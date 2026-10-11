# Semantic code search

Fine-tune a small code embedding model (UniXcoder, 125M) on a single 8 GB GPU to find
functions from natural-language queries, and measure it against BM25 and the base model.

## Phase 1: evaluation and baselines

```bash
pip install -r requirements.txt
python scripts/convert_base_model.py        # -> models/unixcoder-base (safetensors + encoder-only mode token)
python scripts/eval.py bm25
python scripts/eval.py dense                # zero-shot UniXcoder
python scripts/report.py                    # markdown results table
python -m pytest tests
```

**Task.** On the CodeSearchNet Python test split, each query is the first paragraph of a
function's docstring, and its single relevant document is that function's code with the
docstring removed.

**Protocols.**
- `full` (default): rank each query against all 21,763 test functions. This is harder and
  closer to searching a real repo.
- `csn1000`: the original CodeSearchNet setup, with 999 random distractors per query.
  Use it to compare against published numbers.

**Metrics.** MRR@10 and Recall@1/5/10. Tied scores count against the query.

### Results (test split)

| Model | full MRR@10 | R@1 | R@5 | R@10 | csn1000 MRR@10 |
|---|---:|---:|---:|---:|---:|
| BM25 (code-aware tokenizer) | 0.414 | 0.316 | 0.543 | 0.620 | 0.711 |
| UniXcoder-base, zero-shot | 0.499 | 0.395 | 0.638 | 0.720 | 0.800 |
| + fine-tuned (MNRL) | 0.734 | 0.637 | 0.864 | 0.909 | 0.936 |
| + hard negatives | **0.751** | **0.657** | **0.875** | **0.916** | **0.940** |

### Data and eval issues found while building this

- **Docstring leakage.** CodeSearchNet's `func_code_string` still contains the docstring,
  so without stripping it the query appears verbatim inside its own target. The loader
  removes it with `ast`, falling back to a regex for the Python 2 code that `ast` can't
  parse. It also drops 33 pairs where the "docstring" text survives elsewhere in the body.
- **UniXcoder's mode token.** UniXcoder selects encoder-only, decoder-only, or enc-dec
  mode via a token after `<s>`. Loading it directly in sentence-transformers omits that
  token and costs about 12 MRR points zero-shot (0.450 vs 0.571 on a 3k subset). Without
  it the baseline is unfairly weak and fine-tuning gains look inflated.
  `convert_base_model.py` adds the token to the tokenizer's post-processor, so
  sentence-transformers includes it in every encode, during training as well.
- **BM25 speed.** `rank_bm25` scores one query at a time in Python, which would take hours
  over 21k × 21k pairs. `codesearch/bm25.py` precomputes the per-term weights as a sparse
  matrix, so the whole BM25 eval takes about 6 seconds.

## Phase 2: fine-tuning

```bash
python scripts/train.py --run-name smoke --limit 2048 --eval-steps 4   # 2-minute pipeline check
python scripts/train.py --run-name mnrl                                # full run (1 epoch, 401k pairs)
python scripts/view_logs.py                                            # TensorBoard: loss + val MRR@10 curves
python scripts/eval.py dense --model checkpoints/mnrl/final --name mnrl --split validation
python scripts/eval.py dense --model checkpoints/mnrl/final --name mnrl   # test: once per final model
```

- **Training data.** The 401k train pairs only. Validation picks checkpoints and settings;
  test is touched once per final model. CodeSearchNet splits by repository, so test
  functions come from repos the model never saw in training.
- **Loss.** `CachedMultipleNegativesRankingLoss` with batch 256: every query has 255
  in-batch negatives. Gradient caching runs the batch in chunks of 32, so memory scales with
  the chunk size and the number of negatives doesn't.
- **VRAM vs. speed (RTX 3060 Ti, 8 GB).** GradCache mini-batch 32 peaks at about 7.7 GB and
  trains at about 39 pairs/s, roughly 2.8 h per epoch. Mini-batch 64 doesn't raise an
  out-of-memory error. Windows silently spills it into shared system RAM, and training
  slows to 4.3 pairs/s (9× slower). If a run is far slower than expected, check whether
  "Shared GPU memory" in Task Manager is above zero.
- **No-duplicates batching.** Two copies of the same docstring or function in one batch
  would turn the second into a false negative.
- **Checkpoint selection.** Validation MRR@10 on a fixed 5k subset every 250 steps; the best
  checkpoint is saved to `checkpoints/<run>/final`. The pool there is 5k functions rather
  than 21k, so those numbers are higher than eval.py's full-protocol numbers. Use them to
  compare checkpoints, not to fill in the results table.

### Hard negatives

```bash
python scripts/mine_negatives.py      # ~12 min: mine with checkpoints/mnrl/final -> data/hardneg_mnrl.jsonl
python scripts/train.py --run-name mnrl-hn --hard-negatives data/hardneg_mnrl.jsonl
python scripts/eval.py dense --model checkpoints/mnrl-hn/final --name mnrl-hn
```

- **Mining.** The first fine-tuned model embeds all 401k training queries and functions. For
  each query, one negative is picked at random from the 10 highest-ranked candidates that
  pass the filters below. Example: for "Register a function and arguments to be called
  later", the negative is `register_parser_callback`. It's related, but it's the wrong
  function.
- **False-negative filters.** CodeSearchNet has near-copies of the same function under
  different docstrings, and training on one of them as "wrong" teaches the model something
  false. Candidates are skipped when they score ≥ 95% of the correct function's score
  (NV-Retriever's TopK-PercPos), or when their own docstring is identical to the query.
- **Search depth.** With 50 candidates per query, 16% of queries had every candidate
  filtered out and fell back to a random negative. Those were mostly the queries the model
  is least sure about (median positive score 0.48 vs 0.70), exactly where hard negatives
  matter most. Searching 200 deep cut the fallbacks to 7.7%. The rest are mostly
  boilerplate docstrings shared by many functions ("Auto Generated Code" appears 1,779 times).
- **Retraining starts from the base model, not the first fine-tune.** Both runs then differ
  only in the negatives, so the table row isolates their effect.
- **Result.** Full-protocol MRR@10 rises from 0.734 to 0.751 (+1.7), and R@1 from 0.637 to
  0.657. On the 5k validation subset used for checkpoint selection, the best checkpoint
  (step 1500) reaches 0.861, against 0.851 for the first fine-tune. The `codesearch` CLI and
  web app now use `checkpoints/mnrl-hn/final` by default.

## Phase 3: the `codesearch` tool

```bash
pip install -e . --no-deps            # installs the `codesearch` command
git clone --depth 1 https://github.com/psf/requests demo_repos/requests

codesearch "check if a URL should bypass the proxy" demo_repos/requests
codesearch demo_repos/requests        # interactive: load once, search many times
```

```
 1. 0.768  src/requests/utils.py:810  should_bypass_proxies
 2. 0.636  src/requests/utils.py:137  proxy_bypass
 3. 0.568  src/requests/utils.py:99   proxy_bypass_registry
711 functions searched in 14 ms (443 in test code hidden; --tests to include)
```

- **Chunking.** tree-sitter splits the repo into one chunk per function (methods and nested
  functions included, named `Class.method`), never by line count. That's the unit people
  search for and the unit the model was trained on. Unlike `ast`, tree-sitter is
  error-tolerant: Python 2 and broken files still yield their valid functions.
- **Index.** Exact inner-product search in FAISS (`IndexFlatIP`). Cosine similarity on
  normalized vectors, exact rather than approximate, because a repo has thousands of
  functions, not millions. Embeddings are cached in `.index_cache/` per repo. Only files
  whose mtime or size changed are re-embedded, so re-running after an edit is nearly
  instant. Indexing requests (711 functions) takes about 1 s on the GPU.
- **Test code hidden by default.** Test functions say the same words as the code they test
  ("retry", "HTTPError") and crowded out the implementation. `--tests` brings them back.

### Web app

A React + Tailwind frontend on a FastAPI backend: paste any public GitHub URL, watch it
index, and search it.

```bash
cd frontend && npm install && npm run build && cd ..   # once, and after frontend changes
codesearch-web                                         # http://127.0.0.1:8000
```

For frontend work, run `codesearch-web` and `npm run dev` (in `frontend/`) together, then open
http://localhost:5173. Vite hot-reloads the UI and forwards `/api` to the backend.

- **Backend** ([codesearch/server.py](codesearch/server.py)) reuses the CLI's chunker, index and
  model. A repo is shallow-cloned into `.repos/`, then indexed on a background thread with
  progress reporting. Searches stay fast during indexing because the GPU lock is held per
  512-function batch, not per repo. Repos from earlier sessions come back at startup from
  the embedding cache, without network access.
- **Frontend** ([frontend/src](frontend/src)) is React 19, Tailwind 4 and Vite, with
  `prism-react-renderer` for syntax highlighting. Each result links to the exact lines on
  GitHub at the indexed commit. `?repo=…&q=…` URLs are shareable, and `/` focuses the search box.
- **Safety.** Cloned code is only parsed, never executed. Clones are shallow, with
  `core.symlinks=false`, and the indexer skips symlinks, so a malicious repo can't make the
  app read files outside it. Private or misspelled repos fail fast instead of prompting for
  credentials. Repos over 60,000 functions are refused. The server listens on 127.0.0.1
  only. Don't expose it publicly without adding auth and rate limits, because each indexing
  job ties up the GPU.

### Known limitations (found while testing on requests)

- **Long functions are truncated at 256 tokens.** `HTTPAdapter.send` handles retries near
  the end of ~100 lines, past what the model reads, so "where do we retry failed HTTP
  requests" ranks it 20th. Embedding several windows per function and keeping the best
  score would fix this.
- **Query phrasing.** Training queries are docstrings ("Retry failed requests."), not
  questions ("where do we…?"). Phrasing a query like a docstring works best. "max retries
  for a connection" ranks `HTTPAdapter.__init__` 1st, while the question form ranks it
  55th. Training on synthetic question-style queries (the plan's stretch idea) targets
  exactly this.
- **Next steps:** a cross-encoder reranker over the top 50, and BM25 + dense hybrid scoring,
  since exact identifiers like `max_retries` are where BM25 is strong.

## Layout

```
codesearch/data.py      CodeSearchNet loading, docstring stripping, cached JSONL in data/
codesearch/bm25.py      sparse-matrix BM25 + identifier-splitting tokenizer
codesearch/dense.py     sentence-transformers bi-encoder wrapper
codesearch/metrics.py   gold-rank computation (full / pooled) and MRR/Recall
codesearch/chunking.py  tree-sitter: repo -> one chunk per function
codesearch/index.py     cached FAISS index of a repo (incremental re-embedding)
codesearch/cli.py       the `codesearch` command
codesearch/server.py    FastAPI backend for the web app (`codesearch-web`)
frontend/               React + Tailwind + Vite web UI
scripts/train.py        fine-tuning (CachedMNRL, val-based checkpoint selection)
scripts/mine_negatives.py  hard-negative mining with a fine-tuned model
scripts/eval.py         evaluation CLI -> results/*.json + per-query ranks
scripts/report.py       results table
scripts/view_logs.py    TensorBoard launcher that works around the broken TensorFlow install
scripts/prepare_data.py build all cleaned splits (train is needed for Phase 2)
scripts/convert_base_model.py  base model -> local sentence-transformers dir
```

## Environment notes

Tested with torch 2.5.1 + CUDA 12.1 on an RTX 3060 Ti (8 GB).
- `transformers` 5.x needs torch ≥ 2.6, so it is pinned to 4.x. With torch < 2.6,
  `transformers` also refuses to load `.bin` checkpoints, which is why the base model is
  converted to safetensors.
- The scripts set `USE_TF=0` because an installed TensorFlow breaks `transformers` imports.
