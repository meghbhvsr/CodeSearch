"""A cached FAISS index of a repository's functions.

The first search of a repo embeds every function (about 500 functions/s on a 3060 Ti),
and later searches only embed the query. The cache lives in .index_cache/ in this
project, not in the searched repo, and is keyed by the repo's absolute path. On each run,
files whose size and mtime are unchanged reuse their stored embeddings, so after an edit
only the changed files are re-embedded. Changing the model invalidates everything.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from codesearch.chunking import Chunk, chunk_file, is_test_path, iter_python_files

CACHE_ROOT = Path(__file__).resolve().parent.parent / ".index_cache"
CODE_MAX_LEN = 256
QUERY_MAX_LEN = 64
CHUNKER_VERSION = 2  # bump when chunk text changes, so cached embeddings are rebuilt
EMBED_BATCH = 512  # functions embedded between progress callbacks


class RepoIndexError(ValueError):
    """The repo can't be indexed: no Python functions, or more than max_functions."""


def _model_id(model_path: str) -> str:
    p = Path(model_path)
    weights = p / "model.safetensors"
    model = f"{p.resolve()}@{weights.stat().st_mtime_ns}" if weights.exists() else model_path
    return f"{model}#chunker{CHUNKER_VERSION}"


class RepoIndex:
    def __init__(self, repo: Path, chunks: list[Chunk], embeddings: np.ndarray):
        self.repo = repo
        self.chunks = chunks
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # exact search; inner product = cosine on unit vectors
        self.index.add(embeddings)
        self.is_test = np.array([is_test_path(c.path) for c in chunks])

    @classmethod
    def build(cls, repo: Path, encoder, model_path: str, force: bool = False, log=print,
              progress=None, max_functions: int | None = None) -> "RepoIndex":
        """progress(done, total) is called while embedding new functions."""
        repo = repo.resolve()
        cache = CACHE_ROOT / f"{repo.name}-{hashlib.sha1(str(repo).encode()).hexdigest()[:8]}"
        model_id = _model_id(model_path)

        old_files, old_chunks, old_emb = {}, [], None
        meta_path = cache / "meta.json"
        if not force and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["model"] == model_id:
                old_files = meta["files"]
                with (cache / "chunks.jsonl").open(encoding="utf-8") as f:
                    old_chunks = [Chunk(**json.loads(line)) for line in f]
                old_emb = np.load(cache / "embeddings.npy")
        old_rows: dict[str, list[int]] = {}
        for row, c in enumerate(old_chunks):
            old_rows.setdefault(c.path, []).append(row)

        t0 = time.time()
        files, chunks, reuse_rows, new_chunks = {}, [], [], []
        for path in iter_python_files(repo):
            rel = path.relative_to(repo).as_posix()
            st = path.stat()
            files[rel] = [st.st_mtime_ns, st.st_size]
            if old_files.get(rel) == files[rel]:
                rows = old_rows.get(rel, [])
                chunks.extend(old_chunks[r] for r in rows)
                reuse_rows.extend(rows)
            else:
                new_chunks.extend(chunk_file(path, repo))
        n_reused = len(chunks)

        if not chunks and not new_chunks:
            raise RepoIndexError(f"no Python functions found in {repo.name}")
        if max_functions and len(chunks) + len(new_chunks) > max_functions:
            raise RepoIndexError(f"{repo.name} has {len(chunks) + len(new_chunks):,} functions; "
                              f"the limit is {max_functions:,}")
        parts = [old_emb[reuse_rows]] if reuse_rows else []
        if new_chunks:
            log(f"embedding {len(new_chunks):,} functions"
                + (f" ({n_reused:,} unchanged reused from cache)" if n_reused else "") + " ...")
            for start in range(0, len(new_chunks), EMBED_BATCH):
                batch = [c.code for c in new_chunks[start : start + EMBED_BATCH]]
                parts.append(encoder.encode(batch, max_len=CODE_MAX_LEN).cpu().numpy().astype(np.float32))
                if progress:
                    progress(min(start + EMBED_BATCH, len(new_chunks)), len(new_chunks))
            chunks.extend(new_chunks)
        embeddings = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)

        if new_chunks or set(files) != set(old_files):
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / "embeddings.npy", embeddings)
            with (cache / "chunks.jsonl").open("w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(asdict(c)) + "\n")
            meta_path.write_text(json.dumps({"repo": str(repo), "model": model_id, "files": files}))
            log(f"indexed {len(chunks):,} functions from {len(files):,} files in {time.time() - t0:.1f}s")
        return cls(repo, chunks, embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 10, tests: bool = False) -> list[tuple[Chunk, float]]:
        """Top-k functions by cosine similarity; test code is skipped unless tests=True."""
        q = query_embedding.reshape(1, -1).astype(np.float32)
        n = len(self.chunks)
        fetch = k if tests else k + int(self.is_test.sum())  # enough to fill k after dropping tests
        scores, ids = self.index.search(q, min(fetch, n))
        hits = [(i, s) for i, s in zip(ids[0], scores[0]) if i >= 0 and (tests or not self.is_test[i])]
        return [(self.chunks[i], float(s)) for i, s in hits[:k]]
