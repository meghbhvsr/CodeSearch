"""CodeSearchNet (Python) loading and cleaning.

Each example becomes a (query, code) pair:
  query = first paragraph of the docstring (summary line(s), no :param/Args: sections)
  code  = the function source with the docstring removed

Removing the docstring matters: CodeSearchNet's `func_code_string` still contains it,
so leaving it in means the query appears verbatim inside its own target document and
every method (BM25 especially) looks far better than it really is.

Cleaned splits are cached as JSONL under data/ so later phases (training, hard-negative
mining) read exactly the same pairs the evaluation uses.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HF_REPO = "code-search-net/code_search_net"
SPLIT_FILES = {
    "train": "python/train-00000-of-00001.parquet",
    "validation": "python/validation-00000-of-00001.parquet",
    "test": "python/test-00000-of-00001.parquet",
}


@dataclass
class Pair:
    id: str  # repo/path::func_name, for tracing results back to source
    query: str
    code: str
    url: str


# ---------------------------------------------------------------- docstring handling

# Lines that start a docstring's parameter/return section; the query stops before them.
_SECTION_START = re.compile(
    r"^\s*(:param|:type|:return|:rtype|:raises|:arg|@param|@return|"
    r"(Args|Arguments|Parameters|Returns|Return|Yields|Raises|Example|Examples|Note|Notes)\s*:?\s*$|-{3,})",
    re.IGNORECASE,
)

# Fallback for code `ast` can't parse (CodeSearchNet has plenty of Python 2):
# the first triple-quoted string right after the signature's closing colon.
_DOCSTRING_RE = re.compile(
    r"(\)\s*(?:->[^:]*)?:[ \t]*(?:#[^\n]*)?\n)"  # end of signature
    r"[ \t]*[rRuUbB]{0,2}(\"\"\"|''')(?:.|\n)*?\2[ \t]*\n?"
)


def docstring_to_query(doc: str) -> str:
    """First paragraph of a docstring, cut before any param/return section."""
    lines = []
    for line in doc.strip().splitlines():
        if not line.strip():
            if lines:
                break
            continue
        if _SECTION_START.match(line):
            break
        lines.append(line.strip())
    return " ".join(lines)


def strip_docstring(code: str) -> str:
    """Remove the function's own docstring from its source."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):  # Python 2, stray indentation, null bytes...
        return _DOCSTRING_RE.sub(r"\1", code, count=1)

    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return code
    body = tree.body[0].body
    first = body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)):
        return code  # no docstring

    lines = code.splitlines(keepends=True)
    start, end = first.lineno - 1, first.end_lineno  # 0-based line range, end exclusive
    owns_its_lines = lines[start][: first.col_offset].strip() == "" and (len(body) == 1 or body[1].lineno > end)
    if owns_its_lines:
        return "".join(lines[:start] + lines[end:])
    return _DOCSTRING_RE.sub(r"\1", code, count=1)  # e.g. `def f(): "doc"; return 1`


_TRIPLE_QUOTED = re.compile(r"[ \t]*[rRuUbB]{0,2}(\"\"\"|''')(?:.|\n)*?\1[ \t]*\n?")


def remove_string_containing(code: str, text: str) -> str:
    """Drop the triple-quoted literal that contains `text`.

    Catches CodeSearchNet rows whose "docstring" is a string literal further down the body
    (after an import, say), which strip_docstring leaves alone because Python wouldn't
    treat it as a docstring.
    """
    for m in _TRIPLE_QUOTED.finditer(code):
        if text in m.group(0):
            return code[: m.start()] + code[m.end() :]
    return code


# ---------------------------------------------------------------- loading


def _clean_split(split: str) -> tuple[list[Pair], dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, SPLIT_FILES[split], repo_type="dataset")
    columns = ["repository_name", "func_path_in_repository", "func_name",
               "func_code_string", "func_documentation_string", "func_code_url"]
    table = pq.read_table(path, columns=columns)
    pairs: list[Pair] = []
    seen_code: set[str] = set()
    stats = {"raw": table.num_rows, "empty_query": 0, "duplicate_code": 0}
    for row in (r for batch in table.to_batches(10_000) for r in batch.to_pylist()):
        query = docstring_to_query(row["func_documentation_string"])
        if len(query.split()) < 2:
            stats["empty_query"] += 1
            continue
        code = strip_docstring(row["func_code_string"])
        probe = query[:40]
        if probe in code:
            code = remove_string_containing(code, probe)
            if probe in code:  # docstring text still present in some other form
                stats["leaky"] = stats.get("leaky", 0) + 1
                continue
        if code in seen_code:
            # Identical bodies would be indistinguishable targets; keep the first.
            stats["duplicate_code"] += 1
            continue
        seen_code.add(code)
        pairs.append(
            Pair(
                id=f"{row['repository_name']}/{row['func_path_in_repository']}::{row['func_name']}",
                query=query,
                code=code,
                url=row["func_code_url"],
            )
        )
    stats["kept"] = len(pairs)
    return pairs, stats


def load_pairs(split: str = "test", refresh: bool = False) -> list[Pair]:
    """Cleaned (query, code) pairs for a split, built once and cached in data/."""
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {list(SPLIT_FILES)}")
    path = DATA_DIR / f"csn_python_{split}.jsonl"
    if refresh or not path.exists():
        pairs, stats = _clean_split(split)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(asdict(p)) + "\n")
        tmp.replace(path)
        print(f"[data] {split}: {stats} -> {path}")
    with path.open(encoding="utf-8") as f:
        return [Pair(**json.loads(line)) for line in f]
