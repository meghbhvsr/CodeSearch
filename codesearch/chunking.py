"""Split a Python repository into one chunk per function, using tree-sitter.

Functions, not fixed line windows: a function is the unit people search for, and it
matches what the model was trained on (one CodeSearchNet function per document). Methods
and nested functions are included, with qualified names like `HTTPAdapter.send`.

tree-sitter is error-tolerant, so Python 2 files and files with syntax errors still yield
their parseable functions instead of being skipped like `ast` would.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python as tsp
from tree_sitter import Language, Parser

_PARSER = Parser(Language(tsp.language()))

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env", ".env",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "build", "dist",
    "site-packages", ".eggs", ".index_cache",
}
MAX_FILE_BYTES = 1_000_000  # bigger .py files are almost always generated


@dataclass
class Chunk:
    path: str  # relative to the repo root, forward slashes
    name: str  # qualified: Class.method, outer.inner
    start_line: int  # 1-based, includes decorators
    end_line: int
    code: str  # the `def ...` through the end of the body (what gets embedded)


def is_test_path(rel_path: str) -> bool:
    parts = rel_path.split("/")
    name = parts[-1]
    return (
        any(d in ("test", "tests", "testing") for d in parts[:-1])
        or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
    )


def iter_python_files(repo: Path):
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = Path(dirpath) / name
                # Never follow symlinks: a cloned repo could point one at a file outside itself.
                if not path.is_symlink() and path.stat().st_size <= MAX_FILE_BYTES:
                    yield path


def extract_functions(source: bytes, rel_path: str) -> list[Chunk]:
    chunks: list[Chunk] = []

    def visit(node, scope: list[str]):
        for child in node.children:
            target, outer = child, child
            if child.type == "decorated_definition":
                target = child.child_by_field_name("definition")
            if target is None:
                continue
            if target.type in ("function_definition", "class_definition"):
                name = target.child_by_field_name("name").text.decode("utf-8", "replace")
                if target.type == "function_definition":
                    chunks.append(Chunk(
                        path=rel_path,
                        name=".".join(scope + [name]),
                        start_line=outer.start_point[0] + 1,
                        end_line=target.end_point[0] + 1,
                        # The def line has no indentation and the body keeps its original
                        # indentation, the same shape as CodeSearchNet's training functions.
                        # CRLF -> LF: Windows checkouts use CRLF, the model was trained on LF.
                        code=source[target.start_byte : target.end_byte].decode("utf-8", "replace").replace("\r\n", "\n"),
                    ))
                body = target.child_by_field_name("body")
                if body is not None:
                    visit(body, scope + [name])
            elif child.type in ("if_statement", "try_statement", "with_statement", "block",
                                "else_clause", "elif_clause", "except_clause", "finally_clause"):
                visit(child, scope)  # e.g. functions defined under `if sys.version_info ...:`

    visit(_PARSER.parse(source).root_node, [])
    return chunks


def chunk_file(path: Path, repo: Path) -> list[Chunk]:
    return extract_functions(path.read_bytes(), path.relative_to(repo).as_posix())
