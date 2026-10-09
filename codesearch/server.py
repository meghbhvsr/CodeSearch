"""Web backend: clone any public GitHub repo, index its Python functions, search them.

  codesearch-web                 # http://127.0.0.1:8000 (serves the built React app from frontend/dist)
  codesearch-web --port 9000

API
  POST /api/repos          {"url": "https://github.com/psf/requests", "refresh": false}
  GET  /api/repos          all repos and their status
  GET  /api/repos/{owner}/{name}
  POST /api/search         {"repo": "psf/requests", "query": "...", "k": 10, "tests": false}

Indexing runs one repo at a time in a background thread (there's one GPU). Searches can
still run meanwhile, because the GPU lock is taken per embedding batch, not per repo.
Clones live in .repos/ and embeddings in .index_cache/. Repos cloned in earlier sessions
are re-registered at startup from cache, without touching the network.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("USE_TF", "0")  # a broken TensorFlow install otherwise breaks transformers imports
os.environ.setdefault("TQDM_DISABLE", "1")
sys.modules.setdefault("tensorflow", None)  # see scripts/train.py

import argparse  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = ROOT / ".repos"
FRONTEND_DIST = ROOT / "frontend" / "dist"
MODEL_PATH = os.environ.get("CODESEARCH_MODEL", str(ROOT / "checkpoints" / "mnrl" / "final"))
MAX_FUNCTIONS = 60_000  # ~2 min of embedding on a 3060 Ti
CLONE_TIMEOUT_S = 300

GITHUB_URL = re.compile(
    r"^\s*(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9-]{1,39})/([A-Za-z0-9_.-]{1,100}?)(?:\.git)?/?(?:[/?#].*)?\s*$"
)

log = logging.getLogger("codesearch.web")


# ---------------------------------------------------------------- state


@dataclass
class Repo:
    owner: str
    name: str
    status: str = "queued"  # queued | cloning | indexing | ready | error
    message: str = ""
    done: int = 0  # functions embedded so far
    total: int = 0
    commit: str = ""
    n_files: int = 0
    n_functions: int = 0
    n_tests: int = 0
    seconds: float = 0.0
    index: object = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def path(self) -> Path:
        return REPOS_DIR / f"{self.owner}__{self.name}"

    def public(self) -> dict:
        return {
            "id": self.id, "owner": self.owner, "name": self.name, "status": self.status,
            "message": self.message, "done": self.done, "total": self.total,
            "commit": self.commit, "n_files": self.n_files, "n_functions": self.n_functions,
            "n_tests": self.n_tests, "seconds": round(self.seconds, 1),
            "url": f"https://github.com/{self.owner}/{self.name}",
        }


class LockedEncoder:
    """Serializes GPU use between the indexing thread and search requests."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.lock = threading.Lock()

    def encode(self, texts, max_len):
        with self.lock:
            return self.encoder.encode(texts, max_len=max_len)


repos: dict[str, Repo] = {}
repos_lock = threading.Lock()
jobs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer")
encoder: LockedEncoder | None = None


# ---------------------------------------------------------------- indexing


def _git(*args, cwd: Path | None = None, timeout: int = 60) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # private/missing repos fail instead of prompting
    out = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        err = out.stderr.strip().splitlines()
        raise RuntimeError(err[-1] if err else f"git {args[0]} failed")
    return out.stdout.strip()


def _friendly_git_error(msg: str) -> str:
    if "could not read Username" in msg or "not found" in msg.lower() or "Authentication failed" in msg:
        return "Repository not found. It may be private or misspelled."
    return f"git: {msg}"


def index_repo(repo: Repo, refresh: bool) -> None:
    from codesearch.index import RepoIndex, RepoIndexError

    t0 = time.time()
    try:
        REPOS_DIR.mkdir(exist_ok=True)
        if not (repo.path / ".git").exists():
            repo.status, repo.message = "cloning", "Cloning from GitHub ..."
            url = f"https://github.com/{repo.owner}/{repo.name}.git"
            # Shallow, no tags, and symlinks checked out as plain files.
            _git("clone", "--depth", "1", "--single-branch", "--no-tags", "-c", "core.symlinks=false",
                 url, str(repo.path), timeout=CLONE_TIMEOUT_S)
        elif refresh:
            repo.status, repo.message = "cloning", "Pulling latest commit ..."
            _git("fetch", "--depth", "1", "origin", "HEAD", cwd=repo.path, timeout=CLONE_TIMEOUT_S)
            _git("reset", "--hard", "FETCH_HEAD", cwd=repo.path)
        repo.commit = _git("rev-parse", "HEAD", cwd=repo.path)

        repo.status, repo.message = "indexing", "Splitting into functions ..."

        def progress(done, total):
            repo.done, repo.total = done, total
            repo.message = "Embedding functions ..."

        index = RepoIndex.build(repo.path, encoder, MODEL_PATH, log=lambda m: log.info("%s: %s", repo.id, m),
                                progress=progress, max_functions=MAX_FUNCTIONS)
        repo.index = index
        repo.n_functions = len(index.chunks)
        repo.n_tests = int(index.is_test.sum())
        repo.n_files = len({c.path for c in index.chunks})
        repo.status, repo.message = "ready", ""
    except subprocess.TimeoutExpired:
        repo.status, repo.message = "error", f"Cloning took longer than {CLONE_TIMEOUT_S // 60} minutes."
    except RuntimeError as e:
        repo.status, repo.message = "error", _friendly_git_error(str(e))
    except RepoIndexError as e:
        repo.status, repo.message = "error", str(e)
    except Exception as e:  # keep the server alive; surface the problem in the UI
        log.exception("indexing %s failed", repo.id)
        repo.status, repo.message = "error", f"{type(e).__name__}: {e}"
    finally:
        repo.seconds = time.time() - t0


def submit(owner: str, name: str, refresh: bool = False) -> Repo:
    key = f"{owner}/{name}".lower()
    with repos_lock:
        repo = repos.get(key)
        if repo and repo.status in ("queued", "cloning", "indexing"):
            return repo
        if repo and repo.status == "ready" and not refresh:
            return repo
        repo = Repo(owner=owner, name=name)
        repos[key] = repo
    jobs.submit(index_repo, repo, refresh)
    return repo


# ---------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder
    warnings.filterwarnings("ignore")
    from codesearch.dense import DenseEncoder

    log.info("loading model %s", MODEL_PATH)
    dense = DenseEncoder(MODEL_PATH)
    dense.show_progress = False
    encoder = LockedEncoder(dense)
    if REPOS_DIR.exists():  # re-register earlier clones; indexing reuses the embedding cache
        for d in sorted(REPOS_DIR.iterdir()):
            if (d / ".git").exists() and "__" in d.name:
                submit(*d.name.split("__", 1))
    yield
    jobs.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="codesearch", lifespan=lifespan)


class AddRepo(BaseModel):
    url: str = Field(max_length=300)
    refresh: bool = False


class SearchRequest(BaseModel):
    repo: str = Field(max_length=200)
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=10, ge=1, le=50)
    tests: bool = False


def _get(repo_id: str) -> Repo:
    repo = repos.get(repo_id.lower())
    if repo is None:
        raise HTTPException(404, f"{repo_id} hasn't been added")
    return repo


@app.post("/api/repos")
def add_repo(body: AddRepo):
    m = GITHUB_URL.match(body.url)
    if not m or m.group(2) in (".", ".."):
        raise HTTPException(400, "Enter a GitHub repository URL, like https://github.com/psf/requests")
    return submit(m.group(1), m.group(2), refresh=body.refresh).public()


@app.get("/api/repos")
def list_repos():
    return [r.public() for r in reversed(list(repos.values()))]


@app.get("/api/repos/{owner}/{name}")
def repo_status(owner: str, name: str):
    return _get(f"{owner}/{name}").public()


@app.post("/api/search")
def search(body: SearchRequest):
    repo = _get(body.repo)
    if repo.status != "ready":
        raise HTTPException(409, f"{repo.id} is {repo.status}, not ready to search")
    from codesearch.index import QUERY_MAX_LEN

    t0 = time.time()
    q = encoder.encode([body.query], max_len=QUERY_MAX_LEN).cpu().numpy()
    hits = repo.index.search(q, body.k, tests=body.tests)
    blob = f"https://github.com/{repo.owner}/{repo.name}/blob/{repo.commit}"
    return {
        "repo": repo.id,
        "query": body.query,
        "ms": round((time.time() - t0) * 1000),
        "searched": repo.n_functions if body.tests else repo.n_functions - repo.n_tests,
        "hidden_tests": 0 if body.tests else repo.n_tests,
        "results": [
            {
                "name": c.name, "path": c.path, "start_line": c.start_line, "end_line": c.end_line,
                "def_line": c.end_line - len(c.code.splitlines()) + 1,
                "score": round(score, 4), "code": c.code,
                "url": f"{blob}/{c.path}#L{c.start_line}-L{c.end_line}",
            }
            for c, score in hits
        ],
    }


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


def main():
    import uvicorn

    ap = argparse.ArgumentParser(prog="codesearch-web")
    ap.add_argument("--host", default="127.0.0.1", help="default: this machine only")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not FRONTEND_DIST.exists():
        log.warning("frontend/dist not found: run `npm run build` in frontend/ (API still works)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
