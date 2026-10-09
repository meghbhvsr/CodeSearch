import { useCallback, useEffect, useRef, useState } from "react";
import { CircleAlert, Search, Sparkles } from "lucide-react";
import * as api from "./api.js";
import RepoForm from "./components/RepoForm.jsx";
import RepoList from "./components/RepoList.jsx";
import RepoStatus from "./components/RepoStatus.jsx";
import ResultCard from "./components/ResultCard.jsx";
import SearchBar from "./components/SearchBar.jsx";

const EXAMPLE_QUERIES = [
  "parse a configuration file",
  "handle HTTP errors",
  "retry a failed request",
  "validate user input",
  "read a file into memory",
];
const PENDING = new Set(["queued", "cloning", "indexing"]);

// ?repo=owner/name&q=... keeps searches shareable and survives a reload.
const params = new URLSearchParams(window.location.search);

export default function App() {
  const [repos, setRepos] = useState([]);
  const [activeId, setActiveId] = useState(params.get("repo"));
  const [query, setQuery] = useState(params.get("q") ?? "");
  const [k, setK] = useState(10);
  const [tests, setTests] = useState(false);
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  const autoSearched = useRef(false);

  const active = repos.find((r) => r.id.toLowerCase() === activeId?.toLowerCase());

  const refreshRepos = useCallback(async () => {
    try {
      const list = await api.listRepos();
      setRepos(list);
      setActiveId((id) => id ?? list[0]?.id ?? null);
    } catch {
      /* backend restarting; the next poll retries */
    }
  }, []);

  useEffect(() => {
    refreshRepos();
  }, [refreshRepos]);

  // Poll while anything is cloning or indexing.
  const anyPending = repos.some((r) => PENDING.has(r.status));
  useEffect(() => {
    if (!anyPending) return;
    const t = setInterval(refreshRepos, 1000);
    return () => clearInterval(t);
  }, [anyPending, refreshRepos]);

  useEffect(() => {
    const p = new URLSearchParams();
    if (activeId) p.set("repo", activeId);
    if (results?.query) p.set("q", results.query);
    window.history.replaceState(null, "", p.size ? `?${p}` : window.location.pathname);
  }, [activeId, results]);

  const runSearch = useCallback(
    async (q = query, opts = {}) => {
      if (!active || active.status !== "ready" || !q.trim()) return;
      setQuery(q);
      setSearching(true);
      setSearchError("");
      try {
        setResults(await api.search({ repo: active.id, query: q.trim(), k, tests, ...opts }));
      } catch (e) {
        setSearchError(e.message);
      } finally {
        setSearching(false);
      }
    },
    [active, query, k, tests],
  );

  // Run the search from the URL once its repo is ready.
  useEffect(() => {
    if (!autoSearched.current && active?.status === "ready" && query) {
      autoSearched.current = true;
      runSearch(query);
    }
  }, [active, query, runSearch]);

  async function handleAdd(url, refresh = false) {
    const repo = await api.addRepo(url, refresh);
    setActiveId(repo.id);
    setResults(null);
    await refreshRepos();
  }

  function selectRepo(id) {
    setActiveId(id);
    setResults(null);
    setSearchError("");
  }

  const topScore = results?.results[0]?.score ?? 0;

  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 sm:px-6">
      <header className="flex items-center justify-between py-6">
        <a href="/" className="flex items-center gap-2.5">
          <span className="flex size-9 items-center justify-center rounded-xl bg-indigo-500 shadow-lg shadow-indigo-500/30">
            <Search className="size-5 text-white" strokeWidth={2.5} />
          </span>
          <span className="text-lg font-semibold tracking-tight">codesearch</span>
        </a>
        <span className="hidden items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400 sm:inline-flex">
          <Sparkles className="size-3.5 text-indigo-300" />
          UniXcoder fine-tuned on 401k Python functions
        </span>
      </header>

      <main className="flex-1 pb-16">
        <section className="pt-8 pb-10 text-center">
          <h1 className="text-4xl font-bold tracking-tight text-balance sm:text-5xl">
            Search any codebase{" "}
            <span className="bg-gradient-to-r from-indigo-300 to-violet-300 bg-clip-text text-transparent">
              by meaning
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-zinc-400 text-pretty">
            Paste a GitHub repo, describe what you're looking for in plain English, and get the functions that do it,
            even when they don't share a single keyword with your question.
          </p>
        </section>

        <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-4 backdrop-blur sm:p-5">
          <RepoForm onSubmit={handleAdd} />
        </div>

        {repos.length > 0 && (
          <div className="mt-6">
            <RepoList repos={repos} activeId={activeId} onSelect={selectRepo} />
          </div>
        )}

        {active && (
          <section className="mt-6 space-y-5">
            <RepoStatus repo={active} onRefresh={() => handleAdd(active.url, true)} />

            {active.status === "ready" && (
              <>
                <SearchBar
                  value={query}
                  onChange={setQuery}
                  onSubmit={() => runSearch()}
                  busy={searching}
                  k={k}
                  onK={(n) => {
                    setK(n);
                    if (results) runSearch(results.query, { k: n });
                  }}
                  tests={tests}
                  onTests={(t) => {
                    setTests(t);
                    if (results) runSearch(results.query, { tests: t });
                  }}
                />

                {!results && !searching && (
                  <div className="flex flex-wrap items-center gap-2 text-sm text-zinc-500">
                    Examples:
                    {EXAMPLE_QUERIES.map((q) => (
                      <button
                        key={q}
                        onClick={() => runSearch(q)}
                        className="rounded-full border border-zinc-800 px-3 py-1 text-zinc-300 transition hover:border-indigo-500/50 hover:text-white"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                )}

                {searchError && (
                  <p className="flex items-center gap-2 text-sm text-rose-400">
                    <CircleAlert className="size-4" /> {searchError}
                  </p>
                )}

                {results && (
                  <div className={`space-y-4 transition-opacity ${searching ? "opacity-50" : ""}`}>
                    <p className="text-sm text-zinc-500">
                      Top {results.results.length} of {results.searched.toLocaleString()} functions for{" "}
                      <span className="text-zinc-300">“{results.query}”</span> · {results.ms} ms
                      {results.hidden_tests > 0 && <> · {results.hidden_tests.toLocaleString()} test functions hidden</>}
                    </p>
                    {results.results.map((r, i) => (
                      <ResultCard key={`${r.path}:${r.start_line}`} result={r} rank={i + 1} topScore={topScore} />
                    ))}
                  </div>
                )}
              </>
            )}
          </section>
        )}
      </main>

      <footer className="border-t border-zinc-900 py-6 text-center text-xs text-zinc-600">
        Functions are split with tree-sitter, embedded with a fine-tuned UniXcoder bi-encoder, and ranked by cosine
        similarity in FAISS.
      </footer>
    </div>
  );
}
