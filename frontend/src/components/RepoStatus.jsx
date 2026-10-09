import { CircleAlert, LoaderCircle, RefreshCw } from "lucide-react";
import { GitHubIcon } from "./icons.jsx";

const fmt = (n) => n.toLocaleString();

export default function RepoStatus({ repo, onRefresh }) {
  if (repo.status === "error") {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm">
        <CircleAlert className="mt-0.5 size-5 shrink-0 text-rose-400" />
        <div className="flex-1">
          <p className="font-medium text-rose-200">Couldn't index {repo.id}</p>
          <p className="mt-1 text-rose-300/80">{repo.message}</p>
        </div>
        <button
          onClick={onRefresh}
          className="rounded-lg border border-rose-500/30 px-3 py-1 text-rose-200 transition hover:bg-rose-500/20"
        >
          Retry
        </button>
      </div>
    );
  }

  if (repo.status !== "ready") {
    const pct = repo.total ? Math.round((100 * repo.done) / repo.total) : null;
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
        <div className="flex items-center gap-3 text-sm">
          <LoaderCircle className="size-4 animate-spin text-indigo-300" />
          <span className="text-zinc-200">{repo.message || "Waiting in queue ..."}</span>
          {pct !== null && (
            <span className="ml-auto font-mono text-xs text-zinc-400">
              {fmt(repo.done)} / {fmt(repo.total)} functions
            </span>
          )}
        </div>
        <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-zinc-800">
          {pct === null ? (
            <div className="h-full w-1/3 animate-pulse rounded-full bg-indigo-500/60" />
          ) : (
            <div
              className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-400 transition-[width] duration-500"
              style={{ width: `${pct}%` }}
            />
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-zinc-400">
      <a
        href={repo.url}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-2 font-medium text-zinc-200 hover:text-white"
      >
        <GitHubIcon />
        {repo.id}
      </a>
      <span>
        {fmt(repo.n_functions)} functions in {fmt(repo.n_files)} files
      </span>
      <span className="font-mono text-xs text-zinc-500">@ {repo.commit.slice(0, 7)}</span>
      <button
        onClick={onRefresh}
        title="Pull the latest commit and re-index changed files"
        className="ml-auto inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs text-zinc-500 transition hover:bg-zinc-800 hover:text-zinc-200"
      >
        <RefreshCw className="size-3.5" />
        Update
      </button>
    </div>
  );
}
