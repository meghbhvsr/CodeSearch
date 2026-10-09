import { CircleAlert, CircleCheck, LoaderCircle } from "lucide-react";

function StatusIcon({ status }) {
  if (status === "ready") return <CircleCheck className="size-3.5 text-emerald-400" />;
  if (status === "error") return <CircleAlert className="size-3.5 text-rose-400" />;
  return <LoaderCircle className="size-3.5 animate-spin text-indigo-300" />;
}

export default function RepoList({ repos, activeId, onSelect }) {
  if (!repos.length) return null;
  return (
    <nav aria-label="Indexed repositories" className="flex flex-wrap gap-2">
      {repos.map((repo) => {
        const active = repo.id.toLowerCase() === activeId?.toLowerCase();
        return (
          <button
            key={repo.id}
            onClick={() => onSelect(repo.id)}
            aria-current={active ? "true" : undefined}
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition ${
              active
                ? "border-indigo-500/60 bg-indigo-500/15 text-white"
                : "border-zinc-800 bg-zinc-900/60 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200"
            }`}
          >
            <StatusIcon status={repo.status} />
            <span className="font-mono text-[13px]">{repo.id}</span>
          </button>
        );
      })}
    </nav>
  );
}
