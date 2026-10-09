import { useState } from "react";
import { LoaderCircle } from "lucide-react";
import { GitHubIcon } from "./icons.jsx";

const EXAMPLES = ["psf/requests", "pallets/flask", "encode/httpx", "tiangolo/typer"];

export default function RepoForm({ onSubmit }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(value) {
    const target = value.trim();
    if (!target) return;
    setBusy(true);
    setError("");
    try {
      await onSubmit(target.includes("github.com") ? target : `https://github.com/${target}`);
      setUrl("");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(url);
        }}
        className="flex flex-col gap-2 sm:flex-row"
      >
        <label className="group relative flex-1">
          <span className="sr-only">GitHub repository URL</span>
          <GitHubIcon className="pointer-events-none absolute top-1/2 left-4 size-5 -translate-y-1/2 text-zinc-500 transition group-focus-within:text-zinc-300" />
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://github.com/owner/repo"
            spellCheck={false}
            className="h-12 w-full rounded-xl border border-zinc-800 bg-zinc-900/80 pr-4 pl-12 text-[15px] text-zinc-100 placeholder-zinc-500 shadow-inner transition outline-none focus:border-indigo-500/70 focus:ring-4 focus:ring-indigo-500/15"
          />
        </label>
        <button
          type="submit"
          disabled={busy || !url.trim()}
          className="inline-flex h-12 items-center justify-center gap-2 rounded-xl bg-indigo-500 px-6 font-medium text-white shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy && <LoaderCircle className="size-4 animate-spin" />}
          Index repo
        </button>
      </form>
      {error && <p className="mt-2 text-sm text-rose-400">{error}</p>}
      <p className="mt-3 flex flex-wrap items-center gap-2 text-sm text-zinc-500">
        Try
        {EXAMPLES.map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => submit(name)}
            className="rounded-md border border-zinc-800 bg-zinc-900/60 px-2 py-0.5 font-mono text-xs text-zinc-300 transition hover:border-zinc-600 hover:text-white"
          >
            {name}
          </button>
        ))}
      </p>
    </div>
  );
}
