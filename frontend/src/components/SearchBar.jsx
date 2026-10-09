import { useEffect, useRef } from "react";
import { CornerDownLeft, LoaderCircle, Search } from "lucide-react";

export default function SearchBar({ value, onChange, onSubmit, busy, k, onK, tests, onTests }) {
  const input = useRef(null);

  // "/" jumps to the search box, like GitHub.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
        e.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit();
        }}
        className="relative"
      >
        <Search className="pointer-events-none absolute top-1/2 left-5 size-5 -translate-y-1/2 text-zinc-500" />
        <input
          ref={input}
          autoFocus
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Describe the code you're looking for ..."
          aria-label="Search query"
          className="h-14 w-full rounded-2xl border border-zinc-800 bg-zinc-900 pr-28 pl-14 text-base text-zinc-100 placeholder-zinc-500 shadow-xl shadow-black/30 transition outline-none focus:border-indigo-500/70 focus:ring-4 focus:ring-indigo-500/15"
        />
        <button
          type="submit"
          disabled={busy || !value.trim()}
          className="absolute top-1/2 right-2 inline-flex h-10 -translate-y-1/2 items-center gap-2 rounded-xl bg-indigo-500 px-4 text-sm font-medium text-white transition hover:bg-indigo-400 disabled:opacity-40"
        >
          {busy ? <LoaderCircle className="size-4 animate-spin" /> : <CornerDownLeft className="size-4" />}
          Search
        </button>
      </form>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-sm text-zinc-400">
        <label className="inline-flex items-center gap-2">
          Results
          <select
            value={k}
            onChange={(e) => onK(Number(e.target.value))}
            className="rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-zinc-200 outline-none focus:border-indigo-500/70"
          >
            {[5, 10, 20, 50].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <label className="inline-flex cursor-pointer items-center gap-2 select-none">
          <input
            type="checkbox"
            checked={tests}
            onChange={(e) => onTests(e.target.checked)}
            className="size-4 rounded accent-indigo-500"
          />
          Include test code
        </label>
        <span className="ml-auto hidden text-xs text-zinc-600 sm:inline">
          Press <kbd className="rounded border border-zinc-700 px-1.5 font-mono">/</kbd> to search
        </span>
      </div>
    </div>
  );
}
