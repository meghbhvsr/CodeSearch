import { useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, ExternalLink } from "lucide-react";
import CodeBlock from "./CodeBlock.jsx";

const PREVIEW_LINES = 14;

export default function ResultCard({ result, rank, topScore }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const nLines = result.code.split("\n").length;
  const long = nLines > PREVIEW_LINES;
  // Bar width relative to the best hit, so the gap between results is visible.
  const rel = topScore > 0 ? Math.max(0.08, result.score / topScore) : 0;

  async function copyPath() {
    await navigator.clipboard?.writeText(`${result.path}:${result.def_line}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }

  return (
    <article className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/70 shadow-lg shadow-black/20 transition hover:border-zinc-700">
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-zinc-800/80 px-4 py-3">
        <span className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-zinc-800 font-mono text-xs text-zinc-300">
          {rank}
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="truncate font-mono text-[15px] font-medium text-white">{result.name}</h3>
          <div className="flex items-center gap-1.5 text-xs text-zinc-500">
            <span className="truncate font-mono">
              {result.path}:{result.def_line}
            </span>
            <button onClick={copyPath} title="Copy path" className="rounded p-0.5 transition hover:text-zinc-200">
              {copied ? <Check className="size-3.5 text-emerald-400" /> : <Copy className="size-3.5" />}
            </button>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2" title="Cosine similarity between the query and this function">
            <div className="h-1.5 w-16 overflow-hidden rounded-full bg-zinc-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-400"
                style={{ width: `${rel * 100}%` }}
              />
            </div>
            <span className="font-mono text-xs text-zinc-400">{result.score.toFixed(3)}</span>
          </div>
          <a
            href={result.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 rounded-lg border border-zinc-800 px-2.5 py-1 text-xs text-zinc-300 transition hover:border-zinc-600 hover:text-white"
          >
            GitHub <ExternalLink className="size-3" />
          </a>
        </div>
      </header>
      <CodeBlock code={result.code} startLine={result.def_line} maxLines={expanded ? null : PREVIEW_LINES} />
      {long && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex w-full items-center justify-center gap-1.5 border-t border-zinc-800/80 py-2 text-xs text-zinc-400 transition hover:bg-zinc-800/50 hover:text-zinc-200"
        >
          {expanded ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
          {expanded ? "Show less" : `Show all ${nLines} lines`}
        </button>
      )}
    </article>
  );
}
