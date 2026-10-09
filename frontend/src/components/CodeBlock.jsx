import { Highlight, themes } from "prism-react-renderer";

export default function CodeBlock({ code, startLine, maxLines }) {
  const lines = code.split("\n");
  const shown = maxLines ? lines.slice(0, maxLines).join("\n") : code;
  return (
    <Highlight theme={themes.vsDark} code={shown} language="python">
      {({ tokens, getLineProps, getTokenProps }) => (
        <pre className="overflow-x-auto py-3 font-mono text-[13px] leading-6">
          {tokens.map((line, i) => {
            const { key: _k, ...lineProps } = getLineProps({ line });
            return (
              <div key={i} {...lineProps} className="flex px-4 hover:bg-white/[0.03]">
                <span className="mr-5 w-10 shrink-0 text-right text-zinc-600 select-none">{startLine + i}</span>
                <span className="whitespace-pre">
                  {line.map((token, j) => {
                    const { key: _t, ...tokenProps } = getTokenProps({ token });
                    return <span key={j} {...tokenProps} />;
                  })}
                </span>
              </div>
            );
          })}
        </pre>
      )}
    </Highlight>
  );
}
