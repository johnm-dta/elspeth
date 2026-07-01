// ============================================================================
// CodeBlock.tsx — shared, syntax-highlighted value renderer.
//
// Built on the SAME prism-react-renderer highlighter MarkdownRenderer.tsx
// uses (no new dependency).  Two responsibilities:
//
//   1. JSON-shaped values (e.g. invented source data): JSON.parse → 2-space
//      JSON.stringify → Prism `json` grammar (colour-coded).  On parse
//      failure, fall back to plain monospace — NEVER throw, NEVER fabricate
//      structure.  This is the "prettify text fields" surface from the
//      acknowledge-card-stack design.
//   2. Plain highlighted code for a caller-supplied language.
//
// The acknowledgement cards use this for the invented-source value; the
// prompt template keeps its own scroll-gated <pre> (clean structured
// monospace) inside its View expander, so it is not routed through here.
// ============================================================================

import { useCallback, useState, type ReactElement } from "react";
import { Highlight, themes as prismThemes } from "prism-react-renderer";
import { useTheme } from "@/hooks/useTheme";

export interface CodeBlockProps {
  /** The raw value to render. */
  code: string;
  /** Prism grammar to highlight with when not pretty-printing JSON. */
  language?: string;
  /**
   * Attempt to pretty-print the value as JSON (parse → 2-space stringify)
   * before highlighting.  On parse failure, the raw value renders as plain
   * monospace with no highlighting.
   */
  prettyJson?: boolean;
  /** Show a copy-to-clipboard affordance (mirrors MarkdownRenderer). */
  showCopy?: boolean;
  /** Accessible label for the rendered region. */
  ariaLabel?: string;
}

/**
 * Renders a value as syntax-highlighted code.  Pure presentation; reads the
 * resolved theme so the Prism palette tracks light/dark.
 */
export function CodeBlock({
  code,
  language = "text",
  prettyJson = false,
  showCopy = true,
  ariaLabel,
}: CodeBlockProps): ReactElement {
  const { resolvedTheme } = useTheme();
  const [copied, setCopied] = useState(false);

  // Resolve the value to display + the grammar to highlight with.  JSON
  // pretty-printing is best-effort: a parse failure degrades to plain
  // monospace rather than throwing or inventing a shape.
  let displayCode = code;
  let displayLanguage = language;
  let renderPlain = false;
  let dataFormat: "json" | "highlighted" | "plain" = "highlighted";
  if (prettyJson) {
    try {
      displayCode = JSON.stringify(JSON.parse(code), null, 2);
      displayLanguage = "json";
      dataFormat = "json";
    } catch {
      // Never throw, never fabricate structure — show the raw value verbatim.
      renderPlain = true;
      dataFormat = "plain";
    }
  }

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(displayCode);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API unavailable — the user can still select-and-copy.
    }
  }, [displayCode]);

  const prismTheme =
    resolvedTheme === "dark" ? prismThemes.vsDark : prismThemes.vsLight;

  const copyButton = showCopy ? (
    <button
      type="button"
      className="code-block-copy"
      onClick={handleCopy}
      aria-label={copied ? "Copied" : "Copy value"}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  ) : null;

  if (renderPlain) {
    return (
      <div className="code-block-wrapper">
        {copyButton}
        <pre
          className="code-block code-block--plain"
          data-codeblock-format="plain"
          aria-label={ariaLabel}
        >
          {displayCode}
        </pre>
      </div>
    );
  }

  return (
    <div className="code-block-wrapper">
      {copyButton}
      <Highlight
        code={displayCode}
        language={displayLanguage || "text"}
        theme={prismTheme}
      >
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className={`code-block ${className}`}
            style={style}
            data-codeblock-format={dataFormat}
            aria-label={ariaLabel}
          >
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, j) => (
                  <span key={j} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  );
}
