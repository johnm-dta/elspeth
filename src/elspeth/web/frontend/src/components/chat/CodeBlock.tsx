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
//
//      A numeric literal that cannot survive a JSON.parse -> JSON.stringify
//      round trip without losing precision or surface form (an id past
//      Number.MAX_SAFE_INTEGER, a "-0", or an unsafe integer written with a
//      trailing ".0"/exponent) skips the re-stringify step: the ORIGINAL
//      source text renders verbatim, still JSON-highlighted rather than
//      falling back to plain text. `hasLossyNumberLiteral` is exported so
//      StructuredJsonPreview can share the identical detection when it
//      decides whether a value is safe to tabularise.
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
   * monospace with no highlighting. A numeric literal that would lose
   * precision or surface form on re-serialisation (see
   * `hasLossyNumberLiteral`) skips the stringify step and renders the
   * original source text verbatim, still JSON-highlighted.
   */
  prettyJson?: boolean;
  /** Show a copy-to-clipboard affordance (mirrors MarkdownRenderer). */
  showCopy?: boolean;
  /** Accessible label for the rendered region. */
  ariaLabel?: string;
}

const JSON_NUMBER_PATTERN = /^(-?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/;

interface DecimalValue {
  sign: "" | "-";
  digits: string;
  scale: number;
}

/**
 * True if `text` contains a JSON numeric literal that cannot survive a
 * JSON.parse -> JSON.stringify round trip with its value and surface form
 * intact — either the value itself (an id past Number.MAX_SAFE_INTEGER) or
 * its digits/scale (a "-0", or an unsafe integer written as "…e0" /
 * "….0"). Ignores digits inside string literals. Decimal normalisation
 * inside `isLossyJsonNumberLiteral` is deliberately lenient about
 * trailing-zero / exponent forms (e.g. "100.0" is NOT flagged) so ordinary
 * Python-emitted floats keep going through the normal pretty-print/table
 * path instead of being over-flagged as lossy.
 */
export function hasLossyNumberLiteral(text: string): boolean {
  let inString = false;
  let escaped = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\") {
        escaped = true;
        continue;
      }
      if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char !== "-" && !isDigit(char)) {
      continue;
    }

    const start = index;
    if (char === "-") {
      index += 1;
      if (index >= text.length || !isDigit(text[index])) {
        index = start;
        continue;
      }
    }

    if (text[index] === "0") {
      index += 1;
    } else {
      while (index < text.length && isDigit(text[index])) {
        index += 1;
      }
    }

    if (text[index] === ".") {
      index += 1;
      while (index < text.length && isDigit(text[index])) {
        index += 1;
      }
    }

    if (text[index] === "e" || text[index] === "E") {
      index += 1;
      if (text[index] === "+" || text[index] === "-") {
        index += 1;
      }
      while (index < text.length && isDigit(text[index])) {
        index += 1;
      }
    }

    if (isLossyJsonNumberLiteral(text.slice(start, index))) {
      return true;
    }
    index -= 1;
  }

  return false;
}

function isLossyJsonNumberLiteral(token: string): boolean {
  const numericValue = Number(token);
  if (!Number.isFinite(numericValue)) {
    return true;
  }

  const rendered = JSON.stringify(numericValue);
  if (typeof rendered !== "string" || rendered === "null") {
    return true;
  }

  const sourceValue = parseDecimalValue(token);
  const renderedValue = parseDecimalValue(rendered);
  if (sourceValue === null || renderedValue === null) {
    return false;
  }

  return (
    sourceValue.sign !== renderedValue.sign ||
    sourceValue.digits !== renderedValue.digits ||
    sourceValue.scale !== renderedValue.scale
  );
}

function parseDecimalValue(token: string): DecimalValue | null {
  const match = token.match(JSON_NUMBER_PATTERN);
  if (!match) {
    return null;
  }

  const [, rawSign, integerPart, fractionPart = "", exponentPart = "0"] = match;
  const exponent = Number(exponentPart);
  if (!Number.isSafeInteger(exponent)) {
    return null;
  }
  const sign = rawSign === "-" ? "-" : "";

  let digits = `${integerPart}${fractionPart}`.replace(/^0+/, "");
  if (digits.length === 0) {
    return { sign, digits: "0", scale: 0 };
  }

  let scale = fractionPart.length - exponent;
  if (scale < 0) {
    const zerosToAppend = -scale;
    if (zerosToAppend > 400) {
      return null;
    }
    digits += "0".repeat(zerosToAppend);
    scale = 0;
  }

  while (scale > 0 && digits.endsWith("0")) {
    digits = digits.slice(0, -1);
    scale -= 1;
  }

  if (digits.length === 0) {
    return { sign, digits: "0", scale: 0 };
  }

  return {
    sign,
    digits,
    scale,
  };
}

function isDigit(char: string): boolean {
  return char >= "0" && char <= "9";
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
      const parsedValue = JSON.parse(code) as unknown;
      displayLanguage = "json";
      dataFormat = "json";
      if (!hasLossyNumberLiteral(code)) {
        displayCode = JSON.stringify(parsedValue, null, 2);
      }
      // else: a numeric literal in `code` can't survive a JSON.parse ->
      // JSON.stringify round trip without losing precision or surface form
      // (a 20-digit id, a "-0", an unsafe integer written as "….0"/"…e0").
      // Leave displayCode at its default (`code`) so it renders verbatim —
      // still JSON-highlighted below, just not re-serialised.
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
