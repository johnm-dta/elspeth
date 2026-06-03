import { useEffect, useRef, useId, useState, useCallback, type ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import mermaid from "mermaid";
import DOMPurify from "dompurify";
import { Highlight, themes as prismThemes } from "prism-react-renderer";
import { useTheme, type ResolvedTheme } from "@/hooks/useTheme";

type MermaidConfig = NonNullable<Parameters<typeof mermaid.initialize>[0]>;

function cssToken(name: string): string {
  if (typeof document === "undefined") return "";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function mermaidThemeFromTokens(theme: ResolvedTheme): MermaidConfig {
  const themeVariables: Record<string, string> = {};
  const tokenVariables = {
    primaryColor: "--color-surface-elevated",
    primaryBorderColor: "--color-border-strong",
    primaryTextColor: "--color-text",
    lineColor: "--color-text-muted",
    secondaryColor: "--color-surface-raised",
    tertiaryColor: "--color-bg",
  };

  for (const [themeKey, tokenName] of Object.entries(tokenVariables)) {
    const value = cssToken(tokenName);
    if (value) {
      themeVariables[themeKey] = value;
    }
  }

  return {
    startOnLoad: false,
    theme: theme === "dark" ? "dark" : "default",
    themeVariables,
  };
}

// Initial mermaid configuration (dark default, updated reactively by MermaidDiagram)
mermaid.initialize({ startOnLoad: false, theme: "dark" });

interface MarkdownRendererProps {
  content: string;
}

/**
 * Renders markdown content with GFM support and Mermaid diagram rendering.
 *
 * Mermaid code blocks (```mermaid) are rendered as interactive diagrams.
 * All other code blocks render as syntax-highlighted <pre><code>.
 */
export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: SafeLink,
          code: CodeBlock,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function SafeLink({
  href,
  children,
  ...props
}: ComponentPropsWithoutRef<"a">) {
  const isExternal =
    typeof href === "string" &&
    (href.startsWith("http://") || href.startsWith("https://"));
  return (
    <a
      href={href}
      target={isExternal ? "_blank" : undefined}
      rel={isExternal ? "noopener noreferrer" : undefined}
      {...props}
    >
      {children}
    </a>
  );
}

/**
 * Custom code renderer that intercepts mermaid blocks and renders
 * them as diagrams, renders all other fenced blocks with Prism syntax
 * highlighting and a copy-to-clipboard button, and passes inline code
 * through as-is.
 *
 * Pure router — no hooks. Hooks live in FencedCodeBlock so that inline-code
 * and mermaid renders do not subscribe to theme context or allocate state.
 */
function CodeBlock({
  className,
  children,
  ...props
}: ComponentPropsWithoutRef<"code">) {
  const language = className?.replace("language-", "") ?? "";
  const code = String(children).replace(/\n$/, "");

  // Inline code (no language, rendered inside a <p>)
  if (!className) {
    return <code className="inline-code" {...props}>{children}</code>;
  }

  // Mermaid diagrams get special treatment
  if (language === "mermaid") {
    return <MermaidDiagram chart={code} />;
  }

  // All other fenced blocks: delegate to FencedCodeBlock which owns the hooks
  return (
    <FencedCodeBlock
      code={code}
      language={language}
      className={className}
      {...props}
    />
  );
}

interface FencedCodeBlockProps extends ComponentPropsWithoutRef<"code"> {
  code: string;
  language: string;
  className: string;
}

/**
 * Renders a fenced code block with Prism syntax highlighting and a
 * copy-to-clipboard button. Owns all hooks so that CodeBlock (the router)
 * stays hook-free and inline-code / mermaid renders stay lightweight.
 */
function FencedCodeBlock({
  code,
  language,
  className,
  ...props
}: FencedCodeBlockProps) {
  const { resolvedTheme } = useTheme();
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      // 2000ms keeps the confirmation visible long enough for users with
      // higher cognitive load (looking-away to find the paste target and
      // returning) — below this threshold the affordance often reverts
      // before they can confirm the copy succeeded.
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API unavailable — the user can still select-and-copy manually.
    }
  }, [code]);

  const prismTheme =
    resolvedTheme === "dark" ? prismThemes.vsDark : prismThemes.vsLight;

  return (
    <div className="code-block-wrapper">
      <button
        type="button"
        className="code-block-copy"
        onClick={handleCopy}
        aria-label={copied ? "Copied" : "Copy code"}
      >
        {copied ? "Copied" : "Copy"}
      </button>
      <Highlight code={code} language={language || "text"} theme={prismTheme}>
        {({ className: hClass, style, tokens, getLineProps, getTokenProps }) => (
          <pre className={`code-block ${hClass}`} style={style}>
            <code className={className} {...props}>
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  {line.map((token, j) => (
                    <span key={j} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </code>
          </pre>
        )}
      </Highlight>
    </div>
  );
}

/**
 * Renders a Mermaid diagram. Uses mermaid.render() to produce SVG,
 * then injects it via innerHTML (mermaid's API requires this).
 *
 * Falls back to a <pre> block if mermaid parsing fails.
 */
function MermaidDiagram({ chart }: { chart: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const uniqueId = useId().replace(/:/g, "-");
  const { resolvedTheme } = useTheme();
  // Counter forces a unique mermaid render ID when the theme changes,
  // since mermaid.render() caches by ID.
  const [renderCount, setRenderCount] = useState(0);

  // Re-initialize mermaid when theme changes
  useEffect(() => {
    mermaid.initialize(mermaidThemeFromTokens(resolvedTheme));
    setRenderCount((c) => c + 1);
  }, [resolvedTheme]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let cancelled = false;

    mermaid
      .render(`mermaid-${uniqueId}-${renderCount}`, chart)
      .then(({ svg }) => {
        if (!cancelled && container) {
          container.innerHTML = DOMPurify.sanitize(svg, {
            USE_PROFILES: { svg: true, svgFilters: true },
          });
        }
      })
      .catch(() => {
        if (!cancelled && container) {
          container.textContent = chart;
          container.classList.add("mermaid-fallback");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [chart, uniqueId, renderCount]);

  return (
    <div
      ref={containerRef}
      className="mermaid-container"
      role="img"
      aria-label="Mermaid diagram"
    />
  );
}
