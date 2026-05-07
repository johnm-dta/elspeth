import { useEffect, useRef, useId, useState, type ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import mermaid from "mermaid";
import DOMPurify from "dompurify";
import { useTheme, type ResolvedTheme } from "@/hooks/useTheme";

const MERMAID_THEMES: Record<ResolvedTheme, Parameters<typeof mermaid.initialize>[0]> = {
  dark: {
    startOnLoad: false,
    theme: "dark",
    themeVariables: {
      primaryColor: "#1a3d47",
      primaryBorderColor: "#4db89a",
      primaryTextColor: "#dff0ee",
      lineColor: "#6a9898",
      secondaryColor: "#28504a",
      tertiaryColor: "#0f2d35",
    },
  },
  light: {
    startOnLoad: false,
    theme: "default",
    themeVariables: {
      primaryColor: "#eaf2f3",
      primaryBorderColor: "#2a8a70",
      primaryTextColor: "#0f2d35",
      lineColor: "#5a7a84",
      secondaryColor: "#f0f6f7",
      tertiaryColor: "#f4f8f9",
    },
  },
};

// Initial mermaid configuration (dark default, updated reactively by MermaidDiagram)
mermaid.initialize(MERMAID_THEMES.dark);

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
          code: CodeBlock,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

/**
 * Custom code renderer that intercepts mermaid blocks and renders
 * them as diagrams, while passing all other code through normally.
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

  // All other code blocks render as <pre><code>
  return (
    <pre className="code-block">
      <code className={className} {...props}>
        {code}
      </code>
    </pre>
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
    mermaid.initialize(MERMAID_THEMES[resolvedTheme]);
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
