// ============================================================================
// YamlDisplay — pure rendering primitive for a YAML string.
//
// Phase 6B FIX-C: extracted from YamlView so the shared-inspect view can
// render its frozen YAML blob without depending on the session-store
// fetch/proposal machinery that lives in YamlView. YamlDisplay does NOT:
//   - fetch from the API
//   - read or write any store
//   - handle composition proposals
//   - handle 409/error states (callers display the appropriate banner)
//
// What it DOES:
//   - syntax-highlights the supplied YAML via prism-react-renderer
//   - exposes a Copy-to-clipboard button with transient "Copied!" state
//   - exposes a Download button that emits a blob with the supplied filename
//   - chooses a vs-dark / vs-light theme via useTheme()
//
// YamlView retains the fetch, error, empty-state, and proposal-panel
// rendering; once YAML is loaded successfully it delegates display to
// this primitive. SharedInspectView constructs YamlDisplay directly from
// the wire `yaml` string.
// ============================================================================

import { useState, useCallback } from "react";
import { Highlight, themes } from "prism-react-renderer";

import { useTheme } from "@/hooks/useTheme";

interface YamlDisplayProps {
  /** The YAML text to render. */
  yaml: string;
  /**
   * Filename used when the user clicks Download. Defaults to
   * "pipeline.yaml" — callers with version context (YamlView)
   * override with a versioned name.
   */
  filename?: string;
}

const DEFAULT_FILENAME = "pipeline.yaml";

export function YamlDisplay({
  yaml,
  filename = DEFAULT_FILENAME,
}: YamlDisplayProps): JSX.Element {
  const { resolvedTheme } = useTheme();
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(yaml);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API may fail in some contexts (e.g. insecure origin
      // or sandboxed environments). The button stays in its un-copied
      // state in that case — there's no recovery surface in this
      // primitive because the failure is environment-dependent.
    }
  }, [yaml]);

  const handleDownload = useCallback(() => {
    const blob = new Blob([yaml], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [yaml, filename]);

  const highlightTheme = resolvedTheme === "dark" ? themes.vsDark : themes.vsLight;

  return (
    <div className="yaml-view-display" data-testid="yaml-display">
      <div className="yaml-view-toolbar">
        <button
          onClick={handleCopy}
          aria-label={copied ? "Copied to clipboard" : "Copy YAML to clipboard"}
          className="btn yaml-toolbar-btn"
          data-copied={copied ? "true" : "false"}
        >
          {copied ? "Copied!" : "Copy"}
        </button>
        <button
          onClick={handleDownload}
          aria-label="Download YAML file"
          className="btn yaml-toolbar-btn"
        >
          Download
        </button>
      </div>

      <div className="yaml-view-content">
        <Highlight theme={highlightTheme} code={yaml} language="yaml">
          {({ tokens, getLineProps, getTokenProps }) => (
            <pre className="yaml-view-pre">
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })} className="yaml-view-line">
                  <span className="yaml-view-line-number" aria-hidden="true">
                    {i + 1}
                  </span>
                  <span className="yaml-view-line-content">
                    {line.map((token, key) => (
                      <span key={key} {...getTokenProps({ token })} />
                    ))}
                  </span>
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
}
