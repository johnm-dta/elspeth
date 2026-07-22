// src/components/chat/guided/CompletionSummary.tsx
//
// Guided-mode terminal widget for the "completed" outcome (Task 7.10).
// Conventions inherited from ExitToFreeformButton (Task 7.8) and
// Current guided completion projection.
//
//   - Props: { terminal: TerminalState } -- read-only consumer of the terminal.
//   - Renders only when terminal.kind === "completed". The TerminalState
//     discriminator guarantees completed terminals carry pipeline YAML.
//   - Task-oriented actions expose the three next moves users expect after the
//     wizard completes: open freeform editing, review generated YAML, or run
//     validation.
//   - useTheme() for Prism theme-awareness, matching YamlView.tsx:164.
//   - <button type="button"> (never <div onClick>).
//   - CSS via components/chat/guided/guided.css guided-completion-* classes with design tokens.
//   - No auto-focus on mount.

import { useId } from "react";
import { Highlight, themes } from "prism-react-renderer";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { requestValidate } from "@/stores/subscriptions";
import { useTheme } from "@/hooks/useTheme";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";
import type { TerminalState } from "@/types/guided";

// ── Props ─────────────────────────────────────────────────────────────────────

interface CompletionSummaryProps {
  terminal: TerminalState;
  // Concern B: in a tutorial the "Open freeform editor" action is suppressed
  // (the only path out of a tutorial is graduation, never freeform).
  isTutorial?: boolean;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function CompletionSummary({ terminal, isTutorial }: CompletionSummaryProps) {
  if (terminal.kind !== "completed") {
    return null;
  }

  return <CompletionSummaryInner yaml={terminal.pipeline_yaml} isTutorial={isTutorial} />;
}

// ── Inner component (separated so hooks run unconditionally) ──────────────────
//
// React hooks must not be called conditionally.  The outer CompletionSummary
// returns null before reaching any hook calls.  CompletionSummaryInner is the
// always-rendered branch; it can call hooks without violating the Rules of Hooks.

interface CompletionSummaryInnerProps {
  yaml: string;
  isTutorial?: boolean;
}

function CompletionSummaryInner({ yaml, isTutorial }: CompletionSummaryInnerProps) {
  const reactId = useId();
  const headingId = `${reactId}-heading`;
  const preId = `${reactId}-pre`;

  const exitToFreeform = useSessionStore((s) => s.exitToFreeform);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);
  const isValidating = useExecutionStore((s) => s.isValidating);
  const { resolvedTheme } = useTheme();

  // Match the theme-awareness pattern from YamlView.tsx:164.
  const highlightTheme =
    resolvedTheme === "dark" ? themes.vsDark : themes.vsLight;

  function handleExit(): void {
    void exitToFreeform();
  }

  function handleReviewYaml(): void {
    window.dispatchEvent(new CustomEvent(OPEN_YAML_MODAL_EVENT));
  }

  function handleValidate(): void {
    if (activeSessionId === null || compositionState === null) return;
    requestValidate(activeSessionId, compositionState.version);
  }

  return (
    <div className="guided-completion">
      {/* Heading per Task 7.6 M3 convention for primary entity names */}
      <h3 id={headingId} className="guided-completion-heading">
        Pipeline ready
      </h3>

      {/* YAML preview -- syntax-highlighted via prism-react-renderer.
          Theme-aware: dark/light resolved via useTheme() to match YamlView.
          The pre id is used for distinctness testing across instances.
          role=region + tabIndex: the container scrolls (max-height 400px,
          overflow:auto) and holds no focusable content, so without a tab stop
          its overflow is keyboard-unreachable (WCAG 2.1.1; axe
          scrollable-region-focusable, caught live — jsdom never lays out, so
          the a11y suite cannot see scrollability). A role is required for the
          accessible name to be exposed (elspeth-37293a3b7c). */}
      <div
        className="guided-completion-yaml-container"
        role="region"
        aria-label="Pipeline YAML"
        tabIndex={0}
      >
        <Highlight theme={highlightTheme} code={yaml} language="yaml">
          {({ tokens, getLineProps, getTokenProps }) => (
            <pre id={preId} className="guided-completion-pre">
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>

      <div className="guided-completion-actions">
        {!isTutorial && (
          <button
            type="button"
            className="guided-completion-save-btn"
            onClick={handleExit}
          >
            Open freeform editor
          </button>
        )}
        <button
          type="button"
          className="guided-completion-edit-btn"
          onClick={handleReviewYaml}
        >
          Export YAML
        </button>
        <button
          type="button"
          className="guided-completion-edit-btn"
          onClick={handleValidate}
          disabled={activeSessionId === null || compositionState === null || isValidating}
        >
          {isValidating ? "Validating" : "Validate pipeline"}
        </button>
      </div>
    </div>
  );
}
