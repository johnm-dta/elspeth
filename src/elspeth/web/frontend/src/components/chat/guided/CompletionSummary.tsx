// src/components/chat/guided/CompletionSummary.tsx
//
// Guided-mode terminal widget for the "completed" outcome (Task 7.10).
// Conventions inherited from ExitToFreeformButton (Task 7.8) and
// ProposeChainTurn (Task 7.6).
//
//   - Props: { terminal: TerminalState } -- read-only consumer of the terminal.
//   - Renders only when terminal.kind === "completed" AND pipeline_yaml !== null.
//     Returns null otherwise.  No defensive ?? "" coercion: an absent yaml is
//     evidence (absence), not an invitation to render empty syntax-highlighted
//     content.
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
  // Guard: only render in the completed+yaml-present state.
  if (terminal.kind !== "completed" || terminal.pipeline_yaml === null) {
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
          The pre id is used for distinctness testing across instances. */}
      <div className="guided-completion-yaml-container">
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
          Review YAML
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
