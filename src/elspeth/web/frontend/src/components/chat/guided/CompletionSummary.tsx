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
//   - Two action buttons, both calling useSessionStore.exitToFreeform():
//       "Save and exit"                   -- committed-state UX framing.
//       "Drop to freeform to keep editing" -- further-edit UX framing.
//     WIRE-IDENTITY: Both buttons call the same parameterless exitToFreeform().
//     The backend (routes.py + sessionStore.ts:572-583) has ONE handler for
//     control_signal="exit_to_freeform" -- it does NOT distinguish between the
//     two mental models above.  Do NOT introduce two separate wire paths without
//     a backend protocol change (see sessionStore.ts:116).
//   - useTheme() for Prism theme-awareness, matching YamlView.tsx:164.
//   - <button type="button"> (never <div onClick>).
//   - CSS via App.css guided-completion-* classes with design tokens.
//   - No auto-focus on mount.

import { useId } from "react";
import { Highlight, themes } from "prism-react-renderer";
import { useSessionStore } from "@/stores/sessionStore";
import { useTheme } from "@/hooks/useTheme";
import type { TerminalState } from "@/types/guided";

// ── Props ─────────────────────────────────────────────────────────────────────

interface CompletionSummaryProps {
  terminal: TerminalState;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function CompletionSummary({ terminal }: CompletionSummaryProps) {
  // Guard: only render in the completed+yaml-present state.
  if (terminal.kind !== "completed" || terminal.pipeline_yaml === null) {
    return null;
  }

  return <CompletionSummaryInner yaml={terminal.pipeline_yaml} />;
}

// ── Inner component (separated so hooks run unconditionally) ──────────────────
//
// React hooks must not be called conditionally.  The outer CompletionSummary
// returns null before reaching any hook calls.  CompletionSummaryInner is the
// always-rendered branch; it can call hooks without violating the Rules of Hooks.

interface CompletionSummaryInnerProps {
  yaml: string;
}

function CompletionSummaryInner({ yaml }: CompletionSummaryInnerProps) {
  const reactId = useId();
  const headingId = `${reactId}-heading`;
  const preId = `${reactId}-pre`;

  const exitToFreeform = useSessionStore((s) => s.exitToFreeform);
  const { resolvedTheme } = useTheme();

  // Match the theme-awareness pattern from YamlView.tsx:164.
  const highlightTheme =
    resolvedTheme === "dark" ? themes.vsDark : themes.vsLight;

  function handleSaveAndExit(): void {
    void exitToFreeform();
  }

  function handleKeepEditing(): void {
    void exitToFreeform();
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

      {/* Action row.
          Both buttons call exitToFreeform() -- wire-identical by design.
          See WIRE-IDENTITY note in file-level docstring. */}
      <div className="guided-completion-actions">
        <button
          type="button"
          className="guided-completion-save-btn"
          onClick={handleSaveAndExit}
        >
          Save and exit
        </button>
        <button
          type="button"
          className="guided-completion-edit-btn"
          onClick={handleKeepEditing}
        >
          Drop to freeform to keep editing
        </button>
      </div>
    </div>
  );
}
