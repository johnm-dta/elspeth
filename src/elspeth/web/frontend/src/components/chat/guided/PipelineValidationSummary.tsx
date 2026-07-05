// ============================================================================
// PipelineValidationSummary — the in-column "is it OK?" signal (Slice C)
//
// GraphMiniView (the column thumbnail) renders aggregate lanes only and carries
// NO per-node validation markers; the full per-node markers live in GraphView
// inside the App-root GraphModal, one click away (expand). This component is the
// plain-language verification signal that compensates: it reads the SAME
// validationResult the modal markers use and renders a novice-register status.
//
// Findings (ValidationError / ValidationWarning) carry only `component_id` —
// no human node name — so it reuses buildPlainPhraseMap(compositionState) (the
// gloss's mapping) to render "rate each row" rather than the raw "rater".
//
// Error rendering discipline (ux review elspeth-3b35abf148):
//   * Engineer-grade dumps (e.g. "Schema contract violation: edge '…' -> …")
//     are NEVER the role="status" headline — they get a humanised headline and
//     the raw text moves behind a <details> expander, so screen readers hear
//     the plain-language line, not a serialized contract.
//   * The backend's actionable `suggestion` (e.g. "Add the missing secrets via
//     the Secrets panel") is rendered with the error instead of dropped.
//
// Register note (N3): in live-guided the rail SideRailValidationBanner shows
// the same status in a TECHNICAL register; this is the PLAIN-language, novice
// register kept in-column for both surfaces. The overlap is accepted (rail =
// detail, column = signal).
// ============================================================================

import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import {
  resolveNodePlugin,
  stepLabelForPlugin,
} from "../interpretationStepLabel";
import {
  formatFindingBody,
  humaniseValidationMessage,
  makePhraseFor,
} from "@/lib/validationHumaniser";

export interface PipelineValidationSummaryProps {
  /** Tutorial surface flag: the tutorial has no Secrets panel, so a
   *  Secrets-panel suggestion gets an honest availability note. */
  isTutorial?: boolean;
}

export function PipelineValidationSummary({
  isTutorial = false,
}: PipelineValidationSummaryProps = {}): JSX.Element {
  const validationResult = useExecutionStore((s) => s.validationResult);
  const compositionState = useSessionStore((s) => s.compositionState);

  // Neutral pre-validation state: always render a stable root so the mount /
  // D1 / parity tests can find the surface across every state.
  if (validationResult === null) {
    return (
      <p
        className="pipeline-validation-summary pipeline-validation-summary--neutral"
        data-testid="pipeline-validation-summary"
        role="status"
      >
        We'll check your pipeline as you build it.
      </p>
    );
  }

  const phraseFor = makePhraseFor(compositionState);
  // Step labels for the pending-review headline reuse the acknowledgement
  // cards' mapping (stepLabelForPlugin → "Summarise" / "Output" …) so the
  // problems strip names the step exactly as the card it points at. Null when
  // the component id cannot be resolved — the humaniser then uses a generic
  // phrase instead of echoing the internal id.
  const stepLabelFor = (componentId: string): string | null => {
    const plugin = resolveNodePlugin(compositionState, componentId);
    return plugin === null ? null : stepLabelForPlugin(plugin);
  };

  const errors = validationResult.errors ?? [];
  const warnings = validationResult.warnings ?? [];

  let tone: "ok" | "warning" | "error";
  let glyph: string;
  let body: string;
  let suggestion: string | null = null;
  let rawDetail: string | null = null;

  if (errors.length > 0) {
    const first = errors[0];
    tone = "error";
    glyph = "✕";
    const label = errors.length === 1 ? "problem to fix" : "problems to fix";
    const finding = humaniseValidationMessage(first.message, phraseFor, stepLabelFor);
    rawDetail = finding.raw;
    body = formatFindingBody(errors.length, label, finding, first.component_id, phraseFor);
    suggestion = first.suggestion;
  } else if (warnings.length > 0) {
    const first = warnings[0];
    tone = "warning";
    glyph = "⚠";
    const label = warnings.length === 1 ? "warning" : "warnings";
    const finding = humaniseValidationMessage(first.message, phraseFor, stepLabelFor);
    rawDetail = finding.raw;
    body = formatFindingBody(warnings.length, label, finding, first.component_id, phraseFor);
    suggestion = first.suggestion;
  } else {
    tone = "ok";
    glyph = "✓";
    body = "Looks good — no problems found.";
  }

  // Honest availability note (elspeth-3b35abf148 variant 4): the backend's
  // suggestion is written for the freeform surface; the tutorial has no
  // Secrets panel, so say so rather than pointing at an affordance that
  // isn't on screen. The suggestion string itself is UI-safe by design
  // (execution/validation.py never echoes secret values into it).
  const suggestionNote =
    isTutorial && suggestion !== null && /secrets panel/i.test(suggestion)
      ? " (The Secrets panel is part of the full composer, outside this tutorial.)"
      : "";

  return (
    <div
      className={`pipeline-validation-summary pipeline-validation-summary--${tone}`}
      data-testid="pipeline-validation-summary"
    >
      <p className="pipeline-validation-summary-status" role="status">
        <span className="pipeline-validation-summary-glyph" aria-hidden="true">
          {glyph}
        </span>{" "}
        {body}
      </p>
      {suggestion !== null && (
        <p className="pipeline-validation-summary-suggestion">
          {suggestion}
          {suggestionNote}
        </p>
      )}
      {rawDetail !== null && (
        <details className="pipeline-validation-summary-raw">
          <summary>Technical details</summary>
          <pre className="pipeline-validation-summary-raw-text">{rawDetail}</pre>
        </details>
      )}
    </div>
  );
}
