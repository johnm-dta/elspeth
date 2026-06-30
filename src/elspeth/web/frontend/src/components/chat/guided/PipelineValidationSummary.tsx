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
// Register note (N3): in live-guided the rail SideRailValidationBanner shows
// the same status in a TECHNICAL register; this is the PLAIN-language, novice
// register kept in-column for both surfaces. The overlap is accepted (rail =
// detail, column = signal).
// ============================================================================

import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { buildPlainPhraseMap, UNKNOWN_COMPONENT_PHRASE } from "./pipelineGloss";

export function PipelineValidationSummary(): JSX.Element {
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

  const phraseMap = buildPlainPhraseMap(compositionState);
  const phraseFor = (componentId: string | null): string =>
    (componentId !== null ? phraseMap.get(componentId) : undefined) ??
    UNKNOWN_COMPONENT_PHRASE;

  const errors = validationResult.errors ?? [];
  const warnings = validationResult.warnings ?? [];

  let tone: "ok" | "warning" | "error";
  let glyph: string;
  let body: string;

  if (errors.length > 0) {
    const first = errors[0];
    tone = "error";
    glyph = "✕";
    const label = errors.length === 1 ? "problem to fix" : "problems to fix";
    body = `${errors.length} ${label} — '${phraseFor(first.component_id)}': ${first.message}`;
  } else if (warnings.length > 0) {
    const first = warnings[0];
    tone = "warning";
    glyph = "⚠";
    const label = warnings.length === 1 ? "warning" : "warnings";
    body = `${warnings.length} ${label} — '${phraseFor(first.component_id)}': ${first.message}`;
  } else {
    tone = "ok";
    glyph = "✓";
    body = "Looks good — no problems found.";
  }

  return (
    <p
      className={`pipeline-validation-summary pipeline-validation-summary--${tone}`}
      data-testid="pipeline-validation-summary"
      role="status"
    >
      <span className="pipeline-validation-summary-glyph" aria-hidden="true">
        {glyph}
      </span>{" "}
      {body}
    </p>
  );
}
