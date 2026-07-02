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
import type { CompositionState } from "@/types";
import { buildPlainPhraseMap, UNKNOWN_COMPONENT_PHRASE } from "./pipelineGloss";

/**
 * Match the engineer-grade contract-violation dumps that reach this surface.
 * Three backend producers write them (all egress verbatim via the validation
 * endpoint's `message=str(exc)`):
 *   - composer authoring (web/composer/state.py):
 *     "Schema contract violation: 'producer' -> 'consumer'. …"
 *   - DAG runtime preflight (core/dag/graph.py) — the live-verified format:
 *     "Schema contract violation: edge 'producer' → 'consumer'\n  Consumer …"
 *   - edge-contract preflight (web/execution/validation.py):
 *     "Edge contract violation between producer node 'X' (schema 'S') and
 *      consumer node 'Y' (schema 'T'):\n…"
 * Capture groups: [1] = producer id, [2] = consumer id (optional).
 */
const CONTRACT_VIOLATION_RES: readonly RegExp[] = [
  /^(?:Schema|Semantic|Transform) contract violation: (?:edge )?'([^']+)'(?: (?:->|→) '([^']+)')?/,
  /^Edge contract violation between producer node '([^']+)' \(schema '[^']*'\) and consumer node '([^']+)'/,
];

export interface HumanisedFinding {
  /** Plain-language headline safe for a role="status" announcement. */
  headline: string;
  /** The verbatim engineer-grade text, kept behind a details expander.
   *  null when the original message is already headline-register. */
  raw: string | null;
}

/**
 * Humanise one validation message. Contract-violation dumps become a
 * plain-language headline ("Two steps aren't connected correctly: …") with the
 * raw dump preserved for the expander; anything else passes through untouched.
 * Exported for tests.
 */
export function humaniseValidationMessage(
  message: string,
  phraseFor: (componentId: string | null) => string,
): HumanisedFinding {
  let match: RegExpExecArray | null = null;
  for (const pattern of CONTRACT_VIOLATION_RES) {
    match = pattern.exec(message);
    if (match !== null) break;
  }
  if (match === null) {
    return { headline: message, raw: null };
  }
  const producerPhrase = phraseFor(match[1] ?? null);
  const consumerPhrase = match[2] !== undefined ? phraseFor(match[2]) : null;
  const headline =
    consumerPhrase !== null
      ? `Two steps aren't connected correctly: the "${producerPhrase}" step's output doesn't match what "${consumerPhrase}" expects.`
      : `A step isn't connected correctly: "${producerPhrase}" doesn't match what the next step expects.`;
  return { headline, raw: message };
}

/**
 * Component-id → plain-phrase resolver over the pipeline gloss. Contract dumps
 * prefix node ids ("node:rater") while the gloss map keys on the bare id, so
 * the resolver tries verbatim first, then the stripped form. Shared by this
 * summary's headline and the wire-stage blockers list (ChatPanel) so both
 * surfaces name steps identically.
 */
export function makePhraseFor(
  compositionState: CompositionState | null | undefined,
): (componentId: string | null) => string {
  const phraseMap = buildPlainPhraseMap(compositionState);
  return (componentId: string | null): string => {
    if (componentId === null) return UNKNOWN_COMPONENT_PHRASE;
    const direct = phraseMap.get(componentId);
    if (direct !== undefined) return direct;
    const stripped = componentId.replace(/^(node|source|output):/, "");
    return phraseMap.get(stripped) ?? UNKNOWN_COMPONENT_PHRASE;
  };
}

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
    const finding = humaniseValidationMessage(first.message, phraseFor);
    rawDetail = finding.raw;
    body =
      finding.raw !== null
        ? `${errors.length} ${label} — ${finding.headline}`
        : `${errors.length} ${label} — '${phraseFor(first.component_id)}': ${finding.headline}`;
    suggestion = first.suggestion;
  } else if (warnings.length > 0) {
    const first = warnings[0];
    tone = "warning";
    glyph = "⚠";
    const label = warnings.length === 1 ? "warning" : "warnings";
    const finding = humaniseValidationMessage(first.message, phraseFor);
    rawDetail = finding.raw;
    body =
      finding.raw !== null
        ? `${warnings.length} ${label} — ${finding.headline}`
        : `${warnings.length} ${label} — '${phraseFor(first.component_id)}': ${finding.headline}`;
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
