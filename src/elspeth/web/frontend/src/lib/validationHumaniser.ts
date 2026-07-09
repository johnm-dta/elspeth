// ============================================================================
// validationHumaniser — plain-language rendering of backend validation dumps.
//
// Relocated out of components/chat/guided/PipelineValidationSummary.tsx
// (elspeth-d9e5d157cb) so NON-component layers can share it. The chat
// validation-failure injection lives in a store (stores/subscriptions.ts) and
// could not reach the humaniser while it lived in the component tree, so it
// leaked raw engine dumps + a "[type] id:" internal-id prefix into chat. This
// pure module (no React, no store) is importable by the store, the audit panel
// (ReadinessRowDetail), and the chat components alike — so all four
// novice-register surfaces name steps and phrase failures identically.
//
// Depends only on the pure gloss leaf (pipelineGloss) + types; that file
// imports only types/utils, so there is no import cycle.
// ============================================================================

import { buildPlainPhraseMap, UNKNOWN_COMPONENT_PHRASE } from "@/components/chat/guided/pipelineGloss";
import type { CompositionState } from "@/types/index";

/**
 * Match the engineer-grade contract-violation dumps that reach the validation
 * surfaces. Three backend producers write them (all egress verbatim via the
 * validation endpoint's `message=str(exc)`):
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

/**
 * Match the interpretation-review-pending dumps that reach this surface
 * verbatim from `_format_interpretation_site` (web/execution/validation.py):
 *   "pipeline_decision review pending for transform 'guided_xform_1': drop_raw_html_fields"
 * Capture group: [1] = component id. The kind / component_type / user_term are
 * engineer detail — they stay in the raw dump behind the expander
 * (elspeth-016f463ff0).
 */
const INTERPRETATION_REVIEW_PENDING_RE =
  /^[a-z_]+ review pending for [a-z_]+ '([^']+)': /;

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
 * raw dump preserved for the expander; interpretation-review-pending dumps
 * become "The <step> step is waiting for your review." (elspeth-016f463ff0)
 * using the SAME step-label mapping the acknowledgement cards render, when the
 * caller supplies `stepLabelFor`; anything else passes through untouched.
 * Exported for tests.
 */
export function humaniseValidationMessage(
  message: string,
  phraseFor: (componentId: string | null) => string,
  stepLabelFor?: (componentId: string) => string | null,
): HumanisedFinding {
  const pendingMatch = INTERPRETATION_REVIEW_PENDING_RE.exec(message);
  if (pendingMatch !== null && stepLabelFor !== undefined) {
    const stepLabel = stepLabelFor(pendingMatch[1]);
    return {
      // A null step label (component absent from the composition) falls back
      // to a generic phrase — never the raw internal id.
      headline:
        stepLabel !== null
          ? `The ${stepLabel} step is waiting for your review.`
          : "A step is waiting for your review.",
      raw: message,
    };
  }
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
 * the resolver tries verbatim first, then the stripped form. Shared by the
 * summary headline, the wire-stage blockers list (ChatPanel), the audit panel
 * and the chat injection so every surface names steps identically.
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
    const strippedPhrase = phraseMap.get(stripped);
    if (strippedPhrase !== undefined) return strippedPhrase;
    const generatedPhrase = fallbackPhraseForGeneratedId(stripped);
    if (generatedPhrase !== null) return generatedPhrase;
    for (const [knownId, phrase] of phraseMap.entries()) {
      if (componentIdHasKnownToken(stripped, knownId)) {
        return phrase;
      }
    }
    return UNKNOWN_COMPONENT_PHRASE;
  };
}

const MIN_FUZZY_COMPONENT_TOKEN_LENGTH = 4;
const COMPONENT_ROLE_TOKENS = new Set([
  "source",
  "input",
  "read",
  "sink",
  "output",
  "write",
  "node",
  "transform",
  "xform",
]);

function componentIdHasKnownToken(componentId: string, knownId: string): boolean {
  const componentTokens = new Set(componentIdTokens(componentId));
  if (componentTokens.size === 0) return false;
  const meaningfulKnownTokens = componentIdTokens(knownId).filter(
    (token) =>
      token.length >= MIN_FUZZY_COMPONENT_TOKEN_LENGTH &&
      !COMPONENT_ROLE_TOKENS.has(token),
  );
  if (meaningfulKnownTokens.length === 0) return false;
  return meaningfulKnownTokens.every((token) => componentTokens.has(token));
}

function componentIdTokens(componentId: string): string[] {
  return componentId.toLowerCase().split(/[_:-]+/).filter(Boolean);
}

function fallbackPhraseForGeneratedId(componentId: string): string | null {
  const id = componentId.toLowerCase();
  const role = firstGeneratedRole(id);
  if (role === "source") {
    if (/csv/.test(id)) return "read your CSV";
    if (/json/.test(id)) return "read your JSON file";
    if (/(api|http|url|web|scrape)/.test(id)) return "read from an API";
    return "read your data";
  }
  if (role === "output") {
    if (/csv/.test(id)) return "write a CSV";
    if (/json/.test(id)) return "write a JSON file";
    return "write the results";
  }
  if (role === "transform") {
    if (/(scrape|fetch|web|http|url)/.test(id)) return "scrape each page";
    return "process each row";
  }
  if (/(scrape|fetch|web|http|url)/.test(id)) return "scrape each page";
  if (/csv/.test(id)) return "write a CSV";
  if (/json/.test(id)) return "write a JSON file";
  if (/(llm|rate|score|classif|summari[sz]e|xform|transform)/.test(id)) {
    return "process each row";
  }
  return null;
}

type GeneratedRole = "source" | "output" | "transform";

function firstGeneratedRole(id: string): GeneratedRole | null {
  const tokens = componentIdTokens(id);
  for (const token of tokens) {
    if (/^(source|input|read)$/.test(token)) return "source";
    if (/^(sink|output|write)$/.test(token)) return "output";
    if (/^(transform|xform|llm|rate|score|classif.*|summari[sz]e.*)$/.test(token)) {
      return "transform";
    }
  }
  return null;
}

/**
 * Compose the status-line body: "<N> <label> — <headline>", prefixing the
 * finding's plain-language step name ("'rate each row': …") only when the
 * finding is attributed to a resolvable component AND was not already
 * humanised. A settings-level finding (component_id === null — e.g. the
 * missing source/output reframe or empty_pipeline) owns no step, so the
 * possessive prefix would render a bare "'this step':" — omit it there
 * (elspeth-901a404926).
 */
export function formatFindingBody(
  count: number,
  label: string,
  finding: HumanisedFinding,
  componentId: string | null,
  phraseFor: (componentId: string | null) => string,
): string {
  const attributed = finding.raw === null && componentId !== null;
  return attributed
    ? `${count} ${label} — '${phraseFor(componentId)}': ${finding.headline}`
    : `${count} ${label} — ${finding.headline}`;
}
