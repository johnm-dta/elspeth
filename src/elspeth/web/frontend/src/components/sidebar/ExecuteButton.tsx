import { useId, useState } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useAuditReadinessStore } from "@/stores/auditReadinessStore";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { sortedSourceEntries, sourceComponentId } from "@/utils/compositionState";
import type { CompositionState } from "@/types/index";
import type { ReadinessRowId } from "@/types/api";

/**
 * Run-button tooltip text used when a pending interpretation event blocks
 * execution. Exported for the corresponding test so the assertion is pinned
 * against the exact string the spec table requires (18b-phase-5b-frontend.md
 * line 705: `title="Resolve pending interpretation first."`).
 */
export const INTERPRETATION_PENDING_RUN_BLOCK_TITLE =
  "Resolve pending interpretation first.";

/**
 * Transform plugins that reach the network during a run (page fetches and
 * external analysis services). Used only to phrase the pre-run disclosure —
 * which nodes appear is always derived from the actual pipeline config.
 */
const NETWORK_FETCH_PLUGINS = new Set([
  "web_scrape",
  "azure_content_safety",
  "azure_prompt_shield",
  "azure_document_intelligence",
]);

/**
 * Derive the pre-run egress disclosure lines from the actual composition
 * (elspeth-c18ad229cc). Nothing here is hardcoded pipeline content: each
 * line names the configured components (sources, LLM nodes and their model
 * option, network-fetching transforms, output sinks) from the live
 * CompositionState. Exported for tests.
 */
export function buildRunEgressSummary(
  compositionState: CompositionState | null,
): string[] {
  if (!compositionState) return [];
  const lines: string[] = [];

  const sources = sortedSourceEntries(compositionState).map(
    ([sourceName, source]) =>
      `${sourceComponentId(sourceName)} (${source.plugin})`,
  );
  if (sources.length > 0) {
    lines.push(`Reads source data: ${sources.join(", ")}.`);
  }

  // `?.` on options: display-only derivation that must tolerate partially
  // formed compositions mid-authoring rather than crash the Run button.
  const llmNodes = compositionState.nodes
    .filter(
      (node) =>
        typeof node.options?.model === "string" ||
        (node.plugin ?? "").includes("llm"),
    )
    .map((node) =>
      typeof node.options?.model === "string"
        ? `${node.id} (model ${node.options.model})`
        : node.id,
    );
  if (llmNodes.length > 0) {
    lines.push(`Sends rows to the configured LLM: ${llmNodes.join(", ")}.`);
  }

  const networkNodes = compositionState.nodes
    .filter((node) => node.plugin !== null && NETWORK_FETCH_PLUGINS.has(node.plugin))
    .map((node) => `${node.id} (${node.plugin})`);
  if (networkNodes.length > 0) {
    lines.push(`Fetches over the network: ${networkNodes.join(", ")}.`);
  }

  const outputs = compositionState.outputs.map(
    (output) => `${output.name} (${output.plugin})`,
  );
  if (outputs.length > 0) {
    lines.push(`Writes output: ${outputs.join(", ")}.`);
  }

  return lines;
}

/**
 * Which audit-readiness rows are load-bearing for `canExecute` below, and
 * which are informational/advisory only (elspeth-088bf83922 finding T-2,
 * option (a) — legibility, NOT new gating). Read together with
 * `canExecute`: `validationResult?.is_valid === true` corresponds to the
 * `validation` row; `!isRunBlocked` corresponds to the `llm_interpretations`
 * row (both are driven by the same interpretationEventsStore pending/
 * opted-out state used to compute `isRunBlocked` below). The other four
 * rows (plugin_trust, provenance, retention, secrets) never appear in
 * `canExecute` and are always advisory.
 *
 * This file is the single source of truth for what actually gates Run —
 * AuditReadinessRow (components/audit) imports this function rather than
 * re-deriving the classification, so the audit panel's "Blocks Run" /
 * "Advisory" labelling cannot drift from the real predicate below. The
 * exhaustive switch (the `never` default arm) fails the build if a future
 * backend row id is added without an explicit classification here.
 */
export function isRunGatingReadinessRow(id: ReadinessRowId): boolean {
  switch (id) {
    case "validation":
    case "llm_interpretations":
      return true;
    case "plugin_trust":
    case "provenance":
    case "retention":
    case "secrets":
      return false;
    default: {
      const _exhaustive: never = id;
      throw new Error(`unknown readiness row id: ${String(_exhaustive)}`);
    }
  }
}

/** Which of the (up to) three run-blocking gates is currently active, for
 *  the plain-language reason text rendered under the button
 *  (elspeth-088bf83922 T-2). Priority order: an in-flight run takes
 *  precedence (nothing else matters until it finishes); pending
 *  interpretation review is next (it also drives the dedicated
 *  aria-disabled/title/aria-describedby treatment below); structural
 *  validation is the remaining case. Returns null when none apply, i.e.
 *  when `canExecute` is true. Exported for the corresponding test. */
export type RunBlockReason = "running" | "interpretation" | "validation" | "not_validated";

export function primaryRunBlockReason(input: {
  isExecuting: boolean;
  progressRunning: boolean;
  isRunBlocked: boolean;
  validationFailing: boolean;
  validationNotRun: boolean;
}): RunBlockReason | null {
  if (input.isExecuting || input.progressRunning) return "running";
  if (input.isRunBlocked) return "interpretation";
  // "not_validated" (no validation result yet — empty composition, or a
  // snapshot still in flight) is distinct from "validation" (a result exists
  // and reports errors). Conflating them made the button claim "fix the
  // validation errors" when none had been computed and none were shown
  // anywhere (elspeth-088bf83922 review follow-up).
  if (input.validationNotRun) return "not_validated";
  if (input.validationFailing) return "validation";
  return null;
}

/** Plain-language text per `RunBlockReason`, rendered visibly below the
 *  button. The `interpretation` entry reuses
 *  INTERPRETATION_PENDING_RUN_BLOCK_TITLE verbatim rather than restating it
 *  — one string, two channels (the existing title/aria-describedby pair for
 *  mouse/some-AT users, this visible line for everyone else). */
const RUN_BLOCK_REASON_TEXT: Record<RunBlockReason, string> = {
  running: "The pipeline is already running.",
  interpretation: INTERPRETATION_PENDING_RUN_BLOCK_TITLE,
  validation: "Fix the validation errors shown in the Audit panel before running.",
  not_validated: "This pipeline hasn't been validated yet.",
};

/**
 * Run-pipeline button (Phase 2C, with Phase 5b.18b.7 interpretation gating).
 *
 * Gating predicate (spec 18b lines 698-722):
 *
 *   isRunBlocked = !optedOut && Object.keys(pendingBySession[sessionId] ?? {}).length > 0
 *
 * When `isRunBlocked` is true the button:
 *   - stays in the tab order (NO native `disabled` — that would remove the
 *     button from the tab order and make the blocked reason unreachable for
 *     exactly the keyboard/screen-reader users it exists for, WCAG 4.1.2 /
 *     elspeth-94c32de486),
 *   - sets `aria-disabled="true"` so AT users hear the disabled state,
 *   - no-ops in onClick,
 *   - sets `title` to the spec-required tooltip string,
 *   - sets `aria-describedby` pointing to a visually-hidden span with the
 *     same text so screen readers receive the affordance text (a
 *     `title` attribute alone is not reliably announced).
 *
 * Other not-runnable states (validation failing, already executing/running)
 * keep native `disabled` — the WCAG 4.1.2 concern above is specific to the
 * interpretation block, which is the only state that removes reachability
 * from a mouseless/AT user if left natively disabled. They still get a
 * plain-language reason: see `primaryRunBlockReason` below.
 *
 * The opt-out path is the gate's complement: a session that has opted out
 * of interpretation review runs freely (the backend bakes auto-
 * interpretations directly into prompt templates and records them as
 * `interpretation_source='auto_interpreted_opt_out'` in the audit trail).
 *
 * Egress disclosure (elspeth-c18ad229cc): a runnable click first opens a
 * ConfirmDialog summarising what the run will reach (derived from the live
 * composition), mirroring the tutorial's Run-step disclosure. A per-session
 * "don't ask again" opt-out (executionStore.runDisclosureAckBySession)
 * keeps it from becoming click-through noise.
 *
 * Gate legibility (elspeth-088bf83922 T-2, option (a)): the audit-readiness
 * panel's rows other than validation/llm_interpretations never block Run —
 * this button previously gave no hint of that distinction. Two small,
 * NON-gating additions (`canExecute` itself is untouched):
 *   - when disabled, a visible one-line reason ("The pipeline is already
 *     running." / the interpretation-pending line / a validation pointer)
 *     renders below the button, driven by `primaryRunBlockReason`;
 *   - when enabled but the audit snapshot has a non-green advisory row
 *     (plugin_trust/provenance/retention/secrets), a single line notes
 *     that advisory checks don't block Run.
 */
export function ExecuteButton(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);
  const validationResult = useExecutionStore((s) => s.validationResult);
  const isExecuting = useExecutionStore((s) => s.isExecuting);
  const progress = useExecutionStore((s) => s.progress);
  const execute = useExecutionStore((s) => s.execute);
  const disclosureAcknowledged = useExecutionStore((s) =>
    activeSessionId ? s.runDisclosureAckBySession[activeSessionId] === true : false,
  );
  const acknowledgeRunDisclosure = useExecutionStore(
    (s) => s.acknowledgeRunDisclosure,
  );

  // Phase 5b.18b.7 — interpretation-review run-button gating. Subscribe to
  // the per-session sub-maps (not the whole store state) so the button
  // re-renders precisely when its inputs change.
  const pendingInterpretationsBySession = useInterpretationEventsStore(
    (s) => s.pendingBySession,
  );
  const optedOutInterpretationsBySession = useInterpretationEventsStore(
    (s) => s.optedOutBySession,
  );

  // Gate legibility (elspeth-088bf83922 T-2): the version-matched audit
  // snapshot, read the same way AuditReadinessPanel.tsx guards its own
  // snapshot selector — a cached snapshot for a stale composition version
  // must not be used to decide whether to show the advisory note below.
  const compositionVersion = compositionState?.version ?? null;
  const auditSnapshot = useAuditReadinessStore((s) => {
    if (!activeSessionId || compositionVersion === null) return undefined;
    const cached = s.snapshotsBySession[activeSessionId];
    return cached?.composition_version === compositionVersion ? cached : undefined;
  });

  const reactId = useId();
  const describedById = `${reactId}-run-block-reason`;
  const [showRunDisclosure, setShowRunDisclosure] = useState(false);
  const [skipFutureDisclosure, setSkipFutureDisclosure] = useState(false);

  if (!activeSessionId) return null;

  const optedOut = optedOutInterpretationsBySession[activeSessionId] ?? false;
  const pendingCount = Object.keys(
    pendingInterpretationsBySession[activeSessionId] ?? {},
  ).length;
  const isRunBlocked = !optedOut && pendingCount > 0;

  const canExecute =
    validationResult?.is_valid === true &&
    !isExecuting &&
    progress?.status !== "running" &&
    !isRunBlocked;

  // Gate legibility (elspeth-088bf83922 T-2) — derived, non-gating. These
  // three inputs mirror `canExecute`'s own conditions exactly; none of them
  // feed back into `canExecute`.
  const blockReason = primaryRunBlockReason({
    isExecuting,
    progressRunning: progress?.status === "running",
    isRunBlocked,
    validationNotRun: validationResult == null,
    validationFailing: validationResult != null && validationResult.is_valid !== true,
  });
  const advisoryRowsNonGreen =
    auditSnapshot?.rows.some(
      (row) =>
        !isRunGatingReadinessRow(row.id) &&
        (row.status === "warning" || row.status === "error"),
    ) ?? false;

  function handleRunClick(): void {
    // Blocked-but-focusable case (aria-disabled): activation is a no-op.
    if (!canExecute || !activeSessionId) return;
    if (disclosureAcknowledged) {
      void execute(activeSessionId);
      return;
    }
    setShowRunDisclosure(true);
  }

  function handleDisclosureConfirm(): void {
    if (!activeSessionId) return;
    if (skipFutureDisclosure) {
      acknowledgeRunDisclosure(activeSessionId);
    }
    setShowRunDisclosure(false);
    void execute(activeSessionId);
  }

  const egressLines = buildRunEgressSummary(compositionState);

  return (
    <>
      <button
        type="button"
        // Plain .btn, never .btn-primary: CompletionBar's contract (its
        // docstring, per plan 19b §"Scope boundaries") renders Save-for-review
        // / Run / Export YAML as CO-EQUAL verbs with no primary emphasis. A
        // conditional btn-primary here singled Run out as the lone filled
        // accent button whenever the composition was valid, contradicting the
        // documented design (elspeth-0d37694c8c).
        className="btn side-rail-execute-btn"
        onClick={handleRunClick}
        // Native disabled ONLY for reasons without a button-attached
        // explanation. The interpretation-pending block must stay focusable
        // so its aria-describedby reason is reachable (elspeth-94c32de486);
        // shared.css styles .btn[aria-disabled="true"] identically to
        // .btn:disabled, so the visual treatment is unchanged.
        disabled={isRunBlocked ? undefined : !canExecute}
        aria-disabled={!canExecute ? true : undefined}
        aria-label="Run pipeline"
        // `title` (hover tooltip) only when blocked by pending
        // interpretations — the other not-runnable states are natively
        // `disabled`, and a `title` on a disabled element is not reliably
        // reachable by keyboard/AT users, so their reason is carried by the
        // always-visible <p> below instead (elspeth-088bf83922 T-2).
        title={isRunBlocked ? INTERPRETATION_PENDING_RUN_BLOCK_TITLE : undefined}
        aria-describedby={isRunBlocked ? describedById : undefined}
      >
        {isExecuting ? (
          <>
            <span
              className="spinner"
              role="status"
              aria-label="Starting pipeline"
            />
            Starting...
          </>
        ) : (
          "Run pipeline"
        )}
      </button>
      {/*
        Visually-hidden description for AT users. The `title` attribute on
        the button alone is not reliably announced by all screen readers
        (NVDA reads it; VoiceOver and some JAWS configurations ignore it).
        `aria-describedby` pointing at a hidden span is the WCAG-canonical
        way to surface a "why is this button disabled?" reason — which is
        also why the blocked button must remain focusable (no native
        `disabled`) for the description to be reachable at all.
      */}
      {isRunBlocked && (
        <span id={describedById} className="sr-only">
          {INTERPRETATION_PENDING_RUN_BLOCK_TITLE}
        </span>
      )}
      {/* Gate legibility (elspeth-088bf83922 T-2): a visible (not sr-only)
          one-line reason for whichever gate is currently holding Run back.
          Deliberately plain text, not a tooltip — tooltips on natively
          disabled buttons are not reliably reachable by keyboard/AT users
          (see the WCAG 4.1.2 note above), and this line is meant for every
          user, not just the interpretation-pending case that already has
          its own aria-describedby announcement. */}
      {blockReason && (
        <p
          className="side-rail-execute-reason"
          data-run-block-reason={blockReason}
        >
          {RUN_BLOCK_REASON_TEXT[blockReason]}
        </p>
      )}
      {/* Run is enabled, but the audit-readiness panel has a non-green
          advisory row (plugin trust / provenance / retention / secrets).
          These rows never gate Run — say so in one line rather than
          leaving the user to infer it from an amber/red row that did
          nothing when they ran anyway. */}
      {!blockReason && advisoryRowsNonGreen && (
        <p className="side-rail-execute-reason side-rail-execute-reason--advisory">
          Advisory checks don't block Run.
        </p>
      )}
      {showRunDisclosure && (
        <ConfirmDialog
          title="Run pipeline?"
          message={
            egressLines.length > 0
              ? "This run leaves the composer and uses your stored credentials:"
              : "This run leaves the composer and uses your stored credentials."
          }
          confirmLabel="Run pipeline"
          onConfirm={handleDisclosureConfirm}
          onCancel={() => setShowRunDisclosure(false)}
        >
          {egressLines.length > 0 && (
            <ul className="run-disclosure-summary">
              {egressLines.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          )}
          <label className="run-disclosure-opt-out">
            <input
              type="checkbox"
              checked={skipFutureDisclosure}
              onChange={(event) => setSkipFutureDisclosure(event.target.checked)}
            />
            <span>Don't ask again for this session</span>
          </label>
        </ConfirmDialog>
      )}
    </>
  );
}
