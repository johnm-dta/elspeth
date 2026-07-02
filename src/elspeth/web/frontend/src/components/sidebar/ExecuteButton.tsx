import { useId, useState } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { sortedSourceEntries, sourceComponentId } from "@/utils/compositionState";
import type { CompositionState } from "@/types/index";

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
 * keep native `disabled`: they have their own visible surfaces (audit
 * readiness panel, inline banners, the progress view) and carry no
 * button-attached reason to reach.
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
        // `title` only when blocked by pending interpretations — the
        // pre-existing disabled-on-invalid-validation case has its own
        // surface (the audit-readiness panel and inline error banners) and
        // a tooltip here would be redundant.
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
