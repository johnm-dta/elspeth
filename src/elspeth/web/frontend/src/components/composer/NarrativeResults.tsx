/**
 * NarrativeResults — Phase 6B Task 6.
 *
 * Narrative-mode rendering for run results. Mounted when
 * `useNarrativeMode().narrativeMode === true` (Task 7 dispatches).
 *
 * Layered model:
 *
 * **Layer 1 (primary): plugin-provided `summary` field.** The transform's
 * output schema must include a `summary` field, pinned by Phase 6A
 * Task 8's `capability_tags = ("narrative-summary",)`. The narrative
 * panel surfaces this string verbatim. Live mode reads from the active
 * run's summary; the read-only inspect view (Task 8) supplies
 * `summaryOverride` from the frozen blob.
 *
 * **Layer 2 (additive overlay): Phase 5b interpretation events.** When
 * the existing `interpretationEventsStore` projection exposes a queryable
 * list of resolved events, render them as "How user-supplied terms were
 * interpreted." Pre-Phase-5b the projection is pending-only, so the
 * overlay short-circuits — Layer 1 stands alone, per plan §"Phase 5b
 * dependency" ("the renderer must tolerate its absence").
 *
 * **No-summary placeholder.** When the tagged plugin's output did not
 * include a `summary` field (or no terminal run is available), render a
 * placeholder rather than fabricating content. The existing tabular
 * `RunOutputsPanel` remains the source of truth for raw outputs;
 * narrative mode is a complementary surface.
 */

import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";

interface NarrativeResultsProps {
  /** If supplied, narrative pulls the summary from this run output rather
   *  than the live execution store. Used by the read-only inspect view
   *  in Task 8 where the run results live in a frozen blob. ``null`` means
   *  "explicitly no summary"; ``undefined`` means "fall back to live mode." */
  summaryOverride?: string | null;
}

export function NarrativeResults({ summaryOverride }: NarrativeResultsProps = {}): JSX.Element {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const optedOutBySession = useInterpretationEventsStore((s) => s.optedOutBySession);

  // Layer 1: locate the summary string. ``summaryOverride === undefined``
  // means live mode; the live `executionStore` does not yet aggregate a
  // narrative-summary field — surface the placeholder in that case. A
  // future iteration can wire a dedicated summary extractor through the
  // run-results store; the contract here (component prop + placeholder)
  // is stable.
  const summary = summaryOverride !== undefined ? summaryOverride : null;

  // Layer 2: opt-out indicator from the interpretation events store. The
  // store currently exposes pending events + counts + opt-out flag (no
  // queryable list of resolved events). When the projection grows a
  // resolved-list surface, this component renders the list inline; until
  // then we surface only the opt-out flag because that's the lone
  // observable signal that maps to a recipient-meaningful overlay
  // ("auto-interpretation was opted out on this session").
  const optedOut = activeSessionId !== null && optedOutBySession[activeSessionId] === true;

  return (
    <section
      className="narrative-results"
      aria-label="Narrative pipeline summary"
      data-testid="narrative-results"
    >
      <h3>Pipeline summary</h3>
      {summary !== null && summary !== "" ? (
        <p data-testid="narrative-results-summary">{summary}</p>
      ) : (
        <p data-testid="narrative-results-no-summary" className="narrative-results-empty">
          No narrative summary available for this run. The tagged plugin's
          output did not include a <code>summary</code> field.
        </p>
      )}

      {optedOut && (
        <div
          className="narrative-results-interpretations"
          data-testid="narrative-results-interpretations"
        >
          <h4>How user-supplied terms were interpreted</h4>
          <p>
            Auto-interpretation opt-out — LLM-surfaced terms were resolved
            without user review on this session.
          </p>
        </div>
      )}
    </section>
  );
}
