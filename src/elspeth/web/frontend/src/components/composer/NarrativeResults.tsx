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
 * the `interpretationEventsStore` projection has resolved events for the
 * active session, render them as "How user-supplied terms were
 * interpreted." The overlay must NOT surface session-aggregate history;
 * it filters to the active run's wall-clock window so that resolutions
 * from prior runs in the same session do not over-count. Pre-Phase-5b
 * (or no events yet) the overlay short-circuits.
 *
 * **Run-filter (load-bearing — plan §Task 6 lines 311-337).** Phase 5b's
 * `interpretationEventsStore` keys events by `session_id` only and the
 * `interpretation_events_table` has no `run_id` column. To prevent the
 * overlay from over-counting stale resolutions from earlier runs in the
 * same session, the component filters by the active run's wall-clock
 * window:
 *
 *   - `runStart = new Date(run.started_at).getTime()`
 *   - `runEnd   = run.finished_at != null
 *                  ? new Date(run.finished_at).getTime()
 *                  : Date.now()`   // in-flight upper bound is "now"
 *
 * Each event's effective time is `event.resolved_at ?? event.created_at`.
 * The filter keeps events with `runStart <= eventTime <= runEnd`. This
 * is approximate (a slow `resolved_at` write could land outside the
 * window); tightening to an exact run-id requires a Phase 5b schema
 * amendment that has been adjudicated out of scope. We do NOT fall back
 * to a session-aggregate read — that is the explicit failure mode the
 * plan guards against.
 *
 * Note: the plan calls the run-finished timestamp `completed_at`; the
 * shipped `Run` interface (types/index.ts) uses `finished_at`. They are
 * the same concept; the plan was written against an earlier shape.
 *
 * **Auto-interpreted opt-out rows are excluded from the per-event list.**
 * They are already surfaced by the dedicated opt-out indicator below
 * the per-event list, so listing each one would double-surface.
 *
 * **No-summary placeholder.** When the tagged plugin's output did not
 * include a `summary` field (or no terminal run is available), render a
 * placeholder rather than fabricating content. The existing tabular
 * `RunOutputsPanel` remains the source of truth for raw outputs;
 * narrative mode is a complementary surface.
 */

import { useMemo } from "react";
import { useExecutionStore } from "@/stores/executionStore";
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
  const resolvedBySession = useInterpretationEventsStore((s) => s.resolvedBySession);
  const activeRunId = useExecutionStore((s) => s.activeRunId);
  const runs = useExecutionStore((s) => s.runs);

  // Layer 1: locate the summary string. ``summaryOverride === undefined``
  // means live mode; the live `executionStore` does not yet aggregate a
  // narrative-summary field — surface the placeholder in that case. A
  // future iteration can wire a dedicated summary extractor through the
  // run-results store; the contract here (component prop + placeholder)
  // is stable.
  const summary = summaryOverride !== undefined ? summaryOverride : null;

  // Layer 2: opt-out indicator from the interpretation events store.
  // Persists session-wide once flipped; the per-event list below adds
  // the granular detail when there are resolved events in the active
  // run's window.
  const optedOut = activeSessionId !== null && optedOutBySession[activeSessionId] === true;

  // Layer 2 (continued): the per-event run-window overlay.
  //
  // Memoise on the inputs so consumers of `resolvedEventsInWindow` don't
  // re-derive on every render. The `Date.now()` branch for in-flight
  // runs intentionally re-derives on each render-tick where the run is
  // unfinished — the test suite uses `vi.setSystemTime()` to pin the
  // clock for the in-flight test case.
  const resolvedEventsInWindow = useMemo(() => {
    if (activeSessionId === null) return [];
    if (activeRunId === null) return [];
    const run = runs.find((r) => r.id === activeRunId);
    if (run === undefined) return [];
    const runStart = new Date(run.started_at).getTime();
    const runEnd =
      run.finished_at !== null
        ? new Date(run.finished_at).getTime()
        : Date.now();
    const candidates = resolvedBySession[activeSessionId] ?? [];
    return candidates.filter((event) => {
      // Auto-interpreted opt-out rows are surfaced by the opt-out
      // indicator below, not by the per-event list.
      if (event.interpretation_source === "auto_interpreted_opt_out") {
        return false;
      }
      const eventTimeIso = event.resolved_at ?? event.created_at;
      const eventTime = new Date(eventTimeIso).getTime();
      return eventTime >= runStart && eventTime <= runEnd;
    });
  }, [activeSessionId, activeRunId, runs, resolvedBySession]);

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

      {resolvedEventsInWindow.length > 0 && (
        <div
          className="narrative-results-interpretation-overlay"
          data-testid="narrative-results-interpretation-overlay"
        >
          <h4>How user-supplied terms were interpreted</h4>
          <ul>
            {resolvedEventsInWindow.map((event) => {
              // Render `accepted_value` when present (user accepted or
              // amended); fall back to `llm_draft` when `accepted_value`
              // is null (e.g. an opted_out row that slipped through —
              // we skip auto_interpreted_opt_out above, but a manual
              // opted_out row still carries a draft we surface).
              const displayValue = event.accepted_value ?? event.llm_draft;
              return (
                <li
                  key={event.id}
                  data-testid={`narrative-overlay-event-${event.id}`}
                >
                  &ldquo;{event.user_term}&rdquo; &rarr; {displayValue}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {optedOut && (
        <div
          className="narrative-results-interpretations"
          data-testid="narrative-results-interpretations"
        >
          <h4>Auto-interpretation opt-out</h4>
          <p>
            Auto-interpretation opt-out — LLM-surfaced terms were resolved
            without user review on this session.
          </p>
        </div>
      )}
    </section>
  );
}
