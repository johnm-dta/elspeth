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
 * panel surfaces this string verbatim (rendered as Markdown via the
 * reused `MarkdownRenderer` from `components/chat/`). The read-only
 * inspect view (Task 8) supplies `summaryOverride` from the frozen blob;
 * live mode fetches the run's outputs manifest and walks the previews of
 * file artifacts, extracting `summary` values per plan 19b:349.
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
 *
 * **Download affordance (plan 19b:342).** Alongside the narrative
 * summary, the panel exposes a "Download full output" affordance against
 * the active run's first downloadable file artifact. The backend
 * `/content` endpoint requires `Authorization: Bearer` (api/client.ts
 * lines 877-883) so the affordance is a button that invokes
 * `downloadRunOutputContent` and triggers a synthetic anchor — not a
 * plain `<a href>`, which would 401 on top-level navigation. The
 * testid (`narrative-results-download-link`) is preserved from the
 * plan's wording for cross-document grep continuity.
 */

import { useEffect, useMemo, useState } from "react";

import {
  downloadRunOutputContent,
  fetchRunOutputPreview,
  fetchRunOutputs,
} from "@/api/client";
import { MarkdownRenderer } from "@/components/chat/MarkdownRenderer";
import { useExecutionStore } from "@/stores/executionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import type { RunOutputArtifact } from "@/types/index";

interface NarrativeResultsProps {
  /** If supplied, narrative pulls the summary from this run output rather
   *  than the live execution store. Used by the read-only inspect view
   *  in Task 8 where the run results live in a frozen blob. ``null`` means
   *  "explicitly no summary"; ``undefined`` means "fall back to live mode." */
  summaryOverride?: string | null;
}

/** Pick the first file artifact whose `downloadable` flag is not
 *  explicitly false. ``downloadable === undefined`` means a pre-rollout
 *  backend; api/client docstring at types/index.ts:715-720 documents the
 *  "missing → caller treats as optimistic show-the-button" semantic.
 *  Returns null when no candidate exists. */
function pickDownloadableFileArtifact(
  artifacts: ReadonlyArray<RunOutputArtifact>,
): RunOutputArtifact | null {
  for (const a of artifacts) {
    if (a.artifact_type !== "file") continue;
    if (a.downloadable === false) continue;
    return a;
  }
  return null;
}

/** Walk every row in a JSONL or JSON preview, return the rows whose
 *  ``summary`` field is a non-empty string. Plan 19b:349 — "find the last
 *  output row that has a `summary` field; if multiple, concatenate with
 *  blank lines." The preview content is Tier-3 external data (parsed from
 *  bytes the backend serialized to disk) so this is the coerce-and-record
 *  boundary: malformed JSON lines are skipped rather than crashing.
 *  CSV / binary / text content types carry no row-shaped `summary` field
 *  and are skipped entirely (they would require column-name discovery and
 *  full-text parsing the plan does not specify). */
function extractSummariesFromPreviewText(
  text: string,
  contentType: string,
): string[] {
  const summaries: string[] = [];
  if (contentType === "jsonl") {
    for (const rawLine of text.split("\n")) {
      const line = rawLine.trim();
      if (line === "") continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        // Tier-3 boundary: a malformed line is recorded as absence, not
        // a crash. Skip and continue.
        continue;
      }
      if (typeof parsed !== "object" || parsed === null) continue;
      const candidate = (parsed as Record<string, unknown>).summary;
      if (typeof candidate === "string" && candidate.length > 0) {
        summaries.push(candidate);
      }
    }
  } else if (contentType === "json") {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return [];
    }
    if (Array.isArray(parsed)) {
      for (const row of parsed) {
        if (typeof row !== "object" || row === null) continue;
        const candidate = (row as Record<string, unknown>).summary;
        if (typeof candidate === "string" && candidate.length > 0) {
          summaries.push(candidate);
        }
      }
    } else if (typeof parsed === "object" && parsed !== null) {
      const candidate = (parsed as Record<string, unknown>).summary;
      if (typeof candidate === "string" && candidate.length > 0) {
        summaries.push(candidate);
      }
    }
  }
  return summaries;
}

interface LiveOutputsState {
  /** First downloadable file artifact (for the Download affordance). */
  downloadArtifact: RunOutputArtifact | null;
  /** Concatenated non-empty `summary` strings extracted from the previews
   *  of every file artifact in the manifest. ``null`` until the fetch
   *  settles; empty-string when settled-with-no-summaries. */
  extractedSummary: string | null;
}

function triggerBrowserDownload(data: Blob, filename: string): void {
  // Mirrors the RunOutputsPanel:58-67 helper. The /content endpoint
  // requires Authorization: Bearer, so we fetch-then-object-URL rather
  // than rendering an <a href>.
  const url = URL.createObjectURL(data);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export function NarrativeResults({ summaryOverride }: NarrativeResultsProps = {}): JSX.Element {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const optedOutBySession = useInterpretationEventsStore((s) => s.optedOutBySession);
  const resolvedBySession = useInterpretationEventsStore((s) => s.resolvedBySession);
  const activeRunId = useExecutionStore((s) => s.activeRunId);
  const runs = useExecutionStore((s) => s.runs);

  // Live-mode outputs: fetched only when summaryOverride is undefined AND
  // an activeRunId is set. The fetch loads the manifest, then for each
  // file artifact the bounded-preview endpoint, then walks rows looking
  // for a `summary` field per plan 19b:349.
  const [liveOutputs, setLiveOutputs] = useState<LiveOutputsState | null>(null);

  useEffect(() => {
    // When the inspect-view passes summaryOverride, skip the live fetch
    // entirely — the frozen blob is the source of truth and a parallel
    // fetch would only introduce a race the AC4-precedence test guards
    // against.
    if (summaryOverride !== undefined) return;
    if (activeRunId === null) {
      setLiveOutputs(null);
      return;
    }

    const runId = activeRunId;
    let cancelled = false;

    (async () => {
      // Tier-3 boundary: an unhealthy backend or in-flight run can return
      // 4xx/5xx here. The narrative surface degrades to the placeholder
      // rather than crashing the panel; the existing RunOutputsPanel
      // remains the canonical surface for output-fetch error reporting.
      let artifacts: ReadonlyArray<RunOutputArtifact>;
      try {
        const manifest = await fetchRunOutputs(runId);
        artifacts = manifest.artifacts;
      } catch {
        if (cancelled) return;
        setLiveOutputs({ downloadArtifact: null, extractedSummary: "" });
        return;
      }

      const downloadArtifact = pickDownloadableFileArtifact(artifacts);

      const summaries: string[] = [];
      for (const artifact of artifacts) {
        if (artifact.artifact_type !== "file") continue;
        try {
          const preview = await fetchRunOutputPreview(runId, artifact.artifact_id);
          summaries.push(
            ...extractSummariesFromPreviewText(
              preview.preview_text,
              preview.content_type,
            ),
          );
        } catch {
          // A purge-race or content-type mismatch on one artifact must
          // not abort summary extraction for the rest of the manifest.
          continue;
        }
      }

      if (cancelled) return;
      setLiveOutputs({
        downloadArtifact,
        extractedSummary: summaries.join("\n\n"),
      });
    })();

    return () => {
      cancelled = true;
    };
  }, [summaryOverride, activeRunId]);

  // Layer 1: locate the summary string. Precedence: explicit override
  // (frozen-blob inspect view) → live-extracted concatenation → null.
  // ``summaryOverride === null`` is "explicitly no summary"; ``""`` is
  // empty; both surface the placeholder via the truthy gate below.
  const summary: string | null =
    summaryOverride !== undefined
      ? summaryOverride
      : liveOutputs !== null && liveOutputs.extractedSummary !== ""
        ? liveOutputs.extractedSummary
        : null;

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

  // Download affordance source: when the frozen-blob inspect view (Task 8)
  // mounts NarrativeResults, there is no live executionStore run to fetch
  // outputs from — the affordance is suppressed. Live mode surfaces the
  // first downloadable file artifact from the just-fetched manifest.
  const downloadArtifact: RunOutputArtifact | null =
    summaryOverride !== undefined
      ? null
      : (liveOutputs?.downloadArtifact ?? null);

  const handleDownload = async (): Promise<void> => {
    if (downloadArtifact === null) return;
    if (activeRunId === null) return;
    try {
      const { data, filename } = await downloadRunOutputContent(
        activeRunId,
        downloadArtifact.artifact_id,
      );
      triggerBrowserDownload(data, filename);
    } catch {
      // Download failures are surfaced by the canonical RunOutputsPanel
      // banner; the narrative affordance is a complementary entry point,
      // not an error-reporting surface.
    }
  };

  return (
    <section
      className="narrative-results"
      aria-label="Narrative pipeline summary"
      data-testid="narrative-results"
    >
      <h3>Pipeline summary</h3>
      {summary !== null && summary !== "" ? (
        <div data-testid="narrative-results-summary">
          <MarkdownRenderer content={summary} />
        </div>
      ) : (
        <p data-testid="narrative-results-no-summary" className="narrative-results-empty">
          No narrative summary available for this run. The tagged plugin's
          output did not include a <code>summary</code> field.
        </p>
      )}

      {downloadArtifact !== null && (
        <p className="narrative-results-download">
          <button
            type="button"
            data-testid="narrative-results-download-link"
            onClick={() => void handleDownload()}
          >
            Download full output
          </button>
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
