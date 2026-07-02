import { useEffect, useRef, useState } from "react";
import { cancelTutorialRun, runTutorialPipeline } from "@/api/client";
import { AlertBanner } from "@/components/ui";
import type { TutorialRunResponse } from "@/types/api";
import { TUTORIAL_RUN_PREAMBLE, TUTORIAL_SHIELD_OVERRIDE_CAVEAT, TURN_4_PRIMARY_BUTTON } from "./copy";
import type { RunResultRow, TutorialRunResult } from "./tutorialMachine";

interface TutorialTurn4RunProps {
  sessionId: string;
  /**
   * Fired as soon as the run's result arrives (rendered on this turn,
   * before the user clicks Continue). The parent persists the run identity
   * so a reload resumes at the audit step instead of re-executing the
   * pipeline (elspeth-918f4434b3). Optional — standalone mounts (tests)
   * may omit it.
   */
  onResult?: (result: TutorialRunResult) => void;
  onCompleted: (result: TutorialRunResult) => void;
  onCancelled: () => void;
  /**
   * Back affordance. Omitted (undefined) when the run turn has no real prior
   * step to return to — once the guided wizard is completed it is terminal and
   * non-returnable (`previousStep(run)` is null), so HelloWorldTutorial passes
   * no `onBack`. When undefined the Back button is not rendered.
   */
  onBack?: () => void;
}

/**
 * Three-phase narration timing for the run status text. Tied to fixed
 * timers (not streamed progress events — the backend run is an opaque
 * POST). Tuned so AT users hear progress at a rate that matches the
 * typical 8–12s tutorial run on a warm cache.
 */
const PHASE_MODEL_DELAY_MS = 2_000;
const PHASE_WRITE_DELAY_MS = 6_000;
const SHOW_CANCEL_DELAY_MS = 5_000;

type RunPhase = "fetch" | "model" | "write";

interface CachedRun {
  promise: Promise<TutorialRunResponse>;
  controller: AbortController;
}

/**
 * Cached by `sessionId` so React StrictMode's double-invoke of the run
 * effect coalesces to a single backend call. The cache entry stores both
 * the promise and the AbortController owning the fetch's signal — the
 * user-cancel path aborts via the cached controller and removes the entry
 * so a subsequent re-mount triggers a fresh run.
 *
 * The tutorial run is frozen-prompt (``TutorialRunRequest`` carries only
 * ``session_id`` — the backend always runs the canonical tutorial prompt),
 * so the cache key is the session id alone; there is no per-prompt identity
 * to fold into it.
 */
const tutorialRunCache = new Map<string, CachedRun>();

function getTutorialRun(sessionId: string): CachedRun {
  const existing = tutorialRunCache.get(sessionId);
  if (existing !== undefined) {
    return existing;
  }
  const controller = new AbortController();
  const promise = runTutorialPipeline(
    { session_id: sessionId },
    controller.signal,
  ).catch((err: unknown) => {
    // Drop the cache entry on failure so the user can retry without
    // hitting a stale rejected promise.
    tutorialRunCache.delete(sessionId);
    throw err;
  });
  const entry: CachedRun = { promise, controller };
  tutorialRunCache.set(sessionId, entry);
  return entry;
}

function clearTutorialRunCache(sessionId: string): void {
  tutorialRunCache.delete(sessionId);
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function hasErrorType(err: unknown, type: string): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    "error_type" in err &&
    (err as { error_type?: unknown }).error_type === type
  );
}

// Friendly copy for the ``run_already_active`` 409 (a quick retry lands while
// the previous run is still finishing server-side). Deliberately does NOT
// echo the backend's raw detail string ("run already active for session
// ...") — that string names an internal invariant, not user-facing copy.
const TUTORIAL_RUN_STILL_FINISHING_MESSAGE =
  "Your previous tutorial run is still finishing. Give it a moment, then try again.";

export function TutorialTurn4Run({
  sessionId,
  onResult,
  onCompleted,
  onCancelled,
  onBack,
}: TutorialTurn4RunProps): JSX.Element {
  const [result, setResult] = useState<TutorialRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [phase, setPhase] = useState<RunPhase>("fetch");
  const [showCancel, setShowCancel] = useState(false);
  const [retryNonce, setRetryNonce] = useState(0);
  const headingRef = useRef<HTMLHeadingElement | null>(null);

  // Move focus to the turn's heading on mount so screen-reader users
  // hear the new step. `tabIndex={-1}` on the h2 makes it
  // programmatically focusable without entering the tab order.
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  useEffect(() => {
    let active = true;
    setResult(null);
    setError(null);
    setPhase("fetch");
    setShowCancel(false);

    const cached = getTutorialRun(sessionId);
    cached.promise
      .then((response) => {
        if (!active) return;
        const arrived: TutorialRunResult = {
          runId: response.run_id,
          sourceDataHash: response.output.source_data_hash,
          rows: response.output.rows,
          discardedRowCount: response.output.discarded_row_count,
        };
        setResult(arrived);
        // Notify the parent the run identity exists, so it can be
        // persisted for resume BEFORE the user clicks Continue. Uses the
        // same latest-effect-run closure convention as onCancelled below.
        onResult?.(arrived);
      })
      .catch((err: unknown) => {
        if (!active) return;
        // AbortError on the cached promise means another consumer (the
        // user-cancel path below) aborted the run. The cancel handler
        // already dispatched onCancelled; nothing more to render here.
        if (isAbortError(err)) return;
        // Server-side cancellation (another tab's Cancel click, or the run
        // landing on "cancelled" after this fetch was already in flight)
        // surfaces as a 409 with this machine-readable error_type, distinct
        // from the generic live-run-failed error below. Route it to the
        // same cancelled state the local Cancel button drives, rather than
        // rendering it as a raw error.
        if (hasErrorType(err, "tutorial_run_cancelled")) {
          onCancelled();
          return;
        }
        setError(formatError(err));
      });

    const phaseModelTimer = window.setTimeout(() => {
      if (active) setPhase("model");
    }, PHASE_MODEL_DELAY_MS);
    const phaseWriteTimer = window.setTimeout(() => {
      if (active) setPhase("write");
    }, PHASE_WRITE_DELAY_MS);
    const cancelTimer = window.setTimeout(() => {
      if (active) setShowCancel(true);
    }, SHOW_CANCEL_DELAY_MS);

    return () => {
      active = false;
      window.clearTimeout(phaseModelTimer);
      window.clearTimeout(phaseWriteTimer);
      window.clearTimeout(cancelTimer);
      // Do NOT abort the cached controller on effect cleanup — that path
      // fires under React StrictMode's developer double-invoke as well as
      // on real unmount. The cached promise should survive the
      // double-invoke (that's the whole point of the cache); the
      // user-cancel button is the only legitimate abort trigger.
    };
    // Intentional: `onCancelled` is deliberately NOT a dep. HelloWorldTutorial
    // passes an inline arrow function, so a naive dep list would re-run this
    // effect (resetting phase/showCancel and restarting the cancel timers) on
    // every parent re-render — the effect should only re-run for a real
    // session/retry change. The closure below always calls the version
    // captured at the most recent effect run, which is fine here: onCancelled
    // is a stable "dispatch a fixed action" callback, not state-dependent.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, retryNonce]);

  const onCancelClick = (): void => {
    const cached = tutorialRunCache.get(sessionId);
    if (cached !== undefined) {
      cached.controller.abort();
      tutorialRunCache.delete(sessionId);
    }
    // Best-effort server-side cancel, fired alongside the local abort — not
    // awaited, so the UI moves to the cancelled state immediately rather
    // than blocking on network latency. Aborting only the browser fetch
    // left the backend run executing (LLM spend, sink writes continuing);
    // this call is what makes "Cancel run" honest.
    void cancelTutorialRun(sessionId).catch(() => {
      // Nothing to surface: the UI has already moved on via onCancelled()
      // below, and the endpoint is itself idempotent/best-effort.
    });
    onCancelled();
  };

  const onRetryClick = (): void => {
    clearTutorialRunCache(sessionId);
    setRetryNonce((n) => n + 1);
  };

  const phaseText = describePhase(phase);

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-run-title">
      <p className="tutorial-kicker">Run</p>
      <h2 id="tutorial-run-title" ref={headingRef} tabIndex={-1}>
        Running your pipeline.
      </h2>
      <AlertBanner tone="info" className="tutorial-disclosure">
        {TUTORIAL_RUN_PREAMBLE}
      </AlertBanner>
      <p className="tutorial-callout">{TUTORIAL_SHIELD_OVERRIDE_CAVEAT}</p>
      {result === null && error === null && (
        <>
          <div
            role="status"
            aria-busy="true"
            className="tutorial-running"
          >
            <span className="tutorial-progress-bar" aria-hidden="true" />
            <span>{phaseText}</span>
          </div>
          {showCancel && (
            <div className="tutorial-actions">
              <button
                type="button"
                className="btn"
                onClick={onCancelClick}
              >
                Cancel run
              </button>
            </div>
          )}
        </>
      )}
      {error !== null && (
        <>
          <p role="alert" className="tutorial-error">
            {error}
          </p>
          <div className="tutorial-actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={onRetryClick}
            >
              Retry
            </button>
            {onBack !== undefined && (
              <button
                type="button"
                className="tutorial-link-button"
                onClick={onBack}
              >
                Back
              </button>
            )}
          </div>
        </>
      )}
      {result !== null && (
        <>
          <p className="tutorial-run-summary">
            Done. {result.rows.length} rows returned.
          </p>
          {result.discardedRowCount > 0 && (
            <p className="tutorial-run-discarded" role="status">
              {result.discardedRowCount}{" "}
              {result.discardedRowCount === 1 ? "row was" : "rows were"} discarded at the source
              because the data could not be parsed. They were recorded in the audit trail but are
              not shown in the results table.
            </p>
          )}
          <TutorialResultTable rows={result.rows} />
          <div className="tutorial-actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onCompleted(result)}
            >
              {TURN_4_PRIMARY_BUTTON}
            </button>
            {onBack !== undefined && (
              <button
                type="button"
                className="tutorial-link-button"
                onClick={onBack}
                aria-label="Back to the pipeline build"
              >
                Back
              </button>
            )}
          </div>
        </>
      )}
    </section>
  );
}

function describePhase(phase: RunPhase): string {
  switch (phase) {
    case "fetch":
      return "Fetching pages…";
    case "model":
      return "Calling the model…";
    case "write":
      return "Writing the output…";
  }
}

function TutorialResultTable({ rows }: { rows: RunResultRow[] }): JSX.Element {
  const columns = preferredColumns(rows);
  return (
    <div className="tutorial-result-table-wrap">
      <table className="tutorial-result-table">
        <caption className="sr-only">Pipeline run results</caption>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{titleCase(column)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>
                  {stringifyCell(row[column])}
                  {column === "error" && row[column] ? (
                    <span className="tutorial-cell-note">Recorded in audit</span>
                  ) : null}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function preferredColumns(rows: RunResultRow[]): string[] {
  const keys = new Set<string>();
  rows.forEach((row) => Object.keys(row).forEach((key) => keys.add(key)));
  const preferred = ["url", "summary", "error"];
  const ordered = preferred.filter((key) => keys.has(key));
  for (const key of keys) {
    if (!ordered.includes(key)) {
      ordered.push(key);
    }
  }
  return ordered.slice(0, 6);
}

function stringifyCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

function titleCase(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatError(err: unknown): string {
  // run_already_active is an internal one-active-run-per-session invariant
  // (RunAlreadyActiveError, sessions/protocol.py) tripping on a too-quick
  // retry — its raw detail ("run already active for session <uuid>") is
  // implementation detail, not user-facing copy. Fixed friendly language
  // instead of widening what the client echoes.
  if (hasErrorType(err, "run_already_active")) {
    return TUTORIAL_RUN_STILL_FINISHING_MESSAGE;
  }
  if (
    typeof err === "object" &&
    err !== null &&
    "detail" in err &&
    typeof (err as { detail?: unknown }).detail === "string"
  ) {
    return (err as { detail: string }).detail;
  }
  if (err instanceof Error) {
    return err.message;
  }
  return "The tutorial run failed.";
}
