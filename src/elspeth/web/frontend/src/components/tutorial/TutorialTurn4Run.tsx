import { useEffect, useRef, useState } from "react";
import { runTutorialPipeline } from "@/api/client";
import type { TutorialRunResponse } from "@/types/api";
import { TUTORIAL_RUN_PREAMBLE, TUTORIAL_SHIELD_OVERRIDE_CAVEAT, TURN_4_PRIMARY_BUTTON } from "./copy";
import type { RunResultRow, TutorialRunResult } from "./tutorialMachine";

interface TutorialTurn4RunProps {
  sessionId: string;
  prompt: string;
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
 * Cached by `[sessionId, prompt]` so React StrictMode's double-invoke of
 * the run effect coalesces to a single backend call. The cache entry
 * stores both the promise and the AbortController owning the fetch's
 * signal — the user-cancel path aborts via the cached controller and
 * removes the entry so a subsequent re-mount triggers a fresh run.
 */
const tutorialRunCache = new Map<string, CachedRun>();

function getTutorialRun(sessionId: string, prompt: string): CachedRun {
  const key = JSON.stringify([sessionId, prompt]);
  const existing = tutorialRunCache.get(key);
  if (existing !== undefined) {
    return existing;
  }
  const controller = new AbortController();
  const promise = runTutorialPipeline(
    { session_id: sessionId, prompt },
    controller.signal,
  ).catch((err: unknown) => {
    // Drop the cache entry on failure so the user can retry without
    // hitting a stale rejected promise.
    tutorialRunCache.delete(key);
    throw err;
  });
  const entry: CachedRun = { promise, controller };
  tutorialRunCache.set(key, entry);
  return entry;
}

function clearTutorialRunCache(sessionId: string, prompt: string): void {
  const key = JSON.stringify([sessionId, prompt]);
  tutorialRunCache.delete(key);
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

export function TutorialTurn4Run({
  sessionId,
  prompt,
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

    const cached = getTutorialRun(sessionId, prompt);
    cached.promise
      .then((response) => {
        if (!active) return;
        setResult({
          runId: response.run_id,
          sourceDataHash: response.output.source_data_hash,
          rows: response.output.rows,
          seededFromCache: response.seeded_from_cache,
          cacheKey: response.cache_key,
          discardedRowCount: response.output.discarded_row_count,
        });
      })
      .catch((err: unknown) => {
        if (!active) return;
        // AbortError on the cached promise means another consumer (the
        // user-cancel path below) aborted the run. The cancel handler
        // already dispatched onCancelled; nothing more to render here.
        if (isAbortError(err)) return;
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
  }, [prompt, sessionId, retryNonce]);

  const onCancelClick = (): void => {
    const key = JSON.stringify([sessionId, prompt]);
    const cached = tutorialRunCache.get(key);
    if (cached !== undefined) {
      cached.controller.abort();
      tutorialRunCache.delete(key);
    }
    onCancelled();
  };

  const onRetryClick = (): void => {
    clearTutorialRunCache(sessionId, prompt);
    setRetryNonce((n) => n + 1);
  };

  const phaseText = describePhase(phase);

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-run-title">
      <p className="tutorial-kicker">Run</p>
      <h2 id="tutorial-run-title" ref={headingRef} tabIndex={-1}>
        Running your pipeline.
      </h2>
      <p className="tutorial-muted">{TUTORIAL_RUN_PREAMBLE}</p>
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
            Done. {result.rows.length} rows returned
            {result.seededFromCache ? " from the tutorial cache" : ""}.
          </p>
          {result.discardedRowCount > 0 && (
            <p className="tutorial-run-discarded" role="status">
              {result.discardedRowCount}{" "}
              {result.discardedRowCount === 1 ? "row was" : "rows were"} discarded at the source
              because the data could not be parsed. They were recorded in the audit trail but are
              not shown above.
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
                aria-label="Back: edit prompt and start over"
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
