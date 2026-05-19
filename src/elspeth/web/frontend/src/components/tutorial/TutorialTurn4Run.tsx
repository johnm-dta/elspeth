import { useEffect, useState } from "react";
import { runTutorialPipeline } from "@/api/client";
import type { TutorialRunResponse } from "@/types/api";
import { TURN_4_PRIMARY_BUTTON } from "./copy";
import type { RunResultRow, TutorialRunResult } from "./tutorialMachine";

interface TutorialTurn4RunProps {
  sessionId: string;
  prompt: string;
  onCompleted: (result: TutorialRunResult) => void;
}

export function TutorialTurn4Run({
  sessionId,
  prompt,
  onCompleted,
}: TutorialTurn4RunProps): JSX.Element {
  const [result, setResult] = useState<TutorialRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getTutorialRunPromise(sessionId, prompt)
      .then((response) => {
        if (!active) return;
        setResult({
          runId: response.run_id,
          sourceDataHash: response.output.source_data_hash,
          rows: response.output.rows,
          seededFromCache: response.seeded_from_cache,
          cacheKey: response.cache_key,
        });
      })
      .catch((err: unknown) => {
        if (!active) return;
        setError(formatError(err));
      });
    return () => {
      active = false;
    };
  }, [prompt, sessionId]);

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-run-title">
      <p className="tutorial-kicker">Run</p>
      <h2 id="tutorial-run-title">Running your pipeline.</h2>
      {result === null && error === null && (
        <div role="status" className="tutorial-running">
          <span className="tutorial-progress-bar" />
          <span>Fetching pages, rating them, and writing the output...</span>
        </div>
      )}
      {error !== null && (
        <p role="alert" className="tutorial-error">
          {error}
        </p>
      )}
      {result !== null && (
        <>
          <p className="tutorial-run-summary">
            Done. {result.rows.length} rows returned
            {result.seededFromCache ? " from the tutorial cache" : ""}.
          </p>
          <TutorialResultTable rows={result.rows} />
          <div className="tutorial-actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onCompleted(result)}
            >
              {TURN_4_PRIMARY_BUTTON}
            </button>
          </div>
        </>
      )}
    </section>
  );
}

const tutorialRunPromises = new Map<string, Promise<TutorialRunResponse>>();

function getTutorialRunPromise(
  sessionId: string,
  prompt: string,
): Promise<TutorialRunResponse> {
  const key = JSON.stringify([sessionId, prompt]);
  const existing = tutorialRunPromises.get(key);
  if (existing !== undefined) {
    return existing;
  }
  const promise = runTutorialPipeline({ session_id: sessionId, prompt }).catch(
    (err: unknown) => {
      tutorialRunPromises.delete(key);
      throw err;
    },
  );
  tutorialRunPromises.set(key, promise);
  return promise;
}

function TutorialResultTable({ rows }: { rows: RunResultRow[] }): JSX.Element {
  const columns = preferredColumns(rows);
  return (
    <div className="tutorial-result-table-wrap">
      <table className="tutorial-result-table">
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
  const preferred = ["url", "page", "title", "score", "coolness", "rationale", "error"];
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
