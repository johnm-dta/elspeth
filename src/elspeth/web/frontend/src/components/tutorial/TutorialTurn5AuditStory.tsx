import { useEffect, useState } from "react";
import { getRunAuditSummary } from "@/api/client";
import type { RunAuditStoryResponse } from "@/types/api";
import { TURN_5_PRIMARY_BUTTON } from "./copy";

interface TutorialTurn5AuditStoryProps {
  sessionId: string;
  runId: string;
  sourceDataHash: string;
  onContinue: () => void;
}

export function TutorialTurn5AuditStory({
  sessionId,
  runId,
  sourceDataHash,
  onContinue,
}: TutorialTurn5AuditStoryProps): JSX.Element {
  const [summary, setSummary] = useState<RunAuditStoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getRunAuditSummary(sessionId, runId)
      .then((response) => {
        if (!cancelled) setSummary(response);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(formatError(err));
      });
    return () => {
      cancelled = true;
    };
  }, [runId, sessionId]);

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-audit-title">
      <p className="tutorial-kicker">Audit</p>
      <h2 id="tutorial-audit-title">This is the audit story.</h2>
      <p>
        The LLM made a judgment call on each page. ELSPETH keeps the evidence
        needed to explain that judgment later.
      </p>
      {summary === null && error === null && (
        <p role="status" className="tutorial-muted">
          Loading audit evidence...
        </p>
      )}
      {error !== null && (
        <p role="alert" className="tutorial-error">
          {error}
        </p>
      )}
      {summary !== null && (
        <>
          <dl className="tutorial-audit-list">
            <div>
              <dt>Source data hash</dt>
              <dd>{shortHash(sourceDataHash)}</dd>
            </div>
            <div>
              <dt>Output file hash</dt>
              <dd>{shortHash(summary.output_file_hash)}</dd>
            </div>
            <div>
              <dt>LLM calls</dt>
              <dd>{summary.llm_call_count}</dd>
            </div>
            <div>
              <dt>Run</dt>
              <dd>{summary.run_id}</dd>
            </div>
            <div>
              <dt>Started</dt>
              <dd>{new Date(summary.started_at).toLocaleString()}</dd>
            </div>
            <div>
              <dt>Plugin versions</dt>
              <dd>{formatPluginVersions(summary.plugin_versions)}</dd>
            </div>
          </dl>
          <p>
            If someone asks why a page received its score, the run has the
            prompt, response, model details, input hash, output hash, and plugin
            versions tied together.
          </p>
          <div className="tutorial-actions">
            <button type="button" className="btn btn-primary" onClick={onContinue}>
              {TURN_5_PRIMARY_BUTTON}
            </button>
          </div>
        </>
      )}
    </section>
  );
}

function shortHash(hash: string): string {
  return hash.length > 12 ? `${hash.slice(0, 12)}...` : hash;
}

function formatPluginVersions(versions: Record<string, string>): string {
  const entries = Object.entries(versions);
  if (entries.length === 0) {
    return "No plugin versions recorded";
  }
  return entries.map(([name, version]) => `${name} ${version}`).join(", ");
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
  return "The audit story could not be loaded.";
}
