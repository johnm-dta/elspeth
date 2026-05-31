import { useCallback, useEffect, useRef, useState } from "react";
import { getRunAuditSummary } from "@/api/client";
import type { RunAuditStoryResponse } from "@/types/api";
import { TURN_5_PRIMARY_BUTTON } from "./copy";

const COPY_FEEDBACK_DURATION_MS = 2_000;

interface TutorialTurn5AuditStoryProps {
  sessionId: string;
  runId: string;
  sourceDataHash: string;
  onContinue: () => void;
  onBack: () => void;
}

export function TutorialTurn5AuditStory({
  sessionId,
  runId,
  sourceDataHash,
  onContinue,
  onBack,
}: TutorialTurn5AuditStoryProps): JSX.Element {
  const [summary, setSummary] = useState<RunAuditStoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const headingRef = useRef<HTMLHeadingElement | null>(null);

  // Focus the turn's heading on mount (Group D: focus management).
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

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
      <h2 id="tutorial-audit-title" ref={headingRef} tabIndex={-1}>
        This is the audit story.
      </h2>
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
              <dd>
                <HashWithCopy
                  hash={sourceDataHash}
                  label="source data hash"
                />
              </dd>
            </div>
            <div>
              <dt>Output file hash</dt>
              <dd>
                <HashWithCopy
                  hash={summary.output_file_hash}
                  label="output file hash"
                />
              </dd>
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
            <button
              type="button"
              className="tutorial-link-button"
              onClick={onBack}
              aria-label="Back: edit prompt and start over"
            >
              Back
            </button>
          </div>
        </>
      )}
    </section>
  );
}

/**
 * Renders a full hash in <code> with a Copy button next to it. The full
 * hash is shown (not truncated) because the audit-story turn is the
 * screen that has to prove auditability — an auditor cannot verify a
 * 12-character prefix. `overflow-wrap: anywhere` on `.tutorial-audit-list
 * dd` lets the long string wrap inside the existing card.
 */
function HashWithCopy({
  hash,
  label,
}: {
  hash: string;
  label: string;
}): JSX.Element {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | null>(null);

  // Clear any pending feedback timer on unmount so we don't update state
  // after the user has navigated away.
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, []);

  const onCopy = useCallback(async () => {
    // navigator.clipboard requires a secure context. Fall back to the
    // textarea/execCommand pattern when unavailable (older browsers or
    // http:// dev contexts) so the affordance still works.
    try {
      if (navigator.clipboard !== undefined) {
        await navigator.clipboard.writeText(hash);
      } else {
        legacyClipboardCopy(hash);
      }
      setCopied(true);
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
      timerRef.current = window.setTimeout(() => {
        setCopied(false);
        timerRef.current = null;
      }, COPY_FEEDBACK_DURATION_MS);
    } catch {
      // Surface the failure without crashing the audit-story render.
      // The hash itself remains visible in the <code> beside the button.
      setCopied(false);
    }
  }, [hash]);

  return (
    <div className="tutorial-hash">
      <code className="tutorial-hash-value">{hash}</code>
      <button
        type="button"
        className="tutorial-hash-copy"
        onClick={() => void onCopy()}
        aria-label={`Copy full ${label}`}
      >
        {copied ? "Copied" : "Copy"}
      </button>
      <span role="status" className="sr-only">
        {copied ? `Full ${label} copied to clipboard.` : ""}
      </span>
    </div>
  );
}

function legacyClipboardCopy(text: string): void {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
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
