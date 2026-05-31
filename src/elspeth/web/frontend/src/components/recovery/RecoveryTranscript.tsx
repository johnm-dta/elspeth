import { useEffect, useMemo, useState } from "react";
import { fetchRecoveryTranscript } from "@/api/client";
import type { FailedTurn, RecoveryTranscriptRow } from "@/types/recovery";
import type { ToolCall } from "@/types/api";

interface RecoveryTranscriptProps {
  sessionId: string;
  failedTurn: FailedTurn;
}

interface TranscriptState {
  status: "idle" | "loading" | "loaded" | "error";
  rows: RecoveryTranscriptRow[];
}

function findToolRows(
  rows: RecoveryTranscriptRow[],
  assistantId: string,
  toolCallId: string,
): RecoveryTranscriptRow[] {
  return rows.filter(
    (row) =>
      row.role === "tool" &&
      row.parent_assistant_id === assistantId &&
      row.tool_call_id === toolCallId,
  );
}

function toolName(toolCall: ToolCall): string {
  return toolCall.function.name;
}

function renderToolContent(row: RecoveryTranscriptRow): string {
  return row.content;
}

function ToolCallTranscript({
  assistantId,
  rows,
  toolCall,
}: {
  assistantId: string;
  rows: RecoveryTranscriptRow[];
  toolCall: ToolCall;
}) {
  const toolRows = findToolRows(rows, assistantId, toolCall.id);
  return (
    <li className="recovery-transcript-tool-call">
      <div className="recovery-transcript-tool-title">
        <span>{toolName(toolCall)}</span>
        <code>{toolCall.id}</code>
      </div>
      {toolRows.length === 0 ? (
        <p className="recovery-transcript-missing">Missing tool response</p>
      ) : (
        <ul className="recovery-transcript-tool-rows">
          {toolRows.map((row) => (
            <li key={row.id}>
              <pre>{renderToolContent(row)}</pre>
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

export function RecoveryTranscript({
  sessionId,
  failedTurn,
}: RecoveryTranscriptProps) {
  const assistantId = failedTurn.assistant_message_id;
  const [state, setState] = useState<TranscriptState>({
    status: "idle",
    rows: [],
  });

  useEffect(() => {
    if (assistantId === null) {
      setState({ status: "idle", rows: [] });
      return;
    }

    let cancelled = false;
    setState({ status: "loading", rows: [] });
    fetchRecoveryTranscript(sessionId, { limit: 500 })
      .then((rows) => {
        if (!cancelled) {
          setState({ status: "loaded", rows });
        }
      })
      .catch(() => {
        if (!cancelled) {
          setState({ status: "error", rows: [] });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [assistantId, sessionId]);

  const assistantRow = useMemo(() => {
    if (assistantId === null) {
      return null;
    }
    return (
      state.rows.find(
        (row) => row.role === "assistant" && row.id === assistantId,
      ) ?? null
    );
  }, [assistantId, state.rows]);

  if (assistantId === null) {
    return (
      <section
        className="recovery-transcript"
        aria-labelledby="recovery-transcript-title"
      >
        <h3 id="recovery-transcript-title">Failed turn transcript</h3>
        <p>No failed assistant row was recorded for this turn.</p>
        <p>
          ELSPETH saved the partial pipeline state, but the exact assistant
          turn cannot be reconstructed from the transcript index.
        </p>
      </section>
    );
  }

  if (state.status === "loading" || state.status === "idle") {
    return (
      <section
        className="recovery-transcript"
        aria-labelledby="recovery-transcript-title"
      >
        <h3 id="recovery-transcript-title">Failed turn transcript</h3>
        <p>Loading recovery transcript...</p>
      </section>
    );
  }

  if (state.status === "error") {
    return (
      <section
        className="recovery-transcript"
        aria-labelledby="recovery-transcript-title"
      >
        <h3 id="recovery-transcript-title">Failed turn transcript</h3>
        <p>Failed to load the recovery transcript.</p>
      </section>
    );
  }

  if (assistantRow === null) {
    return (
      <section
        className="recovery-transcript"
        aria-labelledby="recovery-transcript-title"
      >
        <h3 id="recovery-transcript-title">Failed turn transcript</h3>
        <p>
          The failed assistant row is not present in the first 500 transcript
          rows. Reopen the session or retry after the transcript index catches
          up.
        </p>
      </section>
    );
  }

  const toolCalls = assistantRow.tool_calls ?? [];
  return (
    <section
      className="recovery-transcript"
      aria-labelledby="recovery-transcript-title"
    >
      <h3 id="recovery-transcript-title">Failed turn transcript</h3>
      <article className="recovery-transcript-assistant">
        <h4>Assistant turn</h4>
        <p>{assistantRow.content}</p>
      </article>
      <div className="recovery-transcript-tools">
        <h4>Tool calls</h4>
        {toolCalls.length === 0 ? (
          <p>No tool calls were recorded for this assistant turn.</p>
        ) : (
          <ul>
            {toolCalls.map((toolCall) => (
              <ToolCallTranscript
                assistantId={assistantRow.id}
                key={toolCall.id}
                rows={state.rows}
                toolCall={toolCall}
              />
            ))}
          </ul>
        )}
      </div>
      {failedTurn.transcript_url === null ? (
        <p className="recovery-transcript-note">
          Transcript URL metadata is not available for this failed turn.
        </p>
      ) : null}
    </section>
  );
}
