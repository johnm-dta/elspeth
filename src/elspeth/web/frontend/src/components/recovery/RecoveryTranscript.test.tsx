import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RecoveryTranscript } from "./RecoveryTranscript";
import type { FailedTurn, RecoveryTranscriptRow } from "@/types/recovery";

function makeFailedTurn(
  overrides: Partial<FailedTurn> = {},
): FailedTurn {
  return {
    assistant_message_id: "assistant-failed",
    tool_calls_attempted: 2,
    tool_responses_persisted: 1,
    transcript_url: null,
    ...overrides,
  };
}

function makeAssistantRow(
  overrides: Partial<RecoveryTranscriptRow> = {},
): RecoveryTranscriptRow {
  return {
    id: "assistant-failed",
    session_id: "session-1",
    role: "assistant",
    content: "I will inspect the source and then update the pipeline.",
    raw_content: "provider hidden chain material",
    tool_calls: [
      {
        id: "call-1",
        type: "function",
        function: { name: "inspect_source", arguments: "{}" },
      },
      {
        id: "call-2",
        type: "function",
        function: { name: "update_pipeline", arguments: "{}" },
      },
    ],
    created_at: "2026-05-14T00:00:00Z",
    composition_state_id: null,
    tool_call_id: null,
    parent_assistant_id: null,
    sequence_no: 10,
    ...overrides,
  };
}

function makeToolRow(
  overrides: Partial<RecoveryTranscriptRow> = {},
): RecoveryTranscriptRow {
  return {
    id: "tool-1",
    session_id: "session-1",
    role: "tool",
    content: '{"status":"ok","redacted":true}',
    raw_content: '{"provider_payload":"do not render"}',
    tool_calls: null,
    created_at: "2026-05-14T00:00:01Z",
    composition_state_id: "state-2",
    tool_call_id: "call-1",
    parent_assistant_id: "assistant-failed",
    sequence_no: 11,
    ...overrides,
  };
}

function mockTranscript(rows: RecoveryTranscriptRow[]): ReturnType<typeof vi.spyOn> {
  const fetchSpy = vi.spyOn(globalThis, "fetch");
  fetchSpy.mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => rows,
  } as Response);
  return fetchSpy;
}

describe("RecoveryTranscript", () => {
  it("fetches transcript rows once per open using tool rows", async () => {
    const fetchSpy = mockTranscript([
      makeAssistantRow(),
      makeToolRow(),
    ]);
    const { rerender } = render(
      <RecoveryTranscript
        sessionId="session-1"
        failedTurn={makeFailedTurn()}
      />,
    );

    expect(screen.getByText("Loading recovery transcript...")).toBeInTheDocument();
    await screen.findByText("inspect_source");
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy.mock.calls[0][0]).toBe(
      "/api/sessions/session-1/messages?include_tool_rows=true&limit=500&offset=0",
    );

    rerender(
      <RecoveryTranscript
        sessionId="session-1"
        failedTurn={makeFailedTurn()}
      />,
    );
    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(1));
    fetchSpy.mockRestore();
  });

  it("filters to the failed assistant row and its matching tool rows", async () => {
    const fetchSpy = mockTranscript([
      makeAssistantRow({ id: "assistant-other", content: "Do not show" }),
      makeToolRow({
        id: "tool-other",
        parent_assistant_id: "assistant-other",
        content: "wrong tool",
      }),
      makeAssistantRow(),
      makeToolRow({ content: '{"preview":"safe redacted result"}' }),
    ]);

    render(
      <RecoveryTranscript
        sessionId="session-1"
        failedTurn={makeFailedTurn()}
      />,
    );

    expect(await screen.findByText("inspect_source")).toBeInTheDocument();
    expect(screen.getByText("update_pipeline")).toBeInTheDocument();
    expect(screen.getByText(/safe redacted result/)).toBeInTheDocument();
    expect(screen.getByText("Missing tool response")).toBeInTheDocument();
    expect(screen.queryByText("Do not show")).not.toBeInTheDocument();
    expect(screen.queryByText("wrong tool")).not.toBeInTheDocument();
    fetchSpy.mockRestore();
  });

  it("renders redaction marker text but never raw content or provider payloads", async () => {
    const fetchSpy = mockTranscript([
      makeAssistantRow(),
      makeToolRow({
        content: '{"note":"<redacted> secret fields removed"}',
        raw_content: '{"provider_payload":"OPENAI_INTERNAL"}',
      }),
    ]);

    render(
      <RecoveryTranscript
        sessionId="session-1"
        failedTurn={makeFailedTurn()}
      />,
    );

    expect(await screen.findByText(/redacted/i)).toBeInTheDocument();
    expect(screen.queryByText(/provider hidden chain material/)).not.toBeInTheDocument();
    expect(screen.queryByText(/OPENAI_INTERNAL/)).not.toBeInTheDocument();
    expect(screen.queryByText(/provider_payload/)).not.toBeInTheDocument();
    fetchSpy.mockRestore();
  });

  it("shows degraded diagnostics when no assistant id is available", () => {
    render(
      <RecoveryTranscript
        sessionId="session-1"
        failedTurn={makeFailedTurn({ assistant_message_id: null })}
      />,
    );

    expect(
      screen.getByText(/No failed assistant row was recorded for this turn./),
    ).toBeInTheDocument();
  });

  it("shows degraded diagnostics when the failed assistant is outside the first page", async () => {
    const fetchSpy = mockTranscript([makeAssistantRow({ id: "assistant-other" })]);
    render(
      <RecoveryTranscript
        sessionId="session-1"
        failedTurn={makeFailedTurn()}
      />,
    );

    expect(
      await screen.findByText(/not present in the first 500 transcript rows/),
    ).toBeInTheDocument();
    fetchSpy.mockRestore();
  });

  it("shows an error state when transcript loading fails", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockRejectedValue(new Error("network down"));

    render(
      <RecoveryTranscript
        sessionId="session-1"
        failedTurn={makeFailedTurn()}
      />,
    );

    expect(
      await screen.findByText(/Failed to load the recovery transcript./),
    ).toBeInTheDocument();
    fetchSpy.mockRestore();
  });
});
