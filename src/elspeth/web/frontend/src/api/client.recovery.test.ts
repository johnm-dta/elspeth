/**
 * Recovery-specific API-client contract tests.
 *
 * These exercise the real client functions over a fetch spy so parseResponse
 * and auth/query construction stay covered together.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchRecoveryTranscript, sendMessage } from "./client";
import type { ApiError, CompositionState } from "@/types/api";

function makePartialState(): CompositionState {
  return {
    id: "state-1",
    version: 7,
    sources: {},
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: "Recovered", description: null },
  };
}

describe("api/client recovery contracts", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("preserves top-level composer recovery fields on ApiError", async () => {
    const partialState = makePartialState();
    fetchSpy.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => ({
        detail: "compose failed",
        error_type: "composer_plugin_crash",
        partial_state: partialState,
        failed_turn: {
          assistant_message_id: "assistant-1",
          tool_calls_attempted: 3,
          tool_responses_persisted: 2,
          transcript_url: "/future/transcript",
        },
        partial_state_save_failed: false,
        partial_state_save_error: null,
      }),
    } as Response);

    let error: ApiError | undefined;
    try {
      await sendMessage("session-1", "recover please");
    } catch (err) {
      error = err as ApiError;
    }

    expect(error).toMatchObject({
      status: 500,
      detail: "compose failed",
      error_type: "composer_plugin_crash",
      partial_state: partialState,
      failed_turn: {
        assistant_message_id: "assistant-1",
        tool_calls_attempted: 3,
        tool_responses_persisted: 2,
        transcript_url: "/future/transcript",
      },
      partial_state_save_failed: false,
      partial_state_save_error: null,
    });
  });

  it("preserves nested FastAPI recovery fields including null transcript_url", async () => {
    const partialState = makePartialState();
    fetchSpy.mockResolvedValue({
      ok: false,
      status: 422,
      statusText: "Unprocessable Entity",
      json: async () => ({
        detail: {
          detail: "compose failed after tool call",
          error_type: "composer_convergence",
          partial_state: partialState,
          failed_turn: {
            assistant_message_id: "assistant-2",
            tool_calls_attempted: 1,
            tool_responses_persisted: 1,
            transcript_url: null,
          },
          partial_state_save_failed: true,
          partial_state_save_error: "audit unavailable",
        },
      }),
    } as Response);

    let error: ApiError | undefined;
    try {
      await sendMessage("session-2", "recover please");
    } catch (err) {
      error = err as ApiError;
    }

    expect(error).toMatchObject({
      status: 422,
      detail: "compose failed after tool call",
      error_type: "composer_convergence",
      partial_state: partialState,
      failed_turn: {
        assistant_message_id: "assistant-2",
        tool_calls_attempted: 1,
        tool_responses_persisted: 1,
        transcript_url: null,
      },
      partial_state_save_failed: true,
      partial_state_save_error: "audit unavailable",
    });
  });

  it("keeps existing provider validation and fanout error parsing", async () => {
    const validationErrors = [
      { component: "source", message: "missing path", severity: "error" },
    ];
    const fanoutGuard = {
      token: "fanout-token",
      provider: "openrouter",
      model: "model-1",
      row_count: 10,
      estimated_provider_calls: null,
      provider_calls_per_row: 2,
      upstream_fanout: ["transform:explode"],
    };
    fetchSpy.mockResolvedValue({
      ok: false,
      status: 428,
      statusText: "Precondition Required",
      json: async () => ({
        detail: {
          detail: "fanout acknowledgement required",
          error_type: "execution_fanout_ack_required",
          provider_detail: "provider says no",
          provider_status_code: 429,
          validation_errors: validationErrors,
          fanout_guard: fanoutGuard,
        },
      }),
    } as Response);

    let error: ApiError | undefined;
    try {
      await sendMessage("session-3", "not recovery");
    } catch (err) {
      error = err as ApiError;
    }

    expect(error).toMatchObject({
      status: 428,
      detail: "fanout acknowledgement required",
      error_type: "execution_fanout_ack_required",
      provider_detail: "provider says no",
      provider_status_code: 429,
      validation_errors: validationErrors,
      fanout_guard: fanoutGuard,
    });
    expect(error?.partial_state).toBeUndefined();
    expect(error?.failed_turn).toBeUndefined();
  });

  it("fetches recovery transcript from active session with tool rows and no raw content", async () => {
    const rows = [
      {
        id: "assistant-1",
        session_id: "session-4",
        role: "assistant",
        content: "working",
        raw_content: null,
        tool_calls: [],
        created_at: "2026-05-14T00:00:00Z",
        composition_state_id: null,
        tool_call_id: null,
        parent_assistant_id: null,
        sequence_no: 1,
      },
    ];
    fetchSpy.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => rows,
    } as Response);

    const result = await fetchRecoveryTranscript("session-4");

    expect(result).toEqual(rows);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(
      "/api/sessions/session-4/messages?include_tool_rows=true&limit=500&offset=0",
    );
    expect(url).not.toContain("since=");
    expect(url).not.toContain("include_raw_content");
  });
});
