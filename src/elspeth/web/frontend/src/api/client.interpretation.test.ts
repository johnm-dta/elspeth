/**
 * Tests for the Phase 5b interpretation-event client surface.
 *
 * Strategy mirrors client.guided.test.ts: spy on globalThis.fetch (NOT
 * vi.mock("./client")) so the real `parseResponse<T>()` + `authHeaders()`
 * code paths execute.  Consumer tests (stores/components) mock the
 * client module; producer tests exercise it.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  listInterpretationEvents,
  resolveInterpretation,
  optOutOfInterpretations,
  getInterpretationOptOutSummary,
} from "./client";
import type {
  InterpretationEvent,
  InterpretationOptOutResponse,
  InterpretationResolveResponse,
  ListInterpretationEventsResponse,
  OptOutSummaryResponse,
} from "@/types/interpretation";
import type { CompositionState } from "@/types/api";
import { compositionStateAuthorityFields } from "@/test/composerFixtures";

// ── Fixtures ────────────────────────────────────────────────────────────────

/** A pending user_approved event — surface fields populated, accepted_value null. */
function makePendingEvent(overrides: Partial<InterpretationEvent> = {}): InterpretationEvent {
  return {
    id: "evt-1",
    session_id: "sess-1",
    composition_state_id: "state-1",
    affected_node_id: "node-1",
    tool_call_id: "tool-1",
    user_term: "cool",
    kind: "vague_term",
    llm_draft: "interesting and engaging",
    accepted_value: null,
    choice: "pending",
    created_at: "2026-05-18T00:00:00Z",
    resolved_at: null,
    actor: "user:owner:u-1",
    interpretation_source: "user_approved",
    model_identifier: "anthropic/claude-opus-4-7",
    model_version: "20260518",
    provider: "anthropic",
    composer_skill_hash: "deadbeef",
    arguments_hash: null,
    hash_domain_version: null,
    runtime_model_identifier_at_resolve: null,
    runtime_model_version_at_resolve: null,
    resolved_prompt_template_hash: null,
    ...overrides,
  };
}

/** Minimal composition state for the resolve response envelope. */
function makeCompositionState(): CompositionState {
  return {
    id: "state-2",
    ...compositionStateAuthorityFields,
    version: 2,
    sources: {},
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

// ── Suites ───────────────────────────────────────────────────────────────────

describe("api/client interpretation functions", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  describe("listInterpretationEvents", () => {
    it("calls GET /api/sessions/:id/interpretations with status=all by default", async () => {
      const body: ListInterpretationEventsResponse = {
        events: [makePendingEvent()],
      };
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => body,
      } as Response);

      const result = await listInterpretationEvents("sess-1");

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/interpretations?status=all");
      expect(init.method).toBe("GET");
      // Unwraps the envelope and returns the events array directly.
      expect(result).toEqual(body.events);
    });

    it("calls GET /api/sessions/:id/interpretations with status=pending when requested", async () => {
      const body: ListInterpretationEventsResponse = { events: [] };
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => body,
      } as Response);

      await listInterpretationEvents("sess-1", "pending");

      const [url] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/interpretations?status=pending");
    });

    it("propagates AbortSignal to fetch", async () => {
      const controller = new AbortController();
      fetchSpy.mockRejectedValue(new DOMException("Aborted", "AbortError"));

      await expect(
        listInterpretationEvents("sess-abort", "all", controller.signal),
      ).rejects.toThrow("Aborted");

      const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(init.signal).toBe(controller.signal);
    });

    it("throws ApiError (does not swallow) when server returns 404", async () => {
      fetchSpy.mockResolvedValue({
        ok: false,
        status: 404,
        statusText: "Not Found",
        json: async () => ({ detail: "session not found" }),
      } as Response);

      await expect(listInterpretationEvents("sess-missing")).rejects.toMatchObject({
        status: 404,
        detail: "session not found",
      });
    });
  });

  describe("resolveInterpretation", () => {
    it("calls POST /api/sessions/:id/interpretations/:event_id/resolve with the body", async () => {
      const resolved = makePendingEvent({
        choice: "accepted_as_drafted",
        accepted_value: "interesting and engaging",
        resolved_at: "2026-05-18T00:01:00Z",
        arguments_hash: "abc123",
        hash_domain_version: "v1",
        resolved_prompt_template_hash: "fedcba",
      });
      const body: InterpretationResolveResponse = {
        event: resolved,
        new_state: makeCompositionState(),
      };
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => body,
      } as Response);

      const result = await resolveInterpretation("sess-1", "evt-1", {
        choice: "accepted_as_drafted",
      });

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/interpretations/evt-1/resolve");
      expect(init.method).toBe("POST");
      expect((init.headers as Record<string, string>)["Content-Type"]).toBe(
        "application/json",
      );
      expect(JSON.parse(init.body as string)).toEqual({
        choice: "accepted_as_drafted",
      });
      expect(result).toEqual(body);
    });

    it("forwards amended_value when choice='amended'", async () => {
      const resolved = makePendingEvent({
        choice: "amended",
        accepted_value: "thoughtful and witty",
      });
      const body: InterpretationResolveResponse = {
        event: resolved,
        new_state: makeCompositionState(),
      };
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => body,
      } as Response);

      await resolveInterpretation("sess-1", "evt-1", {
        choice: "amended",
        amended_value: "thoughtful and witty",
      });

      const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(JSON.parse(init.body as string)).toEqual({
        choice: "amended",
        amended_value: "thoughtful and witty",
      });
    });

    it("surfaces 422 validation errors as typed ApiError", async () => {
      // The backend rejects amended without amended_value with a 422 from
      // the pydantic model_validator.  parseResponse converts that to
      // ApiError { status: 422, detail: ... }.
      fetchSpy.mockResolvedValue({
        ok: false,
        status: 422,
        statusText: "Unprocessable Entity",
        json: async () => ({
          detail: "amended_value is required when choice == 'amended'",
        }),
      } as Response);

      await expect(
        resolveInterpretation("sess-1", "evt-1", { choice: "amended" }),
      ).rejects.toMatchObject({
        status: 422,
        detail: "amended_value is required when choice == 'amended'",
      });
    });

    it("maps nested detail.code onto ApiError.error_type", async () => {
      fetchSpy.mockResolvedValue({
        ok: false,
        status: 422,
        statusText: "Unprocessable Entity",
        json: async () => ({
          detail: {
            code: "interpretation_placeholder_unavailable",
            detail: "The affected LLM prompt no longer contains the expected interpretation placeholder.",
          },
        }),
      } as Response);

      await expect(
        resolveInterpretation("sess-1", "evt-1", {
          choice: "accepted_as_drafted",
        }),
      ).rejects.toMatchObject({
        status: 422,
        error_type: "interpretation_placeholder_unavailable",
        detail:
          "The affected LLM prompt no longer contains the expected interpretation placeholder.",
      });
    });

    it("surfaces 409 conflict (already-resolved event) as typed ApiError", async () => {
      fetchSpy.mockResolvedValue({
        ok: false,
        status: 409,
        statusText: "Conflict",
        json: async () => ({
          detail: "interpretation event already resolved",
          error_type: "interpretation_already_resolved",
        }),
      } as Response);

      await expect(
        resolveInterpretation("sess-1", "evt-1", {
          choice: "accepted_as_drafted",
        }),
      ).rejects.toMatchObject({
        status: 409,
        error_type: "interpretation_already_resolved",
      });
    });
  });

  describe("optOutOfInterpretations", () => {
    it("calls POST /api/sessions/:id/interpretations/opt_out with an empty JSON body", async () => {
      const body: InterpretationOptOutResponse = {
        session_id: "sess-1",
        interpretation_review_disabled: true,
        opted_out_at: "2026-05-18T00:05:00Z",
      };
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => body,
      } as Response);

      const result = await optOutOfInterpretations("sess-1");

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/interpretations/opt_out");
      expect(init.method).toBe("POST");
      expect((init.headers as Record<string, string>)["Content-Type"]).toBe(
        "application/json",
      );
      // Body is an empty JSON object; the route accepts no fields beyond
      // the path's session_id, but Content-Type: application/json wants a
      // matching payload.
      expect(init.body).toBe("{}");
      expect(result).toEqual(body);
    });

    it("surfaces 404 (session not found) as typed ApiError", async () => {
      fetchSpy.mockResolvedValue({
        ok: false,
        status: 404,
        statusText: "Not Found",
        json: async () => ({ detail: "session not found" }),
      } as Response);

      await expect(optOutOfInterpretations("sess-missing")).rejects.toMatchObject({
        status: 404,
      });
    });
  });

  describe("getInterpretationOptOutSummary", () => {
    it("calls GET /api/sessions/:id/interpretations/opt_out_summary", async () => {
      const optOutEvent = makePendingEvent({
        id: "evt-opt-1",
        composition_state_id: null,
        affected_node_id: null,
        tool_call_id: null,
        user_term: null,
        kind: null,
        llm_draft: null,
        choice: "opted_out",
        resolved_at: "2026-05-18T00:10:00Z",
        interpretation_source: "auto_interpreted_opt_out",
        model_identifier: null,
        model_version: null,
        provider: null,
        composer_skill_hash: null,
      });
      const body: OptOutSummaryResponse = { events: [optOutEvent] };
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => body,
      } as Response);

      const result = await getInterpretationOptOutSummary("sess-1");

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/interpretations/opt_out_summary");
      expect(init.method).toBe("GET");
      // Unwraps the envelope.
      expect(result).toEqual([optOutEvent]);
    });

    it("propagates AbortSignal to fetch", async () => {
      const controller = new AbortController();
      fetchSpy.mockRejectedValue(new DOMException("Aborted", "AbortError"));

      await expect(
        getInterpretationOptOutSummary("sess-abort", controller.signal),
      ).rejects.toThrow("Aborted");

      const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(init.signal).toBe(controller.signal);
    });
  });
});
