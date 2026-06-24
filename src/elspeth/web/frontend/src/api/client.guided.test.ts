/**
 * Tests for src/api/client.ts (producer side).
 *
 * Strategy: vi.spyOn(globalThis, "fetch") — NOT vi.mock("./client").
 *
 * Consumer tests (e.g. src/stores/sessionStore.test.ts) mock the
 * @/api/client module because they're testing wiring around it.  This
 * file is the producer; mocking the module we import from would mock
 * the unit under test.  We spy on the underlying fetch instead, which
 * lets us exercise the real getGuided/respondGuided code paths
 * including authHeaders() and parseResponse<T>().
 *
 * Future API-client tests should follow this same pattern.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { getGuided, respondGuided, startGuidedSession } from "./client";
import type {
  GetGuidedResponse,
  GuidedRespondRequest,
  GuidedRespondResponse,
} from "@/types/guided";

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Minimal but well-formed GetGuidedResponse fixture. */
function makeGetGuidedResponse(): GetGuidedResponse {
  return {
    guided_session: {
      step: "step_1_source",
      history: [],
      terminal: null,
      chat_history: [],
      chat_turn_seq: 0,
      profile: null,
    },
    next_turn: null,
    terminal: null,
    composition_state: null,
  };
}

/** Minimal GuidedRespondRequest — all optional fields null. */
function makeRespondRequest(): GuidedRespondRequest {
  return {
    chosen: ["foo"],
    edited_values: null,
    custom_inputs: null,
    accepted_step_index: null,
    edit_step_index: null,
    control_signal: null,
  };
}

/** Minimal GuidedRespondResponse — same shape as GetGuidedResponse. */
function makeRespondResponse(): GuidedRespondResponse {
  return {
    guided_session: {
      step: "step_2_sink",
      history: [
        {
          step: "step_1_source",
          turn_type: "single_select",
          payload_hash: "abc123",
          response_hash: "def456",
          summary: "Source selected: csv",
          emitter: "server",
        },
      ],
      terminal: null,
      chat_history: [],
      chat_turn_seq: 0,
      profile: null,
    },
    next_turn: {
      type: "single_select",
      step_index: 1,
      payload: { options: ["csv", "json"] },
    },
    terminal: null,
    composition_state: null,
  };
}

// ── Suites ───────────────────────────────────────────────────────────────────

describe("api/client guided functions", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  describe("getGuided", () => {
    it("calls GET /api/sessions/:id/guided and returns parsed body", async () => {
      const body = makeGetGuidedResponse();
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => body,
      } as Response);

      const result = await getGuided("sess-1");

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/guided");
      expect(init.method).toBe("GET");
      expect(
        (init.headers as Record<string, string>)["Content-Type"],
      ).toBeUndefined();
      expect(result).toEqual(body);
    });

    it("propagates AbortSignal to fetch", async () => {
      const controller = new AbortController();
      // Mock fetch to throw an AbortError when called — this simulates the
      // native behaviour when the signal fires.  We don't actually abort here;
      // we just confirm the signal was forwarded by inspecting the call args.
      const abortError = new DOMException("Aborted", "AbortError");
      fetchSpy.mockRejectedValue(abortError);

      await expect(getGuided("sess-abort", controller.signal)).rejects.toThrow(
        "Aborted",
      );

      const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(init.signal).toBe(controller.signal);
    });
  });

  describe("respondGuided", () => {
    it("calls POST /api/sessions/:id/guided/respond and returns parsed body", async () => {
      const body = makeRespondResponse();
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => body,
      } as Response);

      const request = makeRespondRequest();
      const result = await respondGuided("sess-1", request);

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/guided/respond");
      expect(init.method).toBe("POST");
      expect((init.headers as Record<string, string>)["Content-Type"]).toBe(
        "application/json",
      );
      // Verify the parsed shape rather than the raw byte sequence: JSON key
      // ordering is incidental, and the server consumes the parsed object.
      const bodyStr = init.body as string;
      expect(JSON.parse(bodyStr)).toEqual(request);
      expect(result).toEqual(body);
    });

    it("propagates AbortSignal to fetch", async () => {
      const controller = new AbortController();
      const abortError = new DOMException("Aborted", "AbortError");
      fetchSpy.mockRejectedValue(abortError);

      const request = makeRespondRequest();
      await expect(
        respondGuided("sess-abort", request, controller.signal),
      ).rejects.toThrow("Aborted");

      const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(init.signal).toBe(controller.signal);
    });

    it("throws ApiError (does not swallow) when server returns 500", async () => {
      // parseResponse throws an ApiError-shaped object for non-2xx responses.
      // Use 500 to avoid the 401 interceptor which dynamically imports authStore.
      fetchSpy.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: async () => ({ detail: "boom" }),
      } as Response);

      const request = makeRespondRequest();
      await expect(respondGuided("sess-err", request)).rejects.toMatchObject({
        status: 500,
        detail: "boom",
      });
    });
  });

  describe("startGuidedSession", () => {
    it("POSTs the profile discriminator to the full guided-start route", async () => {
      const body = makeGetGuidedResponse();
      body.guided_session.profile = {
        coaching: true,
        bookends: true,
        recipe_match: true,
        advisor_checkpoints: true,
      };
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => body,
      } as Response);

      const result = await startGuidedSession("sess-1", "tutorial");

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/guided/start");
      expect(init.method).toBe("POST");
      expect((init.headers as Record<string, string>)["Content-Type"]).toBe(
        "application/json",
      );
      expect(JSON.parse(init.body as string)).toEqual({ profile: "tutorial" });
      expect(result.guided_session.profile?.advisor_checkpoints).toBe(true);
    });
  });
});
