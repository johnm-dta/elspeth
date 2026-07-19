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
import { chatGuided, convertToGuided, forkFromMessage, ForkCommittedResponseError, getGuided, reenterGuided, respondGuided, revertToVersion, startGuidedSession } from "./client";
import type {
  GetGuidedResponse,
  GuidedChatRequest,
  GuidedChatResponse,
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
    operation_id: "00000000-0000-4000-8000-000000000601",
    turn_token: "a".repeat(64),
    chosen: ["foo"],
    edited_values: null,
    custom_inputs: null,
    proposal_id: null,
    draft_hash: null,
    edit_target: null,
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
      turn_token: "b".repeat(64),
      payload: {
        question: "Choose an output",
        options: [
          { id: "csv", label: "CSV", hint: null },
          { id: "json", label: "JSON", hint: null },
        ],
        allow_custom: false,
      },
    },
    terminal: null,
    composition_state: null,
  };
}

function makeProposalResponse(): GetGuidedResponse {
  const sourceId = "00000000-0000-4000-8000-000000000402";
  const nodeId = "00000000-0000-4000-8000-000000000404";
  const outputId = "00000000-0000-4000-8000-000000000405";
  const edgeIds = [403, 406, 408, 409, 410].map(
    (suffix) => `00000000-0000-4000-8000-${String(suffix).padStart(12, "0")}`,
  );
  return {
    ...makeGetGuidedResponse(),
    guided_session: {
      ...makeGetGuidedResponse().guided_session,
      step: "step_3_transforms",
    },
    next_turn: {
      type: "propose_pipeline",
      step_index: 2,
      turn_token: "c".repeat(64),
      payload: {
        proposal_id: "00000000-0000-4000-8000-000000000401",
        draft_hash: "d".repeat(64),
        summary: "guided.proposal.summary.full_graph.v1",
        rationale: "guided.proposal.rationale.review_required.v1",
        component_counts: { sources: 1, nodes: 1, edges: 5, outputs: 1 },
        blockers: [
          {
            code: "policy_review_required",
            category: "policy",
            summary: "guided.proposal.blocker.policy_review_required.v1",
            edit_target: { kind: "node", stable_id: nodeId },
          },
        ],
        graph: {
          sources: [
            { stable_id: sourceId, label: "source-1", plugin: { kind: "source", id: "csv" } },
          ],
          edges: [
            {
              stable_id: edgeIds[0],
              from_endpoint: { kind: "source", stable_id: sourceId },
              to_endpoint: { kind: "node", stable_id: nodeId },
              flow: { kind: "source_success", branch: null },
            },
            {
              stable_id: edgeIds[1],
              from_endpoint: { kind: "source", stable_id: sourceId },
              to_endpoint: { kind: "discard" },
              flow: { kind: "source_validation_failure" },
            },
            {
              stable_id: edgeIds[2],
              from_endpoint: { kind: "node", stable_id: nodeId },
              to_endpoint: { kind: "output", stable_id: outputId },
              flow: { kind: "node_success", branch: null },
            },
            {
              stable_id: edgeIds[3],
              from_endpoint: { kind: "node", stable_id: nodeId },
              to_endpoint: { kind: "discard" },
              flow: { kind: "node_error" },
            },
            {
              stable_id: edgeIds[4],
              from_endpoint: { kind: "output", stable_id: outputId },
              to_endpoint: { kind: "discard" },
              flow: { kind: "output_write_failure" },
            },
          ],
        },
        nodes: [
          {
            stable_id: nodeId,
            label: "node-1",
            node_type: "transform",
            plugin: { kind: "transform", id: "schema_guard" },
            behavior: { kind: "transform" },
          },
        ],
        outputs: [
          { stable_id: outputId, label: "output-1", plugin: { kind: "sink", id: "json" } },
        ],
        edit_targets: [
          { kind: "source", stable_id: sourceId },
          { kind: "node", stable_id: nodeId },
          { kind: "output", stable_id: outputId },
          ...edgeIds.map((stable_id) => ({ kind: "edge" as const, stable_id })),
        ],
      },
    },
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

    it("accepts the exact closed full-graph proposal projection", async () => {
      const body = makeProposalResponse();
      fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

      const decoded = await getGuided("sess-proposal");

      expect(decoded).toEqual(body);
      expect(decoded.next_turn).not.toBe(body.next_turn);
      if (decoded.next_turn?.type !== "propose_pipeline" || body.next_turn?.type !== "propose_pipeline") {
        throw new Error("proposal fixture did not decode as propose_pipeline");
      }
      expect(decoded.next_turn.payload).not.toBe(body.next_turn.payload);
      expect(decoded.next_turn.payload.graph.edges).not.toBe(body.next_turn.payload.graph.edges);
      body.next_turn.payload.graph.edges[0].flow = { kind: "source_validation_failure" };
      expect(decoded.next_turn.payload.graph.edges[0].flow).toEqual({
        kind: "source_success",
        branch: null,
      });
    });

    it("constructs new guided-session, history, and chat-turn objects at the decoder boundary", async () => {
      const body = makeGetGuidedResponse();
      body.guided_session.history = [{
        step: "step_1_source",
        turn_type: "single_select",
        payload_hash: "a".repeat(64),
        response_hash: "b".repeat(64),
        summary: "Selected CSV",
        emitter: "server",
      }];
      body.guided_session.chat_history = [{
        role: "user",
        content: "Use CSV",
        seq: 1,
        step: "step_1_source",
        ts_iso: "2026-07-19T00:00:00Z",
        assistant_message_kind: null,
        synthetic_failure_reason: null,
      }];
      body.guided_session.chat_turn_seq = 1;
      fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

      const decoded = await getGuided("sess-session-identity");

      expect(decoded.guided_session).not.toBe(body.guided_session);
      expect(decoded.guided_session.history).not.toBe(body.guided_session.history);
      expect(decoded.guided_session.history[0]).not.toBe(body.guided_session.history[0]);
      expect(decoded.guided_session.chat_history).not.toBe(body.guided_session.chat_history);
      expect(decoded.guided_session.chat_history[0]).not.toBe(body.guided_session.chat_history[0]);
      body.guided_session.chat_history[0].content = "mutated after decoding";
      expect(decoded.guided_session.chat_history[0].content).toBe("Use CSV");
    });

    it.each(["guided session", "chat turn"])(
      "rejects an unexpected field on the closed %s DTO",
      async (target) => {
        const body = makeGetGuidedResponse();
        if (target === "guided session") {
          (body.guided_session as unknown as Record<string, unknown>).canary = true;
        } else {
          body.guided_session.chat_history = [{
            role: "user",
            content: "Use CSV",
            seq: 1,
            step: "step_1_source",
            ts_iso: "2026-07-19T00:00:00Z",
            assistant_message_kind: null,
            synthetic_failure_reason: null,
          }];
          (body.guided_session.chat_history[0] as unknown as Record<string, unknown>).canary = true;
        }
        fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

        await expect(getGuided("sess-extra-dto-field")).rejects.toThrow(/unexpected canary/i);
      },
    );

    it("requires all current composition response keys and normalizes nullable projections", async () => {
      const body = makeGetGuidedResponse() as unknown as Record<string, unknown>;
      body.composition_state = {
        id: "state-1",
        session_id: "sess-1",
        version: 3,
        sources: null,
        nodes: null,
        edges: null,
        outputs: null,
        metadata: null,
        is_valid: false,
        validation_errors: null,
        validation_warnings: null,
        validation_suggestions: null,
        derived_from_state_id: null,
        created_at: "2026-07-19T00:00:00Z",
        composer_meta: null,
        plugin_policy_findings: [],
      };
      fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

      const result = await getGuided("sess-current-state");

      expect(result.composition_state).toEqual({
        ...(body.composition_state as Record<string, unknown>),
        sources: {},
        nodes: [],
        edges: [],
        outputs: [],
        metadata: { name: null, description: null },
      });
    });

    it.each(["session_id", "is_valid", "validation_errors", "validation_warnings", "validation_suggestions", "derived_from_state_id", "created_at", "composer_meta", "plugin_policy_findings"])(
      "rejects composition response missing required %s",
      async (missingKey) => {
        const body = makeGetGuidedResponse() as unknown as Record<string, unknown>;
        const state: Record<string, unknown> = {
          id: "state-1",
          session_id: "sess-1",
          version: 1,
          sources: {},
          nodes: [],
          edges: [],
          outputs: [],
          metadata: { name: null, description: null },
          is_valid: true,
          validation_errors: null,
          validation_warnings: null,
          validation_suggestions: null,
          derived_from_state_id: null,
          created_at: "2026-07-19T00:00:00Z",
          composer_meta: null,
          plugin_policy_findings: [],
        };
        delete state[missingKey];
        body.composition_state = state;
        fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

        await expect(getGuided("sess-incomplete-state")).rejects.toThrow(/invalid guided response/i);
      },
    );

    it.each([
      ["arbitrary summary", (body: GetGuidedResponse) => {
        if (body.next_turn?.type === "propose_pipeline") body.next_turn.payload.summary = "password123";
      }],
      ["missing validation-failure flow", (body: GetGuidedResponse) => {
        if (body.next_turn?.type === "propose_pipeline") {
          body.next_turn.payload.graph.edges.splice(1, 1);
          body.next_turn.payload.component_counts.edges -= 1;
        }
      }],
      ["component cycle", (body: GetGuidedResponse) => {
        if (body.next_turn?.type === "propose_pipeline") {
          body.next_turn.payload.graph.edges[2].to_endpoint = {
            kind: "node",
            stable_id: body.next_turn.payload.nodes[0].stable_id,
          };
        }
      }],
      ["route alias canary", (body: GetGuidedResponse) => {
        if (body.next_turn?.type === "propose_pipeline") {
          body.next_turn.payload.nodes[0] = {
            stable_id: body.next_turn.payload.nodes[0].stable_id,
            label: "node-1",
            node_type: "gate",
            plugin: null,
            behavior: { kind: "gate", route_aliases: ["Bearer-credential"], fork_branches: [] },
          };
          body.next_turn.payload.graph.edges[2].flow = {
            kind: "gate_route",
            route: "Bearer-credential",
            branch: null,
          };
          body.next_turn.payload.graph.edges.splice(3, 1);
          body.next_turn.payload.component_counts.edges -= 1;
        }
      }],
    ])("rejects proposal projection with %s", async (_label, mutate) => {
      const body = makeProposalResponse();
      mutate(body);
      fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

      await expect(getGuided("sess-invalid-proposal")).rejects.toThrow(/invalid guided response/i);
    });

    it.each([
      [
        "unknown discriminator",
        {
          type: "unknown_turn",
          step_index: 0,
          turn_token: "a".repeat(64),
          payload: {},
        },
      ],
      [
        "extra turn field",
        {
          type: "single_select",
          step_index: 0,
          turn_token: "a".repeat(64),
          payload: { question: "Choose", options: [], allow_custom: false },
          canary: "must-not-reach-store",
        },
      ],
      [
        "mismatched type and payload",
        {
          type: "single_select",
          step_index: 0,
          turn_token: "a".repeat(64),
          payload: { observed: { columns: [], samples: [], warnings: [] } },
        },
      ],
    ])("rejects %s before it reaches the store", async (_label, nextTurn) => {
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ ...makeGetGuidedResponse(), next_turn: nextTurn }),
      } as Response);

      await expect(getGuided("sess-invalid")).rejects.toThrow(
        /invalid guided response/i,
      );
    });

    it.each([
      ["completed reason", { kind: "completed", reason: "user_pressed_exit", pipeline_yaml: "source: {}" }],
      ["completed empty YAML", { kind: "completed", reason: null, pipeline_yaml: "" }],
      ["exited missing reason", { kind: "exited_to_freeform", reason: null, pipeline_yaml: null }],
      ["exited YAML", { kind: "exited_to_freeform", reason: "user_pressed_exit", pipeline_yaml: "source: {}" }],
    ])("rejects terminal with invalid %s cross-fields", async (_label, terminal) => {
      const body = makeGetGuidedResponse() as unknown as Record<string, unknown>;
      (body.guided_session as Record<string, unknown>).terminal = terminal;
      body.terminal = terminal;
      fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

      await expect(getGuided("sess-invalid-terminal")).rejects.toThrow(/invalid guided response/i);
    });

    it.each([
      ["kind", "credential-canary"],
      ["tier", "operator-only"],
      ["item_kind", "filesystem-path"],
    ])("rejects schema-form knob with unknown %s", async (fieldName, value) => {
      const body = makeGetGuidedResponse() as unknown as Record<string, unknown>;
      (body.guided_session as Record<string, unknown>).step = "step_1_source";
      const field: Record<string, unknown> = {
        name: "path",
        label: "Path",
        kind: "string-list",
        required: true,
        nullable: false,
        item_kind: "text",
      };
      field[fieldName] = value;
      body.next_turn = {
        type: "schema_form",
        step_index: 0,
        turn_token: "a".repeat(64),
        payload: {
          mode: "plugin_options",
          plugin: "csv",
          knobs: { fields: [field] },
          prefilled: {},
        },
      };
      fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

      await expect(getGuided("sess-invalid-knob")).rejects.toThrow(/invalid guided response/i);
    });

    it.each([
      ["source", (state: Record<string, unknown>) => {
        ((state.sources as Record<string, unknown>).source as Record<string, unknown>).plugin = 7;
      }],
      ["node type", (state: Record<string, unknown>) => {
        ((state.nodes as Record<string, unknown>[])[0]).node_type = "credential-canary";
      }],
      ["edge type", (state: Record<string, unknown>) => {
        ((state.edges as Record<string, unknown>[])[0]).edge_type = "provider-route";
      }],
      ["output", (state: Record<string, unknown>) => {
        ((state.outputs as Record<string, unknown>[])[0]).options = [];
      }],
      ["metadata", (state: Record<string, unknown>) => {
        (state.metadata as Record<string, unknown>).name = 9;
      }],
      ["validation warning", (state: Record<string, unknown>) => {
        state.validation_warnings = [{ component: "source", message: 7, severity: "high" }];
      }],
      ["policy finding", (state: Record<string, unknown>) => {
        state.plugin_policy_findings = [{
          component_id: "source",
          plugin_id: "source:csv",
          reason_code: "credential-canary",
          snapshot_fingerprint: "a".repeat(64),
        }];
      }],
    ])("rejects malformed nested composition %s", async (_label, mutate) => {
      const state: Record<string, unknown> = {
        id: "state-1",
        session_id: "sess-1",
        version: 1,
        sources: {
          source: {
            plugin: "csv",
            options: {},
            on_success: "records",
            on_validation_failure: "discard",
          },
        },
        nodes: [{
          id: "validate",
          node_type: "transform",
          plugin: "schema_guard",
          input: "records",
          on_success: "valid",
          on_error: "discard",
          options: {},
        }],
        edges: [{
          id: "edge-1",
          from_node: "source",
          to_node: "validate",
          edge_type: "on_success",
          label: null,
        }],
        outputs: [{ name: "result", plugin: "json", options: {}, on_write_failure: "discard" }],
        metadata: { name: null, description: null },
        is_valid: true,
        validation_errors: [],
        validation_warnings: [],
        validation_suggestions: [],
        derived_from_state_id: null,
        created_at: "2026-07-19T00:00:00Z",
        composer_meta: null,
        plugin_policy_findings: [],
      };
      mutate(state);
      const body = makeGetGuidedResponse() as unknown as Record<string, unknown>;
      body.composition_state = state;
      fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

      await expect(getGuided("sess-invalid-state")).rejects.toThrow(/invalid guided response/i);
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

  describe("chatGuided", () => {
    it("sends only the retry identity, current turn token, and visible message", async () => {
      const body: GuidedChatRequest = {
        operation_id: "00000000-0000-4000-8000-000000000001",
        turn_token: "a".repeat(64),
        message: "Use CSV",
      };
      const responseBody: GuidedChatResponse = {
        assistant_message: "CSV selected.",
        assistant_message_kind: "assistant",
        guided_session: makeGetGuidedResponse().guided_session,
        next_turn: {
          type: "schema_form",
          step_index: 0,
          turn_token: "b".repeat(64),
          payload: {
            mode: "plugin_options",
            plugin: "csv",
            knobs: { fields: [] },
            prefilled: {},
          },
        },
        terminal: null,
        composition_state: {
          id: "state-1",
          session_id: "sess-1",
          version: 1,
          nodes: [],
          edges: [],
          sources: {},
          outputs: [],
          metadata: { name: null, description: null },
          is_valid: true,
          validation_errors: null,
          validation_warnings: null,
          validation_suggestions: null,
          derived_from_state_id: null,
          created_at: "2026-07-19T00:00:00Z",
          composer_meta: null,
          plugin_policy_findings: [],
        },
      };
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => responseBody,
      } as Response);

      await expect(chatGuided("sess-1", body)).resolves.toEqual(responseBody);

      expect(fetchSpy).toHaveBeenCalledWith(
        "/api/sessions/sess-1/guided/chat",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify(body),
        }),
      );
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

    it.each([
      ["accept", {
        operation_id: "00000000-0000-4000-8000-000000000611",
        turn_token: "b".repeat(64),
        chosen: ["accept"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: "00000000-0000-4000-8000-000000000612",
        draft_hash: "c".repeat(64),
        edit_target: null,
        control_signal: null,
      }],
      ["reject", {
        operation_id: "00000000-0000-4000-8000-000000000613",
        turn_token: "b".repeat(64),
        chosen: null,
        edited_values: null,
        custom_inputs: null,
        proposal_id: "00000000-0000-4000-8000-000000000612",
        draft_hash: "c".repeat(64),
        edit_target: null,
        control_signal: "reject",
      }],
      ["target-only revise", {
        operation_id: "00000000-0000-4000-8000-000000000614",
        turn_token: "b".repeat(64),
        chosen: null,
        edited_values: null,
        custom_inputs: null,
        proposal_id: "00000000-0000-4000-8000-000000000612",
        draft_hash: "c".repeat(64),
        edit_target: {
          kind: "edge",
          stable_id: "00000000-0000-4000-8000-000000000615",
        },
        control_signal: null,
      }],
    ] satisfies [string, GuidedRespondRequest][])(
      "serializes the exact proposal-bound %s request",
      async (_label, request) => {
        fetchSpy.mockResolvedValue({
          ok: true,
          status: 200,
          json: async () => makeRespondResponse(),
        } as Response);

        await respondGuided("sess-proposal", request);

        const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
        expect(url).toBe("/api/sessions/sess-proposal/guided/respond");
        expect(JSON.parse(init.body as string)).toEqual(request);
      },
    );

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
        advisor_checkpoints: true,
      };
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => body,
      } as Response);

      const result = await startGuidedSession(
        "sess-1",
        "tutorial",
        "00000000-0000-4000-8000-000000000001",
      );

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/guided/start");
      expect(init.method).toBe("POST");
      expect((init.headers as Record<string, string>)["Content-Type"]).toBe(
        "application/json",
      );
      expect(JSON.parse(init.body as string)).toEqual({
        profile: "tutorial",
        operation_id: "00000000-0000-4000-8000-000000000001",
      });
      expect(result.guided_session.profile?.advisor_checkpoints).toBe(true);
    });
  });

  describe("retry-safe mutations", () => {
    it("sends the explicit operation id and accepts only a fork locator", async () => {
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 201,
        json: async () => ({ session_id: "00000000-0000-4000-8000-000000000009" }),
      } as Response);

      const result = await forkFromMessage(
        "sess-1",
        "00000000-0000-4000-8000-000000000008",
        "message-1",
        "edited",
      );

      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/fork");
      expect(JSON.parse(init.body as string)).toEqual({
        operation_id: "00000000-0000-4000-8000-000000000008",
        from_message_id: "message-1",
        new_message_content: "edited",
      });
      expect(result).toEqual({ session_id: "00000000-0000-4000-8000-000000000009" });
    });

    it.each([
      ["missing", {}],
      ["extra legacy fields", { session_id: "00000000-0000-4000-8000-000000000009", title: "legacy" }],
      ["non-string", { session_id: 9 }],
      ["noncanonical", { session_id: "{00000000-0000-4000-8000-000000000009}" }],
    ])("rejects a %s 2xx fork locator as committed-response ambiguity", async (_label, body) => {
      fetchSpy.mockResolvedValue({ ok: true, status: 201, json: async () => body } as Response);

      await expect(
        forkFromMessage("sess-1", "00000000-0000-4000-8000-000000000008", "message-1", "edited"),
      ).rejects.toBeInstanceOf(ForkCommittedResponseError);
    });

    it("tags a truncated 2xx fork body as committed-response ambiguity", async () => {
      fetchSpy.mockResolvedValue({
        ok: true,
        status: 201,
        json: async () => {
          throw new SyntaxError("truncated JSON");
        },
      } as unknown as Response);

      await expect(
        forkFromMessage("sess-1", "00000000-0000-4000-8000-000000000008", "message-1", "edited"),
      ).rejects.toBeInstanceOf(ForkCommittedResponseError);
    });
    it("sends the store-owned operation id for guided conversion", async () => {
      const body = makeGetGuidedResponse();
      fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

      await convertToGuided("sess-1", "00000000-0000-4000-8000-000000000003");

      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/guided/convert");
      expect(JSON.parse(init.body as string)).toEqual({
        operation_id: "00000000-0000-4000-8000-000000000003",
      });
    });
    it("sends the store-owned operation id for guided re-entry", async () => {
      const body = makeGetGuidedResponse();
      fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => body } as Response);

      await reenterGuided("sess-1", "00000000-0000-4000-8000-000000000001");

      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/guided/reenter");
      expect(JSON.parse(init.body as string)).toEqual({
        operation_id: "00000000-0000-4000-8000-000000000001",
      });
    });

    it("sends the same explicit operation id with a state revert", async () => {
      fetchSpy.mockResolvedValue({ ok: true, status: 200, json: async () => ({ id: "state-new" }) } as Response);

      await revertToVersion("sess-1", "state-old", "00000000-0000-4000-8000-000000000002");

      const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
      expect(url).toBe("/api/sessions/sess-1/state/revert");
      expect(JSON.parse(init.body as string)).toEqual({
        operation_id: "00000000-0000-4000-8000-000000000002",
        state_id: "state-old",
      });
    });
  });
});
