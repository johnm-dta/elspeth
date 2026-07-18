// Tests for interpretation.ts — type-assertion style mirroring guided.test.ts.
//
// The compile-time `Equals<A, B>` helper is the load-bearing test: drift
// between the TS union and the Python StrEnum's value set turns into a
// `_exact: never` assignment that fails tsc.  Runtime `expect(...).toHaveLength`
// is belt-and-braces against accidental literal-list edits.

import { describe, expect, it } from "vitest";

import {
  INTERPRETATION_KIND_VALUES,
  isInterpretationKind,
} from "./interpretation";
import type {
  InterpretationChoice,
  InterpretationKind,
  InterpretationSource,
  InterpretationEvent,
  ListInterpretationEventsResponse,
  InterpretationResolveRequest,
  InterpretationResolveResponse,
  InterpretationOptOutResponse,
  OptOutSummaryResponse,
} from "./interpretation";
import type { TurnType } from "./guided";

// Compile-time mutual-extends check.
type Equals<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false;

describe("interpretation protocol types", () => {
  it("InterpretationChoice union has exactly 5 values matching the Python enum", () => {
    const _exact: Equals<
      InterpretationChoice,
      "pending" | "accepted_as_drafted" | "amended" | "opted_out" | "abandoned"
    > = true;
    const all: InterpretationChoice[] = [
      "pending",
      "accepted_as_drafted",
      "amended",
      "opted_out",
      "abandoned",
    ];
    expect(_exact).toBe(true);
    expect(all).toHaveLength(5);
  });

  it("InterpretationSource union has exactly 3 values matching the Python enum", () => {
    const _exact: Equals<
      InterpretationSource,
      "user_approved" | "auto_interpreted_opt_out" | "auto_interpreted_no_surfaces"
    > = true;
    const all: InterpretationSource[] = [
      "user_approved",
      "auto_interpreted_opt_out",
      "auto_interpreted_no_surfaces",
    ];
    expect(_exact).toBe(true);
    expect(all).toHaveLength(3);
  });

  it("InterpretationKind union has exactly 5 values matching the Python enum", () => {
    const _exact: Equals<
      InterpretationKind,
      | "vague_term"
      | "invented_source"
      | "llm_prompt_template"
      | "pipeline_decision"
      | "llm_model_choice"
    > = true;
    const all: InterpretationKind[] = [
      "vague_term",
      "invented_source",
      "llm_prompt_template",
      "pipeline_decision",
      "llm_model_choice",
    ];
    expect(_exact).toBe(true);
    expect(INTERPRETATION_KIND_VALUES).toEqual(all);
    expect(all).toHaveLength(5);
  });

  it("InterpretationKind rejects unknown runtime values", () => {
    expect(isInterpretationKind("vague_term")).toBe(true);
    expect(isInterpretationKind("unknown_kind")).toBe(false);
    expect(isInterpretationKind(null)).toBe(false);
  });

  it("TurnType union has 6 current values", () => {
    const _exact: Equals<
      TurnType,
      | "inspect_and_confirm"
      | "single_select"
      | "multi_select_with_custom"
      | "schema_form"
      | "propose_pipeline"
      | "confirm_wiring"
    > = true;
    const all: TurnType[] = [
      "inspect_and_confirm",
      "single_select",
      "multi_select_with_custom",
      "schema_form",
      "propose_pipeline",
      "confirm_wiring",
    ];
    expect(_exact).toBe(true);
    expect(all).toHaveLength(6);
  });

  it("InterpretationEvent has the exhaustive 23-field shape (compile-time exact-keys check)", () => {
    // Adding/removing a field on the TS interface breaks this assignment.
    // The 23-field count mirrors the InterpretationEventResponse pydantic
    // schema in src/elspeth/web/sessions/schemas.py.
    const _exact: Equals<
      keyof InterpretationEvent,
      | "id"
      | "session_id"
      | "composition_state_id"
      | "affected_node_id"
      | "tool_call_id"
      | "user_term"
      | "kind"
      | "llm_draft"
      | "accepted_value"
      | "choice"
      | "created_at"
      | "resolved_at"
      | "actor"
      | "interpretation_source"
      | "model_identifier"
      | "model_version"
      | "provider"
      | "composer_skill_hash"
      | "arguments_hash"
      | "hash_domain_version"
      | "runtime_model_identifier_at_resolve"
      | "runtime_model_version_at_resolve"
      | "resolved_prompt_template_hash"
    > = true;
    expect(_exact).toBe(true);
  });

  it("InterpretationEvent honours nullability for user_approved pending shape", () => {
    // A pending user_approved row: surface fields populated, accepted_value
    // and resolution provenance null.
    const event: InterpretationEvent = {
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
    };
    expect(event.choice).toBe("pending");
    expect(event.accepted_value).toBeNull();
  });

  it("InterpretationEvent honours all-null surface fields for auto_interpreted_opt_out shape", () => {
    // Opt-out rows: every surface and provenance field is null; only
    // session/choice/source/timestamp/actor are populated.
    const event: InterpretationEvent = {
      id: "evt-opt-1",
      session_id: "sess-1",
      composition_state_id: null,
      affected_node_id: null,
      tool_call_id: null,
      user_term: null,
      kind: null,
      llm_draft: null,
      accepted_value: null,
      choice: "opted_out",
      created_at: "2026-05-18T00:00:00Z",
      resolved_at: "2026-05-18T00:00:00Z",
      actor: "user:owner:u-1",
      interpretation_source: "auto_interpreted_opt_out",
      model_identifier: null,
      model_version: null,
      provider: null,
      composer_skill_hash: null,
      arguments_hash: null,
      hash_domain_version: null,
      runtime_model_identifier_at_resolve: null,
      runtime_model_version_at_resolve: null,
      resolved_prompt_template_hash: null,
    };
    expect(event.composition_state_id).toBeNull();
    expect(event.interpretation_source).toBe("auto_interpreted_opt_out");
  });

  it("InterpretationEvent honours surfaces-null + provenance-required for auto_interpreted_no_surfaces shape", () => {
    // No-surfaces rows: surface fields null but model_identifier and friends
    // ARE populated (LLM was consulted; provenance required).
    const event: InterpretationEvent = {
      id: "evt-no-1",
      session_id: "sess-1",
      composition_state_id: null,
      affected_node_id: null,
      tool_call_id: null,
      user_term: null,
      kind: "llm_prompt_template",
      llm_draft: null,
      accepted_value: null,
      choice: "opted_out",
      created_at: "2026-05-18T00:00:00Z",
      resolved_at: "2026-05-18T00:00:00Z",
      actor: "system:composer",
      interpretation_source: "auto_interpreted_no_surfaces",
      model_identifier: "anthropic/claude-opus-4-7",
      model_version: "20260518",
      provider: "anthropic",
      composer_skill_hash: "deadbeef",
      arguments_hash: null,
      hash_domain_version: null,
      runtime_model_identifier_at_resolve: null,
      runtime_model_version_at_resolve: null,
      resolved_prompt_template_hash: null,
    };
    expect(event.interpretation_source).toBe("auto_interpreted_no_surfaces");
    expect(event.model_identifier).toBe("anthropic/claude-opus-4-7");
    expect(event.user_term).toBeNull();
  });

  it("ListInterpretationEventsResponse and OptOutSummaryResponse share the events-envelope shape", () => {
    // Both envelopes wrap events: InterpretationEvent[].  Same shape on the
    // wire; different routes; type-distinct so downstream code can't
    // accidentally consume the wrong envelope.
    const list: ListInterpretationEventsResponse = { events: [] };
    const summary: OptOutSummaryResponse = { events: [] };
    expect(list.events).toEqual([]);
    expect(summary.events).toEqual([]);
  });

  it("InterpretationResolveRequest narrows choice to the two user-driven values", () => {
    // The resolve request choice union is narrower than InterpretationChoice
    // (no 'pending', 'opted_out', 'abandoned' — those are not user resolve
    // actions).
    const accept: InterpretationResolveRequest = { choice: "accepted_as_drafted" };
    const amend: InterpretationResolveRequest = {
      choice: "amended",
      amended_value: "thoughtful and witty",
    };
    expect(accept.choice).toBe("accepted_as_drafted");
    expect(amend.amended_value).toBe("thoughtful and witty");
  });

  it("InterpretationResolveResponse bundles the resolved event with the new state", () => {
    // Compile-time shape check: response carries event AND new_state in one
    // envelope so the frontend can update atomically.
    type Keys = keyof InterpretationResolveResponse;
    const _exact: Equals<Keys, "event" | "new_state"> = true;
    expect(_exact).toBe(true);
  });

  it("InterpretationOptOutResponse carries the three opt-out fields", () => {
    type Keys = keyof InterpretationOptOutResponse;
    const _exact: Equals<
      Keys,
      "session_id" | "interpretation_review_disabled" | "opted_out_at"
    > = true;
    expect(_exact).toBe(true);
  });
});
