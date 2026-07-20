// ============================================================================
// Guided-mode protocol types
//
// TypeScript mirrors of the pydantic wire schemas for the guided composer
// protocol.  Source of truth:
//   src/elspeth/web/composer/guided/protocol.py   — TurnType, ControlSignal, GuidedStep
//   src/elspeth/web/composer/guided/state_machine.py — TerminalKind, TerminalReason
//   src/elspeth/web/sessions/schemas.py:213-296   — wire response/request shapes
// ============================================================================

import type { CompositionState } from "./index";

/** Mirrors GuidedChatRequest.message max_length in src/elspeth/web/sessions/schemas.py. */
export const GUIDED_CHAT_MESSAGE_MAX_LENGTH = 4096;

// ── Enums ────────────────────────────────────────────────────────────────────

export type TurnType =
  | "inspect_and_confirm"
  | "single_select"
  | "multi_select_with_custom"
  | "schema_form"
  | "review_components"
  | "propose_pipeline"
  | "confirm_wiring";

/**
 * Mirrors protocol.py ControlSignal (source-of-truth enum). "back" and
 * "passthrough" were missing here — pre-existing drift from the backend
 * enum, not a deliberate narrowing — surfaced and fixed alongside the C-3
 * passthrough wiring (composer first-principles review 2026-07-04, C-3).
 * "back" has no frontend caller yet (no Back-nav widget exists — reframe
 * epic elspeth-e7757e5c58 slices D/E/F remain open); it is typed here so
 * the union tracks the backend's closed enum rather than only the signals
 * some widget happens to send today.
 */
export type ControlSignal =
  | "exit_to_freeform"
  | "request_advisor"
  | "reject"
  | "back"
  | "passthrough";

export type GuidedStep =
  | "step_1_source"
  | "step_2_sink"
  | "step_3_transforms"
  | "step_4_wire";

export type TerminalKind = "completed" | "exited_to_freeform";

export type TerminalReason = "user_pressed_exit";

// ── Domain shapes ─────────────────────────────────────────────────────────────

/** Wire: TurnRecordResponse (schemas.py:213-220). Mirrors a single history entry. */
export interface TurnRecord {
  step: GuidedStep;
  turn_type: TurnType;
  payload_hash: string;
  response_hash: string | null;
  summary: string | null;
  /** Closed value domain (state_machine.py:75-82): server-emitted or LLM-emitted. */
  emitter: "server" | "llm";
}

/** Wire: TerminalStateResponse with its runtime cross-field invariant. */
export type TerminalState =
  | {
      kind: "completed";
      reason: null;
      pipeline_yaml: string;
    }
  | {
      kind: "exited_to_freeform";
      reason: "user_pressed_exit";
      pipeline_yaml: null;
    };

/**
 * Wire: ChatTurnResponse (schemas.py — Phase A slice 5).  Mirrors a single
 * entry in GuidedSession.chat_history.  Server-emitted; all values are
 * authoritative (Tier 1).  Ordering is driven by `seq`, not `ts_iso` —
 * two turns produced in the same request share a wall-clock second.
 *
 * `assistant_message_kind` (composer first-principles review 2026-07-04,
 * C-2) discriminates a real LLM reply from a synthetic failure message
 * (scaffold-guard rejection or provider unavailability) — before this
 * field, both looked like an ordinary assistant turn and a synthetic
 * failure could become the "Current decision" headline (guidedRationale.ts)
 * or render with the normal "ELSPETH said:" bubble treatment
 * (GuidedChatHistory.tsx). Only meaningful on `role: "assistant"` entries.
 * The backend always emits both closed discriminator keys. User turns carry
 * null for both; real assistant turns carry `"assistant"` and a null reason;
 * synthetic failures carry their kind and an allowlisted non-null reason.
 */
export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  seq: number;
  step: GuidedStep;
  ts_iso: string;
  assistant_message_kind: "assistant" | "synthetic_failure" | null;
  synthetic_failure_reason: "quality_guard" | "unavailable" | "not_applied" | null;
}

/**
 * Wire: WorkflowProfileResponse (schemas.py — WorkflowProfileResponse).
 * Server-owned workflow profile (the four behavior flags).
 * A `null` `GuidedSession.profile` is the empty/live-guided profile.
 */
export interface WorkflowProfile {
  coaching: boolean;
  bookends: boolean;
  advisor_checkpoints: boolean;
}

/**
 * Wire: GuidedSessionResponse (schemas.py:231-242 post-slice-5).
 * `chat_history` + `chat_turn_seq` were added in Phase A slice 5 to
 * persist per-step chat across reloads; before slice 5 the frontend
 * carried an in-memory `guidedChatHistory` array, which is replaced
 * verbatim by this server-authoritative field.
 */
export interface GuidedSession {
  step: GuidedStep;
  history: TurnRecord[];
  terminal: TerminalState | null;
  chat_history: ChatTurn[];
  chat_turn_seq: number;
  /** Server-owned WorkflowProfile, or `null` for the empty/live-guided profile. */
  profile: WorkflowProfile | null;
}

interface TurnPayloadEnvelope<TType extends TurnType, TPayload> {
  type: TType;
  step_index: number;
  turn_token: string;
  payload: TPayload;
}

/** Closed wire union for TurnPayloadResponse. */
export type TurnPayload =
  | TurnPayloadEnvelope<"inspect_and_confirm", InspectAndConfirmPayload>
  | TurnPayloadEnvelope<"single_select", SingleSelectPayload>
  | TurnPayloadEnvelope<"multi_select_with_custom", MultiSelectWithCustomPayload>
  | TurnPayloadEnvelope<"schema_form", SchemaFormPayload>
  | TurnPayloadEnvelope<"review_components", ComponentReviewPayload>
  | TurnPayloadEnvelope<"propose_pipeline", ProposePipelinePayload>
  | TurnPayloadEnvelope<"confirm_wiring", WireStageData>;

// ── Endpoint envelopes ───────────────────────────────────────────────────────

/** Response for GET /api/sessions/{id}/guided (schemas.py:255-261). */
export interface GetGuidedResponse {
  guided_session: GuidedSession;
  next_turn: TurnPayload | null;
  terminal: TerminalState | null;
  composition_state: CompositionState | null;
}

export type GuidedOperationFailureCode =
  | "provider_unavailable"
  | "provider_timeout"
  | "invalid_provider_response"
  | "stale_conflict"
  | "integrity_error"
  | "custody_error"
  | "quota_exceeded"
  | "operation_failed"
  | "request_cancelled";

export type GuidedStartOperationReconciliation =
  | { status: "in_progress" }
  | { status: "failed"; failure_code: GuidedOperationFailureCode }
  | { status: "completed"; composition_state_id: string };

export interface GuidedEditTarget {
  kind: "source" | "node" | "edge" | "output";
  stable_id: string;
}

interface UnboundProposalFields {
  proposal_id: null;
  draft_hash: null;
  edit_target: null;
}

interface BoundProposalFields {
  proposal_id: string;
  draft_hash: string;
}

export type NonEmptyStringArray = [string, ...string[]];

export type GuidedComponentKind = "source" | "output";

export interface GuidedComponentTarget {
  kind: GuidedComponentKind;
  stable_id: string;
}

export type GuidedComponentAction =
  | { action: "add"; component_kind: GuidedComponentKind }
  | { action: "edit"; target: GuidedComponentTarget }
  | { action: "remove"; target: GuidedComponentTarget }
  | {
      action: "reorder";
      component_kind: GuidedComponentKind;
      stable_ids: NonEmptyStringArray;
    }
  | { action: "finish"; component_kind: GuidedComponentKind };

/** One exact legal response action before retry/turn identity is attached. */
export type GuidedRespondAction =
  | (UnboundProposalFields & {
      chosen: NonEmptyStringArray;
      edited_values: null;
      custom_inputs: null;
      control_signal: null;
    })
  | (UnboundProposalFields & {
      chosen: null;
      edited_values: null;
      custom_inputs: NonEmptyStringArray;
      control_signal: null;
    })
  | (UnboundProposalFields & {
      chosen: NonEmptyStringArray;
      edited_values: null;
      custom_inputs: NonEmptyStringArray;
      control_signal: null;
    })
  | (UnboundProposalFields & {
      chosen: null;
      edited_values: Record<string, unknown>;
      custom_inputs: null;
      control_signal: null;
    })
  | (UnboundProposalFields & {
      chosen: null;
      edited_values: null;
      custom_inputs: null;
      control_signal: Exclude<ControlSignal, "reject">;
    })
  | (BoundProposalFields & {
      chosen: ["accept"];
      edited_values: null;
      custom_inputs: null;
      edit_target: null;
      control_signal: null;
    })
  | (BoundProposalFields & {
      chosen: null;
      edited_values: null;
      custom_inputs: null;
      edit_target: null;
      control_signal: "reject";
    })
  | (BoundProposalFields & {
      chosen: null;
      edited_values: null;
      custom_inputs: null;
      edit_target: GuidedEditTarget;
      control_signal: null;
    })
  | (UnboundProposalFields & {
      chosen: null;
      edited_values: null;
      custom_inputs: null;
      control_signal: null;
      component_action: GuidedComponentAction;
    });

/** Exact proposal control whose retry descriptor remains in local custody. */
export type GuidedProposalRetryAction =
  | { kind: "accept" }
  | { kind: "reject" }
  | { kind: "revise"; edit_target: GuidedEditTarget };

/**
 * Local review lifecycle for one exact durable guided proposal projection.
 * Every state remains bound to the proposal id and draft hash whose controls
 * it describes; a reload never silently transfers an old action to a new
 * proposal.
 */
export type GuidedProposalReviewState =
  | {
      status: "active" | "submitting" | "reloading" | "stale";
      proposal_id: string;
      draft_hash: string;
    }
  | {
      status: "error";
      proposal_id: string;
      draft_hash: string;
      message: string;
      retryable: true;
      retry_action: GuidedProposalRetryAction;
    }
  | {
      status: "error";
      proposal_id: string;
      draft_hash: string;
      message: string;
      retryable: false;
      retry_action: null;
    };

/** Exact request body for POST /api/sessions/{id}/guided/respond. */
export type GuidedRespondRequest = GuidedRespondAction & {
  operation_id: string;
  turn_token: string | null;
};

/** Response for POST /api/sessions/{id}/guided/respond (schemas.py:286-296). */
export interface GuidedRespondResponse {
  guided_session: GuidedSession;
  next_turn: TurnPayload | null;
  terminal: TerminalState | null;
  composition_state: CompositionState | null;
}

/**
 * Response for GET /api/sessions/{id}/guided/tutorial-sample
 * (sessions/schemas.py — TutorialSampleResponse, p4 Task 8a).
 *
 * Runtime-derived inputs for the tutorial worked example: the 3 synthetic
 * sample-page URLs (`sample_urls`) computed from the active tutorial session's
 * resolved origin and appended to the locked STEP_1 prompt so the source driver
 * can parse the runtime-served addresses.
 *
 * No `allowed_hosts` is carried: the synthetic pages are publicly hosted, so the
 * tutorial's web_scrape node relies on the plugin default
 * `allowed_hosts="public_only"` — the client never sets an SSRF allowlist.
 */
export interface TutorialSampleResponse {
  sample_urls: string[];
}

/**
 * Request body for POST /api/sessions/{id}/guided/chat (schemas.py — GuidedChatRequest).
 *
 * The server derives the stage from the checkpoint occurrence identified by
 * `turn_token`; the client cannot restate a positional step. `operation_id`
 * remains stable across an ambiguous transport retry of this exact request.
 */
export interface GuidedChatRequest {
  operation_id: string;
  turn_token: string;
  message: string;
}

/**
 * Response for POST /api/sessions/{id}/guided/chat (schemas.py — GuidedChatResponse).
 *
 * `assistant_message` is the bounded assistant reply or a synthetic failure
 * message. `assistant_message_kind`
 * (C-2) is the top-level discriminator for THIS response's reply —
 * `"synthetic_failure"` covers both a scaffold-guard rejection and provider
 * unavailability; the same value is mirrored onto the tail entry of
 * `guided_session.chat_history` (ChatTurn.assistant_message_kind), which is
 * what the UI actually renders from (GuidedChatHistory reads
 * `guidedSession.chat_history`, not this envelope field directly).
 *
 * All four state fields are the server's authoritative post-operation view.
 * Step 1/2 configuration suggestions may project a pure schema-8 transition;
 * generated inline source bytes remain non-applying until blob custody can
 * participate in the same atomic settlement.
 */
export interface GuidedChatResponse {
  assistant_message: string;
  assistant_message_kind: "assistant" | "synthetic_failure";
  guided_session: GuidedSession;
  next_turn: TurnPayload | null;
  terminal: TerminalState | null;
  composition_state: CompositionState | null;
}


// ── Per-turn payload shapes ───────────────────────────────────────────────────
// Each widget owns its payload type; add yours when you implement the widget.
// Field names use snake_case to mirror the wire (GuidedRespondRequest does too).
//
// SHARED across single_select / multi_select_with_custom:
export interface Option {
  id: string;
  label: string;
  hint: string | null; // null, not optional — wire always sends the key
}

/** Wire: SingleSelectPayload (protocol.py:40-43). */
export interface SingleSelectPayload {
  question: string;
  options: Option[];
  allow_custom: boolean;
}

/** Wire: MultiSelectWithCustomPayload (protocol.py:46-50). */
export interface MultiSelectWithCustomPayload {
  question: string;
  options: Option[];
  /** Option IDs initially checked; subset of `options[].id`. */
  default_chosen: string[];
  /**
   * Server-emitted label for the "let source decide" escape button, or null.
   *
   * The frontend renders this as a first-class escape choice. Submitting it
   * sends the standalone `passthrough` control signal; the backend combines
   * that with the persisted sink intent and records schema_mode="observed".
   */
  escape_label: string | null;
}

/** Wire: _Observed (protocol.py:30-33). Nested inside InspectAndConfirmPayload. */
export interface Observed {
  columns: string[];
  samples: Record<string, unknown>[];
  warnings: string[];
}

/** Wire: InspectAndConfirmPayload (protocol.py:36-37). */
export interface InspectAndConfirmPayload {
  observed: Observed;
}

export type FieldKind =
  | "text"
  | "number-int"
  | "number-float"
  | "checkbox"
  | "enum"
  | "string-list"
  | "blob-ref"
  | "json-object"
  | "json-array"
  | "json-value";

export type FieldTier = "essential" | "common" | "advanced";

export interface VisibilityPredicate {
  field: string;
  equals: unknown;
}

export interface KnobField {
  name: string;
  label: string;
  description?: string;
  kind: FieldKind;
  tier?: FieldTier;
  required: boolean;
  default?: unknown;
  nullable: boolean;
  enum?: string[];
  item_kind?: "text" | "number-int" | "number-float";
  visible_when?: VisibilityPredicate;
}

export interface KnobSchema {
  fields: KnobField[];
}

export interface SchemaFormPayload {
  mode: "plugin_options";
  plugin: string;
  knobs: KnobSchema;
  prefilled: Record<string, unknown>;
}

export type ComponentReviewAction = GuidedComponentAction["action"];

export interface ComponentReviewItem {
  stable_id: string;
  name: string;
  plugin: string;
  status: "reviewed";
}

export interface ComponentReviewPayload {
  component_kind: GuidedComponentKind;
  items: ComponentReviewItem[];
  allowed_actions: ComponentReviewAction[];
}

/**
 * Closed, non-executable projection of a durable pipeline proposal. There are
 * no plugin options, paths, prompts, secret values, or model-authored text in
 * this surface.
 */
export type ProposalBlockerCode =
  | "pipeline_invalid"
  | "policy_review_required"
  | "plugin_unavailable"
  | "interpretation_required";

export type ProposalBlockerCategory =
  | "validation"
  | "policy"
  | "availability"
  | "interpretation";

export interface ProposalBlocker {
  code: ProposalBlockerCode;
  category: ProposalBlockerCategory;
  summary: string;
  edit_target: GuidedEditTarget | null;
}

export interface ProposalPluginRef {
  kind: "source" | "transform" | "sink";
  id: string;
}

export type ProposalEndpoint = {
  kind: "source" | "node" | "output";
  stable_id: string;
};

export type ProposalTargetEndpoint = ProposalEndpoint | { kind: "discard" };

export type ProposalFlow =
  | { kind: "source_success"; branch: string | null }
  | { kind: "source_validation_failure" }
  | { kind: "node_success"; branch: string | null }
  | { kind: "node_error" }
  | { kind: "gate_route"; route: string; branch: string | null }
  | { kind: "gate_fork"; routes: string[]; branch: string }
  | { kind: "queue_continue"; branch: string | null }
  | { kind: "coalesce_success"; branch: string | null }
  | { kind: "output_write_failure" };

export type ProposalNodeBehavior =
  | { kind: "transform" }
  | {
      kind: "gate";
      route_aliases: string[];
      fork_branches: Array<{ routes: string[]; branch: string }>;
    }
  | {
      kind: "aggregation";
      trigger_kinds: Array<"count" | "timeout" | "condition">;
      /** Canonical decimal strings preserve Python integers beyond JS safe range. */
      count: string | null;
      timeout_seconds: number | null;
      output_mode: "default" | "passthrough" | "transform";
      /** Canonical signed decimal string; null mirrors an omitted runtime value. */
      expected_output_count: string | null;
    }
  | { kind: "queue" }
  | {
      kind: "coalesce";
      branch_aliases: string[];
      policy: "require_all" | "quorum" | "best_effort" | "first";
      merge: "union" | "nested" | "select";
    };

export interface ProposePipelinePayload {
  proposal_id: string;
  draft_hash: string;
  summary: string;
  rationale: string;
  component_counts: {
    sources: number;
    nodes: number;
    edges: number;
    outputs: number;
  };
  blockers: ProposalBlocker[];
  graph: {
    sources: Array<{
      stable_id: string;
      label: string;
      plugin: ProposalPluginRef;
    }>;
    edges: Array<{
      stable_id: string;
      from_endpoint: ProposalEndpoint;
      to_endpoint: ProposalTargetEndpoint;
      flow: ProposalFlow;
    }>;
  };
  nodes: Array<{
    stable_id: string;
    label: string;
    node_type: "transform" | "gate" | "aggregation" | "queue" | "coalesce";
    plugin: ProposalPluginRef | null;
    behavior: ProposalNodeBehavior;
  }>;
  outputs: Array<{
    stable_id: string;
    label: string;
    plugin: ProposalPluginRef;
  }>;
  edit_targets: GuidedEditTarget[];
}

/**
 * Wire data for the step-4 wiring review: topology describes source/node/output
 * connection labels, while contracts overlay producer/consumer compatibility.
 * Source ids use `source` or `source:<name>` and output ids use
 * `output:<sink_name>`. Warnings and advisor/signoff fields are optional
 * advisory metadata emitted when the backend has something to report.
 */
export interface WireStageData {
  proposal_id: string;
  draft_hash: string;
  topology: {
    sources: Record<
      string,
      {
        id: string;
        plugin: string;
        on_success: string | null;
        on_validation_failure: string;
      }
    >;
    nodes: Array<{
      id: string;
      node_type: string;
      plugin: string | null;
      input: string | null;
      on_success: string | null;
      on_error: string | null;
      routes: Record<string, string> | null;
      fork_to: string[] | null;
      branches: string[] | Record<string, string> | null;
    }>;
    outputs: Array<{
      id: string;
      sink_name: string;
      plugin: string;
      on_write_failure: string;
    }>;
  };
  edge_contracts: Array<{
    from: string;
    to: string;
    producer_guarantees: string[];
    consumer_requires: string[];
    missing_fields: string[];
    satisfied: boolean;
  }>;
  semantic_contracts: Array<Record<string, unknown>>;
  warnings: Array<Record<string, unknown>>;
  advisor_findings?: string;
  signoff_outcome?: string;
  /**
   * Advisor sign-off passes left AFTER the pass that produced this turn. Present
   * only on a RE-EMITTED wire turn (the two sites where the pass budget is in
   * scope); ABSENT on the initial turn and the advisor-off tutorial, so the
   * wire-stage cost copy gates on `passes_remaining !== undefined`.
   */
  passes_remaining?: number;
}

/**
 * CI mirror for Python `SlotType` (`src/elspeth/web/composer/recipes.py`).
 * Not imported by application code — its members are read by the
 * cross-language drift check `scripts/cicd/check_slot_type_cross_language.py`
 * via regex. Removing this interface breaks the CI smoke test
 * `tests/unit/scripts/cicd/test_check_slot_type_cross_language.py`.
 * Recipe decisions render through `KnobSchema`.
 */
export interface RecipeSlotInput {
  slot_type: "blob_id" | "str" | "float" | "int" | "str_list";
}
