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
  | "propose_chain"
  | "recipe_offer"
  // Phase 5b: guided-mode interpretation-review widget.  Dispatched from
  // GuidedTurn.tsx (the freeform variant uses InterpretationReviewInlineMessage
  // — different file, different component, no shared widget).
  | "interpretation_review"
  | "confirm_wiring";

export type ControlSignal =
  | "exit_to_freeform"
  | "request_advisor"
  | "reject";

export type GuidedStep =
  | "step_1_source"
  | "step_2_sink"
  | "step_2_5_recipe_match"
  | "step_3_transforms"
  | "step_4_wire";

export type TerminalKind = "completed" | "exited_to_freeform";

export type TerminalReason =
  | "user_pressed_exit"
  | "protocol_violation"
  | "solver_exhausted";

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

/** Wire: TerminalStateResponse (schemas.py:223-228). */
export interface TerminalState {
  kind: TerminalKind;
  reason: TerminalReason | null;
  pipeline_yaml: string | null;
}

/**
 * Wire: ChatTurnResponse (schemas.py — Phase A slice 5).  Mirrors a single
 * entry in GuidedSession.chat_history.  Server-emitted; all values are
 * authoritative (Tier 1).  Ordering is driven by `seq`, not `ts_iso` —
 * two turns produced in the same request share a wall-clock second.
 */
export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  seq: number;
  step: GuidedStep;
  ts_iso: string;
}

/**
 * Wire: WorkflowProfileResponse (schemas.py — WorkflowProfileResponse).
 * Server-owned workflow profile, wire-visible subset. `entry_seed` is
 * consumed server-side at POST /api/sessions/{session_id}/guided/start
 * (`/guided/start` shorthand only) and is NOT on the wire.
 * A `null` `GuidedSession.profile` is the empty/live-guided profile.
 */
export interface WorkflowProfile {
  coaching: boolean;
  bookends: boolean;
  recipe_match: boolean;
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

/** Wire: TurnPayloadResponse (schemas.py:239-252). step_index is 0-based ordinal (number). */
export interface TurnPayload {
  type: TurnType;
  step_index: number;
  payload: unknown;
}

// ── Endpoint envelopes ───────────────────────────────────────────────────────

/** Response for GET /api/sessions/{id}/guided (schemas.py:255-261). */
export interface GetGuidedResponse {
  guided_session: GuidedSession;
  next_turn: TurnPayload | null;
  terminal: TerminalState | null;
  composition_state: CompositionState | null;
}

/** Request body for POST /api/sessions/{id}/guided/respond (schemas.py:264-283). */
export interface GuidedRespondRequest {
  chosen: string[] | null;
  edited_values: Record<string, unknown> | null;
  custom_inputs: string[] | null;
  accepted_step_index: number | null;
  edit_step_index: number | null;
  /** Typed as closed enum client-side; server validates and accepts str for graceful stale-client failure. */
  control_signal: ControlSignal | null;
}

/** Response for POST /api/sessions/{id}/guided/respond (schemas.py:286-296). */
export interface GuidedRespondResponse {
  guided_session: GuidedSession;
  next_turn: TurnPayload | null;
  terminal: TerminalState | null;
  composition_state: CompositionState | null;
}

/**
 * Request body for POST /api/sessions/{id}/guided/chat (schemas.py — GuidedChatRequest).
 *
 * `step_index` is the wire form of the user's current step. The server
 * validates it against the live session.step and returns 409 on mismatch
 * (wizard advanced under the client). `message` is capped at 4096 chars
 * server-side; the frontend lets ChatInput's native maxLength enforce the
 * same limit before submit.
 */
export interface GuidedChatRequest {
  message: string;
  step_index: GuidedStep;
}

/**
 * Response for POST /api/sessions/{id}/guided/chat (schemas.py — GuidedChatResponse).
 *
 * `assistant_message` is the LLM's advisory reply, or the synthetic "I'm
 * unavailable" message on transient LLM failure (Phase A does not yet
 * distinguish the two on the wire; slice 5's ComposerChatTurn audit shape
 * adds that discriminator).
 *
 * Most chat is advisory and returns null for the turn/state fields. Step 1
 * source chat may resolve a complete inline source request; then these fields
 * mirror `/guided/respond` so the store can advance atomically.
 */
export interface GuidedChatResponse {
  assistant_message: string;
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
   * sends `chosen: []` and `custom_inputs: []`; the backend combines that with
   * the persisted sink intent and records schema_mode="observed", meaning the
   * source decides the pass-through field set.
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

export interface RecipeContext {
  recipe_name: string;
  description: string;
  alternatives: string[];
}

export type SchemaFormPayload =
  | {
      mode: "plugin_options";
      plugin: string;
      knobs: KnobSchema;
      prefilled: Record<string, unknown>;
    }
  | {
      mode: "recipe_decision";
      knobs: KnobSchema;
      prefilled: Record<string, unknown>;
      recipe_context: RecipeContext;
    };

/**
 * Wire: _ProposedStep (protocol.py:59-62). One step in a proposed chain.
 *
 * options is an arbitrary plugin options dict (Mapping[str, Any] on the wire);
 * typed as Record<string, unknown> -- callers must not assume any specific shape.
 */
export interface ProposedStep {
  plugin: string;
  options: Record<string, unknown>;
  rationale: string;
}

/**
 * Wire: ProposeChainPayload (protocol.py:64-68).
 *
 * steps    -- ordered list of proposed transforms.
 * why      -- LLM's overall rationale for the proposal.
 * blockers -- obstacles identified by the LLM (may be empty).
 *
 * Submit shape (verified against routes.py:2030-2137):
 *   Accept all: { chosen: ["accept"], ... all other fields null }
 *   Reject / per-step Edit / Ask advisor: NOT wired in Phase 4.
 *
 * Tracker: filigree elspeth-2c08408170 (Step-3 backend handler completion).
 */
export interface ProposeChainPayload {
  steps: ProposedStep[];
  why: string;
  blockers: string[];
}

/**
 * Wire data for the step-4 wiring review: topology describes source/node/output
 * connection labels, while contracts overlay producer/consumer compatibility.
 * Source ids use `source` or `source:<name>` and output ids use
 * `output:<sink_name>`. Warnings and advisor/signoff fields are optional
 * advisory metadata emitted when the backend has something to report.
 */
export interface WireStageData {
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
