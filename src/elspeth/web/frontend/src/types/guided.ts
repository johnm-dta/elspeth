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

// ── Enums ────────────────────────────────────────────────────────────────────

export type TurnType =
  | "inspect_and_confirm"
  | "single_select"
  | "multi_select_with_custom"
  | "schema_form"
  | "propose_chain"
  | "recipe_offer";

export type ControlSignal =
  | "exit_to_freeform"
  | "request_advisor"
  | "reject";

export type GuidedStep =
  | "step_1_source"
  | "step_2_sink"
  | "step_2_5_recipe_match"
  | "step_3_transforms";

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
 * Wire: GuidedSessionResponse (schemas.py:231-236).
 * Exactly three wire fields — internal step results never cross the wire.
 */
export interface GuidedSession {
  step: GuidedStep;
  history: TurnRecord[];
  terminal: TerminalState | null;
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

// ── Per-turn payload shapes ───────────────────────────────────────────────────
// Each widget owns its payload type; add yours when you implement the widget.
// Field names use snake_case to mirror the wire (GuidedRespondRequest does too).
//
// SHARED across single_select / multi_select_with_custom / recipe_offer:
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
   * NOTE: As of Task 7.4 this field is INTENTIONALLY NOT consumed by
   * MultiSelectWithCustomTurn. The wire shape for the escape submission
   * requires a cross-layer protocol decision: the plan describes
   * `{edited_values: {schema_mode: "observed", required_fields: []}}`
   * (no `outputs` wrapper, no plugin/options) but the only backend read
   * site (state_machine.py:_advance_step_2 lines 476-483) unconditionally
   * reads `edited_values["outputs"]` as a list of full output dicts. The
   * widget owns neither the plugin nor the options needed to construct
   * that array. The field stays here because the backend still emits it
   * (do not remove).
   *
   * Tracker: filigree elspeth-5e905f3c9d
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

/**
 * Wire: SchemaFormPayload (protocol.py:53-56).
 *
 * schema_block is the output of Pydantic ConfigModel.model_json_schema().
 * The TS type uses Record<string, unknown> because the full JSON Schema spec
 * is not reflected here -- only the subset handled by SchemaFormTurn is consumed
 * (see SchemaFormTurn.tsx SCOPE NOTE for the supported field types).
 *
 * prefilled contains initial field values keyed by property name.
 * Top-level keys in prefilled always correspond to top-level keys in
 * schema_block.properties.
 */
export interface SchemaFormPayload {
  plugin: string;
  schema_block: Record<string, unknown>;
  prefilled: Record<string, unknown>;
}

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
