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
