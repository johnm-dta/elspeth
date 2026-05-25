// ============================================================================
// Interpretation-event protocol types (Phase 5b)
//
// TypeScript mirrors of the pydantic wire schemas for the interpretation
// review surface introduced in Phase 5b.  Source of truth:
//   src/elspeth/web/sessions/schemas.py
//     — InterpretationEventResponse, InterpretationResolveRequest,
//       InterpretationResolveResponse, InterpretationOptOutResponse,
//       ListInterpretationEventsResponse, OptOutSummaryResponse
//   src/elspeth/contracts/composer_interpretation.py
//     — InterpretationChoice, InterpretationSource (StrEnum closed lists)
//
// All field shapes (including nullability) mirror the backend exactly.
// The three structural row shapes (user_approved, auto_interpreted_opt_out,
// auto_interpreted_no_surfaces) are distinguished by `interpretation_source`;
// see InterpretationEventRecord docstring in composer_interpretation.py for
// the per-shape nullability table.  Mirroring the full wire shape here
// (rather than a simplified domain shape) means TS narrows correctly when
// branching on `interpretation_source` and `accepted_value`-vs-null at the
// rendering boundary.
// ============================================================================

import type { CompositionState } from "./index";

// ── Enums (closed lists) ────────────────────────────────────────────────────

/**
 * The user's resolution of an interpretation surface.
 *
 * Mirrors the Python `InterpretationChoice` StrEnum.  CLOSED LIST — adding
 * a value requires (a) amending the Phase 5b plan, (b) extending the
 * Python enum, (c) updating the closed-enum tests, (d) widening this
 * union and the matching backend pydantic Literal.
 *
 *  - pending              — surfaced but not yet resolved (working state).
 *  - accepted_as_drafted  — user accepted the LLM's draft verbatim.
 *  - amended              — user edited the draft and accepted the edit.
 *  - opted_out            — user clicked "stop asking" for this session.
 *  - abandoned            — session ended without resolution (page close,
 *                           timeout).  Phase 11 orphan-cleanup writes this.
 */
export type InterpretationChoice =
  | "pending"
  | "accepted_as_drafted"
  | "amended"
  | "opted_out"
  | "abandoned";

/**
 * Structural source of an interpretation event row.
 *
 * Mirrors the Python `InterpretationSource` StrEnum.  CLOSED LIST — adding
 * a value requires the same governance steps as `InterpretationChoice`.
 *
 *  - user_approved                 — LLM surfaced the term; user approved
 *                                    or amended it.  All surface fields
 *                                    (composition_state_id, affected_node_id,
 *                                    tool_call_id, user_term, llm_draft)
 *                                    are populated.
 *  - auto_interpreted_opt_out      — user clicked "stop asking"; the
 *                                    composer-LLM continues to auto-bake.
 *                                    All surface fields are NULL.
 *  - auto_interpreted_no_surfaces  — rate cap exhausted; LLM baked it in
 *                                    without surfacing.  Surface fields
 *                                    NULL; LLM provenance populated.
 */
export type InterpretationSource =
  | "user_approved"
  | "auto_interpreted_opt_out"
  | "auto_interpreted_no_surfaces";

/**
 * Class of LLM-authored assumption surfaced for review.
 *
 * Mirrors the Python `InterpretationKind` StrEnum.  CLOSED LIST — adding
 * a value requires contract amendment, schema update, closed-enum tests,
 * and writer-path audit.
 */
export const INTERPRETATION_KIND_VALUES = [
  "vague_term",
  "invented_source",
  "llm_prompt_template",
  "pipeline_decision",
  "llm_model_choice",
] as const;

export type InterpretationKind = (typeof INTERPRETATION_KIND_VALUES)[number];

export function isInterpretationKind(value: unknown): value is InterpretationKind {
  return (
    typeof value === "string" &&
    (INTERPRETATION_KIND_VALUES as readonly string[]).includes(value)
  );
}

// ── Wire-domain shape ────────────────────────────────────────────────────────

/**
 * Wire mirror of `InterpretationEventResponse` (schemas.py).
 *
 * Every field's nullability matches the backend exactly.  The three
 * structural row shapes are distinguished by `interpretation_source`:
 *
 *  - user_approved
 *      composition_state_id, affected_node_id, tool_call_id, user_term,
 *      llm_draft are all string.  accepted_value is null until resolved.
 *  - auto_interpreted_opt_out
 *      composition_state_id, affected_node_id, tool_call_id, user_term,
 *      llm_draft are all null.  No LLM was consulted; provenance fields
 *      are also null.
 *  - auto_interpreted_no_surfaces
 *      Surface fields (composition_state_id, affected_node_id, tool_call_id,
 *      user_term, llm_draft) are all null.  LLM provenance
 *      (model_identifier, model_version, provider, composer_skill_hash)
 *      IS populated — the LLM produced the auto-bake; provenance is
 *      required for audit purposes.
 *
 * Discriminating on `interpretation_source` at the rendering boundary is
 * the right move; this widened shape preserves the wire contract so
 * narrowing works inside the component switch.
 */
export interface InterpretationEvent {
  id: string;
  session_id: string;
  // Null for auto_interpreted_opt_out and auto_interpreted_no_surfaces rows.
  composition_state_id: string | null;
  // Null for opt-out / no-surfaces rows.
  affected_node_id: string | null;
  tool_call_id: string | null;
  user_term: string | null;
  // Null only for legacy/session-marker rows that do not represent a
  // specific surfaced assumption.
  kind: InterpretationKind | null;
  llm_draft: string | null;
  // Null until the row is resolved; also null for opt-out rows.
  accepted_value: string | null;
  choice: InterpretationChoice;
  created_at: string;
  // ISO timestamp; null while pending.
  resolved_at: string | null;
  // Mirrors ProposalEventRecord.actor: originator:role:id or system:{component}.
  actor: string;
  interpretation_source: InterpretationSource;
  // ── LLM provenance bound to the draft author ─────────────────────────────
  // Null for auto_interpreted_opt_out rows (no LLM was consulted).
  // Required for user_approved and auto_interpreted_no_surfaces rows.
  model_identifier: string | null;
  model_version: string | null;
  provider: string | null;
  // hex SHA-256 of pipeline_composer.md at draft time.
  composer_skill_hash: string | null;
  // hex rfc8785-canonical hash over the active interpretation hash domain.
  // Populated at resolve time; null until then and for opt-out rows.
  arguments_hash: string | null;
  // "v2" once resolved; null until then and for legacy/session opt-out rows.
  hash_domain_version: string | null;
  // F-19: runtime model snapshot at resolve time (may differ from the
  // composer model that produced the draft if a model swap happened
  // between surfacing and resolution).
  runtime_model_identifier_at_resolve: string | null;
  runtime_model_version_at_resolve: string | null;
  // Cross-DB hash anchor (Option A): hex SHA-256 of the resolved
  // prompt-template string.  Null until resolved; null for opt-out rows.
  resolved_prompt_template_hash: string | null;
}

// ── Endpoint envelopes ───────────────────────────────────────────────────────

/**
 * Response for GET /api/sessions/{id}/interpretations.
 *
 * Envelope shape (not a bare JSON array) consistent with every other list
 * route on the session surface; future pagination metadata can be added
 * without a breaking wire change.
 */
export interface ListInterpretationEventsResponse {
  events: InterpretationEvent[];
}

/**
 * Request body for POST /api/sessions/{id}/interpretations/{event_id}/resolve.
 *
 * `opted_out` and `abandoned` are NOT valid resolve choices — opt-out goes
 * through a separate route, and abandoned is written by the session-end
 * cleanup job.  `pending` is the pre-resolve state and is also not a valid
 * input.  Hence the request union is narrower than `InterpretationChoice`.
 *
 * `amended_value` is required when `choice === "amended"` and MUST be
 * omitted when `choice === "accepted_as_drafted"`.  The backend
 * model_validator enforces this; the type allows both shapes so the
 * caller can build the request without conditional spreads.
 */
export interface InterpretationResolveRequest {
  choice: "accepted_as_drafted" | "amended";
  amended_value?: string;
}

/**
 * Response for POST /api/sessions/{id}/interpretations/{event_id}/resolve.
 *
 * Returns the resolved event row PLUS the new composition state produced
 * by patching the affected LLM transform (provenance:
 * `interpretation_resolve`).  Single-envelope shape lets the frontend
 * update its event-list view and composition-state view atomically.
 */
export interface InterpretationResolveResponse {
  event: InterpretationEvent;
  new_state: CompositionState;
}

/**
 * Response for POST /api/sessions/{id}/interpretations/opt_out.
 *
 * `interpretation_review_disabled` is always `true` on success; surfaced
 * explicitly so the caller can re-render the toggle without a follow-up
 * GET.  `opted_out_at` is the persisted timestamp tied to the
 * interpretation_events_table opt-out row written in the same transaction.
 */
export interface InterpretationOptOutResponse {
  session_id: string;
  interpretation_review_disabled: boolean;
  opted_out_at: string;
}

/**
 * Response for GET /api/sessions/{id}/interpretations/opt_out_summary.
 *
 * Per F-22: after a session has opted out of interpretation review, the
 * composer-LLM continues to auto-bake interpretations (now flagged as
 * `auto_interpreted_opt_out`) and may also write
 * `auto_interpreted_no_surfaces` rows when the rate cap is exhausted.
 * This route lets a user retroactively review every auto-baked
 * interpretation produced during the opted-out portion of the session,
 * closing the audit gap of "click opt-out once, dozens of
 * auto-interpretations accumulate invisibly."
 *
 * Returns rows of both `auto_interpreted_opt_out` and
 * `auto_interpreted_no_surfaces` interpretation_source, ordered by
 * created_at.  `user_approved` rows are excluded; the standard list
 * route is the right surface for those.
 *
 * Envelope shape matches `ListInterpretationEventsResponse` so the two
 * list routes have consistent wire ergonomics.
 */
export interface OptOutSummaryResponse {
  events: InterpretationEvent[];
}
