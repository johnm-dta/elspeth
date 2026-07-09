//
// Re-export all types from the hand-written definitions.
// When openapi-typescript generation is available, change this to:
//   export type { ... } from "./api.generated";
//
export type {
  UserProfile,
  Session,
  ChatMessage,
  ToolCall,
  NodeSpec,
  EdgeSpec,
  CompositionState,
  CompositionStateVersion,
  PipelineMetadata,
  ComposerDensityDefault,
  ComposerPreferences,
  ComposerProgressSnapshot,
  ComposerProgressPhase,
  ComposerTrustMode,
  CompositionProposal,
  MessageWithStateResponse,
  ProposalLifecycleStatus,
  PluginSummary,
  ValidationCheck,
  ValidationResult,
  ValidationError,
  ValidationWarning,
  PipelineStatus,
  Run,
  RunEvent,
  RunEventProgress,
  RunEventError,
  RunEventCompleted,
  RunEventCancelled,
  RunProgress,
  ApiError,
  BlobMetadata,
  BlobCategory,
  SecretInventoryItem,
  ReadinessRowId,
  ReadinessStatus,
  ReadinessRow,
  AuditReadinessSnapshot,
  AuditReadinessExplain,
  InlineSourceSummary,
  InlineSourceProvenance,
  BlobCreationModalityWire,
} from "./index";
export type {
  ComposerRecoveryErrorFields,
  ComposerRecoveryError,
  FailedTurn,
  RecoveryTranscriptRow,
} from "./recovery";

// ── Interpretation events (Phase 5b) ───────────────────────────────────────
// Wire shapes for the LLM-interpretation review surface.  See
// ./interpretation.ts for the full per-type docstrings and the closed-enum
// governance.  Re-exported here so consumer code can `import type { ... }
// from "@/types/api"` uniformly with the other session-surface types.

export type {
  InterpretationChoice,
  InterpretationSource,
  InterpretationEvent,
  ListInterpretationEventsResponse,
  InterpretationResolveRequest,
  InterpretationResolveResponse,
  InterpretationOptOutResponse,
  OptOutSummaryResponse,
} from "./interpretation";

// ── Account-level composer preferences (Phase 1B) ──────────────────────────
// Account-scoped row; distinct from the per-session `ComposerPreferences`
// (trust_mode / density_default) re-exported above. The name disambiguator is
// the `User*` prefix — see plan 13 Task 1 review history for the rationale.

export type ComposerMode = "guided" | "freeform";

// First-run tutorial resume stage (elspeth-918f4434b3). Mirrors the
// backend `TutorialStage` Literal (preferences/models.py) — the frontend
// TutorialStep union minus "welcome" (never persisted; null is the
// no-in-progress-tutorial state).
export type PersistedTutorialStage = "guided" | "run" | "audit" | "graduation";

export interface UserComposerPreferencesPayload {
  default_mode: ComposerMode;
  banner_dismissed_at: string | null;
  tutorial_completed_at: string | null;
  // In-progress tutorial resume state; all four null when no tutorial is
  // in progress. run_id/source_data_hash are recorded once the tutorial
  // run completes so the audit step resumes without re-execution.
  tutorial_stage: PersistedTutorialStage | null;
  tutorial_session_id: string | null;
  tutorial_run_id: string | null;
  tutorial_source_data_hash: string | null;
  // Nullable to mirror the backend Panel-U1 contract: when no DB row
  // exists for the user, the GET response represents the in-server
  // default and updated_at is null (no write event has occurred to
  // associate a timestamp with). Every other response carries the real
  // write time. See src/elspeth/web/preferences/models.py.
  updated_at: string | null;
}

export interface UpdateUserComposerPreferencesPayload {
  default_mode?: ComposerMode;
  banner_dismissed_at?: string | null;
  tutorial_completed_at?: string | null;
  // Absent = unchanged; explicit null = clear. Setting (or clearing)
  // tutorial_completed_at also clears any resume field not supplied in
  // the same PATCH (completion-clears-progress, see backend service).
  tutorial_stage?: PersistedTutorialStage | null;
  tutorial_session_id?: string | null;
  tutorial_run_id?: string | null;
  tutorial_source_data_hash?: string | null;
}

// ── First-run tutorial (Phase 4) ───────────────────────────────────────────

export interface TutorialRunRequest {
  session_id: string;
}

/** Response of POST /api/tutorial/cancel. Idempotent best-effort cancel:
 * `cancelled: false` means there was no active run left to stop. */
export interface TutorialCancelResponse {
  cancelled: boolean;
}

export interface TutorialRunOutput {
  rows: Array<Record<string, unknown>>;
  source_data_hash: string;
  // Rows the source discarded at the boundary (on_validation_failure="discard"):
  // recorded for audit but absent from `rows`. Surfaced so dropped rows are visible.
  discarded_row_count: number;
}

export interface TutorialRunResponse {
  run_id: string;
  output: TutorialRunOutput;
}

export interface RunAuditStoryResponse {
  run_id: string;
  session_id: string;
  llm_call_count: number;
  source_data_hash: string;
  started_at: string;
  plugin_versions: Record<string, string>;
  seeded_from_cache: boolean;
  cache_key: string | null;
}

export interface TutorialOrphanCleanupResponse {
  deleted_count: number;
}

// ── Shareable reviews (Phase 6A) ───────────────────────────────────────────
//
// Wire shapes mirroring the strict Pydantic models at
// src/elspeth/web/shareable_reviews/models.py. The frontend treats these as
// Tier-1 inbound (typed parse; shape drift crashes). The audit_readiness
// field on SharedInspectResponse is the Phase 2 AuditReadinessSnapshot
// reused verbatim — the shared inspect view shows the same six-row panel
// the owner saw at mark-time.

import type {
  AuditReadinessSnapshot as _AuditReadinessSnapshot,
  CompositionState as _CompositionState,
  PipelineMetadata as _PipelineMetadata,
} from "./index";

/** Response from POST /api/sessions/{session_id}/mark-ready-for-review. */
export interface MarkReadyForReviewResponse {
  token: string;
  share_url: string;
  expires_at: string;
  payload_digest: string;
}

/** Response from GET /api/sessions/{session_id}/shareable-link. */
export interface ShareableLinkResponse {
  token: string;
  share_url: string;
  expires_at: string;
  state_id: string;
  payload_digest: string;
}

/** Response from GET /api/sessions/shared/{token}.
 *
 * `pipeline_metadata` and `composition_snapshot` use the canonical
 * front-end shapes (`PipelineMetadata`, `CompositionState`) per plan
 * 19b:100-101 and the FIX-K trust-boundary tightening (Phase 6B
 * gap-analysis). The wire shapes are produced by the strict Pydantic
 * mirrors `PipelineMetadataResponse` / `CompositionStateResponse`
 * (`src/elspeth/web/shareable_reviews/models.py:97-179`).
 *
 * Wire-vs-runtime caveat for `composition_snapshot`: the backend
 * `CompositionState.to_dict()` emits `{version, metadata, sources, nodes,
 * edges, outputs}` — the
 * runtime-only fields on the `CompositionState` interface (`id`,
 * `validation_errors`, `validation_warnings`, `validation_suggestions`)
 * are NOT present on this wire payload. The shared-inspect consumers
 * (notably `GraphMiniView`) read only `.sources`, `.nodes`, `.outputs`,
 * so the absent fields are inert here; treat any downstream consumer
 * that reads `id`/`validation_*` from this surface as a bug.
 */
export interface SharedInspectResponse {
  session_id: string;
  state_id: string;
  pipeline_metadata: _PipelineMetadata;
  composition_snapshot: _CompositionState;
  yaml: string;
  audit_readiness: _AuditReadinessSnapshot;
  created_by_user_id: string;
  created_at: string;
  expires_at: string;
}
