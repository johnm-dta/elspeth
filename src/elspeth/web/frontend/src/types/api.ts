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

export interface UserComposerPreferencesPayload {
  default_mode: ComposerMode;
  banner_dismissed_at: string | null;
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
}

// ── Shareable reviews (Phase 6A) ───────────────────────────────────────────
//
// Wire shapes mirroring the strict Pydantic models at
// src/elspeth/web/shareable_reviews/models.py. The frontend treats these as
// Tier-1 inbound (typed parse; shape drift crashes). The audit_readiness
// field on SharedInspectResponse is the Phase 2 AuditReadinessSnapshot
// reused verbatim — the shared inspect view shows the same six-row panel
// the owner saw at mark-time.

import type { AuditReadinessSnapshot as _AuditReadinessSnapshot } from "./index";

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

/** Response from GET /api/sessions/shared/{token}. */
export interface SharedInspectResponse {
  session_id: string;
  state_id: string;
  pipeline_metadata: Record<string, unknown>;
  composition_snapshot: Record<string, unknown>;
  yaml: string;
  audit_readiness: _AuditReadinessSnapshot;
  created_by_user_id: string;
  created_at: string;
  expires_at: string;
}
