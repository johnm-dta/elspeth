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
} from "./index";
export type {
  ComposerRecoveryErrorFields,
  ComposerRecoveryError,
  FailedTurn,
  RecoveryTranscriptRow,
} from "./recovery";

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
