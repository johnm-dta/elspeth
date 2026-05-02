// ============================================================================
// ELSPETH Frontend Type Definitions
//
// Hand-written types mirroring backend Pydantic schemas. These are the
// contract between frontend and backend. When openapi-typescript generation
// is available, these can be replaced with imports from the generated file.
// ============================================================================

// ── Auth ────────────────────────────────────────────────────────────────────

/**
 * Auth provider configuration returned by GET /api/auth/config.
 * This endpoint is unauthenticated (callable before login).
 * The response is cached in memory for the session lifetime.
 */
export interface AuthConfig {
  provider: "local" | "oidc" | "entra";
  oidc_issuer: string | null;
  oidc_client_id: string | null;
  authorization_endpoint: string | null;
}

/**
 * Minimal identity claims extracted from the JWT or session.
 * Used internally for attribution and display.
 */
export interface UserIdentity {
  user_id: string;
  username: string;
}

/**
 * Full user profile returned by GET /api/auth/me.
 * Extends UserIdentity with display-oriented fields.
 */
export interface UserProfile {
  user_id: string;
  username: string;
  display_name: string | null;
  email: string | null;
  groups: string[];
}

// ── Sessions ────────────────────────────────────────────────────────────────

/** Session summary for sidebar listing. */
export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  forked_from_session_id?: string;
  forked_from_message_id?: string;
}

// ── Messages ────────────────────────────────────────────────────────────────

/**
 * A single tool call within an assistant message.
 * Uses LiteLLM wire format as stored in the chat_messages.tool_calls column.
 * The `arguments` field is a JSON-encoded string, not a parsed object.
 */
export interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
}

/** A chat message in a session. */
export interface ChatMessage {
  id: string;
  session_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  tool_calls: ToolCall[] | null;
  created_at: string;
  local_status?: "pending" | "failed";
  local_error?: string;
  composition_state_id?: string;
}

// ── Composition State ───────────────────────────────────────────────────────

/** Source specification within a pipeline composition. */
export interface SourceSpec {
  plugin: string;
  options: Record<string, unknown>;
  on_success?: string;
  on_validation_failure?: string;
}

/**
 * A node in the pipeline composition DAG.
 * Matches backend CompositionState.to_dict() node serialization.
 */
export interface NodeSpec {
  id: string;
  node_type: "transform" | "gate" | "aggregation" | "coalesce";
  plugin: string | null;
  input: string;
  on_success: string | null;
  on_error: string | null;
  options: Record<string, unknown>;
  condition?: string | null;
  routes?: Record<string, string> | null;
  fork_to?: string[] | null;
  branches?: string[] | null;
  policy?: string | null;
  merge?: string | null;
}

/** An edge connecting two nodes in the DAG. */
export interface EdgeSpec {
  id: string;
  from_node: string;
  to_node: string;
  edge_type: "on_success" | "on_error" | "route_true" | "route_false" | "fork";
  label: string | null;
}

/** Output/sink specification within a pipeline composition. */
export interface OutputSpec {
  name: string;
  plugin: string;
  options: Record<string, unknown>;
}

/** Pipeline-level metadata attached to a composition. */
export interface PipelineMetadata {
  name: string | null;
  description: string | null;
}

/**
 * The full pipeline composition state.
 *
 * This is the central data structure that flows through the system:
 * - Created/updated by the composer tool-use loop
 * - Persisted via SessionService
 * - Validated by the validation pipeline
 * - Rendered by the frontend inspector panel
 * - Converted to YAML for execution
 */
/**
 * A structured validation entry with component attribution.
 * Matches backend ValidationEntryResponse schema.
 */
export interface ValidationEntryDTO {
  component: string;
  message: string;
  severity: string;
}

export interface CompositionState {
  id: string;
  version: number;
  source: SourceSpec | null;
  nodes: NodeSpec[];
  edges: EdgeSpec[];
  outputs: OutputSpec[];
  metadata: PipelineMetadata;
  validation_errors?: string[];
  validation_warnings?: ValidationEntryDTO[];
  validation_suggestions?: ValidationEntryDTO[];
}

/** A version history entry for CompositionState. */
export interface CompositionStateVersion {
  id: string;
  version: number;
  created_at: string;
  node_count: number;
}

// ── Composer Progress ──────────────────────────────────────────────────────

export type ComposerProgressPhase =
  | "idle"
  | "starting"
  | "calling_model"
  | "using_tools"
  | "validating"
  | "saving"
  | "complete"
  | "failed"
  | "cancelled";

/**
 * Stable machine-readable reason code for composer progress events.
 *
 * Mirrors `ComposerProgressReason` in src/elspeth/web/composer/progress.py.
 * This is the public taxonomy the SPA should branch on (instead of parsing
 * `headline` text). The Python validator requires this field for any
 * `phase: "failed"` event, so the SPA can rely on it being present whenever
 * `phase === "failed"`.
 */
export type ComposerProgressReason =
  | "convergence_composition_budget"
  | "convergence_discovery_budget"
  | "convergence_wall_clock_timeout"
  | "provider_auth_failed"
  | "provider_unavailable"
  | "plugin_crash"
  | "runtime_preflight_failed"
  | "service_setup_failed"
  // Required when phase === "cancelled" — distinguishes a client disconnect
  // from a future operator-initiated cancel without parsing the headline.
  | "client_cancelled"
  | "composer_idle"
  | "composer_complete";

/**
 * Latest provider-safe composer progress snapshot for one session.
 *
 * This is a status surface, not a reasoning transcript. Text is produced from
 * visible composer lifecycle boundaries and safe tool categories only.
 */
export interface ComposerProgressSnapshot {
  session_id: string;
  request_id: string | null;
  phase: ComposerProgressPhase;
  headline: string;
  evidence: string[];
  likely_next: string | null;
  reason: ComposerProgressReason | null;
  updated_at: string;
}

// ── Plugin Catalog ──────────────────────────────────────────────────────────

/** Plugin summary from the catalog listing endpoints. */
export interface PluginSummary {
  name: string;
  plugin_type: "source" | "transform" | "sink";
  description: string;
  config_fields: { name: string; type: string; required: boolean; description: string; default: unknown }[];
}

/** Detailed plugin schema info including configuration JSON Schema. */
export interface PluginSchemaInfo {
  name: string;
  plugin_type: "source" | "transform" | "sink";
  description: string;
  json_schema: Record<string, unknown>;
}

// ── Validation ──────────────────────────────────────────────────────────────

/**
 * A single check performed during pipeline validation.
 * Represents one discrete validation step (schema compatibility,
 * route validity, source path security, etc.).
 */
export interface ValidationCheck {
  name: string;
  passed: boolean;
  detail: string;
}

/**
 * A single validation error with per-component attribution.
 * Stage 2 errors include component_id and optional suggestion,
 * unlike Stage 1 errors which are simple strings.
 */
export interface ValidationError {
  component_id: string | null;
  component_type: string | null;
  message: string;
  suggestion: string | null;
}

/**
 * A single validation warning — same shape as ValidationError but non-blocking.
 * Warnings indicate suboptimal configuration but do not prevent execution.
 */
export interface ValidationWarning {
  component_id: string | null;
  component_type: string | null;
  message: string;
  suggestion: string | null;
}

/**
 * Per-edge semantic-contract result.
 *
 * Populated by /validate when the semantic_contracts check runs.
 * Mirrors the backend SemanticEdgeContractResponse Pydantic model
 * (web/execution/schemas.py) and the MCP _SemanticEdgeContractPayload
 * (composer_mcp/server.py) so all three surfaces carry identical shapes.
 *
 * - outcome=satisfied: producer facts match consumer requirement
 * - outcome=conflict:  producer facts violate consumer requirement
 * - outcome=unknown:   producer declared no facts for that field, or
 *                      facts contained an UNKNOWN dimension; under
 *                      consumer unknown_policy=FAIL this is treated as
 *                      a blocking error.
 */
export interface SemanticEdgeContract {
  from_id: string;
  to_id: string;
  consumer_plugin: string;
  producer_plugin: string | null;
  producer_field: string;
  consumer_field: string;
  outcome: "satisfied" | "conflict" | "unknown";
  requirement_code: string;
}

/**
 * Full validation result from POST /api/sessions/{id}/validate.
 * Stage 2 validation with per-component detail.
 */
export interface ValidationResult {
  is_valid: boolean;
  summary?: string;
  checks: ValidationCheck[];
  errors: ValidationError[];
  warnings?: ValidationWarning[];
  semantic_contracts?: SemanticEdgeContract[];
}

/**
 * Derived three-state pipeline validation status.
 *
 * - "valid": no errors, no warnings — fully runnable
 * - "valid-with-warnings": runnable but has non-blocking warnings (yellow)
 * - "invalid": has blocking errors, cannot execute (red)
 * - null: not yet validated
 */
export type PipelineStatus = "valid" | "valid-with-warnings" | "invalid";

// ── Execution ───────────────────────────────────────────────────────────────

/** Counts routed to the virtual discard sink. */
export interface DiscardSummary {
  total: number;
  validation_errors: number;
  transform_errors: number;
  sink_discards: number;
}

/** An execution run. */
export interface Run {
  id: string;
  session_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  rows_processed: number;
  rows_failed: number;
  /** elspeth-5069612f3c — gate route_to_sink MOVE rows. */
  rows_routed_success: number;
  /** elspeth-5069612f3c — transform on_error DIVERT rows. */
  rows_routed_failure: number;
  error: string | null;
  started_at: string;
  finished_at: string | null;
  composition_version: number;
  discard_summary?: DiscardSummary | null;
}

/**
 * A progress event from the WebSocket.
 *
 * The backend sends a nested envelope: top-level fields identify the event,
 * and `data` carries the type-specific payload.
 *
 * Terminal semantics:
 * - "progress" -- non-terminal. Row count update; pipeline still running.
 * - "error" -- non-terminal. Per-row exception; pipeline continues processing.
 *   The frontend appends the error to the exceptions list but does NOT stop
 *   the progress view or close the WebSocket.
 * - "completed" -- terminal. Pipeline finished successfully.
 * - "cancelled" -- terminal. Pipeline was cancelled.
 * - "failed" -- terminal. Pipeline aborted due to an unrecoverable error.
 */
export interface RunEvent {
  run_id: string;
  timestamp: string;
  event_type: "progress" | "error" | "completed" | "cancelled" | "failed";
  data: RunEventProgress | RunEventError | RunEventCompleted | RunEventCancelled | RunEventFailed;
}

export interface RunEventProgress {
  rows_processed: number;
  rows_failed: number;
  /** elspeth-5069612f3c — gate route_to_sink MOVE rows. */
  rows_routed_success: number;
  /** elspeth-5069612f3c — transform on_error DIVERT rows. */
  rows_routed_failure: number;
}

export interface RunEventError {
  message: string;
  node_id: string | null;
  row_id: string | null;
}

export interface RunEventCompleted {
  rows_processed: number;
  rows_succeeded: number;
  rows_failed: number;
  rows_quarantined: number;
  /** elspeth-5069612f3c — gate route_to_sink MOVE rows. */
  rows_routed_success: number;
  /** elspeth-5069612f3c — transform on_error DIVERT rows. */
  rows_routed_failure: number;
  landscape_run_id: string;
}

export interface RunEventCancelled {
  rows_processed: number;
  rows_failed: number;
  /** elspeth-5069612f3c — gate route_to_sink MOVE rows. */
  rows_routed_success: number;
  /** elspeth-5069612f3c — transform on_error DIVERT rows. */
  rows_routed_failure: number;
}

export interface RunEventFailed {
  detail: string;
  node_id: string | null;
}

/** Live progress state derived from RunEvents. */
export interface RunProgress {
  rows_processed: number;
  rows_failed: number;
  /** elspeth-5069612f3c — gate route_to_sink MOVE rows. */
  rows_routed_success: number;
  /** elspeth-5069612f3c — transform on_error DIVERT rows. */
  rows_routed_failure: number;
  recent_errors: RunEventError[];
  status: "running" | "completed" | "cancelled" | "failed";
}

export interface RunDiagnosticNodeState {
  state_id: string;
  token_id: string;
  node_id: string;
  step_index: number;
  attempt: number;
  status: string;
  duration_ms: number | null;
  started_at: string;
  completed_at: string | null;
  error: unknown | null;
  success_reason: unknown | null;
}

export interface RunDiagnosticToken {
  token_id: string;
  row_id: string;
  row_index: number | null;
  branch_name: string | null;
  fork_group_id: string | null;
  join_group_id: string | null;
  expand_group_id: string | null;
  step_in_pipeline: number | null;
  created_at: string;
  terminal_outcome: string | null;
  states: RunDiagnosticNodeState[];
}

export interface RunDiagnosticOperation {
  operation_id: string;
  node_id: string;
  operation_type: string;
  status: string;
  duration_ms: number | null;
  started_at: string;
  completed_at: string | null;
  error_message: string | null;
}

export interface RunDiagnosticArtifact {
  artifact_id: string;
  sink_node_id: string;
  artifact_type: string;
  path_or_uri: string;
  size_bytes: number;
  created_at: string;
}

export interface RunDiagnosticSummary {
  token_count: number;
  preview_limit: number;
  preview_truncated: boolean;
  state_counts: Record<string, number>;
  operation_counts: Record<string, number>;
  latest_activity_at: string | null;
}

export interface RunDiagnostics {
  run_id: string;
  landscape_run_id: string;
  run_status: Run["status"];
  summary: RunDiagnosticSummary;
  tokens: RunDiagnosticToken[];
  operations: RunDiagnosticOperation[];
  artifacts: RunDiagnosticArtifact[];
}

export interface RunDiagnosticsWorkingView {
  headline: string;
  evidence: string[];
  meaning: string;
  next_steps: string[];
}

export interface RunDiagnosticsEvaluation {
  run_id: string;
  generated_at: string;
  explanation: string;
  working_view: RunDiagnosticsWorkingView;
}

// ── API Error Envelope ──────────────────────────────────────────────────────

/**
 * Typed API error response.
 *
 * All non-2xx responses across the entire API use this envelope:
 * - `detail`: Human-readable error message (always present)
 * - `error_type`: Machine-readable discriminator (present on domain errors,
 *   absent on generic HTTP errors)
 * - `validation_errors`: Per-component errors (present on validation failures)
 *
 * The frontend checks error_type first (if present), falls back to HTTP
 * status code, then falls back to detail text.
 */
export interface ApiError {
  status: number;
  detail: string;
  error_type?: string;
  provider_detail?: string;
  provider_status_code?: number;
  validation_errors?: ValidationError[];
}

export interface SystemStatus {
  composer_available: boolean;
  composer_model: string;
  composer_provider: string | null;
  composer_reason: string | null;
  composer_missing_keys: string[];
}

// ── Blob Manager ────────────────────────────────────────────────────────────

/** Blob metadata returned by all blob endpoints. */
export interface BlobMetadata {
  id: string;
  session_id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  content_hash: string | null;
  created_at: string;
  created_by: "user" | "assistant" | "pipeline";
  source_description: string | null;
  status: "ready" | "pending" | "error";
}

/**
 * User-facing file category for the blob manager folder view.
 * Derived from the blob's mime_type and created_by fields.
 */
export type BlobCategory = "source" | "sink" | "other";

// ── Secret References ───────────────────────────────────────────────────────

/** Secret inventory item — browser-safe metadata, never contains values. */
export interface SecretInventoryItem {
  name: string;
  scope: "user" | "server" | "org";
  available: boolean;
  source_kind: string;
}
