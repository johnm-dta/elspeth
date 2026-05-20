// ============================================================================
// ELSPETH Frontend Type Definitions
//
// Hand-written types mirroring backend Pydantic schemas. These are the
// contract between frontend and backend. When openapi-typescript generation
// is available, these can be replaced with imports from the generated file.
// ============================================================================

import type { AuditCharacteristicFlag } from "../components/catalog/auditCharacteristics";
import type { FailedTurn } from "./recovery";

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
 * Full user profile returned by GET /api/auth/me.
 */
export interface UserProfile {
  user_id: string;
  username: string;
  display_name: string | null;
  email: string | null;
  groups: string[];
}

// ── Sessions ────────────────────────────────────────────────────────────────

/** Session summary for session switcher listings. */
export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  forked_from_session_id?: string;
  forked_from_message_id?: string;
  /**
   * True for run-bearing sessions that the user has archived. Unrun sessions
   * are physically deleted; sessions with durable history are hidden instead.
   */
  archived?: boolean;
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
  role: "user" | "assistant" | "system" | "tool" | "audit";
  content: string;
  raw_content?: string | null;
  tool_calls: ToolCall[] | null;
  created_at: string;
  local_status?: "pending" | "failed";
  local_error?: string;
  composition_state_id?: string | null;
  tool_call_id?: string | null;
  parent_assistant_id?: string | null;
  sequence_no?: number | null;
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

// ── Composer Proposal Lifecycle ────────────────────────────────────────────

export type ComposerTrustMode = "explicit_approve" | "auto_commit";
export type ComposerDensityDefault = "high" | "medium" | "low";
export type ProposalLifecycleStatus = "pending" | "committed" | "rejected";

export interface ComposerPreferences {
  session_id: string;
  trust_mode: ComposerTrustMode;
  density_default: ComposerDensityDefault;
  interpretation_review_disabled: boolean;
  updated_at: string;
}

export interface CompositionProposal {
  id: string;
  session_id: string;
  tool_call_id: string;
  tool_name: string;
  status: ProposalLifecycleStatus;
  summary: string;
  rationale: string;
  affects: string[];
  arguments_redacted_json: Record<string, unknown>;
  base_state_id: string | null;
  committed_state_id: string | null;
  audit_event_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface MessageWithStateResponse {
  message: ChatMessage;
  state: CompositionState | null;
  proposals: CompositionProposal[];
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

/** Plugin summary from the catalog listing endpoints.
 *
 * Phase 7A added reference-content fields populated by plugin authors.
 * Unfilled plugins return `null` / empty values; the catalog drawer
 * renders a "see the technical description" fallback for them.
 *
 * ``audit_characteristics`` is typed as the closed vocabulary union to
 * mirror the Python ``DerivedAuditCharacteristics = tuple[AuditCharacteristic, ...]``
 * type. The wire is structurally JSON ``string[]``; the union tightens
 * the in-TS surface so internal call sites cannot construct a
 * PluginSummary with a typo'd flag. Forward compatibility for unknown
 * wire values is preserved by the lookup boundary at
 * ``lookupAuditCharacteristic(flag: string)``, which still accepts
 * ``string`` and returns ``null`` (rendering the grey "unknown" chip)
 * for a flag outside the union.
 */
export interface PluginSummary {
  name: string;
  plugin_type: "source" | "transform" | "sink";
  description: string;
  config_fields: { name: string; type: string; required: boolean; description: string; default: unknown }[];

  // Phase 7B reference-content fields
  usage_when_to_use: string | null;
  usage_when_not_to_use: string | null;
  example_use: string | null;
  capability_tags: string[];
  audit_characteristics: AuditCharacteristicFlag[];
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
export const VALIDATION_CHECK_OUTCOME_CODE_VALUES = [
  "secret_refs.no_refs",
  "secret_refs.resolved",
  "secret_refs.unresolved",
  "secret_refs.skipped_no_service",
  "validation.skipped_after_failure",
] as const;

export type ValidationCheckOutcomeCode = (typeof VALIDATION_CHECK_OUTCOME_CODE_VALUES)[number];

export interface ValidationCheck {
  name: string;
  passed: boolean;
  detail: string;
  affected_nodes: string[];
  outcome_code: ValidationCheckOutcomeCode | null;
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
  error_code?: string | null;
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

export interface ValidationReadinessBlocker {
  code: string;
  component_id: string | null;
  component_type: string | null;
  detail: string;
}

export interface ValidationReadiness {
  authoring_valid: boolean;
  execution_ready: boolean;
  completion_ready: boolean;
  blockers: ValidationReadinessBlocker[];
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
  readiness: ValidationReadiness;
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
export interface DiscardStageSummary {
  stage: "source_validation" | "transform_validation" | "sink_discard";
  node_id: string | null;
  count: number;
}

export interface DiscardSummary {
  total: number;
  validation_errors: number;
  transform_errors: number;
  sink_discards: number;
  stages?: DiscardStageSummary[];
}

/**
 * Single source of truth for the run-status taxonomy.
 *
 * Mirrors the backend `SessionRunStatus` Literal at
 * `src/elspeth/web/sessions/protocol.py:31-32` and the corresponding
 * `SESSION_RUN_STATUS_VALUES` / `SESSION_TERMINAL_RUN_STATUS_VALUES` frozensets.
 *
 * Phase 2.2 (elspeth-0de989c56d): the four-value operator-completion taxonomy
 * (completed / completed_with_failures / failed / empty) plus cancelled gives
 * the five-value terminal set; pending and running are non-terminal.
 *
 * Use `Record<RunStatus, T>` for badge/colour/label maps so adding a new
 * status to the const tuple becomes a compile error in every consumer.
 */
export const RUN_STATUS_VALUES = [
  "pending",
  "running",
  "completed",
  "completed_with_failures",
  "failed",
  "empty",
  "cancelled",
] as const;
export type RunStatus = (typeof RUN_STATUS_VALUES)[number];

export const TERMINAL_RUN_STATUS_VALUES = [
  "completed",
  "completed_with_failures",
  "failed",
  "empty",
  "cancelled",
] as const;
export type TerminalRunStatus = (typeof TERMINAL_RUN_STATUS_VALUES)[number];

export function isTerminalRunStatus(
  status: RunStatus,
): status is TerminalRunStatus {
  return (TERMINAL_RUN_STATUS_VALUES as readonly RunStatus[]).includes(status);
}

// Compile-time assertion: TerminalRunStatus must be a subset of RunStatus.
// If a future widening adds a terminal value to TERMINAL_RUN_STATUS_VALUES
// without adding it to RUN_STATUS_VALUES, this fails to compile.
type _AssertTerminalSubset = TerminalRunStatus extends RunStatus ? true : never;
const _terminalSubsetCheck: _AssertTerminalSubset = true;
void _terminalSubsetCheck;

export interface RunAccountingSource {
  rows_processed: number;
}

export interface RunAccountingTokens {
  emitted: number;
  terminal: number;
  succeeded: number;
  failed: number;
  structural: number;
  pending: number;
}

export interface RunAccountingRouting {
  routed_success: number;
  routed_failure: number;
  quarantined: number;
  discarded: number;
}

export interface RunAccountingIntegrity {
  closure: "closed" | "open" | "unknown";
  missing_terminal_outcomes: number;
  duplicate_terminal_outcomes: number;
}

export interface RunAccounting {
  source: RunAccountingSource;
  sources: Record<string, RunAccountingSource>;
  tokens: RunAccountingTokens;
  routing: RunAccountingRouting;
  integrity: RunAccountingIntegrity;
}

/** An execution run.
 *
 * Mirrors ``RunStatusResponse`` / ``RunResultsResponse`` at
 * ``web/execution/schemas.py``. Accounting is explicit about its unit of
 * account: source rows are ingestion records, while token counts are emitted
 * materialized work. Non-terminal and early-failed runs may not have a
 * Landscape accounting projection yet.
 */
export interface Run {
  id: string;
  session_id: string;
  status: RunStatus;
  cancel_requested?: boolean;
  accounting: RunAccounting | null;
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
  source_rows_processed: number;
  tokens_succeeded: number;
  tokens_failed: number;
  tokens_quarantined: number;
  tokens_routed_success: number;
  tokens_routed_failure: number;
}

export interface RunEventError {
  message: string;
  node_id: string | null;
  row_id: string | null;
}

export interface RunEventCompleted {
  /**
   * Phase 2.2 (elspeth-0de989c56d): backend-supplied operator-completion
   * status. The SSE `event_type="completed"` envelope covers all three
   * operator-completion values; the frontend MUST consume `data.status`
   * verbatim rather than re-deriving from accounting counts (that would duplicate
   * the L0 `failure_indicator` predicate and create dual-source-of-truth
   * drift). Mirrors `CompletedData.status` at `web/execution/schemas.py`.
  */
  status: "completed" | "completed_with_failures" | "empty";
  accounting: RunAccounting;
  landscape_run_id: string;
}

export interface RunEventCancelled {
  status: "cancelled";
  source_rows_processed: number;
  tokens_succeeded: number;
  tokens_failed: number;
  tokens_quarantined: number;
  tokens_routed_success: number;
  tokens_routed_failure: number;
}

export interface RunEventFailed {
  status: "failed";
  detail: string;
  node_id: string | null;
}

/** Live progress state derived from RunEvents.
 *
 * Mirrors the explicit source/token counters from ``RunEventProgress`` plus
 * the recent-error ring buffer. Completed events additionally attach the
 * closed Landscape accounting projection.
 */
export interface RunProgress {
  source_rows_processed: number;
  tokens_succeeded: number;
  tokens_failed: number;
  tokens_quarantined: number;
  tokens_routed_success: number;
  tokens_routed_failure: number;
  cancel_requested?: boolean;
  accounting: RunAccounting | null;
  recent_errors: RunEventError[];
  status: RunStatus;
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

// Focused pointer to the operation that caused a run to fail. The backend
// surfaces the most recent failed operation here so the UI does not have to
// scan the (paged) operations array to find the cause. ``error_message`` is
// the full chain text from Landscape's ``operations.error_message`` — wrapper
// error plus cause(s) plus any truncated HTTP response body captured at the
// LLM-client wrap site.
export interface RunDiagnosticFailureDetail {
  operation_id: string;
  node_id: string;
  operation_type: string;
  error_message: string;
  failed_at: string;
}

export interface RunDiagnostics {
  run_id: string;
  landscape_run_id: string;
  run_status: Run["status"];
  cancel_requested: boolean;
  summary: RunDiagnosticSummary;
  tokens: RunDiagnosticToken[];
  operations: RunDiagnosticOperation[];
  artifacts: RunDiagnosticArtifact[];
  failure_detail: RunDiagnosticFailureDetail | null;
}

export interface CancelRunResponse {
  status: RunStatus;
  cancel_requested: boolean;
}

// ── Run outputs (full audit-evidence manifest + bounded preview) ──────────────
//
// Distinct from RunDiagnosticArtifact (capped operator-UI projection):
// RunOutputArtifact is the per-run unbounded list returned by
// GET /api/runs/{rid}/outputs. The `downloadable` flag is server-computed
// from "is the file in the sink-allowlist AND on disk now?" — used by the
// UI to suppress download buttons that would otherwise 4xx on click.

export interface RunOutputArtifact {
  artifact_id: string;
  sink_node_id: string;
  artifact_type: string;
  path_or_uri: string;
  content_hash: string;
  size_bytes: number;
  created_at: string;
  exists_now: boolean;
  downloadable: boolean;
}

export interface RunOutputsResponse {
  run_id: string;
  landscape_run_id: string;
  artifacts: RunOutputArtifact[];
}

// Renderer hints from the preview endpoint. Mirrors PreviewContentType
// in elspeth/web/execution/preview.py — keep in sync when extending.
export const PREVIEW_CONTENT_TYPE_VALUES = [
  "text",
  "csv",
  "jsonl",
  "json",
  "binary",
] as const;
export type PreviewContentType = (typeof PREVIEW_CONTENT_TYPE_VALUES)[number];

export interface RunOutputArtifactPreview {
  artifact_id: string;
  content_type: PreviewContentType;
  preview_text: string;
  truncated: boolean;
  total_size_bytes: number;
  row_count_preview: number | null;
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

// ── Execution Fanout Guard ─────────────────────────────────────────────────

export interface ExecutionFanoutRisk {
  node_id: string;
  provider: string;
  model: string | null;
  credential_ref: string | null;
  estimated_provider_calls: number | null;
  provider_calls_per_row: number;
  upstream_fanout: string[];
  risk_level: "medium" | "high";
  message: string;
}

export interface ExecutionFanoutGuard {
  ack_token: string;
  risk_level: "medium" | "high";
  summary: string;
  risks: ExecutionFanoutRisk[];
}

export interface ExecutionFanoutAck {
  token: string;
  accepted: true;
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
  partial_state?: CompositionState | null;
  failed_turn?: FailedTurn | null;
  partial_state_save_failed?: boolean;
  partial_state_save_error?: string | null;
  fanout_guard?: ExecutionFanoutGuard;
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

/**
 * Wire form of the closed `creation_modality` enum (Phase 5a Task 2.5).
 * Snake_case mirrors the SQL CHECK constraint exactly. The frontend
 * `InlineSourceSummary.provenance` discriminant uses hyphenated forms;
 * the single translation point is the `fetchBlob` response adapter in
 * `api/client.ts` (`toInlineSourceProvenance`).
 */
export type BlobCreationModalityWire =
  | "verbatim"
  | "llm_generated"
  | "disambiguated"
  | "llm_generated_then_amended";

/**
 * Display form of the creation modality used by
 * `InlineSourceSummary.provenance`. Hyphenated form; see
 * `BlobCreationModalityWire` for the snake_case wire form. The adapter
 * `toInlineSourceProvenance` in `api/client.ts` is the only place wire
 * → display translation is performed.
 */
export type InlineSourceProvenance =
  | "verbatim"
  | "llm-generated"
  | "disambiguated"
  | "llm-generated-then-amended";

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
  // Inline-blob provenance. The wire form is snake_case; the frontend's
  // `InlineSourceSummary.provenance` field is hyphenated. Translation
  // lives in `api/client.ts` only.
  creation_modality: BlobCreationModalityWire;
  created_from_message_id: string | null;
  creating_model_identifier: string | null;
  creating_model_version: string | null;
  creating_provider: string | null;
  creating_composer_skill_hash: string | null;
  creating_arguments_hash: string | null;
}

/**
 * User-facing file category for the blob manager folder view.
 * Derived from the blob's mime_type and created_by fields.
 */
export type BlobCategory = "source" | "sink" | "other";

// ── Secret References ───────────────────────────────────────────────────────

/** Secret inventory item — browser-safe metadata, never contains values. */
export type SecretUnavailabilityReason =
  | "fingerprint_resolver_not_configured"
  | "env_var_not_set"
  | "value_decryption_failed";

export interface SecretInventoryItem {
  name: string;
  scope: "user" | "server" | "org";
  available: boolean;
  source_kind: string;
  reason: SecretUnavailabilityReason | null;
}

// ── Audit Readiness Panel (Phase 2) ────────────────────────────────────────
//
// These types mirror the Pydantic models in
// src/elspeth/web/audit_readiness/models.py (Phase 2A). If a backend literal
// is added, the union here must be widened in the same commit and the
// AuditReadinessPanel's row-renderer switch must add a case — the exhaustive
// `never` default arm fails the build otherwise.

export type ReadinessRowId =
  | "validation"
  | "plugin_trust"
  | "provenance"
  | "retention"
  | "llm_interpretations"
  | "secrets";

export type ReadinessStatus = "ok" | "warning" | "error" | "not_applicable";

export interface ReadinessRow {
  id: ReadinessRowId;
  label: string;
  status: ReadinessStatus;
  summary: string;
  detail: string | null;
  /**
   * IDs of components the row implicates. May reference node ids, source,
   * or sink names. The frontend's click handler resolves these against
   * CompositionState.nodes for jump-to-component navigation; non-node ids
   * fall through to a no-op (no error).
   */
  component_ids: readonly string[];
}

export interface AuditReadinessSnapshot {
  session_id: string;
  composition_version: number;
  checked_at: string;
  rows: readonly ReadinessRow[];
  validation_result: ValidationResult;
}

export interface AuditReadinessExplain {
  session_id: string;
  composition_version: number;
  narrative: string;
}

/**
 * Frontend-derived projection of an inline-blob source attached to the
 * current composition state. Computed from compositionState.source +
 * blob metadata. Never persisted; recomputed on each composition mutation.
 */
export interface InlineSourceSummary {
  blobId: string;
  filename: string;
  mimeType: string;
  /** Truncated content excerpt for display; never the full payload. */
  contentPreview: string;
  /** Best-effort row count from the parsed source; null if unparseable. */
  rowCount: number | null;
  /**
   * SHA-256 of the raw inline content (from session blob metadata).
   *
   * NON-NULLABLE BY CONTRACT. Every persisted blob carries a hash — that's
   * a Tier-1 audit-trail invariant on our data (CLAUDE.md "Auditability
   * Standard": hashes survive payload deletion, integrity is always
   * verifiable). The inline-source projection MUST throw, not
   * coerce, when the wire returns a null or empty hash: silently
   * substituting an empty string into the rendered audit-info pane
   * gives an auditor a value the system never asserted, which is exactly
   * the fabrication CLAUDE.md forbids. The throw lives in
   * `projectInlineSourceSummary` — keep it there.
   */
  contentHash: string;
  /**
   * How this inline source's content was produced. Projected from the
   * server-recorded `creation_modality` column via the `fetchBlob`
   * response adapter in `client.ts`.
   *
   * - "verbatim"                   — user typed the content directly.
   * - "llm-generated"              — LLM generated rows; user confirmed.
   * - "disambiguated"              — LLM interpreted ambiguous input; user confirmed.
   * - "llm-generated-then-amended" — LLM generated rows, user amended via
   *                                  "Edit the list" (F-4). Drives the Edit
   *                                  button visibility alongside "llm-generated".
   *
   * The frontend uses hyphenated forms; the server uses snake_case
   * (`llm_generated`, `llm_generated_then_amended`). The adapter in
   * `client.ts` is the single translation point.
   */
  provenance: InlineSourceProvenance;
}
