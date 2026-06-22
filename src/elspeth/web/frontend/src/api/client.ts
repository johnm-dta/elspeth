// ============================================================================
// ELSPETH API Client
//
// Typed fetch wrappers for all backend endpoints. Auth token injection via
// localStorage. Global 401 interceptor triggers authStore logout.
// All fetch calls use relative paths -- Vite proxy forwards to FastAPI
// in development; same-origin serving works in production.
// ============================================================================

import type {
  ApiError,
  AuthConfig,
  BlobCreationModalityWire,
  BlobMetadata,
  ChatMessage,
  ComposerPreferences,
  CompositionProposal,
  CompositionState,
  CompositionStateVersion,
  ComposerProgressSnapshot,
  ExecutionFanoutAck,
  ExecutionFanoutGuard,
  CancelRunResponse,
  InlineSourceProvenance,
  PluginSchemaInfo,
  PluginSummary,
  Run,
  RunDiagnostics,
  RunDiagnosticsEvaluation,
  RunOutputArtifactPreview,
  RunOutputsResponse,
  WebSocketTicketResponse,
  SecretInventoryItem,
  Session,
  UserProfile,
  ValidationResult,
  SystemStatus,
  MessageWithStateResponse,
} from "@/types/index";
import type {
  GetGuidedResponse,
  GuidedChatRequest,
  GuidedChatResponse,
  GuidedRespondRequest,
  GuidedRespondResponse,
} from "@/types/guided";
import type {
  InterpretationEvent,
  InterpretationOptOutResponse,
  InterpretationResolveRequest,
  InterpretationResolveResponse,
  ListInterpretationEventsResponse,
  OptOutSummaryResponse,
} from "@/types/interpretation";
import type { RecoveryTranscriptRow } from "@/types/recovery";
import type {
  RunAuditStoryResponse,
  TutorialOrphanCleanupResponse,
  TutorialRunRequest,
  TutorialRunResponse,
  UserComposerPreferencesPayload,
  UpdateUserComposerPreferencesPayload,
} from "@/types/api";

// ── Token Management ────────────────────────────────────────────────────────

const TOKEN_KEY = "auth_token";

function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

/**
 * Build headers with auth token injection and optional content type.
 * Every authenticated request includes Authorization: Bearer {token}.
 */
export function authHeaders(contentType?: string): HeadersInit {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  if (contentType) {
    headers["Content-Type"] = contentType;
  }
  return headers;
}

// ── Response Parsing ────────────────────────────────────────────────────────

function ownField(source: unknown, field: string): unknown {
  if (typeof source !== "object" || source === null) {
    return undefined;
  }
  if (!Object.prototype.hasOwnProperty.call(source, field)) {
    return undefined;
  }
  return (source as Record<string, unknown>)[field];
}

function firstDefined<T>(primary: T | undefined, secondary: T | undefined): T | undefined {
  return primary !== undefined ? primary : secondary;
}

/**
 * Parse a response. Throws ApiError for non-2xx status codes.
 *
 * Includes a global 401 interceptor: any API call returning 401 triggers
 * authStore.logout() to handle token expiry mid-session without requiring
 * each caller to check for auth failures. Callers that intentionally handle
 * a non-session capability boundary may opt out of the logout side effect.
 *
 * Error envelope handling priority:
 *   1. error_type (if present in response body) -- most specific
 *   2. HTTP status code -- structural fallback
 *   3. detail text -- human-readable description
 */
interface ParseResponseOptions {
  logoutOnUnauthorized?: boolean;
}

export async function parseResponse<T>(
  response: Response,
  options: ParseResponseOptions = {},
): Promise<T> {
  if (!response.ok) {
    // Global 401 interceptor -- trigger logout on any auth failure.
    // Dynamic import avoids circular dependency at module load time
    // (authStore imports from client, client imports authStore for logout).
    //
    // Skip the logout call when the store already shows no token. Otherwise a
    // 401 from an unauthenticated request (e.g., a pre-login probe) calls
    // logout() against an already-empty session — harmless on its own, but if
    // the response arrives AFTER a successful login completes, it would wipe
    // the freshly-acquired token. Guarding on token!==null defuses that race
    // without changing the legitimate "token expired mid-session" path.
    if (response.status === 401 && options.logoutOnUnauthorized !== false) {
      const { useAuthStore } = await import("@/stores/authStore");
      if (useAuthStore.getState().token !== null) {
        await useAuthStore.getState().logout();
      }
    }

    // Parse the error envelope. All backend errors use `detail` (not
    // `message`) as the human-readable field, matching FastAPI's default
    // HTTPException format.
    let detail = response.statusText;
    let errorType: string | undefined;
    let providerDetail: string | undefined;
    let providerStatusCode: number | undefined;
    let fanoutGuard: ExecutionFanoutGuard | undefined;
    let validationErrors: ApiError["validation_errors"];
    let partialState: ApiError["partial_state"];
    let failedTurn: ApiError["failed_turn"];
    let partialStateSaveFailed: ApiError["partial_state_save_failed"];
    let partialStateSaveError: ApiError["partial_state_save_error"];
    try {
      const body = await response.json();
      const nestedDetail =
        typeof body.detail === "object" && body.detail !== null
          ? body.detail
          : null;

      errorType =
        typeof body.error_type === "string"
          ? body.error_type
          : typeof nestedDetail?.error_type === "string"
            ? nestedDetail.error_type
            : typeof nestedDetail?.code === "string"
              ? nestedDetail.code
              : undefined;

      if (typeof nestedDetail?.detail === "string") {
        detail = nestedDetail.detail;
      } else if (typeof body.detail === "string") {
        detail = body.detail;
      }

      providerDetail =
        typeof body.provider_detail === "string"
          ? body.provider_detail
          : typeof nestedDetail?.provider_detail === "string"
            ? nestedDetail.provider_detail
            : undefined;

      const rawProviderStatusCode =
        typeof body.provider_status_code === "number"
          ? body.provider_status_code
          : typeof nestedDetail?.provider_status_code === "number"
            ? nestedDetail.provider_status_code
            : undefined;
      providerStatusCode = Number.isInteger(rawProviderStatusCode)
        ? rawProviderStatusCode
        : undefined;

      fanoutGuard =
        body.fanout_guard ?? nestedDetail?.fanout_guard;

      validationErrors =
        body.validation_errors ?? nestedDetail?.validation_errors;

      const rawPartialState =
        firstDefined(
          ownField(body, "partial_state"),
          ownField(nestedDetail, "partial_state"),
        );
      partialState = rawPartialState as ApiError["partial_state"];

      const rawFailedTurn =
        firstDefined(
          ownField(body, "failed_turn"),
          ownField(nestedDetail, "failed_turn"),
        );
      failedTurn = rawFailedTurn as ApiError["failed_turn"];

      const rawPartialStateSaveFailed =
        firstDefined(
          ownField(body, "partial_state_save_failed"),
          ownField(nestedDetail, "partial_state_save_failed"),
        );
      partialStateSaveFailed =
        typeof rawPartialStateSaveFailed === "boolean"
          ? rawPartialStateSaveFailed
          : undefined;

      const rawPartialStateSaveError =
        firstDefined(
          ownField(body, "partial_state_save_error"),
          ownField(nestedDetail, "partial_state_save_error"),
        );
      partialStateSaveError =
        typeof rawPartialStateSaveError === "string" ||
        rawPartialStateSaveError === null
          ? rawPartialStateSaveError
          : undefined;
    } catch {
      // Response body wasn't JSON -- use statusText as detail fallback
    }

    const error: ApiError = {
      status: response.status,
      detail,
      error_type: errorType,
      partial_state: partialState,
      failed_turn: failedTurn,
      partial_state_save_failed: partialStateSaveFailed,
      partial_state_save_error: partialStateSaveError,
      fanout_guard: fanoutGuard,
      provider_detail: providerDetail,
      provider_status_code: providerStatusCode,
      validation_errors: validationErrors,
    };
    throw error;
  }

  return response.json() as Promise<T>;
}

// ── Auth ────────────────────────────────────────────────────────────────────

/**
 * Fetch auth provider configuration. Unauthenticated endpoint --
 * callable before login. Returns provider type and OIDC params.
 */
export async function fetchAuthConfig(): Promise<AuthConfig> {
  const response = await fetch("/api/auth/config");
  return parseResponse<AuthConfig>(response);
}

/**
 * Authenticate with username and password (local auth provider).
 * Returns the JWT access token.
 */
export async function login(
  username: string,
  password: string,
): Promise<{ access_token: string }> {
  const response = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return parseResponse<{ access_token: string }>(response);
}

/**
 * Refresh the current auth token. Returns a new access token.
 */
export async function refreshToken(): Promise<{ access_token: string }> {
  const response = await fetch("/api/auth/token", {
    method: "POST",
    headers: authHeaders("application/json"),
  });
  return parseResponse<{ access_token: string }>(response);
}

/**
 * Get the current user's profile. Used to validate the stored token
 * on page load and to populate the user display.
 */
export async function fetchCurrentUser(): Promise<UserProfile> {
  const response = await fetch("/api/auth/me", {
    headers: authHeaders(),
  });
  return parseResponse<UserProfile>(response);
}

/** Return boot-time system readiness for the web UX. */
export async function fetchSystemStatus(): Promise<SystemStatus> {
  const response = await fetch("/api/system/status");
  return parseResponse<SystemStatus>(response);
}

// ── Sessions ────────────────────────────────────────────────────────────────

/** List all sessions for the current user. */
export async function fetchSessions(includeArchived = true): Promise<Session[]> {
  const params = new URLSearchParams();
  if (includeArchived) {
    params.set("include_archived", "true");
  }
  const query = params.size > 0 ? `?${params.toString()}` : "";
  const response = await fetch(`/api/sessions${query}`, {
    headers: authHeaders(),
  });
  return parseResponse<Session[]>(response);
}

/** Create a new session. */
export async function createSession(): Promise<Session> {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify({ title: "New session" }),
  });
  return parseResponse<Session>(response);
}

/** Get a single session by ID. */
export async function getSession(sessionId: string): Promise<Session> {
  const response = await fetch(`/api/sessions/${sessionId}`, {
    headers: authHeaders(),
  });
  return parseResponse<Session>(response);
}

/** Update the user-visible title for a session. */
export async function renameSession(
  sessionId: string,
  title: string,
): Promise<Session> {
  const response = await fetch(`/api/sessions/${sessionId}`, {
    method: "PATCH",
    headers: authHeaders("application/json"),
    body: JSON.stringify({ title }),
  });
  return parseResponse<Session>(response);
}

/** Run the tutorial pipeline for an already-built tutorial session.
 *
 * Accepts an optional `AbortSignal` so the Turn 4 cancel button can abort
 * an in-flight LLM call. Matches the signal convention used elsewhere in
 * this module (see `fetchMessages`, `sendMessage`, etc.). */
export async function runTutorialPipeline(
  body: TutorialRunRequest,
  signal?: AbortSignal,
): Promise<TutorialRunResponse> {
  const response = await fetch("/api/tutorial/run", {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify(body),
    signal,
  });
  return parseResponse<TutorialRunResponse>(response);
}

/** Read the audit-story projection for a completed tutorial run. */
export async function getRunAuditSummary(
  sessionId: string,
  runId: string,
): Promise<RunAuditStoryResponse> {
  const response = await fetch(
    `/api/sessions/${sessionId}/runs/${runId}/audit-story`,
    { headers: authHeaders() },
  );
  return parseResponse<RunAuditStoryResponse>(response);
}

/** Clean up orphaned tutorial sessions for the authenticated user. */
export async function deleteTutorialOrphans(): Promise<TutorialOrphanCleanupResponse> {
  const response = await fetch("/api/tutorial/orphans", {
    method: "DELETE",
    headers: authHeaders(),
  });
  return parseResponse<TutorialOrphanCleanupResponse>(response);
}

/** Archive (soft-delete) a session. Backend returns 204 No Content. */
export async function archiveSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!response.ok) {
    await parseResponse<never>(response);
  }
}

// ── Messages ────────────────────────────────────────────────────────────────

/** Get all messages for a session. */
export async function fetchMessages(sessionId: string): Promise<ChatMessage[]> {
  const response = await fetch(`/api/sessions/${sessionId}/messages`, {
    headers: authHeaders(),
  });
  return parseResponse<ChatMessage[]>(response);
}

/** Fetch the audit-grade recovery transcript for a failed compose turn. */
export async function fetchRecoveryTranscript(
  sessionId: string,
  opts: { limit?: number; offset?: number } = {},
): Promise<RecoveryTranscriptRow[]> {
  const params = new URLSearchParams({
    include_tool_rows: "true",
    limit: String(opts.limit ?? 500),
    offset: String(opts.offset ?? 0),
  });
  const response = await fetch(
    `/api/sessions/${sessionId}/messages?${params}`,
    {
      headers: authHeaders(),
    },
  );
  return parseResponse<RecoveryTranscriptRow[]>(response);
}

/** Get the latest provider-safe composer progress snapshot for a session. */
export async function fetchComposerProgress(
  sessionId: string,
): Promise<ComposerProgressSnapshot> {
  const response = await fetch(
    `/api/sessions/${sessionId}/composer-progress`,
    {
      headers: authHeaders(),
    },
  );
  return parseResponse<ComposerProgressSnapshot>(response);
}

/** Get composer trust and display preferences for a session. */
export async function fetchComposerPreferences(
  sessionId: string,
): Promise<ComposerPreferences> {
  const response = await fetch(
    `/api/sessions/${sessionId}/composer/preferences`,
    {
      headers: authHeaders(),
    },
  );
  return parseResponse<ComposerPreferences>(response);
}

/** Update composer trust and display preferences for a session. */
export async function updateComposerPreferences(
  sessionId: string,
  body: Pick<ComposerPreferences, "trust_mode" | "density_default">,
): Promise<ComposerPreferences> {
  const response = await fetch(
    `/api/sessions/${sessionId}/composer/preferences`,
    {
      method: "PATCH",
      headers: authHeaders("application/json"),
      body: JSON.stringify(body),
    },
  );
  return parseResponse<ComposerPreferences>(response);
}

// ── Account-level composer preferences (Phase 1B) ──────────────────────────
// Account-scoped row keyed by user_id. Distinct from the per-session
// helpers above (trust_mode / density_default).

/** Get the user's account-level composer preferences. */
export async function fetchUserComposerPreferences(): Promise<UserComposerPreferencesPayload> {
  const response = await fetch("/api/composer-preferences", {
    headers: authHeaders(),
  });
  return parseResponse<UserComposerPreferencesPayload>(response);
}

/** Partial-update the user's account-level composer preferences. */
export async function updateUserComposerPreferences(
  payload: UpdateUserComposerPreferencesPayload,
): Promise<UserComposerPreferencesPayload> {
  const response = await fetch("/api/composer-preferences", {
    method: "PATCH",
    headers: authHeaders("application/json"),
    body: JSON.stringify(payload),
  });
  return parseResponse<UserComposerPreferencesPayload>(response);
}

/** List composition proposals for a session. */
export async function fetchCompositionProposals(
  sessionId: string,
): Promise<CompositionProposal[]> {
  const response = await fetch(
    `/api/sessions/${sessionId}/proposals?status=pending`,
    {
      headers: authHeaders(),
    },
  );
  return parseResponse<CompositionProposal[]>(response);
}

/** Accept a pending composition proposal and commit its resulting state. */
export async function acceptCompositionProposal(
  sessionId: string,
  proposalId: string,
): Promise<CompositionProposal> {
  const response = await fetch(
    `/api/sessions/${sessionId}/proposals/${proposalId}/accept`,
    {
      method: "POST",
      headers: authHeaders("application/json"),
    },
  );
  return parseResponse<CompositionProposal>(response);
}

/** Reject a pending composition proposal. */
export async function rejectCompositionProposal(
  sessionId: string,
  proposalId: string,
): Promise<CompositionProposal> {
  const response = await fetch(
    `/api/sessions/${sessionId}/proposals/${proposalId}/reject`,
    {
      method: "POST",
      headers: authHeaders("application/json"),
      body: JSON.stringify({ reason: null }),
    },
  );
  return parseResponse<CompositionProposal>(response);
}

/**
 * Send a user message. The backend runs the composer tool-use loop
 * and returns the assistant response with updated composition state.
 *
 * The response wire format uses `state` (not `compositionState`) --
 * the sessionStore maps the key on destructure.
 */
export async function sendMessage(
  sessionId: string,
  content: string,
  stateId?: string,
  signal?: AbortSignal,
): Promise<MessageWithStateResponse> {
  const body: { content: string; state_id?: string } = { content };
  if (stateId) {
    body.state_id = stateId;
  }
  const response = await fetch(`/api/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify(body),
    signal,
  });
  return parseResponse<MessageWithStateResponse>(response);
}

/** Re-run the composer without inserting a new user message.
 *  Used by the retry flow when the user message is already persisted. */
export async function recompose(
  sessionId: string,
  signal?: AbortSignal,
): Promise<MessageWithStateResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/recompose`, {
    method: "POST",
    headers: authHeaders("application/json"),
    signal,
  });
  return parseResponse<MessageWithStateResponse>(response);
}

/**
 * Fetch the current guided-session state for a session.
 *
 * Returns the active GuidedSession (step + history + terminal), the
 * server-emitted next turn payload (if any), and the current composition
 * state.  When no guided session has started for the session, the server
 * returns an in-memory initial GuidedSession and Step 1 turn without creating
 * a composition-state version.
 */
export async function getGuided(
  sessionId: string,
  signal?: AbortSignal,
): Promise<GetGuidedResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/guided`, {
    method: "GET",
    headers: authHeaders(),
    signal,
  });
  return parseResponse<GetGuidedResponse>(response);
}

/**
 * Post a user response to the active guided turn.
 *
 * Server consumes the response, advances the state machine, and returns
 * the replacement GuidedSession + next turn (or terminal state).  The
 * client is expected to atomically replace its cached guided state with
 * the response shape — no optimistic updates (spec §7.3).
 */
export async function respondGuided(
  sessionId: string,
  body: GuidedRespondRequest,
  signal?: AbortSignal,
): Promise<GuidedRespondResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/guided/respond`, {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify(body),
    signal,
  });
  return parseResponse<GuidedRespondResponse>(response);
}

/**
 * Re-enter guided mode after a deliberate user exit to freeform.
 *
 * Server clears the reversible exited_to_freeform/user_pressed_exit terminal
 * and returns the same envelope shape as GET /guided.
 */
export async function reenterGuided(
  sessionId: string,
  signal?: AbortSignal,
): Promise<GetGuidedResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/guided/reenter`, {
    method: "POST",
    headers: authHeaders(),
    signal,
  });
  return parseResponse<GetGuidedResponse>(response);
}

/**
 * Post a free-text chat message scoped to the user's current wizard step.
 *
 * Most chat is advisory: the server invokes the per-step chat solver with
 * the step-scoped skill briefing and returns the LLM's reply. Step 1 source
 * chat can also resolve a complete inline source request and return updated
 * `next_turn` / `composition_state` fields. Server-side: see
 * _guided_step_chat.solve_step_chat_with_auto_drop; on transient LLM failure
 * the server returns 200 with a synthetic "I'm unavailable" message rather
 * than failing the request.
 *
 * The `step_index` carried in the body lets the server detect that the
 * wizard has advanced under the client (returns 409) so a stale chat
 * does not arrive at the wrong step's skill briefing.
 */
export async function chatGuided(
  sessionId: string,
  body: GuidedChatRequest,
  signal?: AbortSignal,
): Promise<GuidedChatResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/guided/chat`, {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify(body),
    signal,
  });
  return parseResponse<GuidedChatResponse>(response);
}

/** Fork a session from a specific user message. */
export async function forkFromMessage(
  sessionId: string,
  fromMessageId: string,
  newMessageContent: string,
): Promise<{
  session: Session;
  messages: ChatMessage[];
  composition_state: CompositionState | null;
}> {
  const response = await fetch(`/api/sessions/${sessionId}/fork`, {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify({
      from_message_id: fromMessageId,
      new_message_content: newMessageContent,
    }),
  });
  return parseResponse<{
    session: Session;
    messages: ChatMessage[];
    composition_state: CompositionState | null;
  }>(response);
}

// ── Composition State ───────────────────────────────────────────────────────

/** Get the current composition state for a session. Returns null if none exists. */
export async function fetchCompositionState(
  sessionId: string,
): Promise<CompositionState | null> {
  const response = await fetch(`/api/sessions/${sessionId}/state`, {
    headers: authHeaders(),
  });
  if (response.status === 404) {
    return null;
  }
  return parseResponse<CompositionState>(response);
}

/** Get all composition state versions for a session. */
export async function fetchStateVersions(
  sessionId: string,
): Promise<CompositionStateVersion[]> {
  const response = await fetch(`/api/sessions/${sessionId}/state/versions`, {
    headers: authHeaders(),
  });
  return parseResponse<CompositionStateVersion[]>(response);
}

/**
 * Revert the composition state to a prior version.
 * The backend sets the selected version as active and injects a system
 * message into the session's message history.
 */
export async function revertToVersion(
  sessionId: string,
  stateId: string,
): Promise<CompositionState> {
  const response = await fetch(
    `/api/sessions/${sessionId}/state/revert`,
    {
      method: "POST",
      headers: authHeaders("application/json"),
      body: JSON.stringify({ state_id: stateId }),
    },
  );
  return parseResponse<CompositionState>(response);
}

/** Fetch the generated YAML for the current composition state. */
export async function fetchYaml(sessionId: string): Promise<{ yaml: string }> {
  const response = await fetch(`/api/sessions/${sessionId}/state/yaml`, {
    headers: authHeaders(),
  });
  return parseResponse<{ yaml: string }>(response);
}

// ── Plugin Catalog ──────────────────────────────────────────────────────────

/** List available source plugins. */
export async function listSources(): Promise<PluginSummary[]> {
  const response = await fetch("/api/catalog/sources", {
    headers: authHeaders(),
  });
  return parseResponse<PluginSummary[]>(response);
}

/** List available transform plugins. */
export async function listTransforms(): Promise<PluginSummary[]> {
  const response = await fetch("/api/catalog/transforms", {
    headers: authHeaders(),
  });
  return parseResponse<PluginSummary[]>(response);
}

/** List available sink plugins. */
export async function listSinks(): Promise<PluginSummary[]> {
  const response = await fetch("/api/catalog/sinks", {
    headers: authHeaders(),
  });
  return parseResponse<PluginSummary[]>(response);
}

/**
 * Get the full schema for a specific plugin.
 * The plugin type uses singular form ("source", "transform", "sink")
 * matching the CatalogService protocol.
 */
export async function getPluginSchema(
  pluginType: "source" | "transform" | "sink",
  pluginName: string,
): Promise<PluginSchemaInfo> {
  // REST URL uses plural path segments; the route handler translates
  // plural -> singular before calling CatalogService.
  const pluralType = `${pluginType}s`;
  const response = await fetch(
    `/api/catalog/${pluralType}/${pluginName}/schema`,
    { headers: authHeaders() },
  );
  return parseResponse<PluginSchemaInfo>(response);
}

// ── Validation & Execution ──────────────────────────────────────────────────

/**
 * Validate the current pipeline composition.
 * Stage 2 validation with per-component error detail.
 */
export async function validatePipeline(
  sessionId: string,
  stateId?: string,
): Promise<ValidationResult> {
  const params = new URLSearchParams();
  if (stateId) {
    params.set("state_id", stateId);
  }
  const query = params.size > 0 ? `?${params.toString()}` : "";
  const response = await fetch(`/api/sessions/${sessionId}/validate${query}`, {
    method: "POST",
    headers: authHeaders("application/json"),
  });
  return parseResponse<ValidationResult>(response);
}

/**
 * Execute the validated pipeline. Returns the created Run record.
 * The run executes asynchronously; progress streams via WebSocket.
 * Throws 409 if a run is already in progress for this session.
 */
export async function executePipeline(
  sessionId: string,
  fanoutAck?: ExecutionFanoutAck,
  stateId?: string,
): Promise<{ run_id: string }> {
  const params = new URLSearchParams();
  if (stateId) {
    params.set("state_id", stateId);
  }
  const query = params.size > 0 ? `?${params.toString()}` : "";
  const init: RequestInit = {
    method: "POST",
    headers: authHeaders("application/json"),
  };
  if (fanoutAck) {
    init.body = JSON.stringify({ fanout_ack_token: fanoutAck.token });
  }
  const response = await fetch(`/api/sessions/${sessionId}/execute${query}`, init);
  return parseResponse<{ run_id: string }>(response);
}

/** Get the status of a specific run. */
export async function getRunStatus(runId: string): Promise<Run> {
  const response = await fetch(`/api/runs/${runId}`, {
    headers: authHeaders(),
  });
  return parseResponse<Run>(response);
}

/** Cancel a running pipeline execution. */
export async function cancelRun(runId: string): Promise<CancelRunResponse> {
  const response = await fetch(`/api/runs/${runId}/cancel`, {
    method: "POST",
    headers: authHeaders("application/json"),
  });
  return parseResponse<CancelRunResponse>(response);
}

/** Issue a short-lived one-use ticket for the run progress WebSocket. */
export async function createRunWebSocketTicket(
  runId: string,
): Promise<WebSocketTicketResponse> {
  const response = await fetch(`/api/runs/${runId}/ws-ticket`, {
    method: "POST",
    headers: authHeaders("application/json"),
  });
  return parseResponse<WebSocketTicketResponse>(response);
}

/** Get the results/summary of a completed run. */
export async function getRunResults(
  runId: string,
): Promise<Run> {
  const response = await fetch(`/api/runs/${runId}/results`, {
    headers: authHeaders(),
  });
  return parseResponse<Run>(response);
}

/** List runs for a session. */
export async function fetchRuns(sessionId: string): Promise<Run[]> {
  const response = await fetch(`/api/sessions/${sessionId}/runs`, {
    headers: authHeaders(),
  });
  return parseResponse<Run[]>(response);
}

/** Fetch a bounded diagnostics snapshot for a run. */
export async function fetchRunDiagnostics(
  runId: string,
  limit = 50,
): Promise<RunDiagnostics> {
  const response = await fetch(
    `/api/runs/${runId}/diagnostics?limit=${encodeURIComponent(String(limit))}`,
    {
      headers: authHeaders(),
    },
  );
  return parseResponse<RunDiagnostics>(response);
}

/** Ask the configured composer LLM to explain a diagnostics snapshot. */
export async function evaluateRunDiagnostics(
  runId: string,
  limit = 50,
): Promise<RunDiagnosticsEvaluation> {
  const response = await fetch(
    `/api/runs/${runId}/diagnostics/evaluate?limit=${encodeURIComponent(String(limit))}`,
    {
      method: "POST",
      headers: authHeaders("application/json"),
    },
  );
  return parseResponse<RunDiagnosticsEvaluation>(response);
}

/**
 * Fetch the FULL audit-evidence manifest of every sink-write artefact
 * for a run. Distinct from `fetchRunDiagnostics`, whose `artifacts`
 * field is capped at 20 for operator-UI pacing — this endpoint is
 * unbounded and intended for the per-run Outputs section.
 */
export async function fetchRunOutputs(runId: string): Promise<RunOutputsResponse> {
  const response = await fetch(`/api/runs/${runId}/outputs`, {
    headers: authHeaders(),
  });
  return parseResponse<RunOutputsResponse>(response);
}

/**
 * Fetch a bounded head-of-file preview of one sink-write artefact.
 * Bounded to 256 KiB or 100 rows server-side.
 */
export async function fetchRunOutputPreview(
  runId: string,
  artifactId: string,
): Promise<RunOutputArtifactPreview> {
  const response = await fetch(
    `/api/runs/${runId}/outputs/${encodeURIComponent(artifactId)}/preview`,
    {
      headers: authHeaders(),
    },
  );
  return parseResponse<RunOutputArtifactPreview>(response);
}

/**
 * Fetch the full bytes of an artefact and return them as a Blob plus
 * the server-suggested filename (parsed from Content-Disposition).
 *
 * IMPORTANT: this can NOT be a plain `<a href download>` link. The
 * `/content` endpoint requires `Authorization: Bearer ${token}` and
 * the browser does NOT attach localStorage values to top-level
 * navigations. The same fetch-then-objectURL pattern is used for
 * `downloadBlobContent` — see `blobStore.downloadBlob` for the
 * caller-side trigger.
 */
export async function downloadRunOutputContent(
  runId: string,
  artifactId: string,
): Promise<{ data: Blob; filename: string }> {
  const response = await fetch(
    `/api/runs/${runId}/outputs/${encodeURIComponent(artifactId)}/content`,
    { headers: authHeaders() },
  );
  if (!response.ok) {
    await parseResponse<never>(response);
  }
  const disposition = response.headers.get("Content-Disposition");
  const filenameMatch = disposition?.match(/filename="(.+)"/);
  const filename = filenameMatch?.[1] ?? "download";
  const data = await response.blob();
  return { data, filename };
}

// ── Blobs ──────────────────────────────────────────────────────────────────

/** Upload a file as a session-scoped blob. */
export async function uploadBlob(
  sessionId: string,
  file: File,
): Promise<BlobMetadata> {
  const formData = new FormData();
  formData.append("file", file);

  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`/api/sessions/${sessionId}/blobs`, {
    method: "POST",
    headers,
    body: formData,
  });
  return parseResponse<BlobMetadata>(response);
}

/** List all blobs for a session. */
export async function listBlobs(sessionId: string): Promise<BlobMetadata[]> {
  const response = await fetch(`/api/sessions/${sessionId}/blobs`, {
    headers: authHeaders(),
  });
  return parseResponse<BlobMetadata[]>(response);
}

/**
 * Wire → display translation for the inline-blob creation modality
 * (Phase 5a Task 2.5). The server records the modality in snake_case
 * (matching the SQL CHECK constraint); the frontend's
 * `InlineSourceSummary.provenance` discriminant uses the hyphenated form
 * so URL-fragment routing and aria-label rendering work without a
 * second normalisation step. This adapter is the SINGLE translation
 * point — no second mapping in the store, no third one in a component.
 *
 * The mapping is exhaustive over `BlobCreationModalityWire`; a future
 * enum extension at the server forces both `BlobCreationModalityWire`
 * and `InlineSourceProvenance` to widen, and the TypeScript exhaustive
 * `never` arm here turns into a compile error rather than silently
 * dropping into the default branch. That's the discoverability
 * mechanism that lets a future change cascade through both wire and
 * display layers in the same commit.
 */
export function toInlineSourceProvenance(
  wire: BlobCreationModalityWire,
): InlineSourceProvenance {
  switch (wire) {
    case "verbatim":
      return "verbatim";
    case "llm_generated":
      return "llm-generated";
    case "disambiguated":
      return "disambiguated";
    case "llm_generated_then_amended":
      return "llm-generated-then-amended";
    default: {
      const _exhaustive: never = wire;
      throw new Error(
        `Unhandled BlobCreationModalityWire value: ${String(_exhaustive)}`,
      );
    }
  }
}

/** Get metadata for a single blob. */
export async function getBlobMetadata(
  sessionId: string,
  blobId: string,
): Promise<BlobMetadata> {
  const response = await fetch(
    `/api/sessions/${sessionId}/blobs/${blobId}`,
    { headers: authHeaders() },
  );
  return parseResponse<BlobMetadata>(response);
}

/** Download blob content as a Blob (browser Blob, not ELSPETH Blob). */
export async function downloadBlobContent(
  sessionId: string,
  blobId: string,
): Promise<{ data: Blob; filename: string }> {
  const response = await fetch(
    `/api/sessions/${sessionId}/blobs/${blobId}/content`,
    { headers: authHeaders() },
  );
  if (!response.ok) {
    await parseResponse<never>(response);
  }

  const disposition = response.headers.get("Content-Disposition");
  const filenameMatch = disposition?.match(/filename="(.+)"/);
  const filename = filenameMatch?.[1] ?? "download";
  const data = await response.blob();
  return { data, filename };
}

/** Fetch blob content as text for inline preview. */
export async function previewBlobContent(
  sessionId: string,
  blobId: string,
): Promise<string> {
  const response = await fetch(
    `/api/sessions/${sessionId}/blobs/${blobId}/content`,
    { headers: authHeaders() },
  );
  if (!response.ok) {
    await parseResponse<never>(response);
  }
  return response.text();
}

export interface BlobContentPreview {
  text: string;
  truncated: boolean;
  limit: number;
}

/** Fetch bounded blob content as text for inline UI preview. */
export async function previewBlobContentSnippet(
  sessionId: string,
  blobId: string,
  limit: number,
): Promise<BlobContentPreview> {
  const response = await fetch(
    `/api/sessions/${sessionId}/blobs/${blobId}/preview?limit=${limit}`,
    { headers: authHeaders() },
  );
  if (!response.ok) {
    await parseResponse<never>(response);
  }
  const headerLimit = Number(response.headers.get("X-Preview-Limit"));
  return {
    text: await response.text(),
    truncated: response.headers.get("X-Preview-Truncated") === "true",
    limit: Number.isFinite(headerLimit) && headerLimit > 0 ? headerLimit : limit,
  };
}

/** Delete a blob and its backing file. */
export async function deleteBlob(
  sessionId: string,
  blobId: string,
): Promise<void> {
  const response = await fetch(
    `/api/sessions/${sessionId}/blobs/${blobId}`,
    { method: "DELETE", headers: authHeaders() },
  );
  if (!response.ok) {
    await parseResponse<never>(response);
  }
}

// ── Secrets ────────────────────────────────────────────────────────────────

/** List all available secret references (no values). */
export async function listSecrets(): Promise<SecretInventoryItem[]> {
  const response = await fetch("/api/secrets", { headers: authHeaders() });
  return parseResponse<SecretInventoryItem[]>(response);
}

/** Create or update a user-scoped secret. Response never contains the value. */
export async function createSecret(
  name: string,
  value: string,
): Promise<{ name: string; scope: string; available: boolean }> {
  const response = await fetch("/api/secrets", {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify({ name, value }),
  });
  return parseResponse<{ name: string; scope: string; available: boolean }>(response);
}

/** Delete a user-scoped secret. */
export async function deleteSecret(name: string): Promise<void> {
  const response = await fetch(`/api/secrets/${encodeURIComponent(name)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!response.ok) {
    await parseResponse<never>(response);
  }
}

// ── Interpretation events (Phase 5b) ───────────────────────────────────────
//
// HTTP surface mirroring the four routes in
// src/elspeth/web/sessions/routes.py (Phase 5b Tasks 6 + 7):
//
//   GET  /api/sessions/{id}/interpretations[?status=pending|all]
//   POST /api/sessions/{id}/interpretations/{event_id}/resolve
//   POST /api/sessions/{id}/interpretations/opt_out
//   GET  /api/sessions/{id}/interpretations/opt_out_summary
//
// Error envelopes:
// - 422 (validation, e.g. amended_value missing when choice="amended") flows
//   through parseResponse and surfaces as an ApiError with status: 422.
// - 404 (session or event not found, or 404 on IDOR-deflected access) flows
//   through parseResponse and surfaces as an ApiError with status: 404.
// - 409 (already-resolved event, or conflict between concurrent resolves)
//   flows through parseResponse and surfaces as an ApiError with status: 409.
//
// Per the existing client convention, all four functions throw the typed
// ApiError envelope; callers branch on error_type / status code, not on
// detail text.

/**
 * List interpretation events for a session.
 *
 * `status` selects the row subset:
 *   - "pending" — only choice="pending" rows (active review affordances).
 *   - "all"     — every row regardless of choice (for audit-readiness
 *                 counts and the post-resolve event list).
 *
 * The backend default is "all"; this client mirrors that default so the
 * common case (fetch everything on session load for the readiness panel)
 * needs no explicit argument.
 */
export async function listInterpretationEvents(
  sessionId: string,
  status: "pending" | "all" = "all",
  signal?: AbortSignal,
): Promise<InterpretationEvent[]> {
  // Use URLSearchParams to handle the query-string boundary cleanly: it
  // does percent-encoding correctly for any future status-value extension
  // and keeps the call site free of manual string concatenation.
  const qs = new URLSearchParams({ status });
  const response = await fetch(
    `/api/sessions/${sessionId}/interpretations?${qs.toString()}`,
    {
      method: "GET",
      headers: authHeaders(),
      signal,
    },
  );
  const body = await parseResponse<ListInterpretationEventsResponse>(response);
  return body.events;
}

/**
 * Resolve a single interpretation event.
 *
 * `body.choice` is narrowed to the two user-driven values
 * (accepted_as_drafted, amended) — opt-out goes through a separate route,
 * and pending/abandoned are not user resolve actions.
 *
 * The response returns the resolved event row PLUS the new composition
 * state produced by patching the affected LLM transform.  Single-envelope
 * shape so the caller can update its event-list view and
 * composition-state view atomically.
 */
export async function resolveInterpretation(
  sessionId: string,
  eventId: string,
  body: InterpretationResolveRequest,
  signal?: AbortSignal,
): Promise<InterpretationResolveResponse> {
  const response = await fetch(
    `/api/sessions/${sessionId}/interpretations/${eventId}/resolve`,
    {
      method: "POST",
      headers: authHeaders("application/json"),
      body: JSON.stringify(body),
      signal,
    },
  );
  return parseResponse<InterpretationResolveResponse>(response);
}

/**
 * Record the per-session "stop asking about interpretations" decision.
 *
 * No request body: the route is a discrete user decision, parameterised
 * only by the session ID and the authenticated actor (carried by the
 * auth middleware).  On success the backend (a) sets the session's
 * interpretation_review_disabled flag and (b) writes an opt-out row to
 * interpretation_events_table with choice='opted_out' and
 * interpretation_source='auto_interpreted_opt_out'.
 */
export async function optOutOfInterpretations(
  sessionId: string,
  signal?: AbortSignal,
): Promise<InterpretationOptOutResponse> {
  const response = await fetch(
    `/api/sessions/${sessionId}/interpretations/opt_out`,
    {
      method: "POST",
      headers: authHeaders("application/json"),
      // The route accepts an empty body; sending "{}" rather than omitting
      // body entirely so the Content-Type: application/json header has a
      // matching payload (some HTTP intermediaries reject the inverse).
      body: "{}",
      signal,
    },
  );
  return parseResponse<InterpretationOptOutResponse>(response);
}

/**
 * Retroactive audit of auto-baked interpretations (F-22).
 *
 * After a session has opted out, the composer-LLM continues to auto-bake
 * interpretations.  This route lets the user retroactively review every
 * auto-baked event whose interpretation_source is auto_interpreted_opt_out
 * or auto_interpreted_no_surfaces.  user_approved rows are excluded; the
 * standard list route is the right surface for those.
 */
export async function getInterpretationOptOutSummary(
  sessionId: string,
  signal?: AbortSignal,
): Promise<InterpretationEvent[]> {
  const response = await fetch(
    `/api/sessions/${sessionId}/interpretations/opt_out_summary`,
    {
      method: "GET",
      headers: authHeaders(),
      signal,
    },
  );
  const body = await parseResponse<OptOutSummaryResponse>(response);
  return body.events;
}

export {
  fetchAuditReadiness,
  fetchAuditReadinessExplain,
} from "./auditReadiness";
