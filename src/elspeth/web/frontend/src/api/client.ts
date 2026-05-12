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
  BlobMetadata,
  ChatMessage,
  CompositionState,
  CompositionStateVersion,
  ComposerProgressSnapshot,
  ExecutionFanoutAck,
  ExecutionFanoutGuard,
  CancelRunResponse,
  PluginSchemaInfo,
  PluginSummary,
  Run,
  RunDiagnostics,
  RunDiagnosticsEvaluation,
  RunOutputArtifactPreview,
  RunOutputsResponse,
  SecretInventoryItem,
  Session,
  UserProfile,
  ValidationResult,
  SystemStatus,
} from "@/types/index";
import type {
  GetGuidedResponse,
  GuidedRespondRequest,
  GuidedRespondResponse,
} from "@/types/guided";

// ── Token Management ────────────────────────────────────────────────────────

const TOKEN_KEY = "auth_token";

function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

/**
 * Build headers with auth token injection and optional content type.
 * Every authenticated request includes Authorization: Bearer {token}.
 */
function authHeaders(contentType?: string): HeadersInit {
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

/**
 * Parse a response. Throws ApiError for non-2xx status codes.
 *
 * Includes a global 401 interceptor: any API call returning 401 triggers
 * authStore.logout() to handle token expiry mid-session without requiring
 * each caller to check for auth failures.
 *
 * Error envelope handling priority:
 *   1. error_type (if present in response body) -- most specific
 *   2. HTTP status code -- structural fallback
 *   3. detail text -- human-readable description
 */
async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    // Global 401 interceptor -- trigger logout on any auth failure.
    // Dynamic import avoids circular dependency at module load time
    // (authStore imports from client, client imports authStore for logout).
    if (response.status === 401) {
      const { useAuthStore } = await import("@/stores/authStore");
      useAuthStore.getState().logout();
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
    } catch {
      // Response body wasn't JSON -- use statusText as detail fallback
    }

    const error: ApiError = {
      status: response.status,
      detail,
      error_type: errorType,
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
export async function fetchSessions(): Promise<Session[]> {
  const response = await fetch("/api/sessions", {
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
): Promise<{ message: ChatMessage; state: CompositionState | null }> {
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
  return parseResponse<{ message: ChatMessage; state: CompositionState | null }>(
    response,
  );
}

/** Re-run the composer without inserting a new user message.
 *  Used by the retry flow when the user message is already persisted. */
export async function recompose(
  sessionId: string,
  signal?: AbortSignal,
): Promise<{ message: ChatMessage; state: CompositionState | null }> {
  const response = await fetch(`/api/sessions/${sessionId}/recompose`, {
    method: "POST",
    headers: authHeaders("application/json"),
    signal,
  });
  return parseResponse<{ message: ChatMessage; state: CompositionState | null }>(
    response,
  );
}

/**
 * Fetch the current guided-session state for a session.
 *
 * Returns the active GuidedSession (step + history + terminal), the
 * server-emitted next turn payload (if any), and the current composition
 * state.  When no guided session has started for the session, the server
 * initialises one and returns the Step 1 turn.
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
): Promise<ValidationResult> {
  const response = await fetch(`/api/sessions/${sessionId}/validate`, {
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
): Promise<{ run_id: string }> {
  const init: RequestInit = {
    method: "POST",
    headers: authHeaders("application/json"),
  };
  if (fanoutAck) {
    init.body = JSON.stringify({ fanout_ack_token: fanoutAck.token });
  }
  const response = await fetch(`/api/sessions/${sessionId}/execute`, init);
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
