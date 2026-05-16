/**
 * API client for the audit-readiness panel (Phase 2).
 *
 * Two GET endpoints; both return strict Pydantic payloads. Mirrors
 * the auth/parse pattern used by api/client.ts::validatePipeline.
 *
 * Technical debt: getToken/authHeaders/parseResponse are duplicated from
 * api/client.ts (see client.ts:403–424, the account-level preferences block).
 * Phase 8 cleanup task: consolidate these helpers as exported utilities from
 * client.ts. Currently at least two API modules carry the duplicates
 * (client.ts inline + auditReadiness.ts).
 */
import type {
  AuditReadinessSnapshot,
  AuditReadinessExplain,
  ApiError,
} from "../types/api";

// Token key must match src/api/client.ts (TOKEN_KEY = "auth_token") and
// authStore.ts. Phase 8 will replace this with a shared getToken() helper.
function getToken(): string | null {
  return localStorage.getItem("auth_token");
}

function authHeaders(contentType?: string): HeadersInit {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (contentType) headers["Content-Type"] = contentType;
  return headers;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: string | undefined;
    let error_type: string | undefined;
    try {
      const body = (await response.clone().json()) as Record<string, unknown>;
      if (typeof body.detail === "string") detail = body.detail;
      if (typeof body.error_type === "string") error_type = body.error_type;
    } catch {
      // body wasn't JSON; ignore
    }
    const error: ApiError = {
      status: response.status,
      detail: detail ?? response.statusText,
      error_type,
    };
    throw error;
  }
  const body = (await response.json()) as Record<string, unknown>;
  // Trust-boundary check: the backend's _StrictResponse model guarantees this
  // shape, but a proxy / CDN / corrupted-body failure could deliver a
  // different payload with a 200 status. Validate the two discriminating
  // fields both response types share before handing the body to the caller.
  // Both AuditReadinessSnapshot and AuditReadinessExplain carry
  // session_id (string) and composition_version (number ≥ 1). If either
  // is wrong we throw a synthetic ApiError so callers see a typed failure,
  // not a runtime TypeError later in the call chain.
  if (typeof body.session_id !== "string" || typeof body.composition_version !== "number") {
    throw {
      status: response.status,
      detail: "Unexpected response shape from audit-readiness endpoint",
    } as ApiError;
  }
  return body as T;
}

export async function fetchAuditReadiness(
  sessionId: string,
  signal?: AbortSignal,
): Promise<AuditReadinessSnapshot> {
  const response = await fetch(
    `/api/sessions/${sessionId}/audit-readiness`,
    { method: "GET", headers: authHeaders(), signal },
  );
  return parseResponse<AuditReadinessSnapshot>(response);
}

export async function fetchAuditReadinessExplain(
  sessionId: string,
  signal?: AbortSignal,
): Promise<AuditReadinessExplain> {
  const response = await fetch(
    `/api/sessions/${sessionId}/audit-readiness/explain`,
    { method: "GET", headers: authHeaders(), signal },
  );
  return parseResponse<AuditReadinessExplain>(response);
}
