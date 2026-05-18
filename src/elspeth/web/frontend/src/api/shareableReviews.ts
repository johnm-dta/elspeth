/**
 * API client for the shareable-reviews endpoints (Phase 6A backend, Phase 6B frontend).
 *
 * Three endpoints, all returning strict Pydantic payloads:
 *
 *   POST /api/sessions/{session_id}/mark-ready-for-review
 *   GET  /api/sessions/{session_id}/shareable-link
 *   GET  /api/sessions/shared/{token}
 *
 * The first two require ownership; the third grants a specific authenticated
 * user read-only access to a session they don't own (the token is a
 * capability, NOT an authenticator — see plan 19a §"Capability vs
 * authenticator"). All three participate in the global 401 logout
 * interceptor via the shared parseResponse helper in api/client.ts.
 *
 * Wire shapes mirror src/elspeth/web/shareable_reviews/models.py verbatim;
 * any drift here is a typed-parse failure.
 */
import type {
  ApiError,
  AuditReadinessSnapshot,
  MarkReadyForReviewResponse,
  ShareableLinkResponse,
  SharedInspectResponse,
} from "../types/api";
import { authHeaders, parseResponse } from "./client";

function unexpectedShape(status: number, where: string): ApiError {
  return {
    status,
    detail: `Unexpected response shape from ${where} endpoint`,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isAuditReadinessSnapshot(value: unknown): value is AuditReadinessSnapshot {
  // The audit_readiness sub-object is the Phase 2 AuditReadinessSnapshot
  // verbatim. Full shape validation lives in api/auditReadiness.ts; here we
  // do the minimum sanity check (top-level keys present + rows is an array)
  // because re-validating the entire six-row panel on every shared-inspect
  // GET would duplicate that logic. The route layer's response_model has
  // already validated server-side; the runtime check here catches obvious
  // wire-level corruption.
  return (
    isRecord(value) &&
    typeof value.session_id === "string" &&
    typeof value.composition_version === "number" &&
    typeof value.checked_at === "string" &&
    Array.isArray(value.rows)
  );
}

function validateMarkReadyResponse(body: unknown, status: number): MarkReadyForReviewResponse {
  if (
    !isRecord(body) ||
    typeof body.token !== "string" ||
    typeof body.share_url !== "string" ||
    typeof body.expires_at !== "string" ||
    typeof body.payload_digest !== "string"
  ) {
    throw unexpectedShape(status, "mark-ready-for-review");
  }
  return {
    token: body.token,
    share_url: body.share_url,
    expires_at: body.expires_at,
    payload_digest: body.payload_digest,
  };
}

function validateShareableLinkResponse(body: unknown, status: number): ShareableLinkResponse {
  if (
    !isRecord(body) ||
    typeof body.token !== "string" ||
    typeof body.share_url !== "string" ||
    typeof body.expires_at !== "string" ||
    typeof body.state_id !== "string" ||
    typeof body.payload_digest !== "string"
  ) {
    throw unexpectedShape(status, "shareable-link");
  }
  return {
    token: body.token,
    share_url: body.share_url,
    expires_at: body.expires_at,
    state_id: body.state_id,
    payload_digest: body.payload_digest,
  };
}

function validateSharedInspectResponse(body: unknown, status: number): SharedInspectResponse {
  if (
    !isRecord(body) ||
    typeof body.session_id !== "string" ||
    typeof body.state_id !== "string" ||
    !isRecord(body.pipeline_metadata) ||
    !isRecord(body.composition_snapshot) ||
    typeof body.yaml !== "string" ||
    !isAuditReadinessSnapshot(body.audit_readiness) ||
    typeof body.created_by_user_id !== "string" ||
    typeof body.created_at !== "string" ||
    typeof body.expires_at !== "string"
  ) {
    throw unexpectedShape(status, "shared-inspect");
  }
  return {
    session_id: body.session_id,
    state_id: body.state_id,
    pipeline_metadata: body.pipeline_metadata,
    composition_snapshot: body.composition_snapshot,
    yaml: body.yaml,
    audit_readiness: body.audit_readiness,
    created_by_user_id: body.created_by_user_id,
    created_at: body.created_at,
    expires_at: body.expires_at,
  };
}

/** POST /api/sessions/{session_id}/mark-ready-for-review. */
export async function markReadyForReview(
  sessionId: string,
  signal?: AbortSignal,
): Promise<MarkReadyForReviewResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/mark-ready-for-review`, {
    method: "POST",
    headers: authHeaders(),
    signal,
  });
  return validateMarkReadyResponse(await parseResponse<unknown>(response), response.status);
}

/** GET /api/sessions/{session_id}/shareable-link. */
export async function fetchShareableLink(
  sessionId: string,
  signal?: AbortSignal,
): Promise<ShareableLinkResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/shareable-link`, {
    method: "GET",
    headers: authHeaders(),
    signal,
  });
  return validateShareableLinkResponse(await parseResponse<unknown>(response), response.status);
}

/** GET /api/sessions/shared/{token}. */
export async function fetchSharedInspect(
  token: string,
  signal?: AbortSignal,
): Promise<SharedInspectResponse> {
  const response = await fetch(`/api/sessions/shared/${encodeURIComponent(token)}`, {
    method: "GET",
    headers: authHeaders(),
    signal,
  });
  return validateSharedInspectResponse(await parseResponse<unknown>(response), response.status);
}
