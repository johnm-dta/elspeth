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
 * authenticator"). Owner-auth endpoints participate in the global 401 logout
 * interceptor via the shared parseResponse helper in api/client.ts; the
 * shared-inspect capability endpoint preserves the reviewer session so the
 * caller can render invalid/revoked-link errors in place.
 *
 * Wire shapes mirror src/elspeth/web/shareable_reviews/models.py verbatim;
 * any drift here is a typed-parse failure.
 */
import type {
  ApiError,
  CompositionState,
  MarkReadyForReviewResponse,
  PipelineMetadata,
  ShareableLinkResponse,
  SharedInspectResponse,
} from "../types/api";
import { validateAuditReadinessSnapshot } from "./auditReadiness";
import { authHeaders, parseResponse } from "./client";

function unexpectedShape(status: number, where: string, cause?: unknown): ApiError {
  // ``cause`` (optional): attach the inner validation exception so server-
  // side debugging keeps the original locator information (e.g. "row 2
  // status not one of..."). Wire-facing ``detail`` stays the labelled
  // summary; ``cause`` is an out-of-band breadcrumb readers can inspect
  // in the browser console / error reporter. ApiError carries it as a
  // free-form field — we deliberately do not promote it into the public
  // wire-error type because consumers should not branch on it.
  const err: ApiError = {
    status,
    detail: `Unexpected response shape from ${where} endpoint`,
  };
  if (cause !== undefined) {
    (err as ApiError & { cause?: unknown }).cause = cause;
  }
  return err;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

// FIX-K (2026-05-19): tighten the Tier-3 → Tier-1 trust boundary.
//
// The previous implementation deferred audit_readiness validation to the
// renderer ("Full shape validation lives in api/auditReadiness.ts; here
// we do the minimum sanity check") which never invoked that validator at
// this path. A wire-corrupted `rows: [null]` or `rows: [{garbage: 1}]`
// would reach the panel renderer and produce `undefined` cells. Likewise,
// `pipeline_metadata` and `composition_snapshot` were validated only as
// `isRecord(...)` — no per-field guarantees — even though the backend
// Pydantic responses (PipelineMetadataResponse / CompositionStateResponse
// at src/elspeth/web/shareable_reviews/models.py:97-179) are strict.
//
// Resolution per plan 19b:100-101: type both as their canonical front-end
// shapes (`PipelineMetadata`, `CompositionState`) and validate the shape
// at the wire boundary, not the renderer.

function isPipelineMetadata(value: unknown): value is PipelineMetadata {
  return (
    isRecord(value) &&
    (typeof value.name === "string" || value.name === null) &&
    (typeof value.description === "string" || value.description === null)
  );
}

function isCompositionSnapshot(value: unknown): value is CompositionState {
  // Wire shape: CompositionState.to_dict() emits {version, metadata,
  // sources, nodes, edges, outputs} — the runtime-only `id` and
  // `validation_*` fields on the front-end `CompositionState` interface
  // are absent on this wire payload (see SharedInspectResponse's type
  // docstring for the wire-vs-runtime caveat). We validate only the
  // fields the wire actually carries.
  if (!isRecord(value)) return false;
  if (typeof value.version !== "number") return false;
  if (!isPipelineMetadata(value.metadata)) return false;
  if (!isRecord(value.sources)) return false;
  if (!Array.isArray(value.nodes)) return false;
  if (!Array.isArray(value.edges)) return false;
  if (!Array.isArray(value.outputs)) return false;
  return true;
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
    !isPipelineMetadata(body.pipeline_metadata) ||
    !isCompositionSnapshot(body.composition_snapshot) ||
    typeof body.yaml !== "string" ||
    typeof body.created_by_user_id !== "string" ||
    typeof body.created_at !== "string" ||
    typeof body.expires_at !== "string"
  ) {
    throw unexpectedShape(status, "shared-inspect");
  }
  // Full per-row + per-validation-check audit_readiness validation
  // delegated to the shared validator in api/auditReadiness.ts (FIX-K).
  // Wrap to re-label the error under the shared-inspect endpoint so
  // callers can match on a single `where` token.
  let audit_readiness;
  try {
    audit_readiness = validateAuditReadinessSnapshot(body.audit_readiness, status);
  } catch (exc) {
    // Preserve the inner validation cause (e.g. "row 2 status not one
    // of...") on the relabelled ApiError so operators debugging a wire-
    // shape rejection have the locator instead of just the endpoint
    // label.
    throw unexpectedShape(status, "shared-inspect", exc);
  }
  return {
    session_id: body.session_id,
    state_id: body.state_id,
    pipeline_metadata: body.pipeline_metadata,
    composition_snapshot: body.composition_snapshot,
    yaml: body.yaml,
    audit_readiness,
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
  return validateSharedInspectResponse(
    await parseResponse<unknown>(response, { logoutOnUnauthorized: false }),
    response.status,
  );
}
