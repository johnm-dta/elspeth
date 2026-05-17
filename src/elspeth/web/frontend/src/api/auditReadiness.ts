/**
 * API client for the audit-readiness panel (Phase 2).
 *
 * Two GET endpoints; both return strict Pydantic payloads. Reuses the
 * shared parser from api/client.ts so audit-readiness calls participate in
 * the global 401 logout interceptor.
 */
import type {
  AuditReadinessSnapshot,
  AuditReadinessExplain,
  ApiError,
  ReadinessRow,
  ReadinessRowId,
  ReadinessStatus,
} from "../types/api";
import { authHeaders, parseResponse } from "./client";

type AuditReadinessBaseEnvelope = {
  session_id: string;
  composition_version: number;
} & Record<string, unknown>;

const READINESS_ROW_IDS = new Set<ReadinessRowId>([
  "validation",
  "plugin_trust",
  "provenance",
  "retention",
  "llm_interpretations",
  "secrets",
]);
const READINESS_STATUSES = new Set<ReadinessStatus>(["ok", "warning", "error", "not_applicable"]);

function unexpectedShape(status: number): ApiError {
  return {
    status,
    detail: "Unexpected response shape from audit-readiness endpoint",
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function assertBaseEnvelope(body: unknown, status: number): asserts body is AuditReadinessBaseEnvelope {
  if (!isRecord(body) || typeof body.session_id !== "string" || typeof body.composition_version !== "number") {
    throw unexpectedShape(status);
  }
}

function isReadinessRow(row: unknown): row is ReadinessRow {
  return (
    isRecord(row) &&
    typeof row.id === "string" &&
    READINESS_ROW_IDS.has(row.id as ReadinessRowId) &&
    typeof row.label === "string" &&
    typeof row.status === "string" &&
    READINESS_STATUSES.has(row.status as ReadinessStatus) &&
    typeof row.summary === "string" &&
    (typeof row.detail === "string" || row.detail === null) &&
    Array.isArray(row.component_ids) &&
    row.component_ids.every((id) => typeof id === "string")
  );
}

function validateSnapshot(body: unknown, status: number): AuditReadinessSnapshot {
  assertBaseEnvelope(body, status);
  if (typeof body.checked_at !== "string" || !Array.isArray(body.rows)) {
    throw unexpectedShape(status);
  }
  for (const row of body.rows) {
    if (!isReadinessRow(row)) {
      throw unexpectedShape(status);
    }
  }
  return {
    session_id: body.session_id,
    composition_version: body.composition_version,
    checked_at: body.checked_at,
    rows: body.rows,
  };
}

function validateExplain(body: unknown, status: number): AuditReadinessExplain {
  assertBaseEnvelope(body, status);
  if (typeof body.narrative !== "string") {
    throw unexpectedShape(status);
  }
  return {
    session_id: body.session_id,
    composition_version: body.composition_version,
    narrative: body.narrative,
  };
}

export async function fetchAuditReadiness(
  sessionId: string,
  signal?: AbortSignal,
): Promise<AuditReadinessSnapshot> {
  const response = await fetch(
    `/api/sessions/${sessionId}/audit-readiness`,
    { method: "GET", headers: authHeaders(), signal },
  );
  return validateSnapshot(await parseResponse<unknown>(response), response.status);
}

export async function fetchAuditReadinessExplain(
  sessionId: string,
  signal?: AbortSignal,
): Promise<AuditReadinessExplain> {
  const response = await fetch(
    `/api/sessions/${sessionId}/audit-readiness/explain`,
    { method: "GET", headers: authHeaders(), signal },
  );
  return validateExplain(await parseResponse<unknown>(response), response.status);
}
