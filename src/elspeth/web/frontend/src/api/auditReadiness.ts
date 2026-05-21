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
  ValidationCheck,
  ValidationError,
  ValidationResult,
  ValidationWarning,
} from "../types/api";
import { VALIDATION_CHECK_OUTCOME_CODE_VALUES } from "../types/index";
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
const VALIDATION_CHECK_OUTCOME_CODES = new Set<string>(VALIDATION_CHECK_OUTCOME_CODE_VALUES);

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

function isValidationEntry(entry: unknown): entry is ValidationError | ValidationWarning {
  return (
    isRecord(entry) &&
    (typeof entry.component_id === "string" || entry.component_id === null) &&
    (typeof entry.component_type === "string" || entry.component_type === null) &&
    typeof entry.message === "string" &&
    (typeof entry.suggestion === "string" || entry.suggestion === null) &&
    (
      entry.error_code === undefined ||
      typeof entry.error_code === "string" ||
      entry.error_code === null
    )
  );
}

function isValidationCheck(check: unknown): check is ValidationCheck {
  return (
    isRecord(check) &&
    typeof check.name === "string" &&
    typeof check.passed === "boolean" &&
    typeof check.detail === "string" &&
    Array.isArray(check.affected_nodes) &&
    check.affected_nodes.every((node) => typeof node === "string") &&
    (
      check.outcome_code === null ||
      (
        typeof check.outcome_code === "string" &&
        VALIDATION_CHECK_OUTCOME_CODES.has(check.outcome_code)
      )
    )
  );
}

function isSemanticEdgeContract(contract: unknown): boolean {
  return (
    isRecord(contract) &&
    typeof contract.from_id === "string" &&
    typeof contract.to_id === "string" &&
    typeof contract.consumer_plugin === "string" &&
    (typeof contract.producer_plugin === "string" || contract.producer_plugin === null) &&
    typeof contract.producer_field === "string" &&
    typeof contract.consumer_field === "string" &&
    (
      contract.outcome === "satisfied" ||
      contract.outcome === "conflict" ||
      contract.outcome === "unknown"
    ) &&
    typeof contract.requirement_code === "string"
  );
}

function isValidationReadinessBlocker(blocker: unknown): boolean {
  return (
    isRecord(blocker) &&
    typeof blocker.code === "string" &&
    (typeof blocker.component_id === "string" || blocker.component_id === null) &&
    (typeof blocker.component_type === "string" || blocker.component_type === null) &&
    typeof blocker.detail === "string"
  );
}

function isValidationReadiness(readiness: unknown): boolean {
  return (
    isRecord(readiness) &&
    typeof readiness.authoring_valid === "boolean" &&
    typeof readiness.execution_ready === "boolean" &&
    typeof readiness.completion_ready === "boolean" &&
    Array.isArray(readiness.blockers) &&
    readiness.blockers.every(isValidationReadinessBlocker)
  );
}

function isValidationResult(result: unknown): result is ValidationResult {
  return (
    isRecord(result) &&
    typeof result.is_valid === "boolean" &&
    (typeof result.summary === "string" || result.summary === undefined) &&
    Array.isArray(result.checks) &&
    result.checks.every(isValidationCheck) &&
    Array.isArray(result.errors) &&
    result.errors.every(isValidationEntry) &&
    isValidationReadiness(result.readiness) &&
    (
      result.warnings === undefined ||
      (
        Array.isArray(result.warnings) &&
        result.warnings.every(isValidationEntry)
      )
    ) &&
    (
      result.semantic_contracts === undefined ||
      (
        Array.isArray(result.semantic_contracts) &&
        result.semantic_contracts.every(isSemanticEdgeContract)
      )
    )
  );
}

/**
 * Validate a wire-shape `AuditReadinessSnapshot`. Exported so other API
 * clients (e.g. the shareable-reviews `SharedInspectResponse` which embeds
 * the same snapshot verbatim — see plan 19a §"Post-Phase-18 merge fact")
 * can re-use the full per-row + per-validation-check validation without
 * duplicating logic at every consumer. Throws an `ApiError` whose `detail`
 * mentions the audit-readiness endpoint; callers that need a different
 * `where` label should wrap and re-throw with their own context.
 */
export function validateAuditReadinessSnapshot(
  body: unknown,
  status: number,
): AuditReadinessSnapshot {
  assertBaseEnvelope(body, status);
  if (
    typeof body.checked_at !== "string" ||
    !Array.isArray(body.rows) ||
    !isValidationResult(body.validation_result)
  ) {
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
    validation_result: body.validation_result,
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
  return validateAuditReadinessSnapshot(await parseResponse<unknown>(response), response.status);
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
