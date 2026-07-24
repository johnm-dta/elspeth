"""Redacted storage projection for composer tool-invocation evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import canonical_json
from elspeth.web.composer.redaction import (
    MANIFEST,
    redact_tool_call_arguments,
    redact_tool_call_response,
)
from elspeth.web.composer.redaction_telemetry import OtelRedactionTelemetry
from elspeth.web.composer.tool_error_payloads import (
    INVALID_TOOL_ARGUMENTS_REDACTION_STATUS,
    unknown_tool_arguments_redaction,
    unknown_tool_response_redaction,
)


def _hash_canonical_payload(canonical_payload: str) -> str:
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def _load_canonical_mapping(canonical_payload: str | None) -> dict[str, object] | None:
    """Decode Tier-1 canonical audit JSON, failing closed on corruption.

    Dispatch always authors a mapping through ``canonical_json``. A non-None
    payload that no longer decodes to an object is therefore corruption, not
    honest absence. Returning ``None`` would also let callers bypass manifest
    redaction and persist the damaged raw payload.
    """
    if canonical_payload is None:
        return None
    try:
        decoded = json.loads(canonical_payload)
    except json.JSONDecodeError as exc:
        raise AuditIntegrityError("Tier 1 audit anomaly: composer tool-invocation canonical payload is not valid JSON.") from exc
    if type(decoded) is not dict:
        raise AuditIntegrityError("Tier 1 audit anomaly: composer tool-invocation canonical payload must decode to a JSON object.")
    return cast(dict[str, object], decoded)


def _redacted_argument_canonical(
    invocation: ComposerToolInvocation,
    *,
    telemetry: OtelRedactionTelemetry,
) -> str:
    arguments = _load_canonical_mapping(invocation.arguments_canonical)
    if invocation.tool_name not in MANIFEST:
        return canonical_json(unknown_tool_arguments_redaction(telemetry=telemetry))
    if arguments is None:
        return invocation.arguments_canonical
    try:
        redacted = redact_tool_call_arguments(
            invocation.tool_name,
            arguments,
            telemetry=telemetry,
        )
    except PydanticValidationError:
        if invocation.status != ComposerToolStatus.ARG_ERROR:
            raise
        redacted = {
            "_redaction_status": INVALID_TOOL_ARGUMENTS_REDACTION_STATUS,
            "error_class": invocation.error_class,
        }
    return canonical_json(redacted)


def _redacted_result_canonical(
    invocation: ComposerToolInvocation,
    *,
    telemetry: OtelRedactionTelemetry,
) -> str | None:
    result = _load_canonical_mapping(invocation.result_canonical)
    if invocation.tool_name not in MANIFEST:
        return canonical_json(unknown_tool_response_redaction()) if result is not None else None
    if result is None:
        return invocation.result_canonical
    if invocation.status == ComposerToolStatus.ARG_ERROR:
        return invocation.result_canonical
    return canonical_json(
        redact_tool_call_response(
            invocation.tool_name,
            result,
            telemetry=telemetry,
        )
    )


def redacted_tool_invocation_content_and_envelope(
    invocation: ComposerToolInvocation,
) -> tuple[str, dict[str, object]]:
    """Return the one manifest-redacted projection allowed in chat storage.

    ``ComposerToolInvocation`` retains exact in-memory arguments/results for
    convergence and audit binding. Chat storage is a separate persistence
    boundary: hashes are recomputed over the redacted canonical payloads, and
    the canonical pipeline executor hash is carried forward explicitly so
    later proposal settlement can still verify the exact execution result.
    """

    telemetry = OtelRedactionTelemetry()
    arguments_canonical = _redacted_argument_canonical(invocation, telemetry=telemetry)
    result_canonical = _redacted_result_canonical(invocation, telemetry=telemetry)
    if invocation.tool_name == "set_pipeline" and invocation.status is ComposerToolStatus.SUCCESS:
        raw_result = _load_canonical_mapping(invocation.result_canonical)
        if raw_result is not None and ("pipeline_content_hash_schema" in raw_result or "pipeline_content_hash" in raw_result):
            if raw_result.get("pipeline_content_hash_schema") != "composer.pipeline-dispatch-result.v1":
                raise AuditIntegrityError("pipeline dispatch executor-content schema is malformed")
            executor_content_hash = raw_result.get("pipeline_content_hash")
            if type(executor_content_hash) is not str:
                raise AuditIntegrityError("pipeline dispatch executor-content hash is malformed")
            redacted_result = _load_canonical_mapping(result_canonical)
            if redacted_result is None:
                raise AuditIntegrityError("pipeline dispatch redacted result is missing")
            redacted_result["pipeline_content_hash_schema"] = "composer.pipeline-dispatch-result.v1"
            redacted_result["pipeline_content_hash"] = executor_content_hash
            result_canonical = canonical_json(redacted_result)

    invocation_payload: dict[str, Any] = invocation.to_dict()
    invocation_payload["arguments_canonical"] = arguments_canonical
    invocation_payload["arguments_hash"] = _hash_canonical_payload(arguments_canonical)
    invocation_payload["result_canonical"] = result_canonical
    invocation_payload["result_hash"] = _hash_canonical_payload(result_canonical) if result_canonical is not None else None

    if invocation.status == ComposerToolStatus.PLUGIN_CRASH:
        content = json.dumps(
            {
                "error_class": invocation.error_class,
                "error_message": invocation.error_message,
            }
        )
    elif result_canonical is not None:
        content = result_canonical
    else:
        content = json.dumps(
            {
                "error_class": invocation.error_class,
                "error_message": invocation.error_message,
            }
        )
    return content, {"_kind": "audit", "invocation": invocation_payload}


__all__ = ["redacted_tool_invocation_content_and_envelope"]
