"""Strict, audited wrapper for Bedrock Runtime ApplyGuardrail."""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import structlog

import elspeth.contracts.errors as contract_errors
from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.core.canonical import stable_hash
from elspeth.plugins.infrastructure.clients.base import AuditedClientBase, TelemetryEmitCallback

if TYPE_CHECKING:
    from elspeth.contracts.audit_protocols import CallRecorder

logger = structlog.get_logger(__name__)

GuardrailSource = Literal["INPUT", "OUTPUT"]

PROMPT_ATTACK_FILTERS: tuple[str, ...] = ("PROMPT_ATTACK",)
HARMFUL_CONTENT_FILTERS: tuple[str, ...] = ("INSULTS", "HATE", "SEXUAL", "VIOLENCE", "MISCONDUCT")
ALLOWED_FILTERS = frozenset((*PROMPT_ATTACK_FILTERS, *HARMFUL_CONTENT_FILTERS))

ALL_USAGE_KEYS: frozenset[str] = frozenset(
    {
        "automatedReasoningPolicies",
        "automatedReasoningPolicyUnits",
        "contentPolicyImageUnits",
        "contentPolicyUnits",
        "contextualGroundingPolicyUnits",
        "sensitiveInformationPolicyFreeUnits",
        "sensitiveInformationPolicyUnits",
        "topicPolicyUnits",
        "wordPolicyUnits",
    }
)
_REQUIRED_USAGE_KEYS = frozenset(
    {
        "contentPolicyUnits",
        "contextualGroundingPolicyUnits",
        "sensitiveInformationPolicyFreeUnits",
        "sensitiveInformationPolicyUnits",
        "topicPolicyUnits",
        "wordPolicyUnits",
    }
)
_FILTER_ACTIONS = frozenset({"NONE", "BLOCKED"})
_FILTER_CONFIDENCE = frozenset({"NONE", "LOW", "MEDIUM", "HIGH"})
_MAX_USAGE_UNITS = 2**31 - 1
_MAX_OUTPUTS = 8
_MAX_OUTPUT_TEXT = 4096
_MAX_REQUEST_TEXT = 1_000_000
_MAX_ACTION_REASON = 4096
_MAX_LONG = 2**63 - 1
_APPLIED_GUARDRAIL_ID = re.compile(r"[a-z0-9]+\Z")
_APPLIED_GUARDRAIL_VERSION = re.compile(r"(?:[1-9][0-9]{0,7}|DRAFT)\Z")
_APPLIED_GUARDRAIL_ARN = re.compile(r"arn:aws(?:-[^:]+)?:bedrock:[a-z0-9-]{1,20}:[0-9]{12}:guardrail/[a-z0-9]+\Z")
_GUARDRAIL_ORIGINS = frozenset({"REQUEST", "ACCOUNT_ENFORCED", "ORGANIZATION_ENFORCED"})
_GUARDRAIL_OWNERSHIP = frozenset({"SELF", "CROSS_ACCOUNT"})


class GuardrailResponseError(ValueError):
    """Raised when provider data cannot prove a safe or blocked decision."""

    def __init__(self) -> None:
        super().__init__("malformed Bedrock Guardrail response")


class GuardrailServiceError(RuntimeError):
    """Sanitized terminal SDK failure."""

    def __init__(self, *, retryable: bool) -> None:
        super().__init__("Bedrock Guardrail request failed")
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class GuardrailUsage:
    units: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    detected: bool
    intervened: bool
    matched_filters: tuple[str, ...]
    usage: GuardrailUsage
    request_id: str | None


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise GuardrailResponseError
    return value


def _list(value: object, *, exact: int | None = None, minimum: int = 0, maximum: int) -> Sequence[object]:
    if type(value) is not list:
        raise GuardrailResponseError
    if exact is not None and len(value) != exact:
        raise GuardrailResponseError
    if not minimum <= len(value) <= maximum:
        raise GuardrailResponseError
    return value


def _parse_usage(value: object) -> GuardrailUsage:
    usage = _mapping(value)
    keys = frozenset(usage)
    if not keys >= _REQUIRED_USAGE_KEYS or not keys <= ALL_USAGE_KEYS:
        raise GuardrailResponseError
    units: list[tuple[str, int]] = []
    for key, item in usage.items():
        if type(item) is not int or not 0 <= item <= _MAX_USAGE_UNITS:
            raise GuardrailResponseError
        units.append((key, item))
    return GuardrailUsage(tuple(sorted(units)))


def _parse_retry_attempts(value: object) -> int:
    if value is None:
        return 1
    metadata = _mapping(value)
    retry_count = metadata.get("RetryAttempts", 0)
    if type(retry_count) is not int or not 0 <= retry_count <= 10:
        raise GuardrailResponseError
    return retry_count + 1


def _parse_request_metadata(value: object) -> tuple[str | None, int]:
    attempts = _parse_retry_attempts(value)
    if value is None:
        return None, attempts
    metadata = _mapping(value)
    request_id = metadata.get("RequestId")
    if request_id is not None and (type(request_id) is not str or not 1 <= len(request_id) <= 256):
        raise GuardrailResponseError
    return request_id, attempts


def _validate_coverage(value: object) -> None:
    coverage = _mapping(value)
    if not set(coverage) <= {"textCharacters", "images"}:
        raise GuardrailResponseError
    for raw_counts in coverage.values():
        counts = _mapping(raw_counts)
        if not set(counts) <= {"guarded", "total"}:
            raise GuardrailResponseError
        for name in ("guarded", "total"):
            if name in counts:
                count = counts[name]
                if type(count) is not int or not 0 <= count <= _MAX_USAGE_UNITS:
                    raise GuardrailResponseError
        guarded = counts.get("guarded")
        total = counts.get("total")
        if type(guarded) is int and type(total) is int and guarded > total:
            raise GuardrailResponseError


def _validate_invocation_metrics(value: object) -> None:
    metrics = _mapping(value)
    if not set(metrics) <= {"guardrailProcessingLatency", "usage", "guardrailCoverage"}:
        raise GuardrailResponseError
    if "guardrailProcessingLatency" in metrics:
        latency = metrics["guardrailProcessingLatency"]
        if type(latency) is not int or not 0 <= latency <= _MAX_LONG:
            raise GuardrailResponseError
    if "usage" in metrics:
        _parse_usage(metrics["usage"])
    if "guardrailCoverage" in metrics:
        _validate_coverage(metrics["guardrailCoverage"])


def _validate_applied_guardrail_details(value: object) -> None:
    details = _mapping(value)
    allowed = {"guardrailId", "guardrailVersion", "guardrailArn", "guardrailOrigin", "guardrailOwnership"}
    if not set(details) <= allowed:
        raise GuardrailResponseError

    guardrail_id = details.get("guardrailId")
    if guardrail_id is not None and (
        type(guardrail_id) is not str
        or len(guardrail_id) > 2048
        or (guardrail_id != "" and _APPLIED_GUARDRAIL_ID.fullmatch(guardrail_id) is None)
    ):
        raise GuardrailResponseError
    version = details.get("guardrailVersion")
    if version is not None and (type(version) is not str or (version != "" and _APPLIED_GUARDRAIL_VERSION.fullmatch(version) is None)):
        raise GuardrailResponseError
    arn = details.get("guardrailArn")
    if arn is not None and (type(arn) is not str or len(arn) > 2048 or _APPLIED_GUARDRAIL_ARN.fullmatch(arn) is None):
        raise GuardrailResponseError
    origins = details.get("guardrailOrigin")
    if origins is not None:
        raw_origins = _list(origins, minimum=0, maximum=len(_GUARDRAIL_ORIGINS))
        if any(type(origin) is not str or origin not in _GUARDRAIL_ORIGINS for origin in raw_origins):
            raise GuardrailResponseError
        if len(set(raw_origins)) != len(raw_origins):
            raise GuardrailResponseError
    ownership = details.get("guardrailOwnership")
    if ownership is not None and (type(ownership) is not str or ownership not in _GUARDRAIL_OWNERSHIP):
        raise GuardrailResponseError


def _validate_outputs(value: object, *, intervened: bool) -> None:
    outputs = _list(value, minimum=1 if intervened else 0, maximum=_MAX_OUTPUTS if intervened else 0)
    for raw_output in outputs:
        output = _mapping(raw_output)
        if set(output) != {"text"}:
            raise GuardrailResponseError
        text = output["text"]
        if type(text) is not str or len(text) > _MAX_OUTPUT_TEXT:
            raise GuardrailResponseError
        # Validate and discard immediately. Provider-authored canned text is
        # intentionally never copied into a decision, audit, telemetry, or row.


def parse_guardrail_response(
    response: object,
    *,
    required_filters: tuple[str, ...],
) -> tuple[GuardrailDecision, int]:
    """Validate one FULL ApplyGuardrail response without retaining provider text."""
    if required_filters not in (PROMPT_ATTACK_FILTERS, HARMFUL_CONTENT_FILTERS):
        raise ValueError("required_filters must name a supported closed policy")
    root = _mapping(response)
    allowed_root = {
        "usage",
        "action",
        "actionReason",
        "outputs",
        "assessments",
        "guardrailCoverage",
        "ResponseMetadata",
    }
    if not {"usage", "action", "outputs", "assessments"} <= set(root) or set(root) - allowed_root:
        raise GuardrailResponseError

    action = root["action"]
    if action not in ("NONE", "GUARDRAIL_INTERVENED"):
        raise GuardrailResponseError
    action_reason = root.get("actionReason")
    if action_reason is not None and (type(action_reason) is not str or len(action_reason) > _MAX_ACTION_REASON):
        raise GuardrailResponseError
    if "guardrailCoverage" in root:
        _validate_coverage(root["guardrailCoverage"])
    assessments = _list(root["assessments"], exact=1, maximum=1)
    assessment = _mapping(assessments[0])
    if "contentPolicy" not in assessment or set(assessment) - {"contentPolicy", "invocationMetrics", "appliedGuardrailDetails"}:
        raise GuardrailResponseError
    if "invocationMetrics" in assessment:
        _validate_invocation_metrics(assessment["invocationMetrics"])
    if "appliedGuardrailDetails" in assessment:
        _validate_applied_guardrail_details(assessment["appliedGuardrailDetails"])
    content_policy = _mapping(assessment["contentPolicy"])
    if set(content_policy) != {"filters"}:
        raise GuardrailResponseError
    filters = _list(content_policy["filters"], minimum=1, maximum=len(ALLOWED_FILTERS))

    seen: set[str] = set()
    matched: list[str] = []
    blocked: set[str] = set()
    for raw_filter in filters:
        item = _mapping(raw_filter)
        if not {"type", "confidence", "action", "detected"} <= set(item):
            raise GuardrailResponseError
        if set(item) - {"type", "confidence", "filterStrength", "action", "detected"}:
            raise GuardrailResponseError
        filter_type = item["type"]
        confidence = item["confidence"]
        filter_action = item["action"]
        detected = item["detected"]
        if type(filter_type) is not str or filter_type not in ALLOWED_FILTERS or filter_type in seen:
            raise GuardrailResponseError
        if confidence not in _FILTER_CONFIDENCE or filter_action not in _FILTER_ACTIONS or type(detected) is not bool:
            raise GuardrailResponseError
        strength = item.get("filterStrength")
        if strength is not None and strength not in _FILTER_CONFIDENCE:
            raise GuardrailResponseError
        if filter_action == "BLOCKED" and not detected:
            raise GuardrailResponseError
        seen.add(filter_type)
        if detected:
            matched.append(filter_type)
        if filter_action == "BLOCKED":
            blocked.add(filter_type)

    if seen != set(required_filters):
        raise GuardrailResponseError
    intervened = action == "GUARDRAIL_INTERVENED"
    detected_any = bool(matched)
    if intervened:
        if not detected_any or not blocked or not blocked <= set(matched):
            raise GuardrailResponseError
    elif blocked:
        raise GuardrailResponseError
    _validate_outputs(root["outputs"], intervened=intervened)
    usage = _parse_usage(root["usage"])
    request_id, attempts = _parse_request_metadata(root.get("ResponseMetadata"))
    return (
        GuardrailDecision(
            detected=detected_any,
            intervened=intervened,
            matched_filters=tuple(sorted(matched)),
            usage=usage,
            request_id=request_id,
        ),
        attempts,
    )


def _is_retryable_sdk_error(error: Exception) -> bool:
    from botocore.exceptions import ClientError, ConnectionClosedError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError

    if isinstance(error, (ConnectTimeoutError, ConnectionClosedError, EndpointConnectionError, ReadTimeoutError)):
        return True
    if not isinstance(error, ClientError):
        return False
    code = str(error.response.get("Error", {}).get("Code", ""))
    return code in {
        "InternalServerException",
        "ServiceUnavailableException",
        "ThrottlingException",
        "TooManyRequestsException",
    }


class BedrockGuardrailsClient(AuditedClientBase):
    """Single-retry-owner ApplyGuardrail client with audit-first telemetry."""

    def __init__(
        self,
        execution: CallRecorder,
        state_id: str,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        *,
        guardrail_identifier: str,
        guardrail_version: str,
        region: str,
        audit_salt: bytes,
        sdk_client: Any | None = None,
        token_id: str | None = None,
    ) -> None:
        super().__init__(execution, state_id, run_id, telemetry_emit, token_id=token_id)
        if len(audit_salt) < 16:
            raise ValueError("audit_salt must contain at least 16 bytes")
        self._guardrail_identifier = guardrail_identifier
        self._guardrail_version = guardrail_version
        self._region = region
        self._target_fingerprint = hmac.new(
            audit_salt,
            f"{guardrail_identifier}\0{guardrail_version}\0{region}".encode(),
            hashlib.sha256,
        ).hexdigest()
        self._sdk_client = sdk_client if sdk_client is not None else self._build_sdk_client()

    @property
    def sdk_client(self) -> Any:
        return self._sdk_client

    def _build_sdk_client(self) -> Any:
        return build_bedrock_runtime_client(self._region)

    def _emit_after_audit(
        self,
        *,
        status: CallStatus,
        latency_ms: float,
        request_payload: RawCallPayload,
        response_payload: RawCallPayload,
    ) -> None:
        try:
            self._telemetry_emit(
                ExternalCallCompleted(
                    timestamp=datetime.now(UTC),
                    run_id=self._run_id,
                    call_type=CallType.HTTP,
                    provider="aws-bedrock-guardrails",
                    status=status,
                    latency_ms=latency_ms,
                    state_id=self._telemetry_state_id(),
                    token_id=self._telemetry_token_id(),
                    request_hash=stable_hash(request_payload.to_dict()),
                    response_hash=stable_hash(response_payload.to_dict()),
                    request_payload=request_payload,
                    response_payload=response_payload,
                    token_usage=None,
                )
            )
        except contract_errors.TIER_1_ERRORS:
            raise
        except (TypeError, AttributeError, KeyError, NameError):
            raise
        except Exception as error:
            logger.warning(
                "telemetry_emit_failed",
                error_type=type(error).__name__,
                run_id=self._run_id,
                state_id=self._telemetry_state_id(),
                call_type="bedrock_guardrail",
                exc_info=True,
            )

    def apply_guardrail(
        self,
        *,
        text: str,
        source: GuardrailSource,
        required_filters: tuple[str, ...],
    ) -> GuardrailDecision:
        if type(text) is not str or not 1 <= len(text) <= _MAX_REQUEST_TEXT:
            raise ValueError("guardrail text must be a bounded non-empty string")
        if source not in ("INPUT", "OUTPUT"):
            raise ValueError("guardrail source must be INPUT or OUTPUT")
        if required_filters not in (PROMPT_ATTACK_FILTERS, HARMFUL_CONTENT_FILTERS):
            raise ValueError("required_filters must name a supported closed policy")

        call_index = self._next_call_index()
        request_payload = RawCallPayload(
            {
                "operation": "apply_guardrail",
                "target_fingerprint": self._target_fingerprint,
                "source": source,
                "required_filters": required_filters,
            }
        )
        start = time.perf_counter()
        terminal_error: Exception | None = None
        attempts = 1
        try:
            response = self._sdk_client.apply_guardrail(
                guardrailIdentifier=self._guardrail_identifier,
                guardrailVersion=self._guardrail_version,
                source=source,
                outputScope="FULL",
                content=[{"text": {"text": text, "qualifiers": ["guard_content"]}}],
            )
            response_root = _mapping(response)
            attempts = _parse_retry_attempts(response_root.get("ResponseMetadata"))
            decision, parsed_attempts = parse_guardrail_response(response_root, required_filters=required_filters)
            if parsed_attempts != attempts:
                raise GuardrailResponseError
            response_payload = RawCallPayload(
                {
                    "operation": "apply_guardrail",
                    "status": "blocked" if decision.detected else "safe",
                    "attempts": attempts,
                    "detected": decision.detected,
                    "intervened": decision.intervened,
                    "matched_filters": decision.matched_filters,
                    "usage": dict(decision.usage.units),
                    "request_id_present": decision.request_id is not None,
                }
            )
            call_status = CallStatus.SUCCESS
            error_payload = None
        except GuardrailResponseError as error:
            terminal_error = error
            response_payload = RawCallPayload({"operation": "apply_guardrail", "status": "malformed_response", "attempts": attempts})
            call_status = CallStatus.ERROR
            error_payload = RawCallPayload({"type": "malformed_response", "retryable": False})
            decision = None
        except Exception as error:
            retryable = _is_retryable_sdk_error(error)
            terminal_error = GuardrailServiceError(retryable=retryable)
            response_metadata = getattr(error, "response", {}).get("ResponseMetadata", {})
            if isinstance(response_metadata, Mapping):
                raw_retries = response_metadata.get("RetryAttempts", 0)
                if type(raw_retries) is int and 0 <= raw_retries <= 10:
                    attempts = raw_retries + 1
            response_payload = RawCallPayload({"operation": "apply_guardrail", "status": "service_error", "attempts": attempts})
            call_status = CallStatus.ERROR
            error_payload = RawCallPayload({"type": "service_error", "retryable": retryable})
            decision = None

        latency_ms = (time.perf_counter() - start) * 1000
        self._record_call(
            call_index=call_index,
            call_type=CallType.HTTP,
            status=call_status,
            request_data=request_payload,
            response_data=response_payload,
            error=error_payload,
            latency_ms=latency_ms,
        )
        self._emit_after_audit(
            status=call_status,
            latency_ms=latency_ms,
            request_payload=request_payload,
            response_payload=response_payload,
        )
        if terminal_error is not None:
            raise terminal_error
        assert decision is not None
        return decision


def build_bedrock_runtime_client(region: str) -> Any:
    """Build the SDK client with botocore as the sole retry owner."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 3},
            connect_timeout=5,
            read_timeout=15,
        ),
    )
