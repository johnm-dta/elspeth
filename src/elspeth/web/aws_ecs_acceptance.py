"""Sanitized AWS ECS operator-telemetry acceptance coordination.

The live AWS adapters are deliberately injected.  This coordinator owns the
ordering, bounded retry, and evidence projection contracts shared by the
in-task acceptance command: Landscape first, operational telemetry second.
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

_METRIC_NAME = "operator.acceptance.sentinel"
_TRACE_NAME = "RunFinished"
_MAX_IDENTITY_CHARS = 128


class OperatorTelemetryAcceptanceError(RuntimeError):
    """Static acceptance failure safe for an operator receipt."""


class AuditSentinel(Protocol):
    def write_sentinel(self, sentinel_sha256: str) -> str: ...

    def verify_sentinel(self, run_id: str, sentinel_sha256: str) -> bool: ...


class TelemetrySentinelEmitter(Protocol):
    def emit_web_metric(self, sentinel_value: int) -> bool: ...

    def emit_pipeline_lifecycle(self, run_id: str) -> bool: ...

    def stop_collector(self) -> None: ...

    def health_degraded(self) -> bool: ...


class TelemetryQueries(Protocol):
    def metric_observed(self, *, metric_name: str, sentinel_value: int) -> bool: ...

    def trace_observed(self, *, trace_name: str, run_id: str) -> bool: ...


def _bounded_identity(field: str, value: str) -> str:
    if (
        not value.strip()
        or value != value.strip()
        or len(value) > _MAX_IDENTITY_CHARS
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"{field} must be a non-blank bounded string without control characters")
    return value


@dataclass(frozen=True, slots=True)
class SanitizedResourceIdentity:
    """Closed non-content identity persisted in acceptance evidence."""

    service_name: str
    service_version: str
    deployment_environment: str
    cloud_provider: str

    def __post_init__(self) -> None:
        for field in ("service_name", "service_version", "deployment_environment", "cloud_provider"):
            _bounded_identity(field, getattr(self, field))
        if self.cloud_provider != "aws":
            raise ValueError("cloud_provider must be aws")


@dataclass(frozen=True, slots=True)
class AcceptancePolicy:
    attempts: int = 10
    interval_seconds: float = 3.0

    def __post_init__(self) -> None:
        if type(self.attempts) is not int or not 1 <= self.attempts <= 20:
            raise ValueError("attempts must be an integer from 1 through 20")
        if type(self.interval_seconds) not in {int, float}:
            raise ValueError("interval_seconds must be a finite number")
        interval = float(self.interval_seconds)
        if not math.isfinite(interval) or not 0 <= interval <= 30:
            raise ValueError("interval_seconds must be a finite number from 0 through 30")


_DEFAULT_ACCEPTANCE_POLICY = AcceptancePolicy()


@dataclass(frozen=True, slots=True)
class OperatorTelemetryEvidence:
    metric_name: str
    trace_name: str
    observed_at: float
    resource: SanitizedResourceIdentity
    sentinel_sha256: str


@dataclass(frozen=True, slots=True)
class OperatorTelemetryOutageEvidence:
    observed_at: float
    sentinel_sha256: str
    landscape_correct: bool
    telemetry_degraded: bool
    cloud_receipt: bool


def _sentinel_facts(factory: Callable[[], str]) -> tuple[str, int]:
    raw = factory()
    if type(raw) is not str or not raw or len(raw) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise OperatorTelemetryAcceptanceError("acceptance sentinel generation failed validation")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    # Exactly representable in the IEEE-754 integer range used by metric
    # backends while remaining unique enough for a bounded acceptance window.
    return digest, int(digest[:12], 16)


def verify_operator_telemetry(
    *,
    audit: AuditSentinel,
    emitter: TelemetrySentinelEmitter,
    queries: TelemetryQueries,
    resource: SanitizedResourceIdentity,
    policy: AcceptancePolicy = _DEFAULT_ACCEPTANCE_POLICY,
    sleep: Callable[[float], None] = time.sleep,
    sentinel_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    now: Callable[[], float] = time.time,
) -> OperatorTelemetryEvidence:
    """Prove one audit-first metric and lifecycle trace with bounded retries."""

    sentinel_sha256, sentinel_value = _sentinel_facts(sentinel_factory)
    run_id = audit.write_sentinel(sentinel_sha256)
    if not run_id:
        raise OperatorTelemetryAcceptanceError("Landscape sentinel write returned no run identity")
    if not audit.verify_sentinel(run_id, sentinel_sha256):
        raise OperatorTelemetryAcceptanceError("Landscape sentinel was not durable before telemetry")

    metric_delivery = emitter.emit_web_metric(sentinel_value)
    trace_delivery = emitter.emit_pipeline_lifecycle(run_id)
    if not metric_delivery or not trace_delivery:
        raise OperatorTelemetryAcceptanceError("operator telemetry delivery was unavailable")

    metric_seen = False
    trace_seen = False
    for attempt in range(policy.attempts):
        metric_seen = metric_seen or queries.metric_observed(metric_name=_METRIC_NAME, sentinel_value=sentinel_value)
        trace_seen = trace_seen or queries.trace_observed(trace_name=_TRACE_NAME, run_id=run_id)
        if metric_seen and trace_seen:
            break
        if attempt + 1 < policy.attempts:
            sleep(float(policy.interval_seconds))
    if not metric_seen or not trace_seen:
        raise OperatorTelemetryAcceptanceError("bounded CloudWatch/X-Ray observation did not find both signals")

    return OperatorTelemetryEvidence(
        metric_name=_METRIC_NAME,
        trace_name=_TRACE_NAME,
        observed_at=now(),
        resource=resource,
        sentinel_sha256=sentinel_sha256,
    )


def verify_operator_telemetry_outage(
    *,
    audit: AuditSentinel,
    emitter: TelemetrySentinelEmitter,
    sentinel_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    now: Callable[[], float] = time.time,
) -> OperatorTelemetryOutageEvidence:
    """Prove a collector outage cannot undo or impersonate audit evidence."""

    sentinel_sha256, sentinel_value = _sentinel_facts(sentinel_factory)
    run_id = audit.write_sentinel(sentinel_sha256)
    if not run_id:
        raise OperatorTelemetryAcceptanceError("Landscape sentinel write returned no run identity")

    emitter.stop_collector()
    metric_delivery = emitter.emit_web_metric(sentinel_value)
    trace_delivery = emitter.emit_pipeline_lifecycle(run_id)
    landscape_correct = audit.verify_sentinel(run_id, sentinel_sha256)
    telemetry_degraded = emitter.health_degraded()
    cloud_receipt = metric_delivery or trace_delivery
    if not landscape_correct:
        raise OperatorTelemetryAcceptanceError("Landscape sentinel was not durable during telemetry outage")
    if not telemetry_degraded:
        raise OperatorTelemetryAcceptanceError("telemetry outage did not produce degraded health")
    if cloud_receipt:
        raise OperatorTelemetryAcceptanceError("telemetry outage produced a false delivery receipt")

    return OperatorTelemetryOutageEvidence(
        observed_at=now(),
        sentinel_sha256=sentinel_sha256,
        landscape_correct=True,
        telemetry_degraded=True,
        cloud_receipt=False,
    )


__all__ = [
    "AcceptancePolicy",
    "AuditSentinel",
    "OperatorTelemetryAcceptanceError",
    "OperatorTelemetryEvidence",
    "OperatorTelemetryOutageEvidence",
    "SanitizedResourceIdentity",
    "TelemetryQueries",
    "TelemetrySentinelEmitter",
    "verify_operator_telemetry",
    "verify_operator_telemetry_outage",
]
