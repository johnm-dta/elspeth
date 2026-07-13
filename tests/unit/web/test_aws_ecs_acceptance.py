"""Acceptance coordinator tests for AWS operator telemetry."""

from __future__ import annotations

from dataclasses import asdict

from elspeth.web.aws_ecs_acceptance import (
    AcceptancePolicy,
    SanitizedResourceIdentity,
    verify_operator_telemetry,
    verify_operator_telemetry_outage,
)


class _Audit:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def write_sentinel(self, sentinel_sha256: str) -> str:
        assert len(sentinel_sha256) == 64
        self.events.append("audit.write")
        return "landscape-run-internal"

    def verify_sentinel(self, run_id: str, sentinel_sha256: str) -> bool:
        assert run_id == "landscape-run-internal"
        assert len(sentinel_sha256) == 64
        self.events.append("audit.verify")
        return True


class _Emitter:
    def __init__(self, events: list[str], *, delivery: bool = True) -> None:
        self.events = events
        self.delivery = delivery

    def emit_web_metric(self, sentinel_value: int) -> bool:
        assert sentinel_value >= 0
        self.events.append("metric.emit")
        return self.delivery

    def emit_pipeline_lifecycle(self, run_id: str) -> bool:
        assert run_id == "landscape-run-internal"
        self.events.append("trace.emit")
        return self.delivery

    def stop_collector(self) -> None:
        self.events.append("collector.stop")

    def health_degraded(self) -> bool:
        self.events.append("health.degraded")
        return not self.delivery


class _Queries:
    def __init__(self, *, available_on: int) -> None:
        self.available_on = available_on
        self.metric_calls = 0
        self.trace_calls = 0

    def metric_observed(self, *, metric_name: str, sentinel_value: int) -> bool:
        assert metric_name == "operator.acceptance.sentinel"
        assert sentinel_value >= 0
        self.metric_calls += 1
        return self.metric_calls >= self.available_on

    def trace_observed(self, *, trace_name: str, run_id: str) -> bool:
        assert trace_name == "RunFinished"
        assert run_id == "landscape-run-internal"
        self.trace_calls += 1
        return self.trace_calls >= self.available_on


def test_positive_lane_is_audit_first_bounded_and_sanitized() -> None:
    events: list[str] = []
    queries = _Queries(available_on=3)
    sleeps: list[float] = []
    evidence = verify_operator_telemetry(
        audit=_Audit(events),
        emitter=_Emitter(events),
        queries=queries,
        resource=SanitizedResourceIdentity(
            service_name="elspeth-web",
            service_version="0.7.1",
            deployment_environment="acceptance",
            cloud_provider="aws",
        ),
        policy=AcceptancePolicy(attempts=3, interval_seconds=0.25),
        sleep=sleeps.append,
        sentinel_factory=lambda: "non-content-sentinel",
        now=lambda: 1234.5,
    )

    assert events[:4] == ["audit.write", "audit.verify", "metric.emit", "trace.emit"]
    assert queries.metric_calls == 3
    assert queries.trace_calls == 3
    assert sleeps == [0.25, 0.25]
    assert set(asdict(evidence)) == {"metric_name", "trace_name", "observed_at", "resource", "sentinel_sha256"}
    rendered = repr(evidence)
    assert "non-content-sentinel" not in rendered
    assert "landscape-run-internal" not in rendered


def test_negative_lane_keeps_audit_and_reports_no_false_receipt() -> None:
    events: list[str] = []
    evidence = verify_operator_telemetry_outage(
        audit=_Audit(events),
        emitter=_Emitter(events, delivery=False),
        sentinel_factory=lambda: "negative-sentinel",
        now=lambda: 1235.5,
    )

    assert events == [
        "audit.write",
        "collector.stop",
        "metric.emit",
        "trace.emit",
        "audit.verify",
        "health.degraded",
    ]
    assert evidence.landscape_correct is True
    assert evidence.telemetry_degraded is True
    assert evidence.cloud_receipt is False
    assert "negative-sentinel" not in repr(evidence)
