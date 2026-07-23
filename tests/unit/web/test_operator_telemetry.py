"""AWS ECS operator telemetry policy and process bootstrap tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult, MetricsData, PeriodicExportingMetricReader

from elspeth.contracts.events import RunStarted
from elspeth.core.config import TelemetrySettings
from elspeth.telemetry.manager import TelemetryManager
from elspeth.web import operator_telemetry
from elspeth.web.config import WebSettings
from elspeth.web.operator_telemetry import (
    AWS_OTLP_ENDPOINT,
    SAFE_CLOUDWATCH_METRIC_ATTRIBUTES,
    OperatorTelemetryFactories,
    apply_operator_pipeline_telemetry,
    bootstrap_operator_telemetry,
    build_aws_operator_pipeline_telemetry,
    record_operator_pipeline_queue_drops,
    reset_operator_telemetry_for_tests,
)
from tests.fixtures.telemetry import FailingExporter, MockTelemetryConfig


def _web_settings(**overrides: object) -> WebSettings:
    values: dict[str, object] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }
    values.update(overrides)
    return WebSettings(**values)  # type: ignore[arg-type]


def _pipeline_settings(telemetry: TelemetrySettings) -> Any:
    @dataclass(frozen=True)
    class _Settings:
        telemetry: TelemetrySettings

        def model_copy(self, *, update: dict[str, object]) -> _Settings:
            return _Settings(telemetry=update["telemetry"])  # type: ignore[arg-type]

    return _Settings(telemetry=telemetry)


@pytest.fixture(autouse=True)
def _reset_runtime() -> None:
    reset_operator_telemetry_for_tests()
    yield
    reset_operator_telemetry_for_tests()


def test_local_pipeline_telemetry_is_unchanged() -> None:
    authored = TelemetrySettings(
        enabled=True,
        granularity="full",
        fail_on_total_exporter_failure=True,
        exporters=[{"name": "datadog", "options": {"api_key": "authored-secret"}}],
    )
    pipeline = _pipeline_settings(authored)

    effective = apply_operator_pipeline_telemetry(pipeline, _web_settings())

    assert effective is pipeline
    assert effective.telemetry is authored


def test_cloudwatch_metric_dimensions_exclude_unbounded_identity_and_content() -> None:
    forbidden = {
        "account_id",
        "aws_account_id",
        "content",
        "exception_message",
        "prompt",
        "request_id",
        "row_id",
        "run_id",
        "session_id",
        "task_arn",
        "token_id",
        "url",
        "user_id",
    }

    assert SAFE_CLOUDWATCH_METRIC_ATTRIBUTES.isdisjoint(forbidden)
    assert {"reason", "operation", "status"} <= SAFE_CLOUDWATCH_METRIC_ATTRIBUTES


@pytest.mark.parametrize(
    "authored",
    [
        TelemetrySettings(enabled=False),
        TelemetrySettings(enabled=True, granularity="full", exporters=[{"name": "console"}]),
        TelemetrySettings(enabled=True, exporters=[{"name": "azure_monitor", "options": {"connection_string": "secret"}}]),
        TelemetrySettings(enabled=True, exporters=[{"name": "datadog", "options": {"api_key": "secret"}}]),
        TelemetrySettings(
            enabled=True,
            fail_on_total_exporter_failure=True,
            exporters=[{"name": "otlp", "options": {"endpoint": "https://remote.invalid:4317", "headers": {"authorization": "secret"}}}],
        ),
    ],
)
def test_aws_pipeline_telemetry_is_replaced_by_operator_policy(authored: TelemetrySettings) -> None:
    web = _web_settings(
        deployment_target="aws-ecs",
        operator_telemetry="aws-otlp",
        operator_telemetry_service_name="elspeth-web-prod",
        operator_telemetry_environment="production",
        operator_telemetry_release="git-deadbeef",
        operator_telemetry_ecs_cluster="elspeth-production",
        operator_telemetry_ecs_service="elspeth-web",
        operator_telemetry_task_definition_family="elspeth-web-task",
        operator_telemetry_task_definition_revision="42",
        operator_pipeline_telemetry_granularity="rows",
    )

    effective = apply_operator_pipeline_telemetry(_pipeline_settings(authored), web)

    assert effective.telemetry.enabled is True
    assert effective.telemetry.granularity == "rows"
    assert effective.telemetry.fail_on_total_exporter_failure is False
    assert len(effective.telemetry.exporters) == 1
    exporter = effective.telemetry.exporters[0]
    assert exporter.name == "otlp"
    assert exporter.options == {
        "endpoint": AWS_OTLP_ENDPOINT,
        "headers": {},
        "service_name": "elspeth-web-prod",
        "service_version": "git-deadbeef",
        "deployment_environment": "production",
        "cloud_provider": "aws",
        "aws_ecs_cluster_name": "elspeth-production",
        "aws_ecs_service_name": "elspeth-web",
        "aws_ecs_task_family": "elspeth-web-task",
        "aws_ecs_task_revision": "42",
        "batch_size": 100,
    }
    rendered = repr(effective.telemetry.model_dump())
    assert "remote.invalid" not in rendered
    assert "authorization" not in rendered
    assert "secret" not in rendered


def test_aws_pipeline_telemetry_pure_builder_matches_applied_policy() -> None:
    web = _web_settings(
        deployment_target="aws-ecs",
        operator_telemetry="aws-otlp",
        operator_telemetry_service_name="elspeth-web-prod",
        operator_telemetry_environment="production",
        operator_telemetry_release="git-deadbeef",
        operator_telemetry_ecs_cluster="elspeth-production",
        operator_telemetry_ecs_service="elspeth-web",
        operator_telemetry_task_definition_family="elspeth-web-task",
        operator_telemetry_task_definition_revision="42",
        operator_pipeline_telemetry_granularity="rows",
    )

    built = build_aws_operator_pipeline_telemetry(web)
    applied = apply_operator_pipeline_telemetry(_pipeline_settings(TelemetrySettings(enabled=False)), web)

    assert built == applied.telemetry


@dataclass
class _FakeReader:
    kind: str
    exporter: object | None = None
    interval_ms: int | None = None


@dataclass
class _FakeExporter:
    endpoint: str
    insecure: bool
    headers: dict[str, str]
    results: list[MetricExportResult] = field(default_factory=list)

    def export(self, _data: object, timeout_millis: float = 10_000, **_kwargs: object) -> MetricExportResult:
        del timeout_millis
        return self.results.pop(0) if self.results else MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        del timeout_millis
        return True

    def shutdown(self, timeout_millis: float = 30_000, **_kwargs: object) -> None:
        del timeout_millis


class _CapturingMetricExporter(MetricExporter):
    def __init__(self) -> None:
        super().__init__()
        self.exports: list[MetricsData] = []

    def export(self, metrics_data: MetricsData, timeout_millis: float = 10_000, **_kwargs: object) -> MetricExportResult:
        del timeout_millis
        self.exports.append(metrics_data)
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        del timeout_millis
        return True

    def shutdown(self, timeout_millis: float = 30_000, **_kwargs: object) -> None:
        del timeout_millis


def test_aws_metric_export_preserves_only_bounded_acceptance_correlation() -> None:
    inner = _CapturingMetricExporter()
    exporter = operator_telemetry._HealthTrackingMetricExporter(inner, operator_telemetry._ExportHealth())
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60_000)
    provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    try:
        counter = provider.get_meter("acceptance-contract").create_counter("operator.acceptance.sentinel")
        counter.add(
            17,
            attributes={
                "elspeth.acceptance.namespace": "acceptance-run-a",
                "elspeth.acceptance.sentinel": "17",
                "run_id": "must-not-become-a-metric-dimension",
            },
        )

        assert provider.force_flush(timeout_millis=5_000) is True
        assert len(inner.exports) == 1
        point = inner.exports[0].resource_metrics[0].scope_metrics[0].metrics[0].data.data_points[0]  # type: ignore[union-attr]
        assert point.attributes == {
            "elspeth.acceptance.namespace": "acceptance-run-a",
            "elspeth.acceptance.sentinel": "17",
        }
    finally:
        provider.shutdown()


@dataclass
class _FakeProvider:
    readers: list[object]
    resource: object
    views: tuple[object, ...]
    force_flush_calls: list[float] = field(default_factory=list)
    shutdown_calls: list[float] = field(default_factory=list)
    force_flush_error: BaseException | None = None
    gauges: dict[str, list[Any]] = field(default_factory=dict)

    def get_meter(self, _name: str, _version: str) -> _FakeProvider:
        return self

    def create_observable_gauge(self, name: str, *, callbacks: list[Any], **_kwargs: object) -> object:
        self.gauges[name] = callbacks
        return object()

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        self.force_flush_calls.append(timeout_millis)
        if self.force_flush_error is not None:
            raise self.force_flush_error
        return True

    def shutdown(self, timeout_millis: float = 30_000) -> None:
        self.shutdown_calls.append(timeout_millis)


def _factories(record: dict[str, object]) -> OperatorTelemetryFactories:
    def prometheus_reader() -> _FakeReader:
        reader = _FakeReader("prometheus")
        record.setdefault("prometheus", []).append(reader)  # type: ignore[union-attr]
        return reader

    def otlp_exporter(**kwargs: object) -> _FakeExporter:
        exporter = _FakeExporter(**kwargs)  # type: ignore[arg-type]
        record.setdefault("exporters", []).append(exporter)  # type: ignore[union-attr]
        return exporter

    def periodic_reader(exporter: object, *, export_interval_millis: int, export_timeout_millis: int) -> _FakeReader:
        assert export_timeout_millis > 0
        reader = _FakeReader("periodic", exporter, export_interval_millis)
        record.setdefault("periodic", []).append(reader)  # type: ignore[union-attr]
        return reader

    def provider(readers: list[object], *, resource: object, views: tuple[object, ...]) -> _FakeProvider:
        value = _FakeProvider(readers, resource, views)
        record.setdefault("providers", []).append(value)  # type: ignore[union-attr]
        return value

    def set_provider(provider: object) -> None:
        record.setdefault("set_provider", []).append(provider)  # type: ignore[union-attr]

    return OperatorTelemetryFactories(
        prometheus_reader=prometheus_reader,
        otlp_exporter=otlp_exporter,
        periodic_reader=periodic_reader,
        meter_provider=provider,
        set_meter_provider=set_provider,
    )


def test_process_bootstrap_local_is_idempotent_prometheus_only() -> None:
    record: dict[str, object] = {}
    factories = _factories(record)

    first = bootstrap_operator_telemetry(_web_settings(), factories=factories)
    second = bootstrap_operator_telemetry(_web_settings(), factories=factories)

    assert first is second
    assert len(record["providers"]) == 1  # type: ignore[arg-type]
    assert len(record["set_provider"]) == 1  # type: ignore[arg-type]
    assert [reader.kind for reader in first.readers] == ["prometheus"]


def test_process_bootstrap_aws_adds_one_fixed_otlp_reader_and_safe_resource() -> None:
    record: dict[str, object] = {}
    settings = _web_settings(
        deployment_target="aws-ecs",
        operator_telemetry="aws-otlp",
        operator_telemetry_environment="production",
        operator_telemetry_release="git-deadbeef",
        operator_telemetry_ecs_cluster="elspeth-production",
        operator_telemetry_ecs_service="elspeth-web",
        operator_telemetry_task_definition_family="elspeth-web-task",
        operator_telemetry_task_definition_revision="42",
        operator_telemetry_export_interval_seconds=17,
    )

    runtime = bootstrap_operator_telemetry(settings, factories=_factories(record))

    assert [reader.kind for reader in runtime.readers] == ["prometheus", "periodic"]
    exporter = record["exporters"][0]  # type: ignore[index]
    assert exporter.endpoint == AWS_OTLP_ENDPOINT
    assert exporter.insecure is True
    assert exporter.headers == {}
    assert runtime.readers[1].interval_ms == 17_000
    assert runtime.resource.attributes == {
        "service.name": "elspeth-web",
        "service.version": "git-deadbeef",
        "deployment.environment": "production",
        "cloud.provider": "aws",
        "aws.ecs.cluster.name": "elspeth-production",
        "aws.ecs.service.name": "elspeth-web",
        "aws.ecs.task.family": "elspeth-web-task",
        "aws.ecs.task.revision": "42",
    }


@pytest.mark.parametrize(
    ("field", "raw_value"),
    [
        ("operator_telemetry_service_name", "arn:aws:ecs:ap-southeast-2:123456789012:service/elspeth-web"),
        ("operator_telemetry_service_name", "123456789012"),
        ("operator_telemetry_service_name", "elspeth-123456789012-web"),
        ("operator_telemetry_environment", "arn:aws:ecs:ap-southeast-2:123456789012:cluster/production"),
        ("operator_telemetry_environment", "123456789012"),
        ("operator_telemetry_environment", "prod-123456789012-blue"),
    ],
)
def test_aws_bootstrap_defensively_rejects_unvalidated_resource_labels(field: str, raw_value: str) -> None:
    settings = _web_settings(
        deployment_target="aws-ecs",
        operator_telemetry="aws-otlp",
        operator_telemetry_environment="production",
        operator_telemetry_release="git-deadbeef",
        operator_telemetry_ecs_cluster="elspeth-production",
        operator_telemetry_ecs_service="elspeth-web",
        operator_telemetry_task_definition_family="elspeth-web-task",
        operator_telemetry_task_definition_revision="42",
    ).model_copy(update={field: raw_value})

    with pytest.raises(ValueError, match=field) as caught:
        bootstrap_operator_telemetry(settings, factories=_factories({}))

    assert raw_value not in str(caught.value)


def test_pipeline_exporter_failures_are_excluded_from_operator_queue_drop_gauge() -> None:
    record: dict[str, object] = {}
    settings = _web_settings(
        deployment_target="aws-ecs",
        operator_telemetry="aws-otlp",
        operator_telemetry_environment="production",
        operator_telemetry_release="git-deadbeef",
        operator_telemetry_ecs_cluster="elspeth-production",
        operator_telemetry_ecs_service="elspeth-web",
        operator_telemetry_task_definition_family="elspeth-web-task",
        operator_telemetry_task_definition_revision="42",
    )
    runtime = bootstrap_operator_telemetry(settings, factories=_factories(record))
    provider = runtime.provider
    assert isinstance(provider, _FakeProvider)
    gauge_callback = provider.gauges["operator.telemetry.queue_drops"][0]
    assert gauge_callback(None)[0].value == 0
    manager = TelemetryManager(MockTelemetryConfig(), exporters=[FailingExporter()])
    try:
        manager.handle_event(
            RunStarted(
                timestamp=datetime(2026, 7, 14, tzinfo=UTC),
                run_id="run-dropped",
                config_hash="config-hash",
                source_plugin="text",
            )
        )
        manager.flush()
        assert manager.health_metrics["events_dropped"] == 1
        assert manager.health_metrics["queue_drops"] == 0

        record_operator_pipeline_queue_drops(manager.health_metrics["queue_drops"])

        assert gauge_callback(None)[0].value == 0
    finally:
        manager.close()


def test_pipeline_queue_drop_fact_is_observed_by_operator_queue_drop_gauge() -> None:
    record: dict[str, object] = {}
    runtime = bootstrap_operator_telemetry(
        _web_settings(
            deployment_target="aws-ecs",
            operator_telemetry="aws-otlp",
            operator_telemetry_environment="production",
            operator_telemetry_release="git-deadbeef",
            operator_telemetry_ecs_cluster="elspeth-production",
            operator_telemetry_ecs_service="elspeth-web",
            operator_telemetry_task_definition_family="elspeth-web-task",
            operator_telemetry_task_definition_revision="42",
        ),
        factories=_factories(record),
    )
    provider = runtime.provider
    assert isinstance(provider, _FakeProvider)
    gauge_callback = provider.gauges["operator.telemetry.queue_drops"][0]

    record_operator_pipeline_queue_drops(1)

    assert gauge_callback(None)[0].value == 1


@pytest.mark.asyncio
async def test_shutdown_is_bounded_and_once_only() -> None:
    record: dict[str, object] = {}
    settings = _web_settings(
        deployment_target="aws-ecs",
        operator_telemetry="aws-otlp",
        operator_telemetry_environment="production",
        operator_telemetry_release="git-deadbeef",
        operator_telemetry_ecs_cluster="elspeth-production",
        operator_telemetry_ecs_service="elspeth-web",
        operator_telemetry_task_definition_family="elspeth-web-task",
        operator_telemetry_task_definition_revision="42",
    )
    runtime = bootstrap_operator_telemetry(settings, factories=_factories(record))

    await runtime.shutdown()
    await runtime.shutdown()

    provider = record["providers"][0]  # type: ignore[index]
    assert provider.force_flush_calls == [5_000]
    assert provider.shutdown_calls == [5_000]


@pytest.mark.asyncio
@pytest.mark.parametrize("flush_error", [TimeoutError(), ConnectionError("collector unavailable")])
async def test_shutdown_is_still_attempted_once_after_handled_flush_failure(flush_error: BaseException) -> None:
    record: dict[str, object] = {}
    settings = _web_settings(
        deployment_target="aws-ecs",
        operator_telemetry="aws-otlp",
        operator_telemetry_environment="production",
        operator_telemetry_release="git-deadbeef",
        operator_telemetry_ecs_cluster="elspeth-production",
        operator_telemetry_ecs_service="elspeth-web",
        operator_telemetry_task_definition_family="elspeth-web-task",
        operator_telemetry_task_definition_revision="42",
    )
    runtime = bootstrap_operator_telemetry(settings, factories=_factories(record))
    provider = record["providers"][0]  # type: ignore[index]
    provider.force_flush_error = flush_error

    await runtime.shutdown()
    await runtime.shutdown()

    assert provider.force_flush_calls == [5_000]
    assert provider.shutdown_calls == [5_000]
