"""Process-wide web telemetry and AWS ECS pipeline telemetry policy.

Landscape remains the authoritative run record.  This module owns only the
best-effort operational signal path: Prometheus for every web process and a
fixed task-local OTLP metric reader plus pipeline overlay in AWS ECS mode.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import structlog
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult, MetricsData, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from elspeth import __version__
from elspeth.core.config import ElspethSettings, ExporterSettings, TelemetrySettings
from elspeth.telemetry.errors import TELEMETRY_TRANSPORT_ERRORS
from elspeth.web.config import WebSettings

AWS_OTLP_ENDPOINT = "http://127.0.0.1:4317"
_EXPORT_TIMEOUT_MILLIS = 5_000
_SHUTDOWN_WALL_TIMEOUT_SECONDS = 5.5

# The task-local exporter removes every other point attribute before OTLP.
# Resource identity is configured separately and is similarly bounded by
# WebSettings.  The Prometheus reader receives the existing attribute sets so
# its authenticated exposition remains backwards-compatible.
SAFE_CLOUDWATCH_METRIC_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "cap_type",
        "completion_path",
        "completion_verb",
        "component_type",
        "failure_class",
        "from_mode",
        "kind",
        "operation",
        "outcome",
        "probe_status",
        "reason",
        "result",
        "source",
        "status",
        "to_mode",
    }
)

_log = structlog.get_logger(__name__)


class _Provider(Protocol):
    def force_flush(self, timeout_millis: float = 10_000) -> bool: ...

    def shutdown(self, timeout_millis: float = 30_000) -> None: ...


@dataclass(frozen=True, slots=True)
class OperatorTelemetryFactories:
    """Resettable construction seam used by bootstrap unit tests."""

    prometheus_reader: Callable[[], object]
    otlp_exporter: Callable[..., object]
    periodic_reader: Callable[..., object]
    meter_provider: Callable[..., _Provider]
    set_meter_provider: Callable[[object], None]


def _production_factories() -> OperatorTelemetryFactories:
    def _provider(readers: Sequence[object], *, resource: Resource, views: tuple[object, ...]) -> MeterProvider:
        return MeterProvider(
            metric_readers=cast(Sequence[Any], readers),
            resource=resource,
            views=cast(Sequence[Any], views),
            shutdown_on_exit=False,
        )

    return OperatorTelemetryFactories(
        prometheus_reader=PrometheusMetricReader,
        otlp_exporter=OTLPMetricExporter,
        periodic_reader=PeriodicExportingMetricReader,
        meter_provider=_provider,
        set_meter_provider=lambda provider: metrics.set_meter_provider(cast(Any, provider)),
    )


@dataclass(slots=True)
class _ExportHealth:
    attempted: int = 0
    failures: int = 0
    queue_drops: int = 0
    consecutive_failures: int = 0
    last_success_monotonic: float | None = None

    def success(self) -> None:
        self.attempted += 1
        self.consecutive_failures = 0
        self.last_success_monotonic = time.monotonic()

    def failure(self) -> None:
        self.attempted += 1
        self.failures += 1
        self.consecutive_failures += 1


def _safe_point_attributes(attributes: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if attributes is None:
        return None
    return {key: value for key, value in attributes.items() if key in SAFE_CLOUDWATCH_METRIC_ATTRIBUTES}


def _sanitize_metric_data(metrics_data: MetricsData) -> MetricsData:
    """Copy a collection with CloudWatch-safe point attributes and no exemplars."""

    resource_metrics = []
    for resource_metric in metrics_data.resource_metrics:
        scope_metrics = []
        for scope_metric in resource_metric.scope_metrics:
            sanitized_metrics = []
            for metric in scope_metric.metrics:
                data = metric.data
                points = getattr(data, "data_points", None)
                if points is None:
                    sanitized_metrics.append(metric)
                    continue
                sanitized_points = []
                for point in points:
                    replacements: dict[str, object] = {"attributes": _safe_point_attributes(point.attributes)}
                    if hasattr(point, "exemplars"):
                        replacements["exemplars"] = ()
                    sanitized_points.append(dataclasses.replace(point, **replacements))
                sanitized_data = dataclasses.replace(cast(Any, data), data_points=tuple(sanitized_points))
                sanitized_metrics.append(dataclasses.replace(metric, data=sanitized_data))
            scope_metrics.append(dataclasses.replace(scope_metric, metrics=tuple(sanitized_metrics)))
        resource_metrics.append(dataclasses.replace(resource_metric, scope_metrics=tuple(scope_metrics)))
    return MetricsData(resource_metrics=tuple(resource_metrics))


class _HealthTrackingMetricExporter(MetricExporter):
    """Sanitize AWS-bound dimensions and retain aggregate exporter health."""

    def __init__(self, inner: MetricExporter, health: _ExportHealth) -> None:
        super().__init__(
            preferred_temporality=inner._preferred_temporality,
            preferred_aggregation=inner._preferred_aggregation,
        )
        self._inner = inner
        self._health = health

    def _record_transport_failure(self) -> None:
        self._health.failure()
        count = self._health.consecutive_failures
        # Aggregate logs at powers of two.  The metric callback exposes the
        # exact count without recursively recording from inside export().
        if count & (count - 1) == 0:
            _log.warning(
                "operator_otlp_export_unavailable",
                consecutive_failures=count,
                destination="task-local",
            )

    def export(self, metrics_data: MetricsData, timeout_millis: float = 10_000, **kwargs: object) -> MetricExportResult:
        try:
            result = self._inner.export(_sanitize_metric_data(metrics_data), timeout_millis=timeout_millis, **kwargs)
        except TELEMETRY_TRANSPORT_ERRORS:
            self._record_transport_failure()
            return MetricExportResult.FAILURE
        if result is MetricExportResult.SUCCESS:
            self._health.success()
        else:
            self._record_transport_failure()
        return result

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        try:
            return self._inner.force_flush(timeout_millis=timeout_millis)
        except TELEMETRY_TRANSPORT_ERRORS:
            self._record_transport_failure()
            return False

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
        try:
            self._inner.shutdown(timeout_millis=timeout_millis, **kwargs)
        except TELEMETRY_TRANSPORT_ERRORS:
            self._record_transport_failure()


@dataclass(slots=True)
class OperatorTelemetryRuntime:
    """Retained process provider plus bounded, once-only shutdown."""

    mode: str
    provider: _Provider
    readers: tuple[Any, ...]
    resource: Resource
    health: _ExportHealth
    _shutdown_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _shutdown_complete: bool = False

    async def shutdown(self) -> None:
        if self.mode != "aws-otlp":
            return
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            try:
                flushed = await asyncio.wait_for(
                    asyncio.to_thread(self.provider.force_flush, timeout_millis=_EXPORT_TIMEOUT_MILLIS),
                    timeout=_SHUTDOWN_WALL_TIMEOUT_SECONDS,
                )
                if not flushed:
                    _log.warning("operator_otlp_force_flush_incomplete", destination="task-local")
            except TimeoutError:
                _log.warning("operator_otlp_force_flush_timeout", destination="task-local")
            except TELEMETRY_TRANSPORT_ERRORS:
                _log.warning("operator_otlp_force_flush_unavailable", destination="task-local")
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.provider.shutdown, timeout_millis=_EXPORT_TIMEOUT_MILLIS),
                    timeout=_SHUTDOWN_WALL_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                _log.warning("operator_otlp_shutdown_timeout", destination="task-local")
            except TELEMETRY_TRANSPORT_ERRORS:
                _log.warning("operator_otlp_shutdown_unavailable", destination="task-local")
            self._shutdown_complete = True


_runtime: OperatorTelemetryRuntime | None = None
_runtime_lock = threading.Lock()


def _required_aws_resource_identity(settings: WebSettings) -> dict[str, str]:
    configured = {
        "service.version": settings.operator_telemetry_release,
        "aws.ecs.cluster.name": settings.operator_telemetry_ecs_cluster,
        "aws.ecs.service.name": settings.operator_telemetry_ecs_service,
        "aws.ecs.task.family": settings.operator_telemetry_task_definition_family,
        "aws.ecs.task.revision": settings.operator_telemetry_task_definition_revision,
    }
    if any(value is None for value in configured.values()):
        raise ValueError("validated AWS ECS operator telemetry identity is incomplete")
    return {key: cast(str, value) for key, value in configured.items()}


def _wire_health_instruments(provider: _Provider, health: _ExportHealth) -> None:
    get_meter = getattr(provider, "get_meter", None)
    if get_meter is None:  # Narrow test-provider seam; real MeterProvider always exposes get_meter.
        return
    meter = get_meter("elspeth.web.operator_telemetry", __version__)
    meter.create_observable_gauge(
        "operator.telemetry.last_success_age_seconds",
        callbacks=[
            lambda _options: [
                Observation(-1.0 if health.last_success_monotonic is None else max(0.0, time.monotonic() - health.last_success_monotonic))
            ]
        ],
        description="Age of the last successful task-local OTLP metric export; -1 means never exported.",
        unit="s",
    )
    meter.create_observable_gauge(
        "operator.telemetry.export_failures",
        callbacks=[lambda _options: [Observation(health.failures)]],
        description="Cumulative task-local OTLP metric export failures.",
    )
    meter.create_observable_gauge(
        "operator.telemetry.queue_drops",
        callbacks=[lambda _options: [Observation(health.queue_drops)]],
        description="Cumulative task-local metric queue drops.",
    )
    meter.create_observable_gauge(
        "operator.telemetry.collector_unavailable",
        callbacks=[lambda _options: [Observation(1 if health.consecutive_failures else 0)]],
        description="One while the task-local OTLP collector has consecutive failures.",
    )


def bootstrap_operator_telemetry(
    settings: WebSettings,
    *,
    factories: OperatorTelemetryFactories | None = None,
) -> OperatorTelemetryRuntime:
    """Install exactly one process MeterProvider and retain its readers."""

    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime

        selected = factories or _production_factories()
        resource_attributes: dict[str, str] = {
            "service.name": settings.operator_telemetry_service_name,
            "service.version": __version__,
        }
        if settings.operator_telemetry_environment is not None:
            resource_attributes["deployment.environment"] = settings.operator_telemetry_environment
        if settings.operator_telemetry == "aws-otlp":
            resource_attributes["cloud.provider"] = "aws"
            resource_attributes.update(_required_aws_resource_identity(settings))
        # Explicit Resource avoids the SDK's process/environment detectors,
        # which add deployment-varying attributes outside this closed AWS
        # identity contract.
        resource = Resource(resource_attributes)

        readers: list[object] = [selected.prometheus_reader()]
        health = _ExportHealth()
        if settings.operator_telemetry == "aws-otlp":
            raw_exporter = selected.otlp_exporter(endpoint=AWS_OTLP_ENDPOINT, insecure=True, headers={})
            exporter = _HealthTrackingMetricExporter(raw_exporter, health) if isinstance(raw_exporter, MetricExporter) else raw_exporter
            readers.append(
                selected.periodic_reader(
                    exporter,
                    export_interval_millis=settings.operator_telemetry_export_interval_seconds * 1_000,
                    export_timeout_millis=_EXPORT_TIMEOUT_MILLIS,
                )
            )

        provider = selected.meter_provider(readers, resource=resource, views=())
        selected.set_meter_provider(provider)
        _runtime = OperatorTelemetryRuntime(
            mode=settings.operator_telemetry,
            provider=provider,
            readers=tuple(readers),
            resource=resource,
            health=health,
        )
        if settings.operator_telemetry == "aws-otlp":
            _wire_health_instruments(provider, health)
        return _runtime


def apply_operator_pipeline_telemetry(settings: ElspethSettings, web_settings: WebSettings) -> ElspethSettings:
    """Replace web-authored routing with the fixed AWS operator policy."""

    if web_settings.deployment_target != "aws-ecs":
        return settings
    identity = _required_aws_resource_identity(web_settings)
    effective = TelemetrySettings(
        enabled=True,
        granularity=web_settings.operator_pipeline_telemetry_granularity,
        backpressure_mode="drop",
        fail_on_total_exporter_failure=False,
        exporters=[
            ExporterSettings(
                name="otlp",
                options={
                    "endpoint": AWS_OTLP_ENDPOINT,
                    "headers": {},
                    "service_name": web_settings.operator_telemetry_service_name,
                    "service_version": identity["service.version"],
                    "deployment_environment": web_settings.operator_telemetry_environment,
                    "cloud_provider": "aws",
                    "aws_ecs_cluster_name": identity["aws.ecs.cluster.name"],
                    "aws_ecs_service_name": identity["aws.ecs.service.name"],
                    "aws_ecs_task_family": identity["aws.ecs.task.family"],
                    "aws_ecs_task_revision": identity["aws.ecs.task.revision"],
                    "batch_size": 100,
                },
            )
        ],
    )
    return settings.model_copy(update={"telemetry": effective})


def reset_operator_telemetry_for_tests() -> None:
    """Forget the module runtime; callers must inject factories after reset."""

    global _runtime
    with _runtime_lock:
        _runtime = None


__all__ = [
    "AWS_OTLP_ENDPOINT",
    "SAFE_CLOUDWATCH_METRIC_ATTRIBUTES",
    "OperatorTelemetryFactories",
    "OperatorTelemetryRuntime",
    "apply_operator_pipeline_telemetry",
    "bootstrap_operator_telemetry",
    "reset_operator_telemetry_for_tests",
]
