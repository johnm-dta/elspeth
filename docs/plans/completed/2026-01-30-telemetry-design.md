# Telemetry Design: Two-Tier Observability for ELSPETH

> **Priority:** P2 (Operational visibility, complements audit trail)
> **Effort:** Medium (2-3 sessions)
> **Risk:** Low (additive feature, existing EventBus pattern, failure isolation)
> **Status:** Design approved by 4-perspective panel
> **Revision:** v3 - Final revision incorporating re-review feedback (Warning Fatigue fix, default change)

## Executive Summary

ELSPETH needs operational visibility alongside its audit trail. This design introduces a **two-tier telemetry model**:

1. **Global Telemetry (Tier 1)** — Framework-level audit events streamed to external observability platforms via pluggable exporters
2. **Plugin Telemetry (Tier 2)** — Plugin-internal tracing handled autonomously by individual plugins (Azure AI, Langfuse, etc.)

**Key principle:** Landscape remains the legal record. Telemetry is operational visibility.

**Critical design decision (from Systems Thinking review):** Telemetry can be configured to either drop events under load OR apply backpressure. Silent failures are logged loudly, not hidden.

---

## Review Panel Summary

This design was reviewed by a 4-perspective panel across two review rounds:

### Round 1 (v1 → v2)
| Reviewer | Verdict | Key Feedback |
|----------|---------|--------------|
| **Architecture Critic** | ✅ Approve | Design aligns with existing patterns |
| **Python Engineering** | ⚠️ Request Changes | Fix `slots=True`, BoundedBuffer logic, Protocol `name` attribute |
| **Quality Assurance** | ⚠️ Request Changes | Add property-based tests for circuit breaker, buffer, ordering |
| **Systems Thinking** | ⚠️ Request Changes | Replace silent degradation, add backpressure option |

### Round 2 (v2 → v3)
| Reviewer | Verdict | Key Feedback |
|----------|---------|--------------|
| **Architecture Critic** | ⚠️ Request Changes | Backpressure modes `block`/`slow` not implemented - fail fast |
| **Python Engineering** | ✅ Approve | All critical issues fixed |
| **Quality Assurance** | ✅ Approve | All test gaps addressed |
| **Systems Thinking** | ⚠️ Request Changes | Default should be `block`; per-event logging creates Warning Fatigue |

### Final Resolution (v3)
All issues addressed:
- ✅ Python patterns correct (`frozen=True, slots=True`, BoundedBuffer, Protocol)
- ✅ Test strategy comprehensive (property-based, regression, contract tests)
- ✅ Backpressure modes: fail fast at startup if mode != `drop` (until implemented)
- ✅ Default changed to `block` (completeness by default, data loss opt-in)
- ✅ Warning Fatigue fixed: aggregate logging instead of per-event

---

## Problem Statement

ELSPETH's Landscape audit trail provides complete, legally defensible records of pipeline execution. However:

- **No real-time visibility** — Operators can't monitor pipelines during execution
- **No integration with existing observability stacks** — Teams using Datadog, Grafana, Azure Monitor have no native integration
- **No LLM-specific tracing** — LLM calls aren't visible in provider tracing tools (Azure AI, Langfuse)

### What We're NOT Solving

- Replacing Landscape (it remains the source of truth)
- Managing plugin-internal tracing from the framework level

### Design Philosophy (from Systems Thinking Review)

**Avoiding "Shifting the Burden":** Telemetry complements Landscape query tooling, it doesn't replace it. Operators should learn to use `explain()` and Landscape queries for investigations; telemetry provides real-time dashboards and alerting.

**Avoiding "Trust Erosion":** If telemetry drops events, operators lose trust and stop using it. Therefore:
- **Default is `block` mode** — completeness by default, data loss is opt-in
- Telemetry completeness is configurable (backpressure vs. drop)
- Dropped events are logged with aggregate metrics (not per-event to avoid Warning Fatigue)
- Silent failures are prohibited

**Avoiding "Warning Fatigue":** Per-event logging of dropped events creates noise that operators tune out. Therefore:
- Aggregate logging: log every 100 drops, not every single drop
- Include drop rate trends, not just counts
- CRITICAL alerts only for sustained total failures

---

## Architecture Overview

### Two-Tier Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ELSPETH Framework                            │
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │   Engine/    │────▶│  Landscape   │     │      EventBus        │ │
│  │ Orchestrator │     │  Recorder    │     │  (existing)          │ │
│  └──────┬───────┘     └──────────────┘     └──────────┬───────────┘ │
│         │              (audit storage)                │             │
│         │                                             │             │
│         └───────────── TelemetryEvent ───────────────▶│             │
│                                                       │             │
│                              ┌────────────────────────┼─────────────┐
│                              │   Telemetry Exporters  │             │
│                              │   (pluggy plugins)     ▼             │
│                              │  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│                              │  │  OTLP   │ │  Azure  │ │ Console │ │
│                              │  │Exporter │ │ Monitor │ │Exporter │ │
│                              │  └────┬────┘ └────┬────┘ └────┬────┘ │
│                              └───────┼──────────┼───────────┼──────┘│
└──────────────────────────────────────┼──────────┼───────────┼───────┘
                                       ▼          ▼           ▼
                               ┌───────────┐ ┌─────────┐ ┌────────┐
                               │OTLP       │ │Azure    │ │stdout  │
                               │Collector  │ │Monitor  │ │        │
                               └───────────┘ └─────────┘ └────────┘
```

### Key Principles

1. **Landscape is the legal record** — Telemetry events are emitted AFTER successful Landscape recording
2. **EventBus as integration point** — Reuses existing pattern from CLI observability
3. **Exporters are pluggy plugins** — Same registration pattern as sources/transforms/sinks
4. **Full fidelity, configurable filtering** — Capture everything, let operators dial down
5. **Configurable failure handling** — Operators choose: backpressure (complete) or drop (fast)
6. **Loud failures** — No silent degradation; all failures logged with metrics

### Out of Scope for Framework

Plugin-internal telemetry (LLM tracing to Azure AI, Langfuse, etc.) — plugins handle this themselves as T1 code. The framework provides nothing for Tier 2.

---

## Telemetry Event Types

Events mirror what Landscape records, designed for streaming consumption.

### Event Hierarchy

```python
# Base event - all telemetry events inherit from this
# NOTE: All events use frozen=True, slots=True for memory efficiency (Python Engineering review)
@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    timestamp: datetime
    run_id: str

# Lifecycle events (low volume)
@dataclass(frozen=True, slots=True)
class RunStarted(TelemetryEvent):
    config_hash: str
    source_plugin: str

@dataclass(frozen=True, slots=True)
class RunCompleted(TelemetryEvent):
    status: RunStatus
    row_count: int
    duration_ms: float

@dataclass(frozen=True, slots=True)
class PhaseChanged(TelemetryEvent):
    phase: PipelinePhase
    action: PhaseAction

# Row-level events (medium volume)
@dataclass(frozen=True, slots=True)
class RowCreated(TelemetryEvent):
    row_id: str
    token_id: str
    content_hash: str

@dataclass(frozen=True, slots=True)
class TransformCompleted(TelemetryEvent):
    row_id: str
    token_id: str
    node_id: str
    plugin_name: str
    status: NodeStateStatus
    duration_ms: float
    input_hash: str
    output_hash: str

@dataclass(frozen=True, slots=True)
class GateEvaluated(TelemetryEvent):
    row_id: str
    token_id: str
    node_id: str
    plugin_name: str
    routing_mode: RoutingMode
    destinations: tuple[str, ...]

# NOTE: Renamed from TokenOutcome to avoid collision with TokenOutcome enum (Python Engineering review)
@dataclass(frozen=True, slots=True)
class TokenCompleted(TelemetryEvent):
    row_id: str
    token_id: str
    outcome: RowOutcome  # Using the enum, not a same-named class
    sink_name: str | None

# External call events (high volume when enabled)
@dataclass(frozen=True, slots=True)
class ExternalCallCompleted(TelemetryEvent):
    state_id: str
    call_type: CallType  # LLM, HTTP, etc.
    provider: str
    status: CallStatus
    latency_ms: float
    # Optional high-fidelity fields (configurable)
    request_hash: str | None = None
    response_hash: str | None = None
    token_usage: dict[str, int] | None = None  # For LLM calls
```

### Granularity Levels

| Level | Events Emitted | Typical Volume |
|-------|----------------|----------------|
| `lifecycle` | `RunStarted`, `RunCompleted`, `PhaseChanged` | ~10-20 per run |
| `rows` | Above + `RowCreated`, `TransformCompleted`, `GateEvaluated`, `TokenCompleted` | N × M (rows × transforms) |
| `full` | Above + `ExternalCallCompleted` with all details | High (includes all external calls) |

---

## Exporter Protocol & Plugin Registration

### Protocol Definition

```python
# src/elspeth/telemetry/protocols.py

from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class ExporterProtocol(Protocol):
    """Protocol for telemetry exporters.

    Note on 'name' attribute: Implementations MUST define a class attribute
    'name: str' for configuration reference. This is enforced via @property
    in the protocol since class attributes cannot be checked via Protocol.

    Lifecycle:
    - configure() is called once at startup. MUST raise on invalid config (fail fast).
    - export() is called synchronously for each event. MUST NOT raise (log errors instead).
    - flush() is called at run completion. SHOULD be blocking until buffer is empty.
    - close() is called at shutdown. MUST release all resources.
    """

    @property
    def name(self) -> str:
        """Exporter name for configuration reference."""
        ...

    def configure(self, config: dict[str, Any]) -> None:
        """Configure exporter with deployment-specific settings.

        Called once at startup. MUST raise on invalid config (fail fast).
        MAY log warnings for transient connection issues.

        Raises:
            ValueError: If config is invalid or incomplete
            ConnectionError: If endpoint is unreachable (optional - may defer)
        """
        ...

    def export(self, event: TelemetryEvent) -> None:
        """Export a single event.

        Called synchronously from EventBus. Implementations should:
        - Buffer internally if needed for batching
        - Log failures, NEVER raise (would break other exporters)
        - Track metrics: events_exported, events_failed

        Thread safety: Assumes single-threaded EventBus dispatch.
        If used in multithreaded context, implementation must add locking.
        """
        ...

    def flush(self) -> None:
        """Flush any buffered events.

        Called at run completion and shutdown. SHOULD block until
        all buffered events are sent or timeout reached.
        """
        ...

    def close(self) -> None:
        """Release resources.

        Called at shutdown. Close connections, stop background threads.
        MUST be idempotent (safe to call multiple times).
        """
        ...
```

### Hook Specification

```python
# src/elspeth/telemetry/hookspecs.py

from pluggy import HookspecMarker

hookspec = HookspecMarker("elspeth")

class ElspethTelemetrySpec:
    """Hook specifications for telemetry exporters."""

    @hookspec
    def elspeth_get_exporters(self) -> list[type[ExporterProtocol]]:
        """Return telemetry exporter classes."""
```

### Registration Pattern

```python
# src/elspeth/telemetry/exporters/otlp.py

from elspeth.telemetry.hookspecs import hookimpl

class OTLPExporter:
    _name = "otlp"  # Class attribute for storage

    @property
    def name(self) -> str:
        """Exporter name for configuration reference."""
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        if "endpoint" not in config:
            raise ValueError("OTLP exporter requires 'endpoint' in config")
        self._endpoint = config["endpoint"]
        self._headers = config.get("headers", {})
        # Initialize OTLP client...

class OTLPExporterPlugin:
    @hookimpl
    def elspeth_get_exporters(self) -> list[type[ExporterProtocol]]:
        return [OTLPExporter]
```

---

## Configuration

### Secrets Handling

**All secrets MUST come from environment variables, never hardcoded in config files.**

ELSPETH uses Dynaconf's `${ENV_VAR}` substitution pattern. Secrets are resolved at runtime from:
1. **Environment variables** (highest priority)
2. **`.env` file** in project root (development convenience)
3. **`.secrets.yaml`** (optional, gitignored)

| Secret Type | Environment Variable | Example |
|-------------|---------------------|---------|
| OTLP auth token | `OTEL_TOKEN` | `options.headers.Authorization: "Bearer ${OTEL_TOKEN}"` |
| Azure Monitor | `APPLICATIONINSIGHTS_CONNECTION_STRING` | `options.connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}` |
| Datadog API key | `DD_API_KEY` | `options.api_key: ${DD_API_KEY}` |

**Security rules:**
- Config files (`.yaml`) go in git — they contain structure, not secrets
- `.env` files are gitignored — they contain secrets for local dev
- Production secrets come from environment (injected by deployment system)
- Never log secrets — use HMAC fingerprints per CLAUDE.md

### User-Facing Settings

```yaml
# settings.yaml
telemetry:
  enabled: true
  granularity: full  # lifecycle | rows | full

  # Backpressure mode (Systems Thinking review recommendation)
  # - block: Pause pipeline when buffer full until exporter catches up [DEFAULT]
  #          (run completes, but may be slower under high telemetry load)
  # - drop: Drop oldest events when buffer full (fast, may lose events)
  # - slow: Apply gradual backpressure to slow emission rate [NOT YET IMPLEMENTED]
  #
  # NOTE: Default is 'block' to ensure completeness. Data loss is opt-in.
  # NOTE: 'slow' mode not yet implemented - will fail fast at startup if selected.
  # IMPORTANT: 'block' pauses processing temporarily - it does NOT terminate the run.
  backpressure_mode: block  # block | drop | slow

  # Failure handling
  fail_on_total_exporter_failure: false  # If true, crash run when all exporters fail

  exporters:
    - name: otlp
      options:
        endpoint: ${OTEL_ENDPOINT}  # e.g., "http://localhost:4317"
        headers:
          Authorization: "Bearer ${OTEL_TOKEN}"

    - name: azure_monitor
      options:
        connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}

    - name: datadog
      options:
        api_key: ${DD_API_KEY}  # Optional if local agent
        service_name: "elspeth-pipeline"

    - name: console
      options:
        format: json  # json | pretty (for local debugging)
```

### Protocol (contracts/config/protocols.py)

```python
@runtime_checkable
class RuntimeTelemetryProtocol(Protocol):
    """What TelemetryManager requires for event streaming."""
    enabled: bool
    granularity: TelemetryGranularity
    backpressure_mode: BackpressureMode
    fail_on_total_exporter_failure: bool
    exporter_configs: tuple[ExporterConfig, ...]
```

### Runtime Dataclass (contracts/config/runtime.py)

```python
# NOTE: Use (str, Enum) for database/JSON serialization consistency (Python Engineering review)
class TelemetryGranularity(str, Enum):
    LIFECYCLE = "lifecycle"
    ROWS = "rows"
    FULL = "full"

class BackpressureMode(str, Enum):
    BLOCK = "block"    # Pause pipeline until buffer drains [DEFAULT] (run continues, just slower)
    DROP = "drop"      # Drop oldest events when buffer full (fast, events lost)
    SLOW = "slow"      # Gradual backpressure on emission rate [NOT IMPLEMENTED]

# Modes that are implemented
_IMPLEMENTED_MODES = {BackpressureMode.BLOCK, BackpressureMode.DROP}

@dataclass(frozen=True, slots=True)
class ExporterConfig:
    """Configuration for a single exporter."""
    name: str
    options: dict[str, Any]

@dataclass(frozen=True, slots=True)
class RuntimeTelemetryConfig:
    """Runtime telemetry configuration.

    Implements RuntimeTelemetryProtocol for use by TelemetryManager.

    Field Mappings (Settings → Runtime):
        enabled ← enabled
        granularity ← granularity (parsed to enum)
        backpressure_mode ← backpressure_mode (parsed to enum)
        fail_on_total_exporter_failure ← fail_on_total_exporter_failure
        exporter_configs ← exporters (converted to ExporterConfig tuple)
    """
    enabled: bool
    granularity: TelemetryGranularity
    backpressure_mode: BackpressureMode
    fail_on_total_exporter_failure: bool
    exporter_configs: tuple[ExporterConfig, ...]

    @classmethod
    def from_settings(cls, settings: "TelemetrySettings") -> "RuntimeTelemetryConfig":
        """Convert user-facing Settings to runtime config.

        Raises:
            NotImplementedError: If backpressure_mode is not yet implemented (e.g., 'slow')
            ValueError: If granularity or backpressure_mode is invalid
        """
        try:
            granularity = TelemetryGranularity(settings.granularity.lower())
            backpressure_mode = BackpressureMode(settings.backpressure_mode.lower())
        except ValueError as e:
            raise ValueError(
                f"Invalid telemetry configuration: {e}. "
                f"granularity must be one of {[g.value for g in TelemetryGranularity]}, "
                f"backpressure_mode must be one of {[b.value for b in BackpressureMode]}"
            ) from e

        # Fail fast on unimplemented modes (Architecture review requirement)
        if backpressure_mode not in _IMPLEMENTED_MODES:
            raise NotImplementedError(
                f"backpressure_mode='{backpressure_mode.value}' is not yet implemented. "
                f"Use one of: {[m.value for m in _IMPLEMENTED_MODES]}"
            )

        return cls(
            enabled=settings.enabled,
            granularity=granularity,
            backpressure_mode=backpressure_mode,
            fail_on_total_exporter_failure=settings.fail_on_total_exporter_failure,
            exporter_configs=tuple(
                ExporterConfig(name=e.name, options=e.options)
                for e in settings.exporters
            ),
        )

    @classmethod
    def disabled(cls) -> "RuntimeTelemetryConfig":
        """Configuration that disables telemetry."""
        return cls(
            enabled=False,
            granularity=TelemetryGranularity.LIFECYCLE,
            backpressure_mode=BackpressureMode.BLOCK,  # Default is block (completeness)
            fail_on_total_exporter_failure=False,
            exporter_configs=(),
        )
```

### Granularity Filtering

```python
# NOTE: Use match statement for Python 3.10+ (Python Engineering review)
def should_emit(event: TelemetryEvent, granularity: TelemetryGranularity) -> bool:
    """Filter events based on configured granularity."""
    match event:
        case RunStarted() | RunCompleted() | PhaseChanged():
            return True  # Always emit lifecycle
        case RowCreated() | TransformCompleted() | GateEvaluated() | TokenCompleted():
            return granularity in (TelemetryGranularity.ROWS, TelemetryGranularity.FULL)
        case ExternalCallCompleted():
            return granularity == TelemetryGranularity.FULL
        case _:
            return True  # Unknown events pass through
```

---

## Event Emission Points

### TelemetryManager

```python
# src/elspeth/telemetry/manager.py

class TelemetryManager:
    """Coordinates event emission to configured exporters.

    Subscribes to EventBus, filters by granularity, dispatches to exporters.

    Failure handling (per Systems Thinking review):
    - Individual exporter failures: Log warning, continue to other exporters
    - All exporters fail: Log ERROR (aggregate), optionally crash run (configurable)
    - NO silent degradation: All failures are visible in logs and metrics
    - WARNING FATIGUE PREVENTION: Aggregate logging every 100 total failures

    Thread safety: Assumes single-threaded EventBus dispatch from Orchestrator.
    """

    # Log aggregate metrics every N total failures to avoid Warning Fatigue
    _LOG_INTERVAL = 100

    def __init__(
        self,
        config: RuntimeTelemetryProtocol,
        event_bus: EventBusProtocol,
        exporters: list[ExporterProtocol],
    ) -> None:
        self._config = config
        self._exporters = exporters
        self._consecutive_total_failures = 0
        self._max_consecutive_failures = 10

        # Telemetry health metrics (Systems Thinking review)
        self._events_emitted = 0
        self._events_dropped = 0
        self._exporter_failures: dict[str, int] = {}
        self._last_logged_drop_count: int = 0  # For aggregate logging

        if config.enabled:
            for event_type in TELEMETRY_EVENT_TYPES:
                event_bus.subscribe(event_type, self._handle_event)

    def _handle_event(self, event: TelemetryEvent) -> None:
        """Filter and dispatch event to all exporters.

        Logging strategy (Warning Fatigue prevention):
        - Individual exporter failures: Log immediately (actionable per-exporter)
        - All-exporter total failures: Log every 100 drops (aggregate)
        - Threshold breaches: Log CRITICAL immediately (rare, critical)
        """
        if not should_emit(event, self._config.granularity):
            return

        failures = 0
        for exporter in self._exporters:
            try:
                exporter.export(event)
            except Exception as e:
                failures += 1
                self._exporter_failures[exporter.name] = (
                    self._exporter_failures.get(exporter.name, 0) + 1
                )
                # Log individual exporter failures at WARNING
                # (These are actionable - operator can check specific exporter)
                logger.warning(
                    "Telemetry exporter failed",
                    exporter=exporter.name,
                    event_type=type(event).__name__,
                    error=str(e),
                    total_failures=self._exporter_failures[exporter.name],
                )

        if failures == 0:
            self._events_emitted += 1
            self._consecutive_total_failures = 0
        elif failures == len(self._exporters):
            # ALL exporters failed - this is serious (Systems Thinking review)
            self._consecutive_total_failures += 1
            self._events_dropped += 1

            # Aggregate logging: log every _LOG_INTERVAL drops to avoid Warning Fatigue
            # (Per-event ERROR logging creates noise that operators tune out)
            if self._events_dropped - self._last_logged_drop_count >= self._LOG_INTERVAL:
                logger.error(
                    "ALL telemetry exporters failing - events dropped",
                    dropped_since_last_log=self._events_dropped - self._last_logged_drop_count,
                    dropped_total=self._events_dropped,
                    consecutive_failures=self._consecutive_total_failures,
                    exporter_failure_counts=self._exporter_failures,
                )
                self._last_logged_drop_count = self._events_dropped

            if self._consecutive_total_failures >= self._max_consecutive_failures:
                if self._config.fail_on_total_exporter_failure:
                    # Crash loudly per Systems Thinking recommendation
                    raise TelemetryExporterError(
                        f"All {len(self._exporters)} exporters failed "
                        f"{self._max_consecutive_failures} consecutive times. "
                        f"Telemetry is broken. Check exporter configuration."
                    )
                else:
                    # Log CRITICAL immediately (this is rare and actionable)
                    logger.critical(
                        "Telemetry disabled after repeated total failures",
                        consecutive_failures=self._consecutive_total_failures,
                        events_dropped=self._events_dropped,
                        hint="Set fail_on_total_exporter_failure=true to crash instead",
                    )
        else:
            # Partial success - some exporters worked
            self._events_emitted += 1
            self._consecutive_total_failures = 0

    @property
    def health_metrics(self) -> dict[str, Any]:
        """Return telemetry health metrics for monitoring."""
        return {
            "events_emitted": self._events_emitted,
            "events_dropped": self._events_dropped,
            "exporter_failures": self._exporter_failures.copy(),
            "consecutive_total_failures": self._consecutive_total_failures,
        }

    def flush(self) -> None:
        """Flush all exporters."""
        for exporter in self._exporters:
            try:
                exporter.flush()
            except Exception as e:
                logger.warning(
                    "Telemetry exporter flush failed",
                    exporter=exporter.name,
                    error=str(e),
                )

    def close(self) -> None:
        """Close all exporters and log final metrics."""
        # Log final health metrics
        logger.info(
            "Telemetry manager closing",
            **self.health_metrics,
        )

        for exporter in self._exporters:
            try:
                exporter.close()
            except Exception as e:
                logger.warning(
                    "Telemetry exporter close failed",
                    exporter=exporter.name,
                    error=str(e),
                )
```

### Emission Points in Orchestrator

Events are emitted AFTER successful Landscape recording:

| Orchestrator Method | Landscape Call | Telemetry Event |
|---------------------|----------------|-----------------|
| `_begin_run()` | `recorder.begin_run()` | `RunStarted` |
| `_complete_run()` | `recorder.complete_run()` | `RunCompleted` |
| `_process_source_row()` | `recorder.record_row()` | `RowCreated` |
| `_execute_transform()` | `recorder.record_node_state()` | `TransformCompleted` |
| `_execute_gate()` | `recorder.record_routing_event()` | `GateEvaluated` |
| `_record_token_outcome()` | `recorder.record_token_outcome()` | `TokenCompleted` |

### Emission Points in AuditedLLMClient

```python
def chat_completion(self, ...) -> LLMResponse:
    start = time.perf_counter()
    try:
        response = self._client.chat.completions.create(...)
        latency_ms = (time.perf_counter() - start) * 1000

        # Record to Landscape FIRST (existing)
        self._record_call(request, response, CallStatus.SUCCESS, latency_ms)

        # Emit telemetry event AFTER (new)
        self._event_bus.emit(ExternalCallCompleted(
            timestamp=now(),
            run_id=self._run_id,
            state_id=self._state_id,
            call_type=CallType.LLM,
            provider=self._provider,
            status=CallStatus.SUCCESS,
            latency_ms=latency_ms,
            token_usage=response.usage.model_dump() if response.usage else None,
        ))

        return LLMResponse(...)
    except Exception as e:
        # Similar pattern for failures
        ...
```

---

## Default Exporters

### OTLP Exporter

```python
# src/elspeth/telemetry/exporters/otlp.py

class OTLPExporter:
    """Export telemetry events via OpenTelemetry Protocol.

    Converts ELSPETH TelemetryEvents to OTLP spans and ships
    to any OTLP-compatible backend (Jaeger, Tempo, Datadog, Honeycomb, etc.)

    Thread safety: Assumes single-threaded access. Buffer is not thread-safe.
    """
    _name = "otlp"

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        if "endpoint" not in config:
            raise ValueError("OTLP exporter requires 'endpoint' in config")

        self._endpoint = config["endpoint"]
        self._headers = config.get("headers", {})
        self._batch_size = config.get("batch_size", 100)
        self._flush_interval_ms = config.get("flush_interval_ms", 5000)

        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        self._span_exporter = OTLPSpanExporter(
            endpoint=self._endpoint,
            headers=self._headers,
        )
        self._buffer: list[TelemetryEvent] = []

    def export(self, event: TelemetryEvent) -> None:
        self._buffer.append(event)
        if len(self._buffer) >= self._batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        if not self._buffer:
            return
        spans = [self._event_to_span(e) for e in self._buffer]
        self._span_exporter.export(spans)
        self._buffer.clear()

    def _event_to_span(self, event: TelemetryEvent) -> "Span":
        """Convert TelemetryEvent to OpenTelemetry Span.

        Mapping:
        - span.name = event class name (e.g., "TransformCompleted")
        - span.start_time = event.timestamp
        - span.attributes = all event fields as attributes
        - span.trace_id = derived from run_id (consistent within run)
        - span.span_id = derived from event-specific IDs
        """
        # Implementation details...
        ...

    def flush(self) -> None:
        self._flush_batch()

    def close(self) -> None:
        self.flush()
        self._span_exporter.shutdown()
```

### Azure Monitor Exporter

```python
# src/elspeth/telemetry/exporters/azure_monitor.py

class AzureMonitorExporter:
    """Export telemetry to Azure Monitor / Application Insights.

    Uses azure-monitor-opentelemetry-exporter for native integration.
    """
    _name = "azure_monitor"

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        if "connection_string" not in config:
            raise ValueError("Azure Monitor exporter requires 'connection_string' in config")

        self._connection_string = config["connection_string"]

        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
        self._exporter = AzureMonitorTraceExporter(
            connection_string=self._connection_string,
        )
        self._buffer: list[TelemetryEvent] = []
        self._batch_size = config.get("batch_size", 100)

    # ... similar pattern to OTLP
```

### Datadog Exporter

```python
# src/elspeth/telemetry/exporters/datadog.py

class DatadogExporter:
    """Export telemetry to Datadog via native API.

    Uses ddtrace for native Datadog integration with full APM features.

    Config values come pre-resolved by Dynaconf (${DD_API_KEY} → actual value).
    """
    _name = "datadog"

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        # api_key is optional if using local Datadog agent
        # Value comes from config after Dynaconf resolves ${DD_API_KEY}
        self._api_key = config.get("api_key")
        self._service_name = config.get("service_name", "elspeth")
        self._env = config.get("env", "production")

        from ddtrace import tracer
        tracer.configure(
            hostname=config.get("agent_host", "localhost"),
            port=config.get("agent_port", 8126),
        )
        self._tracer = tracer
```

### Console Exporter

```python
# src/elspeth/telemetry/exporters/console.py

class ConsoleExporter:
    """Export telemetry events to stdout for testing/debugging."""
    _name = "console"

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        self._format = config.get("format", "json")  # json | pretty
        self._output = config.get("output", "stdout")  # stdout | stderr

    def export(self, event: TelemetryEvent) -> None:
        if self._format == "json":
            # Custom serialization for datetime handling
            line = json.dumps(self._serialize_event(event))
        else:
            line = f"[{event.timestamp.isoformat()}] {type(event).__name__}: {event.run_id}"

        stream = sys.stdout if self._output == "stdout" else sys.stderr
        print(line, file=stream)

    def _serialize_event(self, event: TelemetryEvent) -> dict[str, Any]:
        """Serialize event for JSON output with proper type handling."""
        from dataclasses import asdict
        data = asdict(event)
        # Convert datetime to ISO format
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
```

---

## Buffer Management

### BoundedBuffer (Fixed per Python Engineering + Systems Thinking reviews)

```python
class BoundedBuffer:
    """Ring buffer that drops oldest events on overflow.

    NOTE: The overflow counting was fixed per Python Engineering review.
    The deque automatically drops the oldest item when maxlen is reached,
    so we detect drops by comparing length before and after append.

    NOTE: Aggregate logging per Systems Thinking review - logs every 100 drops
    instead of per-event to avoid Warning Fatigue.
    """

    # Log aggregate metrics every N drops to avoid Warning Fatigue
    _LOG_INTERVAL = 100

    def __init__(self, max_size: int = 10_000) -> None:
        self._buffer: deque[TelemetryEvent] = deque(maxlen=max_size)
        self._dropped_count: int = 0
        self._last_logged_drop_count: int = 0

    def append(self, event: TelemetryEvent) -> None:
        """Append event to buffer, tracking drops correctly.

        Logging strategy (Warning Fatigue prevention):
        - Logs every 100 drops, not every single drop
        - Includes drop rate trend information
        """
        was_full = len(self._buffer) == self._buffer.maxlen
        self._buffer.append(event)
        if was_full:
            # deque auto-dropped the oldest item
            self._dropped_count += 1

            # Aggregate logging: log every _LOG_INTERVAL drops
            if self._dropped_count - self._last_logged_drop_count >= self._LOG_INTERVAL:
                logger.warning(
                    "Telemetry buffer overflow - events dropped",
                    dropped_since_last_log=self._LOG_INTERVAL,
                    dropped_total=self._dropped_count,
                    buffer_size=self._buffer.maxlen,
                    hint="Consider increasing buffer size or reducing granularity",
                )
                self._last_logged_drop_count = self._dropped_count

    def pop_batch(self, max_count: int) -> list[TelemetryEvent]:
        """Pop up to max_count events from the buffer."""
        batch = []
        for _ in range(min(max_count, len(self._buffer))):
            batch.append(self._buffer.popleft())
        return batch

    @property
    def dropped_count(self) -> int:
        """Number of events dropped due to buffer overflow."""
        return self._dropped_count

    def __len__(self) -> int:
        return len(self._buffer)
```

---

## Plugin-Specific Telemetry (Tier 2)

The framework provides **nothing** for plugin-internal tracing. Plugins are T1 code and handle their own integrations.

### What This Means

| Aspect | Framework Role | Plugin Role |
|--------|----------------|-------------|
| Configuration | None | Plugin defines its own telemetry config fields |
| Dependencies | None | Plugin brings its own SDK (Azure AI, Langfuse, etc.) |
| Initialization | None | Plugin initializes tracing in `__init__` or `on_start()` |
| Instrumentation | None | Plugin wraps its own calls with provider SDK |
| Lifecycle | None | Plugin cleans up in `close()` |

### Example: Azure LLM Plugin

```python
class AzureOpenAIConfig(LLMConfig):
    # Plugin-specific telemetry config (framework doesn't interpret this)
    azure_tracing: AzureTracingConfig | None = None

@dataclass
class AzureTracingConfig:
    enabled: bool = False
    connection_string: str | None = None

class AzureLLMTransform(BaseTransform, BatchTransformMixin):

    def on_start(self, ctx: PluginContext) -> None:
        super().on_start(ctx)

        if self._config.azure_tracing and self._config.azure_tracing.enabled:
            self._setup_azure_tracing()

    def _setup_azure_tracing(self) -> None:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(
            connection_string=self._config.azure_tracing.connection_string,
            enable_live_metrics=True,
        )
```

### Example Configuration

```yaml
transforms:
  - plugin: azure_llm
    options:
      deployment_name: gpt-4
      endpoint: ${AZURE_OPENAI_ENDPOINT}
      api_key: ${AZURE_OPENAI_KEY}

      # Plugin-specific telemetry (framework ignores this)
      azure_tracing:
        enabled: true
        connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
```

---

## Error Handling & Failure Modes

### Principle (Updated per Systems Thinking Review)

**Telemetry completeness is configurable. Silent failures are prohibited.**

### Failure Behavior

| Failure Mode | Behavior | Rationale |
|--------------|----------|-----------|
| Single exporter throws | Log WARNING, continue to other exporters | Partial success is acceptable |
| All exporters fail once | Log ERROR with metrics, continue | Transient failures happen |
| All exporters fail 10× consecutive | Log CRITICAL, optionally crash | Sustained failure = broken config |
| Exporter slow (backpressure) | Per config: block (pause), drop (discard), or slow | Operator chooses tradeoff |
| Invalid config | Fail fast at startup | Config errors should crash early |
| Buffer overflow | Log WARNING per event dropped | Operator sees backpressure |

### No Silent Degradation

Per CLAUDE.md: "A defective plugin that silently produces wrong results is worse than a crash."

The original design had "silent disable after 10 failures." This is replaced with:
1. **Log CRITICAL** when reaching failure threshold
2. **Optionally crash** (configurable via `fail_on_total_exporter_failure`)
3. **Always report metrics** (`events_dropped`, `exporter_failures`)

---

## Package Structure

```
src/elspeth/telemetry/
├── __init__.py           # Re-exports
├── events.py             # TelemetryEvent dataclasses
├── protocols.py          # ExporterProtocol
├── hookspecs.py          # elspeth_get_exporters hook
├── manager.py            # TelemetryManager (EventBus subscriber)
├── filtering.py          # Granularity filtering logic
├── buffer.py             # BoundedBuffer implementation
├── errors.py             # TelemetryExporterError
└── exporters/
    ├── __init__.py       # Exporter plugin registration
    ├── otlp.py           # OTLP exporter
    ├── azure_monitor.py  # Azure Monitor exporter
    ├── datadog.py        # Datadog exporter
    └── console.py        # Console exporter (testing)
```

---

## Implementation Tasks

### Phase 0: Performance Baseline (NEW - per Systems Thinking review)

**Task 0.1: Benchmark EventBus dispatch overhead**
- Measure time for `event_bus.emit(event)` with 0, 1, 3 exporters
- Measure impact on pipeline throughput (rows/sec) with telemetry enabled vs disabled
- Document: "Telemetry adds X% overhead at granularity=full"
- **Acceptance:** Overhead < 5% at granularity=rows, < 10% at granularity=full

### Phase 1: Core Infrastructure

**Task 1.1: Create telemetry package structure**
- Create `src/elspeth/telemetry/` directory
- Create `__init__.py`, `events.py`, `protocols.py`, `hookspecs.py`
- Define `TelemetryEvent` base class and event dataclasses (with `frozen=True, slots=True`)
- Define `ExporterProtocol` (with `@property name`)
- Define `elspeth_get_exporters` hook specification

**Task 1.2: Add configuration contracts**
- Add `TelemetryGranularity` enum to `contracts/enums.py` (as `str, Enum`)
- Add `BackpressureMode` enum to `contracts/enums.py`
- Add `RuntimeTelemetryProtocol` to `contracts/config/protocols.py`
- Add `RuntimeTelemetryConfig` to `contracts/config/runtime.py`
- Add `TelemetrySettings` to `core/config.py`
- Update `ElspethSettings` to include telemetry settings

**Task 1.3: Implement BoundedBuffer**
- Create `buffer.py` with corrected overflow counting
- **Tests:** Property-based tests for buffer behavior (QA review)

**Task 1.4: Implement TelemetryManager**
- Create `manager.py` with `TelemetryManager` class
- Implement EventBus subscription
- Implement granularity filtering (with `match` statement)
- Implement failure handling with metrics (no silent degradation)
- **Tests:** Property-based state machine tests for circuit breaker (QA review)

**Task 1.5: Implement ConsoleExporter**
- Create `exporters/console.py`
- Implement JSON and pretty formats
- Register via pluggy hook
- **Tests:** Unit tests for output formatting

### Phase 2: Event Emission

**Task 2.1: Add telemetry events to Orchestrator**
- Inject `TelemetryManager` into `Orchestrator`
- Emit `RunStarted` after `recorder.begin_run()`
- Emit `RunCompleted` after `recorder.complete_run()`
- Emit `RowCreated` after `recorder.record_row()`
- **Tests:**
  - Integration test verifying events emitted
  - Regression test for Landscape-first ordering (QA review)

**Task 2.2: Add telemetry events to RowProcessor**
- Emit `TransformCompleted` after transform execution
- Emit `GateEvaluated` after gate evaluation
- Emit `TokenCompleted` after token outcome recording
- **Tests:** Integration test for row-level events

**Task 2.3: Add telemetry events to AuditedLLMClient**
- Inject EventBus into `AuditedLLMClient`
- Emit `ExternalCallCompleted` after LLM calls
- Include token usage for LLM calls
- **Tests:** Integration test for external call events

### Phase 3: Production Exporters

**Task 3.1: Implement OTLP Exporter**
- Add `opentelemetry-exporter-otlp` to optional dependencies
- Create `exporters/otlp.py`
- Implement event-to-span conversion
- Implement batching and flush
- Register via pluggy hook
- **Tests:** Unit tests with mock OTLP endpoint

**Task 3.2: Implement Azure Monitor Exporter**
- Add `azure-monitor-opentelemetry-exporter` to optional dependencies
- Create `exporters/azure_monitor.py`
- Implement Azure-specific span attributes
- Register via pluggy hook
- **Tests:** Unit tests with mock Azure endpoint

**Task 3.3: Implement Datadog Exporter**
- Add `ddtrace` to optional dependencies
- Create `exporters/datadog.py`
- Implement Datadog-specific tagging
- Register via pluggy hook
- **Tests:** Unit tests with mock Datadog agent

### Phase 4: Testing & Documentation

**Task 4.1: Add property-based tests (QA review requirements)**
- Circuit breaker state machine test (Hypothesis RuleBasedStateMachine)
- BoundedBuffer overflow behavior test
- Granularity filtering correctness test (all granularity × event combinations)
- Event ordering preservation test

**Task 4.2: Add integration tests**
- Test telemetry emitted alongside Landscape
- Regression test: telemetry only after Landscape success
- Test granularity filtering
- Test exporter failure isolation
- Test loud failure on total exporter failure
- High-volume flooding test (10k+ events)

**Task 4.3: Add contract tests**
- Test all exporters implement `ExporterProtocol` (including calling methods)
- Test `RuntimeTelemetryConfig` implements protocol
- Test config alignment (no orphaned fields)
- Test all TelemetryEvent dataclasses are JSON-serializable

**Task 4.4: Add EventBus tests (QA review)**
- Re-entrance test: handler emitting same-type event doesn't cause stack overflow

**Task 4.5: Update documentation**
- Add telemetry section to CLAUDE.md
- Create `docs/guides/telemetry.md` user guide
- Document exporter configuration options
- Document correlation workflow: "From Datadog Alert to Landscape Explain"

---

## Testing Strategy

### Unit Tests

| Component | Test Focus |
|-----------|------------|
| `TelemetryEvent` dataclasses | Serialization, JSON roundtrip, field completeness |
| `TelemetryManager` | Granularity filtering, exporter dispatch, failure handling, metrics |
| `BoundedBuffer` | Overflow counting, FIFO behavior, property-based invariants |
| `ConsoleExporter` | Output format, stream selection |
| Granularity filtering | Event type → granularity level mapping (all combinations) |

### Property-Based Tests (NEW - per QA review)

```python
# Circuit breaker state machine
class TelemetryManagerStateMachine(RuleBasedStateMachine):
    """Stateful testing for TelemetryManager failure handling."""

    @rule()
    def emit_event_all_exporters_fail(self):
        """Simulate all exporters failing on an event."""

    @rule()
    def emit_event_one_succeeds(self):
        """Simulate at least one exporter succeeding."""

    @invariant()
    def consecutive_failures_tracked_correctly(self):
        """Counter resets on partial success, increments on total failure."""


# Buffer overflow behavior
@given(
    buffer_size=st.integers(min_value=5, max_value=50),
    event_count=st.integers(min_value=10, max_value=200)
)
def test_buffer_drops_oldest_on_overflow(buffer_size, event_count):
    """Property: Buffer drops oldest events when full, counts correctly."""
    buffer = BoundedBuffer(max_size=buffer_size)

    for i in range(event_count):
        buffer.append(make_event(run_id=f"run_{i}"))

    assert len(buffer) == buffer_size
    expected_dropped = max(0, event_count - buffer_size)
    assert buffer.dropped_count == expected_dropped


# Granularity filtering correctness
@given(
    event_type=st.sampled_from([RunStarted, RowCreated, TransformCompleted, ExternalCallCompleted]),
    granularity=st.sampled_from(list(TelemetryGranularity))
)
def test_granularity_filtering_correctness(event_type, granularity):
    """Property: Filtering is consistent with granularity level."""
    event = make_event_of_type(event_type)
    result = should_emit(event, granularity)

    # Verify against specification
    if event_type in [RunStarted, RunCompleted, PhaseChanged]:
        assert result is True  # Always emit lifecycle
    elif event_type in [RowCreated, TransformCompleted]:
        assert result == (granularity in (TelemetryGranularity.ROWS, TelemetryGranularity.FULL))
    elif event_type == ExternalCallCompleted:
        assert result == (granularity == TelemetryGranularity.FULL)
```

### Integration Tests

```python
def test_telemetry_emitted_alongside_landscape():
    """Telemetry events mirror Landscape recording."""
    captured_events: list[TelemetryEvent] = []
    exporter = CapturingExporter(captured_events)

    run_pipeline_with_telemetry(exporter)

    assert any(isinstance(e, RunStarted) for e in captured_events)
    assert any(isinstance(e, RowCreated) for e in captured_events)
    assert any(isinstance(e, RunCompleted) for e in captured_events)


def test_telemetry_only_emitted_after_landscape_success():
    """Regression test: Telemetry emitted ONLY if Landscape recording succeeds."""
    captured_events = []

    # Mock recorder that fails on record_row()
    recorder = Mock(spec=LandscapeRecorder)
    recorder.record_row.side_effect = sqlite3.OperationalError("Disk full")

    orchestrator = Orchestrator(recorder=recorder, event_bus=capturing_bus(captured_events))

    with pytest.raises(sqlite3.OperationalError):
        orchestrator._process_source_row(row_data={"id": 1})

    # NO telemetry events should have been emitted (Landscape failed)
    row_created_events = [e for e in captured_events if isinstance(e, RowCreated)]
    assert len(row_created_events) == 0, "Telemetry emitted before Landscape commit!"


def test_telemetry_failure_logged_loudly():
    """Exporter failures are logged as ERROR when all fail."""
    failing_exporter = AlwaysFailsExporter()

    with capture_logs() as logs:
        result = run_pipeline_with_telemetry(failing_exporter)

    assert result.status == RunStatus.COMPLETED
    assert any("ALL telemetry exporters failed" in log for log in logs)
    assert any(log.level == "ERROR" for log in logs)


def test_high_volume_event_flooding():
    """10k+ events don't overflow or block pipeline."""
    captured_events = []
    exporter = CapturingExporter(captured_events)

    # Run pipeline with 10k rows
    result = run_pipeline_with_rows(10_000, exporter)

    assert result.status == RunStatus.COMPLETED
    # Should have emitted events for all rows (at rows granularity)
    row_events = [e for e in captured_events if isinstance(e, RowCreated)]
    assert len(row_events) == 10_000
```

### Contract Tests

```python
def test_exporter_protocol_compliance():
    """All shipped exporters implement ExporterProtocol with correct behavior."""
    for exporter_cls in [OTLPExporter, AzureMonitorExporter, DatadogExporter, ConsoleExporter]:
        instance = exporter_cls()

        # Test Protocol compliance
        assert isinstance(instance, ExporterProtocol)

        # Test name property works
        assert isinstance(instance.name, str)
        assert len(instance.name) > 0

        # Test configure accepts dict (use minimal valid config)
        instance.configure(get_minimal_config(instance.name))

        # Test export accepts event without raising
        dummy_event = RunStarted(
            timestamp=datetime.now(),
            run_id="test",
            config_hash="hash",
            source_plugin="csv"
        )
        instance.export(dummy_event)  # Should not raise

        instance.flush()
        instance.close()


def test_telemetry_events_json_serializable():
    """All TelemetryEvent dataclasses can be serialized to JSON."""
    for event_cls in TELEMETRY_EVENT_TYPES:
        event = make_event_of_type(event_cls)

        # Should serialize without error
        json_str = json.dumps(asdict(event), default=str)

        # Should deserialize back
        data = json.loads(json_str)
        assert "run_id" in data
        assert "timestamp" in data
```

---

## Success Criteria

### Phase 0 Complete (NEW)

- [ ] EventBus dispatch overhead measured and documented
- [ ] Overhead acceptable: < 5% at rows, < 10% at full

### Phase 1-2 Complete

- [ ] Telemetry events defined for all Landscape recording points (with `frozen=True, slots=True`)
- [ ] `TelemetryManager` subscribes to EventBus and dispatches to exporters
- [ ] Granularity filtering works correctly (using `match` statement)
- [ ] `ConsoleExporter` works for local testing
- [ ] Exporter failures logged loudly with metrics (no silent degradation)
- [ ] `fail_on_total_exporter_failure` option works
- [ ] BoundedBuffer correctly counts dropped events

### Phase 3-4 Complete

- [ ] OTLP, Azure Monitor, Datadog exporters ship
- [ ] All exporters implement `ExporterProtocol` (verified by calling methods)
- [ ] Configuration follows contracts pattern
- [ ] Property-based tests pass (circuit breaker, buffer, filtering)
- [ ] Regression test for Landscape-first ordering passes
- [ ] Integration tests pass including high-volume (10k+)
- [ ] Documentation complete

---

## Dependencies

- **Requires:** Config contracts refactor (in progress) — for `RuntimeTelemetryConfig` pattern
- **Requires:** Existing EventBus infrastructure
- **Optional deps:** `opentelemetry-exporter-otlp`, `azure-monitor-opentelemetry-exporter`, `ddtrace`

---

## Risks and Mitigations (Updated per Review)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hot path overhead > 10% | Medium | High | Phase 0 benchmark; async dispatch option if needed |
| Trust Erosion from dropped events | Medium | Critical | Backpressure mode option; loud logging; health metrics |
| Schema drift (telemetry ↔ Landscape) | High | Medium | Automated alignment tests in CI (future work) |
| Exporter config complexity | Medium | Medium | Fail fast on invalid config; clear error messages |
| Silent failures hide problems | Low | High | No silent degradation; CRITICAL logging; optional crash |

---

## Operational Guidance

### Monitoring Telemetry Health

The telemetry system exposes health metrics via `TelemetryManager.health_metrics`. Operators should monitor:

| Metric | Warning Threshold | Critical Threshold | Action |
|--------|-------------------|-------------------|--------|
| `events_dropped` | > 0 (any drops) | > 1000 in 5 min | Increase buffer or reduce granularity |
| `consecutive_total_failures` | > 5 | > 10 | Check exporter endpoints, credentials |
| `exporter_failures[name]` | Trend ↑ | > 100 per exporter | Check specific exporter config |

### Recommended Alerting

```yaml
# Example Datadog monitor
monitors:
  - name: "ELSPETH Telemetry - Events Dropped"
    query: "sum:elspeth.telemetry.events_dropped{*}.as_count() > 100"
    alert: "Telemetry dropping events - check buffer size or export latency"

  - name: "ELSPETH Telemetry - Total Exporter Failure"
    query: "max:elspeth.telemetry.consecutive_total_failures{*} >= 10"
    alert: "All telemetry exporters failing - check configuration immediately"
```

### Log Patterns to Watch

| Log Level | Message Pattern | Meaning |
|-----------|-----------------|---------|
| WARNING | "Telemetry buffer overflow" | Buffer full, events being dropped (every 100 drops) |
| WARNING | "Telemetry exporter failed" | Single exporter failed (per-failure, actionable) |
| ERROR | "ALL telemetry exporters failing" | No events reaching any backend (every 100 drops) |
| CRITICAL | "Telemetry disabled after repeated total failures" | System gave up (immediate, once) |

### Troubleshooting Decision Tree

```
Events not appearing in observability platform?
├── Check TelemetryManager.health_metrics
│   ├── events_emitted > 0? → Events flowing, check exporter
│   └── events_emitted = 0? → Check granularity filtering
│
├── events_dropped > 0?
│   ├── Buffer overflow → Increase buffer_size or reduce granularity
│   └── Exporter failures → Check exporter_failures dict
│
├── exporter_failures[name] high?
│   ├── OTLP → Check endpoint reachability, headers
│   ├── Azure Monitor → Check connection_string
│   └── Datadog → Check agent connectivity
│
└── No obvious issues?
    └── Enable console exporter to verify events are being emitted
```

### Performance Tuning

| Scenario | Recommended Settings |
|----------|---------------------|
| Development/debugging | `granularity: full`, `console` exporter |
| Production (low volume) | `granularity: rows`, OTLP with batching |
| Production (high volume) | `granularity: lifecycle`, increase `buffer_size` |
| CI/CD pipelines | `enabled: false` (or `lifecycle` only) |

---

## References

- **EventBus implementation:** `src/elspeth/core/events.py`
- **Existing SpanFactory:** `src/elspeth/engine/spans.py`
- **Config contracts refactor:** `docs/plans/in-progress/2026-01-29-config-contracts-refactor.md`
- **Plugin hookspecs:** `src/elspeth/plugins/hookspecs.py`
- **AuditedLLMClient:** `src/elspeth/plugins/clients/llm.py`
- **CLAUDE.md:** Three-Tier Trust Model, No Bug-Hiding Patterns
