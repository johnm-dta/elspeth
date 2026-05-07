---
name: logging-telemetry-policy
description: >
  Full logging and telemetry policy — audit primacy, permitted/forbidden logger uses,
  superset rule, telemetry-only exemptions, and the primacy test. Use when adding or
  modifying logging statements, telemetry emission, or structlog/logger calls.
---

# Logging and Telemetry Policy

> **Model complexity warning:** Logging decisions affect audit integrity. Simple
> models default to `logger.info()` for everything — this creates noise that
> obscures the audit trail and violates the primacy rule. Use the decision table
> below, not judgment.

## Quick Decision Table (Mechanical — No Judgment Required)

For every emission point, match top-to-bottom. First match wins.

| What happened? | Channel | How |
|----------------|---------|-----|
| Row processed, gate routed, transform applied, call made | **Audit + Telemetry** | `recorder.record_*()` first (sync), then `_emit_telemetry()` (async) |
| Run started, finished, phase transition | **Audit + Telemetry** | `recorder.record_*()` first, then telemetry |
| Plugin initialised, config loaded, provider connected, model selected | **Audit + Telemetry** | These establish system state for audit — "which model was active?" is a valid audit question |
| Checkpoint size, cache eviction, throughput counter | **Telemetry only** | `_emit_telemetry()` — no auditor asks "what was the checkpoint size?" |
| Exporter failure count, journal drop rate | **Telemetry only** | Meta-operational, about observability infrastructure itself |
| Audit system is broken (Landscape write failed) | **Logging** | `logger.error("Landscape write failed", exc_info=True)` — last resort |
| Telemetry system is broken (exporter crashed) | **Logging** | `logger.error("Exporter crashed", exc_info=True)` — last resort |
| Temporary debugging you'll remove before merge | **Logging** | `slog.debug(...)` — transitory only |

### Forbidden patterns (never write these)

```python
# NEVER — row-level decisions in logger (duplicates node_states)
logger.info(f"Row {row_id} classified as {category}")

# NEVER — transform outcomes in logger (duplicates Landscape)
logger.info(f"Transform {name} completed for {token_id}")

# NEVER — call results in logger (duplicates calls table)
logger.info(f"LLM response: {response[:100]}")

# NEVER — infrastructure lifecycle in logger (belongs in telemetry)
logger.info("Pipeline started")
logger.info("Checkpoint saved")

# NEVER — silent swallow of telemetry failure
try:
    emit_telemetry(event)
except Exception:
    pass  # Silently lost — violates "no silent failures"
```

## Telemetry (Operational Visibility)

Telemetry provides **real-time operational visibility** alongside the Landscape audit trail.

- **Landscape**: Legal record, complete lineage, persisted forever, source of truth
- **Telemetry**: Operational visibility, real-time streaming, ephemeral, for dashboards/alerting

**No Silent Failures:** Any telemetry emission point MUST either send what it has OR explicitly acknowledge "I have nothing" (with failure reason if applicable). Never silently swallow events or exceptions. This applies to `telemetry_emit` callbacks, `TelemetryManager.emit()`, exporter failures, and disabled states (log once at startup).

**Correlation:** Telemetry events include `run_id` and `token_id`. Use these to cross-reference with `elspeth explain` or the Landscape MCP server.

Full configuration guide (exporters, granularity levels, backpressure): `docs/guides/telemetry.md`.

## Logging Policy

**The Landscape audit system is a must-fire system with absolute primacy.** At every emission point, audit writes first — synchronously, transactionally, crash-on-failure. Only after audit succeeds does telemetry fire (async, best-effort). This ordering is non-negotiable. Logging (`logger`/`structlog`) is the channel of last resort, not a parallel record of pipeline activity.

**Permitted uses of `logger`/`structlog`:**

| Use Case | Example | Why Logging |
|----------|---------|-------------|
| **Transitory debugging** | `slog.debug("pool_state", active=3, idle=2)` | Temporary diagnostic, removed before merge or kept at DEBUG level |
| **Audit system failures** | `logger.error("Landscape write failed", exc_info=True)` | The audit system itself is broken — can't record to it |
| **Telemetry system failures** | `logger.error("Exporter crashed", exc_info=True)` | Both observability systems are down — logging is the last resort |

**Forbidden uses:**

| Anti-Pattern | Why Wrong | Correct Alternative |
|-------------|-----------|-------------------|
| Logging row-level decisions | Duplicates `node_states.success_reason_json` | Enrich `success_reason` metadata |
| Logging transform outcomes | Duplicates `node_states` status + context | Already in Landscape |
| Logging call results | Duplicates `calls` table | Already recorded by `record_call()` |
| Logging for alerting on data patterns | Logs are ephemeral, Landscape is queryable | Query `context_after_json` via MCP |
| Logging infrastructure lifecycle | Startup/shutdown/config events belong in telemetry | Emit via `_emit_telemetry()` — telemetry exporters route to dashboards |

## The Superset Rule

Everything that goes to telemetry should also go to audit, unless it lacks probative value — meaning an auditor subpoenaed to explain a past run's outcomes would never reference it. The key determinant is evidential value, not volume. A high-frequency event that carries probative value (e.g., per-row gate decisions) must be audited regardless of volume. A low-frequency event that lacks probative value (e.g., "cache evicted 3 keys") should not be audited regardless of how easy it would be to include.

**Telemetry-only exemptions** (no audit required):

| Category | Example | Why no probative value |
|----------|---------|----------------------|
| **Operational metrics** | Checkpoint size, cache eviction counts, throughput counters | Describe system pressure, not data outcomes — no auditor asks "what was the checkpoint size when row 42 was processed?" |
| **System health** | Exporter failure counts, journal drop rates, payload store errors | Meta-operational — about the observability infrastructure itself, not the data flowing through the pipeline |

**Must audit** (even if also telemetered):

| Category | Example | Why probative |
|----------|---------|--------------|
| **Pipeline decisions** | Row processed, gate routed, transform applied, call made | Directly explains what happened to each row |
| **Run lifecycle** | Run started/finished, phase transitions | Establishes timeline and system state for the run |
| **Infrastructure lifecycle** | Plugin initialised, config loaded, provider connected, model selected | Establishes what the system state was when decisions were made — "which model was active?" is a valid audit question |

## The Primacy Test

Audit always fires first, synchronously. Telemetry fires second, asynchronously. Logs fire only when the other two can't.

| Question | Channel |
|----------|---------|
| Does this have probative value for explaining a past run's outcomes? | **Audit + Telemetry** — audit first (permanent record), then telemetry (real-time view) |
| Is this useful for operational visibility but not for explaining outcomes? | **Telemetry only** |
| Is the audit or telemetry system itself broken? | **Logging** (last resort) |
