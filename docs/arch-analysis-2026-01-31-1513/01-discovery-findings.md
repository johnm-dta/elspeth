## Discovery Findings (Audit + Telemetry Focus)

### Directory Structure (Top-Level)
- `src/elspeth/core/landscape/`: Audit trail backbone (schema, recorder, repositories, exporter)
- `src/elspeth/telemetry/`: Telemetry subsystem (events, manager, exporters, filtering)
- `src/elspeth/engine/`: Orchestrator + RowProcessor + Executors (where audit/telemetry events are emitted)
- `src/elspeth/plugins/`: Sources, transforms, sinks, and audited clients (LLM/HTTP)
- `docs/guides/telemetry.md`: Telemetry behavior + config guidance
- `docs/guides/data-trust-and-error-handling.md`: Trust-tier rules incl. Landscape invariants

### Entry Points (Telemetry/Audit Relevant)
- `src/elspeth/cli.py`: Builds Orchestrator and runs pipelines (current wiring for audit/telemetry)
- `src/elspeth/engine/orchestrator.py`: Begins runs, creates PluginContext, runs sources, emits lifecycle telemetry
- `src/elspeth/engine/processor.py`: Emits per-row telemetry after Landscape recording
- `src/elspeth/core/landscape/recorder.py`: Primary audit API
- `src/elspeth/telemetry/manager.py`: Telemetry event router/exporter

### Technology Stack (Audit/Telemetry Specific)
- SQLAlchemy Core for Landscape persistence (`core/landscape`)
- structlog for telemetry/exporter logging (`telemetry/manager.py`)
- OpenTelemetry APIs for spans and exporters (`engine/spans.py`, `telemetry/exporters/*`)

### Subsystems Identified (Audit/Telemetry Path)
1. **Landscape Audit Core**: `core/landscape/*` (recorders, repositories, schema, exporter)
2. **Telemetry Core**: `telemetry/*` (events, manager, exporters, filtering)
3. **Engine Emission Layer**: `engine/orchestrator.py`, `engine/processor.py`, `engine/executors.py`
4. **Plugin Integration Layer**: `plugins/context.py`, `plugins/clients/*`, LLM transforms
5. **Payload Store/Canonicalization**: `core/payload_store.py`, `core/canonical.py` (audit payload storage + hashing)

### Notes for Deep Dive
- Audit trail is explicitly the legal record; telemetry is operational visibility.
- Telemetry configuration exists in `core/config.py` and runtime mapping in `contracts/config/runtime.py`.
- Sources, sinks, and LLM plugins have different audit touchpoints; this analysis focuses on how each funnels into Landscape + Telemetry.

### Limitations
- Subagent tooling is unavailable in this environment; analysis is single‑operator with explicit self‑validation later.
