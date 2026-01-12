# Discovery Findings (Holistic Scan)

## Project Snapshot
- **Language/runtime:** Python 3.11+
- **Package root:** `src/elspeth/`
- **Top-level architecture doc:** `docs/design/architecture.md`
- **Primary test suite:** `tests/` (pytest)

## Directory Structure
- `src/elspeth/core/` — canonicalization, config, DAG primitives, payload store, audit “Landscape”
  - `src/elspeth/core/landscape/` — SQL schema + recorder API for audit events
- `src/elspeth/engine/` — runtime execution engine (orchestrator, row processor, executors, tokens, spans, retry)
- `src/elspeth/plugins/` — plugin contracts (protocols), base classes, schemas, results, pluggy manager
  - `src/elspeth/plugins/{sources,transforms,sinks}/` — placeholders (no concrete plugins yet)

## Declared Entry Points
- `pyproject.toml` declares CLI script `elspeth = elspeth.cli:app`
  - **Observation:** `src/elspeth/cli.py` is not present; current CLI entry point is missing.

## Key Dependencies (by intent)
- **Config:** Dynaconf + Pydantic (`src/elspeth/core/config.py`)
- **Audit store:** SQLAlchemy Core (`src/elspeth/core/landscape/schema.py`, `.../recorder.py`)
- **Canonical hashing:** RFC 8785/JCS via `rfc8785` + strict NaN/Inf rejection (`src/elspeth/core/canonical.py`)
- **Graph primitives:** NetworkX DAG wrapper (`src/elspeth/core/dag.py`) — currently not wired into engine execution
- **Plugin system:** pluggy + typed protocols (`src/elspeth/plugins/*`)
- **Tracing:** OpenTelemetry span wrappers (`src/elspeth/engine/spans.py`)
- **Retries:** tenacity wrapper (`src/elspeth/engine/retry.py`)

## Subsystems Identified (4–12 major groups)
1. **Core / Canonicalization** (`elspeth.core.canonical`)
2. **Core / Configuration** (`elspeth.core.config`)
3. **Core / Execution Graph Primitives** (`elspeth.core.dag`)
4. **Core / Payload Store** (`elspeth.core.payload_store`)
5. **Core / Landscape (Audit Backbone)** (`elspeth.core.landscape.*`)
6. **Engine / Orchestration** (`elspeth.engine.orchestrator`, `elspeth.engine.processor`)
7. **Engine / Executors** (`elspeth.engine.executors`)
8. **Engine / Token Model** (`elspeth.engine.tokens`)
9. **Engine / Telemetry** (`elspeth.engine.spans`)
10. **Engine / Resilience** (`elspeth.engine.retry`)
11. **Plugins / Contracts & Types** (`elspeth.plugins.protocols`, `.../results`, `.../schemas`, `.../enums`)
12. **Plugins / Registration Runtime** (`elspeth.plugins.manager`, `.../hookspecs`)

## Notable “Phase Mismatch” Seams (important for further analysis)
- `elspeth.core.landscape.recorder` imports `elspeth.plugins.enums` (core depends on plugins for `NodeType`, `Determinism`).
- `elspeth.plugins.context.PluginContext.landscape` is typed as a minimal protocol (`record_event()`), but the engine passes a full `LandscapeRecorder` implementation with many methods (`type: ignore`).
- Engine sink execution expects a batch sink adapter (`SinkLike.write(rows) -> artifact info`), while plugin `SinkProtocol.write(row) -> None` is per-row.

