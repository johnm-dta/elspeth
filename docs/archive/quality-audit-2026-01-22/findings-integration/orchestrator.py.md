## Summary

Orchestrator resume directly queries Landscape SQLAlchemy tables (`runs_table`, `edges_table`), leaking persistence details into the engine boundary instead of using the LandscapeRecorder interface.

## Severity

- Severity: major
- Priority: P1

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [x] Leaky Abstraction (implementation details cross boundaries)
- [ ] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [ ] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** engine ↔ landscape

**Integration Point:** Resume path reads run schema and edge IDs via direct SQLAlchemy table access.

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: `src/elspeth/engine/orchestrator.py`

`src/elspeth/engine/orchestrator.py:1487-1492`
```python
1487        from sqlalchemy import select
1489        from elspeth.core.landscape.schema import runs_table
1491        with self._db.engine.connect() as conn:
1492            run_row = conn.execute(select(runs_table.c.source_schema_json).where(runs_table.c.run_id == run_id)).fetchone()
```

`src/elspeth/engine/orchestrator.py:1600-1606`
```python
1600        from sqlalchemy import select
1602        from elspeth.core.landscape.schema import edges_table
1605        with self._db.engine.connect() as conn:
1606            edges = conn.execute(select(edges_table).where(edges_table.c.run_id == run_id)).fetchall()
```

### Side B: `src/elspeth/core/landscape/schema.py`

`src/elspeth/core/landscape/schema.py:27-84`
```python
27  runs_table = Table(
28      "runs",
...
74  edges_table = Table(
75      "edges",
76      metadata,
77      Column("edge_id", String(64), primary_key=True),
78      Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
...
```

### Coupling Evidence: `src/elspeth/core/landscape/recorder.py`

`src/elspeth/core/landscape/recorder.py:344-355`
```python
344    def get_run(self, run_id: str) -> Run | None:
353        with self._db.connection() as conn:
354            result = conn.execute(select(runs_table).where(runs_table.c.run_id == run_id))
```

`src/elspeth/core/landscape/recorder.py:694-704`
```python
694    def get_edges(self, run_id: str) -> list[Edge]:
704        query = select(edges_table).where(edges_table.c.run_id == run_id).order_by(edges_table.c.created_at, edges_table.c.edge_id)
```

## Root Cause Hypothesis

Resume needed access to source schema and edge IDs, and the engine bypassed the LandscapeRecorder facade, letting SQLAlchemy schema details leak into the engine layer.

## Recommended Fix

1. Use `LandscapeRecorder.get_run()` to read `source_schema_json` (or add a dedicated accessor returning the schema string).
2. Use `LandscapeRecorder.get_edges()` to build the `edge_map` instead of querying `edges_table` directly.
3. Remove SQLAlchemy imports from `src/elspeth/engine/orchestrator.py` and keep DB schema access inside the landscape subsystem.

## Impact Assessment

- **Coupling Level:** High - engine now depends on landscape schema and SQLAlchemy details
- **Maintainability:** Medium - schema changes require engine edits
- **Type Safety:** Low - no contract boundary enforcing what fields exist
- **Breaking Change Risk:** Medium - schema refactors will ripple into engine

## Related Seams

None identified
---
Template Version: 1.0
---
## Summary

CSV export in Orchestrator dereferences `sink.config["path"]`, but `SinkProtocol` does not require a `config` attribute, so export behavior depends on BaseSink implementation details.

## Severity

- Severity: major
- Priority: P1

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [ ] Leaky Abstraction (implementation details cross boundaries)
- [x] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [ ] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** engine ↔ plugins

**Integration Point:** export sink configuration access for CSV output.

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: `src/elspeth/engine/orchestrator.py`

`src/elspeth/engine/orchestrator.py:1213-1221`
```python
1213        if export_config.format == "csv":
1216            # the path from sink config. CSV format requires file-based sink.
1217            if "path" not in sink.config:
1218                raise ValueError(
1219                    f"CSV export requires file-based sink with 'path' in config, but sink '{sink_name}' has no path configured"
1220                )
1221            artifact_path: str = sink.config["path"]
```

### Side B: `src/elspeth/plugins/protocols.py`

`src/elspeth/plugins/protocols.py:405-416`
```python
405    name: str
406    input_schema: type["PluginSchema"]
407    idempotent: bool  # Can this sink handle retries safely?
408    node_id: str | None  # Set by orchestrator after registration
409
410    # Metadata for Phase 3 audit/reproducibility
411    determinism: Determinism
412    plugin_version: str
414    def __init__(self, config: dict[str, Any]) -> None:
```

### Coupling Evidence: `src/elspeth/plugins/base.py`

`src/elspeth/plugins/base.py:237-243`
```python
237    def __init__(self, config: dict[str, Any]) -> None:
241        Args:
242            config: Plugin configuration
243        self.config = config
```

## Root Cause Hypothesis

Export logic assumes all sinks subclass BaseSink and retain `config`, but the formal SinkProtocol contract never encoded that requirement.

## Recommended Fix

1. Add `config: dict[str, Any]` to `SinkProtocol` (and update docs) to make the assumption explicit.
2. Alternatively, introduce an `ExportSinkProtocol`/`FileSinkProtocol` with a `path` attribute and require that for CSV export.
3. Update any sinks that implement the protocol directly to satisfy the new contract.

## Impact Assessment

- **Coupling Level:** Medium - engine relies on BaseSink internals
- **Maintainability:** Medium - protocol and implementation can drift
- **Type Safety:** Low - missing contract means no static enforcement
- **Breaking Change Risk:** Medium - tightening the protocol affects custom sinks

## Related Seams

`src/elspeth/core/dag.py:364-379`
---
Template Version: 1.0
---
## Summary

Route destination semantics rely on string sentinels ("continue", "fork") in both DAG construction and Orchestrator validation, creating a stringly-typed interface across the boundary.

## Severity

- Severity: minor
- Priority: P2

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [ ] Leaky Abstraction (implementation details cross boundaries)
- [ ] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [x] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** core/dag ↔ engine/orchestrator

**Integration Point:** `route_resolution_map` destination values used for routing validation.

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: `src/elspeth/engine/orchestrator.py`

`src/elspeth/engine/orchestrator.py:264-271`
```python
264        for (gate_node_id, route_label), destination in route_resolution_map.items():
265            # "continue" means proceed to next transform, not a sink
266            if destination == "continue":
267                continue
268
269            # "fork" means fork to multiple paths, not a sink
270            if destination == "fork":
271                continue
```

### Side B: `src/elspeth/core/dag.py`

`src/elspeth/core/dag.py:466-472`
```python
466            # Gate routes to sinks
467            for route_label, target in gate_config.routes.items():
468                if target == "continue":
469                    graph._route_resolution_map[(gid, route_label)] = "continue"
470                elif target == "fork":
471                    # Fork is a special routing mode - handled by fork_to branches
472                    graph._route_resolution_map[(gid, route_label)] = "fork"
```

### Coupling Evidence: `src/elspeth/engine/executors.py`

`src/elspeth/engine/executors.py:586-597`
```python
586        if destination == "continue":
...
596        elif destination == "fork":
```

## Root Cause Hypothesis

Routing destinations were encoded as magic strings to coexist with sink names, without a typed representation for the sentinel values.

## Recommended Fix

1. Introduce a typed `RouteDestination` (enum or dataclass) with explicit variants for `continue`, `fork`, and `sink(name)`.
2. Update `ExecutionGraph` to build a typed route resolution map and adjust Orchestrator validation to match on the type.
3. Update executors and gate configuration validation to use the new type (or a `Literal` union if a full type object is too heavy).

## Impact Assessment

- **Coupling Level:** Medium - multiple layers must share the same magic strings
- **Maintainability:** Medium - new routing semantics require coordinated string changes
- **Type Safety:** Low - errors are runtime-only
- **Breaking Change Risk:** Medium - API changes ripple through graph/executor/orchestrator

## Related Seams

`src/elspeth/core/config.py:230-234`
---
Template Version: 1.0
