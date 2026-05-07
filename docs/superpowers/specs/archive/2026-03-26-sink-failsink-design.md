# Sink Failsink Pattern — Per-Row Write Failure Routing

**Date:** 2026-03-26
**Issue:** elspeth-ad6dc0f117
**Status:** Design

## Problem

Sinks are all-or-nothing at the batch level. When a sink's external system rejects a specific row during `write()` — bad metadata values for ChromaDB, constraint violations for a database, payload too large for an API — the entire pipeline crashes.

There is no "I couldn't write this row" path. The row has nowhere to go.

**Existing audit integrity violation:** ChromaSink already works around this by filtering rows with invalid metadata types internally (lines 191-218 of `chroma_sink.py`), but SinkExecutor opens node_states for ALL tokens in the batch and marks them ALL as `COMPLETED`. Rejected rows get `COMPLETED` outcomes even though they weren't written. This is incorrect — the audit trail claims data was written when it wasn't.

## Trust Model Context

The Tier 2 contract guarantees types are valid (sources validated them). It does NOT guarantee values will work for every external system:

- ChromaDB metadata accepts only `str|int|float|bool|None` — a `datetime` value is correctly typed pipeline data that ChromaDB rejects
- A database sink may hit a unique constraint violation — correctly typed, invalid value
- An API sink may reject a payload that exceeds its size limit — correctly typed, too large

These are value-level failures at the Tier 2 → External boundary. The types are right; the external system has constraints beyond our schema. This is symmetric to source quarantine at the External → Tier 3 boundary: the external system has constraints we must accommodate.

**The decision boundary lives in the plugin.** Only the sink, talking to the external system, knows whether a failure is per-row (divertable) or systemic (crash). A ChromaDB metadata rejection is per-row. A ChromaDB connection failure is systemic. The plugin makes this call in its `write()` method's exception handling.

## Design

### Core Model

The failsink is a **write-time error routing mechanism owned by the plugin**, not an orchestrator-level pre-screen.

1. The sink's `write()` attempts to write each row to the external system
2. When a per-row failure occurs, the plugin calls `self._divert_row(row, reason)` — infrastructure routes the row to a configured failsink (CSV/JSON/XML on disk)
3. When a systemic failure occurs, the plugin re-raises — pipeline crashes, as today
4. `write()` returns a `SinkWriteResult` reporting what happened (N written, M diverted, reasons)
5. SinkExecutor reads the result and records per-token outcomes (`COMPLETED` vs `DIVERTED`)

The orchestrator doesn't pre-screen, doesn't split batches, and doesn't judge diversion rates. It routes data, records everything, and reports facts. Operational alerting on diversion rates is the operator's concern, not the framework's.

### Symmetry With Source Quarantine

| Aspect | Source Quarantine | Sink Failsink |
|--------|-------------------|---------------|
| **Boundary** | External → Tier 3 | Tier 2 → External |
| **Trigger** | Source validation failure | Sink write-time failure |
| **Decision maker** | Source plugin (`yield SourceRow.quarantined(...)`) | Sink plugin (`self._divert_row(row, reason)`) |
| **Destination** | Configured quarantine sink | Configured failsink (CSV/JSON/XML) |
| **Outcome** | `QUARANTINED` | `DIVERTED` |
| **System state** | Healthy — bad external data | Healthy — external system rejected valid data |
| **Config field** | `on_validation_failure` on source | `on_write_failure` on sink |

### What Changes

| Component | Change |
|-----------|--------|
| `BaseSink` | Gains `_failsink` reference + `_divert_row()` helper + `_diversion_log` |
| `write()` return type | `ArtifactDescriptor` → `SinkWriteResult` (contains artifact + diversions) |
| `SinkExecutor` | Reads `SinkWriteResult`, records per-token `COMPLETED` vs `DIVERTED` outcomes |
| `SinkSettings` | Gains `on_write_failure: str \| None` field |
| `SinkProtocol` | Return type of `write()` updated to `SinkWriteResult` |
| DAG builder | Creates `__failsink__` DIVERT edge per sink with `on_write_failure` |
| `RowOutcome` | Gains `DIVERTED` variant |
| `accumulate_row_outcomes()` | Gains `DIVERTED` branch |
| `RunResult` / `ExecutionCounters` | Gains `rows_diverted` counter |
| Config validation | Validates failsink references, enforces CSV/JSON/XML-only, no chains |

### What Does NOT Change

- Orchestrator batch routing — still sends full batches to sinks
- SinkExecutor node_state lifecycle — opens states, calls write, completes states
- `write()` method signature — still `(rows: list[dict], ctx: SinkContext)`
- Sinks with `on_write_failure: discard` — drop failed rows with audit record (no failsink write)

## Detailed Design

### 1. BaseSink Infrastructure

```python
# contracts/sink.py (new types)

@dataclass(frozen=True, slots=True)
class RowDiversion:
    """Record of a single row diverted to failsink during write().

    Created by BaseSink._divert_row() and accumulated in _diversion_log.
    Read by SinkExecutor after write() returns to record per-token outcomes.
    """
    row_index: int       # Index in the original batch passed to write()
    reason: str          # Why the external system rejected this row
    row_data: dict[str, Any]  # The row that was diverted (for failsink write)


@dataclass(frozen=True, slots=True)
class SinkWriteResult:
    """Result of a sink write() call with optional diversion information.

    Replaces ArtifactDescriptor as the return type of write().
    Sinks with no diversions return SinkWriteResult(artifact=..., diversions=()).
    """
    artifact: ArtifactDescriptor
    diversions: tuple[RowDiversion, ...] = ()
```

```python
# On BaseSink (plugins/infrastructure/base.py)

class BaseSink(ABC):
    # ... existing attributes ...

    # Failsink infrastructure — injected by orchestrator before on_start()
    _failsink: BaseSink | None = None
    _diversion_log: list[RowDiversion]

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._output_contract = None
        self._needs_resume_field_resolution = False
        self._failsink = None
        self._diversion_log = []

    # on_write_failure mode — set by orchestrator from SinkSettings (required field)
    # "discard" = drop with audit, else = failsink name
    _on_write_failure: str

    def _divert_row(self, row: dict[str, Any], row_index: int, reason: str) -> None:
        """Divert a row to the failsink or discard it. Called by plugin write() on per-row failure.

        This is the sink-side equivalent of SourceRow.quarantined(). The plugin
        catches a per-row exception from the external system and calls this
        instead of re-raising.

        Two modes (set by orchestrator from config, always present):
        - _on_write_failure is "discard": row is dropped, recorded in diversion_log
        - _on_write_failure is a sink name: row accumulated for failsink write

        In both cases, the diversion is recorded in _diversion_log for the
        executor to read. The executor handles the actual discard-vs-write decision.

        Args:
            row: The row dict that couldn't be written.
            row_index: Index in the original batch (for token correlation).
            reason: Human-readable reason for the diversion.
        """
        self._diversion_log.append(RowDiversion(
            row_index=row_index,
            reason=reason,
            row_data=row,
        ))

    def _reset_diversion_log(self) -> None:
        """Clear diversion log before each write() call. Called by SinkExecutor."""
        self._diversion_log = []

    def _get_diversions(self) -> tuple[RowDiversion, ...]:
        """Return accumulated diversions from the last write() call."""
        return tuple(self._diversion_log)
```

**Key design decisions:**

- `_divert_row()` does NOT write to the failsink immediately. It accumulates diversions in `_diversion_log`. The SinkExecutor writes diverted rows to the failsink AFTER the primary write completes, ensuring clean separation of primary vs failsink artifacts.
- If `_divert_row()` is called without a failsink configured, that's a `FrameworkBugError` — the plugin shouldn't be catching per-row exceptions if there's no failsink to route them to.
- `_diversion_log` is reset before each `write()` call by the executor, not by the sink. This prevents the sink from accidentally clearing its own log.

### 2. Plugin Author Ergonomics

A sink adopts failsink support by wrapping its external operations:

```python
# ChromaSink example (simplified)
def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
    ids, documents, metadatas = [], [], []

    for i, row in enumerate(rows):
        try:
            meta = {field: row[field] for field in self._config.field_mapping.metadata_fields}
            # ChromaDB rejects non-primitive metadata values
            bad_fields = {
                k: type(v).__name__ for k, v in meta.items()
                if v is not None and not isinstance(v, (str, int, float, bool))
            }
            if bad_fields:
                self._divert_row(row, row_index=i, reason=f"Invalid metadata types: {bad_fields}")
                continue
            ids.append(row[self._config.field_mapping.id_field])
            documents.append(row[self._config.field_mapping.document_field])
            metadatas.append(meta)
        except chromadb.errors.ChromaError as e:
            # Infrastructure failure — crash the pipeline
            raise

    # Write the valid batch
    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    artifact = ArtifactDescriptor.for_database(...)
    return SinkWriteResult(artifact=artifact, diversions=self._get_diversions())
```

For future sinks (Azure Blob, Database, etc.), the pattern is the same:
```python
for i, row in enumerate(rows):
    try:
        external_system.write(row)
    except PerRowError as e:
        self._divert_row(row, row_index=i, reason=str(e))
    except SystemicError:
        raise  # crash
```

### 3. SinkExecutor Changes

SinkExecutor.write() currently:
1. Opens node_state per token (at the primary sink's node_id)
2. Calls `sink.write(rows, ctx)` → gets `ArtifactDescriptor`
3. Flushes sink
4. Completes all node_states as COMPLETED
5. Registers artifact
6. Records all token outcomes as the batch's PendingOutcome

With failsink support, the executor receives the resolved failsink as an explicit parameter (not accessed through the sink's private attribute):

```python
def write(
    self,
    sink: SinkProtocol,
    tokens: list[TokenInfo],
    ctx: PluginContext,
    step_in_pipeline: int,
    *,
    sink_name: str,
    pending_outcome: PendingOutcome,
    failsink: SinkProtocol | None = None,  # resolved at graph build time
    on_token_written: Callable[[TokenInfo], None] | None = None,
) -> Artifact | None:
```

**Revised flow:**

1. Opens node_state per token at primary sink's node_id (unchanged)
2. Resets sink diversion log, calls `sink.write(rows, ctx)` → gets `SinkWriteResult`
3. Flushes primary sink (unchanged)
4. Partitions tokens into primary (non-diverted) and diverted using `SinkWriteResult.diversions`
5. Completes primary token node_states as COMPLETED, registers primary artifact, records `COMPLETED` outcomes
6. **If diversions exist AND failsink is not None:**
   a. Opens node_state per diverted token at the failsink's node_id
   b. Writes diverted rows to failsink (enriched with rejection reason — see failsink row format below)
   c. Flushes failsink
   d. Completes diverted token node_states as COMPLETED at failsink
   e. Registers failsink artifact
   f. Records `DIVERTED` outcomes with error_hash for diverted tokens

**Node_state model for diverted tokens:** A diverted token gets ONE node_state at the failsink node_id (where it was actually written). It does NOT get a node_state at the primary sink — the primary sink never wrote it. The diversion is recorded via a `routing_event` with `mode=DIVERT` linking the primary sink to the failsink.

**Failsink row format:** The failsink receives enriched rows, not raw pipeline data:
```python
{
    **original_row,                    # All original row fields
    "__diversion_reason": reason,      # Why the primary sink rejected this row
    "__diverted_from": sink_name,      # Which sink rejected it
    "__diversion_timestamp": iso_ts,   # When the diversion occurred
}
```
The `__` prefix follows the existing convention for system-generated fields and avoids collision with user data fields.

**Error handling for failsink write:**
- If the failsink `write()` or `flush()` fails, complete diverted token states as FAILED and re-raise. The failsink is the last resort — if it fails, that's an infrastructure problem.
- Primary tokens that already wrote successfully keep their COMPLETED states. The primary write is durable and cannot be rolled back.
- This means the two write paths are independent: primary failure does not affect failsink, and failsink failure does not retroactively fail primary tokens.

### 4. New RowOutcome: DIVERTED

```python
class RowOutcome(StrEnum):
    # ... existing variants ...
    DIVERTED = "diverted"   # Row diverted to failsink during sink write
```

`DIVERTED` is terminal — the row reached a destination (the failsink). It carries `error_hash` in `PendingOutcome`, same as `QUARANTINED`.

`accumulate_row_outcomes()` gains a branch:
```python
elif result.outcome == RowOutcome.DIVERTED:
    counters.rows_diverted += 1
```

Note: `DIVERTED` tokens are NOT accumulated into `pending_tokens` — they were already written to the failsink by SinkExecutor. They don't need further routing.

### 5. Configuration

```yaml
# Pipeline YAML
sinks:
  chroma_output:
    plugin: chroma_sink
    on_write_failure: csv_failsink       # divert failed rows to CSV
    options:
      collection: my_collection
      mode: persistent
      persist_directory: ./chroma_data
      field_mapping:
        document_field: content
        id_field: doc_id
        metadata_fields: [topic, source, timestamp]

  csv_output:
    plugin: csv
    on_write_failure: discard            # drop failed rows (audit-recorded)
    options:
      path: ./output/results.csv

  csv_failsink:
    plugin: csv
    on_write_failure: discard            # failsinks use discard (no chains)
    options:
      path: ./output/failsink/chroma_rejects.csv
```

**SinkSettings changes:**
```python
class SinkSettings(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    plugin: str = Field(description="Plugin name")
    options: dict[str, Any] = Field(default_factory=dict)
    on_write_failure: str = Field(
        ...,  # Required — no default
        description=(
            "Per-row write failure handling. Required — pipeline author must decide:\n"
            "- 'discard': drop the row, record the discard in the audit trail\n"
            "- '<sink_name>': divert to named failsink (must be csv, json, or xml)"
        ),
    )
```

**Two-state model — mandatory field (parallel to source `on_validation_failure`):**

`on_write_failure` is required on every sink. There is no "not set" state. The pipeline author must make an explicit choice:

| `on_write_failure` | Behavior | Audit Record |
|--------------------|----------|--------------|
| `"discard"` | Drop row intentionally | `DIVERTED` outcome with `sink_name=__discard__`, routing_event with `reason`, logged at config time ("on_write_failure=discard configured for sink X") and at write time ("row 42 dropped per on_write_failure=discard: {reason}") |
| `"csv_failsink"` | Divert to named failsink | `DIVERTED` outcome with `sink_name=csv_failsink`, full failsink write with artifact |

Making this mandatory forces pipeline authors to think about what happens when a write fails — the same forcing function as requiring `on_validation_failure` on sources. "I didn't think about it" is not a valid configuration.

The `"discard"` path is a deliberate operational decision. The operator is saying "I know some rows will fail here, and I choose to drop them." That choice is recorded in the audit trail at both config validation time and at each discard event — an auditor can always find "this row was discarded, and here's the config that authorized it."

**Naming note:** This is structurally parallel to `TransformSettings.on_error` (per-row error routing to an alternative sink via DIVERT edge). The name `on_write_failure` was chosen over `on_error` to make the trigger point unambiguous: this fires when a sink's write operation fails for a specific row, not on a generic processing error.

**Config validation (at pipeline load time):**
1. `on_write_failure` is required on every sink (no default)
2. Value must be `"discard"` or a sink name defined in the same pipeline
3. If a sink name: referenced sink must use `csv`, `json`, or `xml` plugin type
4. If a sink name: referenced sink must have `on_write_failure: discard` (no chains — failsinks are terminal)
5. Circular reference check (redundant given rule 4, but defense in depth)

This validation runs alongside existing `validate_transform_on_error_destinations()` in `orchestrator/validation.py`.

**DAG builder:** For each sink with `on_write_failure` configured, create a `__failsink__` DIVERT edge from the sink node to the failsink node. This is structural (makes the failsink reachable in the graph for audit trail purposes), parallel to the existing `__quarantine__` DIVERT edges for sources.

### 6. Audit Trail

For a row diverted during ChromaSink write:

| Record | Content |
|--------|---------|
| `routing_event` | `from_node=chroma_output`, `to_node=csv_failsink`, `edge_id=__failsink__`, `mode=DIVERT`, `reason={invalid_metadata_types: {topic: "dict"}}` |
| `node_state` (failsink) | `token_id`, `node_id=csv_failsink`, `status=COMPLETED`, `input_data=enriched_row`, `output_data={row, artifact_path, content_hash}` |
| `token_outcome` | `outcome=DIVERTED`, `error_hash=<sha256(reason)[:16]>`, `sink_name=csv_failsink` |
| `artifact` (failsink) | Path to CSV file, content_hash, size_bytes |

For a row written successfully to the primary sink (same batch):

| Record | Content |
|--------|---------|
| `node_state` (primary) | `token_id`, `node_id=chroma_output`, `status=COMPLETED`, `output_data={row, artifact_path, content_hash}` |
| `token_outcome` | `outcome=COMPLETED`, `sink_name=chroma_output` |
| `artifact` (primary) | ChromaDB collection ref, content_hash, payload_size |

**Key:** Diverted tokens get NO node_state at the primary sink. Their only node_state is at the failsink. The routing_event links the primary to the failsink, providing the audit lineage.

An auditor querying `explain(recorder, run_id, token_id)` for a diverted token sees: source → transforms → routing_event(DIVERT, chroma_output → csv_failsink) → csv_failsink (written).

### 7. Migration Path

**ChromaSink (first consumer):**
- Remove inline metadata filtering from `write()` (lines 191-218)
- Replace with `_divert_row()` calls for invalid metadata types
- Change return type from `ArtifactDescriptor` to `SinkWriteResult`
- Existing tests updated to assert `SinkWriteResult.diversions` instead of checking internal filtering

**All other sinks (future consumers):**
- `write()` return type changes from `ArtifactDescriptor` to `SinkWriteResult`
- Sinks with no per-row failure handling return `SinkWriteResult(artifact=..., diversions=())`
- Adopt `_divert_row()` pattern when they encounter per-row external failures
- Order of adoption: ChromaSink → DatabaseSink → AzureBlobSink → DataverseSink → others

**Backwards compatibility:** Not applicable — ELSPETH has no users yet (per CLAUDE.md No Legacy Code Policy). All sinks change return type in the same commit.

## Scope

### In Scope
- `BaseSink` failsink infrastructure (`_failsink`, `_divert_row()`, `_diversion_log`)
- `SinkWriteResult` and `RowDiversion` contracts
- `SinkExecutor` changes for per-token outcome recording
- `RowOutcome.DIVERTED` and `rows_diverted` counter
- `SinkSettings.on_write_failure` config field
- Config validation (CSV/JSON/XML only, no chains)
- DAG builder `__failsink__` DIVERT edges
- ChromaSink migration (first consumer)
- All other sinks: return type change to `SinkWriteResult` (no-diversion path)
- `accumulate_row_outcomes()` DIVERTED branch

### Out of Scope
- Diversion rate thresholds or alerting (operator concern, not framework)
- Failsink-to-failsink chaining
- Non-file failsink types (database, API)
- Retry-then-divert patterns (handled by existing RetryManager)
- Resume/checkpoint interaction with diverted rows (follow-up design needed)

## Panel Review Summary

Four-agent review was conducted before this design was finalized:

| Reviewer | Key Input Incorporated |
|----------|----------------------|
| **Architecture Critic** | New `DIVERTED` outcome (not reusing `QUARANTINED`); failsinks cannot have failsinks; failsink failure = crash; current ChromaSink filtering is an audit integrity violation |
| **Systems Thinker** | "Shifting the Burden" archetype identified — addressed by making the framework record facts without judging diversion rates; operator alerting is out of scope |
| **Python Engineer** | Failsink reference is orchestration config, not plugin state; pre-flight validation at config time; structured rejection info; follow `on_error` pattern for config/DAG wiring |
| **Quality Engineer** | 7 critical test scenarios; property-based testing for partition completeness; regression risk around `_complete_states_failed()` with split batches; must remove ChromaSink inline filtering after migration |

## Open Questions

1. **`RowDiversion.reason` type:** Start with `str` or use a structured `SinkRejectionReason` TypedDict from day one? The Python engineer recommended structured; starting with `str` is simpler but creates audit query limitations.

2. **Routing_event timing:** The routing_event for DIVERT must be recorded before the failsink node_state is opened. This matches the source quarantine pattern. Need to confirm the recorder supports this ordering for sink-to-sink routing events (current routing_events are transform-to-sink).

3. **Checkpoint/resume interaction:** When a pipeline resumes after a crash, diverted rows from the previous run were already written to the failsink. Resume must not re-divert them. This needs a follow-up design for the checkpoint contract.
