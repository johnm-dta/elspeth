# Landscape Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable cryptographically-signed export of Landscape audit data to any configured sink after a run completes.

**Architecture:** The export happens as a "post-run epilogue" configured in settings. After `complete_run()`, the Orchestrator checks for `landscape.export` config, queries all audit data for that run, formats it as rows, and writes to a configured sink. Optional HMAC signing provides legal-grade integrity verification.

**Tech Stack:** Pydantic (config), existing sink infrastructure, HMAC (signing), existing LandscapeRecorder query methods.

**Review Status:** GO (reviewed 2026-01-14, corrections applied)

---

## Task 0: Add get_edges() Method to LandscapeRecorder (Prerequisite)

**Context:** The LandscapeExporter needs to query edges for a run, but this method doesn't exist yet.

**Files:**

- Modify: `src/elspeth/core/landscape/recorder.py`
- Modify: `src/elspeth/core/landscape/models.py` (if Edge model missing)
- Test: `tests/core/landscape/test_recorder.py`

### Step 1: Write the failing test

```python
# tests/core/landscape/test_recorder.py - add to existing file

def test_get_edges_returns_all_edges_for_run():
    """get_edges should return all edges registered for a run."""
    db = LandscapeDB.from_url("sqlite:///:memory:")
    recorder = LandscapeRecorder(db)

    run = recorder.begin_run(config={}, canonical_version="v1")

    # Register nodes
    recorder.register_node(
        run_id=run.run_id,
        node_id="source_1",
        plugin_name="csv",
        node_type="source",
        plugin_version="1.0.0",
        config={},
    )
    recorder.register_node(
        run_id=run.run_id,
        node_id="sink_1",
        plugin_name="csv",
        node_type="sink",
        plugin_version="1.0.0",
        config={},
    )

    # Register edge
    edge = recorder.register_edge(
        run_id=run.run_id,
        from_node_id="source_1",
        to_node_id="sink_1",
        label="continue",
        mode="move",
    )

    # Query edges
    edges = recorder.get_edges(run.run_id)

    assert len(edges) == 1
    assert edges[0].edge_id == edge.edge_id
    assert edges[0].from_node_id == "source_1"
    assert edges[0].to_node_id == "sink_1"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_recorder.py::test_get_edges_returns_all_edges_for_run -v`
Expected: FAIL with AttributeError ('LandscapeRecorder' has no attribute 'get_edges')

### Step 3: Write minimal implementation

```python
# src/elspeth/core/landscape/recorder.py - add method

def get_edges(self, run_id: str) -> list[Edge]:
    """Get all edges for a run.

    Args:
        run_id: Run ID

    Returns:
        List of Edge models for this run
    """
    from elspeth.core.landscape.schema import edges as edges_table

    query = select(edges_table).where(edges_table.c.run_id == run_id)

    with self._db.connection() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    return [
        Edge(
            edge_id=r.edge_id,
            run_id=r.run_id,
            from_node_id=r.from_node_id,
            to_node_id=r.to_node_id,
            label=r.label,
            mode=r.mode,
            created_at=r.created_at,
        )
        for r in rows
    ]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/core/landscape/test_recorder.py::test_get_edges_returns_all_edges_for_run -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/landscape/test_recorder.py
git commit -m "feat(landscape): add get_edges() query method to LandscapeRecorder"
```

---

## Task 1: Add Export Config Schema

**Files:**

- Modify: `src/elspeth/core/config.py:59-75`
- Test: `tests/core/test_config.py`

### Step 1: Write the failing test

```python
# tests/core/test_config.py - add to existing file

def test_landscape_export_config_defaults():
    """Export config should have sensible defaults."""
    from elspeth.core.config import LandscapeSettings

    settings = LandscapeSettings()
    assert settings.export is not None
    assert settings.export.enabled is False
    assert settings.export.format == "csv"
    assert settings.export.sign is False


def test_landscape_export_config_with_sink():
    """Export config should accept sink reference."""
    from elspeth.core.config import LandscapeSettings

    settings = LandscapeSettings(
        export={
            "enabled": True,
            "sink": "audit_archive",
            "format": "csv",
            "sign": True,
        }
    )
    assert settings.export.enabled is True
    assert settings.export.sink == "audit_archive"
    assert settings.export.sign is True
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_config.py::test_landscape_export_config_defaults -v`
Expected: FAIL with AttributeError (no 'export' attribute)

### Step 3: Write minimal implementation

```python
# src/elspeth/core/config.py - add before LandscapeSettings

class LandscapeExportSettings(BaseModel):
    """Landscape export configuration for audit compliance.

    Exports audit trail to a configured sink after run completes.
    Optional cryptographic signing for legal-grade integrity.
    """

    model_config = {"frozen": True}

    enabled: bool = Field(
        default=False,
        description="Enable audit trail export after run completes",
    )
    sink: str | None = Field(
        default=None,
        description="Sink name to export to (must be defined in sinks)",
    )
    format: Literal["csv", "json"] = Field(
        default="csv",
        description="Export format: csv (human-readable) or json (machine)",
    )
    sign: bool = Field(
        default=False,
        description="HMAC sign each record for integrity verification",
    )


# Modify LandscapeSettings to include export:
class LandscapeSettings(BaseModel):
    """Landscape audit system configuration per architecture."""

    model_config = {"frozen": True}

    enabled: bool = Field(default=True, description="Enable audit trail recording")
    backend: Literal["sqlite", "postgresql"] = Field(
        default="sqlite",
        description="Database backend type",
    )
    url: str = Field(
        default="sqlite:///./runs/audit.db",
        description="Full SQLAlchemy database URL",
    )
    export: LandscapeExportSettings = Field(
        default_factory=LandscapeExportSettings,
        description="Post-run audit export configuration",
    )
```

### Step 4: Run test to verify it passes

Run: `pytest tests/core/test_config.py::test_landscape_export_config_defaults tests/core/test_config.py::test_landscape_export_config_with_sink -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/config.py tests/core/test_config.py
git commit -m "feat(config): add LandscapeExportSettings for audit trail export"
```

---

## Task 2: Validate Export Sink Reference

**Files:**

- Modify: `src/elspeth/core/config.py:130-183`
- Test: `tests/core/test_config.py`

### Step 1: Write the failing test

```python
# tests/core/test_config.py

def test_export_sink_must_exist_when_enabled():
    """If export.enabled=True, export.sink must reference a defined sink."""
    from pydantic import ValidationError
    from elspeth.core.config import ElspethSettings

    with pytest.raises(ValidationError) as exc_info:
        ElspethSettings(
            datasource={"plugin": "csv", "options": {"path": "input.csv"}},
            sinks={"output": {"plugin": "csv", "options": {"path": "out.csv"}}},
            output_sink="output",
            landscape={
                "export": {
                    "enabled": True,
                    "sink": "nonexistent_sink",  # Not in sinks
                }
            },
        )

    assert "export.sink 'nonexistent_sink' not found in sinks" in str(exc_info.value)


def test_export_sink_not_required_when_disabled():
    """If export.enabled=False, sink can be None."""
    from elspeth.core.config import ElspethSettings

    # Should not raise
    settings = ElspethSettings(
        datasource={"plugin": "csv", "options": {"path": "input.csv"}},
        sinks={"output": {"plugin": "csv", "options": {"path": "out.csv"}}},
        output_sink="output",
        landscape={
            "export": {"enabled": False}  # No sink required
        },
    )
    assert settings.landscape.export.sink is None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_config.py::test_export_sink_must_exist_when_enabled -v`
Expected: FAIL (no validation error raised)

### Step 3: Write minimal implementation

```python
# src/elspeth/core/config.py - add validator to ElspethSettings

@model_validator(mode="after")
def validate_export_sink_exists(self) -> "ElspethSettings":
    """Ensure export.sink references a defined sink when enabled."""
    if self.landscape.export.enabled:
        if self.landscape.export.sink is None:
            raise ValueError(
                "landscape.export.sink is required when export is enabled"
            )
        if self.landscape.export.sink not in self.sinks:
            raise ValueError(
                f"landscape.export.sink '{self.landscape.export.sink}' not found in sinks. "
                f"Available sinks: {list(self.sinks.keys())}"
            )
    return self
```

### Step 4: Run test to verify it passes

Run: `pytest tests/core/test_config.py::test_export_sink_must_exist_when_enabled tests/core/test_config.py::test_export_sink_not_required_when_disabled -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/config.py tests/core/test_config.py
git commit -m "feat(config): validate export sink reference"
```

---

## Task 3: Create LandscapeExporter Class

**Files:**

- Create: `src/elspeth/core/landscape/exporter.py`
- Test: `tests/core/landscape/test_exporter.py`

### Step 1: Write the failing test

```python
# tests/core/landscape/test_exporter.py

import pytest
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.core.landscape.exporter import LandscapeExporter


@pytest.fixture
def populated_db():
    """Create a Landscape with one complete run."""
    db = LandscapeDB.from_url("sqlite:///:memory:")
    recorder = LandscapeRecorder(db)

    run = recorder.begin_run(config={"test": True}, canonical_version="v1")

    recorder.register_node(
        run_id=run.run_id,
        node_id="source_1",
        plugin_name="csv",
        node_type="source",
        plugin_version="1.0.0",
        config={"path": "input.csv"},
    )

    # NOTE: Use create_row, not record_source_row (which doesn't exist)
    row = recorder.create_row(
        run_id=run.run_id,
        source_node_id="source_1",
        row_index=0,
        data={"name": "Alice", "value": 100},
    )

    recorder.complete_run(run.run_id, status="completed")

    return db, run.run_id


def test_exporter_extracts_run_metadata(populated_db):
    """Exporter should yield run metadata as first record."""
    db, run_id = populated_db
    exporter = LandscapeExporter(db)

    records = list(exporter.export_run(run_id))

    # Find run record
    run_records = [r for r in records if r["record_type"] == "run"]
    assert len(run_records) == 1
    assert run_records[0]["run_id"] == run_id
    assert run_records[0]["status"] == "completed"


def test_exporter_extracts_rows(populated_db):
    """Exporter should yield row records."""
    db, run_id = populated_db
    exporter = LandscapeExporter(db)

    records = list(exporter.export_run(run_id))

    row_records = [r for r in records if r["record_type"] == "row"]
    assert len(row_records) == 1
    assert row_records[0]["row_index"] == 0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_exporter.py::test_exporter_extracts_run_metadata -v`
Expected: FAIL with ImportError (module doesn't exist)

### Step 3: Write minimal implementation

```python
# src/elspeth/core/landscape/exporter.py
"""Landscape audit trail exporter.

Exports complete audit data for a run in a format suitable for
compliance review and legal inquiry.
"""

from collections.abc import Iterator
from typing import Any

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


class LandscapeExporter:
    """Export Landscape audit data for a run.

    Produces a flat sequence of records suitable for CSV/JSON export.
    Each record has a 'record_type' field indicating its category.

    Record types:
    - run: Run metadata (one per export)
    - node: Registered plugins
    - edge: Graph edges
    - row: Source rows
    - token: Row instances
    - token_parent: Token lineage for forks/joins
    - node_state: Processing records
    - routing_event: Routing decisions
    - call: External calls
    - batch: Aggregation batches
    - batch_member: Batch membership
    - artifact: Sink outputs
    """

    def __init__(self, db: LandscapeDB) -> None:
        self._db = db
        self._recorder = LandscapeRecorder(db)

    def export_run(self, run_id: str) -> Iterator[dict[str, Any]]:
        """Export all audit data for a run.

        Yields flat dict records with 'record_type' field.
        Order: run -> nodes -> edges -> rows -> tokens -> states -> artifacts
        """
        # Run metadata
        run = self._recorder.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        yield {
            "record_type": "run",
            "run_id": run.run_id,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "canonical_version": run.canonical_version,
            "config_hash": run.config_hash,
        }

        # Nodes
        for node in self._recorder.get_nodes(run_id):
            yield {
                "record_type": "node",
                "run_id": run_id,
                "node_id": node.node_id,
                "plugin_name": node.plugin_name,
                "node_type": node.node_type,
                "plugin_version": node.plugin_version,
                "config_hash": node.config_hash,
            }

        # Edges (uses new get_edges method from Task 0)
        for edge in self._recorder.get_edges(run_id):
            yield {
                "record_type": "edge",
                "run_id": run_id,
                "edge_id": edge.edge_id,
                "from_node_id": edge.from_node_id,
                "to_node_id": edge.to_node_id,
                "label": edge.label,
                "mode": edge.mode,
            }

        # Rows and their tokens/states
        for row in self._recorder.get_rows(run_id):
            yield {
                "record_type": "row",
                "run_id": run_id,
                "row_id": row.row_id,
                "row_index": row.row_index,
                "source_node_id": row.source_node_id,
                "source_data_hash": row.source_data_hash,  # CORRECTED field name
            }

            # Tokens for this row
            for token in self._recorder.get_tokens(row.row_id):
                yield {
                    "record_type": "token",
                    "run_id": run_id,
                    "token_id": token.token_id,
                    "row_id": token.row_id,
                    "step_in_pipeline": token.step_in_pipeline,
                }

                # Token parents (for fork/join lineage)
                for parent in self._recorder.get_token_parents(token.token_id):
                    yield {
                        "record_type": "token_parent",
                        "run_id": run_id,
                        "token_id": parent.token_id,
                        "parent_token_id": parent.parent_token_id,
                    }

                # Node states for this token
                for state in self._recorder.get_node_states_for_token(token.token_id):
                    yield {
                        "record_type": "node_state",
                        "run_id": run_id,
                        "state_id": state.state_id,
                        "token_id": state.token_id,
                        "node_id": state.node_id,
                        "status": state.status,
                        "input_hash": state.input_hash,
                        "output_hash": state.output_hash,
                        "started_at": state.started_at.isoformat() if state.started_at else None,
                        "completed_at": state.completed_at.isoformat() if state.completed_at else None,
                    }

                    # Routing events
                    for event in self._recorder.get_routing_events(state.state_id):
                        yield {
                            "record_type": "routing_event",
                            "run_id": run_id,
                            "event_id": event.event_id,
                            "state_id": event.state_id,
                            "edge_id": event.edge_id,
                            "reason_hash": event.reason_hash,
                        }

                    # External calls
                    for call in self._recorder.get_calls(state.state_id):
                        yield {
                            "record_type": "call",
                            "run_id": run_id,
                            "call_id": call.call_id,
                            "state_id": call.state_id,
                            "call_type": call.call_type,  # CORRECTED: was "provider"
                            "request_hash": call.request_hash,
                            "response_hash": call.response_hash,
                            "status": call.status,  # CORRECTED: was "status_code"
                            "latency_ms": call.latency_ms,
                        }

        # Batches
        for batch in self._recorder.get_batches(run_id):
            yield {
                "record_type": "batch",
                "run_id": run_id,
                "batch_id": batch.batch_id,
                "node_id": batch.node_id,
                "status": batch.status,
                "created_at": batch.created_at.isoformat() if batch.created_at else None,
            }

            # Batch members
            for member in self._recorder.get_batch_members(batch.batch_id):
                yield {
                    "record_type": "batch_member",
                    "run_id": run_id,
                    "batch_id": member.batch_id,
                    "token_id": member.token_id,
                }

        # Artifacts
        for artifact in self._recorder.get_artifacts(run_id):
            yield {
                "record_type": "artifact",
                "run_id": run_id,
                "artifact_id": artifact.artifact_id,
                "sink_node_id": artifact.sink_node_id,
                "content_hash": artifact.content_hash,
                "artifact_type": artifact.artifact_type,
            }
```

### Step 4: Update `__init__.py` exports

```python
# src/elspeth/core/landscape/__init__.py - add to exports
from elspeth.core.landscape.exporter import LandscapeExporter

__all__ = [
    # ... existing exports ...
    "LandscapeExporter",
]
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_exporter.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/elspeth/core/landscape/exporter.py src/elspeth/core/landscape/__init__.py tests/core/landscape/test_exporter.py
git commit -m "feat(landscape): add LandscapeExporter for audit trail export"
```

---

## Task 4: Add HMAC Signing to Exporter

**Files:**

- Modify: `src/elspeth/core/landscape/exporter.py`
- Test: `tests/core/landscape/test_exporter.py`

### Step 1: Write the failing test

```python
# tests/core/landscape/test_exporter.py

def test_exporter_signs_records_when_enabled(populated_db):
    """When signing enabled, each record should have signature field."""
    db, run_id = populated_db
    exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

    records = list(exporter.export_run(run_id, sign=True))

    # All records should have signature
    for record in records:
        assert "signature" in record
        assert len(record["signature"]) == 64  # SHA256 hex


def test_exporter_manifest_contains_final_hash(populated_db):
    """Signed export should include manifest with hash of all records."""
    db, run_id = populated_db
    exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

    records = list(exporter.export_run(run_id, sign=True))

    manifest_records = [r for r in records if r["record_type"] == "manifest"]
    assert len(manifest_records) == 1

    manifest = manifest_records[0]
    assert "record_count" in manifest
    assert "final_hash" in manifest
    assert "exported_at" in manifest  # Timestamp for forensics
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_exporter.py::test_exporter_signs_records_when_enabled -v`
Expected: FAIL (no 'signature' field)

### Step 3: Write minimal implementation

```python
# src/elspeth/core/landscape/exporter.py - modify class

import hashlib
import hmac
from datetime import datetime, timezone
from elspeth.core.canonical import canonical_json


class LandscapeExporter:
    """Export Landscape audit data for a run."""

    def __init__(
        self,
        db: LandscapeDB,
        signing_key: bytes | None = None,
    ) -> None:
        self._db = db
        self._recorder = LandscapeRecorder(db)
        self._signing_key = signing_key

    def _sign_record(self, record: dict[str, Any]) -> str:
        """Compute HMAC-SHA256 signature for a record."""
        if self._signing_key is None:
            raise ValueError("Signing key not configured")

        # Canonical JSON ensures consistent hash
        canonical = canonical_json(record)
        return hmac.new(
            self._signing_key,
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def export_run(
        self,
        run_id: str,
        sign: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Export all audit data for a run.

        Args:
            run_id: Run to export
            sign: If True, add HMAC signature to each record

        Yields:
            Flat dict records with 'record_type' field.
            If sign=True, includes 'signature' field and final manifest.
        """
        if sign and self._signing_key is None:
            raise ValueError("Signing requested but no signing_key provided")

        running_hash = hashlib.sha256()
        record_count = 0

        for record in self._iter_records(run_id):
            if sign:
                record["signature"] = self._sign_record(record)
                # Update running hash with signed record
                running_hash.update(record["signature"].encode())

            record_count += 1
            yield record

        # Emit manifest if signing
        if sign:
            manifest = {
                "record_type": "manifest",
                "run_id": run_id,
                "record_count": record_count,
                "final_hash": running_hash.hexdigest(),
                "hash_algorithm": "sha256",
                "signature_algorithm": "hmac-sha256",
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest["signature"] = self._sign_record(manifest)
            yield manifest

    def _iter_records(self, run_id: str) -> Iterator[dict[str, Any]]:
        """Internal: iterate over raw records (no signing)."""
        # ... move existing export_run logic here ...
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_exporter.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/exporter.py tests/core/landscape/test_exporter.py
git commit -m "feat(landscape): add HMAC signing to audit export"
```

---

## Task 5: Integrate Export into Orchestrator

**Files:**

- Modify: `src/elspeth/engine/orchestrator.py`
- Modify: `src/elspeth/cli.py` (pass settings to Orchestrator)
- Test: `tests/engine/test_orchestrator.py`

### Step 1: Write the failing test

```python
# tests/engine/test_orchestrator.py - add new test

def test_orchestrator_exports_landscape_when_configured():
    """Orchestrator should export audit trail after run completes."""
    from elspeth.core.config import ElspethSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig
    from elspeth.plugins.base import BaseSink, BaseSource
    from elspeth.plugins.context import PluginContext

    # Mock source that yields one row
    class MockSource(BaseSource):
        name = "mock"

        def __init__(self):
            super().__init__({})
            self.node_id = None

        def load(self, ctx):
            yield {"id": 1, "value": "test"}

        def close(self):
            pass

    # Mock sink that captures writes
    class CaptureSink(BaseSink):
        name = "capture"

        def __init__(self):
            super().__init__({})
            self.node_id = None
            self.captured_rows = []

        def write(self, row, ctx):
            self.captured_rows.append(row)

        def flush(self):
            pass

        def close(self):
            pass

    # Create in-memory DB
    db = LandscapeDB.from_url("sqlite:///:memory:")

    # Create sinks
    output_sink = CaptureSink()
    export_sink = CaptureSink()

    # Build settings with export enabled
    settings = ElspethSettings(
        datasource={"plugin": "csv", "options": {"path": "input.csv"}},
        sinks={
            "output": {"plugin": "csv", "options": {"path": "out.csv"}},
            "audit_export": {"plugin": "csv", "options": {"path": "audit.csv"}},
        },
        output_sink="output",
        landscape={
            "url": "sqlite:///:memory:",
            "export": {
                "enabled": True,
                "sink": "audit_export",
                "format": "csv",
            }
        },
    )

    source = MockSource()
    pipeline = PipelineConfig(
        source=source,
        transforms=[],
        sinks={
            "output": output_sink,
            "audit_export": export_sink,
        },
        config={},
    )

    # Build graph
    graph = ExecutionGraph.from_plugins(
        source=source,
        transforms=[],
        sinks={"output": output_sink},
        output_sink="output",
    )

    # Run with settings
    orchestrator = Orchestrator(db)
    result = orchestrator.run(pipeline, graph=graph, settings=settings)

    # Export sink should have received audit records
    assert len(export_sink.captured_rows) > 0
    assert any(r.get("record_type") == "run" for r in export_sink.captured_rows)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_orchestrator.py::test_orchestrator_exports_landscape_when_configured -v`
Expected: FAIL (Orchestrator.run doesn't accept settings parameter)

### Step 3: Write minimal implementation

```python
# src/elspeth/engine/orchestrator.py - modify run() method

from elspeth.core.config import ElspethSettings
from elspeth.core.landscape.exporter import LandscapeExporter

class Orchestrator:
    def run(
        self,
        config: PipelineConfig,
        graph: ExecutionGraph | None = None,
        settings: ElspethSettings | None = None,
    ) -> RunResult:
        """Execute a pipeline run.

        Args:
            config: Pipeline configuration with plugins
            graph: Pre-validated execution graph (required)
            settings: Full settings (for post-run hooks like export)
        """
        # ... existing code ...

        try:
            with self._span_factory.run_span(run.run_id):
                result = self._execute_run(recorder, run.run_id, config, graph)

            # Complete run
            recorder.complete_run(run.run_id, status="completed")
            result.status = "completed"

            # Post-run export
            if settings and settings.landscape.export.enabled:
                self._export_landscape(
                    run_id=run.run_id,
                    settings=settings,
                    sinks=config.sinks,
                )

            return result
        # ... rest of method ...

    def _export_landscape(
        self,
        run_id: str,
        settings: ElspethSettings,
        sinks: dict[str, Any],
    ) -> None:
        """Export audit trail to configured sink."""
        export_config = settings.landscape.export

        # Get signing key from environment if signing enabled
        signing_key = None
        if export_config.sign:
            import os
            key_str = os.environ.get("ELSPETH_SIGNING_KEY")
            if not key_str:
                raise ValueError(
                    "ELSPETH_SIGNING_KEY environment variable required for signed export"
                )
            signing_key = key_str.encode("utf-8")

        # Create exporter
        exporter = LandscapeExporter(self._db, signing_key=signing_key)

        # Get target sink
        sink_name = export_config.sink
        if sink_name not in sinks:
            raise ValueError(f"Export sink '{sink_name}' not found")
        sink = sinks[sink_name]

        # Create context for sink
        ctx = PluginContext(run_id=run_id, config={}, landscape=None)

        # Export records to sink
        for record in exporter.export_run(run_id, sign=export_config.sign):
            sink.write(record, ctx)

        sink.flush()
```

### Step 4: Update CLI to pass settings

```python
# src/elspeth/cli.py - modify _execute_pipeline()

# Pass settings to orchestrator
result = orchestrator.run(pipeline_config, graph=graph, settings=config)
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/engine/test_orchestrator.py::test_orchestrator_exports_landscape_when_configured -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/elspeth/engine/orchestrator.py src/elspeth/cli.py tests/engine/test_orchestrator.py
git commit -m "feat(engine): integrate landscape export into Orchestrator"
```

---

## Task 6: Add Export Format Options (CSV vs JSON)

**Files:**

- Create: `src/elspeth/core/landscape/formatters.py`
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/core/landscape/test_formatters.py`

### Step 1: Write the failing test

```python
# tests/core/landscape/test_formatters.py

from elspeth.core.landscape.formatters import CSVFormatter, JSONFormatter


def test_csv_formatter_flattens_nested_fields():
    """CSV formatter should flatten nested dicts to dot notation."""
    formatter = CSVFormatter()

    record = {
        "record_type": "node_state",
        "metadata": {"attempt": 1, "reason": "retry"},
    }

    flat = formatter.flatten(record)

    assert flat["metadata.attempt"] == 1
    assert flat["metadata.reason"] == "retry"


def test_json_formatter_preserves_structure():
    """JSON formatter should preserve nested structure."""
    formatter = JSONFormatter()

    record = {
        "record_type": "node_state",
        "metadata": {"attempt": 1},
    }

    output = formatter.format(record)

    import json
    parsed = json.loads(output)
    assert parsed["metadata"]["attempt"] == 1
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_formatters.py -v`
Expected: FAIL with ImportError

### Step 3: Write minimal implementation

```python
# src/elspeth/core/landscape/formatters.py
"""Export formatters for Landscape data.

Formatters transform audit records for different output formats.
"""

import json
from typing import Any, Protocol


class ExportFormatter(Protocol):
    """Protocol for export formatters."""

    def format(self, record: dict[str, Any]) -> str | dict[str, Any]:
        """Format a record for output."""
        ...


class JSONFormatter:
    """Format records as JSON lines."""

    def format(self, record: dict[str, Any]) -> str:
        """Format as JSON line."""
        return json.dumps(record, default=str)


class CSVFormatter:
    """Format records for CSV output.

    Flattens nested structures using dot notation.
    """

    def flatten(self, record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dict to dot-notation keys."""
        result: dict[str, Any] = {}

        for key, value in record.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self.flatten(value, full_key))
            elif isinstance(value, list):
                # Convert lists to JSON strings for CSV
                result[full_key] = json.dumps(value)
            else:
                result[full_key] = value

        return result

    def format(self, record: dict[str, Any]) -> dict[str, Any]:
        """Format as flat dict for CSV."""
        return self.flatten(record)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_formatters.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/formatters.py tests/core/landscape/test_formatters.py
git commit -m "feat(landscape): add CSV and JSON formatters for export"
```

---

## Task 7: Update Architecture Documentation

**Files:**

- Modify: `docs/design/architecture.md`
- Modify: `docs/design/requirements.md`

### Step 1: Add to architecture.md (after Landscape section)

```markdown
### Audit Trail Export

For compliance and legal inquiry, the Landscape can be exported after a run completes:

```yaml
landscape:
  url: sqlite:///./runs/audit.db
  export:
    enabled: true
    sink: audit_archive       # Reference to configured sink
    format: csv               # csv or json
    sign: true                # HMAC signature per record
```

**Export flow:**
1. Run completes normally
2. Orchestrator queries all audit data for run
3. Records formatted and written to export sink
4. If `sign: true`, each record gets HMAC signature + final manifest

**Signing provides:**
- Per-record integrity verification
- Chain-of-custody proof via running hash
- Manifest with final hash for tamper detection

**Environment:**
- `ELSPETH_SIGNING_KEY`: Required for signed exports (UTF-8 encoded string)

**Redaction note:** Redaction is the responsibility of plugins BEFORE invoking Landscape recording methods. The Landscape is a faithful recorder - it stores what it's given. The export therefore exports exactly what was recorded.
```

### Step 2: Add to requirements.md

Add new requirements section:

```markdown
## LANDSCAPE EXPORT REQUIREMENTS

| Requirement ID | Requirement | Source | Status |
|----------------|-------------|--------|--------|
| EXP-001 | Export audit trail to configured sink | This plan | ✅ Implemented |
| EXP-002 | Optional HMAC signing per record | This plan | ✅ Implemented |
| EXP-003 | Manifest with final hash for tamper detection | This plan | ✅ Implemented |
| EXP-004 | CSV and JSON format options | This plan | ✅ Implemented |
| EXP-005 | Export happens post-run via config, not CLI | This plan | ✅ Implemented |
| EXP-006 | Include all record types (batches, token_parents) | Code review | ✅ Implemented |
```

### Step 3: Commit

```bash
git add docs/design/architecture.md docs/design/requirements.md
git commit -m "docs: add Landscape export requirements and architecture"
```

---

## Task 8: Add Integration Test with Example

**Files:**

- Create: `examples/audit_export/settings.yaml`
- Create: `tests/integration/test_landscape_export.py`

### Step 1: Write the integration test

```python
# tests/integration/test_landscape_export.py
"""Integration test for landscape export."""

from pathlib import Path

import pytest


@pytest.fixture
def export_settings_yaml(tmp_path: Path) -> Path:
    """Create settings file with export enabled."""
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n")

    output_csv = tmp_path / "output.csv"
    audit_csv = tmp_path / "audit_export.csv"
    db_path = tmp_path / "audit.db"

    settings = f"""
datasource:
  plugin: csv
  options:
    path: {input_csv}

sinks:
  output:
    plugin: csv
    options:
      path: {output_csv}
  audit_export:
    plugin: csv
    options:
      path: {audit_csv}

output_sink: output

landscape:
  url: sqlite:///{db_path}
  export:
    enabled: true
    sink: audit_export
    format: csv
    sign: false
"""
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(settings)
    return settings_file


def test_run_with_export_creates_audit_file(export_settings_yaml: Path):
    """Running pipeline with export enabled should create audit CSV."""
    import subprocess

    result = subprocess.run(
        ["uv", "run", "elspeth", "run", "-s", str(export_settings_yaml), "--execute"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    # Check audit export was created
    audit_csv = export_settings_yaml.parent / "audit_export.csv"
    assert audit_csv.exists()

    # Read and verify content
    content = audit_csv.read_text()
    assert "record_type" in content
    assert "run" in content
    assert "row" in content
```

### Step 2: Run test to verify it passes

Run: `pytest tests/integration/test_landscape_export.py -v`
Expected: PASS

### Step 3: Add example to examples folder

Create `examples/audit_export/settings.yaml`:

```yaml
# Example: Pipeline with audit export to CSV
#
# Run with:
#   uv run elspeth run -s examples/audit_export/settings.yaml --execute
#
# For signed exports (legal/compliance use):
#   export ELSPETH_SIGNING_KEY="your-secret-key"
#   uv run elspeth run -s examples/audit_export/settings.yaml --execute

datasource:
  plugin: csv
  options:
    path: examples/audit_export/input.csv

sinks:
  output:
    plugin: csv
    options:
      path: examples/audit_export/output/results.csv

  # Export sink for compliance
  audit_export:
    plugin: csv
    options:
      path: examples/audit_export/output/audit_trail.json

output_sink: output

landscape:
  url: sqlite:///examples/audit_export/runs/audit.db
  export:
    enabled: true
    sink: audit_export
    format: csv
    sign: false  # Set to true and provide ELSPETH_SIGNING_KEY for legal use
```

### Step 4: Commit

```bash
git add examples/audit_export/settings.yaml tests/integration/test_landscape_export.py
git commit -m "test: add landscape export integration test and example"
```

---

## Task 9: Update README with Export Feature

**Files:**

- Modify: `README.md`

### Step 1: Add Audit Export section to README

Add after the existing Landscape section (or create one if it doesn't exist):

```markdown
## Audit Trail Export

ELSPETH can automatically export the complete audit trail after each run for compliance and legal inquiry.

### Configuration

```yaml
landscape:
  url: sqlite:///./runs/audit.db
  export:
    enabled: true
    sink: audit_archive     # Must reference a defined sink
    format: csv             # csv or json
    sign: true              # HMAC signature per record

sinks:
  audit_archive:
    plugin: csv
    options:
      path: exports/audit_trail.csv
```

### Signed Exports

For legal-grade integrity verification, enable signing:

```bash
export ELSPETH_SIGNING_KEY="your-secret-key"
uv run elspeth run -s settings.yaml --execute
```

Each record receives an HMAC-SHA256 signature. A manifest record at the end contains:
- Total record count
- Running hash of all signatures
- Export timestamp

This allows auditors to:
1. Verify no records were added, removed, or modified
2. Trace any row through every processing step
3. Prove chain-of-custody for legal proceedings

### Export Record Types

The export includes all audit data:

| Record Type | Description |
|-------------|-------------|
| `run` | Run metadata (config hash, timestamps, status) |
| `node` | Registered plugins (source, transforms, sinks) |
| `edge` | Graph edges between nodes |
| `row` | Source rows with content hashes |
| `token` | Row instances in pipeline paths |
| `token_parent` | Fork/join lineage |
| `node_state` | Processing records (input/output hashes) |
| `routing_event` | Gate routing decisions |
| `call` | External API calls |
| `batch` | Aggregation batches |
| `batch_member` | Batch membership |
| `artifact` | Sink outputs |
| `manifest` | Final hash and metadata (signed exports only) |
```

### Step 2: Commit

```bash
git add README.md
git commit -m "docs(readme): add Audit Trail Export section"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 0 | Add get_edges() to LandscapeRecorder | recorder.py, test_recorder.py |
| 1 | Add export config schema | config.py, test_config.py |
| 2 | Validate export sink reference | config.py, test_config.py |
| 3 | Create LandscapeExporter class | exporter.py, test_exporter.py |
| 4 | Add HMAC signing | exporter.py, test_exporter.py |
| 5 | Integrate into Orchestrator | orchestrator.py, cli.py |
| 6 | Add format options | formatters.py, test_formatters.py |
| 7 | Update architecture docs | architecture.md, requirements.md |
| 8 | Integration test + example | test_landscape_export.py, export_settings.yaml |
| 9 | Update README | README.md |

**Total commits:** 10
**Approach:** TDD (test first, then implement)
**Review status:** GO (corrections applied from code review)
