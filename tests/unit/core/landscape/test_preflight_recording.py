"""Tests for pre-flight result recording in Landscape audit trail."""

from __future__ import annotations

import json

import pytest

from elspeth.core.dependency_config import (
    CommencementGateResult,
    DependencyRunResult,
    PreflightResult,
)
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import preflight_results_table
from elspeth.core.payload_store import FilesystemPayloadStore


@pytest.fixture()
def factory(tmp_path):
    """Create a factory with a fresh in-memory database."""
    db = LandscapeDB.from_url("sqlite:///:memory:")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    fac = RecorderFactory(db, payload_store=payload_store)
    run = fac.run_lifecycle.begin_run(config={"test": True}, canonical_version="sha256-rfc8785-v1")
    yield fac, run.run_id, db
    db.close()


class TestRecordPreflightResults:
    def test_dependency_run_recorded_and_readable(self, factory) -> None:
        fac, run_id, db = factory
        preflight = PreflightResult(
            dependency_runs=(
                DependencyRunResult(
                    name="indexer",
                    run_id="dep-abc",
                    settings_hash="sha256:deadbeef",
                    duration_ms=1500,
                    indexed_at="2026-03-25T12:00:00Z",
                ),
            ),
            gate_results=(),
        )

        fac.run_lifecycle.record_preflight_results(run_id=run_id, preflight=preflight)

        # Read back from database
        with db.connection() as conn:
            rows = conn.execute(preflight_results_table.select().where(preflight_results_table.c.run_id == run_id)).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row.result_type == "dependency_run"
        assert row.name == "indexer"
        data = json.loads(row.result_json)
        assert data["run_id"] == "dep-abc"
        assert data["settings_hash"] == "sha256:deadbeef"
        assert data["duration_ms"] == 1500

    def test_gate_result_with_nested_snapshot_recorded(self, factory) -> None:
        """Nested MappingProxyType in context_snapshot must serialize correctly (bug #1)."""
        fac, run_id, db = factory
        preflight = PreflightResult(
            dependency_runs=(),
            gate_results=(
                CommencementGateResult(
                    name="corpus_ready",
                    condition="collections['test']['count'] > 0",
                    result=True,
                    context_snapshot={
                        "collections": {"test": {"count": 42, "reachable": True}},
                        "dependency_runs": {"indexer": {"run_id": "dep-abc"}},
                    },
                ),
            ),
        )

        fac.run_lifecycle.record_preflight_results(run_id=run_id, preflight=preflight)

        with db.connection() as conn:
            rows = conn.execute(preflight_results_table.select().where(preflight_results_table.c.run_id == run_id)).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row.result_type == "commencement_gate"
        assert row.name == "corpus_ready"
        data = json.loads(row.result_json)
        assert data["condition"] == "collections['test']['count'] > 0"
        assert data["result"] is True
        # Nested snapshot must round-trip correctly
        assert data["context_snapshot"]["collections"]["test"]["count"] == 42

    def test_empty_preflight_is_noop(self, factory) -> None:
        fac, run_id, db = factory
        preflight = PreflightResult(dependency_runs=(), gate_results=())

        fac.run_lifecycle.record_preflight_results(run_id=run_id, preflight=preflight)

        with db.connection() as conn:
            rows = conn.execute(preflight_results_table.select().where(preflight_results_table.c.run_id == run_id)).fetchall()

        assert len(rows) == 0

    def test_result_json_is_canonical(self, factory) -> None:
        """result_json must use canonical JSON (RFC 8785), not standard json.dumps.

        Canonical JSON produces deterministic key ordering, which means the same
        data always produces the same hash. Standard json.dumps does not guarantee
        key order across Python versions.
        """
        fac, run_id, db = factory
        preflight = PreflightResult(
            dependency_runs=(DependencyRunResult(name="dep", run_id="r", settings_hash="h", duration_ms=100, indexed_at="t"),),
            gate_results=(),
        )

        fac.run_lifecycle.record_preflight_results(run_id=run_id, preflight=preflight)

        with db.connection() as conn:
            rows = conn.execute(preflight_results_table.select().where(preflight_results_table.c.run_id == run_id)).fetchall()

        result_json = rows[0].result_json
        # Canonical JSON (RFC 8785) sorts keys and uses no unnecessary whitespace
        assert result_json == '{"duration_ms":100,"indexed_at":"t","run_id":"r","settings_hash":"h"}'

    def test_mixed_deps_and_gates(self, factory) -> None:
        fac, run_id, db = factory
        preflight = PreflightResult(
            dependency_runs=(
                DependencyRunResult(name="dep1", run_id="r1", settings_hash="h1", duration_ms=100, indexed_at="t1"),
                DependencyRunResult(name="dep2", run_id="r2", settings_hash="h2", duration_ms=200, indexed_at="t2"),
            ),
            gate_results=(CommencementGateResult(name="gate1", condition="True", result=True, context_snapshot={}),),
        )

        fac.run_lifecycle.record_preflight_results(run_id=run_id, preflight=preflight)

        with db.connection() as conn:
            rows = conn.execute(
                preflight_results_table.select().where(preflight_results_table.c.run_id == run_id).order_by(preflight_results_table.c.name)
            ).fetchall()

        assert len(rows) == 3
        types = {r.result_type for r in rows}
        assert types == {"dependency_run", "commencement_gate"}
        names = [r.name for r in rows]
        assert "dep1" in names
        assert "dep2" in names
        assert "gate1" in names


class TestRecordReadinessCheck:
    def test_readiness_check_recorded_and_readable(self, factory) -> None:
        """Readiness check result persists in the preflight_results table."""
        fac, run_id, db = factory

        fac.run_lifecycle.record_readiness_check(
            run_id=run_id,
            name="rag_retrieval",
            collection="test-index",
            reachable=True,
            count=42,
            message="Collection 'test-index' has 42 documents",
        )

        with db.connection() as conn:
            rows = conn.execute(preflight_results_table.select().where(preflight_results_table.c.run_id == run_id)).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row.result_type == "readiness_check"
        assert row.name == "rag_retrieval"
        data = json.loads(row.result_json)
        assert data["collection"] == "test-index"
        assert data["reachable"] is True
        assert data["count"] == 42
        assert "42 documents" in data["message"]

    def test_readiness_json_is_canonical(self, factory) -> None:
        """result_json must use canonical JSON for deterministic hashing."""
        fac, run_id, db = factory

        fac.run_lifecycle.record_readiness_check(
            run_id=run_id,
            name="rag",
            collection="c",
            reachable=True,
            count=1,
            message="m",
        )

        with db.connection() as conn:
            rows = conn.execute(preflight_results_table.select().where(preflight_results_table.c.run_id == run_id)).fetchall()

        result_json = rows[0].result_json
        # Canonical JSON (RFC 8785) sorts keys
        assert result_json == '{"collection":"c","count":1,"message":"m","reachable":true}'


class TestPreflightResult:
    def test_construction(self) -> None:
        dep = DependencyRunResult(name="x", run_id="r", settings_hash="h", duration_ms=0, indexed_at="t")
        gate = CommencementGateResult(name="g", condition="True", result=True, context_snapshot={})
        pf = PreflightResult(dependency_runs=(dep,), gate_results=(gate,))
        assert len(pf.dependency_runs) == 1
        assert len(pf.gate_results) == 1

    def test_empty_tuples(self) -> None:
        pf = PreflightResult(dependency_runs=(), gate_results=())
        assert pf.dependency_runs == ()
        assert pf.gate_results == ()

    def test_frozen(self) -> None:
        pf = PreflightResult(dependency_runs=(), gate_results=())
        with pytest.raises(AttributeError):
            pf.dependency_runs = ()  # type: ignore[misc]
