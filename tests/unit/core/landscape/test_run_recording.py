"""Tests for run lifecycle recording (via RecorderFactory).

Tests cover:
- begin_run (creates run, generates ID, stores config hash, settings JSON)
- complete_run (sets status and completed_at)
- get_run (retrieves run by ID, returns None for unknown)
- get_source_schema (retrieves/validates source schema JSON)
- record/get_source_field_resolution (roundtrip, Tier 1 validation)
- update_run_status (intermediate status changes)
- record/get_secret_resolutions (roundtrip)
- list_runs (with status filter)
- set_export_status (COMPLETED clears error, PENDING clears error)
- finalize_run (computes grade + completes)
"""

from __future__ import annotations

import json

import pytest

from elspeth.contracts import ExportStatus, FieldContract, NodeType, RunStatus, SchemaContract, SecretResolutionInput
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.reproducibility import ReproducibilityGrade
from tests.fixtures.landscape import make_factory, make_landscape_db

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _setup(*, run_id: str = "run-1") -> tuple[LandscapeDB, RecorderFactory]:
    """Create in-memory DB with a run."""
    db = make_landscape_db()
    factory = make_factory(db)
    factory.run_lifecycle.begin_run(config={"key": "value"}, canonical_version="v1", run_id=run_id)
    return db, factory


class TestBeginRun:
    """Tests for begin_run — creates a new run record."""

    def test_returns_run_with_id(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="my-run")
        assert run.run_id == "my-run"

    def test_generates_id_if_not_provided(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        assert run.run_id is not None
        assert len(run.run_id) == 32  # UUID hex

    def test_stores_config_hash(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={"key": "value"}, canonical_version="v1", run_id="r1")
        assert run.config_hash is not None
        assert len(run.config_hash) > 0

    def test_stores_settings_json(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={"key": "value"}, canonical_version="v1", run_id="r1")
        settings = json.loads(run.settings_json)
        assert settings == {"key": "value"}

    def test_initial_status_is_running(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        assert run.status == RunStatus.RUNNING

    def test_stores_reproducibility_grade(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            reproducibility_grade=ReproducibilityGrade.FULL_REPRODUCIBLE,
        )
        assert run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE


class TestGetRun:
    """Tests for get_run — retrieves run by ID."""

    def test_returns_run(self) -> None:
        _db, factory = _setup()
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.run_id == "run-1"

    def test_returns_none_for_unknown(self) -> None:
        _db, factory = _setup()
        assert factory.run_lifecycle.get_run("nonexistent") is None


class TestCompleteRun:
    """Tests for complete_run — finalizes a run."""

    def test_sets_completed_status(self) -> None:
        _db, factory = _setup()
        run = factory.run_lifecycle.complete_run("run-1", RunStatus.COMPLETED)
        assert run.status == RunStatus.COMPLETED
        assert run.completed_at is not None

    def test_sets_failed_status(self) -> None:
        _db, factory = _setup()
        run = factory.run_lifecycle.complete_run("run-1", RunStatus.FAILED)
        assert run.status == RunStatus.FAILED

    def test_stores_reproducibility_grade(self) -> None:
        _db, factory = _setup()
        run = factory.run_lifecycle.complete_run(
            "run-1",
            RunStatus.COMPLETED,
            reproducibility_grade=ReproducibilityGrade.FULL_REPRODUCIBLE,
        )
        assert run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_rejects_non_terminal_status_running(self) -> None:
        _db, factory = _setup()
        with pytest.raises(AuditIntegrityError, match="terminal status"):
            factory.run_lifecycle.complete_run("run-1", RunStatus.RUNNING)

    def test_accepts_interrupted_status(self) -> None:
        _db, factory = _setup()
        run = factory.run_lifecycle.complete_run("run-1", RunStatus.INTERRUPTED)
        assert run.status == RunStatus.INTERRUPTED
        assert run.completed_at is not None

    def test_accepts_completed_status(self) -> None:
        _db, factory = _setup()
        run = factory.run_lifecycle.complete_run("run-1", RunStatus.COMPLETED)
        assert run.status == RunStatus.COMPLETED
        assert run.completed_at is not None

    def test_accepts_failed_status(self) -> None:
        _db, factory = _setup()
        run = factory.run_lifecycle.complete_run("run-1", RunStatus.FAILED)
        assert run.status == RunStatus.FAILED
        assert run.completed_at is not None


class TestGetSourceSchema:
    """Tests for get_source_schema — retrieves source schema for resume."""

    def test_returns_schema_json(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        schema_json = '{"type": "object", "properties": {"id": {"type": "integer"}}}'
        factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            run_id="r1",
            source_schema_json=schema_json,
        )
        result = factory.run_lifecycle.get_source_schema("r1")
        assert result == schema_json

    def test_raises_for_unknown_run(self) -> None:
        _db, factory = _setup()
        with pytest.raises(AuditIntegrityError, match="Run nonexistent not found"):
            factory.run_lifecycle.get_source_schema("nonexistent")

    def test_raises_for_missing_schema(self) -> None:
        _db, factory = _setup()
        with pytest.raises(AuditIntegrityError, match="no source schema stored"):
            factory.run_lifecycle.get_source_schema("run-1")


class TestSourceFieldResolution:
    """Tests for record/get_source_field_resolution — header mapping roundtrip."""

    def test_roundtrip(self) -> None:
        _db, factory = _setup()
        mapping = {"Customer ID": "customer_id", "Amount ($)": "amount"}
        factory.run_lifecycle.record_source_field_resolution("run-1", mapping, normalization_version="v1")
        result = factory.run_lifecycle.get_source_field_resolution("run-1")
        assert result == mapping

    def test_returns_none_when_not_recorded(self) -> None:
        _db, factory = _setup()
        result = factory.run_lifecycle.get_source_field_resolution("run-1")
        assert result is None

    def test_raises_for_unknown_run(self) -> None:
        _db, factory = _setup()
        with pytest.raises(AuditIntegrityError, match="not found"):
            factory.run_lifecycle.get_source_field_resolution("nonexistent")


class TestUpdateRunStatus:
    """Tests for update_run_status — intermediate status changes."""

    def test_updates_status(self) -> None:
        _db, factory = _setup()
        factory.run_lifecycle.update_run_status("run-1", RunStatus.FAILED)
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.status == RunStatus.FAILED

    def test_does_not_set_completed_at(self) -> None:
        _db, factory = _setup()
        factory.run_lifecycle.update_run_status("run-1", RunStatus.FAILED)
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.completed_at is None


class TestSecretResolutions:
    """Tests for record/get_secret_resolutions — Key Vault audit records."""

    def test_roundtrip(self) -> None:
        _db, factory = _setup()
        resolutions = [
            SecretResolutionInput(
                env_var_name="API_KEY",
                source="keyvault",
                vault_url="https://vault.example.com",
                secret_name="api-key",
                timestamp=1705320000.0,
                resolution_latency_ms=150.0,
                fingerprint="a" * 64,
            ),
        ]
        factory.run_lifecycle.record_secret_resolutions("run-1", resolutions)
        results = factory.run_lifecycle.get_secret_resolutions_for_run("run-1")
        assert len(results) == 1
        assert results[0].env_var_name == "API_KEY"
        assert results[0].fingerprint == "a" * 64
        assert results[0].vault_url == "https://vault.example.com"

    def test_empty_resolutions(self) -> None:
        _db, factory = _setup()
        results = factory.run_lifecycle.get_secret_resolutions_for_run("run-1")
        assert results == []


class TestListRuns:
    """Tests for list_runs — lists runs with optional filter."""

    def test_lists_all_runs(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="r1")
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="r2")
        runs = factory.run_lifecycle.list_runs()
        assert len(runs) == 2

    def test_filter_by_status(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="r1")
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="r2")
        factory.run_lifecycle.complete_run("r1", RunStatus.COMPLETED)
        running = factory.run_lifecycle.list_runs(status=RunStatus.RUNNING)
        assert len(running) == 1
        assert running[0].run_id == "r2"


class TestSetExportStatus:
    """Tests for set_export_status — export status management."""

    def test_sets_completed_status(self) -> None:
        _db, factory = _setup()
        factory.run_lifecycle.set_export_status("run-1", ExportStatus.COMPLETED, export_format="json")
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.export_status == ExportStatus.COMPLETED
        assert run.exported_at is not None
        assert run.export_format == "json"

    def test_sets_failed_with_error(self) -> None:
        _db, factory = _setup()
        factory.run_lifecycle.set_export_status("run-1", ExportStatus.FAILED, error="disk full")
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.export_status == ExportStatus.FAILED
        assert run.export_error == "disk full"

    def test_completed_clears_stale_error(self) -> None:
        _db, factory = _setup()
        factory.run_lifecycle.set_export_status("run-1", ExportStatus.FAILED, error="first attempt failed")
        factory.run_lifecycle.set_export_status("run-1", ExportStatus.COMPLETED)
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.export_status == ExportStatus.COMPLETED
        assert run.export_error is None

    def test_pending_clears_stale_error(self) -> None:
        _db, factory = _setup()
        factory.run_lifecycle.set_export_status("run-1", ExportStatus.FAILED, error="old error")
        factory.run_lifecycle.set_export_status("run-1", ExportStatus.PENDING)
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.export_error is None


class TestFinalizeRun:
    """Tests for finalize_run — computes grade + completes."""

    def test_finalize_deterministic_run(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="n1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        run = factory.run_lifecycle.finalize_run("run-1", RunStatus.COMPLETED)
        assert run.status == RunStatus.COMPLETED
        assert run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE
        assert run.completed_at is not None


# ===========================================================================
# Bug 7.3: Run contract hash verification
# ===========================================================================


class TestGetRunContractHashVerification:
    """Tests for get_run_contract hash verification (Bug 7.3)."""

    def _make_contract(self) -> SchemaContract:
        return SchemaContract(
            mode="FIXED",
            fields=(
                FieldContract(
                    normalized_name="id",
                    original_name="id",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )

    def test_get_run_contract_returns_contract(self) -> None:
        """get_run_contract() round-trips a valid contract."""
        _db, factory = _setup()
        contract = self._make_contract()
        factory.run_lifecycle.update_run_contract("run-1", contract)
        restored = factory.run_lifecycle.get_run_contract("run-1")
        assert restored is not None
        assert restored.mode == "FIXED"
        assert len(restored.fields) == 1

    def test_get_run_contract_verifies_hash(self) -> None:
        """get_run_contract() raises AuditIntegrityError on hash mismatch (Bug 7.3)."""
        from sqlalchemy import update

        from elspeth.core.landscape.schema import runs_table

        _db, factory = _setup()
        contract = self._make_contract()
        factory.run_lifecycle.update_run_contract("run-1", contract)

        # Tamper with the stored hash
        with _db.connection() as conn:
            conn.execute(update(runs_table).where(runs_table.c.run_id == "run-1").values(schema_contract_hash="tampered_hash_value"))

        with pytest.raises(AuditIntegrityError, match="hash mismatch"):
            factory.run_lifecycle.get_run_contract("run-1")

    def test_get_run_contract_returns_none_when_no_contract(self) -> None:
        """get_run_contract() returns None when no contract stored."""
        _db, factory = _setup()
        result = factory.run_lifecycle.get_run_contract("run-1")
        assert result is None
