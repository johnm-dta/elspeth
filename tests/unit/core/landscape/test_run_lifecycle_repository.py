# tests/unit/core/landscape/test_run_lifecycle_repository.py
"""Direct unit tests for RunLifecycleRepository.

These tests exercise RunLifecycleRepository directly to pin its contract
and verify Tier 1 crash paths.

Covers 3 untested branch clusters identified in review:
1. get_source_schema — non-string type rejection
2. get_source_field_resolution — corruption paths (bad JSON shape, missing key, non-dict mapping, non-string entries)
3. set_export_status — COMPLETED/PENDING/FAILED branching logic
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select, update

from elspeth.contracts import (
    Determinism,
    ExportStatus,
    NodeType,
    ReproducibilityGrade,
    RunStatus,
    SecretResolutionInput,
)
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import AuditIntegrityError, FrameworkBugError
from elspeth.contracts.plugin_policy_audit import WebPluginPolicyEvidence
from elspeth.contracts.preflight import CommencementGateResult, DependencyRunResult, PreflightResult
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.model_loaders import RunLoader
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.run_lifecycle_repository import (
    RunLifecycleRepository,
    is_valid_sha256_hex,
)
from elspeth.core.landscape.schema import run_attributions_table, run_web_plugin_policy_table, runs_table
from tests.fixtures.landscape import make_factory, make_landscape_db, make_recorder_with_run, register_test_node


def _make_repo(*, run_id: str = "run-1") -> tuple[LandscapeDB, RunLifecycleRepository]:
    """Create in-memory DB + repository with a pre-existing run."""
    db = make_landscape_db()
    ops = DatabaseOps(db)
    repo = RunLifecycleRepository(db, ops, RunLoader())
    repo.begin_run(config={"key": "value"}, canonical_version="v1", run_id=run_id)
    return db, repo


def _corrupt_column(db: LandscapeDB, run_id: str, **values: object) -> None:
    """Directly update a column in the runs table to simulate corruption."""
    with db.connection() as conn:
        conn.execute(update(runs_table).where(runs_table.c.run_id == run_id).values(**values))


def _web_policy_evidence() -> WebPluginPolicyEvidence:
    return WebPluginPolicyEvidence(
        schema_version=1,
        policy_hash="a" * 64,
        snapshot_hash="b" * 64,
        authorized_plugin_ids=("sink:json", "source:csv", "transform:llm"),
        available_plugin_ids=("sink:json", "source:csv", "transform:llm"),
        control_modes=(("content_safety", "recommend"), ("llm", "required")),
        selected_implementations=(("content_safety", None), ("llm", "transform:llm")),
        selected_profile_aliases=(("transform:llm", "tutorial"),),
        plugin_code_identities=(
            ("sink:json", "1.0.0", "sha256:1111111111111111"),
            ("source:csv", "1.0.0", "sha256:2222222222222222"),
            ("transform:llm", "1.0.0", "sha256:3333333333333333"),
        ),
        binding_generation_fingerprint="c" * 64,
        decision_codes=("policy_allowed",),
    )


# ---------------------------------------------------------------------------
# begin_run + get_run — direct repository construction
# ---------------------------------------------------------------------------


class TestBeginRunDirect:
    """Direct tests for begin_run via repository construction."""

    def test_begin_run_returns_run_with_correct_fields(self) -> None:
        """Verify begin_run stores and returns correct field values."""
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        run = repo.begin_run(
            config={"pipeline": "test"},
            canonical_version="v2",
            run_id="explicit-id",
        )
        assert run.run_id == "explicit-id"
        assert run.status == RunStatus.RUNNING
        assert run.config_hash is not None
        assert run.settings_json is not None
        assert run.canonical_version == "v2"
        assert run.started_at is not None

    def test_begin_run_generates_id_when_not_provided(self) -> None:
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        run = repo.begin_run(config={}, canonical_version="v1")
        assert run.run_id  # non-empty generated ID

    def test_begin_run_mints_leader_seat_atomically(self) -> None:
        """Epoch 21 (ADR-030 §B.4 closing line / uniformity rule).

        begin_run creates the run_coordination seat at epoch 1, registers the
        leader's run_workers row (with pid/hostname/entry_point forensics),
        and writes the worker_register + leader_acquire events — every N=1
        run, including every repository-level test fixture, exercises the
        substrate.
        """
        from sqlalchemy import select as sa_select

        from elspeth.core.landscape.schema import (
            run_coordination_events_table,
            run_coordination_table,
            run_workers_table,
        )

        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        explicit_worker = "worker:seat-mint-run:abc123"
        repo.begin_run(
            config={},
            canonical_version="v1",
            run_id="seat-mint-run",
            leader_worker_id=explicit_worker,
        )
        with db.connection() as conn:
            seat = conn.execute(sa_select(run_coordination_table).where(run_coordination_table.c.run_id == "seat-mint-run")).one()
            worker = conn.execute(sa_select(run_workers_table).where(run_workers_table.c.worker_id == explicit_worker)).one()
            event_types = (
                conn.execute(
                    sa_select(run_coordination_events_table.c.event_type)
                    .where(run_coordination_events_table.c.run_id == "seat-mint-run")
                    .order_by(run_coordination_events_table.c.seq)
                )
                .scalars()
                .all()
            )
        assert seat.leader_worker_id == explicit_worker
        assert seat.leader_epoch == 1
        assert seat.leader_heartbeat_expires_at is not None
        assert worker.role == "leader"
        assert worker.status == "active"
        assert worker.entry_point == "run"
        assert worker.pid is not None
        assert event_types == ["worker_register", "leader_acquire"]

    def test_begin_run_uses_public_leader_registration_composition_surface(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        observed_run_ids: list[str] = []

        def spy_register_run_leader_on(self: RunCoordinationRepository, *args: object, **kwargs: object) -> CoordinationToken:
            del self, args
            observed_run_ids.append(str(kwargs["run_id"]))
            return CoordinationToken(
                run_id=str(kwargs["run_id"]),
                worker_id=str(kwargs["worker_id"]),
                leader_epoch=1,
            )

        monkeypatch.setattr(
            RunCoordinationRepository,
            "register_run_leader_on",
            spy_register_run_leader_on,
            raising=False,
        )

        repo.begin_run(
            config={},
            canonical_version="v1",
            run_id="public-composition-run",
            leader_worker_id="worker:public-composition-run:abc123",
        )

        assert observed_run_ids == ["public-composition-run"]

    def test_begin_run_self_mints_worker_identity_when_omitted(self) -> None:
        """Uniformity-for-free: callers that pass no identity still get a seat."""
        from sqlalchemy import select as sa_select

        from elspeth.core.landscape.schema import run_coordination_table

        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        run = repo.begin_run(config={}, canonical_version="v1")
        with db.connection() as conn:
            seat = conn.execute(sa_select(run_coordination_table).where(run_coordination_table.c.run_id == run.run_id)).one()
        assert isinstance(seat.leader_worker_id, str)
        assert seat.leader_worker_id.startswith(f"worker:{run.run_id}:")
        assert seat.leader_epoch == 1

    def test_get_run_roundtrip(self) -> None:
        """get_run returns the same run that begin_run created."""
        _, repo = _make_repo(run_id="roundtrip-run")
        run = repo.get_run("roundtrip-run")
        assert run is not None
        assert run.run_id == "roundtrip-run"
        assert run.status == RunStatus.RUNNING

    def test_get_run_returns_none_for_unknown(self) -> None:
        _, repo = _make_repo()
        assert repo.get_run("nonexistent") is None

    def test_begin_run_completed_status_rejected(self) -> None:
        """COMPLETED must go through complete_run() so completed_at is recorded."""
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        with pytest.raises(AuditIntegrityError, match="complete_run"):
            repo.begin_run(
                config={"pipeline": "test"},
                canonical_version="v1",
                run_id="completed-at-begin",
                status=RunStatus.COMPLETED,
            )

    def test_begin_run_records_web_attribution_without_changing_config_hash(self) -> None:
        """User attribution is audit metadata, not part of the pipeline config hash."""
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        config = {"pipeline": "test"}

        attributed = repo.begin_run(
            config=config,
            canonical_version="v1",
            run_id="attributed-run",
            initiated_by_user_id="alice",
            auth_provider_type="local",
        )
        unattributed = repo.begin_run(config=config, canonical_version="v1", run_id="unattributed-run")

        with db.read_only_connection() as conn:
            row = conn.execute(select(run_attributions_table).where(run_attributions_table.c.run_id == "attributed-run")).one()

        assert attributed.config_hash == unattributed.config_hash
        assert row.initiated_by_user_id == "alice"
        assert row.auth_provider_type == "local"

    def test_begin_run_records_optional_web_plugin_policy_evidence(self) -> None:
        db = make_landscape_db()
        repo = RunLifecycleRepository(db, DatabaseOps(db), RunLoader())

        repo.begin_run(
            config={"pipeline": "test"},
            canonical_version="v1",
            run_id="web-policy-run",
            web_plugin_policy_evidence=_web_policy_evidence(),
        )

        with db.read_only_connection() as conn:
            row = conn.execute(select(run_web_plugin_policy_table).where(run_web_plugin_policy_table.c.run_id == "web-policy-run")).one()
        assert row.policy_hash == "a" * 64
        assert row.authorized_plugin_ids_json == '["sink:json","source:csv","transform:llm"]'
        assert row.selected_profile_aliases_json == '[["transform:llm","tutorial"]]'

    def test_begin_run_cli_path_does_not_fabricate_web_policy_evidence(self) -> None:
        db = make_landscape_db()
        repo = RunLifecycleRepository(db, DatabaseOps(db), RunLoader())
        repo.begin_run(config={}, canonical_version="v1", run_id="cli-run")

        with db.read_only_connection() as conn:
            rows = conn.execute(select(run_web_plugin_policy_table)).all()
        assert rows == []

    def test_web_policy_insert_rolls_back_run_attribution_and_leader_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sqlalchemy.exc import IntegrityError

        from elspeth.core.landscape.schema import run_coordination_table

        db = make_landscape_db()
        repo = RunLifecycleRepository(db, DatabaseOps(db), RunLoader())

        def reject_policy_insert(*_args: object, **_kwargs: object) -> None:
            raise IntegrityError("INSERT", {}, RuntimeError("policy audit unavailable"))

        monkeypatch.setattr(repo, "_insert_web_plugin_policy_evidence", reject_policy_insert)

        with pytest.raises(LandscapeRecordError, match="database rejected audit write"):
            repo.begin_run(
                config={},
                canonical_version="v1",
                run_id="rolled-back-web-run",
                initiated_by_user_id="alice",
                auth_provider_type="local",
                web_plugin_policy_evidence=_web_policy_evidence(),
            )

        with db.read_only_connection() as conn:
            assert conn.execute(select(runs_table).where(runs_table.c.run_id == "rolled-back-web-run")).first() is None
            assert (
                conn.execute(select(run_attributions_table).where(run_attributions_table.c.run_id == "rolled-back-web-run")).first() is None
            )
            assert (
                conn.execute(select(run_coordination_table).where(run_coordination_table.c.run_id == "rolled-back-web-run")).first() is None
            )
            assert (
                conn.execute(
                    select(run_web_plugin_policy_table).where(run_web_plugin_policy_table.c.run_id == "rolled-back-web-run")
                ).first()
                is None
            )

    def test_web_policy_evidence_foreign_key_rejects_orphan_run(self) -> None:
        from sqlalchemy.exc import IntegrityError

        db = make_landscape_db()
        repo = RunLifecycleRepository(db, DatabaseOps(db), RunLoader())

        with pytest.raises(IntegrityError), db.write_connection() as conn:
            repo._insert_web_plugin_policy_evidence(conn, run_id="missing-run", evidence=_web_policy_evidence())

    @pytest.mark.parametrize(
        ("corrupt_values", "message"),
        [
            ({"policy_hash": "A" * 64}, "corrupt"),
            ({"authorized_plugin_ids_json": '["source:csv","sink:json","transform:llm"]'}, "corrupt"),
        ],
    )
    def test_get_web_policy_evidence_rejects_hash_or_canonical_json_corruption(
        self,
        corrupt_values: dict[str, object],
        message: str,
    ) -> None:
        db = make_landscape_db()
        repo = RunLifecycleRepository(db, DatabaseOps(db), RunLoader())
        repo.begin_run(
            config={},
            canonical_version="v1",
            run_id="corrupt-policy-run",
            web_plugin_policy_evidence=_web_policy_evidence(),
        )
        with db.write_connection() as conn:
            conn.execute(
                update(run_web_plugin_policy_table)
                .where(run_web_plugin_policy_table.c.run_id == "corrupt-policy-run")
                .values(**corrupt_values)
            )

        with pytest.raises(AuditIntegrityError, match=message):
            repo.get_web_plugin_policy_evidence("corrupt-policy-run")

    def test_begin_run_rejects_partial_web_attribution(self) -> None:
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())

        with pytest.raises(AuditIntegrityError, match="initiated_by_user_id"):
            repo.begin_run(
                config={"pipeline": "test"},
                canonical_version="v1",
                run_id="partial-attribution",
                auth_provider_type="local",
            )


class TestBeginRunRuntimeValManifest:
    """ADR-010 M3 (issue elspeth-1c8185dfec): runtime VAL manifest recorded at begin_run.

    The declaration-contract + Tier-1-error registries must be serialized
    into the runs row so an auditor can later answer "which VAL contracts
    were in force during run X?" and "are TIER_1_ERRORS the same across
    runs X and Y?".

    Tests explicitly set up the registry state they need via
    ``_snapshot_registry_for_tests`` / ``_restore_registry_snapshot_for_tests``
    rather than relying on module-import side-effects; at unit-test level
    ``pass_through.py`` is not automatically imported so the production
    registry population never runs.
    """

    @staticmethod
    def _fetch_manifest(db: LandscapeDB, run_id: str) -> dict[str, object]:
        from sqlalchemy import select

        with db.connection() as conn:
            row = conn.execute(select(runs_table.c.runtime_val_manifest_json).where(runs_table.c.run_id == run_id)).fetchone()
        assert row is not None, f"runs row for {run_id!r} missing"
        manifest_json = row[0]
        assert manifest_json is not None, "runtime_val_manifest_json is NULL"
        return json.loads(manifest_json)

    def test_manifest_records_declaration_contract_registry(self) -> None:
        """Registered DeclarationContract instances appear in the manifest."""
        # Force registration by importing pass_through — the production
        # bootstrap path reaches it through Orchestrator.run(); unit tests
        # skip that so we import explicitly.
        import elspeth.contracts.declaration_contracts as dc
        from elspeth.engine.executors import pass_through  # noqa: F401  (import side-effect)

        snapshot = dc._snapshot_registry_for_tests()
        try:
            db = make_landscape_db()
            ops = DatabaseOps(db)
            repo = RunLifecycleRepository(db, ops, RunLoader())
            repo.begin_run(config={}, canonical_version="v1", run_id="m3-declarations")
            manifest = self._fetch_manifest(db, "m3-declarations")
        finally:
            dc._restore_registry_snapshot_for_tests(snapshot)

        declaration_entries = manifest["declaration_contracts"]
        assert isinstance(declaration_entries, list)
        names = {entry["name"] for entry in declaration_entries}
        assert "passes_through_input" in names
        for entry in declaration_entries:
            # H2 per-site extension: entries carry dispatch_sites in addition
            # to the 2A keys.
            assert set(entry.keys()) == {"name", "class_name", "class_module", "dispatch_sites", "implementation_hash"}
            assert entry["class_module"].startswith("elspeth.")
            assert isinstance(entry["dispatch_sites"], list)
            assert len(entry["dispatch_sites"]) >= 1
            assert entry["implementation_hash"].startswith("sha256:")

    def test_manifest_records_expected_contract_sites(self) -> None:
        """The EXPECTED_CONTRACT_SITES per-site manifest (N1) is captured verbatim."""
        from elspeth.contracts.declaration_contracts import EXPECTED_CONTRACT_SITES

        db, _ = _make_repo(run_id="m3-manifest")
        manifest = self._fetch_manifest(db, "m3-manifest")
        expected = {name: sorted(sites) for name, sites in EXPECTED_CONTRACT_SITES.items()}
        assert manifest["expected_contract_sites"] == expected

    def test_manifest_records_tier_1_errors(self) -> None:
        """Every Tier-1 error class with its reason is in the manifest."""
        from elspeth.contracts.tier_registry import _TIER_1_ERRORS_VIEW, tier_1_reason

        db, _ = _make_repo(run_id="m3-tier-1")
        manifest = self._fetch_manifest(db, "m3-tier-1")

        tier_1_entries = manifest["tier_1_errors"]
        assert isinstance(tier_1_entries, list)
        assert len(tier_1_entries) > 0  # at least FrameworkBugError + AuditIntegrityError

        recorded_by_name = {(e["class_module"], e["class_name"]): e["reason"] for e in tier_1_entries}
        for entry in tier_1_entries:
            assert entry["implementation_hash"].startswith("sha256:")
        for cls in _TIER_1_ERRORS_VIEW:
            key = (cls.__module__, cls.__name__)
            assert key in recorded_by_name, f"Tier-1 class {key} missing from manifest"
            assert recorded_by_name[key] == tier_1_reason(cls)

    def test_manifest_includes_landscape_record_error(self) -> None:
        """Recorder-failure marker types must be registered and serialized."""
        from elspeth.contracts.tier_registry import _TIER_1_ERRORS_VIEW, tier_1_reason
        from elspeth.core.landscape.errors import LandscapeRecordError

        db, _ = _make_repo(run_id="m3-landscape-record-error")
        manifest = self._fetch_manifest(db, "m3-landscape-record-error")

        assert LandscapeRecordError in _TIER_1_ERRORS_VIEW

        recorded_by_name = {(e["class_module"], e["class_name"]): e["reason"] for e in manifest["tier_1_errors"]}
        key = (LandscapeRecordError.__module__, LandscapeRecordError.__name__)
        assert key in recorded_by_name
        assert recorded_by_name[key] == tier_1_reason(LandscapeRecordError)

    def test_begin_run_fails_when_declaration_registry_is_not_frozen(self) -> None:
        """begin_run must fail closed if the declaration registry is still mutable."""
        import elspeth.contracts.declaration_contracts as dc

        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())

        snapshot = dc._snapshot_registry_for_tests()
        try:
            dc._REGISTRY[:] = []  # type: ignore[attr-defined]  # test-only patch under pytest gate
            for site_list in dc._REGISTRY_BY_SITE.values():  # type: ignore[attr-defined]  # test-only patch under pytest gate
                site_list.clear()
            dc._FROZEN = False  # type: ignore[attr-defined]  # test-only patch under pytest gate

            with pytest.raises(FrameworkBugError, match="requires frozen runtime-VAL registries"):
                repo.begin_run(config={}, canonical_version="v1", run_id="m3-unfrozen")
        finally:
            dc._restore_registry_snapshot_for_tests(snapshot)


class TestFinalizeRunDirect:
    """Direct tests for finalize_run (grade computation + completion)."""

    def test_finalize_sets_status_and_grade(self) -> None:
        """finalize_run computes grade and completes the run.

        Empty pipeline (no nodes) is trivially FULL_REPRODUCIBLE.
        """
        _, repo = _make_repo(run_id="finalize-run")
        run = repo.finalize_run("finalize-run", RunStatus.COMPLETED)
        assert run.status == RunStatus.COMPLETED
        assert run.completed_at is not None
        assert run.reproducibility_grade is not None


# ---------------------------------------------------------------------------
# get_source_schema — Tier 1 crash paths
# ---------------------------------------------------------------------------


class TestGetSourceSchema:
    """Direct tests for get_source_schema Tier 1 validation."""

    def test_returns_stored_schema(self) -> None:
        """Happy path: schema stored via begin_run is retrievable."""
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        schema_json = '{"type": "object", "properties": {}}'
        repo.begin_run(
            config={"key": "value"},
            canonical_version="v1",
            run_id="run-1",
            source_schema_json=schema_json,
        )
        assert repo.get_source_schema("run-1") == schema_json

    def test_run_not_found_raises(self) -> None:
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="not found"):
            repo.get_source_schema("nonexistent-run")

    def test_null_schema_raises(self) -> None:
        _, repo = _make_repo()
        # Run created without source_schema_json → column is NULL
        with pytest.raises(AuditIntegrityError, match="no source schema stored"):
            repo.get_source_schema("run-1")

    @pytest.mark.skip(reason="SQLite type affinity coerces int→str in TEXT columns; branch unreachable with SQLite backend")
    def test_non_string_schema_raises(self) -> None:
        """Tier 1: non-string source_schema_json must crash.

        This branch is defense-in-depth for stricter backends (PostgreSQL)
        that preserve column types. SQLite silently coerces 42 to '42'.
        """
        db, repo = _make_repo()
        _corrupt_column(db, "run-1", source_schema_json=42)
        with pytest.raises(AuditIntegrityError, match="expected str"):
            repo.get_source_schema("run-1")


# ---------------------------------------------------------------------------
# get_source_field_resolution — Tier 1 corruption paths
# ---------------------------------------------------------------------------


class TestGetSourceFieldResolution:
    """Direct tests for get_source_field_resolution Tier 1 validation."""

    def test_roundtrip_happy_path(self) -> None:
        _, repo = _make_repo()
        mapping = {"Original Header": "original_header", "Amount (USD)": "amount_usd"}
        repo.record_source_field_resolution("run-1", mapping, "v1")
        result = repo.get_source_field_resolution("run-1")
        assert result == mapping

    def test_returns_none_when_no_resolution_stored(self) -> None:
        _, repo = _make_repo()
        assert repo.get_source_field_resolution("run-1") is None

    def test_run_not_found_raises(self) -> None:
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="not found"):
            repo.get_source_field_resolution("nonexistent-run")

    def test_corrupt_non_dict_json_raises(self) -> None:
        """Tier 1: resolution JSON that isn't a dict must crash."""
        db, repo = _make_repo()
        _corrupt_column(db, "run-1", source_field_resolution_json='"just a string"')
        with pytest.raises(AuditIntegrityError, match="expected dict"):
            repo.get_source_field_resolution("run-1")

    def test_corrupt_array_json_raises(self) -> None:
        """Tier 1: resolution JSON that is an array must crash."""
        db, repo = _make_repo()
        _corrupt_column(db, "run-1", source_field_resolution_json="[1, 2, 3]")
        with pytest.raises(AuditIntegrityError, match="expected dict"):
            repo.get_source_field_resolution("run-1")

    def test_missing_resolution_mapping_key_raises(self) -> None:
        """Tier 1: dict without resolution_mapping key is corruption."""
        db, repo = _make_repo()
        _corrupt_column(db, "run-1", source_field_resolution_json='{"wrong_key": {}}')
        with pytest.raises(AuditIntegrityError, match="missing required key"):
            repo.get_source_field_resolution("run-1")

    def test_resolution_mapping_not_dict_raises(self) -> None:
        """Tier 1: resolution_mapping that isn't a dict must crash."""
        db, repo = _make_repo()
        bad_json = json.dumps({"resolution_mapping": "not a dict", "normalization_version": None})
        _corrupt_column(db, "run-1", source_field_resolution_json=bad_json)
        with pytest.raises(AuditIntegrityError, match="expected dict"):
            repo.get_source_field_resolution("run-1")

    def test_non_string_value_raises(self) -> None:
        """Tier 1: non-string values in resolution mapping must crash.

        Note: JSON keys are always strings after json.loads(), so the key-type
        check in production is defense-in-depth. This test exercises the value
        type check which IS reachable via corrupted JSON.
        """
        db, repo = _make_repo()
        bad_json = json.dumps({"resolution_mapping": {"header": 42}, "normalization_version": None})
        _corrupt_column(db, "run-1", source_field_resolution_json=bad_json)
        with pytest.raises(AuditIntegrityError, match="expected str->str"):
            repo.get_source_field_resolution("run-1")

    def test_non_string_null_value_raises(self) -> None:
        """Tier 1: null values in resolution mapping must crash."""
        db, repo = _make_repo()
        bad_json = json.dumps({"resolution_mapping": {"header": None}, "normalization_version": None})
        _corrupt_column(db, "run-1", source_field_resolution_json=bad_json)
        with pytest.raises(AuditIntegrityError, match="expected str->str"):
            repo.get_source_field_resolution("run-1")

    def test_corrupt_unparseable_json_raises(self) -> None:
        """Tier 1: syntactically broken JSON in resolution column must crash.

        This exercises the json.JSONDecodeError catch (Fix 2) — distinct from
        the structurally-wrong-JSON tests above which test post-parse validation.
        """
        db, repo = _make_repo()
        _corrupt_column(db, "run-1", source_field_resolution_json="{not valid json!!!")
        with pytest.raises(AuditIntegrityError, match="Corrupt field resolution JSON"):
            repo.get_source_field_resolution("run-1")

    def test_resume_resolution_uses_source_scoped_mapping_when_present(self) -> None:
        """Multi-source resume uses run_sources, not the overwritten run-level singleton."""
        setup = make_recorder_with_run(run_id="run-1", source_node_id="source_orders", source_plugin_name="csv")
        register_test_node(
            setup.data_flow,
            setup.run_id,
            "source_refunds",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
        )
        shared_mapping = {"Order ID": "order_id", "Amount": "amount"}
        setup.run_lifecycle.record_source_field_resolution(
            setup.run_id,
            {"Refund ID": "refund_id", "Amount": "amount"},
            "v1",
        )
        for source_node_id, source_name in (
            ("source_orders", "orders"),
            ("source_refunds", "refunds"),
        ):
            setup.run_lifecycle.record_run_source(
                run_id=setup.run_id,
                source_node_id=source_node_id,
                source_name=source_name,
                plugin_name="csv",
                config_hash=source_name,
                lifecycle_state="loaded",
                field_resolution_mapping=shared_mapping,
                normalization_version="v1",
            )

        assert setup.run_lifecycle.get_resume_field_resolution(setup.run_id) == shared_mapping

    def test_resume_resolution_rejects_distinct_multi_source_mappings(self) -> None:
        """One sink-level mapping cannot safely represent two original header sets."""
        setup = make_recorder_with_run(run_id="run-1", source_node_id="source_orders", source_plugin_name="csv")
        register_test_node(
            setup.data_flow,
            setup.run_id,
            "source_refunds",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
        )
        setup.run_lifecycle.record_source_field_resolution(
            setup.run_id,
            {"Refund ID": "refund_id", "Amount": "amount"},
            "v1",
        )
        setup.run_lifecycle.record_run_source(
            run_id=setup.run_id,
            source_node_id="source_orders",
            source_name="orders",
            plugin_name="csv",
            config_hash="orders",
            lifecycle_state="loaded",
            field_resolution_mapping={"Order ID": "order_id", "Amount": "amount"},
            normalization_version="v1",
        )
        setup.run_lifecycle.record_run_source(
            run_id=setup.run_id,
            source_node_id="source_refunds",
            source_name="refunds",
            plugin_name="csv",
            config_hash="refunds",
            lifecycle_state="loaded",
            field_resolution_mapping={"Refund ID": "refund_id", "Amount": "amount"},
            normalization_version="v1",
        )

        with pytest.raises(AuditIntegrityError, match="different original-header mappings"):
            setup.run_lifecycle.get_resume_field_resolution(setup.run_id)

    def test_resume_resolution_rejects_missing_multi_source_mapping(self) -> None:
        """Missing per-source mapping is ambiguous once multiple sources exist."""
        setup = make_recorder_with_run(run_id="run-1", source_node_id="source_orders", source_plugin_name="csv")
        register_test_node(
            setup.data_flow,
            setup.run_id,
            "source_refunds",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
        )
        setup.run_lifecycle.record_run_source(
            run_id=setup.run_id,
            source_node_id="source_orders",
            source_name="orders",
            plugin_name="csv",
            config_hash="orders",
            lifecycle_state="loaded",
            field_resolution_mapping={"Order ID": "order_id"},
            normalization_version="v1",
        )
        setup.run_lifecycle.record_run_source(
            run_id=setup.run_id,
            source_node_id="source_refunds",
            source_name="refunds",
            plugin_name="csv",
            config_hash="refunds",
            lifecycle_state="loaded",
        )

        with pytest.raises(AuditIntegrityError, match="missing for source"):
            setup.run_lifecycle.get_resume_field_resolution(setup.run_id)


class TestRecordSourceFieldResolutionNonexistentRun:
    """record_source_field_resolution on a nonexistent run must crash."""

    def test_nonexistent_run_raises_audit_integrity(self) -> None:
        """Writing field resolution to a nonexistent run must raise AuditIntegrityError.

        The error comes from execute_update() detecting zero affected rows.
        """
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="Cannot record source field resolution for run 'ghost-run'"):
            repo.record_source_field_resolution(
                "ghost-run",
                {"header": "field"},
                "v1",
            )


# ---------------------------------------------------------------------------
# complete_run — terminal status validation
# ---------------------------------------------------------------------------


class TestCompleteRun:
    """Direct tests for complete_run terminal status enforcement."""

    def test_completed_status_accepted(self) -> None:
        _, repo = _make_repo()
        run = repo.complete_run("run-1", RunStatus.COMPLETED)
        assert run.status == RunStatus.COMPLETED

    def test_failed_status_accepted(self) -> None:
        _, repo = _make_repo()
        run = repo.complete_run("run-1", RunStatus.FAILED)
        assert run.status == RunStatus.FAILED

    def test_interrupted_status_accepted(self) -> None:
        _, repo = _make_repo()
        run = repo.complete_run("run-1", RunStatus.INTERRUPTED)
        assert run.status == RunStatus.INTERRUPTED

    def test_running_status_rejected(self) -> None:
        """Non-terminal RUNNING status must be rejected."""
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="terminal status"):
            repo.complete_run("run-1", RunStatus.RUNNING)

    # Phase 2.2 (elspeth-0de989c56d): four-value terminal taxonomy must
    # round-trip through Landscape — write the value, read it back,
    # confirm the persisted enum equals the input enum.

    def test_completed_with_failures_status_accepted_and_round_trips(self) -> None:
        _, repo = _make_repo(run_id="run-cwf")
        run = repo.complete_run("run-cwf", RunStatus.COMPLETED_WITH_FAILURES)
        assert run.status == RunStatus.COMPLETED_WITH_FAILURES
        # Read back via get_run — confirm persistence preserves the value
        # verbatim through the SQL string round-trip.
        reread = repo.get_run("run-cwf")
        assert reread is not None
        assert reread.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_empty_status_accepted_and_round_trips(self) -> None:
        _, repo = _make_repo(run_id="run-empty")
        run = repo.complete_run("run-empty", RunStatus.EMPTY)
        assert run.status == RunStatus.EMPTY
        reread = repo.get_run("run-empty")
        assert reread is not None
        assert reread.status == RunStatus.EMPTY

    def test_new_terminal_statuses_block_re_completion(self) -> None:
        """COMPLETED_WITH_FAILURES and EMPTY are terminal — same immutability
        guarantee as the pre-existing terminal statuses."""
        _, repo = _make_repo(run_id="run-immut-cwf")
        repo.complete_run("run-immut-cwf", RunStatus.COMPLETED_WITH_FAILURES)
        with pytest.raises(AuditIntegrityError, match="already terminal"):
            repo.complete_run("run-immut-cwf", RunStatus.COMPLETED)

        _, repo2 = _make_repo(run_id="run-immut-empty")
        repo2.complete_run("run-immut-empty", RunStatus.EMPTY)
        with pytest.raises(AuditIntegrityError, match="already terminal"):
            repo2.complete_run("run-immut-empty", RunStatus.FAILED)


# ---------------------------------------------------------------------------
# set_export_status — branching logic
# ---------------------------------------------------------------------------


class TestSetExportStatus:
    """Direct tests for set_export_status branching."""

    def test_completed_sets_exported_at(self) -> None:
        _, repo = _make_repo()
        repo.set_export_status("run-1", ExportStatus.COMPLETED)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.export_status == ExportStatus.COMPLETED
        assert run.exported_at is not None

    def test_completed_clears_stale_error(self) -> None:
        _, repo = _make_repo()
        # Set a FAILED status with error first
        repo.set_export_status("run-1", ExportStatus.FAILED, error="network timeout")
        # Now complete — should clear the error
        repo.set_export_status("run-1", ExportStatus.COMPLETED)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.export_error is None

    def test_pending_clears_stale_error(self) -> None:
        _, repo = _make_repo()
        repo.set_export_status("run-1", ExportStatus.FAILED, error="disk full")
        repo.set_export_status("run-1", ExportStatus.PENDING)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.export_error is None

    def test_pending_clears_exported_at(self) -> None:
        """COMPLETED -> PENDING must clear stale completion evidence."""
        _, repo = _make_repo()
        repo.set_export_status("run-1", ExportStatus.COMPLETED)
        repo.set_export_status("run-1", ExportStatus.PENDING)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.export_status == ExportStatus.PENDING
        assert run.exported_at is None

    def test_failed_with_error(self) -> None:
        _, repo = _make_repo()
        repo.set_export_status("run-1", ExportStatus.FAILED, error="connection refused")
        run = repo.get_run("run-1")
        assert run is not None
        assert run.export_status == ExportStatus.FAILED
        assert run.export_error == "connection refused"

    def test_failed_clears_exported_at(self) -> None:
        """COMPLETED -> FAILED must also clear stale completion evidence."""
        _, repo = _make_repo()
        repo.set_export_status("run-1", ExportStatus.COMPLETED)
        repo.set_export_status("run-1", ExportStatus.FAILED, error="connection refused")
        run = repo.get_run("run-1")
        assert run is not None
        assert run.export_status == ExportStatus.FAILED
        assert run.exported_at is None

    def test_export_format_and_sink_stored(self) -> None:
        _, repo = _make_repo()
        repo.set_export_status(
            "run-1",
            ExportStatus.COMPLETED,
            export_format="csv",
            export_sink="output_sink",
        )
        run = repo.get_run("run-1")
        assert run is not None
        assert run.export_format == "csv"
        assert run.export_sink == "output_sink"

    def test_completed_with_error_raises_integrity_error(self) -> None:
        """Tier 1: COMPLETED + error is contradictory audit state."""
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="only valid with FAILED"):
            repo.set_export_status("run-1", ExportStatus.COMPLETED, error="something")

    def test_pending_with_error_raises_integrity_error(self) -> None:
        """Tier 1: PENDING + error is contradictory audit state."""
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="only valid with FAILED"):
            repo.set_export_status("run-1", ExportStatus.PENDING, error="something")

    def test_nonexistent_run_raises_audit_integrity(self) -> None:
        """Setting export status on a nonexistent run must crash with context."""
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="not found"):
            repo.set_export_status("ghost-run", ExportStatus.COMPLETED)

    def test_nonexistent_run_error_includes_status(self) -> None:
        """Error message includes the requested export status for debugging."""
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="failed"):
            repo.set_export_status("ghost-run", ExportStatus.FAILED, error="oops")


# ---------------------------------------------------------------------------
# record_secret_resolutions — atomicity
# ---------------------------------------------------------------------------


class TestRecordSecretResolutions:
    """Direct tests for record_secret_resolutions atomicity."""

    @staticmethod
    def _make_resolution(env_var: str = "API_KEY") -> SecretResolutionInput:
        return SecretResolutionInput(
            env_var_name=env_var,
            source="keyvault",
            vault_url="https://vault.example.com",
            secret_name=f"{env_var.lower()}-secret",
            timestamp=1709100000.0,
            resolution_latency_ms=42.5,
            fingerprint="a" * 64,  # Valid 64-char lowercase hex (HMAC-SHA256)
        )

    def test_all_resolutions_committed(self) -> None:
        """Normal path: all resolutions stored atomically."""
        _, repo = _make_repo()
        resolutions = [
            self._make_resolution("KEY_1"),
            self._make_resolution("KEY_2"),
            self._make_resolution("KEY_3"),
        ]
        repo.record_secret_resolutions("run-1", resolutions)
        stored = repo.get_secret_resolutions_for_run("run-1")
        assert len(stored) == 3
        assert {r.env_var_name for r in stored} == {"KEY_1", "KEY_2", "KEY_3"}

    def test_empty_list_is_noop(self) -> None:
        """Empty resolutions list should not error."""
        _, repo = _make_repo()
        repo.record_secret_resolutions("run-1", [])
        stored = repo.get_secret_resolutions_for_run("run-1")
        assert len(stored) == 0

    def test_nonexistent_run_raises_landscape_record_error(self) -> None:
        """Missing run_id should surface as an audit-layer write failure."""
        _, repo = _make_repo()
        with pytest.raises(LandscapeRecordError, match="record_secret_resolutions run_id=ghost-run"):
            repo.record_secret_resolutions("ghost-run", [self._make_resolution()])

    def test_atomicity_on_failure(self) -> None:
        """If any insert fails, no resolutions should be persisted.

        We simulate failure by inserting a duplicate resolution_id mid-batch.
        Since record_secret_resolutions uses a single transaction, the entire
        batch should roll back.
        """
        from unittest.mock import patch

        from elspeth.core.landscape._helpers import generate_id as real_generate_id

        _db, repo = _make_repo()
        resolutions = [
            self._make_resolution("KEY_1"),
            self._make_resolution("KEY_2"),
        ]

        # Make generate_id return the same ID twice to trigger a PK violation
        call_count = 0
        fixed_id = real_generate_id()

        def duplicate_id() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return fixed_id  # Same ID for both — second will violate PK
            return real_generate_id()

        with (
            patch("elspeth.core.landscape.run_lifecycle_repository.generate_id", side_effect=duplicate_id),
            pytest.raises(LandscapeRecordError, match="record_secret_resolutions run_id=run-1"),
        ):
            repo.record_secret_resolutions("run-1", resolutions)

        # Verify atomicity: zero records should be stored
        stored = repo.get_secret_resolutions_for_run("run-1")
        assert len(stored) == 0


# ---------------------------------------------------------------------------
# complete_run — crash path coverage
# ---------------------------------------------------------------------------


class TestCompleteRunCrashPath:
    """Tests for complete_run edge cases and crash paths."""

    def test_nonexistent_run_raises(self) -> None:
        """Completing a nonexistent run must crash."""
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="run not found"):
            repo.complete_run("nonexistent-run", RunStatus.COMPLETED)

    def test_complete_preserves_existing_grade_when_none_passed(self) -> None:
        """complete_run with reproducibility_grade=None preserves existing grade.

        Bug 318f74: previously, passing None would overwrite an existing grade
        with NULL. Now the grade column is only included in the UPDATE when
        explicitly provided.
        """
        db, repo = _make_repo()
        # Set a grade via direct column update (simulating begin_run with grade)
        _corrupt_column(db, "run-1", reproducibility_grade="full_reproducible")
        run = repo.complete_run("run-1", RunStatus.COMPLETED)
        assert run.status == RunStatus.COMPLETED
        assert run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_double_completion_rejected(self) -> None:
        """Already-terminal run cannot be completed again.

        Bug 3c77199a70: complete_run() must enforce terminal immutability.
        Once a run reaches COMPLETED/FAILED/INTERRUPTED, the terminal status
        and completed_at timestamp are the legal record and must not be
        overwritten.
        """
        _, repo = _make_repo()
        repo.complete_run("run-1", RunStatus.COMPLETED)
        with pytest.raises(AuditIntegrityError, match="already terminal"):
            repo.complete_run("run-1", RunStatus.FAILED)

    def test_completed_to_completed_rejected(self) -> None:
        """Even same-status double completion is rejected (timestamp overwrite)."""
        _, repo = _make_repo()
        repo.complete_run("run-1", RunStatus.COMPLETED)
        with pytest.raises(AuditIntegrityError, match="already terminal"):
            repo.complete_run("run-1", RunStatus.COMPLETED)

    def test_failed_to_completed_rejected(self) -> None:
        """FAILED run cannot be re-completed as COMPLETED (outcome falsification)."""
        _, repo = _make_repo()
        repo.complete_run("run-1", RunStatus.FAILED)
        with pytest.raises(AuditIntegrityError, match="already terminal"):
            repo.complete_run("run-1", RunStatus.COMPLETED)

    def test_interrupted_then_resume_then_complete_allowed(self) -> None:
        """Resume path: INTERRUPTED → RUNNING (via update_run_status) → COMPLETED.

        The resume path transitions out of terminal state first, then
        complete_run sees RUNNING and succeeds.
        """
        _, repo = _make_repo()
        repo.complete_run("run-1", RunStatus.INTERRUPTED)
        # Resume: transition back to RUNNING first
        repo.update_run_status("run-1", RunStatus.RUNNING)
        # Now complete_run should succeed
        run = repo.complete_run("run-1", RunStatus.COMPLETED)
        assert run.status == RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# update_run_status — backward-transition guard
# ---------------------------------------------------------------------------


class TestUpdateRunStatus:
    """Direct tests for update_run_status transition guards."""

    def test_running_to_running_accepted(self) -> None:
        """Non-terminal to non-terminal transition is valid."""
        _, repo = _make_repo()
        repo.update_run_status("run-1", RunStatus.RUNNING)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.status == RunStatus.RUNNING

    def test_nonexistent_run_raises(self) -> None:
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="not found"):
            repo.update_run_status("ghost-run", RunStatus.RUNNING)

    def test_completed_to_running_rejected(self) -> None:
        """COMPLETED runs are immutable — cannot be transitioned."""
        _, repo = _make_repo()
        repo.complete_run("run-1", RunStatus.COMPLETED)
        with pytest.raises(AuditIntegrityError, match="COMPLETED"):
            repo.update_run_status("run-1", RunStatus.RUNNING)

    def test_completed_status_rejected(self) -> None:
        """COMPLETED must go through complete_run() so completed_at is recorded."""
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="complete_run"):
            repo.update_run_status("run-1", RunStatus.COMPLETED)

    @pytest.mark.parametrize("status", [RunStatus.COMPLETED_WITH_FAILURES, RunStatus.EMPTY])
    def test_success_terminal_status_rejected(self, status: RunStatus) -> None:
        """Successful terminal statuses must go through complete_run()."""
        _, repo = _make_repo()
        with pytest.raises(AuditIntegrityError, match="complete_run"):
            repo.update_run_status("run-1", status)

    @pytest.mark.parametrize("status", [RunStatus.COMPLETED_WITH_FAILURES, RunStatus.EMPTY])
    def test_success_terminal_to_running_rejected(self, status: RunStatus) -> None:
        """Successful terminal runs are immutable — cannot be reopened."""
        _, repo = _make_repo()
        repo.complete_run("run-1", status)
        completed = repo.get_run("run-1")
        assert completed is not None
        assert completed.completed_at is not None

        with pytest.raises(AuditIntegrityError, match=status.value):
            repo.update_run_status("run-1", RunStatus.RUNNING)

        reread = repo.get_run("run-1")
        assert reread is not None
        assert reread.status == status
        assert reread.completed_at == completed.completed_at

    def test_failed_to_running_allowed_for_resume(self) -> None:
        """FAILED→RUNNING is the resume path — must be allowed."""
        _, repo = _make_repo()
        repo.complete_run("run-1", RunStatus.FAILED)
        # Resume path: set back to RUNNING
        repo.update_run_status("run-1", RunStatus.RUNNING)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.status == RunStatus.RUNNING

    def test_interrupted_to_running_allowed_for_resume(self) -> None:
        """INTERRUPTED→RUNNING is the resume path — must be allowed."""
        _, repo = _make_repo()
        repo.complete_run("run-1", RunStatus.INTERRUPTED)
        repo.update_run_status("run-1", RunStatus.RUNNING)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.status == RunStatus.RUNNING

    def test_failed_to_running_clears_completed_at(self) -> None:
        """Regression: FAILED→RUNNING must clear completed_at atomically.

        elspeth-55696f7fa5: previously, update_run_status only set status,
        leaving completed_at set — creating an impossible state where a run
        is simultaneously RUNNING and has a completion timestamp.
        """
        _, repo = _make_repo()
        repo.complete_run("run-1", RunStatus.FAILED)
        # Verify completed_at is set after failure
        run = repo.get_run("run-1")
        assert run is not None
        assert run.completed_at is not None

        # Resume: FAILED → RUNNING
        repo.update_run_status("run-1", RunStatus.RUNNING)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.status == RunStatus.RUNNING
        assert run.completed_at is None  # Must be cleared

    def test_interrupted_to_running_clears_completed_at(self) -> None:
        """INTERRUPTED→RUNNING must also clear completed_at."""
        _, repo = _make_repo()
        repo.complete_run("run-1", RunStatus.INTERRUPTED)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.completed_at is not None

        repo.update_run_status("run-1", RunStatus.RUNNING)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.status == RunStatus.RUNNING
        assert run.completed_at is None


# ---------------------------------------------------------------------------
# finalize_run — nondeterministic and failed edge cases
# ---------------------------------------------------------------------------


class TestFinalizeRunEdgeCases:
    """Tests for finalize_run with varied node configurations."""

    def test_finalize_failed_run(self) -> None:
        """finalize_run with FAILED status still computes grade and completes."""
        _, repo = _make_repo(run_id="fail-run")
        run = repo.finalize_run("fail-run", RunStatus.FAILED)
        assert run.status == RunStatus.FAILED
        assert run.completed_at is not None
        assert run.reproducibility_grade is not None

    def test_finalize_nondeterministic_run(self) -> None:
        """finalize_run with nondeterministic nodes yields REPLAY_REPRODUCIBLE."""
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig

        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        repo.begin_run(config={}, canonical_version="v1", run_id="nd-run")

        # Register a nondeterministic node via the factory (need DataFlowRepository)
        factory = make_factory(db)
        factory.data_flow.register_node(
            run_id="nd-run",
            plugin_name="llm_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="nd-node",
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
            determinism=Determinism.EXTERNAL_CALL,
        )

        run = repo.finalize_run("nd-run", RunStatus.COMPLETED)
        assert run.status == RunStatus.COMPLETED
        assert run.reproducibility_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE


# ---------------------------------------------------------------------------
# list_runs — ordering guarantee
# ---------------------------------------------------------------------------


class TestListRuns:
    """Direct tests for list_runs ordering and filtering."""

    def test_returns_newest_first(self) -> None:
        """list_runs returns runs ordered by started_at descending."""
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        repo.begin_run(config={}, canonical_version="v1", run_id="run-1")
        repo.begin_run(config={}, canonical_version="v1", run_id="run-2")
        repo.begin_run(config={}, canonical_version="v1", run_id="run-3")
        runs = repo.list_runs()
        assert len(runs) == 3
        # Newest first (last created = first returned)
        assert runs[0].run_id == "run-3"
        assert runs[1].run_id == "run-2"
        assert runs[2].run_id == "run-1"

    def test_filter_by_status(self) -> None:
        """list_runs with status filter only returns matching runs."""
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        repo.begin_run(config={}, canonical_version="v1", run_id="r1")
        repo.begin_run(config={}, canonical_version="v1", run_id="r2")
        repo.complete_run("r1", RunStatus.COMPLETED)
        running = repo.list_runs(status=RunStatus.RUNNING)
        assert len(running) == 1
        assert running[0].run_id == "r2"


# ---------------------------------------------------------------------------
# set_export_status — FAILED without error edge case
# ---------------------------------------------------------------------------


class TestSetExportStatusEdgeCases:
    """Edge case tests for set_export_status behavior."""

    def test_failed_without_error_does_not_set_error(self) -> None:
        """FAILED status without error kwarg leaves export_error as None."""
        _, repo = _make_repo()
        repo.set_export_status("run-1", ExportStatus.FAILED)
        run = repo.get_run("run-1")
        assert run is not None
        assert run.export_status == ExportStatus.FAILED
        assert run.export_error is None

    def test_failed_replaces_previous_error(self) -> None:
        """FAILED with new error replaces previous error message."""
        _, repo = _make_repo()
        repo.set_export_status("run-1", ExportStatus.FAILED, error="first error")
        repo.set_export_status("run-1", ExportStatus.FAILED, error="second error")
        run = repo.get_run("run-1")
        assert run is not None
        assert run.export_error == "second error"


class TestPreflightAuditWriteErrors:
    """Regression tests for preflight/readiness audit write error normalization."""

    def test_record_preflight_results_missing_run_raises_landscape_record_error(self) -> None:
        """Preflight writes should not leak raw SQLAlchemy errors."""
        _, repo = _make_repo()
        preflight = PreflightResult(
            dependency_runs=(DependencyRunResult(name="dep", run_id="dep-run", settings_hash="hash", duration_ms=1, indexed_at="now"),),
            gate_results=(CommencementGateResult(name="gate", condition="x", result=True, context_snapshot={}),),
        )
        with pytest.raises(LandscapeRecordError, match="record_preflight_results run_id=ghost-run"):
            repo.record_preflight_results("ghost-run", preflight)

    def test_record_readiness_check_missing_run_raises_landscape_record_error(self) -> None:
        """Readiness writes should not leak raw SQLAlchemy errors."""
        _, repo = _make_repo()
        with pytest.raises(LandscapeRecordError, match="record_readiness_check run_id=ghost-run"):
            repo.record_readiness_check(
                "ghost-run",
                name="probe",
                collection="docs",
                reachable=True,
                count=1,
                message="ok",
            )


class TestSha256HexValidator:
    """Pin the shape of the sha256-hex validator used by Tier-1 write guards."""

    def test_canonical_digest_is_accepted(self) -> None:
        """A real hashlib.sha256 hex digest passes validation."""
        import hashlib

        digest = hashlib.sha256(b"audit-anchor").hexdigest()
        assert is_valid_sha256_hex(digest)

    def test_all_zero_hex_is_accepted(self) -> None:
        """The conftest synthetic snapshot ('0' * 64) must pass — many tests rely on it."""
        assert is_valid_sha256_hex("0" * 64)

    def test_non_hex_string_is_rejected(self) -> None:
        """Non-hex characters fail even at the right length."""
        # 64 chars but with non-hex 'z' — rejected.
        assert not is_valid_sha256_hex("z" * 64)
        # The spec's example: a non-empty non-hex string that the old
        # ``.strip()`` guard would have admitted.
        assert not is_valid_sha256_hex("not-a-sha")

    def test_wrong_length_is_rejected(self) -> None:
        """60 chars of hex is still rejected — exact length required."""
        assert not is_valid_sha256_hex("a" * 60)
        assert not is_valid_sha256_hex("a" * 65)

    def test_uppercase_hex_is_rejected(self) -> None:
        """sha256 hexdigest is lowercase by contract; uppercase signals a bug."""
        assert not is_valid_sha256_hex("A" * 64)

    def test_empty_and_whitespace_rejected(self) -> None:
        assert not is_valid_sha256_hex("")
        assert not is_valid_sha256_hex(" " * 64)


class TestBeginRunOpenrouterCatalogSnapshotValidation:
    """Pin the write-side guard for ``openrouter_catalog_sha256``.

    The guard is the audit-trail integrity check: a non-empty but
    non-hex string must crash rather than corrupting the runs row.
    """

    def test_begin_run_rejects_non_hex_sha256(self) -> None:
        """A non-empty non-hex string passes ``.strip()`` but fails hex validation."""
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        with pytest.raises(AuditIntegrityError, match="64 lowercase hex chars"):
            repo.begin_run(
                config={"pipeline": "test"},
                canonical_version="v1",
                run_id="bad-sha-run",
                openrouter_catalog_sha256="not-a-sha",
                openrouter_catalog_source="bundled",
            )

    def test_begin_run_rejects_none_sha256(self) -> None:
        """``None`` fails the ``type(...) is not str`` guard inside the validator."""
        db = make_landscape_db()
        ops = DatabaseOps(db)
        repo = RunLifecycleRepository(db, ops, RunLoader())
        with pytest.raises(AuditIntegrityError, match="64 lowercase hex chars"):
            repo.begin_run(
                config={"pipeline": "test"},
                canonical_version="v1",
                run_id="none-sha-run",
                openrouter_catalog_sha256=None,  # type: ignore[arg-type]
                openrouter_catalog_source="bundled",
            )


class TestWriteRepositoryOpenrouterCatalogSnapshotValidation:
    """Pin the same hex-shape guard at the synthesised-run write site."""

    def test_record_synthesised_run_rejects_non_hex_sha256(self) -> None:
        from datetime import UTC, datetime

        from elspeth.contracts import NodeType
        from elspeth.contracts.synthesised_audit import SynthesisedNodeSpec
        from elspeth.core.landscape.write_repository import LandscapeWriteRepository

        db = make_landscape_db()
        repo = LandscapeWriteRepository(db)
        node_specs = (
            SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="csv_file", plugin_version="1.0"),
            SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="json_file", plugin_version="1.0"),
        )
        with pytest.raises(LandscapeRecordError, match="openrouter_catalog_sha256 must be 64 lowercase hex chars"):
            repo.record_synthesised_run(
                pipeline_yaml="version: 1",
                rows=(),
                source_data_hash="0" * 64,
                llm_call_count=0,
                node_specs=node_specs,
                started_at=datetime.now(UTC),
                metadata={"seeded_from_cache": True, "cache_key": "c" * 64},
                openrouter_catalog_sha256="not-a-sha",
                openrouter_catalog_source="bundled",
            )


# ---------------------------------------------------------------------------
# Immutability backstop beneath the epoch fence + complete_run diagnosis order
# (ADR-030 §D / §H — slice 2 test campaign §4)
# ---------------------------------------------------------------------------


def _make_repo_with_token(*, run_id: str = "run-fenced") -> tuple[LandscapeDB, RunLifecycleRepository, CoordinationToken]:
    """Run + epoch-1 leader seat + the matching (CURRENT, VALID) token.

    ``begin_run`` mints the seat atomically with the runs row (uniformity
    rule); passing ``leader_worker_id`` lets the test construct the epoch-1
    token without a read-back — exactly what the engine does.
    """
    db = make_landscape_db()
    ops = DatabaseOps(db)
    repo = RunLifecycleRepository(db, ops, RunLoader())
    worker_id = f"worker:{run_id}:unit-fence"
    repo.begin_run(config={"key": "value"}, canonical_version="v1", run_id=run_id, leader_worker_id=worker_id)
    return db, repo, CoordinationToken(run_id=run_id, worker_id=worker_id, leader_epoch=1)


def _bump_seat_epoch(db: LandscapeDB, run_id: str) -> None:
    """The in-DB image of a takeover: depose the token holder by epoch bump."""
    from elspeth.core.landscape.schema import run_coordination_table

    with db.engine.begin() as conn:
        conn.execute(
            update(run_coordination_table)
            .where(run_coordination_table.c.run_id == run_id)
            .values(leader_epoch=run_coordination_table.c.leader_epoch + 1)
        )


class TestImmutabilityBackstopBeneathEpochFence:
    """ADR-030 §B.4 closing line: the epoch fence did NOT replace immutability.

    The durable backstop the design retains beneath the resume() entry guard
    (which now refuses terminal runs itself — the §H test-#1 flip): a
    perfectly-fenced caller holding a CURRENT, VALID token still cannot
    mutate a successful terminal run.
    """

    def test_update_run_status_from_completed_refused_even_with_current_epoch_token(self) -> None:
        _db, repo, token = _make_repo_with_token(run_id="run-immut-fenced")
        repo.complete_run("run-immut-fenced", RunStatus.COMPLETED, token=token)

        # The token is still CURRENT (complete_run does not release the
        # seat), so the fence passes — the refusal below is the immutability
        # guard, not the fence.
        with pytest.raises(AuditIntegrityError, match=r"from COMPLETED .*immutable"):
            repo.update_run_status("run-immut-fenced", RunStatus.RUNNING, token=token)

        run = repo.get_run("run-immut-fenced")
        assert run is not None
        assert run.status == RunStatus.COMPLETED

    def test_update_run_status_immutable_success_target_still_refused_with_token(self) -> None:
        """Setting an immutable-success status via update_run_status stays refused
        (the :922-926 arm), token or no token — completed_at must be recorded
        via complete_run()."""
        _db, repo, token = _make_repo_with_token(run_id="run-immut-target")
        with pytest.raises(AuditIntegrityError, match="complete_run"):
            repo.update_run_status("run-immut-target", RunStatus.COMPLETED, token=token)


class TestCompleteRunDiagnosisOrder:
    """§D rowcount-0 diagnosis order (design :316), pinned arm by arm:

    already-terminal ⇒ AuditIntegrityError (wins even over a stale fence);
    fence mismatch on a non-terminal run ⇒ RunLeadershipLostError;
    residual scheduler work ⇒ OrchestrationInvariantError.
    """

    def test_already_terminal_wins_over_fence_diagnosis(self) -> None:
        """A deposed leader finalizing an ALREADY-TERMINAL run gets the
        immutability diagnosis, not the coordination one — terminal
        diagnosis wins over fence diagnosis."""
        from elspeth.contracts.errors import RunLeadershipLostError

        db, repo, token = _make_repo_with_token(run_id="run-diag-terminal")
        repo.complete_run("run-diag-terminal", RunStatus.COMPLETED, token=token)
        _bump_seat_epoch(db, "run-diag-terminal")  # token is now STALE

        with pytest.raises(AuditIntegrityError, match="already terminal") as exc_info:
            repo.complete_run("run-diag-terminal", RunStatus.FAILED, token=token)
        assert not isinstance(exc_info.value, RunLeadershipLostError)

        run = repo.get_run("run-diag-terminal")
        assert run is not None
        assert run.status == RunStatus.COMPLETED, "the stale finalize mutated nothing"

    def test_fence_mismatch_on_non_terminal_run_is_leadership_lost(self) -> None:
        from elspeth.contracts.errors import RunLeadershipLostError

        db, repo, token = _make_repo_with_token(run_id="run-diag-fence")
        _bump_seat_epoch(db, "run-diag-fence")

        with pytest.raises(RunLeadershipLostError):
            repo.complete_run("run-diag-fence", RunStatus.COMPLETED, token=token)

        run = repo.get_run("run-diag-fence")
        assert run is not None
        assert run.status == RunStatus.RUNNING, "the refused finalize mutated nothing"

    def test_residual_work_with_valid_token_is_orchestration_invariant(self) -> None:
        """One READY journal row + a SUCCESS finalize under a VALID token ⇒
        the in-statement §D quiescence arm refuses (OrchestrationInvariantError)."""
        from datetime import UTC, datetime

        from elspeth.contracts.errors import OrchestrationInvariantError
        from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
        from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

        db, repo, token = _make_repo_with_token(run_id="run-diag-residual")
        factory = make_factory(db)
        source_node = register_test_node(factory.data_flow, "run-diag-residual", "src-1", node_type=NodeType.SOURCE)
        transform_node = register_test_node(factory.data_flow, "run-diag-residual", "t-1")
        row = factory.data_flow.create_row(
            run_id="run-diag-residual",
            source_node_id=source_node,
            row_index=0,
            data={"id": 1},
            source_row_index=0,
            ingest_sequence=0,
        )
        journal_token = factory.data_flow.create_token(row_id=row.row_id)
        TokenSchedulerRepository(db.engine).enqueue_ready(
            run_id="run-diag-residual",
            token_id=journal_token.token_id,
            row_id=row.row_id,
            node_id=transform_node,
            step_index=1,
            ingest_sequence=0,
            row_payload_json=TokenSchedulerRepository.serialize_row_payload(
                PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
            ),
            available_at=datetime.now(UTC),
        )

        with pytest.raises(OrchestrationInvariantError, match="residual scheduler work"):
            repo.complete_run("run-diag-residual", RunStatus.COMPLETED, token=token)

        run = repo.get_run("run-diag-residual")
        assert run is not None
        assert run.status == RunStatus.RUNNING
