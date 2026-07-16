"""Recovery proofs for the dedicated zero-member audit-export effect path."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import cast
from unittest.mock import patch

import pytest
from sqlalchemy import func, select

from elspeth.contracts import NodeType
from elspeth.contracts.audit_export import (
    AuditExportContentDescriptor,
    AuditExportContentStoreResolver,
    IterableBoundAuditExportContentReader,
    RegisteredAuditExportContent,
)
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    AuditExportFormat,
    RestrictedSinkEffectContext,
    SinkEffectAuditExportSnapshotInput,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
)
from elspeth.core.audit_export_content_store import FilesystemAuditExportContentStore
from elspeth.core.config import AuditExportContentStoreSettings, LandscapeExportSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import audit_export_snapshots_table, runs_table
from elspeth.engine.executors.sink_effects import SinkEffectExecutionSeam, SinkEffectInjectedFault
from elspeth.engine.orchestrator.audit_export_effects import execute_audit_export_effect, prepare_audit_export_snapshot
from tests.fixtures.landscape import register_test_node

_COMPLETED_AT = datetime(2026, 7, 16, 7, 8, 9, 123456, tzinfo=UTC)


class _MemoryContentStore:
    def __init__(self, content_store_id: str = "audit-store-v1") -> None:
        self.content_store_id = content_store_id
        self.namespace = "audit/export"
        self.content: dict[str, bytes] = {}
        self.put_count = 0
        self.orphans: list[tuple[str, tuple[AuditExportContentDescriptor, ...]]] = []

    def is_durable(self) -> bool:
        return True

    def put_immutable(self, content: bytes, *, candidate_id: str, object_kind: str) -> str:
        del candidate_id, object_kind
        self.put_count += 1
        content_ref = f"sha256:{sha256(content).hexdigest()}"
        self.content.setdefault(content_ref, content)
        return content_ref

    def open_registered(self, registration: RegisteredAuditExportContent) -> IterableBoundAuditExportContentReader:
        assert registration.content_store_id == self.content_store_id
        return IterableBoundAuditExportContentReader(self.content[registration.descriptor.content_ref])

    def mark_candidate_orphans(
        self,
        candidate_id: str,
        descriptors: tuple[AuditExportContentDescriptor, ...],
    ) -> None:
        self.orphans.append((candidate_id, descriptors))

    def garbage_collect_candidate(self, request: object) -> bool:
        del request
        return False


@dataclass(slots=True)
class _Target:
    publication_count: int = 0
    effect_id: str | None = None
    descriptor: ArtifactDescriptor | None = None
    content: bytes | None = None


class _AuditExportSink:
    def __init__(self, target: _Target) -> None:
        self.target = target

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        del request, ctx
        return SinkEffectInspection(
            mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            reference="audit-export-inspection:v1",
            evidence=MappingProxyType({}),
        )

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        snapshot = cast(SinkEffectAuditExportSnapshotInput, request.effect_input)
        content = b"".join(snapshot.reader.iter_verified_chunks()) + snapshot.reader.read_verified_signed_manifest()
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///tmp/audit-export.json",
            content_hash=sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        return SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=request.input_kind,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=request.inspection.mode,
            target=descriptor.path_or_uri,
            plan_hash=stable_hash(
                {
                    "content_hash": descriptor.content_hash,
                    "effect_id": request.effect_id,
                    "schema": "audit-export-test-plan-v1",
                }
            ),
            payload_hash=descriptor.content_hash,
            expected_descriptor=descriptor,
            safe_evidence={"inspection_reference": request.inspection.reference},
        )

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult:
        del ctx
        assert plan.expected_descriptor is not None
        self.target.publication_count += 1
        self.target.effect_id = plan.effect_id
        self.target.descriptor = plan.expected_descriptor
        return SinkEffectCommitResult(
            descriptor=plan.expected_descriptor,
            evidence={"effect_id": plan.effect_id},
            accepted_ordinals=(),
            diverted_ordinals=(),
        )

    def reconcile_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectReconcileResult:
        del ctx
        if self.target.effect_id is None:
            return SinkEffectReconcileResult.not_applied(evidence={"target": "absent"})
        if self.target.effect_id == plan.effect_id and self.target.descriptor == plan.expected_descriptor:
            assert self.target.descriptor is not None
            return SinkEffectReconcileResult.applied(self.target.descriptor, evidence={"effect_id": plan.effect_id})
        return SinkEffectReconcileResult.unknown(evidence={"target": "divergent"})


def _insert_terminal_run(db: LandscapeDB, run_id: str = "run-export") -> None:
    with db.engine.begin() as connection:
        connection.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=_COMPLETED_AT,
                completed_at=_COMPLETED_AT,
                config_hash="0" * 64,
                settings_json="{}",
                canonical_version="v1",
                status="completed",
                openrouter_catalog_sha256="1" * 64,
                openrouter_catalog_source="bundled",
            )
        )


def _config(**overrides: object) -> LandscapeExportSettings:
    config = LandscapeExportSettings(
        enabled=True,
        sink="output",
        format="json",
        signing_mode="unsigned",
        signer_key_id="UNSIGNED",
        exporter_version="landscape-exporter-v1",
        serialization_version="audit-export-v2",
        chunking_algorithm_version="record-framing-v1",
        total_record_limit=10_000,
        total_byte_limit=10 * 1024 * 1024,
        chunk_limit=100,
        per_chunk_record_limit=100,
        per_chunk_byte_limit=1024 * 1024,
        spool_root=Path(".elspeth/audit-export-spool/recovery"),
        spool_cleanup_age_seconds=3600,
        spool_cleanup_byte_budget=10 * 1024 * 1024,
        spool_cleanup_count_budget=100,
        content_store=AuditExportContentStoreSettings(
            content_store_id="audit-store-v1",
            namespace="audit/export",
            root=Path(".elspeth/audit-export-content-store/recovery"),
            policy_version="v1",
            retention_days=30,
            durability="fsync",
            orphan_grace_period_seconds=3600,
        ),
    )
    return config.model_copy(update=overrides)


@pytest.mark.parametrize(
    ("overrides", "error"),
    (
        ({"total_record_limit": 1, "per_chunk_record_limit": 1}, "max_total_records"),
        ({"total_byte_limit": 700, "per_chunk_byte_limit": 700}, "max_total_bytes"),
        ({"chunk_limit": 1, "per_chunk_record_limit": 1}, "max_chunks"),
    ),
)
def test_configured_total_limits_fail_before_content_store_or_registry_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, int],
    error: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'bounded.db'}")
    store = _MemoryContentStore()
    try:
        _insert_terminal_run(db)
        register_test_node(RecorderFactory(db).data_flow, "run-export", "one-record-is-already-too-many")

        with pytest.raises(ValueError, match=error):
            prepare_audit_export_snapshot(
                db,
                run_id="run-export",
                config=_config(**overrides),
                signing_key=None,
                content_store=store,
            )

        assert store.put_count == 0
        with db.engine.connect() as connection:
            assert connection.scalar(select(func.count()).select_from(audit_export_snapshots_table)) == 0
    finally:
        db.close()


def test_candidate_verification_reads_run_outside_the_write_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk/manifest reread and graph verification must complete before the
    short registry write/CAS transaction begins (elspeth-107ecfec1c)."""
    monkeypatch.chdir(tmp_path)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'verify-outside.db'}")
    store = _MemoryContentStore()
    in_write_transaction = False
    verification_reads_under_write_lock: list[str] = []

    original_open = store.open_registered

    def observing_open(registration: RegisteredAuditExportContent) -> IterableBoundAuditExportContentReader:
        if in_write_transaction:
            verification_reads_under_write_lock.append(registration.descriptor.content_ref)
        return original_open(registration)

    store.open_registered = observing_open  # type: ignore[method-assign]

    real_write_connection = db.write_connection

    @contextmanager
    def observing_write_connection():  # type: ignore[no-untyped-def]
        nonlocal in_write_transaction
        in_write_transaction = True
        try:
            with real_write_connection() as connection:
                yield connection
        finally:
            in_write_transaction = False

    monkeypatch.setattr(db, "write_connection", observing_write_connection)
    try:
        _insert_terminal_run(db)
        snapshot = prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=_config(),
            signing_key=None,
            content_store=store,
        )

        assert snapshot.snapshot_id
        assert verification_reads_under_write_lock == []
    finally:
        db.close()


def test_registry_hit_reuses_verified_winner_without_rewriting_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'reuse.db'}")
    store = _MemoryContentStore()
    try:
        _insert_terminal_run(db)
        first = prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=_config(),
            signing_key=None,
            content_store=store,
        )
        put_count = store.put_count
        with patch(
            "elspeth.core.landscape.export_read_model.ConnectionBoundExportReadModel.get_export_terminal_witness",
            side_effect=lambda *_args, **_kwargs: pytest.fail("registry hit must not query the source run"),
        ):
            second = prepare_audit_export_snapshot(
                db,
                run_id="run-export",
                config=_config(),
                signing_key=None,
                content_store=store,
            )

        assert first.snapshot_id == second.snapshot_id
        assert store.put_count == put_count
        assert b"".join(second.reader.iter_verified_chunks()).endswith(b"\n")
        assert second.reader.read_verified_signed_manifest().endswith(b"}")
    finally:
        db.close()


def test_production_filesystem_store_materializes_and_reopens_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'filesystem-store.db'}")
    config = _config()
    assert config.content_store is not None
    store = FilesystemAuditExportContentStore(config.content_store)
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    try:
        _insert_terminal_run(db)
        first = prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=config,
            signing_key=None,
            content_store=store,
            content_store_resolver=resolver,
        )
        second = prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=config,
            signing_key=None,
            content_store=store,
            content_store_resolver=resolver,
        )

        assert second.snapshot_id == first.snapshot_id
        assert b"".join(second.reader.iter_verified_chunks()).endswith(b"\n")
        assert second.reader.read_verified_signed_manifest().endswith(b"}")
    finally:
        db.close()


def test_hmac_snapshot_streaming_derivation_and_production_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'hmac.db'}")
    store = _MemoryContentStore()
    try:
        _insert_terminal_run(db)
        snapshot = prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=_config(
                signing_mode="hmac_sha256",
                signer_key_id="audit-key-v1",
                signing_secret_ref="AUDIT_EXPORT_TEST_KEY",
            ),
            signing_key=b"integration-signing-key",
            content_store=store,
        )

        record = json.loads(next(snapshot.reader.iter_verified_chunks()))
        manifest = json.loads(snapshot.reader.read_verified_signed_manifest())
        assert isinstance(record["signature"], str) and len(record["signature"]) == 64
        assert isinstance(manifest["signature"], str) and len(manifest["signature"]) == 64
    finally:
        db.close()


def test_single_export_rotation_policy_refuses_a_different_signer_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'single-export.db'}")
    store = _MemoryContentStore()
    try:
        _insert_terminal_run(db)
        prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=_config(
                signing_mode="hmac_sha256",
                signer_key_id="audit-key-v1",
                signing_secret_ref="AUDIT_EXPORT_TEST_KEY_V1",
                signer_rotation_policy="single_export",
            ),
            signing_key=b"first-signing-key",
            content_store=store,
        )

        with pytest.raises(ValueError, match="single_export"):
            prepare_audit_export_snapshot(
                db,
                run_id="run-export",
                config=_config(
                    signing_mode="hmac_sha256",
                    signer_key_id="audit-key-v2",
                    signing_secret_ref="AUDIT_EXPORT_TEST_KEY_V2",
                    signer_rotation_policy="single_export",
                ),
                signing_key=b"second-signing-key",
                content_store=store,
            )

        with db.engine.connect() as connection:
            assert connection.scalar(select(func.count()).select_from(audit_export_snapshots_table)) == 1
    finally:
        db.close()


def test_rotated_store_reuses_prior_winner_only_through_persistent_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'rotation.db'}")
    old_store = _MemoryContentStore("audit-store-v1")
    new_store = _MemoryContentStore("audit-store-v2")
    resolver = AuditExportContentStoreResolver()
    try:
        _insert_terminal_run(db)
        first = prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=_config(),
            signing_key=None,
            content_store=old_store,
            content_store_resolver=resolver,
        )
        second = prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=_config(),
            signing_key=None,
            content_store=new_store,
            content_store_resolver=resolver,
        )

        assert second.snapshot_id == first.snapshot_id
        assert new_store.put_count == 0
        assert b"".join(second.reader.iter_verified_chunks()).endswith(b"\n")

        with pytest.raises(LookupError, match=r"audit-store-v1.*unresolvable"):
            prepare_audit_export_snapshot(
                db,
                run_id="run-export",
                config=_config(),
                signing_key=None,
                content_store=new_store,
                content_store_resolver=AuditExportContentStoreResolver(),
            )
    finally:
        db.close()


@pytest.mark.parametrize(
    "seam",
    (
        SinkEffectExecutionSeam.BEFORE_EFFECT,
        SinkEffectExecutionSeam.AFTER_EFFECT_BEFORE_RETURN,
        SinkEffectExecutionSeam.AFTER_RETURN_BEFORE_FINALIZE,
    ),
)
def test_interrupted_audit_export_effect_reuses_snapshot_and_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seam: SinkEffectExecutionSeam,
) -> None:
    monkeypatch.chdir(tmp_path)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'recovery.db'}")
    store = _MemoryContentStore()
    try:
        _insert_terminal_run(db)
        snapshot = prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=_config(),
            signing_key=None,
            content_store=store,
        )
        factory = RecorderFactory(db)
        sink_node_id = register_test_node(
            factory.data_flow,
            "run-export",
            "audit-export",
            node_type=NodeType.SINK,
            plugin_name="audit-export-test",
        )
        target = _Target()
        injected = False

        def fail_once(observed: SinkEffectExecutionSeam) -> None:
            nonlocal injected
            if observed is seam and not injected:
                injected = True
                raise SinkEffectInjectedFault(seam)

        with pytest.raises(SinkEffectInjectedFault):
            execute_audit_export_effect(
                factory=factory,
                snapshot=snapshot,
                sink=_AuditExportSink(target),
                sink_node_id=sink_node_id,
                target_config={"path": "audit-export.json"},
                worker_id="audit-export-worker",
                fault_hook=fail_once,
            )

        result = execute_audit_export_effect(
            factory=RecorderFactory(db),
            snapshot=snapshot,
            sink=_AuditExportSink(target),
            sink_node_id=sink_node_id,
            target_config={"path": "audit-export.json"},
            worker_id="audit-export-worker",
        )

        assert target.publication_count == 1
        assert result.artifact.sink_effect_id == target.effect_id
        assert result.state_ids == () and result.outcome_ids == ()
    finally:
        db.close()


@pytest.mark.parametrize("signed", [False, True])
def test_json_sink_replays_verified_snapshot_and_exact_manifest_after_response_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signed: bool,
) -> None:
    from elspeth.plugins.sinks import _local_file_effects
    from elspeth.plugins.sinks.json_sink import JSONSink

    monkeypatch.chdir(tmp_path)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'json-adapter.db'}")
    store = _MemoryContentStore()
    try:
        _insert_terminal_run(db)
        snapshot = prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=_config(
                format="json",
                signing_mode="hmac_sha256" if signed else "unsigned",
                signer_key_id="audit-key-v1" if signed else "UNSIGNED",
                signing_secret_ref="AUDIT_EXPORT_TEST_KEY" if signed else None,
            ),
            signing_key=b"integration-signing-key" if signed else None,
            content_store=store,
        )
        assert snapshot.export_format is AuditExportFormat.JSON
        expected = b"".join(snapshot.reader.iter_verified_chunks()) + snapshot.reader.read_verified_signed_manifest()
        output = tmp_path / "audit.jsonl"
        sink_options = {"path": str(output), "format": "jsonl", "mode": "write", "schema": {"mode": "observed"}}
        factory = RecorderFactory(db)
        sink_node_id = register_test_node(
            factory.data_flow,
            "run-export",
            "audit-export-json",
            node_type=NodeType.SINK,
            plugin_name="json",
        )
        publications: list[Path] = []
        monkeypatch.setattr(_local_file_effects, "_after_replace", lambda path: publications.append(path))

        with pytest.raises(SinkEffectInjectedFault):
            execute_audit_export_effect(
                factory=factory,
                snapshot=snapshot,
                sink=JSONSink(sink_options),
                sink_node_id=sink_node_id,
                target_config=sink_options,
                worker_id="audit-export-json-worker",
                fault_hook=lambda seam: (
                    (_ for _ in ()).throw(SinkEffectInjectedFault(seam))
                    if seam is SinkEffectExecutionSeam.AFTER_RETURN_BEFORE_FINALIZE
                    else None
                ),
            )

        result = execute_audit_export_effect(
            factory=RecorderFactory(db),
            snapshot=snapshot,
            sink=JSONSink(sink_options),
            sink_node_id=sink_node_id,
            target_config=sink_options,
            worker_id="audit-export-json-worker",
        )

        assert output.read_bytes() == expected
        assert output.read_bytes().endswith(snapshot.reader.read_verified_signed_manifest())
        assert publications == [output]
        assert result.artifact.content_hash == sha256(expected).hexdigest()
    finally:
        db.close()


def test_csv_sink_recovers_exact_bundle_without_republication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.plugins.sinks import _audit_export_bundle_effects
    from elspeth.plugins.sinks.csv_sink import CSVSink

    monkeypatch.chdir(tmp_path)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'csv-adapter.db'}")
    store = _MemoryContentStore()
    try:
        _insert_terminal_run(db)
        snapshot = prepare_audit_export_snapshot(
            db,
            run_id="run-export",
            config=_config(format="csv"),
            signing_key=None,
            content_store=store,
        )
        assert snapshot.export_format is AuditExportFormat.CSV
        target = tmp_path / "audit-bundle"
        sink_options = {"path": str(target), "mode": "write", "schema": {"mode": "observed"}}
        factory = RecorderFactory(db)
        sink_node_id = register_test_node(
            factory.data_flow,
            "run-export",
            "audit-export-csv",
            node_type=NodeType.SINK,
            plugin_name="csv",
        )
        publications: list[Path] = []
        monkeypatch.setattr(
            _audit_export_bundle_effects,
            "_after_parent_fsync_before_return",
            lambda path: publications.append(path),
        )

        with pytest.raises(SinkEffectInjectedFault):
            execute_audit_export_effect(
                factory=factory,
                snapshot=snapshot,
                sink=CSVSink(sink_options),
                sink_node_id=sink_node_id,
                target_config=sink_options,
                worker_id="audit-export-csv-worker",
                fault_hook=lambda seam: (
                    (_ for _ in ()).throw(SinkEffectInjectedFault(seam))
                    if seam is SinkEffectExecutionSeam.AFTER_RETURN_BEFORE_FINALIZE
                    else None
                ),
            )

        result = execute_audit_export_effect(
            factory=RecorderFactory(db),
            snapshot=snapshot,
            sink=CSVSink(sink_options),
            sink_node_id=sink_node_id,
            target_config=sink_options,
            worker_id="audit-export-csv-worker",
        )

        assert (target / _audit_export_bundle_effects.AUDIT_MANIFEST_NAME).read_bytes() == snapshot.reader.read_verified_signed_manifest()
        assert publications == [target]
        assert result.artifact.path_or_uri.endswith("/audit-bundle")
    finally:
        db.close()
