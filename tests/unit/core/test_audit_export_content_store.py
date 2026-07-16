from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from elspeth.contracts.audit_export import (
    AuditExportContentDescriptor,
    AuditExportOrphanCollectionRequest,
    RegisteredAuditExportContent,
)
from elspeth.core.audit_export_content_store import FilesystemAuditExportContentStore, create_audit_export_content_store
from elspeth.core.config import AuditExportContentStoreSettings, LandscapeExportSettings


def _store_settings(root: Path) -> AuditExportContentStoreSettings:
    return AuditExportContentStoreSettings(
        content_store_id="audit-store-v1",
        namespace="audit/export",
        root=root,
        policy_version="v1",
        retention_days=30,
        durability="fsync",
        orphan_grace_period_seconds=3600,
    )


def _export_settings(root: Path) -> LandscapeExportSettings:
    return LandscapeExportSettings(
        enabled=True,
        sink="audit",
        total_record_limit=100,
        total_byte_limit=10_000,
        chunk_limit=10,
        per_chunk_record_limit=10,
        per_chunk_byte_limit=1_000,
        spool_root=Path(".elspeth/audit-export-spool/test"),
        spool_cleanup_age_seconds=3600,
        spool_cleanup_byte_budget=10_000,
        spool_cleanup_count_budget=10,
        content_store=_store_settings(root),
    )


def test_filesystem_store_puts_and_opens_only_registered_exact_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = FilesystemAuditExportContentStore(_store_settings(Path(".elspeth/audit-export-content-store/test")))
    content = b'{"record_type":"run"}\n'
    digest = sha256(content).hexdigest()
    content_ref = store.put_immutable(content, candidate_id="candidate-1", object_kind="data_chunk")
    registration = RegisteredAuditExportContent(
        snapshot_id="a" * 64,
        content_store_id=store.content_store_id,
        namespace=store.namespace,
        descriptor=AuditExportContentDescriptor(
            content_ref=content_ref,
            content_hash=digest,
            size_bytes=len(content),
            object_kind="data_chunk",
        ),
    )

    assert store.open_registered(registration).read() == content
    assert store.put_immutable(content, candidate_id="candidate-2", object_kind="data_chunk") == content_ref

    wrong_store = RegisteredAuditExportContent(
        snapshot_id="a" * 64,
        content_store_id="other-store",
        namespace=store.namespace,
        descriptor=registration.descriptor,
    )
    with pytest.raises(LookupError):
        store.open_registered(wrong_store)


def test_registered_read_refuses_symlink_substitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path(".elspeth/audit-export-content-store/test")
    store = FilesystemAuditExportContentStore(_store_settings(root))
    content = b'{"record_type":"run"}\n'
    digest = sha256(content).hexdigest()
    content_ref = store.put_immutable(content, candidate_id="candidate-1", object_kind="data_chunk")
    descriptor = AuditExportContentDescriptor(content_ref, digest, len(content), "data_chunk")
    registration = RegisteredAuditExportContent("a" * 64, store.content_store_id, store.namespace, descriptor)
    object_path = root / "audit" / "export" / "objects" / digest[:2] / digest
    attacker = tmp_path / "attacker"
    attacker.write_bytes(content)
    object_path.unlink()
    object_path.symlink_to(attacker)

    with pytest.raises(OSError):
        store.open_registered(registration).read()


def test_production_factory_registers_the_explicit_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = _export_settings(Path(".elspeth/audit-export-content-store/test"))

    store, resolver = create_audit_export_content_store(settings)

    assert resolver.resolve("audit-store-v1") is store
    assert store.is_durable()


def test_candidate_gc_reads_the_complete_orphan_marker_beyond_64_kib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection must parse the complete valid marker, not a 64 KiB prefix (elspeth-4ae79708c2)."""
    monkeypatch.chdir(tmp_path)
    root = Path(".elspeth/audit-export-content-store/test")
    settings = _store_settings(root).model_copy(update={"orphan_grace_period_seconds": 1})
    store = FilesystemAuditExportContentStore(settings)
    content = b'{"record_type":"run"}\n'
    digest = sha256(content).hexdigest()
    descriptor = AuditExportContentDescriptor(
        store.put_immutable(content, candidate_id="candidate-large", object_kind="data_chunk"),
        digest,
        len(content),
        "data_chunk",
    )
    filler = tuple(AuditExportContentDescriptor(f"sha256:{index:064x}", f"{index:064x}", 1, "data_chunk") for index in range(1, 1201))
    store.mark_candidate_orphans("candidate-large", (descriptor, *filler))
    marker_path = root / "audit" / "export" / "candidates" / "candidate-large" / "orphan.json"
    assert marker_path.stat().st_size > 64 * 1024
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["marked_at"] = "2026-01-01T00:00:00.000000Z"
    marker_path.write_text(json.dumps(marker, separators=(",", ":"), sort_keys=True), encoding="utf-8")

    request = AuditExportOrphanCollectionRequest(
        candidate_id="candidate-large",
        namespace=store.namespace,
        descriptors=(descriptor,),
        marked_at=datetime.now(UTC) - timedelta(seconds=2),
        grace_period_seconds=1,
        fresh_winner_reference_check=lambda _ref: False,
    )

    assert store.garbage_collect_candidate(request) is True
    assert not (root / "audit" / "export" / "objects" / digest[:2] / digest).exists()


def test_candidate_gc_refuses_marker_beyond_the_explicit_safe_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker larger than any legitimate writer output must not be parsed
    or collected, and must not crash collection (elspeth-4ae79708c2)."""
    import elspeth.core.audit_export_content_store as content_store_module

    monkeypatch.chdir(tmp_path)
    root = Path(".elspeth/audit-export-content-store/test")
    settings = _store_settings(root).model_copy(update={"orphan_grace_period_seconds": 1})
    store = FilesystemAuditExportContentStore(settings)
    content = b'{"record_type":"run"}\n'
    digest = sha256(content).hexdigest()
    descriptor = AuditExportContentDescriptor(
        store.put_immutable(content, candidate_id="candidate-huge", object_kind="data_chunk"),
        digest,
        len(content),
        "data_chunk",
    )
    store.mark_candidate_orphans("candidate-huge", (descriptor,))
    marker_path = root / "audit" / "export" / "candidates" / "candidate-huge" / "orphan.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["marked_at"] = "2026-01-01T00:00:00.000000Z"
    marker_path.write_text(json.dumps(marker, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(content_store_module, "MAX_AUDIT_EXPORT_ORPHAN_MARKER_BYTES", marker_path.stat().st_size - 1)

    request = AuditExportOrphanCollectionRequest(
        candidate_id="candidate-huge",
        namespace=store.namespace,
        descriptors=(descriptor,),
        marked_at=datetime.now(UTC) - timedelta(seconds=2),
        grace_period_seconds=1,
        fresh_winner_reference_check=lambda _ref: False,
    )

    assert store.garbage_collect_candidate(request) is False
    assert (root / "audit" / "export" / "objects" / digest[:2] / digest).exists()


def test_orphan_marker_writer_enforces_the_reader_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer must never produce a marker the collector would refuse (elspeth-4ae79708c2)."""
    import elspeth.core.audit_export_content_store as content_store_module

    monkeypatch.chdir(tmp_path)
    store = FilesystemAuditExportContentStore(_store_settings(Path(".elspeth/audit-export-content-store/test")))
    content = b'{"record_type":"run"}\n'
    digest = sha256(content).hexdigest()
    descriptor = AuditExportContentDescriptor(
        store.put_immutable(content, candidate_id="candidate-bound", object_kind="data_chunk"),
        digest,
        len(content),
        "data_chunk",
    )
    monkeypatch.setattr(content_store_module, "MAX_AUDIT_EXPORT_ORPHAN_MARKER_BYTES", 16)

    with pytest.raises(ValueError, match="orphan marker"):
        store.mark_candidate_orphans("candidate-bound", (descriptor,))


def test_candidate_gc_requires_fresh_unreferenced_proof_and_deletes_only_owned_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path(".elspeth/audit-export-content-store/test")
    settings = _store_settings(root).model_copy(update={"orphan_grace_period_seconds": 1})
    store = FilesystemAuditExportContentStore(settings)
    content = b'{"record_type":"run"}\n'
    digest = sha256(content).hexdigest()
    descriptor = AuditExportContentDescriptor(
        store.put_immutable(content, candidate_id="candidate-1", object_kind="data_chunk"),
        digest,
        len(content),
        "data_chunk",
    )
    store.mark_candidate_orphans("candidate-1", (descriptor,))
    marker_path = root / "audit" / "export" / "candidates" / "candidate-1" / "orphan.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["marked_at"] = "2026-01-01T00:00:00.000000Z"
    marker_path.write_text(json.dumps(marker, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    marked_at = datetime.now(UTC) - timedelta(seconds=2)

    referenced = AuditExportOrphanCollectionRequest(
        candidate_id="candidate-1",
        namespace=store.namespace,
        descriptors=(descriptor,),
        marked_at=marked_at,
        grace_period_seconds=1,
        fresh_winner_reference_check=lambda _ref: True,
    )
    assert store.garbage_collect_candidate(referenced) is False

    unreferenced = AuditExportOrphanCollectionRequest(
        candidate_id="candidate-1",
        namespace=store.namespace,
        descriptors=(descriptor,),
        marked_at=marked_at,
        grace_period_seconds=1,
        fresh_winner_reference_check=lambda _ref: False,
    )
    assert store.garbage_collect_candidate(unreferenced) is True
    object_path = root / "audit" / "export" / "objects" / digest[:2] / digest
    assert not object_path.exists()
