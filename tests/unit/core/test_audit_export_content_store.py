from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from elspeth.contracts.audit_export import (
    AuditExportContentDescriptor,
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


def test_mark_candidate_orphans_writes_a_durable_marker_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker is the sole durable record of loser-candidate content refs."""
    monkeypatch.chdir(tmp_path)
    root = Path(".elspeth/audit-export-content-store/test")
    store = FilesystemAuditExportContentStore(_store_settings(root))
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
    assert marker["schema"] == "elspeth.audit-export-orphan-candidate.v1"
    assert marker["candidate_id"] == "candidate-1"
    assert marker["content_refs"] == [descriptor.content_ref]

    first_bytes = marker_path.read_bytes()
    store.mark_candidate_orphans("candidate-1", (descriptor,))
    assert marker_path.read_bytes() == first_bytes


def test_orphan_marker_writer_enforces_the_size_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer must never produce a marker beyond the code-owned bound (elspeth-4ae79708c2)."""
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
