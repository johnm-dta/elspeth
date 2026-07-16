"""Immutable audit-export snapshot registry schema contract."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import inspect, update
from sqlalchemy.exc import IntegrityError

from elspeth.core.landscape.database import _REQUIRED_CHECK_CONSTRAINTS, _REQUIRED_INDEXES, _REQUIRED_TRIGGERS, LandscapeDB
from elspeth.core.landscape.schema import audit_export_snapshot_chunks_table, audit_export_snapshots_table, metadata, runs_table

SNAPSHOT_CHECKS = {
    "ck_audit_export_snapshots_manifest_hash_hex",
    "ck_audit_export_snapshots_snapshot_hash_hex",
    "ck_audit_export_snapshots_snapshot_seal_hash_hex",
    "ck_audit_export_snapshots_last_chunk_seal_hash_hex",
    "ck_audit_export_snapshots_final_hash_hex",
    "ck_audit_export_snapshots_signed_manifest_hash_hex",
    "ck_audit_export_snapshots_signed_manifest_ref",
    "ck_audit_export_snapshots_signed_manifest_size",
    "ck_audit_export_snapshots_manifest_schema",
    "ck_audit_export_snapshots_derivation_version",
    "ck_audit_export_snapshots_signing_tuple",
}
COMPLETED_AT = datetime(2026, 7, 16, 1, 2, 3, 456789, tzinfo=UTC)


def _run_values() -> dict[str, object]:
    return {
        "run_id": "run-export",
        "started_at": COMPLETED_AT,
        "completed_at": COMPLETED_AT,
        "config_hash": "0" * 64,
        "settings_json": "{}",
        "canonical_version": "v1",
        "status": "completed",
        "openrouter_catalog_sha256": "1" * 64,
        "openrouter_catalog_source": "bundled",
    }


def _chunk_values() -> dict[str, object]:
    return {
        "snapshot_id": "2" * 64,
        "ordinal": 0,
        "content_ref": f"sha256:{'3' * 64}",
        "content_hash": "3" * 64,
        "size_bytes": 10,
        "record_count": 1,
        "predecessor_seal_hash": None,
        "cumulative_records": 1,
        "cumulative_bytes": 10,
        "chunk_seal_hash": "4" * 64,
    }


def _snapshot_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "snapshot_id": "2" * 64,
        "source_run_id": "run-export",
        "source_status": "completed",
        "source_completed_at": COMPLETED_AT,
        "exported_at": COMPLETED_AT,
        "registry_key_hash": "5" * 64,
        "exporter_version": "v2",
        "serialization_version": "audit-export-v2",
        "export_format": "json",
        "signing_mode": "unsigned",
        "signer_key_id": "UNSIGNED",
        "derivation_version": "audit-export-derivation-v1",
        "public_export_config_hash": "6" * 64,
        "chunking_algorithm_version": "chunk-v1",
        "per_chunk_record_limit": 100,
        "per_chunk_byte_limit": 1024,
        "record_count": 1,
        "total_bytes": 10,
        "chunk_count": 1,
        "terminal_chunk_ordinal": 0,
        "content_store_id": "durable-store",
        "manifest_hash": "7" * 64,
        "last_chunk_seal_hash": "4" * 64,
        "snapshot_hash": "8" * 64,
        "snapshot_seal_hash": "9" * 64,
        "signature_hex": None,
        "record_chain_algorithm": "sha256_concat_record_sha256_v1",
        "final_hash": "a" * 64,
        "signed_manifest_schema": "elspeth.audit-export-manifest.v2",
        "signed_manifest_hash": "b" * 64,
        "signed_manifest_ref": f"sha256:{'b' * 64}",
        "signed_manifest_size_bytes": 128,
    }
    values.update(overrides)
    return values


def _insert_snapshot(db: LandscapeDB, **overrides: object) -> None:
    with db.engine.begin() as connection:
        connection.execute(runs_table.insert().values(**_run_values()))
        connection.execute(audit_export_snapshot_chunks_table.insert().values(**_chunk_values()))
        connection.execute(audit_export_snapshots_table.insert().values(**_snapshot_values(**overrides)))


def test_snapshot_registry_has_exact_structural_fields() -> None:
    table = metadata.tables["audit_export_snapshots"]
    assert set(table.c) == {
        table.c[name]
        for name in (
            "snapshot_id",
            "source_run_id",
            "source_status",
            "source_completed_at",
            "exported_at",
            "registry_key_hash",
            "exporter_version",
            "serialization_version",
            "export_format",
            "signing_mode",
            "signer_key_id",
            "derivation_version",
            "public_export_config_hash",
            "chunking_algorithm_version",
            "per_chunk_record_limit",
            "per_chunk_byte_limit",
            "record_count",
            "total_bytes",
            "chunk_count",
            "terminal_chunk_ordinal",
            "content_store_id",
            "manifest_hash",
            "last_chunk_seal_hash",
            "snapshot_hash",
            "snapshot_seal_hash",
            "signature_hex",
            "record_chain_algorithm",
            "final_hash",
            "signed_manifest_schema",
            "signed_manifest_hash",
            "signed_manifest_ref",
            "signed_manifest_size_bytes",
        )
    }


def test_all_eleven_snapshot_checks_are_required_and_reflected() -> None:
    assert {("audit_export_snapshots", name) for name in SNAPSHOT_CHECKS} <= set(_REQUIRED_CHECK_CONSTRAINTS)
    db = LandscapeDB.in_memory()
    try:
        reflected = {item["name"] for item in inspect(db.engine).get_check_constraints("audit_export_snapshots")}
    finally:
        db.close()
    assert reflected >= SNAPSHOT_CHECKS


def test_export_witness_and_registry_indexes_have_exact_order() -> None:
    runs_index = next(index for index in metadata.tables["runs"].indexes if index.name == "uq_runs_export_witness")
    assert runs_index.unique is True
    assert [column.name for column in runs_index.columns] == ["run_id", "status", "completed_at"]
    registry_index = next(
        index for index in metadata.tables["audit_export_snapshots"].indexes if index.name == "uq_audit_export_snapshots_registry_key"
    )
    assert [column.name for column in registry_index.columns] == [
        "source_run_id",
        "exporter_version",
        "serialization_version",
        "export_format",
        "signing_mode",
        "signer_key_id",
        "public_export_config_hash",
    ]
    assert ("runs", "uq_runs_export_witness") in _REQUIRED_INDEXES
    assert ("audit_export_snapshots", "uq_audit_export_snapshots_registry_key") in _REQUIRED_INDEXES


def test_snapshot_chunk_terminal_index_is_exact() -> None:
    index = next(
        index
        for index in metadata.tables["audit_export_snapshot_chunks"].indexes
        if index.name == "uq_audit_export_snapshot_chunks_terminal"
    )
    assert index.unique is True
    assert [column.name for column in index.columns] == [
        "snapshot_id",
        "ordinal",
        "chunk_seal_hash",
        "cumulative_records",
        "cumulative_bytes",
    ]


def test_valid_unsigned_snapshot_commits_and_physical_triggers_exist() -> None:
    db = LandscapeDB.in_memory()
    try:
        _insert_snapshot(db)
        with db.engine.connect() as connection:
            trigger_names = set(connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'trigger'").scalars())
        assert trigger_names >= set(_REQUIRED_TRIGGERS)
    finally:
        db.close()


@pytest.mark.parametrize(
    "field_name",
    ["manifest_hash", "snapshot_hash", "snapshot_seal_hash", "last_chunk_seal_hash", "final_hash", "signed_manifest_hash"],
)
@pytest.mark.parametrize("bad_hash", ["A" * 64, "g" * 64, "a" * 63, "a" * 65])
def test_registry_hash_checks_reject_every_non_lowercase_sha256(field_name: str, bad_hash: str) -> None:
    db = LandscapeDB.in_memory()
    overrides: dict[str, object] = {field_name: bad_hash}
    if field_name == "signed_manifest_hash":
        overrides["signed_manifest_ref"] = f"sha256:{bad_hash}"
    try:
        with pytest.raises(IntegrityError):
            _insert_snapshot(db, **overrides)
    finally:
        db.close()


@pytest.mark.parametrize("size", [0, -1, 65_537])
def test_registry_rejects_invalid_signed_manifest_size(size: int) -> None:
    db = LandscapeDB.in_memory()
    try:
        with pytest.raises(IntegrityError):
            _insert_snapshot(db, signed_manifest_size_bytes=size)
    finally:
        db.close()


def test_registry_rejects_mismatched_signed_manifest_reference() -> None:
    db = LandscapeDB.in_memory()
    try:
        with pytest.raises(IntegrityError):
            _insert_snapshot(db, signed_manifest_ref=f"sha256:{'c' * 64}")
    finally:
        db.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"signed_manifest_schema": "elspeth.audit-export-manifest.v1"},
        {"derivation_version": "audit-export-derivation-v2"},
        {"signing_mode": "unsigned", "signer_key_id": "key-v1"},
        {"signing_mode": "unsigned", "signature_hex": "c" * 64},
        {"signing_mode": "unsigned", "record_chain_algorithm": "sha256_concat_hmac_sha256_signatures_v1"},
        {"signing_mode": "hmac_sha256", "signer_key_id": "UNSIGNED", "signature_hex": "c" * 64},
        {"signing_mode": "hmac_sha256", "signer_key_id": "key-v1", "signature_hex": None},
        {
            "signing_mode": "hmac_sha256",
            "signer_key_id": "key-v1",
            "signature_hex": "c" * 64,
            "record_chain_algorithm": "sha256_concat_record_sha256_v1",
        },
    ],
)
def test_registry_accepts_only_closed_signing_and_version_tuples(overrides: dict[str, object]) -> None:
    db = LandscapeDB.in_memory()
    try:
        with pytest.raises(IntegrityError):
            _insert_snapshot(db, **overrides)
    finally:
        db.close()


@pytest.mark.parametrize("bad_signature", ["C" * 64, "g" * 64, "c" * 63, "c" * 65])
def test_hmac_signature_is_exact_lowercase_sha256(bad_signature: str) -> None:
    db = LandscapeDB.in_memory()
    try:
        with pytest.raises(IntegrityError):
            _insert_snapshot(
                db,
                signing_mode="hmac_sha256",
                signer_key_id="key-v1",
                signature_hex=bad_signature,
                record_chain_algorithm="sha256_concat_hmac_sha256_signatures_v1",
            )
    finally:
        db.close()


def test_sealed_snapshot_and_chunk_rows_are_immutable() -> None:
    db = LandscapeDB.in_memory()
    try:
        _insert_snapshot(db)
        with pytest.raises(IntegrityError), db.engine.begin() as connection:
            connection.execute(update(audit_export_snapshots_table).values(content_store_id="other"))
        with pytest.raises(IntegrityError), db.engine.begin() as connection:
            connection.execute(update(audit_export_snapshot_chunks_table).values(size_bytes=11))
        with pytest.raises(IntegrityError), db.engine.begin() as connection:
            connection.execute(audit_export_snapshot_chunks_table.delete())
        with pytest.raises(IntegrityError), db.engine.begin() as connection:
            connection.execute(audit_export_snapshots_table.delete())
    finally:
        db.close()
