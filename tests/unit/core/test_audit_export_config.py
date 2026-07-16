"""Audit-export configuration rejects ambiguous resource and identity policy."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from elspeth.contracts.audit_export import (
    AUDIT_EXPORT_MAX_CHUNK_BYTES,
    AUDIT_EXPORT_MAX_CHUNK_RECORDS,
    AUDIT_EXPORT_MAX_CHUNKS,
    AUDIT_EXPORT_MAX_TOTAL_BYTES,
    AUDIT_EXPORT_MAX_TOTAL_RECORDS,
)
from elspeth.core.config import LandscapeExportSettings


def _enabled_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "enabled": True,
        "sink": "archive",
        "format": "json",
        "signing_mode": "unsigned",
        "signer_key_id": "UNSIGNED",
        "signing_secret_ref": None,
        "signer_rotation_policy": "multi_version",
        "total_record_limit": 10_000,
        "total_byte_limit": 10_000_000,
        "chunk_limit": 100,
        "per_chunk_record_limit": 1_000,
        "per_chunk_byte_limit": 1_000_000,
        "spool_root": ".elspeth/audit-export-spool",
        "spool_cleanup_age_seconds": 3600,
        "spool_cleanup_byte_budget": 10_000_000,
        "spool_cleanup_count_budget": 100,
        "content_store": {
            "content_store_id": "archive-primary-v1",
            "namespace": "audit-export",
            "root": ".elspeth/audit-export-content-store/primary",
            "policy_version": "audit-store-policy-v1",
            "retention_days": 365,
            "durability": "fsync",
            "orphan_grace_period_seconds": 7200,
            "reference_safe_gc": True,
            "cleanup_scope": "candidate_unreferenced",
        },
    }
    config.update(overrides)
    return config


def test_enabled_export_requires_complete_explicit_bounded_resource_policy() -> None:
    settings = LandscapeExportSettings(**_enabled_config())
    assert settings.signing_mode == "unsigned"
    assert settings.signer_key_id == "UNSIGNED"
    assert settings.spool_root == Path(".elspeth/audit-export-spool")
    assert settings.content_store is not None
    assert settings.content_store.content_store_id == "archive-primary-v1"
    assert settings.content_store.root == Path(".elspeth/audit-export-content-store/primary")

    for field in (
        "total_record_limit",
        "total_byte_limit",
        "chunk_limit",
        "per_chunk_record_limit",
        "per_chunk_byte_limit",
        "spool_root",
        "spool_cleanup_age_seconds",
        "spool_cleanup_byte_budget",
        "spool_cleanup_count_budget",
        "content_store",
    ):
        config = _enabled_config()
        del config[field]
        with pytest.raises(ValidationError, match=field):
            LandscapeExportSettings(**config)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"signing_mode": "unsigned", "signer_key_id": "key-v1"}, "UNSIGNED"),
        ({"signing_mode": "unsigned", "signing_secret_ref": "ELSPETH_SIGNING_KEY"}, "forbids"),
        ({"signing_mode": "hmac_sha256", "signer_key_id": "UNSIGNED", "signing_secret_ref": "ELSPETH_SIGNING_KEY"}, "reserved"),
        ({"signing_mode": "hmac_sha256", "signer_key_id": "key-v1", "signing_secret_ref": None}, "secret"),
        (
            {
                "signing_mode": "hmac_sha256",
                "signer_key_id": "https://user:pass@example.test/key",
                "signing_secret_ref": "ELSPETH_SIGNING_KEY",
            },
            "credential-free",
        ),
    ],
)
def test_signing_identity_and_secret_reference_are_mode_exact(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        LandscapeExportSettings(**_enabled_config(**updates))


def test_hmac_signing_accepts_credential_free_versioned_key_identity() -> None:
    settings = LandscapeExportSettings(
        **_enabled_config(
            signing_mode="hmac_sha256",
            signer_key_id="audit-signer-2026-q3",
            signing_secret_ref="ELSPETH_SIGNING_KEY",
        )
    )
    public = settings.public_snapshot_config()
    assert public["signer_key_id"] == "audit-signer-2026-q3"
    assert "signing_secret_ref" not in public
    assert "total_record_limit" not in public
    assert "content_store" not in public


def test_signer_rotation_policy_is_explicit_and_single_export_refuses_new_identity() -> None:
    multi = LandscapeExportSettings(**_enabled_config(signer_rotation_policy="multi_version"))
    multi.assert_signer_rotation_allowed(existing_signer_key_id="old-key")

    single = LandscapeExportSettings(**_enabled_config(signer_rotation_policy="single_export"))
    with pytest.raises(ValueError, match="single_export"):
        single.assert_signer_rotation_allowed(existing_signer_key_id="old-key")


@pytest.mark.parametrize(
    ("field", "hard_max"),
    [
        ("total_record_limit", AUDIT_EXPORT_MAX_TOTAL_RECORDS),
        ("total_byte_limit", AUDIT_EXPORT_MAX_TOTAL_BYTES),
        ("chunk_limit", AUDIT_EXPORT_MAX_CHUNKS),
        ("per_chunk_record_limit", AUDIT_EXPORT_MAX_CHUNK_RECORDS),
        ("per_chunk_byte_limit", AUDIT_EXPORT_MAX_CHUNK_BYTES),
    ],
)
def test_each_acceptance_or_chunk_limit_has_a_code_owned_hard_maximum(field: str, hard_max: int) -> None:
    with pytest.raises(ValidationError, match="less than or equal"):
        LandscapeExportSettings(**_enabled_config(**{field: hard_max + 1}))
    with pytest.raises(ValidationError):
        LandscapeExportSettings(**_enabled_config(**{field: 0}))


@pytest.mark.parametrize(
    "updates",
    [
        {"per_chunk_record_limit": 10_001},
        {"per_chunk_byte_limit": 10_000_001},
        {"total_record_limit": 100_001, "chunk_limit": 100, "per_chunk_record_limit": 1_000},
        {"total_byte_limit": 100_000_001, "chunk_limit": 100, "per_chunk_byte_limit": 1_000_000},
        {"total_record_limit": 100, "total_byte_limit": 99},
    ],
)
def test_internally_inconsistent_limits_are_rejected(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match=r"limit|capacity|bytes"):
        LandscapeExportSettings(**_enabled_config(**updates))


@pytest.mark.parametrize("spool_root", ["/tmp/elspeth-export", "../audit-export", "state/audit-export-spool"])
def test_spool_root_must_be_inside_code_owned_private_root(spool_root: str) -> None:
    with pytest.raises(ValidationError, match="spool_root"):
        LandscapeExportSettings(**_enabled_config(spool_root=spool_root))


def test_existing_world_accessible_spool_root_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    spool = tmp_path / ".elspeth" / "audit-export-spool"
    spool.mkdir(parents=True)
    spool.chmod(0o755)
    with pytest.raises(ValidationError, match="private"):
        LandscapeExportSettings(**_enabled_config())


@pytest.mark.parametrize(
    "root",
    ["/tmp/audit-export-content", "../audit-export-content", ".elspeth/audit-export-spool/content"],
)
def test_content_store_root_is_explicit_and_inside_code_owned_root(root: str) -> None:
    content_store = {**_enabled_config()["content_store"], "root": root}  # type: ignore[dict-item]
    with pytest.raises(ValidationError, match="root"):
        LandscapeExportSettings(**_enabled_config(content_store=content_store))


@pytest.mark.parametrize(
    "content_store",
    [
        {**_enabled_config()["content_store"], "durability": "best_effort"},  # type: ignore[dict-item]
        {**_enabled_config()["content_store"], "reference_safe_gc": False},  # type: ignore[dict-item]
        {**_enabled_config()["content_store"], "cleanup_scope": "namespace_prefix"},  # type: ignore[dict-item]
    ],
)
def test_content_store_requires_durable_reference_safe_candidate_cleanup(content_store: object) -> None:
    with pytest.raises(ValidationError):
        LandscapeExportSettings(**_enabled_config(content_store=content_store))
