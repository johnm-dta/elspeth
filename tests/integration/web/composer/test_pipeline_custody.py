"""Integration contracts for custody-safe canonical pipeline proposals."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from elspeth.contracts.enums import CreationModality
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import stable_hash
from elspeth.web.blobs.service import content_hash
from elspeth.web.composer.pipeline_custody import (
    inline_custody_audit_projection,
    prepare_pipeline_custody,
)
from elspeth.web.composer.tools.blobs import _PreparedBlobCreate


def _prepared(tmp_path: Path, *, content: bytes = b"private-inline-value\n42\n") -> _PreparedBlobCreate:
    session_id = str(uuid4())
    blob_id = str(uuid4())
    return _PreparedBlobCreate(
        blob_id=blob_id,
        filename="candidate.csv",
        mime_type="text/csv",
        content_bytes=content,
        content_hash=content_hash(content),
        storage_path=tmp_path / "blobs" / session_id / f"{blob_id}_candidate.csv",
        description="reviewed candidate",
        creation_modality=CreationModality.LLM_GENERATED,
        created_from_message_id=str(uuid4()),
        creating_model_identifier="composer-model",
        creating_model_version="composer-version",
        creating_provider="provider",
        creating_composer_skill_hash="a" * 64,
        creating_arguments_hash="b" * 64,
    )


def _arguments(content: str = "private-inline-value\n42\n") -> dict[str, object]:
    return {
        "source": {
            "plugin": "csv",
            "options": {"schema": {"mode": "observed"}},
            "on_success": "main",
            "on_validation_failure": "discard",
            "inline_blob": {
                "filename": "candidate.csv",
                "mime_type": "text/csv",
                "content": content,
                "description": "reviewed candidate",
            },
        },
        "nodes": [],
        "edges": [],
        "outputs": [],
        "metadata": {"name": "custody proposal"},
    }


def test_prepare_pipeline_custody_is_pure_and_hashes_only_safe_arguments(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    arguments = _arguments()
    session_id = prepared.storage_path.parent.name

    custody = prepare_pipeline_custody(arguments, prepared, session_id=session_id)

    source = custody.arguments["source"]
    assert isinstance(source, dict)
    assert "inline_blob" not in source
    assert source["blob_id"] == str(custody.blob_id)
    assert custody.request.creating_arguments_hash == stable_hash(custody.arguments)
    assert custody.request.content == prepared.content_bytes
    assert not prepared.storage_path.parent.exists()
    assert "private-inline-value" not in repr(prepared)
    assert "private-inline-value" not in repr(custody.request)


def test_audit_projection_recursively_removes_inline_content_from_malformed_and_named_shapes() -> None:
    secret = "redacted inline test value"
    arguments = _arguments(secret)
    arguments["sources"] = {
        "named": {
            "inline_blob": {
                "content": {"malformed": [secret]},
                "filename": "named.csv",
                "mime_type": "text/csv",
            }
        }
    }

    projected = inline_custody_audit_projection(arguments)

    assert secret not in repr(projected)
    assert projected["source"]["inline_blob"]["content"] == "[redacted inline content held for custody]"
    assert projected["sources"]["named"]["inline_blob"]["content"] == "[redacted inline content held for custody]"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda arguments: arguments["source"]["options"].update({"blob_ref": str(uuid4())}),
        lambda arguments: arguments.update({"sources": {"named": {"inline_blob": {"content": "second secret"}}}}),
    ],
)
def test_prepare_pipeline_custody_rejects_manual_or_residual_blob_authority(tmp_path: Path, mutate) -> None:
    prepared = _prepared(tmp_path)
    arguments = _arguments()
    mutate(arguments)

    with pytest.raises(AuditIntegrityError):
        prepare_pipeline_custody(arguments, prepared, session_id=prepared.storage_path.parent.name)

    assert not prepared.storage_path.parent.exists()


@pytest.mark.parametrize(
    ("arguments_patch", "prepared_patch"),
    [
        ({"content": "different private content"}, {}),
        ({"mime_type": "text/plain"}, {}),
        ({"filename": "different.csv"}, {}),
        ({"description": "different description"}, {}),
        ({}, {"content_hash": "f" * 64}),
    ],
)
def test_prepare_pipeline_custody_rejects_candidate_argument_seam_mismatch_without_values(
    tmp_path: Path,
    arguments_patch: dict[str, str],
    prepared_patch: dict[str, str],
) -> None:
    prepared = replace(_prepared(tmp_path), **prepared_patch)
    arguments = _arguments()
    arguments["source"]["inline_blob"].update(arguments_patch)

    with pytest.raises(AuditIntegrityError) as exc_info:
        prepare_pipeline_custody(arguments, prepared, session_id=prepared.storage_path.parent.name)

    assert "different private content" not in str(exc_info.value)
    assert "different description" not in str(exc_info.value)
    assert not prepared.storage_path.parent.exists()
