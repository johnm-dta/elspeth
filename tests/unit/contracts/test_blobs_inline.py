"""Pin widened blob_ref L0 contract shapes."""

from __future__ import annotations

from typing import get_args
from uuid import UUID

import pytest

from elspeth.contracts.blobs_inline import (
    ALLOWED_BLOB_REF_MODES,
    ALLOWED_CONTENT_ENCODINGS,
    BlobContentResolutionError,
    BlobInlineRef,
    BlobRefMode,
    ContentEncoding,
    ResolvedBlobContent,
    WidenedBlobRefShape,
    is_widened_blob_ref,
)

VALID_BLOB_ID = "5b7a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3a4b"
VALID_HASH = "a" * 64


class TestWidenedBlobRefRecognition:
    def test_recognises_inline_content_marker(self) -> None:
        shape = is_widened_blob_ref(
            {
                "blob_ref": VALID_BLOB_ID,
                "mode": "inline_content",
                "sha256": VALID_HASH,
            }
        )

        assert shape == WidenedBlobRefShape(
            blob_id=UUID(VALID_BLOB_ID),
            mode="inline_content",
            sha256=VALID_HASH,
            path=None,
            encoding="utf-8",
        )

    def test_recognises_inline_content_explicit_encoding(self) -> None:
        shape = is_widened_blob_ref(
            {
                "blob_ref": VALID_BLOB_ID,
                "mode": "inline_content",
                "sha256": VALID_HASH,
                "encoding": "utf-16",
            }
        )

        assert shape is not None
        assert shape.encoding == "utf-16"

    def test_recognises_bind_source_marker(self) -> None:
        shape = is_widened_blob_ref(
            {
                "blob_ref": VALID_BLOB_ID,
                "mode": "bind_source",
                "path": "/data/file.csv",
            }
        )

        assert shape == WidenedBlobRefShape(
            blob_id=UUID(VALID_BLOB_ID),
            mode="bind_source",
            sha256=None,
            path="/data/file.csv",
            encoding="utf-8",
        )

    @pytest.mark.parametrize(
        "value",
        [
            {"blob_ref": VALID_BLOB_ID},
            {"blob_ref": VALID_BLOB_ID, "mode": "inline_content"},
            {"blob_ref": VALID_BLOB_ID, "mode": "inline_content", "sha256": VALID_HASH, "path": "/tmp/x"},
            {"blob_ref": VALID_BLOB_ID, "mode": "inline_content", "sha256": "not-a-hash"},
            {"blob_ref": VALID_BLOB_ID, "mode": "inline_content", "sha256": VALID_HASH, "encoding": "cp1252"},
            {"blob_ref": VALID_BLOB_ID, "mode": "bind_source", "sha256": VALID_HASH},
            {"blob_ref": VALID_BLOB_ID, "mode": "bind_source", "encoding": "utf-8"},
            {"blob_ref": "not-a-uuid", "mode": "inline_content", "sha256": VALID_HASH},
            {"blob_ref": VALID_BLOB_ID, "mode": "unknown", "sha256": VALID_HASH},
            {"blob_ref": VALID_BLOB_ID, "mode": "inline_content", "sha256": VALID_HASH, "extra": "x"},
            {"secret_ref": "OPENROUTER_KEY"},
            {"not_a_ref": "x"},
            ["not", "a", "mapping"],
            "not a mapping",
            None,
        ],
    )
    def test_rejects_non_markers_and_malformed_markers(self, value: object) -> None:
        assert is_widened_blob_ref(value) is None


class TestValueObjectValidation:
    def test_blob_inline_ref_accepts_canonical_field_paths(self) -> None:
        for path in (
            "source.options.system_prompt",
            "node:classify.options.system_prompt",
            "output:writeback.options.body_template",
            "node:classify-1.options.prompt.template",
        ):
            ref = BlobInlineRef(
                field_path=path,
                blob_id=UUID(VALID_BLOB_ID),
                sha256=VALID_HASH,
                encoding="utf-8",
            )
            assert ref.field_path == path

    @pytest.mark.parametrize(
        "path",
        [
            "",
            "transforms[2].options.system_prompt",
            "node:classify",
            "node:classify.system_prompt",
            "output:writeback.options",
            "x.options.system_prompt",
        ],
    )
    def test_blob_inline_ref_rejects_non_canonical_field_paths(self, path: str) -> None:
        with pytest.raises(ValueError, match="field_path"):
            BlobInlineRef(
                field_path=path,
                blob_id=UUID(VALID_BLOB_ID),
                sha256=VALID_HASH,
                encoding="utf-8",
            )

    def test_blob_inline_ref_rejects_invalid_hash(self) -> None:
        with pytest.raises(ValueError, match="sha256"):
            BlobInlineRef(
                field_path="source.options.system_prompt",
                blob_id=UUID(VALID_BLOB_ID),
                sha256="A" * 64,
                encoding="utf-8",
            )

    def test_blob_inline_ref_rejects_invalid_encoding(self) -> None:
        with pytest.raises(ValueError, match="encoding"):
            BlobInlineRef(
                field_path="source.options.system_prompt",
                blob_id=UUID(VALID_BLOB_ID),
                sha256=VALID_HASH,
                encoding="cp1252",  # type: ignore[arg-type]
            )

    def test_resolved_blob_content_validates_audit_shape(self) -> None:
        resolved = ResolvedBlobContent(
            field_path="node:classify.options.system_prompt",
            blob_id=UUID(VALID_BLOB_ID),
            content_hash=VALID_HASH,
            byte_length=42,
            mime_type="text/plain",
            encoding="utf-8",
        )

        assert resolved.byte_length == 42
        assert resolved.mime_type == "text/plain"

    def test_resolved_blob_content_rejects_negative_byte_length(self) -> None:
        with pytest.raises(ValueError, match="byte_length"):
            ResolvedBlobContent(
                field_path="source.options.system_prompt",
                blob_id=UUID(VALID_BLOB_ID),
                content_hash=VALID_HASH,
                byte_length=-1,
                mime_type="text/plain",
                encoding="utf-8",
            )

    def test_resolved_blob_content_rejects_invalid_mime_type(self) -> None:
        with pytest.raises(ValueError, match="mime_type"):
            ResolvedBlobContent(
                field_path="source.options.system_prompt",
                blob_id=UUID(VALID_BLOB_ID),
                content_hash=VALID_HASH,
                byte_length=1,
                mime_type="application/octet-stream",  # type: ignore[arg-type]
                encoding="utf-8",
            )


class TestClosedSets:
    def test_content_encoding_runtime_set_matches_literal(self) -> None:
        assert frozenset(get_args(ContentEncoding)) == ALLOWED_CONTENT_ENCODINGS

    def test_blob_ref_mode_runtime_set_matches_literal(self) -> None:
        assert frozenset(get_args(BlobRefMode)) == ALLOWED_BLOB_REF_MODES


class TestBlobContentResolutionError:
    def test_aggregates_all_recoverable_cases(self) -> None:
        err = BlobContentResolutionError(
            missing=["node:c.options.system_prompt"],
            oversized=[("node:c.options.body", 100_000, 64_000)],
            undecodable=[("source.options.x", "utf-8")],
            not_ready=[("source.options.y", "pending")],
            cross_session=["output:s.options.template"],
            malformed=[("node:c.options.z", "missing mode")],
        )

        assert err.missing == ("node:c.options.system_prompt",)
        assert err.oversized == (("node:c.options.body", 100_000, 64_000),)
        assert err.undecodable == (("source.options.x", "utf-8"),)
        assert err.not_ready == (("source.options.y", "pending"),)
        assert err.cross_session == ("output:s.options.template",)
        assert err.malformed == (("node:c.options.z", "missing mode"),)
        assert "missing=1" in str(err)
        assert "oversized=1" in str(err)
