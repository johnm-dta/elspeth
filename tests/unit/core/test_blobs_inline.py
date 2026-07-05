"""Inline blob content resolver tests."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import UTC, datetime
from uuid import UUID

import pytest

from elspeth.contracts.blobs import (
    BlobContentMissingError,
    BlobIntegrityError,
    BlobNotFoundError,
    BlobRecord,
    BlobStateError,
)
from elspeth.contracts.blobs_inline import BlobContentResolutionError, BlobInlineRef, BlobInlineValidationViolation, ResolvedBlobContent
from elspeth.contracts.enums import CreationModality
from elspeth.core.blobs_inline import (
    _discover_blob_content_refs,
    _fetch_blob_contents,
    _substitute_blob_content_refs,
    _validate_blob_content_refs,
    _validate_blob_content_refs_sync,
)

VALID_HASH = "a" * 64
BLOB1 = "5b7a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3a4b"
BLOB2 = "7c3a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3aaa"
SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_SESSION_ID = UUID("22222222-2222-4222-8222-222222222222")


class _AsyncCallRecorder:
    def __init__(self) -> None:
        self.return_value: object = None
        self.side_effect: BaseException | None = None
        self.call_args_list: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args: object, **kwargs: object) -> object:
        self.call_args_list.append((args, kwargs))
        if self.side_effect is not None:
            raise self.side_effect
        return self.return_value

    def assert_awaited_once_with(self, *args: object, **kwargs: object) -> None:
        assert self.call_args_list == [(args, kwargs)]

    def assert_not_called(self) -> None:
        assert self.call_args_list == []


class _BlobServiceDouble:
    def __init__(self) -> None:
        self.read_blob_content = _AsyncCallRecorder()
        self.get_blob = _AsyncCallRecorder()


def _marker(blob_id: str = BLOB1, sha256: str = VALID_HASH) -> dict[str, str]:
    return {"blob_ref": blob_id, "mode": "inline_content", "sha256": sha256}


def _ref(blob_id: str = BLOB1, field_path: str = "source.options.x") -> BlobInlineRef:
    return BlobInlineRef(field_path=field_path, blob_id=UUID(blob_id), sha256=VALID_HASH, encoding="utf-8")


def _ref_with_hash(content: bytes, field_path: str = "source.options.system_prompt", blob_id: str = BLOB1) -> BlobInlineRef:
    return BlobInlineRef(
        field_path=field_path,
        blob_id=UUID(blob_id),
        sha256=hashlib.sha256(content).hexdigest(),
        encoding="utf-8",
    )


def _blob_record(
    *,
    blob_id: str = BLOB1,
    session_id: UUID = SESSION_ID,
    status: str = "ready",
    content_hash: str | None = VALID_HASH,
    size_bytes: int = 12,
) -> BlobRecord:
    return BlobRecord(
        id=UUID(blob_id),
        session_id=session_id,
        filename="prompt.txt",
        mime_type="text/plain",
        size_bytes=size_bytes,
        content_hash=content_hash,
        storage_path="/tmp/prompt.txt",
        created_at=datetime(2026, 5, 24, tzinfo=UTC),
        created_by="user",
        source_description=None,
        status=status,  # type: ignore[arg-type]
        creation_modality=CreationModality.VERBATIM,
        created_from_message_id=None,
        creating_model_identifier=None,
        creating_model_version=None,
        creating_provider=None,
        creating_composer_skill_hash=None,
        creating_arguments_hash=None,
    )


class TestDiscoverBlobContentRefs:
    def test_discovers_source_option_refs(self) -> None:
        config = {"source": {"plugin": "csv", "options": {"system_prompt": _marker()}}}

        refs = _discover_blob_content_refs(config)

        assert len(refs) == 1
        assert refs[0].field_path == "source.options.system_prompt"
        assert refs[0].blob_id == UUID(BLOB1)
        assert refs[0].sha256 == VALID_HASH
        assert refs[0].encoding == "utf-8"

    def test_discovers_plural_source_option_refs_with_source_name(self) -> None:
        config = {
            "sources": {
                "orders": {"plugin": "csv", "options": {"system_prompt": _marker(BLOB1)}},
                "refunds": {"plugin": "json", "options": {"template": _marker(BLOB2)}},
            }
        }

        refs = _discover_blob_content_refs(config)

        assert {ref.field_path for ref in refs} == {
            "source:orders.options.system_prompt",
            "source:refunds.options.template",
        }

    def test_discovers_node_option_refs_across_node_collections(self) -> None:
        config = {
            "transforms": [
                {"name": "classify", "plugin": "llm", "options": {"system_prompt": _marker(BLOB1)}},
            ],
            "gates": [
                {"name": "policy-check", "plugin": "expression", "options": {"template": _marker(BLOB2)}},
            ],
            "aggregations": [
                {"name": "rollup", "plugin": "group_by", "options": {}},
            ],
            "coalesce": [
                {"name": "merge", "plugin": "first", "options": {"explain": _marker(BLOB1)}},
            ],
        }

        refs = _discover_blob_content_refs(config)

        assert {ref.field_path for ref in refs} == {
            "node:classify.options.system_prompt",
            "node:policy-check.options.template",
            "node:merge.options.explain",
        }

    def test_discovers_output_and_sink_refs_with_same_canonical_prefix(self) -> None:
        config = {
            "outputs": {
                "state-view": {"plugin": "json", "options": {"body_template": _marker(BLOB1)}},
            },
            "sinks": {
                "writeback": {"plugin": "csv", "options": {"footer_template": _marker(BLOB2)}},
            },
        }

        refs = _discover_blob_content_refs(config)

        assert {ref.field_path for ref in refs} == {
            "output:state-view.options.body_template",
            "output:writeback.options.footer_template",
        }

    def test_discovers_nested_refs_inside_options(self) -> None:
        config = {
            "source": {
                "plugin": "csv",
                "options": {
                    "auth": {
                        "system_prompt": _marker(),
                    },
                },
            },
        }

        refs = _discover_blob_content_refs(config)

        assert len(refs) == 1
        assert refs[0].field_path == "source.options.auth.system_prompt"

    def test_ignores_bind_source_refs_and_secret_refs(self) -> None:
        config = {
            "source": {
                "plugin": "csv",
                "options": {
                    "blob_ref": BLOB1,
                    "mode": "bind_source",
                    "path": "/tmp/input.csv",
                    "api_key": {"secret_ref": "OPENROUTER_KEY"},
                },
            }
        }

        assert _discover_blob_content_refs(config) == []

    def test_emits_one_ref_per_field_path_even_for_same_blob(self) -> None:
        config = {
            "transforms": [
                {"name": "a", "plugin": "llm", "options": {"system_prompt": _marker(BLOB1)}},
                {"name": "b", "plugin": "llm", "options": {"system_prompt": _marker(BLOB1)}},
            ]
        }

        refs = _discover_blob_content_refs(config)

        assert len(refs) == 2
        assert {ref.field_path for ref in refs} == {
            "node:a.options.system_prompt",
            "node:b.options.system_prompt",
        }

    def test_malformed_markers_raise_batched_error(self) -> None:
        config = {"source": {"plugin": "csv", "options": {"system_prompt": {"blob_ref": BLOB1}}}}

        with pytest.raises(BlobContentResolutionError) as exc_info:
            _discover_blob_content_refs(config)

        assert exc_info.value.malformed == (("source.options.system_prompt", "missing mode"),)

    @pytest.mark.parametrize(
        ("marker", "reason"),
        [
            ({"blob_ref": BLOB1, "mode": []}, "mode must be string"),
            ({"blob_ref": BLOB1, "mode": "inline_content", "sha256": VALID_HASH, "encoding": []}, "encoding must be string"),
        ],
    )
    def test_non_string_marker_fields_are_malformed(self, marker: dict[str, object], reason: str) -> None:
        config = {"source": {"plugin": "csv", "options": {"system_prompt": marker}}}

        with pytest.raises(BlobContentResolutionError) as exc_info:
            _discover_blob_content_refs(config)

        assert exc_info.value.malformed == (("source.options.system_prompt", reason),)

    def test_markers_inside_lists_are_malformed_because_paths_would_be_positional(self) -> None:
        config = {"source": {"plugin": "csv", "options": {"prompts": [_marker()]}}}

        with pytest.raises(BlobContentResolutionError) as exc_info:
            _discover_blob_content_refs(config)

        assert exc_info.value.malformed == (("source.options.prompts", "inline blob refs inside lists are not supported"),)


class TestFetchBlobContents:
    @pytest.mark.asyncio
    async def test_returns_bytes_by_ref(self) -> None:
        blob_service = _BlobServiceDouble()
        blob_service.read_blob_content.return_value = b"content"
        ref = _ref()

        fetched = await _fetch_blob_contents(blob_service, [ref])

        blob_service.read_blob_content.assert_awaited_once_with(UUID(BLOB1))
        assert fetched == {ref: b"content"}

    @pytest.mark.asyncio
    async def test_dedupes_reads_by_blob_id(self) -> None:
        blob_service = _BlobServiceDouble()
        blob_service.read_blob_content.return_value = b"content"
        refs = [
            _ref(BLOB1, "source.options.a"),
            _ref(BLOB1, "source.options.b"),
        ]

        fetched = await _fetch_blob_contents(blob_service, refs)

        blob_service.read_blob_content.assert_awaited_once_with(UUID(BLOB1))
        assert fetched == {refs[0]: b"content", refs[1]: b"content"}

    @pytest.mark.asyncio
    async def test_propagates_integrity_errors(self) -> None:
        blob_service = _BlobServiceDouble()
        blob_service.read_blob_content.side_effect = BlobIntegrityError(BLOB1, expected="a" * 64, actual="b" * 64)

        with pytest.raises(BlobIntegrityError):
            await _fetch_blob_contents(blob_service, [_ref()])

    @pytest.mark.asyncio
    async def test_propagates_content_missing_errors(self) -> None:
        blob_service = _BlobServiceDouble()
        blob_service.read_blob_content.side_effect = BlobContentMissingError(BLOB1, storage_path="/tmp/blob.csv")

        with pytest.raises(BlobContentMissingError):
            await _fetch_blob_contents(blob_service, [_ref()])

    @pytest.mark.asyncio
    async def test_collects_not_found_by_field_path(self) -> None:
        blob_service = _BlobServiceDouble()
        blob_service.read_blob_content.side_effect = BlobNotFoundError(BLOB1)

        with pytest.raises(BlobContentResolutionError) as exc_info:
            await _fetch_blob_contents(blob_service, [_ref()])

        assert exc_info.value.missing == ("source.options.x",)

    @pytest.mark.asyncio
    async def test_collects_not_ready_by_field_path(self) -> None:
        blob_service = _BlobServiceDouble()
        blob_service.read_blob_content.side_effect = BlobStateError(BLOB1, message="Blob is pending")

        with pytest.raises(BlobContentResolutionError) as exc_info:
            await _fetch_blob_contents(blob_service, [_ref()])

        assert exc_info.value.not_ready == (("source.options.x", "Blob is pending"),)


class TestSubstituteBlobContentRefs:
    def test_replaces_marker_with_decoded_string_and_audit_record(self) -> None:
        content = b"You are a helpful assistant."
        ref = _ref_with_hash(content)
        config = {
            "source": {
                "options": {
                    "system_prompt": {
                        "blob_ref": BLOB1,
                        "mode": "inline_content",
                        "sha256": ref.sha256,
                    }
                }
            }
        }

        substituted, audit = _substitute_blob_content_refs(
            deepcopy(config),
            {ref: content},
            refs=[ref],
            blob_metadata={ref.blob_id: ("text/plain", len(content))},
        )

        assert substituted["source"]["options"]["system_prompt"] == "You are a helpful assistant."
        assert audit == [
            ResolvedBlobContent(
                field_path="source.options.system_prompt",
                blob_id=UUID(BLOB1),
                content_hash=ref.sha256,
                byte_length=len(content),
                mime_type="text/plain",
                encoding="utf-8",
            )
        ]

    def test_replaces_plural_source_marker_with_decoded_string_and_named_audit_path(self) -> None:
        content = b"You are a helpful assistant."
        ref = _ref_with_hash(content, "source:orders.options.system_prompt")
        config = {
            "sources": {
                "orders": {
                    "options": {
                        "system_prompt": {
                            "blob_ref": BLOB1,
                            "mode": "inline_content",
                            "sha256": ref.sha256,
                        }
                    }
                }
            }
        }

        substituted, audit = _substitute_blob_content_refs(
            deepcopy(config),
            {ref: content},
            refs=[ref],
            blob_metadata={ref.blob_id: ("text/plain", len(content))},
        )

        assert substituted["sources"]["orders"]["options"]["system_prompt"] == "You are a helpful assistant."
        assert audit == [
            ResolvedBlobContent(
                field_path="source:orders.options.system_prompt",
                blob_id=UUID(BLOB1),
                content_hash=ref.sha256,
                byte_length=len(content),
                mime_type="text/plain",
                encoding="utf-8",
            )
        ]

    def test_replaces_node_and_output_paths(self) -> None:
        node_content = b"Classify strictly."
        output_content = b"Footer text"
        node_ref = _ref_with_hash(node_content, "node:classify.options.system_prompt", BLOB1)
        output_ref = _ref_with_hash(output_content, "output:writeback.options.footer_template", BLOB2)
        config = {
            "transforms": [
                {
                    "name": "classify",
                    "options": {
                        "system_prompt": {
                            "blob_ref": BLOB1,
                            "mode": "inline_content",
                            "sha256": node_ref.sha256,
                        }
                    },
                }
            ],
            "sinks": {
                "writeback": {
                    "options": {
                        "footer_template": {
                            "blob_ref": BLOB2,
                            "mode": "inline_content",
                            "sha256": output_ref.sha256,
                        }
                    }
                }
            },
        }

        substituted, audit = _substitute_blob_content_refs(
            deepcopy(config),
            {node_ref: node_content, output_ref: output_content},
            refs=[node_ref, output_ref],
            blob_metadata={
                node_ref.blob_id: ("text/plain", len(node_content)),
                output_ref.blob_id: ("text/plain", len(output_content)),
            },
        )

        assert substituted["transforms"][0]["options"]["system_prompt"] == "Classify strictly."
        assert substituted["sinks"]["writeback"]["options"]["footer_template"] == "Footer text"
        assert [entry.field_path for entry in audit] == [
            "node:classify.options.system_prompt",
            "output:writeback.options.footer_template",
        ]

    def test_hash_mismatch_propagates_integrity_error(self) -> None:
        content = b"You are a helpful assistant."
        ref = BlobInlineRef(
            field_path="source.options.system_prompt",
            blob_id=UUID(BLOB1),
            sha256="b" * 64,
            encoding="utf-8",
        )
        config = {"source": {"options": {"system_prompt": _marker(BLOB1, ref.sha256)}}}

        with pytest.raises(BlobIntegrityError):
            _substitute_blob_content_refs(
                config,
                {ref: content},
                refs=[ref],
                blob_metadata={ref.blob_id: ("text/plain", len(content))},
            )

    def test_decode_failures_are_batched_without_mutating_config(self) -> None:
        content = b"\xff\xfe\xfd"
        ref = BlobInlineRef(
            field_path="source.options.x",
            blob_id=UUID(BLOB1),
            sha256=hashlib.sha256(content).hexdigest(),
            encoding="utf-8",
        )
        config = {"source": {"options": {"x": _marker(BLOB1, ref.sha256)}}}
        original = deepcopy(config)

        with pytest.raises(BlobContentResolutionError) as exc_info:
            _substitute_blob_content_refs(
                config,
                {ref: content},
                refs=[ref],
                blob_metadata={ref.blob_id: ("text/plain", len(content))},
            )

        assert exc_info.value.undecodable == (("source.options.x", "utf-8"),)
        assert config == original


class TestValidateBlobContentRefs:
    @pytest.mark.asyncio
    async def test_async_returns_missing_violation_without_raising(self) -> None:
        blob_service = _BlobServiceDouble()
        blob_service.get_blob.side_effect = BlobNotFoundError(BLOB1)
        config = {"source": {"options": {"x": _marker(BLOB1, VALID_HASH)}}}

        violations = await _validate_blob_content_refs(blob_service, config, session_id=SESSION_ID)

        assert violations == [
            BlobInlineValidationViolation(
                category="missing",
                field_path="source.options.x",
                detail=f"blob {UUID(BLOB1)} not found",
            )
        ]

    @pytest.mark.asyncio
    async def test_async_treats_cross_session_blob_as_missing_before_metadata_checks(self) -> None:
        blob_service = _BlobServiceDouble()
        blob_service.get_blob.return_value = _blob_record(
            session_id=OTHER_SESSION_ID,
            status="ready",
            content_hash="b" * 64,
            size_bytes=128,
        )
        config = {"source": {"options": {"x": _marker(BLOB1, VALID_HASH)}}}

        violations = await _validate_blob_content_refs(blob_service, config, session_id=SESSION_ID)

        assert violations == [
            BlobInlineValidationViolation(
                category="missing",
                field_path="source.options.x",
                detail=f"blob {UUID(BLOB1)} not found",
            )
        ]

    @pytest.mark.asyncio
    async def test_async_returns_malformed_violation_without_lookup(self) -> None:
        blob_service = _BlobServiceDouble()
        config = {"source": {"options": {"x": {"blob_ref": BLOB1}}}}

        violations = await _validate_blob_content_refs(blob_service, config, session_id=SESSION_ID)

        assert violations == [
            BlobInlineValidationViolation(
                category="malformed",
                field_path="source.options.x",
                detail="missing mode",
            )
        ]
        blob_service.get_blob.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_returns_malformed_violation_for_non_string_marker_fields(self) -> None:
        blob_service = _BlobServiceDouble()
        config = {"source": {"options": {"x": {"blob_ref": BLOB1, "mode": []}}}}

        violations = await _validate_blob_content_refs(blob_service, config, session_id=SESSION_ID)

        assert violations == [
            BlobInlineValidationViolation(
                category="malformed",
                field_path="source.options.x",
                detail="mode must be string",
            )
        ]
        blob_service.get_blob.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_returns_hash_and_size_violations(self) -> None:
        blob_service = _BlobServiceDouble()
        blob_service.get_blob.return_value = _blob_record(content_hash="b" * 64, size_bytes=128)
        config = {"source": {"options": {"x": _marker(BLOB1, VALID_HASH)}}}

        violations = await _validate_blob_content_refs(
            blob_service,
            config,
            session_id=SESSION_ID,
            per_ref_byte_cap=64,
        )

        assert [violation.category for violation in violations] == ["hash_mismatch"]
        assert violations[0].field_path == "source.options.x"

    def test_sync_returns_missing_violation_without_raising(self) -> None:
        config = {"source": {"options": {"x": _marker(BLOB1, VALID_HASH)}}}

        violations = _validate_blob_content_refs_sync(
            blob_get_metadata=lambda _blob_id: None,
            config=config,
        )

        assert violations == [
            BlobInlineValidationViolation(
                category="missing",
                field_path="source.options.x",
                detail=f"blob {UUID(BLOB1)} not found",
            )
        ]

    def test_sync_returns_not_ready_and_oversized_violations(self) -> None:
        pending = _blob_record(status="pending")
        too_large = _blob_record(blob_id=BLOB2, content_hash=VALID_HASH, size_bytes=128)
        config = {
            "source": {"options": {"x": _marker(BLOB1, VALID_HASH)}},
            "sinks": {"writeback": {"options": {"footer": _marker(BLOB2, VALID_HASH)}}},
        }

        def get_metadata(blob_id: UUID) -> BlobRecord | None:
            if blob_id == UUID(BLOB1):
                return pending
            if blob_id == UUID(BLOB2):
                return too_large
            return None

        violations = _validate_blob_content_refs_sync(
            blob_get_metadata=get_metadata,
            config=config,
            per_ref_byte_cap=64,
            aggregate_byte_cap=64,
        )

        assert [violation.category for violation in violations] == ["not_ready", "oversized", "oversized"]
        assert [violation.field_path for violation in violations] == [
            "source.options.x",
            "output:writeback.options.footer",
            "(aggregate)",
        ]
