"""Tests for blob_csv_expand transform."""

from __future__ import annotations

import hashlib

from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

DYNAMIC_SCHEMA = {"mode": "observed", "guaranteed_fields": ["url", "blob_ref"]}


class _PayloadStoreFake:
    def __init__(self, content_by_hash: dict[str, bytes]) -> None:
        self.content_by_hash = content_by_hash

    def retrieve(self, content_hash: str) -> bytes:
        return self.content_by_hash[content_hash]


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _build_transform(**overrides):
    from elspeth.plugins.transforms.blob_csv_expand import BlobCSVExpand

    config = {
        "schema": DYNAMIC_SCHEMA,
        "blob_ref_field": "blob_ref",
    }
    config.update(overrides)
    return BlobCSVExpand(config)


def test_blob_csv_expand_parses_rows_and_preserves_url_order() -> None:
    body = b"id,name\n1,alice\n2,bob\n"
    blob_ref = _hash(body)
    transform = _build_transform()
    transform._payload_store = _PayloadStoreFake({blob_ref: body})

    result = transform.process(
        make_pipeline_row(
            {
                "url": "https://example.test/a.csv",
                "blob_ref": blob_ref,
                "manifest_index": 0,
            }
        ),
        make_context(),
    )

    assert result.status == "success"
    assert result.is_multi_row
    assert result.rows is not None
    assert [row.to_dict() for row in result.rows] == [
        {
            "url": "https://example.test/a.csv",
            "blob_ref": blob_ref,
            "manifest_index": 0,
            "id": "1",
            "name": "alice",
            "csv_row_index": 0,
        },
        {
            "url": "https://example.test/a.csv",
            "blob_ref": blob_ref,
            "manifest_index": 0,
            "id": "2",
            "name": "bob",
            "csv_row_index": 1,
        },
    ]
    assert transform.creates_tokens is True


def test_blob_csv_expand_supports_headerless_columns() -> None:
    body = b"1,alice\n2,bob\n"
    blob_ref = _hash(body)
    transform = _build_transform(columns=["id", "name"])
    transform._payload_store = _PayloadStoreFake({blob_ref: body})

    result = transform.process(
        make_pipeline_row({"url": "https://example.test/a.csv", "blob_ref": blob_ref}),
        make_context(),
    )

    assert result.status == "success"
    assert result.rows is not None
    assert [row.to_dict()["name"] for row in result.rows] == ["alice", "bob"]


def test_blob_csv_expand_fails_on_field_collision() -> None:
    body = b"url,name\nhttps://other.test,alice\n"
    blob_ref = _hash(body)
    transform = _build_transform()
    transform._payload_store = _PayloadStoreFake({blob_ref: body})

    result = transform.process(
        make_pipeline_row({"url": "https://example.test/a.csv", "blob_ref": blob_ref}),
        make_context(),
    )

    assert result.status == "error"
    assert result.reason is not None
    assert result.reason["reason"] == "field_collision"
    assert result.reason["fields"] == ["url"]


def test_blob_csv_expand_enforces_max_output_rows() -> None:
    body = b"id\n1\n2\n"
    blob_ref = _hash(body)
    transform = _build_transform(max_output_rows=1)
    transform._payload_store = _PayloadStoreFake({blob_ref: body})

    result = transform.process(
        make_pipeline_row({"url": "https://example.test/a.csv", "blob_ref": blob_ref}),
        make_context(),
    )

    assert result.status == "error"
    assert result.reason is not None
    assert result.reason["reason"] == "too_many_rows"
    assert result.reason["max_output_rows"] == 1


def test_discovery_registers_blob_csv_expand() -> None:
    from elspeth.plugins.infrastructure.manager import PluginManager

    manager = PluginManager()
    manager.register_builtin_plugins()

    transform = manager.get_transform_by_name("blob_csv_expand")
    assert transform.name == "blob_csv_expand"
