"""Property-based tests for Azure Blob Storage sink plugin.

Verifies hash determinism, JSONL round-trip integrity, and buffering
equivalence using Hypothesis-generated data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import MagicMock, patch

from hypothesis import given
from hypothesis import strategies as st

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_operation_context
from tests.strategies.settings import SLOW_SETTINGS

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FAKE_CONN_STRING = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
PATCH_AUTH = "elspeth.plugins.infrastructure.azure_auth.AzureAuthConfig.create_blob_service_client"
FIXED_SCHEMA: dict[str, Any] = {
    "mode": "fixed",
    "fields": ["id: int", "name: str", "score: float?"],
}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

row_strategy = st.fixed_dictionaries(
    {
        "id": st.integers(min_value=0, max_value=1000),
        "name": st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
        "score": st.one_of(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-1e6,
                max_value=1e6,
            ),
            st.none(),
        ),
    }
)
rows_strategy = st.lists(row_strategy, min_size=1, max_size=5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(**overrides: Any) -> dict[str, Any]:
    """Minimal valid config dict."""
    cfg: dict[str, Any] = {
        "connection_string": FAKE_CONN_STRING,
        "container": "test-container",
        "blob_path": "output.jsonl",
        "schema": FIXED_SCHEMA,
        "format": "jsonl",
    }
    cfg.update(overrides)
    return cfg


def _mock_blob_upload() -> tuple[MagicMock, MagicMock]:
    """Create mock service client returning (service, blob_client) for upload assertions."""
    mock_blob_client = MagicMock()
    mock_container = MagicMock()
    mock_container.get_blob_client.return_value = mock_blob_client
    mock_container.close = MagicMock()
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container
    return mock_service, mock_blob_client


def _make_sink_ctx() -> PluginContext:
    """Build a PluginContext suitable for sink.write() calls."""
    return make_operation_context(
        operation_type="sink_write",
        node_id="sink",
        node_type="SINK",
        plugin_name="azure_blob",
    )


def _get_uploaded_bytes(mock_blob: MagicMock) -> bytes:
    """Extract the bytes passed to upload_blob."""
    data: bytes = mock_blob.upload_blob.call_args[0][0]
    return data


# ---------------------------------------------------------------------------
# Hash properties
# ---------------------------------------------------------------------------


class TestAzureBlobSinkHashProperties:
    """Artifact hash must match SHA-256 of uploaded content."""

    @given(rows=rows_strategy)
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_hash_matches_uploaded_content(
        self,
        mock_create: MagicMock,
        rows: list[dict[str, object]],
    ) -> None:
        """Artifact hash == SHA-256 of captured upload bytes, size matches."""
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        result = sink.write(rows, ctx)

        uploaded = _get_uploaded_bytes(mock_blob)
        expected_hash = hashlib.sha256(uploaded).hexdigest()
        assert result.artifact.content_hash == expected_hash
        assert result.artifact.size_bytes == len(uploaded)

    @given(rows=rows_strategy)
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_same_rows_produce_same_hash(
        self,
        mock_create: MagicMock,
        rows: list[dict[str, object]],
    ) -> None:
        """Same rows written to two separate sink instances produce same hash."""
        hashes = []
        for _ in range(2):
            sink = inject_write_failure(AzureBlobSink(_base_config()))
            ctx = _make_sink_ctx()
            mock_service, _mock_blob = _mock_blob_upload()
            mock_create.return_value = mock_service

            result = sink.write(rows, ctx)
            hashes.append(result.artifact.content_hash)

        assert hashes[0] == hashes[1]


# ---------------------------------------------------------------------------
# JSONL round-trip properties
# ---------------------------------------------------------------------------


class TestAzureBlobSinkJSONLRoundTrip:
    """JSONL output can be parsed back and values match."""

    @given(rows=rows_strategy)
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_jsonl_round_trip(
        self,
        mock_create: MagicMock,
        rows: list[dict[str, object]],
    ) -> None:
        """Write rows as JSONL, parse uploaded bytes back, verify values match."""
        sink = inject_write_failure(AzureBlobSink(_base_config(format="jsonl")))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink.write(rows, ctx)

        uploaded = _get_uploaded_bytes(mock_blob)
        lines = uploaded.decode("utf-8").strip().split("\n")
        assert len(lines) == len(rows)

        for parsed_line, original in zip(lines, rows, strict=True):
            parsed = json.loads(parsed_line)
            for key, expected_value in original.items():
                actual_value = parsed[key]
                if expected_value is None:
                    assert actual_value is None
                elif isinstance(expected_value, float):
                    assert abs(actual_value - expected_value) < 1e-9
                else:
                    assert actual_value == expected_value


# ---------------------------------------------------------------------------
# Buffering properties
# ---------------------------------------------------------------------------


class TestAzureBlobSinkBufferingProperties:
    """Cumulative buffering: write(A) + write(B) == write(A+B)."""

    @given(
        rows_a=rows_strategy,
        rows_b=rows_strategy,
    )
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_two_writes_equals_one_combined_write(
        self,
        mock_create: MagicMock,
        rows_a: list[dict[str, object]],
        rows_b: list[dict[str, object]],
    ) -> None:
        """write(A) then write(B) produces same blob as write(A+B)."""
        # Two-write path
        sink_split = inject_write_failure(AzureBlobSink(_base_config()))
        ctx_split = _make_sink_ctx()
        mock_service_split, mock_blob_split = _mock_blob_upload()
        mock_create.return_value = mock_service_split

        sink_split.write(rows_a, ctx_split)
        sink_split.write(rows_b, ctx_split)

        uploaded_split = _get_uploaded_bytes(mock_blob_split)

        # One-write path
        sink_combined = inject_write_failure(AzureBlobSink(_base_config()))
        ctx_combined = _make_sink_ctx()
        mock_service_combined, mock_blob_combined = _mock_blob_upload()
        mock_create.return_value = mock_service_combined

        sink_combined.write(rows_a + rows_b, ctx_combined)

        uploaded_combined = _get_uploaded_bytes(mock_blob_combined)

        assert uploaded_split == uploaded_combined
