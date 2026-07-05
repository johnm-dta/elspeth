"""Property-based tests for Azure Blob Storage source plugin.

Verifies CSV/JSON round-trip integrity and quarantine completeness
using Hypothesis-generated data.
"""

from __future__ import annotations

import csv
import io
import json
import keyword
from typing import Any
from unittest.mock import patch

from hypothesis import given
from hypothesis import strategies as st

from elspeth.plugins.sources.field_normalization import normalize_field_name
from tests.fixtures.factories import make_operation_context
from tests.strategies.settings import SLOW_SETTINGS

# ---------------------------------------------------------------------------
# Shared constants (mirror unit test conventions)
# ---------------------------------------------------------------------------

FAKE_CONN_STRING = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
DYNAMIC_SCHEMA: dict[str, Any] = {"mode": "observed"}
QUARANTINE_SINK = "quarantine"
PATCH_AUTH = "elspeth.plugins.infrastructure.azure_auth.AzureAuthConfig.create_blob_service_client"


class _BlobDownload:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class _BlobClientFake:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def download_blob(self) -> _BlobDownload:
        return _BlobDownload(self._data)


class _ContainerClientFake:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def get_blob_client(self, *_args: Any, **_kwargs: Any) -> _BlobClientFake:
        return _BlobClientFake(self._data)


class _BlobServiceClientFake:
    def __init__(self, data: bytes = b"") -> None:
        self._data = data

    def get_container_client(self, *_args: Any, **_kwargs: Any) -> _ContainerClientFake:
        return _ContainerClientFake(self._data)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Safe column names: valid ASCII identifiers starting with a letter.
# Avoids normalization surprises (digit-prefix, keyword suffixing).
safe_column_name = st.from_regex(r"[a-z][a-z0-9]{0,9}", fullmatch=True).filter(lambda s: not keyword.iskeyword(s))

# Safe text for CSV values: ASCII letters/digits/spaces,
# no commas, newlines, or quotes.
safe_text = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs"), max_codepoint=0x7E),
).filter(lambda s: s.strip() and "," not in s and "\n" not in s and '"' not in s)

# JSON scalars (no NaN/Inf -- strict JSON compliance)
json_scalar = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.text(
        min_size=0,
        max_size=20,
        alphabet=st.characters(whitelist_categories=("L", "N")),
    ),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.booleans(),
    st.none(),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid config with connection_string auth."""
    config: dict[str, Any] = {
        "connection_string": FAKE_CONN_STRING,
        "container": "test-container",
        "blob_path": "data/input.csv",
        "schema": DYNAMIC_SCHEMA,
        "on_validation_failure": QUARANTINE_SINK,
    }
    config.update(overrides)
    return config


def _mock_blob_download(data: bytes) -> _BlobServiceClientFake:
    """Create a service client fake that returns data from download_blob().readall()."""
    return _BlobServiceClientFake(data)


def _make_source(config: dict[str, Any]) -> Any:
    """Create AzureBlobSource with patched auth."""
    from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

    with patch(PATCH_AUTH, return_value=_BlobServiceClientFake()):
        return AzureBlobSource(config)


# ---------------------------------------------------------------------------
# CSV round-trip properties
# ---------------------------------------------------------------------------


class TestAzureBlobSourceCSVProperties:
    """CSV round-trip: generated columns and values survive source loading."""

    @given(
        columns=st.lists(safe_column_name, min_size=1, max_size=5, unique=True),
        data=st.data(),
    )
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_csv_round_trip_preserves_values(
        self,
        mock_create: Any,
        columns: list[str],
        data: st.DataObject,
    ) -> None:
        """Serialize random CSV, load through source, verify values match."""
        # Generate 1-5 rows of safe text values for each column
        row_count = data.draw(st.integers(min_value=1, max_value=5))
        input_rows: list[dict[str, str]] = []
        for _ in range(row_count):
            row = {col: data.draw(safe_text) for col in columns}
            input_rows.append(row)

        # Serialize to CSV bytes
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns)
        writer.writeheader()
        writer.writerows(input_rows)
        csv_bytes = buf.getvalue().encode("utf-8")

        # Feed through source
        source = _make_source(_base_config())
        ctx = make_operation_context(plugin_name="azure_blob")
        mock_create.return_value = _mock_blob_download(csv_bytes)

        rows = list(source.load(ctx))

        # All valid rows must match input (CSV values are always strings)
        valid_rows = [r for r in rows if not r.is_quarantined]
        assert len(valid_rows) == row_count

        for source_row, expected in zip(valid_rows, input_rows, strict=True):
            for col in columns:
                # Column names are already lowercase identifiers
                assert source_row.row[col] == expected[col]


# ---------------------------------------------------------------------------
# JSON round-trip properties
# ---------------------------------------------------------------------------


class TestAzureBlobSourceJSONProperties:
    """JSON round-trip: generated objects survive source loading."""

    @given(
        keys=st.lists(
            st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True),
            min_size=1,
            max_size=5,
            unique_by=normalize_field_name,
        ),
        data=st.data(),
    )
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_json_round_trip_preserves_structure(
        self,
        mock_create: Any,
        keys: list[str],
        data: st.DataObject,
    ) -> None:
        """Serialize random JSON array, load through source, verify row count.

        Uses integer values for all keys to ensure type homogeneity across rows.
        The observed schema locks types on the first row; mixed types would
        quarantine subsequent rows, which is correct behavior but not what
        this test targets (round-trip structure preservation).
        """
        row_count = data.draw(st.integers(min_value=1, max_value=5))
        # Use integers for all values to guarantee type consistency
        # across rows (observed schema locks types from first row).
        int_value = st.integers(min_value=-1000, max_value=1000)
        input_records: list[dict[str, Any]] = []
        for _ in range(row_count):
            record = {k: data.draw(int_value) for k in keys}
            input_records.append(record)

        blob_bytes = json.dumps(input_records).encode("utf-8")

        source = _make_source(_base_config(format="json"))
        ctx = make_operation_context(plugin_name="azure_blob")
        mock_create.return_value = _mock_blob_download(blob_bytes)

        rows = list(source.load(ctx))

        valid_rows = [r for r in rows if not r.is_quarantined]
        assert len(valid_rows) == row_count


# ---------------------------------------------------------------------------
# Quarantine properties
# ---------------------------------------------------------------------------


class TestAzureBlobSourceQuarantineProperties:
    """Random garbage fed as JSON must never be silently dropped."""

    @given(garbage=st.binary(min_size=1, max_size=200))
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_garbage_json_never_silently_dropped(
        self,
        mock_create: Any,
        garbage: bytes,
    ) -> None:
        """If garbage is not a valid JSON array, all returned rows are quarantined."""
        # Check if garbage accidentally produces a valid JSON array
        is_valid_json_array = False
        try:
            parsed = json.loads(garbage.decode("utf-8"))
            if isinstance(parsed, list):
                is_valid_json_array = True
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            pass

        source = _make_source(_base_config(format="json"))
        ctx = make_operation_context(plugin_name="azure_blob")
        mock_create.return_value = _mock_blob_download(garbage)

        rows = list(source.load(ctx))

        if not is_valid_json_array:
            # Every returned row must be quarantined -- nothing silently dropped
            assert len(rows) > 0, "Garbage input must produce at least one quarantine row"
            assert all(r.is_quarantined for r in rows), "Non-array garbage must quarantine all rows"
