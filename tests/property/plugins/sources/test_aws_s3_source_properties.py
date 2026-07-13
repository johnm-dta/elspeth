"""Property tests for bounded AWS S3 source parsing and transport chunking."""

from __future__ import annotations

import csv
import io
import itertools
import json
import keyword
from dataclasses import dataclass, field
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from elspeth.plugins.sources.field_normalization import normalize_field_name
from tests.strategies.settings import SLOW_SETTINGS

safe_column_name = st.from_regex(r"[a-z][a-z0-9]{0,9}", fullmatch=True).filter(lambda value: not keyword.iskeyword(value))
safe_text = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs"), max_codepoint=0x7E),
).filter(lambda value: value.strip() and "," not in value and "\n" not in value and '"' not in value)
json_scalar = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.text(min_size=0, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.booleans(),
    st.none(),
)


@dataclass
class _Body:
    chunks: list[bytes]
    closed: bool = False

    def read(self, _size: int) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""

    def close(self) -> None:
        self.closed = True


@dataclass
class _Client:
    data: bytes
    split_points: tuple[int, ...] = ()
    bodies: list[_Body] = field(default_factory=list)

    def head_object(self, **_kwargs: Any) -> dict[str, Any]:
        return {"ContentLength": len(self.data), "ETag": '"property"'}

    def get_object(self, **_kwargs: Any) -> dict[str, Any]:
        boundaries = (0, *sorted(point for point in self.split_points if 0 < point < len(self.data)), len(self.data))
        body = _Body([self.data[start:stop] for start, stop in itertools.pairwise(boundaries)])
        self.bodies.append(body)
        return {"ContentLength": len(self.data), "Body": body}

    def close(self) -> None:
        return None


@dataclass
class _Context:
    calls: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[dict[str, Any]] = field(default_factory=list)

    def record_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def record_validation_error(self, **kwargs: Any) -> None:
        self.validation_errors.append(kwargs)


def _load(data: bytes, *, source_format: str, split_points: tuple[int, ...] = ()) -> tuple[list[Any], Any, _Client, _Context]:
    from elspeth.plugins.sources.aws_s3_source import AWSS3Source

    source = AWSS3Source(
        {
            "bucket": "property-bucket",
            "key": f"property.{source_format}",
            "format": source_format,
            "schema": {"mode": "observed"},
            "on_validation_failure": "quarantine",
        }
    )
    client = _Client(data, split_points)
    source._s3_client = client
    context = _Context()
    return list(source.load(context)), source, client, context


@given(
    columns=st.lists(safe_column_name, min_size=1, max_size=5, unique=True),
    values=st.lists(safe_text, min_size=1, max_size=5),
)
@SLOW_SETTINGS
def test_csv_generated_rows_round_trip(columns: list[str], values: list[str]) -> None:
    width = min(len(columns), len(values))
    columns = columns[:width]
    values = values[:width]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerow(values)
    rows, _, client, _ = _load(buffer.getvalue().encode(), source_format="csv")
    assert [row.row for row in rows] == [dict(zip((normalize_field_name(name) for name in columns), values, strict=True))]
    assert client.bodies[0].closed


@given(row=st.dictionaries(safe_column_name, json_scalar, min_size=1, max_size=8))
@SLOW_SETTINGS
def test_json_generated_objects_round_trip(row: dict[str, Any]) -> None:
    rows, _, client, _ = _load(json.dumps([row]).encode(), source_format="json")
    assert [result.row for result in rows] == [{normalize_field_name(key): value for key, value in row.items()}]
    assert client.bodies[0].closed


@given(value=st.one_of(st.integers(), st.text(), st.booleans(), st.none()))
@SLOW_SETTINGS
def test_json_non_array_roots_never_silently_disappear(value: Any) -> None:
    rows, _, client, context = _load(json.dumps(value).encode(), source_format="json")
    assert rows and rows[0].is_quarantined
    assert context.validation_errors
    assert client.bodies[0].closed


@given(split_points=st.lists(st.integers(min_value=1, max_value=80), max_size=12, unique=True).map(tuple))
@SLOW_SETTINGS
def test_transport_chunk_partitions_preserve_rows_contract_hash_and_cleanup(split_points: tuple[int, ...]) -> None:
    data = b'[{"id":1,"name":"Ada"},{"id":2,"name":"Grace"}]'
    baseline_rows, baseline_source, _, baseline_context = _load(data, source_format="json")
    rows, source, client, context = _load(data, source_format="json", split_points=split_points)
    assert [row.row for row in rows] == [row.row for row in baseline_rows]
    assert source.require_schema_contract() == baseline_source.require_schema_contract()
    assert context.calls[0]["response_data"]["content_hash"] == baseline_context.calls[0]["response_data"]["content_hash"]
    assert client.bodies[0].closed
