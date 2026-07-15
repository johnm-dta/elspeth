"""Property checks for bounded AWS S3 sink serialization and cumulative writes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink
from tests.fixtures.base_classes import inject_write_failure


@dataclass
class _Context:
    run_id: str = "property-run"
    contract: Any = None
    landscape: Any = None
    operation_id: str = "property-operation"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _Client:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.bodies: list[bytes] = []

    def put_object(self, **kwargs: Any) -> dict[str, str]:
        self.requests.append(kwargs)
        kwargs["Body"].seek(0)
        self.bodies.append(kwargs["Body"].read())
        return {"ETag": f'"etag-{len(self.requests)}"'}

    def close(self) -> None:
        pass


def _sink(*, format: str = "json", overwrite: bool = True) -> tuple[AWSS3Sink, _Client]:
    sink = inject_write_failure(
        AWSS3Sink(
            {
                "bucket": "property-bucket",
                "key": "property/output",
                "format": format,
                "overwrite": overwrite,
                "schema": {"mode": "observed"},
            }
        )
    )
    client = _Client()
    sink._s3_client = client
    return sink, client


_SAFE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=0,
    max_size=40,
)
_ROWS = st.lists(st.fixed_dictionaries({"id": st.integers(), "name": _SAFE_TEXT}), min_size=1, max_size=15)


@settings(max_examples=40)
@given(rows=_ROWS)
def test_json_and_jsonl_round_trip_arbitrary_safe_rows(rows: list[dict[str, Any]]) -> None:
    for format in ("json", "jsonl"):
        sink, client = _sink(format=format)
        sink.write(rows, _Context())
        if format == "json":
            actual = json.loads(client.bodies[0])
        else:
            actual = [json.loads(line) for line in client.bodies[0].splitlines()]
        assert actual == rows


@settings(max_examples=35)
@given(first=_ROWS, second=_ROWS)
def test_conditional_rewrites_preserve_exact_cumulative_row_order(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
) -> None:
    sink, client = _sink(format="json", overwrite=False)
    context = _Context()
    sink.write(first, context)
    sink.write(second, context)
    assert json.loads(client.bodies[1]) == [*first, *second]
    assert client.requests[0]["IfNoneMatch"] == "*"
    assert client.requests[1]["IfMatch"] == '"etag-1"'


@settings(max_examples=35)
@given(rows=_ROWS)
def test_csv_serialization_keeps_one_record_per_safe_row(rows: list[dict[str, Any]]) -> None:
    sink, client = _sink(format="csv")
    result = sink.write(rows, _Context())
    assert result.diversions == ()
    decoded = client.bodies[0].decode("utf-8").splitlines()
    assert len(decoded) == len(rows) + 1
    assert decoded[0] == "id,name"
