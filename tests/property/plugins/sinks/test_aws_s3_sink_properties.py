"""Property checks for deterministic S3 effect serialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
)
from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink
from tests.fixtures.base_classes import inject_write_failure

_CTX = RestrictedSinkEffectContext(
    run_id="property-run",
    run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
    operation_id="property-operation",
    sink_node_id="property-sink",
)


class _Missing(Exception):
    response: ClassVar[dict[str, object]] = {
        "Error": {"Code": "NoSuchKey"},
        "ResponseMetadata": {"HTTPStatusCode": 404},
    }


class _InspectClient:
    def head_object(self, **_kwargs: object) -> None:
        raise _Missing


def _member(ordinal: int, row: dict[str, Any]) -> SinkEffectMember:
    encoded = canonical_json(row).encode()
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"token-{ordinal}",
        row_id=f"row-{ordinal}",
        ingest_sequence=ordinal,
        lineage_json="[]",
        lineage_hash=sha256(b"[]").hexdigest(),
        payload_hash=sha256(encoded).hexdigest(),
        row=row,
        member_effect_id=sha256(f"member-{ordinal}-{encoded!r}".encode()).hexdigest(),
    )


def _effect_body(rows: list[dict[str, Any]], format_name: str) -> tuple[bytes, tuple[int, ...]]:
    sink = inject_write_failure(
        AWSS3Sink(
            {
                "bucket": "property-bucket",
                "key": "property/output",
                "format": format_name,
                "schema": {"mode": "observed"},
            }
        )
    )
    sink._s3_client = _InspectClient()
    members = tuple(_member(index, row) for index, row in enumerate(rows))
    effect_id = sha256(canonical_json({"format": format_name, "rows": rows}).encode()).hexdigest()
    inspection = sink.inspect_effect(
        SinkEffectInspectionRequest(effect_id=effect_id, target="{}", predecessor_descriptor=None),
        _CTX,
    )
    plan = sink.prepare_effect(
        SinkEffectPrepareRequest(
            effect_id=effect_id,
            effect_input=SinkEffectPipelineMembersInput(members=members, target_snapshot_members=members),
            inspection=inspection,
        ),
        _CTX,
    )
    return Path(str(plan.safe_evidence["staging_path"])).read_bytes(), tuple(plan.safe_evidence["diverted_ordinals"])


_SAFE_TEXT = st.text(alphabet=st.characters(blacklist_categories=("Cs", "Cc")), min_size=0, max_size=40)
_ROWS = st.lists(
    st.fixed_dictionaries({"id": st.integers(min_value=-(2**53) + 1, max_value=2**53 - 1), "name": _SAFE_TEXT}),
    min_size=1,
    max_size=15,
)


@settings(max_examples=40)
@given(rows=_ROWS)
def test_json_and_jsonl_effect_round_trip_arbitrary_safe_rows(rows: list[dict[str, Any]]) -> None:
    for format_name in ("json", "jsonl"):
        body, diverted = _effect_body(rows, format_name)
        actual = json.loads(body) if format_name == "json" else [json.loads(line) for line in body.splitlines()]
        assert actual == rows
        assert diverted == ()


@settings(max_examples=35)
@given(first=_ROWS, second=_ROWS)
def test_cumulative_target_snapshot_preserves_exact_row_order(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
) -> None:
    body, diverted = _effect_body([*first, *second], "json")
    assert json.loads(body) == [*first, *second]
    assert diverted == ()


@settings(max_examples=35)
@given(rows=_ROWS)
def test_csv_effect_serialization_keeps_one_record_per_safe_row(rows: list[dict[str, Any]]) -> None:
    body, diverted = _effect_body(rows, "csv")
    assert diverted == ()
    decoded = body.decode().splitlines()
    assert len(decoded) == len(rows) + 1
    assert decoded[0] == "id,name"
