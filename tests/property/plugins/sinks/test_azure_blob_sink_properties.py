"""Property-based checks for deterministic Azure Blob effect plans."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
)
from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
from tests.fixtures.base_classes import inject_write_failure
from tests.strategies.settings import SLOW_SETTINGS

_CONNECTION = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
_SCHEMA: dict[str, Any] = {
    "mode": "fixed",
    "fields": ["id: int", "name: str", "score: float?"],
}
_CTX = RestrictedSinkEffectContext(
    run_id="property-run",
    run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
    operation_id="property-operation",
    sink_node_id="property-sink",
)


class ResourceNotFoundError(Exception):
    pass


class _Blob:
    def get_blob_properties(self) -> None:
        raise ResourceNotFoundError


class _Container:
    def get_blob_client(self, *_args: object, **_kwargs: object) -> _Blob:
        return _Blob()

    def close(self) -> None:
        return None


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


def _effect_plan(rows: list[dict[str, Any]]):
    sink = inject_write_failure(
        AzureBlobSink(
            {
                "connection_string": _CONNECTION,
                "container": "property-container",
                "blob_path": "output.jsonl",
                "schema": _SCHEMA,
                "format": "jsonl",
            }
        )
    )
    sink._container_client = _Container()  # type: ignore[assignment]
    members = tuple(_member(index, row) for index, row in enumerate(rows))
    effect_id = sha256(canonical_json(rows).encode()).hexdigest()
    inspection = sink.inspect_effect(
        SinkEffectInspectionRequest(effect_id=effect_id, target="{}", predecessor_descriptor=None),
        _CTX,
    )
    return sink.prepare_effect(
        SinkEffectPrepareRequest(
            effect_id=effect_id,
            effect_input=SinkEffectPipelineMembersInput(members=members, target_snapshot_members=members),
            inspection=inspection,
        ),
        _CTX,
    )


row_strategy = st.fixed_dictionaries(
    {
        "id": st.integers(min_value=0, max_value=1000),
        "name": st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        "score": st.one_of(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            st.none(),
        ),
    }
)
rows_strategy = st.lists(row_strategy, min_size=1, max_size=5)


@given(rows=rows_strategy)
@SLOW_SETTINGS
def test_effect_descriptor_hash_matches_staged_content(rows: list[dict[str, Any]]) -> None:
    plan = _effect_plan(rows)
    body = Path(str(plan.safe_evidence["staging_path"])).read_bytes()
    assert plan.expected_descriptor is not None
    assert plan.expected_descriptor.content_hash == sha256(body).hexdigest()
    assert plan.expected_descriptor.size_bytes == len(body)


@given(rows=rows_strategy)
@SLOW_SETTINGS
def test_same_rows_produce_same_effect_payload_hash(rows: list[dict[str, Any]]) -> None:
    assert _effect_plan(rows).payload_hash == _effect_plan(rows).payload_hash


@given(rows=rows_strategy)
@SLOW_SETTINGS
def test_jsonl_effect_round_trip(rows: list[dict[str, Any]]) -> None:
    plan = _effect_plan(rows)
    body = Path(str(plan.safe_evidence["staging_path"])).read_text()
    assert [json.loads(line) for line in body.splitlines()] == rows


@given(rows_a=rows_strategy, rows_b=rows_strategy)
@SLOW_SETTINGS
def test_cumulative_target_snapshot_equals_combined_serialization(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
) -> None:
    combined = [*rows_a, *rows_b]
    plan = _effect_plan(combined)
    body = Path(str(plan.safe_evidence["staging_path"])).read_text()
    assert [json.loads(line) for line in body.splitlines()] == combined
