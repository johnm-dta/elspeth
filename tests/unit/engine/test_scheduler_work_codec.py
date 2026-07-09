"""Parity net for SchedulerWorkCodec (elspeth-6291c51766).

The scheduler row is authoritative for resume, and the idempotent enqueue
reconciles by deterministic ``work_item_id`` + strict field equality. The
field derivation used to live in four hand-synced encoders inside
``RowProcessor`` (source ingest initial claim, READY barrier emission,
idempotent enqueue, resume rehydrate), kept in sync by comments. The codec
owns the mapping in both directions; these tests replace the "keep them in
sync by hand" comments with an enforced round-trip invariant.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime

import pytest

from elspeth.contracts import TokenInfo
from elspeth.contracts.scheduler import BarrierEmission, TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.engine.scheduler_work_codec import (
    TERMINAL_NODE_SENTINEL,
    ScheduledWorkFields,
    SchedulerWorkCodec,
)
from elspeth.engine.work_items import WorkItem

_CONTRACT = SchemaContract(mode="OBSERVED", fields=(), locked=True)
_NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)

_STEP_MAP = {NodeID("normalize"): 3, NodeID("enrich"): 4, NodeID("queue-node"): 5}
_STRUCTURAL = frozenset({NodeID("queue-node")})
_BARRIER = frozenset({NodeID("agg-node")})


def _resolve_node_cursor(node_id: NodeID | None) -> str | None:
    return None if node_id is None else str(node_id)


def _resolve_step_index(node_id: NodeID | None) -> int:
    if node_id is None:
        return max(_STEP_MAP.values(), default=0) + 1
    return _STEP_MAP[node_id]


def _queue_key(item: WorkItem) -> str | None:
    if item.current_node_id in _STRUCTURAL and item.coalesce_name is None:
        return str(item.current_node_id)
    return None


def _barrier_key(item: WorkItem) -> str | None:
    if item.current_node_id in _BARRIER:
        return str(item.current_node_id)
    if item.coalesce_name is not None:
        return str(item.coalesce_name)
    return None


def _create_work_item(
    *,
    token: TokenInfo,
    current_node_id: NodeID | None,
    coalesce_name: CoalesceName | None = None,
    coalesce_node_id: NodeID | None = None,
    on_success_sink: str | None = None,
) -> WorkItem:
    return WorkItem(
        token=token,
        current_node_id=current_node_id,
        coalesce_node_id=coalesce_node_id,
        coalesce_name=coalesce_name,
        on_success_sink=on_success_sink,
    )


def _make_codec(*, resolve_ingest_sequence=None) -> SchedulerWorkCodec:
    def _default_resolver(row_id: str) -> int:
        return 11

    return SchedulerWorkCodec(
        serialize_row_payload=TokenSchedulerRepository.serialize_row_payload,
        deserialize_row_payload=TokenSchedulerRepository.deserialize_row_payload,
        resolve_node_cursor=_resolve_node_cursor,
        resolve_step_index=_resolve_step_index,
        resolve_ingest_sequence=resolve_ingest_sequence or _default_resolver,
        queue_key_for_item=_queue_key,
        barrier_key_for_item=_barrier_key,
        create_work_item=_create_work_item,
    )


def _make_item(**overrides) -> WorkItem:
    token = TokenInfo(
        row_id="row-1",
        token_id="tok-1",
        row_data=PipelineRow({"id": 1, "name": "alpha"}, _CONTRACT),
        branch_name="branch-a",
        fork_group_id="fork-1",
        join_group_id="join-1",
        expand_group_id="expand-1",
    )
    defaults = {
        "token": token,
        "current_node_id": NodeID("normalize"),
        "coalesce_node_id": NodeID("merge-node"),
        "coalesce_name": CoalesceName("merge"),
        "on_success_sink": "default-sink",
    }
    defaults.update(overrides)
    return WorkItem(**defaults)


def _scheduler_row_from_fields(fields: ScheduledWorkFields) -> TokenWorkItem:
    """Materialize the durable row exactly as the repository persists the bundle."""
    return TokenWorkItem(
        work_item_id="wi-1",
        run_id="run-1",
        token_id=fields.token_id,
        row_id=fields.row_id,
        node_id=fields.node_id,
        step_index=fields.step_index,
        ingest_sequence=fields.ingest_sequence,
        row_payload_json=fields.row_payload_json,
        status=TokenWorkStatus.READY,
        attempt=1,
        available_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        queue_key=fields.queue_key,
        barrier_key=fields.barrier_key,
        on_success_sink=fields.on_success_sink,
        branch_name=fields.branch_name,
        fork_group_id=fields.fork_group_id,
        join_group_id=fields.join_group_id,
        expand_group_id=fields.expand_group_id,
        coalesce_node_id=fields.coalesce_node_id,
        coalesce_name=fields.coalesce_name,
    )


class TestReadyFieldsRoundTrip:
    def test_work_item_round_trips_through_scheduler_row(self) -> None:
        codec = _make_codec()
        item = _make_item()

        fields = codec.ready_fields(item)
        rehydrated = codec.work_item_from_scheduler(_scheduler_row_from_fields(fields))

        assert rehydrated.token.row_id == item.token.row_id
        assert rehydrated.token.token_id == item.token.token_id
        assert rehydrated.token.branch_name == item.token.branch_name
        assert rehydrated.token.fork_group_id == item.token.fork_group_id
        assert rehydrated.token.join_group_id == item.token.join_group_id
        assert rehydrated.token.expand_group_id == item.token.expand_group_id
        # PipelineRow has identity equality; payload parity is dict-level.
        assert rehydrated.token.row_data.to_dict() == item.token.row_data.to_dict()
        assert rehydrated.current_node_id == item.current_node_id
        assert rehydrated.coalesce_node_id == item.coalesce_node_id
        assert rehydrated.coalesce_name == item.coalesce_name
        assert rehydrated.on_success_sink == item.on_success_sink

    def test_derived_fields_match_resolvers(self) -> None:
        codec = _make_codec()
        item = _make_item()

        fields = codec.ready_fields(item)

        assert fields.node_id == "normalize"
        assert fields.step_index == 3
        assert fields.ingest_sequence == 11
        # Coalesce items key their barrier by coalesce name.
        assert fields.queue_key is None
        assert fields.barrier_key == "merge"
        assert fields.coalesce_node_id == "merge-node"
        assert fields.coalesce_name == "merge"

    def test_structural_node_derives_queue_key(self) -> None:
        codec = _make_codec()
        item = _make_item(
            current_node_id=NodeID("queue-node"),
            coalesce_node_id=None,
            coalesce_name=None,
        )

        fields = codec.ready_fields(item)

        assert fields.queue_key == "queue-node"
        assert fields.barrier_key is None

    def test_explicit_ingest_sequence_skips_resolver(self) -> None:
        def _forbidden(row_id: str) -> int:
            raise AssertionError("ingest path must not resolve ingest_sequence from the repository")

        codec = _make_codec(resolve_ingest_sequence=_forbidden)
        item = _make_item()

        fields = codec.ready_fields(item, ingest_sequence=7)

        assert fields.ingest_sequence == 7


class TestReadyEmissionParity:
    def test_emission_fields_equal_ready_fields(self) -> None:
        codec = _make_codec()
        item = _make_item()

        fields = codec.ready_fields(item)
        emission = codec.ready_emission(item)

        assert isinstance(emission, BarrierEmission)
        for field in dataclass_fields(ScheduledWorkFields):
            assert getattr(emission, field.name) == getattr(fields, field.name), field.name

    def test_emission_is_ready_lane_shaped(self) -> None:
        codec = _make_codec()
        emission = codec.ready_emission(_make_item())

        # READY continuations never carry sink handoff fields.
        assert emission.sink_name is None
        assert emission.outcome is None
        assert emission.path is None
        assert emission.error_hash is None
        assert emission.error_message is None
        assert emission.attempt == 1


class TestRehydrateCursor:
    @pytest.mark.parametrize("stored_node_id", [None, TERMINAL_NODE_SENTINEL])
    def test_terminal_lane_rehydrates_to_none_cursor(self, stored_node_id: str | None) -> None:
        codec = _make_codec()
        fields = codec.ready_fields(_make_item())
        row = _scheduler_row_from_fields(fields)
        terminal_row = TokenWorkItem(
            **{
                **{f.name: getattr(row, f.name) for f in dataclass_fields(TokenWorkItem)},
                "node_id": stored_node_id,
            }
        )

        rehydrated = codec.work_item_from_scheduler(terminal_row)

        assert rehydrated.current_node_id is None
