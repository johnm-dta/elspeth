"""Unit tests for checkpoint/recovery domain contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts import Checkpoint, ResumeCheck, ResumePoint
from elspeth.contracts.aggregation_checkpoint import (
    AggregationCheckpointState,
    AggregationNodeCheckpoint,
    AggregationTokenCheckpoint,
)
from elspeth.contracts.errors import AuditIntegrityError


def _checkpoint() -> Checkpoint:
    return Checkpoint(
        checkpoint_id="cp-001",
        run_id="run-001",
        token_id="tok-001",
        node_id="node-001",
        sequence_number=1,
        created_at=datetime.now(UTC),
        upstream_topology_hash="a" * 64,
        checkpoint_node_config_hash="b" * 64,
        format_version=Checkpoint.CURRENT_FORMAT_VERSION,
    )


def _make_agg_state() -> AggregationCheckpointState:
    """Create a minimal typed aggregation checkpoint state for testing."""
    return AggregationCheckpointState(
        version="4.0",
        nodes={
            "node-001": AggregationNodeCheckpoint(
                tokens=(
                    AggregationTokenCheckpoint(
                        token_id="tok-buf-001",
                        row_id="row-buf-001",
                        branch_name=None,
                        fork_group_id=None,
                        join_group_id=None,
                        expand_group_id=None,
                        row_data={"buffered_rows": 2},
                        contract_version="abc123",
                        contract={"mode": "FLEXIBLE", "locked": False, "version_hash": "abc123", "fields": []},
                    ),
                ),
                batch_id="batch-001",
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=None,
            ),
        },
    )


def test_resume_check_accepts_true_without_reason() -> None:
    check = ResumeCheck(can_resume=True)
    assert check.can_resume is True
    assert check.reason is None


def test_resume_check_rejects_true_with_reason() -> None:
    with pytest.raises(ValueError, match="can_resume=True should not have a reason"):
        ResumeCheck(can_resume=True, reason="unexpected")


def test_resume_check_rejects_false_without_reason() -> None:
    with pytest.raises(ValueError, match="can_resume=False must have a reason"):
        ResumeCheck(can_resume=False)


def test_resume_point_rejects_empty_token_id() -> None:
    with pytest.raises(ValueError, match="token_id must not be empty"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id="",
            node_id="node-001",
            sequence_number=1,
        )


def test_resume_point_rejects_empty_node_id() -> None:
    with pytest.raises(ValueError, match="node_id must not be empty"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id="tok-001",
            node_id="",
            sequence_number=1,
        )


def test_resume_point_rejects_negative_sequence_number() -> None:
    with pytest.raises(ValueError, match="sequence_number must be >= 0"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id="tok-001",
            node_id="node-001",
            sequence_number=-1,
        )


# === ResumePoint Tier 1 type guards (elspeth-0b184125ca, elspeth-65428b478c) ===


def test_resume_point_rejects_none_token_id() -> None:
    """Regression: elspeth-65428b478c — None must raise TypeError, not ValueError."""
    with pytest.raises(TypeError, match="token_id must be str"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id=None,  # type: ignore[arg-type]
            node_id="node-001",
            sequence_number=1,
        )


def test_resume_point_rejects_none_node_id() -> None:
    """Regression: elspeth-65428b478c — None must raise TypeError, not ValueError."""
    with pytest.raises(TypeError, match="node_id must be str"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id="tok-001",
            node_id=None,  # type: ignore[arg-type]
            sequence_number=1,
        )


def test_resume_point_rejects_int_token_id() -> None:
    """Non-string token_id is corruption, not an empty-string issue."""
    with pytest.raises(TypeError, match="token_id must be str"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id=42,  # type: ignore[arg-type]
            node_id="node-001",
            sequence_number=1,
        )


def test_resume_point_rejects_dict_aggregation_state() -> None:
    """Regression: elspeth-0b184125ca — raw dict must not be accepted as state."""
    with pytest.raises(TypeError, match="aggregation_state must be AggregationCheckpointState"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
            aggregation_state={"version": "3.0", "nodes": {}},  # type: ignore[arg-type]
        )


def test_resume_point_rejects_dict_coalesce_state() -> None:
    """Regression: elspeth-0b184125ca — raw dict must not be accepted as state."""
    with pytest.raises(TypeError, match="coalesce_state must be CoalesceCheckpointState"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
            coalesce_state={"version": "1.0", "pending": []},  # type: ignore[arg-type]
        )


# === ResumePoint checkpoint type guard + sequence_number type guard ===
# (elspeth-dce3a343a7, elspeth-52a31594ee)


def test_resume_point_rejects_non_checkpoint_type() -> None:
    """Regression: elspeth-dce3a343a7 — checkpoint must be Checkpoint, not raw dict."""
    with pytest.raises(TypeError, match="checkpoint must be Checkpoint"):
        ResumePoint(
            checkpoint={"run_id": "r1"},  # type: ignore[arg-type]
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
        )


def test_resume_point_rejects_none_checkpoint() -> None:
    """None checkpoint is corruption — crash, don't propagate."""
    with pytest.raises(TypeError, match="checkpoint must be Checkpoint"):
        ResumePoint(
            checkpoint=None,  # type: ignore[arg-type]
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
        )


def test_resume_point_rejects_float_sequence_number() -> None:
    """Regression: elspeth-52a31594ee — float 0.5 must not pass as sequence number."""
    with pytest.raises(TypeError, match="sequence_number must be int"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id="tok-001",
            node_id="node-001",
            sequence_number=0.5,  # type: ignore[arg-type]
        )


def test_resume_point_rejects_bool_sequence_number() -> None:
    """bool is subclass of int — True (value 1) must not pass as sequence number."""
    with pytest.raises(TypeError, match="sequence_number must be int"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id="tok-001",
            node_id="node-001",
            sequence_number=True,
        )


def test_resume_point_rejects_string_sequence_number() -> None:
    """String sequence number is corruption."""
    with pytest.raises(TypeError, match="sequence_number must be int"):
        ResumePoint(
            checkpoint=_checkpoint(),
            token_id="tok-001",
            node_id="node-001",
            sequence_number="3",  # type: ignore[arg-type]
        )


# === AggregationNodeCheckpoint.from_dict corruption tests ===


def _valid_node_dict() -> dict[str, object]:
    """Complete valid node checkpoint dict for mutation in corruption tests."""
    return {
        "tokens": [
            {
                "token_id": "tok-001",
                "row_id": "row-001",
                "branch_name": None,
                "fork_group_id": None,
                "join_group_id": None,
                "expand_group_id": None,
                "row_data": {"x": 1},
                "contract_version": "abc123",
                "contract": {"mode": "FLEXIBLE"},
            }
        ],
        "batch_id": "batch-001",
        "elapsed_age_seconds": 0.0,
        "count_fire_offset": None,
        "condition_fire_offset": None,
    }


def test_node_from_dict_round_trip() -> None:
    """Valid dict round-trips through from_dict → to_dict."""
    data = _valid_node_dict()
    node = AggregationNodeCheckpoint.from_dict("node-001", data)
    assert node.batch_id == "batch-001"
    assert node.count_fire_offset is None
    assert node.condition_fire_offset is None
    rt = node.to_dict()
    assert rt["count_fire_offset"] is None
    assert rt["condition_fire_offset"] is None


def test_node_from_dict_missing_count_fire_offset_raises() -> None:
    """Missing count_fire_offset is corruption — must crash, not silently return None."""
    data = _valid_node_dict()
    del data["count_fire_offset"]

    with pytest.raises(AuditIntegrityError, match="count_fire_offset"):
        AggregationNodeCheckpoint.from_dict("node-001", data)


def test_node_from_dict_missing_condition_fire_offset_raises() -> None:
    """Missing condition_fire_offset is corruption — must crash, not silently return None."""
    data = _valid_node_dict()
    del data["condition_fire_offset"]

    with pytest.raises(AuditIntegrityError, match="condition_fire_offset"):
        AggregationNodeCheckpoint.from_dict("node-001", data)


def test_node_from_dict_missing_elapsed_age_raises_with_context() -> None:
    """Missing elapsed_age_seconds should raise AuditIntegrityError with node_id context."""
    data = _valid_node_dict()
    del data["elapsed_age_seconds"]

    with pytest.raises(AuditIntegrityError, match="node-001"):
        AggregationNodeCheckpoint.from_dict("node-001", data)


def test_node_from_dict_multiple_missing_fields_reports_all() -> None:
    """All missing fields reported in one error, not just the first one found."""
    data = _valid_node_dict()
    del data["elapsed_age_seconds"]
    del data["count_fire_offset"]
    del data["condition_fire_offset"]

    with pytest.raises(AuditIntegrityError, match="missing required fields") as exc_info:
        AggregationNodeCheckpoint.from_dict("node-001", data)

    msg = str(exc_info.value)
    assert "elapsed_age_seconds" in msg
    assert "count_fire_offset" in msg
    assert "condition_fire_offset" in msg


# === Aggregation checkpoint type guard tests ===


def test_aggregation_token_rejects_non_dict_row_data() -> None:
    """row_data type guard rejects non-dict values with clear error."""
    with pytest.raises(TypeError, match="row_data must be a Mapping"):
        AggregationTokenCheckpoint(
            token_id="tok-001",
            row_id="row-001",
            branch_name=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            row_data="not a dict",  # type: ignore[arg-type]
            contract_version="abc123",
            contract={},
        )


def test_aggregation_token_rejects_non_dict_contract() -> None:
    """contract type guard rejects non-dict values with clear error."""
    with pytest.raises(TypeError, match="contract must be a Mapping"):
        AggregationTokenCheckpoint(
            token_id="tok-001",
            row_id="row-001",
            branch_name=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            row_data={"value": 1},
            contract_version="abc123",
            contract=["not", "a", "dict"],  # type: ignore[arg-type]
        )


# === JSON round-trip tests (tuple→list regression trap) ===


def test_aggregation_checkpoint_json_round_trip_preserves_tokens_tuple() -> None:
    """Round-trip through JSON must restore tokens as tuples, not lists.

    json.dumps converts tuple → list.  json.loads always returns list.
    from_dict must reconstruct tuples so equality holds.
    This is the real persistence path: to_dict → json.dumps → json.loads → from_dict.
    """
    import json

    state = _make_agg_state()
    serialized = json.dumps(state.to_dict())
    deserialized = json.loads(serialized)
    restored = AggregationCheckpointState.from_dict(deserialized)

    assert restored == state
    # Verify tokens is a tuple, not a list
    node = restored.nodes["node-001"]
    assert isinstance(node.tokens, tuple), f"Expected tuple, got {type(node.tokens).__name__}"
    assert len(node.tokens) == 1
    assert node.tokens[0].token_id == "tok-buf-001"


def test_aggregation_checkpoint_json_round_trip_multiple_nodes_and_tokens() -> None:
    """JSON round-trip with multiple nodes and multiple tokens per node."""
    import json

    state = AggregationCheckpointState(
        version="4.0",
        nodes={
            "node-A": AggregationNodeCheckpoint(
                tokens=(
                    AggregationTokenCheckpoint(
                        token_id="tok-1",
                        row_id="row-1",
                        branch_name="path_a",
                        fork_group_id="fork-1",
                        join_group_id=None,
                        expand_group_id=None,
                        row_data={"nested": {"deep": [1, 2, 3]}},
                        contract_version="v1",
                        contract={"mode": "FLEXIBLE", "locked": False, "version_hash": "v1", "fields": ["a", "b"]},
                    ),
                    AggregationTokenCheckpoint(
                        token_id="tok-2",
                        row_id="row-2",
                        branch_name=None,
                        fork_group_id=None,
                        join_group_id="join-1",
                        expand_group_id="expand-1",
                        row_data={"value": 42},
                        contract_version="v2",
                        contract={"mode": "OBSERVED", "locked": True, "version_hash": "v2", "fields": []},
                    ),
                ),
                batch_id="batch-A",
                elapsed_age_seconds=5.5,
                count_fire_offset=1.0,
                condition_fire_offset=2.5,
            ),
            "node-B": AggregationNodeCheckpoint(
                tokens=(),
                batch_id="batch-B",
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=None,
            ),
        },
    )

    serialized = json.dumps(state.to_dict())
    deserialized = json.loads(serialized)
    restored = AggregationCheckpointState.from_dict(deserialized)

    assert restored == state
    # Verify tokens tuples are restored correctly
    node_a = restored.nodes["node-A"]
    assert isinstance(node_a.tokens, tuple)
    assert len(node_a.tokens) == 2
    node_b = restored.nodes["node-B"]
    assert isinstance(node_b.tokens, tuple)
    assert len(node_b.tokens) == 0


def test_state_from_dict_rejects_unknown_underscore_key() -> None:
    """Regression: only _version is a valid metadata key — others must crash."""
    from elspeth.contracts.errors import AuditIntegrityError

    data = {"_version": 1, "_unknown": "corrupt", "node-A": _valid_node_dict()}
    with pytest.raises(AuditIntegrityError, match="unexpected reserved key '_unknown'"):
        AggregationCheckpointState.from_dict(data)


def test_state_from_dict_accepts_version_key() -> None:
    """_version key is valid metadata and must not raise."""
    data = {"_version": 1, "node-A": _valid_node_dict()}
    state = AggregationCheckpointState.from_dict(data)
    assert "node-A" in state.nodes
