"""Tests for __post_init__ validations on checkpoint types.

Covers: AggregationTokenCheckpoint, AggregationNodeCheckpoint, and
AggregationCheckpointState.
"""

from collections import OrderedDict

import pytest

from elspeth.contracts.aggregation_checkpoint import (
    AggregationCheckpointState,
    AggregationNodeCheckpoint,
    AggregationTokenCheckpoint,
)


class TestAggregationTokenCheckpointPostInit:
    def test_rejects_empty_token_id(self) -> None:
        with pytest.raises(ValueError, match="token_id must not be empty"):
            AggregationTokenCheckpoint(
                token_id="",
                row_id="r1",
                branch_name=None,
                fork_group_id=None,
                join_group_id=None,
                expand_group_id=None,
                row_data={},
                contract_version="v1",
                contract={},
            )

    def test_rejects_empty_row_id(self) -> None:
        with pytest.raises(ValueError, match="row_id must not be empty"):
            AggregationTokenCheckpoint(
                token_id="t1",
                row_id="",
                branch_name=None,
                fork_group_id=None,
                join_group_id=None,
                expand_group_id=None,
                row_data={},
                contract_version="v1",
                contract={},
            )

    def test_rejects_empty_contract_version(self) -> None:
        with pytest.raises(ValueError, match="contract_version must not be empty"):
            AggregationTokenCheckpoint(
                token_id="t1",
                row_id="r1",
                branch_name=None,
                fork_group_id=None,
                join_group_id=None,
                expand_group_id=None,
                row_data={},
                contract_version="",
                contract={},
            )

    def test_accepts_valid(self) -> None:
        t = AggregationTokenCheckpoint(
            token_id="t1",
            row_id="r1",
            branch_name=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            row_data={"x": 1},
            contract_version="v1",
            contract={},
        )
        assert t.token_id == "t1"

    def test_accepts_ordered_dict_mapping(self) -> None:
        """OrderedDict is a Mapping subtype — must be accepted (not just dict)."""
        t = AggregationTokenCheckpoint(
            token_id="t1",
            row_id="r1",
            branch_name=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            row_data=OrderedDict({"x": 1}),
            contract_version="v1",
            contract=OrderedDict({"mode": "observed"}),
        )
        assert t.row_data["x"] == 1

    def test_rejects_non_mapping_row_data(self) -> None:
        with pytest.raises(TypeError, match="must be a Mapping"):
            AggregationTokenCheckpoint(
                token_id="t1",
                row_id="r1",
                branch_name=None,
                fork_group_id=None,
                join_group_id=None,
                expand_group_id=None,
                row_data=[1, 2, 3],  # type: ignore[arg-type]
                contract_version="v1",
                contract={},
            )


class TestAggregationNodeCheckpointPostInit:
    def test_rejects_empty_batch_id_with_tokens(self) -> None:
        """Empty batch_id is invalid when tokens are buffered."""
        token = AggregationTokenCheckpoint(
            token_id="t1",
            row_id="r1",
            branch_name=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            row_data={},
            contract_version="v1",
            contract={},
        )
        with pytest.raises(ValueError, match="batch_id must be set when tokens are buffered"):
            AggregationNodeCheckpoint(
                tokens=(token,),
                batch_id="",
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=None,
                accepted_count_total=1,
                completed_flush_count=0,
            )

    def test_rejects_empty_string_batch_id_without_tokens(self) -> None:
        """Empty-string batch_id is rejected even with no tokens (None is the right sentinel)."""
        with pytest.raises(ValueError, match="batch_id must be a non-empty string when set"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="",
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=None,
                accepted_count_total=0,
                completed_flush_count=0,
            )

    def test_accepts_none_batch_id_when_tokens_empty(self) -> None:
        """Counters-only snapshot is allowed: tokens=() + batch_id=None."""
        n = AggregationNodeCheckpoint(
            tokens=(),
            batch_id=None,
            elapsed_age_seconds=0.0,
            count_fire_offset=None,
            condition_fire_offset=None,
            accepted_count_total=6,
            completed_flush_count=2,
        )
        assert n.batch_id is None
        assert n.accepted_count_total == 6
        assert n.completed_flush_count == 2

    def test_rejects_negative_elapsed_age(self) -> None:
        with pytest.raises(ValueError, match="elapsed_age_seconds must be non-negative"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=-1.0,
                count_fire_offset=None,
                condition_fire_offset=None,
                accepted_count_total=0,
                completed_flush_count=0,
            )

    def test_rejects_nan_elapsed_age(self) -> None:
        with pytest.raises(ValueError, match="elapsed_age_seconds must be non-negative and finite"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=float("nan"),
                count_fire_offset=None,
                condition_fire_offset=None,
                accepted_count_total=0,
                completed_flush_count=0,
            )

    def test_rejects_inf_elapsed_age(self) -> None:
        with pytest.raises(ValueError, match="elapsed_age_seconds must be non-negative and finite"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=float("inf"),
                count_fire_offset=None,
                condition_fire_offset=None,
                accepted_count_total=0,
                completed_flush_count=0,
            )

    def test_rejects_negative_count_fire_offset(self) -> None:
        with pytest.raises(ValueError, match="count_fire_offset must be non-negative and finite"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=0.0,
                count_fire_offset=-1.0,
                condition_fire_offset=None,
                accepted_count_total=0,
                completed_flush_count=0,
            )

    def test_rejects_nan_count_fire_offset(self) -> None:
        with pytest.raises(ValueError, match="count_fire_offset must be non-negative and finite"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=0.0,
                count_fire_offset=float("nan"),
                condition_fire_offset=None,
                accepted_count_total=0,
                completed_flush_count=0,
            )

    def test_rejects_negative_condition_fire_offset(self) -> None:
        with pytest.raises(ValueError, match="condition_fire_offset must be non-negative and finite"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=-2.5,
                accepted_count_total=0,
                completed_flush_count=0,
            )

    def test_rejects_nan_condition_fire_offset(self) -> None:
        with pytest.raises(ValueError, match="condition_fire_offset must be non-negative and finite"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=float("nan"),
                accepted_count_total=0,
                completed_flush_count=0,
            )

    def test_rejects_negative_accepted_count_total(self) -> None:
        with pytest.raises(ValueError, match="accepted_count_total must be non-negative"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=None,
                accepted_count_total=-1,
                completed_flush_count=0,
            )

    def test_rejects_bool_accepted_count_total(self) -> None:
        with pytest.raises(TypeError, match="accepted_count_total must be int"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=None,
                accepted_count_total=True,  # type: ignore[arg-type]
                completed_flush_count=0,
            )

    def test_rejects_negative_completed_flush_count(self) -> None:
        with pytest.raises(ValueError, match="completed_flush_count must be non-negative"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=None,
                accepted_count_total=0,
                completed_flush_count=-1,
            )

    def test_rejects_bool_completed_flush_count(self) -> None:
        with pytest.raises(TypeError, match="completed_flush_count must be int"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id="b1",
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=None,
                accepted_count_total=0,
                completed_flush_count=False,  # type: ignore[arg-type]
            )

    def test_accepts_valid(self) -> None:
        token = AggregationTokenCheckpoint(
            token_id="t1",
            row_id="r1",
            branch_name=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            row_data={},
            contract_version="v1",
            contract={},
        )
        n = AggregationNodeCheckpoint(
            tokens=(token,),
            batch_id="b1",
            elapsed_age_seconds=5.0,
            count_fire_offset=None,
            condition_fire_offset=None,
            accepted_count_total=1,
            completed_flush_count=0,
        )
        assert n.batch_id == "b1"

    def test_accepts_valid_with_fire_offsets(self) -> None:
        """Fire offsets are valid only when tokens are buffered (active batch)."""
        token = AggregationTokenCheckpoint(
            token_id="t1",
            row_id="r1",
            branch_name=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            row_data={},
            contract_version="v1",
            contract={},
        )
        n = AggregationNodeCheckpoint(
            tokens=(token,),
            batch_id="b1",
            elapsed_age_seconds=0.0,
            count_fire_offset=1.5,
            condition_fire_offset=3.0,
            accepted_count_total=1,
            completed_flush_count=0,
        )
        assert n.count_fire_offset == 1.5
        assert n.condition_fire_offset == 3.0

    # --- Counter-only invariant (Tier 1: no stale trigger state) ---

    def test_rejects_counter_only_with_count_fire_offset(self) -> None:
        """tokens=() must have count_fire_offset=None — no in-flight trigger to preserve."""
        with pytest.raises(ValueError, match="count_fire_offset must be None when tokens is empty"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id=None,
                elapsed_age_seconds=0.0,
                count_fire_offset=2.5,
                condition_fire_offset=None,
                accepted_count_total=3,
                completed_flush_count=1,
            )

    def test_rejects_counter_only_with_condition_fire_offset(self) -> None:
        """tokens=() must have condition_fire_offset=None — no in-flight trigger to preserve."""
        with pytest.raises(ValueError, match="condition_fire_offset must be None when tokens is empty"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id=None,
                elapsed_age_seconds=0.0,
                count_fire_offset=None,
                condition_fire_offset=4.0,
                accepted_count_total=3,
                completed_flush_count=1,
            )

    def test_rejects_counter_only_with_nonzero_elapsed_age(self) -> None:
        """tokens=() must have elapsed_age_seconds=0.0 — no in-flight age to preserve."""
        with pytest.raises(ValueError, match=r"elapsed_age_seconds must be 0\.0 when tokens is empty"):
            AggregationNodeCheckpoint(
                tokens=(),
                batch_id=None,
                elapsed_age_seconds=12.5,
                count_fire_offset=None,
                condition_fire_offset=None,
                accepted_count_total=3,
                completed_flush_count=1,
            )

    def test_accepts_counter_only_snapshot(self) -> None:
        """Valid counter-only snapshot: empty tokens, no batch_id, no trigger state."""
        n = AggregationNodeCheckpoint(
            tokens=(),
            batch_id=None,
            elapsed_age_seconds=0.0,
            count_fire_offset=None,
            condition_fire_offset=None,
            accepted_count_total=9,
            completed_flush_count=3,
        )
        assert n.tokens == ()
        assert n.batch_id is None
        assert n.accepted_count_total == 9
        assert n.completed_flush_count == 3


class TestAggregationCheckpointStatePostInit:
    def test_rejects_empty_version(self) -> None:
        with pytest.raises(ValueError, match="version must not be empty"):
            AggregationCheckpointState(version="", nodes={})

    # --- Type guard on nodes (elspeth-50f4f87787) ---

    def test_rejects_list_as_nodes(self) -> None:
        """Regression: non-mapping type must raise TypeError, not unhelpful MappingProxyType error."""
        with pytest.raises(TypeError, match="nodes must be a Mapping"):
            AggregationCheckpointState(version="4.0", nodes=[])  # type: ignore[arg-type]

    def test_rejects_string_as_nodes(self) -> None:
        with pytest.raises(TypeError, match="nodes must be a Mapping"):
            AggregationCheckpointState(version="4.0", nodes="not-a-dict")  # type: ignore[arg-type]

    def test_rejects_none_as_nodes(self) -> None:
        with pytest.raises(TypeError, match="nodes must be a Mapping"):
            AggregationCheckpointState(version="4.0", nodes=None)  # type: ignore[arg-type]

    def test_accepts_dict_and_wraps_to_mapping_proxy(self) -> None:
        """Valid dict is accepted and wrapped to MappingProxyType."""
        from types import MappingProxyType

        state = AggregationCheckpointState(version="4.0", nodes={})
        assert isinstance(state.nodes, MappingProxyType)


class TestAggregationNodeCheckpointTokensFreeze:
    """tokens field must be deeply frozen on direct construction."""

    def test_tokens_list_frozen_to_tuple(self) -> None:
        from elspeth.contracts.aggregation_checkpoint import (
            AggregationNodeCheckpoint,
            AggregationTokenCheckpoint,
        )

        token = AggregationTokenCheckpoint(
            token_id="t1",
            row_id="r1",
            branch_name="main",
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            row_data={"value": 42},
            contract_version="v1",
            contract={"mode": "observed"},
        )
        tokens_list = [token]
        node = AggregationNodeCheckpoint(
            tokens=tokens_list,  # type: ignore[arg-type]
            batch_id="b1",
            elapsed_age_seconds=1.0,
            count_fire_offset=None,
            condition_fire_offset=None,
            accepted_count_total=1,
            completed_flush_count=0,
        )
        tokens_list.append(token)
        assert isinstance(node.tokens, tuple)
        assert len(node.tokens) == 1
