"""Tests for __post_init__ validations on engine types.

Covers: _FlushContext coalesce pairing, TriggerEvaluator.restore_from_checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from elspeth.contracts import TokenInfo
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.config import AggregationSettings
from elspeth.testing import make_pipeline_row

if TYPE_CHECKING:
    from elspeth.engine.triggers import TriggerEvaluator


@dataclass(frozen=True, slots=True)
class _TransformFake:
    name: str = "test_transform"


class TestFlushContextPostInit:
    """Tests for _FlushContext.__post_init__ validation."""

    def _make_token(self) -> TokenInfo:
        return TokenInfo(row_id="r1", token_id="t1", row_data=make_pipeline_row({}))

    def _make_transform(self) -> _TransformFake:
        return _TransformFake()

    def _make_settings(self) -> AggregationSettings:
        return AggregationSettings(name="agg", plugin="test_transform", input="source", on_error="discard")

    def test_rejects_empty_node_id(self) -> None:
        from elspeth.engine.processor import _FlushContext

        with pytest.raises(ValueError, match="node_id must not be empty"):
            _FlushContext(
                node_id=NodeID(""),
                transform=self._make_transform(),
                settings=self._make_settings(),
                buffered_tokens=(self._make_token(),),
                batch_id="b1",
                error_msg="test",
                expand_parent_token=self._make_token(),
                triggering_token=None,
                coalesce_node_id=None,
                coalesce_name=None,
            )

    def test_rejects_empty_buffered_tokens(self) -> None:
        from elspeth.engine.processor import _FlushContext

        with pytest.raises(ValueError, match="buffered_tokens must not be empty"):
            _FlushContext(
                node_id=NodeID("n1"),
                transform=self._make_transform(),
                settings=self._make_settings(),
                buffered_tokens=(),
                batch_id="b1",
                error_msg="test",
                expand_parent_token=self._make_token(),
                triggering_token=None,
                coalesce_node_id=None,
                coalesce_name=None,
            )

    def test_rejects_empty_batch_id(self) -> None:
        from elspeth.engine.processor import _FlushContext

        with pytest.raises(ValueError, match="batch_id must not be empty"):
            _FlushContext(
                node_id=NodeID("n1"),
                transform=self._make_transform(),
                settings=self._make_settings(),
                buffered_tokens=(self._make_token(),),
                batch_id="",
                error_msg="test",
                expand_parent_token=self._make_token(),
                triggering_token=None,
                coalesce_node_id=None,
                coalesce_name=None,
            )

    def test_rejects_mismatched_coalesce_id_without_name(self) -> None:
        from elspeth.engine.processor import _FlushContext

        with pytest.raises(ValueError, match="coalesce_node_id and coalesce_name must be both set or both None"):
            _FlushContext(
                node_id=NodeID("n1"),
                transform=self._make_transform(),
                settings=self._make_settings(),
                buffered_tokens=(self._make_token(),),
                batch_id="b1",
                error_msg="test",
                expand_parent_token=self._make_token(),
                triggering_token=None,
                coalesce_node_id=NodeID("c1"),
                coalesce_name=None,
            )

    def test_rejects_mismatched_coalesce_name_without_id(self) -> None:
        from elspeth.engine.processor import _FlushContext

        with pytest.raises(ValueError, match="coalesce_node_id and coalesce_name must be both set or both None"):
            _FlushContext(
                node_id=NodeID("n1"),
                transform=self._make_transform(),
                settings=self._make_settings(),
                buffered_tokens=(self._make_token(),),
                batch_id="b1",
                error_msg="test",
                expand_parent_token=self._make_token(),
                triggering_token=None,
                coalesce_node_id=None,
                coalesce_name=CoalesceName("merge1"),
            )

    def test_accepts_both_coalesce_none(self) -> None:
        from elspeth.engine.processor import _FlushContext

        ctx = _FlushContext(
            node_id=NodeID("n1"),
            transform=self._make_transform(),
            settings=self._make_settings(),
            buffered_tokens=(self._make_token(),),
            batch_id="b1",
            error_msg="test",
            expand_parent_token=self._make_token(),
            triggering_token=None,
            coalesce_node_id=None,
            coalesce_name=None,
        )
        assert ctx.coalesce_node_id is None

    def test_accepts_both_coalesce_set(self) -> None:
        from elspeth.engine.processor import _FlushContext

        ctx = _FlushContext(
            node_id=NodeID("n1"),
            transform=self._make_transform(),
            settings=self._make_settings(),
            buffered_tokens=(self._make_token(),),
            batch_id="b1",
            error_msg="test",
            expand_parent_token=self._make_token(),
            triggering_token=None,
            coalesce_node_id=NodeID("c1"),
            coalesce_name=CoalesceName("merge1"),
        )
        assert ctx.coalesce_node_id == "c1"


class TestTriggerEvaluatorRestoreValidation:
    """Tests for TriggerEvaluator.restore_from_checkpoint input validation."""

    def _make_evaluator(self) -> TriggerEvaluator:
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator

        config = TriggerConfig(count=5)
        return TriggerEvaluator(config)

    def test_rejects_negative_batch_count(self) -> None:
        evaluator = self._make_evaluator()
        with pytest.raises(ValueError, match="batch_count must be non-negative"):
            evaluator.restore_from_checkpoint(batch_count=-1, elapsed_age_seconds=0.0, count_fire_offset=None, condition_fire_offset=None)

    def test_rejects_negative_elapsed_age(self) -> None:
        evaluator = self._make_evaluator()
        with pytest.raises(ValueError, match="elapsed_age_seconds must be non-negative"):
            evaluator.restore_from_checkpoint(batch_count=0, elapsed_age_seconds=-1.0, count_fire_offset=None, condition_fire_offset=None)

    def test_rejects_nan_elapsed_age(self) -> None:
        evaluator = self._make_evaluator()
        with pytest.raises(ValueError, match="elapsed_age_seconds must be non-negative and finite"):
            evaluator.restore_from_checkpoint(
                batch_count=0, elapsed_age_seconds=float("nan"), count_fire_offset=None, condition_fire_offset=None
            )

    def test_rejects_inf_count_fire_offset(self) -> None:
        evaluator = self._make_evaluator()
        with pytest.raises(ValueError, match="count_fire_offset must be non-negative and finite"):
            evaluator.restore_from_checkpoint(
                batch_count=0, elapsed_age_seconds=0.0, count_fire_offset=float("inf"), condition_fire_offset=None
            )

    def test_rejects_negative_condition_fire_offset(self) -> None:
        evaluator = self._make_evaluator()
        with pytest.raises(ValueError, match="condition_fire_offset must be non-negative and finite"):
            evaluator.restore_from_checkpoint(batch_count=0, elapsed_age_seconds=0.0, count_fire_offset=None, condition_fire_offset=-1.0)

    def test_rejects_nan_condition_fire_offset(self) -> None:
        evaluator = self._make_evaluator()
        with pytest.raises(ValueError, match="condition_fire_offset must be non-negative and finite"):
            evaluator.restore_from_checkpoint(
                batch_count=0, elapsed_age_seconds=0.0, count_fire_offset=None, condition_fire_offset=float("nan")
            )

    def test_rejects_inf_condition_fire_offset(self) -> None:
        evaluator = self._make_evaluator()
        with pytest.raises(ValueError, match="condition_fire_offset must be non-negative and finite"):
            evaluator.restore_from_checkpoint(
                batch_count=0, elapsed_age_seconds=0.0, count_fire_offset=None, condition_fire_offset=float("inf")
            )

    def test_accepts_valid(self) -> None:
        evaluator = self._make_evaluator()
        evaluator.restore_from_checkpoint(batch_count=3, elapsed_age_seconds=5.0, count_fire_offset=2.0, condition_fire_offset=None)
        assert evaluator._batch_count == 3

    def test_accepts_valid_with_condition_fire_offset(self) -> None:
        evaluator = self._make_evaluator()
        evaluator.restore_from_checkpoint(batch_count=0, elapsed_age_seconds=0.0, count_fire_offset=None, condition_fire_offset=1.5)
        assert evaluator.get_condition_fire_offset() == 1.5
