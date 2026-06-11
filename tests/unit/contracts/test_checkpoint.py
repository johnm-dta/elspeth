"""Unit tests for checkpoint/recovery domain contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts import Checkpoint, ResumeCheck, ResumePoint
from elspeth.contracts.barrier_scalars import AggregationNodeScalars, BarrierScalars


def _checkpoint() -> Checkpoint:
    return Checkpoint(
        checkpoint_id="cp-001",
        run_id="run-001",
        sequence_number=1,
        created_at=datetime.now(UTC),
        upstream_topology_hash="a" * 64,
        format_version=Checkpoint.CURRENT_FORMAT_VERSION,
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


def test_resume_point_rejects_negative_sequence_number() -> None:
    with pytest.raises(ValueError, match="sequence_number must be >= 0"):
        ResumePoint(
            checkpoint=_checkpoint(),
            sequence_number=-1,
        )


# === ResumePoint Tier 1 type guards (elspeth-0b184125ca) ===


def test_resume_point_rejects_dict_barrier_scalars() -> None:
    """Regression: elspeth-0b184125ca — raw dict must not be accepted as state."""
    with pytest.raises(TypeError, match="barrier_scalars must be BarrierScalars"):
        ResumePoint(
            checkpoint=_checkpoint(),
            sequence_number=1,
            barrier_scalars={"_version": "1.0", "aggregation": {}, "coalesce": []},  # type: ignore[arg-type]
        )


def test_resume_point_accepts_barrier_scalars() -> None:
    """F1: ResumePoint carries scalar barrier metadata, not buffer blobs."""
    scalars = BarrierScalars(
        aggregation={"agg-1": AggregationNodeScalars(count_fire_offset=1.5, condition_fire_offset=None)},
        coalesce={},
    )
    point = ResumePoint(checkpoint=_checkpoint(), sequence_number=1, barrier_scalars=scalars)
    assert point.barrier_scalars is scalars


# === ResumePoint checkpoint type guard + sequence_number type guard ===
# (elspeth-dce3a343a7, elspeth-52a31594ee)


def test_resume_point_rejects_non_checkpoint_type() -> None:
    """Regression: elspeth-dce3a343a7 — checkpoint must be Checkpoint, not raw dict."""
    with pytest.raises(TypeError, match="checkpoint must be Checkpoint"):
        ResumePoint(
            checkpoint={"run_id": "r1"},  # type: ignore[arg-type]
            sequence_number=1,
        )


def test_resume_point_rejects_none_checkpoint() -> None:
    """None checkpoint is corruption — crash, don't propagate."""
    with pytest.raises(TypeError, match="checkpoint must be Checkpoint"):
        ResumePoint(
            checkpoint=None,  # type: ignore[arg-type]
            sequence_number=1,
        )


def test_resume_point_rejects_float_sequence_number() -> None:
    """Regression: elspeth-52a31594ee — float 0.5 must not pass as sequence number."""
    with pytest.raises(TypeError, match="sequence_number must be int"):
        ResumePoint(
            checkpoint=_checkpoint(),
            sequence_number=0.5,  # type: ignore[arg-type]
        )


def test_resume_point_rejects_bool_sequence_number() -> None:
    """bool is subclass of int — True (value 1) must not pass as sequence number."""
    with pytest.raises(TypeError, match="sequence_number must be int"):
        ResumePoint(
            checkpoint=_checkpoint(),
            sequence_number=True,
        )


def test_resume_point_rejects_string_sequence_number() -> None:
    """String sequence number is corruption."""
    with pytest.raises(TypeError, match="sequence_number must be int"):
        ResumePoint(
            checkpoint=_checkpoint(),
            sequence_number="3",  # type: ignore[arg-type]
        )
