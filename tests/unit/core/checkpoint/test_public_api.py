"""Tests for the checkpoint package facade."""

from __future__ import annotations


def test_checkpoint_facade_exports_recovery_companion_api() -> None:
    from elspeth.core import checkpoint
    from elspeth.core.checkpoint.recovery import (
        IncompleteTokenSpec,
        NonResumableRunError,
        check_run_status_resumable,
    )

    assert checkpoint.IncompleteTokenSpec is IncompleteTokenSpec
    assert checkpoint.NonResumableRunError is NonResumableRunError
    assert checkpoint.check_run_status_resumable is check_run_status_resumable
    assert "IncompleteTokenSpec" in checkpoint.__all__
    assert "NonResumableRunError" in checkpoint.__all__
    assert "check_run_status_resumable" in checkpoint.__all__
