# tests/unit/engine/orchestrator/test_checkpoint_leader_fence_guard.py
"""CheckpointCoordinator fail-closed leader-token guard (elspeth-fab455790d).

ADR-030 defense-in-depth: every orchestrator checkpoint create/delete must
run under a bound leader token whose ``run_id`` matches the run being
written. A missing or foreign-run token is an invariant violation and must
crash BEFORE any manager call — never fall through to CheckpointManager's
unfenced plain-write arm (that arm is a deliberate seam for direct
repository/test/tooling callers only, not the coordinator runtime path).

Ordering pins:
- The guard runs AFTER ``_checkpoint_gate``: disabled/unconfigured
  checkpointing still short-circuits token-free (a no-checkpoint run needs
  no leader token).
- In ``maybe_checkpoint`` the guard runs BEFORE the sequence increment, so
  every-N runs fail closed even on rows the frequency gate would skip.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from elspeth.contracts.barrier_scalars import BarrierScalars
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.checkpoint import CheckpointManager
from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator

RUN_ID = "run-fenced"


def _token(run_id: str = RUN_ID) -> CoordinationToken:
    return CoordinationToken(run_id=run_id, worker_id="leader-a", leader_epoch=1)


def _make_coordinator(
    *,
    manager: Mock | None = ...,  # type: ignore[assignment]
    enabled: bool = True,
    frequency: int = 1,
) -> tuple[CheckpointCoordinator, Mock | None]:
    resolved_manager: Mock | None = Mock(spec=CheckpointManager) if manager is ... else manager
    config = Mock(spec=RuntimeCheckpointConfig)
    config.enabled = enabled
    config.frequency = frequency
    coordinator = CheckpointCoordinator(checkpoint_manager=resolved_manager, checkpoint_config=config)
    coordinator.set_active_graph(SimpleNamespace())
    return coordinator, resolved_manager


def _loop_ctx() -> SimpleNamespace:
    processor = SimpleNamespace(get_barrier_scalars=lambda: BarrierScalars(aggregation={}, coalesce={}))
    return SimpleNamespace(processor=processor)


def _fire(coordinator: CheckpointCoordinator, path: str) -> None:
    if path == "run_start":
        coordinator.checkpoint_run_start(RUN_ID)
    elif path == "maybe":
        coordinator.maybe_checkpoint(RUN_ID, barrier_scalars=None)
    elif path == "interrupted":
        coordinator.checkpoint_interrupted_progress(RUN_ID, _loop_ctx())  # type: ignore[arg-type]
    else:
        raise AssertionError(f"unknown path {path!r}")


CREATE_PATHS = ["run_start", "maybe", "interrupted"]


class TestCreatePathsFailClosed:
    @pytest.mark.parametrize("path", CREATE_PATHS)
    def test_missing_token_raises_before_any_manager_write(self, path: str) -> None:
        coordinator, manager = _make_coordinator()
        assert manager is not None
        with pytest.raises(OrchestrationInvariantError, match="no bound leader token"):
            _fire(coordinator, path)
        manager.create_checkpoint.assert_not_called()

    @pytest.mark.parametrize("path", CREATE_PATHS)
    def test_foreign_run_token_raises_before_any_manager_write(self, path: str) -> None:
        coordinator, manager = _make_coordinator()
        assert manager is not None
        coordinator.bind_coordination(_token(run_id="some-other-run"))
        with pytest.raises(OrchestrationInvariantError, match="some-other-run"):
            _fire(coordinator, path)
        manager.create_checkpoint.assert_not_called()

    def test_maybe_checkpoint_guard_precedes_frequency_skip(self) -> None:
        """Every-N runs fail closed even on rows the frequency gate would skip."""
        coordinator, manager = _make_coordinator(frequency=1000)
        assert manager is not None
        with pytest.raises(OrchestrationInvariantError, match="no bound leader token"):
            coordinator.maybe_checkpoint(RUN_ID, barrier_scalars=None)
        manager.create_checkpoint.assert_not_called()

    @pytest.mark.parametrize("path", CREATE_PATHS)
    def test_disabled_checkpointing_short_circuits_token_free(self, path: str) -> None:
        """The gate runs first: a no-checkpoint run never needs a leader token."""
        coordinator, manager = _make_coordinator(enabled=False)
        assert manager is not None
        _fire(coordinator, path)  # must not raise
        manager.create_checkpoint.assert_not_called()

    @pytest.mark.parametrize("path", CREATE_PATHS)
    def test_unconfigured_manager_short_circuits_token_free(self, path: str) -> None:
        coordinator, _manager = _make_coordinator(manager=None)
        _fire(coordinator, path)  # must not raise

    @pytest.mark.parametrize("path", CREATE_PATHS)
    def test_matching_token_writes_with_the_bound_token(self, path: str) -> None:
        coordinator, manager = _make_coordinator()
        assert manager is not None
        token = _token()
        coordinator.bind_coordination(token)
        _fire(coordinator, path)
        assert manager.create_checkpoint.call_count == 1
        assert manager.create_checkpoint.call_args.kwargs["coordination_token"] is token
        assert manager.create_checkpoint.call_args.kwargs["run_id"] == RUN_ID


class TestDeleteFailsClosed:
    def test_missing_token_raises(self) -> None:
        coordinator, manager = _make_coordinator()
        assert manager is not None
        with pytest.raises(OrchestrationInvariantError, match="no bound leader token"):
            coordinator.delete_checkpoints(RUN_ID)
        manager.delete_checkpoints.assert_not_called()

    def test_foreign_run_token_raises(self) -> None:
        coordinator, manager = _make_coordinator()
        assert manager is not None
        coordinator.bind_coordination(_token(run_id="some-other-run"))
        with pytest.raises(OrchestrationInvariantError, match="some-other-run"):
            coordinator.delete_checkpoints(RUN_ID)
        manager.delete_checkpoints.assert_not_called()

    def test_manager_none_returns_silently_without_token(self) -> None:
        """delete has no _checkpoint_gate by design; the manager-None arm stays."""
        coordinator, _manager = _make_coordinator(manager=None)
        coordinator.delete_checkpoints(RUN_ID)  # must not raise

    def test_matching_token_deletes_with_the_bound_token(self) -> None:
        coordinator, manager = _make_coordinator()
        assert manager is not None
        token = _token()
        coordinator.bind_coordination(token)
        coordinator.delete_checkpoints(RUN_ID)
        assert manager.delete_checkpoints.call_count == 1
        assert manager.delete_checkpoints.call_args.kwargs["coordination_token"] is token
