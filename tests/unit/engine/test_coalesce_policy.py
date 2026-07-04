"""Unit tests for the coalesce policy matrix (engine/coalesce_policy.py).

Pins the full policy x event decision table extracted from CoalesceExecutor
(elspeth-2d43291212), including byte-exact failure-reason strings — they are
audit-load-bearing (compute_error_hash -> Landscape serialization) — and the
mutation boundaries previously covered via CoalesceExecutor._should_merge
(TestShouldMergeMutationGaps in test_coalesce_executor.py).

Pure-function tests: no executor construction, no mocks, no I/O.
"""

import pytest

from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.config import CoalesceSettings
from elspeth.engine.coalesce_policy import (
    CoalesceAction,
    CoalesceDecision,
    CoalesceEvent,
    decide_coalesce,
    require_quorum_count,
)


def _settings(
    name: str = "merge",
    branches: list[str] | None = None,
    policy: str = "require_all",
    timeout_seconds: float | None = None,
    quorum_count: int | None = None,
) -> CoalesceSettings:
    """Shorthand for building CoalesceSettings (mirrors test_coalesce_executor.py)."""
    if branches is None:
        branches = ["a", "b", "c"]
    if policy == "best_effort" and timeout_seconds is None:
        timeout_seconds = 60.0  # config validator: best_effort requires a timeout
    return CoalesceSettings(
        name=name,
        branches=branches,
        policy=policy,
        timeout_seconds=timeout_seconds,
        quorum_count=quorum_count,
    )


def _lost(*names: str) -> dict[str, str]:
    return dict.fromkeys(names, "error_routed")


MERGE = CoalesceAction.MERGE
FAIL = CoalesceAction.FAIL
WAIT = CoalesceAction.WAIT

ARRIVAL = CoalesceEvent.ARRIVAL
TIMEOUT = CoalesceEvent.TIMEOUT
FLUSH = CoalesceEvent.FLUSH
LOSS = CoalesceEvent.LOSS


# ===========================================================================
# Full decision matrix — policy x event x (arrived, lost) with exact actions
# and byte-exact failure reasons. All cases use 3 branches (a, b, c);
# quorum cases use quorum_count=2 unless stated.
# ===========================================================================

MATRIX_CASES = [
    # (id, policy, quorum_count, event, arrived, lost_names, action, reason)
    # --- require_all ---
    ("require_all-arrival-complete", "require_all", None, ARRIVAL, 3, (), MERGE, None),
    ("require_all-arrival-partial", "require_all", None, ARRIVAL, 2, (), WAIT, None),
    ("require_all-arrival-loss-does-not-count", "require_all", None, ARRIVAL, 2, ("c",), WAIT, None),
    ("require_all-timeout", "require_all", None, TIMEOUT, 2, (), FAIL, "incomplete_branches"),
    ("require_all-flush", "require_all", None, FLUSH, 2, (), FAIL, "incomplete_branches"),
    ("require_all-flush-zero", "require_all", None, FLUSH, 0, ("a",), FAIL, "incomplete_branches"),
    ("require_all-loss-single", "require_all", None, LOSS, 2, ("c",), FAIL, "branch_lost:c"),
    # --- first ---
    ("first-arrival-one", "first", None, ARRIVAL, 1, (), MERGE, None),
    ("first-arrival-zero", "first", None, ARRIVAL, 0, (), WAIT, None),
    ("first-timeout-zero", "first", None, TIMEOUT, 0, ("a",), FAIL, "first_timeout_no_arrivals"),
    ("first-flush-zero", "first", None, FLUSH, 0, ("a",), FAIL, "all_branches_lost"),
    ("first-loss-all-lost", "first", None, LOSS, 0, ("a", "b", "c"), FAIL, "all_branches_lost"),
    ("first-loss-some-lost", "first", None, LOSS, 0, ("a", "b"), WAIT, None),
    ("first-loss-arrived-waits-not-crashes", "first", None, LOSS, 1, ("a", "b"), WAIT, None),
    # --- quorum (need 2 of 3) ---
    ("quorum-arrival-met", "quorum", 2, ARRIVAL, 2, (), MERGE, None),
    ("quorum-arrival-exceeded", "quorum", 2, ARRIVAL, 3, (), MERGE, None),
    ("quorum-arrival-below", "quorum", 2, ARRIVAL, 1, (), WAIT, None),
    ("quorum-timeout-met", "quorum", 2, TIMEOUT, 2, (), MERGE, None),
    ("quorum-timeout-below", "quorum", 2, TIMEOUT, 1, (), FAIL, "quorum_not_met_at_timeout"),
    ("quorum-flush-met", "quorum", 2, FLUSH, 2, (), MERGE, None),
    ("quorum-flush-below", "quorum", 2, FLUSH, 1, (), FAIL, "quorum_not_met"),
    ("quorum-loss-impossible", "quorum", 2, LOSS, 1, ("b", "c"), FAIL, "quorum_impossible:need=2,max_possible=1"),
    ("quorum-loss-already-met", "quorum", 2, LOSS, 2, ("c",), MERGE, None),
    ("quorum-loss-still-possible", "quorum", 2, LOSS, 1, ("c",), WAIT, None),
    # --- best_effort ---
    ("best_effort-arrival-all-arrived", "best_effort", None, ARRIVAL, 3, (), MERGE, None),
    ("best_effort-arrival-accounted-with-loss", "best_effort", None, ARRIVAL, 2, ("c",), MERGE, None),
    ("best_effort-arrival-unaccounted", "best_effort", None, ARRIVAL, 2, (), WAIT, None),
    ("best_effort-arrival-partial-loss-unaccounted", "best_effort", None, ARRIVAL, 1, ("c",), WAIT, None),
    ("best_effort-timeout-some-arrived", "best_effort", None, TIMEOUT, 1, (), MERGE, None),
    ("best_effort-timeout-none-arrived", "best_effort", None, TIMEOUT, 0, ("a",), FAIL, "best_effort_timeout_no_arrivals"),
    ("best_effort-flush-some-arrived", "best_effort", None, FLUSH, 1, (), MERGE, None),
    ("best_effort-flush-none-arrived", "best_effort", None, FLUSH, 0, ("a",), FAIL, "all_branches_lost"),
    ("best_effort-loss-accounted-with-arrivals", "best_effort", None, LOSS, 2, ("c",), MERGE, None),
    ("best_effort-loss-all-lost", "best_effort", None, LOSS, 0, ("a", "b", "c"), FAIL, "all_branches_lost"),
    ("best_effort-loss-unaccounted", "best_effort", None, LOSS, 1, ("c",), WAIT, None),
]


@pytest.mark.parametrize(
    ("policy", "quorum_count", "event", "arrived", "lost_names", "action", "reason"),
    [case[1:] for case in MATRIX_CASES],
    ids=[case[0] for case in MATRIX_CASES],
)
def test_decision_matrix(
    policy: str,
    quorum_count: int | None,
    event: CoalesceEvent,
    arrived: int,
    lost_names: tuple[str, ...],
    action: CoalesceAction,
    reason: str | None,
) -> None:
    settings = _settings(policy=policy, quorum_count=quorum_count)
    decision = decide_coalesce(
        settings,
        event,
        arrived_count=arrived,
        lost_branches=_lost(*lost_names),
        row_id="row_1",
    )
    assert decision.action is action
    assert decision.failure_reason == reason


def test_require_all_loss_reason_sorts_branch_names() -> None:
    """branch_lost reason lists lost branches sorted, regardless of insert order."""
    settings = _settings(policy="require_all")
    decision = decide_coalesce(
        settings,
        LOSS,
        arrived_count=1,
        lost_branches={"c": "error_routed", "a": "error_routed"},  # unsorted input
        row_id="row_1",
    )
    assert decision.action is FAIL
    assert decision.failure_reason == "branch_lost:a,c"


# ===========================================================================
# Mutation-boundary pins (mirroring TestShouldMergeMutationGaps for the
# extracted matrix — the mutants now live in coalesce_policy.py).
# ===========================================================================


class TestMutationBoundaries:
    def test_first_policy_does_not_fire_at_zero_arrivals(self) -> None:
        """Kill mutant: ``arrived_count >= 1`` -> ``>= 0`` (empty merged token)."""
        decision = decide_coalesce(_settings(policy="first"), ARRIVAL, arrived_count=0, lost_branches={})
        assert decision.action is WAIT

    def test_first_policy_fires_at_exactly_one_arrival(self) -> None:
        decision = decide_coalesce(_settings(policy="first"), ARRIVAL, arrived_count=1, lost_branches={})
        assert decision.action is MERGE

    def test_best_effort_lost_branches_add_not_subtract(self) -> None:
        """Kill mutant: ``arrived + lost`` -> ``arrived - lost`` (barrier stuck forever)."""
        decision = decide_coalesce(
            _settings(policy="best_effort"),
            ARRIVAL,
            arrived_count=2,
            lost_branches=_lost("c"),
        )
        # Correct: 2 + 1 = 3 >= 3 -> MERGE.  Mutant: 2 - 1 = 1 >= 3 -> WAIT.
        assert decision.action is MERGE

    def test_quorum_fires_when_arrivals_exceed_quorum_count(self) -> None:
        """Kill mutant: ``arrived >= quorum`` -> ``==`` (3 of quorum-2 never merges)."""
        decision = decide_coalesce(
            _settings(policy="quorum", quorum_count=2),
            ARRIVAL,
            arrived_count=3,
            lost_branches={},
        )
        assert decision.action is MERGE

    def test_quorum_impossibility_boundary(self) -> None:
        """max_possible == quorum is still possible (``<`` not ``<=``)."""
        decision = decide_coalesce(
            _settings(policy="quorum", quorum_count=2),
            LOSS,
            arrived_count=1,
            lost_branches=_lost("c"),  # max_possible = 3 - 1 = 2, quorum = 2
        )
        assert decision.action is WAIT


# ===========================================================================
# Structural guarantees the executor delegates rely on.
# ===========================================================================


_GRID = tuple(
    (arrived, lost_names)
    for arrived in (0, 1, 2, 3)
    for lost_names in ((), ("c",), ("b", "c"), ("a", "b", "c"))
    if arrived + len(lost_names) <= 3
)


class TestStructuralGuarantees:
    def _all_settings(self) -> list[CoalesceSettings]:
        return [
            _settings(policy="require_all"),
            _settings(policy="first"),
            _settings(policy="quorum", quorum_count=2),
            _settings(policy="best_effort"),
        ]

    def test_arrival_never_fails(self) -> None:
        for settings in self._all_settings():
            for arrived, lost_names in _GRID:
                decision = decide_coalesce(
                    settings,
                    ARRIVAL,
                    arrived_count=arrived,
                    lost_branches=_lost(*lost_names),
                )
                assert decision.action is not FAIL, (settings.policy, arrived, lost_names)

    def test_timeout_and_flush_never_wait(self) -> None:
        for settings in self._all_settings():
            for event in (TIMEOUT, FLUSH):
                for arrived, lost_names in _GRID:
                    if settings.policy == "first" and arrived > 0:
                        continue  # invariant crash, covered separately
                    decision = decide_coalesce(
                        settings,
                        event,
                        arrived_count=arrived,
                        lost_branches=_lost(*lost_names),
                        row_id="row_1",
                    )
                    assert decision.action is not WAIT, (settings.policy, event, arrived, lost_names)

    def test_fail_decisions_always_carry_a_reason(self) -> None:
        for settings in self._all_settings():
            for event in (ARRIVAL, TIMEOUT, FLUSH, LOSS):
                for arrived, lost_names in _GRID:
                    if settings.policy == "first" and event in (TIMEOUT, FLUSH) and arrived > 0:
                        continue  # invariant crash, covered separately
                    decision = decide_coalesce(
                        settings,
                        event,
                        arrived_count=arrived,
                        lost_branches=_lost(*lost_names),
                        row_id="row_1",
                    )
                    if decision.action is FAIL:
                        assert decision.require_failure_reason() == decision.failure_reason
                    else:
                        assert decision.failure_reason is None


# ===========================================================================
# Invariant crashes.
# ===========================================================================


class TestInvariantCrashes:
    @pytest.mark.parametrize("event", [TIMEOUT, FLUSH], ids=["timeout", "flush"])
    def test_first_with_arrivals_at_resolution_crashes(self, event: CoalesceEvent) -> None:
        """'first' merges on arrival — an arrived pending branch at resolution is a bug."""
        with pytest.raises(RuntimeError, match=r"'first' policy should never have arrived pending branches") as exc_info:
            decide_coalesce(
                _settings(name="my_merge", policy="first"),
                event,
                arrived_count=1,
                lost_branches={},
                row_id="row_42",
            )
        assert "my_merge" in str(exc_info.value)
        assert "row_id='row_42'" in str(exc_info.value)

    @pytest.mark.parametrize("event", [ARRIVAL, TIMEOUT, FLUSH, LOSS], ids=["arrival", "timeout", "flush", "loss"])
    def test_quorum_without_quorum_count_crashes(self, event: CoalesceEvent) -> None:
        """quorum_count=None reaching the evaluator is a config validation bug."""
        settings = CoalesceSettings.model_construct(
            name="merge",
            branches={"a": "a", "b": "b", "c": "c"},
            policy="quorum",
            quorum_count=None,
        )
        with pytest.raises(RuntimeError, match="quorum_count is None for quorum policy"):
            decide_coalesce(settings, event, arrived_count=1, lost_branches={}, row_id="row_1")

    def test_require_quorum_count_returns_value(self) -> None:
        assert require_quorum_count(_settings(policy="quorum", quorum_count=2)) == 2

    @pytest.mark.parametrize("event", [ARRIVAL, TIMEOUT, FLUSH, LOSS], ids=["arrival", "timeout", "flush", "loss"])
    def test_unknown_policy_crashes(self, event: CoalesceEvent) -> None:
        settings = CoalesceSettings.model_construct(
            name="merge",
            branches={"a": "a", "b": "b"},
            # Deliberately out-of-vocabulary: pins the fail-closed unknown-policy arm.
            policy="bogus",  # type: ignore[arg-type]
            quorum_count=None,
        )
        with pytest.raises(RuntimeError, match="Unknown coalesce policy: 'bogus'"):
            decide_coalesce(settings, event, arrived_count=1, lost_branches={}, row_id="row_1")

    def test_non_quorum_policy_never_reads_quorum_count(self) -> None:
        """require_quorum_count must not be consulted eagerly for other policies."""
        for policy in ("require_all", "first", "best_effort"):
            settings = _settings(policy=policy, quorum_count=None)
            for event in (ARRIVAL, TIMEOUT, FLUSH, LOSS):
                if policy == "first" and event in (TIMEOUT, FLUSH):
                    continue  # arrived=0 needed to avoid the arrival invariant
                decide_coalesce(settings, event, arrived_count=1, lost_branches={}, row_id="row_1")
            # zero-arrival resolution path for 'first'
            if policy == "first":
                decide_coalesce(settings, TIMEOUT, arrived_count=0, lost_branches={}, row_id="row_1")


class TestCoalesceDecisionInvariant:
    def test_fail_without_reason_rejected(self) -> None:
        with pytest.raises(OrchestrationInvariantError, match="if and only if"):
            CoalesceDecision(action=CoalesceAction.FAIL)

    def test_merge_with_reason_rejected(self) -> None:
        with pytest.raises(OrchestrationInvariantError, match="if and only if"):
            CoalesceDecision(action=CoalesceAction.MERGE, failure_reason="nope")

    def test_wait_with_reason_rejected(self) -> None:
        with pytest.raises(OrchestrationInvariantError, match="if and only if"):
            CoalesceDecision(action=CoalesceAction.WAIT, failure_reason="nope")

    def test_require_failure_reason_on_fail(self) -> None:
        decision = CoalesceDecision(action=CoalesceAction.FAIL, failure_reason="incomplete_branches")
        assert decision.require_failure_reason() == "incomplete_branches"

    def test_require_failure_reason_on_merge_crashes(self) -> None:
        decision = CoalesceDecision(action=CoalesceAction.MERGE)
        with pytest.raises(OrchestrationInvariantError, match="only FAIL decisions carry a reason"):
            decision.require_failure_reason()
