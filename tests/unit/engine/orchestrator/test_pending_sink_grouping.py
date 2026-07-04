# tests/unit/engine/orchestrator/test_pending_sink_grouping.py
"""Tests for SinkFlushCoordinator.write_pending_to_sinks grouping logic.

Bug: elspeth-1569caa900 — The grouping logic uses sort+groupby to separate
tokens by PendingOutcome. If pending_sort_key produces wrong grouping,
QUARANTINED tokens could be written with COMPLETED outcome metadata.

These tests verify:
1. Tokens with different PendingOutcome values are grouped separately
2. Tokens with None pending_outcome are grouped separately from those with outcomes
3. Sort stability — tokens with same outcome stay in original order
4. OrchestrationInvariantError when sink_name is not in config.sinks
"""

from __future__ import annotations

from itertools import groupby
from unittest.mock import Mock

import pytest

from elspeth.contracts import PendingOutcome, TokenInfo
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.engine.orchestrator.types import ExecutionCounters
from elspeth.testing import make_token_info

# =============================================================================
# Helpers — replicate the sort key from sink_flush.py for isolated testing
# =============================================================================


def _pending_sort_key(pair: tuple[TokenInfo, PendingOutcome | None]) -> tuple[bool, str, str, str, bool]:
    """Replica of the pending_sort_key closure from write_pending_to_sinks.

    This must stay in sync with the production code. If the production sort key
    changes, these tests should break, signalling a need to update.
    """
    pending = pair[1]
    if pending is None:
        return (True, "", "", "", False)
    outcome_value = pending.outcome.value if pending.outcome is not None else ""
    return (False, outcome_value, pending.path.value, pending.error_hash or "", pending.scheduler_pending_sink)


def _completed_pending() -> PendingOutcome:
    return PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)


def _routed_pending() -> PendingOutcome:
    return PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.GATE_ROUTED)


def _quarantined_pending(*, error_hash: str) -> PendingOutcome:
    return PendingOutcome(
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.QUARANTINED_AT_SOURCE,
        error_hash=error_hash,
    )


def _pending_pair(pending: PendingOutcome) -> tuple[TerminalOutcome | None, TerminalPath]:
    return pending.outcome, pending.path


def _group_pairs(
    pairs: list[tuple[TokenInfo, PendingOutcome | None]],
) -> list[tuple[PendingOutcome | None, list[TokenInfo]]]:
    """Sort and group pairs using the production sort key, return grouped results."""
    sorted_pairs = sorted(pairs, key=_pending_sort_key)
    result = []
    for pending_outcome, group in groupby(sorted_pairs, key=lambda x: x[1]):
        group_tokens = [token for token, _ in group]
        result.append((pending_outcome, group_tokens))
    return result


# =============================================================================
# Sort key correctness
# =============================================================================


class TestPendingSortKey:
    """Tests for the pending_sort_key closure in isolation."""

    def test_none_outcome_sorts_after_completed(self) -> None:
        """None pending_outcome sorts after non-None (True > False in tuple ordering)."""
        tok_none = make_token_info(token_id="tok-none")
        tok_completed = make_token_info(token_id="tok-completed")

        pair_none = (tok_none, None)
        pair_completed = (tok_completed, _completed_pending())

        # None maps to (True, "", "") which sorts after (False, ..., ...)
        assert _pending_sort_key(pair_none) > _pending_sort_key(pair_completed)

    def test_none_outcome_sorts_after_quarantined(self) -> None:
        """None pending_outcome sorts after QUARANTINED (True > False in tuple ordering)."""
        tok_none = make_token_info(token_id="tok-none")
        tok_quarantined = make_token_info(token_id="tok-q")

        pair_none = (tok_none, None)
        pair_quarantined = (tok_quarantined, _quarantined_pending(error_hash="abc123" * 10 + "abcd"))

        assert _pending_sort_key(pair_none) > _pending_sort_key(pair_quarantined)

    def test_completed_and_quarantined_sort_separately(self) -> None:
        """COMPLETED and QUARANTINED must produce different sort keys."""
        tok_c = make_token_info(token_id="tok-c")
        tok_q = make_token_info(token_id="tok-q")

        key_completed = _pending_sort_key((tok_c, _completed_pending()))
        key_quarantined = _pending_sort_key((tok_q, _quarantined_pending(error_hash="a" * 64)))

        assert key_completed != key_quarantined

    def test_routed_and_completed_sort_separately(self) -> None:
        """ROUTED and COMPLETED must produce different sort keys."""
        tok_r = make_token_info(token_id="tok-r")
        tok_c = make_token_info(token_id="tok-c")

        key_routed = _pending_sort_key((tok_r, _routed_pending()))
        key_completed = _pending_sort_key((tok_c, _completed_pending()))

        assert key_routed != key_completed

    def test_scheduler_handoff_sorts_separately(self) -> None:
        """Scheduler-backed and generated sink tokens must not share a callback group."""
        tok_scheduler = make_token_info(token_id="tok-scheduler")
        tok_generated = make_token_info(token_id="tok-generated")

        scheduler_pending = PendingOutcome(
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.COALESCED,
            scheduler_pending_sink=True,
        )
        generated_pending = PendingOutcome(
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.COALESCED,
            scheduler_pending_sink=False,
        )

        assert _pending_sort_key((tok_scheduler, scheduler_pending)) != _pending_sort_key((tok_generated, generated_pending))

    def test_same_outcome_same_error_hash_produces_equal_keys(self) -> None:
        """Two QUARANTINED tokens with the same error_hash should have equal sort keys."""
        tok1 = make_token_info(token_id="tok-1")
        tok2 = make_token_info(token_id="tok-2")
        error_hash = "b" * 64

        key1 = _pending_sort_key((tok1, _quarantined_pending(error_hash=error_hash)))
        key2 = _pending_sort_key((tok2, _quarantined_pending(error_hash=error_hash)))

        assert key1 == key2

    def test_quarantined_different_error_hashes_sort_separately(self) -> None:
        """QUARANTINED tokens with different error_hashes must have different sort keys."""
        tok1 = make_token_info(token_id="tok-1")
        tok2 = make_token_info(token_id="tok-2")

        key1 = _pending_sort_key((tok1, _quarantined_pending(error_hash="a" * 64)))
        key2 = _pending_sort_key((tok2, _quarantined_pending(error_hash="b" * 64)))

        assert key1 != key2


# =============================================================================
# Groupby correctness — the critical bug surface
# =============================================================================


class TestPendingGrouping:
    """Tests for sort+groupby producing correct groups."""

    def test_quarantined_and_completed_in_separate_groups(self) -> None:
        """QUARANTINED tokens must never appear in the same group as COMPLETED tokens.

        This is the core invariant from bug elspeth-1569caa900.
        """
        tok_c1 = make_token_info(token_id="tok-c1")
        tok_c2 = make_token_info(token_id="tok-c2")
        tok_q1 = make_token_info(token_id="tok-q1")
        tok_q2 = make_token_info(token_id="tok-q2")

        pairs: list[tuple[TokenInfo, PendingOutcome | None]] = [
            (tok_c1, _completed_pending()),
            (tok_q1, _quarantined_pending(error_hash="e" * 64)),
            (tok_c2, _completed_pending()),
            (tok_q2, _quarantined_pending(error_hash="e" * 64)),
        ]

        groups = _group_pairs(pairs)

        # Must produce exactly 2 groups
        assert len(groups) == 2

        # Extract outcome/path pairs and token IDs per group
        for pending_outcome, tokens in groups:
            token_ids = {t.token_id for t in tokens}
            if pending_outcome is not None and _pending_pair(pending_outcome) == (
                TerminalOutcome.SUCCESS,
                TerminalPath.DEFAULT_FLOW,
            ):
                assert token_ids == {"tok-c1", "tok-c2"}
            elif pending_outcome is not None and _pending_pair(pending_outcome) == (
                TerminalOutcome.FAILURE,
                TerminalPath.QUARANTINED_AT_SOURCE,
            ):
                assert token_ids == {"tok-q1", "tok-q2"}
            else:
                pytest.fail(f"Unexpected group outcome: {pending_outcome}")

    def test_none_outcome_grouped_separately(self) -> None:
        """Tokens with None pending_outcome form their own group."""
        tok_none1 = make_token_info(token_id="tok-n1")
        tok_none2 = make_token_info(token_id="tok-n2")
        tok_completed = make_token_info(token_id="tok-c1")

        pairs: list[tuple[TokenInfo, PendingOutcome | None]] = [
            (tok_completed, _completed_pending()),
            (tok_none1, None),
            (tok_none2, None),
        ]

        groups = _group_pairs(pairs)

        assert len(groups) == 2

        none_groups = [(po, toks) for po, toks in groups if po is None]
        assert len(none_groups) == 1
        assert len(none_groups[0][1]) == 2

        completed_groups = [(po, toks) for po, toks in groups if po is not None]
        assert len(completed_groups) == 1
        assert len(completed_groups[0][1]) == 1

    def test_three_distinct_outcomes_produce_three_groups(self) -> None:
        """COMPLETED, ROUTED, and QUARANTINED tokens each get their own group."""
        tok_c = make_token_info(token_id="tok-c")
        tok_r = make_token_info(token_id="tok-r")
        tok_q = make_token_info(token_id="tok-q")

        pairs: list[tuple[TokenInfo, PendingOutcome | None]] = [
            (tok_r, _routed_pending()),
            (tok_c, _completed_pending()),
            (tok_q, _quarantined_pending(error_hash="f" * 64)),
        ]

        groups = _group_pairs(pairs)

        assert len(groups) == 3
        pending_pairs = {_pending_pair(g[0]) if g[0] is not None else None for g in groups}
        assert pending_pairs == {
            (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW),
            (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED),
            (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE),
        }

    def test_sort_stability_preserves_insertion_order(self) -> None:
        """Tokens with the same PendingOutcome must stay in their original order."""
        tokens = [make_token_info(token_id=f"tok-{i}") for i in range(5)]
        outcome = _completed_pending()

        pairs: list[tuple[TokenInfo, PendingOutcome | None]] = [(tok, outcome) for tok in tokens]

        groups = _group_pairs(pairs)

        assert len(groups) == 1
        group_tokens = groups[0][1]
        assert [t.token_id for t in group_tokens] == [f"tok-{i}" for i in range(5)]

    def test_interleaved_outcomes_regroup_correctly(self) -> None:
        """Interleaved COMPLETED and QUARANTINED tokens are regrouped correctly.

        This is the adversarial case: if tokens arrive in alternating order
        (C, Q, C, Q, C), the sort+groupby must still produce two clean groups.
        """
        error_hash = "d" * 64
        pairs: list[tuple[TokenInfo, PendingOutcome | None]] = []
        for i in range(5):
            if i % 2 == 0:
                pairs.append((make_token_info(token_id=f"tok-c{i}"), _completed_pending()))
            else:
                pairs.append((make_token_info(token_id=f"tok-q{i}"), _quarantined_pending(error_hash=error_hash)))

        groups = _group_pairs(pairs)

        assert len(groups) == 2

        for pending_outcome, tokens in groups:
            assert pending_outcome is not None
            if _pending_pair(pending_outcome) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW):
                assert len(tokens) == 3
                assert all(t.token_id.startswith("tok-c") for t in tokens)
            elif _pending_pair(pending_outcome) == (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE):
                assert len(tokens) == 2
                assert all(t.token_id.startswith("tok-q") for t in tokens)
            else:
                pytest.fail(f"Unexpected pending pair: {_pending_pair(pending_outcome)!r}")

    def test_empty_pairs_produces_no_groups(self) -> None:
        """An empty list of pairs should produce no groups."""
        groups = _group_pairs([])
        assert groups == []

    def test_single_token_produces_single_group(self) -> None:
        """A single token produces exactly one group."""
        tok = make_token_info(token_id="tok-solo")
        pairs: list[tuple[TokenInfo, PendingOutcome | None]] = [
            (tok, _completed_pending()),
        ]

        groups = _group_pairs(pairs)

        assert len(groups) == 1
        assert groups[0][1] == [tok]

    def test_quarantined_tokens_with_different_error_hashes_group_by_hash(self) -> None:
        """QUARANTINED tokens with different error_hashes form separate groups.

        The groupby uses the PendingOutcome object identity via __eq__, so
        tokens with different error_hashes must land in different groups
        (each gets a different PendingOutcome passed to sink_executor.write).
        """
        tok_q1 = make_token_info(token_id="tok-q1")
        tok_q2 = make_token_info(token_id="tok-q2")
        tok_q3 = make_token_info(token_id="tok-q3")

        hash_a = "a" * 64
        hash_b = "b" * 64

        pairs: list[tuple[TokenInfo, PendingOutcome | None]] = [
            (tok_q1, _quarantined_pending(error_hash=hash_a)),
            (tok_q2, _quarantined_pending(error_hash=hash_b)),
            (tok_q3, _quarantined_pending(error_hash=hash_a)),
        ]

        groups = _group_pairs(pairs)

        # Two distinct error hashes = two groups (both QUARANTINED)
        quarantine_groups = [
            (po, toks)
            for po, toks in groups
            if po is not None and _pending_pair(po) == (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE)
        ]
        assert len(quarantine_groups) == 2

        hash_to_tokens: dict[str, list[str]] = {}
        for po, toks in quarantine_groups:
            assert po is not None and po.error_hash is not None
            hash_to_tokens[po.error_hash] = [t.token_id for t in toks]

        assert hash_to_tokens[hash_a] == ["tok-q1", "tok-q3"]
        assert hash_to_tokens[hash_b] == ["tok-q2"]


# =============================================================================
# OrchestrationInvariantError — sink_name validation
# =============================================================================


class TestSinkNameValidation:
    """Tests for the OrchestrationInvariantError when sink_name is missing from config."""

    def _make_orchestrator(self) -> Mock:
        """Build a minimal mock SinkFlushCoordinator with write_pending_to_sinks accessible."""
        # Import the actual method to test it directly
        from elspeth.engine.orchestrator.sink_flush import SinkFlushCoordinator

        orchestrator = Mock(spec=SinkFlushCoordinator)
        orchestrator._span_factory = Mock()
        # Bind the real method
        orchestrator.write_pending_to_sinks = SinkFlushCoordinator.write_pending_to_sinks.__get__(orchestrator)
        return orchestrator

    def test_missing_sink_name_raises_orchestration_invariant_error(self) -> None:
        """If pending_tokens references a sink not in config.sinks, raise OrchestrationInvariantError."""
        orchestrator = self._make_orchestrator()

        recorder = Mock()
        config = Mock()
        config.sinks = {"output": Mock()}  # Only "output" exists
        ctx = Mock()

        tok = make_token_info(token_id="tok-1")
        pending_tokens = {
            "nonexistent_sink": [(tok, _completed_pending())],
        }

        with pytest.raises(OrchestrationInvariantError, match="nonexistent_sink"):
            orchestrator.write_pending_to_sinks(
                factory=recorder,
                run_id="test-run",
                config=config,
                ctx=ctx,
                counters=ExecutionCounters(),
                pending_tokens=pending_tokens,
                sink_id_map={"output": "node-1"},
                edge_map={},
                sink_step=5,
            )

    def test_empty_token_list_skips_sink(self) -> None:
        """A sink with an empty token list should be skipped without error."""
        orchestrator = self._make_orchestrator()

        recorder = Mock()
        config = Mock()
        # Even if the sink doesn't exist in config, empty list means we skip before checking
        config.sinks = {}
        ctx = Mock()

        pending_tokens: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {
            "nonexistent_sink": [],
        }

        # Should not raise — empty list triggers `continue` before sink lookup
        orchestrator.write_pending_to_sinks(
            factory=recorder,
            run_id="test-run",
            config=config,
            ctx=ctx,
            counters=ExecutionCounters(),
            pending_tokens=pending_tokens,
            sink_id_map={},
            edge_map={},
            sink_step=5,
        )
