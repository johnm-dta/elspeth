"""Unit tests for evals/composer-rgr Tier 2 §7.7 anti-anchor hint injection.

The hint fires when the LLM has retried the same tool with byte-identical
arguments three times in a row — the empirically-observed "anchored loop"
failure mode where the model stops drifting and locks onto a payload the
validator already rejected.

Tests live alongside other composer-loop tests so the §7.7 logic can be
verified without the cost of spinning up the full compose() coroutine.
"""

from __future__ import annotations

from collections import deque

import pytest

from elspeth.web.composer.anti_anchor import (
    AntiAnchorTracker,
    build_anti_anchor_hint,
    should_inject_hint,
)


def test_three_identical_consecutive_failures_trigger_hint() -> None:
    failures = deque([("set_pipeline", "hash-a"), ("set_pipeline", "hash-a"), ("set_pipeline", "hash-a")], maxlen=5)
    assert should_inject_hint(failures) is True


def test_two_identical_failures_do_not_trigger_hint() -> None:
    failures = deque([("set_pipeline", "hash-a"), ("set_pipeline", "hash-a")], maxlen=5)
    assert should_inject_hint(failures) is False


def test_three_different_hashes_do_not_trigger_hint() -> None:
    failures = deque([("set_pipeline", "hash-a"), ("set_pipeline", "hash-b"), ("set_pipeline", "hash-c")], maxlen=5)
    assert should_inject_hint(failures) is False


def test_two_identical_then_third_different_does_not_trigger() -> None:
    failures = deque([("set_pipeline", "hash-a"), ("set_pipeline", "hash-a"), ("set_pipeline", "hash-b")], maxlen=5)
    assert should_inject_hint(failures) is False


def test_three_different_tools_with_same_hash_do_not_trigger() -> None:
    """Anchor must be on (tool, hash) pair, not hash alone."""
    failures = deque([("upsert_node", "hash-a"), ("set_pipeline", "hash-a"), ("set_output", "hash-a")], maxlen=5)
    assert should_inject_hint(failures) is False


def test_hint_fires_on_most_recent_three_only() -> None:
    """Earlier successes/different attempts upstream must not poison the trigger."""
    failures = deque(
        [
            ("upsert_node", "hash-x"),  # earlier failure, different tool
            ("set_pipeline", "hash-a"),  # 1
            ("set_pipeline", "hash-a"),  # 2
            ("set_pipeline", "hash-a"),  # 3
        ],
        maxlen=5,
    )
    assert should_inject_hint(failures) is True


def test_anti_anchor_tracker_tracks_failures() -> None:
    tracker = AntiAnchorTracker()
    tracker.record_failure("set_pipeline", "hash-a")
    tracker.record_failure("set_pipeline", "hash-a")
    assert tracker.should_fire() is False
    tracker.record_failure("set_pipeline", "hash-a")
    assert tracker.should_fire() is True


def test_anti_anchor_tracker_clears_on_success() -> None:
    """Any tool success breaks the anchor — the model has converged on something."""
    tracker = AntiAnchorTracker()
    tracker.record_failure("set_pipeline", "hash-a")
    tracker.record_failure("set_pipeline", "hash-a")
    tracker.record_success()
    tracker.record_failure("set_pipeline", "hash-a")
    assert tracker.should_fire() is False  # only 1 failure post-success


def test_anti_anchor_tracker_clears_on_hint_fire() -> None:
    """After firing, tracker resets — don't re-fire on the same anchor."""
    tracker = AntiAnchorTracker()
    tracker.record_failure("set_pipeline", "hash-a")
    tracker.record_failure("set_pipeline", "hash-a")
    tracker.record_failure("set_pipeline", "hash-a")
    assert tracker.should_fire() is True
    tracker.consume_fire()
    assert tracker.should_fire() is False
    # And a 4th identical failure does NOT immediately re-fire
    tracker.record_failure("set_pipeline", "hash-a")
    assert tracker.should_fire() is False


def test_hint_message_names_the_anchored_tool() -> None:
    failures = deque([("set_pipeline", "hash-a"), ("set_pipeline", "hash-a"), ("set_pipeline", "hash-a")], maxlen=5)
    hint = build_anti_anchor_hint(failures)
    assert "set_pipeline" in hint
    assert "3" in hint  # references the retry count
    assert "byte-identical" in hint or "identical" in hint


def test_hint_message_actionable_instructions() -> None:
    """The hint must tell the model what to do, not just diagnose the failure."""
    failures = deque([("upsert_node", "hash-z"), ("upsert_node", "hash-z"), ("upsert_node", "hash-z")], maxlen=5)
    hint = build_anti_anchor_hint(failures)
    # Mentions concrete repair steps.
    assert "validator" in hint.lower()
    assert "change" in hint.lower()
    # Includes a system-prefix marker so the model recognises it as system-injected.
    assert hint.startswith("[ELSPETH-SYSTEM-HINT]")


def test_should_inject_hint_handles_empty_deque() -> None:
    failures: deque[tuple[str, str]] = deque(maxlen=5)
    assert should_inject_hint(failures) is False


def test_should_inject_hint_handles_under_three_entries() -> None:
    failures = deque([("set_pipeline", "h")], maxlen=5)
    assert should_inject_hint(failures) is False


def test_tracker_uses_bounded_deque() -> None:
    """Memory-bounded — record_failure beyond maxlen drops oldest, never grows unbounded."""
    tracker = AntiAnchorTracker()
    for i in range(20):
        tracker.record_failure("t", f"h-{i}")
    # Only the last 5 are retained; should_fire checks the last 3.
    assert tracker.should_fire() is False  # last 3 are h-17, h-18, h-19 — all different
    assert len(tracker._failures) == 5  # bounded


@pytest.mark.parametrize("count", [3, 4, 5])
def test_n_or_more_consecutive_identical_all_trigger(count: int) -> None:
    failures: deque[tuple[str, str]] = deque(maxlen=5)
    for _ in range(count):
        failures.append(("set_pipeline", "h"))
    assert should_inject_hint(failures) is True
