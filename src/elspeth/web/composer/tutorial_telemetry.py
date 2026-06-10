"""OpenTelemetry helpers for first-run tutorial signals."""

from __future__ import annotations

from typing import Literal

from opentelemetry import metrics

_CompletionPath = Literal["first_time", "skip", "retake", "repeat"]
_COMPLETION_PATHS: frozenset[str] = frozenset({"first_time", "skip", "retake", "repeat"})

# I6 — silent-failure-hunter remediation. Closed-list of reasons
# ``_store_successful_live_projection`` may refuse to seed the
# tutorial cache. Each name maps 1:1 to a branch in
# ``_cache_seed_skip_reason`` (tutorial_service.py); the conditional
# ordering there resolves overlapping subset relationships
# (quarantined ⊂ failed, routed_failure ⊂ failed) so the emitted
# attribute is always the MORE SPECIFIC reason rather than the
# catch-all ``rows_failed``.
_CacheSkipReason = Literal[
    "status_not_completed",
    "zero_rows_processed",
    "rows_quarantined",
    "rows_routed_failure",
    "rows_failed",
    "rows_routed_success",
    "rows_partial_success",
]
_CACHE_SKIP_REASONS: frozenset[str] = frozenset(
    {
        "status_not_completed",
        "zero_rows_processed",
        "rows_quarantined",
        "rows_routed_failure",
        "rows_failed",
        "rows_routed_success",
        "rows_partial_success",
    }
)

_meter = metrics.get_meter(__name__)
_TUTORIAL_COMPLETED_COUNTER = _meter.create_counter(
    "composer.tutorial.completed_total",
    description=("First-run tutorial completion preference writes. Attributes: completion_path in {first_time, skip, retake, repeat}."),
)
_TUTORIAL_ABANDON_COUNTER = _meter.create_counter(
    "composer.tutorial.abandon_total",
    description="Best-effort tutorial abandon beacons sent during page unload/navigation away.",
)
_TUTORIAL_CACHE_SKIPPED_COUNTER = _meter.create_counter(
    "composer.tutorial.cache_skipped_total",
    description=(
        "First-run tutorial cache-store skips. Emitted when "
        "_store_successful_live_projection refuses to seed the cache "
        "because the live run did not meet the all-rows-succeeded "
        "predicate (status != completed, zero rows, any failed / "
        "routed / quarantined rows, or a partial-success gap). "
        "Attributes: skip_reason in {status_not_completed, "
        "zero_rows_processed, rows_quarantined, rows_routed_failure, "
        "rows_failed, rows_routed_success, rows_partial_success}. "
        "A non-trivial counter floor signals tutorial pipeline "
        "degradation — the canonical tutorial prompt + model + sources "
        "should yield rows_processed > 0 and rows_succeeded == "
        "rows_processed on every run; persistent skips mean every "
        "billed live run is throwing away its cache-seed value (LLM-"
        "billing surge before anyone notices)."
    ),
)


def record_tutorial_completed_path(completion_path: _CompletionPath) -> None:
    """Increment the tutorial completion counter with a server-derived path."""
    if completion_path not in _COMPLETION_PATHS:
        raise ValueError(f"completion_path must be one of {sorted(_COMPLETION_PATHS)!r}; got {completion_path!r}")
    _TUTORIAL_COMPLETED_COUNTER.add(1, attributes={"completion_path": completion_path})
    return None


def record_tutorial_abandoned() -> None:
    """Increment the best-effort tutorial abandon counter."""
    _TUTORIAL_ABANDON_COUNTER.add(1, attributes={})
    return None


def record_tutorial_cache_skipped(skip_reason: _CacheSkipReason) -> None:
    """Increment the tutorial cache-skipped counter.

    Operational telemetry only — emitted from
    ``_store_successful_live_projection`` when the live run cannot
    seed the cache. The closed-list ``skip_reason`` is computed by
    ``_cache_seed_skip_reason`` (tutorial_service.py) from the
    ``RunRecord`` counters; the ordering there resolves overlapping
    subset relationships so the reason is always the more specific
    classification rather than the catch-all ``rows_failed``.
    """
    if skip_reason not in _CACHE_SKIP_REASONS:
        raise ValueError(f"skip_reason must be one of {sorted(_CACHE_SKIP_REASONS)!r}; got {skip_reason!r}")
    _TUTORIAL_CACHE_SKIPPED_COUNTER.add(1, attributes={"skip_reason": skip_reason})
    return None
