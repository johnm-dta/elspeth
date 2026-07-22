"""OpenTelemetry helpers for first-run tutorial signals."""

from __future__ import annotations

from typing import Literal

from opentelemetry import metrics

_CompletionPath = Literal["first_time", "skip", "retake", "repeat", "exit"]
_COMPLETION_PATHS: frozenset[str] = frozenset({"first_time", "skip", "retake", "repeat", "exit"})

_meter = metrics.get_meter(__name__)
_TUTORIAL_COMPLETED_COUNTER = _meter.create_counter(
    "composer.tutorial.completed_total",
    description=(
        "First-run tutorial completion preference writes. Attributes: completion_path in {first_time, skip, retake, repeat, exit}."
    ),
)
_TUTORIAL_ABANDON_COUNTER = _meter.create_counter(
    "composer.tutorial.abandon_total",
    description="Best-effort tutorial abandon beacons sent during page unload/navigation away.",
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
