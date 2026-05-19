"""OpenTelemetry helpers for first-run tutorial signals."""

from __future__ import annotations

from typing import Literal

from opentelemetry import metrics

_CompletionPath = Literal["first_time", "skip", "retake", "repeat"]
_COMPLETION_PATHS: frozenset[str] = frozenset({"first_time", "skip", "retake", "repeat"})

_RuntimeNormalizationKind = Literal["bare_required_field_templates"]
_RUNTIME_NORMALIZATION_KINDS: frozenset[str] = frozenset({"bare_required_field_templates"})

_meter = metrics.get_meter(__name__)
_TUTORIAL_COMPLETED_COUNTER = _meter.create_counter(
    "composer.tutorial.completed_total",
    description=("First-run tutorial completion preference writes. Attributes: completion_path in {first_time, skip, retake, repeat}."),
)
_TUTORIAL_ABANDON_COUNTER = _meter.create_counter(
    "composer.tutorial.abandon_total",
    description="Best-effort tutorial abandon beacons sent during page unload/navigation away.",
)
_TUTORIAL_RUNTIME_NORMALIZATION_COUNTER = _meter.create_counter(
    "composer.tutorial.runtime_normalization_total",
    description=(
        "First-run tutorial pre-execution state normalizations. Attributes: "
        "kind in {bare_required_field_templates}. Emitted whenever "
        "_normalise_current_tutorial_state_for_execution rewrites the "
        "composition state before a live tutorial run."
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


def record_tutorial_runtime_normalization(kind: _RuntimeNormalizationKind) -> None:
    """Increment the tutorial runtime-normalization counter.

    Operational telemetry (not logging) — see ``CLAUDE.md`` "Telemetry and
    Logging" and ``feedback_no_slog_recommendations``. Fires once per
    composition-state save where the tutorial pre-execution normalizer
    rewrote the state.
    """
    if kind not in _RUNTIME_NORMALIZATION_KINDS:
        raise ValueError(f"kind must be one of {sorted(_RUNTIME_NORMALIZATION_KINDS)!r}; got {kind!r}")
    _TUTORIAL_RUNTIME_NORMALIZATION_COUNTER.add(1, attributes={"kind": kind})
    return None
