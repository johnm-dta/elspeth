"""Tests for shared plugin telemetry helpers."""

from __future__ import annotations

from typing import Any

from elspeth.plugins.infrastructure.telemetry import warn_telemetry_before_start


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def warning(self, message: str, **kwargs: Any) -> None:
        self.calls.append((message, kwargs))


def test_warn_telemetry_before_start_records_event_type() -> None:
    logger = _RecordingLogger()

    warn_telemetry_before_start(object(), logger=logger)

    assert logger.calls == [
        (
            "telemetry_emit called before on_start() — event dropped",
            {"event_type": "object"},
        )
    ]
