"""Tests for BaseSink diversion infrastructure."""

from __future__ import annotations

import pytest

from elspeth.contracts import Determinism
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.errors import FrameworkBugError
from elspeth.plugins.infrastructure.base import BaseSink


class StubSink(BaseSink):
    """Minimal concrete sink for testing diversion infrastructure."""

    name = "stub_sink"
    determinism = Determinism.IO_WRITE

    def write(self, rows, ctx):
        for i, row in enumerate(rows):
            if row["should_divert"]:
                self._divert_row(row, row_index=i, reason="bad value")
        return SinkWriteResult(
            artifact=self._null_artifact(),
            diversions=self._get_diversions(),
        )

    def _null_artifact(self):
        from elspeth.contracts.results import ArtifactDescriptor

        return ArtifactDescriptor.for_file(path="/dev/null", content_hash="0" * 12, size_bytes=0)

    def flush(self):
        pass

    def close(self):
        pass


def _make_sink(*, on_write_failure: str | None = "discard") -> StubSink:
    """Create a StubSink with _diversion_log and _on_write_failure initialized."""
    sink = StubSink.__new__(StubSink)
    sink._diversion_log = []
    sink._on_write_failure = on_write_failure
    return sink


class TestDivertRow:
    def test_divert_row_records_diversion(self) -> None:
        sink = _make_sink()
        sink._divert_row({"a": 1}, row_index=0, reason="test reason")
        diversions = sink._get_diversions()
        assert len(diversions) == 1
        assert diversions[0].row_index == 0
        assert diversions[0].reason == "test reason"
        assert diversions[0].row_data["a"] == 1

    def test_get_diversions_returns_tuple(self) -> None:
        sink = _make_sink()
        sink._divert_row({"x": 1}, row_index=0, reason="r1")
        sink._divert_row({"x": 2}, row_index=2, reason="r2")
        result = sink._get_diversions()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_diversion_log_clears(self) -> None:
        sink = _make_sink()
        sink._divert_row({"a": 1}, row_index=0, reason="test")
        sink._reset_diversion_log()
        assert sink._get_diversions() == ()

    def test_get_diversions_freezes_row_data(self) -> None:
        sink = _make_sink()
        sink._divert_row({"nested": {"key": "val"}}, row_index=0, reason="test")
        diversions = sink._get_diversions()
        with pytest.raises(TypeError):
            diversions[0].row_data["new"] = "bad"  # type: ignore[index]

    def test_divert_without_on_write_failure_raises(self) -> None:
        """Calling _divert_row when _on_write_failure is None is a plugin bug."""
        sink = _make_sink(on_write_failure=None)
        with pytest.raises(FrameworkBugError, match="on_write_failure"):
            sink._divert_row({"a": 1}, row_index=0, reason="test")

    def test_divert_with_discard_mode_accumulates(self) -> None:
        """Discard mode still accumulates to log — executor decides outcome."""
        sink = _make_sink(on_write_failure="discard")
        sink._divert_row({"a": 1}, row_index=0, reason="intentional drop")
        diversions = sink._get_diversions()
        assert len(diversions) == 1
        assert diversions[0].reason == "intentional drop"

    def test_divert_with_failsink_name_accumulates(self) -> None:
        """Failsink mode also accumulates to log."""
        sink = _make_sink(on_write_failure="csv_failsink")
        sink._divert_row({"a": 1}, row_index=0, reason="api rejected")
        assert len(sink._get_diversions()) == 1
