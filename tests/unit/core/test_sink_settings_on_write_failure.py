"""Test SinkSettings.on_write_failure mandatory field."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.core.config import SinkSettings


class TestSinkSettingsOnWriteFailure:
    def test_on_write_failure_required(self) -> None:
        """SinkSettings must have on_write_failure — no default."""
        with pytest.raises(ValidationError, match="on_write_failure"):
            SinkSettings(plugin="csv")  # type: ignore[call-arg]

    def test_on_write_failure_discard(self) -> None:
        s = SinkSettings(plugin="csv", on_write_failure="discard")
        assert s.on_write_failure == "discard"

    def test_on_write_failure_sink_name(self) -> None:
        s = SinkSettings(plugin="csv", on_write_failure="csv_failsink")
        assert s.on_write_failure == "csv_failsink"

    def test_on_write_failure_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="on_write_failure"):
            SinkSettings(plugin="csv", on_write_failure="")

    def test_on_write_failure_whitespace_only_rejected(self) -> None:
        with pytest.raises(ValidationError, match="on_write_failure"):
            SinkSettings(plugin="csv", on_write_failure="   ")

    def test_on_write_failure_whitespace_stripped(self) -> None:
        s = SinkSettings(plugin="csv", on_write_failure="  discard  ")
        assert s.on_write_failure == "discard"

    def test_on_write_failure_with_options(self) -> None:
        s = SinkSettings(
            plugin="chroma_sink",
            on_write_failure="csv_failsink",
            options={"collection": "test"},
        )
        assert s.on_write_failure == "csv_failsink"
        assert s.options == {"collection": "test"}
