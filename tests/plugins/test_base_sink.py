"""Tests for BaseSink resume capability."""

import pytest

from elspeth.plugins.base import BaseSink


def test_base_sink_supports_resume_default_false():
    """BaseSink.supports_resume should default to False."""
    assert BaseSink.supports_resume is False


def test_base_sink_configure_for_resume_raises_not_implemented():
    """BaseSink.configure_for_resume should raise NotImplementedError by default."""

    class TestSink(BaseSink):
        name = "test"
        input_schema = None  # type: ignore

        def write(self, rows, ctx):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    sink = TestSink({})

    with pytest.raises(NotImplementedError) as exc_info:
        sink.configure_for_resume()

    assert "TestSink" in str(exc_info.value)
    assert "resume" in str(exc_info.value).lower()
