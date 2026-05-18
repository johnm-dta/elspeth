"""Tests for BaseSink resume capability."""

import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.infrastructure.base import BaseSink


def test_base_sink_supports_resume_default_false():
    """BaseSink.supports_resume should default to False."""
    assert BaseSink.supports_resume is False


def test_base_sink_configure_for_resume_raises_not_implemented():
    """BaseSink.configure_for_resume should raise NotImplementedError by default."""

    class TestSink(BaseSink):
        name = "test"
        determinism = Determinism.IO_WRITE
        input_schema = None
        _on_write_failure: str | None = "discard"

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


def test_base_sink_resume_field_resolution_raises_when_required():
    """A sink that requires resume field resolution must not inherit the no-op."""

    class TestSink(BaseSink):
        name = "test"
        determinism = Determinism.IO_WRITE
        input_schema = None
        _on_write_failure: str | None = "discard"

        def write(self, rows, ctx):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    sink = TestSink({})
    sink._needs_resume_field_resolution = True

    with pytest.raises(NotImplementedError) as exc_info:
        sink.set_resume_field_resolution({"User ID": "user_id"})

    assert "TestSink" in str(exc_info.value)
    assert "field resolution" in str(exc_info.value)
