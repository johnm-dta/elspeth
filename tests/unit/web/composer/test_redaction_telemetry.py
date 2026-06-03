"""Tests for RedactionTelemetry Protocol and NoopRedactionTelemetry impl.

Closes plan-review W4 (telemetry duck-typing). The walker accepts a
typed Protocol instance, never None.

Rev-2 M_telemetry_implementation: the OtelRedactionTelemetry impl uses
module-level create_counter() objects + .add() calls per the project's
established pattern at service.py:135, 148, 824, 868, 1172, 1182.
There is NO _increment_counter helper — that function does not exist.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from elspeth.web.composer.redaction_telemetry import (
    NoopRedactionTelemetry,
    RedactionTelemetry,
)


def test_noop_implements_protocol() -> None:
    noop: RedactionTelemetry = NoopRedactionTelemetry()
    noop.unknown_response_key_redacted(tool_name="t")
    noop.manifest_dispatch(tool_name="t", shape="declarative")
    noop.summarizer_error(tool_name="t")


def test_noop_records_for_assertion_in_tests() -> None:
    noop = NoopRedactionTelemetry()
    noop.unknown_response_key_redacted(tool_name="set_source")
    noop.manifest_dispatch(tool_name="set_source", shape="type_driven")
    noop.summarizer_error(tool_name="set_source")
    assert noop.unknown_response_key_calls == [{"tool_name": "set_source"}]
    assert noop.manifest_dispatch_calls == [{"tool_name": "set_source", "shape": "type_driven"}]
    assert noop.summarizer_error_calls == [{"tool_name": "set_source"}]


def test_otel_telemetry_emits_via_module_level_counters(monkeypatch) -> None:
    """Production impl uses module-level counter objects + .add() calls.

    Rev-2 M_telemetry_implementation: patch the counter objects themselves,
    not a nonexistent _increment_counter helper. The established OTel pattern
    in this project (service.py:135, 148, 824, 868, 1172, 1182) is:
        _FOO_COUNTER = metrics.get_meter(__name__).create_counter(...)
        _FOO_COUNTER.add(1, {"label_key": value})
    """
    import elspeth.web.composer.redaction_telemetry as rt_mod
    from elspeth.web.composer.redaction_telemetry import OtelRedactionTelemetry

    mock_unknown = MagicMock()
    mock_dispatch = MagicMock()
    mock_summarizer = MagicMock()

    monkeypatch.setattr(rt_mod, "_UNKNOWN_RESPONSE_KEY_COUNTER", mock_unknown)
    monkeypatch.setattr(rt_mod, "_MANIFEST_DISPATCH_COUNTER", mock_dispatch)
    monkeypatch.setattr(rt_mod, "_SUMMARIZER_ERROR_COUNTER", mock_summarizer)

    tel = OtelRedactionTelemetry()
    tel.unknown_response_key_redacted(tool_name="set_source")
    tel.manifest_dispatch(tool_name="set_source", shape="type_driven")
    tel.summarizer_error(tool_name="set_source")

    mock_unknown.add.assert_called_once_with(1, {"tool_name": "set_source"})
    mock_dispatch.add.assert_called_once_with(1, {"tool_name": "set_source", "shape": "type_driven"})
    mock_summarizer.add.assert_called_once_with(1, {"tool_name": "set_source"})
