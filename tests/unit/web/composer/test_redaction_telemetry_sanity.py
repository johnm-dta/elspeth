"""Production OtelRedactionTelemetry sanity check.

With the production ``OtelRedactionTelemetry`` instance, exercise the three
named counters via real ``redact_tool_call_arguments`` /
``redact_tool_call_response`` walker paths and assert each counter is
incremented as expected — proving no silent counter-name typo or signature
drift in the production code path.

Sister test to ``test_redaction_telemetry.py::test_otel_telemetry_emits_via_module_level_counters``
which exercises the telemetry instance methods directly.  This file exercises
the *walker* code path (the redaction.py call sites that invoke the telemetry
methods) so a typo in any redactor's keyword argument or a regression in the
walker's emission ordering would surface here even if the telemetry instance
methods themselves still work.

Counters covered:

  * ``composer.redaction.manifest_dispatch`` — fires once per
    ``redact_tool_call_arguments`` and once per ``redact_tool_call_response``
    invocation, with a ``shape`` label (``type_driven`` or ``declarative``).
  * ``composer.redaction.unknown_response_key_redacted`` — fires once per
    unknown key in a declarative-entry response (fail-closed sentinel
    substitution path).
  * ``composer.redaction.summarizer_errors_total`` — fires immediately
    before ``AuditIntegrityError`` raise on summarizer exception OR non-str
    return.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

import elspeth.web.composer.redaction_telemetry as rt_mod
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.redaction import (
    HandlesNoSensitiveDataReason,
    ToolRedaction,
    ToolRedactionPolicy,
    redact_tool_call_arguments,
    redact_tool_call_response,
)
from elspeth.web.composer.redaction_telemetry import OtelRedactionTelemetry


@pytest.fixture
def patched_counters(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, MagicMock]]:
    """Replace the three module-level OTel counters with MagicMocks.

    Returns the dict so each test can assert on whichever counter is
    relevant to the pathway exercised.  Monkeypatch restores the originals
    on teardown so other tests in the suite are not affected.
    """
    counters = {
        "manifest_dispatch": MagicMock(),
        "unknown_response_key": MagicMock(),
        "summarizer_error": MagicMock(),
    }
    monkeypatch.setattr(rt_mod, "_MANIFEST_DISPATCH_COUNTER", counters["manifest_dispatch"])
    monkeypatch.setattr(rt_mod, "_UNKNOWN_RESPONSE_KEY_COUNTER", counters["unknown_response_key"])
    monkeypatch.setattr(rt_mod, "_SUMMARIZER_ERROR_COUNTER", counters["summarizer_error"])
    yield counters


def _patch_manifest_entry(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    entry: ToolRedaction,
) -> None:
    """Extend the module-level MANIFEST with a test entry."""
    from types import MappingProxyType

    import elspeth.web.composer.redaction as _redaction_mod

    new_manifest = MappingProxyType({**_redaction_mod.MANIFEST, tool_name: entry})
    monkeypatch.setattr(_redaction_mod, "MANIFEST", new_manifest)


def _safe_reason() -> HandlesNoSensitiveDataReason:
    return HandlesNoSensitiveDataReason(
        sensitive_data_locations=("no LLM-supplied inputs reach this tool",),
        why_arguments_safe="All arguments are structural metadata only; no user content reaches the handler.",
        why_responses_safe="Response is a closed-set success/version pair; no payload bytes or operator strings are echoed.",
    )


def test_manifest_dispatch_counter_fires_on_argument_redaction(patched_counters: dict[str, MagicMock]) -> None:
    """``redact_tool_call_arguments`` fires manifest_dispatch once per call.

    Exercises the production ``OtelRedactionTelemetry`` against a real
    type-driven manifest entry (``set_source``) — proves the counter name,
    keyword arguments, and label values flow end-to-end through the walker.
    """
    tel = OtelRedactionTelemetry()
    redact_tool_call_arguments(
        "set_source",
        {"plugin": "csv_local", "on_success": "node_a", "on_validation_failure": "node_b", "options": {"path": "/tmp/x"}},
        telemetry=tel,
    )
    patched_counters["manifest_dispatch"].add.assert_called_once_with(1, {"tool_name": "set_source", "shape": "type_driven"})


def test_manifest_dispatch_counter_fires_on_response_redaction(patched_counters: dict[str, MagicMock]) -> None:
    """``redact_tool_call_response`` fires manifest_dispatch once per call.

    Exercises the response walker via a declarative manifest entry
    (``list_sources``) so the ``shape="declarative"`` label is exercised in
    addition to the ``type_driven`` label.
    """
    tel = OtelRedactionTelemetry()
    redact_tool_call_response("list_sources", {}, telemetry=tel)
    # Declarative entries with no sensitive keys still emit the dispatch
    # beacon — the beacon is per-invocation, not per-key.
    patched_counters["manifest_dispatch"].add.assert_called_once_with(1, {"tool_name": "list_sources", "shape": "declarative"})


def test_unknown_response_key_counter_fires_on_declarative_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
    patched_counters: dict[str, MagicMock],
) -> None:
    """An unknown key in a declarative-entry response fires the counter.

    Constructs a test manifest entry with ``known_response_keys=("ok",)``
    and feeds a response carrying an extra key ``"stray"``.  The walker
    fail-closes the stray key with ``REDACTED_UNKNOWN_RESPONSE_KEY`` and
    fires ``unknown_response_key_redacted(tool_name=...)``.
    """
    entry = ToolRedaction(
        policy=ToolRedactionPolicy(
            handles_no_sensitive_data=False,
            known_response_keys=("ok",),
        )
    )
    _patch_manifest_entry(monkeypatch, "telemetry_sanity_tool", entry)

    tel = OtelRedactionTelemetry()
    redact_tool_call_response("telemetry_sanity_tool", {"ok": True, "stray": "garbage"}, telemetry=tel)

    patched_counters["unknown_response_key"].add.assert_called_once_with(1, {"tool_name": "telemetry_sanity_tool"})


def test_summarizer_error_counter_fires_before_audit_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
    patched_counters: dict[str, MagicMock],
) -> None:
    """A summarizer that raises fires summarizer_error_total BEFORE the AuditIntegrityError raise.

    Constructs a test manifest entry with a declarative policy whose
    ``argument_summarizers["sensitive_key"]`` raises ``RuntimeError`` when
    invoked.  The walker fires ``summarizer_error(tool_name=...)`` then
    raises ``AuditIntegrityError`` chained from the underlying.  This pins
    the rev-2 M.8 discipline: telemetry fires BEFORE the raise so the
    counter increment is not lost when the audit path aborts.
    """

    def _boom(value: object) -> str:
        raise RuntimeError("summarizer-deliberately-raises")

    entry = ToolRedaction(
        policy=ToolRedactionPolicy(
            sensitive_argument_keys=("sensitive_key",),
            argument_summarizers={"sensitive_key": _boom},
            handles_no_sensitive_data=False,
            known_response_keys=("ok",),
        )
    )
    _patch_manifest_entry(monkeypatch, "telemetry_summarizer_error_tool", entry)

    tel = OtelRedactionTelemetry()
    with pytest.raises(AuditIntegrityError, match="summarizer"):
        redact_tool_call_arguments(
            "telemetry_summarizer_error_tool",
            {"sensitive_key": "any-value-triggers-boom"},
            telemetry=tel,
        )

    # Counter must have been incremented BEFORE the raise.
    patched_counters["summarizer_error"].add.assert_called_once_with(1, {"tool_name": "telemetry_summarizer_error_tool"})
    # And the safe_reason fixture must not have been needed — sentinel:
    # the test is exercising the summarizer-raise path, not the
    # handles_no_sensitive_data path. The unused-import lint guard keeps
    # this fixture available for sibling tests in the file.
    _ = _safe_reason
