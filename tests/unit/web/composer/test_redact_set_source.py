"""Tracer-bullet: set_source end-to-end through manifest + walker (spec §11).

These tests pin the integration shape established in Task 4 of the Phase 2
redaction plan.  Tasks 13/14/15 replicate the same shape for other tools,
so the assertions here are load-bearing for the bulk-promotion wave.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from elspeth.web.composer.redaction import (
    MANIFEST,
    REDACTED_BLOB_SOURCE_PATH,
    SetSourceArgumentsModel,
    _summarize_set_source_options,
    redact_tool_call_arguments,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry


def test_set_source_manifest_entry_is_type_driven() -> None:
    entry = MANIFEST["set_source"]
    assert entry.argument_model is SetSourceArgumentsModel
    assert entry.policy is None


def test_set_source_argument_model_validates_real_llm_shape() -> None:
    llm_args = {
        "plugin": "csv",
        "options": {"path": "/tmp/data.csv", "header": True},
        "on_success": "rows",
        "on_validation_failure": "discard",
    }
    validated = SetSourceArgumentsModel.model_validate(llm_args)
    assert validated.plugin == "csv"
    assert validated.options == {"path": "/tmp/data.csv", "header": True}
    assert validated.on_success == "rows"
    assert validated.on_validation_failure == "discard"


def test_set_source_argument_model_rejects_missing_required() -> None:
    with pytest.raises(ValidationError):
        SetSourceArgumentsModel.model_validate({})


def test_set_source_argument_model_rejects_wrong_type() -> None:
    with pytest.raises(ValidationError):
        SetSourceArgumentsModel.model_validate(
            {
                "plugin": 42,
                "options": {},
                "on_success": "rows",
                "on_validation_failure": "discard",
            }
        )


def test_set_source_argument_model_rejects_extra_fields() -> None:
    """rev-2 M.1: extra='forbid' prevents argument_canonical/walker drift.

    Without this, a stray ``inline_blob`` or ``label`` field would be
    silently accepted by Pydantic but unrecorded in the walker schema —
    breaking the manifest/canonical-arguments parity invariant the
    adequacy guard relies on.
    """
    with pytest.raises(ValidationError):
        SetSourceArgumentsModel.model_validate(
            {
                "plugin": "csv",
                "options": {"path": "/tmp/x.csv"},
                "on_success": "rows",
                "on_validation_failure": "discard",
                "inline_blob": {"foo": "bar"},  # not a set_source field
            }
        )


def test_redact_substitutes_options_via_summarizer() -> None:
    """Sensitive[options] is replaced by the summarizer string at the top level.

    The summarizer returns canonical JSON of the redacted options dict.
    Because Sensitive() substitutes the ENTIRE marked value, the top-level
    ``options`` slot in the redacted output is a string (the summarizer
    return), not a dict.  This is the load-bearing shape contract: the
    persistence boundary receives a scalar where a dict would otherwise
    sit.
    """
    tel = NoopRedactionTelemetry()
    args = {
        "plugin": "csv",
        "options": {"path": "/internal/blob/path.csv", "blob_ref": "abc"},
        "on_success": "rows",
        "on_validation_failure": "discard",
    }
    redacted = redact_tool_call_arguments("set_source", args, telemetry=tel)
    assert redacted["plugin"] == "csv"
    assert redacted["on_success"] == "rows"
    assert redacted["on_validation_failure"] == "discard"
    # Sensitive substitution: options is now the summarizer's str output.
    assert isinstance(redacted["options"], str)
    # blob_ref triggered redact_source_storage_path → path is the sentinel.
    assert REDACTED_BLOB_SOURCE_PATH in redacted["options"]
    # The original internal path MUST NOT appear in the summary.
    assert "/internal/blob/path.csv" not in redacted["options"]
    # Telemetry recorded the manifest dispatch with the type-driven shape.
    assert tel.manifest_dispatch_calls == [{"tool_name": "set_source", "shape": "type_driven"}]


def test_redact_passes_through_when_no_blob_ref() -> None:
    """Without blob_ref, redact_source_storage_path is a no-op on options.

    The Sensitive substitution still happens (options becomes the
    summarizer's str return), but the path inside the JSON-encoded summary
    is the original path verbatim.
    """
    tel = NoopRedactionTelemetry()
    args = {
        "plugin": "csv",
        "options": {"path": "/tmp/data.csv"},
        "on_success": "rows",
        "on_validation_failure": "discard",
    }
    redacted = redact_tool_call_arguments("set_source", args, telemetry=tel)
    assert isinstance(redacted["options"], str)
    assert "/tmp/data.csv" in redacted["options"]
    assert REDACTED_BLOB_SOURCE_PATH not in redacted["options"]


def test_summarize_set_source_options_accepts_coerced_datetime() -> None:
    """Pin rev-3 A7: summarizer MUST NOT raise on reachable input values.

    Spec §9 RSK-03 requires the summarizer not raise on any reachable
    input value.  Pydantic 2.x can coerce string-like inputs to
    :class:`datetime` when the field accepts ``Any``; :func:`json.dumps`
    raises :class:`TypeError` on ``datetime`` unless ``default=str`` is
    supplied.  This test pins the ``default=str`` argument so a future
    refactor that removes it fails loudly here rather than silently
    violating RSK-03.
    """
    options = {"since": datetime(2026, 1, 1, tzinfo=UTC), "key": "v"}
    result = _summarize_set_source_options(options)
    assert isinstance(result, str)


_CANARY = "CANARY-SENSITIVE-PATH-DO-NOT-LEAK"


def test_serialization_boundary_canary_not_in_json_output() -> None:
    """Pin the Phase 3 cross-boundary integration contract (rev-2 BLOCKER_A).

    Phase 3 passes the result of :func:`redact_tool_call_arguments` through
    :func:`json.dumps` before writing to ``chat_messages.tool_calls``.  This
    test verifies the canary never survives that serialization — even
    though :func:`json.dumps` would otherwise re-emit the canary if it
    appeared anywhere in the dict.  Because the summarizer wraps the
    canary inside ``redact_source_storage_path``, and because the only
    sensitive options field that gets sentinel-replaced is ``path`` when
    ``blob_ref`` is also present, the canary is exposed in the absence of
    ``blob_ref``.  We therefore include ``blob_ref`` so the canary value
    is genuinely substituted.
    """
    args = {
        "plugin": "csv",
        "options": {"path": _CANARY, "blob_ref": "abc123"},
        "on_success": "rows",
        "on_validation_failure": "discard",
    }
    result = redact_tool_call_arguments("set_source", args, telemetry=NoopRedactionTelemetry())
    serialized = json.dumps(result, sort_keys=True)
    assert _CANARY not in serialized, (
        "Sensitive canary value appeared in serialized output. "
        "Redaction did not remove it from the persistence path. "
        f"Serialized: {serialized!r}"
    )
    assert "options" in serialized  # key preserved, value redacted
