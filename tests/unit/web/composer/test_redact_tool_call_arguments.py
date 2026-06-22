"""Tests for redact_tool_call_arguments walker (spec §4.2.6 disposition table).

Disposition-table cells (spec §4.2.6, lines 1015-1023) owned by the arguments
walker:

  • Manifest entry missing for dispatched tool name → AuditIntegrityError.
  • Summarizer raises → telemetry.summarizer_error fires BEFORE
    AuditIntegrityError chained from underlying.
  • Summarizer returns non-str → telemetry.summarizer_error fires BEFORE
    AuditIntegrityError naming the type.
  • Argument summarizer key declared but argument key absent in input → no-op.

(Two further cells, "unknown response key" and "arg fails Pydantic validation",
live on the response walker and the promoted-handler boundary respectively;
they are pinned by ``test_redact_tool_call_response.py`` and the per-tool
handler regression tests at ``test_promote_*.py``.)

Generalised path coverage (Task 8 supersedes Task 4's NotImplementedError
staging boundary for nested paths):

  • Top-level scalar Sensitive (regression: must still substitute as Task 4 did).
  • Nested BaseModel-scalar Sensitive — path like "outer.inner_secret".
  • List-element Sensitive — path like "items[*].secret"; per-element substitution.
  • Dict-element Sensitive — path like "lookup{*}.secret"; per-element substitution.
  • Tuple-element Sensitive — variable-length ``tuple[X, ...]`` form.

Declarative-branch coverage (spec §4.2.6 + §4.3):

  • sensitive_argument_keys with summarizer in argument_summarizers →
    summarizer output substitutes the value.
  • sensitive_argument_keys WITHOUT a summarizer → REDACTED_SENSITIVE_NO_SUMMARIZER
    (spec §4.3 line 1073: "Plain sensitive key → value replaced by literal
    string '<redacted>'.").
  • Non-sensitive keys passthrough.
  • Manifest_dispatch beacon fires once per invocation in BOTH branches.

Walker atomicity (rev-3 W8b / rev-4 W8b): mid-walk raise leaves no partial
dict observable to caller. Canonical sentinel-identity pattern.

Plan task: Phase 2 / Task 8
Spec section: §4.2.6
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel, ConfigDict

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.redaction import (
    REDACTED_SENSITIVE_NO_SUMMARIZER,
    REDACTED_UNKNOWN_ARGUMENT_KEY,
    REDACTED_UNKNOWN_ARGUMENTS_FIELD,
    HandlesNoSensitiveDataReason,
    Sensitive,
    ToolRedaction,
    ToolRedactionPolicy,
    redact_tool_call_arguments,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

# ---------------------------------------------------------------------------
# Helpers (mirror test_redact_tool_call_response.py)
# ---------------------------------------------------------------------------


def _patch_manifest(monkeypatch: pytest.MonkeyPatch, tool_name: str, entry: ToolRedaction) -> None:
    """Extend the module-level MANIFEST with a test entry."""
    from types import MappingProxyType

    import elspeth.web.composer.redaction as _redaction_mod

    new_manifest = MappingProxyType({**_redaction_mod.MANIFEST, tool_name: entry})
    monkeypatch.setattr(_redaction_mod, "MANIFEST", new_manifest)


def _safe_reason() -> HandlesNoSensitiveDataReason:
    return HandlesNoSensitiveDataReason(
        sensitive_data_locations=("no-sensitive-surface",),
        why_arguments_safe="All arguments are structural metadata only; no user content.",
        why_responses_safe="Response contains only structural metadata; no secrets or PII.",
    )


# ---------------------------------------------------------------------------
# Type-driven path: nested-scalar Sensitive (supersedes Task 4 staging boundary)
# ---------------------------------------------------------------------------


def test_type_driven_nested_basemodel_sensitive_field_is_substituted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Sensitive[T] field one level deep inside a nested BaseModel is
    substituted in the redacted output. Path: ``payload.inner_secret``.

    This is the case the Task-4 tracer-bullet raised NotImplementedError for;
    Task 8 generalises it to actual substitution.
    """

    class _InnerModel(BaseModel):
        inner_secret: Annotated[str, Sensitive(summarizer=lambda v: "<sum-inner>")]
        public_field: str

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        payload: _InnerModel
        top_level: str

        model_config = ConfigDict(extra="forbid")

    tool = "t_nested_scalar"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    args = {
        "payload": {"inner_secret": "CANARY_NESTED", "public_field": "shown"},
        "top_level": "also-shown",
    }
    result = redact_tool_call_arguments(tool, args, telemetry=tel)

    assert result["payload"]["inner_secret"] == "<sum-inner>"
    assert result["payload"]["public_field"] == "shown"
    assert result["top_level"] == "also-shown"
    assert "CANARY_NESTED" not in str(result)


# ---------------------------------------------------------------------------
# Type-driven path: list-element Sensitive (per-element substitution)
# ---------------------------------------------------------------------------


def test_type_driven_list_element_sensitive_substitutes_each_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``items: list[Inner]`` where ``Inner.secret`` is Sensitive — every
    element's secret is substituted independently."""

    class _Item(BaseModel):
        secret: Annotated[str, Sensitive(summarizer=lambda v: f"<sum:{v}>")]
        name: str

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        items: list[_Item]

        model_config = ConfigDict(extra="forbid")

    tool = "t_list_element"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    args = {
        "items": [
            {"secret": "ONE", "name": "alpha"},
            {"secret": "TWO", "name": "beta"},
            {"secret": "THREE", "name": "gamma"},
        ]
    }
    result = redact_tool_call_arguments(tool, args, telemetry=tel)

    # The summarizer echoes the raw input back inside its formatted output;
    # to verify each element is independently summarized we assert exact
    # equality (proves both that substitution happened and that the per-leaf
    # value the summarizer saw was the corresponding element's raw secret).
    assert result["items"][0]["secret"] == "<sum:ONE>"
    assert result["items"][1]["secret"] == "<sum:TWO>"
    assert result["items"][2]["secret"] == "<sum:THREE>"
    assert result["items"][0]["name"] == "alpha"
    assert result["items"][1]["name"] == "beta"
    assert result["items"][2]["name"] == "gamma"


def test_type_driven_list_element_sensitive_empty_list_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty list-typed field: no elements, no summarizer calls; passthrough."""

    class _Item(BaseModel):
        secret: Annotated[str, Sensitive(summarizer=lambda v: f"<sum:{v}>")]

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        items: list[_Item]

        model_config = ConfigDict(extra="forbid")

    tool = "t_empty_list"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    result = redact_tool_call_arguments(tool, {"items": []}, telemetry=tel)
    assert result == {"items": []}


# ---------------------------------------------------------------------------
# Type-driven path: dict-element Sensitive (per-element substitution)
# ---------------------------------------------------------------------------


def test_type_driven_dict_element_sensitive_substitutes_each_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``lookup: dict[str, Inner]`` where ``Inner.secret`` is Sensitive — every
    value's secret is substituted independently. Keys are preserved."""

    class _Inner(BaseModel):
        secret: Annotated[str, Sensitive(summarizer=lambda v: f"<sum:{v}>")]
        label: str

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        lookup: dict[str, _Inner]

        model_config = ConfigDict(extra="forbid")

    tool = "t_dict_element"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    args = {
        "lookup": {
            "first": {"secret": "ONE", "label": "alpha"},
            "second": {"secret": "TWO", "label": "beta"},
        }
    }
    result = redact_tool_call_arguments(tool, args, telemetry=tel)

    assert result["lookup"]["first"]["secret"] == "<sum:ONE>"
    assert result["lookup"]["second"]["secret"] == "<sum:TWO>"
    assert result["lookup"]["first"]["label"] == "alpha"
    assert result["lookup"]["second"]["label"] == "beta"


# ---------------------------------------------------------------------------
# Type-driven path: tuple-element Sensitive (variable-length form)
# ---------------------------------------------------------------------------


def test_type_driven_tuple_element_sensitive_substitutes_each_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pair: tuple[Inner, ...]`` — variable-length tuple form; per-element
    substitution. (Fixed-length tuples raise ValueError at walker construction;
    that case is owned by test_walk_model_schema.py.)
    """

    class _Inner(BaseModel):
        secret: Annotated[str, Sensitive(summarizer=lambda v: f"<sum:{v}>")]

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        pair: tuple[_Inner, ...]

        model_config = ConfigDict(extra="forbid")

    tool = "t_tuple_element"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    args = {"pair": [{"secret": "A"}, {"secret": "B"}]}
    result = redact_tool_call_arguments(tool, args, telemetry=tel)

    # pydantic coerces list → tuple at model_validate; model_dump yields a list
    # by default. Either form is acceptable so long as both elements substituted.
    elements = list(result["pair"])
    assert elements[0]["secret"] == "<sum:A>"
    assert elements[1]["secret"] == "<sum:B>"


# ---------------------------------------------------------------------------
# Type-driven path: top-level Sensitive (regression for Task 4 path)
# ---------------------------------------------------------------------------


def test_type_driven_top_level_sensitive_field_with_no_summarizer_substitutes_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level Sensitive[T] without summarizer → REDACTED_SENSITIVE_NO_SUMMARIZER.

    Regression: Task 4's set_source flow already covers the summarizer case at
    the top level; this pins the no-summarizer case symmetrically.
    """

    class _ArgModel(BaseModel):
        secret: Annotated[str, Sensitive()]  # no summarizer
        plain: str

        model_config = ConfigDict(extra="forbid")

    tool = "t_top_level_no_summarizer"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    result = redact_tool_call_arguments(tool, {"secret": "X", "plain": "Y"}, telemetry=tel)
    assert result["secret"] == REDACTED_SENSITIVE_NO_SUMMARIZER
    assert result["plain"] == "Y"


# ---------------------------------------------------------------------------
# Declarative branch
# ---------------------------------------------------------------------------


def test_declarative_sensitive_key_with_summarizer_uses_summarizer_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declarative entry: a key in ``sensitive_argument_keys`` that also has a
    summarizer in ``argument_summarizers`` uses the summarizer's str output."""
    tool = "t_decl_with_summarizer"
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        known_response_keys=("status",),
        argument_summarizers={"path": lambda v: f"<sum:{v}>"},
    )
    entry = ToolRedaction(policy=policy)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    result = redact_tool_call_arguments(tool, {"path": "/secret", "name": "ok"}, telemetry=tel)

    assert result["path"] == "<sum:/secret>"
    assert result["name"] == "ok"
    assert tel.summarizer_error_calls == []


def test_declarative_sensitive_key_without_summarizer_uses_no_summarizer_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §4.3 line 1073: "Plain sensitive key → value replaced by literal
    string '<redacted>'." A declarative ``sensitive_argument_keys`` entry with
    no summarizer in ``argument_summarizers`` substitutes
    REDACTED_SENSITIVE_NO_SUMMARIZER. Mirrors the response walker's behaviour
    at the same line of the implementation."""
    tool = "t_decl_no_summarizer"
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        known_response_keys=("status",),
        # No argument_summarizers — "plain sensitive key" case.
    )
    entry = ToolRedaction(policy=policy)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    result = redact_tool_call_arguments(tool, {"path": "/secret", "name": "ok"}, telemetry=tel)

    assert result["path"] == REDACTED_SENSITIVE_NO_SUMMARIZER
    assert result["name"] == "ok"


def test_declarative_sensitive_key_absent_in_input_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §4.2.6 disposition table: "Argument summarizer key declared but
    argument key absent in input → No-op (key absence is not a fault)."

    Tier-3 input may omit any key. The declarative walker iterates the policy
    keys, not the input keys; absent input keys are simply not substituted and
    not synthesized.
    """
    tool = "t_decl_absent_key"
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=("path", "blob_id"),
        known_response_keys=("status",),
        argument_summarizers={
            "path": lambda v: "<should-not-fire>",
            "blob_id": lambda v: "<should-not-fire>",
        },
    )
    entry = ToolRedaction(policy=policy)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    # Neither sensitive key present in input.
    result = redact_tool_call_arguments(tool, {"name": "only-present-key"}, telemetry=tel)
    assert result == {"name": "only-present-key"}
    assert tel.summarizer_error_calls == []


def test_declarative_non_sensitive_keys_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §4.2.6: declarative arg walker covers ONLY ``sensitive_argument_keys``.
    Keys NOT in that set are passthrough — unlike the response walker, where
    unknown keys are fail-closed sentinelised. Argument inputs are
    LLM-supplied and the manifest enumerates the sensitive surface explicitly;
    enumerating ``known_argument_keys`` is not part of the policy contract.
    """
    tool = "t_decl_passthrough"
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        known_response_keys=("status",),
        argument_summarizers={"path": lambda v: "<sum>"},
    )
    entry = ToolRedaction(policy=policy)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    result = redact_tool_call_arguments(
        tool,
        {"path": "/secret", "name": "alpha", "count": 42},
        telemetry=tel,
    )
    # Sensitive key is substituted; non-sensitive keys passthrough.
    assert result["path"] == "<sum>"
    assert result["name"] == "alpha"
    assert result["count"] == 42


def test_advisor_declarative_unknown_argument_keys_are_sentinelised() -> None:
    """Advisor persists LLM args; unknown keys must not pass through raw."""
    tel = NoopRedactionTelemetry()
    raw_extra_context = "RAW_EXTRA_CONTEXT: private traceback and schema excerpt"

    result = redact_tool_call_arguments(
        "request_advisor_hint",
        {
            "trigger": "proactive_security_safety",
            "problem_summary": "stuck on provider options",
            "recent_errors": ["validator echoed a private column"],
            "attempted_actions": ["set_pipeline with sensitive options"],
            "full_context": raw_extra_context,
        },
        telemetry=tel,
    )

    assert result["problem_summary"].startswith("<advisor-problem-summary:")
    assert "full_context" not in result
    assert result["_unknown_arguments"] == "<redacted-unknown-argument-key>"
    assert raw_extra_context not in str(result)


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("list_blobs", {}),
        ("list_composer_blobs", {}),
        ("get_blob_metadata", {"blob_id": "11111111-1111-4111-8111-111111111111"}),
        ("inspect_source", {"blob_id": "11111111-1111-4111-8111-111111111111"}),
    ],
)
def test_blob_discovery_unknown_argument_keys_are_sentinelised(
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    """Blob discovery arguments are persisted; extra LLM-authored keys must not pass through raw."""
    tel = NoopRedactionTelemetry()
    raw_payload = "RAW_EXTRA_BLOB_DISCOVERY_PAYLOAD: uploaded CSV excerpt or PII"

    result = redact_tool_call_arguments(
        tool_name,
        {**arguments, "note": raw_payload},
        telemetry=tel,
    )

    assert "note" not in result
    assert result[REDACTED_UNKNOWN_ARGUMENTS_FIELD] == REDACTED_UNKNOWN_ARGUMENT_KEY
    assert raw_payload not in str(result)
    for key, value in arguments.items():
        assert result[key] == value


@pytest.mark.parametrize(
    ("tool_name", "arguments", "passthrough_keys", "summarized_keys"),
    [
        ("clear_source", {"source_name": "source"}, ("source_name",), ()),
        ("remove_node", {"id": "normalize_rows"}, ("id",), ()),
        ("remove_edge", {"id": "e_fetch_to_clean"}, ("id",), ()),
        (
            "upsert_edge",
            {
                "id": "e_fetch_to_clean",
                "from_node": "fetch",
                "to_node": "clean",
                "edge_type": "on_success",
                "label": "clean rows",
            },
            ("id", "from_node", "to_node", "edge_type", "label"),
            (),
        ),
        ("remove_output", {"sink_name": "records_out"}, ("sink_name",), ()),
        (
            "upsert_node",
            {
                "id": "normalize_rows",
                "node_type": "transform",
                "input": "raw_rows",
                "plugin": "normalize",
                "on_success": "clean_rows",
                "on_error": "failed_rows",
                "options": {"path": "INNER_MUTATION_VALUE"},
                "condition": "row['enabled']",
                "routes": {"true": "clean_rows", "false": "discard"},
                "fork_to": ["audit_rows"],
                "branches": {"left": "clean_rows", "right": "audit_rows"},
                "policy": "all",
                "merge": "prefer_left",
                "trigger": {"count": 10},
                "output_mode": "transform",
                "expected_output_count": 1,
            },
            (
                "id",
                "node_type",
                "input",
                "plugin",
                "on_success",
                "on_error",
                "condition",
                "fork_to",
                "branches",
                "policy",
                "merge",
                "output_mode",
                "expected_output_count",
            ),
            ("options", "routes", "trigger"),
        ),
        ("set_metadata", {"patch": {"name": "pipeline", "description": "INNER_MUTATION_VALUE"}}, (), ("patch",)),
        (
            "set_output",
            {
                "sink_name": "records_out",
                "plugin": "json",
                "options": {"path": "INNER_MUTATION_VALUE"},
                "on_write_failure": "discard",
            },
            ("sink_name", "plugin", "on_write_failure"),
            ("options",),
        ),
    ],
)
def test_mutation_declarative_unknown_argument_keys_are_sentinelised(
    tool_name: str,
    arguments: dict[str, object],
    passthrough_keys: tuple[str, ...],
    summarized_keys: tuple[str, ...],
) -> None:
    """Mutation tool args are persisted; unexpected LLM keys must fail closed."""
    tel = NoopRedactionTelemetry()
    raw_payload = "RAW_EXTRA_MUTATION_ARGUMENT_PAYLOAD"

    result = redact_tool_call_arguments(
        tool_name,
        {**arguments, "unauthorized_payload": raw_payload},
        telemetry=tel,
    )

    assert "unauthorized_payload" not in result
    assert result[REDACTED_UNKNOWN_ARGUMENTS_FIELD] == REDACTED_UNKNOWN_ARGUMENT_KEY
    assert raw_payload not in str(result)

    for key in passthrough_keys:
        assert result[key] == arguments[key]
    for key in summarized_keys:
        assert key in result
        assert result[key] != arguments[key]


def test_set_metadata_patch_summary_redacts_unknown_patch_key_names() -> None:
    """Nested metadata patch keys are LLM-controlled and must not echo raw."""
    tel = NoopRedactionTelemetry()
    raw_key = "sk-live-secret-as-patch-key"

    result = redact_tool_call_arguments(
        "set_metadata",
        {"patch": {"name": "pipeline", raw_key: "value"}},
        telemetry=tel,
    )

    assert result["patch"] == "<metadata-patch:name,unknown>"
    assert raw_key not in str(result)


# ---------------------------------------------------------------------------
# Failure modes: missing manifest, summarizer raises, summarizer non-str
# ---------------------------------------------------------------------------


def test_missing_manifest_entry_yields_audit_integrity_error() -> None:
    """Spec §4.2.6: "Manifest entry missing for dispatched tool name →
    AuditIntegrityError" (registry-consistency invariant; distinct from
    Tier-3 LLM-hallucinated tool name which is caught earlier in the
    dispatcher).

    The Task-4 tracer-bullet raised plain KeyError here; Task 8 escalates to
    AuditIntegrityError so the audit-write block treats it consistently with
    other Tier-1 violations.
    """
    tel = NoopRedactionTelemetry()
    with pytest.raises(AuditIntegrityError):
        redact_tool_call_arguments("not_in_manifest", {"x": 1}, telemetry=tel)


def test_type_driven_summarizer_raise_fires_telemetry_then_audit_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Type-driven branch: summarizer that raises → walker fires
    summarizer_error counter BEFORE raising AuditIntegrityError chained from
    the underlying exception (rev-2 M.8)."""

    def _crashing(v: Any) -> str:
        raise RuntimeError("boom-args")

    class _ArgModel(BaseModel):
        token: Annotated[str, Sensitive(summarizer=_crashing)]

        model_config = ConfigDict(extra="forbid")

    tool = "t_args_summarizer_raise"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    with pytest.raises(AuditIntegrityError) as exc_info:
        redact_tool_call_arguments(tool, {"token": "SECRET"}, telemetry=tel)

    assert tel.summarizer_error_calls == [{"tool_name": tool}]
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom-args"


def test_type_driven_summarizer_non_str_return_fires_telemetry_then_audit_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Type-driven branch: summarizer that returns non-str → walker fires
    summarizer_error counter BEFORE raising AuditIntegrityError. Message
    names the actual returned type."""

    def _bad_return(v: Any) -> Any:
        return {"x": 1}

    class _ArgModel(BaseModel):
        token: Annotated[str, Sensitive(summarizer=_bad_return)]

        model_config = ConfigDict(extra="forbid")

    tool = "t_args_summarizer_non_str"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    with pytest.raises(AuditIntegrityError) as exc_info:
        redact_tool_call_arguments(tool, {"token": "SECRET"}, telemetry=tel)

    assert tel.summarizer_error_calls == [{"tool_name": tool}]
    assert "dict" in str(exc_info.value)


def test_declarative_summarizer_raise_fires_telemetry_then_audit_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declarative branch: summarizer that raises → walker fires
    summarizer_error counter BEFORE raising AuditIntegrityError chained
    from the underlying exception (rev-2 M.8, symmetric to type-driven)."""

    def _crashing(v: Any) -> str:
        raise RuntimeError("boom-decl")

    tool = "t_decl_summarizer_raise"
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        known_response_keys=("status",),
        argument_summarizers={"path": _crashing},
    )
    entry = ToolRedaction(policy=policy)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    with pytest.raises(AuditIntegrityError) as exc_info:
        redact_tool_call_arguments(tool, {"path": "/secret"}, telemetry=tel)

    assert tel.summarizer_error_calls == [{"tool_name": tool}]
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom-decl"


def test_declarative_summarizer_non_str_return_fires_telemetry_then_audit_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declarative branch: summarizer that returns non-str → walker fires
    summarizer_error counter BEFORE raising AuditIntegrityError."""

    def _bad_return(v: Any) -> Any:
        return 42  # non-str

    tool = "t_decl_summarizer_non_str"
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        known_response_keys=("status",),
        argument_summarizers={"path": _bad_return},
    )
    entry = ToolRedaction(policy=policy)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    with pytest.raises(AuditIntegrityError) as exc_info:
        redact_tool_call_arguments(tool, {"path": "/secret"}, telemetry=tel)

    assert tel.summarizer_error_calls == [{"tool_name": tool}]
    assert "int" in str(exc_info.value)


# ---------------------------------------------------------------------------
# manifest_dispatch beacon emission in both branches
# ---------------------------------------------------------------------------


def test_manifest_dispatch_fires_once_per_invocation_type_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §4.2.4: manifest_dispatch beacon fires once per invocation in the
    type-driven branch (rev-2 M_telemetry_implementation parity)."""

    class _ArgModel(BaseModel):
        name: str

        model_config = ConfigDict(extra="forbid")

    tool = "t_dispatch_type_driven_args"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    redact_tool_call_arguments(tool, {"name": "x"}, telemetry=tel)
    assert tel.manifest_dispatch_calls == [{"tool_name": tool, "shape": "type_driven"}]


def test_manifest_dispatch_fires_once_per_invocation_declarative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §4.2.4: manifest_dispatch beacon fires once per invocation in the
    declarative branch."""
    tool = "t_dispatch_declarative_args"
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        known_response_keys=("status",),
        argument_summarizers={"path": lambda v: "<sum>"},
    )
    entry = ToolRedaction(policy=policy)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    redact_tool_call_arguments(tool, {"path": "/x"}, telemetry=tel)
    assert tel.manifest_dispatch_calls == [{"tool_name": tool, "shape": "declarative"}]


def test_manifest_dispatch_fires_for_declarative_handles_no_sensitive_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Beacon must fire for a declarative entry even when the policy declares
    no sensitive data (handles_no_sensitive_data=True). The dispatch happened;
    operational dashboards depend on the count for both shapes equally."""
    tool = "t_dispatch_no_sensitive_data"
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=(),
        known_response_keys=(),
        handles_no_sensitive_data=True,
        handles_no_sensitive_data_reason_struct=_safe_reason(),
    )
    entry = ToolRedaction(policy=policy)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    result = redact_tool_call_arguments(tool, {"any": "key"}, telemetry=tel)
    # No sensitive keys declared; everything passthrough.
    assert result == {"any": "key"}
    assert tel.manifest_dispatch_calls == [{"tool_name": tool, "shape": "declarative"}]


# ---------------------------------------------------------------------------
# Walker atomicity — mid-walk raise leaves no partial dict observable
# ---------------------------------------------------------------------------


def test_walker_atomicity_on_mid_walk_raise_type_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-walk raise in the type-driven branch leaves no partial dict
    observable to the caller. Mirrors the response walker's canonical
    sentinel-identity pattern (rev-3 W8b / rev-4 W8b).
    """
    call_log: list[str] = []

    def _ok(v: Any) -> str:
        call_log.append("ok")
        return "<ok>"

    def _crashing(v: Any) -> str:
        call_log.append("crash")
        raise RuntimeError("mid-walk crash")

    class _ArgModel(BaseModel):
        key_a: Annotated[str, Sensitive(summarizer=_ok)]
        key_b: Annotated[str, Sensitive(summarizer=_crashing)]
        key_c: str

        model_config = ConfigDict(extra="forbid")

    tool = "t_atomicity_args"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    sentinel = object()
    result: Any = sentinel
    with pytest.raises(AuditIntegrityError):
        result = redact_tool_call_arguments(
            tool,
            {"key_a": "v_a", "key_b": "v_b", "key_c": "v_c"},
            telemetry=tel,
        )
    assert result is sentinel
    assert "ok" in call_log
    assert "crash" in call_log
    assert tel.summarizer_error_calls == [{"tool_name": tool}]


def test_walker_atomicity_on_mid_walk_raise_declarative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declarative-branch atomicity: mid-walk raise → no partial dict observable."""

    def _crashing(v: Any) -> str:
        raise RuntimeError("decl mid-walk crash")

    tool = "t_atomicity_decl"
    policy = ToolRedactionPolicy(
        sensitive_argument_keys=("path",),
        known_response_keys=("status",),
        argument_summarizers={"path": _crashing},
    )
    entry = ToolRedaction(policy=policy)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    sentinel = object()
    result: Any = sentinel
    with pytest.raises(AuditIntegrityError):
        result = redact_tool_call_arguments(
            tool,
            {"path": "/x", "name": "y"},
            telemetry=tel,
        )
    assert result is sentinel
    assert tel.summarizer_error_calls == [{"tool_name": tool}]


def test_counter_fires_once_when_first_list_element_summarizer_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Container path: when the first list element's summarizer crashes,
    iteration stops at first failure; counter fires exactly once.

    Pins the natural raise-on-first-failure pattern so a future refactor that
    accidentally continues past a raise would change the counter count."""

    def _crashing(v: Any) -> str:
        raise RuntimeError("crash-on-first")

    class _Item(BaseModel):
        secret: Annotated[str, Sensitive(summarizer=_crashing)]

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        items: list[_Item]

        model_config = ConfigDict(extra="forbid")

    tool = "t_first_element_crash"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    with pytest.raises(AuditIntegrityError):
        redact_tool_call_arguments(
            tool,
            {"items": [{"secret": "A"}, {"secret": "B"}]},
            telemetry=tel,
        )

    assert len(tel.summarizer_error_calls) == 1


# ---------------------------------------------------------------------------
# manifest_dispatch fires BEFORE model_validate (Tier-3 ValidationError still
# records dispatch happened)
# ---------------------------------------------------------------------------


def test_manifest_dispatch_emitted_before_pydantic_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Tier-3 ValidationError must NOT prevent the manifest_dispatch beacon
    from firing. The dispatch is the event being counted; the validation
    outcome is downstream of that fact.

    Spec §4.2.6 (Tier-3 → Tier-1 boundary): the walker is called with parsed
    arguments, then validates against ``argument_model``. The handler catches
    ``ValidationError`` and re-raises as ToolArgumentError; the dispatch
    beacon is unconditional on success/failure of validation.
    """

    class _ArgModel(BaseModel):
        name: str

        model_config = ConfigDict(extra="forbid")

    tool = "t_dispatch_before_validate"
    entry = ToolRedaction(argument_model=_ArgModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    # Missing required field "name" → ValidationError.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        redact_tool_call_arguments(tool, {}, telemetry=tel)

    # The dispatch happened regardless of validation outcome.
    assert tel.manifest_dispatch_calls == [{"tool_name": tool, "shape": "type_driven"}]
