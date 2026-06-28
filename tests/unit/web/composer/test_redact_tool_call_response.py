"""Tests for redact_tool_call_response walker (spec §4.2.4, §4.2.6).

Covers:
- Known/unknown/sensitive key dispatch (declarative entry path)
- Type-driven response_model path (walk_model_schema)
- Fixed sentinel '<redacted-unknown-response-key>' (W6)
- Summarizer crash discipline (M2/W5/M.8)
- Walker atomicity (W8b)
- Missing manifest entry (registry consistency)

Plan task: Phase 2 / Task 7
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, ConfigDict

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.tier_registry import _TIER_1_ERRORS_VIEW
from elspeth.web.composer.redaction import (
    REDACTED_UNKNOWN_RESPONSE_KEY,
    HandlesNoSensitiveDataReason,
    Sensitive,
    ToolRedaction,
    ToolRedactionPolicy,
    redact_tool_call_response,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

# ---------------------------------------------------------------------------
# Helpers: construct temporary manifest-like entries for test isolation.
# We cannot mutate the module-level MANIFEST (it's a MappingProxyType) so
# tests that need non-set_source tools must patch MANIFEST or call the
# function with a name already in MANIFEST (set_source).
#
# Strategy: use monkeypatch to temporarily extend MANIFEST with test entries.
# ---------------------------------------------------------------------------


def _declarative_entry(
    *,
    sensitive_response_keys: tuple[str, ...] = (),
    known_response_keys: tuple[str, ...] = (),
    argument_summarizers: dict[str, Any] | None = None,
    response_summarizers: dict[str, Any] | None = None,
    handles_no_sensitive_data: bool = False,
    handles_no_sensitive_data_reason_struct: HandlesNoSensitiveDataReason | None = None,
) -> ToolRedaction:
    """Build a declarative ToolRedaction for test fixtures."""
    policy = ToolRedactionPolicy(
        sensitive_response_keys=sensitive_response_keys,
        known_response_keys=known_response_keys,
        argument_summarizers=argument_summarizers or {},
        handles_no_sensitive_data=handles_no_sensitive_data,
        handles_no_sensitive_data_reason_struct=handles_no_sensitive_data_reason_struct,
    )
    return ToolRedaction(policy=policy)


def _safe_reason() -> HandlesNoSensitiveDataReason:
    return HandlesNoSensitiveDataReason(
        sensitive_data_locations=("no-sensitive-surface",),
        why_arguments_safe="All arguments are structural metadata only; no user content.",
        why_responses_safe="Response contains only structural metadata; no secrets or PII.",
    )


def _patch_manifest(monkeypatch: pytest.MonkeyPatch, tool_name: str, entry: ToolRedaction) -> None:
    """Extend the module-level MANIFEST with a test entry.

    Builds a new dict from the existing proxy, adds the test entry, and
    replaces the module binding.  Monkeypatch restores the original value
    on teardown.
    """
    from types import MappingProxyType

    import elspeth.web.composer.redaction as _redaction_mod

    new_manifest = MappingProxyType({**_redaction_mod.MANIFEST, tool_name: entry})
    monkeypatch.setattr(_redaction_mod, "MANIFEST", new_manifest)


# ---------------------------------------------------------------------------
# Test 1: all keys known, none sensitive → passthrough
# ---------------------------------------------------------------------------


def test_passthrough_when_all_keys_known_and_none_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declarative entry: known_response_keys covers all response keys; none
    are sensitive → every value passes through unchanged."""
    tool = "t_passthrough"
    entry = _declarative_entry(
        sensitive_response_keys=(),
        known_response_keys=("status", "count"),
    )
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    response = {"status": "ok", "count": 42}
    result = redact_tool_call_response(tool, response, telemetry=tel)

    assert result == {"status": "ok", "count": 42}
    assert tel.unknown_response_key_calls == []
    assert tel.summarizer_error_calls == []


# ---------------------------------------------------------------------------
# Test 2: sensitive key, no summarizer → REDACTED sentinel
# ---------------------------------------------------------------------------


def test_sensitive_key_without_summarizer_becomes_redacted_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declarative entry: a sensitive_response_key with no summarizer is
    substituted with the no-summarizer sentinel (not the unknown-key sentinel).

    Declarative entries have no response_summarizers; the only available
    substitution for a declarative sensitive key is the sentinel.
    """
    from elspeth.web.composer.redaction import REDACTED_SENSITIVE_NO_SUMMARIZER

    tool = "t_no_summarizer"
    entry = _declarative_entry(
        sensitive_response_keys=("secret",),
        known_response_keys=("status", "secret"),
    )
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    response = {"status": "ok", "secret": "CANARY_SECRET"}
    result = redact_tool_call_response(tool, response, telemetry=tel)

    assert result["status"] == "ok"
    assert result["secret"] == REDACTED_SENSITIVE_NO_SUMMARIZER
    assert "CANARY_SECRET" not in result.values()
    # Unknown-key counter must NOT fire (this is a known sensitive key, not unknown)
    assert tel.unknown_response_key_calls == []


# ---------------------------------------------------------------------------
# Test 3: type-driven response_model, Sensitive field has summarizer
# ---------------------------------------------------------------------------


def test_sensitive_key_with_summarizer_uses_summarizer_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Type-driven entry with response_model: Sensitive field with a summarizer
    uses the summarizer output rather than the raw value or sentinel."""

    class _ResponseModel(BaseModel):
        status: str
        token: Annotated[str, Sensitive(summarizer=lambda v: f"<summarized:{len(v)}>")]

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        query: str

        model_config = ConfigDict(extra="forbid")

    tool = "t_with_summarizer"
    entry = ToolRedaction(argument_model=_ArgModel, response_model=_ResponseModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    response = {"status": "ok", "token": "SECRETTOKEN"}
    result = redact_tool_call_response(tool, response, telemetry=tel)

    assert result["status"] == "ok"
    assert result["token"] == "<summarized:11>"
    assert "SECRETTOKEN" not in str(result)


# ---------------------------------------------------------------------------
# Test 4: unknown key → fixed sentinel + telemetry counter
# ---------------------------------------------------------------------------


def test_unknown_key_becomes_fixed_sentinel_with_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declarative entry: a key in the response that is neither in
    sensitive_response_keys nor known_response_keys is substituted with
    REDACTED_UNKNOWN_RESPONSE_KEY (exact string equality) and increments
    the unknown_response_key_redacted counter once per unknown key."""
    tool = "t_unknown_key"
    entry = _declarative_entry(
        sensitive_response_keys=(),
        known_response_keys=("status",),
    )
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    response = {"status": "ok", "unknown_field": "MYSTERY"}
    result = redact_tool_call_response(tool, response, telemetry=tel)

    assert result["status"] == "ok"
    assert result["unknown_field"] == REDACTED_UNKNOWN_RESPONSE_KEY
    # Exact string equality, not regex/prefix
    assert result["unknown_field"] == "<redacted-unknown-response-key>"
    # Counter fires once per unknown key
    assert tel.unknown_response_key_calls == [{"tool_name": tool}]


def test_unknown_key_counter_fires_once_per_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two unknown keys → counter fires twice (once per key)."""
    tool = "t_two_unknowns"
    entry = _declarative_entry(
        sensitive_response_keys=(),
        known_response_keys=("status",),
    )
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    response = {"status": "ok", "x": 1, "y": 2}
    result = redact_tool_call_response(tool, response, telemetry=tel)

    assert result["x"] == REDACTED_UNKNOWN_RESPONSE_KEY
    assert result["y"] == REDACTED_UNKNOWN_RESPONSE_KEY
    assert len(tel.unknown_response_key_calls) == 2


# ---------------------------------------------------------------------------
# Test 5: type-driven response_model walks via walk_model_schema
# ---------------------------------------------------------------------------


def test_type_driven_response_model_walks_via_schema_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Type-driven entry with response_model: Sensitive[T] fields on the
    model are substituted; non-sensitive fields pass through."""

    class _ResponseModel(BaseModel):
        status: str
        api_key: Annotated[str, Sensitive()]  # no summarizer → sentinel

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        query: str

        model_config = ConfigDict(extra="forbid")

    from elspeth.web.composer.redaction import REDACTED_SENSITIVE_NO_SUMMARIZER

    tool = "t_type_driven_response"
    entry = ToolRedaction(argument_model=_ArgModel, response_model=_ResponseModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    response = {"status": "healthy", "api_key": "sk-super-secret"}
    result = redact_tool_call_response(tool, response, telemetry=tel)

    assert result["status"] == "healthy"
    assert result["api_key"] == REDACTED_SENSITIVE_NO_SUMMARIZER
    assert "sk-super-secret" not in str(result)


def test_get_blob_content_redacts_tool_result_envelope_content() -> None:
    """``get_blob_content`` redacts nested ``data.content`` in ToolResult."""

    response = {
        "success": True,
        "validation": {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "semantic_contracts": [],
            "graph_repair_suggestions": [],
        },
        "affected_nodes": [],
        "version": 2,
        "data": {
            "blob_id": "blob-1",
            "filename": "input.csv",
            "mime_type": "text/csv",
            "content": "secret,row\n1,2\n",
            "truncated": False,
            "size_bytes": 15,
        },
    }

    result = redact_tool_call_response("get_blob_content", response, telemetry=NoopRedactionTelemetry())

    assert result["data"]["content"] == "<blob-content:15-bytes>"
    assert "secret,row" not in str(result)


def test_get_blob_content_redacts_tool_result_failure_envelope() -> None:
    """Recoverable failures have no content leaf and must not crash redaction."""

    response = {
        "success": False,
        "validation": {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "semantic_contracts": [],
            "graph_repair_suggestions": [],
        },
        "affected_nodes": [],
        "version": 2,
        "data": {"error": "Blob 'blob-1' not found."},
    }

    result = redact_tool_call_response("get_blob_content", response, telemetry=NoopRedactionTelemetry())

    assert result == response


def test_get_blob_content_populated_validation_envelope_redacts_only_repair_arguments() -> None:
    """A fully-populated validation envelope round-trips with only ``arguments`` redacted.

    Mechanically pins that ``GetBlobContentValidationModel`` (and its nested
    shadow models) matches the real ``ToolResult.to_dict()`` validation-envelope
    shape — every key produced by ``_semantic_contracts_payload`` and
    ``_graph_repair_suggestions`` in tools/_common.py must be accepted by the
    ``extra="forbid"`` shadow models. The other fixtures use empty lists and
    never exercise a populated ``semantic_contracts`` / ``graph_repair_suggestions``
    element, so a key-name drift between the builder and the shadow model would
    pass them silently; this test fails loudly on such drift.

    It also confirms the ONE Sensitive leaf — the heterogeneous repair-tool-call
    ``arguments`` mapping — is summarized to the structural sketch while every
    surrounding structural scalar (component, message, severity, edge-contract
    fields, repair codes, affected-consumer ids) passes through verbatim. The
    non-sensitive validation metadata MUST survive redaction; only ``arguments``
    is collapsed.
    """
    response = {
        "success": True,
        "validation": {
            "is_valid": False,
            "errors": [
                {"component": "connection:shared", "message": "Duplicate consumer for connection shared", "severity": "high"},
            ],
            "warnings": [
                {"component": "node:t1", "message": "Observed schema in use", "severity": "low"},
            ],
            "suggestions": [
                {"component": "graph", "message": "Consider a fork gate", "severity": "medium"},
            ],
            "semantic_contracts": [
                {
                    "from_id": "source",
                    "to_id": "t1",
                    "consumer_plugin": "passthrough",
                    "producer_plugin": "csv",
                    "producer_field": "url",
                    "consumer_field": "url",
                    "outcome": "satisfied",
                    "requirement_code": "REQ-001",
                },
                {
                    "from_id": "source",
                    "to_id": "t2",
                    "consumer_plugin": "llm",
                    "producer_plugin": None,
                    "producer_field": "rating",
                    "consumer_field": "rating",
                    "outcome": "unsatisfied",
                    "requirement_code": "REQ-002",
                },
            ],
            "graph_repair_suggestions": [
                {
                    "code": "duplicate_consumer_connection",
                    "connection": "shared",
                    "strategy": "insert_fork_gate",
                    "reason": "Give each consumer a unique branch input.",
                    "affected_consumers": [
                        {"id": "t1", "current_input": "shared", "new_input": "shared_to_t1"},
                    ],
                    "tool_sequence": [
                        {"tool": "upsert_node", "arguments": {"id": "t1", "input": "shared_to_t1", "node_type": "transform"}},
                        {"tool": "preview_pipeline", "arguments": {}},
                    ],
                },
            ],
        },
        "affected_nodes": ["t1"],
        "version": 3,
        "data": {
            "blob_id": "blob-1",
            "filename": "input.csv",
            "mime_type": "text/csv",
            "content": "secret,row\n1,2\n",
            "truncated": False,
            "size_bytes": 15,
        },
    }

    result = redact_tool_call_response("get_blob_content", response, telemetry=NoopRedactionTelemetry())

    # Blob bytes redacted (existing behaviour).
    assert result["data"]["content"] == "<blob-content:15-bytes>"

    # The single Sensitive repair-arguments leaf is summarized to sorted keys.
    repair = result["validation"]["graph_repair_suggestions"][0]
    assert repair["tool_sequence"][0]["arguments"] == "<repair-args:id,input,node_type>"
    assert repair["tool_sequence"][1]["arguments"] == "<repair-args:>"

    # All surrounding non-sensitive validation metadata survives verbatim.
    assert result["validation"]["is_valid"] is False
    assert result["validation"]["errors"][0]["message"] == "Duplicate consumer for connection shared"
    assert result["validation"]["semantic_contracts"][1]["producer_plugin"] is None
    assert result["validation"]["semantic_contracts"][0]["requirement_code"] == "REQ-001"
    assert repair["affected_consumers"][0]["new_input"] == "shared_to_t1"
    assert repair["code"] == "duplicate_consumer_connection"

    # The repair-argument VALUES never reach the redacted output (only the
    # affected_consumers descriptor — a non-sensitive structural field — may
    # legitimately echo the new input name).
    assert "'input': 'shared_to_t1'" not in str(result)


def test_declarative_tool_result_redacts_nested_repair_arguments() -> None:
    """Declarative response policies must still descend into repair guidance.

    ``upsert_node`` keeps the ToolResult envelope declarative: top-level
    ``validation`` and ``data`` are known response keys, but both can carry
    nested credential-repair tool calls with open ``arguments`` mappings.
    Those mappings must be structurally summarized before persistence, not
    copied through with credential material intact.
    """
    sentinel = "sk-test-declarative-nested-secret"
    response = {
        "success": False,
        "validation": {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "semantic_contracts": [],
            "graph_repair_suggestions": [
                {
                    "code": "credential_wiring_required",
                    "connection": "source_to_enrich",
                    "strategy": "wire_secret_ref",
                    "reason": "Credential field must be wired by reference.",
                    "affected_consumers": [
                        {"id": "enrich", "current_input": "source", "new_input": "source_to_enrich"},
                    ],
                    "tool_sequence": [
                        {
                            "tool": "upsert_node",
                            "arguments": {
                                "id": "enrich",
                                "options": {"api_key": sentinel},
                            },
                        },
                    ],
                },
            ],
        },
        "affected_nodes": ["enrich"],
        "version": 4,
        "data": {
            "error": "credential wiring required",
            "repair": {
                "tool_sequence": [
                    {
                        "tool": "wire_secret_ref",
                        "arguments": {
                            "name": "OPENAI_API_KEY",
                            "target": "node",
                            "target_id": "enrich",
                            "option_key": "api_key",
                            "proof": sentinel,
                        },
                    },
                ],
                "arguments": {"api_key": sentinel},
            },
        },
    }

    result = redact_tool_call_response("upsert_node", response, telemetry=NoopRedactionTelemetry())

    assert result["validation"]["graph_repair_suggestions"][0]["tool_sequence"][0]["arguments"] == "<repair-args:id,options>"
    repair = result["data"]["repair"]
    assert repair["tool_sequence"][0]["arguments"] == "<repair-args:name,option_key,proof,target,target_id>"
    assert repair["arguments"] == "<repair-args:api_key>"
    assert sentinel not in json.dumps(result, sort_keys=True)


# ---------------------------------------------------------------------------
# Test 6: summarizer raises → telemetry counter BEFORE AuditIntegrityError
# ---------------------------------------------------------------------------


def test_summarizer_raise_yields_audit_integrity_error_and_fires_telemetry_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summarizer that raises RuntimeError → walker fires summarizer_error
    counter BEFORE raising AuditIntegrityError chained from the RuntimeError.

    This is the rev-2 M.8 discipline: counter must fire before raise so OTel
    scrapes see it even when the request dies after the raise.

    Uses the type-driven path (response_model with a crashing summarizer).
    """

    def _crashing_summarizer(v: Any) -> str:
        raise RuntimeError("boom")

    class _ResponseModel(BaseModel):
        status: str
        token: Annotated[str, Sensitive(summarizer=_crashing_summarizer)]

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        query: str

        model_config = ConfigDict(extra="forbid")

    tool = "t_crashing_summarizer"
    entry = ToolRedaction(argument_model=_ArgModel, response_model=_ResponseModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    with pytest.raises(AuditIntegrityError) as exc_info:
        redact_tool_call_response(tool, {"status": "ok", "token": "SECRET"}, telemetry=tel)

    # Telemetry counter fired (BEFORE the raise)
    assert tel.summarizer_error_calls == [{"tool_name": tool}]
    # AuditIntegrityError chains the underlying RuntimeError
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom"


def test_audit_integrity_error_in_tier_1_errors_registry() -> None:
    """AuditIntegrityError is registered in TIER_1_ERRORS per spec §9 RSK-03 / §4.5.

    This test is in the response-walker test file because the task spec
    requires it here; it is not specific to the response walker — it validates
    the registry invariant that the walker's raises depend on.
    """
    tier_1_classes = set(_TIER_1_ERRORS_VIEW)
    assert AuditIntegrityError in tier_1_classes, (
        f"AuditIntegrityError is not in TIER_1_ERRORS. Registered classes: {sorted(c.__name__ for c in tier_1_classes)}"
    )


# ---------------------------------------------------------------------------
# Test 7: summarizer non-str return → telemetry counter BEFORE AuditIntegrityError
# ---------------------------------------------------------------------------


def test_summarizer_non_str_return_yields_audit_integrity_error_and_fires_telemetry_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summarizer that returns a non-str value → walker fires summarizer_error
    counter BEFORE raising AuditIntegrityError with a message naming the type."""

    def _bad_summarizer(v: Any) -> Any:
        return {"x": 1}  # non-str return

    class _ResponseModel(BaseModel):
        status: str
        token: Annotated[str, Sensitive(summarizer=_bad_summarizer)]

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        query: str

        model_config = ConfigDict(extra="forbid")

    tool = "t_bad_return_summarizer"
    entry = ToolRedaction(argument_model=_ArgModel, response_model=_ResponseModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    with pytest.raises(AuditIntegrityError) as exc_info:
        redact_tool_call_response(tool, {"status": "ok", "token": "SECRET"}, telemetry=tel)

    # Telemetry counter fired (BEFORE the raise)
    assert tel.summarizer_error_calls == [{"tool_name": tool}]
    # Message names the actual returned type
    assert "dict" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 8: missing manifest entry → AuditIntegrityError
# ---------------------------------------------------------------------------


def test_missing_manifest_entry_yields_audit_integrity_error() -> None:
    """redact_tool_call_response for a tool name not in MANIFEST raises
    AuditIntegrityError immediately (registry-consistency invariant)."""
    tel = NoopRedactionTelemetry()
    with pytest.raises(AuditIntegrityError):
        redact_tool_call_response("not_in_manifest", {"x": 1}, telemetry=tel)


# ---------------------------------------------------------------------------
# Test 9: empty known_response_keys → every response key sentinelised
# ---------------------------------------------------------------------------


def test_empty_known_response_keys_sentinelises_every_response_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declarative entry with handles_no_sensitive_data=True and
    known_response_keys=(): every response key is unknown → all get the
    fixed sentinel (fail-closed; rev-3 W8c / rev-4 W8c).

    Task 6's ToolRedactionPolicy validator forbids known_response_keys=()
    when handles_no_sensitive_data=False. This fixture therefore uses
    handles_no_sensitive_data=True (the only construction path that permits
    empty known_response_keys).

    The walker applies the policy mechanically: unknown key check fires for
    every key since neither sensitive_response_keys nor known_response_keys
    covers them.
    """
    tool = "t_empty_known"
    entry = _declarative_entry(
        sensitive_response_keys=(),
        known_response_keys=(),  # empty
        handles_no_sensitive_data=True,
        handles_no_sensitive_data_reason_struct=_safe_reason(),
    )
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    response = {"alpha": "a", "beta": "b"}
    result = redact_tool_call_response(tool, response, telemetry=tel)

    assert result == {
        "alpha": REDACTED_UNKNOWN_RESPONSE_KEY,
        "beta": REDACTED_UNKNOWN_RESPONSE_KEY,
    }
    # Counter fires once per unknown key
    assert len(tel.unknown_response_key_calls) == 2


# ---------------------------------------------------------------------------
# Test 10: walker atomicity — mid-walk raise leaves no partial dict observable
# ---------------------------------------------------------------------------


def test_walker_atomicity_on_mid_walk_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mid-walk summarizer raise → no partially-built dict reaches the caller.

    The canonical atomicity test: a sentinel object is assigned BEFORE the
    call. After pytest.raises catches the AuditIntegrityError, the sentinel
    is still the same object (the call returned nothing).

    Fixture: response with three keys; the second key's summarizer raises so
    the first key has been processed but the final dict is never returned.
    """
    call_log: list[str] = []

    def _ok_summarizer(v: Any) -> str:
        call_log.append("ok")
        return "<ok>"

    def _crashing_summarizer(v: Any) -> str:
        call_log.append("crash")
        raise RuntimeError("mid-walk crash")

    class _ResponseModel(BaseModel):
        key_a: Annotated[str, Sensitive(summarizer=_ok_summarizer)]
        key_b: Annotated[str, Sensitive(summarizer=_crashing_summarizer)]
        key_c: str  # non-sensitive, reached only after key_b in schema order

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        query: str

        model_config = ConfigDict(extra="forbid")

    tool = "t_atomicity"
    entry = ToolRedaction(argument_model=_ArgModel, response_model=_ResponseModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    sentinel = object()
    result = sentinel

    with pytest.raises(AuditIntegrityError):
        result = redact_tool_call_response(
            tool,
            {"key_a": "v_a", "key_b": "v_b", "key_c": "v_c"},
            telemetry=tel,
        )

    # The mid-walk crash means no partial dict reached the caller.
    assert result is sentinel
    # key_a's summarizer was called (it appears before key_b in schema order)
    assert "ok" in call_log
    # The crash happened
    assert "crash" in call_log
    # Telemetry counter fired before the raise
    assert tel.summarizer_error_calls == [{"tool_name": tool}]


# ---------------------------------------------------------------------------
# Test 11: manifest_dispatch beacon fires for type-driven response path
# ---------------------------------------------------------------------------


def test_response_walker_emits_manifest_dispatch_for_type_driven_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §4.2.4: manifest_dispatch beacon fires per invocation, not per
    direction. Task 7 fix-up: redact_tool_call_response was silently omitting
    this emission while redact_tool_call_arguments correctly emits it.
    Restoring symmetry so the operational-progress dashboard reflects both
    directions."""

    class _ResponseModel(BaseModel):
        status: str

        model_config = ConfigDict(extra="forbid")

    class _ArgModel(BaseModel):
        query: str

        model_config = ConfigDict(extra="forbid")

    tool = "t_dispatch_type_driven"
    entry = ToolRedaction(argument_model=_ArgModel, response_model=_ResponseModel)
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    redact_tool_call_response(tool, {"status": "ok"}, telemetry=tel)

    assert tel.manifest_dispatch_calls == [{"tool_name": tool, "shape": "type_driven"}]


# ---------------------------------------------------------------------------
# Test 12: manifest_dispatch beacon fires for declarative response path
# ---------------------------------------------------------------------------


def test_response_walker_emits_manifest_dispatch_for_declarative_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §4.2.4 walker-wide emission requirement: declarative branch also
    emits the manifest_dispatch beacon once per invocation."""
    tool = "t_dispatch_declarative"
    entry = _declarative_entry(
        sensitive_response_keys=(),
        known_response_keys=("status",),
    )
    _patch_manifest(monkeypatch, tool, entry)

    tel = NoopRedactionTelemetry()
    redact_tool_call_response(tool, {"status": "ok"}, telemetry=tel)

    assert tel.manifest_dispatch_calls == [{"tool_name": tool, "shape": "declarative"}]
