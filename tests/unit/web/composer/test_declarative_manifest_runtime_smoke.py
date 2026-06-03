"""Per-entry runtime smoke test for declarative manifest entries.

Closes rev-3 W7 / rev-4 W7. Each declarative entry is exercised through
``redact_tool_call_arguments`` AND ``redact_tool_call_response`` with a
minimal representative payload, asserting:

  1. The redaction call returns without raising.
  2. For entries with ``sensitive_argument_keys``: the named keys are
     replaced with the summarizer output (or no-summarizer sentinel) when
     present in the payload; absent keys are no-ops (not added).
  3. For entries with ``sensitive_response_keys``: the named keys are
     replaced when present; unknown keys substitute the fixed sentinel
     ``<redacted-unknown-response-key>``.
  4. For ``handles_no_sensitive_data=True`` entries with no
     ``sensitive_argument_keys``: argument redaction is a passthrough
     (output == input).
  5. For ``handles_no_sensitive_data=True`` entries with
     ``sensitive_argument_keys`` (mixed shape): the sensitive keys are
     redacted; non-sensitive keys are passthrough.
  6. For ``handles_no_sensitive_data=True`` entries the response path is
     not exercised here (no response-surface declaration; the runtime
     wiring decides the disposition).

This is the second line of defense behind the structural adequacy guard
(Tasks 9-12). A misspelled key in a declarative entry passes structural
checks but fails this runtime smoke immediately.
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.redaction import (
    MANIFEST,
    REDACTED_UNKNOWN_RESPONSE_KEY,
    ToolRedaction,
    redact_tool_call_arguments,
    redact_tool_call_response,
)
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

_DECLARATIVE_ENTRIES: list[tuple[str, ToolRedaction]] = [
    (name, entry) for name, entry in sorted(MANIFEST.items()) if entry.policy is not None
]


@pytest.mark.parametrize(
    ("tool_name", "entry"),
    _DECLARATIVE_ENTRIES,
    ids=[name for name, _ in _DECLARATIVE_ENTRIES],
)
def test_declarative_entry_argument_redaction_runtime(tool_name: str, entry: ToolRedaction) -> None:
    """Each declarative entry's argument redaction path executes without raising
    and replaces every declared sensitive key when present in the payload.

    Handles the four shapes documented in the module docstring:

    - ``handles_no_sensitive_data=True`` and ``sensitive_argument_keys`` empty
      → passthrough invariant: ``{}`` → ``{}``.
    - ``handles_no_sensitive_data=True`` and ``sensitive_argument_keys``
      non-empty → mixed shape: build a payload with each declared sensitive
      key set to a sentinel, assert each value is replaced.  Non-sensitive
      passthrough behaviour is covered by sibling unit tests on
      ``_redact_via_policy``; we assert the sensitive-key path here.
    - ``handles_no_sensitive_data=False`` (no current entries in this state
      after Task 16): exercise via sensitive_argument_keys as above.
    """
    policy = entry.policy
    assert policy is not None  # invariant — list filtered by policy is not None
    telemetry = NoopRedactionTelemetry()

    if not policy.sensitive_argument_keys:
        # Passthrough invariant: tools with no sensitive_argument_keys must
        # leave an empty input dict unchanged.  Using ``{}`` rather than a
        # populated dict isolates the test from the tool's actual argument
        # shape (which is irrelevant when no key is declared sensitive).
        out = redact_tool_call_arguments(tool_name, {}, telemetry=telemetry)
        assert out == {}, (
            f"Tool {tool_name!r}: declared handles_no_sensitive_data=True without "
            f"sensitive_argument_keys but redact_tool_call_arguments mutated {{}} → "
            f"{out!r}. Passthrough contract violated."
        )
        return

    # Mixed or pure sensitive-keys shape.  Build a payload containing each
    # declared sensitive key with a fixed sentinel value, then assert every
    # key's value is replaced by the policy summarizer (or the no-summarizer
    # sentinel when no summarizer is registered).
    #
    # Discovery of the right sentinel value for each key:
    #   * ``recent_errors`` / ``attempted_actions`` summarizers expect
    #     ``list[str]`` (request_advisor_hint).  We pass a list there.
    #   * Every other declared sensitive key currently accepts ``str`` or
    #     ``dict``.  ``dict`` is a superset that lands a valid input across
    #     :func:`_summarize_set_source_options` (calls
    #     :func:`redact_source_storage_path` which expects a dict) AND
    #     :func:`_summarize_set_metadata_patch` (expects a dict).  The
    #     :func:`_summarize_advisor_problem_summary` /
    #     :func:`_summarize_advisor_schema_excerpt` summarizers expect ``str``.
    list_keys = {"recent_errors", "attempted_actions"}
    str_keys = {"problem_summary", "schema_excerpt"}
    dict_keys: frozenset[str] = frozenset(policy.sensitive_argument_keys) - list_keys - str_keys

    payload: dict[str, object] = {}
    for key in policy.sensitive_argument_keys:
        if key in list_keys:
            payload[key] = ["error one", "error two"]
        elif key in str_keys:
            payload[key] = "RUNTIME-SMOKE-SENTINEL"
        elif key in dict_keys:
            payload[key] = {"runtime_smoke_key": "RUNTIME-SMOKE-VALUE"}
        else:
            payload[key] = "RUNTIME-SMOKE-SENTINEL"

    out = redact_tool_call_arguments(tool_name, payload, telemetry=telemetry)

    for key in policy.sensitive_argument_keys:
        assert key in out, (
            f"Tool {tool_name!r}: sensitive_argument_key {key!r} disappeared "
            f"from the redacted output. Redaction must preserve key structure."
        )
        assert out[key] != payload[key], (
            f"Tool {tool_name!r}: sensitive_argument_key {key!r} was NOT replaced. "
            f"raw={payload[key]!r}, redacted={out[key]!r}. Common cause: key name "
            f"in MANIFEST does not match the actual argument key the handler reads "
            f"(typo in sensitive_argument_keys tuple)."
        )

    # Absent-key invariant: a sensitive key not present in the input must NOT
    # be added to the output by the redaction path (spec §4.2.6 disposition
    # table — key absence is not a fault).
    out_absent = redact_tool_call_arguments(tool_name, {}, telemetry=telemetry)
    for key in policy.sensitive_argument_keys:
        assert key not in out_absent, (
            f"Tool {tool_name!r}: sensitive_argument_key {key!r} was added to the "
            f"redacted output despite being absent from the input. Spec §4.2.6 "
            f"requires absent keys to be no-ops."
        )


@pytest.mark.parametrize(
    ("tool_name", "entry"),
    _DECLARATIVE_ENTRIES,
    ids=[name for name, _ in _DECLARATIVE_ENTRIES],
)
def test_declarative_entry_response_redaction_runtime(tool_name: str, entry: ToolRedaction) -> None:
    """Each declarative entry's response redaction path either passes through
    cleanly (``handles_no_sensitive_data=True``) or replaces declared sensitive
    keys and falls closed on unknown keys.

    Spec §4.2.6 disposition for declarative response paths:
      * key in ``sensitive_response_keys``         → no-summarizer sentinel
      * key in ``known_response_keys`` only         → passthrough
      * key in neither                              → REDACTED_UNKNOWN_RESPONSE_KEY

    handles_no_sensitive_data=True entries currently DO traverse
    redact_tool_call_response (the manifest_dispatch beacon fires and the
    walker runs); when sensitive_response_keys is empty AND known_response_keys
    is empty, every input key is unknown and substitutes the fixed sentinel —
    this branch is the empty-known set fail-closed path and is exercised
    here too.
    """
    policy = entry.policy
    assert policy is not None
    telemetry = NoopRedactionTelemetry()

    # Sensitive-response-key replacement path (skipped when none declared).
    if policy.sensitive_response_keys:
        raw_value = "RUNTIME-SMOKE-SENTINEL"
        payload: dict[str, object] = dict.fromkeys(policy.sensitive_response_keys, raw_value)
        out = redact_tool_call_response(tool_name, payload, telemetry=telemetry)
        for key in policy.sensitive_response_keys:
            assert key in out, (
                f"Tool {tool_name!r}: sensitive_response_key {key!r} disappeared from "
                f"the redacted response. Redaction must preserve key structure."
            )
            assert out[key] != raw_value, (
                f"Tool {tool_name!r}: sensitive_response_key {key!r} was NOT replaced. "
                f"raw={raw_value!r}, redacted={out[key]!r}. Common cause: key name in "
                f"MANIFEST does not match the actual response key the handler emits "
                f"(typo in sensitive_response_keys tuple)."
            )

    # Unknown-key fail-closed path: a key not in known_response_keys substitutes
    # REDACTED_UNKNOWN_RESPONSE_KEY.  Pick a key name unlikely to collide with
    # any tool's declared response shape.
    unknown_key = "__runtime_smoke_unknown_key__"
    unknown_out = redact_tool_call_response(tool_name, {unknown_key: "x"}, telemetry=telemetry)
    assert unknown_out[unknown_key] == REDACTED_UNKNOWN_RESPONSE_KEY, (
        f"Tool {tool_name!r}: unknown-key fail-closed sentinel did not fire. "
        f"redact_tool_call_response must replace any key not in known_response_keys "
        f"with the fixed sentinel. Got: {unknown_out[unknown_key]!r}."
    )
