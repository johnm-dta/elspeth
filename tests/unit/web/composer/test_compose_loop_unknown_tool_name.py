"""Compose-loop unknown-tool-name routing pins (Task 17).

Two pins were planned for this file:

1. ``test_unknown_tool_name_routes_through_failure_result`` — drives the
   compose loop with an LLM-hallucinated tool name and asserts the
   dispatcher fall-through at ``tools.py:5731`` produces an audit
   invocation with a specific ``error_class`` (the plan's working
   assumption was ``ARG_ERROR``).

   **NOT WRITTEN — escalated to operator (NEEDS_CONTEXT).**

   The plan body carried an escalation rule (rev-2 m_task17_pinning_risk):
   if the actual audit status for the unknown-tool fall-through is NOT
   ``ARG_ERROR``, STOP and surface the discrepancy rather than pin a
   speculative value.

   Investigation (2026-05-12) shows the implementation records
   ``ComposerToolStatus.SUCCESS`` with ``error_class=None`` and
   ``result.success=False`` for this path:

     - ``tools.py:5731`` returns a failure ``ToolResult`` without raising
       (``return _failure_result(state, f"Unknown tool: {tool_name}")``).
     - ``audit.py:732`` therefore enters the SUCCESS branch and records
       ``finish_success`` — no exception was caught.
     - ``composer_audit.py:34-37`` documents this is the intentional
       semantic of SUCCESS: *"handler completed (the underlying tool may
       have returned success=False semantically; that is still a
       successful dispatch — the audit record carries the full result
       payload so an auditor can read the semantic outcome)."*

   The spec's framing (rev-5 §4.2.6 / §5.7.5) describes this as a
   "Tier-3 quarantine" which suggests ARG_ERROR routing. The escalation
   surfaces three resolution options to the operator: (a) accept
   SUCCESS-with-semantic-failure as truthful and update the spec, (b)
   change ``execute_tool`` to raise ``ToolArgumentError`` for unknown
   tool names so the path routes through ARG_ERROR, or (c) introduce a
   new ``ComposerToolStatus.MISSING_TOOL``.

   The existing observational coverage at
   ``test_service.py::TestComposerErrorHandling::test_unknown_tool_returns_error_to_llm``
   (which asserts only message + version, not audit status) does not
   mechanically pin the audit shape, which is how this mismatch escaped
   detection until rev-2 review.

2. ``test_redact_tool_call_arguments_raises_for_unknown_tool`` — the
   Phase 3 call-order precondition pin. **WRITTEN BELOW.**

   The spec's Phase 3 design has ``redact_tool_call_arguments`` called
   from inside the dispatch path. If a Phase 3 implementer inverts the
   ordering (redact-then-check rather than check-then-redact), an
   LLM-hallucinated tool name will be silently converted from a
   graceful Tier-3 dispatcher fall-through into a Tier-1
   ``AuditIntegrityError`` crash. The pin below asserts that
   ``redact_tool_call_arguments`` fails loudly when called for a tool
   name that is not in MANIFEST, so Phase 3's call site is
   mechanically constrained: implementers will see the
   ``AuditIntegrityError`` stack trace citing the missing tool name and
   correct the call order. Closes rev-3 M2 / rev-4 M2.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.redaction import redact_tool_call_arguments
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry


def test_redact_tool_call_arguments_raises_for_unknown_tool() -> None:
    """Phase 3 contract pin: ``redact_tool_call_arguments`` must NOT be
    called for a tool name that is not in MANIFEST.

    The compose loop's existing unknown-tool check (``tools.py:5731`` ->
    ``_failure_result`` with ``Unknown tool: {name}``) MUST fire BEFORE
    the redaction layer.

    If Phase 3 inverts this ordering (redact-then-check), an
    LLM-hallucinated tool name will be silently converted from a
    graceful Tier-3 dispatcher fall-through into a Tier-1
    :class:`AuditIntegrityError` crash. This test asserts that
    :func:`redact_tool_call_arguments` fails loudly when called out of
    order, so Phase 3's call site is mechanically constrained — any
    refactor that inverts the order trips this pin in CI rather than at
    runtime under a hallucinated tool name. Closes rev-3 M2 / rev-4 M2.

    The error message MUST cite the missing tool name so Phase 3
    implementers see the contract violation in stack traces (the
    quoted form from the ``{tool_name!r}`` formatter contains the
    name as a substring).
    """
    with pytest.raises(AuditIntegrityError) as excinfo:
        redact_tool_call_arguments(
            tool_name="nonexistent_tool_name_for_call_order_pin",
            arguments={},
            telemetry=NoopRedactionTelemetry(),
        )
    assert "nonexistent_tool_name_for_call_order_pin" in str(excinfo.value)
