"""Shared fixtures for the mocked-LLM eval suites.

The composer's END authoritative advisor gate (``elspeth-dac6602a2b``) fires
``_run_advisor_checkpoint(phase="end")`` on EVERY no-tool finalize path. With
no advisor provider configured (the default in unit tests), that gate FAILS
CLOSED — it returns an ``is_valid=false`` ``_advisor_blocked_result`` carrying
the shared ``"runtime preflight failed"`` sentinel. That fail-closed result
masks whatever the test is actually driving: a full-loop convergence test that
should reach a valid pipeline instead scores RED on ``advisor_signoff_blocked``.

The sibling package ``tests/unit/web/composer`` neutralises this with the
autouse ``_stub_advisor_end_gate_clean`` fixture in ``_helpers.py``; this
package had no equivalent, so the convergence tests here went RED for an
advisor-infra reason unrelated to the behaviour they assert. This conftest
restores parity. Suites that legitimately exercise the gate override
``service._run_advisor_checkpoint`` per-instance (instance attr wins over the
class patch), so this default never weakens a real gate assertion.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from elspeth.web.composer.service import AdvisorCheckpointVerdict, ComposerServiceImpl


@pytest.fixture(autouse=True)
def _stub_advisor_end_gate_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the END advisor gate a CLEAN no-op (no provider needed in tests).

    Patches the *method* (not the inner ``_call_advisor_with_audit``) so the
    per-call ``llm_calls`` audit counts stay stable. A fresh ``AsyncMock`` per
    test avoids cross-test call-count leakage.
    """
    monkeypatch.setattr(
        ComposerServiceImpl,
        "_run_advisor_checkpoint",
        AsyncMock(return_value=AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN")),
        raising=True,
    )
