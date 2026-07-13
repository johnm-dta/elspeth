"""Envelope-assertion harness for the composer LLM tool-use loop.

This test file is the regression gate for the elspeth-4e79436719 cluster:
unbounded prompt growth across LLM provider turns. The existing
``test_inline_complete_pipeline_replay_uses_atomic_tool_shape`` test
characterizes a happy fake replay; this harness characterizes the
**failure class** the staging incident exposed:

- The compose loop sends a 76 KB skill prompt + 20 KB tools spec on every
  turn, so a 20-turn exchange burns 600 K+ prompt tokens for a trivial
  pipeline-creation request (2026-05-06 staging exchanges referenced in
  the epic).

The harness runs the real ``compose()`` method against a scripted fake
LLM and enforces, on each turn:

1. **Turn ceiling** — total turns ≤ a named constant. Catches a
   regression that lifts the budget without justification.
2. **Per-turn payload byte envelope** — the ``messages`` list and
   ``tools`` array passed into ``_call_llm`` are JSON-serialized and
   their cumulative byte size is asserted against a named constant.
   This is what bites: the bytes are the bytes the production
   ``_build_messages`` / ``_get_litellm_tools`` code actually emitted,
   not whatever the test author scripted into the fake response. A
   regression that re-introduces unbounded transcript accumulation
   (or grows the system-prompt skill without compensating compaction)
   fails this assertion.
3. **Reported prompt-token envelope** — sum of audit-row
   ``prompt_tokens`` ≤ a named constant. Useful as a cheap secondary
   gate; reflects the values the script reports, not the production
   token count, so it must always be paired with the byte-envelope
   assertion above.
4. **Audit completeness** — every fake LLM call appears verbatim in
   ``result.llm_calls``; no synthetic finalize-only records leak in.
5. **Cache-warm assertion (xfail until Phase 3)** — turns ≥ 2 have
   non-zero ``cache_read_input_tokens`` once provider-side prompt
   caching is wired. Marked ``xfail`` today because the
   ``cache_control`` markers have not landed (Phase 3 of the plan).

The fake LLM is patched at ``service._call_llm`` so the dispatch path,
audit recorder, and budget counters all run through the real code. Only
the provider call itself is replaced — this keeps the harness faithful
to production behaviour while staying deterministic and offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from tests.unit.web.composer._helpers import (
    FakeChoice,
    _empty_state,
    _make_llm_response,
    _make_settings,
    _mock_catalog,
    _stub_advisor_end_gate_clean,  # noqa: F401  (autouse end-gate CLEAN stub)
)


@pytest.fixture(autouse=True)
def _composer_available_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the OPENAI_API_KEY availability check.

    Mirrors the autouse fixture in ``test_service.py``: these envelope
    tests verify compose-loop budget mechanics, not local API-key
    presence. Skipping the check keeps the harness deterministic
    offline.
    """

    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


# ---------------------------------------------------------------------------
# Named envelope constants — bumping these requires a paired plan-update PR.
# ---------------------------------------------------------------------------

# Total provider calls across one compose() invocation. Matches current
# settings (composer_max_composition_turns=15 + composer_max_discovery_turns=10),
# capped at a sane upper bound for first-turn creation prompts. A regression
# that lifts this without justification fails the suite loudly.
ENVELOPE_MAX_TURNS_TRIVIAL_PROMPT = 5

# Sum of per-turn prompt_tokens across one trivial-creation compose() call.
# This figure reflects what the harness scripts into fake responses; it is
# NOT a claim about real production prompt sizes. It serves as a cheap
# secondary gate paired with the byte-envelope assertion below, which is
# the one that actually measures production output.
ENVELOPE_MAX_PROMPT_TOKENS_TRIVIAL_PROMPT = 50_000

# Cumulative byte-size envelope on the production-emitted ``messages`` +
# ``tools`` JSON blobs across one compose() call. This IS a real production
# gate: the bytes asserted are what ``_build_messages`` and
# ``_get_litellm_tools`` actually serialize and what would land on the
# provider wire.
#
# Recalibrated baseline (2026-05-18, 1-turn happy script): ~210 KB
# (~177 KB skill prompt + ~31 KB tools spec for 39 tools + small overhead).
# The original baseline (2026-05-06: ~100 KB / 200 KB envelope) was overtaken
# by ~20 incremental skill-prompt commits — recipe-first fork-coalesce
# guidance, mandatory advisor escalation for Recipe #10, fabrication and
# silent-downgrade loophole closures, abuse_contact/scraping_reason
# requirements, audit-backend skill+tool additions — and by five new tool
# definitions added to the LLM-visible toolset. Each commit was a deliberate
# bug-fix; the growth is structural, not regressional. Setting the envelope
# at 300 KB gives ~45 % headroom over the recalibrated baseline so a
# subsequent uncontrolled doubling still bites loudly, without rejecting
# the committed prompt improvements that landed since the original gate
# was set.
#
# Note: Phase 3 (provider prompt caching) will NOT change this number —
# caching reduces the provider's billable token accounting, not the bytes
# we send. This envelope is a pure regression gate, independent of
# whether prompt caching is wired.
ENVELOPE_MAX_PRODUCTION_BYTES_TRIVIAL_PROMPT = 300_000


def _serialize_call(messages: Any, tools: Any) -> int:
    """Return the JSON byte size of the messages+tools sent to a fake _call_llm."""
    import json

    payload = {"messages": messages, "tools": tools}
    return len(json.dumps(payload, default=str).encode("utf-8"))


@dataclass(frozen=True)
class ScriptedTurn:
    """One scripted assistant turn, with optional tool calls and usage payload."""

    content: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()
    prompt_tokens: int = 1_000
    completion_tokens: int = 50
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cached_prompt_tokens: int | None = None


@dataclass
class _ScriptedResponse:
    """Fake LiteLLM response carrying ``choices`` and a ``usage`` object."""

    choices: list[FakeChoice]
    usage: dict[str, Any]
    model: str = "openrouter/test/scripted-model"
    id: str = "chatcmpl-scripted"


def _build_scripted_response(turn: ScriptedTurn) -> _ScriptedResponse:
    base = _make_llm_response(
        content=turn.content,
        tool_calls=list(turn.tool_calls) if turn.tool_calls else None,
    )
    usage: dict[str, Any] = {
        "prompt_tokens": turn.prompt_tokens,
        "completion_tokens": turn.completion_tokens,
        "total_tokens": turn.prompt_tokens + turn.completion_tokens,
    }
    if turn.cached_prompt_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": turn.cached_prompt_tokens}
    if turn.cache_creation_input_tokens is not None:
        usage["cache_creation_input_tokens"] = turn.cache_creation_input_tokens
    if turn.cache_read_input_tokens is not None:
        usage["cache_read_input_tokens"] = turn.cache_read_input_tokens
    return _ScriptedResponse(choices=base.choices, usage=usage)


@dataclass
class _EnvelopeRun:
    """Captured per-turn payload sizes plus the compose() result."""

    result: Any
    per_turn_byte_sizes: tuple[int, ...]


async def _run_envelope(script: Sequence[ScriptedTurn], *, user_message: str = "Build a CSV pipeline.") -> _EnvelopeRun:
    """Run ``compose()`` against the scripted turn sequence.

    Returns both the ``ComposerResult`` and a tuple of per-turn JSON byte
    sizes captured from the production-emitted messages/tools payloads.
    """
    catalog = _mock_catalog()
    settings = _make_settings()
    service = ComposerServiceImpl.for_trained_operator(catalog=catalog, settings=settings)
    state = _empty_state()

    responses = [_build_scripted_response(turn) for turn in script]

    with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = responses
        result = await service.compose(user_message, [], state)

    sizes: list[int] = []
    for invocation in mock_llm.call_args_list:
        # _call_llm is patched as a bound method; positional args are
        # (messages, tools).
        if len(invocation.args) >= 2:
            sizes.append(_serialize_call(invocation.args[0], invocation.args[1]))
        elif "messages" in invocation.kwargs and "tools" in invocation.kwargs:
            sizes.append(_serialize_call(invocation.kwargs["messages"], invocation.kwargs["tools"]))
    return _EnvelopeRun(result=result, per_turn_byte_sizes=tuple(sizes))


# ---------------------------------------------------------------------------
# Scenarios — keep small. Add scenarios here when new failure classes emerge.
# ---------------------------------------------------------------------------


_HAPPY_TRIVIAL_SCRIPT: tuple[ScriptedTurn, ...] = (
    # Turn 1: model emits a text-only reply, no tool calls — terminates.
    ScriptedTurn(
        content="Done — a CSV pipeline is ready in your workspace.",
        prompt_tokens=8_000,
        completion_tokens=40,
    ),
)

# Module-load coupling guard: if a future edit grows ``_HAPPY_TRIVIAL_SCRIPT``
# beyond the asserted turn ceiling, the test would fail because the SCRIPT
# is too long, not because a regression was detected — a false positive that
# erodes trust in the gate. Fail fast at import time so the misalignment is
# obvious. Bumping the ceiling and the script in lockstep is intentional and
# requires touching this guard explicitly.
assert len(_HAPPY_TRIVIAL_SCRIPT) <= ENVELOPE_MAX_TURNS_TRIVIAL_PROMPT, (
    f"_HAPPY_TRIVIAL_SCRIPT has {len(_HAPPY_TRIVIAL_SCRIPT)} turns but the "
    f"asserted ceiling is {ENVELOPE_MAX_TURNS_TRIVIAL_PROMPT}. The script "
    "cannot exceed the ceiling — otherwise the harness fails for the wrong "
    "reason. Either shrink the script or bump ENVELOPE_MAX_TURNS_TRIVIAL_PROMPT "
    "(and justify the bump in the commit message)."
)


# Pathological scenario: a 5-turn run where each turn burns 20K prompt tokens
# (simulating the elspeth-4e79436719 staging case where the full skill +
# tools spec is re-sent on every turn). The harness must reject this:
# 5 x 20K = 100K, well above ENVELOPE_MAX_PROMPT_TOKENS_TRIVIAL_PROMPT.
# Final turn is text-only so the loop terminates cleanly (vs. hitting a
# budget exhaustion path which would also be a positive harness signal).
_PATHOLOGICAL_PROMPT_GROWTH_SCRIPT: tuple[ScriptedTurn, ...] = (
    ScriptedTurn(
        content=None,
        tool_calls=({"id": "call_1", "name": "list_sources", "arguments": {}},),
        prompt_tokens=20_000,
    ),
    ScriptedTurn(
        content=None,
        tool_calls=({"id": "call_2", "name": "list_transforms", "arguments": {}},),
        prompt_tokens=20_000,
    ),
    ScriptedTurn(
        content=None,
        tool_calls=({"id": "call_3", "name": "list_sinks", "arguments": {}},),
        prompt_tokens=20_000,
    ),
    ScriptedTurn(
        content="I have surveyed the catalog.",
        prompt_tokens=20_000,
    ),
)


class TestEnvelopeHarness:
    """Regression gate for unbounded compose-loop growth (elspeth-4e79436719)."""

    @pytest.mark.asyncio
    async def test_happy_trivial_prompt_under_turn_ceiling(self) -> None:
        run = await _run_envelope(_HAPPY_TRIVIAL_SCRIPT)
        assert len(run.result.llm_calls) <= ENVELOPE_MAX_TURNS_TRIVIAL_PROMPT, (
            f"Trivial first-turn creation exceeded the named turn ceiling "
            f"({ENVELOPE_MAX_TURNS_TRIVIAL_PROMPT}); a regression has lifted the "
            "compose-loop budget without justification."
        )

    @pytest.mark.asyncio
    async def test_happy_trivial_prompt_under_production_byte_envelope(self) -> None:
        """Real production prompt-size assertion.

        Captures the JSON byte size of the messages + tools that
        ``_call_llm`` was actually invoked with — i.e., what
        ``_build_messages`` and ``_get_litellm_tools`` emitted. Asserts
        the cumulative byte total stays under the envelope. This is the
        gate that bites a regression which re-introduces unbounded
        transcript growth or ships a system prompt that has become
        dramatically larger.
        """
        run = await _run_envelope(_HAPPY_TRIVIAL_SCRIPT)
        total_bytes = sum(run.per_turn_byte_sizes)
        assert total_bytes <= ENVELOPE_MAX_PRODUCTION_BYTES_TRIVIAL_PROMPT, (
            f"Trivial first-turn creation emitted {total_bytes} bytes of "
            f"messages+tools across {len(run.per_turn_byte_sizes)} turns "
            f"({run.per_turn_byte_sizes}); exceeded named envelope "
            f"({ENVELOPE_MAX_PRODUCTION_BYTES_TRIVIAL_PROMPT}). The compose loop "
            "is re-emitting bytes it should not, or the static prefix grew "
            "without compensating compaction."
        )

    @pytest.mark.asyncio
    async def test_happy_trivial_prompt_under_reported_token_envelope(self) -> None:
        """Secondary gate: audit-row prompt_tokens stay under the named envelope.

        These are the values the script reported, not production token
        counts. Pair with ``test_happy_trivial_prompt_under_production_byte_envelope``
        for actual production-shape coverage.
        """
        run = await _run_envelope(_HAPPY_TRIVIAL_SCRIPT)
        total = sum((c.prompt_tokens or 0) for c in run.result.llm_calls)
        assert total <= ENVELOPE_MAX_PROMPT_TOKENS_TRIVIAL_PROMPT, (
            f"Trivial first-turn creation script reported {total} prompt tokens, "
            f"exceeding the named envelope ({ENVELOPE_MAX_PROMPT_TOKENS_TRIVIAL_PROMPT}); "
            "a regression has re-introduced unbounded transcript accumulation."
        )

    @pytest.mark.asyncio
    async def test_audit_completeness_every_fake_call_recorded(self) -> None:
        """Every fake LLM call must produce exactly one ComposerLLMCall record.

        Synthetic finalize-only or duplicate records fail this assertion.
        """
        run = await _run_envelope(_HAPPY_TRIVIAL_SCRIPT)
        assert len(run.result.llm_calls) == len(_HAPPY_TRIVIAL_SCRIPT)
        for fake_turn, recorded in zip(_HAPPY_TRIVIAL_SCRIPT, run.result.llm_calls, strict=True):
            assert recorded.prompt_tokens == fake_turn.prompt_tokens
            assert recorded.completion_tokens == fake_turn.completion_tokens

    @pytest.mark.asyncio
    async def test_post_warm_turns_record_cache_read_input_tokens(self) -> None:
        """Phase 3 audit-truth assertion: cache hits land on the audit row.

        The harness scripts ``cache_read_input_tokens`` from turn 2 onward
        (turn 1 is a cache miss, no caching benefit yet). The assertion
        verifies that whatever the provider reports about cache hits
        flows through the audit pipeline and lands on the
        ``ComposerLLMCall`` record. Phase 1B wired the extraction;
        Phase 3 wired the cache_control markers; this test gates the
        end-to-end truthfulness contract: the audit row reflects what
        the provider actually said.
        """
        warm_script: tuple[ScriptedTurn, ...] = (
            ScriptedTurn(
                content=None,
                tool_calls=({"id": "c1", "name": "list_sources", "arguments": {}},),
                prompt_tokens=8_000,
                # Turn 1: cold cache. No cache_read_input_tokens.
            ),
            ScriptedTurn(
                content="Done.",
                prompt_tokens=8_000,
                # Turn 2: warm cache. Provider reports it served 7K tokens
                # from cache and 1K of fresh prompt.
                cache_read_input_tokens=7_000,
            ),
        )
        run = await _run_envelope(warm_script)
        assert len(run.result.llm_calls) == 2
        assert run.result.llm_calls[0].cache_read_input_tokens is None
        assert run.result.llm_calls[1].cache_read_input_tokens == 7_000


class TestEnvelopeHarnessSelfTests:
    """Self-tests on the harness — confirms it actually catches regressions.

    A test harness that cannot fail when something is wrong is a placebo.
    These tests verify the harness fires when given a deliberately
    over-budget scenario.
    """

    @pytest.mark.asyncio
    async def test_pathological_script_triggers_envelope_failure(self) -> None:
        """The pathological prompt-growth script either exhausts the loop
        budget (raising ComposerConvergenceError) OR exceeds the envelope.
        Either outcome is a *positive* signal that the harness is doing
        its job — a regression that re-introduces unbounded transcript
        accumulation cannot pass both the production budget AND the
        envelope assertion silently.
        """
        from elspeth.web.composer.protocol import ComposerConvergenceError

        try:
            run = await _run_envelope(_PATHOLOGICAL_PROMPT_GROWTH_SCRIPT)
        except ComposerConvergenceError:
            # Loop budget tripped before envelope — the production guard fired.
            return

        # If we got here, the loop returned a result — the envelope must reject it.
        total = sum((c.prompt_tokens or 0) for c in run.result.llm_calls)
        envelope_breached = total > ENVELOPE_MAX_PROMPT_TOKENS_TRIVIAL_PROMPT
        turn_ceiling_breached = len(run.result.llm_calls) > ENVELOPE_MAX_TURNS_TRIVIAL_PROMPT
        assert envelope_breached or turn_ceiling_breached, (
            f"Harness self-test failed: a 4-turn 20K-token-each script produced "
            f"{len(run.result.llm_calls)} turns x {total} total prompt tokens, "
            "but neither the envelope assertion nor the turn ceiling fired. "
            "The harness would not catch the original elspeth-4e79436719 regression."
        )
