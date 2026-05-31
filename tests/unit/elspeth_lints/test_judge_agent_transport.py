"""Tests for the Claude Agent SDK judge transport (``_call_agent_sdk``).

The ``claude-agent-sdk`` is an OPTIONAL extra (``judge-agent``). These tests
inject a FAKE ``claude_agent_sdk`` module into ``sys.modules`` so they run
without the real SDK installed and, critically, WITHOUT ever making a real
agent query (a real query needs operator credentials — Claude Code CLI login
or ``ANTHROPIC_API_KEY`` — which CI does not have and must never seek).

SDK shape provenance (introspected against ``claude-agent-sdk==0.2.87`` on
2026-05-31, confirmed from the installed package source):

* ``query(*, prompt, options)`` is an async iterator over messages.
* ``AssistantMessage`` has ``.content`` (list of blocks) and ``.error`` (an
  in-band auth/billing error literal: ``'authentication_failed'`` etc.).
* ``TextBlock`` has ``.text``.
* ``ResultMessage`` has ``.usage`` (``dict | None``) and ``.model_usage``
  (``dict | None``, keyed by served model name; the CLI emits it as
  ``modelUsage``).
* Auth/CLI failures surface as ``ClaudeSDKError`` subclasses —
  ``CLINotFoundError`` (CLI not installed) is the operator-actionable
  config signal; there is NO ``AuthenticationError`` class (the plan's
  doc-derived stand-in name does not exist in the real SDK).

The ``usage`` dict's INNER keys (``input_tokens`` / ``cache_read_input_tokens``)
are NOT pinned by the SDK Python source — the SDK forwards the dict opaquely
from the Claude Code CLI, which mirrors the Anthropic Messages API ``usage``
object. They remain doc-derived (Anthropic Messages API convention); the fake
below reproduces that convention.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator, Callable

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict
from elspeth_lints.core.judge import (
    TRANSPORT_AGENT,
    JudgeConfigurationError,
    JudgeContractError,
    JudgeRequest,
    JudgeTransportError,
    call_judge,
)


def _request() -> JudgeRequest:
    return JudgeRequest(
        file_path="core/x.py",
        rule_id="R1",
        symbol="f",
        fingerprint="abc",
        rationale="external call boundary",
        surrounding_code="def f(x):\n    return x.get('a')\n",
    )


_GOOD_JSON = (
    '{"verdict": "ACCEPTED", "rationale": "external boundary; absence recorded as None", "confidence": 0.7, "should_use_decorator": null}'
)


class _Unset:
    """Sentinel distinguishing 'argument omitted' from an explicit ``None``.

    ``_install_fake_sdk`` must be able to inject a GENUINE ``None`` for
    ``usage`` / ``model_usage`` so the implementation's offensive None-guards
    are reachable. A plain ``None`` default can't express that — it collapses
    'omitted' and 'explicitly None' into one. Callers pass ``usage=None`` to
    exercise the guard; omitting the argument keeps the realistic default dict.
    """


_UNSET = _Unset()


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    assistant_text: str,
    served: str = "claude-opus-4-7",
    usage: dict[str, object] | None | _Unset = _UNSET,
    model_usage: dict[str, object] | None | _Unset = _UNSET,
    assistant_error: str | None = None,
    raise_on_query: Exception | None = None,
    emit_result: bool = True,
    is_error: bool = False,
    api_error_status: int | None = None,
    errors: list[str] | None = None,
) -> types.ModuleType:
    """Install a minimal fake ``claude_agent_sdk`` into ``sys.modules``.

    The fake mirrors only the surface ``_call_agent_sdk`` consumes, matching
    the real SDK shape introspected from ``claude-agent-sdk==0.2.87``:
    ``query()`` is an async generator yielding an ``AssistantMessage`` (with a
    ``TextBlock`` in ``.content`` and an optional in-band ``.error`` literal)
    then a ``ResultMessage`` (``.usage`` + ``.model_usage`` dicts, plus the
    ``.is_error`` / ``.api_error_status`` / ``.errors`` error-signal fields).

    ``usage`` / ``model_usage`` accept an explicit ``None`` (distinct from the
    ``_UNSET`` default via the ``_Unset`` sentinel) so the implementation's
    None-guards are testable.

    The real ``CLINotFoundError`` / ``ProcessError`` exception classes are
    attached so ``_is_agent_auth_error`` can ``isinstance``-check against them
    via the injected module (and so the auth test raises a REAL class, not a
    stand-in).
    """
    mod = types.ModuleType("claude_agent_sdk")

    class ClaudeAgentOptions:  # mirror SDK name
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class AssistantMessage:
        def __init__(self, content: list[object], error: str | None = None) -> None:
            self.content = content
            self.error = error

    class ResultMessage:
        def __init__(
            self,
            usage: dict[str, object] | None,
            model_usage: dict[str, object] | None,
            is_error: bool = False,
            api_error_status: int | None = None,
            errors: list[str] | None = None,
        ) -> None:
            self.usage = usage
            self.model_usage = model_usage
            self.is_error = is_error
            self.api_error_status = api_error_status
            self.errors = errors

    # Real SDK exception hierarchy (so isinstance-based discrimination in the
    # implementation matches what the real SDK would raise).
    class ClaudeSDKError(Exception):
        pass

    class CLIConnectionError(ClaudeSDKError):
        pass

    class CLINotFoundError(CLIConnectionError):
        pass

    class ProcessError(ClaudeSDKError):
        pass

    resolved_usage: dict[str, object] | None = {"input_tokens": 200, "cache_read_input_tokens": 50} if isinstance(usage, _Unset) else usage
    resolved_model_usage: dict[str, object] | None = {served: {"input_tokens": 200}} if isinstance(model_usage, _Unset) else model_usage

    async def query(*, prompt: str, options: object) -> AsyncIterator[object]:
        if raise_on_query is not None:
            raise raise_on_query
        yield AssistantMessage(content=[TextBlock(text=assistant_text)], error=assistant_error)
        if emit_result:
            yield ResultMessage(
                usage=resolved_usage,
                model_usage=resolved_model_usage,
                is_error=is_error,
                api_error_status=api_error_status,
                errors=errors,
            )

    mod.ClaudeAgentOptions = ClaudeAgentOptions  # type: ignore[attr-defined]
    mod.TextBlock = TextBlock  # type: ignore[attr-defined]
    mod.AssistantMessage = AssistantMessage  # type: ignore[attr-defined]
    mod.ResultMessage = ResultMessage  # type: ignore[attr-defined]
    mod.ClaudeSDKError = ClaudeSDKError  # type: ignore[attr-defined]
    mod.CLIConnectionError = CLIConnectionError  # type: ignore[attr-defined]
    mod.CLINotFoundError = CLINotFoundError  # type: ignore[attr-defined]
    mod.ProcessError = ProcessError  # type: ignore[attr-defined]
    mod.query = query  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
    return mod


def _import_raising(blocked_name: str) -> Callable[..., object]:
    real_import = __import__

    def _fake(name: str, *args: object, **kwargs: object) -> object:
        if name == blocked_name:
            raise ImportError(f"No module named '{blocked_name}'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    return _fake


def test_agent_transport_produces_validated_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON)
    resp = call_judge(_request(), transport=TRANSPORT_AGENT)
    assert resp.judge_transport == TRANSPORT_AGENT
    assert resp.verdict is JudgeVerdict.ACCEPTED
    assert resp.model_id == "claude-opus-4-7"  # served model from model_usage
    assert resp.prompt_tokens_total == 200
    assert resp.prompt_tokens_cached == 50


def test_agent_transport_preserves_none_cached_distinction(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provider reported a total but NO cached-token count: cached must stay
    # None (absence), not be fabricated to 0. The None-vs-0 distinction is
    # load-bearing for the audit trail per the JudgeResponse docstring.
    _install_fake_sdk(
        monkeypatch,
        assistant_text=_GOOD_JSON,
        usage={"input_tokens": 200},
    )
    resp = call_judge(_request(), transport=TRANSPORT_AGENT)
    assert resp.prompt_tokens_total == 200
    assert resp.prompt_tokens_cached is None


def test_agent_transport_preserves_zero_cached_distinction(monkeypatch: pytest.MonkeyPatch) -> None:
    # Caching on, no hit: cached == 0 must survive as 0 (distinct from None).
    _install_fake_sdk(
        monkeypatch,
        assistant_text=_GOOD_JSON,
        usage={"input_tokens": 200, "cache_read_input_tokens": 0},
    )
    resp = call_judge(_request(), transport=TRANSPORT_AGENT)
    assert resp.prompt_tokens_cached == 0


def test_agent_transport_empty_model_usage_falls_back_to_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    # model_usage present but empty: fall back to the requested id rather than
    # fabricate a served id (mirrors OpenRouter served-vs-requested rule C1-1).
    _install_fake_sdk(
        monkeypatch,
        assistant_text=_GOOD_JSON,
        model_usage={},
    )
    resp = call_judge(_request(), transport=TRANSPORT_AGENT, model_id="claude-sonnet-4-7")
    assert resp.model_id == "claude-sonnet-4-7"


def test_agent_transport_multi_model_usage_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    # A single-shot judge call must resolve to exactly one served model;
    # more than one key is an unexpected shape — crash, don't fabricate.
    _install_fake_sdk(
        monkeypatch,
        assistant_text=_GOOD_JSON,
        model_usage={"a": {"input_tokens": 1}, "b": {"input_tokens": 1}},
    )
    # A response-shape violation: JudgeContractError (not transport). It must
    # propagate unchanged through _call_agent_sdk's except arms, not be remapped.
    with pytest.raises(JudgeContractError):
        call_judge(_request(), transport=TRANSPORT_AGENT)


def test_agent_transport_missing_result_message_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    # No ResultMessage => no usage accounting => contract violation (not a
    # transport failure; propagates unchanged through the except arms).
    _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON, emit_result=False)
    with pytest.raises(JudgeContractError):
        call_judge(_request(), transport=TRANSPORT_AGENT)


def test_agent_transport_missing_sdk_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "claude_agent_sdk", raising=False)
    monkeypatch.setattr("builtins.__import__", _import_raising("claude_agent_sdk"))
    with pytest.raises(JudgeConfigurationError, match="judge-agent"):
        call_judge(_request(), transport=TRANSPORT_AGENT)


def test_agent_transport_auth_failure_names_auth_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real SDK raises CLINotFoundError (a ClaudeSDKError subclass) when the
    # Claude Code CLI is absent — the operator-actionable config signal. The
    # fake attaches the real class names; we raise CLINotFoundError here so the
    # test exercises the same isinstance discrimination the real SDK would hit.
    mod = _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON)
    cli_not_found = mod.CLINotFoundError("Claude Code not found")
    _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON, raise_on_query=cli_not_found)
    with pytest.raises(JudgeConfigurationError, match=r"ANTHROPIC_API_KEY|Claude Code"):
        call_judge(_request(), transport=TRANSPORT_AGENT)


def test_agent_transport_inband_auth_error_names_auth_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real auth failure can also surface IN-BAND as AssistantMessage.error
    # == 'authentication_failed' (a non-exception path the plan's exception-
    # only model didn't anticipate). Map it to the auth-actionable config
    # error too, not the generic "no assistant text" contract error.
    _install_fake_sdk(monkeypatch, assistant_text="", assistant_error="authentication_failed")
    with pytest.raises(JudgeConfigurationError, match=r"ANTHROPIC_API_KEY|Claude Code"):
        call_judge(_request(), transport=TRANSPORT_AGENT)


def test_agent_transport_generic_sdk_error_is_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ProcessError (CLI process failed for a non-auth reason) is a transport
    # failure after configuration, NOT operator-actionable config.
    mod = _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON)
    proc_error = mod.ProcessError("exited 1")
    _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON, raise_on_query=proc_error)
    with pytest.raises(JudgeTransportError):
        call_judge(_request(), transport=TRANSPORT_AGENT)


# --- C1: exhaustive in-band AssistantMessage.error classification ---------
#
# The real Literal set is {authentication_failed, billing_error, rate_limit,
# invalid_request, server_error, unknown}. Auth/billing -> config; every other
# literal -> transport (not a malformed-verdict crash).


def test_agent_transport_inband_billing_error_names_billing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # billing_error is config-class (operator-actionable) but the wording must
    # be about billing, not "log in".
    _install_fake_sdk(monkeypatch, assistant_text="", assistant_error="billing_error")
    with pytest.raises(JudgeConfigurationError, match=r"billing"):
        call_judge(_request(), transport=TRANSPORT_AGENT)


@pytest.mark.parametrize("literal", ["rate_limit", "server_error", "invalid_request", "unknown"])
def test_agent_transport_inband_nonauth_error_is_transport_error(monkeypatch: pytest.MonkeyPatch, literal: str) -> None:
    # A non-auth in-band error (transient / provider-side) must be a transport
    # failure carrying the specific literal — NOT a "no assistant text"
    # contract crash that discards the real cause.
    _install_fake_sdk(monkeypatch, assistant_text="", assistant_error=literal)
    with pytest.raises(JudgeTransportError, match=literal):
        call_judge(_request(), transport=TRANSPORT_AGENT)


def test_agent_transport_errored_result_message_with_no_text_is_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An errored terminal ResultMessage (is_error / api_error_status) with no
    # assistant text is a transport fault, not a malformed verdict. The
    # api_error_status must surface in the message so the operator sees the cause.
    _install_fake_sdk(
        monkeypatch,
        assistant_text="",
        is_error=True,
        api_error_status=503,
        errors=["upstream unavailable"],
    )
    with pytest.raises(JudgeTransportError, match=r"503|api_error_status"):
        call_judge(_request(), transport=TRANSPORT_AGENT)


def test_agent_transport_empty_text_non_errored_result_is_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The genuinely-empty, non-errored case: the model returned nothing usable
    # as a verdict. This is the one path that stays JudgeContractError.
    _install_fake_sdk(monkeypatch, assistant_text="")
    with pytest.raises(JudgeContractError, match=r"no assistant text"):
        call_judge(_request(), transport=TRANSPORT_AGENT)


# --- C2: the offensive None-guards must be reachable + tested -------------


def test_agent_transport_usage_none_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    # usage is dict | None on the real SDK; None on a completed call cannot be
    # accounted -> JudgeContractError. (_UNSET sentinel lets us inject a real None.)
    _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON, usage=None)
    with pytest.raises(JudgeContractError, match=r"usage is None"):
        call_judge(_request(), transport=TRANSPORT_AGENT)


def test_agent_transport_model_usage_none_falls_back_to_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    # model_usage is dict | None; None (absent) falls back to the requested id
    # rather than fabricating a served id.
    _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON, model_usage=None)
    resp = call_judge(_request(), transport=TRANSPORT_AGENT, model_id="claude-sonnet-4-7")
    assert resp.model_id == "claude-sonnet-4-7"


# --- I1: a missing usage key is a contract violation, not a transport error -


def test_agent_transport_missing_input_tokens_key_is_contract_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A usage dict present but missing 'input_tokens' is the same malformed-
    # usage fault class as usage=None -> JudgeContractError (consistent), not a
    # bare KeyError silently reclassified as a transport error.
    _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON, usage={"output_tokens": 10})
    with pytest.raises(JudgeContractError, match=r"input_tokens"):
        call_judge(_request(), transport=TRANSPORT_AGENT)
