"""Tests for the read-only tool-augmented ("investigation") judge transport.

Covers the four moving parts of elspeth-ab5e093fa3:

1. ``AgentToolScope`` / ``build_readonly_tool_scope`` — the scope object.
2. ``_tool_scope_decision`` + ``_build_pretooluse_scope_hook`` — the
   fail-closed PreToolUse guard (THE load-bearing security boundary). The
   guard logic is unit-tested exhaustively here, including a real symlink
   escape; the proof that the live SDK actually *invokes* the hook is the
   acceptance spike recorded on the ticket (it needs operator credentials and
   so cannot run in CI).
3. ``_extract_trailing_verdict_json`` + the tool-mode drain — extracting the
   verdict from a final message that also carries investigation narration,
   and classifying turn-budget exhaustion.
4. Transport wiring — ``_call_agent_sdk(tool_scope=...)`` builds the streaming
   + hook-guarded options; ``_call_openrouter`` / ``call_judge`` reject a tool
   scope on the OpenRouter path; the blinded path is unchanged.

Like ``test_judge_agent_transport``, these inject a FAKE ``claude_agent_sdk``
so they never make a real agent query (which would need operator credentials
CI must never seek).
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict
from elspeth_lints.core.judge import (
    JUDGE_POLICY_HASH,
    TRANSPORT_AGENT,
    TRANSPORT_OPENROUTER,
    AgentToolScope,
    JudgeContractError,
    JudgeRequest,
    _build_pretooluse_scope_hook,
    _call_openrouter,
    _extract_trailing_verdict_json,
    _tool_scope_decision,
    build_readonly_tool_scope,
    call_judge,
)

_VERDICT_JSON = (
    '{"verdict": "ACCEPTED", "rationale": "PRESCRIBED FORM; offensive isinstance->raise at '
    'the construction boundary (composer_interpretation.py:121).", "confidence": 0.8, '
    '"should_use_decorator": null}'
)


def _request() -> JudgeRequest:
    return JudgeRequest(
        file_path="contracts/x.py",
        rule_id="R5",
        symbol="_validate",
        fingerprint="abc",
        rationale="construction boundary",
        surrounding_code="def _validate(v: object) -> None:\n    if not isinstance(v, E):\n        raise ValueError()\n",
    )


@pytest.fixture
def scope(tmp_path: Path) -> AgentToolScope:
    src = tmp_path / "src" / "elspeth"
    allow = tmp_path / "config" / "cicd" / "enforce_tier_model"
    src.mkdir(parents=True)
    allow.mkdir(parents=True)
    return build_readonly_tool_scope(root=src, allowlist_dir=allow)


# --------------------------------------------------------------------------
# AgentToolScope / build_readonly_tool_scope
# --------------------------------------------------------------------------


def test_build_readonly_tool_scope_roots_and_cwd(tmp_path: Path) -> None:
    src = tmp_path / "src" / "elspeth"
    allow = tmp_path / "allow"
    src.mkdir(parents=True)
    allow.mkdir()
    s = build_readonly_tool_scope(root=src, allowlist_dir=allow)
    # roots are realpath-resolved; cwd is the source root (a valid allowed root).
    assert s.cwd == Path(os.path.realpath(src))
    assert s.cwd in s.allowed_roots
    assert Path(os.path.realpath(allow)) in s.allowed_roots
    assert s.max_turns > 0


def test_agent_tool_scope_rejects_empty_roots() -> None:
    with pytest.raises(ValueError, match="at least one allowed root"):
        AgentToolScope(allowed_roots=(), cwd=Path("/x"), max_turns=4)


def test_agent_tool_scope_rejects_cwd_outside_roots() -> None:
    with pytest.raises(ValueError, match="must be one of allowed_roots"):
        AgentToolScope(allowed_roots=(Path("/a"),), cwd=Path("/b"), max_turns=4)


def test_agent_tool_scope_rejects_nonpositive_turns() -> None:
    with pytest.raises(ValueError, match="max_turns must be positive"):
        AgentToolScope(allowed_roots=(Path("/a"),), cwd=Path("/a"), max_turns=0)


# --------------------------------------------------------------------------
# _tool_scope_decision — the fail-closed guard
# --------------------------------------------------------------------------


def test_guard_allows_in_root_read(scope: AgentToolScope) -> None:
    target = str(scope.cwd / "contracts" / "x.py")
    allowed, _ = _tool_scope_decision(scope, "Read", {"file_path": target})
    assert allowed is True


def test_guard_allows_allowlist_dir_read(scope: AgentToolScope) -> None:
    allow_root = next(r for r in scope.allowed_roots if r != scope.cwd)
    allowed, _ = _tool_scope_decision(scope, "Read", {"file_path": str(allow_root / "web.yaml")})
    assert allowed is True


def test_guard_denies_dot_env(scope: AgentToolScope) -> None:
    # An .env physically inside the root is still denied by the basename guard.
    allowed, reason = _tool_scope_decision(scope, "Read", {"file_path": str(scope.cwd / ".env")})
    assert allowed is False
    assert "forbidden file" in reason


def test_guard_denies_out_of_root_absolute(scope: AgentToolScope) -> None:
    allowed, reason = _tool_scope_decision(scope, "Read", {"file_path": "/etc/passwd"})
    assert allowed is False
    assert "outside the permitted roots" in reason


def test_guard_denies_parent_traversal(scope: AgentToolScope) -> None:
    # cwd is .../src/elspeth; climb out to a sibling secret.
    allowed, _ = _tool_scope_decision(scope, "Read", {"file_path": "../../secret.txt"})
    assert allowed is False


def test_guard_fail_closed_on_missing_file_path(scope: AgentToolScope) -> None:
    allowed, reason = _tool_scope_decision(scope, "Read", {})
    assert allowed is False
    assert "fail-closed" in reason


def test_guard_denies_non_read_tools(scope: AgentToolScope) -> None:
    for tool, inp in [
        ("Bash", {"command": "cat /etc/passwd"}),
        ("Write", {"file_path": str(scope.cwd / "x.py")}),
        ("Edit", {"file_path": str(scope.cwd / "x.py")}),
        ("WebFetch", {"url": "http://evil"}),
        ("Task", {"prompt": "x"}),
    ]:
        allowed, reason = _tool_scope_decision(scope, tool, inp)
        assert allowed is False, f"{tool} should be denied"
        assert "not permitted" in reason


def test_guard_allows_pathless_grep_and_glob(scope: AgentToolScope) -> None:
    # No 'path' -> defaults to cwd, which is an allowed root.
    assert _tool_scope_decision(scope, "Grep", {"pattern": "x"})[0] is True
    assert _tool_scope_decision(scope, "Glob", {"pattern": "**/*.py"})[0] is True


def test_guard_denies_out_of_root_grep_path(scope: AgentToolScope) -> None:
    allowed, _ = _tool_scope_decision(scope, "Grep", {"pattern": "x", "path": "/etc"})
    assert allowed is False


def test_guard_denies_symlink_escape(scope: AgentToolScope, tmp_path: Path) -> None:
    # A symlink physically inside the root, pointing OUT to a secret, must be
    # denied — realpath resolution is the load-bearing mechanism.
    secret = tmp_path / "secret_outside.txt"
    secret.write_text("token")
    link = scope.cwd / "innocent_link"
    os.symlink(secret, link)
    allowed, _ = _tool_scope_decision(scope, "Read", {"file_path": str(link)})
    assert allowed is False


def test_guard_allows_in_root_symlink(scope: AgentToolScope) -> None:
    real = scope.cwd / "real.py"
    real.write_text("x = 1\n")
    link = scope.cwd / "inner_link"
    os.symlink(real, link)
    assert _tool_scope_decision(scope, "Read", {"file_path": str(link)})[0] is True


# --------------------------------------------------------------------------
# PreToolUse hook — returns the SDK's decision dict, never raises
# --------------------------------------------------------------------------


def _run_hook(scope: AgentToolScope, payload: dict[str, object]) -> dict:
    hook = _build_pretooluse_scope_hook(scope)
    return asyncio.run(hook(payload, None, None))


def test_hook_allows_in_root(scope: AgentToolScope) -> None:
    out = _run_hook(scope, {"tool_name": "Read", "tool_input": {"file_path": str(scope.cwd / "a.py")}})
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_hook_denies_out_of_root(scope: AgentToolScope) -> None:
    out = _run_hook(scope, {"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}})
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"


def test_hook_fail_closed_on_malformed_input(scope: AgentToolScope) -> None:
    # Missing tool_name / tool_input -> deny, never raise.
    out = _run_hook(scope, {"not_a_tool": 1})
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


# --------------------------------------------------------------------------
# _extract_trailing_verdict_json
# --------------------------------------------------------------------------


def test_extract_pure_json() -> None:
    assert _extract_trailing_verdict_json(_VERDICT_JSON) == _VERDICT_JSON


def test_extract_trailing_after_prose() -> None:
    raw = "Let me check the callers.\nThe helper is called from __post_init__.\n\n" + _VERDICT_JSON
    extracted = _extract_trailing_verdict_json(raw)
    assert extracted == _VERDICT_JSON


def test_extract_prose_only_returns_none() -> None:
    assert _extract_trailing_verdict_json("I cannot decide without more context.") is None


def test_extract_empty_returns_none() -> None:
    assert _extract_trailing_verdict_json("   ") is None


def test_extract_ignores_braces_in_prose() -> None:
    raw = "the literal {a: 1} is irrelevant. Verdict:\n" + _VERDICT_JSON
    extracted = _extract_trailing_verdict_json(raw)
    assert extracted is not None and extracted.endswith("}")
    assert '"verdict"' in extracted


# --------------------------------------------------------------------------
# Transport wiring — fake SDK exercising the tool-mode path
# --------------------------------------------------------------------------


def _install_tool_fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    messages: list[tuple[str, bool]],
    num_turns: int = 2,
    served: str = "claude-opus-4-7",
    capture: dict | None = None,
) -> types.ModuleType:
    """Fake SDK whose ``query`` yields a sequence of assistant messages.

    ``messages`` is a list of (text, has_tool_use) pairs emitted in order,
    followed by a ResultMessage carrying ``num_turns``. ``capture`` (if given)
    records the ClaudeAgentOptions kwargs and the prompt object so wiring can
    be asserted.
    """
    mod = types.ModuleType("claude_agent_sdk")

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            if capture is not None:
                capture["options"] = kwargs

    class TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class ToolUseBlock:
        def __init__(self, name: str = "Read", tool_input: dict | None = None) -> None:
            self.name = name
            self.input = tool_input or {}

    class HookMatcher:
        def __init__(self, *, matcher: object = None, hooks: list | None = None, timeout: object = None) -> None:
            self.matcher = matcher
            self.hooks = hooks or []

    class AssistantMessage:
        def __init__(self, content: list[object], error: str | None = None) -> None:
            self.content = content
            self.error = error

    class ResultMessage:
        def __init__(self) -> None:
            self.usage = {"input_tokens": 100, "cache_read_input_tokens": 20}
            self.model_usage = {served: {"input_tokens": 100}}
            self.is_error = False
            self.api_error_status = None
            self.errors = None
            self.num_turns = num_turns

    class ClaudeSDKError(Exception):
        pass

    async def query(*, prompt: object, options: object) -> AsyncIterator[object]:
        if capture is not None:
            capture["prompt"] = prompt
        for text, has_tool in messages:
            content: list[object] = []
            if has_tool:
                content.append(ToolUseBlock())
            if text:
                content.append(TextBlock(text=text))
            yield AssistantMessage(content=content)
        yield ResultMessage()

    mod.ClaudeAgentOptions = ClaudeAgentOptions  # type: ignore[attr-defined]
    mod.TextBlock = TextBlock  # type: ignore[attr-defined]
    mod.ToolUseBlock = ToolUseBlock  # type: ignore[attr-defined]
    mod.HookMatcher = HookMatcher  # type: ignore[attr-defined]
    mod.AssistantMessage = AssistantMessage  # type: ignore[attr-defined]
    mod.ResultMessage = ResultMessage  # type: ignore[attr-defined]
    mod.ClaudeSDKError = ClaudeSDKError  # type: ignore[attr-defined]
    mod.query = query  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
    return mod


def test_tool_mode_extracts_verdict_from_final_narrated_message(monkeypatch: pytest.MonkeyPatch, scope: AgentToolScope) -> None:
    # Turn 1: narration + tool use (discarded). Turn 2: narration + trailing verdict.
    _install_tool_fake_sdk(
        monkeypatch,
        messages=[
            ("Let me read the callers.", True),
            ("After reading: this is the construction boundary.\n\n" + _VERDICT_JSON, False),
        ],
    )
    resp = call_judge(_request(), transport=TRANSPORT_AGENT, tool_scope=scope)
    assert resp.verdict is JudgeVerdict.ACCEPTED
    assert resp.judge_transport == TRANSPORT_AGENT


def test_tool_mode_builds_streaming_hook_guarded_options(monkeypatch: pytest.MonkeyPatch, scope: AgentToolScope) -> None:
    capture: dict = {}
    _install_tool_fake_sdk(
        monkeypatch,
        messages=[("done\n" + _VERDICT_JSON, False)],
        capture=capture,
    )
    call_judge(_request(), transport=TRANSPORT_AGENT, tool_scope=scope)
    opts = capture["options"]
    # Read/Grep/Glob are NOT auto-approved (would bypass the hook) and NOT
    # disallowed (the hook governs them); Bash/Write/Edit/Web* are hard-denied.
    assert opts["allowed_tools"] == []
    assert "Read" not in opts["disallowed_tools"]
    assert {"Bash", "Write", "Edit"} <= set(opts["disallowed_tools"])
    assert "PreToolUse" in opts["hooks"]
    assert opts["permission_mode"] == "default"
    assert opts["max_turns"] == scope.max_turns
    assert opts["cwd"] == str(scope.cwd)
    # streaming-input prompt (async iterable), required for the hook to fire.
    assert hasattr(capture["prompt"], "__aiter__")
    # tool-mode addendum rides OUTSIDE the hashed policy block.
    assert "TOOL-AUGMENTED INVESTIGATION MODE" in opts["system_prompt"]["append"]


def test_tool_mode_policy_hash_unchanged() -> None:
    # The signed corpus must not need re-signing because of tool mode.
    assert JUDGE_POLICY_HASH == "sha256:08052cb8f2c263c39dc61336444e6f2b2859292e283a902510827744f18d68da"


def test_tool_mode_turn_budget_exhaustion_is_classified(monkeypatch: pytest.MonkeyPatch, scope: AgentToolScope) -> None:
    # Final message is prose-only (no verdict) and the cap was hit.
    _install_tool_fake_sdk(
        monkeypatch,
        messages=[("still investigating, ran out of turns", True)],
        num_turns=scope.max_turns,
    )
    with pytest.raises(JudgeContractError, match="turn budget"):
        call_judge(_request(), transport=TRANSPORT_AGENT, tool_scope=scope)


def test_tool_mode_no_verdict_without_exhaustion_is_contract_error(monkeypatch: pytest.MonkeyPatch, scope: AgentToolScope) -> None:
    # Prose-only final message but turns NOT exhausted -> distinct contract error.
    _install_tool_fake_sdk(
        monkeypatch,
        messages=[("I decline to decide.", False)],
        num_turns=1,
    )
    with pytest.raises(JudgeContractError, match="no trailing verdict"):
        call_judge(_request(), transport=TRANSPORT_AGENT, tool_scope=scope)


# --------------------------------------------------------------------------
# OpenRouter / call_judge reject a tool scope
# --------------------------------------------------------------------------


def test_openrouter_transport_rejects_tool_scope(scope: AgentToolScope) -> None:
    with pytest.raises(ValueError, match="cannot use judge tools"):
        _call_openrouter(_request(), "anthropic/claude-opus-4-7", 1024, tool_scope=scope)


def test_call_judge_openrouter_with_tool_scope_raises(scope: AgentToolScope) -> None:
    with pytest.raises(ValueError, match="cannot use judge tools"):
        call_judge(_request(), transport=TRANSPORT_OPENROUTER, tool_scope=scope)


def test_call_judge_threads_scope_only_when_set() -> None:
    # Backward-compat: with no tool_scope, a 3-arg fake transport_impl (the
    # historical signature) is invoked unchanged.
    seen: dict = {}

    def fake_impl(request: JudgeRequest, model_id: str, max_tokens: int) -> object:
        seen["args"] = (model_id, max_tokens)
        from elspeth_lints.core.judge import _TransportResult

        return _TransportResult(raw_text=_VERDICT_JSON, served_model_id="m", prompt_tokens_total=10, prompt_tokens_cached=None)

    resp = call_judge(_request(), transport=TRANSPORT_AGENT, transport_impl=fake_impl)
    assert resp.verdict is JudgeVerdict.ACCEPTED
    assert seen["args"][1] > 0


# --------------------------------------------------------------------------
# CLI rejection: --judge-tools readonly requires --judge-transport agent
# --------------------------------------------------------------------------


def test_cli_readonly_with_openrouter_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    from elspeth_lints.core.cli import _run_reaudit

    args = argparse.Namespace(
        allowlist_dir=tmp_path,
        root=tmp_path,
        judge_transport="openrouter",
        judge_tools="readonly",
    )
    assert _run_reaudit(args) == 2
    assert "requires --judge-transport agent" in capsys.readouterr().err


def test_cli_justify_readonly_with_openrouter_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Parity: justify (the signing path) accepts --judge-tools just like reaudit,
    # and rejects readonly+openrouter the same way, before any finding scan / HMAC.
    import argparse

    from elspeth_lints.core.cli import _run_justify

    args = argparse.Namespace(
        root=tmp_path,
        repo_root=None,
        allowlist_dir=tmp_path,
        judge_transport="openrouter",
        judge_tools="readonly",
    )
    assert _run_justify(args) == 2
    assert "requires --judge-transport agent" in capsys.readouterr().err
