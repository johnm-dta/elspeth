# Claude-Agent-SDK Judge Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable `--judge-transport agent` path that routes the cicd-judge verdict call through the **Claude Agent SDK** instead of OpenRouter — a cheaper way to produce the *same* `JudgeResponse`, with the transport honestly recorded in the signed audit record.

**Architecture:** `call_judge` becomes a thin dispatcher over a transport seam. Both transports funnel their extracted assistant text through the *identical* `_parse_judge_payload` → validators → `JudgeResponse` path, so a verdict is validated the same way regardless of origin. A new `judge_transport` field (`"openrouter"` / `"claude_agent_sdk"`) is stamped on `JudgeResponse` and carried **inside the HMAC-signed v2 allowlist payload** introduced by the scope-fingerprint plan. OpenRouter stays the default; agent is opt-in.

**Tech Stack:** Python 3.13, `claude-agent-sdk` (new optional dep), `asyncio`, the existing OpenAI/OpenRouter path, `hmac`/`hashlib`, dataclasses, pytest. All code lives under `elspeth-lints/src/elspeth_lints/`. Run tests from `/home/john/elspeth/elspeth-lints` with the repo venv: `../.venv/bin/python -m pytest`.

---

## Dependency & Sequencing (read before starting)

This feature shares the **v2 signed payload** with the scope-fingerprint work (design §5). It is **not** an independent payload change. Two hard ordering constraints:

1. **Lands after** the scope-fingerprint plan's agent-buildable tasks (`docs/superpowers/plans/2026-05-31-judge-scope-fingerprint.md`, Tasks 1–11). Those introduce `signature_version`, `scope_fingerprint`, the version-dispatched `compute_judge_metadata_signature`, the version-aware atomic validator, and the `migrate-judge-scope` command. **This plan modifies that machinery; it does not recreate it.** Task 1 Step 1 verifies the precondition mechanically and halts if absent.

2. **Lands before** the scope-fingerprint plan's operator migration (its Task 12). `judge_transport` joins the v2 payload, and the scope-fingerprint migration is the single event that rewrites all 221 live entries from v1 to v2. Folding `judge_transport` into the migrate backfill (this plan's Task 4) **before** the operator runs it means the corpus migrates **once** to the complete v2 schema (`scope_fingerprint` + `judge_transport`), not twice. If this feature shipped *after* the migration, every freshly-migrated v2 entry would need re-signing to add `judge_transport` — a second keyed operator pass for no reason.

> **If the scope-fingerprint work is genuinely deferred and cannot land first:** do not proceed with this plan as written. The two features must be reconciled to a single v2 payload schema before either ships (design §5). Surface this to the operator rather than minting an independent, conflicting payload bump.

### Operator boundary (same custody rule as scope-fingerprint)

The HMAC signing key (`ELSPETH_JUDGE_METADATA_HMAC_KEY`) is **operator-only and MUST NOT be in any agent's environment** (CLAUDE.md "CICD Judge Gate: HMAC Key Custody"). **Every task in this plan (1–7) is agent-buildable and green-between-commits** using a throwaway test key injected via the existing `hmac_key=` parameter / `monkeypatch.setenv`. The actual re-signing of the 221 live entries — including the `judge_transport="openrouter"` backfill — happens inside the scope-fingerprint plan's **operator-only Task 12**, which this plan's Task 4 prepares. An agent may *propose* that migration but must not execute it.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `elspeth-lints/pyproject.toml` | A `[judge-agent]` optional-dependency extra for `claude-agent-sdk`, mirroring the existing `[judge]` extra. The lint tool stays installable without either SDK. | Modify |
| `core/judge.py` | The transport seam. Factor the OpenRouter body into `_call_openrouter`; add `_call_agent_sdk`; add the `_TransportResult` carrier, transport constants, and the `transport` / `transport_impl` parameters on `call_judge`; add `judge_transport` to `JudgeResponse`. | Modify |
| `core/allowlist.py` | `judge_transport` field on `AllowlistEntry`; loader parse; the field added to the **v2** signed payload; the version-aware atomic validator requires it on v2 and forbids it on v1 / pre-judge. | Modify |
| `core/cli.py` | `--judge-transport {openrouter,agent}` on `justify` and `reaudit`; thread the selected transport into both `call_judge` sites; `_build_yaml_entry_text` emits and signs `judge_transport`; `migrate-judge-scope` backfills `judge_transport="openrouter"`. | Modify |
| `core/reaudit.py` | Thread the chosen transport from `_run_reaudit` down to the per-entry `call_judge(request)` at the judge boundary. | Modify |
| `core/reaudit_sidecar.py` | Round-trip `judge_transport` through the crash-recovery sidecar. | Modify |
| `core/judge_coverage.py` | `judge_transport` participates in `_judge_metadata_payload` (verdict metadata), **not** `_judge_binding_identity` (source binding). | Modify |

The new hash/transport logic deliberately reuses the *single* shared response-contract layer (`_parse_judge_payload`, `_verdict_from_string`, the field validators, the `should_use_decorator`↔`BLOCKED` cross-check). Divergent verdict coercion between transports is the highest-severity failure mode and is structurally prevented: only text extraction and usage accounting differ per transport.

---

## Transport contract (normative — frozen here)

Two transport identities exist. The CLI flag spelling differs from the persisted/signed value:

| CLI flag value | `JudgeResponse.judge_transport` / signed value | Meaning |
|----------------|-----------------------------------------------|---------|
| `openrouter` (default) | `"openrouter"` | OpenAI-compatible SDK → OpenRouter, `temperature=0`. |
| `agent` | `"claude_agent_sdk"` | `claude_agent_sdk.query`, no tools, `claude_code` preset. |

A transport implementation is a callable `(_TransportResult) = impl(request: JudgeRequest, model_id: str, max_tokens: int)`. `call_judge` selects it by name, runs it, then applies the shared validation. Tests inject a fake `impl` at the seam so **no unit test makes a real SDK or network call** — identical posture to the OpenRouter path today.

---

## Task 1: Transport seam refactor + `[judge-agent]` extra (OpenRouter behaviour unchanged)

This is a pure refactor plus a new dataclass field. After it, OpenRouter is still the only working transport, still the default, and the verdict path is byte-for-byte equivalent — the only observable change is a new `judge_transport="openrouter"` on every `JudgeResponse`.

**Files:**
- Modify: `elspeth-lints/pyproject.toml`
- Modify: `elspeth-lints/src/elspeth_lints/core/judge.py`
- Test: `elspeth-lints/tests/core/test_judge.py` (confirm filename with `ls tests/core | grep -i judge`)

- [ ] **Step 1: Verify the scope-fingerprint v2 precondition (halt if absent)**

This plan modifies v2 machinery that the scope-fingerprint plan must have landed first.

Run:
```bash
cd /home/john/elspeth/elspeth-lints && grep -q "signature_version" src/elspeth_lints/core/allowlist.py \
  && grep -q "_run_migrate_judge_scope" src/elspeth_lints/core/cli.py \
  && echo "PRECONDITION OK: v2 payload + migrate command present" \
  || echo "STOP: scope-fingerprint Tasks 1-11 are not landed; do not proceed (see Dependency & Sequencing)"
```
Expected: `PRECONDITION OK`. If `STOP`, halt and surface to the operator — the rest of this plan edits code that does not yet exist.

- [ ] **Step 2: Write the failing tests**

```python
# in tests/core/test_judge.py
from elspeth_lints.core.judge import (
    DEFAULT_JUDGE_MODEL,
    JudgeRequest,
    JudgeResponse,
    TRANSPORT_AGENT,
    TRANSPORT_OPENROUTER,
    _TransportResult,
    call_judge,
)
from elspeth_lints.core.allowlist import JudgeVerdict


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
    '{"verdict": "ACCEPTED", "rationale": "external boundary; coercion recorded", '
    '"confidence": 0.8, "should_use_decorator": null}'
)


def _fake_transport(raw_text: str, served: str = "anthropic/claude-opus-4-7"):
    def _impl(request, model_id, max_tokens):
        assert isinstance(request, JudgeRequest)
        return _TransportResult(
            raw_text=raw_text,
            served_model_id=served,
            prompt_tokens_total=123,
            prompt_tokens_cached=10,
        )

    return _impl


def test_call_judge_stamps_openrouter_transport_by_default() -> None:
    resp = call_judge(_request(), transport_impl=_fake_transport(_GOOD_JSON))
    assert resp.judge_transport == TRANSPORT_OPENROUTER
    assert resp.verdict is JudgeVerdict.ACCEPTED
    assert resp.model_id == "anthropic/claude-opus-4-7"
    assert resp.prompt_tokens_total == 123
    assert resp.prompt_tokens_cached == 10


def test_call_judge_records_selected_transport_name() -> None:
    resp = call_judge(
        _request(), transport=TRANSPORT_AGENT, transport_impl=_fake_transport(_GOOD_JSON)
    )
    assert resp.judge_transport == TRANSPORT_AGENT


def test_call_judge_rejects_unknown_transport() -> None:
    import pytest

    with pytest.raises(ValueError, match="transport"):
        call_judge(_request(), transport="carrier-pigeon", transport_impl=_fake_transport(_GOOD_JSON))


def test_both_transports_share_one_validation_path() -> None:
    # The shared parser's should_use_decorator<->BLOCKED cross-check must fire
    # identically regardless of transport origin.
    import pytest

    from elspeth_lints.core.judge import JudgeContractError

    bad = '{"verdict": "ACCEPTED", "rationale": "x", "confidence": 0.5, "should_use_decorator": "arguments"}'
    with pytest.raises(JudgeContractError):
        call_judge(_request(), transport=TRANSPORT_AGENT, transport_impl=_fake_transport(bad))


def _capture_model_impl(captured: dict):
    def _impl(request, model_id, max_tokens):
        captured["model_id"] = model_id
        return _TransportResult(
            raw_text=_GOOD_JSON, served_model_id="claude-opus-4-7", prompt_tokens_total=1, prompt_tokens_cached=None
        )

    return _impl


def test_openrouter_default_model_keeps_vendor_slug() -> None:
    captured: dict = {}
    call_judge(_request(), transport_impl=_capture_model_impl(captured))
    assert captured["model_id"] == "anthropic/claude-opus-4-7"  # OpenRouter routing slug


def test_agent_transport_default_model_is_unprefixed() -> None:
    # The agent transport must NOT receive the OpenRouter "anthropic/..." slug —
    # the SDK rejects it. call_judge resolves the per-transport default.
    captured: dict = {}
    call_judge(_request(), transport=TRANSPORT_AGENT, transport_impl=_capture_model_impl(captured))
    assert "/" not in captured["model_id"]


def test_explicit_model_id_overrides_transport_default() -> None:
    captured: dict = {}
    call_judge(_request(), model_id="some/explicit-model", transport_impl=_capture_model_impl(captured))
    assert captured["model_id"] == "some/explicit-model"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_judge.py -k "transport" -q`
Expected: FAIL — `ImportError: cannot import name 'TRANSPORT_OPENROUTER'` (and `_TransportResult`).

- [ ] **Step 4: Add transport constants, the result carrier, and the `judge_transport` field**

In `judge.py`, after `DEFAULT_JUDGE_MAX_TOKENS` (:55) add:

```python
# Transport identities. The persisted/signed values are these strings; the
# CLI flag spelling ("openrouter" / "agent") maps onto them in cli.py.
TRANSPORT_OPENROUTER: str = "openrouter"
TRANSPORT_AGENT: str = "claude_agent_sdk"
_VALID_TRANSPORTS: frozenset[str] = frozenset({TRANSPORT_OPENROUTER, TRANSPORT_AGENT})

# Per-transport default model. CRITICAL: DEFAULT_JUDGE_MODEL is an OpenRouter
# *routing slug* ("anthropic/claude-opus-4-7" — the vendor prefix is required
# by OpenRouter, see :54). The Claude Agent SDK (Claude Code CLI /
# ANTHROPIC_API_KEY) expects an UNPREFIXED Anthropic/Claude-Code model id and
# will reject the slug. Each transport therefore has its own default; call_judge
# resolves by transport when the caller passes no explicit model_id.
DEFAULT_AGENT_JUDGE_MODEL: str = "claude-opus-4-7"  # confirm the SDK-accepted id post-install
```

> Confirm the exact model id `ClaudeAgentOptions(model=...)` accepts against the installed SDK (it may be `"claude-opus-4-7"`, a `claude-code`-namespaced alias, or similar). The defect this guards against: passing `"anthropic/claude-opus-4-7"` to the agent path breaks at runtime while the fake-SDK tests — which accept any string — stay green. The capturing test in Step 2 (`test_agent_transport_default_model_is_unprefixed`) is the regression guard, and it exercises `call_judge`'s resolution, which is in our control rather than the SDK's.

Add the result carrier near the error classes (after `JudgeContractError` at :632):

```python
@dataclass(frozen=True, slots=True)
class _TransportResult:
    """What a transport extracts from its provider response.

    The transport-specific code (OpenRouter SDK vs Claude Agent SDK) reduces
    its provider response to exactly these four values; everything downstream
    of here (verdict parsing, contract validation, JudgeResponse construction)
    is shared and transport-agnostic.

    ``prompt_tokens_cached`` preserves the ``None``-vs-``0`` distinction the
    JudgeResponse docstring is explicit about: ``None`` = the provider did not
    report a cached-token count; ``0`` = caching was on but produced no hit.
    """

    raw_text: str
    served_model_id: str
    prompt_tokens_total: int
    prompt_tokens_cached: int | None
```

Add the `judge_transport` field to `JudgeResponse` (after `policy_hash: str` at :719):

```python
    policy_hash: str
    judge_transport: str
```

Extend the `JudgeResponse` docstring (after the cache-accounting paragraph) with:

```
    ``judge_transport`` records which transport produced this verdict
    (``"openrouter"`` or ``"claude_agent_sdk"``). It is carried into the
    HMAC-signed v2 allowlist payload — "how the verdict was produced" is
    verdict metadata, bound to and tamper-evident with the verdict itself.
    The era of any provider-side system prompt (the ``claude_code`` preset
    under the agent transport) is bounded by the already-signed
    ``recorded_at`` timestamp; no separate version is captured.
```

- [ ] **Step 5: Extract `_call_openrouter` from the current `call_judge` body**

Create `_call_openrouter(request, model_id, max_tokens) -> _TransportResult` containing the existing transport-specific body of `call_judge` (:812–937) **verbatim except for the return**: the lazy `openai`/`httpx` import, the `OPENROUTER_API_KEY` check, message construction, the `OpenAI(...)` client, the `completions.create(..., temperature=0, ...)` call, `_extract_text_block`, `_extract_cache_accounting`, and `served_model_id`.

```python
def _call_openrouter(request: JudgeRequest, model_id: str, max_tokens: int) -> _TransportResult:
    """OpenRouter transport: OpenAI-compatible SDK pointed at OpenRouter.

    The static policy block is wrapped in ``cache_control: {"type":
    "ephemeral"}`` so calls within the 5-minute TTL re-use cached tokens.
    """
    try:
        import httpx
        from openai import APIError, OpenAI
    except ImportError as exc:
        raise JudgeConfigurationError(
            "The openai SDK and httpx are required for the OpenRouter transport. "
            "Install with:\n\n    uv pip install -e 'elspeth-lints/[judge]'\n\n"
            "(from the repo root), or select --judge-transport agent."
        ) from exc

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise JudgeConfigurationError(
            "OPENROUTER_API_KEY is not set. The OpenRouter transport calls "
            "OpenRouter (the project-wide LLM gateway) to gate allowlist writes. "
            "Set the key (`export OPENROUTER_API_KEY=sk-or-...`) and re-run, or "
            "select --judge-transport agent."
        )

    user_blocks = _build_user_message_blocks(request)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": _STATIC_POLICY_BLOCK,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
        {"role": "user", "content": user_blocks},
    ]
    # trust_env=False pins the audit-boundary transport: ambient HTTP_PROXY /
    # SSL_CERT_FILE / REQUESTS_CA_BUNDLE cannot silently redirect the judge call.
    client = OpenAI(
        base_url=_OPENROUTER_BASE_URL,
        api_key=api_key,
        http_client=httpx.Client(trust_env=False),
    )
    # temperature=0 is load-bearing: it pins verdict reproducibility and closes
    # C2-4 (phantom WAS_ACCEPTED_NOW_BLOCKED reaudit divergences). Do not remove.
    try:
        completion = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=0,
            messages=cast(Any, messages),
        )
    except (APIError, httpx.HTTPError) as exc:
        raise JudgeTransportError(f"{type(exc).__name__}: {exc}") from exc

    raw_text = _extract_text_block(completion)
    prompt_tokens_total, prompt_tokens_cached = _extract_cache_accounting(completion)
    # Record the *served* model id (OpenRouter may re-route to a fallback);
    # fall back to the requested id only when the transport omits it. Closes C1-1.
    served_model_id = completion.model if completion.model else model_id
    return _TransportResult(
        raw_text=raw_text,
        served_model_id=served_model_id,
        prompt_tokens_total=prompt_tokens_total,
        prompt_tokens_cached=prompt_tokens_cached,
    )
```

> Keep `_extract_text_block` and `_extract_cache_accounting` exactly as they are — they are OpenRouter-shape helpers used only by `_call_openrouter`.

- [ ] **Step 6: Rewrite `call_judge` as the shared dispatcher**

Replace `call_judge` (:786–949) with:

```python
def call_judge(
    request: JudgeRequest,
    *,
    model_id: str | None = None,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    transport: str = TRANSPORT_OPENROUTER,
    transport_impl: Callable[[JudgeRequest, str, int], _TransportResult] | None = None,
) -> JudgeResponse:
    """Send a judge request through the selected transport and return the verdict.

    ``transport`` selects the provider path (``TRANSPORT_OPENROUTER`` /
    ``TRANSPORT_AGENT``). When ``model_id`` is omitted, the default is resolved
    **by transport** — the OpenRouter slug and the Agent-SDK model id are
    different namespaces (see ``DEFAULT_AGENT_JUDGE_MODEL``). ``transport_impl``
    is a test seam: inject a fake to exercise the shared validation path without
    a real provider call. Both transports funnel their extracted assistant text
    through the identical ``_parse_judge_payload`` → validators path, so a
    verdict is validated the same way regardless of origin.

    Raises:
        JudgeConfigurationError: transport SDK not installed, or auth missing.
        JudgeTransportError: the provider call failed after configuration succeeded.
        JudgeContractError: the model returned a malformed response. We crash
            rather than coerce — a malformed judge response is never an
            acceptable audit primitive.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if transport not in _VALID_TRANSPORTS:
        raise ValueError(f"unknown judge transport {transport!r}; expected one of {sorted(_VALID_TRANSPORTS)}")

    if model_id is None:
        # Resolve the default by transport: the OpenRouter routing slug
        # ("anthropic/...") is invalid for the Agent SDK and vice versa.
        model_id = DEFAULT_AGENT_JUDGE_MODEL if transport == TRANSPORT_AGENT else DEFAULT_JUDGE_MODEL

    impl = transport_impl if transport_impl is not None else _TRANSPORTS[transport]
    result = impl(request, model_id, max_tokens)

    parsed = _parse_judge_payload(result.raw_text)
    verdict = _verdict_from_string(parsed["verdict"])
    rationale = _required_str_field(parsed, "rationale")
    confidence = _required_confidence_field(parsed, "confidence")
    should_use_decorator = _optional_str_field(parsed, "should_use_decorator")

    if should_use_decorator is not None and verdict is not JudgeVerdict.BLOCKED:
        raise JudgeContractError(
            f"judge emitted should_use_decorator={should_use_decorator!r} with "
            f"verdict={verdict.value}; should_use_decorator is only valid with BLOCKED."
        )

    return JudgeResponse(
        verdict=verdict,
        model_id=result.served_model_id,
        judge_rationale=rationale,
        recorded_at=datetime.now(UTC),
        should_use_decorator=should_use_decorator,
        confidence=confidence,
        prompt_tokens_total=result.prompt_tokens_total,
        prompt_tokens_cached=result.prompt_tokens_cached,
        policy_hash=JUDGE_POLICY_HASH,
        judge_transport=transport,
    )
```

Add `Callable` to the `typing` import (`from typing import Any, Callable, cast`).

Add the transport registry **after** both `_call_openrouter` and (Task 2) `_call_agent_sdk` are defined. For this task, define it referencing `_call_openrouter` and a placeholder that raises until Task 2 lands:

```python
def _call_agent_sdk(request: JudgeRequest, model_id: str, max_tokens: int) -> _TransportResult:
    raise JudgeConfigurationError(
        "The Claude Agent SDK transport is not yet available. Use --judge-transport openrouter."
    )


_TRANSPORTS: dict[str, Callable[[JudgeRequest, str, int], _TransportResult]] = {
    TRANSPORT_OPENROUTER: _call_openrouter,
    TRANSPORT_AGENT: _call_agent_sdk,
}
```

> The placeholder `_call_agent_sdk` keeps the module importable and the registry total while keeping Task 1 a pure OpenRouter-only refactor. Task 2 replaces its body. Selecting `transport=TRANSPORT_AGENT` in Task 1 still reaches the placeholder — which is why the Task-1 tests inject `transport_impl` and never hit it.

- [ ] **Step 7: Run the judge tests + the existing judge suite**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_judge.py -q`
Expected: PASS. The pre-existing OpenRouter tests still pass — the refactor preserved the body. If any pre-existing test constructs `JudgeResponse(...)` directly, it now needs `judge_transport=...`; update those call sites to pass `judge_transport=TRANSPORT_OPENROUTER` (locked-in-expectations: the new required field surfaces in fixtures).

- [ ] **Step 8: Add the `[judge-agent]` extra**

In `elspeth-lints/pyproject.toml`, after the `judge = ["openai>=1.0,<2"]` line (:20), add:

```toml
# The 'judge-agent' extra pulls in the Claude Agent SDK used by the
# `--judge-transport agent` path. Like 'judge' it is optional: the lint
# tool runs in CI lanes that carry neither SDK. The agent transport
# authenticates via an installed + logged-in Claude Code CLI (subscription
# / Agent-SDK credit pool) or ANTHROPIC_API_KEY or Bedrock/Vertex/Azure.
# Install with:
#     uv pip install -e ".[judge-agent]"  (from the elspeth-lints/ directory)
judge-agent = ["claude-agent-sdk>=0.1"]
```

> Confirm the published distribution name and a sane lower bound with `uv pip index versions claude-agent-sdk` (or check PyPI). Pin the lower bound to the version whose `query` / `ClaudeAgentOptions` / `ResultMessage` shape Task 2 targets.

- [ ] **Step 9: Type-check, lint, commit**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m mypy src/elspeth_lints/core/judge.py && ../.venv/bin/python -m ruff check src/elspeth_lints/core/judge.py`
Expected: no errors.

```bash
git add elspeth-lints/pyproject.toml elspeth-lints/src/elspeth_lints/core/judge.py elspeth-lints/tests/core/test_judge.py
git commit -m "refactor(judge): transport seam + judge_transport field (openrouter unchanged)"
```

---

## Task 2: The Claude Agent SDK transport (`_call_agent_sdk`)

Replace the placeholder with the real agent transport: a no-tools, `claude_code`-preset, single-shot `query` whose assistant text feeds the *same* shared parser. Unit tests inject a fake `claude_agent_sdk` module — **no real SDK call in CI**.

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/judge.py`
- Test: `elspeth-lints/tests/core/test_judge_agent_transport.py`

> **SDK shape is doc-derived, not introspected** — `claude-agent-sdk` is not installed in this workspace. The symbol names below (`query`, `ClaudeAgentOptions`, `ResultMessage`, `AssistantMessage`, `TextBlock`, the `usage` keys `input_tokens` / `cache_read_input_tokens`, `model_usage`) are from the published Python SDK docs (`https://code.claude.com/docs/en/agent-sdk/python`). **Before implementing, install the extra (`uv pip install -e 'elspeth-lints/[judge-agent]'`) and confirm each symbol** with `python -c "import claude_agent_sdk as s; print([n for n in dir(s) if not n.startswith('_')])"` and by inspecting `ResultMessage` / `AssistantMessage` fields. Adjust the extraction to the real shape; keep the *seam contract* (`_TransportResult`) identical. CI correctness rests on the fake-module tests, not on the real SDK.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_judge_agent_transport.py
import sys
import types

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict
from elspeth_lints.core.judge import (
    JudgeConfigurationError,
    JudgeRequest,
    TRANSPORT_AGENT,
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
    '{"verdict": "ACCEPTED", "rationale": "external boundary; absence recorded as None", '
    '"confidence": 0.7, "should_use_decorator": null}'
)


def _install_fake_sdk(monkeypatch, *, assistant_text: str, served="claude-opus-4-7", raise_on_query=None):
    """Install a minimal fake claude_agent_sdk into sys.modules.

    The fake mirrors only the surface _call_agent_sdk consumes: query() is an
    async generator yielding an AssistantMessage(content=[TextBlock(text=...)])
    then a ResultMessage(usage=..., model_usage=...). Adjust attribute names
    here to match the real SDK once confirmed (see the task header note).
    """
    mod = types.ModuleType("claude_agent_sdk")

    class ClaudeAgentOptions:  # noqa: N801 - mirror SDK name
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, content):
            self.content = content

    class ResultMessage:
        def __init__(self, usage, model_usage):
            self.usage = usage
            self.model_usage = model_usage

    async def query(*, prompt, options):
        if raise_on_query is not None:
            raise raise_on_query
        yield AssistantMessage(content=[TextBlock(text=assistant_text)])
        yield ResultMessage(
            usage={"input_tokens": 200, "cache_read_input_tokens": 50},
            model_usage={served: {"input_tokens": 200}},
        )

    mod.ClaudeAgentOptions = ClaudeAgentOptions
    mod.TextBlock = TextBlock
    mod.AssistantMessage = AssistantMessage
    mod.ResultMessage = ResultMessage
    mod.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
    return mod


def test_agent_transport_produces_validated_response(monkeypatch) -> None:
    _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON)
    resp = call_judge(_request(), transport=TRANSPORT_AGENT)
    assert resp.judge_transport == TRANSPORT_AGENT
    assert resp.verdict is JudgeVerdict.ACCEPTED
    assert resp.model_id == "claude-opus-4-7"      # served model from model_usage
    assert resp.prompt_tokens_total == 200
    assert resp.prompt_tokens_cached == 50


def test_agent_transport_missing_sdk_raises_configuration_error(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "claude_agent_sdk", raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _import_raising("claude_agent_sdk"),
    )
    with pytest.raises(JudgeConfigurationError, match="judge-agent"):
        call_judge(_request(), transport=TRANSPORT_AGENT)


def test_agent_transport_auth_failure_names_auth_path(monkeypatch) -> None:
    # Confirm the real SDK's auth-failure exception class and swap it in here.
    class _FakeAuthError(Exception):
        pass

    _install_fake_sdk(monkeypatch, assistant_text=_GOOD_JSON, raise_on_query=_FakeAuthError("not logged in"))
    with pytest.raises(JudgeConfigurationError, match="ANTHROPIC_API_KEY|Claude Code"):
        call_judge(_request(), transport=TRANSPORT_AGENT)


def _import_raising(blocked_name):
    real_import = __import__

    def _fake(name, *args, **kwargs):
        if name == blocked_name:
            raise ImportError(f"No module named '{blocked_name}'")
        return real_import(name, *args, **kwargs)

    return _fake
```

> The `test_agent_transport_auth_failure_names_auth_path` test uses a stand-in `_FakeAuthError`. Once the SDK is installed, replace it with the SDK's real auth/CLI-not-found exception type and map that type in `_call_agent_sdk` (Step 2). If the SDK distinguishes "CLI not installed" from "not authenticated," map both to `JudgeConfigurationError` with path-specific guidance.

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_judge_agent_transport.py -q`
Expected: FAIL — the placeholder `_call_agent_sdk` raises "not yet available".

- [ ] **Step 3: Implement `_call_agent_sdk`**

Replace the placeholder `_call_agent_sdk` from Task 1 with:

```python
def _call_agent_sdk(request: JudgeRequest, model_id: str, max_tokens: int) -> _TransportResult:
    """Claude Agent SDK transport: a no-tools, single-shot query.

    The system prompt is the ``claude_code`` preset with our static policy
    block appended (design §4.2): the preset is the one intentional
    Anthropic-side influence; its era is bounded by the signed recorded_at
    timestamp. No tools are allowed and no project settings are loaded, so the
    judge sees only the excerpt in the prompt — identical to the OpenRouter
    path. The assistant emits the verdict JSON per the policy block's output
    schema; that text feeds the shared ``_parse_judge_payload``.

    Determinism caveat (design §7): the SDK does not expose ``temperature``, so
    agent verdicts are less reproducible than the temperature=0 OpenRouter
    path. The signed ``judge_transport`` lets reaudit attribute a divergence on
    an agent-written entry to transport noise rather than source drift.

    ``model_id`` here is an Agent-SDK model id (unprefixed), NOT an OpenRouter
    slug — call_judge resolves the per-transport default before we are reached
    (see ``DEFAULT_AGENT_JUDGE_MODEL``).

    ``max_tokens`` is accepted for transport-contract uniformity but is NOT
    wired into ``ClaudeAgentOptions`` (the SDK does not surface an equivalent at
    the time of writing — confirm post-install and wire it if it does). One
    consequence: there is no agent equivalent of the OpenRouter path's
    ``finish_reason == "length"`` guard, so a truncated agent response degrades
    to a generic ``_parse_judge_payload`` JSON error rather than the actionable
    "increase max_tokens" message. Acceptable degradation; documented so it is
    not mistaken for missed wiring.

    ``asyncio.run`` below assumes no running event loop. That holds for the
    synchronous justify / reaudit CLI callers today; if a future async caller
    invokes ``call_judge``, this bridge raises ``RuntimeError`` and must be
    revisited.
    """
    try:
        import claude_agent_sdk as sdk
    except ImportError as exc:
        raise JudgeConfigurationError(
            "The claude-agent-sdk is required for --judge-transport agent. "
            "Install with:\n\n    uv pip install -e 'elspeth-lints/[judge-agent]'\n\n"
            "(from the repo root), or use --judge-transport openrouter."
        ) from exc

    options = sdk.ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code", "append": _STATIC_POLICY_BLOCK},
        allowed_tools=[],
        disallowed_tools=["Bash", "Read", "Write", "Edit", "Grep", "Glob", "WebSearch", "WebFetch"],
        setting_sources=[],
        permission_mode="bypassPermissions",
        model=model_id,
    )
    prompt_text = "\n\n".join(block["text"] for block in _build_user_message_blocks(request))

    try:
        return asyncio.run(_drain_agent_query(sdk, prompt_text, options, model_id))
    except JudgeConfigurationError:
        raise
    except JudgeContractError:
        raise
    except Exception as exc:  # SDK transport/auth surface — map by type below.
        # Auth / CLI-not-found is operator-actionable configuration; everything
        # else after configuration is a transport failure. Confirm the SDK's
        # actual exception classes and replace this isinstance() with the real
        # types once installed (see the task header note).
        if _is_agent_auth_error(exc):
            raise JudgeConfigurationError(
                "The Claude Agent SDK could not authenticate. The agent transport "
                "uses an installed + logged-in Claude Code CLI (subscription / "
                "Agent-SDK credit pool) OR ANTHROPIC_API_KEY OR Bedrock/Vertex/Azure. "
                "Log in with the Claude Code CLI, or set ANTHROPIC_API_KEY, then re-run. "
                "(Note: ANTHROPIC_API_KEY is per-token Anthropic billing and may not "
                "be cheaper than OpenRouter.)"
            ) from exc
        raise JudgeTransportError(f"{type(exc).__name__}: {exc}") from exc


async def _drain_agent_query(sdk: Any, prompt_text: str, options: Any, requested_model: str) -> _TransportResult:
    """Drain the async query stream into a _TransportResult.

    Accumulate assistant text blocks; capture the terminal ResultMessage for
    usage + served-model accounting. The shapes here are doc-derived — confirm
    against the installed SDK (task header note).
    """
    text_parts: list[str] = []
    result_message: Any = None
    async for message in sdk.query(prompt=prompt_text, options=options):
        if isinstance(message, sdk.AssistantMessage):
            for block in message.content:
                if isinstance(block, sdk.TextBlock):
                    text_parts.append(block.text)
        elif isinstance(message, sdk.ResultMessage):
            result_message = message

    if result_message is None:
        raise JudgeContractError("agent transport produced no ResultMessage; cannot account usage.")
    raw_text = "".join(text_parts)
    if not raw_text.strip():
        raise JudgeContractError("agent transport produced no assistant text; cannot extract a verdict.")

    served_model_id = _agent_served_model(result_message, requested_model)
    prompt_tokens_total, prompt_tokens_cached = _agent_cache_accounting(result_message)
    return _TransportResult(
        raw_text=raw_text,
        served_model_id=served_model_id,
        prompt_tokens_total=prompt_tokens_total,
        prompt_tokens_cached=prompt_tokens_cached,
    )


def _agent_served_model(result_message: Any, requested_model: str) -> str:
    """Served model id from ResultMessage.model_usage (record what was served).

    ``model_usage`` is keyed by served model name. We take the single key when
    there is exactly one; fall back to the requested id only when the field is
    empty (mirrors the OpenRouter served-vs-requested rule, C1-1). More than one
    key is an unexpected shape for a single-shot judge call — crash rather than
    fabricate a served id.
    """
    model_usage = result_message.model_usage
    keys = list(model_usage)
    if not keys:
        return requested_model
    if len(keys) != 1:
        raise JudgeContractError(
            f"agent ResultMessage.model_usage has {len(keys)} models {keys!r}; "
            "a single-shot judge call must resolve to exactly one served model."
        )
    return keys[0]


def _agent_cache_accounting(result_message: Any) -> tuple[int, int | None]:
    """Map the SDK usage dict onto (prompt_tokens_total, prompt_tokens_cached).

    ``input_tokens`` -> total; ``cache_read_input_tokens`` -> cached subset.
    Preserve None (provider didn't report a cached count) vs 0 (caching on,
    no hit). The total is required on a completed call.
    """
    usage = result_message.usage
    total = usage["input_tokens"]
    if not isinstance(total, int):
        raise JudgeContractError(f"agent usage.input_tokens must be int; got {type(total).__name__}")
    cached = usage.get("cache_read_input_tokens")
    if cached is None:
        return total, None
    if not isinstance(cached, int):
        raise JudgeContractError(f"agent usage.cache_read_input_tokens must be int or None; got {type(cached).__name__}")
    return total, cached


def _is_agent_auth_error(exc: Exception) -> bool:
    """Whether an SDK exception is an operator-actionable auth/config failure.

    Placeholder discriminator until the SDK's real auth/CLI-not-found exception
    classes are confirmed (task header note). Until then, match on the SDK's
    documented auth exception type name to avoid a brittle message-substring
    check. Replace with ``isinstance(exc, sdk.<AuthError>)`` once known.
    """
    return type(exc).__name__ in {"CLINotFoundError", "ProcessError", "AuthenticationError"}
```

> `usage` is shown as a `dict` (`.get`, subscript) per the SDK docs. If the installed SDK exposes `usage` as an object with attributes, switch `_agent_cache_accounting` to attribute access (`usage.input_tokens`) — but note that `.get`/subscript on a genuinely external SDK response dict is a Tier-3 boundary read, not a forbidden defensive `.get` on our own typed data, so either shape is policy-compliant here. Keep the `None`-vs-`0` distinction whichever way.

Add `import asyncio` to the top-of-module imports (after `import hashlib`).

- [ ] **Step 4: Run the agent-transport tests + the full judge suite**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_judge_agent_transport.py tests/core/test_judge.py -q`
Expected: PASS. (`test_agent_transport_auth_failure_names_auth_path` passes because `_FakeAuthError` is not in the placeholder name set → wait: it must be. Either name the fake `AuthenticationError` so `_is_agent_auth_error` matches, or — preferred — once the SDK is installed, set the real class. For the fake-only run, name the test's exception class `AuthenticationError`.)

> Reconcile the fake auth-error class name in the test with `_is_agent_auth_error`'s set so the test is honest about what it exercises. The cleanest version: name the fake `AuthenticationError` and leave a comment that the real SDK class replaces it post-install.

- [ ] **Step 5: Type-check, lint, commit**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m mypy src/elspeth_lints/core/judge.py && ../.venv/bin/python -m ruff check src/elspeth_lints/core/judge.py`
Expected: no errors.

```bash
git add elspeth-lints/src/elspeth_lints/core/judge.py elspeth-lints/tests/core/test_judge_agent_transport.py
git commit -m "feat(judge): Claude Agent SDK transport (no-tools, claude_code preset)"
```

---

## Task 3: `judge_transport` on `AllowlistEntry` + loader + sidecar round-trip (additive)

Add the field and its parse/round-trip plumbing **without** yet signing or requiring it. This is purely additive: an optional field that defaults to `None`, so every existing entry and test stays green.

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/allowlist.py` (`AllowlistEntry` :135–154; `_parse_allow_hits` `AllowlistEntry(...)` construction)
- Modify: `elspeth-lints/src/elspeth_lints/core/reaudit_sidecar.py` (`_entry_to_dict` :898–917; `_entry_from_dict` :920–960)
- Test: `elspeth-lints/tests/core/test_allowlist_schema.py`, `elspeth-lints/tests/core/test_reaudit_sidecar.py`

- [ ] **Step 1: Write the failing tests**

```python
# test_allowlist_schema.py — extend
def test_entry_parses_judge_transport() -> None:
    from elspeth_lints.core.allowlist import _parse_allow_hits

    # Base on an existing fully-populated v2 judge-entry fixture in this file and
    # add "judge_transport": "claude_agent_sdk". Parse schema-only (source_root=None).
    data = {"allow_hits": [{**_V2_JUDGE_ENTRY_FIXTURE, "judge_transport": "claude_agent_sdk"}]}
    entries = _parse_allow_hits(data, source_file="x.yaml", source_root=None)
    assert entries[0].judge_transport == "claude_agent_sdk"


# test_reaudit_sidecar.py — extend
def test_sidecar_round_trips_judge_transport() -> None:
    from elspeth_lints.core.reaudit_sidecar import _entry_to_dict, _entry_from_dict
    from pathlib import Path

    entry = _v2_entry(judge_transport="claude_agent_sdk")  # local helper as elsewhere in this file
    restored = _entry_from_dict(_entry_to_dict(entry), sidecar_path=Path("s"), line_no=1)
    assert restored.judge_transport == "claude_agent_sdk"
```

> Reuse an existing fully-populated v2 judge-entry fixture (after the scope-fingerprint plan, fixtures carry `judge_signature_version: 2` + `scope_fingerprint`). The assertion is only about the new field round-tripping; the atomic validator must already be satisfied by the base fixture.

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_schema.py -k judge_transport tests/core/test_reaudit_sidecar.py -k judge_transport -q`
Expected: FAIL — `AllowlistEntry.__init__() got an unexpected keyword argument 'judge_transport'`.

- [ ] **Step 3: Add the dataclass field**

In `allowlist.py`, in `AllowlistEntry`, after `judge_signature_version: int | None = None` (added by the scope-fingerprint plan's Task 3):

```python
    judge_signature_version: int | None = None
    judge_transport: str | None = None
```

- [ ] **Step 4: Parse it in `_parse_allow_hits`**

After the `judge_signature_version=_optional_signature_version(...)` line (scope-fingerprint Task 3) in the `AllowlistEntry(...)` construction, add:

```python
            judge_transport=_optional_string(entry, "judge_transport", context=ctx),
```

> Confirm the optional-string helper name with `grep -n "_optional_string\|_optional_str\b" src/elspeth_lints/core/allowlist.py`. Use whatever the sibling judge-cluster fields use (`scope_fingerprint` uses `_optional_string` per scope-fingerprint Task 3).

- [ ] **Step 5: Round-trip it through the sidecar**

In `reaudit_sidecar.py` `_entry_to_dict` (:898), after the scope-fingerprint round-trip lines, add:

```python
        "judge_transport": entry.judge_transport,
```

In `_entry_from_dict` (:920), in the `AllowlistEntry(...)` construction after the scope-fingerprint lines, add:

```python
        judge_transport=_optional_str(payload, "judge_transport", sidecar_path, line_no),
```

(`_optional_str` exists at reaudit_sidecar.py:1002.)

- [ ] **Step 6: Run the tests + both suites**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_schema.py tests/core/test_reaudit_sidecar.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/allowlist.py elspeth-lints/src/elspeth_lints/core/reaudit_sidecar.py elspeth-lints/tests/core/
git commit -m "feat(allowlist): add optional judge_transport field + sidecar round-trip"
```

---

## Task 4 (ATOMIC): sign `judge_transport` into the v2 payload — justify, migrate, and the validator together

**This is one commit on purpose.** Adding `judge_transport` to the *signed* v2 payload and making the validator *require* it on v2 entries means **every** site that mints a v2 signature must pass it in the same commit, or a reload of a transport-less v2 entry fails the atomic validator and goes red. The v2-emitting sites are: `compute_judge_metadata_signature` (the sign function), `justify` (`_build_yaml_entry_text`), and `migrate-judge-scope` (the scope-fingerprint plan's `_run_migrate_judge_scope`, which backfills the corpus). All three change here. (Advisor lock: "every v2-emitting site in the commit that makes it required.")

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/allowlist.py` (`compute_judge_metadata_signature`; `_verify_judge_metadata_signature_at_load`; `_validate_judge_metadata_atomic` invariants 4 + 8)
- Modify: `elspeth-lints/src/elspeth_lints/core/cli.py` (`_build_yaml_entry_text`; justify call site; `_run_migrate_judge_scope`)
- Test: `elspeth-lints/tests/core/test_allowlist_signing.py`, `test_cli_justify.py`, `test_cli_migrate_judge_scope.py`

- [ ] **Step 1: Write the failing tests**

```python
# test_allowlist_signing.py — extend the _sig helper from the scope-fingerprint plan's Task 4
def test_v2_signature_includes_judge_transport() -> None:
    a = _sig(signature_version=2, scope_fingerprint="a" * 64, judge_transport="openrouter")
    b = _sig(signature_version=2, scope_fingerprint="a" * 64, judge_transport="claude_agent_sdk")
    # Same logical entry, different transport => different signature (transport is bound).
    import hmac

    assert not hmac.compare_digest(a, b)


def test_v2_signature_default_transport_is_openrouter() -> None:
    # Omitting judge_transport defaults to openrouter (the function-parameter default),
    # so scope-fingerprint's transport-agnostic signing tests stay valid.
    omitted = _sig(signature_version=2, scope_fingerprint="a" * 64)
    explicit = _sig(signature_version=2, scope_fingerprint="a" * 64, judge_transport="openrouter")
    import hmac

    assert hmac.compare_digest(omitted, explicit)


# test_allowlist_atomic.py (or wherever invariant 8 is tested) — extend
def test_v2_judge_entry_requires_judge_transport() -> None:
    import pytest

    with pytest.raises(ValueError, match="judge_transport"):
        _validate_atomic(_v2_entry(scope_fingerprint="a" * 64, judge_transport=None))


def test_v1_judge_entry_forbids_judge_transport() -> None:
    import pytest

    with pytest.raises(ValueError, match="judge_transport"):
        _validate_atomic(_v1_entry(file_fingerprint="b" * 64, judge_transport="openrouter"))


def test_pre_judge_entry_forbids_judge_transport() -> None:
    import pytest

    with pytest.raises(ValueError, match="judge_transport"):
        _validate_atomic(_pre_judge_entry(judge_transport="openrouter"))
```

```python
# test_cli_justify.py — extend the existing v2 justify test
def test_justify_writes_judge_transport(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", "x" * 32)
    # Drive _run_justify with the default (openrouter) transport, stubbing call_judge
    # to return an ACCEPTED JudgeResponse with judge_transport="openrouter".
    # Assert the emitted YAML contains "judge_transport: openrouter", the signature
    # is hmac-sha256:v2:, and re-loading with source_root verifies clean.
    ...
```

```python
# test_cli_migrate_judge_scope.py — extend the scope-fingerprint plan's migrate tests
def test_migrate_backfills_openrouter_transport(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", "x" * 32)
    # Migrate a valid v1 entry to v2; assert the rewritten YAML now carries
    # "judge_transport: openrouter" and the v2 signature verifies. Every existing
    # OpenRouter-produced verdict is truthfully backfilled as openrouter.
    ...
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_signing.py tests/core/test_cli_justify.py tests/core/test_cli_migrate_judge_scope.py -k "judge_transport or transport" -q`
Expected: FAIL — `compute_judge_metadata_signature() got an unexpected keyword argument 'judge_transport'`.

- [ ] **Step 3: Add `judge_transport` to the sign function (v2 payload only)**

In `compute_judge_metadata_signature`, add the parameter (after `scope_fingerprint: str | None = None` from scope-fingerprint Task 4):

```python
    scope_fingerprint: str | None = None,
    judge_transport: str = "openrouter",
```

> The **function-parameter default** is `"openrouter"`. This is what keeps scope-fingerprint's signing-test helper (which never passes `judge_transport`) green and unchanged — exactly as `signature_version=1` defaulting does. The *validator* (Step 5) enforces presence on persisted v2 entries; the function default only covers direct unit callers.

In the `signature_version == 2` branch, add `judge_transport` to the payload. The cleanest place is inside the v2 `binding` dict so it only enters the v2 payload, never v1:

```python
    if signature_version == 2:
        if scope_fingerprint is None:
            raise ValueError("compute_judge_metadata_signature: scope_fingerprint is required for signature_version 2")
        binding: dict[str, Any] = {"scope_fingerprint": scope_fingerprint, "judge_transport": judge_transport}
        prefix = _JUDGE_METADATA_SIGNATURE_PREFIX_V2
```

> `judge_transport` is intentionally **absent** from the v1 branch's `binding`. v1 is legacy, deleted in scope-fingerprint Task 13; its payload shape must not change.

- [ ] **Step 4: Pass the entry's transport at load-time verify**

In `_verify_judge_metadata_signature_at_load`, the `compute_judge_metadata_signature(...)` recompute (scope-fingerprint Task 4 Step 4) must pass the stored transport so the recomputed signature matches:

```python
        signature_version=version,
        file_fingerprint=entry.file_fingerprint,
        scope_fingerprint=entry.scope_fingerprint,
        judge_transport=entry.judge_transport if entry.judge_transport is not None else "openrouter",
```

> `entry.judge_transport` is guaranteed present for valid v2 entries by the validator (Step 5), so the `is not None` fallback only ever applies to v1 entries — where the v2 branch never reads it anyway. Passing it unconditionally is harmless and keeps the call uniform. Do **not** rely on normalization for correctness of v2 entries; the validator guarantees presence.

- [ ] **Step 5: Make the atomic validator require/forbid `judge_transport` by version**

In `_validate_judge_metadata_atomic`, in the **v2 branch** of the version-aware binding block (scope-fingerprint Task 4 Step 5), alongside the `scope_fingerprint` presence check, add:

```python
    if version == 2:
        if entry.scope_fingerprint is None:
            missing_binding.append("scope_fingerprint")
        if entry.judge_transport is None:
            missing_binding.append("judge_transport")
        if entry.file_fingerprint is not None:
            raise ValueError(
                f"{context}: judge_signature_version is 2 but file_fingerprint is present; "
                "v2 entries bind via scope_fingerprint."
            )
    else:
        if entry.file_fingerprint is None:
            missing_binding.append("file_fingerprint")
        if entry.scope_fingerprint is not None:
            raise ValueError(
                f"{context}: judge_signature_version is absent/1 but scope_fingerprint is present; "
                "v1 entries bind via file_fingerprint."
            )
        if entry.judge_transport is not None:
            raise ValueError(
                f"{context}: judge_signature_version is absent/1 but judge_transport is present; "
                "judge_transport is a v2-only signed field. A judge_transport on a v1 entry is corruption."
            )
```

In **invariant 4** (the pre-judge stray-field check; scope-fingerprint Task 4 Step 5 adds `scope_fingerprint` / `judge_signature_version` to it), also add:

```python
        if entry.judge_transport is not None:
            stray.append("judge_transport")
```

- [ ] **Step 6: `justify` emits and signs `judge_transport`**

In `cli.py`, the justify call site already has a `response: JudgeResponse` from `call_judge`. Thread `response.judge_transport` into the entry writer. In the `build_signed_yaml_entry()` closure (scope-fingerprint Task 7), add the argument:

```python
            scope_fingerprint=scope_fingerprint,
            judge_transport=response.judge_transport,
            ast_path=_finding_ast_path(finding),
```

In `_build_yaml_entry_text`, add the `judge_transport: str` parameter, pass it to `compute_judge_metadata_signature(signature_version=2, ..., judge_transport=judge_transport)`, and emit the YAML line alongside the other v2 binding lines (scope-fingerprint Task 7 Step 4):

```python
    lines.append(f"  judge_signature_version: 2")
    lines.append(f"  scope_fingerprint: {_yaml_inline_scalar(scope_fingerprint)}")
    lines.append(f"  judge_transport: {_yaml_inline_scalar(judge_transport)}")
    lines.append(f"  ast_path: {_yaml_inline_scalar(ast_path)}")
    lines.append(f"  judge_metadata_signature: {_yaml_inline_scalar(judge_metadata_signature)}")
```

- [ ] **Step 7: `migrate-judge-scope` backfills `judge_transport="openrouter"`**

In `_run_migrate_judge_scope` (scope-fingerprint Task 10), the re-sign of a valid v1 entry to v2 must pass `judge_transport="openrouter"` to `compute_judge_metadata_signature` **and** write the `judge_transport: openrouter` YAML line. Every existing entry was OpenRouter-produced, so this backfill is truthful, not fabrication.

Locate the migrate re-sign call and the YAML surgery (`grep -n "compute_judge_metadata_signature\|scope_fingerprint:\|judge_signature_version: 2" src/elspeth_lints/core/cli.py` within `_run_migrate_judge_scope`). Add `judge_transport="openrouter"` to the sign call, and add the `judge_transport: openrouter` line to the v2 lines the surgery inserts (immediately after the `scope_fingerprint:` line, matching `_build_yaml_entry_text`'s field order so migrated and freshly-justified entries are byte-congruent).

> If `_run_migrate_judge_scope` reuses `_build_yaml_entry_text` to render the v2 entry, this is a single change: pass `judge_transport="openrouter"`. If it does targeted line surgery, insert the one extra line. Either way, add a migrate test (Step 1) asserting the backfilled line and a clean v2 reload.

> The scope-fingerprint plan's **existing** migrate tests (`test_migrate_rewrites_valid_v1_entry_as_v2`, and its byte-identical-non-binding-fields assertion) now see an extra `judge_transport: openrouter` line in the migrated output. Update those assertions in this same commit (locked-in-expectations — the same move as the golden-signature update in Step 8).

- [ ] **Step 8: Run the signing + justify + migrate + atomic suites**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_allowlist_signing.py tests/core/test_cli_justify.py tests/core/test_cli_migrate_judge_scope.py tests/core/test_allowlist_schema.py -q`
Expected: PASS. If any scope-fingerprint v2 test asserts an exact signature digest (golden), it will shift because the v2 payload now contains `judge_transport`; update that golden (locked-in-expectations) — prefix-only assertions are unaffected.

- [ ] **Step 9: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/allowlist.py elspeth-lints/src/elspeth_lints/core/cli.py elspeth-lints/tests/core/
git commit -m "feat(allowlist): bind judge_transport in v2 signed payload (justify + migrate + validator)"
```

---

## Task 5: `--judge-transport` on `justify` and `reaudit`

Wire the operator-facing flag. The CLI spelling `agent` maps to the stored value `claude_agent_sdk`; the default is `openrouter` (no behaviour change unless opted in).

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/cli.py` (justify parser :274; reaudit parser :428; justify call site :1342; `_run_reaudit`)
- Modify: `elspeth-lints/src/elspeth_lints/core/reaudit.py` (thread transport to the per-entry `call_judge` at :1007)
- Test: `elspeth-lints/tests/core/test_cli_justify.py`, `test_cli_reaudit*.py`

- [ ] **Step 1: Write the failing tests**

```python
# test_cli_justify.py
def test_justify_agent_transport_flag_selects_agent(monkeypatch, tmp_path) -> None:
    # Patch call_judge to capture its `transport` kwarg; run _run_justify with
    # --judge-transport agent; assert call_judge was called with
    # transport="claude_agent_sdk" and the written entry carries
    # judge_transport: claude_agent_sdk.
    ...


# test_cli_reaudit.py
def test_reaudit_agent_transport_flag_threads_to_call_judge(monkeypatch, tmp_path) -> None:
    # Patch call_judge to capture `transport`; run _run_reaudit with
    # --judge-transport agent over one entry; assert call_judge saw
    # transport="claude_agent_sdk".
    ...
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_cli_justify.py tests/core/test_cli_reaudit.py -k "agent_transport" -q`
Expected: FAIL — `unrecognized arguments: --judge-transport`.

- [ ] **Step 3: Add a shared flag helper + the CLI→stored mapping**

Near the top of `cli.py`'s argument-building (or beside the other small helpers), add:

```python
from elspeth_lints.core.judge import TRANSPORT_AGENT, TRANSPORT_OPENROUTER

_CLI_TRANSPORT_CHOICES: dict[str, str] = {"openrouter": TRANSPORT_OPENROUTER, "agent": TRANSPORT_AGENT}


def _add_judge_transport_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-transport",
        choices=tuple(_CLI_TRANSPORT_CHOICES),
        default="openrouter",
        help=(
            "Which transport produces the verdict. 'openrouter' (default) uses "
            "the OpenAI-compatible SDK with temperature=0 (reproducible). 'agent' "
            "uses the Claude Agent SDK (claude_code preset, no tools, cheaper via a "
            "Claude subscription); it cannot pin temperature, so agent verdicts are "
            "less reproducible (see reaudit). Requires the [judge-agent] extra and "
            "Claude Code auth."
        ),
    )
```

Call `_add_judge_transport_arg(justify)` after the justify parser is built (:381) and `_add_judge_transport_arg(reaudit)` after the reaudit parser is built (:534).

- [ ] **Step 4: Thread it into the justify call site**

At the justify `call_judge` invocation (:1342), pass the resolved transport and **drop the hardcoded `model_id`** so `call_judge` resolves the per-transport default (the OpenRouter slug would break the agent path):

```python
        response: JudgeResponse = call_judge(
            request,
            max_tokens=args.max_tokens or DEFAULT_JUDGE_MAX_TOKENS,
            transport=_CLI_TRANSPORT_CHOICES[args.judge_transport],
        )
```

> Removing `model_id=DEFAULT_JUDGE_MODEL` is deliberate: under `--judge-transport agent`, `call_judge` must pick `DEFAULT_AGENT_JUDGE_MODEL`, not the OpenRouter slug. If `DEFAULT_JUDGE_MODEL` is now unused in `cli.py`, drop its import (ruff will flag it).

- [ ] **Step 5: Thread it through reaudit to the judge boundary**

`_run_reaudit` resolves `transport = _CLI_TRANSPORT_CHOICES[args.judge_transport]` and passes it down to the per-entry function that calls `call_judge(request)` at reaudit.py:1007. Add a `transport: str` parameter to that function (and any intermediate callers between `_run_reaudit` and it — trace with `grep -n "def _" src/elspeth_lints/core/reaudit.py` around the call chain), defaulting to `TRANSPORT_OPENROUTER` so existing internal callers/tests are unaffected. At the boundary:

```python
        response = call_judge(request, transport=transport)
```

> This omits `model_id`, so `call_judge` resolves the per-transport default automatically — reaudit under `agent` gets the SDK model id, under `openrouter` the slug. No model-id threading needed in reaudit.

> Keep the existing per-entry try/except exactly as-is: `JudgeConfigurationError` stays sweep-fatal, `JudgeTransportError`/`JudgeContractError` stay per-entry isolated. The agent transport raises the same three error classes, so the isolation contract is unchanged. Record `transport` nowhere new in reaudit — reaudit reports divergence, it does not write entries; the entry's *origin* transport is already in the loaded `entry.judge_transport`.

- [ ] **Step 6: Run the CLI suites**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_cli_justify.py tests/core/test_cli_reaudit*.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/cli.py elspeth-lints/src/elspeth_lints/core/reaudit.py elspeth-lints/tests/core/
git commit -m "feat(cli): --judge-transport {openrouter,agent} on justify and reaudit"
```

---

## Task 6: `judge_transport` participates in the coverage-diff metadata identity

`judge_transport` is **verdict metadata** ("how the verdict was produced"), not source binding. It belongs in `_judge_metadata_payload` (the authenticity-bearing cluster the rotation/coverage diff protects), not in `_judge_binding_identity` (fields a fresh `justify` may legitimately change). A re-justify under a different transport *should* surface as a metadata change.

**Files:**
- Modify: `elspeth-lints/src/elspeth_lints/core/judge_coverage.py` (`_judge_metadata_payload` :403–433)
- Test: `elspeth-lints/tests/core/test_judge_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
def test_judge_metadata_payload_includes_transport() -> None:
    from elspeth_lints.core.judge_coverage import _judge_metadata_payload

    entry = _v2_entry(judge_transport="claude_agent_sdk")
    payload = _judge_metadata_payload(entry)
    assert payload is not None
    assert "claude_agent_sdk" in payload
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_judge_coverage.py -k transport -q`
Expected: FAIL — `"claude_agent_sdk"` not in the payload tuple.

- [ ] **Step 3: Add it to the metadata payload**

In `_judge_metadata_payload` (:420), add `entry.judge_transport` to the returned tuple (after `entry.judge_model`, beside the other judge-cluster fields):

```python
        entry.judge_model,
        entry.judge_transport,
        entry.judge_policy_hash,
```

> Leave `_judge_binding_identity` (:398) untouched — `judge_transport` is not a source-binding field. Leave the pre-judge `None`-guard (:411–418) untouched: it already returns `None` for pre-judge entries, and a pre-judge entry carrying `judge_transport` is rejected upstream by the atomic validator (Task 4 invariant 4), so it never reaches here.

- [ ] **Step 4: Run the coverage suite**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest tests/core/test_judge_coverage.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add elspeth-lints/src/elspeth_lints/core/judge_coverage.py elspeth-lints/tests/core/test_judge_coverage.py
git commit -m "feat(judge-coverage): judge_transport participates in metadata identity"
```

---

## Task 7: Full-suite green gate + docs (auth path, determinism caveat)

**Files:**
- Modify: `CLAUDE.md` (the "CICD Judge Gate" / dev-commands area — document `--judge-transport` and the auth/billing path)
- Modify: any `tier-model-deep-dive` / `engine-patterns-reference` skill text that enumerates the judge transport or the signed payload fields
- Modify: `docs/elspeth-lints/rationale.md` if it documents the judge transport

- [ ] **Step 1: Run the entire elspeth-lints test suite**

Run: `cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m pytest -q`
Expected: PASS (full suite). Fix any fixture that constructs `JudgeResponse` or a v2 `AllowlistEntry` without the new field (locked-in-expectations: the new required v2 field surfaces in older fixtures).

- [ ] **Step 2: Run the tier-model gate against the real tree (keyless shape-only — agent context)**

Run:
```bash
cd /home/john/elspeth && env PYTHONPATH=elspeth-lints/src ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing \
  .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
```
Expected: unchanged from before this work. The 221 live entries are still **v1** (the keyed migration that converts them — and backfills `judge_transport="openrouter"` — is the scope-fingerprint plan's operator Task 12, not run here), so the v2 `judge_transport` requirement applies to **zero** on-disk entries. The shape-only downgrade warning is expected.

- [ ] **Step 3: mypy + ruff across the touched modules**

Run:
```bash
cd /home/john/elspeth/elspeth-lints && ../.venv/bin/python -m mypy src/elspeth_lints && ../.venv/bin/python -m ruff check src/elspeth_lints
```
Expected: no errors.

- [ ] **Step 4: Update docs**

In `CLAUDE.md`, near the judge/dev-commands material, document:
- `--judge-transport {openrouter,agent}` on `justify` and `reaudit` (default `openrouter`).
- The agent path needs the `[judge-agent]` extra and Claude Code auth (CLI login / `ANTHROPIC_API_KEY` / Bedrock-Vertex-Azure); note that `ANTHROPIC_API_KEY` is per-token and may not be cheaper than OpenRouter — the "cheaper" assumption rests on the subscription/credit path.
- The signed v2 payload now binds `judge_transport`; the preset's era is bounded by the signed `judge_recorded_at` timestamp (no preset-version capture).
- Determinism caveat: the agent transport cannot pin `temperature`, so agent-written entries are less reproducible; reaudit under `openrouter` keeps re-checks deterministic regardless of an entry's origin transport.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/
git commit -m "docs: document --judge-transport, agent auth path, and v2 judge_transport binding"
```

---

## Self-Review

**Spec coverage** (against `2026-05-31-agent-sdk-judge-transport-design.md`):
- §2 selectable `--judge-transport`, default openrouter; one shared response contract; signed `judge_transport`; explicit auth path → Tasks 5, 1/2, 4, 2/7. ✓
- §3.1 transport boundary (shared JudgeRequest/JudgeResponse/policy/parser; only text-extraction + usage differ) → Task 1 seam (`_TransportResult`, `_call_openrouter`) + Task 2. ✓
- §3.2 agent transport (no tools, `claude_code` preset append, `setting_sources=[]`, `bypassPermissions`, text→shared parser) → Task 2 `_call_agent_sdk`/`_drain_agent_query`. ✓
- §4.2 `judge_transport` signed; preset era = signed `judge_recorded_at` (no version capture, no second timestamp); `policy_hash` unchanged; served `model_id`; cache `None`-vs-`0` → Tasks 4, 1 (docstring), 2 (`_agent_served_model`/`_agent_cache_accounting`). ✓
- §5 shares v2 payload; migrate backfills `judge_transport="openrouter"`; sequenced with/after scope-fingerprint, before its operator migration → Dependency & Sequencing + Task 4 Step 7. ✓
- §6 `--judge-transport` selection; `JudgeConfigurationError` for missing SDK / no auth; explicit auth path → Tasks 5, 2. ✓
- §7 determinism caveat documented at the seam + reaudit → Task 2 docstring + Task 5 + Task 7 docs. ✓
- §8 affected code (judge.py seam, allowlist v2 payload, cli flags + migrate backfill, pyproject extra, consumers) → Tasks 1–7 map 1:1. ✓
- §9 testing (fake in-process transport; contract-parity; provenance/tamper; config-error) → Task 1 (`_fake_transport` parity + unknown-transport), Task 2 (fake SDK + auth), Task 4 (provenance/tamper via signing + reload), Task 2 (config-error). ✓
- §10 out-of-scope (tool-enabled judge; hashing the preset; agent-as-default) → not implemented; recorded. ✓

**Advisor lock-ins:** migrate folded into the atomic Task 4 (every v2-emitting site in the require-commit). ✓ Two-layer default/requirement: function default `judge_transport="openrouter"` keeps scope-fingerprint signing tests green; validator enforces presence on persisted v2; no None-normalization in verify. ✓ scope-fingerprint dependency is a *checked* precondition (Task 1 Step 1 grep-or-halt) + migrate test asserts the backfill (Task 4 Step 1). ✓ `_call_openrouter` keeps `temperature=0` and `trust_env=False` verbatim. ✓ SDK symbols flagged "confirm at implementation time"; CI rests on fake-module tests. ✓ `judge_transport` → `_judge_metadata_payload`, not `_judge_binding_identity`. ✓ **Model-id namespace split (done-check):** `DEFAULT_JUDGE_MODEL` is an OpenRouter slug invalid for the Agent SDK; per-transport default (`DEFAULT_AGENT_JUDGE_MODEL`) resolved in `call_judge`, justify drops its hardcoded `model_id`, capturing test guards it (Task 1). The fake SDK accepts any string, so only this call_judge-level test catches the defect. ✓ `max_tokens`-unused + `asyncio.run`-sync-assumption documented at the seam (Task 2). ✓

**Type consistency:** `_TransportResult(raw_text, served_model_id, prompt_tokens_total, prompt_tokens_cached)`; `call_judge(..., transport=TRANSPORT_OPENROUTER, transport_impl=None)`; `compute_judge_metadata_signature(..., judge_transport="openrouter")`; `JudgeResponse.judge_transport`; `AllowlistEntry.judge_transport`; CLI `agent`→`claude_agent_sdk` via `_CLI_TRANSPORT_CHOICES` — names used identically across all tasks. ✓

**Known soft spots the engineer resolves in-task (flagged inline, not placeholders):** the published `claude-agent-sdk` symbol/usage shapes (Task 2 header — confirm post-install, CI rests on fakes); the real auth-failure exception class (`_is_agent_auth_error` + Task 2 Step 1 fake); exact test filenames under `tests/core/`; the reaudit call-chain hops between `_run_reaudit` and the `call_judge(request)` boundary; whether `_run_migrate_judge_scope` reuses `_build_yaml_entry_text` or does line surgery (Task 4 Step 7). Each has a concrete `grep`/inspection step and a stated fallback.
