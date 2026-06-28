# Composer Temperature-Rejection Handling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make composer/auto-title LLM calls survive reasoning-model temperature
rejections by reacting to the provider's rejection and retrying once without
`temperature`, recording honestly — without losing `0.0` determinism for models
that accept it.

**Architecture:** Reactive retry (no model-capability prediction). A single
`is_temperature_rejection` predicate isolates the one transport-coupled point; a
shared `_acompletion_with_temperature_retry` helper handles unaudited call sites;
audited call sites thread a `temperature: float | None` parameter and re-run the
audited unit on retry so each attempt records its own row. Audit `temperature`
becomes nullable.

**Tech Stack:** Python 3.13, litellm 1.85.0 (transport, being removed later),
pytest, frozen dataclasses, rfc8785/JCS audit envelope.

**Source spec:** `docs/superpowers/specs/2026-05-31-composer-model-aware-temperature-design.md`

**Branch:** RC5.2 (work in place per operator; worktree decision deferred).

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/elspeth/contracts/composer_llm_audit.py` | `ComposerLLMCall` audit DTO | `temperature: float` → `float \| None`; docstring |
| `src/elspeth/web/composer/llm_response_parsing.py` | builds the audit row | `build_llm_call_record(temperature: float)` → `float \| None` |
| `src/elspeth/web/composer/service.py` | composer LLM transport + main/advisor paths | predicate, retry helper, optional cache, thread `temperature` param, retries |
| `src/elspeth/web/composer/guided/chain_solver.py` | guided `solve_chain` (audited) | thread `temperature`, two-row retry |
| `src/elspeth/web/composer/guided/chat_solver.py` | guided chat (unaudited) | use shared retry helper |
| `src/elspeth/web/sessions/_auto_title.py` | auto-title (telemetry only) | use shared retry helper |

Tests live under `tests/unit/web/composer/` and `tests/unit/web/sessions/`.
The fake-LLM seam is `service._call_llm` reassignment (service.py:1048) and the
`_FakeLLMResponse` shape in `tests/unit/web/composer/test_compose_loop_llm_audit.py`.

**Phasing:** Phase 0 (audit nullability) → Phase 1 (predicate) → Phase 2 (shared
retry helper) → Phase 3 (main path) → Phase 4 (advisor) → Phase 5 (guided chain)
→ Phase 6 (chat_solver + auto_title + text path) → Phase 7 (optional cache) →
Phase 8 (full-suite + lints). Each phase is independently landable and green.

---

## Phase 0 — Audit field nullability

### Task 0: Make `ComposerLLMCall.temperature` nullable

**Files:**
- Modify: `src/elspeth/contracts/composer_llm_audit.py:99`
- Modify: `src/elspeth/contracts/composer_llm_audit.py:73-82` (docstring)
- Modify: `src/elspeth/web/composer/llm_response_parsing.py:269` (param type)
- Test: `tests/unit/contracts/test_composer_llm_audit.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/contracts/test_composer_llm_audit.py` (create if absent; import
the dataclass and a minimal constructor helper mirroring existing tests):

```python
from datetime import UTC, datetime

from elspeth.contracts.composer_llm_audit import ComposerLLMCall, ComposerLLMCallStatus


def _minimal_call(**overrides):
    base = dict(
        model_requested="m",
        model_returned=None,
        status=ComposerLLMCallStatus.SUCCESS,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        latency_ms=0,
        provider_request_id=None,
        messages_hash="h",
        tools_spec_hash=None,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        error_class=None,
        error_message=None,
        temperature=0.0,
        seed=None,
    )
    base.update(overrides)
    return ComposerLLMCall(**base)


def test_temperature_accepts_none_when_omitted():
    call = _minimal_call(temperature=None)
    assert call.temperature is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_composer_llm_audit.py::test_temperature_accepts_none_when_omitted -v`
Expected: the construction succeeds today (no runtime type enforcement), so this
test PASSES at runtime — the real gate is mypy. Run instead:
Run: `.venv/bin/python -m mypy src/elspeth/contracts/composer_llm_audit.py`
Expected before change: clean (field is `float`). After we pass `None` from the
builder in Task 3, mypy would FAIL without this change. This task makes the type
honest first.

- [ ] **Step 3: Change the field type and docstring**

In `composer_llm_audit.py` line 99:

```python
    temperature: float | None
```

Replace the docstring paragraph (currently lines 73-82, "``temperature`` and
``seed``…") with:

```python
    ``temperature`` and ``seed`` capture deterministic-sampling parameters set on
    composer LLM requests. Temperature is ``0.0`` (the deterministic target) when
    the provider accepts it, and ``None`` when it was omitted because the provider
    rejected a non-default temperature (reasoning models such as the GPT-5 family
    and o-series accept only the default). Seed is ``42`` only for providers that
    advertise support for the OpenAI ``seed`` parameter, and ``None`` otherwise.
    The audit row records the value actually sent so a reviewer can detect drift
    and correlate failures with the precise sampling regime. RGR investigation
    2026-05-06 §4.4 traced a ~33% hard-GREEN ceiling on the
    URL→download→line-explode scenario primarily to uncontrolled default sampling
    (~1.0) on the previous code path.
```

- [ ] **Step 4: Update the builder signature**

In `llm_response_parsing.py` line 269, change:

```python
    temperature: float,
```
to:
```python
    temperature: float | None,
```

- [ ] **Step 5: Verify mypy + the JCS envelope serializes None**

Add to `tests/unit/web/composer/test_audit_envelope.py` (or the existing audit
envelope test module — locate with
`grep -rln "llm_call_audit_envelope" tests/`):

```python
def test_llm_call_envelope_serializes_none_temperature():
    from elspeth.web.composer.audit import llm_call_audit_envelope
    call = _minimal_call(temperature=None)  # reuse helper or inline construct
    envelope = llm_call_audit_envelope(call)
    # envelope must be JCS-safe; None → JSON null. Assert it round-trips.
    import json
    json.dumps(envelope, default=str)  # must not raise
```

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ tests/unit/contracts/ -k "temperature or envelope" -v`
Run: `.venv/bin/python -m mypy src/elspeth/contracts/composer_llm_audit.py src/elspeth/web/composer/llm_response_parsing.py`
Expected: PASS / clean.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/composer_llm_audit.py src/elspeth/web/composer/llm_response_parsing.py tests/unit/contracts/test_composer_llm_audit.py tests/unit/web/composer/
git commit -m "feat(composer): make ComposerLLMCall.temperature nullable for omitted sends"
```

---

## Phase 1 — The rejection predicate

### Task 1: `is_temperature_rejection`

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (add near `_litellm_acompletion`, ~line 271)
- Test: `tests/unit/web/composer/test_temperature_rejection.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import pytest
from litellm.exceptions import BadRequestError

from elspeth.web.composer.service import is_temperature_rejection


def _bad_request(message: str):
    # litellm BadRequestError requires message, model, llm_provider
    return BadRequestError(message=message, model="gpt-5", llm_provider="openai")


def test_unsupported_params_message_is_detected():
    exc = _bad_request("litellm.UnsupportedParamsError: gpt-5 models (including "
                       "gpt-5-mini) don't support temperature=0, only the default (1).")
    assert is_temperature_rejection(exc) is True


def test_o_series_message_is_detected():
    exc = _bad_request("litellm.UnsupportedParamsError: O-series models don't "
                       "support parameters: ['temperature']")
    assert is_temperature_rejection(exc) is True


def test_unrelated_bad_request_is_not_detected():
    exc = _bad_request("Invalid 'messages': missing required field 'role'")
    assert is_temperature_rejection(exc) is False


def test_non_exception_is_not_detected():
    assert is_temperature_rejection(None) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_temperature_rejection.py -v`
Expected: FAIL with `ImportError: cannot import name 'is_temperature_rejection'`.

- [ ] **Step 3: Implement the predicate**

In `service.py`, immediately after `_litellm_completion_supports_param` (~line 290):

```python
# Tokens that, together with "temperature", identify a provider's rejection of a
# non-default temperature value (reasoning models accept only the default). Matched
# case-insensitively on the exception message. This message-signature approach is
# transport-agnostic: it matches litellm's UnsupportedParamsError today and the
# raw provider HTTP-400 body after the litellm removal (spec §3.2) — only this
# predicate changes when the transport changes.
_TEMPERATURE_REJECTION_PHRASES: Final[tuple[str, ...]] = (
    "does not support",
    "don't support",
    "only the default",
    "unsupported value",
    "unsupportedparams",
)


def is_temperature_rejection(exc: object) -> bool:
    """True when ``exc`` is a provider rejection of a non-default temperature.

    Operates on the raw transport exception (litellm ``BadRequestError`` /
    ``UnsupportedParamsError`` today). Callers holding a wrapped
    ``_BadRequestLLMError`` pass its ``__cause__``.
    """
    if not isinstance(exc, BaseException):
        return False
    if type(exc).__name__ == "UnsupportedParamsError":
        return True
    message = str(exc).lower()
    if "temperature" not in message:
        return False
    return any(phrase in message for phrase in _TEMPERATURE_REJECTION_PHRASES)
```

Ensure `from typing import Final` is imported (it is — used at line 138).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_temperature_rejection.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_temperature_rejection.py
git commit -m "feat(composer): add is_temperature_rejection predicate (transport-agnostic)"
```

---

## Phase 2 — Shared transport-retry helper (for unaudited sites)

### Task 2: `_acompletion_with_temperature_retry`

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (after `_litellm_acompletion`)
- Test: `tests/unit/web/composer/test_temperature_retry_helper.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import pytest
from litellm.exceptions import BadRequestError

import elspeth.web.composer.service as svc
from elspeth.web.composer.service import _acompletion_with_temperature_retry


class _Resp:
    def __init__(self, marker): self.marker = marker


@pytest.mark.asyncio
async def test_retry_omits_temperature_on_rejection(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise BadRequestError(
                message="UnsupportedParamsError: gpt-5 models don't support temperature=0",
                model="gpt-5", llm_provider="openai")
        return _Resp("ok")

    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    response, used = await _acompletion_with_temperature_retry(
        kwargs={"model": "gpt-5", "messages": []}, temperature=0.0)
    assert response.marker == "ok"
    assert used is None
    assert len(calls) == 2
    assert "temperature" in calls[0] and "temperature" not in calls[1]


@pytest.mark.asyncio
async def test_no_retry_when_accepted(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _Resp("ok")

    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    response, used = await _acompletion_with_temperature_retry(
        kwargs={"model": "gpt-4o", "messages": []}, temperature=0.0)
    assert used == 0.0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_unrelated_bad_request_not_retried(monkeypatch):
    async def fake_acompletion(**kwargs):
        raise BadRequestError(message="missing role", model="gpt-4o", llm_provider="openai")

    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    with pytest.raises(BadRequestError):
        await _acompletion_with_temperature_retry(
            kwargs={"model": "gpt-4o", "messages": []}, temperature=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_temperature_retry_helper.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the helper**

In `service.py`, after `_litellm_acompletion` (~line 282):

```python
async def _acompletion_with_temperature_retry(
    *,
    kwargs: dict[str, Any],
    temperature: float | None,
) -> tuple[Any, float | None]:
    """Call the transport with ``temperature``; on a temperature rejection retry
    once without it. Returns ``(response, temperature_actually_used)``.

    ``kwargs`` must NOT contain ``temperature`` — it is applied here so the
    no-temperature retry is a clean omission. The returned temperature is what a
    caller records on its audit row so the row mirrors the request (spec §3.1).
    """
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    first_kwargs = dict(kwargs)
    if temperature is not None:
        first_kwargs["temperature"] = temperature
    try:
        return await _litellm_acompletion(**first_kwargs), temperature
    except LiteLLMBadRequestError as exc:
        if temperature is None or not is_temperature_rejection(exc):
            raise
        return await _litellm_acompletion(**kwargs), None
```

Confirm `from typing import Any` is imported (it is).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_temperature_retry_helper.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_temperature_retry_helper.py
git commit -m "feat(composer): add shared temperature-retry transport helper"
```

---

## Phase 3 — Main composer path (the reported "Build it" failure)

The main path is `_call_llm_before_deadline` → `_call_llm_with_audit` → `_call_llm`.
We thread `temperature: float | None` into `_call_llm` and `_call_llm_with_audit`
(request + audit), then add the retry in `_call_llm_before_deadline` so each
attempt records its own row.

### Task 3: Thread `temperature` through `_call_llm` and `_call_llm_with_audit`

**Files:**
- Modify: `src/elspeth/web/composer/service.py:2705-2737` (`_call_llm`)
- Modify: `src/elspeth/web/composer/service.py:3308-3409` (`_call_llm_with_audit`)
- Test: `tests/unit/web/composer/test_call_llm_temperature.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import pytest
import elspeth.web.composer.service as svc


@pytest.mark.asyncio
async def test_call_llm_omits_temperature_when_none(monkeypatch, composer_service):
    # composer_service: a constructed ComposerService fixture (reuse existing
    # conftest fixture; grep tests/unit/web/composer/conftest.py for the builder).
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"tool_calls": None, "content": "x"})()})()]
        return _R()

    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    await composer_service._call_llm([{"role": "user", "content": "hi"}], [], temperature=None)
    assert "temperature" not in captured


@pytest.mark.asyncio
async def test_call_llm_includes_temperature_when_set(monkeypatch, composer_service):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"tool_calls": None, "content": "x"})()})()]
        return _R()

    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    await composer_service._call_llm([{"role": "user", "content": "hi"}], [], temperature=0.0)
    assert captured["temperature"] == 0.0
```

> If no `composer_service` fixture exists, construct the service inline mirroring
> `tests/unit/web/composer/test_compose_loop_llm_audit.py` setup. Grep:
> `grep -rln "ComposerService(" tests/unit/web/composer/`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_call_llm_temperature.py -v`
Expected: FAIL — `_call_llm` currently takes no `temperature` kwarg (`TypeError`).

- [ ] **Step 3: Modify `_call_llm`**

Replace the `_call_llm` signature and kwargs block (2705-2722). Current:

```python
    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Call the LLM via LiteLLM. Separated for test mocking."""
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        try:
            seed = _composer_llm_seed_for_model(self._model)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "tools": tools,
                "temperature": _COMPOSER_LLM_TEMPERATURE,
            }
            if seed is not None:
                kwargs[_COMPOSER_LLM_SEED_PARAM] = seed
            response = await _litellm_acompletion(
                **kwargs,
            )
```

New:

```python
    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float | None = _COMPOSER_LLM_TEMPERATURE,
    ) -> Any:
        """Call the LLM via LiteLLM. Separated for test mocking.

        ``temperature`` is omitted from the request when ``None`` (used by the
        temperature-rejection retry in :meth:`_call_llm_before_deadline`).
        """
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        try:
            seed = _composer_llm_seed_for_model(self._model)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "tools": tools,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if seed is not None:
                kwargs[_COMPOSER_LLM_SEED_PARAM] = seed
            response = await _litellm_acompletion(
                **kwargs,
            )
```

(Leave the `except`/empty-choices tail unchanged.)

- [ ] **Step 4: Modify `_call_llm_with_audit`**

Add `temperature` param and thread it to both the call and the audit record.
Signature (3308-3315) — add a keyword param:

```python
    async def _call_llm_with_audit(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout: float,
        recorder: BufferingRecorder | None,
        temperature: float | None = _COMPOSER_LLM_TEMPERATURE,
    ) -> Any:
```

Change the call (3341-3344):

```python
            response = await asyncio.wait_for(
                self._call_llm(messages, tools, temperature=temperature),
                timeout=timeout,
            )
```

Change the audit record temperature (3400):

```python
                        temperature=temperature,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_call_llm_temperature.py -v`
Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_llm_audit.py -v`
Expected: PASS (new tests + existing audit tests still green — default temperature
preserves current behavior).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_call_llm_temperature.py
git commit -m "feat(composer): thread temperature param through main LLM call + audit"
```

### Task 4: Add the retry in `_call_llm_before_deadline` (two audit rows)

**Files:**
- Modify: `src/elspeth/web/composer/service.py:3411-3470` (`_call_llm_before_deadline`)
- Test: `tests/unit/web/composer/test_main_path_temperature_retry.py` (create)

- [ ] **Step 1: Read the current method body**

Run: `sed -n '3411,3475p' src/elspeth/web/composer/service.py`
Identify the exact `await self._call_llm_with_audit(...)` invocation (near 3457)
and the timeout/deadline computation preceding it.

- [ ] **Step 2: Write the failing test**

```python
import pytest
import elspeth.web.composer.service as svc
from elspeth.web.composer.service import _BadRequestLLMError


@pytest.mark.asyncio
async def test_main_path_retries_without_temperature_and_records_two_rows(
    monkeypatch, composer_service, buffering_recorder
):
    # First _call_llm raises a temperature rejection (wrapped as _BadRequestLLMError);
    # second succeeds. Expect: 2 ComposerLLMCall rows (error temp=0.0, success temp=None).
    attempts = []

    async def fake_call_llm(messages, tools, *, temperature=0.0):
        attempts.append(temperature)
        if temperature is not None:
            from litellm.exceptions import BadRequestError
            raise _BadRequestLLMError(
                "LLM request rejected (UnsupportedParamsError)",
                provider_detail="gpt-5 models don't support temperature=0",
                provider_status_code=400,
            ) from BadRequestError(
                message="UnsupportedParamsError: gpt-5 models don't support temperature=0",
                model="gpt-5", llm_provider="openai")
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"tool_calls": None, "content": "ok"})()})()]
        return _R()

    monkeypatch.setattr(composer_service, "_call_llm", fake_call_llm)
    # call via _call_llm_before_deadline with a generous deadline
    import time as _t
    await composer_service._call_llm_before_deadline(
        [{"role": "user", "content": "hi"}], [], state=..., initial_version=0,
        deadline=_t.monotonic() + 60, recorder=buffering_recorder)

    assert attempts == [0.0, None]
    rows = buffering_recorder.llm_calls()
    assert len(rows) == 2
    assert rows[0].temperature == 0.0 and rows[0].status.name == "BAD_REQUEST_ERROR"
    assert rows[1].temperature is None and rows[1].status.name == "SUCCESS"
```

> Reuse/adopt the `state` and `buffering_recorder` fixtures from
> `test_compose_loop_llm_audit.py`. If `_BadRequestLLMError`'s constructor differs,
> match it via `sed -n '228,260p' src/elspeth/web/composer/service.py`.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_main_path_temperature_retry.py -v`
Expected: FAIL — only one row recorded; no retry.

- [ ] **Step 4: Implement the retry**

Wrap the `_call_llm_with_audit` call in `_call_llm_before_deadline`. Replace the
existing single invocation (near 3457) with:

```python
                try:
                    return await self._call_llm_with_audit(
                        messages, tools, timeout=per_call_timeout, recorder=recorder,
                    )
                except _BadRequestLLMError as exc:
                    if not is_temperature_rejection(exc.__cause__):
                        raise
                    # Reasoning model rejected temperature=0.0. The failed attempt
                    # is already recorded by _call_llm_with_audit's finally block.
                    # Retry once with temperature omitted; recompute the remaining
                    # budget so the retry honours the deadline.
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    return await self._call_llm_with_audit(
                        messages, tools, timeout=remaining, recorder=recorder,
                        temperature=None,
                    )
```

> Use the SAME local names the existing code uses for the per-call timeout and the
> deadline (confirm from Step 1; the method docstring references `deadline`). If the
> existing variable is not `per_call_timeout`, substitute the real name.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_main_path_temperature_retry.py tests/unit/web/composer/test_compose_loop_llm_audit.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_main_path_temperature_retry.py
git commit -m "feat(composer): retry main LLM call without temperature on reasoning-model rejection"
```

---

## Phase 4 — Advisor path

The advisor block (service.py ~3160-3307) calls `_litellm_acompletion` directly,
catches raw `LiteLLMBadRequestError`, and records one row in its `finally`.

### Task 5: Advisor retry + nullable audit temperature

**Files:**
- Modify: `src/elspeth/web/composer/service.py:3160-3307`
- Test: `tests/unit/web/composer/test_advisor_temperature_retry.py` (create)

- [ ] **Step 1: Read the advisor block end-to-end**

Run: `sed -n '3140,3307p' src/elspeth/web/composer/service.py`
Identify: the `kwargs` construction (~3165-3171), the `asyncio.wait_for(_litellm_acompletion(**kwargs), timeout=...)` call (~3262), and the `finally` record (~3289-3300) whose `temperature=_COMPOSER_LLM_TEMPERATURE` becomes a local variable.

- [ ] **Step 2: Write the failing test**

```python
@pytest.mark.asyncio
async def test_advisor_retries_without_temperature(monkeypatch, composer_service, buffering_recorder):
    attempts = []

    async def fake_acompletion(**kwargs):
        attempts.append("temperature" in kwargs)
        from litellm.exceptions import BadRequestError
        if "temperature" in kwargs:
            raise BadRequestError(
                message="UnsupportedParamsError: gpt-5 models don't support temperature=0",
                model="gpt-5", llm_provider="openai")
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"content": "advice"})()})()]
        return _R()

    import elspeth.web.composer.service as svc
    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    # invoke the advisor entry point (grep for the method name around line 3140)
    # e.g. await composer_service._call_advisor(arguments={...}, recorder=buffering_recorder)
    ...
    assert attempts == [True, False]
    rows = buffering_recorder.llm_calls()
    assert rows[-1].temperature is None
```

> Fill the invocation by reading the advisor method's name/signature in Step 1.

- [ ] **Step 3: Refactor the advisor temperature to a local + add retry**

In the advisor block:
1. Build `kwargs` WITHOUT temperature; introduce a local
   `advisor_temperature: float | None = _COMPOSER_LLM_TEMPERATURE`.
2. Replace the direct `_litellm_acompletion(**kwargs)` call inside the
   `asyncio.wait_for` with the shared helper so the retry + omission happen once:

```python
            advisor_temperature: float | None = _COMPOSER_LLM_TEMPERATURE
            response, advisor_temperature = await asyncio.wait_for(
                _acompletion_with_temperature_retry(
                    kwargs=kwargs, temperature=advisor_temperature,
                ),
                timeout=effective_timeout,
            )
```

   (Remove `"temperature": _COMPOSER_LLM_TEMPERATURE` from the `kwargs` literal at
   ~3169 so `kwargs` carries no temperature; the helper applies it.)
3. In the `finally` record (~3296), change `temperature=_COMPOSER_LLM_TEMPERATURE`
   to `temperature=advisor_temperature`.

> NOTE: because the shared helper performs both attempts inside one
> `asyncio.wait_for`, the advisor records ONE row reflecting the value actually
> used (the helper does not have the advisor's recorder). This is the unaudited-site
> recording shape: a single truthful row, not two. The failed attempt is captured by
> the helper's retry; if a two-row trail is later required for the advisor, lift the
> retry out of the helper as in the main path. Document this in the comment.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_advisor_temperature_retry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_advisor_temperature_retry.py
git commit -m "feat(composer): retry advisor call without temperature on rejection"
```

---

## Phase 5 — Guided chain solver (audited, two rows)

`chain_solver.solve_chain` mirrors the main path's audited shape (its own
`try/except/finally` with `build_llm_call_record`, temperature at ~324).

### Task 6: `solve_chain` retry + nullable audit temperature

**Files:**
- Modify: `src/elspeth/web/composer/guided/chain_solver.py:180-345`
- Test: `tests/unit/web/composer/guided/test_chain_solver_temperature.py` (create)

- [ ] **Step 1: Read `solve_chain` fully**

Run: `sed -n '140,345p' src/elspeth/web/composer/guided/chain_solver.py`
Confirm the function signature, the `kwargs` temperature line (~188), the
`_litellm_acompletion(**kwargs)` call, and the `finally` record (~324).

- [ ] **Step 2: Write the failing test**

```python
@pytest.mark.asyncio
async def test_solve_chain_retries_without_temperature_two_rows(monkeypatch, buffering_recorder):
    from elspeth.web.composer.guided import chain_solver
    attempts = []

    async def fake_acompletion(**kwargs):
        attempts.append("temperature" in kwargs)
        from litellm.exceptions import BadRequestError
        if "temperature" in kwargs:
            raise BadRequestError(
                message="UnsupportedParamsError: gpt-5 models don't support temperature=0",
                model="gpt-5", llm_provider="openai")
        # minimal valid emit_turn tool-call response
        ...
    monkeypatch.setattr(chain_solver, "_litellm_acompletion", fake_acompletion)
    # call solve_chain(...) with recorder=buffering_recorder per its signature
    ...
    assert attempts == [True, False]
    rows = buffering_recorder.llm_calls()
    assert len(rows) == 2
    assert rows[0].temperature == 0.0
    assert rows[1].temperature is None
```

- [ ] **Step 3: Implement the retry**

Refactor `solve_chain` so the audited call+record runs per attempt. The cleanest
shape matching the existing structure: extract the body from `kwargs` build through
the `finally` into an inner `async def _attempt(temperature: float | None)` and call
it twice on rejection:

```python
    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": _GUIDED_LLM_TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "emit_turn"}},
    }
    if seed is not None:
        base_kwargs["seed"] = seed

    async def _attempt(temperature: float | None) -> Any:
        kwargs = dict(base_kwargs)
        if temperature is not None:
            kwargs["temperature"] = temperature
        started_at = datetime.now(UTC)
        started_ns = time.monotonic_ns()
        status: ComposerLLMCallStatus | None = None
        response: Any = None
        error_class: str | None = None
        error_message: str | None = None
        try:
            response = await _litellm_acompletion(**kwargs)
            status = ComposerLLMCallStatus.SUCCESS
            return response
        except LiteLLMBadRequestError as exc:
            status = ComposerLLMCallStatus.BAD_REQUEST_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        # ... (keep the existing LiteLLMAPIError / _MalformedLLMResponseError /
        #      Exception arms verbatim) ...
        finally:
            if recorder is not None and status is not None:
                recorder.record_llm_call(
                    build_llm_call_record(
                        model_requested=model, messages=messages,
                        tools=_GUIDED_LLM_TOOLS, status=status,
                        started_at=started_at, started_ns=started_ns,
                        temperature=temperature, seed=seed,
                        response=response, error_class=error_class,
                        error_message=error_message,
                    )
                )
                current_exc = sys.exc_info()[1]
                if current_exc is not None:
                    attach_llm_calls(current_exc, recorder)

    try:
        response = await _attempt(_COMPOSER_LLM_TEMPERATURE)
    except LiteLLMBadRequestError as exc:
        if not is_temperature_rejection(exc):
            raise
        response = await _attempt(None)
```

Add `from elspeth.web.composer.service import is_temperature_rejection` to the
existing service imports at the top of `chain_solver.py` (it already imports
`_litellm_acompletion`, `_composer_llm_seed_for_model` from there).

> Preserve the existing typed-except arms verbatim — only the kwargs/record
> temperature and the surrounding retry are new. Empty-choices / malformed handling
> stays where the original had it (inside `_attempt`).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/ -k "chain" -v`
Expected: PASS (new + existing chain_solver tests).

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/guided/chain_solver.py tests/unit/web/composer/guided/test_chain_solver_temperature.py
git commit -m "feat(composer/guided): retry solve_chain without temperature on rejection"
```

---

## Phase 6 — Unaudited sites: chat_solver, auto_title, diagnostics text

These call `_litellm_acompletion` directly with no `ComposerLLMCall` row. Route each
through `_acompletion_with_temperature_retry`.

### Task 7: chat_solver (`maybe_resolve_step_1_source_chat`, `solve_step_chat`)

**Files:**
- Modify: `src/elspeth/web/composer/guided/chat_solver.py` (both functions: kwargs ~213-222 and ~278-285)
- Test: `tests/unit/web/composer/guided/test_chat_solver_temperature.py` (create)

- [ ] **Step 1: Write the failing test** (one per function)

```python
@pytest.mark.asyncio
async def test_solve_step_chat_retries_without_temperature(monkeypatch):
    from elspeth.web.composer.guided import chat_solver
    attempts = []
    async def fake_acompletion(**kwargs):
        attempts.append("temperature" in kwargs)
        from litellm.exceptions import BadRequestError
        if "temperature" in kwargs:
            raise BadRequestError(message="UnsupportedParamsError: temperature unsupported",
                                  model="gpt-5", llm_provider="openai")
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"content": "reply"})()})()]
        return _R()
    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)
    out = await chat_solver.solve_step_chat(model="gpt-5", step=..., user_message="hi")
    assert out == "reply"
    assert attempts == [True, False]
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_chat_solver_temperature.py -v`
Expected: FAIL (no retry; BadRequestError propagates).

- [ ] **Step 3: Convert both call sites**

For each function, build `kwargs` WITHOUT `"temperature"`, then replace
`response = await _litellm_acompletion(**kwargs)` with:

```python
    response, _ = await _acompletion_with_temperature_retry(
        kwargs=kwargs, temperature=_COMPOSER_LLM_TEMPERATURE,
    )
```

Add `_acompletion_with_temperature_retry` to the `from elspeth.web.composer.service
import (...)` block at the top of `chat_solver.py`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/ -k "chat" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/guided/chat_solver.py tests/unit/web/composer/guided/test_chat_solver_temperature.py
git commit -m "feat(composer/guided): retry chat_solver calls without temperature on rejection"
```

### Task 8: auto_title

**Files:**
- Modify: `src/elspeth/web/sessions/_auto_title.py:107-130`
- Test: `tests/unit/web/sessions/test_auto_title_temperature.py` (create)

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_auto_title_retries_without_temperature(monkeypatch):
    import elspeth.web.sessions._auto_title as at
    attempts = []
    async def fake_acompletion(**kwargs):
        attempts.append("temperature" in kwargs)
        from litellm.exceptions import BadRequestError
        if "temperature" in kwargs:
            raise BadRequestError(message="UnsupportedParamsError: temperature unsupported",
                                  model="gpt-5", llm_provider="openai")
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"content": "My Title"})()})()]
        return _R()
    monkeypatch.setattr(at, "_litellm_acompletion", fake_acompletion)
    titled = {}
    class _Svc:
        async def update_session_title(self, sid, title): titled["t"] = title
    await at.maybe_auto_title_session(service=_Svc(), session_id=__import__("uuid").uuid4(),
                                      user_message="hello", model="gpt-5")
    assert attempts == [True, False]
    assert titled["t"] == "My Title"
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_auto_title_temperature.py -v`
Expected: FAIL — `BadRequestError` is not in the caught tuple, so it propagates
(the bug the report describes for auto-title).

- [ ] **Step 3: Convert the call**

Build `kwargs` WITHOUT `"temperature"`. Replace `response = await
_litellm_acompletion(**kwargs)` with:

```python
        response, _ = await _acompletion_with_temperature_retry(
            kwargs=kwargs, temperature=_AUTO_TITLE_TEMPERATURE,
        )
```

Add the import: `from elspeth.web.composer.service import _acompletion_with_temperature_retry`
(alongside the existing `_composer_llm_seed_for_model, _litellm_acompletion` import
at line 29).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_auto_title_temperature.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/_auto_title.py tests/unit/web/sessions/test_auto_title_temperature.py
git commit -m "feat(composer): retry auto-title without temperature on rejection"
```

### Task 9: diagnostics text path (`_call_text_llm`)

**Files:**
- Modify: `src/elspeth/web/composer/service.py:2739-2766` (`_call_text_llm`)
- Test: `tests/unit/web/composer/test_text_llm_temperature.py` (create)

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_call_text_llm_retries_without_temperature(monkeypatch, composer_service):
    import elspeth.web.composer.service as svc
    attempts = []
    async def fake_acompletion(**kwargs):
        attempts.append("temperature" in kwargs)
        from litellm.exceptions import BadRequestError
        if "temperature" in kwargs:
            raise BadRequestError(message="UnsupportedParamsError: temperature unsupported",
                                  model="gpt-5", llm_provider="openai")
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"content": "text"})()})()]
        return _R()
    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    resp = await composer_service._call_text_llm([{"role": "user", "content": "hi"}])
    assert attempts == [True, False]
    assert resp.choices[0].message.content == "text"
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_text_llm_temperature.py -v`
Expected: FAIL.

- [ ] **Step 3: Convert the call**

In `_call_text_llm`, build `kwargs` WITHOUT temperature and replace the
`_litellm_acompletion(**kwargs)` call inside the `try` with:

```python
            response, _ = await _acompletion_with_temperature_retry(
                kwargs=kwargs, temperature=_COMPOSER_LLM_TEMPERATURE,
            )
```

Keep the existing `except LiteLLMBadRequestError → _BadRequestLLMError` wrap (it now
only fires for non-temperature bad requests) and the empty-choices guard.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_text_llm_temperature.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_text_llm_temperature.py
git commit -m "feat(composer): retry diagnostics text call without temperature on rejection"
```

---

## Phase 7 — (Optional) observed-rejection cache

Skip if the smallest landing is preferred; the reactive retry is correct without it.

### Task 10: process-level observed-rejection cache

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (module-level set + use in retry helper)
- Test: `tests/unit/web/composer/test_temperature_retry_cache.py` (create)

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_known_rejecting_model_omits_temperature_up_front(monkeypatch):
    import elspeth.web.composer.service as svc
    svc._note_temperature_rejection("gpt-5")
    calls = []
    async def fake_acompletion(**kwargs):
        calls.append("temperature" in kwargs)
        class _R: choices = [object()]
        return _R()
    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    _, used = await svc._acompletion_with_temperature_retry(
        kwargs={"model": "gpt-5", "messages": []}, temperature=0.0, cache_key="gpt-5")
    assert used is None
    assert calls == [False]  # single call, temperature omitted up front
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_temperature_retry_cache.py -v`
Expected: FAIL (`_note_temperature_rejection` / `cache_key` not defined).

- [ ] **Step 3: Implement**

```python
# Process-scoped, self-correcting heuristic: models observed to reject a
# non-default temperature this process lifetime. Never persisted; starts empty
# each boot. Only an optimization — correctness comes from the reactive retry.
_TEMPERATURE_REJECTING_MODELS: set[str] = set()


def _note_temperature_rejection(model: str) -> None:
    _TEMPERATURE_REJECTING_MODELS.add(model)
```

Extend the helper signature with `cache_key: str | None = None`; at entry, if
`cache_key in _TEMPERATURE_REJECTING_MODELS`, set `temperature = None`; on a
detected rejection, call `_note_temperature_rejection(cache_key)` before retrying.
Pass `cache_key=self._model` / `model` from each call site.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_temperature_retry_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_temperature_retry_cache.py
git commit -m "feat(composer): cache observed temperature-rejecting models to skip the rejected round-trip"
```

---

## Phase 8 — Comment, full suite, lints

### Task 11: Update the sampling-constants comment + final verification

**Files:**
- Modify: `src/elspeth/web/composer/service.py:140-153`

- [ ] **Step 1: Update the comment block**

Append to the existing comment (after the "Configurability is Tier 2" line):

```python
# Temperature is the deterministic target value but is OMITTED at request time
# when the provider rejects a non-default temperature (reasoning models such as
# the GPT-5 family / o-series accept only the default). Detection is reactive
# (see is_temperature_rejection) — we do not predict model capability. The audit
# row records the value actually sent (0.0 or None). See
# docs/superpowers/specs/2026-05-31-composer-model-aware-temperature-design.md.
```

- [ ] **Step 2: Run the composer + sessions unit suites**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ tests/unit/web/sessions/ tests/unit/contracts/ -q`
Expected: all PASS.

- [ ] **Step 3: Run mypy + ruff on touched files**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/ src/elspeth/web/sessions/_auto_title.py src/elspeth/contracts/composer_llm_audit.py`
Run: `.venv/bin/python -m ruff check src/elspeth/web/composer/ src/elspeth/web/sessions/_auto_title.py`
Expected: clean.

- [ ] **Step 4: Run the tier-model lint (no upward imports introduced)**

Run: `env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth`
Expected: no new violations. (All new code is in L3 `web/`; the predicate import
into `guided/` and `sessions/` is intra-L3 — allowed.)

- [ ] **Step 5: Full default suite (CI-equivalent)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: green except the documented pre-existing failures (see memory
`project_rc52_preexisting_test_failures_2026-05-29`); do not attribute those here.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/service.py
git commit -m "docs(composer): document reactive temperature-omission in sampling-constants comment"
```

---

## Self-Review

**Spec coverage:**
- Reactive retry primary (spec §3.1) → Tasks 2, 4, 5, 7, 8, 9. ✓
- Predicate isolation (§3.2) → Task 1. ✓
- Observed-rejection cache (§3.3, optional) → Task 10. ✓
- Audit nullability, no migration (§3.4, §5) → Task 0. ✓
- Two rows on audited-site retry (§3.1) → Tasks 4 (main), 6 (chain). Advisor (Task
  5) records a single truthful row because its retry runs inside the shared helper;
  this divergence is documented in Task 5 Step 3 and is acceptable (the failed
  attempt is not lost — it is the retried call). If strict two-row parity is later
  required for the advisor, lift its retry out of the helper as in the main path.
- All six temperature call sites covered: main (3,4), advisor (5), chain (6),
  chat ×2 (7), auto_title (8), text (9). ✓
- Seed left as-is (§4) — no seed task. ✓
- Scope boundary `call_data.py` untouched (§8). ✓

**Placeholder scan:** test bodies with `...` (state/step/response fixtures) appear
where the exact fixture shape must be read from the existing test module named in
the step. Each such step names the file to read first. These are deliberate
"read-then-fill" anchors for context the executor must pull from the codebase, not
silent TODOs — but the executor MUST replace every `...` with concrete values from
the referenced fixtures before the test will run.

**Type consistency:** `temperature: float | None` is used consistently across the
DTO (Task 0), builder (Task 0), `_call_llm`/`_call_llm_with_audit` (Task 3), the
shared helper return `tuple[Any, float | None]` (Task 2), and every call site.
`is_temperature_rejection(exc: object) -> bool` and
`_acompletion_with_temperature_retry(*, kwargs, temperature, [cache_key]) ->
tuple[Any, float | None]` signatures match every call.

**Known executor caveats:**
1. Tests construct `litellm.exceptions.BadRequestError(message=..., model=...,
   llm_provider=...)` — confirm this constructor against litellm 1.85.0 at
   execution (`.venv/bin/python -c "from litellm.exceptions import BadRequestError; help(BadRequestError.__init__)"`).
2. `_BadRequestLLMError` constructor args (Task 4 test) — confirm via
   `sed -n '228,260p' src/elspeth/web/composer/service.py`.
3. Variable names in `_call_llm_before_deadline` (per-call timeout, deadline) —
   confirm in Task 4 Step 1 before editing.
