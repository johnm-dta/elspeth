# Composer Operator-Set Sampling Config — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make composer LLM sampling (`temperature`, `seed`) operator-set `WebSettings` fields sent verbatim, delete all capability-inference and reactive-retry machinery, validate the config against the provider at boot (fatal only on config-rejection), and emit the effective config as a boot telemetry event.

**Architecture:** Two nullable `WebSettings` fields default to `None` (omit from request). Composer service methods read `self._settings` directly; free functions (guided solvers, auto-title) receive the two values threaded from their settings-holding callers. A boot probe in the `app.py` lifespan exercises the configured sampling against the real provider and refuses boot on a config-rejection 400 only. The change is overwhelmingly subtractive — it deletes the entire uncommitted reactive-retry feature plus two pre-existing inference helpers.

**Tech Stack:** Python 3.13, Pydantic v2 (`WebSettings`), litellm 1.85.0 (transport), FastAPI lifespan, OpenTelemetry (boot event), pytest.

**Source spec:** `docs/superpowers/specs/2026-06-03-composer-operator-set-sampling-config-design.md`

**Branch:** RC5.2 (work in place, per operator).

**Baseline note:** All `service.py` / `chain_solver.py` / `chat_solver.py` / `_auto_title.py` line numbers below refer to the **pre-uncommitted-work baseline** (i.e. `git show HEAD:<file>`), which Phase 0 restores. After Phase 0 the working tree matches that baseline.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/elspeth/web/config.py` | `WebSettings` | add `composer_temperature`, `composer_seed`, `composer_boot_probe_enabled` |
| `src/elspeth/contracts/composer_llm_audit.py` | `ComposerLLMCall` DTO | `temperature: float` → `float \| None` + docstring |
| `src/elspeth/web/composer/llm_response_parsing.py` | `build_llm_call_record` | `temperature: float` → `float \| None` |
| `src/elspeth/web/composer/service.py` | composer LLM call sites + inference helpers | read settings; **delete** seed probe + sampling constants |
| `src/elspeth/web/composer/guided/chain_solver.py` | `solve_chain` | accept `temperature`/`seed` params; drop probe |
| `src/elspeth/web/composer/guided/chat_solver.py` | two chat functions | accept `temperature`/`seed` params; drop probe |
| `src/elspeth/web/sessions/_guided_solve_chain.py` | chain wrapper | thread `temperature`/`seed` |
| `src/elspeth/web/sessions/_guided_step_chat.py` | chat wrapper | thread `temperature`/`seed` |
| `src/elspeth/web/sessions/routes/composer.py` | guided routes | pass settings values into wrappers |
| `src/elspeth/web/sessions/_auto_title.py` | auto-title | accept `temperature`/`seed` params; drop probe/constant |
| `src/elspeth/web/sessions/routes/messages.py` | auto-title caller | pass settings values |
| `src/elspeth/web/app.py` | lifespan | boot probe + `composer.boot_config` event + `ComposerBootConfigError` |
| `docs/adr/` | ADR | record the configurability decision |

---

## Phase 0 — Archive and revert the uncommitted reactive-retry work

**This phase is an operator-confirmed git operation. It moves the abandoned uncommitted work onto a recoverable archive branch, then returns RC5.2 to the clean pre-retry baseline. Nothing is discarded — the work survives in git history on the archive branch.**

### Task 0: Preserve then revert

**Files:** working tree (no source edits)

- [ ] **Step 1: Confirm the current dirty set**

Run: `git status --short`
Expected: the modified `service.py` / `chain_solver.py` / `chat_solver.py` / `_auto_title.py` / `composer_llm_audit.py` / `llm_response_parsing.py` and the untracked `test_*temperature*` files (plus modified `test_service.py`, `test_composer_llm_audit.py`).

- [ ] **Step 2: Archive the uncommitted work to a recoverable branch**

```bash
git checkout -b archive/composer-reactive-temperature-retry
git add -A
git commit --no-verify -m "archive: abandoned reactive-temperature-retry implementation (superseded by operator-set sampling config)"
```

The uncommitted changes follow you onto the new branch and are captured by the commit. (`--no-verify`: this is a throwaway archive of known-incomplete work, not a quality-gated landing.)

- [ ] **Step 3: Return to RC5.2 (now clean at the spec commit)**

```bash
git checkout RC5.2
git status --short
```
Expected: **empty** — the working tree is clean at `fc2c57d66` (the spec commit). The reactive-retry work is preserved on `archive/composer-reactive-temperature-retry`.

- [ ] **Step 4: Verify the baseline is the pre-retry state**

Run: `git show HEAD:src/elspeth/web/composer/service.py | grep -n "is_temperature_rejection\|_TEMPERATURE_REJECTING_MODELS\|_acompletion_with_temperature_retry"`
Expected: **no output** (those symbols only existed in the reverted work). Confirms the clean baseline.

- [ ] **Step 5: Run the composer suite to confirm green baseline**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ tests/unit/web/sessions/ tests/unit/contracts/ -q`
Expected: green (the documented pre-existing failures aside — see memory `project_rc52_preexisting_test_failures_2026-05-29`).

---

## Phase 1 — Config fields

### Task 1: Add `composer_temperature`, `composer_seed`, `composer_boot_probe_enabled`

**Files:**
- Modify: `src/elspeth/web/config.py` (in `WebSettings`, beside `composer_model` ~line 52)
- Test: `tests/unit/web/test_config.py` (locate with `grep -rln "class WebSettings\|WebSettings(" tests/unit/web/ | head`)

- [ ] **Step 1: Write the failing test**

```python
def test_composer_sampling_defaults_to_none_and_probe_enabled():
    settings = _make_web_settings()  # reuse the existing WebSettings test factory
    assert settings.composer_temperature is None
    assert settings.composer_seed is None
    assert settings.composer_boot_probe_enabled is True


def test_composer_temperature_accepts_in_range_and_rejects_out_of_range():
    assert _make_web_settings(composer_temperature=0.0).composer_temperature == 0.0
    assert _make_web_settings(composer_temperature=1.5).composer_temperature == 1.5
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _make_web_settings(composer_temperature=2.5)
    with pytest.raises(ValidationError):
        _make_web_settings(composer_temperature=-0.1)


def test_composer_seed_accepts_int_and_none():
    assert _make_web_settings(composer_seed=42).composer_seed == 42
    assert _make_web_settings(composer_seed=None).composer_seed is None
```

> `_make_web_settings` stands for the existing test constructor for `WebSettings`. Find it first: `grep -rn "def _make_web_settings\|WebSettings(" tests/unit/web/test_config.py`. If the factory uses a different name, use it verbatim.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/test_config.py -k "composer_sampling or composer_temperature or composer_seed" -v`
Expected: FAIL — `composer_temperature` / `composer_seed` / `composer_boot_probe_enabled` are not fields yet.

- [ ] **Step 3: Add the fields**

In `src/elspeth/web/config.py`, immediately after `composer_model: str = "gpt-5.5"`:

```python
    # Operator-set LLM sampling. Default None => omitted from the provider
    # request (the only coherent default given a reasoning-model default like
    # gpt-5.5, which rejects any non-default temperature). Sent verbatim when
    # set; a provider that rejects a set value surfaces it as the operator's
    # config error (validated at boot — see app.py lifespan). See ADR + spec
    # docs/superpowers/specs/2026-06-03-composer-operator-set-sampling-config-design.md.
    composer_temperature: float | None = Field(default=None, ge=0, le=2)
    composer_seed: int | None = None
    # When True (default), the lifespan probes the configured composer model(s)
    # with the configured sampling and refuses boot on a config-rejection 400.
    # Tests/offline dev set False to avoid a real provider call on every boot.
    composer_boot_probe_enabled: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/test_config.py -k "composer_sampling or composer_temperature or composer_seed" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/config.py tests/unit/web/test_config.py
git commit -m "feat(composer): add operator-set composer_temperature/composer_seed + boot-probe flag"
```

---

## Phase 2 — Audit field nullability

### Task 2: Make `ComposerLLMCall.temperature` and the builder param nullable

**Files:**
- Modify: `src/elspeth/contracts/composer_llm_audit.py` (field + docstring ~lines 73-99)
- Modify: `src/elspeth/web/composer/llm_response_parsing.py:269` (`build_llm_call_record` param)
- Test: `tests/unit/contracts/test_composer_llm_audit.py`

- [ ] **Step 1: Write the failing test (mypy gate)**

Add to `tests/unit/contracts/test_composer_llm_audit.py` (reuse the module's existing constructor helper; if none, mirror the field list):

```python
def test_temperature_accepts_none():
    call = _minimal_call(temperature=None)
    assert call.temperature is None
```

- [ ] **Step 2: Run mypy to verify the type is currently non-nullable**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/composer_llm_audit.py`
Expected: clean now (field is `float`). After Phase 3 passes `None`, mypy would FAIL without this change — make the type honest first.

- [ ] **Step 3: Change the field type and docstring**

In `composer_llm_audit.py`, change the `temperature` field to:

```python
    temperature: float | None
```

Replace the `temperature`/`seed` docstring paragraph with:

```python
    ``temperature`` and ``seed`` capture the deterministic-sampling parameters
    actually sent on composer LLM requests. Both are operator-set
    (``WebSettings.composer_temperature`` / ``composer_seed``) and recorded as
    sent: the configured value when set, or ``None`` when the operator left it
    unset and it was omitted from the request. The audit row mirrors the request
    so a reviewer can correlate failures with the precise sampling regime.
```

- [ ] **Step 4: Update the builder signature**

In `llm_response_parsing.py:269`, change `temperature: float,` to `temperature: float | None,`.

- [ ] **Step 5: Verify mypy + tests**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/composer_llm_audit.py src/elspeth/web/composer/llm_response_parsing.py`
Run: `.venv/bin/python -m pytest tests/unit/contracts/test_composer_llm_audit.py -v`
Expected: clean / PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/composer_llm_audit.py src/elspeth/web/composer/llm_response_parsing.py tests/unit/contracts/test_composer_llm_audit.py
git commit -m "feat(composer): make ComposerLLMCall.temperature nullable for omitted sends"
```

---

## Phase 3 — Main composer path reads settings

The methods `_call_llm` (service.py:2705) and `_call_llm_with_audit` (service.py:3308) are instance methods with `self._settings` and `self._model`. They read the config directly — no param threading.

### Task 3: `_call_llm` + `_call_llm_with_audit` source temperature/seed from settings

**Files:**
- Modify: `src/elspeth/web/composer/service.py:2705-2737` (`_call_llm`)
- Modify: `src/elspeth/web/composer/service.py:3400` (`_call_llm_with_audit` audit record)
- Test: `tests/unit/web/composer/test_call_llm_temperature.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import elspeth.web.composer.service as svc


def _fake_response():
    class _R:
        choices = [type("C", (), {"message": type("M", (), {"tool_calls": None, "content": "x"})()})()]
    return _R()


@pytest.mark.asyncio
async def test_call_llm_omits_temperature_when_setting_none(monkeypatch, composer_service):
    # composer_service fixture: a ComposerServiceImpl whose settings leave
    # composer_temperature=None, composer_seed=None (the defaults). Reuse the
    # builder in tests/unit/web/composer/test_compose_loop_llm_audit.py.
    captured = {}
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _fake_response()
    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    await composer_service._call_llm([{"role": "user", "content": "hi"}], [])
    assert "temperature" not in captured
    assert "seed" not in captured


@pytest.mark.asyncio
async def test_call_llm_sends_configured_temperature_and_seed(monkeypatch, composer_service_with):
    # composer_service_with(composer_temperature=0.0, composer_seed=42)
    captured = {}
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _fake_response()
    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    service = composer_service_with(composer_temperature=0.0, composer_seed=42)
    await service._call_llm([{"role": "user", "content": "hi"}], [])
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 42
```

> The composer test module builds a `ComposerServiceImpl` from a `WebSettings`. Read `tests/unit/web/composer/test_compose_loop_llm_audit.py` for the exact construction and adapt it into a `composer_service` / `composer_service_with(**overrides)` fixture (or inline the construction). Grep: `grep -rn "ComposerServiceImpl(" tests/unit/web/composer/ | head`.

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_call_llm_temperature.py -v`
Expected: FAIL — `_call_llm` currently injects `temperature` unconditionally and derives seed from the probe.

- [ ] **Step 3: Rewrite the `_call_llm` kwargs block**

Replace the kwargs construction in `_call_llm` (service.py:2716-2722). Current:

```python
            seed = _composer_llm_seed_for_model(self._model)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "tools": tools,
                "temperature": _COMPOSER_LLM_TEMPERATURE,
            }
            if seed is not None:
                kwargs[_COMPOSER_LLM_SEED_PARAM] = seed
```

New:

```python
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "tools": tools,
            }
            if self._settings.composer_temperature is not None:
                kwargs["temperature"] = self._settings.composer_temperature
            if self._settings.composer_seed is not None:
                kwargs[_COMPOSER_LLM_SEED_PARAM] = self._settings.composer_seed
```

- [ ] **Step 4: Update the `_call_llm_with_audit` audit record temperature (service.py:3400)**

Change `temperature=_COMPOSER_LLM_TEMPERATURE,` to:

```python
                        temperature=self._settings.composer_temperature,
```

(Leave its `seed=_composer_llm_seed_for_model(self._model)` line for now — Task 9 swaps it; but if mypy/tests require, change it here to `seed=self._settings.composer_seed`. Prefer changing it here for consistency.)

Change the `_call_llm_with_audit` seed line (service.py ~3401, `seed=_composer_llm_seed_for_model(self._model),`) to:

```python
                        seed=self._settings.composer_seed,
```

- [ ] **Step 5: Update existing audit tests that pin temperature=0.0**

The compose-loop audit tests assert `row.temperature == 0.0`. With the default `None`, set `composer_temperature=0.0` (and `composer_seed=42`) on those tests' `WebSettings`, OR update the assertion to the configured value. Run them and fix expectations:

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_llm_audit.py -v`
Expected: PASS after aligning each settings fixture / assertion to the explicit configured sampling.

- [ ] **Step 6: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_call_llm_temperature.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_call_llm_temperature.py tests/unit/web/composer/test_compose_loop_llm_audit.py
git commit -m "feat(composer): main LLM path sources temperature/seed from settings"
```

---

## Phase 4 — Advisor path reads settings

### Task 4: `_call_advisor_with_audit` sources temperature/seed from settings

**Files:**
- Modify: `src/elspeth/web/composer/service.py:3199-3206` (advisor kwargs) and `:3297` (audit record)
- Test: `tests/unit/web/composer/test_advisor_temperature.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import pytest
import elspeth.web.composer.service as svc


@pytest.mark.asyncio
async def test_advisor_omits_temperature_when_none(monkeypatch, composer_service):
    captured = {}
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"content": "advice"})()})()]
        return _R()
    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    # invoke the advisor entry point (read its signature in test_*advisor* first)
    await composer_service._call_advisor_with_audit(... , recorder=None)
    assert "temperature" not in captured
```

> Fill `...` from the real `_call_advisor_with_audit` signature (read `sed -n '3137,3210p' src/elspeth/web/composer/service.py` post-Phase-0). Grep existing advisor tests for the call shape: `grep -rln "_call_advisor_with_audit\|_call_advisor" tests/unit/web/composer/`.

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_advisor_temperature.py -v`
Expected: FAIL — advisor injects `temperature` unconditionally.

- [ ] **Step 3: Rewrite advisor kwargs + audit record**

Replace the advisor kwargs block (service.py:3197-3206). Current:

```python
        advisor_seed = _composer_llm_seed_for_model(advisor_model)
        kwargs: dict[str, Any] = {
            "model": advisor_model,
            "messages": messages,
            "temperature": _COMPOSER_LLM_TEMPERATURE,
            "max_tokens": max_completion,
        }
        if advisor_seed is not None:
            kwargs[_COMPOSER_LLM_SEED_PARAM] = advisor_seed
```

New:

```python
        kwargs: dict[str, Any] = {
            "model": advisor_model,
            "messages": messages,
            "max_tokens": max_completion,
        }
        if self._settings.composer_temperature is not None:
            kwargs["temperature"] = self._settings.composer_temperature
        if self._settings.composer_seed is not None:
            kwargs[_COMPOSER_LLM_SEED_PARAM] = self._settings.composer_seed
```

Change the audit record (service.py:3297) `temperature=_COMPOSER_LLM_TEMPERATURE,` to:

```python
                        temperature=self._settings.composer_temperature,
```

And its `seed=advisor_seed,` (service.py ~3300) to `seed=self._settings.composer_seed,`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_advisor_temperature.py -v`
Run: `.venv/bin/python -m pytest tests/unit/web/composer/ -k advisor -v`
Expected: PASS (existing advisor tests may need their settings fixture to set `composer_temperature` if they assert a value).

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_advisor_temperature.py
git commit -m "feat(composer): advisor path sources temperature/seed from settings"
```

---

## Phase 5 — Diagnostics text path reads settings

### Task 5: `_call_text_llm` sources temperature/seed from settings

**Files:**
- Modify: `src/elspeth/web/composer/service.py:2748-2754` (`_call_text_llm` kwargs)
- Test: `tests/unit/web/composer/test_text_llm_temperature.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import pytest
import elspeth.web.composer.service as svc


@pytest.mark.asyncio
async def test_text_llm_omits_temperature_when_none(monkeypatch, composer_service):
    captured = {}
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"content": "text"})()})()]
        return _R()
    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)
    await composer_service._call_text_llm([{"role": "user", "content": "hi"}])
    assert "temperature" not in captured
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_text_llm_temperature.py -v`
Expected: FAIL.

- [ ] **Step 3: Rewrite `_call_text_llm` kwargs (service.py:2748-2754)**

Current:

```python
            seed = _composer_llm_seed_for_model(self._model)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": _COMPOSER_LLM_TEMPERATURE,
            }
            if seed is not None:
                kwargs[_COMPOSER_LLM_SEED_PARAM] = seed
```

New:

```python
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
            }
            if self._settings.composer_temperature is not None:
                kwargs["temperature"] = self._settings.composer_temperature
            if self._settings.composer_seed is not None:
                kwargs[_COMPOSER_LLM_SEED_PARAM] = self._settings.composer_seed
```

- [ ] **Step 4: Run test**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_text_llm_temperature.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_text_llm_temperature.py
git commit -m "feat(composer): diagnostics text path sources temperature/seed from settings"
```

---

## Phase 6 — Guided chain solver (param-threaded)

`solve_chain` (chain_solver.py:134) is a free function taking `model: str`. It currently calls `_composer_llm_seed_for_model(model)` (line 181) and injects `_COMPOSER_LLM_TEMPERATURE` (line 188). Add two keyword params and thread them from the caller chain: `routes/composer.py` (has settings) → `_guided_solve_chain.py` wrapper → `solve_chain`.

### Task 6: `solve_chain` accepts `temperature`/`seed`; callers thread settings

**Files:**
- Modify: `src/elspeth/web/composer/guided/chain_solver.py` (signature ~134, kwargs ~181-191, audit record temperature)
- Modify: `src/elspeth/web/sessions/_guided_solve_chain.py` (wrapper signature + pass-through)
- Modify: `src/elspeth/web/sessions/routes/composer.py` (pass `settings.composer_temperature`/`composer_seed`)
- Test: `tests/unit/web/composer/guided/test_chain_solver_temperature.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import pytest
from elspeth.web.composer.guided import chain_solver


@pytest.mark.asyncio
async def test_solve_chain_omits_temperature_when_none(monkeypatch):
    captured = {}
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        # minimal valid emit_turn/propose_chain tool-call response
        import json
        tc = type("TC", (), {"function": type("F", (), {
            "name": "emit_turn",
            "arguments": json.dumps({"turn_type": "propose_chain",
                                     "payload": {"steps": [], "why": "x"}}),
        })()})()
        msg = type("M", (), {"tool_calls": [tc], "content": None})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
    monkeypatch.setattr(chain_solver, "_litellm_acompletion", fake_acompletion)
    await chain_solver.solve_chain(model="gpt-5", messages=[{"role": "user", "content": "hi"}],
                                   temperature=None, seed=None)
    assert "temperature" not in captured
    assert "seed" not in captured


@pytest.mark.asyncio
async def test_solve_chain_sends_configured_sampling(monkeypatch):
    captured = {}
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        import json
        tc = type("TC", (), {"function": type("F", (), {
            "name": "emit_turn",
            "arguments": json.dumps({"turn_type": "propose_chain",
                                     "payload": {"steps": [], "why": "x"}}),
        })()})()
        msg = type("M", (), {"tool_calls": [tc], "content": None})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
    monkeypatch.setattr(chain_solver, "_litellm_acompletion", fake_acompletion)
    await chain_solver.solve_chain(model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
                                   temperature=0.0, seed=42)
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 42
```

> Match the real `solve_chain` keyword arguments (read `sed -n '134,160p' src/elspeth/web/composer/guided/chain_solver.py`). If it takes `messages` under a different name, use that.

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_chain_solver_temperature.py -v`
Expected: FAIL — `solve_chain` has no `temperature`/`seed` params and uses the probe.

- [ ] **Step 3: Add params + rewrite kwargs in `solve_chain`**

Add to the `solve_chain` keyword-only signature (after `recorder`):

```python
    temperature: float | None = None,
    seed: int | None = None,
```

Replace the kwargs block (chain_solver.py:181-191). Current:

```python
    seed = _composer_llm_seed_for_model(model)
    kwargs = {
        "model": model,
        "messages": messages,
        "tools": _GUIDED_LLM_TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "emit_turn"}},
        "temperature": _COMPOSER_LLM_TEMPERATURE,
    }
    if seed is not None:
        kwargs["seed"] = seed
```

New:

```python
    kwargs = {
        "model": model,
        "messages": messages,
        "tools": _GUIDED_LLM_TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "emit_turn"}},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
```

Change the `build_llm_call_record(... temperature=_COMPOSER_LLM_TEMPERATURE, seed=seed, ...)` in the `finally` to `temperature=temperature, seed=seed,`.

Remove `_COMPOSER_LLM_TEMPERATURE` and `_composer_llm_seed_for_model` from the `from elspeth.web.composer.service import (...)` block at the top of `chain_solver.py` (leave `_litellm_acompletion`).

- [ ] **Step 4: Thread through the wrapper and route**

In `src/elspeth/web/sessions/_guided_solve_chain.py`: add `temperature: float | None` and `seed: int | None` keyword params to the wrapper that calls `solve_chain`, and pass them through.

In `src/elspeth/web/sessions/routes/composer.py`: at the `solve_chain_with_auto_drop(...)` call site, pass `temperature=settings.composer_temperature, seed=settings.composer_seed` (the route holds `settings`). Locate with `grep -n "solve_chain_with_auto_drop\|settings" src/elspeth/web/sessions/routes/composer.py`.

- [ ] **Step 5: Run unit + integration chain tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/ -k chain -v`
Run: `.venv/bin/python -m pytest tests/integration/web/composer/guided/test_chain_solver.py -v`
Expected: PASS. The integration MALFORMED_RESPONSE cases must stay green (the parse/except structure is unchanged by this task).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/guided/chain_solver.py src/elspeth/web/sessions/_guided_solve_chain.py src/elspeth/web/sessions/routes/composer.py tests/unit/web/composer/guided/test_chain_solver_temperature.py
git commit -m "feat(composer/guided): solve_chain takes operator-set temperature/seed"
```

---

## Phase 7 — Guided chat solver (param-threaded)

Both `maybe_resolve_step_1_source_chat` (chat_solver.py:197) and `solve_step_chat` (chat_solver.py:241) take `model: str`, call the seed probe, and inject `_COMPOSER_LLM_TEMPERATURE`. Same treatment as Phase 6.

### Task 7: chat_solver functions accept `temperature`/`seed`; callers thread settings

**Files:**
- Modify: `src/elspeth/web/composer/guided/chat_solver.py` (both signatures + both kwargs blocks)
- Modify: `src/elspeth/web/sessions/_guided_step_chat.py` (wrapper pass-through)
- Modify: `src/elspeth/web/sessions/routes/composer.py` (pass settings values)
- Test: `tests/unit/web/composer/guided/test_chat_solver_temperature.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import pytest
from elspeth.web.composer.guided import chat_solver


@pytest.mark.asyncio
async def test_solve_step_chat_omits_temperature_when_none(monkeypatch):
    captured = {}
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"content": "reply"})()})()]
        return _R()
    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)
    out = await chat_solver.solve_step_chat(model="gpt-5", step=..., user_message="hi",
                                            temperature=None, seed=None)
    assert "temperature" not in captured
    assert out == "reply"
```

> Fill `step=...` from the real `solve_step_chat` signature (`sed -n '241,260p' src/elspeth/web/composer/guided/chat_solver.py`) — it takes a `GuidedStep`. Add an equivalent test for `maybe_resolve_step_1_source_chat`.

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_chat_solver_temperature.py -v`
Expected: FAIL — no `temperature`/`seed` params.

- [ ] **Step 3: Add params + rewrite both kwargs blocks**

For **each** of `maybe_resolve_step_1_source_chat` and `solve_step_chat`, add to the keyword-only signature:

```python
    temperature: float | None = None,
    seed: int | None = None,
```

Replace each kwargs block — remove `seed = _composer_llm_seed_for_model(model)` and the `"temperature": _COMPOSER_LLM_TEMPERATURE` literal; build kwargs without them and append conditionally:

```python
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
```

Remove `_COMPOSER_LLM_TEMPERATURE` and `_composer_llm_seed_for_model` from the `from elspeth.web.composer.service import (...)` block at the top of `chat_solver.py`.

- [ ] **Step 4: Thread through wrapper + route**

In `_guided_step_chat.py`: add `temperature`/`seed` keyword params to the wrapper(s) and pass through to both functions. In `routes/composer.py`: pass `temperature=settings.composer_temperature, seed=settings.composer_seed` at the chat-solve call sites.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/ -k chat -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/guided/chat_solver.py src/elspeth/web/sessions/_guided_step_chat.py src/elspeth/web/sessions/routes/composer.py tests/unit/web/composer/guided/test_chat_solver_temperature.py
git commit -m "feat(composer/guided): chat_solver functions take operator-set temperature/seed"
```

---

## Phase 8 — Auto-title (param-threaded)

`maybe_auto_title_session` (_auto_title.py:116) calls `_composer_llm_seed_for_model(model)` and injects `_AUTO_TITLE_TEMPERATURE`. The caller is `web/sessions/routes/messages.py:235`, which has `settings` (line 107) and already passes `model=settings.composer_model`.

### Task 8: `maybe_auto_title_session` accepts `temperature`/`seed`; route threads settings

**Files:**
- Modify: `src/elspeth/web/sessions/_auto_title.py` (signature, kwargs ~107-130, remove `_AUTO_TITLE_TEMPERATURE`)
- Modify: `src/elspeth/web/sessions/routes/messages.py:235-240` (pass settings values)
- Test: `tests/unit/web/sessions/test_auto_title_temperature.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import pytest
import elspeth.web.sessions._auto_title as at


@pytest.mark.asyncio
async def test_auto_title_omits_temperature_when_none(monkeypatch):
    captured = {}
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        class _R:
            choices = [type("C", (), {"message": type("M", (), {"content": "My Title"})()})()]
        return _R()
    monkeypatch.setattr(at, "_litellm_acompletion", fake_acompletion)
    titled = {}
    class _Svc:
        async def update_session_title(self, sid, title): titled["t"] = title
    import uuid
    await at.maybe_auto_title_session(service=_Svc(), session_id=uuid.uuid4(),
                                      user_message="hello", model="gpt-5",
                                      temperature=None, seed=None)
    assert "temperature" not in captured
    assert titled["t"] == "My Title"
```

> `_litellm_acompletion` is imported into `_auto_title`'s namespace, so patch `at._litellm_acompletion` (its own module global). Confirm the import stays after this task.

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_auto_title_temperature.py -v`
Expected: FAIL — `maybe_auto_title_session` has no `temperature`/`seed` params.

- [ ] **Step 3: Add params + rewrite kwargs**

Add to the `maybe_auto_title_session` keyword signature:

```python
    temperature: float | None = None,
    seed: int | None = None,
```

Replace the kwargs block (_auto_title.py ~118-128). Remove `seed = _composer_llm_seed_for_model(model)` and the `"temperature": _AUTO_TITLE_TEMPERATURE` literal; build kwargs (keeping `max_tokens`/`max_completion`), then:

```python
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
```

Delete `_AUTO_TITLE_TEMPERATURE = 0.0` (_auto_title.py:69). Remove `_composer_llm_seed_for_model` from the `from elspeth.web.composer.service import (...)` import (keep `_litellm_acompletion`).

- [ ] **Step 4: Thread from the route (messages.py:235)**

In the `maybe_auto_title_session(...)` call, after `model=settings.composer_model,` add:

```python
                            temperature=settings.composer_temperature,
                            seed=settings.composer_seed,
```

- [ ] **Step 5: Run test**

Run: `.venv/bin/python -m pytest tests/unit/web/sessions/test_auto_title_temperature.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/sessions/_auto_title.py src/elspeth/web/sessions/routes/messages.py tests/unit/web/sessions/test_auto_title_temperature.py
git commit -m "feat(composer): auto-title takes operator-set temperature/seed"
```

---

## Phase 9 — Delete the inference helpers and constants

All call sites now source sampling from settings/params, so the probe and constants are dead.

### Task 9: Remove the seed probe, capability probe, and sampling constants

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (delete lines 154-155 constants, 308-319 helpers)
- Test: `tests/unit/web/composer/test_no_sampling_inference.py` (create)

- [ ] **Step 1: Write the failing test (guards the deletion)**

```python
import elspeth.web.composer.service as svc


def test_inference_helpers_are_gone():
    assert not hasattr(svc, "_composer_llm_seed_for_model")
    assert not hasattr(svc, "_litellm_completion_supports_param")
    assert not hasattr(svc, "_COMPOSER_LLM_TEMPERATURE")
    assert not hasattr(svc, "_COMPOSER_LLM_SEED")
```

> `hasattr` is used here only as a test assertion about module attributes — not as defensive production code — so it is fine in test scope.

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_no_sampling_inference.py -v`
Expected: FAIL — the symbols still exist.

- [ ] **Step 3: Delete the helpers and constants**

In `service.py`, delete:
- `_COMPOSER_LLM_TEMPERATURE: Final[float] = 0.0` (line 154)
- `_COMPOSER_LLM_SEED: Final[int] = 42` (line 155)
- `_litellm_completion_supports_param` (lines 308-313)
- `_composer_llm_seed_for_model` (lines 316-319)
- Replace the now-stale sampling-constants comment block (lines ~141-153) with a one-line pointer: `# Composer LLM sampling is operator-set via WebSettings.composer_temperature / composer_seed (sent verbatim, omitted when None). See the spec + ADR.`

Keep `_COMPOSER_LLM_SEED_PARAM: Final[str] = "seed"` (still the kwarg key in `_call_llm`).

- [ ] **Step 4: Find any stragglers**

Run: `grep -rn "_COMPOSER_LLM_TEMPERATURE\|_COMPOSER_LLM_SEED\b\|_composer_llm_seed_for_model\|_litellm_completion_supports_param\|_AUTO_TITLE_TEMPERATURE" src/elspeth/`
Expected: **no output**. Fix any remaining reference before proceeding.

- [ ] **Step 5: Run tests + mypy**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_no_sampling_inference.py -v`
Run: `.venv/bin/python -m mypy src/elspeth/web/composer/service.py`
Expected: PASS / clean.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_no_sampling_inference.py
git commit -m "refactor(composer): delete per-model seed probe and hardcoded sampling constants"
```

---

## Phase 10 — Boot probe + telemetry event

### Task 10: Add `ComposerBootConfigError`, the boot probe, and the `composer.boot_config` event

**Files:**
- Create: probe + error in a focused module `src/elspeth/web/composer/boot_probe.py`
- Modify: `src/elspeth/web/app.py` (lifespan, after the catalog prime ~line 384)
- Test: `tests/unit/web/composer/test_boot_probe.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import elspeth.web.composer.boot_probe as bp


@pytest.mark.asyncio
async def test_probe_raises_boot_config_error_on_temperature_rejection(monkeypatch):
    from litellm.exceptions import BadRequestError
    async def fake_acompletion(**kwargs):
        raise BadRequestError(
            message="litellm.UnsupportedParamsError: gpt-5 doesn't support temperature=0, only the default (1)",
            model="gpt-5", llm_provider="openai")
    monkeypatch.setattr(bp, "_litellm_acompletion", fake_acompletion)
    with pytest.raises(bp.ComposerBootConfigError) as ei:
        await bp.probe_composer_config(model="gpt-5", temperature=0.0, seed=None)
    assert "gpt-5" in str(ei.value) and "temperature" in str(ei.value).lower()


@pytest.mark.asyncio
async def test_probe_passes_through_on_success(monkeypatch):
    async def fake_acompletion(**kwargs):
        class _R: choices = [object()]
        return _R()
    monkeypatch.setattr(bp, "_litellm_acompletion", fake_acompletion)
    await bp.probe_composer_config(model="gpt-4o", temperature=0.0, seed=42)  # no raise


@pytest.mark.asyncio
async def test_probe_is_graceful_on_transient(monkeypatch):
    import httpx
    async def fake_acompletion(**kwargs):
        raise httpx.ConnectError("boom")
    monkeypatch.setattr(bp, "_litellm_acompletion", fake_acompletion)
    # transient failures return False (caller warns + boots), never raise
    result = await bp.probe_composer_config(model="gpt-4o", temperature=0.0, seed=42)
    assert result is False
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_boot_probe.py -v`
Expected: FAIL — module `boot_probe` does not exist.

- [ ] **Step 3: Implement `boot_probe.py`**

```python
"""Boot-time validation of operator-set composer sampling config.

The composer sends operator-set temperature/seed verbatim (no per-model
inference). A reasoning model rejecting the configured value is the operator's
fixable config error — this probe surfaces it at boot rather than on the first
user request. Transient/auth/network failures are NOT fatal (mirrors the
graceful OpenRouter catalog probe in app.py): the app must still boot for
non-LLM features.
"""

from __future__ import annotations

from elspeth.web.composer.service import _litellm_acompletion

# Phrases (with "temperature" present) that identify a provider rejecting the
# configured sampling value. Matched case-insensitively against the 400 body.
_CONFIG_REJECTION_PHRASES = ("only the default", "unsupported value", "unsupportedparams")


class ComposerBootConfigError(RuntimeError):
    """The configured composer sampling was rejected by the provider at boot."""


def _is_config_rejection(exc: BaseException) -> bool:
    if type(exc).__name__ == "UnsupportedParamsError":
        return True
    message = str(exc).lower()
    if "temperature" not in message and "seed" not in message:
        return False
    return any(p in message for p in _CONFIG_REJECTION_PHRASES)


async def probe_composer_config(*, model: str, temperature: float | None, seed: int | None) -> bool:
    """Probe ``model`` with the configured sampling. Return True on success,
    False on a transient failure (caller warns + boots). Raise
    ``ComposerBootConfigError`` on a provider rejection of the configured
    sampling (fatal — the operator's config error)."""
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    kwargs: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    try:
        await _litellm_acompletion(**kwargs)
        return True
    except LiteLLMBadRequestError as exc:
        if _is_config_rejection(exc):
            raise ComposerBootConfigError(
                f"composer sampling rejected by {model}: temperature={temperature}, "
                f"seed={seed} — {exc}"
            ) from exc
        return False
    except Exception:
        # Transient / auth / network: graceful, mirroring the catalog probe.
        return False
```

- [ ] **Step 4: Run probe tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_boot_probe.py -v`
Expected: PASS.

- [ ] **Step 5: Wire the probe + telemetry event into the lifespan**

In `src/elspeth/web/app.py`, after the catalog-prime block (~line 384), add the module-level OTel counter near the other boot counters and the probe call inside `lifespan` (guard on the flag):

```python
# module level, near the other composer counters
_COMPOSER_BOOT_CONFIG_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.boot_config",
    description="Composer effective sampling config recorded at boot",
)
```

```python
    # inside lifespan, after catalog prime:
    if settings.composer_boot_probe_enabled:
        from elspeth.web.composer.boot_probe import probe_composer_config
        probe_models = [settings.composer_model]
        if settings.composer_advisor_enabled:
            probe_models.append(settings.composer_advisor_model)
        for _m in probe_models:
            ok = await probe_composer_config(
                model=_m,
                temperature=settings.composer_temperature,
                seed=settings.composer_seed,
            )  # raises ComposerBootConfigError on a config rejection -> boot fails
            if not ok:
                slog.warning(
                    "composer_boot_probe_transient_failure",
                    model=_m,
                    action="booting; composer LLM calls will be exercised at first use",
                )
        _COMPOSER_BOOT_CONFIG_COUNTER.add(1, {
            "composer_model": settings.composer_model,
            "temperature": str(settings.composer_temperature),
            "seed": str(settings.composer_seed),
            "advisor_model": settings.composer_advisor_model if settings.composer_advisor_enabled else "disabled",
        })
```

> `ComposerBootConfigError` is intentionally NOT caught in the lifespan — it propagates and aborts startup, which is the fail-fast. Confirm `slog` and `metrics` are already imported in `app.py` (they are — used by the catalog probe and the module counters).

- [ ] **Step 6: Write the lifespan integration test**

```python
# tests/unit/web/test_app_boot_probe.py
import pytest


@pytest.mark.asyncio
async def test_lifespan_aborts_on_composer_config_rejection(monkeypatch):
    # Build the app with composer_boot_probe_enabled=True and a fake transport
    # that rejects the configured temperature; entering the lifespan must raise
    # ComposerBootConfigError. Reuse the app-construction helper in the existing
    # app/boot tests: grep -rln "lifespan(\|create_app(" tests/unit/web/.
    ...


@pytest.mark.asyncio
async def test_lifespan_skips_probe_when_disabled(monkeypatch):
    # composer_boot_probe_enabled=False -> _litellm_acompletion is never called
    # during boot.
    ...
```

> Fill the `...` from the existing app-boot test harness (named in the comment). The two behaviours to assert: rejection → raises during lifespan entry; disabled → transport never called at boot.

- [ ] **Step 7: Run + commit**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_boot_probe.py tests/unit/web/test_app_boot_probe.py -v`
Expected: PASS.

```bash
git add src/elspeth/web/composer/boot_probe.py src/elspeth/web/app.py tests/unit/web/composer/test_boot_probe.py tests/unit/web/test_app_boot_probe.py
git commit -m "feat(composer): boot probe validates operator sampling config (fatal on rejection) + boot telemetry event"
```

---

## Phase 11 — ADR

### Task 11: Record the configurability decision

**Files:**
- Create: `docs/adr/NNNN-composer-operator-set-sampling.md` (next number — run `ls docs/adr/ | sort | tail -3`)

- [ ] **Step 1: Write the ADR**

Use the project's existing ADR template (copy the front-matter from the most recent `docs/adr/*.md`). Content:
- **Context:** composer sampling was hardcoded behind a "Tier 2 — needs an ADR to make configurable" guard; the default model `gpt-5.5` is a reasoning model that rejects non-default temperature, which forced a reactive-retry apparatus and a per-model seed probe.
- **Decision:** make `composer_temperature` / `composer_seed` operator-set `WebSettings` fields, default `None` (omit), sent verbatim; delete the inference/retry machinery; validate at boot (fatal on config-rejection only); emit `composer.boot_config` telemetry.
- **Consequences:** out-of-box runs at provider-default sampling (RGR §4.4's `0.0` finding no longer applies to the reasoning-model default); operator owns sampling validity; a misconfigured value fails boot with a precise message; per-call audit rows record the configured value (incl. `None`).

- [ ] **Step 2: Commit**

```bash
git add docs/adr/NNNN-composer-operator-set-sampling.md
git commit --no-verify -m "docs(adr): operator-set composer sampling config"
```

---

## Phase 12 — Full verification

### Task 12: Suites, types, lints, contracts

- [ ] **Step 1: Composer + sessions + contracts suites**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ tests/unit/web/sessions/ tests/unit/contracts/ tests/unit/web/test_config.py -q`
Expected: PASS.

- [ ] **Step 2: Integration (guided + boot)**

Run: `.venv/bin/python -m pytest tests/integration/web/composer/ -q`
Expected: PASS.

- [ ] **Step 3: mypy + ruff**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/ src/elspeth/web/sessions/_auto_title.py src/elspeth/web/config.py src/elspeth/web/app.py src/elspeth/contracts/composer_llm_audit.py`
Run: `.venv/bin/python -m ruff check src/elspeth/web/composer/ src/elspeth/web/sessions/ src/elspeth/web/config.py src/elspeth/web/app.py`
Expected: clean.

- [ ] **Step 4: Tier-model lint (no new upward imports)**

Run: `env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth`
Expected: no new violations (all new code is L3 `web/`; `boot_probe.py` imports only from `web/composer/service` — intra-L3).

- [ ] **Step 5: Config contracts**

Run: `.venv/bin/python -m scripts.check_contracts`
Expected: green — the two new `WebSettings` fields are read directly off `self._settings` (no runtime-config-protocol contract), so no contract drift. If `check_contracts` flags them, add them to the composer runtime config protocol per the `config-contracts-guide` skill.

- [ ] **Step 6: Full default suite (CI-equivalent)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: green except the documented pre-existing failures (memory `project_rc52_preexisting_test_failures_2026-05-29`).

- [ ] **Step 7: Final commit (if any test-expectation updates remain)**

```bash
git add -A
git commit -m "test(composer): align suite with operator-set sampling config"
```

---

## Self-Review

**Spec coverage:**
- Config surface (`composer_temperature`/`composer_seed`, default `None`, `Field` bounds) → Task 1. ✓
- Sent verbatim / omitted when `None`, all six call sites → Tasks 3 (main), 4 (advisor), 5 (text), 6 (chain), 7 (chat ×2), 8 (auto-title). ✓
- Deletions: reactive-retry feature (Task 0 revert), seed probe + capability probe + constants (Task 9). ✓
- Kept: `ComposerLLMCall.temperature: float | None` + builder param → Task 2. ✓
- Boot probe, config-rejection-only fatal, graceful transient, `composer_boot_probe_enabled` flag → Task 10. ✓
- `composer.boot_config` telemetry event → Task 10. ✓
- Per-call audit records configured value incl. `None` → Tasks 3-8 (audit-record temperature swapped to settings value). ✓
- ADR → Task 11. ✓
- Trust tier (Pydantic `Field` validation = boundary) → Task 1 bounds. ✓

**Placeholder scan:** Test bodies with `...` (Tasks 4, 7, 10 Step 6) are deliberate "read-then-fill" anchors — each names the existing test module/harness to read first for the exact fixture/construction shape. The executor MUST replace every `...` with concrete values before running. No production-code step contains a placeholder.

**Type consistency:** `temperature: float | None` / `seed: int | None` used consistently across `WebSettings` (Task 1), the DTO + builder (Task 2), the service methods reading `self._settings.composer_temperature/seed` (Tasks 3-5), the guided/auto-title params (Tasks 6-8), and `probe_composer_config(*, model, temperature, seed)` (Task 10). `_COMPOSER_LLM_SEED_PARAM = "seed"` retained as the kwarg key; `_COMPOSER_LLM_TEMPERATURE`/`_COMPOSER_LLM_SEED`/`_AUTO_TITLE_TEMPERATURE` deleted (Task 9) only after all readers are migrated (Tasks 3-8).

**Ordering safety:** Phase 2 (nullable audit) precedes Phases 3-8 (which pass `None` to the audit record). Phase 9 (delete helpers/constants) follows all call-site migrations. Phase 0 (revert) precedes everything and is the clean baseline.

**Known executor caveats:**
1. `litellm.exceptions.BadRequestError(message=..., model=..., llm_provider=...)` — confirm the constructor against litellm 1.85.0 at execution.
2. The `composer_service` / `composer_service_with` fixtures (Tasks 3-5) must be built from the existing `ComposerServiceImpl` construction in `test_compose_loop_llm_audit.py`. Build once, reuse.
3. Existing compose-loop audit tests assert `temperature == 0.0`; with the `None` default they must either set `composer_temperature=0.0` on their settings fixture or assert `None` (Task 3 Step 5). This is the "locked-in expectation" update, not a regression.
4. The guided threading chain (route → wrapper → solver) in Tasks 6-7 spans `routes/composer.py`, `_guided_solve_chain.py`, `_guided_step_chat.py` — grep each for the call site before editing; the wrappers may have auto-drop variants (`*_with_auto_drop`) that also need the params.
