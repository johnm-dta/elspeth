# Composer Operator-Set Sampling Config — Design

**Date:** 2026-06-03
**Status:** Approved design (pre-implementation)
**Supersedes:** `docs/superpowers/specs/2026-05-31-composer-model-aware-temperature-design.md`
(model-aware reactive-retry approach) and abandons its in-flight implementation
plan `docs/superpowers/plans/2026-06-03-composer-temperature-fix.md` plus the
uncommitted reactive-retry work on branch RC5.2.

---

## Problem

The composer hardcodes its LLM sampling parameters as module constants —
`_COMPOSER_LLM_TEMPERATURE = 0.0`, `_COMPOSER_LLM_SEED = 42` — behind a guard
comment: *"Configurability is Tier 2 — do not read from settings/env without an
ADR."* Two consequences followed:

1. **Self-inflicted inference.** The default composer model is `gpt-5.5`
   (`WebSettings.composer_model`), a GPT-5-family *reasoning* model that rejects
   any non-default temperature. Mandating `0.0` against that default forced an
   entire reactive apparatus to cope: a rejection predicate
   (`is_temperature_rejection`), a process-global observed-rejection cache
   (`_TEMPERATURE_REJECTING_MODELS`), and a temperature-omitting retry helper
   triplicated across three modules. We wrote and maintained machinery to avoid
   a config field.

2. **A second inference site.** `_composer_llm_seed_for_model` probes
   `litellm.get_supported_openai_params` per-model to decide whether to send the
   seed — predicting provider capability rather than letting the operator state
   intent.

The operator's directive: stop predicting what models people use and stop
mandating how "hot" they run them. Let the operator set temperature and seed in
config; send them verbatim; validate the configuration against the real provider
once, at boot; and record the effective config as a boot-time fact. The first
time someone asks "why can't I set the temperature," the answer should be "you
can," not a new round of capability inference.

## Goal

Make composer LLM sampling (temperature, seed) operator-configurable, delete all
capability-inference and reactive-retry machinery, validate the operator's config
against the provider at boot (fail fast on a config error), and emit the effective
config as a boot telemetry event.

## Non-Goals

- No separate advisor or auto-title sampling knobs — both share the composer
  values.
- No change to the auto-title `max_tokens` (a task limit, not a sampling
  preference) or to any other `composer_*` budget/timeout field.
- No persistence of the boot config to the Landscape audit trail (operational
  visibility belongs on telemetry; per-call audit rows remain the legal record).

---

## Design

### 1. Config surface

Two new fields on `WebSettings` (`src/elspeth/web/config.py`), beside the existing
`composer_model` / `composer_advisor_model`:

```python
composer_temperature: float | None = Field(default=None, ge=0, le=2)
composer_seed: int | None = None
```

- **Default `None` → omitted from the request.** This is the only default
  coherent with a reasoning-model default (`gpt-5.5`): out of the box we send
  neither parameter, the provider applies its own defaults, and nothing is
  rejected. An operator running a temperature-accepting model who wants
  determinism sets `composer_temperature: 0.0` (and `composer_seed: 42` if their
  model supports it) explicitly.
- **Sent verbatim when set.** No inference, no per-model probe, no reject-retry.
  If a provider rejects a value the operator set, that is the operator's config
  error — surfaced honestly (at boot via the probe; see §3).
- **Shared by all composer LLM paths:** main compose loop, advisor, both
  `chat_solver` functions, the diagnostics text path, and auto-title. Auto-title
  is a free function called from `web/sessions/routes/messages.py`; the route
  threads the two values in (it already holds `WebSettings`).

The RGR §4.4 finding (that `0.0` lifted a ~33% hard-GREEN ceiling) was measured on
an older model that *accepted* `0.0`; it does not apply to the `gpt-5.5` default,
which cannot receive `0.0` at all. Operators on temperature-accepting models can
still opt into `0.0`.

### 2. Deletions

This change is overwhelmingly subtractive.

**The entire uncommitted reactive-retry feature, reverted:**
- `is_temperature_rejection` + `_TEMPERATURE_REJECTION_PHRASES`
- `_TEMPERATURE_REJECTING_MODELS` + `_note_temperature_rejection`
- `_acompletion_with_temperature_retry` — all three copies (`service.py`,
  `guided/chat_solver.py`, `sessions/_auto_title.py`)
- The two-row `_attempt` retry wrappers in `_call_llm_before_deadline`,
  `_call_advisor_with_audit`, and `solve_chain` — these collapse back to a single
  straight-through call+audit; the `_BadRequestLLMError` arm returns to
  "never retry, 400s are not transient"
- All ten new `test_*temperature*` test files (the two *modified* existing test
  files — `test_composer_llm_audit.py`, `test_service.py` — are reverted, not
  deleted)

**Two pre-existing inference helpers (confirmed used nowhere else):**
- `_composer_llm_seed_for_model(model)`
- `_litellm_completion_supports_param(model, param)`

**Three constants, replaced by settings reads:**
- `_COMPOSER_LLM_TEMPERATURE` → `settings.composer_temperature`
- `_COMPOSER_LLM_SEED` → `settings.composer_seed`
- `_AUTO_TITLE_TEMPERATURE` (`_auto_title.py`) → the composer value, threaded from
  the route
- `_COMPOSER_LLM_SEED_PARAM = "seed"` stays as the kwarg-key literal.

**Kept from the uncommitted work:** `ComposerLLMCall.temperature: float | None`
and the matching `build_llm_call_record` parameter. Built to mean "omitted because
rejected," it now honestly means "omitted because the operator did not set it" —
same type, correct new meaning.

**Resulting call-site shape (identical everywhere):**

```python
kwargs = {"model": model, "messages": messages, ...}
if temperature is not None:
    kwargs["temperature"] = temperature
if seed is not None:
    kwargs["seed"] = seed
response = await _litellm_acompletion(**kwargs)
```

### 3. Boot probe + telemetry event

Co-located in the `app.py` lifespan, immediately after the existing OpenRouter
catalog prime probe (`app.py:339`), and gated by a new setting:

```python
composer_boot_probe_enabled: bool = True   # test settings set False
```

When enabled, the probe issues one minimal `_litellm_acompletion` to
`composer_model` (and to `composer_advisor_model` when
`composer_advisor_enabled`), carrying the configured `temperature`/`seed` and
`max_tokens=1` — the cheapest call that exercises the operator's exact sampling
config against the real provider.

**Failure policy — config-rejection only is fatal**, mirroring the existing
catalog probe's deliberate "whose fault is it" split:
- A provider **400 (`BadRequestError`)** raises a typed `ComposerBootConfigError`
  → boot fails, with a message naming the configured parameter values and model
  (e.g. *"composer sampling rejected by gpt-5.5: temperature=0.0, seed=None — only
  the default (1) is supported"*). The discriminator is the **exception class,
  not message prose**: the probe sends a fixed trivial `"ping"` with
  `max_tokens=1`, so the only operator-variable inputs are model/temperature/seed
  — any 400 on that payload is unambiguously a config rejection (and is
  seed-symmetric). This deliberately does **not** re-introduce the prose-matching
  heuristic this redesign deletes. This is the operator's fixable mistake — fail
  fast.
- **Transient / network / auth / 5xx** failures warn and boot, matching the
  existing catalog probe's "staging must still boot for non-LLM features" choice.
  This decouples app availability from provider uptime; a misconfigured sampling
  value is the operator's bug, a provider hiccup is not.

**On success**, emit one structured boot **telemetry** event (`composer.boot_config`)
carrying `{composer_model, composer_temperature, composer_seed,
composer_advisor_model, composer_advisor_enabled, probe_latency_ms}` — the
"record the config at boot" requirement, on the operational-visibility channel per
the audit/telemetry primacy order. The Landscape is not expanded with system
config.

### 4. Per-call audit (mechanism unchanged)

`ComposerLLMCall` rows continue to record the temperature/seed actually sent.
Because those values now come straight from config with no per-model variation,
every row faithfully mirrors the configured regime — one row per call, no two-row
retry pairs. `temperature` reads as `None` when omitted.

### 5. ADR

The removed constant's guard (*"do not read from settings/env without an ADR"*)
is load-bearing: it exists so this change cannot happen silently. The work ships
with a short ADR recording the decision — hardcoded-deterministic → operator-set,
the `gpt-5.5`-default interaction that forces `None` defaults, and the boot-probe
fail-fast that replaces runtime coping — so the next reader learns why the
constants vanished.

### 6. Trust tier

`composer_temperature` / `composer_seed` are operator-authored configuration →
Tier 3. The boundary check is Pydantic `Field` validation at Settings load (range
bounds); the values are then sent verbatim. The provider is the real arbiter of
validity, which the boot probe surfaces deterministically at startup.

---

## Testing

- **Config:** Settings accepts and omits both fields; `Field` bounds reject
  out-of-range temperature; defaults are `None`.
- **Call sites (all six):** `temperature`/`seed` appear in kwargs only when set,
  omitted when `None` — main compose, advisor, `chat_solver` ×2, auto-title, text
  path.
- **Deletions verified:** the inference helpers and reactive-retry symbols are no
  longer importable; existing composer audit tests stay green with
  config-sourced values.
- **Boot probe:** config-rejection 400 → `ComposerBootConfigError` (boot fails,
  message names param/value/model); transient failure → graceful boot + warning;
  `composer_boot_probe_enabled=False` skips the probe entirely; success → the
  `composer.boot_config` telemetry event fires with effective config.
- **Per-call audit:** rows record the configured temperature/seed (including
  `None`).
- **Gates:** mypy, ruff, `trust_tier.tier_model` lint, config-contracts alignment
  (verify the two new fields don't break the Settings→Runtime protocol), full
  default suite.

## Decisions captured

| Question | Decision |
|----------|----------|
| Which knobs become config | `composer_temperature`, `composer_seed` (kill the seed probe) |
| Advisor / auto-title knobs | None — share the composer values |
| Defaults | Both `None` (omit) — forced by the `gpt-5.5` reasoning-model default |
| "Record at boot" channel | Boot telemetry event; per-call audit rows stay the legal record |
| Boot probe failure policy | Fatal on config-rejection only; graceful on transient |
| ADR | Required (removes the Tier-2 configurability guard) |
