# Design: Composer temperature-rejection handling (transport-agnostic)

**Date:** 2026-05-31
**Status:** Approved (design, revised); pending implementation plan
**Scope:** Composer + auto-title LLM call paths only. Pipeline LLM-transform audit
path `contracts/call_data.py` is out of scope (§8).

> **Revision note (2026-05-31):** The original design used a two-layer approach
> whose Layer 1 asked litellm's model-capability DB (`get_optional_params`,
> `drop_params=True`) which sampling params a model would accept. The project has
> since decided to **remove litellm** from the composer (the rest of the product
> already talks to OpenRouter via raw httpx — see §10). Layer 1 was the only part
> coupled to litellm and would be demolished by that removal, so it is **dropped**.
> The reactive retry (former Layer 2) becomes the **primary, transport-agnostic
> mechanism** and survives the litellm removal with only a one-predicate change.

## 1. Problem

On staging, composer/tutorial requests failed at the first outbound LLM call when
the composer model was a GPT-5-family model behind an Azure-hosted OpenAI-compatible
endpoint. The UI symptom was a silent return to the same screen; the backend returned
502 with `LLM request rejected (UnsupportedParamsError)`.

The composer hardcodes `temperature=0.0` and sends it on **every** outbound call
(`_call_llm`, `_call_text_llm`, advisor, the guided chat/chain solvers, and the
auto-title path). GPT-5-family and o-series ("reasoning") models reject a non-default
temperature: only `1` or omission is accepted.

### 1.1 Verified failure mechanism (first-principles, against installed litellm 1.85.0)

The failure is a **value constraint**, not a missing-parameter problem:

| Model | `temperature` in `get_supported_openai_params` | `get_optional_params(temperature=0.0, drop_params=False)` |
|-------|---|---|
| `o1`, `o3-mini` | True | **raises `UnsupportedParamsError`** ("O-series models don't support…") |
| `gpt-5`, `azure/gpt-5` | True | **raises `UnsupportedParamsError`** ("gpt-5 models … don't support temperature=0") |
| `gpt-4o` | True | no raise, sends `temperature=0.0` |

Key consequences (these are *evidence about the failure*, not the basis of the fix):

- `UnsupportedParamsError` **is a subclass of** `litellm.exceptions.BadRequestError`,
  so the composer's existing `except LiteLLMBadRequestError` catches it and renders
  exactly `LLM request rejected (UnsupportedParamsError)`. With litellm, the error is
  raised **client-side**; without litellm (talking to OpenRouter/OpenAI directly) the
  equivalent rejection arrives as an HTTP **400** from the provider.
- Because the constraint is on the **value**, a guard keyed on parameter *presence*
  would not catch recognized reasoning models. And building on litellm's value-constraint
  DB couples the fix to a dependency we are removing. The robust, durable approach is to
  **react to the provider's actual rejection**, not to predict it from a model registry.

### 1.2 Current state in the repository (root-cause confirmation)

- `_COMPOSER_LLM_TEMPERATURE: Final[float] = 0.0` and `_AUTO_TITLE_TEMPERATURE = 0.0`
  are sent unconditionally. No `drop_params`/`modify_params`/`LITELLM_DROP_PARAMS` is set
  anywhere in `src/`, `config/`, `packs/`, any yaml, or the systemd unit — litellm's
  default `drop_params=False` applies, so the bug manifests.
- `seed` already has a model-aware guard (`_composer_llm_seed_for_model`); `temperature`
  does not. The deeper root cause is this inconsistency, but the durable fix routes
  temperature through reaction-to-rejection rather than adding a second predictive guard.
- No commit on any branch makes temperature model-aware. The "fix" referenced in the
  original report existed only on a temporary demo instance (an uncommitted "works on my
  machine" hand-edit) and caused a failed demo. The defect is live in committed code,
  currently *masked* only by staging being configured for a different model
  (`openrouter/openai/gpt-5.4-mini`).

## 2. Goals / non-goals

**Goals**
- Composer and auto-title LLM calls succeed against reasoning models that constrain
  `temperature`, without losing determinism for models that accept `0.0`.
- The audit row records the sampling parameters **actually sent** (audit mirrors request).
- The fix is **transport-agnostic**: correct whether the composer calls litellm today or
  raw OpenRouter HTTP after the litellm removal, with at most a one-predicate change.

**Non-goals**
- Making temperature configurable via settings/env (remains Tier 2; would need an ADR).
- Changing the deterministic intent: `0.0` stays the target value wherever the provider
  accepts it (preserves the RGR 2026-05-06 §4.4 hard-GREEN rationale).
- Reimplementing a model-capability database. We deliberately do **not** predict which
  models reject temperature; we react to actual rejections.
- The pipeline LLM-transform audit path (`contracts/call_data.py`). Off the composer path
  (§8).
- The litellm removal itself (separate follow-up, §10).

## 3. Architecture — reactive retry (primary), with an observed-rejection cache

### 3.1 Reactive retry

Each composer LLM call site sends `temperature=_COMPOSER_LLM_TEMPERATURE` as today.
A shared helper wraps the call:

1. Issue the request with temperature included.
2. If the call fails with a **temperature-rejection signal**, retry **once** with
   `temperature` omitted (other params unchanged).
3. Record honestly — two audit rows:
   - failed attempt → `ComposerLLMCall` (`status=error`, `temperature=0.0`,
     `error_class=<rejection class>`);
   - successful retry → `ComposerLLMCall` (`status=success`, `temperature=None`).

The retry is **once only**; a second failure surfaces normally (no loop). Any rejection
that is **not** a temperature-rejection surfaces unchanged as `_BadRequestLLMError` today.

The helper returns both the response **and** the sampling actually used, so each caller
builds its audit record from the true sent shape (audit fidelity preserved — the retry is
not hidden inside a transport wrapper that cannot signal back what it sent).

### 3.2 Rejection-signal predicate (the one transport-coupled point)

Detection is isolated behind a single predicate
`_is_temperature_rejection(exc) -> bool`:

- **Today (litellm transport):** matches `litellm.exceptions.UnsupportedParamsError`
  (and, defensively, a `BadRequestError` whose message indicates a temperature/value
  constraint). Caught **before** the broader `BadRequestError` handling.
- **After the litellm removal:** the predicate matches the provider's HTTP-400 error body
  signature for an unsupported temperature value. This is the **only** part of the fix
  that changes when the transport changes — the retry/audit machinery is unchanged.

Keeping detection in one named predicate is what makes the fix durable across the
transport swap and keeps the coupling auditable.

### 3.3 Observed-rejection cache (optimization)

To avoid paying a rejected round-trip on every first turn against a reasoning model, a
**process-level** cache records model strings observed to reject temperature. On a cache
hit, the call omits `temperature` up front (and records `temperature=None`). The cache is:

- transport-agnostic (no model DB; populated only by *observed* rejections);
- process-scoped (not persisted) — a conservative, self-correcting heuristic, never a
  source of audited claims beyond what was actually sent;
- safe to start empty on every boot.

This is an optimization, not a correctness requirement; it may be deferred if the plan
prefers the smallest first landing.

### 3.4 Audit field

`ComposerLLMCall.temperature` becomes nullable (`float | None`), mirroring
`seed: int | None`, so an omitted temperature is recorded as `None`.

## 4. Components touched (composer-only)

| File | Change |
|------|--------|
| `contracts/composer_llm_audit.py` | `ComposerLLMCall.temperature: float` → `float \| None`; update docstring (L73–82): temperature is `0.0` when sent, `None` when omitted after a temperature rejection. |
| `composer/llm_response_parsing.py` | `build_llm_call_record(temperature: float)` → `float \| None` (single audit-construction point). |
| `composer/service.py` | Add the retry helper + `_is_temperature_rejection` predicate + (optional) observed-rejection cache; rewire `_call_llm`, `_call_text_llm`, advisor; update comment block L140–153. `seed` handling is left as-is for now (it is not failing and is part of the litellm-removal follow-up). |
| `composer/guided/chat_solver.py`, `composer/guided/chain_solver.py` | Route calls through the shared retry helper. |
| `sessions/_auto_title.py` | Same; `_AUTO_TITLE_TEMPERATURE` flows through the retry helper. |

## 5. Persistence / audit (no migration required)

`ComposerLLMCall` is persisted as a **JSON envelope** in the `tool_calls` JSON column
(`composer/audit.py:llm_call_audit_envelope`), canonicalized with rfc8785/JCS — not as a
typed `NOT NULL` SQL column. `seed: int | None` already serializes `None → null` through
this exact path. Making `temperature` nullable needs **no schema migration and no staging
DB delete**.

## 6. Error handling summary

- Predicate `_is_temperature_rejection` is checked **before** the broader `BadRequestError`
  path; matches retry, non-matches surface unchanged.
- Retry is once only; a second failure surfaces normally.
- LLM transport calls remain system-code (no defensive swallowing) except the targeted
  temperature-rejection catch — a known provider value-constraint at the trust boundary,
  which CLAUDE.md permits.
- Empty-choices Tier-3 guard (`_MalformedLLMResponseError`) is unchanged.

## 7. Testing

- **Predicate unit:** `_is_temperature_rejection` returns True for a litellm
  `UnsupportedParamsError` (and a temperature-signature `BadRequestError`), False for an
  unrelated `BadRequestError`.
- **Retry behavior:** first call raises a temperature rejection → exactly one retry with
  `temperature` omitted → two `ComposerLLMCall` rows with correct `temperature`/`status`.
  Non-temperature `BadRequestError` → no retry, surfaces as `_BadRequestLLMError`.
- **Audit fidelity:** omitted temperature → `ComposerLLMCall.temperature is None` → JCS
  envelope emits `null`.
- **Cache (if included):** second call to a model previously observed to reject omits
  temperature up front (single round-trip, `temperature=None` recorded).
- Integration tests use production code paths (the `_call_llm` mocking seam exists) per
  CLAUDE.md.

## 8. Scope boundary check

`contracts/call_data.py` (its own non-nullable `temperature: float`) is the **pipeline**
LLM-call audit, not referenced by `composer/` or `sessions/_auto_title.py` (verified). Out
of scope; surfaced as a separate follow-up if the same risk exists there.

## 9. Open implementation details (for the plan, not the design)

- Exact extraction shape of the retry helper across the 5 call sites (and whether the
  advisor path, which already wraps in `asyncio.wait_for`, composes cleanly).
- Whether to include the observed-rejection cache (§3.3) in the first landing or defer.
- The precise litellm error-message signature(s) to match in the defensive
  `BadRequestError` arm of the predicate.

## 10. Sequencing context — litellm removal (separate follow-up)

litellm is composer/web-scoped: 16 `web/` references vs 3 in `plugins/` (the only plugin
importer is `transforms/llm/model_catalog.py`, used for model *listing*, not inference).
The pipeline LLM inference plugin (`transforms/llm/providers/openrouter.py`) already uses
**raw httpx** to OpenRouter. Removing litellm from the composer therefore aligns it with
the existing product transport; the reimplementation surface (token-usage, provider-cost,
reasoning-field, prompt-cache extraction, error mapping, and the `model_catalog` listing)
has an existing httpx/OpenRouter precedent. The model-capability DB is the one thing not
reimplemented — which is exactly why this fix is reactive rather than predictive.

**This temperature fix lands first** (live bug, small, transport-agnostic). The litellm
removal is a larger separate piece that inherits the already-portable retry, changing only
the `_is_temperature_rejection` predicate (§3.2).
