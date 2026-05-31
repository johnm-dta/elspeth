# Design: Model-aware composer sampling parameters

**Date:** 2026-05-31
**Status:** Approved (design); pending implementation plan
**Scope:** Composer + auto-title LLM call paths only (pipeline LLM-transform audit
path `contracts/call_data.py` explicitly out of scope — see §8).

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

| Model | `temperature` in `get_supported_openai_params` | `get_optional_params(temperature=0.0, drop_params=False)` | `drop_params=True` |
|-------|---|---|---|
| `o1`, `o3-mini` | **True** | **raises `UnsupportedParamsError`** ("O-series models don't support…") | temperature **dropped** |
| `gpt-5`, `azure/gpt-5` | **True** | **raises `UnsupportedParamsError`** ("gpt-5 models … don't support temperature=0") | temperature **dropped** |
| `gpt-4o` | True | no raise, sends `temperature=0.0` | temperature kept (`0.0`) |

Key consequences:

- `UnsupportedParamsError` **is a subclass of** `litellm.exceptions.BadRequestError`
  (verified), so the composer's existing `except LiteLLMBadRequestError` catches it and
  renders exactly `LLM request rejected (UnsupportedParamsError)`. The error is raised
  **client-side** by litellm (it knows the constraint and `drop_params` is unset), not by
  a network round-trip.
- Because the constraint is on the **value**, a guard keyed on
  `get_supported_openai_params` (param-presence) — the obvious "mirror the seed guard"
  fix — would **not** fix recognized reasoning models: temperature is "supported", so the
  guard would still send `0.0` and still raise. The authority that knows the value
  constraint is litellm's own `drop_params` logic.

### 1.2 Current state in the repository (root-cause confirmation)

- `_COMPOSER_LLM_TEMPERATURE: Final[float] = 0.0` and `_AUTO_TITLE_TEMPERATURE = 0.0`
  are sent unconditionally. No `drop_params`/`modify_params`/`LITELLM_DROP_PARAMS` is set
  anywhere in `src/`, `config/`, `packs/`, any yaml, or the systemd unit — litellm's
  default `drop_params=False` applies, so the bug manifests.
- The `seed` parameter already has a model-aware guard
  (`_composer_llm_seed_for_model` → `_litellm_completion_supports_param`), but
  `temperature` does not. **The deeper root cause is this inconsistency**: one sampling
  parameter is guarded, another is not.
- No commit on any branch makes temperature model-aware. The "fix" referenced in the
  original report existed only on a temporary demo instance (a "works on my machine"
  hand-edit, never committed); it caused a failed demo. The defect is live in committed
  code and is currently *masked* only by staging being configured for a different model
  (`openrouter/openai/gpt-5.4-mini`).

## 2. Goals / non-goals

**Goals**
- Composer and auto-title LLM calls succeed against reasoning models that constrain
  `temperature`, without losing determinism for models that accept `0.0`.
- The audit row records the sampling parameters **actually sent** (audit mirrors request).
- Remove the seed/temperature handling inconsistency so the bug class cannot recur for a
  future sampling parameter.

**Non-goals**
- Making temperature configurable via settings/env (remains Tier 2; would need an ADR).
- Changing the deterministic intent: `0.0` stays the target value wherever the provider
  accepts it (preserves the RGR 2026-05-06 §4.4 hard-GREEN rationale).
- The pipeline LLM-transform audit path (`contracts/call_data.py`). Confirmed off the
  composer path (§8).

## 3. Architecture — two layers

### Layer 1 (preventive): single source of truth for sampling parameters

A new helper in `service.py` computes the **effective** sampling decision for a model
using litellm's own authority:

```
provider = litellm.get_llm_provider(model)            # → (name, custom_llm_provider, …)
effective = litellm.utils.get_optional_params(
    model=name, custom_llm_provider=custom_llm_provider,
    temperature=_COMPOSER_LLM_TEMPERATURE,
    seed=_COMPOSER_LLM_SEED,
    drop_params=True,                                  # ask litellm what it would keep
)
decision = ComposerSamplingDecision(
    temperature=_COMPOSER_LLM_TEMPERATURE if "temperature" in effective else None,
    seed=_COMPOSER_LLM_SEED if "seed" in effective else None,
)
```

- Whatever litellm would **keep** is kept; whatever it would **drop** becomes `None`.
- `get_optional_params` covers **both** the value-constraint case (temperature for
  reasoning models) and the param-presence case (seed for providers without seed support)
  in one uniform mechanism. This **subsumes and replaces** the existing
  `_composer_llm_seed_for_model` / `_litellm_completion_supports_param` pair.
- The single `decision` feeds **both**:
  - the request kwargs — include a key only when its value is not `None`;
  - the audit record — record the value as-is, `None` preserved.
- The real `acompletion` call keeps `drop_params=False`. Anything the computation misses
  then **fails loud** rather than being silently dropped (offensive programming).

`ComposerSamplingDecision` is a small frozen dataclass (`temperature: float | None`,
`seed: int | None`); no container fields, so `frozen=True` suffices (no `deep_freeze`).

Optional: memoize the per-model decision (`functools.lru_cache`) since it is a pure
function of the model string; the current seed helper is uncached, so this is a minor,
non-blocking refinement.

### Layer 2 (reactive): bounded retry for the residual

Layer 1 fixes every model litellm classifies correctly. The **residual** is a custom
deployment alias litellm does *not* recognize as a reasoning model (the model-DB-lag /
"works on my machine" class — e.g. `openrouter/openai/gpt-5.4-mini` if its upstream
rejects `0.0`). There, Layer 1 keeps temperature, the provider rejects it server-side,
and litellm raises `UnsupportedParamsError`.

A small call-site helper:
1. catches **exactly** `UnsupportedParamsError` (caught **before** the broader
   `BadRequestError`);
2. retries **once** with `temperature` omitted (seed unchanged);
3. records honestly:
   - the failed attempt → one `ComposerLLMCall` (`status=error`, `temperature=0.0`,
     `error_class="UnsupportedParamsError"`);
   - the successful retry → a second `ComposerLLMCall` (`status=success`,
     `temperature=None`).

Two audit rows = "no inference; if it's not recorded, it didn't happen." Retry is once
only; a second failure surfaces normally (no loop). Any non-`UnsupportedParamsError`
`BadRequestError` surfaces unchanged as today.

To avoid duplicating retry logic across the call sites, the helper returns both the
response **and** the sampling actually used, so each caller builds its audit record from
the true sent shape (preserving audit fidelity — the retry is not hidden inside the
`_litellm_acompletion` wrapper, which cannot signal back what it sent).

## 4. Components touched (composer-only)

| File | Change |
|------|--------|
| `contracts/composer_llm_audit.py` | `ComposerLLMCall.temperature: float` → `float \| None`; update docstring (L73–82) to state temperature is `0.0` only when the provider accepts it, `None` when omitted. |
| `composer/llm_response_parsing.py` | `build_llm_call_record(temperature: float)` → `float \| None` (single audit-construction point). |
| `composer/service.py` | Add `_composer_sampling_params(model)` + `ComposerSamplingDecision` + Layer-2 retry helper; rewire `_call_llm`, `_call_text_llm`, advisor; update comment block L140–153; **remove** the now-subsumed `_composer_llm_seed_for_model` / `_litellm_completion_supports_param`. |
| `composer/guided/chat_solver.py`, `composer/guided/chain_solver.py` | Route sampling + retry through the shared helpers. |
| `sessions/_auto_title.py` | Same; `_AUTO_TITLE_TEMPERATURE` flows through the effective-params helper. |

## 5. Persistence / audit (no migration required)

`ComposerLLMCall` is persisted as a **JSON envelope** in the `tool_calls` JSON column
(`composer/audit.py:llm_call_audit_envelope`), canonicalized with rfc8785/JCS — not as a
typed `NOT NULL` SQL column. `seed: int | None` already serializes `None → null` through
this exact path. Therefore making `temperature` nullable needs **no schema migration and
no staging DB delete**. `temperature: null` rides the same proven JCS path as `seed`.

## 6. Error handling summary

- Catch order: `UnsupportedParamsError` (retry) **before** `BadRequestError` (surface).
- litellm calls remain system-code (no defensive swallowing) except the one targeted
  `UnsupportedParamsError` catch — a known provider value-constraint at the trust
  boundary, which CLAUDE.md permits.
- Empty-choices Tier-3 guard (`_MalformedLLMResponseError`) is unchanged.

## 7. Testing (against real litellm, not mocks)

The bug is litellm-behavior-dependent; mocking `get_optional_params` would re-introduce
the "works on my machine" risk. Tests exercise real litellm where they assert litellm
behavior, and use the existing `_call_llm` mocking seam for the retry/audit assertions.

- **Layer 1 unit:** `_composer_sampling_params("azure/gpt-5")` and `"o3-mini"` →
  `temperature is None`; `"gpt-4o"` → `temperature == 0.0`; seed parity preserved across
  the unified path.
- **Audit fidelity:** omitted temperature → `ComposerLLMCall.temperature is None` → JCS
  envelope emits `null`.
- **Layer 2:** first call raises `UnsupportedParamsError` → exactly one retry with
  temperature omitted → two `ComposerLLMCall` rows with correct `temperature`/`status`.
- **Request shape:** `temperature` key absent from kwargs when `None`, present (`0.0`)
  when kept.
- **Non-target BadRequestError** still surfaces as `_BadRequestLLMError`.
- Integration tests use production code paths (`ExecutionGraph.from_plugin_instances()` /
  `instantiate_plugins_from_config()` where applicable per CLAUDE.md).

## 8. Scope boundary check

`contracts/call_data.py` (its own non-nullable `temperature: float`, required-keys set
including `temperature`) is the **pipeline** LLM-call audit, **not** referenced by
`composer/` or `sessions/_auto_title.py` (verified). It is out of scope. If the same
hardcoded-temperature risk exists for pipeline LLM transforms, it is surfaced as a
separate follow-up issue — not silently folded in here.

## 9. Open implementation details (for the plan, not the design)

- Exact extraction shape of the Layer-2 retry helper across the 5 call sites.
- Whether to `lru_cache` the per-model decision.
- Confirm `get_optional_params` / `get_llm_provider` call signatures against the pinned
  litellm 1.85.0 at implementation time.
