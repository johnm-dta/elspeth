# Advisor Prompt-Cache Markers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (ultracode) to implement this plan task-by-task. Adapt the pytest-TDD template faithfully — these are real RED→GREEN cycles, not component-mount reuse.

**Goal:** Make the composer **advisor** call (`_call_advisor_with_audit`) *eligible* for Anthropic prompt caching by applying the existing `cache_control` marker — gated on the **advisor** model — and prove the usage-accounting plumbing captures the resulting cache tokens.

**Architecture:** Pure reuse + reroute. The marker helpers (`supports_anthropic_prompt_cache_markers`, `apply_anthropic_cache_markers`) already exist in `llm_response_parsing.py` and are already imported into `service.py`. The primary composer path (`service.py:4160-4162`) is the reference pattern. We mirror it in the advisor method, which currently *deliberately* withholds markers (documented TODO at `service.py:3744-3749`). No new subsystems, no new helpers.

**Tech Stack:** Python 3.12 + FastAPI + LiteLLM; `pytest` (frontend untouched).

**Prerequisites:**
- Branch: `release/0.7.0` (current). This is 0.7.0 scope; commit directly per the release-branch convention. HEAD is `af6e95eed`.
- Main `.venv` active.

---

## Scope reframing (from advisor review — READ BEFORE STARTING)

This change makes the advisor **eligible** for caching. It does **not**, on its own, prove caching is *recovered*. Unit tests in this area are *plumbing* tests (they script `cache_read_input_tokens` into a mock and assert capture) — they prove "if the provider returns cache tokens, we record them," **not** that caching actually happens.

Two economic facts govern whether this is a win or a regression:
- Anthropic `ephemeral` cache: **5-minute TTL**, writes cost ~1.25× base input, reads ~0.1×.
- The advisor fires at early-plan-review and end-sign-off checkpoints. If those land **>5 min apart**, or the deployment is sparse (one tester), every call is a cache **write** on a multi-thousand-token system prompt with **no read to amortize** → **net cost increase**.

**Precondition already verified (static):** `build_system_prompt(data_dir)` is byte-stable — it composes a static module-level `_PIPELINE_SKILL` + a disk-cached deployment skill, joined deterministically. No timestamp / model-list / catalog-sha rides the cacheable head (the catalog sha line-rides the *audit*, not this prompt). So the marker has a stable prefix to cache. ✅

**Precondition NOT verifiable statically (→ DoD gate):** that two advisor calls actually land within the 5-min TTL in real use. This is the load-bearing assumption and it is settled only in the OpenRouter logs.

**Therefore the Definition of Done includes a LIVE check, and "marker applied correctly but cache reads ≈ 0" is a LEGITIMATE outcome meaning reconsider/revert — not ship-and-declare.** Ship this as *"advisor is now eligible for caching, confirmed hitting in OpenRouter logs"*, never *"caching recovered"* on green units alone.

---

## Baseline (capture on a clean tree at HEAD before Task 1)

```bash
# from repo root, main .venv active
pytest tests/unit/web/composer/test_advisor_tool.py \
       tests/unit/web/composer/test_advisor_checkpoint.py \
       tests/unit/web/composer/test_provider_cache_markers.py \
       tests/unit/web/composer/test_compose_loop_envelope.py \
       tests/unit/contracts/test_token_usage.py \
       tests/unit/contracts/test_composer_llm_audit.py -q
```

Record the pass/fail count. Any pre-existing red here is NOT introduced by this work — note it and do not let it mask a new regression. "Green for this slice" = the baseline set plus the new tests pass, and no previously-green test flips.

---

## Task 1: Apply the cache marker in the advisor path (gated on advisor_model)

**Files:**
- Modify: `src/elspeth/web/composer/service.py` — method `_call_advisor_with_audit` (def at `:3718`; docstring `:3744-3749`; messages built `:3768-3771`; kwargs built `:3779-3783`).
- Test: `tests/unit/web/composer/test_advisor_tool.py` (reuse the in-module `_Fake*` mocks / `_make_settings` / `_mock_catalog`).

**Step 1: Write the failing tests**

Add a new section to `test_advisor_tool.py`. These call `_call_advisor_with_audit` with `_litellm_acompletion` monkeypatched to capture the outbound `kwargs`.

```python
# --- Advisor prompt-cache markers (elspeth-4e79436719 follow-up) ---

import elspeth.web.composer.service as _composer_service_mod


def _valid_advisor_arguments() -> dict[str, object]:
    return {
        "trigger": "proactive_security_safety",
        "problem_summary": "demo",
        "recent_errors": ["none"],
        "attempted_actions": ["none"],
    }


@pytest.mark.asyncio
async def test_advisor_anthropic_model_receives_cache_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the advisor model is Anthropic-family, the system message carries
    a cache_control marker — mirroring the primary composer path."""
    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs: object) -> object:
        captured.update(kwargs)
        return _make_advisor_response()

    monkeypatch.setattr(_composer_service_mod, "_litellm_acompletion", _fake_acompletion)

    settings = _make_settings()  # composer_advisor_model default = anthropic/claude-sonnet-4-6
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)

    await service._call_advisor_with_audit(_valid_advisor_arguments(), recorder=None)

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["cache_control"] == {"type": "ephemeral"}
    # User message must NOT be marked (only the stable head is cached).
    assert "cache_control" not in messages[1]


@pytest.mark.asyncio
async def test_advisor_non_anthropic_model_no_cache_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-Anthropic advisor model must NOT carry cache_control markers
    (those providers auto-cache and silently ignore the field)."""
    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs: object) -> object:
        captured.update(kwargs)
        return _make_advisor_response()

    monkeypatch.setattr(_composer_service_mod, "_litellm_acompletion", _fake_acompletion)

    # Advisor must differ from composer_model; use two distinct OpenAI-family ids.
    settings = _make_settings(composer_model="gpt-5.5", composer_advisor_model="gpt-5.4-mini")
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)

    await service._call_advisor_with_audit(_valid_advisor_arguments(), recorder=None)

    messages = captured["messages"]
    assert "cache_control" not in messages[0]
```

> **Adapt to harness reality:** confirm `_call_advisor_with_audit`'s exact signature/keyword names by reading `:3718` (the message body uses `arguments`, `recorder`, optional `timeout`). Confirm `_make_settings` accepts `composer_model`/`composer_advisor_model` overrides; if it does not, extend it minimally. If `pytest.mark.asyncio` is not the project's async style, match the file's existing async test convention.

**Why these tests:** Test 1 proves the marker lands on the stable head for the real default advisor model. Test 2 is the guard the advisor reviewer flagged — gating on the **advisor** model, never `self._model` (which is `gpt-5.5`); a wrong gate would either never fire or wrongly mark an OpenAI payload.

**Step 2: Run to verify they fail**

```bash
pytest tests/unit/web/composer/test_advisor_tool.py -k "cache_control" -q
```
Expected: `test_advisor_anthropic_model_receives_cache_control` FAILS (`KeyError: 'cache_control'`) because markers are not yet applied. The non-anthropic test should already pass (vacuously) — that is fine.

**Step 3: Apply the marker (minimal change)**

In `_call_advisor_with_audit`, immediately AFTER the `messages` list is built (`:3771`) and BEFORE `kwargs` is built (`:3779`), insert:

```python
        # Anthropic-family advisor models honor explicit cache_control markers;
        # gate on the ADVISOR model (self._model is the composer model, gpt-5.5).
        # The marked `messages` flows to BOTH the wire call and the audit record
        # below, so the audit messages_hash is over the bytes actually sent —
        # the same truthfulness contract as the primary path (service.py:4160).
        if supports_anthropic_prompt_cache_markers(advisor_model):
            messages, _ = apply_anthropic_cache_markers(messages, None)
```

(`supports_anthropic_prompt_cache_markers` and `apply_anthropic_cache_markers` are already imported at `service.py:84,88` — verify, do not re-import.)

**Step 4: Update the docstring** at `:3744-3749`. Replace the "deliberately NOT applied" paragraph with the new contract, e.g.:

```python
        Anthropic prompt-cache markers ARE applied here when the advisor
        model is Anthropic-family (gated on ``advisor_model`` via
        ``supports_anthropic_prompt_cache_markers`` — NOT ``self._model``,
        which is the composer model). The marked ``messages`` is what flows
        to LiteLLM and what the audit ``ComposerLLMCall`` record hashes, so
        the audit row is truthful about the wire payload, mirroring the
        primary composer path (``_call_llm_with_audit``). Advisor accounting
        (``cached_prompt_tokens`` in the returned metadata and the audit
        record) is exercised by ``test_advisor_*_cache_*`` and remains
        independent from the primary-composer marker placement.
```

**Step 5: Run to verify GREEN**

```bash
pytest tests/unit/web/composer/test_advisor_tool.py -k "cache_control" -q
```
Both pass. Then run the full baseline set (above) — no previously-green test may flip (watch for any messages_hash snapshot on the advisor path).

**Step 6: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_advisor_tool.py
git commit -m "feat(web/composer): apply Anthropic cache_control marker on advisor calls

The advisor (composer_advisor_model, default anthropic/claude-sonnet-4-6)
deliberately withheld prompt-cache markers; its large stable system head
(build_system_prompt + _ADVISOR_SYSTEM_INSTRUCTIONS) was re-sent uncached
on every checkpoint call. Mirror the primary path: gate on the ADVISOR
model and mark the system message so the stable prefix is cacheable. The
marked messages feed both the wire call and the audit record, keeping the
messages_hash truthful about the bytes sent.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Definition of Done:**
- [ ] Both marker tests pass; baseline set still green (no flips).
- [ ] Gate is on `advisor_model`, not `self._model`.
- [ ] Docstring reflects the new contract.

---

## Task 2: Usage-accounting test (cache tokens captured in metadata + audit)

This is the "focused usage-accounting test" the original code comment demanded. It proves plumbing only (see Scope reframing) — the live read is the real proof.

**Files:**
- Test: `tests/unit/web/composer/test_advisor_tool.py`.
- (Likely no source change — `token_usage_from_response` already parses Anthropic `cache_read_input_tokens` / `cache_creation_input_tokens`; this task confirms it end-to-end through the advisor return + audit record.)

**Step 1: Write the failing/【confirming】test**

```python
@pytest.mark.asyncio
async def test_advisor_records_anthropic_cache_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """A warm advisor call (provider returns cache_read_input_tokens) surfaces
    cached_prompt_tokens in the returned metadata and the audit record."""

    async def _fake_acompletion(**kwargs: object) -> object:
        resp = _make_advisor_response()
        # Anthropic sibling usage shape: cache_read_input_tokens on usage.
        resp.usage.cache_read_input_tokens = 5_000          # adapt: extend _FakeUsage if needed
        resp.usage.cache_creation_input_tokens = None
        return resp

    monkeypatch.setattr(_composer_service_mod, "_litellm_acompletion", _fake_acompletion)

    recorder = <project's standard BufferingRecorder test double — see test_compose_loop_llm_audit.py>
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())

    _guidance, metadata = await service._call_advisor_with_audit(
        _valid_advisor_arguments(), recorder=recorder
    )

    assert metadata["cached_prompt_tokens"] == 5_000
    # Audit sidecar carries the cache token field.
    call_record = recorder.<recorded llm_calls>[-1]
    assert call_record.cache_read_input_tokens == 5_000
```

> **Adapt to harness reality:** `_FakeUsage` currently exposes `prompt_tokens/completion_tokens/total_tokens` — extend it (or build a `SimpleNamespace`) to carry `cache_read_input_tokens` / `cache_creation_input_tokens`, matching the sibling shape in `test_service.py:1989-2007` and `test_compose_loop_envelope.py:142-173`. Use the SAME recorder double the existing audit tests use (`test_compose_loop_llm_audit.py`) and read its recorded `ComposerLLMCall` the same way; the exact attribute names (`cache_read_input_tokens`) come from `contracts/composer_llm_audit.py`. If `token_usage_from_response` maps Anthropic `cache_read_input_tokens` onto `cached_prompt_tokens`, assert that mapping rather than guessing.

**Why this test:** It is the audit-fidelity guard. The original comment warned against "inheriting the primary-composer marker placement by accident" without accounting tests — this closes that gap by proving the cache tokens a warm Anthropic call returns are not dropped on the advisor path.

**Step 2-4: Run RED (if `_FakeUsage` lacks the fields) → extend harness → GREEN → full baseline set green.**

**Step 5: Commit**

```bash
git add tests/unit/web/composer/test_advisor_tool.py   # + service.py only if a mapping fix was required
git commit -m "test(web/composer): advisor surfaces Anthropic cache tokens in metadata + audit

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Definition of Done:**
- [ ] Warm-call test asserts `cached_prompt_tokens` in metadata AND the cache field on the audit `ComposerLLMCall` record.
- [ ] Baseline set green.

---

## Cross-cutting constraints (honor in every task)

| Constraint | Rule |
|---|---|
| **Gate on advisor_model** | Never `self._model` (= gpt-5.5). The reviewer flagged this as the most likely fumble. |
| **Audit truthfulness** | The marked `messages` must feed BOTH the wire call and `build_llm_call_record` (it already does — one `messages` var). Do not split them; the hash must be over bytes sent. |
| **No tools on advisor** | The advisor call has no `tools`; call `apply_anthropic_cache_markers(messages, None)` and discard the tools return. |
| **Tier boundaries** | Touch only the marker application + docstring + tests. Do NOT alter advisor argument validation, the Tier-3 trust-boundary `.get()`-vs-direct-access comments, or the failure-classification clauses. |
| **No scope creep** | The composer/gpt-5.5 staged path is OUT (its zero-caching is by design; padding skills past 1024 tokens is forbidden). This plan is advisor-only. |
| **Commit discipline** | One atomic commit per task; code commits run hooks (no `--no-verify`). Commit only; do NOT push unless the operator asks. |

---

## Definition of Done — SLICE (the live gate is not optional)

1. **Unit:** Tasks 1-2 green; baseline set shows no flips.
2. **Static:** `ruff`/`mypy` clean on the touched lines (run the repo's static set).
3. **LIVE (the real proof — settles precondition #2):** On staging, drive a single guided session that triggers **two** advisor checkpoints close together (e.g. an early plan-review then a sign-off, or Slice-B revision passes that cluster advisor calls within ~5 min). Then read the **OpenRouter logs** for the `anthropic/claude-sonnet-4-6` stream:
   - **Win:** the 2nd+ advisor call reports `cache_read_input_tokens > 0` (or equivalently a cache discount). Report the observed numbers.
   - **Legitimate non-win:** marker present on the wire but reads ≈ 0 across calls → the checkpoints don't cluster within the TTL. **Do not declare recovery.** Record the finding, and recommend either (a) revert (markers add write surcharge with no read benefit = cost increase), or (b) keep only if the audit shows clustered calls in real workloads. Surface this to the operator as a decision.
4. **Framing:** Land it as *"advisor eligible for caching, confirmed hitting in logs"* — never *"caching recovered"* on units alone.

## Tracker

File/locate a P2 issue under the advisor-reliability area (intersects `elspeth-dac6602a2b`) titled "advisor prompt-cache markers" and close it at the slice gate with the commit SHA(s) and the observed live `cache_read` numbers (or the no-read finding + decision).
