# Composer Staged-Path Prompt-Cache Markers — Implementation Plan (v2, post-review)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (ultracode) to implement task-by-task. Real RED→GREEN cycles. Revised against reality + quality plan-reviews (2026-06-30); blocking findings folded in.

**Goal:** Recover prompt caching on the staged guided composer (the `openrouter/anthropic/claude-sonnet-4-6` stream the operator watches) by applying the existing `cache_control` marker at the staged-solver LLM call sites, restructuring fused system messages so the stable per-stage skill is an isolable, markable, byte-stable head.

**Architecture:** Reuse the freeform pattern (`service.py:4148-4162` → `apply_anthropic_cache_markers`, which marks the first `role=="system"` message + trailing tool). Where the system message is already the pure stable skill, mark directly. Where skill + dynamic context are fused into one system message, SPLIT (stable skill → first system message; dynamic context → second system message), then mark. Gate every application on the call's `model`.

**Tech stack:** Python 3.12 + LiteLLM 1.85 (`OpenrouterConfig._move_cache_control_to_content` rewrites our top-level marker into the content-block form OpenRouter→Anthropic honors — verified in source + live probe). `pytest`.

**Prerequisites:** Branch `release/0.7.0` (current); main `.venv` active.

---

## Verified findings (empirical ground; do not relitigate)

1. **Caching works end-to-end.** Live 2-call probe, `openrouter/anthropic/claude-sonnet-4-6`, 18.5k-token prefix + top-level `cache_control`: call 1 `cache_write_tokens=18357` ($0.069); call 2 `cached_tokens=18357` ($0.0056, ~12× cheaper). The marker is effective.
2. **The staged path never marks.** Markers live only in freeform `_call_llm_with_audit` (`service.py:4160`). All 105 staged calls in `data/sessions.db` show zero cache → all unmarked.
3. **OpenRouter cache accounting:** `prompt_tokens_details.cache_write_tokens` (write) + `prompt_tokens_details.cached_tokens` (read); Anthropic siblings ABSENT. App captures the READ as `cached_prompt_tokens` (the live DoD signal); the WRITE is not captured (Task 5, deferred). **Live signal = `cached_prompt_tokens > 0`, NOT `cache_read_input_tokens`.**
4. **Two skill loaders (`guided/prompts.py`):** `load_step_chat_skill(step)` = base + ONE step (used by `chat_solver`); `load_guided_skill()` = base + ALL steps (used by `chain_solver`, ≈ 3768 tok). Per-stage `load_step_chat_skill` sizes (char/4, confirmed against files): step_1 ~1199 ✓, step_2 ~915 ✗, step_3 ~2405 ✓, step_4 ~749 ✗ (Anthropic 1024-tok floor).
5. **`chain_solver.solve_chain` is the PRIMARY step_3 BUILD path** (route `guided.py:2362-2433`, when source+sink committed). `solve_step_chat(STEP_3)` is only the advisory-prose FALLBACK (route `guided.py:2549-2561`). The operator-watched step_3 traffic is `chain_solver`.
6. **`chain_solver` fuses skill+context** into one system message (`chain_solver.py:245`). Marking as-is caches only within a discovery loop, NOT cross-turn → needs a SPLIT.
7. **No baseline test pins a literal `messages_hash`/`tools_spec_hash`** (only an unrelated `error_hash` in `test_freeze_regression.py`). The marker changes the advisor/staged messages_hash but no literal-pinned test flips. **However two tests assert dynamic content in `messages[0]` and WILL require updates after the splits (see Task 2/3).**

---

## Baseline (capture at HEAD before Task 1)

```bash
pytest tests/unit/web/composer/guided tests/integration/web/composer/guided \
       tests/unit/web/composer/test_provider_cache_markers.py \
       tests/unit/contracts/test_token_usage.py -q
```
Record pass/fail. Known 0.7.0 baseline reds exist in the guided suites — note them; do NOT fix here. **NOTE: unlike a pure-additive change, Tasks 2 and 3 INTENTIONALLY update two currently-green tests (listed in those tasks). "Green for this slice" = baseline reds unchanged + the two named tests updated-and-green + new tests pass.**

---

## Scope — per call site (reconciled)

| Site | Structure | Action | Stage skill | Value |
|---|---|---|---|---|
| `solve_step_chat` (`chat_solver.py:860`) | pure `load_step_chat_skill(step)`, `tools=None` | **mark directly** (`apply_anthropic_cache_markers(messages, None)`) | all-step fallback; caches where skill ≥1024 (step_1, step_3-fallback) | Medium (fallback path) |
| `chain_solver.solve_chain` (`chain_solver.py:239-250`) | **fused** `load_guided_skill()` + context + addenda | **SPLIT** skill→`messages[0]` (marked), context+addenda→`messages[1]`; mark in loop | ~3768 tok ✓ | **HIGH (primary step_3)** |
| `maybe_resolve_step_1_source_chat` (`chat_solver.py:405-432`) | **mixed** skill + `hint`/`revise_block` + tool instructions | **SPLIT** skill→`messages[0]` (marked), dynamic→`messages[1]`; mark first system + trailing tool | ~1199 tok ✓ (only skill caches; trailing tool instructions excluded) | Medium |
| `maybe_resolve_step_2_sink_chat` (`chat_solver.py:699-718`) | mixed skill (~915) + tools | **SKIP** (skill < 1024). Document; note tool-array/cumulative caching deferred (marginal). | ~915 tok ✗ | None now |
| step_4 wire | only via `solve_step_chat(STEP_4)` fallback | inert: marked but below-floor no-op (not N/A) | ~749 tok ✗ | None |

OUT: freeform path (already marks), advisor (opus stream, deprioritised).

---

## Task 1: `solve_step_chat` — mark directly

**Files:** `chat_solver.py` (`solve_step_chat`, ~:860-880). Test: `tests/unit/web/composer/guided/test_chat_solver_sampling_config.py` (direct-call + `monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake)`, capture kwargs).

**Step 1 — RED tests:**
- `solve_step_chat(step=STEP_1_SOURCE, model="openrouter/anthropic/claude-sonnet-4-6", ...)` → `captured["messages"][0]["cache_control"] == {"type":"ephemeral"}`; `messages[1]` (user) unmarked.
- Same call with `model="openrouter/openai/gpt-5.5"` → no `cache_control` on `messages[0]`.

**Step 3 — implement.** Add to existing import (`chat_solver.py:34`): `apply_anthropic_cache_markers, supports_anthropic_prompt_cache_markers`. Insert between `:864` (`messages` built) and `:865` (`kwargs` built):
```python
if supports_anthropic_prompt_cache_markers(model):
    messages, _ = apply_anthropic_cache_markers(messages, None)  # solve_step_chat has no tools
```
(The marked `messages` feeds both `kwargs["messages"]` and the audit `build_llm_call_record(messages=messages)` at ~:936 — same object, hash stays truthful.)

**DoD:** marker present (Anthropic) / absent (non-Anthropic); baseline reds unchanged.

---

## Task 2: `chain_solver.solve_chain` — SPLIT + mark (the primary step_3 win)

**Files:** `chain_solver.py` (~:239-285, the loop). Test: `tests/unit/web/composer/guided/test_chain_solver_sampling_config.py`. **UPDATE:** `test_solve_chain_redacts_sample_rows_in_outbound_messages` (assertions move `messages[0]`→`messages[1]`).

**Step 1 — RED tests:**
- Byte-stability + marker: `solve_chain(model="openrouter/anthropic/...", ...)` → `captured["messages"][0]["content"] == load_guided_skill()` (verbatim, stable head) AND `messages[0]["cache_control"] == {"type":"ephemeral"}`; the context block + any addenda are in `messages[1]` (unmarked); `intent` user msg follows.
- Multi-round: across ≥2 discovery rounds, each captured wire payload's `messages[0]` carries the marker; the original growing `messages` list's system dict stays unmarked (per-round `request_messages` snapshot stays hash-truthful).
- Negative: `model="openrouter/openai/gpt-5.5"` → no marker.

**Step 3 — implement.** Replace the fused construction (`:245-250`):
```python
skill = load_guided_skill()
context_block = build_step_3_context_block(source=source, sink=sink)
context_parts = [context_block]
if repair_context is not None:
    context_parts.append(build_repair_addendum(validation_error=repair_context))
if revise_context is not None:
    context_parts.append(build_revise_addendum(revise_instruction=revise_context))
messages: list[dict[str, Any]] = [
    {"role": "system", "content": skill},                         # stable, markable head
    {"role": "system", "content": "\n\n".join(context_parts)},    # dynamic context + addenda
]
if intent is not None:
    messages.append({"role": "user", "content": intent})
```
Then INSIDE the loop, after the snapshot (`request_messages = list(messages)`, ~:284) and before `kwargs` (~:285):
```python
if supports_anthropic_prompt_cache_markers(model):
    request_messages, _t = apply_anthropic_cache_markers(request_messages, tools)
    if _t is not None:
        tools = _t
```
(Re-marking each round is idempotent — `{**msg, "cache_control": ...}`. `request_messages` feeds both `kwargs["messages"]` and the audit record at ~:419.)

**Step 4 — UPDATE the breaking test.** `test_solve_chain_redacts_sample_rows_in_outbound_messages`: the redacted sample summaries now appear in `messages[1]["content"]` (the context block), not `messages[0]`. Update the assertions to index `messages[1]`. This is a deliberate relocation, not a regression — redaction still happens.

**DoD:** `messages[0]` is exactly `load_guided_skill()` + marker; dynamic content in `messages[1]`; updated redaction test green; multi-round marker test green.

---

## Task 3: step_1 source — SPLIT + mark

**Files:** `chat_solver.py` (`_build_step_1_source_tool_prompt` :176, call site :405-432). **UPDATE:** `tests/integration/web/composer/guided/test_step_chat_source_driver.py` (revise-context JSON moves `messages[0]`→`messages[1]`).

**Step 1 — RED tests:** `maybe_resolve_step_1_source_chat(model="openrouter/anthropic/...", plugin_hint="csv", current_source=<src>, ...)` → `messages[0]["content"] == load_step_chat_skill(STEP_1_SOURCE).rstrip()...` (stable skill) + marker; the `hint`/`revise_block` text is in `messages[1]`, unmarked; trailing tool marked.

**Step 3 — implement.** Extract the dynamic portion of `_build_step_1_source_tool_prompt` into `_build_step_1_source_dynamic_block(*, plugin_hint, current_source)` (the `hint` + `revise_block` + the tool-instructions tail). Build:
```python
messages = [
    {"role": "system", "content": load_step_chat_skill(GuidedStep.STEP_1_SOURCE).rstrip()},  # marked
    {"role": "system", "content": _build_step_1_source_dynamic_block(plugin_hint=plugin_hint, current_source=current_source)},
    {"role": "user", "content": user_message},
]
```
Then gated `messages, tools = apply_anthropic_cache_markers(messages, tools)`. NOTE: only the ~1199-tok skill caches; the static tool-instructions tail rides in `messages[1]` (after dynamic content) and is not in the marked prefix — acceptable.

**Step 4 — UPDATE the breaking test.** `test_step_chat_source_driver.py`: the current-source revision JSON now appears in `messages[1]`. Update assertions to `messages[1]`.

**DoD:** stable skill is the marked first message; dynamic + tool instructions in `messages[1]`; resolution behaviour unchanged (parser reads the tool RESPONSE, not the prompt); updated driver test green.

---

## Task 4: document the step_2 / step_4 skips

One-line comments at the step_2 site (skill ~915 < 1024 floor; tool-array/cumulative-prefix caching deferred as marginal — revisit if the skill grows) and a note that `solve_step_chat(STEP_4)` marking is an inert below-floor no-op. No silent caps.

---

## Task 5 (DEFERRED — do NOT pull in without a schema change): capture OpenRouter write side

`token_usage.py` misses `prompt_tokens_details.cache_write_tokens`. **BLOCKING TRAP if attempted:** mapping it onto `cache_creation_input_tokens` trips the dedup at `llm_response_parsing.py:176-177` and NULLS the `cached_tokens → cached_prompt_tokens` READ signal that is the entire DoD. If ever revived: add a SEPARATE field (schema change) + focused test; never reuse the Anthropic write sibling. Deferred — the READ signal is sufficient for this slice.

---

## Cross-cutting constraints

| Constraint | Rule |
|---|---|
| **Gate on the call's `model`** | Each site has `model` (= `settings.composer_model`). Never `self._model`-style or hard-coded. |
| **Marker BEFORE kwargs/snapshot** | solve_step_chat: before `:865`. chain_solver: between the `request_messages` snapshot (`:284`) and `kwargs` (`:285`), inside the loop. step_1: before its kwargs. The marked list MUST be the same object fed to both `_litellm_acompletion` and `build_llm_call_record(messages=...)`. |
| **Two intentional test updates** | Task 2 + Task 3 update `test_solve_chain_redacts_sample_rows…` and `test_step_chat_source_driver`. These are relocations (content → `messages[1]`), not regressions. Do NOT read them as baseline reds. |
| **Below-floor no-pad** | Do NOT enlarge step_2/step_4 skills to clear 1024. |
| **400 safety (F5)** | Probe shows the marker does NOT 400 on this route. Add NO speculative retry. If a 400 ever appears, REMOVE the marker (don't swallow). Live DoD includes an explicit "did not 400" gate. |
| **Tier boundaries** | Touch only message construction + marker application + the two test updates. Do not alter resolution parsers, redaction, or failure-classification. |
| **Imports** | Add the two marker symbols to the EXISTING `llm_response_parsing` import in each solver (no new module, no cycle). |
| **Commit discipline** | One atomic commit per task; hooks run (no `--no-verify`). Commit only; do NOT push unless asked. |

---

## Definition of Done — SLICE

1. **Unit (wire-proof, deterministic):** Tasks 1-3 green incl. the two updated tests + the new marker/byte-stability tests. Baseline reds unchanged.
2. **Byte-stability regression guard:** the `messages[0] == load_guided_skill()` / `== load_step_chat_skill(...)` assertions exist so a future dynamic insertion into the stable head fails loudly.
3. **Audit-truthfulness test:** assert `record.messages_hash == stable_hash(captured_wire_messages)` (hash of the CAPTURED wire list, not a list the test marked itself). Home: `test_audit_emission.py` or the sampling_config modules.
4. **Static:** ruff/mypy clean on touched lines.
5. **LIVE (read-proof — the real win; settles cross-turn caching):** On staging, perform **two separate step_3 Sends in distinct turns** (a build then a follow-up/revise) within ~5 min — NOT relying on within-call discovery rounds. Then read the audit (`data/sessions.db` → `chat_messages.tool_calls` `llm_call_audit` envelopes) / OpenRouter dashboard for `openrouter/anthropic/claude-sonnet-4-6`:
   - **Win:** the 2nd step_3 call shows `cached_prompt_tokens > 0`. Report the number.
   - **Diagnosis on zero:** unit-green (marker on wire) + live-zero ⇒ TTL gap, below-floor stage, or routing — NOT a dropped marker. Record which stages hit.
   - **400-negative gate (F5):** confirm the marked staged calls did not 400 (status `success` in the audit envelopes).
6. **Framing:** "staged guided composer is now eligible for caching; cross-turn step_3 reads confirmed in the audit."

## Tracker

Task #14 (re-scoped to composer staged path). Close at the slice gate with commit SHAs + observed live `cached_prompt_tokens` numbers per stage.
