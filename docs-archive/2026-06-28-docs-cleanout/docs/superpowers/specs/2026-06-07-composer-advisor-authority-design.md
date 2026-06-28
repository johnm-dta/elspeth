# Composer Advisor Authority — Deterministic Checkpoints + Config Mandate (Design)

**Date:** 2026-06-07
**Status:** Design — approved, pre-implementation
**Ticket:** elspeth-dac6602a2b (P1 feature)
**Origin:** 2026-06-07 tutorial-reliability investigation (Case B fix `5deb34f78` + a composer-model-lever experiment). The composer advisor was an opt-in "nice to have"; this work cements it as a MUST-HAVE ("the responsible parent must be set") and takes the trigger decision away from the weak model.

---

## 1. Problem

A two-battery staging experiment (composer model the only variable) showed tutorial composition quality is **model-capability-bound**: `gpt-5.4-mini` graduated 3/6 with 50% execution-validation failures and 33% `vague_term` under-flag; `claude-opus-4-7` graduated 6/6 with zero of either. The failures were **confident** failures — the weak model never realised it was stuck, so it never fired the reactive `request_advisor_hint` escape-hatch. **You cannot fix a metacognition gap by making the self-trigger smarter; you take the trigger decision away from the weak model.**

The fix mirrors the dev-harness loop that caught the two HIGH bugs during the Case-B work (early advisor consult + adversarial end-review). We productise that loop as a **runtime** loop: deterministic, backend-initiated advisor checkpoints, with the advisor as the final authority — and we make it impossible to run the composer without a distinct advisor configured.

## 2. Goals / Non-goals

**Goals (this spec):**
- Advisor engaged **early** (proactive approach review) and as the **final authority** at end sign-off.
- Advisor **mandatory** in config — service will not boot without one.
- Advisor model **distinct** from every active primary (bulk + tutorial), enforced at startup.
- **Two-tier model wiring**: an independent tutorial-path model override.

**Non-goals (deferred to sibling specs — ticket NEXT STEPS §3 says decompose):**
- The deterministic **field-contract checker** (producer→consumer field-availability propagation incl. subtractive transforms) — the *deterministic half* of the end sign-off. Slots in **beside** the advisor gate later (see §9).
- **Reciprocity enforcement** (`{{row.field}}` ↔ `required_input_fields`), currently only taught in the skill.
- **`vague_term` advisory nudge** (deterministic-detect + LLM-reconsider, ~99% recall, NOT fail-closed).

These are explicitly out of scope here; this spec ships the **semantic-authority** slice, which can land while the deterministic layer builds in parallel.

## 3. Decisions (locked with operator)

| # | Decision | Choice |
|---|----------|--------|
| D-1 | Independence depth | **Exact-string distinctness**, normalized (strip provider prefixes). No family/provider detection. `advisor ∉ {composer_model, composer_tutorial_model}`. |
| D-2 | Spec scope | Advisor early pass + end authority + config mandate + **two-tier wiring**. Deterministic checker / reciprocity / vague_term-nudge deferred. |
| D-3 | Hook architecture | **Hybrid (Approach A):** new deterministic checkpoint orchestration that **reuses** the existing audited advisor call (`_call_advisor_with_audit`). NOT a backend-synthesized `request_advisor_hint` (provenance lie); NOT a fully separate subsystem (duplication). |
| D-4 | "Final authority" structure | A **re-review loop**: advisor flags → cheap model repairs → advisor **re-reviews** → finalize. Self-certify-after-advice is not authority. |
| D-5 | Early vs End authority | **EARLY = advisory** (feeds repair loop, never blocks). **END = authoritative** (can block). |
| D-6 | Mandatory enforcement | **Fail-closed at config load / startup.** Remove `composer_advisor_enabled`; the service won't boot without a valid, distinct advisor. |
| D-7 | Advisor-unavailable at end gate | **Fail closed with bounded retry**; surface "couldn't obtain sign-off, retry" to the user. Early-pass advisor failure **degrades silently**. |
| D-8 | Budget | **Separate** advisor-checkpoint budget, independent of `_MAX_REPAIR_TURNS`; an advisor pass can **never** starve a correctness repair. |

## 4. Architecture & components

New **checkpoint orchestrator** on `ComposerService`, deterministic (backend-initiated), reusing the existing audited advisor call.

- **`_run_advisor_checkpoint(phase: "early" | "end", *, state, intent, current_state_id, …) -> AdvisorCheckpointVerdict`**
  - Builds **phase-specific context** (NOT the LLM-supplied `_build_advisor_user_message`):
    - *early:* DAG topology + producer→consumer field contracts (best-effort, descriptive) + the user's stated intent + the composed approach so far.
    - *end:* the final pipeline + intent → asks: intent fulfilment, soundness, subjective-rubric flags.
  - Calls the existing **`_call_advisor_with_audit`** (service.py:3485) — reuses the audited call, redaction summarizers (redaction.py `_summarize_advisor_*`), timeout, model.
  - Returns a structured verdict: `findings: list[str]`, `blocking: bool`, `ok: bool` (call succeeded).

- **Early hook:** fires **once per session**, on the first compose turn that establishes a non-empty pipeline topology (i.e. after a `set_pipeline` leaves the composition non-empty). "Already done" is derived from the advisor audit trail (an early-phase checkpoint event exists for the session) — **no new column**. Findings are injected as a guidance/repair-style message for the cheap model. Advisory — never blocks. (Subsequent-turn approach changes rely on the END gate, which runs at every finalize; re-firing early on "substantial topology change" is deliberately out of scope for v1 — YAGNI.)

- **End hook:** inside **`_surface_and_finalize_no_tools`** (the seam landed in `5deb34f78`), **after** the existing cheap deterministic gates (orphan/interpretation gate, proof gate) and **before** finalize. Runs the re-review loop (§6). Authoritative — can block (fail-closed result, same shape as the orphan gate).

- **Retire** the `reactive_validation_loop` trigger (sessions.py:116 `ADVISOR_TRIGGER_REACTIVE`) — superseded by deterministic checkpoints. **Keep** `proactive_security_safety` / `proactive_red_listed_plugin` (different job — security, not validation).

**Reuse map:** keep `_call_advisor_with_audit`, advisor redaction summarizers, the advisor budget carrier pattern (`advisor_calls_used` in `_compose_loop_carriers.py`). Replace the *trigger* (LLM-decided → backend-decided) and add *new* phase-specific context builders.

## 5. Config contract

In `WebSettings` (`src/elspeth/web/config.py`), validated by a `model_validator` at **load = startup** (fail-closed):

- **Remove `composer_advisor_enabled`** (config.py:83) and its disabled-path branches: `_strip_advisor_content` disabled mode + `_strip_advisor_disabled_fallback` (prompts.py:49/98), `build_system_prompt(advisor_enabled=…)` (prompts.py:110) collapses to always-on, and the `request_advisor_hint` dispatch filter for the disabled case (service.py:3040). Advisor is no longer optional.
- **New `composer_tutorial_model: str | None = None`** — two-tier override. `tutorial_model_id()` (tutorial_service.py:767) returns `composer_tutorial_model or composer_model`. (Tutorial cache key already folds in `composer_model`; update it to the resolved tutorial model.)
- **Distinctness validator:** normalize both sides (strip a leading `openrouter/…/` provider prefix to a canonical model id), then reject at startup if `composer_advisor_model` exact-equals the normalized `composer_model` **or** `composer_tutorial_model` (when set). Error message names the colliding pair and which primary it collided with.
- **New advisor-checkpoint budget knob** (e.g. `composer_advisor_checkpoint_max_passes`, default small, `ge=1`) — see §7. The existing `composer_advisor_model` / token / timeout knobs are reused; `composer_advisor_max_calls_per_compose` continues to bound *total* advisor calls (checkpoints + proactive-security) as a hard ceiling.

One advisor model, shared across both tiers, must differ from both primaries. (With exact-string only: e.g. tutorial=`claude-opus-4-7` + advisor=`claude-opus-4-7` is rejected; operator picks a distinct advisor such as `gpt-5.5`.)

## 6. The two checkpoints — control flow

**Early (advisory):**
```
first set_pipeline succeeds
  → _run_advisor_checkpoint(phase="early")
     → ok + findings → append guidance message; cheap model continues/repairs
     → ok + clean    → continue
     → call FAILED   → log + degrade silently (advisory only); continue
```

**End (authoritative gate).** The "re-review loop" is **not** a tight inner loop — it is realised across outer compose-loop turns via the existing repair-continue mechanism (inject message → return `continue` → outer loop re-runs the model → re-checks at the next finalize), exactly as the orphan/proof gates already work. Per finalization attempt:
```
deterministic gates run FIRST (orphan/interpretation, proof)
  → if FLAGGED → repair-continue, counts the CORRECTNESS budget; (advisor not reached this turn)
  → if PASS:
     _run_advisor_checkpoint(phase="end")
       → CLEAN        → finalize
       → FLAGGED      → repair-continue, counts the ADVISOR budget (inject advisor findings)
                        [next turn: model repairs → loop returns here → advisor RE-reviews]
       → FLAGGED AND advisor budget exhausted → FAIL CLOSED + surface
       → CALL FAILED  → bounded retry (same finalize) → still failing → FAIL CLOSED + surface
```

**Gate order is the budget-attribution mechanism** (see §7): because deterministic gates run before the advisor gate, at most ONE budget decrements per turn — a turn is a *correctness* repair (deterministic flag) OR an *advisor* repair (advisor flag), never both. This is precisely what keeps an advisor pass from ever consuming a correctness-repair turn.

The fail-closed result mirrors the orphan-gate `_TerminateOutcome` / `ComposerResult` shape from `5deb34f78` (authoring_valid / completion_ready / execution_ready all False; a blocking diagnostic the UI surfaces so "run" is never enabled on an unsigned-off pipeline).

## 7. Budget model (the #1 implementation risk)

Two independent budgets, **kept separate by gate order** (§6):
- **Correctness-repair budget** = the existing `_MAX_REPAIR_TURNS = 2` (service.py:936). Consumed only by repair-continues triggered by the **deterministic** gates (proof, orphan/interpretation), which run first.
- **Advisor-checkpoint budget** = a NEW separate counter `advisor_checkpoint_passes_used`, ceiling = `composer_advisor_checkpoint_max_passes`. Consumed by `_run_advisor_checkpoint` invocations and their repair-continues (early + end + re-reviews).

**Mechanism:** since deterministic gates run before the advisor gate within a finalization attempt, the advisor is reached only when no deterministic gate flagged — so any repair-continue that turn is attributed to whichever gate flagged, and at most one budget decrements per turn. The outer compose loop threads BOTH counters and checks them independently; an advisor repair-continue increments `advisor_checkpoint_passes_used` and leaves `repair_turns_used` untouched.

**Invariant (must be tested):** advisor budget exhaustion can NEVER consume or block a correctness-repair turn, and vice-versa. If advisor budget is exhausted while the end gate is still flagged, the gate **fails closed** rather than silently finalizing. The existing `composer_advisor_max_calls_per_compose` remains a hard upper bound across ALL advisor calls (checkpoints + proactive-security) — a backstop above the per-phase budgets.

## 8. Error handling / failure modes

| Condition | Behaviour |
|-----------|-----------|
| End-gate advisor call fails (timeout/provider) | Bounded retry (small N) → still failing → **fail closed + surface** |
| Early-pass advisor call fails | Log + **degrade silently** (advisory only); compose continues |
| Advisor returns malformed/unparseable verdict | Treat as a call failure (retry→fail-closed at end; degrade at early) |
| Config: advisor missing OR collides with a primary | **App won't boot**, precise error naming the pair |
| Advisor budget exhausted, end gate still flagged | **Fail closed + surface** (do not silently finalize) |

## 9. Forward-compatibility with the deferred deterministic layer

The end-gate ordering (cheap deterministic gates → advisor) is chosen so the deferred **field-contract checker** (sibling spec) slots in as another deterministic gate **before** the advisor, with no reordering: mechanical gaps are caught free and the advisor only ever reviews a mechanically-valid pipeline (its findings are purely semantic). The advisor's `phase="end"` context may *reference* a field-contract summary once that lands, but does not depend on it for this spec.

## 10. Testing

**Unit:**
- Distinctness validator: collision vs bulk; collision vs tutorial; provider-prefix-normalized collision (`gpt-5.5` ≡ `openrouter/openai/gpt-5.5`); pass when distinct; tutorial unset → only bulk checked.
- Mandate: removing/omitting the advisor model fails startup.
- Re-review loop: clean→finalize; flagged→repaired→clean→finalize; flagged→budget-exhausted→fail-closed; advisor-unavailable→retry→fail-closed.
- Early pass: findings injected; degrade-on-failure continues.
- **Budget isolation:** advisor-budget exhaustion never decrements the correctness-repair budget (the load-bearing invariant).
- Two-tier: `tutorial_model_id()` returns the override when set, falls back when not; tutorial cache key reflects the resolved model.

**End-to-end:** the staging tutorial battery (the harness that drove Case B) — confirm the cheap-model bulk path with deterministic advisor checkpoints lifts graduation/quality toward the frontier-primary result, and the tutorial path on the stronger model graduates clean.

## 11. Seam references (verified 2026-06-07)

- Config: `src/elspeth/web/config.py` — `composer_model` :53, `composer_advisor_enabled` :83, `composer_advisor_model` :84, advisor budget knobs :85-102.
- Advisor protocol: `src/elspeth/web/composer/protocol.py:673-688`.
- Advisor call core: `service.py:3485` `_call_advisor_with_audit`; `:3936` `_build_advisor_user_message` (LLM-supplied — NOT reused for checkpoints); `:3116` `_validate_advisor_arguments`.
- Triggers: `tools/sessions.py:116-125` (`ADVISOR_TRIGGER_*`); dispatch `tools/_dispatch.py:114` (`request_advisor_hint` def), `:251` (filtered when disabled).
- Finalize seam: `service.py` `_surface_and_finalize_no_tools` (added by `5deb34f78`); `_MAX_REPAIR_TURNS` :936.
- Budget carrier: `_compose_loop_carriers.py` (`advisor_calls_used`).
- Tutorial model: `tutorial_service.py:767` `tutorial_model_id`, cache key :804.
- Prompt strip (to remove): `prompts.py:49` / `:98` / `:110`.
- Env defaults: `deploy/elspeth-web.env:11` (`COMPOSER_MODEL`), `:17-18` (`COMPOSER_ADVISOR_ENABLED` / `COMPOSER_ADVISOR_MODEL`).

## 12. Related

- Case B fix `5deb34f78` (elspeth-e51216d305 CLOSED); follow-ups elspeth-dbc39dd367 (supersede primitive), elspeth-f936a78840 (fork refresh), elspeth-7dc234589a (stale/dup PT events).
- Composer correctness epic elspeth-e1ab67e55a; reactive advisor escape-hatch elspeth-7197f92457.
- Sibling specs to follow (deferred §2): field-contract checker, reciprocity enforcement, vague_term nudge.

---

## 13. Scope amendment (2026-06-07) — frontier-primary pivot; two-tier dropped

After grounding revealed there is **no tutorial-compose model seam** (tutorial compose runs through the same singleton `ComposerServiceImpl`; `tutorial_service.py` only handles the tutorial *run*), the operator changed strategy: **use a frontier model as primary everywhere** rather than cheap-primary + tutorial-override. Consequences, superseding the relevant parts above:

- **Two-tier wiring is DROPPED.** No `composer_tutorial_model` field, no per-compose model override, no tutorial-routing plumbing. (Supersedes Goal §2 "Two-tier model wiring" and D-2's "+ two-tier".)
- **Distinctness simplifies to a single primary:** `advisor_model ≠ composer_model` (normalized exact-string). (Supersedes D-1's `∉ {composer_model, composer_tutorial_model}` → just `≠ composer_model`; the forward-compat "both tiers" wording is moot.)
- **Concrete model pairing:** primary = Sonnet (`claude-sonnet-4-6`), advisor = Opus (`claude-opus-4-7`) — more capable than gpt-5.4-mini, validated via the staging tutorial harness. NOTE: both are Anthropic (same lineage) — this pairing tests *model*-distinctness (which exact-string enforces), not *vendor*-independence. Operator chose it knowingly (Opus is the strongest available reviewer).
- **Rationale broadens, design holds:** with a frontier primary the confident-failure rate drops, so deterministic checkpoints shift from *compensation for a weak model* to an **independence + backstop** mechanism ("the producer is never its own checker," regardless of producer strength). The early/end checkpoint architecture, the re-review loop, the budget model, and the config mandate are all unchanged.

**Retire-trigger reconciliation (clarified against reality):** the proactive security triggers share the *same* `request_advisor_hint` tool as the reactive one. So "retire reactive" = remove `reactive_validation_loop` from `ADVISOR_TRIGGER_VALUES` (tools/sessions.py:116-126) + the reactive-only validation branch (service.py:3179), and **keep** the `request_advisor_hint` tool for `proactive_security_safety` / `proactive_red_listed_plugin`. Removing `composer_advisor_enabled` (advisor now mandatory) drops the tool *filter* (service.py:3039-3040) — the tool is always present — and collapses the advisor-disabled prompt-strip surface (`_strip_advisor_content` becomes dead).

**Grounding-confirmed governance facts (de-risk the plan):**
- `config.py` is NOT in the tier-model allowlist; its `@model_validator`s are not flagged → no allowlist churn for the new validator (uses direct attribute access + explicit raise).
- WebSettings has no `contracts/config` alignment test; the contract is the `ComposerSettings` Protocol (protocol.py:640), mypy-enforced → removing `composer_advisor_enabled` = remove field (config.py:83) + protocol property (protocol.py:673), type-checked.
- Blast radius of removing `composer_advisor_enabled`: src — app.py:387/397 (boot-probe gate + telemetry), tool_batch.py:659 (defense-in-depth branch), service.py:3021/3039 (build_messages arg + tool filter), prompts.py (strip surface). Tests — test_dispatch_arms_characterization.py, test_app.py, test_compose_loop_persistence.py, test_advisor_tool.py. Deploy — elspeth-web.env:17.
- New budget field `composer_advisor_checkpoint_max_passes` is DISTINCT from the existing `composer_advisor_max_calls_per_compose` (config.py:85, which remains the hard ceiling across all advisor calls).

**Budget-model correction (supersedes §7's "early + end + re-reviews" wording):** the EARLY pass fires on the empty→non-empty pipeline transition (structurally ≤ once per session) and does **NOT** consume `composer_advisor_checkpoint_max_passes`. That counter governs **END-gate passes only** (review + re-reviews). Reason: a shared counter would let the early pass starve the end re-review (default 2 → early eats 1 → end gets 1 → a flagged end gate fails closed with no repair). End-only counting preserves the D-4 re-review loop. (Caught in plan self-review 2026-06-07.)
