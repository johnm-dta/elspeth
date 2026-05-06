# Tier 1.5 §7.6 Option C — Empty-State Recovery Nudge: Feasibility & Design

**Date:** 2026-05-07
**Status:** DESIGN-ONLY (recommendation: **CONDITIONAL-GO**)
**Author:** scope agent (composer-tier1.5-hardening)

## TL;DR

After the two structural fixes shipped this session (EdgeContractError 0b176800, empty-state passthrough 2e2c7bc8), the residual 20% RED rate manifests as: model produces no-tool-call reply on a structurally-empty state. Option C proposes that, instead of finalizing, we inject a one-shot system-attributed "you surrendered, try this minimal shape" nudge and let the loop run one more turn.

The shape is sound and orthogonal to the existing §7.7 anti-anchor (which catches byte-identical retries). But we currently have **zero post-fix REDs to characterise the surrender mode against** — every audit-DB capture used as evidence here predates the empty-state passthrough fix. Recommending nudge content design against pre-fix transcripts is guesswork.

**Recommendation: CONDITIONAL-GO. Condition: capture 2–3 post-fix REDs first, then revisit content design.** Pure NO-GO is also defensible on demo-window grounds.

---

## 1. Where this hooks in the compose loop

`src/elspeth/web/composer/service.py` `_compose_loop`:

- **Loop entry per turn:** lines ~1105–1115. LLM call, then test `assistant_message.tool_calls`.
- **No-tool-call branch:** lines 1118–1139. This is the surrender point — the model produced prose only, control flows directly to `_finalize_no_tool_response`.
- **`_finalize_no_tool_response`:** lines 780–874. Already detects `_state_is_structurally_empty(state)` (line 850) and routes to the model-prose passthrough.

**Hook site:** the existing `_state_is_structurally_empty` branch at line 850 is exactly where Option C diverges. Pre-Option C: pass through prose, finalize. Option C: if not yet nudged in this session, append a `role="user"` nudge to `llm_messages`, increment a turn counter, `continue` the outer loop, and call the LLM once more. If that second turn also produces no-tool-call → fall through to existing passthrough.

**Budget interaction.** The nudge-driven turn must charge SOMETHING to prevent infinite recursion if the cap is set wrong:

- `composition_turns_used` — wrong; this turn produced no mutation.
- `discovery_turns_used` — wrong; no discovery tool ran either.
- A new third counter `recovery_turns_used` with cap of 1 (or simply the per-session single-fire guard), enforced both by the tracker AND by checking `composition_turns_used + discovery_turns_used < (max_composition + max_discovery)` so it can't accidentally extend a run that already exhausted budget.

The cleanest control flow: a one-shot tracker (mirror `AntiAnchorTracker` shape — `EmptyStateRecoveryTracker.has_fired() / record_fire()`) plus a sanity check that there is at least 1 unit of budget remaining in either counter. If the model arrived at empty-state-no-tool-call after exhausting both budgets, we don't get a free turn — fall through to existing passthrough. This is structurally important because the loop's existing budget exhaustion paths (lines 1736, 1781) raise `ComposerConvergenceError`, not finalize, so we must not be re-entering after that.

**The hook is NOT inside `_finalize_no_tool_response`.** That method is also reached from the budget-exhaustion bonus call at line 1760, where re-entering the loop is unsafe. Option C goes at the no-tool-call site at line ~1129 in `_compose_loop`, BEFORE calling `_finalize_no_tool_response`, with a guard that we are not arriving from the bonus-call path.

## 2. Nudge content design

Audit-DB inspection of three pre-fix REDs (sessions 2cf59016, 12f061d9, 29ef178e in `/home/john/elspeth/data/sessions.db`):

- All three involve the same task class: scrape URLs → LLM summarise → JSON sink.
- Dominant failure: `web_scrape` options shape — repeated `Field required` for `schema`, `url_field`, `content_field`, `fingerprint_field`, `http`, then once those are present the model gets `schema` shape wrong (4-key dict vs expected single-key `{field_name: type}`).
- Secondary failure: `set_pipeline` missing `source.options` and `outputs[0].options` (the model thinks `csv` source and `json` sink need no options block).
- **Anti-anchor never fired in any of these sessions.** They are *drift retries*, not byte-identical retries — the model changes its args every attempt, so the `(tool_name, arguments_hash)` deque never converges on three identical entries. This is the architectural gap Option C addresses: §7.7 catches stuck-on-same-payload, Option C catches stuck-after-many-different-payloads.

**Two content shapes considered:**

1. **GENERIC (default).** "The pipeline you described could not be assembled. Fall back to a minimal shape — one CSV source, one pass-through (or one transform with all-required fields satisfied), one JSON sink — and once that validates, evolve from there. Re-read `get_plugin_schema` for any plugin whose options you are unsure about, and confirm the `source.options` and `outputs[N].options` blocks are present even when the plugin appears option-less."

2. **SPECIFIC (rejected for now).** Inspect the most recent failed `set_pipeline` arguments and surface a canonical example for each plugin used. This lands closer to the actual failure but requires a content map per plugin and is brittle if the residual REDs aren't web_scrape/llm/json — which we don't know yet because no post-fix REDs have been captured.

**Decision: GENERIC.** Reasons: (a) plugin-specific nudges don't generalise, (b) the GENERIC shape is honest about what we don't know — the model has surrendered, the cheapest recovery is a minimal-shape pipeline that demonstrably validates, (c) within the 30-minute cap on this point, GENERIC is robust regardless of which plugin actually broke. The minimal-shape framing also dovetails with the existing `_compose_empty_state_message` suffix, which already tells the user the pipeline is empty — Option C is the pre-finalize attempt to fix it before we tell the user.

## 3. Budget and loop-trap design

**Single-fire guard, mirror §7.7 shape.** New `EmptyStateRecoveryTracker` per `compose()` call, 100 lines or fewer, lives next to `anti_anchor.py`:

```python
class EmptyStateRecoveryTracker:
    def __init__(self) -> None:
        self._fired = False
    def has_fired(self) -> bool: return self._fired
    def record_fire(self) -> None: self._fired = True
```

The tracker persists for the lifetime of one `compose()` call (one HTTP request). A second `compose()` call from the user — i.e., a new turn after the operator types something — gets a fresh tracker and CAN nudge again. This is correct: each compose call corresponds to one user message, and a user re-prompting after a surrender deserves the same recovery affordance.

**Why one shot per compose call, not per session?** Multi-fire within a single compose is recursive and unsafe — if the model ignores the nudge and produces another no-tool-call empty-state reply, firing again gives identical content and we loop forever (or re-enter every turn until budget exhausts). Single-fire converts the failure mode to "nudge once, fall through to passthrough if it didn't work" which is bounded and predictable.

**Loop-trap proofs:**

- **Trap A:** Model receives nudge → no-tool-call again. Tracker has fired, second arrival at the no-tool-call branch falls through to passthrough. Bounded.
- **Trap B:** Model receives nudge → calls discovery tools → eventually no-tool-call again. Same path: tracker fired, falls through. Bounded.
- **Trap C:** Budget exhausted before nudge fires (via the bonus-call path at line 1760). The bonus call already returns directly to `_finalize_no_tool_response`; Option C must NOT inject a nudge from this path, only from the primary no-tool-call path at line 1129. Distinguished by a parameter/flag passed through the call chain — easiest is to keep the nudge-injection logic outside `_finalize_no_tool_response` entirely, in `_compose_loop` directly.

## 4. Failure modes (not just success path)

| # | Mode | Outcome |
|---|------|---------|
| 1 | Nudge fires → model retries → STILL fails (drift), produces no-tool-call again. | Tracker fired, fall through to existing empty-state passthrough. Worst case: one extra LLM round-trip and the user sees the same prose-with-suffix message. |
| 2 | Nudge fires → model "ignores it" (produces prose only, immediately). | Same as #1. The "extra turn" produces nothing useful. Cost paid, no benefit. |
| 3 | Nudge fires → model retries → succeeds. | RED → GREEN. The hoped-for case. |
| 4 | Nudge content is misaligned with the actual failure. | Likely outcome is #1 or #2 — model can't act on nudge content that doesn't match its problem. We won't know unless we see post-fix REDs and check whether the nudge was relevant. **Detection signal: post-Option-C cohort scorer red_reasons should be inspected for "nudge fired, still RED" cases — needs a telemetry attribute or audit marker.** |
| 5 | Nudge fires from the budget-exhaustion bonus-call path (Trap C above). | Re-entering the loop after budget exhaustion violates the bonus-call contract. **Mitigation: gate the nudge-injection on the source path in `_compose_loop`, not in `_finalize_no_tool_response`.** Mechanical: only inject from line ~1129, never from line ~1760. |
| 6 | Nudge content surfaced verbatim into chat history could leak info. | Generic content has no operator data; safe. If we ever go SPECIFIC, must re-check secret hygiene per the boundary contract on line 762. |

## 5. Risk assessment

**Does the nudge harm first-pass-success runs?** No. Trigger conditions are conjunctive: (a) `assistant_message.tool_calls is None`, AND (b) `_state_is_structurally_empty(state)`. A successful run has `state.source is not None or state.nodes or state.outputs`, failing condition (b). A successful no-state run (e.g., explicit "tell me what's available" task) ends with prose AND empty state but is not REDuced — but here the user got an honest answer to a question that didn't request a build. Option C would inject a "build something minimal" nudge that's tone-deaf to the actual question.

This is a non-zero false-positive risk. Mitigation: require a *prior mutation attempt* to have happened in this compose() call — check `recorder.invocations` for any non-discovery tool invocation. If the operator never asked for a build (no `set_pipeline` etc. attempted), don't nudge. Easy to add, materially reduces false-positive surface.

**Cost on production:** each fired nudge adds one LLM round-trip (~30–90s, ~$0.005–0.02 at current model pricing depending on context size). Option C only fires on already-failed sessions, so the marginal cost is bounded by the RED rate. At 20% RED × 1 extra LLM call, that's a ~20% cost overhead on the failed slice = ~4% across all runs. Acceptable if conversion rate is non-trivial.

**Demo-window opportunity cost:** implementing + testing + verifying Option C is 1–3h. With ~6h demo window remaining and the empty-state passthrough already shipping a graceful failure path, the opportunity cost is real. If Option C converts even 10pp of RED → GREEN it's worth it; if it converts < 5pp it isn't.

**Biggest hole — and the conditional in the recommendation:** all evidence above is from pre-fix sessions. The post-fix 20% RED rate is uncharacterized. The captured REDs are sessions where the OLD synthesizer replaced the model's prose with Pydantic noise — Option C's framing (model surrenders honestly, we nudge) is grounded in those captures, but post-fix the model now sees the empty-state passthrough... wait, it doesn't, because the passthrough fires AT FINALIZE TIME for the operator. The model still surrenders the same way. So the failure mode at the model layer is unchanged by 2e2c7bc8 — only the operator-visible message changed. **This means pre-fix captures are still representative of what the model is doing**, even if not of what the operator sees. The conditional weakens but does not vanish: we should still see ≥ 1 post-fix RED to confirm the model behaviour is what we think it is.

## 6. Verification design

**Detection power for a 20%→10% RED-rate swing:** with a baseline RED rate `p₀ = 0.20` and target `p₁ = 0.10`, a one-sided test at α=0.05 with 80% power needs roughly N ≈ 199 runs per arm, OR N ≈ 50 runs as a single-arm comparison against a known baseline (the 8/10 cohort). Single-arm against a fixed baseline is the realistic option. At ~60s per run and concurrency-1, that's ~50 minutes of staging time plus operator review.

**Within demo budget?** Yes if we descope to a 25-run cohort (~25 min, weaker signal, ~60% power for the same effect size). No if we want statistical confidence — the demo window can't absorb 50+ run verification AND any other in-flight work.

**Cohort plan if GO:**

1. Run 25–50 staging tasks against the rag-text-llm seed task (the dominant pre-fix RED scenario).
2. Tag each run by `nudge_fired: bool` (audit attribute on the recovery tracker, or check `chat_messages` for the system-injected role="user" message with a stable marker like `[ELSPETH-RECOVERY-NUDGE]`).
3. Compare RED rate within `nudge_fired=True` (treatment) vs same-cohort pre-Option-C historical baseline.
4. **Critical secondary metric:** of the `nudge_fired=True` runs, what % are still RED? This tells us whether the nudge was useful (low %) or theatre (high %). Without this, conversion-rate alone could be confounded by the model just having a better day.

**Verification overhead is the binding demo-window constraint.** Either the cohort is descoped (weaker signal) or another in-flight item gets pushed.

## Recommendation: CONDITIONAL-GO

**Condition:** capture 2–3 post-fix REDs from the staging environment first (cheap — ~10 min of running rag-text-llm tasks + audit-DB inspection). Confirm the surrender shape matches the pre-fix captures. THEN implement Option C with GENERIC nudge content.

**Why not HARD GO:** the nudge content is currently inferred from pre-fix transcripts. Empty-state passthrough is a UI-layer fix, not a model-layer fix, so the pre-fix evidence is *probably* still valid — but "probably" is not "verified", and shipping nudge content tuned to a misdiagnosed failure burns a verification cohort and complicates rollback.

**Why not NO-GO:** the architectural gap is real (anti-anchor doesn't catch drift retries, ~20% of staging runs still surface this failure), the implementation is small and well-isolated (mirror anti-anchor pattern), and the existing empty-state passthrough is a graceful fallback when nudge fails — so the downside is bounded.

**Time and cost estimate (GO branch):**

- Capture post-fix REDs: 15 min wallclock + 10 min review.
- Implement Option C: 1.5h (tracker module ~50 LOC, service.py hook ~30 LOC + guard against bonus-call path, audit/telemetry attribute, type-checking, lint).
- Unit tests: 30–45 min (single-fire, fall-through trap A/B, bonus-call path NOT entered, no nudge when no prior mutation attempt).
- Verification cohort: 25–50 min staging time + 30 min cohort scoring.
- **Total: ~3–4h end-to-end. Within demo window budget if other in-flight work is descoped.**

If the demo window is tight and there is no other slack: **default to NO-GO**, ship empty-state passthrough as the floor and accept the 20% RED rate. The operator-visible failure surface is now graceful (model prose + system-attributed retry suggestion), which was the primary aim of §7.6.
