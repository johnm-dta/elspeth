# Composer Tier 1.5 final hardening — agent tasking

**Date issued:** 2026-05-07
**Working window:** 24 hours from epic claim. Hard cap.
**Predecessor work:** Tier 1.5 epic `elspeth-08fafb9873` and its cohort report at `docs/composer/evidence/composer-tier1.5-cohort-report-2026-05-06.md`. Six commits on `RC5-UX` (`f0139356` … `ea4607bb`) currently unpushed.
**Filigree epic:** `elspeth-682aa0c91e` (P0, type=epic, parent=`elspeth-08fafb9873`, grandparent=`elspeth-1d3be32a8a`, labels: `effort:l`, `source:agent`, `changelog:fixed`).
**Recipient:** any agent willing to operate under a hard 24-hour clock with explicit must-have / nice-to-have prioritisation.

---

## 1. Mission

Three deliverables, in priority order. The 24-hour clock is real — if you cannot finish all three, finish in the order given and post a partial cohort report at the cap. **Do not extend past 24 hours; do not trade depth for breadth.**

1. **§7.7 anti-anchor: behavioural proof or fix.** The hint-fire count across the prior cohort was 0. Either (a) construct an adversarial scenario that should trigger the anchor predicate and verify the hint actually fires end-to-end, or (b) determine the predicate is wrong, fix it, and re-verify. Outcome: a captured run with `anti_anchor` evidence in the audit trail OR a code change with the verified-firing run.
2. **Step B scenarios: re-author and re-cohort.** Three scenarios (`fork-and-route`, `aggregation-content-safety`, `rag-text-llm`) currently fail not because the composer is broken but because their prompts reference `/tmp` paths the composer correctly refuses. Re-author each prompt to use a workspace-allowable input shape (see §6 for the contract). Re-run a 6-run cohort per scenario. Outcome: real fan-out signal across pipeline shapes.
3. **Final cohort report supersedes the Tier 1.5 cohort report.** New file: `docs/composer/evidence/composer-tier1.5-final-cohort-report-2026-05-07.md`. Comment posted to **this epic** plus a cross-link comment on `elspeth-08fafb9873`. Outcome: a single artefact the operator can use to make the merge call.

You are NOT being asked to:
- Land any further Tier 2 items beyond §7.7 (already in tree). No §7.5 strict mode, no §7.6 better diagnostics, no §7.8 mutation echo, no §7.9 set_linear_pipeline. If the §7.7 validation surfaces a related bug, file it; do not fix.
- Re-run the persona-driven multi-turn sweep. The simplified-driver Step A is good enough for the merge-readiness call; persona-driven is Tier 3.
- Make the merge decision. The operator does that against your final report.

---

## 2. Mandatory pre-reading

Read in order before starting. Limit pre-reading to ≤45 minutes; the clock is running.

1. **`docs/composer/evidence/composer-tier1.5-cohort-report-2026-05-06.md`** — the cohort report you're superseding. Pay attention to:
   - The "Tier 2 fix shipped this session" section on §7.7 (what was claimed)
   - The Step B "/tmp scenarios" section (line 105-118, what failed)
   - The "Reproducibility / artefacts" section (where existing artefacts live)
2. **`docs/composer/evidence/composer-tier1.5-step-c-diagnosis-2026-05-06.md`** — the Step C diagnosis from commit `f0139356`. The classification is the basis for §7.7's design. Confirm the predicate matches the diagnosis.
3. **`src/elspeth/web/composer/anti_anchor.py`** — the §7.7 implementation, 119 lines. Read it once end-to-end. Note the trigger predicate, the suppression rule from `3abe4d70` (don't clear anchor on discovery-tool successes), and the hint-text payload.
4. **`tests/unit/web/composer/test_anti_anchor.py`** and **`tests/unit/web/composer/test_compose_loop_anti_anchor.py`** — the existing 506 lines of unit/integration tests. Understand what's already covered before adding more.
5. **One Tier 1 GREEN run** — pick any of `evals/composer-rgr/runs/final-*/` that scored GREEN, look at the `messages.json` and `state.json`. Specifically: what shape does the source-blob URL take? What did the model put in `source.options`? This tells you the workspace input contract for Step B.
6. **`docs/superpowers/plans/2026-05-06-composer-tier1.5-test-surface-expansion.md`** — the predecessor tasking. Re-read §11 ("Out of scope"). The "no Tier 2 implementation" rule applied last time and a §7.7 commit landed anyway. **Do not repeat that pattern in this ticket.** §7.7 is grandfathered because it's already in the tree; nothing else gets in.

---

## 3. Required skill invocations

- **`superpowers:using-superpowers`** at session start.
- **`superpowers:writing-plans`** before starting deliverable 1, to lay out a 24-hour budget across deliverables.
- **`superpowers:systematic-debugging`** for §7.7 validation — this is a "does it fire?" diagnostic, not a fix.
- **`pipeline-composer`** before authoring scenario prompts — you must understand what input shapes the composer accepts.
- **`superpowers:verification-before-completion`** before claiming any deliverable complete.

---

## 4. Time budget (hard cap)

24 hours total. Suggested allocation:

| Slice | Deliverable | Cap |
|-------|-------------|-----|
| Pre-reading | §2 list | 0:45 |
| Deliverable 1 | §7.7 behavioural proof or fix | 6:00 |
| Deliverable 2 | Step B re-author + re-cohort | 8:00 |
| Re-baseline existing scenario | confirm 12/12 still holds post-deliverable-1 | 1:00 |
| Deliverable 3 | Final cohort report | 3:00 |
| Buffer / posting / review | | 5:15 |

**At T+18h, stop adding capability. At T+22h, stop running cohorts. At T+24h, post whatever you have.** A partial-but-honest report at the cap beats a complete-but-late one. If you blow through the cap, post a comment explaining what you ran out of time for and which deliverable to pick up next.

---

## 5. Deliverable 1 — §7.7 behavioural proof or fix

**Why first:** the §7.7 commits (`28e8e4ef`, `3abe4d70`) landed in the tree without behavioural verification. Hint-fire count was 0 across the entire prior cohort. The operator has decided to keep §7.7 in the merge candidate, so we need to prove it actually works — not just that the unit tests pass.

### 5.1 Validation approach

Two-phase validation. Both must complete.

**Phase A — Synthetic predicate test.** Write a focused integration test that:
1. Constructs a `ComposeLoopState` (or whatever the predicate consumes) with handcrafted context simulating an anchored-retry pattern (≥N consecutive identical `set_pipeline` calls failing on the same connection-name error).
2. Invokes the predicate directly (or the loop step that consumes it).
3. Asserts the hint is emitted into the next-turn message list.

This proves the predicate fires when the conditions match. It does NOT prove the conditions ever match in production.

The existing `tests/unit/web/composer/test_compose_loop_anti_anchor.py` may already do this. Confirm. If yes, document; if not, add.

**Phase B — Adversarial RGR scenario.** This is the load-bearing test. Write a new RGR scenario explicitly designed to make the model anchor:
- Pick a pipeline shape with non-obvious wiring (e.g., a connection-naming case where the producer's `on_success` value is unusual — `"raw_data_v2"` rather than the typical `"raw"`).
- Phrase the prompt in a way that gives the model a wrong intuition for the connection name. E.g., "Take the input file, decode it, and write the result" — the words "input" and "result" might bias the model toward `node.input = "input"` and `sink_name = "result"`, which won't match.
- Run a 6-run cohort.
- Query the audit DB via `evals/lib/decode_tools.py` (the helper from commit `f0139356`) for any session that retried `set_pipeline` ≥3 times.
- Inspect: did the anti-anchor hint appear in the LLM context window after retry N? Does the next attempt deviate from the prior attempts (success) or persist (failure)?

If the hint fires and the model breaks the loop: §7.7 works, document the evidence.

If the hint fires and the model ignores it: §7.7 mechanism works but payload is weak — file a Tier 2 follow-up observation; do not fix in this ticket.

If the hint never fires under deliberately anchor-seeking conditions: §7.7's predicate is broken. **Fix it.** Likely culprits: trigger threshold too high, comparison too strict (e.g., requiring byte-exact match of arguments_canonical when the model varies whitespace), or the predicate runs after the wrong loop hook. Read `anti_anchor.py` carefully. Adjust the predicate. Re-run Phase B.

### 5.2 Concrete artefacts

- New scenario directory: `evals/composer-rgr/scenarios/anti-anchor-stress/` with its own `scenario.json` plus a `README.md` explaining the adversarial design.
- New evidence file: `docs/composer/evidence/composer-77-validation-2026-05-07.md` with the audit-DB extract showing the hint firing.
- If a fix landed: a single commit `fix(composer): §7.7 predicate <symptom> — <evidence>` with the test that would have caught the original behaviour.

### 5.3 Anti-patterns

- **Don't claim §7.7 works because the unit tests pass.** The unit tests exercise the helper function. The helper function might not be reachable from production code paths under realistic conditions.
- **Don't loosen the predicate to make it fire more often.** If anchoring isn't happening, the predicate being stricter than necessary is a *feature*, not a bug. Better to document "predicate is correct; current model+temperature combination doesn't anchor" than to land a change that fires the hint on every retry.
- **Don't add a new Tier 2 fix because §7.7 is weak.** §7.7's payload quality is out of scope; only its mechanism (does it fire under correct conditions?) is in scope.
- **Don't widen scope to test §7.7 across other models.** gpt-5-mini at temperature=0.0 is the deploy target; that's what we're validating.

---

## 6. Deliverable 2 — Step B scenarios re-author and re-cohort

**Why:** the prior Step B authored prompts referencing `/tmp` paths the composer correctly refuses (Trust Tier 3 boundary working as designed). The 18 RED runs from those scenarios are scenario-author failures, not composer failures. The fan-out goal — validate generalisation beyond URL-download-line-explode — was not met.

### 6.1 Establish the workspace input contract first

Before re-authoring any prompt, learn what the composer accepts. This is non-negotiable; the prior agent skipped this step and authored 18 useless runs.

Two sources of truth:

1. **A working Tier 1 GREEN run.** Look at the `state.json` in any of the `evals/composer-rgr/runs/final-*-GREEN/` directories from the original cohort. Inspect `source.options`. Whatever URL/path/blob shape is in there is one valid input.
2. **The composer's source-acceptance code.** Search `src/elspeth/web/composer/` for the validator that rejected `/tmp`. Find the actual allowlist or pattern. Document it in `docs/composer/evidence/composer-77-validation-2026-05-07.md` (or a separate notes file) under a "Workspace input contract" heading — this saves the next agent a re-derivation.

Likely shapes (verify, don't assume):
- Session-uploaded blob URIs created via the `create_blob` tool (this is what the URL-download-line-explode scenario uses)
- HTTP(S) URLs the `web_scrape` plugin will fetch
- Inline literal text via the `text` source plugin's `text` option

`/tmp`, absolute filesystem paths, and arbitrary local paths are NOT acceptable. The Tier 3 boundary is correct.

### 6.2 Re-author each scenario

Same three scenarios. Same scoring criteria. New prompts that respect the contract.

**`fork-and-route`** — pipeline that loads a small CSV via inline text source or a session blob, routes rows to two named sinks based on a row predicate (e.g., a status column).
- Input: small CSV literal in the prompt, OR instruct the model to call `create_blob` with a CSV literal you provide.
- Don't reference `/tmp/customers.csv`; either inline the CSV or specify a blob.
- GREEN criteria unchanged (gate node + two distinct sinks).

**`aggregation-content-safety`** — pipeline that loads text rows, runs each through Azure content safety, aggregates approved.
- Input: inline text source with 3-5 example messages in the prompt itself.
- Don't reference `/tmp/messages.csv`.
- GREEN criteria unchanged (content-safety transform + threshold + aggregation + sink).
- **Note:** Azure content-safety requires credentials. Confirm staging has them configured. If not, this scenario will RED on environment, not on composer logic — file an observation and skip the scenario.

**`rag-text-llm`** — pipeline that takes URLs, scrapes each, runs through LLM transform, writes summaries.
- Input: 2-3 HTTPS URLs inline in the prompt (e.g., simple Wikipedia article URLs that won't change).
- Don't reference `/tmp/urls.txt`.
- GREEN criteria unchanged (web_scrape + llm + non-empty system_prompt + sink).
- **Note:** llm transform requires upstream LLM credentials. Confirm staging has them.

### 6.3 Cohort each scenario

6 runs per scenario. Capture per-run verdicts. If any scenario's environment is broken (missing credentials, missing plugins), document and skip — do not score those runs as composer failures.

Total budget: 18 runs × ~$0.50 ≈ $9. Plus a fresh 6-run re-baseline of `url-download-line-explode` to confirm 12/12 still holds post-deliverable-1: 6 runs × $0.50 ≈ $3. Total ≈ $12.

### 6.4 Anti-patterns

- **Don't tune scenario prompts until they pass.** The point is to measure naive performance on a realistic prompt. If the prompt is reasonable and the composer fails, that's signal — not a prompt bug.
- **Don't bundle scenarios.** Same rule from Tier 1.5. Each scenario gets its own directory.
- **Don't extend `score.py` with scenario-specific logic.** If the generic `red_criteria`/`green_criteria` framing isn't expressive enough, the scenario is wrong, not the scorer.
- **Don't add a fourth scenario.** Three was the budget; finishing them well > finishing four poorly.

---

## 7. Re-baseline existing scenario

After deliverable 1 lands (any §7.7 fix, plus the new adversarial scenario), re-run the existing `url-download-line-explode` scenario for 6 runs. **Confirm 12/12 still holds across the prior 6 + new 6.**

If it doesn't — i.e., a §7.7 fix you landed in deliverable 1 broke the working scenario — STOP. Revert the fix. The prior 6/6 was real; trading a real win for a hypothetical anchor-fire is a bad trade.

---

## 8. Deliverable 3 — Final cohort report

New file: `docs/composer/evidence/composer-tier1.5-final-cohort-report-2026-05-07.md`. Supersedes the prior 2026-05-06 report. Mirror via comment to **this epic** with a cross-link comment on parent `elspeth-08fafb9873`.

### Required sections

1. **Headline numbers** — single table with hard-GREEN counts across all four scenarios + multi-turn pathology check (re-stated, not re-run).
2. **§7.7 validation evidence** — Phase A test result + Phase B audit-DB extract showing hint fires under adversarial conditions, OR the fix that was required + the verified-firing run.
3. **Step B fan-out** — per-scenario per-run verdicts. Aggregate hard-GREEN rate across the now-valid 18 runs.
4. **Re-baseline** — `url-download-line-explode` 12-run combined verdict (prior 6 + new 6).
5. **Merge-readiness assessment** — single paragraph: against the operator's "reliably green across the board" gate, does the evidence support shipping `RC5-UX` to main? Be specific. If the answer is qualified ("yes, with caveats X and Y"), state the caveats.
6. **What's NOT in scope for this report** — explicit list of known unknowns the operator should be aware of: persona-driven multi-turn (Tier 3), cross-model behaviour (Tier 3), anything else surfaced during the work.
7. **24-hour budget reconciliation** — actual time spent per deliverable vs. the §4 cap. Over-runs documented; under-runs explain what additional capacity went to.

### Format requirements

- Tables for cohort numbers; prose for evidence.
- Audit-DB extracts as code blocks, truncated under 100 lines.
- Reproducibility section: the exact command lines to re-run each cohort.
- "Verification trail" pinned at the bottom — for the operator to spot-check before signing off.

---

## 9. Project-specific anti-patterns (re-asserted)

1. **No skill-prose unit tests.**
2. **No defensive programming in any new helper.** Audit primacy.
3. **No backwards-compat shims.** If you reorganise scenario directories, do it in one commit.
4. **No `git stash`.** Worktree or branch.
5. **No fine-tuning, no RAG over the skill, no model swap.**
6. **No Tier 2 implementation beyond §7.7.** Repeat: this is the third tasking that has said this. The §7.7 grandfather is a one-time exception.
7. **No silent failure handling.**
8. **No closing the parent epic.** Operator owns the merge call.
9. **No pushing to origin without explicit operator approval.** CLAUDE.md git-safety policy.

---

## 10. Out of scope (do not do)

- Persona-driven multi-turn sweep re-run (Tier 3).
- Cross-model RGR (Tier 3).
- §7.5 strict JSON Schema, §7.6 runtime preflight rewrite, §7.8 mutation echo, §7.9 set_linear_pipeline, §7.10 derive-input-from-edges. If §7.7 validation surfaces a real bug in one of these areas, file an observation. Do not fix.
- Refactoring any harness, scorer, scenario layout, or skill.
- Adding more than three scenarios in deliverable 2.
- Pushing the branch.
- Closing the parent Tier 1 epic or the Tier 1.5 epic. Both stay `in_progress` until the operator acts on the final cohort report.

---

## 11. If you cannot finish in 24 hours

Stop at the cap. Post a comment to this epic with:
- Which deliverables completed
- Which deliverable was in flight at the cap and what state it's in
- A single-paragraph recommendation: "ship now with caveat X" / "extend by N hours for Y" / "abort and revert §7.7 because Z."

The operator decides the next step. Do not extend yourself.

---

**End of tasking.** Invoke `superpowers:writing-plans` and post your 24-hour budget plan as a comment on this epic before starting deliverable 1.
