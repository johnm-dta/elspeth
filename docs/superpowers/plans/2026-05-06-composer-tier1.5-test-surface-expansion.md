# Composer Tier 1.5 — pre-merge test surface expansion (agent tasking)

**Date issued:** 2026-05-06
**Issuing investigation:** `docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md` (esp. §3.5, §4.1, §9.2, §9.3, §11)
**Predecessor work:** Tier 1 epic `elspeth-1d3be32a8a` — landed across commits `51bfe46c`, `1ca34527`, `a3eede98`, `9251ff5f`, `fa1de04f`, `8735cabb`. Final cohort 5/6 (83%) hard-GREEN on URL-download-line-explode scenario at single-turn.
**Predecessor tasking:** `docs/superpowers/plans/2026-05-06-composer-tier1-reliability-remediation.md` (read for shape; this doc is the same template).
**Filigree epic:** `elspeth-08fafb9873` (P0, type=epic, parent=`elspeth-1d3be32a8a`, labels: `effort:m`, `source:agent`).
**Recipient:** any agent capable of running bash harnesses, writing JSON scenario files, querying SQLite, and producing a structured outcome report. Three of the four steps are parallelisable across separate agent sessions.

---

## 1. Mission

**Characterise the composer's reliability surface across (a) multi-turn behaviour, (b) failure-mode diversity beyond URL-download-line-explode, and (c) the residual 1/6 hard-RED in the existing Tier 1 final cohort. Produce a single combined cohort report. Do not implement Tier 2 fixes — only measure and diagnose.**

This is **measurement work**, not implementation work. The output is data + diagnosis. The data informs which Tier 2 item ships next; that decision is operator-driven, not agent-driven.

The merge gate on the `RC5-UX` branch is "reliably green across the board" (operator's words, 2026-05-06). Tier 1 demonstrated reliability on **one** scenario at **one** turn-depth. The current branch state cannot meet the merge gate without broader evidence. This ticket produces that evidence.

You are NOT being asked to:
- Implement any Tier 2 item (§7.5–§7.10 of the investigation)
- Add more skill prose to chase a number
- Decide whether the branch ships
- Close the Tier 1 epic

You ARE being asked to:
- Run measurements (Steps A, B)
- Diagnose the residual RED (Step C)
- Land one small reusable diagnostic helper (`evals/lib/decode_tools.py`) as part of Step C
- Produce a single structured cohort report comment on this epic

---

## 2. Mandatory pre-reading

Read in order before starting:

1. **`docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md`** — full investigation. Especially §3.5 (cumulative results table), §4.1 (the audit-DB tool-sequence diagnostic — load-bearing for Step C), §9.2 (multi-turn coverage gap), §9.3 (scenario coverage gap), §11 (review recommendations including the `decode_tools.py` helper request).
2. **`docs/superpowers/plans/2026-05-06-composer-tier1-reliability-remediation.md`** — the Tier 1 tasking. You're following the same shape and the same anti-patterns apply.
3. **`docs/superpowers/plans/2026-05-06-composer-tier1-reliability-implementation.md`** — the per-commit implementation plan from Tier 1 + the four "reality corrections vs the original tasking" up front. Important: Tier 1 moved the harness preflight from `evals/composer-harness/lib/preflight.sh` to **`evals/lib/preflight.sh`** (commit `1ca34527`). The shared `evals/lib/` location is therefore the convention for cross-harness utilities. Step C's `decode_tools.py` helper goes there.
4. **The existing scenario:** `evals/composer-rgr/scenario.json` and `evals/composer-rgr/run_scenario.sh`. You will be cloning the scenario shape, not the harness, in Step B.
5. **CLAUDE.md** — the audit primacy rules apply to your `decode_tools.py` helper; it must read from the audit DB without mutating, and any failures it surfaces must crash loud, not return defaults.

---

## 3. Required skill invocations

- **`superpowers:using-superpowers`** at conversation start.
- **`superpowers:writing-plans`** before starting any step, to lay out per-step commit shape.
- **`superpowers:systematic-debugging`** before Step C — it's a diagnosis task, not a fix task. Resist the urge to propose a fix mid-diagnosis.
- **`pipeline-composer`** before Step B — you need to understand the plugin contracts to author scenarios for fork/aggregation/RAG shapes correctly.
- **`superpowers:verification-before-completion`** before claiming the report comment complete.

If multi-agent parallelisation is used (see §5), each spawned subagent invokes `using-superpowers` independently at its own session start.

---

## 4. Exit criteria (measurable)

You may close this epic only when ALL of the following hold:

1. **Step A delivered.** Multi-turn `evals/composer-harness/` 15-fixture sweep run end-to-end against staging. Per-fixture verdicts captured. The originally-reported `e7d42525-bd73-4838-968c-647ea73cce98` pathology (passivity-as-stalling at turn 2 after user pushback) explicitly checked-for in the verdicts: either reproduces or demonstrably does not.
2. **Step B delivered.** Three new RGR scenarios added under `evals/composer-rgr/scenarios/<name>/` (or sibling dirs — the agent decides the layout, but each scenario must be its own directory so verdicts don't muddle). 6-run cohort recorded per scenario.
3. **Step C delivered.** A specific RED session from the post-Tier-1 cohort identified, audit-DB tool sequence decoded, root-cause assigned to one of {§7.6 needs-better-diagnostics, §7.7 needs-CoT-reset, both, neither + new failure class}. `evals/lib/decode_tools.py` helper landed and used to produce the diagnosis.
4. **Cohort report.** Single comment posted to this epic with:
   - Per-scenario per-cohort hard-GREEN / soft-RED / hard-RED counts (table)
   - Multi-turn sweep result summary (per-fixture or aggregate verdict counts)
   - The Step C diagnosis with audit-DB evidence (decoded tool sequence as a code block)
   - **An evidence-based Tier 2 recommendation** — which §7.x item to ship next, with the cohort numbers as justification
5. **No Tier 2 implementation has been started.** If you find yourself wanting to fix the diagnosed failure mid-task, file an observation via `mcp__filigree__observe` and stop. Tier 2 is a separate ticket.

The cohort report comment is the **deliverable**. It's the document the merge decision rests on, alongside Tier 1's outcome.

---

## 5. Sequencing rules

Steps A, B, C are **independent and parallelisable**. The total wall time can be cut to ~1 day if dispatched across three sessions; serially it's ~2-3 days due to the multi-turn harness's 2-3h sweep duration.

```
                  ┌── Step A (multi-turn sweep, 2-3h harness wall time)  ──┐
START ──→ pre-reading ──┼── Step B (3 scenarios + cohorts, ~half day)        ──┼─→ Cohort report ──→ DONE
                  └── Step C (diagnosis + decode_tools.py, ~half day)    ──┘
```

If you parallelise: spawn **subagents only via `superpowers:dispatching-parallel-agents`**. Each subagent gets a single step and a focused tasking. The parent session synthesises results into the cohort report.

If you don't parallelise: do them in any order. **Order doesn't matter for measurement-only work.**

---

## 6. Step A — Multi-turn persona-harness cohort

**Why:** investigation §9.2. The originally-reported failure was multi-turn (turn 2 after user pushback). Single-turn RGR has not tested this pathology. Tier 1 commit `1ca34527` repaired the multi-turn harness's broken preflight; this is the first time it can run end-to-end since the fix.

**Files & anchors:**
- `evals/composer-harness/` — the harness root.
- `evals/lib/preflight.sh` — repaired in commit `1ca34527`. Confirm the repair is in your working tree before starting.
- The harness has a `--doctor` mode and a full-sweep entry-point. Find them in the harness README or the run scripts; do not assume names.
- `.env` controls model selection; current deploy uses `openrouter/openai/gpt-5.4`. Do NOT change this.

**Procedure:**
1. **Doctor first.** Run the harness in `--doctor` (or equivalent preflight) mode against `https://elspeth.foundryside.dev`. If it doesn't pass, STOP and add a comment to the epic. Don't paper over the failure — the whole point of the Tier 1 commit was to make this work.
2. **Full sweep.** Run the 15-fixture sweep. Wall time ≈ 2-3 hours per investigation §2.1; cost ≈ $8-12. Budget for this; do not abort partway because the cost surprises you. Record the run start/end timestamps.
3. **Capture per-fixture verdicts.** The harness should produce a structured output. If it doesn't — that's an observation worth filing — extract verdicts manually from the run dirs.
4. **Specifically check for two patterns** in the verdicts:
   a. **Passivity-as-stalling at turn 2.** Look for any fixture where the assistant's turn-2 response contains "If you want, I can" / "Should I proceed" / "Do you want me to" with zero tool calls in that turn. This is the originally-reported failure mode. **If it reproduces, the post-Tier-1 deploy has not actually fixed the user-reported bug.**
   b. **Soft-RED tail-offers.** Cases where the pipeline is functionally valid but the assistant ends with a polite tail-offer ("If you want, I can also adjust the output…"). Investigation §3.4 noted iteration 3 didn't eliminate these; this is the first multi-turn measurement of whether they're better, worse, or unchanged post-Tier-1.

**Verification:**
- Per-fixture verdict table captured.
- The two patterns above explicitly checked-for and reported.
- Cost actually paid recorded in the report (for future-budgeting).

**Anti-patterns:**
- Do NOT skip the multi-turn sweep because it's slow or expensive. The single-turn RGR cannot substitute for it. The captured failure session was multi-turn.
- Do NOT extend the sweep with new fixtures while you're in there. Use the existing 15 as-is. New fixtures are Tier 3 work.
- Do NOT propose a fix for any pattern you find. File observations; the report's Tier 2 recommendation is a downstream synthesis.

---

## 7. Step B — RGR scenario fan-out

**Why:** investigation §9.3. One scenario cannot validate "reliably green across the board." Three new failure-mode scenarios test whether the §3.5 GREEN rate generalises beyond URL-download-line-explode.

**Files & anchors:**
- `evals/composer-rgr/scenario.json` — the existing scenario; clone its shape.
- `evals/composer-rgr/run_scenario.sh` — accepts a scenario directory or path. Confirm at edit time how multi-scenario invocation is currently structured. If it's currently single-scenario-only, you may need a tiny wrapper (`run_all_scenarios.sh`) that loops — but **do not refactor `run_scenario.sh` itself**.
- Existing scoring in `evals/composer-rgr/score.py` — already generalises off the scenario's `red_criteria` / `green_criteria`.

**Procedure:**

1. **Choose the layout.** Either (a) move the existing scenario into `evals/composer-rgr/scenarios/url-download-line-explode/scenario.json` and add the three new scenarios as siblings, or (b) keep the existing scenario at the root and add new scenarios as `evals/composer-rgr/scenarios/<name>/`. Either is acceptable; pick the one that minimises churn. Document the choice in the report.

2. **Author three scenarios.** Each is a `scenario.json` with `opening_prompt`, `red_criteria`, `green_criteria`, and any other fields the existing scenario uses.

   **Scenario `fork-and-route`:**
   - Prompt asks for a pipeline that loads CSV rows and routes them to two named sinks based on a row predicate (e.g., `status == "active"` → `active_sink`, else → `inactive_sink`).
   - Tests gate-routing path; exercises `routes` field on a transform node.
   - GREEN criteria: `is_valid: true`, exactly one source, at least one gate node, two sinks with distinct `sink_name` values.

   **Scenario `aggregation-content-safety`:**
   - Prompt asks for a pipeline that loads text rows, runs each through Azure content safety with a threshold, and aggregates approved rows into a single output.
   - Tests aggregation transform + content-safety transform; exercises options-block schema for both.
   - GREEN criteria: `is_valid: true`, content-safety transform present with a threshold value, aggregation transform present, sink wired to aggregation output.

   **Scenario `rag-text-llm`:**
   - Prompt asks for a pipeline that takes a list of URLs, scrapes each, runs the scraped text through an LLM transform with a summarisation prompt, and writes the summaries to a JSON sink.
   - Tests `web_scrape` + `llm` plugin schemas in combination — investigation noted these were only partially exercised by the GREEN runs.
   - GREEN criteria: `is_valid: true`, web_scrape + llm both present, llm has a non-empty system_prompt or template, sink consumes llm output.

3. **Cohort each scenario.** Run 6 runs per new scenario. Total: 18 runs ≈ $9. Plus 6 fresh runs of the existing scenario (post-Tier-1 baseline confirmation): 24 runs total ≈ $12.

4. **Capture results.** Per-scenario per-run verdicts. Aggregate into the report.

**Verification:**
- Each new scenario has its own directory and verdicts file.
- 6 runs per scenario completed; per-run verdicts captured.
- The existing scenario re-cohorted for comparability.

**Anti-patterns:**
- Do NOT bundle all four scenarios into one master scenario file. §11 explicit recommendation.
- Do NOT tune scenario prompts to make them pass. The point is to measure naive performance.
- Do NOT extend `score.py` with scenario-specific logic. If the generic `red_criteria`/`green_criteria` framing isn't expressive enough for a scenario, the scenario is wrong, not the scorer.
- Do NOT attempt to fix any scenario that scores low. That's the entire point of measuring.

---

## 8. Step C — Diagnose the residual RED + land `decode_tools.py`

**Why:** investigation §11 review recommendation. The remaining 1/6 RED in the Tier 1 final cohort was filed as observation `elspeth-obs-fcac7c99ec` referencing §7.6 OR §7.7 — those are different fixes. We need evidence to choose between them.

**Files & anchors:**
- `evals/composer-rgr/runs/final-*/` — the gitignored run dirs from the Tier 1 final cohort. The RED session ID is in `final-<n>/session_id.txt` for whichever final-<n> was the RED.
- `data/sessions.db` — SQLite. Table `chat_messages`. JSON column `tool_calls`.
- `evals/lib/` — convention from Tier 1's commit `1ca34527`. `decode_tools.py` lands here.
- Investigation §4.1 — the SQL pattern and decoded-sequence example. Replicate the analysis.

**Procedure:**

1. **Find the RED session.** Walk `evals/composer-rgr/runs/final-*/` — match the `scoring.json` to find the verdict, capture `session_id.txt` for the RED. If those run dirs have been cleaned up, run a fresh 6-run cohort first and capture the next RED.

2. **Land `evals/lib/decode_tools.py`.** Small, focused. One function:
   ```python
   def decode_tool_sequence(db_path: str, session_id: str) -> list[dict]:
       """Return decoded chat_messages.tool_calls for the given session.

       Each dict has: ts, role, tool_name, arguments_canonical (parsed JSON),
       result_summary (truncated). Crashes loud if the DB or session is missing.
       """
   ```
   - Read-only. Open the SQLite with `mode=ro` URI to make it physically impossible to mutate.
   - No defensive `try/except`; if the DB is missing or the session has zero rows, raise. (CLAUDE.md offensive programming rule.)
   - One CLI entry-point: `python -m evals.lib.decode_tools <db_path> <session_id>` prints the decoded sequence to stdout as pretty JSON.
   - **Do NOT generalise.** No "decode_tool_sequence_to_html" or "filter_by_tool_name" features. One function, one CLI entry-point, that's it.

3. **Run the diagnosis.** Use the new helper to dump the RED session's tool sequence. Inspect each `set_pipeline` (or other mutation) call's `arguments_canonical`. Inspect the runtime-validator error result that came back. Determine:
   - Did the model iterate (≥2 mutation attempts)? → If no, the failure is pre-iteration (passivity / refusal), not convergence.
   - Were the failed attempts identical or did they drift? → Identical = anchored loop = §7.7 (CoT reset hint). Drifting = at least trying = §7.6 (better diagnostics).
   - Did the runtime validator's error message contain enough information for a competent reader to repair the call? → If no = §7.6. If yes but the model didn't act on it = §7.7.
   - Did the model surrender to text-only? → Investigation §1 captured-session pattern (no tool calls at all) is a different mode from convergence-after-attempts.

4. **Assign a category.** One of:
   - **§7.6 needs-better-diagnostics** — runtime errors were unclear; difflib suggester was unhelpful; structured wiring diagnostic would have given the model a fix path.
   - **§7.7 needs-CoT-reset** — runtime errors were clear; model anchored on its prior wrong attempts in context; chain-of-thought reset hint after N retries would break the loop.
   - **Both** — multiple turns, different shapes per turn.
   - **Neither + new failure class** — the failure is something else entirely (e.g., tool-side bug, plugin-options crash, a captured failure shape that wasn't seen in the original investigation). If this, capture full evidence.

**Verification:**
- `evals/lib/decode_tools.py` lands in a single commit, with a one-line README entry pointing at it. Tests required: at least one unit test against a tiny in-memory SQLite fixture proving the function returns the expected structure for a known input. (This IS code, unlike the skill markdown — it gets normal TDD.)
- The diagnosis is documented in the cohort report with the decoded sequence as a code block (truncated if huge — keep it under 100 lines in the comment).

**Anti-patterns:**
- Do NOT attempt to fix the failure. This is diagnosis. Filing a Tier 2 candidate observation IS the conclusion.
- Do NOT skip the unit test on `decode_tools.py`. Code requires tests; the no-skill-prose-tests rule is about skill markdown, not Python.
- Do NOT generalise `decode_tools.py` past one function + one CLI entry-point. Keep it small enough that the next investigation can extend it without inheriting a half-baked framework.

---

## 9. Cohort report shape

Single comment posted to this epic via `mcp__filigree__add_comment`. Suggested structure:

```markdown
# Composer Tier 1.5 cohort report — 2026-05-06

## Multi-turn persona harness (Step A)

| Fixture | Verdict | Notes |
|---------|---------|-------|
| <fixture-1> | GREEN/RED | … |
| … | | |

Aggregate: N/15 GREEN, M/15 soft-RED, K/15 hard-RED.

Originally-reported `e7d42525-…` passivity pattern: REPRODUCED / NOT REPRODUCED.
- Evidence: …

Soft-RED tail-offer count post-Tier-1: N (vs ~2/9 pre-final-cohort iteration 3).

Cost paid: $X. Wall time: H hours.

## Single-turn RGR scenarios (Step B)

| Scenario | hard-GREEN | soft-RED | hard-RED |
|----------|-----------|----------|----------|
| url-download-line-explode (re-cohort) | a/6 | b/6 | c/6 |
| fork-and-route                        | a/6 | b/6 | c/6 |
| aggregation-content-safety            | a/6 | b/6 | c/6 |
| rag-text-llm                          | a/6 | b/6 | c/6 |
| **AGGREGATE**                         | A/24 | B/24 | C/24 |

## Tier 1 residual RED diagnosis (Step C)

RED session: `<sid>`
RGR run dir: `evals/composer-rgr/runs/final-<n>/`

Decoded tool sequence:
```
<truncated decoded sequence — keep under 100 lines>
```

Diagnosis: §7.6 / §7.7 / both / new-class — <one-line justification>.

Evidence:
- Did the model iterate? <yes/no, count>
- Were attempts identical or drifting? <…>
- Was the runtime error informative? <…>
- Did the model give up or push through? <…>

## Tier 2 recommendation

Based on the cohort numbers above, recommend shipping <§7.x> next, because:
- <evidence point>
- <evidence point>

## Merge-readiness assessment

The aggregate hard-GREEN rate is N/(15+24) = X%. <Operator>'s "reliably green across the board" gate <does/does not> appear to be met.

If the operator chooses to ship `RC5-UX` now, the residual failure rate is X%. If the operator chooses to hold the merge, the recommended Tier 2 fix is §7.x with estimated impact of ±Y percentage points.

This is a measurement report; the merge decision is operator-driven.
```

You may adapt the structure but cover all five sections.

---

## 10. Project-specific anti-patterns

Re-asserted from Tier 1's anti-pattern list because they still apply:

1. **No skill-prose unit tests.** No `test_pipeline_composer_skill_contains_X`. (`feedback_no_tests_for_skill_prompts.md`.)
2. **No defensive programming in `decode_tools.py`.** Crashes are informative; silent defaults are evidence tampering. (CLAUDE.md.)
3. **No backwards-compat shims.** If you reorganise `evals/composer-rgr/` for Step B, do it in one commit.
4. **No `git stash`.** Worktree or branch.
5. **No fine-tuning, no RAG over the skill, no model swap.**
6. **No more skill prose in this ticket.** Tier 1.5 is measurement and diagnosis. Skill prose adjustments belong in a future Tier 2 ticket if the diagnosis warrants them.
7. **No silent failure handling in any helper you write.** Audit primacy.
8. **Don't close this epic without the operator's review of the cohort report.** The merge decision is operator-driven; closing prematurely strips the operator's decision context.

---

## 11. Out of scope (do not do)

- **Any Tier 2 implementation.** §7.5 strict JSON Schema, §7.6 runtime preflight rewrite, §7.7 in-loop retry hint, §7.8 mutation echo, §7.9 set_linear_pipeline, §7.10 derive-input-from-edges. File observations if your diagnosis points at one of these, but do not implement.
- **Cross-model RGR (claude-opus-4, gpt-5).** Tier 3. The current deploy is gpt-5.4; that's our reliability target.
- **New persona-harness fixtures.** Use the existing 15 in Step A.
- **Refactoring `run_scenario.sh` or `score.py`.** Surgical scenario additions only.
- **`evals/lib/decode_tools.py` past one function + one CLI entry-point.** Resist the urge to build a "diagnostic toolkit."
- **Filing the merge PR or marking the Tier 1 epic closed.** The cohort report is the deliverable; the merge decision is the operator's.

---

**End of tasking.** When you start, invoke `superpowers:writing-plans` and post your per-step plan as a comment on this epic before beginning Step A / B / C.
