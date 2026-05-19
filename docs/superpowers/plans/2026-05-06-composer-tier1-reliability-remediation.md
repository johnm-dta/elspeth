# Composer Tier 1 reliability remediation — agent tasking

**Date issued:** 2026-05-06
**Issuing investigation:** `docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md` (READ THIS FIRST)
**Branch base:** `RC5-UX` (current HEAD `19317366`)
**Filigree epic:** `elspeth-1d3be32a8a` (P0, type=epic, labels: `effort:m`, `source:agent`, `changelog:fixed`)
**Recipient:** any agent capable of editing Python + markdown + bash, running the local RGR harness, and restarting `elspeth-web.service` on this host.

---

## 1. Mission

Land the five items in **Tier 1** of the program-of-work derived from the 2026-05-06 RGR investigation. Each item is small, low-risk, and on a different reliability axis (sampling determinism, harness test-net, skill placement, skill examples, JSON-Schema hints). The cumulative goal is to lift the composer's hard-GREEN reliability on the canonical URL→download→line-explode scenario from the current baseline (~33% on a 9-run cohort) to **≥66% hard-GREEN on a fresh 6-run cohort**, measured under deterministic sampling.

You are NOT being asked to start Tier 2 work in this ticket. If a Tier 2 candidate becomes obviously easier than expected, file an observation via `mcp__filigree__observe` and continue with Tier 1. Do not scope-creep.

---

## 2. Mandatory pre-reading

Read these in order before starting:

1. **`docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md`** — full investigation. §1 (problem), §3 (empirical baseline), §4 (root cause), §7.1–§7.4 (the items you're shipping), §8 (anti-patterns — explicitly rejected approaches you must not re-propose), §10 (code anchors).
2. **`CLAUDE.md`** — esp. the data-trust tiers, "no-legacy-code" policy, "defensive programming forbidden" rules, and audit primacy.
3. **The skill file you're editing:** `src/elspeth/web/composer/skills/pipeline_composer.md`. Read it end-to-end once. It is 1031 lines and you must understand the existing structure before moving sections around.
4. **Memory entries** — surface via auto-memory:
   - `project_composer_harness_state.md` — skill-loading mechanics; `elspeth-web.service` MUST be restarted after every skill edit because the skill is `@lru_cache`'d at module import.
   - `feedback_no_tests_for_skill_prompts.md` — do NOT write unit tests asserting on skill markdown content.
   - `project_staging_deployment.md` — staging on this host is a source-checkout systemd/Caddy deploy; restart via `systemctl restart elspeth-web.service`, not `scripts/deploy-vm.sh`.
   - `feedback_correctness_beats_performance.md` — frame Tier 1 as a correctness fix, not a perf tradeoff.
   - `feedback_locked_in_buggy_expectations.md` — if existing tests fail after item 1 lands, the tests pinned the bug; update them, do not revert the fix.

---

## 3. Required skill invocations

Invoke each at the appropriate moment. Do not skip:

- **`superpowers:using-superpowers`** at conversation start (always).
- **`superpowers:writing-plans`** before you start coding, to lay out the per-item commit plan.
- **`pipeline-composer`** before editing the skill — gives you the conceptual model the skill is teaching the LLM.
- **`logging-telemetry-policy`** before adding the `temperature` / `seed` audit fields in item 1 — confirm primacy: this is audit (Landscape) not telemetry, not logging.
- **`superpowers:verification-before-completion`** before claiming any item complete.

If you find a fact in the investigation note that contradicts the live code, **trust the live code**, update this tasking via comment on the parent epic, and continue. The investigation was written off a snapshot at commit `19317366`; the line numbers may have drifted by a few lines under your branch but should not have moved structurally.

---

## 4. Exit criteria (measurable)

You may close the parent epic only when ALL of the following hold:

1. **All five items land in source.** A single PR per item is preferred; a single squashed PR with five commits is acceptable. Do not ship item 1 alone and then stop.
2. **Re-baseline measurement.** Run `evals/composer-rgr/run_scenario.sh` six times on a fresh deploy AFTER item 1 has landed but BEFORE items 2–5 have landed. Record the per-run verdicts. This is the *post-temperature-fix-only baseline*. Required for §6's gate decision.
3. **Final cohort.** After all five items have landed, run `evals/composer-rgr/run_scenario.sh` six times again on a fresh deploy. Record the per-run verdicts.
4. **Final cohort hard-GREEN ≥ 4/6** (66%). If the cohort is below 4/6, do NOT close the epic. File a Tier 2 candidate via `mcp__filigree__observe` referencing the cohort numbers and stop. Do not start writing more skill prose to chase the number — §8.6 of the investigation explicitly warns this is unproductive while platform-side fixes are pending.
5. **Audit sidecar verified.** Pull one row from `recorder.llm_calls` (or whatever the schema names the table — confirm at code-read time) for one of the cohort runs. Verify `temperature=0.0` and `seed=42` are persisted as recorded fields. If the columns don't exist on the recorder, you must add them — this is part of item 1.
6. **No skill-prose unit tests added.** Project memory `feedback_no_tests_for_skill_prompts.md`. The RGR harness is the regression net; markdown grep is theatre.
7. **Each Tier 1 item documented** in the parent epic via `mcp__filigree__add_comment` — one comment per item, citing the file:line of what changed, the diff size, and the hard-GREEN delta if you measured one between sub-items (optional; only the post-1 and final cohorts are required).

---

## 5. Sequencing rules

**Order is not negotiable for items 1 and 2.** Items 3, 4, 5 may be done in any internal order or in parallel, but must all land before the final cohort.

```
item 1  →  re-baseline cohort (6 runs)  →  items 2,3,4,5 (any order)  →  final cohort (6 runs)
```

The reason: every impact estimate in the investigation §7 was measured against a noisy 33% baseline that was temperature-variance-dominated. Until item 1 lands, the impact of items 3/4/5 cannot be cleanly attributed. The post-1 cohort gives you the new baseline against which the *cumulative* effect of items 2–5 is measured.

**Items 2–5 should not be measured individually** — the per-item RGR signal is too weak relative to sampling noise on a 6-run cohort. Cumulate them.

---

## 6. The five items

### Item 1 — Set `temperature=0.0` and `seed=42` on composer LLM calls; record both as audit fields

**Why first:** investigation §4.4, §7.1. The single highest-impact change in the entire program. Cheapest item AND biggest single uplift.

**Files & anchors (verify at edit time — line numbers may drift):**
- `src/elspeth/web/composer/service.py` around line 1695 (`_call_llm`) and line 1718 (`_call_text_llm`). Both call `_litellm_acompletion` with no `temperature` parameter.
- The audit recorder for LLM calls — find it from `service.py` (search for the call that persists each LLM round-trip; investigation §10 names `recorder.llm_calls`). Confirm whether `temperature` and `seed` are existing columns or need to be added.

**Change:**
1. Pass `temperature=0.0, seed=42` to both `_litellm_acompletion` call sites.
2. Add (or extend) the audit-sidecar persistence so `temperature` and `seed` are recorded per call. If new columns are required, follow the project's DB-migration policy (memory: `project_db_migration_policy.md` — no Alembic; the operator deletes the old DB, you change the schema declaratively).
3. Do NOT also pass `tool_choice="auto"` — the investigation notes the explicit form, but it's the default and adds churn. Skip it.

**Verification:**
- Restart `elspeth-web.service`.
- Run the RGR harness six times. Capture all six verdicts. This is your re-baseline cohort for §4 exit criterion 2.
- Pick one run, query the `recorder.llm_calls` row (or equivalent), confirm `temperature=0.0` and `seed=42` persist.
- If the post-1 cohort hard-GREEN is already ≥ 4/6: still proceed with items 2–5. The exit criteria require all five to land.
- If the post-1 cohort is *below* the prior baseline (i.e., the temperature change made things worse), STOP and add a comment to the epic. Do not proceed. This would be evidence that the OpenRouter/litellm path silently rejects `seed` or `temperature` and we'd be measuring something else.

**Anti-patterns:**
- Don't read `temperature` / `seed` from settings/env — hardcode for now. Configurability is a Tier 2 concern.
- Don't add a `try/except` around the litellm call — defensive programming forbidden (CLAUDE.md). If the provider rejects the kwargs, we want to know.

---

### Item 2 — Repair `evals/composer-harness/` preflight endpoints

**Why:** investigation §2.1. Off-§7 but cheap and infrastructural. Restores the multi-turn regression net for any future RGR work.

**Files & anchors:**
- `evals/composer-harness/lib/preflight.sh` — the `--doctor` preflight. Currently uses `/api/login` and `/api/catalog`.
- The composer web routes — search `src/elspeth/web/` for the actual auth and catalog endpoints. The investigation states they have moved to `/api/auth/login` and split-per-type catalogs (`/api/catalog/sources`, `/api/catalog/transforms`, `/api/catalog/sinks`). Verify the live route table; do not trust the investigation if reality has moved on.

**Change:**
1. Update the preflight URL constants to the current endpoints.
2. If the catalog is now split per-type, the preflight should hit at least one of each (sources/transforms/sinks). One 404 still fails the preflight.
3. Run `evals/composer-harness/run_harness.sh --doctor` (or whatever the doctor entry-point is named — confirm at edit time). Confirm it exits 0.

**Verification:**
- `--doctor` passes against `https://elspeth.foundryside.dev`.
- Do not run the full 15-fixture sweep. That's Tier 3 work.

**Anti-patterns:**
- Don't refactor `lib/preflight.sh` while you're in there. Surgical URL swap only.

---

### Item 3 — Move "Connection Model" section to top of skill

**Why:** investigation §7.2. Mechanical cut/paste; reduces context distance to the most-violated rule.

**Files & anchors:**
- `src/elspeth/web/composer/skills/pipeline_composer.md`
- "Connection Model" section is currently around line 156. The TERMINATION GATE is around line 39. Re-find both via grep at edit time.

**Change:**
- Cut the "Connection Model" section as it stands today. Paste it immediately after the TERMINATION GATE block and before whatever section currently follows the GATE.
- Do not edit the contents while moving. Do not add a "see also" cross-ref at the old location — the no-legacy-code rule means the section just moves.
- If there are any in-skill references to "Connection Model" by section name, they should still resolve (it's the same section, in a new place).

**Verification:**
- Restart `elspeth-web.service` (skill is `@lru_cache`'d).
- Smoke-test by spot-checking the new prompt structure: log in to staging, open a fresh session, send "what's a connection?" — confirm the model answers from the moved section's content. (This is a smoke test, NOT a regression test.)

---

### Item 4 — Add wiring repair few-shot examples to the Connection Model section

**Why:** investigation §7.3. Two examples cover the empirically-observed failure shapes (input='source' wrong, sink_name doesn't match upstream on_success).

**Files & anchors:**
- `src/elspeth/web/composer/skills/pipeline_composer.md`, inside the now-relocated Connection Model section (post-item-3).

**Change:**
Add a new subsection titled "Wiring repair examples" (or similar). Two examples, each a triplet:
1. Broken `set_pipeline` call (JSON snippet).
2. The `preview_pipeline` error message that would result (verbatim format from `graph.py` — `No producer for connection 'X'. Available connections: Y.`).
3. Fixed JSON snippet, with a one-line explanation of what changed.

The two examples should cover:
- **Example A:** consumer node sets `input: "source"` (treating "source" as a node-id pseudo-name). Fix: change to whatever the source's `on_success` actually publishes.
- **Example B:** sink's `sink_name` doesn't match the publishing upstream's `on_success`. Fix: align the strings.

**Verification:**
- Restart `elspeth-web.service`.
- No isolated RGR run — bundle this with the final cohort.

**Anti-patterns:**
- Do NOT extend beyond two examples. The investigation §8.6 caps prompt-prose marginal value past iteration 3.
- Do NOT add a third example covering fork/route/coalesce — that's Tier 3 (per-failure-mode scenarios).

---

### Item 5 — Enrich JSON Schema descriptions on `input` / `on_success` / `sink_name`

**Why:** investigation §7.4. Each tool call sees better hints; effect is per-call.

**Files & anchors:**
- `src/elspeth/web/composer/tools.py`
  - `set_pipeline.nodes[].input` — investigation cites line 707. Currently `{"type": "string"}` with no description.
  - `upsert_node.input` — line 441; has `"Input connection name."` already; enrich.
  - `upsert_edge` and `set_output` — find the equivalent `from_connection` / `to_connection` / `sink_name` fields and apply the same treatment.

**Change:** for each of the four tools and each relevant field, replace the bare `{"type": "string"}` with:

```python
{
    "type": "string",
    "description": (
        "Connection-name string this node consumes. MUST equal the value of "
        "some upstream's on_success (or routes value, or on_error) field. "
        "Not the upstream node's id. Example: if source.on_success='raw', "
        "the next node sets input='raw'."
    ),
    "examples": ["raw", "fetched_text", "scored_rows"],
}
```

Adapt the description per field role (consumer vs. producer vs. sink). The `examples` array should give realistic connection names for the field's role.

**Verification:**
- Restart `elspeth-web.service`.
- Confirm via the OpenAPI / tools-list endpoint (or by reading the live tool-definition response) that the descriptions surface in the actual tool schemas served to the model.

**Anti-patterns:**
- Don't change field types or required-ness. Descriptions and examples only.
- Don't introduce a `$ref` / shared definition refactor. Inline the descriptions even though they repeat. The duplication is fine; pulling it into a shared schema is a Tier 2 cleanup.

---

## 7. Verification protocol (final cohort)

After all five items have landed and `elspeth-web.service` has been restarted:

```bash
cd /home/john/elspeth
for i in 1 2 3 4 5 6; do
  evals/composer-rgr/run_scenario.sh "final-${i}"
done
```

Record verdicts. Compose a summary in the epic comment:

```
Final cohort (post all Tier 1):
  final-1: GREEN
  final-2: GREEN
  final-3: RED-soft (passivity tail)
  final-4: GREEN
  final-5: GREEN
  final-6: RED-hard (schema construction)

Hard-GREEN: 4/6 (66%) — exit criterion met.
Soft-RED: 1/6 — Tier 2 (anti-tail-offer post-processor or strict-mode) candidate.
Hard-RED: 1/6 — Tier 2 (strict mode + better preflight diagnostics) candidate.
```

Soft-RED and hard-RED counts inform Tier 2 prioritisation but do NOT block epic closure as long as hard-GREEN ≥ 4/6.

---

## 8. Anti-patterns to avoid (project-specific)

These are forbidden — re-proposing them in PR or comment will be rejected without further discussion. Reasons in the investigation §8 and CLAUDE.md.

1. **No skill-prose unit tests.** No `test_pipeline_composer_skill_contains_X`. The RGR harness is the test.
2. **No defensive programming.** No `try/except` around the litellm call to silently fall back to no-temperature. If the provider rejects, we want the crash and the audit trail.
3. **No backwards-compat shims.** When you add `temperature` / `seed` columns, change every call site in the same commit. No "if column exists" guards.
4. **No `git stash`.** Use a worktree or a branch — the project has banned stash due to repeated data loss.
5. **No fine-tuning, no RAG, no model swap, no "fix the schema to accept what the model writes".** All explicitly rejected in §8.
6. **No more skill prose past these five items.** §8.6 caps marginal value of additional rules until platform-side Tier 2 fixes land.
7. **No silent failure handling.** Audit primacy applies — every failure is an event in Landscape, not a swallowed exception.

---

## 9. Reporting

- Update epic status to `in_progress` when you start (`mcp__filigree__update_issue` with `status=in_progress`).
- Comment per item completed (`mcp__filigree__add_comment`).
- Final summary comment with the cohort table.
- Close the epic only when all of §4 holds.
- If you decide to abort or escalate, leave the epic `in_progress`, post a comment with the failure mode, and tag the user. Do not close to `wont_fix` without explicit operator approval — the bug being remediated is in-the-wild user-reported.

## 10. Out of scope (do not do)

- Tier 2 items (§7.5–§7.10 of the investigation): strict JSON Schema mode, runtime preflight rewrite, in-loop retry hint, mutation-result wiring echo, `set_linear_pipeline` shortcut, derive-input-from-edges. File observations if the work overlaps but do not implement.
- Cross-model RGR (claude-opus-4, gpt-5 base) — Tier 3.
- Per-failure-mode RGR scenarios (fork/route, content-safety, RAG) — Tier 3.
- Multi-turn RGR sweep — Tier 3 (item 2 unblocks this but the sweep itself is separate work).
- Anything in §9.6 of the investigation (raw_assistant_content exposure on next turn).

---

**End of tasking.** When you are ready to start, invoke `superpowers:writing-plans` and post your per-commit plan as a comment on the parent epic before touching code.
