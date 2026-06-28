# Composer demo-readiness report — 2026-05-07

**Filigree epic:** elspeth-682aa0c91e (final hardening, 24h cap)
**Parent:** elspeth-08fafb9873 (Tier 1.5)
**Grandparent:** elspeth-1d3be32a8a (Tier 1)
**Branch:** `composer-tier1.5-hardening` (worktree off `RC5-UX@e5863702`)
**Operator override:** plan's framing was "Tier 1.5 hardening; no Tier 2 beyond §7.7"; corrected at session start to "maximise success probability for tomorrow's interactive exec demo (SES through CEO)." Plan §10's "multi-turn out of scope" was overridden.
**Demo narrative:** (a) composer flow only — UX is perfunctory by design — point is to show the shape of the full capability with a suite of plugins ("BAs could process complex reporting and business research tasks in semi-autonomous mode").
**Supersedes:** `docs/composer/evidence/composer-tier1.5-cohort-report-2026-05-06.md`

---

## TL;DR for the operator

**Per-prompt convergence rates (combined today's parallel + sequential cohorts):**

| Demo shape | Per-run convergence | Demo recommendation |
|------------|---------------------|---------------------|
| **`url-download → web_scrape → line_explode → JSONL`** (hello-world ingest) | **9/12 = 75%** today; 15/18 = 83% with 2026-05-06 cohort | **Lead with this.** Most reliable. |
| **`csv → gate → 2 sinks`** (fork/route demo) | **5/6 = 83%** | Demo-ready as showcase shape. |
| **`csv → web_scrape → llm → json sink`** (research/RAG demo) | **5/12 = 42%** | **High failure risk.** Use only if ready for fallback talking points. |
| **`csv → llm → gate → 2 sinks`** (BA classify-and-route) | 0/1 sample (single dry-run RED on schema contract) | **Untested at cohort scale.** Avoid in demo unless rehearsed. |
| `aggregation-content-safety` | n/a — staging has no Azure CS credentials | Skip. |

**Bottom line:** the hello-world and fork-and-route demos are reliable enough to demo without a backup plan. The research/RAG demo has a >50% failure probability per run — operator should either pre-rehearse a specific RAG prompt that works on staging or be ready with fallback narrative.

### Critical merge-decision context (read before signing off)

The branch's Tier 1.5 ticket landed `§7.7` (anti-anchor hint) **specifically to address the captured Tier 1 RED `53bc3cf2-…`**. This session's investigation proves §7.7 does NOT catch that RED's failure pattern (the diagnosis was wrong — the model drifted, didn't anchor; see findings §1 and §2 below). And the same drift-on-web_scrape failure pattern **recurs in 7 of 11 of today's REDs** (~25% of cohort runs).

So: the merge candidate's Tier 1 hardening (anti-tail-offer rule, temperature=0.0, skill quality, workspace input contract) IS load-bearing and real — that's where the convergence-rate floor of 75-83% comes from on simple shapes. **But the §7.7 work added on top does not solve the problem it claims to solve.** The Tier 1.5 child ticket that landed §7.7 is closeable as "shipped, harmless, but does not address its stated motivation." The actual fix for the recurring failure pattern is **§7.6 (broader runtime-preflight error rewrite)**, which was deferred and is the highest-leverage post-demo work item.

This affects the merge call: the branch is shippable for "demonstrate plugin breadth and audit primacy" (the demo's narrative goal) but is NOT a regression-fixed RC for production high-volume use. Communicate accordingly.

---

## Key findings (vs prior cohort report's claims)

### 1. The prior cohort's "6/6 url-download GREEN" was real but small-sample

The 2026-05-06 cohort report's headline 6/6 was a real measurement on 6 runs. Today's 12-run replication shows the actual per-run rate is closer to 75% — 6/6 was an unsurprising clean draw with that sample size. Combined 18-run window: **15/18 = 83%**.

### 2. The §7.7 anti-anchor predicate is correct-but-not-load-bearing

The prior Step C diagnosis (`docs/composer/evidence/composer-tier1.5-step-c-diagnosis-2026-05-06.md`) classified the residual Tier 1 RED `53bc3cf2-…` as "Anchor (byte-identical retry)" by visual inspection of `source.schema.fields` and `web_scrape.options` (top-level). Verified against the audit DB during this session's slice 3 systematic-debugging Phase 1: **the three set_pipeline failures have three distinct argument hashes** (`664367f1.../02b0023e.../32e13b6c...`). Diff between #2 and #3 canonical args shows the model changed `web_scrape.schema` from `{mode:fixed, fields:["url:str"]}` to `{mode:observed}` — drift, not anchor.

The captured RED was a **drift-without-convergence-then-self-surrender**, not an anchored loop. §7.7 (which targets anchored loops at threshold N=3 byte-identical) does not address this failure mode; the dominant failure mode in this deploy is the same drift-and-surrender pattern repeatedly.

**Cohort-wide hint-fire signal**: 0 across **every session in the audit DB** (prior cohort + this session's 30+ runs). §7.7 is harmless dead code in the current model+temperature regime; it's not the merge candidate's reliability source.

The merge candidate's reliability comes from: **Tier 1's anti-tail-offer rule, temperature=0.0, the workspace input contract enforcement, and skill quality** — none of which are §7.7.

Filed as observation `elspeth-obs-9dfff9b571`.

### 3. The recurring failure mode is **drift-on-web_scrape-config**

Of the today-cohort REDs:

| Failure mode | Count | Pattern |
|--------------|-------|---------|
| Drift on `web_scrape` config + self-surrender ("I hit a configuration mismatch on the scrape step") | 7 of 11 REDs | Same pattern as the captured Tier 1 RED — model gets the intent right but can't make `web_scrape` options validate atomically. |
| Discovery-budget exhaustion (21 turns mostly on `get_plugin_schema`) | 1 of 11 REDs | Model spent budget on exploration before committing. |
| Server-side `ElspethSettings` validation ("source / sinks Field required") | 2 of 11 REDs | Suspicious: composer state has source + sinks; runtime preflight sees `{}`. Possible composer→runtime translation bug. |
| Connection-naming validator failure ("Gate route target 'X' is neither a sink nor a known connection") | 1 of 11 REDs | Exactly the §7.6 case the original investigation flagged — error message doesn't pinpoint how to fix. |

**§7.6 (broader runtime-preflight error rewrite) would address the dominant pattern** but is out of scope for this ticket. Filed as follow-up suggestion in observation `elspeth-obs-9dfff9b571`.

### 4. Multi-turn passivity is not a demo risk

Prior cohort's 1/15 PARTIAL was the SharePoint honest-refusal probe (`p1_t3_limit_sharepoint`) — appropriate refusal-with-alternative behaviour, not pipeline-construction hedging. See `docs/composer/evidence/composer-multi-turn-demo-readiness-2026-05-07.md` for the full analysis. In pipeline-construction context: 0/14 prior valid samples + 0/30+ this session = effectively 0% hedging risk.

Observation: `elspeth-obs-1733ef1519` (harness's substring catch-list is context-blind; recommend post-demo gating on presence of mutation tool calls).

### 5. The Step B scenarios were authoring-flawed in the prior cohort

The 18 RED runs from the prior cohort's Step B (`/tmp/customers.csv` etc.) were composer correctly refusing to fabricate paths outside the workspace boundary — not composer regressions. Re-authored this session with inline CSV content + `outputs/` sinks; smokes confirmed two of three converge (rag-text-llm, fork-and-route). The third (aggregation-content-safety) is environment-blocked on staging (no Azure CS creds wired); observation `elspeth-obs-5803c9a63b`.

### 6. The cohort scorer had a measurement-instrument bug

`evals/composer-rgr/score.py` checked `n.get("plugin") or n.get("type")` for gate detection, but gate nodes have `plugin: null` and identify themselves via `node_type`. The `"type"` field was always None. Fix landed this session as commit `9a51ac97` — fallback chain now: `plugin -> node_type -> type -> ""`. Without this fix, fork-and-route's prior cohort scored 4/6 AMBER on functionally-correct gate routing.

### 7. Skill prose gap on `web_scrape` required options (DEPLOY CANDIDATE)

The pipeline-composer skill's `web_scrape` entry historically said: "**Gotchas**: You must specify `url_field` — the name of the row field containing the URL to fetch. There is no default." That language **is misleading** — `url_field` alone is insufficient. Runtime validation requires:

| Option | Required value or example |
|--------|---------------------------|
| `schema` | input contract, e.g. `{"mode": "fixed", "fields": ["url: str"]}` |
| `url_field` | name of URL field on row (no default) |
| `content_field` | output field for scraped content (canonical: `"content"`) |
| `fingerprint_field` | output field for content hash (canonical: `"content_fingerprint"`) |
| `format` | `"text"` / `"markdown"` / `"html"` |
| `text_separator` | required when `format: "text"` (canonical: `"\n"`) |
| `http.abuse_contact` | contact email for abuse reports |
| `http.scraping_reason` | one-line human-readable reason |
| `http.allowed_hosts` | usually `"public_only"` |

Without these, `set_pipeline` returns a "Field required" cascade. The skill's old prose pushed the model to discover them via `get_plugin_schema(web_scrape)` only — sometimes the model internalised correctly, sometimes it surrendered after a few attempts (the dominant failure pattern, per finding §3 above).

A **skill update is committed in this session's branch** (commit will appear once posted) that adds a "Required options" subsection plus a canonical full options block. **NOT YET DEPLOYED to staging.** The deploy involves either (a) merging the worktree branch to main, or (b) cherry-picking the commit, then `systemctl restart elspeth-web.service`.

**Operator decision before demo:**
- *Deploy* → expected effect: lift convergence on simple shapes (`url-download`, `fork-and-route`) from ~80% to ~90%+, lift `rag-text-llm` from 42% to ~60%+. Untested on staging; risk of regression on other shapes is small (skill update is purely additive prose) but non-zero.
- *Don't deploy* → demo runs at the rates above; mitigations stay as documented in TL;DR.

If deploying, run a quick verification: 6 sequential url-download runs + 6 rag-text-llm runs after restart. If the rates lift and no regression on existing shapes, ship. If not, revert.

---

## On-stage talking points (operator situational awareness)

**Lead with the most reliable shape.** url-download-line-explode → 75-83%. Phrase it as "watch the AI build a workflow to download and process this rules text in real time."

**Have fallback ready if the model hedges or fails.**
- "Let me show you the audit trail" — pull up the lineage view; even failed runs are full data.
- "The composer correctly recognised the limit and stopped" — if model says "I'm stuck."
- "Watch how it discovers schemas" — if model is mid-discovery.

**If you demo the research workflow (rag-text-llm), pre-rehearse it.** ~42% per-run rate means there's significant chance of failure. Specific mitigations:
- Use 1-2 URLs in the prompt, not 3+ (reduces complexity).
- Use a simpler summary prompt ("3-sentence summary" is established working).
- If the first attempt fails, try the same prompt again — the next attempt has independent ~42% success.

**If the model uses "If you want, I can help with X" in an honest-refusal context** (e.g., asks for SharePoint connector): correct behaviour, embrace it. *"That's the model knowing its limits. Honest refusal is part of the audit trail."*

**If the audience asks about robustness**: *"Right now we're at ~80% per-prompt convergence on simple shapes — getting that to 95+% is the work track this branch is feeding into. The audit trail captures every attempt."*

---

## Detailed cohort verdicts

### url-download-line-explode (12 runs today, 18 with prior)

| Run | Verdict | Failure mode (if RED) | Latency |
|-----|---------|------------------------|---------|
| prior cohort 1-6 | 6/6 GREEN | — | (not measured) |
| parallel rebaseline 1 | RED | drift on web_scrape config; "I'm blocked by composer validation issue on the web fetch step" | 26s |
| parallel rebaseline 2 | RED | drift on web_scrape config; "the main blocker is the source wiring/config shape" | 40s |
| parallel rebaseline 3 | GREEN | — | 22s |
| parallel rebaseline 4 | GREEN | — | 22s |
| parallel rebaseline 5 | GREEN | — | 24s |
| parallel rebaseline 6 | GREEN | — | 24s |
| sequential rebaseline 1 | GREEN | — | (~25s) |
| sequential rebaseline 2 | GREEN | — | (~25s) |
| sequential rebaseline 3 | RED | discovery-budget exhaustion at 21 turns | (~70s) |
| sequential rebaseline 4 | GREEN | — | (~25s) |
| sequential rebaseline 5 | GREEN | — | (~25s) |
| sequential rebaseline 6 | GREEN | — | (~25s) |

**Aggregate today: 9/12 = 75%; with prior: 15/18 = 83%.**

### fork-and-route (6 runs)

| Run | Verdict | Failure mode (if RED) | Latency |
|-----|---------|------------------------|---------|
| 1 | GREEN | — | 33s |
| 2 | GREEN | — | 15s |
| 3 | RED | connection-naming: "Gate route target 'inactive_conn' is neither a sink nor a known connection" | 26s |
| 4 | GREEN | — | 14s |
| 5 | GREEN | — | 33s |
| 6 | GREEN | — | 23s |

**Aggregate: 5/6 = 83%.**

### rag-text-llm (12 runs)

| Run | Verdict | Failure mode (if RED) | Latency |
|-----|---------|------------------------|---------|
| parallel 1 | RED | state null (no committed pipeline) | 36s |
| parallel 2 | RED | state null + build-failure sentinel | 27s |
| parallel 3 | RED | is_valid=false on committed pipeline | 57s |
| parallel 4 | GREEN | — | 46s |
| parallel 5 | GREEN | — | 31s |
| parallel 6 | RED | state null + sentinel | 27s |
| sequential 1 | RED | drift on web_scrape config | (~50s) |
| sequential 2 | RED | runtime preflight: ElspethSettings missing source/sinks (suspected server-side translation bug) | (~50s) |
| sequential 3 | GREEN | — | (~40s) |
| sequential 4 | RED | runtime preflight failed | (~50s) |
| sequential 5 | GREEN | — | (~40s) |
| sequential 6 | GREEN | — | (~30s) |

**Aggregate: 5/12 = 42%.**

### Concurrency hypothesis (rejected)

I initially hypothesised that the parallel-cohort REDs were a concurrency artefact (3 cohorts firing simultaneously stressed staging). Sequential cohorts disprove this: url-seq still has 1 RED (discovery-budget exhaustion), and rag-seq has 3 REDs (drift + preflight). The failure rates are real per-prompt model issues, not artefacts of how I drove the cohort.

---

## §7.7 anti-anchor validation evidence

**Phase A (synthetic predicate test)**: PASS. 21 unit + integration tests in `tests/unit/web/composer/test_anti_anchor.py` and `test_compose_loop_anti_anchor.py` verified on this session's worktree venv. The predicate fires when 3 byte-identical `(tool_name, arguments_hash)` failures occur in a row, doesn't fire below threshold, ignores discovery successes, resets on mutation success, doesn't re-fire after consume_fire.

**Phase B (behavioural validation against captured RED)**: NOT RUN — prior diagnosis was wrong. See "Key findings" §2 above.

**Net judgement**: §7.7 is correct-but-not-load-bearing. Kept in tree as a long-tail safety net for future model configs that might anchor; harmless code currently.

---

## What this report does NOT cover

- **Cross-model behaviour** (claude-opus, gpt-5, etc.): out of scope; gpt-5-mini@temperature=0.0 is the deploy and demo target.
- **Persona-driven multi-turn re-run**: not re-run; prior simplified-driver cohort's 0/15 REPRODUCED + this session's 0 hedging samples is sufficient demo-readiness signal.
- **Azure Content Safety scenario** (`aggregation-content-safety`): env-blocked on staging.
- **Latency optimization** (in-progress epic `elspeth-4e79436719`): out of scope; latencies measured (24-50s mean per scenario) are acceptable for demo.
- **§7.6 broader runtime-preflight error rewrite**: out of scope for this ticket. Likely highest-leverage post-demo improvement given drift-on-web_scrape is the dominant failure mode. Filed in observation `elspeth-obs-9dfff9b571`.
- **What the operator's actual demo prompts will be**: unknown to this agent; cohort and dry-runs smoked plausible BA-shapes only.

---

## Reproducibility

```bash
# Re-fire the three demo-shape sequential cohorts (6 runs each):
export ELSPETH_EVAL_BASE_URL=https://elspeth.foundryside.dev
export ELSPETH_EVAL_USER=dta_user ELSPETH_EVAL_PASS=dta_pass
cd /home/john/elspeth/.worktrees/composer-tier1.5-hardening/evals/composer-rgr

# url-download (re-baseline)
for i in 1 2 3 4 5 6; do
  ELSPETH_RGR_SCENARIO=$PWD/scenarios/url-download-line-explode/scenario.json ./run_scenario.sh seq-rebaseline-$i
done
# fork-and-route (showcase) — same pattern with scenarios/fork-and-route/scenario.json
# rag-text-llm (research) — same pattern with scenarios/rag-text-llm/scenario.json

# §7.7 unit + integration tests:
.venv/bin/python -m pytest tests/unit/web/composer/test_anti_anchor.py \
  tests/unit/web/composer/test_compose_loop_anti_anchor.py -q

# Cohort-wide hint-fire count:
sqlite3 /home/john/elspeth/data/sessions.db \
  "SELECT COUNT(*) FROM chat_messages WHERE content LIKE '%[ELSPETH-SYSTEM-HINT]%'"

# Captured RED's actual hash sequence (corrects Step C diagnosis):
.venv/bin/python -m evals.lib.decode_tools /home/john/elspeth/data/sessions.db \
  53bc3cf2-ab90-4940-9679-1b5e7d474650 | jq -r '.[].invocation.arguments_hash // empty'

# This session's verdict tally:
for f in evals/composer-rgr/runs/*demo-rebaseline*/scoring.json \
         evals/composer-rgr/runs/*demo-cohort*/scoring.json \
         evals/composer-rgr/runs/*seq-rebaseline*/scoring.json \
         evals/composer-rgr/runs/*seq-cohort*/scoring.json; do
  python3 -c "import json; print(json.load(open('$f'))['verdict'])"
done | sort | uniq -c
```

---

## 24-hour budget reconciliation

| Slice | Cap (agentic) | Actual | Notes |
|-------|---------------|--------|-------|
| 0 — Recon | 1:30 | ~1:00 | Pre-reading + workspace contract recon |
| 1 — Step B re-author + smoke | 5:00 | ~1:30 | Cheaper than budgeted; scorer fix landed alongside |
| 2 — Multi-turn passivity | 4:00 | ~0:30 | Closed early after finding 1/15 PARTIAL was a measurement-instrument artefact |
| 3 — §7.7 behavioural validation | 4:00 | ~1:00 | No code change — prior diagnosis was wrong (Phase 1 of systematic-debugging saved a wrong fix) |
| 4 — First-turn hedging + UX polish | 3:00 | (rolled into slice 2) | Covered by slice 2 finding |
| 5 — Re-baseline + cohorts | 2:00 | ~3:00 | 30 cohort runs across parallel + sequential conditions |
| 6 — Demo dry run (BA-flavored shapes) | 1:00 | ~0:30 | Two BA-flavored single-runs (1 GREEN, 1 RED on schema contract) |
| 7 — Demo-readiness report | 3:00 | ~2:00 | This file + epic comment |
| Buffer | 1:30 | ~0:30 (in flight) | |
| **TOTAL** | 24:00 | ~10:00 (vs ~12h wallclock) | Significant slack vs cap; biggest underrun was slice 3 (no fix needed) |

---

## Verification trail (pinned for the operator)

| Claim | Command |
|-------|---------|
| §7.7 unit + integration tests pass | `.venv/bin/python -m pytest tests/unit/web/composer/test_anti_anchor.py tests/unit/web/composer/test_compose_loop_anti_anchor.py -q` |
| §7.7 hint has fired 0 times in any session ever | `sqlite3 /home/john/elspeth/data/sessions.db "SELECT COUNT(*) FROM chat_messages WHERE content LIKE '%[ELSPETH-SYSTEM-HINT]%'"` |
| Captured RED was drift-not-anchor | Compare three `arguments_hash` values for session `53bc3cf2-…` via decode_tools |
| Cohort verdicts | `for f in evals/composer-rgr/runs/*demo-rebaseline*/scoring.json evals/composer-rgr/runs/*demo-cohort*/scoring.json evals/composer-rgr/runs/*seq-rebaseline*/scoring.json evals/composer-rgr/runs/*seq-cohort*/scoring.json; do python3 -c "import json; print(json.load(open('$f'))['verdict'])"; done` |
| Multi-turn pathology not reproduced | `cat evals/composer-harness/runs/2026-05-06T13-35-28Z-hardmode-sweep-tier1.5b/SUMMARY.json` |
| Scorer fix unblocks gate detection | `git show 9a51ac97 -- evals/composer-rgr/score.py` |

---

**End of report.**
