# Panel-evals handover — 2026-05-09

Self-contained briefing for a Claude Code session picking up the
panel-of-examples evaluation harness work for ELSPETH. Read this
end-to-end before touching anything; do not skim.

## Mission (one sentence)

Build out and run a panel of evaluation scenarios that drive the
composer through every non-Azure non-Docker example in `examples/`
× multiple personas at distinct competence levels, with hard
guarantees that amateur personas stay amateur under composer
pressure.

## Status

| Phase | Status | Commit |
|---|---|---|
| A — persona discipline (competence_ceiling + 2-channel drift detector + anti-helpfulness clause) | **DONE** | `ee46ea59` |
| B — scenario factory + cohort.yaml + RGR scoring step + harness env override | **DONE** | `c40637ab` |
| C — smoke cohort (12 cells) | **BLOCKED on `elspeth-861b0c58f5`** (composer bug, see below) |
| C+ — broad cohort (74 cells) | **NOT STARTED** |

## The bug currently blocking C

**`elspeth-861b0c58f5`** — Composer set_pipeline loops byte-identically
on csv source schema-validation failure.

A separate agent is working this. Discovered on the very first panel-cohort
test cell (`boolean_routing__p3_marketingops`) on 2026-05-08. The composer
fails to recover from a `schema: Field required` failure on csv sources;
the same `[ELSPETH-SYSTEM]` reply byte-repeats across consecutive turns
without state mutation. Affects most simple-shape pipelines that use
`csv` source plugins.

**Implication for the smoke cohort:** of the 12 smoke cells, roughly
8-10 will fail to converge until that bug is fixed (every cell whose
example uses a csv source without explicit schema hints in the example
yaml). Smoke cohort CAN still be run as data — RED verdicts on
csv-source cells will cluster around the same bug, but cells using
text/web_scrape/json sources may converge.

**Wait for the bug fix before declaring smoke results signal vs noise.**

## What you must read before acting

In this order:

1. **`/home/john/CLAUDE.md`** — global project instructions (high-auditability standard, no-defensive-programming, tier model, etc.)
2. **`/home/john/elspeth/CLAUDE.md`** — repo-specific (data manifesto, plugin ownership, layer rules)
3. **`/home/john/.claude/projects/-home-john-elspeth/memory/MEMORY.md`** — persistent feedback that survived prior sessions
4. **The panel-evals epic itself: `mcp__filigree__get_issue elspeth-d573f8e8ab`** — has two long status comments laying out everything that's been done
5. **`evals/lib/dispatch-protocol.md`** — load-bearing for any persona-subagent dispatch you do; the **anti-helpfulness clause** is the difference between probative and performative transcripts
6. **The 4 persona specs at `evals/composer-harness/personas/p{1,2,3,4}_*.md`** — internalize the `competence_ceiling`, `incomprehension_moves`, `concession_rule` sections; these are the load-bearing additions from Phase A

## Key files (cohort artefacts)

```
evals/composer-harness/
├── personas/
│   ├── p1_compliance.md            (Linda — amateur)
│   ├── p2_researcher.md            (Sarah — journeyman_academic)
│   ├── p3_marketingops.md          (Marcus — amateur_overconfident)
│   └── p4_adversarial_engineer.md  (Dev — competence_ceiling: none)
├── cohorts/
│   └── panel-broad-2026-05-09.yaml (74-cell matrix; smoke_subset of 12)
├── scenarios/panel/panel-broad-2026-05-09/
│   └── *.json                      (12 generated smoke scenarios, committed)
├── hardmode/
│   ├── harness.sh                  (reads ELSPETH_EVAL_SCENARIO_ROOT env)
│   ├── post_message.sh             (per-turn driver)
│   ├── finalize_scenario.sh        (auto-fires score_scenario.sh when scenario has red+green criteria)
│   ├── score_scenario.sh           (NEW — RGR verdict from green_criteria)
│   ├── validate_drift.sh           (NEW — Channel 1 structural drift)
│   ├── judge_persona.sh            (NEW — Channel 2 LLM judge via Haiku)
│   ├── generate_cohort.sh          (NEW — walks cohort.yaml × extractor × drafter)
│   ├── aggregate.sh                (NEW: rolls in RGR + drift verdicts)
│   └── sweep_simplified.sh         (reads ELSPETH_EVAL_SCENARIO_ROOT)
├── runs/                           (gitignored)
│   └── panel-smoke-test-2026-05-09/boolean_routing__p3_marketingops/
│                                   (the bug-discovery test run; keep for diff)
└── lib/  (in evals/lib/)
    ├── scenario_from_example.py    (NEW — settings.yaml → green_criteria)
    ├── prompt_drafter.py           (NEW — Haiku persona-flavoured opening prompt)
    └── composer_rgr_score.py       (PRE-EXISTING from convergence-suite)
```

## Resume commands (after the bug is fixed)

The harness expects two env vars set BEFORE you call any of the
hardmode scripts:

```bash
set -a
source /home/john/elspeth/evals/composer-harness/.env  # has ELSPETH_EVAL_BASE_URL/USER/PASS
source /home/john/elspeth/.env                          # has OPENROUTER_API_KEY
set +a
export ELSPETH_EVAL_SCENARIO_ROOT=/home/john/elspeth/evals/composer-harness/scenarios/panel/panel-broad-2026-05-09
export ELSPETH_EVAL_RUNS_DIR=/home/john/elspeth/evals/composer-harness/runs/panel-smoke-2026-05-N    # pick a fresh date
export ELSPETH_EVAL_RUN_TIMEOUT_SEC=180
export ELSPETH_EVAL_RUN_POLL_INTERVAL=3
cd /home/john/elspeth
```

Then for each smoke cell (12 total):

```bash
SCEN=boolean_routing__p3_marketingops   # or any other smoke cell
RUN_DIR="$ELSPETH_EVAL_RUNS_DIR/$SCEN"

# 1. Bootstrap
evals/composer-harness/hardmode/harness.sh "$SCEN"
jq -r '.opening_prompt' "$RUN_DIR/scenario.json" > "$RUN_DIR/turn1.user.txt"

# 2. POST turn 1
evals/composer-harness/hardmode/post_message.sh "$SCEN" 1 "$RUN_DIR/turn1.user.txt"

# 3. For turns 2..5, dispatch a persona subagent via the Agent tool.
#    Subagent prompt template lives in evals/lib/dispatch-protocol.md
#    (read the "Anti-helpfulness clause" section verbatim; it is mandatory).
#    The subagent reads:
#      - personas/<persona_id>.md
#      - dispatch-protocol.md
#      - scenario.json (in the run dir)
#      - turn1.user.txt … turn<N>.user.txt
#      - msg.t1.resp.json … msg.t<N>.resp.json (extract via jq -r '.message.content')
#    It returns either the next user message OR "DONE: <reason>".
#    Save the message to turn<N+1>.user.txt and POST it.
#    Stop when persona says DONE OR the composer signals done OR 5 turns reached.

# 4. Finalize (auto-runs score_scenario.sh because scenario has red/green criteria)
evals/composer-harness/hardmode/finalize_scenario.sh "$SCEN"

# 5. Run both fidelity channels
evals/composer-harness/hardmode/validate_drift.sh "$RUN_DIR"
evals/composer-harness/hardmode/judge_persona.sh "$RUN_DIR"

# After all 12 cells:
evals/composer-harness/hardmode/aggregate.sh "$ELSPETH_EVAL_RUNS_DIR"
# Read SCORECARD.md
```

## Smoke-cohort cells (12)

From `cohorts/panel-broad-2026-05-09.yaml` `smoke_subset`:

```
boolean_routing                    × p3_marketingops, p1_compliance
chroma_rag                         × p2_researcher, p1_compliance
batch_aggregation                  × p4_adversarial_engineer, p1_compliance
transform_pipeline                 × p3_marketingops, p1_compliance
fork_coalesce                      × p4_adversarial_engineer, p1_compliance
schema_contracts_llm_assessment    × p2_researcher, p1_compliance
```

## Decisions already made (don't relitigate)

- **Personas**: keep the 4 we have, add `competence_ceiling` per persona. (Phase A choice, confirmed by operator.)
- **Matrix sparsity**: sparse 2/example + 6 diagnostic full-matrix. 74 cells total.
- **Phase A blocks Phase B/C**: don't run scenarios at scale until persona discipline lands. Discipline landed in `ee46ea59`.
- **Opening-prompt drafting**: LLM-drafted via Haiku, operator spot-checks smoke. No hand-authoring.
- **Generated scenarios committed**: pinned against drafter temperature variation. Re-run with `--force` only on intentional regeneration.
- **Joining rule for fidelity channels**: conservative (either fires = drift) until smoke evidence calibrates it.

## Decisions deferred (handle when relevant)

- **Channel-1↔Channel-2 joining rule calibration**: the contract test on `boolean_routing__p3_marketingops` showed Channel 1 fired on `REAL_LEAD` (a classification literal) while Channel 2 said in-character — needs calibration data from full smoke. Don't change the rule until you have ≥10 cohort cells worth of joined verdicts.
- **Inline drafter/judge dispatch (smoke-cohort prototype)**: documented in `dispatch-protocol.md` but not implemented. Worth prototyping IF the post-hoc Channel 2 catches drift after the fact rather than at composition time.
- **Statistical_batch_plugins**: only `threshold_summary` variant in the cohort right now; the other 9 variants are deferred. Operator can extend cohort.yaml `sparse:` list.

## Things that will surprise / trip you

1. **`runs/` is gitignored.** Don't try to commit run dirs. Commit only the cohort YAML, scenario JSONs (under `scenarios/panel/`), and code.

2. **Pre-commit hooks auto-fix JSON whitespace** then reject the commit. CLAUDE.md no-amend policy: re-add the fixed files and create a NEW commit. Do not `git commit --amend`.

3. **`ELSPETH_EVAL_RUNS_DIR`, NOT `runs_root` in cohort.yaml**, controls where runs land. The cohort.yaml `runs_root` field is documentation only. `generate_cohort.sh` prints copy-paste exports at the end of its run — use those.

4. **The persona-subagent dispatch is half shell, half Agent tool.** Shell scripts handle deterministic plumbing; the Agent tool handles turn 2+ user message generation. You CAN'T fully automate this from a single bash subprocess — each turn requires one Agent dispatch from the parent Claude session.

5. **`scenario.json` field naming is hardmode-aligned.** The opening message lives at `.opening_prompt`, not `.message`. The summary lives at `.task_summary`, not `.summary` (this was an advisor catch during Phase B).

6. **Channel 1 is exempt for P4 (Dev).** `competence_ceiling: **none**` is a literal value, validated as required. Don't add `getattr` defensive guards trying to "handle" missing competence_ceiling — `validate_drift.sh` exits 70 if absent, by design.

7. **Marcus's persona is structurally different from the others.** His pseudo-technical vocab ("schema", "trigger", "webhook", "field mapping") is in-character ONLY with HIS amateur meanings. If the composer corrects him and Marcus then uses these words with the composer's correct meanings, that's **semantic drift** — Channel 1 cannot see it (no new tokens); only Channel 2 catches it. Bear this in mind when reviewing P3 cells.

8. **Composer model on staging is gpt-5-mini.** Don't expect Claude-level convergence. The panel cohort is partly designed to surface its weaknesses.

9. **The composer bug `elspeth-861b0c58f5` was discovered on cell #1 of the smoke test.** Cost: $0.0975. The bug's evidence dir is `runs/panel-smoke-test-2026-05-09/boolean_routing__p3_marketingops/`. Don't delete it — it's the canonical reproduction artefact.

## Operator-facing protocols

- Operator is **John Morrissey**. Email: qacona@gmail.com.
- The operator is **neurodiverse**. If they don't answer a question, do NOT assume tacit consent — relitigate the question, briefly explain why the answer matters.
- Operator strongly dislikes: defensive programming, unnecessary `slog` recommendations, calendar-based shipping commitments, scope dumping, git stash. See `MEMORY.md` for the full list.
- Operator authorizes spending OpenRouter budget on staging-deploy testing within reason ($1-12 ranges OK without asking; ask for >$20).

## What to do FIRST in your new session

1. Read this file fully. Then read the items in "What you must read before acting" (above).
2. Check the bug status: `mcp__filigree__get_issue elspeth-861b0c58f5`. If it's still open, the smoke cohort is still blocked — you can prepare other things but don't burn budget on csv-source cells.
3. Check the panel-evals epic: `mcp__filigree__get_issue elspeth-d573f8e8ab`. The two long status comments are the canonical history.
4. Run `git log --oneline ee46ea59~1..HEAD -- evals/composer-harness/ evals/lib/scenario_from_example.py evals/lib/prompt_drafter.py` to see exactly what's landed.
5. Ask the operator before kicking the smoke cohort. The advisor's pattern: "On tasks longer than a few steps, call advisor at least once before committing to an approach". Use that.

## What NOT to do

- Don't try to "fix" the composer bug yourself. A separate agent is on it.
- Don't relitigate the persona spec format. The operator confirmed the 4-persona-with-competence-ceiling design.
- Don't expand the cohort matrix beyond 74 cells before smoke evidence justifies it.
- Don't run more than 1-2 cells live without operator authorization. Smoke = $12 estimated; that's worth a check-in before kicking.
- Don't change the Channel-1↔Channel-2 joining rule until you have smoke-cohort calibration data.
- Don't commit `runs/` directories (gitignored).

## End

When you finish a substantive change, before declaring done:
1. Make the deliverable durable (write the file, save the result, commit the change).
2. Call advisor() to sanity-check.
3. Update the panel-evals epic (`elspeth-d573f8e8ab`) with a status comment.
4. Mark TaskCreate items completed.

Good luck.
