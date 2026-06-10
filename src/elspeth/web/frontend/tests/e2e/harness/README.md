# Tutorial reliability harness

A repeatable, version-tagged battery that drives the first-run composer tutorial
end-to-end against **staging** (`elspeth.foundryside.dev`), resets between each
run, grades every run on four dimensions, and reports how many worked plus a
classified failure table.

Design: `docs/superpowers/specs/2026-06-06-tutorial-reliability-harness-design.md`
Plan: `docs/superpowers/plans/2026-06-06-tutorial-reliability-harness.md`

## What's in this directory

| File | Responsibility |
|------|----------------|
| `prompt-and-rubric.ts` | The **coupled triple**: the fixed (non-canonical) prompt, the dimension-(c) assumption rubric, the dimension-(d) judge rubric/thresholds, and `HARNESS_VERSION`. Pure data. If the prompt changes, **both** rubrics change in lockstep (spec §11). |
| `types.ts` | `RunRecord` / `FaultSubclass` / `Outcome` — the per-run record schema (spec §8). |
| `aggregate.mjs` | Node script: read per-run records → headline rates + failure table → append a version-stamped trend row → write a markdown report. |

The battery spec itself lives one level up:
`../tutorial-reliability.staging.spec.ts`. Live-API helpers (reset, snapshot,
re-run, diagnostics) live in `../helpers/tutorial-harness.ts`.

## Running the battery

The staging config (`playwright.staging.config.ts`) is single-worker,
`fullyParallel:false`, `retries:0`, so runs are **sequential** (one shared
`dta_user` account) and **all** runs execute even if some fail. Invoke from the
frontend dir:

```bash
cd src/elspeth/web/frontend
HARNESS_BATCH_ID=batch-2026-06-06 HARNESS_BATCH_SIZE=10 \
STAGING_BASE_URL=https://elspeth.foundryside.dev \
STAGING_USERNAME=dta_user STAGING_PASSWORD=dta_pass \
PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
npx playwright test --config=playwright.staging.config.ts tutorial-reliability.staging.spec.ts
```

| Env var | Default | Meaning |
|---------|---------|---------|
| `HARNESS_BATCH_ID` | `skeleton` | Names the batch; results land under `tests/e2e/.harness-results/<batch_id>/` (gitignored) and the report lands at `notes/tutorial-reliability/<batch_id>.md`. |
| `HARNESS_BATCH_SIZE` | `1` | How many independent tutorial runs the battery enumerates. |

The four `STAGING_*` / `PLAYWRIGHT_BACKEND_BASE_URL` vars are required by
`playwright.staging.config.ts` and the global-setup auth step.

Each run writes one record:
`tests/e2e/.harness-results/<batch_id>/run-NN.json` (raw, gitignored).

## Per-run record schema

The authoritative schema is `types.ts` (`RunRecord`). Key fields:

- `outcome` — `pass` | `tutorial_fault` | `infra_fault` (spec §7). The
  `tutorial_fault` rows carry a `fault_subclass` and a `fix_target` — *what we
  fix*; `infra_fault` rows are *noise we ignore* so a prompt change's effect is
  not confounded.
- `dim_a_tutorial_completed` — (a) the tutorial reached graduation with no
  thrown error.
- `dim_b_realsystem_passed` — (b) the **real-system** re-run (normal execute
  path, not the tutorial path) reached terminal `completed`, and normalization
  did not silently repair the pipeline. A run that passes (a) but fails (b) is
  the `normalization-gap` "tutorial lies" class.
- `dim_c_assumptions_ok` — (c) no `under_flagged` and no `over_flagged`
  interpretation kinds vs `ASSUMPTION_RUBRIC` (deterministic, by kind not
  wording).
- `dim_d_solution_quality` — (d) creative-solution quality (see below): carries
  `status` (`pending_judge` | `pass` | `fail`), `judge_score`,
  `source_reachable` (mechanical), `discarded_row_count` (mechanical),
  `substantive_rows` (mechanical).
- `assumptions`, `output_rows`, `landscape`, `stamp`, `timing_s`, `error` —
  raw material + the batch stamp (`composer_skill_hash`, `model_identifier`).

## Mechanical vs judge split (dimension d)

Dimension (d) deliberately splits the **objective** aspects (computed in the
spec, off the judge, to keep them stable) from the **subjective** quality
judgment (spec §6, §11):

- **Mechanical (computed in the spec / `tutorial-harness.ts`, NOT by the
  judge):**
  - `source_reachable` — distinct scrape operations that completed without error
    (`reachableSourceCount` over Landscape diagnostics). `< minReachableSources`
    (5) ⇒ `invented-source-unreachable`.
  - `discarded_row_count` — from the tutorial run output. `> maxDiscardedRows`
    (0) ⇒ `degenerate-output`.
  - `substantive_rows` — rows carrying a meaningful, non-empty colour value not
    matching `/cannot|unknown|n\/a|none/i`. `< minSubstantiveRows` (4) ⇒
    `degenerate-output`.
- **Judge-scored (LLM-judge, irreducibly a judgment):** are the colour values
  real, specific, and plausibly correct — not nulls, not "cannot determine", not
  fabricated when the page shows no clear palette? Scored `0..1` against the
  fixed `JUDGE_RUBRIC.judgePrompt`; pass at `judgePassThreshold` (0.7).

Keeping reachability/discard/substance mechanical removes them as a judge noise
source; the judge model + version belong in the batch stamp so judge changes are
attributable and not confused with composer changes.

## Dimension (d) judging loop

Full LLM-API automation of the judge is **deferred** (spec §9 — first
deliverable is the browser tier on the single fixed prompt). For now the judge
step is a per-batch operator/agent loop:

1. Run the battery (above). Each `run-NN.json` lands with
   `dim_d_solution_quality.status = "pending_judge"` and the mechanical fields
   already populated.
2. For each record, read `output_rows` (the raw rows the run produced) and apply
   `JUDGE_RUBRIC.judgePrompt` from `prompt-and-rubric.ts`. The judge sees the
   **task** and the **output**, not "is this good?" in the abstract (bias
   control, spec §6).
3. Set `dim_d_solution_quality.status` to `pass` / `fail` and record
   `judge_score` (0..1; pass at `JUDGE_RUBRIC.judgePassThreshold`). A mechanical
   fail (`source_reachable < 5/5`, `discarded_row_count > 0`, or
   `substantive_rows < 4/5`) already fails (d) regardless of the judge.
4. Re-run the aggregator to finalize the dim-d column.

## Aggregating a batch

Run from the frontend dir (the relative paths in `aggregate.mjs` resolve against
this cwd → repo-root `notes/`):

```bash
cd src/elspeth/web/frontend
node tests/e2e/harness/aggregate.mjs <batch_id>
```

It prints, and writes, a report to `notes/tutorial-reliability/<batch_id>.md`
and appends a version-stamped line to `notes/tutorial-reliability/trend.jsonl`.
The report carries the **two headline numbers** (spec §7): **tutorial-pass-rate**
(the one driven to 100%) and **infra-noise rate** (so prompt changes are not
confounded), plus per-dimension pass counts and the classified failure table.
Each report is stamped with the git SHA, harness version, model id, and
composer skill hash so cross-batch trend comparison is attributable.
