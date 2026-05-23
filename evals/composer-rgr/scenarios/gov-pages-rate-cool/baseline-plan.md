# gov-pages-rate-cool — baseline run plan

This scenario captures the failure pattern observed in staging session
`47cfbb5e-f269-47c2-9c99-134f494aeba7`. The hardening program has three
co-ordinated parts (skill text, failure schema, telemetry); the four
runs described here attribute the contribution of each part by
comparing scoring deltas across known-state checkpoints.

## Pre-conditions

1. The Part 1 (skill amendment), Part 2 (failure schema), and Part 3
   (telemetry) changes are committed on the branch the staging service
   reads from (likely `main` — confirm with the operator before
   restarting). The composer skill is loaded once at process start
   (`prompts.py:23` — `load_skill("pipeline_composer")` runs at module
   import) and `build_system_prompt` is `@lru_cache`'d, so skill edits
   do NOT take effect until the service restarts.
2. `elspeth-web.service` has been restarted after the merge:
   ```bash
   sudo systemctl restart elspeth-web.service
   ```
3. `evals/composer-rgr/.env` is present (and `chmod 600`) with
   `ELSPETH_EVAL_BASE_URL`, `ELSPETH_EVAL_USER`, `ELSPETH_EVAL_PASS`
   pointing at staging.

## Run plan (four checkpoints)

The harness invocation is the same each time; the differentiator is
the deployed state of the staging service when each run executes.
Each run lands in
`evals/composer-rgr/runs/<utc-ts>-gov-pages-rate-cool-<label>/`.

```bash
cd /home/john/elspeth/evals/composer-rgr
set -a; . .env; set +a
export ELSPETH_RGR_SCENARIO=$PWD/scenarios/gov-pages-rate-cool/scenario.json
```

### 1. RED baseline — pre-merge

Tag: `gov-pages-rate-cool-baseline`

Run before any of Parts 1/2/3 are merged. Establishes the failure
shape as scored by the new rules: expected verdict RED, with all of
`set_pipeline_rejection_without_success`, `max_persisted_tool_calls`,
and `must_be_valid` contributing red reasons.

```bash
./run_scenario.sh gov-pages-rate-cool-baseline
```

### 2. After Part 1 (skill amendment)

Tag: `gov-pages-rate-cool-part1-skill`

Run after the skill text changes that sharpen discover-first guidance
and tighten the "review the rejection envelope before retrying"
language. Expected delta vs. baseline: `must_discover_schema_before_first_mutation`
moves AMBER -> GREEN, `set_pipeline_rejection_without_success` may
move RED -> GREEN, total tool calls likely drops from 13 toward 8.

```bash
sudo systemctl restart elspeth-web.service
./run_scenario.sh gov-pages-rate-cool-part1-skill
```

### 3. After Part 2 (failure schema)

Tag: `gov-pages-rate-cool-part2-failure-schema`

Run after the backend rejection-envelope changes. Expected delta:
when the model retries after a rejection, it gets a more structured
diagnostic and converges faster. `max_persisted_tool_calls` should
move further down; the rejection-without-success rule should be solid
GREEN.

```bash
sudo systemctl restart elspeth-web.service
./run_scenario.sh gov-pages-rate-cool-part2-failure-schema
```

### 4. After Part 3 (telemetry)

Tag: `gov-pages-rate-cool-part3-telemetry`

Run after the pre-commit tool-budget signal is added. Expected delta:
tool-call count stable or down further; no regression of the earlier
gains; all five red criteria GREEN and all four green criteria GREEN.

```bash
sudo systemctl restart elspeth-web.service
./run_scenario.sh gov-pages-rate-cool-part3-telemetry
```

## Attribution analysis

After all four runs land, compare the four `scoring.json` files:

```bash
cd /home/john/elspeth/evals/composer-rgr/runs
for d in $(ls -d *gov-pages-rate-cool-*); do
    echo "=== $d ==="
    cat "$d/scoring.json"
done
```

The `stats.persisted_tool_call_count` field on each result (new in
the 2026-05-23 scoring extensions) is the primary trend signal.
`red_reasons` / `amber_reasons` arrays diff the categorical state
changes across the four runs.

## Notes

- The harness fetches `messages.json` with `?include_tool_rows=true`
  (as of the 2026-05-23 `run_scenario.sh` change). Earlier runs in
  `runs/` were captured WITHOUT tool rows; comparing this scenario's
  scoring against pre-2026-05-23 runs of OTHER scenarios is not
  meaningful for the new tool-sequence stats. The new red rules
  silently no-op when no tool rows are present, so the scoring shape
  of older scenarios is unchanged.
- The "interpretation review for the subjective term 'cool'" check
  is documented in `scenario.json` under
  `_subjective_term_interpretation_review` as a manual-review item.
  Promoting it to a first-class programmatic check is a Part-2 follow-up
  (the structured rejection envelope will surface the relevant
  tool-call dispatch in a form the scorer can key on without a brittle
  arguments-JSON inspection).
- LLM nondeterminism is large at this prompt size. The operator's
  RGR convention (per the harness README) is 3 runs per checkpoint
  when the headline number is consequential; for an initial baseline
  one run per checkpoint is acceptable to surface trend direction,
  with a follow-up 3-run sweep on the final state once Parts 1-3
  have all landed.
