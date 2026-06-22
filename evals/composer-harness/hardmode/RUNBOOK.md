# Hard-Mode Eval Runbook

Step-by-step for running one or all hard-mode scenarios end-to-end.

## Prerequisites

- `jq`, `python3`, `curl` on PATH.
- Network access to `$ELSPETH_EVAL_BASE_URL`.
- Valid credentials for the eval user.
- An LLM provider key configured server-side (the composer reads OpenRouter
  / Azure OpenAI keys from the deploy's secrets store; the harness does not
  pass these).
- A parent agent (Claude in your terminal session) that can dispatch
  `general-purpose` subagents — needed for turn-2+ persona dialogue.

## One-time setup

```bash
cd evals/composer-harness
cp .env.example .env
# Edit .env and set ELSPETH_EVAL_BASE_URL / ELSPETH_EVAL_USER / ELSPETH_EVAL_PASS

hardmode/harness.sh --doctor
# Expected: 'all preflight checks passed'
# If it fails: read the ERROR line and the README.md "Required env" section.
```

## Running a single scenario

The example below uses `p1_t1_happy`. Substitute any scenario_id that has
a fixture under `scenarios/hardmode/`.

### Step 1: Bootstrap

```bash
SCENARIO=p1_t1_happy
export ELSPETH_EVAL_RUNS_DIR="${ELSPETH_EVAL_RUNS_DIR:-runs/$(date -u +%Y-%m-%d)-hardmode}"
RUN_DIR="$ELSPETH_EVAL_RUNS_DIR/$SCENARIO"
hardmode/harness.sh "$SCENARIO"
```

This logs in, creates a composer session, uploads the CSV (if the fixture
has one), and writes scaffolding into:

```
runs/<UTC-date>-hardmode/p1_t1_happy/
  scenario.json  run_manifest.json  session.json  sid.txt  jwt.txt
  blob.req.json  blob.json     harness.log
```

The run root also gets `suite_manifest.json`, which records the hardmode
dispatch contract and message budget for the whole suite.

The script prints the next-step commands at the end. Note the run dir.

### Step 2: Send turn 1 (opening prompt — no subagent)

The opening prompt is verbatim from the fixture. Don't paraphrase it; that's
the deterministic entry condition.

```bash
jq -r '.opening_prompt' "$RUN_DIR/scenario.json" > "$RUN_DIR/turn1.user.txt"
hardmode/post_message.sh "$SCENARIO" 1 "$RUN_DIR/turn1.user.txt"
```

Read the composer's reply:

```bash
jq -r '.message.content' "$RUN_DIR/msg.t1.resp.json"
```

Each posted turn writes `turn<N>.manifest.json` in addition to
`metrics.t<N>.json`. If `/messages` returns non-2xx, the response body is still
preserved and the turn manifest/metrics record `turn_status.status =
"transport_error"`.

Each turn also writes `msg.t<N>.raw.json`, capturing the model's pre-synthesis
prose (the `raw_content` field exposed by `?include_raw_content=true`). When the
empty-state synthesizer at `service.py:_finalize_no_tool_response` augments or
replaces visible content, `msg.t<N>.resp.json.content` will not match the
model's actual final text. `msg.t<N>.raw.json.raw_content` preserves what the
model really said.

Three synthesizer flags are written into `metrics.t<N>.json`:

- `synthesizer_replaced` — the cohort-scoring signal. `true` when visible
  content does not start with `raw_content`, i.e. the synthesizer hid the
  model's prose behind a synthetic `[ELSPETH-SYSTEM]` blocker. Required for
  distinguishing "model converged on useful output that the synthesizer hid"
  from "model genuinely failed to converge" (cf. elspeth-861b0c58f5).
- `synthesizer_augmented` — `true` when visible content starts with
  `raw_content` and is strictly longer, i.e. the synthesizer appended an
  operator-facing suffix to the model's prose. The model's prose is preserved
  in the visible content; the suffix is operator-facing only. Useful for
  distinguishing "model produced honest empty-state diagnostic + suffix" from
  the replacement shape.
- `synthesizer_intercepted` — coarse union (`replaced` OR `augmented`).
  `true` whenever any synthesis activity occurred. Retained for backward
  compatibility with diagnostic dumps; prefer the finer-grained flags above
  for cohort scoring.

### Step 3: Decide whether to continue

Two early-exit conditions:

- The composer's reply is a *clear DONE* (e.g. "the workflow is ready, you
  can run it") AND the persona's stop conditions are met. Skip to step 5.
- 5 turns reached. Force-stop, mark as "convergence-budget exhausted".

Otherwise, continue.

### Step 4: Spawn persona-subagent for turn N+1

This is the parent-agent step. The contract is in
`evals/lib/dispatch-protocol.md`. In Claude Code, dispatch a `general-purpose`
subagent with this prompt template:

```
You are <persona_id> — read the spec at:
  evals/composer-harness/personas/<persona_id>.md

Conversation context (read in order):
  evals/composer-harness/scenarios/hardmode/<scenario_id>_*.json   (your task and the rules)
  evals/composer-harness/runs/<date>-hardmode/<scenario_id>/turn1.user.txt
  evals/composer-harness/runs/<date>-hardmode/<scenario_id>/msg.t1.resp.json
    (assistant's reply — extract with: jq -r '.message.content' msg.t1.resp.json)
  ... (turn2.user.txt, msg.t2.resp.json, etc. for all prior turns)

Reply with EXACTLY ONE of:
  (a) The next user message in character. Plain text. No JSON wrapping, no
      preamble, no quotes around it. This will be saved into turn<N+1>.user.txt
      and POSTed verbatim.
  (b) The literal token "DONE: <one-line reason>" if your stop conditions
      have been met OR you've exhausted your message budget.

Do not narrate. Do not describe what you're going to say. Just say it.
```

Save the subagent's reply into `$RUN_DIR/turn<N+1>.user.txt`.

If it starts with `DONE:`, write the reason to `$RUN_DIR/done_reason.txt`
and skip to step 5. Otherwise:

```bash
hardmode/post_message.sh <scenario_id> <N+1> "$RUN_DIR/turn<N+1>.user.txt"
jq -r '.message.content' "$RUN_DIR/msg.t<N+1>.resp.json"
```

Loop back to step 3.

### Step 5: Finalize

```bash
hardmode/finalize_scenario.sh "$SCENARIO"
```

This runs `/validate`. If valid, runs `/execute` and polls `/api/runs/<rid>`
until terminal status (within `ELSPETH_EVAL_RUN_TIMEOUT_SEC`, default 300s).

Outputs:

```
runs/<date>-hardmode/p1_t1_happy/
  validate.json        # the 9-check matrix
  execute.{json,code}  # the run trigger
  run.json             # final run row, or {} if the engine did not run
  diagnostics.json     # per-token engine diagnostics, or {} if unavailable
  diagnostics.json.code
  final_yaml.json      # the YAML the composer rendered, or {} if unavailable
  final_yaml.json.code
  messages.json        # conversation plus safe LLM-call audit sidecars, or [] if unavailable
  messages.json.code
  artifact_collection_errors.json
  ledger.json          # consolidated summary, including provider usage/cost metadata
```

`/validate` is required infrastructure: if it returns non-2xx, finalization
exits 74 and no ledger is written. Post-validate artifacts are best-effort
evidence collection. If diagnostics, final YAML export, or message export fails,
the finalizer writes a typed fallback, records the HTTP code in
`artifact_collection_errors.json`, and still emits `ledger.json`.
The message export uses `include_llm_audit=true`; those sidecars expose model,
token, and provider-cost metadata, but not raw prompts, tool arguments, or tool
results.

### Step 6 (optional): persona-character check

```bash
hardmode/validate_persona.sh "$RUN_DIR"
cat "$RUN_DIR/persona_check.json"
```

Flags any MUST-AVOID phrase that leaked into the persona's turns (subagent
broke character).

## Running all scenarios

Sequentially is fine — most scenarios converge in 60-180s of LLM time.

```bash
export ELSPETH_EVAL_RUNS_DIR="${ELSPETH_EVAL_RUNS_DIR:-runs/$(date -u +%Y-%m-%d)-hardmode}"
for sid in p1_t1_happy p1_t2_edge p1_t3_limit p1_t4_stress \
           p2_t1_happy p2_t2_edge p2_t3_limit p2_t4_stress \
           p3_t1_happy p3_t2_edge p3_t3_limit p3_t4_stress \
           p4_t1_happy p4_t2_edge p4_t3_limit; do
    echo "=== $sid ==="
    hardmode/harness.sh "$sid"
    # Then: parent agent runs the persona loop manually for each scenario,
    # then finalizes:
    hardmode/finalize_scenario.sh "$sid"
done

hardmode/aggregate.sh "$ELSPETH_EVAL_RUNS_DIR"
cat "$ELSPETH_EVAL_RUNS_DIR/SCORECARD.md"
```

`aggregate.sh` writes four run-root artifacts:

- `aggregate.json` — one record per scenario, including artifact errors,
  provider usage metadata, and cost metadata.
- `aggregate_errors.json` — malformed ledger and aggregation evidence errors.
- `aggregate_summary.json` — suite-level wall time, artifact error totals,
  aggregate error totals, provider token totals, token-usage coverage,
  provider-reported cost, and cost coverage.
- `SCORECARD.md` — human-readable table plus persona/class matrix.

The persona loop in the middle is *not* automatable from bash alone — it
requires the parent agent to dispatch subagents per turn. Plan for ~10
minutes of supervised dispatch per scenario; a full 15-scenario run is a
2-3 hour activity for the parent agent. Use provider billing for actual spend;
the 2026-05-07 full run cost about $3.

## Replaying engine-only

When you've made an engine fix and want to verify it against a captured
scenario without redoing the LLM dialogue:

```bash
hardmode/replay.sh runs/<date>-hardmode/p1_t1_happy
```

Writes a new run into `runs/<date>-hardmode/p1_t1_happy/replays/<utc-ts>/`
with a `replay_summary.json` showing original-vs-replay status delta.

## Recovery scenarios

| Symptom | Cause | Fix |
|---|---|---|
| `harness.sh` says output dir already exists | re-running same scenario | `--fresh` to reset, or `--reuse-sid` to continue |
| `post_message.sh` fails with 401 | JWT expired mid-scenario | The harness auto-refreshes; if it doesn't, re-run `hardmode/harness.sh <sid> --reuse-sid` to force a new login |
| `finalize_scenario.sh` exits 75 (poll timeout) | Engine still running at 300s | Raise `ELSPETH_EVAL_RUN_TIMEOUT_SEC` or read partial run.json |
| `artifact_collection_errors.json` lists `final_yaml`, `messages`, or `diagnostics` | Optional artifact endpoint failed after validate | Keep the ledger; classify the scenario from `validate.json`, `run.json`, and the recorded artifact error |
| `replay.sh` says YAML import failed | The session YAML import route rejected the captured YAML | Check `replays/<utc-ts>/import.json`; replay imports through `POST /api/sessions/{sid}/state/yaml` |
| Persona-subagent breaks character | Subagent saw too much context | Spawn a *fresh* general-purpose subagent each turn — never re-use one across turns |

## Cost accounting

The harness does not estimate dollars from wall time. That heuristic overstated
the 2026-05-07 run: the actual operator-observed provider spend was about $3.

Use OpenRouter/provider billing as the source of truth for USD. The harness
records best-effort provider token metadata in each `ledger.json` and totals it
in `aggregate_summary.json` when diagnostics expose usage. Check
`provider_usage.token_usage_available`; unavailable usage is rendered as `—` in
the scorecard, not as zero.

When the server exposes LLM-call audit sidecars, `ledger.json` also sums
provider-reported `response.usage.cost` into `cost.actual_usd`. Missing cost is
rendered as `—` and counted in `cost_unavailable_scenarios`; it is never
estimated from wall time.
