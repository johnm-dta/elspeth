# Phase 5b interpretation-review regression suite

The Task 12 regression suite for `docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md`. Verifies the composer LLM correctly surfaces subjective user terms via `request_interpretation_review`, stages `{{interpretation:<term>}}` placeholders in LLM-transform `prompt_template` fields, and respects the per-session opt-out flag.

This suite is the gate for Phase 5b's PR. The artifact at `results-<date>.json` (committed alongside the code) must show all four pass thresholds met.

## What this suite tests

Four assertions, each with its own pass threshold (per 18a Task 12):

| # | Assertion | Threshold |
|---|---|---|
| 1 | Hero prompt surfaces `cool` as the interpretation term | ≥8/10 runs |
| 2 | Hero prompt does NOT surface `5` or `1-10` as interpretation terms | ≥9/10 runs |
| 3 | The LLM emits `{{interpretation:cool}}` in the same turn as staging the LLM transform | ≥8/10 runs |
| 4 | With `interpretation_review_disabled=true`, the LLM skips `request_interpretation_review` and writes the auto-interpreted value with `interpretation_source='auto_interpreted_opt_out'` | ≥9/10 runs |

## Scenarios

Nine scenarios. The hero (`hero-cool`) and opt-out (`hero-opt-out`) scenarios are the direct gate inputs — replicate each 10 times. The remaining seven are spot-check breadth: they confirm the surfacing heuristic generalises (other subjective terms) and resists false positives (numeric quantifiers, temporal anchors, fully-concrete prompts).

| Scenario | Purpose | Subjective terms expected | False-positive traps | Replicates |
|---|---|---|---|---|
| `hero-cool` | Gate threshold rows 1, 2, 3 | `cool` | `5`, `1-10`, `government` | 10 |
| `hero-opt-out` | Gate threshold row 4 (opt-out path) | none (opt-out skips surfacing) | n/a | 10 |
| `subjective-important` | Surfacing generalises | `important` | `20`, `latest`, `articles` | 2-3 |
| `subjective-risky` | Surfacing generalises (consequential term) | `risky` | `transactions`, `csv`, `json` | 2-3 |
| `subjective-beautiful` | Surfacing generalises (aesthetic term) | `beautiful` | `10`, `famous`, `paintings` | 2-3 |
| `concrete-no-surface` | Zero subjective terms → zero calls | none | n/a (pure no-surface test) | 2-3 |
| `numeric-quantifier-trap` | Numeric quantifiers must NOT surface | `helpful` | `5`, `top`, `1-10`, `scale` | 2-3 |
| `temporal-anchor-not-subjective` | Temporal anchors must NOT surface | `quality` | `before 2020`, `recent`, `this year` | 2-3 |
| `multiple-subjective-terms` | Multi-surface in one prompt | `engaging`, `authoritative` | `10`, `combined`, `score` | 2-3 |
| `silent-bake-anti-pattern` | Surfacing survives "you decide" pressure | `trustworthy` | `50`, `articles`, `sources` | 2-3 |

Replicate-count for non-gate scenarios is operator discretion. Three runs each is a reasonable spot-check; if any shows >1 RED, raise the replicate count or escalate.

## Running the suite

The scenarios use the standard `evals/composer-rgr/run_scenario.sh` harness with `ELSPETH_RGR_SCENARIO` set to each scenario's `scenario.json`. The harness captures `messages.json`, `state.json`, and a session ID per run.

```bash
cd evals/composer-rgr
set -a; . .env; set +a   # standard staging env (see ../README.md)

# Hero gate — 10 replicates
for i in $(seq 1 10); do
  ELSPETH_RGR_SCENARIO=phase5b-interpretation/scenarios/hero-cool/scenario.json \
    ./run_scenario.sh "hero-cool-r$i"
done

# Opt-out gate — 10 replicates
# IMPORTANT: opt-out requires a pre-step that sets interpretation_review_disabled=true
# on the freshly created session BEFORE the opening prompt is sent. See §Opt-out scenarios below.
for i in $(seq 1 10); do
  ELSPETH_RGR_SCENARIO=phase5b-interpretation/scenarios/hero-opt-out/scenario.json \
    ./run_scenario.sh "hero-opt-out-r$i"
done

# Spot-check breadth — 3 replicates each
for scenario in subjective-important subjective-risky subjective-beautiful \
                concrete-no-surface numeric-quantifier-trap \
                temporal-anchor-not-subjective multiple-subjective-terms \
                silent-bake-anti-pattern; do
  for i in 1 2 3; do
    ELSPETH_RGR_SCENARIO=phase5b-interpretation/scenarios/$scenario/scenario.json \
      ./run_scenario.sh "$scenario-r$i"
  done
done
```

Cost guideline: ~$0.30–$0.60 per run. Full suite (10 + 10 + 8×3 = 44 runs) ≈ $15–$25 per gate verification.

## Manual scoring (the part `score.py` cannot do)

The existing `score.py` consumes `red_criteria` / `green_criteria` and detects passivity, build-failure sentinels, and structural state properties. **It cannot inspect tool-call audit envelopes or interpretation_events rows.** Three of the four Phase 5b assertions need those signals. The operator runs a small post-script per run to derive the Phase 5b booleans:

### 1. `cool_surfaced` (and other surfacing booleans)

The composer drops in-loop tool-call assistant turns from the persisted `chat_messages` table — they live in the per-message `tool_calls` JSON envelope only (per `service.py:1018-1035`). Decode via:

```bash
.venv/bin/python -m evals.lib.decode_tools data/sessions.db "$(cat runs/<run-dir>/session_id.txt)"
```

Search the decoded output for:

```text
tool_name='request_interpretation_review' arguments={'user_term': 'cool', ...}
```

A hit means the LLM called the tool for that term in that session. Walk every tool-call envelope; record the set of `(user_term)` values that triggered a call. Compare against the scenario's `must_call_request_interpretation_review_with_user_term` and `must_not_call_request_interpretation_review_with_user_terms` arrays.

### 2. `placeholder_emitted_same_turn`

Read `runs/<run-dir>/state.json` and look at every node where `node_type='llm_transform'`:

```bash
jq '.nodes[] | select(.node_type == "llm_transform") | .options.prompt_template' \
  runs/<run-dir>/state.json
```

A hit on `{{interpretation:cool}}` (or whichever term) means the placeholder reached the staged state. For same-turn verification: the `decode_tools.py` output shows tool calls in chronological order grouped by assistant message. The `upsert_node` call carrying the placeholder must appear in the SAME assistant message as the `request_interpretation_review` call.

### 3. `opt_out_disables_surfacing`

Query the session database directly:

```bash
sqlite3 data/sessions.db <<SQL
SELECT user_term, interpretation_source, choice
  FROM interpretation_events
 WHERE session_id = '<session-id-from-runs/.../session_id.txt>';
SQL
```

For an opt-out run, expect:

- Zero `request_interpretation_review` tool calls in the decoded audit (confirms surfacing was skipped).
- One or more rows with `interpretation_source='auto_interpreted_opt_out'` (confirms the auto-interpreted writer ran).

## Results artifact format

After running the suite, the operator commits `results-<utc-date>.json` to this directory. One JSON object per run, schema matching 18a Task 12 spec:

```json
{
  "suite": "phase5b-interpretation",
  "model_id": "openrouter/openai/gpt-5.4-mini",
  "skill_hash": "<sha256 of pipeline_composer.md at run time>",
  "service_git_sha": "<HEAD sha of elspeth at staging-deploy time>",
  "started_utc": "2026-05-NN T HH:MM:SSZ",
  "finished_utc": "2026-05-NN T HH:MM:SSZ",
  "runs": [
    {
      "run_index": 1,
      "scenario": "hero-cool",
      "run_dir": "runs/20260519T120000Z-hero-cool-hero-cool-r1",
      "session_id": "...",
      "timestamp_utc": "...",
      "model_id": "openrouter/openai/gpt-5.4-mini",
      "hero_prompt": "create a list of 5 government web pages and use an LLM to rate how cool they are",
      "score_verdict": "GREEN",
      "cool_surfaced": true,
      "numeric_quantifiers_not_surfaced": true,
      "placeholder_emitted_same_turn": true,
      "opt_out_disables_surfacing": null,
      "operator_notes": ""
    }
  ],
  "threshold_summary": {
    "hero_cool_surfaced": "10/10 (≥8 required)",
    "hero_numeric_not_surfaced": "10/10 (≥9 required)",
    "hero_placeholder_same_turn": "9/10 (≥8 required)",
    "opt_out_writes_opt_out_row": "10/10 (≥9 required)"
  },
  "gate_passed": true
}
```

`opt_out_disables_surfacing` is `null` for non-opt-out scenarios (the field only applies to `hero-opt-out` runs). `gate_passed` is `true` only when all four thresholds in `threshold_summary` are met.

## Opt-out scenarios

The `hero-opt-out` scenario requires a pre-step the standard harness does not handle: after `POST /api/sessions` creates the session but before the first `POST /messages`, the harness must opt the session out of interpretation review by calling the dedicated opt-out endpoint shipped in Phase 5b Task 7.

Endpoint: `POST /api/sessions/{session_id}/interpretations/opt_out`. Request body is empty. Response is the `InterpretationOptOutResponse` envelope (`session_id`, `interpretation_review_disabled: true`, `opted_out_at`). The route is idempotent per F-29: repeated POSTs return the original `opted_out_at` from the first call rather than re-stamping.

Until a `--opt-out` flag is added to `run_scenario.sh`, do this manually:

```bash
# After run_scenario.sh starts and prints the session_id, but before the
# 'send opening prompt' step — OR, run the harness once, retrieve the
# session_id, kill before the message send. Cleaner: extend run_scenario.sh.
SID=$(cat runs/<run-dir>/session_id.txt)
curl -sS -X POST "$ELSPETH_EVAL_BASE_URL/api/sessions/$SID/interpretations/opt_out" \
  -H "Authorization: Bearer $JWT"
```

If the opt-out endpoint is unreachable for any reason, the `hero-opt-out` scenario is BLOCKED. Surface to the operator rather than skip-pass.

## What this suite deliberately does not test

- **Rate-limit behaviour** (per-term cap 3, per-session-day cap 10). These are integration-tested directly against the rate-limit code paths; an eval can't reliably exercise the boundary without prompt manipulation that itself shifts surfacing behaviour.
- **Credential-shape rejection** on `user_term` / `llm_draft`. Integration-tested in `tests/integration/web/composer/` per Task 9.
- **Hash-chain integrity** between `interpretation_events.resolved_prompt_template_hash` and `calls.resolved_prompt_template_hash`. That's the Task 9 cross-DB hash-equality test, not an LLM eval.
- **Runtime placeholder resolution errors**. Pipeline execution failures from unresolved `{{interpretation:...}}` placeholders are integration-tested per Task 5 §"Unresolved placeholder runtime detection".

The eval suite tests one specific thing: the composer LLM's surfacing behaviour and its respect for the opt-out flag. Everything else is unit or integration scope.

## Skill prerequisites

Per `feedback_no_tests_for_skill_prompts`: this suite IS the validation for skill changes. After any edit to `src/elspeth/web/composer/skills/pipeline_composer.md` touching the interpretation-surfacing language (Task 8), re-run the gate. The skill is read once at module import (see `prompts.py`), so:

```bash
sudo systemctl restart elspeth-web.service
```

…between every edit and the next gate run, or the eval measures stale skill content.

## When this suite is consulted

Run when:

- Phase 5b PR is being assembled — gate must pass before merge.
- The composer skill is amended in a way that could affect surfacing behaviour (Task 8 amendments, or broader skill rewrites).
- A user reports an interpretation-surfacing failure in production — reproduce by adding their prompt as a new scenario before debugging.

Don't run for:

- Unrelated composer changes (cost adds up; full gate ≈ $15–$25).
- Local skill development before staging deploy. Restart the staging service first or the suite measures stale prompts.
