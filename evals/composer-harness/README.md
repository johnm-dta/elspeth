# ELSPETH Composer Hard-Mode Eval Harness

A reusable, persona-driven evaluation harness for the ELSPETH composer's
LLM-based pipeline-building behaviour. Drives the composer through scripted
HTTP calls plus a parent-agent persona-subagent dispatch loop, captures every
artefact end-to-end, and emits an auditable per-scenario ledger plus a
cross-scenario scorecard.

## Status

This is the *forward-going* harness. The 2026-05-03 historical eval lives
under `evals/2026-05-03-composer/` and is frozen evidence backing the reports
in `docs/composer/evidence/composer-*-2026-05-03.md`. Don't mutate that folder; iterate here.

## Layout

```
evals/
├── lib/                            shared by any future eval suite
│   ├── common.sh                     HTTP/env/JWT/poll primitives
│   ├── preflight.sh                  doctor check (env, login, API roundtrip)
│   └── dispatch-protocol.md          persona-subagent dispatch contract
└── composer-harness/
    ├── README.md                   ← you are here
    ├── .env.example                  required env vars
    ├── personas/                     character specs (in-character drivers)
    │   ├── p1_compliance.md
    │   ├── p2_researcher.md
    │   ├── p3_marketingops.md
    │   └── p4_adversarial_engineer.md
    ├── scenarios/hardmode/           4 personas × 3-4 task classes = 15 scenarios
    │   ├── p1_t{1,2,3,4}_*.json
    │   ├── p2_t{1,2,3,4}_*.json
    │   ├── p3_t{1,2,3,4}_*.json
    │   └── p4_t{1,2,3}_*.json
    ├── hardmode/                     scripts
    │   ├── harness.sh                bootstrap (login → session → blob)
    │   ├── post_message.sh           per-turn driver (POST /messages + metrics)
    │   ├── finalize_scenario.sh      validate → execute → poll-to-terminal → ledger
    │   ├── replay.sh                 engine-only re-run (no LLM)
    │   ├── aggregate.sh              cross-scenario aggregate/summary + SCORECARD.md
    │   ├── validate_persona.sh       linguistic-constraint check on turns
    │   └── RUNBOOK.md                step-by-step
    └── runs/                         outputs (gitignored at this level)
        └── <utc-date>-hardmode/
            └── <scenario_id>/
                ├── scenario.json, session.json, sid.txt, jwt.txt
                ├── turn{N}.user.txt
                ├── msg.t{N}.{req,resp,curl_meta}.json
                ├── state.{before,after}.t{N}.json
                ├── progress.t{N}.json, metrics.t{N}.json
                ├── validate.json, execute.{json,code}
                ├── run.json, diagnostics.json, diagnostics.json.code
                ├── final_yaml.json, final_yaml.json.code
                ├── messages.json, messages.json.code
                ├── artifact_collection_errors.json
                ├── ledger.json
                ├── persona_check.json    (after validate_persona.sh)
                └── replays/<utc-ts>/     (after replay.sh)
```

## Quick start

```bash
cd evals/composer-harness
cp .env.example .env && $EDITOR .env

# Verify env / login / API:
hardmode/harness.sh --doctor

# Bootstrap one scenario:
SCENARIO=p1_t1_happy
export ELSPETH_EVAL_RUNS_DIR="${ELSPETH_EVAL_RUNS_DIR:-runs/$(date -u +%Y-%m-%d)-hardmode}"
RUN_DIR="$ELSPETH_EVAL_RUNS_DIR/$SCENARIO"
hardmode/harness.sh "$SCENARIO"

# Drive the dialogue (turn 1 from the fixture; turn 2+ from a persona-subagent):
jq -r '.opening_prompt' "$RUN_DIR/scenario.json" > "$RUN_DIR/turn1.user.txt"
hardmode/post_message.sh "$SCENARIO" 1 "$RUN_DIR/turn1.user.txt"
# ... see lib/dispatch-protocol.md for the persona-subagent loop

# Finalize:
hardmode/finalize_scenario.sh "$SCENARIO"

# After all scenarios are done:
hardmode/aggregate.sh "$ELSPETH_EVAL_RUNS_DIR"
cat "$ELSPETH_EVAL_RUNS_DIR/SCORECARD.md"
# Also writes aggregate.json and aggregate_summary.json in that run root.
```

Full step-by-step is in [`hardmode/RUNBOOK.md`](hardmode/RUNBOOK.md).

## Composer model and generated model fields

The composer LLM under test is the running web service's
`WebSettings.composer_model`, not a value controlled by this harness. In
settings/env form that field is `ELSPETH_WEB__COMPOSER_MODEL`; the source
default lives in `src/elspeth/web/config.py`. On staging, inspect only the
targeted key in `deploy/elspeth-web.env` before running, and restart
`elspeth-web.service` if you change it.

`openai/gpt-5-mini` appears in the eval suite as a **generated pipeline model
field**, not as the composer model:

1. **`scenarios/hardmode/p4_t1_happy_csv_to_jsonl.json`** — Dev (P4) is the only
   persona who *prescribes* a transform model. Her opening prompt names
   `openai/gpt-5-mini` explicitly, so the composer's `final_yaml.json` for that
   scenario should pin the same id. If it doesn't, that's a probe failure
   (composer ignored a prescriptive ask).

2. **`personas/p4_adversarial_engineer.md`** — references `openrouter/5-mini`
   as Dev's habitual shorthand (a misconception probe — does the composer
   resolve the shorthand or push back?).

For the **P1/P2/P3 scenarios**, the persona deliberately does **not** name a
transform model — the test is whether the composer chooses a model identifier
that actually validates and runs when it builds an LLM transform. The harness
cannot force that generated YAML value from outside the running session.

Changing `ELSPETH_WEB__COMPOSER_MODEL` changes the agent that writes the YAML;
it does **not** force every generated LLM transform to use the same `model:`.
If you want every scenario in the matrix to emit `openai/gpt-5-mini`, make that
an explicit eval contract or composer behavior, then verify it in
`final_yaml.json`.

The harness will faithfully record whatever transform model lands in
`final_yaml.json` per scenario — verify the expected id post-run with:

```bash
RUN_ROOT="${ELSPETH_EVAL_RUNS_DIR:-runs/$(date -u +%Y-%m-%d)-hardmode}"
for d in "$RUN_ROOT"/*/; do
  echo "=== $(basename "$d") ==="
  jq -r '.yaml' "$d/final_yaml.json" 2>/dev/null | grep -E '^\s+model:' || echo "(no model in YAML)"
done
```

## What the harness measures (and what it doesn't)

**Authoritative measurements** (read these directly):
- The composer's literal responses (`msg.t{N}.resp.json` → `.message.content`)
- The final YAML (`final_yaml.json` → `.yaml`)
- The validate matrix (`validate.json` → `.checks[]`)
- Run outcome (`run.json` → `.status`, `.rows_succeeded`, `.error`)
- Per-row engine errors (`diagnostics.json`)
- Optional-artifact failures (`artifact_collection_errors.json` and
  `ledger.json` → `.artifact_collection_errors[]`)
- Run/turn evidence manifests (`suite_manifest.json`, `run_manifest.json`,
  `turn<N>.manifest.json`) with fixture/persona hashes, dispatch contract,
  session id, posted artifact paths, and turn transport status.

**Heuristic signals** (read with appropriate skepticism):
- `metrics.t{N}.json`'s `*_keyword_match` fields are **noisy** — keyword-based
  detection of clarifying questions / volunteered limits. Use as filtering
  aids, not authoritative judgments.
- `persona_check.json` extracts MUST USE / MUST AVOID phrases from a persona
  spec via regex on quoted/backticked phrases — won't catch prose-style
  constraints like "avoid hedges".
- `provider_usage` in `ledger.json` and `aggregate_summary.json` is
  best-effort metadata extracted from diagnostics. It is useful for token
  accounting when the composer exposes provider usage, but it is not a billing
  statement. Check `token_usage_available` before treating token totals as
  observed data; legacy ledgers and diagnostics without usage metadata are
  reported as unavailable, not zero.
- `cost` in `ledger.json` and `aggregate_summary.json` is populated only when
  the composer LLM audit sidecars expose provider-reported cost metadata
  (`response.usage.cost` on the OpenRouter/LiteLLM response path). Missing
  cost remains explicit metadata, not a wall-time estimate.

**Things the harness deliberately does not do**:
- It does not assert a pass/fail outcome — `pass_criterion` is in the fixture
  for human judgment after reading the artefacts. Hard-mode evals on LLM
  output are inherently subjective; the harness produces evidence, not
  verdicts.
- It does not score the composer's responses against a rubric. That's a
  follow-up evaluation step you can add by reading the captured ledgers.
- It does not fabricate dollar cost from wall-clock time. The harness records
  OpenRouter/LiteLLM response cost when the server exposes it via LLM-call audit
  sidecars; otherwise `cost.source = "not_available"`. Use provider billing as
  the reconciliation source for any suite where cost coverage is incomplete.

## When to use replay.sh

You don't need to redo the persona dialogue every time the engine changes.
After capturing a scenario once, `hardmode/replay.sh <run_dir>` re-runs the
engine half (validate + execute + poll) against the captured `final_yaml.json`
in a fresh session, leaving the original capture intact.

This makes the persona dialogue a one-time cost. Engine fixes can be
verified against captured runs in seconds for free, without burning more
LLM credit.

## Artefact integrity

- JWTs are written to `runs/.../<scenario_id>/jwt.txt` at mode 600.
- The harness auto-refreshes the JWT when its `exp` claim falls within
  `ELSPETH_EVAL_JWT_REFRESH_MARGIN_SEC` of expiry.
- Run outputs are date-stamped, so re-runs accumulate side-by-side rather
  than overwriting. Use `--fresh` to reset a single scenario.
- All script invocations append to `runs/.../<scenario_id>/harness.log`
  with timestamped events.
- `harness.sh` writes `suite_manifest.json` at the run root and
  `run_manifest.json` per scenario. These record the hardmode dispatch contract,
  message budget, session id, and scenario/persona hashes.
- `post_message.sh` writes `turn<N>.manifest.json` alongside
  `metrics.t<N>.json`. Non-2xx `/messages` responses are typed as
  `turn_status.status = "transport_error"` and the response body is preserved.
- `finalize_scenario.sh` treats optional artifact collection failures
  (`diagnostics`, final YAML export, message export) as evidence metadata:
  fallback JSON is written, the HTTP code is preserved in `<artifact>.code`,
  and the scenario ledger is still emitted.
- `finalize_scenario.sh` requests `messages?include_llm_audit=true` so
  `messages.json` can include safe LLM-call sidecars with model, token, and
  provider-cost metadata. Full tool-dispatch audit rows remain hidden from the
  HTTP response.
- Each finalize starts by resetting generated `run.json`, `diagnostics.json`,
  and `artifact_collection_errors.json` so a reused run directory cannot leak
  stale engine or usage data into the new ledger.
- `aggregate.sh` writes `aggregate_errors.json`; malformed ledgers are counted
  in `aggregate_summary.json` instead of disappearing from the suite silently.

## Adding a new scenario

1. Add a fixture JSON under `scenarios/hardmode/` named `<persona_id>_t<N>_<class>_<slug>.json`
   (the harness resolves by `<persona_id>_t<N>_*` glob, so the slug is free-form).
2. Required fields: `scenario_id`, `persona`, `task_class`, `task_summary`,
   `opening_prompt`, `probe`, `product_capability_used`, `expected_outcome`,
   `pass_criterion`. Optional: `csv_filename`, `csv_content`.
3. Reference the persona by `persona_id` matching a `personas/<persona_id>.md`
   file.
4. Run `hardmode/harness.sh --doctor` first, then
   `hardmode/harness.sh <scenario_id>` without `--doctor`.

## Adding a new persona

1. Add `personas/<persona_id>.md` following the structure of the existing
   four (Bio, Cognitive style, Linguistic constraints with MUST USE / MUST
   AVOID, Knowledge gaps, Stop conditions, Failure mode).
2. Add at least one scenario per task class (happy/edge/limit) referencing
   the new persona.
3. The `validate_persona.sh` extractor uses double-quoted and backticked
   phrases from the MUST USE / MUST AVOID sections — keep yours quoted.
