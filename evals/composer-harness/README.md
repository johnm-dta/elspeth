# ELSPETH Composer Hard-Mode Eval Harness

A reusable, persona-driven evaluation harness for the ELSPETH composer's
LLM-based pipeline-building behaviour. Drives the composer through scripted
HTTP calls plus a parent-agent persona-subagent dispatch loop, captures every
artefact end-to-end, and emits an auditable per-scenario ledger plus a
cross-scenario scorecard.

## Status

This is the *forward-going* harness. The 2026-05-03 historical eval lives
under `evals/2026-05-03-composer/` and is frozen evidence backing the reports
in `notes/composer-*-2026-05-03.md`. Don't mutate that folder; iterate here.

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
    │   ├── aggregate.sh              cross-scenario aggregate.json + SCORECARD.md
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
                ├── run.json, diagnostics.json
                ├── final_yaml.json, messages.json
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
hardmode/harness.sh p1_t1_happy

# Drive the dialogue (turn 1 from the fixture; turn 2+ from a persona-subagent):
jq -r '.opening_prompt' runs/$(date -u +%Y-%m-%d)-hardmode/p1_t1_happy/scenario.json \
  > runs/$(date -u +%Y-%m-%d)-hardmode/p1_t1_happy/turn1.user.txt
hardmode/post_message.sh p1_t1_happy 1 \
  runs/$(date -u +%Y-%m-%d)-hardmode/p1_t1_happy/turn1.user.txt
# ... see lib/dispatch-protocol.md for the persona-subagent loop

# Finalize:
hardmode/finalize_scenario.sh p1_t1_happy

# After all 9-15 scenarios done:
hardmode/aggregate.sh
cat runs/$(date -u +%Y-%m-%d)-hardmode/SCORECARD.md
```

Full step-by-step is in [`hardmode/RUNBOOK.md`](hardmode/RUNBOOK.md).

## Model under test

The model exercised by the eval is **`openai/gpt-5-mini`** (via OpenRouter).
This name appears in two places that the harness controls:

1. **`scenarios/hardmode/p4_t1_happy_csv_to_jsonl.json`** — Dev (P4) is the only
   persona who *prescribes* a model. Her opening prompt names
   `openai/gpt-5-mini` explicitly, so the composer's `final_yaml.json` for that
   scenario should pin the same id. If it doesn't, that's a probe failure
   (composer ignored a prescriptive ask).

2. **`personas/p4_adversarial_engineer.md`** — references `openrouter/5-mini`
   as Dev's habitual shorthand (a misconception probe — does the composer
   resolve the shorthand or push back?).

For the **P1/P2/P3 scenarios**, the persona deliberately does **not** name a
model — the test is whether the composer auto-selects something that actually
works on OpenRouter (this is the failure surface the historical
`elspeth-obs-f3143acba2` flagged). The harness cannot force the composer's
auto-pick from outside the running session.

If you want every scenario in the matrix to land on `openai/gpt-5-mini`,
you need to set the composer's server-side default model to that id before
running. Two options on the staging deploy:

- **Composer config**: whatever the deploy's composer-LLM-default config key
  is — depends on the running build of `src/elspeth/web/composer/`.
- **Environment override on the web service**: if the composer respects an
  env var for its default model (check the running deploy's systemd unit),
  set it and `systemctl restart elspeth-web.service` before bootstrapping
  any scenarios.

The harness will faithfully record whatever model lands in `final_yaml.json`
per scenario — verify the expected id post-run with:

```bash
for d in runs/$(date -u +%Y-%m-%d)-hardmode/*/; do
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

**Heuristic signals** (read with appropriate skepticism):
- `metrics.t{N}.json`'s `*_keyword_match` fields are **noisy** — keyword-based
  detection of clarifying questions / volunteered limits. Use as filtering
  aids, not authoritative judgments.
- `persona_check.json` extracts MUST USE / MUST AVOID phrases from a persona
  spec via regex on quoted/backticked phrases — won't catch prose-style
  constraints like "avoid hedges".

**Things the harness deliberately does not do**:
- It does not assert a pass/fail outcome — `pass_criterion` is in the fixture
  for human judgment after reading the artefacts. Hard-mode evals on LLM
  output are inherently subjective; the harness produces evidence, not
  verdicts.
- It does not score the composer's responses against a rubric. That's a
  follow-up evaluation step you can add by reading the captured ledgers.
- It does not check for cost overruns. OpenRouter cost is observable through
  the composer's audit DB; the harness deliberately doesn't reach into that
  side-channel.

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

## Adding a new scenario

1. Add a fixture JSON under `scenarios/hardmode/` named `<persona_id>_t<N>_<class>_<slug>.json`
   (the harness resolves by `<persona_id>_t<N>_*` glob, so the slug is free-form).
2. Required fields: `scenario_id`, `persona`, `task_class`, `task_summary`,
   `opening_prompt`, `probe`, `product_capability_used`, `expected_outcome`,
   `pass_criterion`. Optional: `csv_filename`, `csv_content`.
3. Reference the persona by `persona_id` matching a `personas/<persona_id>.md`
   file.
4. Run `hardmode/harness.sh <scenario_id> --doctor` (well, run `--doctor` first
   without scenario, then `harness.sh <scenario_id>` without flag).

## Adding a new persona

1. Add `personas/<persona_id>.md` following the structure of the existing
   four (Bio, Cognitive style, Linguistic constraints with MUST USE / MUST
   AVOID, Knowledge gaps, Stop conditions, Failure mode).
2. Add at least one scenario per task class (happy/edge/limit) referencing
   the new persona.
3. The `validate_persona.sh` extractor uses double-quoted and backticked
   phrases from the MUST USE / MUST AVOID sections — keep yours quoted.
