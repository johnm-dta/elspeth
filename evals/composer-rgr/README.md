# composer-rgr — minimal Red/Green/Refactor harness for the pipeline composer skill

A purpose-built harness for hardening the `pipeline_composer.md` skill against
empirically-observed failure modes. Distinct from `evals/composer-harness/`
(persona-driven 15-scenario sweep) — this one runs a single deterministic
scenario, scores it programmatically, and is designed for fast skill-edit
iteration cycles.

## Why this exists

The user reported two failure modes in the composer:

1. **Passivity** — the LLM asks permission ("If you want, I can…") instead of
   acting on a clearly-authorised request.
2. **Schema-blindness** — the LLM constructs malformed `set_pipeline` calls
   without using the schema/contract evidence available to it.

Both modes were captured in a real session
(`e7d42525-bd73-4838-968c-647ea73cce98`) where the model produced two
text-only assistant turns containing the exact forbidden phrase "If you want,
I can fix this by…" with **zero successful tool calls**.

## What the harness does

`run_scenario.sh`:

1. Logs in to the staging deploy as the eval user.
2. Creates a fresh composer session.
3. POSTs the scenario's `opening_prompt` (a real failure case: "Please create
   a pipeline that downloads this text file: `<URL>` and then explodes it
   into a json file - individual lines").
4. Captures the persisted message thread and the final composition state.
5. Hands them to `score.py` for verdict.

`score.py` reads `scenario.json`'s `red_criteria` / `green_criteria` and
returns a JSON verdict (`RED`, `AMBER`, or `GREEN`).

### Detection rules (in order of reliability)

1. **Build-failure sentinels in final assistant content** — the strings
   "I cannot mark this pipeline complete" / "runtime preflight failed" are
   server-injected (`service.py:_build_runtime_preflight_message`) when the
   model declared completion but the pipeline failed preflight. Definitive
   RED.
2. **`is_valid: false` or null state** — independent of message content,
   catches cases where the model surrenders silently.
3. **Forbidden passivity phrases in final assistant content** — substring
   match against the explicit list the skill forbids. Catches the
   "If you want, I can…" pattern.

Tool-sequence visibility: `run_scenario.sh` fetches the messages thread with
`?include_tool_rows=true&include_raw_content=true&limit=500`, which surfaces
the `role=tool` rows in `messages.json`. Each tool row carries a `success`
discriminator and the raw JSON tool result, so sequence-aware scoring rules
(e.g. `red.max_persisted_tool_calls`,
`green.must_discover_schema_before_first_mutation`) in
`evals/lib/composer_rgr_score.py` operate directly on this stream. For
deeper introspection beyond what the API exposes, query the
`chat_messages.tool_calls` JSON column in `data/sessions.db` (role='tool')
— that captures every audit-recorded invocation including any internal
LLM↔tool turns the API path filters out.

## Usage

```bash
cd evals/composer-rgr
cat > .env <<EOF
ELSPETH_EVAL_BASE_URL=https://elspeth.foundryside.dev
ELSPETH_EVAL_USER=dta_user
ELSPETH_EVAL_PASS=dta_pass
EOF
chmod 600 .env

# Run one cycle (login → session → message → score)
set -a; . .env; set +a
./run_scenario.sh red1     # before edits
# ... edit src/elspeth/web/composer/skills/pipeline_composer.md ...
sudo systemctl restart elspeth-web.service
./run_scenario.sh green1   # after edits
```

Each run lands in `runs/<utc-ts>-<label>/` with `messages.json`,
`state.json`, `scoring.json`, `session_id.txt`, and a session URL printed to
stderr.

### Diagnostic helper

`evals/lib/decode_tools.py` decodes the chronological `chat_messages.tool_calls`
audit envelopes for a session — useful when triaging a RED run. CLI:

```bash
.venv/bin/python -m evals.lib.decode_tools data/sessions.db "$(cat runs/<run-dir>/session_id.txt)"
```

## Restarting matters

The skill is read once at process start (`prompts.py:23` —
`_PIPELINE_SKILL = load_skill("pipeline_composer")` runs at module import,
and `build_system_prompt` is `@lru_cache`'d). To pick up edits to
`pipeline_composer.md`, restart the service:

```bash
sudo systemctl restart elspeth-web.service
```

## Initial RGR results (2026-05-06)

Captured against staging (`openrouter/openai/gpt-5.4`, the deploy's default
composer model).

**Baseline (3 runs, unedited skill):** 3/3 RED. All hit the build-failure
sentinel; all left the pipeline `is_valid: false`. Specific shapes:
- run 1: empty config (no source / no sinks)
- run 2: `web_scrape` only, missing `line_explode`
- run 3: `web_scrape` + `line_explode` but disconnected from source

Plus one captured human session
(`e7d42525-bd73-4838-968c-647ea73cce98`): two assistant turns, zero tool
calls, "If you want, I can fix this by…" twice — the textbook passivity
failure.

**After three skill edits** (TERMINATION GATE, Connection-Model rewrite,
Pattern-1b connection-name idiom, anti-tail-offer reinforcement) **across 9
GREEN-attempt runs**:

- 3/9 hard GREEN (valid pipeline, clean reply) — green3, green6, green9
- 2/9 functional but soft-RED (valid pipeline + "If you want, I can…"
  follow-up offer) — green5, green7
- 4/9 hard RED (model still failed to construct valid set_pipeline call)

Caveat: model nondeterminism on a 3-run-per-iteration sample is large; the
trend is real but the exact percentages should not be over-interpreted.

The dominant remaining failure mode is **schema construction** — the model
sometimes builds `set_pipeline` calls with mismatched connection-name
strings (e.g. `node.input: "source"` when no upstream `on_success` produces
`"source"`) and gives up after a small number of retries. The Connection
Model and Pattern 1b edits target this directly but the model still
occasionally regresses.

The captured passivity phrase pattern (`"If you want, I can fix this by…"`
as a stalling primary response) **did not reproduce** in any post-edit run
— passivity now appears only as a follow-up tail offer, which the
`anti-tail-offer` edit targets.

## What this harness deliberately does not test

- Multi-turn conversations. Real failures often emerge after user pushback;
  this single-turn harness can't reach those. Use the older
  `evals/composer-harness/` for multi-turn persona scenarios (its shared
  `evals/lib/preflight.sh` was repaired in commit `1ca34527` to use
  `/api/catalog/sources`, so `--doctor` now passes).
- Different scenarios. One opening prompt covers URL+download+line-explode;
  other failure modes (e.g. fork-and-route, conflicting consumer
  requirements, content-safety pipelines) need their own scenarios. Add
  them by writing a new `scenario_<id>.json` and pointing `score.py` at it.
- Subjective quality of the model's natural-language replies. The scorer
  judges only on objective signals.

## When to run / not run

Run when:
- You're editing the pipeline composer skill and need empirical evidence the
  edit changed behaviour
- You've fixed a server-side composer bug and want regression evidence
- You're investigating a user-reported composer failure and need a
  reproducer

Don't run for:
- Unrelated composer changes (LLM cost adds up; ~$0.50/run)
- Local skill development before deploying to staging — restart the staging
  service first or the harness measures stale skill content
