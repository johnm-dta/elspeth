# Hard-Mode Eval Agent Prompt

Paste the section between the `BEGIN PROMPT` / `END PROMPT` markers into a new
Claude Code session as the user message. The agent will run the full 15-scenario
hard-mode eval end-to-end and produce a report.

This prompt is self-contained. It tells the agent where to find the harness,
the protocol, and the deliverables. It does NOT re-explain HTTP plumbing or
the persona-subagent contract — it points at the load-bearing reference files
and requires the agent to read them.

---

```
================ BEGIN PROMPT ================

You are running a full hard-mode evaluation of the ELSPETH composer's
LLM-driven pipeline-building behaviour. The composer model under test is the
running service's `WebSettings.composer_model` (`ELSPETH_WEB__COMPOSER_MODEL`
in env form); record it during pre-flight. Do not assume it is
`openai/gpt-5-mini` — that id is a scenario-requested LLM transform model in
P4 T1, not the composer LLM.

## Your mission

Execute every scenario fixture under
`/home/john/elspeth/evals/composer-harness/scenarios/hardmode/` (15 scenarios)
end-to-end using the harness scripts at
`/home/john/elspeth/evals/composer-harness/hardmode/`, then produce a report
in `docs/composer/evidence/composer-eval-hardmode-<UTC-date>.md` summarising findings.

## Authorization

- **Budget**: ~$10 of OpenRouter credit is anticipated. Don't ask before
  running scenarios. Track spend from provider billing and harness cost
  metadata; do not estimate dollars from wall time. If provider billing shows
  actual spend above $20, stop and report.
- **Time**: ~2-3 hours of supervised dispatch work plus engine time. Don't
  ask whether to continue between scenarios; just continue.
- **Side effects authorized**: HTTP calls to the staging deploy at
  `$ELSPETH_EVAL_BASE_URL`, file writes under `evals/composer-harness/runs/`,
  reading `evals/2026-05-03-composer/` for historical comparison.
- **NOT authorized without asking**: git commits/pushes (capture results;
  the user will commit), modifying source code, modifying scenario fixtures
  or persona specs (those are immutable test contracts), running the
  composer outside the eval harness.

## Step 1: Read the load-bearing references

You MUST read these files before starting. They contain the protocol; this
prompt does not duplicate them.

1. `/home/john/elspeth/evals/composer-harness/README.md` — overview, model
   notes, what's measured.
2. `/home/john/elspeth/evals/composer-harness/hardmode/RUNBOOK.md` — the
   step-by-step you will follow.
3. `/home/john/elspeth/evals/lib/dispatch-protocol.md` — the persona-subagent
   contract. Read this twice. The subagent dispatch is the load-bearing
   discipline of this eval.
4. `/home/john/elspeth/evals/composer-harness/personas/*.md` — all 4 persona
   specs. You will dispatch fresh subagents seeded from these.

## Step 2: Pre-flight

```bash
cd /home/john/elspeth/evals/composer-harness
ls .env || echo "ENV MISSING — ask user for ELSPETH_EVAL_BASE_URL/USER/PASS"
hardmode/harness.sh --doctor
grep '^ELSPETH_WEB__COMPOSER_MODEL=' /home/john/elspeth/deploy/elspeth-web.env || \
  echo "composer model not found in deploy env; record the verified source"
```

Expected: `all preflight checks passed`. If `.env` is missing, ask the user
once for the values (don't guess; staging URL changes; don't re-use the
hardcoded values from the historical 2026-05-03 harness — that file is
intentionally sanitised).

## Step 3: Run the 15 scenarios

The full list, in execution order:

```
p1_t1_happy   p1_t2_edge   p1_t3_limit   p1_t4_stress
p2_t1_happy   p2_t2_edge   p2_t3_limit   p2_t4_stress
p3_t1_happy   p3_t2_edge   p3_t3_limit   p3_t4_stress
p4_t1_happy   p4_t2_edge   p4_t3_limit
```

Recommended order: run the limit-probes first (cheapest, validate the harness
works), then happy-paths, then edges, then stress (most expensive last so a
budget-stop doesn't lose the cheap data).

For each scenario, the loop is exactly what RUNBOOK.md describes. Briefly:

```
export ELSPETH_EVAL_RUNS_DIR="${ELSPETH_EVAL_RUNS_DIR:-runs/$(date -u +%Y-%m-%d)-hardmode}"
RUN_DIR="$ELSPETH_EVAL_RUNS_DIR/<sid>"
hardmode/harness.sh <sid>
jq -r '.opening_prompt' "$RUN_DIR/scenario.json" > "$RUN_DIR/turn1.user.txt"
hardmode/post_message.sh <sid> 1 "$RUN_DIR/turn1.user.txt"
# ... persona-subagent loop for turn 2+ until DONE or budget (5 turns)
hardmode/finalize_scenario.sh <sid>
hardmode/validate_persona.sh "$RUN_DIR"
```

### Persona-subagent dispatch — THE LOAD-BEARING PART

For every turn N >= 2, dispatch a fresh `general-purpose` subagent with this
template. **Do not write the persona's reply yourself.** Do not re-use a
single subagent across turns. A *fresh* dispatch each turn is required —
that's what re-grounds the persona from the spec and prevents drift.

```
You are <persona_id> from this spec:
  /home/john/elspeth/evals/composer-harness/personas/<persona_id>.md

Read it, internalise it, do NOT break character.

Your task and the rules of the game:
  /home/john/elspeth/evals/composer-harness/scenarios/hardmode/<scenario_filename>

Conversation so far (read in order):
  /home/john/elspeth/evals/composer-harness/runs/<date>-hardmode/<sid>/turn1.user.txt
  (extract assistant reply with: jq -r '.message.content' /home/john/elspeth/evals/composer-harness/runs/<date>-hardmode/<sid>/msg.t1.resp.json)
  ... (turn2.user.txt, msg.t2.resp.json, ... for all prior turns)

Reply with EXACTLY ONE of:
  (a) The next user message in character. Plain text. No JSON, no quotes,
      no preamble. The orchestrator will save your reply verbatim into
      turn<N>.user.txt and POST it to the composer.
  (b) The literal token "DONE: <one-line reason>" if your stop conditions
      have been met (see persona spec) OR you've hit the 5-turn message
      budget.

Do not narrate what you're going to say. Just say it.
```

Save the subagent's reply to `$RUN_DIR/turn<N>.user.txt`. If it starts with
`DONE:`, write the reason to `$RUN_DIR/done_reason.txt` and skip to the
finalize step. Otherwise call `post_message.sh <sid> <N> "$RUN_DIR/turn<N>.user.txt"`.

### Stop conditions per scenario

- Persona-subagent returned `DONE:`.
- 5 user turns have been sent. Force-stop, write `done_reason.txt =
  "convergence-budget exhausted"`.
- Composer returned a clear DONE-equivalent (e.g. "the workflow is ready,
  you can run it now") AND the persona's stop conditions look met. In
  ambiguous cases, dispatch ONE more subagent turn — it will say DONE if
  it agrees.

### Failure modes per scenario — these are DATA, not problems

- **Convergence-timeout** (composer's 180s ceiling fires): mark and move on.
  This is exactly what the eval is designed to surface.
- **Engine failure on /execute**: capture the run + diagnostics and move on.
  The composer's YAML being wrong is a finding, not a harness bug.
- **Validate-fail at finalize**: capture validate.json's failed_checks and
  move on. This is also a finding.

The harness is designed to keep going — `finalize_scenario.sh` writes the
ledger even on failure. Don't abort the suite on a single scenario failure.

### Recovery from harness errors (vs scenario findings)

- HTTP 401 mid-scenario: the JWT auto-refresh should handle this. If it
  doesn't, re-run `hardmode/harness.sh <sid> --reuse-sid` to force a new
  login on the same session.
- `harness.sh` says dir exists: use `--fresh` to reset (you're starting a
  new eval; nothing to preserve from a prior aborted run of the SAME date).
- `finalize_scenario.sh` exits 75 (run poll timeout): the run may still be
  going. Wait 60s and re-run finalize — the ledger gets overwritten.
- Subagent breaks character (verified via `validate_persona.sh`):
  re-dispatch that turn with a stronger reminder. Do NOT edit the prior
  `turn<N>.user.txt` — re-run the post_message and overwrite.

## Step 4: Aggregate and report

After all 15 scenarios:

```bash
hardmode/aggregate.sh "$ELSPETH_EVAL_RUNS_DIR"
cat "$ELSPETH_EVAL_RUNS_DIR/SCORECARD.md"
```

Then write a report at `/home/john/elspeth/docs/composer/evidence/composer-eval-hardmode-<UTC-date>.md`
following the structure of `docs/composer/evidence/composer-eval-hardmode-2026-05-03.md`. Include:

1. **Headline** — one-line summary of run health (e.g. "12/15 scenarios
   reached engine; 4/15 ran successfully; 3 convergence-timeouts").
2. **Scorecard table** — copy from `SCORECARD.md`.
3. **Per-persona narrative** — one paragraph per persona about what their
   scenarios revealed (P1 Linda, P2 Sarah, P3 Marcus, P4 Dev).
4. **Findings** — observations worth filing back into filigree. Use
   `mcp__filigree__observe` for things you don't want to write a full
   issue for; use `mcp__filigree__create_issue` only if the user explicitly
   tells you to. Default to observations.
5. **Comparison vs 2026-05-03 baseline** — read
   `evals/2026-05-03-composer/hardmode/aggregate.json` and call out what
   changed (which scenarios moved from `failed` to `completed`, which
   regressed, which are new in the matrix).
6. **Cost summary** — report `aggregate_summary.json` cost metadata and
   provider billing observations; do not estimate dollars from wall time.
7. **Path to evidence** — point at `evals/composer-harness/runs/<date>-hardmode/`.

## Things to NOT do

- **Do not write persona replies yourself.** This is the failure mode that
  invalidates the eval. Always dispatch a fresh subagent per turn.
- **Do not skip `validate_persona.sh`.** It's heuristic but data. Even
  ambiguous results are useful.
- **Do not edit scenarios or personas mid-run.** They're immutable test
  contracts. If a fixture is genuinely broken, stop and report.
- **Do not commit anything.** Capture all evidence to `runs/`, write the
  report, then stop. The user will commit.
- **Do not silently retry past a scenario failure.** Capture it as a
  finding and move on; don't try 3 times to make it pass.
- **Do not assert pass/fail on subjective probes.** The `pass_criterion`
  field is for human judgement after evidence is in. The harness produces
  evidence; your report describes it; the user decides pass/fail.

## Progress checkpoint

You should be tracking work with TaskCreate. Suggested task list:
- [Pre-flight] read references, run --doctor
- One task per scenario (15 of them)
- [Aggregate] run aggregate.sh
- [Report] write docs/composer/evidence/composer-eval-hardmode-<date>.md

Mark each scenario `completed` once its `ledger.json` is written, even if
the run inside failed. The ledger existing means the harness side is done;
the run outcome is data.

================ END PROMPT ================
```

## Notes for the operator dispatching the prompt

- Spawn this in a fresh Claude Code session, not inside an existing one with
  context. The agent benefits from clean state for the persona dispatch.
- The 15-scenario suite takes ~2-3 hours of supervised time. Plan accordingly;
  if you need to interrupt, the harness state on disk is fully recoverable
  (`--reuse-sid` resumes a partial scenario).
- The composer model under test is the running service's `composer_model`.
  P4-T1 and the P4 persona mention `gpt-5-mini` as generated pipeline model
  probes; those strings do not set the composer LLM (see README "Composer
  model and generated model fields").
