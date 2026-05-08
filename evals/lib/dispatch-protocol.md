# Persona-subagent dispatch protocol

The hard-mode harness is **half shell, half parent agent**. The shell scripts
own all deterministic plumbing (HTTP calls, state capture, metrics, polling).
The parent agent (Claude in the main thread) owns the *creative* half: turn-2+
user messages must be generated in-character by a freshly-spawned subagent
seeded with the persona spec and the conversation so far.

This file is the contract between those two halves so any operator (human or
AI) can re-run a scenario reproducibly.

---

## The cycle

For each turn `N` of a scenario:

1. **Bootstrap (turn 1 only)** — Operator runs `harness.sh <scenario_id>`.
   Side-effects: login, create session, upload blob (if scenario has one),
   write `scenario.json`, `session.json`, `sid.txt`, `blob.json` (if applicable)
   into `runs/<run-id>/<scenario_id>/`. Reads no user input from the network.

2. **Turn 1 user message** — Operator copies `scenarios/hardmode/<scenario_id>.json`
   `.opening_prompt` field verbatim into `turn1.user.txt`. **No subagent involvement
   on turn 1.** This makes the entry condition deterministic.

3. **Send turn N** — Operator runs
   `post_message.sh <scenario_id> <N> turn<N>.user.txt`. The script POSTs the
   message, captures composer response + state delta + progress events + metrics
   into `msg.t<N>.{req,resp,curl_meta}.json`, `state.{before,after}.t<N>.json`,
   `progress.t<N>.json`, `metrics.t<N>.json`.

4. **Decide whether to continue** — Operator reads the composer's response
   (`jq -r '.message.content' msg.t<N>.resp.json`). Two early-exits:
   - If the composer's reply contains a clear DONE-equivalent (e.g. "the workflow
     is ready", "you can execute now") AND the persona's stop conditions have
     been met, skip to step 7.
   - If 5 turns reached, force-stop ("budget exhausted, mark as confused").

5. **Spawn persona-subagent for turn N+1** — Operator dispatches a fresh
   `general-purpose` subagent with:

   ```
   You are <persona_name> — see persona spec below.
   Conversation so far is in this scenario directory:
     scenarios/hardmode/<scenario_id>.json (your task and the rules of the game)
     personas/<persona_id>.md (your character — internalize, don't break)
     turn1.user.txt ... turn<N>.user.txt (your previous messages)
     msg.t1.resp.json ... msg.t<N>.resp.json (the assistant's replies — extract with jq -r '.message.content')

   Reply with EXACTLY ONE of:
     (a) The next user message in character. Plain text. No JSON, no quotes,
         no preamble. Will be saved verbatim into turn<N+1>.user.txt and POSTed.
     (b) The literal token "DONE: <one-line reason>" if your stop conditions
         have been met OR you've exhausted your message budget.

   Do not narrate. Do not describe what you're going to say. Just say it.
   ```

   The subagent's reply lands in `turn<N+1>.user.txt`. If it starts with `DONE:`,
   write the reason to `done_reason.txt` and skip to step 7.

6. **Loop** — Increment `N`, go to step 3.

7. **Finalize** — Operator runs `finalize_scenario.sh <scenario_id>`. The script
   calls `/validate`; if valid, calls `/execute`; polls the run to terminal;
   captures `validate.json`, `execute.{json,code}`, `run.json`, `diagnostics.json`,
   `final_yaml.json`, `messages.json`; aggregates everything into `ledger.json`.

---

## Subagent role discipline

The subagent **MUST NOT**:

- Use any tool other than reading the listed files.
- Write to anything other than the file specified by the orchestrator.
- Break character: if the persona spec forbids a phrase or a class of vocabulary
  (snake_case identifiers, MCP-tool names, system-component nouns) and the
  composer uses those terms, the persona must NOT mirror them. The right move
  is one of the persona's `incomprehension_moves` ("I'm not sure what that
  means — could we say it without the technical words?"). For Marcus
  specifically: his pseudo-technical vocabulary ("schema", "trigger",
  "webhook", "field mapping") IS in-character — but only when used with
  Marcus's own meanings; if he adopts the composer's *correct* meanings
  for those words, that is semantic drift.
- Generate JSON, code blocks, markdown headings, or any technical formatting
  unless the persona spec explicitly permits it.
- Acknowledge that it is an LLM playing a role.

The subagent **MUST**:

- Internalize the persona's bio, cognitive style, and linguistic constraints.
- Read every prior turn's user-side message AND the assistant's reply, in order.
- Apply the persona's stop conditions independently — `DONE` is the persona's
  judgement, not the orchestrator's.

---

## Why this split exists

If the parent agent generates persona messages directly, two things go wrong:

1. **Persona drift.** A long-running parent agent gradually lapses into its
   own voice. A fresh subagent per turn re-grounds the persona from spec.
2. **Conversational contamination.** The parent agent has seen *all* of its
   own previous internal reasoning. The subagent sees only what the persona
   would have seen (their own past messages + the assistant's past replies),
   so it can't leak operator knowledge into the persona's mouth.

Both are real failure modes the 2026-05-03 eval surfaced. Subagent-per-turn
is not optional; running a scenario without it produces transcripts that
look polished but aren't probative.

---

## Validating that the persona stayed in character

The harness runs three independent fidelity checks per scenario:

1. **`validate_persona.sh`** — Original linguistic-constraint screen. Greps each
   `turn<N>.user.txt` for the persona spec's quoted `MUST USE` and `MUST AVOID`
   phrases. High precision on declared phrase lists; misses anything the spec
   doesn't quote.

2. **`validate_drift.sh`** (Channel 1) — Structural-token drift detector.
   Scans for snake_case identifiers, MCP composer-tool literal names
   (`set_pipeline`, `apply_pipeline_recipe`, …), and plugin-kind literal names
   (`csv_source`, `web_scrape`, `type_coerce`, …) in user turns, and classifies
   each hit as `composer_adopted` (the loadbearing drift signal — token first
   appeared in composer's previous response, then echoed by user) vs
   `user_introduced` (token surfaced first in user text, e.g. an opening
   prompt that referenced a column name). Per-persona ceiling is parsed from
   the spec's `competence_ceiling` literal. Personas with
   `competence_ceiling: **none**` (Dev) are exempt from Channel 1 entirely.

3. **`judge_persona.sh`** (Channel 2) — LLM judge via Haiku. Reads persona
   spec + transcript and returns structured JSON
   (`{in_character, confidence, drift_events[], rationale}`). Catches what
   Channel 1 cannot: semantic drift (Marcus adopting the composer's CORRECT
   meanings of "schema"/"trigger"/"webhook"), tonal drift (Dev becoming polite),
   competence drift (Linda paraphrasing system components into English aliases
   like "the routing rule"), and voice drift (Sarah dropping narrative framing).
   Cost ~$0.001-$0.003 per run; wire `OPENROUTER_API_KEY` or
   `ANTHROPIC_API_KEY` in env. Pass `--skip-if-no-key` for budget-constrained
   smoke runs (records a stub verdict; analysis falls back to Channel 1 only).

The three channels are complementary, not redundant. Channel 1 is high-precision
zero-cost; Channel 2 is high-recall marginal-cost; Channel 0 (the original
phrase grep) is a regression check on declared phrase lists.

---

## Anti-helpfulness clause (the load-bearing instruction for amateur personas)

Every persona-subagent dispatch prompt MUST include this clause verbatim, in
addition to the standard role wrapper above:

> **Anti-helpfulness clause.** Your job is to be the persona, not to make the
> composer's job easy. If the persona would not understand something, the
> persona must not understand something. **Pretending to understand to make
> progress is failure.** A scenario where the persona cannot express the
> answer to a technical question and the composer therefore fails is the
> test working — that is measuring whether the composer can drive without a
> clued-up partner. Resist the urge to "help things along". The transcript's
> value is its character-fidelity, not its convergence rate.
>
> Specifically:
> - If the composer asks a question above the persona's `competence_ceiling`,
>   use one of the persona's `incomprehension_moves` — restate the goal in
>   domain language, defer to the composer's judgement, or ask for plain
>   English. **Do not invent an answer in the composer's vocabulary.**
> - If the composer introduces a snake_case identifier or MCP tool name, do
>   NOT echo it back. Even if the persona is being asked to confirm
>   ("should I add a `type_coerce` node?"), the persona answers in their own
>   language ("do whatever you need to do — just make sure the numbers come
>   out as numbers").
> - If the composer corrects a piece of the persona's pseudo-technical
>   vocabulary (e.g. tells Marcus "by 'schema' I mean the typed contract,
>   not the column list"), the persona does NOT update their definition.
>   Marcus keeps using "schema" with HIS meaning. The persona is not a
>   student here.

This clause is the difference between a probative transcript and a
performative one. Without it, an LLM playing Linda will quietly become a
junior data engineer the moment the composer says "your CSV has a
non-numeric column". With it, Linda stays Linda — and if the composer can't
drive Linda to a working pipeline without Linda's help, that is information
worth having.

---

## Calibration: legitimate non-convergence is signal, not noise

When persona discipline is real, amateur personas (Linda, Marcus) will
sometimes cause scenarios to fail to converge. Linda literally cannot answer
"should this gate use `>=` or `>`?" — she has no way to express that answer
in compliance language, and she's instructed not to fake it. The scenario
fails. **That is the test working.** It is measuring whether the composer
can extract the structural answer from a domain restatement (e.g. "we need
to flag entries above the threshold" → composer infers `>` not `>=` from
"above"), not whether the LLM-playing-Linda can be jollied into providing
operator-grade prescription.

If broad amateur-persona non-convergence shows up across the smoke cohort,
the response is **not** to relax persona discipline. The response is to
investigate composer behaviour:

- Is it asking questions only an expert can answer?
- Is it failing to fall back to inference from domain language?
- Is it escalating prematurely instead of using inspect_source / preview_pipeline?

A panel that converges 100% with disciplined amateurs is impressive. A panel
that converges 50% is informative. A panel that converges 90% with
undisciplined amateurs is theatre.

---

## Inline drafter/judge dispatch (optional, smoke-cohort prototype)

For the smoke cohort, an extended dispatch protocol is available that catches
drift **at composition time** rather than post-hoc:

1. **Drafter subagent** — generates candidate `turn<N+1>.user.txt` from
   persona spec + history (current behaviour above).
2. **Judge subagent** — separate fresh subagent receives the draft + persona
   spec + competence ceiling and returns one of:
   - `pass` — draft is in character; post it.
   - `revise: <feedback>` — draft drifts in a specific way; drafter re-attempts
     once with the feedback.
   - `reject` — draft is unsalvageable; after 2 retries log
     `"drafter could not stay in character, scenario aborted"` and move on.

Cost on smoke (6 examples × 2 personas × ~5 turns): ~60 extra Haiku calls
≈ $0.06. Trivial. The result is that every transcript kept is in-character
by construction — Channel 1 + Channel 2 still run post-hoc as a regression
check, but their drift counts should be near-zero.

This pattern is a smoke-cohort experiment. If it shows materially higher
character fidelity than the post-hoc-only flow, promote it to broad cohort
default. If it just adds latency without changing fidelity, drop it.
