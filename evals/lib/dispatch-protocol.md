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
- Break character: if the persona spec says `MUST AVOID: "schema", "transform"`
  and the assistant uses those words, the persona should say "I'm not sure
  what those mean" rather than mirroring the vocabulary.
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

After a scenario finishes, run `validate_persona.sh <scenario_id>` to check
each `turn<N>.user.txt` against the persona spec's `MUST USE` and `MUST AVOID`
lists. The script reports per-turn linguistic-constraint matches; substantial
violations indicate the subagent broke character and the scenario should be
re-run or its results flagged.
