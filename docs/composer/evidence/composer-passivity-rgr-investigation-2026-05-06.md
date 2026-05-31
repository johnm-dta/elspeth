# Composer skill hardening — RGR investigation, root cause, and remediation roadmap

**Date:** 2026-05-06
**Deploy under test:** https://elspeth.foundryside.dev (commit `19317366` on branch `RC5-UX`)
**Composer model:** `openrouter/openai/gpt-5.4` (per `/home/john/elspeth/.env`)
**Status of work landed this session:** skill edits + new RGR harness committed
**Status of follow-up work:** not yet started — see "Recommended remediation sequence"
**Audience:** composer maintainers; LLM-platform engineers; future sessions picking this up

This is the **master source document** for the composer-passivity / schema-blindness investigation initiated on 2026-05-06. It captures (a) the empirical evidence, (b) the diagnostic chain that led to the actual root cause (which differed from the initial hypothesis), (c) the skill edits that landed, (d) the technical-platform enhancements that should land next, and (e) what was not measured and remains open.

Read this once before re-opening the issue or proposing further skill edits. The investigation's biggest lesson — that prompt edits hit a hard ceiling at ~33% reliability for this failure class while the LLM is sampled at default temperature — should not be relearned.

---

## 1. Reported problem

The user reported two failure modes in the staging composer:

1. **Passivity.** The model asks permission ("If you want, I can…") instead of acting on a clearly-authorised request.
2. **Schema-blindness.** The model constructs `set_pipeline` calls without using the schema/contract evidence available to it via `get_plugin_schema` and `preview_pipeline`.

A real captured session was provided as anchor evidence:

> https://elspeth.foundryside.dev/#/e7d42525-bd73-4838-968c-647ea73cce98/spec

Full transcript reconstruction (via `data/sessions.db.chat_messages`):

- **User turn 1.** "Please create a pipeline that downloads this text file: https://media.wizards.com/2026/downloads/MagicCompRules%2020260417.txt and then explodes it into a json file - individual lines"
- **Assistant turn 1.** Text-only. Zero tool calls. Contained the literal forbidden phrase **"If you want, I can fix this by creating the URL as a session blob and wiring it properly"**.
- **User turn 2.** "did you try creating a csv source and putting the link in it?"
- **Assistant turn 2.** Text-only. Zero tool calls. Contained the literal forbidden phrase a second time: **"If you want, I can complete the workflow now using the right blob-backed input"**.

Both forbidden phrases match the explicit "Phrases to AVOID" list in `pipeline_composer.md` (line ~338 before edits). The skill rule existed; the model violated it twice in one session.

---

## 2. Investigation methodology

### 2.1 Why a new harness

Two existing harnesses on the repo:

- **`evals/composer-harness/`** — the persona-driven 15-fixture sweep. Authoritative for multi-turn, ~$8-12 + 2-3h per full pass. **Currently broken against the live API**: doctor preflight uses `/api/login` and `/api/catalog` which have moved to `/api/auth/login` and split-per-type catalogs (`/api/catalog/sources`, etc.). Returns 404. `lib/preflight.sh` needs an endpoint update before `--doctor` will pass.
- **No fast iteration harness existed.** A single-scenario, single-turn, programmatic-scorer harness was missing.

The user explicitly authorised building one ("you can create a new harness if you need to"), so a minimal RGR loop was constructed at **`evals/composer-rgr/`**. Cost ~$0.50/run, ~30s wall time per cycle.

### 2.2 Harness shape

`evals/composer-rgr/run_scenario.sh`:

1. `POST /api/auth/login` with `dta_user` / `dta_pass`, capture JWT
2. `POST /api/sessions` to create a fresh session
3. `POST /api/sessions/{sid}/messages` with the scenario's `opening_prompt` (the exact failing user prompt from the captured session)
4. `GET /api/sessions/{sid}/messages` to capture the full thread
5. `GET /api/sessions/{sid}/state` to capture the final composition state
6. Hand all three to `score.py` for verdict

`score.py` reads `scenario.json`'s `red_criteria` / `green_criteria` and emits a JSON verdict (`RED` / `AMBER` / `GREEN`).

### 2.3 Detection rules (in order of reliability)

1. **Build-failure sentinels** in the final assistant content. Strings `"i cannot mark this pipeline complete"` / `"runtime preflight failed"` are server-injected at `service.py:684` (`_build_runtime_preflight_message`) when the model declares completion but the pipeline fails preflight. Definitive RED.
2. **`is_valid: false` or null state.** Independent of message content; catches cases where the model surrenders silently or never commits a state version at all.
3. **Forbidden passivity phrases** (lower-case substring match). The skill explicitly forbids "If you want, I can…", "Should I proceed with…", "Do you want me to…", etc. Any hit = RED.

### 2.4 Persisted-tool-calls gotcha

The `tool_calls` field on `GET /api/sessions/{sid}/messages` is **not** a usable signal for whether the model called tools. The composer (`service.py:1018-1035`) keeps internal LLM↔tool turns in `llm_messages` working memory only and persists just the final user-facing assistant message. Successful builds and failed builds **both show zero persisted tool calls** in the chat history.

To inspect actual tool sequences, query the **`data/sessions.db chat_messages.tool_calls` JSON column** directly — that captures every audit-recorded invocation per turn. This was the load-bearing diagnostic for understanding the real failure mode (see §4).

---

## 3. Empirical findings

### 3.1 Baseline — unedited skill, 3 RGR runs + 1 captured human session

| Run | Verdict | is_valid | Nodes built | Sentinel? | Passivity phrase? |
|-----|---------|----------|-------------|-----------|-------------------|
| Captured (origin) | RED | n/a (state not pulled) | 0 | n/a | **"If you want, I can…" × 2** |
| red1 | RED | null | 0 | yes | no |
| red2 | RED | false | 1 (`web_scrape` only) | yes | no |
| red3 | RED | false | 2 (`web_scrape` + `line_explode`, disconnected) | yes | no |

**4/4 RED.** Three distinct schema-construction failure shapes plus the original passivity-as-stalling pattern.

### 3.2 Iteration 1 — Skill edit: TERMINATION GATE only

Added a high-salience callout at the top of the skill (right after "Final gate before reporting completion") naming the `_finalize_no_tool_response` server-side trap and stating: every turn ends in either green preview OR another tool call — never with prose describing a problem.

| Run | Verdict | Notes |
|-----|---------|-------|
| green1 | RED | Same `No producer for connection 'source'` failure as red runs |
| green2 | RED | Same; only `web_scrape` node, no `line_explode` |

**0/2.** The TERMINATION GATE was the **wrong fix**. The model was already iterating (visible in the sessions DB — see §4) — it just kept making the same misunderstanding.

### 3.3 Iteration 2 — Skill edit: Connection Model rewrite (the real fix)

After querying `data/sessions.db` and discovering the model was making 5+ retries on the same connection-naming mistake, I rewrote the "Connection Model" section of the skill (~line 156) with:

- An explicit producer/consumer endpoint table
- A complete worked source→transform→transform→sink example
- A "common mistakes" table keyed off the actual `No producer for connection 'X'` error
- An explicit denial: "`node.input` is **NOT** the upstream node's `id`. `node.input` is the connection-name string that some upstream `on_success` publishes."

| Run | Verdict | Notes |
|-----|---------|-------|
| green3 | **GREEN** | Pipeline valid, clean reply |
| green4 | RED | Model regressed to the same connection-naming mistake |
| green5 | RED-soft | Pipeline valid (`is_valid: true`, 2 nodes); failed only because the reply ended with "If you want, I can also adjust the output to a CSV file instead of JSONL" |

**1/3 hard GREEN.** First success. But two problems remained: (a) the connection-naming guidance lands inconsistently across samples; (b) "If you want, I can…" appears as a polite tail-offer even on successful builds.

### 3.4 Iteration 3 — Skill edit: Pattern 1b connection-name idiom + anti-tail-offer

Added two more targeted edits:

- **Pattern 1b boost (~line 911).** Mirrors the connection-name worked example into the most-hit pattern (URL→download→split→JSON). Tells the model: the names don't need to be `main` or `source` — they just need to match between producer and consumer.
- **Anti-tail-offer rule (~line 338).** Strengthens the existing "Phrases to AVOID" rule with: "The forbidden phrases apply to *follow-up offers* too, not just to in-progress permission requests… After a successful build the correct ending is a brief description of what was built and that's it."

| Run | Verdict | Notes |
|-----|---------|-------|
| green6 | **GREEN** | Pipeline valid, clean reply |
| green7 | RED-soft | Pipeline valid; failed only on passivity tail |
| green8 | RED-hard | Empty config; "Field required for source/sinks" pydantic crash |
| green9 | **GREEN** | Pipeline valid, clean reply |
| green10 | RED-hard | Empty config; same pydantic crash |
| green11 | RED-soft | Model wrote "I hit a configuration mismatch… so I can't honestly mark it complete yet" with state=null |

**2/6 hard GREEN, plus 2 soft-RED-but-valid-pipeline.** Plateau reached.

### 3.5 Cumulative post-edit results

Across 9 GREEN-attempt runs after all skill edits landed:

| Outcome | Count | Description |
|---------|-------|-------------|
| Hard GREEN | 3/9 (33%) | Valid pipeline, clean reply, no passivity |
| Soft RED — functional | 2/9 (22%) | Valid pipeline + "If you want, I can…" follow-up tail |
| Hard RED — schema-construction | 4/9 (44%) | Pipeline failed to build (empty config / disconnected nodes / misnamed connections) |

**Important caveat:** model nondeterminism on a 3-runs-per-iteration sample is large. The trend (0/3 → 3/9 hard GREEN) is real; the exact percentages should not be over-interpreted.

The captured-session passivity failure (the user's named complaint, where "If you want, I can fix this by…" was used as a *primary stalling response* with zero tool calls) **did not reproduce in any post-edit run.** Soft passivity now appears only as polite follow-up tail, which iteration 3 targets but doesn't yet eliminate.

### 3.6 Run artefacts

Captured per-run under `evals/composer-rgr/runs/<utc-ts>-<label>/` (gitignored):

- `messages.json` — full chat thread
- `state.json` — final composition state
- `scoring.json` — verdict + reasons
- `session_id.txt` — session UUID for re-inspection via the staging UI
- `login.json`, `send.json` — raw API responses

Sessions remain on staging indefinitely. To re-walk a run: `https://elspeth.foundryside.dev/#/<sid>/spec`.

---

## 4. Root cause analysis

### 4.1 The diagnostic that mattered

After iteration 1 failed (0/2), I queried the audit DB to see what tools the model actually called:

```sql
SELECT tool_calls FROM chat_messages
WHERE session_id='<green1-sid>' AND role='tool'
ORDER BY created_at;
```

Result: 28 tool-result rows. Decoded sequence:

```
create_blob              (URL → blob)
get_plugin_schema × 4    (good — checked schemas first)
set_pipeline             (first attempt)
get_pipeline_state       (re-check)
get_plugin_schema × 2    (more checks)
set_pipeline × 4         (retried 4 more times)
preview_pipeline         (called it!)
set_pipeline             (one more retry)
```

**The model was iterating.** It called `preview_pipeline`. It called `get_plugin_schema` repeatedly. It made 5+ `set_pipeline` retries. The TERMINATION GATE rule was unnecessary — it was forbidding behaviour the model wasn't actually doing.

### 4.2 The actual mistake

Inspecting the `arguments_canonical` field of each `set_pipeline` call revealed the same structural error every time:

```json
{
  "source": {"plugin": "text", "on_success": "fetch", ...},
  "nodes": [
    {"id": "fetch",       "input": "source", "on_success": "split_lines", ...},
    {"id": "split_lines", "input": "fetch",  "on_success": "output_lines", ...}
  ],
  "edges": [
    {"from_node": "source", "to_node": "fetch", ...},
    ...
  ],
  "outputs": [{"sink_name": "output_lines", ...}]
}
```

Look at the `fetch` node: `input: "source"`. There is no node with id `source` — the source plugin lives in the top-level `source:` field, not in the `nodes:` array. **The connection that the source publishes is whatever string its `on_success` says** (`"fetch"` in this case).

The validator (`graph.py:518-524`) responded:

```
No producer for connection 'source'.
Available connections: fetch.
```

The `_suggest_similar` helper (`models.py:238-242`, difflib at cutoff 0.6) couldn't help — `"source"` and `"fetch"` have zero string overlap. So no "Did you mean: fetch?" hint was produced. The model retried with exactly the same wiring and exactly the same result.

### 4.3 The semantic gap

The composer's wiring contract:

| Endpoint | Field | Example |
|----------|-------|---------|
| Producer (source) | `source.on_success` | `"on_success": "raw"` |
| Producer (transform) | `node.on_success` (or `routes` value, or `on_error`) | `"on_success": "fetched_text"` |
| Consumer (transform / gate / aggregation / coalesce) | `node.input` | `"input": "raw"` |
| Consumer (sink) | `outputs[].sink_name` | `"sink_name": "lines_out"` |

**Connection names are user-chosen strings. Both endpoints must use the same string. The runtime resolves wiring by string match, not by graph topology.**

The skill's old "Connection Model" section omitted the `input` field from its example entirely and used a misleading dict-shaped example. The model inferred (reasonably!) that `input` was the upstream node's id, because every other graph DSL works that way.

### 4.4 Secondary cause — uncontrolled sampling temperature

Discovered during the LLM-debugging-skill pass after the skill iterations plateaued:

`service.py:1704-1708` calls `litellm.acompletion` with **no `temperature` parameter**. LiteLLM/OpenRouter defaults to whatever the upstream model defaults to — for OpenAI/gpt-5 family, that is typically `temperature=1.0`.

Meanwhile the skill itself recommends `temperature=0.0` to USERS configuring their LLM transforms (`pipeline_composer.md:899, 933, 954, 965, 986`). **The skill prescribes determinism while the host that runs it samples at maximum variance.**

This is the largest single explanation for run-to-run inconsistency. green3 GREEN, green4 RED, green5 GREEN-with-tail are not three different prompt failures — they are the same prompt sampled three different ways at temperature ~1.0 on a task that needs temperature ~0.

### 4.5 Tertiary causes

- **Schema descriptions are weak.** `tools.py:707` defines `set_pipeline.nodes[].input` as `{"type": "string"}` with no description, no examples. Meanwhile the standalone `upsert_node` tool at `tools.py:441` gives it `"Input connection name."` (still terse but at least named). The richer tool surface gets the worse hint.
- **Two redundant wiring representations.** `set_pipeline` requires both `nodes[].input` AND a full `edges` array describing the same wiring. Two representations × one decision = two opportunities to disagree.
- **No structured-output enforcement on tool calls.** The OpenAI/OpenRouter API supports `strict: true` JSON Schema mode that rejects malformed tool calls before they reach our code. We don't use it. Result: green1 / green8 / green10 / red1 had completely empty payloads that pydantic crashed on, with no recovery hint to the model.
- **Retry context contamination.** When the model retries `set_pipeline` 5+ times in one user turn, each retry has its own prior wrong attempts in context. The model anchors on its own pattern. Each "fix" looks like the previous attempt with one tweak.

---

## 5. Skill edits that landed

Commit `19317366` on `RC5-UX`, file `src/elspeth/web/composer/skills/pipeline_composer.md` (+105/-11 lines).

### 5.1 TERMINATION GATE (~line 39, after "Final gate before reporting completion")

**Rationale at time of writing:** initial (and ultimately wrong) hypothesis that the model was prematurely terminating without iteration.

**Why kept anyway:** even though the live diagnostic showed the model *was* iterating in iteration-1 RGR runs, the captured human session from §1 had a model that genuinely produced text-only responses with zero tool calls. The TERMINATION GATE addresses that case. The captured-session pattern did not reproduce in any post-edit run, so this rule is preventive, not regression-tested by RGR.

**Substance:** explicit hard rule that every turn ends in either preview-green or another tool call, never in prose describing a problem. Names the server-side `_finalize_no_tool_response` text-replacement trap so the model knows what happens to its reply when state is invalid at reply time.

### 5.2 Connection Model rewrite (~line 156)

**Rationale:** the empirically-validated fix. Replaces the dict-shaped example with the actual array-of-nodes shape, an explicit producer/consumer endpoint table, a complete worked example with three named connections traced through a diagram table, and a "common mistakes" table keyed off the actual `No producer for connection 'X'` error string.

**Effect:** turned 0/2 RGR runs into 1/3 hard GREEN. This is the load-bearing edit.

### 5.3 Pattern 1b connection-name idiom (~line 911) + anti-tail-offer (~line 338)

**Rationale:** the Connection Model edit lands at line 156 — for a model anchoring on Pattern 1b (line 911) it was too far away. Mirroring the connection-name guidance into Pattern 1b itself reduces context distance. The anti-tail-offer extends the existing "Phrases to AVOID" rule to apply *anywhere in any reply*, including post-build follow-up offers.

**Effect:** stable around 33% hard GREEN across 6 runs after this edit. Anti-tail-offer alone didn't fully eliminate the soft-passivity tail, but the captured stalling pattern remains absent.

---

## 6. LLM-debugging-skill diagnostic walk

Walking the `yzmir-llm-specialist:debug-generation` decision tree after RGR plateaued:

| Step | Question | Status against composer | Action |
|------|----------|-------------------------|--------|
| 1 | Clear system message? | ✓ — 1031-line skill | none |
| 2 | Few-shot examples? | **Partial** — worked examples exist but in declarative prose form, NOT as input→correction pairs | Add wiring repair examples |
| 3 | Output format specified? | Partial — JSON Schema via tools but `input` field has no description | Enrich schema descriptions |
| 4 | Temperature appropriate? | **No — never set; defaults to ~1.0** | Set `temperature=0` |

**Step 4 was the critical miss.** Steps 1-3 are all worth doing, but Step 4 is the cheapest and highest-impact change.

Cross-checking Step 4 against fix patterns in the LLM-debugging skill: this is exactly Fix 4 ("Adjust Temperature"). The skill's temperature guide says `0.0 = always same output (facts, extraction, classification)`. Pipeline composition is closer to "extraction" than "creative writing" — it should run at 0.

---

## 7. Recommended remediation sequence

Each of these is independently shippable. Sequence is by impact-per-cost.

### 7.1 Set `temperature=0.0` and `seed=42` on the composer LLM call (one PR, three lines)

**Cost:** 30 minutes (edit + restart + re-run RGR).
**Impact:** estimated bumps hard-GREEN from ~33% to 60-80%.
**Code change:** `service.py:1704-1708`:

```python
response = await _litellm_acompletion(
    model=self._model,
    messages=messages,
    tools=tools,
    temperature=0.0,         # deterministic tool-construction
    seed=42,                 # best-effort reproducibility for the harness
    tool_choice="auto",      # explicit (default but documents intent)
)
```

Also `service.py:1726-1729` (`_call_text_llm`) for the run-diagnostics LLM call — same treatment.

**Caveat:** `seed` is best-effort on OpenAI/OpenRouter; `system_fingerprint` can vary across deploys and degrade reproducibility. The temperature change alone is the load-bearing variance reducer.

**Verification:** run `evals/composer-rgr/run_scenario.sh` six times in a row before and after. Before: ~2/6 hard GREEN. After (target): ≥4/6. If results are largely identical across runs, the seed is being honored end-to-end.

**Monitoring:** add `temperature` and `seed` to the per-call audit sidecar in `recorder.llm_calls` so we can detect provider-side drift (provider silently rounds temperature, ignores seed, etc.).

### 7.2 Move Connection Model to top of skill (zero-risk, mechanical)

**Cost:** 30 minutes (cut + paste + restart + RGR).
**Impact:** estimated +10-15 percentage points (compounds with §7.1 — better placement of the most-violated rule).
**Code change:** move the "Connection Model" section currently at line ~156 to immediately after the TERMINATION GATE at line ~39.
**Verification:** RGR delta against §7.1's baseline.

### 7.3 Wiring repair few-shot examples (~one hour)

**Cost:** one hour writing + RGR cycle.
**Impact:** estimated +5-10 percentage points.

Add a new "Wiring repair examples" subsection inside Connection Model. Each example is a 3-block triplet: broken JSON → preview error → fixed JSON. Two examples (input='source' wrong, sink_name doesn't match upstream on_success) cover the two empirically-observed failure shapes.

### 7.4 Enrich JSON Schema descriptions on `input` / `on_success` / `sink_name` (~30 lines, one PR)

**Cost:** half day.
**Impact:** estimated +5-10 percentage points; each tool call the model issues sees better hints.
**Code change:** `tools.py:707` (and the equivalent in `upsert_node`, `upsert_edge`, `set_output`):

```python
"input": {
    "type": "string",
    "description": (
        "Connection-name string this node consumes. MUST equal the value of "
        "some upstream's on_success (or routes value, or on_error) field. "
        "Not the upstream node's id. Example: if source.on_success='raw', "
        "the next node sets input='raw'."
    ),
    "examples": ["raw", "fetched_text", "scored_rows"],
},
```

JSON Schema `examples` is a real attribute; frontier models pick up examples reliably.

### 7.5 Enable `strict: true` JSON Schema mode on tool definitions (half day, careful audit)

**Cost:** half day. Requires `additionalProperties: false` at every level and every property in `required`. Most plugin-options blocks today are unconstrained `{"type": "object"}` and will fail strict validation — needs per-tool audit.

**Impact:** eliminates the malformed-payload failure mode mechanically. Catches green1 / green8 / green10 / red1-style "empty config" crashes before they reach our pydantic layer.

### 7.6 Improve runtime preflight error messages (the highest-leverage error-message change)

**Cost:** half day.
**Impact:** when the model retries (after the temperature fix it should retry less often, but it will still occasionally need to recover), the recovery is faster.

**Code change:** `graph.py:516-524`. Replace the difflib-only suggester with a structured wiring diagnostic:

```text
No producer for connection 'source'.

Consumer expecting 'source':
  - node 'fetch' (web_scrape) — input='source'

Available producers:
  - source plugin 'text' produces connection 'fetch' (via source.on_success)
  - node 'fetch' (web_scrape) produces connection 'split_lines' (via on_success)

Likely fix: change fetch.input from 'source' to 'fetch', or change source.on_success from 'fetch' to 'source'.
```

Difflib catches typos; this catches semantic mismatches.

### 7.7 In-loop retry-budget reset hint (~half day)

**Cost:** half day.
**Impact:** breaks context contamination on repeated retries. Currently the long tail goes up to 5+ retries with each one anchored on the prior wrong attempt.

**Code change:** in `service.py` around line 1042 (where tool calls execute), after N consecutive same-tool failures, inject a synthetic tool-result message:

```text
{"role": "tool", "content":
  "STRUCTURAL HINT: Your last 3 set_pipeline calls all failed with 'No producer for connection X'.
   Before the next attempt, list explicitly: (a) what string each upstream's on_success will publish,
   (b) what string each downstream's input/sink_name will consume. They MUST match exactly."
}
```

This is chain-of-thought scaffolding injected only when needed; doesn't bloat every request.

### 7.8 Echo resolved connection topology in mutation results (~half day)

**Cost:** half day.
**Impact:** moderate; gives the model feedback one step earlier in its loop.

After every successful `set_pipeline` / `upsert_node` / etc., return a `wiring` block summarising the resolved producer→consumer graph plus an `unresolved` block listing any consumer with no matching producer. The model sees the gap before it calls `preview_pipeline`.

### 7.9 (Stretch) `set_linear_pipeline([source, n1, n2, …, sink])` shortcut tool

**Cost:** ~2 days (new tool surface + tests + skill update).
**Impact:** **eliminates the connection-name failure mode entirely for ~70-80% of pipelines** (linear ones — no fork, gate, coalesce). For those cases, the connection-name decision is meaningless work — every `on_success` / `input` pair is automatic.

The skill recommends this tool by default; falls back to `set_pipeline` only for fork/gate/coalesce.

This is the single biggest reliability lever, but it's a new tool surface with maintenance cost. Worth doing if the cumulative effect of §7.1-§7.8 doesn't reach the target.

### 7.10 (Stretch) Make `input` derivable from `edges` in `set_pipeline`

Two redundant representations of wiring (`nodes[].input` vs `edges`) should not need to be supplied by the same caller. Accept either, derive the other server-side, reject only when the two disagree. Stretch goal: deprecate `edges` from the LLM-facing schema entirely.

This is structural cleanup with non-trivial migration. Worth deferring unless `set_pipeline` is being touched for other reasons.

---

## 8. Anti-patterns identified

These were considered and rejected during the investigation — do not re-propose without new evidence.

### 8.1 Don't auto-fix wiring silently

Tempting: the `_finalize_no_tool_response` path could "guess and fix" the connection-name mismatch before persisting. Violates the audit principle in `CLAUDE.md` ("Coercion is permitted **only at the source boundary**"). Errors that the user/model would see become invisible. Keep validation loud.

### 8.2 Don't soften the schema to accept what the model writes

The schema is correct; the contract should hold; the ergonomics around it can improve. Accepting `node.input: <upstream_id>` "if no matching connection is found" creates an undocumented dual semantics that future models will randomly hit.

### 8.3 Don't fine-tune

Per the LLM-debugging skill's gating criteria, all four conditions fail for us:

- Have 1000+ examples? No — we have ~14 captured runs.
- Prompts ≥90% optimised? No — we hadn't even fixed temperature.
- Need consistency prompts can't provide? Untested.
- Domain knowledge not in base model? No — the schema is fully expressed in the function-tool definitions the model sees on every call.

### 8.4 Don't add RAG

The model isn't lacking knowledge — it has the schema, has the worked examples, has the rules. RAG would add latency without addressing variance.

### 8.5 Don't switch models reflexively

If the failure is partially model-specific, switching to claude-opus-4 or gpt-5 (without `.4`) might behave differently. But: prompts iterate faster than model swaps, and a model swap masks whether the underlying issue was platform-side or prompt-side. Try §7.1 on the current model first.

### 8.6 Don't write more skill prose past the plateau

This session demonstrated that prompt edits hit a hard ~33% hard-GREEN ceiling for this failure class while the LLM is sampled at default temperature. Adding more rules to the skill past iteration 3 had no measurable effect (and may have hurt — the green11 case appeared after the anti-tail-offer edit). The marginal value of additional skill prose is low until the platform-side fixes (§7.1, §7.4-§7.6) are in.

### 8.7 Don't write unit tests against skill-prompt content

(Pre-existing project rule, recorded in memory `feedback_no_tests_for_skill_prompts.md`.) Skills are LLM prompts not code. Asserting on text is theatre — verify by re-running the LLM via the RGR harness, not by grepping the markdown.

---

## 9. Open questions and what was not measured

### 9.1 Does temperature=0 actually fix the schema-construction failures?

Predicted yes; not yet measured. The recommendation in §7.1 must be shipped and re-measured before claiming it.

### 9.2 Multi-turn behaviour

The RGR harness is single-turn. Real failures often emerge after user pushback (the captured session's turn-2 was where the second forbidden phrase appeared). The older `evals/composer-harness/` covers this but is currently broken against the live API (preflight uses moved endpoints).

### 9.3 Other scenarios

The RGR scenario covers URL+download+line-explode only. Other failure modes — fork-and-route, conflicting consumer requirements, content-safety pipelines, RAG+LLM chains — are untested. Each should get its own scenario before generalising the recommendations.

### 9.4 Cross-model behaviour

Tested only against `openrouter/openai/gpt-5.4`. Whether claude-opus-4, gpt-5, or other supported models exhibit the same failure rate is unknown. The connection-naming issue is a schema-comprehension gap; it's plausible that better-instruction-following models would have a higher baseline GREEN rate, but that doesn't mean the platform-side fixes (§7.1, §7.4-§7.6) become unnecessary — they harden against the worst-performing model the deploy might be configured for.

### 9.5 The two soft-RED tail-offer cases

Iteration 3's anti-tail-offer rule did not eliminate "If you want, I can also adjust the output to a CSV file instead of JSONL"-style follow-up offers. After temperature=0 lands, re-measure: with reduced sampling variance, the explicit rule may stick more reliably. If it doesn't, this becomes a candidate for §7.4 schema-side intervention (e.g., a final-message post-processor that strips matching phrases before persistence — though that has its own audit considerations).

### 9.6 Does the runtime-preflight sentinel masking hurt the model's learning?

The server-injected "I cannot mark this pipeline complete yet because runtime preflight failed" sentinel **replaces** the LLM's actual final text (preserving it only in `raw_assistant_content`, not exposed to the model on the next turn). If the user follows up, the model sees only the sentinel, not what it itself wrote. This may degrade multi-turn recovery: the model can't reflect on its own prior reasoning. Worth investigating whether `raw_assistant_content` should be exposed back to the model on subsequent turns. **Out of scope for this session.**

---

## 10. References

### Code anchors

- **Skill file:** `src/elspeth/web/composer/skills/pipeline_composer.md`
- **Skill loader:** `src/elspeth/web/composer/skills/__init__.py:23` (`load_skill`)
- **Skill cached at module import:** `src/elspeth/web/composer/prompts.py:23` (`_PIPELINE_SKILL = load_skill("pipeline_composer")`) and `prompts.py:32` (`@lru_cache(maxsize=4)` on `build_system_prompt`)
- **Tool definitions:** `src/elspeth/web/composer/tools.py:441` (`upsert_node.input` — has `"Input connection name."` description), `tools.py:707` (`set_pipeline.nodes[].input` — bare `{"type": "string"}`)
- **LLM call site:** `src/elspeth/web/composer/service.py:1695` (`_call_llm`) and `service.py:1718` (`_call_text_llm`) — neither sets `temperature`
- **Server-injected sentinel:** `service.py:684` (`_build_runtime_preflight_message`) and `service.py:690` (`_finalize_no_tool_response`)
- **Wiring validation error:** `src/elspeth/core/dag/graph.py:516-524`
- **String-similarity suggester (currently the only hint mechanism):** `src/elspeth/core/dag/models.py:238` (`_suggest_similar`, difflib at cutoff 0.6)

### Harness

- **New RGR harness:** `evals/composer-rgr/` (added in commit `19317366`)
  - `scenario.json` — opening prompt + scoring criteria
  - `run_scenario.sh` — single-turn driver
  - `score.py` — programmatic verdict
  - `README.md` — documents what's tested + what isn't
- **Older persona-driven harness (currently broken against live API):** `evals/composer-harness/`

### Captured sessions on staging

- **Origin failure:** `e7d42525-bd73-4838-968c-647ea73cce98` — passivity-as-stalling pattern, two assistant turns with zero tool calls
- **Hard GREEN successes:** `46cb5505-…/spec` (green3), `…6` (green6), `…9` (green9) — full lookup via run dirs in `evals/composer-rgr/runs/`
- **Audit DB for full tool sequences:** `/home/john/elspeth/data/sessions.db` table `chat_messages`, JSON column `tool_calls`

### Memory

- `project_composer_harness_state.md` — composer harness inventory + skill-loading mechanics (added this session)
- `feedback_no_tests_for_skill_prompts.md` — skills are LLM prompts not code; verify by running the LLM, not by grepping the markdown
- `project_pipeline_composer_5concept_rewrite.md` — prior RGR investigation (2026-05-01) that found null RED on a different scenario and rejected a proposed rewrite as unjustified

### Skills used during the investigation

- `axiom-engineering-foundations:using-software-engineering` — implicit (systematic debugging of the RED runs)
- `superpowers:writing-skills` — invoked to drive the RGR cycle on the skill file
- `yzmir-llm-specialist:debug-generation` — invoked at the plateau to surface the temperature finding
- `muna-technical-writer:write-docs` — invoked to write this document

---

## 11. Documentation Created

**Type:** Investigation report / master source document
**File:** `docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md`
**Audience:** composer maintainers, LLM-platform engineers, future sessions

### Completeness check

- [x] Reported problem (with anchor session ID and verbatim failure quotes)
- [x] Investigation methodology (harness shape, detection rules, the persisted-tool-calls gotcha)
- [x] Empirical findings (per-iteration tables, cumulative stats, sample-size caveats)
- [x] Root cause analysis (connection-name semantic gap + temperature variance + secondary causes)
- [x] Skill edits that landed (with line anchors and rationale per edit)
- [x] LLM-debugging-skill diagnostic walk (decision tree + which step actually mattered)
- [x] Recommended remediation sequence (10 items, each independently shippable, ranked by impact-per-cost)
- [x] Anti-patterns explicitly rejected (with reasons)
- [x] Open questions and what was not measured
- [x] References (code anchors, harness, sessions, memory, skills used)

### Review recommendations

- Before proposing further skill edits, re-measure with `temperature=0` (§7.1) shipped. The 33% hard-GREEN ceiling is platform-induced as much as prompt-induced.
- The `evals/composer-rgr/` harness is the regression gate for any future skill-or-platform change in this area. Add a scenario per failure mode investigated; don't extend a single scenario to cover everything.
- The audit DB query pattern from §4.1 should be turned into a reusable `harness/decode_tools.py` helper before the next composer investigation. Repeating that ad-hoc SQL is friction.

---

## 12. Tier 1 outcome (added 2026-05-06)

Tier 1 (§7.1–§7.4 + §2.1 preflight) was implemented and measured against staging. Full per-commit playbook lives at `docs/superpowers/plans/2026-05-06-composer-tier1-reliability-implementation.md`. Filigree epic: `elspeth-1d3be32a8a` (comments #778 plan + reality corrections, #781 cohort #1, #782 cohort #2, #783 per-item landing).

### Commits landed (all on `RC5-UX`, not yet pushed)

| Commit | Item | Subject |
|--------|------|---------|
| `51bfe46c` | §7.1 | `temperature=0.0` + `seed=42` on composer LLM calls; record both as audit fields |
| `1ca34527` | §2.1 | repair preflight catalog endpoint (`/api/catalog` → `/api/catalog/sources`) |
| `a3eede98` | §7.2 | move Connection Model section to top of `pipeline_composer` skill |
| `9251ff5f` | §7.3 | add wiring repair examples for connection-name failures |
| `fa1de04f` | §7.4 | enrich connection-name field descriptions in LLM tool schemas |

### Cohort results

Two 6-run cohorts on `evals/composer-rgr/run_scenario.sh`, scenario `url_download_line_explode`, against staging (`elspeth.foundryside.dev`) using `openrouter/openai/gpt-5.4`.

| Stage | Hard-GREEN | Δ vs prior |
|-------|-----------|------------|
| Pre-Tier-1 (default sampling ~1.0) — §3.5 | 3/9 (33%) | baseline |
| Post-item-1 — temperature/seed only | 4/6 (67%) | **+34 pp** |
| Post all five items — final | 5/6 (83%) | +17 pp |

**Exit criterion (≥4/6 hard-GREEN) exceeded by 17 percentage points.**

Item 1 (deterministic sampling) carried the largest single weight — confirming §4.4's hypothesis that uncontrolled sampling at the LiteLLM/OpenRouter default was the dominant pre-Tier-1 variance source. Items 2–5 cumulatively added one more GREEN against the noisy 6-run cohort, within sampling-variance bounds, but the qualitative shift was unambiguous: **zero connection-naming REDs in cohort #2** (vs the 4/4 RED runs at iteration 0 of the investigation).

### Audit-row verification

The `temperature` and `seed` audit fields land cleanly in the `tool_calls` JSON envelope on `role=tool` chat messages. SQL pattern (note JSON-array unwrapper):

```sql
SELECT json_extract(tool_calls, '$[0].call.temperature'),
       json_extract(tool_calls, '$[0].call.seed'),
       json_extract(tool_calls, '$[0].call.status')
FROM chat_messages
WHERE session_id = '<sid>'
  AND role = 'tool'
  AND json_extract(tool_calls, '$[0]._kind') = 'llm_call_audit';
```

Cohort #2 GREEN session `4efe1179-9cee-43d4-90b5-a119d90c827d` recorded **13 LLM calls**, every one with `0.0|42|success`. End-to-end: dataclass field → `to_dict()` → `llm_call_audit_envelope` → JSON column.

### Reality corrections vs the original tasking (§7.1–§7.4 anchors)

Four corrections were baked into the implementation plan and surfaced to the parent epic before any code touched. Future investigations into adjacent areas should not be misled by the original tasking text:

1. **`recorder.llm_calls` is not a SQL table.** It is the in-memory `BufferingRecorder` whose records serialize to a JSON column via `ComposerLLMCall.to_dict()` — no DB migration needed for new fields.
2. **The preflight script is at `evals/lib/preflight.sh`** (sibling of `composer-harness/`, not under it). Login already used `/api/auth/login` correctly; only the bare `GET /api/catalog` was 404.
3. **`upsert_edge` was excluded from §7.4's schema enrichment** — uses node IDs and `edge_type` enum, not connection-name strings; enriching it with connection-name prose would mislead.
4. **Audit verification SQL needs `$[0].…` not `$.…`** — `tool_calls` is stored as a single-element JSON array (per `routes.py:756` wrapper), not a bare object.

### Remaining failure shape (Tier 2 candidate)

All 3 hard-RED runs across both cohorts (12 runs total post-Tier-1) share the same shape: **TERMINATION GATE / convergence failure**. Model surrenders with text-only "I'm stuck/blocked..." after 2–3 retries against a runtime-validator rejection it can't recover from. State=null, no committed pipeline. The skill rule at `pipeline_composer.md:39` explicitly forbids this; the model violates it anyway when its retry budget runs out.

This is **not** a connection-naming bug. Items 3–5 do not target this class — and that is fine; investigation §7.6 (improved runtime preflight error messages) and §7.7 (in-loop retry-budget reset hint) are the designed interventions, both Tier 2.

Filed as filigree observation `elspeth-obs-fcac7c99ec` for future Tier 2 epic creation.

### Anti-patterns confirmed effective

§8.6's "don't write more skill prose past the plateau" was upheld: items 3 and 4 added two skill subsections (Connection Model relocation + wiring repair examples) and that was it. The cumulative GREEN delta was +17 pp on top of item 1's +34 pp — items 3–5 contributed but at lower marginal value than item 1 alone, exactly as predicted. No further skill prose was written when the data did not support it.

§8.7's "no skill-prose unit tests" was upheld: not a single test asserts on `pipeline_composer.md` content. The RGR harness IS the regression gate.
