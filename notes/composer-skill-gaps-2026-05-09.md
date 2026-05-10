# Composer skill-coverage gaps — investigation brief

**Date:** 2026-05-09
**Context:** RC5.1 staging composer, post engine-fix commits `3938b6d3` (rejection_mutation leads `validation.errors`) and `ae4acc03` (skill soften discouraging `explain_validation_error` on rejection messages). Observations filed: `elspeth-obs-1159a166c2`, `elspeth-obs-4f31301a40`, `elspeth-obs-af7e73b3b2`.
**Audience:** Engineer or agent picking up composer-skill quality work.

## What we did

We picked five structural (non-LLM, non-cloud) examples at random from `examples/` and asked the staging composer (`https://elspeth.foundryside.dev`, model `openrouter/openai/gpt-5.4-mini`) to build each one from a plain-English user description. The user description deliberately avoided naming any plugin or composer tool; we only described domain intent ("split into two files based on amount", "trim long descriptions", etc.). Each trial uploaded the example's `input.csv` as a session blob first, so the composer had a real source to bind.

The five examples:

| Example | Verdict | Cost | Built vs intended |
|---|---|---|---|
| `schema_contracts_demo` | ✅ structurally faithful | $0.030 | gate-on-amount split, two sinks; chose JSONL where example used CSV |
| `threshold_gate` | ✅ ⚠️ extra nodes | $0.077 | gate routes through redundant `passthrough` node per branch before reaching sink |
| `boolean_routing` | ✅ ⚠️ extra nodes | $0.054 | same redundant-passthrough pattern as threshold_gate |
| `fork_coalesce` | ❌ structure dropped | $0.050 | built only the pre-fork transform + 1 sink; fork-gate, parallel paths, and coalesce all silently dropped |
| `deep_routing` | ❌ no convergence | $0.048 | state ended empty after 5 LLM rounds and 2 rejected `set_pipeline` calls |

Session URLs (live, persisted in `data/sessions.db`):

```
schema_contracts_demo  https://elspeth.foundryside.dev/#/e729f675-1cf3-491d-8413-b02cc0b0dd6b/graph
threshold_gate         https://elspeth.foundryside.dev/#/0d6fb0f7-03ac-427c-9535-0079323ef3a6/graph
boolean_routing        https://elspeth.foundryside.dev/#/7376c7bc-4f4c-4244-9e2e-547bc8a7fa5f/graph
fork_coalesce          https://elspeth.foundryside.dev/#/529a7ebf-e41f-4062-bb78-9f1bf57d231e/graph
deep_routing           https://elspeth.foundryside.dev/#/193c646a-881d-472c-976f-29a091240c4f/graph
```

Three problem clusters emerged. None of them are caused by the engine fix on RC5.1; they predate it. They are skill-text gaps, not engine bugs.

---

## Problem 1 — Redundant `passthrough` nodes after gates

**Observation ID:** `elspeth-obs-1159a166c2` (P2)
**Reproduces in:** sessions `0d6fb0f7` (threshold_gate), `7376c7bc` (boolean_routing). Also visible in `e729f675` (schema_contracts_demo) where a `type_coerce` node sits between source and gate but the gate still routes directly to sinks — so the issue is specifically *post-gate* shimming.

### Evidence

`examples/threshold_gate/settings.yaml` wires source → gate → sink directly:

```yaml
gates:
- name: amount_threshold
  input: gate_in
  condition: row['amount'] > 1000
  routes:
    'true': high_values   # sink name
    'false': output       # sink name
```

The composer built (session `0d6fb0f7`):

```
source → coerce_amount(transform) → split_high_low(gate)
                                      ├── high_copy(passthrough) → high_value_rows_out(csv sink)
                                      └── normal_copy(passthrough) → normal_rows_out(csv sink)
```

Boolean routing showed the same shape with `approved_identity` / `rejected_identity` passthroughs.

### Hypotheses

1. The skill teaches gate routing via node names exclusively; the user-facing `pipeline_composer.md` may not crisply state that gate `routes` values can be **sink names**, not just node names.
2. The LLM is treating gates as pure decision points and assuming sinks must be fed by a node-typed predecessor. This is a model-of-the-system bug, not an engine constraint — the engine accepts gate→sink routing.
3. There may be example or recipe text that always shows gate → transform → sink, and the LLM is pattern-matching against that.

### Suggested investigation

- `grep -n "routes:" src/elspeth/web/composer/skills/pipeline_composer.md` and check how the routing examples are written.
- Look at `evals/composer-rgr/scenarios/` for any scenario that exercises direct gate→sink routing; if none exist, that's a coverage gap.
- Consider adding a one-liner near the gate teaching: "Gate `routes` values can be sink names directly — no intermediate transform needed unless you actually need to mutate the row."
- Add an eval scenario `gate_to_sink_direct` that fails GREEN if the built pipeline contains a passthrough node whose only purpose is to forward data from a gate to a sink (count nodes in the cheapest fix path and reject if > 1).

### Why it matters

This pattern adds 2 nodes per branch on every gate-using pipeline — easily $0.02–$0.05 in extra LLM tokens per trial, plus extra runtime hops. It's the most prevalent of the three issues because every multi-output pipeline goes through a gate.

---

## Problem 2 — Fork/coalesce silently downgraded to linear pipeline

**Observation ID:** `elspeth-obs-4f31301a40` (P1)
**Reproduces in:** session `529a7ebf` (fork_coalesce).

### Evidence

User prompt (verbatim, plain English):

> I uploaded a CSV of products with columns id, product, price, category, description. Some descriptions are quite long — please trim each description to about 40 characters with an ellipsis at the end. Then for every product I want a single output entry that contains two side-by-side copies of the trimmed row data, nested under separate keys (call them path_a and path_b). Save the merged output as JSON Lines.

`examples/fork_coalesce/settings.yaml` is exactly this shape: source → truncate → fork-gate → two parallel paths → coalesce(merge: nested) → sink.

The composer built (session `529a7ebf`):

```
source → trim_description(truncate) → merged_rows(json/jsonl sink)
```

Assistant reply acknowledged the simplification: *"Internally it's a simple CSV input → truncate step → JSONL output."* The LLM knew it had simplified, but didn't surface this as a question or limitation.

### Hypotheses

1. `pipeline_composer.md` likely has no recipe or worked example for fork+coalesce. A search will confirm: `grep -ni "fork\|coalesce" src/elspeth/web/composer/skills/pipeline_composer.md`.
2. With no template to match, the LLM defaulted to the closest pattern it recognized (linear truncate → sink) and rationalized away the unhandled parts of the request.
3. There may be no `request_advisor_hint` budget here either (advisor is disabled on staging — see `composer_advisor_enabled` default in `src/elspeth/web/config.py:59`), so the LLM has no escape hatch when it doesn't know a pattern.

### Suggested investigation

- Audit `pipeline_composer.md` for any mention of fork, coalesce, branches, or merge strategies. If absent, add a "Parallel paths and merging" section with a worked YAML example pulled from `examples/fork_coalesce/settings.yaml`.
- More importantly: teach the composer to **refuse rather than silently downgrade**. Add to the skill: "If the user describes a structural pattern you cannot build (fork+merge, batch aggregation with custom triggers, etc.), reply that you cannot build it rather than building a simpler-shape pipeline that drops parts of the request."
- Add `evals/composer-rgr/scenarios/fork_coalesce_basic/` to lock in the regression — RED criteria should fail any pipeline that lacks both a `fork_to` field and a `coalesce` node when the prompt describes parallel-and-merge.

### Why it matters

Silent downgrade is the worst-shape failure for a tool whose output the user can't easily verify. The user sees "Done — I built a workflow" and trusts it. The actual pipeline is missing core requested behaviour. **Trust cost is much higher than dollar cost here** — we'd rather have the LLM say "I can't build that exact shape; what would you like to do?" than confidently produce a wrong pipeline.

---

## Problem 3 — Multi-gate cascade prompts fail convergence

**Observation ID:** `elspeth-obs-af7e73b3b2` (P2)
**Reproduces in:** session `193c646a` (deep_routing).

### Evidence

User prompt described a 6-rule loan triage cascade: one upstream content filter (4 keyword patterns), three field renames, a notes truncation, and 5 chained gates routing to 7 sinks. Full prompt is ~250 words; pipeline shape is in `examples/deep_routing/settings.yaml` (8-node-deep DAG).

The composer made 5 LLM rounds, 2 rejected `set_pipeline` attempts, never reached `preview_pipeline`, and the session state ended `null` (empty). Total cost $0.048 with no usable output.

The two `set_pipeline` rejections (in order):

1. *"Refusing header-only inline CSV for set_pipeline because ready uploaded CSV blob(s) with matching headers already exist in this session: input.csv (0cc62cc6, 1320 bytes)."* — server-side safety: LLM tried to fabricate inline data instead of using the uploaded blob.
2. *"Invalid options for source 'csv': schema: Field required. Use 'schema: {mode: observed}' to infer types from data, or provide explicit field definitions with mode (fixed/flexible)."* — second attempt omitted the schema block.

Both failures happened at the **source-binding step**, before any of the 5 gates were even attempted. This is interesting: cognitive load from the prompt's complexity appears to have pushed the LLM into rushing the source wiring.

### Hypotheses

1. The skill's instruction to call `list_blobs` first is not strong enough when the user describes a complex pipeline — the LLM jumped straight to `set_pipeline` without discovering the existing upload.
2. The header-only-inline-CSV safety check (in `src/elspeth/web/composer/tools.py`, search for `"Refusing header-only"`) is a good guardrail but the error message could nudge the LLM toward `set_source_from_blob` more clearly.
3. The post-rejection retry path doesn't include the schema field. This may mean the LLM omitted it under cognitive pressure, OR that the rejection-message ordering (now leading with `rejected_mutation` after `3938b6d3`) hides the schema-field hint somewhere the LLM doesn't read.
4. With 5 gates × 2 routes each = 10+ wiring decisions, plus 3 field renames, plus a quarantine route, the per-compose turn budget (`COMPOSER_MAX_COMPOSITION_TURNS=30` in `deploy/elspeth-web.env`) might be tight even when working correctly.

### Suggested investigation

- Replay the prompt and watch the response payloads at each round. Specifically: when the second `set_pipeline` was rejected, did the response surface a clear "schema: Field required. Use 'schema: {mode: observed}'..." rejection_mutation entry? If yes, why did the LLM not include it on the next call? If no, that's a skill or response-shape bug.
- Re-read `src/elspeth/web/composer/skills/pipeline_composer.md` for the upload-discovery rules. A potential rephrasing: "**MANDATORY STEP 1** when a user message mentions an uploaded file or attachment: call `list_blobs`. Do not call `set_pipeline` until you have the blob_id of the relevant upload."
- Consider adding a "build incrementally" recipe for cascades: discover blob → set_source → upsert_node × N gates → set_output × M sinks, instead of one giant `set_pipeline` call. This trades atomicity for incremental visibility into which gate validates and which doesn't.
- Run the `deep_routing` prompt in 3 separate sessions and check whether the failure shape is deterministic or stochastic. If 3/3 fail, it's a structural issue. If 1/3 fails, it's a budget/temperature issue.

### Why it matters

The example `deep_routing` covers a real-world pattern — multi-rule routing for triage workflows is common in compliance, risk, and content-moderation domains. If the composer can't handle this shape end-to-end, the demo's claimed coverage of "real ELSPETH workflows" has a visible hole.

---

## Reproduction recipe (cold)

```bash
# 1. Auth + create session
TOKEN=$(curl -sS -X POST https://elspeth.foundryside.dev/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"dta_user","password":"dta_pass"}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

SID=$(curl -sS -X POST https://elspeth.foundryside.dev/api/sessions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"title":"Replay <example> structural"}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

# 2. Upload input.csv as session blob
CSV_CONTENT=$(cat examples/<example>/input.csv)
curl -sS -X POST "https://elspeth.foundryside.dev/api/sessions/$SID/blobs/inline" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "$(python3 -c "import json,sys; print(json.dumps({'filename':'input.csv','content':sys.argv[1],'mime_type':'text/csv'}))" "$CSV_CONTENT")"

# 3. Send the plain-English prompt (see prompts above for each example)
curl -sS -X POST "https://elspeth.foundryside.dev/api/sessions/$SID/messages" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "$(python3 -c "import json,sys;print(json.dumps({'content':sys.argv[1]}))" "<prompt>")"

# 4. Inspect built state
curl -sS -H "Authorization: Bearer $TOKEN" \
  "https://elspeth.foundryside.dev/api/sessions/$SID/state" | python3 -m json.tool
```

To inspect tool calls / costs, query `data/sessions.db`:

```python
import sqlite3, json
con = sqlite3.connect('/home/john/elspeth/data/sessions.db')
cur = con.execute(
    "SELECT content, tool_calls FROM chat_messages "
    "WHERE session_id=? AND role='tool' ORDER BY created_at",
    (SID,))
# tool_calls JSON has invocation.tool_name / status / error_class / latency_ms /
# arguments_canonical / result_canonical. content with _kind=llm_call_audit
# carries per-round model / total_tokens / provider_cost.
```

## Files most likely relevant

| Path | Relevance |
| --- | --- |
| `src/elspeth/web/composer/skills/pipeline_composer.md` | Primary skill text — all three problems are likely fixable here |
| `src/elspeth/web/composer/recipes.py` | Recipe catalogue — fork/coalesce and multi-gate-cascade probably belong here |
| `src/elspeth/web/composer/tools.py:_execute_set_pipeline` | Header-only inline CSV safety + schema-required validation (problem 3) |
| `src/elspeth/web/composer/prompts.py` | System prompt assembly (advisor stripping) |
| `evals/composer-rgr/scenarios/` | Existing regression scenarios — gaps for gate→sink direct, fork/coalesce, multi-gate cascade |
| `deploy/elspeth-web.env` | `COMPOSER_MAX_COMPOSITION_TURNS=30` — budget (problem 3) |

## Suggested triage order

1. **P1 (trust):** problem 2 — silent downgrade. Either teach the fork/coalesce pattern, or teach the LLM to refuse rather than simplify. The latter is far more important.
2. **P2 (cost):** problem 1 — redundant passthroughs. High-volume issue (every gate-using pipeline), low-complexity fix (skill-text edit).
3. **P2 (coverage):** problem 3 — multi-gate cascade convergence. May need both a skill-text fix and a budget bump.

## Open questions

- Does the post-engine-fix `rejected_mutation` entry actually help when the LLM *next* call still omits the field the rejection asked for? Problem 3's second attempt suggests not always. Worth checking whether the LLM's response context contains the rejection text or only an excerpt.
- Are there other example shapes we haven't tested that hide similar gaps? Candidates: `batch_aggregation`, `landscape_journal`, `template_lookups`, `statistical_batch_plugins`. Worth running through the same plain-English protocol when time allows.
- The user prompt for `fork_coalesce` mentioned "path_a" and "path_b" — almost too LLM-friendly. Even with that hint, the composer dropped the shape. A user not familiar with fork terminology would do worse. The skill needs to recognize generic phrasings like "process the same row two ways and combine."
