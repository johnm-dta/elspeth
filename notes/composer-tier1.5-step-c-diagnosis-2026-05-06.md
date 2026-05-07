# Composer Tier 1.5 — Step C diagnosis (2026-05-06)

**RED session id:** `53bc3cf2-ab90-4940-9679-1b5e7d474650`
**Run dir:** `evals/composer-rgr/runs/20260506T123028Z-final-1/`
**Verdict source:** `scoring.json` → `verdict=RED`, `red_reasons=["final composition state is null (no committed pipeline)"]`
**Helper used:** `evals/lib/decode_tools.py` (this commit) — decoded the chronological tool-call sequence from the read-only audit DB.

---

## Decoded tool sequence (filtered to mutation/discovery calls)

```
[ 0] user        — opening prompt (URL-download-line-explode)
[ 1] assistant   — final response: "I'm stuck on a configuration mismatch in the pipeline editor: the `web_scrape` s…"
[ 2] create_blob              — store URL as inline_blob
[ 3] get_plugin_schema        text (source)
[ 4] get_plugin_schema        web_scrape (transform)
[ 5] get_plugin_schema        line_explode (transform)
[ 6] get_plugin_schema        json (sink)
[ 7] set_pipeline #1          — RESULT: "Invalid options for source 'text': schema: Value error,
                                Field spec at index 0 is a dict with 4 keys. Expected single-key
                                dict like {'field_name': 'type'} or a string like 'field_name: type'."
[ 8] explain_validation_error — model queries the validator (useful mid-flight diagnostic call)
[ 9] set_pipeline #2          — source.schema.fields fixed to ["url: str"];
                                RESULT: "Node 'fetch_text': Invalid options for transform 'web_scrape':
                                schema: Field required / url_field: Field required /
                                content_field: Field required / fingerprint_field: Field required /
                                http: Field required"
[10] get_plugin_schema        web_scrape (transform) — re-fetched in a hope it changed
[11] set_pipeline #3          — IDENTICAL ARGUMENTS to #2; IDENTICAL ERROR to #2
[12-18] llm_call_audit envelopes (system-side LLM call telemetry; not tool calls)
```

The model gave up after attempt #3 with a text-only "I'm stuck" message — no further tool calls.

---

## Drift vs anchor analysis

| Transition | Source.schema.fields | web_scrape options | Error class | Drift? |
|------------|---------------------|---------------------|-------------|--------|
| #1 → #2    | `[{"field_type":"str","name":"url","nullable":false,"required":true}]` → `["url: str"]` | `{}` → `{}` | source-shape error → transform-Field-required error | **Drift (model improved)** |
| #2 → #3    | `["url: str"]` → `["url: str"]` (same) | `{}` → `{}` (same) | identical Field-required cascade | **Anchor (byte-identical retry)** |

So: **the model successfully drifted between attempts 1→2** (it read the explain_validation_error response and corrected the field-spec syntax). Then it **anchored on attempts 2→3** with byte-identical `set_pipeline` arguments despite re-fetching the web_scrape schema in between. The model surrendered after the third identical-args/identical-error cycle.

---

## Mapping to investigation §7.6 / §7.7

The investigation's diagnosis tree (per the Tier 1.5 tasking §8 step 3):

| Question | Answer here |
|----------|-------------|
| Did the model iterate (≥2 mutation attempts)? | YES — three `set_pipeline` calls. |
| Were the failed attempts identical or did they drift? | **HYBRID** — drift between #1→#2, anchor between #2→#3. |
| Did the runtime validator's error message contain enough information for a competent reader to repair the call? | For #1 (source field-spec): **YES** — message was specific, model did repair it. For #2/#3 (`web_scrape options: schema/url_field/content_field/fingerprint_field/http: Field required`): **PARTIAL** — message names the missing fields but doesn't show their expected shapes. The model had `get_plugin_schema(web_scrape)` cached in context (from row [4] at step #4 of the loop) yet sent `options: {}` anyway; re-fetching the schema at row [10] didn't rescue it. |
| Did the model surrender to text-only? | YES — final assistant turn has zero tool calls and contains "I'm stuck". |

**Root-cause classification: §7.7 dominant, §7.6 contributing.**

- §7.7 (in-loop retry-budget reset hint) is the **direct fit** for the #2→#3 anchor: a synthetic STRUCTURAL HINT injected after N consecutive same-tool failures with byte-identical arguments would have broken the anchor by forcing the model to re-evaluate which fields the validator actually named. The §7.7 hint generalises across error classes (connection-naming AND options-validation), which matters here because the failure mode in this RED is options-validation, not connection-naming as the original investigation focussed on.
- §7.6 (improve runtime preflight error messages) is the **secondary contributor** for #2 specifically — the `Field required` cascade does not show what shape each field expects (no "url_field must be a non-empty string", no example). The investigation's §7.6 worked example targets `graph.py:516-524` connection-naming; that specific code site would NOT have improved this RED, because the error here came from Pydantic validation on `web_scrape` plugin options, not from graph wiring. So §7.6 *as-scoped-by-the-investigation* misses this RED; a **broader §7.6** (extend the structured-diagnostic principle to plugin-options validation errors) would help, but that's a larger Tier 2 ticket than the §7.7 fix.

**Tier 2 selection rationale:** ship §7.7 first because it's surgical (one site in `service.py`'s tool-call loop), generalises across error classes, and directly targets the observed anchoring. §7.6 (in its broader form) becomes a follow-on observation.

---

## Anti-anchor hint design

The investigation's §7.7 example hint text is connection-name-specific:

> "STRUCTURAL HINT: Your last 3 set_pipeline calls all failed with 'No producer for connection X'. Before the next attempt, list explicitly: (a) what string each upstream's on_success will publish, (b) what string each downstream's input/sink_name will consume. They MUST match exactly."

For this RED, that hint would have been **useless** — the failure was plugin-options shape, not connection naming. The Tier 2 implementation must use a **generic** hint shape that adapts to the actual error class, e.g.:

> "STRUCTURAL HINT: your last N `set_pipeline` calls all failed with the same error. The previous attempts did not change anything that the validator complained about. Before retrying, list explicitly: (a) which fields the validator named in the error, (b) what value you sent for each, (c) what shape each field expects (re-read the relevant `get_plugin_schema` result). Then change at least one of those fields' values."

Detection criterion: **same tool name + same arguments_hash + non-success result** for ≥3 consecutive calls.

---

## Evidence trail

- Decoded sequence reproducible via:
  ```bash
  .venv/bin/python -m evals.lib.decode_tools data/sessions.db \
      53bc3cf2-ab90-4940-9679-1b5e7d474650
  ```
- The `arguments_hash` field on the audit envelope is canonical — three identical `set_pipeline` payloads share the same hash, so the "byte-identical retry" detection in Tier 2 implementation is *exactly* this comparison.
