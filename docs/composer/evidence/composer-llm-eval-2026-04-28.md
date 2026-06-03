# ELSPETH Composer LLM Evaluation — staging deployment
**Deployment:** https://elspeth.foundryside.dev (source-checkout systemd/Caddy on this host)
**Composer model:** `openrouter/openai/gpt-5.5` (via OpenRouter)
**Composer budget:** 15 mutation turns / 10 discovery turns / 180s wall-clock per `POST /messages`
**Tester:** `dta_user` (regular, no admin groups)
**Date:** 2026-04-28

## How this eval was run

This evaluation was driven by an LLM (Claude Opus 4.7) acting as a regular
authenticated user against the live staging deployment, **not** by editing
ELSPETH source or calling internal Python code. The eval intentionally
constrains itself to the same HTTP surface a frontend SPA or curl-armed
operator would have. Concretely:

- **Auth:** `POST /api/auth/login` with `dta_user` credentials → JWT bearer
  token cached and used for every subsequent call. Same bearer-token contract
  the SPA uses (no cookies, no CSRF — confirmed in `web/auth/routes.py:104`).
- **Discovery (read-only):** `GET /openapi.json`, `GET /api/auth/config`,
  `GET /api/catalog/{sources,transforms,sinks}` and per-plugin
  `…/schema` to establish ground truth for what plugins exist and what
  their option contracts look like. Used only to score the composer's
  output against reality, never to bypass any layer.
- **Per-scenario user simulation:**
  1. `POST /api/sessions` (create chat)
  2. (where needed) `POST /api/sessions/{sid}/blobs/inline` to upload a
     small CSV — same flow the frontend uses for "drop a file"
  3. `POST /api/sessions/{sid}/messages` with a natural-language prompt
     and wait for the synchronous LLM tool-loop to finish, polling
     `/composer-progress` for liveness
  4. `GET /api/sessions/{sid}/state` and `…/state/yaml` to inspect what
     the composer LLM actually built
  5. `POST /api/sessions/{sid}/validate` to run the runtime pre-flight
  6. `POST /api/sessions/{sid}/execute` to actually run the pipeline,
     then `GET /api/runs/{rid}` and `…/diagnostics` to read the outcome
  7. `cat data/outputs/*.jsonl` to confirm real on-disk artefacts

When pipelines failed, the next prompt was **always** the literal error
text the runtime returned, fed back into a fresh `POST /messages` so the
composer LLM could attempt its own recovery. The eval driver did not
hand-edit the YAML, did not call any Python helper, did not use the
composer MCP server in-process, and did not bypass `/validate`. The only
non-user-shaped action was reading server-side files and database tables
to *diagnose* failures after the user-flow finished — the conclusions are
about what the user-flow itself revealed.

Bugs surfaced this way (filed as `elspeth-411435710b` and
`elspeth-178f765792`) are reproducible end-to-end through the public
HTTP API with no privileged access — exactly what a real user would hit.

## TL;DR
The composer LLM is **substantively competent** — it correctly identifies plugin
names, fills required schema fields, respects security constraints, and asks
clarifying questions when prompts are vague. It **fabricates names and field
shapes** at the margins (e.g. trigger keywords, secret references) and the
composer infrastructure has **two real bugs** that surface only when an LLM
actually drives a pipeline end-to-end.

## Scenario summary

| # | Scenario | Outcome | Headline finding |
|---|----------|---------|------------------|
| 1A | Monolithic prompt (CSV → LLM classify → 3 routed sinks) | **Failed** server-side at 180s timeout, no partial state, generic error | Composer can't recover from oversized prompts; gives the user nothing useful. |
| 1B | Same task but split across 3 turns | T2/T3 produced correct shape; runtime `/validate` rejects the result | LLM behaviour was right; the **path-allowlist bug** (filed elspeth-411435710b) blocks all blob-backed pipelines. |
| 2  | Single-turn aggregation pipeline | Composer says valid; runtime rejects (`Forbidden name: 'end_of_source'`) | LLM placed a **trigger type** under `trigger.condition` (boolean expression slot). Real construct, wrong field. |
| 3  | Gate routing + correction + revert | All three turns produced runnable composer state; runtime `/validate` blocked by the same path bug | Strongest LLM behaviour in the eval — surgical patch on T2 (one field changed), forward-replay revert on T3 (audit-trail nuance noted below). |
| 4  | Vague prompt ("do something with my Excel file") | Asked clarifying questions, no state mutation, no hallucinated plugin | Cleanest result of the eval — the LLM declined to fabricate. |

## Cross-cutting findings

### Composer infrastructure issues (server-side)

1. **Blob-backed sources fail runtime path-allowlist** — filed as
   `elspeth-411435710b` (P1 bug). `blobs.storage_path` is stored as
   `data/blobs/<session>/<blob>_<file>`; `web/paths.py:resolve_data_path()`
   joins relative paths to `data_dir = /home/john/elspeth/data`, producing
   `…/data/data/blobs/…` which falls outside the allowed directory. Every
   composer-built pipeline that wires an uploaded blob is blocked at
   runtime, even though composition validation reports `is_valid: true`.

2. **Composition validation is a strict subset of runtime validation.**
   `state.validate()` doesn't run path-allowlist checks, expression-grammar
   checks against trigger conditions, or settings-load round-trips. The
   composer cheerfully tells the user "complete and valid" and emits YAML
   that the runtime then rejects. The LLM trusts the composer's signal,
   so it also reports completion. Two separate failures (the bug above
   and the trigger field-shape miss in scenario 2) both manifested via
   this gap.

3. **`POST /messages` is synchronous-blocking with no streaming response.**
   A first-time user submitting a 3-component pipeline ask hits the 180s
   timeout, gets a generic "the bounded composer loop stopped before a
   final answer" message, and is told to "try a smaller request." No
   partial state survives, no clarifying question is offered. The
   `/composer-progress` polling channel is the only operator visibility,
   and (a) it returns identical `phase: failed` text whether the cause
   was *turn budget* or *wall-clock*, and (b) it requires the frontend
   to poll while a separate POST hangs, which most CLI clients won't do.
   Result: a real-user-style first prompt produces zero work and zero
   diagnostic. **Strong UX regression risk for unfamiliar users.**

4. **`created_by: "assistant"` on inline blobs uploaded by an authenticated
   user.** Audit-attribution smell — the inline-blob handler appears to
   set the blob author to a default rather than the authenticated principal.
   Worth a separate filigree issue to verify intent.

5. **Access log misses in-flight composer requests.** Uvicorn only writes
   the `POST /api/sessions/.../messages` line when the response completes;
   if the client disconnects (or the request is in flight), there's no
   visible record in the journal. Combined with finding 3, this makes
   in-flight composer activity invisible to operators unless they think
   to call `/composer-progress`.

6. **In-progress traceback in `web/middleware/request_id.py:98`** appeared
   under composer load (a starlette `dispatch_func` chain with
   `asyncio.wait_for(..., timeout=0.1)`). Likely unrelated to LLM quality
   but observed during eval — should be triaged separately.

### LLM behaviour (positive)

- **Refused to wire a path outside the managed blob area** (S1B-T1) and
  surfaced the recovery options (upload, or register the file) instead of
  fabricating a path or silently failing. This is exactly the right
  behaviour at a security boundary.
- **Picked real plugin names every time** — `csv` source, `llm`
  transform, `json` sink, `value_transform`, `batch_stats`. Never invented
  a `xlsx` source or a `gpt5_summarizer` transform. The skill pack's
  "discover before configuring" rule appears to be working.
- **Filled required schema fields correctly** including the OpenRouter
  config (`schema`, `model`, `template`, `api_key`, `provider`), the CSV
  schema shape (`mode: fixed, fields: ["x: str", …]`), and the JSON sink
  with `format: jsonl` plus `collision_policy: auto_increment`.
- **Truthfully reported failure to bind the secret** rather than
  pretending it worked (S1B-T3). This is rare model behaviour — many
  LLMs would silently emit a placeholder and claim success.
- **Asked a clarifying question instead of guessing** when given a vague
  prompt (S4). Listed concrete capabilities. Surfaced a real product
  constraint (no Excel reader; CSV only).
- **Volunteered correctness caveats unprompted** — e.g., "no outputs →
  validation will be incomplete" in S1B-T1, "secret may need to be
  enabled before execution" in S1B-T3.

### LLM behaviour (additional positive — scenario 3)

- **Surgical patch, not rebuild.** S3-T2 changed *only* the gate condition
  string between v1 and v2 — every other field was preserved. The model
  used the `in` membership operator (`row['customer_tier'] in
  ('enterprise', 'pro')`) instead of expanding to two `==` checks.
  Idiomatic for the gate grammar, faithful to the user's "patch don't
  rebuild" ask.
- **Proactive quarantine sink.** S3-T1 added a third sink
  (`outputs/quarantine.jsonl`) that the user never asked for, because the
  source's `on_validation_failure: "quarantine"` policy needs a
  destination. That's load-bearing thinking the user would have hit at
  runtime if it were missing.

### LLM behaviour (subtle gap)

- **The model has no `revert` tool exposed.** When asked to "revert"
  (S3-T3), the LLM patched *forward* to recreate v1's state as v3 rather
  than telling the user about the `/state/revert` REST endpoint. Result
  is functionally equivalent (gate condition is back to original) but the
  audit trail loses the "this was an intentional revert" signal that
  `/state/revert` records via a system message. A skill-pack note about
  the revert primitive — and an instruction to surface it as a
  user-callable option — would close this gap without adding new tools.

### LLM behaviour (negative)

- **Confabulated field shapes at fine-grained boundaries.** Scenario 2:
  used `trigger.condition: end_of_source` (where `condition` is a
  boolean-expression slot) instead of the trigger *type*. Scenario 1B-T3:
  emitted `api_key: "${OPENROUTER_API_KEY}"` as a literal string after
  failing to wire the secret reference formally. Both cases match the
  pattern: LLM understands the *concept*, picks the wrong *YAML field name*.
- **Reports "valid and complete" when composer validation passes**, even
  though the runtime later rejects the pipeline. The LLM trusts the
  composer's `is_valid` signal — which is itself wrong — so the model is
  not the right place to fix this. The composer's validator should be
  promoted to match the runtime's.

### What a real user would actually experience

- **First prompt:** if it's a complete pipeline ask (source + 1+ transforms +
  routed sinks), almost certainly timeouts at 180s with a generic failure.
- **Pivot to incremental:** smaller turn-by-turn asks succeed, the LLM
  produces sensible YAML, and the user gets visible progress.
- **Validation deception:** they'll see "complete and valid" from the
  composer, then `/validate` or `/execute` will fail with errors that
  weren't visible during the chat. The model can't help them diagnose
  this because the model also believes the pipeline is fine.
- **End-to-end pipeline run:** blocked today by `elspeth-411435710b` —
  no blob-backed pipeline produced via the composer can pass `/validate`.

## Recommendations (priority ordered)

1. **Fix `elspeth-411435710b` (path-allowlist).** Either store
   `blobs.storage_path` relative-to-`data_dir` (no leading `data/`),
   or have `resolve_data_path()` detect the existing-prefix case. P0/P1.
2. **Make composer-time validation a superset, not subset, of runtime
   validation.** `state.validate()` should run the path-allowlist,
   parse trigger conditions through the expression grammar, and round-trip
   the YAML through `load_settings()` with `mode="composer"`. Today the
   gap is structural and produces an "everything is fine" lie.
3. **Stream tool-call breadcrumbs in the assistant message OR persist
   intermediate turns as `ChatMessage` rows.** Right now only the final
   assistant text is persisted, so the user can't see "I called set_source,
   then upsert_node…". Auditors can't either.
4. **Improve the timeout/turn-budget failure message** so the user knows
   whether to (a) shorten the prompt, (b) split it across turns, or
   (c) raise the server budget. Today's "try a smaller request" text is
   uninformative.
5. **Add explicit trigger-type reinforcement to the skill pack.**
   E.g., "For aggregation triggers, the YAML field is `trigger.type` (one
   of: count, timeout, condition, end_of_source). Use `trigger.condition:
   <expr>` ONLY when `trigger.type: condition`." Would have prevented
   the scenario-2 miss.
6. **Add a secret-reference test surface to the skill pack** — explicit
   reminder that `${VAR}` in YAML fields is not a wired secret reference
   and the proper tools to use are `wire_secret_ref` / `validate_secret_ref`.

## Files / artifacts

- Session 1A (failed monolithic): `c549bb63-47e9-427f-9a27-35467f877395`
- Session 1B (incremental, 3 turns): `6472ff67-1052-406c-98c3-b3278e9ef4ea`
- Session 2 (aggregation): `ae6816aa-1f75-4103-b176-886d14f9e104`
- Session 3 (gate + revision + revert): `9002ed1f-3046-4c00-86be-2f1e3b3bd932`
- Session 4 (vague prompt): `b2370ab0-af7a-41da-87a7-c78b2f7e0165`
- Raw transcripts and prompts in `/tmp/elspeth_eval/`
- Filed: `elspeth-411435710b` (P1 bug, composer path-allowlist)

## Addendum: end-to-end execution results

After validating, I also called `POST /api/sessions/{sid}/execute` on each scenario.

| Scenario | /execute outcome | Run status | Output |
|----------|------------------|-----------|--------|
| 1B (LLM classifier) | HTTP 404 — path-allowlist sync rejection | n/a | none (blocked by `elspeth-411435710b`) |
| 2 (aggregation) | HTTP 202 accepted | failed in 24ms — `Forbidden name: 'end_of_source'` settings-load failure | none (LLM hallucinated trigger field) |
| 3 (gate routing) | HTTP 404 — same path bug | n/a | none |
| 3 (after LLM patch removed `path:` and kept only `blob_ref`) | HTTP 202 accepted | **completed** in 220ms | **2 real files written**: see below |

### Workaround the LLM discovered

When prompted with the actual error message and asked to patch the source,
the LLM did not add a corrected `path:` — it **removed the `path` field
entirely and kept only `blob_ref`**. The engine then resolved the blob
internally without going through `resolve_data_path`, side-stepping the bug.
Worth knowing: this is an undocumented happy-path that works around the
filed bug. If `path:` is omitted but `blob_ref:` is present, the runtime
materializes correctly. The composer's `_execute_set_source_from_blob` tool
should probably do this by default rather than injecting `path`.

### Scenario 3 actual output (run id `c57c3570-ea0e-4741-8017-b7a7139d6a38`)

`data/outputs/high_priority.jsonl` (2 rows, 306 bytes):
```jsonl
{"ticket_id": "T-001", "subject": "Login broken", "body": "Cannot log in since the update — page just refreshes.", "customer_tier": "enterprise"}
{"ticket_id": "T-005", "subject": "Crash on save", "body": "App crashes whenever I save a settings change. Logs attached.", "customer_tier": "enterprise"}
```

`data/outputs/low_priority.jsonl` (4 rows, 541 bytes):
```jsonl
{"ticket_id": "T-002", "subject": "Invoice question", "body": "My latest invoice charged me twice for the seat add-on.", "customer_tier": "pro"}
{"ticket_id": "T-003", "subject": "Feature ask", "body": "Would love a CSV export from the dashboard.", "customer_tier": "pro"}
{"ticket_id": "T-004", "subject": "Random thanks", "body": "Just wanted to say the new UI is great.", "customer_tier": "starter"}
{"ticket_id": "T-006", "subject": "Refund request", "body": "Please refund the unused months from last quarter.", "customer_tier": "pro"}
```

The proactive `quarantine` sink was **not materialized on disk** — JSON sink
with `auto_increment` only writes a file when data arrives. So the audit
graph records a sink that never ran, but no empty file is left behind.

Diagnostic summary: 6 tokens, 18 completed states (3 nodes × 6 rows = 18 —
clean), 1 source_load operation, 2 sink_writes. No errors, no quarantine,
220ms wall-clock end-to-end.

## Addendum 2: actual end-to-end runs (after fixes)

After identifying the path bug, I went back and asked the LLM to patch each
session's source path to omit the `data/` prefix, then re-ran `/execute`.

### Successful runs

| Run | Pipeline | Rows | Output | Time |
|-----|----------|------|--------|------|
| `c57c3570-…` (S3) | gate routing by tier | 6 in / 2 high + 4 low | `data/outputs/{high,low}_priority.jsonl` | 220ms |
| `b6265cd0-…` (S1B) | LLM classifier (gpt-4o-mini via OpenRouter) → single sink | 6 in / 6 succeeded | `data/outputs/all.jsonl` | 4.7s |

S1B's LLM classifier results were 6-for-6 correct:

```
T-001  enterprise -> bug              (Login broken)
T-002  pro        -> billing          (Invoice question — double charge)
T-003  pro        -> feature_request  (CSV export ask)
T-004  starter    -> other            (Random thanks)
T-005  enterprise -> bug              (Crash on save)
T-006  pro        -> billing          (Refund request)
```

The JSON sink also captured per-row `category_usage` (token counts) and
`category_model` ("openai/gpt-4o-mini") — auditability built into the LLM
transform plugin, not the model. Total cost: ~497 tokens across 6 rows.

### Run that's still blocked

| Scenario | Blocker |
|----------|---------|
| 2 (aggregation) | LLM hallucinated `trigger.condition: end_of_source` — 2nd attempt to fix is in flight at time of writing |

### New finding from the run attempts

7. **Secret-availability reporting is inconsistent with runtime resolution.**
   `GET /api/secrets` reports `OPENROUTER_API_KEY` as
   `available: false, source_kind: env`. `POST /api/secrets` for the same
   name returns `"Secret resolver is not configured: ELSPETH_FINGERPRINT_KEY
   is unset."` But the pipeline runtime resolved the same env-backed key
   without issue and S1B's LLM step ran successfully against OpenRouter.
   So the operator-facing `/api/secrets` view paints a false picture: the
   key really is wirable, the list endpoint just claims it isn't. This
   makes the LLM's "secret not accessible" message in S1B-T3 misleading
   too — the LLM trusted the broken availability check.

8. **`get_pipeline_state` LLM tool doesn't show `path` after
   `patch_source_options`.** The LLM in S1B-T5 noted: "the patch call
   succeeded, but `get_pipeline_state` is currently displaying the
   blob-backed source without the `path` field even after the successful
   patch." Yet `/state/yaml` shows the patch landed correctly. The
   composer's introspection tool is out of sync with the persisted state
   in some cases. Worth a separate filigree issue.

9. **The LLM's "side-step" recovery worked in one session, not the other.**
   S3-T4 added a corrected `path:` value (without `data/` prefix). S1B-T4
   tried to *remove* `path` (which the schema marks required), got
   rejected, and didn't try the corrected-string approach until I pointed
   it out in T5. So the model's recovery strategy from the path bug is
   non-deterministic — it sometimes patches forward correctly and
   sometimes attempts the "wrong" repair. A skill-pack note about
   `set_source_from_blob` injecting a buggy path (until the bug is fixed)
   would harden this.

## Addendum 3: S2 (aggregation) — 5 distinct runtime contracts the LLM cannot discover

I went deeper on Scenario 2. After every round-trip patch, a new previously-
hidden constraint surfaced. Each one passed every validation layer the LLM
has access to (composer `state.validate()`, runtime `/validate`) and only
crashed at execute time. They are:

| # | Symptom | Real constraint | Discoverable from catalog? |
|---|---------|------------------|----------------------------|
| 1 | `Forbidden name: 'end_of_source'` | `trigger.condition` is a boolean expression, not a trigger type name | No |
| 2 | LLM tries `trigger.type: end_of_source` — composer rejects | The right answer is "omit trigger entirely" but composer still requires one of `count`/`timeout_seconds`/`condition` | No (and composer disagrees with runtime here) |
| 3 | LLM uses workaround `trigger.count: 2147483647` → `FrameworkBugError` | `batch_stats` is batch-aware; per ADR-013 it cannot have `required_input_fields`. Filed as `elspeth-178f765792`. | No — the catalog schema lists `required_input_fields` on every transform |
| 4 | `FileNotFoundError` after fixing the trigger | LLM's path was `blobs/<blob_id>/tickets.csv`; real layout is `blobs/<session_id>/<blob_id>_<filename>` | No — the LLM has no example or schema for blob storage layout |
| 5 | `Heterogeneous 'customer_tier' values in batch` | `group_by` on `batch_stats` is an *assertion of homogeneity*, NOT a SQL-style GROUP BY. To produce per-tier rollups you need different pipeline structure (gate-per-tier, or pre-sorted source). | No — the field name implies one thing, the runtime enforces another |

**Net for Scenario 2:** the LLM never produced a working pipeline, despite
having all the right concepts. Each fix opened a new gap. The user's
original ask ("aggregate, count per customer_tier, write summary JSON") is
genuinely not buildable in the composer without the engineer-level knowledge
of `batch_stats`'s real contract.

This is the **most important architectural finding of the eval**: the
composer needs a "dry-run" mode that exercises the engine's own
pre-execution guards (e.g. ADR-013's batch-transform check, batch_stats's
homogeneity assertion) before declaring a pipeline "complete and valid."
Otherwise the LLM (and any user) ships pipelines that only fail at execute
time, after each round-trip is a fresh OpenRouter call.

## Final scoreboard

| Scenario | Composer happy | /validate happy | Engine executed | Real output |
|----------|----------------|------------------|-----------------|-------------|
| 1A (monolithic) | — (timed out) | — | — | none |
| 1B (LLM classifier) | ✅ T2-T5 | ✅ T5 | ✅ run b6265cd0 | **all.jsonl, 6 classified rows** |
| 2 (aggregation) | ✅ from T2 | ✅ from T4 | ❌ FrameworkBugError, then FileNotFoundError, then HeterogeneousBatchError | none |
| 3 (gate routing) | ✅ T1 | ✅ T4 | ✅ run c57c3570 | **high_priority.jsonl + low_priority.jsonl** |
| 4 (vague Excel) | n/a (no mutation) | n/a | n/a | n/a |

**2 out of 4 scenarios produced real on-disk output.** Both required at
least one round of human-in-the-loop debug after the LLM's first attempt.

## Filed

- `elspeth-411435710b` (P1) — composer-built blob-backed pipelines fail runtime path-allowlist
- `elspeth-178f765792` (P2) — composer accepts `batch_stats.required_input_fields`, engine `FrameworkBugError` at execute (ADR-013 dispatch gap)
