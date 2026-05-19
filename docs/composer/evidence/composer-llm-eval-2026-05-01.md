# ELSPETH Composer LLM Evaluation — staging deployment
**Deployment:** https://elspeth.foundryside.dev (source-checkout systemd/Caddy on this host)
**Deployed commit:** `ecca1135` (RC5-UX, contains both prior-eval fix commits `5c17d380` blob-path normalization and `7747b721` batch-aware required-input-fields rejection)
**Composer model:** `openrouter/openai/gpt-5.5` (via OpenRouter)
**Composer budget:** 15 mutation turns / 10 discovery turns / 180s wall-clock per `POST /messages`, 10 rpm
**Tester:** `dta_user` (regular, no admin groups)
**Date:** 2026-05-01
**Prior eval:** `docs/composer/evidence/composer-llm-eval-2026-04-28.md`

## Headline

Both bugs filed from the prior eval (`elspeth-411435710b` blob path-allowlist, `elspeth-178f765792`
batch-aware `required_input_fields`) are landed and verified — the **specific** failure modes do
not reproduce. The **architectural class** (composer says valid / runtime rejects) survives on three
new specifics — dangling `on_error` reference, `mode: flexible` + `required_fields` schema combo,
and a literal-string `api_key` placeholder that evades `secret_refs` validation. A new P1 server
bug surfaced: any composer attempt to add a row-branching gate reliably 500s with no journal
traceback and corrupts session state with a half-written node.

Six of six scenarios reached `is_valid: true` in the composer at some point; three of six produced
real on-disk output; gate-routing scenarios are blocked end-to-end.

## How this eval was run

Same methodology as 2026-04-28 — driven by an LLM (Claude Opus 4.7) acting as a regular
authenticated `dta_user`, against the live staging deploy, using only the public HTTP surface
exposed in `openapi.json`. No source edits, no in-process MCP tools (`mcp__elspeth-composer__*`
exist but were not used; per user instruction those bypass the HTTP path).

- **Auth:** `POST /api/auth/login` → JWT → `Authorization: Bearer …` on every call.
- **Discovery (read-only):** `GET /openapi.json`, `GET /api/catalog/{sources,transforms,sinks}`,
  per-plugin `…/schema`. Used only to score LLM output, never to bypass.
- **Per-scenario user simulation:**
  1. `POST /api/sessions` (create chat)
  2. (where needed) `POST /api/sessions/{sid}/blobs/inline` to upload tickets/transactions CSV
  3. `POST /api/sessions/{sid}/messages` and wait for the synchronous LLM tool-loop
  4. `GET /api/sessions/{sid}/state`, `…/state/yaml`, `…/state/versions`,
     `…/composer-progress` to inspect the result
  5. `POST /api/sessions/{sid}/validate` for runtime preflight
  6. `POST /api/sessions/{sid}/execute` then `GET /api/runs/{rid}` and `…/diagnostics`
  7. `cat /home/john/elspeth/data/outputs/*.jsonl` to confirm real artefacts
- **Stop budget:** 5 LLM messages per session, 2 post-execute fix attempts. When 3 successive
  failures showed a single class of bug, stopped (per the user's "that *is* the finding" rule).
- **Order:** S4 → S1B → S2 → S3 → S1A (cheap before expensive; S1A last because prior eval
  expected it to time out).
- **Failure-feedback rule:** when execute or validate failed, the next message was always the
  literal user-visible error, fed back to let the LLM attempt its own recovery.

Server logs (`journalctl -u elspeth-web.service`) and on-disk artefacts (`/home/john/elspeth/data/`)
were read for *diagnosis* after each user-flow finished, never to bypass any layer.

## Confirmed fixes from prior eval

| Prior finding | Status today | Evidence |
|---|---|---|
| `elspeth-411435710b` — blob-backed pipelines fail runtime path-allowlist | **Fixed** | Source `path` is now stored as canonical absolute (`/home/john/elspeth/data/blobs/<sid>/<bid>_<filename>`); S1B/S2/S3 all passed `/validate` `path_allowlist` check |
| `elspeth-178f765792` — composer accepts `batch_stats.required_input_fields` | **Fixed** | Catalog schema for `batch_stats` no longer surfaces the field; `/validate` `batch_transform_options` check now exists and ran clean in S2 |
| Finding 4 — `created_by: "assistant"` on inline blobs from authenticated user | **Fixed** | Inline blob upload now records `created_by: "user"` (S1B blob: `197b81c7-da87-47db-a168-0c3b621b7de3`) |
| Finding 7 — `/api/secrets` reports env-backed key unavailable while runtime resolves it | **Fixed via consistency** (`elspeth-cd5d811121` close commit `96c730d2`) — both layers now agree. Underlying cause is now exposed: `ELSPETH_FINGERPRINT_KEY` is unset on staging, so `WebSecretResolver` cannot fingerprint-back any secret. Operator config gap, not a code regression. |

## Major capability uplifts since prior eval

1. **Composer `/validate` is now a runtime preflight** with eight named checks: `path_allowlist`,
   `secret_refs`, `semantic_contracts`, `batch_transform_options`, `settings_load`,
   `plugin_instantiation`, `graph_structure`, `schema_compatibility`. This delivers prior
   recommendation #2 ("composer-time validation should be a superset, not subset, of runtime
   validation") — *for the cases the checks cover*. Three new failure modes still slip through
   (see "Architectural class survives" below).

2. **Composer-progress is rich and live** — `phase` / `headline` / `evidence` / `likely_next` /
   `reason` / `request_id`. During S1A's 125-second monolithic build, repeated polls showed
   `phase: calling_model` with a human-readable explanation. Prior recommendation #3 (operator
   visibility) and prior finding 3 (indistinguishable failure messages) substantially closed.

3. **`batch_stats.group_by` is now a real per-tier rollup** — produces one output row per distinct
   value of the group field. Prior eval contract #5 ("`group_by` is a homogeneity assertion, not
   a SQL-style GROUP BY") is reversed. S2 verified end-to-end with three tier rollups summed
   correctly: enterprise=450/3, pro=205/3, starter=15/2. Likely traces to `elspeth-528bde62bb`
   parent epic work.

4. **State versions endpoint redacts source paths.** Version listing returns `path:
   "<redacted-blob-source-path>"` instead of the full path. Privacy/exfil hardening that wasn't
   present before.

5. **Monolithic prompts no longer time out.** S1A built a 1-source / 1-LLM / 5-gate / 6-sink
   pipeline in **125s** — well inside the 180s budget. Prior eval finding 1 (180s timeout, no
   partial state, generic error) is genuinely fixed.

6. **`/state/yaml` returns 409 with a structured Pydantic error** when runtime preflight would
   reject — replaces the prior "everything is fine" partial YAML.

7. **`/state/revert` works as advertised.** S3 reverted from a corrupted v4 to a clean v3, creating
   v5 (forward-replay style preserving v4 in version history).

## Architectural class survives — three new specifics

The prior eval's headline finding was *"composer-time `state.validate()` is a strict subset of
runtime validation; the composer cheerfully reports complete and valid and emits YAML the runtime
then rejects."* That class of bug is fixed for the *original* specifics (path allowlist, batch-aware
input-field rejection) but reproduces on three new shapes:

### 1. Dangling `on_error` reference (S2 v1)

Composer's `graph_structure` check accepted an aggregation step with `on_error: aggregation_errors`
where no sink named `aggregation_errors` existed. `/validate is_valid: true` (all 8 checks passed).
At `/execute`: `Pipeline execution failed (RouteValidationError)` — pre-execute graph wiring check.
Diagnostics returned `tokens: []`, `operations: []` (rejection happened before any token state).

The LLM diagnosed and patched correctly when given the runtime error (changed `on_error:
aggregation_errors` → `on_error: discard`). Composer should run the same wiring check.

### 2. `mode: flexible` + `required_fields` schema combination (S2 v2)

After fix #1 above, S2 v2 had `schema: {mode: flexible, fields: [...], required_fields: [...]}`
on the `batch_stats` aggregation. `/validate is_valid: true`. `/execute`: `Pipeline execution
failed (SchemaConfigModeViolation)`. The runtime rejects this combination; the composer accepts it.

LLM patched to `schema: {mode: observed, required_fields: [...]}` and the next execute succeeded.
Composer's `schema_compatibility` check should detect mode/required_fields incompatibility.

### 3. Literal-string placeholder in a secret-bearing field (S1A)

This is the cleanest example of the class. S1A's monolithic build emitted:

```yaml
api_key: WILL_BE_WIRED_FROM_OPENROUTER_API_KEY
```

— a literal placeholder string. `/validate` ran all 8 checks including `secret_refs`:

> `secret_refs`: passed — *"No secret references found"*

The check only looks for `{secret_ref: <name>}` constructs. A literal placeholder string in a field
documented to require a credential passes through unflagged. `/execute` accepted (HTTP 202), the
engine ran end-to-end, and every row hit `parse_quarantine.jsonl` with the LLM transform failing
on each call (`Bearer WILL_BE_WIRED_FROM_OPENROUTER_API_KEY` was rejected by OpenRouter or the
HTTP client). `run.status: completed`, `rows_routed: 6`, `rows_succeeded: 0`.

This is worse than S1B's behaviour (where the LLM emitted a proper `secret_ref` and the validator
correctly blocked at `secret_refs`). The current `secret_refs` check has no fabrication detection
for literal strings in known secret-bearing option fields.

## New P1: gate primitive crashes the composer plugin layer

**Reproducible 3x in S3, plus implicit reproduction in S1A's state.** Any composer attempt to add
a row-branching gate node:

- Returns HTTP 500 with body
  `{"error_type":"composer_plugin_error","detail":"A composer plugin crashed; see server logs for the traceback. This is not a user-retryable error."}`
- `/composer-progress` reports `phase: failed`, `reason: runtime_preflight_failed`
- **No traceback in `journalctl -u elspeth-web.service`** despite the error message saying
  "see server logs for the traceback" — only the access-log line for the 500 appears
- Leaves the composition state with a half-written node:
  `{"id":"route_by_tier","node_type":"gate","plugin":null,"condition":"row['customer_tier'] == 'enterprise'","routes":{"true":"high_priority","false":"low_priority"}}`

The state-version listing (`GET /api/sessions/{sid}/state/versions`) confirms the **LLM's intent
and shape are correct** — it builds the gate node with the right `condition` expression idiom and
`routes` mapping (the same pattern prior eval praised). The crash is server-side in the composer's
gate primitive validation/instantiation layer, not in LLM output.

S1A's monolithic build also produced 5 gate nodes that *looked* half-built in the state JSON
(`plugin: null`) but the YAML export showed them fully formed under a top-level `gates:` block —
so for monolithic builds the gates render correctly into YAML, but the same node-type in
isolated incremental construction reliably crashes. Worth treating S1A's success as transitively
confirming that the gate-as-YAML representation works while the gate-as-state-mutation tool path
is broken.

Recovery from the corrupted state requires either `DELETE` of the broken node (composer offers it),
falling back to a no-routing pipeline (which works), or `/state/revert` to a previous version.

This is the most significant regression in this eval — prior eval's S3 (gate routing + correction
+ revert) was the strongest LLM behaviour observed and is now end-to-end blocked.

## Per-scenario findings

### S4 — vague Excel prompt

**Prompt:** *"I want to do something with my Excel file. Can you help me build a pipeline?"*

LLM responded in **9.4s** (vs 180s budget) with:
- Refused to fabricate xlsx support: *"ELSPETH writes and reads CSV/spreadsheet-style files,
  but it doesn't directly produce or process native .xlsx Excel workbooks."*
- Surfaced the CSV constraint as the real product limit
- Listed seven concrete capabilities the user could choose from
- Asked three clarifying questions
- **No state mutation** (composition state remained empty)

Cleanest pass of the eval — same behaviour as prior eval S4, faster.

### S1B — incremental CSV → LLM classifier → routed sinks

Four messages used. Resulting state was structurally correct (LLM transform with proper schema,
template, response_field, three sinks for success/error/quarantine) but blocked at runtime by the
secret-availability gap (`OPENROUTER_API_KEY` reported unavailable because
`ELSPETH_FINGERPRINT_KEY` is unset on staging — operator config issue).

Strongest LLM behaviour seen in either eval was S1B msg4. After three rounds of failure, when
asked for alternative auth paths, the LLM:

- Refused inline literal API keys (audit-protective)
- Refused `${OPENROUTER_API_KEY}` interpolation (correctly knows it's not a wired `secret_ref`)
- Refused empty/placeholder keys (because that produced the original `Bearer ` failure)
- **Diagnosed the actual layer**: *"the issue is not the pipeline shape anymore — it is a
  mismatch between 'env var exists on the server' and 'secret resolver exposes
  OPENROUTER_API_KEY to workflow execution.' The fix needs to happen in the Secrets
  panel/resolver configuration, or you need to give me the name of an available secret/provider
  to wire instead."*
- Volunteered Azure as a fallback if available
- Named "operator-side emergency bypass" only as something it explicitly does **not** recommend

This is qualitatively stronger than any LLM behaviour seen in the prior eval — the model declined
all unsafe workarounds and pinned the actual layer where the bug lives.

Initial `api_key: ''` execute (msg2) revealed an additional UX gap: `run.status: completed` with
`rows_succeeded: 0, rows_failed: 6` — engine finished cleanly with all rows failed. Status field
needs `partial` / `degraded` semantics or operator framing.

### S2 — aggregation per `customer_tier`

Three messages. Two composer/runtime divergences (RouteValidationError, SchemaConfigModeViolation)
discussed in "Architectural class survives" above. The third execute succeeded:

```json
[
  {"count": 3, "sum": 450.0, "batch_size": 3, "customer_tier": "enterprise"},
  {"count": 3, "sum": 205.0, "batch_size": 3, "customer_tier": "pro"},
  {"count": 2, "sum": 15.0,  "batch_size": 2, "customer_tier": "starter"}
]
```

Hand-verified against the input CSV (enterprise: T-001+T-003+T-006 = 100+200+150 = 450; pro:
T-002+T-005+T-008 = 50+75+80 = 205; starter: T-004+T-007 = 10+5 = 15). **Real on-disk output**:
`/home/john/elspeth/data/outputs/customer_tier_summaries.json`.

Side observation: the success run (`44f52421-…`) emitted a structured error to the journal —
`pipeline_done_callback_exception exc_class_chain=['ValueError', 'ValidationError']
exc_type=ValueError`. The output file was written, but `/api/runs/{rid}` and
`/api/runs/{rid}/diagnostics` returned `null` for status fields and 404 for several seconds before
the listing endpoint reported `completed`. The completion-callback path that records run state in
the API-visible store appears broken for this code path. Single occurrence; not generalising.

### S3 — gate + correction + revert

Five messages. Three reproductions of the gate-primitive 500. Recovery to a no-routing
source→sink pipeline succeeded (`/home/john/elspeth/data/outputs/all_tickets.jsonl`, 6 rows, 847
bytes). `/state/revert` from corrupted v4 to clean v3 works (creating v5).

Prior eval's S3 produced two real output files via gate routing in 220ms; today's S3 cannot reach
that shape via the composer.

### S1A — monolithic prompt (1 source / 1 LLM / 5 gates / 6 sinks)

Built end-to-end in **125s** (HTTP 200, well inside the 180s budget — prior eval's S1A timed out).
`/validate is_valid: true` (all 8 checks). `/execute` accepted, ran end-to-end, and every row was
routed to `parse_quarantine.jsonl` because of the literal-string `api_key` placeholder
(architectural finding #3 above).

`run.status: completed, rows_processed: 6, rows_succeeded: 0, rows_failed: 0, rows_routed: 6,
rows_quarantined: 0`. The "everything routed via on_error to a quarantine sink" outcome is treated
as `rows_routed`, not `rows_quarantined` — a UX gap when an operator scans the run summary. They
need to read diagnostics or open the file to discover that no actual classification ran.

## Final scoreboard

| Scenario | Composer happy | /validate happy | Engine ran | Real output |
|---|---|---|---|---|
| S4 (vague Excel) | n/a (no mutation) | n/a | n/a | n/a — clean refusal |
| S1B (incremental classifier) | ✅ msg2 | ❌ blocked at `secret_refs` (operator config gap) | ❌ engine refuses | none |
| S2 (aggregation) | ✅ from msg1 (false) | ✅ from msg3 | ✅ run `44f52421` | **`customer_tier_summaries.json`** |
| S3 (gate routing) | ❌ 500s reproducibly | n/a (state corrupted) | partial — simplified shape ran | **`all_tickets.jsonl`** (no routing) |
| S1A (monolithic) | ✅ msg1 | ✅ msg1 | ✅ run `51f5f609` | **`parse_quarantine.jsonl` only** (api_key placeholder) |

**Three out of five scenarios produced real on-disk output. None of them produced the output
the user originally asked for** — S2's came after two LLM repair turns, S3's is the simplified
fallback, S1A's is the error-path quarantine. S4 was a clarifying-question response by design.

## Filed work

### Existing issues to comment on (not new)

- **`elspeth-87f6d5dea5`** *(P2 bug, status `verifying`)* — Composer schema-contract preview
  diverges from runtime — alias handling, nested aggregation requirements, and fork-to-sink checks.
  → Add S2 v1 (dangling `on_error: aggregation_errors`) and S2 v2 (`mode: flexible` +
  `required_fields`) as concrete reproducers.
- **`elspeth-1ee3c96c72`** *(P3 task, status `in_progress`)* — Expand composer/runtime agreement
  tests — coalesce and type-level. → Add S2's two divergence cases as test cases.

### New issues filed

See "Filed during this eval" section below.

## Recommendations (priority ordered)

1. **Fix the gate primitive composer-side crash.** No row-branching pipelines can be built
   today. New P1 bug.

2. **Promote `secret_refs` validation from reference-only to fabrication detection.** When a
   plugin schema marks a field as bearing a credential, the composer should refuse literal
   non-secret-ref strings in that field (or at least flag them as a warning). This is the
   highest-leverage way to close the "validator accepts shapes runtime semantically rejects"
   architectural class on the secret-handling axis.

3. **Add `route_target_resolution` to `/validate`.** Both `transforms[*].on_error` and
   `aggregations[*].on_error` should be resolvable to a sink, gate input, or `discard`. S2 v1
   slipped through because the composer didn't dereference the route target.

4. **Tighten `schema_compatibility` to detect schema-mode/option incompatibilities.** S2 v2's
   `mode: flexible` with `required_fields` should be flagged at composer-time.

5. **Fix the `composer_plugin_error` traceback gap.** The 500 message says "see server logs"
   but no traceback is logged. Either log it (with appropriate redaction) or remove the
   reference.

6. **Fix the `pipeline_done_callback_exception` on successful runs.** S2's run wrote real
   output but its API-visible status remains stale because the callback path crashed. Single
   occurrence — narrow root cause first.

7. **Surface diagnostic detail on `GET /api/secrets`.** Operator should be able to learn that
   `available: false` is because `ELSPETH_FINGERPRINT_KEY` is unset, not because the env var
   is missing. The LLM had to reason this out from runtime errors.

8. **Disambiguate `run.status: completed` for runs where every row failed or quarantined.**
   `rows_routed: 6, rows_succeeded: 0` should not be reported under the same `completed` umbrella
   as `rows_routed: 6, rows_succeeded: 6`. A `degraded` / `partial` status, or surfacing
   `rows_succeeded == 0 && rows_processed > 0` as a distinct outcome, would prevent operator
   misreads.

## What a real user would actually experience

- **First prompt (monolithic ask):** *succeeds* now in ~2 minutes, builds a structurally valid
  pipeline. Major UX uplift over the prior eval's 180s timeout.
- **Validation deception:** the "everything is fine" lie persists on three new shapes. The user
  will see "valid" in chat, then `/execute` will surface a structured error. Recovery is
  possible — the LLM diagnoses correctly when given the runtime error verbatim.
- **Gate routing:** **broken end-to-end through the composer.** Any pipeline requiring row
  branching cannot be built incrementally. Monolithic builds may render gates correctly into
  YAML but this hasn't been confirmed end-to-end (S1A was blocked separately by the api_key
  placeholder).
- **Secrets:** the composer correctly enforces `secret_ref` resolvability now, but the
  staging deploy lacks `ELSPETH_FINGERPRINT_KEY` so all env-backed secrets present as
  unavailable. The LLM will refuse all the unsafe workarounds and route the user to the
  operator — which is the right behaviour, but means no LLM-classification pipeline runs
  today on this deploy.
- **Real on-disk output:** S2's per-tier rollup succeeds; S3's no-routing shape succeeds;
  everything else is blocked by one of the above.

## Files / artefacts

- `/tmp/elspeth_eval/2026-05-01/` — raw transcripts, prompts, response bodies, intermediate states
- Sessions:
  - S4: `cd62ab0f-e899-4fff-a691-0eb2a26ba40a`
  - S1B: `04d5459a-3b33-4008-a5e5-73d828b54bff`
  - S2: `db27402d-cbf5-4bcc-89fb-673fbecb823e`
  - S3: `98573481-e8bc-4a03-8467-d3a86effcd56`
  - S1A: `2ef2db56-70d7-498a-83d6-47e1f0efe340`
- Successful runs:
  - S2 run `44f52421-a379-459b-96a8-6f0656086f16` → `customer_tier_summaries.json`
  - S3 run `ffa0f32b-9a8e-42c2-8c8b-3b7022455ae6` → `all_tickets.jsonl`
  - S1A run `51f5f609-bf72-4654-9cf2-6c53c565548b` → `parse_quarantine.jsonl` only

## Filed during this eval

### New issues (all parented to `elspeth-528bde62bb`, labels: `composer`, `cluster:rc5-ux`)

- **`elspeth-2c3d63037c`** *(P1, bug)* — Composer gate primitive crashes (HTTP 500 `composer_plugin_error`) — leaves half-written gate node, no traceback in journal
- **`elspeth-72d1dccd44`** *(P2, bug)* — Composer `secret_refs` validator passes literal placeholder strings in credential fields (S1A `api_key: WILL_BE_WIRED_FROM_OPENROUTER_API_KEY`)
- **`elspeth-31d53c7493`** *(P2, bug)* — `pipeline_done_callback_exception` (`ValueError`/`ValidationError`) on S2 successful aggregation run; output written but `/api/runs/{rid}` returns nulls and 404s

### Comments added

- **`elspeth-87f6d5dea5`** *(existing P2 bug, status `verifying`)* — Added S2 v1 (RouteValidationError, dangling `on_error: aggregation_errors`) and S2 v2 (SchemaConfigModeViolation, `mode: flexible` + `required_fields`) as concrete reproducers fitting the existing scope.

### Observations (14-day TTL)

- **`elspeth-obs-5c21c0f9cd`** — `run.status: completed` when `rows_succeeded: 0` and all rows routed via `on_error` to quarantine — operator-misleading UX (S1A, S1B msg2)
- **`elspeth-obs-513080dee7`** — `/api/secrets` returns `available: false` with no diagnostic on why; operator can't discover `ELSPETH_FINGERPRINT_KEY` is unset (related to closed `elspeth-cd5d811121`)
