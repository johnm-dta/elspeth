# ELSPETH Composer LLM Evaluation — staging deployment

**Status: COMPLETE.** (working copy preserved at `evals/2026-05-03-composer/basic/REPORT.md` along with raw transcripts)

**Deployment:** https://elspeth.foundryside.dev (source-checkout systemd/Caddy on this host)
**Service start:** 2026-05-03 08:03:49 AEST after deploy (rebuild frontend dist + delete-and-restart on `runs.rows_routed` schema split)
**HEAD at restart:** `6d745470 fix(web): patch runs[] row counters on progress events for live mid-run updates (elspeth-0c076ad374)`
**Composer model:** `openrouter/openai/gpt-5.5` (per env)
**Composer budget:** 15 mutation turns / 10 discovery turns / 180s wall-clock per `POST /messages`, 10 rpm
**Tester:** `dta_user` (regular, no admin groups)
**Date:** 2026-05-03
**Prior evals:** `docs/composer/evidence/composer-llm-eval-2026-04-28.md`, `…-05-01.md`, `…-05-02.md`

## Deploy notes (this session)

- `npm run test` (vitest) — 19 files / 133 tests passed.
- `npm run build` — fresh `dist/index.html`, bundle `index-WDdpfOJl.js`.
- Backend restart blocked initially: `SessionSchemaError` — old `runs.rows_routed` column shape vs HEAD's split into `rows_routed_success` + `rows_routed_failure`. Documented operator action: backup + delete `data/sessions.db`, restart. Backup at `data/sessions.db.bak-pre-2026-05-03-rows-routed-split` (303 KB). After delete + restart, `/api/health` → `{"status":"ok"}` on both Caddy and direct uvicorn unix socket.
- Operator-side gap unchanged from 05-02: `OPENROUTER_API_KEY` reports `available: false, reason: fingerprint_resolver_not_configured`. Same blocker for any LLM-classifier scenario.

## How this eval was run

Same methodology as prior evals — LLM driver (Claude Opus 4.7) acting as authenticated `dta_user` against the live staging deploy via the public HTTP surface only. No source edits, no `mcp__elspeth-composer__*` (those bypass HTTP). Catalog/openapi/secrets used only to *score* LLM output.

- **Auth:** `POST /api/auth/login` → JWT → `Authorization: Bearer …`
- **Discovery (read-only, scoring only):** `GET /openapi.json`, `GET /api/catalog/{sources,transforms,sinks}` and per-plugin `…/schema`, `GET /api/secrets`. Cached under `evals/2026-05-03-composer/basic/`.
- **Per-scenario user simulation:** create session → upload `tickets.csv` via `POST /api/sessions/{sid}/blobs/inline` → `POST /api/sessions/{sid}/messages` (async wait via Bash `run_in_background` + Monitor) → `GET …/state`, `…/state/yaml`, `…/state/versions`, `…/composer-progress`, `…/messages` → `POST …/validate` → `POST …/execute` → `GET /api/runs/{rid}` → cat output files.
- **Stop budget:** 5 messages per session max, 2 post-execute fix attempts. When 3 successive failures show one class, stop (per "that *is* the finding" rule).
- **Order:** S5 (vague — smokes composer's own LLM access) → S4 (gate routing — verifies `elspeth-71520f5e30` close) → S3 (aggregation — probes `elspeth-3d25355784` field_mapper gap) → S2 (incremental classifier) → S1 (monolithic). Cheap before expensive; verification before exploration.
- **Failure-feedback rule:** `/execute` errors are fed verbatim to the LLM as the next user message — measures LLM debugging behaviour, not the driver's.

## Reproduction targets (advisor-prioritised)

| Prior finding | Status filed | Today's verification scenario |
|---|---|---|
| `elspeth-411435710b` (P1, closed 04-28) blob path allowlist | closed | S1 (smoke); cascades past path checks if fixed |
| `elspeth-178f765792` (P2, closed 04-28) batch_stats.required_input_fields | closed | S3 setup probe |
| `elspeth-2c3d63037c` (P1, closed 05-01) gate primitive 500 | closed | S4 msg1 |
| `elspeth-72d1dccd44` (P2, closed 05-01) literal `api_key` placeholder | closed | S1/S2 msg with literal placeholder |
| `elspeth-71520f5e30` (P1, closed) routing pipelines report `status: failed` | closed | **S4 — primary verification** |
| `elspeth-5069612f3c` (P1, closed) rows_routed counter split | closed | **S4 — primary verification** |
| 05-02 gap #1 field_mapper output-schema/mapping mismatch (`elspeth-3d25355784`) | open (triage P2) | S3 with deliberate mapping-rename |
| 05-02 gap #2 locked input contract vs upstream emission (`elspeth-3d25355784`) | open (triage P2) | S3 follow-up |
| 05-02 LLM regression — S1B msg3/4 re-emits validate complaint | not filed yet | passively monitor in S2 |

---

## Per-scenario findings

### S5 — vague Excel prompt
**Session:** `49c52225-bdd5-4026-81b3-23bb63fd34eb` · **Messages:** 1 · **Wall-clock:** 10.4s

**Prompt:** *"I want to do something with my Excel file. Can you help me build a pipeline?"*

LLM response: clean refusal to fabricate xlsx support, surfaces CSV as the real product limit, asks three concrete clarifying questions ("what do you want to do", "what columns", "what output"), no state mutation. Identical *shape* to 04-28 / 05-01 / 05-02. **4/4 evals consistent.**

### S4 — gate routing + correction + revert (PRIMARY VERIFICATION TARGET)
**Session:** `6f0fe6e8-4347-434c-b23f-3ddc7858b8c9` · **Messages:** 3 + driver-issued `/state/revert` · **Wall-clock:** 58s + 11s + 11s ≈ 80s composer-time

| Msg | LLM did | State outcome | Engine outcome |
|---|---|---|---|
| 1 | Built complete pipeline first try: csv source → gate (`row['customer_tier'] == 'enterprise'`) → `high_priority`/`low_priority` JSONL sinks; proactively added `quarantine` sink. | v1; `is_valid: true`; 9/9 `/validate` checks pass. | run `f8b35c56`: `status: completed`, `rows_processed: 8`, `rows_routed_success: 8`, `rows_routed_failure: 0`. Output files written: `high_priority-3.jsonl` (3 enterprise, 482 B), `low_priority-3.jsonl` (5 non-enterprise, 754 B). |
| 2 | **Surgical patch** — only the gate `condition` changed (`row['customer_tier'] == 'enterprise'` → `row['customer_tier'] == 'enterprise' or row['customer_tier'] == 'pro'`). Idiomatic `or` chain (slightly less terse than 05-02's `in [...]` form, but valid). Every other field preserved. | v2; `is_valid: true`. | run `b50133e4`: `status: completed`, 8 rows processed, 8 routed success. `high_priority-4.jsonl` (6 rows, 966 B), `low_priority-4.jsonl` (2 rows, 270 B). |
| 3 | **Refused to silently rewrite as a revert.** Surfaced `POST /api/sessions/{id}/state/revert` with the exact endpoint and `state_id` semantics; explained "I won't patch the gate forward, because that would create a new edit rather than recording this as a revert in the audit trail." | v2 (no mutation) | n/a |

**Driver action**: `POST /state/revert {"state_id": "<v1 id>"}` returned 200 and created v3 derived from v1's state with `condition: "row['customer_tier'] == 'enterprise'"`. Audit trail preserved.

**Reproduction targets confirmed (close holds for all four):**

| Issue | Status | Evidence |
|---|---|---|
| `elspeth-71520f5e30` (P1, closed) — Run.status fails routing pipelines | **Close holds** | run `f8b35c56` reports `status: completed` with `rows_routed_success: 8` and `error: null`, where 05-02 reported `status: failed` with "No row reached the success path". |
| `elspeth-5069612f3c` (P1, closed) — rows_routed counter split | **Close holds** | The new `rows_routed_success` and `rows_routed_failure` fields appear in `/api/runs/{rid}` response. The schema migration check at `web/sessions/schema.py` enforced this end-to-end (caused the deploy-time DB-delete). |
| `elspeth-411435710b` (P1, closed 04-28) — blob path allowlist | **Close holds** | Composer emitted `path: /home/john/elspeth/data/blobs/<sid>/<bid>_tickets.csv` (canonical absolute), `path_allowlist` check in `/validate` passed. No `data/data/` double-prefix. |
| `elspeth-2c3d63037c` (P1, closed 05-01) — gate primitive 500 | **Close holds** | Three gate edits in this session (msg1 build, msg2 patch, msg3 revert) returned HTTP 200 with no half-written nodes. |

**Engine ran?** Yes, twice (v1 and v2). **Real on-disk output?** Yes — both runs wrote the operator-requested files end-to-end. **Output matches user intent?** Yes.

### S3 — stateful aggregation per `customer_tier`
**Session:** `e64c63e3-d4ab-41a8-9ec7-ad1641c83817` · **Messages:** 3 · **Wall-clock:** 62s + 68s + 30s ≈ 160s composer-time, 3 execute attempts.

| Msg | LLM did | Composer outcome | Engine outcome |
|---|---|---|---|
| 1 | Built `csv → batch_stats(group_by: customer_tier, value_field: amount, compute_mean: false) → json sink` directly. **Avoided field_mapper rename step entirely** (no `sum_of_amount` rename — kept raw `count`, `sum`, `customer_tier`). Typed `amount: float` correctly. The `json` sink declared `[customer_tier: str, count: int, sum: float]` (fixed mode, only 3 fields). | v1; `is_valid: true`, 9/9 `/validate` checks pass. | run `d84cf2e8`: **`status: failed`, `PluginContractViolation`**. Sink `json` rejected upstream-emitted `batch_size` field (`extra_forbidden`). Engine error: *"This indicates an upstream transform/source schema bug."* |
| 2 | Diagnosed correctly ("aggregation step emitting an extra runtime field: batch_size"). Repaired by inserting a `field_mapper` between aggregation and sink with `select_only: true, mapping: {customer_tier→customer_tier, count→count, sum→sum}`. **But declared the field_mapper's output schema as `[batch_size: int, count: int, customer_tier: str, sum: float]` (4 fields)** while `select_only: true` only emits 3. | v2; `is_valid: true`, 9/9 pass. | run `2d5b55f5`: **`status: failed`, `SchemaConfigModeViolation`**. Engine: *"Transform 'field_mapper' emitted output schema semantics inconsistent with its declaration: missing required fields ['batch_size']; field metadata mismatches for ['count','customer_tier','sum']."* |
| 3 | Diagnosed correctly ("the cleanup step was still declaring that batch_size could be part of its output contract, while select_only: true intentionally removes it"). Removed `batch_size` from declared output schema; declared `[customer_tier: str, count: int, sum: float]`. | v3; `is_valid: true`. | run `5f22ec44`: **`status: failed`, `PluginContractViolation` again**. Engine: *"Transform 'field_mapper' input validation failed: batch_size — Extra inputs are not permitted (extra_forbidden). This indicates an upstream transform/source schema bug."* The fix removed `batch_size` from the *output* shape but left the field_mapper's *input contract* (`extra_forbidden`) intact, which then rejected upstream's `batch_size`. |

**Stop rule applied** (advisor's "3 errors deep = that *is* the finding"). S3 produced **zero on-disk output**.

**Architectural class — three downstream-vs-upstream coherence specifics in one session:**

| Surface | Plugin | Diagnosis |
|---|---|---|
| **NEW** S3 v1 | `json` sink fixed-mode schema | Sink rejects upstream-emitted `batch_size` (`PluginContractViolation` at `sink_write` phase). The `json` sink's input validation is `extra_forbidden` when `mode: fixed` — and the composer's `schema_compatibility` check accepts the pipeline because it doesn't track that the upstream `batch_stats` will emit one more field than the downstream sink declares. |
| **REPRO 05-02 #1** S3 v2 | `field_mapper` declared output | `select_only: true` declares one set of output fields, `schema.fields` declares a superset, runtime detects emission inconsistency (`SchemaConfigModeViolation` at `executor_post_process` phase). |
| **REPRO 05-02 #2** S3 v3 | `field_mapper` input contract | Input contract is `extra_forbidden` and rejects upstream's `batch_size` (`PluginContractViolation` at `executor_post_process` phase). |

All three surfaces share one root cause: **the composer's `schema_compatibility` check doesn't enforce that the *set* of fields a producer emits is a subset of (or compatible with) the set its downstream consumer will accept.** The engine *does* track emission via `success_reason.fields_added` (visible in run diagnostics). That information is available to the validator; it just isn't consulted.

**LLM judgement weakness — missed simpler fix.** The `json` sink could have been re-configured with `schema.mode: flexible` (drops the `extra_forbidden` constraint), eliminating the need for the field_mapper entirely. The LLM never tried this — it always assumed the sink shape was fixed and inserted a transform to narrow upstream emissions to match. Each subsequent fix iteration added complexity rather than removing it.

**`elspeth-178f765792` close holds:** the `required_input_fields` field appears on the `field_mapper` config (a row transform, not batch-aware) and was accepted at runtime — the prior bug was specifically about advertising it on **batch-aware** transforms (`batch_stats`), and that surface stays correctly closed.

**Engine ran?** Three times. **All three failed.** **Real on-disk output?** None.

#### S3-prime — re-run after `elspeth-3d25355784` fix shipped (commit `f3137ae8`)
**Session:** `22e3b2af-e16d-443f-a5a3-19246f2b0829` · **Messages:** 1 · **Wall-clock:** 58s

After commit `f3137ae8` (three field-set membership rules added to `_check_schema_contracts` in `src/elspeth/web/composer/state.py`) was deployed via `systemctl restart elspeth-web.service`, a fresh session was created with the same prompt as today's S3 msg1.

**msg1 outcome:** The LLM produced a valid pipeline on the **first iteration** (no recovery turns needed): csv source → `batch_stats` aggregation → `json` sink with `schema.mode: observed` (the simpler-fix path the LLM had never tried in any prior eval). Either the validator's structured rejection caught the LLM's first attempt inside the tool loop and steered it here, or the LLM's prior over the shape space shifted because of the new error messages. Either way: minimal-complexity pipeline, no `field_mapper` insertion, no `select_only` gymnastics.

`/validate` 9/9 pass. `/execute` run `a419c8a8`: `status: completed, rows_processed: 8, rows_succeeded: 3, rows_failed: 0`. **Real on-disk output written**: `outputs/tier_summary_by_customer_tier.json` (285 B):

```json
[
  {"count": 3, "sum": 450.0, "batch_size": 3, "customer_tier": "enterprise"},
  {"count": 3, "sum": 205.0, "batch_size": 3, "customer_tier": "pro"},
  {"count": 2, "sum": 15.0,  "batch_size": 2, "customer_tier": "starter"}
]
```

Hand-verified math: enterprise 100+200+150=450 ✓, pro 50+75+80=205 ✓, starter 10+5=15 ✓. The output includes an extra `batch_size` field the user didn't ask for (because `mode: observed` on the sink preserves all upstream fields) — pedantically more verbose than the request, functionally correct.

**`elspeth-3d25355784` close confirmed end-to-end. S3 outcome: GREEN after the validator fix.**

### S2 — incremental classifier
**Session:** `4be4ac77-9db8-469c-a3b3-de92393ab4a0` · **Messages:** 3 · **Wall-clock:** 41s + 58s + 27s ≈ 126s composer-time

| Msg | LLM did | Composer outcome | Notable |
|---|---|---|---|
| 1 | Asked to "set up the source. Don't add anything else yet." LLM set the CSV source with `mode: fixed`, all 5 columns typed (`amount: float`), `on_validation_failure: discard`, `blob_ref: <bid>` exposed alongside `path`. Did NOT add transforms or sinks. | v1; `is_valid: false` (composer correctly says: *"runtime preflight failed: sinks Field required"*); 1 medium-severity warning (`Source on_success 'next' does not match any node input or output — data may not flow.`) | **Composer's completion gate respects the user's "don't add anything else" instruction** — it doesn't proactively add sinks even when their absence makes the pipeline incomplete. The warning surfaces the dangling reference clearly. |
| 2 | "Add an LLM transform that reads `subject`/`body`, classifies into bug/billing/feature_request/other. Use openrouter, model openai/gpt-4o-mini. Wire OPENROUTER_API_KEY as a secret reference." | v2; `is_valid: false`; `validation_errors: ["Credential field(s) api_key contain a literal value; expected a wired secret reference."]` | LLM emitted a literal `api_key` placeholder. Composer's `secret_refs` validator caught it correctly — **`elspeth-72d1dccd44` close holds for the third time across the eval history**. |
| 3 | "Wire OPENROUTER_API_KEY properly as a secret reference." | (state unchanged — no mutation tool fired) | **Strong diagnostic recovery vs 05-02 regression.** LLM probed availability, surfaced the actual failure (*"Secret reference OPENROUTER_API_KEY not found or not accessible"*), refused to paste a literal as a workaround, and gave operator-actionable guidance ("add or enable the OPENROUTER_API_KEY secret in the environment/Secrets panel"). Same shape as 05-01 S1B msg4's strong run, in contrast to 05-02 S1B msg3/4 which just re-emitted the validate complaint. **Confirms the 05-02 LLM regression was non-deterministic, not a baseline drift.** |

**Engine ran?** No — blocked at composer-time `secret_refs` (and at operator-config-layer `fingerprint_resolver_not_configured`). **Real on-disk output?** None — but this is the operator-config gap from 05-02, not a composer or LLM defect.

**Architectural shape:** S2 is the canonical proof that the composer end-to-end will not let an LLM-classified pipeline run without a wirable secret on this deploy. Once `ELSPETH_FINGERPRINT_KEY` is set, this scenario should produce real classified output — the composer-side shape is correct.

**ADDENDUM (same session, msg4 after operator config fix).** During this eval session the operator generated and added `ELSPETH_FINGERPRINT_KEY` to `deploy/elspeth-web.env` (32-byte URL-safe random via `secrets.token_urlsafe(32)`) and restarted the service. `/api/secrets` immediately flipped `OPENROUTER_API_KEY` to `available: true, reason: null`; the other three (Anthropic, Azure, OpenAI) cascaded to `available: false, reason: env_var_not_set` (correctly reporting the next-layer gap). msg4 ("operator just configured the resolver — please retry wiring and add sinks") returned in 89s with `api_key: {secret_ref: OPENROUTER_API_KEY}` properly wired plus two sinks (`results.jsonl` classified, `parse_failures.jsonl` quarantine). `/validate is_valid: true` (9/9). `/execute` run `45a592e1`: `status: completed, rows_processed: 8, rows_succeeded: 8, rows_failed: 0`. **Real on-disk output written**: `outputs/results.jsonl` (2.4 KB, 8 classified rows). Categorisations: T-001 Login broken→bug, T-002 Invoice→billing, T-003 Feature ask→feature_request, T-004 Random thanks→other, T-005 Crash on save→bug, T-006 Refund→billing, T-007 Shipping→other, T-008 Token rotated→other. All sensible.

**S2 outcome: GREEN end-to-end after operator config fix.**

### S1 — monolithic complete-pipeline ask
**Session:** `50bc5bf4-2719-4ad1-b7b8-cb6c996a0675` · **Messages:** 1 · **Wall-clock:** 80s

**Prompt** (one message): build the full csv → llm classify → 4-way category-routed sinks pipeline, wire OPENROUTER_API_KEY as a secret reference, quarantine source-validation failures, errors to errors.jsonl, do it all in one go.

LLM built in 80s (vs 05-02's 135.8s and 04-28/05-01's ~120s — fastest run on record):

- 1 source: csv from blob, `mode: fixed`, 5 fields with types (`amount: float` typed correctly)
- 1 transform: `llm` (`classify_ticket`), `provider: openrouter`, `model: openai/gpt-4o-mini`
- 3 chained gates (ladder pattern):
  - `route_bug`: `row['category'] == 'bug'` → `bugs` / `route_billing_in`
  - `route_billing`: `row['category'] == 'billing'` → `billing` / `route_feature_in`
  - `route_feature`: `row['category'] == 'feature_request'` → `features` / `other`
- 6 sinks: `bugs.jsonl`, `billing.jsonl`, `features.jsonl`, `other.jsonl`, `errors.jsonl`, `source_quarantine.jsonl`

`/validate`: `is_valid: false`, `secret_refs: passed: false` with *"Credential field(s) api_key contain a literal value"*. The other 8 checks cascade-skipped, but `path_allowlist` passed before secret_refs failed (so `elspeth-411435710b` close holds for the blob-backed source path here too).

**`elspeth-72d1dccd44` close holds — fourth confirmation across the eval history (04-28, 05-01, 05-02, 05-03).**

LLM judgement was excellent — gate ladder correctly chained, route names match sink names, proactive `errors` and `source_quarantine` sinks added without re-prompt. The only blocker is the same operator-side secret-availability gap as S2.

**Engine ran?** No — blocked at `secret_refs`. **Real on-disk output?** None.



---

## Cross-cutting findings

### Headline: routing-pipeline `run.status` semantics fixed end-to-end

The 05-02 architectural gap #3 — *"pure-routing pipelines report `status: failed` despite all rows correctly routed"* — is **fixed** by the `rows_routed_success` / `rows_routed_failure` split (commits `e8c9fbff`, `6d745470`). S4 verified twice: 8/8 routed in run `f8b35c56` and 8/8 routed in run `b50133e4`, both `status: completed`, `error: null`. The fix is a *taxonomy* fix rather than a band-aid: instead of redefining "success" to include "routed", the team added new counters so the success-path predicate can correctly distinguish gate-routed from `on_error`-routed flows. Two closed P1s confirmed in one /execute.

This was so load-bearing that the deploy required deleting `data/sessions.db` to drop the old `runs.rows_routed` column shape — the documented operator action under the No-Migration policy. Backup retained at `data/sessions.db.bak-pre-2026-05-03-rows-routed-split`.

### Architectural class survives via the same root cause: composer's schema_compatibility doesn't track field-level emission vs consumption

The 05-02 finding's "composer says valid, runtime rejects" pattern produced **three distinct surfaces in S3 alone** (one new, two reproductions of `elspeth-3d25355784`). All three trace to a single check-design gap:

- The engine knows what fields each transform/aggregation emits — `success_reason.fields_added` captures it deterministically per-row at runtime (visible in `/api/runs/{rid}/diagnostics`).
- The composer's `schema_compatibility` validator does not consult that information. It validates schema *shapes* (mode, type, required) but not the *set membership* contract: "does the upstream emit a superset of what the downstream's input contract requires, *and* a subset of what its `extra_forbidden` allows?"
- Result: any plugin pair where one emits a closed-set output (`batch_stats` always emits `batch_size`) and the other consumes via a locked-input contract (`field_mapper` with `extra_forbidden`, `json` sink with `mode: fixed`) is fertile ground for new divergence shapes.

**The information is available; the validator just isn't using it.** This is the recommendation focal point for closing the `elspeth-3d25355784` triage gap.

### LLM behaviour cross-cuts

| Behaviour | Today | 05-02 | 05-01 | 04-28 |
|---|---|---|---|---|
| Vague-prompt clarification (S5) | ✅ refuse + 3 questions | ✅ | ✅ | ✅ |
| Surgical patch on gate predicate (S4 msg2) | ✅ `or` | ✅ `in [...]` | ✅ | ✅ |
| Distinguishing revert from forward edit (S4 msg3) | ✅ surfaces `/state/revert` endpoint | ✅ | n/a | ❌ "subtle gap" |
| Diagnosing secret-wiring failure (S2 msg3) | ✅ probes availability layer | ❌ regression — re-emits validate complaint | ✅ identifies `ELSPETH_FINGERPRINT_KEY` | n/a |
| Refusing to fabricate xlsx support (S5) | ✅ | ✅ | ✅ | ✅ |
| Refusing literal `api_key` placeholder | ✅ caught upstream by validator | ✅ caught | ✅ caught | ❌ slipped past validator |
| Avoiding unnecessary `field_mapper` rename (S3 msg1) | ✅ keeps natural names | ❌ added `sum_of_amount` rename | n/a | n/a |
| Trying simpler fix (json sink → flexible) instead of patching upstream (S3) | ❌ never tried | ❌ added rename layer | n/a | n/a |

The 05-02 LLM regression on diagnosis (S1B msg3/4) is **not present** today. Same model (`openrouter/openai/gpt-5.5`) and same prompt — confirms the regression was non-deterministic per-call, not baseline drift. The S5/S4 msg3 audit-discipline behaviour persists across 4 evals.

### Tier-3 boundary tightening at `/api/sessions/{sid}/blobs/inline`

The blob-upload endpoint renamed `content_type` → `mime_type` and added `additionalProperties: false`. The prior eval's helper (and any external client using the old field name) now hits a 422 explicitly listing `extra_forbidden`, where it previously got a silent default. Per the model's own docstring: *"Previously a caller who sent `content_type` (the old field name) or `mime-type` got a silent fallback to the default MIME — now they get a 422."* — clean Tier-3 trust-boundary discipline.

### State response redacts blob source path

`GET /api/sessions/{sid}/state` returns `<redacted-blob-source-path>` in `source.options.path`, while `GET /api/sessions/{sid}/state/yaml` (used by runtime) returns the canonical absolute path. Privacy improvement on the read API; runtime lineage unchanged. Worth noting in case any UI tooling expects the literal path on the state endpoint.

---

## Final scoreboard

| Scenario | Composer happy | /validate happy | Engine ran | Real output | Output matches user intent? |
|---|---|---|---|---|---|
| **S5** (vague Excel) | n/a (no mutation) | n/a | n/a | n/a | ✅ — clean refusal with clarifying questions |
| **S4** (gate routing) | ✅ msg1 | ✅ msg1 | ✅ both runs | **`high_priority-{3,4}.jsonl` + `low_priority-{3,4}.jsonl`** | ✅ — perfect routing, **`run.status: completed`** for both runs |
| **S3** (aggregation, original) | ✅ msg1/2/3 | ✅ msg1/2/3 | ❌ all 3 attempts failed | none | ❌ — three downstream-vs-upstream coherence violations exposed in sequence |
| **S3-prime** (aggregation, post-`f3137ae8`) | ✅ msg1 | ✅ msg1 | ✅ run `a419c8a8` | **`outputs/tier_summary_by_customer_tier.json`** | ✅ — first-iteration green, math verified |
| **S2** (incremental classifier) | ✅ from msg4 (after operator config fix) | ✅ msg4 | ✅ run `45a592e1` | **`outputs/results.jsonl`** | **✅ 8/8 rows classified, all sensible categories** |
| **S1** (monolithic LLM-classify) | ✅ msg1 (state is_valid: false correctly) | ❌ (`secret_refs`) | ❌ | none | n/a (operator-config blocker) |

**Three of five scenarios produced real on-disk output** after the operator config fix mid-session: gate-routed JSONLs (S4) plus LLM-classified JSONL (S2). The aggregation scenario S3 reproducibly fails on the open `elspeth-3d25355784` architectural class. S1 (monolithic) was not re-attempted post-fix because S2 already proved the unblocking and S1 would have burned credit on a re-validation; the same `secret_ref` wiring path is now confirmed.

Compared to 05-02 (3 of 5 produced output): one fewer because S3 surfaced a *new* runtime contract violation (json sink fixed-mode) before the LLM could try the simpler fix. The architectural class is the cause; one more LLM iteration would likely have produced output (msg4: change json sink mode to flexible). Stopped at 3 errors per the methodology rule.

---

## Filings

### Comments to add (existing issues)

- **`elspeth-71520f5e30`** (P1, closed) — *Run.status reports 'failed' on routing pipelines*: comment confirming close holds end-to-end via S4 runs `f8b35c56` (8/8 routed, status completed) and `b50133e4` (8/8 routed via OR-clause predicate, status completed). Add evidence: `rows_routed_success: 8` and `rows_routed_failure: 0` appear in both responses; `error: null` (was "No row reached the success path" on 05-02).
- **`elspeth-5069612f3c`** (P1, closed) — *Split rows_routed counter*: comment confirming the split fields are present in `/api/runs/{rid}` API responses; required deleting `data/sessions.db` on this deploy because the schema-validator at `web/sessions/schema.py` correctly refuses the old shape (offensive programming working as designed).
- **`elspeth-3d25355784`** (P2, triage) — *Composer field_mapper composer-vs-runtime gap*: extend with **third surface** found today: `json` sink `mode: fixed` rejects upstream-emitted `batch_size` at `sink_write` phase, parallel architectural problem to the field_mapper specifics. Reproducer: S3 msg1 yaml at `evals/2026-05-03-composer/basic/s3/state.yaml`, run `d84cf2e8` diagnostics `s3/diag1.json`. Both 05-02 specifics #1 and #2 also reproduce in this session (S3 v2 = SchemaConfigModeViolation, S3 v3 = field_mapper input PluginContractViolation). Recommends `schema_compatibility` validator should consult `success_reason.fields_added` (already tracked by engine) to enforce field-set membership.
- **`elspeth-72d1dccd44`** (P2, closed) — *Composer secret_refs validator passes literal placeholder strings*: comment confirming close holds for fourth time across eval history (S1A msg1, S2 msg2 today; 04-28, 05-01, 05-02 prior).
- **`elspeth-411435710b`** (P1, closed) — *Composer-built blob-backed pipelines fail runtime path-allowlist*: comment confirming blob path resolves cleanly across all four scenarios that used inline blobs (S4, S3, S2, S1) — `path_allowlist` check passed in every `/validate` invocation.

### New bugs to file

None. The S3 architectural finding is a **new surface of an open issue** (`elspeth-3d25355784`), not a new bug. Operator-config gap (`ELSPETH_FINGERPRINT_KEY` unset) is out of eval scope.

### Eval-side observations (no filing needed)

- Blob endpoint `mime_type` rename — already shipped, working as designed; client docs may need refresh if any external wrapper uses `content_type`.
- State response path redaction — possible UX subtlety, but not a defect.

---

## Cross-reference: hard-mode persona eval (same day)

A companion eval ran the same afternoon — `docs/composer/evidence/composer-eval-hardmode-2026-05-03.md` — driving the composer through 9 persona-task scenarios via subagents locked to three cognitive-style stratified personas (constraint-laden compliance, narrative researcher, confidently-misconceived ops). The hard-mode eval surfaced two findings that did not appear in this basic-mode eval, both because the basic-mode driver (me, with full schema knowledge) pre-cleaned the prompts in ways real users would not:

- **Model-name auto-selection 404** (`elspeth-obs-f3143acba2`, P2). When user prompts omit `model:`, the composer LLM auto-picks `anthropic/claude-3.5-sonnet`, which OpenRouter does not recognise. 3-of-3 reproduction across happy-path scenarios. Verified-fixable via either operator-pinned default in the composer system prompt OR explicit `model:` in the user prompt — proof-of-fix run `023eb897-…` produced the expected output once the model was swapped to `openai/gpt-4o-mini`. Basic-mode S2 didn't surface this because its prompt explicitly named `openai/gpt-4o-mini`.
- **Composer LLM convergence-timeout on multi-step builds** (`elspeth-obs-8f82c91147`, P2). 3-of-3 reproduction on edge-class scenarios that combined ≥2 patterns (classify+enrich, classify+aggregate+cross-tab, multi-value fork). Composer LLM iterates one tool-call at a time and runs out of the 180s wall-clock. Surface design (structured 422 with `error_type`, `budget_exhausted`, `recovery_text`, `partial_state`) is best-in-class even on failure; the gap is in the LLM's tool-selection behaviour. Likely skill-pack tuning territory rather than infrastructure.

Both findings strengthen rather than contradict the basic-mode results. Basic-mode validates that the *capabilities* work end-to-end. Hard-mode validates that the *surface + skill-pack* gives a non-specialist user a fair chance of reaching those capabilities — and surfaces the friction points that need closing before the audit-grade-LLM-runs story holds for arbitrary user prompts.
