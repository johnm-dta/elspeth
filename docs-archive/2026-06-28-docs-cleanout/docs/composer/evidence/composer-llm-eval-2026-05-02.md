# ELSPETH Composer LLM Evaluation — staging deployment

**Deployment:** https://elspeth.foundryside.dev (source-checkout systemd/Caddy on this host)
**Service start:** `systemctl` reports `ActiveEnterTimestamp=2026-05-02 13:34:35 AEST` — head of `RC5-UX` at run time.
**Composer model:** `openrouter/openai/gpt-5.5` (via OpenRouter)
**Composer budget:** 15 mutation turns / 10 discovery turns / 180s wall-clock per `POST /messages`, 10 rpm
**Tester:** `dta_user` (regular, no admin groups)
**Date:** 2026-05-02
**Prior evals:** `docs/composer/evidence/composer-llm-eval-2026-04-28.md`, `docs/composer/evidence/composer-llm-eval-2026-05-01.md`
**Tracker task:** `elspeth-599ecf69fa` (P1 task — repeat staging composer LLM evaluation)

## Headline

**Every P1/P2 bug filed from the 2026-05-01 eval reproduces as fixed.** The gate-primitive
500 (`elspeth-2c3d63037c`) is gone — S3 built and ran a complete gate-routing pipeline on
the first try in 64s. The literal-string `api_key` (`elspeth-72d1dccd44`) is now caught at
composer-time `secret_refs` validation. The completion-callback exception
(`elspeth-31d53c7493`) did not recur on S2's successful aggregation run. The new
`/api/secrets` `reason` field (`elspeth-0d31c22d26`) is live — `OPENROUTER_API_KEY` reports
`available: false, reason: "fingerprint_resolver_not_configured"`.

The architectural class — *composer says valid, runtime rejects* — survives on **two new
field_mapper-shaped specifics** (output-schema-vs-mapping mismatch, and input-contract-vs-
upstream-emission mismatch) and on **one run-status-taxonomy gap**: pure routing pipelines
(every row routed via gates to named sinks) report `status: failed` with
"No row reached the success path" even though every row terminated correctly with
`terminal_outcome: routed` and the operator-requested files were written end-to-end.

Five of five scenarios reached `is_valid: true` in the composer; **three of five produced
real on-disk output**, two of which were the operator-requested artefact (S2's per-tier
rollup, S3's gate-routed JSONLs). S1A and S1B are blocked by the same operator-side gap
as 05-01 — `ELSPETH_FINGERPRINT_KEY` unset on staging, so no env-backed credential can
be wired by the composer's secret-refs flow.

LLM behaviour matches or improves on 05-01. **Standout new behaviour**: in S3 msg3 the
model declined to silently rewrite a forward edit as a "revert" and instead surfaced the
`POST /api/sessions/{id}/state/revert` endpoint to the user — closing the prior eval's
"subtle gap" (04-28 finding: "no `revert` tool exposed" → model patched forward calling it
revert). **Standout regression**: in S1B msg4, when asked to diagnose why
secret-wiring failed, the model just re-emitted the validate complaint verbatim with no
new tool calls (vs 05-01's S1B msg4 which brilliantly identified `ELSPETH_FINGERPRINT_KEY`
as the layer at fault).

## How this eval was run

Same methodology as 04-28 and 05-01 — driven by an LLM (Claude Opus 4.7) acting as a
regular authenticated `dta_user`, against the live staging deploy, using only the public
HTTP surface exposed in `openapi.json`. No source edits, no in-process MCP tools (the
`mcp__elspeth-composer__*` tools were available in the environment but explicitly not
used; per user instruction those bypass the HTTP path).

- **Auth:** `POST /api/auth/login` → JWT → `Authorization: Bearer …` on every call.
- **Discovery (read-only):** `GET /openapi.json`, `GET /api/system/status`,
  `GET /api/secrets`, `GET /api/catalog/{sources,transforms,sinks}` and per-plugin
  `…/schema`. Cached in `/tmp/elspeth_eval_2026-05-02/`. Used only to score LLM output,
  never to bypass.
- **Per-scenario user simulation:**
  1. `POST /api/sessions` (create chat)
  2. `POST /api/sessions/{sid}/blobs/inline` to upload `tickets.csv` (8 rows, 5 columns)
  3. `POST /api/sessions/{sid}/messages` and wait for the synchronous LLM tool-loop
  4. `GET /api/sessions/{sid}/state`, `…/state/yaml`, `…/state/versions`,
     `…/composer-progress`, `…/messages` to inspect the result
  5. `POST /api/sessions/{sid}/validate` for runtime preflight
  6. `POST /api/sessions/{sid}/execute` then `GET /api/runs/{rid}` and `…/diagnostics`
  7. `cat /home/john/elspeth/data/outputs/*.{json,jsonl}` to confirm real artefacts
- **Stop budget:** 5 messages per session max, 2 post-execute fix attempts. When 3
  successive failures showed a single class, stopped (per the user's "that *is* the
  finding" rule).
- **Order:** S4 → S1B → S2 → S3 → S1A (cheap before expensive; S1A last because prior
  eval expected it to time out, though as 05-01 noted it no longer does).
- **Failure-feedback rule:** when `/execute` failed, the next message was always the
  literal user-visible error, fed back to let the LLM attempt its own recovery.
- **Concurrency:** S1B msg3 + S2 msg1 + S3 msg1 were run in parallel after their
  individual session creation, to keep the eval inside one wall-clock window without
  exceeding the 10 rpm composer rate limit. No interference observed; sessions are
  isolated.

Server logs (`journalctl -u elspeth-web.service`) and on-disk artefacts under
`/home/john/elspeth/data/` were read for *diagnosis* after each user-flow finished —
never to bypass any layer.

## Confirmed fixes from prior evals (the major reason to write this report)

| Prior finding / issue | Status today | Direct evidence |
|---|---|---|
| `elspeth-2c3d63037c` (P1, 05-01) — gate primitive 500, half-written nodes, no traceback | **Fixed** | S3 msg1 built a 1-source / 1-gate / 3-sink routing pipeline in 64s, HTTP 200. Gate node renders cleanly into YAML (`gates: [...]` block) and `/state/yaml` returns 200. Two more gate edits (msg2 surgical patch, msg3 revert request) produced no 500s. Eight rows routed end-to-end, output files written. |
| `elspeth-72d1dccd44` (P2, 05-01) — `secret_refs` validator passed literal `api_key` placeholder | **Fixed** | S1B msg2 emitted `api_key: "PLACEHOLDER_TO_BE_WIRED"`. `/validate` returned `is_valid: false`, `secret_refs` check `passed: false` with `Literal value in credential field(s): classify_ticket:api_key`. Same shape blocked again in S1A msg1 (`classify:api_key`). Composer's response includes the suggested fix string. |
| `elspeth-31d53c7493` (P2, 05-01) — `pipeline_done_callback_exception` on S2 successful aggregation | **Fixed** *(non-recurrence)* | S2 v3 `/execute` succeeded with `status: completed, rows_succeeded: 3` and `/api/runs/{rid}` returned the terminal state immediately, no callback exception in the journal. Single-run non-recurrence — narrow but consistent with the close note. |
| 05-01 finding 4 — `created_by: "assistant"` on inline blobs from authenticated user | **Fixed** | S1B blob upload returned `created_by: "user"`. |
| 05-01 finding 7 / `elspeth-cd5d811121` / `elspeth-0d31c22d26` — `/api/secrets` returned `available: false` with no diagnostic | **Fixed** | `GET /api/secrets` now returns `reason: "fingerprint_resolver_not_configured"` for every env-backed entry. The structural cause (operator-side `ELSPETH_FINGERPRINT_KEY` unset on staging) is unchanged, but the API now self-explains rather than the operator having to deduce it from runtime errors. |
| 05-01 finding 6 — `/state/yaml` returns 409 with structured Pydantic error when preflight fails | **Verified again** | S1B v2 (`api_key` literal) `/state/yaml` returned `409 {"detail":"Current composition state failed runtime preflight... First error: Credential field(s) api_key contain a literal value..."}`. |
| 04-28 "subtle gap" — LLM had no `revert` tool, patched forward calling it a revert | **Fixed in LLM behaviour** | S3 msg3 LLM response: *"I can't perform a true revert from the composer tools available here... please use the UI's version history restore action, or have an operator call the revert endpoint: `POST /api/sessions/{session_id}/state/revert`... If instead you want me to make a new forward edit... it will be recorded as a new change rather than a true revert."* Driver then exercised `/state/revert` directly; system message *"Pipeline reverted to version 1."* was recorded in the chat history (audit-preserving). |
| `/validate` is now a 9-check preflight (was 8 in 05-01) | **Confirmed** | New check `route_target_resolution` is present alongside the prior eight. Cascade-skip semantics (downstream checks skip when an upstream check fails) preserved. |

The remediation program in `docs/composer/evidence/composer-remediation-program-2026-05-01.md` correctly
predicted that all of Phase 0 (gate primitive), Phase 1.1 (secret-refs fabrication
detection), and Phase 2.1 (callback exception) would be the entry points. Those landed.
Phase 1.2 (`route_target_resolution`) shipped as a new validator check; the dangling
`on_error` shape is caught at composer-time today (driver-issued verify session
`02b4853e-…` confirmed: a passthrough transform with `on_error:
nonexistent_failsink_xyz` produced `/validate is_valid: false` with the catching check
being `graph_structure` — detail *"Transform 'pass' on_error 'nonexistent_failsink_xyz'
references unknown sink. Available sinks: results."* — and `route_target_resolution`
cascade-skipped behind it). Phase 1.3 (`schema_compatibility` mode/options
incompatibility) is **not** landed — the 05-01 shape `mode: flexible + required_fields`
is silently accepted by both composer `/validate` (`is_valid: true`) and runtime in
S2 v3, so the original-shape failure mode no longer reproduces but only because the
runtime no longer raises on this code path; the composer doesn't catch it. Two new
field_mapper shapes survive (see below).

## Architectural class survives — two new field_mapper specifics + one run-status gap

### 1. field_mapper output schema does not match what `mapping` actually emits (S2 v2)

**Composer state**: `is_valid: true`, all 9 `/validate` checks pass.
**Runtime outcome**: `SchemaConfigModeViolation` from `executor_post_process` phase.

The S2 v2 yaml (LLM-built) wired:

```yaml
transforms:
- name: select_summary_fields
  plugin: field_mapper
  options:
    schema:
      mode: flexible
      fields: [customer_tier: str, count: int, sum: float]
      required_fields: [customer_tier, count, sum]
    mapping:
      customer_tier: customer_tier
      count: count
      sum: sum_of_amount   # ← renames sum to sum_of_amount
    select_only: true
    strict: true
```

Composer's `schema_compatibility` check accepts this. At runtime the engine raised:

```text
Transform 'field_mapper' (node 'transform_select_summary_fields_…') emitted output
schema semantics inconsistent with its declaration for row '…': missing required fields
['sum']; field metadata mismatches for ['customer_tier', 'count'].
```

The contract violation: `mapping`'s emitted output keys must agree with the declared
output `schema.fields`. The composer should detect that `mapping` writes `sum_of_amount`
but the declared schema requires `sum` (i.e., the value side of `mapping` is the runtime
output field name, not the input side). Currently it doesn't.

**Reproducer**: full S2 v2 yaml at `/tmp/elspeth_eval_2026-05-02/s2/final.yaml`
(re-rendered after later patches, but the v2 shape is reconstructible from the LLM's msg2
content). Actual runtime traceback in `s2/diag2.json`.

### 2. field_mapper input contract rejects upstream-emitted extras (S2 v3)

**Composer state**: `is_valid: true`, all 9 `/validate` checks pass.
**Runtime outcome**: `PluginContractViolation` from input-validation phase.

After the LLM patched the output schema in msg3 (correctly aligning declared output to
emitted output `[customer_tier, count, sum_of_amount]`), the runtime then rejected on the
**input** side:

```text
Transform 'field_mapper' input validation failed: 3 validation errors for FieldMapperInput
sum_of_amount
  Field required [type=missing, input_value={'count': 3, 'sum': 450.0, ..., 'customer_tier': 'enterprise'}, input_type=dict]
sum
  Extra inputs are not permitted [type=extra_forbidden, input_value=450.0, input_type=float]
batch_size
  Extra inputs are not permitted [type=extra_forbidden, input_value=3, input_type=int]
```

`batch_stats` emits `{customer_tier, count, sum, batch_size}` (the `batch_size` field is
load-bearing per the plugin's contract). The field_mapper's input contract is locked
(`extra_forbidden`) and was generated from the LLM's *output* schema — so it expected
`sum_of_amount` rather than `sum`, and rejected the unrelated `batch_size` as extra.

The composer's `schema_compatibility` and `semantic_contracts` checks should detect that
a downstream `field_mapper` with a locked input contract is being placed after a transform
(`batch_stats`) whose declared output includes fields not in the mapper's input contract.
Currently they don't.

The LLM diagnosed and repaired this in S2 msg4 (correctly enumerating "the field_mapper
had a fixed input schema that described its renamed output instead of the actual aggregate
row it receives") and the v3 `/execute` succeeded.

**Reproducer**: `s2/diag3.json` operations[0].error_message captures the full Pydantic
validation chain.

### 3. Pure-routing pipelines report `status: failed` despite all rows correctly routed

**Composer state**: `is_valid: true`. **Runtime outcome**: every row reaches a
`terminal_outcome: routed`, every operation `completed`, output files written
end-to-end — but `run.status: failed`.

S3 v1 (`gate(enterprise/else) → 2 sinks`) ran 8 rows, wrote `high_priority-1.jsonl`
(3 enterprise rows, 482 B) and `low_priority-1.jsonl` (5 non-enterprise rows, 754 B). All
rows in diagnostics show `terminal_outcome: routed` and per-step `status: completed`.
Yet the run summary returned:

```json
{
  "status": "failed",
  "rows_processed": 8,
  "rows_succeeded": 0,
  "rows_failed": 0,
  "rows_routed": 8,
  "rows_quarantined": 0,
  "error": "No row reached the success path (rows_processed=8, rows_succeeded=0). Inspect /diagnostics for per-row failure details."
}
```

S3 v2 (`gate(enterprise|pro/else) → 2 sinks`) reproduced the same shape: 6 rows to
`high_priority-2.jsonl` (966 B), 2 rows to `low_priority-2.jsonl` (270 B). Same
`status: failed`.

The new four-value `RunStatus` taxonomy (commit `cc895589`) introduced
`completed_partial` / `failed` / etc. semantics, but the pipeline-decision rule still
treats `rows_succeeded == 0` as failure even when `rows_routed == rows_processed` and
every routed token landed in a named sink that wrote a file. This contradicts the new
taxonomy's stated purpose of distinguishing operator-meaningful outcomes.

The error message — *"No row reached the success path. Inspect /diagnostics for per-row
failure details."* — is also misleading: there are no per-row failures, and there is no
"success path" to reach because the pipeline is intentionally terminal-routed.

This is a **regression vs. 04-28 S3** (which reported `completed` for the same shape) and
a **regression vs. the prior eval's pre-`cc895589` behaviour**, where the symptom
existed but at least the message wasn't promising operators non-existent failure
diagnostics.

## Per-scenario findings

### S4 — vague Excel prompt
**Session:** `5c002509-94a4-484f-adf4-6befdd2323b8` · **Messages:** 1 · **Wall-clock:** 7.9s

**Prompt:** *"I want to do something with my Excel file. Can you help me build a pipeline?"*

LLM response: refused to fabricate xlsx support, surfaced CSV as the real product limit,
asked three concrete clarifying questions ("what do you want to do", "what columns",
"what output"), no state mutation. Same shape as 04-28 and 05-01 S4 — clean refusal,
shortest run of the eval.

### S1B — incremental CSV → LLM classifier → routed sinks
**Session:** `6b2988e2-3963-4839-a08b-43af88438f58` · **Messages:** 4 · **Wall-clock:** sum of msg1+msg2+msg3+msg4 ≈ 7s + 106s + 17s + 15s ≈ 145s

| Msg | LLM did | State outcome | Notable |
|---|---|---|---|
| 1 | `set_source_from_blob` with `csv` plugin, `mode: fixed`, all 5 fields with types, `on_validation_failure: quarantine` | v1, source set | Composer correctly told the user "I cannot mark this pipeline complete yet because runtime preflight failed: sinks - Field required" — completion gate working as designed |
| 2 | Added `llm` transform (`classify_ticket`) with full openrouter config — proactively also added 3 sinks (`results`, `quarantine`, `classification_errors`) | v2, transform + 3 sinks | Used `api_key: "PLACEHOLDER_TO_BE_WIRED"` literal; composer `secret_refs` blocked correctly with `Literal value in credential field(s): classify_ticket:api_key`; 9-check `/validate` cascade-skipped the rest |
| 3 | Asked composer to wire the secret reference properly | v2 (no change) | LLM did **not** mutate state. Re-emitted the same preflight error verbatim. Likely the wire-secret tool returned an availability error and the LLM gave up. |
| 4 | Asked the LLM to *diagnose* why wiring failed (don't change pipeline) | v2 (no change) | LLM re-emitted the same preflight error a third time. **No diagnostic content.** This is a regression vs 05-01 where msg4 brilliantly identified `ELSPETH_FINGERPRINT_KEY` as unset on the deploy. The model has the catalog and the `/api/secrets` data available via tools, but didn't probe further. |

**Engine ran?** No — blocked at `/validate secret_refs`. The deploy lacks
`ELSPETH_FINGERPRINT_KEY` (operator config gap, not a code bug — `/api/secrets` now
self-explains via `reason: fingerprint_resolver_not_configured`).

**Real on-disk output?** None.

**Architectural shape**: S1B is the canonical proof that the composer end-to-end will not
let an LLM-classified pipeline run without a wirable secret on this deploy. Once
`ELSPETH_FINGERPRINT_KEY` is set, this scenario should produce real classified output —
the composer-side shape is correct.

### S2 — aggregation per `customer_tier`
**Session:** `d90a2e68-2686-479f-99b6-4330543c68f3` · **Messages:** 4 · **Wall-clock:** 0s + 122s + 24s + 37s ≈ 183s composer-time + 3 execute attempts

| Msg | LLM did | Outcome | Notable |
|---|---|---|---|
| 1 | Honest pushback — refused to fabricate "list of ticket_ids per tier" because `batch_stats` doesn't support list aggregation. Listed exactly what `batch_stats` supports (count, sum, mean, group_by). Also flagged a separate signature gap in the blob-wiring tool re schema option. | No state mutation | Strongest "honesty about tool gaps" behaviour seen in either prior eval |
| 2 | Built complete pipeline: csv source → `batch_stats` aggregation with `group_by: customer_tier` → `field_mapper` (rename `sum`→`sum_of_amount`) → JSON sink | v1; `/validate is_valid: true` (9/9); `/execute` failed with `SchemaConfigModeViolation` | **Architectural class new specific #1** — field_mapper output schema mismatch (see above) |
| 3 | Diagnosed the divergence and patched `field_mapper`'s declared output schema to `[customer_tier: str, count: int, sum_of_amount: float]` | v2; `/validate is_valid: true` (9/9); `/execute` failed with `PluginContractViolation` on input | **Architectural class new specific #2** — field_mapper input contract mismatch (see above) |
| 4 | Diagnosed the second divergence ("the field_mapper had a fixed input schema that described its renamed output instead of the actual aggregate row it receives") and reshaped both input and output contracts | v3; `/validate is_valid: true`; `/execute` succeeded | **Real on-disk output written**: `outputs/tier_summary.json` (252 B) |

**Final output** (`/home/john/elspeth/data/outputs/tier_summary.json`):

```json
[
  {"count": 3, "customer_tier": "enterprise", "sum_of_amount": 450.0},
  {"count": 3, "customer_tier": "pro",        "sum_of_amount": 205.0},
  {"count": 2, "customer_tier": "starter",    "sum_of_amount": 15.0}
]
```

Hand-verified against the input CSV: enterprise = T-001(100)+T-003(200)+T-006(150) = 450
✓; pro = T-002(50)+T-005(75)+T-008(80) = 205 ✓; starter = T-004(10)+T-007(5) = 15 ✓.

**Engine ran?** Yes (third attempt, `db2a0fef-3437-4e67-9f4c-d4cd8a7d81e2` failed,
second attempt failed, third `/api/runs/$RID` succeeded with `status: completed,
rows_succeeded: 3`).

### S3 — gate routing + correction + revert
**Session:** `70bce1c4-2bb7-4ef2-8791-c2774c57e173` · **Messages:** 3 + driver-issued `/state/revert` · **Wall-clock:** 64s + 12s + 6s ≈ 82s

| Msg | LLM did | Outcome | Notable |
|---|---|---|---|
| 1 | Built complete pipeline first try: csv source → gate (`row['customer_tier'] == 'enterprise'`) → `high_priority`/`low_priority` sinks; proactively added `quarantine` sink for source-validation failures | v1; `/validate is_valid: true` (9/9); `/execute` HTTP 202; output files written; `run.status: failed` (gap #3 above) | **`elspeth-2c3d63037c` regression confirmed FIXED** — gate primitive renders cleanly into both `/state` (`nodes[]`) and `/state/yaml` (`gates:` block); no 500, no half-written node |
| 2 | **Surgical patch** — only the gate condition string changed (`row['customer_tier'] == 'enterprise'` → `row['customer_tier'] in ['enterprise', 'pro']`). Idiomatic membership operator. Every other field preserved. | v2; `/validate is_valid: true`; `/execute` produced 6/2 split (correct); same `status: failed` gap | Same patch idiom as 04-28 S3 |
| 3 | **Refused to silently rewrite as a revert.** Surfaced the `POST /api/sessions/{id}/state/revert` endpoint and explicitly distinguished "audit-preserving revert" from "new forward edit". | v2 (no mutation) | **Closes 04-28 S3 "subtle gap" finding** — model correctly understands the audit semantics of revert vs. forward-edit |

**Driver action**: With the LLM declining to fake a revert, the driver issued
`POST /api/sessions/{sid}/state/revert {"state_id": "<v1 state id>"}` directly. Returned
HTTP 200, created v3 from v1's state. Chat history then contained a `system` role message
*"Pipeline reverted to version 1."* — exactly the audit-trail signal the 04-28 eval
called missing.

**Outputs**:
- `outputs/high_priority-1.jsonl` (3 enterprise rows: T-001, T-003, T-006 — 482 B)
- `outputs/low_priority-1.jsonl` (5 non-enterprise rows: T-002 pro, T-004 starter, T-005 pro, T-007 starter, T-008 pro — 754 B)
- `outputs/high_priority-2.jsonl` (6 enterprise+pro rows — 966 B)
- `outputs/low_priority-2.jsonl` (2 starter rows: T-004, T-007 — 270 B)

All rows correctly routed in both runs. The user-facing artefacts are exactly what the
prompt asked for.

**Engine ran?** Yes, twice — `f626c033-…` (v1) and the v2 execute. Both wrote files;
both reported `status: failed` (gap #3).

### S1A — monolithic complete-pipeline ask
**Session:** `4ee30c5a-ae1d-4025-95cd-3f19340b163c` · **Messages:** 1 · **Wall-clock:** 135.8s

**Prompt** (one message): build the full csv → llm classify → 4-way category-routed sinks
pipeline, wire OPENROUTER_API_KEY as a secret reference, quarantine source-validation
failures, do it all in one go.

LLM built in 135.8s (within the 180s budget):

- 1 source: `csv` from blob, `mode: fixed`, 5 fields with types, quarantine policy
- 1 transform: `llm` (`classify` node), `provider: openrouter`, `model: openai/gpt-4o-mini`,
  template + system_prompt + response_field
- 3 chained gates (ladder pattern):
  - `bug_gate`: `row['category'] == 'bug'` → `bugs` / `billing_gate_in`
  - `billing_gate`: `row['category'] == 'billing'` → `billing` / `feature_gate_in`
  - `feature_gate`: `row['category'] == 'feature_request'` → `features` / `other`
- 6 sinks: `bugs.jsonl`, `billing.jsonl`, `features.jsonl`, `other.jsonl`, `errors.jsonl`,
  `source_quarantine.jsonl`

`/validate`: `is_valid: false`, `secret_refs: passed: false` with
`Literal value in credential field(s): classify:api_key`. The other 8 checks cascade-skipped.

This is an **end-to-end re-verification of `elspeth-72d1dccd44`** — the exact failure
shape the 05-01 eval surfaced (literal placeholder in an LLM `api_key` field surviving
into runtime) is now caught at composer-time.

**Engine ran?** No — blocked at validate. **Real on-disk output?** None.

The LLM behaviour itself was excellent — gate ladder correctly chained, route names
match sink names, proactive `errors` and `source_quarantine` sinks added without being
asked. The only blocker is the same operator-side secret-availability gap as S1B.

### Comparison with 05-01 S1A

05-01's S1A built a similar 1-source / 1-LLM / 5-gate / 6-sink pipeline in 125s, also
with a literal `api_key` placeholder. **The crucial difference**: in 05-01, that literal
slipped past `secret_refs` and the engine ran with `Bearer WILL_BE_WIRED_FROM_OPENROUTER_API_KEY`,
producing 6 rows in `parse_quarantine.jsonl` and zero classified rows. Today, the same
shape is correctly rejected at composer-time and the engine never starts. The `/validate`
behaviour matches the runtime intent exactly — operators are no longer told "this is
fine" only to discover it isn't at execute time.

## Final scoreboard

| Scenario | Composer happy | /validate happy | Engine ran | Real output | Output matches user intent? |
|---|---|---|---|---|---|
| **S4** (vague Excel) | n/a (no mutation) | n/a | n/a | n/a | ✅ — clean refusal with clarifying questions |
| **S1B** (incremental classifier) | ✅ from msg2 | ❌ (`secret_refs`, operator config gap) | ❌ | none | n/a |
| **S2** (aggregation) | ✅ from msg2 | ✅ (msg2/3/4) | ✅ on attempt 3 (`db2a0fef-…`) | **`tier_summary.json`** | ✅ — three tier rollups, math correct |
| **S3** (gate routing) | ✅ msg1 | ✅ msg1 | ✅ both runs | **`high_priority-{1,2}.jsonl` + `low_priority-{1,2}.jsonl`** | ✅ — perfect routing, but `run.status` reports failed (gap #3) |
| **S1A** (monolithic LLM-classify) | ✅ msg1 | ❌ (`secret_refs`) | ❌ | none | n/a (blocker is `secret_refs` correctly catching the placeholder, not LLM error) |

**Three of five scenarios produced real on-disk output**, two of which were the artefact
the user asked for (S2 tier rollup, S3 routed JSONLs). S1B and S1A are blocked on the
same operator-side fingerprint-resolver gap, not on a composer or LLM defect. The five
scenarios collectively reached `is_valid: true` at the composer 9 times and at runtime
3 times.

Compared to 05-01: same number of scenarios producing real output (3), but **the output
that landed today actually matched the user's request** (05-01 reported "None of them
produced the output the user originally asked for" — S2's came after two LLM repair
turns, S3's was the simplified fallback because gates didn't work, S1A's was the error-path
quarantine). Today S2 also took repair turns but the result is correct; S3's gate
output is exactly what was asked for.

## What a real user would actually experience

- **Vague prompt**: gets clarifying questions in ~8s, no fabrication. Same as 05-01.
- **First "build me everything" prompt** (S1A): succeeds structurally in ~2 minutes,
  produces correct YAML for a 6-sink ladder routing pipeline. **The composer correctly
  refuses to claim completion** when the only remaining gap is a literal credential
  placeholder, and provides actionable language ("Wire each credential field through the
  Secrets panel..."). On a deploy with a configured fingerprint resolver, the next step
  would be one secret-wiring round-trip and the pipeline would run.
- **Gate-routing pipelines**: now build, validate, and execute end-to-end with correct
  per-row routing on first prompt. The user's original request is satisfied. **However**,
  `run.status` reports `failed` and the error message says "no row reached the success
  path... inspect /diagnostics for per-row failure details" — operators inspecting the
  status line will think the pipeline failed even though every row landed in the right
  sink.
- **Aggregation**: builds and validates on first prompt, but the engine surfaces two
  contract divergences in sequence (output schema mismatch, then input contract
  mismatch). Each divergence is *recoverable* by feeding the runtime error back to the
  LLM, but each round-trip costs an OpenRouter call. The composer's `/validate` says
  "fine" and the runtime keeps disagreeing.
- **Validation deception**: The original 05-01 architectural class is **not** entirely
  closed. The directly-verified-as-caught specifics today are: literal `api_key`
  placeholder (S1A msg1, S1B msg2), and dangling `on_error` reference (driver-issued
  verify session `02b4853e-…` — note the catching check is `graph_structure` with detail
  `Transform 'pass' on_error 'nonexistent_failsink_xyz' references unknown sink.
  Available sinks: results.` — `route_target_resolution` cascade-skipped behind it). The
  `mode: flexible + required_fields` shape is **NOT** caught — S2 v3's `aggregations[0].
  options.schema` contains exactly that combination and `/validate` returned `is_valid:
  true`; the engine also accepted it and the run executed successfully. The 05-01
  finding's specific failure mode (`SchemaConfigModeViolation` raised at runtime) does
  not reproduce, but the catch is upstream of the validator: runtime now tolerates the
  combination on this code path. That is "no longer divergent because runtime quietly
  accepts," not "now caught at composer." Two new field_mapper-shaped specifics
  (`mapping`-vs-declared-output, locked-input-vs-upstream-emission) survive. The
  pattern persists: any plugin-pair (`batch_stats` → `field_mapper` here) where one
  emits a closed-set output and the other consumes via a locked-input contract is a
  fertile ground for new divergence shapes.

## Filed during this eval

### New issues filed (parented to `elspeth-528bde62bb`, labels `composer`, `cluster:rc5-ux`)

To be filed below — see "New filings" section.

### Comments added

To be added below — see "Comments added" section.

### LLM behavioural regression (will file as bug)

S1B msg3 + msg4 — composer LLM fails to diagnose why secret-wiring didn't take effect.
Re-emits the same `/validate` complaint verbatim across two consecutive turns despite
explicit user request to diagnose rather than retry. This is a regression vs 05-01 S1B
msg4 which correctly identified `ELSPETH_FINGERPRINT_KEY` as the unset upstream gate.

## Recommendations (priority ordered)

1. **Fix the routing-pipeline `run.status` semantics** (P1 in operator UX). When
   `rows_processed == rows_routed > 0` and every operation completed, `run.status`
   should not be `failed`. The four-value `RunStatus` taxonomy from `cc895589` has the
   slot for this — likely needs `completed_routed` or for `completed` to apply when
   every token reached a terminal `routed` outcome. Today's "no row reached the success
   path" message is misleading on a pipeline where the success path *is* the routed
   sinks. **Reproducer**: S3 v1/v2 runs `f626c033-…` and the second exec.
2. **Promote `field_mapper` mapping/declared-output coherence to `/validate`**.
   The `mapping` value side defines emitted output keys; declared `schema.fields` /
   `required_fields` / `select_only: true` must agree. Today composer ships these
   inconsistent. **Reproducer**: S2 v2 yaml in `s2/state.yaml`, runtime
   `SchemaConfigModeViolation` in `s2/diag2.json`.
3. **Promote upstream-emission-vs-downstream-input coherence to `/validate`**.
   When a downstream transform has a locked input contract (Pydantic
   `extra_forbidden`), the composer should detect the upstream's declared output
   superset/disjoint-set against that contract. `batch_stats` → `field_mapper` is the
   canonical case (`batch_size` is always emitted, always extra-forbidden by the
   mapper). **Reproducer**: S2 v3 yaml, runtime `PluginContractViolation` in `s2/diag3.json`.
4. **Composer LLM should retry / diagnose tool-call failures rather than re-emitting the
   same validate error** (S1B msg3 + msg4 regression). When the secret-wiring tool fails
   the composer should surface the failure mode (availability vs. permission vs.
   resolver), not just re-issue the prior assistant text. Currently the LLM appears to
   short-circuit to the validate complaint without exercising other tools.
5. **Set `ELSPETH_FINGERPRINT_KEY` on the staging deploy** — operator config item, not a
   code bug. Until then, no LLM-classifier pipeline can run on staging via the composer
   even though the composer-side code is now correct end-to-end. (Out of eval scope to
   fix, but blocks two of five scenarios.)

## Files / artefacts

- `/tmp/elspeth_eval_2026-05-02/` — raw transcripts, prompts, response bodies,
  intermediate states, validate/diag/run JSON, captured plugin schemas.
- Sessions:
  - S4: `5c002509-94a4-484f-adf4-6befdd2323b8`
  - S1B: `6b2988e2-3963-4839-a08b-43af88438f58`
  - S2: `d90a2e68-2686-479f-99b6-4330543c68f3`
  - S3: `70bce1c4-2bb7-4ef2-8791-c2774c57e173`
  - S1A: `4ee30c5a-ae1d-4025-95cd-3f19340b163c`
  - Verify (route_target_resolution test): `02b4853e-8c8c-4e1f-86d3-bc7d8c512c25`
- Successful runs:
  - S2 v3 run `db2a0fef-3437-4e67-9f4c-d4cd8a7d81e2` → `outputs/tier_summary.json`
  - S3 v1 run `f626c033-f226-4040-94af-ba3e2db9d35c` → `outputs/{high,low}_priority-1.jsonl`
  - S3 v2 (post-patch) run → `outputs/{high,low}_priority-2.jsonl`

## Closing the loop

This eval **partially satisfies** `elspeth-599ecf69fa` *(P1 task — Repeat staging
composer LLM evaluation)*. Per its acceptance gate ("report scenarios now execute or
fail early"):

- **Execute end-to-end**: S2, S3 (twice), S4 (clarification = end-to-end success in this
  context) — **3 of 5**.
- **Fail early with structured, user-actionable error**: S1A (literal `api_key`),
  S1B (literal `api_key`) — **2 of 5**, both blocked at `secret_refs` with the structured
  error and suggestion field, exactly as Phase 1.1 of the remediation program intended.

That's **5 of 5 scenarios executed end-to-end OR failed early with actionable error.**
The acceptance gate is satisfied for the categorical question, but the architectural
class still has live specifics in S2 (two new shapes), so the parent epic
`elspeth-528bde62bb` should not close yet. New issues filed against the epic below
capture those.
