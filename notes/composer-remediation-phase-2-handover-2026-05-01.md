# Phase 2 handover — Composer remediation (operator visibility)

**Parent epic:** `elspeth-528bde62bb`
**Program doc:** `notes/composer-remediation-program-2026-05-01.md`
**Source audit:** `notes/composer-llm-eval-2026-05-01.md`
**Date prepared:** 2026-05-01
**Prior phases:** Phase 0 ✅ · Phase 1 ✅ (1.1, 1.2, 1.3 all landed)

## Executive summary

Phase 2 closes three operator-visibility gaps surfaced by the 2026-05-01 staging eval.
All three items are P2, independent of each other, and can run in parallel. None are
blockers for Phase 4 (eval re-run) but **all three should be addressed before re-running
the eval**, because Phase 4's success criterion includes "operator-facing run summaries
truthfully distinguish success/degraded/failed" and "secret unavailability self-explains."

The unifying theme is **truth-telling at the API layer**: when something goes wrong, the
operator gets either a confident wrong answer (`status: completed` with 0 succeeded) or no
answer at all (`available: false` with no reason; output written but `/api/runs/{rid}`
stale for 30s). Phase 0 fixed the diagnostic surface for composer crashes; Phase 2 does
the same for run-state and secret-state APIs.

## Work items (parallel-safe)

### 2.1 — Pipeline completion callback exception on successful aggregation runs

**Issue:** `elspeth-31d53c7493` *(P2 bug, status `triage`)*

**Symptom:** S2's batch_stats run wrote real output but `/api/runs/{rid}` returned
`null` body fields for ~30s and `/diagnostics` returned 404. journald showed
`pipeline_done_callback_exception exc_class_chain=['ValueError','ValidationError']`.

**Reproducer:**
- Run id: `44f52421-a379-459b-96a8-6f0656086f16`
- Session: `db27402d-cbf5-4bcc-89fb-673fbecb823e`
- Pipeline shape: csv source → `batch_stats` aggregation (`group_by: customer_tier`,
  `value_field: amount`, `mode: observed`) → json sink
- Output written: `/home/john/elspeth/data/outputs/customer_tier_summaries.json`

**Single occurrence in the eval.** Did not reproduce on S3's simple source→sink
completion or S1A's row-failed completion. May be specific to batch-aware aggregation
completion path. **Start narrow** — root-cause first, generalise only if a second shape
shows up.

**Search hint for whoever picks this up:**
- Grep `pipeline_done_callback` in `src/elspeth/engine/` — that's where the structured
  log event fires from
- Run-state writeback path: look for the engine→web bridge that records terminal run
  state into the API store (likely in `src/elspeth/web/execution/` or
  `src/elspeth/web/runs/`)
- Hypothesis: the `ValueError → ValidationError` chain suggests Pydantic validation
  rejecting something the aggregation completion path emits — possibly a numeric type
  coercion (sum: 450.0 vs 450) or a dataclass-vs-dict shape

**Acceptance gate:**
- The S2 successful-run shape (csv → batch_stats with `group_by` → json sink) executes
  with no `pipeline_done_callback_exception` in the journal
- `/api/runs/{rid}` returns the terminal state immediately on completion (no eventual
  consistency delay)
- Regression test under `tests/integration/` exercising the batch_stats + group_by
  completion path against the API store

**Effort:** Likely 1-2 days if the failure reproduces locally; longer if it only repros
in staging shape.

### 2.2 — Run status semantics: `completed` when 0 rows succeeded

**Issue:** `elspeth-0de989c56d` *(P2 task, status `open`)* — promoted from
`elspeth-obs-5c21c0f9cd`

**Symptom:** Two distinct shapes both report `status: completed`:
- Healthy: `rows_processed: 6, rows_succeeded: 6, rows_failed: 0`
- All-failed: `rows_processed: 6, rows_succeeded: 0, rows_failed: 0, rows_routed: 6`
  (S1A — every row routed via `on_error` to quarantine because the LLM transform failed
  on a literal `api_key` placeholder)
- All-failed (no on_error): `rows_processed: 6, rows_succeeded: 0, rows_failed: 6` (S1B
  msg2 — engine ran cleanly with all rows failed, no quarantine route)

Operator scanning a list of runs cannot distinguish these without reading diagnostics
or opening output files.

**Search hint:**
- Run-state model: look for the enum or string field defining run status. Likely in
  `src/elspeth/contracts/` (L0 — would be the right place for a status enum) or
  `src/elspeth/engine/orchestrator/` for the canonical state machine
- The seven row terminal states are defined as `COMPLETED`, `ROUTED`, `FORKED`,
  `CONSUMED_IN_BATCH`, `COALESCED`, `QUARANTINED`, `FAILED`, `EXPANDED` per CLAUDE.md.
  The run-level status is a separate concept (run = aggregate of all rows) — that's the
  layer to extend.

**Design decision (2026-05-02): Option A** — see issue comment on
`elspeth-0de989c56d` for the full rationale, four-value taxonomy
(`completed` / `completed_with_failures` / `failed` / `empty`), and
biconditional invariant pattern. Reasons B and C were rejected:
- Option C derives the verdict at read time, which fails CLAUDE.md's
  "no inference" auditability rule (run-state IS Tier-1 audit data).
- Option B's status×health cross-product introduces representable-but-
  illegal combinations (`status: failed, health: healthy`); the
  biconditional discipline that worked for 1.3 and 2.3 is harder to
  enforce across two fields than one closed-list enum.

The implementer should consult the issue comment before coding — it
records three explicit anti-patterns and one open design question
(`rows_routed`-only outcome → `failed` vs `completed_with_failures`).

**Acceptance gate:**
- S1A's exact run shape (`rows_routed: 6, rows_succeeded: 0`) reports a status string
  distinct from a healthy completion
- S1B msg2's shape (`rows_succeeded: 0, rows_failed: 6`, no on_error route) also
  distinguishable
- A regression test under `elspeth-1ee3c96c72` (composer/runtime agreement test suite)
  encodes the three reproducer shapes
- Audit-trail consistency: the new status value persists in Landscape with the same
  primacy as today's `completed`/`failed` (no silent dropping at the audit boundary)

**Effort:** 2-3 days. Schema/enum change + run-state machine touchpoint + API readback +
test. Touches L0 (contracts) and L2 (engine), so layer-import enforcement should be
checked.

### 2.3 — `/api/secrets` diagnostic reason for `available: false`

**Issue:** `elspeth-0d31c22d26` *(P2 task, status `open`)* — promoted from
`elspeth-obs-513080dee7`

**Symptom:** `GET /api/secrets` on staging returns:

```json
{"name":"OPENROUTER_API_KEY","scope":"server","available":false,"source_kind":"env"}
```

The env var IS set. Root cause is `ELSPETH_FINGERPRINT_KEY` unset, so
`WebSecretResolver` cannot fingerprint-back any secret. The eval's LLM had to deduce
this from runtime errors (S1B msg4) — qualitatively the strongest LLM behaviour
observed, but it shouldn't have needed to.

**Search hint:**
- `WebSecretResolver` likely in `src/elspeth/web/` — find the `/api/secrets` route
  handler and walk back to where availability is determined
- The closed sibling issue `elspeth-cd5d811121` made the API and runtime layers agree
  on availability; this issue extends the API to carry a *reason* alongside the
  agreement
- Audit hygiene constraint (see below) is critical — copy the discipline from
  `elspeth-72d1dccd44`'s `_collect_credential_field_violations` (never echoes the
  offending literal)

**Reason taxonomy proposal:**
- `"fingerprint_resolver_not_configured"` — `ELSPETH_FINGERPRINT_KEY` unset
- `"env_var_not_set"` — name in inventory but missing from process env
- `"fingerprint_mismatch"` — value present but does not match registered fingerprint
- `"resolver_error"` — fallback for unexpected resolver failures

The exact strings are up to the design — what matters is that they're closed-list,
machine-readable (no free-form prose), and never carry secret material.

**Audit-hygiene constraint (load-bearing):**
The `reason` field MUST NOT echo any candidate-secret value, fingerprint bytes, or
deploy-environment file contents. Only the structural failure mode. This is the same
discipline as the secret_refs validator. Reviewers should flag any code path that
interpolates user-supplied or env-supplied data into the reason string.

**Acceptance gate:**
- `/api/secrets` on a deploy with `ELSPETH_FINGERPRINT_KEY` unset returns each entry
  with `reason: "fingerprint_resolver_not_configured"` (or equivalent closed-list value)
- An operator reading the response can identify the configuration gap without recourse
  to source or runtime errors
- Regression test that mocks the resolver into each failure mode and asserts a distinct
  reason string
- Audit-hygiene test: a unit test feeds the resolver an env var with a recognisable
  sentinel value and asserts the sentinel never appears in any `/api/secrets` response

**Effort:** 1-2 days. Single API route touch + resolver-side reason classification +
tests.

## Out-of-scope items the audit surfaced (do not work in Phase 2)

- **`elspeth-obs-c59bc8bf8e`** — duplicate OTel counter name `composer.runtime_preflight.total`
  across two meter scopes (`composer/service.py:90` vs `sessions/routes.py:294`). Not
  from this eval; legitimate observation but not in the program. Promote separately if
  it warrants a P3.
- **`elspeth-obs-d52da9b57a`** — `pipeline_composer.md` skill file 65,405 bytes,
  exceeds 64 KB deployment-overlay cap. Skill-pack hygiene, not composer remediation.
  Owner: skill-pack maintainer.

## Sequencing

```text
Phase 2 work items (all P2, all parallel-safe):

  ┌─────────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────────────┐
  │ 2.1 elspeth-31d53c7493      │  │ 2.2 elspeth-0de989c56d      │  │ 2.3 elspeth-0d31c22d26      │
  │ pipeline_done_callback      │  │ run.status taxonomy         │  │ /api/secrets reason         │
  │ (single-occurrence bug)     │  │ (Tier-1 enum/schema work)   │  │ (API extension + audit hyg) │
  └─────────────────┬───────────┘  └─────────────────┬───────────┘  └─────────────────┬───────────┘
                    │                                │                                │
                    └───────────────┬────────────────┴────────────────┬───────────────┘
                                    ▼                                 ▼
                        Phase 3: regression coverage (elspeth-1ee3c96c72)
                                    │
                                    ▼
                        Phase 4: re-run staging eval (elspeth-599ecf69fa)
                                    │
                                    ▼
                        Close epic elspeth-528bde62bb
```

## Tracker hygiene to clean up

These were not addressed because Phase 1 was reported complete by the implementer; the
issue-tracker statuses are stale relative to landed code. The next person touching these
should transition them — confirming the acceptance gates are met before transitioning.

| Issue | Current status | Suggested transition | Reason |
|---|---|---|---|
| `elspeth-72d1dccd44` (Phase 1.1 secret_refs) | `triage` | `confirmed` or higher | Code landed in commit `3b7ca22b`; S1A reproducer fails `/validate` with structured error per acceptance gate |
| `elspeth-127de6865a` (Phase 1.2 route_target) | `triage` | `confirmed` or higher | Same commit; S2 v1 reproducer (dangling `on_error: aggregation_errors`) covered by new `assemble_and_validate_pipeline_config()` shared helper |

## Cross-reference: standing acceptance gates

These come from the program doc and apply across Phase 2:

- **No silent failures** — Every API endpoint that today returns an inscrutable
  `false`/`null`/empty response in a failure path must return a structured, named
  reason. This is the unifying theme across all three Phase 2 items.
- **Audit-hygiene** — Diagnostic surfaces must not echo candidate-secret material.
  Phase 0 set the precedent (`_safe_frame_strings` in
  `src/elspeth/web/sessions/routes.py`), Phase 1.1 reinforced it
  (`_collect_credential_field_violations` in
  `src/elspeth/web/execution/validation.py` returning field names only). Phase 2.3
  must follow the same pattern; Phase 2.1's run-state writeback should never persist
  user-row data; Phase 2.2's status enum is structural so isn't a hygiene risk.
- **Regression encoding** — Every Phase 2 fix gets a test under `elspeth-1ee3c96c72`
  with the exact reproducer shape from the eval. The "closed list of validator-runtime
  divergence shapes" from the program doc extends to operator-visibility shapes too —
  if it shipped broken once, the test must encode it so it can't ship broken twice.

## Quick-start for whoever picks this up

```bash
# Inspect the three issues in order:
filigree show elspeth-31d53c7493     # 2.1 — pipeline_done_callback
filigree show elspeth-0de989c56d     # 2.2 — run.status taxonomy
filigree show elspeth-0d31c22d26     # 2.3 — /api/secrets reason

# Check valid transitions before claiming:
filigree transitions elspeth-31d53c7493
filigree transitions elspeth-0de989c56d
filigree transitions elspeth-0d31c22d26

# Claim with explicit assignee:
filigree claim elspeth-0de989c56d --assignee <name>

# Read the program doc for context:
less notes/composer-remediation-program-2026-05-01.md

# Read the source audit for the eval scenarios these came from:
less notes/composer-llm-eval-2026-05-01.md
```

## Status snapshot — full program

```text
Phase 0 — Gate primitive (P1)              ✅ landed (elspeth-2c3d63037c, 209b7e3a2b deferred)
Phase 1.1 — secret_refs fabrication-aware  ✅ landed (elspeth-72d1dccd44, status stale)
Phase 1.2 — route_target_resolution        ✅ landed (elspeth-127de6865a, status stale)
Phase 1.3 — schema mode/required_fields    ✅ landed (elspeth-f5f798f797, closed)
Phase 2.1 — pipeline_done_callback         ⬜ ready  (elspeth-31d53c7493, P2 triage)
Phase 2.2 — run.status taxonomy            ⬜ ready  (elspeth-0de989c56d, P2 open)
Phase 2.3 — /api/secrets reason            ⬜ ready  (elspeth-0d31c22d26, P2 open)
Phase 3   — regression coverage            ⬜ pending (elspeth-1ee3c96c72, P3 in_progress)
Phase 4   — staging eval re-run            ⬜ pending (elspeth-599ecf69fa, P1 open)
```

Phase 1's structural fix (`assemble_and_validate_pipeline_config` collapsing
`/validate` and `/execute` into one shared code path) is *stronger than the program
asked for* — it answers the architectural question reserved for after Phase 1 ("why is
`/validate` still not a strict superset of runtime preflight"). Worth surfacing in the
Phase 4 eval report when it's written.
