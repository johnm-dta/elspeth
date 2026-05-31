# Composer remediation program — derived from 2026-05-01 staging eval

**Source audit:** `docs/composer/evidence/composer-llm-eval-2026-05-01.md`
**Parent epic:** `elspeth-528bde62bb` *(Composer LLM evaluation remediation — validator parity, runtime dry-run, and operator visibility)*
**Date:** 2026-05-01
**Author:** Claude (post-eval analysis)

The 2026-04-28 eval's child issues are largely landed. The 2026-05-01 eval confirms
those specific fixes hold but surfaces a P1 regression (gate primitive crash) and three
new specifics of the same architectural class the prior epic was supposed to close. This
document organises the response into a sequenced program with acceptance gates, calls out
issue-tracker gaps, and recommends what to file/re-scope.

## Headline classification

| Class | Severity | What it is | Items |
|---|---|---|---|
| **Regression** | P1 | A previously working capability (gate routing through composer) now 500s reliably | `elspeth-2c3d63037c` |
| **Architectural class** | P2 (each, but cumulative P1) | "Composer says valid, runtime rejects" survives on three new specifics | `elspeth-72d1dccd44`, `elspeth-87f6d5dea5` (existing, scope-stretched), **two issues to file** |
| **Operator visibility** | P2 | Status/diagnostics lie or omit information | `elspeth-31d53c7493`, two observations to promote |
| **Regression coverage** | P2 | Tests must lock the new specifics so future composer/runtime drift fails CI | `elspeth-1ee3c96c72` (existing), needs concrete cases added |

The audit also confirmed substantial uplifts (monolithic prompts no longer time out;
`/validate` is now an 8-check preflight; `/state/yaml` returns 409 + structured errors;
`/state/revert` works; per-tier rollups are real). Those don't need program work — they
need celebration in the eval re-run report.

## Phase 0 — Stop the bleeding (drop everything)

The gate-primitive 500 (`elspeth-2c3d63037c`) blocks **the entire row-branching
capability** through the composer. Prior eval's strongest LLM behaviour scenario (S3 — gate
+ correction + revert) is currently unreachable. This must be fixed before any further
program work because:

1. It silently corrupts session state with half-written nodes (every failed attempt bumps
   the version), so users who hit it leak partial state into their session history.
2. The error message lies — claims "see server logs for the traceback" when no traceback is
   logged. This compromises operator diagnosis and erodes trust in the failure surface.
3. Any composer-shape regressions involving gates will pile up behind it; the audit can't
   exercise them.

**Acceptance gates for Phase 0:**
- `elspeth-2c3d63037c` is closed with a focused regression test reproducing the original
  500 path and proving it now returns either success or a structured 422.
- The `composer_plugin_error` 500 path either logs the traceback with appropriate redaction,
  or the error message no longer references server logs. *Verify in `journalctl -u
  elspeth-web.service` that crashes are now diagnosable.*
- Failed compose mutations leave state untouched (atomic) — confirmed by composing a
  guaranteed-invalid gate node and asserting the version counter does not advance.

`★ Insight ─────────────────────────────────────`
- The "no traceback" symptom is the diagnostic blocker, not a side effect. Whoever picks
  up `2c3d63037c` should fix the logging contract first, then use the now-visible
  traceback to debug the actual crash. Otherwise they're flying blind.
- Atomic state mutation matters even if the underlying bug is fixed — any future failure
  in this code path should not be able to corrupt state. This is a one-line invariant to
  enforce in the message-handler wrapper, and it prevents a class of session-corruption
  bugs we haven't even found yet.
`─────────────────────────────────────────────────`

## Phase 1 — Close the architectural class (validator parity)

The 2026-04-28 eval's headline was "composer-time `state.validate()` is a strict subset of
runtime validation." The 2026-05-01 eval proves the *fix wasn't structural* — the original
three specifics (path allowlist, batch-aware required fields, blob path normalisation) were
patched, but the underlying contract gap reproduces on three new shapes.

**Items in this phase, in dependency order:**

### 1.1 — `secret_refs` becomes fabrication-aware

`elspeth-72d1dccd44` *(P2 bug, filed)*

Today the `secret_refs` check only looks for `{secret_ref: <name>}` constructs. A literal
placeholder string in a known-credential-bearing field passes through unflagged. Fix
direction: walk the plugin's required-secret list (already known to `CatalogServiceImpl`)
and probe each option for either a wired `secret_ref` or a permitted env-marker construct;
reject literal non-empty strings.

**Acceptance gate:** `secret_refs` rejects S1A's
`api_key: WILL_BE_WIRED_FROM_OPENROUTER_API_KEY` shape with a structured error pointing at
the field path. A test case captures the exact placeholder.

### 1.2 — `route_target_resolution` is added to `/validate`

**Not yet a standalone issue.** The audit's recommendation #3 was added as a comment on
`elspeth-87f6d5dea5`, but that issue's stated scope is alias handling / nested aggregation
requirements / fork-to-sink — not route-target dereferencing. **Recommend filing** a new
P2 bug or re-scoping `87f6d5dea5` to absorb this.

The check: every `transforms[*].on_error` and `aggregations[*].on_error` reference must
resolve to either a sink name, a gate input, or `discard`. S2 v1 slipped through because
the composer accepted `on_error: aggregation_errors` where no such sink existed.

**Acceptance gate:** S2 v1's exact YAML fails `/validate` with a structured error naming
the unresolvable target.

### 1.3 — `schema_compatibility` detects mode/options incompatibilities

**Not yet a standalone issue.** Same situation as 1.2 — recommendation #4 is a comment on
`87f6d5dea5` but exceeds that issue's stated scope. **Recommend filing.**

The check: when `schema.mode == flexible`, `required_fields` is incompatible (runtime
raises `SchemaConfigModeViolation`). Composer should reject this combination at validate
time.

**Acceptance gate:** S2 v2's exact YAML fails `/validate` with a structured error naming
the incompatible combination.

### Phase 1 acceptance gate (composite)

A new staging eval run with the prior eval's failure prompts produces zero "validate
green / execute red" outcomes for the *three known shapes*. Any future shape that survives
into production becomes a test case under `elspeth-1ee3c96c72`.

`★ Insight ─────────────────────────────────────`
- The architectural class isn't going away with three more validators. The *structural*
  fix is to make `/validate` runnable as a dry-run of the runtime preflight — same code
  path, mocked execution. `elspeth-34baf10c01` (status `done`) was supposed to deliver
  this; the 2026-05-01 eval shows it didn't, fully. The program should treat the
  three-validator approach as patching, and at the end of Phase 1 the architecture
  question gets re-opened: *why is `/validate` still not a strict superset of runtime
  preflight?*
- Re-scoping `87f6d5dea5` vs filing two new issues is a classic ticket-hygiene tradeoff.
  My recommendation: **file two new issues.** The existing one is in `verifying` — adding
  scope re-opens the verification cycle and the original three fixes get re-litigated.
  Cleaner to file 1.2 and 1.3 as P2 bugs blocked by Phase 0 and parented to the same
  epic.
`─────────────────────────────────────────────────`

## Phase 2 — Operator visibility (the truth-telling layer)

These are P2 in isolation but together they form a single operator-experience gap: when
something goes wrong, operators get a confident wrong answer or no answer at all.

### 2.1 — `pipeline_done_callback_exception` on successful runs

`elspeth-31d53c7493` *(P2 bug, filed)*

Single occurrence in the eval (S2 batch_stats success path). Output landed; API readback
was broken for ~30s. Diagnostic chain shows `ValueError`/`ValidationError` in completion
callback. Single-occurrence root-cause work — start narrow.

**Acceptance gate:** The S2 successful-run shape (csv → batch_stats with group_by → json
sink) executes with no callback exception in the journal, and `/api/runs/{rid}` returns
the terminal state immediately on completion (not after eventual consistency).

### 2.2 — Promote observation: `run.status: completed` semantics

`elspeth-obs-5c21c0f9cd` — operator-misleading run terminal state. Today
`rows_succeeded: 0, rows_routed: 6` (everything routed via `on_error` to quarantine) shows
the same `status: completed` as a healthy run. Operator scanning `/api/runs/...` cannot
distinguish "ran successfully" from "ran with all rows failing."

**Recommendation: promote to a P2 task** parented to `elspeth-528bde62bb`. Likely fix is a
`completed_with_failures` / `degraded` status computed from `rows_succeeded == 0 &&
rows_processed > 0` or `rows_routed_via_on_error >= rows_processed`.

**Acceptance gate:** S1A's exact run shape (`rows_routed: 6, rows_succeeded: 0`) reports
a status distinct from a fully-successful run.

### 2.3 — Promote observation: `/api/secrets` diagnostic gap

`elspeth-obs-513080dee7` — `/api/secrets` returns `available: false` with no detail.
Operator cannot discover that `ELSPETH_FINGERPRINT_KEY` is unset (the actual cause of
S1B's runtime block) without consulting source code or runtime errors. The eval's LLM had
to deduce this from runtime behaviour.

**Recommendation: promote to a P2 task.** Likely fix is to expose a diagnostic field
(`available: false, reason: "fingerprint_key_unset"` or similar) that an operator-tier
client can read.

**Acceptance gate:** Reading `/api/secrets` on a deploy missing `ELSPETH_FINGERPRINT_KEY`
returns a structured diagnostic identifying the gap.

### Phase 2 acceptance gate (composite)

The eval's "What a real user would actually experience" section — specifically the
"Validation deception" and "Secrets" bullets — can be rewritten with the deception
gone: validate failures must produce structured, actionable errors, and secret
unavailability must self-explain.

## Phase 3 — Regression coverage (lock the gains)

`elspeth-1ee3c96c72` *(P3 task, in_progress)* — Expand composer/runtime agreement tests.

The audit explicitly directs adding S2 v1 (RouteValidationError) and S2 v2
(SchemaConfigModeViolation) as test cases. To that I'd add:

- S1A's literal-string `api_key` placeholder as a `secret_refs` test case (covers Phase 1.1)
- S3's gate-primitive happy path as an integration test (covers Phase 0 — must stay green)
- A meta-test: pick three contract-divergence shapes from any future eval, encode them
  here. The archetype is "Shifting the Burden" — every time we patch one specific, we
  push the structural fix off. The test suite must encode the shapes so future drift
  fails CI.

**Acceptance gate:** All five test cases (three from this eval + S1A secret + gate-primitive
integration) green in CI. The closed list of "validator-runtime divergence shapes" is
maintained as a registry that any new finding must extend.

## Phase 4 — Re-run the eval (close the loop)

`elspeth-599ecf69fa` *(P1 task, open)* — Repeat staging composer LLM evaluation.

The 2026-05-01 eval **partially satisfied** this task (it ran the prior eval's
scenarios), but the task's exit criterion is "report scenarios now execute or fail early"
and the 2026-05-01 eval did not meet that — three of five produced real output, none
produced what the user originally asked for, and the architectural class survived.

**After Phases 0-3 land**, run the eval again, with the *combined* prior-eval scenarios
plus the three new shapes from 2026-05-01 (S2 v1, S2 v2, S1A literal placeholder) added as
explicit characterization tests.

**Acceptance gate (this is also the program's exit gate):**
- All five scenarios from the 2026-05-01 eval execute end-to-end OR fail with a structured,
  user-actionable error before `/execute` is called.
- Zero "composer green / runtime red" outcomes on the closed list of known shapes.
- Operator-facing run summaries truthfully distinguish success/degraded/failed.
- Eval report appended to the durable notes directory and parented under
  `elspeth-528bde62bb`.

## Issue tracker gaps and recommendations

Filing recommendations, in priority order:

| Action | Item | Reason |
|---|---|---|
| **File new P2 bug** | route_target_resolution composer check | Recommendation #3, currently a comment on `87f6d5dea5` exceeding that issue's scope |
| **File new P2 bug** | schema-mode/required_fields composer check | Recommendation #4, same reason |
| **Promote obs** | `obs-5c21c0f9cd` (run.status: completed gap) | 14-day TTL; carries recommendation #8 |
| **Promote obs** | `obs-513080dee7` (/api/secrets diagnostic) | 14-day TTL; carries recommendation #7 |
| **Add deps** | `2c3d63037c` blocks (1.2, 1.3, eval re-run) | Phase 0 must precede further composer mutation testing |
| **Add deps** | `599ecf69fa` blocked_by all of Phase 0/1/2 outputs | Re-eval depends on remediation |

## Sequencing summary

```text
[ Phase 0: Gate primitive (P1) ]──┐
                                  │
                                  ▼
[ Phase 1.1 secret_refs ]   [ Phase 1.2 route_target ]   [ Phase 1.3 schema_compat ]
        │                          │                              │
        └──────────┬───────────────┴──────────────┬───────────────┘
                   ▼                              ▼
        [ Phase 2.1 callback ] [ Phase 2.2 status ] [ Phase 2.3 secrets-diag ]
                   └──────────────┬───────────────┘
                                  ▼
                       [ Phase 3: Test coverage ]
                                  │
                                  ▼
                       [ Phase 4: Re-run staging eval ]
                                  │
                                  ▼
                  Close epic elspeth-528bde62bb (RC 5 unblock)
```

Phase 0 is sequential. Phases 1.1/1.2/1.3 can run in parallel after Phase 0; same with
Phases 2.1/2.2/2.3. Phase 3 needs all of Phase 1 done. Phase 4 needs everything else done.

## What this program does NOT cover

- **The structural question**: why is `/validate` still not a strict superset of runtime
  preflight? `elspeth-34baf10c01` was supposed to deliver this and is `done`. The 2026-05-01
  eval suggests the implementation is patch-by-patch (8 named checks), not architectural
  (validate-runs-runtime-preflight-in-mock-mode). After Phase 1 lands, this question should
  be re-asked. If "Shifting the Burden" is the right archetype, no number of named checks
  closes the gap — only collapsing the validator and preflight into one code path does.
- **RC 5 release scope decisions**: this is a remediation program, not a release plan.
  Whether to ship RC 5 with these still open, ship RC 5 incrementally as phases close, or
  hold the release until Phase 4 closes is a release-management decision, not a remediation
  decision. CLAUDE.md memory says ELSPETH does not commit to calendar shipping; pace this
  program to correctness.
