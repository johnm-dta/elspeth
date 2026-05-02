# Split `rows_routed` Counter — MOVE vs DIVERT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** Filigree issue `elspeth-5069612f3c` (in_progress, P1) — "Split rows_routed counter to distinguish DIVERT (on_error) from MOVE (gate-routed)". Phase 1+2 root-cause investigation and design synthesis are recorded in comment 707 on that issue and comment 706 on the user-facing P1 `elspeth-71520f5e30`.

**Goal:** Split `ExecutionCounters.rows_routed` into `rows_routed_success` (intentional MOVE — gate `route_to_sink`) and `rows_routed_failure` (DIVERT — transform `on_error`), propagate the split through L0 contracts → L2 engine → L3 web in lockstep, and update the three mirror invariants so terminal-routed gate pipelines classify as `RunStatus.COMPLETED` instead of `RunStatus.FAILED`.

**Architecture:** Single PR. Eight tasks, executed in order. The architectural change is a producer-site discrimination — `RowOutcome.ROUTED` splits into `RowOutcome.ROUTED` (continues to mean intentional MOVE) and a new `RowOutcome.ROUTED_ON_ERROR` (DIVERT). The accumulator at `engine/orchestrator/outcomes.py:171-180` chooses which counter to increment based on the variant. The success predicate in `derive_terminal_run_status()` adds `rows_routed_success > 0` as a success indicator and `rows_routed_failure > 0` as a failure indicator. Per CLAUDE.md "no legacy code policy", `rows_routed` is removed entirely in the same PR — no shims, no deprecation. The sessions DB uses `metadata.create_all` (no Alembic), so the schema change is a model-declaration update; existing dev/staging session databases require recreation. The Landscape audit DB has no SQL schema diff, but it has a semantic value-domain split: pre-split `token_outcomes.outcome='routed'` rows are ambiguous under the new taxonomy. Checkpoint payloads can also contain the old `RunResult`, `ExecutionCounters`, `AggregationFlushResult`, or progress-counter shape. For dev/staging/pre-1.0 deployments, operators MUST archive/delete/recreate the Landscape audit DB, sessions DB, and configured checkpoint files/directories at deploy time, or explicitly document that historical `routed` rows and pre-split progress/MCP surfaces are legacy-ambiguous and must not be interpreted as MOVE-only evidence.

**Architecture decision — why this is not a predicate-only hotfix:** A narrower hotfix was considered: leave `rows_routed` intact, change only the terminal-status predicate so any routed row counts as success, then schedule the taxonomy split as follow-up. That option is rejected because it would fix the gate-MOVE reproducer by misclassifying transform `on_error` DIVERT rows as success indicators. The inverse predicate-only choice — continue treating all routed rows as failure indicators — preserves the user-facing bug. A heuristic edge lookup at the predicate/accumulator boundary is also rejected: the producer already knows whether it emitted an intentional gate MOVE or an error-handling DIVERT, and deferring that distinction to a later graph lookup creates an optional/defensive inference surface in Tier-1 status accounting. The enum split, counter split, audit value-domain split, API/web schema updates, and dashboard/UI validation therefore ship together so the behavioral fix has a mechanically recorded producer signal from L0 contracts through the user-facing surface. This is the smallest correct unit under the repo's no-legacy-code and audit-attributability constraints.

**Architecture decision — public API field names and naming rule:** The four Pydantic API contracts intentionally expose `rows_routed_success` and `rows_routed_failure` as public JSON fields. These names are stable for the current public web-API horizon: no compatibility alias, no alternate public spelling, and no later rename without an explicit breaking API decision/ADR and OpenAPI/schema test updates. The names reuse the engine vocabulary deliberately because the L0/L3 predicate mirrors must stay mechanically comparable; translating them to softer API names such as `rows_moved` or `rows_error_routed` would hide the run-status predicate role and recreate the drift class that caused this bug. This does create a naming asymmetry with `RowOutcome.ROUTED_ON_ERROR`: row outcomes name the producer/audit circumstance, while aggregate counters name the run-status predicate role. ADR-018 records that rule explicitly. Do not claim that every future outcome/counter pair should be lexically isomorphic; the default pattern is producer-site outcome discrimination plus predicate-role aggregate naming when an aggregate field is part of the status predicate.

**Architecture decision — upgrade-boundary semantics:** This PR is intentionally not backward-compatible across long-lived runtime state. Pre-split `ProgressEvent.rows_succeeded` values may have included all routed rows in the display-success count; post-split progress excludes `rows_routed_failure`. MCP `outcome_distribution` is backed by `token_outcomes.outcome`, so historical `"routed"` buckets are ambiguous while new `"routed"` and `"routed_on_error"` buckets are distinguishable. These are semantic shifts, not data to silently reinterpret. The operator runbook must say to delete checkpoint files and recreate sessions/Landscape stores before upgrade, or record a named accepted limitation for any preserved legacy audit/progress/MCP evidence.

**Tech Stack:** Python 3.13, dataclasses (`@dataclass(frozen=True, slots=True)` for L0 contracts), Pydantic 2.x (L3 schemas), SQLAlchemy 2.x (sessions DB), pytest, structlog. No new dependencies.

---

## File inventory

Each file's responsibility within the change:

### Created
- **No new source modules.** The `RowOutcome.ROUTED_ON_ERROR` enum value is added to existing `contracts/enums.py`; no new files in `src/`.
- **No new test files.** `tests/integration/audit/test_recorder_routing_events.py` already exists; Task 8 Step 9c extends it with a production-path audit distinguishability test class.
- `docs/architecture/adr/018-producer-site-outcome-discrimination.md` — new ADR promoted out of the ADR-004 amendment. Records the project-wide default: when a producer knows a semantically-distinct terminal circumstance, encode it as a producer-site outcome variant instead of a discriminator field. Also records the naming rule: RowOutcome names producer/audit circumstances; aggregate counters that feed run-status predicates name predicate roles.

### Modified — L0 contracts
- `src/elspeth/contracts/enums.py` — add `RowOutcome.ROUTED_ON_ERROR` enum value.
- `src/elspeth/contracts/results.py` — extend `RowResult.__post_init__` invariant (line 406) to require BOTH `sink_name` AND `error` (a `FailureInfo` instance) to be set for `ROUTED_ON_ERROR` (mirror of `DIVERTED`'s contract — `ROUTED_ON_ERROR` is a failure-handling redirect, not an intentional MOVE; the originating transform error must be captured on the outcome record for single-hop audit attributability per the "Terminology and predicate asymmetry" / Tier-1 audit primacy rationale). NOTE: this is a stronger invariant than existing FAILED's `__post_init__` (which currently relies on the recorder layer to enforce `error_hash`); the new ROUTED_ON_ERROR variant uses the strict offensive guard at construction time so any producer site that forgets to pass `error=FailureInfo(...)` crashes immediately, before the row reaches the recorder.
- `src/elspeth/contracts/run_result.py` — add `rows_routed_success: int` and `rows_routed_failure: int` fields on `RunResult`; remove `rows_routed`. Update `derive_terminal_run_status()` predicate. Update `_check_status_invariant` match-statement. Update `to_dict()`.
- `src/elspeth/contracts/errors.py` — split `rows_routed` on `GracefulShutdownError.__init__` (class declared at line 713; `rows_routed: int = 0` parameter at line 731; `self.rows_routed = rows_routed` assignment at line 739) into the two new fields. NOTE: `_RunFailedWithPartialResultError` lives at `engine/orchestrator/core.py:166` (not in `contracts/errors.py`); it carries `partial_result: RunResult` rather than direct `rows_routed` fields, so the L0 RunResult update flows through it automatically and no edit to that class is required.
- `src/elspeth/contracts/events.py` — split `RunSummary.routed: int = 0` (field at line 141, `require_int(self.routed, "routed", ...)` guard at line 150, docstring at lines 128-130) into `routed_success: int = 0` and `routed_failure: int = 0`. RunSummary is the L0 EventBus event consumed by CLI formatters and external CI integrations; `routed_destinations` (the per-sink count map at line 142) is unchanged. This split is required for symmetric audit-distinguishability through the event channel — the operator-visible CLI display continues to show a SUM (`event.routed_success + event.routed_failure`) for UX continuity, while the JSON output and any future programmatic consumer see the split fields explicitly. Without this split, the event surface still carries the conflated single-counter value while the run record carries the split values, producing inconsistent shapes for any consumer reading both.
- `src/elspeth/contracts/engine.py` — add `RowOutcome.ROUTED_ON_ERROR` to `PendingOutcome._FAILURE_OUTCOMES` (the `ClassVar[frozenset[RowOutcome]]` at line 65, currently containing `{QUARANTINED, FAILED}`). The set's actual function is "outcomes requiring `error_hash` on PendingOutcome" — without this addition, `PendingOutcome(RowOutcome.ROUTED_ON_ERROR, error_hash="<hash>")` raises ValueError ("must not have error_hash") at `__post_init__:84`. Because ROUTED_ON_ERROR is the FIRST outcome variant that both routes through the pending-sink pipeline AND requires `error_hash` (existing FAILURE_OUTCOMES are recorded synchronously without going through `_route_to_sink`), this is a real new requirement on the pending pipeline. Update the docstring (lines 76-81) to clarify the set's role: "outcomes requiring error_hash" not "failure outcomes" — the misleading name is preserved as a class variable but the comment/docstring should describe the actual semantic.

### Modified — L2 engine
- `src/elspeth/engine/orchestrator/types.py` — split `rows_routed` on `ExecutionCounters` (line 177) and `AggregationFlushResult` (line 113) into `rows_routed_success` and `rows_routed_failure`. Update `accumulate_flush_result`, `to_flush_result`, `to_run_result`, `__add__`, `to_dict`.
- `src/elspeth/engine/orchestrator/outcomes.py` — three coupled edits (full code blocks in Task 3 Step 3 and Task 3 Step 4):
  - **`accumulate_row_outcomes` (lines 171-213)** — branch on `RowOutcome.ROUTED` → `rows_routed_success` and add a new `RowOutcome.ROUTED_ON_ERROR` branch → `rows_routed_failure`. Both variants increment `counters.routed_destinations[sink_name]` (the destination map tracks "where rows landed" regardless of routing intent).
  - **`_route_to_sink` (lines 47-70)** — extend signature with `error_hash: str | None = None` keyword + offensive guards (ROUTED_ON_ERROR requires it, ROUTED forbids it). Forward `error_hash` to `PendingOutcome(...)`. Add `import hashlib` at module top.
  - **`accumulate_diversion_into_counters` `_decrement_counter` block (lines 104-118)** — remove the dynamic `getattr` / `setattr` helper and replace it with direct branches on `pending_outcome.outcome`. The replacement must preserve the existing three behaviours for ROUTED outcomes (decrement counter, validate `routed_destinations[sink_name] >= diversion_count`, decrement/delete the destination key) and do the same for `ROUTED_ON_ERROR`. `routed_destinations` cleanup applies to both variants because the accumulator increments the map for both. Do not introduce a unified string-field helper for `rows_routed_success` / `rows_routed_failure`; typed field access is the mechanical guard here.
- `src/elspeth/engine/processor.py` — change the producer site at line 2296 (transform on_error path) from `RowOutcome.ROUTED` to `RowOutcome.ROUTED_ON_ERROR`. Line 2364 (gate `route_to_sink`) stays as `RowOutcome.ROUTED`. Do NOT reuse the existing `error_detail = ... else "unknown_error"` fallback for the new `FailureInfo`: `ROUTED_ON_ERROR` must raise `OrchestrationInvariantError` if `transform_result.reason` is falsy so the audit record never fabricates a deterministic `"unknown_error"` message/hash for a Tier-1 outcome.
- `src/elspeth/engine/orchestrator/core.py` — update ~12 call sites that read `counters.rows_routed`, `result.rows_routed`, or pass `rows_routed=` kwargs. Lines 458-459 (the `match` statement on `RowOutcome` that currently has the "rows_routed: excluded from the predicate" comment), 552, 578, 611, 1579, 1616, 1648, 2147, 2420, 2973, 3208, 3282. The `rows_succeeded + rows_routed` additions at lines 2420 and 2973 (ProgressEvent display) become `rows_succeeded + rows_routed_success` (do not include rows_routed_failure in the progress "successful" count — those are operator-visible failures). ADR-004 and the runbook must call out the upgrade-boundary semantic shift: old progress events may have counted both routed forms as display-success; new progress excludes on_error-routed failures.

### Modified — L2 engine (RowOutcome consumers)
- `src/elspeth/core/landscape/data_flow_repository.py` — line 230 (`elif outcome == RowOutcome.ROUTED:` inside `_validate_outcome_fields`) and line 274 (`elif outcome == RowOutcome.DIVERTED:`) — extend the contract-validation block to require BOTH `sink_name` AND `error_hash` for `ROUTED_ON_ERROR` (mirror of the existing `DIVERTED` contract at lines 274-282, which enforces both fields). The new branch matches DIVERTED's shape because both outcomes are failure-handling redirects with an originating error. Preserve the existing terminal `else: raise ValueError(...)` at the end of `_validate_outcome_fields`; if a worker's local branch lacks it, add it while in this area so unknown future `RowOutcome` variants cannot fall through silently. NO Landscape schema change is required: see the "Landscape audit distinguishability" resolution under Task 4 Step 2 for the in-scope analysis (the new `RowOutcome.ROUTED_ON_ERROR` enum value is persisted directly to `token_outcomes.outcome` as a StrEnum, providing primary distinguishability; `RoutingEvent.mode` at `src/elspeth/contracts/audit.py:398` provides a secondary edge-level cross-check; `error_hash` on the outcome record provides the single-hop audit answer to "what error triggered the rerouting?").
- `src/elspeth/core/landscape/model_loaders.py` — lines 547-579 (`TokenOutcomeLoader.load` Tier-1 read-side guards): keep the existing `COMPLETED/ROUTED` sink-name guard and add a separate `ROUTED_ON_ERROR` sink-name guard so the current error-message contract for existing outcomes does not churn. Also expand the existing `DIVERTED` `error_hash` guard to require `error_hash` for `ROUTED_ON_ERROR` as well. This mirrors the write-side `data_flow_repository.py` contract. A stored/corrupt `token_outcomes.outcome='routed_on_error'` row with `sink_name` but `error_hash IS NULL` must crash during read-side audit loading; accepting it would break the Tier-1 "what error triggered this reroute?" attribution guarantee.
- `src/elspeth/testing/__init__.py` — three sites (verify line numbers with `grep -n "rows_routed" src/elspeth/testing/__init__.py`; observed 2026-05-02):
  - line 426: `make_run_result()` factory has `rows_routed: int = 0` parameter and forwards `rows_routed=rows_routed` to `RunResult(...)` at line 445.
  - line 460: `make_flush_result()` factory has the same shape, forwarding to `AggregationFlushResult(rows_routed=...)` at line 474.
  - line 518: `_SINK_OUTCOMES = {RowOutcome.COMPLETED, RowOutcome.ROUTED, RowOutcome.COALESCED}` expands to include `ROUTED_ON_ERROR`.
  - lines 616-641 (`make_run_summary` factory): `routed: int = 0` parameter at the helper signature and `routed=routed` forwarding at line 641. Replace with two parameters `routed_success: int = 0, routed_failure: int = 0` and forward both to `RunSummary(routed_success=routed_success, routed_failure=routed_failure)`. This factory is used in the test suite to construct synthetic RunSummary events.

  The three factory helpers (`make_run_result`, `make_flush_result`, `make_run_summary`) are widely used across the test suite — every test using them breaks at construction time after Task 2's L0 update unless the factories are updated in lockstep.

### Modified — L3 application (CLI/TUI)
- `src/elspeth/cli_formatters.py` — three `event.routed` reads on `RunSummary` (verify with `grep -n "event\.routed\b" src/elspeth/cli_formatters.py`; observed 2026-05-02): line 58 (`if event.routed > 0:`), line 61 (display string `f" | →{event.routed:,} routed"`), line 141 (JSON output dict `"routed": event.routed`). Update strategy:
  - **Console display (lines 58, 61):** sum the two new fields for the operator-visible string. Replace `if event.routed > 0:` with `if (event.routed_success + event.routed_failure) > 0:` and the f-string body with `f" | →{event.routed_success + event.routed_failure:,} routed"`. UX continuity is the goal — operators reading the console output should see the same total as before. The per-sink breakdown (`routed_destinations` at line 59) is unchanged.
  - **JSON output (line 141):** emit BOTH new fields explicitly (`"routed_success": event.routed_success, "routed_failure": event.routed_failure,`). Drop the old `"routed"` key entirely — the JSON consumer is programmatic, not human, and the no-legacy-code policy forbids a transitional alias.

### Modified — L3 web (sessions DB + protocol)
- `src/elspeth/web/sessions/models.py` — line 141: drop `rows_routed` Column; add `rows_routed_success` and `rows_routed_failure` Columns.
- `src/elspeth/web/sessions/protocol.py` — line 203 (RunRecord field), line 374 (constructor parameter): split `rows_routed` into the two new fields.
- `src/elspeth/web/sessions/service.py` — lines 587, 630, 684-685, 910, 1303: update CRUD call sites to read/write the two new columns instead of `rows_routed`.

### Modified — L3 web (execution layer)
- `src/elspeth/web/execution/schemas.py` — lines 109-153 (`_validate_row_decomposition`): update the `sum_terminal` formula to include both new counters. Lines 179-260 (`_check_status_row_count_invariant`): mirror the L0 predicate update — add `rows_routed_failure` to the failure indicator and `rows_routed_success` to the success indicator. FOUR Pydantic response models carry `rows_routed: int = Field(default=0, ge=0)` — verify exact line numbers with `grep -n "rows_routed" src/elspeth/web/execution/schemas.py` before editing (current observed: 274, 295, 533, 591). The model at line 274 is the primary status response; the model at 295 is `CancelledData` (cancelled-run payload); 533 and 591 are the historical/results-export models. All four must drop `rows_routed` and add `rows_routed_success` / `rows_routed_failure`. Update the corresponding `model_validator` invocations (e.g. lines 284, 563, 603) to pass the two new fields into `_check_status_row_count_invariant` / `_validate_row_decomposition`.
- `src/elspeth/web/execution/routes.py` — lines 129, 148, 587: update API route handlers to read the new fields.
- `src/elspeth/web/execution/service.py` — eight `rows_routed` sites:
  - line 133 (`_structural_failure_message` definition): update the synthetic error message phrasing now that gate-routed pipelines no longer hit the FAILED-from-row-shape path. The message becomes: `f"No row reached a success path (rows_processed={rows_processed}, rows_succeeded=0, rows_routed_success=0). All rows either failed terminally or were routed via on_error to a failure sink. Inspect /diagnostics for per-row failure details."`
  - line 510 (`rows_routed=run.rows_routed,` — RunRecord readback): replace with `rows_routed_success=run.rows_routed_success, rows_routed_failure=run.rows_routed_failure,`
  - lines 637, 658 (`CancelledData(rows_processed=0, rows_failed=0, rows_routed=0)`): replace `rows_routed=0` with `rows_routed_success=0, rows_routed_failure=0`
  - lines 862, 886, 936 (completed-event constructions reading `result.rows_routed`): replace `rows_routed=result.rows_routed,` with `rows_routed_success=result.rows_routed_success, rows_routed_failure=result.rows_routed_failure,`
  - lines 954, 967 (graceful-shutdown event constructions reading `gse.rows_routed`): replace `rows_routed=gse.rows_routed,` with `rows_routed_success=gse.rows_routed_success, rows_routed_failure=gse.rows_routed_failure,`
  - Verify the final list with: `grep -n "rows_routed" src/elspeth/web/execution/service.py` (expected: zero matches after this task).

### Modified — L3 web frontend / dashboard
- `src/elspeth/web/frontend/src/types/index.ts` — extend `Run`, `RunEventProgress`, `RunEventCompleted`, `RunEventCancelled`, and `RunProgress` with `rows_routed_success: number` and `rows_routed_failure: number` where the corresponding API/WebSocket payload carries terminal row counters. This file currently has no `rows_routed` or `rowsRouted` identifier, but the dashboard still needs explicit type coverage for the new API fields; "no old frontend field" is not the same as "UI surface validated".
- `src/elspeth/web/frontend/src/stores/executionStore.ts` — when terminal/progress events carry the split counters, preserve them in `progress` and the updated run list instead of dropping them at the store boundary. Do not fabricate counts on events that do not carry them; initialise running progress with both fields as zero and update from payloads when present.
- `src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx` — add the dashboard UI gate in Task 8 Step 9e. This is the frontend validation for the surface where the user saw the incorrect failed status. The repo has Vitest + React Testing Library, not Playwright; use the existing DOM test stack rather than adding a browser dependency for a status-text regression.

### Modified — Tests
- `tests/unit/contracts/test_run_result.py` — REPLACE the existing `rows_routed=N` assertions and the test at lines 289-295 that asserts FAILED-on-rows_routed-shape. Add nineteen new tests/properties in `TestRunStatusRowsRoutedSplitPredicate`: six canonical shapes (gate MOVE → COMPLETED, on_error DIVERT → FAILED, mixed → COMPLETED_WITH_FAILURES, empty → EMPTY, resume → COMPLETED, resume-shaped zero-processed mixed-indicator → COMPLETED_WITH_FAILURES), three additional mixed-counter positive shapes (succeeded+on_error_failure, routed_success+rows_failed, canonical-FAILED-via-rows_failed-only), three matrix tests covering `rows_quarantined` / `rows_coalesce_failed` crossed with the new routed counters, one Hypothesis biconditional round-trip property (`RunResult(status=derive_terminal_run_status(**c), **c)` must not raise), and six negative-invariant tests pinning the `_check_status_invariant` raise-paths (COMPLETED-without-success-indicator, COMPLETED-with-failure-indicator, COMPLETED_WITH_FAILURES-without-each-indicator, EMPTY-with-rows-processed, EMPTY-with-success-indicator). The negative tests use `pytest.raises(ValueError, match=...)` against the dataclass `__post_init__` to confirm the invariant rejects each forbidden shape.
- `tests/unit/engine/orchestrator/test_outcomes.py` — eight sites (verify with `grep -n "rows_routed\|\\brouted\\b" tests/unit/engine/orchestrator/test_outcomes.py`; observed 2026-05-02): line 118 (`assert counters.rows_routed == 1`), line 257 (comment about parameter ordering — `< failed < forked < quarantined < routed` — update to mention both new variants), line 273 (the `_assert_counters` HELPER SIGNATURE has `routed: int = 0` parameter — split into `routed_success: int = 0, routed_failure: int = 0` parameters and update the assertion at line 283 from `counters.rows_routed == routed` to two parallel assertions on the new fields), line 300 (test method name `test_routed_only_increments_routed` — rename to `test_routed_only_increments_routed_success` to reflect the new gate-MOVE semantics), line 308 (helper call `self._assert_counters(counters, routed=1)` — update to `routed_success=1`), line 568 (`counters.rows_routed = 1` direct assignment — split into the two new fields), line 578 (`counters.rows_routed == 0` assertion). The `_assert_counters` helper is used across many tests in the file; updating its signature is the highest-leverage change. Add new tests that exercise `RowOutcome.ROUTED_ON_ERROR` and verify the accumulator increments `rows_routed_failure` (uses Mock-based `_make_result`, so no `error=` field needed in the fixture per the existing pattern).
- `tests/unit/engine/orchestrator/test_types.py` — replace `rows_routed=` kwargs in fixture construction with `rows_routed_success=`/`rows_routed_failure=`.
- `tests/unit/engine/orchestrator/test_aggregation.py` — replace `result.rows_routed` reads at lines 632 and 970.
- `tests/integration/pipeline/orchestrator/test_t18_characterization.py` — line 201: replace `result.rows_routed == 0` with assertions on the two new counters.
- `tests/integration/pipeline/orchestrator/test_graceful_shutdown.py` — line 498: replace `exc_info.value.rows_routed == 0` with assertions on the two new exception fields.
- `tests/integration/pipeline/test_composer_runtime_agreement.py` — INVERT the locked-in-buggy-behavior test at lines 2492-2543. Existing test uses ConditionalErrorTransform which routes via `on_error` → after the split, this still classifies as FAILED (via `rows_routed_failure > 0` and no success indicator), but the assertions and docstring update. Add a new companion test for the gate-routed MOVE shape using a config gate that routes every row to a named sink.
- `tests/unit/contracts/test_freeze_regression.py` — line 342: `PendingOutcome(outcome=RowOutcome.ROUTED)` requires a companion test for the new ROUTED_ON_ERROR variant. Because Task 2 Step 9c admits ROUTED_ON_ERROR to `PendingOutcome._FAILURE_OUTCOMES`, the dataclass `__post_init__` now treats it asymmetrically with ROUTED — error_hash is REQUIRED for ROUTED_ON_ERROR, FORBIDDEN for ROUTED. Add a parallel construction test asserting `PendingOutcome(outcome=RowOutcome.ROUTED_ON_ERROR, error_hash="0123456789abcdef")` succeeds and that `PendingOutcome(outcome=RowOutcome.ROUTED_ON_ERROR)` (missing error_hash) AND `PendingOutcome(outcome=RowOutcome.ROUTED_ON_ERROR, error_hash="")` (empty) both raise ValueError. This pins the new invariant alongside the existing ROUTED freeze-regression coverage.
- `tests/unit/cli/test_cli_formatters.py` — RunSummary fixtures and assertions (verify with `grep -n "routed\b" tests/unit/cli/test_cli_formatters.py`; observed 2026-05-02): six fixture sites at lines 28, 51, 70 (RunSummary fixture without `routed=`, just verify it stays valid), 95, 151, 160 (and any others the grep finds). Each `routed=N` kwarg becomes `routed_success=N_s, routed_failure=N_f` (split N to the appropriate variant based on test intent — for tests that don't differentiate, assume the test exercises gate MOVE and use `routed_success=N, routed_failure=0`). The assertion at line 36 (`"→3 routed (sink_a:2, sink_b:1)"`) and similar string-match assertions remain valid because the console display still shows a sum. The JSON-shape assertion at lines 176-177 expects `"routed": 1` — replace with `"routed_success": 1, "routed_failure": 0` (or split per test intent).
- `tests/integration/pipeline/orchestrator/test_execution_loop.py` — line 398: `assert summary.routed == 0` becomes `assert summary.routed_success == 0 and summary.routed_failure == 0`.
- `tests/integration/web/test_execute_pipeline.py` — ADD a new test class `TestGateRoutedPipelineExecution` with a single test `test_gate_routed_pipeline_classifies_as_completed_via_api` that exercises the user's reproducer through the FastAPI surface (POST `/api/sessions/{id}/execute` → poll GET `/api/runs/{run_id}`) end-to-end. This is the API acceptance gate — the engine-layer test in `test_composer_runtime_agreement.py` confirms the predicate fix at L0; this web-layer test confirms the L3 row-count predicate mirror (`_check_status_row_count_invariant`), the sessions read-side Tier-1 status guards on `RunRecord.__post_init__` (status enum / `finished_at` / `landscape_run_id` / `error` checks — NOT a row-count predicate mirror; see Constraints "NOT a mirror site" subsection), the `_structural_failure_message` helper, and the HTTP API contract all line up. Full structure in Task 8 Step 9b.
- `src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx` — ADD a dashboard DOM test `renders_gate_routed_completed_runs_without_failure_alert` (Task 8 Step 9e). Seed `useExecutionStore` with a run shaped like the fixed API response (`status: "completed"`, `rows_processed > 0`, `rows_failed == 0`, `rows_routed_success > 0`, `rows_routed_failure == 0`, `error: null`) and assert that the Runs dashboard renders `completed`, does not render a failure alert, and never shows the old structural-failure message. This is the UI gate for the dashboard surface where the user observed the wrong failed status; do not close `elspeth-71520f5e30` on the API test alone.
- `tests/integration/audit/test_recorder_routing_events.py` — existing file, extend with `TestRoutingEventDistinguishability`. Pins the Tier-1 audit-trail distinguishability between MOVE-routed and DIVERT-routed tokens via direct `landscape_db` queries on `token_outcomes` and `routing_events`, plus an `explain()` round-trip. Verifies the scoped producer signals (outcome value, error_hash presence, RoutingMode on the corresponding producer edge) Task 4 Step 2's resolution depends on. Per CLAUDE.md "no inference — if it's not recorded, it didn't happen": the audit-distinguishability claim must be PINNED EMPIRICALLY by tests that fail on pre-PR code, otherwise a future regression silently drops one of the three signals and the user-facing P1 re-opens. Mirrors construction patterns from `test_recorder_nodes.py:110-266` (RoutingMode usage) and `test_recorder_explain.py:545+` (GateSettings construction). Full structure in Task 8 Step 9c.
- `tests/integration/pipeline/test_resume_comprehensive.py` — ADD a new test method `test_resume_gate_routed_pipeline_classifies_as_completed` to the existing `TestResumeComprehensive` class (or sibling). Exercises the resume code path end-to-end with a gate-routed pipeline shape, pinning that the `derive_terminal_run_status` call at `core.py:3247` (the resume site, covered by Task 5 Step 2e) correctly accumulates `rows_routed_success` from the actual resume code path — NOT just from a `RunResult` constructor as the Task 1 unit test does. Uses the existing `resume_test_env` fixture pattern (`db`, `checkpoint_manager`, `recovery_manager`, `payload_store`, `checkpoint_config`, `tmp_path`) and the `_setup_failed_run` helper at `test_resume_comprehensive.py:188-242`. Full structure in Task 8 Step 9d.

### Modified — Tests (counter-shape and RowOutcome invariant sweep targets enumerated 2026-05-02)

The following fourteen counter-shape test targets contain ~145 counter-shape sites plus MCP outcome-distribution semantics that the executing agent should sweep BEFORE the final gates run to avoid hitting `pytest -x` cold with ~140 failures and no priority order. The additional RowOutcome enum/invariant/audit-contract targets below are equally required; they are mostly invisible to `rows_routed\b` greps because they fail through enum membership, hand-rolled outcome categories, or read-side audit guards instead of counter field names. Listed in suggested execution order (smallest blast radius first; the property test last because its strategies need redesign, not mechanical rename):

- `tests/unit/contracts/test_enums.py:13-24` — hard-coded `terminal_outcomes` set in `test_terminal_mappings`. **INVISIBLE TO `rows_routed\b` GREP** because the symbol is `RowOutcome.ROUTED`, not `rows_routed`. Add `RowOutcome.ROUTED_ON_ERROR` to the set at line 24 (preserves alphabetical order alongside `RowOutcome.ROUTED` at line 15). Failure mode if missed: assertion failure with `set != set` diff because `RowOutcome.ROUTED_ON_ERROR.is_terminal == True` but the test's hand-rolled set doesn't include it.
- `tests/unit/web/sessions/test_models.py` — single site at line 91. SQLAlchemy column-presence assertion. Replace `rows_routed` column reference with two: `rows_routed_success` and `rows_routed_failure`.
- `tests/unit/web/sessions/test_schema.py` — single site at line 31. Schema-validation test. Replace as above.
- `tests/unit/web/sessions/test_routes.py` — single site at line 627. Replace bare `rows_routed=N` kwarg or `.rows_routed` read with the two new fields.
- `tests/unit/web/sessions/test_service.py` — two sites at lines 642 and 648. CRUD assertion shape; replace as above.
- `tests/unit/web/execution/test_websocket.py` — five sites at lines 168, 181, 225, 238, 254. WebSocket payload assertions on rows_routed; replace assertion shape with the two new field assertions.
- `tests/unit/web/sessions/test_protocol.py` — eleven sites at lines 137, 186, 207, 226, 245, 264, 283, 303, 322, 341, 360. RunRecord constructor / readback tests. Each `rows_routed=N` kwarg becomes `rows_routed_success=N_s, rows_routed_failure=N_f` (split N per the test's intent; tests that don't differentiate use `rows_routed_success=N, rows_routed_failure=0`).
- `tests/unit/web/execution/test_routes.py` — thirteen sites at lines 282, 343, 422, 645, 672, 699, 741, 775, 788, 807, 908, 920, 937. API-route handler tests. Each call site is either a fixture construction or an assertion on response payload shape; replace the field name in both kinds.
- `tests/unit/web/execution/test_service.py` — fifteen sites at lines 209, 217, 406, 471, 703, 761, 779, 818, 1127, 1189, 1254, 1316, 2419, 2625, 2692. Service-layer test fixtures and assertions. Note the lines 2419, 2625, 2692 cluster — likely the structural-failure-message tests; verify the new message phrasing from Task 7 Step 5 (`web/execution/service.py:133` `_structural_failure_message`) matches the assertions.
- `tests/unit/web/execution/test_schemas.py` — twenty-six sites at lines 323, 342, 364, 368, 749, 768, 786, 802, 821, 838, 843, 859, 875, 891, 913, 929, 947, 964, 978, 983, 997, 1011, 1067, 1104, 1110, 1165. Pydantic schema validation tests — the largest single-file sweep target. Many fixtures, many `model_validator` invocation tests, many shape assertions for the four Pydantic response models (`CompletedData`, `CancelledData`, `RunStatusResponse`, `RunResultsResponse`). The negative-validation tests (assertions that `_check_status_row_count_invariant` raises on inconsistent shapes) need updates to use the new field names. Add a public API field-name stability test that introspects each model's `model_json_schema()["properties"]` and asserts `rows_routed_success` and `rows_routed_failure` are present while `rows_routed` is absent. This pins the public JSON names recorded in ADR-018.
- `tests/integration/cli/test_cli.py` — line 185 contains a TEST METHOD NAME `test_invalid_rows_routed_to_quarantine_sink` and line 126 contains a docstring reference to the same. **Invisible to the `rows_routed\b` grep** because the substring appears inside an identifier (test_invalid_**rows_routed**_to_quarantine_sink). After the split, this test exercises the on_error→quarantine path which is now `rows_routed_failure`; rename to `test_invalid_rows_routed_failure_to_quarantine_sink` and update the docstring at line 126 accordingly. Verify whether the test body asserts on `rows_routed` field reads (run `sed -n '180,250p' tests/integration/cli/test_cli.py` to inspect); if so, update those too.
- `tests/unit/contracts/test_events.py:346` — single `routed=3` kwarg on a `RunSummary(...)` construction inside `TestRunSummaryIntValidation`. **Caught only by the bare-`routed` complementary grep**, not by `rows_routed\b`. Replace with `routed_success=3, routed_failure=0` (single-variant fixture; the test exercises require_int validation, not MOVE/DIVERT semantics). Also verify with `grep -n "RunSummary\|routed" tests/unit/contracts/test_events.py` whether other `RunSummary(...)` constructions in the file's TestRunSummaryIntValidation block (around lines 282-330) need the `routed_success`/`routed_failure` parameters — they currently omit `routed=` entirely, so they should still type-check after Task 2 Step 9b makes both fields default-zero. If a test specifically validates `require_int` on the new fields, add parallel cases for `routed_success` and `routed_failure`.
- `tests/unit/mcp/test_analyzer_queries.py` and `tests/unit/mcp/analyzers/test_reports.py` — add MCP outcome-distribution coverage for the new value-domain. Extend `TestGetRunSummary.test_summary_outcome_distribution` (or an adjacent test) with a `RowOutcome.ROUTED_ON_ERROR` token outcome and assert `result["outcome_distribution"]["routed_on_error"] == 1`. Extend the report analyzer test that currently checks terminal/non-terminal outcome rows so one mocked/report row uses `"routed_on_error"` and verifies the serialized `outcome_distribution` preserves that bucket. Add a short assertion comment that MCP does not split historical `"routed"` rows; the runbook/ADR handles the upgrade-boundary limitation.
- `tests/unit/core/landscape/test_model_loaders.py` — read-side audit-contract tests for `TokenOutcomeLoader.load`. Add `"routed_on_error": {"sink_name": "failsink", "error_hash": "e" * 16}` to `_OUTCOME_REQUIRED_FIELDS`, then add three tests next to the existing DIVERTED regression block at lines 1548-1574: valid `routed_on_error` loads with `sink_name` + `error_hash`; missing `sink_name` raises `AuditIntegrityError` matching `ROUTED_ON_ERROR requires sink_name`; missing `error_hash` raises `AuditIntegrityError` matching `ROUTED_ON_ERROR requires error_hash`. These tests pin the read-side half of the Tier-1 contract and must fail before Task 4 Step 3 expands the loader's guards.
- `tests/property/contracts/test_row_result_sink_invariant.py:31-86` — update the hard-coded sink outcome categories for the new runtime invariant. `ROUTED_ON_ERROR` is sink-targeting, but it also requires a real `FailureInfo`, so the positive sink-name property cannot simply add the enum to `SINK_OUTCOMES` and keep constructing `RowResult(..., error=None)`. Add a helper such as `_error_for(outcome: RowOutcome) -> FailureInfo | None` that returns `FailureInfo(exception_type="TransformError", message="boom")` only for `ROUTED_ON_ERROR`; include `ROUTED_ON_ERROR` in `SINK_OUTCOMES`; recompute `NON_SINK_OUTCOMES` from that set; and update docstrings from `COMPLETED/ROUTED/COALESCED` to include `ROUTED_ON_ERROR`. Without this, the property suite treats `ROUTED_ON_ERROR` as non-sink and constructs a shape the new invariant correctly rejects.
- `tests/unit/engine/test_row_outcome.py:16-89` — update `_SINK_OUTCOMES` and the all-enum RowResult construction helpers for the new sink+failure variant. Add `ROUTED_ON_ERROR` to the sink category, but pass `error=FailureInfo(exception_type="TransformError", message="boom")` whenever the loop constructs that outcome. The tests at lines 29-43, 45-53, and 82-89 iterate every `RowOutcome`; if they only set `sink_name`, the planned `RowResult.__post_init__` invariant will reject `ROUTED_ON_ERROR` for missing `FailureInfo`.
- `tests/property/audit/test_terminal_states.py:528-542` — update the hard-coded terminal outcome expectation from ten outcomes to eleven and add `RowOutcome.ROUTED_ON_ERROR` to the expected set. The new enum is terminal by design; leaving this list unchanged creates a deterministic property failure unrelated to counter-shape renames.
- `tests/property/audit/test_recorder_properties.py:624-685` — update the record-outcome required-field tables for the new audit contract. Add a missing-field case that proves `ROUTED_ON_ERROR` rejects missing `sink_name`, add a second missing-field case that proves it rejects missing `error_hash` when `sink_name` is present, and add an accepted-fields case with both `{"sink_name": "failsink", "error_hash": stable_hash({"reason": "transform"})}`. This pins the write-side `record_token_outcome` contract alongside the read-side loader tests.
- `tests/unit/contracts/test_engine_contracts.py:97-122` — mirror the `PendingOutcome` ROUTED_ON_ERROR cases added in `tests/unit/contracts/test_freeze_regression.py`. Add one success test with a non-empty `error_hash`, one missing-hash rejection, one empty-string rejection, and one ROUTED-with-error-hash rejection if not already present. The contract suite and freeze-regression suite should both prove the asymmetry: `ROUTED` forbids `error_hash`; `ROUTED_ON_ERROR` requires it.
- `tests/property/engine/test_orchestrator_lifecycle_properties.py` — at least THIRTY-EIGHT distinct sites across multiple test classes (verify with `grep -n "rows_routed\|\\brouted\\b" tests/property/engine/test_orchestrator_lifecycle_properties.py | wc -l`; observed 2026-05-02). The sites fall in three categories that need DIFFERENT treatment:
  - **Strategy-generator redesign (NOT mechanical rename) — separate arbitrary engine-counter shapes from validated response shapes:** Lines 67 and 84 are `rows_routed=draw(counter_values)` inside the strategies `aggregation_flush_results` and `execution_counters`. ALSO line 626 is a SECOND `routed=st.integers(...)` Hypothesis `@given` parameter inside a different test class (`test_mixed_outcomes_conservation`). For pure monoid / accumulation properties that never construct a `COMPLETED` `RunResult` or a Pydantic web response model, the split fields may still be independently drawn and asserted independently. For any property path that constructs a validated completed/empty/mixed terminal shape (`RunResult(status=COMPLETED...)`, `CompletedData`, `RunStatusResponse`, or anything that calls `_validate_row_decomposition`), independent draws are forbidden because `rows_processed < sum_terminal` can crash the harness at model construction. Use the constrained draw helper below for those paths:

    ```python
    @st.composite
    def completed_row_counter_shapes(draw) -> dict[str, int]:
        """Counters valid for COMPLETED / completed API response construction."""
        rows_succeeded = draw(st.integers(min_value=0, max_value=10))
        rows_failed = draw(st.integers(min_value=0, max_value=10))
        rows_routed_success = draw(st.integers(min_value=0, max_value=10))
        rows_routed_failure = draw(st.integers(min_value=0, max_value=10))
        rows_quarantined = draw(st.integers(min_value=0, max_value=10))
        rows_diverted = draw(st.integers(min_value=0, max_value=10))
        rows_coalesce_failed = draw(st.integers(min_value=0, max_value=10))
        terminal_sum = (
            rows_succeeded
            + rows_failed
            + rows_routed_success
            + rows_routed_failure
            + rows_quarantined
            + rows_diverted
            + rows_coalesce_failed
        )
        rows_processed = draw(
            st.integers(min_value=terminal_sum, max_value=terminal_sum + 10)
        )
        return {
            "rows_processed": rows_processed,
            "rows_succeeded": rows_succeeded,
            "rows_failed": rows_failed,
            "rows_routed_success": rows_routed_success,
            "rows_routed_failure": rows_routed_failure,
            "rows_quarantined": rows_quarantined,
            "rows_diverted": rows_diverted,
            "rows_coalesce_failed": rows_coalesce_failed,
        }
    ```

    If a property needs to assert `COMPLETED` specifically, add an `assume(rows_succeeded > 0 or rows_routed_success > 0)` guard or draw one success indicator as positive. If a property needs `FAILED`, draw `rows_processed >= failure_sum` and at least one failure indicator while leaving success indicators zero. Do not rely on Pydantic constructor crashes as a Hypothesis filter.
  - **Conservation/property assertions across multiple test classes (mechanical, but spread widely):** lines 113, 141, 167, 184, 201, 217 (the `TestFlushResultMonoidProperties` class's commutativity/associativity/identity assertions); lines 251, 257, 275, 287, 313, 320 (zero-state and accumulation assertions in `TestExecutionCountersFlushSemantics`); lines 365, 372, 382, 399, 426 (sequence / sum-equality property assertions); lines 517, 539, 543, 588 (single-outcome `test_*_only_increments_*` properties — note the test name pattern, e.g. `test_routed_only_increments` may need rename to `test_routed_only_increments_routed_success`); lines 630, 639, 644, 645 (the `test_mixed_outcomes_conservation` body using the `routed` parameter — rename parameter and update the conservation-sum assertion at line 644); line 655 (`test_routed_destinations_count_per_sink`); line 674 (test fixture using `rows_routed=1`). Each follows the standard split pattern.
  - **Conservation invariant — special attention.** The conservation assertion at line 644 (`total_increments == completed + failed + routed + quarantined`) MUST become `total_increments == completed + failed + routed_success + routed_failure + quarantined` to preserve the property-test invariant. If the strategy at line 626 draws `routed_success` and `routed_failure` independently, the conservation assertion sums both. Failing to update this assertion in lockstep with the strategy would silently weaken the property: the test would still pass on Hypothesis-drawn shapes, but it would no longer be testing total conservation — an invisible regression in test coverage.

  Run `pytest tests/property/engine/test_orchestrator_lifecycle_properties.py -v --hypothesis-show-statistics` after the redesign to confirm property invariants still hold; if Hypothesis finds a counterexample within the first 200 examples, the strategy redesign has shifted the property semantics and needs review before proceeding.

After this enumeration, run a final per-file grep gate before Task 8 Step 15:

```bash
for f in tests/unit/contracts/test_enums.py \
         tests/unit/contracts/test_events.py \
         tests/unit/web/sessions/test_models.py \
         tests/unit/web/sessions/test_schema.py \
         tests/unit/web/sessions/test_routes.py \
         tests/unit/web/sessions/test_service.py \
         tests/unit/web/execution/test_websocket.py \
         tests/unit/web/sessions/test_protocol.py \
         tests/unit/web/execution/test_routes.py \
         tests/unit/web/execution/test_service.py \
         tests/unit/web/execution/test_schemas.py \
         tests/integration/cli/test_cli.py \
         tests/property/engine/test_orchestrator_lifecycle_properties.py; do
  hits=$(grep -cE "rows_routed\b|\.routed\b|routed=" "$f" 2>/dev/null || echo 0)
  if [ "$hits" -gt "0" ]; then
    echo "REMAINING $hits sites in $f"
    grep -nE "rows_routed\b|\.routed\b|routed=" "$f" | grep -v "routed_success\|routed_failure\|routed_destinations\|routed (sink\|routed_summary\|routed rows\|routed via\|routed to "
  fi
done
```

Expected: zero remaining counter-shape sites across the listed files. Hits in the test_enums.py terminal_outcomes set are caught by the membership-test failure during the regular pytest run, not the grep; MCP analyzer semantics are validated by their dedicated assertions rather than this grep.

### Modified — L3 MCP/analyzers
- `tests/unit/mcp/test_analyzer_queries.py` and `tests/unit/mcp/analyzers/test_reports.py` — add/update MCP analyzer assertions so `outcome_distribution` exposes a distinct `"routed_on_error"` bucket when the Landscape token outcomes contain that enum value. Also add one legacy-boundary assertion/comment in the existing summary distribution tests: historical `"routed"` buckets are not split by the MCP layer and must be treated as legacy ambiguous unless the Landscape DB was recreated at upgrade time. The MCP surface is intentionally a distribution of stored outcome values, not a migration shim.

### Modified — Docs
- `docs/architecture/adr/004-adr-explicit-sink-routing.md` — append a short "Counter split (elspeth-5069612f3c, 2026-05-02)" amendment containing local paragraphs for: (1) the counter split mechanism itself (rows_routed → success/failure, ROUTED_ON_ERROR enum variant, `token_outcomes.outcome` audit-trail attributability); (2) the predicate asymmetry between `RowOutcome.DIVERTED` (sink-write infrastructure failure, NOT a failure indicator) and `rows_routed_failure` (transform-side data failure, IS a failure indicator), with the rationale grounded in failsink absorption semantics; (3) the intentionally-conflated status of `routed_destinations` as a per-sink landed-count map, not a MOVE/DIVERT intent map; and (4) the upgrade-boundary semantic shift for `ProgressEvent.rows_succeeded` and MCP `outcome_distribution`. The project-wide producer-site discrimination default belongs in ADR-018; ADR-004 should cross-reference ADR-018 rather than re-establishing the default locally.
- `docs/architecture/adr/README.md` — add ADR-018 to the index.
- `docs/contracts/token-outcomes/00-token-outcome-contract.md` — add a row to the outcome contract table (line 24+) for `ROUTED_ON_ERROR`.
- `docs/runbooks/database-maintenance.md` — add a "Rows-routed counter split deployment note (2026-05-02)" subsection that treats the Landscape audit DB and checkpoint files as semantically stale after this change unless they are archived/deleted/recreated. The note must state that old `token_outcomes.outcome='routed'` rows predate the `ROUTED_ON_ERROR` split and are legacy-ambiguous; operators may only preserve them with an explicit audit limitation, not reinterpret them as MOVE-only rows. It must also state that pre-upgrade `ProgressEvent.rows_succeeded` evidence and MCP `outcome_distribution["routed"]` evidence cross a semantic boundary and require date/commit-context qualification.

---

## Constraints

- **Migration / stale-semantics strategy: delete the old databases, Landscape audit DB, and checkpoint files for this semantic split.** Per project no-tech-debt policy, the ONLY migration strategy for sessions DB / audit DB schema changes is operator-deletes-the-old-DB. There is no Alembic, no in-place migration script, no schema-version probe at startup, no compatibility shim, no graceful-degradation path against stale DBs. Sessions DB uses `metadata.create_all` (`src/elspeth/web/sessions/schema.py:39`); schema changes propagate by editing the model declaration only. Operator action for the sessions DB: `rm <sessions.db path>` (or equivalent) before redeploying. `metadata.create_all`'s add-but-don't-drop behaviour is acceptable because the operator-mandated step is delete-then-recreate, not migrate-in-place. This PR also has a Landscape **semantic** migration even though it has no SQL schema migration: historical `token_outcomes.outcome='routed'` rows were produced before the `ROUTED_ON_ERROR` taxonomy existed, so they may include both gate MOVE rows and transform on_error DIVERT rows. Checkpoint files/payloads are also stale because they may serialize the old single-counter `RunResult`, `ExecutionCounters`, `AggregationFlushResult`, or progress shape across the upgrade boundary. New code must not reinterpret those historical rows or checkpoint payloads as MOVE-only audit evidence. Operator action for dev/staging/pre-1.0 deployments: stop the service, archive the current Landscape audit DB and configured checkpoint files/directories if retention is required, delete/recreate the Landscape audit DB, sessions DB, and checkpoint files/directories, and then redeploy. If an environment must preserve the old Landscape DB or progress/MCP evidence for retention, the release notes/runbook must explicitly mark pre-split `routed` outcomes, pre-split `ProgressEvent.rows_succeeded`, and pre-split MCP `outcome_distribution["routed"]` as **legacy ambiguous** and accepted audit-limitation evidence; preserving old rows without that limitation is a merge blocker. This applies uniformly: dev, staging, and any pre-1.0 production deployment. Reviewers asking "what about an operator running new code against old state?" — the answer is "the operator deletes the old state, or records a named accepted audit limitation for legacy `routed` rows/progress/MCP evidence before preserving it." Do NOT add defensive code for stale-schema detection.
- **No legacy code (CLAUDE.md).** Drop `rows_routed` entirely in the same PR. No shims, no deprecation, no `rows_routed = rows_routed_success + rows_routed_failure` derived alias. All callers update in lockstep.
- **No defensive programming (CLAUDE.md).** Do NOT add a discriminator field on `RowResult` or do edge-lookup at the accumulator. The producer-site enum split is type-discriminating — every `match` / `if` on `RowOutcome` gets a compiler nudge to handle both variants.
- **Tier 1 invariants stay strict.** The biconditional invariant in `_check_status_invariant` must still raise on inconsistent (status, row-count) shapes. Adding match cases for the new presence-indicator semantics, not relaxing existing ones.
- **Mirror invariants update lockstep.** THREE sites carry the predicate — any change must touch all three:
  1. `src/elspeth/contracts/run_result.py::_check_status_invariant` — full biconditional on `RunResult` (L0 dataclass).
  2. `src/elspeth/web/execution/schemas.py::_check_status_row_count_invariant` — Pydantic mirror at the API surface (lowercase string status; same biconditional shape).
  3. `src/elspeth/web/execution/schemas.py::_validate_row_decomposition` — sum-bound check on terminal counters (`rows_processed >= sum_terminal`); narrower than the biconditional but predicate-adjacent.

  **NOT a mirror site:** `src/elspeth/web/sessions/protocol.py::RunRecord.__post_init__` (verified at protocol.py:209 on 2026-05-02). The dataclass DOES enforce other Tier-1 invariants (status enum validity, `finished_at` presence on terminal statuses, `landscape_run_id` presence on operator-completion statuses, `error` string presence on `failed` status), but it does NOT enforce the row-count biconditional predicate. This is a pre-existing audit-coverage gap (a `runs` row could in principle be persisted with `status='completed'` and `rows_succeeded=0, rows_routed_success=0, rows_routed_failure=0` and `RunRecord` would not crash on read), but adding the row-count predicate to `RunRecord.__post_init__` is **out of scope for this PR** — gold-plating a missing invariant during a counter rename would expand the blast radius without addressing the user-facing P1. Create or promote a separate Filigree issue for this gap before closeout and record the issue ID in the closeout comment; this PR updates only the three sites that genuinely carry the predicate today.

  **Structural-smell follow-up captured (out of scope for this PR, but not ambient):** the three-mirror pattern itself is a Limits-to-Growth signal — every counter-shape change must touch three sites in lockstep, and history shows people forget. The canonical CLAUDE.md fix ("When a New Cross-Layer Need Arises", Option 2 — Extract the primitive) is to define `evaluate_status_predicate(...)` once at L0 in `contracts/` and have all three mirror sites delegate to it. Folding the consolidation refactor into this PR would expand the blast radius significantly (every fixture constructing a RunResult or Pydantic model with arbitrary counter shapes would need re-verification under the consolidated predicate). Capture this as a real Filigree task/feature before closeout if it is still only present as an ambient observation, using the title `Extract shared row-count status predicate to L0 contracts`, blocking it on `elspeth-5069612f3c`, and recording the new issue ID in the closeout comment. This prevents every future predicate-touching PR from rediscovering the same compounding cost through an expiring or unverified observation ID.
- **Intermediate commits are intentionally compile-broken.** The CLAUDE.md "no legacy code / no shims" policy makes per-commit green builds impossible during this lockstep cascade — Task 2's L0 changes deliberately break L2 and L3 import sites, which Tasks 3–7 then rebuild. The PR squashes to a single green merge commit. The Done Conditions (Task 8 Step 19) gate the final merge state. Bisect during this PR would land on broken intermediates by design; if a regression is found post-merge, bisect from the merge commit onward, not into the per-task history.
  - The frequent-commits discipline still applies as a checkpoint mechanism — each task ends with a commit so the work is preserved and reviewable as a unit — but those commits are NOT individually green and should not be expected to be. Keep those broken-intermediate commits local or on a clearly private/non-integration backup branch. Do not push compile-broken task commits to a shared PR branch or any branch other agents might base work on. The shared branch is updated only once the final verification gates pass or through an explicitly marked draft/non-integration branch whose description says the history contains broken intermediates. No stash is used.

## Terminology and predicate asymmetry

**DECISION RATIFIED: Option A (preserve asymmetry).** User-confirmed 2026-05-02. Rationale recorded below; implementing agent proceeds without further escalation. Push for unification (Option B) is captured as a future possibility requiring its own ADR-level discussion — not in scope for this PR.

Three "divert"-flavoured concepts coexist in the codebase after this PR. They are NOT synonyms; conflating them in code or documentation produces the same kind of audit-trail confusion that motivated this PR in the first place. Read this section before writing any code.

### The three terms

| Term | Source | Meaning |
|------|--------|---------|
| `RoutingMode.DIVERT` | `src/elspeth/contracts/enums.py:141` (`RoutingMode(StrEnum)`) | **Config-level intent label on an edge.** Marks an edge as failure-handling routing (transform `on_error` reroute) rather than success-side routing (gate `route_to_sink` MOVE). The intent is recorded on `RoutingEvent.mode` at `src/elspeth/contracts/audit.py:398` and surfaced through the run-time edge configuration. |
| `RowOutcome.DIVERTED` | `src/elspeth/contracts/enums.py:189` | **Terminal token outcome — sink-write infrastructure failure.** Emitted when a row reached the success-path sink but the sink itself failed to write the row (the sink is broken, not the row data). The row is then redirected to a configured failsink. Producers: `src/elspeth/engine/executors/sink.py:954, 1000`. The row data was good; the destination broke. |
| `rows_routed_failure` (NEW, this PR) | `src/elspeth/contracts/run_result.py` (this PR) | **Counter — transform-side data failure routed via DIVERT mode.** Increments when `RowOutcome.ROUTED_ON_ERROR` is emitted by `src/elspeth/engine/processor.py` because a transform's `on_error` clause rerouted the row to a named sink in response to an exception thrown by the transform. The row data caused the failure; the destination is fine. |

### The predicate asymmetry

In the new `_check_status_invariant` predicate (`src/elspeth/contracts/run_result.py`, this PR):

- `rows_routed_failure` IS in the `failure_indicator` OR-chain.
- `rows_diverted` is NOT in the `failure_indicator` OR-chain (and was not in the pre-PR predicate either — verified at the existing `run_result.py:72` `has_failures` definition).

This asymmetry is intentional. Two design options were weighed; the recommendation is documented below.

### Option A — Preserve the asymmetry (recommended)

`rows_diverted` records sink-write infrastructure failures: the row data was good, the destination broke, the failsink absorbed the loss. From the run's perspective, the row's payload was successfully captured (just to a different sink than originally configured); the failsink is the documented escape valve for sink-side breakage. A run can still be `COMPLETED` if other rows reached a success terminal (`rows_succeeded > 0` or `rows_routed_success > 0`) — the sink-write failures are operationally significant but recoverable.

`rows_routed_failure` records transform-side data failures: the row data caused an exception, the engine routed the row to an error sink as configured by the pipeline author's `on_error` clause. The row never reached a value-producing terminal. From the predicate's perspective this is structurally a failure indicator equivalent to `rows_failed` or `rows_quarantined`.

Predicate consequence:

- A run with `rows_succeeded=10, rows_diverted=3, rows_routed_failure=0` classifies as `COMPLETED` (no failure indicator; the diverted rows were captured by failsink).
- A run with `rows_succeeded=10, rows_diverted=0, rows_routed_failure=3` classifies as `COMPLETED_WITH_FAILURES`.
- A run with `rows_succeeded=0, rows_diverted=3, rows_routed_failure=0, rows_processed=3` still classifies as `FAILED` under the current and planned `derive_terminal_run_status(...)` fallback because the run processed rows but has no success indicator. `rows_diverted` is deliberately not a `COMPLETED_WITH_FAILURES` indicator in this PR; the "all rows were sink-side diverted" taxonomy question is a pre-existing gap (predates this PR) tracked separately and is NOT in scope here.

Blast radius: minimal. The predicate change in this PR is purely additive (adds `rows_routed_failure` to the failure indicator); `rows_diverted` predicate semantics are unchanged. Mirror invariants update lockstep at the three documented predicate sites (see Constraints); sink-failure tests under `test_diverted_outcome.py` continue to pass without modification (the one site at line 27 changes only because `rows_routed=0` becomes the two new fields — the test's actual coverage of DIVERTED semantics is untouched).

### Option B — Unify by also making `rows_diverted` a failure indicator

Both counters represent rows that did not reach their intended sink; both could be argued to be operator-visible failures. Adding `rows_diverted > 0` to the `failure_indicator` OR-chain would unify the predicate: any row that didn't land where the pipeline author originally targeted contributes to the failure indicator.

Blast radius: significantly larger. Sink-write-failure integration tests would need re-verification (any test that constructs a run with `rows_diverted > 0` and asserts `RunStatus.COMPLETED` would now need to assert `COMPLETED_WITH_FAILURES`); ADR-004 (failsink semantics) would need a substantive amendment to walk back the "failsink absorbs sink-side failures cleanly" framing; the three mirror invariants would need parallel updates with their own test sweeps.

Because Option B has a meaningful blast radius and is not required to fix the user-facing P1 (`elspeth-71520f5e30`), it is rejected from this PR.

### Recommendation

**Adopt Option A.** Preserve the predicate asymmetry, document the local explicit-sink-routing rationale in ADR-004 (Task 8 Step 16), document the broader producer-site discrimination and naming rule in ADR-018, and treat any push for unification (Option B) as a separate follow-up issue requiring its own ADR-level discussion.

The Task 8 Step 16 ADR-004 update must include a paragraph cross-referencing this section so the explicit-sink-routing rationale is captured in the architecture record. ADR-018 must capture the project-wide producer-site discrimination default and the public API/counter naming rule. The amended ADR text (added to Task 8 Step 16) must:

- Cite this section by name ("Terminology and predicate asymmetry") and by issue (`elspeth-5069612f3c`).
- Spell out that `RowOutcome.DIVERTED` (sink-write failure) and `RowOutcome.ROUTED_ON_ERROR` (transform on_error) are operationally distinct outcomes with distinct counters and distinct predicate contributions.
- Explain why the asymmetry is principled (sink-side vs row-side failure semantics) rather than incidental.
- Note that promoting `rows_diverted` to a failure indicator (Option B) is a future possibility but requires its own ADR amendment.

## Done conditions

The PR is ready to merge when all of the following hold:

1. The full test suite passes (`pytest tests/`).
2. `mypy src/` is clean.
3. `ruff check src/` is clean.
4. `python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` passes.
5. `python scripts/cicd/enforce_freeze_guards.py` passes.
6. The user's reproducer shape (csv → gate → sink_a/sink_b, every row gate-routed) classifies as `RunStatus.COMPLETED` with `rows_routed_success > 0` and `rows_routed_failure == 0`. Verified by FOUR tests in Task 8 (each exercising a structurally distinct code path or user-visible surface):
   - **Engine-layer test** in `tests/integration/pipeline/test_composer_runtime_agreement.py` (Task 8 Step 9) — exercises `Orchestrator.run(...)` directly; confirms the L0 predicate fix.
   - **Web API reproducer test** in `tests/integration/web/test_execute_pipeline.py::TestGateRoutedPipelineExecution` (Task 8 Step 9b) — exercises `Orchestrator.run(...)` via the FastAPI surface (POST `/api/sessions/{id}/execute` → poll GET `/api/runs/{run_id}`); confirms the L3 row-count predicate mirror at `_check_status_row_count_invariant`, the sessions read-side Tier-1 status guards on `RunRecord.__post_init__` (status enum validity, `finished_at` presence, `landscape_run_id` presence, `error` string presence — NOT the row-count predicate; see Constraints "NOT a mirror site" subsection), the `/api/runs/{run_id}` API contract, and the `_structural_failure_message` helper all line up.
   - **Dashboard UI DOM test** in `src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx` (`renders gate-routed completed runs without failure alert`, Task 8 Step 9e) — exercises the React Runs dashboard surface using the fixed API response shape. Confirms the visible dashboard status is `completed` and the old failed-run alert / structural-failure message is absent. **This, plus Step 9b, is the VAL gate for closing `elspeth-71520f5e30`.**
   - **Resume-path test** in `tests/integration/pipeline/test_resume_comprehensive.py::test_resume_gate_routed_pipeline_classifies_as_completed` (Task 8 Step 9d) — exercises `Orchestrator.resume(...)`, a structurally distinct code path with its own `derive_terminal_run_status` site at `core.py:3247` and its own local-accumulator pattern (Task 5 Step 2e). Pins that the resume-side terminal-status derivation correctly accumulates `rows_routed_success` from the resume locals, not just from the pre-resume Landscape records.

   The frontend Step 9e commands are part of this done condition: `npm run test -- src/components/inspector/RunsView.test.tsx` and `npm run build` from `src/elspeth/web/frontend` must both pass.

   The engine, Web API, and resume-path tests must have been verified to FAIL against pre-PR `main` before being treated as behavioral acceptance criteria (Pre-write verification in their respective steps). The dashboard UI test is a post-fix surface gate: it must render the fixed response shape without a failure alert and the frontend build must type-check with the split fields.
7. The on_error reproducer shape (every row triggers transform exception, all routed to error sink) still classifies as `RunStatus.FAILED` — verified by the inverted test in Task 8.
8. No occurrence of the retired aggregate/event field names remains in the source tree, tests, or frontend/stable docs. Verify with THREE commands:
   - Python sources/tests, `rows_routed`: `grep -rn "rows_routed\b" src/elspeth/ tests/ --include="*.py"` — expected zero matches (the `\b` word boundary excludes the new suffix forms because `_` is a word character).
   - Python sources/tests, bare `RunSummary`-style `.routed` consumer or `routed=` producer: `grep -rn "RunSummary[^)]*routed[^_]\|\.routed\b\|\brouted=" src/elspeth/ tests/ --include="*.py" | grep -v "routed_success\|routed_failure\|routed_destinations\|RowOutcome\.ROUTED\|RoutingMode\|RoutingEvent\|routed (sink\|routed_summary\|routed rows\|routed via\|routed to "` — expected zero matches. Pins the RunSummary-level split and catches any missed `event.routed` consumer or `routed=` producer call site.
   - Frontend + stable docs: `grep -rn "rows_routed\b\|rowsRouted\b" src/elspeth/web/frontend/src/ docs/architecture/ docs/contracts/ docs/guides/ docs/runbooks/ docs/reference/` — expected zero matches. The frontend is at `src/elspeth/web/frontend/src/` (NOT `frontend/`, `web/`, or `src/elspeth/web/static/` — those paths do not exist; pre-fix verification 2026-05-02 returned zero hits there). The docs target list is **deliberately narrow**: only published / stable contract subdirs (architecture ADRs, token-outcome contracts, user guides, runbooks, reference docs) are gated. The following docs subdirs are EXCLUDED from the gate because they legitimately discuss the old name in prose (specs in flight, audits, bug write-ups, analysis dumps, archived plans): `docs/superpowers/plans/` (this plan and any sibling spec drafts), `docs/superpowers/specs/`, `docs/audits/`, `docs/bugs/`, `docs/analysis/`, `docs/arch-analysis-*/`, `docs/arch-pack-*/`, `docs/release/`, `docs/requirements/`, `docs/assets/`. Greppping all of `docs/` would self-fail because the active plan file itself contains 150+ legitimate prose references to `rows_routed` (task descriptions, replacement instructions, terminology section, etc.).
9. Public API field names are deliberately recorded and pinned: ADR-018 documents `rows_routed_success` / `rows_routed_failure` as stable web-API names for the current public API horizon, and `tests/unit/web/execution/test_schemas.py` asserts those fields are present and `rows_routed` is absent in the Pydantic model JSON schemas for the four response models.
10. The downstream user-facing P1 `elspeth-71520f5e30` is closed with a closure comment citing this fix's commit SHA.

---

## Task 1: TDD seed at L0 — write failing unit tests

**Files:**
- Modify: `tests/unit/contracts/test_run_result.py`
- Modify: `tests/unit/contracts/test_results.py`

**Goal of this task:** Add nineteen failing `RunResult` tests/properties that pin the new predicate semantics: six canonical happy-path shapes (gate MOVE, on_error DIVERT, mixed MOVE+DIVERT, EMPTY, resume, resume-shaped zero-processed mixed-indicator), three additional mixed-counter positive shapes (succeeded-with-on_error-failures, gate-routed-with-hard-failures, canonical-FAILED-via-rows_failed-only), three matrix shapes covering `rows_quarantined` / `rows_coalesce_failed` crossed with the new routed counters, one Hypothesis biconditional round-trip property asserting `RunResult(status=derive_terminal_run_status(**c), **c)` never raises, and six negative-invariant tests pinning the raise-paths in `_check_status_invariant` (COMPLETED-without-success-indicator, COMPLETED-with-failure-indicator, COMPLETED_WITH_FAILURES-without-success-indicator, COMPLETED_WITH_FAILURES-without-failure-indicator, EMPTY-with-rows-processed, EMPTY-with-success-indicator). They reference fields that don't exist yet (`rows_routed_success`, `rows_routed_failure`) so they fail at compile/import time. Also add four failing `RowResult` invariant tests for the new `RowOutcome.ROUTED_ON_ERROR` variant: valid sink+`FailureInfo`, missing sink, missing error, and wrong error type. This is the TDD seed: every later task is judged against these tests passing.

- [ ] **Step 1: Read the existing test_run_result.py to learn the construction pattern**

Run: `cat tests/unit/contracts/test_run_result.py | head -120`

Note the `_build()` helper or direct `RunResult(...)` construction pattern. The new tests must use the same pattern.
Also add Hypothesis imports if the file does not already have them:

```python
from hypothesis import given, strategies as st
```

- [ ] **Step 2: Add the failing predicate tests as a new test class at the end of the file**

Append to `tests/unit/contracts/test_run_result.py`:

```python
class TestRunStatusRowsRoutedSplitPredicate:
    """elspeth-5069612f3c — predicate behavior for the split rows_routed counters.

    These tests pin the post-split semantics: rows_routed_success counts toward
    the success indicator (gate MOVE pipelines complete cleanly), and
    rows_routed_failure counts toward the failure indicator (on_error DIVERT
    pipelines fail, distinct from rows_failed but predicate-equivalent).

    REPLACES the older test_runstatus_rows_routed_only_classifies_as_failed
    pattern at lines 289-295 of this file, which asserted FAILED for the
    structurally ambiguous rows_routed counter (now removed).
    """

    def _build(
        self,
        *,
        status: RunStatus,
        rows_processed: int = 0,
        rows_succeeded: int = 0,
        rows_failed: int = 0,
        rows_routed_success: int = 0,
        rows_routed_failure: int = 0,
        rows_quarantined: int = 0,
        rows_coalesce_failed: int = 0,
    ) -> RunResult:
        return RunResult(
            run_id="rsp-1",
            status=status,
            rows_processed=rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            rows_routed_success=rows_routed_success,
            rows_routed_failure=rows_routed_failure,
            rows_quarantined=rows_quarantined,
            rows_coalesce_failed=rows_coalesce_failed,
        )

    def test_gate_routed_only_classifies_as_completed(self) -> None:
        """User reproducer shape: csv -> gate -> sink_a/sink_b, every row
        intentionally gate-routed (RowOutcome.ROUTED via MOVE). rows_succeeded
        is 0 because the orchestrator's success-path counter never increments
        for terminally-gate-routed rows; the new rows_routed_success counter
        carries the structural success signal.
        """
        derived = derive_terminal_run_status(
            rows_processed=8,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=8,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED
        # The biconditional invariant accepts the (COMPLETED, 8, 0, 8, 0, 0, 0) shape.
        result = self._build(
            status=RunStatus.COMPLETED,
            rows_processed=8,
            rows_succeeded=0,
            rows_routed_success=8,
        )
        assert result.status == RunStatus.COMPLETED

    def test_on_error_routed_only_classifies_as_failed(self) -> None:
        """S1A reproducer shape: every row triggers a transform exception, all
        routed via on_error to a quarantine/error sink (RowOutcome.ROUTED_ON_ERROR
        via DIVERT). rows_routed_failure carries the structural failure signal;
        no success indicator is present.
        """
        derived = derive_terminal_run_status(
            rows_processed=2,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=2,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.FAILED

    def test_mixed_gate_and_on_error_classifies_as_completed_with_failures(self) -> None:
        """Mixed shape: some rows gate-routed (success) AND some rows
        on_error-routed (failure). Predicate must report
        COMPLETED_WITH_FAILURES — at least one success indicator AND at least
        one failure indicator.
        """
        derived = derive_terminal_run_status(
            rows_processed=10,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=7,
            rows_routed_failure=3,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES

    def test_empty_pipeline_still_classifies_as_empty(self) -> None:
        """Regression: empty source (rows_processed == 0, no failure
        indicator) still classifies as EMPTY after the split.
        """
        derived = derive_terminal_run_status(
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.EMPTY

    def test_resume_continuation_still_classifies_as_completed(self) -> None:
        """Regression: resume / coalesce-continuation shape (rows_processed == 0
        AND rows_succeeded > 0) still classifies as COMPLETED after the split.
        """
        derived = derive_terminal_run_status(
            rows_processed=0,
            rows_succeeded=3,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED

    def test_resume_continuation_with_success_and_failure_indicators_classifies_as_completed_with_failures(self) -> None:
        """Regression for resume-after-coalesce shapes: rows_processed can be 0
        while continuation bookkeeping reports both a success indicator and a
        failure indicator. derive_terminal_run_status() and the L0
        _check_status_invariant must agree on COMPLETED_WITH_FAILURES without
        requiring rows_processed > 0.
        """
        derived = derive_terminal_run_status(
            rows_processed=0,
            rows_succeeded=3,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=1,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=0,
            rows_succeeded=3,
            rows_routed_failure=1,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    # ------------------------------------------------------------------
    # Additional positive-shape coverage (elspeth-5069612f3c review pass)
    # The six canonical shapes above cover the user-facing reproducer
    # scenarios and the resume-after-coalesce zero-processed mixed-indicator
    # regression. These three additional shapes pin mixed-counter cases
    # the canonical set leaves under-tested. Without them the predicate
    # could regress on operationally-common shapes (success-path-with-
    # on-error-failures, gate-routed-with-hard-failures, etc.) without
    # any test catching it.
    # ------------------------------------------------------------------

    def test_succeeded_mixed_with_on_error_routing_classifies_as_completed_with_failures(self) -> None:
        """Mixed-success shape: some rows reached on_success success-path sinks
        (rows_succeeded > 0) while others triggered transform exceptions and
        were on_error-routed (rows_routed_failure > 0). Predicate must report
        COMPLETED_WITH_FAILURES — both indicators are present.
        """
        derived = derive_terminal_run_status(
            rows_processed=10,
            rows_succeeded=7,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=3,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        # Verify the L0 invariant accepts this shape without raising.
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=10,
            rows_succeeded=7,
            rows_routed_failure=3,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_gate_routed_mixed_with_hard_failures_classifies_as_completed_with_failures(self) -> None:
        """Mixed-routing shape: some rows gate-routed via MOVE
        (rows_routed_success > 0) while others reached the canonical FAILED
        terminal via transform exceptions that were NOT on_error-rerouted
        (rows_failed > 0). Both indicators present; predicate is
        COMPLETED_WITH_FAILURES.
        """
        derived = derive_terminal_run_status(
            rows_processed=10,
            rows_succeeded=0,
            rows_failed=4,
            rows_routed_success=6,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        # Verify the L0 invariant accepts this shape without raising.
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=10,
            rows_failed=4,
            rows_routed_success=6,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_canonical_failed_via_rows_failed_only_classifies_as_failed(self) -> None:
        """Canonical FAILED shape: every row reached RowOutcome.FAILED via an
        unhandled transform exception (no on_error reroute, no gate routing).
        rows_failed > 0 is the sole failure indicator; predicate is FAILED.

        This pins the legacy FAILED path that pre-existed the rows_routed
        split — without this test, a regression that only checked the new
        rows_routed_failure indicator could silently bypass rows_failed.
        """
        derived = derive_terminal_run_status(
            rows_processed=5,
            rows_succeeded=0,
            rows_failed=5,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.FAILED

    # ------------------------------------------------------------------
    # Predicate matrix coverage for old failure counters crossed with the
    # new routed counters. These are cheap but important: rows_quarantined
    # and rows_coalesce_failed are pre-existing failure indicators, and the
    # rows_routed_success / rows_routed_failure split must compose with
    # them exactly like rows_failed does.
    # ------------------------------------------------------------------

    def test_gate_routed_with_quarantined_rows_classifies_as_completed_with_failures(self) -> None:
        derived = derive_terminal_run_status(
            rows_processed=8,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=5,
            rows_routed_failure=0,
            rows_quarantined=3,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=8,
            rows_routed_success=5,
            rows_quarantined=3,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_gate_routed_with_coalesce_failures_classifies_as_completed_with_failures(self) -> None:
        derived = derive_terminal_run_status(
            rows_processed=8,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=5,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=3,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=8,
            rows_routed_success=5,
            rows_coalesce_failed=3,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_on_error_routed_with_quarantine_and_coalesce_failures_classifies_as_failed(self) -> None:
        derived = derive_terminal_run_status(
            rows_processed=8,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=4,
            rows_quarantined=2,
            rows_coalesce_failed=2,
        )
        assert derived == RunStatus.FAILED
        result = self._build(
            status=RunStatus.FAILED,
            rows_processed=8,
            rows_routed_failure=4,
            rows_quarantined=2,
            rows_coalesce_failed=2,
        )
        assert result.status == RunStatus.FAILED

    @given(
        rows_processed=st.integers(min_value=0, max_value=20),
        rows_succeeded=st.integers(min_value=0, max_value=20),
        rows_failed=st.integers(min_value=0, max_value=20),
        rows_routed_success=st.integers(min_value=0, max_value=20),
        rows_routed_failure=st.integers(min_value=0, max_value=20),
        rows_quarantined=st.integers(min_value=0, max_value=20),
        rows_coalesce_failed=st.integers(min_value=0, max_value=20),
    )
    def test_derived_status_round_trips_l0_invariant(
        self,
        rows_processed: int,
        rows_succeeded: int,
        rows_failed: int,
        rows_routed_success: int,
        rows_routed_failure: int,
        rows_quarantined: int,
        rows_coalesce_failed: int,
    ) -> None:
        """Biconditional property: any counter tuple classified by
        derive_terminal_run_status() must be accepted by RunResult's L0
        status invariant when used with the derived status.

        This is the cheapest guard against the mirror-drift class that
        produced elspeth-71520f5e30: the predicate function and the
        dataclass invariant must agree for arbitrary non-negative counter
        tuples, including the new routed counters crossed with legacy
        failure counters.
        """
        counters = {
            "rows_processed": rows_processed,
            "rows_succeeded": rows_succeeded,
            "rows_failed": rows_failed,
            "rows_routed_success": rows_routed_success,
            "rows_routed_failure": rows_routed_failure,
            "rows_quarantined": rows_quarantined,
            "rows_coalesce_failed": rows_coalesce_failed,
        }
        derived = derive_terminal_run_status(**counters)
        result = self._build(status=derived, **counters)
        assert result.status == derived

    # ------------------------------------------------------------------
    # Negative invariant coverage (elspeth-5069612f3c review pass)
    # The updated _check_status_invariant has seven raise-paths. Without
    # negative tests, a future regression that relaxes the invariant
    # (admits a shape it should reject) passes every positive test
    # silently. These six negative tests pin the most consequential
    # raise-paths so loosened-invariant regressions are caught.
    # ------------------------------------------------------------------

    def test_completed_without_success_indicator_raises(self) -> None:
        """COMPLETED requires rows_succeeded > 0 OR rows_routed_success > 0.
        With both at zero AND no failure indicator, the run should classify
        as EMPTY (rows_processed == 0) or FAILED (rows_processed > 0) — NOT
        COMPLETED. The invariant must reject this construction.
        """
        with pytest.raises(ValueError, match="status=COMPLETED requires a success indicator"):
            self._build(
                status=RunStatus.COMPLETED,
                rows_processed=5,
                rows_succeeded=0,
                rows_routed_success=0,
            )

    def test_completed_with_failure_indicator_raises(self) -> None:
        """COMPLETED requires NO failure indicator. If any failure counter is
        non-zero, the status is COMPLETED_WITH_FAILURES, not COMPLETED.
        """
        with pytest.raises(ValueError, match="status=COMPLETED requires no failures"):
            self._build(
                status=RunStatus.COMPLETED,
                rows_processed=5,
                rows_succeeded=4,
                rows_failed=1,  # Failure indicator must trigger COMPLETED_WITH_FAILURES, not COMPLETED.
            )

    def test_completed_with_failures_without_success_indicator_raises(self) -> None:
        """COMPLETED_WITH_FAILURES requires BOTH a success indicator AND a
        failure indicator. With only failures present (no success path), the
        status must be FAILED, not COMPLETED_WITH_FAILURES.
        """
        with pytest.raises(ValueError, match="COMPLETED_WITH_FAILURES requires a success indicator"):
            self._build(
                status=RunStatus.COMPLETED_WITH_FAILURES,
                rows_processed=3,
                rows_failed=3,
            )

    def test_completed_with_failures_without_failure_indicator_raises(self) -> None:
        """COMPLETED_WITH_FAILURES requires BOTH indicators. With only success
        present (no failures), the status must be COMPLETED, not
        COMPLETED_WITH_FAILURES.
        """
        with pytest.raises(ValueError, match="COMPLETED_WITH_FAILURES requires at least one failure indicator"):
            self._build(
                status=RunStatus.COMPLETED_WITH_FAILURES,
                rows_processed=3,
                rows_succeeded=3,
            )

    def test_empty_with_rows_processed_raises(self) -> None:
        """EMPTY requires rows_processed == 0 (no input rows reached the
        engine). Any non-zero rows_processed contradicts EMPTY semantics.
        """
        with pytest.raises(ValueError, match="status=EMPTY requires rows_processed == 0"):
            self._build(
                status=RunStatus.EMPTY,
                rows_processed=1,  # Contradicts EMPTY.
            )

    def test_empty_with_success_indicator_raises(self) -> None:
        """EMPTY requires no success indicator. A run with rows_succeeded > 0
        or rows_routed_success > 0 is not EMPTY by definition.
        """
        with pytest.raises(ValueError, match="status=EMPTY requires no success indicator"):
            self._build(
                status=RunStatus.EMPTY,
                rows_processed=0,
                rows_routed_success=1,  # Success indicator contradicts EMPTY.
            )
```

- [ ] **Step 3: Add RowResult invariant tests for ROUTED_ON_ERROR**

In `tests/unit/contracts/test_results.py`, append these tests to `TestRowResultWithFailureInfo` (the file already imports `pytest`, `RowOutcome`, `OrchestrationInvariantError`, `FailureInfo`, `RowResult`, and `_wrap_dict_as_pipeline_row`):

```python
    def test_routed_on_error_with_failure_info(self) -> None:
        """ROUTED_ON_ERROR carries sink_name and typed originating failure evidence."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        error = FailureInfo(
            exception_type="ValueError",
            message="bad row",
        )

        result = RowResult(
            token=token,
            final_data=_wrap_dict_as_pipeline_row({"x": 1}),
            outcome=RowOutcome.ROUTED_ON_ERROR,
            sink_name="error_sink",
            error=error,
        )

        assert result.outcome == RowOutcome.ROUTED_ON_ERROR
        assert result.sink_name == "error_sink"
        assert result.error is error

    def test_routed_on_error_without_sink_name_raises(self) -> None:
        """ROUTED_ON_ERROR must identify the failure sink for audit lineage."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        error = FailureInfo(exception_type="ValueError", message="bad row")

        with pytest.raises(OrchestrationInvariantError, match="ROUTED_ON_ERROR outcome requires sink_name"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"x": 1}),
                outcome=RowOutcome.ROUTED_ON_ERROR,
                error=error,
            )

    def test_routed_on_error_without_error_raises(self) -> None:
        """ROUTED_ON_ERROR must carry the originating transform failure."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))

        with pytest.raises(OrchestrationInvariantError, match=r"ROUTED_ON_ERROR outcome requires error \(FailureInfo\)"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"x": 1}),
                outcome=RowOutcome.ROUTED_ON_ERROR,
                sink_name="error_sink",
            )

    def test_routed_on_error_with_non_failure_info_error_raises(self) -> None:
        """ROUTED_ON_ERROR rejects untyped error evidence at runtime."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))

        with pytest.raises(OrchestrationInvariantError, match="ROUTED_ON_ERROR outcome requires error to be a FailureInfo instance"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"x": 1}),
                outcome=RowOutcome.ROUTED_ON_ERROR,
                sink_name="error_sink",
                error=object(),  # type: ignore[arg-type]
            )
```

- [ ] **Step 4: Run the new tests to verify they fail at import time**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_run_result.py::TestRunStatusRowsRoutedSplitPredicate -v 2>&1 | head -50`

Expected: the new `TestRunStatusRowsRoutedSplitPredicate` tests fail at import / collection time with TypeError or attribute error indicating `rows_routed_success` / `rows_routed_failure` are not valid kwargs on `RunResult`, or `derive_terminal_run_status` does not accept those kwargs. The exact failure mode depends on the import order — the goal is RED. Capture the error message.

After Task 2 lands the L0 contract changes, re-run the same command — expected: all nineteen pass. The twelve positive-shape tests verify `derive_terminal_run_status` and the dataclass invariant accept the canonical and matrix predicate inputs; the Hypothesis property verifies the derived-status biconditional for arbitrary non-negative tuples; the six negative-invariant tests verify `_check_status_invariant` raises with the expected error message on each forbidden (status, counter-tuple) shape.

Then run the RowResult invariant seed:

```bash
.venv/bin/python -m pytest \
  tests/unit/contracts/test_results.py::TestRowResultWithFailureInfo::test_routed_on_error_with_failure_info \
  tests/unit/contracts/test_results.py::TestRowResultWithFailureInfo::test_routed_on_error_without_sink_name_raises \
  tests/unit/contracts/test_results.py::TestRowResultWithFailureInfo::test_routed_on_error_without_error_raises \
  tests/unit/contracts/test_results.py::TestRowResultWithFailureInfo::test_routed_on_error_with_non_failure_info_error_raises \
  -v
```

Expected before Task 2: all four fail because `RowOutcome.ROUTED_ON_ERROR` is not defined. Expected after Task 2 Step 3: all four pass. The non-`FailureInfo` test is the mechanical guard for the runtime invariant; do not rely on the `FailureInfo | None` type annotation alone.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/unit/contracts/test_run_result.py tests/unit/contracts/test_results.py
git commit -m "test(contracts): RED for rows_routed counter split (elspeth-5069612f3c)"
```

---

## Task 2: L0 contracts — split RowOutcome, add new RunResult fields, update predicate and invariant

**Files:**
- Modify: `src/elspeth/contracts/enums.py`
- Modify: `src/elspeth/contracts/results.py:406`
- Modify: `src/elspeth/contracts/run_result.py` (entire file's predicate, fields, invariant, derive function, to_dict)
- Modify: `src/elspeth/contracts/errors.py` (`GracefulShutdownError.__init__`, lines 723-743 — `rows_routed: int = 0` parameter at 731, `self.rows_routed = rows_routed` at 739). `_RunFailedWithPartialResultError` at `engine/orchestrator/core.py:166` carries a `partial_result: RunResult` rather than direct `rows_routed` fields and does NOT need editing here.

**Goal of this task:** Add the new enum value, the two new fields, update the derive function and the biconditional invariant. After this task, the L0 unit tests from Task 1 pass, but the rest of the codebase will not compile (every consumer of `rows_routed` is now broken — that's the next 5 tasks).

- [ ] **Step 1: Locate the RowOutcome enum definition**

Run: `grep -n "class RowOutcome\|^    ROUTED\|^    DIVERTED\|^    COMPLETED" src/elspeth/contracts/enums.py | head -20`

Note the existing enum-value definitions and their order.

- [ ] **Step 2: Add the new RowOutcome variant**

In `src/elspeth/contracts/enums.py`, add `ROUTED_ON_ERROR = "routed_on_error"` immediately after the existing `ROUTED = "routed"` line. The string value uses the same lowercase-snake convention as the rest of the enum.

- [ ] **Step 3: Update the RowResult invariant for the new variant**

In `src/elspeth/contracts/results.py:404-409`, after the existing `if self.outcome == RowOutcome.ROUTED ...` check, add a runtime-enforced invariant for the new variant:

```python
        if self.outcome == RowOutcome.ROUTED_ON_ERROR:
            if self.sink_name is None:
                raise OrchestrationInvariantError(
                    "ROUTED_ON_ERROR outcome requires sink_name to be set"
                )
            if self.error is None:
                raise OrchestrationInvariantError(
                    "ROUTED_ON_ERROR outcome requires error (FailureInfo) to be set — "
                    "the originating transform error must be captured on the outcome "
                    "record for single-hop audit attributability. See "
                    "docs/contracts/token-outcomes/00-token-outcome-contract.md."
                )
            if not isinstance(self.error, FailureInfo):
                raise OrchestrationInvariantError(
                    "ROUTED_ON_ERROR outcome requires error to be a FailureInfo instance"
                )
```

This mirrors the `DIVERTED` contract (sink_name + error_hash), not the `ROUTED` contract (sink_name only). The `RowResult.error` field already exists at line 401 (`error: FailureInfo | None = None`), but that type hint is not a runtime contract; the explicit `isinstance(..., FailureInfo)` check is required so malformed Tier-1 failure evidence crashes at construction instead of being hashed or dereferenced later. The accumulator/pending-sink path converts valid `FailureInfo.message` to the canonical 16-character `error_hash` before sink durability is recorded; `SinkExecutor` then forwards that hash to `record_token_outcome(...)`, where the write-side audit contract is enforced. Note: this is a stronger offensive invariant than existing FAILED currently has at the RowResult layer (FAILED relies on the recorder to enforce `error_hash`); using the strict guard at construction time means any producer that forgets `error=FailureInfo(...)` or passes an untyped object crashes immediately rather than at persistence.

- [ ] **Step 4: Replace `rows_routed` with the two new fields in the RunResult dataclass**

In `src/elspeth/contracts/run_result.py`, change the dataclass field declaration block (lines 22-37). REMOVE `rows_routed: int` and ADD two new fields, preserving alphabetical-by-position consistency:

```python
@dataclass(frozen=True, slots=True)
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: RunStatus
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed_success: int  # MOVE: gate route_to_sink (intentional success-side routing)
    rows_routed_failure: int  # DIVERT: transform on_error reroute to failure sink
    rows_quarantined: int = 0
    rows_forked: int = 0
    rows_coalesced: int = 0
    rows_coalesce_failed: int = 0
    rows_expanded: int = 0
    rows_buffered: int = 0
    rows_diverted: int = 0
    routed_destinations: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
```

Note: `rows_routed_success` and `rows_routed_failure` are required (no default) because the engine always emits them. Existing call sites that omit `rows_routed=N` were emitting `0` by default — those become explicit `rows_routed_success=0, rows_routed_failure=0` in later tasks.

- [ ] **Step 5: Update __post_init__ require_int validation**

In `src/elspeth/contracts/run_result.py:43-54`, replace the line `require_int(self.rows_routed, "rows_routed", min_value=0)` with two lines:

```python
        require_int(self.rows_routed_success, "rows_routed_success", min_value=0)
        require_int(self.rows_routed_failure, "rows_routed_failure", min_value=0)
```

- [ ] **Step 6: Update the biconditional invariant `_check_status_invariant`**

In `src/elspeth/contracts/run_result.py:57-150`, replace the entire method. The new predicate uses unified success/failure presence indicators:

```python
    def _check_status_invariant(self) -> None:
        """elspeth-5069612f3c — biconditional invariant linking ``status`` to
        the row-count shape using unified presence indicators after the
        rows_routed split.

        success_indicator = rows_succeeded > 0 OR rows_routed_success > 0
        failure_indicator = rows_failed > 0 OR rows_quarantined > 0
                            OR rows_coalesce_failed > 0 OR rows_routed_failure > 0

        Non-terminal (``RUNNING``) and signal-bounded (``INTERRUPTED``)
        statuses bypass the predicate.
        """
        success_indicator = self.rows_succeeded > 0 or self.rows_routed_success > 0
        failure_indicator = (
            self.rows_failed > 0
            or self.rows_quarantined > 0
            or self.rows_coalesce_failed > 0
            or self.rows_routed_failure > 0
        )

        match (self.status, self.rows_processed, success_indicator, failure_indicator):
            case (RunStatus.RUNNING, _, _, _):
                return
            case (RunStatus.INTERRUPTED, _, _, _):
                return
            case (RunStatus.COMPLETED, _, True, False):
                return
            case (RunStatus.COMPLETED, _, False, _):
                raise ValueError(
                    f"RunResult: status=COMPLETED requires a success indicator "
                    f"(rows_succeeded > 0 or rows_routed_success > 0); "
                    f"got rows_succeeded={self.rows_succeeded}, "
                    f"rows_routed_success={self.rows_routed_success} "
                    f"(use status=FAILED when no row reached a success path)"
                )
            case (RunStatus.COMPLETED, _, _, True):
                raise ValueError(
                    f"RunResult: status=COMPLETED requires no failures "
                    f"(rows_failed={self.rows_failed}, "
                    f"rows_quarantined={self.rows_quarantined}, "
                    f"rows_coalesce_failed={self.rows_coalesce_failed}, "
                    f"rows_routed_failure={self.rows_routed_failure}); "
                    f"use status=COMPLETED_WITH_FAILURES when at least one row "
                    f"reached a failure terminal state"
                )
            case (RunStatus.COMPLETED_WITH_FAILURES, _, True, True):
                return
            case (RunStatus.COMPLETED_WITH_FAILURES, _, False, _):
                raise ValueError(
                    f"RunResult: status=COMPLETED_WITH_FAILURES requires a success indicator "
                    f"(rows_succeeded > 0 or rows_routed_success > 0); "
                    f"got rows_succeeded={self.rows_succeeded}, "
                    f"rows_routed_success={self.rows_routed_success} "
                    f"(use status=FAILED when no row reached a success path)"
                )
            case (RunStatus.COMPLETED_WITH_FAILURES, _, _, False):
                raise ValueError(
                    f"RunResult: status=COMPLETED_WITH_FAILURES requires at least one failure indicator "
                    f"(rows_failed > 0 or rows_quarantined > 0 or rows_coalesce_failed > 0 "
                    f"or rows_routed_failure > 0); got rows_failed={self.rows_failed}, "
                    f"rows_quarantined={self.rows_quarantined}, "
                    f"rows_coalesce_failed={self.rows_coalesce_failed}, "
                    f"rows_routed_failure={self.rows_routed_failure} "
                    f"(use status=COMPLETED for clean runs)"
                )
            case (RunStatus.FAILED, _, _, _):
                # FAILED has two semantic origins (predicate decision and
                # exception-bounded run) — same biconditional tolerance as
                # before the split.
                return
            case (RunStatus.EMPTY, 0, False, False):
                return
            case (RunStatus.EMPTY, p, _, _) if p > 0:
                raise ValueError(
                    f"RunResult: status=EMPTY requires rows_processed == 0, "
                    f"got rows_processed={p}"
                )
            case (RunStatus.EMPTY, _, True, _):
                raise ValueError(
                    f"RunResult: status=EMPTY requires no success indicator "
                    f"(rows_succeeded={self.rows_succeeded}, "
                    f"rows_routed_success={self.rows_routed_success})"
                )
            case (RunStatus.EMPTY, _, _, True):
                raise ValueError(
                    f"RunResult: status=EMPTY requires no failures "
                    f"(rows_failed={self.rows_failed}, "
                    f"rows_quarantined={self.rows_quarantined}, "
                    f"rows_coalesce_failed={self.rows_coalesce_failed}, "
                    f"rows_routed_failure={self.rows_routed_failure}); "
                    f"use status=FAILED when the run encountered failures with "
                    f"no successful rows"
                )
            case _:
                raise ValueError(
                    f"RunResult: unhandled status/row-count shape: "
                    f"status={self.status!r}, rows_processed={self.rows_processed}, "
                    f"success_indicator={success_indicator}, "
                    f"failure_indicator={failure_indicator}"
                )
```

- [ ] **Step 7: Update `to_dict()` to serialize the two new fields**

In `src/elspeth/contracts/run_result.py:152-174`, replace the `"rows_routed": self.rows_routed,` line with:

```python
            "rows_routed_success": self.rows_routed_success,
            "rows_routed_failure": self.rows_routed_failure,
```

- [ ] **Step 8: Update `derive_terminal_run_status()` signature and body**

In `src/elspeth/contracts/run_result.py:177-229`, replace the function:

```python
def derive_terminal_run_status(
    *,
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
    rows_coalesce_failed: int,
) -> RunStatus:
    """elspeth-5069612f3c — pick a terminal RunStatus from row counts using
    the unified presence-indicator predicate after the rows_routed split.

    success_indicator = rows_succeeded > 0 OR rows_routed_success > 0
    failure_indicator = rows_failed > 0 OR rows_quarantined > 0
                        OR rows_coalesce_failed > 0 OR rows_routed_failure > 0

    Predicate:
    - rows_processed == 0 AND no failure_indicator -> EMPTY (or FAILED if
      a failure indicator is present without source iteration)
    - success_indicator AND not failure_indicator -> COMPLETED
    - success_indicator AND failure_indicator -> COMPLETED_WITH_FAILURES
    - not success_indicator AND rows_processed > 0 -> FAILED

    The result is constrained to the four-value terminal taxonomy
    (COMPLETED / COMPLETED_WITH_FAILURES / FAILED / EMPTY); callers that
    need INTERRUPTED or RUNNING set those values directly.
    """
    success_indicator = rows_succeeded > 0 or rows_routed_success > 0
    failure_indicator = (
        rows_failed > 0
        or rows_quarantined > 0
        or rows_coalesce_failed > 0
        or rows_routed_failure > 0
    )
    if rows_processed == 0 and not success_indicator:
        return RunStatus.FAILED if failure_indicator else RunStatus.EMPTY
    if not success_indicator:
        return RunStatus.FAILED
    if failure_indicator:
        return RunStatus.COMPLETED_WITH_FAILURES
    return RunStatus.COMPLETED
```

- [ ] **Step 9: Update the exception class in errors.py**

Find `GracefulShutdownError.__init__` at `src/elspeth/contracts/errors.py:723` (class declared line 713). Read the surrounding 30 lines to understand the constructor shape, then replace `rows_routed: int = 0` (parameter at line 731) with the two new fields:

```python
        rows_routed_success: int = 0,
        rows_routed_failure: int = 0,
```

And replace `self.rows_routed = rows_routed` with:

```python
        self.rows_routed_success = rows_routed_success
        self.rows_routed_failure = rows_routed_failure
```

- [ ] **Step 9b: Split `RunSummary.routed` in `src/elspeth/contracts/events.py`**

`RunSummary` is the L0 EventBus event consumed by CLI formatters and external CI integrations. Its single `routed` field carries the same MOVE/DIVERT conflation that the predicate split is fixing — leaving it as a single field would create a shape mismatch between the run record (split) and the event surface (conflated). Verify the current shape with `grep -n "routed\b\|class RunSummary" src/elspeth/contracts/events.py`; observed lines 2026-05-02 are 122 (class), 129-130 (docstring), 141 (field), 150 (require_int guard).

Replace the field declaration at line 141:

```python
    routed: int = 0  # Rows routed to non-default sinks
```

with the two new fields:

```python
    routed_success: int = 0  # Rows routed via gate route_to_sink (intentional MOVE)
    routed_failure: int = 0  # Rows routed via transform on_error (DIVERT)
```

Replace the `require_int` guard at line 150:

```python
        require_int(self.routed, "routed", min_value=0)
```

with two parallel guards:

```python
        require_int(self.routed_success, "routed_success", min_value=0)
        require_int(self.routed_failure, "routed_failure", min_value=0)
```

Update the class docstring (lines 128-130) to describe the split:

```python
    Routing breakdown:
    - routed_success: Rows routed via gate route_to_sink (intentional MOVE — success-side routing)
    - routed_failure: Rows routed via transform on_error (DIVERT — failure-side routing)
    - routed_destinations: Count per destination sink {sink_name: count}; the per-sink
      breakdown is not split by routing intent — see ADR-004 for rationale.
```

- [ ] **Step 9c: Add `ROUTED_ON_ERROR` to `PendingOutcome._FAILURE_OUTCOMES` in `src/elspeth/contracts/engine.py`**

`PendingOutcome` carries token outcomes through the pending-sink queue until sink durability is confirmed; its `__post_init__` (lines 75-85) enforces the contract that outcomes in `_FAILURE_OUTCOMES` MUST have `error_hash` and outcomes outside the set MUST NOT have it. ROUTED_ON_ERROR is the FIRST outcome that both (a) routes through the pending-sink pipeline AND (b) requires `error_hash` — existing FAILURE_OUTCOMES (`QUARANTINED`, `FAILED`) are recorded synchronously without going through `_route_to_sink`. Without this addition, the accumulator's call to construct a `PendingOutcome(ROUTED_ON_ERROR, error_hash="<hash>")` (added in Task 3 Step 3) crashes with ValueError.

In `src/elspeth/contracts/engine.py:65-70`, replace:

```python
    _FAILURE_OUTCOMES: ClassVar[frozenset[RowOutcome]] = frozenset(
        {
            RowOutcome.QUARANTINED,
            RowOutcome.FAILED,
        }
    )
```

with:

```python
    # Outcomes that require error_hash on PendingOutcome (the misleading
    # legacy name "_FAILURE_OUTCOMES" is preserved for callsite stability;
    # the actual semantic is "outcomes requiring error_hash"). ROUTED_ON_ERROR
    # joins this set because it is the first outcome that both routes through
    # the pending-sink pipeline AND requires error_hash for single-hop audit
    # attributability — see docs/contracts/token-outcomes/00-token-outcome-contract.md.
    _FAILURE_OUTCOMES: ClassVar[frozenset[RowOutcome]] = frozenset(
        {
            RowOutcome.QUARANTINED,
            RowOutcome.FAILED,
            RowOutcome.ROUTED_ON_ERROR,
        }
    )
```

Also update the docstring at lines 76-81 (`Validate outcome/error_hash consistency`) to describe the new variant:

```python
        """Validate outcome/error_hash consistency.

        QUARANTINED, FAILED, and ROUTED_ON_ERROR outcomes MUST have an
        error_hash — the audit trail needs to reference the error record.
        ROUTED_ON_ERROR is unique among the three in that it ALSO writes
        to a sink (via the pending pipeline); QUARANTINED and FAILED are
        recorded synchronously without sink writes. Other outcomes must NOT
        have an error_hash (an error_hash on COMPLETED would be nonsensical).
        """
```

- [ ] **Step 10: Run the L0 unit tests — they should now pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_run_result.py::TestRunStatusRowsRoutedSplitPredicate -v`

Expected: 19 passed (twelve positive-shape tests + one Hypothesis biconditional property + six negative-invariant tests). If any test fails, debug — do not move on. The negative tests assert `pytest.raises(ValueError, match=...)` against `RunResult.__post_init__::_check_status_invariant`; if a negative test fails with "DID NOT RAISE", the predicate is admitting a shape it should reject — investigate the match-statement branch coverage in Step 6. If the Hypothesis property fails, `derive_terminal_run_status()` and `_check_status_invariant()` disagree on at least one counter tuple; fix the predicate/invariant mirror before touching downstream layers.

- [ ] **Step 11: Run the rest of the L0 contract tests — they will likely fail because pre-existing tests still reference rows_routed**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_run_result.py -v 2>&1 | tail -30`

Expected: many failures with TypeError "unexpected keyword argument 'rows_routed'" or "missing required arguments rows_routed_success / rows_routed_failure". Note this is the natural state — Task 8 sweeps the test suite. Do not fix individual tests yet.

- [ ] **Step 12: Commit the L0 contract changes**

```bash
git add src/elspeth/contracts/enums.py src/elspeth/contracts/results.py src/elspeth/contracts/run_result.py src/elspeth/contracts/errors.py src/elspeth/contracts/events.py src/elspeth/contracts/engine.py
git commit -m "feat(contracts): split RowOutcome.ROUTED and rows_routed into MOVE/DIVERT; split RunSummary.routed; admit ROUTED_ON_ERROR to PendingOutcome._FAILURE_OUTCOMES (elspeth-5069612f3c)"
```

---

## Task 3: L2 engine types — ExecutionCounters, AggregationFlushResult, outcomes accumulator

**Files:**
- Modify: `src/elspeth/engine/orchestrator/types.py` (lines 100-220+)
- Modify: `src/elspeth/engine/orchestrator/outcomes.py` (`_decrement_counter` inner function defined at line 90 must be removed/replaced with direct outcome branches; `_decrement_counter("rows_routed")` call site at line 105 becomes explicit `rows_routed_success` / `rows_routed_failure` field mutation; `accumulate_row_outcomes` ROUTED branch at lines 171-180)

**Goal of this task:** Replace `rows_routed` on `ExecutionCounters` and `AggregationFlushResult` with the two new fields. Update the accumulator to choose the correct counter based on the RowOutcome variant.

- [ ] **Step 1: Update AggregationFlushResult fields**

In `src/elspeth/engine/orchestrator/types.py:103-160`, change the `AggregationFlushResult` dataclass:

Replace `rows_routed: int = 0` (line 113) with:

```python
    rows_routed_success: int = 0
    rows_routed_failure: int = 0
```

Update `to_dict()` (lines 125-142): replace `"rows_routed": self.rows_routed,` (line 134) with two lines for the new fields.

Update `__add__` (lines 144-159): replace `rows_routed=self.rows_routed + other.rows_routed,` (line 151) with two lines for the new fields.

- [ ] **Step 2: Update ExecutionCounters fields**

In `src/elspeth/engine/orchestrator/types.py:163-220`, change the `ExecutionCounters` dataclass:

Replace `rows_routed: int = 0` (line 177) with:

```python
    rows_routed_success: int = 0
    rows_routed_failure: int = 0
```

Update `accumulate_flush_result()` (lines 187-203): replace `self.rows_routed += result.rows_routed` (line 195) with two lines for the new fields.

Update `to_flush_result()` (lines 205-221): replace `rows_routed=self.rows_routed,` (line 213) with two lines for the new fields.

Update `to_run_result()` (lines 223-247): replace `rows_routed=self.rows_routed,` (line 236) with two lines for the new fields.

- [ ] **Step 3: Update the outcomes accumulator branch logic AND extend `_route_to_sink` to carry `error_hash`**

This step has THREE coupled edits in `src/elspeth/engine/orchestrator/outcomes.py`. They must land together — `_route_to_sink`'s signature, the accumulator's new branch, and a new module-level `hashlib` import are all required for `PendingOutcome(ROUTED_ON_ERROR, error_hash=...)` to construct successfully and the recorder's `_validate_outcome_fields` invariant to be satisfied at sink-durability time.

**3a — Add `hashlib` import.** At the top of the file, add the standard-library import in ruff/isort order before the existing `from collections.abc import Iterable` import:

```python
import hashlib
```

This matches the pattern used at `src/elspeth/engine/processor.py:2262`, `src/elspeth/engine/executors/sink.py:288/951/997`, and `src/elspeth/engine/coalesce_executor.py:509/703/1043` — the canonical 16-char SHA-256 prefix recipe for `error_hash` values.

**3b — Extend `_route_to_sink` signature with `error_hash` keyword and offensive guards.** Replace the existing function at `outcomes.py:47-70`:

```python
def _route_to_sink(
    sink_name: str,
    pending_tokens: PendingTokenMap,
    token: TokenInfo,
    pending_outcome: RowOutcome,
    *,
    error_hash: str | None = None,
) -> None:
    """Validate sink exists in pending_tokens and append the token.

    Extracted from accumulate_row_outcomes where multiple outcome branches
    (COMPLETED, ROUTED, ROUTED_ON_ERROR, COALESCED) had identical
    validate+append logic.

    The `error_hash` keyword is required when ``pending_outcome`` is
    ``ROUTED_ON_ERROR`` (mirror of DIVERTED's contract — both outcomes are
    failure-handling redirects with an originating error that must be
    captured on the eventual outcome record). It must be None for ROUTED
    (intentional MOVE — no triggering error exists; passing a hash would
    be fabrication per CLAUDE.md Tier-1).

    Args:
        sink_name: Target sink name from result.sink_name
        pending_tokens: Sink-keyed accumulator to append to
        token: The token to route
        pending_outcome: The RowOutcome variant for the PendingOutcome
            (note: COALESCED tokens use COMPLETED here since they're
            finished from the sink's perspective)
        error_hash: 16-char sha256 prefix capturing the originating error;
            required for ROUTED_ON_ERROR, forbidden for ROUTED.
    """
    if sink_name not in pending_tokens:
        raise OrchestrationInvariantError(
            f"Sink '{sink_name}' not in configured sinks. "
            f"Available: {sorted(pending_tokens.keys())}. Token: {token}"
        )
    # Offensive: ROUTED_ON_ERROR requires error_hash; ROUTED forbids it.
    # The PendingOutcome __post_init__ also enforces this via _FAILURE_OUTCOMES,
    # but a clearer error message at the routing site catches producer bugs
    # before they hit the dataclass invariant.
    if pending_outcome == RowOutcome.ROUTED_ON_ERROR and error_hash is None:
        raise OrchestrationInvariantError(
            f"_route_to_sink: ROUTED_ON_ERROR requires error_hash, got None. "
            f"Token: {token}, sink: {sink_name}"
        )
    if pending_outcome == RowOutcome.ROUTED and error_hash is not None:
        raise OrchestrationInvariantError(
            f"_route_to_sink: ROUTED (intentional MOVE) must not carry error_hash; "
            f"got {error_hash!r}. Token: {token}, sink: {sink_name}"
        )
    pending_tokens[sink_name].append(
        (token, PendingOutcome(pending_outcome, error_hash=error_hash))
    )
```

**3c — Update the `accumulate_row_outcomes` accumulator branches at `outcomes.py:171-213`.** Change the `RowOutcome.ROUTED` branch (lines 176-180) and add a new `RowOutcome.ROUTED_ON_ERROR` branch:

```python
        elif result.outcome == RowOutcome.ROUTED:
            counters.rows_routed_success += 1
            sink_name = _require_sink_name(result)
            counters.routed_destinations[sink_name] += 1
            _route_to_sink(sink_name, pending_tokens, result.token, RowOutcome.ROUTED)
        elif result.outcome == RowOutcome.ROUTED_ON_ERROR:
            # ROUTED_ON_ERROR carries the originating transform error through to
            # the outcome record. The producer site (processor.py) sets
            # result.error to a FailureInfo capturing the upstream exception;
            # here we convert FailureInfo.message to a 16-char sha256 prefix
            # matching the existing pattern used by FAILED, QUARANTINED, and
            # DIVERTED at the recorder layer.
            if result.error is None:
                # Offensive: RowResult.__post_init__ should have caught this
                # at construction time. Reaching this branch with error=None
                # means the producer bypassed the invariant — Tier 1 violation.
                # This guard must run before ANY counter or routed_destinations
                # mutation, otherwise a malformed Tier-1 RowResult partially
                # mutates audit counters before crashing.
                raise OrchestrationInvariantError(
                    f"ROUTED_ON_ERROR result missing error (FailureInfo). "
                    f"Token: {result.token}"
                )
            sink_name = _require_sink_name(result)
            error_hash = hashlib.sha256(
                result.error.message.encode()
            ).hexdigest()[:16]
            counters.rows_routed_failure += 1
            counters.routed_destinations[sink_name] += 1
            _route_to_sink(
                sink_name,
                pending_tokens,
                result.token,
                RowOutcome.ROUTED_ON_ERROR,
                error_hash=error_hash,
            )
```

Note: both variants increment `routed_destinations` because the destination sink is tracked regardless of success vs failure routing intent — that map records "where rows landed" for audit, not "which routing decision was made". The `error_hash` flow ends at `_route_to_sink` → `PendingOutcome(ROUTED_ON_ERROR, error_hash=...)` → `pending_tokens[sink_name]`; from there, `SinkExecutor` at `src/elspeth/engine/executors/sink.py:631-638` reads `pending_outcome.error_hash` and forwards it to `record_token_outcome(...)` after sink durability is confirmed. The recorder's `_validate_outcome_fields` (extended in Task 4 Step 2) then enforces the contract.

- [ ] **Step 4: Replace `_decrement_counter` with direct outcome branches — counter AND routed_destinations cleanup for BOTH variants**

The inner function `_decrement_counter` is defined at `src/elspeth/engine/orchestrator/outcomes.py:90` and uses `getattr(counters, field_name)` / `setattr(counters, field_name, ...)`. Do NOT extend that pattern. CLAUDE.md bans dynamic typed-field access for invariant-sensitive code; the split counters must be updated by direct dataclass field access so typoed field names fail mechanically instead of becoming stringly runtime behaviour.

The existing `if pending_outcome.outcome == RowOutcome.ROUTED:` block runs at lines 104-118 and does THREE things — the replacement must preserve all three for both variants:

1. Decrement the per-outcome counter (`rows_routed` → split into `rows_routed_success` / `rows_routed_failure`).
2. Validate `routed_destinations[sink_name] >= diversion_count` (raises `OrchestrationInvariantError` on counter drift).
3. Decrement `routed_destinations[sink_name]`, deleting the key when remaining == 0.

The accumulator at Task 3 Step 3c increments `routed_destinations[sink_name]` for BOTH `RowOutcome.ROUTED` and `RowOutcome.ROUTED_ON_ERROR` (the destination map tracks "where rows landed" regardless of routing intent). Therefore the decrement path must clean up the destination map for BOTH variants — otherwise diverted ROUTED rows leave the map over-incremented (silent counter drift visible in audit records) and diverted ROUTED_ON_ERROR rows are not cleaned up at all.

Read `src/elspeth/engine/orchestrator/outcomes.py:85-130` first to confirm the surrounding control flow. Then replace the dynamic `_decrement_counter` helper and the following outcome branches with direct branches:

```python
    if pending_outcome.outcome == RowOutcome.COMPLETED:
        if counters.rows_succeeded < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted rows from "
                f"rows_succeeded={counters.rows_succeeded} for sink "
                f"{sink_name!r} and pending outcome {pending_outcome.outcome.value!r}. "
                "This indicates counter drift between processing and sink-write phases."
            )
        counters.rows_succeeded -= diversion_count
        return

    if pending_outcome.outcome == RowOutcome.ROUTED:
        if counters.rows_routed_success < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted rows from "
                f"rows_routed_success={counters.rows_routed_success} for sink "
                f"{sink_name!r} and pending outcome {pending_outcome.outcome.value!r}. "
                "This indicates counter drift between processing and sink-write phases."
            )
        counters.rows_routed_success -= diversion_count
        current_destination_count = counters.routed_destinations[sink_name]
        if current_destination_count < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted routed rows from "
                f"routed_destinations[{sink_name!r}]={current_destination_count} "
                f"(pending_outcome={pending_outcome.outcome.value!r}). "
                "This indicates counter drift between processing and sink-write phases."
            )
        remaining = current_destination_count - diversion_count
        if remaining == 0:
            del counters.routed_destinations[sink_name]
        else:
            counters.routed_destinations[sink_name] = remaining
        return

    if pending_outcome.outcome == RowOutcome.ROUTED_ON_ERROR:
        if counters.rows_routed_failure < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted rows from "
                f"rows_routed_failure={counters.rows_routed_failure} for sink "
                f"{sink_name!r} and pending outcome {pending_outcome.outcome.value!r}. "
                "This indicates counter drift between processing and sink-write phases."
            )
        counters.rows_routed_failure -= diversion_count
        current_destination_count = counters.routed_destinations[sink_name]
        if current_destination_count < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted routed rows from "
                f"routed_destinations[{sink_name!r}]={current_destination_count} "
                f"(pending_outcome={pending_outcome.outcome.value!r}). "
                "This indicates counter drift between processing and sink-write phases."
            )
        remaining = current_destination_count - diversion_count
        if remaining == 0:
            del counters.routed_destinations[sink_name]
        else:
            counters.routed_destinations[sink_name] = remaining
        return

    if pending_outcome.outcome == RowOutcome.QUARANTINED:
        if counters.rows_quarantined < diversion_count:
            raise OrchestrationInvariantError(
                f"Cannot subtract {diversion_count} diverted rows from "
                f"rows_quarantined={counters.rows_quarantined} for sink "
                f"{sink_name!r} and pending outcome {pending_outcome.outcome.value!r}. "
                "This indicates counter drift between processing and sink-write phases."
            )
        counters.rows_quarantined -= diversion_count
```

The ROUTED and ROUTED_ON_ERROR destination-cleanup blocks are intentionally duplicated rather than hidden behind dynamic counter-field names. Duplication here is cheaper than reintroducing `getattr` / `setattr` in Tier-1 counter accounting. The error-message expansion (`pending_outcome={pending_outcome.outcome.value!r}`) is a small but valuable diagnostic addition: if the destination-counter drift fires, the audit log identifies whether the drift originated from a MOVE-side or DIVERT-side outcome, accelerating root-cause analysis.

**Verification:** after the edit, run `pytest tests/unit/engine/orchestrator/test_outcomes.py -v -k "diversion or decrement"` — expected: existing tests pass for the ROUTED variant; add new tests in Task 8 Step 2 (the `test_outcomes.py` sweep) exercising `RowOutcome.ROUTED_ON_ERROR` through `accumulate_diversion_into_counters` with a non-empty `routed_destinations[sink_name]` to confirm the cleanup runs symmetrically.

- [ ] **Step 5: Verify L2 type definitions compile**

Run: `.venv/bin/python -c "from elspeth.engine.orchestrator.types import ExecutionCounters, AggregationFlushResult; c = ExecutionCounters(); print(c.rows_routed_success, c.rows_routed_failure); a = AggregationFlushResult(); print(a.rows_routed_success, a.rows_routed_failure)"`

Expected: `0 0` printed twice. If ImportError or TypeError, debug.

- [ ] **Step 6: Commit the L2 types change**

```bash
git add src/elspeth/engine/orchestrator/types.py src/elspeth/engine/orchestrator/outcomes.py
git commit -m "feat(engine): split ExecutionCounters.rows_routed into success/failure (elspeth-5069612f3c)"
```

---

## Task 4: L2 producers — processor.py + remaining RowOutcome consumers in src/

**Files:**
- Modify: `src/elspeth/engine/processor.py:2293-2298` (the on_error path)
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:230`
- Modify: `src/elspeth/core/landscape/model_loaders.py:545-579`
- Modify: `src/elspeth/testing/__init__.py` — three sites: `make_run_result()` factory at line 426 (parameter) + 445 (forward), `make_flush_result()` factory at line 460 (parameter) + 474 (forward), and `_SINK_OUTCOMES` at line 518.
- Test: `tests/unit/core/landscape/test_model_loaders.py` — add read-side `ROUTED_ON_ERROR` valid/missing-field loader tests.

**Goal of this task:** Switch the on_error producer site to emit the new variant. Update the three remaining consumer sites to handle both ROUTED and ROUTED_ON_ERROR.

- [ ] **Step 1: Update the on_error producer in processor.py**

In `src/elspeth/engine/processor.py:2293-2298`, change the `RowResult` construction. The producer must:
1. Switch outcome to `RowOutcome.ROUTED_ON_ERROR`
2. Construct a `FailureInfo` from the real `transform_result.reason`. Do NOT reuse the current `error_detail = str(transform_result.reason) if transform_result.reason else "unknown_error"` fallback for this new audit-bearing outcome. That fallback is tolerated only by the pre-existing QUARANTINED path until separately remediated; using it for `ROUTED_ON_ERROR` would fabricate Tier-1 audit data and create a deterministic `error_hash` collision across unrelated falsy-error failures.
3. Pass `error=failure` on the RowResult so the new RowResult invariant (Task 2 Step 3) is satisfied AND the accumulator can hash `failure.message` into `error_hash` before the pending-sink record is persisted.

```python
        if not transform_result.reason:
            raise OrchestrationInvariantError(
                "ROUTED_ON_ERROR requires transform_result.reason; refusing to "
                "fabricate FailureInfo.message='unknown_error' for audit hashing"
            )
        error_detail = str(transform_result.reason)

        # Capture the originating transform error so the audit trail records both
        # sink_name and error_hash on the ROUTED_ON_ERROR outcome (mirror of DIVERTED's
        # contract). The accumulator converts FailureInfo.message -> error_hash before
        # the pending-sink record is handed to SinkExecutor for durable recording.
        failure = FailureInfo(exception_type="TransformError", message=error_detail)
        current_result = RowResult(
            token=current_token,
            final_data=current_token.row_data,
            outcome=RowOutcome.ROUTED_ON_ERROR,
            sink_name=error_sink,
            error=failure,
        )
```

The `FailureInfo` class is already imported at `processor.py:63` (`from elspeth.contracts.results import FailureInfo`); no new import is needed. The `exception_type="TransformError"` constant matches the existing pattern at `processor.py:676` (the discard/quarantine FailureInfo uses the same constant). If the receiving sink config carries a more specific error-type label, prefer that — but the default `"TransformError"` is correct when the upstream `transform_result.reason` is a generic transform failure.

`OrchestrationInvariantError` is already imported at `processor.py:55`; no new import is needed. Add a focused regression in `tests/unit/engine/test_processor.py` (or the nearest existing processor on_error test class) that forces a transform-result error route with `transform_result.reason` falsy and asserts `OrchestrationInvariantError` before any `RowResult(ROUTED_ON_ERROR, ...)` or Landscape token outcome is recorded. This pins the no-fabricated-audit-data rule; do not assert on an `"unknown_error"` hash.

The gate-routed producer at line 2364 stays as `outcome=RowOutcome.ROUTED` with NO `error` field (intentional MOVE — no triggering error exists; passing `error=...` would be fabrication per CLAUDE.md Tier-1).

- [ ] **Step 2: Update Landscape data flow repository (contract-validation block)**

**Landscape audit distinguishability — resolution (in-scope, no follow-up):**

The Tier-1 audit-trail distinguishability between gate-routed (MOVE) and on_error-routed (DIVERT) tokens is satisfied by THIS PR with no additional Landscape schema change. Three concrete pieces of evidence:

1. **Primary distinguishability — `token_outcomes.outcome` itself.** `RowOutcome` is a `StrEnum` (`src/elspeth/contracts/enums.py:160`). `data_flow_repository.record_token_outcome` writes `outcome=outcome` directly to the `token_outcomes` table at `src/elspeth/core/landscape/data_flow_repository.py:853`. After this PR, the inserted string for transform on_error rows is the new `'routed_on_error'` value, and for gate-routed rows it remains `'routed'`. `explain(recorder, run_id, token_id)` reads the answer to "did this row succeed via gate route (MOVE) or fail via on_error (DIVERT)?" from a single column in a single table with no JOIN required. This is the strongest possible attributability signal.
2. **Secondary distinguishability — `RoutingEvent.mode`.** The `RoutingEvent` model already carries `mode: RoutingMode` (declared at `src/elspeth/contracts/audit.py:398`, validated by `__post_init__` `_validate_enum` check at line 406). Routing events are recorded against the edge that carried the row, so the auditor can cross-check the token-outcome answer against the edge's `RoutingMode` recorded at the time of routing. This is a redundant guarantee, useful for tier-1 cross-validation but not load-bearing.
3. **No additional schema change required.** The `token_outcomes` table accepts arbitrary `RowOutcome` StrEnum values (the column has no enum constraint on the SQL side; the producer is the contract). The new variant simply adds a previously-unseen string to the column's value domain. No Alembic, no `metadata.create_all` schema diff, no new column, no JOIN.

**Scoped biconditional only — do not assert a global DIVERT⇔ROUTED_ON_ERROR rule.** The empirically-pinned invariant for this PR is:

- gate `route_to_sink` MOVE edge ⇒ `token_outcomes.outcome == 'routed'` and `error_hash IS NULL`;
- transform `on_error` DIVERT edge ⇒ `token_outcomes.outcome == 'routed_on_error'` and `error_hash` is non-empty.

That invariant is intentionally scoped to the two producer sites this PR changes. A global statement like `routing_events.mode == 'divert' IFF token_outcomes.outcome == 'routed_on_error'` is false in the current architecture: `RoutingMode.DIVERT` is also used for source quarantine and sink failsink edges (`contracts/enums.py:146-150`, `engine/orchestrator/core.py:2264`, `engine/executors/sink.py:869`), which map to `RowOutcome.QUARANTINED` and `RowOutcome.DIVERTED`, respectively. Task 8 Step 9c's tests must therefore assert the producer-scoped biconditional above, not a global DIVERT-mode biconditional.

**Historical Landscape DB limitation / operator action.** Because this is a value-domain semantic split rather than a table/column migration, old Landscape rows survive unless the operator deletes/recreates the audit DB. Any pre-split `token_outcomes.outcome='routed'` row is legacy ambiguous: it may represent gate MOVE or transform on_error DIVERT under the old conflated taxonomy. For dev/staging/pre-1.0 deployments, the operator procedure is to archive/delete/recreate the Landscape audit DB during deployment. If a deployment preserves an old Landscape DB for retention, the release/runbook note must mark pre-split `routed` rows as accepted audit-limitation evidence and must forbid interpreting them as MOVE-only rows.

**The implementation step itself is small:** read `src/elspeth/core/landscape/data_flow_repository.py:203-285` (the `_validate_outcome_fields` helper). The `elif outcome == RowOutcome.ROUTED:` branch at line 230 enforces `sink_name is not None` for ROUTED outcomes. Add an analogous branch for `ROUTED_ON_ERROR`:

```python
        elif outcome == RowOutcome.ROUTED_ON_ERROR:
            if sink_name is None:
                raise ValueError(
                    "ROUTED_ON_ERROR outcome requires sink_name but got None. "
                    "Contract violation - see docs/contracts/token-outcomes/00-token-outcome-contract.md"
                )
            if error_hash is None:
                raise ValueError(
                    "ROUTED_ON_ERROR outcome requires error_hash but got None. "
                    "Mirrors DIVERTED's contract — both outcomes are failure-handling "
                    "redirects with an originating error that must be captured on the "
                    "outcome record for single-hop audit attributability. "
                    "Contract violation - see docs/contracts/token-outcomes/00-token-outcome-contract.md"
                )
```

This branch mirrors the existing `DIVERTED` branch at `data_flow_repository.py:274-282` exactly (both checks: `sink_name is not None` AND `error_hash is not None`). Place the branch immediately after the `ROUTED` branch (between line 235 and the `elif outcome == RowOutcome.FORKED:` branch at line 236). This is the only edit required to `data_flow_repository.py`.

While editing this helper, verify the terminal `else: raise ValueError(f"Unhandled RowOutcome variant...")` remains present after the final known outcome branch. Source reality on 2026-05-02 already has this guard at `data_flow_repository.py:290-293`; preserve it. If the worker's branch lacks it because of intervening edits, add it. Also add/keep a small unit test in `tests/unit/core/landscape/test_data_flow_repository.py` that passes an unsupported outcome sentinel through `_validate_outcome_fields` (or otherwise exercises the final branch) and asserts the "Unhandled RowOutcome variant" error, so future enum additions cannot silently bypass contract validation.

The token-outcomes contract document (`docs/contracts/token-outcomes/00-token-outcome-contract.md`) is updated separately in Task 8 Step 17.

- [ ] **Step 3: Update model loaders sink-name and error_hash validation**

In `src/elspeth/core/landscape/model_loaders.py:545-579`, `TokenOutcomeLoader.load()` is the read-side Tier-1 guard for persisted `token_outcomes` rows. Add BOTH the new sink-name guard and the `error_hash` guard. A `ROUTED_ON_ERROR` row with `sink_name` but `error_hash IS NULL` is audit corruption and must crash on load.

Keep the existing `COMPLETED/ROUTED` guard unchanged so current tests that assert its message continue to pin the old contract:

```python
        if outcome in (RowOutcome.COMPLETED, RowOutcome.ROUTED) and row.sink_name is None:
```

Immediately after it, add a dedicated `ROUTED_ON_ERROR` sink guard:

```python
        if outcome == RowOutcome.ROUTED_ON_ERROR and row.sink_name is None:
            raise AuditIntegrityError(
                f"TokenOutcome {oid} has outcome={outcome.value!r} but sink_name is NULL — "
                "audit integrity violation (ROUTED_ON_ERROR requires sink_name)"
            )
```

Then replace the DIVERTED-only `error_hash` check:

```python
        if outcome == RowOutcome.DIVERTED and row.error_hash is None:
            raise AuditIntegrityError(
                f"TokenOutcome {oid} has outcome=DIVERTED but error_hash is NULL — audit integrity violation (DIVERTED requires error_hash)"
            )
```

with:

```python
        if outcome in (RowOutcome.DIVERTED, RowOutcome.ROUTED_ON_ERROR) and row.error_hash is None:
            raise AuditIntegrityError(
                f"TokenOutcome {oid} has outcome={outcome.value!r} but error_hash is NULL — "
                f"audit integrity violation ({outcome.name} requires error_hash)"
            )
```

Leave the existing `FAILED/QUARANTINED` guard in place for synchronous failure outcomes. The resulting read-side contract is:

- `ROUTED`: requires `sink_name`; read-side loader does not require `error_hash`.
- `ROUTED_ON_ERROR`: requires `sink_name` and `error_hash`.
- `DIVERTED`: requires `sink_name` and `error_hash`.

Add the companion loader tests in `tests/unit/core/landscape/test_model_loaders.py`:

```python
    # In _OUTCOME_REQUIRED_FIELDS:
    "routed_on_error": {"sink_name": "failsink", "error_hash": "e" * 16},

    def test_valid_load_routed_on_error(self) -> None:
        """ROUTED_ON_ERROR outcome loads successfully with required fields."""
        sa_row = self._make_outcome_row(
            outcome="routed_on_error",
            is_terminal=1,
            sink_name="failsink",
            error_hash="abc123",
        )

        result = TokenOutcomeLoader().load(sa_row)

        assert result.outcome == RowOutcome.ROUTED_ON_ERROR
        assert result.sink_name == "failsink"
        assert result.error_hash == "abc123"

    def test_routed_on_error_without_sink_name_raises(self) -> None:
        """ROUTED_ON_ERROR outcome with NULL sink_name is audit corruption."""
        sa_row = self._make_outcome_row(
            outcome="routed_on_error",
            is_terminal=1,
            sink_name=None,
            error_hash="abc123",
        )

        with pytest.raises(AuditIntegrityError, match="ROUTED_ON_ERROR requires sink_name"):
            TokenOutcomeLoader().load(sa_row)

    def test_routed_on_error_without_error_hash_raises(self) -> None:
        """ROUTED_ON_ERROR outcome with NULL error_hash is audit corruption."""
        sa_row = self._make_outcome_row(
            outcome="routed_on_error",
            is_terminal=1,
            sink_name="failsink",
            error_hash=None,
        )

        with pytest.raises(AuditIntegrityError, match="ROUTED_ON_ERROR requires error_hash"):
            TokenOutcomeLoader().load(sa_row)
```

- [ ] **Step 4: Update testing helpers — both factories AND `_SINK_OUTCOMES`**

Run `grep -n "rows_routed" src/elspeth/testing/__init__.py` to confirm current line numbers; observed 2026-05-02: 426, 445, 460, 474.

**4a — `make_run_result()` factory (lines 426 and 445):**

In the parameter list at line 426, replace `rows_routed: int = 0,` with:

```python
    rows_routed_success: int = 0,
    rows_routed_failure: int = 0,
```

In the constructor call at line 445, replace `rows_routed=rows_routed,` with:

```python
        rows_routed_success=rows_routed_success,
        rows_routed_failure=rows_routed_failure,
```

**4b — `make_flush_result()` factory (lines 460 and 474):**

Same shape change — replace `rows_routed: int = 0,` at line 460 and `rows_routed=rows_routed,` at line 474 with the two-field forms above.

These two factories are widely consumed across the test suite. Updating them in this task (alongside the producer-site change) keeps the failure surface during Task 8 sweeps confined to direct test-side `rows_routed` references rather than factory-mediated indirect ones.

**4c — `_SINK_OUTCOMES` set (line 518):**

```python
    _SINK_OUTCOMES = {RowOutcome.COMPLETED, RowOutcome.ROUTED, RowOutcome.COALESCED}
```

Becomes:

```python
    _SINK_OUTCOMES = {RowOutcome.COMPLETED, RowOutcome.ROUTED, RowOutcome.ROUTED_ON_ERROR, RowOutcome.COALESCED}
```

**4d — `make_run_summary()` factory (lines 616-641):**

The factory at line 616 (`def make_run_summary(`) builds a `RunSummary` event. Verify the current shape with `grep -n "routed\b\|def make_run_summary" src/elspeth/testing/__init__.py`. Observed 2026-05-02: `def` at line 616, `routed: int = 0,` parameter at line 626, `routed=routed,` forwarding at line 641. All three must be replaced.

Replace the `routed: int = 0,` parameter at line 626 with:

```python
    routed_success: int = 0,
    routed_failure: int = 0,
```

Replace the `routed=routed,` forwarding at line 641 with:

```python
        routed_success=routed_success,
        routed_failure=routed_failure,
```

After all four sub-steps, run `grep -n "rows_routed\|\\brouted=\\|\\.routed\\b" src/elspeth/testing/__init__.py` — expected: zero matches (the factories no longer take or forward a bare `routed` or `rows_routed`).

- [ ] **Step 5: Find and update any remaining match/if on RowOutcome.ROUTED**

Run: `grep -rn "RowOutcome\.ROUTED\b" src/elspeth/ --include="*.py" | grep -v "ROUTED_ON_ERROR"`

For each result, decide: does this site care about routing intent (only intentional MOVE) or all routing? If only MOVE, leave as `ROUTED`. If all routing, expand to `(ROUTED, ROUTED_ON_ERROR)`. Most match statements that handle "the row reached a sink" want both; producers that emit ROUTED should stay as one or the other based on intent.

Likely remaining sites (verify each):
- `src/elspeth/engine/orchestrator/outcomes.py:104` — already handled in Task 3 step 4.
- `src/elspeth/engine/orchestrator/core.py:458` — the `match` statement; needs explicit branches for both. See Task 5.
- `src/elspeth/contracts/results.py:406` — already handled in Task 2 step 3.

- [ ] **Step 6: Verify the engine layer compiles**

Run: `.venv/bin/python -c "from elspeth.engine.processor import RowProcessor; from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes; from elspeth.core.landscape.data_flow_repository import DataFlowRepository; print('imports ok')"`

Expected: `imports ok`. Adjust class names in the import line if needed (locate via grep `class ` in the named files).

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/engine/processor.py src/elspeth/core/landscape/data_flow_repository.py src/elspeth/core/landscape/model_loaders.py src/elspeth/testing/__init__.py tests/unit/core/landscape/test_model_loaders.py
git commit -m "feat(engine,core): emit ROUTED_ON_ERROR for transform on_error path; expand sink-outcome consumers (elspeth-5069612f3c)"
```

---

## Task 5: L2 orchestrator core.py + CLI formatters — propagate the field rename through producer call sites and event consumers

**Files:**
- Modify: `src/elspeth/engine/orchestrator/core.py` (lines 458-459, 552, 578, 611, 1579, 1616, 1648, 2147, 2420, 2973, 3208, 3282) — 12 RunResult/ExecutionCounters/RunSummary call sites
- Modify: `src/elspeth/cli_formatters.py` (lines 58, 61, 141) — three `event.routed` consumer reads on the RunSummary event

**Goal of this task:** Update every site in `core.py` that reads or writes `rows_routed`, and the three `cli_formatters.py` sites that consume the RunSummary event. The two `rows_succeeded + rows_routed` additions at lines 2420 and 2973 become `rows_succeeded + rows_routed_success` (do not include rows_routed_failure in the progress UI's "successful" count — those are operator-visible failures). The CLI display SUMS the new RunSummary fields for operator-visible UX continuity; the JSON output emits the split fields explicitly.

- [ ] **Step 1: Read the match statement and its surrounding context**

Run: `sed -n '450,475p' src/elspeth/engine/orchestrator/core.py`

Note the `case RowOutcome.ROUTED:` handling and the existing comment about rows_routed exclusion.

- [ ] **Step 2: Rewrite the Landscape-query match block at `core.py:444-486` (post-resume / post-finalize terminal-status derivation)**

This block reads accumulated terminal `token_outcomes` records back from Landscape, walks them in a `match outcome.outcome:` block, accumulates per-status local counters, and feeds those locals into `derive_terminal_run_status(...)` at line 479. After the rows_routed split, the block must accumulate the new fields as locals AND pass them to the derive call. THIS IS THE PRIMARY FIX for the user's reproducer bug — the predicate change at L0 alone is not enough; the post-resume derivation also needs the new locals.

**Step 2a — initialize the new locals.** Find the local-counter initialization block immediately before the `for outcome in outcomes:` loop (around line 444 — typically `rows_succeeded = 0; rows_failed = 0; rows_quarantined = 0; rows_processed = 0`). Add two new locals:

```python
        rows_routed_success = 0
        rows_routed_failure = 0
```

**Step 2b — update the match block (lines 458-481).** The existing block has a `case RowOutcome.ROUTED:` branch with the comment "rows_routed: excluded from the predicate (DIVERT vs MOVE ambiguity); still counts as processed." After the split, both ROUTED and ROUTED_ON_ERROR are first-class terminal counters. Replace:

```python
                case RowOutcome.ROUTED:
                    # rows_routed: excluded from the predicate (DIVERT vs MOVE
                    # ambiguity); still counts as processed.
                    rows_processed += 1
```

with:

```python
                case RowOutcome.ROUTED:
                    # Intentional gate MOVE — counts as a success indicator
                    # in the predicate (rows_routed_success > 0).
                    rows_routed_success += 1
                    rows_processed += 1
                case RowOutcome.ROUTED_ON_ERROR:
                    # Transform on_error DIVERT — counts as a failure indicator
                    # in the predicate (rows_routed_failure > 0).
                    rows_routed_failure += 1
                    rows_processed += 1
```

**Step 2c — update the `derive_terminal_run_status(...)` call at line 479.** Add the two new kwargs:

```python
        terminal_status = derive_terminal_run_status(
            rows_processed=rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            rows_routed_success=rows_routed_success,
            rows_routed_failure=rows_routed_failure,
            rows_quarantined=rows_quarantined,
            rows_coalesce_failed=rows_coalesce_failed,
        )
```

**Step 2d — update the return tuple at line 487 (or thereabouts) to include the new locals.** Verify with `sed -n '485,495p' src/elspeth/engine/orchestrator/core.py` first. The current return is `return terminal_status, rows_processed, rows_succeeded, rows_failed, rows_quarantined`. The new return must include the new counters so the caller can pass them through to subsequent constructions. Replace with:

```python
        return (
            terminal_status,
            rows_processed,
            rows_succeeded,
            rows_failed,
            rows_routed_success,
            rows_routed_failure,
            rows_quarantined,
        )
```

Then locate the caller that unpacks this return value (`grep -n "_landscape_terminal_status\|terminal_status, rows_processed" src/elspeth/engine/orchestrator/core.py` — observe the exact name) and update the unpack to receive the two new locals. The caller likely uses these to construct a `RunResult`, so the new locals flow naturally into the `rows_routed_success=`, `rows_routed_failure=` kwargs added in Task 5 Step 3 below.

**Step 2e — verify the other two `derive_terminal_run_status` call sites.** Two additional invocations exist at lines 1524 and 3247 (verify with `grep -n "derive_terminal_run_status\b" src/elspeth/engine/orchestrator/core.py`). Both read fields off an existing `result: RunResult` rather than from local accumulators. After Task 2 lands the L0 split, `result.rows_routed_success` and `result.rows_routed_failure` exist and must be passed:

```python
        terminal_status = derive_terminal_run_status(
            rows_processed=result.rows_processed,
            rows_succeeded=result.rows_succeeded,
            rows_failed=result.rows_failed,
            rows_routed_success=result.rows_routed_success,
            rows_routed_failure=result.rows_routed_failure,
            rows_quarantined=result.rows_quarantined,
            rows_coalesce_failed=result.rows_coalesce_failed,
        )
```

Apply the same shape to BOTH lines 1524 and 3247.

**Step 2f — preserve the inline match-statement RowOutcome handling for the row-result-flow path.** Separately from the Landscape-query block above, the engine also has an inline `match` statement on `RowOutcome` (the original Step 2 anchor). Add a parallel `case RowOutcome.ROUTED_ON_ERROR:` branch immediately after the existing `case RowOutcome.ROUTED:` branch. Both branches share the same handling (the row reached a sink, terminal):

```python
                # ROUTED counts as rows_routed_success (intentional gate MOVE).
                # ROUTED_ON_ERROR counts as rows_routed_failure (transform on_error DIVERT).
                # Both reach a sink; both are terminal.
```

Locate this block via `grep -n "case RowOutcome.ROUTED:" src/elspeth/engine/orchestrator/core.py` — there may be more than one match block; identify each by its surrounding context (Landscape-query reads `token_outcomes`; the row-flow match handles fresh outcomes from the executor) and apply the new variant branch to each.

- [ ] **Step 3: Update each `rows_routed=`, `routed=`, and `counters.rows_routed` site**

For each of the line numbers below, read 5 lines of context with `sed -n '<L-2>,<L+3>p' src/elspeth/engine/orchestrator/core.py`, then:

| Line | Pattern | Replacement |
|------|---------|-------------|
| 552  | `routed=shutdown_exc.rows_routed,` | `routed_success=shutdown_exc.rows_routed_success, routed_failure=shutdown_exc.rows_routed_failure,` |
| 578  | `rows_routed=0,` | `rows_routed_success=0, rows_routed_failure=0,` |
| 611  | `routed=failed_result.rows_routed,` | `routed_success=failed_result.rows_routed_success, routed_failure=failed_result.rows_routed_failure,` |
| 1579 | `routed=result.rows_routed,` | `routed_success=result.rows_routed_success, routed_failure=result.rows_routed_failure,` |
| 1616 | (same shape) | (same replacement) |
| 1648 | (same shape) | (same replacement) |
| 2147 | `rows_routed=counters.rows_routed,` | `rows_routed_success=counters.rows_routed_success, rows_routed_failure=counters.rows_routed_failure,` |
| 2420 | `rows_succeeded=counters.rows_succeeded + counters.rows_routed,` | `rows_succeeded=counters.rows_succeeded + counters.rows_routed_success,` (and update the comment "Include routed rows in success count" to "Include intentionally-routed rows (MOVE) in success count; exclude on_error-routed rows (DIVERT) — those are operator-visible failures") |
| 2973 | (same shape as 2420) | (same replacement; same comment update) |
| 3208 | `rows_routed=0,` | `rows_routed_success=0, rows_routed_failure=0,` |
| 3282 | `routed=result.rows_routed,` | `routed_success=result.rows_routed_success, routed_failure=result.rows_routed_failure,` |

Note: the `routed=` keyword on these call sites is the keyword argument to the `RunSummary` constructor (defined in `src/elspeth/contracts/events.py`). After Task 2 Step 9b splits `RunSummary.routed` → `routed_success` / `routed_failure`, the call sites in this table type-check naturally with the replacements shown. There is no separate `record_routing_event`-style helper between `core.py` and `RunSummary` — the constructor IS the receiver. Confirm with `grep -n "RunSummary(" src/elspeth/engine/orchestrator/core.py` (expected: 6+ sites, each preceded by `RunSummary(...routed=...)` and followed by the next-line `routed_destinations=...`).

- [ ] **Step 4: Update `cli_formatters.py` consumer reads on `RunSummary`**

Three `event.routed` reads exist in `src/elspeth/cli_formatters.py` (verify with `grep -n "event\.routed\b" src/elspeth/cli_formatters.py`). The display strategy is **sum-for-UX-continuity**: console output continues to show one combined "→N routed" total so operator-visible behaviour does not change; JSON output emits the split fields explicitly for programmatic consumers.

At line 58, replace `if event.routed > 0:` with:

```python
        if (event.routed_success + event.routed_failure) > 0:
```

At line 61, replace `routed_summary = f" | →{event.routed:,} routed"` with:

```python
            routed_summary = f" | →{event.routed_success + event.routed_failure:,} routed"
```

At line 141 (inside `_format_run_summary_json`), replace `"routed": event.routed,` with two parallel lines:

```python
                    "routed_success": event.routed_success,
                    "routed_failure": event.routed_failure,
```

The JSON output drops the old `"routed"` key entirely — no transitional alias (no-legacy-code policy).

After this step, run `grep -n "event\.routed\b" src/elspeth/cli_formatters.py` — expected: zero matches.

- [ ] **Step 5: Update the receiving helper function signatures (if any)**

If any `routed=` call site failed to type-check (other than `RunSummary` itself, which Step 9b in Task 2 already updated), the receiving function (likely in `core/landscape/`) needs its parameter renamed too. Make the change in the same task to keep the build green.

- [ ] **Step 6: Verify the engine layer compiles end-to-end**

Run: `.venv/bin/python -c "from elspeth.engine.orchestrator.core import Orchestrator; print('orchestrator imports ok')"`

Expected: `orchestrator imports ok`. If TypeError or ImportError on field names, hunt the missed site with `grep -n "rows_routed\b" src/elspeth/engine/`.

- [ ] **Step 7: Run the L0 unit tests again — they should still pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_run_result.py::TestRunStatusRowsRoutedSplitPredicate -v`

Expected: 19 passed. (Regression check — Task 5 should not have broken Task 1's tests; the positive-shape tests, matrix tests, Hypothesis biconditional property, and negative-invariant tests must continue to pass after the orchestrator-core propagation lands.)

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/engine/orchestrator/core.py src/elspeth/cli_formatters.py
git commit -m "refactor(engine,cli): propagate rows_routed split through orchestrator core and CLI formatters (elspeth-5069612f3c)"
```

---

## Task 6: L3 sessions DB layer — RunRecord, models, service CRUD

**Files:**
- Modify: `src/elspeth/web/sessions/models.py:141`
- Modify: `src/elspeth/web/sessions/protocol.py:203, 374` (RunRecord field + constructor)
- Modify: `src/elspeth/web/sessions/service.py` (lines 587, 630, 684-685, 910, 1303)

**Goal of this task:** Update the persistence layer to store the two new counters. Drop the old column. Existing dev/staging DBs need recreation — there's no Alembic migration; the model declaration drives schema.

- [ ] **Step 1: Update the SQLAlchemy column declaration**

In `src/elspeth/web/sessions/models.py:141`, replace:

```python
    Column("rows_routed", Integer, nullable=False, default=0),
```

with:

```python
    Column("rows_routed_success", Integer, nullable=False, default=0),
    Column("rows_routed_failure", Integer, nullable=False, default=0),
```

- [ ] **Step 2: Update the RunRecord dataclass field**

In `src/elspeth/web/sessions/protocol.py:203` (look for the `rows_routed: int` line in the `RunRecord` dataclass), replace with:

```python
    rows_routed_success: int
    rows_routed_failure: int
```

- [ ] **Step 3: Verify `RunRecord.__post_init__` has NO row-count predicate to update**

Verified 2026-05-02 against `src/elspeth/web/sessions/protocol.py:209`: `RunRecord.__post_init__` enforces only Tier-1 invariants on `status` enum validity, `finished_at` presence on terminal statuses, `landscape_run_id` presence on operator-completion statuses, and `error` string presence on `status='failed'`. It does NOT enforce the row-count biconditional predicate (the predicate is mirrored at three other sites — see the Constraints section). **No edit to `__post_init__` is required in this task.** The dataclass FIELD updates land in Step 2 above; the existing non-row-count invariants stay as-is.

This is a pre-existing audit-coverage gap (a `runs` row could in principle be persisted with `status='completed'` and zero success indicators and `RunRecord` would not crash on read). The gap is OUT OF SCOPE for this PR — adding the row-count predicate to `RunRecord.__post_init__` would gold-plate a missing invariant during a counter rename. **Action item: open a filigree observation flagging the gap as separate follow-up work.** Suggested observation text: "L3 `RunRecord.__post_init__` does not enforce the row-count biconditional predicate that L0 `RunResult.__post_init__` and L3 `_check_status_row_count_invariant` enforce. A `runs` row with inconsistent `(status, row-count)` shape can be deserialized without crashing — Tier-1 read-side audit gap. See `elspeth-5069612f3c` for the in-flight counter split that surfaced this."

- [ ] **Step 4: Update the RunRecord constructor parameter**

In `src/elspeth/web/sessions/protocol.py:374` (the constructor or factory), replace the `rows_routed: int | None = None` parameter with:

```python
        rows_routed_success: int | None = None,
        rows_routed_failure: int | None = None,
```

- [ ] **Step 5: Update sessions/service.py CRUD sites**

For each of lines 587, 630, 684-685, 910, 1303 in `src/elspeth/web/sessions/service.py`, read context with `sed -n '<L-3>,<L+5>p'`, then:

- Line 587: `rows_routed=0,` → `rows_routed_success=0, rows_routed_failure=0,`
- Line 630: `rows_routed: int | None = None,` → two parameter lines
- Lines 684-685: `if rows_routed is not None: values["rows_routed"] = rows_routed` → two parallel `if` blocks for the new fields
- Line 910: `rows_routed=row.rows_routed,` → two lines for the new column reads
- Line 1303: (same shape as 910)

- [ ] **Step 6: Verify the sessions layer compiles**

Run: `.venv/bin/python -c "from elspeth.web.sessions.protocol import RunRecord; from elspeth.web.sessions.models import metadata; print(sorted(c.name for c in metadata.tables['runs'].columns))"`

Expected: a sorted column list including `rows_routed_failure` and `rows_routed_success` and NOT `rows_routed`.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/sessions/models.py src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/service.py
git commit -m "feat(web/sessions): persist rows_routed_success/failure split (elspeth-5069612f3c)"
```

---

## Task 7: L3 web — execution schemas, routes, and service message

**Files:**
- Modify: `src/elspeth/web/execution/schemas.py` (lines 109-153 `_validate_row_decomposition`, 179-260 `_check_status_row_count_invariant`, plus four Pydantic models that carry `rows_routed: int = Field(default=0, ge=0)` — observed at lines 274, 295 `CancelledData`, 533, 591; verify with grep before editing)
- Modify: `src/elspeth/web/execution/routes.py` (lines 129, 148, 587)
- Modify: `src/elspeth/web/execution/service.py` — eight `rows_routed` sites: 133 (`_structural_failure_message`), 510 (RunRecord readback), 637/658 (CancelledData), 862/886/936 (completed-event), 954/967 (graceful-shutdown event)

**Goal of this task:** Update the API surface — Pydantic models, the row-decomposition invariant, the biconditional invariant mirror, and the structural failure message. After this task, the entire src/ tree builds cleanly.

- [ ] **Step 1: Update `_validate_row_decomposition` signature and formula**

In `src/elspeth/web/execution/schemas.py:109-153`, replace the function signature and the `sum_terminal` formula:

```python
def _validate_row_decomposition(
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> None:
    """Enforce rows_processed >= succeeded + failed + routed_success + routed_failure + quarantined.

    elspeth-5069612f3c — rows_routed split into rows_routed_success (MOVE) and
    rows_routed_failure (DIVERT). Both contribute to terminal-state counts.

    NARROW INVARIANT (elspeth-31d53c7493 carry-forward). The original equality
    formulation does not hold for any DAG with aggregation, fork, expansion,
    or coalesce — source rows reach terminal states the formula does not
    account for. The relaxed inequality is preserved here.

    The architecturally-correct formula (full DAG-aware balance) is tracked
    in elspeth-cf84eb1b52. When that lands, this inequality is replaced by
    the full balance.
    """
    sum_terminal = (
        rows_succeeded
        + rows_failed
        + rows_routed_success
        + rows_routed_failure
        + rows_quarantined
    )
    if rows_processed < sum_terminal:
        raise ValueError(
            f"Row count decomposition mismatch (over-counting): rows_processed={rows_processed} "
            f"< rows_succeeded({rows_succeeded}) + rows_failed({rows_failed}) "
            f"+ rows_routed_success({rows_routed_success}) "
            f"+ rows_routed_failure({rows_routed_failure}) "
            f"+ rows_quarantined({rows_quarantined}) = {sum_terminal}. "
            f"Tier 1 anomaly: orchestrator emitted more terminal-state counts than input rows. "
            f"See elspeth-cf84eb1b52 for the full DAG-aware balance equation."
        )
```

- [ ] **Step 2: Update `_check_status_row_count_invariant` signature and predicate**

In `src/elspeth/web/execution/schemas.py:179-260`, replace the function. Same unified-presence-indicator predicate as the L0 mirror, narrowed to the API-surface fields (no `rows_coalesce_failed` per the existing docstring rationale):

```python
def _check_status_row_count_invariant(
    status: str,
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> None:
    """elspeth-5069612f3c — Pydantic mirror of the L0 biconditional after the
    rows_routed split.

    success_indicator = rows_succeeded > 0 OR rows_routed_success > 0
    failure_indicator = rows_failed > 0 OR rows_quarantined > 0
                        OR rows_routed_failure > 0

    The Pydantic mirror does NOT see rows_coalesce_failed (the API schema
    does not surface it) — see the original docstring for the rationale;
    every coalesce failure also increments rows_failed at the engine layer,
    so the failure indicator is preserved.

    Non-terminal (running / pending) and signal-bounded (cancelled) statuses
    bypass the predicate.
    """
    success_indicator = rows_succeeded > 0 or rows_routed_success > 0
    failure_indicator = (
        rows_failed > 0
        or rows_quarantined > 0
        or rows_routed_failure > 0
    )

    if status in {"running", "pending", "cancelled"}:
        return

    if status == "completed":
        if not success_indicator:
            raise ValueError(
                f"status='completed' requires a success indicator "
                f"(rows_succeeded > 0 or rows_routed_success > 0); "
                f"got rows_succeeded={rows_succeeded}, "
                f"rows_routed_success={rows_routed_success} "
                f"(use status='empty' for ingested-zero-rows runs, "
                f"'failed' when rows were processed but none reached a success path)"
            )
        if failure_indicator:
            raise ValueError(
                f"status='completed' requires no failures "
                f"(rows_failed={rows_failed}, "
                f"rows_quarantined={rows_quarantined}, "
                f"rows_routed_failure={rows_routed_failure}); "
                f"use status='completed_with_failures'"
            )
        return

    if status == "completed_with_failures":
        if not success_indicator:
            raise ValueError(
                f"status='completed_with_failures' requires a success indicator "
                f"(rows_succeeded > 0 or rows_routed_success > 0); "
                f"got rows_succeeded={rows_succeeded}, "
                f"rows_routed_success={rows_routed_success} "
                f"(use status='failed' when no row reached a success path)"
            )
        if not failure_indicator:
            raise ValueError(
                f"status='completed_with_failures' requires at least one failure indicator "
                f"(rows_failed > 0 or rows_quarantined > 0 or rows_routed_failure > 0); "
                f"got rows_failed={rows_failed}, rows_quarantined={rows_quarantined}, "
                f"rows_routed_failure={rows_routed_failure} "
                f"(use status='completed' for clean runs)"
            )
        return

    if status == "failed":
        return  # FAILED tolerates any shape (predicate-or-exception origin)

    if status == "empty":
        if rows_processed != 0:
            raise ValueError(
                f"status='empty' requires rows_processed == 0, got {rows_processed}"
            )
        if success_indicator:
            raise ValueError(
                f"status='empty' requires no success indicator "
                f"(rows_succeeded={rows_succeeded}, "
                f"rows_routed_success={rows_routed_success})"
            )
        if failure_indicator:
            raise ValueError(
                f"status='empty' requires no failures "
                f"(rows_failed={rows_failed}, "
                f"rows_quarantined={rows_quarantined}, "
                f"rows_routed_failure={rows_routed_failure}); "
                f"use status='failed' when the run encountered failures with no successful rows"
            )
        return

    raise ValueError(f"Unknown status {status!r}")
```

- [ ] **Step 3: Update the four Pydantic response models**

There are FOUR Pydantic models in this file that declare `rows_routed: int = Field(default=0, ge=0)`. Run `grep -n "rows_routed" src/elspeth/web/execution/schemas.py` to confirm current line numbers (observed 2026-05-02: declarations at 274, 295, 533, 591; `model_validator` self.rows_routed reads at 284, 563, 603). The four models:

1. Line 274 — primary status response model (the `model_validator` at line 284 reads `self.rows_routed`).
2. Line 295 — `CancelledData` (cancelled-run payload; constructed by `web/execution/service.py` at lines 637 and 658). NOTE: this model was added subsequent to earlier review passes and is easy to miss.
3. Line 533 — historical/results model (the `model_validator` at line 563 reads `self.rows_routed`).
4. Line 591 — additional results-shape model (the `model_validator` at line 603 reads `self.rows_routed`).

For each:

1. Find the `rows_routed: int = Field(default=0, ge=0)` line in the model.
2. Replace with two lines:
   ```python
       rows_routed_success: int = Field(default=0, ge=0)
       rows_routed_failure: int = Field(default=0, ge=0)
   ```
3. Find the `model_validator` that calls `_check_status_row_count_invariant` (or `_validate_row_decomposition`) and update the call to pass `self.rows_routed_success` and `self.rows_routed_failure` instead of `self.rows_routed`. (Note: `CancelledData` constructions in `service.py` pass `rows_routed=0` literally — those construction sites are updated in Task 7 Step 5b, not here.)
4. After this step run `grep -n "rows_routed" src/elspeth/web/execution/schemas.py` — expected: zero matches.

- [ ] **Step 4: Update execution/routes.py**

For each of lines 129, 148, 587 in `src/elspeth/web/execution/routes.py`, replace `rows_routed=current.rows_routed,` (or similar) with two lines for the new fields. Read 5 lines of context around each to confirm the surrounding constructor argument shape.

- [ ] **Step 5: Update the `_structural_failure_message` phrasing**

In `src/elspeth/web/execution/service.py:133` (function definition), replace the helper:

```python
def _structural_failure_message(*, rows_processed: int) -> str:
    """elspeth-0de989c56d / elspeth-5069612f3c — synthetic structural error
    for FAILED-from-row-shape after the rows_routed split.

    The L3 RunRecord.__post_init__ invariant requires a non-empty error for
    status='failed'. When the engine returns RunStatus.FAILED from a row-shape
    decision (no exception propagated; no success indicator: rows_succeeded == 0
    AND rows_routed_success == 0), this helper produces a structural fact —
    operator-readable, no candidate-secret material, no echoed user-row data.

    After elspeth-5069612f3c, gate-routed pipelines (rows_routed_success > 0)
    no longer reach this code path — they classify as COMPLETED. This message
    fires only when no row reached EITHER the success-counted terminal state
    OR an intentional gate-routed sink, i.e. when every row failed terminally
    or was diverted via on_error.
    """
    return (
        f"No row reached a success path (rows_processed={rows_processed}, "
        f"rows_succeeded=0, rows_routed_success=0). "
        f"All rows either failed terminally or were routed via on_error to a "
        f"failure sink. Inspect /diagnostics for per-row failure details."
    )
```

The function signature stays the same (`rows_processed` only) — the message body is the only change.

- [ ] **Step 5b: Update the remaining seven `rows_routed` sites in `web/execution/service.py`**

Read each site with `grep -n "rows_routed" src/elspeth/web/execution/service.py` to confirm current line numbers, then apply:

| Line | Current text | Replacement |
|------|--------------|-------------|
| 510  | `rows_routed=run.rows_routed,` | `rows_routed_success=run.rows_routed_success,`<br>`rows_routed_failure=run.rows_routed_failure,` |
| 637  | `CancelledData(rows_processed=0, rows_failed=0, rows_routed=0)` | `CancelledData(rows_processed=0, rows_failed=0, rows_routed_success=0, rows_routed_failure=0)` |
| 658  | (same as 637) | (same replacement) |
| 862  | `rows_routed=result.rows_routed,` | `rows_routed_success=result.rows_routed_success,`<br>`rows_routed_failure=result.rows_routed_failure,` |
| 886  | (same as 862) | (same replacement) |
| 936  | (same as 862) | (same replacement) |
| 954  | `rows_routed=gse.rows_routed,` | `rows_routed_success=gse.rows_routed_success,`<br>`rows_routed_failure=gse.rows_routed_failure,` |
| 967  | (same as 954) | (same replacement) |

After this step, run `grep -n "rows_routed" src/elspeth/web/execution/service.py` again — expected output: empty (zero matches).

- [ ] **Step 6: Verify the web layer compiles**

Run: `.venv/bin/python -c "from elspeth.web.execution.schemas import RunStatusResponse, RunResultsResponse; print('schemas ok')"`

Expected: `schemas ok`. If Pydantic raises validation errors at class-construction time, an invariant call still references `rows_routed`. Hunt with `grep -n "rows_routed\b" src/elspeth/web/execution/schemas.py`.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/execution/schemas.py src/elspeth/web/execution/routes.py src/elspeth/web/execution/service.py
git commit -m "feat(web/execution): split rows_routed in API schemas and update structural failure message (elspeth-5069612f3c)"
```

---

## Task 8: Test sweep, integration reproducer, docs, close downstream P1

**Files:**
- Modify: `tests/unit/engine/orchestrator/test_outcomes.py`
- Modify: `tests/unit/engine/orchestrator/test_types.py`
- Modify: `tests/unit/engine/orchestrator/test_aggregation.py`
- Modify: `tests/unit/contracts/test_run_result.py` (sweep the pre-existing tests at line 289 etc.)
- Modify: `tests/unit/contracts/test_diverted_outcome.py` (one site at line 27 — verify with grep — `rows_routed=0` is incidental to the test, replace with `rows_routed_success=0, rows_routed_failure=0`).
- Modify: `tests/unit/contracts/test_freeze_regression.py:342`
- Modify: `tests/unit/engine/orchestrator/test_graceful_shutdown.py` — three sites (verify with `grep -n "rows_routed" tests/unit/engine/orchestrator/test_graceful_shutdown.py`; observed 2026-05-02: lines 57, 63, 72): a `GracefulShutdownError(rows_routed=5, ...)` construction at 57 and two assertions on `err.rows_routed`. Distinct from the integration-suite shutdown test at `tests/integration/pipeline/orchestrator/test_graceful_shutdown.py:498` listed below.
- Modify: `tests/integration/pipeline/orchestrator/test_t18_characterization.py:201`
- Modify: `tests/integration/pipeline/orchestrator/test_graceful_shutdown.py:498`
- Modify: `tests/integration/pipeline/test_composer_runtime_agreement.py` — TWO test classes touched (verify line numbers via grep before editing):
  - The first (around lines 2350-2543, `test_runstatus_rows_routed_only_classifies_as_failed`) is the locked-in-buggy-behavior test inverted in Step 8 below.
  - A SECOND class around lines 2620-2830 (`agreement_batch_aggregate` / `test_agreement_aggregation_run_counts_construct_completed_data`) has additional `rows_routed` references at observed lines 2727 (docstring), 2804 (`run_result.rows_routed` read in a sum-of-buckets assertion), and 2820 (`rows_routed=run_result.rows_routed` keyword in a `CompletedData` construction). Update the docstring to reference the split, replace `run_result.rows_routed` with `run_result.rows_routed_success + run_result.rows_routed_failure` in the sum-of-buckets check (so the inequality assertion still holds), and replace the `CompletedData(rows_routed=...)` construction with `rows_routed_success=run_result.rows_routed_success, rows_routed_failure=run_result.rows_routed_failure`. This class is the elspeth-31d53c7493 row-decomposition regression pin — leaving it un-updated would silently break Task 7's `_validate_row_decomposition` rewrite at integration time.
- Modify: `docs/architecture/adr/004-adr-explicit-sink-routing.md`
- Modify: `docs/contracts/token-outcomes/00-token-outcome-contract.md`

**Goal of this task:** Sweep every `rows_routed` reference in tests; add the integration test that reproduces the user's bug; update docs; verify the full suite + tier-model enforcer; close the downstream P1.

- [ ] **Step 1: Sweep `tests/unit/engine/orchestrator/test_types.py`**

Run: `grep -n "rows_routed" tests/unit/engine/orchestrator/test_types.py`

For each match (lines 35, 47, 68, 81, 92, 105, 118, 129, 145, 160), update the test fixture: replace `rows_routed=N` kwargs with `rows_routed_success=N, rows_routed_failure=0` (or split N appropriately if the test asserts both). Replace `result.rows_routed == N` reads with assertions on the two new fields.

The default behavior to assume: tests that don't explicitly intend on_error semantics are exercising the MOVE path, so `rows_routed_success=N` is the right replacement.

- [ ] **Step 2: Sweep `tests/unit/engine/orchestrator/test_outcomes.py`**

Run: `grep -n "RowOutcome\.ROUTED\|rows_routed" tests/unit/engine/orchestrator/test_outcomes.py`

For each test that constructs `_make_result(RowOutcome.ROUTED, ...)`, decide whether the test exercises intentional MOVE (keep `ROUTED`) or on_error DIVERT (change to `ROUTED_ON_ERROR`). The test names usually disambiguate (e.g., `test_routed_to_risk_sink` is MOVE; `test_on_error_routed` is DIVERT).

For each `rows_routed` counter assertion, replace with the appropriate `rows_routed_success` or `rows_routed_failure` assertion based on the variant under test.

Add at least two new tests in this file: one that asserts `accumulate_row_outcomes` increments `rows_routed_success` for `RowOutcome.ROUTED` results, and one that asserts it increments `rows_routed_failure` for `RowOutcome.ROUTED_ON_ERROR` results.

- [ ] **Step 3: Sweep `tests/unit/engine/orchestrator/test_aggregation.py`**

For lines 632 and 970, replace `result.rows_routed == 1` with `result.rows_routed_success == 1` (assuming the test exercises MOVE; verify the test's setup uses `RowOutcome.ROUTED` and not `ROUTED_ON_ERROR`).

- [ ] **Step 4: Sweep `tests/unit/contracts/test_run_result.py` (pre-existing tests)**

Run: `grep -n "rows_routed" tests/unit/contracts/test_run_result.py | head -30`

For each pre-existing test (lines 29, 38, 56, 68, 86, 98, 109, 180, 233, 289-295), update the kwargs and assertion shape. The test at lines 289-295 (`rows_routed-only design call`) is the locked-in-buggy-behavior test — REMOVE it (the new `TestRunStatusRowsRoutedSplitPredicate` class added in Task 1 covers the corrected semantics).

- [ ] **Step 5: Sweep `tests/unit/contracts/test_freeze_regression.py:342` — REQUIRED, NOT OPTIONAL**

The existing line constructs `PendingOutcome(outcome=RowOutcome.ROUTED)`. Because Task 2 Step 9c admits `RowOutcome.ROUTED_ON_ERROR` to `PendingOutcome._FAILURE_OUTCOMES`, the dataclass `__post_init__` now treats ROUTED and ROUTED_ON_ERROR ASYMMETRICALLY: `error_hash` is REQUIRED (and must be non-empty) for `ROUTED_ON_ERROR`, FORBIDDEN for `ROUTED`. The freeze-regression suite must pin this new contract — leaving it unpinned would let a future regression silently break the audit-trail invariant.

Add three companion test cases adjacent to the existing `RowOutcome.ROUTED` test at line 342:

```python
def test_pending_outcome_routed_on_error_with_valid_hash_succeeds() -> None:
    """ROUTED_ON_ERROR + non-empty error_hash is the contract-conforming shape."""
    outcome = PendingOutcome(
        outcome=RowOutcome.ROUTED_ON_ERROR,
        error_hash="0123456789abcdef",
    )
    assert outcome.outcome == RowOutcome.ROUTED_ON_ERROR
    assert outcome.error_hash == "0123456789abcdef"


def test_pending_outcome_routed_on_error_without_hash_raises() -> None:
    """ROUTED_ON_ERROR without error_hash violates the _FAILURE_OUTCOMES contract."""
    with pytest.raises(ValueError, match="must have a non-empty error_hash"):
        PendingOutcome(outcome=RowOutcome.ROUTED_ON_ERROR)


def test_pending_outcome_routed_on_error_with_empty_hash_raises() -> None:
    """Empty-string error_hash counts as missing per `__post_init__` whitespace check."""
    with pytest.raises(ValueError, match="must have a non-empty error_hash"):
        PendingOutcome(outcome=RowOutcome.ROUTED_ON_ERROR, error_hash="")
```

After this step, run `pytest tests/unit/contracts/test_freeze_regression.py -v -k "routed"` — expected: four tests passing (the existing ROUTED test plus the three new ROUTED_ON_ERROR cases).

- [ ] **Step 6: Sweep `tests/integration/pipeline/orchestrator/test_t18_characterization.py:201`**

Replace `assert result.rows_routed == 0` with `assert result.rows_routed_success == 0 and result.rows_routed_failure == 0`.

- [ ] **Step 7: Sweep `tests/integration/pipeline/orchestrator/test_graceful_shutdown.py:498`**

Replace `assert exc_info.value.rows_routed == 0` with `assert exc_info.value.rows_routed_success == 0 and exc_info.value.rows_routed_failure == 0`.

- [ ] **Step 7b: Sweep `tests/unit/engine/orchestrator/test_graceful_shutdown.py` (unit-suite — distinct from the integration test above)**

Run: `grep -n "rows_routed" tests/unit/engine/orchestrator/test_graceful_shutdown.py`

Observed 2026-05-02: line 57 (`rows_routed=5,` in a `GracefulShutdownError(...)` construction), line 63 (`assert err.rows_routed == 5`), line 72 (`assert err.rows_routed == 0`).

- Line 57: replace `rows_routed=5,` with `rows_routed_success=5, rows_routed_failure=0,` (the test exercises a normal interrupted run; the rows that had been routed before SIGINT are MOVE-routed gate output, not on_error DIVERT).
- Line 63: replace `assert err.rows_routed == 5` with `assert err.rows_routed_success == 5 and err.rows_routed_failure == 0`.
- Line 72: replace `assert err.rows_routed == 0` with `assert err.rows_routed_success == 0 and err.rows_routed_failure == 0`.

- [ ] **Step 7c: Sweep `tests/unit/contracts/test_diverted_outcome.py`**

Run: `grep -n "rows_routed" tests/unit/contracts/test_diverted_outcome.py`

Observed 2026-05-02: line 27 (`rows_routed=0,` in a `RunResult(...)` construction). The test exercises sink-side DIVERTED, not the new ROUTED_ON_ERROR variant — the `rows_routed=0` is incidental. Replace with `rows_routed_success=0, rows_routed_failure=0,`.

- [ ] **Step 7d: Sweep the SECOND test class in `tests/integration/pipeline/test_composer_runtime_agreement.py` (around lines 2620-2830 — `agreement_batch_aggregate` shape)**

Run: `grep -n "rows_routed" tests/integration/pipeline/test_composer_runtime_agreement.py | awk -F: '$1>=2620'`

Observed 2026-05-02: line 2727 (docstring mentions `rows_routed=0`), line 2804 (`run_result.rows_routed` in a `sum_of_buckets` assertion), line 2820 (`rows_routed=run_result.rows_routed,` in a `CompletedData(...)` construction).

This test pins elspeth-31d53c7493 (the row-decomposition relaxation from `==` to `>=`). The test must continue to pin that contract under the rows_routed split. Update each site:

- Line 2727 (docstring): change the prose `rows_routed=0` to `rows_routed_success=0, rows_routed_failure=0` so the documented expected shape matches the post-split fields.
- Line 2804: replace `sum_of_buckets = run_result.rows_succeeded + run_result.rows_failed + run_result.rows_routed + run_result.rows_quarantined` with `sum_of_buckets = run_result.rows_succeeded + run_result.rows_failed + run_result.rows_routed_success + run_result.rows_routed_failure + run_result.rows_quarantined`. The assertion that `rows_processed > sum_of_buckets` continues to pin the legitimate aggregation inequality.
- Line 2820: replace `rows_routed=run_result.rows_routed,` with `rows_routed_success=run_result.rows_routed_success, rows_routed_failure=run_result.rows_routed_failure,` in the `CompletedData(...)` construction.

Run `grep -n "rows_routed" tests/integration/pipeline/test_composer_runtime_agreement.py` after Steps 7d and 8 — expected output: zero matches.

- [ ] **Step 7e: Sweep `tests/unit/cli/test_cli_formatters.py` (RunSummary fixtures and assertions)**

Run: `grep -n "routed\b" tests/unit/cli/test_cli_formatters.py`

Observed 2026-05-02: six `RunSummary(...routed=N)` fixture sites and matching assertions on the rendered string and JSON output. Update each:

- Line 28 (`routed=3,`) and line 51 (`routed=2,`): replace with `routed_success=3, routed_failure=0,` and `routed_success=2, routed_failure=0,` respectively. Both tests exercise gate MOVE behaviour by name (`includes_routed_destinations`, `handles_empty_routed_destinations_list`), so both rows are MOVE.
- Line 95 fixture: same shape — replace `routed=N` with `routed_success=N, routed_failure=0,`.
- Line 160 (`routed=1,`): the JSON test fixture uses `("error_sink", 1)` as the routed destination. Even though the destination name reads as failure-side, the fixture is testing the JSON-shape contract — preserve as `routed_success=1, routed_failure=0,` (single-variant fixture). If the test is intended to specifically exercise the on_error path, change to `routed_success=0, routed_failure=1,` instead — read the test docstring (line 148: `_with_routed_edge_values`) to confirm intent before editing.
- Line 176 (assertion on JSON shape `"routed": 1`): replace with two assertions — `"routed_success": 1` and `"routed_failure": 0` (or whichever variant matches the fixture chosen above).
- Lines 36 and 59 (string-match assertions on the rendered console output, e.g. `"→3 routed (sink_a:2, sink_b:1)"`): these continue to work unchanged because the console display sums the two new fields. Verify by running the test after the fixture updates.

After this step, run `grep -n "routed\b" tests/unit/cli/test_cli_formatters.py | grep -v "routed_success\|routed_failure\|routed_destinations\|routed (sink\|routed_summary"` — expected: zero matches.

- [ ] **Step 7f: Sweep `tests/integration/pipeline/orchestrator/test_execution_loop.py:398`**

Run: `grep -n "summary\.routed\b\|summary\.routed_success" tests/integration/pipeline/orchestrator/test_execution_loop.py`

Replace `assert summary.routed == 0` at line 398 with `assert summary.routed_success == 0 and summary.routed_failure == 0`.

- [ ] **Step 7g: Sweep the additional counter-shape and RowOutcome invariant files enumerated in the file inventory**

The inventory (`### Modified — Tests (counter-shape and RowOutcome invariant sweep targets enumerated 2026-05-02)`) lists fourteen counter-shape test targets with ~145 counter-shape sites, two MCP analyzer semantic assertions, and six RowOutcome enum/invariant/audit-contract targets that the original Task 8 sweep missed. Execute in the listed order (smallest blast radius first; the property test last because its strategies need redesign). Before editing, run this inspection query so invisible enum/category sites are in view:

```bash
rg -n "SINK_OUTCOMES|terminal_outcomes_count|test_record_outcome_requires_fields|test_record_outcome_accepts_required_fields|_OUTCOME_REQUIRED_FIELDS|TestPendingOutcomePostInit|for outcome in RowOutcome" \
  tests/unit/core/landscape/test_model_loaders.py \
  tests/property/contracts/test_row_result_sink_invariant.py \
  tests/unit/engine/test_row_outcome.py \
  tests/property/audit/test_terminal_states.py \
  tests/property/audit/test_recorder_properties.py \
  tests/unit/contracts/test_engine_contracts.py
```

1. `tests/unit/contracts/test_enums.py:13-24` — add `RowOutcome.ROUTED_ON_ERROR` to the `terminal_outcomes` set. **CRITICAL: invisible to the `rows_routed\b` grep.** Without this fix `test_terminal_mappings` fails with a set-difference assertion.
2. `tests/unit/contracts/test_events.py:346` — single `routed=3` kwarg replacement on a `RunSummary(...)` construction. **Caught only by the bare-`routed` complementary grep.** Replace with `routed_success=3, routed_failure=0` (single-variant fixture).
3. `tests/unit/web/sessions/test_models.py:91` — single-site replacement.
4. `tests/unit/web/sessions/test_schema.py:31` — single-site replacement.
5. `tests/unit/web/sessions/test_routes.py:627` — single-site replacement.
6. `tests/unit/web/sessions/test_service.py:642,648` — two sites.
7. `tests/unit/web/execution/test_websocket.py:168,181,225,238,254` — five sites; assertion shape on WebSocket payload.
8. `tests/unit/web/sessions/test_protocol.py:137,186,207,226,245,264,283,303,322,341,360` — eleven sites; RunRecord constructor / readback. Use `rows_routed_success=N, rows_routed_failure=0` as the default for tests that don't differentiate intent.
9. `tests/unit/web/execution/test_routes.py:282,343,422,645,672,699,741,775,788,807,908,920,937` — thirteen sites; API-route handler tests.
10. `tests/unit/web/execution/test_service.py:209,217,406,471,703,761,779,818,1127,1189,1254,1316,2419,2625,2692` — fifteen sites; service-layer fixtures + assertions. The cluster at 2419/2625/2692 is the structural-failure-message tests — verify Task 7 Step 5's new message phrasing matches the assertions.
11. `tests/unit/web/execution/test_schemas.py` — twenty-six sites at the lines listed in the inventory; the largest single-file sweep target. Pydantic schema validation tests across the four response models. Negative-validation tests (asserting `_check_status_row_count_invariant` raises on inconsistent shapes) need updates to use the new field names. Add the public API field-name stability test from the inventory: for each response model, `model_json_schema()["properties"]` contains `rows_routed_success` and `rows_routed_failure`, and does not contain `rows_routed`.
12. `tests/integration/cli/test_cli.py:126,185` — rename test method `test_invalid_rows_routed_to_quarantine_sink` → `test_invalid_rows_routed_failure_to_quarantine_sink` and update the docstring at line 126. **Invisible to the `rows_routed\b` grep** because the substring is inside the test method identifier. Also inspect the test body for `rows_routed` field reads.
13. `tests/unit/mcp/test_analyzer_queries.py` and `tests/unit/mcp/analyzers/test_reports.py` — add MCP outcome-distribution coverage for the new value-domain. Extend `TestGetRunSummary.test_summary_outcome_distribution` (or an adjacent test) with a `RowOutcome.ROUTED_ON_ERROR` token outcome and assert `result["outcome_distribution"]["routed_on_error"] == 1` while existing `"routed"` buckets, if present, remain stored as `"routed"`. Extend the report analyzer test that currently checks terminal/non-terminal outcome rows so one mocked/report row uses `"routed_on_error"` and verifies the serialized `outcome_distribution` preserves that bucket. Add a short assertion comment that MCP does not split historical `"routed"` rows; the runbook/ADR handles the upgrade-boundary limitation.
14. `tests/property/engine/test_orchestrator_lifecycle_properties.py` — at least THIRTY-EIGHT sites across multiple test classes. **Strategy redesign required, NOT mechanical rename**, at lines 67, 84, AND 626 (THREE Hypothesis strategies / `@given` parameter sites — `aggregation_flush_results`, `execution_counters`, and `test_mixed_outcomes_conservation`). Split this into two generator patterns: pure engine-counter monoid properties may draw `rows_routed_success` and `rows_routed_failure` independently, but any property that constructs a validated terminal model (`RunResult(status=COMPLETED...)`, `CompletedData`, `RunStatusResponse`, or any path invoking `_validate_row_decomposition`) MUST use the constrained helper from the inventory so `rows_processed >= sum_terminal` is true by construction. Do not use invalid constructor crashes as a Hypothesis filter. The mechanical rename sites span four test classes (TestFlushResultMonoidProperties, TestExecutionCountersFlushSemantics, single-outcome property tests, conservation properties): lines 113, 141, 167, 184, 201, 217, 251, 257, 275, 287, 313, 320, 365, 372, 382, 399, 426, 517, 539, 543, 588, 630, 639, 644, 645, 655, 674. **Special attention — line 644 conservation invariant:** the assertion `total_increments == completed + failed + routed + quarantined` MUST become `total_increments == completed + failed + routed_success + routed_failure + quarantined` to preserve the property-test invariant; failing to update this in lockstep with the strategy at line 626 would silently weaken the property (the test would still pass on Hypothesis-drawn shapes, but it would no longer be testing total conservation). Run `pytest tests/property/engine/test_orchestrator_lifecycle_properties.py -v --hypothesis-show-statistics` after the redesign to confirm property invariants still hold; if Hypothesis finds a counterexample within the first 200 examples, the strategy redesign has shifted the property semantics and needs review before proceeding.

Required RowOutcome enum/invariant/audit-contract sweeps:

- `tests/unit/core/landscape/test_model_loaders.py` — update `_OUTCOME_REQUIRED_FIELDS` with `"routed_on_error": {"sink_name": "failsink", "error_hash": "e" * 16}`. Add `test_valid_load_routed_on_error`, `test_routed_on_error_without_sink_name_raises`, and `test_routed_on_error_without_error_hash_raises` adjacent to the DIVERTED regression block. These tests must prove `TokenOutcomeLoader` rejects a read-side `routed_on_error` row that has a sink but lacks `error_hash`.
- `tests/property/contracts/test_row_result_sink_invariant.py` — include `ROUTED_ON_ERROR` in sink-targeting categories, but construct it with a real `FailureInfo`. Do not let it fall into `NON_SINK_OUTCOMES`, and do not make the positive sink-name property rely on `error=None`.
- `tests/unit/engine/test_row_outcome.py` — update the all-enum RowResult loops to pass `FailureInfo` for `ROUTED_ON_ERROR`, including the terminal-outcome loop.
- `tests/property/audit/test_terminal_states.py` — change the terminal count expectation from 10 to 11 and include `RowOutcome.ROUTED_ON_ERROR`.
- `tests/property/audit/test_recorder_properties.py` — extend both the missing-required-field table and the accepted-required-fields table for `ROUTED_ON_ERROR` requiring both `sink_name` and `error_hash`.
- `tests/unit/contracts/test_engine_contracts.py` — add `PendingOutcome` tests proving `ROUTED_ON_ERROR` requires non-empty `error_hash`, while `ROUTED` still forbids it.

**Note on `tests/unit/engine/orchestrator/test_outcomes.py`** (already in the original Task 8 Step 2 sweep, NOT a new file in this fourteen-file list): the `_assert_counters` helper at line 273 has a `routed: int = 0` parameter that must be split into `routed_success: int = 0, routed_failure: int = 0` — this is a leverage point because the helper is called by many tests in the file. Update the helper signature, the body assertion at line 283 (`counters.rows_routed == routed`), all caller sites (e.g. line 308's `self._assert_counters(counters, routed=1)`), and the test method name `test_routed_only_increments_routed` at line 300 (rename to `test_routed_only_increments_routed_success` to reflect new semantics). The original Task 8 Step 2 prose is augmented with this guidance in the inventory.

After the counter-shape sweeps, run the verification loop from the inventory to confirm zero remaining counter sites:

```bash
for f in tests/unit/contracts/test_enums.py tests/unit/contracts/test_events.py tests/unit/web/sessions/test_models.py tests/unit/web/sessions/test_schema.py tests/unit/web/sessions/test_routes.py tests/unit/web/sessions/test_service.py tests/unit/web/execution/test_websocket.py tests/unit/web/sessions/test_protocol.py tests/unit/web/execution/test_routes.py tests/unit/web/execution/test_service.py tests/unit/web/execution/test_schemas.py tests/integration/cli/test_cli.py tests/property/engine/test_orchestrator_lifecycle_properties.py; do
  hits=$(grep -cE "rows_routed\b|\.routed\b|routed=" "$f" 2>/dev/null || echo 0)
  [ "$hits" -gt "0" ] && grep -nE "rows_routed\b|\.routed\b|routed=" "$f" | grep -v "routed_success\|routed_failure\|routed_destinations\|routed (sink\|routed_summary\|routed rows\|routed via\|routed to " | sed "s|^|$f:|"
done
```

Expected: zero output. Then run the per-file pytest in dependency order, including the enum/invariant/audit-contract files that are invisible to the counter grep:

```bash
.venv/bin/python -m pytest tests/unit/contracts/test_enums.py tests/unit/contracts/test_events.py tests/unit/contracts/test_engine_contracts.py tests/unit/core/landscape/test_model_loaders.py tests/property/contracts/test_row_result_sink_invariant.py tests/unit/engine/test_row_outcome.py tests/property/audit/test_terminal_states.py tests/property/audit/test_recorder_properties.py tests/unit/web/sessions/ tests/unit/web/execution/ tests/integration/cli/test_cli.py tests/unit/mcp/test_analyzer_queries.py tests/unit/mcp/analyzers/test_reports.py tests/property/engine/test_orchestrator_lifecycle_properties.py -v
```

If this batch passes, the full pytest run (Step 10) should be tractable — the remaining failures are isolated to test files not enumerated above.

- [ ] **Step 8: Invert the locked-in-buggy-behavior test in test_composer_runtime_agreement.py**

In `tests/integration/pipeline/test_composer_runtime_agreement.py`, replace the `test_runstatus_rows_routed_only_classifies_as_failed` test (lines 2492-2543). The on_error reproducer STILL classifies as FAILED — but for the right structural reason now (`rows_routed_failure > 0` is a failure indicator, no success indicator present). Update the docstring and assertions:

```python
    def test_runstatus_on_error_routed_only_classifies_as_failed(self, landscape_db: LandscapeDB, payload_store) -> None:
        """elspeth-5069612f3c — every row triggers a transform exception and
        is routed via on_error to a quarantine sink. After the rows_routed
        split, this shape produces rows_routed_failure == N (DIVERT) with no
        success indicator, and the predicate classifies as FAILED.

        The verdict (FAILED) matches the prior locked-in test, but the
        structural reason changes: previously the predicate excluded
        rows_routed entirely (sidestepping the DIVERT/MOVE conflation); now
        rows_routed_failure is a first-class failure indicator and contributes
        to the predicate decision directly.

        Companion: test_runstatus_gate_routed_only_classifies_as_completed
        below (the gate MOVE shape).
        """
        source = ListSource([{"value": 1, "fail": True}, {"value": 2, "fail": True}])
        transform = ConditionalErrorTransform(on_success="default", on_error="quarantine")
        default_sink = CollectSink(name="default")
        quarantine_sink = CollectSink(name="quarantine")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        run_result = Orchestrator(landscape_db).run(config, graph=build_production_graph(config), payload_store=payload_store)

        self._assert_engine_landscape_agreement(landscape_db, run_result, RunStatus.FAILED)
        assert run_result.rows_processed == 2
        assert run_result.rows_succeeded == 0
        assert run_result.rows_routed_success == 0
        assert run_result.rows_routed_failure == 2
        assert len(default_sink.results) == 0
        assert len(quarantine_sink.results) == 2
```

- [ ] **Step 9: Add the new gate-routed companion test (the user's reproducer)**

Add immediately after the inverted test in `tests/integration/pipeline/test_composer_runtime_agreement.py`:

```python
    def test_runstatus_gate_routed_only_classifies_as_completed(self, landscape_db: LandscapeDB, payload_store) -> None:
        """elspeth-5069612f3c / elspeth-71520f5e30 — user reproducer shape:
        csv source -> gate routes high-priority rows to one sink, low-priority
        rows to another, no on_success success-path sink. Every row is
        intentionally gate-routed via RoutingMode.MOVE (RowOutcome.ROUTED).

        After the rows_routed split, this shape produces rows_routed_success > 0
        with no failure indicator, and the predicate classifies as COMPLETED.

        Before the split (commit cc895589), this shape misclassified as
        RunStatus.FAILED with the misleading error "No row reached the success
        path" because the predicate excluded rows_routed entirely (DIVERT/MOVE
        conflation). This test pins the corrected behavior.
        """
        source = ListSource(
            [
                {"value": 1, "tier": "high"},
                {"value": 2, "tier": "low"},
                {"value": 3, "tier": "high"},
                {"value": 4, "tier": "low"},
            ],
            on_success="source_out",
        )
        # Config gate using the real GateSettings API (pydantic BaseModel at
        # src/elspeth/core/config.py:476). PipelineConfig.gates is
        # list[GateSettings] — plugin instances are NOT accepted. The
        # condition is an ExpressionParser expression on row data; routes
        # maps the (stringified) condition value to a sink name.
        tier_gate = GateSettings(
            name="tier_gate",
            input="source_out",
            condition="row['tier'] == 'high'",
            routes={"true": "high_priority", "false": "low_priority"},
        )
        high_sink = CollectSink(name="high_priority")
        low_sink = CollectSink(name="low_priority")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "high_priority": as_sink(high_sink),
                "low_priority": as_sink(low_sink),
            },
            gates=[tier_gate],
        )

        run_result = Orchestrator(landscape_db).run(
            config,
            graph=build_production_graph(config),
            payload_store=payload_store,
        )

        self._assert_engine_landscape_agreement(landscape_db, run_result, RunStatus.COMPLETED)
        assert run_result.rows_processed == 4
        assert run_result.rows_succeeded == 0  # No success-path sink
        assert run_result.rows_routed_success == 4  # All routed via MOVE
        assert run_result.rows_routed_failure == 0  # No on_error reroutes
        assert len(high_sink.results) == 2
        assert len(low_sink.results) == 2
```

**API verification:** `GateSettings` is a Pydantic `BaseModel` declared at `src/elspeth/core/config.py:476`. Required fields: `name: str`, `input: str` (named connection from upstream `on_success`), `condition: str` (ExpressionParser expression), `routes: dict[str, str]` (route-label → destination). Optional: `fork_to: list[str] | None`. `PipelineConfig.gates: list[GateSettings]` — Pydantic settings, NOT plugin instances. **Reference patterns to copy from:** `tests/integration/pipeline/test_explicit_sink_routing.py:135, 184` constructs `GateSettings` for fork tests; `tests/integration/audit/test_recorder_explain.py:545` constructs it for terminal-routing tests. Both confirm the (name, input, condition, routes) construction shape.

**Pre-write verification (REQUIRED):** Before proposing this test as the acceptance criterion for closing `elspeth-71520f5e30`, the implementing agent MUST verify the test reproduces the FAILED-classification bug against current `main` (commit cc895589 or the current head of the parent branch — pre-PR state). Run:

```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeAgreement::test_runstatus_gate_routed_only_classifies_as_completed -v
```

Expected on pre-PR code: the test FAILS with the asserted `RunStatus.COMPLETED` not matching the engine's actual `RunStatus.FAILED` classification (or with rows_routed_success not being a known field on RunResult). Expected on post-PR code: passes. If the test passes against pre-PR code, the test does NOT reproduce the user's bug and is invalid as the acceptance criterion — investigate before proceeding.

- [ ] **Step 9b: Add a WEB API reproducer test (API half of the user-facing surface)**

The user reported `elspeth-71520f5e30` against the web execution layer (`/api/runs/{run_id}` returning `status='failed'` for gate-only pipelines). Step 9 above exercises only the engine path (`Orchestrator(...).run(config, ...)`). That path does NOT exercise:
- `RunRecord.__post_init__` (the L3 sessions-DB read-side Tier-1 status guards — status enum validity, `finished_at` presence, `landscape_run_id` presence, `error` string presence; NOT the row-count predicate mirror — see Constraints "NOT a mirror site" subsection)
- `_check_status_row_count_invariant` (the Pydantic L3 row-count predicate mirror — one of the three predicate sites listed in Constraints)
- `/api/sessions/{id}/execute` → `/api/runs/{run_id}` HTTP roundtrip — the user's actual reproducer surface
- The `_structural_failure_message` helper at `web/execution/service.py:133` — the misleading "No row reached the success path" error string the user saw

VAL gap: a pure engine-layer test cannot validate that the user's reported bug is fixed end-to-end. A web API reproducer is required, but this is NOT sufficient by itself because the user observed the failure in the dashboard. Add this API test to `tests/integration/web/test_execute_pipeline.py` (or a new sibling file `test_execute_pipeline_gate_routed.py` if the test author prefers isolation), then add the dashboard DOM gate in Step 9e:

```python
@pytest.mark.integration
class TestGateRoutedPipelineExecution:
    """elspeth-5069612f3c / elspeth-71520f5e30 — API half of the VAL gate
    for the gate-only-pipeline misclassification bug.

    Mirrors the engine-layer test in test_composer_runtime_agreement.py but
    exercises the FastAPI surface the user actually reported the bug
    against. Without this test, the L3 row-count predicate mirror
    (_check_status_row_count_invariant) AND the sessions read-side Tier-1
    status guards (RunRecord.__post_init__ — NOT a row-count mirror, but
    enforces status enum / finished_at / landscape_run_id / error invariants)
    are not exercised end-to-end and the user's reported reproducer goes
    unverified.
    """

    @pytest.mark.asyncio
    async def test_gate_routed_pipeline_classifies_as_completed_via_api(
        self,
        work_dir: Path,
    ) -> None:
        """Gate-only pipeline (every row terminally gate-routed; no on_success
        success-path sink) must surface as RunStatus.COMPLETED on
        /api/runs/{run_id}, NOT FAILED with the misleading
        "No row reached the success path" structural-failure message.

        Pre-PR (commit cc895589): /api/runs/{run_id} returns status='failed'
        with the synthetic structural error. Post-PR: returns
        status='completed' with rows_routed_success > 0.
        """
        from elspeth.web.app import create_app
        from elspeth.web.composer.state import (
            CompositionState,
            EdgeSpec,
            NodeSpec,
            OutputSpec,
            PipelineMetadata,
            SourceSpec,
        )
        from elspeth.web.config import WebSettings
        from elspeth.web.sessions.protocol import CompositionStateData

        # ... settings setup mirrors the existing
        # test_csv_passthrough_csv pattern at lines 72-83 ...

        # Mirror the existing test's auth + session setup (lines 84-100):
        # 1. Register + login test user → auth_headers
        # 2. Create session via REST → session_id

        # 3. Construct a CompositionState that compiles to a gate-only DAG.
        # The gate must use the same ExpressionParser shape as the engine-layer
        # test (Task 8 Step 9): condition="row['tier'] == 'high'", routes
        # mapping "true"→high_priority, "false"→low_priority. Both routes
        # must resolve to terminally-named output sinks (NOT to an on_success
        # success path).
        state = CompositionState(
            source=SourceSpec(
                name="csv_in",
                plugin="csv",
                options={"path": str(work_dir / "blobs" / "input.csv"), ...},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    kind="gate",
                    name="tier_gate",
                    input="csv_in_out",
                    condition="row['tier'] == 'high'",
                    routes={"true": "high_priority", "false": "low_priority"},
                ),
            ),
            edges=(),  # routes wire the gate → outputs
            outputs=(
                OutputSpec(name="high_priority", plugin="csv",
                           options={"path": str(work_dir / "outputs" / "high.csv"), ...},
                           on_write_failure="discard"),
                OutputSpec(name="low_priority", plugin="csv",
                           options={"path": str(work_dir / "outputs" / "low.csv"), ...},
                           on_write_failure="discard"),
            ),
            metadata=PipelineMetadata(name="Gate-routed reproducer",
                                       description="elspeth-71520f5e30 reproducer"),
            version=1,
        )
        # ... save_composition_state via session_service (mirrors lines 148-160)

        # 4. Execute via REST
        resp = await client.post(
            f"/api/sessions/{session_id}/execute",
            headers=auth_headers,
        )
        assert resp.status_code == 202, f"Execute failed: {resp.text}"
        run_id = resp.json()["run_id"]

        # 5. Poll to terminal
        deadline = time.monotonic() + 30
        status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            resp = await client.get(f"/api/runs/{run_id}", headers=auth_headers)
            assert resp.status_code == 200
            status = resp.json()
            if status["status"] in ("completed", "completed_with_failures", "failed", "cancelled", "empty"):
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Pipeline did not complete within 30 seconds")

        # 6. THE BUG: pre-PR returns 'failed'; post-PR must return 'completed'
        assert status["status"] == "completed", (
            f"Gate-routed pipeline misclassified — expected 'completed', "
            f"got {status['status']!r}; error={status.get('error')!r}. "
            f"Pre-fix structural-failure message would be 'No row reached "
            f"the success path' or similar — the rows_routed counter split "
            f"is supposed to fix this end-to-end."
        )
        assert status["rows_processed"] > 0
        assert status["rows_succeeded"] == 0  # No on_success success-path sink
        assert status["rows_routed_success"] > 0  # All rows gate-routed via MOVE
        assert status["rows_routed_failure"] == 0  # No on_error reroutes
        assert status["error"] is None  # No synthetic structural-failure message

        # 7. Verify the L3 mirror invariants accept this shape — by getting here
        # without the API returning 5xx, both _check_status_row_count_invariant
        # and RunRecord.__post_init__'s status validity checks accepted the
        # (status='completed', rows_succeeded=0, rows_routed_success>0) shape.
```

**Add to Tests inventory:**
- `tests/integration/web/test_execute_pipeline.py` (or sibling file) — new web-layer reproducer test class `TestGateRoutedPipelineExecution`. The new test follows the existing pattern at lines 47-209 of `test_execute_pipeline.py`: register/login/session-create/save-state/execute/poll/verify lifecycle. The CompositionState `nodes=` tuple uses `NodeSpec(kind="gate", ...)` to inject a gate node — verify the exact NodeSpec shape with `grep -n "class NodeSpec\|kind=" src/elspeth/web/composer/state.py` before writing; the example above is illustrative.

**Pre-write verification (REQUIRED, mirrors Step 9):** Run this test against pre-PR code (commit cc895589 or current main head):

```bash
.venv/bin/python -m pytest tests/integration/web/test_execute_pipeline.py::TestGateRoutedPipelineExecution::test_gate_routed_pipeline_classifies_as_completed_via_api -v
```

Expected on pre-PR code: the test FAILS with `status['status'] == 'failed'` and `status['error']` containing the synthetic structural-failure message. Expected on post-PR code: passes. If the pre-PR run does NOT reproduce the bug, the test does not validate the fix end-to-end — investigate the pipeline shape before treating it as the VAL gate.

**Why this test is necessary but not sufficient for `elspeth-71520f5e30`** — because the user reported the bug at the web execution surface, the engine-layer test in Step 9 is not enough. This web API test confirms the predicate, the mirror invariants, the structural-failure message, and the API contract all line up. The dashboard UI gate in Step 9e confirms the visible surface renders the corrected terminal state instead of the old failed-run alert.

- [ ] **Step 9c: Add an AUDIT-RECORDER integration test pinning the MOVE-vs-DIVERT distinguishability empirically**

Task 4 Step 2's resolution claims three pieces of evidence for Tier-1 audit distinguishability between gate-routed (MOVE) and on_error-routed (DIVERT) tokens: (1) `RowOutcome.ROUTED_ON_ERROR` persisted to `token_outcomes.outcome`, (2) `RoutingEvent.mode` on the edge as cross-check, (3) `error_hash` on the outcome record. CLAUDE.md Auditability Standard says **"no inference — if it's not recorded, it didn't happen."** The resolution must be PINNED EMPIRICALLY by tests that fail on pre-PR code and pass on post-PR code, otherwise a future regression that silently drops one of the three signals goes undetected.

**Extend the existing test file** at `tests/integration/audit/test_recorder_routing_events.py`. That file already contains `TestRecorderFactoryRouting`; append the new production-path class below the existing recorder-operation tests. Add the imports listed below at file top if they are not already present. These tests deliberately avoid a global `RoutingMode.DIVERT` biconditional because source quarantine and sink failsink edges also use DIVERT; they pin the scoped producer invariant only:

- gate `route_to_sink` MOVE -> `RowOutcome.ROUTED`, `error_hash IS NULL`, route event mode `move`;
- transform `on_error` DIVERT -> `RowOutcome.ROUTED_ON_ERROR`, non-empty `error_hash`, route event mode `divert`.

```python
import re
from collections.abc import Mapping
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from elspeth.contracts import NodeType, RowOutcome, RoutingMode, RoutingSpec
from elspeth.contracts.audit import TokenRef
from elspeth.core.config import GateSettings, SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.lineage import explain
from elspeth.core.landscape.schema import (
    edges_table,
    node_states_table,
    nodes_table,
    routing_events_table,
    token_outcomes_table,
)
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.plugins import CollectSink, ConditionalErrorTransform, ListSource


def _run_mixed_move_and_divert_pipeline(
    landscape_db: LandscapeDB,
    payload_store,
):
    """Run source -> transform -> gate with one MOVE-routed row and one
    on_error-DIVERT-routed row.

    Uses ExecutionGraph.from_plugin_instances(), not a hand-built graph, so the
    routing_events rows are generated by the production DAG wiring path.
    """
    source = ListSource(
        [
            {"id": 1, "priority": "high", "fail": False},
            {"id": 2, "priority": "low", "fail": True},
        ],
        name="audit_source",
        on_success="transform_in",
    )
    transform = ConditionalErrorTransform(
        name="may_fail",
        input_connection="transform_in",
        on_success="route_in",
        on_error="error_sink",
    )
    gate = GateSettings(
        name="priority_gate",
        input="route_in",
        condition="row['priority'] == 'high'",
        routes={"true": "high_priority", "false": "default"},
    )
    sinks = {
        "high_priority": CollectSink("high_priority"),
        "default": CollectSink("default"),
        "error_sink": CollectSink("error_sink"),
    }
    source_settings = SourceSettings(
        plugin="list_source",
        on_success="transform_in",
        options={},
    )
    wired = wire_transforms(
        [transform],
        source_connection="transform_in",
        final_sink="route_in",
        names=["may_fail"],
    )
    graph = ExecutionGraph.from_plugin_instances(
        source=as_source(source),
        source_settings=source_settings,
        transforms=wired,
        sinks={name: as_sink(sink) for name, sink in sinks.items()},
        aggregations={},
        gates=[gate],
    )
    config = PipelineConfig(
        source=as_source(source),
        transforms=[as_transform(transform)],
        sinks={name: as_sink(sink) for name, sink in sinks.items()},
        gates=[gate],
    )

    result = Orchestrator(landscape_db).run(
        config,
        graph=graph,
        payload_store=payload_store,
    )
    return result


def _fetch_single_outcome_for_sink(
    landscape_db: LandscapeDB,
    *,
    run_id: str,
    sink_name: str,
) -> Mapping[str, Any]:
    stmt = (
        select(
            token_outcomes_table.c.token_id,
            token_outcomes_table.c.outcome,
            token_outcomes_table.c.error_hash,
            token_outcomes_table.c.sink_name,
        )
        .where(token_outcomes_table.c.run_id == run_id)
        .where(token_outcomes_table.c.sink_name == sink_name)
        .order_by(token_outcomes_table.c.recorded_at, token_outcomes_table.c.outcome_id)
    )
    with landscape_db.connection() as conn:
        rows = conn.execute(stmt).mappings().all()
    assert len(rows) == 1, f"expected one token_outcomes row for sink {sink_name!r}, got {rows}"
    return rows[0]


def _fetch_routing_modes_to_sink(
    landscape_db: LandscapeDB,
    *,
    run_id: str,
    sink_plugin_name: str,
) -> list[str]:
    stmt = (
        select(routing_events_table.c.mode)
        .select_from(
            routing_events_table.join(
                node_states_table,
                routing_events_table.c.state_id == node_states_table.c.state_id,
            )
            .join(edges_table, routing_events_table.c.edge_id == edges_table.c.edge_id)
            .join(
                nodes_table,
                (edges_table.c.to_node_id == nodes_table.c.node_id)
                & (node_states_table.c.run_id == nodes_table.c.run_id),
            )
        )
        .where(node_states_table.c.run_id == run_id)
        .where(nodes_table.c.plugin_name == sink_plugin_name)
        .order_by(routing_events_table.c.ordinal, routing_events_table.c.event_id)
    )
    with landscape_db.connection() as conn:
        return [str(row.mode) for row in conn.execute(stmt)]


class TestRoutingEventDistinguishability:
    """elspeth-5069612f3c — pin Tier-1 audit distinguishability between
    gate-routed MOVE and transform-on-error-routed DIVERT tokens.
    """

    def test_gate_routed_token_records_routed_outcome_with_null_error_hash(
        self, landscape_db: LandscapeDB, payload_store
    ) -> None:
        """A row routed via gate route_to_sink (intentional MOVE) must record:
        - token_outcomes.outcome == 'routed'
        - token_outcomes.error_hash IS NULL
        - routing_events.mode == 'move' on the edge that carried it
        """
        run_result = _run_mixed_move_and_divert_pipeline(landscape_db, payload_store)

        outcome = _fetch_single_outcome_for_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_name="high_priority",
        )
        assert outcome["outcome"] == RowOutcome.ROUTED.value
        assert outcome["error_hash"] is None

        modes = _fetch_routing_modes_to_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_plugin_name="high_priority",
        )
        assert modes == [RoutingMode.MOVE.value]

    def test_on_error_routed_token_records_routed_on_error_with_non_empty_error_hash(
        self, landscape_db: LandscapeDB, payload_store
    ) -> None:
        """A row routed via transform on_error (DIVERT) must record:
        - token_outcomes.outcome == 'routed_on_error'  (NOT 'routed')
        - token_outcomes.error_hash IS NOT NULL and matches the canonical
          16-char sha256 prefix recipe from processor.py:2262
        - routing_events.mode == 'divert' on the edge that carried it
        """
        run_result = _run_mixed_move_and_divert_pipeline(landscape_db, payload_store)

        outcome = _fetch_single_outcome_for_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_name="error_sink",
        )
        assert outcome["outcome"] == RowOutcome.ROUTED_ON_ERROR.value
        assert outcome["error_hash"] is not None
        assert re.fullmatch(r"[0-9a-f]{16}", str(outcome["error_hash"]))

        modes = _fetch_routing_modes_to_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_plugin_name="error_sink",
        )
        assert modes == [RoutingMode.DIVERT.value]

    def test_explain_recovers_routing_intent_for_both_variants(
        self, landscape_db: LandscapeDB, payload_store
    ) -> None:
        """The explain() function (the contractual audit-attributability surface)
        must distinguish the two variants single-hop. Run a mixed pipeline (some
        rows gate-routed, some on_error-routed), call explain(recorder, run_id,
        token_id) for one of each, and assert the returned record contains:
        - For gate-routed token: outcome=ROUTED, sink_name set, error context absent
        - For on_error token: outcome=ROUTED_ON_ERROR, sink_name set, error context
          present (error_hash retrievable, non-empty)
        """
        run_result = _run_mixed_move_and_divert_pipeline(landscape_db, payload_store)
        factory = RecorderFactory(landscape_db, payload_store=payload_store)

        gate_outcome = _fetch_single_outcome_for_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_name="high_priority",
        )
        error_outcome = _fetch_single_outcome_for_sink(
            landscape_db,
            run_id=run_result.run_id,
            sink_name="error_sink",
        )

        gate_lineage = explain(
            factory.query,
            factory.data_flow,
            run_result.run_id,
            token_id=str(gate_outcome["token_id"]),
        )
        error_lineage = explain(
            factory.query,
            factory.data_flow,
            run_result.run_id,
            token_id=str(error_outcome["token_id"]),
        )

        assert gate_lineage is not None
        assert gate_lineage.outcome is not None
        assert gate_lineage.outcome.outcome == RowOutcome.ROUTED
        assert gate_lineage.outcome.sink_name == "high_priority"
        assert gate_lineage.outcome.error_hash is None
        assert any(event.mode == RoutingMode.MOVE for event in gate_lineage.routing_events)

        assert error_lineage is not None
        assert error_lineage.outcome is not None
        assert error_lineage.outcome.outcome == RowOutcome.ROUTED_ON_ERROR
        assert error_lineage.outcome.sink_name == "error_sink"
        assert error_lineage.outcome.error_hash is not None
        assert re.fullmatch(r"[0-9a-f]{16}", error_lineage.outcome.error_hash)
        assert any(event.mode == RoutingMode.DIVERT for event in error_lineage.routing_events)

    def test_token_outcome_unique_constraint_admits_routed_on_error(
        self, landscape_db: LandscapeDB
    ) -> None:
        """The token_outcomes partial unique index (one terminal outcome per
        token, see docs/contracts/token-outcomes/00-token-outcome-contract.md)
        must admit ROUTED_ON_ERROR like any other terminal outcome. Constructing
        a token with ROUTED_ON_ERROR after another terminal MUST raise the
        unique-violation error (not silently overwrite, not silently drop).
        """
        factory = RecorderFactory(landscape_db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"id": 1, "fail": True},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        ref = TokenRef(token_id=token.token_id, run_id=run.run_id)

        factory.data_flow.record_token_outcome(
            ref,
            RowOutcome.ROUTED_ON_ERROR,
            sink_name="error_sink",
            error_hash="0123456789abcdef",
        )

        with pytest.raises(IntegrityError):
            factory.data_flow.record_token_outcome(
                ref,
                RowOutcome.COMPLETED,
                sink_name="default",
            )
```

**Pre-write verification (REQUIRED):** before treating these tests as the audit-distinguishability acceptance criteria, run them against pre-PR code. Expected failure modes:

| Test | Pre-PR | Post-PR |
|---|---|---|
| `test_gate_routed_token_records_routed_outcome_with_null_error_hash` | passes (existing path correctly records `routed`) | passes |
| `test_on_error_routed_token_records_routed_on_error_with_non_empty_error_hash` | **FAILS** — outcome is `routed` (not `routed_on_error`) and `error_hash IS NULL` — confirms the test catches the bug | passes |
| `test_explain_recovers_routing_intent_for_both_variants` | **FAILS** — explain() can't distinguish the two variants pre-split | passes |
| `test_token_outcome_unique_constraint_admits_routed_on_error` | **FAILS** — `RowOutcome.ROUTED_ON_ERROR` is not a known enum value pre-PR; `AttributeError` at construction | passes |

If any of the three FAIL-expected tests pass on pre-PR code, the test does not pin what it claims and must be revised before being treated as the audit gate.

**Add to Tests inventory:** `tests/integration/audit/test_recorder_routing_events.py` (existing file, extended) — pins the Tier-1 audit-trail distinguishability between MOVE-routed and DIVERT-routed tokens via direct `landscape_db` queries on `token_outcomes` and `routing_events`, plus an `explain()` round-trip. Verifies the scoped producer signals (outcome value, error_hash presence, RoutingMode on the corresponding producer edge) Task 4 Step 2's resolution depends on.

- [ ] **Step 9d: Add a RESUME-PATH integration test for gate-routed pipelines (exercises the resume code path, not just the RunResult constructor)**

The Task 1 TDD seed includes `test_resume_continuation_still_classifies_as_completed`, which calls `derive_terminal_run_status(...)` directly with synthetic counters. That test pins the L0 predicate's behavior on a "resume-shaped" counter tuple (`rows_processed=0, rows_succeeded=3, ...`) but DOES NOT exercise the actual resume code path at `core.py:3104` (`Orchestrator.resume(...)`). After the rows_routed split lands, `core.py:3247` (the resume's terminal-status derivation) reads `result.rows_routed_success` and `result.rows_routed_failure` from the resume-side accumulator (Task 5 Step 2e). A regression in the resume-side accumulator — an off-by-one, a missed kwarg, a broken local — would slip past the L0 unit test entirely.

The web-layer reproducer (Step 9b) doesn't cover resume either: it exercises `Orchestrator.run(...)`, not `Orchestrator.resume(...)`. So the resume code path is currently untested for gate-routed pipelines end-to-end.

**Add a new test method** to `tests/integration/pipeline/test_resume_comprehensive.py` (the file already has the `TestResumeComprehensive` class with the `resume_test_env` fixture and `_setup_failed_run` helper at lines 188-242). The new test mirrors the existing `test_resume_normal_path_with_remaining_rows` shape (lines 244-349) but uses a gate-routed pipeline:

```python
def test_resume_gate_routed_pipeline_classifies_as_completed(
    self,
    resume_test_env: dict[str, Any],
) -> None:
    """elspeth-5069612f3c — pin the resume code path's correct accumulation
    of rows_routed_success.

    The L0 unit test test_resume_continuation_still_classifies_as_completed
    pins the predicate's behavior on a synthetic resume-shaped counter
    tuple, but does NOT exercise core.py:3247 — the actual resume site
    where the terminal status is derived from the resume-side accumulator.
    A regression in the resume-side local accumulators would slip past
    that unit test.

    Scenario:
    1. Failed run with 5 rows (0-4), gate-routed pipeline
       (source -> tier_gate -> high_priority|low_priority sinks)
    2. Rows 0-2 already gate-routed (token_outcomes records ROUTED + sink_name)
    3. Resume processes rows 3-4 (also gate-routed)
    4. Verify: result.status == RunStatus.COMPLETED (not FAILED)
    5. Verify: result.rows_succeeded == 0 (no on_success success-path sink)
    6. Verify: result.rows_routed_success >= 2 (rows 3-4 from resume — the
       resume-side accumulator picks these up via core.py:3247)
    7. Verify: result.rows_routed_failure == 0 (no on_error reroutes)
    """
    db = resume_test_env["db"]
    checkpoint_mgr = resume_test_env["checkpoint_manager"]
    recovery_mgr = resume_test_env["recovery_manager"]
    payload_store = resume_test_env["payload_store"]
    checkpoint_config = resume_test_env["checkpoint_config"]
    tmp_path = resume_test_env["tmp_path"]

    # Set up a 5-row failed run with checkpoint at row 2.
    # The _setup_failed_run helper uses a passthrough transform shape;
    # we override after setup to inject a tier_gate via GateSettings.
    run_id = "resume-gate-routed-test"
    run_id, base_graph = self._setup_failed_run(
        db, payload_store, run_id, num_rows=5, checkpoint_at=2
    )

    # Mark rows 0-2 as gate-routed via MOVE (token_outcomes.outcome='routed',
    # sink_name set, error_hash NULL — the canonical pre-split-fix shape
    # for gate-routed rows).
    factory = make_factory(db)
    for i in range(3):
        # Route 0-1 to high_priority, 2 to low_priority (mimics tier
        # condition behaviour without re-running the gate at setup time).
        sink = "high_priority" if i < 2 else "low_priority"
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=f"t{i}", run_id=run_id),
            outcome=RowOutcome.ROUTED,
            sink_name=sink,
        )

    # Build the resume PipelineConfig with the real GateSettings
    # (mirrors Task 8 Step 9 shape).
    high_sink_path = tmp_path / "resume_high.csv"
    low_sink_path = tmp_path / "resume_low.csv"
    tier_gate = GateSettings(
        name="tier_gate",
        input="source_out",
        condition="row['tier'] == 'high'",
        routes={"true": "high_priority", "false": "low_priority"},
    )
    resume_schema = {"mode": "fixed", "fields": ["id: int", "tier: str"]}
    resume_config = PipelineConfig(
        source=_null_source("source", schema=resume_schema),
        transforms=[],
        sinks={
            "high_priority": CSVSink({
                "path": str(high_sink_path),
                "schema": resume_schema,
                "mode": "append",
            }),
            "low_priority": CSVSink({
                "path": str(low_sink_path),
                "schema": resume_schema,
                "mode": "append",
            }),
        },
        gates=[tier_gate],
    )

    # Build the matching ExecutionGraph (source -> gate -> sinks via
    # routes; the helper at build_production_graph(config) produces the
    # exact graph the engine will compile from this PipelineConfig).
    resume_graph = build_production_graph(resume_config)

    # Confirm the recovery manager agrees this run is resumable.
    assert recovery_mgr.can_resume(run_id, resume_graph).can_resume
    resume_point = recovery_mgr.get_resume_point(run_id, resume_graph)
    assert resume_point is not None

    orchestrator = Orchestrator(
        db,
        checkpoint_manager=checkpoint_mgr,
        checkpoint_config=checkpoint_config,
    )

    # Execute the resume — this is the code path under test. The
    # terminal-status derivation at core.py:3247 must accumulate
    # rows_routed_success from the resume-side locals (Task 5 Step 2e).
    result = orchestrator.resume(
        resume_point=resume_point,
        config=resume_config,
        graph=resume_graph,
        payload_store=payload_store,
    )

    # CORE ASSERTIONS — verify the resume-side accumulator + predicate.
    assert result.status == RunStatus.COMPLETED, (
        f"Resume of gate-routed pipeline misclassified — expected "
        f"COMPLETED, got {result.status!r}. The resume-side "
        f"derive_terminal_run_status call at core.py:3247 must "
        f"accumulate rows_routed_success from the resume locals. "
        f"result={result.to_dict()}"
    )
    assert result.rows_succeeded == 0  # No on_success success-path sink.
    assert result.rows_routed_success >= 2  # Rows 3-4 from resume.
    assert result.rows_routed_failure == 0  # No on_error reroutes.

    # Optional: query Landscape directly to verify the resume's outcome
    # records hit token_outcomes with the expected shape (cross-check
    # with the audit-recorder test in Step 9c).
    with db.engine.connect() as conn:
        from sqlalchemy import select as sa_select
        from elspeth.core.landscape.schema import token_outcomes_table
        outcomes = conn.execute(
            sa_select(token_outcomes_table.c.outcome, token_outcomes_table.c.sink_name)
            .where(token_outcomes_table.c.run_id == run_id)
        ).fetchall()
    routed_outcomes = [o for o in outcomes if o.outcome == "routed"]
    # Pre-resume rows (0-2) plus resume-processed rows (3-4) all routed.
    assert len(routed_outcomes) == 5
```

**Pre-write verification (REQUIRED):** before treating this test as the resume-path acceptance criterion, run it against pre-PR code. Expected on pre-PR: the test FAILS at `assert result.status == RunStatus.COMPLETED` with `result.status == RunStatus.FAILED`, because the pre-PR resume's `derive_terminal_run_status` call at `core.py:3247` excludes `rows_routed` from the predicate (DIVERT/MOVE conflation). On post-PR: passes. If the pre-PR run does NOT reproduce the FAILED classification on the resume code path, the test does not validate the fix end-to-end on the resume side — investigate the resume's locals accumulation shape before treating it as the gate.

**Why this test matters separately from the engine-layer and web-layer reproducers:** the engine-layer test (Step 9) exercises `Orchestrator.run(...)`; the web-layer test (Step 9b) exercises `Orchestrator.run(...)` via the FastAPI surface. Neither exercises `Orchestrator.resume(...)` — a structurally distinct code path with its own terminal-status derivation site (`core.py:3247`) and its own local-accumulator pattern. Without this test, a regression in the resume-side accumulator (an off-by-one in the new locals, a missed kwarg in the `derive_terminal_run_status` call, a broken return-tuple expansion) slips past every other gate.

- [ ] **Step 9e: Add a DASHBOARD UI gate for the user-reported Runs surface**

Step 9b validates the FastAPI response, but the user saw the bug in the dashboard. A corrected API payload can still fail the user if the frontend store drops the split fields, if the run list renders the old failed status, or if the stale structural-failure message still appears in the Runs panel. The repo does not currently depend on Playwright; frontend tests use Vitest + React Testing Library. Use that existing DOM test stack for this status-surface regression instead of adding a new browser dependency.

**Update frontend types/store first:**

1. In `src/elspeth/web/frontend/src/types/index.ts`, add `rows_routed_success: number` and `rows_routed_failure: number` to the run/progress event payload interfaces that carry row counters (`Run`, `RunEventProgress`, `RunEventCompleted`, `RunEventCancelled`, `RunProgress`). If a failed terminal event does not carry row counters today, do not fabricate them on the event type; keep the fields only where the API/WebSocket contract actually sends them.
2. In `src/elspeth/web/frontend/src/stores/executionStore.ts`, initialise `progress` with both routed fields set to zero when execution starts, preserve routed fields from progress/completed/cancelled payloads when present, and write them into the run list on terminal updates. Use explicit property reads; do not use dynamic key access.
3. Update frontend test fixtures (`makeRun`, progress fixtures, event fixtures) to include the new fields so TypeScript catches any future drift.

**Add this test** to `src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx`:

```tsx
it("renders gate-routed completed runs without failure alert", () => {
  useExecutionStore.setState({
    runs: [
      makeRun({
        status: "completed",
        rows_processed: 3,
        rows_failed: 0,
        rows_routed_success: 3,
        rows_routed_failure: 0,
        error: null,
      }),
    ],
  });

  render(<RunsView />);

  expect(screen.getByText("completed")).toBeInTheDocument();
  expect(screen.getByText("3 rows")).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(screen.queryByText(/No row reached/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/Pipeline execution failed/i)).not.toBeInTheDocument();
});
```

**Why this is the UI VAL gate:** this test asserts the visible Runs dashboard accepts the fixed API response shape (`status="completed"`, `rows_routed_success > 0`, `rows_routed_failure == 0`) and does not display the failed-run alert or the historical "No row reached..." structural message. It is intentionally separate from the API test: Step 9b proves the backend emits the fixed shape; Step 9e proves the dashboard surface renders it correctly.

**Verification:**

```bash
cd src/elspeth/web/frontend
npm run test -- src/components/inspector/RunsView.test.tsx
npm run build
```

Expected: the focused frontend test passes and the production frontend build type-checks. If `npm run build` fails because a generated API type or fixture still lacks the split routed fields, update the explicit type/fixture site rather than casting around the error.

- [ ] **Step 10: Run the full test suite — should be green**

Run: `.venv/bin/python -m pytest tests/ -x`

Expected: all pass. If any failure, pytest's exit status must fail the gate. If the output is too long while diagnosing, rerun the focused failing test directly, or capture/truncate output only with `set -o pipefail` active so pytest failures cannot be masked.

- [ ] **Step 11: Run mypy**

Run: `.venv/bin/python -m mypy src/`

Expected: clean. Fix any reported errors.

- [ ] **Step 12: Run ruff**

Run: `.venv/bin/python -m ruff check src/`

Expected: clean.

- [ ] **Step 13: Run the tier-model enforcer**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`

Expected: pass. (No new layer violations; this PR is intra-layer.)

- [ ] **Step 14: Run the freeze-guard enforcer**

Run: `.venv/bin/python scripts/cicd/enforce_freeze_guards.py`

Expected: pass.

- [ ] **Step 15: Verify no retired aggregate/event field names remain**

Run THREE commands:

```bash
grep -rn "rows_routed\b" src/elspeth/ tests/ --include="*.py"
grep -rn "RunSummary[^)]*routed[^_]\|\.routed\b\|\brouted=" src/elspeth/ tests/ --include="*.py" | grep -v "routed_success\|routed_failure\|routed_destinations\|RowOutcome\.ROUTED\|RoutingMode\|RoutingEvent\|routed (sink\|routed_summary\|routed rows\|routed via\|routed to "
grep -rn "rows_routed\b\|rowsRouted\b" src/elspeth/web/frontend/src/ docs/architecture/ docs/contracts/ docs/guides/ docs/runbooks/ docs/reference/
```

Expected: all three return no matches. If matches remain, they are missed retired field sites — fix and re-run.

Coverage breakdown:
- The first command catches missed `rows_routed` (without suffix) field and identifier references — the original Done Condition.
- The second command catches missed `RunSummary.routed` consumer / producer call sites with the existing exclusion list filtering legitimate prose, `routed_destinations`, and the still-valid enum/mode references.
- The third command's docs target list is **deliberately scoped to stable/published contract subdirs only** — gating all of `docs/` would self-fail because the active plan file itself contains 150+ legitimate prose references to `rows_routed`. Subdirs intentionally excluded from the gate (because they discuss the old name in prose by design): `docs/superpowers/plans/`, `docs/superpowers/specs/`, `docs/audits/`, `docs/bugs/`, `docs/analysis/`, `docs/arch-analysis-*/`, `docs/arch-pack-*/`, `docs/release/`, `docs/requirements/`, `docs/assets/`. The frontend lives at `src/elspeth/web/frontend/src/`; the legacy paths `frontend/`, `web/`, and `src/elspeth/web/static/` do not exist in this repo (verified 2026-05-02) and would silently return clean if greppped.

- [ ] **Step 16: Create ADR-018 and update ADR-004**

Create a new ADR for the project-wide producer-site discrimination default. This is not just a local sink-routing footnote: it affects future `RowOutcome` additions, public counter naming, and the decision to encode producer-known semantics as enum variants rather than optional discriminator fields.

Create `docs/architecture/adr/018-producer-site-outcome-discrimination.md`:

```markdown
# ADR-018: Producer-Site Outcome Discrimination

**Date:** 2026-05-02
**Status:** Accepted
**Deciders:** ELSPETH maintainers
**Tags:** contracts, audit, row-outcomes, public-api

## Context

The `rows_routed` counter split in `elspeth-5069612f3c` exposed a broader
design rule. Before the split, `RowOutcome.ROUTED` represented both
intentional gate `route_to_sink` MOVE rows and transform `on_error` DIVERT
rows. The terminal-status predicate could not distinguish those producer
circumstances, so gate-only pipelines were misclassified as failed.

The producer already knows which circumstance occurred. Deferring that
knowledge to a later accumulator, graph lookup, or optional discriminator
field makes Tier-1 status accounting depend on convention rather than a
mechanical type signal.

## Decision

When a producer emits a terminal row outcome and the producer-known
circumstance changes audit obligations, status-predicate contribution, or
operator meaning, encode that circumstance as a distinct `RowOutcome` enum
variant at the producer site.

Do not encode producer-known terminal circumstances as optional discriminator
fields on a shared outcome variant unless a separate ADR justifies the
exception. Optional discriminator fields recreate the invalid state "outcome
known, producer circumstance unknown" and force every consumer to remember a
secondary field.

### Naming Rule

`RowOutcome` variants name the producer/audit circumstance. Aggregate row
counters that feed run-status predicates name the predicate role.

For this PR, that means:

- `RowOutcome.ROUTED` means intentional gate MOVE and contributes to
  `rows_routed_success`.
- `RowOutcome.ROUTED_ON_ERROR` means transform `on_error` reroute and
  contributes to `rows_routed_failure`.

The enum and counter names are intentionally not lexically isomorphic. The
enum answers "what producer circumstance happened to this token?" The counter
answers "how does this aggregate bucket contribute to the run-status
predicate?" Future ADRs must not cite this decision as "make every
outcome/counter pair have the same word"; the pattern is producer-site
outcome discrimination plus predicate-role aggregate naming.

### Public API Naming

The web API exposes `rows_routed_success` and `rows_routed_failure` directly
on the relevant Pydantic response models. These field names are stable for the
current public API horizon. Do not add `rows_moved`, `rows_error_routed`, or a
transitional `rows_routed` alias in this PR. A future rename would be a
breaking API decision requiring its own ADR/API migration plan and OpenAPI
schema test updates.

## Consequences

### Positive Consequences

- Consumer code gets a mechanical prompt to handle new producer
  circumstances through enum branches.
- Audit records preserve producer intent directly in
  `token_outcomes.outcome`.
- L0 contracts, L3 Pydantic response models, and frontend types can compare
  the same predicate-role counter names without translation drift.

### Negative Consequences

- Adding a new producer circumstance requires updating every relevant
  `RowOutcome` branch, even when the transport path is otherwise shared.
- Aggregate counter names may not be lexical siblings of enum variant names.
  The naming rule must be read before adding future counters.
- Public API field names inherit engine predicate vocabulary by design.

## Alternatives Considered

### Alternative 1: Shared outcome variant plus discriminator field

Use `RowOutcome.ROUTED` for both MOVE and DIVERT and add a secondary
`routing_intent: Literal["move", "divert"] | None` field.

Rejected because it creates an optional field every consumer must remember to
read, does not force existing `RowOutcome.ROUTED` branches to change, and
allows the invalid state "routed but unknown intent" to be represented.

### Alternative 2: Accumulator graph lookup

Keep the producer emission unchanged and have the accumulator infer MOVE vs
DIVERT from the graph edge or `RoutingMode`.

Rejected because the producer already knows the answer, while a graph lookup
is a defensive inference path at the Tier-1 counter boundary.

### Alternative 3: Rename public API fields away from engine vocabulary

Expose names such as `rows_moved` and `rows_error_routed` in the web API while
using different L0/L2 names internally.

Rejected for this PR because the bug came from predicate mirror drift. Keeping
L0, L3, and frontend predicate-role fields identical is the mechanical guard.

## Related Decisions

- ADR-004: Explicit Sink Routing
- `docs/superpowers/plans/2026-05-02-rows-routed-counter-split.md`
- Filigree issue `elspeth-5069612f3c`
```

Update `docs/architecture/adr/README.md` to add ADR-018 to the index.

Then append a short "Counter split (elspeth-5069612f3c, 2026-05-02)" amendment at the end of ADR-004 (or in a "Subsequent amendments" section if one exists). ADR-004 should capture the explicit-sink-routing local effect and cross-reference ADR-018 for the project-wide producer-site discrimination default:

```markdown
### Counter split (elspeth-5069612f3c, 2026-05-02)

Subsequent to ADR-010 Phase 2.2 (commit cc895589), the engine's
`ExecutionCounters.rows_routed` was split into `rows_routed_success`
(MOVE — gate `route_to_sink`) and `rows_routed_failure` (DIVERT — transform
`on_error` reroute). The split surfaces the MOVE/DIVERT distinction defined
in this ADR all the way through to the run-status predicate, allowing
gate-only pipelines (whose terminal-routed sinks ARE the success path) to
classify as `RunStatus.COMPLETED` rather than `FAILED`. The corresponding
`RowOutcome.ROUTED_ON_ERROR` enum value carries the producer-site signal
from the transform on_error path through to the accumulator and is
persisted directly to `token_outcomes.outcome` for audit-trail
attributability.

**Predicate asymmetry — `RowOutcome.DIVERTED` vs `RowOutcome.ROUTED_ON_ERROR`.**
Three "divert"-flavoured concepts coexist after this PR and they are NOT
synonymous: `RoutingMode.DIVERT` (config-level intent label on an edge),
`RowOutcome.DIVERTED` (terminal outcome for sink-write infrastructure
failures redirected to failsink), and `rows_routed_failure` (counter for
transform-side data failures routed via on_error). `rows_routed_failure`
contributes to the `failure_indicator` of the run-status predicate;
`rows_diverted` does not. The asymmetry is principled, not incidental: a
sink-write failure (DIVERTED) is sink-side infrastructure breakage that
the failsink absorbs cleanly, leaving the run capable of `COMPLETED`
status; a transform-side data failure (ROUTED_ON_ERROR) is row-data
breakage that prevented the row from reaching a value-producing terminal,
which is the structural definition of a failure indicator. See the
"Terminology and predicate asymmetry" section of the
`2026-05-02-rows-routed-counter-split` plan for the full analysis,
including the rejected Option B (unifying `rows_diverted` into the
failure indicator) and the rationale for treating any future unification
as a separate ADR-level amendment.

**`routed_destinations` remains a landed-count map.** The per-sink
`routed_destinations` map is intentionally not split into MOVE and DIVERT
submaps in this PR. It records where routed rows landed, not why they were
routed. Consumers must not infer routing intent from this map. Use
`rows_routed_success` / `rows_routed_failure` for aggregate predicate role,
`token_outcomes.outcome` for token-level producer circumstance, and
`routing_events.mode` for edge-level route mode. Splitting the per-sink map
would be a separate audit-reporting feature because it changes display and
aggregation semantics beyond the P1 status bug.

**Upgrade-boundary semantics.** This PR also changes the meaning of two
read surfaces at the upgrade boundary. `ProgressEvent.rows_succeeded`
previously counted all routed rows in the progress display path; after the
split, it counts `rows_succeeded + rows_routed_success` and excludes
on_error-routed rows. MCP `outcome_distribution` is a distribution of stored
`token_outcomes.outcome` values; pre-split `"routed"` buckets are legacy
ambiguous, while post-split data has distinct `"routed"` and
`"routed_on_error"` buckets. The database maintenance runbook records the
operator action: delete/recreate stale runtime state at upgrade, or preserve
legacy evidence only with date/commit-context qualification and an accepted
audit limitation.

**Producer-site discrimination record.** The project-wide rule for encoding
producer-known terminal circumstances as `RowOutcome` variants, plus the
naming rule explaining why `RowOutcome.ROUTED_ON_ERROR` maps to the
predicate-role counter `rows_routed_failure`, is recorded in ADR-018
("Producer-Site Outcome Discrimination"). This ADR-004 amendment is only the
explicit-sink-routing local application of that decision.
```

- [ ] **Step 17: Update token-outcomes contract and database maintenance runbook**

Add a row to the outcome contract table (after the existing `ROUTED` row):

```markdown
| ROUTED_ON_ERROR | yes | sink_name, error_hash | RowProcessor (transform on_error reroute via DIVERT) |
```

(Adjust the column shape to match the existing table format — read the current header row first.)

Then update `docs/runbooks/database-maintenance.md` by adding this subsection under `## Procedure`, before the general retention/deletion steps:

```markdown
### Rows-routed counter split deployment note (2026-05-02)

The `rows_routed` counter split (`elspeth-5069612f3c`) is a runtime-state
semantic migration even when the Landscape SQL schema does not change. Before
the split, `token_outcomes.outcome='routed'` represented both gate
`route_to_sink` MOVE rows and transform `on_error` DIVERT rows. After the
split, new transform `on_error` rows are recorded as `routed_on_error`, so
preserving an old Landscape audit database makes historical `routed` rows
legacy ambiguous.

Checkpoint files are also stale across this upgrade. They may contain serialized
`RunResult`, `ExecutionCounters`, `AggregationFlushResult`, or progress payloads
with the old single `rows_routed` shape. Delete the configured checkpoint
files/directories before starting new code. Do not resume a pre-split checkpoint
with post-split code.

For dev, staging, and any pre-1.0 production deployment that accepts destructive
pre-1.0 maintenance, stop the service, archive the current Landscape audit
database and checkpoint directory if retention is required, then delete/recreate
the Landscape audit database, sessions database, and checkpoint files/directories
during the same deployment. Do not run new code against the old Landscape DB and
then interpret pre-split `routed` rows as MOVE-only evidence.

Two read surfaces cross a semantic boundary:

- `ProgressEvent.rows_succeeded` before the split may include all routed rows in
  the display-success count; after the split it excludes on_error-routed rows.
- MCP `outcome_distribution["routed"]` before the split is legacy ambiguous;
  after the split, transform on_error rows appear as `outcome_distribution["routed_on_error"]`.

If an environment must preserve the old Landscape database, checkpoint archive,
or generated audit/progress/MCP evidence, record an accepted audit limitation in
the release notes: pre-split `token_outcomes.outcome='routed'`, pre-split
`ProgressEvent.rows_succeeded`, and pre-split MCP `outcome_distribution["routed"]`
require date/commit-context qualification before being used as audit evidence.
This is an explicit limitation, not a migration shim.
```

- [ ] **Step 18: Commit the test sweep + docs**

```bash
git add tests/ src/elspeth/web/frontend/src/types/index.ts src/elspeth/web/frontend/src/stores/executionStore.ts src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx docs/architecture/adr/004-adr-explicit-sink-routing.md docs/architecture/adr/018-producer-site-outcome-discrimination.md docs/architecture/adr/README.md docs/contracts/token-outcomes/00-token-outcome-contract.md docs/runbooks/database-maintenance.md
git commit -m "test(rows-routed-split): sweep test suite and dashboard for new counter shape (elspeth-5069612f3c)"
```

- [ ] **Step 19: Final verification — full suite + integration + enforcers**

Run all gates one more time as separate commands so the first failing gate is obvious in logs:

```bash
.venv/bin/python -m pytest tests/
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py
npm --prefix src/elspeth/web/frontend run test
npm --prefix src/elspeth/web/frontend run build
```

Expected: every command exits 0. If anything fails, debug before proceeding.

- [ ] **Step 20: Close the downstream P1 elspeth-71520f5e30**

Use the filigree MCP `close_issue` (or the CLI fallback `filigree close elspeth-71520f5e30 --reason="Fixed by elspeth-5069612f3c — rows_routed split landed in commit <SHA>; gate-routed pipelines now classify as RunStatus.COMPLETED, the FastAPI reproducer passes, and the Runs dashboard UI test renders the completed status without the stale failure alert."`) — substitute the actual final commit SHA from `git rev-parse HEAD`.

- [ ] **Step 21: Close elspeth-5069612f3c**

```bash
filigree close elspeth-5069612f3c --reason="rows_routed counter split landed; gate-routed pipelines now classify as COMPLETED (rows_routed_success indicator); on_error pipelines classify as FAILED (rows_routed_failure indicator). Three predicate sites updated lockstep: contracts/run_result.py::_check_status_invariant, web/execution/schemas.py::_check_status_row_count_invariant, web/execution/schemas.py::_validate_row_decomposition. Predicate matrix tests cover rows_quarantined/rows_coalesce_failed crossed with the new routed counters, plus a Hypothesis derived-status biconditional property. API, MCP, and dashboard VAL gates added: FastAPI reproducer, MCP outcome_distribution routed_on_error assertion, and RunsView DOM test proving the visible dashboard renders completed without the stale structural-failure alert. Processor on_error producer refuses to fabricate FailureInfo.message='unknown_error' for ROUTED_ON_ERROR; data_flow_repository preserves a terminal unknown-outcome raise. ADR-018 created for producer-site outcome discrimination, public API field-name stability, and the RowOutcome-vs-counter naming rule; ADR-004 amended with the local explicit-sink-routing counter split, routed_destinations landed-count limitation, predicate-asymmetry note, and ProgressEvent/MCP upgrade-boundary note. Token-outcome contract extended for ROUTED_ON_ERROR (sink_name + error_hash). Database maintenance runbook updated: pre-split Landscape token_outcomes.outcome='routed', pre-split ProgressEvent.rows_succeeded, and pre-split MCP outcome_distribution['routed'] are legacy ambiguous unless the Landscape audit DB is archived/deleted/recreated at deploy time; configured checkpoint files/directories must also be archived/deleted/recreated before upgrade. Follow-up gaps promoted before close: <RUN_RECORD_PREDICATE_GAP_ISSUE_ID> (RunRecord row-count predicate gap — Tier-1 read-side audit gap) and <SHARED_STATUS_PREDICATE_ISSUE_ID> (three-mirror predicate consolidation — real Filigree task/feature; canonical fix is to extract evaluate_status_predicate() to L0 contracts/ and have L3 mirrors delegate)."
```

---

## Self-review (writing-plans final check)

**Spec coverage:** All 8 numbered task scopes from the spec (TDD seed → L0 contract → ExecutionCounters/AggregationFlushResult → producer split → orchestrator core propagation → sessions DB → web schemas/routes/service-message → test sweep + integration + docs + downstream close) are present as Tasks 1-8. The frontend is in scope even though Phase 1 verification found no existing `rows_routed` / `rowsRouted` identifier in `src/elspeth/web/frontend/src/`: absence of the old field does not validate the dashboard surface where the user observed the failure. Task 8 Step 9e adds a Runs dashboard DOM gate and frontend type/store updates so the UI VAL gate is explicit.

**Placeholder scan:** No "TBD", "implement later", "add appropriate", or "similar to Task N" in any step. Every code block contains the actual code. Every command is exact except the final closeout command's explicit `<SHA>` / follow-up issue substitution points, which tell the implementing worker to insert runtime-created IDs. The "RoutedDestinations" Counter merge logic in Task 3 step 2 (`__add__`) was kept as the original `combined_destinations` recipe. Task 4 Step 1 now forbids carrying the pre-existing `"unknown_error"` fallback into `ROUTED_ON_ERROR`; falsy `transform_result.reason` raises `OrchestrationInvariantError` before `FailureInfo` construction. Task 4 Step 2's Landscape audit-distinguishability resolution is fully in-scope: the new `RowOutcome.ROUTED_ON_ERROR` value persisted to `token_outcomes.outcome` is the primary distinguishability mechanism, with `RoutingEvent.mode` providing a producer-scoped edge-level cross-check and `error_hash` providing the single-hop "what error triggered the rerouting?" answer. The plan explicitly rejects a global `DIVERT`⇔`ROUTED_ON_ERROR` biconditional because source quarantine and sink failsink also use `RoutingMode.DIVERT`, and it explicitly preserves/adds a terminal unknown-outcome raise in `_validate_outcome_fields`. The plan also handles stale upgrade semantics for Landscape DB, sessions DB, checkpoint files, `ProgressEvent.rows_succeeded`, and MCP `outcome_distribution`: operators archive/delete/recreate stale state for dev/staging/pre-1.0 deploys, or document legacy evidence as accepted audit-limitation evidence. ADR-004 records that `routed_destinations` remains a per-sink landed-count map, not a MOVE/DIVERT intent map. ADR-018 is now the discoverable project-wide home for producer-site outcome discrimination, public API field-name stability, and the naming rule that maps `RowOutcome.ROUTED_ON_ERROR` to the predicate-role counter `rows_routed_failure`. Task 1 now includes the quarantined/coalesce-failed matrix tests, the Hypothesis derived-status biconditional property, and RowResult `ROUTED_ON_ERROR` invariant tests that require a real `FailureInfo` instance at runtime. Task 6 Step 3 (RunRecord.__post_init__) is explicitly a no-op for this PR — the dataclass does not currently enforce the row-count predicate. **Deferred follow-up work is no longer left as ambient observation only:** create or promote a Filigree issue for the RunRecord row-count predicate gap (Tier-1 read-side audit gap; suggested resolution: add the biconditional check to `RunRecord.__post_init__` mirroring the L0 predicate), and create or promote a real Filigree task/feature for the three-mirror predicate consolidation (Limits-to-Growth structural smell; suggested title: `Extract shared row-count status predicate to L0 contracts`; suggested resolution: extract `evaluate_status_predicate(...)` to L0 `contracts/` and have all three current mirror sites delegate). Both follow-ups must be anchored to specific source-line locations and source-issue-linked to `elspeth-5069612f3c` so they survive plan revisions and compaction.

**Type consistency:** Field names `rows_routed_success` and `rows_routed_failure` are used uniformly across all 8 tasks. `RowOutcome.ROUTED_ON_ERROR` is the new enum value, used uniformly. Function signatures match: `_validate_row_decomposition` and `_check_status_row_count_invariant` use the same parameter list across Task 7 step 1 and step 2. The `derive_terminal_run_status` signature in Task 2 step 8 uses keyword-only parameters with the new fields, matching the L0 unit test invocations in Task 1 step 2.

**Surface gaps caught during writing:** The two `rows_succeeded + rows_routed` additions at orchestrator `core.py:2420,2973` (ProgressEvent display) were NOT in the original spec but are critical — they hard-code "routed counts as success" for the in-flight progress UI. After the split they become `rows_succeeded + rows_routed_success` (excluding rows_routed_failure from the progress "successful" count). Task 5 step 3 covers the code change, and ADR-004/runbook text covers the silent upgrade-boundary semantics. MCP `outcome_distribution` was also added as a validation surface because it exposes stored token-outcome buckets directly; it now gets a `routed_on_error` assertion and a legacy-ambiguous note for pre-split `"routed"` buckets.

**Risk gates:** Task 8 steps 10-15 enforce the done conditions before the downstream P1 closes. The full pytest run (step 10), mypy (step 11), ruff (step 12), tier-model enforcer (step 13), freeze-guard enforcer (step 14), and the no-bare-`rows_routed` grep (step 15) all gate the close steps. No partial-green close path exists.
