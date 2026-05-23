# Multi-Source Token Scheduler Audit — Dedup Map

74 tickets (3 P0, 27 P1, 40 P2, 4 P3) consolidated into 26 groups.

## Section 1: Group Summary

```
[G1]  scheduler-lease-self-steal-on-expiry          | tier-1-must-fix-now    | P0 | canonical: elspeth-941f1508f5 | duplicates: — | merge-into-canonical: —
[G2]  resume-arbitrary-contract-next-iter           | tier-1-must-fix-now    | P0 | canonical: elspeth-01942858c3 | duplicates: elspeth-c09bd87ac7 | merge-into-canonical: elspeth-d5f0194fc8
[G3]  pending-sink-drain-starvation                 | tier-1-must-fix-now    | P0 | canonical: elspeth-5c5e88b071 | duplicates: — | merge-into-canonical: —
[G4]  legacy-source-field-elspethsettings           | tier-2-rc6-structural  | P1 | canonical: elspeth-af87655cdb | duplicates: elspeth-3791835e4c, elspeth-4147d0534d | merge-into-canonical: elspeth-2f54f51d9d, elspeth-287a5c3a98
[G5]  legacy-single-source-facade-build-execution-graph | tier-2-rc6-structural | P1 | canonical: elspeth-781e042709 | duplicates: elspeth-5b1cc1ec49, elspeth-18e44f4750 | merge-into-canonical: —
[G6]  dual-writer-source-contract-singleton-vs-per-source | tier-2-rc6-structural | P1 | canonical: elspeth-2e2f2184ab | duplicates: elspeth-b157fa3fad | merge-into-canonical: elspeth-3ed7516cad
[G7]  dual-drain-path-test-only                     | tier-2-rc6-structural  | P1 | canonical: elspeth-b680e81bce | duplicates: — | merge-into-canonical: —
[G8]  graph-singular-source-accessors-stale         | tier-2-rc6-structural  | P1 | canonical: elspeth-bdc43c911e | duplicates: — | merge-into-canonical: elspeth-a3c9663cea
[G9]  composer-state-singular-source-compat-shim    | tier-2-rc6-structural  | P1 | canonical: elspeth-1ed6db3db4 | duplicates: — | merge-into-canonical: —
[G10] unprocessed-rows-tuple-length-discriminator   | tier-2-rc6-structural  | P1 | canonical: elspeth-5335eb63e4 | duplicates: — | merge-into-canonical: —
[G11] transitional-none-defaults-token-row-identity | tier-2-rc6-structural  | P1 | canonical: elspeth-11a4ed2630 | duplicates: elspeth-f75db57d61 | merge-into-canonical: —
[G12] multi-source-actually-sequential              | tier-2-rc6-structural  | P1 | canonical: elspeth-bc81207798 | duplicates: — | merge-into-canonical: elspeth-e694f1530b
[G13] claude-md-single-source-statement             | tier-3-pre-publish     | P3 | canonical: elspeth-2409a7c7bf | duplicates: elspeth-b84278b7a6 | merge-into-canonical: —
[G14] single-source-doc-corpus-stale (multi-file)   | tier-3-pre-publish     | P3 | canonical: elspeth-e4cf92586c | duplicates: elspeth-65a113a9de, elspeth-00e3ba8eb0, elspeth-9fa14898a3, elspeth-9e77c755b5, elspeth-c02cd6a612 | merge-into-canonical: —
[G15] guarantees-single-threaded-rc3-stale          | tier-3-pre-publish     | P3 | canonical: elspeth-bc91898548 | duplicates: — | merge-into-canonical: —
[G16] composer-multi-source-blind                   | tier-3-pre-publish     | P1 | canonical: elspeth-86de46bcd4 | duplicates: elspeth-f77ec72927, elspeth-5617facba0 | merge-into-canonical: —
[G17] missing-adr-multi-source-and-scheduler        | tier-3-pre-publish     | P1 | canonical: elspeth-57d0031a14 | duplicates: elspeth-f089ed57e5 | merge-into-canonical: elspeth-dfac4da7cf, elspeth-cd7ef70f6b
[G18] landscape-md-missing-run-sources-scheduler-tables | tier-3-pre-publish | P1 | canonical: elspeth-06aecb78a0 | duplicates: — | merge-into-canonical: —
[G19] runbook-and-mcp-guide-scheduler-gaps          | tier-3-pre-publish     | P1 | canonical: elspeth-559bce3459 | duplicates: elspeth-fcacf63e07, elspeth-514aee0e28, elspeth-877cfb99c0 | merge-into-canonical: —
[G20] system-operations-coalesce-row-id-ambiguity   | tier-3-pre-publish     | P1 | canonical: elspeth-c2aa936ad8 | duplicates: — | merge-into-canonical: —
[G21] release-progress-missing-multi-source-scheduler-entry | tier-3-pre-publish | P3 | canonical: elspeth-8c4ca2d89c | duplicates: — | merge-into-canonical: —
[G22] source-row-identity-no-fabricate-discoverability | tier-3-pre-publish  | P1 | canonical: elspeth-7f3ac1ac65 | duplicates: — | merge-into-canonical: —
[G23] redaction-per-source-provenance-decision-undocumented | tier-3-pre-publish | P3 | canonical: elspeth-dde60f76b4 | duplicates: — | merge-into-canonical: —
[G24] audit-test-suite-source-0-node-id-stale       | tier-3-pre-publish     | P3 | canonical: elspeth-9e77c755b5 | duplicates: — | merge-into-canonical: — (rolled into G14)
[G25] test-coverage-multi-source-and-scheduler-gaps | tier-4-test-gaps       | mixed (P1/P2/P3) | canonical group | duplicates per row below
[G26] code-health-scheduler-extractions             | tier-5-code-health     | P2 | canonical: elspeth-54e9c72f1b | duplicates: — | merge-into-canonical: elspeth-eb47c1b234
[G27] scheduler-cas-races-multi-worker-not-enforced | tier-2-rc6-structural  | P1 | canonical: elspeth-4678a5aa73 | duplicates: — | merge-into-canonical: elspeth-f794562898, elspeth-5ab496c4d5, elspeth-a406d5d2c4
[G28] scheduler-pragma-and-perf-edges               | tier-1-must-fix-now    | P0 (PRAGMA) / P2 (release_waiting) | canonical: elspeth-8536552dcb | duplicates: — | merge-into-canonical: elspeth-6682aee9df
[G29] scheduler-state-transitions-not-in-landscape  | tier-2-rc6-structural  | P1 | canonical: elspeth-2b608abbd3 | duplicates: — | merge-into-canonical: —
[G30] sink-exempted-from-queue-requirement          | tier-2-rc6-structural  | P1 | canonical: elspeth-30e7ac9571 | duplicates: — | merge-into-canonical: —
[G31] ingest-sequence-quarantine-gap-semantics      | tier-3-pre-publish     | P3 | canonical: elspeth-1869c9ba64 | duplicates: — | merge-into-canonical: —
[G32] tier-model-allowlist-multi-source-churn-not-pruned | tier-5-code-health | P2 | canonical: elspeth-d869cc0113 | duplicates: — | merge-into-canonical: —
```

Group G25 expansion (test gaps):
```
[G25a] no_e2e_multi_source_crash_resume + no_mid_source_iteration_crash_test → canonical elspeth-71dcedcb66 (P1) | merge: elspeth-66090df487
[G25b] no_multi_source_isolation_tests → canonical elspeth-6116873e3b (P1) standalone
[G25c] property_test_state_machine_pins_old_lifecycle → canonical elspeth-e8a1250782 (P2) standalone
[G25d] recover_expired_leases_under_tested + scheduler_two_workers_test_sequential + claim_pending_sink_untested + no_claim_ready_ordering_load_test → canonical elspeth-0bae6d8a52 (P2) | merge: elspeth-d047c968ca, elspeth-a11b61226c, elspeth-2d0b958024
[G25e] test_concurrent_resume_misnamed → elspeth-40886ef9f8 (P3) standalone
[G25f] reconstruct_resume_state_bypass → elspeth-9c7ae2d60e (P2) standalone (also overlaps G25a)
[G25g] no_source_node_id_attribution_invariant_test → elspeth-e51eaed773 (P2) standalone
[G25h] no_multi_source_chaos_coverage → elspeth-7bb7124e8f (P2) standalone
[G25i] missing-tests-scheduler-edge-cases → elspeth-4162f81771 (P2) — umbrella, merge into G25a/G25d as evidence
```

---

## Section 2: Per-Group Detail

### [G1] scheduler-lease-self-steal-on-expiry
**Canonical**: elspeth-941f1508f5 (dim1, P0) — Worker reaps its own in-flight lease via `recover_expired_leases`, crashing run with AuditIntegrityError.
- File: src/elspeth/core/landscape/scheduler_repository.py:552-592 + engine/processor.py:2553
- Should be: tier-1, P0 (keep)
- No duplicates / no merges.

### [G2] resume-arbitrary-contract-next-iter
**Canonical**: elspeth-01942858c3 (dim1, P0) — `next(iter(schema_contracts_by_source))` picks arbitrary source's contract on multi-source resume; silently validates rows under wrong schema.
- File: src/elspeth/engine/orchestrator/core.py:3497-3512
- Should be: tier-1, P0
**Duplicates to close**:
- elspeth-c09bd87ac7 (dim2) — same `next(iter(...))` at :3512, dim2 framing of latent footgun (no unique evidence beyond dim1).
**Merge into canonical (unique evidence)**:
- elspeth-d5f0194fc8 (dim4) — test-side framing; copy its 4-step test plan (test_resume_*_contradictory_contracts) as a comment so the fix lands with the regression test.

### [G3] pending-sink-drain-starvation
**Canonical**: elspeth-5c5e88b071 (dim1, P0) — `created_pending_sink_this_drain` flag blocks recovery; only ONE pending-sink draws per resume call, crashes invariant check.
- File: src/elspeth/engine/processor.py:2531-2657
- Should be: tier-1, P0
- Standalone.

### [G4] legacy-source-field-elspethsettings
**Canonical**: elspeth-af87655cdb (dim2, P1) — Root of singular/plural facade: `ElspethSettings.source` shim, 26+ `config.source.*` callsites.
- File: src/elspeth/core/config.py:1354-1358,1458-1479
- Should be: tier-2, P1
**Duplicates to close**:
- elspeth-3791835e4c (dim1) — same `replace(config, source=…)` aliasing inside per-source loop; subset of canonical scope.
- elspeth-4147d0534d (dim2) — `_run_main_processing_loop` `next(iter(config.sources))` — symptomatic of same root cause.
**Merge into canonical (unique evidence)**:
- elspeth-2f54f51d9d (dim2) — PluginBundle/cli_helpers carry both source AND sources; defensive reconciliation block at lines 78-81. Copy as "downstream removal checklist."
- elspeth-287a5c3a98 (dim2) — "engine still single-source-shaped, wrapped externally" architecture framing. Copy as motivating description.

### [G5] legacy-single-source-facade-build-execution-graph
**Canonical**: elspeth-781e042709 (dim2, P1) — `build_execution_graph` accepts both `source=` and `sources=`; forces name to `"source"` literal.
- File: src/elspeth/core/dag/builder.py:135-156,233-260
- Should be: tier-2, P1
**Duplicates to close**:
- elspeth-5b1cc1ec49 (dim1) — same finding, same file, same fix.
- elspeth-18e44f4750 (dim3) — policy-framing of same issue; cites CLAUDE.md No Legacy Code.

### [G6] dual-writer-source-contract-singleton-vs-per-source
**Canonical**: elspeth-2e2f2184ab (dim2, P1) — Two writers for source schema contract (`runs.contract_json` + `run_sources`); resume picks reader at read-time.
- File: src/elspeth/core/landscape/run_lifecycle_repository.py:414-502 + orchestrator/core.py:2509-2513
- Should be: tier-2, P1
**Duplicates to close**:
- elspeth-b157fa3fad (dim1) — duplicate finding focused on `runs.source_schema_json`/`source_field_resolution_json` columns; same dual-writer story.
**Merge into canonical**:
- elspeth-3ed7516cad (dim2) — `_record_schema_contract` getattr-like read-before-write pattern at :2509-2513; copy as the surgical-fix location.

### [G7] dual-drain-path-test-only
**Canonical**: elspeth-b680e81bce (dim2, P1) — `_drain_in_memory_work_queue` kept solely for tests; violates CLAUDE.md "never bypass production code paths."
- File: src/elspeth/engine/processor.py:2371-2387,2477-2511
- Should be: tier-2, P1
- Standalone.

### [G8] graph-singular-source-accessors-stale
**Canonical**: elspeth-bdc43c911e (dim1, P1) — `get_source_id`, `get_first_transform_node` raise/return None for multi-source; latent crash for any caller.
- File: src/elspeth/core/dag/graph.py:447-451,570-580
- Should be: tier-2, P1
**Merge into canonical**:
- elspeth-a3c9663cea (dim1) — `GraphArtifacts.source_id: NodeID` singular field populated via `next(iter(...))`; same singular-accessor smell on a different surface. Copy as second-call-site that needs the same cleanup.

### [G9] composer-state-singular-source-compat-shim
**Canonical**: elspeth-1ed6db3db4 (dim2, P1) — `CompositionState.source` singular field auto-synced from `sources` plural; UX-layer mirror of G4.
- File: src/elspeth/web/composer/state.py:1738-1806,1882-1916
- Should be: tier-2, P1
- Standalone (frontend/composer surface, not engine).

### [G10] unprocessed-rows-tuple-length-discriminator
**Canonical**: elspeth-5335eb63e4 (dim2, P1) — `unprocessed_rows` is 3-tuple|4-tuple union discriminated by `len()`; needs `ResumedRow` dataclass.
- File: src/elspeth/core/checkpoint/recovery.py:206-420 + orchestrator/core.py:3089,3149-3170
- Should be: tier-2, P1
- Standalone. Cleanup blocked-by G6 (dual-writer) resolution.

### [G11] transitional-none-defaults-token-row-identity
**Canonical**: elspeth-11a4ed2630 (dim2, P1) — `source_row_index`/`ingest_sequence` declared `int | None = None` but runtime guard mandatory; type system should enforce.
- File: src/elspeth/engine/tokens.py:71-103,156-167
- Should be: tier-2, P1
**Duplicates to close**:
- elspeth-f75db57d61 (dim1) — identical finding, same lines, same fix (delete `_require_source_row_identity`, make params required).

### [G12] multi-source-actually-sequential
**Canonical**: elspeth-bc81207798 (dim1, P1) — Per-source dispatch is sequential `for`-loop, no cross-source interleaving; branch name misleads.
- File: src/elspeth/engine/orchestrator/core.py:3315-3346
- Should be: tier-2, P1
**Merge into canonical**:
- elspeth-e694f1530b (dim1) — same root cause: sequential iteration → time-dependent sources fire at different wall times + missing between-source shutdown check. Copy the between-source-shutdown gap as a sub-task.

### [G13] claude-md-single-source-statement
**Canonical**: elspeth-2409a7c7bf (dim3, P3) — `CLAUDE.md:119` still says "exactly 1 per run."
- File: CLAUDE.md:119
- Should be: tier-3, P3 (correct-today-but-wrong-for-RC6, drop from P1)
**Duplicates to close**:
- elspeth-b84278b7a6 (dim2) — same file, same line, same fix.

### [G14] single-source-doc-corpus-stale
**Canonical**: elspeth-e4cf92586c (dim3, P3) — `docs/reference/configuration.md` "Source Settings" says single-source-only.
- File: docs/reference/configuration.md:150-195
- Should be: tier-3, P3
**Duplicates to close** (all "stale single-source doc" with no unique evidence — one omnibus fix-up):
- elspeth-65a113a9de (dim3) — `docs/contracts/plugin-protocol.md:224`.
- elspeth-00e3ba8eb0 (dim3) — `docs/contracts/execution-graph.md:595` (`get_source()`).
- elspeth-9fa14898a3 (dim3) — `README.md:680-721` pipeline example.
- elspeth-9e77c755b5 (dim3) — `docs/audit/test-suite/u-engine-1-*.md` "source-0" naming.
- elspeth-c02cd6a612 (dim3) — `docs/superpowers/plans/`, `docs/composer/evidence/` "1 source" framing.
- (Note: each cites a distinct file path; close as duplicates of G14 BUT the canonical's fix-list must enumerate each file. Comment the file list on canonical before closing.)

### [G15] guarantees-single-threaded-rc3-stale
**Canonical**: elspeth-bc91898548 (dim3, P3) — `docs/release/guarantees.md §7.1` says "single-threaded in RC-3."
- File: docs/release/guarantees.md:253
- Should be: tier-3, P3 (correct-today-but-wrong-for-RC6)
- Standalone.

### [G16] composer-multi-source-blind
**Canonical**: elspeth-86de46bcd4 (dim3, P1) — `pipeline_composer.md:706` skill still teaches "Every pipeline needs: one source."
- File: src/elspeth/web/composer/skills/pipeline_composer.md:706
- Should be: tier-3, P1 (missing-not-wrong: composer skill must teach multi-source for RC6)
**Duplicates to close**:
- elspeth-f77ec72927 (dim3) — `docs/reference/composer-tools.md:186` says same; resolve together.
- elspeth-5617facba0 (dim3) — guided-mode spec says "Multi-source out of scope for ELSPETH itself" — same composer-blind theme.

### [G17] missing-adr-multi-source-and-scheduler
**Canonical**: elspeth-57d0031a14 (dim2, P1) — No ADR/design doc for multi-source or scheduler; design lives only in code + reviewer's note.
- File: docs/architecture/adr/ (absent)
- Should be: tier-3, P1
**Duplicates to close**:
- elspeth-f089ed57e5 (dim3) — same finding from docs lens, less detailed action items.
**Merge into canonical**:
- elspeth-dfac4da7cf (dim2) — QUEUE node observed-schema design has no design doc; copy as one of the ADR's required sub-decisions.
- elspeth-cd7ef70f6b (dim2) — scheduler multi-worker promised-not-enforced; copy as another required ADR section (lease ownership boundary).

### [G18] landscape-md-missing-run-sources-scheduler-tables
**Canonical**: elspeth-06aecb78a0 (dim3, P1) — `docs/architecture/landscape.md` missing `run_sources`, `token_work_items`, new row columns.
- File: docs/architecture/landscape.md:84-110
- Should be: tier-3, P1 (missing-not-wrong)
- Standalone.

### [G19] runbook-and-mcp-guide-scheduler-gaps
**Canonical**: elspeth-559bce3459 (dim3, P1) — No `docs/runbooks/scheduler-lease-recovery.md`.
- File: docs/runbooks/ (absent)
- Should be: tier-3, P1
**Duplicates to close**:
- elspeth-fcacf63e07 (dim3) — `resume-failed-run.md` missing multi-source + drift-refusal; close as part of runbook umbrella but capture its specific gaps as comment.
- elspeth-514aee0e28 (dim3) — `landscape-mcp-analysis.md` guide; same operator-diagnostic gap.
- elspeth-877cfb99c0 (dim4) — no MCP tools/diagnostic surface for stuck/leaked work_items. Bundle into umbrella (the MCP tool set and the docs go together).

### [G20] system-operations-coalesce-row-id-ambiguity
**Canonical**: elspeth-c2aa936ad8 (dim3, P1) — `system-operations.md §555` Coalesce invariants assume single-source `row_id`.
- File: docs/contracts/system-operations.md:553-555
- Should be: tier-3, P1
- Standalone.

### [G21] release-progress-missing-multi-source-scheduler-entry
**Canonical**: elspeth-8c4ca2d89c (dim3, P3) — `docs/release/elspeth-progress-rc1-to-rc5.md` doesn't mention this delivery.
- File: docs/release/elspeth-progress-rc1-to-rc5.md
- Should be: tier-3, P3
- Standalone.

### [G22] source-row-identity-no-fabricate-discoverability
**Canonical**: elspeth-7f3ac1ac65 (dim3, P1) — "Do not fabricate" lives only in an exception string; needs plugin-protocol doc + lint rule.
- File: notes/branch-review-... (constraint discoverability)
- Should be: tier-3, P1 (missing-not-wrong)
- Standalone. Becomes dead code once G11 deletes `_require_source_row_identity`; but the plugin-protocol doc + lint rule remain valid.

### [G23] redaction-per-source-provenance-decision-undocumented
**Canonical**: elspeth-dde60f76b4 (dim3, P3) — Redaction collapses all source paths to constant `<redacted-blob-source-path>`; no source provenance.
- File: src/elspeth/web/composer/redaction.py + snapshot
- Should be: tier-3, P3 (design decision required; security implications)
- Standalone.

### [G24] (merged into G14)

### [G25a] no_e2e_multi_source_crash_resume
**Canonical**: elspeth-71dcedcb66 (dim4, P1) — Zero crash-and-resume coverage for multi-source.
- File: tests/e2e/recovery/test_crash_and_resume.py
- Should be: tier-4, P1
**Merge into canonical**:
- elspeth-66090df487 (dim4) — mid-source-iteration crash; same suite. Copy its 3 test cases as additional scenarios.

### [G25b] no_multi_source_isolation_tests
**Canonical**: elspeth-6116873e3b (dim4, P1) — Source-isolation tests absent.
- File: tests/integration/pipeline/ (absent file)
- Should be: tier-4, P1 (source-isolation gap)
- Standalone.

### [G25c] property_test_state_machine_pins_old_lifecycle
**Canonical**: elspeth-e8a1250782 (dim4, P2) — Hypothesis state machine models old lifecycle, not scheduler states.
- File: tests/property/engine/test_token_lifecycle_state_machine.py:58-76
- Should be: tier-4, P2
- Standalone.

### [G25d] scheduler-lease-and-claim-edge-test-gaps
**Canonical**: elspeth-0bae6d8a52 (dim4, P2) — `recover_expired_leases` under-tested for multi-expiry.
- File: tests/unit/core/test_multi_source_foundation.py
- Should be: tier-4, P2
**Merge into canonical** (all four are scheduler-test gaps in the same file; consolidate):
- elspeth-d047c968ca (dim4) — `test_scheduler_claim_ready_two_workers` is sequential, not contended.
- elspeth-a11b61226c (dim4) — `claim_pending_sink` has no direct CAS test.
- elspeth-2d0b958024 (dim4) — `claim_ready` ordering load test absent.
- elspeth-4162f81771 (dim1) — umbrella "5 missing scheduler tests" enumeration; copy its list.

### [G25e] test_concurrent_resume_misnamed
**Canonical**: elspeth-40886ef9f8 (dim4, P3) — File doesn't test concurrent resume despite name.
- File: tests/e2e/recovery/test_concurrent_resume.py
- Should be: tier-4, P3 (rename / mostly cosmetic; real coverage tracked by G25a)
- Standalone.

### [G25f] reconstruct_resume_state_bypass
**Canonical**: elspeth-9c7ae2d60e (dim4, P2) — Test calls private `_reconstruct_resume_state` instead of public `Orchestrator.resume`.
- File: tests/integration/pipeline/test_resume_comprehensive.py:440-577
- Should be: tier-4, P2 (test-only path violation)
- Standalone but overlaps G25a (the resume rewrite will subsume this).

### [G25g] no_source_node_id_attribution_invariant_test
**Canonical**: elspeth-e51eaed773 (dim4, P2) — No invariant test that `rows.source_node_id` is NOT NULL across all rows / survives resume.
- File: tests/integration/test_adr_019_cross_table_invariants.py (precedent)
- Should be: tier-4, P2
- Standalone.

### [G25h] no_multi_source_chaos_coverage
**Canonical**: elspeth-7bb7124e8f (dim4, P2) — ChaosLLM/ChaosWeb/ChaosEngine unwired against multi-source/scheduler.
- File: tests/performance/stress/ (no co-located file)
- Should be: tier-4, P2
- Standalone.

### [G26] code-health-scheduler-extractions
**Canonical**: elspeth-54e9c72f1b (dim2, P2) — `processor.py` is 3620 LOC; extract `SchedulerDriver`.
- File: src/elspeth/engine/processor.py
- Should be: tier-5, P2
**Merge into canonical**:
- elspeth-eb47c1b234 (dim2) — `orchestrator/core.py` 3894 LOC; extract `resume.py` + `MultiSourceCoordinator`. Same theme, different file. Copy its extraction plan.

### [G27] scheduler-cas-races-multi-worker-not-enforced
**Canonical**: elspeth-4678a5aa73 (dim1, P1) — `claim_ready`/`claim_pending_sink` SELECT-then-UPDATE; loser raises AuditIntegrityError instead of benign retry.
- File: src/elspeth/core/landscape/scheduler_repository.py:416-550
- Should be: tier-2, P1
**Merge into canonical** (same SELECT-then-UPDATE / multi-worker-race family):
- elspeth-f794562898 (dim1) — `mark_blocked_barrier_terminal` rowcount-mismatch escalates to audit-integrity.
- elspeth-5ab496c4d5 (dim1) — `recover_expired_leases` PENDING_SINK path reuses same `work_item_id` (asymmetric attempt-bump) — same CAS-design family.
- elspeth-a406d5d2c4 (dim1) — `mark_failed` optional `expected_lease_owner` window.

### [G28] scheduler-pragma-and-perf-edges
**Canonical**: elspeth-8536552dcb (dim1, P0 for PRAGMA verification) — Audit PRAGMA discipline on scheduler-bearing connections (`foreign_keys=ON`, `WAL`, `busy_timeout`).
- File: src/elspeth/core/landscape/database.py + scheduler_repository.py
- Should be: tier-1, P0 (PRAGMA verification is correctness-critical; severity in description is "needs investigation" — re-grade after operator verifies)
**Merge into canonical**:
- elspeth-6682aee9df (dim1) — `release_waiting` unbounded UPDATE; performance edge in the same scheduler-DB-discipline family.

### [G29] scheduler-state-transitions-not-in-landscape
**Canonical**: elspeth-2b608abbd3 (dim4, P1) — Scheduler state transitions emit no Landscape audit rows; cannot reconstruct lease-expiry timelines.
- File: src/elspeth/core/landscape/scheduler_repository.py (entire file)
- Should be: tier-2, P1 (audit-primacy gap, structural)
- Standalone. Companion of G17 ADR (the design doc has to commit to whether to add a scheduler_events table).

### [G30] sink-exempted-from-queue-requirement
**Canonical**: elspeth-30e7ac9571 (dim1, P1) — `{QUEUE, SINK, COALESCE}` exemption lets multi-source MOVE-fan into a sink without QUEUE primitive; contract hole.
- File: src/elspeth/core/dag/graph.py:338-353
- Should be: tier-2, P1
- Standalone. Either tighten the rule or document the exemption.

### [G31] ingest-sequence-quarantine-gap-semantics
**Canonical**: elspeth-1869c9ba64 (dim1, P3) — Quarantine rows consume `ingest_sequence` numbers; gap semantics undocumented.
- File: src/elspeth/engine/orchestrator/core.py:2922-2982
- Should be: tier-3, P3 (documentation-only; existing behavior is likely correct)
- Standalone.

### [G32] tier-model-allowlist-multi-source-churn-not-pruned
**Canonical**: elspeth-d869cc0113 (dim3, P2) — Heavy allowlist churn (engine.yaml +262, web.yaml +289, core.yaml +46); needs stale-entry sweep + lifecycle audit.
- File: config/cicd/enforce_tier_model/{core,engine,web,contracts}.yaml
- Should be: tier-5, P2 (operational hygiene; uses existing `cicd-allowlist-audit` skill)
- Standalone.

---

## Summary

- **Total groups**: 32 (G1–G32; G24 absorbed into G14, G25 split into G25a–h)
- **Total tickets**: 74 (3 P0 + 27 P1 + 40 P2 + 4 P3 input)

### Tier distribution
| Tier | Groups | Canonical IDs |
|------|--------|---------------|
| tier-1-must-fix-now | 4 | G1, G2, G3, G28 |
| tier-2-rc6-structural | 13 | G4, G5, G6, G7, G8, G9, G10, G11, G12, G27, G29, G30 + (G26 borderline) |
| tier-3-pre-publish | 11 | G13, G14, G15, G16, G17, G18, G19, G20, G21, G22, G23, G31 |
| tier-4-test-gaps | 8 | G25a–h |
| tier-5-code-health | 2 | G26, G32 |

### Priority distribution (after RC6 re-grade)
- **P0**: 4 canonicals (G1, G2, G3, G28-PRAGMA-half)
- **P1**: 19 canonicals (all tier-2-structural + G16, G17, G18, G19, G20, G22, G25a, G25b)
- **P2**: 6 canonicals (G25c, G25d, G25f, G25g, G25h, G26, G32)
- **P3**: 7 canonicals (G13, G14, G15, G21, G23, G25e, G31)

### Closure counts
- **Duplicates to close (no evidence merge)**: 17 tickets
  - elspeth-c09bd87ac7, elspeth-3791835e4c, elspeth-4147d0534d, elspeth-5b1cc1ec49, elspeth-18e44f4750, elspeth-b157fa3fad, elspeth-f75db57d61, elspeth-f77ec72927, elspeth-5617facba0, elspeth-f089ed57e5, elspeth-65a113a9de, elspeth-00e3ba8eb0, elspeth-9fa14898a3, elspeth-9e77c755b5, elspeth-c02cd6a612, elspeth-b84278b7a6, elspeth-fcacf63e07, elspeth-514aee0e28, elspeth-877cfb99c0
- **Merge-then-close (evidence appended)**: 15 tickets
  - elspeth-d5f0194fc8, elspeth-2f54f51d9d, elspeth-287a5c3a98, elspeth-3ed7516cad, elspeth-a3c9663cea, elspeth-e694f1530b, elspeth-dfac4da7cf, elspeth-cd7ef70f6b, elspeth-66090df487, elspeth-d047c968ca, elspeth-a11b61226c, elspeth-2d0b958024, elspeth-4162f81771, elspeth-eb47c1b234, elspeth-f794562898, elspeth-5ab496c4d5, elspeth-a406d5d2c4, elspeth-6682aee9df
- **Canonicals to keep open (re-graded)**: 32

### Truly standalone (no overlap)
G1, G3, G7, G9, G10, G15, G18, G20, G21, G22, G23, G25b, G25c, G25e, G25g, G25h, G29, G30, G31, G32

### Notes
- All dedupe keys are present in every ticket (no MISSING-DEDUPE-KEY flags).
- G14 is the largest dedup cluster (6 tickets) — five tickets close as duplicates, but the canonical's first comment must enumerate the full file fix-list (configuration.md, plugin-protocol.md, execution-graph.md, README.md, u-engine-1-*.md, run-accounting/composer-eval plans) so the omnibus doc-update doesn't lose individual touchpoints.
- G28 PRAGMA verification straddles "investigation" and "P0 if discipline is broken." Operator should run the verification before re-grading.
- Three known P0s (G1/G2/G3) confirmed exactly as briefed: elspeth-941f1508f5, elspeth-01942858c3 (+ dim4 elspeth-d5f0194fc8 + dim2 elspeth-c09bd87ac7 collapse into it), elspeth-5c5e88b071.
