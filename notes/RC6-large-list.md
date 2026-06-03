  P0 — block any RC6 dev testing (3)

  ┌────────────────────┬──────────────────────────────────────────────────────────────────────────┐
  │         ID         │                                  Title                                   │
  ├────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ elspeth-941f1508f5 │ Worker can steal back its own in-flight lease via recover_expired_leases │
  ├────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ elspeth-01942858c3 │ Resume picks an arbitrary source's schema_contract via next(iter(...))   │
  ├────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ elspeth-5c5e88b071 │ PENDING_SINK starvation: only one pending-sink drained per drain cycle   │
  └────────────────────┴──────────────────────────────────────────────────────────────────────────┘

  P1 — structural correctness + missing docs/tests (34)

  ┌────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
  │         ID         │                                         Title                                          │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-06aecb78a0 │ docs architecture/landscape.md missing run_sources / scheduler / source_node_id schema │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-11a4ed2630 │ TokenManager source_row_index / ingest_sequence accept None defaults                   │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-1e162ad261 │ Forked blob-backed sources — rewrite path drops the plural source map                  │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-1e3ae62d5e │ on_start attribution uses first-source context for every source                        │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-1ed6db3db4 │ CompositionState.source singular field is a UI-layer compatibility shim                │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-2b608abbd3 │ Scheduler state transitions absent from Landscape audit trail                          │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-2e2f2184ab │ Two writers for source schema contract — runs.contract_json + run_sources              │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-30e7ac9571 │ SINK exempted from "multi-producer requires QUEUE" rule                                │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-336e3f704f │ Forked multi-source states — blob-backed sources can be dropped or leaked              │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-34d83daedc │ Type-enforce scheduler engine input — TokenSchedulerRepository(engine: Tier1Engine)    │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-4548f560da │ Execution blob ownership — non-primary named sources bypass blob guards                │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-4678a5aa73 │ claim_ready/claim_pending_sink SELECT-then-UPDATE racy under WAL multi-worker          │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-5335eb63e4 │ unprocessed_rows 3-tuple|4-tuple union discriminated by len()                          │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-543ee35ed3 │ MCP and export read surfaces drop resolved_prompt_template_hash                        │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-559bce3459 │ No runbook covers TokenWorkItem lease recovery                                         │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-57d0031a14 │ No architectural doc explains the multi-source / scheduler design                      │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-6116873e3b │ No test asserts source isolation under concurrent multi-source execution               │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-6ebb263e61 │ headers: original sinks — multi-source writes reuse the wrong contract                 │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-71dcedcb66 │ Zero crash-and-resume coverage for multi-source pipelines                              │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-781e042709 │ legacy_single_source_invocation facade in build_execution_graph                        │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-7ef5d9ff67 │ Run outputs panel — stale artifact manifest across run switches                        │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-7f3ac1ac65 │ Exception message "Do not fabricate source_row_index" should be a doc                  │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-86de46bcd4 │ Composer skill says "Every pipeline needs: one source"                                 │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-9b17af34ca │ Shared review compatibility — snapshots without sources fail validation                │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-9c0a79ed26 │ Scheduler lease retry attempt offset wired to wrong queue path (+ NameError)           │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-9d30da4325 │ Active-run blob guard — named source blobs invisible to lifecycle protection           │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-a7aa07b7ce │ Blob-backed file sources — storage paths leak through redaction                        │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-af87655cdb │ ElspethSettings.source field is a documented legacy shim                               │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-b680e81bce │ Dual drain paths in RowProcessor — legacy in-memory queue preserved for tests          │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-bc81207798 │ Multi-source pipelines run sources sequentially, not concurrently                      │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-bdc43c911e │ get_source_id raises GraphValidationError for multi-source graphs                      │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-c2aa936ad8 │ contracts/system-operations.md Coalesce wording assumes single-source row_id           │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-d8cc46680d │ Named source blob refs — execution validates only the legacy source                    │
  ├────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-de292f9e84 │ CI gate — Row identity contract fixtures fail at import                                │
  └────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

  P2 — polish, test gaps, and review-cluster items (23)

  ┌────────────────────┬─────────────────────────────────────────────────────────────────────────────┐
  │         ID         │                                    Title                                    │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-04980fc019 │ Proof repair gate — named source blob diagnostics skip forced repair        │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-0bae6d8a52 │ No tests for recover_expired_leases when multiple expired items exist       │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-37c1dde240 │ wire_secret_ref — named sources cannot be targeted (phantom legacy source)  │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-3fc847c4be │ Synthesised cache audit — writer rejects multi-source topologies            │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-54e9c72f1b │ processor.py 3620 LOC carrying scheduler state machine — needs subdivision  │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-6db715b7c3 │ Tutorial runtime normalization — named sources dropped on save              │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-7bb7124e8f │ No chaos coverage for multi-source / concurrent token scheduler             │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-80671c0eb3 │ diff_pipeline source summary — named-source-only changes invisible          │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-84d4680346 │ Composer state-claim grounding ignores later named sources                  │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-8536552dcb │ No deferred-FK + WAL PRAGMA discipline on scheduler-bearing connections     │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-9738349228 │ Inline source projection — first non-inline blob hides later inline sources │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-99f992f8bd │ set_source_from_blob source names — invalid names crash plugins             │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-9c7ae2d60e │ test_reconstruct_resume_state bypasses production resume path               │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-9e083c57fe │ set_source_from_blob affected nodes use noncanonical component ids          │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-a4b0c3e00f │ Audit readiness source surfaces — named sources omitted                     │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-af13a34ccd │ SQLite schema epoch — required scheduler row FK ships without epoch advance │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-af612d0470 │ Named blob sources — duplicate refs fail run-setup link insertion           │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-c33c71aafe │ Source secret wiring — target_id ignored for named sources                  │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-d42360f518 │ Pipeline graph view — named sources beyond compatibility invisible          │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-d869cc0113 │ No allowlist sweep / fingerprint rotation for multi-source code shifts      │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-e51eaed773 │ No test that source_node_id is durably attributed on every row              │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-e7463a935b │ Scheduler ready-claim ordering — DB-dependent replay order                  │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ elspeth-e8a1250782 │ Property test models OLD lifecycle, not scheduler states                    │
  └────────────────────┴─────────────────────────────────────────────────────────────────────────────┘

  P3 — pre-publish docs/governance items (12)

  ┌────────────────────┬─────────────────────────────────────────────────────────────────────────┐
  │         ID         │                                  Title                                  │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-1869c9ba64 │ ingest_sequence set from counters AFTER quarantine branch, BEFORE check │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-2409a7c7bf │ CLAUDE.md "Source: Load data — exactly 1 per run"                       │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-40886ef9f8 │ test_concurrent_resume.py misnamed — does not test concurrent resume    │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-8c4ca2d89c │ release/elspeth-progress-rc1-to-rc5.md missing scheduler delivery       │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-8cdc7c4368 │ Row identity types — make required source indexes mechanical            │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-8fe6fc5f24 │ Identity passthrough advisory — named source producers ignored          │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-97f8509b35 │ Promote PRAGMA probe-and-assert from sessions to Landscape engine       │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-addd3dc41f │ PRAGMA verification test on Landscape plain-SQLite path                 │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-bc91898548 │ guarantees.md §7.1 "single-threaded in RC-3" contradicted by scheduler  │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-dde60f76b4 │ Redaction policy doesn't carry per-source provenance                    │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-e22a476972 │ Shared inspect client validation omits sources from snapshot contract   │
  ├────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ elspeth-e4cf92586c │ reference/configuration.md "Source Settings" single-source-only model   │
  └────────────────────┴─────────────────────────────────────────────────────────────────────────┘
