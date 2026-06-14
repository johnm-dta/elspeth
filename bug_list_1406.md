# bug_list_1406 — failing-test remediation worklist

## Process (for each cluster below)

1. Read the test(s) and use systematic debugging to understand the root cause.
2. Search filigree for a related ticket; if none, create one.
3. Resolve by addressing the fault/regression (fix code, or update stale tests
   to a confirmed/operator-decided contract — never mask a real regression).
4. Verify the targeted test file(s) pass fast (no full-suite needed per cluster).
5. Close the ticket, tick the cluster off this list.
6. Parallel agents only where clusters touch disjoint files and cannot collide.

> Self-adjustment (per original step 7): the original "first job — get a full
> bug trace, a test is hanging" is **done**. Root cause of the hang: the suite
> was run with `pytest -n auto` (24 workers) under heavy swap + a
> `--timeout-method=thread` that `os._exit`-ed workers, so xdist hung in
> `teardown_nodes`. Re-run as `pytest tests/ -n 8 --timeout=90
> --timeout-method=signal` completes cleanly: **52 failed, 23 errors, 22193
> passed**. The genuine in-test "hang" is Cluster A (real `time.sleep` retry
> backoff). Worklist below is grouped by ROOT CAUSE, not per-test.

## Diagnostic baseline

- Branch footprint vs `release/0.6.0`: **only** `src/elspeth/plugins/transforms/*`
  + `tests/unit/plugins/**`. Nothing in `web/`, `composer/`, `core/config`,
  audit/ADR durability. → web/composer/config failures are PRE-EXISTING, not
  caused by this branch.

## Worklist

### Branch-caused regressions (this branch owns these)

- [x] **A — LLM sequential-retry hang + stale contract** (THE HANG). DONE — elspeth-1707250b8c; 8 tests updated to B3.7 contract, 164/164 in the 3 files green in ~6.6s.
  `src/elspeth/plugins/transforms/llm/transform.py` B3.7 made sequential
  multi-query do bounded local retry (real `time.sleep`, budget
  `max_capacity_retry_seconds=3600`). Branch updated `test_azure_multi_query_retry.py`
  but left these stale (assert old `retryable=True/"multi_query_failed"` and
  sleep for real → 90s timeout):
  - tests/unit/plugins/llm/test_openrouter_multi_query.py: test_process_row_rate_limit_returns_retryable_error, test_process_row_server_error_returns_retryable_error, test_process_row_network_error_returns_retryable_error, TestHTTPSpecificBehavior::test_handles_connection_error
  - tests/unit/plugins/llm/test_azure_multi_query_profiling.py: TestLoadScenarios::test_rate_limit_error_handling, TestRowAtomicity::test_row_atomicity_high_failure_rate, TestRowAtomicity::test_row_atomicity_under_capacity_errors
  - tests/unit/plugins/llm/test_transform.py::TestMultiQuerySequentialRetryBehavior::test_retryable_error_returns_error_result_not_raises
- [ ] **B — field_mapper composer_hint > 280 chars**.
  tests/unit/contracts/test_plugin_assistance_coverage.py::test_builtin_plugin_publishes_discovery_hints[transform-field_mapper]
  (hint reworked on this branch; trim to <=280).
- [ ] **H — plugin sink/source/transform behavior**.
  - tests/unit/plugins/sinks/test_sink_bug_fixes.py::TestAzureBlobSinkFieldValidation::test_csv_extra_fields_rejected_in_fixed_mode
  - tests/unit/plugins/transforms/azure/test_blob_source.py::TestAzureBlobSourceCSV::test_csv_without_header
  - tests/integration/web/test_composer_tools.py::test_post_call_hints_envelope_populated_for_hinted_plugins[sink-database-...unique constraint]
- [ ] **E — elspeth_lints gate drift (source_file_hash / allowlist)**. Co-land
  source-hash refresh for edited plugin files. Baseline/HMAC re-sign is
  OPERATOR-OWNED — do not blind-regen.
  - tests/unit/elspeth_lints/test_allowlist_loader_unification.py::test_baseline_capture_is_self_consistent
  - tests/unit/elspeth_lints/test_audit_evidence_rules.py::test_audit_evidence_json_mode_succeeds_on_current_codebase
  - tests/unit/elspeth_lints/test_immutability_rules.py::test_existing_yaml_loads_with_core_loader
  - tests/unit/elspeth_lints/test_trust_tier_model_rule.py::TestR1SourceRegressions::test_source_boundary_non_r5_findings_are_site_allowlisted
- [ ] **G — discipline guard tests** (verify whether this branch tripped them).
  - tests/unit/test_mock_discipline_baseline.py::test_unspecced_mock_baseline_does_not_increase
  - tests/unit/test_no_hasattr_branching.py::test_hasattr_in_tests_is_limited_to_direct_surface_assertions

### Pre-existing (NOT this branch — triage: real bug vs environmental)

- [ ] **C — web integration ERRORs (23)**. Root cause: `KeyError: 'source'` at
  tests/integration/web/conftest.py:401 — `composition_state.to_dict()` lacks
  `source`. Single shared fixture fault. Files: test_audit_readiness_routes.py,
  test_completion_flow_e2e.py, test_shareable_reviews_routes.py,
  test_yaml_export_audit_event.py.
- [ ] **D — composer guided integration FAILs (~18)**.
  tests/integration/web/composer/guided/* (default_guided, error_paths,
  get_guided, progressive_disclosure, respond, step_chat, step_handlers).
- [ ] **F — core config FAILs (5)**. tests/unit/core/test_config.py
  (collection probes, env-var expansion, secret-field fingerprint).
- [ ] **I — audit/ADR durability FAILs**.
  - tests/integration/audit/test_exporter_batch_queries.py::TestExporterBatchQueryIntegrity::test_export_run_is_isolated_from_sibling_run_records
  - tests/integration/test_adr_019_sweep_durability.py::test_realtime_invariant_crash_finalizes_failed_and_preserves_witnesses[I1c], [I3]
- [ ] **J — interpretation / execute-pipeline FAILs** (likely same fixture
  family as C/D).
  - tests/integration/web/test_interpretation_opt_out_audit.py::test_opted_out_session_still_records_surface_specific_rows
  - tests/integration/web/test_execute_pipeline.py: TestEndToEndPipelineExecution::test_csv_passthrough_csv, TestGateRoutedPipelineExecution::test_gate_routed_pipeline_classifies_as_completed_via_api
  - tests/integration/web/test_audit_readiness_routes.py::test_secrets_row_surfaces_disallowed_secret_ref_from_real_validate_pipeline
