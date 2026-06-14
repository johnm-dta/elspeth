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
- [x] **B — field_mapper composer_hint > 280 chars**. DONE (commit 03268a124): split hint + re-pinned source_file_hash.
- [x] **H — plugin sink/source/transform behavior**. DONE (commit 345156ad7): B4.4 headerless-CSV reject split; B3.2 azure extra-field divert.
- [x] **G — discipline guards**. DONE: hasattr→getattr/isinstance (345156ad7); mock ratchet 2630→2652 with attribution (f96a860f1).
- [x] **C/D/J — web/composer source→sources migration** (PRE-EXISTING, not this branch). DONE (commit 27324a773): 45 tests; CompositionState.source→.sources across fixtures/attrs/response bodies.
- [x] **F — core config source→sources (ADR-025)** (PRE-EXISTING). DONE (commit e8d66da00): 5 tests.
- [x] **I — audit/ADR durability** (PRE-EXISTING, release/0.6.0 multi-worker). DONE (commit 4f1a3f9b8): epoch-21 create_row identity + check_coordination_latch mock.

### OPERATOR-OWNED GATE SET — NOT fixed here (require HMAC key / tier judgment)

> Per project doctrine: autoroll source-hashes OK, NEVER baseline/allowlist
> snapshots; operator holds the red-gate deliberately. Reconciled at merge.

- [ ] **#3 trust-tier R1/R6 fingerprint drift** — this branch's source edits
  (dataverse B3.1 etc.) drifted tier-model allowlist fingerprints
  (`dataverse.py:R6:DataverseSource:load:fp=…`). Owed: operator fp re-pin/justify at merge.
- [ ] **baseline_capture_is_self_consistent** — pre-existing operator HMAC re-pin (known red).
- [ ] **immutability `8 vs 9` live entries** — allowlist-snapshot drift; operator-owned.
- [ ] **FollowerSeatDeadError undecorated** (audit_evidence gate) — needs a
  tier-1/tier-2 classification in `contracts/errors.py`; follower-epic + tier-model
  decision (pre-existing on release/0.6.0). NOT this branch.

## Final tally

Run2 baseline: **52 failed + 23 errors**. Fixed in commits f98195b10,
03268a124, 345156ad7, f96a860f1, 27324a773, e8d66da00, 4f1a3f9b8:
- Branch-caused: A (8 tests, the hang), B (field_mapper hint), H (azure CSV
  reject/divert), G (hasattr + mock ratchet).
- Pre-existing test-infra (source→sources / ADR-025 / epoch-21), fixed for a
  green suite: C+D+J (45), F (5), I (3).

Remaining = the OPERATOR-OWNED GATE SET only (4 tests above). These require the
operator HMAC key and/or a tier-model classification decision and are
reconciled at merge — deliberately not touched.
