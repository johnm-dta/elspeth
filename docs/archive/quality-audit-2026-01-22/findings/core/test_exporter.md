# Test Quality Review: test_exporter.py

## Summary
Test suite covers basic export functionality and HMAC signing determinism, but has critical gaps in lineage verification, data integrity validation, corrupt data handling, and operation failures. Missing property-based tests for hash chain integrity. Several tests check structure but not semantic correctness.

## Poorly Constructed Tests

### Test: test_exporter_run_metadata_has_required_fields (line 61)
**Issue**: Field presence check without value validation
**Evidence**: Tests only `assert field in run_record`, never validates that values are correct or non-None when they should be populated
**Fix**: Add assertions for value correctness. For a completed run, `started_at` and `completed_at` must be non-None ISO timestamps, `status` must be "completed", `canonical_version` must match what was passed to `begin_run()`
**Priority**: P2

### Test: test_exporter_row_has_required_fields (line 96)
**Issue**: Same pattern - structure check without semantic validation
**Evidence**: Only checks field presence, doesn't verify `source_data_hash` is a valid hash, `row_index` matches expected value (0), or `source_node_id` references a real node
**Fix**: Verify hash format (64 hex chars for SHA256), verify referential integrity (node exists), verify row_index correctness
**Priority**: P2

### Test: test_exporter_extracts_edges (line 147)
**Issue**: Hardcoded assertion for edge count without explicit setup context
**Evidence**: `assert len(edge_records) == 1` - why exactly 1? Test doesn't document this is a simple source→sink pipeline
**Fix**: Add docstring explaining pipeline topology: "Single edge from source to sink", or use `assert len(edge_records) >= 1` and validate the specific edge
**Priority**: P3

### Test: test_exporter_signatures_are_deterministic (line 553)
**Issue**: Excludes manifest from determinism check with special-case handling
**Evidence**: `if r1["record_type"] != "manifest"` - why should manifest signatures differ? The `exported_at` timestamp breaks determinism
**Fix**: Either remove `exported_at` from manifest (use external metadata), or accept that manifest signature cannot be deterministic and document this in test name/docstring
**Priority**: P1 - This is an audit integrity question. If manifests can't be compared across exports, how do we detect tampering?

### Test: test_exporter_final_hash_deterministic_with_multiple_records (line 592)
**Issue**: Creates complex setup but doesn't verify all record types are actually exported
**Evidence**: Creates nodes, edges, rows, tokens, states, routing events, batches - but only checks `final_hash`, doesn't verify every record type made it into the export
**Fix**: Add assertions counting each record type before checking hash. This test should verify determinism AND completeness
**Priority**: P2

### Test: test_exporter_record_order_is_stable (line 690)
**Issue**: Only tests node ordering, ignores all other record types
**Evidence**: `node_ids = [r["node_id"] for r in records if r["record_type"] == "node"]` - what about edges, tokens, states, etc.?
**Fix**: Check ordering stability for all record types, especially nested ones (tokens within rows, states within tokens)
**Priority**: P2

## Missing Critical Tests

### Missing: External call records export
**Issue**: Exporter code includes external call export (lines 316-328 in exporter.py), but no test verifies this
**Evidence**: `test_exporter_extracts_*` covers edges, tokens, states, artifacts, batches, routing events - but NOT calls
**Fix**: Add `test_exporter_extracts_calls()` that creates an external call record via recorder and verifies export includes it with correct fields
**Priority**: P1 - External calls are critical for audit trail ("Full request AND response recorded" per CLAUDE.md)

### Missing: Lineage completeness verification
**Issue**: No test verifies exported data enables complete lineage reconstruction
**Evidence**: Tests check individual record types exist, but never verify "can I reconstruct the path from source row to sink artifact?"
**Fix**: Add `test_exporter_lineage_reconstruction()` that exports a multi-node pipeline and verifies: (1) every token has path to source row, (2) every artifact has path to source data, (3) no orphaned records
**Priority**: P0 - This is the Attributability Test from CLAUDE.md: "For any output, the system must prove complete lineage"

### Missing: Hash integrity validation
**Issue**: Tests check hashes exist (`content_hash == "abc123"`), never verify hashes are correct for the data
**Evidence**: `test_exporter_extracts_artifacts` uses hardcoded hash "abc123" - should compute expected hash from input data and verify match
**Fix**: Add test that creates row with known data, processes through transform, exports, and verifies `source_data_hash` matches `canonical_json(source_data)` hash
**Priority**: P1 - "Hashes survive payload deletion - integrity is always verifiable"

### Missing: Corrupt database handling
**Issue**: No tests for Tier 1 trust violations (corrupt data from OUR database)
**Evidence**: Only one error test (`test_exporter_raises_for_missing_run`), which is a valid "no data" case, not a "bad data" case
**Fix**: Add tests for: (1) NULL in NOT NULL column, (2) Invalid enum value in status field, (3) Broken foreign key reference. All should crash, not coerce
**Priority**: P1 - "Bad data in the audit trail = crash immediately" (Three-Tier Trust Model)

### Missing: export_run_grouped() tests
**Issue**: Method exists in exporter (line 368-398), has different return type and behavior, but no tests
**Evidence**: All tests use `export_run()`, none use `export_run_grouped()`
**Fix**: Add test class `TestLandscapeExporterGrouped` with tests for: (1) correct grouping by record_type, (2) deterministic key order, (3) signing works with grouped export
**Priority**: P2 - Documented use case: "useful for CSV export"

### Missing: Manifest validation test
**Issue**: Tests check manifest fields exist, never validate the hash chain is correct
**Evidence**: `test_exporter_manifest_contains_final_hash` checks field presence, doesn't verify `final_hash` is actually SHA256(signature1 || signature2 || ... || signatureN)
**Fix**: Add test that manually computes expected final hash by concatenating all signatures in order, compare to manifest value
**Priority**: P1 - Hash chain is the tamper-evidence mechanism

### Missing: NodeStateFailed export test
**Issue**: Exporter handles NodeStateOpen, NodeStatePending, NodeStateCompleted, NodeStateFailed (line 284-299), but tests only create completed states
**Evidence**: All `complete_node_state()` calls use `status="completed"`, none use `status="failed"` or test OPEN/PENDING states
**Fix**: Add tests for all state discriminator variants: `test_exporter_extracts_open_state`, `test_exporter_extracts_pending_state`, `test_exporter_extracts_failed_state`
**Priority**: P2 - Quarantined/failed rows are part of audit trail

### Missing: Property-based test for signature stability
**Issue**: Determinism tests use example-based approach (run twice, compare), not property-based
**Evidence**: Would benefit from Hypothesis generating random pipeline structures and verifying export determinism
**Fix**: Add `@given(st.integers(1, 10))` test that generates N nodes/edges/rows and verifies final_hash is identical across multiple exports
**Priority**: P3 - Nice-to-have, but current tests provide decent coverage

### Missing: Empty run edge cases
**Issue**: All tests create at least one row/node. What about: (1) Run with nodes but no rows, (2) Run with rows but no tokens (all quarantined at source?)
**Evidence**: `populated_db` fixture always creates a row. Never tests "valid run structure, zero data processed"
**Fix**: Add `test_exporter_handles_empty_pipeline()` - run with nodes registered but no data processed
**Priority**: P3 - Edge case, but audit trail should handle it

## Misclassified Tests

### Test: test_exporter_final_hash_deterministic_with_multiple_records (line 592)
**Issue**: This is an integration test, not a unit test
**Evidence**: Creates full pipeline with multiple nodes, rows, tokens, states, routing events, and batches - exercises entire recorder and database stack
**Fix**: Move to `tests/integration/test_exporter_determinism.py` or mark with `@pytest.mark.integration`. Keep a simpler unit-level determinism test in this file (single node, single row)
**Priority**: P3 - Test is correct, just wrong location

## Infrastructure Gaps

### Gap: Fixture reuse for complex pipelines
**Issue**: Tests `test_exporter_extracts_edges`, `test_exporter_extracts_tokens`, etc. each rebuild similar pipeline structures inline
**Evidence**: Lines 147-188 (edges), 190-220 (tokens), 222-265 (states) - each creates db, recorder, run, nodes, rows independently
**Fix**: Create parametrized fixtures: `@pytest.fixture(params=["simple", "fork", "aggregation"])` that yields different pipeline topologies. Tests can request the topology they need.
**Priority**: P2 - Reduces duplication, makes tests more maintainable

### Gap: No helper for "verify complete record structure"
**Issue**: Tests manually list required fields and loop to check presence (lines 69-79, 104-113)
**Evidence**: Same pattern repeated 3+ times. Should be a helper function.
**Fix**: Add `def assert_has_fields(record: dict, required: list[str]) -> None` helper at module level or in conftest
**Priority**: P3 - Code quality issue, not a correctness issue

### Gap: No shared signing key constant
**Issue**: Tests use different ad-hoc signing keys: `b"test-key-for-hmac"`, `b"determinism-test-key"`, `b"key-one"`, `b"key-two"`
**Evidence**: Lines 495, 507, 522, 544, 556, 569, 584, 677
**Fix**: Define module-level `TEST_SIGNING_KEY = b"test-hmac-key"` and use consistently. Only tests for "different keys produce different signatures" should use alternate keys
**Priority**: P3 - Clarity and consistency

### Gap: Missing conftest.py with shared landscape fixtures
**Issue**: `populated_db` fixture is defined in this file, but likely useful across all landscape tests
**Evidence**: Other test files probably need similar "db with basic run" setup
**Fix**: Check if `tests/core/landscape/conftest.py` exists, move shared fixtures there. If it doesn't exist, create it
**Priority**: P2 - Reduces duplication across test suite

### Gap: No test isolation verification
**Issue**: All tests use in-memory databases, but no explicit verification that tests don't interfere with each other
**Evidence**: If a test fails to clean up and leaves database state, would next test notice?
**Fix**: Not a problem for in-memory databases (each fixture creates new db), but add a paranoid test: `test_fixtures_are_isolated()` that verifies `populated_db` called twice returns different db instances
**Priority**: P3 - Paranoia, not a real risk with current setup

## Positive Observations

- **HMAC signing determinism tests are thorough**: Tests cover same-key determinism, different-key differences, manifest structure - this is critical for legal-grade exports
- **Good test organization**: Classes group related functionality clearly (metadata, rows, nodes, errors, complex scenarios, signing)
- **Comprehensive record type coverage**: Tests verify export of edges, tokens, node_states, artifacts, batches, batch_members, routing_events, token_parents - excellent breadth
- **Proper use of in-memory databases**: Fast test execution, no cleanup needed, good isolation
- **Error handling tested**: `test_exporter_raises_for_missing_run` and `test_exporter_raises_when_sign_without_key` verify expected failures

## Priority Summary

- **P0**: 1 issue - Lineage completeness verification (core audit requirement)
- **P1**: 5 issues - Manifest determinism, external calls, hash integrity, corrupt data handling, manifest hash validation (audit integrity)
- **P2**: 7 issues - Field value validation, completeness checks, fixture reuse, grouped export, state variants, infrastructure
- **P3**: 7 issues - Documentation, edge cases, test classification, minor refactoring

## Recommended Next Steps

1. **Immediate (P0)**: Add `test_exporter_lineage_reconstruction()` to verify exported data supports complete source-to-sink tracing
2. **High priority (P1)**: Add missing tests for external calls, corrupt data crashes, manifest hash chain validation
3. **Medium priority (P2)**: Refactor fixture structure, add `export_run_grouped()` tests, test all NodeState variants
4. **Low priority (P3)**: Improve documentation, extract helper functions, consider property-based tests
