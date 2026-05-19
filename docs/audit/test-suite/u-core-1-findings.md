# U-CORE-1 — Synthesised Findings

**Scope:** 32 files in `tests/unit/core/landscape/` covering recording, repositories, DB ops, schema, models, serialization, querying, factory/journal/lineage/exporter.
**Method:** 5 specialist agents in parallel, each with a different lens.
**Date:** 2026-05-06.

## Verdict

**Health: Mixed.** Strong Tier-1 crash coverage in compatibility-guards, model-loaders, and run-lifecycle. Real in-memory SQLite, no fake-clock hacks, no test-ordering dependencies. **But:** two structural defects across the chunk seriously undermine its value as audit-DB coverage:

1. **Hash-without-binding theatre on the legal-record signature itself** — `test_exporter.py::_sign_record` checks shape/length/determinism/inequality, never recomputes the HMAC against the test key. The exporter signature is the integrity primitive for legal-record export.
2. **Audit silence** — recording tests verify return values from recorder facades but rarely read the persisted bytes back through the public query API. A regression that returns "looks-recorded" results without writing the DB would pass.

Per-lens verdicts: Mixed (anti-patterns), Some-leaning-Pervasive in three concentrated zones (theatre), Sound-with-policy-violations (Python smells), Strong-with-three-critical-gaps (scenario), Strong-with-three-critical-gaps (SUT coverage).

## Convergent findings (≥2 agents agree — high confidence)

### CONV-1 — Hash-without-binding (most dangerous in this chunk) — 2/5 agents at multiple sites

| Site | Issue |
|---|---|
| `test_call_recording.py:147-204` | `assert call.request_hash is not None`; never compared to `stable_hash(request_data)` |
| `test_token_recording.py:111-120, 170-178` | `assert source_data_hash is not None`; presence not equality |
| `test_exporter.py:386-405` | `_sign_record` tested for shape/length/determinism/inequality; **never compared to `hmac.new(b"test-key", canonical_json(record), sha256).hexdigest()`** |
| `test_recorder_store_payload.py:32-38` | Empty-bytes hash checked for length, not for the well-known `e3b0c44...b855` constant |
| `test_validation_error_noncanonical.py:165-181` | `repr_hash` helper checked for inequality on distinct inputs (SHA-256 doesn't trivially collide — uninteresting) |

**Recommendation:** systematic sweep replacing `is not None` / `len == 64` hash assertions with `== stable_hash(input)` (or `== hmac.new(...).hexdigest()` for signatures). Add a shared fixture asserting `(request_hash, response_hash) == (stable_hash(req), stable_hash(resp))`.

### CONV-2 — `hasattr()` and private-attribute reach-in (CLAUDE.md ban) — 2/5 agents

| Site | Issue |
|---|---|
| `test_query_methods.py:63-64` | `assert not hasattr(factory.query._ops, "execute_insert")` — banned |
| `test_execution_repository.py:395` | `if hasattr(stmt, "is_insert")` in mock callback — banned |
| `test_data_flow_repository.py:1005, 1053, 1091` | Same `hasattr(stmt, "is_insert")` pattern, three sites |
| `test_node_state_recording.py:1814` | `factory.execution._payload_store = mock_store` — private attr injection |
| `test_database_compatibility_guards.py:17-26` | `LandscapeDB.__new__(...)` + manual `_passphrase`, `_engine`, `_journal` assignment — bypasses public API entirely |
| `test_journal.py` 13 sites (515, 560, 575, 589, 606, 619, 632, 654, 675, 687, 723, 742, 763) | `journal._payload_store = Mock()` |

**Recommendation:** **suite-wide CI rule** — extend `enforce_tier_model.py` (or add `enforce_no_hasattr_in_tests.py`) to detect `hasattr(...)` and `_<private>` attribute assignment in test files. Combined with the U-CONTRACTS-1 violations, one CI rule prevents recurrence across the whole suite.

### CONV-3 — Spec-less `Mock()`/`MagicMock()` — 2/5 agents

`test_lineage.py:48-65`, `test_journal.py:54`, `test_database_compatibility_guards.py:289, 406, 569`, `test_call_recording.py:990`, plus 13 `journal._payload_store = Mock()` sites.

**Recommendation:** mechanical `spec=` retrofit (same pattern as U-CONTRACTS-1).

### CONV-4 — `test_data_flow_nan_rejection.py` is structural-AST theatre — 2/5 agents

The file walks `data_flow_repository.py` AST and asserts every `json.dumps` keyword has `allow_nan=False`. Brittle to any benign refactor (helper wrapper, alias) and **duplicates what `enforce_tier_model.py` could enforce statically**.

**Recommendation:** delete the file; replace with a behavioural test that passes `float("nan")` through every public recorder method and asserts `AuditIntegrityError`.

### CONV-5 — Dataclass-machinery tautology cluster — 3/5 agents

`test_models_mutation_gaps.py:32-66, 72`, `test_schema.py:69-118`, `test_row_data.py:13-19, :50`, `test_preflight_recording.py:202-218`, `test_lineage.py:389-404`, `test_factory.py:30, :50, :56, :63-65`. All construct a `@dataclass`, set fields, read them back, assert they match — testing Python's `@dataclass`, not ELSPETH.

**Recommendation:** delete the surveyed sites (~50–80 LOC); keep `test_models_enums.py` Tier-1 tests and the genuine constraint tests.

### CONV-6 — `test_query_methods.py` redundant `single_X` and `scoped_to_run_X` series — 2/5 agents

Each entity (rows/tokens/states/calls/events/outcomes/batches/operations) has `returns_X / empty_X / single_X / scoped_to_run_X` tests. The `single_X` adds nothing over `returns_X_ordered`. The `scoped_to_run_X` is duplicated by `test_where_exactness_consolidated.py`.

**Recommendation:** delete `test_single_X` series; delete `test_*_scoped_to_run` where consolidated coverage exists. ~25-30 deletable test bodies in this one file.

### CONV-7 — `test_journal.py` `_is_write_statement` parametrize candidate — 2/5 agents

8 one-line tests (INSERT/UPDATE/DELETE/REPLACE/SELECT/CREATE/case/whitespace) that should be one parametrize.

## Single-lens findings worth surfacing

### SOLO-1 — `pytest.raises((SchemaCompatibilityError, Exception))` is logically vacuous (Critical)

Found by **python-code-reviewer**. `test_database_sqlcipher.py:90, 111`. `Exception` subsumes `SchemaCompatibilityError`; tuple equivalent to `pytest.raises(Exception)`. Any error including `AssertionError` from a buggy fixture passes.

### SOLO-2 — `test_journal.py:193-203` pins a parser bug as the contract (Critical)

Found by **qa-analyst**. `test_no_column_list_parses_values_as_columns` asserts `table == "calls values"` and `cols == ["1", "2"]`. The test pins broken parser behaviour as the spec — will fail when the bug is fixed. Wrong direction of regression detection.

### SOLO-3 — `test_factory.py::test_plugin_audit_writer_is_adapter` is type-only (High)

Found by **python-code-reviewer**. Asserts `isinstance(writer, _PluginAuditWriterAdapter)` against a private symbol. Tests the class label, not what the adapter does.

### SOLO-4 — `test_error_recording.py:586-600, 746-760` timestamp ordering flake (High)

Found by **python-code-reviewer**. `assert timestamps == sorted(timestamps)` on rapid sequential inserts. Fast hardware can produce timestamp collisions. Use `freezegun` or assert uniqueness separately from ordering.

### SOLO-5 — `test_recorder_store_payload.py` is in the wrong directory (Minor)

Found by **test-suite-reviewer**. File tests `FilesystemPayloadStore` directly, not Landscape recorder. Belongs in `tests/unit/core/payload/`. Move it.

### SOLO-6 — `test_run_lifecycle_repository.py:497` accepts empty status (Suspicious)

Found by **pr-test-analyzer**. `test_empty_status_accepted_and_round_trips` accepts an empty-string status. If empty isn't a valid `RunStatus` enum value, this contradicts the Tier-1 "no coercion" rule. **Verify against the enum definition.**

## Critical production-code gaps (from coverage-gap-analyst)

| Gap | Severity | Why it matters |
|---|---|---|
| `DataFlowRepository.link_validation_error_to_row` — **zero tests** anywhere | **Critical** | Quarantine lineage exactness guarantee per schema.py epoch-4 comment. Without tests, FK can be silently NULL or relinked, destroying `explain()` for quarantined tokens. |
| `_REQUIRED_COMPOSITE_FOREIGN_KEYS` has 12 entries; only 1 (`transform_errors`) is tested | **Critical** | A production DB missing e.g. `token_outcomes(token_id, run_id) → tokens(token_id, run_id)` passes schema validation and allows cross-run outcome contamination. |
| `_validate_token_row_ownership` — never directly tested | **Critical** | Prevents cross-row lineage corruption. A token with a swapped `row_id` produces a valid-looking audit trail attributing wrong source data to a terminal decision. Attributability test fails silently. |
| ADR-019 `sweep_deferred_invariants_or_crash`, `find_orphaned_transient_parents`, `find_orphaned_batch_consumptions` — zero unit tests | **Critical** | Run-end invariant enforcement. SQL regression silently passes orphaned fork/expand parents to storage. |
| **Attributability round-trip is missing** — no test records Source→Transform→Sink and asserts `explain(run_id, token_id)` reproduces the chain | High | The literal "attributability test" from CLAUDE.md is not exercised end-to-end at unit level. |
| Hash↔payload binding survival not explicitly asserted at recorder level | High | Only `test_reproducibility.py` indirectly probes via grade demotion. |
| Closed-set terminal-state coverage incomplete (`EXPANDED`, `COALESCED`, `CONSUMED_IN_BATCH` constraint-checked but never asserted as *only* terminal) | High | "Every row reaches exactly one terminal state" claim not exhaustively verified. |
| `read_only_connection()` unsupported-backend `RuntimeError` not tested in landscape | Medium | MCP-only coverage; if unsupported backend slips config validation, MCP gets writable connection. |

## Top deletion candidates (consensus order)

| # | Target | Lines | Confidence |
|---|---|---|---|
| 1 | `test_data_flow_nan_rejection.py` (entire file) | ~50 | High |
| 2 | `test_query_methods.py` `test_single_X` + `test_*_scoped_to_run` series | ~150 | High |
| 3 | `test_models_mutation_gaps.py:32-66` (`TestRunDataclass`) | ~40 | High |
| 4 | `test_journal.py:69-90` 8 `_is_write_statement` one-liners | ~30 | High (parametrize, not delete) |
| 5 | `test_journal.py:102-127` pass-through identity tests | ~25 | High |
| 6 | `test_schema.py:33-41, 69-118` enum-value + dataclass tautologies | ~55 | High |
| 7 | `test_row_data.py:13-19, :50` enum-value + `frozen=True` tests | ~12 | High |
| 8 | `test_preflight_recording.py:202-218` `TestPreflightResult` | ~17 | High |
| 9 | `test_factory.py:30, :50, :56, :63-65` four tests | ~25 | High |
| 10 | `test_lineage.py:389-404` `test_has_expected_fields` | ~16 | High |
| 11 | `test_database_ops.py:85-87, 147-156, 189-196` cosmetic tests | ~35 | Medium |
| 12 | `test_validation_error_noncanonical.py:165-181` `test_repr_hash_helper` | ~17 | Medium |
| 13 | `test_call_recording.py:67-72` `test_single_allocation` | ~6 | Medium |
| 14 | `test_journal.py:193-203` parser-bug snapshot pair | ~11 | Medium (rewrite, not delete) |

**Total deletable: ~470-500 lines / ~50-60 test bodies.**

## Top "add immediately" candidates

1. **`link_validation_error_to_row` test class** (3 Tier-1 crash branches).
2. **Composite FK exhaustive parametrize** — one test per `_REQUIRED_COMPOSITE_FOREIGN_KEYS` entry.
3. **`_validate_token_row_ownership` direct unit test** with mismatched token_id/row_id pairs.
4. **ADR-019 sweep unit tests** — orphan parent + no child → assert `AuditIntegrityError`.
5. **Attributability round-trip** — record Source→Transform→Sink, call `explain(...)`, assert reproduction.
6. **Hash-binding sweep** — retrofit ~30 hash assertions to `== stable_hash(input)`.
7. **Exporter HMAC verification test** — compare `_sign_record` output to `hmac.new(b"test-key", canonical_json(record), sha256).hexdigest()`.

## Notable strengths (preserve)

- `test_database_compatibility_guards.py` — thorough ADR-019 migration detection with real DDL.
- `test_data_flow_repository.py` — ADR-019 cross-table invariants (I1c, I3) with actual DB state.
- `test_execution_repository.py` — exhaustive `complete_node_state` write-side guards.
- `test_lineage.py::TestExplainTier1Corruption` and `TestExplainParentIntegrity` — full Tier-1 corruption detection.
- `test_call_recording.py::test_payload_integrity_error_raises_audit_integrity` — exemplary template.
- `test_journal.py::test_integrity_error_always_crashes_as_audit_violation` — exemplary `OperationalError` vs `IntegrityError` distinction.

## Filed filigree issues

| ID | Title | Priority | Labels |
|---|---|---|---|
| `elspeth-297dafdf47` | DataFlowRepository.link_validation_error_to_row — zero tests, quarantine lineage exactness unverified | P0 bug | audit-integrity, test-gap, from-test-audit |
| `elspeth-499100db05` | _REQUIRED_COMPOSITE_FOREIGN_KEYS — 11 of 12 entries untested, cross-run audit contamination possible | P0 bug | audit-integrity, test-gap, from-test-audit |
| `elspeth-82c7c028a8` | DataFlowRepository._validate_token_row_ownership — never directly tested, cross-row lineage corruption undetectable | P0 bug | audit-integrity, test-gap, from-test-audit |
| `elspeth-f6f50e9394` | ADR-019 sweep_deferred_invariants_or_crash — zero unit tests for run-end invariant enforcement | P0 bug | audit-integrity, test-gap, from-test-audit, adr-019 |

The other findings (deletable test bodies, hash-without-binding sweep, scenario gaps, attributability round-trip, terminal-state coverage) are not yet filed — they are test-quality cleanups or chunk-wide refactors, best handled as sweep PRs rather than per-test issues. The hash-without-binding and `hasattr` patterns are suite-wide and warrant CI rules (recorded in `docs/audit/test-suite/README.md`).

## Out-of-scope observations (production concerns)

1. `test_journal.py:200-203` pins a real parser bug — production-side fix needed (file as separate observation).
2. `test_run_lifecycle_repository.py:497` accepts empty status — verify against enum, may indicate Tier-1 rule violation in production.
3. The schema `_validate_schema` partial-index predicate is not actually tested against a live DB (only mocked inspectors). DDL regression risk persists undetected because the project's DB-deletion-not-Alembic policy means silent regressions accumulate forever.
