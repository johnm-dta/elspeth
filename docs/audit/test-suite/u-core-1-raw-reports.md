# U-CORE-1 — Raw Agent Reports

Verbatim outputs from the 5 specialist agents that reviewed U-CORE-1 on 2026-05-06.
Preserved in full as the source of truth backing the synthesis in `u-core-1-findings.md`.

Agent IDs (not durable across sessions):
- test-suite-reviewer: `aebda6153674db858`
- quality-assurance-analyst: `ac826a1e2f909c358`
- python-code-reviewer: `a264ae33f507e7736`
- pr-test-analyzer: `affbe7e5c335330ad`
- coverage-gap-analyst: `a1612fd03e555531a`

---

## 1. ordis-quality-engineering:test-suite-reviewer

### Confidence Assessment
High for line-cited findings (direct read evidence). Medium for cross-file pattern generalisation (partial reads on 4 large files). No manufactured findings.

### Risk Assessment
Findings are test quality defects, not production bugs. The most actionable are the frozen-implementation and structural-analysis findings — they make the test suite brittle to legitimate refactors. Tautology findings are low-risk to ignore but represent dead weight.

### Information Gaps
`test_token_recording.py`, `test_node_state_recording.py`, and `test_query_methods.py` were partially read (first 500–1000 lines of 1600–1888). Additional minor issues may exist in unread sections.

### Caveats
ELSPETH's offensive-programming stance means some patterns that look defensive in a normal codebase (e.g., checking for None before asserting) are test-design signals, not style preferences.

### Summary

The 32-file audit subgroup has solid Tier-1 crash coverage (tamper-detect paths, FK constraint enforcement, cross-run contamination guards) and no sleepy assertions or test interdependence. The dominant anti-pattern is frozen-implementation coupling: five files wire tests directly to private attributes, private methods, or implementation internals rather than the public contract, making them brittle to legitimate refactors. A smaller cluster of tautology tests re-states Python `@dataclass` and enum machinery without exercising any ELSPETH-specific behaviour.

### Findings

| File:Line | Category | Severity | Rationale | Recommendation |
|---|---|---|---|---|
| `test_node_state_recording.py:1814` | Frozen-implementation | **Major** | `factory.execution._payload_store = mock_store` monkey-patches a private attribute on a live production object mid-test; breaks on any internal rename | Inject the mock store through the public constructor or a test-helper factory; never reach into `_` attributes |
| `test_database_compatibility_guards.py:17–26` | Frozen-implementation | **Major** | `LandscapeDB.__new__(LandscapeDB)` + manual `instance._passphrase`, `instance._engine`, `instance._journal` assignment bypasses the constructor entirely; coupled to init ordering and private attr names | Expose a factory parameter or `connect_string` override for tests; test the public API, not bypass it |
| `test_data_flow_nan_rejection.py` (whole file) | Structural analysis | **Major** | AST-walks `data_flow_repository.py` source and asserts every `json.dumps` has `allow_nan` in keywords; breaks on any benign refactor (helper wrapper, import alias, file rename) while adding no coverage CI doesn't already provide via `enforce_tier_model.py` | Delete; enforce statically in CI or write a behavioural test that passes `float("nan")` and asserts `AuditIntegrityError` |
| `test_query_methods.py:63–64` | Frozen-implementation + banned `hasattr` | **Major** | `assert not hasattr(factory.query._ops, "execute_insert")` tests structural wiring via a banned construct (`hasattr` is unconditionally prohibited in CLAUDE.md) on a private attribute | Delete the assertion; if wiring matters, test observable behaviour (wrong-method call raises `AuditIntegrityError`) |
| `test_token_recording.py:1495` | Frozen-implementation | **Minor** | `elspeth.core.landscape._helpers.generate_id` and `now` imported from a private module to construct a schema FK collision scenario; private symbol import couples test to implementation | Expose a test-helper in `tests/fixtures/landscape` that generates a collision-scenario token pair |
| `test_journal.py:69–91` | Frozen-implementation | **Minor** | `LandscapeJournal._is_write_statement(...)` called directly across an entire test class; private method tests are implementation theatre | Test journal behaviour through `record()`; if `_is_write_statement` must be tested, extract it to a module-level function with a public name |
| `test_models_mutation_gaps.py:32–85` | Tautology | **Minor** | Survivors after documented pruning: `test_status_is_required_run_status_enum` asserts `isinstance(run.status, RunStatus)` after constructing `Run(status=RunStatus.RUNNING)` — stores what you pass in; `test_registered_at_is_required` tests Python's own required-field `TypeError` | Delete both; they verify `@dataclass` machinery, not ELSPETH behaviour |
| `test_schema.py:33–41` | Tautology | **Minor** | `test_determinism_values` computes `{d.value for d in Determinism}` and asserts equality to a hardcoded set — re-states the enum definition | Delete; a typo in the enum value would require updating both definition and test, yielding zero additional confidence |
| `test_row_data.py:13–19` | Tautology | **Minor** | `test_row_data_state_values` asserts each `RowDataState` member's `.value` matches a string literal — same pattern as `test_schema.py:33` | Delete |
| `test_token_recording.py:111–120` | Weak assertion | **Minor** | `test_stores_source_data_hash` asserts `row.source_data_hash is not None`; does not assert the hash matches `stable_hash(source_data)` | Assert the expected hash value; presence checks don't detect a hash computed from wrong data |
| `test_preflight_recording.py:21–27` | Fixture overreach | **Minor** | `factory` fixture yields a `(fac, run_id, db)` tuple; every test must destructure `fac, run_id, db = factory` — non-idiomatic and fragile if tuple arity changes | Yield a named dataclass or `SimpleNamespace` with `.fac`, `.run_id`, `.db` attributes |
| `test_query_methods.py:283–300` | Branching test logic | **Minor** | `test_never_stored_when_no_payload_ref` contains `if row.source_data_ref is None:` inside the test body — two branches, two scenarios, one test | Split into `test_never_stored_when_no_payload_ref` and `test_stored_when_payload_ref_present` |

### Cross-File Patterns

- **Private attribute reach-in is systemic.** Four separate files (`test_node_state_recording.py`, `test_database_compatibility_guards.py`, `test_query_methods.py`, `test_token_recording.py`) bypass the public API by directly reading or writing `_`-prefixed attributes. Each coupling site is an independent refactor hazard; the pattern should be resolved at the fixture layer, not file by file.
- **Enum-value tautologies cluster in model test files.** `test_schema.py`, `test_row_data.py`, and the survivors in `test_models_mutation_gaps.py` all follow the same pattern: assert that `MyEnum.MEMBER.value == "string_literal"`. These tests add maintenance cost (every rename requires two edits) without catching bugs that type checking and serialisation round-trips wouldn't catch better.
- **Hash assertions stop at presence.** `test_token_recording.py:111–120` and `test_token_recording.py:170–178` verify `is not None` rather than verifying the hash equals the expected canonical value. Given the audit trail's legal-record status and the explicit attributability test in CLAUDE.md (`explain(recorder, run_id, token_id)` must prove complete lineage), presence-only checks undertest the most critical invariant in the subgroup.
- **`test_recorder_store_payload.py` tests the wrong layer.** Its name implies it covers `Recorder.store_payload()` behaviour; its content tests `FilesystemPayloadStore` directly. The file belongs in a payload-store unit directory, not the Landscape audit subgroup. It is not a defect per se but causes confusion when auditors trace "what's covered by U-CORE-1."

### Top 5 Deletion Candidates

1. **`test_data_flow_nan_rejection.py`** — entire file: AST introspection of production source code; structural, brittle, duplicates CI enforcement.
2. **`test_models_mutation_gaps.py::test_status_is_required_run_status_enum`** and **`test_registered_at_is_required`**: both survivors test `@dataclass` machinery (`isinstance` identity and required-field `TypeError`), not ELSPETH.
3. **`test_schema.py::TestNodesDeterminismColumn::test_determinism_values`**: hardcoded enum value set — tautology with zero defect-detection value.
4. **`test_row_data.py::test_row_data_state_values`**: same tautology pattern as above, different enum.
5. **`test_query_methods.py:63–64`** (the `hasattr` assertion only, not the file): uses a banned construct to test structural wiring; delete the two lines, not the surrounding test.

### Out-of-Scope Observations

1. `test_recorder_store_payload.py` tests `FilesystemPayloadStore` directly — its placement in the Landscape audit subgroup is a categorisation error; consider moving to `tests/unit/core/payload/`.
2. `test_lineage.py` uses `Mock()` throughout for `factory.query` and `factory.data_flow` — mock return values are unconstrained, so tests verify orchestration logic but not actual DB contract compliance; an integration counterpart would strengthen lineage coverage.
3. `test_database_sqlcipher.py` inserts `status="RUNNING"` as a raw string — bypasses enum validation and could mask a future serialisation change; out of scope for this review but worth noting.
4. `test_node_state_recording.py` `TestRecordRoutingEvents` class has 5–6 tests each re-registering `sink-2` and `edge-2` with identical boilerplate — not an anti-pattern per se, but a shared fixture would reduce maintenance surface.
5. `test_reproducibility.py` uses raw SQLAlchemy inserts across multiple tables to construct a non-deterministic call scenario — justified complexity given the scenario, but the setup would benefit from a named helper in `tests/fixtures/landscape` to make the intent legible.

---

## 2. axiom-sdlc-engineering:quality-assurance-analyst

### Verdict

**Theatre score: Some, leaning toward Pervasive in three concentrated zones.** Dominant pattern is **hash-without-binding** in recording tests — assertions like `request_hash is not None` or `len(sig) == 64` that verify SHA-256 produces 64 hex chars without binding the hash to its canonical input, despite the codebase explicitly framing hash↔payload binding as the integrity contract. Secondary patterns: dataclass-machinery round-trips, copy-paste duplication where parametrize is obvious, and one frozen-snapshot of known-buggy parser behavior.

### Theatre findings

| File:line | Category | Severity | Rationale | Recommendation |
|---|---|---|---|---|
| `test_call_recording.py:147–153, 168, 187, 204` | Hash-without-binding / audit silence | **High** | Asserts `call.request_hash is not None`, `call.error_json is not None`, `call.response_hash is not None` — never compares to `stable_hash(request_data)` and never reads back from DB. SUT could return random strings and pass. | Compare hash to `stable_hash(request_data)`; query the call back via `factory.execution.get_calls_for_state()` and assert persisted columns. |
| `test_exporter.py:386–405` | Hash-without-binding (signing) | **High** | `_sign_record` tests check `len==64`, hex chars, determinism, and inequality — never recompute HMAC-SHA256 against `b"test-key"` to verify the signature is the *correct* one for that input. The legal-record signature is verified only for "looks like a hash." | Add one test that compares to a hand-computed `hmac.new(b"test-key", canonical_json(record), sha256).hexdigest()`. |
| `test_models_mutation_gaps.py:32–66` | Dataclass-machinery (self-incriminating) | **High** | Author docstring admits original file tested `@dataclass` machinery; survivors `test_status_is_required_run_status_enum` and `test_run_with_all_optional_fields_set` set fields and read them back. Only `test_registered_at_is_required` (line 72) verifies a real contract. | Delete the class except `test_registered_at_is_required`. Subsumed by `test_models_enums.py` Tier-1 tests. |
| `test_journal.py:193–203` | Frozen-snapshot of buggy behavior | **High** | `test_no_column_list_parses_values_as_columns` asserts `table == "calls values"` and `cols == ["1", "2"]`. Test pins a parser bug as the contract. | Either fix the parser to reject malformed inserts, or delete the test and replace with a "rejects malformed insert" guard. |
| `test_journal.py:102–127` | Pass-through identity tests | Medium | `_normalize_parameters({"a":1}) == {"a":1}`, `_normalize_parameters(42) == 42`, `_normalize_parameters("hello") == "hello"`. Type-recursion tests pass through trivially. | Keep `test_tuple_params_converted_to_list`, `test_datetime_serialized`, `test_nested_dict_in_list`. Delete the rest. |
| `test_journal.py:69–91` | Coverage padding (parametrize candidate) | Medium | Eight one-line tests for `_is_write_statement` covering INSERT/UPDATE/DELETE/REPLACE/SELECT/CREATE/case/whitespace. | Parametrize as one matrix. |
| `test_database_ops.py:85–87, 147–156, 189–196` | Frozen-snapshot of error cosmetics | Low | `isinstance(rows, list)`, `"( )" not in message`, `"()" not in message` lock formatting cosmetics. | Delete; cosmetic regressions are reviewer-caught. |
| `test_recorder_store_payload.py:32–38` | Hash-without-binding | Medium | `test_store_payload_empty_bytes` only asserts `len == 64` for SHA-256 of empty bytes. Empty SHA-256 is well-known (`e3b0c44...`) — bind to it. | Replace with constant comparison; or delete (covered by `test_same_content_returns_same_hash`). |
| `test_validation_error_noncanonical.py:165–181` | Hash-without-binding | Medium | `test_repr_hash_helper` checks `repr_hash(42) != repr_hash("42")` (SHA-256 doesn't trivially collide — true) and `len == 64`. | Delete; the helper is exercised by the top-class tests which DO bind hash↔input. |
| `test_preflight_recording.py:202–218` | Dataclass-machinery (clean cluster) | Medium | `TestPreflightResult.test_construction`, `test_empty_tuples`, `test_frozen` — pure dataclass-frozen + tuple-arity tests, unrelated to the audit-recording focus of the rest of the file. | Delete the class. |
| `test_schema.py:69–118` | Dataclass-machinery | Medium | `test_checkpoint_model`, `test_checkpoint_model_with_aggregation_state`, `test_batch_status_is_typed`, `test_node_model_has_determinism_field` — construct dataclass, read field, assert equal. | Delete; contracts already enforced in `test_models_enums.py`. |
| `test_lineage.py:389–404` | Dataclass-machinery | Low | `test_has_expected_fields` constructs `LineageResult` and reads its fields. | Delete. |
| `test_query_methods.py` (multiple) `test_single_X` / `test_empty_for_unknown_X` series | Coverage padding | Medium | Across `TestGetRows/Tokens/States/Calls/Events/...`, the `test_single_X` adds nothing over `test_returns_X_ordered_by_index`. The run-scoping tests overlap with `test_where_exactness_consolidated.py`. | Delete `test_single_*` and `test_*_scoped_to_run` where covered by `test_where_exactness_consolidated.py`. |
| `test_query_methods.py:59–64` | Banned-pattern leakage | Low | Uses `hasattr` (banned by CLAUDE.md) to check repository injection. | Use direct attribute access with `pytest.raises(AttributeError)` or a positive isinstance check. |

### Duplicate clusters

1. **Reproducibility-grade-by-determinism** (`test_reproducibility.py:52–139`): six near-identical tests (`test_all_deterministic_returns_full`, `test_seeded_returns_full`, `test_nondeterministic_returns_replay`, `test_external_call_returns_replay`, `test_io_read_returns_replay`, `test_io_write_returns_replay`). All assert `compute_grade(db, "run-1") == EXPECTED` with one `Determinism` value per test. **Keep one parametrized test over `(determinism, expected_grade)` pairs.**
2. **NaN/Infinity rejection in formatters** (`test_formatters.py:56–70, 315–395, 438–445`): six tests across CSVFormatter (scalar, list, nested-list × NaN+Inf) and JSONFormatter (NaN, Inf), plus the AST scan in `test_data_flow_nan_rejection.py`. **Keep one parametrized matrix `(formatter, location, value)` per Three-Tier rule.**
3. **Run-scoping in query methods** (`test_query_methods.py` multiple `test_*_scoped_to_run` tests vs. `test_where_exactness_consolidated.py:236–321`): the consolidated file is the canonical run-scoping coverage. **Delete the per-entity scoping tests in `test_query_methods.py`.**
4. **`test_call_recording.py:40–72`**: `test_sequential_allocation_starts_at_zero` makes `test_single_allocation` redundant. Keep `test_sequential_allocation_starts_at_zero`.

### Delete with no loss of safety

1. `test_models_mutation_gaps.py:29–66` — `TestRunDataclass` (entire class except line-72 required-field test).
2. `test_preflight_recording.py:202–218` — `TestPreflightResult` (whole class).
3. `test_schema.py:69–118` — `test_checkpoint_model`, `test_checkpoint_model_with_aggregation_state`, `test_batch_status_is_typed`, `test_node_model_has_determinism_field`.
4. `test_journal.py:124–127` — `test_scalar_passes_through` (and `test_dict_params_normalized`/`test_list_params_normalized` at 102/106).
5. `test_database_ops.py:85–87` — `test_returns_list_type` (`isinstance(rows, list)`).
6. `test_database_ops.py:147–156, 189–196` — `test_*_no_context_omits_parens` (formatting cosmetics).
7. `test_validation_error_noncanonical.py:165–181` — `test_repr_hash_helper` (covered by upstream Tier-3 tests).
8. `test_lineage.py:389–404` — `test_has_expected_fields`.
9. `test_call_recording.py:67–72` — `test_single_allocation` (subset of `test_sequential_allocation_starts_at_zero`).
10. `test_journal.py:193–203` — `test_no_column_list_parses_values_as_columns` and `test_missing_close_paren_returns_none_columns` (delete or rewrite to assert rejection — current form pins buggy behavior).

### Coverage padding signals

- **`test_journal.py` is 846 lines** for what is, structurally, a small JSONL writer; statement-classification (4 lines of regex) gets 8 tests, parameter-normalization (recursive type dispatch) gets 6 tests covering pass-through identity. Padding is concentrated in helpers, not in the journal-write integrity path.
- **`test_query_methods.py:1888 lines, 93 tests** with a "rows / tokens / states / calls / events / outcomes / batches / operations …" matrix where each entity gets `returns / empty / single / scoped` — `single` adds no information once `returns_ordered` exists, and `scoped` is duplicated with `test_where_exactness_consolidated.py`.
- **Recording tests (call/token/node-state/batch/graph: ~5,300 lines combined) use 0 raw `db.connection()` reads.** They rely entirely on the returned recorder objects and repository getters. The audit-trail-as-legal-record framing in CLAUDE.md is undermined if no test directly reads the persisted bytes; refactors that change "what gets returned" but break "what gets persisted" would pass.
- **`test_formatters.py` Tier-1 NaN/Inf rejection is tested 6 times** in slight permutations — the contract is "no non-finite anywhere, ever," provable in one parametrized test.

### Out-of-scope observations

- `test_call_recording.py:147` and similar sites would benefit from a shared fixture asserting `(call.request_hash, call.response_hash) == (stable_hash(req), stable_hash(resp))` to retrofit hash-binding cheaply across the file.
- The HMAC signature in `exporter._sign_record` is the legal-record-export integrity primitive; consider an integration test that round-trips through `verify_record(signing_key, record, signature)` rather than only writer-side assertions.
- The existence of a "consolidated" file (`test_where_exactness_consolidated.py`) suggests prior consolidation pass; the per-entity scoping tests in `test_query_methods.py` look like leftovers that consolidation didn't remove.
- `hasattr` use in `test_query_methods.py:63–64` violates CLAUDE.md's banned-pattern rule (test files are not exempt; the ban is unconditional on the `hasattr` failure mode).
- `test_journal.py:200–203` pins a bug as a contract — a structural anti-pattern worth a `filigree observe` for the engine team.

---

## 3. axiom-python-engineering:python-code-reviewer

### Verdict

The landscape test suite has good structural bones — real in-memory SQLite, no `time.sleep`, no network leakage, and generally purposeful test bodies. Two categories of defect demand immediate attention: `hasattr()` calls directly contradict a CLAUDE.md unconditional ban and must be removed before the next review cycle; and two `pytest.raises((SchemaCompatibilityError, Exception))` calls in the SQLCipher file are logically vacuous — the `Exception` arm absorbs everything, rendering the type assertion meaningless. Spec-less mocks are the dominant recurring smell and the primary driver of future test rot.

### Findings Table

| file:line | category | severity | rationale | recommendation |
|-----------|----------|----------|-----------|----------------|
| `test_query_methods.py:63-64` | `hasattr()` — CLAUDE.md ban | **Critical** | `assert not hasattr(factory.query._ops, "execute_insert")` — banned unconditionally; swallows `@property` exceptions | Replace with direct attribute access; catch `AttributeError` explicitly if absence is the contract |
| `test_execution_repository.py:395` | `hasattr()` — CLAUDE.md ban | **Critical** | `if hasattr(stmt, "is_insert") and stmt.is_insert:` inside a mock-side-effect callback | Use `isinstance(stmt, sa.Insert)` instead |
| `test_data_flow_repository.py:1005, 1053, 1091` | `hasattr()` — CLAUDE.md ban | **Critical** | Same `if hasattr(stmt, "is_insert")` pattern repeated three times in mock callbacks | Same fix: `isinstance(stmt, sa.Insert)` |
| `test_database_sqlcipher.py:90, 111` | Catch-all exception tuple | **Critical** | `pytest.raises((SchemaCompatibilityError, Exception))` — `Exception` subsumes `SchemaCompatibilityError`; the test passes for any exception including `AssertionError` | Drop to `pytest.raises(SchemaCompatibilityError)` |
| `test_database_sqlcipher.py:183` | Bare `Exception` class | **Warning** | `pytest.raises(Exception, match="FOREIGN KEY constraint failed")` — `match=` mitigates but the type is a base class; a `RuntimeError` with the same message would pass | Use the specific SQLAlchemy integrity error type |
| `test_lineage.py:48-65` | Spec-less mocks | **Warning** | `_make_factory` builds `Mock()` for `query`, `data_flow`, `factory` with no `spec=`; renamed methods on `QueryRepository` or `DataFlowRepository` pass silently | Add `spec=QueryRepository`, `spec=DataFlowRepository`, `spec=RecorderFactory` |
| `test_journal.py:54` | Spec-less mock | **Warning** | `MagicMock()` in `_make_conn` without `spec=`; drives most journal tests | Add `spec=sqlalchemy.engine.Connection` |
| `test_journal.py:515, 560, 575, 589, 606, 619, 632, 654, 675, 687, 723, 742, 763` | Repeated private injection without spec | **Warning** | `journal._payload_store = Mock()` (~12 occurrences) — spec-less and reaches into private state | Extract a `payload_store` fixture with `spec=PayloadStore`; set via constructor or a supported setter |
| `test_database_compatibility_guards.py:289, 406, 569` | Spec-less mock | **Warning** | `mock_inspector = Mock()` without `spec=sqlalchemy.inspection.Inspector` | Add `spec=sqlalchemy.inspection.Inspector` |
| `test_database_compatibility_guards.py:17-25` | `__new__` bypass — private attrs | **Warning** | `LandscapeDB.__new__()` followed by direct assignment to `_passphrase`, `_journal`, `_engine`, `_require_existing_schema`; any constructor refactor silently breaks these tests | Add a `LandscapeDB.for_testing()` classmethod or inject via constructor |
| `test_call_recording.py:990` | Spec-less mock | **Warning** | `mock_store = MagicMock()` without `spec=` | Add `spec=PayloadStore` |
| `test_factory.py:12` | Private module import | **Warning** | `from elspeth.core.landscape._factory import _PluginAuditWriterAdapter` — private symbol import; breaks on any rename | Import through the public surface or skip the isinstance check |
| `test_factory.py:63-65` | Type-only assertion | **Warning** | `assert isinstance(writer, _PluginAuditWriterAdapter)` is the sole assertion — tests the class label, not behaviour | Assert that `writer.record_call(...)` produces an expected audit record |
| `test_error_recording.py:586-600, 746-760` | Timestamp ordering fragility | **Warning** | `assert timestamps == sorted(timestamps)` after rapid sequential inserts; fast hardware can produce collisions within timestamp resolution | Freeze time with `freezegun` or assert uniqueness separately from ordering |
| `test_recorder_store_payload.py:17, 32, 41` | `tempfile` instead of `tmp_path` | **Suggestion** | `tempfile.TemporaryDirectory()` as context manager misses cleanup on test failure | Replace with pytest `tmp_path` fixture |
| `test_where_exactness_consolidated.py:53` | Unguarded `time.time()` | **Suggestion** | Wall-clock call in `_make_secret_resolution` test-setup helper; not `freeze_time`-controlled | Parameterise with a fixed float constant or freeze via `freezegun` |
| `test_batch_recording.py:897-924` | Parametrize candidate | **Suggestion** | Three identical test bodies varying only by input `status` value | Collapse to `@pytest.mark.parametrize("status", [Status.DRAFT, Status.COMPLETED, Status.EXECUTING])` |

### Recurring Patterns

- **Spec-less `Mock()` / `MagicMock()`** appears in at least six files. This is the single biggest source of future rot: any rename or interface change on the real class passes silently through test runs.
- **`hasattr()` in mock side-effect callbacks** (`test_execution_repository`, `test_data_flow_repository`) — the ban is unconditional; reaching for it inside a lambda/closure makes the violation less visible but no less real.
- **Private attribute injection** (`journal._payload_store`, `LandscapeDB.__new__`) creates a secondary coupling layer on top of spec-less mocks: the tests depend on both the internal layout and the public interface simultaneously.
- **Vacuous exception tuples** — `(SpecificError, Exception)` is a pattern that looks defensive but is logically equivalent to `pytest.raises(Exception)`. The specific type adds no protection.
- **Repeated identical test bodies** for enum-variant exhaustion (batch status trilogy, SQLCipher URL rejection) are an unforced duplication; parametrize is the idiomatic fix.

### Candidates for Deletion or Rewrite

| test body | reason |
|-----------|--------|
| `test_factory.py::test_plugin_audit_writer_is_adapter` | Type-only assertion against a private class; no behaviour is verified. Delete or rewrite to assert audit output. |
| `test_database_sqlcipher.py` lines 90, 111 (both `raises` calls) | The `Exception` arm makes them equivalent to `pytest.raises(Exception)` — rewrite with the specific error type only. |
| `test_database_compatibility_guards.py` `_make_instance` helper | `__new__` bypass is fragile; replace with a factory method on `LandscapeDB` or a `from_url` call with a pre-seeded DB. |

### Pytest Hygiene Observations

- **`tmp_path` adoption** is incomplete; `test_recorder_store_payload.py` still uses `tempfile.TemporaryDirectory()`. A project-wide pass replacing manual tempdir management with `tmp_path` removes the failure-path cleanup gap.
- **Fixture spec coverage** should be added as a linting policy. Consider a custom ruff rule or a CI grep that flags `Mock()` and `MagicMock()` without `spec=` in `tests/unit/core/landscape/`.
- **`freezegun`** is absent from this suite. Two timestamp-ordering assertions (`test_error_recording.py`) and one `time.time()` call (`test_where_exactness_consolidated.py`) would benefit from it; add it to `[dev]` extras if not already present.

### Out-of-Scope Observations

- `test_database_compatibility_guards.py:306` patches `sqlalchemy.inspect` as a global monkeypatch — fragile to SQLAlchemy internal moves; worth flagging for a future SQLAlchemy upgrade cycle.
- `test_call_recording.py:330, 339` reaches through `factory.execution._ops` to inject DB corruption for Tier-1 guard tests — the motivation is legitimate but the coupling is real; a `DatabaseOps.corrupt_for_testing()` helper would decouple these.

### Confidence Assessment
High. All cited findings were verified against source lines with explicit grep and read passes across the full 32-file corpus. The `hasattr` and catch-all exception findings are exact; mock spec coverage was confirmed by direct file read for the highest-frequency files.

### Risk Assessment
The `hasattr` violations are the highest-risk category given the CLAUDE.md unconditional ban — they are test-code defects, not production defects, but they represent banned patterns that could propagate. The vacuous `pytest.raises` tuples are a correctness defect: the tests provide no meaningful type guard.

### Information Gaps
`test_call_recording.py` and `test_where_exactness_consolidated.py` were grep-checked rather than fully read; additional smells in those files beyond the cited lines are possible but unlikely to change the overall verdict.

### Caveats
Timestamp fragility findings depend on hardware speed; they are deterministically reproducible only on machines with coarse timestamp resolution. The `tempfile` finding is a cleanup-on-failure gap, not a correctness issue for passing tests.

---

## 4. pr-review-toolkit:pr-test-analyzer

### Chunk verdict

The 32-file Landscape suite has **strong scenario coverage** for the legal-record audit backbone: Tier-1 crash paths are extensively exercised (model loaders' "wrong status with completed_at" guards, `test_database_compatibility_guards.py`'s schema-shape rejections, `test_lineage.py`'s `TestExplainTier1Corruption` / `TestExplainParentIntegrity`, `TokenOutcomeLoaderTwoAxis` constraint enforcement, `test_data_flow_repository.py`'s `test_record_rejects_each_constraint_row_violation`). The most dangerous gaps are: (a) **no end-to-end attributability round-trip** that records Source→Transform→Sink and asserts `explain(run_id, token_id)` reproduces the recorded chain; (b) **terminal-state coverage of the closed set is partial** — `EXPANDED`, `COALESCED`, `CONSUMED_IN_BATCH` are constraint-checked but never asserted as the *only* terminal a token reaches; (c) **hash↔payload binding survival** (the "hashes survive payload deletion" guarantee) is not explicitly asserted at the recorder level — only `test_reproducibility.py:258` indirectly probes it via grade demotion.

### Per-file scenario table

| File | Primary claim | Scenarios covered | Scenarios missing |
|---|---|---|---|
| test_batch_recording.py | Batch lifecycle recording | begin/complete/fail, attempt N retries, member ordinal | concurrent flush race; batch FK violation crash |
| test_call_recording.py | Call/Operation recording incl. payload-store binding | request/response hash, payload-ref roundtrip, double-complete crash, orphan-payload prevention, integrity-error crash | call recorded with `request_hash` of NaN payload (repr fallback path); cross-run call FK |
| test_database_compatibility_guards.py | Schema epoch & shape guards | future epoch, ADR-018 shapes, missing tables/columns/FKs/indexes/check-constraints, sqlcipher translation | downgrade after partial migration mid-write |
| test_database_ops.py | Read/write helper Tier-1 invariants | rowcount==0 crash, multi-row crash, write rejected on read-only, context in error msg | concurrent writers; DB-locked timeout |
| test_database_sqlcipher.py | Encryption-at-rest invariants | passphrase ordering, wrong/no passphrase, WAL+FK active, postgres/mysql/mem rejected, quote/backslash escaping | empty-passphrase rejection; passphrase length limit |
| test_data_flow_nan_rejection.py | AST audit: every json.dumps has allow_nan=False | static-AST scan one module | none — but see §5 |
| test_data_flow_repository.py | Token/row/edge/outcome write-side | constraint-pair exhaustive, fork/coalesce/expand atomicity rollback, rowcount==0 crash, NaN repr-fallback only when quarantined, cross-run contamination | join token (only fork/coalesce/expand exercised); partial expand rowcount mismatch |
| test_error_recording.py | Validation/transform error recording | VERR/TERR roundtrip, NaN/Inf in row_data, cross-run FK rejection, ordering | what happens when error_id collision is forced |
| test_error_serialization_dispatch.py | Error dataclass→canonical JSON dispatch | to_dict shape, hash stability, cross-type consistency | unknown subclass falls through (negative for dispatch) |
| test_execution_repository.py | Node-state + routing-event lifecycle | (very large; happy + crash paths) | see §3 — terminal-state assertions vs closed set |
| test_exporter.py | Audit export to file formats | format roundtrip, datetime ISO, NaN rejection | export interrupted mid-write leaves no partial; checksum of export |
| test_factory.py | Factory wires repos & shares DB | construction, payload_store propagation, adapter type | factory rejects mismatched DB schema epoch |
| test_formatters.py | CSV/JSON/lineage text formatting | nan/inf rejection, datetime, nested flatten, none handling | extreme-depth recursion limit; circular ref |
| test_graph_recording.py | Node/edge graph recording | (large) | DAG cycle: does the recorder reject it? |
| test_journal.py | Pre-commit journal of writes | write/SELECT split, buffer/flush/rollback, payload enrichment, fail-on-error vs disable-after-N | journal file fsync atomicity; concurrent attach |
| test_lineage.py | explain() lineage queries | terminal/non-terminal filter, sink-equality, parent integrity, group_id validity | the actual SOURCE→ACT chain recovery |
| test_model_loaders.py | DB row → typed dataclass | every status×field invariant for NodeState/Operation/TokenOutcome, enum rejection | corrupt JSON in error_json columns |
| test_models_enums.py | Audit dataclass rejects str/int for enum | 5 fields | partial coverage of enum-typed fields (CallType, Determinism on TokenParent etc.) |
| test_models_mutation_gaps.py | Surviving non-trivial dataclass tests | required-field, all-optional-set | see §4 — these are pointless |
| test_node_state_recording.py | begin/complete node-state | open→completed/failed/pending, success_reason/error_json mutual exclusion, hash determinism | concurrent two writers same state-id |
| test_preflight_recording.py | Preflight/readiness audit | dep run, gate result, canonical JSON, mixed | recording fails mid-list — atomicity |
| test_query_methods.py | Read-side queries | (huge: get_rows, get_tokens, get_calls, hash lookup) | none significant |
| test_recorder_store_payload.py | PayloadStore content-addressed write | sha256 hex, empty bytes, dedup | retrieve from corrupt/missing blob; permission denied on dir |
| test_reproducibility.py | Grade derivation | deterministic/seeded/io/replay paths, post-purge demotion | grade computation when run is mid-flight (RUNNING) |
| test_row_data.py | RowDataState/Result invariants | available requires dict, frozen, repr_fallback, hash_only | none significant |
| test_run_lifecycle_repository.py | Run lifecycle Tier-1 invariants | running→running/failed/completed transition matrix, double-completion crash, contract overwrite crash, atomicity | run-id uniqueness contention |
| test_run_recording.py | Higher-level run recording | begin/complete/finalize, set_export_status, list_runs | settings_json>1MB cliff |
| test_schema.py | SQLAlchemy schema columns | determinism column, checkpoint topology hashes, BatchStatus enum | partial — see §5 |
| test_source_file_hash.py | Node.source_file_hash format | accept None / sha256:16hex; reject prefix/length/empty | hash of large source file |
| test_token_recording.py | Token/row creation | hash determinism, branch_name, fork_group_id, get_row roundtrip | terminal-state assertion |
| test_validation_error_noncanonical.py | Validation error w/ non-canonical row | primitive int audit roundtrip, NaN repr_hash, multiple rows | row_hash collision between distinct rows |
| test_where_exactness_consolidated.py | All queries scope by run_id | every query method's run-isolation | composite-PK exact match (only run_id verified) |

### Critical scenario gaps (chunk-wide)

- **G1 — Attributability round-trip is missing.** No file in this chunk records a Source→Transform→Sink trace and asserts `explain(...)` returns full lineage including config_hash and payload_ref. `test_lineage.py` validates explain *invariants* against synthesized fixtures; it never proves a recorded run can be explained. Should live in a new `test_attributability_roundtrip.py` or extending `test_lineage.py`. Severity 9.
- **G2 — Closed-set terminal-state coverage is incomplete.** The Manifesto requires every row to reach exactly one of 8 terminals (`COMPLETED`, `ROUTED`, `FORKED`, `CONSUMED_IN_BATCH`, `COALESCED`, `QUARANTINED`, `FAILED`, `EXPANDED`). `test_data_flow_repository.py` validates constraint-pair *legality* but no test asserts "for token T, exactly one terminal outcome row exists and it is one of the eight." Should live in `test_data_flow_repository.py::TestRecordTokenOutcomeTwoAxis`. Severity 8.
- **G3 — Hash↔payload binding survival.** "Hashes survive payload deletion" is the cornerstone of audit integrity. No test in `test_call_recording.py` or `test_recorder_store_payload.py` proves: (a) record payload, (b) delete blob, (c) hash still present, (d) `get_call_response_data` returns `PURGED` not `NEVER_STORED`. `test_call_recording.py:1025 test_purged_when_payload_removed` is the closest but does not assert grade demotion or that hash on the row is unchanged. Severity 9.
- **G4 — Concurrent / contention paths.** Audit DB writes are crash-on-failure; no test exercises (a) two recorders racing on the same `state_id`, (b) `INSERT OR IGNORE` collision under contention, (c) WAL checkpoint while a writer holds the lock. `test_database_ops.py:113` is the natural home. Severity 7.
- **G5 — `audit-recorder-was-called` assertion at higher layers.** The closest recordings tests use repository-level fakes/spies. There is no test that wires `make_context` and asserts that, e.g., `record_validation_error` actually persisted to the underlying DB and is visible via the public query API. `test_validation_error_noncanonical.py` does this for one path; it's not generalised. Severity 7.

### Low-effort / pointless tests

- `test_models_mutation_gaps.py:32` `test_status_is_required_run_status_enum` — only asserts `Run` constructs and `isinstance(status, RunStatus)`.
- `test_models_mutation_gaps.py:45` `test_run_with_all_optional_fields_set` — dataclass smoke test.
- `test_models_mutation_gaps.py:72` `test_registered_at_is_required` — tests Python's `dataclass(...)` `TypeError`.
- `test_factory.py:30` `test_creates_all_four_repositories` — `isinstance` checks; tautological.
- `test_factory.py:50` `test_payload_store_propagated` — `assert factory.payload_store is mock_store` against a value just passed in.
- `test_factory.py:56` `test_payload_store_defaults_to_none` — asserts default value.
- `test_factory.py:63` `test_plugin_audit_writer_is_adapter` — `isinstance` on private class.
- `test_recorder_store_payload.py:32` `test_store_payload_empty_bytes` — asserts `len(sha256_hex) == 64` for empty input.
- `test_row_data.py:13` `test_row_data_state_values` — pins `.value` strings of an enum literal-by-literal.
- `test_row_data.py:50` `test_row_data_result_is_frozen` — tests stdlib `frozen=True`.
- `test_schema.py:10`, `:69`, `:84`, `:105` — all four `test_*_model` cases simply construct the dataclass with valid arguments.
- `test_journal.py:69-90` — eight one-liner cases of "does this prefix match `INSERT|UPDATE|DELETE|REPLACE`."
- `test_error_serialization_dispatch.py:67-87` — could collapse to one parametrised assertion.

### Tests that exercise nothing the type system / SQLite engine doesn't already guarantee

- `test_data_flow_nan_rejection.py:16` — useful but **AST-static** scan of one module.
- `test_models_mutation_gaps.py:72` — reproduces `dataclass`'s missing-arg `TypeError`.
- `test_schema.py:10`, `:105` — `assert node.determinism == Determinism.DETERMINISTIC` after passing it in.
- `test_database_ops.py:85` `test_returns_list_type` — already implied by other tests.
- `test_run_lifecycle_repository.py:497` `test_empty_status_accepted_and_round_trips` — accepts an empty-string status (potentially Tier-1 violation).
- `test_call_recording.py:74` `test_seeds_from_database_on_factory_recreation` — exercises SQL `MAX()` semantics.

### Out-of-scope observations

- `test_reproducibility.py:255` `TestUpdateGradeAfterPurge` is the only place that tests the post-payload-deletion grade transition.
- `test_call_recording.py:982` `test_payload_integrity_error_raises_audit_integrity` is exemplary.
- `test_journal.py:638` `test_integrity_error_always_crashes_as_audit_violation` is also exemplary.
- `test_where_exactness_consolidated.py` is a strong WHERE-clause invariant suite.
- The "audit-recorder-was-called" assertion pattern would benefit from a single shared fixture (a `RecordingSpy` wrapping the factory).

---

## 5. ordis-quality-engineering:coverage-gap-analyst

### SUT Footprint Summary

| Test Cluster | Production Module(s) | Coverage Quality |
|---|---|---|
| Recording (batch/call/error/graph/node-state/preflight/run/token) | `data_flow_repository.py`, `execution_repository.py`, `run_lifecycle_repository.py` | Strong |
| Repository (data-flow/execution/run-lifecycle) | Same three repos — direct unit access | Strong |
| DB ops (compatibility-guards/sqlcipher/ops) | `database.py`, `_database_ops.py` | Strong |
| Models (enums/mutation-gaps/loaders) | `model_loaders.py`, contracts enums | Strong |
| Schema | `schema.py` | Thin (structural only, no write-path assertions) |
| Serialization | `formatters.py`, `errors.py`, `row_data.py` | Strong |
| Querying | `query_repository.py` | Strong |
| Other | All modules | Thin-to-Strong varies by file |

### Critical Gaps (Audit-Integrity / Security)

| File:Function | Gap | Why Critical | Suggested Test Type |
|---|---|---|---|
| `test_data_flow_repository.py` : `DataFlowRepository._validate_token_row_ownership` | Zero tests for `_validate_token_row_ownership()`. The method prevents cross-row lineage corruption (token attributed to wrong source row) but is never directly invoked in the unit suite. It is called from `fork_token`, `coalesce_tokens`, `expand_token` — all of which mock or stub the underlying check. | A token with a swapped `row_id` produces a valid-looking audit trail linking the wrong source data to a terminal decision. Attributability test fails silently. | Unit — call the method directly with mismatched `token_id`/`row_id` |
| `test_data_flow_repository.py` : `DataFlowRepository.link_validation_error_to_row` | No test exists anywhere (unit or integration) for `link_validation_error_to_row()`, including its three Tier-1 crash branches: cross-run contamination, non-existent error_id, and re-link to a different row. | This method is the quarantine lineage exactness guarantee referenced in `schema.py` (epoch 4 comment). Without it tested, the `validation_errors.row_id` FK can be silently left NULL or relinked incorrectly, destroying `explain()` ability to resolve exact validation failures for quarantined tokens. | Unit — insert validation_error + row, call the method, assert crash on each mismatch |
| `test_database_compatibility_guards.py` : `_validate_schema` — composite FK path | Composite FK validation (`_REQUIRED_COMPOSITE_FOREIGN_KEYS`, 12 entries) is exercised by only a single test (`test_validate_schema_rejects_stale_single_column_foreign_keys_for_run_scoped_error_tables`) that monkeypatches a 2-entry list covering only `transform_errors`. The 10 remaining composite FK contracts (token_outcomes, node_states, artifacts, batches, batch_members) are never validated by any test. | A production DB missing e.g. `token_outcomes(token_id, run_id) → tokens(token_id, run_id)` will pass schema validation and allow cross-run outcome contamination to enter the audit trail silently. | Unit — monkeypatch each `_REQUIRED_COMPOSITE_FOREIGN_KEYS` entry individually and assert `SchemaCompatibilityError` is raised |

### High-Risk Gaps

| File:Function | Gap | Why High-Risk | Suggested Test Type |
|---|---|---|---|
| `test_data_flow_repository.py` : `sweep_deferred_invariants_or_crash` | Zero unit tests for `sweep_deferred_invariants_or_crash()`, `find_orphaned_transient_parents()`, and `find_orphaned_batch_consumptions()`. Coverage exists only in integration tests outside this cluster's SUT scope. | The sweep is the ADR-019 I1a/I1b invariant enforcement at run-end. A regression in its SQL would produce a silent false-negative: orphaned fork/expand parents pass the sweep and a corrupt audit trail reaches storage. | Unit — insert orphan parent outcomes with no child witnesses; assert `AuditIntegrityError` |
| `test_execution_repository.py` : `allocate_call_index` resume seeding | The resume seeding path (`existing_max` DB seed on first access) is tested in `test_call_recording.py` but only via the facade. No test creates an `ExecutionRepository` directly, pre-inserts call rows for a `state_id`, and verifies the new instance seeds from the DB correctly. | A seeding regression on resume produces `UNIQUE(state_id, call_index)` violations mid-run or silently overwrites prior call records if FK enforcement is off. | Unit — pre-insert 3 calls for `state_id`, recreate `ExecutionRepository`, allocate one index, assert it is 3 |
| `test_run_lifecycle_repository.py` : `get_run` Tier-1 read guard | `RunLoader.load()` reads `status` from DB and maps it to `RunStatus`. No test verifies that an unknown `status` string in the DB crashes (Tier-1 read guard). | A silent coercion or `None` return for a corrupt status would allow a run to be marked "completed" when it isn't. | Unit — corrupt `status` column in DB, call `get_run`, assert `AuditIntegrityError` or enum crash |
| `test_database_compatibility_guards.py` : `read_only_connection` — unsupported backend | `read_only_connection()` raises `RuntimeError` for backends other than SQLite or PostgreSQL. Tested in `tests/unit/mcp/` but not in `test_database_ops.py` or `test_database_sqlcipher.py`. | If an unsupported backend slips through config validation, `read_only_connection` silently yields a writable connection to MCP query tools. | Unit — mock `engine.dialect.name = "mysql"`, assert `RuntimeError` |

### Medium-Risk Gaps

| File:Function | Gap | Suggested Test Type |
|---|---|---|
| `test_run_lifecycle_repository.py` : `record_preflight_results` atomicity | The parallel `_execute_atomic_inserts` path used by `record_preflight_results` and `record_readiness_check` has no atomicity failure test. | Unit — inject failure mid-insert, verify zero rows committed |
| `test_execution_repository.py` : `_materialize_call_ref_after_insert` failure | `LandscapePostCommitError` is raised when payload store fails after the call row is committed. No test covers this path directly. | Unit — mock `PayloadStore.store` to throw, verify `LandscapePostCommitError` with call_id context |
| `test_data_flow_repository.py` : `get_token_outcomes_for_row` run isolation | The JOIN query filters by `run_id` to prevent cross-run outcome leakage, but no test inserts outcomes for two runs with the same `row_id` and verifies isolation. | Unit — two runs, same row_id in each; assert outcomes are isolated |
| `test_schema.py` : `schema.py` — `token_outcomes` partial index definition | The schema file defines a partial unique index. No test verifies this index is actually created with the correct predicate on a live SQLite DB. | Unit — create DB via `LandscapeDB.in_memory()`, reflect the index, assert `sqlite_where` predicate contains `completed = 1` |
| `test_lineage.py` : `explain()` — both `token_id` and `row_id` supplied | When both are provided, `token_id` wins. No test verifies this precedence. | Unit — supply both, assert `token_id` is used |

### Quick Wins

| File | Gap | Value | Effort |
|---|---|---|---|
| `test_schema.py` | Assert the partial unique index predicate on a live in-memory DB | Prevents silent DDL regression allowing duplicate terminal outcomes | < 15 lines |
| `test_lineage.py` | Test `explain()` with both `token_id` and `row_id` supplied | Documents precedence rule | < 10 lines |
| `test_data_flow_repository.py` | Test `link_validation_error_to_row` idempotent re-link | Covers the early-exit branch | < 15 lines |
| `test_run_lifecycle_repository.py` | Test `get_run` with a corrupt `status` string in DB | Tier-1 read guard | < 20 lines |

### Notable Strengths

- `test_database_compatibility_guards.py` has thorough ADR-019 migration detection: stale `is_terminal` column, non-nullable `outcome`, stale index predicate, and PostgreSQL shape — all backed by real DDL, not mocks.
- `test_data_flow_repository.py` covers ADR-019 real-time cross-table invariants (I1c, I3) with actual DB state including the node-state and artifact witnesses.
- `test_execution_repository.py` covers `complete_node_state` write-side guards for all impossible state combinations.
- `test_lineage.py` covers the full Tier-1 corruption detection surface in `explain()`.

### Confidence Assessment

**High confidence** on the gap findings for `link_validation_error_to_row` (zero hits across all test paths confirmed by grep), composite FK coverage gaps (confirmed by reading both the test and source), and `_validate_token_row_ownership` (zero test hits). **Medium confidence** on the `allocate_call_index` resume seeding gap.

### Information Gaps

The integration test suite (outside this cluster's SUT scope) likely covers some of these gaps. The analysis is bounded to the 32 unit files as chartered.

### Caveats

Schema-level gaps (partial index predicate, composite FK exhaustive testing) are low-probability regressions because the schema is not frequently modified. Their criticality is elevated here because the project's explicit migration policy is DB deletion rather than Alembic, meaning any silent DDL regression persists undetected across the entire install lifetime until an auditor runs a query that should have failed a uniqueness constraint.
