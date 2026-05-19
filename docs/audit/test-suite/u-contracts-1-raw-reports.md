# U-CONTRACTS-1 — Raw Agent Reports

Verbatim outputs from the 5 specialist agents that reviewed U-CONTRACTS-1 on 2026-05-06.
Preserved in full as the source of truth backing the synthesis in `u-contracts-1-findings.md`.

Agent IDs (not durable across sessions, recorded for traceability):
- test-suite-reviewer: `a4f1c4ad2a6ee4533`
- quality-assurance-analyst: `a2e58327312e3189a`
- python-code-reviewer: `a7225a6919345092d`
- pr-test-analyzer: `a96d26ce4faeb997e`
- coverage-gap-analyst: `ab8b20e87b17067f6`

---

## 1. ordis-quality-engineering:test-suite-reviewer

### Summary

17 of 25 files contain findings. The dominant anti-pattern is **tautology**: four protocol base-classes collectively contribute ~40 `isinstance`-only assertions that restate the Protocol's type signature — proving nothing a type-checker doesn't already enforce, and breaking on rename rather than on behavioural regression. The overall verdict is **Mixed**: a strong core of deep-freeze, thread-safety, and hash-determinism tests is inflated by a layer of protocol-attribute and lifecycle-smoke tests with near-zero safety value.

### Findings Table

| File:line | Category | Severity | Rationale | Recommendation |
|-----------|----------|----------|-----------|----------------|
| `test_sink_protocol.py:84–104` | Tautology | Major | 5 `isinstance(sink.X, T)` assertions restate the Protocol's declared type signature; mypy enforces these statically | Delete or convert to protocol-compliance integration test that verifies observable runtime behaviour |
| `test_source_protocol.py:69–89` | Tautology | Major | 5 `isinstance(source.X, T)` assertions over `name`, `output_schema`, `determinism`, `plugin_version`, `declared_guaranteed_fields`; same issue | Same as above |
| `test_transform_protocol.py:92–126` | Tautology | Major | 7 `isinstance(transform.X, T)` / `hasattr + isinstance` attribute checks duplicating Protocol definition | Same as above |
| `test_batch_transform_protocol.py:180–214` | Tautology | Major | 7 attribute-presence tests that duplicate `test_transform_protocol.py:92–126` exactly, since `BatchTransformMixin` already implements `TransformProtocol` | Delete; the parent-class tests already cover them |
| `test_sink_protocol.py:320–336` | Frozen-implementation | Major | Line 325 uses `SinkProtocol.__annotations__` membership; line 335 uses `hasattr(SinkProtocol, ...)` — the literal banned pattern in CLAUDE.md; both duplicate what mypy enforces | Delete both; attribute presence is a mypy concern |
| `test_sink_protocol.py:207–220` | Assertion-free smoke | Minor | `test_flush_is_idempotent` calls `flush()` three times with no assertion; pass = no exception thrown, not idempotency | Assert observable state is unchanged (e.g. file size, result hash) across repeated calls |
| `test_sink_protocol.py:222–232` | Assertion-free smoke | Minor | `test_close_is_idempotent` calls `close()` three times, asserts nothing | Same fix |
| `test_source_protocol.py:136–155` | Assertion-free smoke | Minor | `test_close_is_idempotent`, `test_on_start_does_not_raise`, `test_on_complete_does_not_raise` — pass iff no exception | Add post-condition assertions |
| `test_transform_protocol.py:204–242` | Assertion-free smoke | Minor | Same pattern: lifecycle tests assert nothing beyond absence of exception | Same fix |
| `test_plugin_schema.py:7–54` | Tautology (import-shape) | Minor | 6 tests assert `from elspeth.contracts import X` succeeds; tests packaging topology, not behaviour | Delete; packaging is verified by any test that successfully imports the SUT |
| `test_plugin_schema.py:127–134` | One-time migration check | Minor | `test_old_import_path_removed` checks a deleted module; per the no-legacy-code policy, migration guards are not retained | Delete |
| `test_plugin_assistance.py:54–64` | Misleading test name | Minor | `test_required_fields` constructs an object with all required fields already supplied and asserts optional fields default to `()` — tests defaults, not required-ness | Rename to `test_optional_fields_have_empty_defaults`; add a test that omits each required field and expects `TypeError` |
| `test_tier_registry.py:152–170` | Tautology / self-reference | Minor | `test_tests_prefix_only_allowed_under_pytest` asserts `"pytest" in sys.modules` while running under pytest; the subprocess test at line 243 is the real check | Delete this half; keep only the subprocess test |
| `test_azure_content_safety_contract.py:45–51` | Dead code | Minor | `_make_mock_context` defined but never called in this file | Delete |
| `test_azure_multi_query_contract.py:35–42` | Dead code | Minor | `_make_mock_context` defined but never called | Delete |
| `test_azure_prompt_shield_contract.py:41–47` | Dead code | Minor | `_make_mock_context` defined but never called | Delete |
| `test_plugin_protocols.py` | Misleading scope | Minor | Filename implies broad protocol coverage; single test verifies only `CSVSource.__init__` rejects a malformed schema config | Rename to `test_csv_source_init_validation.py` or move test to `test_csv_source_contract.py` |

### Cross-file Patterns

- **Tautological isinstance assertions in protocol base classes.** `test_sink_protocol.py`, `test_source_protocol.py`, `test_transform_protocol.py`, and `test_batch_transform_protocol.py` collectively contain ~40 tests of the form `assert isinstance(plugin.attr, SomeType)`. Every one of these is satisfied by any class that passes a mypy strict-mode check against the Protocol. They detect no bugs; they fail on rename. Deleting them would remove the noisiest failure category during refactors.
- **Lifecycle smoke-by-omission.** All four protocol base-classes contain `test_X_does_not_raise` or `test_X_is_idempotent` tests that call methods with no subsequent assertion. If a method silently deletes state or returns without executing its contract, these tests pass. Real idempotency needs a pre/post state comparison.
- **Dead `_make_mock_context` helpers in 3 Azure contract files.** The autouse `mock_httpx_for_batch` fixture in each file supplies its own inline mock context; the standalone `_make_mock_context` function was never wired up. Three-file copy-paste artefact.
- **Inherited protocol attribute tests duplicated in batch base-class.** `BatchTransformContractTestBase` re-declares every attribute assertion already in `TransformContractTestBase` (lines 180–214 vs 92–126). Since `BatchTransformMixin` implements `TransformProtocol`, the batch subclass inherits coverage automatically; the re-declaration doubles maintenance cost for zero gain.

### Top 5 Deletion Candidates

1. **`test_plugin_schema.py:7–54` (`TestPluginSchemaLocation`)** — Six import-shape tests; any test that imports the SUT already covers this; delete the whole class.
2. **`test_batch_transform_protocol.py:180–214` (attribute assertion block)** — Exact duplicate of `test_transform_protocol.py:92–126` via mixin inheritance; safe to delete with no coverage delta.
3. **`test_plugin_schema.py:127–134` (`test_old_import_path_removed`)** — One-time migration guard for a deleted module; no-legacy-code policy prohibits retaining it.
4. **`test_tier_registry.py:152–170` (`test_tests_prefix_only_allowed_under_pytest` in-process half)** — Asserts `"pytest" in sys.modules` while executing under pytest; the subprocess version at line 243 is the actual proof; this half is a tautology.
5. **`test_sink_protocol.py:320–336` (two standalone protocol attribute tests)** — Uses the CLAUDE.md-banned `hasattr` and `__annotations__` access patterns to test Protocol structure; mypy enforces this at zero test-budget cost.

### Out-of-scope Observations

1. **`test_tier_registry_migration.py:40–72`** — directly mutates `tier_registry._FROZEN` mid-test to bypass the post-bootstrap guard. The correct fix is a sanctioned test-mode hook on the SUT; direct flag-flipping is fragile if the freeze mechanism ever becomes more than a bool.
2. **`test_batch_transform_protocol.py:386, 473`** — `wait_for_results` budgets of 10 s, 30 s, and 60 s. The mechanism is correct (event-signaled, not busy-wait), but the upper bounds are loose enough to hide threading performance regressions silently.

**Confidence Assessment:** High for tautology and dead-code findings (verified line-by-line). Moderate for lifecycle smoke findings — the "assert nothing" pattern is clear, but I cannot confirm whether the test framework itself catches certain side-effects.
**Risk Assessment:** Low. Every deletion candidate is either statically provable by mypy, a one-time migration guard, or a copy-paste duplicate.
**Information Gaps:** I have not confirmed whether any CI gate runs mypy in strict mode against the Protocol definitions; if it does not, the tautology risk for the protocol attribute tests is higher and they may be worth converting rather than deleting.

---

## 2. axiom-sdlc-engineering:quality-assurance-analyst

### Verdict

**Score: Some** (concentrated, not pervasive). The dominant pattern is **sut-mirror / type-only assertions**, clustered in `test_plugin_semantics.py` and `test_plugin_schema.py`. The protocol contract bases (sink/source/transform) carry a substantial layer of attribute roll-call tests that the type system already guarantees, but the property/determinism/idempotency tests in the same files are genuinely valuable. The tier-registry and registry-primitive tests are mostly real behavioural tests (concurrency, allowlist enforcement, freeze semantics) and should be left alone.

### Theatre findings

| File:line | Category | Severity | Rationale | Recommendation |
|-----------|----------|----------|-----------|----------------|
| `test_plugin_semantics.py:318-349` | sut-mirror | **Critical** | The Hypothesis body reimplements the exact branching of `compare_semantic` (`if UNKNOWN -> UNKNOWN; elif kind in accepted and framing in accepted -> SATISFIED; else CONFLICT`) then asserts the SUT agrees. Two copies of the same spec — cannot detect a bug, only divergence. | Replace with discrete examples that pin specific behaviour (already covered at 256-311) and delete the property test, OR rewrite the property to assert an *independent* invariant (e.g. monotonicity: enlarging accepted sets never turns SATISFIED into CONFLICT). |
| `test_plugin_semantics.py:26-32, 51-56, 69-72, 76-79` | field-mirror | Minor | `assert ContentKind.PLAIN_TEXT.value == "plain_text"` etc. Both sides of the assertion are the same source of truth — the enum literal. | Delete the per-member asserts. The closed-membership tests at 38-47 and 58-65 already backstop accidental rename/removal as a CLOSED-LIST trip-wire. |
| `test_plugin_schema.py:9-54` (5 of 6 `_importable_from_contracts` tests) | padding | Major | Asserting `from elspeth.contracts import X` succeeds and that `X(...).attr == passed_value`. The latter is mock-tautology; the former is import existence already enforced by every downstream test. | Keep `test_old_import_path_removed` (127-133, real constraint). Collapse the rest into a single `test_contracts_re_exports` test that imports the names and checks `__all__`. |
| `test_plugin_schema.py:60-121` (CompatibilityResult error-message branches) | sut-mirror | Minor | Each branch test constructs the dataclass with one error category and asserts the rendered string matches the format string in the SUT. Copies the format string verbatim. | Keep one combined test (96-114, the `;`-joined variant — real combinator coverage). Delete the 4 single-category tests; the combined one exercises each branch. |
| `test_sink_protocol.py:84-104` | type-only | Minor | `test_sink_has_name`, `_has_input_schema`, `_has_determinism`, `_has_plugin_version`, `_has_idempotent_flag` — all `isinstance(x.attr, T)`. The type system already enforces this for any SinkProtocol-typed reference. | Collapse to one `test_protocol_attributes_present_and_typed` that uses `runtime_checkable` Protocol check, or delete entirely if a CI lint enforces Protocol conformance. |
| `test_source_protocol.py:69-89` | type-only | Minor | Same pattern: 5 `assert isinstance(...)` on protocol attributes. | Same recommendation. |
| `test_transform_protocol.py:92-126` | type-only | Minor | 7 attribute-presence/type checks. The accompanying `hasattr` calls are also banned by CLAUDE.md as a production pattern but are defensible in tests; even so, low signal. | Same recommendation; the `_skip_if_batch_transform` plumbing is fine. |
| `test_sink_protocol.py:320-336` | frozen-snapshot | Minor | Asserts `"supports_resume" in SinkProtocol.__annotations__` and `hasattr(SinkProtocol, "configure_for_resume")`. Locks Protocol shape. | Delete — replace with a runtime-checkable `isinstance(real_sink, SinkProtocol)` check at one CSVSink test site. |
| `test_plugin_roles.py:137-148` | type-only | Minor | `test_require_declared_output_fields_plugin_returns_typed_plugin` returns the same object and reads two attrs we just set on the literal class body. | Keep the rejection test at 151-159 (real validation logic). Drop the happy-path noop test, or merge into the parametrize at 109-134. |
| `test_passthrough_contract.py:148-176` | sut-mirror (mild) | Minor | Both Hypothesis property tests assert that PassThrough's output equals the input — which is precisely PassThrough's one-line implementation (`return TransformResult.success(deepcopy(row))`). Test mirrors the SUT. | Keep one (deterministic property is genuinely valuable as a regression catch); delete the other. The 47-88 explicit-case tests already cover deep-copy semantics. |
| `test_plugin_protocols.py:6-27` | duplicate | Minor | This is the *whole file*: a single test for "CSVSource validates schema in `__init__`". Already covered by `test_csv_source_contract.py` and source-protocol contract base. | Delete file or fold into `test_csv_source_contract.py`. |

### Duplicate clusters

**Cluster A — Protocol attribute roll-call (sink/source/transform/batch-transform)**
- Files: `test_sink_protocol.py:84-104`, `test_source_protocol.py:69-89`, `test_transform_protocol.py:92-126`, `test_batch_transform_protocol.py:180-214`
- All four assert: `isinstance(x.name, str)`, `issubclass(x.input_schema, PluginSchema)`, etc.
- **Keep**: one parametrized helper test per protocol that does a single Protocol-conformance check; delete the 5–7 individual attribute tests in each base.

**Cluster B — Azure batch-contract scaffolding**
- Files: `test_azure_content_safety_contract.py`, `test_azure_prompt_shield_contract.py`, `test_azure_multi_query_contract.py`
- Each defines an unused `_make_mock_context()` (never referenced after import — dead code masquerading as helper).
- **Keep**: the inheriting `BatchTransformContractTestBase` subclasses (legitimate reuse). **Delete**: the unused `_make_mock_context()` helpers.

**Cluster C — `_importable_from_contracts` (intra-file)**
- File: `test_plugin_schema.py:9-54`
- Six tests doing the same shape: import → instantiate → assert trivial attribute equals what we passed in.
- **Keep**: `test_old_import_path_removed` (real constraint — module deletion check). Collapse rest.

### Delete with no loss of safety

| Test | Why safe to delete |
|------|-------------------|
| `test_plugin_semantics.py::test_compare_semantic_outcome_is_consistent` (318-349) | Re-implements SUT — explicit examples at 256-311 and the closed-membership trip-wires at 38-47, 58-65 backstop. |
| `test_plugin_semantics.py::TestContentKind::test_known_members` (26-32) | `test_membership_is_closed_for_phase_1` (38-47) catches any rename/removal as a value-set diff. |
| `test_plugin_semantics.py::TestTextFraming::test_known_members` (51-56) | Same — closed-membership test at 58-65 backstops. |
| `test_plugin_semantics.py::TestUnknownSemanticPolicy::test_known_members` (69-72), `TestSemanticOutcome::test_known_members` (76-79) | These have *no* closed-membership backstop — but the values are consumed by `compare_semantic` which has explicit-case tests; rename would surface there. **Lower confidence — verify before deleting.** |
| `test_plugin_schema.py:9-25, 26-31, 32-41, 42-54` (5 import smoke tests) | Any downstream consumer test importing these names exercises the re-export path. |
| `test_plugin_schema.py:60-94` (4 single-category error_message tests) | Combined test at 96-114 exercises each branch via the assembled `; `-joined output. |
| `test_sink_protocol.py:320-336` (Protocol annotation introspection) | mypy + a real `inject_write_failure(CSVSink(...))` instantiation prove the same thing. |
| `test_plugin_protocols.py` (entire file, 1 test) | Duplicated by `test_csv_source_contract.py` schema-validation paths. |

### Coverage padding signals

- **Same-file padding pattern**: `test_plugin_schema.py` has six near-identical `*_importable_from_contracts` tests — classic "one test per re-exported symbol" coverage line-up. A single `__all__` assertion would close the same file:line coverage with one test.
- **Type-system duplication**: ~25 `isinstance(x.attr, T)` and `hasattr(x, "attr")` assertions across the three protocol bases. In a strictly-typed codebase running mypy, these test the type checker, not the SUT. CLAUDE.md explicitly bans `hasattr` in production — its presence in tests, while permissible, suggests the same defensive-coding instincts leaking into the test layer.
- **Hypothesis-as-coverage**: `test_passthrough_preserves_arbitrary_dicts` and `test_compare_semantic_outcome_is_consistent` use Hypothesis to generate hundreds of examples, all of which assert the SUT computes what the test re-computes. High example count masks low signal.
- **Dead helper functions**: `_make_mock_context()` defined identically in all three Azure contract files but never referenced. Survived code review three times — suggests reviewers waved through unused scaffolding.

### Out-of-scope observations

- `test_plugin_protocols.py` is a 1-test file with no module docstring beyond a stale opener — file-level orphan that should be folded.
- The `_REGISTRY_LOCK`-acquire patterns at `test_tier_registry.py:284-296, 326-338` use `assert not finished.wait(timeout=0.05)` to verify lock contention. Timing-dependent; potentially flaky on a loaded CI runner.
- `test_tier_registry.py::test_tests_prefix_absent_when_pytest_not_loaded` (243-262) launches a subprocess for what is fundamentally a sys-modules check — heavy machinery for a small invariant, but the pytest-loaded-during-pytest paradox justifies it.

### Confidence Assessment
- **HIGH** for the sut-mirror finding in `test_plugin_semantics.py:318-349` and the import-padding cluster in `test_plugin_schema.py`.
- **MEDIUM** for the Protocol attribute roll-call recommendation — these are defensible as Protocol conformance verification; the suggestion is "consolidate", not "all theatre".
- **LOW–MEDIUM** for deleting the enum `test_known_members` blocks lacking a closed-membership backstop (UnknownSemanticPolicy, SemanticOutcome) — verify downstream coverage before deleting.

### Risk Assessment
- **Quality Risk: LOW.** The chunk is mostly sound; theatre is concentrated in two files and is largely redundant rather than misleading.
- **Primary Risk**: deleting the enum value-mirror tests without confirming downstream coverage could let a silent enum-value rename slip past the closed-membership trip-wire that already exists.

---

## 3. axiom-python-engineering:python-code-reviewer

**Verdict.** The suite's structural skeleton is sound: Hypothesis property tests are wired correctly, the snapshot/restore fixtures show real discipline, and the tier-registry threading tests are technically sophisticated. However, a persistent `Mock()` without `spec=` problem undermines the entire transform-contract subsuite, and a cluster of no-assertion "idempotency" tests and banned `hasattr` calls dilute confidence in the sink and source protocol contracts. The suite is mixed health — structurally coherent at the top level, fragile at the plugin-contract boundaries.

### Findings

| file:line | smell | sev | rationale | recommendation |
|-----------|-------|-----|-----------|----------------|
| `transform_contracts/test_web_scrape_contract.py:30-42` | Mock without spec= | High | `_create_mock_http_response()` returns bare `Mock()` for `response`, `response.request`, `response.raise_for_status`. Any attribute access silently succeeds; refactors to the real HTTP response shape go undetected. | Add `spec=requests.Response` (or the real HTTP client response class) to every `Mock()` in the helper. |
| `transform_contracts/test_azure_content_safety_contract.py:34-51` | Mock without spec= | High | `_make_mock_context()` and `_create_mock_http_response()` both use bare `Mock()`. Same fragility as web_scrape. | Spec every mock to the concrete class it stands in for. |
| `transform_contracts/test_azure_prompt_shield_contract.py:30-47` | Mock without spec= + dead helper | High | Bare `Mock()` throughout; `_make_mock_context()` is defined but never referenced in the test class. | Spec mocks; delete the dead helper. |
| `transform_contracts/test_azure_multi_query_contract.py:26-42` | Mock without spec= + dead helper | High | `_make_mock_context()` defined but unused in its test class. `_make_mock_response()` is bare `Mock()`. | Spec mocks; delete unused helper. |
| `transform_contracts/test_batch_transform_protocol.py:231,248` | Bare `pytest.raises` multi-type tuple, no `match=` | High | `pytest.raises((RuntimeError, AttributeError, ValueError))` — three unrelated exception types, no message constraint. Any exception from any of those three types passes silently, including ones with wrong messages. | Narrow to the single exception type the SUT is specified to raise; add `match=` for the expected message. |
| `transform_contracts/test_batch_transform_protocol.py:236-237` | `contextlib.suppress(Exception)` in teardown | High | `with contextlib.suppress(Exception): batch_transform.close()` — teardown silently swallows contract violations. If `close()` raises unexpectedly, the test suite never knows. | Let teardown raise, or use a `pytest.raises` scoped fixture if `close()` is legitimately expected to fail in that scenario. |
| `transform_contracts/test_transform_protocol.py:94,110,115,120,125,154,198,203,208,213` | `hasattr` — unconditionally banned (CLAUDE.md) | High | Ten call sites. CLAUDE.md: "hasattr is unconditionally banned — it swallows all exceptions from `@property` getters." The `isinstance()` check that immediately follows each `hasattr` makes the `hasattr` dead code in every case. | Delete the `hasattr` guards; keep only the `isinstance` checks. |
| `transform_contracts/test_batch_transform_protocol.py:182,198,203,208,213` | Same `hasattr` pattern | High | Same as above; five instances in the batch variant. | Same fix: delete the `hasattr` lines. |
| `sink_contracts/test_sink_protocol.py:320,336` | `hasattr` on protocol class | High | `assert hasattr(SinkProtocol, "configure_for_resume")` — tests that an attribute name exists on the class. Banned by CLAUDE.md; also fragile (any spurious attribute satisfies it). | Import the protocol and call a method, or use `isinstance` against a concrete stub. |
| `source_contracts/test_source_protocol.py:95-99` | `hasattr` for iterator duck-typing | Med | `hasattr(result, "__iter__")` and `hasattr(result, "__next__")` — banned pattern and weaker than `isinstance(result, Iterator)`. | Replace with `from collections.abc import Iterator; assert isinstance(result, Iterator)` or simply consume the result. |
| `contracts/test_plugin_context_recording.py:169-170,188` | Mock tautology | Med | `mock_landscape.record_transform_error.return_value = "terr_abc123"` then `assert token.error_id == "terr_abc123"`. This asserts the mock library routed a return value, not that the SUT constructed `error_id` correctly. | Assert the construction logic by checking what `record_transform_error` was called with (already done at lines 189-195); delete the `error_id == mock_return_value` assertion. |
| `contracts/test_tier_registry_migration.py:53-72` | Inline private-flag mutation without fixture | Med | Direct `tier_registry._FROZEN = False` with manual rollback in `finally`, duplicating what `test_tier_registry.py`'s `autouse` fixture already handles cleanly. | Extract the `_reset_registry` fixture from `test_tier_registry.py` into a `conftest.py` shared by both files and remove the inline mutation. |
| `sink_contracts/test_sink_protocol.py:207-220,234-250` | No-assertion idempotency tests | Low | `test_flush_is_idempotent` calls flush three times; `test_on_start_does_not_raise` calls on_start once. Neither asserts anything. The implicit contract is "doesn't raise," which is better than nothing but obscures intent. | Add explicit assertions (e.g., call count via a spy, or a sentinel return value), or add a one-line comment `# contract: idempotent, no exception is the requirement` to make intent discoverable. |
| `source_contracts/test_source_protocol.py:136-155` | Same no-assertion pattern | Low | Identical to sink protocol idempotency tests. | Same fix. |
| `transform_contracts/test_azure_multi_query_contract.py:45` | Module-level `autouse` fixture | Low | `@pytest.fixture(autouse=True)` at module scope patches `mock_azure_openai` for every test in the file including `TestMultiQueryLLMSpecific`, which may or may not want it. | Scope the fixture to the class that needs it, or make it class-scoped with explicit `autouse=True` inside the class body. |

### Recurring Patterns

- **Bare `Mock()` without `spec=` in plugin contract tests.** All four Azure/web-scrape transform contract files share this. A one-time `spec=` audit across the contract subsuite would eliminate it.
- **`hasattr` guards before `isinstance` checks.** Fifteen instances across two files; the `hasattr` is always dead code (the `isinstance` handles the same guard). Both files were likely written together and the pattern copied.
- **No-assertion idempotency tests.** Sink and source protocol files both have "call it N times; implicit pass" tests. Intent is not communicated; silent regressions are possible if a method starts raising.
- **Dead helper functions.** `_make_mock_context()` in both test_azure_prompt_shield_contract.py and test_azure_multi_query_contract.py are defined but never called. Suggests copy-paste from a template that was later refactored.
- **Private-flag mutation without fixture isolation.** `test_tier_registry_migration.py` manually manages `_FROZEN` state that `test_tier_registry.py` already handles correctly via a reusable `autouse` fixture.

### Delete / Rewrite Candidates

- **Delete:** All fifteen `hasattr(...)` lines in `test_transform_protocol.py` and `test_batch_transform_protocol.py`. The `isinstance` checks that follow are sufficient; the `hasattr` lines are dead code and violate project policy.
- **Delete:** `_make_mock_context()` in `test_azure_prompt_shield_contract.py:30-47` and `test_azure_multi_query_contract.py:35-42` — unused functions.
- **Rewrite:** `test_batch_transform_protocol.py:231,248` — the multi-type `pytest.raises` tuples should become single-exception assertions with `match=`.
- **Rewrite:** `test_tier_registry_migration.py:53-72` — inline flag mutation should be extracted into a shared conftest fixture.

### Pytest Hygiene Observations

- The `_reset_registry` autouse fixture in `test_tier_registry.py` is a good pattern; it should be in `conftest.py` so `test_tier_registry_migration.py` can reuse it instead of reinventing it.
- The module-level `autouse` fixture in `test_azure_multi_query_contract.py` will silently affect all test classes in the file. Class-scoped autouse fixtures are safer in multi-class modules.
- Hypothesis settings (deadline, suppress_health_check) in `test_registry_snapshot_property.py` are appropriate and well-applied.

### Out-of-Scope Observations

- `transform_contracts/test_transform_protocol.py:196-198` has `# type: ignore[unreachable]` on the multi-row emit branch of the production contract. The `unreachable` annotation means mypy believes that branch cannot execute — worth investigating whether the production `contracts/` type narrowing is correct.
- `mock_landscape.landscape = Mock()` at `test_batch_transform_protocol.py:129-141` creates an unspecced sub-mock on an outer spec'd mock, partially undermining the spec's value for that attribute path.

### Confidence Assessment

High (0.88). All 25 files were read directly. Findings are grounded in exact line numbers. The CLAUDE.md `hasattr` ban is unambiguous. The `pytest.raises` tuple and `contextlib.suppress` findings are textually confirmed.

### Risk Assessment

The bare-Mock findings are the highest risk: a refactor to any Azure client response shape or plugin context protocol will not fail any of these tests. The `contextlib.suppress(Exception)` teardown is operationally risky — it can hide real contract violations from CI. The `hasattr` violations are policy violations but have low runtime risk since the `isinstance` guards cover the same paths.

---

## 4. pr-review-toolkit:pr-test-analyzer

### Chunk verdict — U-CONTRACTS-1: Plugin & registry contracts

**Verdict: Adequate, leaning Strong.** The chunk is structurally well-organized: a strong abstract-base-class scaffold (`Sink/Source/Transform/BatchTransformContractTestBase`) is reused by every plugin-specific file, so cross-plugin contract conformance is uniformly checked. Tier-registry/decoration tests are precise and exercise the right crash paths (frozen-after-bootstrap, plugin-module spoofing, lock contention). However, the chunk is **light on negative scenarios at the plugin level**: most per-plugin contract files only spin up the happy-path subclass and inherit attribute checks, omitting *plugin-specific* rejection paths (oversize input, malformed external response, audit-recorder silence, retryable-vs-fatal classification). The most dangerous gaps: (a) Azure transform contract files never exercise an *unsafe/blocked* response — only happy-path "safe" mocks; (b) no contract test asserts the recorder was actually called for a successful sink/source/transform write; (c) `TransformContractTestBase` accepts both `success` and `error` outcomes for happy-path inputs, so a transform that errors on every valid input would still pass most "process returns TransformResult" tests.

### Per-file scenario table

| File | Primary claim | Covered | Missing (≤3) |
|------|---------------|---------|--------------|
| `sink_contracts/test_csv_sink_contract.py` | CSVSink honours SinkProtocol; hash matches bytes; quoting | hash/size match file; append; quoting commas/quotes/newlines; determinism property | recorder-was-called assertion on write; partial-write/disk-full crash path; mode="write" vs "append" race when file exists |
| `sink_contracts/test_sink_protocol.py` | SinkProtocol attribute + write/lifecycle contract base | attribute presence; SHA-256 shape; idempotent flush/close; empty-batch | wrong-type rows cause crash (Tier-2 plugin contract); on_complete-after-error; supports_resume actually works (only annotation checked L320) |
| `source_contracts/test_csv_source_contract.py` | CSVSource honours SourceProtocol; quarantine/discard/file-not-found | quarantine yields with error; discard records; FileNotFoundError; delimiter; empty/header-only | malformed rows recorder pop_pending linkage assertion; determinism across runs of same file; encoding errors / BOM |
| `source_contracts/test_source_protocol.py` | Source attribute + load/lifecycle contract base | attribute presence; load yields SourceRow; quarantined-row error/destination; close idempotency | source crashes when ctx.landscape is None for quarantine path; load called twice on same source after exhaust |
| `test_plugin_assistance.py` | PluginAssistance/Example deep-freeze + tuple coercion | mappingproxy on dict fields; None preserved; list→tuple; original mutation isolated | issue_code validation (any string accepted); examples with mutable nested values inside before/after dicts |
| `test_plugin_context_recording.py` | record_validation_error / record_transform_error guards + delegation | landscape=None raises; node_id=None raises; row_id derivation (id/hash/repr); quarantine pop_pending; NaN row equality | record_call_error / record_aggregation_error counterparts; double-pop returns None; landscape that raises during record propagates |
| `test_plugin_protocols.py` | Source validates output_schema during __init__ | valid construct; bad fields syntax rejected | unknown mode rejected; missing required schema key; on_validation_failure typo |
| `test_plugin_roles.py` | source/sink role helpers reject cross-role + non-frozenset | inheritance honoured; cross-role rejected; non-frozenset declared_input_fields rejected | declared_required_fields with non-frozenset; empty-frozenset accepted-or-rejected policy; helpers when name attr missing |
| `test_plugin_schema.py` | PluginSchema importable from contracts; CompatibilityResult.error_message formatting | importability of 5 names; combined-error joining; old path removed | check_compatibility on actually-incompatible schemas (only compatible=True tested L52); validate_row with missing field |
| `test_plugin_semantics_imports.py` | L0 purity of plugin_semantics + plugin_assistance | AST scan for L1+ imports | TYPE_CHECKING-only imports excluded? not asserted |
| `test_plugin_semantics.py` | enums closed; contracts immutable; compare_semantic outcomes | enum closure; FrozenInstanceError; coercion; SATISFIED/CONFLICT/UNKNOWN; Hypothesis property | severity values vocabulary closed; compare_semantic with field_name mismatch; unknown_policy=ALLOW behaviour |
| `test_registry_primitive.py` | FrozenRegistry write_unfrozen + freeze waits for in-flight lock | post-freeze raise; freeze blocks during mutation lock | write_unfrozen with no error_factory; nested write_unfrozen reentry; freeze() called twice |
| `test_registry_snapshot_property.py` | snapshot/restore round-trips global + per-site + frozen flag | identity preserved; per-site invariant; frozen flag invariant | restore from foreign snapshot (different process); snapshot/restore under concurrent register |
| `test_tier_decoration_scanner.py` | CI scanner accepts decorated/justified, flags missing/empty | compliant decorator; missing decoration; Violation suffix; TIER-2 with/without justification; qualified decorator; unreadable file | decorator with empty caller_module (covered in tier_registry, not scanner); allowlist behaviour |
| `test_tier_registry_migration.py` | migration parity + live view + no snapshot import | 4 pre-migration members still tier-1; PluginContractViolation excluded; live view; repo audit for forbidden import | reverse: late-deregister also visible; live view from threads |
| `test_tier_registry.py` | @tier_1_error decorator + registry crash paths | decorator semantics; reason validation; freeze guard; module spoof; lock contention; idempotency | concurrent decoration of same class from two threads; `tier_1_reason` for non-registered class |
| `transform_contracts/test_transform_protocol.py` | TransformProtocol attribute/process/lifecycle base | attribute presence; result type; success-data shape; deterministic property; error-result reason+retryable | recorder-was-called for transform call; transform raising (not returning) error — no test that crash propagates |
| `transform_contracts/test_batch_transform_protocol.py` | BatchTransformMixin contract: connect→accept→OutputPort FIFO | connect_output required; double-connect rejected; ctx.token required; FIFO under load; lifecycle | accept after close raises; OutputPort emit failure path; backpressure path (pure non-blocking is asserted only via accept-returns-None) |
| `transform_contracts/test_azure_content_safety_contract.py` | Azure CS honours batch mixin contract via "safe" mock | inherits all batch-contract checks | unsafe/blocked response path; HTTP 4xx/5xx; threshold-mismatch; recorder.record_call assertion |
| `transform_contracts/test_azure_multi_query_contract.py` | LLMTransform query expansion; batch-mixin contract | declared_output_fields contains prefixed; creates_tokens=False; batch contract via Azure mock | malformed JSON response (string content not valid JSON); usage tokens recorded; provider auth failure |
| `transform_contracts/test_azure_prompt_shield_contract.py` | Azure Prompt Shield honours batch contract via "clean" mock | inherits all batch-contract checks | attackDetected=True path; documentsAnalysis with mixed results; HTTP error |
| `transform_contracts/test_keyword_filter_contract.py` | KeywordFilter satisfies process+error contract | happy + blocked-pattern → error | unicode/regex special chars in keyword; case-sensitivity contract; empty `fields` config |
| `transform_contracts/test_passthrough_contract.py` | PassThrough preserves fields, no mutation, deterministic | independent-copy; deep-copy isolation; Hypothesis property | strict-mode `valid_input` is happy-path only (TestPassThroughStrictSchemaContract has no error_input subclass — uses base, never tests strict-rejection); preservation of typing through PipelineRow contract |
| `transform_contracts/test_truncate_contract.py` | Truncate happy + strict-mode rejections | suffix variant; missing-field; type_mismatch (int, None); original_name resolution | truncation actually applied (longest>limit) — only "Short" inputs tested in happy fixtures; max=0 boundary; suffix longer than max |
| `transform_contracts/test_web_scrape_contract.py` | WebScrape honours TransformProtocol via mocked httpx | inherits property+attribute base | non-200 HTTP path; SSRF-rejected URL path; rate-limit acquire=False; payload_store.store failure |

### Critical scenario gaps (chunk-wide)

1. **Recorder-was-called assertions are absent on the happy path.** Every successful sink/source/transform run must record an audit event. Files that never assert this against `factory.plugin_audit_writer()` or `mock_landscape.record_call.assert_called_once`: `sink_contracts/test_csv_sink_contract.py`, `sink_contracts/test_sink_protocol.py` (every base test), `source_contracts/test_source_protocol.py:101-130`, `transform_contracts/test_transform_protocol.py:132-198`, `test_passthrough_contract.py`, `test_truncate_contract.py`, `test_keyword_filter_contract.py`, `test_web_scrape_contract.py`. A transform whose `record_call` is silently no-op would pass every contract test in this chunk. **CLAUDE.md audit-primacy → critical.**
2. **External-failure / negative-response branch missing on Azure mocks.** `test_azure_content_safety_contract.py:23-31` only mocks "all severity 0" and `test_azure_prompt_shield_contract.py:23-27` only mocks `attackDetected: false`. The contract claim that these are "critical for production use" is undercut: a transform that always returns success on `safe` mock data passes — there is no contract pin against the harmful-content branch shape.
3. **Plugin-bug crash propagation untested.** CLAUDE.md: a transform that raises must crash, not be caught. No file asserts that an exception in `transform.process` (or in `OutputPort.emit`) actually propagates uncaught. Belongs in `test_transform_protocol.py` and `test_batch_transform_protocol.py`.
4. **Type-contract-violation crash.** `SinkContractTestBase` (`test_sink_protocol.py:84-184`) tests "MUST have name", but never asserts that a sink receiving wrong-typed pipeline rows crashes (Tier-2). Same for source/transform bases.
5. **Idempotency gaps.** `test_sink_protocol.py:207-232` checks flush/close idempotency but never `write` followed by `close` followed by `write` — does the second write crash? Unspecified.
6. **`TransformContractTestBase.test_process_returns_transform_result` accepts either `success` or `error` for `valid_input`.** L155: `assert result.status in ("success", "error")`. A transform that always errors would pass. The base should split valid_input → must-be-success.

### Low-effort / pointless tests

- `test_plugin_protocols.py:1-27` — entire file is one parametric test fixture; "valid schema accepted, invalid rejected" is duplicated in plenty of CSVSource tests. Marginal.
- `test_plugin_schema.py:9-54` — `test_*_importable_from_contracts` (5 tests). These assert names exist in `__init__.py`. Mypy/import would catch the same.
- `test_plugin_semantics.py:26-32` — `test_known_members` asserts six string equalities of the form `EnumMember.value == "snake_case"`. Tautology against the enum source.
- `test_plugin_assistance.py:54-64` — only checks defaults `()` after construction; a no-op test.
- `test_sink_protocol.py:84-104, 320-336` — "has 'name' attribute / 'determinism' attribute" series.
- `test_source_protocol.py:69-89` — same pattern: attribute-presence checks redundant with type system.
- `test_transform_protocol.py:92-126` — six separate tests, one assertion each, all subsumed by Protocol typing.
- `test_batch_transform_protocol.py:170-214` — duplicates the same attribute-presence litany.
- `test_csv_source_contract.py:79-103` — `test_csv_source_handles_empty_file` only asserts `len(rows) == 0` — does not assert that an audit event recorded "empty source observed."
- `test_keyword_filter_contract.py` — entire file is two fixtures; happy-path inherited; "blocked content" inherited from `TransformErrorContractTestBase`. The file adds no plugin-specific behaviour assertion.
- `test_truncate_contract.py:26-43` — `valid_input` is `"Short title"` against `{title: 20}` — no actual truncation occurs in any happy-path test in the file.

### Tests that exercise nothing the type system doesn't already guarantee

- `test_plugin_schema.py:9-54` — `test_*_importable_from_contracts` (5 tests).
- `test_plugin_semantics.py:34-35` (`test_is_str_subclass`); `:26-32, 51-57, 69-72, 76-79` (member equality tests).
- `test_plugin_assistance.py:26-28` (`test_none_fields_are_left_none` — just asserts `is None` after passing `None`).
- `test_sink_protocol.py:84-104` (`test_sink_has_name`, `_input_schema`, `_determinism`, `_plugin_version`, `_idempotent_flag`).
- `test_sink_protocol.py:320-336` (`test_sink_protocol_requires_supports_resume`, `test_sink_protocol_requires_configure_for_resume`).
- `test_source_protocol.py:69-89` (`test_source_has_name`, `_output_schema`, `_determinism`, `_plugin_version`, `_declared_guaranteed_fields`).
- `test_transform_protocol.py:92-126` (six attribute-presence tests).
- `test_batch_transform_protocol.py:170-214` (six duplicate attribute-presence tests).
- `test_plugin_roles.py:137-148` (`test_require_declared_output_fields_plugin_returns_typed_plugin`).

### Out-of-scope observations

1. `src/elspeth/contracts/plugin_assistance.py` PluginAssistance — `issue_code` is `Optional[str]` accepted as `None` in `test_plugin_assistance.py:60`; no validation that the field is a known issue-code prefix.
2. `test_plugin_semantics_imports.py:34-41` parses imports via AST but does not exclude `if TYPE_CHECKING:` blocks; a TYPE_CHECKING-only import to L1 would be flagged. CLAUDE.md says TYPE_CHECKING is a warning, not failure — the test is stricter than the architecture rule.
3. `test_tier_registry.py:243-262` spawns a subprocess for every run — a fixture-level skip would be cheaper if pytest-xdist multiplies it.
4. `test_csv_sink_contract.py:217-273` Hypothesis strategies filter rows whose `name`/`data` contain `\n`, `,`, `"` — but `TestCSVSinkQuotingCharacters` then asserts those exact characters work. The property test could be widened (with proper quoting expectations) to subsume the example tests.
5. `test_web_scrape_contract.py:99-114` builds a real `Call` object with realistic-looking hashes — the test would silently break if `Call` gained a required field; no factory abstraction.

---

## 5. ordis-quality-engineering:coverage-gap-analyst

### SUT Footprint Summary

| Source Module | Lines | Test File(s) | Coverage Class |
|---|---|---|---|
| `contracts/plugin_context.py` | ~320 | `test_plugin_context_recording.py` | Partial — record_call UNTESTED |
| `contracts/tier_registry.py` | ~230 | `test_tier_registry.py` | Strong |
| `contracts/registry_primitive.py` | ~90 | `test_registry_primitive.py` | Thin — 2 tests only |
| `contracts/plugin_roles.py` | ~180 | `test_plugin_roles.py` | Partial — rejection paths missing |
| `contracts/plugin_semantics.py` | ~165 | `test_plugin_semantics.py` | Behavioral paths covered |
| `contracts/plugin_assistance.py` | ~60 | `test_plugin_assistance.py` | Freeze mechanics covered; secret discipline NOT tested |
| `contracts/declaration_contracts.py` | ~120 | `test_declaration_contracts.py` | Covered |
| `contracts/freeze.py` | ~80 | `test_freeze.py` | Covered |
| `plugins/transforms/web_scrape.py` | ~410 | `test_web_scrape_contract.py` | Happy path only — SSRF boundary untested |
| `plugins/transforms/azure_content_safety.py` | ~260 | `test_azure_content_safety_contract.py` | All-safe path only — rejection untested |
| CI script `enforce_tier_1_decoration.py` | — | `test_tier_decoration_scanner.py` | Subprocess test of CI script, not SUT |
| `contracts/plugin_semantics.py` (import arch) | — | `test_plugin_semantics_imports.py` | AST lint, not behavioral |

### Critical Gaps (Test Immediately)

**Gap C-1: `PluginContext.record_call()` — entire method is untested**

`/home/john/elspeth/src/elspeth/contracts/plugin_context.py`

`record_call()` is the single audit-write path for every external API call in every transform, source, and sink. It contains at minimum five distinct `FrameworkBugError` crash branches: `landscape=None`; both `state_id` and `operation_id` supplied; neither supplied; `state_id=None` after `has_state` returned true; and token mismatch between `ctx.token` and `node_state.token_id`. Every one of these branches represents a framework invariant that, if silently bypassed, would produce an audit trail missing an external call — a direct violation of the Three-Tier trust model and the attributability guarantee.

`test_plugin_context_recording.py` covers `record_validation_error` and `record_transform_error` only. Zero tests exercise `record_call`.

Recommended tests: one unit test per crash branch, plus a happy-path integration test confirming the landscape write actually fires and the returned token is anchored to the correct node state.

**Gap C-2: Azure content safety threshold rejection is untested**

`/home/john/elspeth/src/elspeth/plugins/transforms/azure_content_safety.py`

`test_azure_content_safety_contract.py` uses `_make_safe_response()` exclusively — every category returns severity 0. The entire security purpose of this transform is routing rows to the `on_error` sink when severity meets or exceeds the configured threshold. That path is never executed. A bug in the threshold comparison (off-by-one, wrong comparator, wrong category key) would pass the full test suite.

Recommended tests: severity exactly at threshold (boundary), severity one above threshold, severity below threshold, and at least one HTTP error response to confirm the error-path sink routing logic.

**Gap C-3: `WebScrapeTransform` SSRF boundary and `_final_response_ip` crash sites are untested**

`/home/john/elspeth/src/elspeth/plugins/transforms/web_scrape.py`

`test_web_scrape_contract.py` mocks a 200 response with a valid URL field. Three untested paths carry security-class risk: (1) `validate_url_for_ssrf` rejecting a private IP target — the SSRF boundary that prevents credential exfiltration; (2) missing or wrong-type URL field raising `TypeError` — the input validation path; (3) `_final_response_ip()` raising `FrameworkBugError` when the response object has no `.request`, no `.host`, or a non-IP host — the post-redirect audit integrity path.

Recommended tests: SSRF-blocked URL (assert `SSRFBlockedError` caught and row quarantined), missing URL field, and a monkeypatched response with a stripped request object to exercise the `_final_response_ip` FrameworkBugError.

### High-Risk Gaps (Test This Sprint)

**Gap H-1: `FrozenRegistry.write_unfrozen()` default-RuntimeError branch untested**

`/home/john/elspeth/src/elspeth/contracts/registry_primitive.py`

When `write_unfrozen` is called with no `frozen_error` factory and the registry is frozen, it falls through to `raise RuntimeError(f"{self.name} registry is frozen")`. The two existing tests only cover the custom-factory path (tier_registry supplies `FrameworkBugError`) and the concurrency lock serialisation test. The default branch is the fallback any new registry consumer would hit if they forget to supply a factory.

Recommended test: instantiate a bare `FrozenRegistry`, freeze it, call `write_unfrozen(None)` in a `with` block, assert `RuntimeError` is raised.

**Gap H-2: `plugin_roles.py` rejection paths**

`/home/john/elspeth/src/elspeth/contracts/plugin_roles.py`

Four rejection branches lack test coverage: (1) empty string plugin name passed to `_require_contractable_plugin` — the guard that catches accidentally blank names at registration time; (2) non-string item inside a frozenset passed to `_validated_string_frozenset`; (3) non-bool value for `is_batch_aware` in `require_declared_input_fields_plugin`; (4) non-frozenset passed to `require_declared_output_fields`.

Recommended tests: one parametrised test class per function covering the rejection cases listed above.

### Medium-Risk Gaps

**Gap M-1: `plugin_assistance.py` secret-discipline contract has no runtime or test enforcement**

`/home/john/elspeth/src/elspeth/contracts/plugin_assistance.py`, docstring lines 9-11

The module docstring states assistance fields must not contain URLs, headers, prompts, row data, credentials, or exception strings. There is no runtime guard and no test. The CLAUDE.md notes "Enforcement is by plugin authors and tests (see secret-leakage tests in Phase 3)" — Phase 3 tests do not appear in the 25-file scope. Until those tests exist, the contract is documentation-only.

### Quick Wins

| File | Why Easy | Value |
|---|---|---|
| `test_registry_primitive.py` | Two lines: freeze with no factory, assert `RuntimeError` | Closes H-1; FrozenRegistry is used by all registries |
| `test_plugin_roles.py` — empty name | One parametrised case on existing test class | Closes half of H-2 in minutes |
| `test_azure_content_safety_contract.py` — threshold | Adjust `_make_safe_response` to return severity >= threshold | Closes C-2 boundary path |

### Notable Strengths

`test_tier_registry.py` is genuinely thorough: it covers the module-allowlist guard, plugin-spoof prevention (class `__module__` check separate from caller_module), post-freeze registration, concurrency serialisation for both freeze and registration paths, the pytest-only `tests.` prefix gate including a subprocess test that verifies absence outside pytest, and the empty-`caller_module` rejection. This is the highest-quality test file in the analysed group.

`test_plugin_assistance.py` and `test_freeze.py` correctly test the deep-freeze `__post_init__` contract, including source-dict mutation isolation and list-to-tuple coercion. The immutability surface is well-covered.

### Confidence Assessment

**High** for the three critical gaps — all are confirmed by direct source reading plus advisor review. **Medium** for the H-2 rejection paths — the analysis is based on reading `plugin_roles.py` and `test_plugin_roles.py` and cross-referencing test function names; no execution trace was run. **Medium-low** for M-1 — Phase 3 tests may exist outside the 25-file scope and were not enumerated.

### Risk Assessment

C-1 (`record_call`) is the highest-consequence gap: a regression in any of its crash guards produces a silent audit omission, violating the attributability guarantee that is ELSPETH's core correctness property. C-2 and C-3 are security-boundary gaps; their absence means the security purpose of those transforms is formally unverified by the test suite.

### Information Gaps

1. Whether Phase 3 secret-leakage tests for `plugin_assistance` exist elsewhere in the repo was not verified — only the 25-file scope was examined.
2. `checkpoint_*` methods in `plugin_context.py` were not fully enumerated; checkpoint coverage was treated as out-of-scope per the stated SUT boundary.
3. No execution trace or coverage tool output was available — gap findings are based on source-to-test mapping, not instrumented line hits.

### Caveats

This analysis maps source code to test files by direct inspection. It does not detect parametrised fixture coverage that might exercise record_call indirectly through a higher-level integration harness not in the 25-file scope. The three critical gaps should be treated as confirmed absences pending any such integration harness search.
