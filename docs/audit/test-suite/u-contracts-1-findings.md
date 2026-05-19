# U-CONTRACTS-1 — Synthesised Findings

**Scope:** 25 files in `tests/unit/contracts/` covering plugin protocols, plugin/tier registry, and per-plugin contract bases.
**Method:** 5 specialist agents in parallel, each with a different lens.
**Date:** 2026-05-06.

## Files in scope

```
tests/unit/contracts/sink_contracts/test_csv_sink_contract.py
tests/unit/contracts/sink_contracts/test_sink_protocol.py
tests/unit/contracts/source_contracts/test_csv_source_contract.py
tests/unit/contracts/source_contracts/test_source_protocol.py
tests/unit/contracts/test_plugin_assistance.py
tests/unit/contracts/test_plugin_context_recording.py
tests/unit/contracts/test_plugin_protocols.py
tests/unit/contracts/test_plugin_roles.py
tests/unit/contracts/test_plugin_schema.py
tests/unit/contracts/test_plugin_semantics_imports.py
tests/unit/contracts/test_plugin_semantics.py
tests/unit/contracts/test_registry_primitive.py
tests/unit/contracts/test_registry_snapshot_property.py
tests/unit/contracts/test_tier_decoration_scanner.py
tests/unit/contracts/test_tier_registry_migration.py
tests/unit/contracts/test_tier_registry.py
tests/unit/contracts/transform_contracts/test_azure_content_safety_contract.py
tests/unit/contracts/transform_contracts/test_azure_multi_query_contract.py
tests/unit/contracts/transform_contracts/test_azure_prompt_shield_contract.py
tests/unit/contracts/transform_contracts/test_batch_transform_protocol.py
tests/unit/contracts/transform_contracts/test_keyword_filter_contract.py
tests/unit/contracts/transform_contracts/test_passthrough_contract.py
tests/unit/contracts/transform_contracts/test_transform_protocol.py
tests/unit/contracts/transform_contracts/test_truncate_contract.py
tests/unit/contracts/transform_contracts/test_web_scrape_contract.py
```

## Verdict

**Health: Mixed-with-bright-spots.** Genuinely strong tests for tier registry, deep-freeze invariants, and registry concurrency live alongside ~30+ tautological/duplicate tests in the protocol-base files and a dead-code layer in the Azure contract suite. **Three production-code crash paths sit entirely outside the audit fence** — invisible to every test in the chunk.

Per-lens verdicts: Mixed (anti-patterns), Some-theatre-concentrated (VER/VAL), Mixed (Python smells), Adequate-leaning-Strong (scenario coverage), Strong-with-three-critical-gaps (SUT coverage).

## Convergent findings (≥2 agents agree — high confidence)

### CONV-1 — Protocol attribute roll-call (tautology) — 4/5 agents

`test_sink_protocol.py:84-104`, `test_source_protocol.py:69-89`, `test_transform_protocol.py:92-126`, `test_batch_transform_protocol.py:180-214` collectively contain ~25 `isinstance(plugin.attr, T)` and `hasattr(plugin, "attr")` assertions that re-state Protocol declarations. Mypy enforces these statically. They detect no bugs and break on rename. The batch-transform block is a literal duplicate of the transform block (BatchTransformMixin inherits TransformProtocol).

**Recommendation:** consolidate to one `assert isinstance(real_plugin, ProtocolType)` per protocol; delete the 5–7 individual attribute tests in each base; delete the entire batch-protocol attribute block.

### CONV-2 — `hasattr()` in tests violates CLAUDE.md — 2/5 agents (with citations)

CLAUDE.md unconditionally bans `hasattr` (it swallows `@property` exceptions). Tests violate it 15× in `test_transform_protocol.py` (lines 94, 110, 115, 120, 125, 154, 198, 203, 208, 213) and `test_batch_transform_protocol.py` (lines 182, 198, 203, 208, 213), plus `test_sink_protocol.py:320, 336` uses `hasattr(SinkProtocol, ...)` and `__annotations__` introspection. In every case, an `isinstance` immediately follows, making the `hasattr` dead code.

**Recommendation:** delete every `hasattr(...)` line; consider extending `scripts/cicd/enforce_tier_model.py` to ban it in tests.

### CONV-3 — Dead `_make_mock_context()` helpers in 3 Azure files — 3/5 agents

`test_azure_content_safety_contract.py:45-51`, `test_azure_prompt_shield_contract.py:41-47`, `test_azure_multi_query_contract.py:35-42` each define `_make_mock_context()` that is **never called**. Survived code review three times — copy-paste artefact.

**Recommendation:** delete; flag for code-review-effectiveness signal.

### CONV-4 — Import-shape smoke tests in `test_plugin_schema.py:9-54` — 3/5 agents

Six `test_*_importable_from_contracts` tests assert that `from elspeth.contracts import X` succeeds. Any downstream test that imports those names already proves this. Pure coverage padding.

**Recommendation:** collapse to a single `__all__` assertion; keep `test_old_import_path_removed:127-134` (real legacy-removal guard).

### CONV-5 — Enum value-mirror tests in `test_plugin_semantics.py` — 3/5 agents

Lines `:26-32, :51-56, :69-72, :76-79`: `assert ContentKind.PLAIN_TEXT.value == "plain_text"` etc. Both sides of the assertion are the same source of truth. Closed-membership tests at `:38-47, :58-65` already trip-wire any rename.

**Recommendation:** delete the per-member value asserts for `ContentKind` and `TextFraming` (closed-membership backstop exists). Verify before deleting `UnknownSemanticPolicy` and `SemanticOutcome` member tests — those lack a closed-membership backstop in this file.

### CONV-6 — Lifecycle smoke-by-omission (no-assertion idempotency tests) — 2/5 agents

`test_sink_protocol.py:207-220, 222-232`; `test_source_protocol.py:136-155`; `test_transform_protocol.py:204-242`. Tests call `flush()`/`close()`/`on_start()` repeatedly and assert nothing — pass = no exception. A method that silently drops state passes these.

**Recommendation:** add post-condition assertions (file size unchanged, call count via spy, sentinel return value) or document the contract explicitly with a comment.

## Single-lens findings worth surfacing

### SOLO-1 — `compare_semantic` Hypothesis test re-implements the SUT (Critical theatre)

Found by **qa-analyst**. `test_plugin_semantics.py:318-349` runs hundreds of Hypothesis examples; the test body re-implements `compare_semantic`'s exact branching, then asserts the SUT agrees with the reimplementation. **Two copies of the same spec — cannot detect a bug, only divergence.**

**Recommendation:** rewrite as an *independent* invariant (e.g., monotonicity: enlarging accepted sets never turns SATISFIED into CONFLICT) or delete (explicit cases at `:256-311` cover behaviour).

### SOLO-2 — `pytest.raises((RuntimeError, AttributeError, ValueError))` with no `match=` (Critical Python smell)

Found by **python-code-reviewer**. `test_batch_transform_protocol.py:231, 248`. Three unrelated exception types accepted with no message constraint. Any of those three from anywhere passes silently.

**Recommendation:** narrow to the single specified exception with `match=`.

### SOLO-3 — `contextlib.suppress(Exception)` in teardown (Critical Python smell)

Found by **python-code-reviewer**. `test_batch_transform_protocol.py:236-237`. Teardown silently swallows contract violations during `close()`.

**Recommendation:** let teardown raise, or scope a `pytest.raises` if `close()` is legitimately expected to fail.

### SOLO-4 — Bare `Mock()` without `spec=` across 4 transform_contracts files (High Python smell)

Found by **python-code-reviewer**. `test_web_scrape_contract.py:30-42`, `test_azure_content_safety_contract.py:34-51`, `test_azure_prompt_shield_contract.py:30-47`, `test_azure_multi_query_contract.py:26-42`. Refactors to the real HTTP response shape go undetected.

**Recommendation:** spec every mock to its concrete class — one-time mechanical sweep.

### SOLO-5 — `TransformContractTestBase` happy-path is too permissive (High scenario gap)

Found by **pr-test-analyzer**. `test_transform_protocol.py:155` asserts `result.status in ("success", "error")` for `valid_input`. **A transform that always errors on every valid input would pass this test.** Truncate test inputs are all already-short strings, so the truncation logic itself is never exercised on the happy path.

**Recommendation:** split base into `valid_input` (must-be-success) vs `error_input` (must-be-error); fix per-plugin valid_input fixtures to actually exercise the transform.

### SOLO-6 — Mock tautology in `test_plugin_context_recording.py`

Found by **python-code-reviewer**. Lines 169-170, 188: `mock_landscape.record_transform_error.return_value = "terr_abc123"` then `assert token.error_id == "terr_abc123"`. Asserts the mock library routed a return value, not that the SUT constructed `error_id` correctly. Construction logic is checked at lines 189-195 — the value-equality assertion is redundant.

**Recommendation:** delete the value-equality assertion; keep the call-arguments assertion.

### SOLO-7 — Inline private-flag mutation in `test_tier_registry_migration.py:53-72`

Found by **python-code-reviewer**. Direct `tier_registry._FROZEN = False` with manual rollback in `finally`, duplicating what `test_tier_registry.py`'s `autouse` `_reset_registry` fixture handles cleanly.

**Recommendation:** extract `_reset_registry` into a shared `conftest.py`; remove the inline mutation.

## Critical production-code gaps (from coverage-gap-analyst)

These are bugs waiting to happen, not bad tests. Filed as separate issues — see "Filed issues" below.

| Gap | Severity | Why it matters |
|---|---|---|
| `PluginContext.record_call()` — entire method untested (5 crash branches) | **Critical** | Single audit-write path for every external call in every plugin. Silent regression = audit-trail omission = attributability violation. |
| Azure content safety: threshold-rejection path never tested (only mocks all-severity-0) | **Critical (Security)** | The transform's entire security purpose is routing harmful content to `on_error`. That routing is never exercised. |
| `WebScrapeTransform` SSRF boundary + `_final_response_ip` crashes untested | **Critical (Security)** | SSRF rejection prevents credential exfiltration; `_final_response_ip` covers post-redirect audit integrity. |
| `FrozenRegistry.write_unfrozen()` default `RuntimeError` branch untested | High | Fallback for any new registry consumer that forgets a custom factory. |
| `plugin_roles.py` — 4 rejection branches untested | High | Validation contract whose entire value is rejection. |
| Recorder-was-called assertions absent on the happy path of every contract test | High | A transform whose `record_call` is silently no-op would pass every test in this chunk. |

## Top deletion candidates (consensus order)

| # | Target | Lines | Confidence | Why safe |
|---|---|---|---|---|
| 1 | `test_plugin_schema.py:9-54` (5 import smoke tests) | ~45 | High | Any downstream import covers it |
| 2 | `test_batch_transform_protocol.py:180-214` (attribute block) | ~35 | High | Inherited from `test_transform_protocol.py:92-126` |
| 3 | All 15× `hasattr(...)` lines in transform/sink protocol files | 15 | High (CLAUDE.md mandates) | Banned pattern; `isinstance` on next line covers |
| 4 | `_make_mock_context()` in 3 Azure files | ~22 | High | Never referenced |
| 5 | `test_plugin_semantics.py:26-32, :51-56` (enum value mirrors with backstop) | ~14 | High | Closed-membership tests trip-wire renames |
| 6 | `test_plugin_protocols.py` (entire file, 1 test) | ~27 | Medium | Duplicated by `test_csv_source_contract.py` |
| 7 | `test_plugin_semantics.py:318-349` (compare_semantic Hypothesis SUT-mirror) | ~32 | Medium | Re-implements SUT; explicit cases backstop |
| 8 | `test_sink_protocol.py:320-336` (Protocol annotation introspection via banned `hasattr`) | ~17 | Medium | Mypy + `isinstance` on real instance covers |
| 9 | `test_plugin_schema.py:127-134` (`test_old_import_path_removed`) | ~7 | Medium | One-time migration guard; no-legacy-code policy |
| 10 | `test_tier_registry.py:152-170` (in-process pytest tautology half) | ~18 | Medium | Subprocess version at `:243-262` is the actual proof |

**Total deletable: ~232 lines / ~30 test bodies** with no loss of bug-detection capability.

## Top "add immediately" candidates

1. **`record_call()` happy path + 5 crash branches** in `test_plugin_context_recording.py`.
2. **Azure content safety threshold rejection** (severity at/above/below threshold).
3. **WebScrape SSRF rejection + `_final_response_ip` FrameworkBugError**.
4. **`record_call.assert_called_once_with(...)` on every contract base happy path** — closes the audit-primacy regression hole.
5. **Plugin-bug crash propagation** (transform raises → pipeline crashes, not caught).

## Notable strengths (do not regress)

- `test_tier_registry.py` is genuinely thorough: covers module-allowlist guard, plugin-spoof prevention, post-freeze registration, concurrency serialisation, the pytest-only `tests.` prefix gate (including subprocess test for outside-pytest verification), and empty-`caller_module` rejection.
- `test_plugin_assistance.py` and `test_freeze.py` correctly test the deep-freeze `__post_init__` contract, including source-dict mutation isolation and list-to-tuple coercion.
- `test_registry_snapshot_property.py` Hypothesis settings (deadline, suppress_health_check) are appropriate and well-applied.
- `test_csv_sink_contract.py` quoting tests are real behavioural coverage (commas, quotes, newlines).

## Filed filigree issues

| ID | Title | Priority | Labels |
|---|---|---|---|
| `elspeth-f92ba560ad` | PluginContext.record_call() — entire method untested | P0 bug | audit-integrity, test-gap, from-test-audit |
| `elspeth-8d5558dc25` | Azure content safety — threshold rejection path never tested | P0 bug | security, test-gap, from-test-audit |
| `elspeth-7b7fe68836` | WebScrapeTransform — SSRF boundary and _final_response_ip untested | P0 bug | security, test-gap, from-test-audit |

The other findings in this document (deletable test bodies, scenario-base permissiveness, mock fragility, recorder-was-called assertions) are not yet filed — they are test-quality cleanups, not bugs-waiting-to-happen, and are best handled as a single sweep PR rather than per-test issues.

## Out-of-scope observations (production-code, not test quality)

1. `test_tier_registry_migration.py:40-72` — direct `tier_registry._FROZEN` mutation suggests a sanctioned test-mode hook on the SUT would be cleaner.
2. `test_batch_transform_protocol.py:386, 473` — `wait_for_results` budgets of 10–60s are loose enough to hide threading regressions silently.
3. `test_plugin_semantics_imports.py:34-41` — AST parse does not exclude `if TYPE_CHECKING:` blocks; stricter than CLAUDE.md (warning, not failure).
4. `src/elspeth/contracts/plugin_assistance.py` — secret-discipline contract documented but not runtime-enforced and not tested in this scope.
5. `transform_contracts/test_transform_protocol.py:196-198` — `# type: ignore[unreachable]` annotation suggests possible production type-narrowing issue worth investigating.
