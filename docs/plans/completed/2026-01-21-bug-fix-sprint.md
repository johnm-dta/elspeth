# Bug Fix Sprint Plan - 2026-01-21

## Overview

After comprehensive review of all 55 open bugs by parallel agents, we identified:
- **9 bugs FIXED** (moved to `docs/bugs/closed/`)
- **46 bugs remaining** (11 P1, 24 P2, 7 P3, 4 pending)

This plan organizes fixes into logical sprints based on subsystem, dependencies, and effort.

---

## Sprint 1: Quick Wins (1-2 lines each)

**Theme:** Simple fixes that improve code quality with minimal risk.

| Bug | Fix | Files |
|-----|-----|-------|
| `P1-complete-node-state-empty-output-hash` | Change `if output_data` → `if output_data is not None` | `recorder.py:1063-1065` |
| `P1-fork-to-paths-empty-destinations-allowed` | Add `if not paths: raise ValueError(...)` | `routing.py:92-97` |
| `P2-checkpoint-empty-aggregation-state-dropped` | Change `if aggregation_state` → `if aggregation_state is not None` | `checkpoint/manager.py:57` |
| `P2-recorder-export-status-enum-mismatch` | Add `ExportStatus(row.export_status)` coercion | `recorder.py:330,370` |
| `P2-node-state-ordering-missing-attempt` | Add `.order_by(..., attempt)` | `recorder.py:1729` |
| `P2-node-state-terminal-completed-at-not-validated` | Add `if row.completed_at is None: raise ValueError(...)` | `recorder.py:133-139` |
| `P3-cli-run-prints-enum-status` | Change `result['status']` → `result['status'].value` | `cli.py:184` |
| `P3-transforms-package-exports-incomplete` | Add `BatchStats`, `JSONExplode` to `__all__` | `transforms/__init__.py` |

**Estimated effort:** 30 minutes total

---

## Sprint 2: Schema & Validation Hardening

**Theme:** Fix schema compatibility and validation gaps that cause runtime crashes.

| Bug | Fix | Files |
|-----|-----|-------|
| `P1-schema-compatibility-check-fails-on-optional-and-any` | Handle `Any` type in `_types_compatible()` | `data.py:200-228` |
| `P1-schema-config-yaml-example-crashes-parsing` | Add type check before `spec.strip()` | `schema.py:67,182` |
| `P1-jsonsource-jsonl-parse-errors-not-quarantined` | Wrap `json.loads()` in try/except, quarantine on JSONDecodeError | `json_source.py:113` |
| `P2-non-finite-floats-pass-source-validation` | Add `allow_inf_nan=False` to float field validators | `schema_factory.py` |
| `P2-json-explode-iterable-nonstrict-types` | Add explicit `isinstance(value, list)` check | `json_explode.py:128,138` |

**Estimated effort:** 2-3 hours

---

## Sprint 3: Audit Trail Integrity

**Theme:** Ensure all routing and outcomes are properly recorded for auditability.

| Bug | Fix | Files |
|-----|-----|-------|
| `P1-gate-continue-routing-not-recorded` | Record routing event for continue actions | `executors.py:393-408,565-567` |
| `P1-dag-fork-aggregation-drop` | Propagate aggregation flush outputs downstream | `orchestrator.py` |
| `P2-source-quarantine-silent-drop` | Create audit record when quarantine destination missing | `orchestrator.py:733-744` |
| `P2-transform-errors-ambiguous-transform-id` | Use `node_id` instead of `transform.name` | `executors.py:261` |
| `P2-exporter-missing-expand-group-id` | Add `expand_group_id` to token export | `exporter.py:210-221` |
| `P2-error-tables-missing-foreign-keys` | Add ForeignKey constraints to error tables | `schema.py:274,293` |

**Estimated effort:** 3-4 hours

---

## Sprint 4: CLI Fixes

**Theme:** Fix CLI user experience issues.

| Bug | Fix | Files |
|-----|-----|-------|
| `P1-cli-purge-ignores-payload-store-settings` | Use `config.payload_store.base_path` | `cli.py:560-567` |
| `P2-cli-paths-no-tilde-expansion` | Add `.expanduser()` to Path options | `cli.py` (multiple) |
| `P2-cli-run-does-not-close-landscape-db` | Add try/finally with `db.close()` | `cli.py:300-369` |
| `P2-cli-run-rebuilds-unvalidated-graph` | Reuse validated graph or re-validate | `cli.py:150,315` |
| `P3-cli-purge-resume-silently-create-db` | Check file existence before `from_url()` | `cli.py` |

**Estimated effort:** 2 hours

---

## Sprint 5: Sink Schema Handling

**Theme:** Fix sinks that ignore configured schemas.

| Bug | Fix | Files |
|-----|-----|-------|
| `P1-databasesink-schema-inferred-from-first-row` | Use `schema_config` to create table columns | `database_sink.py:91-107` |
| `P1-shape-changing-transforms-output-schema-mismatch` | Support separate `input_schema`/`output_schema` | `field_mapper.py`, `json_explode.py` |
| `P2-csvsink-fieldnames-inferred-from-first-row` | Use `schema_config.fields` for fieldnames | `csv_sink.py:180` |
| `P2-databasesink-if-exists-replace-ignored` | Implement drop/truncate for replace mode | `database_sink.py` |

**Estimated effort:** 3-4 hours

---

## Sprint 6: Thread Safety & Rate Limiting

**Theme:** Fix concurrency issues in rate limiter.

| Bug | Fix | Files |
|-----|-----|-------|
| `P2-rate-limiter-acquire-not-thread-safe-or-atomic` | Add `with self._lock:` to `acquire()` | `limiter.py:191-199` |
| `P2-rate-limiter-suppression-thread-ident-stale` | Clean up stale thread idents periodically | `limiter.py:57-67` |

**Estimated effort:** 1-2 hours

---

## Sprint 7: Data Integrity & Contracts

**Theme:** Fix contract mismatches and data handling issues.

| Bug | Fix | Files |
|-----|-----|-------|
| `P2-checkpoint-contract-created-at-nullable-mismatch` | Align contract with schema (remove `| None`) | `audit.py:303`, `schema.py:317` |
| `P2-artifact-idempotency-key-column-ignored` | Add `idempotency_key` to Artifact contract | `audit.py:222-233` |
| `P2-forked-token-row-data-shallow-copy-leaks-nested-mutations` | Use `copy.deepcopy()` for forked tokens | `tokens.py:148` |
| `P2-contracts-config-reexport-breaks-leaf-boundary` | Remove core.config imports from contracts | `contracts/config.py` |
| `P2-retention-purge-ignores-call-and-reason-payload-refs` | Include call/routing payload refs in purge query | `purge.py:71-112` |

**Estimated effort:** 2-3 hours

---

## Sprint 8: Cleanup & Tech Debt

**Theme:** Lower-priority cleanup items.

| Bug | Fix | Files |
|-----|-----|-------|
| `P2-node-metadata-hardcoded` | Extract version from config gates | DAG builder |
| `P2-gate-route-destination-name-validation-mismatch` | Unify identifier validation | `config.py:229` |
| `P3-landscape-models-duplication-drift` | Delete or align legacy models.py | `models.py` |
| `P3-plugin-spec-schema-hash-missing-for-config-schemas` | Get schema from instance if class attr is None | `manager.py:76-103` |
| `P3-node-repository-drops-schema-config` | Include schema_mode/fields in load() | `repositories.py:69-86` |
| `P3-defensive-whitelist-review` | Re-audit defensive patterns | Multiple files |

**Estimated effort:** 3-4 hours

---

## Sprint 9: Feature Completion (Larger Items)

**Theme:** Placeholder features that need full implementation.

| Bug | Scope | Files |
|-----|-------|-------|
| `P1-cli-explain-is-placeholder` | Implement full explain TUI with data loading | `cli.py`, `explain_app.py` |
| `P2-exporter-n-plus-one-queries` | Batch loading for export queries | `exporter.py` |
| `P2-exporter-missing-config-in-export` | Include config JSON in export records | `exporter.py` |

**Estimated effort:** 4-6 hours

---

## Pending Bugs (Need Investigation)

| Bug | Status | Next Step |
|-----|--------|-----------|
| `P1-routing-copy-ignored` | NEEDS_VERIFICATION | Write runtime test for COPY vs MOVE semantics |
| `P2-plugin-gate-graph-mismatch` | STILL_OPEN | Decide: remove plugin gate support or fix graph builder |

---

## Recommended Execution Order

1. **Sprint 1** (Quick Wins) - Low risk, high value, builds momentum
2. **Sprint 2** (Schema Validation) - Prevents runtime crashes
3. **Sprint 3** (Audit Trail) - Core to ELSPETH's auditability promise
4. **Sprint 4** (CLI) - User-facing improvements
5. **Sprint 5** (Sink Schemas) - Data integrity
6. **Sprint 6** (Thread Safety) - Prevents subtle concurrency bugs
7. **Sprint 7** (Contracts) - Internal consistency
8. **Sprint 8** (Cleanup) - Tech debt reduction
9. **Sprint 9** (Features) - Larger items, do when time permits

---

## Success Metrics

- [ ] All P1 bugs resolved
- [ ] All truthiness bugs fixed (3 instances)
- [ ] All schema validation gaps closed
- [ ] CLI DB lifecycle properly managed
- [ ] Test coverage added for each fix
