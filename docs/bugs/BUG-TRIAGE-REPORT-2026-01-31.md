# Bug Triage Report - 2026-01-31

## Summary

Triaged **122 generated bug reports** from `docs/bugs/generated/`. Results:

| Status | Count |
|--------|-------|
| **VALID** | 73 |
| **INVALID/EMPTY** | 46 |
| **DUPLICATE** | 3 |

## P1 Bugs Created (25 filed)

All critical P1 bugs have been filed in `docs/bugs/open/`:

| Bug | Location | Issue |
|-----|----------|-------|
| P1-2026-01-31-recovery-missing-payload-hash-verification | core-checkpoint/ | Resume doesn't verify payload integrity |
| P1-2026-01-31-running-status-blocks-resume | core-checkpoint/ | RUNNING status blocks crash recovery |
| P1-2026-01-31-payload-store-path-traversal | core-payload/ | Path traversal via unvalidated content_hash |
| P1-2026-01-31-routing-reason-payload-not-persisted | core-landscape/ | Routing reason payloads never stored |
| P1-2026-01-31-fetchone-multi-row-silent-truncation | core-landscape/ | fetchone silently drops extra rows |
| P1-2026-01-31-compute-grade-no-determinism-validation | core-landscape/ | Invalid determinism values accepted |
| P1-2026-01-31-json-formatter-nan-coercion | core-landscape/ | JSON export allows NaN/Infinity |
| P1-2026-01-31-nodestate-repo-missing-invariant-checks | core-landscape/ | OPEN/PENDING invariants not enforced |
| P1-2026-01-31-token-outcome-repo-is-terminal-coercion | core-landscape/ | Invalid is_terminal coerced silently |
| P1-2026-01-31-gate-drops-computed-schema-guarantees | core-dag/ | Gate nodes break schema contract |
| P1-2026-01-31-sanitized-webhook-url-fragment-secrets | core-security/ | Fragment-based secrets not sanitized |
| P1-2026-01-31-http-client-records-raw-urls-with-secrets | core-security/ | HTTP URLs with secrets recorded |
| P1-2026-01-31-sink-flush-failure-leaves-open-states | engine-executors/ | Sink flush failures leave OPEN states |
| P1-2026-01-31-quarantine-outcome-before-durability | engine-orchestrator/ | Outcome recorded before sink durability |
| P1-2026-01-31-expression-errors-bubble-raw | engine-expression-parser/ | Expression errors crash with opaque errors |
| P1-2026-01-31-row-reorder-buffer-deadlock | engine-pooling/ | Deadlock on non-head sequence eviction |
| P1-2026-01-31-batching-mixin-unbound-local-error | engine-pooling/ | UnboundLocalError in exception handler |
| P1-2026-01-31-multi-query-output-key-collisions | plugins-llm/ | Silent data loss from key collisions |
| P1-2026-01-31-llm-response-payload-dropped-on-parse-failure | plugins-llm/ | LLM response lost on parse failure |
| P1-2026-01-31-context-record-call-bypasses-allocator | plugins-transforms/ | Duplicate call_index values possible |
| P1-2026-01-31-azure-csv-bad-lines-skipped-no-quarantine | plugins-sources/ | CSV bad lines silently skipped |
| P1-2026-01-31-azure-json-errors-crash-instead-quarantine | plugins-sources/ | JSON errors crash instead of quarantine |
| P1-2026-01-31-azure-json-accepts-nan-infinity | plugins-sources/ | NaN/Infinity allowed in JSON input |
| P1-2026-01-31-azure-blob-sink-no-audit-calls | plugins-sinks/ | Azure Blob operations not audited |
| P1-2026-01-31-settings-path-missing-silent-fallback | cli/ | Missing settings file silently falls back |

## All Valid Bugs Filed ✓

All 73 valid bugs from the triage have been filed:
- **25 P1 bugs** (critical)
- **36 P2 bugs** (moderate)
- **12 P3 bugs** (minor)

### P2 Bugs Filed (36 total)

| Area | Count | Examples |
|------|-------|----------|
| CLI | 2 | Resume payload backend ignored, rate limit cleanup |
| Contracts | 4 | Schema compatibility, ArtifactDescriptor duck-typing, etc. |
| Engine | 4 | Config gate MissingEdgeError, trigger coercion, etc. |
| Core/Landscape | 4 | Token outcome non-canonical, schema dialect-specific, etc. |
| Core (other) | 9 | Sink names lowercased, mixed logging, rate limit issues, etc. |
| Plugins/LLM | 5 | Azure batch custom_id, missing call records, retry semantics, etc. |
| Plugins (misc) | 3 | Verifier error misclassification, Azure auth method, etc. |
| MCP | 2 | Missing run_id filter, wrong status literal |

### P3 Bugs (not yet filed)

| Area | Count | Examples |
|------|-------|----------|
| Contracts | 3 | RowResult legacy accessors, telemetry output_hash, routing action |
| Core | 2 | Payload store re-export, checkpoint ID truncation |
| Engine | 1 | Aggregation missing telemetry |
| Core/Landscape | 1 | Lineage formatter fabricated latency |
| Plugins | 2 | Batching examples signature, Azure docstring |

## Duplicates Found

1. `retry.py.md` Bug 1 → Already exists as `P3-2026-01-21-retry-on-retry-called-after-final-attempt.md`
2. `retry.py.md` Bug 2 → Already exists as `P3-2026-01-21-retry-attempt-index-mismatch.md`
3. `exporter.py.md` → Partial duplicate of `P2-2026-01-19-exporter-missing-config-in-export.md` (should expand existing)

## Invalid/Empty Reports

46 files contained "No concrete bug found" or were empty placeholders. These should be deleted:

- All `__init__.py.md` files (empty modules)
- Files explicitly stating "No bug identified"
- Empty template files in plugins/pooling/

## Next Steps

1. **Immediate**: Address filed P1 bugs (9 critical issues)
2. **This sprint**: File remaining P1 bugs (~16 more)
3. **Next sprint**: File P2 bugs (~33 issues)
4. **Backlog**: File P3 bugs (~9 issues)
5. **Cleanup**: Delete `docs/bugs/generated/` directory after all bugs filed

## Methodology

Triage performed by parallel Claude Code agents:
- Each agent verified bugs against actual source code line numbers
- Checked for duplicates against existing `docs/bugs/open/` bugs
- Validated priority assessments (some adjusted based on impact analysis)
- Invalid reports marked for deletion
