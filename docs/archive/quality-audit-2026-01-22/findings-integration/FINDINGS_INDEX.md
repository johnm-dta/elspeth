# Integration Seam Findings Index

Total findings: 10

## Triage Table

| Priority | Source File | Summary | Status | Report |
|----------|-------------|---------|--------|--------|
| P1 | `coalesce_executor` | Coalesce failure outcomes returned from `flush_pending` are never recorded because `CoalesceExecutor | pending | [coalesce_executor.py.md](coalesce_executor.py.md) |
| P1 | `executors` | AggregationExecutor imports and handles `BatchPendingError` from the LLM plugin pack, so core batch  | pending | [executors.py.md](executors.py.md) |
| P1 | `expression_parser` | ExpressionParser.is_boolean_expression classifies any BoolOp as boolean for config validation, but g | pending | [expression_parser.py.md](expression_parser.py.md) |
| P1 | `orchestrator` | Orchestrator resume directly queries Landscape SQLAlchemy tables (`runs_table`, `edges_table`), leak | pending | [orchestrator.py.md](orchestrator.py.md) |
| P1 | `processor` | RowProcessor rejects protocol-only plugins by requiring BaseGate/BaseTransform subclasses, contradic | pending | [processor.py.md](processor.py.md) |
| P1 | `tokens` | TokenManager persists source row payloads itself and passes `payload_ref` into LandscapeRecorder.cre | pending | [tokens.py.md](tokens.py.md) |
| P1 | `triggers` | TriggerConfig’s condition example assumes row-level fields, but TriggerEvaluator evaluates condition | pending | [triggers.py.md](triggers.py.md) |
| P2 | `retry` | RetrySettings exposes `exponential_base`, but RetryConfig/RetryManager drop it, so configured expone | pending | [retry.py.md](retry.py.md) |
| P3 | `spans` | SpanFactory still defines an `aggregation_span` labeled as an aggregation plugin, but aggregation ex | pending | [spans.py.md](spans.py.md) |


## Clean Files (1)

| Source File |
|-------------|
| `artifacts` |


## Triage Status Legend

- **pending**: Not yet reviewed
- **confirmed**: Defect verified, refactoring needed
- **invalid**: False positive, not a real seam issue
- **duplicate**: Already tracked elsewhere
- **fixed**: Defect resolved

---
Last updated: 2026-01-25T16:13:29+00:00
