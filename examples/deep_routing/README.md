# Deep Routing Example

Demonstrates a complex multi-level decision tree with 3 transforms, 5 chained gates, and 7 terminal sinks.

## What This Shows

A loan application triage pipeline that screens, normalizes, and routes applications through a cascade of gates based on amount, credit score, loan type, and term length. The DAG is 8 nodes deep at its longest path.

```
source ─(raw)─> content_screen ─┬─(screened)─> normalize ─> trim_notes ─(trimmed)─>
                                └─(on_error)─> quarantine

  ─> [amount_tier] ─┬─ micro_loans (< $5,000)
                     └─ regular ─> [credit_check] ─┬─ good (>= 700) ─> [loan_type] ─┬─ mortgage ─> [term_check] ─┬─ long_term (>= 240mo)
                                                   │                                 │                            └─ short_term
                                                   │                                 └─ approved_other
                                                   └─ poor ─> [risk_level] ─┬─ high_risk (>= $50k)
                                                                            └─ manual_review
```

## Running

```bash
elspeth run --settings examples/deep_routing/settings.yaml --execute
```

## Output

Each row reaches exactly one of 7 sinks in `output/`:

| Sink | Criteria |
|------|----------|
| `quarantine.csv` | Blocked content (passwords, confidential, etc.) |
| `micro_loans.csv` | Amount < $5,000 |
| `approved_other.csv` | Good credit, non-mortgage |
| `long_term_mortgages.csv` | Good credit, mortgage, term >= 240 months |
| `short_term_mortgages.csv` | Good credit, mortgage, term < 240 months |
| `high_risk.csv` | Poor credit, amount >= $50,000 |
| `manual_review.csv` | Poor credit, amount < $50,000 |

## Key Concepts

- **Chained gates**: 5 gates creating a 4-level deep decision tree — all config-driven, no plugin code
- **Transform pipeline**: `keyword_filter` -> `field_mapper` -> `truncate` before routing
- **on_error diversion**: Content screen routes flagged rows to quarantine instead of dropping them
- **Full audit lineage**: Every row's path through 8 hops is recorded in the Landscape
