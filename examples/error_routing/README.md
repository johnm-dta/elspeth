# Error Routing Example

Demonstrates explicit `on_success` / `on_error` wiring across a multi-gate pipeline with content filtering.

## What This Shows

A deal processing pipeline where a content filter diverts blocked rows to a quarantine sink, then amount-based and category-based gates split surviving rows across three output sinks.

```
source ─(raw)─> content_filter ─┬─(clean)─> truncate ─(trimmed)─> [amount_gate] ─┬─ high_value ─> [category_gate] ─┬─ enterprise
                                │                                                  │                                 └─ commercial
                                │                                                  └─ standard
                                └─(on_error)─> quarantine
```

## Running

```bash
elspeth run --settings examples/error_routing/settings.yaml --execute
```

## Output

Results appear in `output/`:
- `enterprise_deals.csv` — High-value enterprise deals
- `commercial_deals.csv` — High-value non-enterprise deals
- `standard_deals.csv` — Deals under $10,000
- `quarantine.csv` — Rows with blocked content (passwords, confidential, etc.)

## Key Concepts

- **on_error routing**: Blocked rows are routed to a quarantine sink, not silently dropped
- **Declarative wiring**: Every edge is explicitly named (`raw`, `clean`, `trimmed`, etc.)
- **No silent drops**: Every row reaches exactly one terminal sink
- **Audit completeness**: The routing decision for every row is recorded in the Landscape
