# Explicit Routing Example

Demonstrates the declarative `on_success` / `input` wiring pattern that all ELSPETH pipelines use.

## What This Shows

A minimal pipeline where every edge is explicitly named — no positional inference. This is the canonical example of how DAG wiring works in ELSPETH.

```
source ─(validated)─> enrich ─(enriched)─> [value_check] ─┬─ high_value.csv
                                                           └─ standard.csv
```

## Running

```bash
elspeth run --settings examples/explicit_routing/settings.yaml --execute
```

## Output

- `output/high_value.csv` — Rows where `amount >= 5000`
- `output/standard.csv` — Rows where `amount < 5000`

## Key Concepts

- **Named connections**: `source.on_success: validated`, `transform.input: validated`, `transform.on_success: enriched`
- **No implicit ordering**: Unlike some pipeline frameworks, ELSPETH requires every edge to be declared
- **Gate conditions**: `row['amount'] >= 5000` is evaluated by the AST-based expression parser
- **Minimal example**: Good starting point for understanding ELSPETH's wiring model before looking at more complex examples
