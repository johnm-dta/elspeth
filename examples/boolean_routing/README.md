# Boolean Routing Example

Demonstrates a simple gate that splits rows into two sinks based on a field value.

## What This Shows

A CSV source feeds rows through a boolean gate that checks `row['approved'] == 'true'`. Approved rows go to one sink, rejected rows to another.

```
source ─(gate_in)─> [approval_check] ─┬─ approved.csv
                                       └─ rejected.csv
```

## Running

```bash
elspeth run --settings examples/boolean_routing/settings.yaml --execute
```

## Output

- `output/approved.csv` — Rows where `approved == 'true'`
- `output/rejected.csv` — Rows where `approved != 'true'`

## Key Concepts

- **Config-driven gates**: Routing is declared in YAML via `condition` + `routes`, no plugin code needed
- **Binary routing**: Every row reaches exactly one of the two sinks
- **Expression evaluation**: `row['approved'] == 'true'` is parsed by the AST-based expression parser (no `eval`)
