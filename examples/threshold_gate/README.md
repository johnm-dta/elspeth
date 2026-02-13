# Threshold Gate Example

The simplest possible gate example — a single numeric threshold splits rows into two sinks.

## What This Shows

A CSV source feeds rows through a gate that checks `row['amount'] > 1000`. High-value rows go to one sink, the rest to another.

```
source ─(gate_in)─> [amount_threshold] ─┬─ high_values.csv  (amount > 1000)
                                         └─ normal.csv       (amount <= 1000)
```

## Running

```bash
elspeth run --settings examples/threshold_gate/settings.yaml --execute
```

## Output

- `output/high_values.csv` — Rows where `amount > 1000`
- `output/normal.csv` — Rows where `amount <= 1000`

## Key Concepts

- **Minimal gate**: The simplest routing pattern — one condition, two destinations
- **Config-driven**: No plugin code needed; the gate is pure YAML configuration
- **Deterministic routing**: Every row goes to exactly one sink based on the expression result

## See Also

- [`threshold_gate_container`](../threshold_gate_container/) — The same pipeline packaged for Docker
- [`boolean_routing`](../boolean_routing/) — String-based boolean routing
- [`deep_routing`](../deep_routing/) — Complex multi-level gate cascades
