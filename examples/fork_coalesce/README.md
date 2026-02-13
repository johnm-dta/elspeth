# Fork/Coalesce Example

Demonstrates the fork/join DAG pattern where a single row is duplicated into parallel tokens, processed independently, and merged back together.

## What This Shows

After pre-processing (truncating long descriptions), every row is forked into two parallel tokens. Both tokens traverse to a coalesce point that waits for both to arrive, then merges them using the `nested` strategy — the output contains each branch's data under a separate key.

```
source ─(raw)─> truncate ─(preprocessed)─> [fork_gate] ─┬─ path_a ─┐
                                                          └─ path_b ─┤
                                                                     ├─ [merge_results]
                                                           (merged) ─> [route_output] ─> output
```

## Running

**Basic fork/coalesce (no per-branch transforms):**
```bash
elspeth run --settings examples/fork_coalesce/settings.yaml --execute
```

**Per-branch transforms variant (ARCH-15):**
```bash
elspeth run --settings examples/fork_coalesce/settings_per_branch.yaml --execute
```
> **Note:** Per-branch transforms are a new ARCH-15 feature. Validation currently has a schema config bug being fixed in the main implementation.

## Variants

This example has two configuration files demonstrating different fork/coalesce patterns:

### 1. `settings.yaml` — Basic Fork/Coalesce (Direct Wiring)

Uses the **list format** for branches: `branches: [path_a, path_b]`

Both fork paths carry identical data to the coalesce — no transforms run on the branches. This demonstrates the basic merge barrier pattern.

**DAG:**
```
source ─> truncate ─> [fork] ─┬─ path_a ─┐
                               └─ path_b ─┤
                                          ├─ [coalesce] ─> output
```

**Use case:** Redundancy (send to 3 LLMs, take quorum) or fan-out pattern setup.

### 2. `settings_per_branch.yaml` — Per-Branch Transforms (ARCH-15)

Uses the **dict format** for branches: `branches: {path_a: truncated_a, path_b: mapped_b}`

Each fork path runs **different transforms** before coalescing. This is the key ARCH-15 innovation — per-branch processing chains.

**DAG:**
```
source ─> [fork] ─┬─ path_a ─> truncate ─────> truncated_a ─┐
                  └─ path_b ─> field_mapper ─> mapped_b ────┤
                                                             ├─ [coalesce] ─> output
```

**Transforms:**
- **path_a**: Truncates `description` field to 20 chars
- **path_b**: Renames fields (`product` → `product_name`, `id` → `item_id`, `price` → `cost`) and drops `category` and `description`

**Output structure** (nested merge):
```json
{
  "path_a": {
    "id": 1,
    "product": "Widget Pro",
    "price": 2500,
    "category": "electronics",
    "description": "High-performance w..."
  },
  "path_b": {
    "product_name": "Widget Pro",
    "item_id": 1,
    "cost": 2500
  }
}
```

**Use case:** Multi-API enrichment (sentiment API on path_a, entity extraction on path_b), then merge enriched results.

## Output

Results appear in `output/merged_results.json` (basic variant) or `output/per_branch_results.json` (per-branch variant) as JSONL.

See the **Variants** section above for example output structures.

## Why Fork/Coalesce?

In this minimal example both paths carry the same data. The real power emerges in production pipelines where each fork path hits a **different external service**:

| Scenario | Fork Paths | Merge Policy |
|----------|-----------|-------------|
| **LLM redundancy** | 3 different models | `quorum` (2-of-3 agree) |
| **Multi-API enrichment** | Sentiment API + Entity API | `require_all` + `union` |
| **Fastest wins** | Fast model + Accurate model | `first` |
| **Best effort** | Primary + Fallback | `best_effort` with timeout |

## Merge Policies

| Policy | Behaviour |
|--------|-----------|
| `require_all` | Wait for all branches — fail if any missing |
| `quorum` | Wait for N-of-M branches (`quorum_count: 2`) |
| `best_effort` | Merge when timeout expires or all branches arrive |
| `first` | Merge immediately when first branch arrives |

## Merge Strategies

| Strategy | Result Shape |
|----------|-------------|
| `union` | `{field_a: val_a, field_b: val_b, ...}` — flat merge (last branch wins on collisions) |
| `nested` | `{path_a: {...}, path_b: {...}}` — branch-keyed (used in this example) |
| `select` | `{...}` from one chosen branch (`select_branch: path_a`) |

## Key Concepts

- **Fork gate**: `routes: {true: fork}` with `fork_to: [path_a, path_b]` duplicates each row
- **Token lineage**: Each forked token gets its own `token_id` with parent recorded for audit
- **Pre-fork transforms**: Shared processing (like truncation) happens before the fork
- **Post-coalesce routing**: The coalesce name (`merge_results`) becomes a connection for downstream gates
- **Terminal state**: Parent tokens are marked `FORKED`; merged tokens are `COALESCED`

## See Also

- [`deep_routing`](../deep_routing/) — Complex gate cascades (fan-out to sinks, no merge)
- [`explicit_routing`](../explicit_routing/) — Basic wiring model
