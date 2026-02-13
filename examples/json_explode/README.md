# JSON Explode Example

Demonstrates the `json_explode` transform that expands nested JSON arrays into individual rows.

## What This Shows

A JSON source contains order records with an `items` array field. The `json_explode` transform unpacks each array element into its own row, preserving the parent context.

```
source ─(source_out)─> json_explode ─(output)─> order_items.json
```

For example, an order with 3 items becomes 3 output rows, each with the `order_id` plus one `item` object and an `item_index`.

## Running

```bash
elspeth run --settings examples/json_explode/settings.yaml --execute
```

## Output

Results appear in `output/order_items.json` (JSONL format). Each line contains the original `order_id` plus:
- `item` — The individual item object from the array
- `item_index` — The position within the original array (0-based)

## Key Concepts

- **Array expansion**: One input row with N array elements produces N output rows
- **JSON source**: Reads structured JSON input (vs CSV for tabular data)
- **Index tracking**: `include_index: true` adds positional metadata for audit traceability
- **Observed schema**: Output schema is `observed` mode — fields are discovered from the data rather than declared upfront
