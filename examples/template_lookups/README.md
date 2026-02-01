# Template Files and Lookups Example

This example demonstrates ELSPETH's support for external template files and YAML-based lookup tables with two-dimensional lookups.

## Features Demonstrated

1. **External Template File** (`template_file`) - Load Jinja2 templates from separate files
2. **External Lookup File** (`lookup_file`) - Load YAML data for use in templates
3. **Two-Dimensional Lookups** - Use `lookup.X[row.Y]` to map row values to lookup data
4. **Audit Metadata** - Template and lookup sources are tracked in output for traceability

## Use Case: Customer Support Ticket Classification

The example classifies customer support tickets using an LLM. Each ticket has:
- `ticket_id`: Unique identifier
- `category_id`: References a category in the lookup table
- `priority_id`: References priority handling instructions
- `subject`: Ticket subject line
- `body`: Ticket content

The lookup table maps IDs to:
- Full category names with descriptions
- Priority-specific handling instructions

## File Structure

```
template_lookups/
├── README.md
├── settings.yaml           # Sequential pipeline configuration
├── settings_batched.yaml   # Batch aggregation variant
├── input.csv               # Sample support tickets
├── prompts/
│   ├── classify.j2         # External Jinja2 template
│   └── categories.yaml     # Lookup data (categories, priorities)
├── output/
│   ├── results.csv         # Sequential mode results
│   └── results_batched.csv # Batch mode results
└── runs/
    ├── audit.db            # Sequential mode audit trail
    └── audit_batched.db    # Batch mode audit trail
```

## Configuration Variants

| File | Processing Mode | Description |
|------|-----------------|-------------|
| `settings.yaml` | Sequential | One row at a time, simple but slow |
| `settings_batched.yaml` | Batch parallel | Buffer N rows, process in parallel |

The batched variant combines external template/lookup features with batch aggregation:
- Rows are buffered until the trigger fires (default: 5 rows)
- Batch is processed in parallel using a worker pool
- Template and lookup files are still loaded once at config time
- Each row gets its own prompt rendered with `{{ row.* }}` and `{{ lookup.* }}`

## How Two-Dimensional Lookups Work

In the template, you can access lookup data using row values as keys:

```jinja2
{# lookup.categories is a dict, row.category_id selects the entry #}
Category: {{ lookup.categories[row.category_id].name }}

{# lookup.priorities maps priority_id to handling instructions #}
Instructions: {{ lookup.priorities[row.priority_id] }}
```

This pattern is powerful for:
- Mapping codes to full descriptions
- Selecting prompt variations based on input data
- Building dynamic prompts without hardcoding values

## Audit Trail

The output includes audit metadata:
- `llm_response_template_source`: Path to template file
- `llm_response_template_hash`: SHA-256 of template content
- `llm_response_lookup_source`: Path to lookup file
- `llm_response_lookup_hash`: SHA-256 of canonical JSON lookup data

This ensures every classification can be traced back to the exact template and lookup data used.

## Running the Example

```bash
# Set your API key
export OPENROUTER_API_KEY="your-key-here"
export ELSPETH_ALLOW_RAW_SECRETS=true  # For development only

# Run sequential mode
uv run elspeth run -s examples/template_lookups/settings.yaml --execute --verbose

# Run batched mode (parallel processing)
uv run elspeth run -s examples/template_lookups/settings_batched.yaml --execute --verbose

# Check results
cat examples/template_lookups/output/results.csv
cat examples/template_lookups/output/results_batched.csv

# Check landscape database
sqlite3 examples/template_lookups/runs/audit.db "SELECT run_id, status FROM runs;"
sqlite3 examples/template_lookups/runs/audit_batched.db "SELECT run_id, status FROM runs;"
```
