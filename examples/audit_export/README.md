# Audit Export Example

This demonstrates ELSPETH's audit trail export feature for compliance and legal inquiry. After a run completes, the full audit trail is exported to a JSON sink for review.

## Running the Example

```bash
uv run elspeth run -s examples/audit_export/settings.yaml --execute
```

### With Signed Exports (Legal/Compliance Use)

```bash
export ELSPETH_SIGNING_KEY="your-secret-key"
# Update sign: true in settings.yaml
uv run elspeth run -s examples/audit_export/settings.yaml --execute
```

## Output Format

Uses JSON format because audit records are heterogeneous (run, node, row, token records have different fields). For CSV export, separate files per record type would be needed.

## Troubleshooting

### Schema Compatibility Error

If you see an error like:

> SchemaCompatibilityError: Landscape database schema is outdated

This means you have an old `audit.db` from a previous version. Fix by deleting it:

```bash
rm examples/audit_export/runs/audit.db
```

Then re-run the example.
