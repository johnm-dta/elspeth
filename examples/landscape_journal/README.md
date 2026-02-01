# Landscape Journal Example

This example enables the JSONL change journal for the Landscape database.
The journal is an append-only stream of committed database writes for
emergency recovery if the SQLite audit database becomes unavailable.
This is optional, redundant logging (disabled by default).

## Run

```bash
uv run elspeth run -s examples/landscape_journal/settings.yaml --execute
```

## Outputs

- Audit DB: `examples/landscape_journal/runs/audit.db`
- Change journal: `examples/landscape_journal/runs/audit.journal.jsonl`
- Sink output: `examples/landscape_journal/output/output.jsonl`

The journal file is safe to tail during execution:

```bash
tail -f examples/landscape_journal/runs/audit.journal.jsonl
```
