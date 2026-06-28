# Phase 3 Runtime Preflight Gate

Cleared at: 2026-05-24T10:25:09Z

Issue: `elspeth-fdebcaa79a`

## Direction Reuse Check

Command:

```bash
grep -nR --include='*.py' "blob_run_links\b" src/elspeth/ | grep -E '\.direction|direction\b'
```

Observed output:

```text
src/elspeth/web/blobs/service.py:339:                f"Tier 1: blob_run_links.direction is {row.direction!r}, expected one of {sorted(BLOB_RUN_LINK_DIRECTIONS)}"
```

Result: no active query was found that direction-segregates config-content reads away from existing source-data retention semantics. The remaining hit is the Tier-1 read guard for persisted `blob_run_links.direction`.

## Unique Constraint Check

Command:

```bash
grep -n "uq_blob_run_link\|UniqueConstraint.*blob_run_links" src/elspeth/web/sessions/models.py
```

Observed output:

```text
1249:    UniqueConstraint("blob_id", "run_id", "direction", name="uq_blob_run_link"),
```

Result: the dedupe-by-blob-id strategy remains valid for one input lifecycle link per `(blob_id, run_id, direction)`.
