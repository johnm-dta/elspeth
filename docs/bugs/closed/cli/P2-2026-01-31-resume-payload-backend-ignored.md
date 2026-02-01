# Bug Report: Resume ignores payload_store.backend and always uses FilesystemPayloadStore

## Summary

- `elspeth resume` instantiates `FilesystemPayloadStore` without validating `payload_store.backend`, so non-filesystem backends are ignored and resume may read the wrong storage or fail unexpectedly.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/cli.py:1717-1725` - `resume()` instantiates `FilesystemPayloadStore` without checking `payload_store.backend`.
- `src/elspeth/cli.py:526-534` - `run()` explicitly validates `payload_store.backend == "filesystem"` before instantiating the payload store.

## Impact

- User-facing impact: Confusing resume failures when backend is not filesystem
- Data integrity: Could resume against incorrect payloads if local directory exists

## Proposed Fix

- Add backend validation guard in `resume()` mirroring the `run()` check

## Acceptance Criteria

- `elspeth resume` exits with explicit error if `payload_store.backend` is not `filesystem`

## Resolution (2026-02-02)

**Status: FIXED**

Added backend validation guard in `resume()` at `src/elspeth/cli.py:1825-1831` that mirrors the existing check in `run()`:

```python
if settings_config.payload_store.backend != "filesystem":
    typer.echo(
        f"Error: Unsupported payload store backend '{settings_config.payload_store.backend}'. "
        f"Only 'filesystem' is currently supported.",
        err=True,
    )
    raise typer.Exit(1)
```

All 123 resume-related tests pass.
