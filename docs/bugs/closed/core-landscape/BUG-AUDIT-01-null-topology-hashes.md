# Bug Report: Checkpoint Contract Allows NULL Topology Hashes

**Status: CLOSED (Fixed 2026-01-29)**

## Summary

- Checkpoint schema allows `topology_hash: str | None`, but NULL hash is never valid and masks audit database corruption.

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: Branch Bug Scan
- Date: 2026-01-25
- Related run/issue ID: BUG-AUDIT-01

## Evidence

- `src/elspeth/contracts/audit.py` - Schema allows Optional topology hash

## Impact

- Schema tightening: NULL should be rejected at schema level

## Resolution

**Root Cause:** Contract/schema mismatch. The database schema (`schema.py`) correctly enforced `nullable=False`, but the Python `Checkpoint` dataclass in `audit.py` used `str | None`, violating the Data Manifesto's Tier 1 (full trust) principle.

**Fix Applied:**

1. Changed `upstream_topology_hash: str | None` → `upstream_topology_hash: str` in `audit.py`
2. Changed `checkpoint_node_config_hash: str | None` → `checkpoint_node_config_hash: str` in `audit.py`
3. Added `__post_init__` validation to crash if None/empty passed (Tier 1 audit data principle)
4. Removed dead legacy checkpoint handling from `compatibility.py` (per No Legacy Code Policy)
5. Removed unused Alembic migration (pre-release, 0 users)

## Acceptance Criteria

- ✅ Topology hash required in Python contract
- ✅ `__post_init__` crashes on None/empty (Tier 1 crash on garbage)
- ✅ All 461 related tests pass
- ✅ Type checker passes

## Tests

- Existing tests cover this (203 checkpoint tests, 4 schema constraint tests pass)
