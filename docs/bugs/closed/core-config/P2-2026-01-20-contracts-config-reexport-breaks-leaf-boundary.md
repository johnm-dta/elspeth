# Bug Report: contracts/config.py imports core.config (contracts no longer a leaf; circular-import risk)

## Summary

- The Contracts subsystem is intended to be the leaf “shared types” module, but `elspeth.contracts.config` directly imports `elspeth.core.config` and re-exports Pydantic settings models from Core.
- Because `elspeth.contracts.__init__` re-exports these settings models, importing `elspeth.contracts` pulls in Core (and Core pulls in Engine submodules like the expression parser).
- `elspeth.contracts.__init__` explicitly notes that import order is “load-bearing” to avoid circular imports, indicating existing fragility.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A

## Steps To Reproduce

1. In a minimal environment, run `import elspeth.contracts`.
2. Observe that it imports `elspeth.core.config` (and its transitive dependencies), even if the caller only wants basic contracts like enums/results.

## Expected Behavior

- Contracts remain a leaf module: importing `elspeth.contracts` should not import Core/Engine modules.
- Configuration models (if considered contracts) should live in Contracts and be consumed by Core, not the other way around.

## Actual Behavior

- `elspeth.contracts` depends on `elspeth.core.config`, making Contracts non-leaf and increasing the risk of circular imports and heavy import-time side effects.

## Evidence

- Direct dependency from Contracts into Core:
  - `src/elspeth/contracts/config.py:10-23` (`from elspeth.core.config import ...`)
- Contracts package imports config re-exports:
  - `src/elspeth/contracts/__init__.py:75-90`
- `contracts/__init__.py` acknowledges load-bearing import order to avoid circular imports:
  - `src/elspeth/contracts/__init__.py:12-13`

## Impact

- User-facing impact: increased import time and potential circular import failures in edge cases.
- Data integrity / security impact: low.
- Performance or cost impact: higher startup overhead for any consumer importing contracts.

## Root Cause Hypothesis

- To keep Pydantic validation logic in Core, config models were defined there and re-exported through Contracts for import consistency, creating a dependency inversion.

## Proposed Fix

- Code changes (modules/files):
  - Option A (preferred): move the Pydantic settings models into Contracts (e.g., `src/elspeth/contracts/config_models.py`) and have `src/elspeth/core/config.py` import and provide the load/resolve functions and validators around those models.
  - Option B (minimal): stop re-exporting config models from `elspeth.contracts` and require config imports from `elspeth.core.config` directly, restoring Contracts as a leaf.
- Tests to add/update:
  - Add a smoke test asserting `import elspeth.contracts` does not import `elspeth.core.config` (if Option A/B pursued).

## Architectural Deviations

- Spec or doc reference: subsystem boundary described in `docs/arch-analysis-2026-01-20-0105/02-subsystem-catalog.md` (“Outbound: None” for Contracts)
- Observed divergence: Contracts imports Core.
- Alignment plan or decision needed: decide whether config models belong in Contracts or Core.

## Acceptance Criteria

- Importing `elspeth.contracts` does not import `elspeth.core.config`.
- Config model import path is stable and documented (either contracts-first or core-first), without load-bearing import order hacks.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/ -k import`
- New tests required: yes

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID - WORSE THAN REPORTED

**Verified By:** Claude Code P2 verification wave 6c

**Current Code Analysis:**

The bug is confirmed and the impact is actually more severe than originally reported:

1. **Dependency Chain Confirmed:**
   - `/home/john/elspeth-rapid/src/elspeth/contracts/config.py:10-23` still imports from `elspeth.core.config`
   - `/home/john/elspeth-rapid/src/elspeth/contracts/__init__.py:79-92` still re-exports all config models
   - The "load-bearing import order" comment remains at lines 11-13 with `# isort: skip_file`

2. **Measured Impact (Worse Than Expected):**
   - Importing `import elspeth.contracts` loads **1,217 total modules**
   - This includes: 295 pandas modules, 100 numpy modules, 108 sqlalchemy modules, 285 networkx modules
   - All of `elspeth.core` gets loaded: config, landscape, checkpoint, dag, canonical, events, logging, payload_store
   - Even importing just `elspeth.contracts.enums` triggers the full import chain because Python runs `contracts/__init__.py`

3. **Direct Test Results:**
   ```python
   # Importing contracts pulls in core
   import elspeth.contracts
   # Result: elspeth.core.config is loaded ✗

   # Even importing just enums triggers everything via __init__.py
   import elspeth.contracts.enums
   # Result: 1,217 modules loaded including all of core ✗

   # Loading enums.py file directly (bypassing __init__.py)
   # Result: Only 11 modules loaded ✓
   ```

4. **The Import Cycle:**
   - Any import of `elspeth.contracts.*` runs `contracts/__init__.py`
   - `__init__.py` imports `contracts.config`
   - `contracts.config` imports `core.config`
   - `core.config` is clean but importing it triggers package initialization which loads landscape, checkpoint, etc.
   - Total cascade: trying to use a simple enum pulls in the entire database layer, DAG engine, and all heavy dependencies

**Git History:**

- Created in commit `74e6bb6` (2026-01-16): "feat(contracts): add config.py with Pydantic re-exports"
- Commit message explicitly acknowledges the circular import risk
- Uses "load-bearing import order" and `isort: skip_file` as a workaround
- No subsequent commits have attempted to fix this issue
- The workaround has prevented a hard circular import crash but creates massive startup overhead

**Root Cause Confirmed:**

Yes, root cause is exactly as hypothesized:

1. Config models (Pydantic BaseModel subclasses) defined in `core.config`
2. Re-exported through `contracts.config` for "import consistency"
3. This creates an upward dependency: Contracts → Core
4. Violates the leaf module principle where Contracts should have zero outbound dependencies
5. The "load-bearing import order" is a code smell indicating architectural fragility

**Performance Impact Quantified:**

The 1,217 module import overhead means:
- Any CLI command that imports contracts pays ~500ms+ startup cost
- Tests that import contracts pay this cost per test process
- Cannot use lightweight contracts (enums, results) without pulling in the entire framework
- Violates Python best practice of lazy imports and minimal dependencies

**Recommendation:**

**KEEP OPEN - HIGH PRIORITY**

This should be elevated in priority because:

1. **Architectural violation:** Contracts is explicitly documented as a leaf module with "Outbound: None"
2. **Performance impact:** 1,200+ module startup overhead for simple enum access
3. **Maintenance burden:** Load-bearing import order is fragile and prevents refactoring
4. **User experience:** CLI feels slow, test suite slower than necessary

**Proposed fix remains valid (Option B preferred for minimal risk):**
- Remove config re-exports from `contracts/__init__.py`
- Require callers to import config from `elspeth.core.config` directly
- Add regression test: `assert 'elspeth.core' not in sys.modules after 'import elspeth.contracts.enums'`

This preserves current architecture (config stays in core) while restoring the Contracts leaf boundary.
