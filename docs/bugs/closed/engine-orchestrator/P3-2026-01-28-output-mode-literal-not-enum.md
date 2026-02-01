# Bug Report: output_mode uses Literal strings instead of StrEnum

## Summary

- `AggregationSettings.output_mode` is typed as `Literal["single", "passthrough", "transform"]` rather than a `StrEnum`.
- Comparison sites in `processor.py` use raw string literals (`if output_mode == "single":`) which are error-prone and inconsistent with other enums in the codebase.

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: Claude Code (Opus 4.5)
- Date: 2026-01-28
- Related run/issue ID: N/A

## Environment

- Commit/branch: feat/structured-outputs
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Fixing P3 engine-orchestrator bugs; user asked about string literal usage
- Model/version: Claude Opus 4.5
- Tooling and permissions (sandbox/approvals): full workspace access
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection during bug fix session

## Steps To Reproduce

1. Open `src/elspeth/core/config.py:156`
2. Observe `output_mode: Literal["single", "passthrough", "transform"]`
3. Search for usages in `src/elspeth/engine/processor.py`
4. Observe raw string comparisons: `if output_mode == "single":`

## Expected Behavior

- `output_mode` should use a `StrEnum` like other domain concepts (e.g., `NodeStateStatus`, `RowOutcome`, `RunStatus`)
- Comparison sites should use enum members: `if output_mode == OutputMode.SINGLE:`

## Actual Behavior

- `output_mode` uses `Literal[...]` type annotation
- Comparison sites use raw string literals that could be misspelled without type errors

## Evidence

- Config definition: `src/elspeth/core/config.py:156`
  ```python
  output_mode: Literal["single", "passthrough", "transform"] = Field(
      default="single",
      description="How batch produces output rows",
  )
  ```
- Usage in processor (repeated string literals):
  - `src/elspeth/engine/processor.py:414`: `if output_mode == "single":`
  - `src/elspeth/engine/processor.py:522`: `elif output_mode == "transform":`
  - `src/elspeth/engine/processor.py:705`: `if output_mode == "single":`
  - `src/elspeth/engine/processor.py:843`: `elif output_mode == "transform":`

## Impact

- User-facing impact: None (internal code quality)
- Data integrity / security impact: None
- Performance or cost impact: None

## Root Cause Hypothesis

- `Literal[...]` was chosen for config-first simplicity (YAML values are strings).
- Comparison sites didn't get converted to enum usage when other enums were standardized.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/contracts/enums.py` or similar: Add `OutputMode(StrEnum)`
  - `src/elspeth/core/config.py`: Change type to `OutputMode`, Pydantic auto-coerces strings
  - `src/elspeth/engine/processor.py`: Replace string literals with `OutputMode.SINGLE`, etc.
  - `src/elspeth/engine/executors.py`: Same treatment if applicable

- Config or schema changes: None (YAML values remain strings, Pydantic coerces)

- Tests to add/update:
  - Update any tests that compare against string literals

- Risks or migration steps:
  - Low risk: `StrEnum` values equal their string representations, so existing YAML configs work unchanged
  - Pydantic 2.x handles `StrEnum` coercion automatically

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: `output_mode` doesn't follow the enum pattern used by `NodeStateStatus`, `RowOutcome`, `RunStatus`, etc.
- Reason (if known): Probably predates enum standardization or was overlooked
- Alignment plan or decision needed: Decide if enum consistency is worth the refactor

## Acceptance Criteria

- `output_mode` uses a `StrEnum` type
- All comparison sites use enum members instead of string literals
- Existing YAML configs continue to work without modification

## Verification (2026-02-01)

**Status: STILL VALID**

- `AggregationSettings.output_mode` is still typed as a `Literal` in config. (`src/elspeth/core/config.py:156-159`)
- Processor still compares `output_mode` using raw strings (e.g., `"passthrough"`/`"transform"`). (`src/elspeth/engine/processor.py:510-512`, `src/elspeth/engine/processor.py:771-776`)

## Resolution (2026-02-02)

**Status: FIXED**

### Implementation

1. **Added `OutputMode(str, Enum)`** to `contracts/enums.py`:
   ```python
   class OutputMode(str, Enum):
       PASSTHROUGH = "passthrough"
       TRANSFORM = "transform"
   ```

2. **Updated `AggregationSettings.output_mode`** in `core/config.py`:
   - Changed type from `Literal["passthrough", "transform"]` to `OutputMode`
   - Changed default from `"transform"` to `OutputMode.TRANSFORM`

3. **Updated all 7 string comparisons** in `engine/processor.py`:
   - `if output_mode == "passthrough":` → `if output_mode == OutputMode.PASSTHROUGH:`
   - `elif output_mode == "transform":` → `elif output_mode == OutputMode.TRANSFORM:`

### Verification

```bash
.venv/bin/python -m pytest tests/engine/test_processor_batch.py tests/core/test_config_aggregation.py -v
# 32 passed
```

Existing YAML configs continue to work unchanged due to `(str, Enum)` coercion.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor.py -v`
  - `pytest tests/engine/test_processor_batch.py -v`
  - `pytest tests/core/test_config.py -v`
- New tests required: No, but existing tests may need literal→enum updates

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
- Similar enums for reference:
  - `NodeStateStatus` in `src/elspeth/contracts/enums.py`
  - `RowOutcome` in `src/elspeth/contracts/enums.py`
  - `RunStatus` in `src/elspeth/contracts/enums.py`
