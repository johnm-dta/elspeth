# Bug Report: check_compatibility ignores field constraints (false positives)

## Summary

- `check_compatibility` only compares `FieldInfo.annotation` and ignores constraint metadata, so constrained consumers (e.g., `allow_inf_nan=False`) are treated as compatible with unconstrained producers.
- This produces false positives in schema validation and can let pipelines validate even though downstream validation will reject rows.
- The function docstring claims constrained types are handled, so behavior diverges from the contract.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: 81a0925d7d6de0d0e16fdd2d535f63d096a7d052 (fix/rc1-bug-burndown-session-2)
- OS: Linux 6.8.0-90-generic (Ubuntu)
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: SchemaConfig with float field + manual PluginSchema float field

## Agent Context (if relevant)

- Goal or task prompt: deep bug audit of /home/john/elspeth-rapid/src/elspeth/contracts/data.py
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: code inspection + local Python introspection with PYTHONDONTWRITEBYTECODE=1 to compare FieldInfo metadata

## Steps To Reproduce

1. Create a schema via `create_schema_from_config` with `fields: ["x: float"]` (uses FiniteFloat `allow_inf_nan=False`).
2. Create a manual `PluginSchema` with `x: float` (no constraints).
3. Call `check_compatibility(ManualProducer, ConfigConsumer)`.
4. Observe `compatible=True` and no type mismatches, despite the consumer enforcing `allow_inf_nan=False`.

## Expected Behavior

- Compatibility should fail (or at least flag a constraint mismatch) when the consumer has stricter constraints that the producer does not guarantee.

## Actual Behavior

- `check_compatibility` reports compatible because it only compares annotations and ignores constraint metadata.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): Unknown
- Minimal repro input (attach or link): `src/elspeth/plugins/schema_factory.py:23-35` defines `FiniteFloat` with `allow_inf_nan=False`; `src/elspeth/contracts/data.py:152-170` only inspects annotations; `_types_compatible` has no constraint handling at `src/elspeth/contracts/data.py:210-242`.

## Impact

- User-facing impact: pipelines validate as compatible but later fail when constrained consumers reject rows.
- Data integrity / security impact: schema compatibility is overstated, allowing data that violates constraints to propagate.
- Performance or cost impact: reruns and debugging time when runtime validation fails unexpectedly.

## Root Cause Hypothesis

- `check_compatibility` ignores `FieldInfo.metadata` (where Pydantic stores constraints), so constrained types collapse to their base annotations.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/contracts/data.py`: compare constraint metadata (e.g., `allow_inf_nan`, numeric bounds) and require producer constraints to be at least as strict as consumer constraints.
- Config or schema changes: None.
- Tests to add/update:
  - Add unit tests covering `FiniteFloat` compatibility: producer unconstrained vs consumer constrained should be incompatible; producer constrained vs consumer unconstrained should be compatible.
- Risks or migration steps:
  - Stricter compatibility checks may invalidate previously “compatible” pipelines that relied on unconstrained producers.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/contracts/data.py:135-143`
- Observed divergence: constrained type handling is not implemented.
- Reason (if known): compatibility logic only inspects annotations.
- Alignment plan or decision needed: define constraint-compatibility rules (subset/strictness) for schema validation.

## Acceptance Criteria

- `check_compatibility(Producer(float), Consumer(FiniteFloat))` reports incompatible due to constraint mismatch.
- `check_compatibility(Producer(FiniteFloat), Consumer(float))` reports compatible.
- Tests demonstrate constraint evaluation, not just annotation equality.

## Tests

- Suggested tests to run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/plugins/test_schemas.py -k compatibility`
- New tests required: yes

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `docs/bugs/closed/P2-2026-01-19-non-finite-floats-pass-source-validation.md`
