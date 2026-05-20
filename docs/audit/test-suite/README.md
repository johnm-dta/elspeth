# Test Suite Audit

Multi-agent test-quality audit of the ELSPETH test suite (741 `test_*.py` files).
Goal: identify pointless, duplicate, low-effort, defective, and otherwise
underperforming tests, alongside major coverage gaps.

## Method

The suite is partitioned into 27 cohesive **audit chunks** across 4 tiers
(unit, integration, property, end-to-end). Each chunk is reviewed by a wave of
5 specialist agents in parallel, each applying a different lens:

1. `ordis-quality-engineering:test-suite-reviewer` — anti-patterns
2. `axiom-sdlc-engineering:quality-assurance-analyst` — VER/VAL theatre
3. `axiom-python-engineering:python-code-reviewer` — Python-specific smells
4. `pr-review-toolkit:pr-test-analyzer` — scenario coverage
5. `ordis-quality-engineering:coverage-gap-analyst` — SUT coverage gaps

## Chunk progress

| Chunk | Files | Status | Synthesis | Raw reports |
|---|---|---|---|---|
| U-CONTRACTS-1 (Plugin & registry contracts) | 25 | Done | [findings](u-contracts-1-findings.md) | [raw](u-contracts-1-raw-reports.md) |
| U-CORE-1 (Landscape audit-DB recording) | 32 | Done | [findings](u-core-1-findings.md) | [raw](u-core-1-raw-reports.md) |
| I-1 (Audit integration tests) | 32 | Done | [findings](i-1-findings.md) | [raw](i-1-raw-reports.md) |
| U-ENGINE-1 (Declaration/processor/executor tests) | partial | Partial | [findings](u-engine-1-findings.md) | [raw](u-engine-1-raw-reports.md) |
| U-ENGINE-2 (Coalesce/orchestrator/runtime boundary tests) | partial | Partial | [findings](u-engine-2-findings.md) | [raw](u-engine-2-raw-reports.md) |

## Cross-chunk patterns (suite-wide)

These appear in multiple chunks and should be addressed once with a CI rule rather than per-chunk:

- **`hasattr()` in tests violates CLAUDE.md** — found in U-CONTRACTS-1 (~15 sites), U-CORE-1 (5 sites), and the U-ENGINE-1 partial pass. Filed shared infrastructure item `elspeth-2f4978ffbc`; `scripts/cicd/enforce_tier_model.py` already detects banned R3 `hasattr`, but current CI/pre-commit invocations scan `src/elspeth`, not `tests/`.
- **Spec-less `Mock()`/`MagicMock()`** — found in U-CONTRACTS-1 (4 transform_contracts files), U-CORE-1 (~6 files, 13+ `journal._payload_store = Mock()` sites), I-1, and the U-ENGINE-1 partial pass. Filed shared infrastructure item `elspeth-e984600f90`; recommend a CI grep/lint to flag behavioral `Mock()` and `MagicMock()` without `spec=` or a real fake in `tests/`.
- **Hash-without-binding theatre** — found in U-CORE-1, I-1, and the U-ENGINE-1 partial pass. Filed shared infrastructure item `elspeth-e0afd080cc`; recommend a shared fixture asserting `(actual_hash) == stable_hash(input)` for retrofit.
- **Dataclass-machinery tautology cluster** — found in U-CONTRACTS-1, U-CORE-1, and the U-ENGINE-1 partial pass. Tests construct a `@dataclass`, set fields, read them back. Should be deleted on sight in subsequent chunks.
- **Production-code-path bypass in integration tests** — found in I-1: 26 of 32 files in `tests/integration/audit/` never use `ExecutionGraph.from_plugin_instances()` / `Orchestrator.run()`, despite CLAUDE.md mandating that integration tests do. Filed as `elspeth-ae9f541775` (epic), resolved 2026-05-20 by moving repository-level audit persistence tests to `tests/unit/core/landscape/repository_integration/` and adding a directory-discipline gate. The coverage-improvement work remains tracked by the specific prior-wave gaps listed in `i-1-findings.md`.
- **Regression-dump test files** — `tests/integration/audit/test_fixes.py` flagged unanimously by all 5 reviewers as a sprint-task dumping ground that should be dispersed and deleted. Likely pattern: any file named after a sprint, ticket, or "fixes" warrants scrutiny in subsequent chunks.

## Cross-wave verification: revising prior findings

The I-1 cross-reference task **revealed information that revises prior-wave findings.** Whenever a later wave shows a gap is covered elsewhere, the prior issue should be re-scoped, not closed:

| Issue | Original scope | Revised by I-1 verification |
|---|---|---|
| `elspeth-f6f50e9394` (ADR-019 sweep) | "no unit tests" was conflated with "no tests" | `tests/integration/test_adr_019_*.py` covers all three sweep methods at integration layer. Issue scope narrowed to "no *unit* tests"; integration coverage is sufficient. Lower priority. |

## Chunk plan

The full 27-chunk partition is documented in the session transcript that
produced this audit (2026-05-06). Subsequent waves should preserve the same
file-list-up-front approach so findings stay collatable.
