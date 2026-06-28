# Bug Remediation Risk Tree - Elspeth

Reference: `<reference>`
Working directory: `<repo-root>`

Goal: for every P1 bug associated with `<reference>`, validate with evidence, remediate authorized in-scope bugs regression-first, and bubble clustered defect risk to the parent epic. Resolve "associated with `<reference>`" through Filigree/project context. If ambiguous, ask.

## Pre-work

1. Read `AGENTS.md`.
2. Load `engine-patterns-reference`, `tier-model-deep-dive`, `logging-telemetry-policy`, and `config-contracts-guide`. If any fail, halt.
3. Run `filigree session-context`; treat it as canonical for P1 bugs.
4. Resolve `<reference>` through Filigree/project context. Do not guess.
5. Ask whether to use `.worktrees/bug-remediation-<reference>`; default yes.
6. Checkpoint after reference/P1 enumeration; stop for approval before traversal.

## Authorization

- Focused regression tests for confirmed in-scope bugs are authorized.
- Production fixes are authorized only for confirmed in-scope bugs after a failing regression.
- Changing/deleting existing tests requires confirmation per change set.
- Broad refactors, feature work, and unrelated cleanup are not authorized. File separate issues.
- Filigree mutations are authorized, subject to checkpoints and policy checks.

## Budgets

Per P1 bug: max depth 3; max 20 source files; max 20 test files; max 10 sibling risks before checkpoint.

Mandatory checkpoints: after P1 enumeration; after first evidence map; before broad production edits; before parent epic mutation.

## Risk Tree

Parent epic -> P1 bug -> subsystem/folder -> source/test file -> confirmed defect/risk -> evidence -> category -> fix/file/defer.

Use/update stable markers:

`<!-- bug-risk-tree:start <reference> -->`
`<!-- bug-risk-tree:end <reference> -->`

## Per-P1 Workflow

1. Claim one P1 bug with `filigree start-work`. On `INVALID_TRANSITION`, run `filigree transitions <id>` and checkpoint.
2. Read the bug, target source, surrounding code, comments, tests, fixtures, and contracts. Preserve `CLOSED LIST`, `do not extend`, composer invariants, and load-bearing comments.
3. Reproduce or directly prove the defect. Do not fix from description alone.
4. Build an evidence map: symptom; expected behavior; target file/symbol; failing command/path; violated loaded-skill invariant; trust tier/logging/config-contract implications.
5. Add or update a focused regression test, run it, and capture the expected failure.
6. Implement the smallest correct fix. Prefer mechanical constraints over comments.
7. Run focused verification, then appropriate policy/CI checks.
8. Recurse only when the root cause exposes deeper independent sibling defects.

## Categories

Use one primary category: `boundary-validation`, `tier-violation`, `audit-integrity`, `config-contract`, `runtime-invariant`, `data-corruption`, `concurrency`, `error-handling`, `frontend-contract`, `test-only-defect`, `stale-report`, `out-of-scope`.

## Decision Rules

- Confirmed in-scope bug: regression first, then fix.
- Confirmed production bug outside scope: file a bug and link it.
- Disproved/unreproduced report: reject with source evidence.
- Design evaluation without broken behavior: file/retype as task, not bug.
- Test-only defect: fix if in scope; otherwise file test-debt.
- Fix would relax a closed list, trust boundary, or audit invariant: halt and ask.

## Bubble-up

Before parent epic mutation, prepare exact update text and request approval. Cluster bugs by root cause.

Epic update: bugs fixed/filed, rejected non-issues, root causes, coverage added, verification commands, deferred risks, linked issues.

## Deliverables

Filigree bug risk tree; per-P1 bug summary; parent epic defect-risk profile; committed regressions/fixes; issues for sibling/out-of-scope defects; final summary listing bugs inspected, files traversed, confirmed/rejected counts, tests added, fixes made, issues updated, and remaining decisions.
