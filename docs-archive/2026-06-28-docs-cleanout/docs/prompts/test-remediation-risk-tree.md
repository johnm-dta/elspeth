# Test Remediation Risk Tree - Elspeth

Reference: `<reference>`
Working directory: `<repo-root>`

Goal: for every P1 task associated with `<reference>`, audit the test surface, confirm test gaps/defects with evidence, remediate authorized in-scope gaps, and bubble clustered risk back to the parent epic. Resolve "associated with `<reference>`" through Filigree/project context. If ambiguous, ask.

## Pre-work

1. Read `AGENTS.md`.
2. Load `engine-patterns-reference`, `tier-model-deep-dive`, `logging-telemetry-policy`, and `config-contracts-guide`. If any fail, halt.
3. Run `filigree session-context`; treat it as canonical for P1 tasks.
4. Resolve `<reference>` through Filigree/project context. Do not guess.
5. Ask whether to use `.worktrees/test-remediation-<reference>`; default yes.
6. Checkpoint after reference/P1 enumeration; stop for approval before traversal.

## Authorization

- Adding new tests, including new cases in existing files, is authorized.
- Changing/deleting existing test bodies, assertions, fixtures, or harness behavior requires confirmation per change set.
- Source changes outside tests are not authorized. If a gap reveals a production bug, file it via the bug workflow.
- Filigree mutations are authorized, subject to checkpoints and policy checks.

## Budgets

Per P1 task: max depth 3; max 30 test files; max 15 confirmed gaps before checkpoint.

Mandatory checkpoints: after P1 enumeration; after the first coverage map; before deletion/large rewrite; before parent epic mutation.

## Risk Tree

Parent epic -> P1 task -> subsystem/folder -> test file -> confirmed gap/defect -> evidence -> category -> fix/file/defer.

Use/update stable markers:

`<!-- test-gap-tree:start <reference> -->`
`<!-- test-gap-tree:end <reference> -->`

## Per-P1 Workflow

1. Claim one P1 task with `filigree start-work`. On `INVALID_TRANSITION`, run `filigree transitions <id>` and checkpoint.
2. Identify subsystem and test surface: unit, integration, contract, fixtures, conftest, harnesses.
3. Build a coverage map from direct inspection and available coverage data. Use mutation testing only if already supported.
4. For each finding, capture: test path; target file/symbol; covered vs. missing behavior; invariant/contract from loaded skills; flake evidence if any; mock depth; abstraction level; test tier vs. tier-model expectation.
5. Categorize: `missing-coverage`, `flaky`, `brittle`, `wrong-level`, `redundant`, `dead`, `wrong-path`.
6. Recurse only when a test file exposes deeper independent gaps.

## Decision Rules

- In-scope missing coverage: write the test now.
- In-epic but out-of-task coverage: file a test-gap issue.
- Test-side flaky root cause in scope: fix the test.
- Production-side flaky root cause: file a bug; do not mask it.
- Unknown flaky root cause: quarantine per policy and file investigation.
- Brittle/wrong-level/redundant/dead: file test-debt; do not rewrite/delete without confirmation.
- Wrong-path: add canonical-path coverage if in scope; file existing wrong-path test for review.
- If a new test mechanizes a prose-only invariant, file a follow-up to update the relevant skill/contract doc.

## Bubble-up

Before parent epic mutation, prepare exact update text and request approval. Cluster gaps by root cause, e.g. "config contracts not exercised across 4 P1 tasks."

Epic update: coverage delta where measurable, gap count by category, fixed gaps, filed issues, deferred risks/decisions, new/changed test files.

## Deliverables

Filigree risk tree; per-P1 summary; parent epic gap profile; committed new tests; issues for actionable test debt/gaps; final summary listing tasks inspected, test files traversed, confirmed gaps by category, rejected non-issues, tests added, issues updated, and remaining decisions.
