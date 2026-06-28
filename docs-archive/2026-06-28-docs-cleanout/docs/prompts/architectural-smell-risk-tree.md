# Architectural Smell Risk Tree - Elspeth

Reference: `<reference>`
Working directory: `<repo-root>`

Goal: for every P1 architecture/task item associated with `<reference>`, audit smells with source evidence, remediate authorized mechanical improvements, and bubble clustered design risk to the parent epic. Resolve "associated with `<reference>`" through Filigree/project context. If ambiguous, ask.

## Pre-work

1. Read `AGENTS.md`.
2. Load `engine-patterns-reference`, `tier-model-deep-dive`, `logging-telemetry-policy`, and `config-contracts-guide`. If any fail, halt.
3. Run `filigree session-context`; treat it as canonical for P1 architecture/task items.
4. Resolve `<reference>` through Filigree/project context. Do not guess.
5. Ask whether to use `.worktrees/architecture-risk-<reference>`; default yes.
6. Checkpoint after reference/P1 enumeration; stop for approval before traversal.

## Authorization

- Docs, issue filing, and narrow mechanical tests are authorized.
- Source changes require explicit approval after the evidence map unless the task already authorizes the exact refactor.
- Do not move modules, alter public contracts, rewrite architecture, or relax invariants without approval.
- Filigree mutations are authorized, subject to checkpoints and policy checks.

## Budgets

Per P1 item: max depth 3; max 30 source files; max 15 smells before checkpoint.

Mandatory checkpoints: after P1 enumeration; after first architecture map; before source refactors; before parent epic mutation.

## Risk Tree

Parent epic -> P1 item -> subsystem/folder -> source file/symbol -> confirmed smell -> evidence -> category -> fix/file/defer.

Use/update stable markers:

`<!-- architecture-risk-tree:start <reference> -->`
`<!-- architecture-risk-tree:end <reference> -->`

## Per-P1 Workflow

1. Claim one P1 item with `filigree start-work`. On `INVALID_TRANSITION`, run `filigree transitions <id>` and stop.
2. Read the issue, subsystem, module docs, comments, neighboring code, tests, contracts, and plans/reviews. Preserve `CLOSED LIST`, `do not extend`, composer invariants, and load-bearing comments.
3. Build an architecture map: boundaries, data/control flow, tier crossings, audit/logging, config mapping, APIs/contracts, dependency direction, and boundary tests.
4. For each smell, capture: file/symbol; concrete behavior/dependency; violated pattern or loaded-skill invariant; blast radius; affected/missing tests; migration risk; why this is not taste.
5. Categorize and decide fix/file/defer.
6. Recurse only when a file exposes deeper independent architectural risk.

## Categories

Use one primary category: `boundary-leak`, `tier-confusion`, `audit-telemetry-drift`, `config-contract-drift`, `duplicated-abstraction`, `wrong-layer`, `coupling-cycle`, `closed-list-pressure`, `implicit-invariant`, `dead-abstraction`, `testability-gap`, `migration-risk`, `non-issue`.

## Decision Rules

- Low-risk mechanical improvement: propose patch and get approval before source edits unless already authorized.
- Broad migration/design choice: file architecture task/feature.
- Production bug: file via bug workflow.
- Missing tests around invariant: file test-gap issue or hand off to test-remediation prompt.
- Preference-only or justified local pattern: reject with source evidence.
- Any change that relaxes trust tier, audit primacy, config-contract, or closed-list constraints: halt and ask.

## Bubble-up

Before parent epic mutation, prepare exact update text and request approval. Cluster smells by root cause.

Epic update: smell count by category, non-issues, remediation sequence, filed issues, blast radius, deferred decisions, approved changes.

## Deliverables

Filigree architecture risk tree; per-P1 summary; parent epic design-risk profile; committed approved docs/tests/source changes; issues for architecture debt; final summary listing items inspected, files traversed, smells confirmed/rejected, issues updated, changes made, and remaining decisions.
