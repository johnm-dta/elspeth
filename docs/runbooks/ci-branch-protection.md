# CI Branch Protection Runbook

ELSPETH relies on one aggregate required check, `CI Success`, for ordinary CI
jobs rather than requiring each individual `ci.yaml` job in branch protection.
The composer redaction policy gate is a separate governance check because it
lives in its own workflow.

## Required Repository Rules

The `main` ruleset should enforce:

- branch deletion blocked
- non-fast-forward pushes blocked
- pull requests required before merge
- `CI Success` required before merge
- `redaction-gate` from the `composer-redaction-gate` workflow required before
  merge
- Copilot code review optional

The `composer-redaction-gate` workflow is intentionally not declaratively
path-filtered. It starts for every pull request targeting `main`, `master`, or
an `RC*` branch, then its first shell step checks whether any
redaction-sensitive files changed. If none changed, the job exits successfully
with a "gate not applicable" summary. This always-starts/shell-skips shape
keeps the status check available for branch protection without leaving unrelated
PRs blocked on a skipped required workflow.

## CODEOWNERS Compensating Control

This repository is owned by a personal account, so a team-based CODEOWNERS rule
such as `@elspeth/security` is not available. Redaction-sensitive changes are
covered by `.github/workflows/composer-redaction-gate.yml`, which requires the
appropriate redaction-direction label and weakening rationale when the redaction
snapshot shrinks.

## Verification

Check whether the default branch is protected:

```bash
gh api repos/johnm-dta/elspeth/branches/main --jq '{name: .name, protected: .protected}'
```

Inspect active rulesets:

```bash
gh api repos/johnm-dta/elspeth/rulesets --jq '[.[] | {id,name,target,enforcement}]'
```

Inspect the active `main` ruleset, replacing the ID with the value from the
previous command:

```bash
gh api repos/johnm-dta/elspeth/rulesets/12348893 \
  --jq '{name: .name, enforcement: .enforcement, rules: [.rules[] | {type, parameters}]}'
```

Acceptance check:

- `main` reports `protected: true`
- the active branch ruleset targets `~DEFAULT_BRANCH`
- the ruleset includes a required status check for `CI Success`
- the ruleset includes the `redaction-gate` job from
  `.github/workflows/composer-redaction-gate.yml` as a required status check
- pull-request rules remain active
