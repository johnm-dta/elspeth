# CI Branch Protection Runbook

ELSPETH relies on one aggregate required check, `CI Success`, rather than
requiring each individual workflow job in branch protection. This keeps GitHub
rules stable while `ci.yaml` owns the actual required job list.

## Required Repository Rules

The `main` ruleset should enforce:

- branch deletion blocked
- non-fast-forward pushes blocked
- pull requests required before merge
- `CI Success` required before merge
- Copilot code review optional

Do not require `composer-redaction-gate` globally. It is path-filtered and only
runs when redaction-sensitive files change, so making it globally required would
block unrelated PRs where the workflow is correctly skipped.

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
- pull-request rules remain active
