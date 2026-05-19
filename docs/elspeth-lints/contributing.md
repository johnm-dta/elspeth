# Contributing To elspeth-lints

All new ELSPETH-specific static analysis belongs under `elspeth-lints`, not in
new `scripts/cicd/enforce_*.py` scripts.

The migration manifest at `config/cicd/lint_migration_status.yaml` lists legacy
enforcers that are temporarily allowed while they are ported. A new
`scripts/cicd/enforce_*.py` path must not be added to the manifest unless the
work is an explicit migration-state update for an existing legacy script.

Use `docs/elspeth-lints/rule-author-guide.md` for the rule workflow and
`docs/elspeth-lints/protocols.md` for the protocol details.
