# Contributing To elspeth-lints

All new ELSPETH-specific static analysis belongs under `elspeth-lints`, not in
new `scripts/cicd/enforce_*.py` scripts.

The migration manifest at `config/cicd/lint_migration_status.yaml` lists legacy
enforcers that are temporarily allowed while they are ported. A new
`scripts/cicd/enforce_*.py` path must not be added to the manifest unless the
work is an explicit migration-state update for an existing legacy script.

Use `docs/elspeth-lints/rule-author-guide.md` for the rule workflow and
`docs/elspeth-lints/protocols.md` for the protocol details.

## Fixture Convention

Every rule must ship at least one `examples_violation` fixture and one
`examples_clean` fixture under its rule package:

```text
elspeth-lints/src/elspeth_lints/rules/<rule_package>/
├── rule.py
├── metadata.py
└── fixtures/
    ├── examples_violation/
    │   ├── 01_basic.py
    │   └── 01_basic.expected.json
    └── examples_clean/
        └── 01_basic.py
```

Incremental AST rules normally use single `.py` fixtures. Whole-repository rules
may use a directory fixture with an adjacent `<case>.expected.json` file:

```text
fixtures/examples_violation/01_repo_shape/
fixtures/examples_violation/01_repo_shape.expected.json
```

Runtime-backed whole-repository rules may include `_fixture_rule.py` inside a
directory fixture to provide a fixture-local `RULE` object, such as a rule
instance wired to a fake plugin manager. That hook is for fixture dependency
injection only; it does not extend the production `Rule` protocol.

The shared test `tests/unit/elspeth_lints/test_all_rule_fixtures.py` fails when a
rule has no fixtures, when a clean fixture emits findings, or when a violation
fixture does not match its expected JSON. It also checks that
`RuleMetadata.examples_violation_count` and `RuleMetadata.examples_clean_count`
match the discovered fixture inventory.
