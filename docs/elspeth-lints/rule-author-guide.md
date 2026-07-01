# elspeth-lints Rule Author Guide

This guide is for adding a new ELSPETH-specific static-analysis rule. Do not add
new `scripts/cicd/enforce_*.py` scripts; the meta gate rejects unmanifested
bespoke enforcers during the migration window.

Before proposing a new rule, read [the rationale](rationale.md) for the rule
taxonomy, severity policy, allowlist discipline, and lifecycle expectations.

## Add A Rule

1. Create a rule package under `elspeth-lints/src/elspeth_lints/rules/`.
2. Implement the `Rule` protocol from `elspeth_lints.core.protocols`.
3. Export a `RULE` object from the package.
4. Add that object to `BUILTIN_RULES` in
   `elspeth-lints/src/elspeth_lints/rules/__init__.py`.
5. Add `examples_violation` and `examples_clean` fixtures under the rule
   package, plus focused tests under `tests/unit/elspeth_lints/`.
6. If the rule ports an existing `scripts/cicd/enforce_*.py` check, add or
   update the entry in `config/cicd/lint_migration_status.yaml`.

Set `RuleMetadata.path_filter` to the rule's real ownership boundary. The CLI
uses that filter for incremental rules: full-root scans skip out-of-scope files,
and explicit `--files` input outside the selected rules' filters fails with
exit code `2`. This keeps pre-commit trigger scopes and rule ownership
mechanical instead of relying on comments near hook definitions.

Ports must move through the manifest lifecycle:

```text
pending -> shadow -> cutover -> deleted
```

`shadow` entries must include command templates for both the legacy script and
the new rule. The parity harness compares their normalized JSON findings:

```bash
PYTHONPATH=elspeth-lints/src .venv/bin/python scripts/cicd/parity_harness.py \
  --manifest config/cicd/lint_migration_status.yaml \
  --root .
```

The harness treats exit codes `0` and `1` as successful command execution
because enforcers conventionally return `1` when they find violations. Any
finding-set difference fails the run.

## Toy Rule

```python
from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from elspeth_lints.core.protocols import (
    Category,
    Finding,
    RuleContext,
    RuleMetadata,
    RuleScope,
    Severity,
)


class NoPrintRule:
    id = "example.no-print"
    scope = RuleScope.INCREMENTAL
    metadata = RuleMetadata(
        id=id,
        name="No print",
        description="Use audit or explicit user-facing output instead of print.",
        severity=Severity.ERROR,
        category=Category.MANIFEST,
        cwe=(),
        scope=scope,
        path_filter=r".*\.py$",
        examples_violation_count=1,
        examples_clean_count=1,
    )

    def analyze(
        self,
        tree: ast.AST,
        file_path: Path,
        context: RuleContext,
    ) -> Iterable[Finding]:
        del context
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                yield Finding(
                    rule_id=self.id,
                    file_path=str(file_path),
                    line=node.lineno,
                    column=node.col_offset,
                    message="print() is not allowed here",
                    fingerprint=f"print:{node.lineno}:{node.col_offset}",
                    severity=self.metadata.severity,
                )


RULE = NoPrintRule()
```

## Run Locally

```bash
PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check \
  --rules example.no-print \
  --root .
```

For installed-package proof:

```bash
uv pip install --python .venv/bin/python -e ./elspeth-lints
.venv/bin/elspeth-lints check --rules nothing --root /tmp
```

## Worked Example

`plugin_contract.options_metadata` is the v1.0 pilot rule. It demonstrates the
expected package shape:

```text
elspeth-lints/src/elspeth_lints/rules/plugin_contract/options_metadata/
├── __init__.py
├── metadata.py
├── rule.py
└── fixtures/
    ├── examples_violation/
    └── examples_clean/
```

CI runs `plugin_contract.options_metadata` through `elspeth-lints check`, while
`config/cicd/lint_migration_status.yaml` keeps the deleted legacy mapping for
parity history. Use this port as the reference for metadata placement, fixture
counts, command wiring, and protocol-compatible runtime dependencies.
