# elspeth-lints Rule Author Guide

This guide is for adding a new ELSPETH-specific static-analysis rule. Do not add
new `scripts/cicd/enforce_*.py` scripts; the meta gate rejects unmanifested
bespoke enforcers during the migration window.

## Add A Rule

1. Create a module under `elspeth-lints/src/elspeth_lints/rules/`.
2. Implement the `Rule` protocol from `elspeth_lints.core.protocols`.
3. Export a `RULE` object from the module.
4. Add that object to `BUILTIN_RULES` in
   `elspeth-lints/src/elspeth_lints/rules/__init__.py`.
5. Add focused fixtures and tests under `tests/unit/elspeth_lints/`.
6. If the rule ports an existing `scripts/cicd/enforce_*.py` check, add or
   update the entry in `config/cicd/lint_migration_status.yaml`.

Ports must move through the manifest lifecycle:

```text
pending -> shadow -> cutover -> deleted
```

The parity harness task owns the behavior proof for `shadow` entries. Until that
task lands, existing scripts stay `pending`.

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
