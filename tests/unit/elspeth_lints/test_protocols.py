"""Protocol tests for elspeth-lints rule authors."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from elspeth_lints.core.protocols import Category, Finding, Rule, RuleContext, RuleMetadata, RuleScope, Severity


class StubRule:
    """Small rule implementation proving the public protocol shape."""

    id = "stub.rule"
    scope = RuleScope.INCREMENTAL
    metadata = RuleMetadata(
        id=id,
        name="Stub rule",
        description="Used only to prove protocol conformance.",
        severity=Severity.NOTE,
        category=Category.MANIFEST,
        cwe=(),
        scope=scope,
        path_filter=r".*\.py$",
        examples_violation_count=1,
        examples_clean_count=1,
    )

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> Iterable[Finding]:
        assert isinstance(tree, ast.Module)
        assert context.root == Path(".")
        return (
            Finding(
                rule_id=self.id,
                severity=self.metadata.severity,
                file_path=str(file_path),
                line=1,
                column=0,
                message="stub finding",
                fingerprint="stub",
            ),
        )


def test_stub_rule_satisfies_rule_protocol() -> None:
    """A rule author can implement the contract without inheriting a base class."""
    rule: Rule = StubRule()
    findings = list(rule.analyze(ast.Module(body=[], type_ignores=[]), Path("sample.py"), RuleContext(root=Path("."))))

    assert rule.metadata.category is Category.MANIFEST
    assert findings == [
        Finding(
            rule_id="stub.rule",
            severity=Severity.NOTE,
            file_path="sample.py",
            line=1,
            column=0,
            message="stub finding",
            fingerprint="stub",
        )
    ]
