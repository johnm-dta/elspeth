"""Rule registry for elspeth-lints."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from elspeth_lints.core.protocols import Category, Finding, Rule, RuleContext, RuleMetadata, RuleScope, Severity

RuleCallable = Callable[[ast.AST, Path], Iterable[Finding]]


@dataclass(frozen=True, slots=True)
class FunctionRule:
    """Adapter for simple function-based rules."""

    id: str
    func: RuleCallable
    metadata: RuleMetadata
    scope: RuleScope = RuleScope.INCREMENTAL

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> Iterable[Finding]:
        """Run the wrapped function rule."""
        return self.func(tree, file_path)


class RuleRegistry:
    """Registry of named static-analysis rules."""

    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def rule(self, rule_id: str) -> Callable[[RuleCallable], RuleCallable]:
        """Return a decorator that registers a rule callable."""
        if not rule_id:
            raise ValueError("rule_id must be non-empty")

        def decorator(func: RuleCallable) -> RuleCallable:
            self.register_callable(rule_id, func)
            return func

        return decorator

    def register(self, rule: Rule) -> None:
        """Register a protocol-style rule."""
        if not rule.id:
            raise ValueError("rule id must be non-empty")
        if rule.id in self._rules:
            raise ValueError(f"rule {rule.id!r} is already registered")
        self._rules[rule.id] = rule

    def register_callable(self, rule_id: str, func: RuleCallable, *, metadata: RuleMetadata | None = None) -> None:
        """Register a function rule by wrapping it in the public protocol."""
        rule_metadata = metadata
        if rule_metadata is None:
            rule_metadata = RuleMetadata(
                id=rule_id,
                name=rule_id,
                description="Function rule registered without explicit metadata.",
                severity=Severity.ERROR,
                category=Category.MANIFEST,
                cwe=(),
                scope=RuleScope.INCREMENTAL,
                path_filter=r".*\.py$",
                examples_violation_count=0,
                examples_clean_count=0,
            )
        self.register(FunctionRule(id=rule_id, func=func, metadata=rule_metadata, scope=rule_metadata.scope))

    def get(self, rule_id: str) -> Rule:
        """Return a registered rule or raise KeyError."""
        return self._rules[rule_id]

    def ids(self) -> tuple[str, ...]:
        """Return registered rule ids in deterministic order."""
        return tuple(sorted(self._rules))

    def items(self) -> Iterator[tuple[str, Rule]]:
        """Yield registered rules in deterministic order."""
        for rule_id in self.ids():
            yield rule_id, self._rules[rule_id]

    def load_builtin_rules(self) -> None:
        """Load rules shipped inside the workspace package."""
        from elspeth_lints.rules import BUILTIN_RULES

        for rule in BUILTIN_RULES:
            if rule.id not in self._rules:
                self.register(rule)

    def load_entry_points(self, group: str = "elspeth_lints.rules") -> None:
        """Load rule objects or callables exposed through package entry points."""
        entry_points = metadata.entry_points()
        selected = entry_points.select(group=group)
        for entry_point in selected:
            loaded: Any = entry_point.load()
            if isinstance(loaded, Rule):
                if loaded.id not in self._rules:
                    self.register(loaded)
            elif callable(loaded):
                self.register_callable(entry_point.name, loaded)
            else:
                raise TypeError(f"entry point {entry_point.name!r} did not load a rule or callable")


DEFAULT_REGISTRY = RuleRegistry()
