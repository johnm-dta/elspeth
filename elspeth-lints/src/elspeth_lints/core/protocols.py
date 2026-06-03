"""Public contracts for elspeth-lints rules and emitters."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from elspeth_lints.core.allowlist import Allowlist


class Severity(StrEnum):
    """Finding severity levels mapped to SARIF levels."""

    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"


class Category(StrEnum):
    """Rule categories used for migration planning and reviewer ownership."""

    TRUST_TIER = "trust_tier"
    IMMUTABILITY = "immutability"
    AUDIT_EVIDENCE = "audit_evidence"
    PLUGIN_CONTRACT = "plugin_contract"
    COMPOSER = "composer"
    MANIFEST = "manifest"


class RuleScope(StrEnum):
    """Execution scope for a rule."""

    INCREMENTAL = "incremental"
    WHOLE_REPO = "whole_repo"


@dataclass(frozen=True, slots=True)
class Finding:
    """A single static-analysis finding."""

    rule_id: str
    file_path: str
    line: int
    column: int
    message: str
    fingerprint: str
    severity: Severity = Severity.ERROR
    suggestion: str | None = None
    symbol_context: tuple[str, ...] = ()
    ast_path: str = ""

    def canonical_key(self, symbol_context: tuple[str, ...] | None = None) -> str:
        """Return the allowlist key used for exact finding suppression."""
        if symbol_context is None:
            symbol_context = self.symbol_context
        symbol_part = ":".join(symbol_context) if symbol_context else "_module_"
        return f"{self.file_path}:{self.rule_id}:{symbol_part}:fp={self.fingerprint}"


@dataclass(frozen=True, slots=True)
class RuleMetadata:
    """Stable metadata a rule must expose before it can join CI."""

    id: str
    name: str
    description: str
    severity: Severity
    category: Category
    cwe: tuple[str, ...]
    scope: RuleScope
    path_filter: str
    examples_violation_count: int
    examples_clean_count: int


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Repository context shared with rule implementations.

    ``allowlist_dir_override`` lets the CLI (``--allowlist-dir``) force every
    rule to load its allowlist from a single shared directory instead of each
    rule's per-rule default. Useful for shadow runs and cross-branch
    comparisons. ``None`` means "use the rule's own default" — the historical
    behaviour.

    ``repo_root`` is an explicit repository root for whole-repository rules
    whose scan root may be a subdirectory (for example ``src/elspeth``) but
    whose evidence paths are repository-relative (for example ``tests/...``).
    ``None`` means "derive from ``root`` using the rule's documented fallback".
    """

    root: Path
    allowlist: Allowlist | None = None
    allowlist_dir_override: Path | None = None
    repo_root: Path | None = None


@runtime_checkable
class Rule(Protocol):
    """Structural protocol implemented by all elspeth-lints rules."""

    @property
    def id(self) -> str:
        """Stable rule identifier."""

    @property
    def metadata(self) -> RuleMetadata:
        """Reviewer-facing rule metadata."""

    @property
    def scope(self) -> RuleScope:
        """Rule execution scope."""

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> Iterable[Finding]:
        """Analyze one syntax tree or a whole repository root."""
