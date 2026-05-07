# Guard Symmetry CI Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CI enforcer that detects when audit dataclasses have `__post_init__` validation but their corresponding model loaders lack `AuditIntegrityError` guards — preventing asymmetric write/read guard coverage.

**Architecture:** Single-file AST scanner (`scripts/cicd/enforce_guard_symmetry.py`) following the exact pattern of `enforce_freeze_guards.py`. Two-phase scan: (1) discover dataclass→loader pairs by naming convention, (2) check if each loader's `load()` method raises `AuditIntegrityError`. Allowlist via YAML directory under `config/cicd/enforce_guard_symmetry/`.

**Tech Stack:** Python stdlib `ast`, `argparse`, `hashlib`, `dataclasses`; `pyyaml` for allowlist config.

**Filigree Issue:** `elspeth-5f37dcce91` (P1 task)

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `scripts/cicd/enforce_guard_symmetry.py` | The scanner (~400-500 lines) |
| Create | `tests/unit/scripts/cicd/test_enforce_guard_symmetry.py` | Unit tests (~250 lines) |
| Create | `config/cicd/enforce_guard_symmetry/_defaults.yaml` | Allowlist defaults |
| Create | `config/cicd/enforce_guard_symmetry/landscape.yaml` | Initial allowlist for expected findings |
| Create | `.github/workflows/enforce-guard-symmetry.yaml` | CI workflow |

No existing files are modified.

## Design Decisions

**One rule (GS1):** Coarse-grained "does the loader have ANY `AuditIntegrityError`?" — not field-level matching. Future rules can refine.

**Pairing convention:** `ClassName` → `ClassNameLoader`. Override map handles two naming mismatches:
- NodeState variants (4 classes → 1 loader): `NodeStateOpen` → `NodeStateLoader`
- `*Record`-suffixed dataclasses: `TransformErrorRecord` → `TransformErrorLoader`, `ValidationErrorRecord` → `ValidationErrorLoader`

Dataclasses without a matching loader are silently skipped (no finding).

**Detection scope:** Any `@dataclass` with a `__post_init__` method that contains `raise` statements or calls to known validation functions (`require_int`, `_validate_enum`). Freeze-only `__post_init__` methods (only `freeze_fields` calls) are excluded to reduce false positives.

**Abstract loader skip:** Classes ending in `Loader` that have an abstract or stub `load()` method (body is only `pass`, `...`, or `raise NotImplementedError`) are excluded from loader discovery. These are Protocol/ABC definitions, not concrete loaders.

**Pre-commit mode is best-effort only.** This scanner pairs dataclasses (in `contracts/audit.py`) with loaders (in `core/landscape/model_loaders.py`). In pre-commit mode, only changed files are scanned. If only one file of the pair changes, the scanner will find no pair to check and pass silently. **CI full-scan is the authoritative enforcement gate.** Pre-commit provides value when both files change in the same commit, but cannot guarantee detection otherwise.

**Known limitation — indirect AuditIntegrityError:** The scanner only detects `raise AuditIntegrityError(...)` directly within the `load()` method body. If a loader delegates to a helper function that raises AIE, the scanner will not detect it (false negative). This is acceptable for the coarse-grained first pass — developers should add the AIE raise directly in `load()` for scanner visibility, or add an allowlist entry if using a helper pattern.

**Fingerprint scheme differs from other enforcers.** The freeze_guards and tier_model enforcers fingerprint from `ast.dump()` of the detected AST node. This scanner generates findings post-scan from summary `DataclassInfo`/`LoaderInfo` structs (no AST node available at that point). The fingerprint uses `rule_id + loader_file + loader_name + dataclass_name`. This is stable across formatting changes and only shifts on class renames. The divergence is intentional.

**Standalone script (no shared module).** This scanner, like `enforce_freeze_guards.py` and `enforce_tier_model.py`, is a standalone script with no shared imports. The `Finding`, `PerFileRule`, `Allowlist`, and YAML loading code are duplicated across all three enforcers. This is deliberate — CI scripts must be independently executable with no cross-script import dependencies. If the team later decides to extract a shared `scripts/cicd/_allowlist.py` module, all three scripts can be updated together.

---

### Task 1: Scaffold + Data Structures

**Files:**
- Create: `scripts/cicd/enforce_guard_symmetry.py`
- Create: `tests/unit/scripts/cicd/test_enforce_guard_symmetry.py`

- [ ] **Step 1: Create test file with import smoke test**

```python
# tests/unit/scripts/cicd/test_enforce_guard_symmetry.py
"""Unit tests for the guard symmetry enforcement tool."""

from __future__ import annotations

from scripts.cicd.enforce_guard_symmetry import (
    RULES,
    Finding,
    PerFileRule,
    Allowlist,
)


class TestDataStructures:
    """Tests for core data structures."""

    def test_rules_defined(self) -> None:
        assert "GS1" in RULES
        assert RULES["GS1"]["name"] == "missing-read-guard"

    def test_finding_canonical_key(self) -> None:
        f = Finding(
            rule_id="GS1",
            file_path="core/landscape/model_loaders.py",
            line=10,
            col=0,
            symbol_context=("RunLoader", "load"),
            fingerprint="abc123",
            code_snippet="return Run(",
            message="test",
        )
        assert f.canonical_key == "core/landscape/model_loaders.py:GS1:RunLoader:load:fp=abc123"

    def test_allowlist_match(self) -> None:
        rule = PerFileRule(
            pattern="core/landscape/*",
            rules=["GS1"],
            reason="test",
            expires=None,
        )
        al = Allowlist(per_file_rules=[rule])
        f = Finding(
            rule_id="GS1",
            file_path="core/landscape/model_loaders.py",
            line=10,
            col=0,
            symbol_context=("RunLoader",),
            fingerprint="abc123",
            code_snippet="return Run(",
            message="test",
        )
        assert al.match(f) is not None

    def test_allowlist_no_match(self) -> None:
        al = Allowlist(per_file_rules=[])
        f = Finding(
            rule_id="GS1",
            file_path="core/landscape/model_loaders.py",
            line=10,
            col=0,
            symbol_context=("RunLoader",),
            fingerprint="abc123",
            code_snippet="return Run(",
            message="test",
        )
        assert al.match(f) is None
```

- [ ] **Step 2: Run test to verify it fails (module not found)**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_guard_symmetry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.cicd.enforce_guard_symmetry'`

- [ ] **Step 3: Create scanner with data structures**

```python
#!/usr/bin/env python3
"""Enforce write/read guard symmetry for audit dataclasses.

Detects dataclasses with __post_init__ validation whose corresponding
model loaders lack AuditIntegrityError guards. Prevents asymmetric
coverage where write-side checks exist but read-side checks don't.

Usage:
    python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth
    python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth --allowlist config/cicd/enforce_guard_symmetry
    python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth file1.py file2.py
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

# =============================================================================
# Data Structures
# =============================================================================

RULES: dict[str, dict[str, str]] = {
    "GS1": {
        "name": "missing-read-guard",
        "description": "Dataclass has __post_init__ validation but corresponding loader has no AuditIntegrityError",
        "remediation": "Add AuditIntegrityError validation to the loader's load() method for Tier 1 read-side integrity",
    },
}

_ALL_RULE_IDS = frozenset(RULES.keys())

# Dataclass→Loader name overrides for non-standard naming conventions.
# Default rule: ClassName → ClassNameLoader. These cover two patterns:
# 1. NodeState discriminated union: 4 variant classes → 1 loader
# 2. *Record-suffixed dataclasses: loader drops the "Record" suffix
LOADER_OVERRIDES: dict[str, str] = {
    "NodeStateOpen": "NodeStateLoader",
    "NodeStatePending": "NodeStateLoader",
    "NodeStateCompleted": "NodeStateLoader",
    "NodeStateFailed": "NodeStateLoader",
    "TransformErrorRecord": "TransformErrorLoader",
    "ValidationErrorRecord": "ValidationErrorLoader",
}

# Functions called in __post_init__ that indicate validation (not just freezing)
_VALIDATION_FUNCTIONS = frozenset({
    "require_int",
    "_validate_enum",
    "validate",
})


def expected_loader_name(class_name: str) -> str:
    """Map a dataclass name to its expected loader class name."""
    return LOADER_OVERRIDES.get(class_name, f"{class_name}Loader")


@dataclass(frozen=True)
class Finding:
    """A detected guard symmetry violation."""

    rule_id: str
    file_path: str
    line: int
    col: int
    symbol_context: tuple[str, ...]
    fingerprint: str
    code_snippet: str
    message: str

    @property
    def canonical_key(self) -> str:
        symbol_part = ":".join(self.symbol_context) if self.symbol_context else "_module_"
        return f"{self.file_path}:{self.rule_id}:{symbol_part}:fp={self.fingerprint}"


@dataclass
class PerFileRule:
    """A per-file rule that allowlists patterns of specified rules for a file."""

    pattern: str
    rules: list[str]
    reason: str
    expires: date | None
    max_hits: int | None = None
    matched_count: int = field(default=0, compare=False)
    source_file: str = field(default="", compare=False)

    def matches(self, file_path: str, rule_id: str) -> bool:
        if rule_id not in self.rules:
            return False
        return fnmatch.fnmatch(file_path, self.pattern)


@dataclass
class Allowlist:
    """Parsed allowlist configuration."""

    per_file_rules: list[PerFileRule] = field(default_factory=list)
    fail_on_stale: bool = True

    def match(self, finding: Finding) -> PerFileRule | None:
        for rule in self.per_file_rules:
            if rule.matches(finding.file_path, finding.rule_id):
                rule.matched_count += 1
                return rule
        return None

    def get_unused_rules(self) -> list[PerFileRule]:
        return [r for r in self.per_file_rules if r.matched_count == 0]

    def get_expired_rules(self) -> list[PerFileRule]:
        today = datetime.now(UTC).date()
        return [r for r in self.per_file_rules if r.expires and r.expires < today]

    def get_exceeded_rules(self) -> list[PerFileRule]:
        return [r for r in self.per_file_rules if r.max_hits is not None and r.matched_count > r.max_hits]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_guard_symmetry.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/cicd/enforce_guard_symmetry.py tests/unit/scripts/cicd/test_enforce_guard_symmetry.py
git commit -m "feat(ci): scaffold guard symmetry scanner with data structures

Adds Finding, Allowlist, PerFileRule types following enforce_freeze_guards pattern.
Single rule GS1: missing-read-guard for dataclass→loader pair asymmetry."
```

---

### Task 2: Discovery — Dataclass and Loader Detection

**Files:**
- Modify: `tests/unit/scripts/cicd/test_enforce_guard_symmetry.py`
- Modify: `scripts/cicd/enforce_guard_symmetry.py`

- [ ] **Step 1: Write tests for dataclass discovery**

Add to the test file:

```python
import ast
from textwrap import dedent

from scripts.cicd.enforce_guard_symmetry import (
    DataclassInfo,
    LoaderInfo,
    GuardSymmetryVisitor,
)


def scan_source(source: str, filename: str = "test.py") -> GuardSymmetryVisitor:
    """Helper to parse source and run the visitor."""
    tree = ast.parse(source, filename=filename)
    source_lines = source.splitlines()
    visitor = GuardSymmetryVisitor(filename, source_lines)
    visitor.visit(tree)
    return visitor


class TestDataclassDiscovery:
    """Tests for discovering dataclasses with __post_init__ validation."""

    def test_finds_dataclass_with_validation_post_init(self) -> None:
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Token:
                token_id: str
                step: int

                def __post_init__(self) -> None:
                    if not isinstance(self.step, int):
                        raise TypeError("step must be int")
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 1
        assert visitor.dataclasses[0].name == "Token"

    def test_ignores_dataclass_without_post_init(self) -> None:
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Simple:
                name: str
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 0

    def test_ignores_freeze_only_post_init(self) -> None:
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class FreezeOnly:
                data: dict

                def __post_init__(self) -> None:
                    freeze_fields(self, "data")
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 0

    def test_detects_require_int_as_validation(self) -> None:
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Row:
                row_index: int

                def __post_init__(self) -> None:
                    require_int(self.row_index, "row_index", min_value=0)
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 1
        assert visitor.dataclasses[0].name == "Row"

    def test_detects_validate_enum_as_validation(self) -> None:
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class TokenOutcome:
                outcome: str

                def __post_init__(self) -> None:
                    _validate_enum(self.outcome, RowOutcome, "outcome")
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 1

    def test_mixed_freeze_and_validation(self) -> None:
        """__post_init__ with both freeze_fields and validation should be detected."""
        source = dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Mixed:
                data: dict
                step: int

                def __post_init__(self) -> None:
                    freeze_fields(self, "data")
                    require_int(self.step, "step", min_value=0)
        """)
        visitor = scan_source(source)
        assert len(visitor.dataclasses) == 1
```

- [ ] **Step 2: Write tests for loader discovery**

Add to the test file:

```python
class TestLoaderDiscovery:
    """Tests for discovering *Loader classes and checking for AuditIntegrityError."""

    def test_finds_loader_with_audit_integrity_error(self) -> None:
        source = dedent("""
            class TokenOutcomeLoader:
                def load(self, row):
                    if row.is_terminal not in (0, 1):
                        raise AuditIntegrityError("bad is_terminal")
                    return TokenOutcome(outcome_id=row.outcome_id)
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 1
        assert visitor.loaders[0].name == "TokenOutcomeLoader"
        assert visitor.loaders[0].target_class == "TokenOutcome"
        assert visitor.loaders[0].has_audit_integrity_error is True

    def test_finds_loader_without_audit_integrity_error(self) -> None:
        source = dedent("""
            class RunLoader:
                def load(self, row):
                    return Run(run_id=row.run_id, status=RunStatus(row.status))
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 1
        assert visitor.loaders[0].name == "RunLoader"
        assert visitor.loaders[0].target_class == "Run"
        assert visitor.loaders[0].has_audit_integrity_error is False

    def test_ignores_class_without_load_method(self) -> None:
        source = dedent("""
            class NotALoader:
                def process(self, row):
                    return row
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 0

    def test_ignores_class_not_ending_in_loader(self) -> None:
        source = dedent("""
            class RunHelper:
                def load(self, row):
                    return row
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 0

    def test_target_class_derived_from_name(self) -> None:
        source = dedent("""
            class RoutingEventLoader:
                def load(self, row):
                    return RoutingEvent(event_id=row.event_id)
        """)
        visitor = scan_source(source)
        assert visitor.loaders[0].target_class == "RoutingEvent"

    def test_skips_abstract_loader_with_ellipsis_body(self) -> None:
        """Protocol/ABC loaders with stub bodies should not be collected."""
        source = dedent("""
            class SecretLoader:
                def load(self, key: str) -> str:
                    ...
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 0

    def test_skips_abstract_loader_with_pass_body(self) -> None:
        source = dedent("""
            class BaseLoader:
                def load(self, row):
                    pass
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 0

    def test_skips_abstract_loader_with_not_implemented(self) -> None:
        source = dedent("""
            class AbstractLoader:
                def load(self, row):
                    raise NotImplementedError
        """)
        visitor = scan_source(source)
        assert len(visitor.loaders) == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_guard_symmetry.py -v -k "Discovery"`
Expected: FAIL — `ImportError: cannot import name 'GuardSymmetryVisitor'`

- [ ] **Step 4: Implement discovery types and visitor**

Add to `scripts/cicd/enforce_guard_symmetry.py` after the Allowlist class:

```python
@dataclass(frozen=True)
class DataclassInfo:
    """A discovered dataclass with __post_init__ validation."""

    name: str
    file_path: str
    line: int


@dataclass(frozen=True)
class LoaderInfo:
    """A discovered *Loader class with a load() method."""

    name: str
    file_path: str
    line: int
    target_class: str
    has_audit_integrity_error: bool


# =============================================================================
# AST Visitor
# =============================================================================


class GuardSymmetryVisitor(ast.NodeVisitor):
    """AST visitor that discovers dataclass→loader pairs and checks guard coverage."""

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.dataclasses: list[DataclassInfo] = []
        self.loaders: list[LoaderInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Check if this is a @dataclass with __post_init__
        if self._is_dataclass(node):
            post_init = self._find_post_init(node)
            if post_init is not None and self._post_init_has_validation(post_init):
                self.dataclasses.append(
                    DataclassInfo(
                        name=node.name,
                        file_path=self.file_path,
                        line=node.lineno,
                    )
                )

        # Check if this is a *Loader with a concrete load() method
        if node.name.endswith("Loader") and len(node.name) > len("Loader"):
            load_method = self._find_load_method(node)
            if load_method is not None and not self._is_abstract_method(load_method):
                self.loaders.append(
                    LoaderInfo(
                        name=node.name,
                        file_path=self.file_path,
                        line=node.lineno,
                        target_class=node.name.removesuffix("Loader"),
                        has_audit_integrity_error=self._method_raises_aie(load_method),
                    )
                )

        self.generic_visit(node)

    def _is_dataclass(self, node: ast.ClassDef) -> bool:
        """Check if a class has a @dataclass decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                return True
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Name) and func.id == "dataclass":
                    return True
                if isinstance(func, ast.Attribute) and func.attr == "dataclass":
                    return True
        return False

    def _find_post_init(self, node: ast.ClassDef) -> ast.FunctionDef | None:
        """Find __post_init__ method in a class body."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__post_init__":
                return item
        return None

    def _find_load_method(self, node: ast.ClassDef) -> ast.FunctionDef | None:
        """Find load() method in a class body."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "load":
                return item
        return None

    def _post_init_has_validation(self, method: ast.FunctionDef) -> bool:
        """Check if __post_init__ does validation (not just freeze_fields).

        Returns True if the method contains:
        - raise statements
        - Calls to known validation functions (require_int, _validate_enum)
        """
        for node in ast.walk(method):
            # Direct raise statements
            if isinstance(node, ast.Raise):
                return True
            # Calls to known validation functions
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _VALIDATION_FUNCTIONS:
                    return True
        return False

    def _is_abstract_method(self, method: ast.FunctionDef) -> bool:
        """Check if a method is abstract (stub body: pass, ..., or raise NotImplementedError).

        Excludes Protocol/ABC loaders from scanner — they define interfaces,
        not concrete load() implementations that need guard checking.
        """
        body = method.body
        # Skip docstring if present
        stmts = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
        if len(stmts) != 1:
            return False
        stmt = stmts[0]
        # pass
        if isinstance(stmt, ast.Pass):
            return True
        # ... (Ellipsis)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
            return True
        # raise NotImplementedError
        if isinstance(stmt, ast.Raise) and stmt.exc is not None:
            exc = stmt.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return True
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                return True
        return False

    def _method_raises_aie(self, method: ast.FunctionDef) -> bool:
        """Check if a method raises AuditIntegrityError directly in its body.

        Known limitation: does NOT detect AIE raised by helper functions called
        from load(). If a loader delegates to a helper that raises AIE, the
        scanner will report a false negative. This is acceptable for the
        coarse-grained first pass — add the raise directly in load() for
        scanner visibility, or allowlist the finding if using a helper pattern.
        """
        for node in ast.walk(method):
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc = node.exc
                if isinstance(exc, ast.Call):
                    func = exc.func
                    if isinstance(func, ast.Name) and func.id == "AuditIntegrityError":
                        return True
                    if isinstance(func, ast.Attribute) and func.attr == "AuditIntegrityError":
                        return True
        return False
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_guard_symmetry.py -v -k "Discovery"`
Expected: ALL PASSED (11 tests)

- [ ] **Step 6: Commit**

```bash
git add scripts/cicd/enforce_guard_symmetry.py tests/unit/scripts/cicd/test_enforce_guard_symmetry.py
git commit -m "feat(ci): add AST discovery for dataclass/loader pairs

GuardSymmetryVisitor finds frozen dataclasses with __post_init__ validation
and *Loader classes, checking for AuditIntegrityError in load() methods.
Excludes freeze-only __post_init__ to reduce false positives."
```

---

### Task 3: Pairing Logic and Finding Generation

**Files:**
- Modify: `tests/unit/scripts/cicd/test_enforce_guard_symmetry.py`
- Modify: `scripts/cicd/enforce_guard_symmetry.py`

- [ ] **Step 1: Write tests for pairing and finding generation**

Add to the test file:

```python
from scripts.cicd.enforce_guard_symmetry import (
    expected_loader_name,
    find_unguarded_pairs,
)


class TestPairing:
    """Tests for dataclass→loader pairing and finding generation."""

    def test_expected_loader_name_default(self) -> None:
        assert expected_loader_name("Run") == "RunLoader"
        assert expected_loader_name("TokenOutcome") == "TokenOutcomeLoader"

    def test_expected_loader_name_node_state_overrides(self) -> None:
        assert expected_loader_name("NodeStateOpen") == "NodeStateLoader"
        assert expected_loader_name("NodeStatePending") == "NodeStateLoader"
        assert expected_loader_name("NodeStateCompleted") == "NodeStateLoader"
        assert expected_loader_name("NodeStateFailed") == "NodeStateLoader"

    def test_expected_loader_name_record_suffix_overrides(self) -> None:
        """*Record-suffixed dataclasses map to loaders that drop the 'Record' suffix."""
        assert expected_loader_name("TransformErrorRecord") == "TransformErrorLoader"
        assert expected_loader_name("ValidationErrorRecord") == "ValidationErrorLoader"

    def test_unguarded_pair_produces_finding(self) -> None:
        dcs = [DataclassInfo(name="Run", file_path="contracts/audit.py", line=48)]
        loaders = [LoaderInfo(
            name="RunLoader",
            file_path="core/landscape/model_loaders.py",
            line=53,
            target_class="Run",
            has_audit_integrity_error=False,
        )]
        findings = find_unguarded_pairs(dcs, loaders, "core/landscape/model_loaders.py")
        assert len(findings) == 1
        assert findings[0].rule_id == "GS1"
        assert "RunLoader" in findings[0].message

    def test_guarded_pair_produces_no_finding(self) -> None:
        dcs = [DataclassInfo(name="TokenOutcome", file_path="contracts/audit.py", line=642)]
        loaders = [LoaderInfo(
            name="TokenOutcomeLoader",
            file_path="core/landscape/model_loaders.py",
            line=467,
            target_class="TokenOutcome",
            has_audit_integrity_error=True,
        )]
        findings = find_unguarded_pairs(dcs, loaders, "core/landscape/model_loaders.py")
        assert len(findings) == 0

    def test_dataclass_without_loader_skipped(self) -> None:
        dcs = [DataclassInfo(name="Orphan", file_path="contracts/audit.py", line=100)]
        loaders: list[LoaderInfo] = []
        findings = find_unguarded_pairs(dcs, loaders, "core/landscape/model_loaders.py")
        assert len(findings) == 0

    def test_node_state_variants_use_override(self) -> None:
        """All 4 NodeState variants should check NodeStateLoader."""
        dcs = [
            DataclassInfo(name="NodeStateOpen", file_path="contracts/audit.py", line=174),
            DataclassInfo(name="NodeStateCompleted", file_path="contracts/audit.py", line=232),
        ]
        loaders = [LoaderInfo(
            name="NodeStateLoader",
            file_path="core/landscape/model_loaders.py",
            line=253,
            target_class="NodeState",
            has_audit_integrity_error=True,
        )]
        findings = find_unguarded_pairs(dcs, loaders, "core/landscape/model_loaders.py")
        assert len(findings) == 0

    def test_multiple_findings_from_multiple_pairs(self) -> None:
        dcs = [
            DataclassInfo(name="Run", file_path="contracts/audit.py", line=48),
            DataclassInfo(name="Call", file_path="contracts/audit.py", line=298),
        ]
        loaders = [
            LoaderInfo(name="RunLoader", file_path="core/landscape/model_loaders.py",
                       line=53, target_class="Run", has_audit_integrity_error=False),
            LoaderInfo(name="CallLoader", file_path="core/landscape/model_loaders.py",
                       line=190, target_class="Call", has_audit_integrity_error=False),
        ]
        findings = find_unguarded_pairs(dcs, loaders, "core/landscape/model_loaders.py")
        assert len(findings) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_guard_symmetry.py -v -k "Pairing"`
Expected: FAIL — `ImportError: cannot import name 'find_unguarded_pairs'`

- [ ] **Step 3: Implement pairing logic**

Add to `scripts/cicd/enforce_guard_symmetry.py` after the visitor class:

```python
# =============================================================================
# Pairing Logic
# =============================================================================


def find_unguarded_pairs(
    dataclasses: list[DataclassInfo],
    loaders: list[LoaderInfo],
    loader_file_path: str,
) -> list[Finding]:
    """Find dataclass→loader pairs where the loader lacks AuditIntegrityError.

    Args:
        dataclasses: All discovered dataclasses with __post_init__ validation
        loaders: All discovered loader classes
        loader_file_path: Relative path to the loader file (for finding context)

    Returns:
        Findings for each unguarded pair (GS1 rule)
    """
    loader_by_name: dict[str, LoaderInfo] = {loader.name: loader for loader in loaders}
    findings: list[Finding] = []

    for dc in dataclasses:
        loader_name = expected_loader_name(dc.name)
        loader = loader_by_name.get(loader_name)
        if loader is None:
            continue  # No loader — nothing to check
        if loader.has_audit_integrity_error:
            continue  # Loader has guards — OK

        # Unguarded pair: dataclass has validation, loader doesn't
        context = (loader.name, "load")
        # Fingerprint differs from other enforcers (which use ast.dump).
        # Findings are generated post-scan from summary structs — no AST node
        # is available. Uses loader+class names which are stable across
        # formatting changes and only shift on class renames.
        fp_payload = f"GS1|{loader.file_path}|{loader.name}|{dc.name}"
        fingerprint = hashlib.sha256(fp_payload.encode("utf-8")).hexdigest()[:16]

        findings.append(
            Finding(
                rule_id="GS1",
                file_path=loader.file_path,
                line=loader.line,
                col=0,
                symbol_context=context,
                fingerprint=fingerprint,
                code_snippet=f"class {loader.name}:",
                message=(
                    f"{dc.name} has __post_init__ validation (at {dc.file_path}:{dc.line}) "
                    f"but {loader.name}.load() has no AuditIntegrityError guards"
                ),
            )
        )

    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_guard_symmetry.py -v -k "Pairing"`
Expected: ALL PASSED (7 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/cicd/enforce_guard_symmetry.py tests/unit/scripts/cicd/test_enforce_guard_symmetry.py
git commit -m "feat(ci): add dataclass→loader pairing with naming overrides

Pairs dataclasses to loaders by naming convention, generates GS1 findings
when a loader lacks AuditIntegrityError. Handles NodeState discriminated
union (4 variants → 1 loader) and *Record-suffixed dataclasses
(TransformErrorRecord → TransformErrorLoader, etc.)."
```

---

### Task 4: File Scanning and Allowlist Loading

**Files:**
- Modify: `tests/unit/scripts/cicd/test_enforce_guard_symmetry.py`
- Modify: `scripts/cicd/enforce_guard_symmetry.py`

- [ ] **Step 1: Write tests for file scanning**

Add to the test file:

```python
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from scripts.cicd.enforce_guard_symmetry import (
    scan_files,
    load_allowlist,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestFileScanning:
    """Tests for scanning files and producing findings."""

    def test_scan_finds_unguarded_pair_across_files(self, temp_dir: Path) -> None:
        """Dataclass in one file, loader in another — should produce finding."""
        contracts_dir = temp_dir / "contracts"
        contracts_dir.mkdir()
        core_dir = temp_dir / "core" / "landscape"
        core_dir.mkdir(parents=True)

        # Dataclass file
        (contracts_dir / "audit.py").write_text(dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Run:
                run_id: str
                status: str

                def __post_init__(self) -> None:
                    if self.status not in ("running", "done"):
                        raise ValueError("bad status")
        """))

        # Loader file — no AuditIntegrityError
        (core_dir / "model_loaders.py").write_text(dedent("""
            class RunLoader:
                def load(self, row):
                    return Run(run_id=row.run_id, status=row.status)
        """))

        findings = scan_files(temp_dir)
        assert len(findings) == 1
        assert findings[0].rule_id == "GS1"
        assert "RunLoader" in findings[0].message

    def test_scan_no_findings_when_guarded(self, temp_dir: Path) -> None:
        contracts_dir = temp_dir / "contracts"
        contracts_dir.mkdir()
        core_dir = temp_dir / "core" / "landscape"
        core_dir.mkdir(parents=True)

        (contracts_dir / "audit.py").write_text(dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Run:
                run_id: str
                status: str

                def __post_init__(self) -> None:
                    if self.status not in ("running", "done"):
                        raise ValueError("bad status")
        """))

        (core_dir / "model_loaders.py").write_text(dedent("""
            class RunLoader:
                def load(self, row):
                    if row.status not in ("running", "done"):
                        raise AuditIntegrityError("invalid status")
                    return Run(run_id=row.run_id, status=row.status)
        """))

        findings = scan_files(temp_dir)
        assert len(findings) == 0


class TestAllowlistLoading:
    """Tests for allowlist YAML loading."""

    def test_load_from_directory(self, temp_dir: Path) -> None:
        (temp_dir / "_defaults.yaml").write_text("version: 1\ndefaults:\n  fail_on_stale: true\n")
        (temp_dir / "landscape.yaml").write_text(dedent("""
            per_file_rules:
              - pattern: "core/landscape/*"
                rules: [GS1]
                reason: "int-only validation covered by __post_init__"
                expires: null
                max_hits: 5
        """))
        al = load_allowlist(temp_dir)
        assert len(al.per_file_rules) == 1
        assert al.fail_on_stale is True

    def test_load_empty_returns_default(self, temp_dir: Path) -> None:
        nonexistent = temp_dir / "no_such_file.yaml"
        al = load_allowlist(nonexistent)
        assert len(al.per_file_rules) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_guard_symmetry.py -v -k "FileScanning or AllowlistLoading"`
Expected: FAIL — `ImportError: cannot import name 'scan_files'`

- [ ] **Step 3: Implement file scanning and allowlist loading**

Add to `scripts/cicd/enforce_guard_symmetry.py`:

```python
# =============================================================================
# File Scanning
# =============================================================================


def _scan_single_file(file_path: Path, root: Path) -> GuardSymmetryVisitor:
    """Scan a single Python file and return the visitor with collected data."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return GuardSymmetryVisitor("", [])

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}", file=sys.stderr)
        return GuardSymmetryVisitor("", [])

    source_lines = source.splitlines()
    relative_path = str(file_path.relative_to(root))

    visitor = GuardSymmetryVisitor(relative_path, source_lines)
    visitor.visit(tree)
    return visitor


def scan_files(root: Path, files: list[Path] | None = None) -> list[Finding]:
    """Scan Python files and find unguarded dataclass→loader pairs.

    Phase 1: Discover all dataclasses with __post_init__ and all loaders.
    Phase 2: Pair them and report unguarded pairs.

    Args:
        root: Root directory for relative path computation
        files: Specific files to scan (None = scan all *.py under root)

    Returns:
        List of GS1 findings for unguarded pairs
    """
    all_dataclasses: list[DataclassInfo] = []
    all_loaders: list[LoaderInfo] = []

    if files:
        py_files = files
    else:
        py_files = list(root.rglob("*.py"))

    for py_file in py_files:
        resolved = py_file if py_file.is_absolute() else (root / py_file)
        if not resolved.is_file():
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            continue

        visitor = _scan_single_file(resolved, root)
        all_dataclasses.extend(visitor.dataclasses)
        all_loaders.extend(visitor.loaders)

    # Determine loader file path for findings context
    loader_file = ""
    if all_loaders:
        loader_file = all_loaders[0].file_path

    return find_unguarded_pairs(all_dataclasses, all_loaders, loader_file)


# =============================================================================
# Allowlist Handling
# =============================================================================


def _load_yaml_file(path: Path) -> dict:
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in allowlist {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _parse_per_file_rules(data: dict, source_file: str = "") -> list[PerFileRule]:
    rules: list[PerFileRule] = []
    for item in data.get("per_file_rules", []):
        rule_ids = set(item.get("rules", []))
        unknown = rule_ids - _ALL_RULE_IDS
        if unknown:
            ctx = f" in {source_file}" if source_file else ""
            print(
                f"Error: per_file_rules entry for '{item.get('pattern', '?')}'{ctx} uses unknown rule ID(s) {unknown}",
                file=sys.stderr,
            )
            sys.exit(1)

        expires_str = item.get("expires")
        expires_date = None
        if expires_str:
            try:
                expires_date = datetime.strptime(expires_str, "%Y-%m-%d").replace(tzinfo=UTC).date()
            except ValueError:
                print(f"Warning: Invalid date format for expires: {expires_str}", file=sys.stderr)

        max_hits: int | None = None
        raw_max_hits = item.get("max_hits")
        if raw_max_hits is not None:
            try:
                max_hits = int(raw_max_hits)
            except ValueError:
                print(
                    f"Error: non-numeric max_hits for '{item.get('pattern', '?')}': {raw_max_hits!r}",
                    file=sys.stderr,
                )
                sys.exit(1)

        rules.append(
            PerFileRule(
                pattern=item["pattern"],
                rules=item.get("rules", []),
                reason=item.get("reason", ""),
                expires=expires_date,
                max_hits=max_hits,
                source_file=source_file,
            )
        )
    return rules


def load_allowlist(path: Path) -> Allowlist:
    """Load allowlist from a directory of YAML files or a single file."""
    if path.is_dir():
        defaults_path = path / "_defaults.yaml"
        defaults = {}
        if defaults_path.exists():
            defaults_data = _load_yaml_file(defaults_path)
            defaults = defaults_data.get("defaults", {})

        yaml_files = sorted(f for f in path.glob("*.yaml") if f.name != "_defaults.yaml")
        all_rules: list[PerFileRule] = []
        for yaml_file in yaml_files:
            data = _load_yaml_file(yaml_file)
            all_rules.extend(_parse_per_file_rules(data, source_file=yaml_file.name))

        return Allowlist(
            per_file_rules=all_rules,
            fail_on_stale=defaults.get("fail_on_stale", True),
        )

    if not path.exists():
        return Allowlist()

    data = _load_yaml_file(path)
    defaults = data.get("defaults", {})
    return Allowlist(
        per_file_rules=_parse_per_file_rules(data),
        fail_on_stale=defaults.get("fail_on_stale", True),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_guard_symmetry.py -v -k "FileScanning or AllowlistLoading"`
Expected: ALL PASSED (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/cicd/enforce_guard_symmetry.py tests/unit/scripts/cicd/test_enforce_guard_symmetry.py
git commit -m "feat(ci): add cross-file scanning and YAML allowlist loading

Two-phase scan: discovers dataclasses and loaders across all .py files,
then pairs and reports unguarded combinations. Allowlist supports
directory-of-YAML and single-file modes."
```

---

### Task 5: CLI, Reporting, and Entry Point

**Files:**
- Modify: `tests/unit/scripts/cicd/test_enforce_guard_symmetry.py`
- Modify: `scripts/cicd/enforce_guard_symmetry.py`

- [ ] **Step 1: Write tests for reporting and CLI**

Add to the test file:

```python
from scripts.cicd.enforce_guard_symmetry import (
    format_finding,
    run_check,
)


class TestReporting:
    """Tests for finding formatting."""

    def test_format_finding_includes_key_info(self) -> None:
        f = Finding(
            rule_id="GS1",
            file_path="core/landscape/model_loaders.py",
            line=53,
            col=0,
            symbol_context=("RunLoader", "load"),
            fingerprint="abc123",
            code_snippet="class RunLoader:",
            message="Run has __post_init__ validation but RunLoader.load() has no AuditIntegrityError",
        )
        text = format_finding(f)
        assert "model_loaders.py:53:0" in text
        assert "GS1" in text
        assert "missing-read-guard" in text
        assert "RunLoader:load" in text
        assert "fp=abc123" in text


class TestCLI:
    """Tests for the check command."""

    def test_check_returns_0_when_no_findings(self, temp_dir: Path) -> None:
        contracts_dir = temp_dir / "contracts"
        contracts_dir.mkdir()
        (contracts_dir / "audit.py").write_text(dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Foo:
                name: str
        """))

        args = argparse.Namespace(
            root=temp_dir,
            allowlist=temp_dir / "no_such_allowlist",
            files=[],
        )
        assert run_check(args) == 0

    def test_check_returns_1_when_violations(self, temp_dir: Path) -> None:
        contracts_dir = temp_dir / "contracts"
        contracts_dir.mkdir()
        core_dir = temp_dir / "core"
        core_dir.mkdir()

        (contracts_dir / "audit.py").write_text(dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Widget:
                size: int

                def __post_init__(self) -> None:
                    if self.size < 0:
                        raise ValueError("negative size")
        """))

        (core_dir / "loaders.py").write_text(dedent("""
            class WidgetLoader:
                def load(self, row):
                    return Widget(size=row.size)
        """))

        args = argparse.Namespace(
            root=temp_dir,
            allowlist=temp_dir / "no_such_allowlist",
            files=[],
        )
        assert run_check(args) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_guard_symmetry.py -v -k "Reporting or CLI"`
Expected: FAIL — `ImportError: cannot import name 'format_finding'`

- [ ] **Step 3: Implement reporting and CLI**

Add to `scripts/cicd/enforce_guard_symmetry.py`:

```python
# =============================================================================
# Reporting
# =============================================================================


def format_finding(finding: Finding) -> str:
    rule = RULES.get(finding.rule_id, {})
    lines = [
        f"\n{finding.file_path}:{finding.line}:{finding.col}",
        f"  Rule: {finding.rule_id} - {rule.get('name', 'unknown')}",
        f"  Code: {finding.code_snippet}",
        f"  Context: {'.'.join(finding.symbol_context) if finding.symbol_context else '<module>'}",
        f"  Issue: {finding.message}",
        f"  Fix: {rule.get('remediation', 'Review and fix the pattern')}",
        f"  Allowlist key: {finding.canonical_key}",
    ]
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce write/read guard symmetry — detect dataclass→loader pairs missing AuditIntegrityError"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check for guard symmetry violations")
    check_parser.add_argument("--root", type=Path, required=True, help="Root directory to scan")
    check_parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Path to allowlist YAML file or directory",
    )
    check_parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Specific files to check (pre-commit mode). If empty, scans --root directory.",
    )

    args = parser.parse_args()

    if args.command == "check":
        return run_check(args)

    return 0


def run_check(args: argparse.Namespace) -> int:
    root = args.root.resolve()

    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1

    # Load allowlist
    allowlist_path = args.allowlist
    if allowlist_path is None:
        repo_root = Path(__file__).parent.parent.parent
        dir_path = repo_root / "config" / "cicd" / "enforce_guard_symmetry"
        allowlist_path = dir_path if dir_path.is_dir() else repo_root / "config" / "cicd" / "enforce_guard_symmetry.yaml"

    allowlist = load_allowlist(allowlist_path)

    # Scan
    files = [f.resolve() for f in args.files] if args.files else None
    all_findings = scan_files(root, files)

    # Filter allowlisted
    violations: list[Finding] = []
    for finding in all_findings:
        if allowlist.match(finding) is None:
            violations.append(finding)

    # Staleness checks (only in full-scan mode)
    if args.files:
        unused_rules: list[PerFileRule] = []
        expired_rules: list[PerFileRule] = []
        exceeded_rules: list[PerFileRule] = []
    else:
        unused_rules = allowlist.get_unused_rules() if allowlist.fail_on_stale else []
        expired_rules = allowlist.get_expired_rules()
        exceeded_rules = allowlist.get_exceeded_rules()

    has_errors = bool(violations or unused_rules or expired_rules or exceeded_rules)

    # Report
    if violations:
        print(f"\n{'=' * 60}")
        print(f"GUARD SYMMETRY VIOLATIONS: {len(violations)}")
        print("=" * 60)
        for v in violations:
            print(format_finding(v))

    if expired_rules:
        print(f"\n{'=' * 60}")
        print(f"EXPIRED PER-FILE RULES: {len(expired_rules)}")
        print("=" * 60)
        for r in expired_rules:
            print(f"\n  Pattern: {r.pattern}")
            print(f"  Rules: {r.rules}")
            print(f"  Expired: {r.expires}")

    if unused_rules:
        print(f"\n{'=' * 60}")
        print(f"UNUSED PER-FILE RULES: {len(unused_rules)}")
        print("(These rules didn't match any code — remove them)")
        print("=" * 60)
        for r in unused_rules:
            print(f"\n  Pattern: {r.pattern}")
            print(f"  Rules: {r.rules}")
            print(f"  Reason: {r.reason}")

    if exceeded_rules:
        print(f"\n{'=' * 60}")
        print(f"EXCEEDED PER-FILE RULES: {len(exceeded_rules)}")
        print("(These rules matched more findings than max_hits allows)")
        print("=" * 60)
        for r in exceeded_rules:
            print(f"\n  Pattern: {r.pattern}")
            print(f"  Rules: {r.rules}")
            print(f"  Matched: {r.matched_count} (max_hits: {r.max_hits})")

    if has_errors:
        print(f"\n{'=' * 60}")
        print("CHECK FAILED")
        print("=" * 60)
        if violations:
            print(f"\nTo allowlist a finding, add a per_file_rules entry to {allowlist_path}")
    else:
        print("\nNo guard symmetry violations detected. Check passed.")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_guard_symmetry.py -v`
Expected: ALL PASSED (~22 tests)

- [ ] **Step 5: Verify the scanner runs against the real codebase**

Run: `python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth`
Expected: Exit code 1, with several GS1 findings for unguarded loaders (RunLoader, EdgeLoader, RowLoader, TokenLoader, TokenParentLoader, CallLoader, RoutingEventLoader, BatchLoader, ArtifactLoader, OperationLoader, BatchMemberLoader).

Note the exact findings — these will be triaged in Task 7.

- [ ] **Step 6: Commit**

```bash
git add scripts/cicd/enforce_guard_symmetry.py tests/unit/scripts/cicd/test_enforce_guard_symmetry.py
git commit -m "feat(ci): complete guard symmetry scanner with CLI and reporting

Full enforce_freeze_guards-pattern scanner: AST discovery, pairing,
allowlist filtering, staleness checks, pre-commit mode support."
```

---

### Task 6: Configuration Files and CI Workflow

**Files:**
- Create: `config/cicd/enforce_guard_symmetry/_defaults.yaml`
- Create: `config/cicd/enforce_guard_symmetry/landscape.yaml`
- Create: `.github/workflows/enforce-guard-symmetry.yaml`

- [ ] **Step 1: Create allowlist defaults**

```yaml
# config/cicd/enforce_guard_symmetry/_defaults.yaml
version: 1
defaults:
  fail_on_stale: true
```

- [ ] **Step 2: Create initial per-loader allowlist**

Run the scanner first (`python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth`) to see the exact findings. Then create **one allowlist entry per loader** with `max_hits: 1`. This prevents new unguarded loaders from hiding under a broad pattern — each exemption is specific and capped.

```yaml
# config/cicd/enforce_guard_symmetry/landscape.yaml
#
# Guard symmetry allowlist for Landscape loaders.
#
# DESIGN: One entry per loader, each with max_hits: 1.
# This prevents new unguarded loaders from silently passing —
# a new loader MUST either add AuditIntegrityError guards or
# get its own explicit allowlist entry with a reason.
#
# Loaders with complex state validation (NodeStateLoader,
# TokenOutcomeLoader, NodeLoader) correctly use explicit
# AuditIntegrityError and are NOT allowlisted.
#
# These loaders construct dataclasses whose __post_init__
# validation (require_int, _validate_enum) fires during load(),
# providing read-side coverage. The dataclass constructor IS
# the guard — explicit AuditIntegrityError would be redundant.

per_file_rules:
  # --- Int-only validation (require_int fires on construction) ---
  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "RowLoader: Row.__post_init__ calls require_int(row_index) — fires on construction from DB"
    expires: null
    max_hits: 1

  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "TokenLoader: Token.__post_init__ calls require_int(step_in_pipeline) — fires on construction from DB"
    expires: null
    max_hits: 1

  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "TokenParentLoader: TokenParent.__post_init__ calls require_int(ordinal) — fires on construction from DB"
    expires: null
    max_hits: 1

  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "ArtifactLoader: Artifact.__post_init__ calls require_int(size_bytes) — fires on construction from DB"
    expires: null
    max_hits: 1

  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "BatchMemberLoader: BatchMember.__post_init__ calls require_int(ordinal) — fires on construction from DB"
    expires: null
    max_hits: 1

  # --- Enum + int validation (enum conversion in loader, __post_init__ validates) ---
  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "EdgeLoader: Edge.__post_init__ validates enum + int — construction-time guard"
    expires: null
    max_hits: 1

  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "BatchLoader: Batch.__post_init__ validates enums + int — construction-time guard"
    expires: null
    max_hits: 1

  # --- Review candidates: consider adding explicit AuditIntegrityError ---
  # These loaders do more complex work (enum conversion, literal validation,
  # XOR constraints) where explicit read-side guards would add value.
  # Allowlisted for initial scanner deployment; file follow-up issues.
  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "RunLoader: Run.__post_init__ validates enums + ints — REVIEW: consider explicit AIE for status validation"
    expires: null
    max_hits: 1

  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "CallLoader: Call.__post_init__ validates XOR constraint + enums — REVIEW: consider explicit AIE"
    expires: null
    max_hits: 1

  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "RoutingEventLoader: RoutingEvent.__post_init__ validates enum + int — REVIEW: consider explicit AIE"
    expires: null
    max_hits: 1

  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "OperationLoader: Operation.__post_init__ validates lifecycle invariants — REVIEW: consider explicit AIE"
    expires: null
    max_hits: 1

  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "TransformErrorLoader: TransformErrorRecord has no __post_init__ validation — allowlisted (no write-side guard to mirror)"
    expires: null
    max_hits: 1

  - pattern: "core/landscape/model_loaders.py"
    rules: [GS1]
    reason: "ValidationErrorLoader: ValidationErrorRecord has no __post_init__ validation — allowlisted (no write-side guard to mirror)"
    expires: null
    max_hits: 1
```

**Important:** The `max_hits: 1` per entry means any NEW unguarded loader added to `model_loaders.py` will fail CI immediately — it won't hide under an existing entry. After running the scanner, verify the exact finding count matches the number of entries above. Remove entries for loaders that don't produce findings (TransformErrorRecord/ValidationErrorRecord may not if they lack `__post_init__` — confirm by running the scanner).

- [ ] **Step 3: Run scanner with allowlist to verify it passes**

Run: `python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth --allowlist config/cicd/enforce_guard_symmetry`
Expected: Exit code 0 (all findings allowlisted, no stale rules)

If the scanner still fails, adjust `max_hits` or add additional allowlist entries.

- [ ] **Step 4: Create GitHub Actions workflow**

**Note:** This scanner is cross-file (dataclasses in one file, loaders in another). Pre-commit mode (file-specific scanning) is best-effort only — it can miss asymmetries when only one file of the pair changes. The CI full-scan workflow below is the authoritative enforcement gate.

```yaml
# .github/workflows/enforce-guard-symmetry.yaml
#
# Guard Symmetry Enforcement
#
# Detects dataclass→loader pairs where __post_init__ validation exists
# but the loader lacks AuditIntegrityError guards. Fails on:
#   - Unallowlisted guard symmetry violations
#   - Stale allowlist entries
#   - Expired allowlist entries
#
# NOTE: This scanner is cross-file by design. Pre-commit mode is
# best-effort — CI full-scan is the authoritative gate.

name: Enforce Guard Symmetry

on:
  push:
    branches: [main, master]
    paths:
      - 'src/**/*.py'
      - 'scripts/cicd/enforce_guard_symmetry.py'
      - 'config/cicd/enforce_guard_symmetry/**'
  pull_request:
    paths:
      - 'src/**/*.py'
      - 'scripts/cicd/enforce_guard_symmetry.py'
      - 'config/cicd/enforce_guard_symmetry/**'

jobs:
  check:
    name: Check for guard symmetry violations
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pyyaml

      - name: Run guard symmetry enforcement
        run: |
          python scripts/cicd/enforce_guard_symmetry.py check \
            --root src/elspeth \
            --allowlist config/cicd/enforce_guard_symmetry
```

- [ ] **Step 5: Commit**

```bash
git add config/cicd/enforce_guard_symmetry/ .github/workflows/enforce-guard-symmetry.yaml
git commit -m "ci: add guard symmetry workflow and initial allowlist

Allowlists loaders where __post_init__ validation serves as read-side guard.
max_hits cap prevents new unguarded pairs from passing silently."
```

---

### Task 7: Triage Findings and Finalize Allowlist

This task is manual — run the scanner, review each finding, and decide: fix (add AuditIntegrityError to the loader) or allowlist (document why __post_init__ is sufficient).

**Files:**
- Modify: `config/cicd/enforce_guard_symmetry/landscape.yaml` (update `max_hits`)

- [ ] **Step 1: Run scanner and capture output**

Run: `python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth 2>&1`

Review each GS1 finding. Classify as:
- **Fix needed**: Loader does complex operations (enum conversion, state discrimination) where the raw DB value should be validated with AuditIntegrityError before construction
- **Allowlist OK**: Loader constructs the dataclass directly, `__post_init__` catches bad values via `require_int` / `_validate_enum` / `raise`

Expected classification (from issue description):
- **Fix candidates**: CallLoader, OperationLoader, RunLoader, RoutingEventLoader (do enum/literal conversion that could mask bad DB values)
- **Allowlist candidates**: RowLoader, TokenLoader, TokenParentLoader, EdgeLoader, ArtifactLoader, BatchLoader, BatchMemberLoader (simple construction with `__post_init__` coverage)

- [ ] **Step 2: Update max_hits with actual allowlisted count**

After confirming which findings are allowlisted, update `landscape.yaml`:

```yaml
    max_hits: <exact count of allowlisted findings>
```

- [ ] **Step 3: Verify scanner passes with updated allowlist**

Run: `python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth --allowlist config/cicd/enforce_guard_symmetry`
Expected: Exit code 0

- [ ] **Step 4: Commit**

```bash
git add config/cicd/enforce_guard_symmetry/landscape.yaml
git commit -m "ci: finalize guard symmetry allowlist with max_hits cap

Allowlisted N findings where __post_init__ provides sufficient read-side
coverage. Remaining M findings tracked as follow-up work."
```

- [ ] **Step 5: Create follow-up issues for fix candidates**

For each loader that needs AuditIntegrityError guards (not allowlisted), create a Filigree issue:

```
filigree create "Add AuditIntegrityError guards to <LoaderName>" \
  --type=task --priority=2
```

Link these as children of the parent issue `elspeth-5f37dcce91` or document them as follow-up work.

- [ ] **Step 6: Close the scanner issue**

```
filigree close elspeth-5f37dcce91 --reason="Scanner implemented, CI workflow active, initial findings triaged"
```
