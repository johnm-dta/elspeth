#!/usr/bin/env python3
"""AST-based enforcement for contracts package.

Scans the codebase for:
1. dataclasses, TypedDicts, NamedTuples, and Enums used across module boundaries
2. dict[str, Any] type hints that should be typed contracts
3. Settings classes without Runtime counterparts (orphaned settings)

Also validates that all whitelist entries are still valid (not stale).

Usage:
    python scripts/check_contracts.py
    python scripts/check_contracts.py --no-fail-on-stale  # Skip stale check

Exit codes:
    0: All contracts properly centralized
    1: Violations found or stale whitelist entries
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml  # type: ignore[import-untyped]


@dataclass
class Violation:
    """A contract violation found during scanning."""

    file: str
    line: int
    type_name: str
    kind: str
    used_in: list[str]


@dataclass
class DictViolation:
    """A dict[str, Any] usage that should be a typed contract."""

    file: str
    line: int
    context: str  # function name or class.method
    param_name: str  # parameter name or "return"


@dataclass
class StaleEntry:
    """A whitelist entry that doesn't match any code."""

    entry: str
    category: str  # "type" or "dict_pattern"
    reason: str


@dataclass
class WhitelistEntry:
    """Tracked whitelist entry with match status."""

    value: str
    category: str
    matched: bool = field(default=False)


@dataclass
class SettingsViolation:
    """A Settings class without a Runtime counterpart."""

    class_name: str
    file: str
    line: int


@dataclass
class FieldCoverageViolation:
    """A Settings field not accessed in from_settings() method.

    Note: line is always 0 because tracking exact line numbers for Settings
    fields would require significant AST complexity. The settings_class +
    orphaned_field combination is sufficient for locating the issue - users
    can search for "class {settings_class}" and find the field definition.
    """

    settings_class: str
    runtime_class: str
    orphaned_field: str
    file: str
    line: int  # Always 0 - see docstring


@dataclass
class FieldMappingViolation:
    """A field mapping that doesn't match FIELD_MAPPINGS.

    This catches "misrouted" fields where code maps a settings field to
    the wrong runtime field. For example:
        base_delay=settings.max_delay_seconds  # Wrong! Should be initial_delay_seconds

    Note: line is always 0 because tracking exact line numbers for field
    mappings would require additional AST position tracking. The
    runtime_class + runtime_field combination is sufficient for locating
    the issue - users can search for the from_settings() method.
    """

    runtime_class: str
    runtime_field: str
    settings_field: str
    expected_settings_field: str
    file: str
    line: int  # Always 0 - see docstring


@dataclass
class HardcodeViolation:
    """A hardcoded literal in from_settings() not documented in INTERNAL_DEFAULTS.

    This catches undocumented internal defaults where code uses a literal
    value instead of settings.X but the literal is not documented in
    INTERNAL_DEFAULTS. For example:
        jitter=1.0  # OK if INTERNAL_DEFAULTS["retry"]["jitter"] = 1.0
        magic_number=42  # VIOLATION - not documented anywhere

    Note: line is always 0 because tracking exact line numbers for hardcodes
    would require additional AST position tracking. The runtime_class +
    runtime_field combination is sufficient for locating the issue.
    """

    runtime_class: str
    runtime_field: str
    literal_value: str  # String representation of the literal
    subsystem: str  # Expected subsystem key in INTERNAL_DEFAULTS
    file: str
    line: int  # Always 0 - see docstring


def load_whitelist(path: Path) -> tuple[dict[str, set[str]], list[WhitelistEntry]]:
    """Load whitelisted type definitions and dict patterns.

    Returns:
        Tuple of (whitelist dict for matching, list of entries for stale tracking)
    """
    if not path.exists():
        return {"types": set(), "dicts": set()}, []

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    entries: list[WhitelistEntry] = []

    type_entries = data.get("allowed_external_types", [])
    dict_entries = data.get("allowed_dict_patterns", [])

    for t in type_entries:
        entries.append(WhitelistEntry(value=t, category="type"))
    for d in dict_entries:
        entries.append(WhitelistEntry(value=d, category="dict_pattern"))

    return {
        "types": set(type_entries),
        "dicts": set(dict_entries),
    }, entries


def find_type_definitions(file_path: Path) -> list[tuple[str, int, str]]:
    """Find dataclass, TypedDict, NamedTuple, Enum definitions in a file.

    Returns: List of (type_name, line_number, kind)
    """
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that cannot be parsed (syntax errors or invalid encoding)
        return []

    definitions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check for @dataclass decorator
            for decorator in node.decorator_list:
                is_dataclass_name = isinstance(decorator, ast.Name) and decorator.id == "dataclass"
                is_dataclass_call = (
                    isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass"
                )
                if is_dataclass_name or is_dataclass_call:
                    definitions.append((node.name, node.lineno, "dataclass"))

            # Check for TypedDict, NamedTuple, Enum base classes
            for base in node.bases:
                if isinstance(base, ast.Name):
                    if base.id == "TypedDict":
                        definitions.append((node.name, node.lineno, "TypedDict"))
                    elif base.id == "NamedTuple":
                        definitions.append((node.name, node.lineno, "NamedTuple"))
                    elif base.id == "Enum":
                        definitions.append((node.name, node.lineno, "Enum"))
                    elif base.id in ("BaseModel", "PluginSchema"):
                        # Pydantic models in config are OK (trust boundary)
                        pass

    return definitions


def _is_dict_str_any(annotation: ast.expr | None) -> bool:
    """Check if annotation is dict[str, Any] or Dict[str, Any]."""
    if annotation is None:
        return False

    # dict[str, Any] - modern syntax
    if (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id in ("dict", "Dict")
        and isinstance(annotation.slice, ast.Tuple)
        and len(annotation.slice.elts) == 2
    ):
        key_type, value_type = annotation.slice.elts
        if isinstance(key_type, ast.Name) and key_type.id == "str" and isinstance(value_type, ast.Name) and value_type.id == "Any":
            return True
    return False


def _is_list_of_dict_str_any(annotation: ast.expr | None) -> bool:
    """Check if annotation is list[dict[str, Any]]."""
    if annotation is None:
        return False

    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name) and annotation.value.id in ("list", "List"):
        return _is_dict_str_any(annotation.slice)
    return False


def _is_optional_dict(annotation: ast.expr | None) -> bool:
    """Check if annotation is dict[str, Any] | None."""
    if annotation is None:
        return False

    # dict[str, Any] | None
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        left_is_dict = _is_dict_str_any(annotation.left)
        right_is_none = isinstance(annotation.right, ast.Constant) and annotation.right.value is None
        if left_is_dict and right_is_none:
            return True
        # None | dict[str, Any]
        left_is_none = isinstance(annotation.left, ast.Constant) and annotation.left.value is None
        right_is_dict = _is_dict_str_any(annotation.right)
        if left_is_none and right_is_dict:
            return True
    return False


def _is_union_with_dict(annotation: ast.expr | None) -> bool:
    """Check if annotation contains dict[str, Any] in a union."""
    if annotation is None:
        return False

    # Check direct dict
    if _is_dict_str_any(annotation):
        return True

    # Check union types (X | Y | Z)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _is_union_with_dict(annotation.left) or _is_union_with_dict(annotation.right)

    return False


def find_dict_patterns_in_file(file_path: Path) -> list[str]:
    """Find all dict[str, Any] patterns in a file.

    Returns list of qualified names like "path:Class.method:param"
    """
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that cannot be parsed (syntax errors or invalid encoding)
        return []

    patterns = []
    relative_path = str(file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_name = node.name
            class_name = None

            # Try to find enclosing class
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for child in ast.walk(parent):
                        if child is node:
                            class_name = parent.name
                            break

            context = f"{class_name}.{func_name}" if class_name else func_name

            # Check parameters
            for arg in node.args.args + node.args.kwonlyargs:
                param_name = arg.arg
                annotation = arg.annotation

                if _is_dict_str_any(annotation) or _is_optional_dict(annotation) or _is_union_with_dict(annotation):
                    patterns.append(f"{relative_path}:{context}:{param_name}")
                elif _is_list_of_dict_str_any(annotation):
                    patterns.append(f"{relative_path}:{context}:{param_name} (list)")

            # Check return type
            if node.returns:
                if _is_dict_str_any(node.returns) or _is_optional_dict(node.returns) or _is_union_with_dict(node.returns):
                    patterns.append(f"{relative_path}:{context}:return")
                elif _is_list_of_dict_str_any(node.returns):
                    patterns.append(f"{relative_path}:{context}:return (list)")

    return patterns


def find_dict_violations(file_path: Path, whitelist: set[str], matched_entries: dict[str, bool]) -> list[DictViolation]:
    """Find dict[str, Any] type hints that should be typed contracts."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that cannot be parsed (syntax errors or invalid encoding)
        return []

    violations = []
    relative_path = str(file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_name = node.name
            class_name = None

            # Try to find enclosing class
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for child in ast.walk(parent):
                        if child is node:
                            class_name = parent.name
                            break

            context = f"{class_name}.{func_name}" if class_name else func_name

            # Check parameters
            for arg in node.args.args + node.args.kwonlyargs:
                param_name = arg.arg
                annotation = arg.annotation

                if _is_dict_str_any(annotation) or _is_optional_dict(annotation) or _is_union_with_dict(annotation):
                    # Build qualified name for whitelist check
                    qualified = f"{relative_path}:{context}:{param_name}"
                    if qualified in whitelist:
                        matched_entries[qualified] = True
                    else:
                        violations.append(
                            DictViolation(
                                file=relative_path,
                                line=arg.lineno if hasattr(arg, "lineno") else node.lineno,
                                context=context,
                                param_name=param_name,
                            )
                        )
                elif _is_list_of_dict_str_any(annotation):
                    # List types have "(list)" suffix in whitelist
                    qualified = f"{relative_path}:{context}:{param_name} (list)"
                    if qualified in whitelist:
                        matched_entries[qualified] = True
                    else:
                        violations.append(
                            DictViolation(
                                file=relative_path,
                                line=arg.lineno if hasattr(arg, "lineno") else node.lineno,
                                context=context,
                                param_name=f"{param_name} (list)",
                            )
                        )

            # Check return type
            if node.returns:
                if _is_dict_str_any(node.returns) or _is_optional_dict(node.returns) or _is_union_with_dict(node.returns):
                    qualified = f"{relative_path}:{context}:return"
                    if qualified in whitelist:
                        matched_entries[qualified] = True
                    else:
                        violations.append(
                            DictViolation(
                                file=relative_path,
                                line=node.lineno,
                                context=context,
                                param_name="return",
                            )
                        )
                elif _is_list_of_dict_str_any(node.returns):
                    # List types have "(list)" suffix in whitelist
                    qualified = f"{relative_path}:{context}:return (list)"
                    if qualified in whitelist:
                        matched_entries[qualified] = True
                    else:
                        violations.append(
                            DictViolation(
                                file=relative_path,
                                line=node.lineno,
                                context=context,
                                param_name="return (list)",
                            )
                        )

    return violations


def get_top_level_module(file_path: Path, src_dir: Path) -> str:
    """Get the top-level module name for a file.

    For example:
        src/elspeth/tui/types.py -> tui
        src/elspeth/core/config.py -> core
    """
    relative = file_path.relative_to(src_dir)
    parts = relative.parts
    if len(parts) > 0:
        return parts[0]
    return ""


def is_cross_boundary_usage(defining_file: Path, using_file: Path, src_dir: Path) -> bool:
    """Check if usage crosses module boundaries.

    Cross-boundary means the using file is in a different top-level module
    than the defining file.
    """
    defining_module = get_top_level_module(defining_file, src_dir)
    using_module = get_top_level_module(using_file, src_dir)
    return defining_module != using_module


def find_cross_boundary_usages(src_dir: Path, type_name: str, defining_file: Path) -> list[Path]:
    """Find files that import a type from a DIFFERENT top-level module."""
    usages = []
    defining_module = defining_file.relative_to(src_dir).with_suffix("").as_posix().replace("/", ".")

    for py_file in src_dir.rglob("*.py"):
        if py_file == defining_file:
            continue

        # Only count as violation if crossing module boundary
        if not is_cross_boundary_usage(defining_file, py_file, src_dir):
            continue

        try:
            source = py_file.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and defining_module in node.module:
                for alias in node.names:
                    if alias.name == type_name:
                        usages.append(py_file)

    return usages


def validate_type_entry(entry: str, src_dir: Path) -> str | None:
    """Validate that an allowed_external_types entry exists.

    Entry format: "module/path:TypeName"

    Returns None if valid, or an error message if stale.
    """
    try:
        module_path, type_name = entry.rsplit(":", 1)
    except ValueError:
        return "Invalid format (expected 'path:TypeName')"

    # Convert module path to file path
    file_path = src_dir / f"{module_path}.py"

    if not file_path.exists():
        return f"File not found: {file_path}"

    # Check if type exists in file
    definitions = find_type_definitions(file_path)
    type_names = {name for name, _, _ in definitions}

    # Also check for regular class definitions (not just dataclass/TypedDict/etc)
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                type_names.add(node.name)
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that cannot be parsed; rely on definitions from find_type_definitions
        pass

    if type_name not in type_names:
        return f"Type '{type_name}' not found in {file_path}"

    return None


def validate_dict_pattern_entry(entry: str, src_dir: Path) -> str | None:
    """Validate that an allowed_dict_patterns entry exists.

    Entry format: "src/elspeth/path/file.py:Class.method:param"

    Returns None if valid, or an error message if stale.
    """
    try:
        parts = entry.split(":")
        if len(parts) != 3:
            return f"Invalid format (expected 'file:context:param', got {len(parts)} parts)"

        file_path_str, context, param = parts
    except ValueError:
        return "Invalid format (expected 'file:context:param')"

    file_path = Path(file_path_str)

    if not file_path.exists():
        return f"File not found: {file_path}"

    # Find all dict patterns in the file
    patterns = find_dict_patterns_in_file(file_path)

    # Check if this entry matches any pattern
    if entry in patterns:
        return None

    # Check for partial matches to give better error messages
    matching_file_patterns = [p for p in patterns if p.startswith(file_path_str)]
    if not matching_file_patterns:
        return f"No dict[str, Any] patterns found in {file_path}"

    # Check if context exists
    matching_context_patterns = [p for p in matching_file_patterns if f":{context}:" in p]
    if not matching_context_patterns:
        return f"Context '{context}' not found in {file_path}"

    # Parameter doesn't match
    available_params = [p.split(":")[-1] for p in matching_context_patterns]
    return f"Parameter '{param}' not found in {context}. Available: {available_params}"


def find_stale_entries(
    entries: list[WhitelistEntry],
    matched_dict_patterns: dict[str, bool],
    matched_type_patterns: set[str],
    src_dir: Path,
) -> list[StaleEntry]:
    """Find whitelist entries that don't match any code."""
    stale = []

    for entry in entries:
        if entry.category == "type":
            # Check if this type entry is valid
            if entry.value in matched_type_patterns:
                continue
            error = validate_type_entry(entry.value, src_dir)
            if error:
                stale.append(StaleEntry(entry=entry.value, category="type", reason=error))

        elif entry.category == "dict_pattern":
            # Check if this pattern was matched during scanning
            if matched_dict_patterns.get(entry.value, False):
                continue
            # Validate the entry
            error = validate_dict_pattern_entry(entry.value, src_dir)
            if error:
                stale.append(StaleEntry(entry=entry.value, category="dict_pattern", reason=error))

    return stale


class SettingsAccessVisitor(ast.NodeVisitor):
    """Extract all `settings.X` attribute accesses from AST.

    Used to find which Settings fields are accessed in from_settings() methods.
    Looks for patterns like:
        - settings.field_name
        - settings.field_name.nested (captures just field_name)
    """

    def __init__(self, param_name: str = "settings") -> None:
        self.param_name = param_name
        self.accessed_fields: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Capture attribute access on the settings parameter."""
        # Check if this is `settings.X` - direct attribute access on the parameter
        if isinstance(node.value, ast.Name) and node.value.id == self.param_name:
            self.accessed_fields.add(node.attr)
        # Continue visiting children
        self.generic_visit(node)


class FieldMappingVisitor(ast.NodeVisitor):
    """Extract runtime_field=settings.settings_field mappings from AST.

    Used to validate that field mappings in from_settings() match FIELD_MAPPINGS.
    Looks for keyword arguments in constructor calls like:
        cls(
            base_delay=settings.initial_delay_seconds,
            max_delay=settings.max_delay_seconds,
        )

    Captures tuples of (runtime_field, settings_field).
    """

    def __init__(self, param_name: str = "settings") -> None:
        self.param_name = param_name
        self.field_mappings: list[tuple[str, str]] = []  # (runtime_field, settings_field)

    def visit_Call(self, node: ast.Call) -> None:
        """Capture keyword arguments that map settings fields to runtime fields."""
        for keyword in node.keywords:
            if keyword.arg is None:
                # **kwargs - skip
                continue
            runtime_field = keyword.arg
            # Check if value is settings.X
            if (
                isinstance(keyword.value, ast.Attribute)
                and isinstance(keyword.value.value, ast.Name)
                and keyword.value.value.id == self.param_name
            ):
                settings_field = keyword.value.attr
                self.field_mappings.append((runtime_field, settings_field))
        # Continue visiting children (nested calls)
        self.generic_visit(node)


class HardcodeLiteralVisitor(ast.NodeVisitor):
    """Extract runtime_field=<literal> assignments from AST.

    Used to find hardcoded literals in from_settings() methods.
    Looks for keyword arguments in constructor calls like:
        cls(
            jitter=1.0,  # Hardcoded literal
            max_attempts=settings.max_attempts,  # Not a literal (skipped)
        )

    Captures tuples of (runtime_field, literal_value).
    Ignores values that are:
    - settings.X accesses (handled by FieldMappingVisitor)
    - Function calls like float(INTERNAL_DEFAULTS["retry"]["jitter"])
    - Subscripts like INTERNAL_DEFAULTS["retry"]["jitter"]
    - Variable references
    """

    def __init__(self, param_name: str = "settings") -> None:
        self.param_name = param_name
        self.hardcoded_literals: list[tuple[str, object]] = []  # (runtime_field, literal_value)

    def _is_plain_literal(self, node: ast.expr) -> tuple[bool, object]:
        """Check if node is a plain literal (not wrapped in function call).

        Returns (is_literal, value).
        """
        # Plain constants: 1.0, 42, "string", True
        if isinstance(node, ast.Constant):
            return True, node.value
        # Negative numbers: -1.0, -42
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            operand_value = node.operand.value
            # Only negate numeric types
            if isinstance(operand_value, int | float):
                return True, -operand_value
        return False, None

    def visit_Call(self, node: ast.Call) -> None:
        """Capture keyword arguments that use plain literals."""
        for keyword in node.keywords:
            if keyword.arg is None:
                # **kwargs - skip
                continue
            runtime_field = keyword.arg

            # Skip settings.X accesses
            if (
                isinstance(keyword.value, ast.Attribute)
                and isinstance(keyword.value.value, ast.Name)
                and keyword.value.value.id == self.param_name
            ):
                continue

            # Check if it's a plain literal (not wrapped in float(), int(), etc.)
            is_literal, value = self._is_plain_literal(keyword.value)
            if is_literal:
                self.hardcoded_literals.append((runtime_field, value))

        # Continue visiting children (nested calls)
        self.generic_visit(node)


def extract_from_settings_accesses(runtime_path: Path) -> dict[str, set[str]]:
    """Extract all settings.X accesses from from_settings() methods in a file.

    Parses the runtime.py file and finds all Runtime*Config classes with
    from_settings() methods. For each, extracts which settings fields are accessed.

    Args:
        runtime_path: Path to contracts/config/runtime.py

    Returns:
        Dict mapping RuntimeClassName -> set of accessed settings fields
    """
    try:
        source = runtime_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return {}

    result: dict[str, set[str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Runtime") and node.name.endswith("Config"):
            # Find from_settings() method in this class
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "from_settings":
                    # Get the parameter name (first param after cls)
                    # from_settings(cls, settings: "RetrySettings") -> settings
                    param_name = "settings"  # default
                    if len(item.args.args) > 1:
                        param_name = item.args.args[1].arg

                    # Visit the method body to find settings.X accesses
                    visitor = SettingsAccessVisitor(param_name)
                    for stmt in item.body:
                        visitor.visit(stmt)

                    result[node.name] = visitor.accessed_fields
                    break

    return result


def extract_from_settings_field_mappings(runtime_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Extract runtime_field=settings.settings_field mappings from from_settings() methods.

    Parses the runtime.py file and finds all Runtime*Config classes with
    from_settings() methods. For each, extracts the field mappings.

    Args:
        runtime_path: Path to contracts/config/runtime.py

    Returns:
        Dict mapping RuntimeClassName -> list of (runtime_field, settings_field) tuples
    """
    try:
        source = runtime_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return {}

    result: dict[str, list[tuple[str, str]]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Runtime") and node.name.endswith("Config"):
            # Find from_settings() method in this class
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "from_settings":
                    # Get the parameter name (first param after cls)
                    param_name = "settings"  # default
                    if len(item.args.args) > 1:
                        param_name = item.args.args[1].arg

                    # Visit the method body to find runtime_field=settings.X mappings
                    visitor = FieldMappingVisitor(param_name)
                    for stmt in item.body:
                        visitor.visit(stmt)

                    result[node.name] = visitor.field_mappings
                    break

    return result


def check_field_name_mappings(runtime_path: Path) -> list[FieldMappingViolation]:
    """Check that field mappings in from_settings() match FIELD_MAPPINGS.

    For each Runtime*Config class with a from_settings() method:
    1. Extract all runtime_field=settings.settings_field assignments
    2. For each renamed field (in FIELD_MAPPINGS), verify the mapping is correct
    3. Report violations where settings field is mapped to wrong runtime field

    Example violation (misrouted field):
        If FIELD_MAPPINGS says initial_delay_seconds -> base_delay but code has:
            base_delay=settings.max_delay_seconds
        This is a misroute - max_delay_seconds should map to max_delay, not base_delay.

    Args:
        runtime_path: Path to contracts/config/runtime.py (Runtime classes)

    Returns:
        List of FieldMappingViolation for misrouted fields
    """
    from elspeth.contracts.config.alignment import FIELD_MAPPINGS, SETTINGS_TO_RUNTIME

    # Get all runtime_field=settings.X mappings from from_settings() methods
    runtime_mappings = extract_from_settings_field_mappings(runtime_path)

    violations: list[FieldMappingViolation] = []

    # For each Settings -> Runtime mapping that has field renames
    for settings_class, runtime_class in SETTINGS_TO_RUNTIME.items():
        if settings_class not in FIELD_MAPPINGS:
            # No renamed fields for this class - skip
            continue

        if runtime_class not in runtime_mappings:
            # No from_settings() method found - skip (different check handles this)
            continue

        field_renames = FIELD_MAPPINGS[settings_class]
        actual_mappings = runtime_mappings[runtime_class]

        # Build reverse lookup: for each runtime_field that's a rename target,
        # what settings_field SHOULD map to it?
        # field_renames: {settings_field: runtime_field}
        # We need: {runtime_field: expected_settings_field}
        expected_for_runtime: dict[str, str] = {runtime_field: settings_field for settings_field, runtime_field in field_renames.items()}

        # Check each actual mapping
        for runtime_field, actual_settings_field in actual_mappings:
            # Is this runtime_field one that requires a specific settings_field?
            if runtime_field in expected_for_runtime:
                expected_settings_field = expected_for_runtime[runtime_field]
                if actual_settings_field != expected_settings_field:
                    violations.append(
                        FieldMappingViolation(
                            runtime_class=runtime_class,
                            runtime_field=runtime_field,
                            settings_field=actual_settings_field,
                            expected_settings_field=expected_settings_field,
                            file=str(runtime_path),
                            line=0,  # Line number would require more complex tracking
                        )
                    )

    return violations


def extract_from_settings_hardcodes(runtime_path: Path) -> dict[str, list[tuple[str, object]]]:
    """Extract runtime_field=<literal> assignments from from_settings() methods.

    Parses the runtime.py file and finds all Runtime*Config classes with
    from_settings() methods. For each, extracts hardcoded literal values.

    Args:
        runtime_path: Path to contracts/config/runtime.py

    Returns:
        Dict mapping RuntimeClassName -> list of (runtime_field, literal_value) tuples
    """
    try:
        source = runtime_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return {}

    result: dict[str, list[tuple[str, object]]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Runtime") and node.name.endswith("Config"):
            # Find from_settings() method in this class
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "from_settings":
                    # Get the parameter name (first param after cls)
                    param_name = "settings"  # default
                    if len(item.args.args) > 1:
                        param_name = item.args.args[1].arg

                    # Visit the method body to find hardcoded literals
                    visitor = HardcodeLiteralVisitor(param_name)
                    for stmt in item.body:
                        visitor.visit(stmt)

                    result[node.name] = visitor.hardcoded_literals
                    break

    return result


def check_hardcode_documentation(runtime_path: Path) -> list[HardcodeViolation]:
    """Check that hardcoded literals in from_settings() are documented in INTERNAL_DEFAULTS.

    For each Runtime*Config class with a from_settings() method:
    1. Extract all runtime_field=<literal> assignments (plain literals only)
    2. Look up the expected subsystem in RUNTIME_TO_SUBSYSTEM
    3. Check if the literal is documented in INTERNAL_DEFAULTS[subsystem][field]
    4. Report violations for undocumented hardcodes

    Example violation (undocumented hardcode):
        def from_settings(cls, settings):
            return cls(
                jitter=1.0,  # OK if INTERNAL_DEFAULTS["retry"]["jitter"] = 1.0
                magic_number=42,  # VIOLATION - not documented anywhere
            )

    Note: This check only catches PLAIN literals (1.0, 42, "string").
    Wrapped literals like float(INTERNAL_DEFAULTS["retry"]["jitter"]) are not checked
    because they explicitly reference INTERNAL_DEFAULTS (self-documenting).

    Args:
        runtime_path: Path to contracts/config/runtime.py (Runtime classes)

    Returns:
        List of HardcodeViolation for undocumented hardcodes
    """
    from elspeth.contracts.config.alignment import RUNTIME_TO_SUBSYSTEM
    from elspeth.contracts.config.defaults import INTERNAL_DEFAULTS

    # Get all hardcoded literals from from_settings() methods
    runtime_hardcodes = extract_from_settings_hardcodes(runtime_path)

    violations: list[HardcodeViolation] = []

    for runtime_class, hardcodes in runtime_hardcodes.items():
        # Get the subsystem for this runtime class
        subsystem = RUNTIME_TO_SUBSYSTEM.get(runtime_class)
        if subsystem is None:
            # No subsystem mapping - all hardcodes in this class are violations
            for runtime_field, literal_value in hardcodes:
                violations.append(
                    HardcodeViolation(
                        runtime_class=runtime_class,
                        runtime_field=runtime_field,
                        literal_value=repr(literal_value),
                        subsystem="(no subsystem mapping)",
                        file=str(runtime_path),
                        line=0,
                    )
                )
            continue

        # Get the documented defaults for this subsystem
        subsystem_defaults = INTERNAL_DEFAULTS.get(subsystem, {})

        # Check each hardcoded literal
        for runtime_field, literal_value in hardcodes:
            if runtime_field not in subsystem_defaults:
                # Field not documented in INTERNAL_DEFAULTS
                violations.append(
                    HardcodeViolation(
                        runtime_class=runtime_class,
                        runtime_field=runtime_field,
                        literal_value=repr(literal_value),
                        subsystem=subsystem,
                        file=str(runtime_path),
                        line=0,
                    )
                )
            elif subsystem_defaults[runtime_field] != literal_value:
                # Field documented but value doesn't match (more serious!)
                # This means the code has a different value than documented
                violations.append(
                    HardcodeViolation(
                        runtime_class=runtime_class,
                        runtime_field=runtime_field,
                        literal_value=f"{literal_value!r} (documented: {subsystem_defaults[runtime_field]!r})",
                        subsystem=subsystem,
                        file=str(runtime_path),
                        line=0,
                    )
                )

    return violations


def get_settings_class_fields(config_path: Path, class_name: str) -> set[str]:
    """Get all field names from a Settings class using AST.

    Parses the config.py file to find the Settings class and extracts
    field definitions. Handles both Pydantic Field() and simple annotations.

    Args:
        config_path: Path to core/config.py
        class_name: Name of the Settings class (e.g., "RetrySettings")

    Returns:
        Set of field names defined in the class
    """
    try:
        source = config_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields: set[str] = set()
            for item in node.body:
                # Look for annotated assignments: field: Type = ...
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.add(item.target.id)
            return fields

    return set()


def check_from_settings_coverage(
    config_path: Path,
    runtime_path: Path,
) -> list[FieldCoverageViolation]:
    """Check that from_settings() methods access all Settings fields.

    For each Runtime*Config class with a from_settings() method:
    1. Parse the method body to find all settings.X accesses
    2. Get the corresponding Settings class fields
    3. Report any Settings field NOT accessed (potential orphan)

    Why no exemption mechanism for Settings fields:
        If a Settings field exists, it SHOULD be used. Orphaned Settings fields
        are always bugs (like the P2-2026-01-21 exponential_base bug), never
        intentional. The INTERNAL exemption in FIELD_MAPPINGS is for *Runtime*
        fields that don't come from Settings (like `jitter`), not for Settings
        fields to skip. If a field shouldn't be mapped to Runtime, it shouldn't
        be in the Settings class at all.

    Args:
        config_path: Path to core/config.py (Settings classes)
        runtime_path: Path to contracts/config/runtime.py (Runtime classes)

    Returns:
        List of FieldCoverageViolation for orphaned fields
    """
    from elspeth.contracts.config.alignment import SETTINGS_TO_RUNTIME

    # Get all settings.X accesses from from_settings() methods
    runtime_accesses = extract_from_settings_accesses(runtime_path)

    violations: list[FieldCoverageViolation] = []

    # For each Settings -> Runtime mapping, check coverage
    for settings_class, runtime_class in SETTINGS_TO_RUNTIME.items():
        if runtime_class not in runtime_accesses:
            # No from_settings() method found - skip (different check handles this)
            continue

        accessed_fields = runtime_accesses[runtime_class]
        settings_fields = get_settings_class_fields(config_path, settings_class)

        # Find orphaned fields (in Settings but not accessed in from_settings)
        orphaned = settings_fields - accessed_fields

        for field_name in sorted(orphaned):
            violations.append(
                FieldCoverageViolation(
                    settings_class=settings_class,
                    runtime_class=runtime_class,
                    orphaned_field=field_name,
                    file=str(runtime_path),
                    line=0,  # Line number would require more complex tracking
                )
            )

    return violations


def find_settings_classes(config_path: Path) -> list[tuple[str, int]]:
    """Find all Settings classes in core/config.py.

    A Settings class is identified by its name ending in 'Settings'.
    These are Pydantic BaseModel classes that define configuration schemas.

    Args:
        config_path: Path to core/config.py

    Returns:
        List of (class_name, line_number) tuples
    """
    try:
        source = config_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    settings_classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.endswith("Settings"):
            settings_classes.append((node.name, node.lineno))

    return settings_classes


def check_settings_alignment(config_path: Path) -> list[SettingsViolation]:
    """Check that all Settings classes have Runtime counterparts or are exempt.

    Uses SETTINGS_TO_RUNTIME and EXEMPT_SETTINGS from contracts/config/alignment.py
    to determine which Settings classes need Runtime counterparts.

    Args:
        config_path: Path to core/config.py

    Returns:
        List of SettingsViolation for orphaned Settings classes
    """
    # Import alignment mappings
    from elspeth.contracts.config.alignment import EXEMPT_SETTINGS, SETTINGS_TO_RUNTIME

    settings_classes = find_settings_classes(config_path)
    violations = []

    for class_name, line_no in settings_classes:
        # Check if exempt (doesn't need Runtime counterpart)
        if class_name in EXEMPT_SETTINGS:
            continue
        # Check if has Runtime counterpart
        if class_name in SETTINGS_TO_RUNTIME:
            continue
        # Orphaned - no Runtime counterpart and not exempt
        violations.append(
            SettingsViolation(
                class_name=class_name,
                file=str(config_path),
                line=line_no,
            )
        )

    return violations


def main() -> int:
    """Run the contracts enforcement check."""
    parser = argparse.ArgumentParser(description="Check that cross-boundary types are in contracts/ and whitelist entries are valid")
    parser.add_argument(
        "--no-fail-on-stale",
        action="store_true",
        help="Don't fail on stale whitelist entries (just warn)",
    )
    args = parser.parse_args()

    src_dir = Path("src/elspeth")
    contracts_dir = src_dir / "contracts"
    whitelist_path = Path("config/cicd/contracts-whitelist.yaml")

    whitelist, all_entries = load_whitelist(whitelist_path)
    violations: list[Violation] = []
    dict_violations: list[DictViolation] = []
    matched_dict_patterns: dict[str, bool] = dict.fromkeys(whitelist["dicts"], False)
    matched_type_patterns: set[str] = set()

    # Scan all Python files outside contracts/
    for py_file in src_dir.rglob("*.py"):
        if contracts_dir in py_file.parents or py_file.parent == contracts_dir:
            continue  # Skip contracts/ itself

        # Check for type definitions
        definitions = find_type_definitions(py_file)
        for type_name, line_no, kind in definitions:
            qualified_name = f"{py_file.relative_to(src_dir).with_suffix('')}:{type_name}"

            if qualified_name in whitelist["types"]:
                matched_type_patterns.add(qualified_name)
                continue

            # Check if used across module boundaries
            usages = find_cross_boundary_usages(src_dir, type_name, py_file)
            if usages:
                violations.append(
                    Violation(
                        file=str(py_file),
                        line=line_no,
                        type_name=type_name,
                        kind=kind,
                        used_in=[str(u) for u in usages[:3]],  # First 3
                    )
                )

        # Check for dict[str, Any] patterns
        dict_violations.extend(find_dict_violations(py_file, whitelist["dicts"], matched_dict_patterns))

    # Find stale whitelist entries
    stale_entries = find_stale_entries(all_entries, matched_dict_patterns, matched_type_patterns, src_dir)

    # Check Settings → Runtime alignment
    config_path = src_dir / "core" / "config.py"
    runtime_path = contracts_dir / "config" / "runtime.py"
    settings_violations = check_settings_alignment(config_path)

    # Check from_settings() field coverage
    coverage_violations = check_from_settings_coverage(config_path, runtime_path)

    # Check from_settings() field name mappings match FIELD_MAPPINGS
    mapping_violations = check_field_name_mappings(runtime_path)

    # Check hardcoded literals in from_settings() are documented in INTERNAL_DEFAULTS
    hardcode_violations = check_hardcode_documentation(runtime_path)

    has_violations = False
    has_stale = False

    if violations:
        has_violations = True
        print("❌ Type definition violations found:\n")
        for v in violations:
            print(f"  {v.file}:{v.line}: {v.kind} '{v.type_name}'")
            print(f"    Used in: {', '.join(v.used_in)}")
            fix_msg = "    Fix: Move to src/elspeth/contracts/ or add to config/cicd/contracts-whitelist.yaml\n"
            print(fix_msg)

    if dict_violations:
        has_violations = True
        print("❌ dict[str, Any] violations found:\n")
        for dv in dict_violations:
            print(f"  {dv.file}:{dv.line}: {dv.context} - {dv.param_name}")
            print("    Fix: Use TypedDict/dataclass or add to allowed_dict_patterns\n")

    if settings_violations:
        has_violations = True
        print("❌ Orphaned Settings classes found:\n")
        print("  (Settings classes without Runtime counterparts)\n")
        for sv in settings_violations:
            print(f"  {sv.file}:{sv.line}: {sv.class_name}")
            print("    Fix: Add to SETTINGS_TO_RUNTIME mapping in contracts/config/alignment.py")
            print("    Or add to EXEMPT_SETTINGS if no Runtime counterpart is needed\n")

    if coverage_violations:
        has_violations = True
        print("❌ Settings field coverage violations found:\n")
        print("  (Settings fields not accessed in from_settings() methods)\n")
        for cv in coverage_violations:
            print(f"  {cv.settings_class}.{cv.orphaned_field} not accessed in {cv.runtime_class}.from_settings()")
            print("    Fix: Access the field in from_settings() and map it to a Runtime field")
            print("    Or document why the field is unused\n")

    if mapping_violations:
        has_violations = True
        print("❌ Field mapping violations found:\n")
        print("  (Settings fields mapped to wrong Runtime fields - misrouted)\n")
        for mv in mapping_violations:
            print(f"  {mv.runtime_class}: {mv.runtime_field}=settings.{mv.settings_field}")
            print(f"    Expected: {mv.runtime_field}=settings.{mv.expected_settings_field}")
            print(f"    Fix: Update from_settings() to use settings.{mv.expected_settings_field}")
            print("    Or update FIELD_MAPPINGS in contracts/config/alignment.py\n")

    if hardcode_violations:
        has_violations = True
        print("❌ Undocumented hardcoded values in from_settings() found:\n")
        print("  (Literal values in from_settings() must be documented in INTERNAL_DEFAULTS)\n")
        for hv in hardcode_violations:
            print(f"  {hv.runtime_class}.{hv.runtime_field} = {hv.literal_value}")
            print(f"    Subsystem: {hv.subsystem}")
            print(f"    Fix: Add to INTERNAL_DEFAULTS['{hv.subsystem}']['{hv.runtime_field}']")
            print("    in contracts/config/defaults.py\n")

    if stale_entries:
        has_stale = True
        print("❌ Stale whitelist entries found:\n")
        print("  (These entries don't match any code - remove them from the whitelist)\n")
        for se in stale_entries:
            print(f"  [{se.category}] {se.entry}")
            print(f"    Reason: {se.reason}\n")

    if has_violations:
        return 1

    if has_stale:
        if args.no_fail_on_stale:
            print("⚠️  Stale entries found but --no-fail-on-stale was specified")
        else:
            print("❌ Stale whitelist entries cause check failure")
            print("   Use --no-fail-on-stale to warn instead of fail")
            return 1

    print("✅ All cross-boundary types are properly centralized in contracts/")
    if not stale_entries:
        print("✅ All whitelist entries are valid")
    print("✅ All Settings classes have Runtime counterparts or are exempt")
    print("✅ All Settings fields are accessed in from_settings() methods")
    print("✅ All field name mappings match FIELD_MAPPINGS")
    print("✅ All hardcoded values are documented in INTERNAL_DEFAULTS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
