"""Shared hash computation logic for plugin source file hashing.

Provides three operations:
1. compute_source_file_hash — SHA-256 of file content with cross-platform and self-referential normalization
2. extract_plugin_attributes — AST extraction of plugin class attributes
3. fix_source_file_hash — in-place line rewrite to update a stale hash

The hash computation normalizes any existing ``source_file_hash = "sha256:..."``
line to a fixed placeholder before hashing, preventing the self-referential
problem where the hash value changes the hash.

Usage:
    from scripts.cicd.plugin_hash import (
        compute_source_file_hash,
        extract_plugin_attributes,
        fix_source_file_hash,
    )

    hash_val = compute_source_file_hash(Path("my_plugin.py"))
    attrs_list = extract_plugin_attributes(Path("my_plugin.py"))
    fix_source_file_hash(Path("my_plugin.py"), "MyPlugin", hash_val)
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Regex for normalizing source_file_hash lines in raw bytes.
# Matches both plain assignment and annotated assignment forms,
# with any content after "sha256:" (not just hex — handles placeholders
# like "sha256:stale_stale_stale" and "sha256:<computed>"):
#   source_file_hash = "sha256:abcdef0123456789"
#   source_file_hash: str = "sha256:abcdef0123456789"
#   source_file_hash: str | None = "sha256:abcdef0123456789"
#   source_file_hash = "sha256:stale_stale_stale"
_HASH_LINE_PATTERN = re.compile(rb'(\s*source_file_hash\s*(?::[^=]+=\s*|=\s*))"sha256:[^"]+"')

# The normalized placeholder value used during hashing.
_NORMALIZED_HASH_VALUE = b'"sha256:0000000000000000"'


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class PluginAttributes:
    """Extracted plugin class attributes from AST analysis.

    Attributes:
        class_name: The Python class name (e.g., "CSVSink").
        plugin_version: The plugin_version string, or None if absent.
        source_file_hash: The source_file_hash string, or None if absent/None.
        hash_line_number: 1-based line number of the source_file_hash assignment,
            or None if no such assignment exists.
    """

    class_name: str
    plugin_version: str | None
    source_file_hash: str | None
    hash_line_number: int | None


# =============================================================================
# Hash Computation
# =============================================================================


def compute_source_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content with cross-platform normalization.

    Reads raw bytes, normalizes line endings (CRLF/CR → LF) and strips
    UTF-8 BOM, then normalizes any ``source_file_hash = "sha256:..."``
    line to a fixed placeholder.  This ensures the same committed file
    produces the same hash regardless of checkout line-ending settings
    (``core.autocrlf``, ``.gitattributes``).

    Args:
        file_path: Path to the Python source file.

    Returns:
        Hash string in the format ``sha256:<16-hex-chars>``.
    """
    raw = file_path.read_bytes()
    # Normalize line endings: CRLF → LF, then lone CR → LF.
    # Strip UTF-8 BOM if present.
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if normalized.startswith(b"\xef\xbb\xbf"):
        normalized = normalized[3:]
    normalized = _HASH_LINE_PATTERN.sub(lambda m: m.group(1) + _NORMALIZED_HASH_VALUE, normalized)
    digest = hashlib.sha256(normalized).hexdigest()[:16]
    return f"sha256:{digest}"


# =============================================================================
# AST Extraction
# =============================================================================


def _get_class_attribute_value(node: ast.ClassDef, attr_name: str) -> tuple[object | None, int | None]:
    """Extract a class-level attribute value and its line number from a ClassDef.

    Handles both ``ast.Assign`` (``attr = value``) and ``ast.AnnAssign``
    (``attr: type = value``) forms. Only considers simple Name targets.

    Returns:
        (value, line_number) where value is the Python literal or None if the
        AST constant is None, and line_number is 1-based. Returns (None, None)
        if the attribute is not found.
    """
    _SENTINEL = object()

    for item in node.body:
        # Plain assignment: attr = value
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    if isinstance(item.value, ast.Constant):
                        return (item.value.value, item.lineno)
                    return (_SENTINEL, item.lineno)

        # Annotated assignment: attr: type = value
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == attr_name:
            if item.value is not None and isinstance(item.value, ast.Constant):
                return (item.value.value, item.lineno)
            if item.value is None:
                # annotation-only (no default) — not a value assignment
                return (None, None)
            return (_SENTINEL, item.lineno)

    return (None, None)


def _has_name_class_attribute(node: ast.ClassDef) -> bool:
    """Check if a ClassDef has a ``name = "..."`` class-level attribute.

    This is the pluggy plugin identifier. Only class-level string assignments
    count — method definitions named ``name`` do not.
    """
    value, _ = _get_class_attribute_value(node, "name")
    return isinstance(value, str)


def extract_plugin_attributes(file_path: Path) -> list[PluginAttributes]:
    """Extract plugin class attributes from a Python source file using AST.

    Finds all classes that have a ``name = "..."`` class attribute (the pluggy
    plugin identifier) and extracts ``plugin_version`` and ``source_file_hash``
    from their class bodies.

    Args:
        file_path: Path to the Python source file.

    Returns:
        List of PluginAttributes, one per plugin class found. Empty list if
        the file contains no plugin classes.
    """
    source = file_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(file_path))

    results: list[PluginAttributes] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _has_name_class_attribute(node):
            continue

        version_val, _ = _get_class_attribute_value(node, "plugin_version")
        hash_val, hash_line = _get_class_attribute_value(node, "source_file_hash")

        results.append(
            PluginAttributes(
                class_name=node.name,
                plugin_version=version_val if isinstance(version_val, str) else None,
                source_file_hash=hash_val if isinstance(hash_val, str) else None,
                hash_line_number=hash_line,
            )
        )

    return results


# =============================================================================
# Fix Rewrite
# =============================================================================


def fix_source_file_hash(file_path: Path, class_name: str, correct_hash: str) -> None:
    """Rewrite the source_file_hash line for a specific class in-place.

    Uses AST to find the exact line number of ``source_file_hash = "..."``
    in the target class, then replaces that single line preserving indentation.

    Args:
        file_path: Path to the Python source file.
        class_name: Name of the plugin class to update.
        correct_hash: The correct hash value (e.g., ``"sha256:abcdef0123456789"``).

    Raises:
        ValueError: If the class or its source_file_hash assignment cannot be found.
    """
    source = file_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(file_path))

    target_line: int | None = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != class_name:
            continue
        _, hash_line = _get_class_attribute_value(node, "source_file_hash")
        if hash_line is not None:
            target_line = hash_line
            break

    if target_line is None:
        raise ValueError(f"Cannot find source_file_hash assignment in class {class_name!r} in {file_path}")

    lines = source.splitlines(keepends=True)
    # target_line is 1-based
    old_line = lines[target_line - 1]

    # Preserve leading whitespace
    indent = old_line[: len(old_line) - len(old_line.lstrip())]

    # Detect whether the original uses annotation form
    # Match: source_file_hash: str = "...", source_file_hash: str | None = "...",
    # or source_file_hash = "..."
    ann_match = re.match(r"(\s*source_file_hash\s*:[^=]+=\s*)", old_line)
    plain_match = re.match(r"(\s*source_file_hash\s*=\s*)", old_line)

    if ann_match:
        prefix = ann_match.group(1)
    elif plain_match:
        prefix = plain_match.group(1)
    else:
        # Fallback: reconstruct from indent
        prefix = f"{indent}source_file_hash = "

    # Determine line ending from original
    stripped = old_line.rstrip("\r\n")
    ending = old_line[len(stripped) :]

    new_line = f'{prefix}"{correct_hash}"{ending}'
    lines[target_line - 1] = new_line

    file_path.write_text("".join(lines), encoding="utf-8")
