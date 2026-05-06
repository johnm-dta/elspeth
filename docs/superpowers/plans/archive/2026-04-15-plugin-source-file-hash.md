# Plugin Source File Hash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mechanical change detection to all 28 ELSPETH plugins via `source_file_hash` (SHA-256 of entry-point file content), with CI enforcement and landscape audit trail integration.

**Architecture:** Two-channel versioning — `plugin_version` (human semver) + `source_file_hash` (deterministic file hash). CI script computes hashes via AST extraction, enforces freshness. Landscape stores hash per node for reproducibility audit. See `docs/superpowers/specs/2026-04-15-plugin-version-audit-design.md` for full design.

**Tech Stack:** Python AST module (hash extraction), hashlib (SHA-256), SQLAlchemy Core (landscape schema), pluggy (plugin protocols), argparse (CI script CLI)

---

### Task 1: Hash computation module

The hash computation logic is shared between the CI script and the `--fix` mode. Extract it into a standalone module that can be unit-tested independently.

**Files:**
- Create: `scripts/cicd/plugin_hash.py`
- Test: `tests/unit/scripts/cicd/test_plugin_hash.py`
- Create: `tests/unit/scripts/cicd/__init__.py`

- [ ] **Step 1: Write failing tests for hash computation**

```python
# tests/unit/scripts/cicd/__init__.py
# (empty)
```

```python
# tests/unit/scripts/cicd/test_plugin_hash.py
"""Tests for plugin source file hash computation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def test_compute_hash_excludes_own_value(tmp_path: Path) -> None:
    """Hash must be identical regardless of the current source_file_hash value.

    The hash line is normalized to a placeholder before hashing, so changing
    only the hash value does not change the computed hash.
    """
    from scripts.cicd.plugin_hash import compute_source_file_hash

    content_a = textwrap.dedent("""\
        class MyPlugin:
            name = "test"
            plugin_version = "1.0.0"
            source_file_hash = "sha256:aaaaaaaaaaaaaaaa"

            def process(self):
                return True
    """)
    content_b = content_a.replace(
        'source_file_hash = "sha256:aaaaaaaaaaaaaaaa"',
        'source_file_hash = "sha256:bbbbbbbbbbbbbbbb"',
    )

    file_a = tmp_path / "plugin_a.py"
    file_b = tmp_path / "plugin_b.py"
    file_a.write_text(content_a)
    file_b.write_text(content_b)

    hash_a = compute_source_file_hash(file_a)
    hash_b = compute_source_file_hash(file_b)

    assert hash_a == hash_b, "Hash must be stable regardless of declared source_file_hash value"
    assert hash_a.startswith("sha256:")
    assert len(hash_a) == len("sha256:") + 16


def test_compute_hash_changes_on_content_change(tmp_path: Path) -> None:
    """Any non-hash-line change produces a different hash."""
    from scripts.cicd.plugin_hash import compute_source_file_hash

    content = textwrap.dedent("""\
        class MyPlugin:
            name = "test"
            source_file_hash = "sha256:0000000000000000"

            def process(self):
                return True
    """)
    modified = content.replace("return True", "return False")

    file_orig = tmp_path / "orig.py"
    file_mod = tmp_path / "mod.py"
    file_orig.write_text(content)
    file_mod.write_text(modified)

    assert compute_source_file_hash(file_orig) != compute_source_file_hash(file_mod)


def test_compute_hash_uses_raw_bytes(tmp_path: Path) -> None:
    """Hash is computed from raw bytes, not decoded text."""
    from scripts.cicd.plugin_hash import compute_source_file_hash

    content = b'class P:\n    source_file_hash = "sha256:0000000000000000"\n    x = "\xc3\xa9"\n'
    f = tmp_path / "utf8.py"
    f.write_bytes(content)

    h = compute_source_file_hash(f)
    assert h.startswith("sha256:")


def test_compute_hash_file_without_hash_line(tmp_path: Path) -> None:
    """File with no source_file_hash line is hashed as-is (no normalization needed)."""
    from scripts.cicd.plugin_hash import compute_source_file_hash

    content = "class P:\n    name = 'test'\n"
    f = tmp_path / "nohash.py"
    f.write_text(content)

    h = compute_source_file_hash(f)
    assert h.startswith("sha256:")


def test_extract_class_attribute_simple(tmp_path: Path) -> None:
    """AST extraction finds source_file_hash in a simple class body."""
    from scripts.cicd.plugin_hash import extract_plugin_attributes

    content = textwrap.dedent("""\
        class MyPlugin:
            name = "test"
            plugin_version = "1.0.0"
            source_file_hash = "sha256:abcdef0123456789"
    """)
    f = tmp_path / "plugin.py"
    f.write_text(content)

    attrs = extract_plugin_attributes(f)
    assert len(attrs) == 1
    assert attrs[0].class_name == "MyPlugin"
    assert attrs[0].plugin_version == "1.0.0"
    assert attrs[0].source_file_hash == "sha256:abcdef0123456789"
    assert attrs[0].hash_line_number is not None


def test_extract_class_attribute_annotated(tmp_path: Path) -> None:
    """AST extraction handles annotated assignment: source_file_hash: str = ..."""
    from scripts.cicd.plugin_hash import extract_plugin_attributes

    content = textwrap.dedent("""\
        class MyPlugin:
            name = "test"
            plugin_version: str = "2.0.0"
            source_file_hash: str = "sha256:abcdef0123456789"
    """)
    f = tmp_path / "plugin.py"
    f.write_text(content)

    attrs = extract_plugin_attributes(f)
    assert attrs[0].source_file_hash == "sha256:abcdef0123456789"


def test_extract_class_attribute_none_default(tmp_path: Path) -> None:
    """AST extraction detects source_file_hash = None (base class default)."""
    from scripts.cicd.plugin_hash import extract_plugin_attributes

    content = textwrap.dedent("""\
        class MyPlugin:
            name = "test"
            source_file_hash = None
    """)
    f = tmp_path / "plugin.py"
    f.write_text(content)

    attrs = extract_plugin_attributes(f)
    assert attrs[0].source_file_hash is None


def test_extract_ignores_non_plugin_classes(tmp_path: Path) -> None:
    """Helper classes in the same file are not extracted."""
    from scripts.cicd.plugin_hash import extract_plugin_attributes

    content = textwrap.dedent("""\
        class _Helper:
            source_file_hash = "sha256:0000000000000000"

        class MyPlugin:
            name = "test"
            plugin_version = "1.0.0"
            source_file_hash = "sha256:abcdef0123456789"
    """)
    f = tmp_path / "plugin.py"
    f.write_text(content)

    # extract_plugin_attributes returns classes that have BOTH name and source_file_hash
    attrs = extract_plugin_attributes(f)
    plugin_names = [a.class_name for a in attrs]
    assert "MyPlugin" in plugin_names


def test_fix_updates_hash_in_place(tmp_path: Path) -> None:
    """--fix mode rewrites the source_file_hash line to the correct value."""
    from scripts.cicd.plugin_hash import compute_source_file_hash, fix_source_file_hash

    content = textwrap.dedent("""\
        class MyPlugin:
            name = "test"
            plugin_version = "1.0.0"
            source_file_hash = "sha256:stale_stale_stale"

            def process(self):
                return True
    """)
    f = tmp_path / "plugin.py"
    f.write_text(content)

    expected_hash = compute_source_file_hash(f)
    fix_source_file_hash(f, "MyPlugin", expected_hash)

    # Read back and verify
    new_content = f.read_text()
    assert expected_hash in new_content
    assert "stale_stale_stale" not in new_content


def test_fix_is_idempotent(tmp_path: Path) -> None:
    """Running --fix twice produces identical file bytes."""
    from scripts.cicd.plugin_hash import compute_source_file_hash, fix_source_file_hash

    content = textwrap.dedent("""\
        class MyPlugin:
            name = "test"
            source_file_hash = "sha256:0000000000000000"
    """)
    f = tmp_path / "plugin.py"
    f.write_text(content)

    expected = compute_source_file_hash(f)
    fix_source_file_hash(f, "MyPlugin", expected)
    bytes_after_first = f.read_bytes()

    fix_source_file_hash(f, "MyPlugin", compute_source_file_hash(f))
    bytes_after_second = f.read_bytes()

    assert bytes_after_first == bytes_after_second
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_plugin_hash.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.cicd.plugin_hash'`

- [ ] **Step 3: Implement the hash computation module**

```python
# scripts/cicd/plugin_hash.py
"""Plugin source file hash computation and AST extraction.

Shared by the CI enforcement script (enforce_plugin_hashes.py) and the
--fix auto-update mode. Separated for independent unit testing.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_HASH_LINE_PATTERN = re.compile(
    rb'(\s*source_file_hash\s*(?::\s*\w+\s*)?=\s*)"sha256:[0-9a-f]+"',
)
_HASH_PLACEHOLDER = b'"sha256:0000000000000000"'


@dataclass(frozen=True)
class PluginAttributes:
    """Extracted plugin class attributes from AST parsing."""

    class_name: str
    plugin_version: str | None
    source_file_hash: str | None
    hash_line_number: int | None


def compute_source_file_hash(file_path: Path) -> str:
    """Compute the source file hash for a plugin module.

    Reads raw bytes (platform-independent), normalizes the source_file_hash
    line to a placeholder before hashing to avoid self-referential dependency.
    """
    raw = file_path.read_bytes()
    normalized = _HASH_LINE_PATTERN.sub(
        lambda m: m.group(1) + _HASH_PLACEHOLDER,
        raw,
    )
    digest = hashlib.sha256(normalized).hexdigest()[:16]
    return f"sha256:{digest}"


def extract_plugin_attributes(file_path: Path) -> list[PluginAttributes]:
    """Extract plugin class attributes via AST parsing (no imports).

    Finds classes that have a `name` class-level attribute (the pluggy
    plugin identifier) and returns their version/hash declarations.
    """
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    results: list[PluginAttributes] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        attrs: dict[str, tuple[object, int | None]] = {}
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attrs[target.id] = (_extract_value(item.value), item.lineno)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.value is not None:
                    attrs[item.target.id] = (_extract_value(item.value), item.lineno)

        if "name" not in attrs:
            continue

        pv_val, _ = attrs.get("plugin_version", (None, None))
        sfh_val, sfh_line = attrs.get("source_file_hash", (None, None))

        results.append(
            PluginAttributes(
                class_name=node.name,
                plugin_version=str(pv_val) if pv_val is not None else None,
                source_file_hash=str(sfh_val) if sfh_val is not None else None,
                hash_line_number=sfh_line,
            )
        )

    return results


def fix_source_file_hash(
    file_path: Path,
    class_name: str,
    correct_hash: str,
) -> None:
    """Rewrite the source_file_hash line in-place for a specific class.

    Uses AST to find the exact line number, then does a targeted line
    replacement. Preserves all formatting, comments, and surrounding code.
    """
    attrs_list = extract_plugin_attributes(file_path)
    target = next((a for a in attrs_list if a.class_name == class_name), None)
    if target is None:
        msg = f"Class {class_name!r} not found in {file_path}"
        raise ValueError(msg)
    if target.hash_line_number is None:
        msg = f"Class {class_name!r} in {file_path} has no source_file_hash line to fix"
        raise ValueError(msg)

    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    line_idx = target.hash_line_number - 1
    old_line = lines[line_idx]

    # Preserve leading whitespace
    indent = old_line[: len(old_line) - len(old_line.lstrip())]
    lines[line_idx] = f'{indent}source_file_hash = "{correct_hash}"\n'
    file_path.write_text("".join(lines), encoding="utf-8")


def _extract_value(node: ast.expr) -> object:
    """Extract a constant value from an AST expression node."""
    if isinstance(node, ast.Constant):
        return node.value
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_plugin_hash.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/cicd/plugin_hash.py tests/unit/scripts/cicd/__init__.py tests/unit/scripts/cicd/test_plugin_hash.py
git commit -m "feat(cicd): add plugin source file hash computation module

Shared hash computation (SHA-256 of normalized file content), AST
extraction of plugin attributes, and in-place fix rewriting. Core
building block for enforce_plugin_hashes.py CI script.

Spec: docs/superpowers/specs/2026-04-15-plugin-version-audit-design.md"
```

---

### Task 2: Protocol and base class changes

Add `source_file_hash: str | None` to all plugin protocols and base classes.

**Files:**
- Modify: `src/elspeth/contracts/plugin_protocols.py:93,224,368,481`
- Modify: `src/elspeth/plugins/infrastructure/base.py:142,423,689`

- [ ] **Step 1: Add `source_file_hash` to protocols**

In `src/elspeth/contracts/plugin_protocols.py`, add after each `plugin_version` line:

```python
# After SourceProtocol.plugin_version (line 93):
source_file_hash: str | None

# After TransformProtocol.plugin_version (line 224):
source_file_hash: str | None

# After BatchTransformProtocol.plugin_version (line 368):
source_file_hash: str | None

# After SinkProtocol.plugin_version (line 481):
source_file_hash: str | None
```

- [ ] **Step 2: Add `source_file_hash` to base classes**

In `src/elspeth/plugins/infrastructure/base.py`, add after each `plugin_version` line:

```python
# After BaseTransform.plugin_version (line 142):
source_file_hash: str | None = None

# After BaseSink.plugin_version (line 423):
source_file_hash: str | None = None

# After BaseSource.plugin_version (line 689):
source_file_hash: str | None = None
```

- [ ] **Step 3: Run mypy to verify protocol conformance**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/plugin_protocols.py src/elspeth/plugins/infrastructure/base.py`
Expected: PASS (base classes satisfy protocol via `None` default)

- [ ] **Step 4: Run existing tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/unit/plugins/ -x --timeout=30`
Expected: PASS (existing tests unaffected — `None` is the safe default)

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/contracts/plugin_protocols.py src/elspeth/plugins/infrastructure/base.py
git commit -m "feat(contracts): add source_file_hash to plugin protocols and base classes

Nullable str|None attribute with None default. CI enforces non-None;
base class default provides backwards compatibility during rollout."
```

---

### Task 3: Add `source_file_hash` and missing `plugin_version` to all plugins

Use the hash computation module from Task 1 to compute correct hashes, then add the attribute to all 28 plugin files. Also add missing `plugin_version = "1.0.0"` to the 9 plugins that omit it.

**Files:**
- Modify: All 28 plugin files listed in the spec audit (see inventory)

- [ ] **Step 1: Write a helper script to compute all hashes**

```bash
# One-liner: compute hashes for all plugin files
.venv/bin/python -c "
from pathlib import Path
from scripts.cicd.plugin_hash import compute_source_file_hash, extract_plugin_attributes

plugin_dirs = [
    'src/elspeth/plugins/sources',
    'src/elspeth/plugins/sinks',
    'src/elspeth/plugins/transforms',
    'src/elspeth/plugins/transforms/azure',
    'src/elspeth/plugins/transforms/llm',
    'src/elspeth/plugins/transforms/rag',
]
for d in plugin_dirs:
    for f in sorted(Path(d).glob('*.py')):
        if f.name.startswith('_') or f.name == '__init__.py':
            continue
        attrs = extract_plugin_attributes(f)
        for a in attrs:
            h = compute_source_file_hash(f)
            print(f'{f}  {a.class_name}  version={a.plugin_version}  hash={h}')
"
```

- [ ] **Step 2: Add `source_file_hash` to each plugin file**

For each plugin, add `source_file_hash = "sha256:<computed>"` after the `plugin_version` line. For the 7 plugins missing `plugin_version`, also add `plugin_version = "1.0.0"`.

**Pattern (shown for one plugin — repeat for all 32):**

```python
# In src/elspeth/plugins/sources/csv_source.py, after plugin_version = "1.0.0":
    source_file_hash = "sha256:<value-from-step-1>"
```

**The 9 plugins that need `plugin_version = "1.0.0"` added:**
- `sources/azure_blob_source.py` (AzureBlobSource)
- `sources/dataverse.py` (DataverseSource)
- `sinks/dataverse.py` (DataverseSink)
- `transforms/azure/content_safety.py` (AzureContentSafety)
- `transforms/azure/prompt_shield.py` (AzurePromptShield)
- `transforms/rag/transform.py` (RAGRetrievalTransform)
- `transforms/llm/transform.py` (LLMTransform)
<!-- transforms/llm/azure_batch.py (AzureBatchLLMTransform) and
     transforms/llm/openrouter_batch.py (OpenRouterBatchLLMTransform)
     were retired 2026-05-06 per ADR-020 and removed from this list. -->

For these, add both lines after the `name = "..."` attribute:
```python
    plugin_version = "1.0.0"
    source_file_hash = "sha256:<value-from-step-1>"
```

- [ ] **Step 3: Run the hash computation script to verify all hashes**

After adding all attributes, re-run the script from Step 1. Every plugin should now show its declared hash matching the computed hash.

- [ ] **Step 4: Run existing tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/ -x --timeout=60`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/
git commit -m "feat(plugins): add source_file_hash to all 28 plugins

Adds missing plugin_version='1.0.0' to 9 plugins (AzureBlobSource,
DataverseSource, DataverseSink, AzureContentSafety, AzurePromptShield,
RAGRetrievalTransform, LLMTransform, AzureBatchLLMTransform,
OpenRouterBatchLLMTransform).

Adds source_file_hash with correct SHA-256 truncated hash to all 28
plugin entry-point files."
```

---

### Task 4: CI enforcement script

Full CLI script with `check` and `check --fix` subcommands.

**Files:**
- Create: `scripts/cicd/enforce_plugin_hashes.py`
- Create: `config/cicd/enforce_plugin_hashes/` (allowlist directory)
- Test: `tests/unit/scripts/cicd/test_enforce_plugin_hashes.py`

- [ ] **Step 1: Write failing tests for the enforcement script**

```python
# tests/unit/scripts/cicd/test_enforce_plugin_hashes.py
"""Tests for enforce_plugin_hashes.py CI enforcement script."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def plugin_tree(tmp_path: Path) -> Path:
    """Create a minimal plugin directory tree for testing."""
    src = tmp_path / "src" / "elspeth" / "plugins"
    sources = src / "sources"
    sources.mkdir(parents=True)

    # A well-formed plugin
    good = sources / "good_source.py"
    good.write_text(textwrap.dedent("""\
        class GoodSource:
            name = "good"
            plugin_version = "1.0.0"
            source_file_hash = "sha256:placeholder_value_"
    """))

    # Compute and fix the hash so it's correct
    from scripts.cicd.plugin_hash import compute_source_file_hash, fix_source_file_hash

    correct = compute_source_file_hash(good)
    fix_source_file_hash(good, "GoodSource", correct)

    return tmp_path


def _run_check(root: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/cicd/enforce_plugin_hashes.py",
            "check",
            "--root",
            str(root / "src" / "elspeth"),
            *extra_args,
        ],
        capture_output=True,
        text=True,
    )


class TestEnforcePluginHashes:
    def test_passes_on_correct_hashes(self, plugin_tree: Path) -> None:
        result = _run_check(plugin_tree)
        assert result.returncode == 0, result.stderr

    def test_fails_on_stale_hash(self, plugin_tree: Path) -> None:
        plugin = plugin_tree / "src" / "elspeth" / "plugins" / "sources" / "good_source.py"
        content = plugin.read_text()
        plugin.write_text(content + "\n# new comment\n")

        result = _run_check(plugin_tree)
        assert result.returncode != 0
        assert "stale" in result.stdout.lower() or "expected" in result.stdout.lower()

    def test_fails_on_missing_hash(self, plugin_tree: Path) -> None:
        no_hash = plugin_tree / "src" / "elspeth" / "plugins" / "sources" / "nohash.py"
        no_hash.write_text(textwrap.dedent("""\
            class NoHashSource:
                name = "nohash"
                plugin_version = "1.0.0"
        """))

        result = _run_check(plugin_tree)
        assert result.returncode != 0
        assert "no source_file_hash" in result.stdout.lower()

    def test_fails_on_missing_plugin_version(self, plugin_tree: Path) -> None:
        no_ver = plugin_tree / "src" / "elspeth" / "plugins" / "sources" / "nover.py"
        no_ver.write_text(textwrap.dedent("""\
            class NoVerSource:
                name = "nover"
                source_file_hash = "sha256:0000000000000000"
        """))

        result = _run_check(plugin_tree)
        assert result.returncode != 0
        assert "no version" in result.stdout.lower() or "0.0.0" in result.stdout.lower()

    def test_fix_updates_stale_hash(self, plugin_tree: Path) -> None:
        plugin = plugin_tree / "src" / "elspeth" / "plugins" / "sources" / "good_source.py"
        content = plugin.read_text()
        plugin.write_text(content + "\n# changed\n")

        result = _run_check(plugin_tree, "--fix")
        assert result.returncode == 0

        # Verify check now passes
        recheck = _run_check(plugin_tree)
        assert recheck.returncode == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_plugin_hashes.py -v`
Expected: FAIL (script doesn't exist yet)

- [ ] **Step 3: Implement the enforcement script**

```python
# scripts/cicd/enforce_plugin_hashes.py
"""CI enforcement: plugin source_file_hash declarations must match computed values.

Usage:
    python scripts/cicd/enforce_plugin_hashes.py check --root src/elspeth
    python scripts/cicd/enforce_plugin_hashes.py check --root src/elspeth --fix

check:     Verify all plugins have correct source_file_hash (CI mode).
check --fix:  Auto-update stale hashes in-place (developer mode).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.cicd.plugin_hash import (
    compute_source_file_hash,
    extract_plugin_attributes,
    fix_source_file_hash,
)

PLUGIN_DIRS = [
    "plugins/sources",
    "plugins/sinks",
    "plugins/transforms",
    "plugins/transforms/azure",
    "plugins/transforms/llm",
    "plugins/transforms/rag",
]

# Files in plugin directories that are NOT plugin entry points
_EXCLUDED_FILES = frozenset({
    "__init__.py",
    "base.py",
    "config.py",
    "validation.py",
    "templates.py",
    "langfuse.py",
    "tracing.py",
    "multi_query.py",
    "capacity_errors.py",
    "provider.py",
})


def _discover_plugin_files(root: Path) -> list[Path]:
    """Find all plugin entry-point files."""
    files: list[Path] = []
    for rel_dir in PLUGIN_DIRS:
        d = root / rel_dir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.py")):
            if f.name.startswith("_") or f.name in _EXCLUDED_FILES:
                continue
            files.append(f)
    return files


EXPECTED_PLUGIN_COUNT = 28


def run_check(root: Path, *, fix: bool = False) -> int:
    """Run the enforcement check. Returns 0 on success, 1 on failure."""
    files = _discover_plugin_files(root)

    # Count guard: fail loudly if discovery finds fewer plugins than expected
    # (guards against wrong --root, missing directories, or scan regressions)
    plugin_count = sum(
        len(extract_plugin_attributes(f)) for f in files
    )
    if plugin_count < EXPECTED_PLUGIN_COUNT:
        print(
            f"DISCOVERY ERROR: found {plugin_count} plugins, "
            f"expected at least {EXPECTED_PLUGIN_COUNT}. "
            f"Check --root path and PLUGIN_DIRS."
        )
        return 1

    violations: list[str] = []
    fixed: list[str] = []

    for file_path in files:
        attrs_list = extract_plugin_attributes(file_path)
        if not attrs_list:
            continue

        computed = compute_source_file_hash(file_path)
        rel = file_path.relative_to(root)

        for attrs in attrs_list:
            # Check plugin_version
            if attrs.plugin_version is None or attrs.plugin_version == "0.0.0":
                violations.append(
                    f"{rel} ({attrs.class_name}): no version declaration "
                    f"(plugin_version is {attrs.plugin_version!r})"
                )

            # Check source_file_hash
            if attrs.source_file_hash is None:
                violations.append(
                    f"{rel} ({attrs.class_name}): no source_file_hash declaration"
                )
            elif attrs.source_file_hash != computed:
                if fix:
                    fix_source_file_hash(file_path, attrs.class_name, computed)
                    fixed.append(f"{rel} ({attrs.class_name}): updated to {computed}")
                    # Recompute after fix (file content changed)
                    computed = compute_source_file_hash(file_path)
                else:
                    violations.append(
                        f"{rel} ({attrs.class_name}): stale source_file_hash\n"
                        f"  declared: {attrs.source_file_hash}\n"
                        f"  expected: {computed}"
                    )

    if fixed:
        print(f"FIXED {len(fixed)} hash(es):")
        for msg in fixed:
            print(f"  {msg}")
        print()

    if violations:
        print(f"{'=' * 60}")
        print(f"VIOLATIONS FOUND: {len(violations)}")
        print(f"{'=' * 60}")
        print()
        for v in violations:
            print(v)
            print()
        print(f"{'=' * 60}")
        print("CHECK FAILED")
        print(f"{'=' * 60}")
        return 1

    if not fixed:
        print("No bug-hiding patterns detected. Check passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enforce plugin source_file_hash declarations."
    )
    sub = parser.add_subparsers(dest="command")
    check_parser = sub.add_parser("check", help="Verify plugin hashes")
    check_parser.add_argument("--root", type=Path, required=True)
    check_parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-update stale hashes (developer mode, not for CI)",
    )

    args = parser.parse_args()
    if args.command != "check":
        parser.print_help()
        sys.exit(1)

    sys.exit(run_check(args.root, fix=args.fix))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create allowlist directory**

```bash
mkdir -p config/cicd/enforce_plugin_hashes
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/ -v`
Expected: All tests PASS

- [ ] **Step 6: Run the enforcement script against the real codebase**

Run: `.venv/bin/python scripts/cicd/enforce_plugin_hashes.py check --root src/elspeth`
Expected: PASS (all 28 plugins have correct hashes from Task 3)

- [ ] **Step 7: Commit**

```bash
git add scripts/cicd/enforce_plugin_hashes.py config/cicd/enforce_plugin_hashes/ tests/unit/scripts/cicd/test_enforce_plugin_hashes.py
git commit -m "feat(cicd): add enforce_plugin_hashes.py CI enforcement script

Discovers plugin files, computes SHA-256 source file hashes, verifies
against declared source_file_hash attribute via AST extraction.
check mode for CI, check --fix for developer auto-update."
```

---

### Task 5: Landscape schema, Node dataclass, and data path

Add `source_file_hash` column to the landscape nodes table, the `Node` dataclass, the `NodeLoader`, and `register_node()`.

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py:71-99`
- Modify: `src/elspeth/contracts/audit.py:93-120`
- Modify: `src/elspeth/core/landscape/model_loaders.py:101-137`
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:922-1019`
- Test: `tests/unit/core/landscape/test_source_file_hash.py` (or extend existing)

- [ ] **Step 1: Write failing tests for Node validation and landscape round-trip**

```python
# tests/unit/core/landscape/test_source_file_hash.py
"""Tests for source_file_hash in the landscape audit trail."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from elspeth.contracts.audit import Node
from elspeth.contracts.enums import Determinism, NodeType


def _make_node(**overrides) -> Node:
    defaults = dict(
        node_id="node-1",
        run_id="run-1",
        plugin_name="test",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0.0",
        determinism=Determinism.DETERMINISTIC,
        config_hash="a" * 64,
        config_json="{}",
        sequence_in_pipeline=0,
        registered_at=datetime.now(tz=UTC),
        source_file_hash=None,
    )
    defaults.update(overrides)
    return Node(**defaults)


class TestNodeSourceFileHashValidation:
    def test_accepts_none(self) -> None:
        """None is valid (engine nodes like gates and coalesce)."""
        node = _make_node(source_file_hash=None)
        assert node.source_file_hash is None

    def test_accepts_valid_hash(self) -> None:
        """Valid sha256:<16-hex> format is accepted."""
        node = _make_node(source_file_hash="sha256:abcdef0123456789")
        assert node.source_file_hash == "sha256:abcdef0123456789"

    def test_rejects_invalid_format(self) -> None:
        """Invalid format crashes in __post_init__ (Tier 1 crash-on-anomaly)."""
        with pytest.raises((ValueError, AssertionError)):
            _make_node(source_file_hash="not-a-valid-hash")

    def test_rejects_empty_string(self) -> None:
        """Empty string is not a valid source_file_hash."""
        with pytest.raises((ValueError, AssertionError)):
            _make_node(source_file_hash="")

    def test_rejects_wrong_prefix(self) -> None:
        """Must start with sha256:."""
        with pytest.raises((ValueError, AssertionError)):
            _make_node(source_file_hash="md5:abcdef0123456789")

    def test_rejects_wrong_length(self) -> None:
        """Must have exactly 16 hex chars after prefix."""
        with pytest.raises((ValueError, AssertionError)):
            _make_node(source_file_hash="sha256:abc")


class TestNodeSourceFileHashRoundTrip:
    """Spec case 12: register_node() stores source_file_hash and
    NodeLoader.load() reads it back correctly."""

    def test_store_and_retrieve(self, landscape_db) -> None:
        """source_file_hash survives a register_node → load round-trip."""
        from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
        from elspeth.core.landscape.data_flow_repository import DataFlowRepository
        from elspeth.contracts.audit import SchemaConfig

        run_repo = RunLifecycleRepository(landscape_db)
        data_repo = DataFlowRepository(landscape_db)

        run = run_repo.begin_run(pipeline_hash="test", config_json="{}")
        node = data_repo.register_node(
            run_id=run.run_id,
            node_id="transform-1",
            plugin_name="passthrough",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={"on_success": "continue"},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
            source_file_hash="sha256:abcdef0123456789",
        )
        assert node.source_file_hash == "sha256:abcdef0123456789"

    def test_none_for_engine_nodes(self, landscape_db) -> None:
        """Engine-internal nodes (gates, coalesce) store None."""
        from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
        from elspeth.core.landscape.data_flow_repository import DataFlowRepository
        from elspeth.contracts.audit import SchemaConfig

        run_repo = RunLifecycleRepository(landscape_db)
        data_repo = DataFlowRepository(landscape_db)

        run = run_repo.begin_run(pipeline_hash="test", config_json="{}")
        node = data_repo.register_node(
            run_id=run.run_id,
            node_id="gate-1",
            plugin_name="config_gate",
            node_type=NodeType.GATE,
            plugin_version="engine:0.7.8",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
            source_file_hash=None,
        )
        assert node.source_file_hash is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_source_file_hash.py -v`
Expected: FAIL (Node doesn't have source_file_hash field yet)

- [ ] **Step 3: Add `source_file_hash` to Node dataclass with validation**

In `src/elspeth/contracts/audit.py`, add the field to the `Node` dataclass and validation to `__post_init__`:

```python
# After plugin_version field (line 103), add:
source_file_hash: str | None = None

# In __post_init__ (after existing validations), add:
if self.source_file_hash is not None:
    import re
    if not re.fullmatch(r"sha256:[0-9a-f]{16}", self.source_file_hash):
        raise ValueError(
            f"Tier 1: source_file_hash must match 'sha256:<16-hex>' or be None, "
            f"got {self.source_file_hash!r} for node {self.node_id!r}"
        )
```

- [ ] **Step 4: Add column to landscape schema**

In `src/elspeth/core/landscape/schema.py`, add to the `nodes` table definition after `plugin_version` column (line 78):

```python
Column("source_file_hash", String(32), nullable=True),
```

- [ ] **Step 5: Update NodeLoader**

In `src/elspeth/core/landscape/model_loaders.py`, update the `NodeLoader.load()` method to read the new column. Add after the `plugin_version` read:

```python
source_file_hash=row.source_file_hash,
```

Direct attribute access — no `getattr` fallback. Per the no-legacy-code policy, there are no existing databases to support. The column is always present because `schema.py` defines it.

- [ ] **Step 6: Update `register_node()` signature and write**

In `src/elspeth/core/landscape/data_flow_repository.py`:

Add `source_file_hash: str | None = None` parameter to `register_node()` method signature.

In the `Node(...)` construction inside the method, add `source_file_hash=source_file_hash`.

In the database insert dict, add `"source_file_hash": node.source_file_hash`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_source_file_hash.py -v`
Expected: All 6 tests PASS

Run: `.venv/bin/python -m pytest tests/unit/landscape/ tests/integration/audit/ -x --timeout=60`
Expected: PASS (no regressions in existing landscape tests)

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/contracts/audit.py src/elspeth/core/landscape/schema.py src/elspeth/core/landscape/model_loaders.py src/elspeth/core/landscape/data_flow_repository.py tests/unit/core/landscape/test_source_file_hash.py
git commit -m "feat(landscape): add source_file_hash to Node and nodes table

Nullable column (None for engine-internal nodes). Format validated in
Node.__post_init__ (sha256:<16-hex> or None). Direct attribute access
in NodeLoader — no backward-compat fallback per no-legacy-code policy."
```

---

### Task 6: Orchestrator integration

Pass `source_file_hash` from plugin instances to `register_node()` in both the main orchestrator and the export path.

**Files:**
- Modify: `src/elspeth/engine/orchestrator/core.py:1333-1492`
- Modify: `src/elspeth/engine/orchestrator/export.py:100`

- [ ] **Step 1: Update orchestrator `_register_nodes_with_landscape()`**

In `src/elspeth/engine/orchestrator/core.py`, find the block around line 1372-1388 where `plugin_version` is extracted. Add `source_file_hash` extraction with the same pattern:

```python
# After the plugin_version extraction block:
if node_id in config_gate_node_ids or node_id in coalesce_node_ids:
    source_file_hash = None  # Engine-internal nodes
else:
    source_file_hash = plugin.source_file_hash
```

Then in the `factory.data_flow.register_node()` call at line 1406, add:

```python
source_file_hash=source_file_hash,
```

- [ ] **Step 2: Update export path**

In `src/elspeth/engine/orchestrator/export.py`, update the `register_node()` call at line 100 to add:

```python
source_file_hash=sink.source_file_hash,
```

- [ ] **Step 3: Run integration tests**

Run: `.venv/bin/python -m pytest tests/integration/pipeline/ -x --timeout=120`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/engine/orchestrator/core.py src/elspeth/engine/orchestrator/export.py
git commit -m "feat(orchestrator): pass source_file_hash to landscape registration

Plugin nodes pass their declared hash; config gates and coalesce
nodes pass None (engine-internal). Export path also updated."
```

---

### Task 7: Test fixtures and regression verification

Update test fixtures that create mock plugins, then run the full test suite.

**Files:**
- Modify: `tests/fixtures/base_classes.py:44,141,177`
- Modify: `tests/fixtures/landscape.py:122`
- Possibly: other test files creating mock plugins

- [ ] **Step 1: Update test fixture base classes**

In `tests/fixtures/base_classes.py`, add `source_file_hash` to each test base class after `plugin_version`:

```python
# _TestSourceBase (line 44):
source_file_hash = "sha256:00000000000000a0"

# _TestSinkBase (line 141):
source_file_hash = "sha256:00000000000000b0"

# _TestTransformBase (line 177):
source_file_hash = "sha256:00000000000000c0"
```

- [ ] **Step 2: Update landscape test fixture**

In `tests/fixtures/landscape.py`, update `make_recorder_with_run()` at line 122 to pass `source_file_hash=None` (or a test value) to `register_node()`.

- [ ] **Step 3: Search for other fixtures that need updating**

```bash
grep -rn "plugin_version.*=" tests/ | grep -v __pycache__ | grep -v ".pyc"
```

Update any additional fixtures that create mock plugin objects to include `source_file_hash`.

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x --timeout=300`
Expected: PASS (no regressions)

- [ ] **Step 5: Run all enforcement scripts**

```bash
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/
.venv/bin/python scripts/cicd/enforce_plugin_hashes.py check --root src/elspeth
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: update fixtures for source_file_hash attribute

All test plugin mocks, base classes, and landscape fixtures now
include source_file_hash. Full test suite passes."
```

---

### Task 8: Pre-commit integration

Add the enforcement script as a check-only hook.

**Files:**
- Modify: `.pre-commit-config.yaml`

- [ ] **Step 1: Add hook to pre-commit config**

Add after the last local hook entry (enforce-frozen-annotations):

```yaml
  - id: enforce-plugin-hashes
    name: Enforce Plugin Hashes
    entry: .venv/bin/python scripts/cicd/enforce_plugin_hashes.py check --root src/elspeth
    language: system
    types: [python]
    pass_filenames: false
```

- [ ] **Step 2: Verify hook runs**

```bash
.venv/bin/python -m pre_commit run enforce-plugin-hashes --all-files
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "ci: add enforce-plugin-hashes pre-commit hook (check-only)

Matches project convention: hooks are check-only, no auto-fix.
Developers run --fix manually before staging."
```

---

### Task 9: Final verification and allowlist update

Run the full CI gauntlet to confirm zero regressions.

- [ ] **Step 1: Run all pre-commit hooks**

```bash
.venv/bin/python -m pre_commit run --all-files
```

Expected: All hooks PASS

- [ ] **Step 2: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ --timeout=300
```

Expected: PASS (modulo any pre-existing failures unrelated to this feature)

- [ ] **Step 3: Update tier-model allowlist if needed**

If the tier-model enforcement script reports new fingerprint violations from the files touched, update the allowlist entries in `config/cicd/enforce_tier_model/`.

- [ ] **Step 4: Create filigree tracking issue**

Create a filigree task for the future work: "Transitive dependency hashing — extend source_file_hash to cover the full import closure, not just the entry-point file." Priority P3 (backlog). Label `tech-debt`.
