"""Tests for enforce_plugin_hashes.py CI enforcement script.

Runs the script via subprocess to exercise the CLI interface end-to-end.
Uses a ``plugin_tree`` fixture that creates a minimal directory structure
with a single well-formed plugin, and ``--min-plugins 1`` to bypass the
production count guard (which expects 28 plugins).
"""

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

    # A well-formed plugin with a placeholder hash (must use valid hex format
    # so the normalization regex matches during hash computation)
    good = sources / "good_source.py"
    good.write_text(
        textwrap.dedent("""\
        class GoodSource:
            name = "good"
            plugin_version = "1.0.0"
            source_file_hash = "sha256:0000000000000000"
    """)
    )

    # Compute and fix the hash so it's correct
    from scripts.cicd.plugin_hash import compute_source_file_hash, fix_source_file_hash

    correct = compute_source_file_hash(good)
    fix_source_file_hash(good, "GoodSource", correct)

    return tmp_path


def _run_check(root: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.cicd.enforce_plugin_hashes",
            "check",
            "--root",
            str(root / "src" / "elspeth"),
            "--min-plugins",
            "1",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[4]),  # project root
    )


class TestEnforcePluginHashes:
    def test_passes_on_correct_hashes(self, plugin_tree: Path) -> None:
        result = _run_check(plugin_tree)
        assert result.returncode == 0, result.stderr + result.stdout

    def test_fails_on_stale_hash(self, plugin_tree: Path) -> None:
        plugin = plugin_tree / "src" / "elspeth" / "plugins" / "sources" / "good_source.py"
        content = plugin.read_text()
        plugin.write_text(content + "\n# new comment\n")

        result = _run_check(plugin_tree)
        assert result.returncode != 0
        assert "stale" in result.stdout.lower() or "expected" in result.stdout.lower()

    def test_fails_on_missing_hash(self, plugin_tree: Path) -> None:
        no_hash = plugin_tree / "src" / "elspeth" / "plugins" / "sources" / "nohash.py"
        no_hash.write_text(
            textwrap.dedent("""\
            class NoHashSource:
                name = "nohash"
                plugin_version = "1.0.0"
        """)
        )

        result = _run_check(plugin_tree)
        assert result.returncode != 0
        assert "no source_file_hash" in result.stdout.lower()

    def test_fails_on_missing_plugin_version(self, plugin_tree: Path) -> None:
        no_ver = plugin_tree / "src" / "elspeth" / "plugins" / "sources" / "nover.py"
        no_ver.write_text(
            textwrap.dedent("""\
            class NoVerSource:
                name = "nover"
                source_file_hash = "sha256:0000000000000000"
        """)
        )

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
