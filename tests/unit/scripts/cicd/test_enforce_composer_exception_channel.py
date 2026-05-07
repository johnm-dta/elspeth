"""Tests for enforce_composer_exception_channel.py.

Verifies the CI gate detects bare raises of TypeError/ValueError/UnicodeError
inside tool-handler files and accepts raises of ToolArgumentError.

Co-located with the other CI-enforcer tests at tests/unit/scripts/cicd/ —
mirrors test_enforce_plugin_hashes.py / test_enforce_freeze_guards.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the enforcer as a module, with cwd set to the project root.

    Mirrors the pattern in test_enforce_plugin_hashes.py:50-62. The
    module form requires the project root on sys.path so `scripts` is
    importable as a package; we set cwd explicitly rather than relying
    on wherever pytest was invoked from.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.cicd.enforce_composer_exception_channel",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[4]),  # project root
    )


def _make_composer_tree(tmp_path: Path) -> Path:
    """Create a fake src layout so the enforcer's existence assertion holds.

    Layout created:
        tmp_path/web/composer/tools.py
    The enforcer is invoked with --root tmp_path (test-mode), so
    relative_to(root) yields "web/composer/tools.py".
    """
    target_dir = tmp_path / "web" / "composer"
    target_dir.mkdir(parents=True)
    return target_dir / "tools.py"


class TestComposerExceptionChannelEnforcer:
    def test_clean_tree_passes(self, tmp_path: Path) -> None:
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "from elspeth.web.composer.protocol import ToolArgumentError\n"
            "def f(x):\n"
            "    if not isinstance(x, str):\n"
            "        raise ToolArgumentError(argument='x', expected='a string', actual_type=type(x).__name__)\n"
        )
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode == 0, result.stdout + result.stderr

    def test_bare_type_error_fails(self, tmp_path: Path) -> None:
        target = _make_composer_tree(tmp_path)
        target.write_text("def f(x):\n    if not isinstance(x, str):\n        raise TypeError('bad')\n")
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode != 0
        assert "TypeError" in result.stdout

    def test_bare_value_error_fails(self, tmp_path: Path) -> None:
        target = _make_composer_tree(tmp_path)
        target.write_text("def f(x):\n    raise ValueError('bad')\n")
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode != 0

    def test_explicit_raise_inside_try_except_still_flagged(self, tmp_path: Path) -> None:
        """An explicit `raise ValueError` inside a try/except is still flagged.

        The enforcer is intentionally scope-unaware: it detects raise-site
        classes, not control-flow containment. The legitimate contained
        pattern is `try: int(x) except ValueError: ...` (no explicit raise),
        covered by `test_implicit_raise_from_coercion_is_not_flagged` below.
        A handler author who writes `raise ValueError(...)` even inside a
        local try/except is using the wrong exception class for the channel
        and should be flagged.

        False-positive risk: legitimate handler code that catches its own
        raise purely for diagnostic formatting (e.g. raise-then-reformat
        patterns) will also trip this rule. If any such pattern is found
        during Task 5's sweep, resolve by refactoring to `raise
        ToolArgumentError(formatted_message) from exc` rather than
        allowlisting — the narrow exception class carries the
        channel-discipline semantics, and allowlisting would hide the
        design-intent mismatch.
        """
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "def _failure_result(state, msg): return msg\n"
            "def f(x, state):\n"
            "    try:\n"
            "        raise ValueError('bad')\n"
            "    except ValueError as exc:\n"
            "        return _failure_result(state, str(exc))\n"
        )
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode != 0

    def test_implicit_raise_from_coercion_is_not_flagged(self, tmp_path: Path) -> None:
        """Implicit raises (e.g. `int(x)` failing) are not textual raise sites.

        AST-level raise-node walking only sees explicit `raise X` nodes, so
        `int(x)` inside a try/except is invisible to the enforcer — which
        is correct. The handler's job is to wrap the implicit raise and
        either return _failure_result or re-raise as ToolArgumentError.
        """
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "def _failure_result(state, msg): return msg\n"
            "def f(x, state):\n"
            "    try:\n"
            "        int(x)\n"
            "    except ValueError as exc:\n"
            "        return _failure_result(state, str(exc))\n"
        )
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode == 0, result.stdout + result.stderr

    def test_missing_target_file_fails_closed(self, tmp_path: Path) -> None:
        """If web/composer/tools.py does not exist under the root, exit non-zero.

        Closes the threat-analyst fail-open concern: a future rename of
        tools.py would otherwise cause the enforcer to silently scan nothing.
        """
        # No _make_composer_tree() call — target is absent.
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode != 0
        assert "web/composer/tools.py" in (result.stdout + result.stderr)

    def test_allowlist_entry_suppresses_matching_finding(self, tmp_path: Path) -> None:
        """An allowlist entry at the exact (file, line) of a bare raise
        suppresses the finding and the enforcer exits 0.

        The allowlist is load-bearing for legitimate exceptions (e.g. the
        JSON-serializer invariant-violation raise in service.py, if it were
        ever brought into scope). Without this test, the allowlist is an
        unverified mechanism — an implementation bug that dropped entries
        would silently break the escape hatch and surface only when a
        real exemption is needed under time pressure.
        """
        target = _make_composer_tree(tmp_path)
        target.write_text("def f(x):\n    raise TypeError('bad')\n")
        # Line 2 is the raise site in the file above.
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text(
            'allowed:\n  - file: web/composer/tools.py\n    line: 2\n    justification: "test-only exemption; real code never uses this"\n'
        )
        result = _run(
            [
                "check",
                "--root",
                str(tmp_path),
                "--allowlist",
                str(allowlist_dir),
            ]
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_allowlist_entry_with_empty_justification_fails(self, tmp_path: Path) -> None:
        """Allowlist entries MUST have a non-empty justification field.

        Without this gate, the allowlist degrades into a silent dumping
        ground: any developer who hits a CI failure can add an entry with
        no explanation and the build turns green. The justification field
        forces a paper trail for every exemption.
        """
        target = _make_composer_tree(tmp_path)
        target.write_text("def f(x):\n    raise TypeError('bad')\n")
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text('allowed:\n  - file: web/composer/tools.py\n    line: 2\n    justification: ""\n')
        result = _run(
            [
                "check",
                "--root",
                str(tmp_path),
                "--allowlist",
                str(allowlist_dir),
            ]
        )
        assert result.returncode != 0
        assert "justification" in (result.stdout + result.stderr)

    def test_allowlist_entry_missing_justification_key_fails(self, tmp_path: Path) -> None:
        """An allowlist entry that omits the justification key entirely
        (as opposed to providing an empty string) is also rejected."""
        target = _make_composer_tree(tmp_path)
        target.write_text("def f(x):\n    raise TypeError('bad')\n")
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text("allowed:\n  - file: web/composer/tools.py\n    line: 2\n")
        result = _run(
            [
                "check",
                "--root",
                str(tmp_path),
                "--allowlist",
                str(allowlist_dir),
            ]
        )
        assert result.returncode != 0
        assert "justification" in (result.stdout + result.stderr)

    def test_allowlist_entry_missing_file_key_fails(self, tmp_path: Path) -> None:
        """An allowlist entry missing the required 'file' key is rejected
        with a targeted error, not an uninformative KeyError crash."""
        target = _make_composer_tree(tmp_path)
        target.write_text("def f(x):\n    raise TypeError('bad')\n")
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text("allowed:\n  - line: 2\n    justification: 'missing file key'\n")
        result = _run(["check", "--root", str(tmp_path), "--allowlist", str(allowlist_dir)])
        assert result.returncode != 0
        assert "file" in (result.stdout + result.stderr).lower()

    def test_allowlist_entry_missing_line_key_fails(self, tmp_path: Path) -> None:
        """An allowlist entry missing the required 'line' key is rejected
        with a targeted error, not an uninformative KeyError crash."""
        target = _make_composer_tree(tmp_path)
        target.write_text("def f(x):\n    raise TypeError('bad')\n")
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text("allowed:\n  - file: web/composer/tools.py\n    justification: 'missing line key'\n")
        result = _run(["check", "--root", str(tmp_path), "--allowlist", str(allowlist_dir)])
        assert result.returncode != 0
        assert "line" in (result.stdout + result.stderr).lower()

    def test_allowlist_entry_non_integer_line_fails(self, tmp_path: Path) -> None:
        """A non-integer 'line' value is rejected with a targeted error,
        not an uninformative ValueError from int() coercion."""
        target = _make_composer_tree(tmp_path)
        target.write_text("def f(x):\n    raise TypeError('bad')\n")
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "_defaults.yaml").write_text(
            "allowed:\n  - file: web/composer/tools.py\n    line: 'not-a-number'\n    justification: 'bad line'\n"
        )
        result = _run(["check", "--root", str(tmp_path), "--allowlist", str(allowlist_dir)])
        assert result.returncode != 0
        assert "line" in (result.stdout + result.stderr).lower()

    def test_allowlist_path_not_found_fails_closed(self, tmp_path: Path) -> None:
        """If --allowlist points to a non-existent directory, exit non-zero.

        Prevents a workflow typo from masking allowlisted entries by
        silently treating the typo as an empty allowlist. Fail closed.
        """
        target = _make_composer_tree(tmp_path)
        target.write_text(
            "from elspeth.web.composer.protocol import ToolArgumentError\n"
            "def f(x):\n"
            "    raise ToolArgumentError(argument='x', expected='a string', actual_type=type(x).__name__)\n"
        )
        bogus = tmp_path / "does-not-exist"
        result = _run(["check", "--root", str(tmp_path), "--allowlist", str(bogus)])
        assert result.returncode != 0
        assert "does not exist" in (result.stdout + result.stderr).lower()
