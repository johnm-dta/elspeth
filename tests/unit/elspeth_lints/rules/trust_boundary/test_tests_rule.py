"""Tests for the ``trust_boundary.tests`` honesty-gate rule.

The tests rule reads pytest nodeids from ``test_ref`` and walks the referenced
file. We materialise tiny on-disk fixtures inside ``tmp_path`` so the rule's
file-lookup path is exercised end-to-end (no mocking of the filesystem).
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext
from elspeth_lints.rules.trust_boundary.tests import RULE as TESTS_RULE


def _analyze_at(source: str, *, repo_root: Path) -> list[Finding]:
    """Parse ``source`` and run the rule with ``repo_root`` as the scan root.

    The rule reads ``context.root`` and derives the repository root from it
    (``root.parent.parent`` if root looks like ``src/elspeth``, otherwise root
    itself). Tests use the plain ``repo_root`` form so ``tests/...`` paths
    inside ``test_ref`` resolve directly.
    """
    tree = ast.parse(textwrap.dedent(source))
    return list(
        TESTS_RULE.analyze(
            tree, Path("decorated.py"), RuleContext(root=repo_root)
        )
    )


def _write_test_file(repo_root: Path, relative: str, content: str) -> Path:
    target = repo_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content), encoding="utf-8")
    return target


def test_accepts_resolved_test_with_pytest_raises(tmp_path: Path) -> None:
    _write_test_file(
        tmp_path,
        "tests/test_foo.py",
        """
        import pytest

        def test_rejects_bad_input():
            with pytest.raises(ValueError):
                raise ValueError("nope")
        """,
    )
    findings = _analyze_at(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref="tests/test_foo.py::test_rejects_bad_input",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert findings == []


def test_accepts_pytest_raises_as_function_call(tmp_path: Path) -> None:
    _write_test_file(
        tmp_path,
        "tests/test_call.py",
        """
        import pytest

        def test_rejects_bad_input():
            pytest.raises(ValueError, lambda: (_ for _ in ()).throw(ValueError()))
        """,
    )
    findings = _analyze_at(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref="tests/test_call.py::test_rejects_bad_input",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert findings == []


def test_accepts_assertRaises_unittest(tmp_path: Path) -> None:
    _write_test_file(
        tmp_path,
        "tests/test_u.py",
        """
        import unittest

        class Suite(unittest.TestCase):
            def test_rejects(self):
                with self.assertRaises(ValueError):
                    raise ValueError("nope")
        """,
    )
    findings = _analyze_at(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref="tests/test_u.py::Suite::test_rejects",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert findings == []


def test_missing_test_ref_fires_MISSING() -> None:
    """No test_ref kwarg at all -> TBE1."""
    tree = ast.parse(
        textwrap.dedent(
            """
            @trust_boundary(
                tier=3,
                source="x",
                source_param="data",
                suppresses=("R1",),
                invariant="y",
            )
            def foo(data):
                return data["x"]
            """
        )
    )
    findings = list(
        TESTS_RULE.analyze(tree, Path("decorated.py"), RuleContext(root=Path(".")))
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBE1"
    assert "no test_ref" in findings[0].message


def test_explicit_none_test_ref_fires_MISSING(tmp_path: Path) -> None:
    findings = _analyze_at(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref=None,
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBE1"


def test_unresolvable_file_fires_NOTFOUND(tmp_path: Path) -> None:
    findings = _analyze_at(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref="tests/nonexistent.py::test_x",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBE2"
    assert "does not resolve" in findings[0].message


def test_unresolvable_function_fires_NOTFOUND(tmp_path: Path) -> None:
    _write_test_file(
        tmp_path,
        "tests/test_present.py",
        """
        import pytest

        def test_some_other_thing():
            with pytest.raises(ValueError):
                raise ValueError("x")
        """,
    )
    findings = _analyze_at(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref="tests/test_present.py::test_missing_function",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBE2"


def test_resolved_but_no_raises_fires_WEAK(tmp_path: Path) -> None:
    _write_test_file(
        tmp_path,
        "tests/test_no_raise.py",
        """
        def test_smoke():
            x = 1 + 1
            assert x == 2
        """,
    )
    findings = _analyze_at(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref="tests/test_no_raise.py::test_smoke",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBE3"
    assert "no pytest.raises" in findings[0].message


def test_malformed_nodeid_no_double_colon_fires_NOTFOUND(tmp_path: Path) -> None:
    """A test_ref without ``::`` cannot resolve to a function."""
    findings = _analyze_at(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref="tests/test_no_sep.py",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBE2"


def test_async_function_test_ref_resolved(tmp_path: Path) -> None:
    _write_test_file(
        tmp_path,
        "tests/test_async.py",
        """
        import pytest

        async def test_rejects():
            with pytest.raises(ValueError):
                raise ValueError("x")
        """,
    )
    findings = _analyze_at(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref="tests/test_async.py::test_rejects",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert findings == []


def test_non_literal_kwargs_silently_skipped(tmp_path: Path) -> None:
    """Non-literal kwargs are tier_model territory; this rule does not double-report."""
    findings = _analyze_at(
        """
        ref = "tests/x.py::y"
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref=ref,
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert findings == []
