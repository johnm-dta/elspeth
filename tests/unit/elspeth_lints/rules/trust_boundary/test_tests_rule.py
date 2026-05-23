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
    return list(TESTS_RULE.analyze(tree, Path("decorated.py"), RuleContext(root=repo_root)))


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
    findings = list(TESTS_RULE.analyze(tree, Path("decorated.py"), RuleContext(root=Path("."))))
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


def test_non_literal_kwargs_fires_NONLITERAL(tmp_path: Path) -> None:
    """Non-literal kwargs are self-enforced (TBE4), NOT silently deferred.

    Before the C6-4 honesty-gate hardening (epic elspeth-2ed3bb0f7d,
    ticket elspeth-1f4634235a) this rule continued silently when any
    kwarg wasn't a literal, deferring all reporting to
    ``trust_tier.tier_model``. That created a cross-rule bypass —
    suppressing tier_model on a file granted tests-honesty immunity.
    The rule now self-enforces; the redundant finding when tier_model
    is also active is deliberate.
    """
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
    assert len(findings) == 1
    assert findings[0].rule_id == "TBE4"
    assert "test_ref" in findings[0].message
    assert "not a static literal" in findings[0].message


def test_double_star_unpacking_fires_NONLITERAL(tmp_path: Path) -> None:
    """``@trust_boundary(**META)`` is also non-literal — covered by TBE4."""
    findings = _analyze_at(
        """
        META = {"tier": 3, "test_ref": "tests/x.py::y"}
        @trust_boundary(**META)
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBE4"
    assert "unpacking" in findings[0].message


# =============================================================================
# Regression: C6-3 — scope leak in _has_raising_assertion
# =============================================================================
#
# Before the fix, ``_walk_statements`` used :func:`ast.walk` to descend
# through each top-level statement of the named test function's body.
# ``ast.walk`` descends into nested function bodies, so a test that
# delegated its raising-assertion to a nested helper (or to a class-body
# method) had its ``pytest.raises(...)`` call seen by the outer test's
# walk — TBE3 (WEAK) passed silently even though the honesty contract
# (assertion must be visible IN the named test, not buried one
# indirection away) was violated.
#
# The fix replaces the ``ast.walk``-based ``_walk_statements`` with the
# scope-respecting :func:`iter_own_scope` from ast_walker, which
# short-circuits at nested FunctionDef / AsyncFunctionDef / Lambda /
# ClassDef boundaries. The nested helper's ``pytest.raises`` no longer
# satisfies the outer test's contract; TBE3 now fires as the module
# docstring promised.
# =============================================================================


def test_raising_assertion_in_nested_helper_fires_WEAK(tmp_path: Path) -> None:
    """Test delegates raising to nested helper — TBE3 (WEAK) must fire.

    Before the fix: ``ast.walk`` descended into ``_helper``'s body,
    saw the ``pytest.raises(...)`` context manager, and returned True
    from ``_has_raising_assertion``. The decorated function passed
    the TBE3 check silently — a direct violation of the module's
    contract that the raising assertion must be visible in the named
    test, not in a helper.

    After the fix: ``iter_own_scope`` short-circuits at the inner
    FunctionDef boundary; the nested helper's body is never visited,
    so the outer test's body has no raising assertion and TBE3 fires.
    """
    _write_test_file(
        tmp_path,
        "tests/test_helper_indirect.py",
        """
        import pytest

        def test_delegates_to_helper():
            # The visible body has no pytest.raises. The honesty
            # contract requires the assertion to live HERE, not in
            # ``_helper`` — a helper indirection silently changes
            # what the decorator claims is being asserted, and the
            # gate's promise is to refuse that.
            _helper()

        def _helper():
            with pytest.raises(ValueError):
                raise ValueError("hidden")
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
            test_ref="tests/test_helper_indirect.py::test_delegates_to_helper",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert len(findings) == 1, (
        f"Expected exactly one TBE3 (WEAK) finding; got {len(findings)}: {[(f.rule_id, f.message[:80]) for f in findings]}"
    )
    assert findings[0].rule_id == "TBE3"
    assert "no pytest.raises" in findings[0].message


def test_raising_assertion_in_nested_class_body_fires_WEAK(tmp_path: Path) -> None:
    """Class-body raising-assertion does not satisfy the outer test's contract.

    A nested class defined inside the test (rare but legal) whose body
    contains a ``pytest.raises(...)`` call must NOT satisfy the
    outer test's TBE3 contract. The class body is a different
    namespace; the assertion is not visible in the test's own body.

    This case is the symmetric companion to the nested-function test
    above. The :func:`iter_own_scope` walker short-circuits at
    ClassDef for the same reason it short-circuits at FunctionDef:
    nested-scope reads do not satisfy outer-scope contracts.
    """
    _write_test_file(
        tmp_path,
        "tests/test_class_body_indirect.py",
        """
        import pytest

        def test_class_body_raises():
            # The class body executes when the class statement runs,
            # but the raising assertion is inside the class's own
            # namespace, not in the test's body. TBE3 should still
            # fire.
            class _Inner:
                with pytest.raises(ValueError):
                    raise ValueError("hidden in class body")
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
            test_ref="tests/test_class_body_indirect.py::test_class_body_raises",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert len(findings) == 1, (
        f"Expected exactly one TBE3 (WEAK) finding; got {len(findings)}: {[(f.rule_id, f.message[:80]) for f in findings]}"
    )
    assert findings[0].rule_id == "TBE3"


def test_raising_assertion_directly_in_test_body_still_passes(tmp_path: Path) -> None:
    """Sanity check: a test that DOES have ``pytest.raises`` directly in
    its body still passes TBE3 after the C6-3 fix.

    This guards against the symmetric over-correction failure mode:
    we want the scope-respecting walker to stop at nested scopes, NOT
    to stop walking the outer body's own statements.
    """
    _write_test_file(
        tmp_path,
        "tests/test_direct.py",
        """
        import pytest

        def test_direct_raise():
            # Visible in the test's own body. TBE3 must NOT fire.
            with pytest.raises(ValueError):
                raise ValueError("direct")
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
            test_ref="tests/test_direct.py::test_direct_raise",
        )
        def foo(data):
            return data["x"]
        """,
        repo_root=tmp_path,
    )
    assert findings == [], f"Direct in-body pytest.raises must pass TBE3; got: {[(f.rule_id, f.message[:80]) for f in findings]}"
