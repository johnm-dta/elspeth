"""Regression test for the ``elspeth.web.composer.tools`` facade surface.

Every name in ``tools.__all__`` is permanent contract surface this team
commits to defending: it signals "the composer tools package exports this
symbol" to downstream consumers, and breaking it requires a coordinated
release. Dead ``__all__`` entries (declared but never imported by any
external module) impose that maintenance cost for zero benefit.

This test enforces the rule that every facade export has at least one
external consumer. Internal callers within ``elspeth.web.composer.tools.*``
import from sibling submodules directly — they do not (and must not) reach
through the facade. If a name is needed only internally, it does not belong
in the facade or in ``__all__``.

The AST walk performed here is the same one used to prune the original
234%-bloated set down to 62 names (ticket elspeth-b866e0cc8b, RC5.2 review
2026-05-23). If you add a name to ``__all__``, you must also add an external
import for it, or this test will fail the build.

Notes on scope
==============

We treat the following as an "external consumer" of a facade name:

* A statement of the form ``from elspeth.web.composer.tools import X`` in
  any file under ``src/`` or ``tests/`` other than the facade itself.

We do **not** rely on:

* Attribute access via ``tools.X`` after ``import elspeth.web.composer.tools``
  or ``from elspeth.web.composer import tools`` — the prune that motivated
  this test verified that no such reference exists in the current tree.
  Reintroducing one without an explicit ``from`` import would defeat the
  guard, but the operator-facing failure mode (a regression caught by this
  test) is preferable to silently re-bloating ``__all__``.

* String-form ``patch("elspeth.web.composer.tools.X")`` references — the
  prune verified none of these target top-level facade attributes either
  (they target submodule paths like ``...tools.blobs._check_blob_quota``,
  which do not consume the facade).
"""

from __future__ import annotations

import ast
import pathlib

import pytest


def _repo_root() -> pathlib.Path:
    """Locate the repository root from this test file's location.

    Resolves symlinks so that running under a worktree whose ``.venv`` is a
    symlink to the main checkout still walks the correct source tree.
    """

    here = pathlib.Path(__file__).resolve()
    # tests/unit/web/composer/test_tools_facade_surface.py
    #   parents[0] = composer/
    #   parents[1] = web/
    #   parents[2] = unit/
    #   parents[3] = tests/
    #   parents[4] = repo root
    return here.parents[4]


def _names_imported_from_facade() -> set[str]:
    """AST-walk ``src/`` + ``tests/`` for facade imports.

    Returns the set of names appearing in any
    ``from elspeth.web.composer.tools import X`` statement outside the
    facade module itself.
    """

    pkg = "elspeth.web.composer.tools"
    root = _repo_root()
    referenced: set[str] = set()

    for branch in ("src", "tests"):
        branch_root = root / branch
        if not branch_root.is_dir():
            continue
        for py_path in branch_root.rglob("*.py"):
            # The facade re-exports its own names; an import there is not
            # an external consumer.
            rel = py_path.relative_to(root).as_posix()
            if rel.endswith("src/elspeth/web/composer/tools/__init__.py"):
                continue
            try:
                tree = ast.parse(py_path.read_text())
            except SyntaxError:
                # Malformed test fixtures (deliberately broken Python) are
                # not consumers of the facade.
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == pkg:
                    for alias in node.names:
                        referenced.add(alias.name)

    return referenced


def test_tools_all_entries_are_actually_imported_externally() -> None:
    """Every ``tools.__all__`` entry must have at least one external importer.

    Dead ``__all__`` entries are permanent contract surface with no consumer
    benefit. This test fails the build if any future change adds a facade
    export without an accompanying ``from elspeth.web.composer.tools import``
    statement somewhere in ``src/`` or ``tests/``.
    """

    from elspeth.web.composer import tools

    declared = set(tools.__all__)
    referenced = _names_imported_from_facade()
    dead = sorted(declared - referenced)

    assert not dead, (
        "Dead __all__ entries (declared but no external importer): "
        f"{dead}. Either delete from __all__ (and the corresponding "
        "`from ... import` line) or add an external consumer."
    )


def test_tools_all_entries_resolve() -> None:
    """Every ``tools.__all__`` entry must actually be importable.

    A name in ``__all__`` with no backing attribute is a broken export and
    would surface as an ``ImportError`` only when a consumer tried to use
    it. This test catches the typo at build time.
    """

    from elspeth.web.composer import tools

    missing = []
    for name in tools.__all__:
        try:
            getattr(tools, name)
        except AttributeError:
            missing.append(name)

    assert not missing, f"__all__ declares names that are not attributes of the module: {missing}"


def test_tool_result_finalizer_is_intentional_facade_surface() -> None:
    """The batch orchestrator consumes canonical result finalization via the facade."""

    from elspeth.web.composer import tools

    assert callable(tools.finalize_tool_result)
    assert "finalize_tool_result" in tools.__all__


def test_tools_all_is_unique() -> None:
    """``__all__`` must contain no duplicates.

    Ordering is enforced separately by ruff's RUF022 (``__all__`` is not
    sorted) rule, which uses isort's natural-order convention. This test
    only catches the entropy-pump failure mode where the same name appears
    twice across edits — something RUF022 does not catch on its own.
    """

    from elspeth.web.composer import tools

    entries = list(tools.__all__)
    duplicates = sorted(name for name in set(entries) if entries.count(name) > 1)
    assert not duplicates, f"__all__ contains duplicates: {duplicates}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
