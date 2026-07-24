"""Import-aware regression guard for direct web-layer LandscapeDB opens."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True, slots=True)
class _DirectCall:
    path: str
    line: int
    kind: str


@dataclass(frozen=True, slots=True)
class _Offender:
    path: str
    line: int
    reason: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.reason}"


@dataclass(frozen=True, slots=True)
class _Collection:
    direct_calls: tuple[_DirectCall, ...]
    offenders: tuple[_Offender, ...]


def _is_landscape_db_reference(node: ast.expr, *, imported_aliases: frozenset[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "LandscapeDB" or node.id in imported_aliases
    return isinstance(node, ast.Attribute) and node.attr == "LandscapeDB"


def _is_stored_from_url(node: ast.expr, *, imported_aliases: frozenset[str]) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "from_url"
        and _is_landscape_db_reference(node.value, imported_aliases=imported_aliases)
    )


def _collect_landscape_accesses(source: str, *, path: str) -> _Collection:
    tree = ast.parse(source, filename=path)
    imported_aliases = frozenset(
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name == "LandscapeDB"
    )
    direct_calls: list[_DirectCall] = []
    offenders: list[_Offender] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value = node.value
            if _is_stored_from_url(value, imported_aliases=imported_aliases):
                offenders.append(_Offender(path, node.lineno, "stored-from-url-alias"))

        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if _is_landscape_db_reference(function, imported_aliases=imported_aliases):
            direct_calls.append(_DirectCall(path, node.lineno, "constructor"))
            offenders.append(_Offender(path, node.lineno, "direct-constructor"))
            continue
        if not isinstance(function, ast.Attribute) or not _is_landscape_db_reference(
            function.value,
            imported_aliases=imported_aliases,
        ):
            continue
        if function.attr == "in_memory":
            direct_calls.append(_DirectCall(path, node.lineno, "in_memory"))
            offenders.append(_Offender(path, node.lineno, "in-memory-constructor"))
            continue
        if function.attr != "from_url":
            continue

        direct_calls.append(_DirectCall(path, node.lineno, "from_url"))
        create_tables = next((keyword.value for keyword in node.keywords if keyword.arg == "create_tables"), None)
        if create_tables is None:
            offenders.append(_Offender(path, node.lineno, "missing-create-tables-policy"))
        elif isinstance(create_tables, ast.Constant) and create_tables.value is True:
            offenders.append(_Offender(path, node.lineno, "literal-create-tables-true"))

    return _Collection(
        direct_calls=tuple(sorted(direct_calls, key=lambda site: (site.line, site.kind))),
        offenders=tuple(sorted(offenders, key=lambda offender: (offender.line, offender.reason))),
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("LandscapeDB('sqlite:///x.db')", ("synthetic.py:1:direct-constructor",)),
        (
            "from elspeth.core.landscape import LandscapeDB\nLandscapeDB.from_url(url)",
            ("synthetic.py:2:missing-create-tables-policy",),
        ),
        (
            "from elspeth.core.landscape.database import LandscapeDB as LDB\nLDB.in_memory()",
            ("synthetic.py:2:in-memory-constructor",),
        ),
        (
            "import elspeth.core.landscape as landscape\nlandscape.LandscapeDB.from_url(url, create_tables=True)",
            ("synthetic.py:2:literal-create-tables-true",),
        ),
        (
            "import elspeth.core.landscape\nelspeth.core.landscape.LandscapeDB.from_url(url)",
            ("synthetic.py:2:missing-create-tables-policy",),
        ),
        (
            "import elspeth.core.landscape.database as database\ndatabase.LandscapeDB(url)",
            ("synthetic.py:2:direct-constructor",),
        ),
        (
            "import elspeth.core.landscape.database\nelspeth.core.landscape.database.LandscapeDB.in_memory()",
            ("synthetic.py:2:in-memory-constructor",),
        ),
        (
            "from elspeth.core.landscape import database as database\ndatabase.LandscapeDB.from_url(url)",
            ("synthetic.py:2:missing-create-tables-policy",),
        ),
        (
            "from elspeth.web.sessions.routes._helpers import LandscapeDB as LDB\nLDB.from_url(url)",
            ("synthetic.py:2:missing-create-tables-policy",),
        ),
        (
            "import elspeth.web.sessions.routes._helpers as helpers\nhelpers.LandscapeDB.from_url(url, create_tables=True)",
            ("synthetic.py:2:literal-create-tables-true",),
        ),
        (
            "import unfamiliar\nunfamiliar.LandscapeDB(url)",
            ("synthetic.py:2:direct-constructor",),
        ),
        (
            "open_database = LandscapeDB.from_url\nopen_database(url, create_tables=False)",
            ("synthetic.py:1:stored-from-url-alias",),
        ),
        (
            "holder.open_database = LandscapeDB.from_url",
            ("synthetic.py:1:stored-from-url-alias",),
        ),
    ],
)
def test_synthetic_unsafe_forms_are_rejected(source: str, expected: tuple[str, ...]) -> None:
    result = _collect_landscape_accesses(source, path="synthetic.py")

    assert tuple(offender.render() for offender in result.offenders) == expected


@pytest.mark.parametrize(
    "source",
    [
        "LandscapeDB.from_url(url, create_tables=False)",
        "from anywhere import LandscapeDB as LDB\nLDB.from_url(url, create_tables=policy)",
        "import elspeth.core.landscape as landscape\nlandscape.LandscapeDB.from_url(url, create_tables=allowed)",
        "import elspeth.core.landscape\nelspeth.core.landscape.LandscapeDB.from_url(url, create_tables=False)",
        "import elspeth.core.landscape.database as database\ndatabase.LandscapeDB.from_url(url, create_tables=self.create_tables)",
        "from elspeth.core.landscape import database as database\ndatabase.LandscapeDB.from_url(url, create_tables=create_tables)",
        "import elspeth.web.sessions.routes._helpers as helpers\nhelpers.LandscapeDB.from_url(url, create_tables=False)",
    ],
)
def test_synthetic_explicit_policies_are_accepted(source: str) -> None:
    result = _collect_landscape_accesses(source, path="synthetic.py")

    assert result.offenders == ()
    assert len(result.direct_calls) == 1
    assert result.direct_calls[0].kind == "from_url"


def test_real_web_tree_has_only_the_reviewed_direct_call_map() -> None:
    root = Path(__file__).resolve().parents[3]
    web_root = root / "src" / "elspeth" / "web"
    results = [
        _collect_landscape_accesses(path.read_text(encoding="utf-8"), path=path.relative_to(root).as_posix())
        for path in sorted(web_root.rglob("*.py"))
    ]
    offenders = tuple(offender.render() for result in results for offender in result.offenders)
    direct_map = tuple(sorted((site.path, site.kind) for result in results for site in result.direct_calls))

    assert offenders == ()
    assert direct_map == (
        ("src/elspeth/web/app.py", "from_url"),
        ("src/elspeth/web/auth/audit.py", "from_url"),
        ("src/elspeth/web/aws_ecs_acceptance.py", "from_url"),
        ("src/elspeth/web/aws_ecs_acceptance.py", "from_url"),
        ("src/elspeth/web/aws_ecs_acceptance.py", "from_url"),
        ("src/elspeth/web/aws_ecs_acceptance.py", "from_url"),
        ("src/elspeth/web/aws_ecs_acceptance.py", "from_url"),
        ("src/elspeth/web/execution/accounting.py", "from_url"),
        ("src/elspeth/web/execution/diagnostics.py", "from_url"),
        ("src/elspeth/web/execution/discard_summary.py", "from_url"),
        ("src/elspeth/web/execution/outputs.py", "from_url"),
        ("src/elspeth/web/landscape_access.py", "from_url"),
        ("src/elspeth/web/sessions/routes/runs.py", "from_url"),
    )
