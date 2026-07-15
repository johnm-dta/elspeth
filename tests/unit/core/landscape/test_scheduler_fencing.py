"""Scheduler transaction helpers keep fenced and legacy authority explicit."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Protocol, cast, get_type_hints

import pytest
from sqlalchemy import event, insert, select, update
from sqlalchemy.engine import Connection

import elspeth
from elspeth.contracts import RunStatus
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import RunLeadershipLostError
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.scheduler import BarrierJournalRepository, fencing
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import run_coordination_table, run_workers_table, runs_table
from tests.fixtures.landscape import make_landscape_db

RUN_ID = "run-scheduler-fencing"
WORKER_ID = f"worker:{RUN_ID}:leader"
NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)
LEGACY_RECOVERY_HELPER = "legacy_unfenced_recover_expired_leases_write"


class _StrictFencedWrite(Protocol):
    def __call__(
        self,
        engine: Tier1Engine,
        *,
        coordination_token: CoordinationToken,
        now: datetime,
        verb: str,
    ) -> AbstractContextManager[Connection]: ...


class _LegacyUnfencedRecoveryWrite(Protocol):
    def __call__(
        self,
        engine: Tier1Engine,
    ) -> AbstractContextManager[Connection]: ...


def _resolves_legacy_recovery_helper(node: ast.expr, aliases: set[str]) -> bool:
    return (isinstance(node, ast.Name) and node.id in aliases) or (isinstance(node, ast.Attribute) and node.attr == LEGACY_RECOVERY_HELPER)


def _assigned_names(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return {name for element in node.elts for name in _assigned_names(element)}
    return set()


def _legacy_recovery_aliases(tree: ast.AST) -> set[str]:
    aliases = {LEGACY_RECOVERY_HELPER}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            aliases.update(alias.asname or alias.name for alias in node.names if alias.name == LEGACY_RECOVERY_HELPER)

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            targets: set[str] = set()
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets = {name for target in node.targets for name in _assigned_names(target)}
                value = node.value
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                targets = _assigned_names(node.target)
                value = node.value
            if value is not None and _resolves_legacy_recovery_helper(value, aliases):
                new_aliases = targets - aliases
                aliases.update(new_aliases)
                changed = changed or bool(new_aliases)
    return aliases


class _LegacyReferenceVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, *, aliases: set[str]) -> None:
        self._filename = filename
        self._aliases = aliases
        self._function_stack: list[str] = []
        self.references: list[tuple[str, str | None]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self._aliases:
            self.references.append((self._filename, self._function_stack[-1] if self._function_stack else None))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load) and node.attr == LEGACY_RECOVERY_HELPER:
            self.references.append((self._filename, self._function_stack[-1] if self._function_stack else None))
        self.generic_visit(node)


def _legacy_recovery_references(source: str, *, filename: str = "<mutation>") -> list[tuple[str, str | None]]:
    tree = ast.parse(source)
    visitor = _LegacyReferenceVisitor(filename, aliases=_legacy_recovery_aliases(tree))
    visitor.visit(tree)
    return visitor.references


def _strict_fenced_write() -> _StrictFencedWrite:
    helper = getattr(fencing, "fenced_write", None)
    assert helper is not None, "scheduler fencing must expose a strict fenced_write helper"
    return cast(_StrictFencedWrite, helper)


def _legacy_unfenced_recovery_write() -> _LegacyUnfencedRecoveryWrite:
    helper = getattr(fencing, LEGACY_RECOVERY_HELPER, None)
    assert helper is not None, "scheduler fencing must expose a recovery-specific legacy unfenced helper"
    return cast(_LegacyUnfencedRecoveryWrite, helper)


def _resolved_parameter_annotation(method: FunctionType, parameter_name: str) -> object:
    """Resolve one trusted source annotation without evaluating its peers."""
    annotation_probe = type(
        "_AnnotationProbe",
        (),
        {"__annotations__": {"value": method.__annotations__[parameter_name]}},
    )
    return get_type_hints(
        annotation_probe,
        globalns=method.__globals__,
        localns={"datetime": datetime, "Mapping": Mapping},
    )["value"]


def _assert_required_coordination_parameter(method: FunctionType) -> None:
    parameter = inspect.signature(method).parameters["coordination_token"]
    assert parameter.default is inspect.Parameter.empty
    assert _resolved_parameter_annotation(method, "coordination_token") is CoordinationToken


def _seed_leader() -> tuple[LandscapeDB, CoordinationToken]:
    db = make_landscape_db()
    with db.engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=NOW,
                config_hash="cfg",
                settings_json="{}",
                canonical_version="v1",
                status=RunStatus.RUNNING.value,
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
    token = RunCoordinationRepository(db.engine).register_run_leader(
        run_id=RUN_ID,
        worker_id=WORKER_ID,
        now=NOW,
        window_seconds=80.0,
    )
    return db, token


def test_ambiguous_optional_authority_helper_is_removed() -> None:
    assert not hasattr(fencing, "fenced_or_plain_write")


def test_strict_helper_rejects_runtime_none_before_transaction() -> None:
    db = make_landscape_db()
    transactions: list[object] = []

    def record_begin(conn: object) -> None:
        transactions.append(conn)

    event.listen(db.engine, "begin", record_begin)
    try:
        with pytest.raises(TypeError, match="coordination_token"):
            helper = _strict_fenced_write()
            helper(
                db.engine,
                coordination_token=None,  # type: ignore[arg-type]  # runtime trust-boundary regression
                now=NOW,
                verb="strict_probe",
            )
    finally:
        event.remove(db.engine, "begin", record_begin)

    assert transactions == [], "missing authority must be refused before opening a transaction"
    with db.engine.connect() as conn:
        assert conn.execute(select(runs_table.c.run_id)).all() == []


def test_strict_helper_type_contract_forbids_optional_authority() -> None:
    helper = _strict_fenced_write()
    assert get_type_hints(helper)["coordination_token"] is CoordinationToken


@pytest.mark.parametrize(
    "repository_type",
    [BarrierJournalRepository, TokenSchedulerRepository],
    ids=["journal", "facade"],
)
@pytest.mark.parametrize(
    "method_name",
    ["mark_blocked_barrier_terminal", "mark_blocked_barrier_pending_sink_many"],
    ids=["terminal", "pending-sink"],
)
def test_barrier_wrapper_type_contract_requires_authority(
    repository_type: type[BarrierJournalRepository] | type[TokenSchedulerRepository],
    method_name: str,
) -> None:
    method = cast(FunctionType, getattr(repository_type, method_name))
    _assert_required_coordination_parameter(method)


@pytest.mark.parametrize(
    ("annotation", "binding"),
    [
        pytest.param("AuthorityAlias", CoordinationToken, id="alias"),
        pytest.param(
            "coordination.CoordinationToken",
            SimpleNamespace(CoordinationToken=CoordinationToken),
            id="qualified",
        ),
    ],
)
def test_required_coordination_parameter_accepts_equivalent_annotations(
    annotation: str,
    binding: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def probe(*, coordination_token: object) -> None:
        del coordination_token

    probe.__annotations__["coordination_token"] = annotation
    monkeypatch.setitem(probe.__globals__, annotation.partition(".")[0], binding)

    _assert_required_coordination_parameter(cast(FunctionType, probe))


def test_required_coordination_parameter_rejects_wrong_runtime_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    def probe(*, coordination_token: object) -> None:
        del coordination_token

    probe.__annotations__["coordination_token"] = "WrongAuthority"
    monkeypatch.setitem(probe.__globals__, "WrongAuthority", str)

    with pytest.raises(AssertionError):
        _assert_required_coordination_parameter(cast(FunctionType, probe))


def test_strict_helper_accepts_current_token_and_commits() -> None:
    db, token = _seed_leader()
    helper = _strict_fenced_write()
    probe_worker_id = f"worker:{RUN_ID}:probe"

    with helper(
        db.engine,
        coordination_token=token,
        now=NOW,
        verb="strict_probe",
    ) as conn:
        conn.execute(
            insert(run_workers_table).values(
                worker_id=probe_worker_id,
                run_id=RUN_ID,
                role="follower",
                status="active",
                registered_at=NOW,
                heartbeat_expires_at=NOW,
            )
        )

    with db.engine.connect() as conn:
        assert (
            conn.execute(select(run_workers_table.c.worker_id).where(run_workers_table.c.worker_id == probe_worker_id)).scalar_one()
            == probe_worker_id
        )


def test_strict_helper_refuses_stale_token_without_payload_mutation() -> None:
    db, token = _seed_leader()
    with db.engine.begin() as conn:
        conn.execute(
            update(run_coordination_table).where(run_coordination_table.c.run_id == RUN_ID).values(leader_epoch=token.leader_epoch + 1)
        )
    helper = _strict_fenced_write()
    probe_worker_id = f"worker:{RUN_ID}:stale-probe"

    with (
        pytest.raises(RunLeadershipLostError),
        helper(
            db.engine,
            coordination_token=token,
            now=NOW,
            verb="strict_probe",
        ) as conn,
    ):
        conn.execute(  # pragma: no cover - the fence refuses before the helper body
            insert(run_workers_table).values(
                worker_id=probe_worker_id,
                run_id=RUN_ID,
                role="follower",
                status="active",
                registered_at=NOW,
                heartbeat_expires_at=NOW,
            )
        )

    with db.engine.connect() as conn:
        assert (
            conn.execute(select(run_workers_table.c.worker_id).where(run_workers_table.c.worker_id == probe_worker_id)).one_or_none()
            is None
        )


def test_recovery_specific_legacy_helper_is_plain_without_a_verb_selector() -> None:
    db = make_landscape_db()
    helper = _legacy_unfenced_recovery_write()
    assert tuple(inspect.signature(helper).parameters) == ("engine",)

    with helper(db.engine) as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=NOW,
                config_hash="cfg",
                settings_json="{}",
                canonical_version="v1",
                status=RunStatus.RUNNING.value,
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )

    with db.engine.connect() as conn:
        assert conn.execute(select(runs_table.c.run_id)).scalar_one() == RUN_ID


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            f"def probe():\n    {LEGACY_RECOVERY_HELPER}(engine)\n",
            id="bare-call",
        ),
        pytest.param(
            f"def probe():\n    fencing.{LEGACY_RECOVERY_HELPER}(engine)\n",
            id="module-qualified-call",
        ),
        pytest.param(
            f"from elspeth.core.landscape.scheduler.fencing import {LEGACY_RECOVERY_HELPER} as plain_write\n"
            "def probe():\n    plain_write(engine)\n",
            id="import-aliased-call",
        ),
        pytest.param(
            f"import elspeth.core.landscape.scheduler.fencing as fence\ndef probe():\n    fence.{LEGACY_RECOVERY_HELPER}(engine)\n",
            id="module-import-aliased-call",
        ),
        pytest.param(
            f"def probe():\n    plain_write = {LEGACY_RECOVERY_HELPER}\n    plain_write(engine)\n",
            id="assigned-bare-alias-call",
        ),
        pytest.param(
            f"def probe():\n    plain_write = fencing.{LEGACY_RECOVERY_HELPER}\n    plain_write(engine)\n",
            id="assigned-qualified-alias-call",
        ),
        pytest.param(
            f"def probe():\n    registry = ({LEGACY_RECOVERY_HELPER},)\n",
            id="bare-value-reference",
        ),
        pytest.param(
            f"def probe():\n    registry = (fencing.{LEGACY_RECOVERY_HELPER},)\n",
            id="qualified-value-reference",
        ),
    ],
)
def test_legacy_helper_source_contract_detects_reference_mutations(source: str) -> None:
    references = _legacy_recovery_references(source)
    assert references != []
    assert set(references) == {("<mutation>", "probe")}


def test_legacy_helper_reference_is_isolated_to_lease_recovery_across_package() -> None:
    package_dir = Path(elspeth.__file__).parent
    references: list[tuple[str, str | None]] = []

    for path in package_dir.rglob("*.py"):
        references.extend(
            _legacy_recovery_references(
                path.read_text(),
                filename=str(path.relative_to(package_dir)),
            )
        )

    assert references == [("core/landscape/scheduler/leases.py", "recover_expired_leases")]


def test_scheduler_sources_have_no_optional_authority_transaction_selector() -> None:
    scheduler_dir = Path(fencing.__file__).parent
    offenders = sorted(path.name for path in scheduler_dir.glob("*.py") if "fenced_or_plain_write" in path.read_text())
    assert offenders == []


def test_generic_legacy_unfenced_write_selector_is_removed() -> None:
    assert not hasattr(fencing, "legacy_unfenced_write")
