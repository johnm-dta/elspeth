"""PR-diff surfacing for newly-added ``@trust_boundary`` decorators."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from elspeth_lints.core.cli import main
from elspeth_lints.core.trust_boundary_diff import (
    find_new_trust_boundary_decorators,
    render_trust_boundary_diff_summary,
)


def _init_git_fixture(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    src = tmp_path / "src" / "elspeth"
    src.mkdir(parents=True)
    return src


def _commit(repo_root: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(repo_root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo_root), "commit", "-q", "-m", message], check=True)
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_diff_reports_new_trust_boundary_decorator(tmp_path: Path) -> None:
    src = _init_git_fixture(tmp_path)
    target = src / "handler.py"
    target.write_text("def handler(arguments):\n    return arguments.get('x')\n", encoding="utf-8")
    baseline = _commit(tmp_path, "baseline")

    target.write_text(
        textwrap.dedent("""\
        from elspeth.contracts import trust_boundary

        @trust_boundary(
            tier=3,
            source="LLM tool args",
            source_param="arguments",
            suppresses=("R1",),
            invariant="raises on bad args",
            test_ref="tests/test_handler.py::test_rejects_bad_args",
            test_fingerprint="abc123",
        )
        def handler(arguments):
            return arguments.get("x")
    """),
        encoding="utf-8",
    )
    _commit(tmp_path, "add trust boundary")

    report = find_new_trust_boundary_decorators(
        root=src,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    assert len(report.new_decorators) == 1
    decorator = report.new_decorators[0]
    assert decorator.source_file == "src/elspeth/handler.py"
    assert decorator.symbol == "handler"
    assert decorator.source_param == "arguments"
    assert decorator.suppresses == ("R1",)
    assert decorator.source == "LLM tool args"
    assert decorator.test_ref == "tests/test_handler.py::test_rejects_bad_args"


def test_diff_grandfathers_existing_trust_boundary_decorator(tmp_path: Path) -> None:
    src = _init_git_fixture(tmp_path)
    (src / "handler.py").write_text(
        textwrap.dedent("""\
        from elspeth.contracts import trust_boundary

        @trust_boundary(
            tier=3,
            source="LLM tool args",
            source_param="arguments",
            suppresses=("R1",),
            invariant="raises on bad args",
            test_ref="tests/test_handler.py::test_rejects_bad_args",
            test_fingerprint="abc123",
        )
        def handler(arguments):
            return arguments.get("x")
    """),
        encoding="utf-8",
    )
    baseline = _commit(tmp_path, "baseline")

    (src / "handler.py").write_text(
        textwrap.dedent("""\
        from elspeth.contracts import trust_boundary

        @trust_boundary(
            tier=3,
            source="LLM tool args",
            source_param="arguments",
            suppresses=("R1",),
            invariant="raises on bad args",
            test_ref="tests/test_handler.py::test_rejects_bad_args",
            test_fingerprint="abc123",
        )
        def handler(arguments):
            value = arguments.get("x")
            return value
    """),
        encoding="utf-8",
    )
    _commit(tmp_path, "body-only change")

    report = find_new_trust_boundary_decorators(
        root=src,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    assert report.new_decorators == ()


def test_summary_names_new_decorators() -> None:
    from elspeth_lints.core.trust_boundary_diff import TrustBoundaryDecoratorRecord, TrustBoundaryDiffReport

    report = TrustBoundaryDiffReport(
        baseline_ref="abc123",
        root="src/elspeth",
        new_decorators=(
            TrustBoundaryDecoratorRecord(
                source_file="src/elspeth/handler.py",
                line=3,
                symbol="Handler.run",
                source_param="arguments",
                suppresses=("R1", "R5"),
                source="LLM tool args",
                test_ref="tests/test_handler.py::test_rejects_bad_args",
                metadata_readable=True,
                identity_hash="deadbeef",
            ),
        ),
    )

    summary = render_trust_boundary_diff_summary(report)

    assert "New @trust_boundary decorators: 1" in summary
    assert "src/elspeth/handler.py:3 Handler.run" in summary
    assert "source_param=arguments" in summary
    assert "suppresses=R1,R5" in summary


def test_cli_summarizes_new_trust_boundary_decorator(tmp_path: Path, capsys) -> None:
    src = _init_git_fixture(tmp_path)
    target = src / "handler.py"
    target.write_text("def handler(arguments):\n    return arguments.get('x')\n", encoding="utf-8")
    baseline = _commit(tmp_path, "baseline")

    target.write_text(
        textwrap.dedent("""\
        from elspeth.contracts import trust_boundary

        @trust_boundary(
            tier=3,
            source="LLM tool args",
            source_param="arguments",
            suppresses=("R1",),
            invariant="raises on bad args",
            test_ref="tests/test_handler.py::test_rejects_bad_args",
            test_fingerprint="abc123",
        )
        def handler(arguments):
            return arguments.get("x")
    """),
        encoding="utf-8",
    )
    _commit(tmp_path, "add trust boundary")

    status = main(
        [
            "check-trust-boundary-diff",
            "--baseline-ref",
            baseline,
            "--root",
            "src/elspeth",
            "--repo-root",
            str(tmp_path),
        ]
    )

    out = capsys.readouterr().out
    assert status == 0
    assert "New @trust_boundary decorators: 1" in out
    assert "src/elspeth/handler.py" in out
