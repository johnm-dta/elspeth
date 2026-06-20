"""Tests for the session-engine factory contract-invariant rule."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext
from elspeth_lints.rules.contract_invariants.session_engine_factory import RULE as SESSION_ENGINE_FACTORY_RULE
from elspeth_lints.rules.contract_invariants.session_engine_factory.rule import find_session_engine_factory_findings


def test_session_engine_factory_reports_direct_create_engine_under_sessions() -> None:
    findings = _findings(
        "src/elspeth/web/sessions/repository.py",
        """
        from sqlalchemy import create_engine

        def build():
            return create_engine("sqlite:///sessions.db")
        """,
    )

    assert [finding.rule_id for finding in findings] == ["contract_invariants.session_engine_factory"]
    assert findings[0].line == 5
    assert "session-owned path" in findings[0].message


def test_session_engine_factory_reports_aliased_module_create_engine_under_sessions() -> None:
    findings = _findings(
        "src/elspeth/web/sessions/repository.py",
        """
        import sqlalchemy as sa

        def build():
            return sa.create_engine("sqlite:///sessions.db")
        """,
    )

    assert [finding.rule_id for finding in findings] == ["contract_invariants.session_engine_factory"]


def test_session_engine_factory_reports_web_session_db_url() -> None:
    findings = _findings(
        "src/elspeth/web/app.py",
        """
        from sqlalchemy import create_engine as make_engine

        def build(session_db_url: str):
            return make_engine(session_db_url)
        """,
    )

    assert [finding.rule_id for finding in findings] == ["contract_invariants.session_engine_factory"]
    assert "session database URL" in findings[0].message


def test_session_engine_factory_accepts_factory_implementation() -> None:
    findings = _findings(
        "src/elspeth/web/sessions/engine.py",
        """
        from sqlalchemy import create_engine

        def create_session_engine(url: str):
            return create_engine(url)
        """,
    )

    assert findings == []


def test_session_engine_factory_accepts_unrelated_web_engine() -> None:
    findings = _findings(
        "src/elspeth/web/app.py",
        """
        from sqlalchemy import create_engine

        def build(cache_db_url: str):
            return create_engine(cache_db_url)
        """,
    )

    assert findings == []


def test_session_engine_factory_applies_allowlist_override(tmp_path: Path) -> None:
    target = tmp_path / "src" / "elspeth" / "web" / "sessions" / "legacy_test_helper.py"
    target.parent.mkdir(parents=True)
    source = textwrap.dedent(
        """
        from sqlalchemy import create_engine

        def build():
            return create_engine("sqlite:///sessions.db")
        """
    )
    target.write_text(source, encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    context = RuleContext(root=tmp_path)
    finding = next(iter(SESSION_ENGINE_FACTORY_RULE.analyze(tree, target, context)))

    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    (allowlist_dir / "tests.yaml").write_text(
        f"""
allow_hits:
  - key: {finding.canonical_key()}
    owner: test
    reason: Negative fixture intentionally bypasses the sessions factory.
    safety: Synthetic unit test only.
""",
        encoding="utf-8",
    )

    filtered = list(
        SESSION_ENGINE_FACTORY_RULE.analyze(
            tree,
            target,
            RuleContext(root=tmp_path, allowlist_dir_override=allowlist_dir),
        )
    )

    assert filtered == []


def _findings(file_path: str, source: str) -> list[Finding]:
    dedented = textwrap.dedent(source)
    tree = ast.parse(dedented, filename=file_path)
    return find_session_engine_factory_findings(tree, file_path)
