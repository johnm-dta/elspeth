"""Tests for composer elspeth-lints rules."""

from __future__ import annotations

import ast
from pathlib import Path

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.composer.catch_order import RULE as CATCH_ORDER_RULE
from elspeth_lints.rules.composer.exception_channel import RULE as EXCEPTION_CHANNEL_RULE


def test_exception_channel_reports_bare_value_error(tmp_path: Path) -> None:
    target = tmp_path / "web" / "composer" / "tools.py"
    target.parent.mkdir(parents=True)
    source = "def f():\n    raise ValueError('bad')\n"
    tree = ast.parse(source, filename=str(target))

    findings = list(EXCEPTION_CHANNEL_RULE.analyze(tree, target, RuleContext(root=tmp_path)))

    assert [finding.rule_id for finding in findings] == ["CEC1"]
    assert findings[0].file_path == "web/composer/tools.py"
    assert findings[0].line == 2
    assert "ValueError" in findings[0].message


def test_catch_order_reports_broad_before_narrow(tmp_path: Path) -> None:
    target = tmp_path / "web" / "sessions" / "routes.py"
    target.parent.mkdir(parents=True)
    source = (
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except ComposerServiceError as exc:\n"
        "        pass\n"
        "    except ComposerPluginCrashError as crash:\n"
        "        pass\n"
    )
    tree = ast.parse(source, filename=str(target))

    findings = list(CATCH_ORDER_RULE.analyze(tree, target, RuleContext(root=tmp_path)))

    assert [finding.rule_id for finding in findings] == ["CCO1"]
    assert findings[0].file_path == "web/sessions/routes.py"
    assert findings[0].line == 6
    assert "ComposerPluginCrashError" in findings[0].message
