"""Tests for composer elspeth-lints rules."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext
from elspeth_lints.rules.composer.catch_order import RULE as CATCH_ORDER_RULE
from elspeth_lints.rules.composer.catch_order.rule import _SUBCLASS_TO_SUPERCLASSES
from elspeth_lints.rules.composer.exception_channel import RULE as EXCEPTION_CHANNEL_RULE


def test_exception_channel_accepts_tool_argument_error(tmp_path: Path) -> None:
    source = (
        "from elspeth.web.composer.protocol import ToolArgumentError\n"
        "def f(x):\n"
        "    if not isinstance(x, str):\n"
        "        raise ToolArgumentError(argument='x', expected='a string', actual_type=type(x).__name__)\n"
    )
    findings = _exception_channel_findings(tmp_path, source)

    assert findings == []


def test_exception_channel_reports_bare_type_error(tmp_path: Path) -> None:
    source = "def f(x):\n    if not isinstance(x, str):\n        raise TypeError('bad')\n"

    findings = _exception_channel_findings(tmp_path, source)

    assert [finding.rule_id for finding in findings] == ["CEC1"]
    assert "TypeError" in findings[0].message


def test_exception_channel_reports_bare_value_error(tmp_path: Path) -> None:
    source = "def f():\n    raise ValueError('bad')\n"

    findings = _exception_channel_findings(tmp_path, source)

    assert [finding.rule_id for finding in findings] == ["CEC1"]
    assert findings[0].file_path == "web/composer/tools.py"
    assert findings[0].line == 2
    assert "ValueError" in findings[0].message


def test_exception_channel_reports_explicit_raise_inside_try_except(tmp_path: Path) -> None:
    source = (
        "def _failure_result(state, msg): return msg\n"
        "def f(x, state):\n"
        "    try:\n"
        "        raise ValueError('bad')\n"
        "    except ValueError as exc:\n"
        "        return _failure_result(state, str(exc))\n"
    )

    findings = _exception_channel_findings(tmp_path, source)

    assert [finding.rule_id for finding in findings] == ["CEC1"]
    assert findings[0].line == 4


def test_exception_channel_ignores_implicit_raise_from_coercion(tmp_path: Path) -> None:
    source = (
        "def _failure_result(state, msg): return msg\n"
        "def f(x, state):\n"
        "    try:\n"
        "        int(x)\n"
        "    except ValueError as exc:\n"
        "        return _failure_result(state, str(exc))\n"
    )

    findings = _exception_channel_findings(tmp_path, source)

    assert findings == []


def test_catch_order_accepts_narrow_before_broad(tmp_path: Path) -> None:
    source = (
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except ComposerPluginCrashError as crash:\n"
        "        pass\n"
        "    except ComposerServiceError as exc:\n"
        "        pass\n"
    )

    findings = _catch_order_findings(tmp_path, source)

    assert findings == []


def test_catch_order_reports_broad_before_narrow(tmp_path: Path) -> None:
    source = (
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except ComposerServiceError as exc:\n"
        "        pass\n"
        "    except ComposerPluginCrashError as crash:\n"
        "        pass\n"
    )

    findings = _catch_order_findings(tmp_path, source)

    assert [finding.rule_id for finding in findings] == ["CCO1"]
    assert findings[0].file_path == "web/sessions/routes.py"
    assert findings[0].line == 6
    assert "ComposerPluginCrashError" in findings[0].message


def test_catch_order_reports_tuple_handler_shadowing_subclass(tmp_path: Path) -> None:
    source = (
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except (RuntimeError, ComposerServiceError) as exc:\n"
        "        pass\n"
        "    except ComposerPluginCrashError as crash:\n"
        "        pass\n"
    )

    findings = _catch_order_findings(tmp_path, source)

    assert [finding.rule_id for finding in findings] == ["CCO1"]


def test_catch_order_reports_attribute_handler_shadowing_subclass(tmp_path: Path) -> None:
    source = (
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except protocol.ComposerServiceError as exc:\n"
        "        pass\n"
        "    except ComposerPluginCrashError as crash:\n"
        "        pass\n"
    )

    findings = _catch_order_findings(tmp_path, source)

    assert [finding.rule_id for finding in findings] == ["CCO1"]


def test_catch_order_ignores_single_handler_and_unrelated_pairs(tmp_path: Path) -> None:
    assert _catch_order_findings(tmp_path, "def f():\n    try:\n        pass\n    except ComposerPluginCrashError:\n        pass\n") == []
    assert _catch_order_findings(tmp_path, "def f():\n    try:\n        pass\n    except ComposerServiceError:\n        pass\n") == []
    assert (
        _catch_order_findings(
            tmp_path, "def f():\n    try:\n        pass\n    except OSError:\n        pass\n    except ValueError:\n        pass\n"
        )
        == []
    )


def test_catch_order_reports_runtime_preflight_shadowing(tmp_path: Path) -> None:
    source = (
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except ComposerServiceError as exc:\n"
        "        pass\n"
        "    except ComposerRuntimePreflightError as crash:\n"
        "        pass\n"
    )

    findings = _catch_order_findings(tmp_path, source)

    assert [finding.rule_id for finding in findings] == ["CCO1"]
    assert "ComposerRuntimePreflightError" in findings[0].message


def test_catch_order_declared_map_matches_real_composer_exception_mro() -> None:
    import_module("elspeth.web.composer.protocol")
    import_module("elspeth.web.composer.service")

    from elspeth.web.composer.protocol import ComposerServiceError

    composer_family: set[type] = {ComposerServiceError} | _all_subclasses(ComposerServiceError)
    name_to_cls = {cls.__name__: cls for cls in composer_family}
    real_subclasses = {name for name in name_to_cls if name != "ComposerServiceError"}

    assert set(_SUBCLASS_TO_SUPERCLASSES) == real_subclasses
    for sub_name, declared_supers in _SUBCLASS_TO_SUPERCLASSES.items():
        assert "ComposerServiceError" in declared_supers
        cls = name_to_cls[sub_name]
        real_supers = {ancestor.__name__ for ancestor in cls.__mro__[1:] if ancestor in composer_family}
        assert declared_supers == frozenset(real_supers)


def _exception_channel_findings(tmp_path: Path, source: str) -> list[Finding]:
    target = tmp_path / "web" / "composer" / "tools.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(source, filename=str(target))
    return list(EXCEPTION_CHANNEL_RULE.analyze(tree, target, RuleContext(root=tmp_path)))


def _catch_order_findings(tmp_path: Path, source: str) -> list[Finding]:
    target = tmp_path / "web" / "sessions" / "routes.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(source, filename=str(target))
    return list(CATCH_ORDER_RULE.analyze(tree, target, RuleContext(root=tmp_path)))


def _all_subclasses(cls: type) -> set[type]:
    discovered: set[type] = set()
    stack: list[type] = [cls]
    while stack:
        parent = stack.pop()
        for child in parent.__subclasses__():
            if child not in discovered:
                discovered.add(child)
                stack.append(child)
    return discovered
