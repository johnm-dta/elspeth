"""Tests for commencement gate evaluation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from elspeth.contracts.errors import CommencementGateFailedError
from elspeth.core.dependency_config import CommencementGateConfig
from elspeth.engine.commencement import (
    build_preflight_context,
    evaluate_commencement_gates,
)


class TestEvaluateCommencementGates:
    def test_passing_gate(self) -> None:
        gates = [
            CommencementGateConfig(
                name="ready",
                condition="collections['test']['count'] > 0",
            )
        ]
        context = {
            "dependency_runs": {},
            "collections": {"test": {"count": 10, "reachable": True}},
            "env": {"HOME": "/home/user"},
        }
        results = evaluate_commencement_gates(gates, context)
        assert len(results) == 1
        assert results[0].result is True
        assert results[0].name == "ready"

    def test_failing_gate_raises(self) -> None:
        gates = [
            CommencementGateConfig(
                name="ready",
                condition="collections['test']['count'] > 0",
            )
        ]
        context = {
            "dependency_runs": {},
            "collections": {"test": {"count": 0, "reachable": False}},
            "env": {},
        }
        with pytest.raises(CommencementGateFailedError, match="ready"):
            evaluate_commencement_gates(gates, context)

    def test_expression_error_raises(self) -> None:
        gates = [
            CommencementGateConfig(
                name="bad",
                condition="collections['missing']['count'] > 0",
            )
        ]
        context: dict[str, dict[str, object]] = {
            "dependency_runs": {},
            "collections": {},
            "env": {},
        }
        with pytest.raises(CommencementGateFailedError, match="bad"):
            evaluate_commencement_gates(gates, context)

    def test_snapshot_excludes_env_values_but_includes_keys(self) -> None:
        gates = [
            CommencementGateConfig(
                name="ready",
                condition="collections['test']['count'] > 0",
            )
        ]
        context = {
            "dependency_runs": {},
            "collections": {"test": {"count": 5, "reachable": True}},
            "env": {"SECRET_KEY": "abc123", "API_TOKEN": "xyz789"},
        }
        results = evaluate_commencement_gates(gates, context)
        snapshot = results[0].context_snapshot
        # Snapshot includes env_keys (sorted key names) but not env values
        assert set(snapshot.keys()) == {"dependency_runs", "collections", "env_keys"}
        assert snapshot["env_keys"] == ("API_TOKEN", "SECRET_KEY")
        # Values must not appear anywhere in the snapshot
        assert "abc123" not in str(snapshot)
        assert "xyz789" not in str(snapshot)

    def test_snapshot_is_deep_frozen(self) -> None:
        gates = [
            CommencementGateConfig(
                name="ready",
                condition="collections['test']['count'] > 0",
            )
        ]
        context = {
            "dependency_runs": {},
            "collections": {"test": {"count": 5, "reachable": True}},
            "env": {},
        }
        results = evaluate_commencement_gates(gates, context)
        assert isinstance(results[0].context_snapshot, MappingProxyType)

    def test_multiple_gates_all_pass(self) -> None:
        gates = [
            CommencementGateConfig(name="g1", condition="collections['a']['count'] > 0"),
            CommencementGateConfig(name="g2", condition="collections['b']['count'] > 0"),
        ]
        context = {
            "dependency_runs": {},
            "collections": {
                "a": {"count": 5, "reachable": True},
                "b": {"count": 3, "reachable": True},
            },
            "env": {},
        }
        results = evaluate_commencement_gates(gates, context)
        assert len(results) == 2

    def test_second_gate_fails_stops_evaluation(self) -> None:
        gates = [
            CommencementGateConfig(name="g1", condition="collections['a']['count'] > 0"),
            CommencementGateConfig(name="g2", condition="collections['b']['count'] > 0"),
        ]
        context = {
            "dependency_runs": {},
            "collections": {
                "a": {"count": 5, "reachable": True},
                "b": {"count": 0, "reachable": False},
            },
            "env": {},
        }
        with pytest.raises(CommencementGateFailedError, match="g2"):
            evaluate_commencement_gates(gates, context)

    def test_env_accessible_in_expression(self) -> None:
        gates = [
            CommencementGateConfig(
                name="env_check",
                condition="env['ENVIRONMENT'] == 'production'",
            )
        ]
        context = {
            "dependency_runs": {},
            "collections": {},
            "env": {"ENVIRONMENT": "production"},
        }
        results = evaluate_commencement_gates(gates, context)
        assert len(results) == 1
        assert results[0].result is True

    def test_empty_gates_returns_empty(self) -> None:
        results = evaluate_commencement_gates([], {"dependency_runs": {}, "collections": {}, "env": {}})
        assert results == []

    def test_context_mutation_after_evaluation_does_not_affect_snapshot(self) -> None:
        """TOCTOU protection: mutating original context must not change recorded snapshots."""
        gate = CommencementGateConfig(name="always_pass", condition="True")
        context: dict[str, Any] = {
            "dependency_runs": {"dep1": {"run_id": "r1"}},
            "collections": {"col1": {"count": 5, "reachable": True}},
            "env": {"KEY": "value"},
        }
        results = evaluate_commencement_gates([gate], context)

        # Mutate the original context after evaluation
        context["dependency_runs"]["dep1"]["run_id"] = "MUTATED"
        context["collections"]["col1"]["count"] = 999
        context["new_key"] = "injected"

        # Snapshot must reflect pre-mutation state
        snapshot = results[0].context_snapshot
        assert snapshot["dependency_runs"]["dep1"]["run_id"] == "r1"
        assert snapshot["collections"]["col1"]["count"] == 5
        assert "new_key" not in snapshot


class TestCommencementGateCrashThrough:
    """Programming errors must crash through, not be wrapped as gate failures."""

    def _make_context(self) -> dict[str, object]:
        return {
            "dependency_runs": {},
            "collections": {},
            "env": {},
        }

    def test_type_error_crashes_through(self) -> None:
        from unittest.mock import patch

        gate = CommencementGateConfig(name="g", condition="collections['x']['count'] > 0")
        context = self._make_context()

        with (
            patch("elspeth.engine.commencement.ExpressionParser") as mock_cls,
            pytest.raises(TypeError),
        ):
            mock_cls.return_value.evaluate.side_effect = TypeError("bad operand")
            evaluate_commencement_gates([gate], context)

    def test_attribute_error_crashes_through(self) -> None:
        from unittest.mock import patch

        gate = CommencementGateConfig(name="g", condition="True")
        context = self._make_context()

        with (
            patch("elspeth.engine.commencement.ExpressionParser") as mock_cls,
            pytest.raises(AttributeError),
        ):
            mock_cls.return_value.evaluate.side_effect = AttributeError("no attr")
            evaluate_commencement_gates([gate], context)

    def test_name_error_crashes_through(self) -> None:
        from unittest.mock import patch

        gate = CommencementGateConfig(name="g", condition="True")
        context = self._make_context()

        with (
            patch("elspeth.engine.commencement.ExpressionParser") as mock_cls,
            pytest.raises(NameError),
        ):
            mock_cls.return_value.evaluate.side_effect = NameError("undefined")
            evaluate_commencement_gates([gate], context)


class TestBuildPreflightContext:
    def test_includes_all_sections(self) -> None:
        context = build_preflight_context(
            dependency_results={},
            collection_probes={"test": {"count": 5, "reachable": True}},
            env={"HOME": "/home"},
        )
        assert "dependency_runs" in context
        assert "collections" in context
        assert "env" in context

    def test_env_defaults_to_os_environ(self) -> None:
        context = build_preflight_context(
            dependency_results={},
            collection_probes={},
        )
        assert "env" in context
        assert isinstance(context["env"], dict)


class TestCommencementGateTypeEnforcement:
    """Regression: non-boolean gate results must be rejected, not coerced."""

    def test_non_boolean_result_rejected(self) -> None:
        """Gate returning a truthy non-boolean (e.g., int 1) is a config error."""
        gates = [
            CommencementGateConfig(
                name="count_check",
                condition="collections['data']['count']",
            ),
        ]
        context: dict[str, Any] = {
            "collections": {"data": {"count": 5}},
            "dependency_runs": {},
            "env": {},
        }
        with pytest.raises(CommencementGateFailedError, match="not bool"):
            evaluate_commencement_gates(gates, context)

    def test_string_result_rejected(self) -> None:
        """Gate returning a truthy string is a config error."""
        gates = [
            CommencementGateConfig(
                name="env_check",
                condition="env.get('HOME')",
            ),
        ]
        context: dict[str, Any] = {
            "collections": {},
            "dependency_runs": {},
            "env": {"HOME": "/home/user"},
        }
        with pytest.raises(CommencementGateFailedError, match="not bool"):
            evaluate_commencement_gates(gates, context)

    def test_boolean_result_accepted(self) -> None:
        """Gate returning actual bool passes normally."""
        gates = [
            CommencementGateConfig(
                name="real_check",
                condition="collections['data']['count'] > 0",
            ),
        ]
        context: dict[str, Any] = {
            "collections": {"data": {"count": 5}},
            "dependency_runs": {},
            "env": {},
        }
        results = evaluate_commencement_gates(gates, context)
        assert len(results) == 1
        assert results[0].result is True
