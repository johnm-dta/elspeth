# tests/unit/engine/test_bootstrap_preflight.py
"""Unit tests for resolve_preflight() — pure logic, no CLI/DB/IO wiring.

Integration tests for bootstrap_and_run() (the full CLI→orchestrator path)
live in tests/integration/pipeline/test_bootstrap_preflight.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.errors import CommencementGateFailedError
from elspeth.contracts.pipeline_runner import PipelineRunner
from elspeth.core.dependency_config import (
    CommencementGateConfig,
    CommencementGateResult,
    DependencyRunResult,
)


class TestResolvePreflightDirect:
    """Direct unit tests for resolve_preflight() (extracted from bootstrap_and_run)."""

    def test_returns_none_when_nothing_configured(self) -> None:
        """Returns None when neither depends_on nor commencement_gates are configured."""
        from elspeth.engine.bootstrap import resolve_preflight

        mock_config = MagicMock()
        mock_config.depends_on = []
        mock_config.commencement_gates = []

        result = resolve_preflight(mock_config, Path("/fake/pipeline.yaml"))
        assert result is None

    def test_calls_dependency_resolver(self) -> None:
        """When depends_on is configured, calls detect_cycles and resolve_dependencies."""
        from elspeth.core.dependency_config import DependencyConfig
        from elspeth.engine.bootstrap import resolve_preflight

        mock_config = MagicMock()
        mock_config.depends_on = [DependencyConfig(name="indexer", settings="./index.yaml")]
        mock_config.commencement_gates = []

        dep_result = DependencyRunResult(
            name="indexer",
            run_id="dep-run-abc",
            settings_hash="sha256:abc",
            duration_ms=1000,
            indexed_at="2026-03-25T12:00:00Z",
        )

        mock_runner = MagicMock(spec=PipelineRunner)

        with (
            patch("elspeth.engine.dependency_resolver.detect_cycles") as mock_detect,
            patch("elspeth.engine.dependency_resolver.resolve_dependencies", return_value=[dep_result]) as mock_resolve,
        ):
            result = resolve_preflight(mock_config, Path("/fake/pipeline.yaml"), runner=mock_runner)

        mock_detect.assert_called_once_with(Path("/fake/pipeline.yaml"))
        mock_resolve.assert_called_once()
        assert result is not None
        assert len(result.dependency_runs) == 1
        assert result.dependency_runs[0].name == "indexer"

    def test_evaluates_commencement_gates_without_dependencies(self) -> None:
        """When only commencement_gates are configured, gates are evaluated."""
        from elspeth.engine.bootstrap import resolve_preflight

        mock_config = MagicMock()
        mock_config.depends_on = []
        mock_config.commencement_gates = [CommencementGateConfig(name="test_gate", condition="True")]

        gate_result = CommencementGateResult(
            name="corpus_ready",
            condition="collections['test']['count'] > 0",
            result=True,
            context_snapshot={"collections": {"test": {"count": 10}}},
        )

        mock_probes: list[Any] = []

        with (
            patch("elspeth.engine.commencement.evaluate_commencement_gates", return_value=[gate_result]) as mock_eval,
        ):
            result = resolve_preflight(mock_config, Path("/fake/pipeline.yaml"), probes=mock_probes)

        mock_eval.assert_called_once()
        assert result is not None
        assert len(result.gate_results) == 1
        assert result.gate_results[0].name == "corpus_ready"
        assert result.gate_results[0].result is True

    def test_dependency_results_flow_into_gate_context(self) -> None:
        """When both depends_on and gates are configured, dep results are in gate context."""
        from elspeth.core.dependency_config import DependencyConfig
        from elspeth.engine.bootstrap import resolve_preflight

        mock_config = MagicMock()
        mock_config.depends_on = [DependencyConfig(name="indexer", settings="./index.yaml")]
        mock_config.commencement_gates = [CommencementGateConfig(name="test_gate", condition="True")]

        dep_result = DependencyRunResult(
            name="indexer",
            run_id="dep-run-abc",
            settings_hash="sha256:abc",
            duration_ms=1000,
            indexed_at="2026-03-25T12:00:00Z",
        )

        captured_context = {}

        def capture_gate_context(gates: object, context: dict[str, Any]) -> list[Any]:
            captured_context.update(context)
            return []

        mock_runner = MagicMock(spec=PipelineRunner)
        mock_probes: list[Any] = []

        with (
            patch("elspeth.engine.dependency_resolver.detect_cycles"),
            patch("elspeth.engine.dependency_resolver.resolve_dependencies", return_value=[dep_result]),
            patch("elspeth.engine.commencement.evaluate_commencement_gates", side_effect=capture_gate_context),
        ):
            resolve_preflight(mock_config, Path("/fake/pipeline.yaml"), probes=mock_probes, runner=mock_runner)

        assert "dependency_runs" in captured_context
        assert "indexer" in captured_context["dependency_runs"]
        assert captured_context["dependency_runs"]["indexer"]["run_id"] == "dep-run-abc"

    def test_gate_failure_propagates(self) -> None:
        """CommencementGateFailedError propagates through resolve_preflight."""
        from elspeth.engine.bootstrap import resolve_preflight

        mock_config = MagicMock()
        mock_config.depends_on = []
        mock_config.commencement_gates = [CommencementGateConfig(name="test_gate", condition="True")]

        mock_probes: list[Any] = []

        with (
            patch(
                "elspeth.engine.commencement.evaluate_commencement_gates",
                side_effect=CommencementGateFailedError(
                    gate_name="test_gate",
                    condition="False",
                    reason="test failure",
                    context_snapshot={},
                ),
            ),
            pytest.raises(CommencementGateFailedError, match="test_gate"),
        ):
            resolve_preflight(mock_config, Path("/fake/pipeline.yaml"), probes=mock_probes)

    def test_runner_none_with_depends_on_raises_framework_bug(self) -> None:
        """FrameworkBugError when runner=None but depends_on is configured."""
        from elspeth.contracts.errors import FrameworkBugError
        from elspeth.core.dependency_config import DependencyConfig
        from elspeth.engine.bootstrap import resolve_preflight

        mock_config = MagicMock()
        mock_config.depends_on = [DependencyConfig(name="idx", settings="./idx.yaml")]
        mock_config.commencement_gates = []

        with pytest.raises(FrameworkBugError, match="runner is required"):
            resolve_preflight(mock_config, Path("/fake/pipeline.yaml"), runner=None)

    def test_probes_none_with_gates_raises_framework_bug(self) -> None:
        """FrameworkBugError when probes=None but commencement_gates is configured."""
        from elspeth.contracts.errors import FrameworkBugError
        from elspeth.engine.bootstrap import resolve_preflight

        mock_config = MagicMock()
        mock_config.depends_on = []
        mock_config.commencement_gates = [CommencementGateConfig(name="test_gate", condition="True")]

        with pytest.raises(FrameworkBugError, match="probes is required"):
            resolve_preflight(mock_config, Path("/fake/pipeline.yaml"), probes=None)

    def test_duplicate_dependency_names_rejected_before_execution(self) -> None:
        """Duplicate dependency names must raise before resolve_dependencies runs."""
        from elspeth.core.dependency_config import DependencyConfig
        from elspeth.engine.bootstrap import resolve_preflight

        mock_config = MagicMock()
        mock_config.depends_on = [
            DependencyConfig(name="indexer", settings="./a.yaml"),
            DependencyConfig(name="indexer", settings="./b.yaml"),
        ]
        mock_config.commencement_gates = None

        mock_runner = MagicMock(spec=PipelineRunner)

        with (
            patch("elspeth.engine.dependency_resolver.detect_cycles") as mock_detect,
            patch("elspeth.engine.dependency_resolver.resolve_dependencies") as mock_resolve,
            pytest.raises(ValueError, match=r"Duplicate dependency names.*indexer"),
        ):
            resolve_preflight(mock_config, Path("/fake/pipeline.yaml"), runner=mock_runner)

        mock_detect.assert_not_called()
        mock_resolve.assert_not_called()
