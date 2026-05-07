# tests/unit/core/landscape/test_models_mutation_gaps.py
"""Surviving tests from mutation-gap suite — only non-trivial validation.

Original file tested 57 dataclass defaults/required-field patterns.
Most tested Python's @dataclass machinery, not ELSPETH logic.
Retained: enum type validation, required-field contracts for audit models,
full-construction smoke test.

Removed defaults-to-None tests: git log for 2026-04-02 has rationale.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts import (
    Determinism,
    ExportStatus,
    Node,
    NodeType,
    ReproducibilityGrade,
    Run,
    RunStatus,
)


class TestRunDataclass:
    """Verify Run dataclass non-trivial field contracts."""

    def test_status_is_required_run_status_enum(self) -> None:
        """status must be RunStatus enum instance, not string."""
        run = Run(
            run_id="run-001",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status=RunStatus.RUNNING,
        )
        assert isinstance(run.status, RunStatus)
        assert run.status == RunStatus.RUNNING

    def test_run_with_all_optional_fields_set(self) -> None:
        """Verify all optional fields can be set explicitly."""
        now = datetime.now(UTC)
        run = Run(
            run_id="run-002",
            started_at=now,
            config_hash="abc123",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status=RunStatus.COMPLETED,
            completed_at=now,
            reproducibility_grade=ReproducibilityGrade.FULL_REPRODUCIBLE,
            export_status=ExportStatus.COMPLETED,
            export_error=None,
            exported_at=now,
            export_format="csv",
            export_sink="output",
        )
        assert run.completed_at == now
        assert run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE
        assert run.export_status == ExportStatus.COMPLETED
        assert run.export_format == "csv"


class TestNodeDataclass:
    """Verify Node dataclass required-field contracts."""

    def test_registered_at_is_required(self) -> None:
        """registered_at is required (no default)."""
        with pytest.raises(TypeError):
            Node(  # type: ignore[call-arg]
                node_id="node-001",
                run_id="run-001",
                plugin_name="test",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash="x",
                config_json="{}",
                # registered_at missing
            )
