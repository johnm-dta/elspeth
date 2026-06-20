"""End-to-end integration tests for the ``report_assemble`` aggregation.

Exercises the production CLI path: text source -> report_assemble
aggregation -> JSON sink. Verifies pagination metadata is sourced from the
real ``AggregationBatchContext`` rather than from any in-plugin counter.

Two scenarios:

* Count-trigger pagination — three pages from five lines with ``count: 2``.
* End-of-source whole-report — no trigger; one report containing all lines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

runner = CliRunner()


class TestReportAssembleAggregationPipeline:
    """Production-path integration tests for ``report_assemble``."""

    @pytest.fixture
    def input_data(self, tmp_path: Path) -> Path:
        """Create a five-line text input file."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")
        return input_file

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def pipeline_config_count_trigger(self, tmp_path: Path, input_data: Path, output_dir: Path) -> Path:
        """Settings YAML with a ``count: 2`` aggregation trigger."""
        config = {
            "sources": {
                "primary": {
                    "plugin": "text",
                    "on_success": "lines",
                    "options": {
                        "path": str(input_data),
                        "column": "line",
                        "strip_whitespace": False,
                        "skip_blank_lines": False,
                        "schema": {"mode": "observed"},
                        "on_validation_failure": "discard",
                    },
                }
            },
            "aggregations": [
                {
                    "name": "pages",
                    "plugin": "report_assemble",
                    "input": "lines",
                    "on_success": "output",
                    "on_error": "discard",
                    "trigger": {"count": 2},
                    "output_mode": "transform",
                    "options": {
                        "schema": {"mode": "observed"},
                        "text_field": "line",
                    },
                },
            ],
            "sinks": {
                "output": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(output_dir / "pages.json"),
                        "schema": {"mode": "observed"},
                    },
                },
            },
            "landscape": {"url": f"sqlite:///{tmp_path / 'audit.db'}"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    @pytest.fixture
    def pipeline_config_no_trigger(self, tmp_path: Path, input_data: Path, output_dir: Path) -> Path:
        """Settings YAML with no aggregation trigger (end-of-source only)."""
        config = {
            "sources": {
                "primary": {
                    "plugin": "text",
                    "on_success": "lines",
                    "options": {
                        "path": str(input_data),
                        "column": "line",
                        "strip_whitespace": False,
                        "skip_blank_lines": False,
                        "schema": {"mode": "observed"},
                        "on_validation_failure": "discard",
                    },
                }
            },
            "aggregations": [
                {
                    "name": "pages",
                    "plugin": "report_assemble",
                    "input": "lines",
                    "on_success": "output",
                    "on_error": "discard",
                    "output_mode": "transform",
                    "options": {
                        "schema": {"mode": "observed"},
                        "text_field": "line",
                    },
                },
            ],
            "sinks": {
                "output": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(output_dir / "pages.json"),
                        "schema": {"mode": "observed"},
                    },
                },
            },
            "landscape": {"url": f"sqlite:///{tmp_path / 'audit.db'}"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_count_trigger_paginates_into_three_reports(self, pipeline_config_count_trigger: Path, output_dir: Path) -> None:
        """count=2 with 5 input rows -> two full pages + one end-of-source page.

        Pagination context comes from ``AggregationBatchContext``; the asserted
        line_start/line_end/report_index sequence locks in the production
        counter math (see plan §Task 4 for the canonical sequence).
        """
        from elspeth.cli import app

        result = runner.invoke(
            app,
            ["run", "-s", str(pipeline_config_count_trigger), "--execute"],
        )
        assert result.exit_code == 0, f"Pipeline failed: {result.output}"

        output_file = output_dir / "pages.json"
        assert output_file.exists(), "Output file should exist"

        rows: list[dict[str, Any]] = json.loads(output_file.read_text())
        assert len(rows) == 3, f"Expected 3 report rows, got {len(rows)}: {rows!r}"

        # Page 1: lines 1-2, count-triggered.
        assert rows[0]["report_body"] == "line 1\nline 2"
        assert rows[0]["report_index"] == 1
        assert rows[0]["line_start"] == 1
        assert rows[0]["line_end"] == 2
        assert rows[0]["line_count"] == 2
        assert rows[0]["lines_seen_total"] == 2
        assert rows[0]["flush_trigger"] == "count"
        assert rows[0]["is_end_of_source_report"] is False

        # Page 2: lines 3-4, count-triggered.
        assert rows[1]["report_body"] == "line 3\nline 4"
        assert rows[1]["report_index"] == 2
        assert rows[1]["line_start"] == 3
        assert rows[1]["line_end"] == 4
        assert rows[1]["line_count"] == 2
        assert rows[1]["lines_seen_total"] == 4
        assert rows[1]["flush_trigger"] == "count"
        assert rows[1]["is_end_of_source_report"] is False

        # Page 3: line 5 only, emitted at end-of-source.
        assert rows[2]["report_body"] == "line 5"
        assert rows[2]["report_index"] == 3
        assert rows[2]["line_start"] == 5
        assert rows[2]["line_end"] == 5
        assert rows[2]["line_count"] == 1
        assert rows[2]["lines_seen_total"] == 5
        assert rows[2]["flush_trigger"] == "end_of_source"
        assert rows[2]["is_end_of_source_report"] is True

    def test_end_of_source_emits_single_whole_report(self, pipeline_config_no_trigger: Path, output_dir: Path) -> None:
        """No trigger -> one whole-report row containing all five lines."""
        from elspeth.cli import app

        result = runner.invoke(
            app,
            ["run", "-s", str(pipeline_config_no_trigger), "--execute"],
        )
        assert result.exit_code == 0, f"Pipeline failed: {result.output}"

        output_file = output_dir / "pages.json"
        assert output_file.exists(), "Output file should exist"

        rows: list[dict[str, Any]] = json.loads(output_file.read_text())
        assert len(rows) == 1, f"Expected 1 report row, got {len(rows)}: {rows!r}"

        row = rows[0]
        assert row["report_body"] == "line 1\nline 2\nline 3\nline 4\nline 5"
        assert row["report_index"] == 1
        assert row["line_start"] == 1
        assert row["line_end"] == 5
        assert row["line_count"] == 5
        assert row["lines_seen_total"] == 5
        assert row["flush_trigger"] == "end_of_source"
        assert row["is_end_of_source_report"] is True
