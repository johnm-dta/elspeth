# tests/examples/test_llm_examples.py
"""Integration tests for LLM examples.

These tests verify that LLM examples produce correct results by:
1. Deleting existing output files
2. Running the pipeline with actual API calls
3. Verifying output structure and content

REQUIRES: OPENROUTER_API_KEY environment variable to be set.

Run with:
    pytest tests/examples/test_llm_examples.py -v

Skip with:
    pytest tests/examples/test_llm_examples.py -v -k "not integration"
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def _requires_api_key() -> pytest.MarkDecorator:
    """Skip test if OPENROUTER_API_KEY is not set."""
    return pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set",
    )


def _run_pipeline(settings_path: str) -> None:
    """Run elspeth pipeline with the given settings file."""
    import subprocess

    env = os.environ.copy()
    env["ELSPETH_ALLOW_RAW_SECRETS"] = "true"

    result = subprocess.run(
        ["uv", "run", "elspeth", "run", "-s", settings_path, "--execute"],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).parent.parent.parent,  # Project root
    )

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read CSV file and return list of row dicts."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file and return list of row dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _clean_output_dir(output_dir: Path) -> None:
    """Remove all files in output directory."""
    if output_dir.exists():
        for file in output_dir.iterdir():
            if file.is_file():
                file.unlink()


def _clean_runs_dir(runs_dir: Path) -> None:
    """Remove all files in runs directory (audit databases)."""
    if runs_dir.exists():
        for file in runs_dir.iterdir():
            if file.is_file():
                file.unlink()


# Sentiment example configurations
SENTIMENT_EXAMPLES = [
    pytest.param(
        "settings.yaml",
        "output/results.json",
        id="basic",
    ),
    pytest.param(
        "settings_pooled.yaml",
        "output/results_pooled.json",
        id="pooled",
    ),
    pytest.param(
        "settings_batched.yaml",
        "output/results_batched.json",
        id="batched",
    ),
]

# Expected sentiment results (deterministic with temperature=0)
EXPECTED_SENTIMENTS = {
    1: "positive",  # "I absolutely love this product!"
    2: "negative",  # "The service was terrible"
    3: "neutral",  # "It was okay"
    4: "positive",  # "Amazing experience!"
    5: "negative",  # "Completely disappointed"
}


class TestOpenRouterSentiment:
    """Parametrized tests for openrouter_sentiment examples."""

    EXAMPLE_DIR = Path("examples/openrouter_sentiment")

    @_requires_api_key()
    @pytest.mark.parametrize("settings_file,output_file", SENTIMENT_EXAMPLES)
    def test_produces_correct_output(self, settings_file: str, output_file: str) -> None:
        """Sentiment example should classify all 5 rows correctly."""
        settings_path = self.EXAMPLE_DIR / settings_file
        output_path = self.EXAMPLE_DIR / output_file
        runs_dir = self.EXAMPLE_DIR / "runs"

        # Clean up
        _clean_output_dir(output_path.parent)
        _clean_runs_dir(runs_dir)

        # Run pipeline
        _run_pipeline(str(settings_path))

        # Verify output exists
        assert output_path.exists(), f"Output file not created: {output_path}"

        # Read and verify content (JSONL format - sentiment_analysis is already parsed)
        rows = _read_jsonl(output_path)
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

        # Verify required fields and sentiment structure
        for row in rows:
            assert "id" in row, "Missing id field"
            assert "text" in row, "Missing text field"
            assert "sentiment_analysis" in row, "Missing sentiment_analysis field"

            # sentiment_analysis may be a dict or JSON string depending on transform output
            analysis = row["sentiment_analysis"]
            if isinstance(analysis, str):
                analysis = json.loads(analysis)
            assert "sentiment" in analysis, "Missing sentiment field in analysis"
            assert analysis["sentiment"] in ["positive", "negative", "neutral"]
            assert "confidence" in analysis, "Missing confidence field in analysis"
            assert 0 <= analysis["confidence"] <= 1, "Confidence out of range"

        # Verify expected sentiments
        def get_sentiment(row: dict) -> str:
            analysis = row["sentiment_analysis"]
            if isinstance(analysis, str):
                analysis = json.loads(analysis)
            return analysis["sentiment"]

        sentiments = {int(row["id"]): get_sentiment(row) for row in rows}
        for row_id, expected in EXPECTED_SENTIMENTS.items():
            assert sentiments[row_id] == expected, f"Row {row_id}: expected {expected}, got {sentiments[row_id]}"


# Template lookups example configurations
TEMPLATE_LOOKUP_EXAMPLES = [
    pytest.param(
        "settings.yaml",
        "output/results.json",
        id="basic",
    ),
    pytest.param(
        "settings_batched.yaml",
        "output/results_batched.json",
        id="batched",
    ),
]


class TestTemplateLookups:
    """Parametrized tests for template_lookups examples."""

    EXAMPLE_DIR = Path("examples/template_lookups")

    @_requires_api_key()
    @pytest.mark.parametrize("settings_file,output_file", TEMPLATE_LOOKUP_EXAMPLES)
    def test_produces_correct_output(self, settings_file: str, output_file: str) -> None:
        """Template lookups example should classify all 5 support tickets."""
        settings_path = self.EXAMPLE_DIR / settings_file
        output_path = self.EXAMPLE_DIR / output_file
        runs_dir = self.EXAMPLE_DIR / "runs"

        # Clean up
        _clean_output_dir(output_path.parent)
        _clean_runs_dir(runs_dir)

        # Run pipeline
        _run_pipeline(str(settings_path))

        # Verify output exists
        assert output_path.exists(), f"Output file not created: {output_path}"

        # Read and verify content (JSONL format)
        rows = _read_jsonl(output_path)
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

        # Verify classification field and hash tracking
        for row in rows:
            assert "classification" in row, "Missing classification field"
            assert "classification_template_hash" in row, "Missing template hash"
            assert row["classification_template_hash"], "Template hash should not be empty"
            assert "classification_lookup_hash" in row, "Missing lookup hash"
            assert row["classification_lookup_hash"], "Lookup hash should not be empty"
