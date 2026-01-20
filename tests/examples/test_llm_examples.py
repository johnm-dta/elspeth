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


class TestOpenRouterSentimentBasic:
    """Test openrouter_sentiment basic example."""

    EXAMPLE_DIR = Path("examples/openrouter_sentiment")
    SETTINGS = EXAMPLE_DIR / "settings.yaml"
    OUTPUT = EXAMPLE_DIR / "output" / "results.csv"
    RUNS_DIR = EXAMPLE_DIR / "runs"

    @_requires_api_key()
    def test_produces_correct_output(self) -> None:
        """Basic sentiment example should classify all 5 rows."""
        # Clean up
        _clean_output_dir(self.OUTPUT.parent)
        _clean_runs_dir(self.RUNS_DIR)

        # Run pipeline
        _run_pipeline(str(self.SETTINGS))

        # Verify output exists
        assert self.OUTPUT.exists(), f"Output file not created: {self.OUTPUT}"

        # Read and verify content
        rows = _read_csv(self.OUTPUT)
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

        # Verify required fields present
        required_fields = ["id", "text", "sentiment_analysis"]
        for row in rows:
            for field in required_fields:
                assert field in row, f"Missing field: {field}"

        # Verify sentiment analysis contains valid JSON with expected structure
        for row in rows:
            analysis = json.loads(row["sentiment_analysis"])
            assert "sentiment" in analysis, "Missing sentiment field"
            assert analysis["sentiment"] in ["positive", "negative", "neutral"]
            assert "confidence" in analysis, "Missing confidence field"
            assert 0 <= analysis["confidence"] <= 1, "Confidence out of range"

        # Verify expected sentiments (should be deterministic with temperature=0)
        sentiments = {int(row["id"]): json.loads(row["sentiment_analysis"])["sentiment"] for row in rows}
        assert sentiments[1] == "positive"  # "I absolutely love this product!"
        assert sentiments[2] == "negative"  # "The service was terrible"
        assert sentiments[3] == "neutral"  # "It was okay"
        assert sentiments[4] == "positive"  # "Amazing experience!"
        assert sentiments[5] == "negative"  # "Completely disappointed"


class TestOpenRouterSentimentPooled:
    """Test openrouter_sentiment pooled example."""

    EXAMPLE_DIR = Path("examples/openrouter_sentiment")
    SETTINGS = EXAMPLE_DIR / "settings_pooled.yaml"
    OUTPUT = EXAMPLE_DIR / "output" / "results_pooled.csv"
    RUNS_DIR = EXAMPLE_DIR / "runs"

    @_requires_api_key()
    def test_produces_correct_output(self) -> None:
        """Pooled sentiment example should classify all 5 rows with parallel execution."""
        # Clean up
        _clean_output_dir(self.OUTPUT.parent)
        _clean_runs_dir(self.RUNS_DIR)

        # Run pipeline
        _run_pipeline(str(self.SETTINGS))

        # Verify output exists
        assert self.OUTPUT.exists(), f"Output file not created: {self.OUTPUT}"

        # Read and verify content
        rows = _read_csv(self.OUTPUT)
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

        # Verify sentiment analysis structure
        for row in rows:
            analysis = json.loads(row["sentiment_analysis"])
            assert analysis["sentiment"] in ["positive", "negative", "neutral"]

        # Verify expected sentiments match basic example
        sentiments = {int(row["id"]): json.loads(row["sentiment_analysis"])["sentiment"] for row in rows}
        assert sentiments[1] == "positive"
        assert sentiments[2] == "negative"
        assert sentiments[3] == "neutral"
        assert sentiments[4] == "positive"
        assert sentiments[5] == "negative"


class TestOpenRouterSentimentBatched:
    """Test openrouter_sentiment batched example."""

    EXAMPLE_DIR = Path("examples/openrouter_sentiment")
    SETTINGS = EXAMPLE_DIR / "settings_batched.yaml"
    OUTPUT = EXAMPLE_DIR / "output" / "results_batched.csv"
    RUNS_DIR = EXAMPLE_DIR / "runs"

    @_requires_api_key()
    def test_produces_correct_output(self) -> None:
        """Batched sentiment example should classify all 5 rows via batch aggregation."""
        # Clean up
        _clean_output_dir(self.OUTPUT.parent)
        _clean_runs_dir(self.RUNS_DIR)

        # Run pipeline
        _run_pipeline(str(self.SETTINGS))

        # Verify output exists
        assert self.OUTPUT.exists(), f"Output file not created: {self.OUTPUT}"

        # Read and verify content
        rows = _read_csv(self.OUTPUT)
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

        # Verify sentiment analysis structure
        for row in rows:
            analysis = json.loads(row["sentiment_analysis"])
            assert analysis["sentiment"] in ["positive", "negative", "neutral"]

        # Verify expected sentiments match basic example
        sentiments = {int(row["id"]): json.loads(row["sentiment_analysis"])["sentiment"] for row in rows}
        assert sentiments[1] == "positive"
        assert sentiments[2] == "negative"
        assert sentiments[3] == "neutral"
        assert sentiments[4] == "positive"
        assert sentiments[5] == "negative"


class TestTemplateLookups:
    """Test template_lookups basic example."""

    EXAMPLE_DIR = Path("examples/template_lookups")
    SETTINGS = EXAMPLE_DIR / "settings.yaml"
    OUTPUT = EXAMPLE_DIR / "output" / "results.csv"
    RUNS_DIR = EXAMPLE_DIR / "runs"

    @_requires_api_key()
    def test_produces_correct_output(self) -> None:
        """Template lookups example should classify all 5 support tickets."""
        # Clean up
        _clean_output_dir(self.OUTPUT.parent)
        _clean_runs_dir(self.RUNS_DIR)

        # Run pipeline
        _run_pipeline(str(self.SETTINGS))

        # Verify output exists
        assert self.OUTPUT.exists(), f"Output file not created: {self.OUTPUT}"

        # Read and verify content
        rows = _read_csv(self.OUTPUT)
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

        # Verify classification field present
        for row in rows:
            assert "classification" in row, "Missing classification field"

        # Verify template and lookup hashes are tracked
        for row in rows:
            assert "classification_template_hash" in row, "Missing template hash"
            assert row["classification_template_hash"], "Template hash should not be empty"
            assert "classification_lookup_hash" in row, "Missing lookup hash"
            assert row["classification_lookup_hash"], "Lookup hash should not be empty"


class TestTemplateLookupsBatched:
    """Test template_lookups batched example."""

    EXAMPLE_DIR = Path("examples/template_lookups")
    SETTINGS = EXAMPLE_DIR / "settings_batched.yaml"
    OUTPUT = EXAMPLE_DIR / "output" / "results_batched.csv"
    RUNS_DIR = EXAMPLE_DIR / "runs"

    @_requires_api_key()
    def test_produces_correct_output(self) -> None:
        """Batched template lookups example should classify via batch aggregation."""
        # Clean up
        _clean_output_dir(self.OUTPUT.parent)
        _clean_runs_dir(self.RUNS_DIR)

        # Run pipeline
        _run_pipeline(str(self.SETTINGS))

        # Verify output exists
        assert self.OUTPUT.exists(), f"Output file not created: {self.OUTPUT}"

        # Read and verify content
        rows = _read_csv(self.OUTPUT)
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

        # Verify classification field present
        for row in rows:
            assert "classification" in row, "Missing classification field"

        # Verify template and lookup hashes are tracked
        for row in rows:
            assert "classification_template_hash" in row, "Missing template hash"
            assert row["classification_template_hash"], "Template hash should not be empty"
            assert "classification_lookup_hash" in row, "Missing lookup hash"
            assert row["classification_lookup_hash"], "Lookup hash should not be empty"
