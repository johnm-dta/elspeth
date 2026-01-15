# tests/core/test_logging.py
"""Tests for structured logging configuration."""

import json

import pytest


class TestLoggingConfig:
    """Tests for logging configuration."""

    def test_get_logger_exists(self) -> None:
        """get_logger function exists."""
        from elspeth.core.logging import get_logger

        assert callable(get_logger)

    def test_get_logger_returns_logger(self) -> None:
        """get_logger returns a bound logger."""
        from elspeth.core.logging import get_logger

        logger = get_logger("test")
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "bind")

    def test_configure_logging_exists(self) -> None:
        """configure_logging function exists."""
        from elspeth.core.logging import configure_logging

        assert callable(configure_logging)

    def test_logger_outputs_structured(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Logger outputs structured JSON."""
        from elspeth.core.logging import configure_logging, get_logger

        configure_logging(json_output=True)
        logger = get_logger("test")

        logger.info("test message", key="value")

        captured = capsys.readouterr()
        # Should be valid JSON
        log_line = captured.out.strip().split("\n")[-1]
        data = json.loads(log_line)
        assert data["event"] == "test message"
        assert data["key"] == "value"

    def test_logger_console_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Logger outputs human-readable in console mode."""
        from elspeth.core.logging import configure_logging, get_logger

        configure_logging(json_output=False)
        logger = get_logger("test")

        logger.info("test message", key="value")

        captured = capsys.readouterr()
        assert "test message" in captured.out
        # Should NOT be JSON
        assert not captured.out.strip().startswith("{")

    def test_logger_binds_context(self) -> None:
        """Logger can bind context."""
        from elspeth.core.logging import get_logger

        logger = get_logger("test")
        bound = logger.bind(run_id="abc123")

        assert bound is not None
        # Bound logger is a new instance
        assert bound is not logger
