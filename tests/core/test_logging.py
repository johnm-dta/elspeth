# tests/core/test_logging.py
"""Tests for structured logging configuration."""

import json
import logging

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

    def test_logger_outputs_structured(self, capsys: pytest.CaptureFixture[str]) -> None:
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

    def test_noisy_third_party_loggers_silenced(self) -> None:
        """Third-party loggers (Azure SDK, urllib3, etc.) are silenced to WARNING.

        Even when ELSPETH runs in DEBUG mode, we don't want HTTP connection
        spam from Azure SDK, urllib3, and OpenTelemetry internals.
        """
        import logging

        from elspeth.core.logging import configure_logging

        # Configure with DEBUG level (verbose mode)
        configure_logging(level="DEBUG")

        # Root logger should be at DEBUG
        assert logging.getLogger().level == logging.DEBUG

        # Noisy third-party loggers should be silenced to WARNING
        noisy_loggers = [
            "azure",
            "azure.core.pipeline.policies.http_logging_policy",
            "azure.identity",
            "urllib3",
            "opentelemetry",
        ]

        for name in noisy_loggers:
            logger = logging.getLogger(name)
            assert logger.getEffectiveLevel() >= logging.WARNING, (
                f"Logger '{name}' should be WARNING or higher, got level {logger.getEffectiveLevel()}"
            )

    def test_stdlib_loggers_emit_json_when_json_output_enabled(self, capsys: pytest.CaptureFixture[str]) -> None:
        """stdlib loggers emit JSON when json_output=True.

        This is the critical test for P2-2026-01-31-json-logs-mixed-output.
        Modules using logging.getLogger(__name__) must produce JSON output
        when json_output=True, not plain text mixed with structlog JSON.
        """
        from elspeth.core.logging import configure_logging

        configure_logging(json_output=True)

        # Get a stdlib logger (simulates what plugins do)
        stdlib_logger = logging.getLogger("test.stdlib.module")
        stdlib_logger.info("message from stdlib logger")

        captured = capsys.readouterr()
        log_line = captured.out.strip().split("\n")[-1]

        # MUST be valid JSON
        data = json.loads(log_line)
        assert data["event"] == "message from stdlib logger"
        assert "level" in data  # Should have standard structlog fields
        assert "timestamp" in data

    def test_stdlib_loggers_emit_console_when_console_mode(self, capsys: pytest.CaptureFixture[str]) -> None:
        """stdlib loggers emit console format when json_output=False.

        Ensures stdlib loggers go through ConsoleRenderer in console mode.
        """
        from elspeth.core.logging import configure_logging

        configure_logging(json_output=False)

        stdlib_logger = logging.getLogger("test.stdlib.console")
        stdlib_logger.info("message from stdlib logger")

        captured = capsys.readouterr()
        # Should contain the message
        assert "message from stdlib logger" in captured.out
        # Should NOT be JSON
        assert not captured.out.strip().split("\n")[-1].startswith("{")
