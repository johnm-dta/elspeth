# tests/core/test_logging.py
"""Tests for structured logging configuration."""

import json
import logging
from collections.abc import Iterator

import pytest


class TestLoggingConfig:
    """Tests for logging configuration."""

    @pytest.fixture(autouse=True)
    def _restore_root_logger(self) -> Iterator[None]:
        """Save/restore root logger handlers so configure_logging() doesn't leak.

        configure_logging() adds a StreamHandler(sys.stdout) to the root logger.
        Under pytest, sys.stdout is a capture wrapper that gets closed after each
        test. Without cleanup, subsequent tests hit the dead stream (221 times).
        """
        from elspeth.core.logging import _elspeth_handler_ids

        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        saved_ids = set(_elspeth_handler_ids)
        yield
        root.handlers = saved_handlers
        root.level = saved_level
        _elspeth_handler_ids.clear()
        _elspeth_handler_ids.update(saved_ids)

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

    def test_configure_logging_preserves_non_elspeth_handlers(self) -> None:
        """configure_logging must not remove handlers it didn't create.

        Bug: elspeth-afb11494b7 — root.handlers = [] drops ALL handlers
        including pytest caplog. Only ELSPETH-owned handlers should be replaced.
        """
        from elspeth.core.logging import configure_logging

        root = logging.getLogger()

        # Add a foreign handler (simulates pytest caplog or other tools)
        foreign_handler = logging.StreamHandler()
        foreign_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(foreign_handler)

        # Reconfigure logging — should NOT remove the foreign handler
        configure_logging(json_output=True)

        # The foreign handler must still be present
        assert foreign_handler in root.handlers, "configure_logging removed a handler it did not create"

        # Clean up
        root.removeHandler(foreign_handler)

    def test_configure_logging_replaces_own_handler_on_reconfig(self) -> None:
        """Repeated configure_logging calls should not accumulate ELSPETH handlers."""
        from elspeth.core.logging import _elspeth_handler_ids, configure_logging

        root = logging.getLogger()
        handlers_before = list(root.handlers)

        try:
            configure_logging(json_output=True)
            elspeth_count_first = len([h for h in root.handlers if id(h) in _elspeth_handler_ids])

            configure_logging(json_output=False)
            elspeth_count_second = len([h for h in root.handlers if id(h) in _elspeth_handler_ids])

            # Should have exactly one ELSPETH handler after each call
            assert elspeth_count_first == 1
            assert elspeth_count_second == 1
        finally:
            # Restore root logger state to avoid polluting other tests
            root.handlers = handlers_before
            _elspeth_handler_ids.clear()

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

    def test_noisy_loggers_respect_higher_root_threshold(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Noisy loggers must not bypass stricter root thresholds."""
        from elspeth.core.logging import configure_logging

        configure_logging(level="ERROR")
        capsys.readouterr()  # Clear prior output from earlier tests

        # Root logger should be at ERROR
        assert logging.getLogger().level == logging.ERROR

        noisy_logger = logging.getLogger("azure")
        assert noisy_logger.getEffectiveLevel() >= logging.ERROR

        noisy_logger.warning("this warning should be suppressed")
        captured = capsys.readouterr()
        assert "this warning should be suppressed" not in captured.out

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
