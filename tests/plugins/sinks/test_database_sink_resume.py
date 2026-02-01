"""Tests for DatabaseSink resume capability."""

import os

import pytest

from elspeth.plugins.sinks.database_sink import DatabaseSink

# Strict schema for tests - DatabaseSink requires fixed columns
STRICT_SCHEMA = {"mode": "strict", "fields": ["id: int"]}


@pytest.fixture(autouse=True)
def allow_raw_secrets():
    """Allow raw secrets for testing."""
    os.environ["ELSPETH_ALLOW_RAW_SECRETS"] = "true"
    yield
    os.environ.pop("ELSPETH_ALLOW_RAW_SECRETS", None)


def test_database_sink_supports_resume():
    """DatabaseSink should declare supports_resume=True."""
    assert DatabaseSink.supports_resume is True


def test_database_sink_configure_for_resume_sets_append():
    """DatabaseSink.configure_for_resume should set if_exists to append."""
    sink = DatabaseSink(
        {
            "url": "sqlite:///:memory:",
            "table": "test_table",
            "schema": STRICT_SCHEMA,
            "if_exists": "replace",  # Explicit replace mode
        }
    )

    assert sink._if_exists == "replace"

    sink.configure_for_resume()

    assert sink._if_exists == "append"


def test_database_sink_configure_for_resume_idempotent():
    """Calling configure_for_resume multiple times should be safe."""
    sink = DatabaseSink(
        {
            "url": "sqlite:///:memory:",
            "table": "test_table",
            "schema": STRICT_SCHEMA,
        }
    )

    sink.configure_for_resume()
    sink.configure_for_resume()  # Second call

    assert sink._if_exists == "append"
