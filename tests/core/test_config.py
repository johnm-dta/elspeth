# tests/core/test_config.py
"""Tests for configuration schema and loading."""

import pytest
from pydantic import ValidationError


class TestDatabaseSettings:
    """Database configuration validation."""

    def test_valid_sqlite_url(self) -> None:
        from elspeth.core.config import DatabaseSettings

        settings = DatabaseSettings(url="sqlite:///audit.db")
        assert settings.url == "sqlite:///audit.db"
        assert settings.pool_size == 5  # default

    def test_valid_postgres_url(self) -> None:
        from elspeth.core.config import DatabaseSettings

        settings = DatabaseSettings(
            url="postgresql://user:pass@localhost/db",
            pool_size=10,
        )
        assert settings.pool_size == 10

    def test_pool_size_must_be_positive(self) -> None:
        from elspeth.core.config import DatabaseSettings

        with pytest.raises(ValidationError):
            DatabaseSettings(url="sqlite:///test.db", pool_size=0)

    def test_settings_are_frozen(self) -> None:
        from elspeth.core.config import DatabaseSettings

        settings = DatabaseSettings(url="sqlite:///test.db")
        with pytest.raises(ValidationError):
            settings.url = "sqlite:///other.db"


class TestRetrySettings:
    """Retry configuration validation."""

    def test_defaults(self) -> None:
        from elspeth.core.config import RetrySettings

        settings = RetrySettings()
        assert settings.max_attempts == 3
        assert settings.initial_delay_seconds == 1.0
        assert settings.max_delay_seconds == 60.0
        assert settings.exponential_base == 2.0

    def test_max_attempts_must_be_positive(self) -> None:
        from elspeth.core.config import RetrySettings

        with pytest.raises(ValidationError):
            RetrySettings(max_attempts=0)

    def test_delays_must_be_positive(self) -> None:
        from elspeth.core.config import RetrySettings

        with pytest.raises(ValidationError):
            RetrySettings(initial_delay_seconds=-1.0)


class TestElspethSettings:
    """Top-level settings validation."""

    def test_minimal_valid_config(self) -> None:
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            database={"url": "sqlite:///audit.db"},
        )
        assert settings.database.url == "sqlite:///audit.db"
        assert settings.retry.max_attempts == 3  # default

    def test_nested_config(self) -> None:
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            database={"url": "sqlite:///audit.db", "pool_size": 10},
            retry={"max_attempts": 5},
        )
        assert settings.database.pool_size == 10
        assert settings.retry.max_attempts == 5

    def test_run_id_prefix_default(self) -> None:
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(database={"url": "sqlite:///audit.db"})
        assert settings.run_id_prefix == "run"
