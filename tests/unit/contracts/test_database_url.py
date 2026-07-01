"""Tests for shared database URL validation."""

import pytest

from elspeth.contracts.database_url import validate_database_url_format


def test_database_url_bad_port_is_wrapped_as_invalid_format() -> None:
    with pytest.raises(ValueError, match=r"^invalid database URL format:") as exc_info:
        validate_database_url_format("postgresql://host:portx/db")

    assert isinstance(exc_info.value.__cause__, ValueError)
