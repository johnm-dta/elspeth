# tests/plugins/llm/test_capacity_errors.py
"""Tests for capacity error classification."""

from elspeth.plugins.pooling import CapacityError, is_capacity_error
from elspeth.plugins.pooling.errors import CAPACITY_ERROR_CODES


class TestCapacityErrorClassification:
    """Test HTTP status code classification."""

    def test_429_is_capacity_error(self) -> None:
        """429 Too Many Requests is a capacity error."""
        assert is_capacity_error(429)
        assert 429 in CAPACITY_ERROR_CODES

    def test_503_is_capacity_error(self) -> None:
        """503 Service Unavailable is a capacity error."""
        assert is_capacity_error(503)
        assert 503 in CAPACITY_ERROR_CODES

    def test_529_is_capacity_error(self) -> None:
        """529 (Azure overloaded) is a capacity error."""
        assert is_capacity_error(529)
        assert 529 in CAPACITY_ERROR_CODES

    def test_500_is_not_capacity_error(self) -> None:
        """500 Internal Server Error is NOT a capacity error."""
        assert not is_capacity_error(500)
        assert 500 not in CAPACITY_ERROR_CODES

    def test_400_is_not_capacity_error(self) -> None:
        """400 Bad Request is NOT a capacity error."""
        assert not is_capacity_error(400)

    def test_401_is_not_capacity_error(self) -> None:
        """401 Unauthorized is NOT a capacity error."""
        assert not is_capacity_error(401)

    def test_200_is_not_capacity_error(self) -> None:
        """200 OK is NOT a capacity error."""
        assert not is_capacity_error(200)


class TestCapacityErrorException:
    """Test CapacityError exception."""

    def test_capacity_error_stores_status_code(self) -> None:
        """CapacityError should store the status code."""
        error = CapacityError(429, "Rate limited")

        assert error.status_code == 429
        assert str(error) == "Rate limited"

    def test_capacity_error_retryable_flag(self) -> None:
        """CapacityError should always be retryable."""
        error = CapacityError(503, "Service unavailable")

        assert error.retryable is True
