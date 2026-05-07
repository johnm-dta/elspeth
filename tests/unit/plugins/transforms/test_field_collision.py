"""Tests for field collision detection utility."""

from elspeth.contracts.field_collision import detect_field_collisions


class TestDetectFieldCollisions:
    """Unit tests for detect_field_collisions()."""

    def test_no_collision_returns_none(self) -> None:
        existing = {"id", "name", "amount"}
        new = ["llm_response", "llm_response_usage"]
        assert detect_field_collisions(existing, new) is None

    def test_single_collision_returns_sorted_list(self) -> None:
        existing = {"id", "name", "llm_response"}
        new = ["llm_response", "llm_response_usage"]
        assert detect_field_collisions(existing, new) == ["llm_response"]

    def test_multiple_collisions_returns_sorted_list(self) -> None:
        existing = {"id", "fetch_status", "content", "fetch_url_final"}
        new = ["content", "fingerprint", "fetch_status", "fetch_url_final"]
        assert detect_field_collisions(existing, new) == [
            "content",
            "fetch_status",
            "fetch_url_final",
        ]

    def test_empty_new_fields_returns_none(self) -> None:
        existing = {"id", "name"}
        assert detect_field_collisions(existing, []) is None

    def test_empty_existing_fields_returns_none(self) -> None:
        assert detect_field_collisions(set(), ["a", "b"]) is None
