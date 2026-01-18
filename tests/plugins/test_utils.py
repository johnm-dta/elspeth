"""Tests for plugin utilities."""


class TestGetNestedField:
    """Tests for get_nested_field utility."""

    def test_get_nested_field_exists(self) -> None:
        """get_nested_field can be imported."""
        from elspeth.plugins.utils import get_nested_field

        assert get_nested_field is not None

    def test_simple_field_access(self) -> None:
        """Access top-level field."""
        from elspeth.plugins.utils import get_nested_field

        data = {"name": "Alice", "age": 30}
        assert get_nested_field(data, "name") == "Alice"
        assert get_nested_field(data, "age") == 30

    def test_nested_field_access(self) -> None:
        """Access nested field with dot notation."""
        from elspeth.plugins.utils import get_nested_field

        data = {"user": {"name": "Bob", "profile": {"city": "NYC"}}}
        assert get_nested_field(data, "user.name") == "Bob"
        assert get_nested_field(data, "user.profile.city") == "NYC"

    def test_missing_field_returns_sentinel(self) -> None:
        """Missing field returns MISSING sentinel."""
        from elspeth.plugins.sentinels import MISSING
        from elspeth.plugins.utils import get_nested_field

        data = {"name": "Alice"}
        result = get_nested_field(data, "age")
        assert result is MISSING

    def test_missing_nested_field_returns_sentinel(self) -> None:
        """Missing nested field returns MISSING sentinel."""
        from elspeth.plugins.sentinels import MISSING
        from elspeth.plugins.utils import get_nested_field

        data = {"user": {"name": "Alice"}}
        result = get_nested_field(data, "user.email")
        assert result is MISSING

    def test_missing_intermediate_returns_sentinel(self) -> None:
        """Missing intermediate path returns MISSING sentinel."""
        from elspeth.plugins.sentinels import MISSING
        from elspeth.plugins.utils import get_nested_field

        data = {"user": {"name": "Alice"}}
        result = get_nested_field(data, "user.profile.city")
        assert result is MISSING

    def test_custom_default(self) -> None:
        """Custom default value for missing fields."""
        from elspeth.plugins.utils import get_nested_field

        data = {"name": "Alice"}
        result = get_nested_field(data, "age", default=0)
        assert result == 0

    def test_none_value_not_missing(self) -> None:
        """Explicit None is returned, not treated as missing."""
        from elspeth.plugins.sentinels import MISSING
        from elspeth.plugins.utils import get_nested_field

        data = {"value": None}
        result = get_nested_field(data, "value")
        assert result is None
        assert result is not MISSING

    def test_non_dict_intermediate_returns_sentinel(self) -> None:
        """Non-dict intermediate value returns MISSING."""
        from elspeth.plugins.sentinels import MISSING
        from elspeth.plugins.utils import get_nested_field

        data = {"user": "string_not_dict"}
        result = get_nested_field(data, "user.name")
        assert result is MISSING
