"""Tests for Filter transform."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol


class TestFilter:
    """Tests for Filter transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """Filter implements TransformProtocol."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "equals": "active"})
        assert isinstance(transform, TransformProtocol)

    def test_has_required_attributes(self) -> None:
        """Filter has name and schemas."""
        from elspeth.plugins.transforms.filter import Filter

        assert Filter.name == "filter"

    def test_equals_condition_pass(self, ctx: PluginContext) -> None:
        """Row passes when field equals value."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "equals": "active"})
        row = {"id": 1, "status": "active"}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row

    def test_equals_condition_fail(self, ctx: PluginContext) -> None:
        """Row filtered when field does not equal value."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "equals": "active"})
        row = {"id": 1, "status": "inactive"}

        result = transform.process(row, ctx)

        # Filtered rows return success with row=None
        assert result.status == "success"
        assert result.row is None

    def test_not_equals_condition(self, ctx: PluginContext) -> None:
        """Row passes when field does not equal value."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "not_equals": "deleted"})

        active_row = {"id": 1, "status": "active"}
        result = transform.process(active_row, ctx)
        assert result.status == "success"
        assert result.row == active_row

        deleted_row = {"id": 2, "status": "deleted"}
        result = transform.process(deleted_row, ctx)
        assert result.status == "success"
        assert result.row is None

    def test_greater_than_condition(self, ctx: PluginContext) -> None:
        """Row passes when field is greater than value."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "score", "greater_than": 50})

        high_row = {"id": 1, "score": 75}
        result = transform.process(high_row, ctx)
        assert result.row == high_row

        low_row = {"id": 2, "score": 25}
        result = transform.process(low_row, ctx)
        assert result.row is None

    def test_less_than_condition(self, ctx: PluginContext) -> None:
        """Row passes when field is less than value."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "age", "less_than": 18})

        young_row = {"id": 1, "age": 15}
        result = transform.process(young_row, ctx)
        assert result.row == young_row

        adult_row = {"id": 2, "age": 25}
        result = transform.process(adult_row, ctx)
        assert result.row is None

    def test_contains_condition(self, ctx: PluginContext) -> None:
        """Row passes when field contains substring."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "email", "contains": "@example.com"})

        match_row = {"id": 1, "email": "alice@example.com"}
        result = transform.process(match_row, ctx)
        assert result.row == match_row

        nomatch_row = {"id": 2, "email": "bob@other.com"}
        result = transform.process(nomatch_row, ctx)
        assert result.row is None

    def test_matches_regex_condition(self, ctx: PluginContext) -> None:
        """Row passes when field matches regex."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "code", "matches": r"^[A-Z]{3}-\d{4}$"})

        match_row = {"id": 1, "code": "ABC-1234"}
        result = transform.process(match_row, ctx)
        assert result.row == match_row

        nomatch_row = {"id": 2, "code": "invalid"}
        result = transform.process(nomatch_row, ctx)
        assert result.row is None

    def test_in_list_condition(self, ctx: PluginContext) -> None:
        """Row passes when field value is in list."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "in": ["active", "pending"]})

        active_row = {"id": 1, "status": "active"}
        result = transform.process(active_row, ctx)
        assert result.row == active_row

        deleted_row = {"id": 2, "status": "deleted"}
        result = transform.process(deleted_row, ctx)
        assert result.row is None

    def test_missing_field_filters_out(self, ctx: PluginContext) -> None:
        """Row filtered when field is missing (unless allow_missing)."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "status", "equals": "active"})
        row = {"id": 1}  # No status field

        result = transform.process(row, ctx)
        assert result.row is None

    def test_allow_missing_field(self, ctx: PluginContext) -> None:
        """Row passes when field is missing and allow_missing=True."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({
            "field": "status",
            "equals": "active",
            "allow_missing": True,
        })
        row = {"id": 1}  # No status field

        result = transform.process(row, ctx)
        assert result.row == row  # Passes because allow_missing

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Filter on nested field with dot notation."""
        from elspeth.plugins.transforms.filter import Filter

        transform = Filter({"field": "meta.status", "equals": "approved"})

        approved_row = {"id": 1, "meta": {"status": "approved"}}
        result = transform.process(approved_row, ctx)
        assert result.row == approved_row

        pending_row = {"id": 2, "meta": {"status": "pending"}}
        result = transform.process(pending_row, ctx)
        assert result.row is None

    def test_invalid_config_no_condition(self) -> None:
        """Error when no condition is specified."""
        from elspeth.plugins.transforms.filter import Filter

        with pytest.raises(ValueError, match="condition"):
            Filter({"field": "status"})  # No equals, greater_than, etc.
