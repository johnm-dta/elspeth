"""Tests for FieldMatchGate."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import GateProtocol


class TestFieldMatchGate:
    """Tests for FieldMatchGate plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """FieldMatchGate implements GateProtocol."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink", "deleted": "archive_sink"},
        })
        assert isinstance(gate, GateProtocol)

    def test_has_required_attributes(self) -> None:
        """FieldMatchGate has name and schemas."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        assert FieldMatchGate.name == "field_match_gate"
        assert hasattr(FieldMatchGate, "input_schema")
        assert hasattr(FieldMatchGate, "output_schema")

    def test_exact_match_routing(self, ctx: PluginContext) -> None:
        """Route based on exact field value match."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {
                "active": "active_sink",
                "pending": "pending_sink",
                "deleted": "archive_sink",
            },
        })

        active_row = {"id": 1, "status": "active"}
        result = gate.evaluate(active_row, ctx)
        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("active_sink",)

        pending_row = {"id": 2, "status": "pending"}
        result = gate.evaluate(pending_row, ctx)
        assert result.action.destinations == ("pending_sink",)

    def test_no_match_continues(self, ctx: PluginContext) -> None:
        """Continue to next transform when no route matches."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
        })
        row = {"id": 1, "status": "unknown"}

        result = gate.evaluate(row, ctx)

        assert result.action.kind == "continue"

    def test_default_route_on_no_match(self, ctx: PluginContext) -> None:
        """Use default_sink when no route matches."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
            "default_sink": "other_sink",
        })
        row = {"id": 1, "status": "unknown"}

        result = gate.evaluate(row, ctx)

        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("other_sink",)

    def test_regex_route_matching(self, ctx: PluginContext) -> None:
        """Route based on regex pattern match."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "email",
            "mode": "regex",
            "routes": {
                r".*@example\.com$": "internal_sink",
                r".*@partner\.org$": "partner_sink",
            },
        })

        internal_row = {"id": 1, "email": "alice@example.com"}
        result = gate.evaluate(internal_row, ctx)
        assert result.action.destinations == ("internal_sink",)

        partner_row = {"id": 2, "email": "bob@partner.org"}
        result = gate.evaluate(partner_row, ctx)
        assert result.action.destinations == ("partner_sink",)

    def test_list_values_in_routes(self, ctx: PluginContext) -> None:
        """Route multiple values to same sink."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "country",
            "routes": {
                "US,CA,MX": "north_america_sink",  # Comma-separated
                "UK,FR,DE": "europe_sink",
            },
        })

        us_row = {"id": 1, "country": "US"}
        result = gate.evaluate(us_row, ctx)
        assert result.action.destinations == ("north_america_sink",)

        uk_row = {"id": 2, "country": "UK"}
        result = gate.evaluate(uk_row, ctx)
        assert result.action.destinations == ("europe_sink",)

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Access nested field with dot notation."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "meta.type",
            "routes": {"internal": "internal_sink"},
        })
        row = {"id": 1, "meta": {"type": "internal"}}

        result = gate.evaluate(row, ctx)
        assert result.action.destinations == ("internal_sink",)

    def test_missing_field_continues(self, ctx: PluginContext) -> None:
        """Continue when field is missing (no error by default)."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
        })
        row = {"id": 1}  # No status field

        result = gate.evaluate(row, ctx)
        assert result.action.kind == "continue"

    def test_strict_missing_field_raises_error(self, ctx: PluginContext) -> None:
        """Error when field is missing in strict mode."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
            "strict": True,
        })
        row = {"id": 1}

        with pytest.raises(ValueError, match="status"):
            gate.evaluate(row, ctx)

    def test_case_insensitive_matching(self, ctx: PluginContext) -> None:
        """Case-insensitive matching when configured."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
            "case_insensitive": True,
        })

        upper_row = {"id": 1, "status": "ACTIVE"}
        result = gate.evaluate(upper_row, ctx)
        assert result.action.destinations == ("active_sink",)

        mixed_row = {"id": 2, "status": "Active"}
        result = gate.evaluate(mixed_row, ctx)
        assert result.action.destinations == ("active_sink",)

    def test_routing_includes_reason(self, ctx: PluginContext) -> None:
        """RoutingAction includes reason with match details."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate({
            "field": "status",
            "routes": {"active": "active_sink"},
        })
        row = {"id": 1, "status": "active"}

        result = gate.evaluate(row, ctx)

        assert "field" in result.action.reason
        assert "value" in result.action.reason
        assert result.action.reason["value"] == "active"
