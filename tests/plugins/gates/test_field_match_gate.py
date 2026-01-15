"""Tests for FieldMatchGate.

FieldMatchGate returns route LABELS, not sink names.
Config uses 'matches' (field_value -> route_label) instead of 'routes'.
The routes config in settings.yaml maps labels to sinks.
"""

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

        gate = FieldMatchGate(
            {
                "field": "status",
                "matches": {"active": "active_route", "deleted": "archive_route"},
            }
        )
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

        gate = FieldMatchGate(
            {
                "field": "status",
                "matches": {
                    "active": "active_route",
                    "pending": "pending_route",
                    "deleted": "archive_route",
                },
            }
        )

        active_row = {"id": 1, "status": "active"}
        result = gate.evaluate(active_row, ctx)
        assert result.action.kind == "route"
        assert result.action.destinations == ("active_route",)  # Route label

        pending_row = {"id": 2, "status": "pending"}
        result = gate.evaluate(pending_row, ctx)
        assert result.action.destinations == ("pending_route",)

    def test_no_match_uses_default_label(self, ctx: PluginContext) -> None:
        """Use default_label when no match found."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate(
            {
                "field": "status",
                "matches": {"active": "active_route"},
                "default_label": "other",
            }
        )
        row = {"id": 1, "status": "unknown"}

        result = gate.evaluate(row, ctx)

        assert result.action.kind == "route"
        assert result.action.destinations == ("other",)

    def test_no_match_without_default_uses_no_match(self, ctx: PluginContext) -> None:
        """Use 'no_match' label when no default_label specified."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate(
            {
                "field": "status",
                "matches": {"active": "active_route"},
            }
        )
        row = {"id": 1, "status": "unknown"}

        result = gate.evaluate(row, ctx)

        assert result.action.kind == "route"
        assert result.action.destinations == ("no_match",)

    def test_regex_route_matching(self, ctx: PluginContext) -> None:
        """Route based on regex pattern match."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate(
            {
                "field": "email",
                "mode": "regex",
                "matches": {
                    r".*@example\.com$": "internal",
                    r".*@partner\.org$": "partner",
                },
            }
        )

        internal_row = {"id": 1, "email": "alice@example.com"}
        result = gate.evaluate(internal_row, ctx)
        assert result.action.destinations == ("internal",)

        partner_row = {"id": 2, "email": "bob@partner.org"}
        result = gate.evaluate(partner_row, ctx)
        assert result.action.destinations == ("partner",)

    def test_list_values_in_matches(self, ctx: PluginContext) -> None:
        """Route multiple values to same label."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate(
            {
                "field": "country",
                "matches": {
                    "US,CA,MX": "north_america",  # Comma-separated
                    "UK,FR,DE": "europe",
                },
            }
        )

        us_row = {"id": 1, "country": "US"}
        result = gate.evaluate(us_row, ctx)
        assert result.action.destinations == ("north_america",)

        uk_row = {"id": 2, "country": "UK"}
        result = gate.evaluate(uk_row, ctx)
        assert result.action.destinations == ("europe",)

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Access nested field with dot notation."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate(
            {
                "field": "meta.type",
                "matches": {"internal": "internal_route"},
            }
        )
        row = {"id": 1, "meta": {"type": "internal"}}

        result = gate.evaluate(row, ctx)
        assert result.action.destinations == ("internal_route",)

    def test_missing_field_uses_default_label(self, ctx: PluginContext) -> None:
        """Use default_label when field is missing."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate(
            {
                "field": "status",
                "matches": {"active": "active_route"},
                "default_label": "missing_field",
            }
        )
        row = {"id": 1}  # No status field

        result = gate.evaluate(row, ctx)
        assert result.action.kind == "route"
        assert result.action.destinations == ("missing_field",)

    def test_strict_missing_field_raises_error(self, ctx: PluginContext) -> None:
        """Error when field is missing in strict mode."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate(
            {
                "field": "status",
                "matches": {"active": "active_route"},
                "strict": True,
            }
        )
        row = {"id": 1}

        with pytest.raises(ValueError, match="status"):
            gate.evaluate(row, ctx)

    def test_case_insensitive_matching(self, ctx: PluginContext) -> None:
        """Case-insensitive matching when configured."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate(
            {
                "field": "status",
                "matches": {"active": "active_route"},
                "case_insensitive": True,
            }
        )

        upper_row = {"id": 1, "status": "ACTIVE"}
        result = gate.evaluate(upper_row, ctx)
        assert result.action.destinations == ("active_route",)

        mixed_row = {"id": 2, "status": "Active"}
        result = gate.evaluate(mixed_row, ctx)
        assert result.action.destinations == ("active_route",)

    def test_routing_includes_reason(self, ctx: PluginContext) -> None:
        """RoutingAction includes reason with match details."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate

        gate = FieldMatchGate(
            {
                "field": "status",
                "matches": {"active": "active_route"},
            }
        )
        row = {"id": 1, "status": "active"}

        result = gate.evaluate(row, ctx)

        assert "field" in result.action.reason
        assert "value" in result.action.reason
        assert result.action.reason["value"] == "active"
