# tests/unit/plugins/test_context_types.py
"""Tests for PluginContext type alignment.

Migrated from tests/plugins/test_context_types.py.
Tests that require LandscapeDB (test_landscape_type_matches_recorder) are
deferred to integration tier.
"""


class TestPluginContextTypes:
    """Verify PluginContext field types match runtime values."""

    def test_context_landscape_annotation_references_plugin_audit_writer(self) -> None:
        """PluginContext.landscape annotation should reference PluginAuditWriter.

        The annotation should reference elspeth.contracts.audit_protocols.PluginAuditWriter,
        not a stub protocol or the concrete LandscapeRecorder. This test verifies the fix is in place.
        """
        import dataclasses

        from elspeth.contracts.plugin_context import PluginContext

        # Get the field annotations via dataclass introspection
        fields = {f.name: f for f in dataclasses.fields(PluginContext)}
        landscape_field = fields.get("landscape")

        assert landscape_field is not None, "PluginContext should have a 'landscape' field"

        # The type annotation should reference PluginAuditWriter
        annotations = PluginContext.__annotations__
        landscape_annotation = annotations.get("landscape")

        assert landscape_annotation is not None
        assert "PluginAuditWriter" in str(landscape_annotation)

    def test_no_stub_protocol_in_context_module(self) -> None:
        """No stub LandscapeRecorder protocol should exist in plugin_context.py.

        After the migration, plugin_context.py should import PluginAuditWriter
        from contracts, not define any local stub.
        """
        import elspeth.contracts.plugin_context as context_module

        # There should be no locally-defined LandscapeRecorder class
        # that shadows the real one
        local_items = dir(context_module)

        # The module should NOT have its own LandscapeRecorder class defined
        for name in local_items:
            obj = getattr(context_module, name)
            is_local_recorder = (
                isinstance(obj, type) and name == "LandscapeRecorder" and obj.__module__ == "elspeth.contracts.plugin_context"
            )
            if is_local_recorder:
                raise AssertionError(
                    "context.py should not define its own LandscapeRecorder stub. It should import PluginAuditWriter from contracts"
                )
