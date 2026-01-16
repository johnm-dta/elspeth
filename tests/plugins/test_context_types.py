# tests/plugins/test_context_types.py
"""Tests for PluginContext type alignment."""


class TestPluginContextTypes:
    """Verify PluginContext field types match runtime values."""

    def test_landscape_type_matches_recorder(self) -> None:
        """PluginContext.landscape type should accept LandscapeRecorder."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.plugins.context import PluginContext

        # Create a real LandscapeRecorder
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Should be able to create context with real recorder
        # (This tests runtime compatibility)
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=recorder,
        )

        assert ctx.landscape is recorder

    def test_context_landscape_annotation_references_real_recorder(self) -> None:
        """PluginContext.landscape annotation should reference the real LandscapeRecorder.

        The annotation string should reference elspeth.core.landscape.recorder.LandscapeRecorder,
        not a stub protocol. This test verifies the fix is in place.
        """
        import dataclasses

        from elspeth.plugins.context import PluginContext

        # Get the field annotations via dataclass introspection
        fields = {f.name: f for f in dataclasses.fields(PluginContext)}
        landscape_field = fields.get("landscape")

        assert (
            landscape_field is not None
        ), "PluginContext should have a 'landscape' field"

        # The type annotation should reference the real LandscapeRecorder
        # After the fix, __annotations__ should contain 'LandscapeRecorder | None'
        annotations = PluginContext.__annotations__
        landscape_annotation = annotations.get("landscape")

        # Should be a string annotation that contains LandscapeRecorder
        assert landscape_annotation is not None
        assert "LandscapeRecorder" in str(landscape_annotation)

    def test_no_stub_protocol_in_context_module(self) -> None:
        """The stub LandscapeRecorder protocol should be removed from context.py.

        After the fix, context.py should import the real LandscapeRecorder
        (in TYPE_CHECKING block), not define a stub protocol.
        """
        import elspeth.plugins.context as context_module

        # There should be no locally-defined LandscapeRecorder class
        # that shadows the real one
        local_items = dir(context_module)

        # The module should NOT have its own LandscapeRecorder class defined
        # Check if any class defined in this module is named LandscapeRecorder
        for name in local_items:
            obj = getattr(context_module, name)
            # Check if it's a class named LandscapeRecorder defined in this module
            is_local_recorder = (
                isinstance(obj, type)
                and name == "LandscapeRecorder"
                and obj.__module__ == "elspeth.plugins.context"
            )
            if is_local_recorder:
                raise AssertionError(
                    "context.py should not define its own LandscapeRecorder stub. "
                    "It should import from elspeth.core.landscape.recorder"
                )
