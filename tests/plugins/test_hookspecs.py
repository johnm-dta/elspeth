# tests/plugins/test_hookspecs.py
"""Tests for pluggy hook specifications."""


class TestHookspecs:
    """pluggy hook specifications."""

    def test_hookspec_marker_exists(self) -> None:
        from elspeth.plugins.hookspecs import hookspec

        assert hookspec is not None

    def test_hookimpl_marker_exists(self) -> None:
        from elspeth.plugins.hookspecs import hookimpl

        assert hookimpl is not None

    def test_source_hooks_defined(self) -> None:
        from elspeth.plugins.hookspecs import ElspethSourceSpec

        # Check that hook methods exist
        assert hasattr(ElspethSourceSpec, "elspeth_get_source")

    def test_transform_hooks_defined(self) -> None:
        from elspeth.plugins.hookspecs import ElspethTransformSpec

        assert hasattr(ElspethTransformSpec, "elspeth_get_transforms")
        assert hasattr(ElspethTransformSpec, "elspeth_get_gates")

    def test_sink_hooks_defined(self) -> None:
        from elspeth.plugins.hookspecs import ElspethSinkSpec

        assert hasattr(ElspethSinkSpec, "elspeth_get_sinks")
