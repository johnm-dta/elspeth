"""Tests for NullSource - a source that yields nothing for resume operations."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SourceProtocol


class TestNullSource:
    """Tests for NullSource."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_null_source_yields_nothing(self, ctx: PluginContext) -> None:
        """NullSource.load() yields no rows."""
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})

        rows = list(source.load(ctx))

        assert rows == []

    def test_null_source_has_name(self) -> None:
        """NullSource has 'null' as its name."""
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        assert source.name == "null"

    def test_null_source_satisfies_protocol(self) -> None:
        """NullSource satisfies SourceProtocol."""
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        # This should not raise - source satisfies protocol
        assert isinstance(source, SourceProtocol)

    def test_null_source_has_output_schema(self) -> None:
        """NullSource has an output_schema attribute."""
        from elspeth.contracts import PluginSchema
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        assert hasattr(source, "output_schema")
        # output_schema must be a PluginSchema subclass
        assert issubclass(source.output_schema, PluginSchema)

    def test_null_source_close_is_idempotent(self) -> None:
        """close() can be called multiple times safely."""
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        source.close()
        source.close()  # Should not raise

    def test_null_source_has_determinism(self) -> None:
        """NullSource has appropriate determinism marking."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        # NullSource is deterministic - always yields nothing
        assert source.determinism == Determinism.DETERMINISTIC

    def test_null_source_has_plugin_version(self) -> None:
        """NullSource has a plugin_version."""
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({})
        assert hasattr(source, "plugin_version")
        assert isinstance(source.plugin_version, str)
        assert source.plugin_version != ""
