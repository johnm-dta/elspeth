"""Tests ensuring all built-in plugins have proper metadata for audit trail.

Per CLAUDE.md auditability standard: every decision must be traceable to source
data, configuration, AND code version. The plugin_version attribute is critical
for reproducibility - it ties audit records to specific plugin implementations.

This test prevents regression where new plugins are added without plugin_version,
causing audit records to show the placeholder "0.0.0" version.

Bug reference: P3-2026-01-21-sources-missing-plugin-version
"""


class TestBuiltinSourceMetadata:
    """Verify all built-in source plugins have audit-required metadata."""

    def test_csv_source_has_plugin_version(self) -> None:
        """CSVSource must have non-default plugin_version."""
        from elspeth.plugins.sources.csv_source import CSVSource

        assert hasattr(CSVSource, "plugin_version")
        assert isinstance(CSVSource.plugin_version, str)
        assert CSVSource.plugin_version != "0.0.0", "CSVSource has placeholder version"

    def test_json_source_has_plugin_version(self) -> None:
        """JSONSource must have non-default plugin_version."""
        from elspeth.plugins.sources.json_source import JSONSource

        assert hasattr(JSONSource, "plugin_version")
        assert isinstance(JSONSource.plugin_version, str)
        assert JSONSource.plugin_version != "0.0.0", "JSONSource has placeholder version"

    def test_null_source_has_plugin_version(self) -> None:
        """NullSource must have non-default plugin_version."""
        from elspeth.plugins.sources.null_source import NullSource

        assert hasattr(NullSource, "plugin_version")
        assert isinstance(NullSource.plugin_version, str)
        assert NullSource.plugin_version != "0.0.0", "NullSource has placeholder version"


class TestBuiltinSinkMetadata:
    """Verify all built-in sink plugins have audit-required metadata."""

    def test_csv_sink_has_plugin_version(self) -> None:
        """CSVSink must have non-default plugin_version."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        assert hasattr(CSVSink, "plugin_version")
        assert isinstance(CSVSink.plugin_version, str)
        assert CSVSink.plugin_version != "0.0.0", "CSVSink has placeholder version"

    def test_json_sink_has_plugin_version(self) -> None:
        """JSONSink must have non-default plugin_version."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        assert hasattr(JSONSink, "plugin_version")
        assert isinstance(JSONSink.plugin_version, str)
        assert JSONSink.plugin_version != "0.0.0", "JSONSink has placeholder version"

    def test_database_sink_has_plugin_version(self) -> None:
        """DatabaseSink must have non-default plugin_version."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        assert hasattr(DatabaseSink, "plugin_version")
        assert isinstance(DatabaseSink.plugin_version, str)
        assert DatabaseSink.plugin_version != "0.0.0", "DatabaseSink has placeholder version"


class TestBuiltinTransformMetadata:
    """Verify all built-in transform plugins have audit-required metadata."""

    def test_passthrough_has_plugin_version(self) -> None:
        """PassThrough must have non-default plugin_version."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        assert hasattr(PassThrough, "plugin_version")
        assert isinstance(PassThrough.plugin_version, str)
        assert PassThrough.plugin_version != "0.0.0", "PassThrough has placeholder version"

    def test_field_mapper_has_plugin_version(self) -> None:
        """FieldMapper must have non-default plugin_version."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        assert hasattr(FieldMapper, "plugin_version")
        assert isinstance(FieldMapper.plugin_version, str)
        assert FieldMapper.plugin_version != "0.0.0", "FieldMapper has placeholder version"

    def test_truncate_has_plugin_version(self) -> None:
        """Truncate must have non-default plugin_version."""
        from elspeth.plugins.transforms.truncate import Truncate

        assert hasattr(Truncate, "plugin_version")
        assert isinstance(Truncate.plugin_version, str)
        assert Truncate.plugin_version != "0.0.0", "Truncate has placeholder version"

    def test_batch_stats_has_plugin_version(self) -> None:
        """BatchStats must have non-default plugin_version."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        assert hasattr(BatchStats, "plugin_version")
        assert isinstance(BatchStats.plugin_version, str)
        assert BatchStats.plugin_version != "0.0.0", "BatchStats has placeholder version"

    def test_batch_replicate_has_plugin_version(self) -> None:
        """BatchReplicate must have non-default plugin_version."""
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate

        assert hasattr(BatchReplicate, "plugin_version")
        assert isinstance(BatchReplicate.plugin_version, str)
        assert BatchReplicate.plugin_version != "0.0.0", "BatchReplicate has placeholder version"

    def test_json_explode_has_plugin_version(self) -> None:
        """JSONExplode must have non-default plugin_version."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        assert hasattr(JSONExplode, "plugin_version")
        assert isinstance(JSONExplode.plugin_version, str)
        assert JSONExplode.plugin_version != "0.0.0", "JSONExplode has placeholder version"

    def test_keyword_filter_has_plugin_version(self) -> None:
        """KeywordFilter must have non-default plugin_version."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        assert hasattr(KeywordFilter, "plugin_version")
        assert isinstance(KeywordFilter.plugin_version, str)
        assert KeywordFilter.plugin_version != "0.0.0", "KeywordFilter has placeholder version"
