"""Tests for dynamic plugin discovery."""

from pathlib import Path

from elspeth.plugins.base import BaseSink, BaseSource, BaseTransform
from elspeth.plugins.discovery import discover_plugins_in_directory


class TestDiscoverPlugins:
    """Test plugin discovery from directories."""

    def test_discovers_csv_source(self) -> None:
        """Verify CSVSource is discovered in sources directory."""
        plugins_root = Path(__file__).parent.parent.parent / "src" / "elspeth" / "plugins"
        sources_dir = plugins_root / "sources"

        discovered = discover_plugins_in_directory(sources_dir, BaseSource)

        names = [cls.name for cls in discovered]
        assert "csv" in names, f"Expected 'csv' in {names}"

    def test_discovers_passthrough_transform(self) -> None:
        """Verify PassThrough is discovered in transforms directory."""
        plugins_root = Path(__file__).parent.parent.parent / "src" / "elspeth" / "plugins"
        transforms_dir = plugins_root / "transforms"

        discovered = discover_plugins_in_directory(transforms_dir, BaseTransform)

        names = [cls.name for cls in discovered]
        assert "passthrough" in names, f"Expected 'passthrough' in {names}"

    def test_discovers_csv_sink(self) -> None:
        """Verify CSVSink is discovered in sinks directory."""
        plugins_root = Path(__file__).parent.parent.parent / "src" / "elspeth" / "plugins"
        sinks_dir = plugins_root / "sinks"

        discovered = discover_plugins_in_directory(sinks_dir, BaseSink)

        names = [cls.name for cls in discovered]
        assert "csv" in names, f"Expected 'csv' in {names}"

    def test_excludes_non_plugin_files(self) -> None:
        """Verify __init__.py and base.py are not scanned for plugins."""
        plugins_root = Path(__file__).parent.parent.parent / "src" / "elspeth" / "plugins"
        sources_dir = plugins_root / "sources"

        discovered = discover_plugins_in_directory(sources_dir, BaseSource)

        # Should not crash or include base classes
        for cls in discovered:
            assert hasattr(cls, "name"), f"{cls} has no name attribute"
            assert cls.name != "", f"{cls} has empty name"

    def test_skips_abstract_classes(self) -> None:
        """Verify abstract base classes are not included."""
        plugins_root = Path(__file__).parent.parent.parent / "src" / "elspeth" / "plugins"
        sources_dir = plugins_root / "sources"

        discovered = discover_plugins_in_directory(sources_dir, BaseSource)

        class_names = [cls.__name__ for cls in discovered]
        assert "BaseSource" not in class_names
