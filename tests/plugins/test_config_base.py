# tests/plugins/test_config_base.py
"""Tests for plugin configuration base classes."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from elspeth.plugins.config_base import PathConfig, PluginConfig, PluginConfigError


class TestPluginConfig:
    """Tests for PluginConfig base class."""

    def test_rejects_extra_fields(self) -> None:
        """Extra fields should raise validation error."""

        class MyConfig(PluginConfig):
            name: str

        with pytest.raises(ValidationError) as exc_info:
            MyConfig(name="test", unknown_field="value")  # type: ignore[call-arg]

        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_from_dict_wraps_validation_error(self) -> None:
        """from_dict should wrap ValidationError in PluginConfigError."""

        class MyConfig(PluginConfig):
            required_field: str

        with pytest.raises(PluginConfigError) as exc_info:
            MyConfig.from_dict({})  # Missing required_field

        assert "Invalid configuration for MyConfig" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_from_dict_success(self) -> None:
        """from_dict should return valid config on success."""

        class MyConfig(PluginConfig):
            name: str
            count: int = 10

        cfg = MyConfig.from_dict({"name": "test"})

        assert cfg.name == "test"
        assert cfg.count == 10

    def test_from_dict_with_defaults(self) -> None:
        """from_dict should use default values when not provided."""

        class MyConfig(PluginConfig):
            required: str
            optional: str = "default_value"

        cfg = MyConfig.from_dict({"required": "provided"})

        assert cfg.required == "provided"
        assert cfg.optional == "default_value"

    def test_from_dict_rejects_extra_fields(self) -> None:
        """from_dict should reject extra fields via PluginConfigError."""

        class MyConfig(PluginConfig):
            name: str

        with pytest.raises(PluginConfigError) as exc_info:
            MyConfig.from_dict({"name": "test", "typo_field": "value"})

        assert "Invalid configuration for MyConfig" in str(exc_info.value)


class TestPathConfig:
    """Tests for PathConfig base class."""

    def test_rejects_empty_path(self) -> None:
        """Empty path should raise validation error."""

        class FileConfig(PathConfig):
            pass

        with pytest.raises(ValidationError) as exc_info:
            FileConfig(path="")

        assert "path cannot be empty" in str(exc_info.value)

    def test_rejects_whitespace_only_path(self) -> None:
        """Whitespace-only path should raise validation error."""

        class FileConfig(PathConfig):
            pass

        with pytest.raises(ValidationError) as exc_info:
            FileConfig(path="   ")

        assert "path cannot be empty" in str(exc_info.value)

    def test_accepts_valid_path(self) -> None:
        """Valid path should be accepted."""

        class FileConfig(PathConfig):
            pass

        cfg = FileConfig(path="/path/to/file.csv")

        assert cfg.path == "/path/to/file.csv"

    def test_resolved_path_absolute(self) -> None:
        """Absolute path should not change when resolved."""

        class FileConfig(PathConfig):
            pass

        cfg = FileConfig(path="/absolute/path.csv")
        result = cfg.resolved_path()

        assert result == Path("/absolute/path.csv")

    def test_resolved_path_absolute_ignores_base_dir(self) -> None:
        """Absolute path should ignore base_dir when provided."""

        class FileConfig(PathConfig):
            pass

        cfg = FileConfig(path="/absolute/path.csv")
        result = cfg.resolved_path(base_dir=Path("/other/base"))

        assert result == Path("/absolute/path.csv")

    def test_resolved_path_relative_without_base(self) -> None:
        """Relative path without base_dir should return as-is."""

        class FileConfig(PathConfig):
            pass

        cfg = FileConfig(path="relative/path.csv")
        result = cfg.resolved_path()

        assert result == Path("relative/path.csv")

    def test_resolved_path_relative_with_base(self) -> None:
        """Relative path should be resolved against base_dir."""

        class FileConfig(PathConfig):
            pass

        cfg = FileConfig(path="relative/path.csv")
        result = cfg.resolved_path(base_dir=Path("/base"))

        assert result == Path("/base/relative/path.csv")

    def test_path_config_with_additional_fields(self) -> None:
        """PathConfig subclass can have additional validated fields."""

        class CSVConfig(PathConfig):
            delimiter: str = ","
            encoding: str = "utf-8"

        cfg = CSVConfig(path="data.csv", delimiter=";", encoding="latin-1")

        assert cfg.path == "data.csv"
        assert cfg.delimiter == ";"
        assert cfg.encoding == "latin-1"

    def test_path_config_rejects_extra_fields(self) -> None:
        """PathConfig subclass should still reject extra fields."""

        class CSVConfig(PathConfig):
            delimiter: str = ","

        with pytest.raises(ValidationError):
            CSVConfig(path="data.csv", unknown="value")  # type: ignore[call-arg]


class TestPluginConfigInheritance:
    """Tests for config inheritance patterns."""

    def test_deep_inheritance(self) -> None:
        """Multiple levels of inheritance should work correctly."""

        class MiddleConfig(PathConfig):
            compression: str = "none"

        class SpecificConfig(MiddleConfig):
            format_version: int = 1

        cfg = SpecificConfig(path="data.bin", compression="gzip", format_version=2)

        assert cfg.path == "data.bin"
        assert cfg.compression == "gzip"
        assert cfg.format_version == 2
        assert cfg.resolved_path() == Path("data.bin")

    def test_from_dict_returns_correct_subclass_type(self) -> None:
        """from_dict should return instance of the subclass, not base class."""

        class SpecificConfig(PathConfig):
            custom_field: str = "default"

        cfg = SpecificConfig.from_dict({"path": "test.csv", "custom_field": "custom"})

        assert isinstance(cfg, SpecificConfig)
        assert cfg.custom_field == "custom"
