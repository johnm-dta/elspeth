"""Tests for configuration contracts."""


class TestConfigReexports:
    """Verify config types are accessible from contracts."""

    def test_can_import_settings_from_contracts(self) -> None:
        """All settings types importable from contracts."""
        from elspeth.contracts import (
            DatasourceSettings,
            ElspethSettings,
        )

        # Just verify import works
        assert ElspethSettings is not None
        assert DatasourceSettings is not None

    def test_settings_are_pydantic_models(self) -> None:
        """Config types are Pydantic (trust boundary validation)."""
        from pydantic import BaseModel

        from elspeth.contracts import DatasourceSettings, ElspethSettings

        assert issubclass(ElspethSettings, BaseModel)
        assert issubclass(DatasourceSettings, BaseModel)

    def test_settings_are_frozen(self) -> None:
        """Config is immutable after construction."""
        import pytest
        from pydantic import ValidationError

        from elspeth.contracts import DatasourceSettings

        settings = DatasourceSettings(plugin="csv_local")

        with pytest.raises(ValidationError):
            settings.plugin = "other"  # type: ignore[misc]
