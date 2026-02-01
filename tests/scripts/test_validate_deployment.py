"""Tests for deployment validation script."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.validate_deployment import (
    _validate_field_normalization_at_path,
    validate_field_normalization_deployment,
)


class TestValidateDeployment:
    """Tests for deployment validation."""

    def test_validate_field_normalization_deployment_passes_when_complete(self) -> None:
        """Validation passes when all components present."""
        # Should not raise when all components exist (current state)
        validate_field_normalization_deployment()

    def test_validate_passes_when_all_deployed(self, tmp_path: Path) -> None:
        """Validation passes when all components are deployed."""
        base = tmp_path / "src" / "elspeth"
        (base / "plugins" / "sources").mkdir(parents=True)
        (base / "core").mkdir(parents=True)

        # Deploy ALL components
        (base / "plugins" / "sources" / "field_normalization.py").write_text("# normalization module")
        (base / "plugins" / "config_base.py").write_text("class TabularSourceDataConfig:\n    pass\n")
        (base / "core" / "identifiers.py").write_text("# identifiers module")

        # Should not raise
        _validate_field_normalization_at_path(base)

    def test_validate_passes_when_none_deployed(self, tmp_path: Path) -> None:
        """Validation passes when no components are deployed (clean slate)."""
        base = tmp_path / "src" / "elspeth"
        (base / "plugins" / "sources").mkdir(parents=True)
        (base / "core").mkdir(parents=True)

        # Create empty config_base.py (no TabularSourceDataConfig)
        (base / "plugins" / "config_base.py").write_text("# empty config\n")

        # Should not raise - none deployed is a valid state
        _validate_field_normalization_at_path(base)

    def test_validate_fails_with_only_field_normalization(self, tmp_path: Path) -> None:
        """Validation fails when only field_normalization.py is deployed."""
        base = tmp_path / "src" / "elspeth"
        (base / "plugins" / "sources").mkdir(parents=True)
        (base / "core").mkdir(parents=True)

        # Deploy ONLY field_normalization.py
        (base / "plugins" / "sources" / "field_normalization.py").write_text("# normalization module")
        (base / "plugins" / "config_base.py").write_text("# empty config\n")
        # identifiers.py is missing

        with pytest.raises(RuntimeError) as exc_info:
            _validate_field_normalization_at_path(base)

        error_msg = str(exc_info.value)
        assert "DEPLOYMENT VIOLATION" in error_msg
        assert "field_normalization.py" in error_msg
        assert "TabularSourceDataConfig" in error_msg
        assert "core/identifiers.py" in error_msg

    def test_validate_fails_with_missing_field_normalization(self, tmp_path: Path) -> None:
        """Validation fails when field_normalization.py is missing."""
        base = tmp_path / "src" / "elspeth"
        (base / "plugins" / "sources").mkdir(parents=True)
        (base / "core").mkdir(parents=True)

        # Deploy TabularSourceDataConfig and identifiers.py but NOT field_normalization
        (base / "plugins" / "config_base.py").write_text("class TabularSourceDataConfig:\n    pass\n")
        (base / "core" / "identifiers.py").write_text("# identifiers module")
        # field_normalization.py is missing

        with pytest.raises(RuntimeError) as exc_info:
            _validate_field_normalization_at_path(base)

        error_msg = str(exc_info.value)
        assert "DEPLOYMENT VIOLATION" in error_msg
        # field_normalization.py should be in the missing set
        assert "field_normalization.py" in error_msg

    def test_validate_fails_with_missing_identifiers(self, tmp_path: Path) -> None:
        """Validation fails when identifiers.py is missing."""
        base = tmp_path / "src" / "elspeth"
        (base / "plugins" / "sources").mkdir(parents=True)
        (base / "core").mkdir(parents=True)

        # Deploy field_normalization.py and TabularSourceDataConfig but NOT identifiers
        (base / "plugins" / "sources" / "field_normalization.py").write_text("# normalization module")
        (base / "plugins" / "config_base.py").write_text("class TabularSourceDataConfig:\n    pass\n")
        # identifiers.py is missing

        with pytest.raises(RuntimeError) as exc_info:
            _validate_field_normalization_at_path(base)

        error_msg = str(exc_info.value)
        assert "DEPLOYMENT VIOLATION" in error_msg
        assert "core/identifiers.py" in error_msg

    def test_validate_fails_with_missing_tabular_config(self, tmp_path: Path) -> None:
        """Validation fails when TabularSourceDataConfig is missing."""
        base = tmp_path / "src" / "elspeth"
        (base / "plugins" / "sources").mkdir(parents=True)
        (base / "core").mkdir(parents=True)

        # Deploy field_normalization.py and identifiers but NOT TabularSourceDataConfig
        (base / "plugins" / "sources" / "field_normalization.py").write_text("# normalization module")
        (base / "plugins" / "config_base.py").write_text("# no config class\n")
        (base / "core" / "identifiers.py").write_text("# identifiers module")

        with pytest.raises(RuntimeError) as exc_info:
            _validate_field_normalization_at_path(base)

        error_msg = str(exc_info.value)
        assert "DEPLOYMENT VIOLATION" in error_msg
        assert "TabularSourceDataConfig" in error_msg

    def test_error_message_lists_deployed_and_missing(self, tmp_path: Path) -> None:
        """Error message clearly lists both deployed and missing components."""
        base = tmp_path / "src" / "elspeth"
        (base / "plugins" / "sources").mkdir(parents=True)
        (base / "core").mkdir(parents=True)

        # Deploy only identifiers.py
        (base / "plugins" / "config_base.py").write_text("# empty\n")
        (base / "core" / "identifiers.py").write_text("# identifiers module")

        with pytest.raises(RuntimeError) as exc_info:
            _validate_field_normalization_at_path(base)

        error_msg = str(exc_info.value)
        # Should list what is deployed
        assert "Deployed:" in error_msg
        assert "core/identifiers.py" in error_msg
        # Should list what is missing
        assert "Missing:" in error_msg
        assert "field_normalization.py" in error_msg
        assert "TabularSourceDataConfig" in error_msg
        # Should explain the risk
        assert "AUDIT TRAIL CORRUPTION" in error_msg
