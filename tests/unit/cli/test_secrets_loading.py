# tests/cli/test_secrets_loading.py
"""Tests for CLI secret loading integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


class TestCLISecretsLoading:
    """Tests for secret loading in CLI run command."""

    def test_run_with_keyvault_secrets_loads_before_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Secrets should be loaded before config resolution."""
        from elspeth.cli import app

        # Set fingerprint key required for Key Vault secret loading
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        # Create a settings file that references a secret
        # NOTE: vault_url must be literal per P1-6
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
secrets:
  source: keyvault
  vault_url: https://test-vault.vault.azure.net
  mapping:
    TEST_SECRET: test-secret

sources:
  primary:
    plugin: null
    options:
      on_success: output

sinks:
  output:
    plugin: null

""")

        with patch("elspeth.core.security.secret_loader.KeyVaultSecretLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_loader.get_secret.return_value = ("loaded-from-keyvault", MagicMock())
            MockLoader.return_value = mock_loader

            # Run with --dry-run to avoid full execution
            runner.invoke(app, ["--no-dotenv", "run", "--settings", str(settings_file), "--dry-run"])

            # Verify secret was loaded
            mock_loader.get_secret.assert_called_once_with("test-secret")

    def test_run_with_keyvault_failure_exits_with_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Key Vault failure should exit with clear error message."""
        from elspeth.cli import app
        from elspeth.core.security.secret_loader import SecretNotFoundError

        # Set fingerprint key required for Key Vault secret loading
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
secrets:
  source: keyvault
  vault_url: https://test-vault.vault.azure.net
  mapping:
    MISSING_SECRET: nonexistent-secret

sources:
  primary:
    plugin: null
    options:
      on_success: output

sinks:
  output:
    plugin: null

""")

        with patch("elspeth.core.security.secret_loader.KeyVaultSecretLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_loader.get_secret.side_effect = SecretNotFoundError("Not found")
            MockLoader.return_value = mock_loader

            result = runner.invoke(app, ["--no-dotenv", "run", "--settings", str(settings_file), "--dry-run"])

            assert result.exit_code == 1
            assert "nonexistent-secret" in result.output or "Secret" in result.output

    def test_run_with_env_source_skips_keyvault(self, tmp_path: Path) -> None:
        """source: env should not attempt Key Vault connection."""
        from elspeth.cli import app

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
secrets:
  source: env

sources:
  primary:
    plugin: null
    options:
      on_success: output

sinks:
  output:
    plugin: null

""")

        with patch("elspeth.core.security.secret_loader.KeyVaultSecretLoader") as MockLoader:
            runner.invoke(app, ["--no-dotenv", "run", "--settings", str(settings_file), "--dry-run"])

            # Key Vault client should NOT be called
            MockLoader.assert_not_called()

    def test_run_with_secrets_validation_error_shows_friendly_message(self, tmp_path: Path) -> None:
        """Invalid secrets config should show user-friendly error."""
        from elspeth.cli import app

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
secrets:
  source: keyvault
  vault_url: http://not-https.vault.azure.net
  mapping:
    KEY: secret

sources:
  primary:
    plugin: null
    options:
      on_success: output

sinks:
  output:
    plugin: null

""")

        result = runner.invoke(app, ["--no-dotenv", "run", "--settings", str(settings_file), "--dry-run"])

        assert result.exit_code == 1
        assert "HTTPS" in result.output or "secrets" in result.output.lower()

    def test_run_with_missing_vault_url_shows_error(self, tmp_path: Path) -> None:
        """Missing vault_url when source is keyvault should show error."""
        from elspeth.cli import app

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
secrets:
  source: keyvault
  mapping:
    KEY: secret

sources:
  primary:
    plugin: null
    options:
      on_success: output

sinks:
  output:
    plugin: null

""")

        result = runner.invoke(app, ["--no-dotenv", "run", "--settings", str(settings_file), "--dry-run"])

        assert result.exit_code == 1
        assert "vault_url" in result.output.lower() or "secrets" in result.output.lower()

    def test_run_with_default_env_source_no_secrets_section(self, tmp_path: Path) -> None:
        """Omitting secrets section should default to env source."""
        from elspeth.cli import app

        # No secrets section at all - should default to source: env
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
sources:
  primary:
    plugin: null
    options:
      on_success: output

sinks:
  output:
    plugin: null

""")

        with patch("elspeth.core.security.secret_loader.KeyVaultSecretLoader") as MockLoader:
            runner.invoke(app, ["--no-dotenv", "run", "--settings", str(settings_file), "--dry-run"])

            # Key Vault client should NOT be called (env is default)
            MockLoader.assert_not_called()
            # Should at least not fail on secrets loading
            # (may fail elsewhere, but that's fine for this test)

    def test_run_with_env_var_in_vault_url_rejected(self, tmp_path: Path) -> None:
        """Environment variable reference in vault_url should be rejected."""
        from elspeth.cli import app

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
secrets:
  source: keyvault
  vault_url: ${AZURE_KEYVAULT_URL}
  mapping:
    KEY: secret

sources:
  primary:
    plugin: null
    options:
      on_success: output

sinks:
  output:
    plugin: null

""")

        result = runner.invoke(app, ["--no-dotenv", "run", "--settings", str(settings_file), "--dry-run"])

        assert result.exit_code == 1
        # Should mention that ${VAR} is not allowed
        assert "${" in result.output or "VAR" in result.output or "variable" in result.output.lower()
