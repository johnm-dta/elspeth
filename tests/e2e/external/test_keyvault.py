"""Live E2E tests for Azure Key Vault secret resolution."""

from __future__ import annotations

import os

import pytest

_KEYVAULT_URL = os.environ.get("TEST_KEYVAULT_URL")

pytestmark = pytest.mark.e2e

_FINGERPRINT_SECRET_NAME = os.environ.get("TEST_KEYVAULT_FINGERPRINT_SECRET_NAME", "elspeth-fingerprint-key")
_MISSING_SECRET_NAME = os.environ.get("TEST_KEYVAULT_MISSING_SECRET_NAME", "elspeth-e2e-secret-that-should-not-exist")


@pytest.fixture(scope="module", autouse=True)
def _keyvault_prerequisites() -> None:
    if not _KEYVAULT_URL:
        pytest.skip("TEST_KEYVAULT_URL not set — requires Azure Key Vault credentials")
    pytest.importorskip("azure.keyvault.secrets")
    pytest.importorskip("azure.identity")


def _fingerprint_key_config():
    from elspeth.core.config import SecretsConfig

    assert _KEYVAULT_URL is not None
    return SecretsConfig(
        source="keyvault",
        vault_url=_KEYVAULT_URL,
        mapping={
            "ELSPETH_FINGERPRINT_KEY": _FINGERPRINT_SECRET_NAME,
        },
    )


class TestKeyVault:
    """Azure Key Vault secret resolution E2E tests."""

    def test_keyvault_secret_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Key Vault secrets can be resolved and loaded into config."""
        from elspeth.core.security.config_secrets import load_secrets_from_config

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

        resolutions = load_secrets_from_config(_fingerprint_key_config())

        assert os.environ.get("ELSPETH_FINGERPRINT_KEY")
        assert len(resolutions) == 1
        assert resolutions[0].env_var_name == "ELSPETH_FINGERPRINT_KEY"
        assert resolutions[0].secret_name == _FINGERPRINT_SECRET_NAME
        assert resolutions[0].source == "keyvault"

    def test_keyvault_secret_fingerprinting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolved secrets are fingerprinted for audit trail storage."""
        from elspeth.core.security.config_secrets import load_secrets_from_config

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

        resolutions = load_secrets_from_config(_fingerprint_key_config())

        assert len(resolutions) == 1
        fingerprint = resolutions[0].fingerprint
        assert len(fingerprint) == 64
        assert set(fingerprint) <= set("0123456789abcdef")
        assert fingerprint != os.environ["ELSPETH_FINGERPRINT_KEY"]

    def test_keyvault_missing_secret_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing secrets in Key Vault produce clear error messages."""
        from elspeth.core.config import SecretsConfig
        from elspeth.core.security.config_secrets import SecretLoadError, load_secrets_from_config

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        assert _KEYVAULT_URL is not None
        config = SecretsConfig(
            source="keyvault",
            vault_url=_KEYVAULT_URL,
            mapping={
                "ELSPETH_FINGERPRINT_KEY": _FINGERPRINT_SECRET_NAME,
                "ELSPETH_E2E_MISSING_SECRET": _MISSING_SECRET_NAME,
            },
        )

        with pytest.raises(SecretLoadError, match=_MISSING_SECRET_NAME):
            load_secrets_from_config(config)

        assert os.environ.get("ELSPETH_E2E_MISSING_SECRET") is None

    def test_keyvault_resolution_recorded_in_landscape(
        self,
        system_landscape_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Secret resolutions are recorded in the Landscape audit trail."""
        from elspeth.core.security.config_secrets import load_secrets_from_config

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        run = system_landscape_factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        resolutions = load_secrets_from_config(_fingerprint_key_config())

        system_landscape_factory.run_lifecycle.record_secret_resolutions(run.run_id, resolutions)
        stored = system_landscape_factory.run_lifecycle.get_secret_resolutions_for_run(run.run_id)

        assert len(stored) == 1
        assert stored[0].run_id == run.run_id
        assert stored[0].env_var_name == "ELSPETH_FINGERPRINT_KEY"
        assert stored[0].source == "keyvault"
        assert stored[0].vault_url == _KEYVAULT_URL
        assert stored[0].secret_name == _FINGERPRINT_SECRET_NAME
        assert stored[0].fingerprint == resolutions[0].fingerprint
