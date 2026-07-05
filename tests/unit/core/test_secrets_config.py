# tests/core/test_secrets_config.py
"""Tests for SecretsConfig Pydantic model validation."""

import pytest
from pydantic import ValidationError


class TestSecretsConfigValidation:
    """Tests for SecretsConfig schema validation."""

    def test_env_source_requires_no_additional_fields(self) -> None:
        """source: env should work with no other fields."""
        from elspeth.core.config import SecretsConfig

        config = SecretsConfig(source="env")
        assert config.source == "env"
        assert config.vault_url is None
        assert config.mapping == {}

    def test_keyvault_source_requires_vault_url(self) -> None:
        """source: keyvault must have vault_url."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="vault_url is required"):
            SecretsConfig(source="keyvault", mapping={"KEY": "key"})

    def test_keyvault_source_requires_mapping(self) -> None:
        """source: keyvault must have non-empty mapping."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="mapping is required"):
            SecretsConfig(
                source="keyvault",
                vault_url="https://my-vault.vault.azure.net",
                mapping={},
            )

    def test_keyvault_source_valid_config(self) -> None:
        """Valid keyvault config passes validation."""
        from elspeth.core.config import SecretsConfig

        config = SecretsConfig(
            source="keyvault",
            vault_url="https://my-vault.vault.azure.net",
            mapping={
                "AZURE_OPENAI_KEY": "azure-openai-key",
                "AZURE_OPENAI_ENDPOINT": "openai-endpoint",
            },
        )
        assert config.source == "keyvault"
        assert config.vault_url == "https://my-vault.vault.azure.net"
        assert len(config.mapping) == 2

    @pytest.mark.parametrize(
        "bad_name",
        ["BAD=NAME", "", "1STARTS_WITH_DIGIT", "HAS SPACE", "HAS-DASH", "HAS.DOT", "NUL\x00BYTE", "TRAILING\n", "\nLEADING"],
    )
    def test_keyvault_mapping_rejects_invalid_env_var_names(self, bad_name: str) -> None:
        """Invalid env-var names in the mapping are rejected at config time.

        Regression for elspeth-1afd07cb77: mapping keys become os.environ
        assignments in the sequential apply phase; an invalid name ('BAD=NAME'
        -> ValueError, '' -> OSError) would fail late, after a valid earlier
        entry had already mutated process env. Reject mechanically before any
        Key Vault I/O.
        """
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="not a valid environment"):
            SecretsConfig(
                source="keyvault",
                vault_url="https://my-vault.vault.azure.net",
                mapping={bad_name: "secret"},
            )

    def test_keyvault_mapping_accepts_valid_env_var_names(self) -> None:
        """POSIX-style env-var names (leading letter/underscore) are accepted."""
        from elspeth.core.config import SecretsConfig

        config = SecretsConfig(
            source="keyvault",
            vault_url="https://my-vault.vault.azure.net",
            mapping={"AZURE_OPENAI_KEY": "k1", "_PRIVATE": "k2", "K3_v2": "k3"},
        )
        assert len(config.mapping) == 3

    def test_invalid_source_rejected(self) -> None:
        """Invalid source value is rejected."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="Input should be 'env' or 'keyvault'"):
            SecretsConfig(source="invalid")

    def test_default_source_is_env(self) -> None:
        """Default source is 'env' when not specified."""
        from elspeth.core.config import SecretsConfig

        config = SecretsConfig()
        assert config.source == "env"

    # P0-3: Vault URL format validation tests
    def test_vault_url_must_be_https(self) -> None:
        """vault_url must use HTTPS protocol."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="must use HTTPS"):
            SecretsConfig(
                source="keyvault",
                vault_url="http://my-vault.vault.azure.net",  # HTTP not allowed
                mapping={"KEY": "key"},
            )

    def test_vault_url_rejects_env_var_reference(self) -> None:
        """vault_url cannot contain ${VAR} references (chicken-egg problem)."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match=r"cannot contain.*\$\{"):
            SecretsConfig(
                source="keyvault",
                vault_url="${AZURE_KEYVAULT_URL}",  # Not allowed
                mapping={"KEY": "key"},
            )

    def test_vault_url_rejects_malformed_url(self) -> None:
        """vault_url must be a valid URL."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="Invalid URL"):
            SecretsConfig(
                source="keyvault",
                vault_url="not-a-valid-url",
                mapping={"KEY": "key"},
            )

    def test_vault_url_with_trailing_slash_normalized(self) -> None:
        """vault_url with trailing slash should be accepted."""
        from elspeth.core.config import SecretsConfig

        config = SecretsConfig(
            source="keyvault",
            vault_url="https://my-vault.vault.azure.net/",
            mapping={"KEY": "key"},
        )
        # Trailing slash should be stripped for consistency
        assert config.vault_url == "https://my-vault.vault.azure.net"

    def test_vault_url_rejects_non_string(self) -> None:
        """Non-string vault_url should be rejected with type error."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="str"):
            SecretsConfig(
                source="keyvault",
                vault_url=123,  # Integer instead of string
                mapping={"KEY": "key"},
            )


class TestVaultUrlSSRFHardening:
    """SSRF / credential-boundary hardening for vault_url (elspeth-7572facbc6).

    A well-formed Azure Key Vault endpoint carries no userinfo, no non-standard
    port, no path/query/fragment, and its host is an approved Key Vault suffix.
    These checks stop an operator settings file from aiming a
    DefaultAzureCredential-backed client at an arbitrary HTTPS host.
    """

    def test_vault_url_rejects_userinfo(self) -> None:
        """Embedded user:pass@ is rejected (credential-in-URL / host confusion)."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="userinfo"):
            SecretsConfig(
                source="keyvault",
                vault_url="https://user:pass@my-vault.vault.azure.net",
                mapping={"KEY": "key"},
            )

    def test_vault_url_rejects_unexpected_port(self) -> None:
        """A non-443 port is rejected."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="port"):
            SecretsConfig(
                source="keyvault",
                vault_url="https://my-vault.vault.azure.net:8443",
                mapping={"KEY": "key"},
            )

    def test_vault_url_allows_explicit_standard_port(self) -> None:
        """An explicit :443 is the standard HTTPS port and is accepted."""
        from elspeth.core.config import SecretsConfig

        config = SecretsConfig(
            source="keyvault",
            vault_url="https://my-vault.vault.azure.net:443",
            mapping={"KEY": "key"},
        )
        assert config.vault_url == "https://my-vault.vault.azure.net:443"

    def test_vault_url_rejects_path(self) -> None:
        """A path component is rejected — a vault endpoint is host-only."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="path"):
            SecretsConfig(
                source="keyvault",
                vault_url="https://my-vault.vault.azure.net/secrets/steal",
                mapping={"KEY": "key"},
            )

    def test_vault_url_rejects_query_string(self) -> None:
        """A query string is rejected."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="query"):
            SecretsConfig(
                source="keyvault",
                vault_url="https://my-vault.vault.azure.net?x=1",
                mapping={"KEY": "key"},
            )

    def test_vault_url_rejects_fragment(self) -> None:
        """A fragment is rejected."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="fragment"):
            SecretsConfig(
                source="keyvault",
                vault_url="https://my-vault.vault.azure.net#frag",
                mapping={"KEY": "key"},
            )

    def test_vault_url_rejects_non_keyvault_host(self) -> None:
        """A host outside the approved Key Vault suffixes is rejected (SSRF)."""
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="approved Azure Key Vault"):
            SecretsConfig(
                source="keyvault",
                vault_url="https://evil.example.com",
                mapping={"KEY": "key"},
            )

    def test_vault_url_rejects_suffix_lookalike(self) -> None:
        """A host that only embeds the suffix as a substring is rejected.

        ``my-vault.vault.azure.net.attacker.com`` ends with ``.attacker.com`` —
        the leading-dot suffix match must not be fooled.
        """
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="approved Azure Key Vault"):
            SecretsConfig(
                source="keyvault",
                vault_url="https://my-vault.vault.azure.net.attacker.com",
                mapping={"KEY": "key"},
            )

    @pytest.mark.parametrize(
        "url",
        [
            "https://my-vault.vault.azure.net",  # public cloud
            "https://my-vault.vault.azure.cn",  # Azure China (Mooncake)
            "https://my-vault.vault.usgovcloudapi.net",  # US Government
            "https://my-vault.vault.microsoftazure.de",  # legacy Germany
        ],
    )
    def test_vault_url_accepts_sovereign_cloud_suffixes(self, url: str) -> None:
        """All sovereign-cloud Key Vault suffixes are accepted."""
        from elspeth.core.config import SecretsConfig

        config = SecretsConfig(source="keyvault", vault_url=url, mapping={"KEY": "key"})
        assert config.vault_url == url

    @pytest.mark.parametrize(
        "raw",
        [
            " https://my-vault.vault.azure.net",  # leading space
            "\thttps://my-vault.vault.azure.net",  # leading tab
            "https://my-vault.vault.azure.net\t",  # trailing tab
            "  https://my-vault.vault.azure.net  ",  # both ends
            "https://my-vault.vault.azure.net/ ",  # trailing slash + space
        ],
    )
    def test_vault_url_strips_surrounding_whitespace(self, raw: str) -> None:
        """Edge whitespace is normalised so the stored value == the validated value.

        urlparse drops leading whitespace (and a trailing tab) while parsing, so
        these inputs would otherwise pass validation on their parsed form but be
        persisted — and handed to the Key Vault client — with the whitespace
        intact. The validator must strip first (elspeth-7572facbc6).
        """
        from elspeth.core.config import SecretsConfig

        config = SecretsConfig(source="keyvault", vault_url=raw, mapping={"KEY": "key"})
        assert config.vault_url == "https://my-vault.vault.azure.net"

    def test_vault_url_rejects_bare_trailing_dot_fqdn(self) -> None:
        """A trailing-dot FQDN fails closed rather than matching the suffix.

        ``my-vault.vault.azure.net.`` is a valid absolute-DNS spelling of the real
        vault, but the leading-dot suffix check does not match it, so it is
        rejected — the operator drops the dot. This locks the commit's
        "trailing-dot FQDNs fail closed" claim (elspeth-7572facbc6), distinct from
        the look-alike ``…azure.net.attacker.com`` case above.
        """
        from elspeth.core.config import SecretsConfig

        with pytest.raises(ValidationError, match="approved Azure Key Vault"):
            SecretsConfig(
                source="keyvault",
                vault_url="https://my-vault.vault.azure.net.",
                mapping={"KEY": "key"},
            )
