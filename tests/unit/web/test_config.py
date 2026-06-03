"""Tests for WebSettings configuration model."""

from __future__ import annotations

import types
import typing
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from elspeth.web.config import WebSettings


class TestWebSettingsValidation:
    """Tests for field validation."""

    def test_invalid_auth_provider_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                auth_provider="invalid",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_invalid_auth_provider_kerberos_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                auth_provider="kerberos",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_port_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                port=0,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_port_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                port=-1,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_port_above_65535_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                port=65536,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_max_composition_turns_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                composer_max_composition_turns=0,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_max_discovery_turns_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=0,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_max_tool_calls_per_turn_defaults_to_16(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.composer_max_tool_calls_per_turn == 16

    def test_composer_max_tool_calls_per_turn_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_max_tool_calls_per_turn=0,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_timeout_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_timeout_must_leave_transport_headroom(self) -> None:
        with pytest.raises(ValidationError, match="transport idle ceiling"):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=300.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_timeout_allows_explicit_larger_transport_ceiling(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=300.0,
            composer_transport_idle_ceiling_seconds=360.0,
            composer_transport_headroom_seconds=30.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.composer_timeout_seconds == 300.0

    def test_composer_rate_limit_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=0,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_composer_fields_required_no_defaults(self) -> None:
        """Regression: composer fields must be explicitly configured — no silent defaults."""
        with pytest.raises(ValidationError):
            WebSettings(shareable_link_signing_key=b"\x00" * 32)  # type: ignore[call-arg]  # intentionally omitted to test validation

    def test_composer_model_default_uses_current_openai_frontier_model(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )

        assert settings.composer_model == "gpt-5.5"

    def test_max_upload_bytes_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            WebSettings(
                max_upload_bytes=0,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_payload_store_retention_days_zero_rejected(self) -> None:
        """The audit-readiness retention row reads this setting and surfaces
        it to operators ("System retention: N days"). An accidental ``ge=0``
        relaxation would render as "System retention: 0 days" — wire-valid
        but semantically meaningless. The ``ge=1`` floor is load-bearing.
        """
        with pytest.raises(ValidationError):
            WebSettings(
                payload_store_retention_days=0,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )


class TestWebSettingsDerivedAccessors:
    """Tests for get_landscape_url() and get_payload_store_path()."""

    def test_get_landscape_url_default_derives_from_data_dir(self) -> None:
        settings = WebSettings(
            data_dir=Path("/app/data"),
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_landscape_url()
        assert url == "sqlite:////app/data/runs/audit.db"

    def test_get_landscape_url_explicit_value_returned(self) -> None:
        settings = WebSettings(
            landscape_url="postgresql://db/audit",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_landscape_url()
        assert url == "postgresql://db/audit"

    def test_get_payload_store_path_default_derives_from_data_dir(self) -> None:
        settings = WebSettings(
            data_dir=Path("/app/data"),
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        path = settings.get_payload_store_path()
        assert path == Path("/app/data/payloads")

    def test_get_payload_store_path_explicit_value_returned(self) -> None:
        settings = WebSettings(
            payload_store_path=Path("/mnt/payloads"),
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        path = settings.get_payload_store_path()
        assert path == Path("/mnt/payloads")

    def test_default_data_dir_landscape_url(self) -> None:
        """Default data_dir='data' is resolved to an absolute path at
        validation time, so derived URLs are anchored to a fixed
        location regardless of later os.chdir calls."""
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_landscape_url()
        expected_data_dir = Path("data").resolve()
        assert url == f"sqlite:///{expected_data_dir / 'runs' / 'audit.db'}"

    def test_default_data_dir_payload_store_path(self) -> None:
        """Default data_dir='data' is resolved at validation time;
        payload path inherits that absolute prefix."""
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        path = settings.get_payload_store_path()
        assert path == Path("data").resolve() / "payloads"

    def test_get_session_db_url_default_derives_from_data_dir(self) -> None:
        settings = WebSettings(
            data_dir=Path("/app/data"),
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_session_db_url()
        assert url == "sqlite:////app/data/sessions.db"

    def test_get_session_db_url_explicit_value_returned(self) -> None:
        settings = WebSettings(
            session_db_url="postgresql://db/sessions",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_session_db_url()
        assert url == "postgresql://db/sessions"

    def test_default_data_dir_session_db_url(self) -> None:
        """Default data_dir='data' is resolved to an absolute path at
        validation time; session DB URL inherits that prefix."""
        settings = WebSettings(
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        url = settings.get_session_db_url()
        expected_data_dir = Path("data").resolve()
        assert url == f"sqlite:///{expected_data_dir / 'sessions.db'}"


class TestSecretKeyGuard:
    """Tests for the secret_key production guard validator."""

    def test_default_secret_key_rejected_on_non_local_host(self) -> None:
        with pytest.raises(ValidationError, match="secret_key must be set"):
            WebSettings(
                host="0.0.0.0",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_default_secret_key_allowed_on_localhost(self) -> None:
        # Should not raise
        settings = WebSettings(
            host="127.0.0.1",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.secret_key == "change-me-in-production"

    def test_default_secret_key_allowed_on_localhost_name(self) -> None:
        settings = WebSettings(
            host="localhost",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.secret_key == "change-me-in-production"

    def test_default_secret_key_allowed_on_ipv6_loopback(self) -> None:
        settings = WebSettings(
            host="::1",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.secret_key == "change-me-in-production"

    def test_custom_secret_key_allowed_on_any_host(self) -> None:
        # DC-2 FIX-L: non-loopback hosts also need a non-weak signing key.
        # Uniform-byte placeholders are rejected by the weak-key validator.
        settings = WebSettings(
            host="0.0.0.0",
            secret_key="this-non-loopback-secret-is-long-enough",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\xab\xcd" * 16,
        )
        assert settings.secret_key == "this-non-loopback-secret-is-long-enough"

    def test_short_secret_key_rejected_on_non_local_host(self) -> None:
        with pytest.raises(ValidationError, match="secret_key must be at least 32 bytes"):
            WebSettings(
                host="0.0.0.0",
                secret_key="short-secret",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\xab\xcd" * 16,
            )

    def test_short_secret_key_allowed_on_localhost(self) -> None:
        settings = WebSettings(
            host="127.0.0.1",
            secret_key="short-secret",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.secret_key == "short-secret"

    @pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
    @pytest.mark.parametrize("secret_key", ["", "   "])
    def test_blank_secret_key_rejected_on_all_hosts(self, host: str, secret_key: str) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                host=host,
                secret_key=secret_key,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )


class TestAuthFieldValidation:
    """Tests for OIDC/Entra conditional field requirements."""

    def test_local_provider_no_oidc_fields_required(self) -> None:
        """Local auth (default) should work without any OIDC fields."""
        settings = WebSettings(
            auth_provider="local",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.auth_provider == "local"

    @pytest.mark.parametrize(
        "field_name",
        [
            "oidc_issuer",
            "oidc_audience",
            "oidc_client_id",
            "oidc_authorization_endpoint",
            "entra_tenant_id",
        ],
    )
    def test_local_provider_rejects_oidc_entra_fields(self, field_name: str) -> None:
        """Local auth must not accept inert OIDC/Entra configuration."""
        with pytest.raises(ValidationError, match=field_name):
            WebSettings(
                auth_provider="local",
                **{field_name: "https://issuer.example.com"},
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_oidc_provider_missing_fields_raises(self) -> None:
        """OIDC provider without required fields should raise."""
        with pytest.raises(ValidationError, match="OIDC auth requires"):
            WebSettings(
                auth_provider="oidc",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )


class TestPathFieldValidation:
    """Tests for path normalization and blank-path rejection."""

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("data_dir", ""),
            ("data_dir", "   "),
            ("payload_store_path", ""),
            ("payload_store_path", "   "),
        ],
    )
    def test_blank_path_fields_rejected(self, field_name: str, value: str) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                **{
                    field_name: value,
                    "composer_max_composition_turns": 15,
                    "composer_max_discovery_turns": 10,
                    "composer_timeout_seconds": 85.0,
                    "composer_rate_limit_per_minute": 10,
                    "shareable_link_signing_key": b"\x00" * 32,
                }
            )

    def test_data_dir_expands_user_home(self) -> None:
        settings = WebSettings(
            data_dir="~/data",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        expected = Path("~/data").expanduser().resolve()
        assert settings.data_dir == expected
        assert settings.data_dir.is_absolute()
        assert settings.get_landscape_url() == f"sqlite:///{expected / 'runs' / 'audit.db'}"
        assert settings.get_session_db_url() == f"sqlite:///{expected / 'sessions.db'}"

    def test_payload_store_path_expands_user_home(self) -> None:
        settings = WebSettings(
            payload_store_path="~/payloads",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.get_payload_store_path() == Path("~/payloads").expanduser().resolve()
        assert settings.get_payload_store_path().is_absolute()

    def test_relative_data_dir_resolved_at_validation_immune_to_chdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The download/preview path-allowlist check used to silently
        depend on the running process CWD: a relative data_dir was
        resolved at every callsite, so a chdir between sink-write and
        download-time would move the allowlist with the process. After
        validation-time resolve(), the data_dir is pinned for the
        process lifetime regardless of later chdir.
        """
        monkeypatch.chdir(tmp_path)
        settings = WebSettings(
            data_dir="data",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        # Captured at validation time, anchored to tmp_path.
        expected = (tmp_path / "data").resolve()
        assert settings.data_dir == expected
        # Now chdir somewhere else — settings.data_dir MUST NOT move.
        other_dir = tmp_path / "elsewhere"
        other_dir.mkdir()
        monkeypatch.chdir(other_dir)
        assert settings.data_dir == expected


class TestServerSecretAllowlistValidation:
    """Tests for server_secret_allowlist field validation."""

    def test_reserved_elspeth_server_secret_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ELSPETH_"):
            WebSettings(
                server_secret_allowlist=("ELSPETH_FINGERPRINT_KEY",),
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )


class TestAuthFieldValidationContinued:
    """Additional OIDC/Entra field requirement coverage."""

    def test_oidc_provider_with_all_fields_valid(self) -> None:
        """OIDC provider with all required fields should succeed."""
        settings = WebSettings(
            auth_provider="oidc",
            oidc_issuer="https://issuer.example.com",
            oidc_audience="my-audience",
            oidc_client_id="my-client-id",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.auth_provider == "oidc"
        assert settings.oidc_issuer == "https://issuer.example.com"

    def test_oidc_provider_partial_fields_raises(self) -> None:
        """OIDC provider with only some fields should name the missing ones."""
        with pytest.raises(ValidationError, match="oidc_audience"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_entra_provider_missing_fields_raises(self) -> None:
        """Entra provider without required fields should raise."""
        with pytest.raises(ValidationError, match="Entra auth requires"):
            WebSettings(
                auth_provider="entra",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_entra_provider_missing_tenant_id_raises(self) -> None:
        """Entra with OIDC fields but no tenant_id should raise."""
        with pytest.raises(ValidationError, match="entra_tenant_id"):
            WebSettings(
                auth_provider="entra",
                oidc_issuer="https://login.microsoftonline.com/t/v2.0",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

    def test_entra_provider_with_all_fields_valid(self) -> None:
        """Entra provider with all required fields should succeed."""
        settings = WebSettings(
            auth_provider="entra",
            oidc_issuer="https://login.microsoftonline.com/t/v2.0",
            oidc_audience="my-audience",
            oidc_client_id="my-client-id",
            entra_tenant_id="my-tenant-id",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        assert settings.auth_provider == "entra"
        assert settings.entra_tenant_id == "my-tenant-id"


class TestOIDCBlankStringRejection:
    """Blank/whitespace-only OIDC/Entra fields must be rejected at config time."""

    _COMPOSER_DEFAULTS: typing.ClassVar[dict[str, object]] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }

    def test_oidc_empty_string_issuer_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_whitespace_issuer_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="   ",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_empty_audience_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="",
                oidc_client_id="my-client-id",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_empty_client_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="my-audience",
                oidc_client_id="",
                **self._COMPOSER_DEFAULTS,
            )

    def test_entra_empty_tenant_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="entra",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                entra_tenant_id="",
                **self._COMPOSER_DEFAULTS,
            )

    def test_entra_whitespace_tenant_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="entra",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                entra_tenant_id="   ",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_empty_authorization_endpoint_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                oidc_authorization_endpoint="",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_http_authorization_endpoint_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be an HTTPS URL"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                oidc_authorization_endpoint="http://issuer.example.com/oauth2/authorize",
                **self._COMPOSER_DEFAULTS,
            )

    def test_oidc_cross_origin_authorization_endpoint_rejected(self) -> None:
        with pytest.raises(ValidationError, match="same origin as issuer"):
            WebSettings(
                auth_provider="oidc",
                oidc_issuer="https://issuer.example.com",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                oidc_authorization_endpoint="https://evil.example.com/oauth2/authorize",
                **self._COMPOSER_DEFAULTS,
            )

    def test_entra_cross_origin_authorization_endpoint_rejected(self) -> None:
        with pytest.raises(ValidationError, match="same origin as issuer"):
            WebSettings(
                auth_provider="entra",
                oidc_audience="my-audience",
                oidc_client_id="my-client-id",
                entra_tenant_id="test-tenant-id",
                oidc_authorization_endpoint="https://evil.example.com/oauth2/authorize",
                **self._COMPOSER_DEFAULTS,
            )

    def test_local_auth_blank_oidc_field_still_rejected(self) -> None:
        """Field validator fires regardless of auth_provider — blank is always invalid."""
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(
                auth_provider="local",
                oidc_issuer="",
                **self._COMPOSER_DEFAULTS,
            )


class TestDBURLValidation:
    """Database URL format and passphrase compatibility validation."""

    _COMPOSER_DEFAULTS: typing.ClassVar[dict[str, object]] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }

    def test_landscape_url_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(landscape_url="", **self._COMPOSER_DEFAULTS)

    def test_landscape_url_whitespace_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(landscape_url="   ", **self._COMPOSER_DEFAULTS)

    def test_landscape_url_malformed_rejected(self) -> None:
        with pytest.raises(ValidationError, match="invalid database URL"):
            WebSettings(landscape_url="not-a-url", **self._COMPOSER_DEFAULTS)

    def test_landscape_url_valid_sqlite_accepted(self) -> None:
        settings = WebSettings(landscape_url="sqlite:///path/audit.db", **self._COMPOSER_DEFAULTS)
        assert settings.landscape_url == "sqlite:///path/audit.db"

    def test_landscape_url_valid_postgresql_accepted(self) -> None:
        settings = WebSettings(landscape_url="postgresql://host/db", **self._COMPOSER_DEFAULTS)
        assert settings.landscape_url == "postgresql://host/db"

    def test_session_db_url_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(session_db_url="", **self._COMPOSER_DEFAULTS)

    def test_session_db_url_whitespace_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(session_db_url="   ", **self._COMPOSER_DEFAULTS)

    def test_session_db_url_malformed_rejected(self) -> None:
        with pytest.raises(ValidationError, match="invalid database URL"):
            WebSettings(session_db_url="garbage", **self._COMPOSER_DEFAULTS)

    def test_session_db_url_valid_accepted(self) -> None:
        settings = WebSettings(session_db_url="sqlite:///sessions.db", **self._COMPOSER_DEFAULTS)
        assert settings.session_db_url == "sqlite:///sessions.db"

    def test_passphrase_with_postgresql_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires a SQLite"):
            WebSettings(
                landscape_url="postgresql://host/db",
                landscape_passphrase="secret",
                **self._COMPOSER_DEFAULTS,
            )

    def test_passphrase_with_sqlite_accepted(self) -> None:
        settings = WebSettings(
            landscape_url="sqlite:///audit.db",
            landscape_passphrase="secret",
            **self._COMPOSER_DEFAULTS,
        )
        assert settings.landscape_passphrase == "secret"

    def test_passphrase_without_explicit_url_accepted(self) -> None:
        """When landscape_url is None, default is SQLite — passphrase is valid."""
        settings = WebSettings(landscape_passphrase="secret", **self._COMPOSER_DEFAULTS)
        assert settings.landscape_passphrase == "secret"
        assert settings.landscape_url is None

    def test_passphrase_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(landscape_passphrase="", **self._COMPOSER_DEFAULTS)

    def test_passphrase_whitespace_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WebSettings(landscape_passphrase="   ", **self._COMPOSER_DEFAULTS)


def _optional_string_field_names() -> list[str]:
    """Enumerate all str | None fields on WebSettings for parametrized testing."""
    fields: list[str] = []
    for name, field_info in WebSettings.model_fields.items():
        # field_info.annotation is typed as `type[Any] | None` by pydantic,
        # but at runtime a `str | None` field is stored as a types.UnionType
        # instance.  Widen to Any so the isinstance narrowing is honest —
        # mypy otherwise treats the branch body as unreachable.
        ann: Any = field_info.annotation
        if isinstance(ann, types.UnionType):
            args = typing.get_args(ann)
            if str in args and type(None) in args:
                fields.append(name)
    return fields


class TestFieldValidatorCoverage:
    """Structural test: every str | None field must reject blank strings.

    Prevents the 'Drifting Goals' pattern — new optional string fields added
    without blank-string validators silently accumulate validation gaps.
    """

    _COMPOSER_DEFAULTS: typing.ClassVar[dict[str, object]] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }

    @pytest.mark.parametrize("field_name", _optional_string_field_names())
    def test_blank_string_rejected_for_all_optional_string_fields(self, field_name: str) -> None:
        """Every str | None field must reject empty strings at config time."""
        with pytest.raises(ValidationError):
            WebSettings(**{field_name: "", **self._COMPOSER_DEFAULTS})


class TestJWKSFailureRetryFloor:
    """Regression guard for the cold-start DoS shield (elspeth-32982f17cf).

    ``jwks_failure_retry_seconds`` has a schema floor of ``ge=10``, not
    ``ge=1``.  The floor is load-bearing: during an IdP outage the FIRST
    caller pays the httpx timeout (~15s worst case) while subsequent
    callers short-circuit on the cached negative result.  A configured
    1-second retry collapses that shield — concurrent auth requests
    re-hit the dead IdP almost immediately, reinstating a partial DoS.
    Ten seconds is tight enough for fixtures to advance the window
    deliberately and loose enough that operators cannot configure the
    throttle away.  These tests fail loudly if the floor is silently
    relaxed to ``ge=1`` again.
    """

    _COMPOSER_DEFAULTS: typing.ClassVar[dict[str, object]] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }

    @pytest.mark.parametrize("below_floor", [1, 2, 5, 9])
    def test_below_floor_rejected(self, below_floor: int) -> None:
        """Any value below 10 must raise ValidationError at schema time."""
        with pytest.raises(ValidationError):
            WebSettings(jwks_failure_retry_seconds=below_floor, **self._COMPOSER_DEFAULTS)

    def test_floor_value_accepted(self) -> None:
        """The boundary value ``10`` is the minimum legal configuration."""
        settings = WebSettings(jwks_failure_retry_seconds=10, **self._COMPOSER_DEFAULTS)
        assert settings.jwks_failure_retry_seconds == 10

    def test_above_floor_accepted(self) -> None:
        """Operators may raise the retry window (trading stale-serve risk for safety)."""
        settings = WebSettings(jwks_failure_retry_seconds=600, **self._COMPOSER_DEFAULTS)
        assert settings.jwks_failure_retry_seconds == 600

    def test_zero_rejected(self) -> None:
        """Zero would disable the throttle entirely — must be rejected."""
        with pytest.raises(ValidationError):
            WebSettings(jwks_failure_retry_seconds=0, **self._COMPOSER_DEFAULTS)

    def test_negative_rejected(self) -> None:
        """Negative values are schema-illegal."""
        with pytest.raises(ValidationError):
            WebSettings(jwks_failure_retry_seconds=-1, **self._COMPOSER_DEFAULTS)
