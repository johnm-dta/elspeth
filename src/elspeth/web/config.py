"""Web application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from elspeth.contracts.auth import AuthProviderType
from elspeth.web.validation import (
    SERVER_SECRET_RESERVED_PREFIX,
    is_reserved_server_secret_name,
    validate_secret_name,
)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DEFAULT_COMPOSER_TRANSPORT_IDLE_CEILING_SECONDS = 300.0
_DEFAULT_COMPOSER_TRANSPORT_HEADROOM_SECONDS = 30.0


class WebSettings(BaseModel):
    """Configuration for the ELSPETH web application.

    All fields have sensible defaults for local development.
    auth_provider uses a Literal type so Pydantic rejects invalid
    values automatically -- no manual @field_validator needed.

    Frozen to prevent accidental mutation in async request handlers —
    settings are constructed once and shared via app.state.
    """

    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=8451, ge=1, le=65535)
    auth_provider: AuthProviderType = "local"
    registration_mode: Literal["open", "email_verified", "closed"] = "open"
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    data_dir: Path = Field(default=Path("data"), validate_default=True)
    composer_model: str = "gpt-5.5"
    composer_max_composition_turns: int = Field(..., ge=1)
    composer_max_discovery_turns: int = Field(..., ge=1)
    composer_timeout_seconds: float = Field(..., gt=0)
    composer_transport_idle_ceiling_seconds: float = Field(
        default=_DEFAULT_COMPOSER_TRANSPORT_IDLE_CEILING_SECONDS,
        gt=0,
    )
    composer_transport_headroom_seconds: float = Field(
        default=_DEFAULT_COMPOSER_TRANSPORT_HEADROOM_SECONDS,
        gt=0,
    )
    composer_runtime_preflight_timeout_seconds: float = Field(default=5.0, gt=0)
    composer_rate_limit_per_minute: int = Field(..., ge=1)
    composer_expose_provider_errors: bool = False
    # Advisor escape hatch: lets the composer LLM phone a frontier model
    # for guidance when stuck. Disabled by default; enabling it filters
    # the request_advisor_hint tool into get_tool_definitions(). Budget
    # is per-compose-request (local counter), not per-session lifetime.
    composer_advisor_enabled: bool = False
    composer_advisor_model: str = "anthropic/claude-sonnet-4-6"
    composer_advisor_max_calls_per_compose: int = Field(
        default=4,
        ge=0,
        description=(
            "Maximum advisor calls per compose request (NOT per session-across-time). "
            "Each new user prompt starts with a fresh budget. The default of 4 is "
            "sized to accommodate one proactive intro call (security/red-list trigger) "
            "plus three reactive recovery calls — the proactive call should not crowd "
            "out reactive recovery. A user prompting 10 times in one session may make "
            "up to 40 advisor calls total over the session lifetime; session-lifetime "
            "cost is bounded by composer_rate_limit_per_minute, not this setting. "
            "Raise this for heavyweight workloads (e.g. business-analysis pipelines "
            "with many plugins where the LLM benefits from multiple intro consultations)."
        ),
    )
    composer_advisor_max_prompt_tokens: int = Field(default=4000, ge=1)
    composer_advisor_max_completion_tokens: int = Field(default=1500, ge=1)
    composer_advisor_timeout_seconds: float = Field(default=60.0, gt=0)
    auth_rate_limit_per_minute: int = Field(default=20, ge=1)
    secret_key: str = (
        "change-me-in-production"  # Security rule S3 (seam-contracts.md): Sub-2 startup guard enforces non-default in production
    )
    max_upload_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    max_blob_storage_per_session_bytes: int = Field(default=500 * 1024 * 1024, ge=1)
    server_secret_allowlist: tuple[str, ...] = (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_API_KEY",
    )
    orphan_run_max_age_seconds: int = Field(default=3600, ge=60)
    orphan_run_check_interval_seconds: int = Field(default=300, ge=30)

    # Execution infrastructure — defaults derive from data_dir when not explicitly set
    landscape_url: str | None = None
    landscape_passphrase: str | None = None
    payload_store_path: Path | None = None

    # OIDC / Entra-specific (optional)
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_client_id: str | None = None
    oidc_authorization_endpoint: str | None = None
    entra_tenant_id: str | None = None

    # JWKS cache tuning (OIDC / Entra). Defaults match the provider
    # defaults; operators may lower or raise them. Raising the failure
    # retry makes stale-serve windows longer (safer during brief IdP
    # outages); lowering it shrinks the partial-DoS blast radius during
    # a sustained outage — see elspeth-32982f17cf.
    #
    # ``jwks_failure_retry_seconds`` floor is 10, not 1: the whole point
    # of the throttle is that during an IdP outage the FIRST caller pays
    # the httpx timeout (≈15s worst case) and the REST short-circuit.
    # A configured 1-second retry window means concurrent auth requests
    # re-hit the dead IdP almost immediately, collapsing the shield
    # back toward the per-request timeout and reinstating the cold-start
    # DoS the throttle exists to close.  Ten seconds is tight enough for
    # test fixtures to advance the window deliberately and loose enough
    # that production operators cannot configure the throttle away.
    jwks_cache_ttl_seconds: int = Field(default=3600, ge=1)
    jwks_failure_retry_seconds: int = Field(default=300, ge=10)

    # Session database (sessions, messages, composition states, runs)
    # Separate from landscape_url (audit DB)
    session_db_url: str | None = None

    @field_validator(
        "oidc_issuer",
        "oidc_audience",
        "oidc_client_id",
        "oidc_authorization_endpoint",
        "entra_tenant_id",
    )
    @classmethod
    def _reject_blank_auth_fields(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.strip():
            raise ValueError("must not be blank (omit the field or set to a non-empty value)")
        return v

    @field_validator("landscape_url", "session_db_url")
    @classmethod
    def _validate_db_url(cls, v: str | None) -> str | None:
        """Reject blank and malformed database URLs at config time."""
        if v is None:
            return None
        from elspeth.contracts.database_url import validate_database_url_format

        return validate_database_url_format(v)

    @field_validator("landscape_passphrase")
    @classmethod
    def _reject_blank_passphrase(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.strip():
            raise ValueError("must not be blank (omit the field to disable encryption)")
        return v

    @field_validator("secret_key")
    @classmethod
    def _reject_blank_secret_key(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("data_dir", "payload_store_path", mode="before")
    @classmethod
    def _reject_blank_path_strings(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("data_dir", "payload_store_path")
    @classmethod
    def _normalize_paths(cls, v: Path | None) -> Path | None:
        if v is None:
            return None
        if not str(v).strip():
            raise ValueError("must not be blank")
        # Resolve to an absolute path at validation time so downstream
        # consumers do not depend on the running process CWD. Without
        # this, a relative `data_dir` (e.g. the default Path("data"))
        # is interpreted against whatever CWD the systemd unit happens
        # to have when the audit DB is opened vs. when the
        # sink-allowlist is checked — same code, different answers.
        # `.resolve()` makes the answer immutable for the process
        # lifetime regardless of later os.chdir calls.
        return v.expanduser().resolve()

    @field_validator("server_secret_allowlist")
    @classmethod
    def _validate_server_secret_allowlist(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_secret_name(name, field_name="server_secret_allowlist entry") for name in v)
        reserved = tuple(name for name in validated if is_reserved_server_secret_name(name))
        if reserved:
            raise ValueError(f"server_secret_allowlist entries must not start with {SERVER_SECRET_RESERVED_PREFIX}: {sorted(reserved)}")
        return validated

    @model_validator(mode="after")
    def _validate_auth_fields(self) -> WebSettings:
        """Enforce that OIDC/Entra providers have their required fields."""
        if self.auth_provider == "oidc":
            missing = [
                name
                for name, val in (
                    ("oidc_issuer", self.oidc_issuer),
                    ("oidc_audience", self.oidc_audience),
                    ("oidc_client_id", self.oidc_client_id),
                )
                if not val
            ]
            if missing:
                raise ValueError(f"OIDC auth requires: {', '.join(missing)}")
        elif self.auth_provider == "entra":
            # oidc_issuer is NOT required — EntraAuthProvider derives it
            # from entra_tenant_id (login.microsoftonline.com/{tid}/v2.0).
            missing = [
                name
                for name, val in (
                    ("oidc_audience", self.oidc_audience),
                    ("oidc_client_id", self.oidc_client_id),
                    ("entra_tenant_id", self.entra_tenant_id),
                )
                if not val
            ]
            if missing:
                raise ValueError(f"Entra auth requires: {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def _validate_composer_timeout_transport_headroom(self) -> WebSettings:
        """Keep composer wall-clock failures ahead of browser/proxy aborts."""
        max_backend_timeout_seconds = self.composer_transport_idle_ceiling_seconds - self.composer_transport_headroom_seconds
        if max_backend_timeout_seconds <= 0:
            raise ValueError("composer_transport_headroom_seconds must be less than composer_transport_idle_ceiling_seconds")
        if self.composer_timeout_seconds > max_backend_timeout_seconds:
            raise ValueError(
                "composer_timeout_seconds must leave transport idle ceiling headroom: "
                f"got {self.composer_timeout_seconds}s, maximum {max_backend_timeout_seconds}s "
                f"(transport idle ceiling {self.composer_transport_idle_ceiling_seconds}s - "
                f"headroom {self.composer_transport_headroom_seconds}s)"
            )
        return self

    @model_validator(mode="after")
    def _validate_passphrase_requires_sqlite(self) -> WebSettings:
        """Reject landscape_passphrase with non-SQLite URLs at config time."""
        if self.landscape_passphrase is not None and self.landscape_url is not None:
            from sqlalchemy.engine.url import make_url

            driver = make_url(self.landscape_url).drivername.split("+")[0]
            if driver != "sqlite":
                raise ValueError(
                    f"landscape_passphrase requires a SQLite landscape_url, "
                    f"got driver '{driver}'. Either remove the passphrase "
                    f"or change landscape_url to sqlite:///path/to/audit.db"
                )
        return self

    @model_validator(mode="after")
    def _enforce_secret_key_in_production(self) -> WebSettings:
        """Reject the default secret key when host suggests non-local deployment."""
        if self.secret_key == "change-me-in-production" and self.host not in _LOCAL_HOSTS:
            raise ValueError(
                "secret_key must be set to a secure value for non-local deployments "
                "(host is not a loopback address). Set ELSPETH_WEB__SECRET_KEY or pass secret_key explicitly."
            )
        return self

    def get_landscape_url(self) -> str:
        """Resolve landscape DB URL, defaulting to data_dir-relative path."""
        if self.landscape_url is not None:
            return self.landscape_url
        db_path = self.data_dir / "runs" / "audit.db"
        return f"sqlite:///{db_path}"

    def get_payload_store_path(self) -> Path:
        """Resolve payload store path, defaulting to data_dir-relative path."""
        if self.payload_store_path is not None:
            return self.payload_store_path
        return self.data_dir / "payloads"

    def get_session_db_url(self) -> str:
        """Resolve session DB URL, defaulting to data_dir-relative path."""
        if self.session_db_url is not None:
            return self.session_db_url
        db_path = self.data_dir / "sessions.db"
        return f"sqlite:///{db_path}"
