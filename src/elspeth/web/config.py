"""Web application configuration."""

from __future__ import annotations

import os
import sys
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, SecretBytes, field_validator, model_validator

from elspeth.contracts.auth import AuthProviderType
from elspeth.core.config import PayloadStoreSettings
from elspeth.plugins.infrastructure.url_validation import validate_credential_safe_https_url
from elspeth.web.auth.urls import validate_oidc_authorization_endpoint, validate_oidc_issuer
from elspeth.web.validation import (
    SERVER_SECRET_RESERVED_PREFIX,
    is_reserved_server_secret_name,
    validate_secret_name,
)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_MIN_NON_LOCAL_JWT_SECRET_KEY_BYTES = 32
_DEFAULT_COMPOSER_TRANSPORT_IDLE_CEILING_SECONDS = 300.0
_DEFAULT_COMPOSER_TRANSPORT_HEADROOM_SECONDS = 30.0
# Mechanical link to core retention default: if
# core/config.py:PayloadStoreSettings.retention_days changes, this value
# tracks it automatically. Prevents the silent divergence called out in
# docs/composer/ux-redesign-2026-05/14a-phase-2a-backend.md
# §"Retention default divergence guard".
_DEFAULT_PAYLOAD_STORE_RETENTION_DAYS: int = PayloadStoreSettings.model_fields["retention_days"].default


def _allow_insecure_test_keys(host: str) -> bool:
    return host in _LOCAL_HOSTS and ("pytest" in sys.modules or os.environ.get("ELSPETH_ENV") == "test")


def is_default_secret_key_placeholder(secret_key: str) -> bool:
    return secret_key == "change-me-in-production"


def is_undersized_secret_key(secret_key: str) -> bool:
    return len(secret_key.encode("utf-8")) < _MIN_NON_LOCAL_JWT_SECRET_KEY_BYTES


def is_uniform_byte_key(key_bytes: bytes) -> bool:
    return len(set(key_bytes)) == 1


def _is_loopback_or_private_origin(value: str) -> bool:
    parsed = urlparse(value)
    hostname = parsed.hostname
    if hostname is None:
        return True
    if hostname.casefold() == "localhost":
        return True
    try:
        address = ip_address(hostname)
    except ValueError:
        return False
    return not address.is_global


def _is_loopback_origin(value: str) -> bool:
    parsed = urlparse(value)
    hostname = parsed.hostname
    if hostname is None:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        address = ip_address(hostname)
    except ValueError:
        return False
    return address.is_loopback


class WebSettings(BaseModel):
    """Configuration for the ELSPETH web application.

    All fields have sensible defaults for local development.
    auth_provider uses a Literal type so Pydantic rejects invalid
    values automatically -- no manual @field_validator needed.

    Frozen to prevent accidental mutation in async request handlers —
    settings are constructed once and shared via app.state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8451, ge=1, le=65535)
    auth_provider: AuthProviderType = "local"
    # ``default`` preserves current behavior; ``aws-ecs`` is strictly validated by
    # web/deployment_contract.py::validate_aws_ecs_settings.
    deployment_target: Literal["default", "aws-ecs"] = "default"
    registration_mode: Literal["open", "email_verified", "closed"] = "open"
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    data_dir: Path = Field(default=Path("data"), validate_default=True)
    # Trusted externally visible origin used for generated user-facing links.
    # Required for email-verified registration when binding to a non-local host;
    # never derive emailed links from request Host headers.
    public_base_url: str | None = Field(default=None)
    # Phase p4: override for the public base URL the tutorial synthetic-scrape
    # pages are reachable at. When None (the default), the canonical public
    # GitHub Pages copy (TUTORIAL_SAMPLE_PAGES_BASE_URL) is used. Used ONLY to
    # build {base}/tutorial-site/project-N.html for the tutorial's web_scrape
    # node; that node uses the plugin default allowed_hosts="public_only" and the
    # server injects no allowlist. Set this only to host your own copy (a fork).
    tutorial_sample_base_url: str | None = Field(default=None)
    composer_model: str = "gpt-5.5"
    # Operator-set LLM sampling. Default None means omitted from the
    # provider request, which is the coherent default for reasoning-model
    # defaults like gpt-5.5 that reject non-default temperature values.
    # Sent verbatim when set; provider rejection is the operator's config
    # error and is validated at boot. See
    # docs/superpowers/specs/2026-06-03-composer-operator-set-sampling-config-design.md.
    composer_temperature: float | None = Field(default=None, ge=0, le=2)
    composer_seed: int | None = None
    # Tests/offline development can disable the real provider boot probe.
    composer_boot_probe_enabled: bool = True
    composer_max_composition_turns: int = Field(..., ge=1)
    composer_max_discovery_turns: int = Field(..., ge=1)
    composer_max_tool_calls_per_turn: int = Field(default=16, ge=1)
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
    e2e_state_seed_enabled: bool = False
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
    composer_advisor_checkpoint_max_passes: int = Field(
        default=2,
        ge=1,
        description=(
            "Max END advisor-checkpoint passes per compose request (the initial "
            "end sign-off plus its re-reviews), counted SEPARATELY from "
            "_MAX_REPAIR_TURNS. The EARLY advisory pass is separate and does NOT "
            "consume this counter (spec §13). On the last budgeted pass a "
            "still-flagged end gate fails closed (no repair — it cannot "
            "re-review). Checkpoint calls are bounded SOLELY by this knob and are "
            "NOT counted against composer_advisor_max_calls_per_compose (which "
            "bounds LLM-initiated hints + proactive-security). "
            "(Spec §7 envisioned composer_advisor_max_calls_per_compose as a "
            "unified backstop across all advisor calls including checkpoints; "
            "that unification is not implemented — checkpoints are bounded "
            "separately. Operator decision pending.)"
        ),
    )
    composer_advisor_max_prompt_tokens: int = Field(default=4000, ge=1)
    composer_advisor_max_completion_tokens: int = Field(default=1500, ge=1)
    composer_advisor_timeout_seconds: float = Field(default=60.0, gt=0)
    # Phase 5b Task 5 — interpretation-event rate limits (F-30/F-31).
    #
    # Both limits are read at compose-loop initialisation and passed to
    # ``_check_interpretation_rate_limits`` as keyword arguments; changing
    # them requires a service restart (not a per-request reload). The
    # per-day window is UTC midnight, not a sliding 24-hour window —
    # simpler for operators to reason about and produces predictable
    # reset behaviour. See ``web/composer/tools.py`` for the helper that
    # consumes these values.
    composer_interpretation_rate_limit_per_term: int = Field(
        default=3,
        ge=1,
        description=(
            "Max times the composer LLM may surface the same (session, user_term, "
            "composition_state_id) tuple for user review. Exceeding this cap "
            "raises ToolArgumentError; the compose loop falls back to "
            "AUTO_INTERPRETED_NO_SURFACES."
        ),
    )
    composer_interpretation_rate_limit_per_session_day: int = Field(
        default=10,
        ge=1,
        description=(
            "Max request_interpretation_review invocations per session per UTC day. "
            "Window resets at UTC midnight (not a sliding 24-hour window). "
            "Exceeding this cap raises ToolArgumentError; the compose loop falls "
            "back to AUTO_INTERPRETED_NO_SURFACES."
        ),
    )
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
        "AZURE_CONTENT_SAFETY_KEY",
    )
    orphan_run_max_age_seconds: int = Field(default=3600, ge=60)
    orphan_run_check_interval_seconds: int = Field(default=300, ge=30)

    # Execution infrastructure — defaults derive from data_dir when not explicitly set
    landscape_url: str | None = None
    landscape_passphrase: str | None = None
    payload_store_path: Path | None = None
    payload_store_retention_days: int = Field(
        default=_DEFAULT_PAYLOAD_STORE_RETENTION_DAYS,
        ge=1,
        description=(
            "Payload retention in days surfaced by the audit-readiness "
            "panel. Mirrors the core default sourced from "
            "src/elspeth/core/config.py:PayloadStoreSettings.retention_days "
            "via _DEFAULT_PAYLOAD_STORE_RETENTION_DAYS (mechanical link, "
            "not a hand-copied literal). The panel row is informational "
            "only in Phase 2A — there is no user-stated requirement to "
            "compare against yet."
        ),
    )

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

    # Phase 6A — shareable-review token signing.
    #
    # The HMAC key backs the ``ShareTokenSigner`` primitive at
    # ``web/shareable_reviews/signer.py``. Required (Field(...)): the web
    # service refuses to start without it — there is no test-friendly
    # default because rotation/recovery is "re-issue all links," and a
    # silent dev-mode default would let a misconfigured staging deploy
    # ship outstanding links signed with the well-known dev key. Pair the
    # operator-action runbook entry at
    # ``docs/guides/sharing-pipelines.md`` (Task 12 of plan 19a) with the
    # generation step ``openssl rand -base64 32``.
    #
    # 32-byte minimum matches HMAC-SHA256's digest (output) size — the
    # natural entropy floor for a tag of this hash, and the byte count
    # produced by the documented operator recipe ``openssl rand -base64 32``
    # after base64 decode. (HMAC-SHA256's *block* size is 64 bytes; the
    # floor here is the digest size, not the block size.) Longer keys
    # are accepted as opaque key material.
    #
    # Rotating this key invalidates EVERY outstanding shareable link.
    # There is no dual-key acceptance window in v1 — key rotation tooling
    # is out of scope.
    #
    # ``SecretBytes`` masks the value in ``repr()`` so tracebacks, debug
    # logs, and REPL inspection do not exfiltrate the HMAC key. Pydantic v2's
    # default repr otherwise prints every field value in plaintext.
    #
    # ``strict=True`` forbids the lax ``str → bytes`` utf-8 coercion. Without
    # this, a 31-character string containing a 2-byte codepoint
    # (``'a' * 30 + 'ñ'``) silently passes the 32-byte floor with only 31
    # characters of entropy. Strict mode rejects non-bytes inputs at the
    # boundary.
    #
    # Consumer site (web/app.py:276) unwraps with ``.get_secret_value()``
    # when constructing the ``ShareTokenSigner`` primitive — the signer's
    # constructor signature is unchanged (still ``bytes``).
    shareable_link_signing_key: SecretBytes = Field(
        ...,
        strict=True,
        description=(
            "HMAC-SHA256 key for shareable-review tokens. Required — the "
            "service refuses to start without it. Rotating invalidates "
            "ALL outstanding shareable links. Generate with "
            "``openssl rand -base64 32`` and store as utf-8 bytes. "
            "Wrapped in ``SecretBytes``: ``repr()`` shows a mask; call "
            "``.get_secret_value()`` to obtain the raw bytes."
        ),
    )

    # Phase 6A — shareable-review token lifetime.
    #
    # Stamps ``expires_at = now() + this delta`` on every newly minted
    # signed token. The signer verifies expiry on resolve (see
    # ``ShareTokenSigner.verify``). Default is 30 days; operators may
    # lower or raise as appropriate to their review cadence.
    shareable_link_lifetime_seconds: int = Field(
        default=30 * 24 * 3600,
        gt=0,
        description=(
            "Lifetime (in seconds) for shareable-review tokens. Default: "
            "30 days. The service stamps expires_at = now() + this delta "
            "when creating a token."
        ),
    )

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

    @field_validator("tutorial_sample_base_url")
    @classmethod
    def _reject_blank_tutorial_sample_base_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.strip():
            raise ValueError("must not be blank (omit the field or set to a non-empty base URL)")
        return v

    @field_validator("public_base_url")
    @classmethod
    def _validate_public_base_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        safe_url = validate_credential_safe_https_url(
            v,
            field_name="public_base_url",
            allow_http_loopback=True,
        ).rstrip("/")
        parsed = urlparse(safe_url)
        if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
            raise ValueError("public_base_url must be an origin without path, query, or fragment")
        if _is_loopback_or_private_origin(safe_url) and not (parsed.scheme == "http" and _is_loopback_origin(safe_url)):
            raise ValueError("public_base_url must target a public origin unless using HTTP loopback for local development")
        return safe_url

    @field_validator("secret_key")
    @classmethod
    def _reject_blank_secret_key(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("shareable_link_signing_key", mode="before")
    @classmethod
    def _decode_signing_key_from_string(cls, v: object) -> object:
        """Explicit base64 decoding for str inputs.

        Env-var ingestion (``ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY``)
        delivers the key as a ``str``. With ``strict=True`` on the field,
        Pydantic would reject the str outright; without it, Pydantic would
        silently utf-8-encode the str — and that is the bug: a 31-character
        string containing a 2-byte codepoint
        (``'a' * 30 + 'ñ'``) encodes to 32 utf-8 bytes and slips past the
        byte-length floor with only 31 characters of entropy.

        The documented operator recipe is ``openssl rand -base64 32`` (44-char
        base64 string, 32 raw bytes of entropy). We decode str as base64
        explicitly here — same encoding the operator used to generate the
        key. Invalid base64 raises ``ValueError`` (no fall-back to utf-8
        coercion). Bytes pass through unchanged; strict mode then rejects any
        remaining non-bytes types.

        Multibyte-utf-8 ambiguity is foreclosed at the boundary by requiring
        base64 — the only way to express N raw bytes through a string medium
        without character-vs-byte conflation.
        """
        if isinstance(v, str):
            import base64
            import binascii

            try:
                return base64.b64decode(v, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(
                    "shareable_link_signing_key string inputs must be base64-encoded (e.g. ``openssl rand -base64 32``). Decoding failed."
                ) from exc
        return v

    @field_validator("shareable_link_signing_key")
    @classmethod
    def _signing_key_min_length(cls, v: SecretBytes) -> SecretBytes:
        """Phase 6A — reject signing keys shorter than HMAC-SHA256's digest size.

        32 bytes is the minimum — the digest (output) size of HMAC-SHA256
        and the natural entropy floor for a tag produced by this hash.
        ``openssl rand -base64 32`` produces 44 utf-8 characters that
        base64-decode to exactly 32 raw bytes — the floor.
        Shorter keys reduce the effective entropy of the HMAC tag and make
        brute-force token forgery easier; pre-release ELSPETH refuses to start
        a service with such a key.

        The floor is on raw byte length. The ``mode="before"`` companion
        validator above performs explicit base64 decoding for str inputs, so
        this byte count equals the operator-supplied raw byte count —
        multibyte-utf-8 ambiguity is foreclosed at the boundary, not papered
        over here.
        """
        if len(v.get_secret_value()) < 32:
            raise ValueError("shareable_link_signing_key must be at least 32 bytes")
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
        if self.registration_mode == "email_verified" and self.host not in _LOCAL_HOSTS:
            if self.public_base_url is None:
                raise ValueError("email_verified registration on a non-local host requires public_base_url")
            if _is_loopback_or_private_origin(self.public_base_url):
                raise ValueError("public_base_url for a non-local email_verified host must be publicly reachable")

        if self.auth_provider == "local":
            configured = [
                name
                for name, val in (
                    ("oidc_issuer", self.oidc_issuer),
                    ("oidc_audience", self.oidc_audience),
                    ("oidc_client_id", self.oidc_client_id),
                    ("oidc_authorization_endpoint", self.oidc_authorization_endpoint),
                    ("entra_tenant_id", self.entra_tenant_id),
                )
                if val is not None
            ]
            if configured:
                raise ValueError(f"Local auth does not use OIDC/Entra fields: {', '.join(configured)}")
        elif self.auth_provider == "oidc":
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
            assert self.oidc_issuer is not None
            object.__setattr__(self, "oidc_issuer", validate_oidc_issuer(self.oidc_issuer))
            if self.oidc_authorization_endpoint is not None:
                object.__setattr__(
                    self,
                    "oidc_authorization_endpoint",
                    validate_oidc_authorization_endpoint(
                        self.oidc_authorization_endpoint,
                        issuer=self.oidc_issuer,
                    ),
                )
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
            if self.oidc_authorization_endpoint is not None:
                assert self.entra_tenant_id is not None
                object.__setattr__(
                    self,
                    "oidc_authorization_endpoint",
                    validate_oidc_authorization_endpoint(
                        self.oidc_authorization_endpoint,
                        issuer=f"https://login.microsoftonline.com/{self.entra_tenant_id}/v2.0",
                    ),
                )
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
    def _validate_advisor_distinct_from_primary(self) -> WebSettings:
        """The advisor must be a different model from the primary composer.

        Independence of failure modes: a model checking its own work shares
        its blind spots. Exact-string distinctness on the canonical model id
        (final path segment, so provider prefixes like ``openrouter/openai/``
        do not mask a same-model pairing). The advisor is mandatory — there is
        no enable flag — so this runs for every boot.
        """

        def _canonical(model_id: str) -> str:
            return model_id.rsplit("/", 1)[-1].strip()

        if _canonical(self.composer_advisor_model) == _canonical(self.composer_model):
            raise ValueError(
                "composer_advisor_model must differ from composer_model "
                f"(both resolve to {_canonical(self.composer_model)!r}); the advisor "
                "is the independent reviewer and cannot be the primary composer"
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
        """Reject default or undersized JWT HMAC keys outside explicit test contexts."""
        if _allow_insecure_test_keys(self.host):
            return self
        if is_default_secret_key_placeholder(self.secret_key):
            raise ValueError(
                "secret_key must be set to a secure value outside explicit test contexts. "
                "Set ELSPETH_WEB__SECRET_KEY or pass secret_key explicitly."
            )
        if is_undersized_secret_key(self.secret_key):
            raise ValueError(
                f"secret_key must be at least {_MIN_NON_LOCAL_JWT_SECRET_KEY_BYTES} bytes outside explicit test contexts. "
                "Generate a high-entropy key for ELSPETH_WEB__SECRET_KEY."
            )
        return self

    @model_validator(mode="after")
    def _reject_known_weak_signing_key(self) -> WebSettings:
        """Refuse uniform-byte placeholder signing keys outside explicit test contexts.

        Test fixtures across the suite use ``b'\\x00' * 32`` and ``b'0' * 32``
        as convenient 32-byte placeholders. Those values are operationally
        indistinguishable from "the operator forgot to generate a real key" —
        outside an explicit test context that is a security incident waiting to
        happen. Mirrors the shape of ``_enforce_secret_key_in_production`` above.

        A real key from ``openssl rand -base64 32`` is uniformly distributed;
        a single repeated byte (any value) is the signature of a placeholder.
        The check is intentionally simple — "all bytes identical" — to avoid
        false positives on legitimate (if unusual) high-entropy keys.
        """
        if _allow_insecure_test_keys(self.host):
            return self
        raw_key = self.shareable_link_signing_key.get_secret_value()
        if is_uniform_byte_key(raw_key):
            raise ValueError(
                "shareable_link_signing_key is a known-weak placeholder "
                "(uniform-byte pattern detected); generate a real key with "
                "``openssl rand -base64 32`` outside explicit test contexts."
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
