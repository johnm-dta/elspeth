# src/elspeth/contracts/url.py
"""URL sanitization types for audit-safe storage.

These types GUARANTEE URLs cannot contain credentials when stored in the audit trail.
Callers must explicitly sanitize using the factory methods, making accidental
secret leaks impossible at the type level.

Usage:
    from elspeth.contracts.url import SanitizedDatabaseUrl, SanitizedWebhookUrl

    # Database URLs - reuses existing _sanitize_dsn infrastructure
    sanitized = SanitizedDatabaseUrl.from_raw_url("postgresql://user:secret@host/db")
    # sanitized.sanitized_url = "postgresql://user@host/db"
    # sanitized.fingerprint = "abc123..." (HMAC of password)

    # Webhook URLs - handles query params and Basic Auth
    sanitized = SanitizedWebhookUrl.from_raw_url("https://api.example.com?token=sk-xxx")
    # sanitized.sanitized_url = "https://api.example.com"
    # sanitized.fingerprint = "def456..." (HMAC of token value only)
"""

from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# NOTE: Fingerprint functions are imported lazily inside methods to avoid
# breaking the contracts leaf module boundary. Importing from elspeth.core
# at module level would pull in 1,200+ modules (pandas, numpy, sqlalchemy, etc.)
# FIX: P2-2026-01-20-contracts-config-reexport-breaks-leaf-boundary

# Sensitive query parameter names that should be stripped from webhook URLs.
# Expanded list per code review to cover OAuth, API keys, signed URLs, etc.
SENSITIVE_PARAMS = frozenset(
    {
        # Common API authentication
        "token",
        "api_key",
        "apikey",
        "key",
        "secret",
        "password",
        "auth",
        # OAuth patterns
        "access_token",
        "client_secret",
        "api_secret",
        "bearer",
        # Signed URL patterns
        "signature",
        "sig",
        # Header-style params sometimes in query strings
        "authorization",
        "x-api-key",
        # Credential patterns
        "credential",
        "credentials",
    }
)


@dataclass(frozen=True)
class SanitizedDatabaseUrl:
    """Database URL with credentials removed. Cannot contain secrets.

    This is a frozen dataclass that guarantees the URL stored in `sanitized_url`
    has had any password removed. The `fingerprint` field contains an HMAC-SHA256
    of the original password (if present) for audit traceability.

    Use the `from_raw_url` factory method to create instances.
    """

    sanitized_url: str
    fingerprint: str | None  # None if original had no password

    @classmethod
    def from_raw_url(
        cls,
        url: str,
        *,
        fail_if_no_key: bool = True,
    ) -> "SanitizedDatabaseUrl":
        """Create sanitized URL from raw database connection URL.

        Reuses existing `_sanitize_dsn` infrastructure from config.py to ensure
        consistent behavior with Landscape database URL sanitization.

        Args:
            url: Raw database connection URL (SQLAlchemy format)
            fail_if_no_key: If True (default), raise SecretFingerprintError when
                            password is found but ELSPETH_FINGERPRINT_KEY is not set.
                            If False (dev mode), sanitize without fingerprint.

        Returns:
            SanitizedDatabaseUrl with credentials removed and fingerprint if available

        Raises:
            SecretFingerprintError: If password found, no key available,
                                    and fail_if_no_key=True
        """
        # Import here to avoid circular dependency (config imports from contracts)
        from elspeth.core.config import _sanitize_dsn

        sanitized, fingerprint, _ = _sanitize_dsn(url, fail_if_no_key=fail_if_no_key)
        return cls(sanitized_url=sanitized, fingerprint=fingerprint)


@dataclass(frozen=True)
class SanitizedWebhookUrl:
    """Webhook URL with tokens removed. Cannot contain secrets.

    This is a frozen dataclass that guarantees the URL stored in `sanitized_url`
    has had any sensitive query parameters and Basic Auth credentials removed.
    The `fingerprint` field contains an HMAC-SHA256 of the removed secret values
    (not the full URL) for audit traceability.

    Use the `from_raw_url` factory method to create instances.
    """

    sanitized_url: str
    fingerprint: str | None  # None if original had no secrets

    @classmethod
    def from_raw_url(
        cls,
        url: str,
        *,
        fail_if_no_key: bool = True,
    ) -> "SanitizedWebhookUrl":
        """Create sanitized URL from raw webhook URL.

        Handles both:
        - Query parameter tokens (e.g., ?token=xxx, ?api_key=xxx)
        - Basic Auth credentials (e.g., https://user:pass@host/path)

        The fingerprint is computed from ONLY the secret values (not the full URL),
        so you can verify "same token was used" even if endpoint paths differ.

        Args:
            url: Raw webhook URL that may contain tokens
            fail_if_no_key: If True (default), raise SecretFingerprintError when
                            secrets are found but ELSPETH_FINGERPRINT_KEY is not set.
                            If False (dev mode), sanitize without fingerprint.

        Returns:
            SanitizedWebhookUrl with secrets removed and fingerprint if available

        Raises:
            SecretFingerprintError: If secrets found, no key available,
                                    and fail_if_no_key=True
        """
        # Import here to access the error type
        from elspeth.core.config import SecretFingerprintError

        parsed = urlparse(url)
        query_params = parse_qs(parsed.query, keep_blank_values=True)

        # Track which sensitive keys are present (even if empty)
        has_sensitive_keys = any(k.lower() in SENSITIVE_PARAMS for k in query_params)

        # Collect only non-empty sensitive values for fingerprinting
        sensitive_values: list[str] = []

        # Check query params for sensitive keys
        for key, values in query_params.items():
            if key.lower() in SENSITIVE_PARAMS:
                # Only add non-empty values to fingerprint
                sensitive_values.extend(v for v in values if v)

        # Check for Basic Auth credentials (user:pass@host OR user@host)
        # SECURITY: Treat BOTH username and password as sensitive.
        # Many services use username for bearer tokens (e.g., https://token@github.com)
        has_basic_auth = parsed.username is not None or parsed.password is not None
        if parsed.username:
            sensitive_values.append(parsed.username)
        if parsed.password:
            sensitive_values.append(parsed.password)

        # If no sensitive keys or Basic Auth found, return URL unchanged
        if not has_sensitive_keys and not has_basic_auth:
            return cls(sanitized_url=url, fingerprint=None)

        # Compute fingerprint only if there are non-empty values
        fingerprint: str | None = None
        if sensitive_values:
            # Lazy import to avoid breaking contracts leaf boundary
            from elspeth.core.security.fingerprint import (
                get_fingerprint_key,
                secret_fingerprint,
            )

            # We have non-empty secrets - need to fingerprint them
            try:
                get_fingerprint_key()
                have_key = True
            except ValueError:
                have_key = False

            if have_key:
                # Sort for deterministic fingerprint regardless of param order
                combined = "|".join(sorted(sensitive_values))
                fingerprint = secret_fingerprint(combined)
            elif fail_if_no_key:
                raise SecretFingerprintError(
                    "Webhook URL contains tokens but ELSPETH_FINGERPRINT_KEY "
                    "is not set. Either set the environment variable or use "
                    "ELSPETH_ALLOW_RAW_SECRETS=true for development "
                    "(not recommended for production)."
                )
            # else: dev mode - sanitize without fingerprint
        # else: only empty values (e.g., ?token=) - no fingerprint needed

        # Remove sensitive query params
        sanitized_params = {k: v for k, v in query_params.items() if k.lower() not in SENSITIVE_PARAMS}

        # Reconstruct netloc without ANY Basic Auth credentials
        # SECURITY: Strip entire userinfo section when credentials present
        if has_basic_auth:
            # Remove both username and password - rebuild netloc without userinfo
            port_str = f":{parsed.port}" if parsed.port else ""
            # IPv6 addresses need brackets (hostname strips them, netloc preserves them)
            if parsed.hostname and ":" in parsed.hostname:
                netloc = f"[{parsed.hostname}]{port_str}"
            else:
                netloc = f"{parsed.hostname}{port_str}"
        else:
            netloc = parsed.netloc

        # Reconstruct URL without secrets
        sanitized = urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                urlencode(sanitized_params, doseq=True) if sanitized_params else "",
                parsed.fragment,
            )
        )

        return cls(sanitized_url=sanitized, fingerprint=fingerprint)
