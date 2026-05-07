"""URL sanitization types for audit-safe storage.

These types GUARANTEE URLs cannot contain credentials when stored in the audit trail.
Callers must explicitly sanitize using the factory methods, making accidental
secret leaks impossible at the type level.

Usage:
    from elspeth.contracts.url import SanitizedDatabaseUrl, SanitizedWebhookUrl

    # Database URLs - extracts password, fingerprints it, returns sanitized URL
    sanitized = SanitizedDatabaseUrl.from_raw_url("postgresql://user:secret@host/db")
    # sanitized.sanitized_url = "postgresql://user@host/db"
    # sanitized.fingerprint = "abc123..." (HMAC of password)

    # Webhook URLs - handles query params and Basic Auth
    sanitized = SanitizedWebhookUrl.from_raw_url("https://api.example.com?token=sk-xxx")
    # sanitized.sanitized_url = "https://api.example.com"
    # sanitized.fingerprint = "def456..." (HMAC of token value only)
"""

import json as json_module
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse

from elspeth.contracts.security import (
    SecretFingerprintError,
    get_fingerprint_key,
    secret_fingerprint,
)


def _extract_raw_port(netloc: str) -> str:
    """Extract raw port string (including colon) from a URL netloc.

    Unlike ``urlparse().port`` — which calls ``int()`` and raises
    ``ValueError`` on non-numeric ports — this function returns the raw
    string.  This handles templated ports like ``${PORT}`` and malformed
    DSNs that should be passed through unchanged.

    Returns empty string if no port is present.
    """
    # Strip userinfo (user:pass@)
    host_port = netloc.rsplit("@", 1)[-1]

    if host_port.startswith("["):
        # IPv6: [::1]:port or [::1]
        bracket_close = host_port.find("]")
        if bracket_close == -1:
            return ""
        after = host_port[bracket_close + 1 :]
        return after if after.startswith(":") else ""

    # Regular host or IPv4: host:port or host
    colon = host_port.rfind(":")
    return host_port[colon:] if colon != -1 else ""


# Sensitive query parameter names that should be stripped from webhook URLs.
# Expanded list per code review to cover OAuth, API keys, signed URLs, etc.
def _base_param_name(key: str) -> str:
    """Extract base parameter name, stripping bracket/dot suffixes.

    parse_qs preserves bracket notation (api_key[0] -> "api_key[0]") and
    dot notation (token.value -> "token.value") as literal key strings.
    This function extracts the base name for matching against SENSITIVE_PARAMS.

    Examples:
        _base_param_name("api_key[0]") -> "api_key"
        _base_param_name("token.value") -> "token"
        _base_param_name("api_key") -> "api_key"
    """
    # Bracket notation: take everything before first '['
    bracket = key.find("[")
    if bracket != -1:
        key = key[:bracket]
    # Dot notation: take everything before first '.'
    dot = key.find(".")
    if dot != -1:
        key = key[:dot]
    return key


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


@dataclass(frozen=True, slots=True)
class SanitizedDatabaseUrl:
    """Database URL with credentials removed. Cannot contain secrets.

    This is a frozen dataclass that guarantees the URL stored in `sanitized_url`
    has had any password removed. The `fingerprint` field contains an HMAC-SHA256
    of the original password (if present) for audit traceability.

    Use the `from_raw_url` factory method to create instances.
    """

    sanitized_url: str
    fingerprint: str | None  # None if original had no password

    def __post_init__(self) -> None:
        """Enforce invariant: sanitized_url must not contain credentials."""
        parsed = urlparse(self.sanitized_url)
        if parsed.password is not None:
            raise ValueError(
                "SanitizedDatabaseUrl cannot contain a password in the URL. Use SanitizedDatabaseUrl.from_raw_url() to sanitize first."
            )

    @classmethod
    def from_raw_url(
        cls,
        url: str,
        *,
        fail_if_no_key: bool = True,
    ) -> "SanitizedDatabaseUrl":
        """Create sanitized URL from raw database connection URL.

        Uses stdlib ``urlparse`` to extract and remove passwords from database
        connection URLs. Handles SQLAlchemy-style URLs like
        ``postgresql+psycopg2://user:pass@host:5432/db``.

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
        parsed = urlparse(url)

        if parsed.password is None:
            return cls(sanitized_url=url, fingerprint=None)

        # Compute fingerprint if we have a key
        fingerprint: str | None = None
        try:
            get_fingerprint_key()
            have_key = True
        except ValueError:
            have_key = False

        if have_key:
            # Decode percent-encoding before fingerprinting so the fingerprint
            # represents the actual secret value, not the URL encoding.
            # urlparse().password preserves percent-encoding (e.g., "p%40ss" for "p@ss").
            fingerprint = secret_fingerprint(unquote(parsed.password))
        elif fail_if_no_key:
            raise SecretFingerprintError(
                "Database URL contains a password but ELSPETH_FINGERPRINT_KEY "
                "is not set. Either set the environment variable or use "
                "ELSPETH_ALLOW_RAW_SECRETS=true for development "
                "(not recommended for production)."
            )
        # else: dev mode - just remove password without fingerprint

        # Reconstruct netloc without password.
        # hostname can be None for Unix-socket DSNs like
        # postgresql://user:pass@/dbname?host=/var/run/postgresql
        host_part = ""
        if parsed.hostname:
            if ":" in parsed.hostname:
                # IPv6 addresses need brackets
                host_part = f"[{parsed.hostname}]"
            else:
                host_part = parsed.hostname

        port_str = _extract_raw_port(parsed.netloc)

        if parsed.username:
            netloc = f"{parsed.username}@{host_part}{port_str}"
        else:
            netloc = f"{host_part}{port_str}"

        sanitized = urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

        return cls(sanitized_url=sanitized, fingerprint=fingerprint)


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Enforce invariant: sanitized_url must not contain credentials."""
        parsed = urlparse(self.sanitized_url)
        # Check for ANY userinfo in netloc (username, password, or both).
        # Must match the sanitizer's rule: username OR password = sensitive.
        # Many services use username for bearer tokens (e.g., https://token@host).
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "SanitizedWebhookUrl cannot contain userinfo (username/password) in the URL. "
                "Use SanitizedWebhookUrl.from_raw_url() to sanitize first."
            )
        # Check for sensitive query parameters
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        sensitive_in_query = [k for k in query_params if _base_param_name(k.lower()) in SENSITIVE_PARAMS]
        if sensitive_in_query:
            raise ValueError(
                f"SanitizedWebhookUrl cannot contain sensitive query parameters: "
                f"{sensitive_in_query}. "
                f"Use SanitizedWebhookUrl.from_raw_url() to sanitize first."
            )
        # Check for sensitive fragment parameters
        fragment_params = parse_qs(parsed.fragment, keep_blank_values=True)
        sensitive_in_fragment = [k for k in fragment_params if _base_param_name(k.lower()) in SENSITIVE_PARAMS]
        if sensitive_in_fragment:
            raise ValueError(
                f"SanitizedWebhookUrl cannot contain sensitive fragment parameters: "
                f"{sensitive_in_fragment}. "
                f"Use SanitizedWebhookUrl.from_raw_url() to sanitize first."
            )

    @classmethod
    def from_raw_url(
        cls,
        url: str,
        *,
        fail_if_no_key: bool = True,
    ) -> "SanitizedWebhookUrl":
        """Create sanitized URL from raw webhook URL.

        Handles:
        - Query parameter tokens (e.g., ?token=xxx, ?api_key=xxx)
        - Fragment tokens (e.g., #access_token=xxx) - common in OAuth implicit flow
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
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query, keep_blank_values=True)

        # Parse fragment as query params (e.g., #access_token=xxx&state=yyy)
        # SECURITY: OAuth implicit flow and some APIs put tokens in fragments
        fragment_params = parse_qs(parsed.fragment, keep_blank_values=True)

        # Track which sensitive keys are present (even if empty)
        has_sensitive_query_keys = any(_base_param_name(k.lower()) in SENSITIVE_PARAMS for k in query_params)
        has_sensitive_fragment_keys = any(_base_param_name(k.lower()) in SENSITIVE_PARAMS for k in fragment_params)

        # Collect only non-empty sensitive values for fingerprinting
        sensitive_values: list[str] = []

        # Check query params for sensitive keys
        for key, values in query_params.items():
            if _base_param_name(key.lower()) in SENSITIVE_PARAMS:
                # Only add non-empty values to fingerprint
                sensitive_values.extend(v for v in values if v)

        # Check fragment params for sensitive keys
        for key, values in fragment_params.items():
            if _base_param_name(key.lower()) in SENSITIVE_PARAMS:
                # Only add non-empty values to fingerprint
                sensitive_values.extend(v for v in values if v)

        # Check for Basic Auth credentials (user:pass@host OR user@host)
        # SECURITY: Treat BOTH username and password as sensitive.
        # Many services use username for bearer tokens (e.g., https://token@github.com)
        has_basic_auth = parsed.username is not None or parsed.password is not None
        # Decode percent-encoding before fingerprinting so the fingerprint
        # represents the actual secret value, not the URL encoding.
        if parsed.username:
            sensitive_values.append(unquote(parsed.username))
        if parsed.password:
            sensitive_values.append(unquote(parsed.password))

        # If no sensitive keys in query, fragment, or Basic Auth found, return URL unchanged
        if not has_sensitive_query_keys and not has_sensitive_fragment_keys and not has_basic_auth:
            return cls(sanitized_url=url, fingerprint=None)

        # Compute fingerprint only if there are non-empty values
        fingerprint: str | None = None
        if sensitive_values:
            # We have non-empty secrets - need to fingerprint them
            try:
                get_fingerprint_key()
                have_key = True
            except ValueError:
                have_key = False

            if have_key:
                # Use canonical JSON array encoding for unambiguous fingerprinting.
                # Pipe-delimited join is collision-prone: "a|b" as one value
                # collides with "a" and "b" as two values. JSON array encoding
                # preserves structural boundaries between values.
                combined = json_module.dumps(sorted(sensitive_values), separators=(",", ":"))
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
        sanitized_params = {k: v for k, v in query_params.items() if _base_param_name(k.lower()) not in SENSITIVE_PARAMS}

        # Remove sensitive fragment params
        sanitized_fragment_params = {k: v for k, v in fragment_params.items() if _base_param_name(k.lower()) not in SENSITIVE_PARAMS}

        # Reconstruct netloc without ANY Basic Auth credentials
        # SECURITY: Strip entire userinfo section when credentials present
        if has_basic_auth:
            # Remove both username and password - rebuild netloc without userinfo
            port_str = _extract_raw_port(parsed.netloc)
            # IPv6 addresses need brackets (hostname strips them, netloc preserves them)
            if parsed.hostname and ":" in parsed.hostname:
                netloc = f"[{parsed.hostname}]{port_str}"
            else:
                netloc = f"{parsed.hostname or ''}{port_str}"
        else:
            netloc = parsed.netloc

        # Reconstruct fragment from sanitized params
        # Only include fragment if there are remaining params
        sanitized_fragment = urlencode(sanitized_fragment_params, doseq=True) if sanitized_fragment_params else ""

        # Reconstruct URL without secrets
        sanitized = urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                urlencode(sanitized_params, doseq=True) if sanitized_params else "",
                sanitized_fragment,
            )
        )

        return cls(sanitized_url=sanitized, fingerprint=fingerprint)
