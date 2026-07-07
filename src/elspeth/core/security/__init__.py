"""Security utilities for ELSPETH.

The stable eager facade is intentionally small: fingerprint helpers are cheap
contract functions used by core config and plugin code. Secret loading and web
SSRF helpers remain available as lazy compatibility exports so lightweight
fingerprint callers do not initialize Azure/secret-loader or DNS resolver state.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from elspeth.contracts.security import (
    get_fingerprint_key,
    secret_fingerprint,
)

if TYPE_CHECKING:
    from elspeth.core.security.config_secrets import (
        SecretLoadError as SecretLoadError,
    )
    from elspeth.core.security.config_secrets import (
        load_secrets_from_config as load_secrets_from_config,
    )
    from elspeth.core.security.secret_loader import (
        CachedSecretLoader as CachedSecretLoader,
    )
    from elspeth.core.security.secret_loader import (
        CompositeSecretLoader as CompositeSecretLoader,
    )
    from elspeth.core.security.secret_loader import (
        EnvSecretLoader as EnvSecretLoader,
    )
    from elspeth.core.security.secret_loader import (
        KeyVaultSecretLoader as KeyVaultSecretLoader,
    )
    from elspeth.core.security.secret_loader import (
        SecretLoader as SecretLoader,
    )
    from elspeth.core.security.secret_loader import (
        SecretNotFoundError as SecretNotFoundError,
    )
    from elspeth.core.security.secret_loader import (
        SecretRef as SecretRef,
    )
    from elspeth.core.security.web import (
        ALWAYS_BLOCKED_RANGES as ALWAYS_BLOCKED_RANGES,
    )
    from elspeth.core.security.web import (
        NetworkError as NetworkError,
    )
    from elspeth.core.security.web import (
        SSRFBlockedError as SSRFBlockedError,
    )
    from elspeth.core.security.web import (
        SSRFSafeRequest as SSRFSafeRequest,
    )
    from elspeth.core.security.web import (
        validate_literal_ip_for_ssrf as validate_literal_ip_for_ssrf,
    )
    from elspeth.core.security.web import (
        validate_url_for_ssrf as validate_url_for_ssrf,
    )
    from elspeth.core.security.web import (
        validate_url_scheme as validate_url_scheme,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ALWAYS_BLOCKED_RANGES": ("elspeth.core.security.web", "ALWAYS_BLOCKED_RANGES"),
    "CachedSecretLoader": ("elspeth.core.security.secret_loader", "CachedSecretLoader"),
    "CompositeSecretLoader": ("elspeth.core.security.secret_loader", "CompositeSecretLoader"),
    "EnvSecretLoader": ("elspeth.core.security.secret_loader", "EnvSecretLoader"),
    "KeyVaultSecretLoader": ("elspeth.core.security.secret_loader", "KeyVaultSecretLoader"),
    "NetworkError": ("elspeth.core.security.web", "NetworkError"),
    "SSRFBlockedError": ("elspeth.core.security.web", "SSRFBlockedError"),
    "SSRFSafeRequest": ("elspeth.core.security.web", "SSRFSafeRequest"),
    "SecretLoadError": ("elspeth.core.security.config_secrets", "SecretLoadError"),
    "SecretLoader": ("elspeth.core.security.secret_loader", "SecretLoader"),
    "SecretNotFoundError": ("elspeth.core.security.secret_loader", "SecretNotFoundError"),
    "SecretRef": ("elspeth.core.security.secret_loader", "SecretRef"),
    "load_secrets_from_config": ("elspeth.core.security.config_secrets", "load_secrets_from_config"),
    "validate_literal_ip_for_ssrf": ("elspeth.core.security.web", "validate_literal_ip_for_ssrf"),
    "validate_url_for_ssrf": ("elspeth.core.security.web", "validate_url_for_ssrf"),
    "validate_url_scheme": ("elspeth.core.security.web", "validate_url_scheme"),
}


def __getattr__(name: str) -> object:
    """Resolve compatibility facade exports without eager module imports."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "ALWAYS_BLOCKED_RANGES",
    "CachedSecretLoader",
    "CompositeSecretLoader",
    "EnvSecretLoader",
    "KeyVaultSecretLoader",
    "NetworkError",
    "SSRFBlockedError",
    "SSRFSafeRequest",
    "SecretLoadError",
    "SecretLoader",
    "SecretNotFoundError",
    "SecretRef",
    "get_fingerprint_key",
    "load_secrets_from_config",
    "secret_fingerprint",
    "validate_literal_ip_for_ssrf",
    "validate_url_for_ssrf",
    "validate_url_scheme",
]
