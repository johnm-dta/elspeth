"""Shared ChromaDB connection configuration.

Used by ChromaSearchProviderConfig and ChromaSinkConfig, which validate their
connection fields by constructing a ChromaConnectionConfig (triggering its
validators) and discarding the instance.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from elspeth.core.security.web import NetworkError, SSRFBlockedError, validate_url_for_ssrf
from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode_enabled

_LOOPBACK_HOSTS = {"localhost"}
_LOOPBACK_ALLOWED_RANGES = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
)


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower().removesuffix(".")
    if normalized in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized.strip("[]")).is_loopback
    except ValueError:
        return False


def _host_url(host: str, port: int, *, ssl: bool) -> str:
    if "://" in host or "/" in host or "?" in host or "#" in host:
        raise ValueError(f"host must be a bare hostname or IP address, got {host!r}")
    try:
        parsed_ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        url_host = host
    else:
        url_host = f"[{parsed_ip}]" if parsed_ip.version == 6 else str(parsed_ip)
    scheme = "https" if ssl else "http"
    return f"{scheme}://{url_host}:{port}/"


def _validate_chroma_http_target(host: str, port: int, *, ssl: bool) -> None:
    """Apply the project SSRF policy before handing the target to Chroma SDK.

    The SDK owns the actual HTTP transport, so we cannot use
    AuditedHTTPClient's pinned-IP request path here. This preflight still
    enforces the same host/IP blocklist for user-authored Chroma targets and
    keeps loopback-only local development explicitly bounded.
    """
    allowed_ranges = _LOOPBACK_ALLOWED_RANGES if _is_loopback_host(host) else ()
    try:
        validate_url_for_ssrf(_host_url(host, port, ssl=ssl), allowed_ranges=allowed_ranges)
    except (SSRFBlockedError, NetworkError) as exc:
        raise ValueError(f"ChromaDB host {host!r} is blocked by the SSRF policy: {exc}") from exc


class ChromaConnectionConfig(BaseModel):
    """Shared ChromaDB connection fields with cross-field validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection: str = Field(description="ChromaDB collection name")

    @field_validator("collection")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        if len(v) < 3:
            raise ValueError(f"collection name must be at least 3 characters, got {len(v)}")
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$", v):
            raise ValueError(
                f"collection must contain only alphanumeric characters, hyphens, and underscores "
                f"(and start/end with alphanumeric), got {v!r}."
            )
        return v

    mode: Literal["persistent", "client"] = Field(description="Connection mode: persistent (local disk) or client (remote HTTP)")
    persist_directory: str | None = Field(
        default=None,
        description="Path to ChromaDB data directory (persistent mode only)",
    )

    @field_validator("persist_directory")
    @classmethod
    def reject_path_traversal(cls, v: str | None) -> str | None:
        if v is not None and ".." in v.split("/"):
            raise ValueError(f"persist_directory must not contain '..' path components, got {v!r}")
        return v

    host: str | None = Field(
        default=None,
        description="ChromaDB server hostname (client mode only)",
    )
    port: int = Field(default=8000, ge=1, le=65535, description="ChromaDB server port")
    ssl: bool = Field(default=True, description="Use HTTPS for client connections")
    distance_function: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Distance function for collection creation",
    )

    @model_validator(mode="after")
    def validate_mode_fields(self) -> ChromaConnectionConfig:
        if self.mode == "persistent":
            if self.persist_directory is None:
                raise ValueError("persist_directory is required when mode='persistent'")
            if self.host is not None:
                raise ValueError("host must not be set when mode='persistent'")
        elif self.mode == "client":
            if self.host is None:
                raise ValueError("host is required when mode='client'")
            if self.persist_directory is not None:
                raise ValueError("persist_directory must not be set when mode='client'")
            if not self.ssl and not _is_loopback_host(self.host):
                raise ValueError(
                    f"HTTPS (ssl=True) is required for remote ChromaDB hosts, "
                    f"got host={self.host!r} with ssl=False. "
                    f"Non-SSL connections are only permitted for localhost."
                )
            # The SSRF check resolves DNS, so it must not run during preflight
            # (preflight validates config purely, without network I/O). It runs
            # at real runtime — before the Chroma SDK opens any connection.
            if not plugin_preflight_mode_enabled():
                _validate_chroma_http_target(self.host, self.port, ssl=self.ssl)
        return self
