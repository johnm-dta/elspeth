"""Shared ChromaDB connection configuration.

Used by ChromaSearchProviderConfig and ChromaSinkConfig, which validate their
connection fields by constructing a ChromaConnectionConfig (triggering its
validators) and discarding the instance.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
            if not self.ssl and self.host not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError(
                    f"HTTPS (ssl=True) is required for remote ChromaDB hosts, "
                    f"got host={self.host!r} with ssl=False. "
                    f"Non-SSL connections are only permitted for localhost."
                )
        return self
