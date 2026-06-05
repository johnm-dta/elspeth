"""Tests for shared ChromaDB connection configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.plugins.infrastructure.clients.retrieval.connection import (
    ChromaConnectionConfig,
)


class TestChromaConnectionConfig:
    def test_persistent_mode_requires_persist_directory(self) -> None:
        with pytest.raises(ValidationError, match="persist_directory"):
            ChromaConnectionConfig(
                collection="test",
                mode="persistent",
            )

    def test_persistent_mode_forbids_host(self) -> None:
        with pytest.raises(ValidationError, match="host"):
            ChromaConnectionConfig(
                collection="test",
                mode="persistent",
                persist_directory="./data",
                host="example.com",
            )

    def test_client_mode_requires_host(self) -> None:
        with pytest.raises(ValidationError, match="host"):
            ChromaConnectionConfig(
                collection="test",
                mode="client",
            )

    def test_client_mode_forbids_persist_directory(self) -> None:
        with pytest.raises(ValidationError, match="persist_directory"):
            ChromaConnectionConfig(
                collection="test",
                mode="client",
                host="example.com",
                persist_directory="./data",
            )

    def test_client_mode_requires_https_for_remote(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)ssl|https"):
            ChromaConnectionConfig(
                collection="test",
                mode="client",
                host="remote.example.com",
                ssl=False,
            )

    def test_client_mode_allows_no_ssl_for_localhost(self) -> None:
        config = ChromaConnectionConfig(
            collection="test",
            mode="client",
            host="localhost",
            ssl=False,
        )
        assert config.host == "localhost"
        assert config.ssl is False

    def test_client_mode_allows_no_ssl_for_127_0_0_1(self) -> None:
        config = ChromaConnectionConfig(
            collection="test",
            mode="client",
            host="127.0.0.1",
            ssl=False,
        )
        assert config.host == "127.0.0.1"

    def test_client_mode_allows_no_ssl_for_ipv6_loopback(self) -> None:
        config = ChromaConnectionConfig(
            collection="test",
            mode="client",
            host="::1",
            ssl=False,
        )
        assert config.host == "::1"

    def test_client_mode_blocks_link_local_metadata_host(self) -> None:
        """An internal/metadata target (cloud metadata IP) is rejected by the SSRF policy."""
        with pytest.raises(ValidationError, match=r"(?i)ssrf"):
            ChromaConnectionConfig(
                collection="test",
                mode="client",
                host="169.254.169.254",
                ssl=True,
            )

    def test_client_mode_blocks_private_rfc1918_host(self) -> None:
        """A private-range target is rejected by the SSRF policy (no DNS needed for an IP)."""
        with pytest.raises(ValidationError, match=r"(?i)ssrf"):
            ChromaConnectionConfig(
                collection="test",
                mode="client",
                host="10.0.0.5",
                ssl=True,
            )

    def test_persistent_mode_valid(self) -> None:
        config = ChromaConnectionConfig(
            collection="science-facts",
            mode="persistent",
            persist_directory="./chroma_data",
            distance_function="cosine",
        )
        assert config.collection == "science-facts"
        assert config.mode == "persistent"
        assert config.persist_directory == "./chroma_data"

    def test_defaults(self) -> None:
        config = ChromaConnectionConfig(
            collection="test",
            mode="persistent",
            persist_directory="./data",
        )
        assert config.port == 8000
        assert config.distance_function == "cosine"
        assert config.ssl is True

    def test_frozen(self) -> None:
        config = ChromaConnectionConfig(
            collection="test",
            mode="persistent",
            persist_directory="./data",
        )
        with pytest.raises(ValidationError):
            config.collection = "other"  # type: ignore[misc]

    def test_rejects_port_zero(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ChromaConnectionConfig(
                collection="test",
                mode="client",
                host="localhost",
                port=0,
            )

    def test_rejects_port_above_65535(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 65535"):
            ChromaConnectionConfig(
                collection="test",
                mode="client",
                host="localhost",
                port=70000,
            )

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            ChromaConnectionConfig(
                collection="test",
                mode="persistent",
                persist_directory="./data",
                unknown_field="value",  # type: ignore[call-arg]
            )
