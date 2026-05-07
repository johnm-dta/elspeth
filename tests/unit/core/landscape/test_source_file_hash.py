"""Tests for Node.source_file_hash validation.

The source_file_hash field tracks the hash of the plugin source file
for auditability. It must be None (for engine nodes) or match the
format 'sha256:<16-hex-chars>'.

Per Data Manifesto: The audit database is OUR data. Bad data = crash.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from elspeth.contracts.audit import Node
from elspeth.contracts.enums import Determinism, NodeType


def _make_node(**overrides: Any) -> Node:
    """Build a Node with sensible defaults for testing."""
    defaults: dict[str, Any] = {
        "node_id": "node-1",
        "run_id": "run-1",
        "plugin_name": "test",
        "node_type": NodeType.TRANSFORM,
        "plugin_version": "1.0.0",
        "determinism": Determinism.DETERMINISTIC,
        "config_hash": "a" * 64,
        "config_json": "{}",
        "sequence_in_pipeline": 0,
        "registered_at": datetime.now(tz=UTC),
        "source_file_hash": None,
    }
    defaults.update(overrides)
    return Node(**defaults)


class TestNodeSourceFileHashValidation:
    """Tier 1 validation: source_file_hash must be 'sha256:<16-hex>' or None."""

    def test_accepts_none(self) -> None:
        """None is valid for engine nodes that have no source file."""
        node = _make_node(source_file_hash=None)
        assert node.source_file_hash is None

    def test_accepts_valid_hash(self) -> None:
        """A correctly formatted sha256 truncated hash is accepted."""
        node = _make_node(source_file_hash="sha256:abcdef0123456789")
        assert node.source_file_hash == "sha256:abcdef0123456789"

    def test_rejects_invalid_format(self) -> None:
        """Arbitrary strings must be rejected."""
        with pytest.raises(ValueError, match="source_file_hash must match"):
            _make_node(source_file_hash="not-a-valid-hash")

    def test_rejects_empty_string(self) -> None:
        """Empty string is not None and not a valid hash."""
        with pytest.raises(ValueError, match="source_file_hash must match"):
            _make_node(source_file_hash="")

    def test_rejects_wrong_prefix(self) -> None:
        """Only sha256 prefix is accepted."""
        with pytest.raises(ValueError, match="source_file_hash must match"):
            _make_node(source_file_hash="md5:abcdef0123456789")

    def test_rejects_wrong_length(self) -> None:
        """The hex portion must be exactly 16 characters."""
        with pytest.raises(ValueError, match="source_file_hash must match"):
            _make_node(source_file_hash="sha256:abc")
