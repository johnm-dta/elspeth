"""Tests for RAG ingestion error types."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from elspeth.contracts.errors import (
    CommencementGateFailedError,
    DependencyFailedError,
    DuplicateDocumentError,
    RetrievalNotReadyError,
)


class TestDependencyFailedError:
    def test_empty_dependency_name_raises(self) -> None:
        with pytest.raises(ValueError, match="dependency_name must not be empty"):
            DependencyFailedError(dependency_name="", run_id="y", reason="z")

    def test_empty_run_id_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id must not be empty"):
            DependencyFailedError(dependency_name="x", run_id="", reason="z")

    def test_empty_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="reason must not be empty"):
            DependencyFailedError(dependency_name="x", run_id="y", reason="")


class TestCommencementGateFailedError:
    def test_context_snapshot_is_deeply_frozen(self) -> None:
        """context_snapshot must be frozen — it's named 'snapshot' so it must snapshot."""
        original = {"collections": {"test": {"count": 0, "reachable": True}}}
        err = CommencementGateFailedError(
            gate_name="g",
            condition="True",
            reason="r",
            context_snapshot=original,
        )
        # Outer dict is frozen
        assert isinstance(err.context_snapshot, MappingProxyType)
        # Nested dict is also frozen
        assert isinstance(err.context_snapshot["collections"], MappingProxyType)
        assert isinstance(err.context_snapshot["collections"]["test"], MappingProxyType)

    def test_empty_gate_name_raises(self) -> None:
        with pytest.raises(ValueError, match="gate_name must not be empty"):
            CommencementGateFailedError(gate_name="", condition="True", reason="r", context_snapshot={})

    def test_empty_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="condition must not be empty"):
            CommencementGateFailedError(gate_name="g", condition="", reason="r", context_snapshot={})

    def test_empty_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="reason must not be empty"):
            CommencementGateFailedError(gate_name="g", condition="True", reason="", context_snapshot={})

    def test_context_snapshot_mutation_isolated(self) -> None:
        """Mutating the original dict after construction must not affect the error."""
        original: dict[str, Any] = {"key": {"nested": 1}}
        err = CommencementGateFailedError(
            gate_name="g",
            condition="True",
            reason="r",
            context_snapshot=original,
        )
        original["key"]["nested"] = 999
        original["new_key"] = "injected"
        assert err.context_snapshot["key"]["nested"] == 1
        assert "new_key" not in err.context_snapshot


class TestRetrievalNotReadyError:
    def test_empty_collection_raises(self) -> None:
        with pytest.raises(ValueError, match="collection must not be empty"):
            RetrievalNotReadyError(collection="", reason="unreachable")

    def test_empty_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="reason must not be empty"):
            RetrievalNotReadyError(collection="test", reason="")


class TestDuplicateDocumentError:
    def test_empty_duplicate_ids_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            DuplicateDocumentError(collection="test", duplicate_ids=[])
