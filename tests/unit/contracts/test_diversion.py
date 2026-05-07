"""Tests for sink diversion contracts."""

from __future__ import annotations

import pytest

from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.errors import PluginContractViolation
from elspeth.contracts.results import ArtifactDescriptor


class TestRowDiversion:
    def test_create_with_required_fields(self) -> None:
        d = RowDiversion(row_index=0, reason="bad metadata", row_data={"a": 1})
        assert d.row_index == 0
        assert d.reason == "bad metadata"
        assert d.row_data["a"] == 1

    def test_negative_row_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="row_index must be >= 0"):
            RowDiversion(row_index=-1, reason="test", row_data={})

    def test_bool_row_index_rejected(self) -> None:
        """bool is a subclass of int — require_int rejects it."""
        with pytest.raises(TypeError, match="row_index must be int"):
            RowDiversion(row_index=True, reason="test", row_data={})


class TestSinkWriteResult:
    def _make_artifact(self) -> ArtifactDescriptor:
        return ArtifactDescriptor.for_file(
            path="/tmp/test.csv",
            content_hash="abc123def456",
            size_bytes=100,
        )

    def test_no_diversions_default(self) -> None:
        result = SinkWriteResult(artifact=self._make_artifact())
        assert result.diversions == ()
        assert result.artifact.path_or_uri == "file:///tmp/test.csv"

    def test_with_diversions(self) -> None:
        divs = (
            RowDiversion(row_index=1, reason="bad type", row_data={"x": 1}),
            RowDiversion(row_index=3, reason="too long", row_data={"x": 2}),
        )
        result = SinkWriteResult(artifact=self._make_artifact(), diversions=divs)
        assert len(result.diversions) == 2
        assert result.diversions[0].row_index == 1
        assert result.diversions[1].row_index == 3

    def test_duplicate_row_index_rejected(self) -> None:
        """Duplicate row_index values crash — would silently collapse in executor."""
        divs = (
            RowDiversion(row_index=1, reason="first", row_data={"x": 1}),
            RowDiversion(row_index=1, reason="second", row_data={"x": 2}),
        )
        with pytest.raises(PluginContractViolation, match="duplicate diversion row_index=1"):
            SinkWriteResult(artifact=self._make_artifact(), diversions=divs)

    def test_artifact_must_be_artifact_descriptor(self) -> None:
        """artifact field rejects non-ArtifactDescriptor values."""
        with pytest.raises(PluginContractViolation, match="artifact must be ArtifactDescriptor"):
            SinkWriteResult(artifact="not_an_artifact")  # type: ignore[arg-type]

    def test_artifact_rejects_dict(self) -> None:
        """A dict that looks like an artifact is still not an ArtifactDescriptor."""
        with pytest.raises(PluginContractViolation, match="artifact must be ArtifactDescriptor"):
            SinkWriteResult(artifact={"artifact_type": "file", "content_hash": "abc", "size_bytes": 1})  # type: ignore[arg-type]

    def test_diversions_must_be_tuple(self) -> None:
        """Diversions must be a tuple (immutable), not a list."""
        divs = [RowDiversion(row_index=0, reason="test", row_data={"a": 1})]
        with pytest.raises(PluginContractViolation, match="diversions must be a tuple"):
            SinkWriteResult(artifact=self._make_artifact(), diversions=divs)  # type: ignore[arg-type]

    def test_diversion_elements_must_be_row_diversion(self) -> None:
        """Each element in diversions must be a RowDiversion instance."""
        with pytest.raises(PluginContractViolation, match="diversions\\[0\\] must be RowDiversion"):
            SinkWriteResult(
                artifact=self._make_artifact(),
                diversions=({"row_index": 0, "reason": "test", "row_data": {}},),  # type: ignore[arg-type]
            )

    def test_diversions_tuple_default(self) -> None:
        """Default diversions is an empty tuple."""
        result = SinkWriteResult(artifact=self._make_artifact())
        assert isinstance(result.diversions, tuple)
