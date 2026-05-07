"""Tests for retrieval type dataclasses."""

from types import MappingProxyType

import pytest

from elspeth.contracts.errors import PluginRetryableError
from elspeth.plugins.infrastructure.clients.retrieval.base import RetrievalError
from elspeth.plugins.infrastructure.clients.retrieval.types import RetrievalChunk


class TestRetrievalChunkScoreValidation:
    def test_score_at_lower_bound(self):
        chunk = RetrievalChunk(content="text", score=0.0, source_id="doc1", metadata={})
        assert chunk.score == 0.0

    def test_score_at_upper_bound(self):
        chunk = RetrievalChunk(content="text", score=1.0, source_id="doc1", metadata={})
        assert chunk.score == 1.0

    def test_score_below_lower_bound_raises(self):
        with pytest.raises(ValueError, match=r"normalized to.*0\.0.*1\.0"):
            RetrievalChunk(content="text", score=-0.1, source_id="doc1", metadata={})

    def test_score_above_upper_bound_raises(self):
        with pytest.raises(ValueError, match=r"normalized to.*0\.0.*1\.0"):
            RetrievalChunk(content="text", score=1.1, source_id="doc1", metadata={})

    def test_mid_range_score(self):
        chunk = RetrievalChunk(content="text", score=0.75, source_id="doc1", metadata={})
        assert chunk.score == 0.75


class TestRetrievalChunkMetadataValidation:
    def test_valid_metadata(self):
        chunk = RetrievalChunk(
            content="text",
            score=0.5,
            source_id="doc1",
            metadata={"page": 3, "section": "intro"},
        )
        assert chunk.metadata == {"page": 3, "section": "intro"}

    def test_empty_metadata(self):
        chunk = RetrievalChunk(content="text", score=0.5, source_id="doc1", metadata={})
        assert chunk.metadata == {}

    def test_non_serializable_metadata_raises(self):
        """Provider must coerce non-primitive types at Tier 3 boundary."""
        import datetime

        with pytest.raises(ValueError, match="JSON-serializable"):
            RetrievalChunk(
                content="text",
                score=0.5,
                source_id="doc1",
                metadata={"timestamp": datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)},
            )

    def test_bytes_metadata_raises(self):
        with pytest.raises(ValueError, match="JSON-serializable"):
            RetrievalChunk(
                content="text",
                score=0.5,
                source_id="doc1",
                metadata={"data": b"binary"},
            )

    def test_nested_metadata_ok(self):
        chunk = RetrievalChunk(
            content="text",
            score=0.5,
            source_id="doc1",
            metadata={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )
        assert chunk.metadata["nested"]["key"] == "value"

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
    def test_non_finite_metadata_rejected(self, bad_value: float):
        """Non-finite floats in metadata must be rejected at the Tier 3 boundary."""
        with pytest.raises(ValueError, match="JSON-serializable"):
            RetrievalChunk(
                content="text",
                score=0.5,
                source_id="doc1",
                metadata={"score": bad_value},
            )

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
    def test_nested_non_finite_metadata_rejected(self, bad_value: float):
        """Non-finite floats nested in metadata must also be rejected."""
        with pytest.raises(ValueError, match="JSON-serializable"):
            RetrievalChunk(
                content="text",
                score=0.5,
                source_id="doc1",
                metadata={"nested": {"value": bad_value}},
            )


class TestRetrievalChunkMetadataImmutability:
    """Regression tests: metadata must be deeply frozen after construction."""

    def test_metadata_is_mapping_proxy(self):
        chunk = RetrievalChunk(
            content="text",
            score=0.5,
            source_id="doc1",
            metadata={"page": 3},
        )
        assert isinstance(chunk.metadata, MappingProxyType)

    def test_top_level_mutation_raises(self):
        chunk = RetrievalChunk(
            content="text",
            score=0.5,
            source_id="doc1",
            metadata={"page": 3},
        )
        with pytest.raises(TypeError):
            chunk.metadata["new_key"] = "value"

    def test_nested_dict_mutation_raises(self):
        chunk = RetrievalChunk(
            content="text",
            score=0.5,
            source_id="doc1",
            metadata={"nested": {"key": "value"}},
        )
        with pytest.raises(TypeError):
            chunk.metadata["nested"]["key"] = "mutated"


class TestRetrievalError:
    def test_retryable_error(self):
        err = RetrievalError("timeout", retryable=True, status_code=429)
        assert err.retryable is True
        assert err.status_code == 429
        assert str(err) == "timeout"

    def test_non_retryable_error(self):
        err = RetrievalError("bad request", retryable=False, status_code=400)
        assert err.retryable is False
        assert err.status_code == 400

    def test_is_plugin_retryable_error(self):
        err = RetrievalError("test", retryable=True)
        assert isinstance(err, PluginRetryableError)

    def test_status_code_defaults_none(self):
        err = RetrievalError("test", retryable=False)
        assert err.status_code is None
