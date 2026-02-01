"""Tests for ArtifactRepository and BatchMemberRepository."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from elspeth.core.landscape.repositories import (
    ArtifactRepository,
    BatchMemberRepository,
)


class TestArtifactRepository:
    """Tests for ArtifactRepository."""

    def test_load_artifact_all_fields(self) -> None:
        """Load returns Artifact with all fields mapped correctly."""
        now = datetime.now(UTC)
        row = MagicMock(
            artifact_id="art_123",
            run_id="run_1",
            produced_by_state_id="state_456",
            sink_node_id="sink_output",
            artifact_type="csv",
            path_or_uri="/output/results.csv",
            content_hash="hash_abc",
            size_bytes=1024,
            created_at=now,
            idempotency_key="idem_key_1",
        )

        repo = ArtifactRepository(MagicMock())
        result = repo.load(row)

        assert result.artifact_id == "art_123"
        assert result.run_id == "run_1"
        assert result.produced_by_state_id == "state_456"
        assert result.sink_node_id == "sink_output"
        assert result.artifact_type == "csv"
        assert result.path_or_uri == "/output/results.csv"
        assert result.content_hash == "hash_abc"
        assert result.size_bytes == 1024
        assert result.created_at == now
        assert result.idempotency_key == "idem_key_1"

    def test_load_artifact_optional_fields_none(self) -> None:
        """Load handles None optional fields correctly."""
        now = datetime.now(UTC)
        row = MagicMock(
            artifact_id="art_456",
            run_id="run_2",
            produced_by_state_id="state_789",
            sink_node_id="sink_json",
            artifact_type="json",
            path_or_uri="/output/data.json",
            content_hash="hash_def",
            size_bytes=2048,
            created_at=now,
            idempotency_key=None,
        )

        repo = ArtifactRepository(MagicMock())
        result = repo.load(row)

        assert result.artifact_id == "art_456"
        assert result.idempotency_key is None


class TestBatchMemberRepository:
    """Tests for BatchMemberRepository."""

    def test_load_batch_member(self) -> None:
        """Load returns BatchMember with all fields mapped correctly."""
        row = MagicMock(
            batch_id="batch_1",
            token_id="token_1",
            ordinal=0,
        )

        repo = BatchMemberRepository(MagicMock())
        result = repo.load(row)

        assert result.batch_id == "batch_1"
        assert result.token_id == "token_1"
        assert result.ordinal == 0

    def test_load_batch_member_higher_ordinal(self) -> None:
        """Load handles non-zero ordinal values correctly."""
        row = MagicMock(
            batch_id="batch_2",
            token_id="token_5",
            ordinal=4,
        )

        repo = BatchMemberRepository(MagicMock())
        result = repo.load(row)

        assert result.batch_id == "batch_2"
        assert result.token_id == "token_5"
        assert result.ordinal == 4
