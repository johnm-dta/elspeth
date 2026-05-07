"""Tests for collection readiness probes."""

from __future__ import annotations

import pytest

from elspeth.contracts.probes import CollectionProbe, CollectionReadinessResult


class TestCollectionReadinessResult:
    """Tests for the unified collection readiness result type."""

    def test_construction_with_all_fields(self) -> None:
        result = CollectionReadinessResult(
            collection="science-facts",
            reachable=True,
            count=450,
            message="Collection 'science-facts' has 450 documents",
        )
        assert result.collection == "science-facts"
        assert result.reachable is True
        assert result.count == 450
        assert result.message == "Collection 'science-facts' has 450 documents"

    def test_frozen_immutability(self) -> None:
        result = CollectionReadinessResult(
            collection="test",
            reachable=True,
            count=10,
            message="ok",
        )
        with pytest.raises(AttributeError):
            result.count = 99  # type: ignore[misc]

    def test_unreachable_with_none_count(self) -> None:
        """Unreachable collections can have count=None (unknown).

        Bug fix: elspeth-ed5125c3e1. Previously, probes were forced to
        fabricate count=0 when the count was unknown.
        """
        result = CollectionReadinessResult(
            collection="missing",
            reachable=False,
            count=None,
            message="Collection 'missing' unreachable",
        )
        assert result.reachable is False
        assert result.count is None

    def test_reachable_with_none_count(self) -> None:
        """Reachable collections can have count=None (e.g., malformed response).

        Bug fix: elspeth-ed5125c3e1. A probe that reaches the server
        but can't parse the count should not fabricate count=0.
        """
        result = CollectionReadinessResult(
            collection="weird",
            reachable=True,
            count=None,
            message="Index returned non-integer $count body",
        )
        assert result.reachable is True
        assert result.count is None

    def test_empty_collection_result(self) -> None:
        result = CollectionReadinessResult(
            collection="empty",
            reachable=True,
            count=0,
            message="Collection 'empty' is empty",
        )
        assert result.reachable is True
        assert result.count == 0

    def test_empty_collection_raises(self) -> None:
        with pytest.raises(ValueError, match="collection must not be empty"):
            CollectionReadinessResult(collection="", reachable=True, count=0, message="ok")

    def test_negative_count_raises(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 0"):
            CollectionReadinessResult(collection="test", reachable=True, count=-1, message="ok")

    def test_bool_count_rejected(self) -> None:
        """bool is a subclass of int in Python — require_int rejects it."""
        with pytest.raises(TypeError, match="count must be int"):
            CollectionReadinessResult(collection="test", reachable=True, count=True, message="ok")

    def test_unreachable_with_known_count_raises(self) -> None:
        """Unreachable with a known count is contradictory.

        Bug fix: elspeth-ed5125c3e1. If we can't reach the collection,
        we can't know its count. count must be None when unreachable.
        """
        with pytest.raises(ValueError, match=r"[Uu]nreachable.*count"):
            CollectionReadinessResult(collection="test", reachable=False, count=5, message="down")

    def test_valid_construction(self) -> None:
        result = CollectionReadinessResult(
            collection="my-collection",
            reachable=True,
            count=0,
            message="ok",
        )
        assert result.collection == "my-collection"
        assert result.count == 0


class TestCollectionProbe:
    """Tests for the CollectionProbe protocol."""

    def test_non_compliant_missing_probe_method(self) -> None:
        class BadProbe:
            collection_name: str = "test"

        assert not isinstance(BadProbe(), CollectionProbe)

    def test_non_compliant_missing_collection_name(self) -> None:
        class BadProbe:
            def probe(self) -> CollectionReadinessResult:
                return CollectionReadinessResult(collection="x", reachable=True, count=1, message="ok")

        # Python 3.12+ runtime_checkable checks non-callable protocol
        # members as instance attributes. BadProbe has no collection_name
        # instance attribute, so isinstance returns False.
        assert not isinstance(BadProbe(), CollectionProbe)
