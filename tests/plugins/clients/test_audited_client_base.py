# tests/plugins/clients/test_audited_client_base.py
"""Tests for AuditedClientBase thread safety."""

import threading
from unittest.mock import MagicMock

import pytest

from elspeth.plugins.clients.base import AuditedClientBase


class ConcreteAuditedClient(AuditedClientBase):
    """Concrete implementation for testing."""

    pass


class TestCallIndexThreadSafety:
    """Test that _next_call_index is thread-safe."""

    def test_concurrent_call_index_no_duplicates(self) -> None:
        """Multiple threads should get unique call indices."""
        mock_recorder = MagicMock()
        client = ConcreteAuditedClient(
            recorder=mock_recorder,
            state_id="test-state",
        )

        indices: list[int] = []
        lock = threading.Lock()
        barrier = threading.Barrier(10)  # Synchronize thread starts

        def get_indices(count: int) -> None:
            barrier.wait()  # All threads start simultaneously
            for _ in range(count):
                idx = client._next_call_index()
                with lock:
                    indices.append(idx)

        # Spawn 10 threads, each getting 100 indices
        threads = [threading.Thread(target=get_indices, args=(100,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 1000 indices should be unique
        assert len(indices) == 1000
        assert len(set(indices)) == 1000, "Duplicate call indices detected!"

        # Should be 0-999
        assert sorted(indices) == list(range(1000))

    @pytest.mark.parametrize("iteration", range(10))
    def test_concurrent_call_index_repeated(self, iteration: int) -> None:
        """Repeated test to increase chance of catching race conditions."""
        mock_recorder = MagicMock()
        client = ConcreteAuditedClient(
            recorder=mock_recorder,
            state_id=f"test-state-{iteration}",
        )

        indices: list[int] = []
        lock = threading.Lock()
        barrier = threading.Barrier(20)  # More threads for more contention

        def get_indices(count: int) -> None:
            barrier.wait()  # Maximize contention by starting all at once
            for _ in range(count):
                idx = client._next_call_index()
                with lock:
                    indices.append(idx)

        # 20 threads, 50 indices each = 1000 total
        threads = [threading.Thread(target=get_indices, args=(50,)) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 1000 indices should be unique
        assert len(indices) == 1000
        assert len(set(indices)) == 1000, f"Duplicate call indices detected on iteration {iteration}!"
