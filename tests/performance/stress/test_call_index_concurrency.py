# tests/performance/stress/test_call_index_concurrency.py
"""Concurrent stress tests for allocate_call_index thread safety.

Validates that ExecutionRepository.allocate_call_index() and
allocate_operation_call_index() correctly serialize concurrent access
via threading.Lock, producing contiguous gap-free index sequences
with no duplicates.

Target code: src/elspeth/core/landscape/execution_repository.py
(allocate_call_index, allocate_operation_call_index)
"""

from __future__ import annotations

import threading

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory

pytestmark = pytest.mark.stress

NUM_THREADS = 8
CALLS_PER_THREAD = 500

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


@pytest.fixture
def factory() -> RecorderFactory:
    """Create a RecorderFactory with in-memory database.

    Pre-seeds per-thread state_ids by calling allocate_call_index once
    from the main thread. This ensures the DB-seeding slow path runs on
    the main thread (where the SQLite connection was created), avoiding
    SQLite threading limitations with SingletonThreadPool.
    """
    db = LandscapeDB.in_memory()
    fact = RecorderFactory(db)
    # Pre-seed per-thread state_ids on the main thread so worker threads
    # only hit the fast path (in-memory counter). The loop covers state_0
    # through state_7, which includes state_1 used by the shared-key test.
    # Each allocate consumes index 0; tests account for this.
    for i in range(NUM_THREADS):
        fact.execution.allocate_call_index(f"state_{i}")
    return fact


@pytest.fixture
def factory_with_operation() -> tuple[RecorderFactory, str]:
    """Create a factory with a run, node, and operation for operation call index tests.

    Pre-seeds operation call indices on the main thread.
    """
    db = LandscapeDB.in_memory()
    fact = RecorderFactory(db)
    fact.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
    fact.data_flow.register_node(
        run_id="run-1",
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=_DYNAMIC_SCHEMA,
    )
    op = fact.execution.begin_operation("run-1", "source-0", "source_load")
    # Pre-seed the operation call index from the main thread
    fact.execution.allocate_operation_call_index(op.operation_id)
    return fact, op.operation_id


class TestAllocateCallIndexConcurrency:
    """Thread-safety stress tests for allocate_call_index."""

    def test_concurrent_same_state_id(self, factory: RecorderFactory) -> None:
        """8 threads allocating indices for the same state_id produce contiguous set.

        All threads call allocate_call_index("state_1") 500 times each.
        The returned indices must form {1, 2, ..., 4000} with no gaps or duplicates.
        (Index 0 was consumed by the pre-seed in the fixture.)
        This validates the lock correctly serializes access for a shared state_id.
        """
        barrier = threading.Barrier(NUM_THREADS)
        all_indices: list[list[int]] = [[] for _ in range(NUM_THREADS)]

        def worker(thread_idx: int) -> None:
            barrier.wait()
            local: list[int] = []
            for _ in range(CALLS_PER_THREAD):
                idx = factory.execution.allocate_call_index("state_1")
                local.append(idx)
            all_indices[thread_idx] = local

        threads = [threading.Thread(target=worker, args=(i,), name=f"ci-same-{i}") for i in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), f"Thread {t.name} did not finish within timeout"

        # Flatten and verify contiguous set with no gaps or duplicates
        flat = [idx for batch in all_indices for idx in batch]
        expected_total = NUM_THREADS * CALLS_PER_THREAD
        assert len(flat) == expected_total
        # Indices start at 1 (0 consumed by pre-seed)
        assert set(flat) == set(range(1, expected_total + 1))

    def test_concurrent_independent_state_ids(self, factory: RecorderFactory) -> None:
        """8 threads with unique state_ids produce independent contiguous sequences.

        Each thread calls allocate_call_index with its own state_id 500 times.
        Each state_id's indices must independently form {1, 2, ..., 500}.
        (Index 0 was consumed by the pre-seed in the fixture.)
        This validates per-state-id counters don't interfere across threads.
        """
        barrier = threading.Barrier(NUM_THREADS)
        all_indices: list[list[int]] = [[] for _ in range(NUM_THREADS)]

        def worker(thread_idx: int) -> None:
            barrier.wait()
            state_id = f"state_{thread_idx}"
            local: list[int] = []
            for _ in range(CALLS_PER_THREAD):
                idx = factory.execution.allocate_call_index(state_id)
                local.append(idx)
            all_indices[thread_idx] = local

        threads = [threading.Thread(target=worker, args=(i,), name=f"ci-indep-{i}") for i in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), f"Thread {t.name} did not finish within timeout"

        # Each thread's state_id should have an independent {1..500} sequence
        for thread_idx in range(NUM_THREADS):
            indices = all_indices[thread_idx]
            assert len(indices) == CALLS_PER_THREAD
            assert set(indices) == set(range(1, CALLS_PER_THREAD + 1)), (
                f"Thread {thread_idx} (state_{thread_idx}): expected {{1..{CALLS_PER_THREAD}}}, got {len(set(indices))} unique values"
            )


class TestAllocateOperationCallIndexConcurrency:
    """Thread-safety stress tests for allocate_operation_call_index (H6)."""

    def test_concurrent_same_operation(
        self,
        factory_with_operation: tuple[RecorderFactory, str],
    ) -> None:
        """8 threads allocating indices for the same operation_id produce contiguous set.

        Index 0 was consumed by the pre-seed. Remaining indices: {1..4000}.
        """
        factory, operation_id = factory_with_operation
        barrier = threading.Barrier(NUM_THREADS)
        all_indices: list[list[int]] = [[] for _ in range(NUM_THREADS)]

        def worker(thread_idx: int) -> None:
            barrier.wait()
            local: list[int] = []
            for _ in range(CALLS_PER_THREAD):
                idx = factory.execution.allocate_operation_call_index(operation_id)
                local.append(idx)
            all_indices[thread_idx] = local

        threads = [threading.Thread(target=worker, args=(i,), name=f"oci-same-{i}") for i in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), f"Thread {t.name} did not finish within timeout"

        flat = [idx for batch in all_indices for idx in batch]
        expected_total = NUM_THREADS * CALLS_PER_THREAD
        assert len(flat) == expected_total
        assert set(flat) == set(range(1, expected_total + 1))

    def test_concurrent_independent_operations(
        self,
        factory_with_operation: tuple[RecorderFactory, str],
    ) -> None:
        """8 threads with independent operation_ids produce independent sequences.

        Creates additional operations and pre-seeds each. Each thread's indices
        must independently form {1..500}.
        """
        factory, _op_id = factory_with_operation

        # Create independent operations for each thread
        operation_ids: list[str] = []
        for _i in range(NUM_THREADS):
            # Reuse the same run/node but create separate operations
            # (operation_ids are unique, not constrained per-node)
            op = factory.execution.begin_operation("run-1", "source-0", "source_load")
            # Pre-seed on main thread
            factory.execution.allocate_operation_call_index(op.operation_id)
            operation_ids.append(op.operation_id)

        barrier = threading.Barrier(NUM_THREADS)
        all_indices: list[list[int]] = [[] for _ in range(NUM_THREADS)]

        def worker(thread_idx: int) -> None:
            barrier.wait()
            local: list[int] = []
            for _ in range(CALLS_PER_THREAD):
                idx = factory.execution.allocate_operation_call_index(operation_ids[thread_idx])
                local.append(idx)
            all_indices[thread_idx] = local

        threads = [threading.Thread(target=worker, args=(i,), name=f"oci-indep-{i}") for i in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), f"Thread {t.name} did not finish within timeout"

        for thread_idx in range(NUM_THREADS):
            indices = all_indices[thread_idx]
            assert len(indices) == CALLS_PER_THREAD
            assert set(indices) == set(range(1, CALLS_PER_THREAD + 1)), (
                f"Thread {thread_idx} (op {operation_ids[thread_idx]}): "
                f"expected {{1..{CALLS_PER_THREAD}}}, got {len(set(indices))} unique values"
            )
