"""Performance baseline tests for token expansion deepcopy overhead.

Measures the cost of copy.deepcopy() in expand_token for various row sizes
and expansion ratios. Critical for validating that audit integrity (isolation
of sibling tokens) doesn't create unacceptable performance overhead.

Related: P2-2026-01-21-expand-token-shared-row-data (the fix being measured)
"""

import copy
import time
from typing import Any

import pytest


class TestExpandTokenDeepCopyPerformance:
    """Benchmark deepcopy overhead in expand_token scenarios.

    These tests measure the raw deepcopy cost, isolated from database
    operations, to establish baselines for the token expansion path.
    """

    @staticmethod
    def _create_small_row() -> dict[str, Any]:
        """Small flat row (~100 bytes serialized)."""
        return {
            "id": 12345,
            "name": "test_record",
            "value": 42.5,
            "active": True,
        }

    @staticmethod
    def _create_medium_row() -> dict[str, Any]:
        """Medium nested row (~5KB serialized)."""
        return {
            "id": 12345,
            "metadata": {
                "source": "api",
                "timestamp": "2026-01-29T12:00:00Z",
                "tags": ["important", "validated", "processed"],
            },
            "payload": {
                "items": [{"sku": f"SKU-{i}", "quantity": i * 10, "price": i * 1.5} for i in range(100)],
            },
            "nested": {
                "level1": {
                    "level2": {
                        "level3": {"deep_value": "found"},
                    },
                },
            },
        }

    @staticmethod
    def _create_large_row() -> dict[str, Any]:
        """Large LLM-response-like row (~50KB serialized)."""
        return {
            "id": 12345,
            "llm_response": {
                "content": "x" * 40000,  # ~40KB of content
                "model": "gpt-4",
                "usage": {"prompt_tokens": 1000, "completion_tokens": 2000},
            },
            "extracted_entities": [
                {
                    "type": "PERSON",
                    "value": f"Person {i}",
                    "confidence": 0.95,
                    "span": {"start": i * 10, "end": i * 10 + 8},
                }
                for i in range(200)
            ],
            "classification": {
                "category": "technical",
                "subcategories": ["engineering", "software", "data"],
                "scores": {f"label_{i}": 0.1 * (i % 10) for i in range(50)},
            },
        }

    @pytest.mark.performance
    def test_deepcopy_small_rows_baseline(self) -> None:
        """Baseline: deepcopy cost for small flat rows.

        Measures 1000 deepcopy operations to establish µs-per-copy baseline.
        """
        row = self._create_small_row()
        iterations = 1000

        start = time.perf_counter()
        for _ in range(iterations):
            _ = copy.deepcopy(row)
        elapsed = time.perf_counter() - start

        us_per_copy = (elapsed / iterations) * 1_000_000

        # Baseline: small rows should copy in < 10µs each
        assert us_per_copy < 10, f"Small row deepcopy: {us_per_copy:.2f}µs (expected < 10µs)"

        print(f"\nSmall row deepcopy: {us_per_copy:.2f}µs/copy ({iterations} iterations)")

    @pytest.mark.performance
    def test_deepcopy_medium_rows_baseline(self) -> None:
        """Medium: deepcopy cost for nested structures (~5KB).

        Measures 1000 deepcopy operations for medium complexity rows.
        """
        row = self._create_medium_row()
        iterations = 1000

        start = time.perf_counter()
        for _ in range(iterations):
            _ = copy.deepcopy(row)
        elapsed = time.perf_counter() - start

        us_per_copy = (elapsed / iterations) * 1_000_000

        # Baseline: medium rows should copy in < 200µs each
        # Note: 100-150µs typical on modern hardware, 200µs allows headroom
        assert us_per_copy < 200, f"Medium row deepcopy: {us_per_copy:.2f}µs (expected < 200µs)"

        print(f"\nMedium row deepcopy: {us_per_copy:.2f}µs/copy ({iterations} iterations)")

    @pytest.mark.performance
    def test_deepcopy_large_rows_baseline(self) -> None:
        """Large: deepcopy cost for LLM response payloads (~50KB).

        Measures 100 deepcopy operations for large complex rows.
        """
        row = self._create_large_row()
        iterations = 100

        start = time.perf_counter()
        for _ in range(iterations):
            _ = copy.deepcopy(row)
        elapsed = time.perf_counter() - start

        us_per_copy = (elapsed / iterations) * 1_000_000
        ms_per_copy = us_per_copy / 1000

        # Baseline: large rows should copy in < 5ms each
        assert ms_per_copy < 5, f"Large row deepcopy: {ms_per_copy:.2f}ms (expected < 5ms)"

        print(f"\nLarge row deepcopy: {ms_per_copy:.2f}ms/copy ({iterations} iterations)")

    @pytest.mark.performance
    def test_expand_token_simulation_small(self) -> None:
        """Simulate expand_token with small rows: 1 -> 10 expansion.

        Measures the realistic scenario of expanding one row into 10 children,
        as happens in deaggregation or multi-row transforms.
        """
        row = self._create_small_row()
        expansion_count = 10
        iterations = 100

        start = time.perf_counter()
        for _ in range(iterations):
            # Simulate expand_token loop (tokens.py:269-278)
            _children = [copy.deepcopy(row) for _ in range(expansion_count)]
        elapsed = time.perf_counter() - start

        ms_per_expansion = (elapsed / iterations) * 1000

        # Baseline: 10x expansion of small rows should complete in < 1ms
        assert ms_per_expansion < 1, f"10x small expansion: {ms_per_expansion:.2f}ms (expected < 1ms)"

        print(f"\n10x small row expansion: {ms_per_expansion:.3f}ms/expansion ({iterations} iterations)")

    @pytest.mark.performance
    def test_expand_token_simulation_high_fanout(self) -> None:
        """High fan-out: 1 row -> 100 expanded rows.

        Measures extreme deaggregation scenarios where a single batch
        produces many output rows.
        """
        row = self._create_small_row()
        expansion_count = 100
        iterations = 50

        start = time.perf_counter()
        for _ in range(iterations):
            _children = [copy.deepcopy(row) for _ in range(expansion_count)]
        elapsed = time.perf_counter() - start

        ms_per_expansion = (elapsed / iterations) * 1000

        # Baseline: 100x expansion of small rows should complete in < 10ms
        assert ms_per_expansion < 10, f"100x small expansion: {ms_per_expansion:.2f}ms (expected < 10ms)"

        print(f"\n100x small row expansion: {ms_per_expansion:.2f}ms/expansion ({iterations} iterations)")

    @pytest.mark.performance
    def test_expand_token_simulation_large_high_fanout(self) -> None:
        """Worst case: large rows with high fan-out (1 -> 50).

        Measures the most expensive realistic scenario: LLM responses
        being expanded into multiple output rows.
        """
        row = self._create_large_row()
        expansion_count = 50
        iterations = 10

        start = time.perf_counter()
        for _ in range(iterations):
            _children = [copy.deepcopy(row) for _ in range(expansion_count)]
        elapsed = time.perf_counter() - start

        ms_per_expansion = (elapsed / iterations) * 1000

        # Baseline: 50x expansion of large rows should complete in < 500ms
        # This is the worst case - if this passes, typical usage is fine
        assert ms_per_expansion < 500, f"50x large expansion: {ms_per_expansion:.2f}ms (expected < 500ms)"

        print(f"\n50x large row expansion: {ms_per_expansion:.2f}ms/expansion ({iterations} iterations)")

    @pytest.mark.performance
    def test_shallow_vs_deep_copy_comparison(self) -> None:
        """Reference: compare shallow copy vs deep copy overhead.

        This is for documentation purposes only - shallow copy is NOT
        a valid alternative (would break audit integrity), but helps
        quantify the cost of correctness.
        """
        row = self._create_medium_row()
        iterations = 1000

        # Shallow copy (reference only - not usable in production)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = row.copy()  # Shallow copy
        shallow_elapsed = time.perf_counter() - start

        # Deep copy (what we actually use)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = copy.deepcopy(row)
        deep_elapsed = time.perf_counter() - start

        shallow_us = (shallow_elapsed / iterations) * 1_000_000
        deep_us = (deep_elapsed / iterations) * 1_000_000
        overhead_factor = deep_us / shallow_us if shallow_us > 0 else float("inf")

        print("\nMedium row copy comparison:")
        print(f"  Shallow copy: {shallow_us:.2f}µs")
        print(f"  Deep copy: {deep_us:.2f}µs")
        print(f"  Overhead factor: {overhead_factor:.1f}x")
        print("  (Deep copy is required for audit integrity)")

        # No assertion - this is informational only


class TestMemoryAmplification:
    """Measure memory impact of deepcopy in expansion scenarios."""

    @pytest.mark.performance
    def test_memory_amplification_factor(self) -> None:
        """Measure memory amplification from expansion.

        When 1 row expands to N rows via deepcopy, memory usage
        becomes N * sizeof(row). This test documents that cost.
        """
        import sys

        # Create a medium-sized row
        row: dict[str, Any] = {
            "id": 12345,
            "data": [{"value": i, "label": f"item_{i}"} for i in range(100)],
        }

        # Measure single row size (approximate via sys.getsizeof recursively)
        def get_deep_size(obj: Any, seen: set[int] | None = None) -> int:
            """Recursively calculate object memory usage."""
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                size += sum(get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                size += sum(get_deep_size(item, seen) for item in obj)
            return size

        single_size = get_deep_size(row)
        expansion_count = 10

        # Create expanded copies
        expanded = [copy.deepcopy(row) for _ in range(expansion_count)]
        total_size = sum(get_deep_size(r) for r in expanded)

        amplification = total_size / single_size

        print(f"\nMemory amplification ({expansion_count}x expansion):")
        print(f"  Single row: {single_size:,} bytes")
        print(f"  {expansion_count} copies: {total_size:,} bytes")
        print(f"  Amplification factor: {amplification:.2f}x (expected: ~{expansion_count}x)")

        # Memory should scale linearly with expansion count (within 20% tolerance)
        expected_min = expansion_count * 0.8
        expected_max = expansion_count * 1.2
        assert expected_min <= amplification <= expected_max, (
            f"Memory amplification {amplification:.2f}x outside expected range [{expected_min}, {expected_max}]"
        )
