"""Performance benchmarks for token expansion deepcopy overhead.

Migrated from tests/performance/test_token_expansion_performance.py.
Measures copy.deepcopy() cost for various row sizes and expansion ratios.
Critical for validating that audit integrity (isolation of sibling tokens)
doesn't create unacceptable performance overhead.

Related: P2-2026-01-21-expand-token-shared-row-data
"""

from __future__ import annotations

import copy
import time
from typing import Any

import pytest

from tests.performance.benchmarks._deep_size import deep_sizeof


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
                "content": "x" * 40000,
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

        Measures 1000 deepcopy operations to establish us-per-copy baseline.
        """
        row = self._create_small_row()
        iterations = 1000

        start = time.perf_counter()
        for _ in range(iterations):
            _ = copy.deepcopy(row)
        elapsed = time.perf_counter() - start

        us_per_copy = (elapsed / iterations) * 1_000_000

        # Baseline: small rows should copy in < 10us each
        assert us_per_copy < 10, f"Small row deepcopy: {us_per_copy:.2f}us (expected < 10us)"

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

        # Baseline: medium rows should copy in < 200us each
        # Note: 100-150us typical on modern hardware, 200us allows headroom
        assert us_per_copy < 200, f"Medium row deepcopy: {us_per_copy:.2f}us (expected < 200us)"

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
            _children = [copy.deepcopy(row) for _ in range(expansion_count)]
        elapsed = time.perf_counter() - start

        ms_per_expansion = (elapsed / iterations) * 1000

        # Baseline: 10x expansion of small rows should complete in < 1ms
        assert ms_per_expansion < 1, f"10x small expansion: {ms_per_expansion:.2f}ms (expected < 1ms)"

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
        assert ms_per_expansion < 500, f"50x large expansion: {ms_per_expansion:.2f}ms (expected < 500ms)"

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
            _ = row.copy()
        shallow_us = (time.perf_counter() - start) / iterations * 1_000_000

        # Deep copy (what we actually use)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = copy.deepcopy(row)
        deep_us = (time.perf_counter() - start) / iterations * 1_000_000

        # Deep copy must be slower than shallow (sanity check) and
        # must complete in reasonable time.
        assert deep_us > shallow_us, "Deep copy should be slower than shallow copy"
        assert deep_us < 500, f"Deep copy too slow: {deep_us:.2f}us (expected < 500us)"


class TestMemoryAmplification:
    """Measure memory impact of deepcopy in expansion scenarios."""

    @pytest.mark.performance
    def test_memory_amplification_factor(self) -> None:
        """Measure memory amplification from expansion.

        When 1 row expands to N rows via deepcopy, memory usage
        becomes N * sizeof(row). This test documents that cost.
        """

        # Create a medium-sized row
        row: dict[str, Any] = {
            "id": 12345,
            "data": [{"value": i, "label": f"item_{i}"} for i in range(100)],
        }

        single_size = deep_sizeof(row)
        expansion_count = 10

        # Create expanded copies
        expanded = [copy.deepcopy(row) for _ in range(expansion_count)]
        total_size = sum(deep_sizeof(r) for r in expanded)

        amplification = total_size / single_size

        # Memory should scale linearly with expansion count (within 20% tolerance)
        expected_min = expansion_count * 0.8
        expected_max = expansion_count * 1.2
        assert expected_min <= amplification <= expected_max, (
            f"Memory amplification {amplification:.2f}x outside expected range [{expected_min}, {expected_max}]"
        )
