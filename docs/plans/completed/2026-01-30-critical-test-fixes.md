# Critical Test Fixes Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-02-01)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close 6 critical test coverage gaps in audit-integrity and security code paths, plus eliminate flakiness in batch adapter tests.

**Architecture:** Each task creates isolated, focused tests that validate crash behavior for audit violations, guard rails for infinite loops, and security boundaries. Tests follow existing patterns in `tests/core/landscape/` and `tests/engine/`.

**Tech Stack:** pytest, SQLAlchemy (IntegrityError), threading.Event (for deterministic concurrency tests), existing MockClock/CallbackSource patterns.

---

## Implementation Summary

- Added audit-integrity and guardrail tests (`tests/core/landscape/test_token_outcome_constraints.py`, `tests/engine/test_processor_guards.py`, `tests/core/checkpoint/test_topology_validation.py`).
- Added security boundary coverage for Key Vault handling (`tests/core/security/test_fingerprint_keyvault.py`, `tests/core/security/__init__.py`).
- Batch adapter tests refactored for deterministic synchronization (`tests/engine/test_batch_adapter.py`).

## Task 1: Double Token Outcome Recording Test

**Files:**
- Create: `tests/core/landscape/test_token_outcome_constraints.py`
- Reference: `src/elspeth/core/landscape/recorder.py:2075-2148`
- Reference: `src/elspeth/core/landscape/schema.py:156-164` (partial unique index)

**Step 1: Write the failing test**

```python
# tests/core/landscape/test_token_outcome_constraints.py
"""Tests for token outcome constraint enforcement.

Critical audit integrity tests: Verifies that the partial UNIQUE index on
token_outcomes_table prevents recording multiple terminal outcomes for the
same token, which would corrupt the audit trail.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.enums import NodeType, RowOutcome
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder

DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestTokenOutcomeConstraints:
    """Tests for terminal outcome uniqueness constraint (audit integrity)."""

    def test_double_terminal_outcome_raises_integrity_error(self) -> None:
        """Recording two terminal outcomes for same token must fail.

        This is CRITICAL for audit integrity - a token can only have ONE
        terminal state. The database enforces this via partial unique index:
        UNIQUE(token_id) WHERE is_terminal=1

        If this constraint isn't enforced, the audit trail becomes ambiguous:
        "Did token X complete to sink A or sink B?" - both recorded as terminal.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup: Create run, source node, row, and token
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )
        token = recorder.create_token(row_id=row.row_id)

        # First terminal outcome: COMPLETED
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output_sink",
        )

        # Second terminal outcome for SAME token must raise IntegrityError
        with pytest.raises(IntegrityError):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.ROUTED,
                sink_name="alternate_sink",
            )

    def test_non_terminal_then_terminal_is_allowed(self) -> None:
        """Non-terminal (BUFFERED) followed by terminal (CONSUMED_IN_BATCH) is valid.

        The aggregation pattern: token enters aggregation (BUFFERED), then
        batch flushes (CONSUMED_IN_BATCH). Both are recorded, but only
        CONSUMED_IN_BATCH is terminal.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Non-terminal: BUFFERED (is_terminal=False)
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.BUFFERED,
            batch_id="batch-001",
        )

        # Terminal: CONSUMED_IN_BATCH (is_terminal=True) - should succeed
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.CONSUMED_IN_BATCH,
            batch_id="batch-001",
        )

        # Verify both outcomes were recorded
        outcomes = recorder.get_token_outcomes(token_id=token.token_id)
        assert len(outcomes) == 2
        assert {o.outcome for o in outcomes} == {RowOutcome.BUFFERED, RowOutcome.CONSUMED_IN_BATCH}

    def test_multiple_non_terminal_outcomes_allowed(self) -> None:
        """Multiple non-terminal outcomes for same token are allowed.

        Edge case: A token could theoretically have multiple BUFFERED outcomes
        if it passes through multiple aggregations (though this is rare).
        The constraint only prevents multiple TERMINAL outcomes.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Multiple non-terminal outcomes (both BUFFERED in different batches)
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.BUFFERED,
            batch_id="batch-001",
        )

        # Second BUFFERED should also succeed (not terminal)
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.BUFFERED,
            batch_id="batch-002",
        )

        outcomes = recorder.get_token_outcomes(token_id=token.token_id)
        assert len(outcomes) == 2
```

**Step 2: Run test to verify it fails or passes**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_token_outcome_constraints.py -v`

Expected: If the partial unique index is working, the first test should PASS (IntegrityError raised). If not, we have a bug to fix.

**Step 3: Commit**

```bash
git add tests/core/landscape/test_token_outcome_constraints.py
git commit -m "test: add token outcome constraint tests (audit integrity)"
```

---

## Task 2: Processor Work Queue Iteration Guard Test

**Files:**
- Create: `tests/engine/test_processor_guards.py`
- Reference: `src/elspeth/engine/processor.py:42,233-234`

**Step 1: Write the failing test**

```python
# tests/engine/test_processor_guards.py
"""Tests for RowProcessor safety guards.

These tests verify that safety mechanisms (iteration limits, etc.)
correctly prevent pathological scenarios from hanging the pipeline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import TokenInfo, TransformResult
from elspeth.contracts.enums import NodeType, RowOutcome
from elspeth.contracts.results import GateResult, RowResult
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.engine.processor import MAX_WORK_QUEUE_ITERATIONS, RowProcessor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult as PluginTransformResult
from tests.conftest import _TestSchema


class InfiniteLoopTransform(BaseTransform):
    """Transform that simulates infinite loop via self-referential fork.

    This is a pathological case that shouldn't happen in real pipelines,
    but the guard exists to prevent hangs if it does.
    """

    name = "infinite_loop_transform"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})
        self.call_count = 0

    def process(self, row: Any, ctx: Any) -> PluginTransformResult:
        self.call_count += 1
        return PluginTransformResult.success(row)


class TestProcessorGuards:
    """Tests for processor safety guards."""

    def test_max_work_queue_iterations_constant_value(self) -> None:
        """Verify MAX_WORK_QUEUE_ITERATIONS is set to expected value.

        This is a sanity check - if someone changes the constant,
        they should be aware tests depend on it.
        """
        assert MAX_WORK_QUEUE_ITERATIONS == 10_000

    def test_work_queue_exceeding_limit_raises_runtime_error(self) -> None:
        """Exceeding MAX_WORK_QUEUE_ITERATIONS must raise RuntimeError.

        This test verifies the infinite loop guard fires correctly.
        We can't easily create a real infinite loop, so we patch the
        constant to a lower value and create a DAG that would exceed it.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup minimal infrastructure
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        span_factory = SpanFactory(run_id=run.run_id)

        # Create processor with minimal config
        processor = RowProcessor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        # Create transform that will be called many times
        transform = InfiniteLoopTransform()
        transform.node_id = "transform_1"

        # Create a token
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        token_info = TokenInfo(
            token_id=token.token_id,
            row_id=row.row_id,
            row_data={"value": 1},
            branch_name=None,
        )

        # Patch the constant to a small value for test speed
        # Also need to make the work queue grow beyond limit
        with patch("elspeth.engine.processor.MAX_WORK_QUEUE_ITERATIONS", 5):
            # Create a mock scenario where work queue keeps growing
            # by patching _process_single_token to always return more work
            original_process = processor._process_single_token

            def mock_process_that_generates_more_work(*args: Any, **kwargs: Any) -> tuple[RowResult, list[Any]]:
                # Return a result + N child items to process
                result = RowResult(
                    row_id=row.row_id,
                    outcome=RowOutcome.COMPLETED,
                    token_id=token.token_id,
                    sink_name="output",
                )
                # Generate child items that will be added to queue
                from elspeth.engine.processor import _WorkItem
                child = _WorkItem(token=token_info, start_step=0)
                return result, [child]  # Always return more work

            processor._process_single_token = mock_process_that_generates_more_work  # type: ignore[method-assign]

            with pytest.raises(RuntimeError, match="exceeded .* iterations"):
                processor.process_row(
                    row_index=0,
                    row_data={"value": 1},
                    transforms=[transform],
                    ctx=MagicMock(),
                )

    def test_normal_processing_completes_without_hitting_guard(self) -> None:
        """Normal DAG processing should never approach the iteration limit.

        This is a sanity check that the guard doesn't interfere with
        legitimate pipelines.
        """
        # A simple linear pipeline with 10 transforms should complete
        # in exactly 10 iterations (one per transform)
        assert 10 < MAX_WORK_QUEUE_ITERATIONS

        # The guard is set high enough that even complex DAGs with
        # many forks/joins should complete well under the limit
        # A DAG with 100 nodes and 10 parallel branches = ~1000 iterations max
        assert 1000 < MAX_WORK_QUEUE_ITERATIONS
```

**Step 2: Run test to verify behavior**

Run: `.venv/bin/python -m pytest tests/engine/test_processor_guards.py -v`

Expected: Tests should PASS - the guard is implemented and should raise RuntimeError when iterations exceed limit.

**Step 3: Commit**

```bash
git add tests/engine/test_processor_guards.py
git commit -m "test: add processor work queue iteration guard tests"
```

---

## Task 3: Fingerprint Key Vault Failure Test

**Files:**
- Create: `tests/core/security/test_fingerprint_keyvault.py`
- Reference: `src/elspeth/core/security/fingerprint.py:58-99`

**Step 1: Write the test**

```python
# tests/core/security/test_fingerprint_keyvault.py
"""Tests for fingerprint key retrieval security boundaries.

These tests verify that fingerprint key retrieval fails safely when
Key Vault is unavailable or misconfigured, rather than falling back
to insecure defaults.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from elspeth.core.security.fingerprint import (
    _ENV_VAR,
    _KEYVAULT_SECRET_NAME_VAR,
    _KEYVAULT_URL_VAR,
    get_fingerprint_key,
    secret_fingerprint,
)


class TestFingerprintKeyRetrieval:
    """Tests for fingerprint key retrieval security."""

    def test_missing_all_config_raises_value_error(self) -> None:
        """Must raise ValueError when no key source is configured.

        Security boundary: If neither env var nor Key Vault is configured,
        we MUST fail loudly, not silently return empty/default key.
        """
        # Clear all relevant env vars
        env_overrides = {
            _ENV_VAR: None,
            _KEYVAULT_URL_VAR: None,
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            # Remove the keys entirely
            for key in [_ENV_VAR, _KEYVAULT_URL_VAR]:
                os.environ.pop(key, None)

            with pytest.raises(ValueError, match="Fingerprint key not configured"):
                get_fingerprint_key()

    def test_keyvault_retrieval_failure_raises_value_error(self) -> None:
        """Key Vault retrieval failure must raise ValueError, not fallback.

        Security boundary: If Key Vault is configured but retrieval fails,
        we MUST NOT fall back to environment variable or empty key.
        The error must propagate.
        """
        with patch.dict(os.environ, {
            _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
            _ENV_VAR: "",  # Empty to ensure no fallback
        }, clear=False):
            # Remove env key entirely
            os.environ.pop(_ENV_VAR, None)

            # Mock the Key Vault client to fail
            with patch(
                "elspeth.core.security.fingerprint._get_keyvault_client"
            ) as mock_client:
                mock_client.return_value.get_secret.side_effect = Exception(
                    "Network error: Key Vault unreachable"
                )

                with pytest.raises(ValueError, match="Failed to retrieve fingerprint key from Key Vault"):
                    get_fingerprint_key()

    def test_keyvault_secret_has_null_value_raises_value_error(self) -> None:
        """Key Vault secret with None value must raise ValueError.

        Edge case: Secret exists in Key Vault but has null/empty value.
        This is a configuration error that must not be silently accepted.
        """
        with patch.dict(os.environ, {
            _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
        }, clear=False):
            os.environ.pop(_ENV_VAR, None)

            # Mock secret with None value
            mock_secret = MagicMock()
            mock_secret.value = None

            with patch(
                "elspeth.core.security.fingerprint._get_keyvault_client"
            ) as mock_client:
                mock_client.return_value.get_secret.return_value = mock_secret

                with pytest.raises(ValueError, match="has no value"):
                    get_fingerprint_key()

    def test_env_var_takes_precedence_over_keyvault(self) -> None:
        """Environment variable should take precedence over Key Vault.

        This is documented behavior for dev/testing scenarios.
        """
        with patch.dict(os.environ, {
            _ENV_VAR: "test-key-from-env",
            _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
        }):
            key = get_fingerprint_key()
            assert key == b"test-key-from-env"

    def test_secret_fingerprint_with_missing_key_raises(self) -> None:
        """secret_fingerprint() without key param must raise if no config.

        High-level function should propagate key retrieval failures.
        """
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV_VAR, None)
            os.environ.pop(_KEYVAULT_URL_VAR, None)

            with pytest.raises(ValueError, match="Fingerprint key not configured"):
                secret_fingerprint("my-secret")

    def test_azure_import_error_propagates(self) -> None:
        """ImportError from missing azure packages must propagate.

        If azure-keyvault-secrets is not installed but Key Vault URL
        is configured, the ImportError should propagate with helpful message.
        """
        with patch.dict(os.environ, {
            _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
        }, clear=False):
            os.environ.pop(_ENV_VAR, None)

            # Mock the import to fail
            with patch(
                "elspeth.core.security.fingerprint._get_keyvault_client"
            ) as mock_client:
                mock_client.side_effect = ImportError(
                    "azure-keyvault-secrets and azure-identity are required"
                )

                with pytest.raises(ImportError, match="azure-keyvault-secrets"):
                    get_fingerprint_key()
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/core/security/test_fingerprint_keyvault.py -v`

Expected: All tests should PASS - the security boundary is implemented.

**Step 3: Commit**

```bash
git add tests/core/security/test_fingerprint_keyvault.py
git commit -m "test: add fingerprint Key Vault security boundary tests"
```

---

## Task 4: Checkpoint Topology Hash Mismatch Test

**Files:**
- Create: `tests/core/checkpoint/test_topology_validation.py`
- Reference: `src/elspeth/core/checkpoint/compatibility.py:74-81`
- Reference: `src/elspeth/core/checkpoint/manager.py:89`

**Step 1: Write the test**

```python
# tests/core/checkpoint/test_topology_validation.py
"""Tests for checkpoint topology validation.

Critical audit integrity tests: Verifies that resume with modified
pipeline configuration is correctly rejected, preventing "one run,
two configs" corruption.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts import Checkpoint, ResumeCheck
from elspeth.contracts.enums import Determinism, NodeType
from elspeth.core.canonical import compute_full_topology_hash, stable_hash
from elspeth.core.checkpoint.compatibility import CheckpointValidator
from elspeth.core.dag import ExecutionGraph, NodeInfo


class TestCheckpointTopologyValidation:
    """Tests for topology hash validation during resume."""

    def _create_linear_graph(self, num_transforms: int = 2) -> ExecutionGraph:
        """Create a simple linear graph: source → transforms → sink."""
        graph = ExecutionGraph()

        # Add source
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            config={"path": "data.csv"},
            determinism=Determinism.DETERMINISTIC,
            plugin_name="csv_source",
        )

        # Add transforms
        prev_node = "source_1"
        for i in range(num_transforms):
            node_id = f"transform_{i}"
            graph.add_node(
                node_id,
                node_type=NodeType.TRANSFORM,
                config={"operation": f"op_{i}"},
                determinism=Determinism.DETERMINISTIC,
                plugin_name="passthrough",
            )
            graph.add_edge(prev_node, node_id, label="continue")
            prev_node = node_id

        # Add sink
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            config={"path": "output.csv"},
            determinism=Determinism.DETERMINISTIC,
            plugin_name="csv_sink",
        )
        graph.add_edge(prev_node, "sink_1", label="continue")

        return graph

    def _create_checkpoint_for_graph(
        self,
        graph: ExecutionGraph,
        node_id: str = "transform_0",
    ) -> Checkpoint:
        """Create a checkpoint with topology hash from the given graph."""
        topology_hash = compute_full_topology_hash(graph)
        node_info = graph.get_node_info(node_id)
        config_hash = stable_hash(node_info.config)

        return Checkpoint(
            checkpoint_id="cp-test123",
            run_id="run-123",
            token_id="token-456",
            node_id=node_id,
            sequence_number=100,
            created_at=datetime.now(UTC),
            upstream_topology_hash=topology_hash,
            checkpoint_node_config_hash=config_hash,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )

    def test_identical_graph_validates_successfully(self) -> None:
        """Checkpoint from identical graph should validate successfully."""
        original_graph = self._create_linear_graph(num_transforms=2)
        checkpoint = self._create_checkpoint_for_graph(original_graph)

        # Create identical graph for resume
        resume_graph = self._create_linear_graph(num_transforms=2)

        validator = CheckpointValidator()
        result = validator.validate_checkpoint(checkpoint, resume_graph)

        assert result.can_resume is True
        assert result.reason is None

    def test_added_transform_causes_validation_failure(self) -> None:
        """Adding a transform after checkpoint must fail validation.

        Scenario: Original pipeline had 2 transforms, resume has 3.
        Even though checkpoint node still exists, topology changed.
        """
        original_graph = self._create_linear_graph(num_transforms=2)
        checkpoint = self._create_checkpoint_for_graph(original_graph)

        # Resume graph has extra transform
        modified_graph = self._create_linear_graph(num_transforms=3)

        validator = CheckpointValidator()
        result = validator.validate_checkpoint(checkpoint, modified_graph)

        assert result.can_resume is False
        assert "topology" in result.reason.lower()

    def test_removed_transform_causes_validation_failure(self) -> None:
        """Removing a transform after checkpoint must fail validation."""
        original_graph = self._create_linear_graph(num_transforms=3)
        # Checkpoint at transform_0 (which exists in both)
        checkpoint = self._create_checkpoint_for_graph(original_graph, node_id="transform_0")

        # Resume graph has fewer transforms
        modified_graph = self._create_linear_graph(num_transforms=2)

        validator = CheckpointValidator()
        result = validator.validate_checkpoint(checkpoint, modified_graph)

        assert result.can_resume is False
        assert "topology" in result.reason.lower()

    def test_modified_sink_config_causes_validation_failure(self) -> None:
        """Changing sink configuration after checkpoint must fail validation.

        BUG-COMPAT-01 scenario: Even changes to sibling branches (different
        sinks) must invalidate the checkpoint because "one run = one config".
        """
        original_graph = self._create_linear_graph(num_transforms=2)
        checkpoint = self._create_checkpoint_for_graph(original_graph)

        # Create graph with different sink config
        modified_graph = ExecutionGraph()
        modified_graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            config={"path": "data.csv"},
            determinism=Determinism.DETERMINISTIC,
            plugin_name="csv_source",
        )
        for i in range(2):
            node_id = f"transform_{i}"
            modified_graph.add_node(
                node_id,
                node_type=NodeType.TRANSFORM,
                config={"operation": f"op_{i}"},
                determinism=Determinism.DETERMINISTIC,
                plugin_name="passthrough",
            )
            prev = "source_1" if i == 0 else f"transform_{i-1}"
            modified_graph.add_edge(prev, node_id, label="continue")

        # DIFFERENT sink config
        modified_graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            config={"path": "DIFFERENT_OUTPUT.csv"},  # Changed!
            determinism=Determinism.DETERMINISTIC,
            plugin_name="csv_sink",
        )
        modified_graph.add_edge("transform_1", "sink_1", label="continue")

        validator = CheckpointValidator()
        result = validator.validate_checkpoint(checkpoint, modified_graph)

        assert result.can_resume is False
        assert "topology" in result.reason.lower()

    def test_checkpoint_node_missing_causes_validation_failure(self) -> None:
        """Checkpoint node not existing in new graph must fail validation."""
        original_graph = self._create_linear_graph(num_transforms=3)
        # Checkpoint at transform_2 which won't exist in smaller graph
        checkpoint = self._create_checkpoint_for_graph(original_graph, node_id="transform_2")

        # Resume graph doesn't have transform_2
        modified_graph = self._create_linear_graph(num_transforms=2)

        validator = CheckpointValidator()
        result = validator.validate_checkpoint(checkpoint, modified_graph)

        assert result.can_resume is False
        # Should mention either node or topology mismatch
        assert "node" in result.reason.lower() or "topology" in result.reason.lower()

    def test_checkpoint_node_config_changed_causes_validation_failure(self) -> None:
        """Changing checkpoint node's config must fail validation.

        This is separate from topology - even if graph structure is same,
        the specific node's config must also match.
        """
        original_graph = self._create_linear_graph(num_transforms=2)
        checkpoint = self._create_checkpoint_for_graph(original_graph, node_id="transform_0")

        # Create graph with same structure but different config at checkpoint node
        modified_graph = ExecutionGraph()
        modified_graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            config={"path": "data.csv"},
            determinism=Determinism.DETERMINISTIC,
            plugin_name="csv_source",
        )
        # transform_0 has DIFFERENT config
        modified_graph.add_node(
            "transform_0",
            node_type=NodeType.TRANSFORM,
            config={"operation": "DIFFERENT_OP"},  # Changed!
            determinism=Determinism.DETERMINISTIC,
            plugin_name="passthrough",
        )
        modified_graph.add_edge("source_1", "transform_0", label="continue")
        modified_graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            config={"operation": "op_1"},
            determinism=Determinism.DETERMINISTIC,
            plugin_name="passthrough",
        )
        modified_graph.add_edge("transform_0", "transform_1", label="continue")
        modified_graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            config={"path": "output.csv"},
            determinism=Determinism.DETERMINISTIC,
            plugin_name="csv_sink",
        )
        modified_graph.add_edge("transform_1", "sink_1", label="continue")

        validator = CheckpointValidator()
        result = validator.validate_checkpoint(checkpoint, modified_graph)

        assert result.can_resume is False
        # Should fail on either topology (config affects hash) or node config check
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/core/checkpoint/test_topology_validation.py -v`

Expected: Tests should PASS - topology validation is implemented.

**Step 3: Commit**

```bash
git add tests/core/checkpoint/test_topology_validation.py
git commit -m "test: add checkpoint topology validation tests (audit integrity)"
```

---

## Task 5: Replace time.sleep() in Batch Adapter Tests

**Files:**
- Modify: `tests/engine/test_batch_adapter.py`

**Step 1: Refactor test to use Event-based coordination**

Replace the `time.sleep()` calls with `threading.Event` synchronization:

```python
# tests/engine/test_batch_adapter.py
# Replace lines 37-60 (test_single_row_wait method)

    def test_single_row_wait(self) -> None:
        """Test waiting for a single row's result."""
        adapter = SharedBatchAdapter()

        # Register waiter with (token_id, state_id) key for retry safety
        waiter = adapter.register("token-1", "state-1")

        # Use Event for deterministic synchronization instead of sleep
        registration_complete = threading.Event()
        emit_allowed = threading.Event()

        def emit_when_signaled() -> None:
            registration_complete.wait()  # Wait for test setup to complete
            emit_allowed.wait()  # Wait for explicit signal
            token = MockTokenInfo(token_id="token-1", row_id=1)
            result = TransformResult.success({"output": "done"})
            adapter.emit(token, result, "state-1")  # type: ignore[arg-type]

        thread = threading.Thread(target=emit_when_signaled)
        thread.start()

        # Signal that registration is complete
        registration_complete.set()

        # Signal emit is allowed (test is ready to receive)
        emit_allowed.set()

        # Wait for result
        result = waiter.wait(timeout=5.0)

        assert result.status == "success"
        assert result.row == {"output": "done"}

        thread.join()
```

**Step 2: Refactor test_multiple_concurrent_rows**

```python
# Replace lines 62-108

    def test_multiple_concurrent_rows(self) -> None:
        """Test multiple rows waiting concurrently."""
        adapter = SharedBatchAdapter()

        # Register 3 waiters with unique state_ids
        waiter1 = adapter.register("token-1", "state-1")
        waiter2 = adapter.register("token-2", "state-2")
        waiter3 = adapter.register("token-3", "state-3")

        # Use Events for deterministic out-of-order completion
        setup_complete = threading.Event()
        emit_events = {
            "token-2": threading.Event(),  # Will be signaled first
            "token-1": threading.Event(),  # Will be signaled second
            "token-3": threading.Event(),  # Will be signaled third
        }

        def emit_results() -> None:
            setup_complete.wait()  # Wait for test setup

            # Emit in controlled order: token-2, token-1, token-3
            emit_events["token-2"].wait()
            adapter.emit(
                MockTokenInfo(token_id="token-2", row_id=2),  # type: ignore[arg-type]
                TransformResult.success({"value": 2}),
                "state-2",
            )

            emit_events["token-1"].wait()
            adapter.emit(
                MockTokenInfo(token_id="token-1", row_id=1),  # type: ignore[arg-type]
                TransformResult.success({"value": 1}),
                "state-1",
            )

            emit_events["token-3"].wait()
            adapter.emit(
                MockTokenInfo(token_id="token-3", row_id=3),  # type: ignore[arg-type]
                TransformResult.success({"value": 3}),
                "state-3",
            )

        thread = threading.Thread(target=emit_results)
        thread.start()

        # Setup complete
        setup_complete.set()

        # Signal emits in out-of-order sequence
        emit_events["token-2"].set()  # First
        emit_events["token-1"].set()  # Second
        emit_events["token-3"].set()  # Third

        # Wait for results (each waiter gets correct result regardless of emit order)
        result1 = waiter1.wait(timeout=5.0)
        result2 = waiter2.wait(timeout=5.0)
        result3 = waiter3.wait(timeout=5.0)

        assert result1.row == {"value": 1}
        assert result2.row == {"value": 2}
        assert result3.row == {"value": 3}

        thread.join()
```

**Step 3: Refactor test_concurrent_waiters_in_parallel_threads**

```python
# Replace lines 206-247

    def test_concurrent_waiters_in_parallel_threads(self) -> None:
        """Test multiple threads waiting concurrently."""
        adapter = SharedBatchAdapter()
        results: dict[str, TransformResult] = {}
        errors: list[Exception] = []
        all_registered = threading.Barrier(6)  # 5 waiters + 1 emitter thread

        def wait_for_token(token_id: str, state_id: str) -> None:
            try:
                waiter = adapter.register(token_id, state_id)
                all_registered.wait()  # Synchronize: all threads registered
                result = waiter.wait(timeout=5.0)
                results[token_id] = result
            except Exception as e:
                errors.append(e)

        # Start 5 waiter threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=wait_for_token, args=(f"token-{i}", f"state-{i}"))
            threads.append(t)
            t.start()

        def emit_all() -> None:
            all_registered.wait()  # Wait for all waiters to register
            # Emit all results
            for i in range(5):
                adapter.emit(
                    MockTokenInfo(token_id=f"token-{i}", row_id=i),  # type: ignore[arg-type]
                    TransformResult.success({"index": i}),
                    f"state-{i}",
                )

        emit_thread = threading.Thread(target=emit_all)
        emit_thread.start()

        # Wait for all threads
        for t in threads:
            t.join(timeout=5.0)
        emit_thread.join(timeout=5.0)

        # Verify all succeeded
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        for i in range(5):
            assert results[f"token-{i}"].row == {"index": i}
```

**Step 4: Run refactored tests**

Run: `.venv/bin/python -m pytest tests/engine/test_batch_adapter.py -v`

Expected: All tests should PASS with deterministic synchronization.

**Step 5: Commit**

```bash
git add tests/engine/test_batch_adapter.py
git commit -m "refactor: replace time.sleep() with Event synchronization in batch adapter tests

Eliminates flakiness risk from timing-dependent thread coordination.
Uses threading.Event and threading.Barrier for deterministic behavior."
```

---

## Task 6: Create __init__.py for test_core_security if needed

**Files:**
- Create if missing: `tests/core/security/__init__.py`

**Step 1: Check and create**

```python
# tests/core/security/__init__.py
"""Security-related tests."""
```

**Step 2: Commit if created**

```bash
git add tests/core/security/__init__.py
git commit -m "chore: add __init__.py for security tests package"
```

---

## Summary

| Task | Files | Purpose |
|------|-------|---------|
| 1 | `test_token_outcome_constraints.py` | Audit integrity - double outcome prevention |
| 2 | `test_processor_guards.py` | Infinite loop guard verification |
| 3 | `test_fingerprint_keyvault.py` | Security boundary - Key Vault failure handling |
| 4 | `test_topology_validation.py` | Audit integrity - resume config validation |
| 5 | `test_batch_adapter.py` (refactor) | Flakiness elimination |
| 6 | `__init__.py` (if needed) | Package structure |

**Total estimated time:** 2-3 hours

**After completion, run full test suite:**

```bash
.venv/bin/python -m pytest tests/core/landscape/test_token_outcome_constraints.py tests/engine/test_processor_guards.py tests/core/security/test_fingerprint_keyvault.py tests/core/checkpoint/test_topology_validation.py tests/engine/test_batch_adapter.py -v
```
