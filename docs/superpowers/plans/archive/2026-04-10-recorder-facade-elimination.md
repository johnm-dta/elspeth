# Recorder Facade Elimination — Direct Repository Injection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the 93-method `LandscapeRecorder` facade and have all callers inject the specific repositories they need, eliminating a no-legacy-code policy violation and unblocking the Engine API extraction (`elspeth-1119dc22ef`).

**Architecture:** `LandscapeRecorder` is a pure delegation facade over 4 domain repositories (`RunLifecycleRepository`, `ExecutionRepository`, `DataFlowRepository`, `QueryRepository`). Every method is `return self._repo.method(args)` with zero logic. Callers will inject the 1-3 repositories they actually use. For the `contracts/` layer (L0), which cannot import `core/` (L1), we define a structural protocol (`PluginAuditWriter`) that a thin adapter satisfies by composing 3 repositories. A `RecorderFactory` replaces the 11 construction sites that currently call `LandscapeRecorder(db)`.

**Review-driven changes (2026-04-10):** Four-agent review (architecture, systems, Python, quality) identified: (1) missing `get_node_state`, `get_source_field_resolution`, `record_readiness_check` from `PluginAuditWriter`; (2) `display_headers.py` and `rag/transform.py` absent from file map; (3) unjustified `Any` annotations — most types live in L0; (4) `store_payload` semantic mismatch — route through `PayloadStore` directly; (5) deprecated aliases violate no-legacy-code policy — bulk-rename instead; (6) `ResumeState.recorder` rename needs explicit consumer tracing; (7) adapter delegation tests needed; (8) mypy should run earlier; (9) `hasattr` in conformance tests is banned.

**Tech Stack:** Python dataclasses, `typing.Protocol`, pluggy, SQLAlchemy Core, pytest

**Filigree issue:** `elspeth-50521f45ec`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/contracts/audit_protocols.py` | 3 protocols for cross-layer boundary |
| Create | `src/elspeth/core/landscape/factory.py` | `RecorderFactory` — constructs repos from `LandscapeDB` |
| Modify | `src/elspeth/contracts/contexts.py` | Replace `LandscapeRecorder` with protocol types |
| Modify | `src/elspeth/contracts/plugin_context.py` | Replace `LandscapeRecorder` field with protocol type |
| Modify | `src/elspeth/engine/tokens.py` | Accept `DataFlowRepository` |
| Modify | `src/elspeth/engine/executors/state_guard.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/engine/executors/gate.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/engine/executors/transform.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/engine/executors/sink.py` | Accept `ExecutionRepository` + `DataFlowRepository` |
| Modify | `src/elspeth/engine/executors/aggregation.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/engine/coalesce_executor.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/engine/processor.py` | Accept specific repos |
| Modify | `src/elspeth/engine/orchestrator/core.py` | Construct repos via factory, pass to components |
| Modify | `src/elspeth/engine/orchestrator/export.py` | Use factory |
| Modify | `src/elspeth/engine/orchestrator/aggregation.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/engine/orchestrator/types.py` | Replace `LandscapeRecorder` in `ResumeState` |
| Modify | `src/elspeth/core/operations.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/core/landscape/lineage.py` | Accept `QueryRepository` + `DataFlowRepository` |
| Modify | `src/elspeth/core/landscape/exporter.py` | Use factory |
| Modify | `src/elspeth/core/checkpoint/recovery.py` | Use factory |
| Modify | `src/elspeth/mcp/analyzer.py` | Use factory |
| Modify | `src/elspeth/tui/screens/explain_screen.py` | Use factory |
| Modify | `src/elspeth/cli.py` | Use factory |
| Modify | `src/elspeth/plugins/infrastructure/clients/base.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/plugins/infrastructure/clients/http.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/plugins/infrastructure/clients/llm.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/plugins/infrastructure/clients/replayer.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/plugins/infrastructure/clients/verifier.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py` | Accept `ExecutionRepository` |
| Modify | `src/elspeth/plugins/transforms/web_scrape.py` | Replace `store_payload` with PayloadStore |
| Modify | `src/elspeth/plugins/transforms/llm/transform.py` | `PluginAuditWriter` type annotation |
| Modify | `src/elspeth/plugins/transforms/llm/openrouter_batch.py` | `PluginAuditWriter` type annotation |
| Modify | `src/elspeth/plugins/transforms/llm/providers/azure.py` | `PluginAuditWriter` type annotation |
| Modify | `src/elspeth/plugins/transforms/llm/providers/openrouter.py` | `PluginAuditWriter` type annotation |
| Modify | `src/elspeth/plugins/transforms/azure/base.py` | `PluginAuditWriter` type annotation |
| Modify | `src/elspeth/plugins/infrastructure/display_headers.py` | `PluginAuditWriter` type annotation |
| Modify | `src/elspeth/plugins/transforms/rag/transform.py` | `PluginAuditWriter` type annotation |
| Modify | `src/elspeth/core/landscape/__init__.py` | Remove `LandscapeRecorder` export, add factory |
| Delete | `src/elspeth/core/landscape/recorder.py` | The facade itself |
| Modify | `tests/fixtures/landscape.py` | Replace `make_recorder()` with factory |
| Create | `tests/unit/contracts/test_audit_protocols.py` | Protocol structural conformance tests |
| Create | `tests/unit/core/landscape/test_factory.py` | Factory construction tests |

---

## Caller → Repository Mapping

This table drives all the caller-side changes. Each row shows what the caller currently receives (`LandscapeRecorder`) and what it should receive after this change.

| Caller | Methods used | New injection |
|--------|-------------|---------------|
| **TokenManager** | `create_row`, `create_token`, `fork_token`, `coalesce_tokens`, `expand_token` | `DataFlowRepository` |
| **NodeStateGuard** | `begin_node_state`, `complete_node_state` | `ExecutionRepository` |
| **GateExecutor** | `record_routing_event`, `record_routing_events` | `ExecutionRepository` |
| **TransformExecutor** | (indirect via StateGuard) | `ExecutionRepository` |
| **SinkExecutor** | `begin_node_state`, `complete_node_state`, `record_routing_event`, `record_token_outcome`, `register_artifact` | `ExecutionRepository` + `DataFlowRepository` |
| **AggregationExecutor** | `create_batch`, `add_batch_member`, `update_batch_status`, `begin_node_state`, `complete_node_state` | `ExecutionRepository` |
| **CoalesceExecutor** | `begin_node_state`, `complete_node_state`, `get_completed_row_ids_for_nodes` | `ExecutionRepository` |
| **RowProcessor** | `record_token_outcome`, `record_transform_error`, `get_node_contracts`, `update_node_output_contract` + executor construction | `ExecutionRepository` + `DataFlowRepository` |
| **Orchestrator** | Everything — run lifecycle, graph, export | All 4 repos via factory |
| **Plugin clients** | `allocate_call_index`, `record_call`, `find_call_by_request_hash`, `get_call_response_data` | `ExecutionRepository` |
| **PluginContext** | `record_call`, `allocate_call_index`, `record_validation_error`, `record_transform_error`, `get_node_state`, `get_source_field_resolution`, `record_readiness_check` | `PluginAuditWriter` protocol |
| **operations.py** | `begin_operation`, `complete_operation`, `allocate_operation_call_index`, `record_operation_call` | `ExecutionRepository` |
| **lineage.py** | Read-only across QueryRepository + DataFlowRepository | `QueryRepository` + `DataFlowRepository` |
| **exporter.py** | Read-only across all repos | Factory (constructs what it needs) |
| **MCP/TUI/CLI** | Read-only queries | Factory (constructs what it needs) |

---

## Task 1: Define audit protocols in contracts/

Protocols for the L0→L1 boundary. Only needed where `contracts/` code references audit capabilities — specifically `PluginContext` and `contexts.py`.

**Files:**
- Create: `src/elspeth/contracts/audit_protocols.py`
- Test: `tests/unit/contracts/test_audit_protocols.py`

- [ ] **Step 1: Write structural conformance tests**

These tests verify that the adapter and repositories satisfy the protocols. Write them first — they define the protocol contract.

Note: `hasattr` is banned by CLAUDE.md. Use live instances with `isinstance` or direct method calls.

```python
# tests/unit/contracts/test_audit_protocols.py
"""Verify audit protocol conformance via live instances."""
from __future__ import annotations

from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory


class TestPluginAuditWriterConformance:
    """_PluginAuditWriterAdapter must satisfy PluginAuditWriter structurally."""

    def test_adapter_satisfies_protocol(self) -> None:
        """Adapter constructed by factory must pass isinstance check."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        writer = factory.plugin_audit_writer()
        # isinstance on Protocol checks method name presence (not signatures).
        # This is a coarse check — delegation tests below verify routing.
        assert isinstance(writer, PluginAuditWriter)


class TestPluginAuditWriterDelegation:
    """Verify each adapter method routes to the correct underlying repository."""

    def test_call_recording_routes_to_execution(self) -> None:
        """allocate_call_index, record_call should call ExecutionRepository."""
        from unittest.mock import MagicMock

        from elspeth.core.landscape.data_flow_repository import DataFlowRepository
        from elspeth.core.landscape.execution_repository import ExecutionRepository
        from elspeth.core.landscape.factory import _PluginAuditWriterAdapter
        from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository

        mock_exec = MagicMock(spec=ExecutionRepository)
        mock_df = MagicMock(spec=DataFlowRepository)
        mock_rl = MagicMock(spec=RunLifecycleRepository)
        mock_exec.allocate_call_index.return_value = 0

        adapter = _PluginAuditWriterAdapter(mock_exec, mock_df, mock_rl)
        adapter.allocate_call_index("state-1")
        mock_exec.allocate_call_index.assert_called_once_with("state-1")
        mock_df.allocate_call_index.assert_not_called()

    def test_error_recording_routes_to_data_flow(self) -> None:
        """record_validation_error should call DataFlowRepository."""
        from unittest.mock import MagicMock

        from elspeth.core.landscape.data_flow_repository import DataFlowRepository
        from elspeth.core.landscape.execution_repository import ExecutionRepository
        from elspeth.core.landscape.factory import _PluginAuditWriterAdapter
        from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository

        mock_exec = MagicMock(spec=ExecutionRepository)
        mock_df = MagicMock(spec=DataFlowRepository)
        mock_rl = MagicMock(spec=RunLifecycleRepository)
        mock_df.record_validation_error.return_value = "err-1"

        adapter = _PluginAuditWriterAdapter(mock_exec, mock_df, mock_rl)
        adapter.record_validation_error("run-1", "node-1", {}, "bad", "strict", "quarantine")
        mock_df.record_validation_error.assert_called_once()
        mock_exec.record_validation_error.assert_not_called()

    def test_get_node_state_routes_to_execution(self) -> None:
        """get_node_state should call ExecutionRepository."""
        from unittest.mock import MagicMock

        from elspeth.core.landscape.data_flow_repository import DataFlowRepository
        from elspeth.core.landscape.execution_repository import ExecutionRepository
        from elspeth.core.landscape.factory import _PluginAuditWriterAdapter
        from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository

        mock_exec = MagicMock(spec=ExecutionRepository)
        mock_df = MagicMock(spec=DataFlowRepository)
        mock_rl = MagicMock(spec=RunLifecycleRepository)
        mock_exec.get_node_state.return_value = None

        adapter = _PluginAuditWriterAdapter(mock_exec, mock_df, mock_rl)
        adapter.get_node_state("state-1")
        mock_exec.get_node_state.assert_called_once_with("state-1")

    def test_run_lifecycle_routes_to_run_lifecycle(self) -> None:
        """get_source_field_resolution, record_readiness_check → RunLifecycleRepository."""
        from unittest.mock import MagicMock

        from elspeth.core.landscape.data_flow_repository import DataFlowRepository
        from elspeth.core.landscape.execution_repository import ExecutionRepository
        from elspeth.core.landscape.factory import _PluginAuditWriterAdapter
        from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository

        mock_exec = MagicMock(spec=ExecutionRepository)
        mock_df = MagicMock(spec=DataFlowRepository)
        mock_rl = MagicMock(spec=RunLifecycleRepository)
        mock_rl.get_source_field_resolution.return_value = None

        adapter = _PluginAuditWriterAdapter(mock_exec, mock_df, mock_rl)
        adapter.get_source_field_resolution("run-1")
        mock_rl.get_source_field_resolution.assert_called_once_with("run-1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_audit_protocols.py -v`
Expected: ImportError — `audit_protocols` module doesn't exist yet.

- [ ] **Step 3: Create the protocols module**

```python
# src/elspeth/contracts/audit_protocols.py
"""Audit protocols for cross-layer boundary typing.

These protocols define what contracts/ code (L0) needs from core/landscape (L1)
repositories. They exist because L0 cannot import L1 directly at runtime.

Satisfied by _PluginAuditWriterAdapter (core/landscape/factory.py), which
composes ExecutionRepository + DataFlowRepository + RunLifecycleRepository.

Method-count budget: This protocol has ~17 methods across 3 repositories.
If a new caller needs a method not listed here, inject the repository
directly rather than growing this protocol. Do not exceed 20 methods.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from elspeth.contracts import (
    Call,
    CallStatus,
    CallType,
    NodeState,
    RoutingEvent,
    RoutingMode,
    RoutingReason,
    RoutingSpec,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.call_data import CallPayload
from elspeth.contracts.errors import ContractViolation, TransformErrorReason
from elspeth.contracts.schema_contract import SchemaContract

if TYPE_CHECKING:
    from elspeth.contracts.schema_contract import PipelineRow


class PluginAuditWriter(Protocol):
    """What PluginContext exposes for audit recording.

    Satisfied by _PluginAuditWriterAdapter which delegates to 3 repos:
    - ExecutionRepository: call recording, node state lookup, routing
    - DataFlowRepository: error recording, schema contracts
    - RunLifecycleRepository: field resolution, readiness checks
    """

    # ── Call recording (ExecutionRepository) ─────────────────────────
    def allocate_call_index(self, state_id: str) -> int: ...

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
    ) -> Call: ...

    # ── Operation call recording (ExecutionRepository) ───────────────
    def record_operation_call(
        self,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
    ) -> Call: ...

    # ── Node state lookup (ExecutionRepository) ──────────────────────
    def get_node_state(self, state_id: str) -> NodeState | None: ...

    # ── Routing recording (ExecutionRepository) ──────────────────────
    def record_routing_event(
        self,
        state_id: str,
        edge_id: str,
        mode: RoutingMode,
        reason: RoutingReason | None = None,
        *,
        event_id: str | None = None,
        routing_group_id: str | None = None,
        ordinal: int = 0,
        reason_ref: str | None = None,
    ) -> RoutingEvent: ...

    def record_routing_events(
        self,
        state_id: str,
        routes: list[RoutingSpec],
        reason: RoutingReason | None = None,
    ) -> list[RoutingEvent]: ...

    # ── Error recording (DataFlowRepository) ─────────────────────────
    def record_validation_error(
        self,
        run_id: str,
        node_id: str | None,
        row_data: Any,
        error: str,
        schema_mode: str,
        destination: str,
        *,
        contract_violation: ContractViolation | None = None,
    ) -> str: ...

    def record_transform_error(
        self,
        ref: TokenRef,
        transform_id: str,
        row_data: Mapping[str, object] | PipelineRow,
        error_details: TransformErrorReason,
        destination: str,
    ) -> str: ...

    # ── Schema contract updates (DataFlowRepository) ─────────────────
    def update_node_output_contract(
        self,
        run_id: str,
        node_id: str,
        contract: SchemaContract,
    ) -> None: ...

    def get_node_contracts(
        self,
        run_id: str,
        node_id: str,
        *,
        allow_missing: bool = False,
    ) -> tuple[SchemaContract | None, SchemaContract | None]: ...

    # ── Run lifecycle reads (RunLifecycleRepository) ─────────────────
    def get_source_field_resolution(self, run_id: str) -> dict[str, str] | None: ...

    def record_readiness_check(
        self,
        run_id: str,
        *,
        name: str,
        collection: str,
        reachable: bool,
        count: int | None,
        message: str,
    ) -> None: ...
```

**Design notes (post-review):**
- `PluginQueryReader` removed — return types are all L0 contracts types, so read-only consumers should accept `QueryRepository` directly (L2/L3 callers can import L1). No protocol needed for the read path.
- `runtime_checkable` removed from `PluginAuditWriter` — no production `isinstance` check exists. The conformance test uses mock-based delegation verification instead.
- `allocate_operation_call_index` removed — `PluginContext` never calls it via `ctx.landscape` (the operation path uses `record_operation_call` directly which handles indexing).
- `store_payload` NOT on this protocol — `web_scrape.py` should use `PayloadStore` directly (see Task 6).
- `Any` minimized — only `row_data` in `record_validation_error` remains `Any` (accepts heterogeneous external data). All other types are concrete L0 imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_audit_protocols.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/contracts/audit_protocols.py tests/unit/contracts/test_audit_protocols.py
git commit -m "feat: add PluginAuditWriter protocol for cross-layer boundary typing

Define PluginAuditWriter protocol in contracts/ so that PluginContext
and contexts.py can reference audit capabilities
without importing core/ (L1) types at runtime.

Part of recorder facade elimination (elspeth-50521f45ec)."
```

---

## Task 2: Create RecorderFactory

A factory that constructs repositories from `LandscapeDB` + `PayloadStore`, replacing the 11 `LandscapeRecorder(db)` construction sites.

**Files:**
- Create: `src/elspeth/core/landscape/factory.py`
- Create: `tests/unit/core/landscape/test_factory.py`

- [ ] **Step 1: Write factory tests**

```python
# tests/unit/core/landscape/test_factory.py
"""Tests for RecorderFactory construction."""
from __future__ import annotations

from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.factory import RecorderFactory, _PluginAuditWriterAdapter
from elspeth.core.landscape.query_repository import QueryRepository
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository


class TestRecorderFactory:
    def test_creates_all_four_repositories(self) -> None:
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        assert isinstance(factory.run_lifecycle, RunLifecycleRepository)
        assert isinstance(factory.execution, ExecutionRepository)
        assert isinstance(factory.data_flow, DataFlowRepository)
        assert isinstance(factory.query, QueryRepository)

    def test_repositories_share_database_ops(self) -> None:
        """All repos should share the same DatabaseOps instance."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        # Verify a round-trip: register a run via lifecycle, query it via query
        run = factory.run_lifecycle.begin_run(
            config={}, canonical_version="v1"
        )
        rows = factory.query.get_rows(run.run_id)
        assert rows == []  # No rows yet, but query didn't crash

    def test_payload_store_propagated(self) -> None:
        """PayloadStore should be passed to repos that need it."""
        from unittest.mock import MagicMock

        db = LandscapeDB.in_memory()
        mock_store = MagicMock()
        factory = RecorderFactory(db, payload_store=mock_store)
        assert factory.execution is not None
        assert factory.payload_store is mock_store

    def test_plugin_audit_writer_is_adapter(self) -> None:
        """Factory should produce a _PluginAuditWriterAdapter."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        writer = factory.plugin_audit_writer()
        assert isinstance(writer, _PluginAuditWriterAdapter)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_factory.py -v`
Expected: ImportError — `factory` module doesn't exist yet.

- [ ] **Step 3: Create the factory module**

```python
# src/elspeth/core/landscape/factory.py
"""RecorderFactory: constructs landscape repositories from LandscapeDB.

Replaces the LandscapeRecorder facade. Instead of one god-object that
delegates 93 methods, callers get the specific repositories they need.

Construction sites that previously called LandscapeRecorder(db) now call
RecorderFactory(db) and access the repository they need via properties.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from elspeth.contracts import (
    Call,
    CallStatus,
    CallType,
    NodeState,
    RoutingEvent,
    RoutingMode,
    RoutingReason,
    RoutingSpec,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.call_data import CallPayload
from elspeth.contracts.errors import ContractViolation, TransformErrorReason
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.model_loaders import (
    ArtifactLoader,
    BatchLoader,
    BatchMemberLoader,
    CallLoader,
    EdgeLoader,
    NodeLoader,
    NodeStateLoader,
    OperationLoader,
    RoutingEventLoader,
    RowLoader,
    RunLoader,
    TokenLoader,
    TokenOutcomeLoader,
    TokenParentLoader,
    TransformErrorLoader,
    ValidationErrorLoader,
)
from elspeth.core.landscape.query_repository import QueryRepository
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.contracts.schema_contract import PipelineRow


class _PluginAuditWriterAdapter:
    """Adapter composing 3 repositories into PluginAuditWriter.

    Exists because PluginContext (L0) needs a single audit writer
    reference, and the methods it needs span three domain repositories.

    Method-count budget: ~17 methods. Do not exceed 20.
    If a new caller needs a method not here, inject the repository
    directly rather than growing this adapter.
    """

    def __init__(
        self,
        execution: ExecutionRepository,
        data_flow: DataFlowRepository,
        run_lifecycle: RunLifecycleRepository,
    ) -> None:
        self._execution = execution
        self._data_flow = data_flow
        self._run_lifecycle = run_lifecycle

    # ── Delegated to ExecutionRepository ──────────────────────────────

    def allocate_call_index(self, state_id: str) -> int:
        return self._execution.allocate_call_index(state_id)

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
    ) -> Call:
        return self._execution.record_call(
            state_id, call_index, call_type, status,
            request_data, response_data, error, latency_ms,
            request_ref=request_ref, response_ref=response_ref,
        )

    def record_operation_call(
        self,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
    ) -> Call:
        return self._execution.record_operation_call(
            operation_id, call_type, status,
            request_data, response_data, error, latency_ms,
            request_ref=request_ref, response_ref=response_ref,
        )

    def get_node_state(self, state_id: str) -> NodeState | None:
        return self._execution.get_node_state(state_id)

    def record_routing_event(
        self,
        state_id: str,
        edge_id: str,
        mode: RoutingMode,
        reason: RoutingReason | None = None,
        *,
        event_id: str | None = None,
        routing_group_id: str | None = None,
        ordinal: int = 0,
        reason_ref: str | None = None,
    ) -> RoutingEvent:
        return self._execution.record_routing_event(
            state_id, edge_id, mode, reason,
            event_id=event_id, routing_group_id=routing_group_id,
            ordinal=ordinal, reason_ref=reason_ref,
        )

    def record_routing_events(
        self,
        state_id: str,
        routes: list[RoutingSpec],
        reason: RoutingReason | None = None,
    ) -> list[RoutingEvent]:
        return self._execution.record_routing_events(state_id, routes, reason)

    # ── Delegated to DataFlowRepository ──────────────────────────────

    def record_validation_error(
        self,
        run_id: str,
        node_id: str | None,
        row_data: Any,
        error: str,
        schema_mode: str,
        destination: str,
        *,
        contract_violation: ContractViolation | None = None,
    ) -> str:
        return self._data_flow.record_validation_error(
            run_id, node_id, row_data, error, schema_mode, destination,
            contract_violation=contract_violation,
        )

    def record_transform_error(
        self,
        ref: TokenRef,
        transform_id: str,
        row_data: Mapping[str, object] | PipelineRow,
        error_details: TransformErrorReason,
        destination: str,
    ) -> str:
        return self._data_flow.record_transform_error(
            ref, transform_id, row_data, error_details, destination,
        )

    def update_node_output_contract(
        self,
        run_id: str,
        node_id: str,
        contract: SchemaContract,
    ) -> None:
        self._data_flow.update_node_output_contract(run_id, node_id, contract)

    def get_node_contracts(
        self,
        run_id: str,
        node_id: str,
        *,
        allow_missing: bool = False,
    ) -> tuple[SchemaContract | None, SchemaContract | None]:
        return self._data_flow.get_node_contracts(
            run_id, node_id, allow_missing=allow_missing,
        )

    # ── Delegated to RunLifecycleRepository ──────────────────────────

    def get_source_field_resolution(self, run_id: str) -> dict[str, str] | None:
        return self._run_lifecycle.get_source_field_resolution(run_id)

    def record_readiness_check(
        self,
        run_id: str,
        *,
        name: str,
        collection: str,
        reachable: bool,
        count: int | None,
        message: str,
    ) -> None:
        self._run_lifecycle.record_readiness_check(
            run_id, name=name, collection=collection,
            reachable=reachable, count=count, message=message,
        )


class RecorderFactory:
    """Constructs landscape repositories from LandscapeDB.

    Replaces LandscapeRecorder as the single construction point.
    Instead of one 93-method facade, callers get specific repositories.

    Usage:
        factory = RecorderFactory(db, payload_store=store)
        run = factory.run_lifecycle.begin_run(config={}, ...)
        factory.data_flow.create_row(run.run_id, ...)
    """

    def __init__(
        self,
        db: LandscapeDB,
        *,
        payload_store: PayloadStore | None = None,
    ) -> None:
        ops = DatabaseOps(db)

        # Shared loader instances
        run_loader = RunLoader()
        node_loader = NodeLoader()
        edge_loader = EdgeLoader()
        row_loader = RowLoader()
        token_loader = TokenLoader()
        token_parent_loader = TokenParentLoader()
        call_loader = CallLoader()
        operation_loader = OperationLoader()
        routing_event_loader = RoutingEventLoader()
        batch_loader = BatchLoader()
        node_state_loader = NodeStateLoader()
        validation_error_loader = ValidationErrorLoader()
        transform_error_loader = TransformErrorLoader()
        token_outcome_loader = TokenOutcomeLoader()
        artifact_loader = ArtifactLoader()
        batch_member_loader = BatchMemberLoader()

        self._run_lifecycle = RunLifecycleRepository(db, ops, run_loader)

        self._execution = ExecutionRepository(
            db, ops,
            node_state_loader=node_state_loader,
            routing_event_loader=routing_event_loader,
            call_loader=call_loader,
            operation_loader=operation_loader,
            batch_loader=batch_loader,
            batch_member_loader=batch_member_loader,
            artifact_loader=artifact_loader,
            payload_store=payload_store,
        )

        self._data_flow = DataFlowRepository(
            db, ops,
            token_outcome_loader=token_outcome_loader,
            node_loader=node_loader,
            edge_loader=edge_loader,
            validation_error_loader=validation_error_loader,
            transform_error_loader=transform_error_loader,
            payload_store=payload_store,
        )

        self._query = QueryRepository(
            ops,
            row_loader=row_loader,
            token_loader=token_loader,
            token_parent_loader=token_parent_loader,
            node_state_loader=node_state_loader,
            routing_event_loader=routing_event_loader,
            call_loader=call_loader,
            token_outcome_loader=token_outcome_loader,
            payload_store=payload_store,
        )

        self._payload_store = payload_store

    @property
    def run_lifecycle(self) -> RunLifecycleRepository:
        return self._run_lifecycle

    @property
    def execution(self) -> ExecutionRepository:
        return self._execution

    @property
    def data_flow(self) -> DataFlowRepository:
        return self._data_flow

    @property
    def query(self) -> QueryRepository:
        return self._query

    @property
    def payload_store(self) -> PayloadStore | None:
        return self._payload_store

    def plugin_audit_writer(self) -> PluginAuditWriter:
        """Create a PluginAuditWriter for PluginContext injection.

        Returns an adapter composing ExecutionRepository,
        DataFlowRepository, and RunLifecycleRepository.
        """
        return _PluginAuditWriterAdapter(
            self._execution, self._data_flow, self._run_lifecycle,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_factory.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/core/landscape/factory.py tests/unit/core/landscape/test_factory.py
git commit -m "feat: add RecorderFactory for direct repository construction

Constructs all 4 landscape repositories from LandscapeDB + PayloadStore.
Includes _PluginAuditWriterAdapter for cross-layer PluginContext injection.

Part of recorder facade elimination (elspeth-50521f45ec)."
```

---

## Task 3: Update PluginContext and context protocols

Replace `LandscapeRecorder` references with `PluginAuditWriter` in the contracts layer.

**Files:**
- Modify: `src/elspeth/contracts/contexts.py`
- Modify: `src/elspeth/contracts/plugin_context.py`
- Modify: `src/elspeth/plugins/transforms/llm/transform.py` (type annotation: `self._recorder`)
- Modify: `src/elspeth/plugins/transforms/llm/openrouter_batch.py` (type annotation: `self._recorder`)
- Modify: `src/elspeth/plugins/transforms/llm/providers/azure.py` (type annotation: `recorder` param)
- Modify: `src/elspeth/plugins/transforms/llm/providers/openrouter.py` (type annotation: `recorder` param)
- Modify: `src/elspeth/plugins/transforms/azure/base.py` (type annotation: `self._recorder`)
- Modify: `src/elspeth/plugins/infrastructure/display_headers.py` (type annotation)
- Modify: `src/elspeth/plugins/transforms/rag/transform.py` (type annotation)
- Modify: `tests/unit/contracts/test_context_protocols.py`
- Modify: `tests/unit/contracts/test_plugin_context_recording.py`
- Modify: `tests/unit/plugins/test_context.py`
- Modify: `tests/unit/plugins/test_context_types.py`

- [ ] **Step 1: Update contexts.py protocols**

In `src/elspeth/contracts/contexts.py`, replace the `LandscapeRecorder` import and property type:

Replace:
```python
if TYPE_CHECKING:
    from elspeth.contracts import Call, CallStatus, CallType
    from elspeth.contracts.batch_checkpoint import BatchCheckpointState
    from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig
    from elspeth.contracts.identity import TokenInfo
    from elspeth.contracts.plugin_context import ValidationErrorToken
    from elspeth.contracts.schema_contract import SchemaContract
    from elspeth.core.landscape.recorder import LandscapeRecorder
    from elspeth.core.rate_limit import RateLimitRegistry
```

With:
```python
if TYPE_CHECKING:
    from elspeth.contracts import Call, CallStatus, CallType
    from elspeth.contracts.batch_checkpoint import BatchCheckpointState
    from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig
    from elspeth.contracts.identity import TokenInfo
    from elspeth.contracts.plugin_context import ValidationErrorToken
    from elspeth.contracts.schema_contract import SchemaContract
    from elspeth.core.rate_limit import RateLimitRegistry

from elspeth.contracts.audit_protocols import PluginAuditWriter
```

Then replace all three protocol property signatures:
```python
    # In SourceContext:
    @property
    def landscape(self) -> PluginAuditWriter | None: ...

    # In TransformContext:
    @property
    def landscape(self) -> PluginAuditWriter | None: ...

    # In SinkContext:
    @property
    def landscape(self) -> PluginAuditWriter | None: ...
```

Note: `PluginAuditWriter` is imported at runtime (not TYPE_CHECKING) because it's a `runtime_checkable` Protocol used in isinstance checks. This is a proper L0→L0 import — no layer violation.

- [ ] **Step 2: Update plugin_context.py**

In `src/elspeth/contracts/plugin_context.py`, find the `landscape` field and change its type from `LandscapeRecorder | None` to `PluginAuditWriter | None`. Remove the `LandscapeRecorder` import. Add the `PluginAuditWriter` import.

Find the TYPE_CHECKING block and remove the `from elspeth.core.landscape.recorder import LandscapeRecorder` line. Add at the top-level imports:

```python
from elspeth.contracts.audit_protocols import PluginAuditWriter
```

Change the field:
```python
    landscape: PluginAuditWriter | None
```

Also update any internal methods that reference `LandscapeRecorder` in type hints — change them to `PluginAuditWriter`.

- [ ] **Step 2b: Update transform plugin type annotations**

Five transform files store `ctx.landscape` in a `self._recorder` field typed as `LandscapeRecorder | None`. Change the type annotation to `PluginAuditWriter | None` and replace the TYPE_CHECKING import of `LandscapeRecorder` with a runtime import of `PluginAuditWriter`.

In each file, replace:
```python
if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder
```
With:
```python
from elspeth.contracts.audit_protocols import PluginAuditWriter
```

And replace all `LandscapeRecorder` type annotations with `PluginAuditWriter`.

Files:
- `src/elspeth/plugins/transforms/llm/transform.py:59,1138`
- `src/elspeth/plugins/transforms/llm/openrouter_batch.py:56,239`
- `src/elspeth/plugins/transforms/llm/providers/azure.py:27,102`
- `src/elspeth/plugins/transforms/llm/providers/openrouter.py:39,104`
- `src/elspeth/plugins/transforms/azure/base.py:39,133`
- `src/elspeth/plugins/infrastructure/display_headers.py:192` (calls `get_source_field_resolution`)
- `src/elspeth/plugins/transforms/rag/transform.py:141` (calls `record_readiness_check`)

- [ ] **Step 3: Run existing context protocol tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_context_protocols.py tests/unit/plugins/test_context.py tests/unit/plugins/test_context_types.py tests/unit/contracts/test_plugin_context_recording.py -v`

These tests verify PluginContext satisfies the context protocols. They will need updates if they construct a `LandscapeRecorder` to pass as the `landscape` field — change those to use `RecorderFactory(db).plugin_audit_writer()`.

- [ ] **Step 4: Fix any test failures**

Tests that construct `PluginContext(landscape=LandscapeRecorder(db))` should change to `PluginContext(landscape=RecorderFactory(db).plugin_audit_writer())`.

Look for patterns like:
```python
recorder = LandscapeRecorder(db)
ctx = PluginContext(landscape=recorder, ...)
```
Replace with:
```python
factory = RecorderFactory(db)
ctx = PluginContext(landscape=factory.plugin_audit_writer(), ...)
```

**Critical:** Also update `tests/fixtures/factories.py` — `make_context()` creates `Mock(spec=LandscapeRecorder)` and configures `get_node_state` on it. Change to `Mock(spec=_PluginAuditWriterAdapter)`:
```python
from elspeth.core.landscape.factory import _PluginAuditWriterAdapter
landscape = Mock(spec=_PluginAuditWriterAdapter)
node_state_mock = Mock()
node_state_mock.token_id = token.token_id
landscape.get_node_state.return_value = node_state_mock
```
Note: `get_node_state` IS on `_PluginAuditWriterAdapter` (added per review). The mock will correctly reflect the adapter's interface.

- [ ] **Step 5: Run full contracts test suite**

Run: `.venv/bin/python -m pytest tests/unit/contracts/ -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/contexts.py src/elspeth/contracts/plugin_context.py tests/unit/contracts/ tests/unit/plugins/
git commit -m "refactor: replace LandscapeRecorder with PluginAuditWriter in contracts

PluginContext and context protocols now use PluginAuditWriter (L0)
instead of LandscapeRecorder (L1), eliminating a TYPE_CHECKING cross-layer
import.

Part of recorder facade elimination (elspeth-50521f45ec)."
```

---

## Task 4: Update engine executors

Change executors from accepting `LandscapeRecorder` to accepting specific repositories. This is the largest batch of mechanical changes.

**Files:**
- Modify: `src/elspeth/engine/executors/state_guard.py`
- Modify: `src/elspeth/engine/executors/gate.py`
- Modify: `src/elspeth/engine/executors/transform.py`
- Modify: `src/elspeth/engine/executors/sink.py`
- Modify: `src/elspeth/engine/executors/aggregation.py`
- Modify: `src/elspeth/engine/coalesce_executor.py`
- Modify: `src/elspeth/engine/tokens.py`
- Modify: associated test files

- [ ] **Step 1: Update NodeStateGuard**

In `src/elspeth/engine/executors/state_guard.py`:

Replace `LandscapeRecorder` import with `ExecutionRepository`:
```python
from elspeth.core.landscape.execution_repository import ExecutionRepository
```

Change constructor:
```python
class NodeStateGuard:
    def __init__(
        self,
        execution: ExecutionRepository,
        ...
    ) -> None:
        self._execution = execution
        ...
```

Replace all `self._recorder.begin_node_state(...)` → `self._execution.begin_node_state(...)`.
Replace all `self._recorder.complete_node_state(...)` → `self._execution.complete_node_state(...)`.

- [ ] **Step 2: Update GateExecutor**

In `src/elspeth/engine/executors/gate.py`:

Replace import, change constructor parameter from `recorder: LandscapeRecorder` to `execution: ExecutionRepository`.

Replace `self._recorder.record_routing_event(...)` → `self._execution.record_routing_event(...)`.
Replace `self._recorder.record_routing_events(...)` → `self._execution.record_routing_events(...)`.

- [ ] **Step 3: Update TransformExecutor**

In `src/elspeth/engine/executors/transform.py`:

Replace import, change constructor. TransformExecutor passes its recorder to NodeStateGuard — after Step 1, it should pass `ExecutionRepository` instead.

Change `recorder: LandscapeRecorder` → `execution: ExecutionRepository` in constructor. Update the NodeStateGuard construction inside.

- [ ] **Step 4: Update SinkExecutor**

In `src/elspeth/engine/executors/sink.py`:

SinkExecutor needs both `ExecutionRepository` (for node states, routing, artifacts) and `DataFlowRepository` (for `record_token_outcome`).

Change constructor:
```python
def __init__(
    self,
    execution: ExecutionRepository,
    data_flow: DataFlowRepository,
    span_factory: ...,
    run_id: str,
) -> None:
    self._execution = execution
    self._data_flow = data_flow
```

Replace:
- `self._recorder.begin_node_state(...)` → `self._execution.begin_node_state(...)`
- `self._recorder.complete_node_state(...)` → `self._execution.complete_node_state(...)`
- `self._recorder.record_routing_event(...)` → `self._execution.record_routing_event(...)`
- `self._recorder.register_artifact(...)` → `self._execution.register_artifact(...)`
- `self._recorder.record_token_outcome(...)` → `self._data_flow.record_token_outcome(...)`

- [ ] **Step 5: Update AggregationExecutor**

In `src/elspeth/engine/executors/aggregation.py`:

Change to accept `ExecutionRepository`. All methods called (`create_batch`, `add_batch_member`, `update_batch_status`, `begin_node_state`, `complete_node_state`) are on ExecutionRepository.

- [ ] **Step 6: Update CoalesceExecutor**

In `src/elspeth/engine/coalesce_executor.py`:

Change to accept `ExecutionRepository`. Methods: `begin_node_state`, `complete_node_state`, `get_completed_row_ids_for_nodes` — all on ExecutionRepository.

- [ ] **Step 7: Update TokenManager**

In `src/elspeth/engine/tokens.py`:

Change to accept `DataFlowRepository`. Methods: `create_row`, `create_token`, `fork_token`, `coalesce_tokens`, `expand_token` — all on DataFlowRepository.

```python
from elspeth.core.landscape.data_flow_repository import DataFlowRepository

class TokenManager:
    def __init__(
        self,
        data_flow: DataFlowRepository,
        *,
        step_resolver: StepResolver,
    ) -> None:
        self._data_flow = data_flow
        ...
```

Replace all `self._recorder.xxx(...)` → `self._data_flow.xxx(...)`.

- [ ] **Step 8: Run executor tests**

Run: `.venv/bin/python -m pytest tests/unit/engine/ -v`

Tests that construct executors with `LandscapeRecorder` will fail. Fix them by either:
1. Constructing a `RecorderFactory` and passing the appropriate repository, or
2. Constructing the repository directly for unit tests that mock it.

For tests that mock the recorder, change the mock target:
```python
# Before:
mock_recorder = MagicMock(spec=LandscapeRecorder)
guard = NodeStateGuard(mock_recorder, ...)

# After:
mock_execution = MagicMock(spec=ExecutionRepository)
guard = NodeStateGuard(mock_execution, ...)
```

- [ ] **Step 9: Run full engine test suite**

Run: `.venv/bin/python -m pytest tests/unit/engine/ tests/integration/pipeline/ -v`
Expected: All PASS.

- [ ] **Step 10: Commit**

```bash
git add src/elspeth/engine/ tests/unit/engine/
git commit -m "refactor: executors and TokenManager accept specific repositories

- NodeStateGuard, GateExecutor, TransformExecutor, AggregationExecutor,
  CoalesceExecutor → ExecutionRepository
- SinkExecutor → ExecutionRepository + DataFlowRepository
- TokenManager → DataFlowRepository

Part of recorder facade elimination (elspeth-50521f45ec)."
```

---

## Task 5: Update RowProcessor and orchestrator

The orchestrator is the construction root — it creates the recorder and passes it down. After this task, it creates repositories via `RecorderFactory` and passes them to executors/processor.

**Files:**
- Modify: `src/elspeth/engine/processor.py`
- Modify: `src/elspeth/engine/orchestrator/core.py`
- Modify: `src/elspeth/engine/orchestrator/export.py`
- Modify: `src/elspeth/engine/orchestrator/aggregation.py`
- Modify: `src/elspeth/engine/orchestrator/types.py`
- Modify: `src/elspeth/core/operations.py`

- [ ] **Step 1: Update RowProcessor**

In `src/elspeth/engine/processor.py`:

RowProcessor needs `ExecutionRepository` (for passing to executors it constructs), `DataFlowRepository` (for `record_token_outcome`, `record_transform_error`, `get_node_contracts`, `update_node_output_contract`), and it constructs TokenManager (which needs `DataFlowRepository`).

Change constructor:
```python
def __init__(
    self,
    execution: ExecutionRepository,
    data_flow: DataFlowRepository,
    span_factory: ...,
    run_id: str,
    ...
) -> None:
    self._execution = execution
    self._data_flow = data_flow
```

Update internal executor construction to pass `self._execution` (and `self._data_flow` for SinkExecutor). Update direct method calls.

- [ ] **Step 2: Update ResumeState**

In `src/elspeth/engine/orchestrator/types.py`:

`ResumeState` is `@dataclass(frozen=True)` and holds a `recorder: LandscapeRecorder` field. Change it to `factory: RecorderFactory`. Note: `RecorderFactory` is a mutable object, but `frozen=True` only prevents reassignment — `freeze_fields` in `__post_init__` only freezes `restored_aggregation_state`, not all fields. No freeze guard change needed.

```python
from elspeth.core.landscape.factory import RecorderFactory

@dataclass
class ResumeState:
    factory: RecorderFactory
    ...
```

**Critical:** After renaming this field, grep for ALL consumers in `core.py`:
```bash
grep -n "state\.recorder\|resume_state\.recorder" src/elspeth/engine/orchestrator/core.py
```
Known sites (must all change to `state.factory` or destructure to specific repos):
- Line 2742: `recorder = state.recorder` — change to factory destructuring
- Line 2750: `recorder.finalize_run(...)` — change to `factory.run_lifecycle.finalize_run(...)`
- Line 2806: `recorder=recorder` passed to `_process_resumed_rows` — pass factory
- Line 2824: `recorder.finalize_run(...)` — change to `factory.run_lifecycle.finalize_run(...)`
- Line 2862: `recorder` passed to `_emit_interrupted_ceremony` — pass factory
- Line 2870: `recorder` passed to `_emit_failed_ceremony` — pass factory
- Line 2879: `_process_resumed_rows` signature — change parameter type

- [ ] **Step 3: Update operations.py**

In `src/elspeth/core/operations.py`:

Change to accept `ExecutionRepository`. Methods used: `begin_operation`, `complete_operation`, `allocate_operation_call_index`, `record_operation_call` — all on ExecutionRepository.

- [ ] **Step 4: Update orchestrator/core.py**

This is the largest single change. In `src/elspeth/engine/orchestrator/core.py`:

Replace `LandscapeRecorder(db, payload_store=...)` with `RecorderFactory(db, payload_store=...)`.

In `_initialize_database_phase`:
```python
factory = RecorderFactory(self._db, payload_store=payload_store)
run = factory.run_lifecycle.begin_run(
    config=config.config,
    canonical_version=self._canonical_version,
    ...
)
if secret_resolutions:
    factory.run_lifecycle.record_secret_resolutions(
        run_id=run.run_id,
        resolutions=secret_resolutions,
    )
return factory, run
```

Change the return type from `tuple[LandscapeRecorder, Run]` to `tuple[RecorderFactory, Run]`.

In `_build_processor`, change `recorder: LandscapeRecorder` to `factory: RecorderFactory`:
```python
token_manager = TokenManager(factory.data_flow, step_resolver=step_resolver)
coalesce_executor = CoalesceExecutor(
    execution=factory.execution,
    ...
)
processor = RowProcessor(
    execution=factory.execution,
    data_flow=factory.data_flow,
    ...
)
```

Update all other methods that pass `recorder` through:
- `_register_nodes` → use `factory.data_flow.register_node(...)`
- `_register_edges` → use `factory.data_flow.register_edge(...)`
- `_build_plugin_context` → use `factory.plugin_audit_writer()` for the `landscape` field
- `_execute_source_load` → use `factory.execution.begin_node_state(...)` etc.
- `finalize_run` → use `factory.run_lifecycle.finalize_run(...)`
- `_execute_export_phase` → use `factory.run_lifecycle.set_export_status(...)`

Systematically grep for `recorder.` in the file and replace each call with the appropriate `factory.<repo>.method(...)`.

- [ ] **Step 5: Update orchestrator/export.py**

Replace `LandscapeRecorder(db)` with `RecorderFactory(db)`, access specific repos for queries.

- [ ] **Step 6: Update orchestrator/aggregation.py**

Change `handle_incomplete_batches` parameter from `LandscapeRecorder` to `ExecutionRepository`. Methods used: `get_incomplete_batches`, `update_batch_status`, `retry_batch` — all ExecutionRepository.

- [ ] **Step 7: Run orchestrator tests**

Run: `.venv/bin/python -m pytest tests/unit/engine/orchestrator/ tests/integration/pipeline/orchestrator/ -v`

Fix any test failures by updating recorder construction to use `RecorderFactory`.

- [ ] **Step 8: Run full pipeline integration tests**

Run: `.venv/bin/python -m pytest tests/integration/pipeline/ -v`
Expected: All PASS.

- [ ] **Step 8b: Run mypy (early check — catches method misattribution)**

Run: `.venv/bin/python -m mypy src/elspeth/engine/ src/elspeth/core/operations.py`

This catches the most common error in this refactor: calling a DataFlowRepository method on an ExecutionRepository reference (or vice versa). Don't defer to Task 9.

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/engine/ src/elspeth/core/operations.py tests/
git commit -m "refactor: orchestrator uses RecorderFactory, passes specific repos

Orchestrator constructs RecorderFactory instead of LandscapeRecorder.
RowProcessor accepts ExecutionRepository + DataFlowRepository.
operations.py accepts ExecutionRepository.

Part of recorder facade elimination (elspeth-50521f45ec)."
```

---

## Task 6: Update plugin clients

Change audited client base and subclasses to accept `ExecutionRepository`.

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/clients/base.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/http.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/llm.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/replayer.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/verifier.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py`
- Modify: `src/elspeth/plugins/transforms/web_scrape.py`

- [ ] **Step 1: Update AuditedClientBase**

In `src/elspeth/plugins/infrastructure/clients/base.py`:

Replace `LandscapeRecorder` import with `ExecutionRepository`. Change constructor parameter. Update `self._recorder` → `self._execution` throughout.

- [ ] **Step 2: Update HTTP, LLM, Replayer, Verifier clients**

In each file, the recorder reference comes from the base class. If they reference `self._recorder` directly, update to `self._execution`. If they only inherit and use base class methods, they may need no changes beyond the import path.

Also update the retrieval clients (`chroma.py`, `azure_search.py`) — these accept `recorder: LandscapeRecorder` in their constructors and pass it to `AuditedClientBase`. Change to `execution: ExecutionRepository`.

- [ ] **Step 3: Update web_scrape.py**

`web_scrape.py` calls `recorder.store_payload()`, which is the one method on the facade that doesn't delegate to any repository — it goes directly to `PayloadStore`.

`store_payload` does NOT belong on `PluginAuditWriter` — it's payload persistence, not audit recording (review finding). Route through `PayloadStore` directly instead.

`web_scrape.py` gets its recorder via `ctx.landscape` (PluginContext). Since `PluginContext` does not currently have a `payload_store` field, add one:

1. In `src/elspeth/contracts/plugin_context.py`, add a field:
```python
from elspeth.contracts.payload_store import PayloadStore
...
payload_store: PayloadStore | None = None
```

2. In `web_scrape.py`, replace:
```python
self._recorder.store_payload(content.encode(), purpose="processed_content")
```
With:
```python
if ctx.payload_store is None:
    from elspeth.contracts.errors import FrameworkBugError
    raise FrameworkBugError(
        "store_payload called but PluginContext has no payload_store. "
        "Orchestrator must configure payload_store for web-scrape transforms."
    )
ctx.payload_store.store(content.encode())
```

3. In orchestrator `_build_plugin_context`, pass `payload_store=factory.payload_store`.

4. Migrate `tests/unit/core/landscape/test_recorder_store_payload.py` to test the new path via `PluginContext.payload_store` instead of `LandscapeRecorder.store_payload()`. The `purpose` parameter was already call-site documentation only (never persisted) — drop it.

- [ ] **Step 4: Run plugin client tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/clients/ tests/unit/plugins/transforms/ -v`

Fix test failures by updating mock/recorder construction.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/ src/elspeth/contracts/audit_protocols.py src/elspeth/core/landscape/factory.py tests/unit/plugins/
git commit -m "refactor: plugin clients accept ExecutionRepository directly

AuditedClientBase and subclasses inject ExecutionRepository.
web_scrape.py store_payload routed through PluginContext.payload_store.

Part of recorder facade elimination (elspeth-50521f45ec)."
```

---

## Task 7: Update read-only consumers

MCP, TUI, exporter, lineage, CLI, and checkpoint recovery all construct `LandscapeRecorder` locally. Replace with `RecorderFactory`.

**Files:**
- Modify: `src/elspeth/mcp/analyzer.py`
- Modify: `src/elspeth/tui/screens/explain_screen.py`
- Modify: `src/elspeth/core/landscape/exporter.py`
- Modify: `src/elspeth/core/landscape/lineage.py`
- Modify: `src/elspeth/core/checkpoint/recovery.py`
- Modify: `src/elspeth/cli.py`
- Modify: `src/elspeth/cli_helpers.py`

- [ ] **Step 1: Update lineage.py**

In `src/elspeth/core/landscape/lineage.py`:

The `explain()` function takes a `LandscapeRecorder`. Change it to accept `QueryRepository` + `DataFlowRepository` (it calls methods on both).

```python
def explain(
    query: QueryRepository,
    data_flow: DataFlowRepository,
    run_id: str,
    row_id: str,
    ...
) -> LineageResult:
```

Replace `recorder.get_token(...)` → `query.get_token(...)`, `recorder.get_node_states_for_token(...)` → `query.get_node_states_for_token(...)`, etc.

For DataFlowRepository calls: `recorder.get_token_outcomes_for_row(...)` → `data_flow.get_token_outcomes_for_row(...)`, `recorder.get_validation_errors_for_row(...)` → `data_flow.get_validation_errors_for_row(...)`, etc.

- [ ] **Step 2: Update exporter.py**

Replace `LandscapeRecorder(db)` with `RecorderFactory(db)`. Access repos via factory properties.

```python
factory = RecorderFactory(db)
run = factory.run_lifecycle.get_run(run_id)
nodes = factory.data_flow.get_nodes(run_id)
...
```

- [ ] **Step 3: Update MCP analyzer.py**

Replace `self._recorder = LandscapeRecorder(self._db)` with `self._factory = RecorderFactory(self._db)`. Pass specific repositories to analyzer submodules.

- [ ] **Step 4: Update TUI explain_screen.py**

Replace `LandscapeRecorder(db)` constructions with `RecorderFactory(db)` and access appropriate repos.

- [ ] **Step 5: Update CLI (cli.py, cli_helpers.py)**

Replace `recorder = LandscapeRecorder(db)` with `factory = RecorderFactory(db)` and access `factory.run_lifecycle`, `factory.data_flow`, etc. as needed.

- [ ] **Step 6: Update checkpoint/recovery.py**

Replace `recorder = LandscapeRecorder(self._db)` with `factory = RecorderFactory(self._db)`. Access `factory.run_lifecycle.get_run_contract(...)` etc.

- [ ] **Step 7: Run consumer tests**

Run: `.venv/bin/python -m pytest tests/unit/mcp/ tests/unit/cli/ tests/unit/core/landscape/test_query_methods.py tests/e2e/ -v`

Fix failures by updating recorder construction in tests.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/mcp/ src/elspeth/tui/ src/elspeth/core/landscape/exporter.py src/elspeth/core/landscape/lineage.py src/elspeth/core/checkpoint/ src/elspeth/cli.py src/elspeth/cli_helpers.py tests/
git commit -m "refactor: read-only consumers use RecorderFactory

MCP, TUI, exporter, lineage, CLI, and recovery construct repositories
via RecorderFactory instead of LandscapeRecorder.

Part of recorder facade elimination (elspeth-50521f45ec)."
```

---

## Task 8: Update test fixtures

The central test fixture `make_recorder()` and `RecorderSetup` need to use `RecorderFactory`. This affects ~100 test files.

**Files:**
- Modify: `tests/fixtures/landscape.py`
- Modify: `tests/fixtures/factories.py`
- Modify: `tests/fixtures/multi_run.py`
- Modify: Various test files that construct `LandscapeRecorder` directly

- [ ] **Step 1: Update landscape.py fixtures**

```python
# tests/fixtures/landscape.py
from elspeth.core.landscape.factory import RecorderFactory

def make_factory(db: LandscapeDB | None = None, *, payload_store=None) -> RecorderFactory:
    """Factory for RecorderFactory (replaces make_recorder)."""
    if db is None:
        db = make_landscape_db()
    return RecorderFactory(db, payload_store=payload_store)


@dataclass
class RecorderSetup:
    """Result from make_recorder_with_run()."""
    db: LandscapeDB
    factory: RecorderFactory
    run_id: str
    source_node_id: str

    # Convenience properties for common access patterns
    @property
    def run_lifecycle(self) -> RunLifecycleRepository:
        return self.factory.run_lifecycle

    @property
    def execution(self) -> ExecutionRepository:
        return self.factory.execution

    @property
    def data_flow(self) -> DataFlowRepository:
        return self.factory.data_flow

    @property
    def query(self) -> QueryRepository:
        return self.factory.query
```

Update `make_recorder_with_run()` to use `RecorderFactory`:
```python
def make_recorder_with_run(...) -> RecorderSetup:
    db = make_landscape_db()
    factory = make_factory(db)
    run = factory.run_lifecycle.begin_run(**begin_kwargs)
    node = factory.data_flow.register_node(**register_kwargs)
    return RecorderSetup(db=db, factory=factory, run_id=run.run_id, ...)
```

Update `register_test_node()` to accept `DataFlowRepository`:
```python
def register_test_node(
    data_flow: DataFlowRepository,
    run_id: str,
    node_id: str,
    ...
) -> str:
    node = data_flow.register_node(...)
    return node.node_id
```

**No deprecated alias.** The no-legacy-code policy forbids compatibility shims. Bulk-rename `make_recorder` → `make_factory` and `setup.recorder` → `setup.factory` (or `setup.data_flow`, `setup.execution`, etc.) across all ~100 test files in one pass.

- [ ] **Step 2: Bulk-rename across all test files**

```bash
# Find all files referencing make_recorder or setup.recorder
grep -rl "make_recorder\|setup\.recorder\|\.recorder\." tests/ --include="*.py" | sort -u
```

For each file:
- `make_recorder(` → `make_factory(`
- `setup.recorder.` → route to the appropriate repo property (`setup.data_flow.`, `setup.execution.`, `setup.run_lifecycle.`, `setup.query.`)
- `register_test_node(recorder,` → `register_test_node(setup.data_flow,` (or `factory.data_flow`)

This is mechanical. Many test files use `setup.recorder.begin_node_state(...)` which maps to `setup.execution.begin_node_state(...)`, or `setup.recorder.create_row(...)` which maps to `setup.data_flow.create_row(...)`. Refer to the Caller → Repository Mapping table to determine which repo each method belongs to.

- [ ] **Step 3: Update pytest fixtures**

```python
@pytest.fixture
def landscape_factory(landscape_db: LandscapeDB) -> RecorderFactory:
    """Function-scoped RecorderFactory."""
    return RecorderFactory(landscape_db)
```

No `recorder` fixture alias — delete the old one.

- [ ] **Step 4: Update tests/fixtures/factories.py and multi_run.py**

Replace `LandscapeRecorder` construction with `RecorderFactory`. Update `make_context()` mock spec (already covered in Task 3 Step 4).

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -x --timeout=120`

With `-x`, the first failure stops the run. Fix and re-run until green.

- [ ] **Step 6: Verify zero LandscapeRecorder references remain**

```bash
grep -r "LandscapeRecorder\|make_recorder\|setup\.recorder" tests/ --include="*.py"
```
Expected: zero results.

- [ ] **Step 7: Commit**

```bash
git add tests/
git commit -m "refactor: test fixtures use RecorderFactory — bulk rename

No deprecated aliases. All ~100 test files updated to use make_factory(),
setup.data_flow, setup.execution, etc. directly.

Part of recorder facade elimination (elspeth-50521f45ec)."
```

---

## Task 9: Delete LandscapeRecorder and clean up

Remove the facade and all remaining references.

**Files:**
- Delete: `src/elspeth/core/landscape/recorder.py`
- Modify: `src/elspeth/core/landscape/__init__.py`
- Modify: `tests/fixtures/landscape.py` (remove deprecated aliases)

- [ ] **Step 1: Grep for any remaining LandscapeRecorder references**

Run: `grep -r "LandscapeRecorder" src/ tests/ --include="*.py" -l`

Every file in the output must be updated. If any remain from previous tasks, fix them now.

- [ ] **Step 2: Delete recorder.py**

```bash
rm src/elspeth/core/landscape/recorder.py
```

- [ ] **Step 3: Update __init__.py**

In `src/elspeth/core/landscape/__init__.py`:

Remove:
```python
from elspeth.core.landscape.recorder import LandscapeRecorder
```

Remove `"LandscapeRecorder"` from `__all__`.

Add:
```python
from elspeth.core.landscape.factory import RecorderFactory
```

Add `"RecorderFactory"` to `__all__`.

Review the 4 repository exports (`DataFlowRepository`, `ExecutionRepository`, `QueryRepository`, `RunLifecycleRepository`) — these should STAY in `__all__` since they're now the public API.

- [ ] **Step 4: Migrate store_payload tests**

`tests/unit/core/landscape/test_recorder_store_payload.py` tests `LandscapeRecorder.store_payload()`. Rewrite to test the `PayloadStore` path via `PluginContext.payload_store` (the new routing from Task 6). Key test cases to preserve: stores content, raises without store, empty bytes.

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ --timeout=120`
Expected: All PASS with zero references to `LandscapeRecorder`.

- [ ] **Step 6: Run type checker**

Run: `.venv/bin/python -m mypy src/elspeth/`
Expected: No new type errors related to recorder/repository types.

- [ ] **Step 7: Run linter**

Run: `.venv/bin/python -m ruff check src/elspeth/`
Expected: No new lint errors.

- [ ] **Step 8: Run tier model enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`

This verifies no new layer violations were introduced. The `PluginAuditWriter` import in `contracts/` is L0→L0 (clean). Repository imports in engine/ are L2→L1 (clean).

- [ ] **Step 9: Commit**

```bash
git rm src/elspeth/core/landscape/recorder.py
git add src/elspeth/core/landscape/__init__.py tests/fixtures/landscape.py
git commit -m "refactor: delete LandscapeRecorder facade

93-method pure-delegation facade removed. All callers now inject
specific repositories directly. RecorderFactory handles construction.

Closes elspeth-50521f45ec."
```

---

## Risk Mitigation Notes

**Blast radius:** ~45 source files + ~100 test files. Every change is mechanical (type annotation swap + attribute rename). No logic changes. The risk is typos and missed call sites, not design errors.

**Incremental safety:** Each task produces a working codebase. Tasks 1-2 are additive (new files only). Tasks 3-8 each touch a different caller group. Task 9 is the only destructive step (file deletion) and should only happen after all tests pass.

**Rollback:** If any task breaks unexpectedly, `git revert` the single commit. No task depends on another task's commits being squashed.

**Test strategy:** Existing test suite for behavioral verification. New delegation routing tests (Task 1) verify adapter wiring correctness. `store_payload` tests migrated to new path (Task 9 Step 4). mypy runs after Tasks 5 and 9 to catch method misattribution.

**Duck-typed callers:** `grep` for `LandscapeRecorder` will NOT catch files that call methods via `ctx.landscape` without importing the type (e.g., `display_headers.py`, `rag/transform.py`). The protocol and adapter methods were derived by grepping for `ctx.landscape.` and `self.landscape.` across the entire codebase, not just import statements.

**`store_payload` migration:** Routed through `PluginContext.payload_store` directly (not the adapter). `PayloadStore` protocol is already in L0. `purpose` parameter was call-site documentation only (never persisted) — confirmed by reading `recorder.py:494-521`.

**Adapter method-count budget:** `_PluginAuditWriterAdapter` has ~17 methods across 3 repositories. A code comment enforces a ceiling of 20. If future callers need more methods via `ctx.landscape`, inject the repository directly — do not grow the adapter toward facade territory.

**No deprecated aliases.** All renames (fixture functions, fixture names, field names) are done as bulk changes in a single commit per task. The no-legacy-code policy is respected throughout.
