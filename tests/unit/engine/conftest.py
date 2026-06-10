# tests/unit/engine/conftest.py
"""Engine unit test fixtures."""

from typing import ClassVar

from pydantic import ConfigDict

from elspeth.contracts import PluginSchema
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.types import NodeID, StepResolver
from elspeth.core.config import CoalesceSettings
from elspeth.engine.coalesce_executor import CoalesceExecutor


class MockCoalesceExecutor(CoalesceExecutor):
    """Test wrapper that auto-provides output_schema for union merge.

    Production code computes output_schema via the DAG builder's merge_union_fields().
    Tests bypass the DAG builder, so this wrapper provides an OBSERVED-mode schema
    by default, matching the contract mode used by test fixtures.

    An OBSERVED output_schema routes _execute_merge() through the runtime
    merge_union_contracts() path (the all-OBSERVED union path), which shares
    its core algorithm with merge_union_fields().
    """

    def register_coalesce(
        self,
        settings: CoalesceSettings,
        node_id: NodeID,
        branch_schemas: dict[str, tuple[str, ...]] | None = None,
        output_schema: SchemaContract | None = None,
    ) -> None:
        if settings.on_success is None:
            settings = settings.model_copy(update={"on_success": "default"})
        if output_schema is None and settings.merge == "union":
            output_schema = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        super().register_coalesce(settings, node_id, branch_schemas, output_schema)


# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class _TestSchema(PluginSchema):
    """Dynamic schema for engine test plugins."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


def make_test_step_resolver(step_map: dict[str, int] | None = None) -> StepResolver:
    """Create a permissive step resolver for testing.

    Returns a mapped step for known nodes, or 1 for unknown nodes. Use this
    when the test doesn't care about specific step values and just needs a
    resolver that won't crash.

    For tests that verify step values flow correctly, prefer
    make_strict_step_resolver() to catch accidentally wrong node IDs.
    """
    _map = {NodeID(k): v for k, v in (step_map or {}).items()}

    def resolve(node_id: NodeID) -> int:
        if node_id in _map:
            return _map[node_id]
        return 1  # Default step for tests

    return resolve


def make_strict_step_resolver(step_map: dict[str, int]) -> StepResolver:
    """Create a strict step resolver that crashes on unmapped node IDs.

    Mirrors production behavior (OrchestrationInvariantError on unknown nodes).
    Use this when the test exercises specific nodes and should verify the
    correct node_id is passed to the resolver.
    """
    _map = {NodeID(k): v for k, v in step_map.items()}

    def resolve(node_id: NodeID) -> int:
        if node_id not in _map:
            raise AssertionError(f"Unexpected NodeID in step resolver: {node_id!r}. Known nodes: {set(_map.keys())}")
        return _map[node_id]

    return resolve
