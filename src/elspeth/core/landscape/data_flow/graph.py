"""Execution-graph audit persistence (split from ``DataFlowRepository``).

Owns the ``nodes`` and ``edges`` audit aggregates: node registration with
audit-safe config sanitization and schema/contract serialization, edge
registration, the graph read models, and post-inference output-contract
updates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from sqlalchemy import select

from elspeth.contracts import (
    ContractAuditRecord,
    Determinism,
    Edge,
    Node,
    NodeType,
    RoutingMode,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import generate_id, now
from elspeth.core.landscape.model_loaders import EdgeLoader, NodeLoader
from elspeth.core.landscape.schema import edges_table, nodes_table

if TYPE_CHECKING:
    from elspeth.contracts.schema import SchemaConfig

__all__ = ["GraphAuditRepository"]


class GraphAuditRepository:
    """Execution-graph registration and read models.

    NOTE: nodes table has composite PK (node_id, run_id). Always filter
    by both columns when querying individual nodes.
    """

    def __init__(
        self,
        ops: DatabaseOps,
        *,
        node_loader: NodeLoader,
        edge_loader: EdgeLoader,
    ) -> None:
        self._ops = ops
        self._node_loader = node_loader
        self._edge_loader = edge_loader

    def _sanitize_node_config_for_audit(self, config: Mapping[str, object], *, plugin_name: str | None) -> Mapping[str, object]:
        """Return an audit-safe node config with secrets fingerprinted.

        Database plugin configs additionally get placement-based DSN
        sanitization: ``url`` is credential-bearing by placement, not by
        name, so the generic secret-name heuristic misses it
        (elspeth-6169a16809). Mirrors the full-config path's database-sink
        gate in ``_fingerprint_config_for_audit``.
        """
        import os

        from elspeth.core.config import _fingerprint_secrets, _sanitize_dsn_option_for_audit

        thawed = deep_thaw(config)
        if type(thawed) is not dict:
            raise TypeError(f"Node config must thaw to dict[str, object], got {type(thawed).__name__}: {thawed!r}")

        allow_raw = False
        if "ELSPETH_ALLOW_RAW_SECRETS" in os.environ:
            allow_raw = os.environ["ELSPETH_ALLOW_RAW_SECRETS"].lower() == "true"
        sanitized = _fingerprint_secrets(thawed, fail_if_no_key=not allow_raw)
        if plugin_name == "database":
            # Node config is flat: the DSN sits at top-level `url`.
            _sanitize_dsn_option_for_audit(
                sanitized,
                option_name="url",
                fingerprint_name="url_password_fingerprint",
                redacted_name="url_password_redacted",
                fail_if_no_key=not allow_raw,
            )
        return sanitized

    def register_node(
        self,
        run_id: str,
        plugin_name: str,
        node_type: NodeType,
        plugin_version: str,
        config: Mapping[str, object],
        *,
        node_id: str | None = None,
        sequence: int | None = None,
        schema_hash: str | None = None,
        determinism: Determinism = Determinism.DETERMINISTIC,
        schema_config: SchemaConfig,
        source_file_hash: str | None = None,
        input_contract: SchemaContract | None = None,
        output_contract: SchemaContract | None = None,
    ) -> Node:
        """Register a node in the execution graph.

        Args:
            run_id: Run this node belongs to
            plugin_name: Name of the plugin (None for gates and coalesces, which are config-driven)
            node_type: NodeType enum (SOURCE, TRANSFORM, GATE, AGGREGATION, COALESCE, SINK)
            plugin_version: Version of the plugin (None for non-plugin nodes)
            config: Node configuration
            node_id: Optional node ID (generated if not provided)
            sequence: Position in pipeline
            schema_hash: Optional input/output schema hash
            determinism: Determinism enum (defaults to DETERMINISTIC)
            schema_config: Schema configuration for audit trail (WP-11.99)
            source_file_hash: Optional truncated SHA-256 hash of the plugin source file
            input_contract: Optional input schema contract (what node requires)
            output_contract: Optional output schema contract (what node guarantees)

        Returns:
            Node model
        """
        node_id = node_id or generate_id()
        audit_safe_config = self._sanitize_node_config_for_audit(config, plugin_name=plugin_name)
        config_json = canonical_json(audit_safe_config)
        config_hash = stable_hash(audit_safe_config)
        timestamp = now()

        # Extract schema info for audit (WP-11.99)
        schema_fields_json: str | None = None
        schema_fields_list: list[dict[str, object]] | None = None

        # Extract schema mode directly - no translation needed
        schema_mode = schema_config.mode
        if not schema_config.is_observed and schema_config.fields:
            # FieldDefinition.to_dict() returns dict[str, str | bool]
            # Cast each dict to wider type for storage
            field_dicts = [f.to_dict() for f in schema_config.fields]
            schema_fields_list = [dict(d) for d in field_dicts]
            schema_fields_json = canonical_json(field_dicts)

        # Convert schema contracts to audit records if provided
        input_contract_json: str | None = None
        output_contract_json: str | None = None
        if input_contract is not None:
            input_contract_json = ContractAuditRecord.from_contract(input_contract).to_json()
        if output_contract is not None:
            output_contract_json = ContractAuditRecord.from_contract(output_contract).to_json()

        node = Node(
            node_id=node_id,
            run_id=run_id,
            plugin_name=plugin_name,
            node_type=node_type,
            plugin_version=plugin_version,
            determinism=determinism,
            config_hash=config_hash,
            config_json=config_json,
            source_file_hash=source_file_hash,
            schema_hash=schema_hash,
            sequence_in_pipeline=sequence,
            registered_at=timestamp,
            schema_mode=schema_mode,
            schema_fields=schema_fields_list,
        )

        self._ops.execute_insert(
            nodes_table.insert().values(
                node_id=node.node_id,
                run_id=node.run_id,
                plugin_name=node.plugin_name,
                node_type=node.node_type,
                plugin_version=node.plugin_version,
                determinism=node.determinism,
                config_hash=node.config_hash,
                config_json=node.config_json,
                source_file_hash=node.source_file_hash,
                schema_hash=node.schema_hash,
                sequence_in_pipeline=node.sequence_in_pipeline,
                registered_at=node.registered_at,
                schema_mode=node.schema_mode,
                schema_fields_json=schema_fields_json,
                input_contract_json=input_contract_json,
                output_contract_json=output_contract_json,
            )
        )

        return node

    def register_edge(
        self,
        run_id: str,
        from_node_id: str,
        to_node_id: str,
        label: str,
        mode: RoutingMode,
        *,
        edge_id: str | None = None,
    ) -> Edge:
        """Register an edge in the execution graph.

        Args:
            run_id: Run this edge belongs to
            from_node_id: Source node
            to_node_id: Destination node
            label: Edge label ("continue", route name, etc.)
            mode: RoutingMode enum (MOVE or COPY)
            edge_id: Optional edge ID (generated if not provided)

        Returns:
            Edge model
        """
        edge_id = edge_id or generate_id()
        timestamp = now()

        edge = Edge(
            edge_id=edge_id,
            run_id=run_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            label=label,
            default_mode=mode,
            created_at=timestamp,
        )

        self._ops.execute_insert(
            edges_table.insert().values(
                edge_id=edge.edge_id,
                run_id=edge.run_id,
                from_node_id=edge.from_node_id,
                to_node_id=edge.to_node_id,
                label=edge.label,
                default_mode=edge.default_mode,
                created_at=edge.created_at,
            )
        )

        return edge

    def get_node(self, node_id: str, run_id: str) -> Node | None:
        """Get a node by its composite primary key (node_id, run_id).

        NOTE: The nodes table has a composite PK (node_id, run_id). The same
        node_id can exist in multiple runs, so run_id is required to identify
        the specific node.

        Args:
            node_id: Node ID to retrieve
            run_id: Run ID the node belongs to

        Returns:
            Node model or None if not found
        """
        query = select(nodes_table).where((nodes_table.c.node_id == node_id) & (nodes_table.c.run_id == run_id))
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._node_loader.load(row)

    def get_nodes(self, run_id: str) -> list[Node]:
        """Get all nodes for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Node models, ordered by sequence (NULL sequences last)
        """
        query = (
            select(nodes_table)
            .where(nodes_table.c.run_id == run_id)
            # Use nullslast() for consistent NULL handling across databases
            # Nodes without sequence (e.g., dynamically added) sort last
            # Tiebreakers (registered_at, node_id) ensure deterministic ordering
            # for export signing when sequence_in_pipeline is NULL
            .order_by(
                nodes_table.c.sequence_in_pipeline.nullslast(),
                nodes_table.c.registered_at,
                nodes_table.c.node_id,
            )
        )
        rows = self._ops.execute_fetchall(query)
        return [self._node_loader.load(row) for row in rows]

    def get_node_contracts(
        self, run_id: str, node_id: str, *, allow_missing: bool = False
    ) -> tuple[SchemaContract | None, SchemaContract | None]:
        """Get input and output contracts for a node.

        Retrieves stored schema contracts and verifies integrity via hash.

        Args:
            run_id: Run ID the node belongs to
            node_id: Node ID to query
            allow_missing: If False (default), crash when node not found
                (Tier 1 invariant — our audit data must be present).
                Set to True only for external query paths (MCP, analysis).

        Returns:
            Tuple of (input_contract, output_contract), either may be None
            if the node exists but has no contracts recorded.

        Raises:
            AuditIntegrityError: If node not found and allow_missing is False
            ValueError: If stored contract fails integrity verification
        """
        query = select(
            nodes_table.c.input_contract_json,
            nodes_table.c.output_contract_json,
        ).where((nodes_table.c.node_id == node_id) & (nodes_table.c.run_id == run_id))
        row = self._ops.execute_fetchone(query)

        if row is None:
            if allow_missing:
                return None, None
            raise AuditIntegrityError(
                f"Node not found in audit trail: node_id={node_id!r}, run_id={run_id!r}. Expected node to exist (Tier 1 data)."
            )

        input_contract: SchemaContract | None = None
        output_contract: SchemaContract | None = None

        if row.input_contract_json is not None:
            audit_record = ContractAuditRecord.from_json(row.input_contract_json)
            input_contract = audit_record.to_schema_contract()

        if row.output_contract_json is not None:
            audit_record = ContractAuditRecord.from_json(row.output_contract_json)
            output_contract = audit_record.to_schema_contract()

        return input_contract, output_contract

    def get_edges(self, run_id: str) -> list[Edge]:
        """Get all edges for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Edge models for this run, ordered by created_at then edge_id
            for deterministic export signatures.
        """
        query = select(edges_table).where(edges_table.c.run_id == run_id).order_by(edges_table.c.created_at, edges_table.c.edge_id)
        rows = self._ops.execute_fetchall(query)
        return [self._edge_loader.load(row) for row in rows]

    def get_edge(self, edge_id: str) -> Edge:
        """Get a single edge by ID.

        Tier 1: crash on missing — an edge_id from our own routing_events
        table MUST resolve. Missing means audit DB corruption.

        Args:
            edge_id: Edge ID to look up

        Returns:
            Edge model

        Raises:
            AuditIntegrityError: If edge not found (audit integrity violation)
        """
        query = select(edges_table).where(edges_table.c.edge_id == edge_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            raise AuditIntegrityError(
                f"Audit integrity violation: edge '{edge_id}' not found. "
                f"A routing_event references a non-existent edge. "
                f"This indicates database corruption."
            )
        return self._edge_loader.load(row)

    def get_edge_map(self, run_id: str) -> dict[tuple[str, str], str]:
        """Get edge mapping for a run (from_node_id, label) -> edge_id.

        Args:
            run_id: Run to query

        Returns:
            Dictionary mapping (from_node_id, label) to edge_id

        Raises:
            AuditIntegrityError: If run has no edges registered (data corruption).
                DAG compilation always registers edges, so an empty map
                indicates the run was never properly initialized.

        Note:
            This encapsulates Landscape schema access for Orchestrator resume.
            Edge IDs are required for FK integrity when recording routing events.
        """
        query = select(edges_table).where(edges_table.c.run_id == run_id)
        edges = self._ops.execute_fetchall(query)

        edge_map: dict[tuple[str, str], str] = {}
        for edge in edges:
            edge_map[(edge.from_node_id, edge.label)] = edge.edge_id

        if not edge_map:
            raise AuditIntegrityError(
                f"Run {run_id!r} has no edges registered — cannot build edge map. "
                f"DAG compilation always registers edges; an empty map indicates "
                f"the run was never properly initialized or database corruption."
            )

        return edge_map

    def update_node_output_contract(
        self,
        run_id: str,
        node_id: str,
        contract: SchemaContract,
    ) -> None:
        """Update a node's output_contract after first-row inference or schema evolution.

        Called in two scenarios:
        1. Source infers schema from first valid row during OBSERVED mode
        2. Transform adds fields during execution (schema evolution)

        Args:
            run_id: Run containing the node
            node_id: Node to update (source or transform node)
            contract: SchemaContract with inferred/evolved fields

        Note:
            This is the complement to ``update_run_source_contract()`` for
            node-level contracts (the per-source ``run_sources`` writer that
            superseded the deleted run-level singleton ``update_run_contract``).
            Used for dynamic schema discovery and transform schema evolution.
        """
        audit_record = ContractAuditRecord.from_contract(contract)
        output_contract_json = audit_record.to_json()

        self._ops.execute_update(
            nodes_table.update()
            .where((nodes_table.c.run_id == run_id) & (nodes_table.c.node_id == node_id))
            .values(output_contract_json=output_contract_json)
        )
