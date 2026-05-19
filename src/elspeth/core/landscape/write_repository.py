"""Write-side helpers for synthesised Landscape audit records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import Determinism, NodeType, RunStatus
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape._helpers import generate_id
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table


class LandscapeWriteRepository:
    """Write-side repository for synthetic Landscape run records."""

    def __init__(self, db: LandscapeDB) -> None:
        self._db = db

    def record_synthesised_run(
        self,
        *,
        pipeline_yaml: str,
        rows: Sequence[Mapping[str, Any]],
        source_data_hash: str,
        llm_call_count: int,
        plugin_versions: Mapping[str, str],
        started_at: datetime,
        metadata: Mapping[str, Any],
    ) -> str:
        """Insert a cache-replay run plus row/node audit facts.

        The cache stores content, not identity. This method mints a fresh
        Landscape run id for the replay and records the replay marker
        (`seeded_from_cache`, `cache_key`) on the run row.
        """
        if not plugin_versions:
            raise LandscapeRecordError("record_synthesised_run requires at least one plugin version")
        seeded_from_cache = metadata["seeded_from_cache"]
        cache_key = metadata["cache_key"]
        if type(seeded_from_cache) is not bool:
            raise LandscapeRecordError("record_synthesised_run metadata['seeded_from_cache'] must be bool")
        if type(cache_key) is not str:
            raise LandscapeRecordError("record_synthesised_run metadata['cache_key'] must be str")

        run_id = generate_id()
        config = {
            "pipeline_yaml": pipeline_yaml,
            "metadata": dict(metadata),
        }
        node_items = tuple(plugin_versions.items())
        source_node_id = self._node_id(run_id, 0)

        try:
            with self._db.connection() as conn:
                conn.execute(
                    runs_table.insert().values(
                        run_id=run_id,
                        started_at=started_at,
                        completed_at=started_at,
                        config_hash=stable_hash(config),
                        settings_json=canonical_json(config),
                        reproducibility_grade=None,
                        canonical_version="phase4-tutorial-cache-v1",
                        source_schema_json=None,
                        source_field_resolution_json=None,
                        status=RunStatus.COMPLETED.value,
                        export_status=None,
                        export_error=None,
                        exported_at=None,
                        export_format=None,
                        export_sink=None,
                        schema_contract_json=None,
                        schema_contract_hash=None,
                        runtime_val_manifest_json=None,
                        llm_call_count=llm_call_count,
                        seeded_from_cache=seeded_from_cache,
                        cache_key=cache_key,
                    )
                )
                for index, (plugin_name, plugin_version) in enumerate(node_items):
                    node_config = {"plugin_name": plugin_name, "plugin_version": plugin_version}
                    conn.execute(
                        nodes_table.insert().values(
                            node_id=self._node_id(run_id, index),
                            run_id=run_id,
                            plugin_name=plugin_name,
                            node_type=self._node_type(index, len(node_items)).value,
                            plugin_version=plugin_version,
                            source_file_hash=None,
                            determinism=Determinism.DETERMINISTIC.value,
                            config_hash=stable_hash(node_config),
                            config_json=canonical_json(node_config),
                            schema_hash=None,
                            sequence_in_pipeline=index,
                            registered_at=started_at,
                            schema_mode=None,
                            schema_fields_json=None,
                            input_contract_json=None,
                            output_contract_json=None,
                        )
                    )
                for row_index, _row in enumerate(rows):
                    conn.execute(
                        rows_table.insert().values(
                            row_id=f"{run_id}-row-{row_index}",
                            run_id=run_id,
                            source_node_id=source_node_id,
                            row_index=row_index,
                            source_data_hash=source_data_hash,
                            source_data_ref=None,
                            created_at=started_at,
                        )
                    )
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"record_synthesised_run failed — database rejected audit write: {type(exc).__name__}: {exc}"
            ) from exc

        return run_id

    @staticmethod
    def _node_id(run_id: str, index: int) -> str:
        return f"{run_id}-n{index}"

    @staticmethod
    def _node_type(index: int, count: int) -> NodeType:
        if index == 0:
            return NodeType.SOURCE
        if index == count - 1:
            return NodeType.SINK
        return NodeType.TRANSFORM
