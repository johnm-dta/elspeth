"""Write-side helpers for synthesised Landscape audit records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import Determinism, NodeType, RunStatus
from elspeth.contracts.synthesised_audit import SynthesisedNodeSpec
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape._helpers import generate_id
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.run_lifecycle_repository import is_valid_sha256_hex, validate_run_attribution
from elspeth.core.landscape.schema import nodes_table, rows_table, run_attributions_table, runs_table

# Re-export for back-compat with existing import sites (``from
# elspeth.core.landscape.write_repository import SynthesisedNodeSpec``).
# Canonical location is ``elspeth.contracts.synthesised_audit``.
__all__ = ["LandscapeWriteRepository", "SynthesisedNodeSpec"]


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
        node_specs: Sequence[SynthesisedNodeSpec],
        started_at: datetime,
        metadata: Mapping[str, Any],
        openrouter_catalog_sha256: str,
        openrouter_catalog_source: str,
        initiated_by_user_id: str | None = None,
        auth_provider_type: str | None = None,
    ) -> str:
        """Insert a cache-replay run plus row/node audit facts.

        The cache stores content, not identity. This method mints a fresh
        Landscape run id for the replay and records the replay marker
        (``seeded_from_cache``, ``cache_key``) on the run row, plus one
        ``nodes`` row per element of ``node_specs`` carrying the YAML-declared
        role. Structural invariants (exactly one SOURCE at index 0, at least
        one SINK) are enforced offensively before any write — a misshapen
        sequence is a caller bug, not an audit anomaly to silently absorb.
        When requester attribution is supplied, it is written in the same
        transaction as the run row so audit/export queries see cache replays
        the same way they see live executions.
        """
        self._validate_node_specs(node_specs)
        validate_run_attribution(initiated_by_user_id=initiated_by_user_id, auth_provider_type=auth_provider_type)
        seeded_from_cache = metadata["seeded_from_cache"]
        cache_key = metadata["cache_key"]
        if type(seeded_from_cache) is not bool:
            raise LandscapeRecordError("record_synthesised_run metadata['seeded_from_cache'] must be bool")
        if type(cache_key) is not str:
            raise LandscapeRecordError("record_synthesised_run metadata['cache_key'] must be str")
        if type(openrouter_catalog_sha256) is not str or not is_valid_sha256_hex(openrouter_catalog_sha256):
            raise LandscapeRecordError(
                f"record_synthesised_run openrouter_catalog_sha256 must be 64 lowercase hex chars, got {openrouter_catalog_sha256!r}"
            )
        if openrouter_catalog_source not in ("live", "bundled"):
            raise LandscapeRecordError(
                f"record_synthesised_run openrouter_catalog_source must be 'live' or 'bundled', got {openrouter_catalog_source!r}"
            )

        run_id = generate_id()
        config = {
            "pipeline_yaml": pipeline_yaml,
            "metadata": dict(metadata),
        }
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
                        openrouter_catalog_sha256=openrouter_catalog_sha256,
                        openrouter_catalog_source=openrouter_catalog_source,
                    )
                )
                if initiated_by_user_id is not None and auth_provider_type is not None:
                    conn.execute(
                        run_attributions_table.insert().values(
                            run_id=run_id,
                            recorded_at=started_at,
                            initiated_by_user_id=initiated_by_user_id,
                            auth_provider_type=auth_provider_type,
                        )
                    )
                for index, spec in enumerate(node_specs):
                    node_config = {"plugin_name": spec.plugin_name, "plugin_version": spec.plugin_version}
                    conn.execute(
                        nodes_table.insert().values(
                            node_id=self._node_id(run_id, index),
                            run_id=run_id,
                            plugin_name=spec.plugin_name,
                            node_type=spec.node_type.value,
                            plugin_version=spec.plugin_version,
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
                # Cache-replay rows: exactly one source per replayed run, so
                # ``source_row_index = row_index`` (position within the
                # source) and ``ingest_sequence = row_index`` (monotone
                # run-wide order). This is *recording reality*, not
                # fabrication — the cache stores a deterministic single-
                # source sequence; the three fields are genuinely equal.
                #
                # Load-bearing assumption: this path mints exactly ONE
                # source node (``source_node_id = self._node_id(run_id, 0)``
                # at the head of this method). A future contributor adding
                # multi-source cache replay MUST re-derive
                # ``source_row_index`` per source and ``ingest_sequence``
                # globally — the equality above is single-source-specific
                # and would otherwise produce per-source row-index
                # collisions on the ``UniqueConstraint("run_id",
                # "ingest_sequence")`` (filigree elspeth-56c3cda89b).
                for row_index, _row in enumerate(rows):
                    conn.execute(
                        rows_table.insert().values(
                            row_id=f"{run_id}-row-{row_index}",
                            run_id=run_id,
                            source_node_id=source_node_id,
                            row_index=row_index,
                            source_row_index=row_index,
                            ingest_sequence=row_index,
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
    def _validate_node_specs(node_specs: Sequence[SynthesisedNodeSpec]) -> None:
        if len(node_specs) == 0:
            raise LandscapeRecordError("record_synthesised_run requires at least one node spec")
        if node_specs[0].node_type is not NodeType.SOURCE:
            raise LandscapeRecordError(f"record_synthesised_run: first node must be SOURCE, got {node_specs[0].node_type.value}")
        source_count = sum(1 for spec in node_specs if spec.node_type is NodeType.SOURCE)
        if source_count != 1:
            raise LandscapeRecordError(f"record_synthesised_run requires exactly one SOURCE node, got {source_count}")
        sink_count = sum(1 for spec in node_specs if spec.node_type is NodeType.SINK)
        if sink_count < 1:
            raise LandscapeRecordError("record_synthesised_run requires at least one SINK node")
