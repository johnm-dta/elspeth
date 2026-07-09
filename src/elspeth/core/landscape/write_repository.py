"""Write-side helpers for synthesised Landscape audit records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import Determinism, NodeType, RunStatus
from elspeth.contracts.synthesised_audit import SynthesisedNodeSpec
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.ids import generate_id
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.run_lifecycle_repository import is_valid_sha256_hex, validate_run_attribution
from elspeth.core.landscape.schema import nodes_table, rows_table, run_attributions_table, runs_table

# Re-export for back-compat with existing import sites (``from
# elspeth.core.landscape.write_repository import SynthesisedNodeSpec``).
# Canonical location is ``elspeth.contracts.synthesised_audit``.
__all__ = ["LandscapeWriteRepository", "SynthesisedNodeSpec"]

_ROW_IDENTITY_KEYS = frozenset({"source_node_index", "source_row_index", "ingest_sequence", "source_data_hash"})
_SYNTHESISED_RUN_METADATA_KEYS = frozenset({"seeded_from_cache", "cache_key"})


@dataclass(frozen=True, slots=True)
class _SynthesisedRowIdentity:
    """Explicit row provenance for synthesised Landscape rows."""

    source_node_index: int
    source_row_index: int
    ingest_sequence: int
    source_data_hash: str


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
        role. Structural invariants (one or more leading SOURCE nodes, at
        least one SINK) are enforced offensively before any write — a
        misshapen sequence is a caller bug, not an audit anomaly to silently
        absorb. Single-source cache rows retain the legacy positional identity
        fallback. Multi-source rows must carry explicit row identity fields
        (``source_node_index``, ``source_row_index``, ``ingest_sequence``,
        ``source_data_hash``) so the writer never guesses provenance.
        When requester attribution is supplied, it is written in the same
        transaction as the run row so audit/export queries see cache replays
        the same way they see live executions.
        """
        source_node_indices = self._validate_node_specs(node_specs)
        validate_run_attribution(initiated_by_user_id=initiated_by_user_id, auth_provider_type=auth_provider_type)
        metadata_keys = frozenset(metadata)
        if metadata_keys != _SYNTHESISED_RUN_METADATA_KEYS:
            details = []
            if _SYNTHESISED_RUN_METADATA_KEYS - metadata_keys:
                details.append("missing required metadata keys")
            if metadata_keys - _SYNTHESISED_RUN_METADATA_KEYS:
                details.append("unexpected metadata keys")
            raise LandscapeRecordError(
                f"record_synthesised_run metadata must contain only seeded_from_cache and cache_key ({'; '.join(details)})"
            )
        seeded_from_cache = metadata["seeded_from_cache"]
        cache_key = metadata["cache_key"]
        if type(seeded_from_cache) is not bool:
            raise LandscapeRecordError("record_synthesised_run metadata['seeded_from_cache'] must be bool")
        if type(cache_key) is not str:
            raise LandscapeRecordError("record_synthesised_run metadata['cache_key'] must be str")
        if not is_valid_sha256_hex(cache_key):
            raise LandscapeRecordError(f"record_synthesised_run cache_key must be 64 lowercase hex chars, got {cache_key!r}")
        if type(llm_call_count) is not int or llm_call_count < 0:
            raise LandscapeRecordError(f"record_synthesised_run llm_call_count must be a non-negative integer, got {llm_call_count!r}")
        if type(openrouter_catalog_sha256) is not str or not is_valid_sha256_hex(openrouter_catalog_sha256):
            raise LandscapeRecordError(
                f"record_synthesised_run openrouter_catalog_sha256 must be 64 lowercase hex chars, got {openrouter_catalog_sha256!r}"
            )
        if type(source_data_hash) is not str or not is_valid_sha256_hex(source_data_hash):
            raise LandscapeRecordError(f"record_synthesised_run source_data_hash must be 64 lowercase hex chars, got {source_data_hash!r}")
        if openrouter_catalog_source not in ("live", "bundled"):
            raise LandscapeRecordError(
                f"record_synthesised_run openrouter_catalog_source must be 'live' or 'bundled', got {openrouter_catalog_source!r}"
            )

        run_id = generate_id()
        audit_metadata = {
            "seeded_from_cache": seeded_from_cache,
            "cache_key": cache_key,
        }
        config = {
            "pipeline_yaml": pipeline_yaml,
            "metadata": audit_metadata,
        }
        source_node_ids_by_index = {index: self._node_id(run_id, index) for index in source_node_indices}

        try:
            with self._db.write_connection() as conn:
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
                # Single-source cache rows can safely use the legacy
                # positional identity fallback: source_row_index and
                # ingest_sequence both equal row_index because there is only
                # one source. Multi-source rows must carry explicit
                # provenance and are rejected below if any identity field is
                # absent or malformed.
                for row_index, row in enumerate(rows):
                    row_identity = self._row_identity(
                        row,
                        row_index=row_index,
                        source_node_indices=source_node_indices,
                        fallback_source_data_hash=source_data_hash,
                    )
                    conn.execute(
                        rows_table.insert().values(
                            row_id=f"{run_id}-row-{row_index}",
                            run_id=run_id,
                            source_node_id=source_node_ids_by_index[row_identity.source_node_index],
                            row_index=row_index,
                            source_row_index=row_identity.source_row_index,
                            ingest_sequence=row_identity.ingest_sequence,
                            source_data_hash=row_identity.source_data_hash,
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
    def _validate_node_specs(node_specs: Sequence[SynthesisedNodeSpec]) -> tuple[int, ...]:
        if len(node_specs) == 0:
            raise LandscapeRecordError("record_synthesised_run requires at least one node spec")
        if node_specs[0].node_type is not NodeType.SOURCE:
            raise LandscapeRecordError(f"record_synthesised_run: first node must be SOURCE, got {node_specs[0].node_type.value}")
        source_indices = tuple(index for index, spec in enumerate(node_specs) if spec.node_type is NodeType.SOURCE)
        first_non_source_index = next(
            (index for index, spec in enumerate(node_specs) if spec.node_type is not NodeType.SOURCE), len(node_specs)
        )
        if any(index > first_non_source_index for index in source_indices):
            raise LandscapeRecordError("record_synthesised_run: SOURCE nodes must precede transforms and sinks")
        sink_count = sum(1 for spec in node_specs if spec.node_type is NodeType.SINK)
        if sink_count < 1:
            raise LandscapeRecordError("record_synthesised_run requires at least one SINK node")
        return source_indices

    @staticmethod
    def _row_identity(
        row: Mapping[str, Any],
        *,
        row_index: int,
        source_node_indices: tuple[int, ...],
        fallback_source_data_hash: str,
    ) -> _SynthesisedRowIdentity:
        if len(source_node_indices) == 1:
            return _SynthesisedRowIdentity(
                source_node_index=source_node_indices[0],
                source_row_index=row_index,
                ingest_sequence=row_index,
                source_data_hash=fallback_source_data_hash,
            )

        explicit_keys = _ROW_IDENTITY_KEYS.intersection(row.keys())
        if not explicit_keys:
            raise LandscapeRecordError(
                "record_synthesised_run: multi-source synthesised rows require explicit row identity fields "
                "source_node_index, source_row_index, ingest_sequence, and source_data_hash"
            )
        if explicit_keys != _ROW_IDENTITY_KEYS:
            missing = sorted(_ROW_IDENTITY_KEYS - explicit_keys)
            raise LandscapeRecordError(f"record_synthesised_run row identity is incomplete; missing {missing}")

        source_node_index = row["source_node_index"]
        source_row_index = row["source_row_index"]
        ingest_sequence = row["ingest_sequence"]
        row_source_data_hash = row["source_data_hash"]
        if type(source_node_index) is not int or source_node_index < 0:
            raise LandscapeRecordError("record_synthesised_run row source_node_index must be a non-negative integer")
        if source_node_index not in source_node_indices:
            raise LandscapeRecordError(f"record_synthesised_run row source_node_index {source_node_index} does not reference a SOURCE node")
        if type(source_row_index) is not int or source_row_index < 0:
            raise LandscapeRecordError("record_synthesised_run row source_row_index must be a non-negative integer")
        if type(ingest_sequence) is not int or ingest_sequence < 0:
            raise LandscapeRecordError("record_synthesised_run row ingest_sequence must be a non-negative integer")
        if type(row_source_data_hash) is not str or not is_valid_sha256_hex(row_source_data_hash):
            raise LandscapeRecordError(
                f"record_synthesised_run row source_data_hash must be 64 lowercase hex chars, got {row_source_data_hash!r}"
            )
        return _SynthesisedRowIdentity(
            source_node_index=source_node_index,
            source_row_index=source_row_index,
            ingest_sequence=ingest_sequence,
            source_data_hash=row_source_data_hash,
        )
