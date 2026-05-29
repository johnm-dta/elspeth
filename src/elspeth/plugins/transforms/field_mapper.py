"""FieldMapper transform plugin.

Renames, selects, and reorganizes row fields.

IMPORTANT: Transforms use allow_coercion=False to catch upstream bugs.
If the source outputs wrong types, the transform crashes immediately.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import Field, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.contract_propagation import narrow_contract_to_output
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.infrastructure.sentinels import MISSING
from elspeth.plugins.infrastructure.utils import get_nested_field


class FieldMapperConfig(TransformDataConfig):
    """Configuration for field mapper transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {mode: observed}' for dynamic field handling.
    """

    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from existing input field names to output field names.",
    )
    select_only: bool = Field(default=False, description="When true, emit only fields named in the mapping.")
    strict: bool = Field(default=False, description="When true, fail if any mapped source field is missing from an input row.")

    @model_validator(mode="after")
    def _reject_duplicate_targets(self) -> FieldMapperConfig:
        """Reject mappings where multiple sources map to the same target.

        Duplicate targets cause silent data loss: the last write wins,
        overwriting the value from the earlier mapping without any error.
        This also produces incorrect contract metadata (type/original_name
        lineage from the wrong source field).
        """
        if not self.mapping:
            return self
        targets: list[str] = list(self.mapping.values())
        seen: set[str] = set()
        duplicates: set[str] = set()
        for target in targets:
            if target in seen:
                duplicates.add(target)
            seen.add(target)
        if duplicates:
            # Build source->target details for the error message
            collisions: dict[str, list[str]] = {}
            for source, target in self.mapping.items():
                if target in duplicates:
                    collisions.setdefault(target, []).append(source)
            msg = (
                f"Mapping has duplicate target field names: "
                f"{', '.join(f'{t!r} <- {srcs}' for t, srcs in sorted(collisions.items()))}. "
                f"Multiple sources mapping to the same target causes silent data loss."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _reject_overlapping_rename_graphs(self) -> FieldMapperConfig:
        """Reject mappings whose targets are also sources.

        FieldMapper preserves audit lineage for renames. Swaps/chains require
        target fields to carry metadata from a different source while also
        existing as source fields themselves, which is ambiguous without a more
        expressive contract model.
        """
        if not self.mapping:
            return self

        sources = set(self.mapping)
        overlaps: dict[str, list[str]] = {}
        for source, target in self.mapping.items():
            if source != target and target in sources:
                if target not in overlaps:
                    overlaps[target] = []
                overlaps[target].append(source)

        if overlaps:
            details = ", ".join(f"{target!r} is both target for {srcs} and source" for target, srcs in sorted(overlaps.items()))
            raise ValueError(
                "Mapping contains an overlapping rename graph: "
                f"{details}. Targets that are also sources are order-dependent and can cause silent data loss."
            )

        return self


class FieldMapper(BaseTransform):
    """Map, rename, and select row fields.

    Config options:
        schema: Required. Schema for input/output (use {mode: observed} for any fields)
        mapping: Dict of source_field -> target_field
            - Simple: {"old": "new"} renames old to new
            - Nested: {"meta.source": "origin"} extracts nested field
        select_only: If True, only include mapped fields (default: False)
        strict: If True, error on missing source fields (default: False)
    """

    name = "field_mapper"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:946dd005c41af030"
    config_model = FieldMapperConfig

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "mapping": {
                "field_mapper_probe_source": "field_mapper_probe_target",
            },
            "strict": True,
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = FieldMapperConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._mapping: dict[str, str] = cfg.mapping
        self._select_only: bool = cfg.select_only
        self._strict: bool = cfg.strict
        self._schema_config = cfg.schema_config

        self.declared_output_fields = self._derive_declared_output_fields(cfg)

        self.input_schema, self.output_schema = self._create_schemas(
            cfg.schema_config,
            "FieldMapper",
            adds_fields=True,
        )
        self._output_schema_config = self._build_field_mapper_output_schema_config(cfg)

    @staticmethod
    def _is_static_normalized_source(source: str) -> bool:
        """Return True when constructor-time contract math can use source."""
        return "." not in source and source.isidentifier()

    @staticmethod
    def _is_unresolved_original_source(source: str) -> bool:
        """Return True when source may be an original header resolved only at runtime."""
        return "." not in source and not source.isidentifier()

    @classmethod
    def _mapping_target_is_guaranteed(
        cls,
        cfg: FieldMapperConfig,
        source: str,
        base_guaranteed: set[str],
    ) -> bool:
        """Whether target exists on every successful row for this mapping."""
        if cls._is_unresolved_original_source(source):
            return False
        if cfg.strict:
            return True
        return cls._is_static_normalized_source(source) and source in base_guaranteed

    @classmethod
    def _derive_declared_output_fields(cls, cfg: FieldMapperConfig) -> frozenset[str]:
        """Derive targets safe for executor-level declared-output checks."""
        base_guaranteed = set(cfg.schema_config.guaranteed_fields or ())
        return frozenset(
            target
            for source, target in cfg.mapping.items()
            if source != target and cls._mapping_target_is_guaranteed(cfg, source, base_guaranteed)
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the real rename/drop path for the backward invariant."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name="field_mapper_probe_source",
                value="mapped",
            )
        ]

    def _build_field_mapper_output_schema_config(self, cfg: FieldMapperConfig) -> SchemaConfig:
        """Build output schema config reflecting the mapped output shape.

        FieldMapper is shape-changing: it removes source fields and adds target fields.
        The base _build_output_schema_config() incorrectly copies input fields into
        output guarantees. This method builds the correct output field set.

        When select_only=True: output guarantees are ONLY the mapping targets.
        When select_only=False: output guarantees are input fields MINUS removed
            sources PLUS new targets.
        """
        base_guaranteed = set(cfg.schema_config.guaranteed_fields or ())
        guaranteed_targets = {
            target for source, target in cfg.mapping.items() if self._mapping_target_is_guaranteed(cfg, source, base_guaranteed)
        }

        if cfg.select_only:
            # Only mapped targets appear in output
            output_fields = guaranteed_targets
        else:
            # Input fields minus removed sources plus new targets.
            # A source is removed from output when it's renamed to a different target.
            has_unresolved_original_removal = any(
                source != target and self._is_unresolved_original_source(source) for source, target in cfg.mapping.items()
            )
            if has_unresolved_original_removal:
                passthrough_fields: set[str] = set()
            else:
                removed_sources = {
                    source
                    for source, target in cfg.mapping.items()
                    if source != target and self._is_static_normalized_source(source) and source in base_guaranteed
                }
                passthrough_fields = base_guaranteed - removed_sources
            output_fields = passthrough_fields | guaranteed_targets

        # Always include declared_output_fields (targets that aren't also sources)
        output_fields |= self.declared_output_fields

        # Preserve None-vs-empty-tuple semantics: None = abstain, () = explicitly empty.
        # If upstream declared guarantees or we computed non-empty output, declare explicitly.
        upstream_declared = cfg.schema_config.guaranteed_fields is not None
        if upstream_declared or output_fields:
            guaranteed_fields_result = tuple(sorted(output_fields))
        else:
            guaranteed_fields_result = None

        return SchemaConfig(
            mode=cfg.schema_config.mode,
            fields=cfg.schema_config.fields,
            guaranteed_fields=guaranteed_fields_result,
            audit_fields=cfg.schema_config.audit_fields,
            required_fields=cfg.schema_config.required_fields,
        )

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Apply field mapping to row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with mapped row data

        Raises:
            PluginContractViolation: Raised by executor if row fails input schema
                validation. This indicates a bug in the upstream source/transform.
        """
        # Keep a normalized dict view only for validation and dotted-path lookups.
        row_data = row.to_dict()

        # Start with empty or copy depending on select_only
        if self._select_only:
            output: dict[str, Any] = {}
        else:
            output = copy.deepcopy(row_data)

        # Apply mappings
        applied_mappings: dict[str, str] = {}
        for source, target in self._mapping.items():
            if "." in source:
                # Dotted-path navigation is an operation on Tier-2 row values, not a
                # type-contract check. Contracts are flat (no nested-shape guarantee),
                # so a PRESENT non-dict intermediate (e.g. ``user`` is a str when the
                # mapping expects ``user.name``) is operation-unsafe data, not an
                # upstream type-contract violation. Route the offending row to on_error
                # — recorded and attributable — rather than raising and crashing the
                # whole run on a single malformed nested value. A genuinely absent
                # intermediate still returns MISSING (handled below), preserving the
                # strict/non-strict distinction for true absence.
                try:
                    value = get_nested_field(row_data, source)
                except TypeError as exc:
                    return TransformResult.error(
                        {
                            "reason": "type_mismatch",
                            "field": source,
                            "error": str(exc),
                            "message": f"Dotted-path field '{source}' is not navigable on this row: {exc}",
                        }
                    )
            elif source in row:
                value = row[source]
            else:
                value = MISSING

            if value is MISSING:
                if self._strict:
                    return TransformResult.error(
                        {"reason": "missing_field", "field": source, "message": f"Required field '{source}' not found in row"}
                    )
                continue  # Skip missing fields in non-strict mode

            # Remove old key if it exists (for rename within same dict)
            if not self._select_only and "." not in source and source in row:
                if source in output:
                    del output[source]
                else:
                    normalized_source = row.contract.resolve_name(source)
                    if normalized_source in output:
                        del output[normalized_source]

            output[target] = value
            applied_mappings[source] = target

        # Track field changes
        fields_modified: list[str] = []
        fields_added: list[str] = []
        for target in applied_mappings.values():
            if target in row:
                fields_modified.append(target)
            else:
                fields_added.append(target)

        # Update contract to reflect field mapping (renames and removals)
        output_contract = narrow_contract_to_output(
            input_contract=row.contract,
            output_row=output,
            renamed_fields=applied_mappings,
        )
        output_contract = self._align_output_contract(output_contract)

        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "mapped",
                "fields_modified": fields_modified,
                "fields_added": fields_added,
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name="field_mapper",
                issue_code=None,
                summary="Rename, drop, or reorder row fields. Stateless and shape-changing — declares new field names in output_schema.",
                composer_hints=(
                    "Use 'mapping' to rename; 'drop' to discard; 'include' to whitelist; 'rename_only' to skip drop semantics.",
                    "Use select_only: true when cleanup means 'save only these fields'; with select_only true, mapping should whitelist exactly the saved output fields.",
                    "For scraped-content cleanup before a user-facing sink, field_mapper is the only utility transform that actually removes raw fields. Place it immediately before the sink and omit raw content and fingerprint fields from mapping.",
                    "For web_scrape content enriched by an LLM and saved without raw page bodies, the final topology is source -> web_scrape -> llm -> field_mapper(cleanup) -> sink. A JSON sink named cleanup is not a cleanup transform.",
                    "A validator-valid direct route from web_scrape or an LLM to the sink is still incomplete when raw scraped-content cleanup is required; insert or restore this field_mapper immediately before the sink.",
                    "A cleanup stream name is not a cleanup node; if an upstream producer points to a cleanup stream, create a field_mapper that consumes it before stopping, and do not offer to repair it later.",
                    "For final cleanup routing, set the upstream LLM or scraper on_success to the cleanup mapper, and set the cleanup mapper on_success to the sink.",
                    "If an LLM routes directly to a JSON sink whose name sounds like cleanup, cleanup is still missing; the LLM passes through raw scrape fields until this field_mapper whitelists them.",
                    "If a final cleanup field_mapper points to an intermediate stream with no downstream node, route the mapper directly to the existing sink by setting on_success to the sink name or by using an on_success edge. Do not remove the cleanup mapper or output to clear the validation error.",
                    "A field_mapper before web_scrape or before raw scraped fields exist cannot satisfy scraped-content cleanup; source-shaping mappers are separate from final cleanup.",
                    "Final cleanup should preserve requested enrichment, extraction, scoring, or LLM response fields unless the user explicitly asked to drop them.",
                    "If the user already asked to remove, drop, exclude, or avoid saving raw scrape fields, that request is the authorization and requirement to add the cleanup field_mapper; do not ask whether to add cleanup later.",
                    "For scraped-content cleanup review, use user_term 'drop_raw_html_fields' even when the configured raw body field is named content, html, raw_html, or another page-body field.",
                    "When field_mapper implements a cleanup, retention, drop, or output-shaping choice that the user did not spell out mechanically, stage a pipeline_decision interpretation requirement on that field_mapper node before set_pipeline and request its review after mutation succeeds.",
                    "The pipeline_decision review records the row-shaping decision for audit; it is not permission to omit the cleanup node.",
                    "naming a sink or output like cleanup, filtered, or final does not clean data; only a field_mapper row-shaping transform changes which fields reach the sink.",
                    "Renames are pure transformations — no coercion happens here. Use type_coerce for type changes.",
                    "If the downstream consumer expects a specific field name not in the source, field_mapper is the right tool.",
                ),
            )
        return None
