"""Composer sessions plane — pipeline-state, recipe application, and interpretation-review handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Final, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.composer_interpretation import (
    InterpretationChoice,
    InterpretationEventRecord,
    InterpretationKind,
    InterpretationSource,
)
from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.recipes import (
    RecipeValidationError,
    apply_recipe,
)
from elspeth.web.composer.redaction import (
    ApplyPipelineRecipeArgumentsModel,
    SetPipelineArgumentsModel,
    redact_source_storage_path,
)
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    EdgeType,
    NodeSpec,
    NodeType,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationSummary,
    _batch_aware_placement_error,
    _batch_aware_required_input_fields_error,
    _validate_gate_expression,
    _validate_gate_route_parity,
)
from elspeth.web.composer.tools._common import (
    _DEFAULT_SOURCE_VALIDATION_FAILURE,
    _FULL_STATE_COMPONENT_ALIAS_SET,
    _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
    ToolContext,
    ToolResult,
    _credential_wiring_contract_failure,
    _discovery_result,
    _failure_result,
    _graph_repair_suggestions,
    _missing_output_options_repair_error,
    _mutation_result,
    _options_with_default_llm_reviews,
    _prevalidate_sink,
    _prevalidate_source,
    _prevalidate_transform,
    _resolver_owned_interpretation_requirement_error,
    _runtime_owned_llm_option_error,
    _semantic_contracts_payload,
    _serialize_full_pipeline_state,
    _serialize_node,
    _serialize_output,
    _serialize_source,
    _validate_mutation_arguments,
    _validate_plugin_name,
    _validate_sink_path,
    _validate_source_path,
    _validate_transform_provider_config_path,
    _validate_transform_provider_config_policy,
    _vf_destination_note,
    validate_composer_file_sink_collision_policy,
)
from elspeth.web.composer.tools.blobs import (
    _blob_create_payload,
    _blob_creation_provenance,
    _persist_prepared_blob_create,
    _prepare_blob_create,
    _PreparedBlobCreate,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
)
from elspeth.web.composer.tools.sources import (
    _MIME_TO_SOURCE,
    _delimiter_extra_for_csv_blob,
    _header_only_inline_csv_conflict,
    _options_with_source_blob_review,
    _reject_manual_source_authoring,
    _reject_manual_source_blob_ref,
    _resolve_source_blob,
    _ResolvedSourceBlob,
    _source_authoring_options,
    _source_component_id,
)
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    SOURCE_AUTHORING_KEY,
    SOURCE_COMPONENT_ID,
    composition_review_contract_error,
    interpretation_sites,
    transform_vague_term_site_tuples,
    vague_term_wiring_count,
    validate_pipeline_decision_node_semantics,
)
from elspeth.web.validation import (
    _reject_credential_shaped_content,
    _validate_accepted_value_content,
)

ADVISOR_TRIGGER_PROACTIVE_SECURITY: Final[str] = "proactive_security_safety"

ADVISOR_TRIGGER_PROACTIVE_RED_LISTED: Final[str] = "proactive_red_listed_plugin"

ADVISOR_TRIGGER_VALUES: Final[tuple[str, ...]] = (
    ADVISOR_TRIGGER_PROACTIVE_SECURITY,
    ADVISOR_TRIGGER_PROACTIVE_RED_LISTED,
)

# Backend-synthesized triggers for the deterministic advisor checkpoints
# (early/end). Deliberately NOT in ADVISOR_TRIGGER_VALUES: those are the
# LLM-selectable set validated against Tier-3 input; these are produced by the
# trusted compose loop itself and bypass the Tier-3 trigger allowlist.
ADVISOR_TRIGGER_DETERMINISTIC_EARLY: Final[str] = "deterministic_early_checkpoint"

ADVISOR_TRIGGER_DETERMINISTIC_END: Final[str] = "deterministic_end_checkpoint"


class _RequestInterpretationReviewArgumentsModel(BaseModel):
    """Tier-3 trust-boundary model for the ``request_interpretation_review`` tool.

    All fields are LLM-supplied and constrained mechanically:

    * ``affected_node_id`` — short identifier; 256-char cap matches the wire
      cap used by ``upsert_node.id``.
    * ``kind`` — closed interpretation class for the review row.
    * ``user_term`` and ``llm_draft`` — capped at 8192 chars to defend against
      pathological inputs that would distend the audit row beyond the
      schema's 8192-byte expectation (see ``interpretation_events_table``
      column definitions; the schema-level body limit also bounds these
      at the ASGI boundary).

    ``extra="forbid"`` rejects unknown keys structurally so a misrouted
    argument shape (e.g., the LLM passing ``id`` instead of
    ``affected_node_id``) fails fast with a clear ARG_ERROR rather than
    silently dropping the typo and validating a partial payload.
    """

    affected_node_id: str = Field(min_length=1, max_length=256)
    kind: InterpretationKind
    user_term: str = Field(min_length=1, max_length=8192)
    llm_draft: str = Field(min_length=1, max_length=8192)

    model_config = ConfigDict(extra="forbid")


def _validate_source_artifact_review_content(value: str) -> None:
    """Validate generated source-artifact review text.

    Invented-source reviews carry source artifacts (CSV/JSONL/URL lists), not
    single-phrase interpretation text. They may be multiline, but still must
    not carry template metacharacters, credential-shaped content, or
    non-whitespace control characters.
    """

    if "{{" in value or "}}" in value:
        raise ValueError("source artifact review content must not contain template metacharacters {{ or }}")
    for character in value:
        codepoint = ord(character)
        if codepoint == 0x7F or (codepoint < 0x20 and character not in "\t\n\r"):
            raise ValueError("source artifact review content must not contain non-printable control characters")
    for line in value.splitlines():
        if len(line) > 1024:
            raise ValueError("source artifact review content has a line exceeding the 1024-character limit")
    _reject_credential_shaped_content(value)


def _options_with_inline_blob_source_review(
    options: Mapping[str, Any],
    prepared_blob: _PreparedBlobCreate,
) -> Mapping[str, Any]:
    """Ensure LLM-authored inline source blobs carry a Class 2 review gate."""
    return _options_with_source_blob_review(
        options,
        mime_type=prepared_blob.mime_type,
        content=prepared_blob.content_bytes.decode("utf-8"),
    )


@trust_boundary(
    tier=3,
    source="LLM composer tool-call arguments",
    source_param="args",
    suppresses=("R1", "R5"),
    invariant="raises ToolArgumentError on SetPipelineArgumentsModel shape mismatch; never coerces",
    test_ref="tests/unit/web/composer/test_promote_set_pipeline.py::TestPromoteSetPipelineArgErrorRouting::test_empty_arguments_raise_tool_argument_error",
    test_fingerprint="02c5bd7c9f9aa90bd1af5d67ac7d60764bfc0306de78c6216ee84a5d905d362b",
)
def _execute_set_pipeline(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Atomically replace the entire pipeline composition state.

    Tier-3 boundary: ``args`` is an LLM-supplied dict.  Validated via the
    Pydantic redaction-bearing model :class:`SetPipelineArgumentsModel` —
    the single source of truth for the argument schema, superseding the
    deleted ``_TOOL_REQUIRED_PATHS["set_pipeline"]`` entry in
    ``service.py``.

    On :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing at
    ``service.py:2480`` receives the right exception class.

    Dispatcher-wired kwargs
    -----------------------
    The dispatcher at :func:`execute_tool` (``tools.py:5530-5540``) supplies
    ``session_engine`` and ``session_id`` as kwargs.  These are NOT part of
    the LLM-supplied ``arguments`` dict — they are wired from the composer
    service request context — so they are NOT modelled in
    :class:`SetPipelineArgumentsModel`.  The Pydantic model validates only
    the LLM-supplied dict; the kwargs enter through the function signature.

    Semantic vs argument-shape failures
    ------------------------------------
    Pydantic enforces argument shape (type, required-fields, extra=forbid).
    Per-component semantic checks (plugin existence in catalog, path
    allowlist, manual blob_ref injection rejection, source.blob_id +
    source.inline_blob exclusivity, gate-condition expression validity)
    remain in this handler and produce recoverable ``_failure_result``
    responses with repair hints.  Two channels for two failure shapes
    (type vs semantic) — same pattern as
    :class:`SetSourceArgumentsModel` plugin-not-in-catalog handling.
    """
    catalog = context.catalog
    data_dir = context.data_dir
    session_engine = context.session_engine
    session_id = context.session_id
    user_message_id = context.user_message_id
    max_blob_storage_per_session_bytes = context.max_blob_storage_per_session_bytes

    try:
        validated = SetPipelineArgumentsModel.model_validate(args)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="set_pipeline arguments",
            expected="object conforming to SetPipelineArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc

    if validated.source is not None and validated.sources is not None:
        return _failure_result(state, "set_pipeline must use either source or sources, not both.")
    if validated.source is None and validated.sources is None:
        return _failure_result(state, "set_pipeline requires source or sources.")

    source_specs: dict[str, SourceSpec] = {}
    prepared_inline_blob: _PreparedBlobCreate | None = None
    resolved_source_blob: _ResolvedSourceBlob | None = None
    single_source_on_vf: str | None = None

    if validated.sources is not None:
        if not validated.sources:
            return _failure_result(state, "set_pipeline sources must include at least one named source.")
        for source_name, source_model in validated.sources.items():
            if not source_name.strip():
                return _failure_result(state, "set_pipeline sources keys must be non-empty source names.")
            if source_model.blob_id is not None or source_model.inline_blob is not None:
                return _failure_result(
                    state,
                    f"set_pipeline sources.{source_name} cannot use blob_id or inline_blob in v1. "
                    "Bind blob-backed sources with set_source_from_blob, or use source for a single blob-backed pipeline.",
                )
            src_plugin = source_model.plugin
            plugin_error = _validate_plugin_name(catalog, "source", src_plugin)
            if plugin_error is not None:
                return _failure_result(state, f"Source '{source_name}': {plugin_error}")
            src_options = dict(source_model.options)
            manual_blob_ref_error = _reject_manual_source_blob_ref(src_options, tool_name="set_pipeline")
            if manual_blob_ref_error is not None:
                return _failure_result(state, f"Source '{source_name}': {manual_blob_ref_error}")
            manual_authoring_error = _reject_manual_source_authoring(src_options, tool_name="set_pipeline")
            if manual_authoring_error is not None:
                return _failure_result(state, f"Source '{source_name}': {manual_authoring_error}")
            review_metadata_error = _resolver_owned_interpretation_requirement_error(src_options, tool_name="set_pipeline")
            if review_metadata_error is not None:
                return _failure_result(state, f"Source '{source_name}': {review_metadata_error}")
            credential_error = _credential_wiring_contract_failure(
                state,
                component_id=_source_component_id(source_name),
                component_type="source",
                plugin_type="source",
                plugin_name=src_plugin,
                options=src_options,
            )
            if credential_error is not None:
                return credential_error
            src_on_vf = source_model.on_validation_failure or _DEFAULT_SOURCE_VALIDATION_FAILURE
            path_error = _validate_source_path(src_options, data_dir)
            if path_error is not None:
                return _failure_result(state, f"Source '{source_name}': {path_error}")
            src_prevalidation = _prevalidate_source(src_plugin, src_options, src_on_vf)
            if src_prevalidation is not None:
                return _failure_result(state, f"Source '{source_name}': {src_prevalidation}")
            source_specs[source_name] = SourceSpec(
                plugin=src_plugin,
                on_success=source_model.on_success,
                options=src_options,
                on_validation_failure=src_on_vf,
            )
    else:
        legacy_source_model = validated.source
        if legacy_source_model is None:
            raise AssertionError("validated.source unexpectedly None after source/sources gate")
        src_plugin = legacy_source_model.plugin
        plugin_error = _validate_plugin_name(catalog, "source", src_plugin)
        if plugin_error is not None:
            return _failure_result(state, plugin_error)

        # Inline user-provided source data can be materialised as a blob inside
        # this same atomic pipeline mutation. The generated path/blob_ref are
        # authoritative exactly as if create_blob + set_source_from_blob had been
        # called, but the LLM gets one audited tool decision instead of a serial
        # blob-then-source-then-pipeline conversation.
        legacy_src_options: Mapping[str, Any] = dict(legacy_source_model.options)
        manual_blob_ref_error = _reject_manual_source_blob_ref(
            legacy_src_options,
            tool_name="set_pipeline",
            inline_blob_supported=True,
        )
        if manual_blob_ref_error is not None:
            return _failure_result(state, manual_blob_ref_error)
        manual_authoring_error = _reject_manual_source_authoring(legacy_src_options, tool_name="set_pipeline")
        if manual_authoring_error is not None:
            return _failure_result(state, manual_authoring_error)
        review_metadata_error = _resolver_owned_interpretation_requirement_error(legacy_src_options, tool_name="set_pipeline")
        if review_metadata_error is not None:
            return _failure_result(state, review_metadata_error)
        credential_error = _credential_wiring_contract_failure(
            state,
            component_id="source",
            component_type="source",
            plugin_type="source",
            plugin_name=src_plugin,
            options=legacy_src_options,
        )
        if credential_error is not None:
            return credential_error
        source_blob_id = legacy_source_model.blob_id
        inline_blob = legacy_source_model.inline_blob
        src_on_vf = legacy_source_model.on_validation_failure or _DEFAULT_SOURCE_VALIDATION_FAILURE
        single_source_on_vf = src_on_vf
        if source_blob_id is not None and inline_blob is not None:
            return _failure_result(state, "set_pipeline source must use either an existing blob_id or inline_blob, not both.")
        if source_blob_id is not None:
            resolved = _resolve_source_blob(
                blob_id=source_blob_id,
                explicit_plugin=src_plugin,
                caller_options=legacy_src_options,
                on_validation_failure=src_on_vf,
                state=state,
                catalog=catalog,
                session_engine=session_engine,
                session_id=session_id,
                tool_name="set_pipeline",
            )
            if isinstance(resolved, ToolResult):
                return resolved
            resolved_source_blob = resolved
            src_plugin = resolved.plugin
            legacy_src_options = resolved.options
        if inline_blob is not None:
            if session_engine is None or session_id is None:
                return _failure_result(state, "set_pipeline source.inline_blob requires session context.")
            if data_dir is None:
                return _failure_result(state, "set_pipeline source.inline_blob requires data_dir for storage.")
            # _prepare_blob_create raises ToolArgumentError on invalid LLM
            # arguments (CEC1 channel discipline) — propagate to the
            # compose loop's ARG_ERROR branch rather than masking as
            # SUCCESS-with-success=False. The inline_blob contents are already
            # type-validated by ``_InlineBlobModel``
            # (str/str/str + extra=forbid), so the isinstance guards inside
            # _prepare_blob_create are unreachable from this caller — see
            # the cleanup that removes them.
            provenance = _blob_creation_provenance(inline_blob.content, context)
            prepared_inline_blob = _prepare_blob_create(
                inline_blob.model_dump(),
                data_dir=data_dir,
                session_id=session_id,
                creation_modality=provenance.creation_modality,
                created_from_message_id=user_message_id,
                creating_model_identifier=provenance.creating_model_identifier,
                creating_model_version=provenance.creating_model_version,
                creating_provider=provenance.creating_provider,
                creating_composer_skill_hash=provenance.creating_composer_skill_hash,
                creating_arguments_hash=provenance.creating_arguments_hash,
            )
            header_conflict = _header_only_inline_csv_conflict(
                prepared_inline_blob,
                session_engine=session_engine,
                session_id=session_id,
            )
            if header_conflict is not None:
                return _failure_result(state, header_conflict)

            # ``prepared_inline_blob.mime_type`` was validated by
            # ``_prepare_blob_create`` against ``_ALLOWED_BLOB_MIME_TYPES``,
            # which is the exact key set of ``_MIME_TO_SOURCE``. A KeyError
            # here means those constants drifted, not a recoverable LLM input
            # condition.
            inferred_plugin, inferred_options = _MIME_TO_SOURCE[prepared_inline_blob.mime_type]
            mime_options: dict[str, str] = inferred_options if inferred_plugin == src_plugin else {}
            # A ``.tsv`` inline blob (uploaded as text/csv) must bind a tab
            # delimiter; ``_MIME_TO_SOURCE`` is mime-keyed and cannot express it.
            # Derive it from the filename like ``inspect_blob_content`` does, gated
            # on the plugin actually being bound and not clobbering a caller value.
            delimiter_options = _delimiter_extra_for_csv_blob(
                src_plugin,
                prepared_inline_blob.filename,
                legacy_src_options,
            )
            legacy_src_options = {
                **legacy_src_options,
                **mime_options,
                **delimiter_options,
                "path": str(prepared_inline_blob.storage_path),
                "blob_ref": prepared_inline_blob.blob_id,
                **_source_authoring_options(prepared_inline_blob.creation_modality, prepared_inline_blob.content_hash),
            }
            legacy_src_options = _options_with_inline_blob_source_review(legacy_src_options, prepared_inline_blob)

        path_error = _validate_source_path(legacy_src_options, data_dir)
        if path_error is not None:
            return _failure_result(state, path_error)

        src_prevalidation = _prevalidate_source(src_plugin, legacy_src_options, src_on_vf)
        if src_prevalidation is not None:
            return _failure_result(state, src_prevalidation)
        source_specs["source"] = SourceSpec(
            plugin=src_plugin,
            on_success=legacy_source_model.on_success,
            options=legacy_src_options,
            on_validation_failure=src_on_vf,
        )

    # 2. Validate node plugins and options
    for node in validated.nodes:
        node_id = node.id
        node_type = node.node_type
        node_plugin = node.plugin
        node_options = node.options
        runtime_owned_error = _runtime_owned_llm_option_error(
            node_plugin,
            node_options,
            tool_name="set_pipeline",
        )
        if runtime_owned_error is not None:
            return _failure_result(state, f"Node '{node_id}': {runtime_owned_error}")
        credential_error = _credential_wiring_contract_failure(
            state,
            component_id=node_id,
            component_type="node",
            plugin_type="transform" if node_plugin is not None else None,
            plugin_name=node_plugin,
            options=node_options,
        )
        if credential_error is not None:
            return credential_error
        if node_type in ("transform", "aggregation") and node_plugin is not None:
            plugin_error = _validate_plugin_name(catalog, "transform", node_plugin)
            if plugin_error is not None:
                return _failure_result(state, f"Node '{node_id}': {plugin_error}")
            batch_placement_error = _batch_aware_placement_error(node_id, node_type, node_plugin, node.output_mode)
            if batch_placement_error is not None:
                return _failure_result(state, f"Node '{node_id}': {batch_placement_error}")
            batch_required_error = _batch_aware_required_input_fields_error(node_id, node_plugin, node_options)
            if batch_required_error is not None:
                return _failure_result(state, f"Node '{node_id}': {batch_required_error}")

            review_options = _options_with_default_llm_reviews(
                node_id=node_id,
                plugin=node_plugin,
                options=node_options,
            )
            node_prevalidation = _prevalidate_transform(node_plugin, review_options)
            if node_prevalidation is not None:
                return _failure_result(state, f"Node '{node_id}': {node_prevalidation}")

            provider_policy_error = _validate_transform_provider_config_policy(node_options, plugin=node_plugin)
            if provider_policy_error is not None:
                return _failure_result(state, f"Node '{node_id}': {provider_policy_error}")

            # S2: confine nested provider_config persist_directory (RAG
            # retrieval). Parity with the per-output sink-path check below so
            # a bulk set_pipeline cannot wave through an escaping transform
            # path while rejecting an escaping sink path.
            provider_path_error = _validate_transform_provider_config_path(node_options, data_dir)
            if provider_path_error is not None:
                return _failure_result(state, f"Node '{node_id}': {provider_path_error}")

        # Validate gate condition expression at composition time.
        if node_type == "gate" and node.condition is not None:
            expr_error = _validate_gate_expression(node.condition)
            if expr_error is not None:
                return _failure_result(state, f"Node '{node_id}': {expr_error}")
            parity_error = _validate_gate_route_parity(node.condition, node.routes)
            if parity_error is not None:
                return _failure_result(state, f"Node '{node_id}': {parity_error}")

    # 3. Validate output plugins and options
    #
    # ``options_missing`` distinguishes "operator omitted the options key
    # entirely" from "operator supplied options: {}".  Post-Pydantic the
    # default-factory replaces an absent key with ``{}`` on the model side,
    # so we look at the raw ``args`` dict to recover the operator's
    # original intent for the repair-hint branch.  The semantic-validation
    # branch (file-sink collision policy, path allowlist) still runs on
    # the validated dict.
    # ``args`` is typed ``dict[str, Any]`` per the function signature
    # (raw LLM tool-call payload). The Tier-3 LLM-boundary read is
    # ``args.get("outputs")`` — the value may be absent, present-as-list,
    # or present-as-some-other-shape; downstream isinstance checks
    # narrow the value. A pre-isinstance on ``args`` itself is
    # redundant (the type system already proves it is a Mapping).
    raw_outputs = args.get("outputs")
    for index, output in enumerate(validated.outputs):
        out_name = output.sink_name
        out_plugin = output.plugin
        plugin_error = _validate_plugin_name(catalog, "sink", out_plugin)
        if plugin_error is not None:
            return _failure_result(state, f"Output '{out_name}': {plugin_error}")
        out_options = output.options
        raw_out_args: Mapping[str, Any] = {}
        if isinstance(raw_outputs, list) and 0 <= index < len(raw_outputs):
            raw_entry = raw_outputs[index]
            if isinstance(raw_entry, Mapping):
                raw_out_args = raw_entry
        options_missing = "options" not in raw_out_args
        if options_missing:
            out_prevalidation = _prevalidate_sink(out_plugin, out_options)
            out_collision_error = validate_composer_file_sink_collision_policy(
                out_plugin,
                out_options,
                require_explicit=data_dir is not None,
            )
            validation_error = out_prevalidation if out_prevalidation is not None else out_collision_error
            if validation_error is not None:
                return _failure_result(
                    state,
                    _missing_output_options_repair_error(
                        sink_name=out_name,
                        plugin_name=out_plugin,
                        on_write_failure=output.on_write_failure if output.on_write_failure is not None else "discard",
                        validation_error=validation_error,
                    ),
                )
        credential_error = _credential_wiring_contract_failure(
            state,
            component_id=out_name,
            component_type="output",
            plugin_type="sink",
            plugin_name=out_plugin,
            options=out_options,
        )
        if credential_error is not None:
            return credential_error
        out_path_error = _validate_sink_path(out_options, data_dir)
        if out_path_error is not None:
            return _failure_result(state, f"Output '{out_name}': {out_path_error}")
        out_prevalidation = _prevalidate_sink(out_plugin, out_options)
        if out_prevalidation is not None:
            return _failure_result(state, f"Output '{out_name}': {out_prevalidation}")
        out_collision_error = validate_composer_file_sink_collision_policy(
            out_plugin,
            out_options,
            require_explicit=data_dir is not None,
        )
        if out_collision_error is not None:
            return _failure_result(state, f"Output '{out_name}': {out_collision_error}")

    # 4. Construct specs (same field extraction as individual handlers)
    # ``node_type`` / ``edge_type`` are typed as ``str`` on
    # ``_PipelineNodeModel`` / ``_PipelineEdgeModel`` to preserve Tier-3
    # LLM-recoverable feedback (the handler reports unknown enum values
    # via semantic _failure_result; a Pydantic Literal rejection would
    # surface as ARG_ERROR with no repair guidance).  At the point we
    # construct :class:`NodeSpec` / :class:`EdgeSpec`, the downstream
    # validation (``_validate_plugin_name``, graph topology) and the
    # ``_batch_aware_placement_error`` checks above have not yet
    # rejected non-canonical enum values explicitly — that responsibility
    # remains on the state-level dataclass invariants.  The ``cast`` here
    # narrows the static type without re-validating; semantically wrong
    # enum values flow through to be rejected at runtime by NodeSpec /
    # EdgeSpec / CompositionState invariants.
    node_specs = []
    for n in validated.nodes:
        fork_to = tuple(n.fork_to) if n.fork_to is not None else None
        branches = dict(n.branches) if isinstance(n.branches, Mapping) else tuple(n.branches) if n.branches is not None else None
        nt = n.node_type
        node_specs.append(
            NodeSpec(
                id=n.id,
                node_type=cast(NodeType, nt),
                plugin=n.plugin,
                input=n.input,
                on_success=n.on_success,
                on_error=n.on_error or ("discard" if nt in ("transform", "aggregation") else None),
                options=_options_with_default_llm_reviews(
                    node_id=n.id,
                    plugin=n.plugin,
                    options=n.options,
                ),
                condition=n.condition,
                routes=n.routes,
                fork_to=fork_to,
                branches=branches,
                policy=n.policy,
                merge=n.merge,
                # ``n.trigger`` is a typed :class:`_NodeTriggerModel` (or None) per F3 —
                # convert to a plain dict at the NodeSpec boundary because
                # :class:`NodeSpec.trigger` is typed ``Mapping[str, Any] | None`` and
                # is deep-frozen by ``freeze_fields("trigger")``; the freeze contract
                # requires a Mapping, not a Pydantic model instance.
                trigger=n.trigger.model_dump() if n.trigger is not None else None,
                output_mode=n.output_mode,
                expected_output_count=n.expected_output_count,
            )
        )

    edge_specs = []
    for e in validated.edges:
        edge_specs.append(
            EdgeSpec(
                id=e.id,
                from_node=e.from_node,
                to_node=e.to_node,
                edge_type=cast(EdgeType, e.edge_type),
                label=e.label,
            )
        )

    output_specs = []
    for o in validated.outputs:
        output_specs.append(
            OutputSpec(
                name=o.sink_name,
                plugin=o.plugin,
                options=o.options,
                on_write_failure=o.on_write_failure if o.on_write_failure is not None else "discard",
            )
        )

    # PipelineMetadata's __init__ supplies its own defaults for ``name`` and
    # ``description``; honour those by passing through only explicitly
    # supplied fields.  ``validated.metadata`` is None when the LLM omitted
    # the ``metadata`` key entirely.
    meta_kwargs: dict[str, str] = {}
    if validated.metadata is not None:
        if validated.metadata.name is not None:
            meta_kwargs["name"] = validated.metadata.name
        if validated.metadata.description is not None:
            meta_kwargs["description"] = validated.metadata.description
    metadata_spec = PipelineMetadata(**meta_kwargs)

    # 5. Build new state
    new_state = CompositionState(
        sources=source_specs,
        nodes=tuple(node_specs),
        edges=tuple(edge_specs),
        outputs=tuple(output_specs),
        metadata=metadata_spec,
        version=state.version + 1,
    )
    review_contract_error = composition_review_contract_error(new_state)
    if review_contract_error is not None:
        return _failure_result(state, review_contract_error)

    if prepared_inline_blob is not None:
        if session_engine is None or session_id is None:
            return _failure_result(state, "set_pipeline source.inline_blob requires session context.")
        quota_error = _persist_prepared_blob_create(
            prepared_inline_blob,
            session_engine=session_engine,
            session_id=session_id,
            max_blob_storage_per_session_bytes=max_blob_storage_per_session_bytes,
        )
        if quota_error is not None:
            return _failure_result(state, quota_error)

    # 6. Report all nodes + sources + outputs as affected
    affected = (*(_source_component_id(name) for name in source_specs), *(n.id for n in node_specs), *(o.name for o in output_specs))
    data: dict[str, Any] | None = _vf_destination_note(new_state, single_source_on_vf) if single_source_on_vf is not None else None
    if resolved_source_blob is not None:
        source_blob_payload = {"source_blob": resolved_source_blob.payload}
        data = source_blob_payload if data is None else {**data, **source_blob_payload}
    if prepared_inline_blob is not None:
        inline_payload = {"inline_blob": _blob_create_payload(prepared_inline_blob)}
        data = inline_payload if data is None else {**data, **inline_payload}
    return _mutation_result(
        new_state,
        affected,
        data=data,
    )


def _execute_apply_pipeline_recipe(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Validate a recipe's slots, build set_pipeline args, and dispatch to set_pipeline.

    Tier-3 boundary: ``arguments`` is an LLM-supplied dict.  Validated
    via :class:`ApplyPipelineRecipeArgumentsModel` — the single source
    of truth for the argument schema, superseding the deleted
    ``_TOOL_REQUIRED_PATHS["apply_pipeline_recipe"]`` entry in
    ``service.py``.  On
    :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing
    at ``service.py:2480`` receives the right exception class.

    Semantic vs argument-shape failures
    ------------------------------------
    Pydantic enforces argument shape (type, required-fields, extra=forbid).
    The empty-``recipe_name`` semantic check and the
    :class:`RecipeValidationError` slot-shape check remain in this handler
    and produce recoverable ``_failure_result`` responses with repair
    hints (``Call list_recipes to discover available recipes``).  Two
    channels for two failure shapes (type vs semantic) — same pattern as
    :class:`SetSourceArgumentsModel` plugin-not-in-catalog handling.

    ``set_pipeline`` is full state replacement, so a ``replaced_pipeline_note`` is
    emitted to make the destructive replacement visible to the LLM/operator. The
    note is suppressed when the prior pipeline is empty — a no-op replacement
    needs no flag, and emitting one would be noise on a fresh-session apply.

    Co-located with :func:`_execute_set_pipeline` (rather than living in a
    sibling ``tools/recipes.py``) because the recipe-application handler
    delegates the destructive state-replacement to ``_execute_set_pipeline``
    on the success path. Keeping them in the same plane closes the
    cross-plane private-access edge the slice 1-5 refactor introduced.
    """
    try:
        validated = ApplyPipelineRecipeArgumentsModel.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="apply_pipeline_recipe arguments",
            expected="object conforming to ApplyPipelineRecipeArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc

    recipe_name = validated.recipe_name
    raw_slots = validated.slots
    if not recipe_name:
        # Empty-string recipe_name passes Pydantic's ``str`` validation
        # but the handler treats it as a recoverable semantic failure
        # with a repair-hint pointing the LLM at list_recipes (rather
        # than the generic ARG_ERROR envelope a Pydantic min_length=1
        # would produce).
        return _failure_result(
            state,
            "apply_pipeline_recipe requires a non-empty 'recipe_name' string. Call list_recipes to discover available recipes.",
        )

    try:
        pipeline_args = apply_recipe(recipe_name, dict(raw_slots))
    except RecipeValidationError as exc:
        return _failure_result(state, str(exc))

    # Capture pre-replacement counts BEFORE delegating to the destructive
    # set_pipeline path. Frozen-dataclass fields, so capturing the integers
    # now is sufficient — the post-call result.updated_state is a fresh
    # CompositionState produced by set_pipeline.
    pre_source_present = bool(state.sources)
    pre_node_count = len(state.nodes)
    pre_output_count = len(state.outputs)

    # Delegate to the existing set_pipeline executor — recipes produce the
    # exact arguments shape set_pipeline accepts, so validation and state
    # mutation flow through the canonical mutation path.
    result = _execute_set_pipeline(pipeline_args, state, context)

    # Only annotate successful replacements over a non-empty prior state.
    # On failure, set_pipeline returned ``state`` unchanged and the note
    # would be misleading. On a fresh-session apply, there is nothing to
    # call out and a note would be noise.
    if not result.success:
        return result
    if not (pre_source_present or pre_node_count or pre_output_count):
        return result

    note = (
        f"apply_pipeline_recipe replaced the existing pipeline "
        f"(prior state had source={'set' if pre_source_present else 'unset'}, "
        f"{pre_node_count} node(s), {pre_output_count} output(s)). "
        "Recipes are full-state scaffolds; the prior composition was discarded."
    )

    # Preserve any existing data payload from set_pipeline (e.g. inline-blob
    # creation summary) by merging into a single dict. set_pipeline's
    # ``data`` is currently None on the recipe path because recipes don't
    # use inline_blob, but merging is forward-compatible.
    #
    # ``_execute_set_pipeline`` declares ``data: dict[str, Any] | None``
    # (see the local annotation at the success-path construction site).
    # That contract is system code, not Tier-3 LLM data, so the value is
    # ``None`` or a ``dict`` — we authored it. We access it directly: the
    # dict-spread on the non-None path crashes naturally (``TypeError``) if a
    # future return path violates the contract, which is the correct offensive
    # response to a system-code bug — not a silent wrap into a payload the LLM
    # would treat as valid.
    existing_data = result.data
    if existing_data is None:
        merged_data: dict[str, Any] = {"replaced_pipeline_note": note}
    else:
        merged_data = {**existing_data, "replaced_pipeline_note": note}

    return replace(result, data=merged_data)


_APPLY_PIPELINE_RECIPE_DECLARATION = ToolDeclaration(
    name="apply_pipeline_recipe",
    handler=_execute_apply_pipeline_recipe,
    kind=ToolKind.MUTATION,
    description=(
        "Apply a registered pipeline recipe with operator-supplied slot values and replace "
        "the current pipeline state with the resulting configuration. Slots are validated "
        "against the recipe's declared schema before scaffolding — invalid slots are "
        "rejected with a repair hint. Call list_recipes to discover available recipes and "
        "their slot schemas. The resulting state is identical to a hand-authored "
        "set_pipeline call; the model can refine via patch_*_options afterwards."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "recipe_name": {
                "type": "string",
                "description": "Recipe identifier (e.g., 'classify-rows-llm-jsonl')",
            },
            "slots": {
                "type": "object",
                "description": "Operator-supplied slot values; must match the recipe's slot schema",
            },
        },
        "required": ["recipe_name", "slots"],
        "additionalProperties": False,
    },
    augments_on_failure=True,
)


def _handle_set_pipeline(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _execute_set_pipeline(arguments, state, context)


_SET_PIPELINE_DECLARATION = ToolDeclaration(
    name="set_pipeline",
    handler=_handle_set_pipeline,
    kind=ToolKind.MUTATION,
    description="Atomically replace the entire pipeline. Provide the "
    "complete source, nodes, edges, outputs, and metadata in one call. "
    "This is more efficient than calling set_source + upsert_node + "
    "upsert_edge + set_output sequentially.",
    json_schema={
        "type": "object",
        "properties": {
            "source": {
                "type": "object",
                "description": (
                    "Source configuration: {plugin, on_success, options?, on_validation_failure?, blob_id?, inline_blob?}. "
                    "Use blob_id to bind an already uploaded session blob, or inline_blob to "
                    "materialize user-provided literal data while atomically setting the full pipeline."
                ),
                "properties": {
                    "plugin": {"type": "string"},
                    "blob_id": {
                        "type": "string",
                        "description": (
                            "Existing ready session blob ID to bind as this source. "
                            "The tool resolves path/blob_ref authoritatively exactly like set_source_from_blob."
                        ),
                    },
                    "options": {
                        "type": "object",
                        "description": (
                            "Plugin-specific source config. Required by most file/data sources even though "
                            "the schema leaves it optional so the handler can return plugin-specific repair "
                            "feedback instead of a generic missing-argument error."
                        ),
                    },
                    "on_success": {
                        "type": "string",
                        "description": (
                            "Connection-name string the source PUBLISHES. Some downstream "
                            "consumer (node 'input' or output 'sink_name') MUST equal this. "
                            "Connections match by string, not by node id."
                        ),
                        "examples": ["raw_url_rows", "csv_rows", "fetched_text"],
                    },
                    "on_validation_failure": {
                        "type": "string",
                        "description": _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
                    },
                    "inline_blob": {
                        "type": "object",
                        "description": (
                            "Optional inline source content to create as a session blob before binding the source. "
                            "Fields mirror create_blob: filename, mime_type, content, and optional description."
                        ),
                        "properties": {
                            "filename": {"type": "string"},
                            "mime_type": {
                                "type": "string",
                                "enum": [
                                    "text/plain",
                                    "application/json",
                                    "text/csv",
                                    "application/x-jsonlines",
                                    "application/jsonl",
                                    "text/jsonl",
                                ],
                            },
                            "content": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["filename", "mime_type", "content"],
                    },
                },
                "required": ["plugin", "on_success"],
            },
            "sources": {
                "type": "object",
                "description": (
                    "Named source roots keyed by stable source name. Use this instead of source for multi-source pipelines. "
                    "Each value has the same shape as source, but blob_id and inline_blob are only supported on the legacy source field in v1."
                ),
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "plugin": {"type": "string"},
                        "options": {"type": "object"},
                        "on_success": {"type": "string"},
                        "on_validation_failure": {
                            "type": "string",
                            "description": _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
                        },
                    },
                    "required": ["plugin", "on_success"],
                },
            },
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "node_type": {"type": "string"},
                        "plugin": {"type": "string"},
                        "input": {
                            "type": "string",
                            "description": (
                                "Connection-name string this node CONSUMES. MUST equal some "
                                "upstream's on_success/routes value/on_error. NOT the upstream "
                                "node's id. If source.on_success='raw_url_rows', this node sets "
                                "input='raw_url_rows'."
                            ),
                            "examples": ["raw_url_rows", "fetched_text", "scored_rows"],
                        },
                        "on_success": {
                            "type": "string",
                            "description": (
                                "Connection-name string this node PUBLISHES (transform/aggregation/"
                                "coalesce). Some downstream input/sink_name MUST equal this. Omit "
                                "for gates (routing is via condition+routes)."
                            ),
                            "examples": ["fetched_text", "scored_rows", "lines_out"],
                        },
                        "on_error": {"type": "string"},
                        "options": {"type": "object"},
                        "condition": {"type": "string"},
                        "routes": {
                            "type": "object",
                            "description": (
                                "Gate route mapping to sink names, downstream connection names, 'fork', or "
                                "'discard' for an audited terminal drop."
                            ),
                        },
                        "fork_to": {"type": "array", "items": {"type": "string"}},
                        "branches": {
                            "type": ["array", "object"],
                            "items": {"type": "string"},
                            "additionalProperties": {"type": "string"},
                        },
                        "policy": {"type": "string"},
                        "merge": {"type": "string"},
                        "trigger": {"type": "object"},
                        "output_mode": {"type": "string"},
                        "expected_output_count": {"type": "integer"},
                    },
                    "required": ["id", "node_type", "input"],
                },
                "description": "Array of node specs: [{id, input, plugin?, node_type, options?, on_success?, on_error?, condition?, routes?, fork_to?, branches?, policy?, merge?, trigger?, output_mode?, expected_output_count?}]",
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "from_node": {"type": "string"},
                        "to_node": {"type": "string"},
                        "edge_type": {"type": "string"},
                        "label": {"type": ["string", "null"]},
                    },
                    "required": ["id", "from_node", "to_node", "edge_type"],
                },
                "description": "Array of edge specs: [{id, from_node, to_node, edge_type}]",
            },
            "outputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sink_name": {
                            "type": "string",
                            "description": (
                                "Sink name. BOTH the sink's identifier AND the connection-name "
                                "the sink consumes — it MUST equal some upstream's on_success "
                                "value. Pick a descriptive name; it does not need to match an "
                                "upstream node's id."
                            ),
                            "examples": ["lines_out", "scored_results", "errors_quarantine"],
                        },
                        "plugin": {"type": "string"},
                        "options": {
                            "type": "object",
                            "description": (
                                "Plugin-specific sink config. For csv/json file sinks in runnable web "
                                "pipelines, include path, schema, and explicit collision_policy."
                            ),
                        },
                        "on_write_failure": {"type": "string"},
                    },
                    "required": ["sink_name", "plugin"],
                    "examples": [
                        {
                            "sink_name": "results",
                            "plugin": "json",
                            "options": {
                                "path": "outputs/results.json",
                                "schema": {"mode": "observed"},
                                "mode": "write",
                                "collision_policy": "auto_increment",
                            },
                            "on_write_failure": "discard",
                        }
                    ],
                },
                "description": (
                    "Array of output specs: [{sink_name, plugin, options, on_write_failure?}]. "
                    "For csv/json file sinks in runnable web pipelines, options must include "
                    "path, schema, explicit mode ('write' or 'append'), and explicit collision_policy."
                ),
            },
            "metadata": {
                "type": "object",
                "description": "Pipeline metadata: {name?, description?}",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "required": ["nodes", "edges", "outputs"],
        "additionalProperties": False,
    },
    augments_on_failure=True,
)


def _is_full_state_component_alias(component: Any) -> bool:
    """Return whether a component argument explicitly requests full state."""
    return isinstance(component, str) and component.strip().lower() in _FULL_STATE_COMPONENT_ALIAS_SET


def _execute_get_pipeline_state(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Return full pipeline state including all options.

    If ``component`` is specified, returns only that component's details.
    Otherwise returns the full state: source, all nodes with options, all
    outputs with options, edges, and metadata.
    """
    del context  # unused; signature uniformity with the other handlers.
    component = args.get("component")

    if component == "source":
        data: Any = {"sources": {name: _serialize_source(source) for name, source in state.sources.items()}}
    elif component is not None:
        # Try node, then output
        node = next((n for n in state.nodes if n.id == component), None)
        if node is not None:
            data = {"node": _serialize_node(node)}
        else:
            output = next((o for o in state.outputs if o.name == component), None)
            if output is not None:
                data = {"output": _serialize_output(output)}
            elif _is_full_state_component_alias(component):
                data = _serialize_full_pipeline_state(state, requested_component=component)
            else:
                return _failure_result(
                    state,
                    f"Component '{component}' not found. Specify 'source', a node ID, an output name, "
                    "or a full-state alias ('full', 'all', 'pipeline', or empty string).",
                )
    else:
        data = _serialize_full_pipeline_state(state, requested_component=None)

    data = redact_source_storage_path(data)
    return _discovery_result(state, data)


_GET_PIPELINE_STATE_DECLARATION = ToolDeclaration(
    name="get_pipeline_state",
    handler=_execute_get_pipeline_state,
    kind=ToolKind.DISCOVERY,
    description="Inspect the full current pipeline state including all "
    "options for source, nodes, and outputs. Use this during correction "
    "loops to see what is currently configured before patching.",
    json_schema={
        "type": "object",
        "properties": {
            "component": {
                "type": "string",
                "description": (
                    "Optional: return only one component — 'source', a node ID, or an output name. "
                    "Accepted full-state aliases: omit component, pass 'full', 'all', 'pipeline', "
                    "or pass the empty string."
                ),
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    cacheable=False,
)


def _authoring_validation_payload(state: CompositionState, validation: ValidationSummary) -> dict[str, Any]:
    return {
        "is_valid": validation.is_valid,
        "errors": [e.to_dict() for e in validation.errors],
        "warnings": [e.to_dict() for e in validation.warnings],
        "suggestions": [e.to_dict() for e in validation.suggestions],
        "edge_contracts": [ec.to_dict() for ec in validation.edge_contracts],
        "semantic_contracts": _semantic_contracts_payload(validation.semantic_contracts),
        "graph_repair_suggestions": _graph_repair_suggestions(state, validation),
    }


def _find_node_or_raise(state: CompositionState, affected_node_id: str) -> NodeSpec:
    node = next((n for n in state.nodes if n.id == affected_node_id), None)
    if node is None:
        known = sorted(n.id for n in state.nodes)
        raise ToolArgumentError(
            argument="affected_node_id",
            expected=f"id of an existing LLM transform (known ids: {known!r})",
            actual_type=f"unknown id {affected_node_id!r}",
        )
    return node


def _matching_interpretation_sites(
    state: CompositionState,
    affected_node_id: str,
    kind: InterpretationKind,
    user_term: str,
) -> list[str]:
    normalized_user_term = user_term.strip()
    try:
        sites = interpretation_sites(state)
    except (TypeError, ValueError) as exc:
        raise ToolArgumentError(
            argument="affected_node_id",
            expected="well-formed interpretation authoring metadata",
            actual_type=f"invalid interpretation metadata: {exc}",
        ) from exc
    return [
        site.user_term
        for site in sites
        if site.component_id == affected_node_id and site.kind is kind and site.user_term == normalized_user_term
    ]


@trust_boundary(
    tier=3,
    source="LLM composer tool-call state (composer/LLM-authored composition state re-read from session storage)",
    source_param="state",
    suppresses=("R1", "R5"),
    invariant="raises ToolArgumentError on review-component shape/semantic mismatch; never coerces",
    test_ref="tests/unit/web/composer/test_request_interpretation_review_tool.py::test_04_wrong_kind_node_raises",
    test_fingerprint="bb71cdbf2a60eb1348ab865028d80f2c1303d84495f82b5eb2572e170a96ef1d",
)
def _assert_affected_component(
    state: CompositionState,
    affected_node_id: str,
    kind: InterpretationKind,
    user_term: str,
    llm_draft: str | None = None,
) -> None:
    """Tier-3 boundary check on the LLM-supplied review component.

    Raises :class:`ToolArgumentError` with an actionable message when:

    * ``invented_source`` does not target the source component, or the
      source lacks composer-authored source metadata;
    * prompt and vague-term kinds do not target an existing LLM node;
    * pipeline decisions do not target an existing node with matching review
      metadata;
    * pending review sites do not match the structured authoring metadata
      (with legacy placeholder fallback for vague terms).

    Each branch raises ARG_ERROR (not a Tier-1 crash) because the LLM can
    recover by staging the right source/node metadata and retrying the tool
    call. Term matching remains strict after stripping surrounding whitespace.
    """
    if kind is InterpretationKind.INVENTED_SOURCE:
        if affected_node_id != SOURCE_COMPONENT_ID:
            raise ToolArgumentError(
                argument="affected_node_id",
                expected="'source' for invented_source",
                actual_type="node id",
            )
        source = state.sources.get(SOURCE_COMPONENT_ID)
        if source is None or SOURCE_AUTHORING_KEY not in source.options:
            raise ToolArgumentError(
                argument="affected_node_id",
                expected="source with composer-authored source metadata",
                actual_type="source without metadata",
            )
        if INTERPRETATION_REQUIREMENTS_KEY not in source.options:
            raise ToolArgumentError(
                argument="affected_node_id",
                expected=f"source to contain a pending {kind.value} requirement for the requested term",
                actual_type=f"missing pending {kind.value} review site",
            )
        matched_terms = _matching_interpretation_sites(state, affected_node_id, kind, user_term)
        if not matched_terms:
            raise ToolArgumentError(
                argument="affected_node_id",
                expected=f"source to contain a pending {kind.value} requirement for the requested term",
                actual_type=f"missing pending {kind.value} review site",
            )
        if llm_draft is not None:
            # The presence guard above already proved the key is present on
            # the resolved source, and ``state`` is a frozen CompositionState
            # that cannot change in between. Direct subscript: a ``KeyError``
            # here would mean our guard logic broke. The retrieved value's
            # contents remain Tier-3 LLM-authored, so the shape check below
            # stays.
            requirements = source.options[INTERPRETATION_REQUIREMENTS_KEY]
            draft = None
            if isinstance(requirements, (list, tuple)):
                for requirement in requirements:
                    if (
                        isinstance(requirement, Mapping)
                        and requirement.get("kind") == kind.value
                        and requirement.get("user_term") == user_term
                        and requirement.get("status") == "pending"
                    ):
                        draft_value = requirement.get("draft")
                        draft = draft_value if isinstance(draft_value, str) else None
                        break
            if draft is not None and draft != llm_draft:
                raise ToolArgumentError(
                    argument="llm_draft",
                    expected="the exact source review requirement draft staged in source.options.interpretation_requirements",
                    actual_type="invented_source event draft does not match the source review requirement draft",
                )
        return

    if kind is InterpretationKind.PIPELINE_DECISION:
        node = _find_node_or_raise(state, affected_node_id)
        matched_terms = _matching_interpretation_sites(state, affected_node_id, kind, user_term)
        if not matched_terms:
            raise ToolArgumentError(
                argument="affected_node_id",
                expected=f"node {affected_node_id!r} to contain a pending {kind.value} requirement for the requested term",
                actual_type=f"missing pending {kind.value} review site",
            )
        requirements = node.options.get(INTERPRETATION_REQUIREMENTS_KEY)
        draft = None
        if isinstance(requirements, (list, tuple)):
            for requirement in requirements:
                if (
                    isinstance(requirement, Mapping)
                    and requirement.get("kind") == kind.value
                    and requirement.get("user_term") == user_term
                    and requirement.get("status") == "pending"
                ):
                    draft_value = requirement.get("draft")
                    draft = draft_value if isinstance(draft_value, str) else None
                    break
        try:
            validate_pipeline_decision_node_semantics(
                node=node,
                all_nodes=state.nodes,
                user_term=user_term,
                draft=draft,
                context="request_interpretation_review",
            )
        except ValueError as exc:
            raise ToolArgumentError(
                argument="affected_node_id",
                expected=str(exc),
                actual_type="pipeline_decision node that failed semantic review",
            ) from exc
        return

    node = _find_node_or_raise(state, affected_node_id)
    plugin = node.plugin
    if plugin != "llm":
        raise ToolArgumentError(
            argument="affected_node_id",
            expected="id of a node whose plugin is 'llm'",
            actual_type=f"node {affected_node_id!r} has plugin={plugin!r}",
        )

    options = node.options if node.options else {}
    # ``llm_model_choice`` reviews a different field of the same LLM node;
    # the prompt-template existence guard does not apply (a model can
    # legitimately be set before the prompt is authored mid-compose).
    if kind is not InterpretationKind.LLM_MODEL_CHOICE:
        prompt_template = options.get("prompt_template")
        if not isinstance(prompt_template, str) or not prompt_template:
            raise ToolArgumentError(
                argument="affected_node_id",
                expected=f"node {affected_node_id!r} to declare non-empty options.prompt_template",
                actual_type=f"options.prompt_template is {type(prompt_template).__name__}",
            )
    else:
        prompt_template = None

    matched_terms = _matching_interpretation_sites(state, affected_node_id, kind, user_term)
    if not matched_terms:
        expected_kind = kind.value
        raise ToolArgumentError(
            argument="affected_node_id",
            expected=(f"node {affected_node_id!r} to contain a pending {expected_kind} requirement or placeholder for the requested term"),
            actual_type=f"missing pending {expected_kind} review site",
        )
    # A pending vague_term requirement is only a *resolvable* handoff when the
    # prompt actually carries the substitution wiring the resolver consumes. A
    # requirement staged without a matching ``prompt_template_parts``
    # ``interpretation_ref`` (or legacy placeholder) produces a review the
    # operator can approve but never resolve — the resolver raises at
    # ``prompt_template_parts is required`` (a dead-end 422) or, worse, silently
    # drops the accepted value from the prompt. Reject at the Tier-3 boundary so
    # the dead event is never created; the LLM can recover by wiring the node.
    if kind is InterpretationKind.VAGUE_TERM and vague_term_wiring_count(options, user_term=user_term) != 1:
        raise ToolArgumentError(
            argument="affected_node_id",
            expected=(
                f"node {affected_node_id!r} to wire the requested interpretation into the prompt — "
                "exactly one prompt_template_parts entry "
                '{"kind": "interpretation_ref", "requirement_id": "<the pending requirement id>"} '
                "referencing the requirement, or exactly one legacy interpretation placeholder in options.prompt_template"
            ),
            actual_type=("pending vague_term requirement with no resolvable prompt wiring (the operator's resolve would dead-end)"),
        )
    if kind is InterpretationKind.LLM_PROMPT_TEMPLATE and llm_draft is not None and llm_draft != prompt_template:
        raise ToolArgumentError(
            argument="llm_draft",
            expected=f"current options.prompt_template for node {affected_node_id!r}",
            actual_type="stale prompt-template draft",
        )
    if kind is InterpretationKind.LLM_MODEL_CHOICE and llm_draft is not None:
        current_model = options.get("model")
        if not isinstance(current_model, str) or not current_model:
            raise ToolArgumentError(
                argument="affected_node_id",
                expected=f"node {affected_node_id!r} to declare non-empty options.model",
                actual_type=f"options.model is {type(current_model).__name__}",
            )
        if llm_draft != current_model:
            raise ToolArgumentError(
                argument="llm_draft",
                expected=f"current options.model for node {affected_node_id!r}",
                actual_type="stale model-choice draft",
            )


def _assert_affected_llm_node(
    state: CompositionState,
    affected_node_id: str,
    user_term: str,
) -> None:
    """Compatibility wrapper for older tests/importers of the vague-term-only helper."""
    _assert_affected_component(state, affected_node_id, InterpretationKind.VAGUE_TERM, user_term)


def _detect_unresolved_interpretation_placeholders_typed(
    nodes: Sequence[NodeSpec],
) -> list[tuple[str, str]]:
    """Return (node_id, term) tuples for every unresolved interpretation site.

    F-17 runtime detector operating directly on ``CompositionState.nodes``
    (a ``Sequence[NodeSpec]``). It accepts both structured pending
    interpretation requirements and legacy ``{{interpretation:…}}``
    placeholders during the migration window.

    Each unresolved site produces exactly one tuple per
    ``(node_id, term)`` pair, deduplicated within a node by insertion
    order so a repeated placeholder or requirement in one node does not
    inflate telemetry / error surface area.  Cross-node duplicates ARE
    preserved (the same ``term`` on two different nodes is two distinct
    unresolved sites).

    The return type is a ``list`` (not a ``tuple``) to match the
    dict-shaped sibling and the spec at
    ``docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md`` §F-17
    (``list[str]`` of terms, lifted to ``list[tuple[str, str]]`` here
    because the typed call site needs the ``node_id`` for both the
    telemetry attribute and the user-actionable error message).
    """
    for node in nodes:
        if node.plugin != "llm":
            continue
        if "prompt_template" not in node.options:
            continue
        prompt_template = node.options["prompt_template"]
        if not isinstance(prompt_template, str):
            raise ToolArgumentError(
                argument="nodes[].options.prompt_template",
                expected="a string",
                actual_type=type(prompt_template).__name__,
            )
    return list(transform_vague_term_site_tuples(tuple(nodes)))


RATE_CAP_PER_TERM_CODE: Final[str] = "RATE_CAP_PER_TERM"

RATE_CAP_PER_SESSION_DAY_CODE: Final[str] = "RATE_CAP_PER_SESSION_DAY"

# Dispatch discriminant for the dedup-on-resolved branch. Mirrors the
# RATE_CAP_* code constants: the compose loop and downstream telemetry
# read ``ToolArgumentError.code`` to distinguish branches without grepping
# the message string. Set when the LLM tries to re-stage a logical review
# (same kind + user_term + affected_node_id) that the user has already
# resolved in this composition branch.
DUPLICATE_RESOLVED_INTERPRETATION_CODE: Final[str] = "DUPLICATE_RESOLVED_INTERPRETATION"

RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {
        RATE_CAP_PER_TERM_CODE: "per_term",
        RATE_CAP_PER_SESSION_DAY_CODE: "per_session_day",
    }
)


def _utc_day_start(now: datetime) -> datetime:
    """Return the UTC-midnight start of the calendar day containing ``now``.

    F-30 fixed-window helper. The rate-limit window is the *calendar day in
    UTC*, not a sliding 24-hour window — simpler for operators to reason
    about and produces predictable reset behaviour ("the counter resets at
    UTC midnight"). The caller compares ``event.created_at >= utc_day_start(now)``.
    """
    aware_now = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    aware_now_utc = aware_now.astimezone(UTC)
    return aware_now_utc.replace(hour=0, minute=0, second=0, microsecond=0)


async def _check_duplicate_interpretation(
    *,
    session_id: UUID,
    composition_state_id: UUID,
    kind: InterpretationKind,
    user_term: str,
    affected_node_id: str,
    state: CompositionState,
    list_events_fn: Callable[..., Awaitable[list[InterpretationEventRecord]]],
) -> ToolResult | None:
    """Dedup a re-stage of an existing logical interpretation review.

    The dedup key is the tuple ``(kind, user_term, affected_node_id)``
    within the active composition branch. The composer skill prompt tells
    the LLM to "carry pending interpretation requirements forward
    unchanged"; this is the staging-side defence against the LLM emitting
    ``request_interpretation_review`` twice for the same logical review
    within one tool-loop turn (observed in session
    2766a814-2112-4a5c-b1f0-62f85169281a — see fix history).

    ``AUTO_INTERPRETED_OPT_OUT`` and ``AUTO_INTERPRETED_NO_SURFACES`` rows
    are excluded from the lookup because they are suppression / cap-hit
    audit records, not reviews — each such attempt is its own audit fact
    and must not be deduplicated against. The DB-side filter
    ``sources=[USER_APPROVED]`` excludes both.

    Three outcomes:

    - **No prior match** → returns ``None``; caller continues to the
      rate-limit gate and normal pending-row creation.
    - **Prior match still ``pending``** → returns a SUCCESS ``ToolResult``
      whose ``data`` payload reuses the existing event id and signals an
      idempotent re-stage via ``_kind='interpretation_review_pending_idempotent'``.
      No new DB row is written.
    - **Prior match already resolved** (any choice in
      ``{accepted_as_drafted, accepted_with_amendment, dismissed, …}``)
      → raises :class:`ToolArgumentError` with
      ``code=DUPLICATE_RESOLVED_INTERPRETATION_CODE``. This is a contract
      violation by the LLM and the compose loop's ARG_ERROR routing
      surfaces the error back to it for a retry that drops the duplicate.

    Async because ``list_events_fn`` (the injected
    ``list_interpretation_events`` service method) is async — it reads
    the session DB.
    """
    # Filter at the DB by interpretation_source — opt-out rows are
    # excluded so repeated opt-out audit events keep flowing through
    # unimpeded. ``status='all'`` returns both pending and resolved rows;
    # we need both branches.
    prior_events = await list_events_fn(
        session_id,
        status="all",
        composition_state_id=composition_state_id,
        sources=[InterpretationSource.USER_APPROVED],
    )
    matches = [
        event
        for event in prior_events
        if event.kind is kind and event.user_term == user_term and event.affected_node_id == affected_node_id
    ]
    if not matches:
        return None
    # ``list_interpretation_events`` orders by ``(created_at, id)`` —
    # the earliest match is the original event whose id the idempotent
    # response must echo so the frontend correlates with the existing card.
    original = matches[0]
    if original.choice is InterpretationChoice.PENDING:
        return ToolResult(
            success=True,
            updated_state=state,  # no state change — original review is still pending
            validation=state.validate(),
            affected_nodes=(affected_node_id,),
            data={
                "_kind": "interpretation_review_pending_idempotent",
                "event_id": str(original.id),
                "affected_node_id": affected_node_id,
                "kind": kind.value,
                "interpretation_source": original.interpretation_source.value,
                "message": "Interpretation review is already pending; reusing the existing event.",
            },
        )
    # Resolved — re-staging an already-decided review is forbidden.
    raise ToolArgumentError(
        argument="user_term",
        expected=(
            "a fresh interpretation review (this kind+user_term+affected_node_id "
            "tuple has already been resolved in this composition branch — carry "
            "the resolved value forward, do not re-stage)"
        ),
        actual_type="re-staging an already-resolved interpretation review",
        code=DUPLICATE_RESOLVED_INTERPRETATION_CODE,
    )


async def _check_interpretation_rate_limits(
    *,
    session_id: UUID,
    user_term: str,
    composition_state_id: UUID,
    list_events_fn: Callable[..., Awaitable[list[InterpretationEventRecord]]],
    per_term_cap: int,
    per_session_day_cap: int,
    now: datetime,
) -> None:
    """Enforce the two per-session interpretation-review rate limits (F-30/F-31).

    Two structural limits apply, in order:

    1. **Per-term cap (default 3):** the same ``(session_id, user_term)`` pair,
       scoped to the active ``composition_state_id`` branch, may be surfaced
       at most ``per_term_cap`` times before the LLM must fall back to
       AUTO_INTERPRETED_NO_SURFACES. Counted against rows that are NOT
       AUTO_INTERPRETED_OPT_OUT shape (those have NULL ``user_term``) and
       whose ``composition_state_id`` matches.

    2. **Per-session-day cap (default 10):** the session may make at most
       ``per_session_day_cap`` ``request_interpretation_review`` invocations
       per UTC calendar day. The window starts at UTC midnight (NOT a
       sliding 24-hour window) so the reset behaviour is predictable.

    On cap exceeded: raises :class:`ToolArgumentError`. The compose loop
    catches this and is expected (per the composer skill) to fall back to
    a non-LLM interpretation with ``interpretation_source =
    'auto_interpreted_no_surfaces'``.

    Async because ``list_events_fn`` (the injected
    ``list_interpretation_events`` service method) is async — it reads the
    session DB. A sync helper would force a blocking ``asyncio.run(...)``
    inside an already-async caller, which deadlocks.

    The cap values are injected as kwargs (not read from settings inside
    the helper) so this function is testable without a live settings
    object. Production callers thread ``WebSettings.composer_interpretation_*``
    in.
    """
    events = await list_events_fn(session_id, status="all")
    # Per-term cap — count rows for this composition branch with matching user_term.
    per_term_count = sum(
        1
        for event in events
        if event.composition_state_id == composition_state_id and event.user_term is not None and event.user_term == user_term
    )
    if per_term_count >= per_term_cap:
        raise ToolArgumentError(
            argument="user_term",
            expected=f"at most {per_term_cap} interpretation requests per term in this composition",
            actual_type=(
                f"per-term cap would be exceeded on request {per_term_count + 1}; use a direct interpretation "
                "in the prompt template instead"
            ),
            # Compose-loop discriminant (F-6): the rate-cap branch is the
            # trigger for the
            # AUTO_INTERPRETED_NO_SURFACES writer + F-15 telemetry. The
            # ``code`` field lets the loop distinguish this exception from a
            # generic ARG_ERROR without grepping the message string.
            code=RATE_CAP_PER_TERM_CODE,
        )
    # Per-session-day cap — UTC-midnight fixed window. Only rows with
    # populated ``user_term`` (i.e. not opt-out skeletons) count toward the
    # invocation budget; opt-out is a different action with its own row.
    day_start = _utc_day_start(now)
    per_day_count = sum(1 for event in events if event.user_term is not None and event.created_at >= day_start)
    if per_day_count >= per_session_day_cap:
        raise ToolArgumentError(
            argument="user_term",
            expected=f"at most {per_session_day_cap} interpretation requests per session per UTC day",
            actual_type=(
                f"session would record {per_day_count + 1} requests today — the compose loop should "
                f"fall back to auto-interpretation (AUTO_INTERPRETED_NO_SURFACES)"
            ),
            # See per-term cap above for the ``code`` field rationale.
            code=RATE_CAP_PER_SESSION_DAY_CODE,
        )


async def _handle_request_interpretation_review(
    arguments: object,
    state: CompositionState,
    *,
    session_id: UUID,
    composition_state_id: UUID,
    tool_call_id: str,
    now: datetime,
    per_term_cap: int,
    per_session_day_cap: int,
    model_identifier: str,
    model_version: str,
    provider: str,
    composer_skill_hash: str,
    create_pending_interpretation_event: Callable[..., Awaitable[InterpretationEventRecord]],
    list_interpretation_events: Callable[..., Awaitable[list[InterpretationEventRecord]]],
) -> ToolResult:
    """Stage a pending interpretation event for user review.

    Returns a SUCCESS :class:`ToolResult` whose ``data`` payload signals
    the frontend to surface the review affordance. Does NOT advance
    composition state version — state mutation happens at /resolve time
    when the user accepts or amends the draft.

    Async because it awaits the injected service methods. Registered in
    :data:`_SESSION_AWARE_TOOL_HANDLERS` (NOT the synchronous
    :data:`_MUTATION_TOOLS` registry — see the dual-registry invariant
    documented at the registry block above).
    """
    parsed = cast(
        _RequestInterpretationReviewArgumentsModel,
        _validate_mutation_arguments(
            _RequestInterpretationReviewArgumentsModel,
            arguments,
            "request_interpretation_review arguments",
        ),
    )
    # Backend owns prompt-template surfacing (elspeth-e51216d305 Case B). The
    # ``llm_prompt_template`` review is auto-staged on every LLM node and the
    # BACKEND surfaces its EVENT against the FINAL frozen skeleton at turn
    # finalization, so it can never go stale against a later skeleton mutation.
    # The LLM must NOT surface it mid-build via this tool; reject the kind at
    # the Tier-3 boundary immediately after the parse, before any service call.
    if parsed.kind is InterpretationKind.LLM_PROMPT_TEMPLATE:
        raise ToolArgumentError(
            argument="kind",
            expected="vague_term, invented_source, pipeline_decision, or llm_model_choice",
            actual_type=("llm_prompt_template — surfaced automatically by the backend at turn finalization; do not request it"),
        )
    # F-34 credential prefilter: Tier-3 boundary check before any DB write.
    # ``_reject_credential_shaped_content`` raises ``ValueError``; we wrap
    # as ToolArgumentError so the compose loop's ARG_ERROR routing catches
    # it (a bare ValueError would land in the plugin-crash catch-all and
    # mis-classify the failure as a Tier-1 plugin bug).
    for field_name, field_value in (("user_term", parsed.user_term), ("llm_draft", parsed.llm_draft)):
        try:
            _reject_credential_shaped_content(field_value)
        except ValueError as exc:
            raise ToolArgumentError(
                argument=field_name,
                expected="content that does not match a known credential shape",
                actual_type="credential-shaped content rejected at the tool boundary",
            ) from exc
    # F-2 prompt-injection guard on llm_draft. Vague-term and invented-source
    # drafts become accepted values/artifacts; reject template metacharacters
    # before any DB write. Prompt-template reviews deliberately carry real
    # Jinja such as ``{{ row.html }}``, so they keep the credential prefilter
    # above but skip the accepted-value validator.
    if parsed.kind in {
        InterpretationKind.VAGUE_TERM,
        InterpretationKind.PIPELINE_DECISION,
        InterpretationKind.LLM_MODEL_CHOICE,
    }:
        # ``llm_model_choice`` drafts are short model identifier strings
        # (e.g. ``anthropic/claude-sonnet-4.5``) selected from a catalog
        # at compose time; they are treated as accepted-value content
        # because they flow into the runtime config as-is, never as
        # template fragments — the same content rules apply (no Jinja,
        # no control characters, no credential shapes).
        try:
            _validate_accepted_value_content(parsed.llm_draft)
        except ValueError as exc:
            raise ToolArgumentError(
                argument="llm_draft",
                expected="content without template metacharacters, control characters, or credential patterns",
                actual_type="rejected by accepted-value content validator",
            ) from exc
    elif parsed.kind is InterpretationKind.INVENTED_SOURCE:
        try:
            _validate_source_artifact_review_content(parsed.llm_draft)
        except ValueError as exc:
            raise ToolArgumentError(
                argument="llm_draft",
                expected="source artifact content without template metacharacters, credential patterns, or non-printable controls",
                actual_type="rejected by source-artifact content validator",
            ) from exc
    # Tier-3 boundary check on the LLM-supplied component/kind pair. Missing
    # component, wrong component kind, and absent review metadata all raise
    # ARG_ERROR so the LLM can retry after fixing composition state. This must
    # happen before rate limiting so cap-exceeded audit rows are emitted only
    # for valid pending review sites.
    _assert_affected_component(state, parsed.affected_node_id, parsed.kind, parsed.user_term, parsed.llm_draft)
    # Dedup gate. Runs BEFORE the rate-limit check on purpose: a duplicate
    # re-stage is zero logical progress and must not consume the per-term
    # budget, which is reserved for legitimate user-resolved churn (accept,
    # then LLM re-surfaces a refined draft, etc.). Scoped to the active
    # composition branch (same ``composition_state_id``) so an edit that
    # changes the pipeline can legitimately re-surface a review.
    # ``AUTO_INTERPRETED_OPT_OUT`` rows are excluded — those are suppression
    # audit records, not reviews; each opt-out attempt is its own audit fact.
    idempotent_result = await _check_duplicate_interpretation(
        session_id=session_id,
        composition_state_id=composition_state_id,
        kind=parsed.kind,
        user_term=parsed.user_term,
        affected_node_id=parsed.affected_node_id,
        state=state,
        list_events_fn=list_interpretation_events,
    )
    if idempotent_result is not None:
        return idempotent_result
    # Rate-limit gate. Cap-exceeded raises ToolArgumentError; the compose
    # loop is expected to react by writing an AUTO_INTERPRETED_NO_SURFACES
    # event (handled in service.py — see ``record_auto_interpreted_no_surfaces_event``).
    await _check_interpretation_rate_limits(
        session_id=session_id,
        user_term=parsed.user_term,
        composition_state_id=composition_state_id,
        list_events_fn=list_interpretation_events,
        per_term_cap=per_term_cap,
        per_session_day_cap=per_session_day_cap,
        now=now,
    )
    # Persist the pending row. The service method re-validates affected_node_id
    # under the session write lock (defence in depth — the compose-state could
    # in principle race with another writer; the service is the authoritative
    # consistency gate).
    event = await create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=composition_state_id,
        affected_node_id=parsed.affected_node_id,
        tool_call_id=tool_call_id,
        user_term=parsed.user_term,
        kind=parsed.kind,
        llm_draft=parsed.llm_draft,
        model_identifier=model_identifier,
        model_version=model_version,
        provider=provider,
        composer_skill_hash=composer_skill_hash,
    )
    if event.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT:
        return ToolResult(
            success=True,
            updated_state=state,
            validation=state.validate(),
            affected_nodes=(),
            data={
                "_kind": "interpretation_review_suppressed_by_opt_out",
                "event_id": str(event.id),
                "kind": parsed.kind.value,
                "interpretation_source": event.interpretation_source.value,
                "interpretation_review_disabled": True,
                "message": "Interpretation review suppressed because this session has opted out.",
            },
        )
    return ToolResult(
        success=True,
        updated_state=state,  # no state change yet — /resolve advances version
        validation=state.validate(),
        affected_nodes=(parsed.affected_node_id,),
        data={
            "_kind": "interpretation_review_pending",
            "event_id": str(event.id),
            "affected_node_id": parsed.affected_node_id,
            "kind": parsed.kind.value,
            "interpretation_source": event.interpretation_source.value,
            "message": (
                "Interpretation review staged for user review. Waiting for user acceptance/amendment before the pipeline can finalise."
            ),
        },
    )


SessionAwareToolHandler = Callable[..., Awaitable[ToolResult]]


_SESSION_AWARE_TOOL_HANDLERS: dict[str, SessionAwareToolHandler] = {
    "request_interpretation_review": _handle_request_interpretation_review,
}

# The ``is_session_aware_tool`` predicate lives in
# ``elspeth.web.composer.tools.discovery`` alongside the other classification
# predicates so the tool-name vocabulary has one source of truth. Import it
# from there.


TOOLS_IN_MODULE: tuple[ToolDeclaration, ...] = (
    _GET_PIPELINE_STATE_DECLARATION,
    _SET_PIPELINE_DECLARATION,
    _APPLY_PIPELINE_RECIPE_DECLARATION,
)
"""Every tool declared in this module, in stable order.

``_dispatch.py`` aggregates this tuple alongside every other plane's
TOOLS_IN_MODULE to build the registered-tool universe.

Note: ``request_interpretation_review`` (the session-aware async handler) is
dispatched outside ``execute_tool`` and is intentionally NOT migrated to the
ToolDeclaration model in Step 3 — its per-call kwarg surface differs from
the synchronous ``ToolContext`` (9 extra kwargs: session_id,
composition_state_id, tool_call_id, now, per_term_cap, per_session_day_cap,
model_identifier, model_version, provider, composer_skill_hash, plus two
``Awaitable`` callbacks). The migration is captured in filigree ticket
elspeth-f5da936747 (P3, parent elspeth-6c9972ccbf); option-A requires
widening the ``ToolHandler`` alias to a sync-or-async union and adding an
escape hatch on ``ToolDeclaration`` for the extra kwargs. The inline schema
for this tool remains in ``_dispatch.py:get_tool_definitions()`` until that
work lands."""
