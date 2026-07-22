"""Composer generation plane — preview, diff, explain, plugin schema, proof diagnostics."""

from __future__ import annotations

import ast
import csv
import io
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, TypedDict

from opentelemetry import metrics
from sqlalchemy import Engine

from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.schema import get_aggregation_contract_options, get_raw_schema_config
from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.contracts.value_source import get_catalog_values
from elspeth.core.expression_parser import ExpressionEvaluationError, ExpressionParser
from elspeth.plugins.infrastructure.manager import (
    PluginNotFoundError,
    get_shared_plugin_manager,
)
from elspeth.plugins.transforms.llm.model_catalog import (
    MODEL_CATALOG_OPENROUTER,
    OPENROUTER_LITELLM_PREFIX,
    read_litellm_model_list,
)
from elspeth.web.catalog.protocol import PluginKind
from elspeth.web.composer._producer_resolver import ProducerResolver, is_source_producer_id
from elspeth.web.composer.source_inspection import (
    SourceInspectionFacts,
    derive_extra_column_risk,
    derive_required_header_mismatch_risk,
    inspect_blob_content,
    inspect_csv_source_content,
)
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    SourceSpec,
    _source_options_have_schema,
    _validate_gate_expression,
)
from elspeth.web.composer.tools._common import (
    _DATA_ERROR_KEY,
    ToolContext,
    ToolResult,
    _discovery_result,
    _failure_result,
    _plugin_policy_failure,
    _validate_plugin_name,
    diff_states,
)
from elspeth.web.composer.tools.blobs import (
    _sync_get_blob,
    _verify_blob_content_integrity,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
)
from elspeth.web.composer.tools.sessions import (
    _authoring_validation_payload,
)

_AUTHORING_VALIDATION_COUNTER = metrics.get_meter("elspeth.web.composer.tools").create_counter(
    "composer.authoring_validation.total",
    description="Total authoring (Stage 1) validation outcomes from preview_pipeline",
)


@trust_boundary(
    tier=3,
    source="CSV-source 'delimiter' option from external / LLM-authored composer source options",
    source_param="options",
    suppresses=("R1",),
    invariant="raises ValueError when 'delimiter' is present but not a single-character str; never coerces",
    test_ref="tests/unit/web/composer/test_generation_source_option_boundaries.py::test_csv_source_delimiter_rejects_non_string",
    test_fingerprint="70dcc85656bdccb86ef9d40258841bfbb2ef3123bbe836f9a7ca4266091d3716",
)
def _csv_source_delimiter(options: Mapping[str, Any]) -> str:
    raw = options.get("delimiter")
    if raw is None:
        return ","
    if type(raw) is not str:
        raise ValueError(f"csv source delimiter must be str when present; got {type(raw).__name__}")
    if len(raw) != 1:
        raise ValueError(f"csv source delimiter must be one character; got {raw!r}")
    return raw


@trust_boundary(
    tier=3,
    source="CSV-source 'skip_rows' option from external / LLM-authored composer source options",
    source_param="options",
    suppresses=("R1",),
    invariant="raises ValueError when 'skip_rows' is present but not a non-negative int; never coerces",
    test_ref="tests/unit/web/composer/test_generation_source_option_boundaries.py::test_csv_source_skip_rows_rejects_non_int",
    test_fingerprint="975c5318f2f88801e228ad4934c33851d6bf6ecb82891032a22d305ccf7b2f21",
)
def _csv_source_skip_rows(options: Mapping[str, Any]) -> int:
    raw = options.get("skip_rows")
    if raw is None:
        return 0
    if type(raw) is not int:
        raise ValueError(f"csv source skip_rows must be int when present; got {type(raw).__name__}")
    if raw < 0:
        raise ValueError(f"csv source skip_rows must be non-negative; got {raw}")
    return raw


@trust_boundary(
    tier=3,
    source="CSV-source 'columns' option from external / LLM-authored composer source options",
    source_param="options",
    suppresses=("R1", "R5"),
    invariant="raises ValueError when 'columns' is present but not a list/tuple of str; never coerces",
    test_ref="tests/unit/web/composer/test_generation_source_option_boundaries.py::test_csv_source_columns_rejects_non_sequence",
    test_fingerprint="b5bb902a03d873d14c816d002ab58e9dd8d99c5f51c2136166480bdaf6ac94e6",
)
def _csv_source_columns(options: Mapping[str, Any]) -> tuple[str, ...] | None:
    raw = options.get("columns")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"csv source columns must be a list of strings when present; got {type(raw).__name__}")
    columns: list[str] = []
    for idx, item in enumerate(raw):
        if type(item) is not str:
            raise ValueError(f"csv source columns[{idx}] must be str; got {type(item).__name__}")
        columns.append(item)
    return tuple(columns)


@trust_boundary(
    tier=3,
    source="CSV-source 'field_mapping' option from external / LLM-authored composer source options",
    source_param="options",
    suppresses=("R5",),
    invariant="raises ValueError when 'field_mapping' is present but not a Mapping of str→str; never coerces",
    test_ref="tests/unit/web/composer/test_generation_field_mapping.py::test_csv_source_field_mapping_rejects_non_mapping",
    test_fingerprint="0b60b2320a348028e1b74cecfe0ed49335417a9844e7478f1a652eb2725e68ac",
)
def _csv_source_field_mapping(options: Mapping[str, Any]) -> dict[str, str] | None:
    raw = options["field_mapping"] if "field_mapping" in options else None
    if raw is None:
        return None
    # Accept any Mapping, not just dict: CompositionState options are deep-frozen,
    # so field_mapping arrives as a MappingProxyType. A `type(raw) is dict` check
    # wrongly rejects the frozen mapping (Tier-3 structural validation of external/
    # LLM-authored config — isinstance is the correct, subtype-aware guard here).
    if not isinstance(raw, Mapping):
        raise ValueError(f"csv source field_mapping must be a mapping when present; got {type(raw).__name__}")
    mapping: dict[str, str] = {}
    for key, value in raw.items():
        if type(key) is not str:
            raise ValueError(f"csv source field_mapping keys must be str; got {type(key).__name__}")
        if type(value) is not str:
            raise ValueError(f"csv source field_mapping[{key!r}] must be str; got {type(value).__name__}")
        mapping[key] = value
    return mapping


@trust_boundary(
    tier=3,
    source="source 'schema.required_fields' from external / LLM-authored composer source options",
    source_param="schema",
    suppresses=("R1", "R5"),
    invariant="raises ValueError when 'required_fields' is present but not a list/tuple of str; never coerces",
    test_ref="tests/unit/web/composer/test_generation_source_option_boundaries.py::test_schema_required_fields_rejects_non_sequence",
    test_fingerprint="b3c30bf7a9ac3f09269f01b4d43a3330440da5a4f2046abf025b1ab272c047ee",
)
def _schema_required_fields(schema: Mapping[str, Any]) -> tuple[str, ...]:
    raw = schema.get("required_fields")
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"schema.required_fields must be a list of strings when present; got {type(raw).__name__}")
    required: list[str] = []
    for idx, item in enumerate(raw):
        if type(item) is not str:
            raise ValueError(f"schema.required_fields[{idx}] must be str; got {type(item).__name__}")
        required.append(item)
    return tuple(required)


_EXPRESSION_GRAMMAR = """\
Gate Expression Syntax Reference
=================================

Variables:
  row      - The current row as a dict. Access fields via row['field_name'].

Field access:
  row['field_name']       Direct access (raises KeyError if missing)
  row.get('field_name')   Returns None if missing (NO default argument allowed)

Operators:
  ==, !=, <, >, <=, >=   Comparison
  and, or, not            Boolean logic
  in, not in              Membership test
  +, -, *, /, //, %       Arithmetic

Built-in functions (only these are allowed):
  len()    Length of a sequence or string
  abs()    Absolute value of a number

Type coercion functions (int, str, float, bool) are NOT available.
Types are guaranteed by the source schema — no coercion is needed in expressions.

Examples:
  row['confidence'] >= 0.85
  row['status'] == 'approved'
  row['category'] in ('A', 'B', 'C')
  row.get('optional_field') is not None
  row['score'] > 0.5 and row['status'] != 'rejected'
  len(row['name']) > 0

Forbidden:
  row.get('field', default)   Default values fabricate data — use 'is not None' test
  int(row['x'])               Type coercion — coerce at source schema instead
  Imports, lambdas, comprehensions, attribute access (except row.get)
"""


def get_expression_grammar() -> str:
    """Return the gate expression syntax reference."""
    return _EXPRESSION_GRAMMAR


def _handle_get_plugin_schema(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    plugin_type = arguments["plugin_type"]
    name = arguments["name"]
    policy_error = _validate_plugin_name(context, plugin_type, name)
    if policy_error is not None:
        return _plugin_policy_failure(state, policy_error)
    try:
        schema = context.catalog.get_schema(plugin_type, name)
        return _discovery_result(state, schema)
    except (ValueError, KeyError) as exc:
        # ValueError: catalog contract for "unknown plugin/type"
        # KeyError: LLM omitted required argument (Tier 3)
        return _failure_result(state, str(exc))


_GET_PLUGIN_SCHEMA_DECLARATION = ToolDeclaration(
    name="get_plugin_schema",
    handler=_handle_get_plugin_schema,
    kind=ToolKind.DISCOVERY,
    description="Get the full configuration schema for a plugin.",
    json_schema={
        "type": "object",
        "properties": {
            "plugin_type": {
                "type": "string",
                "enum": ["source", "transform", "sink"],
                "description": "Plugin type.",
            },
            "name": {
                "type": "string",
                "description": "Plugin name (e.g. 'csv').",
            },
        },
        "required": ["plugin_type", "name"],
        "additionalProperties": False,
    },
    cacheable=True,
)


def _handle_get_expression_grammar(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    del context  # unused; signature uniformity with the other handlers.
    return _discovery_result(state, get_expression_grammar())


_GET_EXPRESSION_GRAMMAR_DECLARATION = ToolDeclaration(
    name="get_expression_grammar",
    handler=_handle_get_expression_grammar,
    kind=ToolKind.DISCOVERY,
    description="Get the gate expression syntax reference.",
    json_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    cacheable=True,
)


_VALIDATION_ERROR_PATTERNS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        r"No source configured",
        "The pipeline has no data source. Every pipeline needs exactly one source to read input data from.",
        "Use set_source to configure a source plugin (e.g. csv, json, dataverse).",
    ),
    (
        r"No sinks configured",
        "The pipeline has no outputs. At least one sink is needed to write results.",
        "Use set_output to add an output (e.g. csv, json).",
    ),
    (
        r"references unknown node '(.+)' as from_node",
        "An edge references a node that doesn't exist in the pipeline as its source.",
        "Check the edge's from_node value. Either add the missing node with upsert_node or fix the edge with upsert_edge.",
    ),
    (
        r"references unknown node '(.+)' as to_node",
        "An edge references a node or output that doesn't exist in the pipeline as its target.",
        "Check the edge's to_node value. Either add the missing node/output or fix the edge.",
    ),
    (
        r"Duplicate node ID: '(.+)'",
        "Two nodes have the same ID. Each node must have a unique identifier.",
        "Rename one of the duplicate nodes using upsert_node with a different id.",
    ),
    (
        r"Duplicate output name: '(.+)'",
        "Two outputs have the same name. Each output must have a unique name.",
        "Rename one of the duplicate outputs using set_output with a different sink_name.",
    ),
    (
        r"Duplicate edge ID: '(.+)'",
        "Two edges have the same ID. Each edge must have a unique identifier.",
        "Remove the duplicate edge with remove_edge and re-add with a unique id.",
    ),
    (
        r"Gate '(.+)' is missing required field '(.+)'",
        "A gate node is missing a required configuration field (condition or routes).",
        "Update the gate with upsert_node, providing the missing field.",
    ),
    (
        r"Transform '(.+)' must not have '(.+)' field",
        "A transform node has a field that only gates should have (condition or routes).",
        "Update the node with upsert_node. Set node_type to 'gate' if routing is needed, or remove the field.",
    ),
    (
        r"Coalesce '(.+)' is missing required field '(.+)'",
        "A coalesce node is missing a required field (branches or policy).",
        "Update the coalesce node with upsert_node, providing the missing field.",
    ),
    (
        r"aggregation_missing_plugin|Aggregation '(.+)' is missing required field 'plugin'",
        "An aggregation node needs a plugin to define its aggregation behaviour.",
        "Update the aggregation with upsert_node, specifying the plugin name.",
    ),
    (
        r"coalesce_branch_unreachable|branches .* are not reachable",
        "A coalesce branches mapping names an incoming connection that no runtime routing field produces. The usual cause is "
        "the WIRING AROUND the coalesce, not the coalesce itself: a branch transform's on_success routes past the coalesce "
        "(e.g. straight to a sink), so nothing arrives under the connection name the branches value claims. The rejection's "
        "connectivity facts list each unreachable branches value and every connection the pipeline actually produces.",
        "Wire each fork branch end-to-end: the gate fork_to name is the branch transform's input; that transform's on_success "
        "must be a unique connection name (NOT a sink); the coalesce branches VALUE for that branch is exactly that connection. "
        "A branch with no transforms uses its fork branch name as the value. Repair the branch transforms' on_success together "
        "with the coalesce, drawing every branches value from the connectivity facts' produced_connections.",
    ),
    (
        r"node_input_not_reachable|input '(.+)' is not reachable",
        "A node's input connection point is not produced by the runtime routing fields.",
        "Set source.on_success or an upstream node's on_success/on_error/route/fork_to so it matches this node's input.",
    ),
    (
        r"Unknown .+ plugin '(.+)'",
        "The specified plugin name is not available in the catalog.",
        "Use list_sources, list_transforms, or list_sinks to see available plugins.",
    ),
    (
        r"Path violation \(S2\).*[Ss]ource",
        "The source file path is outside the allowed directories.",
        "Source paths must be under the blobs/ directory. Upload a file first or use set_source_from_blob.",
    ),
    (
        r"Path violation \(S2\).*[Ss]ink",
        "The sink output path is outside the allowed directories.",
        "Sink output paths must be under the outputs/ or blobs/ directory.",
    ),
    (
        r"Path violation \(S2\)",
        "A file path is outside the allowed directories.",
        "Source paths must be under the blobs/ directory. Sink output paths must be under the outputs/ or blobs/ directory.",
    ),
    (
        r"Invalid options for source '(.+)':",
        "The source plugin configuration is invalid. A required option may be missing or have an invalid value.",
        "Use get_pipeline_state with component='source' to see current options, then use patch_source_options to fix.",
    ),
    (
        r"Invalid options for transform '(.+)':",
        "A transform node has invalid configuration. A required option may be missing or have an invalid value.",
        "Use get_pipeline_state to see the node's current options, then use patch_node_options to fix.",
    ),
    (
        r"Invalid options for sink '(.+)':",
        "A sink output has invalid configuration. A required option may be missing (e.g. path for file-based sinks).",
        "Use get_pipeline_state to see the output's current options, then use patch_output_options to fix.",
    ),
    # Contract-family entries are ORDER-SENSITIVE: the extras (locked-input)
    # messages and the sink-edge messages both start with "Schema contract
    # violation:", so the more specific patterns must precede the generic one.
    (
        r"sink_locked_extras|Extra fields rejected by sink input contract",
        "The sink schema is locked (mode: fixed) and the upstream producer emits extra fields the sink does not accept. "
        "The rejection's contract facts name the producer, the sink, and the extra field names.",
        "Change ONLY that edge: relax the sink schema (mode: flexible), add the extra field names to the sink schema fields, "
        "or insert a field_mapper with select_only: true before the sink to drop them.",
    ),
    (
        r"locked_input_extras|Extra fields rejected by consumer input contract",
        "The consumer node's input schema is locked (mode: fixed) and the upstream producer emits extra fields it does not accept. "
        "The rejection's contract facts name the producer, the consumer, and the extra field names.",
        "Change ONLY that edge: relax the consumer schema (mode: flexible), add the extra field names to the consumer schema fields, "
        "or insert a field_mapper with select_only: true upstream to drop them.",
    ),
    (
        r"transform_contract_violation|Transform contract violation:",
        "A transform's declared output schema promises fields its own configuration cannot emit "
        "(e.g. field_mapper with select_only: true only emits its mapping targets). "
        "The rejection's contract facts name the node and the declared-but-unemitted field names.",
        "Change ONLY that node: remove the missing field names from its schema declaration, extend its mapping so they are "
        "actually produced, or set select_only: false.",
    ),
    (
        r"semantic_contract_violation|Semantic contract violation:|Semantic contract:",
        "A consumer requires a field to satisfy a semantic contract (content kind, framing, or value type) that its upstream "
        "producer does not declare or actively conflicts with. The rejection's contract facts name the producer and consumer.",
        "Call get_plugin_assistance for the consumer plugin to see which producers satisfy its semantic requirements, then "
        "change ONLY that edge: swap the producer, or route through a transform that produces the required content kind.",
    ),
    (
        r"contract_config_invalid|Invalid contract config:",
        "A schema or contract declaration failed to parse — malformed fields spec, required_fields/required_input_fields with "
        "the wrong shape (must be a list of strings), or an invalid mode.",
        "Re-emit ONLY the offending component's schema options. Field specs are single-key dicts like {'field_name': 'str'} or "
        "strings like 'field_name: str'; required_input_fields is a list of field-name strings; mode is one of observed, "
        "flexible, fixed.",
    ),
    (
        r"sink_contract_violation|Schema contract violation: '.*' -> 'output:[^']+'",
        "A sink schema requires fields that its upstream producer does not guarantee. "
        "The rejection's contract facts name the producer, the sink, and the missing field names.",
        "Call preview_pipeline to inspect edge_contracts, then either relax the sink schema with patch_output_options or update the upstream schema with patch_source_options or patch_node_options and re-preview until the edge shows satisfied=true.",
    ),
    (
        r"schema_contract_violation|Schema contract violation:",
        "A downstream node requires fields that its upstream producer does not guarantee. "
        "The rejection's contract facts name the producer, the consumer, and the missing field names.",
        "Call preview_pipeline to inspect edge_contracts, then update the upstream schema with patch_source_options or patch_node_options and re-preview until the edge shows satisfied=true.",
    ),
    # ── Closed structural node-shape codes ──────────────────────────────────
    # The planner repair feedback strips validation messages; these codes are
    # the only signal it carries, so each must be explainable here.
    (
        r"unknown[ _]node_type",
        "The node_type is not one of the composer's node kinds: aggregation, coalesce, gate, queue, transform. There is no 'fork' node_type.",
        "Keep your current pipeline shape and change ONLY the invalid node: forking is expressed with a GATE node — node_type='gate', condition='True', routes={'true': 'fork', 'false': 'fork'}, fork_to=['branch_a', 'branch_b']; each branch node reads one branch name as its input, and branches rejoin at a COALESCE node (branches + policy + merge). A node after the coalesce consumes it by setting input='<coalesce id>' — the coalesce's own on_success may only name a sink.",
    ),
    (
        r"coalesce_on_success_must_be_sink|coalesce_on_success_unknown_sink|Coalesce on_success must point to a sink|Coalesce '(.+)' on_success references unknown sink",
        "A coalesce node's on_success may only name an existing sink; merged rows otherwise flow to whichever node reads the coalesce id as its input.",
        "Either set the coalesce on_success to a sink name from outputs[], or leave on_success null and give the downstream node input='<coalesce node id>'.",
    ),
    (
        r"coalesce_missing_policy|Coalesce '(.+)' is missing required field 'policy'",
        "A coalesce node must declare how it settles branches.",
        "Set policy='require_all' (wait for every branch) and merge='union' (combine branch fields into one row); branches maps "
        "each fork branch name to the connection ARRIVING at the coalesce — the branch's last transform on_success (e.g. "
        "branches={'branch_a': 'a_done', 'branch_b': 'b_done'}), or the fork branch name itself only when that branch has no transforms.",
    ),
    (
        r"coalesce_missing_branches|Coalesce '(.+)' is missing required field 'branches'",
        "A coalesce node must name the branch connections it rejoins.",
        "Set branches to a mapping of fork branch name -> the connection ARRIVING at the coalesce: the branch's last transform "
        "on_success (e.g. branches={'branch_a': 'a_done', 'branch_b': 'b_done'}), or the fork branch name itself only when that "
        "branch has no transforms. Add policy='require_all' and merge='union'.",
    ),
    (
        r"transform_missing_on_success|Transform '(.+)' is missing required field 'on_success'",
        "Every transform must route its successful rows somewhere.",
        "Set the transform's on_success to the next connection or sink name; use on_error='discard' unless failed rows need a "
        "quarantine sink. A transform on a fork branch that rejoins at a coalesce must publish the connection named by that "
        "coalesce's branches value for its branch — not a sink.",
    ),
    (
        r"transform_missing_on_error|Transform '(.+)' is missing required field 'on_error'",
        "Every transform must declare where failed rows go.",
        "Set the transform's on_error — 'discard' is the simplest safe choice, or route to a quarantine sink name.",
    ),
    (
        r"gate_missing_condition|Gate '(.+)' is missing required field 'condition'",
        "A gate node needs a boolean row expression to route on.",
        "Set condition to a row expression (call get_expression_grammar for syntax); a pure fan-out gate uses condition='True'.",
    ),
    (
        r"gate_missing_routes|Gate '(.+)' is missing required field 'routes'",
        "A gate node needs a routes mapping for its condition outcomes.",
        "Set routes={'true': <connection-or-'fork'>, 'false': <connection-or-'fork'>}; use 'fork' with fork_to=[...] to fan a row out to several branches.",
    ),
    (
        r"fork_branch_no_destination|fork branch '(.+)' with no destination",
        "Every gate fork_to branch name must land somewhere: as a KEY in a coalesce 'branches' mapping, or as an exact sink name.",
        "Key the coalesce branches by the gate's fork branch names — e.g. fork_to=['branch_a','branch_b'] pairs with branches={'branch_a': '<connection arriving from that branch>', 'branch_b': '<connection>'} — where each value is the connection reaching the coalesce after any per-branch transforms (the transform's on_success).",
    ),
    (
        r"coalesce_policy_invalid|coalesce_merge_invalid",
        "Coalesce policy and merge use the engine's closed vocabularies.",
        "Set policy to one of: require_all, quorum, best_effort, first — and merge to one of: union, nested, select. For an A/B rejoin that combines branch fields into one row: policy='require_all', merge='union'.",
    ),
    (
        r"pipeline_decision_unregistered",
        "A pipeline_decision interpretation review may only use one of the registered decision kinds — novel decision terms can never be reviewed or resolved.",
        "Remove the pipeline_decision entry from the node's interpretation_requirements (record the rationale in metadata.description instead), or use an llm_prompt_template review for prompt-shaped decisions. Registered kinds: drop_raw_html_fields, web_scrape_http_identity, prompt_injection_shield_recommendation.",
    ),
    (
        r"interpretation_requirements_invalid",
        "A node's interpretation_requirements entry is malformed. interpretation_requirements is a list of review entry objects, each carrying string fields kind, user_term, and draft; the server-owned id and status fields are filled automatically.",
        'Simplest fix: omit interpretation_requirements entirely — the required LLM reviews (prompt template, model choice) are auto-staged by the server, and the prompt-injection-shield recommendation is advisory. If you must stage a review, re-emit each entry as an object {kind, user_term, draft} with non-empty string values — e.g. {"kind": "pipeline_decision", "user_term": "prompt_injection_shield_recommendation", "draft": "<recommendation text>"} — and never author id, status, or resolved review metadata.',
    ),
    (
        r"plugin_options_invalid",
        "One or more of the component's options failed its plugin schema (missing required option, wrong shape, flexible schema without fields, or — for llm — a missing/invalid operator profile alias).",
        "Call get_plugin_schema(<plugin_type>, <plugin_name>) for the exact option shapes and allowed values (the llm transform's 'profile' enum lists the operator-approved aliases), fix only the offending options, and re-emit. For schema options: use {mode: observed} to infer types, or provide explicit fields with mode fixed/flexible.",
    ),
    (
        r"transform_on_success_dangling|aggregation_on_success_dangling|source_on_success_dangling|is neither a sink nor a known connection",
        "An on_success destination must be an existing sink name or a connection another node reads as its input.",
        "Point on_success at one of outputs[].sink_name exactly, or at the connection name a downstream node declares as its input. Call get_pipeline_state to list the current sink names and node input connections, then re-emit with a matching destination.",
    ),
    (
        r"transform_on_error_unknown_sink|references unknown sink",
        "An on_error destination may only be 'discard' or an existing sink name.",
        "Set on_error='discard', or point it at one of outputs[].sink_name exactly.",
    ),
    (
        r"gate_route_labels_mismatch|route labels don't match",
        'A boolean gate condition routes on exactly the labels "true" and "false" — both must be present even when they share a destination.',
        'Use routes={"true": <destination>, "false": <destination>}. For a pure fan-out gate: condition=\'True\', routes={"true": "fork", "false": "fork"}, fork_to=[<branch connections>]. Only string-returning conditions may use custom route labels.',
    ),
    (
        r"no_source_configured|No source configured",
        "The pipeline has no source; every pipeline reads rows from exactly one configured source (or named sources).",
        "Include a source block in the set_pipeline call — plugin, on_success connection, and options with a schema.",
    ),
    (
        r"no_sinks_configured|No sinks configured",
        "The pipeline has no outputs; rows must land in at least one sink.",
        "Include at least one outputs[] entry — sink_name, plugin, and options (file-based sinks need a path under outputs/).",
    ),
    (
        r"source_name_invalid",
        "A named source uses an invalid name.",
        "Source names must be short lowercase identifiers; rename the source key and re-emit.",
    ),
    (
        r"reserved_node_id|Reserved node id|reserved source producer namespace",
        "The id 'source' and the 'source:<name>' namespace are reserved for pipeline sources; no node or queue may use them.",
        "Rename the node to a descriptive id (e.g. 'clean_rows') and update every routing field that referenced the old id.",
    ),
    (
        r"duplicate_node_id|Duplicate node ID",
        "Two nodes share the same id; node ids must be unique.",
        "Rename one of the nodes and update the routing fields that reference it.",
    ),
    (
        r"duplicate_output_name|Duplicate output name",
        "Two outputs share the same sink name; sink names must be unique.",
        "Rename one of the outputs and update any on_success/on_error/route that targets it.",
    ),
    (
        r"duplicate_edge_id|Duplicate edge ID",
        "Two edges share the same id.",
        "Edges are UI-only; drop the duplicate edge entry (runtime routing uses the connection fields, not edges).",
    ),
    (
        r"edge_unknown_node|references unknown node",
        "An edge names a from_node or to_node that does not exist in the pipeline.",
        "Edges are UI-only; drop the stale edge or fix its node reference — runtime routing uses the connection fields.",
    ),
    (
        r"duplicate_connection_producer|Duplicate producer for connection",
        "Two different components publish rows to the same connection name; every connection has exactly one producer.",
        "Rename one producer's on_success (or route/fork_to target) to a distinct connection name and point the intended "
        "consumer's input at it. To merge branches, use a coalesce node — not a shared connection name.",
    ),
    (
        r"duplicate_connection_consumer|Duplicate consumer for connection",
        "Two different nodes read the same connection as their input; every connection has exactly one consumer.",
        "Give each consumer its own input connection (fan out with a gate fork_to if both need the same rows).",
    ),
    (
        r"connection_sink_name_overlap|Connection names overlap with sink names|collides with a sink of the same name",
        "A connection name and a sink name are identical; connection and sink names must be disjoint so routing is unambiguous.",
        "Rename the connection (the node id or on_success value) so it no longer matches any outputs[].sink_name.",
    ),
    (
        r"gate_condition_invalid",
        "A gate's condition expression failed to parse or uses disallowed syntax.",
        "Set condition to a valid row expression (call get_expression_grammar for the syntax); a pure fan-out gate uses condition='True'.",
    ),
    (
        r"aggregation_missing_on_error|Aggregation '(.+)' is missing required field 'on_error'",
        "Every aggregation must declare where failed rows go.",
        "Set the aggregation's on_error — 'discard' is the simplest safe choice, or route to a quarantine sink name.",
    ),
    (
        r"aggregation_output_mode_invalid",
        "An aggregation's output_mode is not one of the allowed values.",
        "Set output_mode to 'passthrough' or 'transform' (or omit it for the default).",
    ),
    (
        r"batch_transform_misplaced",
        "A batch-aware plugin is configured as a row-level transform; batch plugins only run as aggregations.",
        "Re-emit the node with node_type='aggregation' and a trigger (e.g. trigger={'count': N}), or pick a row-level transform plugin.",
    ),
    (
        r"batch_required_fields_invalid",
        "A batch-aware aggregation's required_input_fields declaration is invalid for its plugin.",
        "Call get_plugin_schema('transform', <plugin>) for the exact option shapes and re-emit only the offending options.",
    ),
    (
        r"batch_value_field_not_numeric",
        "batch_distribution_profile.value_field must reference a numeric (int/float) field, but the upstream schema declares a non-numeric type.",
        "Point value_field at a numeric field, or use batch_top_k for categorical distributions.",
    ),
    (
        r"queue_config_invalid",
        "A queue node's shape is invalid — a queue's id must equal its input, with no plugin, routing, or options beyond a description.",
        "Re-emit the queue with id == input and only a description in options, or remove the queue node.",
    ),
    (
        r"queue_no_consumer",
        "A queue has no downstream consumer; a queue must feed exactly one ordinary node.",
        "Give one downstream node input='<queue id>', or remove the queue.",
    ),
    (
        r"queue_name_collision|collides with a source of the same name",
        "A queue's id collides with a source name; source and queue names must be globally unique.",
        "Rename the queue node (its id and the consumer's input) so it no longer matches any source name.",
    ),
    (
        r"aggregation_trigger_invalid|trigger is invalid",
        "An aggregation's trigger failed to parse.",
        "Set trigger to a valid shape, e.g. trigger={'count': N} to aggregate every N rows, or omit it to aggregate at end of source.",
    ),
    (
        r"web_scrape_http_identity_invalid",
        "web_scrape.http identity fields (abuse_contact, scraping_reason) carry a placeholder or undeliverable value; these ship "
        "as HTTP headers to the scraped host and must be real operator-supplied identity.",
        "Ask the operator for the deployment's abuse contact email and scraping reason, or leave the http block for the operator "
        "to fill — never invent an email or domain.",
    ),
)


# The closed validation codes the planner's redacted repair feedback can carry
# (see ``pipeline_planner._allowlisted_candidate_feedback``). Every entry MUST
# resolve through :func:`explain_validation_code` — the explain tool's fallback
# advertises this list and its fuzzy route scans for these substrings, so a
# dead entry would route the model to a code that then explains nothing
# (test_closed_code_catalogue_is_fully_explainable pins the invariant).
_CLOSED_VALIDATION_ERROR_CODES: Final[tuple[str, ...]] = (
    "unknown_node_type",
    "coalesce_on_success_must_be_sink",
    "coalesce_missing_policy",
    "coalesce_missing_branches",
    "coalesce_policy_invalid",
    "coalesce_merge_invalid",
    "transform_missing_on_success",
    "transform_missing_on_error",
    "transform_on_success_dangling",
    "transform_on_error_unknown_sink",
    "aggregation_on_success_dangling",
    "source_on_success_dangling",
    "gate_missing_condition",
    "gate_missing_routes",
    "gate_route_labels_mismatch",
    "fork_branch_no_destination",
    "pipeline_decision_unregistered",
    "interpretation_requirements_invalid",
    "plugin_options_invalid",
    # ── Schema-contract family (2026-07-22 codeless-rejection closure) ──────
    # Emitted with a SchemaContractDetail naming producer/consumer/fields so
    # the planner's redacted repair feedback carries actionable structure.
    "schema_contract_violation",
    "sink_contract_violation",
    "locked_input_extras",
    "sink_locked_extras",
    "transform_contract_violation",
    "semantic_contract_violation",
    "contract_config_invalid",
    # ── Structural rejections (same closure sweep) ──────────────────────────
    "no_source_configured",
    "no_sinks_configured",
    "source_name_invalid",
    "reserved_node_id",
    "duplicate_node_id",
    "duplicate_output_name",
    "duplicate_edge_id",
    "edge_unknown_node",
    "duplicate_connection_producer",
    "duplicate_connection_consumer",
    "connection_sink_name_overlap",
    "gate_condition_invalid",
    "aggregation_missing_plugin",
    "aggregation_missing_on_error",
    "aggregation_output_mode_invalid",
    "batch_transform_misplaced",
    "batch_required_fields_invalid",
    "batch_value_field_not_numeric",
    "queue_config_invalid",
    "queue_no_consumer",
    "queue_name_collision",
    "coalesce_branch_unreachable",
    "node_input_not_reachable",
    "coalesce_on_success_unknown_sink",
    "aggregation_trigger_invalid",
    "web_scrape_http_identity_invalid",
)


def _extract_validator_expected_hint(error_text: str) -> str | None:
    """Pull the ``Expected ...`` span out of a validator error string.

    Pydantic and our schema-spec validators frequently emit errors like
    ``"Field spec at index 0 is a dict with 2 keys. Expected single-key
    dict like {'field_name': 'type'} or a string like 'field_name: type'."``.
    The static catalogue fix below the substring discards that hint, so
    the model only sees ``"Use get_pipeline_state ... patch_source_options"``
    — which doesn't tell it what shape to actually emit. Returning the
    ``Expected ...`` span verbatim lets the caller append it to
    ``suggested_fix`` so the model can copy the shape directly.

    The hint terminates at the next sentence boundary (``.`` followed by
    whitespace or end-of-string) so a trailing ``"Got X. Other noise."``
    doesn't get swept up.
    """
    idx = error_text.find("Expected ")
    if idx == -1:
        return None
    rest = error_text[idx:]
    end = len(rest)
    for i, ch in enumerate(rest):
        if ch == "." and (i + 1 == len(rest) or rest[i + 1].isspace()):
            end = i + 1
            break
    return rest[:end].strip()


def _augment_with_expected_hint(fix: str, error_text: str) -> str:
    """Append the validator ``Expected ...`` hint to ``fix`` when present."""
    hint = _extract_validator_expected_hint(error_text)
    if hint is None:
        return fix
    return f"{fix} {hint}"


def explain_validation_code(code: str) -> tuple[str, str] | None:
    """Resolve a closed validation ``error_code`` to ``(explanation, suggested_fix)``.

    The one-shot pipeline planner's repair feedback strips raw validation
    messages before returning a rejection to the model — a redaction boundary,
    since a raw message can quote plugin names, option values, or row content
    (see ``pipeline_planner._allowlisted_candidate_feedback``). The closed
    ``error_code`` is the only signal that survives. The "Closed structural
    node-shape codes" entries in :data:`_VALIDATION_ERROR_PATTERNS` deliberately
    embed those codes as regex alternations precisely so the *code alone*
    resolves to the same guidance the ``explain_validation_error`` tool returns
    for the full message — this accessor is what lets the planner feedback carry
    that fix (e.g. "there is no 'fork' node_type; fork with a gate") without
    re-opening the message boundary.

    Returns ``None`` when no pattern matches, so callers attach nothing rather
    than a misleading generic. The ``_augment_with_expected_hint`` span is
    intentionally NOT applied: there is no error_text to mine an ``Expected …``
    hint from — only the bare code.
    """
    if type(code) is not str or not code:
        return None
    for pattern, explanation, fix in _VALIDATION_ERROR_PATTERNS:
        if re.search(pattern, code):
            return explanation, fix
    return None


def _execute_explain_validation_error(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Explain a validation error with human-readable diagnosis and fix."""
    validation = context.catalog.validate_composition_state(state).validation
    error_text = args["error_text"]
    for pattern, explanation, fix in _VALIDATION_ERROR_PATTERNS:
        if re.search(pattern, error_text):
            return ToolResult(
                success=True,
                updated_state=state,
                validation=validation,
                affected_nodes=(),
                data={
                    "error_text": error_text,
                    "explanation": explanation,
                    "suggested_fix": _augment_with_expected_hint(fix, error_text),
                },
            )
    # Fuzzy route: live planners pass junk like "ValidationError" or
    # "node:X ValidationError validation_error" instead of the message or the
    # bare code. A closed code buried in noise (any case) still resolves —
    # the direct pattern pass above is case-sensitive, so this catches e.g.
    # an upper-cased code echoed from a log line.
    lowered = error_text.lower()
    for code in _CLOSED_VALIDATION_ERROR_CODES:
        if code in lowered:
            guidance = explain_validation_code(code)
            if guidance is None:  # pragma: no cover — catalogue invariant is test-pinned
                continue
            explanation, fix = guidance
            return ToolResult(
                success=True,
                updated_state=state,
                validation=validation,
                affected_nodes=(),
                data={
                    "error_text": error_text,
                    "error_code": code,
                    "explanation": explanation,
                    "suggested_fix": _augment_with_expected_hint(fix, error_text),
                },
            )
    # No match at all — teach usage instead of an unhelpful generic. The codes
    # are a public static catalogue, so listing them leaks nothing.
    return ToolResult(
        success=True,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data={
            "error_text": error_text,
            "explanation": "The text does not match any known validation message or closed error_code.",
            "suggested_fix": _augment_with_expected_hint(
                "Call explain_validation_error with the exact error_code string from the rejection "
                "feedback's validation.errors[].error_code. Closed codes: " + ", ".join(_CLOSED_VALIDATION_ERROR_CODES) + ".",
                error_text,
            ),
            "known_codes": list(_CLOSED_VALIDATION_ERROR_CODES),
        },
    )


_EXPLAIN_VALIDATION_ERROR_DECLARATION = ToolDeclaration(
    name="explain_validation_error",
    handler=_execute_explain_validation_error,
    kind=ToolKind.DISCOVERY,
    description="Get a human-readable explanation of a validation error "
    "with suggested fixes. Pass the exact error text from a validation result.",
    json_schema={
        "type": "object",
        "properties": {
            "error_text": {
                "type": "string",
                "description": "The validation error message to explain.",
            },
        },
        "required": ["error_text"],
        "additionalProperties": False,
    },
    cacheable=True,
)


def _serialize_plugin_assistance_example(
    example: Any,
) -> dict[str, Any]:
    """Serialize a PluginAssistanceExample for LLM consumption.

    Mirrors the serialize-at-the-boundary pattern used by
    ``_semantic_edge_contract_to_payload`` (composer_mcp/server.py) and
    ``serialize_semantic_contracts`` (web/execution/_semantic_helpers.py):
    L0 contract types intentionally have no ``.to_dict()``; the rendering
    site owns the JSON shape so contracts stay free of encoding concerns.
    """
    return {
        "title": example.title,
        "before": deep_thaw(example.before) if example.before is not None else None,
        "after": deep_thaw(example.after) if example.after is not None else None,
    }


def _execute_get_plugin_assistance(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Return plugin-owned guidance for a source, transform, or sink.

    Dual-use by ``issue_code``:

    * ``issue_code is None`` (or absent) — discovery-time guidance. The
      plugin returns a one-line ``summary`` and ``composer_hints``
      (same surface that list_* and get_plugin_schema already carry).
    * ``issue_code is not None`` — failure-time guidance. The
      semantic validator emits ``requirement_code`` values like
      ``line_explode.source_field.line_framed_text``; the agent echoes
      that code in to retrieve ``suggested_fixes`` + example
      before/after configs.

    When the plugin has no assistance to publish, returns success with
    a "no assistance published" payload (summary=None, empty lists)
    rather than failing — the absence is itself a useful signal.

    Unknown plugin name or invalid plugin_type surfaces here as a tool
    failure with the original message so the agent can correct the call.
    """
    plugin_type_raw = args["plugin_type"]
    plugin_name = args["plugin_name"]
    # ``args`` is LLM tool-call arguments (Tier 3). ``plugin_type``/``plugin_name``
    # are required (json_schema ``required``) so direct subscript lets a KeyError
    # surface an LLM contract violation; ``issue_code`` is optional (discovery vs
    # failure mode), so its absence is recorded honestly as ``None`` via the
    # membership form rather than a defensive ``.get``.
    issue_code = args["issue_code"] if "issue_code" in args else None

    if plugin_type_raw not in ("source", "transform", "sink"):
        return _failure_result(
            state,
            f"Unknown plugin_type: {plugin_type_raw!r}. Must be one of: 'source', 'transform', 'sink'.",
        )
    plugin_type: PluginKind = plugin_type_raw

    policy_error = _validate_plugin_name(context, plugin_type, plugin_name)
    if policy_error is not None:
        return _plugin_policy_failure(state, policy_error)
    try:
        context.catalog.get_schema(plugin_type, plugin_name)
    except (ValueError, KeyError) as exc:
        return _failure_result(state, str(exc))

    manager = get_shared_plugin_manager()
    try:
        if plugin_type == "source":
            plugin_cls: Any = manager.get_source_by_name(plugin_name)
        elif plugin_type == "transform":
            plugin_cls = manager.get_transform_by_name(plugin_name)
        else:
            plugin_cls = manager.get_sink_by_name(plugin_name)
    except PluginNotFoundError as exc:
        return _failure_result(state, str(exc))

    assistance = plugin_cls.get_agent_assistance(issue_code=issue_code)

    if assistance is None:
        payload: dict[str, Any] = {
            "plugin_type": plugin_type,
            "plugin_name": plugin_name,
            "issue_code": issue_code,
            "summary": None,
            "suggested_fixes": [],
            "examples": [],
            "composer_hints": [],
        }
        return _discovery_result(state, payload)

    payload = {
        "plugin_type": plugin_type,
        "plugin_name": assistance.plugin_name,
        "issue_code": assistance.issue_code,
        "summary": assistance.summary,
        "suggested_fixes": list(assistance.suggested_fixes),
        "examples": [_serialize_plugin_assistance_example(ex) for ex in assistance.examples],
        "composer_hints": list(assistance.composer_hints),
    }
    return _discovery_result(state, payload)


_GET_PLUGIN_ASSISTANCE_DECLARATION = ToolDeclaration(
    name="get_plugin_assistance",
    handler=_execute_get_plugin_assistance,
    kind=ToolKind.DISCOVERY,
    description=(
        "Retrieve plugin-owned guidance for a source, transform, or sink. "
        "Two modes by ``issue_code``:\n"
        "  * Omit ``issue_code`` (or pass null) to get discovery-time guidance "
        "    — a summary of the plugin and composer_hints. (The same hints "
        "    are also carried on list_sources / list_transforms / list_sinks / "
        "    get_plugin_schema responses; this tool is the explicit path.)\n"
        "  * Pass an ``issue_code`` (validators emit these as requirement_code "
        "    on semantic_contracts entries) to get failure-time guidance — "
        "    summary, suggested_fixes, and example before/after configurations."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "plugin_type": {
                "type": "string",
                "enum": ["source", "transform", "sink"],
                "description": "Plugin family. 'source', 'transform', or 'sink'.",
            },
            "plugin_name": {
                "type": "string",
                "description": "Plugin name (e.g. 'csv', 'web_scrape', 'database').",
            },
            "issue_code": {
                "type": ["string", "null"],
                "description": (
                    "Optional. Stable issue identifier owned by the plugin "
                    "for failure-time guidance. Omit or pass null for "
                    "discovery-time guidance."
                ),
            },
        },
        "required": ["plugin_type", "plugin_name"],
        "additionalProperties": False,
    },
    cacheable=True,
)


def _execute_get_audit_info(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Return constant facts about the Landscape audit trail.

    Audit is mandatory (`LandscapeSettings` rejects `enabled=false` at
    config validation time) and the backend URL is operator-managed via
    `WebSettings.get_landscape_url()` — security fix S1, see
    `web/composer/yaml_generator.py:179`. Letting the composer set the
    audit backend would let a user prompt redirect the audit trail to an
    attacker DB, disable encryption, or split audit across stores.

    The returned payload is a constant — no `WebSettings` access — so
    operator-internal config (URL, backend type, encryption-key env var)
    never reaches the LLM context. The model paraphrases `summary`; it
    does not need the URL itself.
    """
    del context  # unused; signature uniformity with the other handlers.
    payload = {
        "enabled": True,
        "composer_modifiable": False,
        "summary": (
            "Landscape audit is mandatory and always on for every pipeline run. "
            "The audit backend (database type, location, encryption) is configured "
            "by the operator at deploy time and is intentionally NOT addressable "
            "from the composer — letting the composer set it would be a security "
            "regression (audit-DSN injection, encryption bypass, audit split-brain). "
            "When a user asks for 'audit logging', 'SQLite audit', or similar: "
            "acknowledge that audit is already enabled for every run, do NOT add a "
            "sink shape for it, and do NOT silently remove an audit-shaped node by "
            "treating it as 'unconnected'. To inspect past runs, point the user at "
            "the Landscape MCP forensic tools."
        ),
        "audit_export_summary": (
            "A separate optional feature ('landscape.export') can copy each run's "
            "audit data to an additional sink for offline review. This is also "
            "operator-configured and is not currently composer-controllable. If a "
            "user asks for 'export the audit data to a file', explain that this is "
            "an operator-side configuration and is not part of the pipeline the "
            "composer is building."
        ),
    }
    return _discovery_result(state, payload)


_GET_AUDIT_INFO_DECLARATION = ToolDeclaration(
    name="get_audit_info",
    handler=_execute_get_audit_info,
    kind=ToolKind.DISCOVERY,
    description=(
        "Return facts about ELSPETH's Landscape audit trail. Call this BEFORE "
        "answering any user question that mentions audit logging, audit "
        "database, SQLite/Postgres audit, audit backend, audit export, "
        "Landscape, or 'how do I record what the pipeline did'. Audit is "
        "mandatory and operator-managed; the composer cannot configure the "
        "backend (security boundary — see yaml_generator.py:179, fix S1). "
        "Returns enabled status, composer_modifiable flag, and a canonical "
        "summary to paraphrase. Does NOT return the audit URL/path/DSN — "
        "that is operator-internal and intentionally not surfaced to the LLM."
    ),
    json_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    cacheable=True,
)


def _execute_list_models(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """List available LLM model identifiers.

    Without a provider filter, returns provider names and model counts
    to avoid dumping hundreds of entries. With a provider filter,
    returns matching model IDs capped at ``limit``.

    For ``provider="openrouter/"`` the slugs are read from the live
    OpenRouter catalog primed at boot (or the bundled litellm fallback
    when the primer hasn't run / is offline) via
    :func:`get_catalog_values` against :data:`MODEL_CATALOG_OPENROUTER` —
    the same catalog accessor the value-source compliance walker
    consults at preflight. Sharing one source of truth between the
    composer's discovery surface and the validator closes the
    "composer recommended a model the validator then rejects" loop
    that previously surfaced as ``ValueSourceValidationError`` on
    tutorial runs. Non-OpenRouter providers continue to be served from
    the bundled litellm list because no per-provider live primer exists
    yet.
    """
    del context  # unused; signature uniformity with the other handlers.
    all_models: list[str] = list(read_litellm_model_list())

    # ``args`` is LLM tool-call arguments (Tier 3). Both ``provider`` and
    # ``limit`` are optional (json_schema ``required: []``). ``provider``
    # absence is recorded honestly as ``None`` (membership form). ``limit``
    # absence resolves to the documented default of 50 (the json_schema
    # description states "default 50") — a meaning-preserving substitution, not
    # fabrication. The scalar type checks use ``type() is`` so a bool ``limit``
    # (``isinstance(True, int)`` is True) is correctly rejected at the boundary.
    provider = args["provider"] if "provider" in args else None
    limit = args["limit"] if "limit" in args else 50
    if type(limit) is not int or limit < 1:
        limit = 50

    if provider is not None and type(provider) is str:
        normalised = provider.rstrip("/")
        if normalised == OPENROUTER_LITELLM_PREFIX.rstrip("/"):
            # Live-catalog read returns un-prefixed slugs already (e.g.
            # ``anthropic/claude-sonnet-4.5``) — exactly the form the
            # OpenRouterLLMProvider config field ``model`` expects and
            # the value-source walker compares against. No prefix strip
            # needed; the litellm fallback path inside
            # ``get_catalog_values`` already strips ``openrouter/`` from
            # its bundled list.
            filtered = sorted(get_catalog_values(MODEL_CATALOG_OPENROUTER))
        elif provider == "":
            # Empty string means "models without a provider prefix"
            filtered = [m for m in all_models if "/" not in m]
        else:
            filtered = [m for m in all_models if m.startswith(provider)]
        truncated = len(filtered) > limit
        data: dict[str, Any] = {
            "models": filtered[:limit],
            "count": len(filtered),
            "truncated": truncated,
        }
    else:
        # Group by provider prefix to avoid token waste. ``providers`` is our
        # own freshly-constructed accumulator dict, not a Tier-3 boundary, so
        # the offensive-programming rule rejects ``providers.get(prefix, 0)``
        # as a defensive read on data we wrote. Initialize the slot
        # explicitly before increment instead. (``Counter`` would also work
        # but introduces an import-shift that cascades fingerprint rotations
        # across this whole module — keep the diff local.)
        providers: dict[str, int] = {}
        for m in all_models:
            prefix = m.split("/", 1)[0] if "/" in m else ""
            if prefix not in providers:
                providers[prefix] = 0
            providers[prefix] += 1
        # Replace the bundled-litellm openrouter count with the live
        # catalog size so the unfiltered summary advertises the same
        # numbers a follow-up ``provider="openrouter/"`` filter will
        # return. The litellm bundled list and the live OpenRouter
        # catalog drift apart as Anthropic / OpenAI rotate model slugs;
        # advertising the bundled count would invite the composer to
        # request a "known" model that the validator then rejects.
        openrouter_key = OPENROUTER_LITELLM_PREFIX.rstrip("/")
        live_openrouter = get_catalog_values(MODEL_CATALOG_OPENROUTER)
        if openrouter_key in providers:
            bundled = providers[openrouter_key]
            providers[openrouter_key] = len(live_openrouter)
            total_models = len(all_models) - bundled + len(live_openrouter)
        else:
            providers[openrouter_key] = len(live_openrouter)
            total_models = len(all_models) + len(live_openrouter)
        data = {
            "providers": providers,
            "total_models": total_models,
            "hint": "Use provider parameter to list models for a specific provider. An empty string key means models without a provider prefix.",
        }

    return _discovery_result(state, data)


_LIST_MODELS_DECLARATION = ToolDeclaration(
    name="list_models",
    handler=_execute_list_models,
    kind=ToolKind.DISCOVERY,
    description="List available LLM model identifiers. Without a provider "
    "filter, returns provider names and counts. With a provider filter, "
    "returns matching model IDs (capped at limit). For provider='openrouter/' "
    "the returned slugs are normalised to OpenRouter's HTTP API form "
    "(without the litellm-internal 'openrouter/' routing prefix) — these "
    "are the values to put directly in `model:`.",
    json_schema={
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "description": "Provider prefix to filter by (e.g. 'openrouter/', 'azure/'). "
                "Omit to get a provider summary instead of individual models.",
            },
            "limit": {
                "type": "integer",
                "description": "Max models to return (default 50).",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    cacheable=True,
)


_BLOCKING_DIAGNOSTIC_CODES: Final[frozenset[str]] = frozenset(
    {
        "aggregation_numeric_value_field_type_mismatch_against_source_schema",
        "csv_duplicate_headers",
        "csv_fixed_schema_omits_observed_columns",
        "csv_source_blob_header_mismatch",
        "csv_source_field_resolution_error",
        "gate_expression_type_mismatch_against_source_schema",
        "text_source_url_without_web_scrape",
        "source_inspection_failed",
    }
)


class _BlockingDiagnosticPayload(TypedDict):
    code: str
    severity: Literal["blocking"]
    message: str
    suggested_repair: str
    evidence_locator: Mapping[str, Any]


def _blocking_diagnostic(
    *,
    code: str,
    message: str,
    suggested_repair: str,
    evidence_locator: Mapping[str, Any],
) -> _BlockingDiagnosticPayload:
    """Construct a blocking diagnostic dict and assert the code is registered.

    Offensive: a contributor who adds a new blocker without registering it in
    ``_BLOCKING_DIAGNOSTIC_CODES`` (and the matching skill-markdown vocabulary)
    crashes immediately rather than shipping an unrecognised code into the
    audit trail and the LLM's repair-message context.
    """
    if code not in _BLOCKING_DIAGNOSTIC_CODES:
        raise AssertionError(
            f"blocking diagnostic code {code!r} is not registered in "
            f"_BLOCKING_DIAGNOSTIC_CODES. Add it there (and to the skill-"
            f"markdown vocabulary the composer LLM consumes) before emitting "
            f"a blocking diagnostic with this code."
        )
    return {
        "code": code,
        "severity": "blocking",
        "message": message,
        "suggested_repair": suggested_repair,
        "evidence_locator": evidence_locator,
    }


def _csv_source_field_resolution_error_diagnostic(
    *,
    blob_id: object,
    facts: SourceInspectionFacts,
    exc: ValueError,
) -> _BlockingDiagnosticPayload:
    return _blocking_diagnostic(
        code="csv_source_field_resolution_error",
        message=(
            "CSV source header resolution failed before proof diagnostics could compare "
            "declared fields to observed headers. CSVSource would reject this shape at "
            "runtime, so preview_pipeline is blocking it for repair. The raw resolver "
            "error text and observed header values are withheld: header-resolution "
            "failure means a headerless or malformed CSV can make the first data row "
            "look like headers, so those values are treated as potential row content."
        ),
        suggested_repair=(
            "Rename colliding or invalid CSV headers, correct field_mapping keys and "
            "values to match normalized source headers, or declare explicit `columns` "
            "for headerless input, then re-run preview_pipeline."
        ),
        evidence_locator={
            "source": "blob",
            "blob_id": str(blob_id),
            "error_class": type(exc).__name__,
            "observed_header_count": len(facts.observed_headers or ()),
            "observed_headers_redacted": True,
        },
    )


def _csv_source_schema_config_error_diagnostic(
    *,
    blob_id: object,
    facts: SourceInspectionFacts,
    exc: ValueError,
) -> _BlockingDiagnosticPayload:
    """Distinct repair guidance for a ``schema.*``-block parse failure.

    Shares the registered ``csv_source_field_resolution_error`` code (the
    composer LLM's repair vocabulary and the skill markdown are keyed on that
    code; adding a new code would silently half-wire the LLM contract), but the
    repair text points at the ``schema.*`` knob rather than at header /
    field_mapping / columns resolution. A failure from ``get_raw_schema_config``
    is a malformed *schema declaration* (bad ``schema.mode``, a malformed field
    spec, a non-bool required flag), NOT a header-resolution problem — so the
    generic field-resolution repair text would misdirect the operator and the
    LLM. The raw ``{exc}`` is NOT interpolated: it can quote observed header /
    field values that, under a failed header resolution, may be row content; the
    exception class is surfaced structurally instead and the repair text points
    at the ``schema.*`` knob.
    """
    return _blocking_diagnostic(
        code="csv_source_field_resolution_error",
        message=(
            "CSV source schema declaration failed to parse before proof diagnostics "
            "could compare declared fields to observed headers. CSVSource would reject "
            "this schema at runtime, so preview_pipeline is blocking it for repair. The "
            "raw parser error text and observed header values are withheld (they may "
            "carry row content for a headerless or malformed CSV)."
        ),
        suggested_repair=(
            "Correct the source `schema` block: `schema.mode` must be one of "
            "'fixed'/'flexible'/'observed', each entry in `schema.fields` must be a "
            "valid field spec (a 'name: type' string or a mapping with a str name and a "
            "bool required flag), then re-run preview_pipeline."
        ),
        evidence_locator={
            "source": "blob",
            "blob_id": str(blob_id),
            "error_class": type(exc).__name__,
            "observed_header_count": len(facts.observed_headers or ()),
            "observed_headers_redacted": True,
        },
    )


def _source_schema_mode(source: SourceSpec) -> str | None:
    # ``source.options`` is composer/user-authored config re-read from persisted
    # session state — Tier-3 origin (we authored the container, they authored the
    # values), so absence is recorded as ``None`` and shape is validated, never
    # coerced. Membership form (not ``.get``) makes the honest absence→None
    # explicit at the boundary.
    schema = source.options["schema"] if "schema" in source.options else None
    if not isinstance(schema, Mapping):
        return None
    mode = schema["mode"] if "mode" in schema else None
    if type(mode) is not str:
        return None
    return mode.strip().lower()


def _sample_csv_rows(content: bytes, *, filename: str, max_rows: int = 100) -> tuple[dict[str, str], ...]:
    text = content[: 8 * 1024].decode("utf-8", errors="replace")
    delimiter = "\t" if filename.lower().endswith(".tsv") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader):
        if index >= max_rows:
            break
        rows.append({key: value for key, value in row.items() if type(key) is str and value is not None})
    return tuple(rows)


def _row_fields_referenced_by_condition(condition: str) -> tuple[str, ...]:
    tree = ast.parse(condition, mode="eval")
    fields: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "row"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            fields.append(node.slice.value)
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "row"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            fields.append(node.args[0].value)
    return tuple(dict.fromkeys(fields))


def _gate_expression_type_diagnostics_for_observed_csv(
    state: CompositionState,
    source: SourceSpec,
    *,
    blob_id: str,
    filename: str,
    content: bytes,
) -> list[Mapping[str, Any]]:
    """Evaluate direct source-fed gates against sampled observed CSV rows.

    Observed CSV sources emit raw strings because there are no declared field
    types to coerce against. A gate such as ``row['amount'] >= 1000`` is
    syntactically valid but fails at runtime when the evaluator compares
    ``str`` with ``int``. This preview proof step uses the same expression
    evaluator against bounded raw rows and reports the type mismatch without
    surfacing row values.
    """
    if _source_schema_mode(source) != "observed":
        return []

    rows = _sample_csv_rows(content, filename=filename)
    if not rows:
        return []

    diagnostics: list[Mapping[str, Any]] = []
    direct_gate_nodes = (
        node for node in state.nodes if node.node_type == "gate" and node.input == source.on_success and node.condition is not None
    )
    for node in direct_gate_nodes:
        condition = node.condition
        if condition is None:
            continue
        if _validate_gate_expression(condition) is not None:
            continue
        parser = ExpressionParser(condition)
        fields = _row_fields_referenced_by_condition(condition)
        field = fields[0] if fields else None
        for row_index, row in enumerate(rows):
            try:
                parser.evaluate(row)
            except ExpressionEvaluationError:
                diagnostics.append(
                    _blocking_diagnostic(
                        code="gate_expression_type_mismatch_against_source_schema",
                        message=(
                            f"Gate '{node.id}' condition {condition!r} fails against sampled observed CSV "
                            "rows before runtime. Observed CSV source values are strings unless the "
                            "source schema declares explicit field types. (The raw evaluation error is "
                            "withheld: it can quote sampled row values and the observed field/key set.)"
                        ),
                        suggested_repair=(
                            "Patch the source schema to declare the compared field with an explicit numeric "
                            "type, for example schema.mode='fixed' or 'flexible' with schema.fields including "
                            f"{field + ': int' if field is not None else '<field>: int'}, then re-run preview_pipeline."
                        ),
                        evidence_locator={
                            "source": "blob",
                            "blob_id": str(blob_id),
                            "node_id": node.id,
                            "field": field,
                            "fields": list(fields),
                            "sample_row_index": row_index,
                            "source_schema_mode": "observed",
                        },
                    )
                )
                break
    return diagnostics


_NUMERIC_VALUE_FIELD_AGGREGATION_PLUGINS: Final[frozenset[str]] = frozenset(
    {
        "batch_distribution_profile",
        "batch_outlier_annotator",
        "batch_stats",
        "batch_threshold_summary",
    }
)


def _value_transform_preserves_field(node: NodeSpec, field_name: str) -> bool:
    # ``node.options`` is composer/user-authored config re-read from persisted
    # session state — Tier-3 origin. Membership form (not ``.get``) records the
    # honest absence→None at the boundary; the subtype guards below remain
    # because frozen config arrives as MappingProxyType/tuple subtypes.
    operations = node.options["operations"] if "operations" in node.options else None
    if not isinstance(operations, (list, tuple)):
        return False
    for operation in operations:
        if not isinstance(operation, Mapping):
            return False
        target = operation["target"] if "target" in operation else None
        if target == field_name:
            return False
    return True


def _source_field_reaches_connection_without_type_change(
    state: CompositionState,
    connection_name: str,
    *,
    field_name: str,
) -> bool:
    """Return True when a source field flows to a connection unchanged.

    This intentionally recognises only field-preserving nodes. Unknown
    transforms may coerce, overwrite, delete, or synthesize the field, so the
    proof step abstains instead of emitting a false positive.
    """
    resolver = ProducerResolver.build(
        source=None,
        sources=state.sources,
        nodes=state.nodes,
        sink_names=frozenset(output.name for output in state.outputs),
    )
    current = connection_name
    visited: set[str] = set()
    while True:
        if current in visited:
            return False
        visited.add(current)

        producer = resolver.find_producer_for(current)
        if producer is None:
            return False
        if is_source_producer_id(producer.producer_id):
            return True

        node = resolver.get_node(producer.producer_id)
        if node is None:
            return False
        if node.node_type == "gate":
            current = node.input
            continue
        if node.plugin == "value_transform" and _value_transform_preserves_field(node, field_name):
            current = node.input
            continue
        if node.plugin == "passthrough":
            current = node.input
            continue
        return False


def _numeric_aggregation_diagnostics_for_observed_csv(
    state: CompositionState,
    source: SourceSpec,
    *,
    blob_id: str,
    inferred_types: Mapping[str, str] | None,
    observed_headers: tuple[str, ...] | None,
) -> list[Mapping[str, Any]]:
    """Block observed CSV strings before numeric aggregation runtime failures."""
    if _source_schema_mode(source) != "observed" or observed_headers is None:
        return []

    observed_header_set = set(observed_headers)
    diagnostics: list[Mapping[str, Any]] = []
    for node in state.nodes:
        if node.node_type != "aggregation" or node.plugin not in _NUMERIC_VALUE_FIELD_AGGREGATION_PLUGINS:
            continue
        options, _owner = get_aggregation_contract_options(node.options, owner=f"node:{node.id}")
        # ``options`` is the external-origin (composer/user-authored) node
        # options Mapping returned by the contract helper — Tier-3. Membership
        # form records honest absence→None; the following ``type() is`` check is
        # the boundary validation of the scalar value.
        value_field = options["value_field"] if "value_field" in options else None
        if type(value_field) is not str or not value_field.strip():
            continue
        value_field = value_field.strip()
        if value_field not in observed_header_set:
            continue
        if not _source_field_reaches_connection_without_type_change(state, node.input, field_name=value_field):
            continue

        # ``inferred_types`` is our own Tier-2 derived inspection output. When
        # present it is built by ``_inspect_csv`` as ``{h: ... for h in
        # headers}`` against the SAME ``headers`` tuple it returns as
        # ``observed_headers`` — i.e. it carries exactly one entry per observed
        # header, never omitting one (``_merge_types`` always yields a type).
        # The whole map can legitimately be absent (None) for an empty/parse-
        # error blob, so that absence is handled explicitly. But ``value_field``
        # was already confirmed present in ``observed_header_set`` above, so a
        # non-None map is GUARANTEED to contain it: a missing key would be a
        # Tier-2 invariant break in our own inspection output, which must crash
        # (subscript), not be silently coerced to None by ``.get()``.
        inferred_type = inferred_types[value_field] if inferred_types is not None else None
        diagnostics.append(
            _blocking_diagnostic(
                code="aggregation_numeric_value_field_type_mismatch_against_source_schema",
                message=(
                    f"Aggregation '{node.id}' ({node.plugin}) uses numeric value_field '{value_field}', "
                    "but it is flowing from an observed CSV source. Observed CSV source values are strings "
                    "unless the source schema declares explicit field types or an upstream type_coerce node "
                    "converts the field before aggregation."
                ),
                suggested_repair=(
                    "Patch the source schema to declare the aggregated field with an explicit numeric type "
                    f"(for example {value_field}: float), or insert a type_coerce node upstream of the aggregation. "
                    "If the field is categorical and you want counts/frequencies, use batch_top_k instead of a "
                    "numeric aggregation."
                ),
                evidence_locator={
                    "source": "blob",
                    "blob_id": str(blob_id),
                    "node_id": node.id,
                    "plugin": node.plugin,
                    "field": value_field,
                    "observed_type": "str",
                    "inferred_sample_type": inferred_type or "unknown",
                    "source_runtime_type": "str",
                    "source_schema_mode": "observed",
                },
            )
        )

    return diagnostics


def compute_proof_diagnostics(
    state: CompositionState,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> list[Mapping[str, Any]]:
    """Compute machine-readable proof diagnostics for a composer state.

    Promotes ``preview_pipeline`` from a "state validates" check into a
    "state is plausibly runnable against observed input" proof. Returns a
    machine-readable list of diagnostics — each entry has::

        {
            "code": "csv_fixed_schema_omits_observed_columns",
            "severity": "blocking" | "warning" | "info",
            "message": "human-readable description",
            "suggested_repair": "tool/options the LLM should call",
            "evidence_locator": {"source": "...", "node_id": "...", ...},
        }

    Diagnostics surfaced:

      * ``csv_fixed_schema_omits_observed_columns`` — fixed CSV schema +
        on_validation_failure=discard + at least one observed column
        absent from declared fields. The combination silently discards
        every row, which is the #1 historical convergence-failure mode.
      * ``csv_source_blob_header_mismatch`` — CSV source with required
        declared fields but no overlap with the parsed blob header. This
        catches headerless CSV/list inputs before the first row is consumed
        as a header and every data row is discarded.
      * ``csv_source_field_resolution_error`` — CSVSource-style header
        normalization or field_mapping resolution failed before schema
        comparison could proceed.
      * ``text_source_url_without_web_scrape`` — text source whose blob
        content is a single URL but no web_scrape node downstream. The
        URL string itself reaches sinks instead of the URL's content.
      * ``gate_expression_type_mismatch_against_source_schema`` — observed
        CSV source values are still strings, and a direct source-fed gate
        condition fails when evaluated against sampled rows before runtime.
      * ``aggregation_numeric_value_field_type_mismatch_against_source_schema`` —
        observed CSV strings flow unchanged into a numeric aggregation
        ``value_field`` before runtime can reject the batch.
      * ``source_inspection_warning`` — every warning surfaced by
        ``inspect_blob_content`` is mirrored here at ``info`` severity
        so the model sees them in the same array as blocking issues.

    Bounded I/O: at most one blob read per call, bounded by
    ``inspect_blob_content``'s 8 KiB / 100 row caps.

    No-op (returns an empty list) if the source is not blob-backed or
    if session context is absent.
    """
    diagnostics: list[Mapping[str, Any]] = []

    source = state.sources["source"] if "source" in state.sources else None
    blob_id: Any | None = None
    if source is not None and "blob_ref" in source.options:
        blob_id = source.options["blob_ref"]
    if blob_id is None:
        for candidate_source in state.sources.values():
            if "blob_ref" in candidate_source.options:
                source = candidate_source
                blob_id = candidate_source.options["blob_ref"]
                break
    if source is None:
        return diagnostics

    # Only blob-backed sources are inspectable from preview_pipeline; for
    # path-based sources we have no bytes to peek at. ``source.options`` is
    # composer/user-authored config re-read from persisted session state —
    # Tier-3 origin, so the absence of a ``blob_ref`` is recorded as ``None``
    # (membership form, not ``.get``) and the caller treats None as "not
    # blob-backed", returning the empty diagnostics list.
    blob_id = source.options["blob_ref"] if "blob_ref" in source.options else None
    if blob_id is None or session_engine is None or session_id is None:
        return diagnostics

    blob = _sync_get_blob(session_engine, str(blob_id), session_id)
    # ``blob`` is a BlobToolRecord (TypedDict produced by
    # ``_blob_row_to_tool_dict`` from a validated blobs row). Direct
    # subscript access is mandatory — a missing key is a Tier-1
    # contract violation in our own dict shape, not external data.
    if blob is None or blob["status"] != "ready":
        return diagnostics

    storage_path = Path(blob["storage_path"])
    if not storage_path.exists():
        diagnostics.append(
            _blocking_diagnostic(
                code="source_inspection_failed",
                message=(f"Source blob '{blob_id}' storage file is missing — pipeline cannot run until the blob is re-uploaded."),
                suggested_repair="create_blob with the original content and re-wire via set_source_from_blob",
                evidence_locator={"source": "blob", "blob_id": str(blob_id)},
            )
        )
        return diagnostics

    # Tier 1 (our data, our file): an OSError between exists() and
    # read_bytes() is a real anomaly (concurrent delete, fs corruption,
    # permission revocation). Per CLAUDE.md offensive-programming
    # policy, let it propagate so the operator sees an informative
    # exception rather than a synthesised soft-degraded diagnostic
    # that could let downstream act on absent bytes.
    content = storage_path.read_bytes()

    # Tier 1 integrity verification — same shared helper as the two
    # other composer-tool blob readers. Without this, the proof step
    # would feed unverified bytes into ``inspect_blob_content`` and
    # repair-loop, undermining the audit trail's "decisions made on
    # verified inputs" invariant.
    _verify_blob_content_integrity(blob, content)

    facts = inspect_blob_content(
        content=content,
        filename=blob["filename"],
        mime_type=blob["mime_type"],
        content_hash=blob["content_hash"],
    )
    if source.plugin == "csv":
        # ``source.options`` is composer/user-authored config re-read from
        # persisted session state — Tier-3 origin (see the long note on the
        # ``get_raw_schema_config`` catch below). ``_csv_source_delimiter`` /
        # ``_csv_source_skip_rows`` / ``_csv_source_columns`` each re-validate a
        # persisted external-origin option and raise a raw ``ValueError`` on a
        # malformed value (non-single-char delimiter, non-int/negative skip_rows,
        # non-sequence/non-str columns). ``compute_proof_diagnostics`` is called
        # UNWRAPPED from ``_execute_preview_pipeline`` and the forced-repair gate,
        # and the tool dispatcher only catches ``ToolArgumentError`` — so an
        # unhandled ``ValueError`` here would escape and crash the preview_pipeline
        # tool over recoverable bad external input. Wrap the whole inspect call,
        # turn the failure into a per-blob blocking diagnostic (quarantine-
        # equivalent), and early-return: every downstream diagnostic section
        # (schema-mismatch, gate/aggregation type checks, text-URL, inspection
        # warnings) depends on ``facts``, which we could not compute, so there is
        # no independent diagnostic left to produce for this blob. The early
        # return also makes the later ``_csv_source_columns(source.options)`` call
        # safe-by-construction — reaching it means ``columns`` already parsed here.
        try:
            facts = inspect_csv_source_content(
                content=content,
                filename=blob["filename"],
                mime_type=blob["mime_type"],
                delimiter=_csv_source_delimiter(source.options),
                skip_rows=_csv_source_skip_rows(source.options),
                columns=_csv_source_columns(source.options),
                content_hash=blob["content_hash"],
            )
        except ValueError as exc:
            diagnostics.append(
                _csv_source_field_resolution_error_diagnostic(
                    blob_id=blob_id,
                    facts=facts,
                    exc=exc,
                )
            )
            return diagnostics

    # 1. Fixed CSV schema omits observed columns + discard => silent all-row drop.
    if facts.source_kind in {"csv", "json", "jsonl"}:
        # ``source.options`` is composer/user-authored configuration re-read
        # from persisted session state — Tier-3 origin, not Tier-1. We authored
        # the container (``SourceSpec`` / the options Mapping) and the schema
        # that validated it at load time, but the *values* were authored by the
        # operator or the composer LLM. Persistence does not promote that to
        # Tier-1: the store is mutable (hand-edits, stale restore) and the
        # validating contract can drift between write and read, so re-reading it
        # here is a FRESH Tier-3 boundary, not a Tier-1 invariant check. A
        # ``ValueError`` from ``get_raw_schema_config`` is therefore recoverable
        # bad external-origin input — exactly like a malformed source row — and
        # is caught and turned into a per-blob blocking diagnostic
        # (quarantine-equivalent: record what we found, skip the dependent
        # schema-mismatch analysis for this blob) rather than being allowed to
        # escape and crash the preview_pipeline tool. This mirrors the
        # ``_csv_source_field_mapping`` try/except below. The typed
        # ``SchemaConfig`` it returns replaces the prior raw-dict probing of
        # mode/fields; declared field specs are bridged back into the spec form
        # ``derive_*_risk`` consumes via ``to_dict()`` so the typed field
        # names/required flags flow through unchanged.
        try:
            schema_config = get_raw_schema_config(source.options, owner=f"source:{source.plugin}")
        except ValueError as exc:
            diagnostics.append(
                _csv_source_schema_config_error_diagnostic(
                    blob_id=blob_id,
                    facts=facts,
                    exc=exc,
                )
            )
            # Parse failed and the diagnostic is the record. Fall through with a
            # ``None`` schema_config so the existing ``is not None`` guard skips
            # the schema-mismatch analysis for this blob, mirroring how
            # ``field_resolution_failed`` short-circuits dependent work below.
            schema_config = None
        if schema_config is not None and schema_config.mode in {"fixed", "flexible"}:
            declared: tuple[Mapping[str, Any], ...] = tuple(schema_config.to_dict()["fields"] or ())
            headerless_columns = source.plugin == "csv" and _csv_source_columns(source.options) is not None
            field_mapping: dict[str, str] | None = None
            field_resolution_failed = False
            if source.plugin == "csv" and not headerless_columns:
                try:
                    field_mapping = _csv_source_field_mapping(source.options)
                except ValueError as exc:
                    diagnostics.append(
                        _csv_source_field_resolution_error_diagnostic(
                            blob_id=blob_id,
                            facts=facts,
                            exc=exc,
                        )
                    )
                    field_resolution_failed = True

            missing_declared: tuple[str, ...] = ()
            if not field_resolution_failed and not headerless_columns and facts.source_kind == "csv":
                try:
                    missing_declared = derive_required_header_mismatch_risk(
                        facts,
                        declared,
                        explicit_required_fields=schema_config.required_fields or (),
                        field_mapping=field_mapping,
                    )
                except ValueError as exc:
                    diagnostics.append(
                        _csv_source_field_resolution_error_diagnostic(
                            blob_id=blob_id,
                            facts=facts,
                            exc=exc,
                        )
                    )
                    field_resolution_failed = True
            if missing_declared and source.on_validation_failure == "discard":
                observed_header_count = len(facts.observed_headers or ())
                diagnostics.append(
                    _blocking_diagnostic(
                        code="csv_source_blob_header_mismatch",
                        message=(
                            f"CSV source declares required field(s) {list(missing_declared)} "
                            f"but the bound blob's parsed header has {observed_header_count} column(s) "
                            "with no overlapping field names. Header values are redacted because "
                            "headerless CSV input can make the first data row look like headers. "
                            "Every row will fail validation; "
                            "with on_validation_failure='discard', the run will terminate empty. "
                            "Either prepend a header row containing the declared field names or set "
                            "source options.columns to those names for headerless CSV input."
                        ),
                        suggested_repair=(
                            "For headered CSV, update the blob so line 1 contains the declared "
                            "schema field names. For headerless CSV, patch_source_options with "
                            "`columns` set to the declared field names, then re-run preview_pipeline. "
                            "See pipeline_composer.md rule 10."
                        ),
                        evidence_locator={
                            "source": "blob",
                            "blob_id": str(blob_id),
                            "declared_required_fields": list(missing_declared),
                            "observed_header_count": observed_header_count,
                            "observed_headers_redacted": True,
                            "source_plugin": source.plugin,
                        },
                    )
                )
            elif schema_config.mode == "fixed" and not field_resolution_failed:
                missing: tuple[str, ...] = ()
                if not headerless_columns:
                    try:
                        missing = derive_extra_column_risk(
                            facts,
                            declared,
                            field_mapping=field_mapping if source.plugin == "csv" else None,
                        )
                    except ValueError as exc:
                        diagnostics.append(
                            _csv_source_field_resolution_error_diagnostic(
                                blob_id=blob_id,
                                facts=facts,
                                exc=exc,
                            )
                        )
                    if missing and source.on_validation_failure == "discard":
                        diagnostics.append(
                            _blocking_diagnostic(
                                code="csv_fixed_schema_omits_observed_columns",
                                message=(
                                    f"Source schema is mode=fixed but {len(missing)} observed column(s) "
                                    "are not declared in schema.fields. Combined with "
                                    "on_validation_failure='discard', every row will be dropped because "
                                    "each contains an undeclared column. (Observed and missing column "
                                    "values are withheld: an observed-mode or headerless CSV can make a "
                                    "data row look like column headers.)"
                                ),
                                suggested_repair=(
                                    "patch_source_options with schema.mode='flexible' to accept extra "
                                    "columns, OR add the missing columns to schema.fields, OR set "
                                    "on_validation_failure to a configured output for inspection."
                                ),
                                evidence_locator={
                                    "source": "blob",
                                    "blob_id": str(blob_id),
                                    "missing_column_count": len(missing),
                                    "observed_column_count": len(facts.observed_headers or ()),
                                    "observed_columns_redacted": True,
                                },
                            )
                        )

    # 2. Observed CSV + numeric gate predicate => preview/runtime agreement gap.
    if facts.source_kind == "csv":
        diagnostics.extend(
            _gate_expression_type_diagnostics_for_observed_csv(
                state,
                source,
                blob_id=str(blob_id),
                filename=blob["filename"],
                content=content,
            )
        )
        diagnostics.extend(
            _numeric_aggregation_diagnostics_for_observed_csv(
                state,
                source,
                blob_id=str(blob_id),
                inferred_types=facts.inferred_types,
                observed_headers=facts.observed_headers,
            )
        )

    # 3. Text source containing a single URL but no web_scrape downstream.
    if facts.source_kind == "text" and facts.url_candidates:
        node_plugins = {(n.plugin or "").lower() for n in state.nodes}
        if "web_scrape" not in node_plugins:
            diagnostics.append(
                _blocking_diagnostic(
                    code="text_source_url_without_web_scrape",
                    message=(
                        f"Source blob contains URL(s) {list(facts.url_candidates)} but no "
                        "web_scrape transform is wired downstream. The URL string itself will "
                        "flow to sinks, not the URL's content."
                    ),
                    suggested_repair=(
                        "upsert_node({node_type: 'transform', plugin: 'web_scrape', "
                        "input: <source on_success>, options: {url_field: '<column>'}}) and route "
                        "the source on_success to it."
                    ),
                    evidence_locator={
                        "source": "blob",
                        "blob_id": str(blob_id),
                        "url_candidates": list(facts.url_candidates),
                    },
                )
            )

    # 4. Surface inspection warnings as info-severity diagnostics so the model
    #    sees them in the same array as blocking issues. These are *advisory*
    #    only — the model can ignore them if the operator's intent justifies.
    #
    #    Exception: ``csv_duplicate_headers`` is promoted to blocking. Duplicate
    #    headers cause silent column collapse in csv.DictReader (last-write-
    #    wins) and similar libraries, fabricating a single column from multiple
    #    source columns. That is a Tier-1 audit-integrity violation — the
    #    audit trail would silently contain data that "looks single-column"
    #    when the source had two — and must force the repair loop, not pass
    #    through as advisory. The repair vocabulary is: rename headers,
    #    declare ``columns`` explicitly, configure ``field_mapping``, or set
    #    ``on_validation_failure`` to a configured quarantine output.
    for warning in facts.warnings:
        if warning.startswith("csv_duplicate_headers:"):
            diagnostics.append(
                _blocking_diagnostic(
                    code="csv_duplicate_headers",
                    message=warning,
                    suggested_repair=(
                        "Rename the duplicate header(s) at the source, OR declare "
                        "explicit `columns` in the source options, OR configure "
                        "`field_mapping` to disambiguate the collapsed names, OR "
                        "set `on_validation_failure` to a configured quarantine "
                        "output so the silent column collapse does not poison the "
                        "audit trail."
                    ),
                    evidence_locator={"source": "blob", "blob_id": str(blob_id)},
                )
            )
            continue
        diagnostics.append(
            {
                "code": "source_inspection_warning",
                "severity": "info",
                "message": warning,
                "suggested_repair": None,
                "evidence_locator": {"source": "blob", "blob_id": str(blob_id)},
            }
        )

    inspected_blob_id = str(blob_id)
    node_plugins = {(n.plugin or "").lower() for n in state.nodes}
    if "web_scrape" not in node_plugins and session_engine is not None and session_id is not None:
        for source_name, candidate_source in state.sources.items():
            if "blob_ref" not in candidate_source.options:
                continue
            candidate_blob_id = str(candidate_source.options["blob_ref"])
            if candidate_blob_id == inspected_blob_id:
                continue
            if candidate_source.plugin != "text":
                continue
            candidate_blob = _sync_get_blob(session_engine, candidate_blob_id, session_id)
            if candidate_blob is None or candidate_blob["status"] != "ready":
                continue
            candidate_storage_path = Path(candidate_blob["storage_path"])
            if not candidate_storage_path.exists():
                continue
            candidate_content = candidate_storage_path.read_bytes()
            _verify_blob_content_integrity(candidate_blob, candidate_content)
            candidate_facts = inspect_blob_content(
                content=candidate_content,
                filename=candidate_blob["filename"],
                mime_type=candidate_blob["mime_type"],
                content_hash=candidate_blob["content_hash"],
            )
            if candidate_facts.source_kind != "text" or not candidate_facts.url_candidates:
                continue
            diagnostics.append(
                _blocking_diagnostic(
                    code="text_source_url_without_web_scrape",
                    message=(
                        f"Source blob contains URL(s) {list(candidate_facts.url_candidates)} but no "
                        "web_scrape transform is wired downstream. The URL string itself will "
                        "flow to sinks, not the URL's content."
                    ),
                    suggested_repair=(
                        "upsert_node({node_type: 'transform', plugin: 'web_scrape', "
                        "input: <source on_success>, options: {url_field: '<column>'}}) and route "
                        "the source on_success to it."
                    ),
                    evidence_locator={
                        "source": "blob",
                        "source_name": source_name,
                        "blob_id": candidate_blob_id,
                        "url_candidates": list(candidate_facts.url_candidates),
                    },
                )
            )

    return diagnostics


def _execute_preview_pipeline(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Preview pipeline configuration — dry-run validation with source summary.

    Returns ``authoring_validation`` (Stage 1), ``runtime_preflight``
    (Stage 2 from the caller-supplied callback), and ``proof_diagnostics``
    (Stage 3 — operator-input-aware proof against the observed source
    blob). The presence of any blocking ``proof_diagnostics`` entry means
    ``is_valid=False`` even when authoring + runtime checks pass.
    """
    validation = context.catalog.validate_composition_state(state).validation
    _AUTHORING_VALIDATION_COUNTER.add(
        1,
        {"outcome": "valid" if validation.is_valid else "invalid"},
    )
    authoring_payload = _authoring_validation_payload(state, validation)
    runtime_result = context.runtime_preflight(state) if context.runtime_preflight is not None else None

    proof_diagnostics = compute_proof_diagnostics(
        state,
        session_engine=context.session_engine,
        session_id=context.session_id,
    )
    has_blocking_proof = any(d["severity"] == "blocking" for d in proof_diagnostics)

    is_valid = validation.is_valid
    if runtime_result is not None:
        is_valid = is_valid and runtime_result.is_valid
    if has_blocking_proof:
        is_valid = False

    summary: dict[str, Any] = {
        "is_valid": is_valid,
        "errors": authoring_payload["errors"],
        "warnings": authoring_payload["warnings"],
        "suggestions": authoring_payload["suggestions"],
        "edge_contracts": authoring_payload["edge_contracts"],
        "semantic_contracts": authoring_payload["semantic_contracts"],
        "graph_repair_suggestions": authoring_payload["graph_repair_suggestions"],
        "authoring_validation": authoring_payload,
        "runtime_preflight": runtime_result.model_dump() if runtime_result is not None else None,
        "proof_diagnostics": proof_diagnostics,
        "sources": {
            name: {
                "plugin": source.plugin,
                "on_success": source.on_success,
                "has_schema_config": _source_options_have_schema(source.options),
            }
            for name, source in state.sources.items()
        },
        "node_count": len(state.nodes),
        "output_count": len(state.outputs),
        "nodes": [{"id": n.id, "node_type": n.node_type, "plugin": n.plugin} for n in state.nodes],
        "outputs": [{"name": o.name, "plugin": o.plugin} for o in state.outputs],
    }

    return ToolResult(
        success=True,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data=summary,
        runtime_preflight=runtime_result,
    )


_PREVIEW_PIPELINE_DECLARATION = ToolDeclaration(
    name="preview_pipeline",
    handler=_execute_preview_pipeline,
    kind=ToolKind.DISCOVERY,
    description="Preview the current pipeline configuration — returns "
    "validation status, source summary, and node/output overview "
    "without executing. Use this to confirm the pipeline is set up "
    "correctly before running.",
    json_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    cacheable=False,
)


def _execute_diff_pipeline(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Compute a diff/change summary against a baseline state.

    The baseline is passed explicitly by the MCP server or web composer
    via ``context.baseline``. If no baseline is available, returns a
    notice instead.

    Pre-computed current-state validation is threaded through
    ``context.current_validation`` so this handler avoids redundant
    recomputation when the caller already has the live ValidationSummary
    in hand.
    """
    baseline = context.baseline
    current_validation = context.current_validation
    if baseline is None:
        return _discovery_result(
            state,
            {
                _DATA_ERROR_KEY: "No baseline available. Load or create a session first.",
                "current_version": state.version,
            },
        )

    baseline_validation = context.catalog.validate_composition_state(baseline).validation
    changes = diff_states(
        baseline,
        state,
        baseline_validation=baseline_validation,
        current_validation=current_validation,
    )
    return _discovery_result(state, changes)


_DIFF_PIPELINE_DECLARATION = ToolDeclaration(
    name="diff_pipeline",
    handler=_execute_diff_pipeline,
    kind=ToolKind.DISCOVERY,
    description="Show what changed since the session was loaded or created. "
    "Returns added, removed, and modified nodes/edges/outputs, "
    "plus warnings introduced or resolved.",
    json_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    cacheable=False,
)


TOOLS_IN_MODULE: tuple[ToolDeclaration, ...] = (
    _GET_PLUGIN_SCHEMA_DECLARATION,
    _GET_EXPRESSION_GRAMMAR_DECLARATION,
    _EXPLAIN_VALIDATION_ERROR_DECLARATION,
    _GET_PLUGIN_ASSISTANCE_DECLARATION,
    _GET_AUDIT_INFO_DECLARATION,
    _LIST_MODELS_DECLARATION,
    _PREVIEW_PIPELINE_DECLARATION,
    _DIFF_PIPELINE_DECLARATION,
)
"""Every tool declared in this module, in stable order.

``_dispatch.py`` aggregates this tuple alongside every other plane's
TOOLS_IN_MODULE to build the registered-tool universe."""
