"""Generic semantic-contract validator.

For each consumer node with declared input_semantic_requirements(),
walks back to the effective upstream producer (using ProducerResolver),
asks the producer for its output_semantics(), and compares facts to
requirements per field. Emits structured SemanticEdgeContract records
and high-severity ValidationEntry errors on CONFLICT or on UNKNOWN +
FAIL policy.

This module reuses ProducerResolver — there must be exactly ONE
walk-back implementation across composer state.

Pass-through propagation is intentionally NOT performed in Phase 1.
A pass-through transform between a declared producer and a declared
consumer breaks the chain — the consumer sees outcome=UNKNOWN, which
combined with line_explode's FAIL policy means the chain rejects.
This is by design: it forces a real propagation API decision rather
than making an ad hoc choice.

Layer: L3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from elspeth.contracts.plugin_semantics import (
    FieldSemanticFacts,
    OutputSemanticDeclaration,
    SemanticEdgeContract,
    SemanticOutcome,
    UnknownSemanticPolicy,
    compare_semantic,
)
from elspeth.web.composer._producer_resolver import (
    ProducerEntry,
    ProducerResolver,
    is_source_producer_id,
)
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    Severity,
    ValidationEntry,
)

if TYPE_CHECKING:
    from elspeth.plugins.infrastructure.base import BaseTransform


def _is_config_probe_exception(exc: Exception) -> bool:
    """Same expected-error set used by _check_schema_contracts for probes."""
    from elspeth.plugins.infrastructure.config_base import PluginConfigError
    from elspeth.plugins.infrastructure.manager import PluginNotFoundError
    from elspeth.plugins.infrastructure.templates import TemplateError
    from elspeth.plugins.infrastructure.validation import UnknownPluginTypeError

    if isinstance(exc, (PluginConfigError, PluginNotFoundError, TemplateError, UnknownPluginTypeError)):
        return True
    return type(exc) is ValueError and str(exc).startswith("Invalid configuration for transform ")


def _instantiate_consumer(node: NodeSpec) -> BaseTransform | None:
    """Construct a consumer transform instance to read its requirements.

    Returns None when:
    - node.plugin is unset (typed-None on a draft node)
    - construction fails with one of the established probe-tolerant exception
      types (PluginConfigError, PluginNotFoundError, TemplateError,
      UnknownPluginTypeError, or "Invalid configuration for transform"
      ValueError). This mirrors the producer-probe tolerance in
      ``_safe_output_semantics`` and the existing schema-contract probe
      tolerance in ``state._check_schema_contracts``.

    Why both probes share tolerance: ``CompositionState.validate()`` has an
    established contract that draft pipelines (incomplete config, unregistered
    plugins) do not crash validation — see
    ``test_contract_probe_constructor_exception_falls_back_instead_of_crashing``.
    The semantic validator runs inside ``validate()`` and must respect that
    contract. Phase 3 semantic-validation fixtures (``_wardline_state``)
    construct valid producer+consumer configs and assert specific outcomes
    (``len(errors) == 1``, ``outcome == CONFLICT``), so they never enter the
    tolerant branch — silent skip would surface as a positive-assertion
    failure, preserving the vacuousness guard.

    Unexpected exceptions PROPAGATE — a plugin method raising mid-construction
    is a system bug per CLAUDE.md plugin-as-system-code policy.
    """
    from elspeth.contracts.freeze import deep_thaw
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.web.interpretation_state import strip_authoring_options

    if node.plugin is None:
        return None
    try:
        return cast(
            "BaseTransform",
            get_shared_plugin_manager().create_transform(
                node.plugin,
                strip_authoring_options(deep_thaw(node.options)),
            ),
        )
    except Exception as exc:
        if _is_config_probe_exception(exc):
            return None
        raise


def _instantiate_producer(producer: ProducerEntry) -> BaseTransform | None:
    """Construct a producer transform/source instance to read its facts."""
    from elspeth.contracts.freeze import deep_thaw
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.web.interpretation_state import strip_authoring_options

    if producer.plugin_name is None or producer.producer_id == "source":
        # Sources don't expose output_semantics() in Phase 1 — return None.
        # Phase 2 (post-Wardline) can extend BaseSource the same way.
        return None
    return cast(
        "BaseTransform",
        get_shared_plugin_manager().create_transform(
            producer.plugin_name,
            strip_authoring_options(deep_thaw(producer.options)),
        ),
    )


def _safe_output_semantics(producer: ProducerEntry) -> OutputSemanticDeclaration | None:
    """Construct producer and read its semantics, tolerating draft-config errors.

    Returns None when:
    - Producer plugin has no instance (source — see _instantiate_producer)
    - Plugin construction fails with an expected draft/config probe error

    Unexpected exceptions PROPAGATE — they indicate a framework bug
    (per CLAUDE.md plugin-as-system-code policy: a plugin method that
    raises is a bug we MUST know about).
    """
    try:
        instance = _instantiate_producer(producer)
    except Exception as exc:
        if _is_config_probe_exception(exc):
            return None
        raise

    if instance is None:
        return None
    return instance.output_semantics()


def _find_producer_facts(
    declaration: OutputSemanticDeclaration,
    field_name: str,
) -> FieldSemanticFacts | None:
    for facts in declaration.fields:
        if facts.field_name == field_name:
            return facts
    return None


def _targeted_conflict_repair_hint(
    *,
    consumer_plugin: str,
    producer_plugin: str | None,
    requirement_code: str,
    facts: FieldSemanticFacts | None,
) -> str | None:
    if (
        consumer_plugin == "json_explode"
        and producer_plugin == "llm"
        and requirement_code == "json_explode.array_field.list"
        and facts is not None
        and facts.value_type.value == "str"
    ):
        return (
            "json_explode requires list field; LLM response_field is str. "
            "Use structured-output parsing or a transform that parses JSON object/string first."
        )
    return None


def validate_semantic_contracts(
    state: CompositionState,
) -> tuple[tuple[ValidationEntry, ...], tuple[SemanticEdgeContract, ...]]:
    """Validate semantic contracts across the composition.

    Returns (errors, contracts):
    - errors: ValidationEntry records suitable for ValidationSummary.errors
    - contracts: SemanticEdgeContract records for ValidationSummary.semantic_contracts
    """
    errors: list[ValidationEntry] = []
    contracts: list[SemanticEdgeContract] = []
    seen_edges: set[tuple[str, str, str, str]] = set()

    sink_names = frozenset(output.name for output in state.outputs)
    resolver = ProducerResolver.build(
        source=None,
        sources=state.sources,
        nodes=state.nodes,
        sink_names=sink_names,
    )

    for node in state.nodes:
        if node.node_type != "transform" or node.plugin is None:
            continue

        # CONSUMER probe failures must NOT be silently tolerated. The
        # consumer is the entry point — if it can't construct, the
        # test/composition has a bug we MUST surface, otherwise the
        # validator silently skips the case it's meant to check.
        # Producer probes are tolerant (separate _safe_output_semantics).
        consumer = _instantiate_consumer(node)
        if consumer is None:
            continue
        requirements = consumer.input_semantic_requirements()
        if not requirements.fields:
            continue

        upstream_producer = resolver.walk_to_real_producer(node.input)
        if upstream_producer is None:
            # No declared producer (coalesce, ambiguous, missing).
            # Schema-contract layer handles missing-field cases; semantic
            # layer treats this as unknown for any FAIL-policy requirement.
            for req in requirements.fields:
                edge_key = ("?source", node.id, req.field_name, req.field_name)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                contract = SemanticEdgeContract(
                    from_id="?",
                    to_id=node.id,
                    consumer_plugin=node.plugin,
                    producer_plugin=None,
                    producer_field=req.field_name,
                    consumer_field=req.field_name,
                    producer_facts=None,
                    requirement=req,
                    outcome=SemanticOutcome.UNKNOWN,
                )
                contracts.append(contract)
                if req.unknown_policy is UnknownSemanticPolicy.FAIL:
                    errors.append(
                        ValidationEntry(
                            f"node:{node.id}",
                            (
                                f"Semantic contract: '{node.plugin}' node '{node.id}' "
                                f"requires field '{req.field_name}' "
                                f"({req.requirement_code}) but the upstream producer is "
                                f"undeclared (coalesce, ambiguous, or unreachable)."
                            ),
                            cast(Severity, req.severity),
                        )
                    )
            continue

        # SOURCE → TRANSFORM edges are out of scope for Phase 1.
        # BaseSource has no output_semantics() API yet; treating source
        # producers as UNKNOWN under FAIL would break every working
        # csv → line_explode pipeline. Skip these edges entirely;
        # extending BaseSource is a separate plan.
        if is_source_producer_id(upstream_producer.producer_id):
            continue

        producer_decl = _safe_output_semantics(upstream_producer)

        for req in requirements.fields:
            facts = _find_producer_facts(producer_decl, req.field_name) if producer_decl is not None else None
            outcome = compare_semantic(facts, req)

            edge_key = (upstream_producer.producer_id, node.id, req.field_name, req.field_name)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            contracts.append(
                SemanticEdgeContract(
                    from_id=upstream_producer.producer_id,
                    to_id=node.id,
                    consumer_plugin=node.plugin,
                    producer_plugin=upstream_producer.plugin_name,
                    producer_field=req.field_name,
                    consumer_field=req.field_name,
                    producer_facts=facts,
                    requirement=req,
                    outcome=outcome,
                )
            )

            if outcome is SemanticOutcome.CONFLICT:
                producer_label = upstream_producer.plugin_name if upstream_producer.plugin_name is not None else "(unknown plugin)"
                facts_kind = facts.content_kind.value if facts else "unknown"
                facts_framing = facts.text_framing.value if facts else "unknown"
                facts_value_type = facts.value_type.value if facts else "unknown"
                # Generic diagnostic: name producer, consumer, field,
                # observed producer facts, and the requirement_code.
                # Do NOT enumerate accepted enum sets — that prints
                # values like "markdown" / "newline_framed" which read
                # as fix prose ("use markdown"). Fix prose belongs in
                # PluginAssistance, addressed by requirement_code.
                repair_hint = _targeted_conflict_repair_hint(
                    consumer_plugin=node.plugin,
                    producer_plugin=upstream_producer.plugin_name,
                    requirement_code=req.requirement_code,
                    facts=facts,
                )
                message = (
                    f"Semantic contract violation: '{upstream_producer.producer_id}' "
                    f"-> '{node.id}'. Consumer ({node.plugin}) requires field "
                    f"'{req.field_name}' to satisfy {req.requirement_code}, "
                    f"but producer ({producer_label}) declares "
                    f"content_kind={facts_kind}, text_framing={facts_framing}, "
                    f"value_type={facts_value_type}."
                )
                if repair_hint is not None:
                    message = f"{message} {repair_hint}"
                errors.append(
                    ValidationEntry(
                        f"node:{node.id}",
                        message,
                        cast(Severity, req.severity),
                    )
                )
            elif outcome is SemanticOutcome.UNKNOWN and req.unknown_policy is UnknownSemanticPolicy.FAIL:
                producer_label = upstream_producer.plugin_name if upstream_producer.plugin_name is not None else "(unknown plugin)"
                if facts is None:
                    semantic_detail = (
                        "declares no semantic facts for that field. "
                        "Producers semantically feeding this consumer must "
                        "declare output_semantics()."
                    )
                else:
                    semantic_detail = (
                        "does not declare all semantic dimensions required "
                        "for that field. Declared facts: "
                        f"content_kind={facts.content_kind.value}, "
                        f"text_framing={facts.text_framing.value}, "
                        f"value_type={facts.value_type.value}."
                    )
                errors.append(
                    ValidationEntry(
                        f"node:{node.id}",
                        (
                            f"Semantic contract: '{node.plugin}' node '{node.id}' "
                            f"requires field '{req.field_name}' "
                            f"({req.requirement_code}) but upstream producer "
                            f"'{upstream_producer.producer_id}' ({producer_label}) "
                            f"{semantic_detail}"
                        ),
                        cast(Severity, req.severity),
                    )
                )

    return tuple(errors), tuple(contracts)
