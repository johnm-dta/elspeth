"""Structured interpretation-review state for composer-authored LLM prompts.

Layer: L3 web application.

The runtime LLM plugin owns ``prompt_template`` as real Jinja prompt text.
Human-review workflow state is represented here as structured authoring
metadata on the web composition node and stripped before engine configuration.
Legacy ``{{interpretation:<term>}}`` prompts are still detected so older session
states can be opened and resolved during the migration window.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final, Literal, NotRequired, TypedDict

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.enums import CreationModality
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.state import CompositionState, NodeSpec, SourceSpec
from elspeth.web.validation import INTERPRETATION_PLACEHOLDER_RE

INTERPRETATION_REQUIREMENTS_KEY = "interpretation_requirements"
PROMPT_TEMPLATE_PARTS_KEY = "prompt_template_parts"
SOURCE_AUTHORING_KEY = "source_authoring"
SOURCE_COMPONENT_ID = "source"
INTERPRETATION_REVIEW_PENDING_CODE = "interpretation_review_pending"
PENDING_INTERPRETATION_AUTHORING_TEXT = "pending interpretation"
RAW_HTML_CLEANUP_USER_TERM: Final[str] = "drop_raw_html_fields"
RAW_HTML_CLEANUP_REVIEW_DRAFT: Final[str] = "Drop the scraped raw HTML and fingerprint fields before saving the JSON output."
PROMPT_SHIELD_USER_TERM: Final[str] = "prompt_injection_shield_recommendation"
PROMPT_SHIELD_WARNING_DRAFT: Final[str] = (
    "Recommend inserting azure_prompt_shield (or the deployment equivalent prompt-injection shield) "
    "between the external-content fetch step and this LLM. The current draft routes "
    "internet-controlled text directly into the LLM without that shield, which is a prompt-injection "
    "exposure on untrusted remote content, but continuing without it is allowed. "
    "[user_term: prompt_injection_shield_recommendation]"
)

_RAW_HTML_CLEANUP_DRAFT_MARKERS: Final[tuple[str, ...]] = ("raw html", "fingerprint")

# Transform plugins whose output is externally-controlled remote content for
# prompt-injection-defence purposes. web_scrape returns whatever the fetched
# page served, which is by definition untrusted.
_UNTRUSTED_REMOTE_CONTENT_PRODUCER_PLUGINS: Final[frozenset[str]] = frozenset({"web_scrape"})

# Transform plugins that constitute an authorized prompt-injection shield
# sitting between an untrusted producer and a downstream LLM consumer.
# Content-moderation (azure_content_safety) is deliberately NOT in this set:
# content moderation and prompt-injection shielding are different controls.
_AUTHORIZED_PROMPT_SHIELD_PLUGINS: Final[frozenset[str]] = frozenset({"azure_prompt_shield"})

AUTHORING_METADATA_OPTION_KEYS: frozenset[str] = frozenset(
    {
        INTERPRETATION_REQUIREMENTS_KEY,
        PROMPT_TEMPLATE_PARTS_KEY,
        SOURCE_AUTHORING_KEY,
    }
)


class InterpretationRequirement(TypedDict):
    id: str
    kind: str
    user_term: str
    status: Literal["pending", "resolved"]
    draft: str | None
    event_id: str | None
    accepted_value: str | None
    accepted_artifact_hash: str | None
    resolved_prompt_template_hash: str | None


class PromptTextPart(TypedDict):
    kind: Literal["text"]
    text: str


class PromptInterpretationRefPart(TypedDict):
    kind: Literal["interpretation_ref"]
    requirement_id: str


class PromptPart(TypedDict):
    kind: str
    text: NotRequired[str]
    requirement_id: NotRequired[str]


class SourceAuthoringMetadata(TypedDict):
    modality: str
    content_hash: str
    review_event_id: str | None
    resolved_kind: str | None


@dataclass(frozen=True, slots=True)
class InterpretationReviewSite:
    component_id: str
    component_type: Literal["source", "transform"]
    user_term: str
    kind: InterpretationKind


@dataclass(frozen=True, slots=True)
class InterpretationReviewPending:
    """Execution/readiness blocker for unresolved interpretation review."""

    sites: Sequence[InterpretationReviewSite]

    def __post_init__(self) -> None:
        freeze_fields(self, "sites")


def strip_authoring_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return runtime options with web-only authoring metadata removed."""

    return {key: value for key, value in options.items() if key not in AUTHORING_METADATA_OPTION_KEYS}


def validate_pipeline_decision_semantics(
    *,
    node_id: str,
    plugin: str | None,
    options: Mapping[str, Any],
    user_term: str,
    draft: str | None,
    context: str,
) -> None:
    """Validate that reviewed pipeline-shaping decisions match node behavior."""

    if not _is_raw_html_cleanup_decision(user_term=user_term, draft=draft):
        return
    if plugin != "field_mapper":
        raise ValueError(
            f"{context}: raw-html cleanup decision {user_term!r} must be implemented by a field_mapper node; "
            f"node {node_id!r} has plugin {plugin!r}"
        )
    if options.get("select_only") is not True:
        raise ValueError(f"{context}: raw-html cleanup decision {user_term!r} requires field_mapper.select_only=true on node {node_id!r}")
    mapping = options.get("mapping")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(
            f"{context}: raw-html cleanup decision {user_term!r} requires a non-empty field_mapper.mapping on node {node_id!r}"
        )
    preserved_raw_fields = sorted(
        {
            field_name
            for source_field, target_field in mapping.items()
            for field_name in _validated_mapping_pair(source_field, target_field, context=context, node_id=node_id)
            if _looks_like_raw_html_field(field_name)
        }
    )
    if preserved_raw_fields:
        raise ValueError(
            f"{context}: raw-html cleanup decision {user_term!r} preserves raw HTML/fingerprint field(s) "
            f"on node {node_id!r}: {preserved_raw_fields}. Remove them from mapping when select_only=true."
        )


def raw_html_cleanup_review_contract_error(state: CompositionState) -> str | None:
    """Return a composer-facing error for unreviewed or contradictory raw cleanup."""
    web_scrape_raw_fields = _web_scrape_raw_fields(state.nodes)
    if not web_scrape_raw_fields:
        return None
    for node in state.nodes:
        requirement_error = _raw_html_cleanup_requirement_contract_error(node)
        if requirement_error is not None:
            return requirement_error
    for node in state.nodes:
        if node.plugin != "field_mapper" or node.options.get("select_only") is not True:
            continue
        mapping = node.options.get("mapping")
        if not isinstance(mapping, Mapping) or not mapping:
            continue
        preserved_fields = _preserved_mapping_fields(mapping)
        preserved_raw_fields = sorted(field for field in web_scrape_raw_fields if field in preserved_fields)
        if preserved_raw_fields and _looks_like_cleanup_node_id(node.id):
            return (
                f"Node {node.id!r} is named like raw HTML cleanup but preserves web-scrape raw field(s) "
                f"{preserved_raw_fields}. Remove those fields from field_mapper.mapping when select_only=true."
            )
        if preserved_raw_fields:
            continue
        requirements = _requirements(node.options)
        if _requirement_for_kind(requirements, InterpretationKind.PIPELINE_DECISION) is None:
            return (
                f"Node {node.id!r} drops web-scrape raw field(s) {sorted(web_scrape_raw_fields)} with "
                "field_mapper.select_only=true. Stage a pending pipeline_decision interpretation_requirements "
                f"entry on that node with user_term {RAW_HTML_CLEANUP_USER_TERM!r} and draft "
                f"{RAW_HTML_CLEANUP_REVIEW_DRAFT!r}, then call request_interpretation_review. "
                "Do not put interpretation_requirements inside field_mapper.mapping; mapping contains "
                "only data fields to preserve. interpretation_requirements must be a sibling of mapping "
                "inside the field_mapper options object. "
                "If this came from a rejected set_pipeline call, resubmit the full pipeline with that "
                "requirement on the cleanup node; rejected set_pipeline calls do not persist partial nodes."
            )
    return None


def prompt_shield_recommendation_warning_pairs(state: CompositionState) -> tuple[tuple[str, str], ...]:
    """Return advisory warnings for unshielded untrusted content entering an LLM."""

    producer_by_output_stream = _producer_by_output_stream(state.nodes)
    warnings: list[tuple[str, str]] = []
    for node in state.nodes:
        if node.plugin != "llm":
            continue
        if not _llm_consumes_untrusted_remote_content(node, producer_by_output_stream):
            continue
        if _llm_has_shield_recommendation(node):
            continue
        warnings.append(
            (
                f"node:{node.id}",
                (
                    f"LLM node {node.id!r} consumes externally-fetched content from a web_scrape upstream "
                    "without an authorized prompt-injection shield between them. "
                    f"{PROMPT_SHIELD_WARNING_DRAFT}"
                ),
            )
        )
    return tuple(warnings)


def _producer_by_output_stream(nodes: Sequence[NodeSpec]) -> dict[str, NodeSpec]:
    producers: dict[str, NodeSpec] = {}
    for node in nodes:
        if isinstance(node.on_success, str) and node.on_success:
            producers[node.on_success] = node
    return producers


def _llm_consumes_untrusted_remote_content(
    node: NodeSpec,
    producer_by_output_stream: Mapping[str, NodeSpec],
) -> bool:
    """Walk upstream from ``node``; return True iff untrusted producer reached without a shield."""

    if node.plugin != "llm":
        return False
    stream = node.input
    visited: set[str] = set()
    while isinstance(stream, str) and stream and stream not in visited:
        visited.add(stream)
        producer = producer_by_output_stream.get(stream)
        if producer is None:
            return False
        if producer.plugin in _AUTHORIZED_PROMPT_SHIELD_PLUGINS:
            return False
        if producer.plugin in _UNTRUSTED_REMOTE_CONTENT_PRODUCER_PLUGINS:
            return True
        stream = producer.input
    return False


def _llm_has_shield_recommendation(node: NodeSpec) -> bool:
    requirements = _requirements(node.options)
    if requirements is None:
        return False
    for requirement in requirements:
        if InterpretationKind(requirement["kind"]) is not InterpretationKind.PIPELINE_DECISION:
            continue
        if requirement["user_term"].strip() == PROMPT_SHIELD_USER_TERM:
            return True
    return False


def _raw_html_cleanup_requirement_contract_error(node: NodeSpec) -> str | None:
    try:
        requirements = _requirements(node.options)
    except (KeyError, TypeError, ValueError) as exc:
        return f"Node {node.id!r} has invalid interpretation_requirements: {exc}"
    if requirements is None:
        return None
    for requirement in requirements:
        if InterpretationKind(requirement["kind"]) is not InterpretationKind.PIPELINE_DECISION:
            continue
        if not _is_raw_html_cleanup_decision(user_term=requirement["user_term"], draft=requirement["draft"]):
            continue
        try:
            validate_pipeline_decision_semantics(
                node_id=node.id,
                plugin=node.plugin,
                options=node.options,
                user_term=requirement["user_term"],
                draft=requirement["draft"],
                context="raw-html cleanup review contract",
            )
        except ValueError as exc:
            return str(exc)
    return None


def _is_raw_html_cleanup_decision(*, user_term: str, draft: str | None) -> bool:
    normalized_term = user_term.strip()
    if normalized_term != RAW_HTML_CLEANUP_USER_TERM:
        return False
    if not isinstance(draft, str):
        return False
    normalized_draft = draft.lower()
    return all(marker in normalized_draft for marker in _RAW_HTML_CLEANUP_DRAFT_MARKERS)


def _validated_mapping_pair(source_field: object, target_field: object, *, context: str, node_id: str) -> tuple[str, str]:
    if not isinstance(source_field, str) or not isinstance(target_field, str):
        raise ValueError(f"{context}: field_mapper.mapping on node {node_id!r} must map string field names to string field names")
    return (source_field, target_field)


def _looks_like_raw_html_field(field_name: str) -> bool:
    normalized = field_name.strip().lower().replace("-", "_")
    return normalized == "content" or "html" in normalized or "fingerprint" in normalized


def _looks_like_cleanup_node_id(node_id: str) -> bool:
    normalized = node_id.strip().lower().replace("-", "_")
    return "drop" in normalized or "cleanup" in normalized or "clean" in normalized or "raw" in normalized or "html" in normalized


def _preserved_mapping_fields(mapping: Mapping[str, Any]) -> frozenset[str]:
    fields: set[str] = set()
    for source_field, target_field in mapping.items():
        if isinstance(source_field, str):
            fields.add(source_field.strip())
        if isinstance(target_field, str):
            fields.add(target_field.strip())
    return frozenset(fields)


def interpretation_sites(state: CompositionState) -> tuple[InterpretationReviewSite, ...]:
    """Return unresolved interpretation-review sites across source and transforms."""

    sites: list[InterpretationReviewSite] = []
    if state.source is not None:
        sites.extend(_pending_source_sites(state.source))
    web_scrape_raw_fields = _web_scrape_raw_fields(state.nodes)
    for node in state.nodes:
        node_sites = [*_pending_node_sites(node), *_legacy_placeholder_sites(node)]
        sites.extend(node_sites)
        sites.extend(_missing_raw_html_cleanup_review_sites(node, web_scrape_raw_fields=web_scrape_raw_fields))
        if not any(site.kind is InterpretationKind.LLM_PROMPT_TEMPLATE for site in node_sites):
            sites.extend(_missing_prompt_template_review_sites(node))
        if not any(site.kind is InterpretationKind.LLM_MODEL_CHOICE for site in node_sites):
            sites.extend(_missing_model_choice_review_sites(node))
    return tuple(dict.fromkeys(sites))


def transform_vague_term_site_tuples(nodes: Sequence[NodeSpec]) -> tuple[tuple[str, str], ...]:
    """Compatibility view for vague-term LLM handoff paths.

    Older prompt-repair and handoff paths consume only ``(node_id, term)``
    tuples for transform vague-term sites. Keep that legacy view mechanical
    instead of letting source or prompt-template review sites bleed into this
    narrow tuple API.
    """

    sites: list[tuple[str, str]] = []
    for node in nodes:
        for site in (*_pending_node_sites(node), *_legacy_placeholder_sites(node)):
            if site.component_type == "transform" and site.kind is InterpretationKind.VAGUE_TERM:
                sites.append((site.component_id, site.user_term))
    return tuple(dict.fromkeys(sites))


def materialize_state_for_authoring(state: CompositionState) -> CompositionState:
    """Return a validation-safe authoring state without mutating ``state``."""

    changed = False
    materialized_nodes: list[NodeSpec] = []
    for node in state.nodes:
        materialized = _materialize_node_for_authoring(node)
        materialized_nodes.append(materialized)
        changed = changed or materialized is not node
    if not changed:
        return state
    return replace(state, nodes=tuple(materialized_nodes))


def materialize_state_for_execution(state: CompositionState) -> CompositionState | InterpretationReviewPending:
    """Materialize resolved interpretation state or return pending sites."""

    pending_sites = interpretation_sites(state)
    if pending_sites:
        return InterpretationReviewPending(sites=pending_sites)

    changed = False
    materialized_source = state.source
    if state.source is not None:
        materialized_source = _materialize_source_for_execution(state.source)
        changed = changed or materialized_source is not state.source
    materialized_nodes: list[NodeSpec] = []
    for node in state.nodes:
        materialized = _materialize_node_for_execution(node, state.nodes)
        materialized_nodes.append(materialized)
        changed = changed or materialized is not node
    if not changed:
        return state
    return replace(state, source=materialized_source, nodes=tuple(materialized_nodes))


def _materialize_node_for_authoring(node: NodeSpec) -> NodeSpec:
    if node.plugin != "llm":
        return node
    options = node.options
    parts = _prompt_parts(options)
    if parts is not None:
        prompt = _render_prompt_parts(parts, _requirements_by_id(options), unresolved_text=PENDING_INTERPRETATION_AUTHORING_TEXT)
        return _replace_prompt_if_changed(node, prompt, include_hash=False)

    if "resolved_prompt_template_hash" in options:
        return node

    prompt_template = options.get("prompt_template")
    if not isinstance(prompt_template, str):
        return node
    masked = INTERPRETATION_PLACEHOLDER_RE.sub(PENDING_INTERPRETATION_AUTHORING_TEXT, prompt_template)
    return _replace_prompt_if_changed(node, masked, include_hash=False)


def _materialize_node_for_execution(node: NodeSpec, all_nodes: Sequence[NodeSpec]) -> NodeSpec:
    _validate_pipeline_decision_review(node, all_nodes)
    if node.plugin != "llm":
        return node
    model = node.options.get("model")
    if isinstance(model, str) and model:
        _validate_model_choice_review(node, model)
    parts = _prompt_parts(node.options)
    if parts is None:
        prompt_template = node.options.get("prompt_template")
        if isinstance(prompt_template, str) and prompt_template:
            requirement = _prompt_template_review_requirement(node.options)
            if requirement is not None:
                _validate_prompt_template_review(node, prompt_template)
                return _ensure_prompt_template_hash(node)
        return node
    prompt = _render_prompt_parts(parts, _requirements_by_id(node.options), unresolved_text=None)
    _validate_prompt_template_review(node, prompt)
    return _replace_prompt_if_changed(node, prompt, include_hash=True)


def _materialize_source_for_execution(source: SourceSpec) -> SourceSpec:
    metadata = _source_authoring_metadata(source.options)
    if metadata is None or not _is_llm_authored_modality(metadata["modality"]):
        return source
    requirements = _requirements(source.options)
    resolved = _resolved_requirement_for_kind(requirements, InterpretationKind.INVENTED_SOURCE)
    if resolved is None:
        raise ValueError("invented source review requirement is required before execution")
    accepted_hash = resolved["accepted_artifact_hash"]
    if accepted_hash != metadata["content_hash"]:
        raise ValueError("invented source review drift: reviewed content hash does not match current source content hash")
    return source


def _replace_prompt_if_changed(node: NodeSpec, prompt: str, *, include_hash: bool) -> NodeSpec:
    current = node.options.get("prompt_template")
    current_hash = node.options.get("resolved_prompt_template_hash")
    next_hash = stable_hash(prompt) if include_hash else current_hash
    if current == prompt and current_hash == next_hash:
        return node
    options = dict(node.options)
    options["prompt_template"] = prompt
    if include_hash:
        options["resolved_prompt_template_hash"] = next_hash
    return replace(node, options=options)


def _ensure_prompt_template_hash(node: NodeSpec) -> NodeSpec:
    prompt_template = node.options.get("prompt_template")
    if not isinstance(prompt_template, str) or not prompt_template:
        return node
    return _replace_prompt_if_changed(node, prompt_template, include_hash=True)


def _pending_source_sites(source: SourceSpec) -> tuple[InterpretationReviewSite, ...]:
    metadata = _source_authoring_metadata(source.options)
    if metadata is None or not _is_llm_authored_modality(metadata["modality"]):
        return ()
    requirements = _requirements(source.options)
    requirement = _requirement_for_kind(requirements, InterpretationKind.INVENTED_SOURCE)
    if requirement is None:
        return (
            InterpretationReviewSite(
                component_id=SOURCE_COMPONENT_ID,
                component_type="source",
                user_term="llm_generated_source",
                kind=InterpretationKind.INVENTED_SOURCE,
            ),
        )
    if requirement["status"] == "resolved":
        return ()
    return (
        InterpretationReviewSite(
            component_id=SOURCE_COMPONENT_ID,
            component_type="source",
            user_term=requirement["user_term"].strip(),
            kind=InterpretationKind.INVENTED_SOURCE,
        ),
    )


def _pending_node_sites(node: NodeSpec) -> tuple[InterpretationReviewSite, ...]:
    requirements = _requirements(node.options)
    if requirements is None:
        return ()
    sites: list[InterpretationReviewSite] = []
    for requirement in requirements:
        status = requirement["status"]
        if status == "pending":
            kind = InterpretationKind(requirement["kind"])
            if node.plugin != "llm" and kind is not InterpretationKind.PIPELINE_DECISION:
                continue
            sites.append(
                InterpretationReviewSite(
                    component_id=node.id,
                    component_type="transform",
                    user_term=requirement["user_term"].strip(),
                    kind=kind,
                )
            )
    return tuple(sites)


def _web_scrape_raw_fields(nodes: Sequence[NodeSpec]) -> frozenset[str]:
    fields: set[str] = set()
    for node in nodes:
        if node.plugin != "web_scrape":
            continue
        content_field = node.options.get("content_field")
        fingerprint_field = node.options.get("fingerprint_field")
        if isinstance(content_field, str) and content_field.strip():
            fields.add(content_field.strip())
        if isinstance(fingerprint_field, str) and fingerprint_field.strip():
            fields.add(fingerprint_field.strip())
    return frozenset(fields)


def _missing_raw_html_cleanup_review_sites(
    node: NodeSpec,
    *,
    web_scrape_raw_fields: frozenset[str],
) -> tuple[InterpretationReviewSite, ...]:
    """Return the required review site for unreviewed web-scrape field cleanup."""
    if not web_scrape_raw_fields:
        return ()
    if node.plugin != "field_mapper":
        return ()
    requirements = _requirements(node.options)
    if _requirement_for_kind(requirements, InterpretationKind.PIPELINE_DECISION) is not None:
        return ()
    if node.options.get("select_only") is not True:
        return ()
    mapping = node.options.get("mapping")
    if not isinstance(mapping, Mapping) or not mapping:
        return ()
    preserved_fields = {
        field_name
        for source_field, target_field in mapping.items()
        if isinstance(source_field, str) and isinstance(target_field, str)
        for field_name in (source_field.strip(), target_field.strip())
    }
    if any(field in preserved_fields for field in web_scrape_raw_fields):
        return ()
    return (
        InterpretationReviewSite(
            component_id=node.id,
            component_type="transform",
            user_term=RAW_HTML_CLEANUP_USER_TERM,
            kind=InterpretationKind.PIPELINE_DECISION,
        ),
    )


def _legacy_placeholder_sites(node: NodeSpec) -> tuple[InterpretationReviewSite, ...]:
    if node.plugin != "llm":
        return ()
    prompt_template = node.options.get("prompt_template")
    if not isinstance(prompt_template, str):
        return ()
    return tuple(
        InterpretationReviewSite(
            component_id=node.id,
            component_type="transform",
            user_term=term,
            kind=InterpretationKind.VAGUE_TERM,
        )
        for term in _legacy_terms(prompt_template)
    )


def _missing_prompt_template_review_sites(node: NodeSpec) -> tuple[InterpretationReviewSite, ...]:
    if node.plugin != "llm":
        return ()
    prompt_template = node.options.get("prompt_template")
    if not isinstance(prompt_template, str) or not prompt_template:
        return ()
    requirement = _prompt_template_review_requirement(node.options)
    if requirement is None:
        return (
            InterpretationReviewSite(
                component_id=node.id,
                component_type="transform",
                user_term=f"llm_prompt_template:{node.id}",
                kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
            ),
        )
    if requirement["status"] == "resolved":
        return ()
    return (
        InterpretationReviewSite(
            component_id=node.id,
            component_type="transform",
            user_term=_requirement_user_term_or_default(requirement, "prompt_template"),
            kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        ),
    )


def _missing_model_choice_review_sites(node: NodeSpec) -> tuple[InterpretationReviewSite, ...]:
    """Enumerate an unreviewed model choice as a pending review site.

    Parallel to :func:`_missing_prompt_template_review_sites` for the
    ``llm_model_choice`` review kind. The mutation-time auto-stager
    (:func:`_options_with_default_model_choice_review`) catches new
    options-mutations; this enumerator catches pre-existing state that
    pre-dates the auto-stager (e.g. composition rows loaded from
    ``sessions.db`` before the gate was added). Together they guarantee
    that no LLM node with a non-empty ``options.model`` ever reaches a
    runnable state without a surfaced model-choice review.
    """
    if node.plugin != "llm":
        return ()
    model = node.options.get("model")
    if not isinstance(model, str) or not model:
        return ()
    requirement = _model_choice_review_requirement(node.options)
    if requirement is None:
        return (
            InterpretationReviewSite(
                component_id=node.id,
                component_type="transform",
                user_term=f"llm_model_choice:{node.id}",
                kind=InterpretationKind.LLM_MODEL_CHOICE,
            ),
        )
    if requirement["status"] == "resolved":
        return ()
    return (
        InterpretationReviewSite(
            component_id=node.id,
            component_type="transform",
            user_term=_requirement_user_term_or_default(requirement, "model"),
            kind=InterpretationKind.LLM_MODEL_CHOICE,
        ),
    )


def _requirement_user_term_or_default(requirement: InterpretationRequirement | None, default: str) -> str:
    if requirement is None:
        return default
    return requirement["user_term"].strip()


def _requirements_by_id(options: Mapping[str, Any]) -> dict[str, InterpretationRequirement]:
    requirements = _requirements(options)
    if requirements is None:
        return {}
    by_id: dict[str, InterpretationRequirement] = {}
    for requirement in requirements:
        requirement_id = requirement["id"]
        if requirement_id in by_id:
            raise ValueError(f"duplicate interpretation requirement id {requirement_id!r}")
        by_id[requirement_id] = requirement
    return by_id


def _requirements(options: Mapping[str, Any]) -> tuple[InterpretationRequirement, ...] | None:
    value = options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in options else None
    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        raise TypeError("interpretation_requirements must be a list")
    requirements: list[InterpretationRequirement] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("interpretation requirement entries must be mappings")
        requirements.append(_coerce_requirement(item))
    return tuple(requirements)


def _coerce_requirement(value: Mapping[str, Any]) -> InterpretationRequirement:
    requirement_id = value["id"]
    user_term = value["user_term"]
    status = value["status"]
    kind_value = value["kind"] if "kind" in value else InterpretationKind.VAGUE_TERM.value
    if not isinstance(requirement_id, str) or not requirement_id.strip():
        raise TypeError("interpretation requirement id must be a non-empty string")
    if not isinstance(user_term, str) or not user_term.strip():
        raise TypeError("interpretation requirement user_term must be a non-empty string")
    if status not in ("pending", "resolved"):
        raise ValueError(f"unknown interpretation requirement status {status!r}")
    if not isinstance(kind_value, str):
        raise TypeError("interpretation requirement kind must be a string")
    try:
        kind = InterpretationKind(kind_value)
    except ValueError as exc:
        raise ValueError(f"unknown interpretation requirement kind {kind_value!r}") from exc
    accepted_value = value["accepted_value"] if "accepted_value" in value else None
    if status == "resolved" and not isinstance(accepted_value, str):
        raise TypeError("resolved interpretation requirement must carry accepted_value")
    accepted_artifact_hash = value["accepted_artifact_hash"] if "accepted_artifact_hash" in value else None
    if accepted_artifact_hash is not None and not isinstance(accepted_artifact_hash, str):
        raise TypeError("interpretation requirement accepted_artifact_hash must be a string or None")
    resolved_prompt_template_hash = value["resolved_prompt_template_hash"] if "resolved_prompt_template_hash" in value else None
    if resolved_prompt_template_hash is not None and not isinstance(resolved_prompt_template_hash, str):
        raise TypeError("interpretation requirement resolved_prompt_template_hash must be a string or None")
    return InterpretationRequirement(
        id=requirement_id,
        kind=kind.value,
        user_term=user_term,
        status=status,
        draft=value["draft"] if "draft" in value else None,
        event_id=value["event_id"] if "event_id" in value else None,
        accepted_value=accepted_value,
        accepted_artifact_hash=accepted_artifact_hash,
        resolved_prompt_template_hash=resolved_prompt_template_hash,
    )


def _source_authoring_metadata(options: Mapping[str, Any]) -> SourceAuthoringMetadata | None:
    value = options[SOURCE_AUTHORING_KEY] if SOURCE_AUTHORING_KEY in options else None
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("source_authoring must be a mapping")
    modality = value["modality"]
    content_hash = value["content_hash"]
    review_event_id = value["review_event_id"] if "review_event_id" in value else None
    resolved_kind = value["resolved_kind"] if "resolved_kind" in value else None
    if not isinstance(modality, str) or not modality:
        raise TypeError("source_authoring.modality must be a non-empty string")
    if not isinstance(content_hash, str) or not content_hash:
        raise TypeError("source_authoring.content_hash must be a non-empty string")
    if review_event_id is not None and not isinstance(review_event_id, str):
        raise TypeError("source_authoring.review_event_id must be a string or None")
    if resolved_kind is not None:
        if not isinstance(resolved_kind, str):
            raise TypeError("source_authoring.resolved_kind must be a string or None")
        InterpretationKind(resolved_kind)
    return SourceAuthoringMetadata(
        modality=modality,
        content_hash=content_hash,
        review_event_id=review_event_id,
        resolved_kind=resolved_kind,
    )


def _is_llm_authored_modality(modality: str) -> bool:
    try:
        creation_modality = CreationModality(modality)
    except ValueError as exc:
        raise ValueError(f"unknown source authoring modality {modality!r}") from exc
    return creation_modality.requires_llm_provenance()


def _requirement_for_kind(
    requirements: Sequence[InterpretationRequirement] | None,
    kind: InterpretationKind,
) -> InterpretationRequirement | None:
    if requirements is None:
        return None
    matching = tuple(requirement for requirement in requirements if InterpretationKind(requirement["kind"]) is kind)
    if len(matching) > 1:
        raise ValueError(f"multiple interpretation requirements for kind {kind.value!r}")
    return matching[0] if matching else None


def _prompt_template_review_requirement(options: Mapping[str, Any]) -> InterpretationRequirement | None:
    return _requirement_for_kind(_requirements(options), InterpretationKind.LLM_PROMPT_TEMPLATE)


def _model_choice_review_requirement(options: Mapping[str, Any]) -> InterpretationRequirement | None:
    return _requirement_for_kind(_requirements(options), InterpretationKind.LLM_MODEL_CHOICE)


def _resolved_requirement_for_kind(
    requirements: Sequence[InterpretationRequirement] | None,
    kind: InterpretationKind,
) -> InterpretationRequirement | None:
    requirement = _requirement_for_kind(requirements, kind)
    if requirement is None or requirement["status"] != "resolved":
        return None
    return requirement


def _validate_prompt_template_review(node: NodeSpec, prompt_template: str) -> None:
    requirements = _requirements(node.options)
    resolved = _resolved_requirement_for_kind(requirements, InterpretationKind.LLM_PROMPT_TEMPLATE)
    if resolved is None:
        return
    parts = _prompt_parts(node.options)
    if parts is not None:
        # Structured nodes attest the prompt *skeleton*, not the substituted
        # text. The vague-term slot values are reviewed independently, so the
        # prompt-template review must be invariant under their resolution —
        # otherwise resolving a vague term (which rewrites the rendered prompt)
        # spuriously drifts a prompt-template review the operator already
        # approved. A genuine edit to a fixed text segment, or re-pointing a
        # slot to a different requirement, still changes the skeleton and drifts.
        expected_hash = prompt_structure_hash(parts)
    else:
        expected_hash = stable_hash(prompt_template)
    if resolved["resolved_prompt_template_hash"] != expected_hash:
        raise ValueError(f"llm node {node.id!r} prompt-template review hash drifted")


def _validate_model_choice_review(node: NodeSpec, model: str) -> None:
    """Tier-1 read guard — resolved model choice must still match options.model.

    If a previously accepted ``llm_model_choice`` review exists but the
    node's current ``options.model`` no longer hashes to the same value,
    something changed the model after acceptance — that violates the
    "every model choice surfaced to the user" contract and the audit
    trail can no longer attribute the run's model to a user decision.
    Crash rather than silently let a drifted choice through.
    """
    requirements = _requirements(node.options)
    resolved = _resolved_requirement_for_kind(requirements, InterpretationKind.LLM_MODEL_CHOICE)
    if resolved is None:
        return
    expected_hash = stable_hash(model)
    if resolved["resolved_prompt_template_hash"] != expected_hash:
        raise ValueError(f"llm node {node.id!r} model-choice review hash drifted")


def pipeline_decision_artifact_hash(
    node: NodeSpec,
    all_nodes: Sequence[NodeSpec],
    *,
    user_term: str,
) -> str:
    """Canonical artifact hash for a pipeline-decision review.

    The hash domain is the *minimum* state projection that, if changed,
    would invalidate the prior review. Different decision kinds adjudicate
    different facts about the graph, so each one routes through its own
    projection helper. A whole-node hash would invalidate the review on
    unrelated edits (e.g. swapping the LLM model) — operationally noisy
    without auditability gain because the review's premise is unchanged.

    Both the write side (sessions/service when an interpretation-resolve
    event lands) and the read side (preflight materialisation) call this
    function so the hash is produced by exactly one piece of code.

    Adding a new pipeline-decision kind requires a registered helper —
    unknown user_terms raise rather than fall through to a permissive
    default.
    """

    normalized = user_term.strip()
    if normalized == PROMPT_SHIELD_USER_TERM:
        return _prompt_shield_artifact_hash(node, all_nodes)
    if normalized == RAW_HTML_CLEANUP_USER_TERM:
        return _raw_html_cleanup_artifact_hash(node, all_nodes)
    raise ValueError(f"pipeline_decision_artifact_hash: unknown pipeline_decision user_term {user_term!r}")


def _prompt_shield_artifact_hash(node: NodeSpec, all_nodes: Sequence[NodeSpec]) -> str:
    """Material-scoped hash for the prompt-shield recommendation review.

    The review accepts the recommendation that an authorized prompt-injection
    shield (currently azure_prompt_shield) be inserted between an
    untrusted-remote-content producer (currently web_scrape) and this LLM.
    The hash binds to exactly that adjudication:

    - this LLM's node id (the review attaches to a specific node)
    - the upstream chain from this LLM's input back to either an authorized
      shield, an untrusted producer, or the end of the chain — captured as
      ``(producer_id, producer_plugin)`` pairs in stream order.

    Fields like ``model``, ``temperature``, ``prompt_template``, ``api_key``
    and ``schema`` are intentionally NOT in scope: they don't change whether
    a shield is needed. Swapping the model after review should leave the
    review intact.
    """

    producer_by_output_stream = _producer_by_output_stream(all_nodes)
    chain: list[tuple[str, str | None]] = []
    stream = node.input
    visited: set[str] = set()
    while isinstance(stream, str) and stream and stream not in visited:
        visited.add(stream)
        if stream not in producer_by_output_stream:
            break
        producer = producer_by_output_stream[stream]
        chain.append((producer.id, producer.plugin))
        if producer.plugin in _AUTHORIZED_PROMPT_SHIELD_PLUGINS:
            break
        if producer.plugin in _UNTRUSTED_REMOTE_CONTENT_PRODUCER_PLUGINS:
            break
        stream = producer.input
    return stable_hash(
        {
            "review_kind": "prompt_shield_recommendation",
            "llm_node_id": node.id,
            "upstream_chain": chain,
        }
    )


def _raw_html_cleanup_artifact_hash(node: NodeSpec, all_nodes: Sequence[NodeSpec]) -> str:
    """Material-scoped hash for the raw-html cleanup review.

    The review accepts that this field_mapper drops the upstream web_scrape's
    raw content/fingerprint fields. The hash binds to:

    - this field_mapper's node id
    - its ``mapping`` and ``select_only`` flag (the topological adjudication)
    - the set of raw fields any upstream web_scrape exposes (so adding a new
      raw field upstream re-stages the review — that's a material change to
      what "drop the raw fields" means).

    ``schema.guaranteed_fields`` on the field_mapper itself is excluded —
    it's a downstream consequence of the mapping, not adjudication input.
    """

    upstream_raw_fields = sorted(_web_scrape_raw_fields(all_nodes))
    raw_mapping = node.options["mapping"] if "mapping" in node.options else None
    mapping: dict[str, Any] = dict(raw_mapping) if isinstance(raw_mapping, Mapping) else {}
    select_only = "select_only" in node.options and node.options["select_only"] is True
    return stable_hash(
        {
            "review_kind": "raw_html_cleanup",
            "field_mapper_node_id": node.id,
            "mapping": mapping,
            "select_only": select_only,
            "upstream_web_scrape_raw_fields": upstream_raw_fields,
        }
    )


def _validate_pipeline_decision_review(node: NodeSpec, all_nodes: Sequence[NodeSpec]) -> None:
    requirements = _requirements(node.options)
    resolved = _resolved_requirement_for_kind(requirements, InterpretationKind.PIPELINE_DECISION)
    if resolved is None:
        return
    validate_pipeline_decision_semantics(
        node_id=node.id,
        plugin=node.plugin,
        options=node.options,
        user_term=resolved["user_term"],
        draft=resolved["draft"],
        context="interpretation_state",
    )
    expected_hash = pipeline_decision_artifact_hash(node, all_nodes, user_term=resolved["user_term"])
    if resolved["accepted_artifact_hash"] != expected_hash:
        raise ValueError(f"node {node.id!r} pipeline-decision review hash drifted")


def _prompt_parts(options: Mapping[str, Any]) -> tuple[PromptPart, ...] | None:
    value = options[PROMPT_TEMPLATE_PARTS_KEY] if PROMPT_TEMPLATE_PARTS_KEY in options else None
    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        raise TypeError("prompt_template_parts must be a list")
    parts: list[PromptPart] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("prompt_template_parts entries must be mappings")
        kind = item["kind"]
        if kind == "text":
            text = item["text"]
            if not isinstance(text, str):
                raise TypeError("text prompt part requires string text")
            parts.append(PromptPart(kind="text", text=text))
        elif kind == "interpretation_ref":
            requirement_id = item["requirement_id"]
            if not isinstance(requirement_id, str) or not requirement_id:
                raise TypeError("interpretation_ref prompt part requires requirement_id")
            parts.append(PromptPart(kind="interpretation_ref", requirement_id=requirement_id))
        else:
            raise ValueError(f"unknown prompt_template_parts kind {kind!r}")
    return tuple(parts)


def prompt_structure_hash(parts: tuple[PromptPart, ...]) -> str:
    """Canonical hash of a prompt's *structure* — text segments and the
    requirement each interpretation slot references — independent of whether
    those slots are pending or resolved and of their accepted values.

    This is the attestation domain for the ``llm_prompt_template`` review: that
    review approves the LLM-authored prompt skeleton. The vague-term reviews
    approve the slot *values* separately. Anchoring the prompt-template review
    to this skeleton makes it invariant under interpretation resolution — so
    resolving a vague term (which rewrites the rendered ``prompt_template``)
    does not drift the prompt-template review, while a genuine edit to a fixed
    text segment or a re-pointed slot still does.

    The projection deliberately excludes requirement status and accepted_value:
    those belong to the vague-term reviews' own attestations.
    """
    skeleton: list[tuple[str, str]] = []
    for part in parts:
        kind = part["kind"]
        if kind == "text":
            skeleton.append(("text", part["text"]))
        elif kind == "interpretation_ref":
            skeleton.append(("interpretation_ref", part["requirement_id"]))
        else:
            raise ValueError(f"unknown prompt part kind {kind!r}")
    return stable_hash(skeleton)


def prompt_structure_hash_from_options(options: Mapping[str, Any]) -> str | None:
    """Skeleton hash for a node's prompt parts, or ``None`` for a legacy
    (no-parts) node. Single derivation shared by the resolve-time anchor and the
    execution-time drift guard so they cannot diverge."""
    parts = _prompt_parts(options)
    if parts is None:
        return None
    return prompt_structure_hash(parts)


def _render_prompt_parts(
    parts: tuple[PromptPart, ...],
    requirements_by_id: Mapping[str, InterpretationRequirement],
    *,
    unresolved_text: str | None,
) -> str:
    rendered: list[str] = []
    for part in parts:
        kind = part["kind"]
        if kind == "text":
            rendered.append(part["text"])
            continue
        if kind != "interpretation_ref":
            raise ValueError(f"unknown prompt part kind {kind!r}")
        requirement_id = part["requirement_id"]
        if requirement_id not in requirements_by_id:
            raise ValueError(f"prompt part references unknown interpretation requirement {requirement_id!r}")
        requirement = requirements_by_id[requirement_id]
        if requirement["status"] == "pending":
            if unresolved_text is None:
                raise ValueError(f"interpretation requirement {requirement_id!r} is still pending")
            rendered.append(unresolved_text)
            continue
        accepted = requirement["accepted_value"]
        if not isinstance(accepted, str):
            raise TypeError(f"resolved interpretation requirement {requirement_id!r} has no accepted value")
        rendered.append(accepted)
    return "".join(rendered)


def _legacy_terms(prompt_template: str) -> tuple[str, ...]:
    return tuple(match.group(1).strip() for match in INTERPRETATION_PLACEHOLDER_RE.finditer(prompt_template))


def vague_term_wiring_count(options: Mapping[str, Any], *, user_term: str) -> int:
    """Count the resolvable ``vague_term`` wirings for ``user_term`` in a node's options.

    This is the single source of truth for "is a vague-term review actually
    resolvable?", shared by every staging-time gate so they cannot drift from
    the resolver contract. It mirrors the substitution wiring the resolver
    consumes in ``sessions/service.py::_patch_llm_transform_prompt``:

    * **Structured form** (``interpretation_requirements`` present): there must
      be exactly one *pending* ``vague_term`` requirement whose ``user_term``
      matches, AND a well-formed ``prompt_template_parts`` carrying at least one
      ``interpretation_ref`` part that references that requirement's ``id``.
      Returns ``1`` when wired; ``0`` when the requirement exists but no part
      references it (``prompt_template_parts`` absent, malformed, or missing the
      ref) — the deterministic root cause of the resolve-time 422 and the
      latent silent-drop; or the count of matching requirements when that count
      is not exactly one (caller treats any value ``!= 1`` as unresolvable).
    * **Legacy form** (no ``interpretation_requirements``): the number of
      ``{{interpretation:<user_term>}}`` placeholders in
      ``options.prompt_template``.

    Reads are lenient (Tier-3 staging idiom): a malformed sub-shape contributes
    ``0`` so the node reads as *unresolvable* and is routed back to the composer
    for repair, rather than crashing the request. Strict offensive validation
    lives at the resolve boundary, where the operator-approved mutation runs.
    """
    normalized_user_term = user_term.strip()
    requirements = options[INTERPRETATION_REQUIREMENTS_KEY] if INTERPRETATION_REQUIREMENTS_KEY in options else None
    matching_ids: list[Any] = []
    if isinstance(requirements, (list, tuple)):
        matching_ids = [
            requirement["id"]
            for requirement in requirements
            if isinstance(requirement, Mapping)
            and requirement.get("status") == "pending"
            and isinstance(requirement.get("user_term"), str)
            and requirement["user_term"].strip() == normalized_user_term
            and requirement.get("kind", InterpretationKind.VAGUE_TERM.value) == InterpretationKind.VAGUE_TERM.value
        ]
    if len(matching_ids) == 1:
        requirement_id = matching_ids[0]
        if not isinstance(requirement_id, str) or not requirement_id:
            return 0
        parts = options[PROMPT_TEMPLATE_PARTS_KEY] if PROMPT_TEMPLATE_PARTS_KEY in options else None
        if not isinstance(parts, (list, tuple)):
            return 0
        ref_count = sum(
            1
            for part in parts
            if isinstance(part, Mapping) and part.get("kind") == "interpretation_ref" and part.get("requirement_id") == requirement_id
        )
        return 1 if ref_count >= 1 else 0
    if len(matching_ids) > 1:
        return len(matching_ids)
    # No matching pending vague_term requirement: the term is either wired by a
    # legacy ``{{interpretation:<term>}}`` placeholder (which coexists with the
    # auto-staged prompt-template / model-choice requirements) or not wired at
    # all. Mirror the resolver's legacy fall-back and count placeholders.
    prompt_template = options["prompt_template"] if "prompt_template" in options else None
    if isinstance(prompt_template, str):
        return sum(1 for term in _legacy_terms(prompt_template) if term == normalized_user_term)
    return 0
