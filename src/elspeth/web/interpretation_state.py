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
from typing import Any, Literal, NotRequired, TypedDict

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.enums import CreationModality
from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.state import CompositionState, NodeSpec, SourceSpec
from elspeth.web.validation import INTERPRETATION_PLACEHOLDER_RE

INTERPRETATION_REQUIREMENTS_KEY = "interpretation_requirements"
PROMPT_TEMPLATE_PARTS_KEY = "prompt_template_parts"
SOURCE_AUTHORING_KEY = "source_authoring"
SOURCE_COMPONENT_ID = "source"
INTERPRETATION_REVIEW_PENDING_CODE = "interpretation_review_pending"
PENDING_INTERPRETATION_AUTHORING_TEXT = "pending interpretation"

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


def strip_authoring_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return runtime options with web-only authoring metadata removed."""

    return {key: value for key, value in options.items() if key not in AUTHORING_METADATA_OPTION_KEYS}


def interpretation_sites(state: CompositionState) -> tuple[InterpretationReviewSite, ...]:
    """Return unresolved interpretation-review sites across source and transforms."""

    sites: list[InterpretationReviewSite] = []
    if state.source is not None:
        sites.extend(_pending_source_sites(state.source))
    for node in state.nodes:
        node_sites = [*_pending_node_sites(node), *_legacy_placeholder_sites(node)]
        sites.extend(node_sites)
        if not any(site.kind is InterpretationKind.VAGUE_TERM for site in node_sites):
            sites.extend(_missing_prompt_template_review_sites(node))
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
        materialized = _materialize_node_for_execution(node)
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


def _materialize_node_for_execution(node: NodeSpec) -> NodeSpec:
    if node.plugin != "llm":
        return node
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
    if requirement is None or requirement["status"] == "resolved":
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
    if node.plugin != "llm":
        return ()
    requirements = _requirements(node.options)
    if requirements is None:
        return ()
    sites: list[InterpretationReviewSite] = []
    for requirement in requirements:
        status = requirement["status"]
        if status == "pending":
            sites.append(
                InterpretationReviewSite(
                    component_id=node.id,
                    component_type="transform",
                    user_term=requirement["user_term"].strip(),
                    kind=InterpretationKind(requirement["kind"]),
                )
            )
    return tuple(sites)


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
    if requirement is None or requirement["status"] == "resolved":
        return ()
    return (
        InterpretationReviewSite(
            component_id=node.id,
            component_type="transform",
            user_term=_requirement_user_term_or_default(requirement, "prompt_template"),
            kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
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
    if "kind" not in value:
        raise TypeError("interpretation requirement kind is required")
    kind_value = value["kind"]
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
    expected_hash = stable_hash(prompt_template)
    if resolved["resolved_prompt_template_hash"] != expected_hash:
        raise ValueError(f"llm node {node.id!r} prompt-template review hash drifted")


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
