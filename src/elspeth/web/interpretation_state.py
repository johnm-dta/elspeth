"""Structured interpretation-review state for composer-authored LLM prompts.

Layer: L3 web application.

The runtime LLM plugin owns ``prompt_template`` as real Jinja prompt text.
Human-review workflow state is represented here as structured authoring
metadata on the web composition node and stripped before engine configuration.
Legacy ``{{interpretation:<term>}}`` prompts are still detected so older session
states can be opened and resolved during the migration window.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, NotRequired, TypedDict

from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.state import CompositionState, NodeSpec
from elspeth.web.validation import INTERPRETATION_PLACEHOLDER_RE

INTERPRETATION_REQUIREMENTS_KEY = "interpretation_requirements"
PROMPT_TEMPLATE_PARTS_KEY = "prompt_template_parts"
INTERPRETATION_REVIEW_PENDING_CODE = "interpretation_review_pending"
PENDING_INTERPRETATION_AUTHORING_TEXT = "pending interpretation"

AUTHORING_METADATA_OPTION_KEYS: frozenset[str] = frozenset(
    {
        INTERPRETATION_REQUIREMENTS_KEY,
        PROMPT_TEMPLATE_PARTS_KEY,
    }
)


class InterpretationRequirement(TypedDict):
    id: str
    user_term: str
    status: Literal["pending", "resolved"]
    draft: str | None
    event_id: str | None
    accepted_value: str | None
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


@dataclass(frozen=True, slots=True)
class InterpretationReviewPending:
    """Execution/readiness blocker for unresolved interpretation review."""

    sites: tuple[tuple[str, str], ...]


def strip_authoring_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return runtime options with web-only authoring metadata removed."""

    return {key: value for key, value in options.items() if key not in AUTHORING_METADATA_OPTION_KEYS}


def interpretation_sites(nodes: tuple[NodeSpec, ...]) -> tuple[tuple[str, str], ...]:
    """Return ``(node_id, user_term)`` sites with unresolved interpretation review."""

    sites: list[tuple[str, str]] = []
    for node in nodes:
        if node.plugin != "llm":
            continue
        structured = _pending_structured_sites(node)
        if structured:
            sites.extend(structured)
            continue
        if "resolved_prompt_template_hash" in node.options:
            continue
        prompt_template = node.options.get("prompt_template")
        if isinstance(prompt_template, str):
            sites.extend((node.id, term) for term in _legacy_terms(prompt_template))
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

    pending_sites = interpretation_sites(state.nodes)
    if pending_sites:
        return InterpretationReviewPending(sites=pending_sites)

    changed = False
    materialized_nodes: list[NodeSpec] = []
    for node in state.nodes:
        materialized = _materialize_node_for_execution(node)
        materialized_nodes.append(materialized)
        changed = changed or materialized is not node
    if not changed:
        return state
    return replace(state, nodes=tuple(materialized_nodes))


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
        return node
    prompt = _render_prompt_parts(parts, _requirements_by_id(node.options), unresolved_text=None)
    return _replace_prompt_if_changed(node, prompt, include_hash=True)


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


def _pending_structured_sites(node: NodeSpec) -> tuple[tuple[str, str], ...]:
    requirements = _requirements(node.options)
    if requirements is None:
        return ()
    sites: list[tuple[str, str]] = []
    for requirement in requirements:
        status = requirement["status"]
        if status == "pending":
            sites.append((node.id, requirement["user_term"].strip()))
    return tuple(sites)


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
    if not isinstance(requirement_id, str) or not requirement_id.strip():
        raise TypeError("interpretation requirement id must be a non-empty string")
    if not isinstance(user_term, str) or not user_term.strip():
        raise TypeError("interpretation requirement user_term must be a non-empty string")
    if status not in ("pending", "resolved"):
        raise ValueError(f"unknown interpretation requirement status {status!r}")
    accepted_value = value["accepted_value"] if "accepted_value" in value else None
    if status == "resolved" and not isinstance(accepted_value, str):
        raise TypeError("resolved interpretation requirement must carry accepted_value")
    return InterpretationRequirement(
        id=requirement_id,
        user_term=user_term,
        status=status,
        draft=value["draft"] if "draft" in value else None,
        event_id=value["event_id"] if "event_id" in value else None,
        accepted_value=accepted_value,
        resolved_prompt_template_hash=value["resolved_prompt_template_hash"] if "resolved_prompt_template_hash" in value else None,
    )


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
