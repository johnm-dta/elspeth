"""Shared serialization + assistance helpers for semantic contracts.

These helpers serve every consumer that needs to render or annotate the
output of ``validate_semantic_contracts`` — currently:

* ``elspeth.web.execution.validation`` — /validate response payload.
* ``elspeth.web.execution.errors`` (via SemanticContractViolationError) —
  structured exception carried out of /execute when semantic contracts
  reject a pipeline.
* ``elspeth.web.execution.routes`` — 422 handler that turns
  SemanticContractViolationError into a JSON payload.

Hoisting the two helpers out of ``validation.py`` keeps every surface
that renders semantic contracts on a single source of truth — adding a
field updates one site rather than three. The shapes consciously
mirror ``elspeth.composer_mcp.server._SemanticEdgeContractPayload`` so
HTTP, MCP, and exception-path payloads stay aligned.
"""

from __future__ import annotations

from typing import cast

from elspeth.contracts.plugin_semantics import SemanticEdgeContract
from elspeth.web.composer.state import ValidationEntry
from elspeth.web.execution.schemas import SemanticEdgeContractResponse


def serialize_semantic_contracts(
    contracts: tuple[SemanticEdgeContract, ...],
) -> list[SemanticEdgeContractResponse]:
    """Convert internal SemanticEdgeContract records to the wire response model.

    Field shape mirrors composer_mcp/server.py::_SemanticEdgeContractPayload.
    Operators want to confirm "yes, semantic_contracts: 1 satisfied" in the
    UI banner even on success paths — the response carries the same
    structured payload regardless of the overall pass/fail outcome.
    """
    return [
        SemanticEdgeContractResponse(
            from_id=c.from_id,
            to_id=c.to_id,
            consumer_plugin=c.consumer_plugin,
            producer_plugin=c.producer_plugin,
            producer_field=c.producer_field,
            consumer_field=c.consumer_field,
            outcome=c.outcome.value,
            requirement_code=c.requirement.requirement_code,
        )
        for c in contracts
    ]


def assistance_suggestion_for(
    entry: ValidationEntry,
    contracts: tuple[SemanticEdgeContract, ...],
) -> str | None:
    """Look up plugin-owned guidance for a semantic error.

    Uses SemanticEdgeContract.consumer_plugin (and producer_plugin as
    a fallback) to address a SPECIFIC plugin class. Looping every
    registered transform and returning the first match was registry-
    order dependent — fixed by carrying the plugin names on the
    contract (Phase 1 Task 1.3).
    """
    from elspeth.plugins.infrastructure.base import BaseTransform
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    component_id = entry.component.removeprefix("node:")
    matching = next((c for c in contracts if c.to_id == component_id), None)
    if matching is None:
        return None

    manager = get_shared_plugin_manager()
    issue_code = matching.requirement.requirement_code

    # Consumer plugin owns the requirement, so it's the authoritative
    # source for guidance about the requirement_code. Verified method
    # name: get_transform_by_name (manager.py:183), NOT get_transform_class.
    # The registry returns type[TransformProtocol]; assistance lives on
    # BaseTransform — every in-tree plugin is a BaseTransform subclass,
    # so the cast is sound (per CLAUDE.md plugin-as-system-code policy).
    consumer_cls = cast(type[BaseTransform], manager.get_transform_by_name(matching.consumer_plugin))
    consumer_assistance = consumer_cls.get_agent_assistance(issue_code=issue_code)
    if consumer_assistance is not None:
        return consumer_assistance.summary

    # Producer plugin may also publish guidance for the producer-side
    # fact_code. The validator could attach that fact_code on the
    # contract in a later phase; for now, only consumer assistance is
    # surfaced as suggestion text.
    return None
