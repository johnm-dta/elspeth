"""Canonical, custody-safe pipeline proposal integrity envelope.

``PipelineProposal`` wraps the exact canonical ``set_pipeline`` arguments. It
does not define another pipeline topology model. The draft hash uses the
``composer.pipeline-proposal-envelope.v2`` domain because it covers every
authority-bearing envelope field. This intentionally supersedes the older
design's ``composer.pipeline-proposal.v1`` pipeline-only preimage; accepting
both would allow two different integrity meanings to share one draft concept.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, cast
from uuid import UUID

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import FrozenJsonArray, deep_thaw, freeze_fields
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.composer.bounded_json import (
    JSON_MAX_ITEMS,
    JSON_MAX_TOTAL_UTF8_BYTES,
    JsonBoundaryError,
    JsonTraversalBudget,
)

if TYPE_CHECKING:
    from elspeth.web.composer.state import CompositionState

_DRAFT_HASH_SCHEMA = "composer.pipeline-proposal-envelope.v2"
_REVIEWED_ANCHOR_SCHEMA = "guided.reviewed-anchors.v1"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_PROPOSAL_FIELDS = frozenset(
    {
        "pipeline",
        "draft_hash",
        "base",
        "reviewed_anchor_hash",
        "surface",
        "repair_count",
        "skill_hash",
        "covered_deferred_intent_ids",
        "supersedes_draft_hash",
    }
)


class PlannerSurface(StrEnum):
    """Authoring controller that requested the shared planner."""

    FREEFORM = "freeform"
    GUIDED_FULL = "guided_full"
    GUIDED_STAGED = "guided_staged"
    TUTORIAL_PROFILE = "tutorial_profile"


@dataclass(frozen=True, slots=True)
class AbsentBase:
    """Explicit assertion that no current composition state exists."""

    kind: Literal["absent"] = "absent"

    def __post_init__(self) -> None:
        if self.kind != "absent":
            raise AuditIntegrityError("AbsentBase kind must be 'absent'")


@dataclass(frozen=True, slots=True)
class PresentBase:
    """Binding to one exact existing composition state and its content."""

    state_id: UUID
    composition_content_hash: str
    kind: Literal["present"] = "present"

    def __post_init__(self) -> None:
        if self.kind != "present":
            raise AuditIntegrityError("PresentBase kind must be 'present'")
        if type(self.state_id) is not UUID:
            raise AuditIntegrityError("PresentBase state_id must be a UUID")
        _require_hash(self.composition_content_hash, "PresentBase composition_content_hash")


ProposalBase = AbsentBase | PresentBase


class PipelineProposalData(TypedDict):
    """Strict JSON-safe persistence shape returned by ``to_dict``."""

    pipeline: dict[str, Any]
    draft_hash: str
    base: dict[str, str]
    reviewed_anchor_hash: str
    surface: str
    repair_count: int
    skill_hash: str
    covered_deferred_intent_ids: list[str]
    supersedes_draft_hash: str | None


def _require_hash(value: object, field_name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
        raise AuditIntegrityError(f"{field_name} must be exactly 64 lowercase hexadecimal characters")


def _base_to_dict(base: ProposalBase) -> dict[str, str]:
    if type(base) is AbsentBase:
        return {"kind": "absent"}
    if type(base) is PresentBase:
        return {
            "kind": "present",
            "state_id": str(base.state_id),
            "composition_content_hash": base.composition_content_hash,
        }
    raise AuditIntegrityError("PipelineProposal base must be AbsentBase or PresentBase")


def _canonical_uuid_text(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise AuditIntegrityError(f"{field_name} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise AuditIntegrityError(f"{field_name} must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise AuditIntegrityError(f"{field_name} must be a canonical lowercase UUID string")
    return value


def _validate_and_freeze_strict_json_value(
    value: object,
    field_name: str,
    *,
    path: str,
    depth: int,
    budget: JsonTraversalBudget,
    active_container_ids: set[int],
) -> Any:
    """Validate and detach one strict JSON value in a single recursive pass."""
    budget.check_depth(depth, label=field_name)
    if type(value) in {dict, MappingProxyType}:
        mapping_value = cast(Mapping[object, object], value)
        container_id = id(value)
        if container_id in active_container_ids:
            raise AuditIntegrityError(f"{field_name} contains a recursive mapping at {path}")
        active_container_ids.add(container_id)
        try:
            frozen_children: dict[str, Any] = {}
            budget.consume_items(len(mapping_value), label=field_name)
            for key, child in mapping_value.items():
                if type(key) is not str:
                    raise AuditIntegrityError(f"{field_name} key at {path} must be an exact str")
                budget.consume_text(key, label=field_name)
                frozen_children[key] = _validate_and_freeze_strict_json_value(
                    child,
                    field_name,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                    budget=budget,
                    active_container_ids=active_container_ids,
                )
        finally:
            active_container_ids.remove(container_id)
        return MappingProxyType(frozen_children)

    if type(value) in {list, FrozenJsonArray}:
        sequence_value = cast(Sequence[object], value)
        container_id = id(value)
        if container_id in active_container_ids:
            raise AuditIntegrityError(f"{field_name} contains a recursive list at {path}")
        active_container_ids.add(container_id)
        try:
            budget.consume_items(len(sequence_value), label=field_name)
            frozen_items = FrozenJsonArray(
                _validate_and_freeze_strict_json_value(
                    child,
                    field_name,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    budget=budget,
                    active_container_ids=active_container_ids,
                )
                for index, child in enumerate(sequence_value)
            )
        finally:
            active_container_ids.remove(container_id)
        return frozen_items

    if value is None or type(value) in {bool, str, int, float}:
        if type(value) is str:
            budget.consume_text(value, label=field_name)
        return value

    raise AuditIntegrityError(
        f"{field_name} value at {path} must be an exact JSON leaf or a list/mapping container, got {type(value).__name__}"
    )


def _validate_and_freeze_canonical_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    """Snapshot an arbitrary Mapping once, then validate exact JSON containers."""
    try:
        snapshot: dict[object, object] = {}
        for index, item in enumerate(value.items()):
            if index >= JSON_MAX_ITEMS:
                raise JsonBoundaryError(f"{field_name} exceeds the {JSON_MAX_ITEMS}-item JSON limit")
            key, child = item
            snapshot[key] = child
    except JsonBoundaryError as exc:
        raise AuditIntegrityError(f"{field_name} violates bounded JSON: {exc}") from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise AuditIntegrityError(f"{field_name} must be a mapping") from exc
    try:
        frozen = _validate_and_freeze_strict_json_value(
            snapshot,
            field_name,
            path="$",
            depth=0,
            budget=JsonTraversalBudget(),
            active_container_ids=set(),
        )
        canonical = canonical_json(frozen)
        if len(canonical.encode("utf-8")) > JSON_MAX_TOTAL_UTF8_BYTES:
            raise JsonBoundaryError(f"{field_name} exceeds the {JSON_MAX_TOTAL_UTF8_BYTES}-byte canonical JSON limit")
    except JsonBoundaryError as exc:
        raise AuditIntegrityError(f"{field_name} violates bounded JSON: {exc}") from exc
    except RecursionError as exc:
        raise AuditIntegrityError(f"{field_name} exceeds bounded JSON recursion") from exc
    except (TypeError, ValueError) as exc:
        raise AuditIntegrityError(f"{field_name} contains a number outside the RFC 8785 JSON domain") from exc
    return cast(Mapping[str, Any], frozen)


def _validate_covered_intent_ids(intent_ids: tuple[str, ...]) -> None:
    if type(intent_ids) is not tuple:
        raise AuditIntegrityError("covered_deferred_intent_ids must be a tuple")
    seen: set[str] = set()
    for index, intent_id in enumerate(intent_ids):
        canonical_id = _canonical_uuid_text(intent_id, f"covered_deferred_intent_ids[{index}]")
        if canonical_id in seen:
            raise AuditIntegrityError("covered_deferred_intent_ids must not contain duplicates")
        seen.add(canonical_id)


def reviewed_anchor_hash(reviewed_facts: Mapping[str, Any]) -> str:
    """Hash a detached, recursively immutable snapshot of reviewed facts."""
    frozen_facts = _validate_and_freeze_canonical_mapping(reviewed_facts, "reviewed_facts")
    preimage = {
        "schema": _REVIEWED_ANCHOR_SCHEMA,
        "facts": frozen_facts,
    }
    canonical_json(preimage)
    return stable_hash(preimage)


def pipeline_draft_hash(
    *,
    pipeline: Mapping[str, Any],
    base: ProposalBase,
    reviewed_anchor_hash: str,
    surface: PlannerSurface,
    repair_count: int,
    skill_hash: str,
    covered_deferred_intent_ids: tuple[str, ...],
    supersedes_draft_hash: str | None,
) -> str:
    """Hash the complete authority-bearing proposal envelope under v2."""
    frozen_pipeline = _validate_and_freeze_canonical_mapping(pipeline, "pipeline")
    _require_hash(reviewed_anchor_hash, "reviewed_anchor_hash")
    if type(surface) is not PlannerSurface:
        raise AuditIntegrityError("surface must be PlannerSurface")
    if type(repair_count) is not int or repair_count < 0:
        raise AuditIntegrityError("repair_count must be a non-negative integer")
    _require_hash(skill_hash, "skill_hash")
    _validate_covered_intent_ids(covered_deferred_intent_ids)
    _require_hash(supersedes_draft_hash, "supersedes_draft_hash", optional=True)

    preimage = {
        "schema": _DRAFT_HASH_SCHEMA,
        "pipeline": frozen_pipeline,
        "base": _base_to_dict(base),
        "reviewed_anchor_hash": reviewed_anchor_hash,
        "surface": surface.value,
        "repair_count": repair_count,
        "skill_hash": skill_hash,
        "covered_deferred_intent_ids": list(covered_deferred_intent_ids),
        "supersedes_draft_hash": supersedes_draft_hash,
    }
    canonical_json(preimage)
    return stable_hash(preimage)


@dataclass(frozen=True, slots=True)
class PipelineProposal:
    """Deeply immutable, hash-verified exact ``set_pipeline`` arguments."""

    pipeline: Mapping[str, Any]
    draft_hash: str
    base: ProposalBase
    reviewed_anchor_hash: str
    surface: PlannerSurface
    repair_count: int
    skill_hash: str
    covered_deferred_intent_ids: tuple[str, ...]
    supersedes_draft_hash: str | None

    def __post_init__(self) -> None:
        frozen_pipeline = _validate_and_freeze_canonical_mapping(self.pipeline, "pipeline")
        _base_to_dict(self.base)
        _require_hash(self.draft_hash, "draft_hash")
        _require_hash(self.reviewed_anchor_hash, "reviewed_anchor_hash")
        if type(self.surface) is not PlannerSurface:
            raise AuditIntegrityError("surface must be PlannerSurface")
        if type(self.repair_count) is not int or self.repair_count < 0:
            raise AuditIntegrityError("repair_count must be a non-negative integer")
        _require_hash(self.skill_hash, "skill_hash")
        _validate_covered_intent_ids(self.covered_deferred_intent_ids)
        _require_hash(self.supersedes_draft_hash, "supersedes_draft_hash", optional=True)

        expected_draft_hash = pipeline_draft_hash(
            pipeline=frozen_pipeline,
            base=self.base,
            reviewed_anchor_hash=self.reviewed_anchor_hash,
            surface=self.surface,
            repair_count=self.repair_count,
            skill_hash=self.skill_hash,
            covered_deferred_intent_ids=self.covered_deferred_intent_ids,
            supersedes_draft_hash=self.supersedes_draft_hash,
        )
        if self.draft_hash != expected_draft_hash:
            raise AuditIntegrityError("PipelineProposal draft_hash mismatch")
        object.__setattr__(self, "pipeline", frozen_pipeline)
        freeze_fields(self, "pipeline", "covered_deferred_intent_ids")

    @classmethod
    def create(
        cls,
        *,
        pipeline: Mapping[str, Any],
        base: ProposalBase,
        reviewed_facts: Mapping[str, Any],
        surface: PlannerSurface,
        repair_count: int,
        skill_hash: str,
        covered_deferred_intent_ids: tuple[str, ...],
        supersedes_draft_hash: str | None,
        supplied_draft_hash: str | None = None,
        supplied_reviewed_anchor_hash: str | None = None,
    ) -> Self:
        """Construct from reviewed facts, optionally verifying supplied hashes."""
        frozen_pipeline = _validate_and_freeze_canonical_mapping(pipeline, "pipeline")
        computed_anchor_hash = reviewed_anchor_hash(reviewed_facts)
        if supplied_reviewed_anchor_hash is not None and supplied_reviewed_anchor_hash != computed_anchor_hash:
            raise AuditIntegrityError("PipelineProposal reviewed_anchor_hash mismatch")
        computed_draft_hash = pipeline_draft_hash(
            pipeline=frozen_pipeline,
            base=base,
            reviewed_anchor_hash=computed_anchor_hash,
            surface=surface,
            repair_count=repair_count,
            skill_hash=skill_hash,
            covered_deferred_intent_ids=covered_deferred_intent_ids,
            supersedes_draft_hash=supersedes_draft_hash,
        )
        if supplied_draft_hash is not None and supplied_draft_hash != computed_draft_hash:
            raise AuditIntegrityError("PipelineProposal draft_hash mismatch")
        return cls(
            pipeline=frozen_pipeline,
            draft_hash=computed_draft_hash,
            base=base,
            reviewed_anchor_hash=computed_anchor_hash,
            surface=surface,
            repair_count=repair_count,
            skill_hash=skill_hash,
            covered_deferred_intent_ids=covered_deferred_intent_ids,
            supersedes_draft_hash=supersedes_draft_hash,
        )

    def to_dict(self) -> PipelineProposalData:
        """Return the strict JSON-safe envelope used for persistence."""
        return {
            "pipeline": deep_thaw(self.pipeline),
            "draft_hash": self.draft_hash,
            "base": _base_to_dict(self.base),
            "reviewed_anchor_hash": self.reviewed_anchor_hash,
            "surface": self.surface.value,
            "repair_count": self.repair_count,
            "skill_hash": self.skill_hash,
            "covered_deferred_intent_ids": list(self.covered_deferred_intent_ids),
            "supersedes_draft_hash": self.supersedes_draft_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, reviewed_facts: Mapping[str, Any]) -> Self:
        """Strictly restore and reverify an envelope against reviewed facts."""
        payload = _validate_and_freeze_canonical_mapping(payload, "pipeline proposal payload")
        if set(payload) != _PROPOSAL_FIELDS:
            raise AuditIntegrityError("PipelineProposal persisted fields are malformed")
        _require_hash(payload["draft_hash"], "PipelineProposal draft_hash")
        _require_hash(payload["reviewed_anchor_hash"], "PipelineProposal reviewed_anchor_hash")

        raw_base = payload["base"]
        if type(raw_base) is not MappingProxyType:
            raise AuditIntegrityError("PipelineProposal base must be a mapping")
        base_mapping = cast(Mapping[str, Any], raw_base)
        base_kind = base_mapping["kind"] if "kind" in base_mapping else None
        if base_kind == "absent":
            if set(base_mapping) != {"kind"}:
                raise AuditIntegrityError("AbsentBase persisted fields are malformed")
            base: ProposalBase = AbsentBase()
        elif base_kind == "present":
            if set(base_mapping) != {"kind", "state_id", "composition_content_hash"}:
                raise AuditIntegrityError("PresentBase persisted fields are malformed")
            state_id_text = _canonical_uuid_text(base_mapping["state_id"], "base.state_id")
            base = PresentBase(
                state_id=UUID(state_id_text),
                composition_content_hash=base_mapping["composition_content_hash"],
            )
        else:
            raise AuditIntegrityError("PipelineProposal base kind is malformed")

        raw_surface = payload["surface"]
        if type(raw_surface) is not str:
            raise AuditIntegrityError("PipelineProposal surface is malformed")
        try:
            surface = PlannerSurface(raw_surface)
        except ValueError as exc:
            raise AuditIntegrityError("PipelineProposal surface is malformed") from exc

        raw_covered_ids = payload["covered_deferred_intent_ids"]
        if type(raw_covered_ids) is not FrozenJsonArray:
            raise AuditIntegrityError("PipelineProposal covered_deferred_intent_ids must be a list")

        return cls.create(
            pipeline=payload["pipeline"],
            base=base,
            reviewed_facts=reviewed_facts,
            surface=surface,
            repair_count=payload["repair_count"],
            skill_hash=payload["skill_hash"],
            covered_deferred_intent_ids=tuple(raw_covered_ids),
            supersedes_draft_hash=payload["supersedes_draft_hash"],
            supplied_draft_hash=payload["draft_hash"],
            supplied_reviewed_anchor_hash=payload["reviewed_anchor_hash"],
        )


def composition_content_hash(state: CompositionState) -> str:
    """Hash authored composition content, excluding version and guided metadata.

    The preimage is byte-for-byte equivalent to the helper moved from the
    guided route; changing it would invalidate existing base bindings.
    """
    state_d = state.to_dict()
    return stable_hash(
        {
            "sources": state_d["sources"],
            "nodes": state_d["nodes"],
            "edges": state_d["edges"],
            "outputs": state_d["outputs"],
            "metadata": state_d["metadata"],
        }
    )
