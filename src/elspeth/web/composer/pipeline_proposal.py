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
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict
from uuid import UUID

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_freeze, deep_thaw, freeze_fields
from elspeth.core.canonical import canonical_json, stable_hash

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


def _validate_canonical_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditIntegrityError(f"{field_name} must be a mapping")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise AuditIntegrityError(f"{field_name} must contain canonical JSON values") from exc
    return value


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
    frozen_facts = deep_freeze(reviewed_facts)
    _validate_canonical_mapping(frozen_facts, "reviewed_facts")
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
    covered_deferred_intent_ids: tuple[str, ...] = (),
    supersedes_draft_hash: str | None = None,
) -> str:
    """Hash the complete authority-bearing proposal envelope under v2."""
    frozen_pipeline = deep_freeze(pipeline)
    _validate_canonical_mapping(frozen_pipeline, "pipeline")
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
    covered_deferred_intent_ids: tuple[str, ...] = ()
    supersedes_draft_hash: str | None = None

    def __post_init__(self) -> None:
        freeze_fields(self, "pipeline", "covered_deferred_intent_ids")
        _validate_canonical_mapping(self.pipeline, "pipeline")
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
            pipeline=self.pipeline,
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
        covered_deferred_intent_ids: tuple[str, ...] = (),
        supersedes_draft_hash: str | None = None,
        supplied_draft_hash: str | None = None,
        supplied_reviewed_anchor_hash: str | None = None,
    ) -> Self:
        """Construct from reviewed facts, optionally verifying supplied hashes."""
        frozen_pipeline = deep_freeze(pipeline)
        _validate_canonical_mapping(frozen_pipeline, "pipeline")
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
        if not isinstance(payload, Mapping) or set(payload) != _PROPOSAL_FIELDS:
            raise AuditIntegrityError("PipelineProposal persisted fields are malformed")
        _require_hash(payload["draft_hash"], "PipelineProposal draft_hash")
        _require_hash(payload["reviewed_anchor_hash"], "PipelineProposal reviewed_anchor_hash")

        raw_base = payload["base"]
        if not isinstance(raw_base, Mapping):
            raise AuditIntegrityError("PipelineProposal base must be a mapping")
        base_kind = raw_base.get("kind")
        if base_kind == "absent":
            if set(raw_base) != {"kind"}:
                raise AuditIntegrityError("AbsentBase persisted fields are malformed")
            base: ProposalBase = AbsentBase()
        elif base_kind == "present":
            if set(raw_base) != {"kind", "state_id", "composition_content_hash"}:
                raise AuditIntegrityError("PresentBase persisted fields are malformed")
            state_id_text = _canonical_uuid_text(raw_base["state_id"], "base.state_id")
            base = PresentBase(
                state_id=UUID(state_id_text),
                composition_content_hash=raw_base["composition_content_hash"],
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
        if type(raw_covered_ids) is not list:
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
