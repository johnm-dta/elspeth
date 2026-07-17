"""Integrity contracts for the shared canonical pipeline proposal envelope."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any
from uuid import UUID

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.composer.pipeline_proposal import (
    AbsentBase,
    PipelineProposal,
    PlannerSurface,
    PresentBase,
    composition_content_hash,
    pipeline_draft_hash,
    reviewed_anchor_hash,
)

_STATE_ID = UUID("00000000-0000-4000-8000-000000000001")
_INTENT_A = "00000000-0000-4000-8000-00000000000a"
_INTENT_B = "00000000-0000-4000-8000-00000000000b"


def _pipeline() -> dict[str, Any]:
    return {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {
                "credentials": {"secret_ref": "CSV_API_TOKEN"},
                "columns": ["name", "score"],
            },
        },
        "nodes": [
            {
                "id": "clean",
                "plugin": "normalize",
                "options": {"rules": [{"column": "name", "operation": "strip"}]},
            }
        ],
        "edges": [{"from": "rows", "to": "clean"}],
        "outputs": [
            {
                "sink_name": "clean",
                "plugin": "json",
                "options": {"destination": {"secret_ref": "OUTPUT_PATH"}},
            }
        ],
    }


def _reviewed_facts() -> dict[str, Any]:
    return {
        "source": {"plugin": "csv", "columns": ["name", "score"]},
        "outputs": [{"plugin": "json", "required_fields": ["path"]}],
    }


def _create_proposal(
    *,
    pipeline: dict[str, Any] | None = None,
    base: AbsentBase | PresentBase | None = None,
    reviewed_facts: dict[str, Any] | None = None,
    surface: PlannerSurface = PlannerSurface.GUIDED_FULL,
    repair_count: int = 1,
    skill_hash: str = "c" * 64,
    covered_deferred_intent_ids: tuple[str, ...] = (_INTENT_A, _INTENT_B),
    supersedes_draft_hash: str | None = "d" * 64,
    supplied_draft_hash: str | None = None,
    supplied_reviewed_anchor_hash: str | None = None,
) -> PipelineProposal:
    return PipelineProposal.create(
        pipeline=_pipeline() if pipeline is None else pipeline,
        base=PresentBase(state_id=_STATE_ID, composition_content_hash="b" * 64) if base is None else base,
        reviewed_facts=_reviewed_facts() if reviewed_facts is None else reviewed_facts,
        surface=surface,
        repair_count=repair_count,
        skill_hash=skill_hash,
        covered_deferred_intent_ids=covered_deferred_intent_ids,
        supersedes_draft_hash=supersedes_draft_hash,
        supplied_draft_hash=supplied_draft_hash,
        supplied_reviewed_anchor_hash=supplied_reviewed_anchor_hash,
    )


def test_pipeline_proposal_is_recursively_immutable_and_detached() -> None:
    pipeline = _pipeline()
    proposal = _create_proposal(pipeline=pipeline)

    assert isinstance(proposal.pipeline, MappingProxyType)
    assert isinstance(proposal.pipeline["source"], MappingProxyType)
    assert isinstance(proposal.pipeline["source"]["options"]["columns"], tuple)
    assert isinstance(proposal.pipeline["nodes"][0]["options"]["rules"][0], MappingProxyType)

    with pytest.raises(TypeError):
        proposal.pipeline["nodes"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        proposal.pipeline["source"]["options"]["new"] = "value"  # type: ignore[index]
    with pytest.raises(AttributeError):
        proposal.pipeline["nodes"].append({})

    pipeline["source"]["options"]["columns"].append("late_mutation")
    pipeline["nodes"][0]["options"]["rules"][0]["column"] = "changed"
    assert proposal.pipeline["source"]["options"]["columns"] == ("name", "score")
    assert proposal.pipeline["nodes"][0]["options"]["rules"][0]["column"] == "name"


def test_secret_ref_markers_are_preserved_in_canonical_pipeline() -> None:
    pipeline = _pipeline()
    proposal = _create_proposal(pipeline=pipeline)

    assert deep_thaw(proposal.pipeline) == pipeline
    assert canonical_json(proposal.pipeline) == canonical_json(pipeline)
    assert proposal.pipeline["source"]["options"]["credentials"] == {"secret_ref": "CSV_API_TOKEN"}
    assert proposal.pipeline["outputs"][0]["options"]["destination"] == {"secret_ref": "OUTPUT_PATH"}


def test_reviewed_anchor_hash_uses_versioned_deep_frozen_facts() -> None:
    facts = _reviewed_facts()
    frozen_shape = {
        "schema": "guided.reviewed-anchors.v1",
        "facts": facts,
    }

    expected = stable_hash(frozen_shape)
    assert reviewed_anchor_hash(facts) == expected
    assert reviewed_anchor_hash(facts) == reviewed_anchor_hash(_reviewed_facts())

    facts["source"]["columns"].append("late_mutation")
    assert reviewed_anchor_hash(facts) != expected


def test_draft_hash_has_explicit_v2_expanded_envelope_golden_preimage() -> None:
    """v2 deliberately supersedes the design's older pipeline-only v1 hash."""
    proposal = _create_proposal()
    expected_preimage = {
        "schema": "composer.pipeline-proposal-envelope.v2",
        "pipeline": _pipeline(),
        "base": {
            "kind": "present",
            "state_id": str(_STATE_ID),
            "composition_content_hash": "b" * 64,
        },
        "reviewed_anchor_hash": reviewed_anchor_hash(_reviewed_facts()),
        "surface": "guided_full",
        "repair_count": 1,
        "skill_hash": "c" * 64,
        "covered_deferred_intent_ids": [_INTENT_A, _INTENT_B],
        "supersedes_draft_hash": "d" * 64,
    }
    expected_canonical_json = (
        '{"base":{"composition_content_hash":"' + "b" * 64 + '","kind":"present","state_id":"00000000-0000-4000-8000-000000000001"},'
        '"covered_deferred_intent_ids":["00000000-0000-4000-8000-00000000000a",'
        '"00000000-0000-4000-8000-00000000000b"],"pipeline":{"edges":[{"from":"rows","to":"clean"}],'
        '"nodes":[{"id":"clean","options":{"rules":[{"column":"name","operation":"strip"}]},'
        '"plugin":"normalize"}],"outputs":[{"options":{"destination":{"secret_ref":"OUTPUT_PATH"}},'
        '"plugin":"json","sink_name":"clean"}],"source":{"on_success":"rows","options":{"columns":'
        '["name","score"],"credentials":{"secret_ref":"CSV_API_TOKEN"}},"plugin":"csv"}},"repair_count":1,'
        '"reviewed_anchor_hash":"'
        + reviewed_anchor_hash(_reviewed_facts())
        + '","schema":"composer.pipeline-proposal-envelope.v2","skill_hash":"'
        + "c" * 64
        + '","supersedes_draft_hash":"'
        + "d" * 64
        + '","surface":"guided_full"}'
    )

    assert canonical_json(expected_preimage) == expected_canonical_json
    assert proposal.draft_hash == stable_hash(expected_preimage)
    assert proposal.draft_hash == pipeline_draft_hash(
        pipeline=proposal.pipeline,
        base=proposal.base,
        reviewed_anchor_hash=proposal.reviewed_anchor_hash,
        surface=proposal.surface,
        repair_count=proposal.repair_count,
        skill_hash=proposal.skill_hash,
        covered_deferred_intent_ids=proposal.covered_deferred_intent_ids,
        supersedes_draft_hash=proposal.supersedes_draft_hash,
    )


@pytest.mark.parametrize(
    "change",
    [
        "pipeline",
        "base_kind",
        "base_state_id",
        "base_content_hash",
        "reviewed_anchor",
        "surface",
        "repair_count",
        "skill_hash",
        "covered_content",
        "covered_order",
        "supersedes",
    ],
)
def test_draft_hash_covers_every_authority_bearing_envelope_field(change: str) -> None:
    baseline = _create_proposal()
    kwargs: dict[str, Any] = {}
    if change == "pipeline":
        changed_pipeline = _pipeline()
        changed_pipeline["nodes"][0]["plugin"] = "filter"
        kwargs["pipeline"] = changed_pipeline
    elif change == "base_kind":
        kwargs["base"] = AbsentBase()
    elif change == "base_state_id":
        kwargs["base"] = PresentBase(
            state_id=UUID("00000000-0000-4000-8000-000000000002"),
            composition_content_hash="b" * 64,
        )
    elif change == "base_content_hash":
        kwargs["base"] = PresentBase(state_id=_STATE_ID, composition_content_hash="e" * 64)
    elif change == "reviewed_anchor":
        facts = _reviewed_facts()
        facts["source"]["columns"].append("extra")
        kwargs["reviewed_facts"] = facts
    elif change == "surface":
        kwargs["surface"] = PlannerSurface.GUIDED_STAGED
    elif change == "repair_count":
        kwargs["repair_count"] = 2
    elif change == "skill_hash":
        kwargs["skill_hash"] = "e" * 64
    elif change == "covered_content":
        kwargs["covered_deferred_intent_ids"] = (_INTENT_A,)
    elif change == "covered_order":
        kwargs["covered_deferred_intent_ids"] = (_INTENT_B, _INTENT_A)
    elif change == "supersedes":
        kwargs["supersedes_draft_hash"] = None

    assert _create_proposal(**kwargs).draft_hash != baseline.draft_hash


def test_absent_and_present_bases_have_unambiguous_serialized_tags() -> None:
    absent = _create_proposal(base=AbsentBase()).to_dict()["base"]
    present = _create_proposal().to_dict()["base"]

    assert absent == {"kind": "absent"}
    assert present == {
        "kind": "present",
        "state_id": str(_STATE_ID),
        "composition_content_hash": "b" * 64,
    }


@pytest.mark.parametrize(
    "base_factory",
    [
        lambda: PresentBase(
            state_id="00000000-0000-4000-8000-000000000001",  # type: ignore[arg-type]
            composition_content_hash="b" * 64,
        ),
        lambda: PresentBase(state_id=_STATE_ID, composition_content_hash=""),
        lambda: PresentBase(state_id=_STATE_ID, composition_content_hash="B" * 64),
        lambda: PresentBase(state_id=_STATE_ID, composition_content_hash="not-a-hash"),
        lambda: AbsentBase(kind="present"),  # type: ignore[arg-type]
    ],
)
def test_base_rejects_invalid_uuid_tag_or_hash(base_factory: Callable[[], AbsentBase | PresentBase]) -> None:
    with pytest.raises(AuditIntegrityError):
        _create_proposal(base=base_factory())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repair_count", -1),
        ("repair_count", True),
        ("skill_hash", ""),
        ("skill_hash", "C" * 64),
        ("skill_hash", "c" * 63),
        ("supersedes_draft_hash", ""),
        ("supersedes_draft_hash", "not-a-hash"),
    ],
)
def test_proposal_rejects_negative_repair_or_malformed_nonempty_hashes(field: str, value: object) -> None:
    with pytest.raises(AuditIntegrityError):
        _create_proposal(**{field: value})


@pytest.mark.parametrize(
    "intent_ids",
    [
        (_INTENT_A, _INTENT_A),
        ("",),
        ("not-a-uuid",),
        ("00000000-0000-4000-8000-00000000000A",),
        (UUID(_INTENT_A),),
    ],
)
def test_proposal_rejects_duplicate_or_malformed_covered_intent_ids(intent_ids: tuple[object, ...]) -> None:
    with pytest.raises(AuditIntegrityError):
        _create_proposal(covered_deferred_intent_ids=intent_ids)  # type: ignore[arg-type]


def test_proposal_preserves_covered_intent_order() -> None:
    proposal = _create_proposal(covered_deferred_intent_ids=(_INTENT_B, _INTENT_A))
    assert proposal.covered_deferred_intent_ids == (_INTENT_B, _INTENT_A)
    assert proposal.to_dict()["covered_deferred_intent_ids"] == [_INTENT_B, _INTENT_A]


def test_construction_rejects_supplied_draft_or_reviewed_anchor_mismatch() -> None:
    with pytest.raises(AuditIntegrityError, match="draft_hash mismatch"):
        _create_proposal(supplied_draft_hash="0" * 64)
    with pytest.raises(AuditIntegrityError, match="reviewed_anchor_hash mismatch"):
        _create_proposal(supplied_reviewed_anchor_hash="0" * 64)


def test_direct_construction_rejects_draft_hash_mismatch() -> None:
    with pytest.raises(AuditIntegrityError, match="draft_hash mismatch"):
        replace(_create_proposal(), draft_hash="0" * 64)


def test_strict_restore_round_trip_and_hash_reverification() -> None:
    proposal = _create_proposal()
    restored = PipelineProposal.from_dict(proposal.to_dict(), reviewed_facts=_reviewed_facts())
    assert restored == proposal

    bad_draft = proposal.to_dict()
    bad_draft["draft_hash"] = "0" * 64
    with pytest.raises(AuditIntegrityError, match="draft_hash mismatch"):
        PipelineProposal.from_dict(bad_draft, reviewed_facts=_reviewed_facts())

    bad_anchor = proposal.to_dict()
    bad_anchor["reviewed_anchor_hash"] = "0" * 64
    with pytest.raises(AuditIntegrityError, match="reviewed_anchor_hash mismatch"):
        PipelineProposal.from_dict(bad_anchor, reviewed_facts=_reviewed_facts())


@pytest.mark.parametrize("field", ["draft_hash", "reviewed_anchor_hash"])
@pytest.mark.parametrize("corrupt_value", [None, "", 42])
def test_restore_rejects_null_empty_or_wrong_type_required_hash_before_resealing(
    field: str,
    corrupt_value: object,
) -> None:
    payload = _create_proposal().to_dict()
    payload[field] = corrupt_value  # type: ignore[literal-required,typeddict-item]

    with pytest.raises(AuditIntegrityError, match=rf"PipelineProposal {field} must be exactly 64 lowercase hexadecimal characters"):
        PipelineProposal.from_dict(payload, reviewed_facts=_reviewed_facts())


@pytest.mark.parametrize("field", ["draft_hash", "reviewed_anchor_hash"])
def test_restore_rejects_missing_required_hash(field: str) -> None:
    payload = _create_proposal().to_dict()
    del payload[field]  # type: ignore[literal-required,misc]

    with pytest.raises(AuditIntegrityError, match="persisted fields are malformed"):
        PipelineProposal.from_dict(payload, reviewed_facts=_reviewed_facts())


@pytest.mark.parametrize("mutation", ["extra_top_level", "missing_top_level", "unknown_base", "extra_base"])
def test_restore_rejects_noncanonical_envelope_shapes(mutation: str) -> None:
    payload = _create_proposal().to_dict()
    if mutation == "extra_top_level":
        payload["rationale"] = "model-authored text must not enter the envelope"
    elif mutation == "missing_top_level":
        del payload["skill_hash"]
    elif mutation == "unknown_base":
        payload["base"] = {"kind": "wildcard"}
    elif mutation == "extra_base":
        payload["base"]["version"] = 3

    with pytest.raises(AuditIntegrityError):
        PipelineProposal.from_dict(payload, reviewed_facts=_reviewed_facts())


def test_envelope_defines_no_duplicate_topology_dataclasses_or_rationale() -> None:
    import elspeth.web.composer.pipeline_proposal as module

    dataclass_names = {
        name
        for name, value in vars(module).items()
        if isinstance(value, type) and value.__module__ == module.__name__ and is_dataclass(value)
    }
    assert dataclass_names == {"AbsentBase", "PresentBase", "PipelineProposal"}
    assert [field.name for field in fields(PipelineProposal)] == [
        "pipeline",
        "draft_hash",
        "base",
        "reviewed_anchor_hash",
        "surface",
        "repair_count",
        "skill_hash",
        "covered_deferred_intent_ids",
        "supersedes_draft_hash",
    ]
    assert "rationale" not in PipelineProposal.__slots__
    assert "why" not in PipelineProposal.__slots__


def test_composition_content_hash_exactly_matches_moved_legacy_preimage() -> None:
    class _State:
        def to_dict(self) -> dict[str, Any]:
            return {
                "version": 17,
                "sources": {"main": {"plugin": "csv"}},
                "nodes": [{"id": "clean"}],
                "edges": [{"from": "main", "to": "clean"}],
                "outputs": [{"sink_name": "clean"}],
                "metadata": {"name": "example"},
                "composer_meta": {"guided_session": {"step": "step_4_wire"}},
            }

    state = _State()
    state_d = state.to_dict()
    legacy_hash = stable_hash(
        {
            "sources": state_d["sources"],
            "nodes": state_d["nodes"],
            "edges": state_d["edges"],
            "outputs": state_d["outputs"],
            "metadata": state_d["metadata"],
        }
    )
    assert composition_content_hash(state) == legacy_hash


@pytest.mark.parametrize("pipeline", [[], {"nodes": float("nan")}])
def test_proposal_rejects_non_mapping_or_noncanonical_pipeline(pipeline: object) -> None:
    with pytest.raises(AuditIntegrityError):
        _create_proposal(pipeline=pipeline)  # type: ignore[arg-type]
