"""Unit tests for the composer-interpretation contract (Phase 5b Task 1).

These tests pin the closed enums, the frozen-dataclass record shape, and the
active hash-domain frozenset. The record is a Tier-1 read-side audit type — the
suite enforces required-field discipline (TypeError on missing kwargs) and
frozen-attribute discipline (FrozenInstanceError on assignment).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from elspeth.contracts.composer_interpretation import (
    INTERPRETATION_HASH_DOMAIN_V1,
    INTERPRETATION_HASH_DOMAIN_V2,
    InterpretationChoice,
    InterpretationEventRecord,
    InterpretationKind,
    InterpretationSource,
)


def _resolved_record_kwargs() -> dict[str, object]:
    """Kwargs for a fully resolved user_approved row (all fields populated)."""
    return {
        "id": uuid4(),
        "session_id": uuid4(),
        "composition_state_id": uuid4(),
        "affected_node_id": "node-1",
        "tool_call_id": "tool-call-abc",
        "user_term": "cool",
        "kind": InterpretationKind.VAGUE_TERM,
        "llm_draft": "subjectively interesting or impressive",
        "accepted_value": "subjectively interesting or impressive",
        "choice": InterpretationChoice.ACCEPTED_AS_DRAFTED,
        "created_at": datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
        "resolved_at": datetime(2026, 5, 18, 12, 0, 30, tzinfo=UTC),
        "actor": "john@pgpl.net",
        "model_identifier": "anthropic/claude-opus-4-7",
        "model_version": "2026-01-15",
        "provider": "anthropic",
        "composer_skill_hash": "a" * 64,
        "arguments_hash": "b" * 64,
        "hash_domain_version": "v2",
        "interpretation_source": InterpretationSource.USER_APPROVED,
        "runtime_model_identifier_at_resolve": "anthropic/claude-opus-4-7",
        "runtime_model_version_at_resolve": "2026-01-15",
        "resolved_prompt_template_hash": "c" * 64,
    }


def _opted_out_record_kwargs() -> dict[str, object]:
    """Kwargs for an auto_interpreted_opt_out row (all nullable fields None)."""
    return {
        "id": uuid4(),
        "session_id": uuid4(),
        "composition_state_id": None,
        "affected_node_id": None,
        "tool_call_id": None,
        "user_term": None,
        "kind": None,
        "llm_draft": None,
        "accepted_value": None,
        "choice": InterpretationChoice.OPTED_OUT,
        "created_at": datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
        "resolved_at": datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
        "actor": "john@pgpl.net",
        "model_identifier": None,
        "model_version": None,
        "provider": None,
        "composer_skill_hash": None,
        "arguments_hash": None,
        "hash_domain_version": None,
        "interpretation_source": InterpretationSource.AUTO_INTERPRETED_OPT_OUT,
        "runtime_model_identifier_at_resolve": None,
        "runtime_model_version_at_resolve": None,
        "resolved_prompt_template_hash": None,
    }


def _no_surfaces_record_kwargs() -> dict[str, object]:
    """Kwargs for an auto_interpreted_no_surfaces row."""
    return {
        "id": uuid4(),
        "session_id": uuid4(),
        "composition_state_id": None,
        "affected_node_id": None,
        "tool_call_id": None,
        "user_term": None,
        "kind": InterpretationKind.VAGUE_TERM,
        "llm_draft": None,
        "accepted_value": None,
        "choice": InterpretationChoice.OPTED_OUT,
        "created_at": datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
        "resolved_at": datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
        "actor": "composer-llm",
        "model_identifier": "anthropic/claude-opus-4-7",
        "model_version": "2026-01-15",
        "provider": "anthropic",
        "composer_skill_hash": "a" * 64,
        "arguments_hash": None,
        "hash_domain_version": None,
        "interpretation_source": InterpretationSource.AUTO_INTERPRETED_NO_SURFACES,
        "runtime_model_identifier_at_resolve": None,
        "runtime_model_version_at_resolve": None,
        "resolved_prompt_template_hash": None,
    }


def _surface_opt_out_record_kwargs() -> dict[str, object]:
    """Kwargs for a surface-specific auto_interpreted_opt_out row."""
    return _resolved_record_kwargs() | {
        "user_term": "inline_source_url_list",
        "kind": InterpretationKind.INVENTED_SOURCE,
        "llm_draft": "https://example.gov.au",
        "accepted_value": "https://example.gov.au",
        "choice": InterpretationChoice.OPTED_OUT,
        "actor": "composer-llm",
        "interpretation_source": InterpretationSource.AUTO_INTERPRETED_OPT_OUT,
        "arguments_hash": "d" * 64,
        "hash_domain_version": "v2",
        "runtime_model_identifier_at_resolve": None,
        "runtime_model_version_at_resolve": None,
        "resolved_prompt_template_hash": None,
    }


def test_interpretation_choice_has_exactly_five_values() -> None:
    """InterpretationChoice is a closed enum with exactly 5 members."""
    members = {member.value for member in InterpretationChoice}
    assert members == {
        "pending",
        "accepted_as_drafted",
        "amended",
        "opted_out",
        "abandoned",
    }
    assert len(InterpretationChoice) == 5


def test_interpretation_source_has_exactly_three_values() -> None:
    """InterpretationSource is a closed enum with exactly 3 members."""
    members = {member.value for member in InterpretationSource}
    assert members == {
        "user_approved",
        "auto_interpreted_opt_out",
        "auto_interpreted_no_surfaces",
    }
    assert len(InterpretationSource) == 3


def test_interpretation_kind_closed_set() -> None:
    assert [member.value for member in InterpretationKind] == [
        "vague_term",
        "invented_source",
        "llm_prompt_template",
        "pipeline_decision",
        "llm_model_choice",
    ]


def test_v2_hash_domain_includes_kind_and_retires_v1_writes() -> None:
    assert "kind" in INTERPRETATION_HASH_DOMAIN_V2
    assert (
        frozenset(
            {
                "session_id",
                "composition_state_id",
                "affected_node_id",
                "tool_call_id",
                "user_term",
                "kind",
                "llm_draft",
                "accepted_value",
                "actor",
                "model_identifier",
                "model_version",
                "provider",
                "composer_skill_hash",
            }
        )
        == INTERPRETATION_HASH_DOMAIN_V2
    )


def test_record_is_frozen() -> None:
    """Assigning to any field on a constructed record raises FrozenInstanceError."""
    record = InterpretationEventRecord(**_resolved_record_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.actor = "someone-else"  # type: ignore[misc]


def test_record_missing_required_field_raises_typeerror() -> None:
    """Constructing without a required field crashes with TypeError."""
    kwargs = _resolved_record_kwargs()
    del kwargs["actor"]
    with pytest.raises(TypeError):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


def test_record_rejects_literal_cast_choice() -> None:
    """Tier-1 records require enum members, not raw DB strings."""
    kwargs = _resolved_record_kwargs()
    kwargs["choice"] = "accepted_as_drafted"
    with pytest.raises(ValueError, match=r"choice.*InterpretationChoice"):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


def test_record_rejects_literal_cast_interpretation_source() -> None:
    """Tier-1 records require enum members, not raw DB strings."""
    kwargs = _resolved_record_kwargs()
    kwargs["interpretation_source"] = "user_approved"
    with pytest.raises(ValueError, match=r"interpretation_source.*InterpretationSource"):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


def test_record_roundtrips_through_asdict() -> None:
    """dataclasses.asdict() works on a resolved record."""
    kwargs = _resolved_record_kwargs()
    record = InterpretationEventRecord(**kwargs)
    snapshot = dataclasses.asdict(record)
    # Every field on the record appears as a key in the snapshot.
    field_names = {f.name for f in dataclasses.fields(record)}
    assert set(snapshot.keys()) == field_names
    # Enum values round-trip as their enum members under asdict (StrEnum
    # instances are preserved; asdict does not coerce them to strings).
    assert snapshot["choice"] == InterpretationChoice.ACCEPTED_AS_DRAFTED
    assert snapshot["interpretation_source"] == InterpretationSource.USER_APPROVED


def test_opted_out_record_constructs_successfully() -> None:
    """An auto_interpreted_opt_out row with all nullable fields None constructs cleanly."""
    record = InterpretationEventRecord(**_opted_out_record_kwargs())
    assert record.choice is InterpretationChoice.OPTED_OUT
    assert record.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT
    assert record.composition_state_id is None
    assert record.affected_node_id is None
    assert record.tool_call_id is None
    assert record.user_term is None
    assert record.kind is None
    assert record.llm_draft is None
    assert record.accepted_value is None
    assert record.model_identifier is None
    assert record.model_version is None
    assert record.provider is None
    assert record.composer_skill_hash is None
    assert record.arguments_hash is None
    assert record.hash_domain_version is None
    assert record.runtime_model_identifier_at_resolve is None
    assert record.runtime_model_version_at_resolve is None
    assert record.resolved_prompt_template_hash is None


def test_opted_out_record_roundtrips_through_asdict() -> None:
    """dataclasses.asdict() works on an opt-out record."""
    kwargs = _opted_out_record_kwargs()
    record = InterpretationEventRecord(**kwargs)
    snapshot = dataclasses.asdict(record)
    field_names = {f.name for f in dataclasses.fields(record)}
    assert set(snapshot.keys()) == field_names
    assert snapshot["choice"] == InterpretationChoice.OPTED_OUT
    assert snapshot["interpretation_source"] == InterpretationSource.AUTO_INTERPRETED_OPT_OUT
    # All nullable fields surface as None in the snapshot.
    for nullable in (
        "composition_state_id",
        "affected_node_id",
        "tool_call_id",
        "user_term",
        "kind",
        "llm_draft",
        "accepted_value",
        "model_identifier",
        "model_version",
        "provider",
        "composer_skill_hash",
        "arguments_hash",
        "hash_domain_version",
        "runtime_model_identifier_at_resolve",
        "runtime_model_version_at_resolve",
        "resolved_prompt_template_hash",
    ):
        assert snapshot[nullable] is None


def test_no_surfaces_record_constructs_successfully() -> None:
    """auto_interpreted_no_surfaces has NULL surfaces and populated provenance."""
    record = InterpretationEventRecord(**_no_surfaces_record_kwargs())
    assert record.choice is InterpretationChoice.OPTED_OUT
    assert record.interpretation_source is InterpretationSource.AUTO_INTERPRETED_NO_SURFACES
    assert record.composition_state_id is None
    assert record.affected_node_id is None
    assert record.tool_call_id is None
    assert record.user_term is None
    assert record.kind is InterpretationKind.VAGUE_TERM
    assert record.llm_draft is None
    assert record.model_identifier == "anthropic/claude-opus-4-7"
    assert record.provider == "anthropic"
    assert record.composer_skill_hash == "a" * 64


def test_surface_opt_out_record_constructs_successfully() -> None:
    """Surface-specific opt-out rows carry kind, content, provenance, and V2 hash."""
    record = InterpretationEventRecord(**_surface_opt_out_record_kwargs())
    assert record.choice is InterpretationChoice.OPTED_OUT
    assert record.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT
    assert record.kind is InterpretationKind.INVENTED_SOURCE
    assert record.accepted_value == "https://example.gov.au"
    assert record.hash_domain_version == "v2"
    assert record.arguments_hash == "d" * 64


def test_surface_opt_out_record_rejects_v1_hash_domain() -> None:
    """Surface-specific opt-out rows are current V2 writes, never legacy V1."""
    kwargs = _surface_opt_out_record_kwargs()
    kwargs["hash_domain_version"] = "v1"
    with pytest.raises(ValueError, match=r"surface-specific auto_interpreted_opt_out.*v2"):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs_factory", "source_name"),
    [
        (_opted_out_record_kwargs, "auto_interpreted_opt_out"),
        (_surface_opt_out_record_kwargs, "auto_interpreted_opt_out"),
        (_no_surfaces_record_kwargs, "auto_interpreted_no_surfaces"),
    ],
)
def test_auto_interpreted_rows_require_opted_out_choice(
    kwargs_factory: Callable[[], dict[str, object]],
    source_name: str,
) -> None:
    """Auto-interpreted rows are born resolved as opted_out, not accepted/abandoned."""
    kwargs = kwargs_factory()
    kwargs["choice"] = InterpretationChoice.ABANDONED
    with pytest.raises(ValueError, match=rf"{source_name}.*opted_out"):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


def test_user_approved_record_requires_surface_and_provenance_fields() -> None:
    """USER_APPROVED mirrors ck_interpretation_events_user_approved_required."""
    kwargs = _resolved_record_kwargs()
    kwargs["user_term"] = None
    kwargs["model_identifier"] = None
    with pytest.raises(ValueError, match=r"user_approved.*user_term.*model_identifier"):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


def test_opted_out_record_requires_null_surface_and_provenance_fields() -> None:
    """AUTO_INTERPRETED_OPT_OUT mirrors ck_interpretation_events_source_nullability."""
    kwargs = _opted_out_record_kwargs()
    kwargs["user_term"] = "cool"
    kwargs["provider"] = "anthropic"
    with pytest.raises(ValueError, match=r"auto_interpreted_opt_out.*user_term.*provider"):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs_factory", "overrides", "expected_fields"),
    [
        (
            _opted_out_record_kwargs,
            {"arguments_hash": "b" * 64, "hash_domain_version": "v2"},
            ("arguments_hash", "hash_domain_version"),
        ),
        (
            _surface_opt_out_record_kwargs,
            {"accepted_value": None, "arguments_hash": None, "hash_domain_version": None},
            ("accepted_value", "arguments_hash", "hash_domain_version"),
        ),
    ],
)
def test_shape_violation_summary_includes_lifecycle_field_offenders(
    kwargs_factory: Callable[[], dict[str, object]],
    overrides: dict[str, object],
    expected_fields: tuple[str, ...],
) -> None:
    """The offending-fields summary must include every field the shape validator can flag."""
    kwargs = kwargs_factory()
    kwargs.update(overrides)

    with pytest.raises(ValueError) as exc_info:
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]

    message = str(exc_info.value)
    offender_summary = message.split("offending fields: ", maxsplit=1)[1].split(";", maxsplit=1)[0]
    for expected_field in expected_fields:
        assert expected_field in offender_summary


def test_no_surfaces_record_requires_null_surfaces_and_provenance() -> None:
    """AUTO_INTERPRETED_NO_SURFACES mirrors ck_interpretation_events_no_surfaces_shape."""
    kwargs = _no_surfaces_record_kwargs()
    kwargs["tool_call_id"] = "tool-call-abc"
    kwargs["model_identifier"] = None
    with pytest.raises(ValueError, match=r"auto_interpreted_no_surfaces.*tool_call_id.*model_identifier"):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("choice", "resolved_at"),
    [
        (InterpretationChoice.PENDING, datetime(2026, 5, 18, 12, 0, 30, tzinfo=UTC)),
        (InterpretationChoice.ACCEPTED_AS_DRAFTED, None),
    ],
)
def test_record_validates_choice_resolved_at_coupling(
    choice: InterpretationChoice,
    resolved_at: datetime | None,
) -> None:
    """choice and resolved_at mirror ck_interpretation_events_resolved_at_status."""
    kwargs = _resolved_record_kwargs()
    kwargs["choice"] = choice
    kwargs["resolved_at"] = resolved_at
    if choice is InterpretationChoice.PENDING:
        kwargs["accepted_value"] = None
        kwargs["arguments_hash"] = None
        kwargs["hash_domain_version"] = None
        kwargs["runtime_model_identifier_at_resolve"] = None
        kwargs["runtime_model_version_at_resolve"] = None
        kwargs["resolved_prompt_template_hash"] = None
    with pytest.raises(ValueError, match="resolved_at"):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("choice", "accepted_value"),
    [
        (InterpretationChoice.ACCEPTED_AS_DRAFTED, None),
        (InterpretationChoice.OPTED_OUT, "something cool"),
    ],
)
def test_record_validates_choice_accepted_value_coupling(
    choice: InterpretationChoice,
    accepted_value: str | None,
) -> None:
    """choice and accepted_value mirror ck_interpretation_events_accepted_value_status."""
    kwargs = _resolved_record_kwargs()
    kwargs["choice"] = choice
    kwargs["accepted_value"] = accepted_value
    with pytest.raises(ValueError, match="accepted_value"):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments_hash", "hash_domain_version"),
    [
        ("b" * 64, None),
        (None, "v1"),
    ],
)
def test_record_validates_arguments_hash_domain_coupling(
    arguments_hash: str | None,
    hash_domain_version: str | None,
) -> None:
    """arguments_hash and hash_domain_version are created as a pair."""
    kwargs = _resolved_record_kwargs()
    kwargs["arguments_hash"] = arguments_hash
    kwargs["hash_domain_version"] = hash_domain_version
    with pytest.raises(ValueError, match=r"arguments_hash.*hash_domain_version"):
        InterpretationEventRecord(**kwargs)  # type: ignore[arg-type]


def test_hash_domain_v1_names_are_valid_record_fields() -> None:
    """F-12 CI guard: every name in INTERPRETATION_HASH_DOMAIN_V1 is a real field.

    Catches typos like ``"sesion_id"`` that would silently drop a field out of
    the hash domain without any test surface noticing.
    """
    field_names = {f.name for f in dataclasses.fields(InterpretationEventRecord)}
    missing = INTERPRETATION_HASH_DOMAIN_V1 - field_names
    assert missing == frozenset(), f"INTERPRETATION_HASH_DOMAIN_V1 references unknown fields: {missing}"


def test_hash_domain_v2_names_are_valid_record_fields() -> None:
    """F-12 CI guard: every name in INTERPRETATION_HASH_DOMAIN_V2 is a real field."""
    field_names = {f.name for f in dataclasses.fields(InterpretationEventRecord)}
    missing = INTERPRETATION_HASH_DOMAIN_V2 - field_names
    assert missing == frozenset(), f"INTERPRETATION_HASH_DOMAIN_V2 references unknown fields: {missing}"
