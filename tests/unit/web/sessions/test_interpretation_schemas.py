"""Wire-schema tests for Phase 5b Task 3.

Covers the pydantic models that mirror the Phase 5b interpretation-event
contract types:

- ``InterpretationEventResponse`` — read-side wire mirror of
  ``InterpretationEventRecord``.
- ``InterpretationResolveRequest`` — POST body for
  ``/api/sessions/{id}/interpretations/{event_id}/resolve``.  Carries the
  choice + (optional) ``amended_value``, plus a model-validator that
  enforces the choice/value consistency contract.
- ``InterpretationResolveResponse`` — strict response wrapping the
  resolved event and the new composition state produced by the resolve.
- ``InterpretationOptOutResponse`` — strict response for the per-session
  "stop asking" opt-out route.
- ``ListInterpretationEventsResponse`` — strict response for the list
  route.

The shared content-check helpers (``_validate_accepted_value_content``
and ``_reject_credential_shaped_content``) live in
``elspeth.web.validation`` per Phase 5b spec F-34.  They are tested
indirectly via ``InterpretationResolveRequest`` (the schema-layer
defense-in-depth) and directly in ``test_validation_helpers.py`` where
the tool-boundary path also exercises them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from elspeth.contracts.composer_interpretation import (
    InterpretationChoice,
    InterpretationKind,
    InterpretationSource,
)
from elspeth.web.sessions.schemas import (
    CompositionStateResponse,
    InterpretationEventResponse,
    InterpretationOptOutResponse,
    InterpretationResolveRequest,
    InterpretationResolveResponse,
    ListInterpretationEventsResponse,
)


def _valid_event_kwargs() -> dict[str, object]:
    """Build a complete-field kwargs dict for a resolved user_approved event.

    All optional fields populated so per-field rejection tests can prune
    individual keys to confirm the cap fires per field, not as a side
    effect of another required field being absent.
    """
    return {
        "id": uuid4(),
        "session_id": uuid4(),
        "composition_state_id": uuid4(),
        "affected_node_id": "transform_rate_coolness",
        "tool_call_id": "call_abc123",
        "user_term": "cool",
        "kind": "vague_term",
        "llm_draft": "visually appealing and well-organised",
        "accepted_value": "visually appealing and well-organised",
        "choice": "accepted_as_drafted",
        "created_at": datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
        "resolved_at": datetime(2026, 5, 18, 12, 0, 5, tzinfo=UTC),
        "actor": "user:abc",
        "interpretation_source": "user_approved",
        "model_identifier": "anthropic/claude-opus-4-7",
        "model_version": "claude-opus-4-7-20260101",
        "provider": "anthropic",
        "composer_skill_hash": "a" * 64,
        "arguments_hash": "b" * 64,
        "hash_domain_version": "v2",
        "runtime_model_identifier_at_resolve": "anthropic/claude-opus-4-7",
        "runtime_model_version_at_resolve": "claude-opus-4-7-20260101",
        "resolved_prompt_template_hash": "c" * 64,
    }


class TestInterpretationEventResponse:
    def test_closed_list_fields_use_contract_enums(self) -> None:
        """Wire schema must not hand-duplicate the interpretation closed lists."""
        assert InterpretationEventResponse.model_fields["choice"].annotation is InterpretationChoice
        assert InterpretationEventResponse.model_fields["interpretation_source"].annotation is InterpretationSource
        assert InterpretationEventResponse.model_fields["kind"].annotation == InterpretationKind | None

    def test_full_user_approved_row_constructs(self) -> None:
        event = InterpretationEventResponse(**_valid_event_kwargs())  # type: ignore[arg-type]
        assert event.choice == "accepted_as_drafted"
        assert event.interpretation_source == "user_approved"
        assert event.kind == "vague_term"
        assert event.hash_domain_version == "v2"

    def test_opted_out_row_constructs_with_nullable_fields(self) -> None:
        kwargs = _valid_event_kwargs()
        # opted_out rows have NULL surface and provenance fields per the
        # InterpretationEventRecord contract (composer_interpretation.py).
        kwargs.update(
            composition_state_id=None,
            affected_node_id=None,
            tool_call_id=None,
            user_term=None,
            kind=None,
            llm_draft=None,
            accepted_value=None,
            model_identifier=None,
            model_version=None,
            provider=None,
            composer_skill_hash=None,
            arguments_hash=None,
            hash_domain_version=None,
            runtime_model_identifier_at_resolve=None,
            runtime_model_version_at_resolve=None,
            resolved_prompt_template_hash=None,
            choice="opted_out",
            interpretation_source="auto_interpreted_opt_out",
        )
        event = InterpretationEventResponse(**kwargs)  # type: ignore[arg-type]
        assert event.choice == "opted_out"
        assert event.composition_state_id is None
        assert event.composer_skill_hash is None

    def test_rejects_extra_field(self) -> None:
        kwargs = _valid_event_kwargs()
        kwargs["mystery_field"] = "boom"
        with pytest.raises(ValidationError) as exc:
            InterpretationEventResponse(**kwargs)  # type: ignore[arg-type]
        # Pydantic v2 raises "extra_forbidden" for ConfigDict(extra="forbid").
        assert any(err["type"] == "extra_forbidden" for err in exc.value.errors())

    def test_rejects_invalid_choice(self) -> None:
        kwargs = _valid_event_kwargs()
        kwargs["choice"] = "not_a_real_choice"
        with pytest.raises(ValidationError):
            InterpretationEventResponse(**kwargs)  # type: ignore[arg-type]

    def test_rejects_invalid_interpretation_source(self) -> None:
        kwargs = _valid_event_kwargs()
        kwargs["interpretation_source"] = "fabricated_source"
        with pytest.raises(ValidationError):
            InterpretationEventResponse(**kwargs)  # type: ignore[arg-type]

    def test_rejects_invalid_kind(self) -> None:
        kwargs = _valid_event_kwargs()
        kwargs["kind"] = "fabricated_kind"
        with pytest.raises(ValidationError):
            InterpretationEventResponse(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field", "length"),
        [
            ("affected_node_id", 257),
            ("tool_call_id", 257),
            ("user_term", 8193),
            ("llm_draft", 8193),
            ("accepted_value", 8193),
            ("actor", 257),
            ("model_identifier", 257),
            ("model_version", 129),
            ("provider", 65),
            ("composer_skill_hash", 65),
            ("arguments_hash", 65),
            ("hash_domain_version", 17),
            ("runtime_model_identifier_at_resolve", 257),
            ("runtime_model_version_at_resolve", 129),
            ("resolved_prompt_template_hash", 65),
        ],
    )
    def test_max_length_caps_enforced(self, field: str, length: int) -> None:
        kwargs = _valid_event_kwargs()
        kwargs[field] = "x" * length
        with pytest.raises(ValidationError) as exc:
            InterpretationEventResponse(**kwargs)  # type: ignore[arg-type]
        # Make sure the failing field is the one we exceeded — otherwise the
        # test could pass for an unrelated reason (e.g. an unrelated default
        # also breaching a cap).
        assert any(err["loc"] == (field,) for err in exc.value.errors())

    def test_actor_min_length_one(self) -> None:
        kwargs = _valid_event_kwargs()
        kwargs["actor"] = ""
        with pytest.raises(ValidationError):
            InterpretationEventResponse(**kwargs)  # type: ignore[arg-type]

    def test_response_is_frozen(self) -> None:
        event = InterpretationEventResponse(**_valid_event_kwargs())  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            event.choice = "amended"  # type: ignore[misc]


class TestInterpretationResolveRequest:
    def test_accepts_as_drafted_without_amended_value(self) -> None:
        req = InterpretationResolveRequest(choice="accepted_as_drafted")
        assert req.choice == "accepted_as_drafted"
        assert req.amended_value is None

    def test_accepts_amended_with_clean_value(self) -> None:
        req = InterpretationResolveRequest(
            choice="amended",
            amended_value="visually appealing and easy to use",
        )
        assert req.choice == "amended"
        assert req.amended_value == "visually appealing and easy to use"

    def test_rejects_amended_without_value(self) -> None:
        with pytest.raises(ValidationError) as exc:
            InterpretationResolveRequest(choice="amended")
        assert "amended_value is required" in str(exc.value)

    def test_rejects_amended_with_empty_value(self) -> None:
        # Empty string is falsy; the validator forbids it for choice=amended
        # because a zero-length amendment cannot be the user's intended
        # interpretation.  The schema-layer cap is min_length=0 on the field
        # itself; the model_validator catches the empty case.
        with pytest.raises(ValidationError) as exc:
            InterpretationResolveRequest(choice="amended", amended_value="")
        assert "amended_value is required" in str(exc.value)

    def test_rejects_accepted_as_drafted_with_amended_value(self) -> None:
        with pytest.raises(ValidationError) as exc:
            InterpretationResolveRequest(
                choice="accepted_as_drafted",
                amended_value="not allowed here",
            )
        assert "must be omitted" in str(exc.value)

    def test_rejects_invalid_choice(self) -> None:
        with pytest.raises(ValidationError):
            InterpretationResolveRequest(choice="opted_out")  # type: ignore[arg-type]

    def test_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError) as exc:
            InterpretationResolveRequest(  # type: ignore[call-arg]
                choice="accepted_as_drafted",
                rogue_field="ignored",
            )
        assert any(err["type"] == "extra_forbidden" for err in exc.value.errors())

    def test_amended_value_max_length_cap(self) -> None:
        with pytest.raises(ValidationError):
            InterpretationResolveRequest(
                choice="amended",
                amended_value="x" * 8193,
            )

    def test_amended_value_rejects_template_metachars_open(self) -> None:
        with pytest.raises(ValidationError) as exc:
            InterpretationResolveRequest(
                choice="amended",
                amended_value="something {{ injected }} here",
            )
        assert "template metacharacters" in str(exc.value)

    def test_amended_value_rejects_template_metachars_close_only(self) -> None:
        with pytest.raises(ValidationError) as exc:
            InterpretationResolveRequest(
                choice="amended",
                amended_value="closing }} brace",
            )
        assert "template metacharacters" in str(exc.value)

    def test_amended_value_permits_newline(self) -> None:
        # Newlines are permitted: pipeline_decision and vague_term amended
        # values can legitimately span lines (a multi-paragraph rubric or
        # decision rationale). The per-line length cap below still bounds
        # pathological single-line pastes.
        req = InterpretationResolveRequest(
            choice="amended",
            amended_value="line one\nline two",
        )
        assert req.amended_value == "line one\nline two"

    def test_amended_value_permits_carriage_return(self) -> None:
        # CRLF-terminated content from operators on Windows must round-trip.
        req = InterpretationResolveRequest(
            choice="amended",
            amended_value="line one\r\nline two",
        )
        assert req.amended_value == "line one\r\nline two"

    def test_amended_value_rejects_other_control_chars(self) -> None:
        # \x00 (NUL), \x07 (BEL), \x0b (VT), \x0c (FF), \x1b (ESC), \x1f
        # (US), and \x7f (DEL) must be rejected. \t \n \r are explicitly
        # permitted.
        for char in ("\x00", "\x07", "\x0b", "\x0c", "\x1b", "\x1f", "\x7f"):
            with pytest.raises(ValidationError):
                InterpretationResolveRequest(
                    choice="amended",
                    amended_value=f"prefix{char}suffix",
                )

    def test_amended_value_permits_horizontal_tab(self) -> None:
        req = InterpretationResolveRequest(
            choice="amended",
            amended_value="prefix\tsuffix",
        )
        assert req.amended_value == "prefix\tsuffix"

    def test_amended_value_per_line_length_cap_single_line(self) -> None:
        long_single_line = "a" * 1025
        with pytest.raises(ValidationError) as exc:
            InterpretationResolveRequest(
                choice="amended",
                amended_value=long_single_line,
            )
        assert "1024-character" in str(exc.value)

    def test_amended_value_per_line_length_cap_multi_line(self) -> None:
        # The per-line cap must fire even when the offending line is in
        # the middle of a multi-line payload — the previous single-line
        # cap silently allowed this case.
        payload = "ok\n" + ("b" * 1025) + "\ntail"
        with pytest.raises(ValidationError) as exc:
            InterpretationResolveRequest(
                choice="amended",
                amended_value=payload,
            )
        assert "1024-character" in str(exc.value)

    def test_amended_value_rejects_aws_access_key(self) -> None:
        with pytest.raises(ValidationError) as exc:
            InterpretationResolveRequest(
                choice="amended",
                amended_value="key is AKIAIOSFODNN7EXAMPLE here",  # secret-scan: allow-this-line
            )
        assert "credential" in str(exc.value).lower()

    def test_amended_value_rejects_bearer_token(self) -> None:
        with pytest.raises(ValidationError) as exc:
            InterpretationResolveRequest(
                choice="amended",
                amended_value="auth Bearer abcdefghijklmnopqrstuv",
            )
        assert "credential" in str(exc.value).lower()

    def test_amended_value_rejects_github_pat(self) -> None:
        with pytest.raises(ValidationError):
            InterpretationResolveRequest(
                choice="amended",
                amended_value="token ghp_" + "a" * 36,
            )

    def test_amended_value_rejects_jwt_shape(self) -> None:
        # A contiguous three-segment base64url string (no whitespace between
        # segments) MUST trip the JWT rejection.
        jwt_like = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcDEF123_-xyz"
        with pytest.raises(ValidationError):
            InterpretationResolveRequest(
                choice="amended",
                amended_value=f"token={jwt_like}",
            )

    def test_amended_value_permits_benign_prose_with_periods(self) -> None:
        # F-32 regression: prose sentences with periods (e.g. "well-organized,
        # and easy to use.") must not be misclassified as JWTs.  The JWT
        # pattern requires contiguous base64url segments — periods in prose
        # have whitespace around them, so the contiguous match fails.
        req = InterpretationResolveRequest(
            choice="amended",
            amended_value=("The term 'cool' means visually appealing, well-organized, and easy to use."),
        )
        assert req.amended_value is not None

    def test_amended_value_rejects_anthropic_api_key(self) -> None:
        with pytest.raises(ValidationError):
            InterpretationResolveRequest(
                choice="amended",
                amended_value="key=sk-ant-" + "x" * 50,
            )

    def test_amended_value_rejects_openai_api_key(self) -> None:
        with pytest.raises(ValidationError):
            InterpretationResolveRequest(
                choice="amended",
                amended_value="key=sk-" + "X" * 48,
            )


class TestInterpretationResolveResponse:
    def test_constructs_and_is_strict(self) -> None:
        event = InterpretationEventResponse(**_valid_event_kwargs())  # type: ignore[arg-type]
        state = CompositionStateResponse(
            id=str(uuid4()),
            session_id=str(event.session_id),
            version=2,
            is_valid=True,
            created_at=datetime(2026, 5, 18, 12, 0, 6, tzinfo=UTC),
        )
        response = InterpretationResolveResponse(event=event, new_state=state)
        assert response.event.id == event.id
        assert response.new_state.version == 2

    def test_rejects_extra_field(self) -> None:
        event = InterpretationEventResponse(**_valid_event_kwargs())  # type: ignore[arg-type]
        state = CompositionStateResponse(
            id=str(uuid4()),
            session_id=str(event.session_id),
            version=2,
            is_valid=True,
            created_at=datetime(2026, 5, 18, 12, 0, 6, tzinfo=UTC),
        )
        with pytest.raises(ValidationError):
            InterpretationResolveResponse(  # type: ignore[call-arg]
                event=event,
                new_state=state,
                extra="boom",
            )


class TestInterpretationOptOutResponse:
    def test_constructs(self) -> None:
        resp = InterpretationOptOutResponse(
            session_id=uuid4(),
            interpretation_review_disabled=True,
            opted_out_at=datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
        )
        assert resp.interpretation_review_disabled is True

    def test_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError):
            InterpretationOptOutResponse(  # type: ignore[call-arg]
                session_id=uuid4(),
                interpretation_review_disabled=True,
                opted_out_at=datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
                ghost_field=42,
            )


class TestListInterpretationEventsResponse:
    def test_empty_list_is_valid(self) -> None:
        response = ListInterpretationEventsResponse(events=[])
        assert response.events == []

    def test_with_events(self) -> None:
        event = InterpretationEventResponse(**_valid_event_kwargs())  # type: ignore[arg-type]
        response = ListInterpretationEventsResponse(events=[event])
        assert len(response.events) == 1

    def test_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError):
            ListInterpretationEventsResponse(events=[], extra="forbidden")  # type: ignore[call-arg]
