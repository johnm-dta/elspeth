"""Tests for the state-claim grounding detector.

Reproduces the panel-evals T4/T5 corpus from issue
``elspeth-c028f7d186`` (boolean_routing__p1_compliance cell):

- T4 prose: "still uses on_validation_failure: discard" while state has
  ``rejected_records`` — forward contradiction.
- T5 prose: "I just fixed it" while state was unchanged from T4 —
  backward contradiction.

Plus edge cases that protect against false positives (matching claims,
missing source, paraphrases that legitimately discuss state, multi-
output configurations).
"""

from __future__ import annotations

from elspeth.web.composer.state import (
    CompositionState,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.state_claim_grounding import (
    ActionClaim,
    StateClaim,
    check_state_claim_grounding,
    compose_grounded_message,
    extract_action_claims,
    extract_state_claims,
    format_grounding_correction,
    verify_action_claims,
    verify_state_claims,
)


def _state_with_source(on_validation_failure: str = "discard") -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="rows",
            options={"path": "/tmp/x.csv"},
            on_validation_failure=on_validation_failure,
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_outputs(*on_write_failures: str) -> CompositionState:
    outputs = tuple(
        OutputSpec(
            name=f"out_{i}",
            plugin="json",
            options={"path": f"/tmp/out_{i}.json"},
            on_write_failure=owf,
        )
        for i, owf in enumerate(on_write_failures)
    )
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=outputs,
        metadata=PipelineMetadata(),
        version=1,
    )


class TestExtractStateClaims:
    """The detector must recognise the prose patterns documented in the
    panel-evals T4 corpus and common paraphrases, without matching
    unrelated prose."""

    def test_extract_field_colon_value_with_backticks(self) -> None:
        # Exact T4 prose shape from the panel-evals findings.
        prose = "the source still uses `on_validation_failure: discard`"
        claims = extract_state_claims(prose)
        assert len(claims) == 1
        assert claims[0].field_name == "on_validation_failure"
        assert claims[0].scope == "source"
        assert claims[0].claimed_value == "discard"

    def test_extract_field_colon_value_no_backticks(self) -> None:
        prose = "Currently, on_validation_failure: rejected_records is set."
        claims = extract_state_claims(prose)
        assert len(claims) == 1
        assert claims[0].claimed_value == "rejected_records"

    def test_extract_field_is_value_paraphrase(self) -> None:
        prose = "The on_validation_failure is discard right now."
        claims = extract_state_claims(prose)
        assert len(claims) == 1
        assert claims[0].claimed_value == "discard"

    def test_extract_field_set_to_value_paraphrase(self) -> None:
        prose = "on_validation_failure is set to rejected_records."
        claims = extract_state_claims(prose)
        assert len(claims) == 1
        assert claims[0].claimed_value == "rejected_records"

    def test_extract_field_configured_to_value(self) -> None:
        prose = "The source is configured with on_validation_failure configured to quarantine."
        claims = extract_state_claims(prose)
        assert len(claims) == 1
        assert claims[0].claimed_value == "quarantine"

    def test_extract_value_then_field_paraphrase(self) -> None:
        prose = "It uses discard for on_validation_failure."
        claims = extract_state_claims(prose)
        assert len(claims) == 1
        assert claims[0].claimed_value == "discard"

    def test_extract_arrow_separator(self) -> None:
        prose = "on_validation_failure -> rejected_records"
        claims = extract_state_claims(prose)
        assert len(claims) == 1
        assert claims[0].claimed_value == "rejected_records"

    def test_extract_dedupes_same_field_value(self) -> None:
        # If both pattern shapes match the same (field, value), only one
        # StateClaim is returned. Prevents duplicate violations from
        # paraphrase overlap.
        prose = "on_validation_failure: discard. We use discard for on_validation_failure."
        claims = extract_state_claims(prose)
        assert len(claims) == 1

    def test_extract_two_distinct_values(self) -> None:
        # If prose mentions two different values for the same field
        # (rare but possible — e.g., describing a transition), both
        # are extracted.
        prose = "Currently on_validation_failure: discard, but we want on_validation_failure: rejected_records."
        claims = extract_state_claims(prose)
        values = {c.claimed_value for c in claims}
        assert values == {"discard", "rejected_records"}

    def test_extract_case_insensitive_field_and_value(self) -> None:
        prose = "ON_VALIDATION_FAILURE: Discard"
        claims = extract_state_claims(prose)
        assert len(claims) == 1
        assert claims[0].claimed_value == "discard"

    def test_no_extract_when_value_not_in_closed_list(self) -> None:
        # Value tokens outside the closed list are not extracted —
        # avoids false matches against arbitrary identifiers.
        prose = "on_validation_failure: my_custom_handler"
        claims = extract_state_claims(prose)
        assert claims == ()

    def test_no_extract_for_substring_field_match(self) -> None:
        # Field name is anchored on word boundary — a longer identifier
        # that contains "on_validation_failure" as a substring is not
        # matched. (The current regex uses ``\b`` which treats ``_`` as
        # part of the word, so e.g. ``custom_on_validation_failure``
        # also avoids matching.)
        prose = "configured my_custom_on_validation_failure_handler: discard"
        claims = extract_state_claims(prose)
        assert claims == ()

    def test_no_extract_for_unrelated_prose(self) -> None:
        prose = (
            "I configured the pipeline with a CSV source and a JSON output. "
            "The source reads from /tmp/x.csv and the output writes to "
            "/tmp/out.json."
        )
        claims = extract_state_claims(prose)
        assert claims == ()

    def test_extract_on_write_failure_field(self) -> None:
        prose = "The output uses on_write_failure: quarantine."
        claims = extract_state_claims(prose)
        assert len(claims) == 1
        assert claims[0].field_name == "on_write_failure"
        assert claims[0].scope == "outputs"
        assert claims[0].claimed_value == "quarantine"


class TestExtractActionClaims:
    """Action-claim detection must be conservative — only ``just <verb>``
    forms count, to avoid mis-flagging legitimate references to earlier
    turns."""

    def test_extract_just_fixed(self) -> None:
        prose = "I just fixed the workflow behavior."
        claims = extract_action_claims(prose)
        assert len(claims) == 1
        assert claims[0].verb == "fixed"

    def test_extract_just_changed(self) -> None:
        prose = "I just changed the source configuration."
        claims = extract_action_claims(prose)
        assert len(claims) == 1
        assert claims[0].verb == "changed"

    def test_extract_ive_just_updated(self) -> None:
        prose = "I've just updated the pipeline metadata."
        claims = extract_action_claims(prose)
        assert len(claims) == 1
        assert claims[0].verb == "updated"

    def test_no_extract_for_bare_past_tense(self) -> None:
        # Bare "I changed X" without "just" is ambiguous (could refer
        # to an earlier turn in this session). Conservative detector
        # does not flag.
        prose = "I changed the source configuration."
        claims = extract_action_claims(prose)
        assert claims == ()

    def test_no_extract_for_third_person(self) -> None:
        # "The composer just fixed X" — plausible report from server
        # narration, not a model self-claim. Detector is anchored on
        # first person.
        prose = "The composer just fixed the routing."
        claims = extract_action_claims(prose)
        assert claims == ()

    def test_extract_multiple_distinct_verbs(self) -> None:
        prose = "I just fixed the source. I just updated the metadata."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert verbs == {"fixed", "updated"}


class TestVerifyStateClaims:
    """A claim is a violation iff it contradicts state. Matching claims
    are not flagged. Claims about absent state (no source, no outputs)
    are not flagged either — those are hypotheticals, not contradictions."""

    def test_t4_forward_contradiction_is_flagged(self) -> None:
        # The exact T4 case from the panel-evals findings: prose claims
        # discard while state has rejected_records.
        state = _state_with_source(on_validation_failure="rejected_records")
        claims = (
            StateClaim(
                field_name="on_validation_failure",
                scope="source",
                claimed_value="discard",
                span=(0, 30),
            ),
        )
        violations = verify_state_claims(claims, state)
        assert len(violations) == 1
        assert violations[0].kind == "state_claim"
        assert violations[0].claimed_value == "discard"
        assert violations[0].actual_value == "rejected_records"
        assert "rejected_records" in violations[0].explanation

    def test_matching_claim_is_not_flagged(self) -> None:
        state = _state_with_source(on_validation_failure="rejected_records")
        claims = (
            StateClaim(
                field_name="on_validation_failure",
                scope="source",
                claimed_value="rejected_records",
                span=(0, 30),
            ),
        )
        assert verify_state_claims(claims, state) == ()

    def test_claim_about_absent_source_is_not_flagged(self) -> None:
        # No source => the model is discussing a hypothetical, not
        # contradicting state. Empty-state finalize paths handle this.
        state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )
        claims = (
            StateClaim(
                field_name="on_validation_failure",
                scope="source",
                claimed_value="discard",
                span=(0, 30),
            ),
        )
        assert verify_state_claims(claims, state) == ()

    def test_output_claim_matches_one_of_two_outputs(self) -> None:
        # In a multi-output pipeline, a claim about on_write_failure
        # is satisfied if any output has the claimed value.
        state = _state_with_outputs("discard", "quarantine")
        claims = (
            StateClaim(
                field_name="on_write_failure",
                scope="outputs",
                claimed_value="quarantine",
                span=(0, 30),
            ),
        )
        assert verify_state_claims(claims, state) == ()

    def test_output_claim_matches_no_output(self) -> None:
        state = _state_with_outputs("discard", "discard")
        claims = (
            StateClaim(
                field_name="on_write_failure",
                scope="outputs",
                claimed_value="quarantine",
                span=(0, 30),
            ),
        )
        violations = verify_state_claims(claims, state)
        assert len(violations) == 1
        assert violations[0].claimed_value == "quarantine"
        assert violations[0].actual_value == "discard"


class TestVerifyActionClaims:
    """An action-claim is a violation iff no mutation succeeded this turn
    AND state did not change. Either signal alone disqualifies the
    violation — a successful identity mutation is fine, and a
    state-version bump from a non-mutating tool is fine."""

    def test_t5_unmotivated_action_claim_is_flagged(self) -> None:
        # Exact T5 case: "I just fixed it" with no mutation this turn.
        claims = (ActionClaim(verb="fixed", span=(0, 20)),)
        violations = verify_action_claims(
            claims,
            mutation_success_seen=False,
            state_changed=False,
        )
        assert len(violations) == 1
        assert violations[0].kind == "action_claim"
        assert violations[0].claimed_value == "fixed"

    def test_action_claim_with_mutation_seen_is_not_flagged(self) -> None:
        claims = (ActionClaim(verb="fixed", span=(0, 20)),)
        violations = verify_action_claims(
            claims,
            mutation_success_seen=True,
            state_changed=True,
        )
        assert violations == ()

    def test_action_claim_with_state_change_only_is_not_flagged(self) -> None:
        # State changed but no mutation tool returned success — defensive:
        # both signals together disqualify, either alone disqualifies.
        claims = (ActionClaim(verb="fixed", span=(0, 20)),)
        violations = verify_action_claims(
            claims,
            mutation_success_seen=False,
            state_changed=True,
        )
        assert violations == ()

    def test_dedupes_repeated_verbs(self) -> None:
        # If the model says "I just fixed X. I just fixed Y." we only
        # emit one violation per verb.
        claims = (
            ActionClaim(verb="fixed", span=(0, 20)),
            ActionClaim(verb="fixed", span=(30, 50)),
        )
        violations = verify_action_claims(
            claims,
            mutation_success_seen=False,
            state_changed=False,
        )
        assert len(violations) == 1


class TestCheckGroundingTopLevel:
    """End-to-end check on the panel-evals T4/T5 corpus."""

    def test_t4_corpus(self) -> None:
        prose = (
            "I see what you mean — the source still uses "
            "`on_validation_failure: discard`, so rejected rows would be dropped. "
            "Let me apply the fix now."
        )
        # State has rejected_records (the fix HAS already been applied,
        # the prose is wrong about it).
        state = _state_with_source(on_validation_failure="rejected_records")
        violations = check_state_claim_grounding(
            prose=prose,
            state=state,
            mutation_success_seen=True,
            state_changed=True,
        )
        assert len(violations) == 1
        assert violations[0].kind == "state_claim"
        assert violations[0].claimed_value == "discard"
        assert violations[0].actual_value == "rejected_records"

    def test_t5_corpus(self) -> None:
        prose = "I just fixed the workflow behavior. The source now routes rejected rows correctly."
        # State is unchanged from T4 — fix happened a turn earlier, not now.
        state = _state_with_source(on_validation_failure="rejected_records")
        violations = check_state_claim_grounding(
            prose=prose,
            state=state,
            mutation_success_seen=False,
            state_changed=False,
        )
        # The action-claim "I just fixed" is flagged. The prose's
        # "now routes rejected rows correctly" is generic narrative
        # and is not a state-claim shape, so it does not produce a
        # second violation.
        assert len(violations) == 1
        assert violations[0].kind == "action_claim"
        assert violations[0].claimed_value == "fixed"

    def test_combined_t4_t5_prose(self) -> None:
        # Pathological case: prose contains both shapes.
        prose = "The source still uses `on_validation_failure: discard`. I just fixed it."
        state = _state_with_source(on_validation_failure="rejected_records")
        violations = check_state_claim_grounding(
            prose=prose,
            state=state,
            mutation_success_seen=False,
            state_changed=False,
        )
        kinds = {v.kind for v in violations}
        assert kinds == {"state_claim", "action_claim"}

    def test_grounded_prose_no_violations(self) -> None:
        prose = "I configured the source with on_validation_failure: rejected_records. The pipeline preflighted clean."
        state = _state_with_source(on_validation_failure="rejected_records")
        violations = check_state_claim_grounding(
            prose=prose,
            state=state,
            mutation_success_seen=True,
            state_changed=True,
        )
        assert violations == ()

    def test_empty_prose(self) -> None:
        state = _state_with_source(on_validation_failure="discard")
        violations = check_state_claim_grounding(
            prose="",
            state=state,
            mutation_success_seen=False,
            state_changed=False,
        )
        assert violations == ()


class TestFormatCorrection:
    """The correction suffix must satisfy the augmentation prefix
    invariant when concatenated to prose."""

    def test_empty_violations_returns_empty_suffix(self) -> None:
        assert format_grounding_correction(()) == ""

    def test_compose_grounded_message_starts_with_prose(self) -> None:
        # The audit-integrity contract enforced at the service.py
        # call site is augmented.startswith(content). Every
        # compose_grounded_message output must satisfy this.
        prose = "the source still uses on_validation_failure: discard"
        state = _state_with_source(on_validation_failure="rejected_records")
        violations = check_state_claim_grounding(
            prose=prose,
            state=state,
            mutation_success_seen=True,
            state_changed=True,
        )
        message = compose_grounded_message(prose=prose, violations=violations)
        assert message.startswith(prose)
        assert "[ELSPETH-SYSTEM]" in message
        assert "rejected_records" in message

    def test_compose_with_no_violations_returns_prose_unchanged(self) -> None:
        prose = "All looks good."
        message = compose_grounded_message(prose=prose, violations=())
        assert message == prose

    def test_compose_with_empty_prose(self) -> None:
        # Augmentation prefix invariant is trivially satisfied
        # ("".startswith("") is True). Suffix-only output is permitted
        # by the existing _enforce_augmentation_prefix_invariant
        # contract.
        violations = verify_action_claims(
            (ActionClaim(verb="fixed", span=(0, 20)),),
            mutation_success_seen=False,
            state_changed=False,
        )
        message = compose_grounded_message(prose="", violations=violations)
        assert message.startswith("")
        assert "[ELSPETH-SYSTEM]" in message
