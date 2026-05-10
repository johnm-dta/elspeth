"""Tests for the state-claim grounding detector.

Reproduces three panel-evals corpora:

- T4 prose: "still uses on_validation_failure: discard" while state has
  ``rejected_records`` — forward contradiction (issue
  ``elspeth-c028f7d186``, boolean_routing__p1_compliance cell).
- T5 prose: "I just fixed it" while state was unchanged from T4 —
  backward contradiction (same issue, same cell).
- Cells #2/#4 prose: "you're right, I'll change that" /
  "I fixed the workflow behavior so source validation is no longer
  silently dropping rows" while no mutation tool was called this turn
  (issue ``elspeth-905fe2a3d8``).

Plus edge cases that protect against false positives (matching claims,
missing source, paraphrases that legitimately discuss state, multi-
output configurations, bare past tense without consequence clauses,
present-perfect with negation/progressive, references to earlier
turns).
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


class TestExtractActionClaimsAgreementPromise:
    """Agreement-opener + first-person commitment detection.

    Closes the panel-smoke cells #2/#4 case (issue
    ``elspeth-905fe2a3d8``) — the model agrees verbally without calling
    a mutation tool. Verbs surface in past tense even though the
    matched prose is in base/future form, so ``verify_action_claims``
    can dedupe across patterns on the same verb identity."""

    def test_extract_youre_right_ill_change(self) -> None:
        prose = "You're right, I'll change that to rejected_records."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "changed" in verbs

    def test_extract_yes_i_can_fix(self) -> None:
        prose = "Yes, I can fix that for you."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "fixed" in verbs

    def test_extract_good_catch_ill_switch(self) -> None:
        prose = "Good catch — I'll switch the routing."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "switched" in verbs

    def test_extract_absolutely_ill_update(self) -> None:
        prose = "Absolutely, I will update the source."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "updated" in verbs

    def test_extract_as_you_asked_i_should_remove(self) -> None:
        prose = "As you asked, I should remove that node."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "removed" in verbs

    def test_extract_yes_i_can_confirm(self) -> None:
        # Issue elspeth-905fe2a3d8 lists "I confirmed" as one of three
        # example phrases the detector must catch. The base-form
        # variant in agreement context exercises both the verb-list
        # entry and the base-to-past normalisation.
        prose = "Yes, I can confirm that change."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "confirmed" in verbs

    def test_no_extract_for_agreement_without_action_verb(self) -> None:
        # "Yes, I see what you mean" — agreement opener but no action
        # verb in the closed list. No match.
        prose = "Yes, I see what you mean."
        claims = extract_action_claims(prose)
        assert claims == ()

    def test_no_extract_for_action_verb_without_agreement_opener(self) -> None:
        # "I'll change that" without a leading agreement opener does
        # not match this pattern (other patterns may catch it but the
        # agreement-promise pattern alone is conservative).
        prose = "I'll change that."
        claims = extract_action_claims(prose)
        # The bare-past pattern requires consequence; present-perfect
        # requires "have/'ve"; just-pattern requires "just". None
        # match either.
        assert claims == ()


class TestExtractActionClaimsPresentPerfect:
    """Present-perfect detection ("I've fixed", "I have changed").

    Less ambiguous than bare simple past — present perfect English
    usage strongly implies "before now, possibly just now". Negation
    and progressive forms are excluded by negative lookahead."""

    def test_extract_ive_fixed(self) -> None:
        prose = "I've fixed the schema for you."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "fixed" in verbs

    def test_extract_i_have_updated(self) -> None:
        prose = "I have updated the source."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "updated" in verbs

    def test_extract_ive_configured(self) -> None:
        prose = "I've configured the output to write JSON lines."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "configured" in verbs

    def test_extract_i_confirmed_present_perfect(self) -> None:
        # Issue elspeth-905fe2a3d8 lists "I confirmed" as one of three
        # example phrases the detector must catch. Also covers the
        # bare past form via the bare-past + consequence pattern when
        # paired with a consequence clause (separate test below); this
        # one pins the present-perfect variant.
        prose = "I have confirmed the change to rejected_records."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "confirmed" in verbs

    def test_no_extract_for_negated_present_perfect(self) -> None:
        prose = "I have not changed the source configuration."
        claims = extract_action_claims(prose)
        # The just-pattern requires "just" (no), the agreement-promise
        # requires an opener (no), the bare-past requires consequence
        # (no), and the present-perfect explicitly excludes "not".
        assert claims == ()

    def test_no_extract_for_havent_contraction(self) -> None:
        prose = "I haven't changed the source yet."
        claims = extract_action_claims(prose)
        # Note: ``I haven't`` parses as ``I + 've + n't`` under typical
        # tokenisation. The negative lookahead rejects ``n't`` after
        # the present-perfect anchor. Confirm no match.
        assert claims == ()

    def test_no_extract_for_progressive_present_perfect(self) -> None:
        prose = "I have been adjusting the schema across turns."
        claims = extract_action_claims(prose)
        # Present-perfect progressive ("have been <verb-ing>") refers
        # to ongoing or intermittent action, not a fresh completion.
        # Negative lookahead rejects "been".
        assert claims == ()


class TestExtractActionClaimsBarePastWithConsequence:
    """Bare simple past followed by a present-tense consequence clause.

    Closes the panel-smoke T5 case (issue ``elspeth-c028f7d186``,
    backward-contradiction shape with bare "I fixed" instead of "I
    just fixed"). The consequence clause is the disambiguator — a
    reference to an earlier-turn action would not narrate a fresh
    present-tense consequence."""

    def test_extract_t5_corpus_so_clause(self) -> None:
        # Exact T5 prose from boolean_routing__p1_compliance cell.
        prose = "I fixed the workflow behavior so source validation is no longer silently dropping rows from the record set."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "fixed" in verbs

    def test_extract_and_now_clause(self) -> None:
        prose = "I changed the routing and now everything goes to rejected_records."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        assert "changed" in verbs

    def test_extract_is_now_consequence(self) -> None:
        prose = "I configured the source. I updated the output and the pipeline is now writing CSV."
        claims = extract_action_claims(prose)
        verbs = {c.verb for c in claims}
        # "I updated the output and ... is now writing" matches the
        # bare-past + consequence pattern. "I configured the source."
        # alone has no consequence clause and so does not match.
        assert "updated" in verbs

    def test_no_extract_for_bare_past_without_consequence(self) -> None:
        # Bare "I fixed it" without a consequence clause remains
        # ambiguous in multi-turn dialogue (could refer to an earlier
        # turn). Conservative detector does not flag.
        prose = "I fixed it."
        claims = extract_action_claims(prose)
        assert claims == ()

    def test_no_extract_for_earlier_turn_reference(self) -> None:
        # The consequence clause requires present-tense narration that
        # a prior-turn reference would not produce.
        prose = "As I mentioned earlier, I changed that already."
        claims = extract_action_claims(prose)
        assert claims == ()

    def test_no_extract_for_consequence_without_determiner(self) -> None:
        # Pattern requires a determiner (the/that/this/it/your) after
        # the verb to anchor the object. Bare "I fixed everything so
        # things work" lacks a determiner.
        prose = "I fixed everything so things work."
        claims = extract_action_claims(prose)
        assert claims == ()

    def test_no_extract_for_consequence_across_sentence_boundary(self) -> None:
        # The consequence-clause search uses [^.!?] so it does not
        # cross sentence boundaries. A bare past in one sentence and
        # a "now" claim in the next does not match.
        prose = "I fixed the source. The pipeline is now valid."
        claims = extract_action_claims(prose)
        assert claims == ()


class TestExtractActionClaimsCrossPattern:
    """Pattern overlap and verb de-duplication across the four
    detector categories."""

    def test_just_and_present_perfect_dedupe_on_verb(self) -> None:
        # "I've fixed" matches present-perfect; "I just fixed" matches
        # the just-pattern. Both surface verb "fixed". The verifier
        # dedupes — at most one violation per verb per turn.
        prose = "I've fixed the source. I just fixed the output."
        claims = extract_action_claims(prose)
        violations = verify_action_claims(
            claims,
            mutation_success_seen=False,
            state_changed=False,
        )
        assert len(violations) == 1
        assert violations[0].claimed_value == "fixed"

    def test_agreement_promise_normalises_to_past_tense(self) -> None:
        # Base-form verbs from agreement-promise pattern surface as
        # past-tense in ActionClaim.verb so dedup across patterns
        # works on a single verb identity.
        prose = "Yes, I'll change that."
        claims = extract_action_claims(prose)
        assert any(c.verb == "changed" for c in claims)
        assert all(c.verb != "change" for c in claims)

    def test_distinct_verbs_across_patterns_all_surface(self) -> None:
        prose = "Yes, I'll change the routing. I've fixed the output. I configured the source so it now uses rejected_records."
        claims = extract_action_claims(prose)
        violations = verify_action_claims(
            claims,
            mutation_success_seen=False,
            state_changed=False,
        )
        verbs = {v.claimed_value for v in violations}
        assert verbs == {"changed", "fixed", "configured"}


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

    def test_panel_smoke_cell_2_corpus(self) -> None:
        # Issue elspeth-905fe2a3d8 — model agrees verbally without
        # calling a mutation tool. State unchanged, no mutation seen.
        prose = "You're right, I'll change that to rejected_records."
        state = _state_with_source(on_validation_failure="discard")
        violations = check_state_claim_grounding(
            prose=prose,
            state=state,
            mutation_success_seen=False,
            state_changed=False,
        )
        assert any(v.kind == "action_claim" and v.claimed_value == "changed" for v in violations)

    def test_issue_905fe2a3d8_third_example_phrase(self) -> None:
        # Issue elspeth-905fe2a3d8 names three example phrases the
        # detector must catch: "you're right", "I confirmed",
        # "yes I can change that". The first is exercised by
        # ``test_panel_smoke_cell_2_corpus`` above; the third by
        # ``test_extract_yes_i_can_fix`` (same shape with verb=fix).
        # This pins the literal "I confirmed" example from the issue
        # text — present-perfect path, state unchanged, no mutation.
        prose = "I have confirmed the change to rejected_records."
        state = _state_with_source(on_validation_failure="rejected_records")
        violations = check_state_claim_grounding(
            prose=prose,
            state=state,
            mutation_success_seen=False,
            state_changed=False,
        )
        assert any(v.kind == "action_claim" and v.claimed_value == "confirmed" for v in violations)

    def test_panel_smoke_t5_corpus_widened(self) -> None:
        # Issue elspeth-c028f7d186 — exact T5 prose without "just".
        # Caught by the bare-past + consequence pattern.
        prose = "I fixed the workflow behavior so source validation is no longer silently dropping rows from the record set."
        state = _state_with_source(on_validation_failure="rejected_records")
        violations = check_state_claim_grounding(
            prose=prose,
            state=state,
            mutation_success_seen=False,
            state_changed=False,
        )
        assert len(violations) == 1
        assert violations[0].kind == "action_claim"
        assert violations[0].claimed_value == "fixed"

    def test_present_perfect_completion_with_mutation_is_not_flagged(self) -> None:
        # The verifier gate protects truthful turns: "I've fixed it"
        # paired with a successful mutation produces no violation
        # regardless of which pattern matched.
        prose = "I've fixed the source so it now uses rejected_records."
        state = _state_with_source(on_validation_failure="rejected_records")
        violations = check_state_claim_grounding(
            prose=prose,
            state=state,
            mutation_success_seen=True,
            state_changed=True,
        )
        # State claim "rejected_records" matches actual; action claim
        # is gated out by mutation_success_seen.
        assert violations == ()

    def test_agreement_acknowledgement_with_mutation_is_not_flagged(self) -> None:
        # Same gate protection for the agreement-promise pattern:
        # "Yes, I'll change that" paired with a successful mutation
        # this turn does not produce a violation.
        prose = "Yes, I'll change that to rejected_records."
        state = _state_with_source(on_validation_failure="rejected_records")
        violations = check_state_claim_grounding(
            prose=prose,
            state=state,
            mutation_success_seen=True,
            state_changed=True,
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
