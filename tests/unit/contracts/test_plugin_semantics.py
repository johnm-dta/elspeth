"""Tests for plugin semantics contract types."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elspeth.contracts.plugin_semantics import (
    ContentKind,
    FieldSemanticFacts,
    FieldSemanticRequirement,
    InputSemanticRequirements,
    OutputSemanticDeclaration,
    SemanticEdgeContract,
    SemanticOutcome,
    SemanticValueType,
    TextFraming,
    UnknownSemanticPolicy,
    compare_semantic,
)


class TestContentKind:
    def test_is_str_subclass(self):
        assert isinstance(ContentKind.PLAIN_TEXT, str)

    def test_membership_is_closed_for_phase_1(self):
        # Phase 1 vocabulary — additions require explicit plan amendment.
        assert {member.value for member in ContentKind} == {
            "unknown",
            "plain_text",
            "markdown",
            "html_raw",
            "json_structured",
            "binary",
        }


class TestTextFraming:
    def test_membership_is_closed_for_phase_1(self):
        assert {member.value for member in TextFraming} == {
            "unknown",
            "not_text",
            "compact",
            "newline_framed",
            "line_compatible",
        }


class TestSemanticValueType:
    def test_membership_is_closed_for_phase_1(self):
        assert {member.value for member in SemanticValueType} == {
            "unknown",
            "str",
            "list",
        }


class TestUnknownSemanticPolicy:
    def test_membership_is_closed_for_phase_1(self):
        assert {member.value for member in UnknownSemanticPolicy} == {
            "allow",
            "warn",
            "fail",
        }


class TestSemanticOutcome:
    def test_membership_is_closed_for_phase_1(self):
        assert {member.value for member in SemanticOutcome} == {
            "satisfied",
            "conflict",
            "unknown",
        }


class TestFieldSemanticFacts:
    def test_construct(self):
        facts = FieldSemanticFacts(
            field_name="content",
            content_kind=ContentKind.PLAIN_TEXT,
            text_framing=TextFraming.COMPACT,
            fact_code="web_scrape.content.compact_text",
            configured_by=("format", "text_separator"),
        )
        assert facts.field_name == "content"
        assert facts.content_kind is ContentKind.PLAIN_TEXT
        assert facts.text_framing is TextFraming.COMPACT
        assert facts.value_type is SemanticValueType.UNKNOWN
        assert facts.fact_code == "web_scrape.content.compact_text"
        assert facts.configured_by == ("format", "text_separator")

    def test_immutable(self):
        facts = FieldSemanticFacts(
            field_name="x",
            content_kind=ContentKind.PLAIN_TEXT,
            fact_code="t.x.basic",
        )
        with pytest.raises(FrozenInstanceError):
            facts.field_name = "y"

    def test_default_configured_by_is_empty_tuple(self):
        facts = FieldSemanticFacts(
            field_name="x",
            content_kind=ContentKind.UNKNOWN,
            fact_code="t.x.unknown",
        )
        assert facts.configured_by == ()
        assert facts.value_type is SemanticValueType.UNKNOWN


class TestFieldSemanticRequirement:
    def test_construct_and_compare_against_satisfied_facts(self):
        requirement = FieldSemanticRequirement(
            field_name="content",
            accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT, ContentKind.MARKDOWN}),
            accepted_text_framings=frozenset({TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE}),
            requirement_code="line_explode.source_field.line_framed_text",
            unknown_policy=UnknownSemanticPolicy.FAIL,
        )
        assert requirement.field_name == "content"
        assert ContentKind.PLAIN_TEXT in requirement.accepted_content_kinds
        assert TextFraming.LINE_COMPATIBLE in requirement.accepted_text_framings
        assert requirement.severity == "high"  # default

    def test_immutable(self):
        requirement = FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT}),
            accepted_text_framings=frozenset({TextFraming.NEWLINE_FRAMED}),
            requirement_code="t.x.req",
        )
        with pytest.raises(FrozenInstanceError):
            requirement.field_name = "y"

    def test_set_inputs_are_coerced_to_frozenset(self):
        # Type annotations name frozensets, but Python does not enforce that.
        # A caller passing a mutable set must produce an immutable
        # frozenset rather than a live set reference.
        kinds = {ContentKind.PLAIN_TEXT}
        framings = {TextFraming.NEWLINE_FRAMED}
        value_types = {SemanticValueType.LIST}
        configured_by = ["source_field"]
        requirement = FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=kinds,
            accepted_text_framings=framings,
            requirement_code="t.x.req",
            accepted_value_types=value_types,
            configured_by=configured_by,
        )
        assert isinstance(requirement.accepted_content_kinds, frozenset)
        assert isinstance(requirement.accepted_text_framings, frozenset)
        assert isinstance(requirement.accepted_value_types, frozenset)
        assert isinstance(requirement.configured_by, tuple)
        # Mutating the original containers MUST NOT affect the frozen fields.
        kinds.add(ContentKind.MARKDOWN)
        framings.add(TextFraming.LINE_COMPATIBLE)
        value_types.add(SemanticValueType.STR)
        configured_by.append("extra")
        assert ContentKind.MARKDOWN not in requirement.accepted_content_kinds
        assert TextFraming.LINE_COMPATIBLE not in requirement.accepted_text_framings
        assert SemanticValueType.STR not in requirement.accepted_value_types
        assert "extra" not in requirement.configured_by

    @pytest.mark.parametrize("severity", ["critical", "High"])
    def test_rejects_out_of_vocabulary_severity(self, severity):
        with pytest.raises(ValueError, match="severity"):
            FieldSemanticRequirement(
                field_name="x",
                accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT}),
                accepted_text_framings=frozenset({TextFraming.NEWLINE_FRAMED}),
                requirement_code="t.x.req",
                severity=severity,
            )


class TestFieldSemanticFactsCoercion:
    def test_list_configured_by_coerced_to_tuple(self):
        configured_by = ["format", "text_separator"]
        facts = FieldSemanticFacts(
            field_name="x",
            content_kind=ContentKind.PLAIN_TEXT,
            text_framing=TextFraming.NEWLINE_FRAMED,
            fact_code="t.x.nl",
            configured_by=configured_by,
        )
        assert isinstance(facts.configured_by, tuple)
        configured_by.append("extra")
        assert "extra" not in facts.configured_by


class TestSemanticContractConstructionInvariants:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "field_name": "",
                "content_kind": ContentKind.PLAIN_TEXT,
                "fact_code": "t.x",
            },
            {
                "field_name": "x",
                "content_kind": "plain_text",
                "fact_code": "t.x",
            },
            {
                "field_name": "x",
                "content_kind": ContentKind.PLAIN_TEXT,
                "text_framing": "compact",
                "fact_code": "t.x",
            },
            {
                "field_name": "x",
                "content_kind": ContentKind.PLAIN_TEXT,
                "value_type": "str",
                "fact_code": "t.x",
            },
            {
                "field_name": "x",
                "content_kind": ContentKind.PLAIN_TEXT,
                "fact_code": "",
            },
            {
                "field_name": "x",
                "content_kind": ContentKind.PLAIN_TEXT,
                "fact_code": "t.x",
                "configured_by": ("format", 1),
            },
        ],
    )
    def test_field_semantic_facts_rejects_malformed_values(self, kwargs):
        with pytest.raises((TypeError, ValueError)):
            FieldSemanticFacts(**kwargs)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"field_name": ""},
            {"accepted_content_kinds": frozenset({"plain_text"})},
            {"accepted_text_framings": frozenset({"compact"})},
            {"accepted_value_types": frozenset({"list"})},
            {"requirement_code": ""},
            {"severity": ""},
            {"unknown_policy": "fail"},
            {"configured_by": ("source_field", object())},
        ],
    )
    def test_field_semantic_requirement_rejects_malformed_values(self, overrides):
        kwargs = {
            "field_name": "x",
            "accepted_content_kinds": frozenset({ContentKind.PLAIN_TEXT}),
            "accepted_text_framings": frozenset({TextFraming.NEWLINE_FRAMED}),
            "requirement_code": "t.x.req",
        }
        kwargs.update(overrides)

        with pytest.raises((TypeError, ValueError)):
            FieldSemanticRequirement(**kwargs)

    def test_output_declaration_rejects_non_fact_fields(self):
        with pytest.raises(TypeError):
            OutputSemanticDeclaration(fields=(object(),))

    def test_input_requirements_rejects_non_requirement_fields(self):
        with pytest.raises(TypeError):
            InputSemanticRequirements(fields=(object(),))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"from_id": ""},
            {"to_id": ""},
            {"consumer_plugin": ""},
            {"producer_plugin": 7},
            {"producer_field": ""},
            {"consumer_field": ""},
            {"producer_facts": object()},
            {"requirement": object()},
            {"outcome": "satisfied"},
        ],
    )
    def test_semantic_edge_contract_rejects_malformed_values(self, overrides):
        facts = FieldSemanticFacts("x", ContentKind.PLAIN_TEXT, fact_code="t.x")
        requirement = FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT}),
            accepted_text_framings=frozenset({TextFraming.NEWLINE_FRAMED}),
            requirement_code="c.x.req",
        )
        kwargs = {
            "from_id": "a",
            "to_id": "b",
            "consumer_plugin": "line_explode",
            "producer_plugin": "web_scrape",
            "producer_field": "x",
            "consumer_field": "x",
            "producer_facts": facts,
            "requirement": requirement,
            "outcome": SemanticOutcome.SATISFIED,
        }
        kwargs.update(overrides)

        with pytest.raises((TypeError, ValueError)):
            SemanticEdgeContract(**kwargs)


class TestOutputSemanticDeclaration:
    def test_default_is_empty(self):
        decl = OutputSemanticDeclaration()
        assert decl.fields == ()

    def test_carries_facts(self):
        f1 = FieldSemanticFacts("a", ContentKind.PLAIN_TEXT, fact_code="t.a")
        f2 = FieldSemanticFacts("b", ContentKind.MARKDOWN, fact_code="t.b")
        decl = OutputSemanticDeclaration(fields=(f1, f2))
        assert decl.fields == (f1, f2)

    def test_list_input_is_coerced_to_tuple(self):
        f1 = FieldSemanticFacts("a", ContentKind.PLAIN_TEXT, fact_code="t.a")
        fields = [f1]
        decl = OutputSemanticDeclaration(fields=fields)
        assert isinstance(decl.fields, tuple)
        # Mutating the source list MUST NOT affect the frozen field.
        fields.append(FieldSemanticFacts("b", ContentKind.MARKDOWN, fact_code="t.b"))
        assert len(decl.fields) == 1


class TestInputSemanticRequirements:
    def test_default_is_empty(self):
        reqs = InputSemanticRequirements()
        assert reqs.fields == ()

    def test_list_input_is_coerced_to_tuple(self):
        req = FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT}),
            accepted_text_framings=frozenset({TextFraming.NEWLINE_FRAMED}),
            requirement_code="t.x.req",
        )
        fields = [req]
        reqs = InputSemanticRequirements(fields=fields)
        assert isinstance(reqs.fields, tuple)
        # Mutating the source list MUST NOT affect the frozen field.
        fields.append(req)
        assert len(reqs.fields) == 1


class TestSemanticEdgeContract:
    def test_construct(self):
        facts = FieldSemanticFacts("x", ContentKind.PLAIN_TEXT, fact_code="t.x")
        req = FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT}),
            accepted_text_framings=frozenset({TextFraming.UNKNOWN, TextFraming.LINE_COMPATIBLE}),
            requirement_code="c.x.req",
        )
        edge = SemanticEdgeContract(
            from_id="a",
            to_id="b",
            consumer_plugin="line_explode",
            producer_plugin="web_scrape",
            producer_field="x",
            consumer_field="x",
            producer_facts=facts,
            requirement=req,
            outcome=SemanticOutcome.SATISFIED,
        )
        assert edge.outcome is SemanticOutcome.SATISFIED
        assert edge.consumer_plugin == "line_explode"
        assert edge.producer_plugin == "web_scrape"


class TestCompareSemantic:
    def _req(self, kinds, framings, policy=UnknownSemanticPolicy.FAIL):
        return FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=frozenset(kinds),
            accepted_text_framings=frozenset(framings),
            requirement_code="t.x.req",
            unknown_policy=policy,
        )

    def test_satisfied_when_facts_within_acceptance(self):
        facts = FieldSemanticFacts(
            "x",
            ContentKind.PLAIN_TEXT,
            text_framing=TextFraming.NEWLINE_FRAMED,
            fact_code="t.x.nl",
        )
        req = self._req(
            {ContentKind.PLAIN_TEXT, ContentKind.MARKDOWN},
            {TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE},
        )
        assert compare_semantic(facts, req) is SemanticOutcome.SATISFIED

    def test_conflict_on_content_kind_mismatch(self):
        facts = FieldSemanticFacts(
            "x",
            ContentKind.HTML_RAW,
            text_framing=TextFraming.NOT_TEXT,
            fact_code="t.x.raw",
        )
        req = self._req({ContentKind.PLAIN_TEXT}, {TextFraming.NEWLINE_FRAMED})
        assert compare_semantic(facts, req) is SemanticOutcome.CONFLICT

    def test_conflict_on_framing_mismatch(self):
        facts = FieldSemanticFacts(
            "x",
            ContentKind.PLAIN_TEXT,
            text_framing=TextFraming.COMPACT,
            fact_code="t.x.compact",
        )
        req = self._req(
            {ContentKind.PLAIN_TEXT},
            {TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE},
        )
        assert compare_semantic(facts, req) is SemanticOutcome.CONFLICT

    def test_unknown_when_facts_are_none(self):
        req = self._req({ContentKind.PLAIN_TEXT}, {TextFraming.NEWLINE_FRAMED})
        assert compare_semantic(None, req) is SemanticOutcome.UNKNOWN

    def test_unknown_when_either_dimension_is_unknown(self):
        facts_kind_unknown = FieldSemanticFacts(
            "x",
            ContentKind.UNKNOWN,
            text_framing=TextFraming.NEWLINE_FRAMED,
            fact_code="t.x.kindless",
        )
        facts_framing_unknown = FieldSemanticFacts(
            "x",
            ContentKind.PLAIN_TEXT,
            text_framing=TextFraming.UNKNOWN,
            fact_code="t.x.framingless",
        )
        req = self._req({ContentKind.PLAIN_TEXT}, {TextFraming.NEWLINE_FRAMED})
        assert compare_semantic(facts_kind_unknown, req) is SemanticOutcome.UNKNOWN
        assert compare_semantic(facts_framing_unknown, req) is SemanticOutcome.UNKNOWN

    def test_conflict_on_value_type_mismatch(self):
        facts = FieldSemanticFacts(
            "x",
            ContentKind.UNKNOWN,
            text_framing=TextFraming.UNKNOWN,
            value_type=SemanticValueType.STR,
            fact_code="t.x.str",
        )
        req = FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=frozenset(),
            accepted_text_framings=frozenset(),
            requirement_code="json_explode.array_field.list",
            accepted_value_types=frozenset({SemanticValueType.LIST}),
        )
        assert compare_semantic(facts, req) is SemanticOutcome.CONFLICT

    def test_unknown_when_required_value_type_is_unknown(self):
        facts = FieldSemanticFacts(
            "x",
            ContentKind.UNKNOWN,
            text_framing=TextFraming.UNKNOWN,
            value_type=SemanticValueType.UNKNOWN,
            fact_code="t.x.unknown",
        )
        req = FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=frozenset(),
            accepted_text_framings=frozenset(),
            requirement_code="json_explode.array_field.list",
            accepted_value_types=frozenset({SemanticValueType.LIST}),
        )
        assert compare_semantic(facts, req) is SemanticOutcome.UNKNOWN


_CONTENT_KINDS = list(ContentKind)
_FRAMINGS = list(TextFraming)
_KNOWN_CONTENT_KINDS = [kind for kind in ContentKind if kind is not ContentKind.UNKNOWN]
_KNOWN_FRAMINGS = [framing for framing in TextFraming if framing is not TextFraming.UNKNOWN]


@given(
    content_kind=st.sampled_from(_KNOWN_CONTENT_KINDS),
    text_framing=st.sampled_from(_KNOWN_FRAMINGS),
    extra_kinds=st.sets(st.sampled_from(_CONTENT_KINDS)),
    extra_framings=st.sets(st.sampled_from(_FRAMINGS)),
)
def test_compare_semantic_acceptance_is_monotonic(
    content_kind,
    text_framing,
    extra_kinds,
    extra_framings,
):
    facts = FieldSemanticFacts(
        field_name="x",
        content_kind=content_kind,
        text_framing=text_framing,
        fact_code="t.x.gen",
    )
    exact_requirement = FieldSemanticRequirement(
        field_name="x",
        accepted_content_kinds=frozenset({content_kind}),
        accepted_text_framings=frozenset({text_framing}),
        requirement_code="c.x.req",
    )
    expanded_requirement = FieldSemanticRequirement(
        field_name="x",
        accepted_content_kinds=frozenset({content_kind} | extra_kinds),
        accepted_text_framings=frozenset({text_framing} | extra_framings),
        requirement_code="c.x.req.expanded",
    )

    assert compare_semantic(facts, exact_requirement) is SemanticOutcome.SATISFIED
    assert compare_semantic(facts, expanded_requirement) is SemanticOutcome.SATISFIED
