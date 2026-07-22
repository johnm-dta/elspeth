# tests/unit/web/composer/test_validation_error_codes.py
"""Every candidate-path rejection carries a closed error_code + structural facts.

Guided A/B session 5113b7ac (attempts 6/10/12, 2026-07-22) died
REPAIR_EXHAUSTED with ``rejection_codes=[]``: the planner's redacted repair
feedback (``_allowlisted_candidate_feedback``) strips raw validation messages
and keys its enrichment on the closed ``error_code`` — so a ``ValidationEntry``
emitted without one forwards NOTHING actionable. A rejection with no code and
no message is unrepairable by construction.

These tests pin the closure:

- the schema-contract family carries ``schema_contract_violation`` /
  ``sink_contract_violation`` / ``locked_input_extras`` / ``sink_locked_extras``
  plus a structured ``contract`` detail naming producer, consumer, and the
  missing/extra FIELD NAMES (pipeline identifiers and schema field names from
  validated config — never user row content, hence redaction-safe);
- representative structural rejections each carry their closed code;
- the closed-code catalogue stays containment-free (no code is a substring of
  another), which the explain tool's fuzzy route and the regex alternations
  in ``_VALIDATION_ERROR_PATTERNS`` both depend on;
- the planner feedback projection forwards code + static guidance + contract
  facts, and the per-attempt trail can never report an empty code list for a
  rejection that carried entries.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationEntry,
    ValidationSummary,
)
from elspeth.web.composer.tools.generation import (
    _CLOSED_VALIDATION_ERROR_CODES,
    explain_validation_code,
)


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


def _make_source(
    on_success: str = "t1",
    options: dict[str, Any] | None = None,
) -> SourceSpec:
    return SourceSpec(
        plugin="csv",
        on_success=on_success,
        options={"path": "/data/input.csv", **(options or {})},
        on_validation_failure="quarantine",
    )


def _make_transform(
    id: str,
    input: str,
    on_success: str,
    options: dict[str, Any] | None = None,
) -> NodeSpec:
    return NodeSpec(
        id=id,
        node_type="transform",
        plugin="value_transform",
        input=input,
        on_success=on_success,
        on_error="discard",
        options={
            "schema": {"mode": "observed"},
            "operations": [{"target": "_placeholder", "expression": "row['text']"}],
            **(options or {}),
        },
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def _make_output(name: str = "main", options: dict[str, Any] | None = None) -> OutputSpec:
    return OutputSpec(
        name=name,
        plugin="csv",
        options={"path": f"outputs/{name}.csv", "schema": {"mode": "observed"}, **(options or {})},
        on_write_failure="discard",
    )


def _make_edge(id: str, from_node: str, to_node: str) -> EdgeSpec:
    return EdgeSpec(id=id, from_node=from_node, to_node=to_node, edge_type="on_success", label=None)


def _entries_with_code(result: ValidationSummary, code: str) -> list[ValidationEntry]:
    return [e for e in result.errors if e.error_code == code]


class TestSchemaContractFamilyCodes:
    """The five schema-contract emitters carry codes + structured facts."""

    def test_node_contract_violation_carries_code_and_facts(self) -> None:
        state = _empty_state()
        state = state.with_source(_make_source(options={"schema": {"mode": "observed"}}))
        state = state.with_node(_make_transform("t1", "t1", "main", options={"required_input_fields": ["text"]}))
        state = state.with_output(_make_output())
        state = state.with_edge(_make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        entries = _entries_with_code(result, "schema_contract_violation")
        assert entries, [e.to_dict() for e in result.errors]
        detail = entries[0].contract
        assert detail is not None
        assert detail.consumer == "t1"
        assert detail.producer  # source producer id
        assert detail.missing_fields == ("text",)

    def test_sink_contract_violation_carries_code_and_facts(self) -> None:
        state = _empty_state()
        state = state.with_source(_make_source(on_success="main", options={"schema": {"mode": "fixed", "fields": ["other: str"]}}))
        state = state.with_output(_make_output(options={"schema": {"mode": "observed", "required_fields": ["text"]}}))
        result = state.validate()
        assert not result.is_valid
        entries = _entries_with_code(result, "sink_contract_violation")
        assert entries, [e.to_dict() for e in result.errors]
        detail = entries[0].contract
        assert detail is not None
        assert detail.consumer == "output:main"
        assert detail.missing_fields == ("text",)

    def test_locked_input_extras_carries_code_and_facts(self) -> None:
        state = _empty_state()
        state = state.with_source(_make_source(options={"schema": {"mode": "fixed", "fields": ["text: str", "extra: str"]}}))
        state = state.with_node(
            _make_transform(
                "t1",
                "t1",
                "main",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_output(_make_output())
        result = state.validate()
        assert not result.is_valid
        entries = _entries_with_code(result, "locked_input_extras")
        assert entries, [e.to_dict() for e in result.errors]
        detail = entries[0].contract
        assert detail is not None
        assert detail.consumer == "t1"
        assert "extra" in detail.extra_fields

    def test_sink_locked_extras_carries_code_and_facts(self) -> None:
        state = _empty_state()
        state = state.with_source(
            _make_source(on_success="main", options={"schema": {"mode": "fixed", "fields": ["text: str", "extra: str"]}})
        )
        state = state.with_output(_make_output(options={"schema": {"mode": "fixed", "fields": ["text: str"]}}))
        result = state.validate()
        assert not result.is_valid
        entries = _entries_with_code(result, "sink_locked_extras")
        assert entries, [e.to_dict() for e in result.errors]
        detail = entries[0].contract
        assert detail is not None
        assert detail.consumer == "output:main"
        assert "extra" in detail.extra_fields

    def test_contract_config_parse_failure_carries_code(self) -> None:
        state = _empty_state()
        state = state.with_source(_make_source(options={"schema": {"mode": "observed"}}))
        state = state.with_node(_make_transform("t1", "t1", "main", options={"required_input_fields": "text"}))
        state = state.with_output(_make_output())
        result = state.validate()
        assert not result.is_valid
        assert _entries_with_code(result, "contract_config_invalid"), [e.to_dict() for e in result.errors]


class TestStructuralRejectionCodes:
    """Representative structural rejections each carry their closed code."""

    def test_empty_state_names_missing_source_and_sinks(self) -> None:
        result = _empty_state().validate()
        assert not result.is_valid
        assert _entries_with_code(result, "no_source_configured")
        assert _entries_with_code(result, "no_sinks_configured")

    def test_unreachable_input_carries_code(self) -> None:
        state = _empty_state()
        state = state.with_source(_make_source(on_success="rows"))
        state = state.with_node(_make_transform("t1", "nowhere", "main"))
        state = state.with_output(_make_output())
        result = state.validate()
        assert not result.is_valid
        assert _entries_with_code(result, "node_input_not_reachable"), [e.to_dict() for e in result.errors]

    def test_duplicate_connection_producer_carries_code(self) -> None:
        state = _empty_state()
        state = state.with_source(_make_source(on_success="shared"))
        state = state.with_node(_make_transform("t1", "shared", "shared"))
        state = state.with_node(_make_transform("t2", "shared", "main"))
        state = state.with_output(_make_output())
        result = state.validate()
        assert not result.is_valid
        assert _entries_with_code(result, "duplicate_connection_producer"), [e.to_dict() for e in result.errors]

    def test_aggregation_missing_on_error_carries_code(self) -> None:
        state = _empty_state()
        state = state.with_source(_make_source(on_success="agg_in"))
        state = state.with_node(
            NodeSpec(
                id="agg",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg_in",
                on_success="main",
                on_error=None,
                options={"schema": {"mode": "observed"}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                trigger={"count": 10},
            )
        )
        state = state.with_output(_make_output())
        result = state.validate()
        assert not result.is_valid
        assert _entries_with_code(result, "aggregation_missing_on_error"), [e.to_dict() for e in result.errors]

    def test_battery_states_emit_no_codeless_errors(self) -> None:
        """Sweep: every error the battery produces carries a closed code.

        This is the campaign's boundary rule stated as an invariant: the
        planner feedback can only enrich what carries a code, so a codeless
        error is a blind (unrepairable) rejection by construction.
        """
        battery: list[CompositionState] = []

        battery.append(_empty_state())

        s = _empty_state()
        s = s.with_source(_make_source(options={"schema": {"mode": "observed"}}))
        s = s.with_node(_make_transform("t1", "t1", "main", options={"required_input_fields": ["text"]}))
        s = s.with_output(_make_output())
        battery.append(s)

        s = _empty_state()
        s = s.with_source(_make_source(on_success="rows"))
        s = s.with_node(_make_transform("t1", "nowhere", "main"))
        s = s.with_output(_make_output())
        battery.append(s)

        s = _empty_state()
        s = s.with_source(_make_source(on_success="shared"))
        s = s.with_node(_make_transform("t1", "shared", "shared"))
        s = s.with_node(_make_transform("t2", "shared", "main"))
        s = s.with_output(_make_output())
        battery.append(s)

        s = _empty_state()
        s = s.with_source(_make_source(on_success="main", options={"schema": {"mode": "fixed", "fields": ["other: str"]}}))
        s = s.with_output(_make_output(options={"schema": {"mode": "observed", "required_fields": ["text"]}}))
        battery.append(s)

        for state in battery:
            result = state.validate()
            assert not result.is_valid
            codeless = [e.to_dict() for e in result.errors if not e.error_code]
            assert not codeless, f"codeless rejection(s) escaped the closed-code sweep: {codeless}"


class TestClosedCodeCatalogueInvariants:
    def test_schema_contract_codes_are_registered_and_explainable(self) -> None:
        for code in (
            "schema_contract_violation",
            "sink_contract_violation",
            "locked_input_extras",
            "sink_locked_extras",
            "contract_config_invalid",
            "node_input_not_reachable",
            "duplicate_connection_producer",
            "duplicate_connection_consumer",
            "no_source_configured",
            "no_sinks_configured",
            "aggregation_missing_on_error",
            "coalesce_branch_unreachable",
        ):
            assert code in _CLOSED_VALIDATION_ERROR_CODES, code
            guidance = explain_validation_code(code)
            assert guidance is not None, f"{code} does not resolve to catalogue guidance"
            explanation, fix = guidance
            assert explanation and fix

    def test_codes_are_containment_free(self) -> None:
        """No closed code may be a substring of another.

        The explain tool's fuzzy route scans codes by substring containment
        and the catalogue patterns embed codes as regex alternations — a
        contained code would mis-resolve to whichever entry scans first.
        """
        codes = _CLOSED_VALIDATION_ERROR_CODES
        offenders = [(a, b) for a in codes for b in codes if a != b and a in b]
        assert not offenders, offenders


class TestPlannerFeedbackCarriesStructuralFacts:
    def test_allowlisted_feedback_projects_contract_facts(self) -> None:
        from elspeth.web.composer.pipeline_planner import _allowlisted_candidate_feedback
        from elspeth.web.composer.state import SchemaContractDetail
        from elspeth.web.composer.tools import ToolResult

        entry = ValidationEntry(
            component="node:llm_tone",
            message="Schema contract violation: 'fork_ab' -> 'llm_tone'. (raw message must NOT be forwarded)",
            severity="high",
            error_code="schema_contract_violation",
            contract=SchemaContractDetail(
                producer="fork_ab",
                consumer="llm_tone",
                missing_fields=("color_name", "hex"),
            ),
        )
        result = ToolResult(
            success=False,
            updated_state=_empty_state(),
            validation=ValidationSummary(is_valid=False, errors=(entry,), warnings=(), suggestions=()),
            affected_nodes=(),
        )
        feedback = _allowlisted_candidate_feedback(result)
        projected = feedback["validation"]["errors"][0]
        assert projected["error_code"] == "schema_contract_violation"
        assert "message" not in projected
        assert projected["explanation"]
        assert projected["suggested_fix"]
        assert projected["contract"] == {
            "producer": "fork_ab",
            "consumer": "llm_tone",
            "missing_fields": ["color_name", "hex"],
        }

    def test_allowlisted_feedback_codeless_entry_still_names_a_code(self) -> None:
        from elspeth.web.composer.pipeline_planner import _allowlisted_candidate_feedback
        from elspeth.web.composer.tools import ToolResult

        entry = ValidationEntry(component="node:x", message="anything", severity="high")
        result = ToolResult(
            success=False,
            updated_state=_empty_state(),
            validation=ValidationSummary(is_valid=False, errors=(entry,), warnings=(), suggestions=()),
            affected_nodes=(),
        )
        feedback = _allowlisted_candidate_feedback(result)
        assert feedback["validation"]["errors"][0]["error_code"] == "validation_error"

    def test_review_contract_guidance_quotes_the_recognition_constants_verbatim(self) -> None:
        """The repair guidance must BE the literal minimal delta.

        Tutorial op 18b4cee7 (session c98e8561, 2026-07-22, post-356d839a8):
        four generations including the opus hatch each drew the single code
        ``interpretation_review_contract_unsatisfied`` WITH guidance live.
        The contract recognizes the cleanup row only when user_term equals
        RAW_HTML_CLEANUP_USER_TERM AND the draft's lowercase contains every
        _RAW_HTML_CLEANUP_DRAFT_MARKERS substring — guidance inviting a
        free-text draft steers the planner into an unrecognized-row loop
        where the identical code fires forever. The suggested_fix must quote
        the registered user_term and the canonical draft constant verbatim
        so a copy-paste repair is guaranteed to be recognized.
        """
        from elspeth.web.composer.tools.generation import explain_validation_code
        from elspeth.web.interpretation_state import (
            RAW_HTML_CLEANUP_REVIEW_DRAFT,
            RAW_HTML_CLEANUP_USER_TERM,
        )

        guidance = explain_validation_code("interpretation_review_contract_unsatisfied")
        assert guidance is not None
        _explanation, suggested_fix = guidance
        assert RAW_HTML_CLEANUP_USER_TERM in suggested_fix
        assert RAW_HTML_CLEANUP_REVIEW_DRAFT in suggested_fix

    def test_rejected_mutation_gates_stale_state_errors_out_of_feedback_and_trail(self) -> None:
        """A pre-application rejection must not carry the unchanged state's errors.

        Tutorial session 38e3e7f8 (op 1152d7e3, 2026-07-22): every semantic
        set_pipeline rejection on the empty-seed surface reached the planner
        as ``['no_sinks_configured', 'no_source_configured', 'validation_error']``
        — the real reason reduced to a bare placeholder and two red herrings
        describing a state the planner was not editing (it authors a full
        replacement pipeline). The planner "converged" by dropping every node.
        When a ``rejected_mutation`` entry is present, feedback and trail must
        carry ONLY the rejection entries.
        """
        from elspeth.web.composer.pipeline_planner import (
            _allowlisted_candidate_feedback,
            _candidate_rejection_codes,
        )
        from elspeth.web.composer.tools._common import _failure_result

        result = _failure_result(
            _empty_state(),
            "File sink 'json' must set mode explicitly. Use 'write' or 'append'.",
        )
        # The empty state contributes no_source_configured/no_sinks_configured
        # to validation.errors — stale-state noise for a full-replacement tool.
        assert {entry.error_code for entry in result.validation.errors} >= {"no_source_configured", "no_sinks_configured"}

        feedback = _allowlisted_candidate_feedback(result)
        assert [entry["component"] for entry in feedback["validation"]["errors"]] == ["rejected_mutation"]
        assert _candidate_rejection_codes(result) == ("validation_error",)

    def test_validated_candidate_rejections_pass_through_ungated(self) -> None:
        """Without a rejected_mutation entry, every real error must survive.

        Guards the instance-1 class (built candidate validated, real errors,
        e.g. coalesce_branch_unreachable) against over-gating.
        """
        from elspeth.web.composer.pipeline_planner import (
            _allowlisted_candidate_feedback,
            _candidate_rejection_codes,
        )
        from elspeth.web.composer.tools import ToolResult

        entries = (
            ValidationEntry(component="node:merge", message="m", severity="high", error_code="coalesce_branch_unreachable"),
            ValidationEntry(component="node:copy", message="m", severity="high", error_code="node_input_not_reachable"),
        )
        result = ToolResult(
            success=False,
            updated_state=_empty_state(),
            validation=ValidationSummary(is_valid=False, errors=entries, warnings=(), suggestions=()),
            affected_nodes=(),
        )
        feedback = _allowlisted_candidate_feedback(result)
        assert [entry["error_code"] for entry in feedback["validation"]["errors"]] == [
            "coalesce_branch_unreachable",
            "node_input_not_reachable",
        ]
        assert _candidate_rejection_codes(result) == ("coalesce_branch_unreachable", "node_input_not_reachable")

    def test_rejection_trail_codes_never_empty_when_entries_exist(self) -> None:
        """The per-attempt trail must name every rejection, coded or not.

        A codeless entry surfaces as the 'validation_error' placeholder in
        rejection_codes rather than silently vanishing — REPAIR_EXHAUSTED
        with rejection_codes=[] while entries existed is the exact blindness
        session 5113b7ac exposed.
        """
        from elspeth.web.composer.pipeline_planner import _candidate_rejection_codes
        from elspeth.web.composer.tools import ToolResult

        entries = (
            ValidationEntry(component="node:a", message="m", severity="high"),
            ValidationEntry(component="node:b", message="m", severity="high", error_code="schema_contract_violation"),
        )
        result = ToolResult(
            success=False,
            updated_state=_empty_state(),
            validation=ValidationSummary(is_valid=False, errors=entries, warnings=(), suggestions=()),
            affected_nodes=(),
        )
        codes = _candidate_rejection_codes(result)
        assert codes == ("validation_error", "schema_contract_violation")


def _make_gate(id: str, input: str, fork_to: tuple[str, ...]) -> NodeSpec:
    return NodeSpec(
        id=id,
        node_type="gate",
        plugin=None,
        input=input,
        on_success=None,
        on_error=None,
        options={},
        condition="'all'",
        routes={"all": "fork"},
        fork_to=fork_to,
        branches=None,
        policy=None,
        merge=None,
    )


def _make_coalesce(id: str, branches: Any) -> NodeSpec:
    return NodeSpec(
        id=id,
        node_type="coalesce",
        plugin=None,
        input="branches",
        on_success=None,
        on_error=None,
        options={},
        condition=None,
        routes=None,
        fork_to=None,
        branches=branches,
        policy="require_all",
        merge="union",
    )


def _orphaned_coalesce_state(branches: Any) -> CompositionState:
    """Fork/coalesce pipeline whose branch transforms bypass the coalesce.

    Reconstruction of guided session 277fb6c4 (attempts 3/6/9/10, 2026-07-22):
    the per-branch transforms publish straight to the sink — legal in
    isolation, so no companion code fires — leaving the coalesce's branches
    values naming connections nothing produces. The ONLY rejection is
    ``coalesce_branch_unreachable``, exactly matching the observed
    single-code per-attempt trail.
    """
    state = _empty_state()
    state = state.with_source(_make_source(on_success="rows"))
    state = state.with_node(_make_gate("fan_out", "rows", ("branch_a", "branch_b")))
    state = state.with_node(_make_transform("t_a", "branch_a", "main"))
    state = state.with_node(_make_transform("t_b", "branch_b", "main"))
    state = state.with_node(_make_coalesce("merge", branches))
    state = state.with_node(_make_transform("tidy", "merge", "main"))
    state = state.with_output(_make_output())
    return state


class TestCoalesceReachabilityFacts:
    """The coalesce reachability rejection carries instance wiring facts.

    Guided session 277fb6c4 died REPAIR_EXHAUSTED on four identical
    ``coalesce_branch_unreachable`` rejections: the static guidance directs
    the repair at the coalesce node, but the observed miswiring lives in the
    branch transforms' ``on_success`` — a repair the planner cannot find
    from a bare code. These facts name each unreachable branches value and
    the connections the pipeline actually produces (node ids and connection
    names the planner itself authored — never user row content).
    """

    def test_orphaned_coalesce_rejects_with_the_single_observed_code(self) -> None:
        state = _orphaned_coalesce_state({"branch_a": "a_done", "branch_b": "b_done"})
        result = state.validate()
        assert not result.is_valid
        assert [e.error_code for e in result.errors] == ["coalesce_branch_unreachable"]

    def test_reachability_facts_name_unreachable_pairs_and_produced_connections(self) -> None:
        from elspeth.web.composer.state import coalesce_reachability_facts

        state = _orphaned_coalesce_state({"branch_a": "a_done", "branch_b": "b_done"})
        facts = coalesce_reachability_facts(state)
        assert facts == {
            "merge": {
                "unreachable_branches": {"branch_a": "a_done", "branch_b": "b_done"},
                # Sink names and the coalesce's own published id are excluded:
                # both pass the membership walk today but are not connections a
                # branch value should be steered toward.
                "produced_connections": ["branch_a", "branch_b", "rows"],
                # The lure, named: each unreachable branch whose branch-side
                # transform publishes to a SINK instead of the expected
                # connection (guided attempt 14, session 04200b45 — the model
                # wired branch transforms to the reviewed sink 3x with the
                # bare facts live).
                "sink_targeting_branches": [
                    {"node_id": "t_a", "on_success_sink": "main", "expected_connection": "a_done"},
                    {"node_id": "t_b", "on_success_sink": "main", "expected_connection": "b_done"},
                ],
            }
        }

    def test_reachability_facts_handle_list_form_branches(self) -> None:
        from elspeth.web.composer.state import coalesce_reachability_facts

        state = _orphaned_coalesce_state(("a_done", "b_done"))
        facts = coalesce_reachability_facts(state)
        # List-form branch keys are the arriving connection names themselves —
        # nothing consumes them as an input, so no branch-side transform chain
        # exists to attribute a sink lure to.
        assert facts == {
            "merge": {
                "unreachable_branches": {"a_done": "a_done", "b_done": "b_done"},
                "produced_connections": ["branch_a", "branch_b", "rows"],
            }
        }

    def test_sink_lure_attribution_follows_transform_chains_and_skips_non_sink_dangles(self) -> None:
        """The lure walk follows a branch's transform CHAIN to the sink hop.

        branch_a: t_a -> x_mid -> t_mid -> main (sink): the transform to
        repair is t_mid, the chain's sink-publishing hop. branch_b: t_b
        publishes a dangling non-sink name — unreachable, but not the sink
        lure, so no attribution entry.
        """
        from elspeth.web.composer.state import coalesce_reachability_facts

        state = _empty_state()
        state = state.with_source(_make_source(on_success="rows"))
        state = state.with_node(_make_gate("fan_out", "rows", ("branch_a", "branch_b")))
        state = state.with_node(_make_transform("t_a", "branch_a", "x_mid"))
        state = state.with_node(_make_transform("t_mid", "x_mid", "main"))
        state = state.with_node(_make_transform("t_b", "branch_b", "b_dangle"))
        state = state.with_node(_make_coalesce("merge", {"branch_a": "a_done", "branch_b": "b_done"}))
        state = state.with_node(_make_transform("tidy", "merge", "main"))
        state = state.with_output(_make_output())
        facts = coalesce_reachability_facts(state)
        assert facts["merge"]["sink_targeting_branches"] == [
            {"node_id": "t_mid", "on_success_sink": "main", "expected_connection": "a_done"},
        ]

    def test_reachability_facts_empty_for_correctly_wired_coalesce(self) -> None:
        from elspeth.web.composer.state import coalesce_reachability_facts

        state = _empty_state()
        state = state.with_source(_make_source(on_success="rows"))
        state = state.with_node(_make_gate("fan_out", "rows", ("branch_a", "branch_b")))
        state = state.with_node(_make_transform("t_a", "branch_a", "a_done"))
        state = state.with_node(_make_transform("t_b", "branch_b", "b_done"))
        state = state.with_node(_make_coalesce("merge", {"branch_a": "a_done", "branch_b": "b_done"}))
        state = state.with_node(_make_transform("tidy", "merge", "main"))
        state = state.with_output(_make_output())
        assert state.validate().is_valid
        assert coalesce_reachability_facts(state) == {}

    def test_allowlisted_feedback_projects_connectivity_facts(self) -> None:
        from elspeth.web.composer.pipeline_planner import _allowlisted_candidate_feedback
        from elspeth.web.composer.tools import ToolResult

        state = _orphaned_coalesce_state({"branch_a": "a_done", "branch_b": "b_done"})
        result = ToolResult(
            success=False,
            updated_state=state,
            validation=state.validate(),
            affected_nodes=(),
        )
        feedback = _allowlisted_candidate_feedback(result)
        projected = feedback["validation"]["errors"][0]
        assert projected["error_code"] == "coalesce_branch_unreachable"
        assert "message" not in projected
        assert projected["connectivity"] == {
            "unreachable_branches": {"branch_a": "a_done", "branch_b": "b_done"},
            "produced_connections": ["branch_a", "branch_b", "rows"],
            "sink_targeting_branches": [
                {"node_id": "t_a", "on_success_sink": "main", "expected_connection": "a_done"},
                {"node_id": "t_b", "on_success_sink": "main", "expected_connection": "b_done"},
            ],
        }

    def test_allowlisted_feedback_omits_connectivity_for_other_codes(self) -> None:
        from elspeth.web.composer.pipeline_planner import _allowlisted_candidate_feedback
        from elspeth.web.composer.tools import ToolResult

        entry = ValidationEntry(
            component="node:merge",
            message="anything",
            severity="high",
            error_code="coalesce_missing_policy",
        )
        result = ToolResult(
            success=False,
            updated_state=_empty_state(),
            validation=ValidationSummary(is_valid=False, errors=(entry,), warnings=(), suggestions=()),
            affected_nodes=(),
        )
        feedback = _allowlisted_candidate_feedback(result)
        assert "connectivity" not in feedback["validation"]["errors"][0]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
