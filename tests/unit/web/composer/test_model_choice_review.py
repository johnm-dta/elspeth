"""Tests for the ``llm_model_choice`` interpretation review surface.

The composer cannot ship an LLM model identifier without surfacing it
for user review. This file pins the auto-stager (mutation-time gate),
the missing-site enumerator (state-load gate), the execution-time hash
guard, and the resolution-time draft equality check.

Tests use the public ``InterpretationKind`` enum and the private helpers
in ``elspeth.web.composer.tools._common`` and
``elspeth.web.interpretation_state`` because those are the surfaces a
future regression would touch first.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.web.composer.state import CompositionState, NodeSpec, PipelineMetadata
from elspeth.web.composer.tools._common import (
    _options_with_default_llm_reviews,
    _options_with_default_model_choice_review,
)
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    _missing_model_choice_review_sites,
    _validate_model_choice_review,
    materialize_state_for_execution,
)


def _llm_node(node_id: str, options: dict[str, object]) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type="transform",
        plugin="llm",
        input="rows",
        on_success="results",
        on_error="discard",
        options=options,
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


class TestModelChoiceAutoStager:
    """Mutation-time gate: setting ``options.model`` stages the review."""

    def test_stages_review_when_model_set(self) -> None:
        options = {"model": "anthropic/claude-sonnet-4.5"}
        staged = _options_with_default_model_choice_review(node_id="llm1", plugin="llm", options=options)
        requirements = staged[INTERPRETATION_REQUIREMENTS_KEY]
        assert len(requirements) == 1
        assert requirements[0]["kind"] == InterpretationKind.LLM_MODEL_CHOICE.value
        assert requirements[0]["user_term"] == "llm_model_choice:llm1"
        assert requirements[0]["draft"] == "anthropic/claude-sonnet-4.5"
        assert requirements[0]["status"] == "pending"
        assert requirements[0]["id"] == "model_choice_review:llm1"

    def test_noop_for_non_llm_plugin(self) -> None:
        options = {"model": "anthropic/claude-sonnet-4.5"}
        staged = _options_with_default_model_choice_review(node_id="t1", plugin="field_mapper", options=options)
        assert INTERPRETATION_REQUIREMENTS_KEY not in staged

    def test_noop_when_model_absent(self) -> None:
        staged = _options_with_default_model_choice_review(node_id="llm1", plugin="llm", options={"prompt_template": "Hi"})
        assert INTERPRETATION_REQUIREMENTS_KEY not in staged

    def test_noop_when_model_empty(self) -> None:
        staged = _options_with_default_model_choice_review(node_id="llm1", plugin="llm", options={"model": ""})
        assert INTERPRETATION_REQUIREMENTS_KEY not in staged

    def test_idempotent_when_model_unchanged(self) -> None:
        """Re-staging with the same model keeps the existing requirement object."""
        options = {"model": "anthropic/claude-sonnet-4.5"}
        first = _options_with_default_model_choice_review(node_id="llm1", plugin="llm", options=options)
        second = _options_with_default_model_choice_review(node_id="llm1", plugin="llm", options=first)
        # Same draft → same single requirement, not duplicated.
        assert len(second[INTERPRETATION_REQUIREMENTS_KEY]) == 1
        assert second[INTERPRETATION_REQUIREMENTS_KEY][0]["draft"] == "anthropic/claude-sonnet-4.5"

    def test_replaces_when_model_changes(self) -> None:
        """Changing the model rotates the pending requirement's draft."""
        first = _options_with_default_model_choice_review(node_id="llm1", plugin="llm", options={"model": "old/model"})
        second = _options_with_default_model_choice_review(
            node_id="llm1",
            plugin="llm",
            options={**dict(first), "model": "new/model"},
        )
        requirements = second[INTERPRETATION_REQUIREMENTS_KEY]
        assert len(requirements) == 1
        assert requirements[0]["draft"] == "new/model"


class TestCompositeAutoStager:
    """``_options_with_default_llm_reviews`` stages every default LLM gate."""

    def test_composite_stages_both_prompt_and_model(self) -> None:
        options = {
            "model": "anthropic/claude-sonnet-4.5",
            "prompt_template": "Score {{ row.x }}",
        }
        staged = _options_with_default_llm_reviews(node_id="llm1", plugin="llm", options=options)
        kinds = [r["kind"] for r in staged[INTERPRETATION_REQUIREMENTS_KEY]]
        assert InterpretationKind.LLM_PROMPT_TEMPLATE.value in kinds
        assert InterpretationKind.LLM_MODEL_CHOICE.value in kinds


class TestMissingSiteEnumerator:
    """State-load gate: pre-auto-stager sessions still get a review site."""

    def test_emits_site_when_no_requirement(self) -> None:
        node = _llm_node("llm1", {"model": "anthropic/claude-sonnet-4.5"})
        sites = _missing_model_choice_review_sites(node)
        assert len(sites) == 1
        assert sites[0].kind is InterpretationKind.LLM_MODEL_CHOICE
        assert sites[0].component_id == "llm1"
        assert sites[0].user_term == "llm_model_choice:llm1"

    def test_silent_when_resolved(self) -> None:
        node = _llm_node(
            "llm1",
            {
                "model": "anthropic/claude-sonnet-4.5",
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "model_choice_review:llm1",
                        "kind": "llm_model_choice",
                        "user_term": "llm_model_choice:llm1",
                        "status": "resolved",
                        "draft": "anthropic/claude-sonnet-4.5",
                        "event_id": "e1",
                        "accepted_value": "anthropic/claude-sonnet-4.5",
                        "accepted_artifact_hash": None,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            },
        )
        assert _missing_model_choice_review_sites(node) == ()

    def test_silent_for_non_llm_node(self) -> None:
        node = _llm_node("t1", {"model": "x"})
        node = NodeSpec(
            id="t1",
            node_type="transform",
            plugin="field_mapper",
            input="rows",
            on_success="results",
            on_error="discard",
            options={"model": "x"},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        assert _missing_model_choice_review_sites(node) == ()


class TestExecutionGuard:
    """Resolved model-choice hash must match the live ``options.model``."""

    def test_hash_match_passes(self) -> None:
        from elspeth.contracts.hashing import stable_hash

        chosen = "anthropic/claude-sonnet-4.5"
        node = _llm_node(
            "llm1",
            {
                "model": chosen,
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "model_choice_review:llm1",
                        "kind": "llm_model_choice",
                        "user_term": "llm_model_choice:llm1",
                        "status": "resolved",
                        "draft": chosen,
                        "event_id": "e1",
                        "accepted_value": chosen,
                        "accepted_artifact_hash": None,
                        "resolved_prompt_template_hash": stable_hash(chosen),
                    }
                ],
            },
        )
        _validate_model_choice_review(node, chosen)  # no raise

    def test_hash_drift_raises(self) -> None:
        from elspeth.contracts.hashing import stable_hash

        node = _llm_node(
            "llm1",
            {
                "model": "anthropic/claude-sonnet-4.5",
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "model_choice_review:llm1",
                        "kind": "llm_model_choice",
                        "user_term": "llm_model_choice:llm1",
                        "status": "resolved",
                        "draft": "anthropic/claude-sonnet-4.5",
                        "event_id": "e1",
                        "accepted_value": "anthropic/claude-sonnet-4.5",
                        "accepted_artifact_hash": None,
                        "resolved_prompt_template_hash": stable_hash("anthropic/claude-sonnet-4.5"),
                    }
                ],
            },
        )
        with pytest.raises(ValueError, match="model-choice review hash drifted"):
            _validate_model_choice_review(node, "openai/gpt-5")

    def test_no_resolved_requirement_passes_silently(self) -> None:
        node = _llm_node("llm1", {"model": "anthropic/claude-sonnet-4.5"})
        _validate_model_choice_review(node, "anthropic/claude-sonnet-4.5")  # no raise

    @pytest.mark.parametrize("legacy_status", ["pending", "resolved"])
    def test_operator_profile_model_ignores_legacy_user_review(self, legacy_status: str) -> None:
        """Profile rotation is operator policy even when old review metadata remains."""
        from elspeth.contracts.hashing import stable_hash

        old_model = "anthropic/claude-sonnet-4.5"
        requirement: dict[str, object] = {
            "id": "model_choice_review:llm1",
            "kind": "llm_model_choice",
            "user_term": "llm_model_choice:llm1",
            "status": legacy_status,
            "draft": old_model,
            "event_id": "e1" if legacy_status == "resolved" else None,
            "accepted_value": old_model if legacy_status == "resolved" else None,
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": stable_hash(old_model) if legacy_status == "resolved" else None,
        }
        node = _llm_node(
            "llm1",
            {
                "model": "bedrock/zai.glm-5",
                INTERPRETATION_REQUIREMENTS_KEY: [requirement],
            },
        )
        state = CompositionState(
            nodes=(node,),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

        materialized = materialize_state_for_execution(
            state,
            operator_resolved_model_node_ids=frozenset({"llm1"}),
        )

        assert isinstance(materialized, CompositionState)

    def test_profile_resolved_llm_node_is_exempt_without_explicit_arg(self) -> None:
        """Profile-aliased model needs no review even when the caller omits the exemption.

        Parity regression (freeform session 0c59fbca): the run-readiness path
        computes ``operator_resolved_model_node_ids`` and passes it, but the
        execute path (``execution.service`` -> ``materialize_state_for_execution``)
        called it with no argument, so an ``ab_assess`` node carrying a profile
        alias plus a stale pending ``llm_model_choice`` review 422'd at /execute
        while readiness said READY. The exemption is a property of the state
        (an LLM node bound to a ``profile`` alias), so it must hold with no
        explicit argument.
        """
        node = _llm_node(
            "ab_assess",
            {
                "profile": "sonnet",
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "model_choice_review:ab_assess",
                        "kind": "llm_model_choice",
                        "user_term": "llm_model_choice:ab_assess",
                        "status": "pending",
                        "draft": "sonnet",
                        "event_id": None,
                        "accepted_value": None,
                        "accepted_artifact_hash": None,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            },
        )
        state = CompositionState(
            nodes=(node,),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

        materialized = materialize_state_for_execution(state)

        assert isinstance(materialized, CompositionState)
