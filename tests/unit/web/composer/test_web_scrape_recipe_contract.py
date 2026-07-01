"""State-level contract + shield-advisory tests for the web-scrape recipe (D11).

Pins the rev-4 re-polarized shield behaviour: the built pipeline omits the
azure_prompt_shield HARD NODE but the medium-severity prompt-shield ADVISORY
warning IS present (elspeth-abb2cb0931 is a conditional 'restore once plugins
gate on secret availability' ticket, NOT a licence to hide the signal).

The state-builder routes through ``apply_recipe`` (not ``_build_web_scrape_recipe``
directly) so the optional ``provider`` / ``rating_template`` / ``output_path``
slot defaults are injected — mirroring ``TestWebScrapeRecipeBuild._build`` in
``test_recipes.py``. Calling ``_build_web_scrape_recipe`` on the raw slots would
KeyError because those defaults are applied by ``validate_slots``.
"""

from __future__ import annotations

from typing import Any

from elspeth.web.composer.recipes import apply_recipe
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    RAW_HTML_CLEANUP_USER_TERM,
    composition_review_contract_error,
    prompt_shield_recommendation_warning_pairs,
)

_SLOTS = {
    "source_blob_id": "a1b2c3d4-0000-0000-0000-0000000000aa",
    "source_plugin": "json",
    "model": "anthropic/claude-sonnet-4.6",
    "api_key_secret": "OPENROUTER_API_KEY",
    "abuse_contact": "web-scrape-contact@dta.gov.au",
    "scraping_reason": "Tutorial exercise: fetch public pages for rating",
    "output_path": "outputs/ratings.jsonl",
}


def _node_from_args(node_args: dict[str, Any]) -> NodeSpec:
    return NodeSpec(
        id=node_args["id"],
        node_type=node_args["node_type"],
        plugin=node_args["plugin"],
        input=node_args["input"],
        on_success=node_args.get("on_success"),
        on_error=node_args.get("on_error"),
        options=node_args["options"],
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def _state_from_recipe() -> CompositionState:
    # apply_recipe runs validate_slots, which injects the provider /
    # rating_template / output_path defaults the builder reads.
    args = apply_recipe("web-scrape-llm-rate-jsonl", _SLOTS)
    src = args["source"]
    source = SourceSpec(
        plugin=src["plugin"],
        on_success=src["on_success"],
        options=src["options"],
        on_validation_failure=src["on_validation_failure"],
    )
    out = args["outputs"][0]
    output = OutputSpec(
        name=out["sink_name"],
        plugin=out["plugin"],
        options=out["options"],
        on_write_failure=out["on_write_failure"],
    )
    return CompositionState(
        source=source,
        nodes=tuple(_node_from_args(n) for n in args["nodes"]),
        edges=(),
        outputs=(output,),
        metadata=PipelineMetadata(),
        version=1,
    )


def test_built_recipe_passes_blocking_cleanup_contract() -> None:
    """The staged pipeline_decision satisfies raw_html_cleanup_review_contract_error,
    so composition_review_contract_error is None.

    This is also the load-bearing *orphan discriminator*: a None here means the
    raw-HTML cleanup is STAGED (a pending pipeline_decision requirement on the
    field_mapper), not orphaned/missing. See
    test_cleanup_node_stages_not_orphans_raw_html_review for the matching
    positive pin.
    """
    state = _state_from_recipe()
    assert composition_review_contract_error(state) is None


def test_prompt_shield_advisory_is_present_no_hard_node() -> None:
    """Re-polarized shield (rev 4): assert (a) no azure_prompt_shield HARD NODE,
    AND (b) the medium-severity prompt-shield ADVISORY warning IS present — pin
    the PRESENCE of the security signal, not its absence. The flagship example
    must not be the one web_scrape→llm pipeline that hides the warning the rest
    of the system shows.

    See elspeth-abb2cb0931 (conditional 'restore the shield advice once plugins
    gate on secret availability' ticket).
    """
    state = _state_from_recipe()

    # (a) no unbuildable hard node
    assert all(node.plugin != "azure_prompt_shield" for node in state.nodes)

    # (b) the advisory IS present (web_scrape → llm without a shield)
    warning_pairs = prompt_shield_recommendation_warning_pairs(state)
    assert warning_pairs, "expected the medium-severity prompt-shield advisory to fire"
    components = {component for component, _message in warning_pairs}
    assert "node:rate_pages" in components


def test_prompt_shield_advisory_surfaces_in_validate_warnings() -> None:
    """The same advisory rides validate().warnings at 'medium' severity
    (state.py:2421), which is the payload the wire stage renders
    (_authoring_validation_payload['warnings'])."""
    state = _state_from_recipe()
    summary = state.validate()
    shield_warnings = [w for w in summary.warnings if "prompt-injection shield" in w.message and w.severity == "medium"]
    assert shield_warnings, "prompt-shield advisory must ride validate().warnings at medium severity"


def test_cleanup_node_drops_raw_fields() -> None:
    """Data minimization: the raw content/fingerprint fields are NOT preserved
    by the field_mapper(select_only) cleanup node."""
    state = _state_from_recipe()
    fm = next(n for n in state.nodes if n.plugin == "field_mapper")
    mapping = fm.options["mapping"]
    preserved = set(mapping) | set(mapping.values())
    assert "content" not in preserved
    assert "content_fingerprint" not in preserved


def test_cleanup_node_stages_not_orphans_raw_html_review() -> None:
    """The raw-HTML cleanup is STAGED, not orphaned/missing.

    A pending staged requirement and an orphaned missing-cleanup site are the
    *same* InterpretationReviewSite coordinates (_pending_node_sites emits one
    for the staged pending requirement; _missing_raw_html_cleanup_review_sites
    emits an identical one ONLY when the requirement is absent). The two are
    therefore indistinguishable via interpretation_sites — the orphan check is
    the blocking *contract* (composition_review_contract_error), asserted here
    alongside a positive pin that the field_mapper carries a pending
    pipeline_decision requirement with the canonical user_term.
    """
    state = _state_from_recipe()

    # No orphan: the blocking cleanup contract passes because the requirement
    # is staged (not missing).
    assert composition_review_contract_error(state) is None

    # Positive pin: the staged requirement IS on the field_mapper, pending,
    # carrying the canonical raw-HTML-cleanup user_term.
    fm = next(n for n in state.nodes if n.plugin == "field_mapper")
    reqs = fm.options[INTERPRETATION_REQUIREMENTS_KEY]
    decision = next(r for r in reqs if r["kind"] == "pipeline_decision")
    assert decision["user_term"] == RAW_HTML_CLEANUP_USER_TERM
    assert decision["status"] == "pending"
