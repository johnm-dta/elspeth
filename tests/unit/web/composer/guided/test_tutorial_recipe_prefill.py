"""p4 Task 8b (reimagined) — tutorial recipe-slot prefill (pure helper).

A passive tutorial learner cannot TYPE the operator-fillable recipe slots, so
for a TUTORIAL session the server moves the worked-example-fillable required
slots out of ``RecipeMatch.unsatisfied_slots`` into ``slots`` with honest
worked-example values. The recipe_offer then surfaces them PREFILLED (read-only)
with no required-empty knob, so the learner's single "Apply recipe" click is
enabled.

These tests pin the pure helper ``prefill_tutorial_recipe_slots`` against a REAL
``match_recipe`` output (not a hand-built fixture), so they catch drift between
the matcher's unsatisfied set and the prefill's value map.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from elspeth.web.composer.guided.emitters import build_step_2_5_recipe_offer_turn
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.recipe_match import RecipeMatch, match_recipe
from elspeth.web.composer.guided.resolved import (
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
)
from elspeth.web.composer.guided.tutorial_recipe_prefill import (
    TUTORIAL_ABUSE_CONTACT,
    TUTORIAL_SCRAPING_REASON,
    prefill_tutorial_recipe_slots,
)
from elspeth.web.config import WebSettings

WEB_SCRAPE_RECIPE = "web-scrape-llm-rate-jsonl"


def _settings(tmp_path: Path, *, allowlist: tuple[str, ...] | None = None) -> WebSettings:
    kwargs: dict[str, object] = {
        "data_dir": tmp_path,
        "composer_model": "anthropic/claude-sonnet-4.6",
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }
    if allowlist is not None:
        kwargs["server_secret_allowlist"] = allowlist
    return WebSettings(**kwargs)  # type: ignore[arg-type]


def _web_scrape_match() -> RecipeMatch:
    """A REAL web-scrape RecipeMatch with the 4 operator-fillable unsatisfied slots."""
    source = SourceResolved(
        plugin="json",
        options={"blob_ref": "blob-123"},
        observed_columns=("url",),
        sample_rows=({"url": "http://127.0.0.1/tutorial-site/project-1.html"},),
    )
    sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="json",
                options={"path": "out.jsonl"},
                required_fields=(),
                schema_mode="observed",
            ),
        )
    )
    match = match_recipe(source, sink)
    assert match is not None
    assert match.recipe_name == WEB_SCRAPE_RECIPE
    # Precondition: the four operator-fillable slots are unsatisfied to start.
    assert set(match.unsatisfied_slots) == {
        "model",
        "api_key_secret",
        "abuse_contact",
        "scraping_reason",
    }
    return match


def test_prefill_moves_all_four_slots_into_slots(tmp_path: Path) -> None:
    match = _web_scrape_match()
    out = prefill_tutorial_recipe_slots(recipe_match=match, settings=_settings(tmp_path))

    # All four required slots are now satisfied → unsatisfied is empty → Apply
    # is enabled with nothing required-empty.
    assert dict(out.unsatisfied_slots) == {}
    assert out.slots["model"] == "anthropic/claude-sonnet-4.6"  # == settings.composer_model
    assert out.slots["api_key_secret"] == "OPENROUTER_API_KEY"
    assert out.slots["abuse_contact"] == TUTORIAL_ABUSE_CONTACT == "noreply@demo.com"
    assert out.slots["scraping_reason"] == TUTORIAL_SCRAPING_REASON


def test_prefill_preserves_resolved_slots(tmp_path: Path) -> None:
    match = _web_scrape_match()
    out = prefill_tutorial_recipe_slots(recipe_match=match, settings=_settings(tmp_path))
    # The three matcher-derived slots ride through untouched.
    assert out.slots["source_blob_id"] == "blob-123"
    assert out.slots["source_plugin"] == "json"
    # The resolver carries the operator-set sink path verbatim ("out.jsonl"),
    # not the recipe's default — prefill must not disturb it.
    assert out.slots["output_path"] == "out.jsonl"
    assert out.recipe_name == WEB_SCRAPE_RECIPE


def test_prefill_model_rides_deployment_composer_model(tmp_path: Path) -> None:
    match = _web_scrape_match()
    settings = _settings(tmp_path)
    out = prefill_tutorial_recipe_slots(recipe_match=match, settings=settings)
    assert out.slots["model"] == settings.composer_model


def test_prefill_is_deterministic(tmp_path: Path) -> None:
    match = _web_scrape_match()
    settings = _settings(tmp_path)
    first = prefill_tutorial_recipe_slots(recipe_match=match, settings=settings)
    second = prefill_tutorial_recipe_slots(recipe_match=match, settings=settings)
    assert dict(first.slots) == dict(second.slots)
    assert dict(first.unsatisfied_slots) == dict(second.unsatisfied_slots)


def test_prefill_fails_closed_when_secret_ref_not_allowlisted(tmp_path: Path) -> None:
    match = _web_scrape_match()
    # Deployment has NOT allowlisted the openrouter secret-ref → the tutorial
    # cannot wire an LLM credential, so prefill fails closed rather than
    # fabricating one.
    settings = _settings(tmp_path, allowlist=("ANTHROPIC_API_KEY",))
    with pytest.raises(InvariantError):
        prefill_tutorial_recipe_slots(recipe_match=match, settings=settings)


def test_prefilled_offer_emits_empty_knobs(tmp_path: Path) -> None:
    """The real web-scrape tutorial offer: all 4 slots prefilled -> knobs.fields == [].

    This is the brand-new empty-knobs UI state the passive web-scrape learner
    hits at STEP_2.5 (before this change every offer had >=1 unsatisfied slot).
    Pin that the emitter produces a well-formed empty-knobs payload whose
    ``prefilled`` carries all seven slots for the frontend to resubmit.
    """
    match = _web_scrape_match()
    prefilled = prefill_tutorial_recipe_slots(recipe_match=match, settings=_settings(tmp_path))
    turn = build_step_2_5_recipe_offer_turn(prefilled)
    payload = turn["payload"]

    assert payload["mode"] == "recipe_decision"
    assert payload["knobs"]["fields"] == []
    assert set(payload["prefilled"]) == {
        "source_blob_id",
        "source_plugin",
        "output_path",
        "model",
        "api_key_secret",
        "abuse_contact",
        "scraping_reason",
    }


def test_prefill_noop_when_no_overlap(tmp_path: Path) -> None:
    # A RecipeMatch with no tutorial-fillable unsatisfied slots is returned
    # unchanged (defensive: the gate is the tutorial profile, not the recipe).
    match = RecipeMatch(
        recipe_name=WEB_SCRAPE_RECIPE,
        slots={"source_blob_id": "blob-123", "source_plugin": "json", "output_path": "x.jsonl"},
        unsatisfied_slots={},
    )
    out = prefill_tutorial_recipe_slots(recipe_match=match, settings=_settings(tmp_path))
    assert out is match
