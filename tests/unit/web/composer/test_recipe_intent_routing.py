"""Tests for deterministic freeform recipe intent routing."""

from __future__ import annotations

import pytest

from elspeth.web.composer.recipe_intent_routing import FreeformRecipeIntentMatch, match_freeform_recipe_intent

FORK_COALESCE_PROMPT = """Please create a pipeline that processes the following customer rows. Each row should be processed two ways in parallel and combined into a single merged output row at outputs/merged.jsonl: path A keeps the original row unchanged, path B truncates the description field to 30 characters with suffix '...'. Combine both branches under separate keys `path_a` and `path_b` in each merged output row -- one input row produces one output row containing both branches side-by-side. Customer rows (CSV):
name,description
alice,this is a moderately long description for testing the truncation behaviour
bob,short note
charlie,another lengthy customer description that exceeds thirty characters comfortably"""


def test_fork_coalesce_truncate_prompt_matches_registered_recipe() -> None:
    match = match_freeform_recipe_intent(FORK_COALESCE_PROMPT)

    assert match is not None
    assert match.recipe_name == "fork-coalesce-truncate-jsonl"
    assert match.inline_blob is not None
    assert match.inline_blob.filename == "inline-fork-coalesce.csv"
    assert match.inline_blob.mime_type == "text/csv"
    assert match.inline_blob.content.startswith("name,description\nalice,")
    assert match.slots == {
        "truncate_field": "description",
        "max_chars": 30,
        "truncation_suffix": "...",
        "output_path": "outputs/merged.jsonl",
        "key_a": "path_a",
        "key_b": "path_b",
    }


def test_non_fork_coalesce_request_does_not_match() -> None:
    prompt = "Classify each CSV row as urgent or normal and write outputs/classified.jsonl."

    assert match_freeform_recipe_intent(prompt) is None


def test_freeform_recipe_intent_match_deep_freezes_slots() -> None:
    match = FreeformRecipeIntentMatch(
        recipe_name="example",
        slots={"nested": {"items": ["a"]}},
    )

    with pytest.raises(TypeError):
        match.slots["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        match.slots["nested"]["items"] = ["b"]  # type: ignore[index,call-overload]
