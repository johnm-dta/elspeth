"""Zero-LLM compose gate for the web-scrape recipe (D11, §4.1).

Building the canonical recipe is a pure, deterministic function: it must make
ZERO LLM provider calls. Pins the §4.1 claim that the canonical pipeline
composes with no frontier round-trip at recipe-build time.

P4 owns the recipe build, not the dispatch wiring (P2). This test proves the
recipe-build path itself is provider-free: building the set_pipeline args and
validating the slots calls the LLM zero times. The llm node IS present in the
COMPOSED pipeline (it runs at RUN time, never at compose time).

_SLOTS carries provider + rating_template explicitly because
``_build_web_scrape_recipe`` reads ``slots["provider"]`` / ``slots["rating_template"]``
directly; those keys are only injected by ``validate_slots`` when the build is
reached through ``apply_recipe``. Passing them explicitly keeps the direct-build
test honest while remaining valid input to ``apply_recipe`` (both are declared
slots).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

from elspeth.web.composer.recipes import _build_web_scrape_recipe, apply_recipe

_SLOTS = {
    "source_blob_id": str(uuid4()),
    "source_plugin": "json",
    "model": "anthropic/claude-sonnet-4.6",
    "api_key_secret": "OPENROUTER_API_KEY",
    "provider": "openrouter",
    "rating_template": "Rate this page from 1-10:\n\n{{ row['content'] }}",
    "abuse_contact": "web-scrape-contact@dta.gov.au",
    "scraping_reason": "Tutorial exercise: fetch public pages for rating",
    "output_path": "outputs/ratings.jsonl",
}


def test_build_web_scrape_recipe_makes_zero_llm_calls() -> None:
    with patch(
        "elspeth.web.composer.service._litellm_acompletion",
        new_callable=AsyncMock,
    ) as mock_acomp:
        args = _build_web_scrape_recipe(_SLOTS)
        # llm node IS present in the COMPOSED pipeline (it runs at RUN time,
        # not compose time) — but the build itself called no provider.
        assert any(n["plugin"] == "llm" for n in args["nodes"])
    assert mock_acomp.call_count == 0


def test_apply_web_scrape_recipe_makes_zero_llm_calls() -> None:
    with patch(
        "elspeth.web.composer.service._litellm_acompletion",
        new_callable=AsyncMock,
    ) as mock_acomp:
        args = apply_recipe("web-scrape-llm-rate-jsonl", _SLOTS)
        assert args["metadata"]["name"] == "web-scrape-llm-rate-jsonl"
    assert mock_acomp.call_count == 0
