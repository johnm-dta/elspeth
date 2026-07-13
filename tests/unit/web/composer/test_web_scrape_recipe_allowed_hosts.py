"""Phase p4 Task 6 — the optional allowed_hosts SSRF slot on the web-scrape recipe.

Behaviour-preserving: no slot value -> the web_scrape node omits allowed_hosts
(the field default "public_only" applies, the current behaviour). A non-empty
CIDR list -> the node carries it under ``http``. The list is operator/recipe
supplied and constrained by the web_scrape enforcement boundary (CidrStr
validation + the "public_only" field default), NEVER fabricated by the LLM.
"""

from __future__ import annotations

from elspeth.web.composer.recipes import get_recipe, validate_slots

_BASE_SLOTS = {
    "source_blob_id": "00000000-0000-0000-0000-000000000001",
    "source_plugin": "json",
    "profile": "tutorial-default",
    "rating_template": "Extract fields and return JSON.",
    "abuse_contact": "noreply@dta.gov.au",
    "scraping_reason": "DTA technical demonstration",
    "output_path": "outputs/out.jsonl",
}


def _web_scrape_node(args: dict) -> dict:
    return next(n for n in args["nodes"] if n["plugin"] == "web_scrape")


def test_default_omits_allowed_hosts() -> None:
    recipe = get_recipe("web-scrape-llm-rate-jsonl")
    assert recipe is not None
    slots = validate_slots(recipe, dict(_BASE_SLOTS))
    args = recipe.build(slots)
    node = _web_scrape_node(args)
    # Behaviour-preserving: no allowed_hosts key -> field default public_only.
    # The allowlist is a WebScrapeHTTPConfig field, so its canonical location is
    # under ``http`` (and it must not be stranded at the top level either).
    assert "allowed_hosts" not in node["options"]["http"]
    assert "allowed_hosts" not in node["options"]


def test_supplied_cidr_list_nests_under_http() -> None:
    recipe = get_recipe("web-scrape-llm-rate-jsonl")
    assert recipe is not None
    slots = validate_slots(recipe, {**_BASE_SLOTS, "allowed_hosts": ["127.0.0.1/32", "::1/128"]})
    args = recipe.build(slots)
    node = _web_scrape_node(args)
    # allowed_hosts is a WebScrapeHTTPConfig field -> nests under ``http`` (the
    # plugin rejects a top-level key with extra:forbid). It must not be stranded
    # at the top level.
    assert node["options"]["http"]["allowed_hosts"] == ["127.0.0.1/32", "::1/128"]
    assert "allowed_hosts" not in node["options"]


def test_empty_slot_omits_allowed_hosts() -> None:
    # An explicitly empty slot list -> omit (the field default "public_only"
    # applies), same as supplying no slot at all.
    recipe = get_recipe("web-scrape-llm-rate-jsonl")
    assert recipe is not None
    slots = validate_slots(recipe, {**_BASE_SLOTS, "allowed_hosts": []})
    args = recipe.build(slots)
    node = _web_scrape_node(args)
    assert "allowed_hosts" not in node["options"]["http"]
    assert "allowed_hosts" not in node["options"]
