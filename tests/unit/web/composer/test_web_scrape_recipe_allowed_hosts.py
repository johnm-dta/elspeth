"""Phase p4 Task 6 — the optional allowed_hosts SSRF slot on the web-scrape recipe.

Behaviour-preserving: no slot value -> the web_scrape node omits allowed_hosts
(the field default "public_only" applies, the current behaviour). A non-empty
CIDR list (loopback dev) -> the node carries it. The list comes from the Task 2
resolver, NEVER from the LLM (SSRF control).
"""

from __future__ import annotations

from elspeth.web.composer.recipes import get_recipe, validate_slots
from elspeth.web.composer.tutorial_sample import resolve_tutorial_allowed_hosts

_BASE_SLOTS = {
    "source_blob_id": "00000000-0000-0000-0000-000000000001",
    "source_plugin": "json",
    "model": "anthropic/claude-sonnet-4.6",
    "api_key_secret": "OPENROUTER_API_KEY",
    "provider": "openrouter",
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


def test_loopback_cidr_list_is_emitted() -> None:
    recipe = get_recipe("web-scrape-llm-rate-jsonl")
    assert recipe is not None
    hosts = resolve_tutorial_allowed_hosts(base_url="http://127.0.0.1:5173")
    assert isinstance(hosts, list)
    slots = validate_slots(recipe, {**_BASE_SLOTS, "allowed_hosts": hosts})
    args = recipe.build(slots)
    node = _web_scrape_node(args)
    # allowed_hosts is a WebScrapeHTTPConfig field -> nests under ``http`` (the
    # plugin rejects a top-level key with extra:forbid). The previous assertion
    # checked the top level and only passed because it never applied the dict
    # through WebScrapeConfig.
    assert node["options"]["http"]["allowed_hosts"] == ["127.0.0.1/32", "::1/128"]
    assert "allowed_hosts" not in node["options"]


def test_public_resolver_yields_omit() -> None:
    # The public resolver returns "public_only" (scalar), which the tutorial
    # threading maps to an EMPTY slot list -> omit (field default applies).
    recipe = get_recipe("web-scrape-llm-rate-jsonl")
    assert recipe is not None
    hosts = resolve_tutorial_allowed_hosts(base_url="https://elspeth.foundryside.dev")
    assert hosts == "public_only"
    # public_only is the field default; the slot stays empty -> omit.
    slots = validate_slots(recipe, {**_BASE_SLOTS, "allowed_hosts": []})
    args = recipe.build(slots)
    node = _web_scrape_node(args)
    assert "allowed_hosts" not in node["options"]["http"]
    assert "allowed_hosts" not in node["options"]
