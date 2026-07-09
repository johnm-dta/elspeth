"""Tests for ASCII-folding of web_scrape wire-visible header fields.

``http.scraping_reason`` and ``http.abuse_contact`` are sent verbatim as the
``X-Scraping-Reason`` / ``X-Abuse-Contact`` request headers, which must be
ASCII-encodable (enforced by ``WebScrapeHTTPConfig``). The guided LLM composer
routinely emits typographic punctuation (em/en dashes, curly quotes, ellipses),
so the composer folds those to ASCII as options are finalized \u2014 letting
composer-built pipelines (the first-run tutorial) round-trip. Characters with no
ASCII mapping are left untouched so the config validator still rejects them as a
configuration error for hand-authored / YAML configs that bypass the composer.

These pin the auto-folder composed into ``_options_with_default_llm_reviews``.
"""

from __future__ import annotations

from elspeth.web.composer.tools._common import _options_with_default_llm_reviews


def _scrape_options(scraping_reason: str = "Routine fetch", abuse_contact: str = "ops@dta.gov.au") -> dict[str, object]:
    return {
        "url_field": "url",
        "http": {"abuse_contact": abuse_contact, "scraping_reason": scraping_reason},
    }


def test_folds_em_dash_in_scraping_reason() -> None:
    """The exact tutorial failure: an em dash (U+2014) is folded to a hyphen."""
    options = _scrape_options(scraping_reason="Demo pipeline \u2014 fetching own demo pages")
    staged = _options_with_default_llm_reviews(node_id="scrape", plugin="web_scrape", options=options)
    reason = staged["http"]["scraping_reason"]
    assert reason == "Demo pipeline - fetching own demo pages"
    assert reason.isascii()


def test_folds_typographic_punctuation_in_both_header_fields() -> None:
    """En dash, curly quotes, and ellipsis fold to ASCII in both fields."""
    options = _scrape_options(
        scraping_reason="Summarise pages \u2013 \u201cgov\u201d demo\u2026",
        abuse_contact="ops\u2019team@dta.gov.au",
    )
    staged = _options_with_default_llm_reviews(node_id="scrape", plugin="web_scrape", options=options)
    assert staged["http"]["scraping_reason"] == 'Summarise pages - "gov" demo...'
    assert staged["http"]["abuse_contact"] == "ops'team@dta.gov.au"
    assert staged["http"]["scraping_reason"].isascii()
    assert staged["http"]["abuse_contact"].isascii()


def test_noop_for_non_web_scrape_plugin() -> None:
    """A non-scrape plugin's body text (e.g. an LLM prompt) is never folded."""
    options = {"prompt_template": "Write 2\u20134 sentences"}
    staged = _options_with_default_llm_reviews(node_id="t1", plugin="field_mapper", options=options)
    assert staged["prompt_template"] == "Write 2\u20134 sentences"


def test_noop_when_http_block_absent() -> None:
    """web_scrape options without an http block pass through unchanged."""
    options: dict[str, object] = {"url_field": "url"}
    staged = _options_with_default_llm_reviews(node_id="scrape", plugin="web_scrape", options=options)
    assert staged == options


def test_leaves_unmappable_non_ascii_for_the_validator() -> None:
    """Characters with no ASCII mapping are NOT stripped \u2014 the config validator
    must still reject them, so the composer must not silently launder them."""
    options = _scrape_options(scraping_reason="Citizen caf\u00e9 audit \U0001f600")
    staged = _options_with_default_llm_reviews(node_id="scrape", plugin="web_scrape", options=options)
    assert not staged["http"]["scraping_reason"].isascii()
