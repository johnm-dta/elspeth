"""Phase p4 — deterministic tutorial-sample URL resolver.

The synthetic page URLs are computed by the SEAM, never by the LLM. The pages
are publicly hosted (GitHub Pages for local dev, the deployment's own origin in
production), so the tutorial's web_scrape node relies on the plugin default
``allowed_hosts="public_only"`` — the server injects no SSRF allowlist.
"""

from __future__ import annotations

from pydantic import SecretBytes

from elspeth.web.composer.tutorial_sample import (
    TUTORIAL_SAMPLE_PAGES_BASE_URL,
    resolve_tutorial_sample_urls,
    tutorial_sample_base_url,
)
from elspeth.web.config import WebSettings


def _settings(**kw) -> WebSettings:
    # Required field on WebSettings (config.py:250, strict, no default).
    # Mirror the established fixture pattern at
    # tests/unit/web/preferences/test_tutorial_cache.py:264.
    base: dict[str, object] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": SecretBytes(b"\x00" * 32),
    }
    base.update(kw)
    return WebSettings(**base)


def test_urls_are_built_under_base() -> None:
    urls = resolve_tutorial_sample_urls(base_url="https://elspeth.foundryside.dev")
    assert urls == (
        "https://elspeth.foundryside.dev/tutorial-site/project-1.html",
        "https://elspeth.foundryside.dev/tutorial-site/project-2.html",
        "https://elspeth.foundryside.dev/tutorial-site/project-3.html",
    )


def test_urls_strip_trailing_slash_on_base() -> None:
    urls = resolve_tutorial_sample_urls(base_url="http://127.0.0.1:5173/")
    assert urls[0] == "http://127.0.0.1:5173/tutorial-site/project-1.html"


def test_github_pages_base_builds_public_urls() -> None:
    # The first-run tutorial's synthetic pages are published to the project's
    # public GitHub Pages (https://johnm-dta.github.io/elspeth/tutorial-site/).
    # Pointing ``tutorial_sample_base_url`` there makes a loopback dev box scrape
    # a PUBLIC origin, so the tutorial's web_scrape node uses the plugin default
    # ``public_only`` with no server-injected allowlist. This pins the
    # public-hosting decision: the URLs asserted here are the ones served live.
    base = "https://johnm-dta.github.io/elspeth"
    assert resolve_tutorial_sample_urls(base_url=base) == (
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-1.html",
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-2.html",
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-3.html",
    )


def test_base_url_prefers_configured_setting() -> None:
    settings = _settings(tutorial_sample_base_url="https://configured.example")
    assert tutorial_sample_base_url(settings=settings) == "https://configured.example"


def test_base_url_defaults_to_github_pages_when_unset() -> None:
    # Unset -> the canonical public GitHub Pages copy (never the request origin,
    # which produced loopback URLs the web_scrape SSRF gate refuses).
    settings = _settings()
    assert tutorial_sample_base_url(settings=settings) == TUTORIAL_SAMPLE_PAGES_BASE_URL
    assert TUTORIAL_SAMPLE_PAGES_BASE_URL == "https://johnm-dta.github.io/elspeth"
