"""Deterministic resolver for the tutorial synthetic-scrape scenario (p4).

Turns the canonical sample-page base into the 3 synthetic-page URLs for the
tutorial's ``web_scrape`` node. The URLs are computed by the SEAM — the LLM
never sets them (it would otherwise fabricate them, violating the Tier-1
"exception not implicit fabrication" doctrine). The 3 concrete URLs are handed
to the p1 source driver as the dataset (contract §2.2/§3.4).

The synthetic pages are published to the project's PUBLIC GitHub Pages, which is
the canonical default base (``TUTORIAL_SAMPLE_PAGES_BASE_URL``). Two reasons:
the content is operator-controlled, so the always-on prompt-injection-shield
teaching moment stays deterministic; and it needs no local hosting, so the
tutorial works on any deployment — including a pure loopback dev box — without
the app serving the pages itself. Because the base is a public origin, the
tutorial's ``web_scrape`` node relies on the plugin default
``allowed_hosts="public_only"`` with no server-injected allowlist — full parity
with a normal backend run. (Earlier revisions derived the base from the request
origin and minted a loopback CIDR allowlist for localhost dev boxes; both were
removed — the request-origin derivation produced loopback URLs the execution
validator correctly refuses.)
"""

from __future__ import annotations

from elspeth.web.config import WebSettings

_PAGES = ("project-1.html", "project-2.html", "project-3.html")

# Canonical public base for the synthetic tutorial pages. Operator-controlled
# content published to GitHub Pages. A fork republishing its own copy overrides
# this via ``ELSPETH_WEB__TUTORIAL_SAMPLE_BASE_URL``.
TUTORIAL_SAMPLE_PAGES_BASE_URL = "https://johnm-dta.github.io/elspeth"


def resolve_tutorial_sample_urls(*, base_url: str) -> tuple[str, str, str]:
    """Build the 3 absolute synthetic-page URLs under ``base_url``."""
    root = base_url.rstrip("/")
    urls = tuple(f"{root}/tutorial-site/{page}" for page in _PAGES)
    return (urls[0], urls[1], urls[2])


def tutorial_sample_base_url(*, settings: WebSettings) -> str:
    """Resolve the tutorial sample base URL.

    An operator-configured ``WebSettings.tutorial_sample_base_url`` wins; when
    unset, the canonical public GitHub Pages copy applies. Never derived from the
    request origin — that produced loopback URLs on local-dev boxes that the
    ``web_scrape`` SSRF gate (correctly) refuses, and a public, operator-owned
    base is what keeps the prompt-shield teaching moment deterministic.
    """
    if settings.tutorial_sample_base_url:
        return settings.tutorial_sample_base_url
    return TUTORIAL_SAMPLE_PAGES_BASE_URL
