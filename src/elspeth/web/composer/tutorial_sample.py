"""Deterministic resolver for the tutorial synthetic-scrape scenario (p4).

Turns the app's reachable origin into the 3 synthetic-page URLs and the SSRF
``allowed_hosts`` value for the tutorial's ``web_scrape`` node. Everything here
is computed by the SEAM — the LLM never sets the URLs (it would otherwise
fabricate them, violating the Tier-1 "exception not implicit fabrication"
doctrine) and never sets ``allowed_hosts`` (an SSRF control). The 3 concrete
URLs are handed to the p1 source driver as the dataset (contract §2.2/§3.4).
"""

from __future__ import annotations

import ipaddress
from typing import Literal
from urllib.parse import urlsplit

from elspeth.web.config import WebSettings

_PAGES = ("project-1.html", "project-2.html", "project-3.html")
# Tight loopback CIDR for local dev. NOT "allow_private" — the tightest
# allowlist that lets the backend reach its own SPA mount on loopback.
_LOOPBACK_CIDRS = ["127.0.0.1/32", "::1/128"]


def resolve_tutorial_sample_urls(*, base_url: str) -> tuple[str, str, str]:
    """Build the 3 absolute synthetic-page URLs under ``base_url``."""
    root = base_url.rstrip("/")
    urls = tuple(f"{root}/tutorial-site/{page}" for page in _PAGES)
    return (urls[0], urls[1], urls[2])


def resolve_tutorial_allowed_hosts(*, base_url: str) -> Literal["public_only"] | list[str]:
    """Derive the web_scrape SSRF allowlist from the resolved host class.

    Public host -> the default ``"public_only"`` policy is sufficient and is
    the tightest correct value. Loopback / private host (local dev) -> a tight
    CIDR list, NEVER the blanket ``"allow_private"``.
    """
    host = urlsplit(base_url).hostname
    if host is None:
        # Undeterminable host: fail safe to the tight loopback list rather than
        # widening egress. This is a genuine fail-safe, not dead code:
        # tutorial_sample_base_url returns the configured/origin value VERBATIM
        # with no URL validation, so a malformed base (e.g. a bare/relative
        # string whose urlsplit().hostname is None) reaches this arm. Failing to
        # the tightest list is the safe-by-default choice. Covered by
        # test_allowed_hosts_none_host_fails_safe_to_loopback.
        return list(_LOOPBACK_CIDRS)
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A bare hostname. RFC 6761 reserves ``localhost`` (and ``*.localhost``)
        # for loopback, so the common local-dev base ``http://localhost:PORT``
        # gets the same tight loopback CIDR as ``127.0.0.1`` — otherwise it would
        # be misclassified as ``public_only`` and the scraper would block the
        # synthetic tutorial pages. Any other hostname (e.g.
        # elspeth.foundryside.dev) -> ``public_only`` covers it.
        if host == "localhost" or host.endswith(".localhost"):
            return list(_LOOPBACK_CIDRS)
        return "public_only"
    if address.is_global:
        return "public_only"
    return list(_LOOPBACK_CIDRS)


def tutorial_sample_base_url(*, settings: WebSettings, request_origin: str | None) -> str:
    """Resolve the base URL: configured setting wins, else the request origin.

    Raises ``ValueError`` when neither is available — we never fabricate a host
    for an SSRF-controlled fetch.
    """
    if settings.tutorial_sample_base_url:
        return settings.tutorial_sample_base_url
    if request_origin:
        return request_origin
    raise ValueError(
        "tutorial_sample_base_url is unset and no request origin is available; "
        "set WebSettings.tutorial_sample_base_url for this deployment."
    )
