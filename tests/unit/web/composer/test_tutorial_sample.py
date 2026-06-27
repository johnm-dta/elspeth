"""Phase p4 — deterministic tutorial-sample URL + SSRF allowed_hosts resolver.

The synthetic URLs and the web_scrape allowed_hosts value are computed by the
SEAM, never by the LLM (allowed_hosts is an SSRF control). Both a public host
and a loopback host must be covered (spec mandate, highest-risk detail).
"""

from __future__ import annotations

import pytest
from pydantic import SecretBytes

from elspeth.web.composer.tutorial_sample import (
    resolve_tutorial_allowed_hosts,
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


def test_public_host_uses_public_only() -> None:
    assert resolve_tutorial_allowed_hosts(base_url="https://elspeth.foundryside.dev") == "public_only"


def test_github_pages_base_resolves_public_only_and_public_urls() -> None:
    # The first-run tutorial's synthetic pages are published to the project's
    # public GitHub Pages (https://johnm-dta.github.io/elspeth/tutorial-site/).
    # Pointing ``tutorial_sample_base_url`` there makes a loopback dev box scrape
    # a PUBLIC origin -> ``public_only`` -> the server never injects the loopback
    # CIDR allowlist the execution preflight rejects, so the tutorial executes
    # end-to-end off localhost with NO change to the SSRF boundary. This pins the
    # public-hosting decision: the URLs asserted here are the ones served live.
    base = "https://johnm-dta.github.io/elspeth"
    assert resolve_tutorial_allowed_hosts(base_url=base) == "public_only"
    assert resolve_tutorial_sample_urls(base_url=base) == (
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-1.html",
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-2.html",
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-3.html",
    )


def test_loopback_host_uses_tight_cidr_not_allow_private() -> None:
    hosts = resolve_tutorial_allowed_hosts(base_url="http://127.0.0.1:5173")
    assert hosts == ["127.0.0.1/32", "::1/128"]
    assert hosts != "allow_private"


def test_ipv6_loopback_host_uses_tight_cidr() -> None:
    hosts = resolve_tutorial_allowed_hosts(base_url="http://[::1]:8451")
    assert hosts == ["127.0.0.1/32", "::1/128"]


def test_private_rfc1918_host_uses_tight_cidr() -> None:
    # A private (non-public) host must NOT fall through to public_only, and must
    # land on the SAME tight CIDR list as loopback — never a wider allowlist.
    # Exact equality (not just isinstance list) is the security-critical
    # invariant: a resolver bug returning e.g. ["0.0.0.0/0"] must fail this.
    hosts = resolve_tutorial_allowed_hosts(base_url="http://192.168.1.50:8451")
    assert hosts == ["127.0.0.1/32", "::1/128"]
    assert hosts != "allow_private"


def test_localhost_hostname_uses_tight_cidr() -> None:
    # RFC 6761 reserves ``localhost`` (and ``*.localhost``) for loopback. The
    # common local-dev base http://localhost:PORT must NOT fall through to
    # public_only (``ip_address("localhost")`` raises) — it gets the same tight
    # CIDR as 127.0.0.1, otherwise the scraper blocks the synthetic tutorial pages.
    assert resolve_tutorial_allowed_hosts(base_url="http://localhost:5173") == ["127.0.0.1/32", "::1/128"]
    assert resolve_tutorial_allowed_hosts(base_url="http://app.localhost:8000") == ["127.0.0.1/32", "::1/128"]


def test_base_url_prefers_configured_setting() -> None:
    settings = _settings(tutorial_sample_base_url="https://configured.example")
    assert tutorial_sample_base_url(settings=settings, request_origin="https://ignored.example") == "https://configured.example"


def test_base_url_falls_back_to_request_origin() -> None:
    settings = _settings()
    assert (
        tutorial_sample_base_url(settings=settings, request_origin="https://elspeth.foundryside.dev") == "https://elspeth.foundryside.dev"
    )


def test_base_url_raises_when_undeterminable() -> None:
    settings = _settings()
    with pytest.raises(ValueError):
        tutorial_sample_base_url(settings=settings, request_origin=None)


def test_allowed_hosts_none_host_fails_safe_to_loopback() -> None:
    # A base whose urlsplit().hostname is None (e.g. a bare/relative string)
    # must fail SAFE to the tight loopback CIDR list, NEVER widen egress. This
    # exercises the resolver's `if host is None` fail-safe arm directly — a real
    # branch, not a caller-error-only path.
    assert resolve_tutorial_allowed_hosts(base_url="not-a-url") == ["127.0.0.1/32", "::1/128"]
