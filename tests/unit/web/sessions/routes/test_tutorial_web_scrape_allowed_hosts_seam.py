"""Unit test: the tutorial web_scrape allowed_hosts injection-DECISION seam.

``_tutorial_web_scrape_allowed_hosts`` decides WHAT SSRF allowlist the server
injects into a TUTORIAL session's web_scrape node (the LLM never sets an SSRF
control). It is the glue between ``resolve_tutorial_allowed_hosts`` (host-class
-> allowlist) and ``handle_step_3_chain_accept`` (the injection point). This pins
the decision branch-by-branch: tutorial + loopback origin -> tight CIDR; tutorial
+ public origin -> ``public_only``; live (``EMPTY_PROFILE``) -> ``None``; missing
settings -> ``None``; and an unresolvable base degrades SAFELY to ``None`` rather
than raising at accept.
"""

from __future__ import annotations

from pydantic import SecretBytes

from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.config import WebSettings
from elspeth.web.sessions.routes._helpers import _tutorial_web_scrape_allowed_hosts


def _settings(**kw: object) -> WebSettings:
    base: dict[str, object] = {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": SecretBytes(b"\x00" * 32),
    }
    base.update(kw)
    return WebSettings(**base)


_LOOPBACK = ["127.0.0.1/32", "::1/128"]


def test_tutorial_loopback_origin_gets_tight_cidr() -> None:
    session = GuidedSession.initial(profile=TUTORIAL_PROFILE)
    hosts = _tutorial_web_scrape_allowed_hosts(session, _settings(), request_origin="http://localhost:5173")
    assert hosts == _LOOPBACK


def test_tutorial_public_origin_gets_public_only() -> None:
    session = GuidedSession.initial(profile=TUTORIAL_PROFILE)
    hosts = _tutorial_web_scrape_allowed_hosts(
        session,
        _settings(tutorial_sample_base_url="https://elspeth.foundryside.dev"),
        request_origin=None,
    )
    assert hosts == "public_only"


def test_live_session_gets_no_injection() -> None:
    # EMPTY_PROFILE (live guided): the operator owns its own allowlist there.
    session = GuidedSession.initial()  # defaults to EMPTY_PROFILE
    assert _tutorial_web_scrape_allowed_hosts(session, _settings(), request_origin="http://localhost:5173") is None


def test_none_settings_gets_no_injection() -> None:
    session = GuidedSession.initial(profile=TUTORIAL_PROFILE)
    assert _tutorial_web_scrape_allowed_hosts(session, None, request_origin="http://localhost:5173") is None


def test_unresolvable_base_degrades_to_none_not_raise() -> None:
    # No configured base AND no request origin -> tutorial_sample_base_url raises;
    # the seam degrades SAFELY to None (no injection, web_scrape's default
    # public_only) rather than 500 the accept on a misconfigured local-dev edge.
    session = GuidedSession.initial(profile=TUTORIAL_PROFILE)
    assert _tutorial_web_scrape_allowed_hosts(session, _settings(), request_origin=None) is None
