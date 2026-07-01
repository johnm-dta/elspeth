"""p4 Task 8a — GET /guided/tutorial-sample surface.

Exposes the runtime-derived synthetic-scrape inputs for the active TUTORIAL
session: the 3 sample URLs. The URLs are runtime-derived (they cannot ride the
frozen profile constants). The synthetic pages are publicly hosted, so the
tutorial's web_scrape node carries no server-injected SSRF allowlist (it relies
on the plugin default ``allowed_hosts="public_only"``).
"""

from __future__ import annotations

from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "tutorial-sample-test"})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _start(client: TestClient, session_id: str, profile: str) -> None:
    resp = client.post(f"/api/sessions/{session_id}/guided/start", json={"profile": profile})
    assert resp.status_code == 200, resp.text


def _set_base_url(client: TestClient, base_url: str | None) -> None:
    """WebSettings is frozen — reconstruct with the tutorial base url override."""
    client.app.state.settings = client.app.state.settings.model_copy(update={"tutorial_sample_base_url": base_url})


def _get_sample(client: TestClient, session_id: str):
    return client.get(f"/api/sessions/{session_id}/guided/tutorial-sample")


def test_tutorial_sample_default_is_github_pages(composer_test_client: TestClient) -> None:
    """No configured base -> the canonical public GitHub Pages URLs (never the
    request origin, which produced loopback URLs the web_scrape gate refuses)."""
    session_id = _create_session(composer_test_client)
    _start(composer_test_client, session_id, "tutorial")

    resp = _get_sample(composer_test_client, session_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["sample_urls"] == [
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-1.html",
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-2.html",
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-3.html",
    ]


def test_tutorial_sample_configured_base_builds_urls_under_it(composer_test_client: TestClient) -> None:
    """A configured (publicly-hosted) base -> the 3 URLs built under that base."""
    session_id = _create_session(composer_test_client)
    _start(composer_test_client, session_id, "tutorial")
    _set_base_url(composer_test_client, "https://johnm-dta.github.io/elspeth")

    resp = _get_sample(composer_test_client, session_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["sample_urls"] == [
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-1.html",
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-2.html",
        "https://johnm-dta.github.io/elspeth/tutorial-site/project-3.html",
    ]


def test_tutorial_sample_exposes_only_sample_urls(composer_test_client: TestClient) -> None:
    """The tutorial-sample wire carries ONLY the runtime-derived sample URLs."""
    session_id = _create_session(composer_test_client)
    _start(composer_test_client, session_id, "tutorial")

    resp = _get_sample(composer_test_client, session_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert set(body.keys()) == {"sample_urls"}


def test_tutorial_sample_rejects_non_tutorial_session(composer_test_client: TestClient) -> None:
    """A live (non-tutorial) guided session has no tutorial sample surface -> 400."""
    session_id = _create_session(composer_test_client)
    _start(composer_test_client, session_id, "live")

    resp = _get_sample(composer_test_client, session_id)
    assert resp.status_code == 400, resp.text
