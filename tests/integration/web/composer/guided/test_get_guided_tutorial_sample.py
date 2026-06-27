"""p4 Task 8a — GET /guided/tutorial-sample surface.

Exposes the runtime-derived synthetic-scrape inputs for the active TUTORIAL
session: the 3 sample URLs + the SSRF host-class (``allowed_hosts``). The URLs
are runtime-derived (they cannot ride the frozen profile constants) and the
host-class is the deterministic resolver output. The wire carries only those two
runtime-derived fields.
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


def test_tutorial_sample_default_origin_is_public_only(composer_test_client: TestClient) -> None:
    """No configured base -> request origin (host 'test') is a public host -> public_only."""
    session_id = _create_session(composer_test_client)
    _start(composer_test_client, session_id, "tutorial")

    resp = _get_sample(composer_test_client, session_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["sample_urls"]) == 3
    for i, url in enumerate(body["sample_urls"], start=1):
        assert url.endswith(f"/tutorial-site/project-{i}.html"), url
    assert body["allowed_hosts"] == "public_only"


def test_tutorial_sample_configured_loopback_base_yields_cidr(composer_test_client: TestClient) -> None:
    """A configured loopback base -> the tight loopback CIDR list, urls under that base."""
    session_id = _create_session(composer_test_client)
    _start(composer_test_client, session_id, "tutorial")
    _set_base_url(composer_test_client, "http://127.0.0.1:8000")

    resp = _get_sample(composer_test_client, session_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["allowed_hosts"] == ["127.0.0.1/32", "::1/128"]
    assert body["sample_urls"] == [
        "http://127.0.0.1:8000/tutorial-site/project-1.html",
        "http://127.0.0.1:8000/tutorial-site/project-2.html",
        "http://127.0.0.1:8000/tutorial-site/project-3.html",
    ]


def test_tutorial_sample_exposes_only_urls_and_allowed_hosts(composer_test_client: TestClient) -> None:
    """The tutorial-sample wire carries ONLY the runtime-derived inputs."""
    session_id = _create_session(composer_test_client)
    _start(composer_test_client, session_id, "tutorial")

    resp = _get_sample(composer_test_client, session_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert set(body.keys()) == {"sample_urls", "allowed_hosts"}


def test_tutorial_sample_rejects_non_tutorial_session(composer_test_client: TestClient) -> None:
    """A live (non-tutorial) guided session has no tutorial sample surface -> 400."""
    session_id = _create_session(composer_test_client)
    _start(composer_test_client, session_id, "live")

    resp = _get_sample(composer_test_client, session_id)
    assert resp.status_code == 400, resp.text
