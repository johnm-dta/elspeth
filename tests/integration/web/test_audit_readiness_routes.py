"""Integration tests for /api/sessions/{sid}/audit-readiness routes.

The tests stand up the full ELSPETH web app via ``create_app`` so the
audit-readiness routes exercise the real ``validate_pipeline`` /
scoped-secret-resolver paths. Fixtures live in
``tests/integration/web/conftest.py``.

Two of the tests below — ``test_provenance_row_component_ids_populated_*``
and ``test_secrets_row_uses_scoped_resolver`` — guard real-system seams
that unit tests cannot cover:

* **C1 guard.** ``identity_node_advisory``'s ``affected_nodes`` field
  must propagate from ``execution/validation.py`` into the
  ``ReadinessService`` provenance row.
* **C4 guard.** ``ReadinessService`` must call ``list_refs`` through the
  scoped secret resolver (which has ``auth_provider_type`` baked in)
  rather than through the raw secret service. A misroute here would
  ``TypeError`` at runtime — visible only through the real wiring.

These were previously staged in
``tests/integration/web/audit_readiness/test_readiness_service_integration.py``
behind a ``pytest.mark.skip`` until Task 5 scaffolded the fixtures.
That staging file has been removed; the two tests live here, where the
plan §"Step 2b" routes them.
"""

from __future__ import annotations

import uuid
from uuid import UUID

from fastapi.testclient import TestClient

from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions.protocol import CompositionStateData


def test_snapshot_returns_six_canonical_rows(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    client, session_id = audit_readiness_client_with_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == str(session_id)
    assert body["composition_version"] >= 1
    assert body["checked_at"].endswith("Z") or body["checked_at"].endswith("+00:00")
    assert {row["id"] for row in body["rows"]} == {
        "validation",
        "plugin_trust",
        "provenance",
        "retention",
        "llm_interpretations",
        "secrets",
    }


def test_snapshot_404_when_no_state(
    audit_readiness_client_without_state: tuple[TestClient, UUID],
) -> None:
    client, session_id = audit_readiness_client_without_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 404


def test_snapshot_404_on_cross_user_access(
    audit_readiness_test_client: TestClient,
    audit_readiness_other_user_session_id: UUID,
) -> None:
    """IDOR guard: cross-user access returns 404, NOT 403.

    Returning 403 would disclose that the session exists. The
    ``verify_session_ownership`` helper canonicalizes both
    'session-not-found' and 'session-owned-by-other-user' to the same
    404 body so an attacker cannot probe arbitrary UUIDs.
    """
    response = audit_readiness_test_client.get(f"/api/sessions/{audit_readiness_other_user_session_id}/audit-readiness")
    assert response.status_code == 404


def test_snapshot_404_on_auth_provider_type_mismatch(
    audit_readiness_test_client: TestClient,
    audit_readiness_mismatched_provider_session_id: UUID,
) -> None:
    """IDOR guard: ``auth_provider_type`` mismatch returns 404, NOT 403.

    ``verify_session_ownership`` rejects when EITHER ``user_id`` OR
    ``auth_provider_type`` differs (sessions/ownership.py:49). Only the
    ``user_id`` branch had explicit coverage above; this test exercises
    the second comparator so a future refactor cannot collapse it (e.g.
    by dropping the right-hand-side of the ``or``) without a test
    failure. The session is owned by the authenticated user but bound
    to ``auth_provider_type="oidc"`` while ``settings.auth_provider``
    is ``"local"``, isolating the second comparator.
    """
    response = audit_readiness_test_client.get(f"/api/sessions/{audit_readiness_mismatched_provider_session_id}/audit-readiness")
    assert response.status_code == 404


def test_snapshot_requires_auth(
    audit_readiness_client_anonymous: TestClient,
) -> None:
    any_session_id = uuid.uuid4()
    response = audit_readiness_client_anonymous.get(f"/api/sessions/{any_session_id}/audit-readiness")
    assert response.status_code == 401


def test_explain_returns_narrative(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    client, session_id = audit_readiness_client_with_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness/explain")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == str(session_id)
    assert body["composition_version"] >= 1
    assert body["narrative"].startswith("When you run this pipeline, ELSPETH will record:")


def test_explain_404_when_no_state(
    audit_readiness_client_without_state: tuple[TestClient, UUID],
) -> None:
    client, session_id = audit_readiness_client_without_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness/explain")
    assert response.status_code == 404


def test_explain_requires_auth(
    audit_readiness_client_anonymous: TestClient,
) -> None:
    any_session_id = uuid.uuid4()
    response = audit_readiness_client_anonymous.get(f"/api/sessions/{any_session_id}/audit-readiness/explain")
    assert response.status_code == 401


def test_snapshot_includes_no_store_cache_header(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """The panel must reflect the current composition_version on every
    render; cache reuse across edits would silently desynchronize the
    UI from the underlying state."""
    client, session_id = audit_readiness_client_with_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-store"


def test_explain_includes_no_store_cache_header(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    client, session_id = audit_readiness_client_with_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness/explain")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-store"


def test_rejects_malformed_session_id(
    audit_readiness_test_client: TestClient,
) -> None:
    """FastAPI's ``UUID`` path-param parser must 422 on non-UUID input.

    The route's ownership/state logic never runs for malformed input,
    so this guards the boundary against bypassing the IDOR check via
    an invalid path segment.
    """
    response = audit_readiness_test_client.get("/api/sessions/not-a-uuid/audit-readiness")
    assert response.status_code == 422


# ── Real-system integration guards (C1 + C4) ───────────────────────────────


def test_provenance_row_component_ids_populated_via_real_validate_pipeline(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """C1 guard: affected_nodes wired in execution/validation.py must propagate to component_ids.

    If the _CHECK_IDENTITY_NODE_ADVISORY site in execution/validation.py
    (grep for ``name=_CHECK_IDENTITY_NODE_ADVISORY``) does not pass
    ``affected_nodes``, this assertion will fail even if the unit test
    passes (because the unit test supplies affected_nodes manually). The
    fixture's passthrough node triggers the advisory.
    """
    client, session_id = audit_readiness_client_with_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200, response.text
    rows = {r["id"]: r for r in response.json()["rows"]}
    assert rows["provenance"]["status"] == "warning", (
        "provenance row must be 'warning' — fixture includes a passthrough node to trigger the identity_node_advisory"
    )
    assert rows["provenance"]["component_ids"], (
        "provenance row status is 'warning' but component_ids is empty — "
        "execution/validation.py must pass affected_nodes=(node_id,) to "
        "ValidationCheck(name='identity_node_advisory')"
    )


def test_secrets_row_uses_scoped_resolver(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    """C4 guard: ReadinessService must call list_refs via scoped_secret_resolver.

    If wired to the raw ``secret_service`` (which requires
    ``auth_provider_type``), this will ``TypeError`` at runtime rather
    than in unit tests. The only assertion that matters is "no 500
    from a TypeError"; the body's ``status`` value is allowed to be
    any of the canonical Literal values for this composition shape.
    """
    client, session_id = audit_readiness_client_with_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200, response.text
    rows = {r["id"]: r for r in response.json()["rows"]}
    assert rows["secrets"]["status"] in ("ok", "error", "not_applicable")


def test_secrets_row_surfaces_disallowed_secret_ref_from_real_validate_pipeline(
    audit_readiness_test_client: TestClient,
) -> None:
    settings = audit_readiness_test_client.app.state.settings
    (settings.data_dir / "blobs").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "outputs").mkdir(parents=True, exist_ok=True)
    session_service = audit_readiness_test_client.app.state.session_service
    session_id = uuid.uuid4()

    async def _seed() -> None:
        record = await session_service.create_session(
            user_id="alice",
            title="audit-readiness disallowed secret-ref fixture",
            auth_provider_type=settings.auth_provider,
        )
        (settings.data_dir / "blobs" / str(record.id)).mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "outputs" / str(record.id)).mkdir(parents=True, exist_ok=True)
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="src_out",
                options={
                    "path": str(settings.data_dir / "blobs" / str(record.id) / "audit_readiness_fixture.csv"),
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="scrape",
                    node_type="transform",
                    plugin="web_scrape",
                    input="src_out",
                    on_success="out",
                    on_error="discard",
                    options={
                        "url_field": "url",
                        "http": {
                            "abuse_contact": {"secret_ref": "ANY_SECRET"},
                            "scraping_reason": "research",
                            # ``public_only`` is the sole allowed_hosts value web-authored
                            # web_scrape pipelines may use; a list/CIDR allowlist now trips
                            # the web_scrape_network_policy gate (14c2f4a73), which returns
                            # is_valid=False BEFORE the secret_refs check this test exercises.
                            "allowed_hosts": "public_only",
                        },
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(
                OutputSpec(
                    name="out",
                    plugin="csv",
                    options={
                        "path": str(settings.data_dir / "outputs" / str(record.id) / "audit_readiness_fixture_out.csv"),
                        "schema": {"mode": "observed"},
                    },
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(
                name="audit-readiness disallowed secret-ref fixture",
                description="Fixture used by audit-readiness route tests.",
            ),
            version=1,
        )
        state_d = state.to_dict()
        await session_service.save_composition_state(
            record.id,
            CompositionStateData(
                sources=state_d["sources"],
                nodes=state_d["nodes"],
                edges=state_d["edges"],
                outputs=state_d["outputs"],
                metadata_=state_d["metadata"],
                is_valid=True,
                validation_errors=None,
            ),
            provenance="session_seed",
        )
        nonlocal session_id
        session_id = record.id

    import asyncio

    asyncio.run(_seed())
    response = audit_readiness_test_client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200, response.text
    rows = {r["id"]: r for r in response.json()["rows"]}
    assert rows["secrets"]["status"] == "error"
    assert "credential-bearing fields" in (rows["secrets"]["detail"] or "")
    assert "scrape" in rows["secrets"]["component_ids"]


def test_audit_readiness_routes_are_rate_limited(
    audit_readiness_client_with_state: tuple[TestClient, UUID],
) -> None:
    client, session_id = audit_readiness_client_with_state

    for suffix in ("", "/explain"):
        client.app.state.rate_limiter = ComposerRateLimiter(limit=1)
        first_response = client.get(f"/api/sessions/{session_id}/audit-readiness{suffix}")
        assert first_response.status_code == 200
        response = client.get(f"/api/sessions/{session_id}/audit-readiness{suffix}")
        assert response.status_code == 429
