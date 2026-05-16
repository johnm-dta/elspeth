"""C1/C4 guard tests — staged for Task 5 to wire in.

These tests are the Step 2b deliverable from Phase 2A Task 3 (plan §lines
1274-1333). They guard two real-system seams that unit tests cannot cover:

  - C1: identity_node_advisory's affected_nodes propagating from
    execution/validation.py into ReadinessService's provenance row.
  - C4: ReadinessService calling list_refs through the scoped secret
    resolver (which has auth_provider_type baked in) rather than the raw
    secret_service (which requires auth_provider_type as a kwarg).

Task 5 owns the integration-fixture scaffolding
(``audit_readiness_client_with_state``) and the long-term home for these
tests in ``tests/integration/web/test_audit_readiness_routes.py``. Until
those fixtures land, the tests are pytest.skip-ped so the suite stays
green; Task 5 removes the skip markers when the fixtures are scaffolded.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="Task 5 scaffolds the audit_readiness_client_with_state fixture",
)


def test_provenance_row_component_ids_populated_via_real_validate_pipeline(
    audit_readiness_client_with_state,
):
    """C1 guard: affected_nodes wired in execution/validation.py must propagate to component_ids.

    If execution/validation.py:1248 ValidationCheck(name='identity_node_advisory', ...)
    does not pass affected_nodes, this assertion will fail even if the unit
    test passes (because the unit test supplies affected_nodes manually).
    """
    client, session_id = audit_readiness_client_with_state
    # The session fixture must include a passthrough node (triggers the advisory).
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200
    rows = {r["id"]: r for r in response.json()["rows"]}
    # Fixture's passthrough node must trigger the advisory; component_ids must be populated.
    assert rows["provenance"]["status"] == "warning", (
        "provenance row must be 'warning' — fixture must include a passthrough node to trigger the identity_node_advisory"
    )
    assert rows["provenance"]["component_ids"], (
        "provenance row status is 'warning' but component_ids is empty — "
        "execution/validation.py:1248 must pass affected_nodes=(node_id,) to ValidationCheck"
    )


def test_secrets_row_uses_scoped_resolver(audit_readiness_client_with_state):
    """C4 guard: ReadinessService must call list_refs via scoped_secret_resolver.

    If wired to the raw secret_service (which requires auth_provider_type),
    this will TypeError at runtime rather than in unit tests.
    """
    client, session_id = audit_readiness_client_with_state
    # A session whose composition references a secret ref exercises the resolver.
    # Wire the secret ref into the session fixture's CompositionState, or add
    # a separate fixture variant with a secret ref pre-populated.
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200
    rows = {r["id"]: r for r in response.json()["rows"]}
    # A session with a secret ref that resolves → ok; unresolved → error.
    # Either is acceptable; what must NOT happen is a 500 from a TypeError.
    assert rows["secrets"]["status"] in ("ok", "error", "not_applicable")
