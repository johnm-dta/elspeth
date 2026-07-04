"""Step-1 source schema prefill integration tests."""

from __future__ import annotations

from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _create_session(client: TestClient, title: str) -> str:
    resp = client.post("/api/sessions", json={"title": title})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _seed_csv_blob(client: TestClient, session_id: str) -> str:
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={
            "filename": "data.csv",
            "content": "name,age\nAda,37\n",
            "mime_type": "text/csv",
        },
    )
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _single_select_response(plugin: str) -> dict[str, object]:
    return {
        "chosen": [plugin],
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": None,
    }


def _get_guided(client: TestClient, session_id: str) -> dict:
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _select_source(client: TestClient, session_id: str, plugin: str) -> dict:
    resp = client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json=_single_select_response(plugin),
    )
    assert resp.status_code == 200, resp.json()
    return resp.json()


def test_step1_prefilled_from_uploaded_blob_inspection_facts(composer_test_client: TestClient) -> None:
    sess = _create_session(composer_test_client, "step1-prefill")
    _seed_csv_blob(composer_test_client, sess)

    initial = _get_guided(composer_test_client, sess)
    assert initial["next_turn"]["type"] == "single_select"

    selected = _select_source(composer_test_client, sess, "csv")

    assert selected["next_turn"]["type"] == "schema_form"
    schema_prefill = selected["next_turn"]["payload"]["prefilled"]["schema"]
    assert schema_prefill["mode"] == "flexible"
    assert "name: str" in schema_prefill["fields"]
    assert "age: int" in schema_prefill["fields"]


def test_step1_prefill_survives_guided_get_rebuild(composer_test_client: TestClient) -> None:
    sess = _create_session(composer_test_client, "step1-prefill-rebuild")
    _seed_csv_blob(composer_test_client, sess)

    _get_guided(composer_test_client, sess)
    selected = _select_source(composer_test_client, sess, "csv")
    schema_prefill = selected["next_turn"]["payload"]["prefilled"]["schema"]
    assert schema_prefill["fields"] == ["name: str", "age: int"]

    rebuilt = _get_guided(composer_test_client, sess)

    assert rebuilt["next_turn"]["type"] == "schema_form"
    rebuilt_prefill = rebuilt["next_turn"]["payload"]["prefilled"]["schema"]
    assert rebuilt_prefill["fields"] == ["name: str", "age: int"]


def test_step1_prefill_stays_observed_for_env_var_header(composer_test_client: TestClient) -> None:
    # A Tier-3 uploaded CSV whose header is a ${VAR} placeholder must not be
    # promoted into explicit schema.fields specs: those strings later flow through
    # the runtime YAML loader, where ${VAR} would resolve host env on the CLI path.
    sess = _create_session(composer_test_client, "step1-prefill-env-header")
    resp = composer_test_client.post(
        f"/api/sessions/{sess}/blobs/inline",
        json={
            "filename": "secrets.csv",
            "content": "${AWS_SECRET_ACCESS_KEY},ok\nsecret,1\n",
            "mime_type": "text/csv",
        },
    )
    assert resp.status_code == 201, resp.json()

    initial = _get_guided(composer_test_client, sess)
    assert initial["next_turn"]["type"] == "single_select"

    selected = _select_source(composer_test_client, sess, "csv")

    assert selected["next_turn"]["type"] == "schema_form"
    schema_prefill = selected["next_turn"]["payload"]["prefilled"]["schema"]
    assert schema_prefill == {"mode": "observed"}
