"""End-to-end integration test: CSV -> passthrough -> CSV through web layer.

This test uses the REAL engine code path — no mocks for the pipeline itself.
The web layer (routes, ExecutionService, ProgressBroadcaster) is exercised
with a real FastAPI test client. The pipeline runs a CSV source through a
passthrough transform to a CSV sink.

Test strategy (confirmed by panel review):
- REST for session creation (exercises auth + IDOR + route)
- Programmatic CompositionState construction + save via session service
  (bypasses LLM composer — we're testing execution, not composition)
- REST for execute/poll/results (exercises the Sub-5 path end-to-end)
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_CSV = FIXTURES_DIR / "test_input.csv"

# elspeth-5069612f3c — gate-routed reproducer CSV with the ``tier`` column.
GATE_ROUTED_CSV_CONTENT = "id,tier\n1,high\n2,low\n3,high\n4,low\n"


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Create a working directory with the test CSV and output dir."""
    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir()
    csv_dest = blobs_dir / "input.csv"
    shutil.copy(TEST_CSV, csv_dest)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    audit_dir = tmp_path / "runs"
    audit_dir.mkdir()
    payloads_dir = tmp_path / "payloads"
    payloads_dir.mkdir(mode=0o700)
    return tmp_path


@pytest.mark.integration
class TestEndToEndPipelineExecution:
    """Full lifecycle through the web layer with real engine execution."""

    @pytest.mark.asyncio
    async def test_csv_passthrough_csv(self, work_dir: Path) -> None:
        """
        1. Register + login test user
        2. Create session via REST
        3. Save CompositionState programmatically (bypass LLM composer)
        4. Execute via REST -> get run_id
        5. Poll status -> eventually 'completed'
        6. Verify results: accounting.source.rows_processed > 0 and accounting.tokens.failed == 0
        7. Verify landscape_run_id links to audit trail
        """
        from elspeth.web.app import create_app
        from elspeth.web.composer.state import (
            CompositionState,
            OutputSpec,
            PipelineMetadata,
            SourceSpec,
        )
        from elspeth.web.config import WebSettings
        from elspeth.web.sessions.protocol import CompositionStateData

        settings = WebSettings(
            data_dir=work_dir,
            landscape_url=f"sqlite:///{work_dir}/runs/audit.db",
            payload_store_path=work_dir / "payloads",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        app = create_app(settings=settings)

        # Create test user via the auth provider directly (no /register endpoint)
        auth_provider = app.state.auth_provider
        auth_provider.create_user("testuser", "testpass123", display_name="Test User")

        from asgi_lifespan import LifespanManager

        async with LifespanManager(app), AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Authenticate via /login endpoint
            resp = await client.post(
                "/api/auth/login",
                json={"username": "testuser", "password": "testpass123"},
            )
            assert resp.status_code == 200, f"Login failed: {resp.text}"
            token = resp.json()["access_token"]
            auth_headers = {"Authorization": f"Bearer {token}"}

            # 1. Create session via REST
            resp = await client.post(
                "/api/sessions",
                headers=auth_headers,
                json={"title": "Integration test session"},
            )
            assert resp.status_code == 201, f"Session creation failed: {resp.text}"
            session_id = resp.json()["id"]  # SessionResponse.id, NOT session_id

            # 2. Save composition state programmatically
            csv_path = str(work_dir / "blobs" / "input.csv")
            output_path = str(work_dir / "outputs" / "result.csv")

            state = CompositionState(
                source=SourceSpec(
                    plugin="csv",
                    on_success="primary",
                    options={
                        "path": csv_path,
                        "schema": {
                            "mode": "fixed",
                            "fields": ["id: int", "name: str", "value: int"],
                        },
                    },
                    on_validation_failure="discard",
                ),
                nodes=(),
                edges=(),
                outputs=(
                    OutputSpec(
                        name="primary",
                        plugin="csv",
                        options={
                            "path": output_path,
                            "schema": {
                                "mode": "fixed",
                                "fields": ["id: int", "name: str", "value: int"],
                            },
                        },
                        on_write_failure="discard",
                    ),
                ),
                metadata=PipelineMetadata(
                    name="Integration Test Pipeline",
                    description="CSV passthrough for Sub-5 integration test",
                ),
                version=1,
            )

            # Convert to CompositionStateData for saving
            state_d = state.to_dict()
            state_data = CompositionStateData(
                sources=state_d["sources"],
                nodes=state_d["nodes"],
                edges=state_d["edges"],
                outputs=state_d["outputs"],
                metadata_=state_d["metadata"],  # metadata_ (underscore)
                is_valid=True,
                validation_errors=None,
            )
            session_service = app.state.session_service
            await session_service.save_composition_state(UUID(session_id), state_data, provenance="session_seed")

            # 3. Execute via REST
            resp = await client.post(
                f"/api/sessions/{session_id}/execute",
                headers=auth_headers,
            )
            assert resp.status_code == 202, f"Execute failed: {resp.text}"
            run_id = resp.json()["run_id"]

            # 4. Poll to completion (timeout after 30s)
            deadline = time.monotonic() + 30
            status: dict[str, Any] = {}
            while time.monotonic() < deadline:
                resp = await client.get(
                    f"/api/runs/{run_id}",
                    headers=auth_headers,
                )
                assert resp.status_code == 200
                status = resp.json()
                if status["status"] in ("completed", "failed", "cancelled"):
                    break
                await asyncio.sleep(0.5)
            else:
                pytest.fail("Pipeline did not complete within 30 seconds")

            # 5. Verify results
            assert status["status"] == "completed", f"Pipeline failed: {status.get('error')}"
            accounting = status["accounting"]
            assert accounting is not None
            assert accounting["source"]["rows_processed"] > 0
            assert accounting["tokens"]["failed"] == 0

            # 6. Verify landscape_run_id links to real audit trail
            assert status["landscape_run_id"] is not None

            # 7. Verify output file was created
            output_file = work_dir / "outputs" / "result.csv"
            assert output_file.exists()

            # 8. Verify audit database exists
            audit_db = work_dir / "runs" / "audit.db"
            assert audit_db.exists()

            # 9. Verify results endpoint
            resp = await client.get(
                f"/api/runs/{run_id}/results",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            results = resp.json()
            assert results["accounting"]["source"]["rows_processed"] > 0
            assert results["landscape_run_id"] is not None


@pytest.mark.integration
class TestGateRoutedPipelineExecution:
    """elspeth-5069612f3c / elspeth-71520f5e30 — API half of the VAL gate
    for the gate-only-pipeline misclassification bug.

    Mirrors the engine-layer test in test_composer_runtime_agreement.py but
    exercises the FastAPI surface the user actually reported the bug
    against. Without this test, the L3 row-count predicate mirror
    (_check_status_row_count_invariant) AND the sessions read-side Tier-1
    status guards (RunRecord.__post_init__ — NOT a row-count mirror, but
    enforces status enum / finished_at / landscape_run_id / error invariants)
    are not exercised end-to-end and the user's reported reproducer goes
    unverified.
    """

    @pytest.mark.asyncio
    async def test_gate_routed_pipeline_classifies_as_completed_via_api(
        self,
        work_dir: Path,
    ) -> None:
        """Gate-only pipeline (every row terminally gate-routed; no on_success
        success-path sink) must surface as RunStatus.COMPLETED on
        /api/runs/{run_id}, NOT FAILED with the misleading
        "No row reached the success path" structural-failure message.

        Pre-PR (commit cc895589): /api/runs/{run_id} returns status='failed'
        with the synthetic structural error. Post-PR: returns
        status='completed' with routed successes represented as both lifecycle
        token successes and the accounting.routing.routed_success subset counter.
        """
        from elspeth.web.app import create_app
        from elspeth.web.composer.state import (
            CompositionState,
            EdgeSpec,
            NodeSpec,
            OutputSpec,
            PipelineMetadata,
            SourceSpec,
        )
        from elspeth.web.config import WebSettings
        from elspeth.web.sessions.protocol import CompositionStateData

        # Overwrite the input.csv prepared by ``work_dir`` with the
        # gate-routed reproducer schema (id, tier).
        gate_csv = work_dir / "blobs" / "input.csv"
        gate_csv.write_text(GATE_ROUTED_CSV_CONTENT)

        settings = WebSettings(
            data_dir=work_dir,
            landscape_url=f"sqlite:///{work_dir}/runs/audit.db",
            payload_store_path=work_dir / "payloads",
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        app = create_app(settings=settings)

        auth_provider = app.state.auth_provider
        auth_provider.create_user("gateuser", "gatepass123", display_name="Gate User")

        from asgi_lifespan import LifespanManager

        async with LifespanManager(app), AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/login",
                json={"username": "gateuser", "password": "gatepass123"},
            )
            assert resp.status_code == 200, f"Login failed: {resp.text}"
            token = resp.json()["access_token"]
            auth_headers = {"Authorization": f"Bearer {token}"}

            resp = await client.post(
                "/api/sessions",
                headers=auth_headers,
                json={"title": "Gate-routed reproducer session"},
            )
            assert resp.status_code == 201, f"Session creation failed: {resp.text}"
            session_id = resp.json()["id"]

            csv_path = str(gate_csv)
            high_path = str(work_dir / "outputs" / "high.csv")
            low_path = str(work_dir / "outputs" / "low.csv")
            row_schema = {
                "mode": "fixed",
                "fields": ["id: int", "tier: str"],
            }

            state = CompositionState(
                source=SourceSpec(
                    plugin="csv",
                    on_success="csv_in_out",
                    options={"path": csv_path, "schema": row_schema},
                    on_validation_failure="discard",
                ),
                nodes=(
                    NodeSpec(
                        id="tier_gate",
                        node_type="gate",
                        plugin=None,
                        input="csv_in_out",
                        on_success=None,
                        on_error=None,
                        options={},
                        condition="row['tier'] == 'high'",
                        routes={"true": "high_priority", "false": "low_priority"},
                        fork_to=None,
                        branches=None,
                        policy=None,
                        merge=None,
                    ),
                ),
                edges=(
                    EdgeSpec(
                        id="e_source_gate",
                        from_node="source",
                        to_node="tier_gate",
                        edge_type="on_success",
                        label=None,
                    ),
                    EdgeSpec(
                        id="e_gate_high",
                        from_node="tier_gate",
                        to_node="high_priority",
                        edge_type="route_true",
                        label="true",
                    ),
                    EdgeSpec(
                        id="e_gate_low",
                        from_node="tier_gate",
                        to_node="low_priority",
                        edge_type="route_false",
                        label="false",
                    ),
                ),
                outputs=(
                    OutputSpec(
                        name="high_priority",
                        plugin="csv",
                        options={"path": high_path, "schema": row_schema},
                        on_write_failure="discard",
                    ),
                    OutputSpec(
                        name="low_priority",
                        plugin="csv",
                        options={"path": low_path, "schema": row_schema},
                        on_write_failure="discard",
                    ),
                ),
                metadata=PipelineMetadata(
                    name="Gate-routed reproducer",
                    description="elspeth-71520f5e30 reproducer",
                ),
                version=1,
            )

            state_d = state.to_dict()
            state_data = CompositionStateData(
                sources=state_d["sources"],
                nodes=state_d["nodes"],
                edges=state_d["edges"],
                outputs=state_d["outputs"],
                metadata_=state_d["metadata"],
                is_valid=True,
                validation_errors=None,
            )
            session_service = app.state.session_service
            await session_service.save_composition_state(UUID(session_id), state_data, provenance="session_seed")

            resp = await client.post(
                f"/api/sessions/{session_id}/execute",
                headers=auth_headers,
            )
            assert resp.status_code == 202, f"Execute failed: {resp.text}"
            run_id = resp.json()["run_id"]

            deadline = time.monotonic() + 30
            status: dict[str, Any] = {}
            while time.monotonic() < deadline:
                resp = await client.get(
                    f"/api/runs/{run_id}",
                    headers=auth_headers,
                )
                assert resp.status_code == 200
                status = resp.json()
                if status["status"] in (
                    "completed",
                    "completed_with_failures",
                    "failed",
                    "cancelled",
                    "empty",
                ):
                    break
                await asyncio.sleep(0.5)
            else:
                pytest.fail("Pipeline did not complete within 30 seconds")

            # The bug: pre-PR returns 'failed'; post-PR must return
            # 'completed'.
            assert status["status"] == "completed", (
                "Gate-routed pipeline misclassified — expected 'completed', "
                f"got {status['status']!r}; error={status.get('error')!r}. "
                "Pre-fix structural-failure message would be 'No row reached "
                "the success path' or similar — the routed accounting split is "
                "supposed to fix this end-to-end."
            )
            accounting = status["accounting"]
            assert accounting is not None
            assert accounting["source"]["rows_processed"] > 0
            assert accounting["tokens"]["succeeded"] >= accounting["routing"]["routed_success"]
            assert accounting["routing"]["routed_success"] > 0
            assert accounting["routing"]["routed_failure"] == 0
            assert status["error"] is None
