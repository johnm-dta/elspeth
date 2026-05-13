"""Route-boundary exception handlers for compose-loop persistence failures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from elspeth.contracts.errors import AuditIntegrityError, FailedTurnMetadata
from elspeth.web.app import create_app
from elspeth.web.config import WebSettings
from elspeth.web.sessions.protocol import StaleComposeStateError


def _settings(tmp_path: Path) -> WebSettings:
    return WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
    )


@pytest.mark.asyncio
async def test_audit_integrity_error_handler_returns_static_500(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    handler = app.exception_handlers[AuditIntegrityError]

    response = await handler(
        cast(Request, object()),
        AuditIntegrityError(
            "hidden sql detail",
            failed_turn=FailedTurnMetadata(
                assistant_message_id=None,
                tool_calls_attempted=2,
                tool_responses_persisted=0,
            ),
        ),
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error_type"] == "audit_integrity_error"
    assert body["failed_turn"] == {
        "assistant_message_id": None,
        "tool_calls_attempted": 2,
        "tool_responses_persisted": 0,
        "transcript_url": None,
    }
    assert "hidden sql detail" not in response.body.decode()


def test_create_app_wires_existing_session_service_into_composer(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    assert app.state.composer_service._sessions_service is app.state.session_service


@pytest.mark.asyncio
async def test_audit_integrity_error_handler_returns_degraded_body_without_failed_turn(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    handler = app.exception_handlers[AuditIntegrityError]

    response = await handler(cast(Request, object()), AuditIntegrityError("outside compose loop"))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error_type"] == "audit_integrity_error"
    assert body["diagnostic"] == "no_failed_turn_metadata"
    assert body["reason"] == "originated outside compose-loop annotation scope"
    assert "failed_turn" not in body


@pytest.mark.asyncio
async def test_stale_compose_state_error_handler_returns_409(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    handler = app.exception_handlers[StaleComposeStateError]

    response = await handler(cast(Request, object()), StaleComposeStateError("stale"))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 409
    assert json.loads(response.body)["error_type"] == "stale_compose_state"
