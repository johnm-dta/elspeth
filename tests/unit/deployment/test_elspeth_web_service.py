"""Deployment guardrails for the staging web systemd unit."""

from __future__ import annotations

from pathlib import Path


_SERVICE_PATH = Path("deploy/elspeth-web.service")


def _service_text() -> str:
    return _SERVICE_PATH.read_text(encoding="utf-8")


def test_elspeth_web_service_pins_repo_src_on_pythonpath() -> None:
    service_text = _service_text()

    assert "Environment=PYTHONPATH=/home/john/elspeth/src" in service_text
    assert service_text.index("Environment=PYTHONPATH=/home/john/elspeth/src") < service_text.index("ExecStart=")


def test_elspeth_web_service_documents_prefixed_secret_key() -> None:
    """The env file guidance must name the variable create_app() actually reads."""
    service_text = _service_text()

    assert "ELSPETH_WEB__SECRET_KEY" in service_text
    assert "# SECRET_KEY:" not in service_text


def test_elspeth_web_service_marks_uds_reverse_proxy_as_non_local() -> None:
    """UDS startup must trip WebSettings' production secret-key guard."""
    service_text = _service_text()

    assert "--uds /run/elspeth/uvicorn.sock" in service_text
    assert "Environment=ELSPETH_WEB__HOST=0.0.0.0" in service_text
    assert service_text.index("Environment=ELSPETH_WEB__HOST=0.0.0.0") < service_text.index("ExecStart=")
