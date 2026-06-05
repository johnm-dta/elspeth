"""Deployment guardrails for the staging web systemd unit."""

from __future__ import annotations

from pathlib import Path


def _service_text() -> str:
    return Path("deploy/elspeth-web.service").read_text(encoding="utf-8")


def _active_service_text() -> str:
    return "\n".join(line for line in _service_text().splitlines() if not line.lstrip().startswith("#"))


def test_elspeth_web_service_pins_repo_src_on_pythonpath() -> None:
    service_text = _service_text()

    assert "Environment=PYTHONPATH=/home/john/elspeth/src" in service_text
    assert service_text.index("Environment=PYTHONPATH=/home/john/elspeth/src") < service_text.index("ExecStart=")


def test_elspeth_web_service_trusts_forwarded_headers_over_uds_boundary() -> None:
    service_text = _active_service_text()

    assert "--proxy-headers" in service_text
    assert "--forwarded-allow-ips='*'" in service_text
    assert "--forwarded-allow-ips=/run/elspeth/uvicorn.sock" not in service_text


def test_elspeth_web_service_omits_clean_exit_request_recycling() -> None:
    service_text = _active_service_text()

    assert "Restart=on-failure" in service_text
    assert "--limit-max-requests" not in service_text


def test_elspeth_web_service_documents_prefixed_secret_key() -> None:
    """The env-file guidance must name the variable create_app() actually reads."""
    service_text = _service_text()

    assert "ELSPETH_WEB__SECRET_KEY" in service_text
    assert "# SECRET_KEY:" not in service_text


def test_elspeth_web_service_marks_uds_reverse_proxy_as_non_local() -> None:
    """UDS startup must trip WebSettings' production secret-key guard (host not loopback)."""
    service_text = _active_service_text()

    assert "--uds /run/elspeth/uvicorn.sock" in service_text
    assert "Environment=ELSPETH_WEB__HOST=0.0.0.0" in service_text
    assert service_text.index("Environment=ELSPETH_WEB__HOST=0.0.0.0") < service_text.index("ExecStart=")
