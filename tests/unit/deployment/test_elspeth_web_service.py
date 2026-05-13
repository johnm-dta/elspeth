"""Deployment guardrails for the staging web systemd unit."""

from __future__ import annotations

from pathlib import Path


def test_elspeth_web_service_pins_repo_src_on_pythonpath() -> None:
    service_text = Path("deploy/elspeth-web.service").read_text(encoding="utf-8")

    assert "Environment=PYTHONPATH=/home/john/elspeth/src" in service_text
    assert service_text.index("Environment=PYTHONPATH=/home/john/elspeth/src") < service_text.index("ExecStart=")
