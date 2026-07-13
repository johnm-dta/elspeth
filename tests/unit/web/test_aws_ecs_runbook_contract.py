"""Executable contract for the Plan 14 CloudWatch runbook section."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

RUNBOOK = Path(__file__).parents[3] / "docs" / "runbooks" / "aws-ecs-deployment.md"


def _text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def _json_documents() -> list[object]:
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", _text(), flags=re.DOTALL)]


def _yaml_documents() -> list[object]:
    return [yaml.safe_load(block) for block in re.findall(r"```yaml\n(.*?)\n```", _text(), flags=re.DOTALL)]


def test_runbook_pins_task_local_nonessential_healthy_sidecar() -> None:
    task = next(document for document in _json_documents() if isinstance(document, dict) and "containerDefinitions" in document)
    containers = {container["name"]: container for container in task["containerDefinitions"]}
    sidecar = containers["cloudwatch-agent"]
    app = containers["elspeth-web"]

    assert sidecar["essential"] is False
    assert re.search(r"@sha256:\$\{CLOUDWATCH_AGENT_IMAGE_SHA256\}$", sidecar["image"])
    assert "portMappings" not in sidecar
    assert app["dependsOn"] == [{"containerName": "cloudwatch-agent", "condition": "HEALTHY"}]
    assert "127.0.0.1:4317" in _text()


def test_runbook_has_versioned_hashed_bounded_agent_config() -> None:
    text = _text()
    for required in (
        "elspeth.cloudwatch-agent.v1",
        "memory_limiter",
        "batch",
        "awscloudwatch",
        "awsxray",
        "sha256sum",
        "ELSPETH/Operator",
    ):
        assert required in text
    otel = _yaml_documents()[0]
    assert set(otel["exporters"]) == {"awscloudwatch", "awsxray"}


def test_runbook_separates_task_role_and_execution_role_without_credentials() -> None:
    text = _text()
    for action in (
        "cloudwatch:PutMetricData",
        "xray:PutTraceSegments",
        "xray:PutTelemetryRecords",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
    ):
        assert action in text
    assert "Execution role" in text
    assert "Task role" in text
    assert "Do not add keys, profiles, credential headers, role overrides" in text


def test_runbook_documents_dashboard_alarms_cardinality_cost_and_audit_authority() -> None:
    text = _text()
    for required in (
        "run failures",
        "run duration",
        "external-call failures",
        "external-call latency",
        "LLM token",
        "LLM cost",
        "export failure",
        "queue drop",
        "stale export",
        "missing sidecar signal",
        "Landscape",
        "lifecycle",
        "rows",
        "full",
        "retention",
    ):
        assert required in text
    manifest = next(
        document
        for document in _json_documents()
        if isinstance(document, dict) and document.get("schema") == "elspeth.cloudwatch-dimensions.v1"
    )
    dimensions = set(manifest["dimensions"])
    assert dimensions == {
        "service.name",
        "deployment.environment",
        "service.version",
        "task.definition.family",
        "task.definition.revision",
        "reason",
        "operation",
        "status",
        "outcome",
    }
    forbidden_dimensions = {
        "user_id",
        "session_id",
        "run_id",
        "row_id",
        "token_id",
        "task_arn",
        "account_id",
        "request_id",
        "prompt",
        "content",
        "url",
        "exception",
    }
    assert dimensions.isdisjoint(forbidden_dimensions)
