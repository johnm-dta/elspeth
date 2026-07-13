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
    environment = {entry["name"]: entry["value"] for entry in app["environment"]}
    assert environment == {
        "ELSPETH_WEB__OPERATOR_TELEMETRY": "aws-otlp",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_ENVIRONMENT": "production",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_RELEASE": "${ELSPETH_RELEASE_SHA_OR_DIGEST}",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_ECS_CLUSTER": "${ECS_CLUSTER_NAME}",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_ECS_SERVICE": "${ECS_SERVICE_NAME}",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_TASK_DEFINITION_FAMILY": "${ECS_TASK_DEFINITION_FAMILY}",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_TASK_DEFINITION_REVISION": "${ECS_TASK_DEFINITION_REVISION}",
    }
    assert "127.0.0.1:4317" in _text()


def test_runbook_wires_versioned_config_into_executable_sidecar_startup() -> None:
    task = next(document for document in _json_documents() if isinstance(document, dict) and "containerDefinitions" in document)
    sidecar = next(container for container in task["containerDefinitions"] if container["name"] == "cloudwatch-agent")

    assert sidecar["entryPoint"] == ["/bin/sh", "-ceu"]
    environment = {entry["name"]: entry["value"] for entry in sidecar["environment"]}
    assert environment == {
        "ELSPETH_CW_AGENT_CONFIG_JSON_B64": "${CLOUDWATCH_AGENT_CONFIG_JSON_B64}",
        "ELSPETH_CW_AGENT_CONFIG_JSON_SHA256": "${CLOUDWATCH_AGENT_CONFIG_JSON_SHA256}",
        "ELSPETH_CW_AGENT_OTEL_YAML_B64": "${CLOUDWATCH_AGENT_OTEL_YAML_B64}",
        "ELSPETH_CW_AGENT_OTEL_YAML_SHA256": "${CLOUDWATCH_AGENT_OTEL_YAML_SHA256}",
    }
    assert len(sidecar["command"]) == 1
    script = sidecar["command"][0]
    json_path = "/tmp/elspeth-cloudwatch-agent/elspeth.cloudwatch-agent.v1.json"
    otel_path = "/tmp/elspeth-cloudwatch-agent/elspeth.cloudwatch-agent.v1.otel.yaml"
    assert f'base64 -d > "{json_path}"' in script
    assert f'base64 -d > "{otel_path}"' in script
    json_verify = f'"$ELSPETH_CW_AGENT_CONFIG_JSON_SHA256  {json_path}" | sha256sum -c -'
    otel_verify = f'"$ELSPETH_CW_AGENT_OTEL_YAML_SHA256  {otel_path}" | sha256sum -c -'
    fetch = f'-a fetch-config -m auto -c "file:{json_path}" -s'
    append = f'-a append-config -m auto -c "file:{otel_path}" -s'
    assert json_verify in script
    assert otel_verify in script
    assert fetch in script
    assert append in script
    assert script.index(json_verify) < script.index(fetch)
    assert script.index(otel_verify) < script.index(append)
    assert script.index(fetch) < script.index(append)


def test_runbook_uses_supported_auto_mode_for_running_agent_health() -> None:
    task = next(document for document in _json_documents() if isinstance(document, dict) and "containerDefinitions" in document)
    sidecar = next(container for container in task["containerDefinitions"] if container["name"] == "cloudwatch-agent")

    assert sidecar["healthCheck"]["command"] == [
        "CMD-SHELL",
        '/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a status -m auto | grep -q \'"status": "running"\'',
    ]
    assert "-m ecs" not in json.dumps(sidecar)


def test_runbook_has_versioned_hashed_bounded_agent_config() -> None:
    text = _text()
    for required in (
        "elspeth.cloudwatch-agent.v1",
        "memory_limiter/elspeth",
        "batch/elspeth",
        "awsemf/elspeth",
        "awsxray/elspeth",
        "sha256sum",
        "ELSPETH/Operator",
    ):
        assert required in text
    otel = _yaml_documents()[0]
    assert set(otel["receivers"]) == {"otlp/elspeth"}
    assert otel["receivers"]["otlp/elspeth"] == {"protocols": {"grpc": {"endpoint": "127.0.0.1:4317"}}}
    assert set(otel["processors"]) == {"memory_limiter/elspeth", "batch/elspeth"}
    assert set(otel["exporters"]) == {"awsemf/elspeth", "awsxray/elspeth"}
    assert otel["exporters"]["awsemf/elspeth"] == {
        "namespace": "ELSPETH/Operator",
        "log_group_name": "/elspeth/operator/metrics",
        "log_stream_name": "telemetry",
        "dimension_rollup_option": "NoDimensionRollup",
        "retain_initial_value_of_delta_metric": True,
        "resource_to_telemetry_conversion": {"enabled": True},
    }
    assert otel["exporters"]["awsxray/elspeth"] == {}
    assert otel["service"]["pipelines"] == {
        "metrics/elspeth": {
            "receivers": ["otlp/elspeth"],
            "processors": ["memory_limiter/elspeth", "batch/elspeth"],
            "exporters": ["awsemf/elspeth"],
        },
        "traces/elspeth": {
            "receivers": ["otlp/elspeth"],
            "processors": ["memory_limiter/elspeth", "batch/elspeth"],
            "exporters": ["awsxray/elspeth"],
        },
    }
    rendered = json.dumps(otel)
    assert '"awscloudwatch"' not in rendered
    assert '"debug' not in rendered
    assert '"file' not in rendered
    assert "role_arn" not in rendered
    assert "endpoint_override" not in rendered


def test_runbook_separates_task_role_and_execution_role_without_credentials() -> None:
    text = _text()
    policy = next(
        document
        for document in _json_documents()
        if isinstance(document, dict) and document.get("Version") == "2012-10-17" and "Statement" in document
    )
    actions = {
        action
        for statement in policy["Statement"]
        for action in ([statement["Action"]] if isinstance(statement["Action"], str) else statement["Action"])
    }
    assert {
        "xray:PutTraceSegments",
        "xray:PutTelemetryRecords",
        "logs:DescribeLogStreams",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
    } <= actions
    assert "cloudwatch:PutMetricData" not in actions
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
        "cloud.provider",
        "aws.ecs.cluster.name",
        "aws.ecs.service.name",
        "aws.ecs.task.family",
        "aws.ecs.task.revision",
        "cap_type",
        "completion_path",
        "completion_verb",
        "component_type",
        "failure_class",
        "from_mode",
        "kind",
        "reason",
        "operation",
        "probe_status",
        "result",
        "source",
        "status",
        "outcome",
        "to_mode",
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
