"""Executable contract for the AWS ECS deployment runbook."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "aws-ecs-deployment.md"
RUNBOOK_INDEX = REPO_ROOT / "docs" / "runbooks" / "index.md"
DOCKER_GUIDE = REPO_ROOT / "docs" / "guides" / "docker.md"


def _text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def _fences(language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", _text(), flags=re.DOTALL)


def _json_documents() -> list[object]:
    return [json.loads(block) for block in _fences("json")]


def _yaml_documents() -> list[object]:
    return [yaml.safe_load(block) for block in _fences("yaml")]


def test_runbook_preserves_task_local_nonessential_healthy_sidecar() -> None:
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


def test_runbook_preserves_versioned_config_sidecar_startup() -> None:
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
    json_verify = f'"$ELSPETH_CW_AGENT_CONFIG_JSON_SHA256  {json_path}" | sha256sum -c -'
    otel_verify = f'"$ELSPETH_CW_AGENT_OTEL_YAML_SHA256  {otel_path}" | sha256sum -c -'
    fetch = f'-a fetch-config -m auto -c "file:{json_path}" -s'
    append = f'-a append-config -m auto -c "file:{otel_path}" -s'
    assert f'base64 -d > "{json_path}"' in script
    assert f'base64 -d > "{otel_path}"' in script
    assert json_verify in script
    assert otel_verify in script
    assert fetch in script
    assert append in script
    assert script.index(json_verify) < script.index(fetch)
    assert script.index(otel_verify) < script.index(append)
    assert script.index(fetch) < script.index(append)


def test_runbook_preserves_supported_agent_health_mode() -> None:
    task = next(document for document in _json_documents() if isinstance(document, dict) and "containerDefinitions" in document)
    sidecar = next(container for container in task["containerDefinitions"] if container["name"] == "cloudwatch-agent")

    assert sidecar["healthCheck"]["command"] == [
        "CMD-SHELL",
        '/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a status -m auto | grep -q \'"status": "running"\'',
    ]
    assert "-m ecs" not in json.dumps(sidecar)


def test_runbook_preserves_versioned_hashed_bounded_agent_config() -> None:
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

    otel = next(document for document in _yaml_documents() if isinstance(document, dict) and "receivers" in document)
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


def test_runbook_preserves_role_separation_without_credentials() -> None:
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


def test_runbook_preserves_dashboard_cardinality_cost_and_audit_authority() -> None:
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
    assert dimensions.isdisjoint(
        {
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
    )


def test_every_bash_fence_is_syntactically_valid() -> None:
    fences = _fences("bash")
    assert len(fences) >= 8

    for index, script in enumerate(fences, start=1):
        result = subprocess.run(
            ["bash", "-n"],
            input=script,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"bash fence {index}: {result.stderr}"


def test_every_bash_fence_routes_aws_calls_through_protected_capture() -> None:
    for script in _fences("bash"):
        for line in script.splitlines():
            assert not line.lstrip().startswith("aws "), line


def test_container_health_command_is_parseable_json_and_liveness_only() -> None:
    documents: list[Any] = []
    for fence in _fences("json"):
        documents.append(json.loads(fence))

    health_checks = [
        document["healthCheck"] for document in documents if isinstance(document, dict) and isinstance(document.get("healthCheck"), dict)
    ]
    assert len(health_checks) == 1
    command = health_checks[0]["command"]
    assert command[:2] == ["CMD", "python"]
    assert "http.client.HTTPConnection('127.0.0.1',8451,timeout=5)" in command[3]
    assert "r.status == 200" in command[3]
    assert "/api/health" in command[3]
    assert "/api/ready" not in command[3]
    assert "elspeth health" not in command[3]


def test_runbook_pins_core_runtime_and_identity_contracts() -> None:
    text = _text()
    required = (
        "ELSPETH_WEB__DEPLOYMENT_TARGET=aws-ecs",
        "AWS Secrets Manager",
        "Azure equivalent",
        "PostgreSQL",
        "session_db_url",
        "landscape_url",
        "payload_store_path",
        "Cognito/OIDC",
        "auth.db",
        "DELETE",
        "ELSPETH_WEB__OIDC_AUTHORIZATION_ALLOWED_ORIGINS",
        "ELSPETH_WEB__OIDC_TOKEN_ENDPOINT",
        "ELSPETH_WEB__OIDC_AUDIENCE_CLAIM",
        "token_use=access",
        "AllowedOAuthFlowsUserPoolClient",
        "AllowedOAuthFlows",
        "AllowedOAuthScopes",
        "CallbackURLs",
        'INSTALL_EXTRAS="webui llm aws postgres"',
        "TARGET_PLATFORM",
        "runtimePlatform",
        "linux/amd64",
        "X86_64",
        "linux/arm64",
        "ARM64",
        "healthCheckGracePeriodSeconds",
        "startPeriod",
        "minimum ACU",
    )
    for marker in required:
        assert marker in text, marker


def test_runbook_pins_guardrail_telemetry_and_task_role_contracts() -> None:
    text = _text()
    required = (
        "bedrock:InvokeModel",
        "bedrock:ApplyGuardrail",
        "aws_bedrock_prompt_shield",
        "aws_bedrock_content_safety",
        "DRAFT",
        "CloudWatch Agent",
        "127.0.0.1:4317",
        "RunStarted",
        "RunFinished",
        "Landscape-first",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "ELSPETH_ACCEPTANCE_S3_PREFIX",
        "taskRoleArn",
        "executionRoleArn",
        "ssmmessages:CreateControlChannel",
        "ssmmessages:CreateDataChannel",
        "ssmmessages:OpenControlChannel",
        "ssmmessages:OpenDataChannel",
    )
    for marker in required:
        assert marker in text, marker


def test_runbook_pins_ordered_rollout_rollback_and_schema_authority() -> None:
    text = _text()
    required = (
        "DEPLOYMENT_MODE",
        "first-recovery",
        "FIRST_DEPLOY_LISTENER_RULE_ARN",
        "FIRST_DEPLOY_DISABLED_ACTIONS",
        "DOCTOR_TASK_DEFINITION",
        "DOCTOR_CONTAINER_NAME",
        "DOCTOR_NETWORK_CONFIGURATION",
        "DOCTOR_OVERRIDES",
        "DOCTOR_TASK_ARN",
        "OBSERVATION_START_EPOCH_MS",
        "ACCEPTANCE_START_UTC",
        "ECS_DEPLOYMENT_EVENT_RULE",
        "ECS_DEPLOYMENT_EVENT_TARGET_ID",
        "ECS_DEPLOYMENT_EVENT_LOG_GROUP",
        "WEB_LOG_GROUP",
        "WEB_LOG_STREAM_PREFIX",
        "DOCTOR_LOG_GROUP",
        "DOCTOR_LOG_STREAM_PREFIX",
        "--launch-type FARGATE",
        "application-autoscaling describe-scalable-targets",
        'TargetType == "ip"',
        "HealthCheckEnabled == true",
        'HealthCheckPath == "/api/ready"',
        'Matcher.HttpCode == "200"',
        "HealthCheckTimeoutSeconds >= 6",
        "--force-new-deployment",
        "services-stable",
        "SERVICE_DEPLOYMENT_FAILED",
        "privateIpv4Address",
        "every 30 seconds for 20 iterations",
        "readiness_check_not_ready",
        "release/schema compatibility record",
        "rollback_permitted",
        "Unknown or unapproved compatibility is NO-GO",
    )
    for marker in required:
        assert marker in text, marker


def test_every_desired_one_update_forces_a_new_deployment() -> None:
    for fence in _fences("bash"):
        normalized = fence.replace("\\\n", " ")
        for command in re.findall(r"aws ecs update-service.*?(?=\n\n|$)", normalized, flags=re.DOTALL):
            if "--desired-count 1" in command:
                assert "--force-new-deployment" in command
                assert "minimumHealthyPercent" in command
                assert "maximumPercent" in command


def test_runbook_rejects_unsafe_probe_evidence_and_promotion_regressions() -> None:
    text = _text()
    assert "curl -fsS" not in text
    assert "raw logs are never printed or persisted" in text
    assert "Promotion is forbidden before Plan 12 final GO" in text
    assert "container healthCheck" in text
    assert "elspeth health" in text
    assert re.search(r"elspeth health.*not wired", text, flags=re.IGNORECASE | re.DOTALL)
    assert "any old target may be ignored only while `draining`" in text


def test_runbook_pins_disposable_cleanup_and_orphan_sweep() -> None:
    text = _text()
    required = (
        "ACCEPTANCE_RUN_ID",
        "teardown deadline",
        "Terraform",
        "Cognito test identity",
        "rollback-baseline ECR tag",
        "candidate acceptance ECR tag",
        "Cleanup failure is itself NO-GO",
        "orphan-sweep",
        "DELETE_IN_PROGRESS",
        "CLEANUP_REQUIRED=0",
        "promotion",
    )
    for marker in required:
        assert marker in text, marker


def test_runbook_is_linked_from_operator_indexes() -> None:
    assert (
        "| [AWS ECS Deployment](aws-ecs-deployment.md) | Deploying ELSPETH web to AWS ECS Fargate with Aurora PostgreSQL |"
    ) in RUNBOOK_INDEX.read_text(encoding="utf-8")
    assert (
        "[AWS ECS Deployment Runbook](../runbooks/aws-ecs-deployment.md) - Production ECS/Fargate deployment contract"
    ) in DOCKER_GUIDE.read_text(encoding="utf-8")
