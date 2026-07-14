"""Executable contract for the AWS ECS deployment runbook."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from elspeth.web.aws_ecs_acceptance import SCENARIO_ASSIGNMENT_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "aws-ecs-deployment.md"
RUNBOOK_INDEX = REPO_ROOT / "docs" / "runbooks" / "index.md"
DOCKER_GUIDE = REPO_ROOT / "docs" / "guides" / "docker.md"
PLAN12 = REPO_ROOT / "docs" / "superpowers" / "plans" / "aws" / "2026-07-08-aws-ecs-12-integration-closeout.md"


def _text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def _plan12_text() -> str:
    return PLAN12.read_text(encoding="utf-8")


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
        "ELSPETH_ACCEPTANCE_CANDIDATE_SHA": "${CANDIDATE_SHA}",
        "ELSPETH_ACCEPTANCE_SCENARIO_ID": "${SCENARIO_ID}",
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


def test_protected_command_wrappers_are_bounded_redacted_and_complete() -> None:
    text = _text()
    capture = text[text.index("export ELSPETH_COMMAND_OUTPUT_LIMIT_BYTES") : text.index("### Closed lifecycle helper wrappers")]

    for helper in ("aws_capture", "aws_ecr_login", "terraform_capture", "verify_tf_binding"):
        assert f"{helper}() (" in capture
    for marker in (
        "ELSPETH_COMMAND_OUTPUT_LIMIT_BYTES=2097152",
        "ulimit -f 4096",
        "timeout --signal=TERM --kill-after=5s",
        '--cli-connect-timeout "$AWS_CLI_CONNECT_TIMEOUT"',
        '--cli-read-timeout "$AWS_CLI_READ_TIMEOUT"',
        "trap 'rm -f",
        "aws_command_failed",
        "terraform_command_failed",
        "docker login --username AWS --password-stdin",
        "elspeth.aws-ecs-tf-binding.v1",
        "terraform_binding_live_mismatch",
    ):
        assert marker in capture
    assert 'cat "$stderr_file"' not in capture
    assert "get-login-password" in capture


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


def test_runbook_defines_every_packaged_lifecycle_wrapper_before_first_runtime_use() -> None:
    text = _text()
    for helper in (
        "persist_sanitized_receipt",
        "require_signed_tf_plan_approval",
        "require_signed_tf_destroy_approval",
        "load_scenario",
        "run_orphan_sweep",
        "finalize_cleanup_evidence",
    ):
        definition = text.index(f"{helper}() {{")
        invocations = [match.start() for match in re.finditer(rf"(?m)^\s*{helper}(?:\s|$)", text)]
        assert not invocations or min(invocations) >= definition
    for command in (
        "receipt-store",
        "approval-verify",
        "scenario-load",
        "orphan-sweep",
        "cleanup-evidence-finalize",
    ):
        assert command in text


def test_load_scenario_clears_every_closed_assignment_before_loading() -> None:
    text = _text()
    wrapper = text[text.index("load_scenario() {") : text.index("run_orphan_sweep() {")]
    unset_block = wrapper[: wrapper.index("assignments=$(uv run")]

    for name in SCENARIO_ASSIGNMENT_NAMES:
        assert re.search(rf"\b{name}\b", unset_block), name


def test_runbook_pins_exact_oidc_redirect_phases_and_closed_evidence() -> None:
    text = _text()
    assert 'OIDC_REDIRECT_URI="${ALB_BASE_URL}/"' in text
    assert '--arg callback "$OIDC_REDIRECT_URI"' in text
    assert "STAGING_BASE_URL is the slashless origin" not in text
    for phase in (
        "previous-before-candidate",
        "candidate-initial",
        "previous-after-rollback",
        "candidate-after-redeploy",
    ):
        assert phase in text
    for field in (
        "subject_sha256",
        "auth_me_status",
        "session_create_status",
        "session_read_status",
        "session_delete_status",
        "session_round_trip",
    ):
        assert field in text
    assert "[chromium] aws-ecs-oidc.staging.spec.ts" in text


def test_runbook_pins_replacement_then_persistence_role_and_drained_local_auth_order() -> None:
    text = _text()
    sequence = text[text.index("#### Persistence, replacement") : text.index("### 6. Observe")]
    ordered = (
        "aws_ecs_acceptance capture",
        "PRE_REPLACEMENT_TASK_ARN",
        "aws_ecs_acceptance verify-api",
        "PAYLOAD_VERIFIER_TASK_DEFINITION",
        'run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-s3',
        "LOCAL_AUTH_VERIFIER_TASK_DEFINITION",
    )
    positions = [sequence.index(marker) for marker in ordered]
    assert positions == sorted(positions)
    assert 'task-level `user: "1000:1000"`' in text
    assert "root-running ECS Exec" in text


def test_runbook_uses_canonical_orphan_and_two_phase_finalizer_commands() -> None:
    text = _text()
    assert '--file "$CONTROL_MANIFEST" --acceptance-run-id "$ACCEPTANCE_RUN_ID"' in text
    assert "--control-manifest" not in text
    assert "--scenario all" not in text
    assert "finalize_cleanup_evidence prepare" in text
    assert "finalize_cleanup_evidence commit" in text
    assert text.index("finalize_cleanup_evidence prepare") < text.index("finalize_cleanup_evidence commit")


def test_runbook_projects_cognito_before_capture_and_persists_local_evidence_paths_first() -> None:
    text = _text()
    cognito = text[text.index("COGNITO_CLIENT_JSON=$(aws_capture") : text.index("unset COGNITO_CLIENT_JSON")]
    assert "UserPoolClient.{clientId:ClientId" in cognito
    assert ".UserPoolClient as $c" not in cognito
    oidc_create = text.index("OIDC_EVIDENCE_DIR=$(mktemp")
    oidc_update = text.index('--oidc-evidence-dir "$OIDC_EVIDENCE_DIR"', oidc_create)
    oidc_run = text.index("run_oidc_evidence()", oidc_create)
    assert oidc_create < oidc_update < oidc_run
    state_create = text.index("ACCEPTANCE_STATE=$(mktemp")
    state_update = text.index('--acceptance-state-path "$ACCEPTANCE_STATE"', state_create)
    capture = text.index("aws_ecs_acceptance capture", state_create)
    assert state_create < state_update < capture
    assert "--receipt-stdin" in text


def test_plan12_executes_protected_scenario_order_and_durable_cleanup_contract() -> None:
    text = _plan12_text()
    assert re.search(r"aws_capture\s+(?!aws\b)", text) is None
    assert "run_acceptance_tf_bounded" not in text
    assert "run_cleanup_bounded terraform" not in text
    assert re.search(r"(?m)^\s*terraform\s+-chdir", text) is None
    assert text.count("load_scenario A") >= 7 and text.count("load_scenario B") >= 8
    replacement = text[text.index("ORIGINAL_TASK_ARN=") : text.index("Complete the runbook's 20 consecutive")]
    assert replacement.index("verify-api") < replacement.index("PAYLOAD_TASK_ARN") < replacement.index("run_candidate_role_checks")
    assert text.index('--oidc-evidence-dir "$OIDC_EVIDENCE_DIR"') < text.index("COGNITO_CLIENT_PREFLIGHT=")
    assert text.index('--acceptance-state-path "$ACCEPTANCE_STATE"') < text.index("aws_ecs_acceptance capture")
    assert '"$SCENARIO_A_TF_BINDING_FILE"' in text and '"$SCENARIO_B_TF_BINDING_FILE"' in text
    assert "elspeth.aws-ecs-evidence-export.v1" in text
    assert '--evidence-export-receipt "$EVIDENCE_EXPORT_RECEIPT"' in text
    assert '--final-evidence-export-receipt "$FINAL_EVIDENCE_EXPORT_RECEIPT"' in text
    assert "elspeth.aws-ecs-scenario-inventory.v4" in text
    assert "elspeth.aws-ecs-control-manifest.v4" in text
    assert "elspeth.aws-ecs-retained-evidence.v1" in text
    assert "control-manifest bind-retained-evidence" in text
    assert "control-manifest bind-scenario" in text
    assert "gate-ledger bind-candidate" in text
    assert "IDENTITY_CLEANUP_CONFIRMED" not in text
    assert "SHARED_RESOURCE_CLEANUP_CONFIRMED" not in text
    assert "PRECLEANUP_EVIDENCE_EXPORT_CONFIRMED" not in text
    assert text.index("record_gate_check task8.check04") < text.index('--final-evidence-export-receipt "$FINAL_EVIDENCE_EXPORT_RECEIPT"')
    assert "emergency_horizon_started_or_renewed" not in text
    for task, count in {1: 13, 2: 8, 3: 7, 4: 2, 5: 10, 6: 6, 7: 30, 8: 5}.items():
        assert f"task{task}.check01" in text
        assert f"task{task}.check{count:02d}" in text


def test_plan12_binds_each_resolved_inventory_in_apply_order_and_preserves_noop_identity() -> None:
    text = _plan12_text()
    apply_function = text[text.index("plan_and_apply_scenario()") : text.index("export SCENARIO_A_RESOLVED_INVENTORY")]
    assert apply_function.index('noop_plan_sha="$(sha256sum "$plan"') < apply_function.index(
        '--terraform-noop-receipt "$scenario_id:$noop_plan_sha:$durable_receipt"'
    )
    assert apply_function.index('--terraform-noop-receipt "$scenario_id:$noop_plan_sha:$durable_receipt"') < apply_function.index(
        "control-manifest bind-scenario"
    )
    assert apply_function.index("approval-require-current") < apply_function.index('terraform_capture -chdir="$dir" apply')
    calls_start = text.index("export SCENARIO_A_RESOLVED_INVENTORY")
    scenario_calls = text[calls_start : text.index("load_scenario A", calls_start)]
    assert scenario_calls.index("plan_and_apply_scenario scenario_a A") < scenario_calls.index("plan_and_apply_scenario scenario_b B")
    assert '"$SCENARIO_A_RESOLVED_INVENTORY"' in scenario_calls
    assert '"$SCENARIO_B_RESOLVED_INVENTORY"' in scenario_calls
    assert text.index("record_gate_check task1.check13") < text.index("gate-ledger bind-candidate")
    destroy = text[text.index("destroy_acceptance_stack()") : text.index("ORPHAN_SWEEP_CONFIRMED=0")]
    assert destroy.index("approval-require-current") < destroy.index('terraform_capture -chdir="$dir" apply')


def test_plan12_task8_rows_are_retryable_and_task9_finalizes_only_on_go() -> None:
    text = _plan12_text()
    task8 = text[text.index("### Task 8:") : text.index("### Task 9:")]
    assert task8.count("- [ ]") == 5
    mapping = (
        "`task8.check01` is protected cleanup resume",
        "`task8.check02` is the initial",
        "`task8.check03` is the preserved original",
        "`task8.check04` is identity/shared-resource",
        "`task8.check05` is successful aggregate coordinator completion",
    )
    assert all(marker in task8 for marker in mapping)
    for check_id in range(1, 5):
        assert f"record_gate_check task8.check{check_id:02d} 0" in task8
    assert "TASK8_CLEANUP_EXIT" not in task8
    assert 'record_gate_check task8.check04 "$TASK8_CLEANUP_EXIT"' not in task8
    task9 = text[text.index("### Task 9:") :]
    no_go = task9[task9.index("Return **NO-GO**") : task9.index("On GO only")]
    go = task9[task9.index("On GO only") :]
    assert "gate-ledger finalize" not in no_go
    assert "leave the protected ledger\n  unfinalized" in no_go
    assert "gate-ledger finalize" in go


def test_plan12_binds_post_observation_evidence_without_reopening_resource_inventories() -> None:
    text = _plan12_text()
    retained = text.index("control-manifest bind-retained-evidence")
    task7 = text.index("### Task 7:")
    task8 = text.index("### Task 8:")
    assert task7 < retained < task8
    retained_section = text[retained - 900 : retained + 300]
    assert "resource inventories remain immutable" in retained_section
    assert "elspeth.aws-ecs-retained-evidence.v1" in retained_section


def test_runbook_verifies_the_initialized_remote_backend_identity() -> None:
    text = _text()
    assert 'backend_metadata="$directory/.terraform/terraform.tfstate"' in text
    assert '.backend.type == "s3"' in text
    assert ".backend.config.encrypt" in text
    assert ".backend.config.use_lockfile" in text
    assert ".backend.config.dynamodb_table" in text
    assert "backend_key_hash=$(jq -jr" in text
    assert ".backend_state_key_sha256 == $backend_key_hash" in text


def test_runbook_refreshes_emergency_deadline_inside_every_protected_call() -> None:
    text = _text()
    timeout_helper = text[text.index("protected_timeout_seconds()") : text.index("aws_capture()")]
    assert "ELSPETH_CLEANUP_MODE" in timeout_helper
    assert "control-manifest load-cleanup" in timeout_helper
    assert "EMERGENCY_CLEANUP_DEADLINE_UTC" in timeout_helper


def test_runbook_pins_operator_telemetry_positive_outage_replacement_positive_order() -> None:
    text = _text()
    sequence = text[text.index("# Run once in Scenario A") : text.index("LOCAL_AUTH_OVERRIDES=")]
    markers = (
        'run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry outage',
        'OUTAGE_TASK_ARN="$CANDIDATE_TASK_ARN"',
        "--force-new-deployment",
        'test "$CANDIDATE_TASK_ARN" != "$OUTAGE_TASK_ARN"',
        'run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry positive',
    )
    positions = [sequence.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "--container cloudwatch-agent" in sequence


def test_runbook_is_linked_from_operator_indexes() -> None:
    assert (
        "| [AWS ECS Deployment](aws-ecs-deployment.md) | Deploying ELSPETH web to AWS ECS Fargate with Aurora PostgreSQL |"
    ) in RUNBOOK_INDEX.read_text(encoding="utf-8")
    assert (
        "[AWS ECS Deployment Runbook](../runbooks/aws-ecs-deployment.md) - Production ECS/Fargate deployment contract"
    ) in DOCKER_GUIDE.read_text(encoding="utf-8")
