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
OIDC_PLAYWRIGHT_CONFIG = REPO_ROOT / "src" / "elspeth" / "web" / "frontend" / "playwright.oidc.config.ts"


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
        "ELSPETH_WEB__PLUGIN_ALLOWLIST": "${ELSPETH_WEB__PLUGIN_ALLOWLIST}",
        "ELSPETH_WEB__PLUGIN_PREFERENCES": "${ELSPETH_WEB__PLUGIN_PREFERENCES}",
        "ELSPETH_WEB__PLUGIN_CONTROL_MODES": "${ELSPETH_WEB__PLUGIN_CONTROL_MODES}",
        "ELSPETH_WEB__LLM_PROFILES": "${ELSPETH_WEB__LLM_PROFILES}",
        "ELSPETH_WEB__TUTORIAL_LLM_PROFILE": "${ELSPETH_WEB__TUTORIAL_LLM_PROFILE}",
        "ELSPETH_WEB__BEDROCK_GUARDRAIL_PROFILES": "${ELSPETH_WEB__BEDROCK_GUARDRAIL_PROFILES}",
        "ELSPETH_WEB__BEDROCK_GUARDRAIL_DEFAULT_PROFILES": "${ELSPETH_WEB__BEDROCK_GUARDRAIL_DEFAULT_PROFILES}",
        "ELSPETH_ACCEPTANCE_PLUGIN_POLICY_BINDING_SHA256": "${ELSPETH_ACCEPTANCE_PLUGIN_POLICY_BINDING_SHA256}",
        "ELSPETH_BEDROCK_LIVE_TEST_MODEL": "${ELSPETH_BEDROCK_LIVE_TEST_MODEL}",
        "AWS_REGION": "${AWS_REGION}",
        "ELSPETH_WEB__OPERATOR_TELEMETRY": "aws-otlp",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_ENVIRONMENT": "production",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_RELEASE": "${ELSPETH_RELEASE_SHA_OR_DIGEST}",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_ECS_CLUSTER": "${ECS_CLUSTER_NAME}",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_ECS_SERVICE": "${ECS_SERVICE_NAME}",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_TASK_DEFINITION_FAMILY": "${ECS_TASK_DEFINITION_FAMILY}",
        "ELSPETH_WEB__OPERATOR_TELEMETRY_TASK_DEFINITION_REVISION": "${ECS_TASK_DEFINITION_REVISION}",
        "ELSPETH_ACCEPTANCE_RUN_ID": "${ACCEPTANCE_RUN_ID}",
        "ELSPETH_ACCEPTANCE_CANDIDATE_SHA": "${CANDIDATE_SHA}",
        "ELSPETH_ACCEPTANCE_SCENARIO_ID": "${SCENARIO_ID}",
        "ELSPETH_ACCEPTANCE_S3_BUCKET": "${ELSPETH_TEST_S3_BUCKET}",
        "ELSPETH_ACCEPTANCE_S3_PREFIX": "${SCENARIO_RESOURCE_NAMESPACE}/${ACCEPTANCE_RUN_ID}",
    }
    assert "127.0.0.1:4317" in _text()


def test_runbook_consumes_complete_web_plugin_policy_handoff() -> None:
    text = _text()

    for required in (
        "ELSPETH_WEB__PLUGIN_ALLOWLIST",
        "ELSPETH_WEB__PLUGIN_PREFERENCES",
        "ELSPETH_WEB__PLUGIN_CONTROL_MODES",
        "ELSPETH_WEB__LLM_PROFILES",
        "ELSPETH_WEB__TUTORIAL_LLM_PROFILE",
        "ELSPETH_WEB__BEDROCK_GUARDRAIL_PROFILES",
        "ELSPETH_WEB__BEDROCK_GUARDRAIL_DEFAULT_PROFILES",
        "ELSPETH_BEDROCK_LIVE_TEST_MODEL",
        "GET /api/system/status",
        "tutorial_required_control_coverage",
        "typed HTTP 409",
        "plugin_policy",
        "landscape_evidence",
        "run_web_plugin_policy",
    ):
        assert required in text

    assert "register a new task-definition revision" in text
    assert "force a new deployment" in text
    assert "validate-task-definition-policy" in text
    assert "resolve_bound_task_definition CANDIDATE_TASK_DEFINITION" in text
    assert "resolve_bound_task_definition DOCTOR_TASK_DEFINITION" in text
    assert "resolve_bound_task_definition PAYLOAD_VERIFIER_TASK_DEFINITION" in text
    assert "resolve_bound_task_definition LOCAL_AUTH_VERIFIER_TASK_DEFINITION" in text
    assert "resolve_bound_task_definition ROLLBACK_DOCTOR_TASK_DEFINITION" in text
    assert "--plugin-policy-binding-sha256" in text
    assert "target_llm" in text
    assert "selected_controls" in text
    assert "prompt-shield" in text
    assert "content-safety" in text


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
        "log_group_name": "${OPERATOR_METRICS_LOG_GROUP}",
        "log_stream_name": "telemetry",
        "dimension_rollup_option": "NoDimensionRollup",
        "retain_initial_value_of_delta_metric": True,
        "resource_to_telemetry_conversion": {"enabled": True},
    }
    assert otel["exporters"]["awsxray/elspeth"] == {}
    assert "OPERATOR_METRICS_LOG_GROUP" in SCENARIO_ASSIGNMENT_NAMES
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
        "cloudwatch:GetMetricData",
        "xray:BatchGetTraces",
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

    for helper in ("aws_capture_with_kind", "aws_ecr_login", "terraform_capture", "verify_tf_binding"):
        assert f"{helper}() (" in capture
    for helper in ("aws_capture", "aws_waiter_capture"):
        assert f"{helper}() {{" in capture
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
    terraform = capture[capture.index("terraform_capture()") : capture.index("verify_tf_binding()")]
    assert "ulimit -f" not in terraform
    assert "ELSPETH_AWS_EXEC_CEILING_SECONDS=420" in capture
    assert "aws_exec_capture()" in capture


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
        "elspeth:ecs-0.7.1-closeout",
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


def test_verifier_overrides_do_not_repeat_the_python_module_entrypoint() -> None:
    text = _text()

    assert '"entryPoint": ["python", "-m", "elspeth.web.aws_ecs_acceptance"]' in text
    assert 'command:["verify-payloads","--landscape-run-id",$run]' in text
    assert 'command:["verify-local-auth"]' in text
    assert 'command:["python","-m","elspeth.web.aws_ecs_acceptance","verify-payloads"' not in text
    assert 'command:["python","-m","elspeth.web.aws_ecs_acceptance","verify-local-auth"]' not in text


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
    for marker in (
        "OIDC_BEARER_HANDOFF_FILE",
        'ELSPETH_ACCEPTANCE_BEARER_TOKEN="$OIDC_BEARER_TOKEN"',
        'run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry positive "$OIDC_LANDSCAPE_RUN_ID"',
        "--landscape-run-id",
    ):
        assert marker in text
    handoff = text[text.index("capture_oidc_lifecycle_handoff() {") : text.index("## ECS probe wiring")]
    assert handoff.index("unset OIDC_BEARER_TOKEN") < handoff.index("OIDC_LANDSCAPE_RUN_ID=")
    role_check = text[text.index("run_candidate_role_check() {") : text.index("# Run once in Scenario A")]
    assert "ELSPETH_ACCEPTANCE_BEARER_TOKEN" not in role_check


def test_fresh_account_https_uses_per_scenario_certificate_and_exact_browser_spki_pin() -> None:
    text = _text()
    config = OIDC_PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")

    for marker in (
        "tls_self_signed_cert",
        "aws_acm_certificate",
        "aws_lb.web.dns_name",
        "ACCEPTANCE_TLS_CA_BUNDLE",
        "ACCEPTANCE_TLS_SPKI_SHA256",
        '--cacert "$ACCEPTANCE_TLS_CA_BUNDLE"',
        'SSL_CERT_FILE="$ACCEPTANCE_TLS_CA_BUNDLE"',
        'NODE_EXTRA_CA_CERTS="$ACCEPTANCE_TLS_CA_BUNDLE"',
        'OIDC_TLS_SPKI_SHA256="$ACCEPTANCE_TLS_SPKI_SHA256"',
    ):
        assert marker in text
    assert "--ignore-certificate-errors-spki-list=" in config
    assert "OIDC_TLS_SPKI_SHA256" in config
    assert "ignoreHTTPSErrors: true" not in config


def test_fresh_account_bootstrap_is_manifest_armed_backends_initialized_and_destroyed_before_orphan_sweep() -> None:
    text = _text()
    bootstrap = text[text.index("### Fresh-account shared bootstrap") : text.index("### Temporary image publication")]
    cleanup = text[text.index("## Disposable acceptance cleanup") :]

    for marker in (
        'BOOTSTRAP_TF_DIR="$IAC_PACKAGE_DIR/bootstrap"',
        'BACKEND_STATE_BUCKET="elspeth-acc-${ACCEPTANCE_RUN_ID//-/}"',
        "arm_external_cleanup",
        'terraform_capture -chdir="$BOOTSTRAP_TF_DIR" apply',
        "checkpoint_cleanup shared_resource_cleanup pending",
        "initialize_scenario_backend A",
        "initialize_scenario_backend B",
        "backend_state_bucket",
        "ECR_REPOSITORY",
    ):
        assert marker in bootstrap
    assert bootstrap.index("arm_external_cleanup") < bootstrap.index('terraform_capture -chdir="$BOOTSTRAP_TF_DIR" apply')
    assert bootstrap.index("initialize_scenario_backend A") < text.index("### Temporary image publication")
    assert 'terraform_capture -chdir="$BOOTSTRAP_TF_DIR" plan -destroy' in cleanup
    assert cleanup.index("if ! destroy_scenario B") < cleanup.index('if test "$scenario_a_destroyed" = 1')
    assert cleanup.index('if test "$scenario_a_destroyed" = 1') < cleanup.index("run_orphan_sweep")
    shared_destroy = cleanup[cleanup.index("destroy_shared_bootstrap() {") : cleanup.index("if ! destroy_scenario A")]
    assert shared_destroy.index('terraform_capture -chdir="$BOOTSTRAP_TF_DIR" plan -destroy') < shared_destroy.index(
        "checkpoint_cleanup shared_resource_cleanup confirmed"
    )


def test_fresh_scenarios_bootstrap_schema_before_first_or_upgrade_candidate() -> None:
    text = _text()
    first = text[text.index("### Fresh Scenario A database baseline") : text.index("### Fresh Scenario B upgrade baseline")]
    first_ordered = (
        'test "$ACTIVE_SCENARIO_ID" = A',
        '"doctor","aws-ecs","--init-schema","--json"',
        '--task-definition "$DOCTOR_TASK_DEFINITION"',
    )
    first_positions = [first.index(marker) for marker in first_ordered]
    assert first_positions == sorted(first_positions)
    assert '--task-definition "$CANDIDATE_TASK_DEFINITION"' not in first

    section = text[text.index("### Fresh Scenario B upgrade baseline") : text.index("## ECS probe wiring")]
    ordered = (
        'test "$ACTIVE_SCENARIO_ID" = B',
        '"doctor","aws-ecs","--init-schema","--json"',
        '--task-definition "$ROLLBACK_DOCTOR_TASK_DEFINITION"',
        '--task-definition "$PREVIOUS_TASK_DEFINITION"',
        "run_oidc_evidence previous-before-candidate",
    )
    positions = [section.index(marker) for marker in ordered]
    assert positions == sorted(positions)
    assert '--task-definition "$CANDIDATE_TASK_DEFINITION"' not in section


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
    assert sequence.index("require_compatibility_record_current") < sequence.index("PRE_REPLACEMENT_TASK_ARN")
    assert 'task-level `user: "1000:1000"`' in text
    assert "root-running ECS Exec" in text

    phase = text[text.index("### 5. Perform candidate-aware acceptance") : text.index("### 6. Observe")]
    helper = phase[phase.index("run_candidate_role_check()") : phase.index("verify_candidate_target_mapping\n")]
    assert phase.index("run_candidate_role_check()") < phase.index('run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-s3')
    assert phase.index("checkpoint_operator_retained_evidence()") < phase.index(
        'run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry'
    )
    assert helper.index("persist_sanitized_receipt") < helper.index("checkpoint_operator_retained_evidence")
    assert helper.index("checkpoint_operator_retained_evidence") < helper.index('rm -f "$receipt_file"')


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
    identity = text[text.index("provision_scenario_b_test_identity()") : text.index("render_resolved_inventory()")]
    assert "admin-create-user" in identity
    assert "admin-set-user-password" in identity
    assert "admin-get-user" in identity
    render = text[text.index("render_resolved_inventory()") : text.index("plan_and_apply_scenario()")]
    assert render.index("provision_scenario_b_test_identity") < render.index("cognito_subject_sub = $subject")
    cognito = text[text.index("prepare_scenario_b_oidc()") : text.index("### Real-browser OIDC evidence")]
    assert "UserPoolClient.{clientId:ClientId" in cognito
    assert ".UserPoolClient as $c" not in cognito
    assert "admin-get-user" in cognito
    assert ".sub == $subject" in cognito
    oidc_create = text.index('OIDC_EVIDENCE_DIR="/tmp/elspeth-oidc-${ACCEPTANCE_RUN_ID}"')
    oidc_update = text.index('--oidc-evidence-dir "$OIDC_EVIDENCE_DIR"', oidc_create)
    oidc_run = text.index("run_oidc_evidence()", oidc_create)
    assert oidc_create < oidc_update < oidc_run
    browser_preflight = text.index("playwright install chromium")
    assert browser_preflight < text.index("\narm_external_cleanup\n", browser_preflight)
    scenario_b = text.index("load_scenario B")
    assert scenario_b < text.index("prepare_scenario_b_oidc", scenario_b)
    state_create = text.index("ACCEPTANCE_STATE=$(mktemp")
    state_update = text.index('--acceptance-state-path "$ACCEPTANCE_STATE"', state_create)
    capture = text.index("aws_ecs_acceptance capture", state_create)
    assert state_create < state_update < capture
    assert "--receipt-stdin" in text


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
    outage = sequence[sequence.index('OUTAGE_TASK_ARN="$CANDIDATE_TASK_ARN"') :]
    assert outage.index("require_compatibility_record_current") < outage.index("aws_capture aws ecs update-service")
    assert "--container cloudwatch-agent" in sequence


def test_runbook_starts_connection_observation_on_a_future_minute_boundary() -> None:
    text = _text()
    observe = text[text.index("### 6. Observe for ten minutes") : text.index("### 7. Roll back")]
    assert "OBSERVATION_ALIGNMENT_SECONDS=$((60 - 10#$(date -u +%S)))" in observe
    assert observe.index('sleep "$OBSERVATION_ALIGNMENT_SECONDS"') < observe.index("ACCEPTANCE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:00Z)")
    assert "ACCEPTANCE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" not in observe


def test_runbook_validates_task_definitions_and_compatibility_before_baseline_mutation() -> None:
    text = _text()
    for scenario, end_marker in (("A", "### Fresh Scenario B upgrade baseline"), ("B", "## ECS probe wiring")):
        start = text.index(f"load_scenario {scenario}")
        block = text[start : text.index(end_marker, start)]
        assert block.index("validate_scenario_task_definitions") < block.index("bind_compatibility_record")
        assert block.index("bind_compatibility_record") < block.index("provision_scenario_storage")
        assert block.index("provision_scenario_storage") < block.index("--init-schema")
    assert ".containers[] | select(.essential == true)" not in text
    assert "require_stopped_task_success" in text


def test_runbook_binds_bootstrap_approvals_and_terminal_receipt_lifecycle() -> None:
    text = _text()
    bootstrap_apply = text[
        text.index('BOOTSTRAP_PLAN="$BOOTSTRAP_TF_DIR/bootstrap.tfplan"') : text.index("checkpoint_cleanup shared_resource_cleanup pending")
    ]
    assert bootstrap_apply.index("persist_sanitized_receipt bootstrap terraform-plan") < bootstrap_apply.index("approval-require-current")
    assert bootstrap_apply.index("approval-require-current") < bootstrap_apply.rindex('sha256sum "$BOOTSTRAP_PLAN"')
    assert bootstrap_apply.rindex('sha256sum "$BOOTSTRAP_PLAN"') < bootstrap_apply.index(" apply -input=false")
    bootstrap_destroy = text[text.index("destroy_shared_bootstrap()") : text.index("scenario_a_destroyed=0")]
    assert "persist_sanitized_receipt bootstrap terraform-destroy-plan" in bootstrap_destroy
    assert bootstrap_destroy.index("approval-require-current") < bootstrap_destroy.rindex('sha256sum "$plan"')
    assert bootstrap_destroy.rindex('sha256sum "$plan"') < bootstrap_destroy.index(" apply -input=false")
    assert 'SANITIZED_ORPHAN_RECEIPT="/tmp/elspeth-orphan-${ACCEPTANCE_RUN_ID}.json"' in text
    orphan = text[text.index("ORPHAN_RECEIPT_DIR=") : text.index('if test "${#cleanup_failures[@]}"')]
    assert "install -m 600 /dev/null" not in orphan
    assert orphan.index("mktemp") < orphan.index("run_orphan_sweep") < orphan.index("mv -fT")
    assert "protected_timeout_seconds orphan-sweep" in text
    removal = text[text.index("remove_local_acceptance_evidence()") : text.index("Use these transitions")]
    assert 'rm -f -- "$SANITIZED_ORPHAN_RECEIPT"' in removal


def test_runbook_is_linked_from_operator_indexes() -> None:
    assert (
        "| [AWS ECS Deployment](aws-ecs-deployment.md) | Deploying ELSPETH web to AWS ECS Fargate with Aurora PostgreSQL |"
    ) in RUNBOOK_INDEX.read_text(encoding="utf-8")
    assert (
        "[AWS ECS Deployment Runbook](../runbooks/aws-ecs-deployment.md) - Production ECS/Fargate deployment contract"
    ) in DOCKER_GUIDE.read_text(encoding="utf-8")
