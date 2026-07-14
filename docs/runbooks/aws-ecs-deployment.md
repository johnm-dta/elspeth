# Runbook: AWS ECS Fargate deployment

Deploy one ELSPETH web task to AWS ECS Fargate with Aurora PostgreSQL, EFS,
Cognito/OIDC, task-role S3 and Bedrock access, and CloudWatch operator
telemetry. This runbook permits planned downtime: it deliberately uses a
zero-overlap, single-task deployment.

---

## Symptoms

Use this runbook when you need to:

- deploy or upgrade ELSPETH web on ECS Fargate;
- prove schema, persistence, authentication, S3, Bedrock, Guardrail, and
  operator-telemetry behavior before admitting traffic;
- roll back one immutable task definition without guessing about database
  compatibility; or
- tear down a disposable acceptance environment without leaving run-scoped
  AWS resources.

Do not use it to publish a durable image, run more than one web task, automate
a destructive database reset, or infer that CloudWatch evidence replaces the
Landscape audit record.

---

## Contract summary

This is the canonical operator entry point for the AWS ECS deployment. The
application and collector have deliberately separate evidence roles:

- Landscape is the permanent source of truth for lineage, replay, and run
  decisions. Its write must succeed before an operational signal is emitted.
- CloudWatch and X-Ray are best-effort operational telemetry. A receipt there
  never proves an audit write, and loss of the collector never rolls back a
  committed Landscape record.

`ELSPETH_WEB__DEPLOYMENT_TARGET=aws-ecs` requires PostgreSQL
`session_db_url` and `landscape_url` values (including
`postgresql+psycopg://...`), a pre-provisioned writable `data_dir`, an explicit
writable `payload_store_path`, and non-placeholder
`ELSPETH_WEB__SECRET_KEY` and
`ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY` secrets. Web startup validates this
contract but never creates, migrates, drops, or repairs either schema.

AWS credentials are provided by ECS task roles and AWS Secrets Manager
injection — never baked into the image or passed on the command line. ECS
injects secrets via AWS Secrets Manager into the task definition; Azure
deployments use the Azure equivalent (Key Vault — see
[configure-keyvault-secrets.md](configure-keyvault-secrets.md)). The task
`executionRoleArn` pulls the image, writes awslogs, and resolves approved
Secrets Manager references. The distinct `taskRoleArn` supplies only runtime
S3, Bedrock, Guardrail, EFS, ECS Exec, CloudWatch, and X-Ray permissions.

AWS web tasks set `ELSPETH_WEB__OPERATOR_TELEMETRY=aws-otlp`, an explicit
environment, release SHA or image digest, ECS cluster/service, and task
definition family/revision. The bounded service name and 1–3600 second export
interval remain optional overrides. The receiver is fixed by the application
at `http://127.0.0.1:4317`; it is not an operator endpoint setting. Uploaded
pipeline telemetry routing is replaced with one lifecycle-or-rows OTLP
exporter, empty headers, and best-effort failure policy.

## Prerequisites

- A database operator has approved the Landscape and session PostgreSQL
  targets and the runtime principal.
- A release operator has approved an immutable ELSPETH image and an immutable
  CloudWatch Agent image digest.
- A telemetry owner has approved the namespace, log group, X-Ray destination,
  retention, alarm actions, and the cost budget below.
- AWS authentication comes only from the ECS task role/default credential
  chain. Do not add keys, profiles, credential headers, role overrides, or
  AWS service endpoint overrides to either container.
- AWS CLI v2, `jq`, `curl`, Docker, the Session Manager plugin, Node 22/npm,
  and Playwright Chromium are installed from reviewed locks before mutation.
- Every web, doctor, payload-verifier, and local-auth-verifier task definition
  uses Linux Fargate 1.4.0 or `LATEST`, declares `runtimePlatform`, and uses
  the approved EFS access point and immutable image digest.

### Protected command capture

Start every operator shell with strict mode and protected capture. Every AWS
CLI call in rollout, acceptance, diagnosis, and cleanup goes through
`aws_capture`; raw logs are never printed or persisted. A non-zero AWS call
emits only a static class. Plan 10's packaged acceptance module supplies the
closed `sanitize-evidence` projectors used below.

```bash
set -Eeuo pipefail
umask 077
export AWS_PAGER=""
export AWS_CLI_CONNECT_TIMEOUT=5
export AWS_CLI_READ_TIMEOUT=30

aws_capture() {
  local stdout_file stderr_file status
  stdout_file=$(mktemp)
  stderr_file=$(mktemp)
  chmod 600 "$stdout_file" "$stderr_file"
  if "$@" >"$stdout_file" 2>"$stderr_file"; then
    cat "$stdout_file"
    rm -f "$stdout_file" "$stderr_file"
    return 0
  fi
  status=$?
  rm -f "$stdout_file" "$stderr_file"
  printf '%s\n' 'aws_command_failed' >&2
  return "$status"
}
```

A plain `aws ... | sanitize-evidence` pipe is forbidden because AWS stderr
would bypass the projector. The Task 3 version also caps both channels at 2
MiB and enforces the control-manifest deadline plus a per-call cleanup ceiling.

## Authentication and secret injection

Cognito/OIDC is recommended. Configure `auth_provider=oidc`, the user-pool
`oidc_issuer`, and set both `oidc_audience` and `oidc_client_id` to the public
app-client ID. Set the hosted/custom-domain
`oidc_authorization_endpoint` and same-origin `oidc_token_endpoint`, exposed
as `ELSPETH_WEB__OIDC_AUTHORIZATION_ENDPOINT` and
`ELSPETH_WEB__OIDC_TOKEN_ENDPOINT`. The task environment also sets:

```bash
export ELSPETH_WEB__OIDC_AUTHORIZATION_ALLOWED_ORIGINS='["https://example.auth.ap-southeast-2.amazoncognito.com"]'
export ELSPETH_WEB__OIDC_AUDIENCE_CLAIM=client_id
```

The allowlist accepts exact normalized HTTPS origins only. Wildcard or suffix
matching, paths, and automatic Cognito-domain inference are rejected.
`client_id` mode is exclusively for Cognito access tokens and requires
`token_use=access`; generic OIDC continues to validate `aud`. The browser uses
the authorization code flow with S256 PKCE. The public client has no client
secret, and the implicit flow must not be enabled.

Before browser acceptance, query the one approved pool/client through the
protected capture wrapper and project only booleans and counts:

```bash
COGNITO_CLIENT_JSON=$(aws_capture aws cognito-idp describe-user-pool-client \
  --user-pool-id "$COGNITO_USER_POOL_ID" \
  --client-id "$OIDC_EXPECTED_AUDIENCE" \
  --output json)

jq -e --arg origin "$ALB_BASE_URL" '
  .UserPoolClient as $c
  | ($c.ClientSecret // "") == ""
  and $c.AllowedOAuthFlowsUserPoolClient == true
  and ($c.AllowedOAuthFlows | index("code") != null)
  and ($c.AllowedOAuthFlows | index("implicit") == null)
  and (["openid", "profile", "email"] - $c.AllowedOAuthScopes | length == 0)
  and ($c.CallbackURLs | index($origin) != null)
' <<<"$COGNITO_CLIENT_JSON" >/dev/null
unset COGNITO_CLIENT_JSON
```

`ALB_BASE_URL` is an exact HTTPS origin with no path, query, fragment, or
trailing slash. ELSPETH disables Uvicorn's raw request-line access logger so
the PKCE callback code is not logged. If ALB access logging is enabled, its
short retention and access policy must be separately approved because those
logs retain callback codes even though PKCE prevents redemption without the
verifier.

Local auth (`auth_provider=local`) is an explicit single-task option. Mount
`data_dir/auth.db` on EFS and keep SQLite journal mode `DELETE`, never WAL.
`elspeth doctor aws-ecs` never opens `auth.db`; the drained, explicit-UID
local-auth verifier opens `auth.db` read-only only after traffic is fixed at
503 and desired count is zero. A third `auth.db` reminder: never inspect it
through root-running ECS Exec beside uvicorn.

## ECS probe wiring

This runbook is the canonical operator entry point. Keep these four facts in
sync with [AWS ECS health and readiness](../operator/aws-ecs-health-and-readiness.md):

| Surface | Wiring | Meaning |
|---|---|---|
| container `healthCheck` | loopback `GET /api/health` | liveness only |
| ALB target group | `GET /api/ready` | traffic readiness |
| `elspeth doctor aws-ecs` | one-shot pre-traffic `run-task` | deployment contract and schema gate |
| `elspeth health` | not wired to any ECS probe | generic CLI health only |

The container healthCheck for the web task uses installed Python, bypasses proxy
settings, does not follow redirects, and accepts exactly HTTP 200:

```json
{
  "healthCheck": {
    "command": ["CMD", "python", "-c", "import http.client,sys; c=http.client.HTTPConnection('127.0.0.1',8451,timeout=5); c.request('GET','/api/health'); r=c.getresponse(); sys.exit(0 if r.status == 200 else 1)"],
    "interval": 30,
    "timeout": 6,
    "retries": 3,
    "startPeriod": 150
  }
}
```

Validate-only startup runs before uvicorn binds, so the port can be closed for
the two-cold-cluster worst case. Set both container `startPeriod >= 150` and
service `healthCheckGracePeriodSeconds >= 150`; the approximately 90-second
database budget excludes probe time. A zero grace period creates a restart
loop rather than readiness.

## Packaging and platform identity

Build a temporary acceptance image for one explicitly approved platform:

```bash
set -Eeuo pipefail
TARGET_PLATFORM=${TARGET_PLATFORM:?approved linux/amd64 or linux/arm64 required}
case "$TARGET_PLATFORM" in
  linux/amd64) ECS_CPU_ARCHITECTURE=X86_64 ;;
  linux/arm64) ECS_CPU_ARCHITECTURE=ARM64 ;;
  *) printf '%s\n' 'unsupported TARGET_PLATFORM' >&2; exit 2 ;;
esac

docker build --platform "$TARGET_PLATFORM" \
  --build-arg INSTALL_EXTRAS="webui llm aws postgres" \
  -t "$ACCEPTANCE_ECR_REPOSITORY:$ACCEPTANCE_RUN_ID-candidate" .
```

Every web, doctor, verifier, and rollback definition declares
`runtimePlatform.operatingSystemFamily == LINUX` and the mapped
`runtimePlatform.cpuArchitecture` (`linux/amd64` → `X86_64`, `linux/arm64` →
`ARM64`). A host-native image with no recorded target platform is NO-GO. The
lean image omits the `azure` extra; `azure_blob` pipelines need the default
`all` image or an expanded `INSTALL_EXTRAS`. Operators push their own temporary
tag to ECR. The published GHCR/ACR default remains `all`.

## Storage provisioning and cold start

Before doctor, the infrastructure/release operator provisions `data_dir`,
explicit `payload_store_path`, and the derived blob directory on the intended
EFS access point. The non-root `1000:1000` user must create, read, fsync, and
delete a probe file in each directory. Mounting only the parent is
insufficient: doctor deliberately never calls `mkdir`, including with
`--init-schema`, because an overlay child would mask a displaced EFS mount.
Record the filesystem/access-point identity and non-root probe booleans, never
paths or credentials.

`elspeth doctor aws-ecs --init-schema` has one 10-second connection attempt and
no retry budget. Before first use against min-0-ACU Aurora, set a non-zero
minimum ACU or warm the connection and re-run. Doctor is idempotent.

The web/payload/local-auth tasks use a DML-only PostgreSQL principal with no
schema ownership or `CREATE`. An init-capable doctor uses a distinct
database-operator-approved schema-owner secret. The EFS task role has only
`elasticfilesystem:ClientMount` and `elasticfilesystem:ClientWrite` scoped to
the exact filesystem/access point; `ClientRootAccess` needs separate approval.

## Bedrock, Guardrails, and S3 task-role shape

Grant the runtime task role resource-scoped `bedrock:InvokeModel`. Configure
the ordinary `region_name` and a `bedrock/anthropic...` model identifier; do
not embed AWS keys. For the two run-scoped Guardrails, grant resource-scoped
`bedrock:ApplyGuardrail` and grant `bedrock:GetGuardrail` only if the approved
preflight uses it. Terraform creates two acceptance-run-tagged Guardrails,
publishes immutable numeric versions, injects private identifier/version/
region bindings behind opaque profile aliases, records configuration hashes,
and destroys them during reviewed teardown. `DRAFT` is forbidden.

Plan 15B's kind-qualified allowlist, ordered preferences, and control modes
place explicit `aws_bedrock_prompt_shield` before the target LLM and
`aws_bedrock_content_safety` on returned content. Raw web bindings, static
credentials, and treating provider refusal as shield evidence are forbidden.
The runbook records the documented Bedrock/Azure category gap. Shared
pre-existing Guardrails are not accepted because their ownership and
configuration can drift.

For S3, grant only `s3:GetObject` for approved source prefixes and
`s3:PutObject` for approved sink prefixes. Never grant wildcard buckets. The
disposable acceptance role additionally gets `s3:DeleteObject` only for
`ELSPETH_ACCEPTANCE_S3_BUCKET` plus its UUID-scoped
`ELSPETH_ACCEPTANCE_S3_PREFIX`; the Plan 12 operator receives the same narrow
cleanup backstop. Steady-state production does not inherit test-only delete.

For ECS Exec, grant exactly `ssmmessages:CreateControlChannel`,
`ssmmessages:CreateDataChannel`, `ssmmessages:OpenControlChannel`, and
`ssmmessages:OpenDataChannel`; never `ssmmessages:*`. Require private-subnet
SSM Messages reachability and any approved KMS/session-log permissions. ECS
Exec runs as root, so it proves task-role S3/Bedrock behavior only—not EFS
permissions or the single-process local-auth contract.

## Versioned collector configuration

Store these two files in the deployment repository under a versioned
`telemetry/elspeth.cloudwatch-agent.v1/` directory. Compute both digests with
`sha256sum "$AGENT_CONFIG_JSON" "$AGENT_OTEL_YAML"`, record them in the
reviewed task-definition manifest, and render each non-secret file as
single-line base64 plus its lowercase SHA-256 into the sidecar environment.
Base64 is transport encoding, not a credential or secrecy mechanism. The
sidecar entrypoint decodes both files into its task-local writable directory,
verifies both hashes before use, then runs the agent's required `fetch-config`
followed by `append-config` sequence. A mismatch or either control-script
failure stops the sidecar. The task definition must refer to that exact
manifest version; mutable “latest” configuration is not accepted.

CloudWatch Agent JSON (`elspeth.cloudwatch-agent.v1.json`):

```json
{
  "agent": {
    "metrics_collection_interval": 60,
    "usage_data": false
  }
}
```

Appended OpenTelemetry YAML (`elspeth.cloudwatch-agent.v1.otel.yaml`):

```yaml
receivers:
  otlp/elspeth:
    protocols:
      grpc:
        endpoint: 127.0.0.1:4317
processors:
  memory_limiter/elspeth:
    check_interval: 5s
    limit_mib: 128
    spike_limit_mib: 32
  batch/elspeth:
    send_batch_size: 512
    send_batch_max_size: 1024
    timeout: 5s
exporters:
  awsemf/elspeth:
    namespace: ELSPETH/Operator
    log_group_name: /elspeth/operator/metrics
    log_stream_name: telemetry
    dimension_rollup_option: NoDimensionRollup
    retain_initial_value_of_delta_metric: true
    resource_to_telemetry_conversion:
      enabled: true
  awsxray/elspeth: {}
service:
  pipelines:
    metrics/elspeth:
      receivers: [otlp/elspeth]
      processors: [memory_limiter/elspeth, batch/elspeth]
      exporters: [awsemf/elspeth]
    traces/elspeth:
      receivers: [otlp/elspeth]
      processors: [memory_limiter/elspeth, batch/elspeth]
      exporters: [awsxray/elspeth]
```

The suffix on every receiver, processor, exporter, and pipeline prevents
merge conflicts with pipelines generated by the CloudWatch Agent itself. The
`awsemf` exporter is a component shipped in the current agent: it converts the
bounded OTLP metrics to Embedded Metric Format, writes them to the pre-created
`/elspeth/operator/metrics` log group, and extracts them into the
`ELSPETH/Operator` CloudWatch namespace. `NoDimensionRollup` prevents the
exporter from silently creating additional dimension sets, and retaining the
first delta value preserves low-frequency acceptance and failure counters.
The `awsxray` exporter sends traces to X-Ray. Both exporters use the default
AWS credential chain and therefore the ECS task role; neither accepts an
endpoint, role override, profile, or static credential here.

Production permits only these two supported exporters. The unsupported
`awscloudwatch` collector exporter must not be used. Diagnostic `debug` output
and local-disk/file exporters are forbidden because they duplicate content
into an unreviewed retention surface.

## Task-definition shape

Resolve an approved CloudWatch Agent repository and its 64-lowercase-hex
digest into `CLOUDWATCH_AGENT_IMAGE_SHA256`. The rendered image reference must
contain the digest and no tag. The approved ECS runtime variant must include
the AWS control script plus `/bin/sh`, `base64`, `sha256sum`, `grep`, and
`sleep`; those are part of the reviewed image contract and are exercised by
the entrypoint below:

```json
{
  "containerDefinitions": [
    {
      "name": "cloudwatch-agent",
      "image": "${CLOUDWATCH_AGENT_IMAGE_REPOSITORY}@sha256:${CLOUDWATCH_AGENT_IMAGE_SHA256}",
      "essential": false,
      "memoryReservation": 192,
      "entryPoint": ["/bin/sh", "-ceu"],
      "command": ["CONFIG_DIR=/tmp/elspeth-cloudwatch-agent; CTL=/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl; mkdir -p \"$CONFIG_DIR\"; printf '%s' \"$ELSPETH_CW_AGENT_CONFIG_JSON_B64\" | base64 -d > \"/tmp/elspeth-cloudwatch-agent/elspeth.cloudwatch-agent.v1.json\"; printf '%s' \"$ELSPETH_CW_AGENT_OTEL_YAML_B64\" | base64 -d > \"/tmp/elspeth-cloudwatch-agent/elspeth.cloudwatch-agent.v1.otel.yaml\"; printf '%s\\n' \"$ELSPETH_CW_AGENT_CONFIG_JSON_SHA256  /tmp/elspeth-cloudwatch-agent/elspeth.cloudwatch-agent.v1.json\" | sha256sum -c -; printf '%s\\n' \"$ELSPETH_CW_AGENT_OTEL_YAML_SHA256  /tmp/elspeth-cloudwatch-agent/elspeth.cloudwatch-agent.v1.otel.yaml\" | sha256sum -c -; \"$CTL\" -a fetch-config -m auto -c \"file:/tmp/elspeth-cloudwatch-agent/elspeth.cloudwatch-agent.v1.json\" -s; \"$CTL\" -a append-config -m auto -c \"file:/tmp/elspeth-cloudwatch-agent/elspeth.cloudwatch-agent.v1.otel.yaml\" -s; while \"$CTL\" -a status -m auto | grep -q '\"status\": \"running\"'; do sleep 30; done; exit 1"],
      "environment": [
        {"name": "ELSPETH_CW_AGENT_CONFIG_JSON_B64", "value": "${CLOUDWATCH_AGENT_CONFIG_JSON_B64}"},
        {"name": "ELSPETH_CW_AGENT_CONFIG_JSON_SHA256", "value": "${CLOUDWATCH_AGENT_CONFIG_JSON_SHA256}"},
        {"name": "ELSPETH_CW_AGENT_OTEL_YAML_B64", "value": "${CLOUDWATCH_AGENT_OTEL_YAML_B64}"},
        {"name": "ELSPETH_CW_AGENT_OTEL_YAML_SHA256", "value": "${CLOUDWATCH_AGENT_OTEL_YAML_SHA256}"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a status -m auto | grep -q '\"status\": \"running\"'"],
        "interval": 10,
        "timeout": 5,
        "retries": 6,
        "startPeriod": 30
      }
    },
    {
      "name": "elspeth-web",
      "dependsOn": [{"containerName": "cloudwatch-agent", "condition": "HEALTHY"}],
      "environment": [
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY", "value": "aws-otlp"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_ENVIRONMENT", "value": "production"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_RELEASE", "value": "${ELSPETH_RELEASE_SHA_OR_DIGEST}"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_ECS_CLUSTER", "value": "${ECS_CLUSTER_NAME}"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_ECS_SERVICE", "value": "${ECS_SERVICE_NAME}"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_TASK_DEFINITION_FAMILY", "value": "${ECS_TASK_DEFINITION_FAMILY}"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_TASK_DEFINITION_REVISION", "value": "${ECS_TASK_DEFINITION_REVISION}"}
      ]
    }
  ]
}
```

Do not add a task port map for the collector. Fargate containers in the task
share the task network namespace, so the application reaches the loopback
receiver while no collector listener is published through the ENI, security
group, load balancer, or service discovery. The sidecar is required to become
healthy for initial application startup but remains non-essential afterward:
an observability outage degrades alarms without killing healthy audited work.

## IAM separation

Pre-create `/elspeth/operator/metrics`, apply the retention policy below, and
do not grant the task permission to create arbitrary log groups. The
**Task role** carries only application-time EMF and trace delivery. The EMF
exporter needs stream discovery/creation and event writes on that exact group;
X-Ray write APIs do not support narrower resource scoping:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PutElspethOperatorMetricEmf",
      "Effect": "Allow",
      "Action": ["logs:DescribeLogStreams", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": [
        "arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT}:log-group:/elspeth/operator/metrics",
        "arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT}:log-group:/elspeth/operator/metrics:log-stream:telemetry"
      ]
    },
    {
      "Sid": "PutElspethTraces",
      "Effect": "Allow",
      "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
      "Resource": "*"
    }
  ]
}
```

The **Execution role** remains limited to image pull, approved secret
injection, and the ECS log driver. It does not receive the EMF or trace
delivery actions. `cloudwatch:PutMetricData` is not required: `awsemf` delivers
EMF with `logs:PutLogEvents`, and CloudWatch extracts the configured namespace.
The application container does not call CloudWatch or X-Ray APIs; the sidecar
uses the task role.

## Dashboard and alarm contract

Publish the dashboard as version `elspeth-aws-operator-v1`. Every widget uses
the exact service/environment/release/task-definition resource filters. Its
panels cover run failures, run duration, external-call failures,
external-call latency, existing non-sensitive LLM token and LLM cost totals,
composer/runtime failures, export failure and queue drop rate, stale export
age, and missing sidecar signal.

Create these alarms with an owner action and the matching Landscape workflow:

| Alarm | Trigger | Owner action and Landscape correlation |
|---|---|---|
| RunFailureRate | non-zero failed-run rate for 5 minutes | Runtime owner queries Landscape terminal runs for the alarm window and treats Landscape as authoritative. |
| RunDurationP95 | approved p95 breached for 15 minutes | Runtime owner compares audited run timestamps and inspects slow nodes/calls. |
| ExternalCallFailureRate | closed failure reason rises for 5 minutes | Plugin owner queries audited external calls by provider/operation and verifies their recorded outcomes. |
| ExternalCallLatencyP95 | approved p95 breached for 15 minutes | Plugin owner compares audited call latency and provider status. |
| OperatorExportFailures | export failure or queue drop is non-zero for 5 minutes | Platform owner checks sidecar health and network/task-role delivery; business processing remains audit-led. |
| OperatorExportStale | last successful export age exceeds 180 seconds | Platform owner checks the sidecar and confirms current Landscape writes independently. |
| OperatorSignalMissing | missing sidecar signal for 180 seconds | On-call checks task health and collector logs, then queries Landscape for runs during the blind window. |

Alarm notifications contain only alarm name, service/environment identity,
time window, and the approved operator action. They contain no content,
credentials, exception messages, raw provider responses, or AWS identities.

## Cardinality, volume, cost, and retention

CloudWatch metric dimensions are a closed set: bounded service, environment,
release (`service.version`), ECS cluster/service, task-definition
family/revision, the constant cloud provider, and the closed operational
attributes enumerated below. User, session, pipeline-run, row, token, prompt,
content, URL, exception, account, task, and request identities are forbidden
as metric dimensions. Pipeline-run correlation remains a bounded trace
attribute used to return from an alarm to Landscape, never a metric dimension.

The versioned metric-dimension manifest consumed by deployment review is:

```json
{
  "schema": "elspeth.cloudwatch-dimensions.v1",
  "dimensions": [
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
    "to_mode"
  ]
}
```

Before deployment, the telemetry owner records the expected daily run count
`R`, admitted row count `N`, and external-call count `C`:

- `lifecycle`: approximately `2R` pipeline spans; this is the production
  default and lowest-cost profile.
- `rows`: approximately `2R + 3N` spans; it requires explicit operator cost
  acceptance before production enablement.
- `full`: approximately `2R + 3N + 2C` spans for generic CLI/batch use. The AWS
  web overlay rejects this content-bearing profile.
- Web metrics use a fixed instrument inventory and the closed dimensions
  above; no per-row or per-run time series may be created.

Set a monthly custom-metric/trace budget and alarms at 50%, 80%, and 100% of
that budget. Keep sidecar logs for 30 days unless the organisation approves a
shorter incident window; use the approved X-Ray trace retention and do not
copy traces into another store. Review retention and volume after the first
seven production days and after any move from lifecycle to rows.

## Acceptance and outage drill

Run the in-task operator-telemetry acceptance helper with a fresh non-content
sentinel. It must write and re-read the Landscape sentinel first, emit one
`operator.acceptance.sentinel` web metric and one `RunStarted`/`RunFinished`
lifecycle trace, then query CloudWatch/X-Ray with bounded retries. The durable
receipt contains only metric/trace name, observation timestamp, sanitized
resource identity, and the SHA-256 of the sentinel. Raw AWS service responses
are discarded.

For the negative lane, stop or install an invalid configuration in the
disposable sidecar only after the Landscape sentinel commits. The metric and
trace deliveries must report unavailable, operator telemetry health must
degrade within 180 seconds, the application must stay alive, and the
Landscape sentinel must remain correct. Any CloudWatch success receipt during
the forced outage is a false-receipt failure. Restore the exact reviewed
configuration and repeat the positive lane before promotion.

The deployment posture is explicitly Landscape-first: operational signals
help an operator find work, while Landscape remains the audit authority.

---

## Ordered rollout and rollback

The release operator owns every gate and records only sanitized resolved
inputs, immutable digests, timestamps, counts, booleans, and receipt hashes.
The database operator owns destructive schema action. Stop between every
numbered phase; a failed phase never falls through to the next one.

### 1. Resolve and validate inventory without service mutation

Set `DEPLOYMENT_MODE` to `upgrade`, `first`, or the narrow `first-recovery`.
`first-recovery` can restart only the same immutable candidate after a failed
first deploy left desired count zero and traffic fixed at 503. Validate these
inputs from the approved inventory:

- `TARGET_PLATFORM`, `AWS_REGION`, `ECS_CLUSTER`, `ECS_SERVICE`,
  `WEB_CONTAINER_NAME`, `TARGET_GROUP_ARN`, and `ALB_BASE_URL`;
- `CANDIDATE_TASK_DEFINITION`, `DOCTOR_TASK_DEFINITION`,
  `DOCTOR_CONTAINER_NAME`, and the complete JSON
  `DOCTOR_NETWORK_CONFIGURATION`;
- `WEB_LOG_GROUP`, `WEB_LOG_STREAM_PREFIX`, `DOCTOR_LOG_GROUP`, and
  `DOCTOR_LOG_STREAM_PREFIX`;
- `ECS_DEPLOYMENT_EVENT_RULE`, `ECS_DEPLOYMENT_EVENT_TARGET_ID`, and
  `ECS_DEPLOYMENT_EVENT_LOG_GROUP`;
- `PREVIOUS_TASK_DEFINITION` for upgrades; and
- `FIRST_DEPLOY_LISTENER_RULE_ARN`, `FIRST_DEPLOY_FORWARD_ACTIONS`, and
  `FIRST_DEPLOY_DISABLED_ACTIONS` for first/first-recovery.

`ECS_CLUSTER` and `ECS_SERVICE` are short cluster/service names matching
`^[A-Za-z0-9_-]+$`, never ARNs. Application Auto Scaling uses the literal
resource ID `service/$ECS_CLUSTER/$ECS_SERVICE`. Resolve every supplied task
definition and replace it with the returned exact taskDefinitionArn. Require
`ACTIVE`, the approved image digest, matching Linux CPU architecture, named
container, network configuration, log group, and stream prefix.

```bash
: "${DEPLOYMENT_MODE:?} ${TARGET_PLATFORM:?} ${AWS_REGION:?}"
: "${ECS_CLUSTER:?} ${ECS_SERVICE:?} ${WEB_CONTAINER_NAME:?}"
: "${TARGET_GROUP_ARN:?} ${ALB_BASE_URL:?}"
: "${CANDIDATE_TASK_DEFINITION:?} ${DOCTOR_TASK_DEFINITION:?}"
: "${DOCTOR_CONTAINER_NAME:?} ${DOCTOR_NETWORK_CONFIGURATION:?}"
: "${WEB_LOG_GROUP:?} ${WEB_LOG_STREAM_PREFIX:?}"
: "${DOCTOR_LOG_GROUP:?} ${DOCTOR_LOG_STREAM_PREFIX:?}"
: "${ECS_DEPLOYMENT_EVENT_RULE:?} ${ECS_DEPLOYMENT_EVENT_TARGET_ID:?}"
: "${ECS_DEPLOYMENT_EVENT_LOG_GROUP:?} ${ACCEPTANCE_RUN_ID:?}"

case "$DEPLOYMENT_MODE" in
  upgrade) : "${PREVIOUS_TASK_DEFINITION:?}" ;;
  first|first-recovery)
    : "${FIRST_DEPLOY_LISTENER_RULE_ARN:?}"
    : "${FIRST_DEPLOY_FORWARD_ACTIONS:?} ${FIRST_DEPLOY_DISABLED_ACTIONS:?}"
    ;;
  *) printf '%s\n' 'invalid DEPLOYMENT_MODE' >&2; exit 2 ;;
esac
[[ "$ECS_CLUSTER" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "$ECS_SERVICE" =~ ^[A-Za-z0-9_-]+$ ]]

OBSERVATION_START_EPOCH_MS=$(($(date +%s) * 1000))
SERVICE_JSON=$(aws_capture aws ecs describe-services \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" --output json)
SCALING_JSON=$(aws_capture aws application-autoscaling describe-scalable-targets \
  --service-namespace ecs \
  --resource-ids "service/$ECS_CLUSTER/$ECS_SERVICE" --output json)
jq -e '.failures | length == 0' <<<"$SERVICE_JSON" >/dev/null
jq -e '.ScalableTargets | length == 0' <<<"$SCALING_JSON" >/dev/null
unset SERVICE_JSON SCALING_JSON
```

The returned service ARN/name must agree with the short names. The
`ScalableTargets` array must be empty; any registered target is NO-GO. Require
the rolling `ECS` controller, Fargate 1.4.0 or `LATEST`,
`enableExecuteCommand == true`, zero-overlap deployment settings, current
`desiredCount == 1` for upgrade and zero for first modes. After launch,
`describe-tasks` must report the named container's
`ExecuteCommandAgent.lastStatus == RUNNING`.

Validate the ALB target group mechanically: `TargetType == "ip"`,
`HealthCheckEnabled == true`, `HealthCheckPath == "/api/ready"`,
`Matcher.HttpCode == "200"`, and `HealthCheckTimeoutSeconds >= 6`. Require
the container health command above, `startPeriod >= 150`, and service
`healthCheckGracePeriodSeconds >= 150`.

The EventBridge rule is enabled and has the exact ECS deployment-failure
pattern. Its CloudWatch Logs target has no `RoleArn`. Use
`aws logs describe-resource-policies` to require only
`logs:CreateLogStream`/`logs:PutLogEvents` for `events.amazonaws.com` and
`delivery.logs.amazonaws.com` on the exact log destination. Never spoof
AWS-owned `aws.ecs`. Create a temporary custom-source canary rule named from
`ACCEPTANCE_RUN_ID`, deliver a uniquely correlated event to the same log
group, retain only a sanitized receipt, and remove and prove absence of that
canary before deployment.

### 2. Run the hardened one-shot doctor

Use the candidate digest in the doctor definition. The schema-owner variant is
used only for a database-operator-approved `--init-schema` action.

```bash
DOCTOR_OVERRIDES=$(jq -cn --arg name "$DOCTOR_CONTAINER_NAME" \
  '{containerOverrides:[{name:$name,command:["doctor","aws-ecs","--json"]}]}')

RUN_TASK_JSON=$(aws_capture aws ecs run-task \
  --cluster "$ECS_CLUSTER" \
  --task-definition "$DOCTOR_TASK_DEFINITION" \
  --launch-type FARGATE \
  --network-configuration "$DOCTOR_NETWORK_CONFIGURATION" \
  --count 1 \
  --overrides "$DOCTOR_OVERRIDES" \
  --output json)
jq -e '(.failures | length) == 0 and (.tasks | length) == 1' \
  <<<"$RUN_TASK_JSON" >/dev/null
DOCTOR_TASK_ARN=$(jq -er '.tasks[0].taskArn | select(length > 0)' \
  <<<"$RUN_TASK_JSON")
unset RUN_TASK_JSON

aws_capture aws ecs wait tasks-stopped \
  --cluster "$ECS_CLUSTER" --tasks "$DOCTOR_TASK_ARN" >/dev/null
DOCTOR_RESULT=$(aws_capture aws ecs describe-tasks \
  --cluster "$ECS_CLUSTER" --tasks "$DOCTOR_TASK_ARN" --output json)
ESSENTIAL_NAMES=$(aws_capture aws ecs describe-task-definition \
  --task-definition "$DOCTOR_TASK_DEFINITION" --output json \
  | jq -c '[.taskDefinition.containerDefinitions[] | select(.essential == true) | .name]')
jq -e --arg arn "$DOCTOR_TASK_ARN" --argjson essential "$ESSENTIAL_NAMES" '
  (.failures | length) == 0
  and .tasks[0].taskArn == $arn
  and ([.tasks[0].containers[] | select(.name as $n | $essential | index($n))
        | .exitCode == 0] | all)
' <<<"$DOCTOR_RESULT" >/dev/null
unset DOCTOR_RESULT ESSENTIAL_NAMES
```

Missing/null `exitCode`, a name mismatch, non-zero essential-container exit,
or sanitized doctor failure blocks service mutation. Diagnose only through
bounded `aws logs filter-log-events` calls captured by `aws_capture` and sent
directly to `sanitize-evidence`; raw logs are never printed or persisted.

### 3. Apply the schema compatibility gate

`--init-schema` may initialize the session schema only when it is MISSING; a
partially present session schema is STALE. It may initialize the landscape
schema when MISSING or complete verified-shape PARTIAL state, except a
non-empty pre-epoch-23 Landscape missing `run_web_plugin_policy` is STALE and
must not be additively repaired. Aurora detection is structural-only: a
semantics-only schema-epoch change can still appear CURRENT.

Attach the approved release/schema compatibility record before deployment.
AWS ECS validate-only startup must fail closed before uvicorn binds for
missing, partial, stale, or incompatible state. Code rollback cannot undo an
incompatible database schema. STALE/incompatible state requires the
database-operator-owned archive decision and drop/recreate procedure followed
by `--init-schema`; never automate it. Code rollback to epoch 22 after an
epoch-23 recreation is unsafe.

### 4. Deploy exactly one candidate task

Capture the previous primary deployment ID/time. Changing desired count alone
does not prove a new deployment. Every intentional launch, relaunch, upgrade,
or manual upgrade rollback uses a forced new deployment:

```bash
aws_capture aws ecs update-service \
  --cluster "$ECS_CLUSTER" \
  --service "$ECS_SERVICE" \
  --task-definition "$CANDIDATE_TASK_DEFINITION" \
  --desired-count 1 \
  --force-new-deployment \
  --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
  >/dev/null

aws_capture aws ecs wait services-stable \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
```

A non-zero waiter, `rolloutState=FAILED`, `SERVICE_DEPLOYMENT_FAILED`, absent
distinct primary deployment/task identity, or more than one running/pending
task fails rollout. For first modes, automatic rollback has no different
known-good revision; keep traffic fixed at 503 and scale to zero.

### 5. Perform candidate-aware acceptance before traffic

Require one completed primary deployment on the candidate definition,
`desiredCount == 1`, `runningCount == 1`, and `pendingCount == 0`. Use
`aws ecs list-tasks` and exact-ARN `aws ecs describe-tasks`; reject any second
running or pending task. Resolve the service `loadBalancers` entry for
`TARGET_GROUP_ARN`, then the named container's ENI `privateIpv4Address` and
container port. The exact `(Id,Port)` pair must be healthy. After zero-overlap
replacement, any old target may be ignored only while `draining`, never while
serving.

Only then, in first/first-recovery, change
`FIRST_DEPLOY_LISTENER_RULE_ARN` from `FIRST_DEPLOY_DISABLED_ACTIONS` to
`FIRST_DEPLOY_FORWARD_ACTIONS` and verify it forwards only to
`TARGET_GROUP_ARN`.

```bash
health_body=$(mktemp)
ready_body=$(mktemp)
chmod 600 "$health_body" "$ready_body"
trap 'rm -f "$health_body" "$ready_body"' EXIT

health_status=$(curl --connect-timeout 5 --max-time 10 --max-redirs 0 \
  -sS -o "$health_body" -w '%{http_code}' "$ALB_BASE_URL/api/health")
ready_status=$(curl --connect-timeout 5 --max-time 10 --max-redirs 0 \
  -sS -o "$ready_body" -w '%{http_code}' "$ALB_BASE_URL/api/ready")
[[ "$health_status" == 200 ]]
[[ "$ready_status" == 200 ]]
jq -e '.ready == true' "$ready_body" >/dev/null
```

### 6. Observe for ten minutes

Set `ACCEPTANCE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)` immediately after
the first complete pass. Poll every 30 seconds for 20 iterations, repeating
the exact service/deployment counts, candidate task/target mapping, target
health, and exact-200 bounded `/api/health` and `/api/ready` checks.

```bash
ACCEPTANCE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
for iteration in $(seq 1 20); do
  printf 'acceptance_sample=%s\n' "$iteration"
  SERVICE_SAMPLE=$(aws_capture aws ecs describe-services \
    --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" --output json)
  jq -e --arg task_definition "$CANDIDATE_TASK_DEFINITION" '
    .services[0]
    | .desiredCount == 1 and .runningCount == 1 and .pendingCount == 0
      and (.deployments | length) == 1
      and .deployments[0].taskDefinition == $task_definition
      and .deployments[0].rolloutState == "COMPLETED"
  ' <<<"$SERVICE_SAMPLE" >/dev/null
  unset SERVICE_SAMPLE
  sleep 30
done
```

`SERVICE_DEPLOYMENT_FAILED`, `rolloutState=FAILED`, launch/placement failures,
non-zero essential exits, or a `stoppedReason` require rollback and stopped
task inspection. Candidate `Target.ResponseCodeMismatch`, `Target.Timeout`, or
`Target.FailedHealthChecks` require diagnosis and rollback if persistent.
Non-200 liveness/readiness, `ready != true`, `readiness_check_not_ready`,
startup/schema failure, or new unhandled `ERROR`/`CRITICAL` blocks acceptance.
Retain only allowlisted checks, classes, counts, and hashes.

### 7. Roll back without crossing the schema stop

For an upgrade, first prove the release/schema compatibility record permits
the previous code on the current schema, then force a distinct deployment:

```bash
aws_capture aws ecs update-service \
  --cluster "$ECS_CLUSTER" \
  --service "$ECS_SERVICE" \
  --task-definition "$PREVIOUS_TASK_DEFINITION" \
  --desired-count 1 \
  --force-new-deployment \
  --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
  >/dev/null
aws_capture aws ecs wait services-stable \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
```

Repeat exact-one-task, previous-revision target mapping/health, public probes,
CloudWatch, and EventBridge checks. If backward compatibility is unknown, keep
traffic drained and escalate instead of rolling old code over the schema.

For first/first-recovery, remove traffic before compute, verify the listener's
fixed 503 action, then scale to zero:

```bash
aws_capture aws elbv2 modify-rule \
  --rule-arn "$FIRST_DEPLOY_LISTENER_RULE_ARN" \
  --actions "$FIRST_DEPLOY_DISABLED_ACTIONS" >/dev/null
aws_capture aws ecs update-service \
  --cluster "$ECS_CLUSTER" \
  --service "$ECS_SERVICE" \
  --desired-count 0 \
  --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
  >/dev/null
aws_capture aws ecs wait services-stable \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
```

Require zero running/pending tasks and no non-draining registered targets.
Keep failed task-definition, stopped-task, event, and sanitized log receipts
until the incident record is complete.

## Release/schema compatibility record

The database operator is the author/approver and the release operator is the
countersigner before doctor/deploy. Store this record in the approved change
system; Plan 12 binds its ID to both scenarios.

```yaml
record_id: change-record-id
timestamp_utc: 2026-01-01T00:00:00Z
candidate git SHA: 40-lowercase-hex
immutable image digest: sha256:64-lowercase-hex
candidate_task_definition_arn: exact-arn
doctor_task_definition_arn: exact-arn
candidate_package_version: 0.7.0
candidate_session_schema_epoch: approved-value
candidate_landscape_schema_epoch: 23
run_web_plugin_policy_present: true
structural_schema_changes: reviewed-summary
semantics-only changes: reviewed-summary
archive_export_decision: approved-value
destructive_reset_required: false
previous_identity: exact-digest-and-task-definition-or-not_applicable
previous_schema_epochs: approved-values-or-not_applicable
forward_compatible: yes-or-no
backward_compatible: yes-or-no-or-not_applicable
rollback_permitted: yes-or-no
rollback_evidence: approved-reference
code_schema_diff_inspector: operator-identity
database_operator_approval: operator-identity-and-timestamp
release_operator_countersignature: operator-identity-and-timestamp
expiry_or_supersession: timestamp-or-record-id
```

Unknown or unapproved compatibility is NO-GO.

## Disposable acceptance cleanup

Before either Terraform scenario is created, record infrastructure owner,
identity owner, sanitized evidence destination, non-secret UUID
`ACCEPTANCE_RUN_ID`, and a teardown deadline no later than four hours after
the live gate. Tag every disposable resource. Promotion is forbidden before Plan 12 final GO
and while cleanup is pending.

On success, failure, interruption, or timeout:

1. Export only approved sanitized evidence.
2. Destroy both Terraform stacks and verify separately bound state is empty.
3. Delete/rotate the Cognito test identity and test secrets.
4. Delete the rollback-baseline ECR tag and candidate acceptance ECR tag.
5. Run the closed `orphan-sweep` across ECS, ALB, Aurora, EFS, Secrets
   Manager, CloudWatch Logs, EventBridge, Cognito, ECR, and Guardrails.
6. Clear `CLEANUP_REQUIRED=0` only after all independent surfaces pass.

```bash
python -m elspeth.web.aws_ecs_acceptance orphan-sweep \
  --control-manifest "$CONTROL_MANIFEST" \
  --scenario all \
  >"$SANITIZED_ORPHAN_RECEIPT"

python -m elspeth.web.aws_ecs_acceptance control-manifest validate \
  --file "$CONTROL_MANIFEST" \
  --cleanup-only \
  --require-cleanup-cleared
```

Terraform state-empty alone is insufficient after partial apply. The orphan
sweep returns non-zero for an empty/malformed run ID, any API failure, or any
unapproved survivor. Run-scoped ECS task definitions require no `ACTIVE`
revision, explicit deregister/delete receipts, zero dependants, and either
absence or owner-tracked `DELETE_IN_PROGRESS` with a 24-hour poll/escalation
deadline. Cleanup failure is itself NO-GO and escalates to the named owner.

Durable lean-image publication is separate release-owner work after GO: build
the GO SHA for each approved platform, verify SBOM/provenance/signature and
vulnerability policy, and repeat live acceptance for any rebuilt digest.
Mere retagging of the temporary candidate is not publication evidence.

## See also

- [AWS ECS health and readiness](../operator/aws-ecs-health-and-readiness.md)
- [Configure Azure Key Vault](configure-keyvault-secrets.md)
- [Docker guide](../guides/docker.md)
