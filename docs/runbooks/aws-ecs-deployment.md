# AWS ECS deployment

## Contract summary

This is the canonical operator entry point for the AWS ECS deployment. The
application and collector have deliberately separate evidence roles:

- Landscape is the permanent source of truth for lineage, replay, and run
  decisions. Its write must succeed before an operational signal is emitted.
- CloudWatch and X-Ray are best-effort operational telemetry. A receipt there
  never proves an audit write, and loss of the collector never rolls back a
  committed Landscape record.

AWS web tasks set `ELSPETH_WEB__OPERATOR_TELEMETRY=aws-otlp`, an explicit
`ELSPETH_WEB__OPERATOR_TELEMETRY_ENVIRONMENT`, and optionally the bounded
service name and 1–3600 second export interval. The receiver is fixed by the
application at `http://127.0.0.1:4317`; it is not an operator endpoint setting.
Uploaded pipeline telemetry routing is replaced with one lifecycle-or-rows
OTLP exporter, empty headers, and best-effort failure policy.

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

## Versioned collector configuration

Store these two files in the deployment repository under a versioned
`telemetry/elspeth.cloudwatch-agent.v1/` directory. Compute both digests with
`sha256sum "$AGENT_CONFIG_JSON" "$AGENT_OTEL_YAML"`, record them in the
reviewed task-definition manifest, and have the sidecar entrypoint verify the
files before starting. A mismatch is a deployment failure. The task definition
must refer to that exact manifest version; mutable “latest” configuration is
not accepted.

CloudWatch Agent JSON (`elspeth.cloudwatch-agent.v1.json`):

```json
{
  "schema": "elspeth.cloudwatch-agent.v1",
  "agent": {"metrics_collection_interval": 60},
  "metrics": {
    "namespace": "ELSPETH/Operator",
    "force_flush_interval": 5
  },
  "logs": {
    "force_flush_interval": 5,
    "log_stream_name": "elspeth-telemetry-sidecar"
  }
}
```

Appended OpenTelemetry YAML (`elspeth.cloudwatch-agent.v1.otel.yaml`):

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 127.0.0.1:4317
processors:
  memory_limiter:
    check_interval: 5s
    limit_mib: 128
    spike_limit_mib: 32
  batch:
    send_batch_size: 512
    send_batch_max_size: 1024
    timeout: 5s
exporters:
  awscloudwatch:
    namespace: ELSPETH/Operator
  awsxray: {}
service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [awscloudwatch]
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [awsxray]
```

Production permits only the two exporters above. Diagnostic console output
and local-disk exporters are forbidden because they duplicate content into an
unreviewed retention surface.

## Task-definition shape

Resolve an approved CloudWatch Agent repository and its 64-lowercase-hex
digest into `CLOUDWATCH_AGENT_IMAGE_SHA256`. The rendered image reference must
contain the digest and no tag:

```json
{
  "containerDefinitions": [
    {
      "name": "cloudwatch-agent",
      "image": "${CLOUDWATCH_AGENT_IMAGE_REPOSITORY}@sha256:${CLOUDWATCH_AGENT_IMAGE_SHA256}",
      "essential": false,
      "memoryReservation": 192,
      "healthCheck": {
        "command": ["CMD-SHELL", "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -m ecs -a status >/dev/null"],
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
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_ENVIRONMENT", "value": "production"}
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

The **Task role** carries only application-time delivery. Use the approved
namespace condition and exact log-group resources; X-Ray write APIs do not
support narrower resource scoping:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PutElspethOperatorMetrics",
      "Effect": "Allow",
      "Action": "cloudwatch:PutMetricData",
      "Resource": "*",
      "Condition": {"StringEquals": {"cloudwatch:namespace": "ELSPETH/Operator"}}
    },
    {
      "Sid": "PutElspethTraces",
      "Effect": "Allow",
      "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
      "Resource": "*"
    },
    {
      "Sid": "PutTelemetrySidecarLogs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT}:log-group:/elspeth/operator:*"
    }
  ]
}
```

The **Execution role** remains limited to image pull, approved secret
injection, and the ECS log driver. It does not receive the metric or trace
actions. The application container does not call CloudWatch or X-Ray APIs;
the sidecar uses the task role.

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
release, task-definition family/revision, and existing closed
reason/operation/status/outcome enums. User, session, pipeline-run, row,
token, prompt, content, URL, exception, account, task, and request identities
are forbidden as metric dimensions. Pipeline-run correlation remains a
bounded trace attribute used to return from an alarm to Landscape, never a
metric dimension.

The versioned metric-dimension manifest consumed by deployment review is:

```json
{
  "schema": "elspeth.cloudwatch-dimensions.v1",
  "dimensions": [
    "service.name",
    "deployment.environment",
    "service.version",
    "task.definition.family",
    "task.definition.revision",
    "reason",
    "operation",
    "status",
    "outcome"
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
