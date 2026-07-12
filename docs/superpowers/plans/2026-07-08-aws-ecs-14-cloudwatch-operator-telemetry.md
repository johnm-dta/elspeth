# AWS ECS CloudWatch Operator Telemetry Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:using-git-worktrees, superpowers:test-driven-development,
> logging-telemetry-policy, and superpowers:executing-plans. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Give AWS ECS operators a supported, task-role-authenticated CloudWatch
telemetry path for ELSPETH pipeline events and web/operator metrics without
weakening Landscape audit primacy or exposing a new arbitrary-egress surface.

**Problem:** ELSPETH already has two complementary observability surfaces:

1. `LandscapeDB` is the durable, complete audit and lineage record.
2. Pipeline telemetry has built-in `console`, generic `otlp`,
   `azure_monitor`, and `datadog` exporters, while the web process publishes
   OpenTelemetry metrics through an authenticated Prometheus `/metrics` route.

The AWS ECS program currently deploys the Landscape database and captures
container logs, ECS events, and a few infrastructure metrics, but it never
configures or live-proves application telemetry. The generic OTLP exporter can
reach an AWS collector, but no AWS connector contract stamps useful resource
identity, prevents credential/header/remote-endpoint configuration, or tells
operators how the pipeline spans and web metrics arrive in CloudWatch. The
current synthetic OTLP spans also use an empty resource, which makes production
service/environment attribution too weak for dependable dashboards and alarms.

**Architecture:** Keep the signals and their authority separate:

- **Landscape is Tier 1 and authoritative.** Existing engine and plugin paths
  record the auditable event first. No CloudWatch success may compensate for a
  Landscape failure, and CloudWatch is never evidence for lineage or replay.
- **CloudWatch is Tier 2 and operational.** Reuse the built-in generic `otlp`
  exporter to emit the existing telemetry-event spans over OTLP/gRPC to a local
  collector. An AWS ECS operator overlay replaces pipeline-authored telemetry
  routing at the web execution boundary; this is a deployment policy over the
  vendor-neutral transport, not a redundant CloudWatch SDK exporter or a second
  audit path. CLI/batch operation retains the generic exporter behavior.
- Run the AWS-supported CloudWatch Agent as a non-essential ECS sidecar pinned
  by image digest. Its appended OpenTelemetry configuration exposes one
  task-local OTLP receiver and exports metrics and traces to CloudWatch/X-Ray
  through the ECS task role. The task definition publishes no collector port.
- Move web meter-provider construction out of module import and into one
  process bootstrap that retains the Prometheus reader and, in AWS ECS mode,
  adds a periodic OTLP metric reader targeting the same task-local receiver.
  `/metrics` remains authenticated and is not scraped using a bearer token.
- Resource identity is bounded and non-content-bearing: service name/version,
  deployment environment, cloud provider, ECS cluster/service, task-definition
  family/revision, and release SHA/digest. Run, row, user, prompt, URL, token,
  payload, exception-message, AWS account, and request IDs are forbidden as
  CloudWatch metric dimensions. Pipeline run IDs remain trace attributes only
  where already required for alert-to-Landscape correlation.
- Telemetry delivery remains best effort, but it is observable: exporter
  failures, queue drops, disabled exporters, and sidecar health produce bounded
  counters/alarms. The AWS operator overlay fixes
  `fail_on_total_exporter_failure=false`; losing Tier-2 telemetry cannot roll
  back or fail Tier-1 audited business processing. Generic CLI/batch telemetry
  retains its existing configurable policy.

This follows AWS's current recommendation to receive OpenTelemetry metrics and
traces through the CloudWatch Agent, and the agent's supported OTLP receiver
and CloudWatch/X-Ray exporters. The sidecar uses only the default AWS credential
chain/task role; ELSPETH never accepts or forwards AWS keys, profiles, role
ARNs, SigV4 material, or CloudWatch API endpoints.

**Depends on:** Plan 13 and the shared signed-tier/trust-boundary/Wardline
baseline `elspeth-8166b310e7`. Plan 13 is a deliberate file-order dependency:
both plans edit `web/config.py`, `web/app.py`, their tests, and the deployment
runbook contract. Plan 14 must be committed before Plan 10 records the
pre-packaging rollback baseline. Plan 10's task definition/runbook/harness and
Plan 12's live closeout depend on this plan.

**Primary references:** AWS CloudWatch, “Collect metrics and traces with
OpenTelemetry”; AWS CloudWatch, “Amazon CloudWatch agent”; AWS Distro for
OpenTelemetry, ECS sidecar deployment guidance; OpenTelemetry semantic
conventions for resource attributes.

**Tech Stack:** OpenTelemetry Python SDK/OTLP gRPC exporter, CloudWatch Agent,
CloudWatch Metrics/Transaction Search, AWS X-Ray, ECS/Fargate, Pydantic v2,
FastAPI, pytest, Wardline.

---

### Task 0: Atomic ownership and current-surface inventory

- [ ] From the repository root, require `release/0.7.1`, preserve unrelated
  changes, and create a clean isolated implementation worktree at the live
  release tip. Do not stash, reset, or stage another worker's files.
- [ ] Read the Plan 14 Filigree step and require Plan 13 plus
  `elspeth-8166b310e7` closed. Atomically start the exact step with
  `filigree start-work ...`; do not use claim-plus-status.
- [ ] Record `BASE_SHA=$(git rev-parse HEAD)`, the pre-existing dirty path set,
  and the current built-in exporter registry. Re-read Plans 10 and 12 before
  editing so task-definition, IAM, evidence, and rollback contracts remain
  aligned.
- [ ] Pin the authority split in tests and docs before implementation:
  Landscape is permanent/source-of-truth/must-fire; telemetry is
  operational/ephemeral/best-effort. No test may treat a CloudWatch receipt as
  proof that an audit write occurred.

---

### Task 1: Harden generic OTLP for the AWS operator profile

**Files:**

- Modify: `src/elspeth/telemetry/exporters/otlp.py`
- Modify: `src/elspeth/telemetry/serialization.py`
- Modify: `src/elspeth/telemetry/manager.py`
- Modify: `tests/unit/telemetry/exporters/test_otlp.py`
- Modify: `tests/unit/telemetry/test_factory.py`
- Modify: `tests/unit/telemetry/test_manager.py`
- Modify: `tests/unit/telemetry/test_plugin_wiring.py`
- Modify: `docs/guides/telemetry.md`

**Interface:**

```yaml
telemetry:
  enabled: true
  granularity: lifecycle
  fail_on_total_exporter_failure: true
  exporters:
    - name: otlp
      options:
        endpoint: http://127.0.0.1:4317
        service_name: elspeth
        deployment_environment: production
        batch_size: 100
```

Generic `otlp` gains bounded resource attributes while retaining its documented
endpoint/header behavior for CLI and non-AWS deployments. AWS web execution
does not trust the uploaded `telemetry` block: Task 2 replaces it with an
operator-owned effective config whose endpoint is exactly
`http://127.0.0.1:4317`, headers are empty, granularity is `lifecycle` or
`rows`, and only service/version/environment/batch values come from trusted
deployment settings. It rejects `full` because current full-granularity events
can contain LLM/HTTP request and response content.

- [ ] RED: table-drive bounded resource config validation. Reject blank or
  control-bearing identity fields, overlong values, booleans as integers, and
  invalid batch sizes. Add endpoint parser differentials: reject userinfo,
  query, fragment, control characters, and malformed URLs. Errors name only the
  field/check and never echo endpoint or raw input.
- [ ] RED: prove the built-in registry remains `console`, `otlp`,
  `azure_monitor`, and `datadog`; no boto3/botocore/AWS service SDK is added to
  the exporter. The CloudWatch Agent, not ELSPETH, owns SigV4 and AWS calls.
- [ ] RED: construct a real synthetic span and assert its resource contains
  bounded `service.name`, optional `service.version`,
  `deployment.environment`, and `cloud.provider=aws`. Assert status semantics
  distinguish a failed `RunFinished`/external-call result from a successful
  event without copying raw error text into status description or attributes.
- [ ] RED: pin the attribute allowlist. Content values, prompts, response text,
  secrets, URLs, local paths, exception messages, bearer material, and raw AWS
  identifiers must not appear in an exported span. Existing approved event
  attributes remain available, including bounded run correlation.
- [ ] Extend the generic OTLP span builder with an explicit immutable Resource
  and status input. Do not make the generic exporter silently AWS-specific and
  do not create another exporter thread/queue outside `TelemetryManager`.
- [ ] Correct delivery accounting: buffered/enqueued is not delivered. A failed
  batch that is cleared increments failed/dropped counts for every member, and
  health metrics expose attempted, delivered, failed, dropped, last-success,
  and consecutive-failure facts without double counting.
- [ ] RED/GREEN failure tests: OTLP `FAILURE`, timeout, unavailable sidecar,
  flush failure, and close failure feed the existing circuit-breaker and health
  metrics. Programming/type errors still escape; transport failures never
  masquerade as successful delivery.
- [ ] Add the AWS example and authority warning to the telemetry guide. State
  that CLI/batch authors opt into lifecycle/row/full event volume, while the
  AWS web overlay is operator-owned, mandatory, and ignores uploaded telemetry
  routing; production AWS permits lifecycle/rows only.
- [ ] Run the focused exporter/factory/manager/plugin-wiring suites and the
  serialization property tests. Recompute any required plugin source hash and
  generated catalog golden with repository tooling; never hand-edit them.
- [ ] Stage only Task 1 files and commit
  `feat(telemetry): harden OTLP for AWS operator telemetry`.

---

### Task 2: Export operator web metrics over OTLP without weakening `/metrics`

**Files:**

- Create: `src/elspeth/web/operator_telemetry.py`
- Modify: `src/elspeth/web/config.py`
- Modify: `src/elspeth/web/app.py`
- Modify: `src/elspeth/web/execution/service.py`
- Modify: `src/elspeth/web/sessions/telemetry.py`
- Create: `tests/unit/web/test_operator_telemetry.py`
- Modify: `tests/unit/web/test_config.py`
- Modify: `tests/unit/web/test_app.py`

**Interfaces:**

```python
class WebSettings:
    operator_telemetry: Literal["prometheus", "aws-otlp"] = "prometheus"
    operator_telemetry_service_name: str = "elspeth-web"
    operator_telemetry_environment: str | None = None
    operator_telemetry_export_interval_seconds: int = 60
    operator_pipeline_telemetry_granularity: Literal["lifecycle", "rows"] = "lifecycle"
```

In `deployment_target="aws-ecs"`, `operator_telemetry` must be `aws-otlp` and
the environment must be an explicit bounded value. The OTLP destination is not
a setting: it is the fixed task-local receiver.

- [ ] RED settings/deployment-contract tests: local default remains
  Prometheus-only; AWS ECS rejects an explicit off/prometheus mode, missing
  environment, unknown modes, bad intervals, blank/control/oversized resource
  names, and any attempted endpoint/header/credential fields. Static errors do
  not contain the rejected value.
- [ ] RED execution tests: in local mode the submitted pipeline telemetry block
  is unchanged. In AWS ECS web mode, disabled telemetry, console/Azure/Datadog,
  remote OTLP endpoints, headers, `full` granularity, and custom failure policy
  are all replaced by one enabled OTLP exporter using the fixed loopback
  endpoint, `fail_on_total_exporter_failure=false`, and operator settings.
  Persist the sanitized effective telemetry
  config/hash in Landscape before the first telemetry event so an investigation
  can distinguish authored from effective policy without storing secrets.
- [ ] RED bootstrap tests around a resettable provider seam. One process gets
  exactly one `MeterProvider`; local mode has one `PrometheusMetricReader`;
  AWS mode has that reader plus one `PeriodicExportingMetricReader` using an
  OTLP metric exporter to task-local `127.0.0.1:4317`. Repeated `create_app`
  calls in tests do not replace the global provider or duplicate export.
- [ ] Move the import-time provider side effect from `app.py` behind an
  idempotent bootstrap that runs before application instruments are created.
  Keep the existing counters/histograms and authenticated `/metrics` response
  byte-for-byte compatible. Do not add an unauthenticated loopback scrape route
  and do not put a long-lived bearer token in collector configuration.
- [ ] Add safe resource attributes and a view/cardinality contract. Metric
  dimensions may include service/environment/release/task-definition and
  existing closed reason/operation enums. They may not include user/session/run/
  row/token IDs, prompts, content, URLs, exception strings, account numbers,
  task ARNs, or request IDs.
- [ ] Add export-health instruments for last successful export age, failures,
  queue drops, and collector unavailability. Recording these metrics must not
  recurse through the failing exporter or flood logs; use bounded aggregate
  logging and the existing health semantics.
- [ ] RED/GREEN lifespan shutdown tests: force-flush and shutdown are bounded,
  called once, and cannot corrupt an already committed Landscape record.
  Programming failures still fail tests; expected exporter/collector outages
  are static, redacted operational failures.
- [ ] Run focused web config/app/session telemetry tests and existing `/metrics`
  authentication/exposition regressions, plus web execution overlay tests.
- [ ] Stage only Task 2 files and commit
  `feat(web): export operator metrics to task-local OTLP`.

---

### Task 3: ECS CloudWatch Agent sidecar, IAM, dashboards, and alarms

**Files:**

- Modify: `docs/runbooks/aws-ecs-deployment.md`
- Modify: `tests/unit/web/test_aws_ecs_runbook_contract.py`
- Modify: `src/elspeth/web/aws_ecs_acceptance.py`
- Modify: `tests/unit/web/test_aws_ecs_acceptance.py`
- Modify: `docs/release/guarantees.md`

- [ ] Add a CloudWatch Agent sidecar pinned by immutable image digest. It is
  non-essential so an observability outage does not kill a healthy audited
  run, but the application container depends on the agent reaching `HEALTHY`
  at initial startup. No collector port is mapped outside the task; the OTLP
  receiver is reachable only within the task network namespace.
- [ ] Store the agent's JSON and appended OTel YAML in an operator-controlled,
  versioned deployment artifact with a SHA-256 pinned in the task definition.
  The configuration has one OTLP receiver and bounded batch/memory-limit
  processors; metrics go to the approved CloudWatch namespace and traces to
  the approved CloudWatch/X-Ray destination. Debug/file exporters are forbidden
  in production.
- [ ] Extend task-role least privilege for only the required CloudWatch/X-Ray
  put actions and exact log groups/namespaces where scoping is supported.
  Execution-role and task-role duties remain separate. No static AWS key,
  profile, role override, endpoint override, or credential secret is present
  in app or sidecar environment/config.
- [ ] Add a versioned dashboard and alarms for run failures/duration,
  external-call failures/latency, LLM token/cost counters where already
  non-sensitive, composer/runtime failures, telemetry exporter failure/drop
  rate, stale export age, and missing sidecar signal. Every alarm names an
  owner action and Landscape query/correlation workflow.
- [ ] Add cardinality/cost budgets and retention. A test enumerates every
  metric dimension and rejects unbounded identity/content dimensions. Document
  expected custom-metric and trace volume for lifecycle/rows/full telemetry;
  production defaults to lifecycle unless an operator explicitly accepts the
  rows/full cost.
- [ ] Update the acceptance helper to emit unique non-content sentinels through
  one web metric and one pipeline lifecycle trace, then query CloudWatch/X-Ray
  with bounded retries. Persist only metric/trace name, timestamp, sanitized
  resource identity, and sentinel hash—not raw service responses.
- [ ] Add a negative lane: stop or misconfigure the disposable sidecar after a
  successful audit write. Prove Landscape remains correct, telemetry health
  becomes degraded/alertable within the bound, and no process crash or false
  CloudWatch receipt occurs.
- [ ] Run runbook-contract and acceptance-helper unit tests. Stage only Task 3
  files and commit `docs(aws-ecs): add CloudWatch telemetry sidecar and alarms`.

---

### Task 4: Integrated gates and Filigree handoff

- [ ] Run focused telemetry, web, deployment-contract, runbook, and acceptance
  suites, then Ruff format/check and mypy over changed surfaces.
- [ ] Run plugin hashes/catalog contracts and the logging/telemetry policy
  tests. Mechanically prove audit-first ordering at every touched path and that
  telemetry-only exceptions contain no probative run decision.
- [ ] Run signed-tier/trust-boundary diagnosis. Operators alone repair/sign
  diagnosed allowlist rows; agents never receive signing keys or hand-edit
  signatures.
- [ ] Run `wardline scan . --fail-on ERROR`. Fix findings at the input boundary;
  do not baseline or waive Plan 14 findings.
- [ ] Run pre-commit on the exact staged paths, prove changed-path equals
  staged-path, commit the final Task 4 corrections, and require a clean Plan 14
  worktree.
- [ ] Add a Filigree comment with commits, exact tests, gate results, and the
  Plan 10/12 handoff; close only the Plan 14 step. Do not start Plan 10.

**Acceptance handoff:** Plan 10 must include the pinned CloudWatch Agent
sidecar, IAM, dashboard/alarm/runbook, and bounded acceptance helper in the
candidate task definition. Plan 12 must prove, with zero skips, both a pipeline
lifecycle trace and the operator web metrics in CloudWatch on the candidate
digest, plus the negative audit-survives-telemetry-outage lane. Missing signals,
unbounded dimensions, raw-content evidence, sidecar credentials, or treating
CloudWatch as audit evidence is NO-GO.
