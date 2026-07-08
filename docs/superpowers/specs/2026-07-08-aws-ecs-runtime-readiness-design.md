# AWS ECS Runtime Readiness Design

Date: 2026-07-08
Status: approved design, pending implementation plan
Branch context: `release/0.7.0`

## Purpose

Make Elspeth usable in a Terraform-provisioned AWS environment without adding
Terraform to this repository. The target runtime is a single ECS Fargate web
task using Aurora Serverless v2 PostgreSQL for relational state, EFS for app
local state, and AWS Secrets Manager values injected by the ECS task
definition.

This design also adds AWS storage parity for pipeline authors by introducing
`aws_s3` source and sink plugins equivalent in capability to the current
`azure_blob` source and sink plugins.

## Scope

The implementation plan that follows this design should cover application and
runtime integration only:

- Strict AWS ECS web deployment contract.
- One-shot `elspeth doctor aws-ecs` preflight and schema-initialization CLI.
- Validate-only web startup for ECS mode.
- Aurora PostgreSQL support as a production runtime dependency.
- EFS-backed local state validation.
- Separate liveness and readiness HTTP endpoints.
- `aws_s3` pipeline source and sink plugins.
- Bedrock readiness for LLM usage: verification of the existing LiteLLM
  surfaces for the web composer, plus a new LiteLLM-backed `bedrock`
  provider for pipeline LLM transforms.
- Documentation and tests for the above.

## Non-Goals

Terraform implementation is out of scope. Terraform will provision ECS,
Aurora, EFS, Secrets Manager, IAM, ALB, logging, and network resources outside
this repository.

SQLite-to-Aurora migration is excluded by design. Elspeth is still pre-1.0, so
the pre-release rule remains: stale runtime data is dropped and recreated by
the operator rather than migrated in place.

Horizontal or multi-container web scaling is a future goal. The current
milestone supports one running web process in one ECS task. The follow-up must
account for Elspeth's existing per-app scaling model and any ECS shape with
multiple containers in one task, not just multiple ECS tasks.

S3-backed internal web blobs and payload storage are out of scope. Internal web
blobs and payloads remain filesystem-backed on EFS for this milestone.

## Existing Context

The current web settings already distinguish the session DB URL from the
landscape/audit DB URL through `ELSPETH_WEB__SESSION_DB_URL` and
`ELSPETH_WEB__LANDSCAPE_URL`. Without explicit values, web state defaults to
SQLite files under `data_dir`.

The web startup currently initializes session schema during app construction.
That is acceptable for local development but not for ECS production startup.
In ECS mode, schema creation must happen in a one-shot command before the web
task receives traffic.

The current `elspeth health` command is a deployment probe. It is not a doctor
surface and should remain safe and mostly read-only. It currently checks
`DATABASE_URL`, which is not the same contract as the web session and landscape
URLs.

The web app intentionally rejects multi-worker mode because progress streaming
and rate limiting are process-local. Aurora and EFS do not remove that design
constraint.

The existing Azure Blob source and sink plugins provide the pattern for the
new S3 plugins: cloud object storage source/sink, CSV/JSON/JSONL support,
external-system audit calls, strict config validation, and unit/property tests.

The web composer uses LiteLLM. The pipeline LLM transform provider registry is
currently provider-specific, with only `azure` and `openrouter` entries and no
LiteLLM path. Bedrock support therefore splits: the composer already works
through LiteLLM model identifiers and needs verification, while pipeline
transforms need a new, explicit LiteLLM-backed Bedrock provider.

## Deployment Contract

Add a web deployment target setting:

```text
ELSPETH_WEB__DEPLOYMENT_TARGET=aws-ecs
```

When the web deployment target is `aws-ecs`, startup and doctor validation must
use strict production rules:

- `ELSPETH_WEB__SESSION_DB_URL` is required.
- `ELSPETH_WEB__LANDSCAPE_URL` is required.
- Both URLs must use a PostgreSQL SQLAlchemy scheme, including driver variants
  such as `postgresql+psycopg`.
- Session and landscape URLs should point to separate logical Aurora targets,
  either separate databases or separate schemas, even when they share one
  Aurora cluster.
- No web database may silently fall back to SQLite on EFS.
- `ELSPETH_WEB__DATA_DIR` is required and must be writable.
- `ELSPETH_WEB__PAYLOAD_STORE_PATH` is required and must be writable.
- Production secrets and signing keys must be present and must not be known
  placeholders.
- The web host must be suitable for container serving, normally `0.0.0.0`.
- The app validates the resulting environment contract. It does not call AWS
  Secrets Manager directly in this milestone.

The aws-ecs contract validates a specific, enumerated placeholder set rather
than an open-ended notion of "known placeholder": `ELSPETH_WEB__SECRET_KEY`
(rejects `change-me-in-production` and values below the existing length
floor) and `ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY` (rejects uniform-byte or
otherwise degenerate keys, using the existing heuristic). The doctor command
and the startup contract validate exactly this enumerated set; no other
secret-backed setting is placeholder-checked. `landscape_passphrase` is
deliberately absent: it is SQLite-only landscape encryption and cannot be set
alongside the PostgreSQL landscape URL this contract requires. OIDC settings
carry no client secret (bearer validation against the issuer's JWKS), so
there is nothing to placeholder-check on that path.

Terraform is responsible for injecting Secrets Manager values into the ECS task
definition. Elspeth cannot prove a value came from Secrets Manager after ECS
injects it, so the enforceable application rule is: web deployments in strict
deployment-target mode must have all secret-backed settings present and
production-shaped. Documentation must state that ECS uses AWS Secrets Manager
and Azure deployments use the Azure equivalent.

Batch and CLI use remain flexible. Operators can still provide manual config
for batch mode, including local SQLite if they choose. The strict contract
applies to the web deployment target.

## Schema Initialization

Add a new CLI group and command:

```bash
elspeth doctor aws-ecs --json
elspeth doctor aws-ecs --init-schema --json
```

The command is read-only by default. It validates:

- deployment target and required web settings;
- PostgreSQL URL scheme for session and landscape DBs;
- DB connectivity;
- session schema state;
- landscape schema state;
- data directory writability;
- payload store path writability;
- blob directory writability;
- required production secrets;
- required runtime dependencies;
- plugin availability relevant to the AWS runtime.

`--init-schema` is the only mutating mode. It may create missing schema objects
in fresh Aurora databases. It must not drop, truncate, or auto-repair existing
data.

`--init-schema` takes a PostgreSQL advisory lock (`pg_advisory_lock`) around
schema creation for each target database, so two concurrent doctor
invocations against the same database serialize instead of racing. This
covers deploy retries and parallel deploys that both attempt initialization.

If a database contains stale or incompatible schema, doctor must fail with
explicit pre-1.0 operator instructions: drop or recreate the affected Aurora
database/schema through the environment's normal procedures, then rerun
`elspeth doctor aws-ecs --init-schema`.

The web app must use validate-only schema behavior in `aws-ecs` mode. If the
schema is missing, stale, or only partially present, web startup fails closed
and points the operator at the doctor command rather than racing the
initializer to completion. Local development can keep create-if-missing
behavior.

## HTTP Health And Readiness

Keep `GET /api/health` as shallow liveness. It should answer whether the
process is responsive and must not depend on Aurora, EFS, or remote providers.

Add a dependency-aware readiness endpoint:

```http
GET /api/ready
```

Readiness should return structured JSON with per-check status and no secrets or
credential-bearing URLs. It should validate:

- session DB connectivity;
- session schema validity;
- landscape DB connectivity;
- landscape schema validity;
- `data_dir` writability;
- payload store writability;
- blob directory writability;
- configured auth mode requirements.

ECS/ALB can use readiness-style checks for traffic routing. Liveness remains
safe from dependency-induced restart storms.

### ECS Probe Wiring

The design's probe surfaces have distinct roles and must be wired distinctly:

- ECS container `healthCheck` calls `GET /api/health` (liveness only).
- ALB target-group health check calls `GET /api/ready`.
- `elspeth doctor aws-ecs` runs as a one-shot pre-traffic task, for example an
  ECS `run-task` before the service update that shifts traffic to the new
  task definition.
- `elspeth health` is not part of the aws-ecs contract. Its `DATABASE_URL`
  contract is orthogonal to the web session and landscape URLs (see Existing
  Context) and must not be wired to any ECS probe.

### Operational Budgets

Readiness and startup carry explicit budgets so an unauthenticated,
dependency-touching endpoint cannot become a latency tax or a DoS surface:

- Each `/api/ready` dependency check has a 2-second timeout.
- The whole `/api/ready` request has a 5-second ceiling.
- Readiness results are cached for 2 seconds so ALB polling cannot stampede
  Aurora or EFS; a poll within the cache window returns the cached result
  instead of re-probing dependencies.
- `/api/ready` stays unauthenticated, because ALB target-group health checks
  cannot present credentials. The 2-second cache is the stampede control, not
  authentication, and this design states that explicitly rather than leaving
  it implicit.
- The `aws_s3` source enforces a configurable `max_object_bytes` guard,
  defaulting to 256 MiB. An object over the limit fails closed with a clear
  error instead of being read fully into memory.
- Aurora connection pools must be explicitly sized rather than left at
  driver defaults: a small fixed pool (for example `pool_size=5`,
  `max_overflow=5`) with connection pre-ping enabled.
- Validate-only startup, including one Aurora cold start, must complete
  within 60 seconds.

### Aurora Serverless v2 Considerations

Aurora Serverless v2 can be configured with a minimum capacity of 0 ACU,
which pauses the cluster and adds a cold-start delay to the next connection.
The design tolerates that possibility without depending on it:

- The startup validate path uses a 10-second connection timeout and a
  bounded retry with backoff, up to 45 seconds total, before failing. The
  ceiling is deliberately below the 60-second startup budget so schema
  validation retains headroom after worst-case connection acquisition.
- The readiness probe cannot absorb a multi-second scale-from-zero cold
  start: with a 2-second per-check timeout, `/api/ready` reports not-ready
  while the cluster wakes. Cold-start tolerance therefore rests on ALB
  threshold configuration, not the probe budget. Operators should either set
  a non-zero minimum ACU or configure ALB healthy/unhealthy thresholds wide
  enough that a single cold start does not pull the task; this design does
  not mandate which.
- `/api/health` never touches Aurora, so liveness cannot hold the cluster
  awake and cannot itself trigger a cold start.
- Readiness caching (see Operational Budgets) bounds how often probe traffic
  can wake a paused cluster.

## Local State And Auth

For the first ECS milestone, EFS is the local-state substrate:

- `ELSPETH_WEB__DATA_DIR` points to the EFS mount, for example `/app/data`.
- `ELSPETH_WEB__PAYLOAD_STORE_PATH` points under that mount, for example
  `/app/data/payloads`.
- Web composer uploaded or generated blobs continue to use the existing
  filesystem-backed blob service.
- The payload store remains filesystem-backed.
- Local auth is allowed as an explicit single-task option, with its SQLite
  auth DB on EFS.
- Cognito/OIDC remains the recommended production auth path.

The doctor and readiness checks cannot reliably prove from inside the
container that a path is EFS. They must prove the runtime contract Elspeth
needs: create, read, fsync or close, and delete a probe file as the non-root
runtime user, without leaking path contents or secrets.

Local auth on EFS is not a multi-task production identity design. If future
multi-task support keeps local auth, that follow-up must move auth state to a
shared relational store or otherwise provide a safe concurrency model.

SQLite over NFS-class filesystems, which is what EFS is, carries a known
corruption hazard: byte-range lock semantics diverge from local disk, and
both SQLite's own documentation and AWS warn against concurrent access to a
SQLite file on EFS. The single-task steady-state posture mitigates this, but
`elspeth doctor aws-ecs` and the web task can both be alive during a deploy,
so the hazard must be scoped explicitly:

- `elspeth doctor aws-ecs` must not open `auth.db`.
- Local auth on EFS is safe only under the strict single-task,
  single-process posture this design already requires; it is not safe once
  more than one task or process can open the file concurrently.
- The SQLite journal mode for `auth.db` on EFS must not be WAL, because WAL
  requires shared memory that EFS does not provide; the default rollback
  journal (`DELETE` mode) must be used instead.
- Cognito/OIDC is the only concurrency-safe option once more than one task
  can exist. Local auth on EFS is a single-task convenience, not a step
  toward a multi-task-capable design.

## AWS S3 Pipeline Plugins

Add `aws_s3` as both a source plugin and a sink plugin, mirroring the current
`azure_blob` source and sink pattern.

The source plugin loads one configured S3 object. It supports:

- CSV;
- JSON array;
- JSONL;
- the same schema, field-normalization, quarantine, and safe validation
  semantics as comparable file and Azure Blob sources.

The sink plugin writes one configured S3 object key. It supports:

- CSV;
- JSON array;
- JSONL;
- Jinja-rendered keys matching Azure sink behavior;
- an `overwrite` option.

Configuration should include:

- `bucket`;
- `key`;
- `format`;
- source CSV and JSON options equivalent to Azure source options;
- sink CSV/header options equivalent to Azure sink options;
- optional `region_name`;
- optional `endpoint_url`, restricted to CLI/batch authorship (see below).

Configuration must not include AWS access-key, secret-key, or session-token
fields. The plugin uses the AWS default credential chain. In ECS, credentials
come from the task role. Local users can use `AWS_PROFILE`, standard AWS env
vars, or standard AWS config outside the pipeline document.

Web-authored `aws_s3` configuration must not set `endpoint_url`. An
author-controlled endpoint combined with server-ambient task-role credentials
is a pipeline-payload exfiltration and SSRF vector: the sink would PUT
pipeline output to an arbitrary author-chosen host, and the source would let
the server issue signed requests to an author-chosen host and ingest the
reply as rows. This mirrors the existing LLM `base_url` doctrine, where a
web-authored custom endpoint is rejected because the server resolves the
credential and a custom target would redirect it to an author-chosen
destination. The gate lives at the same seam as the existing
`web_llm_base_url_policy_error` check: `web/execution/validation.py`, backed
by `provider_config_policy`. `endpoint_url` remains available to
CLI/batch-authored pipelines (LocalStack, tests), where the single-machine
threat model holds.

Audit records should use provider `aws_s3` and record operation, bucket, key,
byte count, content hash where available, overwrite behavior, latency, and
sanitized error class. They must not record credentials, presigned URLs, raw
secret-bearing endpoint strings, or unbounded provider error bodies.

The sink should avoid client-side exists-then-put races. When overwrite is
disabled, use S3 server-side conditional semantics if the selected boto3 API
supports them for the intended call path. If a limitation remains, document it
and test the behavior.

## Bedrock LLM Readiness

Bedrock support has two different postures. The web composer already routes
through LiteLLM and needs verification. The pipeline LLM transform registry
has no LiteLLM path today and needs a committed addition.

For the web composer, verification-first is the right posture:

- confirm LiteLLM accepts Bedrock model identifiers, including Anthropic
  Bedrock identifiers such as `bedrock/anthropic...`;
- confirm AWS default credential chain behavior works in ECS through the task
  role;
- confirm provider errors remain redacted;
- confirm token, cost, prompt-cache, and response metadata parsing handles
  Bedrock-shaped LiteLLM responses.

For pipeline LLM transforms, the current `_PROVIDERS` registry contains only
`azure` and `openrouter`, with no LiteLLM path, so verification is not the
open question here: the design commits to adding a LiteLLM-backed `bedrock`
provider registered in `_PROVIDERS`.

- The `bedrock` provider is implemented through LiteLLM rather than a
  bespoke direct AWS SDK transport.
- The provider must satisfy the existing `LLMProvider` protocol, preserve
  typed error classification, and keep raw provider responses inside the
  provider boundary.
- Provider config must use AWS default credentials, optional `region_name`,
  and a required model identifier. It must not contain AWS access-key or
  secret-key fields.

Docs must show the expected ECS shape: task-role permissions for Bedrock,
region, model identifier, and no embedded AWS keys.

Composer acceptance must be falsifiable rather than a confirmation
narrative: unit tests assert that Bedrock-shaped LiteLLM responses parse
token, cost, and prompt-cache metadata correctly, and an opt-in live smoke
test completes one `bedrock/anthropic...` request when explicit AWS
credentials, region, and model settings are present. Unit tests for the
pipeline `bedrock` provider cover model/config validation and safe error
handling without requiring AWS access.

## Packaging

Add production extras so the runtime image can avoid `--extra all`:

- `postgres` for the chosen PostgreSQL DBAPI driver;
- `aws` for boto3/botocore and any Bedrock/LiteLLM AWS support dependencies
  not already covered by `llm`;
- existing web and LLM extras as needed.

The Docker build should support a production install with only production
extras, for example web, llm, aws, and postgres. Dev and test dependencies,
including testcontainers, should not be required in the final runtime image.

The existing non-root runtime user remains required. AWS credentials should be
provided by ECS task roles and Secrets Manager injection, not baked into the
image or command line.

## Testing Strategy

Unit tests:

- `WebSettings` deployment-target validation;
- ECS mode rejects missing session and landscape URLs;
- ECS mode rejects SQLite session and landscape URLs;
- ECS mode accepts explicit PostgreSQL URL schemes;
- production secret placeholder rejection;
- doctor read-only validation;
- doctor `--init-schema` creation against empty databases;
- doctor stale-schema failure with drop/recreate instructions;
- EFS path validation success and permission failure;
- `/api/ready` success and failure cases with redacted output;
- `/api/health` returns 200 with Aurora and EFS dependencies unavailable
  (liveness independence);
- `aws_s3` source config validation;
- `aws_s3` sink config validation;
- `aws_s3` CSV/JSON/JSONL parsing and writing;
- `aws_s3` overwrite behavior;
- `aws_s3` audit recording and safe error handling;
- web-authored `aws_s3` config with `endpoint_url` set is rejected by
  composer/web validation; CLI/batch config with `endpoint_url` is accepted;
- `aws_s3` source `max_object_bytes` guard fails closed on an oversized
  object;
- Bedrock model/config handling for the pipeline `bedrock` provider;
- Bedrock-shaped LiteLLM responses parse token, cost, and prompt-cache
  metadata correctly for the web composer;
- Bedrock/LiteLLM safe failure classification.

Integration tests:

- PostgreSQL schema create and validate for session DB;
- PostgreSQL schema create and validate for landscape DB;
- `elspeth doctor aws-ecs --init-schema --json` against fresh PostgreSQL
  test databases;
- two concurrent `elspeth doctor aws-ecs --init-schema` runs against the same
  database serialize via advisory lock and do not corrupt or duplicate
  schema;
- web startup in ECS validate-only mode after doctor initialization;
- optional LocalStack-backed S3 source/sink tests if LocalStack is available;
- optional live Bedrock smoke test only when explicitly configured.

Security and boundary checks:

- run Wardline for new external-input boundaries;
- ensure provider errors, DB URLs, and secret-backed settings are redacted in
  doctor, readiness, and plugin error surfaces;
- preserve fail-closed behavior for unknown provider modes and stale schemas.

## Acceptance Criteria

- `elspeth doctor aws-ecs --init-schema --json` succeeds against fresh Aurora
  PostgreSQL targets and writable EFS-mounted state paths.
- `elspeth doctor aws-ecs --json` fails clearly when required ECS web settings
  are missing, malformed, or SQLite-backed.
- In `ELSPETH_WEB__DEPLOYMENT_TARGET=aws-ecs`, the web app refuses to start
  unless explicit PostgreSQL session and landscape URLs are provided.
- In ECS mode, web startup validates existing schema and does not create or
  mutate schema objects.
- Stale Aurora schema fails with operator instructions to drop/recreate data;
  no Elspeth command auto-drops Aurora data.
- `/api/health` remains shallow liveness and returns 200 even when Aurora,
  EFS, or provider dependencies are unavailable (liveness independence).
- `/api/ready` reports dependency readiness with structured, redacted output,
  honoring the per-probe timeout, whole-endpoint ceiling, and result cache
  from Operational Budgets.
- Internal web blobs and payloads work on EFS.
- Local auth works as an explicit single-task option on EFS; `elspeth doctor
  aws-ecs` never opens `auth.db`.
- Cognito/OIDC is documented as the recommended production auth path.
- `aws_s3` source and sink read/write CSV, JSON, and JSONL using AWS default
  credentials.
- `aws_s3` pipeline config contains no AWS access-key or secret-key fields.
- Web-authored `aws_s3` configs cannot set `endpoint_url`; CLI/batch-authored
  configs can.
- `aws_s3` source rejects an object over `max_object_bytes` with a clear,
  fail-closed error.
- Unit tests assert Bedrock-shaped LiteLLM responses parse token, cost, and
  prompt-cache metadata correctly for the web composer; an opt-in live smoke
  test completes one `bedrock/anthropic...` request.
- Pipeline LLM transforms gain a LiteLLM-backed `bedrock` provider registered
  in `_PROVIDERS`, satisfying the existing `LLMProvider` protocol.
- Two concurrent `elspeth doctor aws-ecs --init-schema` runs against the same
  database serialize through the advisory lock without corrupting schema.
- Production image installation no longer depends on `--extra all`.

## Follow-Up Work

Multi-task and multi-container scaling needs a separate design. That design
must address process-local progress streaming, rate limiting, local auth, and
the existing per-app scaling model.

S3-backed internal web blob and payload storage needs a separate design. It
must account for composer blob ownership, storage identifiers, retention,
payload hashes, shareable reviews, and audit/readiness behavior.

Post-1.0 schema migrations need a separate design. Until then, pre-release
Aurora schema incompatibility remains an operator drop/recreate workflow.
