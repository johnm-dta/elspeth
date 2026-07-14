# AWS ECS Packaging & Deployment Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Give AWS ECS operators a lean production image, an executable
deployment/rollback runbook, and a reproducible live persistence/auth harness
for the runtime-readiness acceptance gate.

**Architecture:** Add a validated `INSTALL_EXTRAS` build ARG to the Dockerfile
so the builder stage's `uv sync` installs a caller-supplied extras set; ECS
builds pass `webui llm aws postgres`. Bind the image architecture to the ECS
task definitions, and add `docs/runbooks/aws-ecs-deployment.md`, cross-linked
from the runbooks index and Docker guide.

**Tech Stack:** uv, Docker multi-stage build, Markdown, AWS CLI v2, jq, curl.

**Program orchestration:** Execute this plan directly in the one shared
`/home/john/elspeth/.worktrees/aws-ecs-program` worktree on
`feat/aws-ecs-program`. The orchestration run sheet supersedes any generic
per-plan worktree, branch, claim, merge, or close convention. Task 0 freezes
the clean current program tip before any Plan 10 edit; release-branch movement
is outside this plan. Run-sheet Stage 9 later fixes and reconciles the release
tip as `RECONCILED_RELEASE_SHA`. Plan 12 then requires that ref to remain fixed
through Tasks 1–8; Task 9 alone performs the final exact-SHA fast-forward.

**Depends on:** every implementation slice in Plans 01–09, 11, 13–14, and 15A–15C,
including both split slices of Plans 03 and 08. This broad dependency is
intentional: Task 0 must bind a mechanically complete pre-Plan-10 rollback
baseline before this plan edits packaging, docs, or harnesses. In particular,
Plan 02 provides the `postgres` extra and its lock slice, Plan 06 creates the
initial `aws` extra, and Plan 07 owns the final Jinja2-extended source+sink
`aws` extra and regenerated lock. Plan 03 provides the doctor, Plan 09
provides Bedrock behavior, Plan 11 closes request-path schema mutation, and
Plan 13 makes real Cognito hosted login safe and executable, Plan 14 provides
operator-owned CloudWatch telemetry over task-local OTLP, and Plans 15A–15C provide
explicit Bedrock prompt/content shields. Plan 10 consumes those results; it
never text-merges or independently reconstructs `uv.lock`.

**Deviations:** the brief's `web` extra is `webui` in `pyproject.toml:152`; used throughout. The shared GHCR/ACR image's default (`--extra all`, `Dockerfile:52,60`) stays unchanged — Azure Container Apps' Ansible deploy pulls that image (`docs/runbooks/ansible-ubuntu-deployment.md:2271`) and needs its Azure plugin pack. Production extras become an opt-in `INSTALL_EXTRAS` build arg for a lean ECS image instead, matching the spec's `webui,llm,aws,postgres` example.

**Plan 15B handoff:** bind the rollback baseline only after the complete 15B
acceptance commit is present. The image must preserve the core-only default,
kind-qualified optional allowlist, ordered capability preferences, opaque LLM
profiles, usable tutorial profile with fail-closed typed-409 required-control
coverage gate, and Landscape epoch-23
`run_web_plugin_policy` evidence. ECS task configuration supplies the complete
settings bundle documented in `docs/reference/configuration.md`:
`ELSPETH_WEB__PLUGIN_ALLOWLIST`, `ELSPETH_WEB__PLUGIN_PREFERENCES`,
`ELSPETH_WEB__PLUGIN_CONTROL_MODES`, `ELSPETH_WEB__LLM_PROFILES`,
`ELSPETH_WEB__TUTORIAL_LLM_PROFILE`,
`ELSPETH_WEB__BEDROCK_GUARDRAIL_PROFILES`, and
`ELSPETH_WEB__BEDROCK_GUARDRAIL_DEFAULT_PROFILES`. Changing any member requires
registering a new task-definition revision and forcing a new deployment; an
ECS Exec override is not a rollout. Acceptance must recheck the six-row
`GET /api/system/status` readiness contract and the authenticated typed HTTP
409 tutorial launch gate. The in-task Guardrail proof must also bind the exact
effective policy and protected seven-setting hash, privately correlate the
tutorial model/region with the live Bedrock target, retain the intentional
tutorial coverage blocker and immutable Guardrail versions, bind the required
prompt/content control aliases, and atomically store/read back the Landscape
`run_web_plugin_policy` evidence into its sanitized receipt.
The rollback candidate must understand epoch 23 and the same profile/policy
schema. Do not qualify an epoch-22 image after the database has been recreated
at epoch 23.

**Global Constraints (spec §Packaging, §ECS Probe Wiring, §Local State And Auth, §Bedrock LLM Readiness):**
- "a production install with only production extras, for example web, llm, aws, and postgres. Dev and test dependencies, including testcontainers, should not be required in the final runtime image."
- "The existing non-root runtime user remains required. AWS credentials should be provided by ECS task roles and Secrets Manager injection, not baked into the image or command line."
- The web, payload-verifier, and local-auth-verifier task definitions use a DML-only runtime PostgreSQL principal with CONNECT/USAGE and required table/sequence privileges but no schema CREATE/ownership. Init-capable doctor task definitions use a distinct schema-owner secret approved by the database operator. The runbook records only boolean privilege attestations and distinct secret-reference identities; it never records database role names, URLs, or credentials. Code-level `create_tables=False` and database-level DDL denial are independent controls.
- container `healthCheck`→`GET /api/health` (liveness only); ALB target group→`GET /api/ready`; `elspeth doctor aws-ecs`→one-shot pre-traffic `run-task`; `elspeth health` "is not part of the aws-ecs contract... must not be wired to any ECS probe."
- "Documentation must state that ECS uses AWS Secrets Manager and Azure deployments use the Azure equivalent."
- "Local auth is allowed as an explicit single-task option, with its SQLite auth DB on EFS. Cognito/OIDC remains the recommended production auth path." "`elspeth doctor aws-ecs` must not open `auth.db`." Journal mode "must not be WAL... the default rollback journal (`DELETE` mode) must be used instead."
- "Docs must show the expected ECS shape: task-role permissions for Bedrock, region, model identifier, and no embedded AWS keys."

---

### Task 0: Bind the qualified rollback-baseline source commit

**Files:** no changes; this captures the fully integrated tree before Plan 10
edits it.

- [ ] Start only after the Filigree DAG shows Plans 01–09, 11, 13–14, and 15A–15C
  (including both split slices of Plans 03 and 08) done. Record
  `ROLLBACK_BASELINE_SHA=$(git rev-parse HEAD)` and require a clean worktree.
  The dependency graph enforces this state; a hand-selected older 0.7.0 image
  is invalid because it lacks Plan 13's Cognito contract, Plan 14's mandatory
  AWS operator telemetry posture, Plan 15B's effective web plugin policy, and Plan 15C's Guardrail transforms.
- [ ] Run the exact unfiltered auth regressions against this SHA:
  `uv run pytest tests/unit/web/auth/ tests/unit/web/test_config.py tests/unit/web/test_app.py -v`,
  `npm --prefix src/elspeth/web/frontend test -- src/components/auth/LoginPage.test.tsx`,
  and `npm --prefix src/elspeth/web/frontend run typecheck`. Record the SHA,
  timestamp, exact commands, exit codes, approved evidence reference, and the
  literal marker `aws-ecs-rollback-baseline-source` in one non-secret
  Filigree `comment_add` on issue `elspeth-6285c29c07`. Do not tag it as a
  release or call it live-qualified yet.
  Plan 12 builds the immutable image, binds its exact task definition and
  compatibility record, and earns that status with the first browser phase.

**Definition of Done:** one clean, mechanically complete pre-Plan-10 SHA is
durably recorded for the Scenario B rollback baseline.

---

### Task 1: Lean ECS production Docker build via `INSTALL_EXTRAS`

**Files:**
- Modify: `Dockerfile:48-60`, `Dockerfile:107-114` (the two `uv sync`
  calls, selected-extras comments, and the stale orchestrator-health example)
- Modify: `tests/unit/test_build_push_release_checks.py:138,154-156,166-168`
  (three tests assert the exact `--extra all` literal this task removes; add
  validation/default/health-comment contract coverage alongside them)

`tests/unit/test_build_push_release_checks.py` already has a real pytest harness over the Dockerfile's text (`test_release_dockerfile_builds_frontend_dist_before_python_install`, `test_release_dockerfile_copies_local_uv_sources_before_dependency_sync`, `test_release_dockerfile_copies_frontend_dist_into_installed_package`) asserting the literal substrings `"uv sync --frozen --extra all --no-editable --active"` / `"...--no-install-project --active"`, both removed by this task's edit — update them below, do not leave them red. Beyond that harness, no pytest coverage exercises an actual `docker build`; runtime verification still uses `docker build`/`docker run` as red/green steps.

**Interfaces:**
- Consumes: `webui` extra (`pyproject.toml:152-165`), `llm` extra (`pyproject.toml:124-130`), final source+sink `aws` extra and lock (plan 07 after plan 06), `postgres` extra (plan 02)
- Produces: Dockerfile `ARG INSTALL_EXTRAS="all"` (default preserves shared-image behavior)

- [ ] Hard precondition (this plan starts after its dependencies, so this is a
  passing gate, not an impossible historical RED step):
  `uv sync --frozen --extra webui --extra llm --extra aws --extra postgres --no-install-project --dry-run`
  exits 0. A non-zero result means Plan 02/06/07's final integrated
  `pyproject.toml`/`uv.lock` state is absent; return ownership there instead of
  reconstructing or regenerating the lock in this task.
- [ ] RED (test harness): before editing the Dockerfile, update the three
  literal assertions in `tests/unit/test_build_push_release_checks.py` to the
  new stable substrings — `"uv sync --frozen \"$@\" --no-install-project --active"`
  and `"...--no-editable --active"` — and add tests requiring the default
  `ARG INSTALL_EXTRAS="all"`, non-empty input, token validation in both sync
  layers, and the corrected orchestrator-health comment. Leave the ordering
  assertions (`npm run build` before install-project sync; that sync before
  the frontend-dist `copytree`) unchanged. Run
  `uv run pytest tests/unit/test_build_push_release_checks.py -v` and require
  the new/changed assertions to fail against the old Dockerfile for the
  expected missing contract, not an unrelated error.
- [ ] Add `ARG INSTALL_EXTRAS="all"` above the first `uv sync`. Replace both
  hardcoded `--extra all` flags with the same validated loop over the
  space-separated value. Reject an empty value and every token outside
  `[a-z0-9][a-z0-9-]*`, build positional arguments with `set --`, and pass
  `"$@"`; a caller must not be able to smuggle a uv option through this
  build argument. Chain with `&&` throughout (not `;`) so any earlier failure
  still aborts the layer:
  ```
  RUN uv venv /opt/venv && \
      . /opt/venv/bin/activate && \
      test -n "$INSTALL_EXTRAS" && \
      set -f && \
      set -- && \
      for e in $INSTALL_EXTRAS; do \
          case "$e" in [a-z0-9]*) ;; *) exit 2 ;; esac; \
          case "$e" in *[!a-z0-9-]*) exit 2 ;; esac; \
          set -- "$@" --extra "$e"; \
      done && \
      test "$#" -gt 0 && \
      uv sync --frozen "$@" --no-install-project --active
  ```
  Mirror the validation into the second call. Update the builder comments so
  they describe the selected extras rather than falsely promising that every
  image bundles all plugins.
- [ ] Correct the existing Dockerfile orchestrator comment while this surface
  is open: web task definitions probe loopback `GET /api/health`; ALBs probe
  `GET /api/ready`; batch tasks use process exit; the image itself still has
  no `HEALTHCHECK`. It must not recommend `elspeth health` for a web container.
- [ ] GREEN (test harness):
  `uv run pytest tests/unit/test_build_push_release_checks.py -v` → all tests
  pass, including empty, whitespace-only, leading-hyphen, uppercase,
  glob/metacharacter,
  default, and four-production-extra cases across both sync layers.
- [ ] Execute the fail-closed build boundary, not only text assertions:
  `docker build --build-arg INSTALL_EXTRAS="   " .` and
  `docker build --build-arg INSTALL_EXTRAS="--no-dev" .` both fail in the
  extras-validation layer; a missing Docker daemon is blocked, not a pass.
- [ ] GREEN, default unchanged: set `TARGET_PLATFORM=linux/amd64` for the
  release acceptance inventory (or the one explicitly approved platform),
  record `EXPECTED_VERSION=$(uv run python -c 'from importlib.metadata import version; print(version("elspeth"))')`;
  `docker build --platform "$TARGET_PLATFORM" -t elspeth:default-test .`
  succeeds; `docker run --rm elspeth:default-test --version` prints a line
  containing `elspeth version $EXPECTED_VERSION`.
- [ ] GREEN, default still bundles dev/test + Azure (inverse of the lean check, guards against the loop silently narrowing the default install): `docker run --rm --entrypoint python elspeth:default-test -c "import pytest, testcontainers, azure.storage.blob"` exits 0.
- [ ] GREEN, lean build: `docker build --platform "$TARGET_PLATFORM" --build-arg INSTALL_EXTRAS="webui llm aws postgres" -t elspeth:ecs-prod-test .`
  succeeds; `docker image inspect` reports the OS/architecture represented by
  `TARGET_PLATFORM`; `docker run --rm elspeth:ecs-prod-test --version` prints a
  line containing `elspeth version $EXPECTED_VERSION`.
- [ ] No dev/test deps (the Dockerfile's `ENTRYPOINT ["elspeth"]` intercepts bare args, so override it to reach a real interpreter): `docker run --rm --entrypoint python elspeth:ecs-prod-test -c "import importlib.util as u,sys; missing=[m for m in ('testcontainers','pytest','mypy','ruff') if u.find_spec(m) is None]; sys.exit(0 if len(missing)==4 else 1)"` exits 0.
- [ ] Production deps present: `docker run --rm --entrypoint python elspeth:ecs-prod-test -c "import boto3, botocore, ijson, jinja2, psycopg, litellm, fastapi"` exits 0.
- [ ] Non-root user unchanged: `docker run --rm --entrypoint id elspeth:ecs-prod-test` prints output starting with `uid=1000(elspeth) gid=1000(elspeth)` (real `id` output appends a trailing `groups=...` field — match the prefix, not the full line).
- [ ] No baked-in AWS keys: capture `docker history --no-trunc` to a protected
  temporary file and require the command itself exits 0, then use explicit
  `if grep ...; then exit 1; fi` absence checks over both Dockerfile and
  history. A failed `docker history` must not be mistaken for a clean grep.
- [ ] `git add Dockerfile tests/unit/test_build_push_release_checks.py && git commit -m "feat(docker): add INSTALL_EXTRAS build arg for lean AWS ECS production images"`

All GREEN steps above are real `docker build`/`docker run` invocations, not
simulated. No CI job builds the `INSTALL_EXTRAS` path today
(`build-push.yaml` never passes that build arg); the final review retains this
as a post-0.7.1 supply-chain strengthening item, not evidence for this exact
temporary acceptance artifact.

### Task 2: AWS ECS deployment runbook + cross-links

**Files:**
- Create: `docs/runbooks/aws-ecs-deployment.md`
- Create: `tests/unit/web/test_aws_ecs_runbook_contract.py`
- Modify: `docs/runbooks/index.md:17` (Quick Reference table row)
- Modify: `docs/guides/docker.md:424-428` (See Also link only; its Kubernetes-probe section is unrelated and unchanged)

**Interfaces:** none (documentation only).

- [ ] Write `docs/runbooks/aws-ecs-deployment.md` (mirror the Symptoms/Prerequisites/Steps shape of `docs/runbooks/configure-keyvault-secrets.md`), covering exactly:
  1. **Contract summary** — `ELSPETH_WEB__DEPLOYMENT_TARGET=aws-ecs` requires PostgreSQL `session_db_url`/`landscape_url` (incl. `postgresql+psycopg`), writable `data_dir`/`payload_store_path`, rejects placeholder secrets (`ELSPETH_WEB__SECRET_KEY`, `ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY`); enforced by the sibling contract plan, described here for operators. State the general credentials principle once, here, for the whole ECS contract: "AWS credentials are provided by ECS task roles and Secrets Manager injection — never baked into the image or passed on the command line" (the Bedrock bullet below restates this only for its own case, not as the sole place it's said).
  2. **Secrets statement** (spec-required substance, plan-authored wording — not a verbatim spec quote): "ECS injects secrets via AWS Secrets Manager into the task definition; Azure deployments use the Azure equivalent (Key Vault — see [configure-keyvault-secrets.md](configure-keyvault-secrets.md))."
  3. **Auth** — Cognito/OIDC is recommended. Document the exact operator
     settings: `auth_provider=oidc`; the user-pool
     `oidc_issuer`; `oidc_audience` and `oidc_client_id` set to the app-client
     ID; the hosted/custom-domain `oidc_authorization_endpoint`;
     the same-origin hosted/custom-domain `oidc_token_endpoint`;
     `ELSPETH_WEB__OIDC_AUTHORIZATION_ALLOWED_ORIGINS='["https://example.auth.ap-southeast-2.amazoncognito.com"]'`;
     and `ELSPETH_WEB__OIDC_AUDIENCE_CLAIM=client_id`. The allowlist accepts
     exact normalized HTTPS origins only—wildcards, suffix matching, paths,
     and automatic Cognito-domain inference are rejected. `client_id` mode is
     for Cognito access tokens and requires `token_use=access`; it is never a
     fallback for generic OIDC `aud` validation. Local auth
     (`auth_provider=local`) is an explicit single-task option, with `auth.db`
     on EFS; `elspeth doctor aws-ecs` never opens it and EFS journal mode stays
     `DELETE`, never WAL. For Cognito, require `COGNITO_USER_POOL_ID` and use a
     secret-filtered `aws cognito-idp describe-user-pool-client` preflight.
     The public app client must have no client secret,
     `AllowedOAuthFlowsUserPoolClient=true`, include the `code` flow and not
     the `implicit` flow, use S256 PKCE,
     include all of `openid`, `profile`, and `email` in
     `AllowedOAuthScopes`, and contain the exact HTTPS `ALB_BASE_URL` origin
     (no path or trailing slash) in `CallbackURLs`; otherwise browser
     acceptance is operationally impossible and deployment is NO-GO. ELSPETH
     disables Uvicorn's raw request-line access logger because Cognito returns
     the PKCE-bound code in the callback query. If upstream ALB access logging
     is enabled, document and approve its short retention/access policy because
     those logs retain callback codes even though PKCE prevents redemption
     without the verifier.
  4. **ECS Probe Wiring** — condensed 4-row table (container `healthCheck`→`/api/health`, liveness only; ALB target group→`/api/ready`; `elspeth doctor aws-ecs`→one-shot pre-traffic `run-task`; `elspeth health`→not wired to any probe), plus one line declaring this runbook the canonical operator entry point and cross-linking the fuller write-up plan 05 creates at `docs/operator/aws-ecs-health-and-readiness.md` — same four facts, kept in sync manually; don't let the two drift. Give the exact task-definition JSON command, using the runtime image's installed Python rather than an absent `curl`/`wget`: `["CMD","python","-c","import http.client,sys; c=http.client.HTTPConnection('127.0.0.1',8451,timeout=5); c.request('GET','/api/health'); r=c.getresponse(); sys.exit(0 if r.status == 200 else 1)"]`. The contract test parses this as JSON. It bypasses proxy settings, does not follow redirects, and requires exactly HTTP 200. Include the startup-window sizing obligation (round-3; authoritative wording in spec §Operational Budgets): the validate-only gate runs before uvicorn binds the socket, so the port is **closed** (connection-refused, not 503) for up to the two-cold-cluster worst case — set **both** the container `healthCheck.startPeriod` **and** the ECS service `healthCheckGracePeriodSeconds` to that worst case plus margin (at least 150s; the ~90s figure excludes probe time). With the default (zero) grace period, ALB kills a cold-starting task mid-validation into a restart loop that never converges.
  5. **Bedrock ECS shape** — task-role IAM for `bedrock:InvokeModel`, `region_name`, `bedrock/anthropic...` model id, no embedded AWS keys (references the principle stated in item 1).
  5a. **Bedrock Guardrails shape** — two immutable numeric Guardrail versions,
     resource-scoped `bedrock:ApplyGuardrail` (and only if used by preflight,
     `bedrock:GetGuardrail`), Plan-15B allowlisting/preferences/control modes,
     opaque Plan-15C profile aliases lowered to task-role-only private
     bindings, explicit `aws_bedrock_prompt_shield` and
     `aws_bedrock_content_safety` placement, no `DRAFT`, raw web binding,
     static credentials, or provider-refusal-as-shield claim, and the
     documented Bedrock/Azure category gap. Terraform creates two run-scoped
     Guardrails tagged with the acceptance run ID, publishes immutable numeric
     versions, injects their private bindings into opaque aliases, records the
     version/configuration hashes in acceptance evidence, and destroys the
     resources in the reviewed teardown. Shared pre-existing Guardrails are
     not accepted by this milestone because ownership/configuration drift would
     make safe/blocked evidence non-reproducible.
  5b. **CloudWatch operator telemetry shape** — the Plan 14 AWS web overlay,
     fixed task-local OTLP endpoint, lifecycle/rows-only pipeline events,
     web OTLP metrics, digest-pinned CloudWatch Agent sidecar with no published
     port, bounded queue/memory, task-role-only CloudWatch/X-Ray delivery,
     resource/cardinality allowlists, dashboard/alarms, retention, and explicit
     Landscape-first authority. The sidecar is non-essential after startup but
     must be healthy before the app starts; later signal loss blocks acceptance
     and alerts without invalidating an already committed audit record.
  Add the sibling S3 task-role contract here as well: grant only the required
  object actions (`s3:GetObject` for source reads and `s3:PutObject` for sink
  writes) on approved bucket/prefix ARNs; never grant a wildcard bucket or
  inject static AWS keys. The disposable acceptance task role additionally
  gets `s3:DeleteObject` only on `ELSPETH_ACCEPTANCE_S3_BUCKET` plus its
  UUID-scoped `ELSPETH_ACCEPTANCE_S3_PREFIX`; the Plan-12 operator has the same
  narrow cleanup permission as a backstop. Steady-state production does not
  inherit this test-only delete permission.
  6. **Packaging and platform identity** — the Task 1
     `INSTALL_EXTRAS="webui llm aws postgres"` build with the exact
     `docker build --platform "$TARGET_PLATFORM"` command. The approved value
     is explicit, with the closed mapping `linux/amd64` → `X86_64` and
     `linux/arm64` → `ARM64`; every web, doctor, verifier, and rollback task
     definition declares `runtimePlatform.operatingSystemFamily == LINUX` and
     the mapped `cpuArchitecture`. A host-native image with no recorded
     platform is NO-GO. Note that the lean image drops the `azure` extra —
     pipelines using `azure_blob` need the default `all` build or an expanded
     `INSTALL_EXTRAS`. Operators build and push their own tagged acceptance
     image to ECR; the published GHCR/ACR default remains the `all` image.
  7. **Mounted directory provisioning before doctor** — the infrastructure/release operator must provision the configured `data_dir`, explicit `payload_store_path`, and derived blob directory on the intended EFS access point before any doctor task runs, with ownership/mode allowing the non-root task user to create, read, fsync, and delete a file. Mounting only the parent is insufficient: doctor never calls `mkdir`, including under `--init-schema`, because creating a missing child in container overlay storage would mask a displaced or incomplete EFS mount. Record the resolved mount/access-point identity and directory provisioning evidence in the change record; the doctor's active ephemeral probes remain the runtime-user permission proof.
  8. **Cold-start precondition for first deploy** — `elspeth doctor aws-ecs --init-schema` uses a single 10-second connection probe with **no retry budget** (fail-fast by design for an interactive/one-shot task, unlike web startup's 45s bounded retry); against a min-0-ACU Aurora cluster it can false-fail while the cluster wakes. Before the first `--init-schema`, either set a non-zero minimum ACU or issue a warm-up connection and re-run — doctor is idempotent and safe to re-run.
- [ ] After those eight contract areas, add a narrowly scoped **Ordered rollout and rollback** procedure. The **release operator** owns every gate and records only sanitized, allowlisted resolved inputs, task-definition/image digests, timestamps, task/event/log evidence, and command results in the change record; the **database operator** owns any destructive schema action. Define and validate these environment-specific inputs once from the approved deployment inventory: `DEPLOYMENT_MODE` (`upgrade`, `first`, or `first-recovery`), `TARGET_PLATFORM`, `AWS_REGION`, `ECS_CLUSTER`, `ECS_SERVICE`, `WEB_CONTAINER_NAME`, `TARGET_GROUP_ARN`, `ALB_BASE_URL`, `CANDIDATE_TASK_DEFINITION`, `DOCTOR_TASK_DEFINITION`, `DOCTOR_CONTAINER_NAME`, `DOCTOR_NETWORK_CONFIGURATION` (the complete `awsvpcConfiguration` JSON), `WEB_LOG_GROUP`, `WEB_LOG_STREAM_PREFIX`, `DOCTOR_LOG_GROUP`, `DOCTOR_LOG_STREAM_PREFIX`, `ECS_DEPLOYMENT_EVENT_RULE`, `ECS_DEPLOYMENT_EVENT_TARGET_ID`, and `ECS_DEPLOYMENT_EVENT_LOG_GROUP`. `PREVIOUS_TASK_DEFINITION` is required only for `upgrade`; `FIRST_DEPLOY_LISTENER_RULE_ARN` is required for `first` and `first-recovery`; `COGNITO_USER_POOL_ID` is required for a Cognito/OIDC scenario. Reject missing/invalid inputs. `ECS_CLUSTER` and `ECS_SERVICE` must be the short cluster/service names (validate each against `^[A-Za-z0-9_-]+$` and cross-check the returned ARN/name from `describe-services`), never ARNs; this is required because Application Auto Scaling's resource ID is literally `service/$ECS_CLUSTER/$ECS_SERVICE`. Resolve supplied task definitions through `describe-task-definition` and replace them with the returned exact `taskDefinitionArn` values before any comparison; validate each is `ACTIVE`, its named image digest and Linux `runtimePlatform.cpuArchitecture` match `TARGET_PLATFORM`, the doctor task uses the candidate image digest and named container, the JSON network configuration names the intended subnets/security groups, and the log groups/stream prefixes match the task definitions. `first-recovery` is deliberately narrow: it may only restart the same immutable candidate after the `first` rollback path left desired count zero and traffic fixed at 503; it is not a general third deployment mode. This milestone requires exactly one running web task and accepts deployment downtime: every intentional launch, relaunch, upgrade, or manual rollback uses `desiredCount=1`, `minimumHealthyPercent=0`, `maximumPercent=100`, and `--force-new-deployment`; capture the previous primary deployment ID/created time and require a distinct new primary plus task ARN before accepting the waiter. Changing `desiredCount` alone does not create a deployment. Prescribe the following order with a hard stop between phases:
  1. **Preflight (no service mutation):** require AWS CLI v2, `jq`, `curl`, the Session Manager plugin, Node 22/npm, and Playwright/Chromium installed from the frontend lock before any AWS mutation. Start copyable shells with `set -Eeuo pipefail`, `umask 077`, `AWS_PAGER=""`, bounded curl connect/total timeouts, and cleanup traps. Document the least-privilege ECS, ELBv2, CloudWatch Logs, EventBridge, IAM, SSM Messages, and Application Auto Scaling permissions. Bind each task definition's exact approved `taskRoleArn` (runtime S3/Bedrock/default-chain) and `executionRoleArn` (ECR pull, awslogs, Secrets Manager), forbid plaintext secret/static credential/profile/role/endpoint fields, and require approved ELSPETH secrets through Secrets Manager references. Mechanically require the task role's four exact `ssmmessages` channel actions, private-subnet Session Manager Messages reachability, and any configured KMS/session-log permissions; no `ssmmessages:*`. For EFS, require exact filesystem/access point, encryption and IAM authorization, `rootDirectory` absent or `/`, a read-write mount to the configured data-dir ancestor, task-role `elasticfilesystem:ClientMount`/`ClientWrite` scoped to that filesystem/access point, and no `ClientRootAccess` unless separately approved. Set `OBSERVATION_START_EPOCH_MS=$(($(date +%s) * 1000))` before any doctor/deployment command. Require the rolling `ECS` deployment controller, Linux Fargate platform `1.4.0` or `LATEST`, current `desiredCount` of 1 in `upgrade` mode (0 in `first`/`first-recovery`), `enableExecuteCommand == true`, and the approved zero-overlap deployment configuration. Because ECS Exec cannot be enabled retroactively on existing tasks, it must be enabled before any acceptance task launches; after launch, `describe-tasks` must show the 1.4.0 platform family, `enableExecuteCommand == true`, and the named container's `ExecuteCommandAgent.lastStatus == RUNNING`. ECS Exec runs as root: it may prove task-role S3/Bedrock behavior but never non-root EFS permission or local-auth single-process behavior; those use explicit-UID one-shot verifier tasks. Require the web container's exact task-definition `healthCheck.command` to be the Python loopback `/api/health` command above, not `elspeth health` or `/api/ready`, with bounded timeout/retries and `startPeriod >= 150`; require service `healthCheckGracePeriodSeconds >= 150`. Run the no-autoscaling and exact ALB target-group checks described below. In `upgrade`, bind the completed distinct previous revision; in `first`, require desired-zero and no distinct known-good revision; in `first-recovery`, require desired-zero, fixed-503, the same candidate task definition, and prior rollback evidence. Validate the exact deployment JSON. For the EventBridge durable failure signal, require the enabled exact event pattern and CloudWatch Logs target, require that target has no `RoleArn`, and use `aws logs describe-resource-policies` to prove a resource-policy statement grants `events.amazonaws.com` and `delivery.logs.amazonaws.com` only `logs:CreateLogStream`/`logs:PutLogEvents` on the exact destination ARN. Do not spoof the AWS-owned `aws.ecs` source. Create a temporary canary rule named from `ACCEPTANCE_RUN_ID` that matches a dedicated custom source, target the same CloudWatch Logs destination under the same resource policy, publish one uniquely correlated custom canary, and poll until it reaches that log group. Persist only the sanitized correlation receipt, then remove the canary target/rule and prove absence before deployment. The production rule's exact ECS pattern is inspected separately; delivery-failure metrics alone do not prove the target works. Retain only allowlisted projected fields, never raw rule, target, policy, or log payloads.
  2. **Hardened one-shot doctor:** construct `DOCTOR_OVERRIDES=$(jq -cn --arg name "$DOCTOR_CONTAINER_NAME" '{containerOverrides:[{name:$name,command:["doctor","aws-ecs","--json"]}]}')`; for an approved initialization only, construct the same JSON with `command:["doctor","aws-ecs","--init-schema","--json"]`. Invoke `aws ecs run-task --cluster "$ECS_CLUSTER" --task-definition "$DOCTOR_TASK_DEFINITION" --launch-type FARGATE --network-configuration "$DOCTOR_NETWORK_CONFIGURATION" --count 1 --overrides "$DOCTOR_OVERRIDES"` only through the protected stdout/stderr capture wrapper; parse successful stdout in memory and persist only a `sanitize-evidence` receipt. Require `(.failures | length) == 0` and `(.tasks | length) == 1` in the `run-task` JSON, extract the one non-empty exact `DOCTOR_TASK_ARN=.tasks[0].taskArn`, and pass only that ARN to `aws ecs wait tasks-stopped --cluster "$ECS_CLUSTER" --tasks "$DOCTOR_TASK_ARN"`. Obtain the essential container-name set from `aws ecs describe-task-definition --task-definition "$DOCTOR_TASK_DEFINITION"` (`containerDefinitions[] | select(.essential == true) | .name`), then `describe-tasks` for that exact ARN; require no describe failures, `.tasks[0].taskArn == $DOCTOR_TASK_ARN`, and an explicit `exitCode == 0` for every named essential container. Missing/null exit codes, a name mismatch, a non-zero exit, or any sanitized doctor failure blocks `update-service`; run `aws logs filter-log-events --log-group-name "$DOCTOR_LOG_GROUP" --log-stream-name-prefix "$DOCTOR_LOG_STREAM_PREFIX" --start-time "$OBSERVATION_START_EPOCH_MS"` for diagnosis and retain the sanitized output.
  3. **Schema initialization and compatibility gate:** distinguish the two databases explicitly. `--init-schema` may initialize the **session** schema only when it is MISSING; a partially present session table set is shape-unverifiable and therefore classifies STALE, requiring the operator-controlled drop/recreate path. It may initialize/complete the **landscape** schema when MISSING or when the probe reports verified-shape PARTIAL, except that Plan 15B's missing `run_web_plugin_policy` on any non-empty pre-epoch-23 Landscape is deliberately STALE and never additively completed. That one-way change requires approved archive/export where retention applies, database-operator drop/recreate, and fresh owner initialization. STALE/incompatible in either database blocks deployment. State that Aurora schema detection is structural-only: a semantics-only schema-epoch change can still probe as CURRENT, so release compatibility must come from the approved release/schema compatibility record, not CURRENT alone. The release operator must attach that record before deployment.
  4. **Deploy:** run `aws ecs update-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" --task-definition "$CANDIDATE_TASK_DEFINITION" --desired-count 1 --force-new-deployment --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}'`, then wait for stability. A non-zero waiter result, `rolloutState=FAILED`, `SERVICE_DEPLOYMENT_FAILED` event, absent distinct fresh deployment identity/task ARN, or more than one running/pending task fails the rollout. In first modes there is no different completed revision for circuit-breaker rollback; failure keeps/drains traffic to fixed 503 and scales desired count to zero. The rollback branches remain as below and also force a new deployment.
  5. **Candidate-aware post-deploy acceptance:** after the waiter succeeds, require `describe-services` to show exactly one deployment, `rolloutState=COMPLETED`, the candidate task definition as primary, `desiredCount == 1`, `runningCount == 1`, and `pendingCount == 0`. Use `aws ecs list-tasks --cluster "$ECS_CLUSTER" --service-name "$ECS_SERVICE" --desired-status RUNNING`, pass the exact returned task ARN to `aws ecs describe-tasks`, and require exactly one task whose `taskDefinitionArn == CANDIDATE_TASK_DEFINITION`; any second running or pending web task is a contract violation. Resolve the service's `loadBalancers[]` entry whose `targetGroupArn == TARGET_GROUP_ARN` and require its `containerName == WEB_CONTAINER_NAME`; use that entry's `containerPort` with the candidate task's `WEB_CONTAINER_NAME` ENI `privateIpv4Address` to construct the exact ALB target `(Id,Port)` pair. Compare that pair with `aws elbv2 describe-target-health --target-group-arn "$TARGET_GROUP_ARN"`: the candidate pair must be `healthy`; after zero-overlap replacement any old target may be ignored only while `draining`, never while serving traffic. In `first` or `first-recovery`, only after those pre-traffic checks pass, run `aws elbv2 modify-rule --rule-arn "$FIRST_DEPLOY_LISTENER_RULE_ARN" --actions "$FIRST_DEPLOY_FORWARD_ACTIONS"` and verify the rule now forwards only to `TARGET_GROUP_ARN`. For both public probes, write the bounded body to a protected temporary file, use `curl --connect-timeout 5 --max-time 10 --max-redirs 0 -sS -o ... -w '%{http_code}'`, and require the status string equals `200`; only then parse `.ready == true`. `curl -f` alone is insufficient because it accepts redirects.
  6. **Ten-minute observation and actionable signals:** set `ACCEPTANCE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)` immediately after the first complete candidate-aware acceptance pass. Poll every 30 seconds for 20 iterations (10 minutes), repeating the service deployment/count checks, candidate task/target mapping, candidate target-health assertion, and both exact-200 bounded curl commands. On every iteration, invoke the protected stdout/stderr capture wrapper for each bounded `filter-log-events` call and pass successful stdout to `sanitize-evidence` before anything is printed or persisted; retain only the allowlisted receipts. Include a source/signal/action table: `SERVICE_DEPLOYMENT_FAILED`, `rolloutState=FAILED`, repeated launch/placement failures, non-zero essential-container exits, or task `stoppedReason` fail the rollout and require rollback plus stopped-task inspection; candidate `Target.ResponseCodeMismatch`, `Target.Timeout`, or `Target.FailedHealthChecks` requires probe/grace-period diagnosis and rollback if persistent; non-200 `/api/health`, non-200 or `ready != true` `/api/ready`, `readiness_check_not_ready`, startup contract/schema validation failures, or new unhandled `ERROR`/`CRITICAL` log events blocks acceptance and triggers rollback when introduced by the candidate. Preserve sanitized receipts; use check names, exception classes, ECS events, stopped-task metadata, and counts, never messages, secrets, raw URLs, or raw AWS JSON. Acceptance requires all 20 samples to pass and records `ACCEPTANCE_START_UTC` plus the final timestamp.
  7. **Manual rollback:** if acceptance fails after ECS declared stability, or automatic rollback is unavailable/incomplete, branch on mode. In `upgrade`, run `aws ecs update-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" --task-definition "$PREVIOUS_TASK_DEFINITION" --desired-count 1 --force-new-deployment --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}'`, prove a distinct deployment/task identity, wait for stability, then repeat the exact-one-task, previous-revision target mapping/health, `/api/health`, `/api/ready`, CloudWatch, and EventBridge checks against the restored revision. In `first` or `first-recovery`, remove traffic before stopping compute: run `aws elbv2 modify-rule --rule-arn "$FIRST_DEPLOY_LISTENER_RULE_ARN" --actions "$FIRST_DEPLOY_DISABLED_ACTIONS"`, verify the fixed 503 action, then run `aws ecs update-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" --desired-count 0 --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}'`; wait for stability and require zero running/pending tasks and no registered non-draining targets. Keep the failed task definition, stopped-task evidence, event capture, and sanitized log receipts until the incident record is complete.
  8. **Schema-compatibility rollback stop:** state that task-definition/code rollback cannot undo an incompatible database schema and is safe only when the previous code is compatible with the current schema according to the release/schema compatibility record. AWS ECS validate-only startup must fail closed before uvicorn binds when the structural probe reports missing, partial, stale, or incompatible; web startup must never create, migrate, drop, or repair schema. Session MISSING-only and landscape MISSING/verified-PARTIAL are the only `--init-schema` cases, subject to the epoch-23 exception above. Session partial and a non-empty pre-23 Landscape missing `run_web_plugin_policy` classify STALE. STALE/incompatible schema requires the pre-1.0 drop/recreate procedure owned and approved by the database operator, followed by `--init-schema`; never automate that destructive action or treat repeated code rollback as schema repair. Code rollback to epoch 22 after an epoch-23 recreation is unsafe. In `upgrade`, if compatibility with `PREVIOUS_TASK_DEFINITION` cannot be proven, keep traffic drained and escalate instead of rolling code back over the incompatible schema. In `first` or `first-recovery`, keep the listener on fixed 503 and the service at desired count zero until the database operator repairs/recreates the schema and the full gate restarts.
  9. **Release/schema compatibility record authority:** include a copyable record template and make the **database operator** its author/approver and the **release operator** its countersigner before doctor/deploy. Required fields: record ID and timestamp; candidate git SHA, immutable image digest, exact candidate/doctor task-definition ARNs, package version, session-schema epoch, landscape-schema epoch (23 for this candidate), presence of `run_web_plugin_policy`, structural schema changes, semantics-only changes, archive/export decision, and destructive reset requirement; for upgrades, previous package version, immutable image/task-definition identity, both previous schema epochs, forward/backward compatibility decisions, and an explicit `rollback_permitted: yes|no` with evidence; for first deploy, mark previous identity/rollback compatibility `not_applicable` and bind recovery to fixed-503 + desired-zero. Record who inspected the code/schema diff, the evidence locations, database-operator approval, release-operator countersignature, and expiry/supersession. Unknown or unapproved compatibility is NO-GO. Store the completed record in the approved change-management system (not in a secret-bearing URL or task environment); Plan 12 binds its record ID to each live scenario.
  10. **Disposable-acceptance cleanup:** before either environment is created,
      record the infrastructure owner, identity owner, evidence destination,
      a non-secret UUID `ACCEPTANCE_RUN_ID` applied as a tag to every
      disposable resource, and a teardown deadline no later than four hours
      after the live gate.
      On success, failure, interruption, or timeout, first export the approved
      sanitized evidence, then destroy both Terraform scenario stacks and
      verify their state is empty; delete/rotate the Cognito test identity and
      test secrets; and delete both rollback-baseline and candidate acceptance
      ECR tags. Promotion is forbidden during cleanup and before Plan 12's
      final GO. Durable lean-image publication is out of this plan's scope and
      requires a separate release-owner issue/workflow that builds the GO SHA
      for approved platforms, emits and verifies SBOM/provenance, signs the
      resulting digest, and only then considers tagging/deployment. The
      `TARGET_PLATFORM` digest must equal Plan 12's recorded live-accepted
      digest or repeat its artifact/live acceptance; every other platform
      requires its own acceptance. Plan 12 GO alone never authorizes a rebuilt
      digest, and mere retagging of the temporary acceptance digest is
      insufficient. Cleanup failure is
      itself NO-GO and escalates
      to the named owner; a disposable environment is never the release's
      steady-state deployment. Terraform state-empty is not enough after a
      partial apply: require a tag-based AWS inventory sweep plus explicit
      checks for ECS, ALB, Aurora, EFS, Secrets Manager, CloudWatch Logs,
      EventBridge, and Cognito surfaces, with zero live resources or a signed
      shared-resource restoration receipt. ECS task definitions use the AWS
      asynchronous deletion contract: require no `ACTIVE` run-scoped revision,
      explicit deregister/delete receipts, zero dependants, and either absence
      or owner-tracked `DELETE_IN_PROGRESS` with a 24-hour poll/escalation
      deadline. Provide the tested `orphan-sweep` acceptance subcommand and
      copyable runbook invocation; it performs those closed inventories, emits
      only a sanitized count/receipt, and returns non-zero for an
      empty/malformed run ID, API failure, or any unapproved survivor.
      `CLEANUP_REQUIRED=0` is set only after that command
      succeeds and all other cleanup surfaces pass.
- [ ] Verify (run against the new file `F=docs/runbooks/aws-ecs-deployment.md`): `grep -q "AWS Secrets Manager" "$F" && grep -q "Azure equivalent" "$F" && grep -q "elspeth health" "$F" && grep -qi "not wired" "$F" && grep -qi "cognito\|oidc" "$F" && grep -qi "bedrock" "$F" && grep -q "INSTALL_EXTRAS" "$F" && grep -q "healthCheckGracePeriodSeconds" "$F" && grep -q "startPeriod" "$F" && grep -qi "minimum ACU\|min.*ACU" "$F"`; `test "$(grep -c "auth.db" "$F")" -ge 3`. Guard the rollout/rollback material separately: `grep -q "DEPLOYMENT_MODE" "$F" && grep -q "FIRST_DEPLOY_LISTENER_RULE_ARN" "$F" && grep -q "FIRST_DEPLOY_DISABLED_ACTIONS" "$F" && grep -q "DOCTOR_TASK_DEFINITION" "$F" && grep -q "DOCTOR_CONTAINER_NAME" "$F" && grep -q "DOCTOR_NETWORK_CONFIGURATION" "$F" && grep -q "DOCTOR_OVERRIDES" "$F" && grep -q "OBSERVATION_START_EPOCH_MS" "$F" && grep -q "ACCEPTANCE_START_UTC" "$F" && grep -q "ECS_DEPLOYMENT_EVENT_RULE" "$F" && grep -q "ECS_DEPLOYMENT_EVENT_TARGET_ID" "$F" && grep -q "ECS_DEPLOYMENT_EVENT_LOG_GROUP" "$F" && grep -q "WEB_LOG_GROUP" "$F" && grep -q "WEB_LOG_STREAM_PREFIX" "$F" && grep -q "DOCTOR_LOG_GROUP" "$F" && grep -q "DOCTOR_LOG_STREAM_PREFIX" "$F" && grep -q "exact.*taskDefinitionArn" "$F" && grep -Fq -- "--launch-type FARGATE" "$F" && grep -q "application-autoscaling describe-scalable-targets" "$F" && grep -q 'TargetType == "ip"' "$F" && grep -q 'HealthCheckEnabled == true' "$F" && grep -q 'HealthCheckPath == "/api/ready"' "$F" && grep -q 'Matcher.HttpCode == "200"' "$F" && grep -q 'HealthCheckTimeoutSeconds >= 6' "$F" && grep -q "filter-log-events" "$F" && grep -q "run-task" "$F" && grep -q "(.failures | length) == 0" "$F" && grep -q "DOCTOR_TASK_ARN" "$F" && grep -q "select(.essential == true)" "$F" && grep -q "exitCode == 0" "$F" && grep -Fq -- "--deployment-configuration '{\"deploymentCircuitBreaker\":{\"enable\":true,\"rollback\":true},\"minimumHealthyPercent\":0,\"maximumPercent\":100}'" "$F" && grep -q "services-stable" "$F" && grep -q "SERVICE_DEPLOYMENT_FAILED" "$F" && grep -q "list-tasks" "$F" && grep -q "describe-tasks" "$F" && grep -q "loadBalancers" "$F" && grep -q "privateIpv4Address" "$F" && grep -q "any old target may be ignored only while.*draining" "$F" && grep -q -- "--connect-timeout" "$F" && grep -q -- "--max-time" "$F" && grep -q -- "--max-redirs 0" "$F" && grep -q "%{http_code}" "$F" && grep -q "status.*200" "$F" && grep -q "every 30 seconds for 20 iterations" "$F" && grep -q "readiness_check_not_ready" "$F" && grep -q "session.*only when it is MISSING" "$F" && grep -q "partially present session.*STALE" "$F" && grep -q "landscape.*verified-shape PARTIAL" "$F" && grep -q "structural-only" "$F" && grep -q "semantics-only schema-epoch change" "$F" && grep -q "release/schema compatibility record" "$F" && grep -qi "code rollback cannot undo" "$F" && grep -qi "fail closed" "$F" && grep -qi "drop/recreate" "$F"`.
- [ ] Guard the Cognito operator contract explicitly:
  `grep -q "OIDC_AUTHORIZATION_ALLOWED_ORIGINS" "$F" && grep -q "OIDC_TOKEN_ENDPOINT" "$F" && grep -q "OIDC_AUDIENCE_CLAIM" "$F" && grep -q "client_id" "$F" && grep -q "token_use=access" "$F" && grep -qi "authorization.code" "$F" && grep -q "S256\|PKCE" "$F" && grep -qi "no client secret" "$F" && grep -qi "implicit.*not\|not.*implicit" "$F" && grep -qi "wildcard" "$F" && grep -q "describe-user-pool-client" "$F" && grep -q "AllowedOAuthFlowsUserPoolClient" "$F" && grep -q "AllowedOAuthFlows" "$F" && grep -q "AllowedOAuthScopes" "$F" && grep -q "CallbackURLs" "$F"`.
- [ ] Guard the final single-task/recovery prerequisites explicitly: `grep -q "AWS CLI v2" "$F" && grep -q "first-recovery" "$F" && grep -q "short cluster/service names" "$F" && grep -Fq 'service/$ECS_CLUSTER/$ECS_SERVICE' "$F" && grep -q "empty.*ScalableTargets" "$F" && grep -q "desiredCount == 1" "$F" && grep -q "minimumHealthyPercent.*0" "$F" && grep -q "maximumPercent.*100" "$F"`.
- [ ] Guard the platform, health, Exec, durable-signal, and launch contracts:
  `grep -q "TARGET_PLATFORM" "$F" && grep -q "runtimePlatform" "$F" && grep -q "force-new-deployment" "$F" && grep -q "enableExecuteCommand" "$F" && grep -q "ExecuteCommandAgent" "$F" && grep -q "http.client" "$F" && grep -q "startPeriod.*150" "$F" && grep -q "healthCheckGracePeriodSeconds.*150" "$F" && grep -q "describe-resource-policies" "$F" && grep -q "delivery.logs.amazonaws.com" "$F"`.
- [ ] Guard the compatibility authority: `grep -q "database operator.*author/approver" "$F" && grep -q "release operator.*countersigner" "$F" && grep -q "rollback_permitted" "$F" && grep -q "semantics-only changes" "$F" && grep -q "candidate git SHA" "$F" && grep -q "immutable image digest" "$F" && grep -q "Unknown or unapproved compatibility is NO-GO" "$F"`.
- [ ] Guard the disposable cleanup lifecycle:
  `grep -q "teardown deadline" "$F" && grep -q "Terraform" "$F" && grep -q "Cognito test identity" "$F" && grep -q "rollback-baseline ECR tag" "$F" && grep -q "Cleanup failure is itself NO-GO" "$F"`.
- [ ] Insert a row into `docs/runbooks/index.md`'s Quick Reference table: `| [AWS ECS Deployment](aws-ecs-deployment.md) | Deploying ELSPETH web to AWS ECS Fargate with Aurora PostgreSQL |`.
- [ ] Add a link under `docs/guides/docker.md`'s "See Also": `- [AWS ECS Deployment Runbook](../runbooks/aws-ecs-deployment.md) - Production ECS/Fargate deployment contract`.
- [ ] Create `tests/unit/web/test_aws_ecs_runbook_contract.py`. It must extract
  every Bash fence and pass each through `bash -n`, pin the core positive
  contracts above, and reject unsafe regressions: unforced desired-zero
  launch, raw log retention, premature candidate promotion, missing task
  architecture/health/Exec checks, or **container health-command** use of
  `elspeth health` or `/api/ready` (ALB/public readiness still requires
  `/api/ready`). It specifically rejects the obsolete `curl -fsS` probe form.
  Run
  `uv run pytest tests/unit/web/test_aws_ecs_runbook_contract.py tests/unit/test_build_push_release_checks.py -v`
  and require all tests pass.
- [ ] `git add docs/runbooks/aws-ecs-deployment.md docs/runbooks/index.md docs/guides/docker.md tests/unit/web/test_aws_ecs_runbook_contract.py && git commit -m "docs: add executable AWS ECS deployment runbook"`. Do not bypass hooks; the runbook is operational code.

### Task 3: Reproducible live persistence/auth acceptance harness

**Files:**
- Create: `src/elspeth/web/aws_ecs_acceptance.py`
- Create: `tests/unit/web/test_aws_ecs_acceptance.py`
- Modify: `src/elspeth/web/config.py`
- Modify: `src/elspeth/web/app.py`
- Modify: `tests/unit/web/test_app.py`
- Create: `src/elspeth/web/frontend/playwright.oidc.config.ts`
- Create: `src/elspeth/web/frontend/tests/e2e/aws-ecs-oidc.staging.spec.ts`
- Create: `src/elspeth/web/frontend/tests/e2e/harness/oidc-evidence.ts`
- Create: `src/elspeth/web/frontend/tests/e2e/harness/oidc-evidence.test.ts`
- Create: `src/elspeth/web/frontend/tests/e2e/harness/oidc-redacting-reporter.ts`
- Create: `src/elspeth/web/frontend/tests/e2e/harness/oidc-redacting-reporter.test.ts`
- Modify: `src/elspeth/web/frontend/package.json`
- Modify: `docs/runbooks/aws-ecs-deployment.md` (harness commands and evidence fields)

**Interface:** `python -m elspeth.web.aws_ecs_acceptance` with subcommands
`capture`, `verify-api`, `verify-payloads`, `verify-local-auth`, `verify-s3`,
`verify-bedrock`, `verify-bedrock-guardrails`, `verify-operator-telemetry`,
`extract-exec-receipt`, `sanitize-evidence`,
`control-manifest`, `gate-ledger`, `receipt-store`, `approval-verify`,
`scenario-load`, `validate-task-definition-policy`, `orphan-sweep`, and
`cleanup-evidence-finalize`.
This is
operator tooling shipped in the production package so the exact same module
can drive the public API from the release workspace and validate payload/EFS
state from inside the running task. It is not an HTTP endpoint and exposes no
  new remote surface.

- [ ] Extract `app.py`'s private `_settings_from_env` and collection-field
  handling into one public, tested `settings_from_env()` helper in
  `web/config.py`; update app startup and the acceptance module to call that
  exact helper. Do not duplicate environment parsing or import an app-private
  function. Preserve Plan 13's tuple-field registry, explicit-field provenance,
  unknown-key, JSON-collection, null, and raw-input-redacted failure tests in
  `test_app.py` plus the new harness tests.

- [ ] RED: write unit tests that pin every interface mode above and initially fail with
  `ModuleNotFoundError`. Tests must prove: credentials/tokens are read only
  from `ELSPETH_ACCEPTANCE_USERNAME` + `ELSPETH_ACCEPTANCE_PASSWORD` or
  `ELSPETH_ACCEPTANCE_BEARER_TOKEN`, never CLI args or output; the state file
  is mode 0600 and contains only non-secret IDs/hashes/status; HTTP failures,
  timeouts, malformed JSON, non-terminal/failed runs, missing artifacts, hash
  mismatch, zero payload refs, payload retrieval/integrity failure, non-local
  auth, missing `auth.db`, or journal mode other than `delete` exit non-zero
  with redacted class/check detail. `verify-bedrock` tests additionally prove
  missing/invalid model or region, provider failure, empty content, malformed
  response metadata, and any credential/ARN/request-id sentinel fail with only
  a static check/class message; no raw provider text reaches JSON/stderr.
  `verify-s3` tests pin default-chain-only source/sink construction, bounded
  object/hash/collision checks, `finally` cleanup, and static redacted failures.
  `verify-bedrock-guardrails` pins safe/intervened decisions for both explicit
  controls, numeric versions, default-chain-only calls, audit-first records,
  payload-free receipts, and the candidate composer's target-LLM context exposes
  exactly the enabled/preferred security-control inventory with approved
  reference placeholders only. `verify-operator-telemetry` pins one web metric and
  one pipeline lifecycle trace, bounded CloudWatch/X-Ray polling, exact run
  correlation back to Landscape, forbidden-content scans, and static failures.
  `sanitize-evidence` tests feed credential, URL, provider-response, log, and
  task-definition sentinels and prove that only the closed evidence schema is
  emitted. Run
  `uv run pytest tests/unit/web/test_aws_ecs_acceptance.py -v` and require the
  expected red result.
- [ ] Implement the shared HTTP boundary first. Read the one base origin only
  from `ELSPETH_ACCEPTANCE_BASE_URL`; require an exact normalized origin with
  no userinfo, path, query, or fragment, and HTTPS except the exact loopback
  hosts `127.0.0.1`, `::1`, or `localhost` (`urlsplit().hostname` returns the
  IPv6 host without brackets). Reject ambiguous auth input:
  local username/password is one mode, bearer token is the other, and mixed or
  partial credentials fail before any request. Use a client with redirects
  disabled and same-origin response validation: connect 5s, read 15s, write
  10s, pool 5s; 1 MiB maximum JSON/error response; 8 MiB maximum expected
  blob/artifact response; five-minute overall run-poll deadline with a
  one-second bounded interval. These constants are named and unit-tested;
  no credential may be replayed to a redirect or different origin. Tests must
  exercise userinfo, suffix-host, port, path/query/fragment, redirect, timeout,
  oversized body, malformed JSON, and cross-origin cases.
- [ ] Implement `capture --state-file PATH`. In the disposable local-auth scenario only,
  `ELSPETH_ACCEPTANCE_REGISTER=1` permits one `POST /api/auth/register` using
  the environment credentials (200 creates the user and returns its token;
  409 falls back to
  `/login`; any other status fails); `verify-api` never registers: in local
  mode it logs in, while bearer mode uses only the supplied bearer token.
  Never log
  the credential/token. Create a
  session; upload the fixed bytes `id,name\n1,alpha\n`; compute their SHA-256;
  import a fixed no-LLM CSV-source→CSV-sink pipeline through
  `POST /api/sessions/{session_id}/state/yaml`, binding the uploaded blob with
  `source_blob_ids`; execute through
  `POST /api/sessions/{session_id}/execute`; poll `GET /api/runs/{run_id}` to a
  bounded terminal deadline; require `completed`, positive source rows, zero
  failed tokens, and a non-null `landscape_run_id`; fetch `/results`,
  `/outputs`, the downloadable sink artifact content, and the original blob
  content. Persist only session/blob/run/landscape/artifact IDs, the exact
  uploaded/blob/artifact SHA-256 values, accounting assertions, and timestamps
  to the state file. Write under `umask 077` using a same-directory exclusive
  mode-0600 temporary regular file, reject symlink/non-regular/foreign-owned or
  permissive destinations, fsync file and directory, and atomically replace.
  Reads are no-follow, size-bounded, exact-schema validated, and validate every
  UUID, SHA-256, enum/count, and timestamp before use. Never persist bearer
  tokens, passwords, URLs, paths, response headers, or raw content.
- [ ] Pin the fixed YAML independently of mocked HTTP: parse it through
  `composition_state_from_runtime_yaml`, run the ordinary pipeline validation,
  and assert explicit CSV schema/options, sink mode/collision policy, and an
  allowlisted unique relative sink path such as
  `outputs/aws-ecs-acceptance-<session-id>.csv`. A mock accepting arbitrary
  YAML is not evidence that the live importer/runtime will execute it.
- [ ] Implement `verify-api --state-file PATH`. Re-authenticate from the
  current environment after task replacement, then require the session, blob
  metadata/content, terminal run results, outputs manifest, and sink content
  are still retrievable and byte/hash-identical to the capture record. This
  mode must not mutate or delete the acceptance session.
- [ ] Implement `verify-payloads --landscape-run-id ID` for execution in a
  dedicated one-shot `PAYLOAD_VERIFIER_TASK_DEFINITION`, using the candidate
  digest, same PostgreSQL/EFS settings, explicit `user: "1000:1000"`, and a
  Python module entrypoint. Do not use ECS Exec: it runs as root and would mask
  the non-root EFS permission contract. Load `WebSettings` through the same generic environment
  loader as the app; open the configured landscape DB read-only with the
  configured passphrase; select every non-null `rows.source_data_ref` for the
  exact landscape run through `LandscapeDB.from_url(..., create_tables=False,
  read_only=True)`; close it on every path. Require the configured payload root
  already exists as the expected mounted directory **before** constructing
  `FilesystemPayloadStore` (its constructor otherwise creates a missing path);
  require at least one ref and retrieve every ref through that store, thereby using
  its built-in hash/integrity check. Emit only JSON counts and content hashes,
  never row payloads, DB URLs, paths, or secrets.
- [ ] Implement `verify-local-auth` for execution only as a one-shot verifier
  task while the web service is drained to fixed 503 and scaled to zero. Local
  auth's EFS SQLite contract is one-task/one-process; opening it through ECS
  Exec beside uvicorn is forbidden. The verifier task definition uses the
  candidate digest and same EFS/settings, overrides the Docker image entrypoint
  explicitly to `python -m elspeth.web.aws_ecs_acceptance verify-local-auth`,
  and is checked like the doctor task. Load settings, require
  `auth_provider == "local"`, require
  `data_dir / "auth.db"` exists, open it read-only, and require
  `PRAGMA journal_mode` is `delete`. Emit only the check names/results. The
  surrounding runbook separately binds `data_dir` to the approved EFS mount;
  this mode must not claim it can identify the filesystem type from inside the
  container. Open SQLite with a URI `mode=ro` connection so a missing database
  cannot be created, and close it on every path.
- [ ] Implement `verify-s3` for execution inside every healthy candidate web
  task through ECS Exec. Require only `ELSPETH_ACCEPTANCE_S3_BUCKET`, a
  UUID-scoped `ELSPETH_ACCEPTANCE_S3_PREFIX`, and
  `AWS_REGION`/`AWS_DEFAULT_REGION`; reject endpoint/profile,
  static-key/session-token, or role-override inputs. Exercise the shipped
  `AWSS3Sink` and `AWSS3Source` through their ordinary plugin APIs and AWS
  default chain: bounded fixed write/read/hash, conditional-collision check,
  and cleanup in `finally`. Emit only counts, check names, and hashes. The
  acceptance task role is least-privilege `GetObject`/`PutObject`/`DeleteObject`
  on that prefix; missing default-chain credentials, permission, integrity, or
  cleanup is NO-GO.
- [ ] Implement `verify-bedrock` for execution **inside the web task**. Require
  `ELSPETH_BEDROCK_LIVE_TEST_MODEL` in `bedrock/<model-id>` form and region from
  `AWS_REGION` or `AWS_DEFAULT_REGION`; accept neither credential nor endpoint
  arguments. Await the composer's production `_litellm_acompletion` with one
  fixed prompt and a bounded 60-second deadline, allowing boto3/LiteLLM to use
  the task role through the ordinary default chain. Require non-empty content
  and pass the real response through the ordinary composer token/cost/cache,
  safe-model, and request-id parsing path. Emit only a bounded JSON receipt:
  check status, returned-model hash, provider-request-id hash, presence/absence
  flags for prompt/completion/cache counts, finite non-negative cost when
  available, and the cost-source enum. Never emit content, model id, request
  id, credentials, account/role ARN, raw provider response, or raw exception.
  Suppress LiteLLM debug/callback output and capture all library stdout/stderr
  before it reaches ECS Exec; sentinel tests at the file-descriptor boundary
  must prove provider text, request IDs, ARNs, credentials, and model IDs never
  escape on success or failure.
- [ ] Implement `verify-bedrock-guardrails` inside the web task by adapting
  Plan 15C's reusable `guardrails_live_check`. Load the two approved opaque
  profile aliases through the same frozen WebSettings/profile resolver as the
  application; never accept raw identifiers/versions/regions as command
  arguments or emit their resolved values. Require safe and approved
  attack/content sentinel outcomes, `guard_content` on every request, and
  blocking when `detected=true` even if AWS detect mode returns top-level
  `NONE`. Before either call, compile the candidate task's complete
  seven-setting web policy bundle through the application's ordinary resolver.
  Recompute and require the protected scenario inventory's policy-binding
  hash. Require the tutorial profile's private model and region to equal the
  live Bedrock smoke target and resolved AWS region, while keeping both out of
  the receipt. Require the tutorial profile to be usable but the canonical
  core-only tutorial to remain fail-closed with
  `tutorial_required_control_coverage`; do not auto-insert controls. Require
  both Bedrock controls to be locally available and operator-preferred,
  both modes to be `required`, and both default opaque profile aliases to
  match the live inputs. Persist that exact `WebPluginPolicyEvidence`
  atomically with the acceptance run, read it back from Landscape, and fail on
  any difference. Prove each external call is present in Landscape before its
  payload-free operator telemetry. The sanitized receipt adds only
  `policy_hash`, `snapshot_hash`, policy-binding hash, tutorial-profile-ready
  and intentional tutorial-blocker fields, tutorial/target-LLM IDs, selected
  plugin IDs, safe aliases, required modes, immutable numeric Guardrail
  versions, and `landscape_evidence: true`; it never emits model, Guardrail
  identifier/region, raw setting, or provider values.
  The controller-side `validate-task-definition-policy` command reads each
  returned candidate, doctor, payload-verifier, local-auth-verifier, previous,
  and rollback-doctor task-definition document and compares the named
  container's seven raw settings, binding hash, live model, and AWS region
  byte-for-byte with the protected bound scenario inventory. A substituted
  bundle with a freshly substituted self-consistent hash is a failure.
  Preserve Plan 15C's exact live-test contract in the runbook/controller lane:
  marker `live_aws`, env gate
  `ELSPETH_RUN_LIVE_BEDROCK_GUARDRAILS=1`, and command
  `uv run --extra aws pytest tests/integration/plugins/transforms/aws/test_bedrock_guardrails_live.py -m live_aws -q -rs`.
  The protected environment also supplies the application-owned
  `ELSPETH_WEB__PLUGIN_ALLOWLIST` and
  `ELSPETH_WEB__BEDROCK_GUARDRAIL_PROFILES`, plus
  `ELSPETH_LIVE_BEDROCK_{PROMPT,CONTENT}_PROFILE_ALIAS`,
  `ELSPETH_LIVE_BEDROCK_{PROMPT,CONTENT}_{SAFE,BLOCKED}_TEXT`, and
  `ELSPETH_LIVE_BEDROCK_{PROMPT,CONTENT}_EXPECTED_VERSION`. Values are never
  copied into command arguments, receipts, logs, or evidence.
  The test must fail (not skip) when the env gate is set but an approved
  alias/fixture/policy input is absent. Unit tests pin this command and the
  acceptance subcommand to the same reusable checker. The runbook captures
  output through the protected wrapper, requires a non-zero passed count, and
  rejects the words `skipped` or `deselected`; `pytest -rs` reporting alone is
  not a zero-skip assertion. Adapt the Plan 15C callable directly rather than
  forking its request/parser/receipt logic. Plan 10 owns the acceptance-module,
  task-role/IAM, image, task-definition, and runbook integration; Plan 12
  executes both proofs and owns the verdict.
- [ ] Implement `verify-operator-telemetry`. Emit a uniquely hashed web metric,
  run one lifecycle-only pipeline, then use bounded AWS API polling to locate
  the metric and `RunStarted`/`RunFinished` trace for the exact run correlation.
  Query Landscape for the same run and require the terminal status agrees.
  Scan the sanitized signal projection for prompt/content/credential sentinels,
  and add a disposable negative lane proving a collector outage leaves the
  already committed Landscape record intact while export health degrades.
  A positive closed receipt includes only the exact allowlisted CloudWatch
  metric query (with its acceptance-run/scenario namespace) and deterministic
  X-Ray trace ID. `control-manifest checkpoint-operator-evidence` consumes that
  receipt and creates plus binds the next immutable strict-superset retained-
  evidence checkpoint before the live helper returns. Raw CloudWatch/X-Ray
  responses are never printed or persisted.
- [ ] Give `verify-s3` and `verify-bedrock` one shared ECS-Exec transport
  contract. Their only process output is exactly one bounded
  `ELSPETH_ACCEPTANCE_RECEIPT_V1:<base64url-json>` sentinel whose decoded JSON
  has a closed schema, names the requested check, and has `ok: true`. The
  local `extract-exec-receipt` binds it to the expected candidate SHA, exact
  task ARN, and scenario ID in the sanitized evidence envelope. For
  `verify-bedrock-guardrails`, it additionally requires the receipt's
  `plugin_policy.binding_sha256` to equal the controller's protected inventory
  value, and `receipt-store` repeats that manifest-backed comparison before
  persistence. The runbook's
  `run_candidate_role_checks` captures the interactive Session Manager stream
  without terminal echo, requires exactly one sentinel, decodes and validates
  it locally, and persists only the sanitized receipt. Missing, duplicate,
  malformed, oversized, wrong-task, wrong-SHA, or `ok: false` receipts fail.
  Unit tests include Session Manager banners/noise and prove no provider,
  credential, ARN, model, URL, or raw-error sentinel can escape either the
  remote process or local extractor.
- [ ] Implement a durable, interruption-safe `control-manifest` helper for
  Plan 12. `init`, `update`, `validate`, `bind-scenario`,
  `bind-retained-evidence`, `checkpoint-operator-evidence`, closed-field `get`,
  and `load-cleanup` operate on an
  owner-only mode-0600 regular file using no-follow reads, a same-directory
  exclusive temporary file, fsync, and atomic replace. The closed schema binds
  the acceptance run ID, candidate SHA, release PR number, exact-SHA CI run
  ID, two separately named scenario
  inventory/state identities, external IaC revisions and binding hashes,
  AWS account/region, run-scoped ECR tags/digests, evidence paths, teardown
  deadline, `cleanup_required`, and per-surface cleanup states. Secret values,
  raw Terraform plans/state, credentials, URLs, usernames, and commands are
  forbidden. `load-cleanup` emits only a closed shell-assignment allowlist and
  rejects foreign-owned, permissive, symlinked, malformed, or inconsistent
  manifests. Expiry blocks ordinary validation and all further acceptance,
  but `load-cleanup` still enters cleanup-only mode, emits
  `DEADLINE_EXPIRED=1`, and supplies the teardown identities; an overdue
  manifest must never make emergency cleanup impossible or permit return to
  Task 7. `validate --cleanup-only --require-cleanup-cleared` is the sole
  allow-expired validation mode: it verifies the final cleared state and
  preserved deadline-failure record from a fresh process while continuing to
  forbid Task 7. Every mutation/checkpoint is atomic so a new shell or emergency
  owner can resume Task 8 after INT/TERM, deadline expiry, or process loss.
- [ ] Own every Plan-12 lifecycle helper as a tested acceptance-module
  subcommand plus copyable runbook wrapper: `receipt-store` implements
  `persist_sanitized_receipt`; `approval-verify` implements
  `require_signed_tf_plan_approval` and
  `require_signed_tf_destroy_approval`; `scenario-load` implements
  `load_scenario`; `orphan-sweep` replaces the in-shell orphan function; and
  `cleanup-evidence-finalize` has a prepare phase that verifies durable copies
  before local deletion and a commit phase that appends final cleanup
  manifest/ledger/receipt hashes, verifies checksums, refuses unless every
  required cleanup state is confirmed, and only then atomically clears
  `cleanup_required`. Approval
  verification binds scenario ID, plan-receipt hash, acceptance run ID,
  approver identity/authority, decision, timestamp, and expiry and accepts only
  an unexpired explicit approval. Scenario loading clears prior generic
  variables, validates every resource against run and scenario IDs, and emits
  a closed assignment allowlist. Every helper has hostile-file/schema,
  wrong-run/scenario, expired approval, interruption, idempotent-resume, and
  redaction tests; the runbook contract test proves all wrapper names are
  defined before first use.
- [ ] Implement `gate-ledger init|get|record|record-cleanup|bind-candidate|finalize`
  with the same protected-file discipline. Initialization durably binds the
  reviewed Plan 12 SHA-256, immutable program-base SHA, reconciled-release
  SHA, branch, and starting SHA; closed-field `get` reconstructs only those
  non-secret anchors plus the bound candidate for process-safe resume. A
  record contains only a stable command/check ID, candidate SHA, start/end UTC,
  exit status, and sanitized receipt/output hash; it never stores an expanded
  command line or raw stdout/stderr. Tasks 1–7 use one strict success stream;
  Task 8 uses an independent strict cleanup stream so a post-mutation Task 7
  failure can still complete and prove teardown without fabricating skipped
  success rows. Plan 12 records every Task 1–8 checkbox, finalizes one checksum
  over both streams, and exports that manifest as the exact-command/exit-status
  evidence ledger.
- [ ] Implement `sanitize-evidence --kind web-log|doctor-log|deployment-event|task-definition|terraform-plan|terraform-destroy-plan`.
  It reads a bounded JSON document from stdin and emits a closed, allowlisted
  receipt containing only timestamps, event/check/class names, severities,
  task/deployment revision identities, counts, and booleans. It never emits
  free-form message, URL, command, environment value, exception text, provider
  text, or raw AWS response. The runbook must pipe `filter-log-events` and
  other diagnostic JSON directly through this projector before persistence;
  raw output is neither printed nor retained. Define a copyable `aws_capture`
  wrapper (and use it for **every** AWS CLI call in rollout, acceptance, and
  cleanup). It captures
  upstream stdout **and stderr** into mode-0600 temporary files without
  printing, enforces a 2 MiB cap on each, and deletes both on every exit. It
  sets bounded AWS CLI connect/read timeouts; in cleanup mode it also enforces
  the control manifest's effective cleanup deadline and a per-call ceiling so
  no AWS request can prevent later independent cleanup surfaces. If the
  original acceptance deadline is expired, `load-cleanup` records NO-GO and
  atomically establishes a bounded emergency-cleanup horizon; AWS cleanup uses
  that horizon rather than refusing to run. On
  exit 0 it releases stdout only to the caller's command substitution,
  allowlisted jq projection, or `sanitize-evidence`; on non-zero it emits only a
  static failure class before securely deleting both buffers. Define a
  separate `aws_ecr_login REGISTRY REGION` helper for the sole secret-output
  exception: it captures AWS CLI and `docker login --password-stdin` stderr in
  protected bounded files, streams the password only through the pipe, never
  returns/persists it, uses `pipefail`, and emits only a static failure class.
  A plain
  `aws ... | sanitize-evidence` pipe is forbidden because AWS CLI stderr would
  bypass the projector. Sentinel tests exercise both channels.
  The runbook also defines `terraform_capture` and `verify_tf_binding`:
  `terraform_capture` applies the same protected bounded-channel rules to
  Terraform JSON and enforces the acceptance deadline minus cleanup reserve
  (or the effective emergency-cleanup deadline in Task 8);
  `verify_tf_binding` fail-closes unless the external clean IaC
  commit, Terraform/provider lock, remote encrypted+locked backend, hashed
  state key, workspace, AWS account/region, vars hash, and combined binding
  hash exactly match the Plan 12 control manifest, with Scenario A and B state
  identities distinct. Raw plan/state JSON is never printed or persisted.
- [ ] GREEN: run
  `uv run pytest tests/unit/web/test_aws_ecs_acceptance.py -v`,
  `uv run ruff check src/elspeth/web/aws_ecs_acceptance.py tests/unit/web/test_aws_ecs_acceptance.py`,
  `uv run ruff format --check src/elspeth/web/aws_ecs_acceptance.py tests/unit/web/test_aws_ecs_acceptance.py`,
  and `uv run mypy src/elspeth/web/aws_ecs_acceptance.py`; all must exit 0.
- [ ] Add a real-browser Cognito/OIDC harness, invoked from the repository root
  as `npm --prefix src/elspeth/web/frontend run test:e2e:oidc`. Its dedicated
  Playwright config targets the deployed
  `STAGING_BASE_URL`, has no local `webServer`, `globalSetup`, or
  pre-authenticated `storageState`, uses `workers=1`, `retries=0`, bounded test
  and navigation timeouts, and has an exact
  `testMatch` for `aws-ecs-oidc.staging.spec.ts` only. It disables
  trace/video/screenshots and uses only the custom redacting console reporter;
  no HTML/blob/JUnit artifact is permitted. The reporter replaces tokens,
  credentials, callback URL/query/fragment, cookies, and headers in titles,
  errors, attachments, stdout, and stderr, with sentinel tests. The spec must:
  fetch `/api/auth/config` and require provider `oidc`, the exact
  `OIDC_EXPECTED_ISSUER`, `oidc_client_id == OIDC_EXPECTED_AUDIENCE`, and a
  HTTPS authorization endpoint whose origin exactly equals
  `OIDC_EXPECTED_AUTHORIZATION_ORIGIN`, plus a HTTPS token endpoint on that
  same exact origin;
  click the ELSPETH **Sign in with SSO** control; fill the approved Cognito
  test account from `OIDC_TEST_USERNAME`/`OIDC_TEST_PASSWORD` using
  `input[name="username"], input#signInFormUsername` and
  `input[name="password"], input#signInFormPassword`, selecting the first
  visible match; submit through
  `input[name="signInSubmitButton"], button:has-text("Sign in")`; wait for
  return to `STAGING_BASE_URL`; require the frontend consumed and removed the
  callback code/state from the URL and removed `oidc_transaction` from session
  storage before accepting or rejecting the callback; require that no access
  token ever appeared in the callback URL and the public-client S256 PKCE token
  exchange completed without a client secret, as implemented and unit-tested
  by Plan 13. Call `/api/auth/me`
  and create/read/delete one session with the acquired bearer token. Decode
  claims only in memory and require exact issuer, non-empty subject, unexpired
  token, and the exact claim named by `OIDC_EXPECTED_AUDIENCE_CLAIM` (`aud` or
  `client_id`) equals `OIDC_EXPECTED_AUDIENCE`; `client_id` mode also requires
  `token_use=access`. Write a mode-0600 `OIDC_EVIDENCE_FILE` containing only
  this exact schema: `phase`, `timestamp`, `issuer`, `authorization_origin`,
  `audience_claim`, `audience`, `subject_sha256`, `auth_me_status` (200),
  `session_create_status` (201), `session_read_status` (200),
  `session_delete_status` (204), and `session_round_trip` (`true`) — no
  token, authorization code, PKCE verifier, username, password, callback URL/query/fragment, cookies, page HTML,
  or headers. The evidence runner keeps no independent token artifact; the
  accepted access token follows ELSPETH's existing auth-store session policy,
  deletes the created
  session in `finally`, and writes evidence only after cleanup succeeds. The
  evidence writer follows the same no-follow, mode-0600, exclusive
  same-directory temporary file, file+directory fsync, atomic-replace, strict
  schema, and bounded-size contract as the Python state file; tests cover
  symlink and permissive/pre-existing destination attacks.
- [ ] Unit-test the OIDC claim/evidence helper with wrong issuer/audience,
  wrong/missing audience-claim mode, wrong `token_use`, missing subject,
  expired token, and secret-redaction cases. Run
  `npm --prefix src/elspeth/web/frontend test -- tests/e2e/harness/oidc-evidence.test.ts`,
  `npm --prefix src/elspeth/web/frontend run typecheck`, and
  `npm --prefix src/elspeth/web/frontend run test:e2e:oidc -- --list`; the list
  command must work without credentials and assert the exact expected spec and
  test count, not merely exit 0. The live spec itself runs in Plan 12
  at four upgrade/rollback checkpoints.
- [ ] Run the repository's incremental contract/trust-boundary and manifest
  gates for every changed Python/TypeScript file, plus the read-only signed
  trust-tier/trust-boundary diagnosis. If signed drift is reported, stop for
  the operator-held signing workflow, then rerun diagnosis and the exact
  required-signature verification. Do not defer first discovery to Plan 12.
  At minimum run:
  ```bash
  uv run python scripts/cicd/generate_skill_inventory.py --check
  uv run python scripts/check_contracts.py
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules manifest.contract_manifest --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules manifest.symbol_inventory,manifest.test_to_source_mapping --root .
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing uv run elspeth-lints diagnose-judge-signatures --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model --format text
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing uv run elspeth-lints check --rules trust_tier.tier_model --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier --root src/elspeth
  ```
  If an operator repairs diagnosed signed YAML, inspect and stage only those
  exact files, never backups, then rerun all scoped tests/static/trust gates on
  the new HEAD. Required-mode verification uses the operator-supplied key in
  the environment and never prints or stores it.
- [ ] Run `wardline scan . --fail-on ERROR`. On exit 1, explain and fix every
  finding at the API/file/DB input boundary, then rescan; exit 2 blocks. No
  baseline or waiver is permitted for this harness.
- [ ] Add the exact `capture` → forced task replacement → `verify-api`,
  explicit-UID one-shot `verify-payloads`, ECS Exec `verify-s3`/`verify-bedrock`,
  and drained explicit-UID one-shot `verify-local-auth` sequence to the runbook, with
  ECS Exec enablement, the local Session Manager plugin, and
  `execute-command`/SSM permissions as explicit prerequisites.
- [ ] Run `git diff --check`, affected auth/session/blob/execution/output/payload
  backend tests, full frontend lint/typecheck/build/Vitest, staged pre-commit
  hooks, and the exact Playwright list gate. A narrow new-test-only pass is not
  closeout evidence.
- [ ] `git add src/elspeth/web/aws_ecs_acceptance.py src/elspeth/web/config.py src/elspeth/web/app.py tests/unit/web/test_aws_ecs_acceptance.py tests/unit/web/test_app.py src/elspeth/web/frontend/playwright.oidc.config.ts src/elspeth/web/frontend/tests/e2e/aws-ecs-oidc.staging.spec.ts src/elspeth/web/frontend/tests/e2e/harness/oidc-evidence.ts src/elspeth/web/frontend/tests/e2e/harness/oidc-evidence.test.ts src/elspeth/web/frontend/tests/e2e/harness/oidc-redacting-reporter.ts src/elspeth/web/frontend/tests/e2e/harness/oidc-redacting-reporter.test.ts src/elspeth/web/frontend/package.json docs/runbooks/aws-ecs-deployment.md && git commit -m "test(aws-ecs): add live persistence, S3, Bedrock, and OIDC acceptance harnesses"`.
