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

### Web plugin policy rollout

Render this complete policy bundle into every candidate, rollback, recovery,
and verifier task definition from the owner-approved protected scenario
inventory. `scenario-load` exports these exact names and the computed
`ELSPETH_ACCEPTANCE_PLUGIN_POLICY_BINDING_SHA256`:

- `ELSPETH_WEB__PLUGIN_ALLOWLIST`
- `ELSPETH_WEB__PLUGIN_PREFERENCES`
- `ELSPETH_WEB__PLUGIN_CONTROL_MODES`
- `ELSPETH_WEB__LLM_PROFILES`
- `ELSPETH_WEB__TUTORIAL_LLM_PROFILE`
- `ELSPETH_WEB__BEDROCK_GUARDRAIL_PROFILES`
- `ELSPETH_WEB__BEDROCK_GUARDRAIL_DEFAULT_PROFILES`

The same protected scenario inventory supplies
`ELSPETH_BEDROCK_LIVE_TEST_MODEL`; it must exactly equal the private model in
the selected tutorial profile, and `AWS_REGION` must equal that profile's
private region.

Values use the JSON/opaque-alias contract in
[`configuration.md`](../reference/configuration.md). Never retain their raw
values or rendered task-definition JSON as evidence. Changing any member
requires the operator to register a new task-definition revision. The operator
must force a new deployment and restart candidate acceptance; ECS Exec cannot
change the process-frozen policy.

`verify-api` requires all six sanitized boot-readiness rows from
`GET /api/system/status`, then rechecks a separately captured canonical
core-only tutorial candidate. Because both controls are required and the
canonical tutorial does not auto-insert them, acceptance requires the typed
HTTP 409 `tutorial_not_ready` / `tutorial_required_control_coverage` response
without creating a tutorial run. `verify-bedrock-guardrails` requires the
effective policy's `tutorial_profile_ready: true`, `tutorial_ready: false`,
exact tutorial blocker and `target_llm` selection, plus prompt-shield and
content-safety entries in `selected_controls` with `required` modes and opaque
aliases. It privately correlates the tutorial model/region with the live
Bedrock smoke and verifies the protected policy binding hash. Its sanitized
receipt retains the binding hash and immutable numeric Guardrail versions,
then sets `landscape_evidence` true only after the atomic
`run_web_plugin_policy` row is read back unchanged.

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
export ELSPETH_COMMAND_OUTPUT_LIMIT_BYTES=2097152
export ELSPETH_AWS_CALL_CEILING_SECONDS=60
export ELSPETH_TERRAFORM_CALL_CEILING_SECONDS=900
export ELSPETH_CLEANUP_RESERVE_SECONDS=5400

protected_timeout_seconds() {
  local kind="$1" ceiling deadline now_epoch deadline_epoch remaining reserve refresh_file
  if test "$kind" = terraform; then
    ceiling="${ELSPETH_TERRAFORM_CALL_CEILING_SECONDS:?set Terraform call ceiling}"
  else
    ceiling="${ELSPETH_AWS_CALL_CEILING_SECONDS:?set AWS call ceiling}"
  fi
  test "$ceiling" -gt 0 2>/dev/null || {
    printf '%s\n' 'command_timeout_invalid' >&2
    return 1
  }
  deadline="${ACCEPTANCE_TEARDOWN_DEADLINE_UTC:-}"
  if test "${ELSPETH_CLEANUP_MODE:-0}" = 1 && test -n "$deadline" && test -z "${EMERGENCY_CLEANUP_DEADLINE_UTC:-}"; then
    now_epoch=$(date -u +%s)
    deadline_epoch=$(date -u -d "$deadline" +%s 2>/dev/null) || {
      printf '%s\n' 'command_deadline_invalid' >&2
      return 1
    }
    if test "$now_epoch" -ge "$deadline_epoch"; then
      test -n "${CONTROL_MANIFEST:-}" || {
        printf '%s\n' 'cleanup_control_manifest_missing' >&2
        return 1
      }
      refresh_file=$(mktemp -p /tmp elspeth-cleanup-timeout.XXXXXX) || return 1
      chmod 600 "$refresh_file" || { rm -f -- "$refresh_file"; return 1; }
      if ! uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest load-cleanup \
          --file "$CONTROL_MANIFEST" --shell-assignments >"$refresh_file"; then
        rm -f -- "$refresh_file"
        return 1
      fi
      . "$refresh_file"
      rm -f -- "$refresh_file"
    fi
  fi
  if test "${ELSPETH_CLEANUP_MODE:-0}" = 1 && test -n "${EMERGENCY_CLEANUP_DEADLINE_UTC:-}"; then
    deadline="$EMERGENCY_CLEANUP_DEADLINE_UTC"
  fi
  if test -z "$deadline"; then
    printf '%s\n' "$ceiling"
    return 0
  fi
  now_epoch=$(date -u +%s)
  deadline_epoch=$(date -u -d "$deadline" +%s 2>/dev/null) || {
    printf '%s\n' 'command_deadline_invalid' >&2
    return 1
  }
  remaining=$((deadline_epoch - now_epoch))
  if test "$kind" = terraform && test "${ELSPETH_CLEANUP_MODE:-0}" != 1; then
    reserve="${ELSPETH_CLEANUP_RESERVE_SECONDS:?set cleanup reserve}"
    test "$reserve" -ge 0 2>/dev/null || {
      printf '%s\n' 'command_timeout_invalid' >&2
      return 1
    }
    remaining=$((remaining - reserve))
  fi
  test "$remaining" -gt 0 || {
    printf '%s\n' 'command_deadline_exhausted' >&2
    return 1
  }
  if test "$remaining" -lt "$ceiling"; then
    printf '%s\n' "$remaining"
  else
    printf '%s\n' "$ceiling"
  fi
}

aws_capture() (
  local stdout_file stderr_file status stdout_size stderr_size timeout_seconds
  test "${1:-}" = aws || {
    printf '%s\n' 'aws_command_invalid' >&2
    exit 1
  }
  stdout_file=$(mktemp)
  stderr_file=$(mktemp)
  trap 'rm -f -- "$stdout_file" "$stderr_file"' EXIT HUP INT TERM
  chmod 600 "$stdout_file" "$stderr_file"
  timeout_seconds=$(protected_timeout_seconds aws) || exit 1
  if (ulimit -f 4096; timeout --signal=TERM --kill-after=5s "${timeout_seconds}s" "$@" \
      --cli-connect-timeout "$AWS_CLI_CONNECT_TIMEOUT" --cli-read-timeout "$AWS_CLI_READ_TIMEOUT") \
      >"$stdout_file" 2>"$stderr_file"; then
    status=0
  else
    status=$?
  fi
  stdout_size=$(stat -c %s "$stdout_file")
  stderr_size=$(stat -c %s "$stderr_file")
  if test "$stdout_size" -gt "$ELSPETH_COMMAND_OUTPUT_LIMIT_BYTES" || test "$stderr_size" -gt "$ELSPETH_COMMAND_OUTPUT_LIMIT_BYTES"; then
    printf '%s\n' 'aws_command_output_too_large' >&2
    exit 1
  fi
  test "$status" -eq 0 || {
    printf '%s\n' 'aws_command_failed' >&2
    exit "$status"
  }
  cat "$stdout_file"
)

aws_ecr_login() (
  local registry="$1" region="$2" aws_stderr docker_stderr status aws_size docker_size timeout_seconds
  [[ "$registry" =~ ^[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(\.cn)?$ ]] || {
    printf '%s\n' 'ecr_login_input_invalid' >&2
    exit 1
  }
  [[ "$region" =~ ^[a-z]{2}(-gov)?-[a-z]+-[0-9]+$ ]] || {
    printf '%s\n' 'ecr_login_input_invalid' >&2
    exit 1
  }
  aws_stderr=$(mktemp)
  docker_stderr=$(mktemp)
  trap 'rm -f -- "$aws_stderr" "$docker_stderr"' EXIT HUP INT TERM
  chmod 600 "$aws_stderr" "$docker_stderr"
  timeout_seconds=$(protected_timeout_seconds aws) || exit 1
  if (
    set -o pipefail
    ulimit -f 4096
    timeout --signal=TERM --kill-after=5s "${timeout_seconds}s" aws ecr get-login-password --region "$region" \
      --cli-connect-timeout "$AWS_CLI_CONNECT_TIMEOUT" --cli-read-timeout "$AWS_CLI_READ_TIMEOUT" 2>"$aws_stderr" |
      timeout --signal=TERM --kill-after=5s "${timeout_seconds}s" docker login --username AWS --password-stdin "$registry" \
        >/dev/null 2>"$docker_stderr"
  ); then
    status=0
  else
    status=$?
  fi
  aws_size=$(stat -c %s "$aws_stderr")
  docker_size=$(stat -c %s "$docker_stderr")
  if test "$aws_size" -gt "$ELSPETH_COMMAND_OUTPUT_LIMIT_BYTES" || test "$docker_size" -gt "$ELSPETH_COMMAND_OUTPUT_LIMIT_BYTES"; then
    printf '%s\n' 'ecr_login_output_too_large' >&2
    exit 1
  fi
  test "$status" -eq 0 || {
    printf '%s\n' 'ecr_login_failed' >&2
    exit "$status"
  }
)

terraform_capture() (
  local stdout_file stderr_file status stdout_size stderr_size timeout_seconds
  stdout_file=$(mktemp)
  stderr_file=$(mktemp)
  trap 'rm -f -- "$stdout_file" "$stderr_file"' EXIT HUP INT TERM
  chmod 600 "$stdout_file" "$stderr_file"
  timeout_seconds=$(protected_timeout_seconds terraform) || exit 1
  if (ulimit -f 4096; timeout --signal=TERM --kill-after=5s "${timeout_seconds}s" terraform "$@") \
    >"$stdout_file" 2>"$stderr_file"; then
    status=0
  else
    status=$?
  fi
  stdout_size=$(stat -c %s "$stdout_file")
  stderr_size=$(stat -c %s "$stderr_file")
  if test "$stdout_size" -gt "$ELSPETH_COMMAND_OUTPUT_LIMIT_BYTES" || test "$stderr_size" -gt "$ELSPETH_COMMAND_OUTPUT_LIMIT_BYTES"; then
    printf '%s\n' 'terraform_command_output_too_large' >&2
    exit 1
  fi
  test "$status" -eq 0 || {
    printf '%s\n' 'terraform_command_failed' >&2
    exit "$status"
  }
  cat "$stdout_file"
)

verify_tf_binding() (
  local scenario_id="$1" directory="$2" vars_file="$3" expected_hash="$4" binding_file="$5"
  local binding_hash repository_commit lock_hash vars_hash terraform_json terraform_version workspace git_status
  local backend_metadata backend_key_hash
  case "$scenario_id" in A|B) ;; *) printf '%s\n' 'terraform_binding_invalid' >&2; exit 1 ;; esac
  [[ "$expected_hash" =~ ^[0-9a-f]{64}$ ]] || {
    printf '%s\n' 'terraform_binding_invalid' >&2
    exit 1
  }
  test ! -L "$directory" && test -d "$directory" && test ! -L "$vars_file" && test -f "$vars_file" || {
    printf '%s\n' 'terraform_binding_file_invalid' >&2
    exit 1
  }
  test ! -L "$binding_file" && test -f "$binding_file" || {
    printf '%s\n' 'terraform_binding_file_invalid' >&2
    exit 1
  }
  test "$(stat -c %u "$binding_file")" = "$(id -u)" && test "$(stat -c %a "$binding_file")" = 600 || {
    printf '%s\n' 'terraform_binding_file_invalid' >&2
    exit 1
  }
  test "$(stat -c %s "$binding_file")" -le 65536 || {
    printf '%s\n' 'terraform_binding_file_invalid' >&2
    exit 1
  }
  jq -e --arg run "$ACCEPTANCE_RUN_ID" --arg scenario "$scenario_id" '
    type == "object" and
    keys == ["acceptance_run_id","aws_account_id","aws_region","backend_encrypted","backend_locked","backend_state_key_sha256","backend_type","repository_commit","scenario_id","schema","terraform_lock_sha256","terraform_version","vars_sha256","workspace"] and
    .schema == "elspeth.aws-ecs-tf-binding.v1" and
    .acceptance_run_id == $run and .scenario_id == $scenario and
    (.repository_commit | test("^[0-9a-f]{40}$")) and
    (.terraform_lock_sha256 | test("^[0-9a-f]{64}$")) and
    (.vars_sha256 | test("^[0-9a-f]{64}$")) and
    (.backend_state_key_sha256 | test("^[0-9a-f]{64}$")) and
    .backend_type == "s3" and .backend_encrypted == true and .backend_locked == true and
    (.terraform_version | type == "string" and length > 0 and length <= 64) and
    (.workspace | test("^[A-Za-z0-9][A-Za-z0-9._-]{0,89}$")) and
    (.aws_account_id | test("^[0-9]{12}$")) and
    (.aws_region | test("^[a-z]{2}(-gov)?-[a-z]+-[0-9]+$"))
  ' "$binding_file" >/dev/null 2>/dev/null || {
    printf '%s\n' 'terraform_binding_schema_invalid' >&2
    exit 1
  }
  binding_hash=$(jq -cS . "$binding_file" 2>/dev/null | sha256sum | awk '{print $1}') || {
    printf '%s\n' 'terraform_binding_hash_invalid' >&2
    exit 1
  }
  test "$binding_hash" = "$expected_hash" || {
    printf '%s\n' 'terraform_binding_hash_mismatch' >&2
    exit 1
  }
  git_status=$(git -C "$directory" status --porcelain=v1 --untracked-files=all 2>/dev/null) || {
    printf '%s\n' 'terraform_repository_invalid' >&2
    exit 1
  }
  test -z "$git_status" || {
    printf '%s\n' 'terraform_repository_dirty' >&2
    exit 1
  }
  repository_commit=$(git -C "$directory" rev-parse --verify HEAD 2>/dev/null) || {
    printf '%s\n' 'terraform_repository_invalid' >&2
    exit 1
  }
  test -f "$directory/.terraform.lock.hcl" || {
    printf '%s\n' 'terraform_lock_missing' >&2
    exit 1
  }
  lock_hash=$(sha256sum "$directory/.terraform.lock.hcl" 2>/dev/null | awk '{print $1}') || {
    printf '%s\n' 'terraform_lock_invalid' >&2
    exit 1
  }
  vars_hash=$(sha256sum "$vars_file" 2>/dev/null | awk '{print $1}') || {
    printf '%s\n' 'terraform_vars_invalid' >&2
    exit 1
  }
  terraform_json=$(terraform_capture -chdir="$directory" version -json) || exit 1
  terraform_version=$(jq -er '.terraform_version | select(type == "string")' <<<"$terraform_json" 2>/dev/null) || {
    printf '%s\n' 'terraform_version_invalid' >&2
    exit 1
  }
  workspace=$(terraform_capture -chdir="$directory" workspace show) || exit 1
  [[ "$workspace" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,89}$ ]] || {
    printf '%s\n' 'terraform_workspace_invalid' >&2
      exit 1
    }
  backend_metadata="$directory/.terraform/terraform.tfstate"
  test ! -L "$backend_metadata" && test -f "$backend_metadata" && test "$(stat -c %s "$backend_metadata")" -le 1048576 || {
    printf '%s\n' 'terraform_backend_metadata_invalid' >&2
    exit 1
  }
  jq -e --arg region "$AWS_REGION" '
    .backend.type == "s3" and
    .backend.config.region == $region and
    (.backend.config.encrypt == true or .backend.config.encrypt == "true") and
    ((.backend.config.use_lockfile == true or .backend.config.use_lockfile == "true") or
     (.backend.config.dynamodb_table | type == "string" and length > 0)) and
    (.backend.config.key | type == "string" and length > 0)
  ' "$backend_metadata" >/dev/null 2>/dev/null || {
    printf '%s\n' 'terraform_backend_live_mismatch' >&2
    exit 1
  }
  backend_key_hash=$(jq -jr '.backend.config.key' "$backend_metadata" 2>/dev/null | sha256sum | awk '{print $1}') || {
    printf '%s\n' 'terraform_backend_live_mismatch' >&2
    exit 1
  }
  jq -e \
    --arg commit "$repository_commit" --arg lock "$lock_hash" --arg vars "$vars_hash" \
    --arg version "$terraform_version" --arg workspace "$workspace" \
    --arg account "$AWS_ACCOUNT_ID" --arg region "$AWS_REGION" --arg backend_key_hash "$backend_key_hash" '
      .repository_commit == $commit and .terraform_lock_sha256 == $lock and
      .vars_sha256 == $vars and .terraform_version == $version and
      .workspace == $workspace and .aws_account_id == $account and .aws_region == $region and
      .backend_state_key_sha256 == $backend_key_hash
    ' "$binding_file" >/dev/null 2>/dev/null || {
      printf '%s\n' 'terraform_binding_live_mismatch' >&2
      exit 1
    }
)
```

A plain `aws ... | sanitize-evidence` pipe is forbidden because AWS stderr
would bypass the projector. The wrappers cap each captured channel at 2 MiB,
delete protected buffers on success, failure, signal, or timeout, and enforce
the control-manifest deadline plus a per-call ceiling. `aws_ecr_login` is the
only secret-output exception: the ECR password exists only in its pipe to
`docker login --password-stdin`.

Before control-manifest initialization, prepare distinct owner-only mode-0600
`SCENARIO_A_TF_BINDING_FILE` and `SCENARIO_B_TF_BINDING_FILE` documents. Each
uses schema `elspeth.aws-ecs-tf-binding.v1` and the exact closed fields enforced
by `verify_tf_binding`: run/scenario, clean repository commit, Terraform lock
hash/version, `s3` backend with encryption and locking asserted, hashed state
key, workspace, account/region, and vars hash. Its canonical sorted JSON hash
is the scenario's `SCENARIO_*_TF_BINDING_SHA`; the receipt stays outside the
IaC worktree so the clean-tree check remains meaningful.

### Closed lifecycle helper wrappers

Define these wrappers before the first Terraform plan, receipt, scenario load,
or cleanup call. The acceptance module validates the mode-0600 control
manifest and emits only hashes or its closed shell-assignment allowlist.
The manifest schema is `elspeth.aws-ecs-control-manifest.v5`; it preserves
each scenario's original pre-apply path/hash separately from the resolved
resource inventory and binds post-observation retained evidence independently.
`ELSPETH_ACCEPTANCE_APPROVAL_KEYRING` names a mode-0600 protected JSON document
with schema `elspeth.aws-ecs-approval-keyring.v1` and a non-empty `keys` map of
approved key IDs to raw 32-byte Ed25519 public keys encoded as unpadded
base64url. Approval files carry the matching key ID and unpadded base64url
signature; absent, malformed, unknown, expired, or invalid authority fails
closed. The manifest retains the protected approval path, hash, and expiry and
reopens them through `approval-require-current` immediately before both apply
and destroy use, and again when an approved apply is recorded; verification
performed before expiry cannot authorize later use after expiry.

Each protected scenario inventory uses
`elspeth.aws-ecs-scenario-inventory.v5` and binds the run ID, candidate SHA,
account, region, scenario ID, Terraform binding, and the closed `values`
assignment set, including the protected binding-receipt path. The initial
immutable `preapply` document leaves provider-generated identities empty; a
different immutable `resolved` document is bound after apply through
`control-manifest bind-scenario` only after plan, exact apply, and no-op
receipts are present. The bind happens for Scenario A before Scenario B begins,
preserves the pre-apply document, and rejects deterministic identity changes;
ordinary scenario loads reject an unresolved inventory. Its additional closed
`orphan_sweep` object records
`tag_key: ACCEPTANCE_RUN_ID`, a non-secret cleanup owner, ECS task-definition
families, listener ARNs, DB instance identifiers, EFS creation tokens/file
systems/access points, secret IDs, log groups, dashboard/alarm names, X-Ray
group/sampling-rule names, EventBridge rule/target identities, owned Bedrock
Guardrail identifiers/versions, the Cognito subject and pool-ownership flag,
and the pre-mutation Transaction Search state hash. Exact retained CloudWatch
metric queries and X-Ray trace IDs do not exist at resource-bind time; instead,
bind mode-0600 `elspeth.aws-ecs-retained-evidence.v1` receipts monotonically.
Every positive
operator-telemetry proof exposes its exact allowlisted metric query and X-Ray
trace ID; the controller immediately creates and binds a new immutable strict-
superset checkpoint before any later live operation. Before Plan 12 records
the `live` phase, revalidate the latest bound checkpoint with
`--require-complete` so both scenarios have non-empty matching counts.
Omitted fields, duplicate identities, count/identity disagreement, drift from
either pre-mutation hash, or a foreign run/scenario fail closed.

```bash
persist_sanitized_receipt() {
  local scenario_id="$1" kind="$2" subject_id="$3" receipt_source="$4"
  if test -f "$receipt_source" && test ! -L "$receipt_source"; then
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance receipt-store \
      --file "$CONTROL_MANIFEST" \
      --scenario-id "$scenario_id" \
      --kind "$kind" \
      --subject-id "$subject_id" \
      --receipt-file "$receipt_source"
  else
    printf '%s' "$receipt_source" | \
      uv run --frozen python -m elspeth.web.aws_ecs_acceptance receipt-store \
        --file "$CONTROL_MANIFEST" \
        --scenario-id "$scenario_id" \
        --kind "$kind" \
        --subject-id "$subject_id" \
        --receipt-stdin
  fi
}

require_signed_tf_plan_approval() {
  local scenario_id="$1" receipt_hash="$2"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance approval-verify \
    --file "$CONTROL_MANIFEST" \
    --scenario-id "$scenario_id" \
    --kind terraform-plan \
    --plan-receipt-hash "$receipt_hash" \
    --approval-file "$TERRAFORM_PLAN_APPROVAL_FILE"
}

require_signed_tf_destroy_approval() {
  local scenario_id="$1" receipt_hash="$2"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance approval-verify \
    --file "$CONTROL_MANIFEST" \
    --scenario-id "$scenario_id" \
    --kind terraform-destroy-plan \
    --plan-receipt-hash "$receipt_hash" \
    --approval-file "$TERRAFORM_DESTROY_APPROVAL_FILE"
}

load_scenario() {
  local scenario_id="$1" assignments
  unset ACTIVE_SCENARIO_ID ACCEPTANCE_RUN_ID DEPLOYMENT_MODE TARGET_PLATFORM \
    AWS_REGION ECS_CLUSTER ECS_SERVICE WEB_CONTAINER_NAME TARGET_GROUP_ARN \
    ELSPETH_WEB__PLUGIN_ALLOWLIST ELSPETH_WEB__PLUGIN_PREFERENCES \
    ELSPETH_WEB__PLUGIN_CONTROL_MODES ELSPETH_WEB__LLM_PROFILES \
    ELSPETH_WEB__TUTORIAL_LLM_PROFILE \
    ELSPETH_WEB__BEDROCK_GUARDRAIL_PROFILES \
    ELSPETH_WEB__BEDROCK_GUARDRAIL_DEFAULT_PROFILES \
    ELSPETH_ACCEPTANCE_PLUGIN_POLICY_BINDING_SHA256 \
    ELSPETH_BEDROCK_LIVE_TEST_MODEL \
    ALB_BASE_URL ALB_ARN CANDIDATE_TASK_DEFINITION DOCTOR_TASK_DEFINITION \
    DOCTOR_CONTAINER_NAME DOCTOR_NETWORK_CONFIGURATION \
    PAYLOAD_VERIFIER_TASK_DEFINITION LOCAL_AUTH_VERIFIER_TASK_DEFINITION \
    ROLLBACK_DOCTOR_TASK_DEFINITION WEB_LOG_GROUP WEB_LOG_STREAM_PREFIX \
    DOCTOR_LOG_GROUP DOCTOR_LOG_STREAM_PREFIX ECS_DEPLOYMENT_EVENT_RULE \
    ECS_DEPLOYMENT_EVENT_TARGET_ID ECS_DEPLOYMENT_EVENT_LOG_GROUP \
    PREVIOUS_TASK_DEFINITION FIRST_DEPLOY_LISTENER_RULE_ARN \
    FIRST_DEPLOY_FORWARD_ACTIONS FIRST_DEPLOY_DISABLED_ACTIONS \
    COGNITO_USER_POOL_ID DB_CLUSTER_IDENTIFIER ELSPETH_TEST_S3_BUCKET \
    OIDC_EXPECTED_ISSUER OIDC_EXPECTED_AUDIENCE \
    OIDC_EXPECTED_AUTHORIZATION_ORIGIN OIDC_EXPECTED_AUDIENCE_CLAIM \
    SCENARIO_TF_DIR SCENARIO_TF_VARS SCENARIO_TF_BINDING_SHA \
    SCENARIO_TF_BINDING_FILE
  assignments=$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance scenario-load \
    --file "$CONTROL_MANIFEST" --scenario-id "$scenario_id" --shell-assignments)
  eval "$assignments"
}

run_orphan_sweep() {
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance orphan-sweep \
    --file "$CONTROL_MANIFEST" --acceptance-run-id "$ACCEPTANCE_RUN_ID"
}

finalize_cleanup_evidence() {
  local phase="$1"
  if test "$phase" = commit; then
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance cleanup-evidence-finalize \
      --file "$CONTROL_MANIFEST" --ledger "$GATE_LEDGER" \
      --phase commit --clear-cleanup-required
  else
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance cleanup-evidence-finalize \
      --file "$CONTROL_MANIFEST" --ledger "$GATE_LEDGER" --phase prepare
  fi
}

# These transitions are the minimum interruption-safe state for a live run.
# Call each one immediately after the named operation succeeds, never before.
arm_external_cleanup() {
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
    --file "$CONTROL_MANIFEST" --cleanup-required true \
    --ecr-registry "$ECR_REGISTRY" --ecr-repository "$ECR_REPOSITORY" \
    --ecr-baseline-tag "$ROLLBACK_BASELINE_TAG" --ecr-candidate-tag "$CANDIDATE_TAG"
}

checkpoint_ecr_digests() {
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
    --file "$CONTROL_MANIFEST" \
    --ecr-baseline-digest "$ROLLBACK_BASELINE_DIGEST" \
    --ecr-candidate-digest "$IMAGE_DIGEST"
}

checkpoint_terraform_plan() {
  local scenario_id="$1" plan_sha="$2" receipt_hash="$3" approval_hash="$4"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
    --file "$CONTROL_MANIFEST" \
    --terraform-plan-receipt "$scenario_id:$plan_sha:$receipt_hash:$approval_hash"
}

checkpoint_terraform_apply() {
  local scenario_id="$1" plan_sha="$2" receipt_hash="$3" approval_hash="$4"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
    --file "$CONTROL_MANIFEST" \
    --terraform-applied "$scenario_id:$plan_sha:$receipt_hash:$approval_hash"
}

checkpoint_terraform_noop_and_bind() {
  local scenario_id="$1" noop_plan_sha="$2" receipt_hash="$3" inventory="$4"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
    --file "$CONTROL_MANIFEST" \
    --terraform-noop-receipt "$scenario_id:$noop_plan_sha:$receipt_hash"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest bind-scenario \
    --file "$CONTROL_MANIFEST" --scenario-id "$scenario_id" --inventory "$inventory"
}

checkpoint_cleanup() {
  local surface="$1" state="$2"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
    --file "$CONTROL_MANIFEST" --cleanup-checkpoint "$surface:$state"
}

bind_initial_evidence_export() {
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
    --file "$CONTROL_MANIFEST" --evidence-export-receipt "$1"
}

bind_final_evidence_export() {
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
    --file "$CONTROL_MANIFEST" --final-evidence-export-receipt "$1"
}

load_cleanup_state() {
  local assignments
  assignments="$(mktemp -p /tmp elspeth-cleanup-state.XXXXXX)"
  chmod 600 "$assignments"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest load-cleanup \
    --file "$CONTROL_MANIFEST" --shell-assignments >"$assignments"
  . "$assignments"
  rm -f -- "$assignments"
}

remove_local_acceptance_images() {
  local ref
  for ref in \
    elspeth:ecs-rollback-baseline \
    elspeth:ecs-0.7.1-closeout \
    "$ECR_REGISTRY/$ECR_REPOSITORY:$ROLLBACK_BASELINE_TAG" \
    "$ECR_REGISTRY/$ECR_REPOSITORY:$CANDIDATE_TAG" \
    "$ROLLBACK_BASELINE_IMAGE" \
    "$CANDIDATE_IMAGE"; do
    test -n "$ref" || continue
    docker image inspect "$ref" >/dev/null 2>&1 && docker image rm "$ref" >/dev/null || true
    docker image inspect "$ref" >/dev/null 2>&1 && return 1
  done
}

remove_local_acceptance_evidence() {
  if test -n "${ACCEPTANCE_STATE:-}"; then
    case "$ACCEPTANCE_STATE" in /tmp/*) rm -f -- "$ACCEPTANCE_STATE" ;; *) return 1 ;; esac
  fi
  if test -n "${OIDC_EVIDENCE_DIR:-}"; then
    case "$OIDC_EVIDENCE_DIR" in /tmp/*) rm -rf -- "$OIDC_EVIDENCE_DIR" ;; *) return 1 ;; esac
  fi
}
```

Use these transitions around the real deployment, not as a substitute for it:

1. Reserve unique run-scoped ECR tags, call `arm_external_cleanup`, then push.
2. Resolve and verify both registry digests, then call
   `checkpoint_ecr_digests`.
3. For Scenario A and then Scenario B, store the sanitized plan receipt and
   signed approval, call `checkpoint_terraform_plan`, recheck the approval,
   apply that exact saved plan, and call `checkpoint_terraform_apply`.
4. Run a no-change plan, store its sanitized receipt, create the protected
   resolved inventory, and call `checkpoint_terraform_noop_and_bind` before
   loading or exercising the scenario.

The controller will reject calls made out of order or against a different
candidate, run, state binding, receipt, approval, inventory, tag, or digest.

### Temporary image publication

The acceptance run uses unique temporary tags for both the Plan 10 rollback
baseline and the frozen candidate. The control manifest is armed before the
first push so interruption always routes to cleanup:

```bash
set -Eeuo pipefail
test -n "${ROLLBACK_BASELINE_SHA:?restore the Plan 10 baseline SHA}"
test "$ROLLBACK_BASELINE_SHA" != "$CANDIDATE_SHA"
git merge-base --is-ancestor "$ROLLBACK_BASELINE_SHA" "$CANDIDATE_SHA"
test "${TARGET_PLATFORM:?}" = linux/amd64 || test "$TARGET_PLATFORM" = linux/arm64

git archive "$ROLLBACK_BASELINE_SHA" | docker buildx build \
  --platform "$TARGET_PLATFORM" --load \
  --label "org.opencontainers.image.revision=$ROLLBACK_BASELINE_SHA" \
  -t elspeth:ecs-rollback-baseline -

export ECR_REGISTRY="${ECR_REGISTRY:?set approved account registry host}"
export ECR_REPOSITORY="${ECR_REPOSITORY:?set approved repository name}"
export ROLLBACK_BASELINE_TAG="acceptance-${ACCEPTANCE_RUN_ID}-baseline-${ROLLBACK_BASELINE_SHA}"
export CANDIDATE_TAG="acceptance-${ACCEPTANCE_RUN_ID}-0.7.1-${CANDIDATE_SHA}"

test "$(aws_capture aws sts get-caller-identity --query Account --output text)" = "$AWS_ACCOUNT_ID"
REPOSITORY_IDENTITY="$(aws_capture aws ecr describe-repositories \
  --region "$AWS_REGION" --repository-names "$ECR_REPOSITORY" \
  --query 'repositories[0].{registryId:registryId,repositoryUri:repositoryUri}' --output json)"
jq -e --arg account "$AWS_ACCOUNT_ID" --arg uri "$ECR_REGISTRY/$ECR_REPOSITORY" \
  '.registryId == $account and .repositoryUri == $uri' <<<"$REPOSITORY_IDENTITY" >/dev/null

for tag in "$ROLLBACK_BASELINE_TAG" "$CANDIDATE_TAG"; do
  listing="$(aws_capture aws ecr list-images --region "$AWS_REGION" \
    --repository-name "$ECR_REPOSITORY" --filter tagStatus=TAGGED --output json)"
  jq -e --arg tag "$tag" '[.imageIds[] | select(.imageTag == $tag)] | length == 0' \
    <<<"$listing" >/dev/null
done

arm_external_cleanup
(
  export DOCKER_CONFIG="$(mktemp -d)"
  chmod 700 "$DOCKER_CONFIG"
  trap 'rm -rf -- "$DOCKER_CONFIG"' EXIT
  aws_ecr_login "$ECR_REGISTRY" "$AWS_REGION"
  docker tag elspeth:ecs-rollback-baseline \
    "$ECR_REGISTRY/$ECR_REPOSITORY:$ROLLBACK_BASELINE_TAG"
  docker tag elspeth:ecs-0.7.1-closeout \
    "$ECR_REGISTRY/$ECR_REPOSITORY:$CANDIDATE_TAG"
  docker push "$ECR_REGISTRY/$ECR_REPOSITORY:$ROLLBACK_BASELINE_TAG"
  docker push "$ECR_REGISTRY/$ECR_REPOSITORY:$CANDIDATE_TAG"
  docker logout "$ECR_REGISTRY"
)

export ROLLBACK_BASELINE_DIGEST="$(aws_capture aws ecr describe-images \
  --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" \
  --image-ids imageTag="$ROLLBACK_BASELINE_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
export IMAGE_DIGEST="$(aws_capture aws ecr describe-images \
  --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" \
  --image-ids imageTag="$CANDIDATE_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
test -n "$ROLLBACK_BASELINE_DIGEST" && test "$ROLLBACK_BASELINE_DIGEST" != None
test -n "$IMAGE_DIGEST" && test "$IMAGE_DIGEST" != None
export ROLLBACK_BASELINE_IMAGE="$ECR_REGISTRY/$ECR_REPOSITORY@$ROLLBACK_BASELINE_DIGEST"
export CANDIDATE_IMAGE="$ECR_REGISTRY/$ECR_REPOSITORY@$IMAGE_DIGEST"
checkpoint_ecr_digests
```

### Saved-plan apply and scenario binding

Apply Scenario A completely before Scenario B. Terraform plan files may
contain secrets: keep them mode 0600 under `/tmp`, apply only the reviewed
saved plan, and delete them immediately. The durable receipt is a sanitized
projection and hash, not the plan itself.

```bash
plan_and_apply_scenario() {
  local scenario_id="$1" directory="$2" vars="$3" expected_binding="$4"
  local binding_file="$5" resolved_inventory="$6"
  local work plan receipt plan_sha receipt_hash approval_hash noop_sha
  work="$(mktemp -d -p /tmp "elspeth-${scenario_id}-tf.XXXXXX")"
  chmod 700 "$work"
  plan="$work/apply.tfplan"
  receipt="$work/apply.receipt.json"

  terraform_capture -chdir="$directory" init -input=false -lock-timeout=5m >/dev/null
  verify_tf_binding "$scenario_id" "$directory" "$vars" "$expected_binding" "$binding_file"
  terraform_capture -chdir="$directory" validate >/dev/null
  terraform_capture -chdir="$directory" plan -input=false -lock-timeout=5m \
    -var-file="$vars" -var="acceptance_run_id=$ACCEPTANCE_RUN_ID" \
    -var="scenario_id=$scenario_id" -var="candidate_image=$CANDIDATE_IMAGE" \
    -var="rollback_baseline_image=$ROLLBACK_BASELINE_IMAGE" -out="$plan" >/dev/null
  chmod 600 "$plan"
  terraform_capture -chdir="$directory" show -json "$plan" | \
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
      sanitize-evidence --kind terraform-plan >"$receipt"
  chmod 600 "$receipt"
  plan_sha="$(sha256sum "$plan" | awk '{print $1}')"
  receipt_hash="$(persist_sanitized_receipt "$scenario_id" terraform-plan "$plan_sha" "$receipt")"
  approval_hash="$(require_signed_tf_plan_approval "$scenario_id" "$receipt_hash")"
  checkpoint_terraform_plan "$scenario_id" "$plan_sha" "$receipt_hash" "$approval_hash"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance approval-require-current \
    --file "$CONTROL_MANIFEST" --scenario-id "$scenario_id" --kind terraform-plan \
    --plan-receipt-hash "$receipt_hash" --approval-hash "$approval_hash"
  terraform_capture -chdir="$directory" apply -input=false -lock-timeout=5m "$plan" >/dev/null
  checkpoint_terraform_apply "$scenario_id" "$plan_sha" "$receipt_hash" "$approval_hash"

  plan="$work/noop.tfplan"
  receipt="$work/noop.receipt.json"
  terraform_capture -chdir="$directory" plan -input=false -lock-timeout=5m \
    -detailed-exitcode -var-file="$vars" \
    -var="acceptance_run_id=$ACCEPTANCE_RUN_ID" -var="scenario_id=$scenario_id" \
    -var="candidate_image=$CANDIDATE_IMAGE" \
    -var="rollback_baseline_image=$ROLLBACK_BASELINE_IMAGE" -out="$plan" >/dev/null
  chmod 600 "$plan"
  terraform_capture -chdir="$directory" show -json "$plan" | \
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
      sanitize-evidence --kind terraform-plan >"$receipt"
  chmod 600 "$receipt"
  noop_sha="$(sha256sum "$plan" | awk '{print $1}')"
  receipt_hash="$(persist_sanitized_receipt "$scenario_id" terraform-noop "$noop_sha" "$receipt")"
  checkpoint_terraform_noop_and_bind "$scenario_id" "$noop_sha" \
    "$receipt_hash" "$resolved_inventory"
  rm -rf -- "$work"
}

plan_and_apply_scenario A "$SCENARIO_A_TF_DIR" "$SCENARIO_A_TF_VARS" \
  "$SCENARIO_A_TF_BINDING_SHA" "$SCENARIO_A_TF_BINDING_FILE" \
  "$SCENARIO_A_RESOLVED_INVENTORY"
load_scenario A
plan_and_apply_scenario B "$SCENARIO_B_TF_DIR" "$SCENARIO_B_TF_VARS" \
  "$SCENARIO_B_TF_BINDING_SHA" "$SCENARIO_B_TF_BINDING_FILE" \
  "$SCENARIO_B_RESOLVED_INVENTORY"
load_scenario B
```

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
  --query "UserPoolClient.{clientId:ClientId,allowedOAuthFlowsUserPoolClient:AllowedOAuthFlowsUserPoolClient,allowedOAuthFlows:AllowedOAuthFlows,allowedOAuthScopes:AllowedOAuthScopes,callbackURLs:CallbackURLs,hasClientSecret:contains(keys(@), 'ClientSecret')}" \
  --output json)

OIDC_REDIRECT_URI="${ALB_BASE_URL}/"
jq -e --arg callback "$OIDC_REDIRECT_URI" '
  keys == ["allowedOAuthFlows","allowedOAuthFlowsUserPoolClient","allowedOAuthScopes","callbackURLs","clientId","hasClientSecret"]
  and .hasClientSecret == false
  and .allowedOAuthFlowsUserPoolClient == true
  and (.allowedOAuthFlows | index("code") != null)
  and (.allowedOAuthFlows | index("implicit") == null)
  and (["openid", "profile", "email"] - .allowedOAuthScopes | length == 0)
  and (.callbackURLs | index($callback) != null)
' <<<"$COGNITO_CLIENT_JSON" >/dev/null
unset COGNITO_CLIENT_JSON
```

`ALB_BASE_URL` is an exact HTTPS origin with no path, query, fragment, or
trailing slash. `OIDC_REDIRECT_URI` is the exact slash-bearing root URL the
frontend sends during authorization and token exchange. ELSPETH disables
Uvicorn's raw request-line access logger so
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

### Real-browser OIDC evidence

Install Chromium through the frontend lock before any AWS mutation, and prove
the dedicated configuration discovers exactly one test without credentials:

```bash
npm --prefix src/elspeth/web/frontend ci
npm --prefix src/elspeth/web/frontend exec -- playwright install chromium
OIDC_LIST_OUTPUT=$(npm --prefix src/elspeth/web/frontend run test:e2e:oidc -- --list)
grep -Fq '[chromium] aws-ecs-oidc.staging.spec.ts' <<<"$OIDC_LIST_OUTPUT"
test "$(grep -Fc '[chromium] aws-ecs-oidc.staging.spec.ts' <<<"$OIDC_LIST_OUTPUT")" = 1
```

The live runner accepts credentials only through the existing environment,
uses no storage state, screenshot, trace, video, or secondary reporter, and
writes one owner-only evidence document only after the created session is
deleted successfully:

```bash
OIDC_EVIDENCE_DIR=$(mktemp -d -p /tmp elspeth-oidc.XXXXXX)
chmod 700 "$OIDC_EVIDENCE_DIR"
uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
  --file "$CONTROL_MANIFEST" --oidc-evidence-dir "$OIDC_EVIDENCE_DIR"

run_oidc_evidence() {
  local phase="$1"
  STAGING_BASE_URL="$ALB_BASE_URL" \
  OIDC_TEST_USERNAME="$OIDC_TEST_USERNAME" \
  OIDC_TEST_PASSWORD="$OIDC_TEST_PASSWORD" \
  OIDC_EXPECTED_ISSUER="$OIDC_EXPECTED_ISSUER" \
  OIDC_EXPECTED_AUDIENCE="$OIDC_EXPECTED_AUDIENCE" \
  OIDC_EXPECTED_AUTHORIZATION_ORIGIN="$OIDC_EXPECTED_AUTHORIZATION_ORIGIN" \
  OIDC_EXPECTED_AUDIENCE_CLAIM="$OIDC_EXPECTED_AUDIENCE_CLAIM" \
  OIDC_EVIDENCE_PHASE="$phase" \
  OIDC_EVIDENCE_FILE="$OIDC_EVIDENCE_DIR/$phase.json" \
    npm --prefix src/elspeth/web/frontend run test:e2e:oidc
  test "$(stat -c '%a' "$OIDC_EVIDENCE_DIR/$phase.json")" = 600
}
```

Run exactly `previous-before-candidate`, `candidate-initial`,
`previous-after-rollback`, and `candidate-after-redeploy`. Each JSON document
contains only `phase`, `timestamp`, `issuer`, `authorization_origin`,
`audience_claim`, `audience`, `subject_sha256`, `auth_me_status: 200`,
`session_create_status: 201`, `session_read_status: 200`,
`session_delete_status: 204`, and `session_round_trip: true`. It never contains a token,
authorization code, verifier, credential, callback URL/query/fragment,
cookie, header, page HTML, or artifact path.

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
        {"name": "ELSPETH_WEB__PLUGIN_ALLOWLIST", "value": "${ELSPETH_WEB__PLUGIN_ALLOWLIST}"},
        {"name": "ELSPETH_WEB__PLUGIN_PREFERENCES", "value": "${ELSPETH_WEB__PLUGIN_PREFERENCES}"},
        {"name": "ELSPETH_WEB__PLUGIN_CONTROL_MODES", "value": "${ELSPETH_WEB__PLUGIN_CONTROL_MODES}"},
        {"name": "ELSPETH_WEB__LLM_PROFILES", "value": "${ELSPETH_WEB__LLM_PROFILES}"},
        {"name": "ELSPETH_WEB__TUTORIAL_LLM_PROFILE", "value": "${ELSPETH_WEB__TUTORIAL_LLM_PROFILE}"},
        {"name": "ELSPETH_WEB__BEDROCK_GUARDRAIL_PROFILES", "value": "${ELSPETH_WEB__BEDROCK_GUARDRAIL_PROFILES}"},
        {"name": "ELSPETH_WEB__BEDROCK_GUARDRAIL_DEFAULT_PROFILES", "value": "${ELSPETH_WEB__BEDROCK_GUARDRAIL_DEFAULT_PROFILES}"},
        {"name": "ELSPETH_ACCEPTANCE_PLUGIN_POLICY_BINDING_SHA256", "value": "${ELSPETH_ACCEPTANCE_PLUGIN_POLICY_BINDING_SHA256}"},
        {"name": "ELSPETH_BEDROCK_LIVE_TEST_MODEL", "value": "${ELSPETH_BEDROCK_LIVE_TEST_MODEL}"},
        {"name": "AWS_REGION", "value": "${AWS_REGION}"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY", "value": "aws-otlp"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_ENVIRONMENT", "value": "production"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_RELEASE", "value": "${ELSPETH_RELEASE_SHA_OR_DIGEST}"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_ECS_CLUSTER", "value": "${ECS_CLUSTER_NAME}"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_ECS_SERVICE", "value": "${ECS_SERVICE_NAME}"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_TASK_DEFINITION_FAMILY", "value": "${ECS_TASK_DEFINITION_FAMILY}"},
        {"name": "ELSPETH_WEB__OPERATOR_TELEMETRY_TASK_DEFINITION_REVISION", "value": "${ECS_TASK_DEFINITION_REVISION}"},
        {"name": "ELSPETH_ACCEPTANCE_RUN_ID", "value": "${ACCEPTANCE_RUN_ID}"},
        {"name": "ELSPETH_ACCEPTANCE_CANDIDATE_SHA", "value": "${CANDIDATE_SHA}"},
        {"name": "ELSPETH_ACCEPTANCE_SCENARIO_ID", "value": "${SCENARIO_ID}"}
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
container, network configuration, log group, and stream prefix. The
manifest-backed validator below also compares the returned named container's
seven policy settings, binding hash, live Bedrock model, and AWS region byte
for byte with the loaded protected scenario inventory; recomputing a matching
hash over a substituted bundle is not sufficient.

```bash
: "${DEPLOYMENT_MODE:?} ${TARGET_PLATFORM:?} ${AWS_REGION:?}"
: "${ECS_CLUSTER:?} ${ECS_SERVICE:?} ${WEB_CONTAINER_NAME:?}"
: "${TARGET_GROUP_ARN:?} ${ALB_BASE_URL:?}"
: "${CANDIDATE_TASK_DEFINITION:?} ${DOCTOR_TASK_DEFINITION:?}"
: "${PAYLOAD_VERIFIER_TASK_DEFINITION:?} ${LOCAL_AUTH_VERIFIER_TASK_DEFINITION:?}"
: "${ROLLBACK_DOCTOR_TASK_DEFINITION:?}"
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

resolve_bound_task_definition() {
  local variable="$1" container_name="$2" reference raw validated resolved
  reference=${!variable}
  raw=$(aws_capture aws ecs describe-task-definition \
    --task-definition "$reference" --output json)
  validated=$(printf '%s' "$raw" \
    | uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
        validate-task-definition-policy --file "$CONTROL_MANIFEST" \
        --scenario-id "$ACTIVE_SCENARIO_ID" --container-name "$container_name")
  resolved=$(jq -er '.task_definition_arn | select(length > 0)' <<<"$validated")
  printf -v "$variable" '%s' "$resolved"
  unset raw validated resolved reference
}

resolve_bound_task_definition CANDIDATE_TASK_DEFINITION "$WEB_CONTAINER_NAME"
resolve_bound_task_definition DOCTOR_TASK_DEFINITION "$DOCTOR_CONTAINER_NAME"
resolve_bound_task_definition PAYLOAD_VERIFIER_TASK_DEFINITION "$WEB_CONTAINER_NAME"
resolve_bound_task_definition LOCAL_AUTH_VERIFIER_TASK_DEFINITION "$WEB_CONTAINER_NAME"
resolve_bound_task_definition ROLLBACK_DOCTOR_TASK_DEFINITION "$DOCTOR_CONTAINER_NAME"
if test "$DEPLOYMENT_MODE" = upgrade; then
  resolve_bound_task_definition PREVIOUS_TASK_DEFINITION "$WEB_CONTAINER_NAME"
fi

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

#### Persistence, replacement, task-role, and local-auth sequence

Run these checks in this order; a later check never substitutes for an earlier
one:

1. In the disposable local-auth scenario, capture one fixed pipeline through
   the public API into a protected state file.
2. Force a distinct candidate deployment and prove the old task was replaced.
3. Re-authenticate from a fresh process and run `verify-api` against that same
   protected state without mutating the session. Require the six-row
   `GET /api/system/status` contract and the typed HTTP 409
   `tutorial_required_control_coverage` recheck as part of this command.
4. Start the explicit `1000:1000` one-shot payload verifier with the candidate
   digest and the same PostgreSQL/EFS settings; do not use root-running ECS
   Exec for this proof.
5. From every contributing healthy candidate task, use ECS Exec for the S3,
   Bedrock, Guardrail, and operator-telemetry checks and locally extract the
   one sanitized receipt sentinel.
6. Drain traffic to fixed 503, scale the service to zero, then start the
   explicit `1000:1000` local-auth verifier against the same EFS mount.

```bash
ACCEPTANCE_STATE=$(mktemp -p /tmp elspeth-aws-ecs-state.XXXXXX)
rm -f "$ACCEPTANCE_STATE"
uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
  --file "$CONTROL_MANIFEST" --acceptance-state-path "$ACCEPTANCE_STATE"
ELSPETH_ACCEPTANCE_REGISTER=1 \
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance capture \
    --state-file "$ACCEPTANCE_STATE"
LANDSCAPE_RUN_ID=$(jq -er '.landscape_run_id' "$ACCEPTANCE_STATE")

PRE_REPLACEMENT_TASK_ARN="$CANDIDATE_TASK_ARN"
aws_capture aws ecs update-service \
  --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
  --task-definition "$CANDIDATE_TASK_DEFINITION" --desired-count 1 \
  --force-new-deployment \
  --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
  >/dev/null
aws_capture aws ecs wait services-stable \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
CANDIDATE_TASK_ARN=$(aws_capture aws ecs list-tasks \
  --cluster "$ECS_CLUSTER" --service-name "$ECS_SERVICE" \
  --desired-status RUNNING --query 'taskArns[0]' --output text)
test "$CANDIDATE_TASK_ARN" != "$PRE_REPLACEMENT_TASK_ARN"

uv run --frozen python -m elspeth.web.aws_ecs_acceptance verify-api \
  --state-file "$ACCEPTANCE_STATE"

PAYLOAD_OVERRIDES=$(jq -cn --arg name "$WEB_CONTAINER_NAME" --arg run "$LANDSCAPE_RUN_ID" \
  '{containerOverrides:[{name:$name,command:["python","-m","elspeth.web.aws_ecs_acceptance","verify-payloads","--landscape-run-id",$run]}]}')
PAYLOAD_TASK=$(aws_capture aws ecs run-task \
  --cluster "$ECS_CLUSTER" --task-definition "$PAYLOAD_VERIFIER_TASK_DEFINITION" \
  --launch-type FARGATE --network-configuration "$DOCTOR_NETWORK_CONFIGURATION" \
  --count 1 --overrides "$PAYLOAD_OVERRIDES" --query 'tasks[0].taskArn' --output text)
aws_capture aws ecs wait tasks-stopped --cluster "$ECS_CLUSTER" --tasks "$PAYLOAD_TASK" >/dev/null
aws_capture aws ecs describe-tasks --cluster "$ECS_CLUSTER" --tasks "$PAYLOAD_TASK" \
  | jq -e '(.failures | length) == 0 and (.tasks | length) == 1 and all(.tasks[0].containers[] | select(.essential == true); .exitCode == 0)' >/dev/null

checkpoint_operator_retained_evidence() {
  local exec_receipt="$1" receipt_hash checkpoint
  test -d "${RETAINED_EVIDENCE_DIR:?set the owner-approved protected retained-evidence directory}"
  test ! -L "$RETAINED_EVIDENCE_DIR"
  test "$(stat -c '%a' "$RETAINED_EVIDENCE_DIR")" = "700"
  receipt_hash=$(sha256sum "$exec_receipt" | awk '{print $1}')
  checkpoint="$RETAINED_EVIDENCE_DIR/operator-${ACTIVE_SCENARIO_ID}-${receipt_hash}.json"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
    control-manifest checkpoint-operator-evidence --file "$CONTROL_MANIFEST" \
    --exec-receipt "$exec_receipt" --checkpoint "$checkpoint"
}

run_candidate_role_check() {
  local task_arn="$1" check="$2" phase="${3:-}" stream receipt_file command
  local -a binding_args=()
  case "$phase" in
    "") command="python -m elspeth.web.aws_ecs_acceptance $check" ;;
    positive|outage)
      test "$check" = verify-operator-telemetry || return 64
      command="python -m elspeth.web.aws_ecs_acceptance $check --phase $phase"
      ;;
    *) return 64 ;;
  esac
  if test "$check" = verify-bedrock-guardrails; then
    binding_args=(--plugin-policy-binding-sha256 "$ELSPETH_ACCEPTANCE_PLUGIN_POLICY_BINDING_SHA256")
  fi
  stream=$(aws_capture aws ecs execute-command \
    --cluster "$ECS_CLUSTER" --task "$task_arn" --container "$WEB_CONTAINER_NAME" \
    --interactive --command "$command")
  receipt_file=$(mktemp -p /tmp elspeth-exec-receipt.XXXXXX)
  chmod 600 "$receipt_file"
  printf '%s' "$stream" | uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
    extract-exec-receipt --check "$check" --candidate-sha "$CANDIDATE_SHA" \
    --task-arn "$task_arn" --scenario-id "$ACTIVE_SCENARIO_ID" \
    "${binding_args[@]}" >"$receipt_file"
  unset stream
  persist_sanitized_receipt "$ACTIVE_SCENARIO_ID" "$check" "$task_arn" "$receipt_file"
  if test "$check" = verify-operator-telemetry && test "$phase" != outage; then
    checkpoint_operator_retained_evidence "$receipt_file"
  fi
  rm -f "$receipt_file"
}

run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-s3
run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock
run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock-guardrails
run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry positive

# Run once in Scenario A after the complete positive role-check pass.
aws_capture aws ecs execute-command \
  --cluster "$ECS_CLUSTER" --task "$CANDIDATE_TASK_ARN" \
  --container cloudwatch-agent --interactive \
  --command "/bin/sh -c 'kill -TERM 1'" >/dev/null
run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry outage
OUTAGE_TASK_ARN="$CANDIDATE_TASK_ARN"
aws_capture aws ecs update-service \
  --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
  --task-definition "$CANDIDATE_TASK_DEFINITION" --desired-count 1 \
  --force-new-deployment \
  --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
  >/dev/null
aws_capture aws ecs wait services-stable \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
CANDIDATE_TASK_ARN=$(aws_capture aws ecs list-tasks \
  --cluster "$ECS_CLUSTER" --service-name "$ECS_SERVICE" \
  --desired-status RUNNING --query 'taskArns[0]' --output text)
test -n "$CANDIDATE_TASK_ARN" && test "$CANDIDATE_TASK_ARN" != None
test "$CANDIDATE_TASK_ARN" != "$OUTAGE_TASK_ARN"
run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry positive

aws_capture aws elbv2 modify-rule --rule-arn "$FIRST_DEPLOY_LISTENER_RULE_ARN" \
  --actions "$FIRST_DEPLOY_DISABLED_ACTIONS" >/dev/null
aws_capture aws ecs update-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
  --desired-count 0 \
  --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
  >/dev/null
aws_capture aws ecs wait services-stable --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
LOCAL_AUTH_OVERRIDES=$(jq -cn --arg name "$WEB_CONTAINER_NAME" \
  '{containerOverrides:[{name:$name,command:["python","-m","elspeth.web.aws_ecs_acceptance","verify-local-auth"]}]}')
LOCAL_AUTH_TASK=$(aws_capture aws ecs run-task \
  --cluster "$ECS_CLUSTER" --task-definition "$LOCAL_AUTH_VERIFIER_TASK_DEFINITION" \
  --launch-type FARGATE --network-configuration "$DOCTOR_NETWORK_CONFIGURATION" \
  --count 1 --overrides "$LOCAL_AUTH_OVERRIDES" --query 'tasks[0].taskArn' --output text)
aws_capture aws ecs wait tasks-stopped --cluster "$ECS_CLUSTER" --tasks "$LOCAL_AUTH_TASK" >/dev/null
aws_capture aws ecs describe-tasks --cluster "$ECS_CLUSTER" --tasks "$LOCAL_AUTH_TASK" \
  | jq -e '(.failures | length) == 0 and (.tasks | length) == 1 and all(.tasks[0].containers[] | select(.essential == true); .exitCode == 0)' >/dev/null
```

The `verify-bedrock-guardrails` receipt must contain `plugin_policy` with the
exact `target_llm` and the prompt-shield/content-safety entries in
`selected_controls`, including both opaque aliases and `required` modes, plus
`landscape_evidence: true`. That final flag proves the acceptance run's atomic
`run_web_plugin_policy` row was read back unchanged; a Guardrail API success
without this policy proof is NO-GO.

Both one-shot definitions set task-level `user: "1000:1000"`, explicitly
override the image entrypoint, use the candidate digest, and reuse the exact
approved EFS access point and database settings. The role-check transport
requires the Session Manager plugin, running `ExecuteCommandAgent`, and only
the module's single bounded receipt sentinel; interactive output is never
echoed or retained.

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
set -o pipefail
load_cleanup_state
bind_initial_evidence_export "$EVIDENCE_EXPORT_RECEIPT"
cleanup_failures=()

destroy_scenario() {
  local scenario_id="$1" directory="$2" vars="$3" binding="$4"
  local binding_file="$5" approval_file="$6" surface="$7"
  if (
    set -Eeuo pipefail
    work="$(mktemp -d -p /tmp "elspeth-${scenario_id}-destroy.XXXXXX")"
    chmod 700 "$work"
    trap 'rm -rf -- "$work"' EXIT
    plan="$work/destroy.tfplan"
    receipt="$work/destroy.receipt.json"
    terraform_capture -chdir="$directory" init -input=false -lock-timeout=5m >/dev/null
    verify_tf_binding "$scenario_id" "$directory" "$vars" "$binding" "$binding_file"
    state_before="$(terraform_capture -chdir="$directory" state list)"
    if test -z "$state_before"; then
      exit 0
    fi
    terraform_capture -chdir="$directory" plan -destroy -input=false -lock-timeout=5m \
      -var-file="$vars" -var="acceptance_run_id=$ACCEPTANCE_RUN_ID" \
      -var="scenario_id=$scenario_id" -var="candidate_image=$CANDIDATE_IMAGE" \
      -var="rollback_baseline_image=$ROLLBACK_BASELINE_IMAGE" -out="$plan" >/dev/null
    chmod 600 "$plan"
    terraform_capture -chdir="$directory" show -json "$plan" | \
      uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
        sanitize-evidence --kind terraform-destroy-plan >"$receipt"
    chmod 600 "$receipt"
    plan_sha="$(sha256sum "$plan" | awk '{print $1}')"
    receipt_hash="$(persist_sanitized_receipt "$scenario_id" terraform-destroy-plan "$plan_sha" "$receipt")"
    approval_hash="$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
      approval-verify --file "$CONTROL_MANIFEST" --scenario-id "$scenario_id" \
      --kind terraform-destroy-plan --plan-receipt-hash "$receipt_hash" \
      --approval-file "$approval_file")"
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance approval-require-current \
      --file "$CONTROL_MANIFEST" --scenario-id "$scenario_id" \
      --kind terraform-destroy-plan --plan-receipt-hash "$receipt_hash" \
      --approval-hash "$approval_hash"
    terraform_capture -chdir="$directory" apply -input=false -lock-timeout=5m "$plan" >/dev/null
    test -z "$(terraform_capture -chdir="$directory" state list)"
  ); then
    checkpoint_cleanup "$surface" confirmed
  else
    checkpoint_cleanup "$surface" failed || true
    return 1
  fi
}

delete_ecr_tag() {
  local label="$1" tag="$2" surface="ecr_$1" listing count result
  if (
    set -Eeuo pipefail
    listing="$(aws_capture aws ecr list-images --region "$AWS_REGION" \
      --repository-name "$ECR_REPOSITORY" --filter tagStatus=TAGGED --output json)"
    count="$(jq --arg tag "$tag" '[.imageIds[] | select(.imageTag == $tag)] | length' <<<"$listing")"
    test "$count" = 0 || test "$count" = 1
    if test "$count" = 1; then
      result="$(aws_capture aws ecr batch-delete-image --region "$AWS_REGION" \
        --repository-name "$ECR_REPOSITORY" --image-ids imageTag="$tag")"
      jq -e '(.failures | length) == 0 and (.imageIds | length) == 1' <<<"$result" >/dev/null
    fi
    listing="$(aws_capture aws ecr list-images --region "$AWS_REGION" \
      --repository-name "$ECR_REPOSITORY" --filter tagStatus=TAGGED --output json)"
    jq -e --arg tag "$tag" '[.imageIds[] | select(.imageTag == $tag)] | length == 0' \
      <<<"$listing" >/dev/null
  ); then
    checkpoint_cleanup "$surface" confirmed
  else
    checkpoint_cleanup "$surface" failed || true
    return 1
  fi
}

if ! destroy_scenario A "$SCENARIO_A_TF_DIR" "$SCENARIO_A_TF_VARS" \
  "$SCENARIO_A_TF_BINDING_SHA" "$SCENARIO_A_TF_BINDING_FILE" \
  "$SCENARIO_A_DESTROY_APPROVAL_FILE" terraform_scenario_a; then
  cleanup_failures+=(terraform_scenario_a)
fi
if ! destroy_scenario B "$SCENARIO_B_TF_DIR" "$SCENARIO_B_TF_VARS" \
  "$SCENARIO_B_TF_BINDING_SHA" "$SCENARIO_B_TF_BINDING_FILE" \
  "$SCENARIO_B_DESTROY_APPROVAL_FILE" terraform_scenario_b; then
  cleanup_failures+=(terraform_scenario_b)
fi

if SANITIZED_ORPHAN_RECEIPT="$(mktemp -p /tmp elspeth-orphan.XXXXXX)" && \
  chmod 600 "$SANITIZED_ORPHAN_RECEIPT" && \
  run_orphan_sweep >"$SANITIZED_ORPHAN_RECEIPT"; then
  checkpoint_cleanup orphan_sweep confirmed
else
  checkpoint_cleanup orphan_sweep failed || true
  cleanup_failures+=(orphan_sweep)
fi

if ! delete_ecr_tag baseline "$ROLLBACK_BASELINE_TAG"; then
  cleanup_failures+=(ecr_baseline)
fi
if ! delete_ecr_tag candidate "$CANDIDATE_TAG"; then
  cleanup_failures+=(ecr_candidate)
fi

if test "${#cleanup_failures[@]}" -ne 0; then
  printf 'cleanup failures: %s\n' "${cleanup_failures[*]}" >&2
  exit 1
fi
```

Identity and shared-resource cleanup are owner actions rather than shell
substitutes. After validating their signed completion/restoration receipts,
call `checkpoint_cleanup identity_cleanup confirmed` and
`checkpoint_cleanup shared_resource_cleanup confirmed`. Failed or timed-out
owner actions are checkpointed `failed`; they do not prevent the independent
Terraform, orphan-sweep, and ECR attempts above.

The evidence owner binds an initial export before deletion and a distinct
final export after the cleanup receipts exist. Plan 12 then prepares final
cleanup evidence, removes local images and temporary evidence through
`remove_local_acceptance_images` and `remove_local_acceptance_evidence`,
checkpoints those operations, and commits the finalizer. No checkpoint may be
marked `confirmed` until its real operation and absence/recovery check pass.

```bash
bind_final_evidence_export "$FINAL_EVIDENCE_EXPORT_RECEIPT"
checkpoint_cleanup evidence_export confirmed
finalize_cleanup_evidence prepare
checkpoint_cleanup final_evidence_prepare confirmed
remove_local_acceptance_images
checkpoint_cleanup local_images confirmed
remove_local_acceptance_evidence
checkpoint_cleanup local_evidence confirmed
uv run --frozen python -c 'from datetime import UTC, datetime; import os; assert datetime.now(UTC) <= datetime.fromisoformat(os.environ["ACCEPTANCE_TEARDOWN_DEADLINE_UTC"].replace("Z", "+00:00"))'
checkpoint_cleanup teardown_deadline confirmed
finalize_cleanup_evidence commit
uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest validate \
  --file "$CONTROL_MANIFEST" --cleanup-only --require-cleanup-cleared
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
