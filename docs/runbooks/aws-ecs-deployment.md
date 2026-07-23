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
- AWS CLI v2, `jq`, `curl`, Docker, the Session Manager plugin, Node 24/npm 11,
  and Playwright Chromium are installed from reviewed locks before mutation.
- Every web, doctor, payload-verifier, and local-auth-verifier task definition
  uses Linux Fargate 1.4.0 or `LATEST`, declares `runtimePlatform`, and uses
  the approved EFS access point and immutable image digest.

### Fresh-account HTTPS

Acceptance does not assume that the disposable AWS account owns a DNS zone or
already has an ACM certificate. Each scenario Terraform root owns a different
short-lived self-signed certificate whose only DNS SAN is that scenario's real
ALB DNS name, imports it into ACM, and attaches it to the scenario HTTPS
listener. The private key exists only in that scenario's encrypted, locked
remote state. Both the ACM certificate and ALB are tagged with the full
`ACCEPTANCE_RUN_ID` and scenario ID and are destroyed with the scenario.

The scenario stack uses this shape; the ALB itself can exist before the HTTPS
listener, so the generated SAN does not introduce a dependency cycle:

```hcl
resource "tls_private_key" "alb" {
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "tls_self_signed_cert" "alb" {
  private_key_pem       = tls_private_key.alb.private_key_pem
  validity_period_hours = 24
  early_renewal_hours   = 1
  dns_names             = [aws_lb.web.dns_name]
  allowed_uses          = ["key_encipherment", "digital_signature", "server_auth"]

  subject {
    organization = "ELSPETH disposable acceptance"
  }
}

resource "aws_acm_certificate" "alb" {
  private_key      = tls_private_key.alb.private_key_pem
  certificate_body = tls_self_signed_cert.alb.cert_pem
  tags             = local.acceptance_tags
}

output "acceptance_tls_ca_pem" {
  value     = tls_self_signed_cert.alb.cert_pem
  sensitive = true
}
```

After each scenario apply and resolved inventory bind, write that public
certificate to the protected live-work directory and derive its exact
Chromium SPKI pin. Repeat this after every `load_scenario`; the active CA and
pin must belong to the same scenario as `ALB_BASE_URL`:

```bash
ACCEPTANCE_TLS_DIR="/tmp/elspeth-acceptance-tls-${ACCEPTANCE_RUN_ID}"
if test -e "$ACCEPTANCE_TLS_DIR"; then
  test ! -L "$ACCEPTANCE_TLS_DIR" && test -d "$ACCEPTANCE_TLS_DIR"
  test "$(stat -c %u "$ACCEPTANCE_TLS_DIR")" = "$(id -u)"
  test "$(stat -c %a "$ACCEPTANCE_TLS_DIR")" = 700
else
  mkdir -m 700 -- "$ACCEPTANCE_TLS_DIR"
fi

prepare_scenario_tls() {
  local scenario_id="$1" terraform_dir="$2" alb_base_url="$3" hostname ca_file spki
  case "$scenario_id" in A|B) ;; *) return 2 ;; esac
  ca_file="$ACCEPTANCE_TLS_DIR/scenario-${scenario_id,,}-ca.pem"
  terraform_capture -chdir="$terraform_dir" output -raw acceptance_tls_ca_pem >"$ca_file"
  chmod 600 "$ca_file"
  hostname=$(uv run --frozen python -c 'import sys, urllib.parse; print(urllib.parse.urlsplit(sys.argv[1]).hostname or "")' "$alb_base_url")
  test -n "$hostname"
  openssl x509 -in "$ca_file" -noout -checkend 3600
  openssl x509 -in "$ca_file" -noout -checkhost "$hostname"
  spki=$(openssl x509 -in "$ca_file" -pubkey -noout \
    | openssl pkey -pubin -outform DER \
    | openssl dgst -sha256 -binary \
    | openssl base64 -A)
  [[ "$spki" =~ ^[A-Za-z0-9+/]{43}=$ ]]
  printf -v "SCENARIO_${scenario_id}_TLS_CA_BUNDLE" '%s' "$ca_file"
  printf -v "SCENARIO_${scenario_id}_TLS_SPKI_SHA256" '%s' "$spki"
}

activate_scenario_tls() {
  local scenario_id="$1" ca_name="SCENARIO_${1}_TLS_CA_BUNDLE" spki_name="SCENARIO_${1}_TLS_SPKI_SHA256"
  ACCEPTANCE_TLS_CA_BUNDLE=${!ca_name:?prepare scenario TLS first}
  ACCEPTANCE_TLS_SPKI_SHA256=${!spki_name:?prepare scenario TLS first}
  export ACCEPTANCE_TLS_CA_BUNDLE ACCEPTANCE_TLS_SPKI_SHA256
}
```

Host-side HTTP clients trust only the active scenario certificate. Use
`--cacert "$ACCEPTANCE_TLS_CA_BUNDLE"` for curl,
`SSL_CERT_FILE="$ACCEPTANCE_TLS_CA_BUNDLE"` for Python/httpx, and both
`NODE_EXTRA_CA_CERTS="$ACCEPTANCE_TLS_CA_BUNDLE"` and the exact Chromium SPKI
pin for the OIDC browser. Never use curl `-k`, `verify=False`, Playwright
`ignoreHTTPSErrors`, or a general Chromium certificate-error bypass. The CA
files and pins are live-only material and are removed with the protected local
acceptance evidence after final export.

### Protected command capture

Start every operator shell with strict mode and protected capture. Every AWS
CLI call in rollout, acceptance, diagnosis, and cleanup goes through
`aws_capture`, or through `aws_waiter_capture` for an ECS waiter; raw logs are
never printed or persisted. A non-zero AWS call emits only a static class.
Plan 10's packaged acceptance module supplies the closed `sanitize-evidence`
projectors used below.

```bash
set -Eeuo pipefail
umask 077
export AWS_PAGER=""
export AWS_CLI_CONNECT_TIMEOUT=5
export AWS_CLI_READ_TIMEOUT=30
export ELSPETH_COMMAND_OUTPUT_LIMIT_BYTES=2097152
export ELSPETH_AWS_CALL_CEILING_SECONDS=60
export ELSPETH_AWS_WAITER_CEILING_SECONDS=900
export ELSPETH_AWS_EXEC_CEILING_SECONDS=420
export ELSPETH_ORPHAN_SWEEP_CEILING_SECONDS=900
# Aurora create/delete can legitimately consume the provider's two-hour
# timeout. This outer ceiling adds ten minutes for provider bookkeeping.
export ELSPETH_TERRAFORM_CALL_CEILING_SECONDS=7800
export ELSPETH_CLEANUP_RESERVE_SECONDS=5400

protected_timeout_seconds() {
  local kind="$1" ceiling deadline now_epoch deadline_epoch remaining reserve refresh_file
  case "$kind" in
    terraform) ceiling="${ELSPETH_TERRAFORM_CALL_CEILING_SECONDS:?set Terraform call ceiling}" ;;
    aws-waiter) ceiling="${ELSPETH_AWS_WAITER_CEILING_SECONDS:?set AWS waiter ceiling}" ;;
    aws-exec) ceiling="${ELSPETH_AWS_EXEC_CEILING_SECONDS:?set ECS Exec ceiling}" ;;
    orphan-sweep) ceiling="${ELSPETH_ORPHAN_SWEEP_CEILING_SECONDS:?set orphan sweep ceiling}" ;;
    aws) ceiling="${ELSPETH_AWS_CALL_CEILING_SECONDS:?set AWS call ceiling}" ;;
    *) printf '%s\n' 'command_kind_invalid' >&2; return 1 ;;
  esac
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

aws_capture_with_kind() (
  local timeout_kind="$1" stdout_file stderr_file status stdout_size stderr_size timeout_seconds
  shift
  test "${1:-}" = aws || {
    printf '%s\n' 'aws_command_invalid' >&2
    exit 1
  }
  stdout_file=$(mktemp)
  stderr_file=$(mktemp)
  trap 'rm -f -- "$stdout_file" "$stderr_file"' EXIT HUP INT TERM
  chmod 600 "$stdout_file" "$stderr_file"
  timeout_seconds=$(protected_timeout_seconds "$timeout_kind") || exit 1
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

aws_capture() {
  aws_capture_with_kind aws "$@"
}

aws_waiter_capture() {
  aws_capture_with_kind aws-waiter "$@"
}

aws_exec_capture() {
  aws_capture_with_kind aws-exec "$@"
}

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
  if timeout --signal=TERM --kill-after=5s "${timeout_seconds}s" terraform "$@" \
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
  binding_hash=$(jq -cjS . "$binding_file" 2>/dev/null | sha256sum | awk '{print $1}') || {
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
systems/access points, secret IDs, exact task/execution IAM role names, log
groups, exact CloudWatch Logs resource-policy names, dashboard/alarm names, X-Ray
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

Every Terraform root derives its AWS resource namespace identically. The
input is the canonical lowercase UUID, a literal NUL byte, and the uppercase
scenario ID; take the first 20 lowercase hex characters of SHA-256 and prefix
them with `a-` or `b-`. The full UUID remains in tags, telemetry dimensions,
and the terminal S3 prefix segment; it is not placed in length-constrained AWS
resource names.

```hcl
locals {
  scenario_id_upper          = upper(var.scenario_id)
  scenario_resource_digest  = substr(sha256("${lower(var.acceptance_run_id)}\u0000${local.scenario_id_upper}"), 0, 20)
  scenario_resource_namespace = "${lower(local.scenario_id_upper)}-${local.scenario_resource_digest}"
}
```

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

request_signed_tf_approval() {
  local scenario_id="$1" kind="$2" receipt_hash="$3" receipt_file="$4" approval_file="$5"
  local handoff="${TERRAFORM_APPROVAL_HANDOFF:?set the reviewed operator approval handoff executable}"
  case "$kind" in terraform-plan|terraform-destroy-plan) ;; *) return 64 ;; esac
  test "${handoff#/}" != "$handoff"
  test ! -L "$handoff" && test -f "$handoff" && test -x "$handoff"
  test "$(stat -c %u "$handoff")" = "$(id -u)"
  # The handoff receives only the sanitized receipt and its hash. It blocks
  # until the operator has reviewed that receipt and written the signed,
  # mode-0600 approval document to approval_file.
  "$handoff" "$scenario_id" "$kind" "$receipt_hash" "$receipt_file" "$approval_file"
  test ! -L "$approval_file" && test -f "$approval_file"
  test "$(stat -c %u "$approval_file")" = "$(id -u)"
  test "$(stat -c %a "$approval_file")" = 600
}

load_scenario() {
  local scenario_id="$1" assignments tls_ca_name
  unset ACTIVE_SCENARIO_ID ACCEPTANCE_RUN_ID DEPLOYMENT_MODE TARGET_PLATFORM \
    AWS_REGION ECS_CLUSTER ECS_SERVICE WEB_CONTAINER_NAME TARGET_GROUP_ARN \
    ELSPETH_WEB__DATA_DIR ELSPETH_WEB__PAYLOAD_STORE_PATH \
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
    DOCTOR_LOG_GROUP DOCTOR_LOG_STREAM_PREFIX OPERATOR_METRICS_LOG_GROUP \
    ECS_DEPLOYMENT_EVENT_RULE \
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
  tls_ca_name="SCENARIO_${scenario_id}_TLS_CA_BUNDLE"
  if test -n "${ACCEPTANCE_TLS_DIR:-}" && test -n "${!tls_ca_name:-}"; then
    activate_scenario_tls "$scenario_id"
  fi
}

run_orphan_sweep() {
  local timeout_seconds
  timeout_seconds=$(protected_timeout_seconds orphan-sweep)
  timeout --signal=TERM --kill-after=5s "${timeout_seconds}s" \
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
  local assignments iac_parent bootstrap_state_entries
  assignments="$(mktemp -p /tmp elspeth-cleanup-state.XXXXXX)"
  chmod 600 "$assignments"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest load-cleanup \
    --file "$CONTROL_MANIFEST" --shell-assignments >"$assignments"
  . "$assignments"
  rm -f -- "$assignments"
  iac_parent=$(dirname "$SCENARIO_A_TF_DIR")
  test "$(dirname "$SCENARIO_B_TF_DIR")" = "$iac_parent"
  BOOTSTRAP_TF_DIR="$iac_parent/bootstrap"
  BOOTSTRAP_STATE="$BOOTSTRAP_TF_DIR/terraform.tfstate"
  BACKEND_STATE_BUCKET="elspeth-acc-${ACCEPTANCE_RUN_ID//-/}"
  ACCEPTANCE_TLS_DIR="/tmp/elspeth-acceptance-tls-${ACCEPTANCE_RUN_ID}"
  SANITIZED_ORPHAN_RECEIPT="/tmp/elspeth-orphan-${ACCEPTANCE_RUN_ID}.json"
  test ! -L "$BOOTSTRAP_TF_DIR" && test -d "$BOOTSTRAP_TF_DIR"
  if test -e "$BOOTSTRAP_STATE"; then
    test ! -L "$BOOTSTRAP_STATE" && test -f "$BOOTSTRAP_STATE"
    test "$(stat -c %u "$BOOTSTRAP_STATE")" = "$(id -u)"
    test "$(stat -c %a "$BOOTSTRAP_STATE")" = 600
    bootstrap_state_entries=$(terraform_capture -chdir="$BOOTSTRAP_TF_DIR" state list)
    if test -n "$bootstrap_state_entries"; then
      test "$(terraform_capture -chdir="$BOOTSTRAP_TF_DIR" output -raw backend_state_bucket)" = "$BACKEND_STATE_BUCKET"
    fi
  fi
  [[ "$BACKEND_STATE_BUCKET" =~ ^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$ ]]
  case "$SANITIZED_ORPHAN_RECEIPT" in /tmp/elspeth-orphan-*.json) ;; *) return 1 ;; esac
  test ! -L "$SANITIZED_ORPHAN_RECEIPT"
  export BOOTSTRAP_TF_DIR BOOTSTRAP_STATE BACKEND_STATE_BUCKET ACCEPTANCE_TLS_DIR SANITIZED_ORPHAN_RECEIPT
  export ACCEPTANCE_TEARDOWN_DEADLINE_UTC
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
  return 0
}

remove_local_acceptance_evidence() {
  if test -n "${ACCEPTANCE_STATE:-}"; then
    case "$ACCEPTANCE_STATE" in /tmp/*) rm -f -- "$ACCEPTANCE_STATE" ;; *) return 1 ;; esac
  fi
  if test -n "${OIDC_EVIDENCE_DIR:-}"; then
    case "$OIDC_EVIDENCE_DIR" in /tmp/*) rm -rf -- "$OIDC_EVIDENCE_DIR" ;; *) return 1 ;; esac
  fi
  if test -n "${ACCEPTANCE_TLS_DIR:-}"; then
    case "$ACCEPTANCE_TLS_DIR" in /tmp/*) rm -rf -- "$ACCEPTANCE_TLS_DIR" ;; *) return 1 ;; esac
  fi
  if test -n "${SANITIZED_ORPHAN_RECEIPT:-}"; then
    case "$SANITIZED_ORPHAN_RECEIPT" in /tmp/elspeth-orphan-*.json) rm -f -- "$SANITIZED_ORPHAN_RECEIPT" ;; *) return 1 ;; esac
    test ! -e "$SANITIZED_ORPHAN_RECEIPT"
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

### Fresh-account shared bootstrap

In a new disposable account the encrypted S3 backend and ECR repository do
not exist yet. Prepare one separate bootstrap Terraform root outside the
repository with protected local state. It owns exactly the run-scoped S3
backend state bucket and ECR repository; it does not own either scenario's
application resources. The bucket enables versioning, SSE-S3 or SSE-KMS,
public-access blocking, and native S3 lockfiles. The ECR repository enables
scan-on-push. Both resources are tagged with the full `ACCEPTANCE_RUN_ID` and
the cleanup owner. Their names are known before creation, so the two scenario
binding receipts and pre-apply inventories can bind distinct hashed state keys
before Plan 12 initializes the control manifest.

Plan 12 must initialize and validate the control manifest before this block.
The three Terraform roots share one protected parent directory: `bootstrap/`,
`scenario-a/`, and `scenario-b/`. The backend bucket name is derived only from
the manifest-bound account and run UUID, so cleanup can recover it even if the
bootstrap apply is interrupted before local state is written. Arm cleanup with
the planned ECR identity before the bootstrap apply, keep the bootstrap state
under that protected parent, and checkpoint the still-pending shared cleanup
immediately after the mutation:

```bash
SCENARIO_A_TF_DIR=$(jq -er '.values.SCENARIO_TF_DIR' "$SCENARIO_A_INVENTORY")
SCENARIO_A_TF_VARS=$(jq -er '.values.SCENARIO_TF_VARS' "$SCENARIO_A_INVENTORY")
SCENARIO_A_TF_BINDING_FILE=$(jq -er '.values.SCENARIO_TF_BINDING_FILE' "$SCENARIO_A_INVENTORY")
SCENARIO_B_TF_DIR=$(jq -er '.values.SCENARIO_TF_DIR' "$SCENARIO_B_INVENTORY")
SCENARIO_B_TF_VARS=$(jq -er '.values.SCENARIO_TF_VARS' "$SCENARIO_B_INVENTORY")
SCENARIO_B_TF_BINDING_FILE=$(jq -er '.values.SCENARIO_TF_BINDING_FILE' "$SCENARIO_B_INVENTORY")
IAC_PACKAGE_DIR=$(dirname "${SCENARIO_A_TF_DIR:?set Scenario A Terraform root}")
test "$(dirname "${SCENARIO_B_TF_DIR:?set Scenario B Terraform root}")" = "$IAC_PACKAGE_DIR"
BOOTSTRAP_TF_DIR="$IAC_PACKAGE_DIR/bootstrap"
BOOTSTRAP_STATE="$BOOTSTRAP_TF_DIR/terraform.tfstate"
BACKEND_STATE_BUCKET="elspeth-acc-${ACCEPTANCE_RUN_ID//-/}"
export ECR_REGISTRY="${ECR_REGISTRY:?set approved account registry host}"
export ECR_REPOSITORY="${ECR_REPOSITORY:?set run-scoped ECR repository name}"
export ROLLBACK_BASELINE_TAG="acceptance-${ACCEPTANCE_RUN_ID}-baseline-${ROLLBACK_BASELINE_SHA}"
export CANDIDATE_TAG="acceptance-${ACCEPTANCE_RUN_ID}-0.7.1-${CANDIDATE_SHA}"

test ! -L "$BOOTSTRAP_TF_DIR" && test -d "$BOOTSTRAP_TF_DIR"
test "$BOOTSTRAP_STATE" = "$BOOTSTRAP_TF_DIR/terraform.tfstate"
[[ "$BACKEND_STATE_BUCKET" =~ ^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$ ]]
test ! -e "$BOOTSTRAP_STATE" || { test ! -L "$BOOTSTRAP_STATE" && test "$(stat -c '%a' "$BOOTSTRAP_STATE")" = 600; }
uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest validate \
  --file "$CONTROL_MANIFEST" --acceptance-run-id "$ACCEPTANCE_RUN_ID" \
  --candidate-sha "$CANDIDATE_SHA"
test "$(aws_capture aws sts get-caller-identity --query Account --output text)" = "$AWS_ACCOUNT_ID"

# Complete the credential-free browser installation/discovery and require the
# operator-provided identity inputs before the first AWS mutation.
test -n "${OIDC_TEST_USERNAME:-}" && test -n "${OIDC_TEST_PASSWORD:-}"
[[ "$OIDC_TEST_USERNAME" =~ ^[A-Za-z0-9][A-Za-z0-9._@+-]{0,127}$ ]]
test "${#OIDC_TEST_PASSWORD}" -ge 12 && test "${#OIDC_TEST_PASSWORD}" -le 256
npm --prefix src/elspeth/web/frontend ci
npm --prefix src/elspeth/web/frontend exec -- playwright install chromium
OIDC_LIST_OUTPUT=$(npm --prefix src/elspeth/web/frontend run test:e2e:oidc -- --list)
grep -Fq '[chromium] aws-ecs-oidc.staging.spec.ts' <<<"$OIDC_LIST_OUTPUT"
test "$(grep -Fc '[chromium] aws-ecs-oidc.staging.spec.ts' <<<"$OIDC_LIST_OUTPUT")" = 1
unset OIDC_LIST_OUTPUT
arm_external_cleanup

terraform_capture -chdir="$BOOTSTRAP_TF_DIR" init -input=false >/dev/null
terraform_capture -chdir="$BOOTSTRAP_TF_DIR" validate >/dev/null
BOOTSTRAP_PLAN="$BOOTSTRAP_TF_DIR/bootstrap.tfplan"
BOOTSTRAP_PLAN_RECEIPT=$(mktemp -p /tmp elspeth-bootstrap-plan.XXXXXX)
chmod 600 "$BOOTSTRAP_PLAN_RECEIPT"
terraform_capture -chdir="$BOOTSTRAP_TF_DIR" plan -input=false \
  -var="acceptance_run_id=$ACCEPTANCE_RUN_ID" \
  -var="aws_account_id=$AWS_ACCOUNT_ID" \
  -var="aws_region=$AWS_REGION" \
  -var="backend_state_bucket=$BACKEND_STATE_BUCKET" \
  -var="ecr_repository=$ECR_REPOSITORY" \
  -out="$BOOTSTRAP_PLAN" >/dev/null
chmod 600 "$BOOTSTRAP_PLAN"
BOOTSTRAP_PLAN_SHA=$(sha256sum "$BOOTSTRAP_PLAN" | awk '{print $1}')
terraform_capture -chdir="$BOOTSTRAP_TF_DIR" show -json "$BOOTSTRAP_PLAN" \
  | uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
      sanitize-evidence --kind terraform-plan >"$BOOTSTRAP_PLAN_RECEIPT"
BOOTSTRAP_PLAN_RECEIPT_HASH=$(persist_sanitized_receipt bootstrap terraform-plan \
  "$BOOTSTRAP_PLAN_SHA" "$BOOTSTRAP_PLAN_RECEIPT")
request_signed_tf_approval bootstrap terraform-plan "$BOOTSTRAP_PLAN_RECEIPT_HASH" \
  "$BOOTSTRAP_PLAN_RECEIPT" "${BOOTSTRAP_PLAN_APPROVAL_FILE:?set bootstrap apply approval file}"
BOOTSTRAP_PLAN_APPROVAL_HASH=$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
  approval-verify --file "$CONTROL_MANIFEST" --scenario-id bootstrap --kind terraform-plan \
  --plan-receipt-hash "$BOOTSTRAP_PLAN_RECEIPT_HASH" \
  --approval-file "$BOOTSTRAP_PLAN_APPROVAL_FILE")
uv run --frozen python -m elspeth.web.aws_ecs_acceptance approval-require-current \
  --file "$CONTROL_MANIFEST" --scenario-id bootstrap --kind terraform-plan \
  --plan-receipt-hash "$BOOTSTRAP_PLAN_RECEIPT_HASH" \
  --approval-hash "$BOOTSTRAP_PLAN_APPROVAL_HASH"
test "$(sha256sum "$BOOTSTRAP_PLAN" | awk '{print $1}')" = "$BOOTSTRAP_PLAN_SHA"
terraform_capture -chdir="$BOOTSTRAP_TF_DIR" apply -input=false \
  "$BOOTSTRAP_PLAN" >/dev/null
rm -f -- "$BOOTSTRAP_PLAN" "$BOOTSTRAP_PLAN_RECEIPT"
test ! -L "$BOOTSTRAP_STATE" && test "$(stat -c '%a' "$BOOTSTRAP_STATE")" = 600
test "$(terraform_capture -chdir="$BOOTSTRAP_TF_DIR" output -raw backend_state_bucket)" = "$BACKEND_STATE_BUCKET"
test "$(terraform_capture -chdir="$BOOTSTRAP_TF_DIR" output -raw ecr_repository)" = "$ECR_REPOSITORY"
checkpoint_cleanup shared_resource_cleanup pending

aws_capture aws s3api head-bucket --bucket "$BACKEND_STATE_BUCKET" >/dev/null
aws_capture aws ecr describe-repositories --region "$AWS_REGION" \
  --repository-names "$ECR_REPOSITORY" >/dev/null
```

The bootstrap provider sets `region = var.aws_region` and
`allowed_account_ids = [var.aws_account_id]`; the same closed account binding
is required in both scenario providers.

Each scenario root is deliberately disposable: Aurora sets
`deletion_protection = false` and `skip_final_snapshot = true`, every owned
Secrets Manager secret sets `recovery_window_in_days = 0`, and the scenario
S3 bucket sets `force_destroy = true`. The resolved inventory emits the exact
task/execution IAM role names and CloudWatch Logs resource-policy name so the
terminal sweep can prove their absence; Resource Groups Tagging API is not a
substitute for either surface.

Initialize and live-verify both distinct S3 backends now, before any image is
pushed. This does not apply Scenario B or create application resources:

```bash
initialize_scenario_backend() {
  local scenario_id="$1" directory="$2" vars="$3" binding="$4" binding_file="$5" workspace
  workspace=$(jq -er '.workspace | select(test("^[A-Za-z0-9_-]{1,90}$"))' "$binding_file")
  terraform_capture -chdir="$directory" init -input=false -reconfigure -lock-timeout=5m >/dev/null
  terraform_capture -chdir="$directory" workspace select "$workspace" >/dev/null \
    || terraform_capture -chdir="$directory" workspace new "$workspace" >/dev/null
  verify_tf_binding "$scenario_id" "$directory" "$vars" "$binding" "$binding_file"
}

initialize_scenario_backend A "$SCENARIO_A_TF_DIR" "$SCENARIO_A_TF_VARS" \
  "$SCENARIO_A_TF_BINDING_SHA" "$SCENARIO_A_TF_BINDING_FILE"
initialize_scenario_backend B "$SCENARIO_B_TF_DIR" "$SCENARIO_B_TF_VARS" \
  "$SCENARIO_B_TF_BINDING_SHA" "$SCENARIO_B_TF_BINDING_FILE"
```

Only after both checks pass may an image be pushed. Cleanup destroys Scenario
A and B first, removes the two image tags, then destroys this bootstrap root
before the terminal orphan sweep. Its run-scoped bucket may use
`force_destroy` because it contains only these two disposable state objects;
no shared or pre-existing bucket is permitted.

### Temporary image publication

The acceptance run uses unique temporary tags for both the Plan 10 rollback
baseline and the frozen candidate. The control manifest is armed before the
first push so interruption always routes to cleanup:

The image still contains the exact rollback source tree consumed by the
runtime. Only `Dockerfile` and `.dockerignore` come from the frozen candidate,
because those two Plan 10 packaging controls make the qualified pre-Plan-10
source buildable without changing its application or lock content.

```bash
set -Eeuo pipefail
test -n "${ROLLBACK_BASELINE_SHA:?restore the Plan 10 baseline SHA}"
test "$ROLLBACK_BASELINE_SHA" != "$CANDIDATE_SHA"
git merge-base --is-ancestor "$ROLLBACK_BASELINE_SHA" "$CANDIDATE_SHA"
test "${TARGET_PLATFORM:?}" = linux/amd64 || test "$TARGET_PLATFORM" = linux/arm64

# The rollback SHA is the pre-Plan-10 source identity. Its historical Docker
# context predates Plan 10's production-only test-source exclusions and is not
# itself buildable. Package that exact runtime source and lock tree with the
# frozen candidate's reviewed Dockerfile and .dockerignore. The image still
# contains the exact rollback source tree consumed by the runtime; only its
# packaging control files come from the candidate that earned this runbook.
(
  ROLLBACK_CONTEXT="$(mktemp -d -p /tmp elspeth-rollback-context.XXXXXX)"
  chmod 700 "$ROLLBACK_CONTEXT"
  trap 'rm -rf -- "$ROLLBACK_CONTEXT"' EXIT HUP INT TERM
  git archive "$ROLLBACK_BASELINE_SHA" | tar -x -C "$ROLLBACK_CONTEXT"
  git show "$CANDIDATE_SHA:Dockerfile" >"$ROLLBACK_CONTEXT/Dockerfile"
  git show "$CANDIDATE_SHA:.dockerignore" >"$ROLLBACK_CONTEXT/.dockerignore"
  chmod 600 "$ROLLBACK_CONTEXT/Dockerfile" "$ROLLBACK_CONTEXT/.dockerignore"
  docker buildx build \
    --platform "$TARGET_PLATFORM" --load \
    --build-arg INSTALL_EXTRAS="webui llm aws postgres" \
    --label "org.opencontainers.image.revision=$ROLLBACK_BASELINE_SHA" \
    -t elspeth:ecs-rollback-baseline "$ROLLBACK_CONTEXT"
)
test "$(docker image inspect elspeth:ecs-rollback-baseline --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')" = "$ROLLBACK_BASELINE_SHA"
test "$(docker image inspect elspeth:ecs-rollback-baseline --format '{{.Os}}/{{.Architecture}}')" = "$TARGET_PLATFORM"

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
projection and hash, not the plan itself. `TERRAFORM_APPROVAL_HANDOFF` is the
reviewed owner-side executable defined above: it receives the newly created
sanitized receipt and blocks until the operator writes the matching signed
approval. An approval file cannot validly be prepared before its receipt hash
exists.

```bash
provision_scenario_b_test_identity() (
  set -Eeuo pipefail
  local pool_id="$1" users count work create_input password_input verified subject
  test -n "$pool_id"
  work=$(mktemp -d -p /tmp elspeth-oidc-identity.XXXXXX)
  chmod 700 "$work"
  trap 'rm -rf -- "$work"' EXIT
  create_input="$work/create.json"
  password_input="$work/password.json"

  users=$(aws_capture aws cognito-idp list-users --region "$AWS_REGION" \
    --user-pool-id "$pool_id" --filter "username = \"$OIDC_TEST_USERNAME\"" \
    --attributes-to-get sub --limit 60 --output json)
  count=$(jq -er --arg username "$OIDC_TEST_USERNAME" \
    '[.Users[] | select(.Username == $username)] | length' <<<"$users")
  test "$count" = 0 || test "$count" = 1
  if test "$count" = 0; then
    jq -cn --arg pool "$pool_id" --arg username "$OIDC_TEST_USERNAME" \
      --arg password "$OIDC_TEST_PASSWORD" \
      '{UserPoolId:$pool,Username:$username,TemporaryPassword:$password,MessageAction:"SUPPRESS"}' \
      >"$create_input"
    chmod 600 "$create_input"
    aws_capture aws cognito-idp admin-create-user --region "$AWS_REGION" \
      --cli-input-json "file://$create_input" --query 'User.UserStatus' --output text \
      | grep -Fxq FORCE_CHANGE_PASSWORD
  fi

  jq -cn --arg pool "$pool_id" --arg username "$OIDC_TEST_USERNAME" \
    --arg password "$OIDC_TEST_PASSWORD" \
    '{UserPoolId:$pool,Username:$username,Password:$password,Permanent:true}' \
    >"$password_input"
  chmod 600 "$password_input"
  aws_capture aws cognito-idp admin-set-user-password --region "$AWS_REGION" \
    --cli-input-json "file://$password_input" >/dev/null
  verified=$(aws_capture aws cognito-idp admin-get-user --region "$AWS_REGION" \
    --user-pool-id "$pool_id" --username "$OIDC_TEST_USERNAME" \
    --query '{username:Username,enabled:Enabled,status:UserStatus,sub:UserAttributes[?Name==`sub`]|[0].Value}' \
    --output json)
  subject=$(jq -er --arg username "$OIDC_TEST_USERNAME" '
    select(keys == ["enabled","status","sub","username"])
    | select(.username == $username and .enabled == true and .status == "CONFIRMED")
    | .sub | select(type == "string" and length > 0 and length <= 128)
  ' <<<"$verified")
  printf '%s\n' "$subject"
)

render_resolved_inventory() (
  local scenario_id="$1" directory="$2" output_path="$3" rendered rebound pool_id subject
  rendered=$(mktemp -p /tmp "elspeth-${scenario_id}-inventory.XXXXXX")
  rebound=""
  trap 'test -z "$rendered" || rm -f -- "$rendered"; test -z "$rebound" || rm -f -- "$rebound"' EXIT
  chmod 600 "$rendered"
  terraform_capture -chdir="$directory" output -json resolved_inventory >"$rendered"
  if test "$scenario_id" = B; then
    test "$(jq -er '.orphan_sweep.cognito_subject_sub' "$rendered")" = ""
    pool_id=$(jq -er '.values.COGNITO_USER_POOL_ID | select(type == "string" and length > 0)' "$rendered")
    subject=$(provision_scenario_b_test_identity "$pool_id")
    rebound=$(mktemp -p /tmp elspeth-B-inventory-bound.XXXXXX)
    chmod 600 "$rebound"
    jq --arg subject "$subject" '.orphan_sweep.cognito_subject_sub = $subject' \
      "$rendered" >"$rebound"
    mv -fT -- "$rebound" "$rendered"
    rebound=""
  fi
  jq -e --arg run "$ACCEPTANCE_RUN_ID" --arg candidate "$CANDIDATE_SHA" \
    --arg account "$AWS_ACCOUNT_ID" --arg region "$AWS_REGION" --arg scenario "$scenario_id" '
      type == "object"
      and .schema == "elspeth.aws-ecs-scenario-inventory.v5"
      and .phase == "resolved"
      and .acceptance_run_id == $run
      and .candidate_sha == $candidate
      and .aws_account_id == $account
      and .aws_region == $region
      and .scenario_id == $scenario
    ' "$rendered" >/dev/null
  if test -e "$output_path"; then
    test ! -L "$output_path" && test -f "$output_path"
    test "$(stat -c %u "$output_path")" = "$(id -u)"
    test "$(stat -c %a "$output_path")" = 600
    cmp -s "$rendered" "$output_path"
  else
    mv -- "$rendered" "$output_path"
    rendered=""
    chmod 600 "$output_path"
  fi
)

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
    -var="scenario_id=$scenario_id" -var="aws_account_id=$AWS_ACCOUNT_ID" \
    -var="candidate_image=$CANDIDATE_IMAGE" \
    -var="rollback_baseline_image=$ROLLBACK_BASELINE_IMAGE" -out="$plan" >/dev/null
  chmod 600 "$plan"
  terraform_capture -chdir="$directory" show -json "$plan" | \
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
      sanitize-evidence --kind terraform-plan >"$receipt"
  chmod 600 "$receipt"
  plan_sha="$(sha256sum "$plan" | awk '{print $1}')"
  receipt_hash="$(persist_sanitized_receipt "$scenario_id" terraform-plan "$plan_sha" "$receipt")"
  request_signed_tf_approval "$scenario_id" terraform-plan "$receipt_hash" \
    "$receipt" "$TERRAFORM_PLAN_APPROVAL_FILE"
  approval_hash="$(require_signed_tf_plan_approval "$scenario_id" "$receipt_hash")"
  checkpoint_terraform_plan "$scenario_id" "$plan_sha" "$receipt_hash" "$approval_hash"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance approval-require-current \
    --file "$CONTROL_MANIFEST" --scenario-id "$scenario_id" --kind terraform-plan \
    --plan-receipt-hash "$receipt_hash" --approval-hash "$approval_hash"
  test "$(sha256sum "$plan" | awk '{print $1}')" = "$plan_sha"
  terraform_capture -chdir="$directory" apply -input=false -lock-timeout=5m "$plan" >/dev/null
  checkpoint_terraform_apply "$scenario_id" "$plan_sha" "$receipt_hash" "$approval_hash"

  plan="$work/noop.tfplan"
  receipt="$work/noop.receipt.json"
  terraform_capture -chdir="$directory" plan -input=false -lock-timeout=5m \
    -detailed-exitcode -var-file="$vars" \
    -var="acceptance_run_id=$ACCEPTANCE_RUN_ID" -var="scenario_id=$scenario_id" \
    -var="aws_account_id=$AWS_ACCOUNT_ID" \
    -var="candidate_image=$CANDIDATE_IMAGE" \
    -var="rollback_baseline_image=$ROLLBACK_BASELINE_IMAGE" -out="$plan" >/dev/null
  chmod 600 "$plan"
  terraform_capture -chdir="$directory" show -json "$plan" | \
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
      sanitize-evidence --kind terraform-plan >"$receipt"
  chmod 600 "$receipt"
  noop_sha="$(sha256sum "$plan" | awk '{print $1}')"
  receipt_hash="$(persist_sanitized_receipt "$scenario_id" terraform-noop "$noop_sha" "$receipt")"
  render_resolved_inventory "$scenario_id" "$directory" "$resolved_inventory"
  checkpoint_terraform_noop_and_bind "$scenario_id" "$noop_sha" \
    "$receipt_hash" "$resolved_inventory"
  rm -rf -- "$work"
}

activate_scenario_stack() {
  local scenario_id="$1" directory vars binding binding_file inventory
  case "$scenario_id" in
    A)
      directory="$SCENARIO_A_TF_DIR"
      vars="$SCENARIO_A_TF_VARS"
      binding="$SCENARIO_A_TF_BINDING_SHA"
      binding_file="$SCENARIO_A_TF_BINDING_FILE"
      inventory="${SCENARIO_A_INVENTORY%.json}.resolved.json"
      ;;
    B)
      directory="$SCENARIO_B_TF_DIR"
      vars="$SCENARIO_B_TF_VARS"
      binding="$SCENARIO_B_TF_BINDING_SHA"
      binding_file="$SCENARIO_B_TF_BINDING_FILE"
      inventory="${SCENARIO_B_INVENTORY%.json}.resolved.json"
      ;;
    *) return 64 ;;
  esac
  plan_and_apply_scenario "$scenario_id" "$directory" "$vars" \
    "$binding" "$binding_file" "$inventory"
  load_scenario "$scenario_id"
  prepare_scenario_tls "$scenario_id" "$directory" "$ALB_BASE_URL"
  activate_scenario_tls "$scenario_id"
}

# First pass only. Do not apply B until A has completed rollback section 7.
activate_scenario_stack A
```

After Scenario A completes section 7, return here once and run
`activate_scenario_stack B`; then execute only the Scenario B baseline and
repeat sections 1 through 7. The B backend was initialized in the pre-push
bootstrap gate, but no B application plan is applied while A acceptance is in
progress. Cleanup remains deferred until both isolated scenarios have
completed or an interruption/failure requires Task 6.

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
prepare_scenario_b_oidc() {
  local cognito_client_json identity_json inventory_path expected_sub OIDC_REDIRECT_URI
  test "$ACTIVE_SCENARIO_ID" = B
  cognito_client_json=$(aws_capture aws cognito-idp describe-user-pool-client \
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
  ' <<<"$cognito_client_json" >/dev/null
  inventory_path=$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
    control-manifest get --file "$CONTROL_MANIFEST" --field scenarios.B.inventory_path)
  expected_sub=$(jq -er '.orphan_sweep.cognito_subject_sub \
    | select(type == "string" and length > 0)' "$inventory_path")
  identity_json=$(aws_capture aws cognito-idp admin-get-user --region "$AWS_REGION" \
    --user-pool-id "$COGNITO_USER_POOL_ID" --username "$OIDC_TEST_USERNAME" \
    --query '{username:Username,enabled:Enabled,status:UserStatus,sub:UserAttributes[?Name==`sub`]|[0].Value}' \
    --output json)
  jq -e --arg username "$OIDC_TEST_USERNAME" --arg subject "$expected_sub" '
    keys == ["enabled","status","sub","username"]
    and .username == $username and .sub == $subject
    and .enabled == true and .status == "CONFIRMED"
  ' <<<"$identity_json" >/dev/null
  OIDC_EVIDENCE_DIR="/tmp/elspeth-oidc-${ACCEPTANCE_RUN_ID}"
  if test -e "$OIDC_EVIDENCE_DIR"; then
    test ! -L "$OIDC_EVIDENCE_DIR" && test -d "$OIDC_EVIDENCE_DIR"
    test "$(stat -c %u "$OIDC_EVIDENCE_DIR")" = "$(id -u)"
    test "$(stat -c %a "$OIDC_EVIDENCE_DIR")" = 700
  else
    mkdir -m 700 -- "$OIDC_EVIDENCE_DIR"
  fi
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
    --file "$CONTROL_MANIFEST" --oidc-evidence-dir "$OIDC_EVIDENCE_DIR"
}
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

Chromium installation and exact test discovery already completed in the
fresh-account preflight before `arm_external_cleanup`; never defer them until
after bootstrap or Scenario A mutation.

The live runner accepts credentials only through the existing environment,
uses no storage state, screenshot, trace, video, or secondary reporter, and
writes one owner-only evidence document only after the created session is
deleted successfully:

```bash
run_oidc_evidence() {
  local phase="$1"
  STAGING_BASE_URL="$ALB_BASE_URL" \
  NODE_EXTRA_CA_CERTS="$ACCEPTANCE_TLS_CA_BUNDLE" \
  OIDC_TLS_SPKI_SHA256="$ACCEPTANCE_TLS_SPKI_SHA256" \
  OIDC_TEST_USERNAME="$OIDC_TEST_USERNAME" \
  OIDC_TEST_PASSWORD="$OIDC_TEST_PASSWORD" \
  OIDC_EXPECTED_ISSUER="$OIDC_EXPECTED_ISSUER" \
  OIDC_EXPECTED_AUDIENCE="$OIDC_EXPECTED_AUDIENCE" \
  OIDC_EXPECTED_AUTHORIZATION_ORIGIN="$OIDC_EXPECTED_AUTHORIZATION_ORIGIN" \
  OIDC_EXPECTED_AUDIENCE_CLAIM="$OIDC_EXPECTED_AUDIENCE_CLAIM" \
  OIDC_EVIDENCE_PHASE="$phase" \
  OIDC_EVIDENCE_FILE="$OIDC_EVIDENCE_DIR/$phase.json" \
  OIDC_BEARER_HANDOFF_FILE="${OIDC_BEARER_HANDOFF_FILE:-}" \
    npm --prefix src/elspeth/web/frontend run test:e2e:oidc
  test "$(stat -c '%a' "$OIDC_EVIDENCE_DIR/$phase.json")" = 600
}

capture_oidc_lifecycle_handoff() {
  local handoff_dir handoff_file
  test "$ACTIVE_SCENARIO_ID" = B
  handoff_dir=$(mktemp -d -p /tmp elspeth-oidc-bearer.XXXXXX)
  chmod 700 "$handoff_dir"
  handoff_file="$handoff_dir/access-token"
  OIDC_LIFECYCLE_STATE="$OIDC_EVIDENCE_DIR/candidate-lifecycle.state"
  test ! -e "$OIDC_LIFECYCLE_STATE" && test ! -L "$OIDC_LIFECYCLE_STATE"
  (
    set -Eeuo pipefail
    trap 'unset OIDC_BEARER_TOKEN; rm -rf -- "$handoff_dir"' EXIT HUP INT TERM
    OIDC_BEARER_HANDOFF_FILE="$handoff_file" run_oidc_evidence candidate-initial
    test -f "$handoff_file" && test ! -L "$handoff_file"
    test "$(stat -c '%a' "$handoff_file")" = 600
    test "$(wc -c <"$handoff_file")" -le 16384
    OIDC_BEARER_TOKEN=$(<"$handoff_file")
    rm -f -- "$handoff_file"
    [[ "$OIDC_BEARER_TOKEN" =~ ^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$ ]]
    SSL_CERT_FILE="$ACCEPTANCE_TLS_CA_BUNDLE" \
    ELSPETH_ACCEPTANCE_BEARER_TOKEN="$OIDC_BEARER_TOKEN" \
      uv run --frozen python -m elspeth.web.aws_ecs_acceptance capture \
        --state-file "$OIDC_LIFECYCLE_STATE"
    unset OIDC_BEARER_TOKEN
  )
  OIDC_LANDSCAPE_RUN_ID=$(jq -er \
    '.landscape_run_id | select(test("^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$"))' \
    "$OIDC_LIFECYCLE_STATE")
}
```

Run exactly `previous-before-candidate`, `candidate-initial`, and
`candidate-after-rollback-refusal`. Each JSON document
contains only `phase`, `timestamp`, `issuer`, `authorization_origin`,
`audience_claim`, `audience`, `subject_sha256`, `auth_me_status: 200`,
`session_create_status: 201`, `session_read_status: 200`,
`session_delete_status: 204`, and `session_round_trip: true`. It never contains a token,
authorization code, verifier, credential, callback URL/query/fragment,
cookie, header, page HTML, or artifact path.

### Bound release/schema compatibility record

Before either storage provisioner or schema initializer runs, the database
operator supplies a protected mode-0600 JSON record and the release operator
countersigns it. Set `SCENARIO_A_COMPATIBILITY_RECORD_FILE` and
`SCENARIO_B_COMPATIBILITY_RECORD_FILE`. Scenario B has exactly this shape:

```json
{
  "schema": "elspeth.aws-ecs-compatibility-record.v2",
  "record_id": "change-record-id",
  "acceptance_run_id": "manifest-run-uuid",
  "scenario_id": "B",
  "candidate_sha": "40-lowercase-hex",
  "candidate_image_digest": "sha256:64-lowercase-hex",
  "candidate_task_definition": "exact-candidate-task-definition-arn",
  "candidate_doctor_task_definition": "exact-candidate-doctor-task-definition-arn",
  "candidate_package_version": "0.7.1",
  "previous_source_sha": "40-lowercase-hex",
  "previous_image_digest": "sha256:64-lowercase-hex",
  "previous_task_definition": "exact-previous-task-definition-arn",
  "rollback_doctor_task_definition": "exact-rollback-doctor-task-definition-arn",
  "previous_package_version": "0.7.0",
  "schema_facts": {
    "candidate": {"session_epoch": 35, "landscape_epoch": 29, "run_web_plugin_policy_present": true},
    "previous": {"session_epoch": 27, "landscape_epoch": 23, "run_web_plugin_policy_present": true},
    "structural_changes": "landscape_epoch_23_to_29_token_ownership_artifact_idempotency_sink_effect_ledger_coalesce_receipts_per_member_failsink_provenance_output_contract_hash_run_scoped_validation_errors_and_token_ancestry_batch_expansion_claim_and_sidecar_journal_outbox",
    "semantics_only_changes": "none",
    "archive_export_decision": "required_before_forward_migration",
    "destructive_reset_required": false
  },
  "forward_compatible": true,
  "backward_compatible": false,
  "rollback_permitted": false,
  "decision": "approved",
  "approver_identity": "database-operator",
  "countersigner_identity": "release-operator",
  "approved_at": "RFC3339-UTC",
  "countersigned_at": "RFC3339-UTC",
  "expires_at": "RFC3339-UTC"
}
```

Scenario A uses the same field set with `scenario_id: "A"`; empty strings for
`previous_source_sha`, `previous_image_digest`, `previous_task_definition`,
`rollback_doctor_task_definition`, and `previous_package_version`;
`schema_facts.previous: null`; `structural_changes: "initial_create"`;
`archive_export_decision: "not_applicable"`; and both
`backward_compatible` and `rollback_permitted` false.

The controller binds the record to the manifest, image digest, exact task
and doctor definitions, candidate and previous package/image identities,
session epoch 35, Landscape epoch 29 and `run_web_plugin_policy` presence,
change/reset facts, decision, two distinct approvals, and expiry. It
stores only a sanitized receipt and document hash. Reopen and revalidate the
raw record before init-capable doctor, ordinary doctor, candidate deploy, and
any later deployment action. The 0.7.0 image understands Landscape epoch 23,
not epoch 29. Pre-1.0 candidates do not migrate predecessor schemas: the old
deployment is stopped and uninstalled, required evidence is archived/exported,
and the databases are recreated before the candidate is installed. The previous
image cannot reopen the recreated current database, so Scenario B rollback is
forbidden after recreation. Unknown or unapproved compatibility is NO-GO;
expiry or identity drift is also NO-GO.

Before either schema initializer runs, create and prove the required EFS
children with the candidate image's explicit `1000:1000` one-shot definition.
The command creates only the configured payload directory and `data_dir/blobs`
under the already-mounted `data_dir`, then performs create/read/fsync/delete
probes in all three directories. It emits no paths:

```bash
resolve_bound_task_definition() {
  local variable="$1" container_name="$2" expected_user="${3:-}"
  local reference raw validated resolved expected_image expected_log_group expected_log_prefix expected_arch
  local -a user_args=() image_role_args=()
  if test -n "$expected_user"; then
    user_args=(--expected-user "$expected_user")
  fi
  reference=${!variable}
  raw=$(aws_capture aws ecs describe-task-definition \
    --task-definition "$reference" --output json)
  case "$variable" in
    PREVIOUS_TASK_DEFINITION|ROLLBACK_DOCTOR_TASK_DEFINITION)
      expected_image="$ROLLBACK_BASELINE_IMAGE"
      image_role_args=(--expected-image-role rollback-baseline) ;;
    *) expected_image="$CANDIDATE_IMAGE" ;;
  esac
  case "$variable" in
    DOCTOR_TASK_DEFINITION|ROLLBACK_DOCTOR_TASK_DEFINITION)
      expected_log_group="$DOCTOR_LOG_GROUP"; expected_log_prefix="$DOCTOR_LOG_STREAM_PREFIX" ;;
    *) expected_log_group="$WEB_LOG_GROUP"; expected_log_prefix="$WEB_LOG_STREAM_PREFIX" ;;
  esac
  case "$TARGET_PLATFORM" in
    linux/amd64) expected_arch=X86_64 ;;
    linux/arm64) expected_arch=ARM64 ;;
    *) return 64 ;;
  esac
  jq -e --arg container "$container_name" --arg image "$expected_image" \
    --arg arch "$expected_arch" --arg log_group "$expected_log_group" \
    --arg log_prefix "$expected_log_prefix" '
      .taskDefinition.status == "ACTIVE"
      and .taskDefinition.networkMode == "awsvpc"
      and (.taskDefinition.requiresCompatibilities | index("FARGATE") != null)
      and .taskDefinition.runtimePlatform.operatingSystemFamily == "LINUX"
      and .taskDefinition.runtimePlatform.cpuArchitecture == $arch
      and (.taskDefinition.executionRoleArn | type == "string" and length > 0)
      and (.taskDefinition.taskRoleArn | type == "string" and length > 0)
      and .taskDefinition.executionRoleArn != .taskDefinition.taskRoleArn
      and ([.taskDefinition.containerDefinitions[] | select(.name == $container and .essential == true)] | length) == 1
      and ([.taskDefinition.containerDefinitions[] | select(.name == $container)
        | select(.image == $image)
        | select(.logConfiguration.logDriver == "awslogs")
        | select(.logConfiguration.options["awslogs-group"] == $log_group)
        | select(.logConfiguration.options["awslogs-stream-prefix"] == $log_prefix)] | length) == 1
    ' <<<"$raw" >/dev/null
  validated=$(printf '%s' "$raw" \
    | uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
        validate-task-definition-policy --file "$CONTROL_MANIFEST" \
        --scenario-id "$ACTIVE_SCENARIO_ID" --container-name "$container_name" \
        "${user_args[@]}" "${image_role_args[@]}")
  resolved=$(jq -er '.task_definition_arn | select(length > 0)' <<<"$validated")
  printf -v "$variable" '%s' "$resolved"
}

validate_scenario_task_definitions() {
  resolve_bound_task_definition CANDIDATE_TASK_DEFINITION "$WEB_CONTAINER_NAME"
  resolve_bound_task_definition DOCTOR_TASK_DEFINITION "$DOCTOR_CONTAINER_NAME"
  resolve_bound_task_definition PAYLOAD_VERIFIER_TASK_DEFINITION "$WEB_CONTAINER_NAME" 1000:1000
  resolve_bound_task_definition LOCAL_AUTH_VERIFIER_TASK_DEFINITION "$WEB_CONTAINER_NAME" 1000:1000
  if test "$DEPLOYMENT_MODE" = upgrade; then
    resolve_bound_task_definition ROLLBACK_DOCTOR_TASK_DEFINITION "$DOCTOR_CONTAINER_NAME"
    resolve_bound_task_definition PREVIOUS_TASK_DEFINITION "$WEB_CONTAINER_NAME"
  fi
  TASK_DEFINITIONS_VALIDATED_FOR="$ACTIVE_SCENARIO_ID"
}

bind_compatibility_record() {
  local variable="SCENARIO_${ACTIVE_SCENARIO_ID}_COMPATIBILITY_RECORD_FILE" record receipt raw_sha
  record=${!variable:?set the protected compatibility record for the active scenario}
  receipt=$(mktemp -p /tmp "elspeth-compatibility-${ACTIVE_SCENARIO_ID}.XXXXXX")
  chmod 600 "$receipt"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
    compatibility-record-validate --file "$CONTROL_MANIFEST" \
    --scenario-id "$ACTIVE_SCENARIO_ID" --record "$record" >"$receipt"
  raw_sha=$(jq -cS . "$record" | sha256sum | awk '{print $1}')
  test "$(jq -er '.record_sha256' "$receipt")" = "$raw_sha"
  persist_sanitized_receipt "$ACTIVE_SCENARIO_ID" compatibility-record "$raw_sha" "$receipt" >/dev/null
  COMPATIBILITY_RECORD_FILE="$record"
  COMPATIBILITY_RECORD_SHA256="$raw_sha"
  rm -f -- "$receipt"
}

require_compatibility_record_current() {
  local receipt current_sha
  receipt=$(mktemp -p /tmp "elspeth-compatibility-current-${ACTIVE_SCENARIO_ID}.XXXXXX")
  chmod 600 "$receipt"
  current_sha=$(jq -cS . "$COMPATIBILITY_RECORD_FILE" | sha256sum | awk '{print $1}')
  test "$current_sha" = "$COMPATIBILITY_RECORD_SHA256"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
    compatibility-record-validate --file "$CONTROL_MANIFEST" \
    --scenario-id "$ACTIVE_SCENARIO_ID" --record "$COMPATIBILITY_RECORD_FILE" >"$receipt"
  test "$(jq -er '.record_sha256' "$receipt")" = "$COMPATIBILITY_RECORD_SHA256"
  jq -e '
    .forward_compatible == true
    and .backward_compatible == false
    and .rollback_permitted == false
  ' "$receipt" >/dev/null
  rm -f -- "$receipt"
}

set_traffic_action() {
  local mode="$1" actions expected observed
  case "$mode" in
    forward) actions="$FIRST_DEPLOY_FORWARD_ACTIONS" ;;
    disabled) actions="$FIRST_DEPLOY_DISABLED_ACTIONS" ;;
    *) return 64 ;;
  esac
  aws_capture aws elbv2 modify-rule --rule-arn "$FIRST_DEPLOY_LISTENER_RULE_ARN" \
    --actions "$actions" >/dev/null
  observed=$(aws_capture aws elbv2 describe-rules \
    --rule-arns "$FIRST_DEPLOY_LISTENER_RULE_ARN" --output json)
  expected=$(jq -cS . <<<"$actions")
  test "$(jq -cS '.Rules | select(length == 1) | .[0].Actions' <<<"$observed")" = "$expected"
}

verify_task_definition_target_mapping() {
  local expected_task_definition="$1"
  local service running pending task task_ip task_port target_health
  service=$(aws_capture aws ecs describe-services \
    --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" --output json)
  jq -e --arg task_definition "$expected_task_definition" '
    (.failures | length) == 0 and (.services | length) == 1
    and .services[0].desiredCount == 1 and .services[0].runningCount == 1
    and .services[0].pendingCount == 0
    and (.services[0].deployments | length) == 1
    and .services[0].deployments[0].status == "PRIMARY"
    and .services[0].deployments[0].taskDefinition == $task_definition
    and .services[0].deployments[0].rolloutState == "COMPLETED"
  ' <<<"$service" >/dev/null
  task_port=$(jq -er --arg target_group "$TARGET_GROUP_ARN" --arg container "$WEB_CONTAINER_NAME" '
    [.services[0].loadBalancers[]
      | select(.targetGroupArn == $target_group and .containerName == $container)]
    | select(length == 1) | .[0].containerPort
    | select(type == "number" and . >= 1 and . <= 65535)
  ' <<<"$service")
  running=$(aws_capture aws ecs list-tasks --cluster "$ECS_CLUSTER" \
    --service-name "$ECS_SERVICE" --desired-status RUNNING --output json)
  pending=$(aws_capture aws ecs list-tasks --cluster "$ECS_CLUSTER" \
    --service-name "$ECS_SERVICE" --desired-status PENDING --output json)
  jq -e '.taskArns | length == 1' <<<"$running" >/dev/null
  jq -e '.taskArns | length == 0' <<<"$pending" >/dev/null
  ACTIVE_TASK_ARN=$(jq -er '.taskArns[0]' <<<"$running")
  task=$(aws_capture aws ecs describe-tasks --cluster "$ECS_CLUSTER" \
    --tasks "$ACTIVE_TASK_ARN" --output json)
  jq -e --arg task "$ACTIVE_TASK_ARN" --arg definition "$expected_task_definition" \
    --arg container "$WEB_CONTAINER_NAME" '
    (.failures | length) == 0 and (.tasks | length) == 1
    and .tasks[0].taskArn == $task
    and .tasks[0].taskDefinitionArn == $definition
    and .tasks[0].lastStatus == "RUNNING"
    and ([.tasks[0].containers[] | select(.name == $container)] | length) == 1
    and ([.tasks[0].containers[] | select(.name == $container)
      | .managedAgents[]
      | select(.name == "ExecuteCommandAgent" and .lastStatus == "RUNNING")] | length) == 1
  ' <<<"$task" >/dev/null
  task_ip=$(jq -er '
    [.tasks[0].attachments[] | select(.type == "ElasticNetworkInterface")
      | .details[] | select(.name == "privateIPv4Address") | .value]
    | select(length == 1) | .[0]
    | select(test("^[0-9]{1,3}(\\.[0-9]{1,3}){3}$"))
  ' <<<"$task")
  target_health=$(aws_capture aws elbv2 describe-target-health \
    --target-group-arn "$TARGET_GROUP_ARN" --output json)
  jq -e --arg id "$task_ip" --argjson port "$task_port" '
    [.TargetHealthDescriptions[]
      | select(.Target.Id == $id and .Target.Port == $port and .TargetHealth.State == "healthy")]
      | length == 1
  ' <<<"$target_health" >/dev/null
  jq -e --arg id "$task_ip" --argjson port "$task_port" '
    all(.TargetHealthDescriptions[];
      (.Target.Id == $id and .Target.Port == $port) or .TargetHealth.State == "draining")
  ' <<<"$target_health" >/dev/null
}

verify_public_probes() {
  local health_body ready_body health_status ready_status
  health_body=$(mktemp -p /tmp elspeth-health.XXXXXX)
  ready_body=$(mktemp -p /tmp elspeth-ready.XXXXXX)
  chmod 600 "$health_body" "$ready_body"
  if health_status=$(curl --cacert "$ACCEPTANCE_TLS_CA_BUNDLE" \
      --connect-timeout 5 --max-time 10 --max-redirs 0 --max-filesize 1048576 \
      -sS -o "$health_body" -w '%{http_code}' "$ALB_BASE_URL/api/health") \
    && ready_status=$(curl --cacert "$ACCEPTANCE_TLS_CA_BUNDLE" \
      --connect-timeout 5 --max-time 10 --max-redirs 0 --max-filesize 1048576 \
      -sS -o "$ready_body" -w '%{http_code}' "$ALB_BASE_URL/api/ready") \
    && test "$health_status" = 200 && test "$ready_status" = 200 \
    && jq -e '.ready == true' "$ready_body" >/dev/null; then
    rm -f -- "$health_body" "$ready_body"
    return 0
  fi
  rm -f -- "$health_body" "$ready_body"
  return 1
}

require_stopped_task_success() {
  local task_definition="$1" task_arn="$2" definition essential result
  definition=$(aws_capture aws ecs describe-task-definition \
    --task-definition "$task_definition" --output json)
  essential=$(jq -ce '
    [.taskDefinition.containerDefinitions[] | select(.essential != false) | .name] as $names
    | select(($names | length) > 0 and ($names | length) == ($names | unique | length))
    | ($names | sort)' <<<"$definition")
  result=$(aws_capture aws ecs describe-tasks \
    --cluster "$ECS_CLUSTER" --tasks "$task_arn" --output json)
  jq -e --arg task_definition "$task_definition" --argjson essential "$essential" '
    (.failures | length) == 0 and (.tasks | length) == 1
    and .tasks[0].taskDefinitionArn == $task_definition
    and .tasks[0].lastStatus == "STOPPED"
    and ([.tasks[0].containers[].name] | sort) as $actual
    and ($essential - $actual | length) == 0
    and all($essential[] as $name;
      ([.tasks[0].containers[] | select(.name == $name and (.exitCode | type == "number") and .exitCode == 0)] | length) == 1)
  ' <<<"$result" >/dev/null
}

provision_scenario_storage() {
  local overrides task
  overrides=$(jq -cn --arg name "$WEB_CONTAINER_NAME" \
    '{containerOverrides:[{name:$name,command:["provision-storage"]}]}')
  task=$(aws_capture aws ecs run-task \
    --cluster "$ECS_CLUSTER" --task-definition "$PAYLOAD_VERIFIER_TASK_DEFINITION" \
    --launch-type FARGATE --network-configuration "$DOCTOR_NETWORK_CONFIGURATION" \
    --count 1 --overrides "$overrides" --query 'tasks[0].taskArn' --output text)
  aws_waiter_capture aws ecs wait tasks-stopped \
    --cluster "$ECS_CLUSTER" --tasks "$task" >/dev/null
  require_stopped_task_success "$PAYLOAD_VERIFIER_TASK_DEFINITION" "$task"
}
```

### Fresh Scenario A database baseline

Scenario A is a true first deploy. Its Terraform stack creates the database,
EFS paths, disabled 503 listener rule, task definitions, and service at desired
count zero, but application startup never creates schemas. While traffic is
still disabled, run the candidate image's init-capable doctor once under the
database-operator-approved schema-owner secret. The ordinary ordered rollout
then runs the same candidate's read-only doctor before launching the first web
task.

Execute this block only on the first pass, immediately after
`activate_scenario_stack A`. Skip the Scenario B baseline below and continue
with ordered rollout sections 1 through 7.

```bash
load_scenario A
test "$ACTIVE_SCENARIO_ID" = A
validate_scenario_task_definitions
bind_compatibility_record
require_compatibility_record_current
SERVICE_JSON=$(aws_capture aws ecs describe-services \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" --output json)
jq -e '(.failures | length) == 0 and (.services | length) == 1
  and .services[0].desiredCount == 0 and .services[0].runningCount == 0
  and .services[0].pendingCount == 0' <<<"$SERVICE_JSON" >/dev/null
unset SERVICE_JSON
provision_scenario_storage

require_compatibility_record_current
FIRST_INIT_OVERRIDES=$(jq -cn --arg name "$DOCTOR_CONTAINER_NAME" \
  '{containerOverrides:[{name:$name,command:["doctor","aws-ecs","--init-schema","--json"]}]}')
FIRST_INIT_TASK=$(aws_capture aws ecs run-task \
  --cluster "$ECS_CLUSTER" \
  --task-definition "$DOCTOR_TASK_DEFINITION" \
  --launch-type FARGATE --network-configuration "$DOCTOR_NETWORK_CONFIGURATION" \
  --count 1 --overrides "$FIRST_INIT_OVERRIDES" \
  --query 'tasks[0].taskArn' --output text)
aws_waiter_capture aws ecs wait tasks-stopped \
  --cluster "$ECS_CLUSTER" --tasks "$FIRST_INIT_TASK" >/dev/null
require_stopped_task_success "$DOCTOR_TASK_DEFINITION" "$FIRST_INIT_TASK"
unset FIRST_INIT_OVERRIDES
```

Failure leaves desired count zero and the listener fixed at 503. Do not run
the candidate service or switch the listener until the later read-only doctor
and schema compatibility gate both pass.

### Fresh Scenario B upgrade baseline

Scenario B proves an upgrade, but a fresh account has neither schema nor a
running previous revision. Its Terraform stack therefore creates the service
at desired count zero while registering both the rollback-baseline web task
and its database-operator-approved init-capable doctor definition. After the
resolved inventory, TLS material, and test identity are bound, initialize the
empty database with the rollback baseline, then launch that previous web
revision. Only this bootstrap establishes the pre-candidate state; it is not a
candidate deployment or a substitute for the later rollback-refusal and
forward-recovery proof.

Execute this block only after Scenario A has completed section 7 and the
operator has returned to the saved-plan section to run
`activate_scenario_stack B`. Do not rerun the Scenario A baseline.

```bash
load_scenario B
test "$ACTIVE_SCENARIO_ID" = B
validate_scenario_task_definitions
bind_compatibility_record
require_compatibility_record_current
prepare_scenario_b_oidc
set_traffic_action disabled
SERVICE_JSON=$(aws_capture aws ecs describe-services \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" --output json)
jq -e '(.failures | length) == 0 and (.services | length) == 1
  and .services[0].desiredCount == 0 and .services[0].runningCount == 0
  and .services[0].pendingCount == 0' <<<"$SERVICE_JSON" >/dev/null
unset SERVICE_JSON
provision_scenario_storage

require_compatibility_record_current
ROLLBACK_INIT_OVERRIDES=$(jq -cn --arg name "$DOCTOR_CONTAINER_NAME" \
  '{containerOverrides:[{name:$name,command:["doctor","aws-ecs","--init-schema","--json"]}]}')
ROLLBACK_INIT_TASK=$(aws_capture aws ecs run-task \
  --cluster "$ECS_CLUSTER" \
  --task-definition "$ROLLBACK_DOCTOR_TASK_DEFINITION" \
  --launch-type FARGATE --network-configuration "$DOCTOR_NETWORK_CONFIGURATION" \
  --count 1 --overrides "$ROLLBACK_INIT_OVERRIDES" \
  --query 'tasks[0].taskArn' --output text)
aws_waiter_capture aws ecs wait tasks-stopped \
  --cluster "$ECS_CLUSTER" --tasks "$ROLLBACK_INIT_TASK" >/dev/null
require_stopped_task_success "$ROLLBACK_DOCTOR_TASK_DEFINITION" "$ROLLBACK_INIT_TASK"
unset ROLLBACK_INIT_OVERRIDES

require_compatibility_record_current
aws_capture aws ecs update-service \
  --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
  --task-definition "$PREVIOUS_TASK_DEFINITION" --desired-count 1 \
  --force-new-deployment \
  --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
  >/dev/null
aws_waiter_capture aws ecs wait services-stable \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
SERVICE_JSON=$(aws_capture aws ecs describe-services \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" --output json)
jq -e --arg previous "$PREVIOUS_TASK_DEFINITION" '
  (.failures | length) == 0 and (.services | length) == 1
  and .services[0].taskDefinition == $previous
  and .services[0].desiredCount == 1 and .services[0].runningCount == 1
  and .services[0].pendingCount == 0' <<<"$SERVICE_JSON" >/dev/null
unset SERVICE_JSON
verify_task_definition_target_mapping "$PREVIOUS_TASK_DEFINITION"
set_traffic_action forward
verify_public_probes
run_oidc_evidence previous-before-candidate
set_traffic_action disabled
```

From this point the ordinary upgrade preflight requires desired/running count
one on `PREVIOUS_TASK_DEFINITION`. Any failed baseline doctor, unstable
service, unhealthy target, or failed browser round trip blocks the candidate.

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

Task 4's inspected `elspeth:ecs-0.7.1-closeout` image is the only candidate.
The earlier fresh-account publication step binds that exact local image to its
registry digest before either scenario apply; do not rebuild or retag a second
candidate here. Validate the approved platform mapping only:

```bash
set -Eeuo pipefail
TARGET_PLATFORM=${TARGET_PLATFORM:?approved linux/amd64 or linux/arm64 required}
case "$TARGET_PLATFORM" in
  linux/amd64) ECS_CPU_ARCHITECTURE=X86_64 ;;
  linux/arm64) ECS_CPU_ARCHITECTURE=ARM64 ;;
  *) printf '%s\n' 'unsupported TARGET_PLATFORM' >&2; exit 2 ;;
esac

```

Every web, doctor, verifier, and rollback definition declares
`runtimePlatform.operatingSystemFamily == LINUX` and the mapped
`runtimePlatform.cpuArchitecture` (`linux/amd64` → `X86_64`, `linux/arm64` →
`ARM64`). A host-native image with no recorded target platform is NO-GO. The
lean image omits the `azure` extra; `azure_blob` pipelines need the default
`all` image or an expanded `INSTALL_EXTRAS`. The published GHCR/ACR default
remains `all`.

## Storage provisioning and cold start

Before doctor, run `provision_scenario_storage` above. It provisions
`data_dir`, explicit `payload_store_path`, and the derived blob directory on
the intended EFS access point and proves them as the non-root `1000:1000`
user. Mounting only the parent is insufficient: doctor deliberately never
calls `mkdir`, including with `--init-schema`, because an overlay child would
mask a displaced EFS mount. Record the filesystem/access-point identity and
non-root probe booleans, never paths or credentials.

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
    log_group_name: ${OPERATOR_METRICS_LOG_GROUP}
    log_stream_name: telemetry
    dimension_rollup_option: NoDimensionRollup
    retain_initial_value_of_delta_metric: true
    resource_to_telemetry_conversion:
      enabled: true
  awsxray/elspeth:
    indexed_attributes: [run_id, status]
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
scenario-owned `OPERATOR_METRICS_LOG_GROUP`, and extracts them into the
`ELSPETH/Operator` CloudWatch namespace. `NoDimensionRollup` prevents the
exporter from silently creating additional dimension sets, and retaining the
first delta value preserves low-frequency acceptance and failure counters.
The `awsxray` exporter sends traces to X-Ray and indexes only the bounded
`run_id` and `status` lifecycle attributes that the acceptance query reads as
annotations. Both exporters use the default AWS credential chain and therefore
the ECS task role; neither accepts an endpoint, role override, profile, or
static credential here.

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
        {"name": "ELSPETH_ACCEPTANCE_SCENARIO_ID", "value": "${SCENARIO_ID}"},
        {"name": "ELSPETH_ACCEPTANCE_S3_BUCKET", "value": "${ELSPETH_TEST_S3_BUCKET}"},
        {"name": "ELSPETH_ACCEPTANCE_S3_PREFIX", "value": "${SCENARIO_RESOURCE_NAMESPACE}/${ACCEPTANCE_RUN_ID}"}
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

Pre-create the scenario-owned `OPERATOR_METRICS_LOG_GROUP`, include it in the
protected scenario inventory and orphan sweep, apply the retention policy
below, and do not grant the task permission to create arbitrary log groups.
The **Task role** carries application-time EMF and trace delivery. The
temporary acceptance task role also carries the two read APIs used by the
in-task positive/outage verifier; a durable production role may omit those
reads. The EMF exporter needs stream discovery/creation and event writes on
that exact group. CloudWatch metric reads and X-Ray APIs do not support useful
narrower resource scoping:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PutElspethOperatorMetricEmf",
      "Effect": "Allow",
      "Action": ["logs:DescribeLogStreams", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": [
        "arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT}:log-group:${OPERATOR_METRICS_LOG_GROUP}",
        "arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT}:log-group:${OPERATOR_METRICS_LOG_GROUP}:log-stream:telemetry"
      ]
    },
    {
      "Sid": "PutElspethTraces",
      "Effect": "Allow",
      "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
      "Resource": "*"
    },
    {
      "Sid": "ReadElspethAcceptanceTelemetry",
      "Effect": "Allow",
      "Action": ["cloudwatch:GetMetricData", "xray:BatchGetTraces"],
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
- `WEB_LOG_GROUP`, `WEB_LOG_STREAM_PREFIX`, `DOCTOR_LOG_GROUP`,
  `DOCTOR_LOG_STREAM_PREFIX`, and `OPERATOR_METRICS_LOG_GROUP`;
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
: "${DOCTOR_CONTAINER_NAME:?} ${DOCTOR_NETWORK_CONFIGURATION:?}"
: "${WEB_LOG_GROUP:?} ${WEB_LOG_STREAM_PREFIX:?}"
: "${DOCTOR_LOG_GROUP:?} ${DOCTOR_LOG_STREAM_PREFIX:?} ${OPERATOR_METRICS_LOG_GROUP:?}"
: "${ECS_DEPLOYMENT_EVENT_RULE:?} ${ECS_DEPLOYMENT_EVENT_TARGET_ID:?}"
: "${ECS_DEPLOYMENT_EVENT_LOG_GROUP:?} ${ACCEPTANCE_RUN_ID:?}"

case "$DEPLOYMENT_MODE" in
  upgrade) : "${PREVIOUS_TASK_DEFINITION:?} ${ROLLBACK_DOCTOR_TASK_DEFINITION:?}" ;;
  first|first-recovery)
    : "${FIRST_DEPLOY_LISTENER_RULE_ARN:?}"
    : "${FIRST_DEPLOY_FORWARD_ACTIONS:?} ${FIRST_DEPLOY_DISABLED_ACTIONS:?}"
    ;;
  *) printf '%s\n' 'invalid DEPLOYMENT_MODE' >&2; exit 2 ;;
esac
[[ "$ECS_CLUSTER" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "$ECS_SERVICE" =~ ^[A-Za-z0-9_-]+$ ]]

test "$TASK_DEFINITIONS_VALIDATED_FOR" = "$ACTIVE_SCENARIO_ID"

OBSERVATION_START_EPOCH_MS=$(($(date +%s) * 1000))
SERVICE_JSON=$(aws_capture aws ecs describe-services \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" --output json)
SCALING_JSON=$(aws_capture aws application-autoscaling describe-scalable-targets \
  --service-namespace ecs \
  --resource-ids "service/$ECS_CLUSTER/$ECS_SERVICE" --output json)
jq -e --arg cluster "$ECS_CLUSTER" --arg service "$ECS_SERVICE" \
  --arg target_group "$TARGET_GROUP_ARN" --arg container "$WEB_CONTAINER_NAME" \
  --arg mode "$DEPLOYMENT_MODE" '
    (.failures | length) == 0 and (.services | length) == 1
    and (.services[0].serviceArn | endswith("/" + $service))
    and (.services[0].clusterArn | endswith("/" + $cluster))
    and .services[0].launchType == "FARGATE"
    and (.services[0].platformVersion == "1.4.0" or .services[0].platformVersion == "LATEST")
    and .services[0].deploymentController.type == "ECS"
    and .services[0].enableExecuteCommand == true
    and .services[0].deploymentConfiguration.minimumHealthyPercent == 0
    and .services[0].deploymentConfiguration.maximumPercent == 100
    and .services[0].healthCheckGracePeriodSeconds >= 150
    and ([.services[0].loadBalancers[]
      | select(.targetGroupArn == $target_group and .containerName == $container)] | length) == 1
    and (if $mode == "upgrade" then
      .services[0].desiredCount == 1 and .services[0].runningCount == 1
      and .services[0].pendingCount == 0
    else
      .services[0].desiredCount == 0 and .services[0].runningCount == 0
      and .services[0].pendingCount == 0
    end)
  ' <<<"$SERVICE_JSON" >/dev/null
jq -e '.ScalableTargets | length == 0' <<<"$SCALING_JSON" >/dev/null
TARGET_GROUP_JSON=$(aws_capture aws elbv2 describe-target-groups \
  --target-group-arns "$TARGET_GROUP_ARN" --output json)
jq -e --arg arn "$TARGET_GROUP_ARN" '
  (.TargetGroups | length) == 1
  and .TargetGroups[0].TargetGroupArn == $arn
  and .TargetGroups[0].TargetType == "ip"
  and .TargetGroups[0].HealthCheckEnabled == true
  and .TargetGroups[0].HealthCheckPath == "/api/ready"
  and .TargetGroups[0].Matcher.HttpCode == "200"
  and .TargetGroups[0].HealthCheckTimeoutSeconds >= 6
' <<<"$TARGET_GROUP_JSON" >/dev/null

CANDIDATE_DEFINITION_JSON=$(aws_capture aws ecs describe-task-definition \
  --task-definition "$CANDIDATE_TASK_DEFINITION" --output json)
jq -e --arg container "$WEB_CONTAINER_NAME" '
  [.taskDefinition.containerDefinitions[] | select(.name == $container)] | length == 1
  and .[0].healthCheck.startPeriod >= 150
  and .[0].healthCheck.command == ["CMD","python","-c",
    "import http.client,sys; c=http.client.HTTPConnection('\''127.0.0.1'\'',8451,timeout=5); c.request('\''GET'\','\''/api/health'\''); r=c.getresponse(); sys.exit(0 if r.status == 200 else 1)"]
' <<<"$CANDIDATE_DEFINITION_JSON" >/dev/null
unset SERVICE_JSON SCALING_JSON TARGET_GROUP_JSON CANDIDATE_DEFINITION_JSON
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

```bash
DEPLOYMENT_RULE_JSON=$(aws_capture aws events describe-rule \
  --region "$AWS_REGION" --name "$ECS_DEPLOYMENT_EVENT_RULE" --event-bus-name default)
jq -e '
  .State == "ENABLED"
  and (.EventPattern | fromjson
    | keys == ["detail","detail-type","source"]
    and .source == ["aws.ecs"]
    and .["detail-type"] == ["ECS Deployment State Change"]
    and .detail == {"eventName":["SERVICE_DEPLOYMENT_FAILED"]})
' <<<"$DEPLOYMENT_RULE_JSON" >/dev/null
DEPLOYMENT_TARGETS_JSON=$(aws_capture aws events list-targets-by-rule \
  --region "$AWS_REGION" --rule "$ECS_DEPLOYMENT_EVENT_RULE" --event-bus-name default)
EXPECTED_EVENT_LOG_ARN="arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT_ID}:log-group:${ECS_DEPLOYMENT_EVENT_LOG_GROUP}"
jq -e --arg id "$ECS_DEPLOYMENT_EVENT_TARGET_ID" --arg arn "$EXPECTED_EVENT_LOG_ARN" '
  (.Targets | length) == 1 and .Targets[0].Id == $id and .Targets[0].Arn == $arn
  and (.Targets[0] | has("RoleArn") | not)
' <<<"$DEPLOYMENT_TARGETS_JSON" >/dev/null
LOG_POLICIES_JSON=$(aws_capture aws logs describe-resource-policies --region "$AWS_REGION")
ACTIVE_INVENTORY_PATH=$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
  control-manifest get --file "$CONTROL_MANIFEST" \
  --field "scenarios.${ACTIVE_SCENARIO_ID}.inventory_path")
LOG_RESOURCE_POLICY_NAME=$(jq -er '
  .orphan_sweep.log_resource_policy_names
  | select(type == "array" and length == 1) | .[0]
' "$ACTIVE_INVENTORY_PATH")
jq -e --arg name "$LOG_RESOURCE_POLICY_NAME" --arg arn "${EXPECTED_EVENT_LOG_ARN}:*" '
  [.resourcePolicies[] | select(.policyName == $name)] as $policies
  | ($policies | length) == 1
  and ($policies[0].policyDocument | fromjson
    | (.Statement | type == "array" and length == 1)
    and (.Statement[0] as $statement
      | (($statement | keys - ["Sid"] | sort) == ["Action","Effect","Principal","Resource"])
      and $statement.Effect == "Allow"
      and $statement.Resource == $arn
      and (($statement.Action | if type == "string" then [.] else . end | sort)
        == ["logs:CreateLogStream","logs:PutLogEvents"])
      and (($statement.Principal | keys) == ["Service"])
      and (($statement.Principal.Service | if type == "string" then [.] else . end | sort)
        == ["delivery.logs.amazonaws.com","events.amazonaws.com"])))
' <<<"$LOG_POLICIES_JSON" >/dev/null
unset DEPLOYMENT_RULE_JSON DEPLOYMENT_TARGETS_JSON LOG_POLICIES_JSON \
  ACTIVE_INVENTORY_PATH LOG_RESOURCE_POLICY_NAME

run_event_delivery_canary() (
  set -Eeuo pipefail
  local namespace rule target correlation event_pattern event_entry events delivered=0 created=0
  namespace=$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
    scenario-namespace --acceptance-run-id "$ACCEPTANCE_RUN_ID" \
    --scenario-id "$ACTIVE_SCENARIO_ID")
  rule="${namespace}-event-canary"
  target="${namespace}-canary-target"
  correlation=$(uv run --frozen python -c 'import uuid; print(uuid.uuid4())')
  event_pattern='{"source":["elspeth.acceptance"]}'
  event_entry=$(jq -cn --arg correlation "$correlation" \
    '[{Source:"elspeth.acceptance",DetailType:"acceptance-canary",
      Detail:({correlation:$correlation}|tojson)}]')
  cleanup_canary() {
    test "$created" = 1 || return 0
    aws_capture aws events remove-targets --region "$AWS_REGION" --event-bus-name default \
      --rule "$rule" --ids "$target" >/dev/null 2>&1 || true
    aws_capture aws events delete-rule --region "$AWS_REGION" --event-bus-name default \
      --name "$rule" >/dev/null 2>&1 || true
  }
  trap cleanup_canary EXIT
  aws_capture aws events put-rule --region "$AWS_REGION" --event-bus-name default \
    --name "$rule" --state ENABLED --event-pattern "$event_pattern" \
    --tags "Key=ACCEPTANCE_RUN_ID,Value=$ACCEPTANCE_RUN_ID" \
      "Key=SCENARIO_ID,Value=$ACTIVE_SCENARIO_ID" >/dev/null
  created=1
  aws_capture aws events put-targets --region "$AWS_REGION" --event-bus-name default \
    --rule "$rule" --targets "Id=$target,Arn=$EXPECTED_EVENT_LOG_ARN" \
    | jq -e '.FailedEntryCount == 0' >/dev/null
  aws_capture aws events put-events --region "$AWS_REGION" --entries "$event_entry" \
    | jq -e '.FailedEntryCount == 0' >/dev/null
  for _attempt in $(seq 1 30); do
    events=$(aws_capture aws logs filter-log-events --region "$AWS_REGION" \
      --log-group-name "$ECS_DEPLOYMENT_EVENT_LOG_GROUP" --filter-pattern "\"$correlation\"")
    if jq -e '(.events | length) >= 1' <<<"$events" >/dev/null; then
      delivered=1
      break
    fi
    sleep 5
  done
  test "$delivered" = 1
  cleanup_canary
  created=0
  test "$(aws_capture aws events list-rules --region "$AWS_REGION" \
    --event-bus-name default --name-prefix "$rule" \
    --query "length(Rules[?Name=='$rule'])" --output text)" = 0
  printf '%s' '{"schema":"elspeth.aws-ecs-event-canary.v1","delivered":true,"removed":true}' \
    | persist_sanitized_receipt "$ACTIVE_SCENARIO_ID" deployment-event-canary \
      "$ECS_DEPLOYMENT_EVENT_RULE"
)

run_event_delivery_canary
```

### 2. Run the hardened one-shot doctor

Use the candidate digest in the doctor definition. The schema-owner variant is
used only for a database-operator-approved `--init-schema` action.

```bash
require_compatibility_record_current
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

aws_waiter_capture aws ecs wait tasks-stopped \
  --cluster "$ECS_CLUSTER" --tasks "$DOCTOR_TASK_ARN" >/dev/null
require_stopped_task_success "$DOCTOR_TASK_DEFINITION" "$DOCTOR_TASK_ARN"
```

Missing/null `exitCode`, a name mismatch, non-zero essential-container exit,
or sanitized doctor failure blocks service mutation. Diagnose only through
bounded `aws logs filter-log-events` calls captured by `aws_capture` and sent
directly to `sanitize-evidence`; raw logs are never printed or persisted.

### 3. Apply the schema compatibility gate

`--init-schema` may initialize a session or Landscape schema only when it is
MISSING; a partially present or predecessor schema is STALE. Before 1.0 there
are no SQLite or PostgreSQL in-place exceptions. Read-only and
`create_tables=False` inspection opens report incompatibility without mutation.
The database owner archives/exports required evidence, drops and recreates the
store, then initializes the current release. Aurora detection is
structural-only: a semantics-only schema-epoch change can still appear CURRENT.

Attach the approved release/schema compatibility record before deployment.
AWS ECS validate-only startup must fail closed before uvicorn binds for
missing, partial, stale, or incompatible state. Code rollback cannot undo an
incompatible database schema. STALE/incompatible state requires the
database-operator-owned archive decision and drop/recreate procedure followed
by `--init-schema`; never automate it. Predecessor archives are evidence, not
recovery inputs for the current release. If the fresh candidate fails, fix it
forward and repeat the uninstall/recreate/reinstall procedure.

### 4. Deploy exactly one candidate task

Capture the previous primary deployment ID/time. Changing desired count alone
does not prove a new deployment. Every intentional launch, relaunch, upgrade,
or manual upgrade rollback uses a forced new deployment:

```bash
require_compatibility_record_current
set_traffic_action disabled
aws_capture aws ecs update-service \
  --cluster "$ECS_CLUSTER" \
  --service "$ECS_SERVICE" \
  --task-definition "$CANDIDATE_TASK_DEFINITION" \
  --desired-count 1 \
  --force-new-deployment \
  --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
  >/dev/null

aws_waiter_capture aws ecs wait services-stable \
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
verify_candidate_target_mapping() {
  verify_task_definition_target_mapping "$CANDIDATE_TASK_DEFINITION"
  CANDIDATE_TASK_ARN="$ACTIVE_TASK_ARN"
}

run_candidate_role_check() {
  local task_arn="$1" check="$2" phase="${3:-}" landscape_run_id="${4:-}"
  local stream receipt_file command
  local -a binding_args=()
  case "$phase" in
    "") command="python -m elspeth.web.aws_ecs_acceptance $check" ;;
    positive|outage)
      test "$check" = verify-operator-telemetry || return 64
      command="python -m elspeth.web.aws_ecs_acceptance $check --phase $phase"
      if test -n "$landscape_run_id"; then
        test "$phase" = positive || return 64
        [[ "$landscape_run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$ ]] || return 64
        command="$command --landscape-run-id $landscape_run_id"
      fi
      ;;
    *) return 64 ;;
  esac
  if test "$check" = verify-bedrock-guardrails; then
    binding_args=(--plugin-policy-binding-sha256 "$ELSPETH_ACCEPTANCE_PLUGIN_POLICY_BINDING_SHA256")
  fi
  stream=$(aws_exec_capture aws ecs execute-command \
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

verify_candidate_target_mapping
run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-s3
run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock
run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock-guardrails
set_traffic_action forward
verify_public_probes
```

#### Persistence, replacement, task-role, and local-auth sequence

Run these checks in this order; a later check never substitutes for an earlier
one:

1. Before admitting traffic, run S3, Bedrock, and both Guardrail checks through
   the initial candidate task role.
2. In the disposable local-auth scenario, capture one fixed pipeline through
   the public API into a protected state file.
3. Force a distinct candidate deployment and prove the old task was replaced.
4. Re-authenticate from a fresh process and run `verify-api` against that same
   protected state without mutating the session. Require the six-row
   `GET /api/system/status` contract and the typed HTTP 409
   `tutorial_required_control_coverage` recheck as part of this command.
5. Start the explicit `1000:1000` one-shot payload verifier with the candidate
   digest and the same PostgreSQL/EFS settings; do not use root-running ECS
   Exec for this proof.
6. From every contributing healthy candidate task, use ECS Exec for the S3,
   Bedrock, Guardrail, and operator-telemetry checks and locally extract the
   one sanitized receipt sentinel.
7. Drain traffic to fixed 503, scale the service to zero, then start the
   explicit `1000:1000` local-auth verifier against the same EFS mount.

```bash
case "$ACTIVE_SCENARIO_ID" in
  A)
    ACCEPTANCE_STATE=$(mktemp -p /tmp elspeth-aws-ecs-state.XXXXXX)
    rm -f "$ACCEPTANCE_STATE"
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
      --file "$CONTROL_MANIFEST" --acceptance-state-path "$ACCEPTANCE_STATE"
    SSL_CERT_FILE="$ACCEPTANCE_TLS_CA_BUNDLE" \
    ELSPETH_ACCEPTANCE_REGISTER=1 \
      uv run --frozen python -m elspeth.web.aws_ecs_acceptance capture \
        --state-file "$ACCEPTANCE_STATE"
    LANDSCAPE_RUN_ID=$(jq -er '.landscape_run_id' "$ACCEPTANCE_STATE")
    ;;
  B)
    capture_oidc_lifecycle_handoff
    LANDSCAPE_RUN_ID="$OIDC_LANDSCAPE_RUN_ID"
    ;;
  *) exit 64 ;;
esac

require_compatibility_record_current
PRE_REPLACEMENT_TASK_ARN="$CANDIDATE_TASK_ARN"
aws_capture aws ecs update-service \
  --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
  --task-definition "$CANDIDATE_TASK_DEFINITION" --desired-count 1 \
  --force-new-deployment \
  --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
  >/dev/null
aws_waiter_capture aws ecs wait services-stable \
  --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
verify_candidate_target_mapping
test "$CANDIDATE_TASK_ARN" != "$PRE_REPLACEMENT_TASK_ARN"
verify_public_probes

if test "$ACTIVE_SCENARIO_ID" = A; then
  SSL_CERT_FILE="$ACCEPTANCE_TLS_CA_BUNDLE" \
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance verify-api \
    --state-file "$ACCEPTANCE_STATE"
else
  test "$ACTIVE_SCENARIO_ID" = B
  test "$(jq -er '.landscape_run_id' "$OIDC_LIFECYCLE_STATE")" = "$LANDSCAPE_RUN_ID"
fi

PAYLOAD_OVERRIDES=$(jq -cn --arg name "$WEB_CONTAINER_NAME" --arg run "$LANDSCAPE_RUN_ID" \
  '{containerOverrides:[{name:$name,command:["verify-payloads","--landscape-run-id",$run]}]}')
PAYLOAD_TASK=$(aws_capture aws ecs run-task \
  --cluster "$ECS_CLUSTER" --task-definition "$PAYLOAD_VERIFIER_TASK_DEFINITION" \
  --launch-type FARGATE --network-configuration "$DOCTOR_NETWORK_CONFIGURATION" \
  --count 1 --overrides "$PAYLOAD_OVERRIDES" --query 'tasks[0].taskArn' --output text)
aws_waiter_capture aws ecs wait tasks-stopped --cluster "$ECS_CLUSTER" --tasks "$PAYLOAD_TASK" >/dev/null
require_stopped_task_success "$PAYLOAD_VERIFIER_TASK_DEFINITION" "$PAYLOAD_TASK"

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

run_connection_budget_check() {
  local task_arn="$1" stream envelope_file details_file command
  [[ "$ACCEPTANCE_START_UTC" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]
  [[ "$ACCEPTANCE_CONNECTION_BUDGET" =~ ^[1-9][0-9]{0,8}$ ]]
  [[ "$ACCEPTANCE_CONNECTION_SAFETY_MARGIN" =~ ^[0-9]{1,9}$ ]]
  command="python -m elspeth.web.aws_ecs_acceptance verify-connection-budget"
  command="$command --cluster-id $DB_CLUSTER_IDENTIFIER --start-time $ACCEPTANCE_START_UTC"
  command="$command --approved-budget $ACCEPTANCE_CONNECTION_BUDGET"
  command="$command --safety-margin $ACCEPTANCE_CONNECTION_SAFETY_MARGIN"
  stream=$(aws_exec_capture aws ecs execute-command \
    --cluster "$ECS_CLUSTER" --task "$task_arn" --container "$WEB_CONTAINER_NAME" \
    --interactive --command "$command")
  envelope_file=$(mktemp -p /tmp elspeth-connection-envelope.XXXXXX)
  details_file=$(mktemp -p /tmp elspeth-connection-budget.XXXXXX)
  chmod 600 "$envelope_file" "$details_file"
  printf '%s' "$stream" | uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
    extract-exec-receipt --check verify-connection-budget \
    --candidate-sha "$CANDIDATE_SHA" --task-arn "$task_arn" \
    --scenario-id "$ACTIVE_SCENARIO_ID" >"$envelope_file"
  unset stream
  jq -e '.details' "$envelope_file" >"$details_file"
  persist_sanitized_receipt "$ACTIVE_SCENARIO_ID" connection-budget \
    "$DB_CLUSTER_IDENTIFIER" "$details_file" >/dev/null
  rm -f -- "$envelope_file" "$details_file"
}

run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-s3
run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock
run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock-guardrails
case "$ACTIVE_SCENARIO_ID" in
  A) run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry positive ;;
  B) run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry positive "$OIDC_LANDSCAPE_RUN_ID" ;;
  *) exit 64 ;;
esac

# Run once in Scenario A after the complete positive role-check pass.
if test "$ACTIVE_SCENARIO_ID" = A; then
  aws_exec_capture aws ecs execute-command \
    --cluster "$ECS_CLUSTER" --task "$CANDIDATE_TASK_ARN" \
    --container cloudwatch-agent --interactive \
    --command "/bin/sh -c 'kill -TERM 1'" >/dev/null
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry outage
  OUTAGE_TASK_ARN="$CANDIDATE_TASK_ARN"
  require_compatibility_record_current
  aws_capture aws ecs update-service \
    --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
    --task-definition "$CANDIDATE_TASK_DEFINITION" --desired-count 1 \
    --force-new-deployment \
    --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
    >/dev/null
  aws_waiter_capture aws ecs wait services-stable \
    --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
  verify_candidate_target_mapping
  test "$CANDIDATE_TASK_ARN" != "$OUTAGE_TASK_ARN"
  verify_public_probes
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry positive

  aws_capture aws elbv2 modify-rule --rule-arn "$FIRST_DEPLOY_LISTENER_RULE_ARN" \
    --actions "$FIRST_DEPLOY_DISABLED_ACTIONS" >/dev/null
  aws_capture aws ecs update-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
    --desired-count 0 \
    --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
    >/dev/null
  aws_waiter_capture aws ecs wait services-stable --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
  LOCAL_AUTH_OVERRIDES=$(jq -cn --arg name "$WEB_CONTAINER_NAME" \
    '{containerOverrides:[{name:$name,command:["verify-local-auth"]}]}')
  LOCAL_AUTH_TASK=$(aws_capture aws ecs run-task \
    --cluster "$ECS_CLUSTER" --task-definition "$LOCAL_AUTH_VERIFIER_TASK_DEFINITION" \
    --launch-type FARGATE --network-configuration "$DOCTOR_NETWORK_CONFIGURATION" \
    --count 1 --overrides "$LOCAL_AUTH_OVERRIDES" --query 'tasks[0].taskArn' --output text)
  aws_waiter_capture aws ecs wait tasks-stopped --cluster "$ECS_CLUSTER" --tasks "$LOCAL_AUTH_TASK" >/dev/null
  require_stopped_task_success "$LOCAL_AUTH_VERIFIER_TASK_DEFINITION" "$LOCAL_AUTH_TASK"

  require_compatibility_record_current
  PRE_LOCAL_AUTH_RELAUNCH_TASK_ARN="$CANDIDATE_TASK_ARN"
  aws_capture aws ecs update-service \
    --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
    --task-definition "$CANDIDATE_TASK_DEFINITION" --desired-count 1 \
    --force-new-deployment \
    --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
    >/dev/null
  aws_waiter_capture aws ecs wait services-stable \
    --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
  verify_candidate_target_mapping
  test "$CANDIDATE_TASK_ARN" != "$PRE_LOCAL_AUTH_RELAUNCH_TASK_ARN"
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-s3
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock-guardrails
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry positive
  aws_capture aws elbv2 modify-rule --rule-arn "$FIRST_DEPLOY_LISTENER_RULE_ARN" \
    --actions "$FIRST_DEPLOY_FORWARD_ACTIONS" >/dev/null
  verify_public_probes
fi
```

The `verify-bedrock-guardrails` receipt must contain `plugin_policy` with the
exact `target_llm` and the prompt-shield/content-safety entries in
`selected_controls`, including both opaque aliases and `required` modes, plus
`landscape_evidence: true`. That final flag proves the acceptance run's atomic
`run_web_plugin_policy` row was read back unchanged; a Guardrail API success
without this policy proof is NO-GO.

Both one-shot definitions set task-level `user: "1000:1000"`, override the
image entrypoint exactly once with
`{"entryPoint": ["python", "-m", "elspeth.web.aws_ecs_acceptance"]}`,
use the candidate digest, and reuse the exact
approved EFS access point and database settings. The role-check transport
requires the Session Manager plugin, running `ExecuteCommandAgent`, and only
the module's single bounded receipt sentinel; interactive output is never
echoed or retained.

### 6. Observe for ten minutes

Wait for the next UTC minute boundary, then set an exact `:00Z`
`ACCEPTANCE_START_UTC` immediately after the first complete pass. Poll every 30 seconds for 20 iterations, repeating
the exact service/deployment counts, candidate task/target mapping, target
health, and exact-200 bounded `/api/health` and `/api/ready` checks. Before the
loop, set integer `ACCEPTANCE_CONNECTION_BUDGET` and
`ACCEPTANCE_CONNECTION_SAFETY_MARGIN` from the database operator's approved
record. At the end, the candidate task reads PostgreSQL's live
`max_connections` and queries the `AWS/RDS` `DatabaseConnections` maximums
for the exact cluster and observation window; a missing datapoint, excess over
budget, or insufficient remaining margin fails acceptance.

```bash
OBSERVATION_ALIGNMENT_SECONDS=$((60 - 10#$(date -u +%S)))
sleep "$OBSERVATION_ALIGNMENT_SECONDS"
while test "$(date -u +%S)" != 00; do
  OBSERVATION_ALIGNMENT_SECONDS=$((60 - 10#$(date -u +%S)))
  sleep "$OBSERVATION_ALIGNMENT_SECONDS"
done
ACCEPTANCE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:00Z)
unset OBSERVATION_ALIGNMENT_SECONDS
for iteration in $(seq 1 20); do
  printf 'acceptance_sample=%s\n' "$iteration"
  verify_candidate_target_mapping
  verify_public_probes
  sleep 30
done
run_connection_budget_check "$CANDIDATE_TASK_ARN"
```

`SERVICE_DEPLOYMENT_FAILED`, `rolloutState=FAILED`, launch/placement failures,
non-zero essential exits, or a `stoppedReason` require rollback and stopped
task inspection. Candidate `Target.ResponseCodeMismatch`, `Target.Timeout`, or
`Target.FailedHealthChecks` require diagnosis and rollback if persistent.
Non-200 liveness/readiness, `ready != true`, `readiness_check_not_ready`,
startup/schema failure, or new unhandled `ERROR`/`CRITICAL` blocks acceptance.
Retain only allowlisted checks, classes, counts, and hashes.

### 7. Prove rollback refusal without crossing the schema stop

The current upgrade record proves the opposite of rollback authorization. Once
the candidate has recreated Landscape at epoch 29, the 0.7.0 image must
never be deployed against that database. Scenario B therefore exercises a
fail-closed rollback refusal and forward recovery: revalidate and persist the
sanitized compatibility receipt, prove the candidate task remains the active
target, repeat its role/public probes, and capture fresh OIDC evidence. Do not
disable traffic, invoke `update-service`, or start
`PREVIOUS_TASK_DEFINITION` in this phase.

```bash
if test "$DEPLOYMENT_MODE" = upgrade; then
  require_compatibility_record_current
  ROLLBACK_REFUSAL_RECEIPT=$(mktemp -p /tmp \
    "elspeth-rollback-refusal-${ACTIVE_SCENARIO_ID}.XXXXXX")
  chmod 600 "$ROLLBACK_REFUSAL_RECEIPT"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
    compatibility-record-validate --file "$CONTROL_MANIFEST" \
    --scenario-id "$ACTIVE_SCENARIO_ID" \
    --record "$COMPATIBILITY_RECORD_FILE" >"$ROLLBACK_REFUSAL_RECEIPT"
  jq -e '
    .backward_compatible == false
    and .rollback_permitted == false
    and .schema_facts.previous.landscape_epoch == 23
    and .schema_facts.candidate.landscape_epoch == 29
  ' "$ROLLBACK_REFUSAL_RECEIPT" >/dev/null
  persist_sanitized_receipt "$ACTIVE_SCENARIO_ID" compatibility-record \
    "$COMPATIBILITY_RECORD_SHA256" "$ROLLBACK_REFUSAL_RECEIPT" >/dev/null
  rm -f -- "$ROLLBACK_REFUSAL_RECEIPT"

  verify_candidate_target_mapping
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-s3
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock-guardrails
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry positive "$OIDC_LANDSCAPE_RUN_ID"
  verify_public_probes
  run_oidc_evidence candidate-after-rollback-refusal
fi
```

The compatibility receipt plus `candidate-after-rollback-refusal` evidence is
the refusal/forward-recovery record. If the candidate is unhealthy, keep traffic
drained and repair forward with epoch-35 session/epoch-29 Landscape code.
Predecessor database restoration and code downgrade are not supported repair
paths. Never roll old code over the recreated schema.

For first/first-recovery, remove traffic before compute, verify the listener's
fixed 503 action, then scale to zero:

```bash
if test "$DEPLOYMENT_MODE" != upgrade; then
  PRE_FIRST_RECOVERY_TASK_ARN="$CANDIDATE_TASK_ARN"
  aws_capture aws elbv2 modify-rule \
    --rule-arn "$FIRST_DEPLOY_LISTENER_RULE_ARN" \
    --actions "$FIRST_DEPLOY_DISABLED_ACTIONS" >/dev/null
  RULE_JSON=$(aws_capture aws elbv2 describe-rules \
    --rule-arns "$FIRST_DEPLOY_LISTENER_RULE_ARN" --output json)
  jq -e --argjson actions "$FIRST_DEPLOY_DISABLED_ACTIONS" \
    '(.Rules | length) == 1 and .Rules[0].Actions == $actions' <<<"$RULE_JSON" >/dev/null
  unset RULE_JSON
  aws_capture aws ecs update-service \
    --cluster "$ECS_CLUSTER" \
    --service "$ECS_SERVICE" \
    --desired-count 0 \
    --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
    >/dev/null
  aws_waiter_capture aws ecs wait services-stable \
    --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
  RUNNING_TASKS_JSON=$(aws_capture aws ecs list-tasks --cluster "$ECS_CLUSTER" \
    --service-name "$ECS_SERVICE" --desired-status RUNNING --output json)
  PENDING_TASKS_JSON=$(aws_capture aws ecs list-tasks --cluster "$ECS_CLUSTER" \
    --service-name "$ECS_SERVICE" --desired-status PENDING --output json)
  jq -e '.taskArns | length == 0' <<<"$RUNNING_TASKS_JSON" >/dev/null
  jq -e '.taskArns | length == 0' <<<"$PENDING_TASKS_JSON" >/dev/null
  TARGET_HEALTH_JSON=$(aws_capture aws elbv2 describe-target-health \
    --target-group-arn "$TARGET_GROUP_ARN" --output json)
  jq -e 'all(.TargetHealthDescriptions[]; .TargetHealth.State == "draining")' \
    <<<"$TARGET_HEALTH_JSON" >/dev/null
  test "$(curl --cacert "$ACCEPTANCE_TLS_CA_BUNDLE" --connect-timeout 5 \
    --max-time 10 --max-redirs 0 --max-filesize 1048576 -sS -o /dev/null \
    -w '%{http_code}' "$ALB_BASE_URL/api/health")" = 503
  unset RUNNING_TASKS_JSON PENDING_TASKS_JSON TARGET_HEALTH_JSON

  # Prove the narrow first-recovery transition with the same immutable image
  # after the first-deploy recovery rehearsal left traffic disabled and zero.
  require_compatibility_record_current
  DEPLOYMENT_MODE=first-recovery
  aws_capture aws ecs update-service \
    --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
    --task-definition "$CANDIDATE_TASK_DEFINITION" --desired-count 1 \
    --force-new-deployment \
    --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' \
    >/dev/null
  aws_waiter_capture aws ecs wait services-stable \
    --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" >/dev/null
  verify_candidate_target_mapping
  test "$CANDIDATE_TASK_ARN" != "$PRE_FIRST_RECOVERY_TASK_ARN"
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-s3
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-bedrock-guardrails
  aws_capture aws elbv2 modify-rule --rule-arn "$FIRST_DEPLOY_LISTENER_RULE_ARN" \
    --actions "$FIRST_DEPLOY_FORWARD_ACTIONS" >/dev/null
  verify_public_probes
  SSL_CERT_FILE="$ACCEPTANCE_TLS_CA_BUNDLE" \
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance verify-api \
      --state-file "$ACCEPTANCE_STATE"
  run_candidate_role_check "$CANDIDATE_TASK_ARN" verify-operator-telemetry positive
fi
```

The drain half requires zero running/pending tasks and no non-draining
registered targets. The recovery half must create a distinct candidate task,
restore only the approved listener action, and read back the same persisted
Landscape/session state. Keep failed task-definition, stopped-task, event, and
sanitized log receipts until the incident record is complete.

## Disposable acceptance cleanup

Before either Terraform scenario is created, record infrastructure owner,
identity owner for the Cognito test identity, sanitized evidence destination, non-secret UUID
`ACCEPTANCE_RUN_ID`, and a teardown deadline no later than four hours after
the live gate. Tag every disposable resource. Promotion is forbidden before Plan 12 final GO
and while cleanup is pending.

On success, failure, interruption, or timeout:

1. Export only approved sanitized evidence.
2. Destroy both Terraform stacks, verify separately bound state is empty, and
   prove the Scenario B-owned Cognito pool/identity is absent.
3. Delete the rollback-baseline ECR tag and candidate acceptance ECR tag.
4. Destroy the shared bootstrap bucket/repository only after both remote
   scenario states are empty.
5. Run the closed terminal `orphan-sweep` across ECS, ALB, Aurora, EFS, Secrets
   Manager, CloudWatch Logs, EventBridge, Cognito, ECR, and Guardrails.
6. Clear `CLEANUP_REQUIRED=0` only after all independent surfaces pass.

```bash
set -o pipefail
load_cleanup_state
test "$(aws_capture aws sts get-caller-identity --query Account --output text)" = "$AWS_ACCOUNT_ID"
: "${EVIDENCE_EXPORT_RECEIPT:?set a new protected initial-export receipt path}"
: "${INITIAL_EVIDENCE_ARTIFACT_COUNT:?set the verified initial export artifact count}"
uv run --frozen python -m elspeth.web.aws_ecs_acceptance evidence-export-receipt \
  --file "$CONTROL_MANIFEST" --ledger "$GATE_LEDGER" \
  --output "$EVIDENCE_EXPORT_RECEIPT" \
  --artifact-count "$INITIAL_EVIDENCE_ARTIFACT_COUNT" >/dev/null
bind_initial_evidence_export "$EVIDENCE_EXPORT_RECEIPT"
cleanup_failures=()
SCENARIO_B_COGNITO_POOL_ID=$(jq -er '.values.COGNITO_USER_POOL_ID // ""' "$SCENARIO_B_INVENTORY")
SCENARIO_B_COGNITO_POOL_OWNED=$(jq -r '.orphan_sweep.cognito_pool_owned' "$SCENARIO_B_INVENTORY")
case "$SCENARIO_B_COGNITO_POOL_OWNED" in true|false) ;; *) exit 1 ;; esac

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
      -var="scenario_id=$scenario_id" -var="aws_account_id=$AWS_ACCOUNT_ID" \
      -var="candidate_image=$CANDIDATE_IMAGE" \
      -var="rollback_baseline_image=$ROLLBACK_BASELINE_IMAGE" -out="$plan" >/dev/null
    chmod 600 "$plan"
    terraform_capture -chdir="$directory" show -json "$plan" | \
      uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
        sanitize-evidence --kind terraform-destroy-plan >"$receipt"
    chmod 600 "$receipt"
    plan_sha="$(sha256sum "$plan" | awk '{print $1}')"
    receipt_hash="$(persist_sanitized_receipt "$scenario_id" terraform-destroy-plan "$plan_sha" "$receipt")"
    request_signed_tf_approval "$scenario_id" terraform-destroy-plan "$receipt_hash" \
      "$receipt" "$approval_file"
    approval_hash="$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
      approval-verify --file "$CONTROL_MANIFEST" --scenario-id "$scenario_id" \
      --kind terraform-destroy-plan --plan-receipt-hash "$receipt_hash" \
      --approval-file "$approval_file")"
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance approval-require-current \
      --file "$CONTROL_MANIFEST" --scenario-id "$scenario_id" \
      --kind terraform-destroy-plan --plan-receipt-hash "$receipt_hash" \
      --approval-hash "$approval_hash"
    test "$(sha256sum "$plan" | awk '{print $1}')" = "$plan_sha"
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
  local label="$1" tag="$2" surface="ecr_$1" repositories listing count result
  if (
    set -Eeuo pipefail
    repositories=$(aws_capture aws ecr describe-repositories --region "$AWS_REGION" \
      --query "length(repositories[?repositoryName=='$ECR_REPOSITORY'])" --output text)
    test "$repositories" = 0 || test "$repositories" = 1
    if test "$repositories" = 0; then
      exit 0
    fi
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

destroy_shared_bootstrap() {
  if (
    set -Eeuo pipefail
    if test -e "$BOOTSTRAP_STATE"; then
      state_before=$(terraform_capture -chdir="$BOOTSTRAP_TF_DIR" state list)
      if test -n "$state_before"; then
        plan="$BOOTSTRAP_TF_DIR/destroy.tfplan"
        receipt="$BOOTSTRAP_TF_DIR/destroy.receipt.json"
        trap 'rm -f -- "$plan" "$receipt"' EXIT
        terraform_capture -chdir="$BOOTSTRAP_TF_DIR" plan -destroy -input=false \
          -var="acceptance_run_id=$ACCEPTANCE_RUN_ID" \
          -var="aws_account_id=$AWS_ACCOUNT_ID" \
          -var="aws_region=$AWS_REGION" \
          -var="backend_state_bucket=$BACKEND_STATE_BUCKET" \
          -var="ecr_repository=$ECR_REPOSITORY" -out="$plan" >/dev/null
        chmod 600 "$plan"
        terraform_capture -chdir="$BOOTSTRAP_TF_DIR" show -json "$plan" | \
          uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
            sanitize-evidence --kind terraform-destroy-plan >"$receipt"
        chmod 600 "$receipt"
        plan_sha=$(sha256sum "$plan" | awk '{print $1}')
        receipt_hash=$(persist_sanitized_receipt bootstrap terraform-destroy-plan "$plan_sha" "$receipt")
        request_signed_tf_approval bootstrap terraform-destroy-plan "$receipt_hash" \
          "$receipt" "${BOOTSTRAP_DESTROY_APPROVAL_FILE:?set bootstrap destroy approval file}"
        approval_hash=$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
          approval-verify --file "$CONTROL_MANIFEST" --scenario-id bootstrap \
          --kind terraform-destroy-plan --plan-receipt-hash "$receipt_hash" \
          --approval-file "$BOOTSTRAP_DESTROY_APPROVAL_FILE")
        uv run --frozen python -m elspeth.web.aws_ecs_acceptance approval-require-current \
          --file "$CONTROL_MANIFEST" --scenario-id bootstrap --kind terraform-destroy-plan \
          --plan-receipt-hash "$receipt_hash" --approval-hash "$approval_hash"
        test "$(sha256sum "$plan" | awk '{print $1}')" = "$plan_sha"
        terraform_capture -chdir="$BOOTSTRAP_TF_DIR" apply -input=false "$plan" >/dev/null
      fi
      test -z "$(terraform_capture -chdir="$BOOTSTRAP_TF_DIR" state list)"
    else
      # Recovery for interruption before the first local-state write. At this
      # point no scenario backend could have been initialized, so the planned
      # bucket must be empty.
      if test "$(aws_capture aws s3api list-buckets \
          --query "length(Buckets[?Name=='$BACKEND_STATE_BUCKET'])" --output text)" = 1; then
        aws_capture aws s3api delete-bucket --bucket "$BACKEND_STATE_BUCKET" \
          --region "$AWS_REGION" >/dev/null
      fi
      if test "$(aws_capture aws ecr describe-repositories --region "$AWS_REGION" \
          --query "length(repositories[?repositoryName=='$ECR_REPOSITORY'])" --output text)" = 1; then
        aws_capture aws ecr delete-repository --region "$AWS_REGION" \
          --repository-name "$ECR_REPOSITORY" --force >/dev/null
      fi
    fi
    test "$(aws_capture aws s3api list-buckets \
      --query "length(Buckets[?Name=='$BACKEND_STATE_BUCKET'])" --output text)" = 0
    test "$(aws_capture aws ecr describe-repositories --region "$AWS_REGION" \
      --query "length(repositories[?repositoryName=='$ECR_REPOSITORY'])" --output text)" = 0
  ); then
    checkpoint_cleanup shared_resource_cleanup confirmed
  else
    checkpoint_cleanup shared_resource_cleanup failed || true
    return 1
  fi
}

scenario_a_destroyed=0
scenario_b_destroyed=0
EARLY_BOOTSTRAP_ONLY=$(jq -r '
  (.ecr.baseline_digest == null) and (.ecr.candidate_digest == null)
  and (.scenarios.A.terraform_applied == false)
  and (.scenarios.B.terraform_applied == false)
' "$CONTROL_MANIFEST")
case "$EARLY_BOOTSTRAP_ONLY" in true|false) ;; *) exit 1 ;; esac
if test "$EARLY_BOOTSTRAP_ONLY" = true; then
  # Application mutation is impossible before bootstrap and image publication.
  checkpoint_cleanup terraform_scenario_a confirmed
  checkpoint_cleanup terraform_scenario_b confirmed
  scenario_a_destroyed=1
  scenario_b_destroyed=1
else
  if ! destroy_scenario A "$SCENARIO_A_TF_DIR" "$SCENARIO_A_TF_VARS" \
    "$SCENARIO_A_TF_BINDING_SHA" "$SCENARIO_A_TF_BINDING_FILE" \
    "$SCENARIO_A_DESTROY_APPROVAL_FILE" terraform_scenario_a; then
    cleanup_failures+=(terraform_scenario_a)
  else
    scenario_a_destroyed=1
  fi
  if ! destroy_scenario B "$SCENARIO_B_TF_DIR" "$SCENARIO_B_TF_VARS" \
    "$SCENARIO_B_TF_BINDING_SHA" "$SCENARIO_B_TF_BINDING_FILE" \
    "$SCENARIO_B_DESTROY_APPROVAL_FILE" terraform_scenario_b; then
    cleanup_failures+=(terraform_scenario_b)
  else
    scenario_b_destroyed=1
  fi
fi
unset EARLY_BOOTSTRAP_ONLY

if test "$scenario_b_destroyed" = 1 && (
  set -Eeuo pipefail
  if test -n "$SCENARIO_B_COGNITO_POOL_ID"; then
    test "$SCENARIO_B_COGNITO_POOL_OWNED" = true
    test "$(aws_capture aws cognito-idp list-user-pools --region "$AWS_REGION" \
      --max-results 60 --query \
      "length(UserPools[?Id=='$SCENARIO_B_COGNITO_POOL_ID'])" --output text)" = 0
  else
    test "$SCENARIO_B_COGNITO_POOL_OWNED" = false
  fi
); then
  # Scenario B owns the disposable pool. Its successful Terraform destroy and
  # this independent absence query delete the user and invalidate all sessions.
  checkpoint_cleanup identity_cleanup confirmed
else
  checkpoint_cleanup identity_cleanup failed || true
  cleanup_failures+=(identity_cleanup)
fi

if ! delete_ecr_tag baseline "$ROLLBACK_BASELINE_TAG"; then
  cleanup_failures+=(ecr_baseline)
fi
if ! delete_ecr_tag candidate "$CANDIDATE_TAG"; then
  cleanup_failures+=(ecr_candidate)
fi
if test "$scenario_a_destroyed" = 1 && test "$scenario_b_destroyed" = 1; then
  destroy_shared_bootstrap || cleanup_failures+=(shared_resource_cleanup)
else
  checkpoint_cleanup shared_resource_cleanup failed || true
  cleanup_failures+=(shared_resource_cleanup)
fi

ORPHAN_RECEIPT_DIR=$(dirname -- "$SANITIZED_ORPHAN_RECEIPT")
test ! -e "$SANITIZED_ORPHAN_RECEIPT" || {
  test ! -L "$SANITIZED_ORPHAN_RECEIPT" && test -f "$SANITIZED_ORPHAN_RECEIPT"
  test "$(stat -c %u "$SANITIZED_ORPHAN_RECEIPT")" = "$(id -u)"
  test "$(stat -c %a "$SANITIZED_ORPHAN_RECEIPT")" = 600
}
ORPHAN_RECEIPT_TMP=$(mktemp -p "$ORPHAN_RECEIPT_DIR" .elspeth-orphan.XXXXXX)
chmod 600 "$ORPHAN_RECEIPT_TMP"
if run_orphan_sweep >"$ORPHAN_RECEIPT_TMP" && \
  mv -fT -- "$ORPHAN_RECEIPT_TMP" "$SANITIZED_ORPHAN_RECEIPT"; then
  checkpoint_cleanup orphan_sweep confirmed
else
  rm -f -- "$ORPHAN_RECEIPT_TMP"
  checkpoint_cleanup orphan_sweep failed || true
  cleanup_failures+=(orphan_sweep)
fi
unset ORPHAN_RECEIPT_DIR ORPHAN_RECEIPT_TMP

if test "${#cleanup_failures[@]}" -ne 0; then
  printf 'cleanup failures: %s\n' "${cleanup_failures[*]}" >&2
  exit 1
fi
```

In this fresh-account topology Scenario B owns the complete disposable Cognito
pool. Destroying that stack deletes the test identity and invalidates its
sessions; the independent paginated absence query above is the identity
cleanup proof. A deployment that reuses an approved pool must instead delete
the disposable user, invalidate its sessions, validate the owner receipt, and
checkpoint that action before destroying Scenario B. The fresh-account shared
bucket and ECR repository are removed by `destroy_shared_bootstrap` after both
scenario states and image tags are gone and before the terminal orphan sweep.
Failed or timed-out actions are checkpointed `failed`; they do not prevent
other independent cleanup attempts unless a later action depends on them.

The evidence owner binds an initial export before deletion and a distinct
final export after the cleanup receipts exist. Plan 12 then prepares final
cleanup evidence, removes local images and temporary evidence through
`remove_local_acceptance_images` and `remove_local_acceptance_evidence`,
checkpoints those operations, and commits the finalizer. No checkpoint may be
marked `confirmed` until its real operation and absence/recovery check pass.

```bash
: "${FINAL_EVIDENCE_EXPORT_RECEIPT:?set a distinct protected final-export receipt path}"
: "${FINAL_EVIDENCE_ARTIFACT_COUNT:?set the verified final export artifact count}"
uv run --frozen python -m elspeth.web.aws_ecs_acceptance evidence-export-receipt \
  --file "$CONTROL_MANIFEST" --ledger "$GATE_LEDGER" \
  --output "$FINAL_EVIDENCE_EXPORT_RECEIPT" \
  --artifact-count "$FINAL_EVIDENCE_ARTIFACT_COUNT" >/dev/null
bind_final_evidence_export "$FINAL_EVIDENCE_EXPORT_RECEIPT"
checkpoint_cleanup evidence_export confirmed
finalize_cleanup_evidence prepare
checkpoint_cleanup final_evidence_prepare confirmed
remove_local_acceptance_images
checkpoint_cleanup local_images confirmed
remove_local_acceptance_evidence
checkpoint_cleanup local_evidence confirmed
if uv run --frozen python -c 'from datetime import UTC, datetime; import os; assert datetime.now(UTC) <= datetime.fromisoformat(os.environ["ACCEPTANCE_TEARDOWN_DEADLINE_UTC"].replace("Z", "+00:00"))'; then
  checkpoint_cleanup teardown_deadline confirmed
else
  # Record the terminal verdict and bounded three-hour emergency window if the
  # original deadline expired during cleanup; resource removal still finishes.
  if test -z "${EMERGENCY_CLEANUP_DEADLINE_UTC:-}"; then
    EMERGENCY_CLEANUP_DEADLINE_UTC=$(date -u -d '+3 hours' +%Y-%m-%dT%H:%M:%SZ)
    export EMERGENCY_CLEANUP_DEADLINE_UTC
    uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update \
      --file "$CONTROL_MANIFEST" --verdict-failure teardown_deadline \
      --emergency-cleanup-deadline-utc "$EMERGENCY_CLEANUP_DEADLINE_UTC" \
      --cleanup-escalation teardown_deadline
  fi
  checkpoint_cleanup teardown_deadline failed
fi
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
