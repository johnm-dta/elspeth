# AWS ECS Runtime Readiness Integration Closeout Plan

> **Execution rule:** run this plan in order from
> `/home/john/elspeth/.worktrees/aws-ecs-program` on
> `feat/aws-ecs-program`. This is the final AWS ECS program plan. Completing
> Task 7 completes the program; there is no additional acceptance layer.

**Goal:** Deliver the AWS ECS integration for one exact 0.7.1 candidate in the
disposable AWS environment, clean up everything created, then fast-forward
that exact SHA into `release/0.7.1` and issue the final GO. Checks and evidence
exist only to support safe deployment, operation, cleanup, and release.

**Owner:** one integration coordinator owns the candidate, the phase order,
the interruption-safe control manifest, cleanup, and final release transition.
Infrastructure, database, identity, evidence, cleanup, and release operators
act only on their approved surfaces.

## Required order

1. Freeze the candidate.
2. Run static and contract gates.
3. Run PostgreSQL, the complete suite, and coverage.
4. Build and inspect the production image.
5. Run live AWS acceptance.
6. Export evidence and clean up.
7. Fast-forward the exact candidate and close the program.

Tasks 1–4 are local and make no external changes. Any repository change after
Task 1 invalidates the candidate and restarts Task 1. Task 5 begins external
mutation. From that point, every success, failure, timeout, signal, or host
loss routes through Task 6 before the candidate may change or another attempt
may begin. Evidence from different candidate SHAs must never be combined.

Wardline and hosted CI are not part of this project closeout. Do not run
Wardline, push a candidate branch, create a PR or temporary ref, or query
hosted workflow results.

The gate ledger is a six-phase resume checkpoint, not a second test system. It
contains one outcome each for `candidate`, `static`, `tests`, `image`, `live`,
and `cleanup`. Detailed operational evidence belongs to the deployment
runbook, control manifest, and sanitized evidence store.

## Hard stops

- Use `uv run` for Python tooling.
- Do not skip, deselect, xfail, lower a threshold, add a waiver, or treat a
  non-zero command as a pass.
- A required operator, credential, browser, Docker, Terraform, database, or
  AWS facility that is unavailable is blocked, not passed.
- Never put credentials, secret values, raw provider responses, prompts,
  content, URLs, ARNs, account identifiers, or raw exceptions into the ledger,
  Filigree, or retained evidence.
- The control manifest must exist before the first registry push or Terraform
  apply and must be updated after every external mutation.
- Do not update `release/0.7.1`, create `v0.7.1`, or publish a durable image
  before Task 7.

---

### Task 1: Freeze the exact candidate

- [ ] Verify the worktree, release anchor, prerequisite closure, version, lock,
  and candidate SHA:

  ```bash
  set -Eeuo pipefail
  umask 077
  test "$PWD" = "/home/john/elspeth/.worktrees/aws-ecs-program"
  test "$(git branch --show-current)" = "feat/aws-ecs-program"
  test -z "$(git status --porcelain)"
  git diff --check
  test -n "${PROGRAM_BASE_SHA:?restore the recorded program base}"
  test -n "${RECONCILED_RELEASE_SHA:?restore the Stage 9 release anchor}"
  git merge-base --is-ancestor "$PROGRAM_BASE_SHA" HEAD
  git merge-base --is-ancestor "$RECONCILED_RELEASE_SHA" HEAD
  test "$(git rev-parse release/0.7.1)" = "$RECONCILED_RELEASE_SHA"

  filigree plan elspeth-6343920a47 --json --detail full > /tmp/aws-ecs-plan12-start.json
  uv run --frozen python - <<'PY'
  import json

  payload = json.load(open("/tmp/aws-ecs-plan12-start.json", encoding="utf-8"))
  steps = [step for phase in payload["phases"] for step in phase["steps"]]
  plan12 = "elspeth-05396fed38"
  incomplete = [step["issue_id"] for step in steps if step["issue_id"] != plan12 and step["status_category"] != "done"]
  assert not incomplete, f"incomplete prerequisites: {incomplete}"
  PY

  test "$(uv run --frozen python -c 'from importlib.metadata import version; print(version("elspeth"))')" = "0.7.1"
  test "$(uv run --frozen elspeth --version)" = "elspeth version 0.7.1"
  test "$(uv run --frozen python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "3.13"
  uv lock --check
  uv sync --frozen --all-extras --dev

  export CANDIDATE_SHA="$(git rev-parse HEAD)"
  test "$(git rev-parse HEAD^{tree})" = "$(git write-tree)"
  ```

- [ ] Initialize a fresh mode-0600 phase ledger outside the repository and bind
  it to the candidate. If an earlier attempt entered cleanup, preserve it and
  use a new path:

  ```bash
  export GATE_LEDGER="${GATE_LEDGER:?set a fresh protected ledger path}"
  export PLAN12_SHA256="$(sha256sum docs/superpowers/plans/aws/2026-07-08-aws-ecs-12-integration-closeout.md | awk '{print $1}')"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger init \
    --file "$GATE_LEDGER" \
    --branch feat/aws-ecs-program \
    --starting-sha "$CANDIDATE_SHA" \
    --plan-sha256 "$PLAN12_SHA256" \
    --program-base-sha "$PROGRAM_BASE_SHA" \
    --reconciled-release-sha "$RECONCILED_RELEASE_SHA"

  uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger record \
    --file "$GATE_LEDGER" --check-id candidate --candidate-sha "$CANDIDATE_SHA" \
    --exit-status 0 \
    --receipt-hash "$(printf 'candidate\0%s\0complete' "$CANDIDATE_SHA" | sha256sum | awk '{print $1}')"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger bind-candidate \
    --file "$GATE_LEDGER" --candidate-sha "$CANDIDATE_SHA"
  ```

**Outcome:** one clean 0.7.1 SHA is frozen; all later work refers to that SHA.

---

### Task 2: Run static and contract gates

- [ ] Run the local enforcing commands:

  ```bash
  test "$(git rev-parse HEAD)" = "$CANDIDATE_SHA"
  test -z "$(git status --porcelain)"

  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  uv run python scripts/cicd/check_slot_type_cross_language.py
  uv run python scripts/cicd/generate_skill_inventory.py --check
  uv run python scripts/check_contracts.py
  PYTHONPATH=elspeth-lints/src uv run python scripts/cicd/parity_harness.py \
    --manifest config/cicd/lint_migration_status.yaml --root .

  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules plugin_contract.options_metadata --root .
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules plugin_contract.component_type,plugin_contract.plugin_hashes --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules immutability.freeze_guards,immutability.frozen_annotations --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules audit_evidence.nominal_base,audit_evidence.tier_1_decoration,audit_evidence.guard_symmetry,audit_evidence.gve_attribution --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules 'composer/*' --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules 'contract_invariants/*' --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules contract_invariants.session_engine_factory --root .
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules manifest.contract_manifest --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules manifest.symbol_inventory,manifest.test_to_source_mapping --root .
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules meta.no-new-bespoke-cicd-enforcer --root .
  uv run python scripts/cicd/enforce_adapter_budget.py

  test -n "${ELSPETH_JUDGE_METADATA_HMAC_KEY:-}"
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=required PYTHONPATH=elspeth-lints/src \
    uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=required PYTHONPATH=elspeth-lints/src \
    uv run python -m elspeth_lints.core.cli check --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier --root src/elspeth

  test -z "$(git status --porcelain)"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger record \
    --file "$GATE_LEDGER" --check-id static --candidate-sha "$CANDIDATE_SHA" \
    --exit-status 0 \
    --receipt-hash "$(printf 'static\0%s\0complete' "$CANDIDATE_SHA" | sha256sum | awk '{print $1}')"
  ```

**Outcome:** formatting, lint, strict typing, generated contracts, repository
invariants, and signed trust checks pass on the frozen candidate.

---

### Task 3: Run PostgreSQL, the complete suite, and coverage

- [ ] Run the load-bearing PostgreSQL tests first, with zero skips:

  ```bash
  docker info
  uv run pytest \
    tests/testcontainer/web/test_schema_probe_postgres.py \
    tests/testcontainer/web/test_doctor_aws_ecs_postgres.py \
    tests/testcontainer/web/test_aws_ecs_validate_only_startup.py \
    tests/testcontainer/web/test_aws_ecs_readiness_postgres.py \
    tests/testcontainer/web/test_landscape_write_gate_postgres.py \
    -m testcontainer -q
  ```

- [ ] Run the complete suite once, then the coverage lane and subsystem floors:

  ```bash
  uv run pytest tests/ -v -m ""
  uv run pytest tests/ --cov=src/elspeth --cov-report=xml --cov-report=term-missing \
    --cov-fail-under=85 -v -m "not slow and not stress and not performance and not testcontainer"
  uv run coverage report --include="src/elspeth/core/landscape/*" --fail-under=92
  uv run coverage report --include="src/elspeth/core/canonical.py" --fail-under=99
  uv run coverage report --include="src/elspeth/engine/orchestrator/*" --fail-under=90
  uv run coverage report --include="src/elspeth/contracts/*" --fail-under=62

  test "$(git rev-parse HEAD)" = "$CANDIDATE_SHA"
  test -z "$(git status --porcelain)"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger record \
    --file "$GATE_LEDGER" --check-id tests --candidate-sha "$CANDIDATE_SHA" \
    --exit-status 0 \
    --receipt-hash "$(printf 'tests\0%s\0complete' "$CANDIDATE_SHA" | sha256sum | awk '{print $1}')"
  ```

**Outcome:** the five PostgreSQL files, the complete repository suite, overall
coverage, and all four subsystem floors pass on the frozen candidate.

---

### Task 4: Build and inspect the production image

- [ ] Build the production-extras image for the approved platform and verify
  its source identity:

  ```bash
  export TARGET_PLATFORM="${TARGET_PLATFORM:?set linux/amd64 or linux/arm64}"
  test "$TARGET_PLATFORM" = "linux/amd64" || test "$TARGET_PLATFORM" = "linux/arm64"
  docker buildx build --platform "$TARGET_PLATFORM" --load \
    --build-arg INSTALL_EXTRAS="webui llm aws postgres" \
    --label "org.opencontainers.image.revision=$CANDIDATE_SHA" \
    -t elspeth:ecs-0.7.1-closeout .
  test "$(docker image inspect elspeth:ecs-0.7.1-closeout --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')" = "$CANDIDATE_SHA"
  test "$(docker image inspect elspeth:ecs-0.7.1-closeout --format '{{.Os}}/{{.Architecture}}')" = "$TARGET_PLATFORM"
  ```

- [ ] Verify the runtime contents and identity:

  ```bash
  test "$(docker run --rm elspeth:ecs-0.7.1-closeout --version)" = "elspeth version 0.7.1"
  docker run --rm --entrypoint python elspeth:ecs-0.7.1-closeout \
    -c "import boto3, botocore, ijson, jinja2, psycopg, litellm, fastapi"
  docker run --rm --entrypoint python elspeth:ecs-0.7.1-closeout \
    -c "import importlib.util as u,sys; missing=[m for m in ('testcontainers','pytest','mypy','ruff') if u.find_spec(m) is None]; sys.exit(0 if len(missing)==4 else 1)"
  docker run --rm --entrypoint sh elspeth:ecs-0.7.1-closeout \
    -c 'test "$(id -u)" = 1000 && test "$(id -g)" = 1000'

  ! grep -iE "AWS_ACCESS_KEY|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN" Dockerfile
  HISTORY_FILE="$(mktemp -p /tmp elspeth-history.XXXXXX)"
  chmod 600 "$HISTORY_FILE"
  trap 'rm -f -- "$HISTORY_FILE"' EXIT
  docker history --no-trunc elspeth:ecs-0.7.1-closeout >"$HISTORY_FILE"
  ! grep -iE "AWS_ACCESS_KEY|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN" "$HISTORY_FILE"
  rm -f -- "$HISTORY_FILE"
  trap - EXIT

  test "$(git rev-parse HEAD)" = "$CANDIDATE_SHA"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger record \
    --file "$GATE_LEDGER" --check-id image --candidate-sha "$CANDIDATE_SHA" \
    --exit-status 0 \
    --receipt-hash "$(printf 'image\0%s\0complete' "$CANDIDATE_SHA" | sha256sum | awk '{print $1}')"
  ```

**Outcome:** the lean non-root production image runs 0.7.1, contains its
runtime dependencies but no development tools or baked credentials, matches
the approved platform, and is labeled with the frozen SHA.

---

### Task 5: Run live AWS acceptance

This task consumes the approved out-of-repository Terraform stacks and the
deployment runbook at `docs/runbooks/aws-ecs-deployment.md`. The runbook owns
the command wrappers, protected receipt schemas, inventory validation, AWS
capture rules, rollout commands, observation window, rollback, and cleanup
mechanics. Do not duplicate or improvise those commands here.

- [ ] Before mutation, obtain and validate:

  - a usable default AWS credential chain and the approved account/region;
  - distinct Scenario A and B Terraform roots, vars, workspaces, encrypted
    locked remote state, binding receipts, and pre-apply inventories;
  - the Plan 10 rollback-baseline SHA and the approved ECR registry/repository;
  - the evidence destination and teardown deadline;
  - database retention/schema approval;
  - Cognito/OIDC test identity inputs for Scenario B;
  - infrastructure, database, identity, evidence, cleanup, and release owners.

  Missing input or authority is a genuine external blocker. Do not fabricate
  it and do not substitute a mock.

- [ ] Create the control manifest before the first ECR push or Terraform
  apply:

  ```bash
  export AWS_PAGER=""
  export CONTROL_MANIFEST="${CONTROL_MANIFEST:?set a fresh protected manifest path}"
  export ACCEPTANCE_RUN_ID="${ACCEPTANCE_RUN_ID:?set a fresh UUID}"
  export AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?set approved account}"
  export AWS_REGION="${AWS_REGION:?set approved region}"
  export SCENARIO_A_INVENTORY="${SCENARIO_A_INVENTORY:?set Scenario A pre-apply inventory}"
  export SCENARIO_B_INVENTORY="${SCENARIO_B_INVENTORY:?set Scenario B pre-apply inventory}"
  export SCENARIO_A_TF_BINDING_SHA="${SCENARIO_A_TF_BINDING_SHA:?set Scenario A binding hash}"
  export SCENARIO_B_TF_BINDING_SHA="${SCENARIO_B_TF_BINDING_SHA:?set Scenario B binding hash}"
  export EVIDENCE_DESTINATION_SHA256="${EVIDENCE_DESTINATION_SHA256:?set evidence destination hash}"
  export ACCEPTANCE_TEARDOWN_DEADLINE_UTC="${ACCEPTANCE_TEARDOWN_DEADLINE_UTC:?set deadline}"

  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest init \
    --file "$CONTROL_MANIFEST" \
    --acceptance-run-id "$ACCEPTANCE_RUN_ID" \
    --candidate-sha "$CANDIDATE_SHA" \
    --aws-account-id "$AWS_ACCOUNT_ID" \
    --aws-region "$AWS_REGION" \
    --scenario-a-inventory "$SCENARIO_A_INVENTORY" \
    --scenario-b-inventory "$SCENARIO_B_INVENTORY" \
    --scenario-a-tf-binding "$SCENARIO_A_TF_BINDING_SHA" \
    --scenario-b-tf-binding "$SCENARIO_B_TF_BINDING_SHA" \
    --evidence-destination-sha256 "$EVIDENCE_DESTINATION_SHA256" \
    --gate-ledger "$GATE_LEDGER" \
    --teardown-deadline-utc "$ACCEPTANCE_TEARDOWN_DEADLINE_UTC"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest validate \
    --file "$CONTROL_MANIFEST" --acceptance-run-id "$ACCEPTANCE_RUN_ID" \
    --candidate-sha "$CANDIDATE_SHA"
  ```

- [ ] Execute the deployment runbook in this order for both isolated
  scenarios:

  1. validate protected inventory and bindings;
  2. push run-scoped baseline and candidate images and resolve digests;
  3. review and apply the saved Terraform plan, then prove a no-op plan;
  4. bind the resolved inventory;
  5. run doctor and the schema compatibility gate;
  6. deploy exactly one candidate task;
  7. prove S3 and Bedrock/Guardrails through the candidate task role;
  8. prove web health, persistence, payload verification, and local-auth EFS
     behavior;
  9. in Scenario B, prove real-browser Cognito authorization-code + PKCE and
     exact-origin behavior;
  10. prove zero-overlap replacement and first-deploy recovery;
  11. complete the observation window with Landscape-correlated CloudWatch and
      X-Ray evidence and the approved Aurora connection budget;
  12. rehearse both rollback branches without crossing the schema stop.

  Scenario A is local-auth. Scenario B is Cognito/OIDC. Both use the same
  candidate digest and remain isolated by run/scenario tags, Terraform state,
  namespace, database, EFS, service, and evidence identifiers.

- [ ] Bind the final retained-evidence checkpoint, require zero verdict
  failures, and record the live phase:

  ```bash
  RETAINED_EVIDENCE_RECEIPT="$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest get --file "$CONTROL_MANIFEST" --field evidence.retained_evidence_path)"
  test -n "$RETAINED_EVIDENCE_RECEIPT"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance \
    control-manifest bind-retained-evidence --file "$CONTROL_MANIFEST" \
    --receipt "$RETAINED_EVIDENCE_RECEIPT" --require-complete
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest validate \
    --file "$CONTROL_MANIFEST" --acceptance-run-id "$ACCEPTANCE_RUN_ID" \
    --candidate-sha "$CANDIDATE_SHA"
  test "$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest get --file "$CONTROL_MANIFEST" --field verdict_failures)" = "[]"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger record \
    --file "$GATE_LEDGER" --check-id live --candidate-sha "$CANDIDATE_SHA" \
    --exit-status 0 \
    --receipt-hash "$(printf 'live\0%s\0complete' "$CANDIDATE_SHA" | sha256sum | awk '{print $1}')"
  ```

**Outcome:** both scenarios pass deployment, task-role, persistence, identity,
telemetry, connection-budget, and rollback acceptance on the exact image
digest and candidate SHA.

---

### Task 6: Export evidence and clean up

Task 6 is mandatory after any Task 5 mutation, whether Task 5 passed, failed,
timed out, or was interrupted. Resume from `control-manifest load-cleanup`, not
from remembered shell variables. Follow the runbook's “Disposable acceptance
cleanup” section; continue independent cleanup attempts after one fails.

- [ ] Export the available sanitized evidence before deletion, then clean up:

  - both Terraform states using reviewed destroy plans and current approvals;
  - run-scoped ECS services, tasks, task definitions, ALB resources, Aurora,
    EFS/access points, secrets, log groups, dashboards, alarms, X-Ray,
    EventBridge, Guardrails, and Cognito resources;
  - the test identity and sessions;
  - both run-scoped ECR tags and local images;
  - every resource found by the run/scenario tag and explicit inventory sweep.

  Follow the runbook's lifecycle helpers while doing this work. Bind the
  initial export before deletion with
  `bind_initial_evidence_export "$EVIDENCE_EXPORT_RECEIPT"`. After each independent cleanup succeeds,
  checkpoint its actual surface immediately; do not batch-confirm surfaces.
  The required surfaces are `identity_cleanup`, `shared_resource_cleanup`,
  `terraform_scenario_a`, `terraform_scenario_b`, `orphan_sweep`,
  `ecr_baseline`, and `ecr_candidate`.

  Shared resources must be restored to their recorded pre-run state. No ACTIVE
  task-definition revision or unaccounted resource may remain. AWS-managed
  `DELETE_IN_PROGRESS` task-definition revisions require a named owner and
  bounded follow-up receipt.

- [ ] Bind the final evidence export and commit cleanup completion:

  ```bash
  bind_final_evidence_export "$FINAL_EVIDENCE_EXPORT_RECEIPT"
  checkpoint_cleanup evidence_export confirmed
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance cleanup-evidence-finalize \
    --file "$CONTROL_MANIFEST" --ledger "$GATE_LEDGER" --phase prepare
  checkpoint_cleanup final_evidence_prepare confirmed

  # Remove the run-scoped local images and protected temporary evidence only
  # after the final export and prepare step have succeeded.
  remove_local_acceptance_images
  checkpoint_cleanup local_images confirmed
  remove_local_acceptance_evidence
  checkpoint_cleanup local_evidence confirmed
  uv run --frozen python -c 'from datetime import UTC, datetime; import os; assert datetime.now(UTC) <= datetime.fromisoformat(os.environ["ACCEPTANCE_TEARDOWN_DEADLINE_UTC"].replace("Z", "+00:00"))'
  checkpoint_cleanup teardown_deadline confirmed

  uv run --frozen python -m elspeth.web.aws_ecs_acceptance cleanup-evidence-finalize \
    --file "$CONTROL_MANIFEST" --ledger "$GATE_LEDGER" --phase commit \
    --clear-cleanup-required
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest validate \
    --file "$CONTROL_MANIFEST" --cleanup-only --require-cleanup-cleared
  ```

The finalizer writes the single `cleanup` ledger outcome. Cleanup failure is
NO-GO even when live acceptance passed.

**Outcome:** durable sanitized evidence exists, both Terraform states are
empty, every disposable resource is gone or has an explicit bounded AWS
deletion receipt, identities and tags are removed, and `cleanup_required` is
false.

---

### Task 7: Fast-forward the exact candidate and close the program

- [ ] Reconstruct the anchors from the protected ledger and verify the final
  state before any release mutation:

  ```bash
  set -Eeuo pipefail
  export PROGRAM_WORKTREE="/home/john/elspeth/.worktrees/aws-ecs-program"
  RECONCILED_RELEASE_SHA="$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger get --file "$GATE_LEDGER" --field reconciled_release_sha)"
  CANDIDATE_SHA="$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger get --file "$GATE_LEDGER" --field candidate_sha)"
  PROGRAM_BASE_SHA="$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger get --file "$GATE_LEDGER" --field program_base_sha)"

  test "$(git -C "$PROGRAM_WORKTREE" branch --show-current)" = "feat/aws-ecs-program"
  test "$(git -C "$PROGRAM_WORKTREE" rev-parse HEAD)" = "$CANDIDATE_SHA"
  test -z "$(git -C "$PROGRAM_WORKTREE" status --porcelain)"
  test "$(git -C /home/john/elspeth branch --show-current)" = "release/0.7.1"
  test -z "$(git -C /home/john/elspeth status --porcelain)"

  LOCAL_RELEASE_SHA="$(git -C /home/john/elspeth rev-parse release/0.7.1)"
  REMOTE_RELEASE_SHA="$(git ls-remote --heads origin refs/heads/release/0.7.1 | awk '{print $1}')"
  case "$LOCAL_RELEASE_SHA" in
      "$RECONCILED_RELEASE_SHA"|"$CANDIDATE_SHA") ;;
      *) exit 1 ;;
  esac
  case "$REMOTE_RELEASE_SHA" in
      ""|"$RECONCILED_RELEASE_SHA"|"$CANDIDATE_SHA") ;;
      *) exit 1 ;;
  esac

  test -z "$(git tag --list v0.7.1)"
  test -z "$(git ls-remote --tags origin refs/tags/v0.7.1 'refs/tags/v0.7.1^{}')"
  git -C "$PROGRAM_WORKTREE" merge-base --is-ancestor "$PROGRAM_BASE_SHA" "$CANDIDATE_SHA"
  git -C "$PROGRAM_WORKTREE" merge-base --is-ancestor "$RECONCILED_RELEASE_SHA" "$CANDIDATE_SHA"

  uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger finalize \
    --file "$GATE_LEDGER" --candidate-sha "$CANDIDATE_SHA"
  ```

- [ ] Perform the single idempotent fast-forward and verify it:

  ```bash
  if test "$LOCAL_RELEASE_SHA" = "$RECONCILED_RELEASE_SHA"; then
      git -C /home/john/elspeth merge --ff-only "$CANDIDATE_SHA"
  fi
  test "$(git -C /home/john/elspeth rev-parse release/0.7.1)" = "$CANDIDATE_SHA"
  test -z "$(git -C /home/john/elspeth status --porcelain)"

  if test "$REMOTE_RELEASE_SHA" != "$CANDIDATE_SHA"; then
      git -C /home/john/elspeth push origin "$CANDIDATE_SHA:refs/heads/release/0.7.1"
  fi
  test "$(git ls-remote --heads origin refs/heads/release/0.7.1 | awk '{print $1}')" = "$CANDIDATE_SHA"
  ```

- [ ] On GO, add one sanitized Filigree comment containing the candidate SHA,
  image digest, AWS change-record ID, and cleanup receipt; close
  `elspeth-05396fed38` at `release/0.7.1@$CANDIDATE_SHA`. On any failure, leave
  it open with the exact blocker and do not claim partial GO.

**GO means:** the exact SHA, digest, and platform passed local and live runtime
acceptance, cleanup completed, and `release/0.7.1` now points at that SHA. It
does not approve a rebuilt digest or another platform without its own image
and live acceptance.

**Plan complete:** once the release and Filigree assertions above pass, the AWS
ECS runtime-readiness program is complete. Do not add another review, gate,
report, or closeout phase after Task 7.
