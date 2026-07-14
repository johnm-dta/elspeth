# AWS ECS Runtime Readiness Integration Closeout Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to run this plan in order. This is an owned
> integration gate, not a parallel implementation slice. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Produce one exact-version candidate on `feat/aws-ecs-program`, prove
its local and hosted gates plus the design's live Aurora/EFS/ECS/ALB acceptance
criteria, then fast-forward that exact tested SHA into `release/0.7.1` before
issuing runtime GO.

**Architecture:** Plans 01–11, 13–14, and 15A–15C own implementation and scoped tests. This plan
starts only after all of them have landed, installs the exact locked
environment, then runs the repository's whole-program test, static-analysis,
contract, trust-boundary, lean-image, hosted-CI, and live-AWS gates in a fixed
order. Any failure returns ownership to the plan or code surface that caused
it; after repair, the closeout owner restarts this plan from Task 1 so the
final verdict always describes one unchanged commit and its deployed image.

**Tech Stack:** uv, pytest, coverage.py, Ruff, strict mypy, elspeth-lints,
Wardline, Docker, GitHub CLI.

**Depends on:** every implementation task and commit in plans 01–11, 13–14, and 15A–15C. In
particular, plan 02's `postgres` dependency and plan 06's `aws` dependency must
already coexist in one regenerated `uv.lock`; plan 08's endpoint gate must
precede the plan 06/07 web-reachable `aws_s3` registrations; plan 11's
landscape-write guard must be present. Do not start this plan against a branch
that merely has open or partially integrated versions of those plans.

**Owner:** one integration coordinator owns the sequence, durable control
manifest, tracker claim, and final verdict. Infrastructure, database,
identity, release, evidence, and emergency-cleanup operators perform only
their approved surfaces; they do not issue the final go/no-go verdict. The
coordinator confirms each named operator's authority and availability before
Task 1 starts and records the matrix in the protected gate ledger.

**Program branch contract:** Run Tasks 1–8 in the one shared
`/home/john/elspeth/.worktrees/aws-ecs-program` worktree on
`feat/aws-ecs-program`. The orchestration run sheet first merges the
then-current `release/0.7.1` tip into the program branch and records it as
`RECONCILED_RELEASE_SHA`; documentation-only release movement before that
reconciliation is not a blocker. Use the program branch or a
candidate-derived immutable exact-SHA temporary ref for hosted CI. From Task 1
through Task 9, the release ref must remain at `RECONCILED_RELEASE_SHA`; later
movement invalidates the candidate and restarts reconciliation plus Task 1.
If external mutation has begun or `CLEANUP_REQUIRED=1`, Task 8 must first
finish evidence export and every independent cleanup attempt before program
HEAD changes or another reconciliation begins. Cleanup failure is NO-GO and
forbids restart; evidence from the invalidated candidate must not be mixed
into the next run.
After Tasks 1–8 pass on one unchanged candidate and cleanup completes, Task 9
fast-forwards the reconciled release branch to that exact SHA without creating
a new merge commit, verifies all slice anchors in release ancestry, and only
then may issue GO and close Plan 12.

## Hard-stop rules

- Run Tasks 1–8 from the program worktree with `feat/aws-ecs-program` checked
  out. Run only Task 9's final release-boundary commands from the release
  checkout after verifying its exact branch, anchor, and cleanliness.
- Execute every Task 1–7 command block in a Bash shell already configured with
  `set -Eeuo pipefail` and `umask 077` (`AWS_PAGER=""` for AWS work). Task 8 is
  the deliberate exception: it keeps `pipefail` but collects failures instead
  of using `errexit` so every independent cleanup surface is attempted.
- Use `uv run` for Python tools. Bare `pytest`, `ruff`, or `mypy` may select a
  different interpreter and is not evidence.
- Do not use `|| true`, deselect a failing test, lower a threshold, add a
  baseline/waiver, or treat a non-zero result as a warning to complete this
  plan.
- A dependency-gated or environment-gated command that cannot run is
  **blocked**, not passed. Obtain the required environment or operator action
  before continuing.
- Before Task 7 creates an acceptance resource or registry tag, a command
  failure stops the run: fix the owning surface, commit, and restart at Task 1.
  After any such external mutation, set `CLEANUP_REQUIRED=1`; every failure or
  interruption skips remaining acceptance work and routes to Task 8 first.
  Task 8 exports available sanitized evidence, attempts every independent
  cleanup surface, aggregates failures, and returns NO-GO. Only after Task 8
  finishes may the owner repair and restart at Task 1. Results from different
  commits may never be combined into a go verdict.
- Release movement after reconciliation invalidates the candidate. Before any
  external mutation, return directly to the run sheet's reconciliation
  protocol. After external mutation begins or while `CLEANUP_REQUIRED=1`, do
  not change program HEAD or reconcile again until Task 8 has completed
  evidence export and every independent cleanup attempt. Cleanup failure is
  NO-GO and forbids restart.
- Do not push or update `release/0.7.1` before Task 9's final fast-forward.
  Do not create a VCS release tag, promote the candidate image beyond the
  acceptance environment, or call the AWS ECS program complete until Task 9
  returns GO. Task 7's candidate-specific ECR tag is evidence plumbing, not a
  release tag.

---

### Task 1: Create and freeze the exact-version integrated candidate

**Files:**

- Modify if still at 0.7.0: `pyproject.toml`
- Modify if still at 0.7.0: `uv.lock`
- Modify if 0.7.1 entry is absent: `CHANGELOG.md`
- Verify: `docs/superpowers/plans/aws/2026-07-08-aws-ecs-00-overview.md`

- [ ] Confirm the closeout is running on the program integration branch and
  starts from a clean tree:

  ```bash
  test "$(git branch --show-current)" = "feat/aws-ecs-program"
  test -z "$(git status --porcelain)"
  git diff --check
  test -n "${RECONCILED_RELEASE_SHA:-}"
  git merge-base --is-ancestor "$RECONCILED_RELEASE_SHA" HEAD
  test "$(git rev-parse release/0.7.1)" = "$RECONCILED_RELEASE_SHA"
  ```

  **Expected result:** every command exits 0. If plan work is still
  uncommitted, the integration owner is on another branch, the reconciliation
  anchor is absent from candidate ancestry, or the release ref moved after
  reconciliation, invalidate the candidate and return to the run sheet's
  reconciliation protocol before restarting Task 1. The main checkout's dirty
  state is irrelevant until Task 9's final update surface.

- [ ] Initialize the owner-approved protected evidence ledger outside the
  repository before the first gate. Its parent directory is mode 0700 on
  durable encrypted storage; the ledger is mode 0600 and starts bound to the
  current branch/starting SHA. Execute every checkbox in Tasks 1–8 through its
  stable check ID and record only timing, exit status, later candidate SHA, and
  sanitized receipt/output hashes—never expanded commands, secrets, or raw
  output:

  ```bash
  export GATE_LEDGER="${GATE_LEDGER:?set approved durable mode-0600 ledger path}"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger init --file "$GATE_LEDGER" --branch feat/aws-ecs-program --starting-sha "$(git rev-parse HEAD)"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger record --file "$GATE_LEDGER" --check-id task1.clean-start --exit-status 0 --receipt-hash "$(printf '%s' clean-start | sha256sum | awk '{print $1}')"
  ```

  **Expected result:** a fresh-process ledger validation succeeds. An existing
  ledger is accepted only through its exact-owner resume validation; never
  truncate or silently replace prior evidence.

- [ ] Before changing the version or lock boundary, prove from the executable
  Filigree graph that every implementation slice is closed, then atomically
  start (or validate an exact-owner resume of) the Plan 12 issue:

  ```bash
  export INTEGRATION_OWNER="${INTEGRATION_OWNER:?set the named integration coordinator}"
  filigree plan elspeth-6343920a47 --json --detail full > /tmp/aws-ecs-plan12-start.json
  uv run --frozen python -c 'import json,sys; p=json.load(open("/tmp/aws-ecs-plan12-start.json")); steps=[s for phase in p["phases"] for s in phase["steps"]]; required={"elspeth-b9e8b5d24b","elspeth-9070fb0a45","elspeth-8166b310e7","elspeth-c0103e6c88","elspeth-e8dc754360","elspeth-7fe6aa531f","elspeth-dffe064287","elspeth-03cf981d4a","elspeth-1a1c31bcce","elspeth-74717426b7","elspeth-a342f333a4","elspeth-397ac915b8","elspeth-6285c29c07","elspeth-25286192ee","elspeth-5e729216f4","elspeth-f5d5dddddf","elspeth-130dc48252","elspeth-0674a06468","elspeth-7d1f35e3d8"}; done={s["issue_id"] for s in steps if s["status_category"] == "done"}; missing=required-done; assert not missing, f"implementation steps not done: {sorted(missing)}"; assert any(s["issue_id"] == "elspeth-05396fed38" for s in steps), "Plan 12 tracker step missing"'
  while read -r anchor; do
      [[ "$anchor" =~ ^feat/aws-ecs-program@[0-9a-f]{40}$ ]]
      CLOSE_SHA="${anchor##*@}"
      git merge-base --is-ancestor "$CLOSE_SHA" HEAD
  done < <(jq -r '.phases[].steps[] | select(.issue_id != "elspeth-05396fed38") | .close_commit' /tmp/aws-ecs-plan12-start.json)
  filigree show elspeth-7d1f35e3d8 --json | jq -e '(.blocked_by | sort) as $deps | ($deps | index("elspeth-0674a06468")) != null and ($deps | index("elspeth-7fe6aa531f")) != null' >/dev/null
  PLAN12_JSON="$(filigree show elspeth-05396fed38 --json)"
  PLAN12_STATUS="$(jq -r .status <<<"$PLAN12_JSON")"
  PLAN12_ASSIGNEE="$(jq -r '.assignee // ""' <<<"$PLAN12_JSON")"
  if test "$PLAN12_STATUS" = "pending" && test -z "$PLAN12_ASSIGNEE"; then
      filigree start-work elspeth-05396fed38 --assignee "$INTEGRATION_OWNER" --actor "$INTEGRATION_OWNER" --commit "feat/aws-ecs-program@$(git rev-parse HEAD)"
  else
      test "$PLAN12_STATUS" = "in_progress"
      test "$PLAN12_ASSIGNEE" = "$INTEGRATION_OWNER"
  fi
  ```

  **Expected result:** all 19 pre-closeout tracker steps (18 application
  slices plus the verification-baseline task) are done, Plan 15C still
  depends directly on Plan 15B and Plan 06, and Plan 12 is
  held in `in_progress` by exactly the integration coordinator. A missing
  dependency stops before any release-candidate commit. This is an execution
  instruction only; plan-review agents must leave the issue pending and
  unclaimed.

- [ ] Own the release version boundary. Before the version commit, add a
  `CHANGELOG.md` 0.7.1 section with the actual candidate date, the AWS ECS
  runtime-readiness deliverables from plans 01–11, 13–14, and 15A–15C, the schema/drop-recreate and
  one-task/downtime constraints, and the operator runbook link. Do not copy the
  implementation-plan review history into public release notes. Then accept
  only an unbumped 0.7.0 tree or
  an already-idempotently-bumped 0.7.1 tree; any other starting version is a
  blocker:

  ```bash
  CURRENT_VERSION="$(uv version --short)"
  test "$CURRENT_VERSION" = "0.7.0" || test "$CURRENT_VERSION" = "0.7.1"
  if test "$CURRENT_VERSION" = "0.7.0"; then
      uv version 0.7.1 --no-sync
  fi
  test "$(uv version --short)" = "0.7.1"
  grep -Eq '^## 0\.7\.1 - [0-9]{4}-[0-9]{2}-[0-9]{2}' CHANGELOG.md
  git diff --check
  if ! git diff --quiet -- pyproject.toml uv.lock CHANGELOG.md; then
      git add pyproject.toml uv.lock CHANGELOG.md
      git commit -m "chore: prepare release 0.7.1"
  fi
  test -z "$(git status --porcelain)"
  ```

  **Expected result:** the tree is clean and `pyproject.toml` plus `uv.lock`
  identify 0.7.1. The version bump is part of the candidate, never a later tag
  or image-label fiction. If a failure later reopens implementation, amend or
  replace the candidate as appropriate and restart this plan from Task 1.

- [ ] Verify that the lockfile represents the serially integrated 02/06/07
  dependency set; do not regenerate it during closeout:

  ```bash
  uv lock --check
  uv sync --frozen --all-extras
  uv run --frozen python -c "import sys; assert sys.version_info[:2] == (3, 13), sys.version"
  test "$(uv run --frozen python -c 'from importlib.metadata import version; print(version("elspeth"))')" = "0.7.1"
  ```

  **Expected result:** all four commands exit 0 without modifying `uv.lock`.
  Python 3.13 is the release and authoritative coverage interpreter in live
  CI; a 3.12-only result does not close this plan.
  `uv sync --frozen --all-extras` is the same dependency installation posture
  used by the live CI test and static-analysis jobs. If either command reports
  lock drift, reopen the owning slice: Plan 02 for PostgreSQL, Plan 06 for the
  initial AWS packages, or Plan 07 for the final Jinja2-extended source+sink
  AWS extra. Reopen the owner and affected descendants, repair and re-execute
  them in dependency order on the current program tip, run `uv lock` under the
  owning slice, commit the regenerated file, and restart this plan. Never
  resolve `uv.lock` by editing conflict markers or choosing one side.

- [ ] Capture the candidate commit for the final evidence record:

  ```bash
  git rev-parse HEAD
  ```

  **Expected result:** one 40-character commit SHA. Every later task must run
  against this SHA unless a failure causes a repair and full restart.
  Bind that SHA into the protected gate ledger; subsequent records reject any
  different worktree HEAD. Any code, documentation, lockfile, or generated-file
  change after this point invalidates every candidate receipt and restarts this
  plan from Task 1; evidence from different SHAs may never be combined.

**Definition of Done:**

- [ ] `feat/aws-ecs-program` is clean, contains `RECONCILED_RELEASE_SHA`, and
  `release/0.7.1` still equals that reconciliation anchor.
- [ ] Project and installed distribution versions are exactly 0.7.1.
- [ ] `CHANGELOG.md` has a dated 0.7.1 release entry covering this program's
  public behavior and operational constraints.
- [ ] The uv environment uses Python 3.13.
- [ ] `uv lock --check` and frozen all-extras sync pass.
- [ ] The candidate SHA is recorded.
- [ ] All implementation slices in plans 01–11, 13–14, and 15A–15C are done in Filigree and
  the Plan 12 closeout step is atomically held by the integration coordinator.

---

### Task 2: Run formatting, lint, strict typing, and repository contract guards

**Files:**

- Verify: `src/`
- Verify: `tests/`
- Verify: `scripts/`
- Verify: `examples/`
- Verify: `elspeth-lints/src/`
- Reference: `.github/workflows/ci.yaml`
- No file changes are expected.

- [ ] Run the three primary static gates exactly as live CI runs them:

  ```bash
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  ```

  **Expected result:** every command exits 0. The mypy command is strict
  because `pyproject.toml` enables `strict = true` for the gated source trees.
  Do not substitute `mypy .`; tests and scripts intentionally have different
  typing scope.

- [ ] Run the CI-owned cross-language, inventory, and settings-to-runtime
  contract scripts:

  ```bash
  uv run python scripts/cicd/check_slot_type_cross_language.py
  uv run python scripts/cicd/generate_skill_inventory.py --check
  uv run python scripts/check_contracts.py
  PYTHONPATH=elspeth-lints/src uv run python scripts/cicd/parity_harness.py --manifest config/cicd/lint_migration_status.yaml --root .
  ```

  **Expected result:** every command exits 0 and the inventory check reports
  no generated-file drift. The parity harness must report no divergence
  between migrated and shadow-mode lints. These checks cover the
  guided/composer and plugin surfaces changed by plans 06–09 as well as the
  deployment settings added by plans 01–05, 11, and 13.

- [ ] Run the directly enforcing elspeth-lints suites from live CI:

  ```bash
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
  ```

  **Expected result:** every command exits 0. The SARIF-emission jobs in CI
  re-run subsets of these rules for reporting; they are not a separate local
  acceptance condition once the enforcing commands above pass.

- [ ] Run the signed trust-tier and trust-boundary gates in required
  verification mode. The operator supplies
  `ELSPETH_JUDGE_METADATA_HMAC_KEY` through the shell environment; never print
  it, copy it into a command line, or add it to a file:

  ```bash
  test -n "${ELSPETH_JUDGE_METADATA_HMAC_KEY:-}"
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=required PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=required PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier --root src/elspeth
  ```

  **Expected result:** all commands exit 0. A missing key blocks closeout. If
  either gate reports signature, fingerprint, scope, or AST-path drift, stop
  and use the repository's operator-owned signed-allowlist diagnosis/repair
  procedure; the integration agent must not sign around the finding. After
  the operator repair lands, restart Task 1.

**Definition of Done:**

- [ ] Ruff check and format-check pass without edits.
- [ ] Strict mypy passes on both CI-gated source trees.
- [ ] Every direct CI contract/inventory guard exits 0.
- [ ] Required-mode trust-tier and trust-boundary verification exits 0.

---

### Task 3: Run the unfiltered test suite and CI coverage floors

**Files:**

- Verify: `tests/`
- Verify: `src/elspeth/`
- Verify: `tests/testcontainer/web/test_schema_probe_postgres.py`
- Verify: `tests/testcontainer/web/test_doctor_aws_ecs_postgres.py`
- Verify: `tests/testcontainer/web/test_aws_ecs_validate_only_startup.py`
- Verify: `tests/testcontainer/web/test_aws_ecs_readiness_postgres.py`
- Verify: `tests/testcontainer/web/test_landscape_write_gate_postgres.py`
- No file changes are expected beyond ignored pytest/coverage artifacts.

- [ ] Run the five load-bearing PostgreSQL files explicitly before the broad
  suite. Docker availability and test execution are acceptance requirements,
  not optional skips:

  ```bash
  docker info
  uv run pytest tests/testcontainer/web/test_schema_probe_postgres.py tests/testcontainer/web/test_doctor_aws_ecs_postgres.py tests/testcontainer/web/test_aws_ecs_validate_only_startup.py tests/testcontainer/web/test_aws_ecs_readiness_postgres.py tests/testcontainer/web/test_landscape_write_gate_postgres.py -m testcontainer -q
  ```

  **Expected result:** `docker info` exits 0 and all five files execute and
  pass with zero skips. Unavailable Docker, a missing file, deselection, or any
  skip is `BLOCKED`; do not continue to the broad suite.

- [ ] Run the complete pytest tree without the per-plan selectors or CI's
  slow/stress/performance/testcontainer exclusions:

  ```bash
  uv run pytest tests/ -v -m ""
  ```

  **Expected result:** pytest exits 0. The explicit empty marker expression
  overrides `pyproject.toml`'s default exclusion of
  slow/stress/performance/testcontainer tests while preserving its
  `--strict-markers` and `--strict-config` options. Test-authored skips for
  unavailable opt-in services are acceptable when they are part of the test's
  declared contract; adding a non-empty `-m`, `-k`, `--ignore`, or environment
  switch to hide a failing test is not. This unfiltered command is the
  integration proof that scoped plan tests cannot provide.

- [ ] Re-run the live Python 3.13 CI coverage lane exactly, then enforce its
  audit-critical subsystem floors:

  ```bash
  uv run pytest tests/ --cov=src/elspeth --cov-report=xml --cov-report=term-missing --cov-fail-under=85 -v -m "not slow and not stress and not performance and not testcontainer"
  uv run coverage report --include="src/elspeth/core/landscape/*" --fail-under=92
  uv run coverage report --include="src/elspeth/core/canonical.py" --fail-under=99
  uv run coverage report --include="src/elspeth/engine/orchestrator/*" --fail-under=90
  uv run coverage report --include="src/elspeth/contracts/*" --fail-under=62
  ```

  **Expected result:** pytest and every coverage report exit 0. Do not lower a
  floor to close this plan.

**Definition of Done:**

- [ ] The unfiltered `tests/` suite passes.
- [ ] All five load-bearing PostgreSQL files pass explicitly with zero skips.
- [ ] The live-CI coverage lane passes at 85% or higher.
- [ ] All four audit-critical coverage floors pass.

---

### Task 4: Run the Wardline trust-boundary gate

**Files:**

- Verify: repository root (`.`), including the S3 source/sink, doctor,
  readiness, deployment-contract, Bedrock-provider, and web validation
  boundaries introduced by plans 01–11, 13–14, and 15A–15C.
- Reference: `AGENTS.md` (Wardline project gate)
- No file changes are expected.

- [ ] Run the mandatory repository scan:

  ```bash
  wardline scan . --fail-on ERROR
  ```

  **Expected result:** exit 0. Exit 1 means the gate found an ERROR; fix the
  finding at the external-input boundary, not at a downstream sink. Exit 2 is
  a Wardline/tooling error and also blocks closeout. Do not baseline or waive
  a new finding merely to complete this plan. After any repair, restart at
  Task 1.

**Definition of Done:**

- [ ] Wardline exits 0 on the same candidate commit used for Tasks 1–3.

---

### Task 5: Build and inspect the lean AWS ECS image

**Files:**

- Verify: `Dockerfile`
- Verify: `pyproject.toml`
- Verify: `uv.lock`
- No file changes are expected.

- [ ] Build the actual production-extras path from plan 10:

  ```bash
  set -Eeuo pipefail
  CANDIDATE_SHA="$(git rev-parse HEAD)"
  TARGET_PLATFORM="${TARGET_PLATFORM:?set approved linux/amd64 or linux/arm64}"
  test "$TARGET_PLATFORM" = "linux/amd64" || test "$TARGET_PLATFORM" = "linux/arm64"
  docker buildx build --platform "$TARGET_PLATFORM" --load --build-arg INSTALL_EXTRAS="webui llm aws postgres" --label "org.opencontainers.image.revision=$CANDIDATE_SHA" -t elspeth:ecs-0.7.1-closeout .
  test "$(docker image inspect elspeth:ecs-0.7.1-closeout --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')" = "$CANDIDATE_SHA"
  IMAGE_OS_ARCH="$(docker image inspect elspeth:ecs-0.7.1-closeout --format '{{.Os}}/{{.Architecture}}')"
  test "$IMAGE_OS_ARCH" = "$TARGET_PLATFORM"
  ```

  **Expected result:** all commands exit 0 and the image is labeled with the
  exact candidate SHA. A successful default `all` image is not a substitute;
  this command must exercise `INSTALL_EXTRAS`.

- [ ] Verify the CLI, production imports, omitted development tools, and
  non-root runtime identity:

  ```bash
  test "$(docker run --rm elspeth:ecs-0.7.1-closeout --version)" = "elspeth version 0.7.1"
  docker run --rm --entrypoint python elspeth:ecs-0.7.1-closeout -c "import boto3, botocore, ijson, jinja2, psycopg, litellm, fastapi"
  docker run --rm --entrypoint python elspeth:ecs-0.7.1-closeout -c "import importlib.util as u,sys; missing=[m for m in ('testcontainers','pytest','mypy','ruff') if u.find_spec(m) is None]; sys.exit(0 if len(missing)==4 else 1)"
  docker run --rm --entrypoint sh elspeth:ecs-0.7.1-closeout -c 'test "$(id -u)" = 1000 && test "$(id -g)" = 1000'
  ```

  **Expected result:** all four commands exit 0; the installed image reports
  exactly `elspeth version 0.7.1`, the production imports succeed, all four
  dev/test modules are absent, and the runtime UID/GID are both 1000.

- [ ] Prove the Dockerfile and image history do not bake AWS credentials:

  ```bash
  set -Eeuo pipefail
  assert_no_aws_credential_marker() {
      local file="$1" status=0
      grep -iE "AWS_ACCESS_KEY|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN" "$file" >/dev/null || status=$?
      if test "$status" = "0"; then return 1; fi
      test "$status" = "1"
  }
  assert_no_aws_credential_marker Dockerfile
  HISTORY_FILE="$(mktemp -p /tmp elspeth-history.XXXXXX)"
  chmod 600 "$HISTORY_FILE"
  trap 'rm -f -- "$HISTORY_FILE"' EXIT
  docker history --no-trunc elspeth:ecs-0.7.1-closeout >"$HISTORY_FILE"
  assert_no_aws_credential_marker "$HISTORY_FILE"
  rm -f -- "$HISTORY_FILE"
  trap - EXIT
  ```

  **Expected result:** both commands exit 0 and print no match. AWS runtime
  credentials come from the ECS task role and Secrets Manager injection, not
  image layers.

**Definition of Done:**

- [ ] The lean `webui llm aws postgres` image builds.
- [ ] Its inspected OS/architecture equals the approved `TARGET_PLATFORM`.
- [ ] The image's OCI revision label equals the Task 1 candidate SHA.
- [ ] The CLI reports exactly 0.7.1 and production dependencies run in that
  image.
- [ ] Development/test tools are absent.
- [ ] The runtime remains non-root.
- [ ] No AWS credential variable is baked into the Dockerfile or image
  history.

---

### Task 6: Require the hosted CI umbrella on the candidate SHA

**Files:**

- Reference: `.github/workflows/ci.yaml`
- No file changes are expected.

- [ ] Publish only the exact program-branch candidate for hosted review and CI.
  Do not push or update `release/0.7.1`, and do not use it as the PR head.

  ~~~bash
  export CANDIDATE_SHA="$(git rev-parse HEAD)"
  export PROGRAM_BRANCH="feat/aws-ecs-program"
  test "$(git branch --show-current)" = "$PROGRAM_BRANCH"
  test "$(git rev-parse release/0.7.1)" = "$RECONCILED_RELEASE_SHA"
  REMOTE_PROGRAM_SHA="$(git ls-remote --heads origin "refs/heads/$PROGRAM_BRANCH" | awk '{print $1}')"
  if test -n "$REMOTE_PROGRAM_SHA"; then
      git merge-base --is-ancestor "$REMOTE_PROGRAM_SHA" "$CANDIDATE_SHA"
  fi
  git push origin "$CANDIDATE_SHA:refs/heads/$PROGRAM_BRANCH"
  test "$(git ls-remote --heads origin "refs/heads/$PROGRAM_BRANCH" | awk '{print $1}')" = "$CANDIDATE_SHA"
  PR_NUMBER="$(gh pr list --state open --base release/0.7.1 --head feat/aws-ecs-program --json number,headRefOid --jq 'map(select(.headRefOid == env.CANDIDATE_SHA)) | first | .number // empty')"
  if test -z "$PR_NUMBER"; then
      PR_URL="$(gh pr create --base release/0.7.1 --head feat/aws-ecs-program --title 'release: AWS ECS 0.7.1 candidate' --body 'Candidate review surface only. Do not merge this PR; Plan 12 Task 9 performs the final exact-SHA fast-forward after Tasks 1-8 pass.')"
      PR_NUMBER="${PR_URL##*/}"
  fi
  test "$(gh pr view "$PR_NUMBER" --json state,baseRefName,headRefName,headRefOid --jq '.state + ":" + .baseRefName + ":" + .headRefName + ":" + .headRefOid')" = "OPEN:release/0.7.1:feat/aws-ecs-program:$CANDIDATE_SHA"
  test "$(git rev-parse release/0.7.1)" = "$RECONCILED_RELEASE_SHA"
  ~~~

  If the coordinator lacks feature-branch push authority, obtain the named
  release operator action. A non-fast-forward remote program branch, missing
  authority, or any release-branch movement is blocked. Hosted exact-SHA proof
  comes from the immutable temporary ref below; the unmerged candidate-review
  PR exists to bind review/control-manifest metadata and its merge ref may not
  substitute.

- [ ] Trigger separate exact-SHA runs through the workflows' existing `RC*`
  push selector. Use a candidate-derived, immutable temporary ref; never
  force-update or reuse a ref that points elsewhere. A push event's `headSha`
  is the pushed tip, so this closes the merge-ref ambiguity without changing
  `ci.yaml` during closeout:

  ```bash
  set -Eeuo pipefail
  export CI_REF="RC0.7.1-${CANDIDATE_SHA:0:12}"
  delete_ci_ref() {
      local remote_sha=""
      remote_sha="$(git ls-remote --heads origin "refs/heads/$CI_REF" | awk '{print $1}')"
      if test -z "$remote_sha"; then return 0; fi
      test "$remote_sha" = "$CANDIDATE_SHA"
      git push origin --delete "$CI_REF"
      test -z "$(git ls-remote --heads origin "refs/heads/$CI_REF")"
  }
  cleanup_ci_ref_on_exit() {
      local status=$?
      trap - EXIT
      if ! delete_ci_ref; then status=1; fi
      exit "$status"
  }
  trap cleanup_ci_ref_on_exit EXIT

  REMOTE_CI_SHA="$(git ls-remote --heads origin "refs/heads/$CI_REF" | awk '{print $1}')"
  if test -n "$REMOTE_CI_SHA"; then
      test "$REMOTE_CI_SHA" = "$CANDIDATE_SHA"
  else
      git push origin "${CANDIDATE_SHA}:refs/heads/$CI_REF"
  fi

  EXPECTED_WORKFLOWS=(ci.yaml codeql.yaml enforce-allowlist-judge-gates.yaml)
  CI_RUN_ID=""
  for workflow in "${EXPECTED_WORKFLOWS[@]}"; do
      workflow_run_id=""
      for _ in $(seq 1 30); do
          workflow_run_id="$(gh run list --workflow "$workflow" --branch "$CI_REF" --event push --commit "$CANDIDATE_SHA" --limit 10 --json databaseId,event,headBranch,headSha --jq 'map(select(.event == "push" and .headBranch == env.CI_REF and .headSha == env.CANDIDATE_SHA)) | sort_by(.databaseId) | last | .databaseId // empty')"
          test -z "$workflow_run_id" || break
          sleep 10
      done
      test -n "$workflow_run_id"
      gh run watch "$workflow_run_id" --exit-status --interval 30
      test "$(gh run view "$workflow_run_id" --json event --jq .event)" = "push"
      test "$(gh run view "$workflow_run_id" --json headBranch --jq .headBranch)" = "$CI_REF"
      test "$(gh run view "$workflow_run_id" --json headSha --jq .headSha)" = "$CANDIDATE_SHA"
      test "$(gh run view "$workflow_run_id" --json status --jq .status)" = "completed"
      test "$(gh run view "$workflow_run_id" --json conclusion --jq .conclusion)" = "success"
      gh run view "$workflow_run_id" --json databaseId,url,event,headBranch,headSha,status,conclusion
      if test "$workflow" = "ci.yaml"; then CI_RUN_ID="$workflow_run_id"; fi
  done
  test -n "$CI_RUN_ID"
  test "$(gh run view "$CI_RUN_ID" --json event --jq .event)" = "push"
  test "$(gh run view "$CI_RUN_ID" --json headBranch --jq .headBranch)" = "$CI_REF"
  test "$(gh run view "$CI_RUN_ID" --json headSha --jq .headSha)" = "$CANDIDATE_SHA"
  test "$(gh run view "$CI_RUN_ID" --json status --jq .status)" = "completed"
  test "$(gh run view "$CI_RUN_ID" --json conclusion --jq .conclusion)" = "success"
  test "$(gh run view "$CI_RUN_ID" --json jobs --jq '[.jobs[] | select(.name == "CI Success" and .conclusion == "success")] | length')" = "1"
  delete_ci_ref
  trap - EXIT
  ```

  Record all three run IDs/URLs plus `CI_REF`, then delete the compare-verified
  temporary ref before leaving Task 6. The EXIT trap performs the same guarded
  deletion on failure, because Task 8 has not yet been armed. An absent
  eligible run, a moved pre-existing ref, failed ref deletion, or a
  non-push/merge-ref run is blocked, not passed.

  **Expected result:** every command exits 0. The `CI Success` job is the
  repository-owned aggregate for static analysis, the Python 3.12/3.13 test
  matrix, dependency/license audit, frontend unit tests and typecheck, and
  Playwright end-to-end tests. Local Tasks 2–5 provide inspectable closeout
  evidence; they do not replace any hosted required lane.

**Definition of Done:**

- [ ] A completed successful `ci.yaml` run exists for the exact candidate SHA.
- [ ] The remote `feat/aws-ecs-program` branch and its unmerged review PR to
  `release/0.7.1` point at the candidate while the release branch remains at
  `RECONCILED_RELEASE_SHA`; the acceptance run was an exact-SHA `push` event on its
  immutable temporary `RC*` ref.
- [ ] Its `CI Success` umbrella job concluded successfully.
- [ ] The exact-SHA CI, CodeQL, and signed-allowlist run IDs/URLs are recorded,
  all succeeded and the temporary RC ref is absent.

---

### Task 7: Prove live AWS runtime and rollback acceptance

**Files:**

- Execute: `docs/runbooks/aws-ecs-deployment.md` (created by plan 10)
- Execute: `python -m elspeth.web.aws_ecs_acceptance` (created by plan 10)
- Execute: `tests/integration/plugins/sources/test_aws_s3_source_live.py` (created by plan 06)
- Execute: `tests/integration/plugins/sinks/test_aws_s3_sink_live.py` (created by plan 07)
- Execute inside the candidate task: `python -m elspeth.web.aws_ecs_acceptance verify-bedrock` (created by plan 10, consuming plan 09)
- Reference: the approved external deployment inventory and change record
- No repository file changes are expected; AWS service changes and evidence
  capture are intentional operator actions within the acceptance environment.

All AWS CLI examples below assume Plan 10's `aws_capture` is defined in the
current shell. Direct `aws ...` invocation is forbidden: scalar queries use
command substitution, JSON assertions use an allowlisted `jq` projection,
mutations/waiters discard successful stdout, and sanitized evidence uses
`sanitize-evidence`; wrapper failures expose only a static class and never raw
AWS stderr. The same rule applies through Task 8 cleanup.

- [ ] Require **two isolated disposable acceptance scenarios** defined by the
  out-of-repository Terraform stack. This checkbox binds and validates the
  stacks but does **not** apply them yet; immutable image digests are resolved
  in the next checkbox and injected into the reviewed saved plans. Both need
  ECS Fargate, Aurora
  Serverless v2 PostgreSQL, a real EFS mount, ALB/IP target group, Secrets
  Manager injection, CloudWatch Logs, the durable EventBridge
  deployment-failure target, ECS Exec, a one-shot local-auth verifier task,
  and controlled downtime. Scenario A is
  a desired-zero first deployment using local auth and fresh Aurora/EFS;
  Scenario B is provisioned desired-zero with its listener fixed at 503, then
  bootstraps the qualified pre-Plan-10 rollback baseline recorded by Plan 10
  Task 0 after its immutable digest and task definition are resolved below.
  That baseline must contain every earlier implementation
  slice, especially Plan 13; an unmodified 0.7.0 release cannot satisfy the
  Cognito contract. Scenario B uses the recommended OIDC posture and an
  approved candidate/baseline schema-compatibility record. Each
  has separate service, database, EFS, listener, target, log, and change-record
  identities. Missing infrastructure, permissions, test identities, operator
  approval, or database-owner approval is **blocked**, never a local-only pass.
  Before any external mutation, prove AWS CLI v2, Session Manager plugin,
  Node 22, npm, and the frontend-lock-owned Playwright/Chromium environment.
  Run `npm ci` in `src/elspeth/web/frontend`, require the locally installed
  Playwright version equals the lockfile version, install Chromium through
  that local binary, then run the credential-free exact-spec
  `test:e2e:oidc -- --list` assertion and require exactly the expected single
  test. Recheck the repository is clean. A root-package Playwright, global
  browser, dependency install, or browser download after external mutation is
  too late and is NO-GO.

  ```bash
  test "$(node --version | sed -E 's/^v([0-9]+).*/\1/')" = "22"
  npm --prefix src/elspeth/web/frontend ci
  LOCKED_PLAYWRIGHT_VERSION="$(node -e 'const p=require("./src/elspeth/web/frontend/package-lock.json"); console.log(p.packages["node_modules/@playwright/test"].version)')"
  INSTALLED_PLAYWRIGHT_VERSION="$(npm --prefix src/elspeth/web/frontend exec -- playwright --version | awk '{print $2}')"
  test "$INSTALLED_PLAYWRIGHT_VERSION" = "$LOCKED_PLAYWRIGHT_VERSION"
  npm --prefix src/elspeth/web/frontend exec -- playwright install chromium
  OIDC_LIST_OUTPUT="$(npm --prefix src/elspeth/web/frontend run test:e2e:oidc -- --list)"
  grep -Fq 'aws-ecs-oidc.staging.spec.ts' <<<"$OIDC_LIST_OUTPUT"
  test "$(grep -Ec '^[[:space:]]*\[[^]]+\].*aws-ecs-oidc\.staging\.spec\.ts' <<<"$OIDC_LIST_OUTPUT")" = "1"
  test -z "$(git status --porcelain)"
  ```

  **Expected result:** the acceptance browser is installed from the same
  frontend lock and Node major used by CI, exactly one dedicated OIDC spec is
  listed, and ignored dependency/browser artifacts do not dirty the candidate.

  Record exact inventory inputs `SCENARIO_A_DB_CLUSTER_IDENTIFIER` and
  `SCENARIO_B_DB_CLUSTER_IDENTIFIER`. The database operator obtains and signs
  the numeric `SHOW max_connections` result for each exact target, then
  approves a connection budget covering long-lived session/Landscape pools,
  doctor/verifier tasks, and Plan 11's short-lived per-auth-event Landscape
  engines at the configured auth rate/concurrency limit for the exact one-task
  service, with an explicit numeric safety margin. During each observation
  window, call `aws_capture cloudwatch get-metric-statistics` for namespace
  `AWS/RDS`, metric `DatabaseConnections`, the exact
  `DBClusterIdentifier` dimension, 60-second period, and `Maximum`, bounded by
  that scenario's observation timestamps. Project the successful JSON in
  memory to timestamp/count pairs plus one numeric high-water value; retain
  only that sanitized receipt and require `high_water <= approved_budget` and
  `max_connections - high_water >= safety_margin`. Missing datapoints,
  identifier mismatch, unsigned max-connections value, or exceeded budget is
  NO-GO. Do not persist raw CloudWatch JSON or describe the per-event auth
  engines as sharing the long-lived pool.
  Before provisioning, record `SCENARIO_A_TF_DIR`, `SCENARIO_A_TF_VARS`,
  `SCENARIO_B_TF_DIR`, `SCENARIO_B_TF_VARS`, each external IaC repository's
  clean 40-character commit, `.terraform.lock.hcl` hash, Terraform version,
  backend type plus hashed state key, workspace, AWS account/region, and a
  combined binding hash over those non-secret identities and the vars-file
  hash. Require remote encrypted state with locking, distinct A/B state
  identities, and no local backend. Also record the infrastructure, database,
  identity, release, evidence, and emergency-cleanup owners, sanitized
  evidence destination, and a fresh non-secret UUID `ACCEPTANCE_RUN_ID`.
  Require the out-of-repository stacks to apply that tag
  to every disposable resource and fail the pre-apply plan review on any
  uncovered taggable resource; bind the value into the sanitized inventory.
  Also record
  `ACCEPTANCE_TEARDOWN_DEADLINE_UTC` (at most four hours after the planned
  gate), plus a 90-minute cleanup reserve and phase estimates. Before starting
  any new live phase, require the remaining time exceeds that phase's estimate
  plus the cleanup reserve; otherwise route directly to Task 8. Task 8 remains
  mandatory after any success, failure, interruption, or timeout once a
  registry tag or environment exists.

  Create the Plan-10 mode-0600 atomic control manifest **before** the first
  external mutation. It binds all identities above, two separately namespaced
  inventory documents, evidence paths, candidate SHA, teardown deadline, and
  `cleanup_required: false`; validate it from a fresh process. Every later
  mutation atomically updates this manifest. An INT/TERM trap records the
  interrupted phase and routes to Task 8; process/host loss is recovered with
  `control-manifest load-cleanup`. Never rely on shell variables as the only
  cleanup record. Also initialize the protected gate ledger and record every
  Task 1–8 checkbox by stable check ID, timestamps, exit status, SHA, and
  sanitized receipt hash.

  ```bash
  export CONTROL_MANIFEST="${CONTROL_MANIFEST:?set approved durable mode-0600 control-manifest path}"
  export SCENARIO_A_INVENTORY="${SCENARIO_A_INVENTORY:?set closed sanitized Scenario A inventory path}"
  export SCENARIO_B_INVENTORY="${SCENARIO_B_INVENTORY:?set closed sanitized Scenario B inventory path}"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest init \
      --file "$CONTROL_MANIFEST" \
      --acceptance-run-id "$ACCEPTANCE_RUN_ID" \
      --candidate-sha "$CANDIDATE_SHA" \
      --release-pr-number "$PR_NUMBER" \
      --ci-run-id "$CI_RUN_ID" \
      --aws-account-id "$AWS_ACCOUNT_ID" \
      --aws-region "$AWS_REGION" \
      --scenario-a-inventory "$SCENARIO_A_INVENTORY" \
      --scenario-b-inventory "$SCENARIO_B_INVENTORY" \
      --scenario-a-tf-binding "$SCENARIO_A_TF_BINDING_SHA" \
      --scenario-b-tf-binding "$SCENARIO_B_TF_BINDING_SHA" \
      --gate-ledger "$GATE_LEDGER" \
      --teardown-deadline-utc "$ACCEPTANCE_TEARDOWN_DEADLINE_UTC"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest validate --file "$CONTROL_MANIFEST" --acceptance-run-id "$ACCEPTANCE_RUN_ID" --candidate-sha "$CANDIDATE_SHA"
  ```

  **Expected result:** init fails if the destination exists; exact-owner resume
  uses `validate` rather than replacement. A fresh process can load both
  isolated inventories and every cleanup identity without shell history.

  The scenario inventories are closed documents, not a shared mutable variable
  bag. Define `load_scenario A|B` to clear all generic scenario variables,
  load exactly one inventory through `control-manifest`, validate every ARN
  and resource against both `ACCEPTANCE_RUN_ID` and the scenario ID, and stamp
  that ID into every receipt. Invoke it before every generic
  `ECS_CLUSTER`/`ECS_SERVICE`/listener/target/database command and never carry a
  task ARN or receipt across scenarios.

  Define and use this closed connection-budget receipt for each scenario's
  exact final observation window:

  ```bash
  capture_connection_budget() {
      local cluster_id="$1" start_utc="$2" end_utc="$3" max_connections="$4" approved_budget="$5" safety_margin="$6"
      local raw="" receipt=""
      raw="$(aws_capture cloudwatch get-metric-statistics --region "$AWS_REGION" --namespace AWS/RDS --metric-name DatabaseConnections --dimensions "Name=DBClusterIdentifier,Value=$cluster_id" --start-time "$start_utc" --end-time "$end_utc" --period 60 --statistics Maximum --output json)"
      receipt="$(jq -ce --arg cluster_hash "$(printf '%s' "$cluster_id" | sha256sum | awk '{print $1}')" --argjson max "$max_connections" --argjson budget "$approved_budget" --argjson margin "$safety_margin" '
          [.Datapoints[] | {timestamp:.Timestamp,count:.Maximum}] as $points
          | ($points | length) > 0
          and all($points[]; (.count | type) == "number")
          | if . then ($points | map(.count) | max) as $high
            | {schema:"elspeth.rds-connection-budget.v1",cluster_id_sha256:$cluster_hash,points:$points,high_water:$high,max_connections:$max,approved_budget:$budget,safety_margin:$margin,ok:($high <= $budget and ($max-$high) >= $margin)}
            else error("missing or non-numeric DatabaseConnections datapoints") end
      ' <<<"$raw")"
      unset raw
      jq -e '.ok == true' <<<"$receipt" >/dev/null
      persist_sanitized_receipt "$ACTIVE_SCENARIO_ID" connection-budget "$cluster_id" "$receipt"
  }
  ```

  Invoke it with the database-operator-signed numeric values and the timestamps
  bracketing that scenario's 20-sample window. The receipt stores only a hash
  of the identifier, timestamp/count pairs, high-water, approved numbers, and
  inequalities; raw CloudWatch JSON and database role/URL data are discarded.

- [ ] Build and publish the two temporary acceptance artifacts before
  infrastructure mutation. Obtain the
  approved `ECR_REGISTRY` (host only) and `ECR_REPOSITORY` (repository name),
  plus the `ROLLBACK_BASELINE_SHA` durably recorded in Plan 10's Filigree
  comment. Build that source commit and push both images under
  candidate-specific tags, then resolve each tag back to an immutable registry
  digest:

  ```bash
  set -Eeuo pipefail
  CANDIDATE_SHA="$(git rev-parse HEAD)"
  test "$(aws_capture sts get-caller-identity --query Account --output text)" = "$AWS_ACCOUNT_ID"
  REPOSITORY_IDENTITY="$(aws_capture ecr describe-repositories --region "$AWS_REGION" --repository-names "$ECR_REPOSITORY" --query 'repositories[0].{registryId:registryId,repositoryUri:repositoryUri}' --output json)"
  jq -e --arg account "$AWS_ACCOUNT_ID" --arg uri "$ECR_REGISTRY/$ECR_REPOSITORY" '.registryId == $account and .repositoryUri == $uri' <<<"$REPOSITORY_IDENTITY" >/dev/null
  test -n "${ROLLBACK_BASELINE_SHA:-}"
  test "$ROLLBACK_BASELINE_SHA" != "$CANDIDATE_SHA"
  git merge-base --is-ancestor "$ROLLBACK_BASELINE_SHA" "$CANDIDATE_SHA"
  git grep -q "oidc_audience_claim" "$ROLLBACK_BASELINE_SHA" -- src/elspeth/web/config.py
  git grep -q "oidc_authorization_allowed_origins" "$ROLLBACK_BASELINE_SHA" -- src/elspeth/web/config.py
  git grep -q "allowed_origins" "$ROLLBACK_BASELINE_SHA" -- src/elspeth/web/auth/urls.py
  git grep -q "token_use" "$ROLLBACK_BASELINE_SHA" -- src/elspeth/web/auth/oidc.py
  git grep -Eq 'response_type.*code|response_type=code' "$ROLLBACK_BASELINE_SHA" -- src/elspeth/web/frontend/src/components/auth/LoginPage.tsx
  git grep -Eq 'code_challenge_method.*S256|code_challenge_method=S256' "$ROLLBACK_BASELINE_SHA" -- src/elspeth/web/frontend/src/components/auth/LoginPage.tsx
  git grep -q 'removeItem("oidc_transaction")' "$ROLLBACK_BASELINE_SHA" -- src/elspeth/web/frontend/src/components/auth/LoginPage.tsx
  test "${TARGET_PLATFORM:?}" = "linux/amd64" || test "$TARGET_PLATFORM" = "linux/arm64"
  git archive "$ROLLBACK_BASELINE_SHA" | docker buildx build --platform "$TARGET_PLATFORM" --load --label "org.opencontainers.image.revision=$ROLLBACK_BASELINE_SHA" -t elspeth:ecs-rollback-baseline -
  test "$(docker image inspect elspeth:ecs-rollback-baseline --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')" = "$ROLLBACK_BASELINE_SHA"
  test "$(docker image inspect elspeth:ecs-rollback-baseline --format '{{.Os}}/{{.Architecture}}')" = "$TARGET_PLATFORM"
  test "$(docker image inspect elspeth:ecs-0.7.1-closeout --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')" = "$CANDIDATE_SHA"
  test "$(docker image inspect elspeth:ecs-0.7.1-closeout --format '{{.Os}}/{{.Architecture}}')" = "$TARGET_PLATFORM"
  ROLLBACK_BASELINE_TAG="acceptance-${ACCEPTANCE_RUN_ID}-baseline-${ROLLBACK_BASELINE_SHA}"
  CANDIDATE_TAG="acceptance-${ACCEPTANCE_RUN_ID}-0.7.1-${CANDIDATE_SHA}"
  for tag in "$ROLLBACK_BASELINE_TAG" "$CANDIDATE_TAG"; do
      listing="$(aws_capture ecr list-images --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" --filter tagStatus=TAGGED --output json)"
      jq -e --arg tag "$tag" '[.imageIds[] | select(.imageTag == $tag)] | length == 0' <<<"$listing" >/dev/null
  done

  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update --file "$CONTROL_MANIFEST" --cleanup-required true --ecr-baseline-tag "$ROLLBACK_BASELINE_TAG" --ecr-candidate-tag "$CANDIDATE_TAG"
  CLEANUP_REQUIRED=1
  (
      export DOCKER_CONFIG="$(mktemp -d)"
      chmod 700 "$DOCKER_CONFIG"
      trap 'rm -rf -- "$DOCKER_CONFIG"' EXIT
      aws_ecr_login "$ECR_REGISTRY" "$AWS_REGION"
      docker tag elspeth:ecs-rollback-baseline "$ECR_REGISTRY/$ECR_REPOSITORY:$ROLLBACK_BASELINE_TAG"
      docker tag elspeth:ecs-0.7.1-closeout "$ECR_REGISTRY/$ECR_REPOSITORY:$CANDIDATE_TAG"
      docker push "$ECR_REGISTRY/$ECR_REPOSITORY:$ROLLBACK_BASELINE_TAG"
      docker push "$ECR_REGISTRY/$ECR_REPOSITORY:$CANDIDATE_TAG"
      docker logout "$ECR_REGISTRY"
  )
  ROLLBACK_BASELINE_DIGEST="$(aws_capture ecr describe-images --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" --image-ids imageTag="$ROLLBACK_BASELINE_TAG" --query 'imageDetails[0].imageDigest' --output text)"
  IMAGE_DIGEST="$(aws_capture ecr describe-images --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" --image-ids imageTag="$CANDIDATE_TAG" --query 'imageDetails[0].imageDigest' --output text)"
  test -n "$ROLLBACK_BASELINE_DIGEST" && test "$ROLLBACK_BASELINE_DIGEST" != "None"
  test -n "$IMAGE_DIGEST" && test "$IMAGE_DIGEST" != "None"
  ROLLBACK_BASELINE_IMAGE="$ECR_REGISTRY/$ECR_REPOSITORY@$ROLLBACK_BASELINE_DIGEST"
  CANDIDATE_IMAGE="$ECR_REGISTRY/$ECR_REPOSITORY@$IMAGE_DIGEST"
  docker pull "$ROLLBACK_BASELINE_IMAGE"
  docker pull "$CANDIDATE_IMAGE"
  test "$(docker image inspect "$ROLLBACK_BASELINE_IMAGE" --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')" = "$ROLLBACK_BASELINE_SHA"
  test "$(docker image inspect "$CANDIDATE_IMAGE" --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')" = "$CANDIDATE_SHA"
  test "$(docker image inspect "$ROLLBACK_BASELINE_IMAGE" --format '{{.Os}}/{{.Architecture}}')" = "$TARGET_PLATFORM"
  test "$(docker image inspect "$CANDIDATE_IMAGE" --format '{{.Os}}/{{.Architecture}}')" = "$TARGET_PLATFORM"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update --file "$CONTROL_MANIFEST" --ecr-baseline-digest "$ROLLBACK_BASELINE_DIGEST" --ecr-candidate-digest "$IMAGE_DIGEST"
  ```

- [ ] Plan and apply both disposable stacks from the bound external IaC
  revisions using those exact immutable image references. The Plan-10 runbook
  defines `verify_tf_binding` and a protected `terraform_capture` wrapper with
  the same bounded stdout/stderr and static-error discipline as `aws_capture`.
  Each stack must declare the closed inputs `acceptance_run_id`,
  `scenario_id`, `candidate_image`, and `rollback_baseline_image`; no mutable
  image tag may enter a task definition.

  ```bash
  set -Eeuo pipefail
  TF_PLAN_DIR="$(mktemp -d -p /tmp elspeth-tf-plans.XXXXXX)"
  chmod 700 "$TF_PLAN_DIR"
  trap 'rm -rf -- "$TF_PLAN_DIR"' EXIT
  run_acceptance_tf_bounded() {
      local seconds=""
      seconds="$(uv run --frozen python -c 'import sys; from datetime import datetime,timedelta,timezone; d=datetime.fromisoformat(sys.argv[1].replace("Z","+00:00"))-timedelta(minutes=90); remaining=max(0,int((d-datetime.now(timezone.utc)).total_seconds())); print(min(1200,remaining))' "$ACCEPTANCE_TEARDOWN_DEADLINE_UTC")"
      test "$seconds" -gt 0 || return 124
      timeout --foreground "${seconds}s" "$@"
  }

  plan_and_apply_scenario() {
      local name="$1" scenario_id="$2" dir="$3" vars="$4" expected_binding="$5"
      local plan="$TF_PLAN_DIR/$name.tfplan" receipt="$TF_PLAN_DIR/$name.plan.receipt.json"
      local plan_sha="" durable_receipt="" approval_receipt=""
      verify_tf_binding "$scenario_id" "$dir" "$vars" "$expected_binding"
      run_acceptance_tf_bounded terraform -chdir="$dir" init -input=false -lock-timeout=5m
      run_acceptance_tf_bounded terraform -chdir="$dir" validate
      run_acceptance_tf_bounded terraform -chdir="$dir" plan -input=false -lock-timeout=5m \
          -var-file="$vars" \
          -var="acceptance_run_id=$ACCEPTANCE_RUN_ID" \
          -var="scenario_id=$scenario_id" \
          -var="candidate_image=$CANDIDATE_IMAGE" \
          -var="rollback_baseline_image=$ROLLBACK_BASELINE_IMAGE" \
          -out="$plan"
      chmod 600 "$plan"
      terraform_capture -chdir="$dir" show -json "$plan" | \
          uv run --frozen python -m elspeth.web.aws_ecs_acceptance sanitize-evidence --kind terraform-plan >"$receipt"
      chmod 600 "$receipt"
      plan_sha="$(sha256sum "$plan" | awk '{print $1}')"
      durable_receipt="$(persist_sanitized_receipt "$scenario_id" terraform-plan "$plan_sha" "$receipt")"
      approval_receipt="$(require_signed_tf_plan_approval "$scenario_id" "$durable_receipt")"
      uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update --file "$CONTROL_MANIFEST" --terraform-plan-receipt "$scenario_id:$plan_sha:$durable_receipt:$approval_receipt"
      run_acceptance_tf_bounded terraform -chdir="$dir" apply -input=false -lock-timeout=5m "$plan"
      rm -f -- "$plan" "$receipt"
      uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update --file "$CONTROL_MANIFEST" --terraform-applied "$scenario_id"
      plan="$TF_PLAN_DIR/$name.noop.tfplan"
      receipt="$TF_PLAN_DIR/$name.noop.receipt.json"
      run_acceptance_tf_bounded terraform -chdir="$dir" plan -input=false -lock-timeout=5m -detailed-exitcode \
          -var-file="$vars" \
          -var="acceptance_run_id=$ACCEPTANCE_RUN_ID" \
          -var="scenario_id=$scenario_id" \
          -var="candidate_image=$CANDIDATE_IMAGE" \
          -var="rollback_baseline_image=$ROLLBACK_BASELINE_IMAGE" \
          -out="$plan"
      chmod 600 "$plan"
      terraform_capture -chdir="$dir" show -json "$plan" | \
          uv run --frozen python -m elspeth.web.aws_ecs_acceptance sanitize-evidence --kind terraform-plan >"$receipt"
      durable_receipt="$(persist_sanitized_receipt "$scenario_id" terraform-noop "$(sha256sum "$plan" | awk '{print $1}')" "$receipt")"
      uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update --file "$CONTROL_MANIFEST" --terraform-noop-receipt "$scenario_id:$durable_receipt"
      rm -f -- "$plan" "$receipt"
  }

  plan_and_apply_scenario scenario_a A "$SCENARIO_A_TF_DIR" "$SCENARIO_A_TF_VARS" "$SCENARIO_A_TF_BINDING_SHA"
  plan_and_apply_scenario scenario_b B "$SCENARIO_B_TF_DIR" "$SCENARIO_B_TF_VARS" "$SCENARIO_B_TF_BINDING_SHA"
  ```

  `terraform plan -out` files can contain cleartext sensitive values. They
  stay mode 0600 under `/tmp`, are never committed/exported, and are deleted
  immediately after exact-plan apply; durable evidence stores only their hash,
  sanitized closed projection, approval receipt, apply result, and post-apply
  no-change result. Exit 2 from the final `-detailed-exitcode` command means
  drift and is NO-GO. Any partial apply leaves `cleanup_required: true` in the
  already durable manifest and routes directly to Task 8.

  Through the approved external inventory, resolve exact ARNs for
  `CANDIDATE_TASK_DEFINITION`, `DOCTOR_TASK_DEFINITION`,
  `PAYLOAD_VERIFIER_TASK_DEFINITION`, `LOCAL_AUTH_VERIFIER_TASK_DEFINITION`, Scenario B's
  `PREVIOUS_TASK_DEFINITION`, and `ROLLBACK_DOCTOR_TASK_DEFINITION`. Candidate,
  doctor, payload verifier, and local-auth verifier named containers use exactly
  `CANDIDATE_IMAGE`; previous and rollback-doctor named containers use exactly
  `ROLLBACK_BASELINE_IMAGE`. Compare the returned task-definition JSON, never
  tags. For every relevant definition mechanically require
  `runtimePlatform.operatingSystemFamily == LINUX` and the closed architecture
  mapping `linux/amd64` → `X86_64`, `linux/arm64` → `ARM64`,
  non-root user, approved `taskRoleArn` (runtime S3/Bedrock/default-chain) and
  `executionRoleArn` (ECR pull, awslogs, Secrets Manager), no AWS access-key,
  session-token, profile, role, or endpoint override in environment/secrets/
  command, and every secret-backed ELSPETH setting present only in
  `containerDefinitions[].secrets`. Require web/payload/local-auth verifier
  database secret references to resolve, under database-operator attestation,
  to a DML-only runtime PostgreSQL principal with no schema CREATE or ownership;
  require init-capable candidate/rollback doctor definitions to use a distinct
  approved schema-owner secret. Preserve only boolean privilege results and
  distinct-reference hashes, never database role names, URLs, SQL, or secret
  values. Require the exact approved EFS
  `fileSystemId`, access point, `transitEncryption=ENABLED`,
  `authorizationConfig.iam=ENABLED`, `rootDirectory` absent or `/`, and a
  read-write mount at the configured data-dir ancestor. Require task-role
  `elasticfilesystem:ClientMount`/`ClientWrite` on the exact filesystem/access
  point and forbid `ClientRootAccess` unless separately approved.
  Because ECS Exec is load-bearing, mechanically require the task role's exact
  `ssmmessages:CreateControlChannel`, `ssmmessages:CreateDataChannel`,
  `ssmmessages:OpenControlChannel`, and `ssmmessages:OpenDataChannel` grants
  without a broader `ssmmessages:*`; prove the task subnets can reach Session
  Manager Messages (the exact interface endpoint when private-only). If Exec
  uses a customer KMS key or CloudWatch/S3 session logging, also prove the
  exact task/caller KMS and log-destination permissions plus reachable
  endpoints. Missing connectivity or permissions blocks before the first
  acceptance task rather than being discovered during a live role check.
  Candidate/previous web health
  checks use the direct Python loopback `/api/health` command—never
  `elspeth health` or `/api/ready`—with bounded timeout/retries and
  `startPeriod >= 150`; the service has
  `healthCheckGracePeriodSeconds >= 150`, `enableExecuteCommand == true`, and
  Fargate platform `1.4.0` or `LATEST`; every launched task must report the
  1.4.0 platform family required by EFS and ECS Exec. Payload/local-auth
  verifier definitions use explicit `user: "1000:1000"` so EFS checks cannot
  pass as root. Verifier/doctor definitions explicitly replace the image entrypoint where
  they run `python -m ...`; a container override command alone would be passed
  to the Dockerfile's `elspeth` entrypoint. The candidate doctor/verifier must
  report `elspeth version 0.7.1`. Persist only an allowlisted sanitized
  projection of these checks, never raw environment, secret ARN/value, command,
  or task-definition JSON. Record the SHA, immutable image digest, exact
  task-definition ARNs, rollback-baseline SHA/image/task definitions,
  target-group ARN, EFS access point/mount identity, and Aurora cluster
  identity in the sanitized change record. Add the same non-secret baseline
  identities to the Plan 10 Filigree step using `comment_add`. Do not record
  secret values, the ECR login token, or raw credential-bearing URLs.

- [ ] Before each live scenario, complete plan 10's release/schema
  compatibility record template. The database operator authors and approves
  it; the release operator countersigns it. Bind the candidate SHA, immutable
  image digest, exact candidate/doctor task-definition ARNs, package version,
  both schema epochs, structural and semantics-only changes, reset posture,
  and rollback decision. Require Landscape epoch 23, the
  `run_web_plugin_policy` table, and the approved archive/export plus
  drop/recreate decision for any non-empty pre-23 database; rolling candidate
  state back to epoch-22 code is not permitted. The upgrade scenario must additionally bind the
  qualified rollback-baseline version/SHA/image/task-definition and state
  `rollback_permitted: yes` with evidence before the rollback rehearsal. The
  first-deploy scenario records previous compatibility as not applicable and
  binds recovery to fixed-503 plus desired-zero. Record each compatibility
  record ID in the acceptance evidence; an unknown, expired, unsigned, or
  `rollback_permitted: no` upgrade record blocks deployment.

- [ ] Prove both S3 plugins against real S3 before either scenario can earn
  runtime GO. Supply a dedicated disposable `ELSPETH_TEST_S3_BUCKET` from the
  approved acceptance inventory and use the acceptance controller's ordinary
  AWS default credential chain — no `endpoint_url`, access-key, secret-key, or
  session-token field may appear in pipeline config. Run:

  ```bash
  test -n "${ELSPETH_TEST_S3_BUCKET:-}"
  uv run --frozen pytest tests/integration/plugins/sources/test_aws_s3_source_live.py tests/integration/plugins/sinks/test_aws_s3_sink_live.py -m "slow and integration" -q
  ```

  Require all source and sink CSV/JSON/JSONL cases to execute and pass with
  zero skips. The sink lane must also prove first-create `IfNoneMatch`,
  cumulative `IfMatch`, fresh-sink collision, and intervening-writer rejection
  without clobbering external bytes. Retain sanitized bucket/prefix, content
  hashes, and audit assertions as evidence; both tests delete every
  UUID-scoped object in `finally`. Missing bucket, credentials, permissions,
  cleanup, or any skip is BLOCKED/NO-GO, not an optional smoke.

- [ ] Prove the reusable Bedrock Guardrail lane with Plan 15C's exact marker
  contract before candidate-task adaptation. Supply only the approved opaque
  profile aliases and operator-owned fixture values through the protected
  environment names pinned by Plan 10
  (`ELSPETH_LIVE_BEDROCK_{PROMPT,CONTENT}_PROFILE_ALIAS`,
  `ELSPETH_LIVE_BEDROCK_{PROMPT,CONTENT}_{SAFE,BLOCKED}_TEXT`, and
  `ELSPETH_LIVE_BEDROCK_{PROMPT,CONTENT}_EXPECTED_VERSION`), then run:

  ```bash
  export ELSPETH_RUN_LIVE_BEDROCK_GUARDRAILS=1
  LIVE_GUARDRAIL_OUTPUT="$(uv run --extra aws pytest \
    tests/integration/plugins/transforms/aws/test_bedrock_guardrails_live.py \
    -m live_aws -q -rs 2>&1)"
  printf '%s' "$LIVE_GUARDRAIL_OUTPUT" | grep -Eq '[0-9]+ passed'
  ! printf '%s' "$LIVE_GUARDRAIL_OUTPUT" | grep -Eqi 'skipped|deselected'
  unset LIVE_GUARDRAIL_OUTPUT
  ```

  The live test itself must fail, not skip, when the env gate is set but an
  approved alias/fixture/policy-version input is missing. Retain only its
  bounded sanitized receipt/hash. Any skip, deselection, raw binding/text, or
  policy-version mismatch is BLOCKED/NO-GO. This controller proof does not
  replace the per-candidate `verify-bedrock-guardrails` ECS-Exec proof below.
  Plan 12 owns both executions, Landscape/telemetry correlation, cleanup, and
  GO/NO-GO; Plan 15C supplies the reusable checker and bounded receipt but does
  not claim live ECS acceptance.

- [ ] Also prove the runtime credential path from **each healthy candidate ECS
  task**. Resolve its exact task ARN and use ECS Exec to run
  `python -m elspeth.web.aws_ecs_acceptance verify-s3`. The task receives only
  the approved non-secret bucket, UUID prefix, and region; its task role grants
  scoped `GetObject`/`PutObject`/`DeleteObject` on that prefix and no static
  credential, profile, role, or endpoint override. Require the bounded
  write/read/hash/collision receipt and confirmed cleanup. Controller-side
  pytest proves deeper plugin semantics but cannot substitute for this
  task-role/default-chain and lean-image proof.

- [ ] Prove Bedrock through the **candidate ECS task role**, not the acceptance
  controller's credentials. The approved task definition supplies a non-secret
  `ELSPETH_BEDROCK_LIVE_TEST_MODEL` in `bedrock/<model-id>` form and
  `AWS_REGION`/`AWS_DEFAULT_REGION`; it supplies no AWS access-key, secret-key,
  session-token, endpoint, profile, or role override. After the candidate task
  is healthy in each scenario, resolve its exact task ARN and use ECS Exec to
  run both checks through one closed extractor:

  ```bash
  run_candidate_role_check() {
      local task_arn="$1" check="$2" stream="" receipt=""
      case "$check" in
        verify-s3|verify-bedrock|verify-bedrock-guardrails|verify-operator-telemetry) ;;
        *) return 64 ;;
      esac
      stream="$(aws_capture ecs execute-command --cluster "$ECS_CLUSTER" --task "$task_arn" --container "$WEB_CONTAINER_NAME" --interactive --command "python -m elspeth.web.aws_ecs_acceptance $check")"
      receipt="$(printf '%s' "$stream" | uv run --frozen python -m elspeth.web.aws_ecs_acceptance extract-exec-receipt --check "$check" --candidate-sha "$CANDIDATE_SHA" --task-arn "$task_arn" --scenario-id "$ACTIVE_SCENARIO_ID")"
      unset stream
      jq -e --arg check "$check" --arg sha "$CANDIDATE_SHA" --arg task "$task_arn" --arg scenario "$ACTIVE_SCENARIO_ID" '.schema == "elspeth.aws-ecs-exec-receipt.v1" and .ok == true and .check == $check and .candidate_sha == $sha and .task_arn == $task and .scenario_id == $scenario' <<<"$receipt" >/dev/null
      persist_sanitized_receipt "$ACTIVE_SCENARIO_ID" "$check" "$task_arn" "$receipt"
  }

  run_candidate_role_checks() {
      local task_arn="$1"
      require_exec_agent_running "$task_arn"
      run_candidate_role_check "$task_arn" verify-s3
      run_candidate_role_check "$task_arn" verify-bedrock
      run_candidate_role_check "$task_arn" verify-bedrock-guardrails
      run_candidate_role_check "$task_arn" verify-operator-telemetry
  }
  ```

  `aws_capture` retains the interactive Session Manager stream only in its
  bounded protected buffers/variable; it is never echoed or persisted. The
  extractor requires exactly one
  `ELSPETH_ACCEPTANCE_RECEIPT_V1:<base64url-json>` sentinel and fails on
  missing, duplicate, malformed, oversized, wrong-check, or `ok: false`
  receipts before adding the local task/SHA/scenario binding.

  Require exit 0 and the helper's bounded sanitized receipt: non-empty
  completion accepted, hashed returned model/request id, honest token/cache
  presence flags, and finite non-negative provider cost plus its source when
  available. The receipt must contain no prompt/content, model id, request id,
  credential, account/role ARN, raw response, or raw provider error. Missing
  model/region, task-role permission, default-chain credentials, metadata, or
  any failure is BLOCKED/NO-GO; there is no live-test skip in closeout. Retain
  the sanitized receipt with the scenario evidence.

  The Guardrails receipt additionally requires safe and intervened decisions
  for both explicit control families, immutable versions, and audit-first
  Landscape records. The operator-telemetry receipt requires a web metric plus
  `RunStarted`/`RunFinished` trace correlated to the same Landscape run and a
  negative collector-outage proof. Neither receipt may contain raw fixture,
  provider body, metric/trace service response, Guardrail messaging, ARN,
  account/request ID, URL, credential, or exception text.

  Invoke `run_candidate_role_checks TASK_ARN` for every distinct candidate
  task that contributes
  evidence: Scenario A initial task, forced replacement, and first-recovery;
  Scenario B initial candidate and candidate-after-rollback redeploy. A receipt
  from an earlier task ARN cannot be reused. S3 inputs are already present in
  each candidate task definition as `ELSPETH_ACCEPTANCE_S3_BUCKET`,
  UUID-scoped `ELSPETH_ACCEPTANCE_S3_PREFIX`, and region; ECS Exec does not
  inject replacement environment variables.

- [ ] In Scenario B, prepare the real-browser OIDC evidence runner. Supply the
  Cognito test credentials only through pre-existing environment variables;
  never put them in the command, evidence, trace, screenshot, or change record:

  ```bash
  test -n "${OIDC_TEST_USERNAME:-}"
  test -n "${OIDC_TEST_PASSWORD:-}"
  test -n "${OIDC_EXPECTED_ISSUER:-}"
  test -n "${OIDC_EXPECTED_AUDIENCE:-}"
  test -n "${OIDC_EXPECTED_AUTHORIZATION_ORIGIN:-}"
  test -n "${COGNITO_USER_POOL_ID:-}"
  test -n "${ALB_ARN:-}"
  test "${OIDC_EXPECTED_AUDIENCE_CLAIM:-}" = "client_id"
  uv run --frozen python -c 'import sys; from urllib.parse import urlsplit; u=urlsplit(sys.argv[1]); assert u.scheme == "https" and u.netloc and u.hostname is not None and u.username is None and u.password is None and u.path == "" and not u.query and not u.fragment; origin = f"https://{u.hostname.lower()}" + (f":{u.port}" if u.port not in (None, 443) else ""); assert sys.argv[1] == origin' "$ALB_BASE_URL"
  OIDC_EVIDENCE_DIR="$(mktemp -d -p /tmp elspeth-oidc.XXXXXX)"
  chmod 700 "$OIDC_EVIDENCE_DIR"
  ACCEPTANCE_STATE=""
  COGNITO_CLIENT_PREFLIGHT="$OIDC_EVIDENCE_DIR/cognito-app-client.preflight"
  : > "$COGNITO_CLIENT_PREFLIGHT"
  chmod 600 "$COGNITO_CLIENT_PREFLIGHT"
  ALB_ATTRIBUTES="$(aws_capture elbv2 describe-load-balancer-attributes --load-balancer-arn "$ALB_ARN" --output json)"
  jq -e '[.Attributes[] | select(.Key == "access_logs.s3.enabled")][0].Value == "false"' <<<"$ALB_ATTRIBUTES" >/dev/null
  aws_capture cognito-idp describe-user-pool-client --region "$AWS_REGION" --user-pool-id "$COGNITO_USER_POOL_ID" --client-id "$OIDC_EXPECTED_AUDIENCE" --query "UserPoolClient.{clientId:ClientId,allowedOAuthFlowsUserPoolClient:AllowedOAuthFlowsUserPoolClient,allowedOAuthFlows:AllowedOAuthFlows,allowedOAuthScopes:AllowedOAuthScopes,callbackURLs:CallbackURLs,hasClientSecret:contains(keys(@), 'ClientSecret')}" --output json > "$COGNITO_CLIENT_PREFLIGHT"
  jq -e --arg client "$OIDC_EXPECTED_AUDIENCE" --arg callback "$ALB_BASE_URL" '.clientId == $client and .allowedOAuthFlowsUserPoolClient == true and .hasClientSecret == false and (.allowedOAuthFlows | index("code")) != null and (.allowedOAuthFlows | index("implicit")) == null and ((["openid","profile","email"] - .allowedOAuthScopes) | length) == 0 and (.callbackURLs | index($callback)) != null' "$COGNITO_CLIENT_PREFLIGHT"
  run_oidc_evidence() {
      phase="$1"
      STAGING_BASE_URL="$ALB_BASE_URL" OIDC_TEST_USERNAME="$OIDC_TEST_USERNAME" OIDC_TEST_PASSWORD="$OIDC_TEST_PASSWORD" OIDC_EXPECTED_ISSUER="$OIDC_EXPECTED_ISSUER" OIDC_EXPECTED_AUDIENCE="$OIDC_EXPECTED_AUDIENCE" OIDC_EXPECTED_AUTHORIZATION_ORIGIN="$OIDC_EXPECTED_AUTHORIZATION_ORIGIN" OIDC_EXPECTED_AUDIENCE_CLAIM="$OIDC_EXPECTED_AUDIENCE_CLAIM" OIDC_EVIDENCE_PHASE="$phase" OIDC_EVIDENCE_FILE="$OIDC_EVIDENCE_DIR/$phase.json" npm --prefix src/elspeth/web/frontend run test:e2e:oidc
      test "$(stat -c '%a' "$OIDC_EVIDENCE_DIR/$phase.json")" = "600"
  }
  ```

  The Playwright harness must drive ELSPETH's SSO control and Cognito hosted
  UI through the public-client authorization-code + S256 PKCE flow, return
  through the real callback with no bearer token in the URL, validate issuer/audience/subject and
  expiry in memory, and successfully call `/api/auth/me` plus session
  create/read/delete. A bearer token injected without that browser flow is not
  evidence. The sanitized app-client preflight is also mandatory evidence;
  never capture the raw `describe-user-pool-client` response because a
  confidential client response can contain `ClientSecret`.
  The disposable acceptance ALB must have access logging disabled as proven
  above, because Cognito necessarily delivers the short-lived PKCE-bound code
  in the callback query before browser cleanup. Production deployments that
  enable ALB access logs require the separately documented approved retention
  and access policy; browser redaction cannot sanitize an upstream request log.

- [ ] Bootstrap Scenario B's **fresh** Aurora state before collecting the first
  OIDC phase. Keep the listener fixed at 503 and service desired-zero. Run the
  hardened one-shot doctor with `ROLLBACK_DOCTOR_TASK_DEFINITION` first using
  `doctor aws-ecs --init-schema --json`, then again without `--init-schema`;
  require empty ECS failures, one exact stopped task, every essential exit 0,
  and every sanitized check successful. This proves the rollback-baseline
  image—not candidate code—can initialize and read its own schema. A
  snapshot-restored alternative is allowed only with the exact snapshot ARN,
  schema epochs, baseline compatibility evidence, and a passing non-mutating
  rollback doctor. Then require no running/pending service tasks and the exact
  `PREVIOUS_TASK_DEFINITION` using `ROLLBACK_BASELINE_IMAGE`. Capture the
  pre-launch primary deployment identity/created time, force a fresh
  zero-overlap deployment, and prove a distinct post-launch primary plus sole
  healthy task:

  ```bash
  BASELINE_PRE_DEPLOYMENT="$(aws_capture ecs describe-services --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" --query 'services[0].deployments[?status==`PRIMARY`].[id,createdAt]' --output json)"
  aws_capture ecs update-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" --task-definition "$PREVIOUS_TASK_DEFINITION" --desired-count 1 --force-new-deployment --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' >/dev/null
  aws_capture ecs wait services-stable --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE"
  BASELINE_TASK_ARN="$(aws_capture ecs list-tasks --cluster "$ECS_CLUSTER" --service-name "$ECS_SERVICE" --desired-status RUNNING --query 'taskArns[0]' --output text)"
  test -n "$BASELINE_TASK_ARN" && test "$BASELINE_TASK_ARN" != "None"
  aws_capture ecs describe-services --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" | jq -e --arg td "$PREVIOUS_TASK_DEFINITION" '(.services | length) == 1 and .services[0].desiredCount == 1 and .services[0].runningCount == 1 and .services[0].pendingCount == 0 and ([.services[0].deployments[] | select(.status == "PRIMARY" and .taskDefinition == $td and .rolloutState == "COMPLETED")] | length) == 1'
  aws_capture ecs describe-tasks --cluster "$ECS_CLUSTER" --tasks "$BASELINE_TASK_ARN" | jq -e --arg td "$PREVIOUS_TASK_DEFINITION" '(.failures | length) == 0 and (.tasks | length) == 1 and .tasks[0].taskDefinitionArn == $td'
  BASELINE_POST_DEPLOYMENT="$(aws_capture ecs describe-services --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" --query 'services[0].deployments[?status==`PRIMARY`].[id,createdAt]' --output json)"
  test "$BASELINE_POST_DEPLOYMENT" != "$BASELINE_PRE_DEPLOYMENT"
  ```

  Before enabling traffic, rerun
  `aws_capture elbv2 describe-target-groups --target-group-arns "$TARGET_GROUP_ARN"`
  and retain the sanitized response as evidence. Require exactly one target
  group with `TargetType == "ip"`, `HealthCheckEnabled == true`,
  `HealthCheckPath == "/api/ready"`, `Matcher.HttpCode == "200"`, and
  `HealthCheckTimeoutSeconds >= 6`; a successful manual curl is not a
  substitute. Then run Plan 10's exact task-to-service load-balancer/ENI/port mapping and
  target-health assertions with the rollback baseline as the expected
  revision. Only then switch the listener from fixed 503 to the target group,
  require HTTP 200 `/api/health` and `.ready == true` from `/api/ready`, and
  complete 20 consecutive 30-second service/target/probe/log/event samples.
  Now run `run_oidc_evidence previous-before-candidate`; this successful phase
  is what qualifies the recorded source as the live rollback baseline. Any
  failure drains traffic, returns Scenario B to desired-zero, and is NO-GO;
  candidate deployment must not begin.

- [ ] In Scenario A, execute plan 10's hardened one-shot doctor with
  `doctor aws-ecs --init-schema --json` against fresh Aurora databases and the
  mounted EFS state paths, then execute `doctor aws-ecs --json` against the
  initialized state. Scenario B's rollback doctor owned initialization above;
  after the baseline has been live-qualified and before candidate deployment,
  run the **candidate** `DOCTOR_TASK_DEFINITION` non-mutating against that same
  state. Every run requires an empty ECS `failures` array, the exact
  stopped task ARN, exit code 0 for every essential container, and every
  returned doctor check successful. Testcontainers or a skipped integration
  test is not a substitute.

- [ ] Execute the implemented runbook from preflight through candidate-aware
  acceptance in **both** scenarios: `DEPLOYMENT_MODE=first` in A and
  `DEPLOYMENT_MODE=upgrade` in B. Enforce one web task, no service autoscaling,
  zero-overlap stop-before-start deployment, an IP target group whose enabled
  health check uses `/api/ready`, matcher `200`, and timeout at least six
  seconds, the candidate
  task-definition ARN, candidate ENI/port target health, HTTP 200
  `/api/health`, and `.ready == true` from `/api/ready`. Any schema, task,
  target, probe, event, or log failure invokes that scenario's runbook rollback
  path and returns NO-GO.

- [ ] After Scenario B's initial candidate acceptance, run
  `run_oidc_evidence candidate-initial`. Its issuer, audience, and
  SHA-256(subject) must match the previous-release evidence.

- [ ] In local-auth Scenario A, use the shipped acceptance harness to prove
  the exact public API journey and capture non-secret evidence before task
  replacement. Supply the test username/password only through the named
  environment variables; the module must not print or persist them:

  ```bash
  test -n "${ELSPETH_ACCEPTANCE_USERNAME:-}"
  test -n "${ELSPETH_ACCEPTANCE_PASSWORD:-}"
  export ELSPETH_ACCEPTANCE_REGISTER=1
  export ELSPETH_ACCEPTANCE_BASE_URL="$ALB_BASE_URL"
  ACCEPTANCE_STATE="$(mktemp -p /tmp elspeth-acceptance.XXXXXX)"
  chmod 600 "$ACCEPTANCE_STATE"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance capture --state-file "$ACCEPTANCE_STATE"
  SESSION_ID="$(jq -er .session_id "$ACCEPTANCE_STATE")"
  APP_RUN_ID="$(jq -er .run_id "$ACCEPTANCE_STATE")"
  LANDSCAPE_RUN_ID="$(jq -er .landscape_run_id "$ACCEPTANCE_STATE")"
  BLOB_SHA256="$(jq -er .blob_sha256 "$ACCEPTANCE_STATE")"
  ARTIFACT_SHA256="$(jq -er .artifact_sha256 "$ACCEPTANCE_STATE")"
  ```

  `capture` must have authenticated, created a session, uploaded the fixed CSV,
  imported the fixed CSV→CSV YAML through `/state/yaml`, executed it to
  completion, and verified `/results`, `/outputs`, blob bytes, artifact bytes,
  accounting, and hashes. Retain the 0600 state file only for this scenario's
  replacement/rollback checks.

- [ ] Force a zero-overlap replacement using the same candidate task
  definition, prove that the task ARN changed, then run every verification
  surface against the replacement:

  ```bash
  ORIGINAL_TASK_ARN="$(aws_capture ecs list-tasks --cluster "$ECS_CLUSTER" --service-name "$ECS_SERVICE" --desired-status RUNNING --query 'taskArns[0]' --output text)"
  test -n "$ORIGINAL_TASK_ARN" && test "$ORIGINAL_TASK_ARN" != "None"
  aws_capture ecs update-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" --force-new-deployment --desired-count 1 --deployment-configuration '{"deploymentCircuitBreaker":{"enable":true,"rollback":true},"minimumHealthyPercent":0,"maximumPercent":100}' >/dev/null
  aws_capture ecs wait services-stable --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE"
  REPLACEMENT_TASK_ARN="$(aws_capture ecs list-tasks --cluster "$ECS_CLUSTER" --service-name "$ECS_SERVICE" --desired-status RUNNING --query 'taskArns[0]' --output text)"
  test -n "$REPLACEMENT_TASK_ARN" && test "$REPLACEMENT_TASK_ARN" != "None"
  test "$ORIGINAL_TASK_ARN" != "$REPLACEMENT_TASK_ARN"
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance verify-api --state-file "$ACCEPTANCE_STATE"
  run_candidate_role_checks "$REPLACEMENT_TASK_ARN"
  ```

  Separately run `PAYLOAD_VERIFIER_TASK_DEFINITION` as a one-shot Fargate task
  with an override supplying only `verify-payloads --landscape-run-id
  $LANDSCAPE_RUN_ID`; require empty launch failures, the exact stopped task,
  explicit UID/GID 1000, and exit 0 for every essential container. ECS Exec is
  deliberately not used because its agent runs commands as root and would
  mask the EFS permission contract.

  Re-run the runbook's exact-one-task, candidate task-definition, target-health,
  health, and readiness assertions. `verify-api` re-authenticates and requires
  the session, blob content, run results, output manifest, and artifact content
  to match `BLOB_SHA256`/`ARTIFACT_SHA256`. `verify-payloads` queries the exact
  landscape run inside the task and integrity-retrieves every non-null payload
  ref from the configured EFS payload store; `verify-s3` proves the candidate
  task-role/default-chain path. Any zero-ref, missing-content, hash, S3, or
  cleanup failure is NO-GO. Scenario B's browser/OIDC evidence
  is owned by `run_oidc_evidence`, not this bearer/local-auth API harness.

- [ ] Complete the runbook's 20 consecutive 30-second samples in each scenario
  after its final candidate redeploy: exact-one-task service state, candidate
  target health, both HTTP probes, candidate CloudWatch logs, and the
  EventBridge CloudWatch Logs target. Preserve separate sanitized timestamps
  and outputs. Any failed or missing sample resets that scenario's ten-minute
  window after repair/redeploy.

- [ ] Rehearse **both** manual rollback branches; success in one mode never
  substitutes for the other:

  - Scenario A (`first`): switch the listener to fixed 503 before scaling the
    service to zero; require no running/pending tasks and no serving targets.
    Only now run `LOCAL_AUTH_VERIFIER_TASK_DEFINITION` as a one-shot task on
    the same EFS/settings, with explicit Python entrypoint, and require one
    exact stopped task, every essential exit 0, local provider, existing
    `auth.db`, and read-only `DELETE` journal proof. Never open `auth.db` from
    ECS Exec while uvicorn is running. Run the candidate non-mutating doctor
    while traffic is still fixed 503 and before service mutation. Set
    `DEPLOYMENT_MODE=first-recovery` and restart the same immutable
    candidate through the constrained recovery preflight, which requires
    desired-zero, fixed-503 traffic, no tasks, and the primary task definition
    equal to the candidate; use `--force-new-deployment` and prove a new
    primary deployment identity/task ARN. Then run `verify-api`, one-shot
    `verify-payloads`, `run_candidate_role_checks` for the recovery task, all
    service/target/probe checks, and
    the complete 20-sample observation. Keep the mode-0600
    `ACCEPTANCE_STATE` file for Task 8's evidence export; it contains only the
    non-secret IDs/hashes declared by the harness contract.
  - Scenario B (`upgrade`): require the signed compatibility record permits
    rollback. If current-state compatibility is uncertain, run the
    non-mutating rollback-baseline doctor before service mutation. Restore the
    exact previous task-definition ARN with `--force-new-deployment` and the
    zero-overlap posture, prove a new deployment/task identity, and verify the previous one-task service, targets,
    and probes; run `run_oidc_evidence previous-after-rollback`. Set
    `DEPLOYMENT_MODE=upgrade` again: the restored
    previous revision is current and the immutable candidate is the distinct
    target. Run the candidate non-mutating doctor **before** mutation, then
    execute ordinary upgrade preflight and force-new deployment. Only after
    the candidate target/probes pass run
    `run_candidate_role_checks` for the redeployed candidate task,
    `run_oidc_evidence candidate-after-redeploy`, and the complete
    20-sample observation. Require
    `jq -s -e 'length == 4 and ([.[].phase] | unique | length) == 4 and ([.[].issuer] | unique | length) == 1 and ([.[].authorization_origin] | unique | length) == 1 and ([.[].audience] | unique | length) == 1 and ([.[].subject_sha256] | unique | length) == 1 and all(.[]; .audience_claim == "client_id" and .auth_me_status == 200 and .session_create_status == 201 and .session_read_status == 200 and .session_delete_status == 204 and .session_round_trip == true)' "$OIDC_EVIDENCE_DIR"/*.json`.

  Leave both disposable environments in their approved candidate state only
  until Task 8 exports evidence and destroys them. If either branch cannot be
  rehearsed, the program is NO-GO; it may not be
  waived or scoped down after the fact. Keep the mode-0700 OIDC directory and
  its sanitized app-client preflight plus four mode-0600 evidence JSON files
  intact for Task 8. There is deliberately no evidence-deleting EXIT trap:
  after external mutation, even an early failure must preserve available
  sanitized evidence until the cleanup coordinator exports it.

**Definition of Done:**

- [ ] Both run-scoped image digests were resolved before reviewed saved
  Terraform plans, exact-plan applies, and post-apply no-change proofs bound to
  distinct remote state identities.
- [ ] Real Aurora initialization and non-mutating doctor checks pass from the
  exact candidate image on mounted EFS.
- [ ] ALB health/readiness and one-task zero-overlap deployment pass.
- [ ] Blob/payload/local-auth persistence across replacement is proven in
  Scenario A; four real Cognito hosted-UI login phases pass in Scenario B with
  stable issuer/audience/subject evidence.
- [ ] Payload/local-auth EFS checks pass in explicit-UID one-shot tasks, and S3
  source/sink behavior passes from every candidate task through its task role.
- [ ] Bedrock completes inside each candidate ECS task through its task role,
  with bounded sanitized token/cache/cost provenance evidence and zero skips.
- [ ] Bedrock prompt shielding and content safety both execute independently
  through `ApplyGuardrail` inside each candidate task, with exact safe/blocked
  outcomes, `guard_content`, detect-only-positive blocking, audit-first
  evidence, immutable versions, redaction, and zero skips.
- [ ] In each candidate task, the target LLM's request context lists exactly
  the Plan-15B-authorized and locally available prompt/content controls, marks
  the operator-preferred compatible implementation and opaque profile alias,
  selects no disabled/unavailable implementation, and leaks no secret name,
  resolved config, Guardrail binding, environment name, AWS identity, or
  failure detail.
- [ ] The two acceptance Guardrails are run-scoped Terraform resources tagged
  with the acceptance run ID; evidence records immutable version and normalized
  configuration hashes, and teardown destroys both resources plus verifies no
  matching run tag remains. A shared/pre-existing Guardrail cannot satisfy this
  gate.
- [ ] AWS operator telemetry delivers one web metric and the exact pipeline
  lifecycle trace to CloudWatch/X-Ray, correlates terminal status back to
  Landscape, passes forbidden-dimension/content checks, alarms on loss, and
  proves telemetry outage does not erase or replace the audit record.
- [ ] First-deploy drain/scale-zero/first-recovery and rollback-baseline
  upgrade rollback/redeploy are both rehearsed safely, each followed by 20
  consecutive passing observation samples.
- [ ] Both final observation windows have non-empty exact-cluster
  `DatabaseConnections` receipts satisfying the approved budget and safety
  margin against signed `max_connections` values.
- [ ] Sanitized evidence binds the live result to the Task 1 SHA and immutable
  candidate plus rollback-baseline image/task-definition identities.

---

### Task 8: Export evidence and tear down disposable acceptance state

**Files:** no repository changes are expected. This task mutates only the two
approved Terraform acceptance stacks, their test identities/secrets, and the
run-scoped ECR tags. It reads and checkpoints the protected control manifest
and gate ledger at the owner-approved durable control location.

- [ ] Start or resume from the durable control manifest, never from remembered
  shell state. `control-manifest validate` must bind the expected candidate
  SHA/run ID and `load-cleanup` supplies only the closed cleanup assignments.
  Install INT/TERM handlers that atomically record the interrupted cleanup
  surface before exiting NO-GO. Confirm at least the 90-minute cleanup reserve
  remains; every Terraform/AWS/Docker/identity operation below receives a
  bounded timeout derived from the teardown deadline and checkpoints success
  or failure before the next independent surface. A timeout records failure
  and continues, so one wedged destroy cannot prevent the other stack, ECR,
  identity, evidence, or orphan sweeps from being attempted. Confirmed
  idempotent surfaces are skipped on resume; incomplete ones are retried.

  ```bash
  CLEANUP_ENV="$(mktemp -p /tmp elspeth-cleanup-env.XXXXXX)"
  chmod 600 "$CLEANUP_ENV"
  trap 'rm -f -- "$CLEANUP_ENV"' EXIT
  uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest load-cleanup --file "$CONTROL_MANIFEST" --shell-assignments >"$CLEANUP_ENV"
  . "$CLEANUP_ENV"
  rm -f -- "$CLEANUP_ENV"
  trap - EXIT
  test "$CLEANUP_REQUIRED" = "1"
  if test "${DEADLINE_EXPIRED:-0}" = "1"; then
      ACCEPTANCE_REENTRY_FORBIDDEN=1
  fi
  test "${ACCEPTANCE_REENTRY_FORBIDDEN:-0}" = "0" || test "${DEADLINE_EXPIRED:-0}" = "1"
  ```

  **Expected result:** the closed assignments reconstruct every cleanup
  identity and prior surface state. Deadline expiry forces cleanup-only mode;
  it never prevents teardown.

- [ ] Attempt evidence export **first**, even after a failed Task 7. Copy every
  available sanitized change/compatibility record, command result, CI link,
  app-client preflight, OIDC evidence file, acceptance state, and Terraform
  plan/inventory identifier to the approved durable destination. Verify its
  manifest and checksums, then set `EVIDENCE_EXPORT_CONFIRMED=1`. Never export
  credentials, tokens, raw database URLs, ECR login material, or raw payloads.
  If export fails, keep the mode-protected local evidence for owner recovery,
  set the flag to 0, checkpoint `evidence_export: failed`, and continue the
  remaining cleanup attempts. The export operation has a deadline-derived
  timeout; success checkpoints `evidence_export: confirmed` before continuing.
- [ ] Require the Task-7 inventory/evidence manifest contains the original
  `ACCEPTANCE_RUN_ID` and pre-apply tag-coverage proof. Do not generate a new
  cleanup-time value; the original tag is the orphan-recovery key after partial
  applies.
- [ ] Independently attempt identity/session cleanup. The identity owner
  deletes the disposable Cognito user or rotates its password to a new
  unrecorded value and revokes active sessions. Store only a signed completion
  receipt keyed by subject hash—never username or credential—and set
  `IDENTITY_CLEANUP_CONFIRMED=1` only on success. Likewise set
  `SHARED_RESOURCE_CLEANUP_CONFIRMED=1` after the inventory proves there were
  no shared exclusions or every excluded resource has a signed deletion or
  restoration receipt. The coordinator bounds the wait for each owner action
  by the teardown deadline, checkpoints identity and shared-resource surfaces
  independently as confirmed/failed, and continues after a timeout. No
  release/promotion tag may be created in this task.
- [ ] Run the remaining coordinator in one Bash shell. It deliberately does
  not use `set -e`: every operation is checked in an `if` branch, every failure
  is recorded, and all independent cleanup surfaces are attempted before the
  final aggregate non-zero result:

  ```bash
  set -o pipefail
  CLEANUP_FAILURES=()
  VERDICT_FAILURES=()
  record_cleanup_failure() { CLEANUP_FAILURES+=("$1"); }
  checkpoint_cleanup() {
      if ! uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update --file "$CONTROL_MANIFEST" --cleanup-checkpoint "$1:$2"; then
          CLEANUP_FAILURES+=("control_manifest_checkpoint_${1}")
          return 1
      fi
  }
  cleanup_surface_needed() {
      local state=""
      state="$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest get --file "$CONTROL_MANIFEST" --field "cleanup_states.$1")"
      test "$state" != "confirmed"
  }
  cleanup_interrupted() {
      checkpoint_cleanup coordinator interrupted
      exit 1
  }
  trap cleanup_interrupted INT TERM
  if test "${DEADLINE_EXPIRED:-0}" = "1"; then
      VERDICT_FAILURES+=(teardown_deadline)
      if test -n "${EMERGENCY_CLEANUP_DEADLINE_UTC:-}" && uv run --frozen python -c 'import sys; from datetime import datetime,timezone; assert datetime.now(timezone.utc) < datetime.fromisoformat(sys.argv[1].replace("Z","+00:00"))' "$EMERGENCY_CLEANUP_DEADLINE_UTC"; then
          CLEANUP_EFFECTIVE_DEADLINE_UTC="$EMERGENCY_CLEANUP_DEADLINE_UTC"
      else
          CLEANUP_EFFECTIVE_DEADLINE_UTC="$(uv run --frozen python -c 'from datetime import datetime,timedelta,timezone; print((datetime.now(timezone.utc)+timedelta(hours=4)).isoformat().replace("+00:00","Z"))')"
          uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update --file "$CONTROL_MANIFEST" --verdict-failure teardown_deadline --emergency-cleanup-deadline-utc "$CLEANUP_EFFECTIVE_DEADLINE_UTC" --cleanup-escalation emergency_horizon_started_or_renewed
      fi
  else
      CLEANUP_EFFECTIVE_DEADLINE_UTC="$ACCEPTANCE_TEARDOWN_DEADLINE_UTC"
  fi
  run_cleanup_bounded() {
      local seconds=""
      seconds="$(uv run --frozen python -c 'import sys; from datetime import datetime,timezone; d=datetime.fromisoformat(sys.argv[1].replace("Z","+00:00")); remaining=max(0,int((d-datetime.now(timezone.utc)).total_seconds())); print(min(1200,remaining))' "$CLEANUP_EFFECTIVE_DEADLINE_UTC")"
      test "$seconds" -gt 0 || return 124
      timeout --foreground "${seconds}s" "$@"
  }

  test "${CLEANUP_REQUIRED:-0}" = "1" || record_cleanup_failure cleanup_state_not_armed
  if test "${EVIDENCE_EXPORT_CONFIRMED:-0}" = "1"; then checkpoint_cleanup evidence_export confirmed; else record_cleanup_failure evidence_export; checkpoint_cleanup evidence_export failed; fi
  if test "${IDENTITY_CLEANUP_CONFIRMED:-0}" = "1"; then checkpoint_cleanup identity_cleanup confirmed; else record_cleanup_failure identity_cleanup; checkpoint_cleanup identity_cleanup failed; fi
  if test "${SHARED_RESOURCE_CLEANUP_CONFIRMED:-0}" = "1"; then checkpoint_cleanup shared_resource_cleanup confirmed; else record_cleanup_failure shared_resource_cleanup; checkpoint_cleanup shared_resource_cleanup failed; fi

  destroy_acceptance_stack() {
      local name="$1" scenario_id="$2" dir="$3" vars="$4" binding="$5"
      local state_output="" plan="" receipt="" plan_sha="" durable_receipt="" approval_receipt=""
      local failures_before="${#CLEANUP_FAILURES[@]}"
      if test -z "$dir" || test -z "$vars" || test -z "$binding"; then
          record_cleanup_failure "terraform_${name}_inputs"
          checkpoint_cleanup "terraform_${name}" failed
          return
      fi
      if ! verify_tf_binding "$scenario_id" "$dir" "$vars" "$binding"; then
          record_cleanup_failure "terraform_${name}_binding"
          checkpoint_cleanup "terraform_${name}" failed
          return
      fi
      if ! run_cleanup_bounded terraform -chdir="$dir" init -input=false -lock-timeout=5m; then
          record_cleanup_failure "terraform_${name}_init"
          checkpoint_cleanup "terraform_${name}_init" failed
          return
      fi
      plan="$(mktemp -p /tmp "elspeth-${name}-destroy.XXXXXX")"
      receipt="$(mktemp -p /tmp "elspeth-${name}-destroy-receipt.XXXXXX")"
      chmod 600 "$plan" "$receipt"
      if ! run_cleanup_bounded terraform -chdir="$dir" plan -destroy -input=false -lock-timeout=5m \
          -var-file="$vars" \
          -var="acceptance_run_id=$ACCEPTANCE_RUN_ID" \
          -var="scenario_id=$scenario_id" \
          -var="candidate_image=$CANDIDATE_IMAGE" \
          -var="rollback_baseline_image=$ROLLBACK_BASELINE_IMAGE" \
          -out="$plan"; then
          record_cleanup_failure "terraform_${name}_destroy_plan"
      elif ! terraform_capture -chdir="$dir" show -json "$plan" | \
          uv run --frozen python -m elspeth.web.aws_ecs_acceptance sanitize-evidence --kind terraform-destroy-plan >"$receipt"; then
          record_cleanup_failure "terraform_${name}_destroy_projection"
      elif ! plan_sha="$(sha256sum "$plan" | awk '{print $1}')"; then
          record_cleanup_failure "terraform_${name}_destroy_hash"
      elif ! durable_receipt="$(persist_sanitized_receipt "$scenario_id" terraform-destroy-plan "$plan_sha" "$receipt")"; then
          record_cleanup_failure "terraform_${name}_destroy_receipt"
      elif ! approval_receipt="$(require_signed_tf_destroy_approval "$scenario_id" "$durable_receipt")"; then
          record_cleanup_failure "terraform_${name}_destroy_approval"
      elif ! run_cleanup_bounded terraform -chdir="$dir" apply -input=false -lock-timeout=5m "$plan"; then
          record_cleanup_failure "terraform_${name}_destroy_apply"
      fi
      rm -f -- "$plan" "$receipt"
      if ! state_output="$(run_cleanup_bounded terraform -chdir="$dir" state list)"; then
          record_cleanup_failure "terraform_${name}_state_read"
      elif test -n "$state_output"; then
          record_cleanup_failure "terraform_${name}_state_not_empty"
      fi
      if test "${#CLEANUP_FAILURES[@]}" = "$failures_before"; then
          checkpoint_cleanup "terraform_${name}" confirmed
      else
          checkpoint_cleanup "terraform_${name}" failed
      fi
  }
  if cleanup_surface_needed terraform_scenario_a; then
      destroy_acceptance_stack scenario_a A "${SCENARIO_A_TF_DIR:-}" "${SCENARIO_A_TF_VARS:-}" "${SCENARIO_A_TF_BINDING_SHA:-}"
  fi
  if cleanup_surface_needed terraform_scenario_b; then
      destroy_acceptance_stack scenario_b B "${SCENARIO_B_TF_DIR:-}" "${SCENARIO_B_TF_VARS:-}" "${SCENARIO_B_TF_BINDING_SHA:-}"
  fi

  ORPHAN_SWEEP_CONFIRMED=0
  if ! cleanup_surface_needed orphan_sweep; then
      ORPHAN_SWEEP_CONFIRMED=1
  elif run_cleanup_bounded uv run --frozen python -m elspeth.web.aws_ecs_acceptance orphan-sweep --file "$CONTROL_MANIFEST" --acceptance-run-id "${ACCEPTANCE_RUN_ID:-}"; then
      ORPHAN_SWEEP_CONFIRMED=1
      checkpoint_cleanup orphan_sweep confirmed
  else
      record_cleanup_failure orphan_sweep
      checkpoint_cleanup orphan_sweep failed
  fi

  delete_candidate_tag() {
      local label="$1" tag="$2" listing="" count="" result=""
      if test -z "${AWS_REGION:-}" || test -z "${ECR_REPOSITORY:-}" || test -z "$tag"; then
          record_cleanup_failure "ecr_${label}_inputs"
          return
      fi
      if ! listing="$(aws_capture ecr list-images --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" --filter tagStatus=TAGGED --output json)"; then
          record_cleanup_failure "ecr_${label}_list"
          return
      fi
      if ! count="$(jq --arg tag "$tag" '[.imageIds[] | select(.imageTag == $tag)] | length' <<<"$listing")"; then
          record_cleanup_failure "ecr_${label}_list_parse"
          return
      fi
      if test "$count" = "0"; then
          return
      fi
      if test "$count" != "1"; then
          record_cleanup_failure "ecr_${label}_duplicate_tag"
          return
      fi
      if ! result="$(aws_capture ecr batch-delete-image --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" --image-ids imageTag="$tag")"; then
          record_cleanup_failure "ecr_${label}_delete"
      elif ! jq -e '(.failures | length) == 0 and (.imageIds | length) == 1' <<<"$result" >/dev/null; then
          record_cleanup_failure "ecr_${label}_delete_result"
      fi
      if ! listing="$(aws_capture ecr list-images --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" --filter tagStatus=TAGGED --output json)"; then
          record_cleanup_failure "ecr_${label}_post_delete_list"
      elif ! jq -e --arg tag "$tag" '[.imageIds[] | select(.imageTag == $tag)] | length == 0' <<<"$listing" >/dev/null; then
          record_cleanup_failure "ecr_${label}_tag_still_present"
      fi
  }

  cleanup_ecr_surface() {
      local label="$1" tag="$2" failures_before="${#CLEANUP_FAILURES[@]}"
      delete_candidate_tag "$label" "$tag"
      if test "${#CLEANUP_FAILURES[@]}" = "$failures_before"; then
          checkpoint_cleanup "ecr_${label}" confirmed
      else
          checkpoint_cleanup "ecr_${label}" failed
      fi
  }

  if ! cleanup_surface_needed ecr_baseline; then
      :
  elif test -n "${ROLLBACK_BASELINE_TAG:-}"; then
      cleanup_ecr_surface baseline "$ROLLBACK_BASELINE_TAG"
  else
      record_cleanup_failure ecr_baseline_tag
      checkpoint_cleanup ecr_baseline failed
  fi
  if ! cleanup_surface_needed ecr_candidate; then
      :
  elif test -n "${CANDIDATE_TAG:-}"; then
      cleanup_ecr_surface candidate "$CANDIDATE_TAG"
  else
      record_cleanup_failure ecr_candidate_tag
      checkpoint_cleanup ecr_candidate failed
  fi

  docker_image_present() {
      local ref="$1" status=0
      run_cleanup_bounded docker image inspect "$ref" >/dev/null 2>&1 || status=$?
      if test "$status" = "0"; then return 0; fi
      if test "$status" = "124"; then return 2; fi
      if ! run_cleanup_bounded docker info >/dev/null 2>&1; then return 2; fi
      return 1
  }

  if cleanup_surface_needed local_images; then
    DOCKER_FAILURES_BEFORE="${#CLEANUP_FAILURES[@]}"
    if run_cleanup_bounded docker info >/dev/null 2>&1; then
      LOCAL_IMAGE_REFS=(
          "elspeth:ecs-rollback-baseline"
          "elspeth:ecs-0.7.1-closeout"
      )
      if test -n "${ECR_REGISTRY:-}" && test -n "${ECR_REPOSITORY:-}" && test -n "${ROLLBACK_BASELINE_TAG:-}"; then
          LOCAL_IMAGE_REFS+=("$ECR_REGISTRY/$ECR_REPOSITORY:$ROLLBACK_BASELINE_TAG")
      fi
      if test -n "${ROLLBACK_BASELINE_IMAGE:-}"; then LOCAL_IMAGE_REFS+=("$ROLLBACK_BASELINE_IMAGE"); fi
      if test -n "${ECR_REGISTRY:-}" && test -n "${ECR_REPOSITORY:-}" && test -n "${CANDIDATE_TAG:-}"; then
          LOCAL_IMAGE_REFS+=("$ECR_REGISTRY/$ECR_REPOSITORY:$CANDIDATE_TAG")
      fi
      if test -n "${CANDIDATE_IMAGE:-}"; then LOCAL_IMAGE_REFS+=("$CANDIDATE_IMAGE"); fi
      for ref in "${LOCAL_IMAGE_REFS[@]}"; do
          test -n "$ref" || continue
          image_status=0
          if docker_image_present "$ref"; then
              if ! run_cleanup_bounded docker image rm "$ref"; then
                  record_cleanup_failure local_image_remove
              fi
          else
              image_status=$?
              if test "$image_status" = "2"; then record_cleanup_failure local_image_inspect; fi
          fi
      done
      for ref in "${LOCAL_IMAGE_REFS[@]}"; do
          test -n "$ref" || continue
          image_status=0
          if docker_image_present "$ref"; then
              record_cleanup_failure local_image_still_present
          else
              image_status=$?
              if test "$image_status" = "2"; then record_cleanup_failure local_image_verify; fi
          fi
      done
    else
        record_cleanup_failure docker_unavailable
    fi
    if test "${#CLEANUP_FAILURES[@]}" = "$DOCKER_FAILURES_BEFORE"; then
        checkpoint_cleanup local_images confirmed
    else
        checkpoint_cleanup local_images failed
    fi
  fi

  FINAL_EVIDENCE_PREPARED=0
  if run_cleanup_bounded uv run --frozen python -m elspeth.web.aws_ecs_acceptance cleanup-evidence-finalize --file "$CONTROL_MANIFEST" --ledger "$GATE_LEDGER" --phase prepare; then
      FINAL_EVIDENCE_PREPARED=1
      checkpoint_cleanup final_evidence_prepare confirmed
  else
      record_cleanup_failure final_evidence_prepare
      checkpoint_cleanup final_evidence_prepare failed
  fi

  LOCAL_EVIDENCE_FAILURES_BEFORE="${#CLEANUP_FAILURES[@]}"
  if test "${EVIDENCE_EXPORT_CONFIRMED:-0}" = "1" && test "$FINAL_EVIDENCE_PREPARED" = "1"; then
      if test -n "${ACCEPTANCE_STATE:-}"; then
          case "$ACCEPTANCE_STATE" in
              /tmp/*) if ! rm -f -- "$ACCEPTANCE_STATE"; then record_cleanup_failure acceptance_state_remove; fi ;;
              *) record_cleanup_failure acceptance_state_path ;;
          esac
      fi
      if test -n "${OIDC_EVIDENCE_DIR:-}"; then
          case "$OIDC_EVIDENCE_DIR" in
              /tmp/*) if ! rm -rf -- "$OIDC_EVIDENCE_DIR"; then record_cleanup_failure oidc_evidence_remove; fi ;;
              *) record_cleanup_failure oidc_evidence_path ;;
          esac
      fi
  fi
  if test "${#CLEANUP_FAILURES[@]}" = "$LOCAL_EVIDENCE_FAILURES_BEFORE"; then
      checkpoint_cleanup local_evidence confirmed
  else
      checkpoint_cleanup local_evidence failed
  fi

  if ! uv run --frozen python -c 'import sys; from datetime import datetime, timezone; deadline=datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00")); assert datetime.now(timezone.utc) <= deadline' "${ACCEPTANCE_TEARDOWN_DEADLINE_UTC:-}"; then
      if test "${#VERDICT_FAILURES[@]}" = "0"; then VERDICT_FAILURES+=(teardown_deadline); fi
      uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest update --file "$CONTROL_MANIFEST" --verdict-failure teardown_deadline
  else
      checkpoint_cleanup teardown_deadline confirmed
  fi
  test "$ORPHAN_SWEEP_CONFIRMED" = "1" || record_cleanup_failure orphan_sweep_unconfirmed

  if test "${#CLEANUP_FAILURES[@]}" -ne 0; then
      printf 'cleanup failure: %s\n' "${CLEANUP_FAILURES[@]}" >&2
      exit 1
  fi
  if ! run_cleanup_bounded uv run --frozen python -m elspeth.web.aws_ecs_acceptance cleanup-evidence-finalize --file "$CONTROL_MANIFEST" --ledger "$GATE_LEDGER" --phase commit --clear-cleanup-required; then
      record_cleanup_failure final_evidence_commit
      exit 1
  fi
  if ! uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest validate --file "$CONTROL_MANIFEST" --acceptance-run-id "$ACCEPTANCE_RUN_ID" --candidate-sha "$CANDIDATE_SHA" --cleanup-only --require-cleanup-cleared; then
      record_cleanup_failure cleanup_manifest_not_cleared
      exit 1
  fi
  CLEANUP_REQUIRED=0
  if test "${#VERDICT_FAILURES[@]}" -ne 0; then
      printf 'acceptance failure after completed cleanup: %s\n' "${VERDICT_FAILURES[@]}" >&2
      exit 1
  fi
  ```

  Empty Terraform state is necessary but not sufficient after a partial apply.
  Plan 10's tested `orphan-sweep` subcommand performs the following closed
  checks. It uses the Resource Groups Tagging API with
  `ACCEPTANCE_RUN_ID` and explicit service inventories for surfaces it may not
  cover reliably: ECS services/tasks/task definitions, ALB/listener rules/
  target groups, Aurora, EFS/access points, Secrets Manager deletion state,
  CloudWatch log groups/metric namespaces/dashboards/alarms, X-Ray/Transaction
  Search acceptance artifacts, EventBridge rules/targets, the two disposable
  Bedrock Guardrail versions when the inventory owns them, and the Cognito identity.
  Require zero live resources except for AWS's asynchronous ECS
  task-definition deletion lifecycle: require no `ACTIVE` run-scoped revision;
  for each revision, receipts must prove deregistration through `INACTIVE`, a
  delete request, and zero dependent tasks/services, while its current state
  is either absent or `DELETE_IN_PROGRESS`.
  `DELETE_IN_PROGRESS` is not silently treated as gone; record an owner,
  follow-up deadline within 24 hours, and a poll/escalation receipt, while
  forbidding any `ACTIVE` revision at Task 8 completion. The only other
  exception is an inventory-declared shared resource with a signed owner
  restoration receipt; record it without secret values. Any unaccounted
  orphan keeps cleanup failed even when both Terraform states are empty.

  The Terraform stacks own ECS services/task definitions, Aurora databases,
  EFS/access points and acceptance data, ALB/listeners/target groups, Secrets
  Manager test values, CloudWatch logs, and EventBridge targets. Attach the
  aggregate coordinator result, empty-state plus tag/inventory sweep evidence,
  owner receipts, and ECR
  results to the change record. Any recorded failure or missed deadline is
  NO-GO and escalates after all cleanup attempts finish; never restart Task 1
  or issue GO while `CLEANUP_REQUIRED` remains 1.

**Definition of Done:** durable sanitized evidence exists, both Terraform
states are empty, the acceptance-run tag and explicit service sweeps find no
unaccounted or active resources (with any ECS task-definition
`DELETE_IN_PROGRESS` revision carrying the bounded receipt above), the test
identity/secrets are disposed or rotated, both run-scoped ECR tags are deleted,
and cleanup completed on time.

---

### Task 9: Issue the final go/no-go verdict

**Files:**

- Verify: entire integrated worktree.
- No file changes are expected.

- [ ] Confirm that no gate generated or left an uncommitted repository change:

  ```bash
  test -z "$(git status --porcelain)"
  git diff --check
  git rev-parse HEAD
  ```

  **Expected result:** all commands exit 0, and the SHA matches the candidate
  recorded in Task 1.

- [ ] Reconfirm the candidate and hosted-CI boundary, then perform the one
  final fast-forward into the reconciled release branch. This is the only step
  authorized to update release/0.7.1.

  ~~~bash
  set -Eeuo pipefail
  test "$(git branch --show-current)" = "feat/aws-ecs-program"
  test "$(git rev-parse HEAD)" = "$CANDIDATE_SHA"
  test -z "$(git status --porcelain)"

  CI_REF="$(printf 'RC0.7.1-%s' "$(printf '%s' "$CANDIDATE_SHA" | cut -c1-12)")"
  PR_NUMBER="$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest get --file "$CONTROL_MANIFEST" --field release_pr_number)"
  CI_RUN_ID="$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance control-manifest get --file "$CONTROL_MANIFEST" --field ci_run_id)"
  test "$(git ls-remote --heads origin refs/heads/feat/aws-ecs-program | awk '{print $1}')" = "$CANDIDATE_SHA"
  test "$(gh pr view "$PR_NUMBER" --json state,baseRefName,headRefName,headRefOid --jq '.state + ":" + .baseRefName + ":" + .headRefName + ":" + .headRefOid')" = "OPEN:release/0.7.1:feat/aws-ecs-program:$CANDIDATE_SHA"
  test "$(gh run view "$CI_RUN_ID" --json event,headSha,status,conclusion --jq '.event + ":" + .headSha + ":" + .status + ":" + .conclusion')" = "push:$CANDIDATE_SHA:completed:success"
  test -z "$(git ls-remote --heads origin "refs/heads/$CI_REF")"
  test "$(gh release list --limit 100 --json tagName --jq '[.[] | select(.tagName == "v0.7.1")] | length')" = "0"
  test -z "$(git tag --list v0.7.1)"
  test -z "$(git ls-remote --tags origin refs/tags/v0.7.1 'refs/tags/v0.7.1^{}')"

  test "$(git -C /home/john/elspeth branch --show-current)" = "release/0.7.1"
  test -z "$(git -C /home/john/elspeth status --porcelain)"
  test "$(git -C /home/john/elspeth rev-parse HEAD)" = "$RECONCILED_RELEASE_SHA"
  test "$(git -C /home/john/elspeth rev-parse release/0.7.1)" = "$RECONCILED_RELEASE_SHA"
  git -C "$PROGRAM_WORKTREE" merge-base --is-ancestor "$RECONCILED_RELEASE_SHA" "$CANDIDATE_SHA"
  REMOTE_RELEASE_SHA="$(git ls-remote --heads origin refs/heads/release/0.7.1 | awk '{print $1}')"
  test -z "$REMOTE_RELEASE_SHA" || test "$REMOTE_RELEASE_SHA" = "$RECONCILED_RELEASE_SHA"
  git -C /home/john/elspeth merge --ff-only "$CANDIDATE_SHA"
  test "$(git -C /home/john/elspeth rev-parse HEAD)" = "$CANDIDATE_SHA"
  test "$(git -C /home/john/elspeth rev-parse release/0.7.1)" = "$CANDIDATE_SHA"
  test -z "$(git -C /home/john/elspeth status --porcelain)"
  git -C /home/john/elspeth push origin release/0.7.1
  test "$(git ls-remote --heads origin refs/heads/release/0.7.1 | awk '{print $1}')" = "$CANDIDATE_SHA"

  filigree plan elspeth-6343920a47 --json --detail full > /tmp/aws-ecs-final-plan.json
  test "$(jq '[.phases[].steps[] | select(.issue_id != "elspeth-05396fed38")] | length' /tmp/aws-ecs-final-plan.json)" -eq 19
  while read -r anchor; do
      test -n "$anchor"
      [[ "$anchor" =~ ^feat/aws-ecs-program@[0-9a-f]{40}$ ]]
      CLOSE_SHA="$(printf '%s' "$anchor" | cut -d@ -f2)"
      git -C /home/john/elspeth merge-base --is-ancestor "$CLOSE_SHA" release/0.7.1
  done < <(jq -r '.phases[].steps[] | select(.issue_id != "elspeth-05396fed38") | .close_commit' /tmp/aws-ecs-final-plan.json)
  ~~~

  **Expected result:** the program worktree and candidate are unchanged, the
  exact-SHA hosted run remains green, no temporary/ref/tag leak exists, the
  local and remote release branch started at RECONCILED_RELEASE_SHA and now
  equal CANDIDATE_SHA by fast-forward, both worktrees are clean, and all 19
  prerequisite close SHAs are release ancestors. Any failure is NO-GO; do not
  create a merge commit, force-push, or issue a partial verdict.

- [ ] Return **GO** only when every checkbox in Tasks 1–8 is backed by an exit-0
  result or explicit live acceptance evidence bound to that SHA. Report the
  SHA and exact commands run to the release owner, plus the hosted CI run ID
  and URL and the sanitized AWS change-record identifier.
- [ ] State explicitly that GO is runtime-readiness acceptance, not durable
  image or cross-platform artifact acceptance. The recorded rollback baseline
  is a source-qualified synthetic rehearsal artifact, not proof that a
  retained historical production artifact exists. A separate release-owner
  issue/workflow must build the lean extras from this GO SHA for every approved
  platform, emit and verify SBOM/provenance, scan and sign the resulting
  digest, and only then consider a durable tag. For `TARGET_PLATFORM`, the
  rebuilt digest must equal the recorded live-accepted digest; if it differs,
  that digest must rerun Tasks 5–8 before publication. Every other platform
  needs its own artifact and live-runtime acceptance. Plan 12 GO alone cannot
  authorize publication or deployment of a rebuilt/different digest. The
  temporary local/ECR acceptance tags were deleted in Task 8 and must not be
  recreated or retagged as a release shortcut.
- [ ] Return **NO-GO** for any failure, missing operator prerequisite, changed
  SHA, dirty worktree, skipped required command, or incomplete dependency.
  Name the failing command and owning plan/surface, finalize the protected
  ledger, and add a Filigree comment with the sanitized blocker/evidence IDs;
  leave `elspeth-05396fed38` open and assigned for resume. Do not summarize a
  partial run as "go with warnings."
- [ ] On GO only, finalize and checksum the protected gate ledger, add the
  candidate SHA, exact-SHA workflow IDs, AWS change-record ID, cleanup receipt,
  and verdict as a sanitized Filigree comment, then close
  `elspeth-05396fed38` with that reason under `INTEGRATION_OWNER` and exact
  anchor `release/0.7.1@$CANDIDATE_SHA`. Re-read the issue and require done
  status, matching assignee/close commit, and no missing evidence fields. No
  earlier task may close it.

**Definition of Done:**

- [ ] One unchanged `feat/aws-ecs-program` candidate passed every required
  gate, then `release/0.7.1` was fast-forwarded to that exact SHA with no merge
  commit.
- [ ] The exact candidate image passed live AWS deployment, persistence,
  observation, and rollback acceptance.
- [ ] Both disposable scenarios and temporary identities/tags passed the Task 8
  evidence-retention and cleanup gate.
- [ ] The release owner received an unambiguous GO with SHA, command, hosted-CI,
  and live-environment evidence, or a NO-GO naming the exact blocker.
