# AWS ECS Runtime Readiness Orchestration Run Sheet

> **For agentic workers:** REQUIRED SUB-SKILLS: the integration coordinator
> uses `superpowers:subagent-driven-development`; every implementation worker
> uses the exact skills named by its plan. Use `superpowers:using-git-worktrees`
> before dispatch and `superpowers:verification-before-completion` before every
> integration or close. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the complete AWS ECS runtime-readiness program in dependency
order, integrate every slice into `release/0.7.1`, and hand one immutable
candidate to Plan 12 for final live GO/NO-GO acceptance.

**Architecture:** One named integration coordinator owns the release worktree,
live Filigree DAG, serialized merges, close anchors, and stage barriers. Workers
operate only in per-slice `.worktrees/` created from the then-current integrated
release tip. Filigree readiness is necessary but not sufficient: a dependency
is consumable only when its close commit is an ancestor of `release/0.7.1` and
its integrated handoff gates are green.

**Tech Stack:** Git worktrees, Filigree milestone
`elspeth-6343920a47`, Python/uv, Node/npm, Docker/testcontainers, Wardline,
ELSPETH trust-tier gates, Terraform, AWS ECS/Aurora/EFS/ALB/Cognito/S3/Bedrock,
CloudWatch/X-Ray, and the plan-specific verification commands.

---

## 1. What the orchestrator is executing

The folder contains 18 dated AWS program-plan Markdown files plus this run
sheet:

- Plan 00 is the non-executable index.
- Plans 01–14 plus 15A/15B/15C are 17 executable plan documents.
- Plan 03 maps to two tracker slices: 03A (Tasks 1–2) and 03B (Tasks 3–4,
  including the full Plan 03 handoff).
- Plan 08 maps to three tracker slices: verification baseline (Task 0), 08A
  (Tasks 1–3), and 08B (Task 4).
- Plan 12 is the final closeout, not ordinary parallel implementation.

The resulting live milestone has 20 tracker steps: 19 prerequisites before
Plan 12 (including the non-application verification-baseline task), then Plan
12 itself. Never infer execution order from filenames, Filigree phase display,
or the words “Wave 1”; use the hard dependencies below.

Primary orientation files:

- [Plan 00 index](2026-07-08-aws-ecs-00-overview.md)
- [Aggregate plan review](2026-07-08-aws-ecs.review.json)
- [Runtime-readiness design](../../specs/2026-07-08-aws-ecs-runtime-readiness-design.md)
- [Universal web-plugin-policy design](../../specs/2026-07-12-universal-web-plugin-policy-design.md)

Planning approval means `GO_TO_IMPLEMENT`. It is not runtime approval. Only a
successful Plan 12 closeout may issue AWS runtime GO.

### Ready-to-paste orchestration directive

> Work in `/home/john/elspeth` as the sole AWS ECS integration coordinator.
> Read `AGENTS.md` and this run sheet completely, then execute milestone
> `elspeth-6343920a47` stage by stage. Dispatch only the exact issue IDs listed
> here, create each worker in an ignored `.worktrees/` worktree from the latest
> integrated `release/0.7.1` tip, and never close a slice before its commit and
> handoff gates are integrated. Refresh and validate the live DAG before every
> dispatch. Preserve the 02→06→07 lockfile chain and every security/audit gate.
> Stop on any condition in section 11 rather than guessing or weakening a gate.
> Run Plan 10 and Plan 12 exclusively under their special checkpoint rules.
> Continue until Plan 12 returns an evidence-backed GO or a precise NO-GO with
> cleanup complete and the owning slice reopened.

## 2. Authority and non-negotiable rules

1. **One coordinator:** set one stable coordinator identity for the entire run.
   The coordinator alone mutates `release/0.7.1`, integrates branches, and
   closes tracker slices.
2. **Exact IDs only:** this repository currently has hundreds of unrelated
   ready issues. Never use `start-next-work`, numeric “next plan” assumptions,
   or a generic ready-queue worker. Dispatch only the IDs in this document.
3. **Atomic claims:** use `filigree start-work`, never `claim` followed by a
   status update. A conflict means another owner won; stop rather than steal.
4. **Integration precedes closure:** workers commit and report evidence but do
   not close from feature branches. The coordinator closes only after the
   implementation is in `release/0.7.1` and integrated gates pass.
5. **Live ancestry:** a done dependency whose `close_commit` is not an ancestor
   of the current release tip is not satisfied. Stop and repair the tracker or
   integration state.
6. **Fresh worktrees:** create each worktree only when its slice is dispatched,
   from the then-current integrated tip. Do not pre-create all stage branches.
7. **Preserve user work:** never stash, reset, checkout-over, or broad-stage
   unrelated changes. Use the exact staging allowlists in each plan.
8. **No lockfile text merges:** `uv.lock` ownership is strictly Plan 02 → Plan
   06 → Plan 07. Each owner rebases on the previous integrated lock and
   regenerates it.
9. **Audit before telemetry:** Landscape remains authoritative. No stage may
   weaken audit ordering, replace audit evidence with telemetry, or retain raw
   provider/user content in operational evidence.
10. **No security bypasses:** no waiver, baseline, allowlist broadening,
    threshold reduction, skipped mandatory lane, or partial Wardline scan may
    turn a failing gate green.
11. **Operator-held authority stays operator-held:** agents diagnose signed
    trust metadata but never receive signing keys or self-sign repairs.
12. **Current-scope defects stay in scope:** do not hide an implementation gap
    as a short-lived observation. Fix it, expand the issue, or add a real
    dependency/issue before continuing.
13. **The run sheet owns orchestration mechanics:** for slices other than Plan
    12, Sections 5 and 6 supersede any plan checkbox that claims/releases/closes
    a Filigree issue, chooses a worker identity, creates/removes a worktree, or
    chooses a branch/base SHA. Those bootstrap/handoff checkboxes are satisfied
    by following this run sheet and must not be executed twice. Every plan
    implementation, test, evidence, staging-allowlist, and commit checkbox
    remains binding. Plan 12 alone follows its own coordinator-owned Task 1
    claim/resume and final close instructions.

## 3. Coordinator bootstrap

- [ ] **Step 1: Stabilize the planning commit before implementation**

  The relocation, this run sheet, tracker path updates, and dependency repairs
  must be committed and present on `release/0.7.1` before any implementation
  worktree is created. Record that SHA as `PLAN_SET_SHA`; it is not the later
  Plan-10 rollback baseline.

- [ ] **Step 2: Establish coordinator identity and clean release state**

  ```bash
  set -Eeuo pipefail
  umask 077
  cd /home/john/elspeth
  export AWS_PAGER=""
  export AWS_ECS_COORDINATOR="${AWS_ECS_COORDINATOR:?set the named integration coordinator}"
  test "$(git branch --show-current)" = "release/0.7.1"
  test -z "$(git status --porcelain)"
  git check-ignore -q .worktrees
  export PLAN_SET_SHA="$(git rev-parse HEAD)"
  ```

  Expected: the coordinator owns a clean release worktree, `.worktrees/` is
  ignored, and the immutable planning baseline is recorded.

  Every coordinator and worker command block runs in Bash with
  `set -Eeuo pipefail` and `umask 077`; AWS-capable blocks also export
  `AWS_PAGER=""`. Treat each complete numbered protocol (Section 5 dispatch,
  Section 6 integration, and Stage 8 Plan 10 setup) as one persistent Bash
  session; fenced blocks are readability breaks, not new shells. Exports do not
  cross agent boundaries. At session start, explicitly pass or redeclare
  `AWS_ECS_COORDINATOR`, `WORKER_ID`, `ISSUE_ID`, `PLAN_BASE`, `PLAN_FILE`,
  `REVIEW_FILE`, `PLAN_SET_SHA`, `WORKTREE_PATH`, `WORKTREE_NAME`, and
  `BRANCH_NAME`. If a shell is interrupted, restart the protocol's documented
  exact-owner resume path and reconstruct values from Filigree/Git; never rely
  on a vanished environment variable.

- [ ] **Step 3: Snapshot live tracker state**

  ```bash
  filigree session-context
  filigree plan elspeth-6343920a47 --json --detail full > /tmp/aws-ecs-plan.json
  filigree critical-path --json > /tmp/aws-ecs-critical-path.json
  jq -e '[.phases[].steps[]] | length == 20' /tmp/aws-ecs-plan.json
  ```

  Expected: 20 milestone steps. Temporary tracker snapshots are diagnostic only
  and must not be committed or treated as durable acceptance evidence.

  If the milestone is still `planning`, atomically start its lifecycle under
  coordinator ownership before dispatching children; otherwise require an
  exact-owner resume:

  ```bash
  MILESTONE_STATUS="$(jq -r '.milestone.status' /tmp/aws-ecs-plan.json)"
  if test "$MILESTONE_STATUS" = planning; then
      filigree start-work elspeth-6343920a47 \
        --target-status active \
        --assignee "$AWS_ECS_COORDINATOR" \
        --actor "$AWS_ECS_COORDINATOR" \
        --commit "release/0.7.1@$PLAN_SET_SHA"
  else
      test "$MILESTONE_STATUS" = active
      jq -e --arg owner "$AWS_ECS_COORDINATOR" \
        '.milestone.assignee == $owner' /tmp/aws-ecs-plan.json >/dev/null
  fi
  ```

  Validate every direct dependency rather than only the step count:

  ```bash
  uv run --frozen python - <<'PY'
  import json
  from pathlib import Path

  expected = {
      "elspeth-8166b310e7": set(),
      "elspeth-b9e8b5d24b": set(),
      "elspeth-9070fb0a45": set(),
      "elspeth-c0103e6c88": {"elspeth-8166b310e7"},
      "elspeth-130dc48252": {"elspeth-8166b310e7"},
      "elspeth-dffe064287": {"elspeth-b9e8b5d24b", "elspeth-9070fb0a45"},
      "elspeth-03cf981d4a": {"elspeth-b9e8b5d24b", "elspeth-9070fb0a45"},
      "elspeth-5e729216f4": {"elspeth-b9e8b5d24b", "elspeth-8166b310e7"},
      "elspeth-7fe6aa531f": {"elspeth-9070fb0a45", "elspeth-c0103e6c88"},
      "elspeth-1a1c31bcce": {"elspeth-03cf981d4a"},
      "elspeth-25286192ee": {"elspeth-b9e8b5d24b", "elspeth-9070fb0a45", "elspeth-03cf981d4a"},
      "elspeth-f5d5dddddf": {"elspeth-5e729216f4", "elspeth-8166b310e7"},
      "elspeth-74717426b7": {"elspeth-7fe6aa531f", "elspeth-c0103e6c88"},
      "elspeth-a342f333a4": {"elspeth-7fe6aa531f", "elspeth-c0103e6c88"},
      "elspeth-e8dc754360": {"elspeth-7fe6aa531f"},
      "elspeth-0674a06468": {
          "elspeth-130dc48252", "elspeth-1a1c31bcce", "elspeth-c0103e6c88",
          "elspeth-e8dc754360", "elspeth-25286192ee", "elspeth-f5d5dddddf",
      },
      "elspeth-7d1f35e3d8": {"elspeth-0674a06468", "elspeth-7fe6aa531f"},
      "elspeth-397ac915b8": {
          "elspeth-dffe064287", "elspeth-7fe6aa531f", "elspeth-74717426b7",
          "elspeth-e8dc754360", "elspeth-f5d5dddddf", "elspeth-7d1f35e3d8",
      },
      "elspeth-6285c29c07": {
          "elspeth-9070fb0a45", "elspeth-dffe064287", "elspeth-397ac915b8",
          "elspeth-03cf981d4a", "elspeth-1a1c31bcce", "elspeth-7fe6aa531f",
          "elspeth-74717426b7", "elspeth-a342f333a4", "elspeth-e8dc754360",
          "elspeth-25286192ee", "elspeth-5e729216f4", "elspeth-f5d5dddddf",
          "elspeth-7d1f35e3d8",
      },
      "elspeth-05396fed38": {
          "elspeth-397ac915b8", "elspeth-1a1c31bcce", "elspeth-a342f333a4",
          "elspeth-6285c29c07", "elspeth-25286192ee", "elspeth-5e729216f4",
          "elspeth-f5d5dddddf", "elspeth-7d1f35e3d8",
      },
  }

  payload = json.loads(Path("/tmp/aws-ecs-plan.json").read_text(encoding="utf-8"))
  steps = [step for phase in payload["phases"] for step in phase["steps"]]
  actual = {step["issue_id"]: set(step.get("blocked_by", [])) for step in steps}
  assert actual == expected, {
      issue_id: {"expected": sorted(expected.get(issue_id, set())), "actual": sorted(actual.get(issue_id, set()))}
      for issue_id in sorted(set(actual) | set(expected))
      if actual.get(issue_id, set()) != expected.get(issue_id, set())
  }
  PY
  ```

  Expected: exit 0. Any difference is a contract change requiring review, not
  an instruction to silently update the run sheet.

- [ ] **Step 4: Verify plan and review inventory**

  ```bash
  test "$(find docs/superpowers/plans/aws -maxdepth 1 -name '2026-07-08-aws-ecs-*.md' | wc -l)" -eq 18
  test "$(find docs/superpowers/plans/aws -maxdepth 1 -name '2026-07-08-aws-ecs-*.review.json' | wc -l)" -eq 18
  find docs/superpowers/plans/aws -maxdepth 1 -name '*.review.json' -print0 | xargs -0 -n1 jq -e . >/dev/null
  jq -e '.scope_verdict == "GO_TO_IMPLEMENT" and .runtime_verdict == "PENDING_PLAN_12"' \
    docs/superpowers/plans/aws/2026-07-08-aws-ecs.review.json
  ```

  Expected: all plans and their review sidecars are readable, and runtime GO is
  still explicitly pending Plan 12.

- [ ] **Step 5: Confirm external authority before dependent stages**

  Name the owners and verify availability for:

  - signed trust-metadata repair;
  - Docker/testcontainers;
  - GitHub CI and temporary exact-SHA refs;
  - PostgreSQL/Aurora schema ownership and retention approval;
  - AWS/Terraform/ECR/ECS/EFS/ALB/Cognito/S3/Bedrock/CloudWatch permissions;
  - protected evidence storage and emergency cleanup.

  Persist only identity labels and booleans in a protected coordinator
  checkpoint, never credentials, account details, URLs, or ARNs:

  ```bash
  export AUTHORITY_MATRIX="${AUTHORITY_MATRIX:?set protected mode-0600 authority matrix path}"
  jq -e '
    .schema == "elspeth.aws-ecs-authority-matrix.v1" and
    (.authorities | type == "array" and length == 7) and
    all(.authorities[];
      (keys | sort) == ["authorized","available","owner_label","surface"] and
      (.surface | IN("trust_signing","docker","github_ci","database_retention","aws_runtime","evidence_storage","cleanup")) and
      (.owner_label | type == "string" and test("^[A-Za-z0-9._-]{1,64}$")) and
      (.authorized | type == "boolean") and (.available | type == "boolean")
    ) and
    ([.authorities[].surface] | unique | length == 7)
  ' "$AUTHORITY_MATRIX" >/dev/null
  test "$(stat -c '%a' "$AUTHORITY_MATRIX")" = "600"
  ```

  Missing authority does not block early local stages, but the coordinator
  must revalidate this checkpoint and require `authorized and available` for
  each surface before dispatching its first dependent lane. Never substitute
  mocks for a mandatory live gate.

## 4. Exact hard-edge DAG

| Slice | Filigree ID | Plan scope | Direct prerequisites |
|---|---|---|---|
| Baseline | `elspeth-8166b310e7` | [Plan 08 Task 0](2026-07-08-aws-ecs-08-s3-endpoint-gate.md) | none |
| 01 | `elspeth-b9e8b5d24b` | [Plan 01](2026-07-08-aws-ecs-01-deployment-contract.md) | none |
| 02 | `elspeth-9070fb0a45` | [Plan 02](2026-07-08-aws-ecs-02-postgres-schema-support.md) | none |
| 08A | `elspeth-c0103e6c88` | [Plan 08 Tasks 1–3](2026-07-08-aws-ecs-08-s3-endpoint-gate.md) | Baseline |
| 15A | `elspeth-130dc48252` | [Plan 15A](2026-07-08-aws-ecs-15a-text-sink.md) | Baseline |
| 03A | `elspeth-dffe064287` | [Plan 03 Tasks 1–2](2026-07-08-aws-ecs-03-doctor-cli.md) | 01, 02 |
| 04 | `elspeth-03cf981d4a` | [Plan 04](2026-07-08-aws-ecs-04-validate-only-startup.md) | 01, 02 |
| 13 | `elspeth-5e729216f4` | [Plan 13](2026-07-08-aws-ecs-13-cognito-authorization-origin.md) | Baseline, 01 |
| 06 | `elspeth-7fe6aa531f` | [Plan 06](2026-07-08-aws-ecs-06-s3-source.md) | 02, 08A |
| 05 | `elspeth-1a1c31bcce` | [Plan 05](2026-07-08-aws-ecs-05-readiness-endpoint.md) | 04 |
| 11 | `elspeth-25286192ee` | [Plan 11](2026-07-08-aws-ecs-11-landscape-write-gate.md) | 01, 02, 04 |
| 14 | `elspeth-f5d5dddddf` | [Plan 14](2026-07-08-aws-ecs-14-cloudwatch-operator-telemetry.md) | Baseline, 13 |
| 07 | `elspeth-74717426b7` | [Plan 07](2026-07-08-aws-ecs-07-s3-sink.md) | 06, 08A |
| 08B | `elspeth-a342f333a4` | [Plan 08 Task 4](2026-07-08-aws-ecs-08-s3-endpoint-gate.md) | 06, 08A |
| 09 | `elspeth-e8dc754360` | [Plan 09](2026-07-08-aws-ecs-09-bedrock-provider.md) | 06 |
| 15B | `elspeth-0674a06468` | [Plan 15B](2026-07-08-aws-ecs-15b-universal-web-plugin-policy.md) | 15A, 05, 08A, 09, 11, 14 |
| 15C | `elspeth-7d1f35e3d8` | [Plan 15C](2026-07-08-aws-ecs-15c-bedrock-guardrail-shields.md) | 15B, 06 |
| 03B | `elspeth-397ac915b8` | [Plan 03 Tasks 3–4](2026-07-08-aws-ecs-03-doctor-cli.md) | 03A, 06, 07, 09, 14, 15C |
| 10 | `elspeth-6285c29c07` | [Plan 10](2026-07-08-aws-ecs-10-packaging-docker.md) | 02, 03A, 03B, 04, 05, 06, 07, 08B, 09, 11, 13, 14, 15C |
| 12 | `elspeth-05396fed38` | [Plan 12](2026-07-08-aws-ecs-12-integration-closeout.md) | 03B, 05, 08B, 10, 11, 13, 14, 15C |

The expected critical path is:

```text
Baseline -> 08A -> 06 -> 09 -> 15B -> 15C -> 03B -> 10 -> 12
```

If live Filigree differs from this table, stop. Determine whether the reviewed
documents or tracker is stale, repair the authoritative contract, and rerun the
plan review before dispatching affected descendants.

### Dispatch manifest

| Key | Worktree | Branch | Plan/review base name | Assigned scope |
|---|---|---|---|---|
| baseline | `.worktrees/aws-ecs-baseline` | `ops/aws-ecs-baseline` | `2026-07-08-aws-ecs-08-s3-endpoint-gate` | Task 0 |
| 01 | `.worktrees/aws-ecs-01` | `feat/aws-ecs-01-deployment-contract` | `2026-07-08-aws-ecs-01-deployment-contract` | all tasks |
| 02 | `.worktrees/aws-ecs-02` | `feat/aws-ecs-02-postgres` | `2026-07-08-aws-ecs-02-postgres-schema-support` | all tasks |
| 08A | `.worktrees/aws-ecs-08a` | `feat/aws-ecs-08a-endpoint-gate` | `2026-07-08-aws-ecs-08-s3-endpoint-gate` | Tasks 1–3 |
| 15A | `.worktrees/aws-ecs-15a` | `feat/aws-ecs-15a-text-sink` | `2026-07-08-aws-ecs-15a-text-sink` | all tasks |
| 03A | `.worktrees/aws-ecs-03a` | `feat/aws-ecs-03a-doctor-base` | `2026-07-08-aws-ecs-03-doctor-cli` | Tasks 1–2 |
| 04 | `.worktrees/aws-ecs-04` | `feat/aws-ecs-04-validate-startup` | `2026-07-08-aws-ecs-04-validate-only-startup` | all tasks |
| 13 | `.worktrees/aws-ecs-13` | `feat/aws-ecs-13-cognito-pkce` | `2026-07-08-aws-ecs-13-cognito-authorization-origin` | all tasks |
| 06 | `.worktrees/aws-ecs-06` | `feat/aws-ecs-06-s3-source` | `2026-07-08-aws-ecs-06-s3-source` | all tasks |
| 05 | `.worktrees/aws-ecs-05` | `feat/aws-ecs-05-readiness` | `2026-07-08-aws-ecs-05-readiness-endpoint` | all tasks |
| 11 | `.worktrees/aws-ecs-11` | `feat/aws-ecs-11-landscape-gate` | `2026-07-08-aws-ecs-11-landscape-write-gate` | all tasks |
| 14 | `.worktrees/aws-ecs-14` | `feat/aws-ecs-14-cloudwatch-otlp` | `2026-07-08-aws-ecs-14-cloudwatch-operator-telemetry` | all tasks |
| 07 | `.worktrees/aws-ecs-07` | `feat/aws-ecs-07-s3-sink` | `2026-07-08-aws-ecs-07-s3-sink` | all tasks |
| 08B | `.worktrees/aws-ecs-08b` | `feat/aws-ecs-08b-guided-s3` | `2026-07-08-aws-ecs-08-s3-endpoint-gate` | Task 4 |
| 09 | `.worktrees/aws-ecs-09` | `feat/aws-ecs-09-bedrock-provider` | `2026-07-08-aws-ecs-09-bedrock-provider` | all tasks |
| 15B | `.worktrees/aws-ecs-15b` | `feat/aws-ecs-15b-plugin-policy` | `2026-07-08-aws-ecs-15b-universal-web-plugin-policy` | all tasks |
| 15C | `.worktrees/aws-ecs-15c` | `feat/aws-ecs-15c-guardrails` | `2026-07-08-aws-ecs-15c-bedrock-guardrail-shields` | all tasks |
| 03B | `.worktrees/aws-ecs-03b` | `test/aws-ecs-03b-doctor-postgres` | `2026-07-08-aws-ecs-03-doctor-cli` | Tasks 3–4 |
| 10 | `.worktrees/aws-ecs-10` | `feat/aws-ecs-10-packaging` | `2026-07-08-aws-ecs-10-packaging-docker` | all tasks |
| 12 | N/A: coordinator release worktree | `release/0.7.1` | `2026-07-08-aws-ecs-12-integration-closeout` | all tasks; Plan 12 protocol only |

For a row, set:

```bash
PLAN_FILE="docs/superpowers/plans/aws/${PLAN_BASE}.md"
REVIEW_FILE="docs/superpowers/plans/aws/${PLAN_BASE}.review.json"
WORKTREE_NAME="${WORKTREE_PATH#.worktrees/}"
```

`ISSUE_ID`, `PLAN_BASE`, `WORKTREE_PATH`, and `BRANCH_NAME` come from that
exact row and the DAG table; `WORKER_ID` is the named agent assigned by the
coordinator. These are typed dispatch inputs, not free-form choices. The Plan
12 row deliberately has no worker worktree inputs and is never passed to the
generic worker or integration protocols.

## 5. Worker dispatch protocol

Use this protocol for every slice except Plan 12. Replace shell variables with
the exact row from the DAG table; do not hand-edit the commands themselves.
This protocol has precedence over plan-local claim, release, worktree, branch,
and close commands as stated in rule 13. Do not repeat those plan-local
orchestration commands while executing Step 6.

- [ ] **Step 1: Read the exact plan slice and review artifact**

  For split plans, execute only the tasks assigned to the tracker slice. Verify
  the sibling review verdict is approved before claiming:

  ```bash
  BASE_SHA="$(git rev-parse release/0.7.1)"
  test -z "$(git status --porcelain -- docs/superpowers/plans/aws docs/superpowers/specs)"
  git merge-base --is-ancestor "$PLAN_SET_SHA" "$BASE_SHA"
  PLAN_SHA256="$(sha256sum "$PLAN_FILE" | awk '{print $1}')"
  jq -e --arg sha "$PLAN_SHA256" \
    '(.verdict | startswith("APPROVED")) and .plan_sha256 == $sha' \
    "$REVIEW_FILE"
  ```

  A checksum mismatch means the plan changed after review. Stop, run the plan
  review again, update its sidecar, commit those reviewed planning changes on
  `release/0.7.1`, and advance the coordinator's recorded `PLAN_SET_SHA` before
  dispatch. This applies when an earlier implementation plan intentionally
  edits a later AWS plan; an old approved verdict never survives plan drift.

- [ ] **Step 2: Require live readiness and integrated dependency ancestry**

  ```bash
  cd /home/john/elspeth
  ISSUE_JSON="$(filigree show "$ISSUE_ID" --json)"
  while read -r dependency_id; do
      DEP_JSON="$(filigree show "$dependency_id" --json)"
      jq -e '.status_category == "done" and (.close_commit | type == "string" and length > 0)' <<<"$DEP_JSON" >/dev/null
      DEP_SHA="$(jq -r '.close_commit | split("@")[-1]' <<<"$DEP_JSON")"
      git merge-base --is-ancestor "$DEP_SHA" "$BASE_SHA"
  done < <(jq -r '.blocked_by[]' <<<"$ISSUE_JSON")
  ```

  Expected: every direct dependency is done and anchored in the current release
  tip. An empty/malformed anchor or failed ancestry check is a stop condition.

  The verification-baseline close is historical evidence, not a permanent
  waiver. Before every baseline-dependent dispatch, and again after integrating
  a change to trust-boundary-sensitive source, validate the current release
  tip in the clean coordinator worktree:

  ```bash
  case "$ISSUE_ID" in
    elspeth-c0103e6c88|elspeth-130dc48252|elspeth-5e729216f4|elspeth-7fe6aa531f|elspeth-f5d5dddddf|elspeth-74717426b7|elspeth-a342f333a4|elspeth-e8dc754360|elspeth-0674a06468|elspeth-7d1f35e3d8|elspeth-397ac915b8|elspeth-6285c29c07)
      ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing \
        uv run elspeth-lints check --rules trust_tier.tier_model --root src/elspeth
      PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check \
        --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier \
        --root src/elspeth
      wardline scan . --fail-on ERROR
      ;;
  esac
  ```

  Signed binding drift returns to the named operator; Wardline exit 1 requires
  a boundary repair and exit 2 is a tool failure. Neither rewrites the
  historical baseline issue.

- [ ] **Step 3: Atomically start or validate an exact-owner resume**

  The coordinator first atomically starts the issue's parent phase if it is
  still pending; otherwise it requires exact-owner resume. Only the exact phase
  IDs from the live 20-node plan are allowed:

  ```bash
  ISSUE_JSON="$(filigree show "$ISSUE_ID" --json)"
  PHASE_ID="$(jq -r .parent_id <<<"$ISSUE_JSON")"
  case "$PHASE_ID" in
    elspeth-db28fb3293|elspeth-da8e5447c8|elspeth-ab8f9e11d9|elspeth-ef74690eb2) ;;
    *) exit 1 ;;
  esac
  PHASE_STATUS="$(filigree show "$PHASE_ID" --json | jq -r .status)"
  if test "$PHASE_STATUS" = pending; then
      filigree start-work "$PHASE_ID" \
        --target-status active \
        --assignee "$AWS_ECS_COORDINATOR" \
        --actor "$AWS_ECS_COORDINATOR" \
        --commit "release/0.7.1@$BASE_SHA"
  else
      test "$PHASE_STATUS" = active
      filigree show "$PHASE_ID" --json | jq -e \
        --arg owner "$AWS_ECS_COORDINATOR" '.assignee == $owner' >/dev/null
  fi
  ```

  ```bash
  ISSUE_JSON="$(filigree show "$ISSUE_ID" --json)"
  if jq -e '.is_ready == true and ((.assignee // "") == "")' <<<"$ISSUE_JSON" >/dev/null; then
      filigree start-work "$ISSUE_ID" \
        --assignee "$WORKER_ID" \
        --actor "$WORKER_ID" \
        --commit "release/0.7.1@$BASE_SHA"
      FRESH_DISPATCH=1
      WORKTREE_BASE_SHA="$BASE_SHA"
  else
      jq -e --arg worker "$WORKER_ID" \
        '.status_category == "wip" and .assignee == $worker and
         (.claim_commit | type == "string" and test("^release/0\\.7\\.1@[0-9a-f]{40}$"))' \
        <<<"$ISSUE_JSON" >/dev/null
      WORKTREE_BASE_SHA="$(jq -r '.claim_commit | split("@")[-1]' <<<"$ISSUE_JSON")"
      git cat-file -e "$WORKTREE_BASE_SHA^{commit}"
      git merge-base --is-ancestor "$WORKTREE_BASE_SHA" "$BASE_SHA"
      FRESH_DISPATCH=0
  fi
  ```

  Expected: a fresh issue enters its working status under `WORKER_ID`, or an
  interrupted dispatch resumes only for the exact assignee and its original
  well-formed claim anchor, which must remain an ancestor of the latest release
  tip. A sibling merge after the claim does not rewrite that historical anchor.
  Any other state is a conflict and stops dispatch.

- [ ] **Step 4: Create the worktree after the claim**

  ```bash
  git check-ignore -q .worktrees
  if test "$FRESH_DISPATCH" = 1; then
      test ! -e ".worktrees/$WORKTREE_NAME"
      ! git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"
      git worktree add ".worktrees/$WORKTREE_NAME" -b "$BRANCH_NAME" "$WORKTREE_BASE_SHA"
  else
      if test -d ".worktrees/$WORKTREE_NAME"; then
          test "$(git -C ".worktrees/$WORKTREE_NAME" branch --show-current)" = "$BRANCH_NAME"
          git -C ".worktrees/$WORKTREE_NAME" merge-base --is-ancestor "$WORKTREE_BASE_SHA" HEAD
      elif git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
          git merge-base --is-ancestor "$WORKTREE_BASE_SHA" "$BRANCH_NAME"
          git worktree add ".worktrees/$WORKTREE_NAME" "$BRANCH_NAME"
      else
          git worktree add ".worktrees/$WORKTREE_NAME" -b "$BRANCH_NAME" "$WORKTREE_BASE_SHA"
      fi
  fi
  cd ".worktrees/$WORKTREE_NAME"
  ```

  If worktree/setup creation fails, keep the exact claim so the partial-state
  resume path can recover it. Add a sanitized failure checkpoint and stop:

  ```bash
  filigree add-comment "$ISSUE_ID" \
    '{"schema":"elspeth.aws-ecs-setup-failure.v1","stage":"worktree_or_baseline_setup","retry":"exact-owner-resume"}' \
    --expected-assignee "$WORKER_ID" \
    --actor "$AWS_ECS_COORDINATOR"
  exit 1
  ```

- [ ] **Step 5: Establish the plan-specific baseline**

  For a fresh dispatch, establish deterministic worktree-local dependencies
  before the plan's exact initial tests:

  ```bash
  if test "$FRESH_DISPATCH" = 1; then
      uv sync --frozen --all-extras
      if rg -q 'npm (--prefix src/elspeth/web/frontend|ci|run|test)|src/elspeth/web/frontend/' "$PLAN_FILE"; then
          npm --prefix src/elspeth/web/frontend ci
      fi
      test -z "$(git status --porcelain)"
  fi
  ```

  Then, for a fresh dispatch, run the exact initial tests and remaining setup in
  the plan. A failing baseline is not permission to continue. On exact-owner
  resume, preserve any in-progress dirty changes and resume from the last
  sanitized Filigree task-checkpoint comment; do not rerun destructive setup or
  discard partial work. If the checkpoint and worktree disagree, stop for
  coordinator diagnosis.

  Refresh claim liveness after every completed plan task and at least once per
  hour during long tests:

  ```bash
  filigree heartbeat-work "$ISSUE_ID" \
    --expected-assignee "$WORKER_ID" \
    --lease-hours 4 \
    --actor "$WORKER_ID"
  ```

  After each completed plan task, add a sanitized checkpoint comment containing
  only schema `elspeth.aws-ecs-task-checkpoint.v1`, the stable task ID, current
  commit SHA, and `complete: true`. This is the resume cursor; it contains no
  command, output, path, URL, or provider/user content.

  The coordinator audits assignee and `claim_expires_at` before integration;
  an expired or changed claim is a stop condition, not permission to overwrite
  the owner.

- [ ] **Step 6: Execute every assigned checkbox**

  Follow every assigned implementation, test, evidence, staging-allowlist, and
  commit checkbox exactly. Treat any plan-local claim, release, worktree,
  branch/base selection, or close checkbox as already fulfilled by Sections 5
  and 6; do not execute it again. Follow TDD, static gates, signed-tier
  diagnosis, and Wardline instructions exactly. Mandatory Docker/live lanes
  must execute with zero skips where the plan says so. The worker may create
  additional commits to repair in-scope defects but must not widen unrelated
  scope.

- [ ] **Step 7: Report worker evidence without closing**

  Require a clean feature worktree. Create a bounded verification summary at
  `/tmp/${ISSUE_ID}-verification.json`; it must be a non-empty JSON array with
  one object per stable plan check ID and only these fields:

  ```json
  [{"check_id":"task1.unit-tests","exit_code":0,"collected":42,"passed":42,"skipped":0}]
  ```

  `check_id` must match `^[a-z0-9][a-z0-9._-]{0,79}$`. Every completed check
  has `exit_code == 0`, `skipped == 0`, and `passed == collected`; a non-test
  gate uses zero for all three counts. IDs are unique. Deferred mandatory proof
  is not a completed handoff and cannot appear here. Do not include commands,
  arguments, output, exception text, paths, URLs, provider/user content, or
  additional fields. Validate it and add the structured array to a sanitized
  comment:

  Derive IDs mechanically from reviewed plan order: the Mth checkbox under
  `Task N` is `taskN.checkM`; the Mth plan-level handoff checkbox is
  `handoff.checkM`. Because plan checksum drift blocks dispatch, these ordinals
  are stable for the reviewed plan. Test checks record their collected/passed
  counts; successful non-test gates (Ruff, mypy, Docker, Wardline, scripts)
  record `collected=0`, `passed=0`, and `skipped=0`.

  ```bash
  test -z "$(git status --porcelain)"
  IMPLEMENTATION_SHA="$(git rev-parse HEAD)"
  VERIFICATION_FILE="/tmp/${ISSUE_ID}-verification.json"
  jq -e '
    type == "array" and length > 0 and
    all(.[];
      (keys | sort) == ["check_id","collected","exit_code","passed","skipped"] and
      (.check_id | type == "string" and test("^[a-z0-9][a-z0-9._-]{0,79}$")) and
      ([.exit_code,.collected,.passed,.skipped] |
        all(type == "number" and floor == . and . >= 0)) and
      (.exit_code == 0 and .skipped == 0 and .passed == .collected)
    ) and
    ([.[].check_id] | length == (unique | length))
  ' "$VERIFICATION_FILE" >/dev/null
  EVIDENCE="$(jq -cn \
    --arg schema "elspeth.aws-ecs-worker-handoff.v1" \
    --arg plan "$PLAN_FILE" \
    --arg worktree_base_sha "$WORKTREE_BASE_SHA" \
    --arg release_tip_at_handoff "$(git rev-parse release/0.7.1)" \
    --arg implementation_sha "$IMPLEMENTATION_SHA" \
    --argjson verification "$(<"$VERIFICATION_FILE")" \
    '{schema:$schema,plan:$plan,worktree_base_sha:$worktree_base_sha,release_tip_at_handoff:$release_tip_at_handoff,implementation_sha:$implementation_sha,verification:$verification}')"
  filigree add-comment "$ISSUE_ID" "$EVIDENCE" \
    --expected-assignee "$WORKER_ID" \
    --actor "$WORKER_ID"
  ```

  Stable check IDs identify plan gates without reproducing their commands. Raw
  logs remain in the plan-approved protected evidence surface, never in the
  tracker comment.

## 6. Coordinator integration and close protocol

Use this protocol for every slice except Plan 12. Plan 12 runs directly in the
coordinator's checked-out `release/0.7.1` worktree and owns its Task 1
claim/resume, candidate commit, evidence ledger, cleanup, and final close.

- [ ] **Step 1: Recheck ownership and release cleanliness**

  ```bash
  cd /home/john/elspeth
  test "$(git branch --show-current)" = "release/0.7.1"
  test -z "$(git status --porcelain)"
  IMPLEMENTATION_SHA="$(git -C ".worktrees/$WORKTREE_NAME" rev-parse HEAD)"
  VERIFICATION_FILE="/tmp/${ISSUE_ID}-verification.json"
  test -s "$VERIFICATION_FILE"
  jq -e '
    type == "array" and length > 0 and
    all(.[];
      (keys | sort) == ["check_id","collected","exit_code","passed","skipped"] and
      (.check_id | type == "string" and test("^[a-z0-9][a-z0-9._-]{0,79}$")) and
      ([.exit_code,.collected,.passed,.skipped] |
        all(type == "number" and floor == . and . >= 0)) and
      (.exit_code == 0 and .skipped == 0 and .passed == .collected)
    ) and
    ([.[].check_id] | length == (unique | length))
  ' "$VERIFICATION_FILE" >/dev/null
  filigree show "$ISSUE_ID" --json | jq -e \
    --arg worker "$WORKER_ID" '.status_category == "wip" and .assignee == $worker' >/dev/null
  ```

- [ ] **Step 2: Make the worker rebase, not the coordinator improvise conflicts**

  ```bash
  git -C ".worktrees/$WORKTREE_NAME" rebase release/0.7.1
  ```

  Trivial clean rebases continue. A semantic conflict returns to the plan owner,
  who resolves it and reruns the complete plan handoff. The coordinator never
  guesses at a conflict across config, schema, security, or audit surfaces.

- [ ] **Step 3: Rerun the integrated handoff gates**

  In the rebased worktree, rerun the plan's complete final verification block,
  not only its nearest unit tests. Require clean status and record the rebased
  implementation SHA:

  ```bash
  uv sync --directory ".worktrees/$WORKTREE_NAME" --frozen --all-extras
  if rg -q 'npm (--prefix src/elspeth/web/frontend|ci|run|test)|src/elspeth/web/frontend/' "$PLAN_FILE"; then
      npm --prefix ".worktrees/$WORKTREE_NAME/src/elspeth/web/frontend" ci
  fi
  test -z "$(git -C ".worktrees/$WORKTREE_NAME" status --porcelain)"
  REBASED_IMPLEMENTATION_SHA="$(git -C ".worktrees/$WORKTREE_NAME" rev-parse HEAD)"
  INTEGRATED_VERIFICATION_FILE="/tmp/${ISSUE_ID}-integrated-verification.json"
  jq -e --arg sha "$REBASED_IMPLEMENTATION_SHA" '
    .schema == "elspeth.aws-ecs-integrated-verification.v1" and
    .rebased_implementation_sha == $sha and
    (.checks | type == "array" and length > 0) and
    all(.checks[];
      (keys | sort) == ["check_id","collected","exit_code","passed","skipped"] and
      (.check_id | type == "string" and test("^[a-z0-9][a-z0-9._-]{0,79}$")) and
      ([.exit_code,.collected,.passed,.skipped] |
        all(type == "number" and floor == . and . >= 0)) and
      (.exit_code == 0 and .skipped == 0 and .passed == .collected)
    ) and
    ([.checks[].check_id] | length == (unique | length))
  ' "$INTEGRATED_VERIFICATION_FILE" >/dev/null
  ```

  The coordinator creates that envelope only from the just-completed
  post-rebase rerun. Its `checks` use the Section 5 schema, and its bound SHA
  must equal the current rebased worktree HEAD. The pre-rebase worker file
  cannot satisfy this gate.

- [ ] **Step 4: Fast-forward the release branch**

  ```bash
  cd /home/john/elspeth
  git merge --ff-only "$BRANCH_NAME"
  INTEGRATION_SHA="$(git rev-parse HEAD)"
  test -z "$(git status --porcelain)"
  git merge-base --is-ancestor "$REBASED_IMPLEMENTATION_SHA" "$INTEGRATION_SHA"
  ```

- [ ] **Step 5: Close only on integrated evidence**

  Add a second, post-rebase handoff comment using the same bounded verification
  array schema from Section 5, with `schema` set to
  `elspeth.aws-ecs-integrated-handoff.v1`, the original implementation SHA, the
  rebased implementation SHA, and the integration SHA. Then close:

  ```bash
  INTEGRATED_EVIDENCE="$(jq -cn \
    --arg schema "elspeth.aws-ecs-integrated-handoff.v1" \
    --arg implementation_sha "$IMPLEMENTATION_SHA" \
    --arg rebased_implementation_sha "$REBASED_IMPLEMENTATION_SHA" \
    --arg integration_sha "$INTEGRATION_SHA" \
    --argjson verification "$(jq -c .checks "$INTEGRATED_VERIFICATION_FILE")" \
    '{schema:$schema,implementation_sha:$implementation_sha,rebased_implementation_sha:$rebased_implementation_sha,integration_sha:$integration_sha,verification:$verification}')"
  filigree add-comment "$ISSUE_ID" "$INTEGRATED_EVIDENCE" \
    --expected-assignee "$WORKER_ID" \
    --actor "$AWS_ECS_COORDINATOR"
  filigree close "$ISSUE_ID" \
    --expected-assignee "$WORKER_ID" \
    --actor "$AWS_ECS_COORDINATOR" \
    --commit "release/0.7.1@$INTEGRATION_SHA" \
    --reason "integrated into release/0.7.1; plan handoff gates passed"
  ```

  Immediately verify the returned close anchor and refresh the live DAG before
  any successor is claimed:

  ```bash
  CLOSED_JSON="$(filigree show "$ISSUE_ID" --json)"
  jq -e \
    --arg worker "$WORKER_ID" \
    --arg anchor "release/0.7.1@$INTEGRATION_SHA" \
    '.status_category == "done" and .assignee == $worker and .close_commit == $anchor' \
    <<<"$CLOSED_JSON" >/dev/null
  CLOSE_SHA="$(jq -r '.close_commit | split("@")[-1]' <<<"$CLOSED_JSON")"
  git merge-base --is-ancestor "$CLOSE_SHA" release/0.7.1
  filigree plan elspeth-6343920a47 --json --detail full > /tmp/aws-ecs-plan.json
  ```

- [ ] **Step 6: Remove only the completed worktree**

  After the close is verified and no repair is pending:

  ```bash
  git worktree remove ".worktrees/$WORKTREE_NAME"
  git branch -d "$BRANCH_NAME"
  ```

## 7. Recommended stage schedule

The hard-edge DAG allows some event-driven overlap. This barrier schedule is
slightly stricter where shared files, goldens, or guided/composer surfaces make
parallel integration expensive. An orchestrator may dispatch a next-stage lane
early only when all its direct dependencies are integrated and it does not
violate a serialized ownership chain below.

### Stage 1: Parallel foundations

- [ ] Baseline — `elspeth-8166b310e7`
- [ ] Plan 01 — `elspeth-b9e8b5d24b`
- [ ] Plan 02 — `elspeth-9070fb0a45`

Exit conditions:

- Plan 01 and 02 plan-level gates are green on the integrated tree.
- Plan 02 owns and regenerates the initial PostgreSQL `uv.lock` change.
- Signed-tier diagnosis is clean or operator-repaired; no agent signs.
- Trust-boundary tests/scope/tier pass.
- `wardline scan . --fail-on ERROR` exits 0; exit 1 requires boundary repair,
  exit 2 is a tool failure and blocks the stage.

### Stage 2: Parallel early consumers

- [ ] 08A — `elspeth-c0103e6c88`
- [ ] 15A — `elspeth-130dc48252`
- [ ] 03A — `elspeth-dffe064287` (Plan 03 Tasks 1–2 only)
- [ ] Plan 04 — `elspeth-03cf981d4a`
- [ ] Plan 13 — `elspeth-5e729216f4`

Exit conditions:

- 08A is integrated before any `aws_s3` source or sink registration.
- 15A passes strict byte/hash/collision/resume and text round-trip proof.
- 03A completes only its base doctor tasks; Task 3 remains unclaimed.
- Plan 04 proves validate-only startup and no AWS-ECS schema/mount mutation.
- Plan 13 proves authorization-code + S256 PKCE and exact-origin behavior.
- Merge/rebase shared `config.py`, `app.py`, and CLI surfaces one branch at a
  time; every later branch reruns its full handoff after rebase.

### Stage 3: Parallel runtime and integration prerequisites

- [ ] Plan 06 — `elspeth-7fe6aa531f`
- [ ] Plan 05 — `elspeth-1a1c31bcce`
- [ ] Plan 11 — `elspeth-25286192ee`
- [ ] Plan 14 — `elspeth-f5d5dddddf`

Exit conditions:

- Plan 06 rebases on integrated Plan 02/08A and regenerates the authoritative
  PostgreSQL + AWS lock; its registration guard is load-bearing.
- Plan 05 merges after Plan 04 and preserves shallow `/api/health` plus bounded
  `/api/ready`.
- Plan 11 consumes Plans 01/02/04, proves DML under DDL denial, and seals every
  request-time Landscape opener.
- Plan 14 consumes Plan 13 and the baseline, keeps telemetry task-local and
  operator-owned, and preserves audit-first behavior.

### Stage 4: Parallel AWS plugin finishing

- [ ] Plan 07 — `elspeth-74717426b7`
- [ ] 08B — `elspeth-a342f333a4` (Plan 08 Task 4 only)
- [ ] Plan 09 — `elspeth-e8dc754360`

Exit conditions:

- Plan 07 extends the integrated AWS extra with Jinja2 and performs the final
  Plan 02 → 06 → 07 lock regeneration. Never text-merge `uv.lock`.
- 08B adds guided-source parity without weakening 08A.
- Plan 09 registers the keyless Bedrock provider and passes audit/catalog/error
  gates. Local smoke evidence does not replace Plan 12 task-role proof.
- Complete this barrier before 15B even though 07/08B are not direct 15B
  dependencies; this avoids stale guided/composer tests and goldens.

### Stage 5: Universal web plugin policy convergence

- [ ] Plan 15B — `elspeth-0674a06468`

Entry: 15A, 05, 08A, 09, 11, and 14 are closed and integrated.

Exit: one frozen policy/snapshot governs catalog, freeform, guided, recipes,
import, validation, execution, delayed export, readiness, caches, and audit;
private bindings stay memory-only; CLI remains unrestricted; epoch-23 SQLite
and PostgreSQL refusal/recreation tests pass.

### Stage 6: Bedrock Guardrail controls

- [ ] Plan 15C — `elspeth-7d1f35e3d8`

Entry: 15B and 06 are closed and integrated.

Exit: recursive installed-model validation, strict detect/intervention handling,
prompt/content transforms, opaque profiles, full-vs-web schema parity, audit
before telemetry, and reusable `live_aws` proof are green. Plan 15C registration
must never appear in a tree without integrated Plan 15B.

### Stage 7: Integrated doctor proof

- [ ] 03B — `elspeth-397ac915b8` (Plan 03 Tasks 3–4)

Entry: 03A, 06, 07, 09, 14, and 15C are integrated.

Exit: the real PostgreSQL doctor proof sees the complete S3, Bedrock provider,
telemetry, universal-policy, and Guardrail registration set; all Plan 03
handoff gates pass with mandatory Docker lanes executed.

### Stage 8: Packaging and deployment runbook

- [ ] Plan 10 — `elspeth-6285c29c07`

Plan 10 is exclusive integration work and specializes Section 5 Steps 3–5.
After the normal exact-ID dependency, ancestry, plan-checksum, and ownership
checks—but before creating a worktree or editing a Plan-10 file—atomically
claim the slice at the clean integrated source SHA:

```bash
cd /home/john/elspeth
test -z "$(git status --porcelain)"
export ROLLBACK_BASELINE_SHA="$(git rev-parse release/0.7.1)"
export WORKER_ID="${WORKER_ID:?set named Plan 10 owner}"
PLAN10_JSON="$(filigree show elspeth-6285c29c07 --json)"
if jq -e '.is_ready == true and ((.assignee // "") == "")' <<<"$PLAN10_JSON" >/dev/null; then
    filigree start-work elspeth-6285c29c07 \
      --assignee "$WORKER_ID" \
      --actor "$WORKER_ID" \
      --commit "release/0.7.1@$ROLLBACK_BASELINE_SHA"
    PLAN10_FRESH=1
else
    jq -e \
      --arg worker "$WORKER_ID" \
      --arg anchor "release/0.7.1@$ROLLBACK_BASELINE_SHA" \
      '.status_category == "wip" and .assignee == $worker and .claim_commit == $anchor' \
      <<<"$PLAN10_JSON" >/dev/null
    PLAN10_FRESH=0
fi
if test "$PLAN10_FRESH" = 1; then
    trap 'filigree add-comment elspeth-6285c29c07 "{\"schema\":\"elspeth.aws-ecs-setup-failure.v1\",\"stage\":\"plan10_task0_or_worktree_setup\",\"retry\":\"exact-owner-resume\"}" --expected-assignee "$WORKER_ID" --actor "$AWS_ECS_COORDINATOR" || true' ERR
fi
test "$(git rev-parse release/0.7.1)" = "$ROLLBACK_BASELINE_SHA"
```

Now execute Plan 10 Task 0's three exact, unfiltered source-qualification
commands in the coordinator release worktree. Capture exit codes under
`set -Eeuo pipefail`; all must be zero. Require the tree and release SHA to
remain unchanged. Only then record the qualified source baseline:

```bash
export APPROVED_EVIDENCE_REF="${APPROVED_EVIDENCE_REF:?set protected Task 0 evidence reference}"
test "${#APPROVED_EVIDENCE_REF}" -le 128
[[ "$APPROVED_EVIDENCE_REF" =~ ^[A-Za-z0-9._-]{1,128}$ ]]
BASELINE_COMMENTS="$(filigree get-comments elspeth-6285c29c07 --json)"
BASELINE_RECEIPT_COUNT="$(jq --arg sha "$ROLLBACK_BASELINE_SHA" '[.items[] | .text | fromjson? | select(.schema == "elspeth.aws-ecs-rollback-baseline.v1" and .marker == "aws-ecs-rollback-baseline-source" and .sha == $sha)] | length' <<<"$BASELINE_COMMENTS")"
test "$BASELINE_RECEIPT_COUNT" -le 1
if test "$BASELINE_RECEIPT_COUNT" = 0; then
  uv sync --frozen --all-extras
  uv run pytest tests/unit/web/auth/ tests/unit/web/test_config.py tests/unit/web/test_app.py -v
  npm --prefix src/elspeth/web/frontend ci
  npm --prefix src/elspeth/web/frontend test -- src/components/auth/LoginPage.test.tsx
  npm --prefix src/elspeth/web/frontend run typecheck
  test -z "$(git status --porcelain)"
  test "$(git rev-parse release/0.7.1)" = "$ROLLBACK_BASELINE_SHA"
  BASELINE_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  filigree add-comment elspeth-6285c29c07 \
    "$(jq -cn \
      --arg schema elspeth.aws-ecs-rollback-baseline.v1 \
      --arg marker aws-ecs-rollback-baseline-source \
      --arg sha "$ROLLBACK_BASELINE_SHA" \
      --arg timestamp "$BASELINE_TIMESTAMP" \
      --arg evidence_ref "$APPROVED_EVIDENCE_REF" \
      --arg auth_pytest 'uv run pytest tests/unit/web/auth/ tests/unit/web/test_config.py tests/unit/web/test_app.py -v' \
      --arg auth_frontend 'npm --prefix src/elspeth/web/frontend test -- src/components/auth/LoginPage.test.tsx' \
      --arg frontend_typecheck 'npm --prefix src/elspeth/web/frontend run typecheck' \
      '{schema:$schema,marker:$marker,sha:$sha,timestamp:$timestamp,source_qualified_only:true,evidence_ref:$evidence_ref,checks:[{command:$auth_pytest,exit_code:0},{command:$auth_frontend,exit_code:0},{command:$frontend_typecheck,exit_code:0}]}')" \
    --expected-assignee "$WORKER_ID" \
    --actor "$AWS_ECS_COORDINATOR"
fi
if test -d .worktrees/aws-ecs-10; then
    test "$(git -C .worktrees/aws-ecs-10 branch --show-current)" = feat/aws-ecs-10-packaging
    git -C .worktrees/aws-ecs-10 merge-base --is-ancestor "$ROLLBACK_BASELINE_SHA" HEAD
else
    if git show-ref --verify --quiet refs/heads/feat/aws-ecs-10-packaging; then
        git merge-base --is-ancestor "$ROLLBACK_BASELINE_SHA" feat/aws-ecs-10-packaging
        git worktree add .worktrees/aws-ecs-10 feat/aws-ecs-10-packaging
    else
        git worktree add .worktrees/aws-ecs-10 -b feat/aws-ecs-10-packaging "$ROLLBACK_BASELINE_SHA"
    fi
fi
cd .worktrees/aws-ecs-10
uv sync --frozen --all-extras
npm --prefix src/elspeth/web/frontend ci
if test "$PLAN10_FRESH" = 1; then
    test -z "$(git status --porcelain)"
fi
trap - ERR
```

The detailed test output stays only at `APPROVED_EVIDENCE_REF`; the tracker
record contains fixed command names and exit codes, not output. A qualification
failure leaves no baseline comment and blocks Plan 10. The baseline is
source-qualified only, not live production qualification.

After this specialized setup, execute Plan 10 Task 1 onward through Section 5
Step 6, worker evidence, and the normal Section 6 integration/close protocol.

Any prerequisite change or unexpected non-Plan-10 release commit before Plan
10 integration invalidates the source baseline: discard the Plan-10 branch,
integrate the prerequisite repair, and restart Task 0 on a fresh integrated
SHA. Plan 10's own reviewed commits and final fast-forward do not invalidate
the historical pre-Plan-10 source baseline.

If a prerequisite is reopened after Plan 10 has integrated, stop for an
operator-approved release-line reconstruction. The current post-Plan-10 HEAD
must never be relabelled as a fresh pre-Plan-10 baseline. Retire any Plan 12
candidate and evidence after required cleanup; preserve the old baseline and
candidate anchors; reconstruct a replacement pre-Plan-10 lineage from the last
qualified baseline plus the repaired prerequisite/descendant commits; rerun
Task 0; then re-execute Plan 10 so the new baseline is an ancestor of the
replacement candidate. Replacing `release/0.7.1` with that non-fast-forward
lineage requires explicit repository-owner approval and branch-protection
procedure; without it, return NO-GO.

Exit: lean platform-bound image, executable runbook, acceptance tooling,
IAM/EFS/health/Exec/CloudWatch/Guardrail/OIDC contracts, and all Plan-10 static
and security gates are green.

### Stage 9: Final Plan 12 closeout

- [ ] Plan 12 — `elspeth-05396fed38`

Plan 12 runs alone. No implementation branch, plan repair, lockfile change, or
unrelated release commit may run in parallel.

Before Plan 12 Task 1 claims or resumes its issue:

- verify all 19 prerequisite tracker steps are done;
- verify every close anchor is an ancestor of the release tip;
- verify the tree is clean and version boundaries match Plan 12;
- establish the named infrastructure, database, identity, release, evidence,
  and cleanup operators;
- initialize the protected gate ledger/control manifest before external
  mutation; and
- record the clean integrated starting SHA and prerequisite-anchor ledger.

Because Plan 15C intentionally may edit Plan 12, bind its final reviewed text
immediately before Task 1:

```bash
PLAN12_FILE=docs/superpowers/plans/aws/2026-07-08-aws-ecs-12-integration-closeout.md
PLAN12_REVIEW=docs/superpowers/plans/aws/2026-07-08-aws-ecs-12-integration-closeout.review.json
PLAN12_SHA256="$(sha256sum "$PLAN12_FILE" | awk '{print $1}')"
jq -e --arg sha "$PLAN12_SHA256" \
  '(.verdict | startswith("APPROVED")) and .plan_sha256 == $sha' \
  "$PLAN12_REVIEW"
```

A mismatch stops closeout for re-review, sidecar update, reviewed planning
commit, and `PLAN_SET_SHA` advancement. Plan 12 never consumes stale approval.

Atomically start Wave 4 or validate its exact coordinator-owned resume before
entering Plan 12 Task 1:

```bash
STARTING_SHA="$(git rev-parse release/0.7.1)"
WAVE4_JSON="$(filigree show elspeth-ef74690eb2 --json)"
case "$(jq -r .status <<<"$WAVE4_JSON")" in
  pending)
    filigree start-work elspeth-ef74690eb2 \
      --target-status active \
      --assignee "$AWS_ECS_COORDINATOR" \
      --actor "$AWS_ECS_COORDINATOR" \
      --commit "release/0.7.1@$STARTING_SHA"
    ;;
  active)
    jq -e --arg owner "$AWS_ECS_COORDINATOR" \
      '.assignee == $owner and
       (.claim_commit | type == "string" and test("^release/0\\.7\\.1@[0-9a-f]{40}$"))' \
      <<<"$WAVE4_JSON" >/dev/null
    WAVE4_BASE_SHA="$(jq -r '.claim_commit | split("@")[-1]' <<<"$WAVE4_JSON")"
    git merge-base --is-ancestor "$WAVE4_BASE_SHA" "$STARTING_SHA"
    ;;
  *) exit 1 ;;
esac
```

Run Plan 12 directly in the coordinator's checked-out `release/0.7.1` worktree;
Sections 5 and 6 do not apply. Task 1 atomically claims or validates exact-owner
resume, prepares the release commit, and only then freezes the resulting
candidate SHA. Execute Plan 12 from Task 1 without combining evidence across
SHAs. Before any
external mutation, a failure returns to the owning plan and restarts Task 1 on
a repaired candidate. After external mutation, set cleanup-required, execute
the complete cleanup/evidence-export path, return NO-GO, reopen the owner, and
restart on a new candidate. Only Plan 12's final GO path may close the issue.

Plan 12 GO is limited to the exact tested source, digest, and platform. It is
not durable signed multi-platform publication approval.

## 8. Serialized ownership chains

| Shared surface | Required integration order | Rule |
|---|---|---|
| `uv.lock` / extras | 02 → 06 → 07 | Rebase and regenerate; never text-merge. Plan 10/12 consume only. |
| startup/readiness app wiring | 04 → 05 | Plan 05 assumes Plan 04's final lifespan/schema posture. |
| auth/config/telemetry | 13 → 14 | Plan 14 consumes the final auth/config surface. |
| guarded Landscape openers | 04 → 11 | Plan 11 consumes Plan 04 and must preserve Plan 02 schema behavior. |
| universal web integration | 05 + 11 + 14 + 15A + 08A + 09 → 15B | 15B is the integration owner across these shared surfaces. |
| Bedrock controls | 15B → 15C | No Guardrail registration before universal policy enforcement. |
| integrated inventory proof | 15C + 07 + 09 + 14 + 03A → 03B | Doctor proof runs only against the complete registration set. |
| packaging/live harness | all prior → 10 → 12 | Plan 10 records baseline first; Plan 12 owns runtime GO. |

## 9. Stage-exit audit

At every stage barrier:

- [ ] Every stage issue is done in Filigree.
- [ ] Every `close_commit` resolves to a SHA that is an ancestor of
  `release/0.7.1`.
- [ ] Every plan's complete handoff command block passed after final rebase.
- [ ] Mandatory Docker/live lanes owned by slices completed in this stage
      executed with zero skips; intentionally deferred task-role/live proof is
      still explicitly owned by Plan 10 or Plan 12.
- [ ] Signed-tier diagnosis and operator repairs are complete where required.
- [ ] `wardline scan . --fail-on ERROR` exited 0 for external-input changes.
- [ ] `uv lock --check` passes; the current lock owner is correct.
- [ ] `git diff --check` passes and the release worktree is clean.
- [ ] Live `filigree plan` and `filigree critical-path` were refreshed.
- [ ] No successor was claimed solely because a feature-branch test passed.

After the audit, the coordinator may close an active phase only when a fresh
milestone snapshot proves every one of that phase's children done. Use the
current release anchor; never skip an unfinished child:

```bash
filigree plan elspeth-6343920a47 --json --detail full > /tmp/aws-ecs-plan.json
for phase_id in elspeth-db28fb3293 elspeth-da8e5447c8 elspeth-ab8f9e11d9 elspeth-ef74690eb2; do
    PHASE_JSON="$(jq -c --arg id "$phase_id" '.phases[] | select(.phase.issue_id == $id)' /tmp/aws-ecs-plan.json)"
    PHASE_STATUS="$(jq -r '.phase.status' <<<"$PHASE_JSON")"
    if jq -e '(.steps | length > 0) and all(.steps[]; .status_category == "done")' <<<"$PHASE_JSON" >/dev/null; then
        if test "$PHASE_STATUS" = active; then
            filigree close "$phase_id" \
              --expected-assignee "$AWS_ECS_COORDINATOR" \
              --actor "$AWS_ECS_COORDINATOR" \
              --commit "release/0.7.1@$(git rev-parse release/0.7.1)" \
              --reason "all phase children integrated and closed"
        else
            test "$(jq -r '.phase.status_category' <<<"$PHASE_JSON")" = done
        fi
    fi
done
```

On Plan 12 GO, close Wave 4 after Plan 12, then close milestone
`elspeth-6343920a47` with `release/0.7.1@$CANDIDATE_SHA` after verifying that
the candidate is still the release HEAD, using
`--expected-assignee "$AWS_ECS_COORDINATOR"`. On NO-GO, reopened work, or
incomplete cleanup, keep the affected phase and milestone active. Heartbeat the
active milestone and every active phase under the exact coordinator assignee
at each dispatch and stage barrier.

## 10. Failure, reopen, and evidence invalidation

### Before integration

Keep the issue assigned/in progress, add a sanitized blocker comment, and stop
its descendants. Release the claim only for an explicit owner handoff. Do not
close as partial, convert an in-scope defect to an observation, or weaken a
gate.

### After integration or close

Freeze all dispatch and stop active descendant workers first. From the exact
DAG in Section 4, enumerate the owning slice's full transitive descendant set
and check it against the refreshed live DAG. Reopen every already-closed
descendant in **reverse topological order**, then reopen the owning slice. This
includes Plan 10 and Plan 12 whenever they are descendants:

```bash
filigree plan elspeth-6343920a47 --json --detail full > /tmp/aws-ecs-plan.json
export ISSUE_ID
uv run --frozen python - <<'PY'
import hashlib
import json
import os
from collections import defaultdict, deque
from pathlib import Path

payload = json.loads(Path("/tmp/aws-ecs-plan.json").read_text(encoding="utf-8"))
steps = [step for phase in payload["phases"] for step in phase["steps"]]
by_id = {step["issue_id"]: step for step in steps}
owner = os.environ["ISSUE_ID"]
assert owner in by_id, owner

children: dict[str, set[str]] = defaultdict(set)
indegree = {issue_id: 0 for issue_id in by_id}
for issue_id, step in by_id.items():
    for dependency in step.get("blocked_by", []):
        assert dependency in by_id, (issue_id, dependency)
        children[dependency].add(issue_id)
        indegree[issue_id] += 1

queue = deque(sorted(issue_id for issue_id, degree in indegree.items() if degree == 0))
topological: list[str] = []
while queue:
    issue_id = queue.popleft()
    topological.append(issue_id)
    for child in sorted(children[issue_id]):
        indegree[child] -= 1
        if indegree[child] == 0:
            queue.append(child)
assert len(topological) == len(by_id), "live milestone DAG contains a cycle"

descendants: set[str] = set()
pending = list(children[owner])
while pending:
    issue_id = pending.pop()
    if issue_id not in descendants:
        descendants.add(issue_id)
        pending.extend(children[issue_id])
reverse_topological = [issue_id for issue_id in reversed(topological) if issue_id in descendants]
phase_by_issue = {
    step["issue_id"]: phase["phase"]["issue_id"]
    for phase in payload["phases"]
    for step in phase["steps"]
}
phases = sorted({phase_by_issue[issue_id] for issue_id in descendants | {owner}})

descendant_path = Path("/tmp/aws-descendants-reverse-topo.txt")
phase_path = Path("/tmp/aws-affected-phases.txt")
descendant_path.write_text("".join(f"{issue_id}\n" for issue_id in reverse_topological), encoding="utf-8")
phase_path.write_text("".join(f"{phase_id}\n" for phase_id in phases), encoding="utf-8")
for path in (descendant_path, phase_path):
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())
PY

# Review both exact-ID files and their printed hashes before any reopen.
mapfile -t DESCENDANTS_REVERSE_TOPO < /tmp/aws-descendants-reverse-topo.txt
mapfile -t AFFECTED_PHASE_IDS < /tmp/aws-affected-phases.txt

# Reopen completed lifecycle containers first so repaired children can resume.
MILESTONE_JSON="$(filigree show elspeth-6343920a47 --json)"
if test "$(jq -r .status_category <<<"$MILESTONE_JSON")" = done; then
    filigree reopen elspeth-6343920a47 --actor "$AWS_ECS_COORDINATOR"
fi
for phase_id in "${AFFECTED_PHASE_IDS[@]}"; do
    case "$phase_id" in
      elspeth-db28fb3293|elspeth-da8e5447c8|elspeth-ab8f9e11d9|elspeth-ef74690eb2) ;;
      *) exit 1 ;;
    esac
    PHASE_JSON="$(filigree show "$phase_id" --json)"
    if test "$(jq -r .status_category <<<"$PHASE_JSON")" = done; then
        filigree reopen "$phase_id" --actor "$AWS_ECS_COORDINATOR"
    fi
done

for descendant_id in "${DESCENDANTS_REVERSE_TOPO[@]}"; do
    test -n "$descendant_id" || continue
    DESCENDANT_JSON="$(filigree show "$descendant_id" --json)"
    if test "$(jq -r .status_category <<<"$DESCENDANT_JSON")" = "done"; then
        filigree reopen "$descendant_id" --actor "$AWS_ECS_COORDINATOR"
    fi
done
OWNER_JSON="$(filigree show "$ISSUE_ID" --json)"
case "$(jq -r .status_category <<<"$OWNER_JSON")" in
  done) filigree reopen "$ISSUE_ID" --actor "$AWS_ECS_COORDINATOR" ;;
  wip) jq -e '.assignee | type == "string" and length > 0' <<<"$OWNER_JSON" >/dev/null ;;
  *) exit 1 ;;
esac
```

`reopen` restores the last non-done status; it does not establish a fresh
atomic claim. For each reopened issue, release any stale assignee only through
the tracker-supported exact-owner handoff, then reuse Section 5 Step 3 with the
exact manifest `ISSUE_ID`, named `WORKER_ID`, and current release SHA. Use
`--advance` only when Filigree reports a soft transition. Never use claim plus
status update.

Treat the owner and every transitive descendant's prior base, test,
close-anchor, Plan-10 rollback-baseline, and Plan-12 candidate evidence as
invalid. Repair/reclose the owner first, then re-execute and reclose descendants
in topological order with new anchors. A descendant may not remain closed
merely because its old tests passed. A WIP descendant keeps no valid evidence;
after the owner repair, it must be restarted or explicitly exact-owner resumed
on a fresh integrated base.

The generated phase file is the deduplicated parent-phase set of the owner plus
all transitive descendants. The reverse-topological descendant file is derived
only from the refreshed live graph after Section 3's exact DAG parity check has
passed. After reopen, milestone and phase assignees must still equal
`$AWS_ECS_COORDINATOR`; otherwise stop for an exact-owner handoff before child
redispatch.

Any prerequisite reopen after the Plan-10 rollback baseline invalidates that
baseline and all Plan-10 evidence. Any code/SHA change after Plan 12 freezes a
candidate invalidates the candidate and restarts Plan 12.

## 11. Immediate stop conditions

Stop and surface the exact owner/remedy when any of these occurs:

- plan/document/tracker dependency disagreement;
- dirty coordinator release worktree;
- dependency marked done but close SHA absent from release ancestry;
- claim conflict or unexpected assignee;
- failing baseline before implementation;
- required operator, credential, Docker, AWS, Terraform, browser, or evidence
  facility unavailable;
- required test skipped, deselected, xfailed, or not collected;
- Wardline exit 1 without a boundary fix, or exit 2;
- signed metadata drift without operator-authorized repair;
- lockfile or generated-golden merge conflict;
- potential secret, raw provider response, prompt/content, URL, ARN, account,
  role, credential, or raw exception leakage into evidence;
- S3 registration without 08A, Guardrail registration without 15B,
  request-path DDL without 11, or redirectable operator telemetry without 14;
- Plan-10 rollback baseline recorded after Plan-10 edits;
- Plan-12 local/remote/CI/task/image SHA mismatch; or
- incomplete external cleanup/evidence export.

The orchestrator never guesses past a stop condition. It repairs the contract,
records a real dependency/owner decision, or returns a precise NO-GO.
