# AWS ECS Runtime Readiness Orchestration Run Sheet

> **For agentic workers:** REQUIRED SUB-SKILLS: the integration coordinator
> uses `superpowers:subagent-driven-development`; every implementation worker
> uses the exact skills named by its plan. Use
> `superpowers:using-git-worktrees` once to establish the program worktree and
> `superpowers:verification-before-completion` before every close and final
> fast-forward. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Project applicability correction:** Wardline is suite-installed but is not
> enabled for Elspeth because the project has no Wardline markers. Do not run
> it, stop on it, or create evidence for it. Hosted CI, candidate
> PRs, and temporary remote refs are likewise outside this program. Stage 9
> uses local verification plus live AWS only.

**Goal:** Execute the complete AWS ECS runtime-readiness program in dependency
order on `feat/aws-ecs-program`, hand one immutable candidate to Plan 12 for
final live GO/NO-GO acceptance, then fast-forward that exact tested SHA into
`release/0.7.1` as one final merge.

**Architecture:** One named integration coordinator owns one ignored worktree
at `/home/john/elspeth/.worktrees/aws-ecs-program`, branch
`feat/aws-ecs-program`, the live Filigree DAG, serialized mutations, close
anchors, and stage barriers. The recorded program base remains immutable in
program ancestry, while the main release checkout and `release/0.7.1` may
advance independently for unrelated planning work through Stages 1–8. They are
not program coordination surfaces until the explicit pre-candidate
reconciliation in Stage 9. Filigree readiness is necessary but not sufficient:
a dependency is consumable only when its close commit is
an ancestor of the current program tip and its focused program-branch handoff
gates are green.

**Tech Stack:** Git worktrees, Filigree milestone
`elspeth-6343920a47`, Python/uv, Node/npm, Docker/testcontainers,
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

> Bootstrap from `/home/john/elspeth` as the sole AWS ECS integration
> coordinator, then work in the one ignored program worktree at
> `/home/john/elspeth/.worktrees/aws-ecs-program` on
> `feat/aws-ecs-program`.
> Read `AGENTS.md` and this run sheet completely, then execute milestone
> `elspeth-6343920a47` stage by stage. Dispatch only the exact issue IDs listed
> here, serialize all writes and tracker/external mutations through the
> coordinator, and never create a plan-specific worktree or integration
> branch. Close each slice truthfully at `feat/aws-ecs-program@<SHA>` after its
> focused gates pass on the shared program branch. Preserve the 02→06→07
> lockfile chain and every security/audit gate. Hold the complete unfiltered
> suite until Plan 12 final washup. Ignore unrelated main-checkout and release
> branch activity through Stages 1–8. Immediately before Plan 12 freezes its
> candidate, reconcile the then-current `release/0.7.1` tip into the program
> branch, rerun the affected-area checkpoint, and bind that consumed tip.
> After Tasks 1–6 pass on one unchanged SHA, require the release tip still to
> equal that reconciliation anchor and fast-forward it to the exact tested SHA
> before Task 7 issues GO. Stop on any condition in section 11 rather than
> guessing or weakening a gate.

## 2. Authority and non-negotiable rules

1. **One coordinator:** set one stable coordinator identity for the entire run.
   The coordinator alone serializes mutations in the shared program worktree,
   Filigree transitions, external/AWS changes, and the final release
   fast-forward.
2. **Exact IDs only:** this repository currently has hundreds of unrelated
   ready issues. Never use `start-next-work`, numeric “next plan” assumptions,
   or a generic ready-queue worker. Dispatch only the IDs in this document.
3. **Atomic claims:** use `filigree start-work`, never `claim` followed by a
   status update. A conflict means another owner won; stop rather than steal.
4. **Program-branch evidence precedes closure:** each slice commits directly to
   `feat/aws-ecs-program`. The coordinator closes only after focused handoff
   gates pass on that program tip, using `feat/aws-ecs-program@<SHA>`.
5. **Live ancestry:** a done dependency whose `close_commit` is not an ancestor
   of the current program tip is not satisfied. Stop and repair the tracker or
   program-branch state.
6. **One worktree:** create or resume only
   `/home/john/elspeth/.worktrees/aws-ecs-program`. Do not create per-slice
   worktrees or branches.
7. **Preserve user work:** never stash, reset, checkout-over, or broad-stage
   unrelated changes. Use the exact staging allowlists in each plan and admit
   only one writer at a time to shared state.
8. **No lockfile text merges:** `uv.lock` ownership is strictly Plan 02 → Plan
   06 → Plan 07. Each owner starts from the previous program-branch lock
   commit and regenerates it.
9. **Audit before telemetry:** Landscape remains authoritative. No stage may
   weaken audit ordering, replace audit evidence with telemetry, or retain raw
   provider/user content in operational evidence.
10. **No security bypasses:** no waiver, baseline, allowlist broadening,
    threshold reduction, or skipped mandatory lane may turn a failing gate
    green.
11. **Operator-held authority stays operator-held:** agents diagnose signed
    trust metadata but never receive signing keys or self-sign repairs.
12. **Current-scope defects stay in scope:** do not hide an implementation gap
    as a short-lived observation. Fix it, expand the issue, or add a real
    dependency/issue before continuing.
13. **The run sheet owns orchestration mechanics:** Sections 3, 5, and 6
    supersede every plan-local checkbox that claims/releases/closes a Filigree
    issue, chooses a worker identity, creates/removes a worktree, chooses a
    branch/base SHA, performs obsolete worker-branch integration, or bans
    closure before release integration. Those bootstrap/handoff checkboxes are
    satisfied by
    this run sheet and must not be executed twice. Every plan implementation,
    test, evidence, staging-allowlist, and commit checkbox remains binding.
    Plan 10 retains its rollback-baseline semantics. Plan 12 retains its owned
    Task 1 claim/resume, exact-candidate evidence, cleanup, and final close
    instructions as amended for the program branch and final fast-forward.
14. **Delegation authority:** the coordinator may invoke every relevant
    installed skill and use as many subagents as useful without further
    permission. This authority does not broaden scope, weaken gates, transfer
    operator credentials, or permit conflicting concurrent writes. Parallel
    work is read-only or demonstrably non-conflicting; the coordinator
    serializes all mutations.
15. **Reasoning effort:** use `xhigh` for the coordinator. Default
    implementation and review subagents to `high`; elevate technically
    complex, security-sensitive, cross-cutting, concurrency-heavy, ambiguous,
    or repeatedly failing work to `xhigh`. If the surface cannot set effort,
    state it in the dispatch and use the highest supported setting without
    blocking solely on that limitation.
16. **Baseline signing freeze:** from the final signed-entry diagnosis through
    operator repair, post-repair diagnosis, baseline gates, baseline commit,
    and baseline close, freeze all program-worktree code/config mutation.
    Preserve queued sibling work, commits, and checkpoints intact, but pause
    their integration and further mutation until the baseline closes.
17. **Concurrent release planning is out of scope until Stage 9:** after the
    program worktree exists, do not inspect, clean, reset, stash, commit, or
    otherwise manage `/home/john/elspeth` or concurrent `release/0.7.1`
    planning. Release movement or a dirty main checkout during Stages 1–8 is
    not a stop condition. `PROGRAM_BASE_SHA` is the immutable ancestry marker
    for the program branch. Stage 9 alone reconciles the then-current release
    tip before candidate freeze; movement after that reconciliation
    invalidates the candidate and restarts Plan 12 Task 1. Documentation-only
    changes outside AWS program plans, reviews, designs, and specifications,
    repository policy, generated contracts, runtime code, and tests are
    non-substantive: absorb them at the Stage 9 merge and do not reopen slices.
    Every non-documentation change is
    substantive. Documentation that touches any of those named program
    surfaces is also substantive and uses the normal impact/evidence
    invalidation rules.

## 3. Coordinator bootstrap

- [ ] **Step 1: Stabilize the planning commit and record the program base**

  The relocation, this run sheet, tracker path updates, and dependency repairs
  must be committed and present on `release/0.7.1` before the autonomous run
  starts. Record that tip as both the initial `PLAN_SET_SHA` and immutable
  `PROGRAM_BASE_SHA`; it is not the later Plan-10 rollback baseline. Persist
  both in the protected coordinator checkpoint. Subsequent reviewed planning
  commits advance `PLAN_SET_SHA` on the program branch without changing
  `PROGRAM_BASE_SHA`. After worktree creation, concurrent main-checkout and
  release-branch planning is out of scope until Stage 9 reconciliation.

- [ ] **Step 2: Establish the coordinator and the one program worktree**

  ```bash
  set -Eeuo pipefail
  umask 077
  export AWS_PAGER=""
  export AWS_ECS_COORDINATOR="${AWS_ECS_COORDINATOR:?set the named integration coordinator}"
  export PROGRAM_WORKTREE="/home/john/elspeth/.worktrees/aws-ecs-program"
  export PROGRAM_BRANCH="feat/aws-ecs-program"

  if test -d "$PROGRAM_WORKTREE"; then
      cd "$PROGRAM_WORKTREE"
      : "${PROGRAM_BASE_SHA:?restore the recorded immutable program base from the protected checkpoint}"
      : "${PLAN_SET_SHA:?restore the latest reviewed planning commit from the protected checkpoint}"
      test "$(git branch --show-current)" = "$PROGRAM_BRANCH"
      git merge-base --is-ancestor "$PROGRAM_BASE_SHA" HEAD
      git merge-base --is-ancestor "$PLAN_SET_SHA" HEAD
  elif git -C /home/john/elspeth show-ref --verify --quiet "refs/heads/$PROGRAM_BRANCH"; then
      cd /home/john/elspeth
      git check-ignore -q .worktrees
      : "${PROGRAM_BASE_SHA:?restore the recorded immutable program base from the protected checkpoint}"
      : "${PLAN_SET_SHA:?restore the latest reviewed planning commit from the protected checkpoint}"
      git merge-base --is-ancestor "$PROGRAM_BASE_SHA" "$PROGRAM_BRANCH"
      git merge-base --is-ancestor "$PLAN_SET_SHA" "$PROGRAM_BRANCH"
      git worktree add "$PROGRAM_WORKTREE" "$PROGRAM_BRANCH"
  else
      cd /home/john/elspeth
      git check-ignore -q .worktrees
      export PROGRAM_BASE_SHA="$(git rev-parse release/0.7.1)"
      export PLAN_SET_SHA="$PROGRAM_BASE_SHA"
      git worktree add "$PROGRAM_WORKTREE" -b "$PROGRAM_BRANCH" "$PROGRAM_BASE_SHA"
  fi

  test -z "$(git -C "$PROGRAM_WORKTREE" status --porcelain)"
  cd "$PROGRAM_WORKTREE"
  export PROGRAM_TIP="$(git rev-parse HEAD)"
  ```

  Expected: the coordinator owns one clean program worktree descended from
  immutable `PROGRAM_BASE_SHA`; the current reviewed `PLAN_SET_SHA` is in its
  ancestry; no plan-specific AWS ECS worktree or integration branch is
  created; and no assertion or mutation targets the concurrent main checkout
  or current release tip after worktree creation.

  Every coordinator and worker command block runs in Bash with
  `set -Eeuo pipefail` and `umask 077`; AWS-capable blocks also export
  `AWS_PAGER=""`. Treat each complete numbered protocol (Section 5 dispatch,
  Section 6 close, Stage 8 Plan 10 setup, and Stage 9 release reconciliation)
  as one persistent Bash
  session; fenced blocks are readability breaks, not new shells. Exports do not
  cross agent boundaries. At session start, explicitly pass or redeclare
  `AWS_ECS_COORDINATOR`, `WORKER_ID`, `ISSUE_ID`, `PLAN_BASE`, `PLAN_FILE`,
  `REVIEW_FILE`, `PLAN_SET_SHA`, `PROGRAM_BASE_SHA`, `PROGRAM_WORKTREE`, and
  `PROGRAM_BRANCH`. If a shell is interrupted, restart the protocol's documented
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

  Before Baseline dispatch, reconcile the live structured tracker contract for
  `elspeth-8166b310e7` and `elspeth-c0103e6c88`; prose comments are not a
  substitute. Baseline structured targets/verification must include
  `state.py`, the exact direct boundary-test node, and the signed metadata
  directory as operator-owned scope. 08A structured targets/verification must
  remove `state.py`, the direct boundary test, fingerprint repair, and baseline
  signing work. Use
  the tracker-supported structured update surface, refresh both issues with
  `filigree show ... --json`, and stop dispatch until the live fields match
  this split exactly. This run-sheet edit does not itself mutate the tracker.
  The exact signed-YAML repair set does not exist yet and must not be invented:
  immediately after the final signed-entry diagnosis and before operator
  repair, update Baseline's structured target files to replace the directory
  placeholder with only the exact receipt-listed YAML paths, refresh the issue,
  and require a byte-for-byte match to the diagnosis receipt before proceeding.

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
        --commit "feat/aws-ecs-program@$PROGRAM_TIP"
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
  - PostgreSQL/Aurora schema ownership and retention approval;
  - AWS/Terraform/ECR/ECS/EFS/ALB/Cognito/S3/Bedrock/CloudWatch permissions;
  - protected evidence storage and emergency cleanup.

  Persist only identity labels and booleans in a protected coordinator
  checkpoint, never credentials, account details, URLs, or ARNs:

  ```bash
  export AUTHORITY_MATRIX="${AUTHORITY_MATRIX:?set protected mode-0600 authority matrix path}"
  jq -e '
    .schema == "elspeth.aws-ecs-authority-matrix.v1" and
    (.authorities | type == "array" and length == 6) and
    all(.authorities[];
      (keys | sort) == ["authorized","available","owner_label","surface"] and
      (.surface | IN("trust_signing","docker","database_retention","aws_runtime","evidence_storage","cleanup")) and
      (.owner_label | type == "string" and test("^[A-Za-z0-9._-]{1,64}$")) and
      (.authorized | type == "boolean") and (.available | type == "boolean")
    ) and
    ([.authorities[].surface] | unique | length == 6)
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

Every row uses the same worktree
`/home/john/elspeth/.worktrees/aws-ecs-program` and branch
`feat/aws-ecs-program`.

| Key | Plan/review base name | Assigned scope |
|---|---|---|
| baseline | `2026-07-08-aws-ecs-08-s3-endpoint-gate` | Task 0 |
| 01 | `2026-07-08-aws-ecs-01-deployment-contract` | all tasks |
| 02 | `2026-07-08-aws-ecs-02-postgres-schema-support` | all tasks |
| 08A | `2026-07-08-aws-ecs-08-s3-endpoint-gate` | Tasks 1–3 |
| 15A | `2026-07-08-aws-ecs-15a-text-sink` | all tasks |
| 03A | `2026-07-08-aws-ecs-03-doctor-cli` | Tasks 1–2 |
| 04 | `2026-07-08-aws-ecs-04-validate-only-startup` | all tasks |
| 13 | `2026-07-08-aws-ecs-13-cognito-authorization-origin` | all tasks |
| 06 | `2026-07-08-aws-ecs-06-s3-source` | all tasks |
| 05 | `2026-07-08-aws-ecs-05-readiness-endpoint` | all tasks |
| 11 | `2026-07-08-aws-ecs-11-landscape-write-gate` | all tasks |
| 14 | `2026-07-08-aws-ecs-14-cloudwatch-operator-telemetry` | all tasks |
| 07 | `2026-07-08-aws-ecs-07-s3-sink` | all tasks |
| 08B | `2026-07-08-aws-ecs-08-s3-endpoint-gate` | Task 4 |
| 09 | `2026-07-08-aws-ecs-09-bedrock-provider` | all tasks |
| 15B | `2026-07-08-aws-ecs-15b-universal-web-plugin-policy` | all tasks |
| 15C | `2026-07-08-aws-ecs-15c-bedrock-guardrail-shields` | all tasks |
| 03B | `2026-07-08-aws-ecs-03-doctor-cli` | Tasks 3–4 |
| 10 | `2026-07-08-aws-ecs-10-packaging-docker` | all tasks |
| 12 | `2026-07-08-aws-ecs-12-integration-closeout` | all tasks; Plan 12 protocol only |

For a row, set:

```bash
PLAN_FILE="docs/superpowers/plans/aws/${PLAN_BASE}.md"
REVIEW_FILE="docs/superpowers/plans/aws/${PLAN_BASE}.review.json"
```

`ISSUE_ID` and `PLAN_BASE` come from that exact row and the DAG table;
`WORKER_ID` is the named implementation agent assigned by the coordinator.
These are typed dispatch inputs, not free-form choices. Plan 12 is never passed
to the generic slice protocol.

## 5. Shared-program slice protocol

Use this protocol for every slice except Plan 12. The coordinator may delegate
implementation and review, but every slice uses the existing program worktree
and branch. This section supersedes plan-local worktree, branch, claim, merge,
release, and close instructions as stated in rule 13.

- [ ] **Step 1: Read the exact plan slice and current review artifact**

  For split plans, execute only the tasks assigned to the tracker slice. From
  the program worktree, verify the review verdict and checksum before claiming:

  ~~~bash
  cd "$PROGRAM_WORKTREE"
  test "$(git branch --show-current)" = "$PROGRAM_BRANCH"
  test -z "$(git status --porcelain -- docs/superpowers/plans/aws docs/superpowers/specs)"
  PROGRAM_TIP="$(git rev-parse HEAD)"
  git merge-base --is-ancestor "$PROGRAM_BASE_SHA" "$PROGRAM_TIP"
  git merge-base --is-ancestor "$PLAN_SET_SHA" "$PROGRAM_TIP"
  PLAN_SHA256="$(sha256sum "$PLAN_FILE" | awk '{print $1}')"
  jq -e --arg sha "$PLAN_SHA256" \
    '(.verdict | startswith("APPROVED")) and .plan_sha256 == $sha' \
    "$REVIEW_FILE"
  ~~~

  A checksum mismatch means the plan changed after review. Stop, re-review it,
  update the sidecar, commit the reviewed planning change to the program branch,
  and advance PLAN_SET_SHA before dispatch. An old approval never survives
  plan drift.

- [ ] **Step 2: Require live readiness and program-branch ancestry**

  ~~~bash
  ISSUE_JSON="$(filigree show "$ISSUE_ID" --json)"
  while read -r dependency_id; do
      DEP_JSON="$(filigree show "$dependency_id" --json)"
      jq -e '.status_category == "done" and
             (.close_commit | type == "string" and
              test("^feat/aws-ecs-program@[0-9a-f]{40}$"))' \
        <<<"$DEP_JSON" >/dev/null
      DEP_SHA="$(jq -r '.close_commit | split("@")[-1]' <<<"$DEP_JSON")"
      git merge-base --is-ancestor "$DEP_SHA" "$PROGRAM_TIP"
  done < <(jq -r '.blocked_by[]' <<<"$ISSUE_JSON")
  ~~~

  Expected: every direct dependency is done and its truthful program-branch
  close SHA is an ancestor of the current program tip. A malformed anchor or
  failed ancestry check stops dispatch.

  The verification-baseline close is historical evidence, not a permanent
  waiver. Before each baseline-dependent dispatch, and after any
  trust-boundary-sensitive change, run the current trust-tier and
  trust-boundary gates from the clean program worktree. Signed binding drift
  returns to the named operator.

- [ ] **Step 3: Atomically start or validate exact-owner resume**

  The coordinator alone performs Filigree transitions. Start the exact parent
  phase if needed, then the exact issue; never use claim plus status update.

  ~~~bash
  ISSUE_JSON="$(filigree show "$ISSUE_ID" --json)"
  PHASE_ID="$(jq -r .parent_id <<<"$ISSUE_JSON")"
  case "$PHASE_ID" in
    elspeth-db28fb3293|elspeth-da8e5447c8|elspeth-ab8f9e11d9|elspeth-ef74690eb2) ;;
    *) exit 1 ;;
  esac
  PHASE_JSON="$(filigree show "$PHASE_ID" --json)"
  case "$(jq -r .status <<<"$PHASE_JSON")" in
    pending)
      filigree start-work "$PHASE_ID" \
        --target-status active \
        --assignee "$AWS_ECS_COORDINATOR" \
        --actor "$AWS_ECS_COORDINATOR" \
        --commit "feat/aws-ecs-program@$PROGRAM_TIP"
      ;;
    active)
      jq -e --arg owner "$AWS_ECS_COORDINATOR" \
        '.assignee == $owner' <<<"$PHASE_JSON" >/dev/null
      ;;
    *) exit 1 ;;
  esac

  ISSUE_JSON="$(filigree show "$ISSUE_ID" --json)"
  if jq -e '.is_ready == true and ((.assignee // "") == "")' \
      <<<"$ISSUE_JSON" >/dev/null; then
      filigree start-work "$ISSUE_ID" \
        --assignee "$WORKER_ID" \
        --actor "$AWS_ECS_COORDINATOR" \
        --commit "feat/aws-ecs-program@$PROGRAM_TIP"
      FRESH_DISPATCH=1
      SLICE_BASE_SHA="$PROGRAM_TIP"
  else
      jq -e --arg worker "$WORKER_ID" \
        '.status_category == "wip" and .assignee == $worker and
         (.claim_commit | type == "string" and
          test("^feat/aws-ecs-program@[0-9a-f]{40}$"))' \
        <<<"$ISSUE_JSON" >/dev/null
      SLICE_BASE_SHA="$(jq -r '.claim_commit | split("@")[-1]' <<<"$ISSUE_JSON")"
      git cat-file -e "$SLICE_BASE_SHA^{commit}"
      git merge-base --is-ancestor "$SLICE_BASE_SHA" "$PROGRAM_TIP"
      FRESH_DISPATCH=0
  fi
  ~~~

  A sibling slice may have committed after the claim; do not rewrite the
  historical claim anchor. A claim conflict or unexpected assignee stops that
  lane.

- [ ] **Step 4: Establish the serialized shared-worktree boundary**

  ~~~bash
  cd "$PROGRAM_WORKTREE"
  test "$(git branch --show-current)" = "$PROGRAM_BRANCH"
  git merge-base --is-ancestor "$PROGRAM_BASE_SHA" HEAD
  git merge-base --is-ancestor "$SLICE_BASE_SHA" HEAD
  if test "$FRESH_DISPATCH" = 1; then
      test -z "$(git status --porcelain)"
      uv sync --frozen --all-extras
      if rg -q 'npm (--prefix src/elspeth/web/frontend|ci|run|test)|src/elspeth/web/frontend/' "$PLAN_FILE"; then
          npm --prefix src/elspeth/web/frontend ci
      fi
      test -z "$(git status --porcelain)"
  fi
  ~~~

  Admit only one writer at a time. Subagents may run parallel read-only or
  demonstrably non-conflicting work; the coordinator serializes every edit,
  index operation, commit, dependency/generated-file operation, conflicting
  test, tracker transition, and external mutation. On exact-owner resume,
  preserve in-progress changes and resume from the last sanitized checkpoint;
  never discard partial work. If checkpoint and worktree disagree, stop.

- [ ] **Step 5: Execute every assigned implementation and verification checkbox**

  Follow every assigned implementation, test, evidence, staging-allowlist, and
  commit checkbox. Treat plan-local claim, release, worktree, branch/base,
  obsolete branch-integration, and close checkboxes as fulfilled by this run
  sheet and do not execute them again. Use TDD. Run all relevant focused unit,
  integration,
  property, frontend, Docker, PostgreSQL, live, static, typing, trust,
  coverage, manifest, packaging, and affected-area checks in
  proportion to the change. Hold only the complete unfiltered repository suite
  until Plan 12 final washup.

  Refresh claim liveness after each completed plan task and at least hourly
  during long checks. After each task, add a sanitized
  elspeth.aws-ecs-task-checkpoint.v1 comment containing only stable task ID,
  current commit SHA, and complete=true. The coordinator verifies the exact
  plan staging allowlist, commits coherent increments directly on
  feat/aws-ecs-program, and leaves the shared worktree clean at handoff.

- [ ] **Step 6: Produce bounded slice evidence**

  Create /tmp/$ISSUE_ID-verification.json as a non-empty array with one object
  per stable plan check ID:

  ~~~json
  [{"check_id":"task1.unit-tests","exit_code":0,"collected":42,"passed":42,"skipped":0}]
  ~~~

  Check IDs match ^[a-z0-9][a-z0-9._-]{0,79}$; every completed check has
  exit_code=0, skipped=0, and passed=collected; non-test gates use zero counts.
  Do not include commands, output, exception text, paths, URLs, provider/user
  content, or extra fields.

  ~~~bash
  test -z "$(git status --porcelain)"
  IMPLEMENTATION_SHA="$(git rev-parse HEAD)"
  VERIFICATION_FILE="/tmp/$ISSUE_ID-verification.json"
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
  git merge-base --is-ancestor "$SLICE_BASE_SHA" "$IMPLEMENTATION_SHA"
  ~~~

## 6. Coordinator verification and close protocol

Use this protocol for every slice except Plan 12. There is no per-slice branch
integration or worktree removal: the current shared program tip is already the
integrated result.

- [ ] **Step 1: Recheck the exclusive mutation boundary**

  ~~~bash
  cd "$PROGRAM_WORKTREE"
  test "$(git branch --show-current)" = "$PROGRAM_BRANCH"
  test -z "$(git status --porcelain)"
  PROGRAM_TIP="$(git rev-parse HEAD)"
  git merge-base --is-ancestor "$PROGRAM_BASE_SHA" "$PROGRAM_TIP"
  git merge-base --is-ancestor "$IMPLEMENTATION_SHA" "$PROGRAM_TIP"
  filigree show "$ISSUE_ID" --json | jq -e \
    --arg worker "$WORKER_ID" \
    '.status_category == "wip" and .assignee == $worker' >/dev/null
  ~~~

- [ ] **Step 2: Run the complete focused handoff on the program tip**

  Rerun the plan's complete final focused verification block on PROGRAM_TIP,
  plus any additional affected checks derived from Loomweave, Warpline, and
  the owning plan
  ownership. Mandatory Docker/live lanes execute with zero skips where the plan
  requires them. Create /tmp/$ISSUE_ID-integrated-verification.json with schema
  elspeth.aws-ecs-integrated-verification.v1, program_tip, and a non-empty
  checks array using the bounded Section 5 schema. The envelope is valid only
  when program_tip equals PROGRAM_TIP; pre-change evidence cannot satisfy it.

  Baseline is an evidence dependency for closure, not a new hard-DAG
  implementation prerequisite. Plan 01 and Plan 02 may be claimed,
  implemented, tested, and committed in parallel with Baseline. However, no
  non-Baseline slice, including Plan 01 or Plan 02, may enter Step 3 until the
  Baseline close commit is in current program ancestry and the current
  trust-tier and trust-boundary gates pass on that exact program tip. This
  closure rule does not add or imply a tracker dependency edge.

  `IMPLEMENTATION_SHA` normally equals `PROGRAM_TIP`. An ancestor-only match is
  permitted solely because the shared program lineage may have admitted later
  coordinator-reviewed commits while this issue waited for Baseline evidence
  or resumed after evidence invalidation. Step 2 is therefore mandatory on the
  complete current tip: it binds the closure to the integrated tree, and Step
  3 records that exact `PROGRAM_TIP` rather than the older implementation SHA.

- [ ] **Step 3: Comment and close truthfully on program-branch evidence**

  ~~~bash
  INTEGRATED_VERIFICATION_FILE="/tmp/$ISSUE_ID-integrated-verification.json"
  jq -e --arg sha "$PROGRAM_TIP" '
    .schema == "elspeth.aws-ecs-integrated-verification.v1" and
    .program_tip == $sha and
    (.checks | type == "array" and length > 0) and
    all(.checks[];
      (keys | sort) == ["check_id","collected","exit_code","passed","skipped"] and
      (.check_id | type == "string" and test("^[a-z0-9][a-z0-9._-]{0,79}$")) and
      ([.exit_code,.collected,.passed,.skipped] |
        all(type == "number" and floor == . and . >= 0)) and
      (.exit_code == 0 and .skipped == 0 and .passed == .collected)
    )
  ' "$INTEGRATED_VERIFICATION_FILE" >/dev/null

  EVIDENCE="$(jq -cn \
    --arg schema "elspeth.aws-ecs-program-handoff.v1" \
    --arg plan "$PLAN_FILE" \
    --arg slice_base_sha "$SLICE_BASE_SHA" \
    --arg implementation_sha "$IMPLEMENTATION_SHA" \
    --arg program_tip "$PROGRAM_TIP" \
    --argjson verification "$(jq -c .checks "$INTEGRATED_VERIFICATION_FILE")" \
    '{schema:$schema,plan:$plan,slice_base_sha:$slice_base_sha,
      implementation_sha:$implementation_sha,program_tip:$program_tip,
      verification:$verification}')"
  filigree add-comment "$ISSUE_ID" "$EVIDENCE" \
    --expected-assignee "$WORKER_ID" \
    --actor "$AWS_ECS_COORDINATOR"
  filigree close "$ISSUE_ID" \
    --expected-assignee "$WORKER_ID" \
    --actor "$AWS_ECS_COORDINATOR" \
    --commit "feat/aws-ecs-program@$PROGRAM_TIP" \
    --reason "integrated on feat/aws-ecs-program; focused handoff gates passed"
  ~~~

- [ ] **Step 4: Verify close ancestry before successor dispatch**

  ~~~bash
  CLOSED_JSON="$(filigree show "$ISSUE_ID" --json)"
  jq -e \
    --arg worker "$WORKER_ID" \
    --arg anchor "feat/aws-ecs-program@$PROGRAM_TIP" \
    '.status_category == "done" and .assignee == $worker and
     .close_commit == $anchor' <<<"$CLOSED_JSON" >/dev/null
  CLOSE_SHA="$(jq -r '.close_commit | split("@")[-1]' <<<"$CLOSED_JSON")"
  git merge-base --is-ancestor "$CLOSE_SHA" HEAD
  filigree plan elspeth-6343920a47 --json --detail full \
    > /tmp/aws-ecs-plan.json
  ~~~

  This focused, integrated program-branch close is the truthful evidence that
  advances the DAG. After the final fast-forward, Plan 12 re-verifies that
  every slice close SHA is also an ancestor of release/0.7.1.

## 7. Recommended stage schedule

The hard-edge DAG allows parallel analysis, review, and demonstrably
non-conflicting checks. The coordinator still admits every mutation to the one
program worktree serially. A next-stage lane may begin early only when all its
direct dependencies are closed in current program ancestry and it does not
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
- Edit shared `config.py`, `app.py`, and CLI surfaces one slice at a time;
  every later slice reruns its full affected handoff on the new program tip.

### Stage 3: Parallel runtime and integration prerequisites

- [ ] Plan 06 — `elspeth-7fe6aa531f`
- [ ] Plan 05 — `elspeth-1a1c31bcce`
- [ ] Plan 11 — `elspeth-25286192ee`
- [ ] Plan 14 — `elspeth-f5d5dddddf`

Exit conditions:

- Plan 06 starts from integrated Plan 02/08A and regenerates the authoritative
  PostgreSQL + AWS lock; its registration guard is load-bearing.
- Plan 05 commits after Plan 04 and preserves shallow `/api/health` plus bounded
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

- [ ] Plan 10 — elspeth-6285c29c07

Plan 10 is exclusive shared-worktree work. After the normal exact-ID
dependency, ancestry, plan-checksum, and ownership checks, but before any Plan
10 edit, freeze the clean current program tip as the rollback-baseline source:

~~~bash
cd "$PROGRAM_WORKTREE"
test "$(git branch --show-current)" = "$PROGRAM_BRANCH"
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "$PROGRAM_BASE_SHA" HEAD
export ROLLBACK_BASELINE_SHA="$(git rev-parse HEAD)"
: "${WORKER_ID:?set named Plan 10 owner}"
PLAN10_JSON="$(filigree show elspeth-6285c29c07 --json)"
if jq -e '.is_ready == true and ((.assignee // "") == "")'     <<<"$PLAN10_JSON" >/dev/null; then
    filigree start-work elspeth-6285c29c07       --assignee "$WORKER_ID"       --actor "$AWS_ECS_COORDINATOR"       --commit "feat/aws-ecs-program@$ROLLBACK_BASELINE_SHA"
else
    jq -e       --arg worker "$WORKER_ID"       --arg anchor "feat/aws-ecs-program@$ROLLBACK_BASELINE_SHA"       '.status_category == "wip" and .assignee == $worker and
       .claim_commit == $anchor' <<<"$PLAN10_JSON" >/dev/null
fi
~~~

Execute Plan 10 Task 0's three exact source-qualification commands on that SHA.
They are focused auth/frontend commands, not the held whole-repository suite.
Capture all exit codes; require zero; require HEAD, the clean tree, and
`PROGRAM_BASE_SHA` ancestry to remain unchanged. Do not inspect or bind the
concurrent release tip here. Only then add the existing sanitized
elspeth.aws-ecs-rollback-baseline.v1 receipt, bound to ROLLBACK_BASELINE_SHA,
and continue with Plan 10 Task 1 in the same program worktree. Do not create a
Plan 10 branch or worktree.

The rollback baseline remains source-qualified only, not live production
qualification. Plan 10's own reviewed commits do not invalidate its historical
pre-Plan-10 baseline. Any prerequisite repair before Plan 10 closes invalidates
the baseline and descendant evidence; integrate the repair on the program
branch, freeze a new eligible pre-Plan-10 source SHA, rerun Task 0, and
re-execute Plan 10.

If a prerequisite reopens after Plan 10 closes, stop for repository-owner
approval of a replacement program lineage that omits the invalid Plan 10
result. Never relabel a post-Plan-10 tip as a new pre-Plan-10 baseline. Retire
any Plan 12 candidate after required cleanup, preserve historical anchors,
reconstruct from the last qualified baseline plus repaired
prerequisite/descendant commits, rerun Task 0, and re-execute Plan 10. Do not
touch concurrent release work during that reconstruction. Rewriting the
program branch requires explicit owner approval; without it, return NO-GO.

Exit: lean platform-bound image, executable runbook, acceptance tooling,
IAM/EFS/health/Exec/CloudWatch/Guardrail/OIDC contracts, and all Plan-10 static
and security gates are green.

### Stage 9: Final Plan 12 closeout

- [ ] Plan 12 — elspeth-05396fed38

Plan 12 runs alone in the program worktree. No implementation, plan repair,
lockfile change, unrelated commit, tracker mutation, or external mutation may
run in parallel.

Before release reconciliation:

- verify all 19 prerequisite tracker steps are done;
- verify every close anchor has the feat/aws-ecs-program@SHA form and its SHA is
  an ancestor of the current program tip;
- verify the program worktree is clean and version boundaries match Plan 12;
- establish the named infrastructure, database, identity, release, evidence,
  and cleanup operators.

Then perform the one pre-candidate release reconciliation. This is the first
time since worktree creation that concurrent `release/0.7.1` state becomes a
program input:

~~~bash
cd "$PROGRAM_WORKTREE"
test "$(git branch --show-current)" = "$PROGRAM_BRANCH"
if git rev-parse -q --verify MERGE_HEAD >/dev/null; then
    # Exact conflict-resume path after the merge command's documented exit 2.
    export PRE_RECONCILIATION_PROGRAM_SHA="$(git rev-parse HEAD)"
    export RECONCILED_RELEASE_SHA="$(git rev-parse MERGE_HEAD)"
    test "$(git rev-parse ORIG_HEAD)" = "$PRE_RECONCILIATION_PROGRAM_SHA"
else
    test -z "$(git status --porcelain)"
    export PRE_RECONCILIATION_PROGRAM_SHA="$(git rev-parse HEAD)"
    export RECONCILED_RELEASE_SHA="$(git rev-parse release/0.7.1)"
fi
git merge-base --is-ancestor "$PROGRAM_BASE_SHA" "$PRE_RECONCILIATION_PROGRAM_SHA"
export RELEASE_RECONCILIATION_BASE_SHA="$(git merge-base "$PRE_RECONCILIATION_PROGRAM_SHA" "$RECONCILED_RELEASE_SHA")"
git log --oneline "$RELEASE_RECONCILIATION_BASE_SHA..$RECONCILED_RELEASE_SHA"
git diff --name-status "$RELEASE_RECONCILIATION_BASE_SHA" "$RECONCILED_RELEASE_SHA"
~~~

Review and classify that exact release-only commit/path inventory before
continuing. Documentation-only paths outside the named substantive surfaces
are non-substantive. Any non-documentation path, or documentation touching an
AWS program plan, review, design, or specification, repository policy,
generated contract, runtime code, or test, is substantive and must enter the
impact/evidence protocol below. If the substantive delta affects Plan 10
itself or any prerequisite already consumed by closed Plan 10,
do not merge through generic Section 10 recovery: use Stage 8's explicit
post-Plan-10 owner-approved replacement-lineage protocol, rebuild/reclose the
affected lineage through Plan 10, and restart Stage 9.

~~~bash
if git rev-parse -q --verify MERGE_HEAD >/dev/null; then
    test "$(git rev-parse MERGE_HEAD)" = "$RECONCILED_RELEASE_SHA"
    test -z "$(git diff --name-only --diff-filter=U)"
    test -z "$(git diff --name-only)"
    test -z "$(git ls-files --others --exclude-standard)"
    git diff --cached --check
    git commit -m "chore: reconcile release planning before Plan 12"
elif ! git merge-base --is-ancestor "$RECONCILED_RELEASE_SHA" HEAD; then
    if ! git merge --no-ff --no-commit "$RECONCILED_RELEASE_SHA"; then
        git status --short
        if git rev-parse -q --verify MERGE_HEAD >/dev/null &&
           test -n "$(git diff --name-only --diff-filter=U)"; then
            exit 2
        fi
        exit 1
    fi
    git diff --cached --check
    git commit -m "chore: reconcile release planning before Plan 12"
fi
git merge-base --is-ancestor "$PRE_RECONCILIATION_PROGRAM_SHA" HEAD
git merge-base --is-ancestor "$RECONCILED_RELEASE_SHA" HEAD
test -z "$(git status --porcelain)"
export PROGRAM_TIP="$(git rev-parse HEAD)"
~~~

This reconciliation is a merge into the program branch, never a rebase or
reset, so every historical slice close anchor remains in ancestry. The final
release update is still a fast-forward and creates no new merge commit. Review
the release-only delta before the merge. Only a conflict-class exit 2 leaves
`MERGE_HEAD`; any other merge failure exits 1 and is not a conflict-resume
case. After exit 2, the coordinator resolves only the exact unmerged paths
semantically, stages every resolution, runs the affected checks, and reruns
the entire Stage 9 reconciliation protocol. Its initial block reconstructs
and validates the exact reviewed anchors from `HEAD`/`ORIG_HEAD` and
`MERGE_HEAD`;
never resample a moving release ref, choose one branch wholesale, or discard
unrelated work. If the delta
changes any completed slice's
implementation, AWS program plan, review, design, specification, repository gate,
generated artifact, or security boundary not covered by the Stage 8 special
case, apply Section 10
evidence invalidation and re-execute the affected owners/descendants before
Plan 12. Planning-only changes outside the AWS program still require the
reconciliation affected-area checkpoint but do not reopen unrelated slices.
Re-run the checks derived from the reconciled delta. Do not freeze a candidate
until this reconciliation checkpoint passes and the program worktree is clean.

After reconciliation and any required owner/descendant re-execution, refresh
the authoritative prerequisite and ancestry proof on the final program tip:

~~~bash
filigree plan elspeth-6343920a47 --json --detail full > /tmp/aws-ecs-plan12-reconciled.json
test "$(jq '[.phases[].steps[] | select(.issue_id != "elspeth-05396fed38")] | length' /tmp/aws-ecs-plan12-reconciled.json)" -eq 19
jq -e 'all(.phases[].steps[]; .issue_id == "elspeth-05396fed38" or .status_category == "done")' /tmp/aws-ecs-plan12-reconciled.json >/dev/null
while read -r anchor; do
    [[ "$anchor" =~ ^feat/aws-ecs-program@[0-9a-f]{40}$ ]]
    CLOSE_SHA="${anchor##*@}"
    git merge-base --is-ancestor "$CLOSE_SHA" HEAD
done < <(jq -r '.phases[].steps[] | select(.issue_id != "elspeth-05396fed38") | .close_commit' /tmp/aws-ecs-plan12-reconciled.json)
export PROGRAM_TIP="$(git rev-parse HEAD)"
~~~

Because Plan 15C may intentionally edit Plan 12, verify its final reviewed text
immediately before Task 1. A checksum mismatch stops closeout for re-review,
sidecar update, and a reviewed planning commit on the program branch. Plan 12
never consumes stale approval.

Only now bind the final reviewed Plan 12 checksum and initialize its protected
phase ledger under Task 1. Initialize the candidate-bound control manifest at
Plan 12's documented point before the first external mutation, not before
reconciliation or candidate freeze.

Validate the exact coordinator-owned Plan 12 resume using the current
feat/aws-ecs-program@PROGRAM_TIP anchor. Then follow Plan 12 Task 1 to prepare
and freeze CANDIDATE_SHA. The phase ledger binds the program branch,
PROGRAM_BASE_SHA, RECONCILED_RELEASE_SHA, and CANDIDATE_SHA.

Run Tasks 1–6 on that one unchanged candidate. This final washup explicitly
includes the exact complete-suite command:

~~~bash
uv run pytest tests/ -v -m ""
~~~

It also includes the local quality, image, live AWS, rollback,
evidence-export, and cleanup work required by Plan 12. Never push or
update release/0.7.1 early. Any
code, documentation, lockfile, generated-file, or candidate-SHA change
invalidates all candidate evidence and restarts Task 1. Evidence from
different SHAs may never be combined.

After Tasks 1–6 pass and cleanup is complete, hand control directly to Task 7.
Plan 12 Task 7 owns the single idempotent release transition. The run sheet
performs no release mutation of its own; it only confirms the inputs Task 7
must reconstruct from the protected ledger:

~~~bash
test "$(git -C "$PROGRAM_WORKTREE" rev-parse HEAD)" = "$CANDIDATE_SHA"
test -z "$(git -C "$PROGRAM_WORKTREE" status --porcelain)"
git -C "$PROGRAM_WORKTREE" merge-base --is-ancestor "$RECONCILED_RELEASE_SHA" "$CANDIDATE_SHA"
test "$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger get --file "$GATE_LEDGER" --field reconciled_release_sha)" = "$RECONCILED_RELEASE_SHA"
test "$(uv run --frozen python -m elspeth.web.aws_ecs_acceptance gate-ledger get --file "$GATE_LEDGER" --field candidate_sha)" = "$CANDIDATE_SHA"
~~~

Task 7 freezes the complete ledger and advances local and remote
`release/0.7.1` through the closed
`{RECONCILED_RELEASE_SHA,CANDIDATE_SHA}` state machine. Only after that single
transition and the release-ancestry audit may Task 7 issue GO and close Plan 12
with release/0.7.1@CANDIDATE_SHA.

A release tip that moves after reconciliation invalidates the candidate. If
external mutation has begun or `CLEANUP_REQUIRED=1`, finish Plan 12 Task 6's
evidence export and every independent cleanup attempt before changing program
HEAD or starting another reconciliation; cleanup failure is NO-GO and forbids
restart. Then preserve the concurrent work, merge the new tip into the program
branch, repeat the reconciliation impact audit, and restart Plan 12 Task 1. A
non-fast-forward final update, dirty final release-update
surface, failed ancestry audit, missing cleanup, or changed candidate is
NO-GO. Preserve the release branch at its last verified state and report the
exact owner/remedy. Plan 12 GO remains limited to the exact tested source,
digest, and platform; it is not durable signed multi-platform publication
approval.

## 8. Serialized ownership chains

| Shared surface | Required integration order | Rule |
|---|---|---|
| `uv.lock` / extras | 02 → 06 → 07 | Regenerate serially on the shared branch; never text-merge. Plan 10/12 consume only. |
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
- [ ] Every `close_commit` resolves to a SHA that is an ancestor of the
  current `feat/aws-ecs-program` tip. After the final fast-forward, every
  prerequisite close SHA is also an ancestor of `release/0.7.1`.
- [ ] Every plan's complete focused handoff command block passed on its final
  program tip.
- [ ] Mandatory Docker/live lanes owned by slices completed in this stage
      executed with zero skips; intentionally deferred task-role/live proof is
      still explicitly owned by Plan 10 or Plan 12.
- [ ] Signed-tier diagnosis and operator repairs are complete where required.
- [ ] `uv lock --check` passes; the current lock owner is correct.
- [ ] `git diff --check` passes, the program worktree is clean, and
  `PROGRAM_BASE_SHA` remains in its ancestry. Do not inspect or gate on the
  concurrent main checkout or release tip at Stages 1–8 barriers.
- [ ] Live `filigree plan` and `filigree critical-path` were refreshed.
- [ ] No successor was claimed before its program-branch dependency anchors
  and focused handoff evidence passed.

Plan 12 closure is the endpoint for this job. Parent phase and milestone
administration are not additional closeout requirements.

## 10. Failure, reopen, and evidence invalidation

### Before a slice close

Keep the issue assigned/in progress, add a sanitized blocker comment, and stop
its descendants. Release the claim only for an explicit owner handoff. Do not
close as partial, convert an in-scope defect to an observation, or weaken a
gate.

### Baseline-reopen trust evidence overlay

Reopening Baseline freezes dispatch and pauses every baseline-evidence-dependent
closure, without adding tracker DAG edges. It invalidates prior trust-tier and
trust-boundary handoffs, including Plan 01, Plan 02, and their hard-DAG
descendants even when they are not Baseline descendants in the Section 4 graph.
Preserve every implementation commit and checkpoint; pause work in progress
rather than discarding it. Reopen affected closed issues in reverse topological
order, including Plan 01/02 and their hard descendants, while treating this
paragraph as an evidence-invalidation overlay rather than a dependency rewrite.

Repair and reclose Baseline first. Then exact-owner resume each paused or
reopened issue, integrate its preserved implementation on the current program
lineage, rerun its complete focused handoff, and rerun the trust-tier and
trust-boundary gates at that affected issue's exact tip. Reclose affected issues
in topological order with fresh anchors. No prior baseline trust result survives
a Baseline reopen.

### After a program-branch commit or close

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
baseline_id = "elspeth-8166b310e7"
if owner == baseline_id:
    # Baseline owns the signed-tier and trust-boundary evidence substrate. Its
    # reopen invalidates every prior slice handoff, including independent roots
    # Plan 01/02 and their descendants, without changing the tracker DAG.
    descendants = set(by_id) - {owner}
else:
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
exact manifest `ISSUE_ID`, named `WORKER_ID`, and current program SHA. Use
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
- dirty program worktree at a required clean boundary;
- dependency marked done but close SHA absent from the current
  `feat/aws-ecs-program` ancestry before final fast-forward, or absent from
  `release/0.7.1` ancestry after final fast-forward;
- claim conflict or unexpected assignee;
- failing baseline before implementation;
- required operator, credential, Docker, AWS, Terraform, browser, or evidence
  facility unavailable;
- required test skipped, deselected, xfailed, or not collected;
- signed metadata drift without operator-authorized repair;
- lockfile or generated-golden ownership collision;
- potential secret, raw provider response, prompt/content, URL, ARN, account,
  role, credential, or raw exception leakage into evidence;
- S3 registration without 08A, Guardrail registration without 15B,
  request-path DDL without 11, or redirectable operator telemetry without 14;
- Plan-10 rollback baseline recorded after Plan-10 edits;
- Plan-12 local/task/image SHA mismatch; or
- release movement after Stage 9 reconciliation without candidate
  invalidation, re-reconciliation, and a Task 1 restart; or
- incomplete external cleanup/evidence export.

The orchestrator never guesses past a stop condition. It repairs the contract,
records a real dependency/owner decision, or returns a precise NO-GO.
