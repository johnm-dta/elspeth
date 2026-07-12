# AWS ECS Runtime Readiness — Plan Index

> **For agentic workers:** This is the index, not a plan. Execute the numbered
> subplans with superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans. Plans 01–11 and 13 define scoped implementation
> verification and carry
> their own commits, but dependency-gated tasks are not independently
> executable. Plan 12 owns acceptance of the fully integrated program.

**Spec:** `docs/superpowers/specs/2026-07-08-aws-ecs-runtime-readiness-design.md`
(reviewed against the 11 canonical failure modes; findings closed —
`docs/superpowers/specs/review-2026-07-08.md`).

**Shared interface contract:** all plans were written against one pinned
brief. The load-bearing shared symbols: `deployment_contract.py`
(`DEPLOYMENT_TARGET_AWS_ECS`, `ContractCheck`, `validate_aws_ecs_settings`),
`schema_probe.py` (`SchemaState`, `probe_/init_session|landscape_schema` —
note `probe_session_schema` never returns `PARTIAL`; session
partial-presence is unverifiable and classifies `STALE`),
`readiness.py` (`readiness_report`), `landscape_access.py`
(`open_landscape_db`, `landscape_create_tables_allowed` — plan 11),
`aws_s3_common.build_s3_client`, provider key `bedrock` in `_PROVIDERS`,
extras `postgres` and `aws`.

## Subplans

| # | File | Delivers | Depends on |
|---|------|----------|------------|
| 01 | `…-01-deployment-contract.md` | `deployment_target` setting; pure aws-ecs contract validator; placeholder predicates extracted from existing validators | — |
| 02 | `…-02-postgres-schema-support.md` | `postgres` extra; schema-shape probes on the owning modules; `schema_probe.py` with `pg_advisory_lock`-serialized init; pool kwargs passthrough | — |
| 03 | `…-03-doctor-cli.md` | `elspeth doctor aws-ecs [--init-schema] --json`; non-persisting by default (filesystem probes are ephemeral and never mkdir in either mode); never opens `auth.db`; sanitized error classes | Tasks 1–2: 01, 02 · **Task 3 (integration proof): also 06, 07, 09** |
| 04 | `…-04-validate-only-startup.md` | Pre-auth fail-closed target/schema validation in aws-ecs mode (incl. orphan-reconciliation `create_tables`); 10 s connect/45 s per-probe budgets; startup requires pre-provisioned data/payload/blob directories and never creates them | 01, 02 |
| 05 | `…-05-readiness-endpoint.md` | `GET /api/ready` with bounded per-label admission (2 s/check, 5 s route ceiling, cancellation-safe 2 s cache, redacted); shallow `/api/health`; mechanical ALB-path proof | 04 (and transitively 01, 02) |
| 06 | `…-06-s3-source.md` | `aws_s3` source; `max_object_bytes` (256 MiB default) enforced pre-read; `aws` extra; `aws_s3_common.build_s3_client` | tracker/integration: 02 and 08A (lockfile + complete pre-registration web gate) |
| 07 | `…-07-s3-sink.md` | bounded `aws_s3` sink; Jinja keys; first-create `IfNoneMatch` plus cumulative `IfMatch`; audit-safe lifecycle; offline botocore and real-S3 proof | build: 06 · **registration: 08 core must already be committed** |
| 08 | `…-08-s3-endpoint-gate.md` | Web-authorship rejection of `aws_s3` `endpoint_url` (design review's Critical): `provider_config_policy` + `validate_pipeline` source/output adjudication + every composer mutation path (08A), then guided-source prompt parity (08B) | gate baseline: none, executable now · 08A: gate baseline · 08B: 06 + 08A |
| 09 | `…-09-bedrock-provider.md` | LiteLLM-backed `bedrock` pipeline provider in `_PROVIDERS`; composer Bedrock-response parsing tests; opt-in local smoke plus Plan-10/12 in-task task-role acceptance | 06 (AWS SDK support; transitively the shared gate baseline) |
| 10 | `…-10-packaging-docker.md` | Binds a fully integrated pre-Plan-10 rollback-baseline SHA; validated platform-bound lean image; force-new zero-overlap first/upgrade/recovery runbook with task-definition/EFS/IAM/health/Exec and schema authority; sanitized API/non-root-EFS/local-auth/S3-task-role/Bedrock/OIDC acceptance harnesses; orphan-safe cleanup | all implementation slices in 01–09, 11, and 13, including both 03/08 split slices |
| 11 | `…-11-landscape-write-gate.md` | Fail-closed `open_landscape_db(settings)` writer policy; migrates the per-run, tutorial-projection, four auth-audit methods, and all 39 current service test seams; import-aware AST seal covers direct/aliased/qualified constructors and `in_memory`; DDL-denied PostgreSQL request-writer proof | 01, 02, 04 |
| 12 | `…-12-integration-closeout.md` | Owned final gate: atomic tracker ownership and exact 0.7.1 version, lockfile/dependency reconciliation, full pytest/static/contracts/Wardline, PR checks plus exact-SHA RC-push CI/CodeQL/allowlist gates, run-scoped platform-bound candidate + qualified rollback-baseline images, reviewed saved-plan Terraform apply/destroy with durable interruption-safe cleanup state, baseline-first schema initialization, live Aurora/EFS/ECS/ALB task-role/non-root deployment and rollback acceptance, sanitized evidence, and tag/inventory-verified teardown; any failure hard-stops source/runtime-contract GO, which never authorizes a rebuilt or other-platform artifact | **All implementation plans 01–11 and 13** |
| 13 | `…-13-cognito-authorization-origin.md` | Authorization code + S256 PKCE for the browser public client; exact operator-declared HTTPS authorization/token origin; explicit Cognito `client_id` access-token validation; one-use expiring callback transaction | 01 · shared signed-tier/Wardline verification baseline |

## Filigree execution graph

The executable tracker graph is milestone `elspeth-6343920a47` (AWS ECS
runtime readiness for `release/0.7.1`). The plan-repair work that created and
verified this graph is tracked separately as `elspeth-e1e2e0f99f`; it is not
an implementation step. Start implementation work through Filigree's atomic
`work_start` / `work_start_next` surfaces. No implementation step is claimed
by this planning repair.

| Plan slice | Filigree step |
|------------|---------------|
| 01 | `elspeth-b9e8b5d24b` |
| 02 | `elspeth-9070fb0a45` |
| 03 base (Tasks 1–2) | `elspeth-dffe064287` |
| 03 integration proof (Task 3) | `elspeth-397ac915b8` |
| 04 | `elspeth-03cf981d4a` |
| 05 | `elspeth-1a1c31bcce` |
| 06 | `elspeth-7fe6aa531f` |
| 07 | `elspeth-74717426b7` |
| 08 gate-baseline prerequisite | `elspeth-8166b310e7` |
| 08A complete security gate (Tasks 1–3) | `elspeth-c0103e6c88` |
| 08B guided-source parity (Task 4) | `elspeth-a342f333a4` |
| 09 | `elspeth-e8dc754360` |
| 10 | `elspeth-6285c29c07` |
| 11 | `elspeth-25286192ee` |
| 12 | `elspeth-05396fed38` |
| 13 | `elspeth-5e729216f4` |

Plans 03 and 08 are split only at their real dependency boundaries. The shared
gate-baseline prerequisite is separately visible because it is operator/tooling
work and currently blocks 08A, then transitively Plans 06/07/09; it must not be
mistaken for application code.
Keeping either plan as one tracker node would create false readiness or a dependency cycle:
their base work can start earlier, while their integration/parity work must
wait for later plugin registrations. `filigree plan elspeth-6343920a47
--json` is the source of truth for current readiness.

## Execution order

- **Wave 1 (parallel foundations, then gated source integration):** the shared
  gate-baseline prerequisite may run in parallel with 01 and 02. Only
  08A waits for that prerequisite to close (the gate
  keys on plugin name + option key, so the policy, pipeline-validation wiring,
  and composer mutation backstops are implementable and testable before the plugins exist). Plan 06
  starts only after both 02 and all of 08A are done; this is enforced by the
  Filigree DAG, not left to merge prose. Merge 02 with its regenerated
  `uv.lock`, then implement/rebase 06 on that tree, rerun `uv lock`, and commit
  the freshly regenerated lockfile with 06; never text-merge `uv.lock`. Plans
  01 may merge wherever its ordinary file-level conflicts permit.
  This keeps 08's gate in the tree before any web-reachable `aws_s3`
  registration and gives the 02↔06 dependency additions one authoritative
  lockfile history.
- **Wave 2:** 03 Tasks 1–2 and 04 (need 01+02), then 05 (needs 04); 07 and 09 (both need 06; 07 also needs the
  already-closed 08A); 08B Task 4 is guided-source prompt parity and starts only
  after 06 and 08A are closed. Plan 07 extends the `aws` extra with
  Jinja2 for sink key templates and regenerates `uv.lock` a final time; Plans
  10/12 consume that 02 -> 06 -> 07 result. **Within-wave order:** merge 04
  before 05 — both edit the same `app.py:293-294` lifespan orphan call, and
  05's wrapper instructions assume 04's `create_tables` kwarg is already
  present at that site. Plan 09 waits for 06 because LiteLLM's real Bedrock
  transport requires the `aws` extra even though its unit tests mock the call.
- **Wave 3:** 03 Task 3 (integration proof — needs 06, 07, 09 all
  registered), 11 (needs 01, 02, 04), and 13 (needs 01) may proceed as their
  dependencies allow. Plan 10 starts last, only after every implementation
  slice in 01–09, 11, and 13 is done; the Filigree edges enforce this. Its
  Task 0 records that clean integrated pre-Plan-10 SHA as the only eligible
  Scenario B rollback-baseline source before packaging/runbook/harness edits.
- **Wave 4 — owned closeout, never parallel with implementation:** 12 starts
  only after every task and commit from 01–11 and 13 is present in one integrated
  `release/0.7.1` tree. A failure in 12 reopens the owning implementation
  surface; after repair, restart 12's gate sequence from the beginning.

**Security ordering constraint:** the plan-08 gate must be in the tree
before (never merely "in the same merge as") the first web-reachable
`aws_s3` registration from 06/07 — see the within-wave order above. The live
tracker enforces completion of all 08A Tasks 1–3 before Plan 06 starts; Tasks
1–2 are the mechanically load-bearing execution gate and Task 3 completes the
immediate authoring-surface defenses.
**Round 3 — this is now mechanically enforced, not a merge discipline:**
plan 06 Task 3 and plan 07 Task 1 each land a guard test in their
registration commits (`test_registered_aws_s3_source_is_endpoint_url_gated`
/ `test_registered_aws_s3_sink_is_endpoint_url_gated`) that drives
`validate_pipeline` and is deliberately red on any tree carrying an
`aws_s3` registration without plan 08A's core gate — a tree that merges 06/07
ahead of the gate cannot pass CI. A tree with 06/07 merged but not 08
carries the exact web-authorable exfiltration/SSRF surface the design
review rated Critical. Do not deploy the web app from such a tree. Likewise, do not call the tree aws-ecs-safe
until plan 11 has landed: without it, pipeline runs, tutorial projections,
and login events can still emit schema DDL from request paths.

## Cross-plan follow-ups (owed, tracked here so they don't vanish)

1. **CLI e2e acceptance for `endpoint_url` — RESOLVED, now in-plan.** The
   positive half of the acceptance criterion ("CLI/batch configs *can* set
   `endpoint_url`") is owned directly: plan 06 Task 3
   (`test_cli_aws_s3_endpoint_url_accepted`, source) and plan 07 Task 3
   (`test_cli_aws_s3_sink_endpoint_url_accepted`, sink), both driving
   `load_settings_from_yaml_string` + `instantiate_plugins_from_config` (the
   real acceptance gate — `SourceSettings`/`SinkSettings.options` are untyped
   dicts; only plugin instantiation runs the config's field validation).
   Plan 08 retains the import-boundary test (`elspeth.core.config` never
   imports `elspeth.web`) as the negative-half proof. No longer owed.
2. **Plan 01's extracted predicates** (`is_default_secret_key_placeholder`,
   `is_undersized_secret_key`, `is_uniform_byte_key` in `web/config.py`) are
   public and available to sibling implementers; they are not in the pinned
   brief.
3. **Full-program verification — RESOLVED, now executable as plan 12.** Plan
   12 owns the exact 0.7.1 version boundary, tracker-completion verification,
   the unscoped full-suite run, the CI-aligned static, formatting, typing,
   contract, Wardline, and lean-image gates, the hosted `CI Success` umbrella,
   and a live Aurora/EFS/ECS/ALB deployment, persistence, observation, and
   rollback rehearsal on the exact integrated `release/0.7.1` SHA. Scoped
   per-plan pytest runs remain useful during implementation, but they do not
   satisfy runtime acceptance.
4. **`aws_s3` class naming — RESOLVED in the Plan-06 repair.** Plans 06 and
   07 now use the house all-caps-acronym convention consistently:
   `AWSS3Source`/`AWSS3SourceConfig` and `AWSS3Sink`/`AWSS3SinkConfig`, matching
   `CSVSink`, `HTTPCallRequest`, and `LLMResponse`.
5. **Request-path landscape schema mutation — RESOLVED, now executable as
   plan 11.** The second review round (four independent no-go verdicts)
   correctly rejected this as a tracked-but-unowned follow-up: the plan
   set's headline claim ("aws-ecs never mutates schema outside doctor") was
   falsified by `execution/service.py:1100`, `composer/tutorial_service.py:204`,
   **and four `auth/audit.py` writer sites the review didn't name**
   (`:174,197,226,246` — found while scoping the fix; every login event
   could emit DDL under version skew). Plan 11 gates all six through
   `open_landscape_db(settings)` / an explicit `create_tables` field and
   seals the pattern with an AST guard. No longer owed.
6. **Session-schema PARTIAL asymmetry — RESOLVED by reclassification
   (plan 02).** The original design had the session probe report missing
   tables as PARTIAL (additively completable) *without verifying the
   present tables' shape* — `create_all` could then build around a
   divergent survivor set. Plan 02 now maps session partial-presence to
   STALE (fail-closed, drop/recreate), because `_validate_current_schema`
   raises on any table-set mismatch before checking shapes, making partial
   session state genuinely unverifiable. `probe_session_schema` never
   returns PARTIAL; `init_session_schema` creates on MISSING only.
   Residual (future, not owed for 0.7.1): making session honestly
   PARTIAL-aware needs `_validate_current_schema` to tolerate a partial
   table set.
7. **uv.lock ownership — RESOLVED by dependency order** (02 -> 06 -> 07):
   Plan 02 writes the `postgres` slice, Plan 06 rebases and regenerates for the
   initial `aws` slice, and Plan 07 finally regenerates after adding Jinja2 to
   the standalone source+sink `aws` extra. A text merge is forbidden. Plans
   10 and 12 consume and frozen-check Plan 07's final integrated lock.
8. **Trusted publication for the lean `INSTALL_EXTRAS` Docker build** (plan 10
   Task 1 verifies by real local `docker build`/`docker run`, but
   `build-push.yaml` never exercises the `INSTALL_EXTRAS` build-arg path,
   so a later Dockerfile edit could silently break the lean ECS image).
   Strengthening, not a spec requirement — the temporary acceptance tags are
   deleted before GO. Durable publication needs a separate trusted workflow
   that builds every approved platform with
   `--build-arg INSTALL_EXTRAS="webui llm aws postgres"`, runs the lean smoke,
   emits/verifies SBOM and provenance, and signs/verifies the digest before a
   release tag is created. The existing `build-push.yaml` default-`all`
   GHCR/ACR artifact is not that lean publication path.
9. **Real-S3 conditional-write proof — RESOLVED by the Plan-07 repair.**
   Plan 07 now adds both a real boto3/botocore `Stubber` lane and a slow
   default-chain real-S3 test. Plan 12 runs the live test with zero skips and
   proves first-create `IfNoneMatch`, cumulative `IfMatch`, fresh-sink
   collision, intervening-writer rejection, byte preservation, and cleanup.

## Review status

The initial ten-plan set (plans 01–10) was panel-reviewed by a
four-perspective Sonnet 5 panel
(solution architect, systems thinker, Python engineer/reality-checker,
quality engineer) plus API, threat, and LLM-safety specialists where the
slice warranted, and a single Fable reviewer on the plan-08 egress gate:
**3 critical, 57 high, 75 medium, 130 low** across those original ten plans.
Those counts predate plan 11. Every
critical and high was adjudicated and closed by a per-plan Sonnet 5 fixer
(each finding FIXED / SKIPPED-with-reason / REJECTED-verified-wrong on its
own evidence, never by agreement count); the fix round also surfaced two
new defects the panel missed (a schema-probe check-ordering bug in plan 02,
a startup-lifespan landscape-outage gap in plan 05), both fixed. A final
holistic Fable pass then read the spec, this index, and that ten-plan set
end-to-end for cross-plan coherence. The second review round added plan 11;
rounds two and three evaluated that expanded eleven-plan implementation set.
Plan 12 is the subsequent repair that turns the previously ownerless final
integration checks into an executable closeout gate.

**Second review round (post-commit, four independent read-only reviewers,
unanimous no-go — all six blockers verified real and repaired):**
(1) request-path landscape schema mutation was tracked-but-unowned →
**plan 11 added** (scope grew to six sites once the four `auth/audit.py`
writers were found); (2) session PARTIAL could `create_all` around
unverified surviving tables → plan 02 reclassifies session
partial-presence as fail-closed STALE; (3) doctor **and** readiness
writability probes `mkdir`'d missing directories, masking an unmounted EFS
volume → both now fail closed and the operator provisions all runtime
directories before doctor/startup (`--init-schema` creates schema only); (4) plan 03's integration proof needed
06/07/09 → dependency split, Task 3 moved to Wave 3; (5) plan 07's sink
config omitted the `csv_options`/`headers` defaults its own acceptance
test requires → field declarations now mirror `azure_blob_sink.py:178-185`;
(6) spec's unqualified 60s startup budget contradicted plan 04's per-probe
45s model → spec amended to the per-probe model (~60s nominal, ~90s
two-cold-cluster worst case, `startPeriod` sized to worst case). Warnings:
gate sequencing hardened to a within-wave merge order; probe semantics
unified across 03/05; plan 07's `get_agent_assistance` moved into its
registration commit; lean-image CI recorded as follow-up 8. The
`chat_solver.py` guided-parity warning was **initially mis-rejected here**
(a path typo — `composer/chat_solver.py` — read as file-not-found), then
verified REAL on re-check: `composer/guided/chat_solver.py:444` hardcodes
the guided-mode valid-source prompt list, and plan 06 carried the update
only as an ownerless "alongside plan 08" note. Now owned by **plan 08
Task 4**, sequenced after both the plugin (06) and the complete gate (08A Tasks 1–3)
exist. Recorded verbatim as a caution: a reviewer warning naming a file
you cannot find means *find the file*, not reject the warning.

**Third review round (post-repair, four independent read-only reviewers —
reality/architecture/quality/systems — full report in
`2026-07-08-aws-ecs.review.json`): 3× go-with-warnings, 1× no-go; both
no-go blockers verified against the tree and repaired, plus eight MEDIUMs
and the cheap LOWs folded in.** All six round-2 blocker repairs were
independently verified present in the plan text (three reviewers
re-censused every web-layer `LandscapeDB` construction; plan 11's six-site
enumeration is exact). The two new blockers were both *interactions with
unmodified existing code* — the class of defect per-plan review structurally
misses: (1) `create_app()`'s unconditional `data_dir` mkdir
(`app.py:818-819`) silently defeated the round-2 no-mkdir/unmounted-EFS
repair for the running web task (readiness would report READY while run
data landed on the overlay FS) → plan 04 Task 2 now gates it, fail-closed,
with tests; (2) the validate-only gate runs inside `create_app()` before
uvicorn binds the socket, so the port is *closed* (not 503) for up to the
~90s+ worst case, and only the container `startPeriod` was documented —
never the ECS service `healthCheckGracePeriodSeconds` that governs
ALB-driven task replacement → spec §Operational Budgets amended; plan 05
ops doc and plan 10 runbook now size both knobs with margin (~150s).
MEDIUM repairs: plan 11's `deployment_target="local"` → `"default"`
(plan 01's vocabulary); plan 11 now records the auth-audit fail-closed
posture as a decision with a pinning test (integrity over availability —
deliberate asymmetry with 04/05's graceful degrade); the 08-before-06/07
security ordering became mechanical via registration-commit guard tests;
doctor's cold-start fail-fast got a runbook precondition (plan 10 item 7);
the YAML-import route to the `validate_pipeline` chokepoint is pinned by a
test (plan 08 Task 2); readiness failures now emit structured logs
(plan 05 `_finalize`); plan 07's fake-only `IfNoneMatch` residual was
accepted in that round and was later closed by the dedicated Plan-07 repair
(follow-up 9); the Aurora no-epoch-backstop
limitation is stated in plan 02 and the spec. LOWs: `pg_advisory_lock`
wait bounded via `lock_timeout` + test (plan 02); 256 MiB exact-boundary,
one-over, and empty-object pins (plan 06); session-STALE doctor message
explains the interrupted-init case (plan 03); Wave-2 04-before-05 merge
order hardened; plan 05 round-3 accepted residuals recorded; `web` →
`webui` here. Not re-fixed (recorded): the AST guard's syntactic boundary
is now stated in plan 11; the former S3 casing split is resolved by follow-up
4 above.

**Fourth review round (release/0.7.1 repair, four independent lenses —
APPROVED for implementation):** the live-reality, architecture, quality, and
systems passes found no remaining blocker after repair. This round retargeted
the program and executable Filigree graph to `release/0.7.1`; repaired Plan
01's target check, redaction, environment provenance, commands, static gates,
and Wardline handoff; normalized every scoped Python test command to the
project environment; added Plan 12's owned exact-version/local/hosted/live
closeout; and added Plan 13's exact Cognito authorization-origin plus explicit
`client_id` access-token boundary. The final interaction pass also bound a
Plan-13-capable pre-Plan-10 rollback baseline, made the Cognito app-client and
four-phase browser evidence executable, and required aggregate failure-path
cleanup of both disposable stacks, identities, tags, images, and evidence
before Task 9 can issue GO. The only non-blocking release residual remaining
from that list is follow-up 8: hosted CI does not yet build the lean
`INSTALL_EXTRAS` path. Follow-up 9 is now closed by Plan 07's mandatory
real-S3 conditional-write lane. Accordingly, this is a **GO to execute the
implementation plans**, not runtime GO; runtime GO is owned solely by a fully
successful Plan 12.

**Dedicated Plan-13 review (2026-07-12 — APPROVED after repairs, dependency-
controlled):** three independent code-reality, security, and quality passes
found the original exact-origin/client-ID direction sound but rejected its
production use of Cognito's implicit token grant. The repaired plan now owns
authorization code + S256 PKCE, exact paired authorization/token endpoints,
issuer-bound discovery, parser-differential defenses, an expiring one-use
browser transaction, required Cognito access-token claims and RS256, redacted
configuration/token errors, full shared-provider/trust gates, isolated staging,
and atomic Filigree handoff. Plans 10/12 and the specification now require a
public client with no secret, code flow only, and live PKCE exchange. Execution
remains blocked on Plan 01 and `elspeth-8166b310e7`; review did not claim or
transition Plan 13.

**Dedicated Plan-09 review (2026-07-12 — APPROVED after repairs, not
startable yet):** three independent code-grounded passes plus installed
LiteLLM 1.85.0 inspection found and repaired the false no-dependency claim,
unsafe dynamic provider-property reads, incomplete config/lifecycle/redaction
tests, missing plugin hash/catalog golden/static/signed/Wardline gates, and the
absence of an ECS task-role Bedrock proof. Plan 09 now depends on Plan 06,
which provides boto3/botocore and transitively enforces the shared gate
baseline. Plans 10/12 now execute `verify-bedrock` inside each candidate ECS
task with zero skips/failures. Full evidence is in
`2026-07-08-aws-ecs-09-bedrock-provider.review.json`.
