# AWS ECS Runtime Readiness — Plan Index

> **For agentic workers:** This is the index, not a plan. Execute the numbered
> subplans with superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans. Each subplan is independently testable and
> carries its own commits.

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
| 03 | `…-03-doctor-cli.md` | `elspeth doctor aws-ecs [--init-schema] --json`; read-only by default (incl. filesystem — probes never mkdir outside `--init-schema`, and never the `data_dir` root); never opens `auth.db`; sanitized error classes | Tasks 1–2: 01, 02 · **Task 3 (integration proof): also 06, 07, 09** |
| 04 | `…-04-validate-only-startup.md` | Fail-closed startup schema validation in aws-ecs mode (incl. the hidden orphan-reconciliation `create_tables` path); 10 s connect/45 s per-probe budgets (~60 s nominal, ~90 s worst case); aws-ecs startup never creates `data_dir` — missing root fails closed as an unmounted EFS volume | 01, 02 |
| 05 | `…-05-readiness-endpoint.md` | `GET /api/ready` (2 s/check, 5 s ceiling, 2 s TTL cache, redacted); liveness-independence proof for `/api/health`; probe-wiring doc | 01, 02 |
| 06 | `…-06-s3-source.md` | `aws_s3` source; `max_object_bytes` (256 MiB default) enforced pre-read; `aws` extra; `aws_s3_common.build_s3_client` | build: — · **deploy: 08 gate must land first** (security ordering) |
| 07 | `…-07-s3-sink.md` | `aws_s3` sink; Jinja keys; `overwrite=False` via `put_object(IfNoneMatch="*")` (verified against botocore 1.43.42); catalog/determinism parity registrations | build: 06 · **deploy: 08 gate must land first** (security ordering) |
| 08 | `…-08-s3-endpoint-gate.md` | Web-authorship rejection of `aws_s3` `endpoint_url` (design review's Critical): `provider_config_policy` + `validate_pipeline` source/output adjudication + composer mutation paths + guided-mode prompt parity (Task 4) | build: — (Tasks 1–2 wave 1) · Task 3 composer paths (wave 2) · Task 4 needs 06 + Tasks 1–3 (wave 2, last) |
| 09 | `…-09-bedrock-provider.md` | LiteLLM-backed `bedrock` pipeline provider in `_PROVIDERS`; composer Bedrock-response parsing tests; opt-in live smoke | — |
| 10 | `…-10-packaging-docker.md` | Production install with extras `webui,llm,aws,postgres` (no `--extra all`); Docker build; operator deployment docs incl. `startPeriod` **and** `healthCheckGracePeriodSeconds` sizing (the port is closed, not 503, during startup validation) and the first-deploy Aurora cold-start precondition | 02, 06, 09 |
| 11 | `…-11-landscape-write-gate.md` | `open_landscape_db(settings)` factory gating `create_tables` on deployment target; migrates the per-run, tutorial-projection, and four auth-audit landscape writers; AST guard banning ungated `LandscapeDB` construction under `web/` | 01, 02, 04 |

## Execution order

- **Wave 1 (parallel-safe):** 01, 02, 06, 09 — plus 08 Tasks 1–2 (the gate
  keys on plugin name + option key, so the policy and pipeline-validation
  wiring are implementable and testable before the plugins exist).
  **Within-wave order (hardened):** *merge* 08 Tasks 1–2 before *merging*
  06 — they can be built in parallel, but 08's gate commits land in the
  tree first, so no commit ever exists carrying a web-reachable `aws_s3`
  registration without the gate.
- **Wave 2:** 03 Tasks 1–2, 04, 05 (need 01+02); 07 (needs 06; merge after
  08 Tasks 1–2, same hardening); 08 Task 3 (composer mutation paths), then
  08 Task 4 (guided-mode prompt parity — needs 06 landed and 08 Tasks 1–3
  in the tree; never before the gate). **Within-wave order:** merge 04
  before 05 — both edit the same `app.py:293-294` lifespan orphan call, and
  05's wrapper instructions assume 04's `create_tables` kwarg is already
  present at that site.
- **Wave 3:** 03 Task 3 (integration proof — needs 06, 07, 09 all
  registered); 10 (needs 02, 06, 09); 11 (needs 01, 02, 04).

**Security ordering constraint:** the plan-08 gate must be in the tree
before (never merely "in the same merge as") the first web-reachable
`aws_s3` registration from 06/07 — see the within-wave order above.
**Round 3 — this is now mechanically enforced, not a merge discipline:**
plan 06 Task 2 and plan 07 Task 1 each land a guard test in their
registration commits (`test_registered_aws_s3_source_is_endpoint_url_gated`
/ `test_registered_aws_s3_sink_is_endpoint_url_gated`) that drives
`validate_pipeline` and is deliberately red on any tree carrying an
`aws_s3` registration without plan 08 Tasks 1–2 — a tree that merges 06/07
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
3. **Unscoped full-suite run at the slice boundary** before merging the
   whole epic to `release/0.7.0` (scoped per-plan pytest runs miss
   parity/baseline/lint-pin gates), plus one `wardline scan . --fail-on
   ERROR` over the new external-input boundaries (S3 source/sink, doctor,
   readiness).
4. **Unify the `aws_s3` class naming.** Plan 06 uses `AwsS3Source`/
   `AwsS3SourceConfig`; plan 07 uses `AWSS3Sink`/`AWSS3SinkConfig` (the
   house all-caps-acronym convention — `CSVSink`, `HTTPCallRequest`,
   `LLMResponse`). Unify on the **house form** (`AWSS3Source`/
   `AWSS3SourceConfig`) when 06 is implemented, so the same plugin family
   reads consistently. Neither fixer could edit the other's file; this is
   the reconciliation owner's call, recorded here so it is not lost.
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
   Residual (future, not owed for 0.7.0): making session honestly
   PARTIAL-aware needs `_validate_current_schema` to tolerate a partial
   table set.
7. **uv.lock parallel-wave conflict** (02 ↔ 06): both regenerate and commit
   `uv.lock` in wave 1 (02 for the `postgres` extra, 06 for `aws` — split
   ownership; plan 10 only consumes the result). Run in parallel they
   conflict on the lockfile; the second to merge must re-run `uv lock`
   (regenerate), never text-merge. Sequence the two lock commits, or land
   one then rebase the other.
8. **CI coverage for the lean `INSTALL_EXTRAS` Docker build** (plan 10
   Task 1 verifies by real local `docker build`/`docker run`, but
   `build-push.yaml` never exercises the `INSTALL_EXTRAS` build-arg path,
   so a later Dockerfile edit could silently break the lean ECS image).
   Strengthening, not a spec requirement — owed as a CI job that builds
   `--build-arg INSTALL_EXTRAS="webui llm aws postgres"` and runs
   `--version`, after 0.7.0 if need be.
9. **Optional LocalStack/real-S3 smoke for `IfNoneMatch` overwrite
   semantics** (round 3). Plan 07's overwrite tests all drive a hand-rolled
   fake, so the conditional-put behaviour itself is asserted against
   botocore's request model, never proven against real S3 — an accepted
   residual recorded in plan 07 Task 3. A LocalStack (or live-bucket) smoke
   exercising `overwrite=False` twice and asserting the second write raises
   would close it. Optional, after 0.7.0 if need be.

## Review status

Each subplan was panel-reviewed by a four-perspective Sonnet 5 panel
(solution architect, systems thinker, Python engineer/reality-checker,
quality engineer) plus API, threat, and LLM-safety specialists where the
slice warranted, and a single Fable reviewer on the plan-08 egress gate:
**3 critical, 57 high, 75 medium, 130 low** across the ten plans. Every
critical and high was adjudicated and closed by a per-plan Sonnet 5 fixer
(each finding FIXED / SKIPPED-with-reason / REJECTED-verified-wrong on its
own evidence, never by agreement count); the fix round also surfaced two
new defects the panel missed (a schema-probe check-ordering bug in plan 02,
a startup-lifespan landscape-outage gap in plan 05), both fixed. A final
holistic Fable pass reads the spec + this index + all ten amended plans
end-to-end for cross-plan coherence.

**Second review round (post-commit, four independent read-only reviewers,
unanimous no-go — all six blockers verified real and repaired):**
(1) request-path landscape schema mutation was tracked-but-unowned →
**plan 11 added** (scope grew to six sites once the four `auth/audit.py`
writers were found); (2) session PARTIAL could `create_all` around
unverified surviving tables → plan 02 reclassifies session
partial-presence as fail-closed STALE; (3) doctor **and** readiness
writability probes `mkdir`'d missing directories, masking an unmounted EFS
volume → both now fail closed, `--init-schema` alone creates (subdirs
only, never the `data_dir` root); (4) plan 03's integration proof needed
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
Task 4**, sequenced after both the plugin (06) and the gate (08 Tasks 1–3)
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
(plan 05 `_finalize`); plan 07's fake-only `IfNoneMatch` residual is
accepted in writing (+ follow-up 9); the Aurora no-epoch-backstop
limitation is stated in plan 02 and the spec. LOWs: `pg_advisory_lock`
wait bounded via `lock_timeout` + test (plan 02); 256 MiB exact-boundary,
one-over, and empty-object pins (plan 06); session-STALE doctor message
explains the interrupted-init case (plan 03); Wave-2 04-before-05 merge
order hardened; plan 05 round-3 accepted residuals recorded; `web` →
`webui` here. Not re-fixed (recorded): the AST guard's syntactic boundary
is now stated in plan 11; the `AwsS3Source`/`AWSS3Sink` casing split stays
follow-up 4.
