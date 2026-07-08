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
`schema_probe.py` (`SchemaState`, `probe_/init_session|landscape_schema`),
`readiness.py` (`readiness_report`), `aws_s3_common.build_s3_client`,
provider key `bedrock` in `_PROVIDERS`, extras `postgres` and `aws`.

## Subplans

| # | File | Delivers | Depends on |
|---|------|----------|------------|
| 01 | `…-01-deployment-contract.md` | `deployment_target` setting; pure aws-ecs contract validator; placeholder predicates extracted from existing validators | — |
| 02 | `…-02-postgres-schema-support.md` | `postgres` extra; schema-shape probes on the owning modules; `schema_probe.py` with `pg_advisory_lock`-serialized init; pool kwargs passthrough | — |
| 03 | `…-03-doctor-cli.md` | `elspeth doctor aws-ecs [--init-schema] --json`; read-only by default; never opens `auth.db`; sanitized error classes | 01, 02 |
| 04 | `…-04-validate-only-startup.md` | Fail-closed startup schema validation in aws-ecs mode (incl. the hidden orphan-reconciliation `create_tables` path); 10 s/45 s/60 s budgets | 01, 02 |
| 05 | `…-05-readiness-endpoint.md` | `GET /api/ready` (2 s/check, 5 s ceiling, 2 s TTL cache, redacted); liveness-independence proof for `/api/health`; probe-wiring doc | 01, 02 |
| 06 | `…-06-s3-source.md` | `aws_s3` source; `max_object_bytes` (256 MiB default) enforced pre-read; `aws` extra; `aws_s3_common.build_s3_client` | build: — · **deploy: 08 gate must land first** (security ordering) |
| 07 | `…-07-s3-sink.md` | `aws_s3` sink; Jinja keys; `overwrite=False` via `put_object(IfNoneMatch="*")` (verified against botocore 1.43.42); catalog/determinism parity registrations | build: 06 · **deploy: 08 gate must land first** (security ordering) |
| 08 | `…-08-s3-endpoint-gate.md` | Web-authorship rejection of `aws_s3` `endpoint_url` (design review's Critical): `provider_config_policy` + `validate_pipeline` source/output adjudication + composer mutation paths | build: — (Tasks 1–2 wave 1) · Task 3 refs composer paths (wave 2) |
| 09 | `…-09-bedrock-provider.md` | LiteLLM-backed `bedrock` pipeline provider in `_PROVIDERS`; composer Bedrock-response parsing tests; opt-in live smoke | — |
| 10 | `…-10-packaging-docker.md` | Production install with extras `web,llm,aws,postgres` (no `--extra all`); Docker build; operator deployment docs | 02, 06, 09 |

## Execution order

- **Wave 1 (parallel-safe):** 01, 02, 06, 09 — plus 08 Tasks 1–2 (the gate
  keys on plugin name + option key, so the policy and pipeline-validation
  wiring are implementable and testable before the plugins exist).
- **Wave 2:** 03, 04, 05 (need 01+02); 07 (needs 06); 08 Task 3 (composer
  mutation paths).
- **Wave 3:** 10 (needs 02, 06, 09).

**Security ordering constraint:** the plan-08 gate must be in the tree
before (or in the same merge as) the first web-reachable `aws_s3`
registration from 06/07. A tree with 06/07 merged but not 08 carries the
exact web-authorable exfiltration/SSRF surface the design review rated
Critical. Do not deploy the web app from such a tree.

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
5. **Audit the remaining landscape create-if-missing paths** (surfaced by
   plan 02's and plan 04's fixers, out of their edit scope): TWO per-run
   sites construct a `LandscapeDB` that eagerly creates schema —
   `execution/service.py:1100` and `composer/tutorial_service.py:204`
   (`from_url` with `create_tables` defaulting True). Plan 04 gates the
   startup + orphan-reconciliation paths; these two per-run sites are
   separate create-if-missing surfaces that must also be validate-only in
   `aws-ecs` mode. Owed before the web app is considered aws-ecs-safe.
6. **Session-schema PARTIAL asymmetry** (documented inline in plan 02):
   `_validate_current_schema` raises on any table-set mismatch before
   checking column shapes, so the session probe cannot classify
   "existing-tables-only" as PARTIAL the way the landscape probe now does.
   Landscape honors the binding PARTIAL semantics; session's
   incomplete-but-correct-shape case still surfaces as a hard failure.
   Acceptable for the drop/recreate pre-1.0 posture, but the asymmetry is
   real — a future pass that makes session PARTIAL-aware needs surgery on
   `_validate_current_schema`.
7. **uv.lock parallel-wave conflict** (02 ↔ 06): both regenerate and commit
   `uv.lock` in wave 1 (02 for the `postgres` extra, 06 for `aws` — split
   ownership; plan 10 only consumes the result). Run in parallel they
   conflict on the lockfile; the second to merge must re-run `uv lock`
   (regenerate), never text-merge. Sequence the two lock commits, or land
   one then rebase the other.

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
