# AWS ECS Packaging & Deployment Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Give AWS ECS operators a Docker build path installing only production extras (no `--extra all`, no dev/test deps) plus a runbook documenting the ECS contract, secrets, auth, probe wiring, and Bedrock shape.

**Architecture:** Add an `INSTALL_EXTRAS` build ARG to the Dockerfile so the builder stage's `uv sync` installs a caller-supplied extras set; ECS builds pass `webui llm aws postgres`. Add `docs/runbooks/aws-ecs-deployment.md`, cross-linked from the runbooks index and Docker guide.

**Tech Stack:** uv, Docker multi-stage build, Markdown.

**Depends on:** `2026-07-08-aws-ecs-02-postgres-schema-support.md` (`postgres` extra + `schema_probe.py`; its Task 1 commits `pyproject.toml` **and** a regenerated `uv.lock` together), `2026-07-08-aws-ecs-06-s3-source.md` (`aws` extra + `aws_s3` plugins), and `2026-07-08-aws-ecs-09-bedrock-provider.md` (Task 2's Bedrock-ECS-shape content is sourced from the spec's Bedrock LLM Readiness section, not from plan 09's artifacts directly — no code dependency). Task 1 needs the `aws`/`postgres` extras in `pyproject.toml` *and* represented in `uv.lock`: plan 02 owns its own lock commit; plan 06 owns committing the `aws`-extra regeneration (binding decision — plan 10 references this precondition, it does not re-do the `uv lock` step itself).

**Deviations:** the brief's `web` extra is `webui` in `pyproject.toml:152`; used throughout. The shared GHCR/ACR image's default (`--extra all`, `Dockerfile:52,60`) stays unchanged — Azure Container Apps' Ansible deploy pulls that image (`docs/runbooks/ansible-ubuntu-deployment.md:2271`) and needs its Azure plugin pack. Production extras become an opt-in `INSTALL_EXTRAS` build arg for a lean ECS image instead, matching the spec's `webui,llm,aws,postgres` example.

**Global Constraints (spec §Packaging, §ECS Probe Wiring, §Local State And Auth, §Bedrock LLM Readiness):**
- "a production install with only production extras, for example web, llm, aws, and postgres. Dev and test dependencies, including testcontainers, should not be required in the final runtime image."
- "The existing non-root runtime user remains required. AWS credentials should be provided by ECS task roles and Secrets Manager injection, not baked into the image or command line."
- container `healthCheck`→`GET /api/health` (liveness only); ALB target group→`GET /api/ready`; `elspeth doctor aws-ecs`→one-shot pre-traffic `run-task`; `elspeth health` "is not part of the aws-ecs contract... must not be wired to any ECS probe."
- "Documentation must state that ECS uses AWS Secrets Manager and Azure deployments use the Azure equivalent."
- "Local auth is allowed as an explicit single-task option, with its SQLite auth DB on EFS. Cognito/OIDC remains the recommended production auth path." "`elspeth doctor aws-ecs` must not open `auth.db`." Journal mode "must not be WAL... the default rollback journal (`DELETE` mode) must be used instead."
- "Docs must show the expected ECS shape: task-role permissions for Bedrock, region, model identifier, and no embedded AWS keys."

---

### Task 1: Lean ECS production Docker build via `INSTALL_EXTRAS`

**Files:**
- Modify: `Dockerfile:50-52`, `Dockerfile:59-60` (the two `uv sync` calls)
- Modify: `tests/unit/test_build_push_release_checks.py:138,154-156,166-168` (three tests assert the exact `--extra all` literal this task removes)

`tests/unit/test_build_push_release_checks.py` already has a real pytest harness over the Dockerfile's text (`test_release_dockerfile_builds_frontend_dist_before_python_install`, `test_release_dockerfile_copies_local_uv_sources_before_dependency_sync`, `test_release_dockerfile_copies_frontend_dist_into_installed_package`) asserting the literal substrings `"uv sync --frozen --extra all --no-editable --active"` / `"...--no-install-project --active"`, both removed by this task's edit — update them below, do not leave them red. Beyond that harness, no pytest coverage exercises an actual `docker build`; runtime verification still uses `docker build`/`docker run` as red/green steps.

**Interfaces:**
- Consumes: `webui` extra (`pyproject.toml:152-165`), `llm` extra (`pyproject.toml:124-130`), `aws` extra (plan 06), `postgres` extra (plan 02)
- Produces: Dockerfile `ARG INSTALL_EXTRAS="all"` (default preserves shared-image behavior)

- [ ] RED (extras gate): `uv sync --frozen --extra webui --extra llm --extra aws --extra postgres --no-install-project --dry-run` must exit non-zero until plans 02/06 land (today: `error: Extra \`aws\` is not defined in the project's \`optional-dependencies\` table`, exit 2 — wording varies by uv version, so gate on exit code, not string match). uv reports this identical error whether an extra is missing from `pyproject.toml` or merely unrepresented in `uv.lock`; don't proceed to GREEN until it exits 0. If still non-zero after 02/06/09 have all landed, plan 06's `uv.lock` commit (see Depends-on) is the missing precondition — not something to fix here.
- [ ] RED (test harness): before editing the Dockerfile, update the three literal assertions in `tests/unit/test_build_push_release_checks.py` to the new stable substrings — `"uv sync --frozen $EXTRA_FLAGS --no-install-project --active"` and `"...--no-editable --active"` — leaving the ordering assertions (`npm run build` before install-project sync; that sync before the frontend-dist `copytree`) unchanged, since the edit doesn't reorder anything. Run `pytest tests/unit/test_build_push_release_checks.py -v` → the 3 tests fail with `ValueError: substring not found` (Dockerfile still says `--extra all` literally).
- [ ] Add `ARG INSTALL_EXTRAS="all"` above the first `uv sync`. Replace both hardcoded `--extra all` flags with a loop over `INSTALL_EXTRAS` (space-separated) building repeated `--extra <name>` flags, chained with `&&` throughout (not `;`) so any earlier step's failure — e.g. `uv venv` failing on an unwritable path — still aborts the `RUN` immediately instead of falling through to a `uv sync` that masks the real error:
  ```
  RUN uv venv /opt/venv && \
      . /opt/venv/bin/activate && \
      EXTRA_FLAGS="" && \
      for e in $INSTALL_EXTRAS; do EXTRA_FLAGS="$EXTRA_FLAGS --extra $e"; done && \
      uv sync --frozen $EXTRA_FLAGS --no-install-project --active
  ```
  Mirror into the second call (`. /opt/venv/bin/activate && EXTRA_FLAGS="" && for ... && uv sync --frozen $EXTRA_FLAGS --no-editable --active`).
- [ ] GREEN (test harness): `pytest tests/unit/test_build_push_release_checks.py -v` → 11/11 pass.
- [ ] GREEN, default unchanged: `docker build -t elspeth:default-test .` (no build-arg) succeeds; `docker run --rm elspeth:default-test --version` prints a line containing `elspeth version 0.7.0` (the real output of `cli.py`'s `version_callback`, not the bare version string).
- [ ] GREEN, default still bundles dev/test + Azure (inverse of the lean check, guards against the loop silently narrowing the default install): `docker run --rm --entrypoint python elspeth:default-test -c "import pytest, testcontainers, azure.storage.blob"` exits 0.
- [ ] GREEN, lean build: `docker build --build-arg INSTALL_EXTRAS="webui llm aws postgres" -t elspeth:ecs-prod-test .` succeeds; `docker run --rm elspeth:ecs-prod-test --version` prints a line containing `elspeth version 0.7.0` (exercises the actual lean-image deliverable, not just the default image).
- [ ] No dev/test deps (the Dockerfile's `ENTRYPOINT ["elspeth"]` intercepts bare args, so override it to reach a real interpreter): `docker run --rm --entrypoint python elspeth:ecs-prod-test -c "import importlib.util as u,sys; missing=[m for m in ('testcontainers','pytest','mypy','ruff') if u.find_spec(m) is None]; sys.exit(0 if len(missing)==4 else 1)"` exits 0.
- [ ] Production deps present: `docker run --rm --entrypoint python elspeth:ecs-prod-test -c "import boto3, psycopg, litellm, fastapi"` exits 0.
- [ ] Non-root user unchanged: `docker run --rm --entrypoint id elspeth:ecs-prod-test` prints output starting with `uid=1000(elspeth) gid=1000(elspeth)` (real `id` output appends a trailing `groups=...` field — match the prefix, not the full line).
- [ ] No baked-in AWS keys: `grep -iE "AWS_ACCESS_KEY|AWS_SECRET_ACCESS_KEY|aws_session_token" Dockerfile` and `docker history --no-trunc elspeth:ecs-prod-test | grep -iE "AWS_ACCESS_KEY|AWS_SECRET"` both return nothing.
- [ ] `git add Dockerfile tests/unit/test_build_push_release_checks.py && git commit -m "feat(docker): add INSTALL_EXTRAS build arg for lean AWS ECS production images"`

All GREEN steps above are real `docker build`/`docker run` invocations, not simulated — this task's verification is by executing the actual build, not by inspection. No CI job builds the `INSTALL_EXTRAS` path today (`build-push.yaml` never passes that build-arg); adding one is a reasonable follow-up but out of this task's scope since it's a strengthening suggestion, not a spec requirement.

### Task 2: AWS ECS deployment runbook + cross-links

**Files:**
- Create: `docs/runbooks/aws-ecs-deployment.md`
- Modify: `docs/runbooks/index.md:17` (Quick Reference table row)
- Modify: `docs/guides/docker.md:424-428` (See Also link only; its Kubernetes-probe section is unrelated and unchanged)

**Interfaces:** none (documentation only).

- [ ] Write `docs/runbooks/aws-ecs-deployment.md` (mirror the Symptoms/Prerequisites/Steps shape of `docs/runbooks/configure-keyvault-secrets.md`), covering exactly:
  1. **Contract summary** — `ELSPETH_WEB__DEPLOYMENT_TARGET=aws-ecs` requires PostgreSQL `session_db_url`/`landscape_url` (incl. `postgresql+psycopg`), writable `data_dir`/`payload_store_path`, rejects placeholder secrets (`ELSPETH_WEB__SECRET_KEY`, `ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY`); enforced by the sibling contract plan, described here for operators. State the general credentials principle once, here, for the whole ECS contract: "AWS credentials are provided by ECS task roles and Secrets Manager injection — never baked into the image or passed on the command line" (the Bedrock bullet below restates this only for its own case, not as the sole place it's said).
  2. **Secrets statement** (spec-required substance, plan-authored wording — not a verbatim spec quote): "ECS injects secrets via AWS Secrets Manager into the task definition; Azure deployments use the Azure equivalent (Key Vault — see [configure-keyvault-secrets.md](configure-keyvault-secrets.md))."
  3. **Auth** — Cognito/OIDC (`auth_provider=oidc`, Cognito User Pool OIDC endpoint; `oidc_issuer`/`oidc_audience`/`oidc_client_id`/`oidc_authorization_endpoint`, `config.py:226-229`) is recommended; local auth (`auth_provider=local`) is an explicit single-task option, `auth.db` on EFS (`app.py:832`); `elspeth doctor aws-ecs` never opens it; EFS journal mode stays `DELETE`, never WAL.
  4. **ECS Probe Wiring** — condensed 4-row table (container `healthCheck`→`/api/health`, liveness only; ALB target group→`/api/ready`; `elspeth doctor aws-ecs`→one-shot pre-traffic `run-task`; `elspeth health`→not wired to any probe), plus one line declaring this runbook the canonical operator entry point and cross-linking the fuller write-up plan 05 creates at `docs/operator/aws-ecs-health-and-readiness.md` — same four facts, kept in sync manually; don't let the two drift. Include the startup-window sizing obligation (round-3; authoritative wording in spec §Operational Budgets): the validate-only gate runs before uvicorn binds the socket, so the port is **closed** (connection-refused, not 503) for up to the two-cold-cluster worst case — set **both** the container `healthCheck.startPeriod` **and** the ECS service `healthCheckGracePeriodSeconds` to that worst case plus margin (~150s; the ~90s figure excludes probe time). With the default (zero) grace period, ALB kills a cold-starting task mid-validation into a restart loop that never converges.
  5. **Bedrock ECS shape** — task-role IAM for `bedrock:InvokeModel`, `region_name`, `bedrock/anthropic...` model id, no embedded AWS keys (references the principle stated in item 1).
  6. **Packaging** — the Task 1 `INSTALL_EXTRAS="webui llm aws postgres"` build with the exact `docker build` command; note that the lean image drops the `azure` extra (the Dockerfile's "All plugins bundled" comment no longer holds for it) — pipelines using `azure_blob` need the default `all` build or an expanded `INSTALL_EXTRAS`; and that operators must build-and-push their own tagged image (e.g. to ECR) for the ECS task definition to reference — the published GHCR/ACR image from `build-push.yaml` still uses the Dockerfile's default `--extra all` and is not a lean image.
  7. **Cold-start precondition for first deploy** — `elspeth doctor aws-ecs --init-schema` uses a single 10-second connection probe with **no retry budget** (fail-fast by design for an interactive/one-shot task, unlike web startup's 45s bounded retry); against a min-0-ACU Aurora cluster it can false-fail while the cluster wakes. Before the first `--init-schema`, either set a non-zero minimum ACU or issue a warm-up connection and re-run — doctor is idempotent and safe to re-run.
- [ ] Verify (run against the new file `F=docs/runbooks/aws-ecs-deployment.md`): `grep -q "AWS Secrets Manager" $F && grep -q "Azure equivalent" $F && grep -q "elspeth health" $F && grep -qi "not wired" $F && grep -qi "cognito\|oidc" $F && grep -qi "bedrock" $F && grep -q "INSTALL_EXTRAS" $F && grep -q "healthCheckGracePeriodSeconds" $F && grep -q "startPeriod" $F && grep -qi "minimum ACU\|min.*ACU" $F`; `grep -c "auth.db" $F` reports ≥3.
- [ ] Insert a row into `docs/runbooks/index.md`'s Quick Reference table: `| [AWS ECS Deployment](aws-ecs-deployment.md) | Deploying ELSPETH web to AWS ECS Fargate with Aurora PostgreSQL |`.
- [ ] Add a link under `docs/guides/docker.md`'s "See Also": `- [AWS ECS Deployment Runbook](../runbooks/aws-ecs-deployment.md) - Production ECS/Fargate deployment contract`.
- [ ] `git add docs/runbooks/aws-ecs-deployment.md docs/runbooks/index.md docs/guides/docker.md && git commit -m "docs: add AWS ECS deployment runbook"` (doc-only commit; `--no-verify` permitted per repo convention).
