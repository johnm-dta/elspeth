# Ansible Ubuntu Deployment Guide

## What This Document Is

**This document is a specification, not a shipped Ansible tree.** No
`deploy/ansible/` directory exists in this repository yet; the code
blocks below are the reference material from which an operations
repository (or a future in-tree `deploy/ansible/`) will be authored.

Concretely:

- **The role/playbook/inventory paths named below do not yet resolve.**
  `ansible-playbook -i inventories/non-cloud.yml playbooks/elspeth-vm.yml`
  is the *target* command; running it today fails because the files do
  not exist. The operator's first task is to transcribe the code blocks
  into a real Ansible tree (in an operations repo, or under
  `deploy/ansible/` in this repo) and adapt the variable values to
  their environment.
- **The `VERIFY BEFORE MERGE` markers scattered through code blocks
  refer to merging the transcribed code into the operator's Ansible
  tree** — i.e., they flag points where the API shape of an upstream
  Ansible module, Azure REST endpoint, or `ansible-core` Jinja
  type-preservation behavior varies across versions and must be
  re-confirmed against the operator's installed toolchain before the
  task is trusted. They are **not** unresolved TODOs in this document;
  they are operator-side verification points by design.
- **What this document does commit to**: the deployment topology
  (source-checkout VM, Azure Front Door fronting a VM, Container Apps
  with revision traffic-shifting), the gate sequence (CI verification
  → snapshot → deploy → probe → traffic-shift or rollback), the risk
  register, and the rationale for every load-bearing choice. Those are
  durable. The module-version-specific quirks are operator-side because
  they change faster than this document can.

This guide covers three deployment contexts:

- Non-cloud Ubuntu host running the source-checkout web service behind Caddy.
- Azure Ubuntu VM running the same service, published through Azure Front Door.
- Azure container deployment using a built container image and Azure Container
  Apps or a comparable container host.

Do not paste production secrets into inventory files; use Ansible Vault, an
external secret manager, or Azure Key Vault.

### Pre-Adoption Verification Checklist

Before relying on the transcribed playbooks in production, work through
the `VERIFY BEFORE MERGE` markers in your installed toolchain:

1. `azure.azcollection` version: confirm the `azure_rm_resource_info`
   failure shape for a 404 against your installed collection version
   (see "Distinguish 404 from auth / permission / network failures").
2. `ansible-core` version: confirm Jinja type-preservation for
   `template: "{{ elspeth_container_template.template }}"` (see
   "Container Template — Single Source Of Truth").
3. `caddy` version: confirm `format filter` field-redaction syntax
   against your installed Caddy version (the runbook is validated
   against Caddy 2.11.2).
4. Container Apps API version (`api_version: "2024-03-01"`):
   confirm `revisions` subresource shape and per-revision FQDN field
   path against the API version available in your subscription's
   region.

Each verification is a one-off probe — once observed and pinned, the
corresponding code block can be treated as load-bearing.

## Preflight Decisions

Resolve these decisions before adapting the playbooks:

| Decision | VM default | Azure VM + Front Door | Azure containers |
|----------|------------|-----------------------|------------------|
| Public hostname | `elspeth.example.com` points to the VM | Public hostname points to Front Door | Public hostname points to Container Apps or Front Door |
| Origin TLS | Caddy gets or reads the certificate | Caddy must present a certificate valid for the Front Door origin host header | Container Apps terminates TLS at ingress |
| Session database | SQLite only for single-worker installs | SQLite for small single-worker installs; PostgreSQL for scale or HA | External database required for production |
| Payload store | Filesystem path on local disk | Filesystem path on managed disk | Mounted filesystem volume; not container-local storage |
| Secrets | Ansible Vault or host secret manager | Ansible Vault, Key Vault, or managed identity handoff | Key Vault references with managed identity |
| Rollback unit | Previous Git commit/tag plus service restart | Previous Git commit/tag plus service restart | Previous immutable image tag or Container Apps revision |

Do not defer the origin TLS decision for Azure Front Door. If Front Door uses
`HttpsOnly`, the VM origin must serve a certificate that matches the
`origin_host_header`. Use DNS-01 ACME, a pre-provisioned certificate, or a
dedicated origin hostname with matching DNS and certificate automation. HTTP-01
certificate issuance can fail when the VM only accepts inbound traffic from
Azure Front Door.

## Deployment Matrix

| Context | Ubuntu target | Ingress | Runtime | Persistent state |
|---------|---------------|---------|---------|------------------|
| Non-cloud VM | 24.04 or 22.04 | Caddy terminates TLS | Source checkout, virtualenv, systemd, uvicorn UDS | Local disk or managed database |
| Azure VM | 24.04 or 22.04 | Azure Front Door, then Caddy on the VM | Source checkout, virtualenv, systemd, uvicorn UDS | Managed disk, Azure Database for PostgreSQL, or SQLite for small/single-worker installs |
| Azure containers | Ubuntu 24.04/22.04 image when policy requires Ubuntu; host OS is managed by Azure | Azure Container Apps ingress, optionally Azure Front Door | Container image | External database and mounted or managed payload storage |

ELSPETH requires Python `>=3.12` according to `pyproject.toml`. Ubuntu 24.04
normally satisfies this with distribution packages. Ubuntu 22.04 deployments
must install or provide Python 3.12 explicitly before creating the virtualenv.

## Repository Facts The Playbooks Must Preserve

The web deployment is a source-checkout service:

- Application import target: `elspeth.web.app:create_app`
- Frontend bundle served from: `src/elspeth/web/frontend/dist/`
- Backend dependency extra: `.[webui]`
- Azure plugin dependency extra when using Azure services: `.[azure]`
- Uvicorn transport: Unix domain socket at `/run/elspeth/uvicorn.sock`
- Static frontend build command: `npm run build` from `src/elspeth/web/frontend`
- Frontend build dependency: Node.js 24.x with npm 11.x
- Required non-local settings (web service refuses to start without these —
  no Python default):
  - `ELSPETH_WEB__SECRET_KEY` (non-loopback hosts; reject default)
  - `ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY` (HMAC for Phase 6A capability tokens)
  - `ELSPETH_WEB__COMPOSER_MAX_COMPOSITION_TURNS`
  - `ELSPETH_WEB__COMPOSER_MAX_DISCOVERY_TURNS`
  - `ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS`
  - `ELSPETH_WEB__COMPOSER_RATE_LIMIT_PER_MINUTE`
- Default web state under `ELSPETH_WEB__DATA_DIR`, with overrides:
  - `ELSPETH_WEB__SESSION_DB_URL`
  - `ELSPETH_WEB__LANDSCAPE_URL`
  - `ELSPETH_WEB__PAYLOAD_STORE_PATH`

**Two SQLite DBs, distinct trust tiers.** `sessions.db` and `audit.db` are
both SQLite, but they are NOT interchangeable. `audit.db` (Landscape) is
the project's legal record — back up before every deploy, never delete.
`sessions.db` was historically "in-flight composer scratch" but Phase 5b
(`interpretation_events`) and Phase 6A (`composer_completion_events`)
moved part of the composer audit surface into it. The "delete on schema
change" policy applies to `sessions.db` only and discards the composer-
side audit rows it carries; see **Database Lifecycle → Composer Audit-Table
Inventory In sessions.db** for the export-before-delete pattern.

Keep `WEB_CONCURRENCY` unset or set to `1` when using SQLite session state.
The web app has a startup guard for unsafe multi-worker SQLite deployments.

## Control Node Setup

Install Ansible and the Azure collection on the machine that runs the
playbooks:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ansible-core
ansible-galaxy collection install azure.azcollection
python -m pip install -r ~/.ansible/collections/ansible_collections/azure/azcollection/requirements.txt
```

For Azure resource provisioning, authenticate the control node with one of the
methods supported by `azure.azcollection`, for example a service principal:

```bash
export AZURE_SUBSCRIPTION_ID="00000000-0000-0000-0000-000000000000"
export AZURE_CLIENT_ID="00000000-0000-0000-0000-000000000000"
export AZURE_SECRET="use-ansible-vault-or-ci-secret-store"
export AZURE_TENANT="00000000-0000-0000-0000-000000000000"
```

Use `ansible-vault encrypt_string` or your CI secret store for real values.

## Suggested Ansible Layout

```text
ansible/
  inventories/
    non-cloud.yml
    azure-vm.yml
    azure-containers.yml
  group_vars/
    elspeth_web.yml
    azure.yml
    vault.yml
  playbooks/
    elspeth-vm.yml
    azure-vm-front-door.yml
    azure-containers.yml
  roles/
    elspeth_common/
    elspeth_web_source/
    caddy_origin/
    azure_vm/
    azure_front_door/
    azure_containers/
```

Split cloud resource creation from host configuration. That gives you a clean
failure boundary: Azure resources can converge before SSH-based VM setup runs.

## Deployment Gates And Rollback

Use the same release gates for non-cloud and Azure deployments:

1. **Pre-deploy CI gates pass** for the target commit, on a build host that is
   not the production host:
   - `pytest tests/`
   - `mypy src/`
   - `ruff check src/`
   - `python -m scripts.check_contracts`
   - `elspeth-lints check --rules trust_tier.tier_model --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model`
   - `elspeth-lints check --rules immutability.freeze_guards,immutability.frozen_annotations --root src/elspeth`
   Run the trust-tier gate from a CI or operator shell that has
   `ELSPETH_JUDGE_METADATA_HMAC_KEY`; the packaged linter fails closed without
   the key unless CI explicitly selects the fork-PR shape-only mode.
   The playbooks below do not run these gates; they assume the commit named in
   `elspeth_repo_version` has already passed them. The preflight role
   *does* machine-verify that the named SHA passed CI when
   `elspeth_ci_status_url` is configured — see "CI Status Verification"
   below. When it is not configured, the playbook prints a warning and
   proceeds; an unverified-SHA deploy is an operator decision, not a
   silent default.
2. **Pin `elspeth_repo_version` to a tag or SHA**, never a branch. Re-runs of
   the playbook with a branch reference will silently pull whatever the branch
   points to *now* and the deploy is no longer reproducible.
3. **Deploy to staging first.** The active staging target for this project is
   `elspeth.foundryside.dev` (source-checkout systemd/Caddy on the build host).
   Verify staging end-to-end before promoting the same `elspeth_repo_version`
   to production.
4. **Capture the resolved revision** post-deploy (`git rev-parse HEAD` task
   already in the role) and record it in the deployment log alongside the
   intended `elspeth_repo_version`.
5. **Back up `sessions.db` and `audit.db` before the upgrade** — see
   [Database Lifecycle](#database-lifecycle) below.
6. **Run smoke checks on the target host before changing public routing.**
7. **Verify `/api/health`** through the local transport and through the public
   edge.
8. **Watch the service** for at least one health-probe interval after cutover;
   trigger automatic rollback on N consecutive failures (see
   [Rollback Automation](#rollback-automation) below).
9. **Roll back** by redeploying the previous Git commit/tag or container image
   tag, restoring the pre-deploy database snapshot, then rerunning the same
   verification.

### CI Status Verification

The "Pre-deploy CI gates pass" step above is a **default-required
gate**. Three variables control it:

```yaml
# group_vars/elspeth_web.yml — example for GitHub status checks
elspeth_ci_status_verification_required: true   # default-true; flip to false ONLY for one-off dev deploys
elspeth_ci_status_url: "https://api.github.com/repos/your-org/elspeth/commits/{{ elspeth_repo_version }}/status"
elspeth_ci_status_headers:
  Authorization: "Bearer {{ vault_github_ci_token }}"
  Accept: "application/vnd.github+json"
elspeth_ci_status_success_path: "state"      # JSON field to read
elspeth_ci_status_success_value: "success"   # value that means "green"
```

**Default posture is strict.** When
`elspeth_ci_status_verification_required` is true (the default), the
preflight role requires `elspeth_ci_status_url` to be set AND requires
the CI status at that URL to report success. Either condition failing
refuses the deploy.

To run a one-off dev / canary deploy of an unverified SHA, the
operator must set `elspeth_ci_status_verification_required: false`
*as an explicit variable change in Git history*. Omitting
`elspeth_ci_status_url` while leaving `_required` at its default is
no longer a silent skip — the assert below refuses to proceed. Same
discipline as the HSTS one-way levers earlier in this section:
load-bearing posture changes are auditable variable bumps, not
omissions.

Preflight tasks:

```yaml
- name: Refuse deploy when CI status verification is required but unconfigured
  ansible.builtin.assert:
    that:
      - elspeth_ci_status_url is defined
      - elspeth_ci_status_url | length > 0
    fail_msg: >-
      elspeth_ci_status_verification_required=true (default) but
      elspeth_ci_status_url is not set. Either configure the CI status
      URL in group_vars, or explicitly set
      elspeth_ci_status_verification_required=false to acknowledge an
      unverified-SHA deploy. Refusing to proceed: silent omission of
      the URL is not an opt-out.
  when: elspeth_ci_status_verification_required | default(true) | bool

- name: Verify CI status for elspeth_repo_version
  ansible.builtin.uri:
    url: "{{ elspeth_ci_status_url }}"
    method: GET
    headers: "{{ elspeth_ci_status_headers | default({}) }}"
    return_content: true
    status_code: 200
  register: elspeth_ci_status
  when: elspeth_ci_status_verification_required | default(true) | bool

- name: Refuse deploy if CI status is not success
  ansible.builtin.assert:
    that:
      - (elspeth_ci_status.json[elspeth_ci_status_success_path | default('state')]
          | string) == (elspeth_ci_status_success_value | default('success'))
    fail_msg: >-
      CI status for {{ elspeth_repo_version }} is not
      "{{ elspeth_ci_status_success_value | default('success') }}".
      Refusing to deploy an unverified revision.
  when: elspeth_ci_status_verification_required | default(true) | bool

- name: Warn loudly when CI verification has been explicitly disabled
  ansible.builtin.debug:
    msg: >-
      WARNING: elspeth_ci_status_verification_required=false. This
      deploy is proceeding WITHOUT machine-verifying that
      {{ elspeth_repo_version }} passed pre-deploy CI gates. This is
      an operator-asserted unverified deploy. Confirm the variable
      bump appears in Git history alongside the deploy SHA.
  when: not (elspeth_ci_status_verification_required | default(true) | bool)
```

GitLab, Jenkins, and Azure DevOps all expose equivalent commit-status
endpoints; only the URL template and the JSON path of the success field
change. Vault the CI token, never check it in.

**Build-pipeline honesty.** The VM playbooks below are a *source-checkout
deployment*: `git pull`, `pip install -e`, and `npm ci` all run on the
production host. This is acceptable for single-host or small fleets where
ordered rollout is operator-driven, but it is **not** a "build once, deploy
everywhere" model — two hosts in a fleet can converge on materially different
bundles if PyPI or the npm registry shifts between runs. For multi-host
fleets, build a frontend artifact + Python wheel/venv tarball in CI, push
to an artifact store, and replace the per-host `git`/`pip`/`npm` tasks with
`ansible.builtin.unarchive` from that store. The container path in the
"Azure Containers Configuration" section already follows this model.

For VM deployments, the safest rollback unit is `elspeth_repo_version`. Keep
the previous known-good commit or tag in the release record:

```yaml
elspeth_repo_version: "542f34874"           # pinned SHA, not a branch
elspeth_previous_repo_version: "previous-known-good-sha"
```

For container deployments, use immutable tags such as `sha-<commit>`. Never use
`latest` for staging or production. Container Apps revisions can support gradual
traffic shifting; use that for canaries once health and telemetry gates exist.

## Common Variables

Put shared variables in `group_vars/elspeth_web.yml`:

```yaml
elspeth_user: elspeth
elspeth_group: elspeth
# Dedicated reverse-proxy socket group. Do NOT reuse elspeth_group here:
# elspeth_group owns app data, while this group only grants access to
# /run/elspeth/uvicorn.sock. Caddy must never receive app-secret/data group
# membership just to reach the Unix-domain socket.
elspeth_socket_group: elspeth-web-socket
elspeth_app_dir: /opt/elspeth
elspeth_repo_url: https://github.com/johnm-dta/elspeth.git
# Pin to a tag or commit SHA, never a branch. Branch references are not
# reproducible across re-runs of the playbook.
elspeth_repo_version: "542f34874"
elspeth_previous_repo_version: ""           # set before each deploy for rollback
elspeth_domain: elspeth.example.com
elspeth_python: /usr/bin/python3.12
elspeth_python_packages:
  - python3.12
  - python3.12-venv
  - python3.12-dev
elspeth_node_binary: /usr/bin/node
# How Node.js 24.x gets onto the host. Pick ONE:
#   - elspeth_node_packages populated with apt-installable names
#     (the apt source must ship Node 24.x; Ubuntu defaults do
#     not). Leave elspeth_install_nodesource_repo at false.
#   - elspeth_install_nodesource_repo: true
#     adds the official NodeSource Node 24.x repo before package install.
#     Sets elspeth_node_packages to ['nodejs'] if it's still empty.
#   - Bake Node.js into the VM image and leave both empty; the role's
#     verify-Node-version assert will accept any pre-installed Node
#     24.x at elspeth_node_binary.
elspeth_node_packages: []
elspeth_install_nodesource_repo: false
elspeth_system_packages:
  - build-essential
  - ca-certificates
  - curl
  - git
  - pkg-config
elspeth_extras:
  - webui
elspeth_web_bind_socket: /run/elspeth/uvicorn.sock
elspeth_data_dir: /var/lib/elspeth
elspeth_payload_store_path: /var/lib/elspeth/payloads
elspeth_session_db_url: sqlite:////var/lib/elspeth/sessions.db
elspeth_landscape_url: sqlite:////var/lib/elspeth/runs/audit.db
elspeth_registration_mode: closed
elspeth_auth_provider: local
elspeth_cors_origins:
  - https://elspeth.example.com

# Required WebSettings fields with NO Python default (Field(...)). The
# service refuses to start if any is unset; pick values appropriate to
# the deployment tier. Surfaced in the runbook (rather than left to env-
# var-discovery) because forgetting any one of them produces a
# pydantic.ValidationError at uvicorn bootstrap with no audit trail.
elspeth_composer_max_composition_turns: 25
elspeth_composer_max_discovery_turns: 6
elspeth_composer_timeout_seconds: 240
elspeth_composer_rate_limit_per_minute: 30

# ---- Composer LLM identity -------------------------------------------
# Model the composer-loop LLM runs against. Default in code is "gpt-5.5";
# staging or dev environments may want a cheaper model. See the Composer
# Runtime Tuning section below for the model-selection considerations.
elspeth_composer_model: gpt-5.5

# ---- Composer transport timing (browser/proxy abort racing) ----------
# These TWO knobs are the operator's lever against composer wall-clock
# failures racing browser/proxy idle aborts. The model validator
# enforces composer_timeout_seconds <= idle_ceiling - headroom; bumping
# composer_timeout_seconds without raising the ceiling FAILS BOOTSTRAP
# with a precise error message. See the "Composer Runtime Tuning"
# section for the full invariant and how to size against Caddy / Azure
# Front Door idle timeouts.
elspeth_composer_transport_idle_ceiling_seconds: 300.0
elspeth_composer_transport_headroom_seconds: 30.0
elspeth_composer_runtime_preflight_timeout_seconds: 5.0
elspeth_composer_max_tool_calls_per_turn: 16

# ---- Composer-LLM error surface --------------------------------------
# DEFAULTS TO FALSE. Set true only in dev/staging where exposing
# upstream provider error detail to the composer UI is acceptable.
# Production deployments MUST leave this false — surfacing provider
# errors leaks vendor-specific implementation details and may include
# request fingerprints in error bodies.
elspeth_composer_expose_provider_errors: false

# ---- Composer interpretation-event rate limits (F-30 / F-31) ---------
# Per-(session, user_term, composition_state_id) and per-session-per-UTC-
# day caps on request_interpretation_review tool calls. Exceeding either
# cap raises ToolArgumentError and the compose loop falls back to
# AUTO_INTERPRETED_NO_SURFACES. UTC-midnight window (not sliding 24h)
# for predictable reset behaviour.
elspeth_composer_interpretation_rate_limit_per_term: 3
elspeth_composer_interpretation_rate_limit_per_session_day: 10

# ---- Shareable-review token lifetime ---------------------------------
# Wall-clock validity stamped on freshly minted tokens. Default 30 days;
# compliance regimes may want shorter (e.g. 86400 = 24h). Tokens past
# this point fail signer.verify() with InvalidToken before any payload-
# store lookup runs.
elspeth_shareable_link_lifetime_seconds: 2592000     # 30 * 24 * 3600

# ---- Auth + JWKS tuning ----------------------------------------------
# auth_rate_limit_per_minute: per-IP auth-attempt rate limit.
# jwks_cache_ttl_seconds: how long a fetched JWKS document is reused.
# jwks_failure_retry_seconds: floor 10. Sized to prevent cold-start DoS
# during an IdP outage — the first caller pays the httpx timeout (~15s
# worst case) and subsequent callers short-circuit through this window.
# Do NOT lower below 10 even in test fixtures (the validator rejects it).
# jwks_max_stale_seconds: absolute lifetime of cached keys from the last
# successful JWKS fetch. Retry windows never renew this hard limit.
elspeth_auth_rate_limit_per_minute: 20
elspeth_jwks_cache_ttl_seconds: 3600
elspeth_jwks_failure_retry_seconds: 300
elspeth_jwks_max_stale_seconds: 86400

# ---- Upload + blob limits --------------------------------------------
# max_upload_bytes: per-request body cap on POST /api/blobs upload path.
# max_blob_storage_per_session_bytes: total blob storage one composer
# session may hold. Defaults are 100MB / 500MB; tune for tier.
elspeth_max_upload_bytes: 104857600                  # 100 * 1024 * 1024
elspeth_max_blob_storage_per_session_bytes: 524288000  # 500 * 1024 * 1024

# ---- Composer secret allowlist ---------------------------------------
# The closed list of env-var-named secrets the composer LLM may reference
# in pipeline configs via the {{ secrets.NAME }} substitution path.
# DefaultS to the four widely-used model-provider keys. Add an entry here
# (and provision the env var in the same deploy) before the LLM can wire
# a plugin that needs that key. Reserved-prefix names are rejected.
elspeth_server_secret_allowlist:
  - OPENROUTER_API_KEY
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
  - AZURE_API_KEY
  - AZURE_CONTENT_SAFETY_KEY

# ---- Payload-store retention -----------------------------------------
# Mirrors core/config.py:PayloadStoreSettings.retention_days. Surfaced
# on the audit-readiness panel as informational; the live retention
# enforcement lives in `elspeth purge`. Tune for the deployment's
# legal/compliance retention requirement.
elspeth_payload_store_retention_days: 90

# ---- Orphan-run cleanup ----------------------------------------------
# Background sweeper that marks abandoned runs `failed` after max_age.
# check_interval is the sweeper cadence; raise both for low-traffic
# deployments where premature reaping risks aborting a slow run.
elspeth_orphan_run_max_age_seconds: 3600
elspeth_orphan_run_check_interval_seconds: 300

# HSTS rollout. Every one of these levers is a one-way decision per
# browser session — once a client caches the policy, you cannot make
# it shorter or weaker for that client until the cached max-age
# elapses. Stage the ramp by bumping elspeth_hsts_max_age in discrete
# variable changes that show up clearly in Git history:
#
#   300       (5 min)   — first deploy of a new domain
#   86400     (1 day)   — after 24h of clean TLS operation
#   31536000  (1 year)  — production target, once TLS is fully stable
#
# elspeth_hsts_include_subdomains and elspeth_hsts_preload are even
# more binding than max-age:
#   - includeSubDomains extends the pin to every subdomain. Once
#     enabled, every subdomain MUST serve valid HTTPS for the full
#     max-age window or the subdomain becomes unreachable in
#     compliant browsers.
#   - preload submits the domain to the browser-shipped HSTS preload
#     list. Removal requires an explicit submission and takes weeks
#     to roll through browser releases; treat it as effectively
#     permanent.
#
# An auditor reading the Git history for "when did we commit to a
# 1-year HSTS pin" should see a discrete bump to elspeth_hsts_max_age,
# not a Caddyfile diff. Same for the other two flags.
elspeth_hsts_max_age: 300
elspeth_hsts_include_subdomains: false
elspeth_hsts_preload: false
```

The `auth_provider: local` + `registration_mode: closed` defaults are
operator-bootstrapped single-org defaults. For Azure Front Door deployments
exposed to a wider population, swap to OIDC (Entra) per the active Web auth
hardening cluster (filigree epic `elspeth-250f698aaf`). Leaving these as
`local`/`closed` in a publicly-reachable deployment means every account is
provisioned by hand against the local password backend — that is rarely
the right answer once the service is behind Azure Front Door.

Put secrets in `group_vars/vault.yml` and encrypt the file:

```yaml
vault_elspeth_web_secret_key: "replace-with-generated-token"
vault_openrouter_api_key: ""
vault_azure_openai_api_key: ""
vault_azure_openai_endpoint: ""
vault_fingerprint_key: "replace-with-stable-hmac-key"
# Phase 6A — shareable-review HMAC-SHA256 signing key. The web service
# refuses to start without this key (Field(...) with no default). MUST
# be ≥32 raw bytes; the documented generation recipe produces exactly
# 32 bytes of entropy as 44 base64 characters. Rotating this key
# invalidates EVERY outstanding shareable link, in-flight or not — there
# is no dual-key acceptance window in v1. Treat rotation as a deliberate
# operational event, not a routine secret refresh.
vault_elspeth_web_shareable_link_signing_key: "replace-with-generated-token"
# Only required when elspeth_install_nodesource_repo=true. Obtain
# from a trusted workstation:
#   curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sha256sum
vault_nodesource_apt_key_sha256: ""
```

Generate the web secret key on a trusted workstation:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

Generate the shareable-review signing key on a trusted workstation:

```bash
openssl rand -base64 32
```

The output is 44 base64 characters that decode to exactly 32 raw bytes —
HMAC-SHA256's digest size, which is the natural entropy floor for a tag
produced by this hash. The web settings validator decodes the base64
form, rejects shorter keys with a `ValueError`, and refuses to start on
non-loopback hosts when the decoded key is a uniform-byte placeholder
(e.g. `b"\x00" * 32`, `b"0" * 32`) — those are operationally
indistinguishable from "operator forgot to generate a real key."

## Composer Runtime Tuning

This section explains the composer-facing `WebSettings` knobs the
runbook surfaces as Ansible variables. Every variable named below has
a corresponding `elspeth_<name>` in **Common Variables** above and a
matching `ELSPETH_WEB__<NAME>` line in the env-file / Container Apps
template. The knobs are grouped by purpose; the operational
invariants (especially the transport-timing one) are load-bearing.

### Transport Timing — Browser/Proxy Idle-Abort Racing (CRITICAL)

The composer-loop POST is long-running by design — a single
compose request can take tens of seconds to a few minutes against
slower models. That puts it in the death zone for two concurrent
idle-abort budgets:

* **Browser fetch abort.** Browsers do not impose a fixed body-read
  timeout, but transparent proxies (CDN, Front Door, Caddy upstream)
  do. If the proxy aborts the request before the composer finishes,
  the client sees a generic 5xx with no audit trail and the LLM call
  may continue running on the backend.
* **Backend wall-clock.** The composer enforces its own
  `composer_timeout_seconds` budget so a stuck call doesn't waste an
  unbounded number of worker requests.

The runbook surfaces three knobs to keep these in correct ordering.
The invariant the `WebSettings` model-validator enforces is:

```text
composer_timeout_seconds <= composer_transport_idle_ceiling_seconds
                          - composer_transport_headroom_seconds
```

Violating it makes the service refuse to start with a precise error
message naming the three values. The intent of the invariant is that
**the backend gives up before the proxy gives up**, so the composer
emits an audit-trail-complete failure rather than a torn-mid-response
generic 5xx that the proxy attributes to the upstream.

**Sizing recipe for a typical reverse-proxy deployment:**

| Layer | Knob | Suggested value | Why |
|-------|------|-----------------|-----|
| Caddy / Front Door | upstream `idle_timeout` | 360s | Concrete budget on the proxy side; longer than the backend so the backend wins the race. |
| WebSettings | `composer_transport_idle_ceiling_seconds` | 300.0 | Backend's belief about the proxy ceiling; MUST stay strictly below the proxy's actual idle_timeout to leave network jitter slack. |
| WebSettings | `composer_transport_headroom_seconds` | 30.0 | Slack between the backend's all-stop and the ceiling — the period during which the backend writes the audit row, emits the failure event, and returns the response. |
| WebSettings | `composer_timeout_seconds` | 240 | Per-compose wall-clock budget. Must be `<= ceiling - headroom = 270`; 240 gives an additional 30s comfort margin. |

**Why these specific defaults are conservative.** The 30s headroom is
sized for the longest plausible single audit-emit-and-respond path
even under pressure: `INSERT` into the audit DB, optional payload-
store write, fingerprint sign, JSON serialise, gzip, network flush.
Shrinking headroom below 15s starts to risk the audit row landing
*after* the proxy decides the connection is dead — the backend's
response gets discarded and the client sees the proxy's 504. The
audit row is still durable, but the operator-side debugging story
breaks.

**Recipe for tightening or loosening:**

* **Loosen for slow models or large composition payloads.** Raise the
  proxy's idle_timeout first, then the backend's
  `composer_transport_idle_ceiling_seconds` (keep ~60s below the
  proxy), then `composer_timeout_seconds` (keep `<= ceiling - 30`).
* **Tighten for fast models and stricter SLOs.** Lower
  `composer_timeout_seconds` first; the ceiling/headroom can stay.
  Watch for compose loops that legitimately take longer than the new
  budget — they will fail-fast with `ComposerTimeoutError` audit
  events.

### Composer LLM Identity and Caps

`elspeth_composer_model` selects the model the composer loop runs
against. The shipped default (`gpt-5.5`) targets production tier;
staging / dev are best served with a smaller model (e.g.
`gpt-5.5-mini` or `o4-mini`) to keep iteration cost down. The model
identifier passes through to the LLM provider via the OpenRouter
abstraction; the composer does not validate the model exists at
config-load time — typos surface as 404s from the provider on first
compose POST.

Caps that bound a single compose request:

| Knob | Purpose | Notes |
|------|---------|-------|
| `composer_max_composition_turns` | LLM iteration budget per compose POST. Hitting it aborts with an audit-trail `ComposerBudgetExceeded` event. | Sized at 25 for production; staging may run higher for debugging. |
| `composer_max_discovery_turns` | Sub-budget for the discovery loop that fires before composition. | Smaller than the composition budget; 6 is a balanced starting point. |
| `composer_max_tool_calls_per_turn` | Per-turn cap on tool invocations. DoS-relevant: a runaway LLM cannot exhaust the audit DB by issuing thousands of tool calls in one turn. | Default 16. Lowering reduces noise on a particularly chatty model; raising can stall convergence on models that prefer many small calls. |
| `composer_runtime_preflight_timeout_seconds` | Validation timeout fired before run-execute. Catches malformed pipelines without running them. | 5s default. Raise only if your validation is unusually slow (rare). |

### Composer-LLM Error Surface

`composer_expose_provider_errors` controls whether upstream provider
error detail flows through to the composer UI. **Production must
leave this `false`.** Setting it `true` in production leaks
vendor-specific implementation details and may include
provider-side request fingerprints in error bodies, which surface to
non-operator users with composer access.

Dev / staging can set `true` while debugging an upstream issue —
revert before any production deploy. The runbook explicitly surfaces
this knob so an "expose for debug, forget to revert" leak is visible
in the diff of `group_vars/elspeth_web.yml`.

### Composer Interpretation-Event Rate Limits (F-30 / F-31)

The interpretation-event surface (Phase 5b) lets the composer LLM
flag ambiguous user terms for human review. Two caps prevent abuse:

| Knob | Cap | Window |
|------|-----|--------|
| `composer_interpretation_rate_limit_per_term` | Max times the LLM can surface the same `(session, user_term, composition_state_id)` tuple. | Lifetime of the composition state. |
| `composer_interpretation_rate_limit_per_session_day` | Max `request_interpretation_review` invocations per session per UTC day. | UTC midnight (NOT a sliding 24h). |

Either cap exceeded raises `ToolArgumentError`; the compose loop
falls back to `AUTO_INTERPRETED_NO_SURFACES` and the LLM is told to
make its best guess. Defaults (3 per term, 10 per session per day)
are sized for typical operator workloads — heavy multi-plugin
sessions may legitimately need a higher per-day cap.

The UTC-midnight reset (rather than sliding 24h) is deliberate:
operators reading the audit trail get a clean daily boundary, and a
session that exhausts its budget recovers predictably at the same
wall-clock time each day.

### Shareable-Review Token Lifetime

`shareable_link_lifetime_seconds` stamps `expires_at` on every
freshly-minted token. The signer rejects tokens past this point at
`ShareTokenSigner.verify()` before any payload-store lookup. Default
is 30 days (2 592 000s); compliance regimes may want shorter (24h =
86 400s).

There is no per-token revocation in v1 — the only emergency response
to a leaked link is to rotate `shareable_link_signing_key`, which
invalidates every outstanding link. See **Secret-Rotation Playbooks**
under Database Lifecycle for the rotation procedure.

### Auth Rate-Limit and JWKS Cache

| Knob | Purpose | Notes |
|------|---------|-------|
| `auth_rate_limit_per_minute` | Per-IP auth-attempt cap. | Default 20. Behind Front Door / Caddy this fires after the CDN cap; in direct-bind deployments it is the only auth-throttle. |
| `jwks_cache_ttl_seconds` | How long a fetched JWKS document is reused before re-fetch from the IdP. | 3600s default. Lowering accelerates key-rotation pickup at the cost of more IdP load. |
| `jwks_failure_retry_seconds` | Cool-down between JWKS re-fetch attempts when the IdP is unreachable. | Floor of 10 enforced. The first caller pays the httpx timeout (~15s); subsequent callers short-circuit to the stale JWKS via this window. Raising it makes the stale-serve window longer (safer during brief outages); lowering it shrinks the partial-DoS blast radius during a sustained outage. Do NOT lower the floor in test fixtures — the validator rejects values below 10. |
| `jwks_max_stale_seconds` | Hard upper bound on cached-key use after the last successful, fully validated JWKS fetch. | 86400s (24h) default; minimum 1. Transient refresh failures may serve stale keys only below this age. Once reached, authentication fails closed with 503 until a refresh succeeds; retry throttling still applies. |

### Upload and Blob Limits

| Knob | Default | Cap on |
|------|---------|--------|
| `max_upload_bytes` | 100 MB | Per-request body on `POST /api/blobs`. |
| `max_blob_storage_per_session_bytes` | 500 MB | Total blob storage one composer session may hold across all uploads. |

Both apply to user-driven file uploads (CSV / parquet / JSON sources
plugged into the composer). Tune for the deployment's typical
payload size — for image-heavy or large-CSV workflows, raise both.

### Composer Secret Allowlist

`server_secret_allowlist` is the closed list of env-var-named secrets
the composer LLM may reference in pipeline configs via
`{{ secrets.NAME }}` substitution. The default four entries
(`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`AZURE_API_KEY`) cover the widely-used model-provider keys. Adding
an entry requires both (a) appending to this list, and (b)
provisioning the env var in the same deploy — the allowlist tracks
*permitted names*, not *available values*. Names matching the
reserved prefix (per `validation.SERVER_SECRET_RESERVED_PREFIX`) are
rejected at config-validation time.

### Payload-Store Retention

`payload_store_retention_days` mirrors the core
`PayloadStoreSettings.retention_days` default. The audit-readiness
panel surfaces this value as informational so operators see what
retention the composer believes is in effect.

The **live retention enforcement** is `elspeth purge`, not the web
service. Configure a periodic systemd timer or cron entry to run
`elspeth purge --retention-days $(retention_days)` against the
deployment's payload store. Setting the WebSettings value without
running `elspeth purge` produces an audit-readiness panel that
misrepresents the actual retention horizon.

### Orphan-Run Cleanup

| Knob | Default | Purpose |
|------|---------|---------|
| `orphan_run_max_age_seconds` | 3600 | Runs without progress updates for this long get marked `failed` by the sweeper. |
| `orphan_run_check_interval_seconds` | 300 | Sweeper cadence. |

For low-traffic deployments where a slow legitimate run could exceed
`max_age`, raise both values. For high-traffic deployments where
abandoned runs accumulate fast, the defaults are appropriate.

### Composer-Advisor Escape Hatch — Future-Release

The `composer_advisor_*` settings (enabled, model, budgets,
timeouts) wire an escape hatch that lets the composer LLM consult a
frontier model when stuck on a particular composition. This feature
is **disabled by default** and is queued behind further hardening —
do NOT enable it in production today. The settings exist in
`WebSettings` so the wiring is in place, but the operator-runbook
posture for v1 is "leave disabled."

When the feature is approved for general availability, this section
will document the budget knobs (`max_calls_per_compose`,
`max_prompt_tokens`, `max_completion_tokens`, `timeout_seconds`) and
the operational implications of enabling it (cost exposure,
auditability of escalations, model-provider dependency).

## Ubuntu 24.04 And 22.04 Package Handling

Use the same role for both Ubuntu releases, but make Python provisioning
explicit. The frontend lockfile currently includes packages that require
Node.js 24.x, so do not rely on the Ubuntu
`nodejs` package unless your apt source is pinned to a compatible version.

```yaml
- name: Verify supported Ubuntu release
  ansible.builtin.assert:
    that:
      - ansible_distribution == "Ubuntu"
      - ansible_distribution_version in ["22.04", "24.04"]
    fail_msg: "ELSPETH Ansible deployment supports Ubuntu 22.04 and 24.04."

- name: Optionally add an organization-approved Python 3.12 repository on Ubuntu 22.04
  ansible.builtin.apt_repository:
    repo: "{{ elspeth_python312_apt_repository }}"
    state: present
    update_cache: true
  when:
    - ansible_distribution_version == "22.04"
    - elspeth_python312_apt_repository is defined

- name: Provision NodeSource Node 24.x apt repo (optional)
  when: elspeth_install_nodesource_repo | default(false) | bool
  block:
    - name: Ensure NodeSource apt key directory exists
      ansible.builtin.file:
        path: /etc/apt/keyrings
        state: directory
        owner: root
        group: root
        mode: "0755"

    - name: Install NodeSource apt key
      ansible.builtin.get_url:
        url: https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key
        dest: /etc/apt/keyrings/nodesource.asc
        owner: root
        group: root
        mode: "0644"
        checksum: "sha256:{{ vault_nodesource_apt_key_sha256 }}"
      # The checksum pin is the supply-chain defense: if NodeSource is
      # compromised and serves a different key, get_url fails the task
      # before the key is written. The SHA256 lives in the vault
      # (vault.yml: vault_nodesource_apt_key_sha256) so it survives
      # rotations cleanly. To obtain the current value, run from a
      # trusted workstation:
      #
      #   curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      #     | sha256sum
      #
      # Capture the output, cross-check against the NodeSource
      # signature documentation (https://github.com/nodesource/distributions),
      # and commit the value to the encrypted vault.yml. Rotate when
      # NodeSource publishes a new key (rare — typically once every
      # 2-3 years).
      #
      # If you prefer to defer the pin to a follow-up commit (NOT
      # recommended for production), comment out the `checksum:` line
      # and accept the risk. The risk-register entry "NodeSource apt
      # key fetched without checksum pin" applies.

    - name: Configure NodeSource apt source for Node 24.x
      ansible.builtin.copy:
        dest: /etc/apt/sources.list.d/nodesource.list
        owner: root
        group: root
        mode: "0644"
        content: |
          deb [signed-by=/etc/apt/keyrings/nodesource.asc] https://deb.nodesource.com/node_24.x nodistro main

    - name: Add nodejs to the install list when NodeSource is in use
      ansible.builtin.set_fact:
        elspeth_node_packages: "{{ ['nodejs'] if (elspeth_node_packages | length == 0) else elspeth_node_packages }}"

- name: Install system packages
  ansible.builtin.apt:
    name: "{{ elspeth_system_packages + elspeth_python_packages + elspeth_node_packages }}"
    state: present
    update_cache: true

- name: Verify Python 3.12 is available
  ansible.builtin.command: "{{ elspeth_python }} --version"
  register: elspeth_python_version
  changed_when: false

- name: Fail if Python is not 3.12+
  ansible.builtin.assert:
    that:
      - elspeth_python_version.stdout is match('Python 3\\.(1[2-9]|[2-9][0-9])\\.')
    fail_msg: "Install Python 3.12 or newer before deploying ELSPETH."

- name: Verify Node.js is available
  ansible.builtin.command: "{{ elspeth_node_binary }} --version"
  register: elspeth_node_version
  failed_when: false
  changed_when: false

- name: Fail with an actionable message if Node.js is missing
  ansible.builtin.assert:
    that:
      - elspeth_node_version.rc == 0
    fail_msg: >-
      Node.js was not found at {{ elspeth_node_binary }}
      (exit code {{ elspeth_node_version.rc }}). The frontend bundle
      build needs Node.js 24.x. Resolve by one of:
        (a) set elspeth_node_packages to a list installable by apt
            (e.g. ['nodejs'] when your apt source ships Node 24.x),
        (b) enable elspeth_install_nodesource_repo=true to add the
            official NodeSource apt repo for Node 24.x,
        (c) bake Node.js into the VM image and set elspeth_node_binary
            to its install path.

- name: Fail if Node.js is too old for the frontend lockfile
  ansible.builtin.assert:
    that:
      - elspeth_node_version.stdout is match('v24\\.')
    fail_msg: >-
      Node.js at {{ elspeth_node_binary }} is
      {{ elspeth_node_version.stdout }}, but the frontend lockfile
      requires Node.js 24.x. The default Ubuntu nodejs package depends
      on the apt mirror's snapshot and may be too old.
      Resolve by one of:
        (a) point elspeth_node_packages at a newer apt source,
        (b) set elspeth_install_nodesource_repo=true to provision the
            NodeSource Node 24.x repo,
        (c) re-bake the VM image with Node.js 24.x pre-installed.
```

If your organization does not permit third-party apt repositories on Ubuntu
22.04, bake Python 3.12 into the image and set `elspeth_python` to that path.
Use the same image-baking or approved-repository approach for Node.js.

## VM Source-Checkout Role

The VM role should install the app, build the frontend, template the environment
file, and restart systemd only when inputs change.

```yaml
- name: Refuse per-host source-checkout build on multi-host plays
  ansible.builtin.assert:
    that:
      - (play_hosts | length <= 1) or
        (elspeth_allow_per_host_build | default(false) | bool)
    fail_msg: >-
      This role builds the frontend and installs the Python venv on the
      production host. With more than one host in the play, two hosts can
      converge on materially different bundles if PyPI or the npm registry
      shifts between runs (see "Build-pipeline honesty" above). Either
      reduce the play to a single host, switch to the build-artifact
      playbook variant, or set elspeth_allow_per_host_build=true to
      acknowledge the risk. play_hosts={{ play_hosts | length }}.
  # play_hosts (the hosts in scope for THIS run) is the correct gate, not
  # groups['elspeth_web'] (the full inventory). A 10-host inventory run
  # with --limit elspeth-prod-1 is a single-host play and must pass.

- name: Create ELSPETH group
  ansible.builtin.group:
    name: "{{ elspeth_group }}"
    system: true

- name: Create ELSPETH socket-access group
  ansible.builtin.group:
    name: "{{ elspeth_socket_group }}"
    system: true

- name: Create ELSPETH user
  ansible.builtin.user:
    name: "{{ elspeth_user }}"
    group: "{{ elspeth_group }}"
    home: "{{ elspeth_app_dir }}"
    shell: /usr/sbin/nologin
    system: true

- name: Create data directories with owner-only access
  # mode 0700 — only the elspeth user can enter elspeth_data_dir. The
  # earlier 0750 leaked the audit DB to anyone in the elspeth group;
  # Caddy is in that group for UDS access (/run/elspeth/uvicorn.sock,
  # managed by systemd's RuntimeDirectory and granted by the socket's
  # 0660 mode), so 0750 here meant Caddy could read /var/lib/elspeth/
  # at file-level. Caddy has no business reading the audit DB; the
  # "Access-Log Hygiene" section's blast-radius argument applies here
  # in the inverse direction. 0700 closes the path; UDS access is
  # unaffected because it lives under /run, not /var/lib/elspeth.
  ansible.builtin.file:
    path: "{{ item }}"
    state: directory
    owner: "{{ elspeth_user }}"
    group: "{{ elspeth_group }}"
    mode: "0700"
  loop:
    - "{{ elspeth_data_dir }}"
    - "{{ elspeth_data_dir }}/runs"
    - "{{ elspeth_payload_store_path }}"

- name: Refuse to deploy with an unpinned repo version
  ansible.builtin.assert:
    that:
      - elspeth_repo_version | length > 0
      - elspeth_repo_version is match('^([0-9a-f]{7,40}|v?[0-9]+\\.[0-9]+\\.[0-9]+([.-][A-Za-z0-9.+-]+)?)$')
    fail_msg: >-
      elspeth_repo_version must be a commit SHA or tag, not a branch.
      Got: {{ elspeth_repo_version }}

- name: Read SHA that last deployed against this DB (if any)
  # The project uses SQLAlchemy `metadata.create_all()` for the
  # session DB and does NOT write PRAGMA user_version or any other
  # in-DB schema marker. So the runbook cannot PRAGMA-check schema
  # compatibility. Instead, we write a sidecar marker on every
  # successful deploy recording the SHA-that-last-deployed-against-
  # this-DB. On the next preflight, we read the marker. If the
  # incoming SHA differs from the last-deploy SHA AND the operator
  # has not explicitly acknowledged a schema-drift deploy, refuse.
  # The project policy is "operator deletes sessions.db on schema
  # change" (see "Database Lifecycle"); this preflight makes that
  # decision explicit instead of silent. The marker is rotated to
  # the new SHA at the end of a successful deploy.
  ansible.builtin.slurp:
    src: "{{ elspeth_data_dir }}/.last-deploy-sha"
  register: elspeth_last_deploy_sha_raw
  failed_when: false                # first deploy: marker does not exist
  changed_when: false

- name: Compute previous-deploy SHA from marker
  ansible.builtin.set_fact:
    elspeth_previous_deploy_sha: >-
      {{ (elspeth_last_deploy_sha_raw.content | b64decode | trim)
         if (elspeth_last_deploy_sha_raw.content is defined)
         else '' }}

- name: Refuse schema-drift deploy without explicit acknowledgement
  # If the SHA changed since the last deploy, the running code may
  # expect a different session-DB schema than what's on disk. The
  # project's pre-1.0 policy is "operator deletes sessions.db on
  # schema change" — but "did the schema change?" is currently a
  # human decision, not a mechanical one. Either:
  #   (a) operator confirms no schema change: set
  #       elspeth_acknowledge_schema_compatible=true (the deploy is
  #       a code-only update, not a schema change), or
  #   (b) operator confirms schema change: set
  #       elspeth_force_session_db_delete=true (the role deletes
  #       sessions.db post-deploy; in-flight composer sessions
  #       are lost as documented in "Database Lifecycle"), or
  #   (c) neither set: refuse, because silently starting the new
  #       code against the old schema either crashes on first write
  #       or — worse — opens, mismatches, and corrupts in-flight
  #       rows.
  ansible.builtin.assert:
    that:
      - (elspeth_previous_deploy_sha == elspeth_repo_version)
        or (elspeth_acknowledge_schema_compatible | default(false) | bool)
        or (elspeth_force_session_db_delete | default(false) | bool)
    fail_msg: >-
      sessions.db was last touched by SHA={{ elspeth_previous_deploy_sha }};
      this deploy is SHA={{ elspeth_repo_version }}. Either set
      elspeth_acknowledge_schema_compatible=true (code-only update, no
      schema change), or set elspeth_force_session_db_delete=true (schema
      changed; delete sessions.db and drop in-flight composer sessions).
      Project policy is "operator deletes sessions.db on schema change";
      this gate makes the decision explicit. See "Database Lifecycle".
  when: elspeth_previous_deploy_sha | length > 0

- name: Probe whether each SQLite DB exists (first deploy: they don't yet)
  ansible.builtin.stat:
    path: "{{ item }}"
  loop:
    - "{{ elspeth_data_dir }}/sessions.db"
    - "{{ elspeth_data_dir }}/runs/audit.db"
  register: elspeth_db_stat

- name: Snapshot session and audit databases before upgrade (WAL-safe)
  # Use sqlite3 ".backup" rather than ansible.builtin.copy. The .backup
  # command invokes the SQLite Online Backup API, which serializes
  # against writers and produces a consistent point-in-time copy
  # regardless of WAL/SHM state. A raw byte-copy under WAL captures
  # the main .db file but NOT the .db-wal sidecar — any committed-
  # but-not-yet-checkpointed transactions are absent from the
  # snapshot, and the "atomic" snapshot is actually torn. See
  # https://www.sqlite.org/backup.html for the API semantics.
  #
  # The .backup destination path is passed UNQUOTED in argv[2]: argv
  # bypasses the shell, so any literal `'` characters would land
  # inside SQLite's parser as part of the filename rather than being
  # stripped. We rely on elspeth_data_dir + the SHA-and-timestamp
  # suffix containing no shell-significant characters — both are
  # asserted earlier in the role (data_dir under /var/lib/elspeth
  # by convention; SHA matches a strict regex). If you change
  # elspeth_data_dir to a path containing whitespace, use
  # ansible.builtin.shell with explicit quoting instead.
  ansible.builtin.command:
    argv:
      - sqlite3
      - "{{ item.item }}"
      - ".backup {{ item.item }}.pre-deploy-of-{{ elspeth_repo_version }}-{{ ansible_date_time.iso8601_basic_short }}"
  loop: "{{ elspeth_db_stat.results }}"
  loop_control:
    label: "{{ item.item }}"
  when: item.stat.exists
  register: elspeth_db_snapshot
  changed_when: true
  # The snapshot filename embeds the SHA we are ABOUT TO DEPLOY, not the
  # SHA currently live. To roll back FROM a failing SHA Y back TO the
  # prior X, the rollback role selects the snapshot named
  # "pre-deploy-of-Y-*" — i.e., it restores the state that existed
  # *before* Y started writing to the DB. See "Rollback Automation".

- name: Tighten snapshot permissions to owner-only
  ansible.builtin.file:
    path: "{{ item.item }}.pre-deploy-of-{{ elspeth_repo_version }}-{{ ansible_date_time.iso8601_basic_short }}"
    owner: "{{ elspeth_user }}"
    group: "{{ elspeth_group }}"
    mode: "0600"
  loop: "{{ elspeth_db_stat.results }}"
  loop_control:
    label: "{{ item.item }}"
  when: item.stat.exists
  # sqlite3 .backup writes the file as the invoking user (root, when
  # become: true) with default umask. Re-stat to elspeth:elspeth 0600
  # so the snapshot has the same owner-only readability as the live
  # DB after the elspeth_data_dir tightening (see "Common Variables").

- name: Check out ELSPETH
  ansible.builtin.git:
    repo: "{{ elspeth_repo_url }}"
    dest: "{{ elspeth_app_dir }}"
    version: "{{ elspeth_repo_version }}"
    update: true
  become_user: "{{ elspeth_user }}"
  notify: restart elspeth web

- name: Record deployed Git revision
  ansible.builtin.command: git rev-parse HEAD
  args:
    chdir: "{{ elspeth_app_dir }}"
  register: elspeth_deployed_revision
  changed_when: false

- name: Create virtualenv
  ansible.builtin.command:
    cmd: "{{ elspeth_python }} -m venv {{ elspeth_app_dir }}/.venv"
    creates: "{{ elspeth_app_dir }}/.venv/bin/python"
  become_user: "{{ elspeth_user }}"

- name: Install Python dependencies
  ansible.builtin.command:
    argv:
      - "{{ elspeth_app_dir }}/.venv/bin/python"
      - -m
      - pip
      - install
      - -e
      - ".[{{ elspeth_extras | join(',') }}]"
    chdir: "{{ elspeth_app_dir }}"
  become_user: "{{ elspeth_user }}"
  notify: restart elspeth web

- name: Pre-compile Python bytecode
  ansible.builtin.command:
    argv:
      - "{{ elspeth_app_dir }}/.venv/bin/python"
      - -m
      - compileall
      - -q
      - "{{ elspeth_app_dir }}/src"
  become_user: "{{ elspeth_user }}"
  changed_when: false
  notify: restart elspeth web
  # Runtime mount is read-only (ProtectSystem=strict makes /opt read-only,
  # and the elspeth user's home is the app dir). If bytecode is not
  # precompiled at deploy time, the first import attempt will silently fail
  # to write a __pycache__ entry on every cold start.

- name: Install frontend dependencies (reproducible)
  ansible.builtin.command:
    cmd: npm ci
    chdir: "{{ elspeth_app_dir }}/src/elspeth/web/frontend"
  become_user: "{{ elspeth_user }}"
  # `npm ci` is the production default: it refuses to run if package.json
  # and package-lock.json disagree, and it does not mutate the lockfile.
  # Use `npm install` only as a recovery procedure when the lockfile is
  # genuinely stale, and re-commit the regenerated lockfile from a
  # workstation — never from the production host.

- name: Build frontend bundle
  ansible.builtin.command:
    cmd: npm run build
    chdir: "{{ elspeth_app_dir }}/src/elspeth/web/frontend"
  become_user: "{{ elspeth_user }}"
  notify: restart elspeth web

- name: Add caddy user to socket-access group (UDS access only)
  ansible.builtin.user:
    name: caddy
    groups: "{{ elspeth_socket_group }}"
    append: true
  notify: restart caddy
  # The systemd unit runs with Group={{ elspeth_socket_group }} and
  # UMask=0007, so uvicorn creates /run/elspeth/uvicorn.sock as mode
  # 0660 owned by elspeth:{{ elspeth_socket_group }}. Debian's Caddy
  # package runs as user `caddy`; membership in this dedicated group is
  # enough for reverse_proxy UDS access without granting Caddy read access
  # to /etc/elspeth/elspeth-web.env or app data owned by elspeth_group.
  # The corresponding handler:
  #
  #   - name: restart caddy
  #     ansible.builtin.systemd:
  #       name: caddy.service
  #       state: restarted
  #
  # belongs in the caddy_origin role.
```

Template `/etc/elspeth/elspeth-web.env` with secret-safe permissions:

```ini
ELSPETH_WEB__HOST=0.0.0.0
ELSPETH_WEB__AUTH_PROVIDER={{ elspeth_auth_provider }}
ELSPETH_WEB__REGISTRATION_MODE={{ elspeth_registration_mode }}
ELSPETH_WEB__SECRET_KEY={{ vault_elspeth_web_secret_key }}
ELSPETH_WEB__DATA_DIR={{ elspeth_data_dir }}
ELSPETH_WEB__SESSION_DB_URL={{ elspeth_session_db_url }}
ELSPETH_WEB__LANDSCAPE_URL={{ elspeth_landscape_url }}
ELSPETH_WEB__PAYLOAD_STORE_PATH={{ elspeth_payload_store_path }}
ELSPETH_WEB__CORS_ORIGINS={{ elspeth_cors_origins | to_json }}
# Required no-default Field(...) values — uvicorn refuses to start if any
# is missing. See the matching variables under "Common Variables".
ELSPETH_WEB__COMPOSER_MAX_COMPOSITION_TURNS={{ elspeth_composer_max_composition_turns }}
ELSPETH_WEB__COMPOSER_MAX_DISCOVERY_TURNS={{ elspeth_composer_max_discovery_turns }}
ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS={{ elspeth_composer_timeout_seconds }}
ELSPETH_WEB__COMPOSER_RATE_LIMIT_PER_MINUTE={{ elspeth_composer_rate_limit_per_minute }}
# Phase 6A — shareable-review HMAC signing key (Field(...), no default).
# Service refuses to start without this; rotating invalidates ALL
# outstanding shareable links.
ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY={{ vault_elspeth_web_shareable_link_signing_key }}
# Composer LLM identity
ELSPETH_WEB__COMPOSER_MODEL={{ elspeth_composer_model }}
# Composer transport timing — protects against browser/proxy idle abort
ELSPETH_WEB__COMPOSER_TRANSPORT_IDLE_CEILING_SECONDS={{ elspeth_composer_transport_idle_ceiling_seconds }}
ELSPETH_WEB__COMPOSER_TRANSPORT_HEADROOM_SECONDS={{ elspeth_composer_transport_headroom_seconds }}
ELSPETH_WEB__COMPOSER_RUNTIME_PREFLIGHT_TIMEOUT_SECONDS={{ elspeth_composer_runtime_preflight_timeout_seconds }}
ELSPETH_WEB__COMPOSER_MAX_TOOL_CALLS_PER_TURN={{ elspeth_composer_max_tool_calls_per_turn }}
# Composer error surface (false in production)
ELSPETH_WEB__COMPOSER_EXPOSE_PROVIDER_ERRORS={{ elspeth_composer_expose_provider_errors | string | lower }}
# Composer interpretation rate limits (F-30/F-31)
ELSPETH_WEB__COMPOSER_INTERPRETATION_RATE_LIMIT_PER_TERM={{ elspeth_composer_interpretation_rate_limit_per_term }}
ELSPETH_WEB__COMPOSER_INTERPRETATION_RATE_LIMIT_PER_SESSION_DAY={{ elspeth_composer_interpretation_rate_limit_per_session_day }}
# Shareable-review token lifetime
ELSPETH_WEB__SHAREABLE_LINK_LIFETIME_SECONDS={{ elspeth_shareable_link_lifetime_seconds }}
# Auth + JWKS
ELSPETH_WEB__AUTH_RATE_LIMIT_PER_MINUTE={{ elspeth_auth_rate_limit_per_minute }}
ELSPETH_WEB__JWKS_CACHE_TTL_SECONDS={{ elspeth_jwks_cache_ttl_seconds }}
ELSPETH_WEB__JWKS_FAILURE_RETRY_SECONDS={{ elspeth_jwks_failure_retry_seconds }}
ELSPETH_WEB__JWKS_MAX_STALE_SECONDS={{ elspeth_jwks_max_stale_seconds }}
# Upload + blob limits
ELSPETH_WEB__MAX_UPLOAD_BYTES={{ elspeth_max_upload_bytes }}
ELSPETH_WEB__MAX_BLOB_STORAGE_PER_SESSION_BYTES={{ elspeth_max_blob_storage_per_session_bytes }}
# Composer secret allowlist (JSON-encoded list)
ELSPETH_WEB__SERVER_SECRET_ALLOWLIST={{ elspeth_server_secret_allowlist | to_json }}
# Payload-store retention (informational; live enforcement via `elspeth purge`)
ELSPETH_WEB__PAYLOAD_STORE_RETENTION_DAYS={{ elspeth_payload_store_retention_days }}
# Orphan-run cleanup loop
ELSPETH_WEB__ORPHAN_RUN_MAX_AGE_SECONDS={{ elspeth_orphan_run_max_age_seconds }}
ELSPETH_WEB__ORPHAN_RUN_CHECK_INTERVAL_SECONDS={{ elspeth_orphan_run_check_interval_seconds }}
ELSPETH_FINGERPRINT_KEY={{ vault_fingerprint_key }}
OPENROUTER_API_KEY={{ vault_openrouter_api_key }}
AZURE_OPENAI_API_KEY={{ vault_azure_openai_api_key }}
AZURE_OPENAI_ENDPOINT={{ vault_azure_openai_endpoint }}
```

Install it with root-only permissions. systemd reads `EnvironmentFile=` as root
before dropping privileges to `User={{ elspeth_user }}`, so neither the
service user nor Caddy needs direct filesystem read access to this file.

```yaml
- name: Create environment directory
  ansible.builtin.file:
    path: /etc/elspeth
    state: directory
    owner: root
    group: root
    mode: "0700"

- name: Install web environment file
  ansible.builtin.template:
    src: elspeth-web.env.j2
    dest: /etc/elspeth/elspeth-web.env
    owner: root
    group: root
    mode: "0600"
  no_log: true
  notify: restart elspeth web
```

Install the systemd unit from a template:

```ini
[Unit]
Description=ELSPETH Web Server
After=network-online.target
Wants=network-online.target

[Service]
User={{ elspeth_user }}
Group={{ elspeth_socket_group }}
WorkingDirectory={{ elspeth_app_dir }}
RuntimeDirectory=elspeth
EnvironmentFile=/etc/elspeth/elspeth-web.env
Environment=PYTHONPATH={{ elspeth_app_dir }}/src
# Pin single-worker mode at the systemd layer so an env-file override or
# operator habit cannot accidentally trip the multi-worker SQLite guard
# in elspeth.web.app (see src/elspeth/web/app.py:517).
Environment=WEB_CONCURRENCY=1
ExecStart={{ elspeth_app_dir }}/.venv/bin/uvicorn elspeth.web.app:create_app \
    --factory \
    --uds {{ elspeth_web_bind_socket }} \
    --proxy-headers \
    --forwarded-allow-ips='*' \
    --no-server-header \
    --limit-concurrency 100
# --forwarded-allow-ips takes IP CIDRs (or '*'), not socket paths. For a
# UDS deployment, trust is established by filesystem permissions on the
# socket (mode 0660, {{ elspeth_socket_group }} group); the only peer that
# can connect is Caddy, so '*' is the correct value. Without this,
# X-Forwarded-For and X-Forwarded-Proto from Caddy are dropped and the
# audit trail records the wrong remote_addr.
#
# --limit-max-requests is deliberately omitted: it causes uvicorn to
# exit 0 after N requests, which Restart=on-failure does *not* catch.
# In a single-worker SQLite deployment this becomes a periodic outage
# with no automatic recovery. If you re-introduce it, also change
# Restart=on-failure to Restart=always.
Restart=on-failure
RestartSec=2
UMask=0007
NoNewPrivileges=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectSystem=strict
ProtectHome=read-only
ProtectControlGroups=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
LockPersonality=yes
RestrictRealtime=yes
RestrictNamespaces=yes
SystemCallArchitectures=native
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
ReadWritePaths=/run/elspeth {{ elspeth_data_dir }}
# If you override elspeth_session_db_url, elspeth_landscape_url, or
# elspeth_payload_store_path to point outside elspeth_data_dir, you MUST
# add those paths to ReadWritePaths or the service will silently fail to
# write. The preflight role asserts this:
#
#   - name: Assert writable paths live under elspeth_data_dir
#     ansible.builtin.assert:
#       that:
#         - elspeth_session_db_url is search('sqlite:////' ~ elspeth_data_dir | regex_escape)
#         - elspeth_landscape_url is search('sqlite:////' ~ elspeth_data_dir | regex_escape)
#         - elspeth_payload_store_path is match('^' ~ elspeth_data_dir | regex_escape)
#       fail_msg: >-
#         Override paths fall outside elspeth_data_dir; add them to
#         ReadWritePaths or set elspeth_data_dir to a common parent.

[Install]
WantedBy=multi-user.target
```

Handler:

```yaml
- name: restart elspeth web
  ansible.builtin.systemd:
    name: elspeth-web.service
    state: restarted
    daemon_reload: true
    enabled: true
```

## Non-Cloud VM Configuration

Inventory:

```yaml
all:
  children:
    elspeth_web:
      hosts:
        elspeth-prod-1:
          ansible_host: 203.0.113.10
          ansible_user: ubuntu
      vars:
        elspeth_domain: elspeth.example.com
        elspeth_cors_origins:
          - https://elspeth.example.com
```

Caddy origin template (`templates/elspeth.caddyfile.j2`). This file is
rendered by Ansible via `ansible.builtin.template`, **not** shipped as
a static `copy:` payload — the `{{ elspeth_domain }}`,
`{{ elspeth_hsts_max_age }}`, and `{% if %}` directives in the body
below are Jinja expressions that must be interpolated at render time.
The install task is shown immediately after the template body:

```caddyfile
{{ elspeth_domain }} {
    encode zstd gzip

    header {
        # HSTS header is constructed from elspeth_hsts_max_age,
        # elspeth_hsts_include_subdomains, and elspeth_hsts_preload —
        # see "Common Variables" for the ramp schedule and the
        # auditability rationale. Do NOT hand-edit this directive;
        # each one-way lever should be a discrete variable bump that
        # shows up in Git history as such.
        Strict-Transport-Security "max-age={{ elspeth_hsts_max_age }}{% if elspeth_hsts_include_subdomains %}; includeSubDomains{% endif %}{% if elspeth_hsts_preload %}; preload{% endif %}"
        X-Content-Type-Options "nosniff"
        Referrer-Policy "strict-origin-when-cross-origin"
        -Server
    }

    log {
        # Operational telemetry, NOT the audit trail. The legal record
        # lives in the Landscape DB at ELSPETH_WEB__LANDSCAPE_URL. Caddy
        # access logs may legitimately be sampled, rotated away, or
        # truncated for disk pressure; do not rely on them for decision
        # provenance. The audit-trail policy is not, however, a license
        # for sloppy access-log hygiene — operational logs still leak
        # PII if left at defaults. See "Access-Log Hygiene" below.
        output file /var/log/caddy/elspeth-access.log {
            roll_size 50MiB
            roll_keep 5
            roll_keep_for 168h   # 7 days on-host; ship off-host before this
            mode 0640            # operator-readable only; not world-readable
        }
        # The "filter" format wraps json and lets us redact fields that
        # commonly carry PII before they hit disk. The fields shown are
        # the minimum set; expand as needed for your workload. Exact
        # syntax varies across Caddy versions — verify against the
        # caddyserver/transform-encoder module documentation for the
        # version installed by your apt source, and test with
        # `caddy validate` before reloading.
        format filter {
            wrap json
            fields {
                request>headers>Authorization replace "REDACTED"
                request>headers>Cookie        replace "REDACTED"
                request>headers>Set-Cookie    replace "REDACTED"
                # Strip query strings from the logged URI. Query
                # parameters can carry session bootstrap tokens, search
                # terms, OIDC state, and similar. The path portion is
                # retained; only the "?..." tail is dropped.
                request>uri regexp "^([^?]*).*$" "$1"
            }
        }
    }

    reverse_proxy unix//run/elspeth/uvicorn.sock
}
```

Install task (lives in the `caddy_origin` role):

```yaml
- name: Render Caddy site file from Jinja template
  ansible.builtin.template:
    src: elspeth.caddyfile.j2
    dest: /etc/caddy/sites-available/elspeth
    owner: root
    group: caddy
    mode: "0640"
    validate: 'caddy validate --config %s --adapter caddyfile'
  notify: restart caddy
  # `validate:` runs `caddy validate` against the rendered candidate
  # before installing it. An unrendered `{{ }}` left over from a
  # missing variable, or a Caddy syntax error from a version drift
  # in the `log_filters` syntax, fails the task BEFORE the running
  # config is touched. Without this, a bad render replaces the live
  # file and the next caddy reload either takes the service down or
  # silently keeps the previous in-memory config until the next
  # restart — both modes hide the failure.
  #
  # Expected `caddy validate` warning: "Caddyfile input is not
  # formatted; run 'caddy fmt --overwrite' to fix inconsistencies".
  # Caddy's canonical formatting uses tab indentation; this runbook's
  # Markdown examples use 4-space indentation for cross-renderer
  # readability. The warning is cosmetic — the config is valid (the
  # task exits 0). If the noise bothers you, either reformat your
  # `elspeth.caddyfile.j2` with tabs, or add a `caddy fmt --overwrite`
  # post-render task before this one and the warning disappears.
  # Validated against Caddy 2.11.2 on 2026-05-18.

- name: Enable the elspeth Caddy site
  ansible.builtin.file:
    src: /etc/caddy/sites-available/elspeth
    dest: /etc/caddy/sites-enabled/elspeth
    state: link
    owner: root
    group: root
  notify: restart caddy
  # If your distribution's Caddy package does not use the
  # sites-available / sites-enabled split (some packagings include the
  # site directly via Caddyfile's `import` directive instead), skip
  # this task and add the corresponding `import` line to your main
  # Caddyfile under operator control.
```

### Caddy TLS Provisioning Modes

Caddy auto-provisions TLS via ACME on first reload when the site
block names a domain (e.g. `{{ elspeth_domain }} { ... }`). On a
non-cloud VM this is convenient but the failure modes look like
"playbook is broken" unless the operator picks a mode deliberately.
Four modes, picked per environment:

**1. HTTP-01 ACME (default, public host).** Caddy listens on port 80
for the ACME challenge and on port 443 for HTTPS. Requires:

- Public DNS A/AAAA pointing `{{ elspeth_domain }}` at the VM's
  public IP.
- Inbound port 80 reachable from the public Internet — the NSG / UFW
  / `iptables` rule for the VM MUST allow `0.0.0.0/0:80` for the
  duration of the challenge. ACME issuance retries indefinitely on
  failure, so closing port 80 after issuance is fine but you cannot
  *skip* opening it the first time.
- ACME-provider rate limits (Let's Encrypt: ~50 certs / 7d / domain;
  ~5 failed validations / 1h / domain). Pinning
  `elspeth_repo_version` to a SHA + running the role repeatedly does
  not re-issue certs — Caddy caches; rate limits matter mainly if
  the operator tears down and recreates the VM frequently.

No additional Caddy config needed beyond the site block above.

**2. DNS-01 ACME (recommended when port 80 is closed, e.g. behind a
bastion or private subnet).** Caddy proves domain ownership by
writing a TXT record via the DNS provider's API. Requires:

- A Caddy DNS plugin module compiled in (the apt-installed Caddy
  *does not* include DNS plugins by default; use the
  `xcaddy`-built binary, or a vendor RPM/DEB that bundles your
  provider). Common providers: Cloudflare, Route53, Azure DNS.
- API credentials in the vault and surfaced via environment to the
  Caddy systemd unit (e.g. `Environment=CLOUDFLARE_API_TOKEN=...`
  in a drop-in).
- Site-block directive naming the provider:

```caddyfile
{{ elspeth_domain }} {
    tls {
        dns cloudflare {env.CLOUDFLARE_API_TOKEN}
    }
    # ... reverse_proxy etc as above
}
```

Port 80 can stay closed end-to-end.

**3. `tls internal` (lab / canary / private domains).** Caddy's
built-in internal CA issues a cert that browsers do not trust by
default. Requires:

- Operator-managed trust: each client either trusts Caddy's local
  root (printed on first run at `~/.local/share/caddy/pki/...`),
  uses `--insecure` for one-off probes, or skips browsers entirely
  in favor of API-only access.
- Suitable for staging-style hosts where TLS-ness is wanted but
  public CA issuance is impossible (RFC1918 IPs, internal-only DNS).

```caddyfile
{{ elspeth_domain }} {
    tls internal
    # ... reverse_proxy etc as above
}
```

Append `elspeth_caddy_tls_internal: true` as an opt-in variable in
group_vars when this mode is desired; the template renders the `tls
internal` directive only when the flag is set, so production hosts
never fall into internal-CA mode by accident.

**4. Pre-provisioned certificate (compliance jurisdictions, EV
certs, or organizational PKI).** Operator drops cert + key files
into a controlled path and points Caddy at them:

```caddyfile
{{ elspeth_domain }} {
    tls /etc/caddy/certs/elspeth.pem /etc/caddy/certs/elspeth.key
    # ... reverse_proxy etc as above
}
```

The Ansible role must template both file paths into the site
template and ship the cert/key from the vault (or, preferably,
from the host's secret manager — Azure Key Vault file mount, etc.;
do NOT carry production private keys through Ansible Vault if
avoidable).

**Picking a mode.** The decision lives in `group_vars`; treat it
as load-bearing. The role's preflight should `assert` exactly one
mode is selected (the four variables are mutually exclusive) so a
half-configured site block does not silently fall back to HTTP-01
on a host with port 80 closed.

### Access-Log Hygiene

The Caddy access log is operational telemetry, not the audit trail —
but it is *still* a log file that records every request, including
auth-error paths, redirects, and metadata about authenticated users.
The default Caddy configuration is permissive about what lands in
that file; treat the runbook's `log { ... }` block as the *minimum*
sanitization, not the ceiling.

**Retention on-host.** The template sets `roll_keep 5` (five rolled
files) and `roll_keep_for 168h` (seven days). For compliance
jurisdictions with shorter retention windows (e.g., a 24-hour
operational-log policy), reduce both. For incident-response support
where seven days is too short, extend `roll_keep_for` *only* if
off-host shipping is in place — the on-host file is an emergency
local aid, not a long-term store.

**Off-host shipping.** Ship the rolled files to your central log
store (Loki/Promtail, Elastic, Splunk, Azure Log Analytics, etc.)
before they age out on-host. The runbook does not prescribe the
shipper because it depends on your environment; common options:
`promtail`, `vector`, `filebeat`, or Azure Monitor Agent for Azure
VMs. Whichever you choose:

- The shipper must use a service account with read-only access to
  `/var/log/caddy/`. Do *not* run it as root.
- The shipper must not write back into `/var/log/caddy/` (no
  position files, no checkpoints in that directory).
- The off-host store inherits the redaction discipline of the log
  file — if `Authorization` headers were not redacted on-host, they
  will appear in the central store and the operator-readable scope
  has just expanded enormously.

**File mode and ownership.** The template sets `mode 0640`, owned by
the Caddy service user. That makes the file readable only by Caddy
itself and members of the Caddy group. The Ansible role should NOT
add `elspeth` (or any other application service account) to the
Caddy group; the elspeth service does not need access to Caddy's
access log and adding it expands the blast radius of an elspeth
compromise.

**What's still leaked even with the filter above.**

- **Client IP**: `request.remote_addr` is logged at full resolution.
  For privacy-regulated jurisdictions, replace the IP-logging field
  with a truncated form (`/24` for IPv4, `/64` for IPv6) using a
  similar `regexp` field filter, or hash with a daily-rotated key.
- **User-Agent**: can contain version strings that uniquely identify
  the client. Acceptable for most operational use but flag it in
  threat-model reviews.
- **Referer**: not logged by default in the JSON formatter, but
  worth re-checking if you add it for debugging — Referer URLs can
  contain query strings from the prior page.
- **Path**: the URI *path* is retained (only the query string is
  stripped). Paths can themselves carry IDs (`/api/sessions/<uuid>`).
  These are typically required for operational diagnosis; if your
  workload routes PII into URL paths, redact more aggressively
  rather than relying on this template's defaults.

**Why this matters even with an audit trail.** The Landscape DB is
the legal record of *decisions*. The access log is the operational
record of *requests*. A breach of the access log exposes who
accessed what when — that is a sensitive-data exposure regardless
of whether the audit trail is intact. Audit-trail discipline is the
upper bound on what we can prove happened; access-log discipline is
the upper bound on what we accidentally help an attacker reconstruct.

Playbook:

```yaml
- name: Deploy ELSPETH to a non-cloud Ubuntu VM
  hosts: elspeth_web
  become: true
  roles:
    - elspeth_common
    - elspeth_web_source
    - caddy_origin
```

Run:

```bash
ansible-playbook -i inventories/non-cloud.yml playbooks/elspeth-vm.yml --ask-vault-pass
```

## Azure VM With Azure Front Door

Use this path when you want VM-level operational control but Azure-managed edge
TLS, WAF, global routing, and health probes.

Recommended Azure shape:

- Resource group for the app.
- Virtual network and subnet.
- Network security group attached to the VM subnet or NIC.
- Ubuntu 24.04 or 22.04 VM.
- Public IP or load-balanced origin endpoint reachable by Azure Front Door.
- Azure Front Door Standard or Premium profile.
- Origin group with `/api/health` as the health probe.
- Route for `/*` with caching disabled so WebSocket upgrades are forwarded.
- WAF policy at Front Door when exposing the service publicly.

Azure Front Door supports WebSocket on Standard and Premium tiers, but cache
must stay disabled on WebSocket routes. Front Door has idle and connection
duration limits, so keep client reconnect behavior enabled in the web UI.
WAF rules apply during WebSocket establishment, not after the connection is
upgraded.

Azure resource variables:

```yaml
azure_resource_group: rg-elspeth-prod
azure_location: australiaeast
azure_vnet_name: vnet-elspeth-prod
azure_subnet_name: snet-elspeth-web
azure_vm_name: vm-elspeth-prod-01
azure_vm_size: Standard_B2ms
azure_admin_username: azureuser
azure_admin_ssh_public_key: "ssh-ed25519 AAAA... operator@example"
azure_admin_source_cidr: 198.51.100.0/24
azure_ubuntu_release: "24.04"
azure_vm_image_by_ubuntu_release:
  "24.04":
    publisher: Canonical
    offer: ubuntu-24_04-lts
    sku: server
    version: latest
  "22.04":
    publisher: Canonical
    offer: 0001-com-ubuntu-server-jammy
    sku: 22_04-lts-gen2
    version: latest
azure_frontdoor_profile: afd-elspeth-prod
azure_frontdoor_endpoint: elspeth-prod
azure_frontdoor_origin_group: og-elspeth-web
azure_frontdoor_origin: origin-elspeth-vm
azure_frontdoor_sku: standard_azurefrontdoor
azure_vm_origin_host_name: origin-elspeth.example.com
azure_frontdoor_id: "00000000-0000-0000-0000-000000000000"
```

Verify marketplace image URNs against your Azure region before first use. Azure
image offers and SKUs can change independently of this repository.

Provision the VM and network resources first:

```yaml
- name: Create resource group
  azure.azcollection.azure_rm_resourcegroup:
    name: "{{ azure_resource_group }}"
    location: "{{ azure_location }}"

- name: Create virtual network
  azure.azcollection.azure_rm_virtualnetwork:
    resource_group: "{{ azure_resource_group }}"
    name: "{{ azure_vnet_name }}"
    address_prefixes:
      - 10.40.0.0/16

- name: Create web subnet
  azure.azcollection.azure_rm_subnet:
    resource_group: "{{ azure_resource_group }}"
    virtual_network_name: "{{ azure_vnet_name }}"
    name: "{{ azure_subnet_name }}"
    address_prefix_cidr: 10.40.1.0/24

- name: Create public IP for Front Door origin
  azure.azcollection.azure_rm_publicipaddress:
    resource_group: "{{ azure_resource_group }}"
    name: pip-elspeth-web
    allocation_method: Static
    sku: Standard

- name: Create network security group
  azure.azcollection.azure_rm_securitygroup:
    resource_group: "{{ azure_resource_group }}"
    name: nsg-elspeth-web
    rules:
      - name: AllowAdminSsh
        protocol: Tcp
        source_address_prefix: "{{ azure_admin_source_cidr }}"
        source_port_range: "*"
        destination_address_prefix: "*"
        destination_port_range: 22
        access: Allow
        priority: 90
        direction: Inbound
      - name: AllowAzureFrontDoorBackendHttps
        protocol: Tcp
        source_address_prefix: AzureFrontDoor.Backend
        source_port_range: "*"
        destination_address_prefix: "*"
        destination_port_range: 443
        access: Allow
        priority: 100
        direction: Inbound
      - name: AllowAzureInfrastructureHttps
        protocol: Tcp
        source_address_prefix: AzureLoadBalancer
        source_port_range: "*"
        destination_address_prefix: "*"
        destination_port_range: 443
        access: Allow
        priority: 110
        direction: Inbound
      - name: DenyInternetHttps
        protocol: Tcp
        source_address_prefix: Internet
        source_port_range: "*"
        destination_address_prefix: "*"
        destination_port_range: 443
        access: Deny
        priority: 200
        direction: Inbound

- name: Create network interface
  azure.azcollection.azure_rm_networkinterface:
    resource_group: "{{ azure_resource_group }}"
    name: nic-elspeth-web
    virtual_network: "{{ azure_vnet_name }}"
    subnet: "{{ azure_subnet_name }}"
    security_group: nsg-elspeth-web
    ip_configurations:
      - name: ipconfig1
        public_ip_address_name: pip-elspeth-web
        primary: true

- name: Create Ubuntu VM
  azure.azcollection.azure_rm_virtualmachine:
    resource_group: "{{ azure_resource_group }}"
    name: "{{ azure_vm_name }}"
    vm_size: "{{ azure_vm_size }}"
    admin_username: "{{ azure_admin_username }}"
    ssh_password_enabled: false
    ssh_public_keys:
      - path: "/home/{{ azure_admin_username }}/.ssh/authorized_keys"
        key_data: "{{ azure_admin_ssh_public_key }}"
    network_interfaces:
      - nic-elspeth-web
    managed_disk_type: Premium_LRS
    image: "{{ azure_vm_image_by_ubuntu_release[azure_ubuntu_release] }}"
    state: present
```

Then provision the Front Door profile with the Standard/Premium modules in
`azure.azcollection`:

```yaml
- name: Create Azure Front Door profile
  azure.azcollection.azure_rm_cdnprofile:
    resource_group: "{{ azure_resource_group }}"
    name: "{{ azure_frontdoor_profile }}"
    location: global
    sku: "{{ azure_frontdoor_sku }}"
    state: present

- name: Create Azure Front Door endpoint
  azure.azcollection.azure_rm_afdendpoint:
    resource_group: "{{ azure_resource_group }}"
    profile_name: "{{ azure_frontdoor_profile }}"
    name: "{{ azure_frontdoor_endpoint }}"
    state: present

- name: Create origin group
  azure.azcollection.azure_rm_afdorigingroup:
    resource_group: "{{ azure_resource_group }}"
    profile_name: "{{ azure_frontdoor_profile }}"
    name: "{{ azure_frontdoor_origin_group }}"
    health_probe_settings:
      probe_path: /api/health
      probe_request_type: GET
      probe_protocol: HTTPS
      probe_interval_in_seconds: 30
    state: present

- name: Create VM origin
  azure.azcollection.azure_rm_afdorigin:
    resource_group: "{{ azure_resource_group }}"
    profile_name: "{{ azure_frontdoor_profile }}"
    origin_group_name: "{{ azure_frontdoor_origin_group }}"
    name: "{{ azure_frontdoor_origin }}"
    host_name: "{{ azure_vm_origin_host_name }}"
    origin_host_header: "{{ azure_vm_origin_host_name }}"
    http_port: 80
    https_port: 443
    priority: 1
    weight: 1000
    enabled_state: Enabled
    state: present

- name: Probe origin TLS before enabling HttpsOnly forwarding
  ansible.builtin.command:
    argv:
      - curl
      - --fail
      - --silent
      - --show-error
      - --resolve
      - "{{ azure_vm_origin_host_name }}:443:{{ azure_vm_public_ip }}"
      - "https://{{ azure_vm_origin_host_name }}/api/health"
  register: elspeth_origin_tls_probe
  changed_when: false
  failed_when: elspeth_origin_tls_probe.rc != 0
  # If this fails, the route below will pin Front Door to HttpsOnly
  # against an origin that cannot present a valid certificate for the
  # origin_host_header, and every health probe will fail. Resolve TLS
  # provisioning (DNS-01 ACME, pre-provisioned cert, or origin hostname
  # with matching DNS) before enabling the route.

- name: Route all app traffic
  azure.azcollection.azure_rm_afdroute:
    resource_group: "{{ azure_resource_group }}"
    profile_name: "{{ azure_frontdoor_profile }}"
    endpoint_name: "{{ azure_frontdoor_endpoint }}"
    name: route-elspeth-web
    origin_group: "{{ azure_frontdoor_origin_group }}"
    disable_cache_configuration: true
    route:
      enabled_state: Enabled
      forwarding_protocol: HttpsOnly
      https_redirect: Enabled
      link_to_default_domain: Enabled
      patterns_to_match:
        - /*
      supported_protocols:
        - Https
    state: present
  when: elspeth_origin_tls_probe is succeeded
```

Secure the VM origin. Microsoft recommends both network filtering and Front
Door identifier validation for public IP origins:

The `nsg-elspeth-web` example above allows `AzureFrontDoor.Backend`, allows the
Azure load-balancer service path, and denies direct Internet HTTPS. Reconcile it
with your bastion or private-access model before opening SSH. Microsoft also
recommends allowing the platform infrastructure addresses used by Azure; check
the current Azure Front Door origin-security documentation before locking the
NSG in production.

Add an origin header check in Caddy when `azure_frontdoor_id` is known:

```caddyfile
{{ azure_vm_origin_host_name }} {
    @wrong_frontdoor_id {
        not header X-Azure-FDID {{ azure_frontdoor_id }}
    }
    respond @wrong_frontdoor_id 403

    encode zstd gzip
    reverse_proxy unix//run/elspeth/uvicorn.sock
}
```

Do not rely on the header check alone. Other clients can spoof HTTP headers if
they can reach the origin directly; the NSG rule closes that path. Conversely,
do not rely on the NSG alone because other Azure customers use the same Front
Door backend service tag.

Run the Azure resource playbook first, then the VM configuration playbook:

```bash
ansible-playbook -i inventories/azure-vm.yml playbooks/azure-vm-front-door.yml --ask-vault-pass
ansible-playbook -i inventories/azure-vm.yml playbooks/elspeth-vm.yml --ask-vault-pass
```

## Azure Containers Configuration

Use this path when you want image-based deployment rather than a mutable source
checkout. The app still needs the same web environment variables, but state
should move out of the container.

Container guidance:

- Build an immutable image tag, for example `sha-<commit>`.
- Install ELSPETH with the extras required by the workload, usually
  `webui,azure,llm`.
- Build `src/elspeth/web/frontend/dist/` inside the image.
- Run uvicorn on TCP inside the container, not a Unix socket.
- Use an external database for session and landscape state in production.
- Use a mounted filesystem volume for `ELSPETH_WEB__PAYLOAD_STORE_PATH`.
  The current runtime payload store is filesystem-backed; Azure Blob is
  available to source/sink plugins, not as a drop-in web payload-store backend.
- Use managed identity for image pulls and Azure resource access where possible.
- Use Key Vault references for production Container Apps secrets. Direct
  `value:` entries are acceptable only for throwaway development environments.

Container Apps variables:

```yaml
azure_containerapps_environment: cae-elspeth-prod
azure_container_app_name: ca-elspeth-web-prod
azure_acr_login_server: acrelspethprod.azurecr.io
azure_container_pull_identity_id: "/subscriptions/.../userAssignedIdentities/id-elspeth-pull"
azure_container_runtime_identity_id: "/subscriptions/.../userAssignedIdentities/id-elspeth-runtime"
azure_keyvault_elspeth_web_secret_key_url: "https://kv-elspeth.vault.azure.net/secrets/elspeth-web-secret-key/<version>"
azure_keyvault_fingerprint_key_url: "https://kv-elspeth.vault.azure.net/secrets/elspeth-fingerprint-key/<version>"
# Phase 6A — shareable-review HMAC signing key (Field(...), no default).
# The web service refuses to start without it; the field is `strict=True`
# so the env var MUST decode cleanly as base64 (the runtime decodes the
# str to bytes at config-load time). Rotating invalidates ALL outstanding
# shareable links.
azure_keyvault_elspeth_web_shareable_link_signing_key_url: "https://kv-elspeth.vault.azure.net/secrets/elspeth-web-shareable-link-signing-key/<version>"
elspeth_image_tag: sha-542f34874

# Traffic-shifting flow. Set elspeth_previous_image_tag to the SHA
# currently serving 100% traffic BEFORE running the deploy — the role
# uses it both for the initial-state traffic split and as the rollback
# target when the new revision's health probe fails. The probe variables
# are conservative defaults: 10 retries x 15s = 150s probe budget.
# elspeth_revision_probe_business_endpoints defaults to empty; populate
# it per-environment with the application's _smoke endpoints (e.g.,
# ['/api/sessions/_smoke', '/api/auth/_status']). The traffic-shifting
# flow honors any non-empty list automatically — no role change needed.
# These endpoints are APPLICATION CODE, not deploy code: if they don't
# exist in the running revision they return 404 and the revision is
# deactivated as if it were broken. Add them in the application first.
elspeth_previous_image_tag: ""
elspeth_revision_probe_path: /api/health
elspeth_revision_probe_retries: 10
elspeth_revision_probe_delay: 15
elspeth_revision_probe_business_endpoints: []
```

Container Apps **revision name length budget.** Azure documents
`revisionSuffix` as accepting up to 64 characters, but the *full*
revision name that gets constructed — `<azure_container_app_name>--<suffix>`
— is itself subject to a Container Apps name-length limit. Plan for:

```text
max(len(elspeth_image_tag)) = 64 - len(azure_container_app_name) - 2
```

For the worked example with `azure_container_app_name: ca-elspeth-web-prod`
(19 characters), the suffix budget is 43 characters. The
`sha-<commit>` convention with a 7-character short SHA (`sha-542f348`,
11 characters) fits comfortably; a 40-character long SHA
(`sha-542f34874a2b...`) is 44 characters and would just barely fit
even with the headroom budget. If your `azure_container_app_name` is
longer, shorten the SHA portion of the tag or rename the app.

**Revision suffix character constraints.** Lowercase alphanumeric and
hyphens only. Tags containing `.`, `_`, or uppercase characters will
fail revision creation with a non-obvious error mid-deploy. Choose the
tag scheme accordingly — the `sha-<commit>` convention satisfies the
character set as well as the length budget.

If one user-assigned identity handles both ACR pulls and Key Vault reads, set
`azure_container_pull_identity_id` and `azure_container_runtime_identity_id` to
the same resource ID and grant it both roles.

Container command:

```yaml
command:
  - /app/.venv/bin/uvicorn
args:
  - elspeth.web.app:create_app
  - --factory
  - --host
  - 0.0.0.0
  - --port
  - "8451"
  - --proxy-headers
  - --forwarded-allow-ips=*
  - --no-server-header
```

Azure Container Apps can be provisioned with `azure_rm_resource` when the
specific Container Apps modules in your installed `azure.azcollection` version
do not cover the properties you need.

The container app's **initial creation** and **subsequent revision
deploys** are *different operations*: the initial create is a one-shot
bootstrap that sets `activeRevisionsMode: Multiple` and seeds the
first revision; subsequent deploys go through "Revision
Traffic-Shifting For Production Deploys" below and must NOT re-run
the initial-create task, because that task's body is a full-replace
PUT and would clobber the running multi-revision configuration. The
discovery task below sets `elspeth_container_app_exists`, and the two
operations gate on it:

```yaml
- name: Discover whether the container app already exists
  azure.azcollection.azure_rm_resource_info:
    resource_group: "{{ azure_resource_group }}"
    provider: App
    resource_type: containerApps
    resource_name: "{{ azure_container_app_name }}"
    api_version: "2024-03-01"
  register: elspeth_container_app_lookup
  failed_when: false                # decision made by the next task

- name: Distinguish 404 from auth / permission / network failures
  ansible.builtin.assert:
    that:
      # VERIFY BEFORE MERGE: azure_rm_resource_info's failure shape
      # varies across azure.azcollection versions and Azure regions.
      # Across released versions we have observed:
      #   - newer versions: 404 returns `failed=false` with empty
      #     `response`, so the `not failed` branch matches and the
      #     fact below evaluates to "not exists" correctly.
      #   - older versions: 404 returns `failed=true` with a message
      #     containing "ResourceNotFound" or "was not found" (we have
      #     also seen "404 Not Found" and "Resource ... not found"
      #     across regions; this list is NOT exhaustive).
      #
      # Run a one-off probe in your environment with `ansible-playbook
      # --check` against a non-existent app and confirm the actual
      # failure shape BEFORE relying on this assert. If your installed
      # version emits a phrase not in the match list below, a
      # legitimate 404 will fail the assert and block deploys. Two
      # safer alternatives if the string match is too fragile:
      #   (a) call the ARM REST endpoint directly with `uri:` and
      #       check `.status` explicitly (200 vs 404 vs other);
      #   (b) treat the assert as advisory and gate the bootstrap
      #       task on `elspeth_container_app_exists OR (lookup failed
      #       with a phrase in the operator-curated allowlist)`.
      #
      # The match list below is the starting point, not the answer.
      - >-
        not (elspeth_container_app_lookup.failed | default(false))
        or ('ResourceNotFound' in (elspeth_container_app_lookup.msg | default('')))
        or ('was not found' in (elspeth_container_app_lookup.msg | default('')))
        or ('404' in (elspeth_container_app_lookup.msg | default('')))
    fail_msg: >-
      Container app lookup failed for a non-404 reason:
      {{ elspeth_container_app_lookup.msg | default('unknown error') }}.
      This is typically an authentication or RBAC error — verify the
      Ansible runner identity has at least Reader on
      {{ azure_resource_group }} and Container Apps Reader (or higher)
      on {{ azure_container_app_name }}, that the subscription is
      reachable, and that the credentials have not expired. Refusing to
      proceed: the bootstrap task below would otherwise misread the
      lookup as "app does not exist" and attempt to create over the
      running app.

- name: Set container-app existence fact
  ansible.builtin.set_fact:
    elspeth_container_app_exists: >-
      {{ (not (elspeth_container_app_lookup.failed | default(false)))
         and ((elspeth_container_app_lookup.response | default([]) | length) > 0) }}
    # Two conditions are required for "exists": the lookup must have
    # succeeded (not failed-and-suppressed-as-404), AND the response
    # must contain at least one resource. A failed-but-not-404 lookup
    # has already tripped the assert above and we never reach here.

- name: Load shared container template (single source of truth)
  ansible.builtin.set_fact:
    elspeth_container_template: "{{ lookup('template', 'container-app-template.yml.j2') | from_yaml }}"
  # See "Container Template — Single Source Of Truth" below for the
  # template body. Both the bootstrap PUT and the revision-shift PATCH
  # consume `elspeth_container_template.template` for the `template:`
  # subtree of the request body. Without this single source, the PATCH
  # body's abbreviated container spec would silently drop env vars,
  # secret refs, and probes from every redeploy.
  #
  # VERIFY BEFORE MERGE: the downstream `body:` parameters reference
  # this fact as `template: "{{ elspeth_container_template.template }}"`
  # — a quoted single-expression Jinja that should resolve to a dict.
  # Ansible's type-preservation behavior for this pattern varies
  # across versions: most modern ansible-core releases pass the dict
  # through correctly, but some `jinja2_native` configurations
  # stringify it (which Azure ARM then rejects as a malformed body).
  # Test against your installed ansible-core with
  # `ansible -m debug -a 'var=elspeth_container_template.template'`
  # and confirm the output is a YAML mapping, not a quoted string.
  # If type-preservation is broken in your environment, build the
  # entire request body via a second `set_fact` and pass
  # `body: "{{ elspeth_container_app_body }}"` to the azure_rm_resource
  # task instead — that pattern is unambiguous across all versions.

- name: Create Container Apps environment
  azure.azcollection.azure_rm_resource:
    resource_group: "{{ azure_resource_group }}"
    provider: App
    resource_type: managedEnvironments
    resource_name: "{{ azure_containerapps_environment }}"
    api_version: "2024-03-01"
    body:
      location: "{{ azure_location }}"
      properties: {}

# GATE: the next task runs ONLY when the container app does not yet
# exist (see the `when:` clause at the end of the task). On every
# subsequent deploy, the traffic-shifting playbook below is the sole
# mutator. Removing the `when:` would PUT a single-revision config
# over a running multi-revision app, deleting the previous revision
# before traffic-shifting can stage the new one — exactly the
# failure mode the flow is designed to avoid.
- name: Create ELSPETH container app (one-shot bootstrap; runs only on first-ever deploy)
  azure.azcollection.azure_rm_resource:
    resource_group: "{{ azure_resource_group }}"
    provider: App
    resource_type: containerApps
    resource_name: "{{ azure_container_app_name }}"
    api_version: "2024-03-01"
    body:
      location: "{{ azure_location }}"
      identity:
        type: UserAssigned
        userAssignedIdentities:
          "{{ azure_container_pull_identity_id }}": {}
          "{{ azure_container_runtime_identity_id }}": {}
      properties:
        managedEnvironmentId: "{{ azure_containerapps_environment_id }}"
        configuration:
          # Multiple revision mode is REQUIRED for traffic-splitting. Set
          # it at first-create — switching from Single to Multiple later
          # forces a non-trivial migration of traffic state. See
          # "Revision Traffic-Shifting For Production Deploys" below.
          activeRevisionsMode: Multiple
          ingress:
            external: true
            targetPort: 8451
            transport: auto
            allowInsecure: false
            traffic:
              # On first deploy elspeth_previous_image_tag is empty and
              # the new revision takes 100% traffic — there is no prior
              # revision to split against. On subsequent deploys the
              # traffic-shifting playbook below overrides this block.
              - revisionName: "{{ azure_container_app_name }}--{{ elspeth_image_tag }}"
                weight: 100
          registries:
            - server: "{{ azure_acr_login_server }}"
              identity: "{{ azure_container_pull_identity_id }}"
          secrets:
            - name: elspeth-web-secret-key
              keyVaultUrl: "{{ azure_keyvault_elspeth_web_secret_key_url }}"
              identity: "{{ azure_container_runtime_identity_id }}"
            - name: elspeth-fingerprint-key
              keyVaultUrl: "{{ azure_keyvault_fingerprint_key_url }}"
              identity: "{{ azure_container_runtime_identity_id }}"
            # Phase 6A — shareable-review HMAC signing key. Field(...)
            # on WebSettings means the service refuses to start without
            # this secret. The Key Vault entry MUST contain the
            # operator-generated 32-byte key as base64 (output of
            # `openssl rand -base64 32`).
            - name: elspeth-web-shareable-link-signing-key
              keyVaultUrl: "{{ azure_keyvault_elspeth_web_shareable_link_signing_key_url }}"
              identity: "{{ azure_container_runtime_identity_id }}"
        # `template:` is consumed from the shared
        # `elspeth_container_template` fact loaded above. Do NOT inline
        # a container spec here — the traffic-shift PATCH task below
        # consumes the SAME fact, and any drift between the two
        # destinations silently drops env vars, secret refs, or probes
        # on every redeploy. See "Container Template — Single Source
        # Of Truth" below for the template body.
        template: "{{ elspeth_container_template.template }}"
  when: not elspeth_container_app_exists
  # `no_log: true` is intentionally NOT set here. Secrets are passed by
  # Key Vault reference (keyVaultUrl + identity), not by value, so the
  # task body contains no secret material. Hiding the task output makes
  # first-run debugging materially harder. The env-file template task in
  # the VM role does use `no_log: true` because that file does contain
  # plaintext secrets.
```

When publishing Container Apps through Azure Front Door, create the Front Door
origin from the Container App ingress FQDN instead of the VM hostname. Keep the
route caching disabled and reuse `/api/health` as the probe path.

### Container Template — Single Source Of Truth

The bootstrap PUT and the traffic-shift PATCH both consume the SAME
container spec, loaded once into the `elspeth_container_template` fact
by the "Load shared container template" task above. The fact reads from
`templates/container-app-template.yml.j2`, which lives in the
`azure_containers` role:

```yaml
# templates/container-app-template.yml.j2 — single source of truth
# for the Container Apps `template:` subtree.
template:
  # revisionSuffix makes revision names deterministic
  # (<appName>--<suffix>). Without it Container Apps generates random
  # suffixes and the traffic-shifting tasks cannot construct the
  # per-revision FQDN deterministically.
  revisionSuffix: "{{ elspeth_image_tag }}"
  containers:
    - name: elspeth-web
      image: "{{ azure_acr_login_server }}/elspeth:{{ elspeth_image_tag }}"
      env:
        - name: ELSPETH_WEB__HOST
          value: "0.0.0.0"
        - name: ELSPETH_WEB__SECRET_KEY
          secretRef: elspeth-web-secret-key
        - name: ELSPETH_FINGERPRINT_KEY
          secretRef: elspeth-fingerprint-key
        # Phase 6A — shareable-review HMAC signing key. Field(...) on
        # WebSettings means uvicorn refuses to start without this; the
        # Key Vault entry MUST be the operator-generated 32-byte key
        # (`openssl rand -base64 32` output).
        - name: ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY
          secretRef: elspeth-web-shareable-link-signing-key
        - name: ELSPETH_WEB__AUTH_PROVIDER
          value: "{{ elspeth_auth_provider }}"
        - name: ELSPETH_WEB__REGISTRATION_MODE
          value: "{{ elspeth_registration_mode }}"
        - name: ELSPETH_WEB__SESSION_DB_URL
          value: "{{ elspeth_session_db_url }}"
        - name: ELSPETH_WEB__LANDSCAPE_URL
          value: "{{ elspeth_landscape_url }}"
        - name: ELSPETH_WEB__PAYLOAD_STORE_PATH
          value: "{{ elspeth_payload_store_path }}"
        # Required Field(...) composer values — uvicorn refuses to start
        # if any is missing. Plain values (not secrets) but still
        # required because there is no Python default.
        - name: ELSPETH_WEB__COMPOSER_MAX_COMPOSITION_TURNS
          value: "{{ elspeth_composer_max_composition_turns }}"
        - name: ELSPETH_WEB__COMPOSER_MAX_DISCOVERY_TURNS
          value: "{{ elspeth_composer_max_discovery_turns }}"
        - name: ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS
          value: "{{ elspeth_composer_timeout_seconds }}"
        - name: ELSPETH_WEB__COMPOSER_RATE_LIMIT_PER_MINUTE
          value: "{{ elspeth_composer_rate_limit_per_minute }}"
        # Composer-aware tunables (defaults exist in Python but are
        # surfaced here so the Front Door / Container Apps deploy can
        # tune them without source edits).
        - name: ELSPETH_WEB__COMPOSER_MODEL
          value: "{{ elspeth_composer_model }}"
        - name: ELSPETH_WEB__COMPOSER_TRANSPORT_IDLE_CEILING_SECONDS
          value: "{{ elspeth_composer_transport_idle_ceiling_seconds }}"
        - name: ELSPETH_WEB__COMPOSER_TRANSPORT_HEADROOM_SECONDS
          value: "{{ elspeth_composer_transport_headroom_seconds }}"
        - name: ELSPETH_WEB__COMPOSER_RUNTIME_PREFLIGHT_TIMEOUT_SECONDS
          value: "{{ elspeth_composer_runtime_preflight_timeout_seconds }}"
        - name: ELSPETH_WEB__COMPOSER_MAX_TOOL_CALLS_PER_TURN
          value: "{{ elspeth_composer_max_tool_calls_per_turn }}"
        - name: ELSPETH_WEB__COMPOSER_EXPOSE_PROVIDER_ERRORS
          value: "{{ elspeth_composer_expose_provider_errors | string | lower }}"
        - name: ELSPETH_WEB__COMPOSER_INTERPRETATION_RATE_LIMIT_PER_TERM
          value: "{{ elspeth_composer_interpretation_rate_limit_per_term }}"
        - name: ELSPETH_WEB__COMPOSER_INTERPRETATION_RATE_LIMIT_PER_SESSION_DAY
          value: "{{ elspeth_composer_interpretation_rate_limit_per_session_day }}"
        - name: ELSPETH_WEB__SHAREABLE_LINK_LIFETIME_SECONDS
          value: "{{ elspeth_shareable_link_lifetime_seconds }}"
        - name: ELSPETH_WEB__AUTH_RATE_LIMIT_PER_MINUTE
          value: "{{ elspeth_auth_rate_limit_per_minute }}"
        - name: ELSPETH_WEB__JWKS_CACHE_TTL_SECONDS
          value: "{{ elspeth_jwks_cache_ttl_seconds }}"
        - name: ELSPETH_WEB__JWKS_FAILURE_RETRY_SECONDS
          value: "{{ elspeth_jwks_failure_retry_seconds }}"
        - name: ELSPETH_WEB__JWKS_MAX_STALE_SECONDS
          value: "{{ elspeth_jwks_max_stale_seconds }}"
        - name: ELSPETH_WEB__MAX_UPLOAD_BYTES
          value: "{{ elspeth_max_upload_bytes }}"
        - name: ELSPETH_WEB__MAX_BLOB_STORAGE_PER_SESSION_BYTES
          value: "{{ elspeth_max_blob_storage_per_session_bytes }}"
        - name: ELSPETH_WEB__SERVER_SECRET_ALLOWLIST
          value: "{{ elspeth_server_secret_allowlist | to_json }}"
        - name: ELSPETH_WEB__PAYLOAD_STORE_RETENTION_DAYS
          value: "{{ elspeth_payload_store_retention_days }}"
        - name: ELSPETH_WEB__ORPHAN_RUN_MAX_AGE_SECONDS
          value: "{{ elspeth_orphan_run_max_age_seconds }}"
        - name: ELSPETH_WEB__ORPHAN_RUN_CHECK_INTERVAL_SECONDS
          value: "{{ elspeth_orphan_run_check_interval_seconds }}"
      probes:
        - type: Liveness
          httpGet:
            path: /api/health
            port: 8451
          initialDelaySeconds: 30
          periodSeconds: 30
```

**Why this is required, not a stylistic preference.** Azure ARM
PATCH-merge semantics on the `Microsoft.App/containerApps` resource
treat `containers[]` as a *whole-array replacement*: any field absent
from the PATCH body is dropped from the new revision. An inlined
abbreviated container spec in the PATCH task — with just `name` and
`image` and a comment that env vars are "identical to the initial
creation task" — silently ships a broken revision whose only
environment is what's missing. The probe against the per-revision
FQDN then fails (because `ELSPETH_WEB__SECRET_KEY` is unset, etc.),
the auto-rollback deactivates the revision, the operator sees
"revision failed health probe" and chases the wrong root cause.

**Adding a field.** Add it to `container-app-template.yml.j2` only.
The bootstrap PUT and the traffic-shift PATCH both consume it on the
next deploy. Do not edit the inline `template:` references in the
task bodies — they only contain `"{{ elspeth_container_template.template }}"`
and nothing else.

**If you must inline a container spec for some reason** (e.g., a
debugging session that needs to vary one field), use a *full* container
spec, not an abbreviated one. The runbook's risk register entry for
"PATCH with abbreviated body" is load-bearing.

### Revision Traffic-Shifting For Production Deploys

Production Container Apps deploys must not put 100% of traffic on a new
revision before its health has been confirmed. The flow below creates
the new revision at 0% traffic, probes its **per-revision FQDN**
(probing the app's stable FQDN can hit the previous revision and
produce a false-positive green), then shifts traffic to 100% on
success or deactivates the failed revision and leaves the previous
revision serving.

The flow assumes the initial container creation has already run with
`activeRevisionsMode: Multiple` and a `revisionSuffix` (both set in
the task above). **Every task in this section is gated on
`when: elspeth_container_app_exists`** — the fact set by the
discovery task in the previous section. On first-ever deploy the
container app did not exist before the role started, so this whole
flow is skipped and the one-shot initial-create handles the bootstrap;
on every subsequent deploy the discovery task finds the app and the
traffic-shifting flow runs.

Pre-conditions checked in the role:

```yaml
- name: Refuse traffic-shift deploy without a previous-image reference
  ansible.builtin.assert:
    that:
      - elspeth_previous_image_tag is defined
      - elspeth_previous_image_tag | length > 0
      - elspeth_previous_image_tag != elspeth_image_tag
    fail_msg: >-
      Traffic-shifting deploy requires elspeth_previous_image_tag (the
      SHA currently serving 100% traffic) and it must differ from
      elspeth_image_tag. For first-ever deploy use the initial-create
      task above; for redeploy of the same tag use a no-op playbook.
  when: elspeth_container_app_exists
  # The remaining tasks in this section all carry the same
  # `when: elspeth_container_app_exists` guard. The most concise way
  # to express this in a real role is a `block:` wrapper:
  #
  #   - block:
  #       - name: Refuse traffic-shift deploy ...
  #       - name: Deploy new revision at 0% traffic
  #       - ...
  #     when: elspeth_container_app_exists
  #
  # The tasks below are shown unwrapped for readability; co-locate
  # them inside a single block in your role.

- name: Set new and previous revision name facts
  ansible.builtin.set_fact:
    elspeth_new_revision_name: "{{ azure_container_app_name }}--{{ elspeth_image_tag }}"
    elspeth_previous_revision_name: "{{ azure_container_app_name }}--{{ elspeth_previous_image_tag }}"
  when: elspeth_container_app_exists
```

Deploy the new revision at 0% traffic. The PATCH preserves the previous
revision serving 100% and adds the new revision deactivated-for-traffic:

```yaml
- name: Deploy new revision at 0% traffic
  azure.azcollection.azure_rm_resource:
    resource_group: "{{ azure_resource_group }}"
    provider: App
    resource_type: containerApps
    resource_name: "{{ azure_container_app_name }}"
    api_version: "2024-03-01"
    method: PATCH
    body:
      properties:
        configuration:
          activeRevisionsMode: Multiple
          ingress:
            external: true
            targetPort: 8451
            transport: auto
            allowInsecure: false
            traffic:
              - revisionName: "{{ elspeth_previous_revision_name }}"
                weight: 100
              - revisionName: "{{ elspeth_new_revision_name }}"
                weight: 0
        # `template:` is consumed from the shared
        # `elspeth_container_template` fact loaded near the top of the
        # Azure Containers Configuration section. This is REQUIRED, not
        # a stylistic preference: Azure ARM PATCH-merge semantics for
        # the containers[] array are "replace the entire array" — any
        # field that the PATCH body omits (env vars, secret refs,
        # probes) is dropped from the new revision. Inlining a partial
        # container spec here is the most common way to ship a broken
        # production redeploy. Do not do it.
        template: "{{ elspeth_container_template.template }}"
```

Fetch the per-revision FQDN. Azure assigns a deterministic FQDN of the
form `<revisionName>.<envDomain>` when ingress is external:

```yaml
- name: Fetch new revision metadata
  azure.azcollection.azure_rm_resource_info:
    resource_group: "{{ azure_resource_group }}"
    provider: App
    resource_type: containerApps
    resource_name: "{{ azure_container_app_name }}"
    subresource:
      - type: revisions
        # Azure's revisions API expects the FULL revision name
        # (<appName>--<suffix>), not the bare suffix. Passing just
        # elspeth_image_tag returns 404 — the assert two tasks below
        # would then fire on undefined FQDN. Always use the fact set
        # at the top of this section.
        name: "{{ elspeth_new_revision_name }}"
    api_version: "2024-03-01"
  register: elspeth_new_revision_info

- name: Extract per-revision FQDN
  ansible.builtin.set_fact:
    elspeth_new_revision_fqdn: "{{ elspeth_new_revision_info.response[0].properties.fqdn }}"

- name: Refuse to proceed without a per-revision FQDN
  ansible.builtin.assert:
    that:
      - elspeth_new_revision_fqdn is defined
      - elspeth_new_revision_fqdn | length > 0
    fail_msg: >-
      Azure did not return a per-revision FQDN. Check that ingress.external=true
      and that activeRevisionsMode=Multiple in the container app. Without a
      per-revision FQDN the probe below would hit the previous revision and
      give a false-positive green.
```

Probe `/api/health` against the per-revision FQDN, with retries. Any
endpoints listed in `elspeth_revision_probe_business_endpoints` are
probed after the health check passes; an empty list (the default)
runs only the health probe. Populate the list per-environment in
`group_vars` once the application exposes appropriate `_smoke`
endpoints — see the variable comment for the constraint:

```yaml
- name: Probe new revision /api/health
  ansible.builtin.uri:
    url: "https://{{ elspeth_new_revision_fqdn }}{{ elspeth_revision_probe_path }}"
    method: GET
    status_code: 200
    return_content: false
  register: elspeth_revision_probe
  retries: "{{ elspeth_revision_probe_retries }}"
  delay: "{{ elspeth_revision_probe_delay }}"
  until: elspeth_revision_probe.status == 200
  failed_when: false                # decision handled below

- name: Probe new revision business endpoints
  ansible.builtin.uri:
    url: "https://{{ elspeth_new_revision_fqdn }}{{ item }}"
    method: GET
    status_code: 200
  loop: "{{ elspeth_revision_probe_business_endpoints }}"
  register: elspeth_business_probe
  failed_when: false                # decision handled below
  when: elspeth_revision_probe.status == 200

- name: Decide whether the new revision is healthy
  ansible.builtin.set_fact:
    elspeth_revision_healthy: >-
      {{ (elspeth_revision_probe.status | default(0)) == 200 and
         (elspeth_business_probe.results | default([])
           | rejectattr('status', 'equalto', 200)
           | list | length == 0) }}
```

Shift traffic on success, deactivate on failure. Exactly one of these two
tasks runs each deploy:

```yaml
- name: Shift 100% traffic to new revision
  azure.azcollection.azure_rm_resource:
    resource_group: "{{ azure_resource_group }}"
    provider: App
    resource_type: containerApps
    resource_name: "{{ azure_container_app_name }}"
    api_version: "2024-03-01"
    method: PATCH
    body:
      properties:
        configuration:
          ingress:
            traffic:
              - revisionName: "{{ elspeth_new_revision_name }}"
                weight: 100
              - revisionName: "{{ elspeth_previous_revision_name }}"
                weight: 0
  when: elspeth_revision_healthy

- name: Deactivate failed revision and leave previous revision serving
  azure.azcollection.azure_rm_resource:
    resource_group: "{{ azure_resource_group }}"
    provider: App
    resource_type: containerApps
    resource_name: "{{ azure_container_app_name }}"
    subresource:
      - type: revisions
        # Same correctness property as "Fetch new revision metadata"
        # above: the revisions API expects the full revision name,
        # not the bare image-tag suffix.
        name: "{{ elspeth_new_revision_name }}"
      - type: deactivate
    method: POST
    api_version: "2024-03-01"
  when: not elspeth_revision_healthy

- name: Fail the play if rollback was triggered
  ansible.builtin.fail:
    msg: >-
      New revision {{ elspeth_new_revision_name }} failed health probes
      ({{ elspeth_revision_probe.status | default('no-response') }} on
      {{ elspeth_revision_probe_path }}). Revision has been deactivated;
      {{ elspeth_previous_revision_name }} continues serving 100% traffic.
      Investigate the failed revision before retrying.
  when: not elspeth_revision_healthy
```

**Gradual ramp variation.** If a one-shot 0% → 100% shift is too
aggressive (high-traffic deploy, or you want canary metrics), replace
the "Shift 100% traffic" task with a loop that ramps weights in stages
and re-probes between stages:

```yaml
- name: Ramp traffic to new revision
  azure.azcollection.azure_rm_resource:
    resource_group: "{{ azure_resource_group }}"
    provider: App
    resource_type: containerApps
    resource_name: "{{ azure_container_app_name }}"
    api_version: "2024-03-01"
    method: PATCH
    body:
      properties:
        configuration:
          ingress:
            traffic:
              - revisionName: "{{ elspeth_new_revision_name }}"
                weight: "{{ item }}"
              - revisionName: "{{ elspeth_previous_revision_name }}"
                weight: "{{ 100 - item }}"
  loop: [10, 50, 100]
  loop_control:
    pause: 60     # seconds between stages; raise for slow-burn canaries
  when: elspeth_revision_healthy
```

The loop is non-rollbacking by design: once it commits to 10% it
assumes the per-revision probe already proved health. Add a second
per-stage probe against the *app's stable FQDN* if you want each ramp
stage gated on real-traffic health.

**Why the per-revision FQDN matters.** Container Apps ingress routes
traffic across revisions according to the `traffic[]` weights. A probe
against the app's stable FQDN at any weight other than 0% can hit
*either* revision, so it cannot distinguish "new revision is healthy"
from "new revision is broken but the old one is still serving." The
per-revision FQDN bypasses the weighted router and addresses the
specific revision unconditionally. This is the single most important
correctness property of the flow.

### Front Door And Traffic-Shifting

When the Container App sits behind Azure Front Door, the per-revision
probe above runs end-to-end against Azure-internal DNS — Front Door
origins should still target the *app's stable FQDN*, not a per-revision
FQDN, because revisions come and go. Front Door observes the traffic
weights set on the Container App ingress.

## Verification

After each playbook run, verify the host or container without printing secrets.

VM checks:

```yaml
- name: Check systemd service state
  ansible.builtin.command: systemctl is-active elspeth-web.service
  register: elspeth_service_state
  changed_when: false
  failed_when: elspeth_service_state.stdout != "active"

- name: Check local health over Unix socket
  ansible.builtin.command:
    cmd: curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
  changed_when: false

- name: Check built frontend entrypoint exists
  ansible.builtin.stat:
    path: "{{ elspeth_app_dir }}/src/elspeth/web/frontend/dist/index.html"
  register: elspeth_frontend_index

- name: Fail if frontend bundle is missing
  ansible.builtin.assert:
    that:
      - elspeth_frontend_index.stat.exists

- name: Confirm uvicorn is running single-worker
  ansible.builtin.shell: |
    set -o pipefail
    pgrep -afu {{ elspeth_user }} uvicorn | grep -c 'elspeth.web.app:create_app'
  args:
    executable: /bin/bash
  register: elspeth_worker_count
  changed_when: false
  failed_when: elspeth_worker_count.stdout | int != 1
  # If this returns >1 the multi-worker SQLite guard at
  # src/elspeth/web/app.py:517 has been bypassed (env-file override,
  # systemd drop-in, or a process supervisor wrapping uvicorn).
  # The guard fires at create_app() time, so a misconfigured WEB_CONCURRENCY
  # only crashes on the next restart — which is too late to catch in this
  # verification window. Treat this assertion as load-bearing.
```

**Flush handlers before verifying.** Every install task in the role
above (`git`, `pip install -e`, `npm ci`, `npm run build`,
`compileall`, env-file template, unit template) notifies the
`restart elspeth web` handler. Ansible handlers fire at *end of play*
by default — so without an explicit flush the verification probes
below would run against the *pre-restart* binary, the marker
rotation would lock in the new SHA as verified, and the actual
restart with the new code would happen with no further check. Force
handlers to run *before* verification:

```yaml
- name: Flush pending restart handlers before verifying
  ansible.builtin.meta: flush_handlers
  # CRITICAL: without this, the public-edge probes below report on
  # the old code while the role is mid-deploy. The handler-driven
  # restart would happen at end-of-play, after marker rotation, with
  # no further verification. flush_handlers forces the queued
  # `restart elspeth web` + `restart caddy` handlers to fire here,
  # so verification exercises the new binary + new ingress config.
```

**Public-edge verification (Ansible-asserted).** Replace the
copy-pastable bash one-liners with `ansible.builtin.uri` tasks so the
play fails fast on the local-vs-edge mismatch named in the next
paragraph. The play runs these from the *Ansible controller* against
the public hostname, not from the VM itself, so they exercise DNS,
TLS, ingress, and the reverse-proxy path end-to-end:

```yaml
- name: Verify public HTTPS edge (Ansible-asserted; delegated to controller)
  ansible.builtin.uri:
    url: "https://{{ elspeth_domain }}/api/health"
    method: GET
    status_code: 200
    validate_certs: "{{ elspeth_public_probe_validate_certs | default(not (elspeth_caddy_tls_internal | default(false) | bool)) }}"
    return_content: false
    follow_redirects: none
  register: elspeth_public_health
  retries: 6
  delay: 10
  until: elspeth_public_health.status == 200
  delegate_to: localhost
  run_once: true
  # delegate_to: localhost runs the probe from the Ansible controller,
  # not from the VM. From the VM itself, a "public" probe can short-
  # circuit through /etc/hosts, the local Caddy instance, or a
  # split-horizon DNS resolver and report green without ever touching
  # the public ingress path. Probing from the controller traverses
  # the real DNS + TLS + Front Door / NSG path, which is the property
  # we want to assert.
  #
  # validate_certs defaults to TRUE except when elspeth_caddy_tls_internal
  # is set — Caddy's internal-CA mode (see "Caddy TLS Provisioning
  # Modes") issues certs the controller does not trust by default,
  # so probing such hosts with validate_certs=true would always
  # fail. Container Apps uses Azure-managed public-CA TLS, never
  # tls internal, so the container check below keeps validate_certs
  # hard-coded to true.

- name: Verify Azure Front Door endpoint when configured
  ansible.builtin.uri:
    url: "https://{{ azure_frontdoor_endpoint }}.azurefd.net/api/health"
    method: GET
    status_code: 200
    validate_certs: "{{ elspeth_public_probe_validate_certs | default(not (elspeth_caddy_tls_internal | default(false) | bool)) }}"
    return_content: false
    follow_redirects: none
  register: elspeth_frontdoor_health
  retries: 6
  delay: 10
  until: elspeth_frontdoor_health.status == 200
  delegate_to: localhost
  run_once: true
  when: azure_frontdoor_endpoint is defined
  # Probing both the Front Door endpoint AND the public-host alias
  # catches misconfigured custom-domain bindings on Front Door —
  # i.e., the *.azurefd.net endpoint is green but elspeth_domain
  # is not yet bound to the Front Door endpoint, or the binding
  # is stale.

- name: Verify Container Apps FQDN when configured
  ansible.builtin.uri:
    url: "https://{{ azure_container_app_fqdn }}/api/health"
    method: GET
    status_code: 200
    validate_certs: true
    return_content: false
  register: elspeth_container_health
  retries: 6
  delay: 10
  until: elspeth_container_health.status == 200
  delegate_to: localhost
  run_once: true
  when: azure_container_app_fqdn is defined
```

For interactive one-off probes during incident response, the
copy-pastable equivalents are:

```bash
curl -fsS https://elspeth.example.com/api/health
curl -fsS "https://${FRONT_DOOR_ENDPOINT}.azurefd.net/api/health"
curl -fsS "https://${AZURE_CONTAINER_APP_FQDN}/api/health"
```

These are operator tools, not deploy gates — the deploy gate is the
Ansible-asserted block above.

**Post-deploy: rotate the deploy-SHA marker and handle forced
session-DB deletion.** These tasks run AFTER public-edge verification
has passed; a failed verification short-circuits the play before
either task fires, so a half-deployed host keeps its old marker (and
its old `sessions.db`) and is ready for the operator to re-run the
playbook against the previous SHA:

```yaml
- name: Delete sessions.db when schema-change deploy was acknowledged
  ansible.builtin.file:
    path: "{{ elspeth_data_dir }}/sessions.db"
    state: absent
  when: elspeth_force_session_db_delete | default(false) | bool
  notify: restart elspeth web
  # In-flight composer sessions are lost. This is the documented
  # project policy for schema-change deploys ("Database Lifecycle").
  # The corresponding .db-wal / .db-shm sidecars are deleted by the
  # systemd unit on next start (SQLite handles fresh sidecar
  # creation); explicit cleanup here would race with any handler
  # that hasn't fired yet.

- name: Rotate deploy-SHA marker (records this SHA as last to touch the DB)
  ansible.builtin.copy:
    content: "{{ elspeth_repo_version }}"
    dest: "{{ elspeth_data_dir }}/.last-deploy-sha"
    owner: "{{ elspeth_user }}"
    group: "{{ elspeth_group }}"
    mode: "0600"
  # The marker is the basis for the schema-drift preflight on the
  # NEXT deploy. Rotate AFTER successful public-edge verification —
  # rotating before would lock in this SHA as the new baseline even
  # if the public-edge probe later fails and the deploy is rolled
  # back. The rollback playbook is the same playbook re-run with
  # elspeth_repo_version pointed at the previous-known-good SHA and
  # elspeth_restore_db_from_snapshot=true; on a successful rollback
  # this same task fires and rotates the marker to the rolled-back-
  # to SHA, which is the correct new baseline. On a *failed*
  # rollback the play aborts before this task and the marker is
  # untouched, preserving the failing-SHA baseline for operator
  # forensics.
```

Deployment evidence to retain:

- Git commit or immutable image tag deployed.
- Frontend build timestamp or artifact digest.
- Local health-check response.
- Public health-check response.
- Azure Front Door origin health status, when applicable.
- Rollback target commit, tag, or revision.

If the local VM socket check passes but the public check fails, inspect the
ingress layer first: Caddy, NSG, Azure Front Door origin health, route caching,
certificate validation, and origin host header.

## Database Lifecycle

ELSPETH persists three categories of state on the VM. They have different
upgrade semantics and you must handle them differently across deploys.

| Store | URL variable | Default path | Upgrade policy |
|-------|--------------|--------------|----------------|
| Web session DB | `ELSPETH_WEB__SESSION_DB_URL` | `/var/lib/elspeth/sessions.db` | **Operator-deletes on schema change.** No Alembic migrations. Deleting drops in-flight composer sessions AND a partial slice of composer audit history (interpretation events, completion-gesture events, YAML-export events — see the composer audit-table inventory below). |
| Landscape (audit trail) | `ELSPETH_WEB__LANDSCAPE_URL` | `/var/lib/elspeth/runs/audit.db` | **Back up before every deploy.** This is the legal record of every prior decision. Loss is not recoverable. |
| Payload store | `ELSPETH_WEB__PAYLOAD_STORE_PATH` | `/var/lib/elspeth/payloads` | Survives across deploys. Purge via `elspeth purge --retention-days N`, never delete-on-deploy. |

**Current project policy** (as of 2026-05-18) is that schema changes ship
without migrations: when the running code and the on-disk DB schema diverge,
the operator deletes the on-disk DB. This is intentional for the
pre-1.0 line; it will change in a future release that introduces the
migration runner. For now, the deploy procedure is:

1. **Snapshot both databases** (the role's "Snapshot session and audit
   databases before upgrade" task above does this automatically). Snapshots
   are written next to the live DB with the deploying SHA *and* an
   ISO-8601 suffix:
   `sessions.db.pre-deploy-of-<sha>-20260518T143012` and
   `runs/audit.db.pre-deploy-of-<sha>-20260518T143012`. The SHA prefix
   lets the rollback role unambiguously locate the right baseline when
   more than one deploy happened in the same session — see
   [Rollback Automation](#rollback-automation).

   **WAL safety.** The snapshot task uses `sqlite3 ".backup"` rather
   than a raw `cp` / `ansible.builtin.copy` of the `.db` file. Under
   WAL mode (which ELSPETH's web app uses for concurrent reads), a
   raw byte-copy captures the main `.db` but not the `.db-wal`
   sidecar — any committed-but-not-yet-checkpointed transactions are
   absent from the snapshot, and the resulting file is torn. The
   SQLite Online Backup API (invoked by `.backup`) serializes
   against writers and produces a consistent point-in-time copy
   regardless of WAL state. The restore path likewise stops the
   service, removes stale `.db-wal` / `.db-shm` sidecars (which
   otherwise replay over the restored file on next open), and
   archives the pre-rollback `audit.db` off-host before overwrite.
2. **For a schema-incompatible 0.7.1 upgrade from 0.7.0**, stop the service and
   recreate the session database before handing the site back to users:
   ```bash
   sudo -u elspeth rm /var/lib/elspeth/sessions.db
   sudo systemctl restart elspeth-web.service
   ```
   The exact paths and SQLite sidecar handling depend on the deployment; use
   [staging-session-db-recreation.md](staging-session-db-recreation.md) as the
   current operational source of truth. Keep the epoch-22 Landscape audit DB
   for a direct 0.7.0→0.7.1 upgrade. Deployments crossing from an older release
   must also apply the historical 0.7.0 two-database reset. `data/auth.db` is
   separate and survives both reset paths.
3. **On rollback**, restore both pre-deploy snapshots before re-running the
   previous-version playbook. See [Rollback Automation](#rollback-automation).
4. **Snapshot retention**: keep at least the last three pre-deploy snapshots
   of `audit.db` off-host (S3, Azure Blob, or backup share). The on-host
   snapshots are an emergency local rollback aid, not a backup.

For container deployments, mount the audit DB on durable storage external
to the container. Container-local SQLite for audit data is unsafe regardless
of whether the schema is stable.

### SESSION_SCHEMA_EPOCH — Mechanical Schema-Change Detection

`sessions.db` does NOT carry a migration runner. The compatibility
contract is mechanical: the model layer declares
`SESSION_SCHEMA_EPOCH` (an integer in `src/elspeth/web/sessions/models.py`)
and the bootstrap stamps it into `PRAGMA user_version` on fresh databases.
On every subsequent startup, `_assert_schema_sentinels`
(`src/elspeth/web/sessions/schema.py`) reads `PRAGMA user_version` back
and crashes the service with an actionable error if it diverges from the
constant the running code expects:

```text
SessionSchemaError: Session DB schema version 35 does not match
SESSION_SCHEMA_EPOCH=36. Pre-release ELSPETH does not migrate session
databases. Delete the session DB file and restart.
```

A second sentinel, `PRAGMA application_id = 0x454C5350` (which spells
"ELSP" in ASCII), is also enforced so the bootstrap refuses to touch a
SQLite file that happens to live at the configured path but belongs to
some other application entirely.

**Operator detection cadence.**
The "schema-incompatible upgrade" decision is *not* a judgement call —
the running code asserts it on startup. The deploy procedure should:

1. Archive `sessions.db` as evidence (per the standard snapshot task above).
2. Restart the service (the role's standard restart-and-verify gate
   already covers this).
3. If the restart fails with `SessionSchemaError`, the error message
   itself is the actionable instruction: delete the file, restart again.
   The archive is diagnostic evidence, not a downgrade database; recreate the
   live DB with the current release.

**What happens if you skip the delete.**
The service refuses to start at all. There is no partial-functionality
mode; `_assert_schema_sentinels` runs before any other initialisation.
A staging deploy that bumps the epoch without operator-side `rm`
becomes a crash-restart loop until systemd's `StartLimitBurst`
exhausts and the unit goes into `failed` state.

**Recent epoch history** (current code under `models.py:SESSION_SCHEMA_EPOCH`):

| Epoch | Rationale |
|-------|-----------|
| 1 | Initial schema. |
| 2 | `interpretation_events_table` added (Phase 5b). |
| 3 | `composition_proposals.user_message_id` added (chat-message provenance). |
| 4 | `composer_completion_events_table` added (Phase 6A — `mark_ready_for_review`, `export_yaml`). |
| 5 | Per-event-type partial CHECK constraints on `composer_completion_events` (Phase 6A post-merge hardening). |
| 26 | 0.7.0 schema; includes guided Composer state, local auth/email-verification support, and first-run tutorial resume fields. |
| 27 | Adds the account-wide freeform-primer dismissal preference. |
| 28 | Adds the cross-dialect application/store/epoch identity proof. |
| 29 | Adds guided schema 8 and durable fenced guided-operation reservations. |
| 30 | Adds the closed `quota_exceeded` failure code for stable HTTP 413 fork replay. |
| 31 | Adds guided schema 9 and durable pipeline-proposal replay locators. |
| 32 | Binds exact audit evidence to failed guided operations. |
| 33 | Adds the durable guided-start negative admission barrier. |
| 34 | Adds guided schema 10 and removes obsolete advisor counters. |
| 35 | Adds exclusive guided-confirmation proposal admission. |
| 36 | Current schema; adds durable retry state for post-commit blob tombstone cleanup. |

The constant should be the authoritative reference; this table is a
durability aid for operators reading the runbook in isolation.

### Append-Only Trigger Set — Required At Startup

`sessions.db` carries six SQLite triggers that enforce audit invariants
the column-level NOT NULL / CHECK surface cannot express on its own.
`_validate_required_triggers` enumerates the required set and refuses
service startup if any trigger is missing:

| Trigger | Purpose |
|---------|---------|
| `trg_interpretation_events_immutable_resolved` | Resolved interpretation-event rows are immutable. |
| `trg_interpretation_events_no_delete_resolved` | Resolved interpretation-event rows cannot be deleted (PENDING rows remain deletable for orphan recovery). |
| `trg_composer_completion_events_no_update` | Completion events (`mark_ready_for_review`, `export_yaml`) are unconditionally append-only — UPDATE is forbidden. |
| `trg_composer_completion_events_no_delete` | Completion events are unconditionally append-only — DELETE is forbidden. |
| `trg_chat_messages_immutable_content` | `chat_messages.content` is append-only once written (chat is an audit anchor via `blobs.created_from_message_id`). |
| `trg_chat_messages_no_delete` | `chat_messages` rows cannot be deleted directly; whole-session archival is the bounded lifecycle purge path via `sessions` cascade. |

**Operator implication.**
Direct DDL manipulation on `sessions.db` (e.g. attaching with the
`sqlite3` CLI and running `DROP TRIGGER ...` to "unblock" a workflow)
makes the service refuse to start. The fix is the standard fix:
delete the DB and let bootstrap recreate it. Do NOT recreate the
trigger by hand — the `event.listen(after_create)` registration is
table-scoped and the trigger DDL is in code; the hand-recreated SQL
will drift from the model definition the next time it changes.

### Composer Audit-Table Inventory In sessions.db

The "delete sessions.db on schema change" framing is correct for
recovery, but it understates the audit surface that lives in
`sessions.db` alongside the in-flight composer state. The following
tables carry composer-side audit data — they are NOT ephemeral session
chrome, even though they reset on epoch bump:

| Table | Phase | Append-only triggers | Carries |
|-------|-------|----------------------|---------|
| `composition_states` | 1 | (no — CHECK on `provenance` closed enum) | Every committed composition state. `provenance` enum names which writer produced it. |
| `composition_proposals` | 1A | (no) | Explicit-approval proposals awaiting accept/reject. `user_message_id` ties them to chat-message provenance. |
| `chat_messages` | 1 | content-immutable + no-delete | The composer-session transcript. Audit anchor for the blob lineage walk. |
| `interpretation_events` | 5b | resolved-row immutable + resolved-row no-delete | LLM-interpretation surface events (one per surfaced term). PENDING rows remain deletable for orphan recovery. |
| `composer_completion_events` | 6A | unconditional no-update + no-delete | `mark_ready_for_review` and `export_yaml` audit events. Per-event-type partial CHECKs enforce `payload_digest`/`expires_at`/`composition_state_id` invariants. |
| `runs` | 1 | (CHECK on `status` closed enum) | Pipeline runs initiated from the composer. |

**Operational consequence.**
When the runbook says "deleting `sessions.db` drops in-flight composer
sessions," that is true but incomplete. It also discards every
composer-side audit row enumerated above. For deployments where the
composer audit trail must be preserved across schema bumps, the
operator playbook is: snapshot `sessions.db` per the standard task,
*export* the audit-relevant rows from the snapshot (e.g.
`sqlite3 sessions.db.bak-... .dump composer_completion_events > evidence.sql`),
archive the export, then proceed with the delete. There is no
out-of-the-box migration to carry these rows forward — only export +
archive.

### Closed-CHECK Enum Reference (sessions.db)

Several `sessions.db` columns are governance-locked closed enums
enforced by SQLite CHECK constraints. An operator opening
`sqlite3 sessions.db ".schema"` will see these constraints, but
they are NOT stylistic — they are paired contracts with corresponding
Python `Literal` type aliases in the source. Extending one without
the other lets the dataclass validator pass while the DB rejects the
row, or vice versa. **Adding a value requires a spec amendment and a
schema-change cohort with epoch bump.**

| Table | Column | Allowed values | Paired Python type |
|-------|--------|----------------|--------------------|
| `composition_states` | `provenance` | `tool_call`, `convergence_persist`, `plugin_crash_persist`, `preflight_persist`, `tutorial_normalization`, `post_compose`, `session_seed`, `session_fork`, `interpretation_resolve` | `web/sessions/protocol.py::CompositionStateProvenance` |
| `composer_completion_events` | `event_type` | `mark_ready_for_review`, `export_yaml` | Inline check, see `web/sessions/models.py` |
| `runs` | `status` | `pending`, `running`, `completed`, `completed_with_failures`, `failed`, `empty`, `cancelled` | `web/sessions/protocol.py::SessionRunStatus` |
| `interpretation_events` | `event_type` | (Phase 5b — see `web/sessions/models.py` for current set) | `web/sessions/protocol.py` |
| `proposal_events` | `event_type` | `proposal.created`, `proposal.accepted`, `proposal.rejected`, `trust_mode.changed` | Governance-locked; see `composer_completion_events` for the rationale (split into a separate table to avoid conflating decision families). |
| `audit_access_log` | `writer_principal` | (closed set; see `web/sessions/models.py`) | Governance-locked. |

`composer_completion_events` additionally carries the per-event-type
partial CHECK constraints landed in Phase 6A:

* `ck_composer_completion_events_digest_iff_mark_ready` —
  `payload_digest IS NOT NULL` iff `event_type = 'mark_ready_for_review'`.
* `ck_composer_completion_events_expires_iff_mark_ready` —
  `expires_at IS NOT NULL` iff `event_type = 'mark_ready_for_review'`.
* `ck_composer_completion_events_composition_state_id_required` —
  `composition_state_id IS NOT NULL` for both event types.

An operator hand-inserting a row that violates any of these CHECKs
will see an `IntegrityError` from SQLite. The error message is the
constraint name, which is named-for-purpose — the operator can map
back to this table without source-diving.

### Secret-Rotation Playbooks

The web service carries multiple HMAC-style keys with different
rotation semantics. Each row below names a key, the variable that
holds it, and the operational consequence of rotating it. **No
operator should rotate one of these keys without first understanding
the consequence column** — there is no rotation-aware migration for
any of them.

| Key | Vault var | Where it signs | Rotation consequence |
|-----|-----------|----------------|----------------------|
| `secret_key` | `vault_elspeth_web_secret_key` | Session/cookie signing | Existing sessions are invalidated; users must re-authenticate. Safe to rotate. |
| `fingerprint_key` | `vault_fingerprint_key` | Audit-record fingerprints in the Landscape DB | **DO NOT ROTATE without migrating audit DB.** Existing fingerprints become unverifiable; the audit trail's integrity claim breaks for every pre-rotation row. The env-file comment that ships with this variable explicitly forbids rotation. |
| `shareable_link_signing_key` | `vault_elspeth_web_shareable_link_signing_key` | Shareable-review capability tokens (`GET /api/sessions/shared/{token}`) | **Invalidates EVERY outstanding shareable link.** There is no dual-key acceptance window in v1; recipients with active links receive 401 on next use. Acceptable as an emergency leak-response action. Otherwise treat as a deliberate operational event with stakeholder communication. |
| `landscape_passphrase` | `vault_landscape_passphrase` (if encrypting) | SQLCipher passphrase for the Landscape DB at rest | Re-keying the live DB requires `PRAGMA rekey` against the running file. There is no playbook for this in-runbook today; if you need this, build the playbook before rotation, not after. |

For the `shareable_link_signing_key` specifically, the rotation
procedure is:

1. Generate the replacement key on a trusted workstation:
   `openssl rand -base64 32`.
2. Update the vault file (`vault_elspeth_web_shareable_link_signing_key`)
   or push the new value to Key Vault and bump the version reference in
   `azure_keyvault_elspeth_web_shareable_link_signing_key_url`.
3. Notify any stakeholders with outstanding shareable links that their
   links will return 401 after the next deploy.
4. Run the standard deploy.
5. Recipients of broken links request fresh ones via the composer UI.

There is intentionally NO "soft rotation" mode — the threat model for
this key (HMAC of a capability that travels in a URL) does not permit
dual-key verify without expanding the wire format. A v2 envelope
shape supporting dual-key verify is a future-release item.

### What Rollback Does To The Audit Trail

When a rollback restores `audit.db` from a pre-deploy snapshot, every
row written by the failing version between snapshot time and rollback
time is **discarded**. This is the correct operational behaviour — the
failing version's writes are themselves suspect, and resuming with a
mixed-version audit DB would leave the on-disk schema in a state the
running code does not expect — but it is *also* an auditability event
in its own right that must be recorded.

**Before any rollback that sets `elspeth_restore_db_from_snapshot=true`:**

1. **Capture the discarded-window boundary** from the live
   `audit.db` BEFORE running the rollback playbook. The on-host
   `runs/audit.db` is about to be overwritten by the snapshot.
   The snapshot's mtime is the floor of the discarded window — it
   is embedded in the snapshot filename, e.g.
   `runs/audit.db.pre-deploy-of-<sha>-20260518T143012` →
   `2026-05-18T14:30:12`. Pass that timestamp as the `WHERE` floor:

   ```bash
   # Replace SNAPSHOT_TS with the ISO-8601 timestamp from the
   # pre-deploy snapshot filename (the suffix after "pre-deploy-of-<sha>-").
   # Replace column names to match your Landscape schema; the point is
   # to record the boundary, not to copy this query verbatim.
   SNAPSHOT_TS='2026-05-18T14:30:12'
   sqlite3 /var/lib/elspeth/runs/audit.db <<SQL
     SELECT MAX(run_id)     AS lost_run_id_ceiling,
            MAX(token_id)   AS lost_token_id_ceiling,
            COUNT(*)        AS discarded_row_count,
            MIN(started_at) AS discarded_window_start,
            MAX(started_at) AS discarded_window_end
       FROM runs
      WHERE started_at >= '$SNAPSHOT_TS';
   SQL
   ```

   The earlier draft of this query used a `(SELECT ... ORDER BY
   started_at DESC LIMIT 1)` subquery as the floor — that returns the
   *single most recent row's* timestamp, so the outer `WHERE >= max`
   matches exactly that one row. The discarded-row count was
   structurally `1`, not "rows the failing version wrote." If you
   copy-pasted that earlier form, re-run the corrected query above
   before treating the boundary as authoritative.

   Capture the output to the deploy log alongside `elspeth_repo_version`
   and `elspeth_rollback_from_repo_version`.

2. **Note the discarded SHA in the post-rollback audit attestation.**
   The rollback role writes a single row into the *restored* `audit.db`
   recording: "rolled back FROM `elspeth_rollback_from_repo_version`
   TO `elspeth_repo_version`, discarding N rows written between T1
   and T2." That row is the only evidence inside the audit trail that
   the discarded window existed. Without it, the trail looks
   contiguous — a future auditor querying "what happened between T1
   and T2?" sees nothing, with no signal that anything was discarded.
   The implementing tasks are in "Rollback Automation" below, gated
   on `elspeth_restore_db_from_snapshot=true`.

3. **The off-host snapshot is the record of what was discarded.** The
   pre-deploy snapshot is named
   `runs/audit.db.pre-deploy-of-<sha>-<ts>`; the *failing* version's
   writes accumulated on top of that snapshot before being discarded
   by rollback. The full pre-rollback `audit.db` (the one being
   overwritten) should be copied off-host before the playbook restore
   step runs, so the discarded rows are recoverable if a later auditor
   needs them. The runbook's "Snapshot retention" item above covers
   the *pre-deploy* snapshot but not the *pre-rollback* snapshot —
   make sure both are shipped.

**The trade-off named.** Restoring `audit.db` from snapshot favours
*consistency* over *retention*: the audit trail post-rollback is
internally consistent (no rows from a version that no longer runs
anywhere) but is missing the discarded window. The alternative —
leaving the failing version's audit rows in place and only rolling
back code — favours retention over consistency: every row is kept,
but the code now reads/writes a schema it wasn't written for, and any
column added or repurposed by the failing version produces wrong
answers on rollback queries. ELSPETH's pre-1.0 policy (no Alembic;
delete-the-DB on schema change) makes the consistency choice the only
safe one. When the migration runner ships, this trade-off can be
re-evaluated.

Cross-references:

- [Backup and Recovery runbook](backup-and-recovery.md) — full backup/restore
  procedures for both databases.
- [Database Maintenance runbook](database-maintenance.md) — purge, vacuum,
  and growth management.

## Rollback Automation

Rollback for VM deployments is a re-run of the same playbook with three
variables overridden:

```bash
ansible-playbook -i inventories/non-cloud.yml playbooks/elspeth-vm.yml \
  --vault-password-file "$ANSIBLE_VAULT_PASSWORD_FILE" \
  -e elspeth_repo_version="$PREVIOUS_KNOWN_GOOD_SHA" \
  -e elspeth_rollback_from_repo_version="$FAILED_SHA" \
  -e elspeth_restore_db_from_snapshot=true
```

`elspeth_rollback_from_repo_version` names the SHA you are rolling back
*away from* — i.e., the failing deploy. The role uses it to locate the
snapshot taken just before that SHA's deploy began. Without this
variable the role refuses to restore, on the grounds that an `mtime`-only
selection (the previous version of this role) can silently restore the
wrong snapshot when two deploys ran in the same session.

The role honors `elspeth_restore_db_from_snapshot` by replacing
`sessions.db` and `runs/audit.db` with the snapshot taken before
`elspeth_rollback_from_repo_version` started writing, *before*
restarting the service:

```yaml
- name: Refuse rollback DB-restore without a failed-SHA reference
  ansible.builtin.assert:
    that:
      - elspeth_rollback_from_repo_version is defined
      - elspeth_rollback_from_repo_version | length > 0
    fail_msg: >-
      elspeth_restore_db_from_snapshot=true requires
      elspeth_rollback_from_repo_version (the SHA whose pre-deploy
      snapshot should be restored). The previous mtime-only selection
      can silently restore the wrong snapshot when two deploys ran in
      the same session.
  when: elspeth_restore_db_from_snapshot | default(false) | bool

- name: Locate pre-deploy snapshots for the failed revision
  ansible.builtin.find:
    paths: "{{ elspeth_data_dir }}"
    patterns:
      - "sessions.db.pre-deploy-of-{{ elspeth_rollback_from_repo_version }}-*"
      - "runs/audit.db.pre-deploy-of-{{ elspeth_rollback_from_repo_version }}-*"
    recurse: true
  register: elspeth_db_snapshots
  when: elspeth_restore_db_from_snapshot | default(false) | bool

- name: Fail loudly if no matching snapshot exists
  ansible.builtin.assert:
    that:
      - elspeth_db_snapshots.files | length > 0
    fail_msg: >-
      No pre-deploy snapshot found for
      elspeth_rollback_from_repo_version={{ elspeth_rollback_from_repo_version }}.
      Refusing to roll back code without restoring DB state — that
      combination produces a known-good binary running against a
      known-bad on-disk schema/data. Restore from off-host backup or
      abort the rollback.
  when: elspeth_restore_db_from_snapshot | default(false) | bool

- name: Stop the web service before restoring DBs
  ansible.builtin.systemd:
    name: elspeth-web.service
    state: stopped
  when: elspeth_restore_db_from_snapshot | default(false) | bool
  # The restore sequence below removes stale .db-wal / .db-shm sidecars
  # from the failing version's writer. Doing that while the service is
  # running races against any in-flight checkpoints. Stop first, restore
  # main DB + sidecar cleanup, restart via handler at end-of-play.

- name: Capture pre-rollback audit DB off-host before overwrite
  # The pre-rollback audit.db contains the rows the failing version
  # wrote between snapshot time and rollback time. The restore step
  # below is about to overwrite it. If a later auditor needs to
  # recover the discarded rows, the off-host copy named here is the
  # only source. The runbook section "What Rollback Does To The Audit
  # Trail" prescribes this capture; the task below DOES it. The
  # default sink is local under elspeth_data_dir; override to a remote
  # location (S3, Azure Blob, SSH target) in your environment.
  ansible.builtin.copy:
    src: "{{ elspeth_data_dir }}/runs/audit.db"
    dest: "{{ elspeth_pre_rollback_audit_db_archive | default(elspeth_data_dir ~ '/runs/audit.db.pre-rollback-from-' ~ elspeth_rollback_from_repo_version ~ '-' ~ ansible_date_time.iso8601_basic_short) }}"
    remote_src: true
    owner: "{{ elspeth_user }}"
    group: "{{ elspeth_group }}"
    mode: "0600"
  when: elspeth_restore_db_from_snapshot | default(false) | bool

- name: Restore each database from its newest snapshot for the failed revision
  ansible.builtin.copy:
    src: "{{ (elspeth_db_snapshots.files
              | selectattr('path', 'match', '^' ~ elspeth_data_dir ~ '/' ~ dest_relative | regex_escape ~ '\\.pre-deploy-of-')
              | sort(attribute='mtime', reverse=true)
              | list)[0].path }}"
    dest: "{{ elspeth_data_dir }}/{{ dest_relative }}"
    remote_src: true
    owner: "{{ elspeth_user }}"
    group: "{{ elspeth_group }}"
    mode: "0600"
  loop:
    - sessions.db
    - runs/audit.db
  loop_control:
    loop_var: dest_relative
  when: elspeth_restore_db_from_snapshot | default(false) | bool
  # If the same SHA was deployed twice (rare — usually means the first
  # attempt failed before the snapshot task ran on the second attempt),
  # we still pick the newest snapshot for that SHA, which is the
  # closest-to-failed-state baseline we have. The SHA filter prevents
  # crossing over into a different deploy's snapshots.

- name: Remove stale WAL/SHM sidecars from the failing version
  # CRITICAL: after restoring the .db main file, any leftover .db-wal
  # and .db-shm from the failing version contain writes against the
  # failing schema. SQLite replays the WAL on next open, corrupting
  # the just-restored file. Deleting the sidecars forces SQLite to
  # initialize fresh ones on first open of the restored DB.
  ansible.builtin.file:
    path: "{{ elspeth_data_dir }}/{{ item }}"
    state: absent
  loop:
    - sessions.db-wal
    - sessions.db-shm
    - runs/audit.db-wal
    - runs/audit.db-shm
  when: elspeth_restore_db_from_snapshot | default(false) | bool

- name: Derive snapshot floor timestamp for attestation
  # The pre-deploy snapshot filename embeds the deploying SHA and an
  # ISO-8601 basic-short timestamp, e.g.:
  #   runs/audit.db.pre-deploy-of-<sha>-20260518T143012
  # The suffix after "pre-deploy-of-<sha>-" is the floor of the
  # discarded window (the moment the snapshot was taken). The
  # attestation row records that floor as discarded_window_start_basic.
  ansible.builtin.set_fact:
    elspeth_restored_audit_snapshot_path: "{{ (elspeth_db_snapshots.files
      | selectattr('path', 'match', '^' ~ elspeth_data_dir ~ '/runs/audit\\.db\\.pre-deploy-of-')
      | sort(attribute='mtime', reverse=true)
      | list)[0].path }}"
  when: elspeth_restore_db_from_snapshot | default(false) | bool

- name: Extract snapshot floor timestamp
  ansible.builtin.set_fact:
    elspeth_discarded_window_floor: "{{ elspeth_restored_audit_snapshot_path
      | regex_replace('^.*\\.pre-deploy-of-' ~ elspeth_rollback_from_repo_version ~ '-', '') }}"
  when: elspeth_restore_db_from_snapshot | default(false) | bool

- name: Ensure rollback_attestations table exists in the restored audit DB
  # The attestation lives in a dedicated table — co-mingling with the
  # operational `runs` table would force every auditor query to
  # filter on a synthetic discriminator. A separate table named
  # rollback_attestations is queryable as a first-class audit event.
  # CREATE TABLE IF NOT EXISTS is idempotent across re-runs.
  ansible.builtin.command:
    argv:
      - sqlite3
      - "{{ elspeth_data_dir }}/runs/audit.db"
      - >-
        CREATE TABLE IF NOT EXISTS rollback_attestations (
          attested_at_iso8601   TEXT NOT NULL,
          rolled_back_from_sha  TEXT NOT NULL,
          rolled_back_to_sha    TEXT NOT NULL,
          snapshot_floor_basic  TEXT NOT NULL,
          actor                 TEXT NOT NULL,
          notes                 TEXT
        );
  when: elspeth_restore_db_from_snapshot | default(false) | bool
  changed_when: false

- name: Write rollback attestation row into the restored audit DB
  # This row is the only signal inside the audit trail that a
  # discarded window existed. The off-host pre-rollback archive
  # (captured by "Capture pre-rollback audit DB off-host before
  # overwrite") is the only place the discarded rows themselves live;
  # the attestation row points to the archive's filename via
  # `notes`. An auditor querying the rollback window sees the
  # attestation row, learns the SHAs and floor timestamp, and
  # retrieves the archive for the discarded rows if needed.
  ansible.builtin.command:
    argv:
      - sqlite3
      - "{{ elspeth_data_dir }}/runs/audit.db"
      - >-
        INSERT INTO rollback_attestations (
          attested_at_iso8601, rolled_back_from_sha,
          rolled_back_to_sha, snapshot_floor_basic, actor, notes
        ) VALUES (
          '{{ ansible_date_time.iso8601 }}',
          '{{ elspeth_rollback_from_repo_version }}',
          '{{ elspeth_repo_version }}',
          '{{ elspeth_discarded_window_floor }}',
          '{{ elspeth_deploy_actor | default(ansible_user_id) | default("unattended-rollback", true) }}',
          'pre-rollback archive at {{ elspeth_pre_rollback_audit_db_archive | default(elspeth_data_dir ~ "/runs/audit.db.pre-rollback-from-" ~ elspeth_rollback_from_repo_version ~ "-" ~ ansible_date_time.iso8601_basic_short) }}'
        );
  when: elspeth_restore_db_from_snapshot | default(false) | bool
  changed_when: true
  notify: restart elspeth web
```

**Automatic rollback on post-deploy probe failure.** After cutover, run a
probe loop and trigger the rollback playbook if probes fail N
consecutive times.

**Where the probe script and wrappers live.** Both the probe loop
and its wrappers live on the **Ansible controller**, not on the
production host. The script invokes `ansible-playbook` against an
inventory + playbook tree and reads
`ANSIBLE_VAULT_PASSWORD_FILE` from a controller-local path — none of
that infrastructure exists on the deploy host (and shipping it there
would mean either deploying the whole operations repo + Ansible
runtime + vault to production, or maintaining a controller-on-prod
shape neither of which is desirable). Concretely:

- `post-deploy-probe.sh` lives in the operations repo at
  `scripts/post-deploy-probe.sh` (alongside the `inventories/`,
  `playbooks/`, and `roles/` trees referenced earlier in this
  runbook). Permissions: `0750`, owned by the controller's deploy
  user.
- `templates/run-deploy.sh.j2` (production wrapper) and
  `scripts/run-deploy-staging.sh` (staging wrapper) likewise live in
  the operations repo. The Jinja-rendered production wrapper is
  rendered *by Ansible at deploy time* into the controller's
  build/deploy area — the runbook shows it as a template so that
  the rendered `{{ elspeth_repo_version }}` and
  `{{ elspeth_previous_repo_version }}` values are pinned at the
  moment the deploy decision is made, not at the moment bash
  executes the wrapper.
- `ANSIBLE_VAULT_PASSWORD_FILE` likewise points at a
  controller-local 0400 file (read from the controller's secret
  manager: systemd credentials, Azure Key Vault file mount,
  HashiCorp Vault agent template). It is never copied to the
  deploy host.

If your deployment topology has the controller running on the
production host itself (single-host shop, lab setup), the
distinction collapses but the file paths still belong logically to
the operations repo, not to `elspeth_app_dir`. Either way, do NOT
deploy `post-deploy-probe.sh` to `{{ elspeth_app_dir }}/scripts/` —
that path is reserved for application-side scripts that the
running service or its operators need on the production host.

The script body below populates the operations repo at
`scripts/post-deploy-probe.sh`. The probe loop:

1. Hits every path in `PROBE_ENDPOINTS` (space-separated) against the
   base URL. Any non-2xx response counts as a failure for that
   iteration. Defaults to `/api/health` for backwards compatibility,
   but production deploys should expand the list to a representative
   sample of read-path and write-path endpoints. Staging deploys can
   probe more aggressively (more endpoints, tighter thresholds).
2. *Optionally* parses the Caddy JSON access log over the last
   `PROBE_ERROR_RATE_WINDOW` seconds and counts requests with HTTP
   status ≥ 500 against total. If the ratio exceeds
   `PROBE_ERROR_RATE_THRESHOLD`, the iteration counts as a failure
   even if the endpoint probes themselves returned 2xx. This catches
   regressions that break *real* traffic while leaving the probe
   endpoints intact.
3. **Treats "no traffic in the window" as inconclusive**, not as
   healthy. A pure synthetic probe against a service that no real
   client is using produces 0/0 — distinguishing that from "service
   is healthy and serving 1000 req/s with zero errors" requires
   either real traffic or an explicit decision. The script does not
   trigger rollback on inconclusive but also does not reset the
   consecutive-failure counter on it.

The rollback path is **non-interactive by design**: the probe can fire
after the operator has left, so the script reads the vault password
from a file referenced by `ANSIBLE_VAULT_PASSWORD_FILE` rather than
prompting on stdin. The file should live on a path readable only by
the deploy account (mode `0400`, owned by the deploy user) and be
sourced from your host secret manager (systemd credentials, Azure
Key Vault file mount, HashiCorp Vault agent template, etc.).

```bash
#!/usr/bin/env bash
# scripts/post-deploy-probe.sh
set -euo pipefail

BASE_URL="${1:?usage: post-deploy-probe.sh BASE_URL PREVIOUS_SHA FAILED_SHA}"
PREVIOUS_SHA="${2:?usage: post-deploy-probe.sh BASE_URL PREVIOUS_SHA FAILED_SHA}"
FAILED_SHA="${3:?usage: post-deploy-probe.sh BASE_URL PREVIOUS_SHA FAILED_SHA}"
: "${ANSIBLE_VAULT_PASSWORD_FILE:?ANSIBLE_VAULT_PASSWORD_FILE must point at a 0400 file readable by the deploy user; --ask-vault-pass is not usable from an unattended probe}"

WINDOW="${PROBE_WINDOW_SECONDS:-300}"
INTERVAL="${PROBE_INTERVAL_SECONDS:-10}"
FAIL_THRESHOLD="${PROBE_FAIL_THRESHOLD:-3}"

# Space-separated list of paths to hit each iteration. Override per env:
#   PROBE_ENDPOINTS="/api/health /api/sessions/_smoke /api/auth/_status"
# Paths must not contain whitespace; URL-encode if necessary.
PROBE_ENDPOINTS="${PROBE_ENDPOINTS:-/api/health}"

# Optional error-rate gate. Leave PROBE_ERROR_RATE_LOG unset to disable.
# Requires jq on the probe host.
PROBE_ERROR_RATE_LOG="${PROBE_ERROR_RATE_LOG:-}"
PROBE_ERROR_RATE_WINDOW="${PROBE_ERROR_RATE_WINDOW:-60}"
PROBE_ERROR_RATE_THRESHOLD="${PROBE_ERROR_RATE_THRESHOLD:-0.01}"

probe_endpoints() {
    # Returns 0 if all endpoints returned 2xx, 1 otherwise. Prints which
    # endpoint failed (and its HTTP status) to stderr.
    local path status
    for path in $PROBE_ENDPOINTS; do
        # curl -w '%{http_code}' writes the status code to stdout with
        # NO trailing newline. On total failure (DNS, TLS, connection
        # refused) curl writes nothing and we substitute '000' via
        # `printf` — also with no newline — so the integer comparison
        # below sees a clean three-digit token in both branches.
        # `echo` would append a newline that some shells include in
        # the captured string and break the comparison under stricter
        # locales / shells.
        status=$(curl -s -o /dev/null -w '%{http_code}' "${BASE_URL%/}${path}" || printf '000')
        if [ "$status" -lt 200 ] || [ "$status" -ge 300 ]; then
            echo "probe ${path} returned ${status}" >&2
            return 1
        fi
    done
    return 0
}

check_error_rate() {
    # Returns 0 = healthy, 1 = exceeded threshold, 2 = inconclusive (no
    # traffic in window, log missing, jq unavailable, etc.). Caller
    # treats 2 as "do not change failure counter."
    [ -z "$PROBE_ERROR_RATE_LOG" ] && return 0  # gate disabled
    [ -r "$PROBE_ERROR_RATE_LOG" ] || return 2
    command -v jq >/dev/null || return 2

    local cutoff total errors
    cutoff=$(( $(date -u +%s) - PROBE_ERROR_RATE_WINDOW ))

    # Caddy emits ts as a Unix epoch float. The -R flag reads each
    # line as raw text and `fromjson?` parses it, silently dropping
    # any line that is not valid JSON. Without this, a single
    # malformed line (Caddy config glitch, partial write during
    # rotation, log-format change between versions) makes jq exit
    # non-zero, which under `set -e` + `set -o pipefail` aborts the
    # whole probe — turning the error-rate gate into a new silent
    # regression. fromjson? is the canonical tolerance pattern.
    total=$(jq -rcR --argjson cutoff "$cutoff" \
        'fromjson? | select((.ts // 0) >= $cutoff) | 1' \
        "$PROBE_ERROR_RATE_LOG" 2>/dev/null | wc -l)
    [ "$total" -eq 0 ] && return 2  # inconclusive — no traffic in window

    errors=$(jq -rcR --argjson cutoff "$cutoff" \
        'fromjson? | select((.ts // 0) >= $cutoff) | select((.status // 0) >= 500) | 1' \
        "$PROBE_ERROR_RATE_LOG" 2>/dev/null | wc -l)

    # awk exit 1 = "rate exceeded threshold" (function returns 1)
    if awk -v e="$errors" -v t="$total" -v thresh="$PROBE_ERROR_RATE_THRESHOLD" \
        'BEGIN { exit !((e / t) > thresh) }'; then
        echo "error rate ${errors}/${total} over ${PROBE_ERROR_RATE_WINDOW}s exceeded threshold ${PROBE_ERROR_RATE_THRESHOLD}" >&2
        return 1
    fi
    return 0
}

rollback() {
    echo "post-deploy probe failed ${FAIL_THRESHOLD} times; rolling back from ${FAILED_SHA} to ${PREVIOUS_SHA}" >&2
    ansible-playbook -i inventories/non-cloud.yml playbooks/elspeth-vm.yml \
        --vault-password-file "$ANSIBLE_VAULT_PASSWORD_FILE" \
        -e elspeth_repo_version="$PREVIOUS_SHA" \
        -e elspeth_rollback_from_repo_version="$FAILED_SHA" \
        -e elspeth_restore_db_from_snapshot=true
    exit 1
}

deadline=$(( $(date +%s) + WINDOW ))
consecutive_failures=0

while [ "$(date +%s)" -lt "$deadline" ]; do
    iteration_failed=0
    iteration_inconclusive=0

    if ! probe_endpoints; then
        iteration_failed=1
    fi

    # Only run the error-rate gate if endpoint probes passed. If they
    # already failed there is no point checking traffic; the iteration
    # is already counted as failed.
    if [ "$iteration_failed" -eq 0 ]; then
        check_error_rate
        case $? in
            0) ;;                                  # healthy
            1) iteration_failed=1 ;;              # exceeded threshold
            2) iteration_inconclusive=1 ;;        # no traffic / log missing
        esac
    fi

    if [ "$iteration_failed" -eq 1 ]; then
        consecutive_failures=$(( consecutive_failures + 1 ))
        if [ "$consecutive_failures" -ge "$FAIL_THRESHOLD" ]; then
            rollback
        fi
    elif [ "$iteration_inconclusive" -eq 0 ]; then
        # Only reset on a CLEAN pass — neither failed nor inconclusive.
        # An inconclusive iteration (no traffic in the error-rate
        # window, log file missing, jq not installed) MUST leave the
        # consecutive-failure counter unchanged, as documented in the
        # "Treats no traffic in the window as inconclusive" paragraph
        # above. The earlier version of this script reset the counter
        # on inconclusive iterations, contradicting its own docs.
        consecutive_failures=0
    fi
    sleep "$INTERVAL"
done
echo "post-deploy probe window elapsed cleanly"
```

Invoke it as the final step of the deploy pipeline. `FAILED_SHA` is
the SHA you *just deployed* (the candidate that may need rolling
back); `PREVIOUS_SHA` is the known-good revision you would fall back
to. Endpoint lists, error-rate thresholds, and the Caddy log path
should be set per-environment in your deploy wrapper.

The two examples below are **Ansible-rendered wrapper templates**
(e.g. `templates/run-deploy.sh.j2`), not literal bash scripts. The
`{{ elspeth_previous_repo_version }}` / `{{ elspeth_repo_version }}`
placeholders are Jinja, evaluated by Ansible at render time — by the
time bash runs the wrapper, the SHAs are baked in as plain strings.
Do not copy-paste the production example into a hand-edited
`.sh` file: bash will pass the literal four-character `{{ }}` tokens
as SHAs, the rollback role's snapshot lookup will find nothing
matching `pre-deploy-of-{{...}}-*`, and the rollback assert will
fire. The staging example uses `"$PREVIOUS_SHA"` / `"$FAILED_SHA"`
shell variables instead, so it can be used as a literal script
when those vars are exported by the caller.

```bash
# templates/run-deploy.sh.j2 — production wrapper, rendered by Ansible.
# The {{ }} placeholders are Jinja; bash never sees them after render.
ANSIBLE_VAULT_PASSWORD_FILE=/etc/ansible/vault.pass \
PROBE_ENDPOINTS="/api/health /api/sessions/_smoke /api/auth/_status" \
PROBE_ERROR_RATE_LOG=/var/log/caddy/elspeth-access.log \
PROBE_ERROR_RATE_WINDOW=60 \
PROBE_ERROR_RATE_THRESHOLD=0.01 \
ELSPETH_DEPLOY_ACTOR="{{ elspeth_deploy_actor | default('unknown-ci') }}" \
scripts/post-deploy-probe.sh "https://elspeth.example.com" \
    "{{ elspeth_previous_repo_version }}" \
    "{{ elspeth_repo_version }}"
# ELSPETH_DEPLOY_ACTOR is recorded into the audit-trail rollback
# attestation row (see "What Rollback Does To The Audit Trail").
# Populate elspeth_deploy_actor from CI metadata when the wrapper
# is rendered — e.g., GITHUB_ACTOR for GitHub Actions, GITLAB_USER_LOGIN
# for GitLab CI, BUILD_USER_ID for Jenkins. The lookup("env","USER")
# fallback the runbook previously used is empty under unattended
# rollback (probe-loop-driven), which is the case where attribution
# matters most.

# Staging wrapper — usable as a literal bash script. Caller exports
# PREVIOUS_SHA and FAILED_SHA before invoking. More endpoints, tighter
# error-rate threshold, and a lower fail-count so staging trips
# rollback faster than production.
ANSIBLE_VAULT_PASSWORD_FILE=/etc/ansible/vault.pass \
PROBE_ENDPOINTS="/api/health /api/sessions/_smoke /api/auth/_status /api/runs/_smoke /api/plugins/_list" \
PROBE_ERROR_RATE_LOG=/var/log/caddy/elspeth-access.log \
PROBE_ERROR_RATE_THRESHOLD=0.005 \
PROBE_FAIL_THRESHOLD=2 \
scripts/post-deploy-probe.sh "https://elspeth.foundryside.dev" \
    "$PREVIOUS_SHA" "$FAILED_SHA"
```

**Endpoint selection.** Choose endpoints that are cheap (sub-50ms
typical), idempotent (no side effects on repeated probing), and
representative of the *real* request path — both the framework layer
and the application layer. Authentication-gated endpoints can use
a dedicated `_smoke` route that returns a fixed JSON body to a
fixed bearer token from `ANSIBLE_VAULT_PASSWORD_FILE`-adjacent
storage; do *not* hardcode test tokens or check them into the repo.

**`_smoke` endpoints are application code, not deploy code.** If the
endpoints listed in `PROBE_ENDPOINTS` don't exist, add them in the
application before adding them here. A probe against a 404 is a
permanent rollback trigger, not a probe.

For container deployments, the equivalent is Container Apps revision
traffic-shifting: deploy the new revision with 0% traffic, probe the
per-revision FQDN, then either shift traffic to 100% on success or
deactivate the failed revision and leave the previous revision serving.
See [Revision Traffic-Shifting For Production Deploys](#revision-traffic-shifting-for-production-deploys)
above for the full playbook. Unlike the VM rollback (which restores DB
state from a SHA-tagged snapshot), container rollback is *traffic-only*:
the previous revision was never stopped, the new revision never received
real traffic, and external state (Key Vault references, external
database, mounted payload store) is unchanged.

## Risk Register Starter

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ubuntu 22.04 image lacks Python 3.12 | Playbook creates an unusable virtualenv or fails midway | Preflight Python version, use an approved Python repository, or bake Python 3.12 into the image |
| Front Door origin certificate does not match host header | Edge health probes fail; users see 502/503 | Decide `azure_vm_origin_host_name` and certificate automation before enabling `HttpsOnly`; route task is gated on the origin-TLS probe in the "Azure VM With Azure Front Door" section |
| Direct origin access bypasses Front Door WAF | Public VM skips WAF and edge controls | Combine `AzureFrontDoor.Backend` filtering with `X-Azure-FDID` validation |
| Container uses local SQLite or local payload files | Data disappears on restart or scale-out | Use external databases and mounted filesystem storage |
| Secret values appear in CI output | Credential exposure | Use Vault or Key Vault references; `no_log: true` only on the env-file template task; never on tasks whose body contains only Key Vault *references* |
| Front Door caching enabled on WebSocket routes | WebSocket upgrade fails | Set `disable_cache_configuration: true` on app routes |
| `elspeth_repo_version` set to a branch | Deploy is not reproducible; re-runs ship different code | "Refuse to deploy with an unpinned repo version" assert task; default is a SHA |
| `--forwarded-allow-ips` set to the socket path | `X-Forwarded-For` from Caddy is dropped; audit records wrong remote_addr | Systemd unit uses `--forwarded-allow-ips='*'` (trust established by UDS filesystem permissions) |
| Caddy cannot open the elspeth UDS | First deploy returns 502 on every request | "Add caddy user to socket-access group" task plus `Group={{ elspeth_socket_group }}` in the systemd unit |
| `--limit-max-requests` causes uvicorn to exit 0 mid-traffic | Periodic outage; `Restart=on-failure` does not catch a clean exit | `--limit-max-requests` removed; if re-introduced, switch to `Restart=always` |
| Multi-worker SQLite bypass via `WEB_CONCURRENCY` override | Next restart trips the startup guard and the service refuses to start | Systemd unit pins `Environment=WEB_CONCURRENCY=1`; verification asserts worker count |
| Schema-incompatible upgrade overwrites session DB | In-flight composer sessions lost; audit DB unrecoverable if conflated | "Snapshot session and audit databases before upgrade" task; Database Lifecycle section documents the operator-deletes policy for sessions only |
<!-- Two HSTS rows below (consolidated from a pre-L1 entry that said
"Caddy template ships max-age=300 by default with documented ramp" —
that mitigation is now the variable-driven mechanism captured in the
HSTS max-age and includeSubDomains/preload rows further down). -->
| Post-deploy regression undetected | Service is broken in production until human notices | Rollback Automation section ships a probe-loop that triggers playbook re-run with `elspeth_previous_repo_version` on N consecutive failures |
| Multi-host fleet builds drift between hosts | Two production hosts converge on materially different bundles | "Refuse per-host source-checkout build on multi-host plays" assert; opt-out via `elspeth_allow_per_host_build=true` for the rare deliberate case |
| Unverified SHA reaches production | Code that never passed CI gates ships | Default-required gate: `elspeth_ci_status_verification_required: true` is the default; omitting `elspeth_ci_status_url` while the gate is required refuses the deploy. Opt-out requires an explicit variable bump visible in Git history |
| Auto-rollback cannot unlock vault from unattended probe | Rollback never runs; broken deploy stays live until human returns | `post-deploy-probe.sh` requires `ANSIBLE_VAULT_PASSWORD_FILE`; `--ask-vault-pass` removed from the rollback path |
| Rollback restores the wrong DB snapshot when multiple deploys ran in one session | Known-good binary runs against known-bad on-disk state | Snapshot filename embeds the deploying SHA (`pre-deploy-of-<sha>-<ts>`); rollback role requires `elspeth_rollback_from_repo_version` and refuses to restore on `mtime` alone |
| Container deploy puts 100% traffic on a new revision before health check | Broken revision serves all production traffic until human notices | "Revision Traffic-Shifting For Production Deploys" deploys at 0% traffic, probes the per-revision FQDN, then either shifts to 100% on success or deactivates on failure |
| Container deploy probe hits the previous revision and returns false-positive green | New revision marked healthy without ever being exercised | Probe targets the *per-revision FQDN* (deterministic via `revisionSuffix`), not the app's stable FQDN; assert refuses to proceed without a per-revision FQDN |
| Container app created in `Single` revision mode | Traffic-shifting flow is structurally impossible — every new revision instantly takes 100% | Initial container creation sets `activeRevisionsMode: Multiple` and an explicit `traffic[]` block from first deploy |
| Image tag contains characters invalid for Container Apps revision suffix | Revision creation fails with cryptic error mid-deploy | Variables section documents the constraint (lowercase alphanumeric + hyphens, ≤64 chars); the `sha-<commit>` convention satisfies it |
| Post-deploy probe greenlights a deploy where `/api/health` is OK but real endpoints 500 | Broken release stays live; probe was theatre | `PROBE_ENDPOINTS` env var lets each environment add representative read-path and write-path endpoints; optional `PROBE_ERROR_RATE_LOG` gate catches regressions visible in real traffic |
| Error-rate gate treats "no traffic in window" as healthy | Late-night or low-traffic deploy silently bypasses the gate | `check_error_rate` distinguishes inconclusive (return 2) from healthy (return 0); inconclusive does not reset the consecutive-failure counter |
| `PROBE_ENDPOINTS` listed before the application exposes them | Permanent 404 → permanent rollback trigger | Runbook prescribes: `_smoke` endpoints are application code, add them in the application before listing in the probe wrapper |
| Caddy access log retains `Authorization`/`Cookie` headers and query strings | PII / session-token exposure in operational log | `log { format filter ... }` template redacts `Authorization`, `Cookie`, `Set-Cookie`, and the query-string portion of the URI; `mode 0640` restricts on-host readers |
| Off-host log shipper inherits unredacted fields | PII leaked to central log store at much greater scale | Access-Log Hygiene section: the shipper inherits on-host redaction discipline; redact at source, not at destination |
| `elspeth` service account added to `caddy` group for log access | An elspeth compromise gains read access to all Caddy access logs | Runbook explicitly says NOT to add `elspeth` to the Caddy group; elspeth does not need Caddy log access |
| HSTS `includeSubDomains` or `preload` enabled without explicit audit | One-way pin extends to every subdomain or to the browser preload list; rollback requires waiting out cached max-age or a multi-week preload-removal cycle | `elspeth_hsts_include_subdomains` and `elspeth_hsts_preload` are explicit booleans defaulting to `false`; both require a discrete variable change visible in Git history |
| HSTS max-age bumped to 1 year before TLS posture is proven | Botched cert during the year-long pin bricks the domain for clients that cached the policy | `elspeth_hsts_max_age` variable with documented three-stage ramp (300 → 86400 → 31536000); ramp progression is an auditable variable change, not a Caddyfile diff |
| Node.js missing or too old crashes the deploy late with an opaque message | Operator wastes time triaging "Node too old" without knowing which lever to pull | Two-step assert: first checks Node is present and names the three resolution paths (apt source, NodeSource repo, image bake); second requires Node 24.x and repeats the resolution paths with the observed version |
| NodeSource apt key fetched without checksum pin | Supply-chain compromise of nodesource.com replaces the key and silently changes the package source | NodeSource get_url task pins the key by SHA256 via `vault_nodesource_apt_key_sha256`; runbook documents how to obtain and rotate the value |
| Container app discovery treats auth/permission failures as "app does not exist" | One-shot bootstrap task tries to create over the running app; deploy ends in a half-applied confused state | "Distinguish 404 from auth / permission / network failures" assert fires loudly with operator guidance on the typical fix (Container Apps Reader role) |
| Traffic-shift PATCH abbreviates the container spec | ARM PATCH-merge replaces `containers[]` wholesale and drops env vars, secret refs, probes; new revision starts up without `SECRET_KEY`, fails health probe, gets deactivated, and operator chases the wrong root cause | Container spec extracted into `templates/container-app-template.yml.j2`, loaded once into `elspeth_container_template` fact, consumed by both bootstrap PUT and traffic-shift PATCH; runbook calls this REQUIRED, not stylistic |
| Probe script resets `consecutive_failures` on inconclusive iterations | Late-night low-traffic deploy with the error-rate gate enabled silently bypasses the consecutive-failure window | Probe loop tracks `iteration_failed` and `iteration_inconclusive` separately; counter only resets on a clean (neither failed nor inconclusive) pass |
| Audit-row-loss on VM rollback is undocumented | Auditor querying the rollback window sees no rows but no signal that any were discarded | Database Lifecycle now has "What Rollback Does To The Audit Trail" section prescribing pre-rollback boundary capture, post-rollback attestation row, and off-host shipping of the pre-rollback `audit.db` |
| Raw `cp` of SQLite under WAL produces a torn snapshot | Snapshot is the rollback target; if it's torn (committed-but-not-checkpointed rows missing), rollback restores corrupt state | Snapshot task uses `sqlite3 ".backup"` (Online Backup API serializes against writers); restore task stops the service, copies the snapshot, removes stale `.db-wal`/`.db-shm`, restarts |
| Stale WAL replays over a restored DB | Sidecars (`.db-wal`/`.db-shm`) from the failing version are replayed on first open, re-corrupting the restored file with the failing version's writes | Rollback role removes both sidecars after the restore copy, before service restart |
| Caddy receives app-secret/data group membership for UDS access | Operational reverse proxy can read `/etc/elspeth/elspeth-web.env` or app data; an edge compromise crosses into app authentication and provider-key custody | `elspeth_socket_group` is dedicated to `/run/elspeth/uvicorn.sock`; Caddy is not added to `elspeth_group`, app data remains 0700, and the env directory/file are root-only |
| Caddy ACME HTTP-01 fails because port 80 is closed | First Caddy reload floods logs with ACME failures; operator misreads this as "playbook broken"; cert never issues so HTTPS never works | "Caddy TLS Provisioning Modes" section names four mutually-exclusive modes (HTTP-01, DNS-01, `tls internal`, pre-provisioned) and assigns each to a deployment posture; preflight `assert` refuses to render the site template when zero or more than one mode is selected |
| Schema-change deploy proceeds without operator acknowledgement | Service starts against an incompatible `sessions.db`, crashes on first session-DB write, or silently corrupts in-flight rows | Sidecar deploy-SHA marker `.last-deploy-sha` records the SHA that last touched the DB; preflight refuses to proceed when SHA differs unless the operator sets `elspeth_acknowledge_schema_compatible=true` (code-only) or `elspeth_force_session_db_delete=true` (schema-change, drop in-flight sessions) |
| Local socket probe passes but public edge is broken | Play reports green; users see 502/503; "verification" was theatre | "Public-edge verification" block now uses `ansible.builtin.uri` delegated to the Ansible controller (not the VM), exercising the real DNS + TLS + ingress path with `retries: 6 delay: 10` and `validate_certs` gated on TLS mode; the play fails fast on any non-200 |
| Verification probes run against the pre-restart binary because handlers fire at end-of-play | Marker rotation locks in the new SHA as verified while the actual restart still hasn't happened with the new code; broken deploy ships labeled "verified" | `ansible.builtin.meta: flush_handlers` inserted immediately before the verification block forces `restart elspeth web` + `restart caddy` to fire mid-play, so verification exercises the new binary + new ingress config |

## Operational Notes

- Never print `/etc/elspeth/elspeth-web.env` or decrypted Vault values in CI
  logs.
- Restart `elspeth-web.service` after Python code, dependency, environment, or
  systemd changes.
- A frontend-only deploy still needs `npm run build`; the running FastAPI app
  serves the rebuilt files from disk.
- For Azure Front Door, keep caching disabled for app routes that may carry
  WebSocket upgrade traffic.
- For public Azure VM origins, use the `AzureFrontDoor.Backend` service tag and
  validate `X-Azure-FDID`; either control alone is incomplete.
- For containers, do not rely on container-local SQLite for production. Use a
  managed database or another durable external store.
- Keep image tags immutable in container deployments. Avoid `latest` outside
  development.

## References

- Azure Front Door WebSocket support: <https://learn.microsoft.com/azure/frontdoor/standard-premium/websocket>
- Azure Front Door origin security: <https://learn.microsoft.com/azure/frontdoor/origin-security>
- Ansible Azure collection: <https://docs.ansible.com/ansible/latest/collections/azure/azcollection/index.html>
- Azure Container Apps ingress: <https://learn.microsoft.com/azure/container-apps/ingress-overview>
- Azure Container Apps managed identity image pull: <https://learn.microsoft.com/azure/container-apps/managed-identity-image-pull>
