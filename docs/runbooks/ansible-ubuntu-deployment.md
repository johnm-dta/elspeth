# Ansible Ubuntu Deployment Guide

This guide describes how to automate ELSPETH web deployments with Ansible on
Ubuntu 24.04 and Ubuntu 22.04. It covers three deployment contexts:

- Non-cloud Ubuntu host running the source-checkout web service behind Caddy.
- Azure Ubuntu VM running the same service, published through Azure Front Door.
- Azure container deployment using a built container image and Azure Container
  Apps or a comparable container host.

The examples are templates for an operations repository or a future
`deploy/ansible/` tree. Do not paste production secrets into inventory files;
use Ansible Vault, an external secret manager, or Azure Key Vault.

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
- Frontend build dependency: Node.js 20.19+ or a newer compatible LTS release
- Required non-local setting: `ELSPETH_WEB__SECRET_KEY`
- Default web state under `ELSPETH_WEB__DATA_DIR`, with overrides:
  - `ELSPETH_WEB__SESSION_DB_URL`
  - `ELSPETH_WEB__LANDSCAPE_URL`
  - `ELSPETH_WEB__PAYLOAD_STORE_PATH`

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
   - `python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
   - `python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards`
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

The "Pre-deploy CI gates pass" step above is operator-asserted unless
you wire a CI-status check into the playbook. Two variables turn the
assertion into a mechanical gate:

```yaml
# group_vars/elspeth_web.yml — example for GitHub status checks
elspeth_ci_status_url: "https://api.github.com/repos/your-org/elspeth/commits/{{ elspeth_repo_version }}/status"
elspeth_ci_status_headers:
  Authorization: "Bearer {{ vault_github_ci_token }}"
  Accept: "application/vnd.github+json"
elspeth_ci_status_success_path: "state"      # JSON field to read
elspeth_ci_status_success_value: "success"   # value that means "green"
```

Preflight tasks:

```yaml
- name: Verify CI status for elspeth_repo_version
  ansible.builtin.uri:
    url: "{{ elspeth_ci_status_url }}"
    method: GET
    headers: "{{ elspeth_ci_status_headers | default({}) }}"
    return_content: true
    status_code: 200
  register: elspeth_ci_status
  when:
    - elspeth_ci_status_url is defined
    - elspeth_ci_status_url | length > 0

- name: Refuse deploy if CI status is not success
  ansible.builtin.assert:
    that:
      - (elspeth_ci_status.json[elspeth_ci_status_success_path | default('state')]
          | string) == (elspeth_ci_status_success_value | default('success'))
    fail_msg: >-
      CI status for {{ elspeth_repo_version }} is not
      "{{ elspeth_ci_status_success_value | default('success') }}".
      Refusing to deploy an unverified revision.
  when:
    - elspeth_ci_status_url is defined
    - elspeth_ci_status_url | length > 0

- name: Warn when CI status verification is unconfigured
  ansible.builtin.debug:
    msg: >-
      WARNING: elspeth_ci_status_url is not set. This deploy will proceed
      without machine-verifying that {{ elspeth_repo_version }} passed
      pre-deploy CI gates. See "CI Status Verification" in the runbook.
  when:
    - (elspeth_ci_status_url is not defined) or
      (elspeth_ci_status_url | length == 0)
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
# How Node.js 20.19+ gets onto the host. Pick ONE:
#   - elspeth_node_packages populated with apt-installable names
#     (the apt source must ship Node 20.19+; Ubuntu 22.04 defaults do
#     not). Leave elspeth_install_nodesource_repo at false.
#   - elspeth_install_nodesource_repo: true
#     adds the official NodeSource Node 20.x repo before package install.
#     Sets elspeth_node_packages to ['nodejs'] if it's still empty.
#   - Bake Node.js into the VM image and leave both empty; the role's
#     verify-Node-version assert will accept any pre-installed Node
#     20.19+ at elspeth_node_binary.
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
```

Generate the web secret key on a trusted workstation:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Ubuntu 24.04 And 22.04 Package Handling

Use the same role for both Ubuntu releases, but make Python provisioning
explicit. The frontend lockfile currently includes packages that require
Node.js 20.19+ or a newer compatible LTS release, so do not rely on the Ubuntu
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

- name: Provision NodeSource Node 20.x apt repo (optional)
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
      # The key is fetched at provision time. Pinning the key by SHA256
      # in a real production posture (checksum: sha256:...) is strongly
      # recommended; the unpinned form above is the runbook example.

    - name: Configure NodeSource apt source for Node 20.x
      ansible.builtin.copy:
        dest: /etc/apt/sources.list.d/nodesource.list
        owner: root
        group: root
        mode: "0644"
        content: |
          deb [signed-by=/etc/apt/keyrings/nodesource.asc] https://deb.nodesource.com/node_20.x nodistro main

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
      build needs Node.js 20.19+. Resolve by one of:
        (a) set elspeth_node_packages to a list installable by apt
            (e.g. ['nodejs'] when your apt source ships Node 20.19+),
        (b) enable elspeth_install_nodesource_repo=true to add the
            official NodeSource apt repo for Node 20.x (Ubuntu 22.04
            does NOT ship Node 20.19+ in the default repos),
        (c) bake Node.js into the VM image and set elspeth_node_binary
            to its install path.

- name: Fail if Node.js is too old for the frontend lockfile
  ansible.builtin.assert:
    that:
      - elspeth_node_version.stdout is match('v(20\\.(1[9]|[2-9][0-9])|2[2-9]\\.|[3-9][0-9]\\.)')
    fail_msg: >-
      Node.js at {{ elspeth_node_binary }} is
      {{ elspeth_node_version.stdout }}, but the frontend lockfile
      requires Node.js 20.19+ or a newer compatible LTS. The default
      Ubuntu 22.04 nodejs package is too old; the default Ubuntu
      24.04 nodejs package depends on the apt mirror's snapshot.
      Resolve by one of:
        (a) point elspeth_node_packages at a newer apt source,
        (b) set elspeth_install_nodesource_repo=true to provision the
            NodeSource Node 20.x repo,
        (c) re-bake the VM image with Node.js 20.19+ pre-installed.
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

- name: Create ELSPETH user
  ansible.builtin.user:
    name: "{{ elspeth_user }}"
    group: "{{ elspeth_group }}"
    home: "{{ elspeth_app_dir }}"
    shell: /usr/sbin/nologin
    system: true

- name: Create data directories
  ansible.builtin.file:
    path: "{{ item }}"
    state: directory
    owner: "{{ elspeth_user }}"
    group: "{{ elspeth_group }}"
    mode: "0750"
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

- name: Snapshot session and audit databases before upgrade
  ansible.builtin.copy:
    src: "{{ item }}"
    dest: "{{ item }}.pre-deploy-of-{{ elspeth_repo_version }}-{{ ansible_date_time.iso8601_basic_short }}"
    remote_src: true
    owner: "{{ elspeth_user }}"
    group: "{{ elspeth_group }}"
    mode: "0640"
  loop:
    - "{{ elspeth_data_dir }}/sessions.db"
    - "{{ elspeth_data_dir }}/runs/audit.db"
  failed_when: false                # first deploy: files do not exist yet
  register: elspeth_db_snapshot
  changed_when: elspeth_db_snapshot is not skipped
  # The snapshot filename embeds the SHA we are ABOUT TO DEPLOY, not the
  # SHA currently live. To roll back FROM a failing SHA Y back TO the
  # prior X, the rollback role selects the snapshot named
  # "pre-deploy-of-Y-*" — i.e., it restores the state that existed
  # *before* Y started writing to the DB. See "Rollback Automation".

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

- name: Add caddy user to elspeth group (UDS access)
  ansible.builtin.user:
    name: caddy
    groups: "{{ elspeth_group }}"
    append: true
  notify: restart caddy
  # The systemd unit creates the UDS at /run/elspeth/uvicorn.sock with
  # UMask=0007, so the socket is mode 0660 owned by elspeth:elspeth.
  # Debian's Caddy package runs as user `caddy`; without group membership
  # the reverse_proxy returns 502s on every request. The corresponding
  # handler:
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
ELSPETH_FINGERPRINT_KEY={{ vault_fingerprint_key }}
OPENROUTER_API_KEY={{ vault_openrouter_api_key }}
AZURE_OPENAI_API_KEY={{ vault_azure_openai_api_key }}
AZURE_OPENAI_ENDPOINT={{ vault_azure_openai_endpoint }}
```

Install it with:

```yaml
- name: Create environment directory
  ansible.builtin.file:
    path: /etc/elspeth
    state: directory
    owner: root
    group: "{{ elspeth_group }}"
    mode: "0750"

- name: Install web environment file
  ansible.builtin.template:
    src: elspeth-web.env.j2
    dest: /etc/elspeth/elspeth-web.env
    owner: root
    group: "{{ elspeth_group }}"
    mode: "0640"
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
Group={{ elspeth_group }}
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
# socket (mode 0660, elspeth group); the only peer that can connect is
# Caddy, so '*' is the correct value. Without this, X-Forwarded-For and
# X-Forwarded-Proto from Caddy are dropped and the audit trail records
# the wrong remote_addr.
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

Container Apps **revision suffix constraint**: `elspeth_image_tag` is used
as the revision suffix, which must be lowercase alphanumeric or hyphens
and ≤ 64 characters. The `sha-<commit>` convention satisfies this.
Image tags containing `.`, `_`, or uppercase characters will fail
revision creation; choose the tag scheme accordingly.

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
  failed_when: false                # 404 is an expected first-deploy state

- name: Set container-app existence fact
  ansible.builtin.set_fact:
    elspeth_container_app_exists: >-
      {{ (elspeth_container_app_lookup.response | default([]) | length) > 0 }}

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
        template:
          # revisionSuffix makes revision names deterministic
          # (<appName>--<suffix>). Without it Container Apps generates
          # random suffixes and the traffic-shifting tasks below cannot
          # construct the per-revision FQDN deterministically.
          revisionSuffix: "{{ elspeth_image_tag }}"
          containers:
            - name: elspeth-web
              image: "{{ azure_acr_login_server }}/elspeth:{{ elspeth_image_tag }}"
              env:
                - name: ELSPETH_WEB__HOST
                  value: 0.0.0.0
                - name: ELSPETH_WEB__SECRET_KEY
                  secretRef: elspeth-web-secret-key
                - name: ELSPETH_FINGERPRINT_KEY
                  secretRef: elspeth-fingerprint-key
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
              probes:
                - type: Liveness
                  httpGet:
                    path: /api/health
                    port: 8451
                  initialDelaySeconds: 30
                  periodSeconds: 30
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
        template:
          revisionSuffix: "{{ elspeth_image_tag }}"
          containers:
            - name: elspeth-web
              image: "{{ azure_acr_login_server }}/elspeth:{{ elspeth_image_tag }}"
              # env/secrets/probes block is identical to the initial
              # creation task above; omitted here for brevity. Templating
              # tip: extract the container spec into a single Jinja
              # include and reference it from both tasks so they cannot
              # drift apart.
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

Public checks:

```bash
curl -fsS https://elspeth.example.com/api/health
```

Azure Front Door checks:

```bash
curl -fsS https://<front-door-endpoint>.azurefd.net/api/health
curl -fsS https://elspeth.example.com/api/health
```

Container checks:

```bash
curl -fsS "https://${AZURE_CONTAINER_APP_FQDN}/api/health"
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
| Web session DB | `ELSPETH_WEB__SESSION_DB_URL` | `/var/lib/elspeth/sessions.db` | **Operator-deletes on schema change.** No Alembic migrations. Deleting drops in-flight composer sessions. |
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
2. **For a schema-incompatible upgrade**, after a successful deploy and
   verification, delete the *session* DB:
   ```bash
   sudo -u elspeth rm /var/lib/elspeth/sessions.db
   sudo systemctl restart elspeth-web.service
   ```
   Do **not** delete `audit.db`. The landscape schema is append-only at
   the row level; if a code change requires a landscape schema migration,
   that change is currently blocked from shipping (it is a future-release
   gate, not an operator decision).
3. **On rollback**, restore both pre-deploy snapshots before re-running the
   previous-version playbook. See [Rollback Automation](#rollback-automation).
4. **Snapshot retention**: keep at least the last three pre-deploy snapshots
   of `audit.db` off-host (S3, Azure Blob, or backup share). The on-host
   snapshots are an emergency local rollback aid, not a backup.

For container deployments, mount the audit DB on durable storage external
to the container. Container-local SQLite for audit data is unsafe regardless
of whether the schema is stable.

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
    mode: "0640"
  loop:
    - sessions.db
    - runs/audit.db
  loop_control:
    loop_var: dest_relative
  when: elspeth_restore_db_from_snapshot | default(false) | bool
  notify: restart elspeth web
  # If the same SHA was deployed twice (rare — usually means the first
  # attempt failed before the snapshot task ran on the second attempt),
  # we still pick the newest snapshot for that SHA, which is the
  # closest-to-failed-state baseline we have. The SHA filter prevents
  # crossing over into a different deploy's snapshots.
```

**Automatic rollback on post-deploy probe failure.** After cutover, run a
probe loop and trigger the rollback playbook if probes fail N
consecutive times. The probe loop:

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
        status=$(curl -s -o /dev/null -w '%{http_code}' "${BASE_URL%/}${path}" || echo "000")
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
    if ! probe_endpoints; then
        iteration_failed=1
    fi

    # Only run the error-rate gate if endpoint probes passed. If they
    # already failed there is no point checking traffic; the iteration
    # is already counted.
    if [ "$iteration_failed" -eq 0 ]; then
        check_error_rate
        case $? in
            0) ;;                       # healthy
            1) iteration_failed=1 ;;    # exceeded threshold
            2) ;;                       # inconclusive — leave counter as-is
        esac
    fi

    if [ "$iteration_failed" -eq 1 ]; then
        consecutive_failures=$(( consecutive_failures + 1 ))
        if [ "$consecutive_failures" -ge "$FAIL_THRESHOLD" ]; then
            rollback
        fi
    else
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
should be set per-environment in your deploy wrapper:

```bash
# Example: production wrapper
ANSIBLE_VAULT_PASSWORD_FILE=/etc/ansible/vault.pass \
PROBE_ENDPOINTS="/api/health /api/sessions/_smoke /api/auth/_status" \
PROBE_ERROR_RATE_LOG=/var/log/caddy/elspeth-access.log \
PROBE_ERROR_RATE_WINDOW=60 \
PROBE_ERROR_RATE_THRESHOLD=0.01 \
scripts/post-deploy-probe.sh "https://elspeth.example.com" \
    "{{ elspeth_previous_repo_version }}" \
    "{{ elspeth_repo_version }}"

# Example: staging wrapper — more endpoints, tighter threshold
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
| Caddy cannot open the elspeth UDS | First deploy returns 502 on every request | "Add caddy user to elspeth group" task in the VM role |
| `--limit-max-requests` causes uvicorn to exit 0 mid-traffic | Periodic outage; `Restart=on-failure` does not catch a clean exit | `--limit-max-requests` removed; if re-introduced, switch to `Restart=always` |
| Multi-worker SQLite bypass via `WEB_CONCURRENCY` override | Next restart trips the startup guard and the service refuses to start | Systemd unit pins `Environment=WEB_CONCURRENCY=1`; verification asserts worker count |
| Schema-incompatible upgrade overwrites session DB | In-flight composer sessions lost; audit DB unrecoverable if conflated | "Snapshot session and audit databases before upgrade" task; Database Lifecycle section documents the operator-deletes policy for sessions only |
<!-- Two HSTS rows below (consolidated from a pre-L1 entry that said
"Caddy template ships max-age=300 by default with documented ramp" —
that mitigation is now the variable-driven mechanism captured in the
HSTS max-age and includeSubDomains/preload rows further down). -->
| Post-deploy regression undetected | Service is broken in production until human notices | Rollback Automation section ships a probe-loop that triggers playbook re-run with `elspeth_previous_repo_version` on N consecutive failures |
| Multi-host fleet builds drift between hosts | Two production hosts converge on materially different bundles | "Refuse per-host source-checkout build on multi-host plays" assert; opt-out via `elspeth_allow_per_host_build=true` for the rare deliberate case |
| Unverified SHA reaches production | Code that never passed CI gates ships | "Verify CI status for elspeth_repo_version" preflight task; warning emitted when `elspeth_ci_status_url` is unset |
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
| Node.js missing or too old crashes the deploy late with an opaque message | Operator wastes time triaging "Node too old" without knowing which lever to pull | Two-step assert: first checks Node is present and names the three resolution paths (apt source, NodeSource repo, image bake); second checks the version is ≥ 20.19 and repeats the resolution paths with the observed version |
| NodeSource apt key fetched without checksum pin | Supply-chain compromise of nodesource.com replaces the key and silently changes the package source | Runbook documents the unpinned form as the example but explicitly notes "pinning the key by SHA256 in a real production posture is strongly recommended" |

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
