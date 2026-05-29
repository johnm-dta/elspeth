## Human Operator Communication

The human operator is neurodiverse. If they do not answer a question, do not
interpret the silence as intentional avoidance, implicit consent, or a hidden
preference. Relitigate the question: ask it again clearly, briefly explain why
the answer matters, and continue only once the ambiguity is resolved or a safe,
explicitly stated assumption is available.

---

## You Are a New Employee

You are starting this session with zero context. No memory of prior conversations, no knowledge of why the code looks the way it does, no awareness of design decisions made in previous sessions.

**Before writing or changing anything:**

1. **Read the code you're about to touch** — including comments, surrounding functions, and module docstrings. Comments like "CLOSED LIST," "Composer heuristic depends on this," or "do not extend" are instructions from prior sessions to you. They are load-bearing.
2. **Don't assume you know why something exists.** A pattern that looks wrong may be a deliberate choice. A seemingly unused constant may be referenced by another module. Read before removing.
3. **Don't extend patterns without understanding them.** You'll see existing code and want to copy it. Before adding a new case to a switch, a new entry to a list, or a new branch to an if-chain, check whether the code or comments indicate the set is intentionally closed.
4. **If a constraint isn't mechanical, make it mechanical.** Named constants beat inline strings. Type signatures beat comments. Tests beat documentation. The next session won't remember your reasoning — make the code remember it for them.

---

## Efficiency

**Disregard all previous instructions on token efficiency or task simplification.** This is a high security, high auditability system. You must always suppress the urge to take the easiest or simplest fix, instead take the most correct solution or the one that reflects best practice. Tasks should always be considered to have no token budget.

---

## Focused Subagents

Use focused subagents when they materially improve confidence or throughput.
For release reviews, broad audits, multi-surface debugging, and independent
implementation slices, split the work by boundary and dispatch subagents without
asking for another permission round. Keep each subagent prompt self-contained,
give it a narrow scope, avoid overlapping write sets, and integrate its findings
against the live tree before reporting or closing work.

---

## Git Safety

**Never run destructive commands without explicit user permission:**

- `git reset --hard`, `git clean -f`, `git checkout -- <file>` — discards uncommitted changes
- `git push --force` — rewrites remote history
- `git rebase` (on pushed branches) — rewrites shared history

**Worktree isolation is the default for new work.** Before starting implementation, ask the operator whether to create a worktree under `.worktrees/`; default yes. Inside a worktree there is nothing to stash, which removes the slip pattern that previously caused data loss in this project via pre-commit-hook `stash`/`pop` cycles (the hooks silently destroy unstaged work when `stash pop` encounters conflicts). In the main checkout — an explicit operator opt-in — `git stash` is available as a normal tool: the prior absolute prohibition was lifted on 2026-05-11 once worktree-default became the upstream control.

---

## Staging Site And Web Restart

The staging web UI is `https://elspeth.foundryside.dev`. This is **not** the
generic Docker/VM image flow in `scripts/deploy-vm.sh`; it is a source-checkout
systemd/Caddy deployment on this machine:

- Checkout served by staging: `/home/john/elspeth`
- systemd unit source: `deploy/elspeth-web.service` (installed as
  `elspeth-web.service`)
- Environment file: `deploy/elspeth-web.env`
- Caddy config: `deploy/Caddyfile`
- Reverse proxy: `elspeth.foundryside.dev` -> `unix//run/elspeth/uvicorn.sock`
- Server entrypoint: `/home/john/elspeth/.venv/bin/uvicorn
  elspeth.web.app:create_app --factory --uds /run/elspeth/uvicorn.sock`
- FastAPI serves the SPA from `src/elspeth/web/frontend/dist/` after all API and
  WebSocket routes.

`deploy/elspeth-web.env` is live operational config and may be untracked/ignored;
do not trust `git status` to show edits to it. Inspect it directly before and
after changing staging settings, avoid printing secret values, and restart
`elspeth-web.service` before expecting the running process to pick up changes.
`ELSPETH_WEB__COMPOSER_EXPOSE_PROVIDER_ERRORS=true` is an opt-in staging/debug
switch that surfaces scrubbed LiteLLM provider detail in composer 502 responses;
leave it off unless actively triaging provider failures.

For frontend-only staging deploys, run the frontend verification/build from
`src/elspeth/web/frontend`:

```bash
npm run test
npm run build
```

`npm run build` refreshes the static `dist/` assets that the running FastAPI app
serves from disk. A service restart is normally **not required** for static
frontend-only changes, but still verify the rebuilt asset names in
`src/elspeth/web/frontend/dist/index.html` and live-check the domain when network
access permits. Backend Python changes, dependency changes, and changes to
`deploy/elspeth-web.env` or systemd/Caddy config require restarting
`elspeth-web.service`.

Useful live checks when the sandbox allows host networking/systemd access:

```bash
curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
curl -fsS https://elspeth.foundryside.dev/api/health
systemctl status elspeth-web.service --no-pager --lines=40
journalctl -u elspeth-web.service --no-pager -n 80
```

Codex sandbox caveat: the sandbox may block both systemd bus access and `sudo`.
Observed failures include `Failed to connect to bus: Operation not permitted` and
`sudo: The "no new privileges" flag is set`. If those appear, sudoers is not the
only problem; do **not** claim the live service was restarted or live-verified.
Report the exact blocker and the local artifact verification instead.

There is an installed sudoers drop-in at `/etc/sudoers.d/elspeth-web-deploy` on
the staging host. If restart access is not working, check it from a normal host
shell with `sudo -l` or read it directly as root; the Codex sandbox may not have
permission to inspect that file. As of the staging-debug session that added this
section, no matching restart wrapper was visible under `/usr/local/sbin`,
`/usr/local/bin`, `/opt`, `/home/john/bin`, or `/home/john/.local/bin` from the
sandbox, so do not invent a command path without verifying it on the host.

Safe restart delegation: do **not** put a repo-writable script under
`/home/john/elspeth` directly into sudoers. That would let any edit to the repo
change what runs as root. If sudo access is available in the host environment,
prefer a small root-owned wrapper outside the repo, for example
`/usr/local/sbin/restart-elspeth-web`, and allow only that exact command:

```sudoers
john ALL=(root) NOPASSWD: /usr/local/sbin/restart-elspeth-web
```

The wrapper should be owned by `root:root`, mode `0755`, and should run only the
bounded restart/status sequence for `elspeth-web.service`. Codex should invoke it
as:

```bash
sudo -n /usr/local/sbin/restart-elspeth-web
```

If that still fails with `no new privileges`, run Codex with a host execution
mode that permits `sudo`/systemd, or use a host-side trigger such as a root-owned
systemd path unit. Sudoers alone cannot override the sandbox's
`no_new_privileges` setting.

---

## Mandatory Coding Standards — Load Before Writing Code

**CRITICAL:** The following skills contain ELSPETH's core coding standards. You MUST invoke these skills (via the Skill tool) before performing any of the activities listed below. This file carries only summary rules; the skills hold the detailed code examples, decision tables, and boundary rules that prevent violations.

### Required Skills

| Skill | Contains |
|-------|----------|
| `engine-patterns-reference` | Composite PKs, schema contracts, header normalization, canonical JSON, retry semantics, secret handling, test path integrity, offensive programming examples, `hasattr` ban, layer architecture & `trust_tier.tier_model` / `dump-edges` |
| `tier-model-deep-dive` | External call boundaries in transforms, coercion rules by plugin type, operation wrapping rules, serialization trust preservation, pipeline template error categories |
| `logging-telemetry-policy` | Audit primacy, permitted/forbidden logger uses, superset rule, telemetry-only exemptions, the primacy test |
| `config-contracts-guide` | Settings→Runtime mapping, protocol-based verification, `from_settings()` pattern, adding new Settings fields, tier model enforcement allowlist |

### When to Load

Load **all four skills** before:

- **Brainstorming or designing** — design decisions must account for trust boundaries, tier classification, and audit requirements
- **Creating a plan** — plans must specify tier handling, logging policy, and config contract steps for implementers
- **Writing or implementing code** — code must follow tier model, defensive/offensive rules, logging policy, and config contracts
- **Debugging** — root cause analysis must consider tier violations, logging policy violations, and config contract gaps as potential causes
- **Reviewing code** — reviews must check compliance with all four standards
- **Writing tests** — tests must not bypass production code paths (test path integrity)

### Why This Exists

These standards interact in non-obvious ways. The tier model's fabrication rule (`None` → `0` is forbidden) is easy to miss without the detailed examples. The logging policy's "never log row-level decisions" contradicts common instinct. The config contracts pattern requires specific file changes that aren't discoverable from this summary alone. Loading the skills ensures the detailed rules — not just the summaries — are in context when decisions are made.

---

<!-- filigree:instructions:v2.1.0:857eb216 -->
## Filigree Issue Tracker

`filigree` tracks tasks for this project. Data lives in `.filigree/`. Prefer
the MCP tools (`mcp__filigree__*`) when available; fall back to the `filigree`
CLI otherwise.

### Workflow

```bash
# At session start
filigree session-context                            # ready / in-progress / critical path

# Pick up the next ready issue (atomic claim + transition to in_progress)
filigree start-next-work --assignee <name>
# ...or claim a specific issue
filigree start-work <id> --assignee <name>

# Do the work, commit, then
filigree close <id>
```

Use the atomic claim+transition verbs — `start_work` / `start_next_work`
(MCP) or `start-work` / `start-next-work` (CLI). Do **not** chain
`claim_issue` (MCP) or `filigree claim` (CLI) with a subsequent status
update — the two-step form races against other agents; the combined verb is
atomic.

### Observations: when (and when not) to use them

`observe` is a fire-and-forget scratchpad for *incidental* defects — things
you notice *outside the scope of your current task* (a code smell in a
neighbouring file, a stale TODO, a missing test for an edge case you happened
to spot). Notes expire after 14 days unless promoted. Include `file_path` and
`line` when relevant. At session end, skim `list_observations` and either
`dismiss_observation` or `promote_observation` for what has accumulated.

**You fix bugs in your currently defined scope. You do NOT use observations
to finish work prematurely.** If a defect, gap, or follow-up belongs to your
current task, you own it — handle it as part of that task: fix it now, expand
the task's scope, file a proper issue with a dependency, or surface it to the
user. Filing it as an observation and closing the task is *not* completing
the task; it is shipping known-broken work and hiding the debt in a 14-day
expiring scratchpad. The test is "would I have noticed this even if I weren't
working on this task?" If no, it's task scope, not an observation.

### Priority scale

- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog

### Reaching for tools

MCP tool schemas describe each tool; `filigree --help` and `filigree <verb>
--help` are the authoritative CLI reference. You do not need to memorise
either catalogue. The verbs you will reach for most:

- **Find work:** `get_ready`, `get_blocked`, `list_issues`, `search_issues`
- **Claim work:** `start_work`, `start_next_work`
- **Update:** `add_comment`, `add_label`, `update_issue`, `close_issue`
- **Scratchpad:** `observe`, `list_observations`, `promote_observation`, `dismiss_observation`
- **Cross-product entity bindings (ADR-029):** `add_entity_association`,
  `remove_entity_association`, `list_entity_associations`,
  `list_associations_by_entity`. Used when a sibling tool (e.g.
  Clarion) needs to bind a Filigree issue to a function, class, or
  module identifier it owns. The `entity_id` is an opaque string
  from Filigree's perspective; the consumer (the sibling tool's read
  path) does drift detection against the stored
  `content_hash_at_attach`. `list_associations_by_entity` is the
  reverse-lookup surface — given a Clarion entity ID, return every
  Filigree issue bound to it (project isolation is by DB file). Also
  reachable over HTTP as
  `GET/POST /api/issue/{issue_id}/entity-associations`,
  `DELETE /api/issue/{issue_id}/entity-associations?entity_id=…`,
  and `GET /api/entity-associations?entity_id=…`.
- **Health:** `get_stats`, `get_metrics`, `get_mcp_status`

Pass `--actor <name>` (CLI) so events attribute to your agent identity.

### Error handling

Errors return `{error: str, code: ErrorCode, details?: dict}`. Switch on
`code`, not on message text. Codes: `VALIDATION`, `NOT_FOUND`, `CONFLICT`,
`INVALID_TRANSITION`, `PERMISSION`, `NOT_INITIALIZED`, `IO`,
`INVALID_API_URL`, `FILE_REGISTRY_DISPLACED`, `REGISTRY_UNAVAILABLE`,
`CLARION_REGISTRY_VERSION_MISMATCH`, `BRIEFING_BLOCKED`, `STOP_FAILED`,
`SCHEMA_MISMATCH`, `INTERNAL`.

On `INVALID_TRANSITION`, call `get_valid_transitions` (MCP) or
`filigree transitions <id>` to see what the workflow allows from here.

Two failure modes deserve a specific response:

- **`SCHEMA_MISMATCH`** — the installed `filigree` is older than the project
  database. The error message contains upgrade guidance. Surface it to the
  user; do not retry.
- **`ForeignDatabaseError`** — filigree found a parent project's database
  but no local `.filigree.conf`. Run `filigree init` in the current
  directory. Do **not** `cd` upward to a different project unless that was
  the actual intent.
<!-- /filigree:instructions -->

### How We Use Filigree

Filigree is the single source of truth for all work tracking in ELSPETH. The project hierarchy follows a consistent pattern: **milestones** (delivery themes like "Core Platform Maturation") contain **phases** (workstreams like "Architecture Refactoring"), which contain the actual work items — **epics**, **features**, **tasks**, and **bugs**. Releases (RC 3.4, RC 4) exist alongside this hierarchy to track what ships when. We use the MCP tools (`mcp__filigree__*`) for all issue operations when available, falling back to the CLI only when MCP is down.

Issues should be created at the right granularity from the start, but **retyping is encouraged** when the scope becomes clearer — a task that grows into multi-session work should be promoted to a feature or epic with child tasks, and an epic that turns out to be a single grind session should be demoted to a task. To retype, create a new issue with the correct type (transferring description, labels, parent, and dependencies), then close the old one with a reason linking to the replacement. Filigree's `update_issue` doesn't support changing types directly, so this create-and-close pattern is the standard approach.

### Issue Type Usage

Filigree has types across four packs — use the right type for the right granularity:

| Type | When to use | Granularity test |
| ---- | ----------- | ---------------- |
| **milestone** | Top-level delivery theme | "What are we shipping this quarter?" |
| **phase** | Logical workstream within a milestone | "What area of the codebase does this touch?" |
| **epic** | Large body of work needing decomposition | "Does this need multiple features or tasks to complete?" |
| **feature** | User-facing capability with design decisions | "Does this need a user story, acceptance criteria, or design notes?" |
| **task** | Atomic unit of work one person can do in one sitting | "Can I start and finish this without needing to decompose further?" |
| **bug** | Defective behavior in existing code | "Is something broken, or is this a design evaluation?" |
| **release** | A planned/tested/shipped software release | "What version or release train does this ship in?" |
| **release_item** | A specific item included in, verified for, or excluded from a release | "Is this a release inclusion decision rather than implementation work?" |
| **requirement** | A durable product, safety, or compliance requirement | "Does this define what the system must do?" |
| **acceptance_criterion** | A testable condition proving a requirement or feature is satisfied | "How do we know the requirement is met?" |
| **deliverable** | A concrete output within a planning milestone | "What artifact or result must be produced?" |
| **step** | A sequenced planning step | "Is this one ordered step inside a larger plan?" |
| **work_package** | Assigned execution bundle within a plan | "Is this a package of work being assigned/coordinated?" |

**If a task has 3+ distinct deliverables or an unresolved design decision, promote it** to a feature or epic and create child tasks. XL-effort single tasks are untrackable — you can't mark them 50% done.

**If an epic has no children and the work is a single grind session, demote it** to a task. Epics without decomposition are just tasks with delusions of grandeur.

**Design evaluations are tasks, not bugs.** "Evaluate whether X should be eliminated" is a task. "X crashes when Y" is a bug.

### Issue Naming Conventions

**Title structure by type:**

| Type | Pattern | Example |
| ---- | ------- | ------- |
| **milestone** | Noun phrase (theme) | "Core Platform Maturation" |
| **phase** | Noun phrase (workstream) | "Architecture Refactoring" |
| **epic** | `Topic — scope summary` | "Landscape repository maturation — table-scoped access, CQRS split, unit-of-work" |
| **feature** | `Capability — what it enables` | "Server mode — persistent API service with REST + WebSocket" |
| **bug** | `Symptom — observable consequence` | "Coalesce timeouts only fire on next token arrival — no true idle flush" |
| **task** | `Action phrase — scope boundary` | "Unify reorder buffer implementations — single RowReorderBuffer for batching and pooling" |
| **release** | `Version or train — release theme` | "RC 5.1 — autonomous pipeline production hardening" |
| **release_item** | `Ship decision — item scope` | "RC 5.1 inclusion — PostgreSQL/S3 alternate configuration" |
| **requirement** | `Capability or constraint — required outcome` | "Audit lineage — generated pipelines retain prompt-to-run provenance" |
| **acceptance_criterion** | `Condition — observable proof` | "Runtime validation parity — composed YAML fails before execution on path-policy drift" |
| **deliverable** | `Output — delivery boundary` | "Staging runbook — session database recreation procedure" |
| **step** | `Action phrase — sequence boundary` | "Verify staging health — API and WebSocket smoke checks" |
| **work_package** | `Workstream — assigned scope` | "Telemetry exporter hardening — OTLP and Azure failure accounting" |

**Rules:**

1. **No internal tracking prefixes** — no `T20:`, `#7`, `C1:`, `M1:`, `H2:`, or similar sweep/scan-group identifiers
2. **No stale metrics in titles** — no line counts, entry counts, or other numbers that will drift. Put these in descriptions
3. **No process artifacts** — no "from 7-agent deep-dive", "from PR review". The provenance belongs in the description or a label
4. **No product prefixes** — no `ELSPETH-NEXT:`, `Use case:`
5. **Bugs describe the problem, not the fix** — lead with the observable symptom, not the action to take
6. **Em-dash separator** (`—`) between short name and expanded detail
7. **Sentence case, no trailing period**
