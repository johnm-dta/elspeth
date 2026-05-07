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

## Git Safety

**Never run destructive commands without explicit user permission:**

- `git reset --hard`, `git clean -f`, `git checkout -- <file>` — discards uncommitted changes
- `git push --force` — rewrites remote history
- `git rebase` (on pushed branches) — rewrites shared history

**No git stash.** The stash/pop cycle has caused repeated data loss in this project — pre-commit hooks that stash/unstash silently destroy unstaged work when `stash pop` encounters conflicts. If you need to preserve work, commit it to a branch.

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

**CRITICAL:** The following skills contain ELSPETH's core coding standards. You MUST invoke these skills (via the Skill tool) before performing any of the activities listed below. CLAUDE.md contains summary rules, but the skills contain the detailed code examples, decision tables, and boundary rules that prevent violations.

### Required Skills

| Skill | Contains |
|-------|----------|
| `engine-patterns-reference` | Composite PKs, schema contracts, header normalization, canonical JSON, retry semantics, secret handling, test path integrity, offensive programming examples, `hasattr` ban, layer architecture & `enforce_tier_model.py` (`check` and `dump-edges`) |
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

These standards interact in non-obvious ways. The tier model's fabrication rule (`None` → `0` is forbidden) is easy to miss without the detailed examples. The logging policy's "never log row-level decisions" contradicts common instinct. The config contracts pattern requires specific file changes that aren't discoverable from CLAUDE.md alone. Loading the skills ensures the detailed rules — not just the summaries — are in context when decisions are made.

---

<!-- filigree:instructions:v1.6.0:84820288 -->
## Filigree Issue Tracker

Use `filigree` for all task tracking in this project. Data lives in `.filigree/`.

### MCP Tools (Preferred)

When MCP is configured, prefer `mcp__filigree__*` tools over CLI commands — they're
faster and return structured data. Key tools:

- `get_ready` / `get_blocked` — find available work
- `get_issue` / `list_issues` / `search_issues` — read issues
- `create_issue` / `update_issue` / `close_issue` — manage issues
- `claim_issue` / `claim_next` — atomic claiming
- `add_comment` / `add_label` — metadata
- `list_labels` / `get_label_taxonomy` — discover labels and reserved namespaces
- `create_plan` / `get_plan` — milestone planning
- `get_stats` / `get_metrics` — project health
- `get_valid_transitions` — workflow navigation
- `observe` / `list_observations` / `dismiss_observation` / `promote_observation` — agent scratchpad
- `trigger_scan` / `trigger_scan_batch` / `get_scan_status` / `preview_scan` / `list_scanners` — automated code scanning
- `get_finding` / `list_findings` / `update_finding` / `batch_update_findings` — scan finding triage
- `promote_finding` / `dismiss_finding` — finding lifecycle (promote to issue or dismiss)

Observations are fire-and-forget notes that expire after 14 days. Use `list_issues --label=from-observation` to find promoted observations.

**Observations are ambient.** While doing other work, use `observe` whenever you
notice something worth noting — a code smell, a potential bug, a missing test, a
design concern. Don't stop what you're doing; just fire off the observation and
carry on. They're ideal for "I don't have time to investigate this right now, but
I want to come back to it." Include `file_path` and `line` when relevant so the
observation is anchored to code. At session end, skim `list_observations` and
either `dismiss_observation` (not worth tracking) or `promote_observation`
(deserves an issue) for anything that's accumulated.

Fall back to CLI (`filigree <command>`) when MCP is unavailable.

### CLI Quick Reference

```bash
# Finding work
filigree ready                              # Show issues ready to work (no blockers)
filigree list --status=open                 # All open issues
filigree list --status=in_progress          # Active work
filigree list --label=bug --label=P1        # Filter by multiple labels (AND)
filigree list --label-prefix=cluster:       # Filter by label namespace prefix
filigree list --not-label=wontfix           # Exclude issues with label
filigree show <id>                          # Detailed issue view

# Creating & updating
filigree create "Title" --type=task --priority=2          # New issue
filigree update <id> --status=in_progress                # Claim work
filigree close <id>                                      # Mark complete
filigree close <id> --reason="explanation"               # Close with reason

# Dependencies
filigree add-dep <issue> <depends-on>       # Add dependency
filigree remove-dep <issue> <depends-on>    # Remove dependency
filigree blocked                            # Show blocked issues

# Comments & labels
filigree add-comment <id> "text"            # Add comment
filigree get-comments <id>                  # List comments
filigree add-label <id> <label>             # Add label
filigree remove-label <id> <label>          # Remove label
filigree labels                             # List all labels by namespace
filigree taxonomy                           # Show reserved namespaces and vocabulary

# Workflow templates
filigree types                              # List registered types with state flows
filigree type-info <type>                   # Full workflow definition for a type
filigree transitions <id>                   # Valid next states for an issue
filigree packs                              # List enabled workflow packs
filigree validate <id>                      # Validate issue against template
filigree guide <pack>                       # Display workflow guide for a pack

# Atomic claiming
filigree claim <id> --assignee <name>            # Claim issue (optimistic lock)
filigree claim-next --assignee <name>            # Claim highest-priority ready issue

# Batch operations
filigree batch-update <ids...> --priority=0      # Update multiple issues
filigree batch-close <ids...>                    # Close multiple with error reporting

# Planning
filigree create-plan --file plan.json            # Create milestone/phase/step hierarchy

# Event history
filigree changes --since 2026-01-01T00:00:00    # Events since timestamp
filigree events <id>                             # Event history for issue
filigree explain-state <type> <state>            # Explain a workflow state

# All commands support --json and --actor flags
filigree --actor bot-1 create "Title"            # Specify actor identity
filigree list --json                             # Machine-readable output

# Project health
filigree stats                              # Project statistics
filigree search "query"                     # Search issues
filigree doctor                             # Health check
```

### File Records & Scan Findings (API)

The dashboard exposes REST endpoints for file tracking and scan result ingestion.
Use `GET /api/files/_schema` for available endpoints and valid field values.

Key endpoints:
- `GET /api/files/_schema` — Discovery: valid enums, endpoint catalog
- `POST /api/v1/scan-results` — Ingest scan results (SARIF-lite format)
- `GET /api/files` — List tracked files with filtering and sorting
- `GET /api/files/{file_id}` — File detail with associations and findings summary
- `GET /api/files/{file_id}/findings` — Findings for a specific file

### Workflow
1. `filigree ready` to find available work
2. `filigree show <id>` to review details
3. `filigree transitions <id>` to see valid state changes
4. `filigree update <id> --status=in_progress` to claim it
5. Do the work, commit code
6. `filigree close <id>` when done

### Session Start
When beginning a new session, run `filigree session-context` to load the project
snapshot (ready work, in-progress items, critical path). This provides the
context needed to pick up where the previous session left off.

### Priority Scale
- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog
<!-- /filigree:instructions -->

### How We Use Filigree

Filigree is the single source of truth for all work tracking in ELSPETH. The project hierarchy follows a consistent pattern: **milestones** (delivery themes like "Core Platform Maturation") contain **phases** (workstreams like "Architecture Refactoring"), which contain the actual work items — **epics**, **features**, **tasks**, and **bugs**. Releases (RC 3.4, RC 4) exist alongside this hierarchy to track what ships when. We use the MCP tools (`mcp__filigree__*`) for all issue operations when available, falling back to the CLI only when MCP is down.

Issues should be created at the right granularity from the start, but **retyping is encouraged** when the scope becomes clearer — a task that grows into multi-session work should be promoted to a feature or epic with child tasks, and an epic that turns out to be a single grind session should be demoted to a task. To retype, create a new issue with the correct type (transferring description, labels, parent, and dependencies), then close the old one with a reason linking to the replacement. Filigree's `update_issue` doesn't support changing types directly, so this create-and-close pattern is the standard approach.

### Issue Type Usage

Filigree has types across three packs — use the right type for the right granularity:

| Type | When to use | Granularity test |
| ---- | ----------- | ---------------- |
| **milestone** | Top-level delivery theme | "What are we shipping this quarter?" |
| **phase** | Logical workstream within a milestone | "What area of the codebase does this touch?" |
| **epic** | Large body of work needing decomposition | "Does this need multiple features or tasks to complete?" |
| **feature** | User-facing capability with design decisions | "Does this need a user story, acceptance criteria, or design notes?" |
| **task** | Atomic unit of work one person can do in one sitting | "Can I start and finish this without needing to decompose further?" |
| **bug** | Defective behavior in existing code | "Is something broken, or is this a design evaluation?" |

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

**Rules:**

1. **No internal tracking prefixes** — no `T20:`, `#7`, `C1:`, `M1:`, `H2:`, or similar sweep/scan-group identifiers
2. **No stale metrics in titles** — no line counts, entry counts, or other numbers that will drift. Put these in descriptions
3. **No process artifacts** — no "from 7-agent deep-dive", "from PR review". The provenance belongs in the description or a label
4. **No product prefixes** — no `ELSPETH-NEXT:`, `Use case:`
5. **Bugs describe the problem, not the fix** — lead with the observable symptom, not the action to take
6. **Em-dash separator** (`—`) between short name and expanded detail
7. **Sentence case, no trailing period**
