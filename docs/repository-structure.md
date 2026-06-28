# Repository Directory Strategy

Every top-level directory in this repository has **one clear, distinct purpose**.
This document is the map of those purposes and — more importantly — the rule for
**where a new file belongs**. If you cannot tell from this page where something
should go, that is a bug in the layout; fix the layout (and this page), don't
guess.

Scope: the top level and one level down where it aids placement. The internal
layout of `src/elspeth/` and `tests/` is intentionally *not* repeated here — see
the [Repository Architecture](../README.md#repository-architecture) section of
the root README and [ARCHITECTURE.md](../ARCHITECTURE.md) for the code tree.

## Purpose categories

| Category | Directory(ies) | One-line purpose |
| --- | --- | --- |
| Product code | `src/` | The shipped `elspeth` package (the only thing in `pyproject` `packages`). See README/ARCHITECTURE for internals. |
| Tests | `tests/` | The test suite (unit / integration / property / e2e / performance). |
| Active documentation | `docs/` | Current, audience-facing docs. Index: [`docs/README.md`](README.md). |
| Archived documentation | `docs-archive/` | Dated, point-in-time doc cleanouts (`<date>-docs-cleanout/`), each with a `MANIFEST.md`. Same public repo; de-emphasised, not deleted. |
| Examples | `examples/` | Runnable reference pipelines and configs for users (one folder per scenario). |
| Developer & CI automation | `scripts/` | Runnable repo automation: CI check logic, eval drivers, git hooks, audits, deploy helpers. |
| Deliverable-artifact build | `tools/` | Build pipelines that render *distributable artifacts* (currently `tools/pdf/`). |
| CI/CD & governance config | `config/`, `.github/`, `.githooks/` | Declarative policy + workflow triggers + local hook bindings (see [§ CI three-way](#ci-the-three-way-split)). |
| Deployment | `deploy/`, root `Dockerfile` / `docker-compose.yaml` | How the service is shipped and run (systemd unit + env; container image/compose). |
| Internal evaluation | `evals/` | Self-contained LLM/composer evaluation harness + dated run records. Imported only by itself; not part of the shipped package. |
| Engineering notes | `notes/` | Ad-hoc engineering memos and baselines — explicitly internal, low-ceremony. |
| Auxiliary package | `elspeth-lints/` | The CI tier-model linter — its own Python package (own `pyproject.toml`) consumed by CI, not by `src/`. |
| Marketing / landing site | `website/` | Standalone static site (HTML/CSS/JS), built and served independently of the app frontend. |
| Runtime / working data | `data/`, `state/` | App working data and the audit database (see [§ Working-state](#working-state-where-runtime-data-lives)). Mostly gitignored. |
| Local tool/runtime state | the gitignored dot-dirs | `.venv`, `.ruff_cache`, `.mypy_cache`, `.pytest_cache`, `.hypothesis`, `.uv-cache`, `node_modules`, `.loomweave`, `.filigree`, `.weft`, `.clarion`, `.claude`, `.codex`, `.agents`, `.superpowers`, `.worktrees`, `scratch/`, … — never shipped, never relied on by tracked code. One bucket; do not itemise. |
| Root metadata & manifests | root files | Governance/community docs (`LICENSE`, `GOVERNANCE.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SUPPORT.md`), product docs (`README.md`, `ARCHITECTURE.md`, `PLUGIN.md`, `CHANGELOG.md`), and build/tooling config (`pyproject.toml`, `uv.lock`, `package*.json`, `Dockerfile`, `weft.toml`, `.pre-commit-config.yaml`, `.mcp.json`, dot-config). |

> Agent-instruction files (`CLAUDE.md`, `AGENTS.md`) are deliberately
> **gitignored** — they are local agent configuration, not shipped repo content.

## One level down (where it aids placement)

- **`config/`** → `cicd/` (one folder per CI policy: cells, allowlists, defaults), `mcp/` (MCP server configuration).
- **`scripts/`** → `cicd/`, `eval/`, `git-hooks/`, `scan-groups/`, `skill_rgr/`, `archive/` (superseded one-offs), plus top-level utility scripts (`generate_test_data.py`, `run_mutation_testing.py`, `validate_deployment.py`, `deploy-vm.sh`, `smoke-test.sh`).
- **`tools/`** → `pdf/` (Typst/Pandoc artifact builds).
- **`evals/`** → dated run folders (`2026-05-03-composer/` …), `composer-harness/`, `composer-rgr/`, `lib/` (shared eval code).
- **`data/`** → `skills/` (deployment examples of the live composer skill prompt); runtime DBs live here at deploy time (gitignored).
- **`docs/`** → see its own [index](README.md); plans/specs that are implemented move to `docs-archive/`.

## Decision rules — where does a new file go?

| You are adding… | Put it in… |
| --- | --- |
| A new CI policy / allowlist / rule config | `config/cicd/<policy>/` |
| The runnable logic for a CI check | `scripts/cicd/` |
| A GitHub Actions workflow | `.github/workflows/` |
| A general dev/repo automation script | `scripts/` (or the matching `scripts/<area>/`) |
| A build that renders a distributable PDF/artifact | `tools/<artifact>/` (the release assurance PDF set lives in `docs/release/pdf/`) |
| A new example pipeline | `examples/<scenario>/` |
| A design/plan/spec doc | `docs/` while active → `docs-archive/<date>-docs-cleanout/` once implemented |
| An evaluation scenario or harness change | `evals/` |
| A deployable artifact (service unit, env template) | `deploy/` |
| A throwaway working file | `scratch/` (gitignored) — **never** commit it |
| Runtime/working data the app reads or writes | `data/` (sessions/working) or `state/` (audit DB) |

The test for any folder: *would a contributor know, without asking, where to put
a new file?* Where two folders could both plausibly hold it, the boundary is
adjudicated below.

## Distinctness adjudication (overlap clusters)

These are the places where "one clear, distinct purpose" is non-obvious. Each
either has a stated boundary (✓) or a flagged problem (⚠, see
[Findings](#findings--recommended-actions)).

### CI: the three-way split
✓ Distinct by *axis*:
- `.github/workflows/` — **when** CI runs (triggers, job orchestration).
- `config/cicd/` — **what** is enforced (declarative policy cells, allowlists, defaults).
- `scripts/cicd/` — **how** a check executes (the runnable analysis logic).

### scripts/ vs tools/
✓ `scripts/` = automation run *during development/CI*; `tools/` = build pipelines
that produce *distributable artifacts*. A helper that emits a deliverable PDF is a
tool; a helper that lints, tests, deploys, or generates fixtures is a script.

### PDF pipelines: tools/pdf/ vs docs/release/pdf/
✓ Distinct by *output*: `docs/release/pdf/` builds the **release assurance set**
(executive-summary, architecture, composer, guarantees, data-trust); `tools/pdf/`
builds the **architecture presentation pack** (`build-arch-pack.sh`). The stale
2026-05-03 composer briefing/walkthrough builders were archived 2026-06-28, so the
two pipelines no longer overlap.

### Deployment spread
✓ Distinct by *target*: `deploy/` = host service (systemd unit + env); root
`Dockerfile`/`docker-compose.yaml` = container image/compose; `scripts/deploy-vm.sh`
+ `validate_deployment.py` = the automation that drives a deploy.

### Working-state: where runtime data lives
✓ Distinct by *role*: `data/` = app/session working data (sessions DB, skill
examples); `state/` = the `audit.db` landscape database (its path is hardcoded in
`core/config.py`). Scratch is settled by `.gitignore`: `.scratch/` is the canonical
composer-MCP scratch dir (kept via `.gitkeep`), `scratch/` is the ignored
duplicate. `.elspeth/rotations.log` is deliberately force-tracked as a tier-model
rotation audit trail.

### prompts overlap
✓ Resolved — the root `prompts/` one-off was archived 2026-06-28; agent prompts
that must live in-tree belong under a single, named home, not a second top-level
`prompts/`.

## Findings & recommended actions

Genuine *purpose* problems only — overlap, cruft, or ambiguity. Folders that are
merely internal (e.g. `evals/`, `notes/`) are **not** flagged: "internal eval
harness" and "engineering notes" are clear, distinct purposes.

| # | Issue | Resolution (2026-06-28) |
| --- | --- | --- |
| 1 | Root `prompts/` (1 file) overlaps the archived `docs/prompts/` | **Done** — relocated to `docs-archive/2026-06-28-docs-cleanout/prompts/` (no inbound refs) |
| 2 | `.scratch/` vs `scratch/` looked duplicated | **No change** — `.gitignore` documents `.scratch/` as the canonical composer-MCP scratch dir and `scratch/` as the ignored duplicate; already resolved |
| 3 | `.benchmarks/` (root) empty and **not** gitignored | **Done** — deleted and added `/.benchmarks/` to `.gitignore` |
| 4 | `.elspeth/rotations.log` under version control | **No change** — deliberately force-tracked (`!.elspeth/rotations.log`) as a tier-model rotation audit trail |
| 5 | `tools/pdf/` briefing/walkthrough builders tied to archived 2026-05-03 evidence | **Done** — archived the two builders + their metadata; `build-arch-pack.sh` and shared infra retained |
| 6 | `data/` vs `state/` runtime split not self-evident | **Documented** here (`state/audit.db` path is hardcoded in `core/config.py`, so it stays put) |

Not flagged (clear, distinct purpose): `src/`, `tests/`, `docs/`, `docs-archive/`,
`examples/`, `scripts/`, `tools/`, `config/`, `.github/`, `.githooks/`, `deploy/`,
`evals/`, `notes/`, `elspeth-lints/`, `website/`.
