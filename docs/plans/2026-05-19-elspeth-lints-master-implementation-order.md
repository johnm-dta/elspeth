# elspeth-lints Master Implementation Order

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`
> or `superpowers:subagent-driven-development` when implementing any task from
> this order. Claim the relevant Filigree issue before changing code.

**Goal:** Consolidate ELSPETH's bespoke CI/CD enforcement scripts into the
workspace-only `elspeth-lints` package without changing enforcement behavior
during migration, while preserving a separate lane for analyzer-quality and
allowlist-burn-down fixes.

**Architecture:** Treat the migration as a behavior-preserving refactor guarded
by a parity harness. New rules run in shadow mode until old-vs-new findings are
identical, then cut over and delete the old script. Behavior-changing analyzer
fixes, such as R1/R5 rule precision improvements, must land as separate issues
with their own regression proof.

**Tech stack:** Python `ast`, uv workspace package, pre-commit, GitHub Actions,
SARIF 2.1.0, Filigree, existing `scripts/cicd/enforce_*.py` gates.

---

## Scope Of This Review

This file reconciles the Markdown corpus for the current CI analyzer body of
work, plus the live Filigree issue graph. It is a master ordering document, not
the detailed implementation plan for every child issue.

Markdown inputs reviewed:

- `docs/architecture/adr/023-custom-python-ci-analyzer.md`
- `docs/architecture/adr/README.md`
- `docs/audit/2026-05-19-cicd-allowlist-audit.md`
- `docs/audit/findings/_tickets-to-file.md`
- `docs/audit/findings/fp-analyst.md`
- `docs/audit/findings/python-reviewer.md`
- `docs/audit/findings/refactoring-architect.md`
- `docs/audit/findings/silent-failure-hunter.md`

Tracker inputs reviewed:

- `elspeth-8843308cfe` - consolidation epic
- All 19 direct child issues under that epic
- `elspeth-797cac825e` - companion rationale document task

Scope note: the repository contains many unrelated Markdown documents. This
ordering focuses on the Markdown and tracker artifacts that describe the
`elspeth-lints` / CI enforcement work.

---

## Non-Negotiables

1. Use a dedicated worktree for implementation tranches under `.worktrees/`.
2. Start each tranche with `filigree session-context`, then `filigree start-work
   <issue> --assignee <name>` or the MCP equivalent.
3. Load the ELSPETH coding-standard skills before design, planning, code,
   tests, debugging, or review:
   - `engine-patterns-reference`
   - `tier-model-deep-dive`
   - `logging-telemetry-policy`
   - `config-contracts-guide`
4. Existing scripts are the behavioral source of truth until their port reaches
   parity and cutover.
5. Every port follows `shadow -> cutover -> deleted` in
   `config/cicd/lint_migration_status.yaml`.
6. A port PR may refactor packaging, invocation, and output format. It must not
   silently change what is enforced.
7. Analyzer behavior fixes are allowed, but only as separate issues/PRs with
   failing regression tests or before/after scanner fixtures.
8. Do not weaken audit primacy, tier-boundary semantics, or config-contract
   enforcement to make a lint migration easier.
9. Do not close the epic from memory. Run the explicit closeout verification
   issue and cite evidence for every acceptance criterion.

---

## Current State

ADR-023 accepts custom Python as the toolchain for ELSPETH-specific invariants
and rejects porting the current rules to CodeQL, Semgrep, or ast-grep. CodeQL
stays as a complementary generic vulnerability analyzer.

The consolidation epic is `elspeth-8843308cfe`, titled "Consolidate CI/CD
enforcement scripts into elspeth-lints package". Its corrected plan says:

1. Foundation first.
2. Pilot port second.
3. Medium category ports third.
4. `trust_tier` last.
5. CI, pre-commit, SARIF upload, rationale docs, and verification complete the
   work.

The CI allowlist audit is adjacent but not identical work. It found that all
current gates pass, but the tier-model allowlist is decaying: the allowlist grew
from 447 entries on 2026-04-17 to roughly 674-675 entries on 2026-05-19, with
most entries permanent. Those findings should become rule-quality and lifecycle
work, not be smuggled into behavior-preserving migration commits.

---

## Implementation Order

### Phase 0 - Decision Artifact And Baseline

**Primary issues:**

- `elspeth-80a0ab611c` - CodeQL/custom-Python ADR

**Do first:**

1. Land `docs/architecture/adr/023-custom-python-ci-analyzer.md`.
2. Land the ADR index update in `docs/architecture/adr/README.md`.
3. Link ADR-023 from the future `elspeth-lints` README/rationale surfaces once
   those files exist.
4. Record the baseline before any migration:
   - current `scripts/cicd/enforce_*.py` inventory
   - current `.pre-commit-config.yaml` static-analysis hooks
   - current `.github/workflows/ci.yaml` job count
   - current allowlist counts
   - current gate pass/fail status

**Why first:** ADR-023 answers the external-review question "why not CodeQL?"
and unblocks the companion rationale task. The baseline prevents later
migration PRs from guessing what changed.

**Do not:** start porting rules before the parity-harness design exists.

---

### Phase 1 - Foundation PR: Skeleton, Protocols, And Meta-Gate

**Primary issues:**

- `elspeth-8e71b27ba0` - bootstrap package skeleton
- `elspeth-37acb657e6` - core protocols and frozen CLI contract
- `elspeth-d62f4b7ad7` - meta-CI gate forbidding new bespoke `enforce_*.py`
  scripts

**Recommended PR shape:** one co-landed foundation PR.

**Files expected:**

- `elspeth-lints/pyproject.toml`
- `elspeth-lints/src/elspeth_lints/core/ast_walker.py`
- `elspeth-lints/src/elspeth_lints/core/allowlist.py`
- `elspeth-lints/src/elspeth_lints/core/cli.py`
- `elspeth-lints/src/elspeth_lints/core/findings.py`
- `elspeth-lints/src/elspeth_lints/core/protocols.py`
- `elspeth-lints/src/elspeth_lints/core/registry.py`
- `docs/elspeth-lints/protocols.md`
- `docs/elspeth-lints/rule-author-guide.md`
- `tests/...` smoke/protocol tests for the skeleton

**Order inside the PR:**

1. Create package skeleton and editable install path.
2. Define `Rule`, `Finding`, `Severity`, `Category`, `RuleMetadata`, and the
   allowlist schema.
3. Define the `elspeth-lints check` CLI contract and exit codes.
4. Add a stub rule to prove registry discovery and CLI invocation.
5. Add the meta-gate that blocks any new `scripts/cicd/enforce_*.py` outside
   the migration manifest/legacy allowlist.
6. Document how rule authors add new rules under `elspeth-lints` instead of
   adding scripts under `scripts/cicd/`.

**Gate:** `elspeth-lints check --rules nothing --root /tmp` exits 0 with an
empty finding set.

**Why co-land:** without protocols, the first port accidentally defines the
interface. Without the meta-gate, new bespoke scripts can appear during the
migration window.

---

### Phase 2 - Foundation Verification And Output Surfaces

**Primary issues:**

- `elspeth-e63801dee1` - findings-parity harness
- `elspeth-9421b0bc92` - shared fixture convention
- `elspeth-4e6cb5b995` - SARIF/github/text emitters

**Recommended PR shape:** parallel-ready after Phase 1, but all must close
before the pilot port closes.

**Required deliverables:**

1. `scripts/cicd/parity_harness.py`
   - compares old script output to new `elspeth-lints` output
   - normalizes findings by file, line, column, rule id, fingerprint, and
     allowlist match
   - exits non-zero on any diff
2. `config/cicd/lint_migration_status.yaml`
   - per-rule lifecycle: `shadow`, `cutover`, `deleted`
3. `elspeth_lints.core.fixture_harness`
   - requires every rule to provide `examples_violation` and `examples_clean`
4. `elspeth_lints.core.emitters`
   - `sarif.py`
   - `github.py`
   - `text.py`
5. Golden-file tests for every output format.

**Gate:** a mock old-vs-new pair with a deliberate one-line drift must fail the
parity harness. A no-drift pair must pass.

**Why before pilot:** the pilot must validate the real migration lifecycle, not
just rule execution.

---

### Phase 3 - Pilot Port And Interface Freeze

**Primary issue:**

- `elspeth-8e8b49d58e` - pilot port for `enforce_options_metadata.py`

**Recommended PR shape:** one pilot PR after all Phase 1 and Phase 2 blockers
are closed.

**Order:**

1. Add `plugin_contract/options_metadata` as the first real rule.
2. Add at least three violation fixtures and three clean fixtures.
3. Add a migration manifest entry in `shadow`.
4. Run the old script and new rule side by side through the parity harness.
5. Add pre-commit/CI shadow invocation.
6. Freeze the protocol surface at `v1.0` in `docs/elspeth-lints/protocols.md`.

**Gate:** parity reports zero diff across the required trees. Old script still
runs; new rule is shadow-only until the cutover criteria are met.

**Why this rule:** `enforce_options_metadata.py` is small enough to port in one
bounded task but complex enough to exercise class traversal, metadata semantics,
allowlist loading, fixtures, emitters, and CI wiring.

**Do not:** change the protocol surface after this closes without a separate
interface-evolution issue.

---

### Phase 4 - Medium Category Ports

**Primary issues:**

- `elspeth-a3f6004c83` - immutability rules
- `elspeth-64d67da0de` - audit-evidence rules
- `elspeth-90576d9826` - remaining plugin-contract rules
- `elspeth-48db98bd88` - composer and manifest/inventory rules

**Parallelism:** these are parallel-ready after the pilot freezes the protocol.
If one implementer is doing them serially, use this order:

1. Immutability: closest to the pilot's class/dataclass traversal shape.
2. Audit evidence: introduces shared audit call-site visitors.
3. Remaining plugin contract: includes `component_type` and plugin hash manifest
   parity.
4. Composer + manifest/inventory: introduces the broadest AST-statement and
   whole-repo manifest surface.

**Port workflow for each category:**

1. Add or import fixtures for the old behavior.
2. Implement the new rule package under
   `elspeth-lints/src/elspeth_lints/rules/<category>/`.
3. Declare every rule's metadata, including `scope = incremental | whole_repo`.
4. Add/port allowlist loading with paths preserved where possible.
5. Add migration manifest entry in `shadow`.
6. Run parity.
7. Once parity is green in CI, cut over.
8. Delete the old script only after cutover evidence exists.

**Coordination rule:** the first AST-statement-pattern port that needs helpers
owns the first helper design under `core/walkers/` or `core/ast_helpers/`.
Coordinate before adding a second helper with a conflicting shape.

**Do not:** use the medium ports to change findings. Any "this rule should be
smarter" discovery becomes a separate rule-quality issue.

---

### Phase 5 - trust_tier Port Last

**Primary issue:**

- `elspeth-73990ea9f8` - port `enforce_tier_model.py`

**Start only after:**

- Pilot port closed.
- All four medium category ports are closed.
- Parity harness and fixture harness are battle-tested.
- SARIF/text/github emitters have already handled real findings.

**Order inside the task:**

1. Reconfirm the current rule behavior with the gate command:
   `env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth`
2. Port current behavior into `rules/trust_tier/` with submodules for the
   individual defensive-pattern/layer rules.
3. Preserve allowlist path compatibility:
   `config/cicd/enforce_tier_model/<module>.yaml`.
4. Preserve fingerprint rotation behavior.
5. Preserve `dump-edges --no-timestamp` byte-identical output through
   `elspeth-lints dump-edges`.
6. Land in shadow.
7. Run parity across the required trees.
8. Cut over only after zero-diff parity.
9. Delete the old script in the cutover-confirmed commit.

**Special warning:** the allowlist audit identified real rule-quality work in
R1, R5, lifecycle, and owner taxonomy. Do not mix those changes into the
behavior-preserving `trust_tier` port. Either:

- land them before this port as old-script behavior changes, update the baseline,
  then port that corrected behavior, or
- land them after this port as new `elspeth-lints` rule-quality changes.

The second option is usually safer because the port can prove equivalence first.

---

### Phase 6 - Analyzer Quality And Allowlist Burn-Down Lane

**Primary parent:** `elspeth-297b8f5c5d` and any subtasks created from
`docs/audit/findings/_tickets-to-file.md`.

This lane is related to `elspeth-lints`, but it is not the same as the migration
epic. Keep it separate unless a rule-quality fix is explicitly made a blocker.

**Time-sensitive first:**

1. Renew/correct the near-expiry entry expiring on 2026-06-07:
   `plugins/infrastructure/clients/http.py:R4:AuditedHTTPClient:_emit_telemetry_after_audit`.
2. Renew/correct the near-expiry entry expiring on 2026-06-15:
   `contracts/contract_builder.py:R6:ContractBuilder:process_first_row`.

**Then implement independent rule-quality fixes:**

1. R1 receiver-type heuristic:
   - distinguish `httpx.AsyncClient.get`, `aiohttp.ClientSession.get`,
     `asyncio.Queue.get`, and similar non-dict `.get()` calls from `dict.get`.
   - remove confirmed wallpaper allowlist entries in the same PR.
2. R5 split:
   - R5a: frozen dataclass `__post_init__` offensive guards.
   - R5b: Tier-3 boundary validators.
   - R5c: forbidden Tier-2 defensive `isinstance` checks.
3. Lifecycle:
   - promote unmatched/orphaned allowlist entries to CI-failing.
   - default new entries to bounded expiry, not `expires: null`.
   - add permanent-entry growth reporting.
4. Owner taxonomy:
   - replace `bugfix`, `feature`, and `refactor` owners with subsystem owners.
   - require non-null expiry for ticket-ID owners.
5. Code suppressions identified as restructurable:
   - exporter sparse indexes: align `defaultdict` annotations and use direct
     subscription instead of `.get(key, [])`.
   - sink display mapping: add `display_name_for`, rewrite repeated
     `display_map.get(field, field)` call sites.
   - `normalize_type_for_contract`: consider sentinel/Result return instead of
     `TypeError` as protocol signal.

**Gate:** every rule-quality change needs a focused regression fixture proving
the old behavior was wrong or too noisy and the new behavior is intentional.

---

### Phase 7 - CI Graph, Pre-Commit, And GitHub Code Scanning

**Primary issues:**

- `elspeth-16f04d5049` - collapse CI jobs
- `elspeth-0b6034e09b` - split pre-commit incremental vs PR CI full-repo
- `elspeth-b79958739e` - upload SARIF to GitHub Code Scanning

**Recommended order:**

1. Collapse the static-analysis CI surface once `elspeth-lints check --rule-set
   full --format sarif` is real. If CI runtime is painful earlier, do a
   temporary shell-loop collapse over existing scripts, but treat that as stage
   1 of the CI task.
2. Split pre-commit from CI only after real ported rules have honest
   `metadata.scope` values:
   - `incremental` rules can run on changed files.
   - `whole_repo` rules stay full-repo or CI-only.
3. Upload SARIF only after the SARIF emitter and consolidated static-analysis
   job both exist.

**Gates:**

- CI job count is at most 8.
- `ci-success` branch-protection aggregation still works.
- Median pre-commit wall-clock on a one-file change is under 10 seconds.
- A deliberate violation PR produces both a PR annotation and a Security-tab
  finding with SARIF category `elspeth-lints`.

---

### Phase 8 - Rationale, Naming Cleanup, And Closeout

**Primary issues:**

- `elspeth-797cac825e` - `docs/elspeth-lints/rationale.md`
- `elspeth-3d1227a289` - rename ADR/phase-numbered CI scripts to invariant
  names
- `elspeth-b50bd1b8f0` - epic-close verification

**Order:**

1. Write `docs/elspeth-lints/rationale.md` after ADR-023 lands and after the
   rule-author guide/protocol docs exist.
2. Do naming cleanup during the manifest/composer port when possible:
   - `adr019_symbol_inventory.py` -> `symbol_inventory.py`
   - `adr019_test_inventory.py` -> `test_to_source_mapping.py`
   - phase-numbered trailer language -> stable invariant naming
3. Run epic-close verification as its own task.

**Closeout evidence required:**

- `scripts/cicd/enforce_*.py` inventory shows all migrated scripts gone, or
  `enforce_adapter_budget.py` has an explicit waiver.
- `find elspeth-lints/src/elspeth_lints/rules -name '*.py'` matches the expected
  taxonomy.
- Parity checks are green for every migrated rule.
- SARIF upload is live.
- CI job count is at most 8.
- Meta-CI gate rejects new bespoke scripts.
- ADR-023 and `docs/elspeth-lints/rationale.md` are merged and linked.
- No ADR/phase-numbered CI scripts remain.
- `docs/elspeth-lints/protocols.md` declares interface `v1.0`.
- The workspace-only packaging decision is documented.

---

## Dependency Graph Summary

| Phase | Issue | Role | Must happen before |
|---|---|---|---|
| 0 | `elspeth-80a0ab611c` | ADR-023 / custom Python decision | rationale, epic close |
| 1 | `elspeth-8e71b27ba0` | package skeleton | everything in the package |
| 1 | `elspeth-37acb657e6` | protocol surface | all ports |
| 1 | `elspeth-d62f4b7ad7` | meta-gate | pilot |
| 2 | `elspeth-e63801dee1` | parity harness | all ports |
| 2 | `elspeth-9421b0bc92` | fixture convention | all ports |
| 2 | `elspeth-4e6cb5b995` | SARIF emitters | all ports, SARIF upload |
| 3 | `elspeth-8e8b49d58e` | pilot port / interface freeze | medium ports |
| 4 | `elspeth-a3f6004c83` | immutability ports | trust_tier, pre-commit split |
| 4 | `elspeth-64d67da0de` | audit-evidence ports | trust_tier, pre-commit split |
| 4 | `elspeth-90576d9826` | plugin-contract ports | trust_tier |
| 4 | `elspeth-48db98bd88` | composer + manifest ports | trust_tier |
| 5 | `elspeth-73990ea9f8` | trust_tier port | epic close |
| 7 | `elspeth-16f04d5049` | CI graph collapse | SARIF upload, epic close |
| 7 | `elspeth-0b6034e09b` | pre-commit split | epic close |
| 7 | `elspeth-b79958739e` | Code Scanning upload | epic close |
| 8 | `elspeth-797cac825e` | rationale doc | epic close |
| 8 | `elspeth-3d1227a289` | invariant naming cleanup | epic close |
| 8 | `elspeth-b50bd1b8f0` | evidence-backed closeout | epic close |

---

## Recommended PR Sequence

1. **PR 0 - Decision docs**
   - ADR-023 and ADR index.
   - This master ordering file.
2. **PR 1 - Foundation substrate**
   - skeleton, protocols, meta-gate.
3. **PR 2 - Foundation verification**
   - parity harness, migration manifest, fixture harness, emitters.
4. **PR 3 - Pilot**
   - `options_metadata` in shadow mode, interface `v1.0` frozen.
5. **PR 4A - Immutability port**
6. **PR 4B - Audit-evidence port**
7. **PR 4C - Plugin-contract port**
8. **PR 4D - Composer + manifest port**
9. **PR 5 - trust_tier port**
10. **PR 6 - Analyzer quality lane**
    - separate from migration PRs; can be split further by R1, R5, lifecycle,
      and source-code suppression cleanup.
11. **PR 7 - CI/pre-commit/SARIF upload**
12. **PR 8 - Rationale/naming/epic close verification**

Parallel execution is safe only where the dependency graph says it is safe.
Parallel workers must use disjoint write sets and coordinate shared helper APIs.

---

## Verification Commands To Keep Handy

Use the repo virtualenv unless a task explicitly documents a different toolchain.
Adapt exact test paths to the files touched in the tranche.

```bash
.venv/bin/python -m pytest tests/path/to/targeted_tests.py -v
.venv/bin/python -m mypy src/ elspeth-lints/src/
.venv/bin/python -m ruff check src/ tests/ elspeth-lints/ scripts/cicd/
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
.venv/bin/pre-commit run --all-files
git diff --check
```

Port-specific verification must include:

```bash
.venv/bin/python scripts/cicd/parity_harness.py --rule <rule-id>
elspeth-lints check --rules <rule-id> --root src/elspeth --format text
elspeth-lints check --rules <rule-id> --root src/elspeth --format sarif > /tmp/<rule-id>.sarif
```

---

## Rules For Splitting Behavior Changes From Migration

Behavior-preserving migration PRs may:

- move code into `elspeth-lints`
- normalize finding objects
- add output emitters
- preserve existing allowlist paths
- add fixtures that characterize current behavior
- add shims from old script names to the new CLI
- delete old scripts after cutover

Behavior-preserving migration PRs must not:

- delete allowlist entries because they "look stale" without proving the rule no
  longer fires
- change R1/R5/R6 semantics
- change path filters
- change fingerprint algorithms without proving parity
- collapse multiple old findings into one new finding unless the old behavior is
  intentionally changed in a separate rule-quality issue

Rule-quality PRs may change behavior, but must:

- carry a focused regression fixture or test
- update allowlists in the same PR
- name whether they are changing the old script, the new `elspeth-lints` rule,
  or both
- update the migration baseline if they land before the relevant port

---

## Risk Register

| Risk | Mitigation |
|---|---|
| First port accidentally defines the whole interface | Protocol task before pilot; pilot freezes only after real rule proof |
| Migration changes enforcement behavior | Parity harness blocks cutover |
| New bespoke scripts appear during migration | Meta-CI gate blocks new `scripts/cicd/enforce_*.py` |
| R1/R5 allowlist noise hides real regressions | Separate analyzer-quality lane with focused rule fixes |
| `trust_tier` fingerprint rotation invalidates hundreds of entries | Preserve fingerprint algorithm; add explicit stability tests |
| SARIF exists but no one sees it | Code Scanning upload task is required, not optional |
| Slow pre-commit encourages `--no-verify` | Scope metadata enables changed-file pre-commit and full PR CI |
| Epic closes on assumption | Dedicated epic-close verification task with evidence for every criterion |

---

## Done For This Master Order

This master order is complete when:

- It reconciles the ADR, audit, findings, and Filigree child issue graph.
- It names the implementation order and blockers.
- It separates migration from behavior-changing analyzer fixes.
- It identifies which work can be parallelized and which work is on the
  critical path.
- It can be handed to future implementers without requiring them to rediscover
  the structure from scattered Markdown files.
