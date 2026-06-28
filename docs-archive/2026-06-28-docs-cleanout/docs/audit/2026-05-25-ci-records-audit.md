# CI Records Audit — 2026-05-25

Verdict: **partially trustworthy**.

The live enforcement model is real, but not uniformly trustworthy. Branch
protection is active through ruleset `12348893`, and it requires the expected
top-level contexts: `CI Success`, `CodeQL`, `Check cohort-attribution trailers
on PR commits`, and `redaction-gate`. The current local checkout is
`RC5.2` at `080e4ecdd14dbaaabda3fdf68c4bdc594a19ef16`, one commit ahead of
`origin/RC5.2` (`62e2133f590575a0cf1b75294b6127f1e40d8bab`), so GitHub Actions
state proves the pushed head while local scanner runs prove this checkout.

Post-remediation note: this changeset repaired the two local static-gate
failures captured below. The audit findings preserve the original failure
evidence; the active-debt section distinguishes remaining live governance debt
from gates that now pass locally.

## Findings

### P0 — Redaction label gate misses the current composer tools package

- **File:** `.github/workflows/composer-redaction-gate.yml:40-47`,
  `tests/unit/web/composer/test_label_gate_yaml.py:27-35`
- **Record/key:** `REDACTION_PATHS`, `REQUIRED_PATHS`
- **Active surface:** GitHub Actions; required by live ruleset as
  `redaction-gate`
- **Evidence:** the workflow still watches
  `src/elspeth/web/composer/tools.py`, but that file does not exist at HEAD.
  The live tool implementation is the `src/elspeth/web/composer/tools/`
  package, and modules inside it import/use redaction policy and registry
  invariants. The structural test encodes the same stale path, so it preserves
  the gap instead of catching it.
- **Impact:** a PR that changes redaction-relevant composer tool code under
  `src/elspeth/web/composer/tools/` can make the required `redaction-gate`
  conclude "No redaction-sensitive files changed" unless the snapshot also
  changes. This weakens a branch-protection-required audit/security gate.
- **Recommended action:** replace the stale path with a directory path set that
  covers `src/elspeth/web/composer/tools/**` or the exact redaction-bearing
  plane modules, update `test_label_gate_yaml.py`, and add a negative fixture
  proving a `tools/` change activates the gate.

### P0 — Audit-time local CI static gates were red despite policy records claiming coverage

- **File:** `.github/workflows/ci.yaml:118-123`,
  `src/elspeth/web/interpretation_state.py:83-88`
- **Record/key:** `immutability.frozen_annotations`
- **Active surface:** GitHub Actions; partially pre-commit
- **Evidence:** local live run:
  `env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules immutability.freeze_guards,immutability.frozen_annotations --root src/elspeth`
  failed with
  `web/interpretation_state.py:84:0: FG3: Frozen dataclass 'InterpretationReviewPending' has container fields ['sites'] but no __post_init__`.
- **Impact:** a configured CI gate failed on this checkout. Pre-commit has a
  changed-file `frozen_annotations` hook, but the whole-repo CI invocation is
  stricter and was red at audit capture.
- **Recommended action:** fix or explicitly justify
  `InterpretationReviewPending.sites`; do not widen the allowlist unless the
  container mutability is deliberately load-bearing and documented with an
  expiry.

### P0 — Audit-time local contract gate was red on untracked `dict[str, Any]` returns

- **File:** `.github/workflows/ci.yaml:295-300`,
  `src/elspeth/web/sessions/service.py:515-552`
- **Record/key:** `config/cicd/contracts-whitelist.yaml`
- **Active surface:** GitHub Actions; pre-commit only when contract files or
  config surfaces change
- **Evidence:** local live run:
  `.venv/bin/python scripts/check_contracts.py` failed with
  `src/elspeth/web/sessions/service.py:515: _interpretation_hash_domain_v2 - return`.
  The function returns `dict[str, Any]`, but no matching entry exists in
  `contracts-whitelist.yaml`.
- **Impact:** the Settings/runtime contract gate and dict-pattern whitelist are
  not in sync with code at this checkout. Because the function builds audit hash
  material, this is not just type hygiene; it touches audit determinism.
- **Recommended action:** prefer a `TypedDict` or frozen dataclass for the hash
  domain. If an exemption is truly required, add a narrow whitelist entry with
  owner, rationale, and expiry.

### P1 — Release image publishing validates only `CI Success`, not the full branch-protection policy

- **File:** `.github/workflows/build-push.yaml:61-68`,
  `.github/workflows/build-push.yaml:87-111`
- **Record/key:** `Verify CI success for image commit`
- **Active surface:** GitHub Actions on `workflow_run`, tags, and manual
  dispatch; not required by branch protection
- **Evidence:** the publish job is allowed on tags/manual dispatch and queries
  only `check_name=CI Success`. The live ruleset separately requires `CodeQL`,
  `redaction-gate`, and `Check cohort-attribution trailers on PR commits`.
- **Impact:** tag or manual image publication can proceed for a commit with a
  successful aggregate CI check even if another branch-protection-required
  governance check was missing, stale, or failed.
- **Recommended action:** make build-push validate the same required context set
  as the active ruleset, or restrict tag/manual publication to signed release
  commits whose required contexts are all green.

### P1 — `enforce_adapter_budget.py` is a live script but dormant policy

- **File:** `config/cicd/lint_migration_status.yaml:15-18`,
  `scripts/cicd/enforce_adapter_budget.py:1-7`
- **Record/key:** `scripts/cicd/enforce_adapter_budget.py`
- **Active surface:** dormant; not GitHub Actions, not pre-commit
- **Evidence:** the migration manifest records this script as `status: pending`
  with `new_rule: null` and `migration_issue: null`. `parity_harness.py` reported
  `No shadow migration entries configured.` A direct run passed
  (`_PluginAuditWriterAdapter has 13/20 public methods`), but no policy surface
  runs it.
- **Impact:** a facade-regression guard exists as a record but is dark. The meta
  rule prevents new unmanifested enforcers; it does not force pending enforcers
  to run.
- **Recommended action:** either wire the adapter-budget check into CI/pre-commit
  or create an elspeth-lints rule and move the manifest entry through
  `shadow -> cutover`.

### P1 — CI allowlist budgets are saturated and still dominated by permanent entries

- **File:** `config/cicd/enforce_tier_model/_defaults.yaml:160-178`,
  `config/cicd/enforce_tier_model/_defaults.yaml:224-275`
- **Record/key:** `enforce_tier_model` allowlist budget
- **Active surface:** GitHub Actions and pre-commit for policy/config changes
- **Evidence:** structured count at HEAD: `580` `allow_hits` and `92`
  `per_file_rules`, exactly matching `max_allow_hits: 580`,
  `max_per_file_rules: 92`, and `max_total_entries: 672`. Of those entries,
  `410` are permanent (`expires: null`), exactly matching
  `max_permanent_total_entries: 410`.
- **Impact:** the budget is enforcing a ceiling, but it has zero headroom and
  still carries a large permanent corpus. A scanner pass can be green while the
  exemption set remains too broad to distinguish real tier-model defects from
  institutionalized suppressions.
- **Recommended action:** ratchet permanent entries down before accepting more
  broad branch work; require new exact hits to be bounded by default; keep
  tracking owner/rule/file deltas in every CI-policy change.

### P1 — Multiple non-tier allowlists have no expiry discipline

- **File:** `config/cicd/enforce_frozen_annotations/existing.yaml:1-16`,
  `config/cicd/enforce_freeze_guards/contracts.yaml:1-35`,
  `config/cicd/symbol_inventory/migration_files.yaml:1-10`,
  `config/cicd/test_to_source_mapping/migration_files.yaml:1-14`
- **Record/key:** frozen annotations, freeze guards, ADR-019 migration
  inventories
- **Active surface:** GitHub Actions; some pre-commit surfaces
- **Evidence:** current counts include `16/16` frozen-annotation exact hits with
  no expiry, `15/15` freeze-guard per-file rules with no expiry, `2/2`
  symbol-inventory migration rules with no expiry, and `4/4`
  test-to-source mapping rules with no expiry.
- **Impact:** entries described as migration windows or pending content audit can
  become permanent governance exceptions. This weakens the same audit/tier-model
  discipline that the lints are meant to enforce.
- **Recommended action:** add expiry dates and tracker links to migration-window
  records, or reclassify truly permanent records as deliberate architecture
  exceptions with owner and review cadence.

### P2 — Telemetry-backfill allowlist is CI-only despite README claiming pre-commit parity

- **File:** `config/cicd/enforce_telemetry_backfill_trailer/README.md:46-48`,
  `.githooks/commit-msg-telemetry-backfill:88-99`,
  `.githooks/check-commit-range-telemetry-backfill.sh:30-96`
- **Record/key:** `config/cicd/enforce_telemetry_backfill_trailer/*.yaml`
- **Active surface:** GitHub Actions and commit-msg hook, but allowlist reading
  is only implemented in the CI range checker
- **Evidence:** the README says both the hook and CI backstop read every YAML
  allowlist file. The range checker implements `load_allowlist`; the local
  commit-msg hook only documents the idea and then requires trailers directly.
  The current range check over `origin/main..HEAD` passed and used all 36
  allowlisted pre-hook commits.
- **Impact:** current enforcement is acceptable for historical PR commits, but
  the policy record overstates pre-commit behavior. Future operators may add an
  allowlist entry expecting local commit-msg behavior that cannot work by
  commit SHA before the commit exists.
- **Recommended action:** correct the README to say the SHA allowlist is a CI
  backstop mechanism, or redesign local exemptions around a pre-SHA mechanism
  such as a reviewed trailer token.

### P2 — Actionlint has configuration but no enforcement surface

- **File:** `.github/actionlint.yaml:1-4`
- **Record/key:** `self-hosted-runner.labels`
- **Active surface:** configured but not wired
- **Evidence:** `actionlint` is not installed in this environment, and no
  references to `actionlint` exist in `.github/workflows/`,
  `.pre-commit-config.yaml`, `scripts/`, `config/`, `docs/`, `pyproject.toml`,
  or `package.json`.
- **Impact:** workflow syntax and runner-label policy are not checked by the
  repo's CI even though a configuration file exists. This matters because this
  repo uses self-hosted runner routing and branch-protection-required workflows.
- **Recommended action:** add an actionlint CI/pre-commit surface or delete the
  config file if the project intentionally relies on GitHub's workflow parser
  only.

### P2 — Dependabot version-update config exists, but security alerting is disabled

- **File:** `.github/dependabot.yml:1-70`
- **Record/key:** Dependabot `uv`, root `npm`, frontend `npm`,
  `github-actions`, and `docker` update records
- **Active surface:** GitHub Dependabot configuration; not a CI check
- **Evidence:** `gh api repos/johnm-dta/elspeth/vulnerability-alerts` returned
  `404 Vulnerability alerts are disabled`; automated security fixes reported
  `{"enabled":false,"paused":false}`. No dependency-labeled PRs were found.
  The CI supply-chain lane does run `pip-audit` and license checks, but that is
  separate from repository alerting and automated fix PRs.
- **Impact:** version update records exist, but GitHub's advisory alert/fix
  surfaces are not active. Operators must not treat `dependabot.yml` as proof of
  security alert coverage.
- **Recommended action:** enable vulnerability alerts and automated security
  fixes if GitHub-native advisory governance is desired, or document that
  `pip-audit` is the sole enforced dependency vulnerability gate.

## Inventory

| Record family | Records | GitHub Actions | Pre-commit / local hook | Branch protection | Status |
| --- | ---: | --- | --- | --- | --- |
| `ci.yaml` aggregate | 1 workflow, 6 required `ci-success.needs` jobs | yes | no | required through `CI Success` | active, but local HEAD has static gate failures |
| CodeQL | 1 workflow + `.github/codeql/codeql-config.yml` | yes | no | required as `CodeQL` | active; open code-scanning alerts query returned `[]` |
| Composer redaction label gate | 1 workflow + 2 scripts + structural tests | yes | no | required as `redaction-gate` | active but broken path predicate |
| Telemetry-backfill trailer | 1 workflow + 2 `.githooks` scripts + 36-entry allowlist | yes | commit-msg hook | required as `Check cohort-attribution trailers on PR commits` | active; allowlist is CI-only |
| Build/push/release | 1 workflow | yes | no | not required | active publication workflow; only checks `CI Success` |
| Mutation testing | 1 workflow | scheduled/manual | no | not required | advisory; explicitly non-gating |
| Dependabot | 5 ecosystem records | GitHub service, not workflow CI | no | not required | configured; vulnerability alerts and security fixes disabled |
| Actionlint | 1 config file | no | no | not required | configured but not wired |
| `elspeth-lints` migrated rules | 11 active rule groups | yes via `ci.yaml` | mixed whole-repo/config/changed-file hooks | inside `CI Success` | active; immutability failed at audit capture, now passes locally |
| `lint_migration_status.yaml` | 15 deleted, 1 pending, 0 shadow | parity harness runs in CI | meta rule in pre-commit on metadata changes | inside `CI Success` | pending adapter budget is dark |
| `contracts-whitelist.yaml` | 15 external types + 458 dict patterns | yes via `scripts/check_contracts.py` | yes on config/contract changes | inside `CI Success` | active; failed at audit capture, now passes locally |
| Tier-model allowlist | 580 exact + 92 per-file entries | yes | yes for policy/config changes and changed Python | inside `CI Success` | active, saturated |
| Freeze/frozen allowlists | 15 freeze per-file + 16 frozen exact entries | yes | partial | inside `CI Success` | active, no expiry discipline |
| Manifest inventories | 2 symbol + 4 test-to-source per-file rules | yes | config/rule changes | inside `CI Success` | active, migration entries lack expiry |
| Smoke test config | `config/cicd/smoke-test.yaml` | not referenced by current workflows | no | not required | documentation/config-only in current tree |

## Active Gate Debt

- `composer-redaction-gate` is required but its path predicate is stale for the
  current `tools/` package.
- Local `immutability.frozen_annotations` failure on
  `InterpretationReviewPending.sites` was repaired in this changeset.
- Local `scripts/check_contracts.py` failure on interpretation-review hash and
  tool helper dict shapes was repaired in this changeset.
- PR #39 is blocked in live GitHub by `Test (Python 3.13)` and `CI Success` on
  pushed head `62e2133f590575a0cf1b75294b6127f1e40d8bab`; `Static analysis`
  was green for that pushed head. The local static failures captured during the
  audit have been repaired and rerun successfully in this changeset.

## Dormant Or Broken Policy Debt

- `scripts/cicd/enforce_adapter_budget.py` is tracked as pending but has no
  enforcement surface.
- `.github/actionlint.yaml` is not wired into CI or pre-commit.
- GitHub's workflow registry still lists old standalone policy workflows from
  `main`, while this branch deletes them and consolidates policy under
  `ci.yaml`. These old records are live GitHub workflow records until the branch
  lands, but they are not part of this branch's PR checks.
- Dependabot version update records exist, but vulnerability alerts and
  automated security fixes are disabled.

## Exemption Corpus Audit

Every YAML exemption/allowlist record under `config/cicd/` was mechanically
loaded and checked for count, active surface, scanner result, expiry, and tracker
link evidence. "Missing tracker" below means no `elspeth-...` issue id appears
in the entry fields inspected by the audit script.

| Corpus | Entries | Active surface | Live scanner result | Missing reason | Missing expiry | Expired | Missing tracker | Corpus verdict |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| `contracts-whitelist.yaml` | 15 external types + 458 dict patterns | CI + pre-commit on contract/config changes | failed at audit capture; passes after remediation | n/a | schema has no expiry | n/a | schema has no tracker field | active; repaired in this changeset |
| `enforce_tier_model/` | 580 exact + 92 per-file | CI + pre-commit | pass | 0 | 410 | 0 | 656 | active, budget-saturated, over-broad permanent corpus |
| `enforce_frozen_annotations/` | 16 exact | CI + changed-file pre-commit | failed at audit capture; passes after remediation | 0 | 16 | 0 | 16 | active; immediate FG3 failure repaired, expiry debt remains |
| `enforce_freeze_guards/` | 15 per-file | CI + config/rule pre-commit | pass as a rule, but grouped CI invocation fails on sibling frozen rule | 0 | 15 | 0 | 15 | active, permanent corpus |
| `enforce_guard_symmetry/` | 1 per-file | CI + config/rule pre-commit | pass | 0 | 1 | 0 | 1 | active, permanent structural exemption |
| `enforce_gve_attribution/` | 2 per-file | CI + config/rule pre-commit | pass | 0 | 2 | 0 | 2 | active, permanent structural exemptions |
| `symbol_inventory/` | 2 per-file | CI + config/rule pre-commit | pass | 0 | 2 | 0 | 2 | active migration exemptions with no expiry |
| `test_to_source_mapping/` | 4 per-file | CI + config/rule pre-commit | pass | 0 | 4 | 0 | 4 | active migration exemptions with no expiry |
| `enforce_telemetry_backfill_trailer/` | 36 commit entries | CI range checker; commit-msg hook does not read allowlist | pass over `origin/main..HEAD` | 0 | 0 | 0 | 36 | active CI-only historical exception set; all expire 2026-06-30 |
| `enforce_options_metadata/` | 0 | CI + pre-commit | pass | 0 | 0 | 0 | 0 | active, no exemptions |
| `enforce_component_type/` | 0 | CI + pre-commit | pass | 0 | 0 | 0 | 0 | active, no exemptions |
| `enforce_audit_evidence_nominal/` | 0 | CI + pre-commit | pass | 0 | 0 | 0 | 0 | active, no exemptions |
| `enforce_tier_1_decoration/` | 0 | CI + pre-commit | pass | 0 | 0 | 0 | 0 | active, no exemptions |
| `enforce_composer_exception_channel/` | 0 | CI + pre-commit | pass via `composer/*` | 0 | 0 | 0 | 0 | active, no exemptions |
| `enforce_composer_catch_order/` | 0 | CI + pre-commit | pass via `composer/*` | 0 | 0 | 0 | 0 | active, no exemptions |

Tier-model composition at HEAD:

| Dimension | Highest counts |
| --- | --- |
| Rule families | `R5` 221, `R1` 199, `R6` 151, `R2` 32, `R4` 27, `R7` 15, `R9` 13 |
| Owners | `web-sessions` 87, `architecture` 77, `web-execution` 61, `composer-tools-rearchitect` 49, `composer-audit` 36, `trust-tier-maintenance` 35, `source-boundary` 33, `web-composer` 32, `bugfix` 30, `web-interpretation` 30 |
| Files | `web/sessions/service.py` 41, `web/composer/tools/generation.py` 40, `web/composer/llm_response_parsing.py` 30, `web/interpretation_state.py` 30, `web/sessions/routes/_helpers.py` 28 |
| Near-expiry | 14 entries expire `2026-07-02`; 1 entry expires `2026-06-07` |

The expiry/tracker gap is not uniform. The telemetry-backfill exceptions have
explicit short TTLs but no tracker ids; the tier-model corpus has a live budget
and some bounded entries, but `410/672` entries are still permanent; the
non-tier migration and structural per-file allowlists generally lack both
expiry and tracker linkage. None of the YAML entries were expired as of
2026-05-25.

## Live GitHub Governance

- Repository: `johnm-dta/elspeth`; default branch: `main`.
- Branch protection endpoint for `main` returned `404 Branch not protected`.
  Enforcement is via active ruleset `12348893`.
- Active ruleset `12348893` targets `~DEFAULT_BRANCH`, blocks deletion and
  non-fast-forward updates, requires pull requests, enables Copilot review on
  push, and requires status contexts:
  - `CI Success`
  - `CodeQL`
  - `Check cohort-attribution trailers on PR commits`
  - `redaction-gate`
- Actions permissions:
  - `allowed_actions: all`
  - `sha_pinning_required: true`
  - `default_workflow_permissions: read`
  - `can_approve_pull_request_reviews: false`
- Open code-scanning alerts query returned `[]`.
- Open secret-scanning alerts query returned `[]`.
- Dependabot vulnerability alerts returned disabled; automated security fixes
  returned disabled.

## Scanner Evidence

Passed locally at `080e4ecdd14dbaaabda3fdf68c4bdc594a19ef16`:

- `trust_tier.tier_model`
- `audit_evidence.nominal_base,audit_evidence.tier_1_decoration,audit_evidence.guard_symmetry,audit_evidence.gve_attribution`
- `plugin_contract.options_metadata`
- `plugin_contract.component_type,plugin_contract.plugin_hashes`
- `composer/*`
- `manifest.contract_manifest`
- `manifest.symbol_inventory,manifest.test_to_source_mapping`
- `meta.no-new-bespoke-cicd-enforcer`
- `scripts/cicd/check_slot_type_cross_language.py`
- `scripts/cicd/generate_skill_inventory.py --check`
- `scripts/cicd/enforce_adapter_budget.py` direct run
- `scripts/cicd/parity_harness.py --manifest config/cicd/lint_migration_status.yaml --root .`
- `.githooks/check-commit-range-telemetry-backfill.sh origin/main..HEAD`

Failed locally at audit-capture commit
`080e4ecdd14dbaaabda3fdf68c4bdc594a19ef16`:

- `immutability.freeze_guards,immutability.frozen_annotations`
- `scripts/check_contracts.py`

Re-run after remediation in this changeset and passed:

- `immutability.freeze_guards,immutability.frozen_annotations`
- `scripts/check_contracts.py`

## Not Checked

- `actionlint` execution: no `actionlint` binary is installed locally and the
  repo does not wire it into CI/pre-commit.
- Full Python test suite and frontend tests at local HEAD: out of scope for the
  CI-record audit; live GitHub already shows `Test (Python 3.13)` red on the
  pushed PR head.
- Build/push image publication: not executed, because this is a read-only audit.
- Dependabot alert list details: vulnerability alerts are disabled, so there
  are no alert records to inspect.

## Suggested Filigree Grouping

Do not create these unless explicitly requested.

- **P0 bug:** Redaction gate path predicate misses `web/composer/tools/**`.
- **P0 bug:** Local CI static drift — fix `InterpretationReviewPending` frozen
  annotation and `_interpretation_hash_domain_v2` contract surface.
- **P1 task:** Align build-push release proof with live ruleset-required
  contexts.
- **P1 task:** Reactivate or migrate `_PluginAuditWriterAdapter` budget guard.
- **P1 feature:** CI exemption lifecycle ratchet — expiry/tracker requirements
  for all allowlist families, not only tier-model exact hits.
- **P2 task:** Wire actionlint or remove the unused config.
- **P2 task:** Clarify telemetry-backfill allowlist semantics as CI-only or
  implement a real local-hook exemption design.
- **P2 task:** Decide whether GitHub Dependabot security alerts are part of the
  dependency governance model; enable or document replacement by `pip-audit`.
