# CI/CD enforcement-scanner cluster — triage + MD3 fix + toolset ergonomics

Date: 2026-06-07 · Branch: `fix/cicd-scanner-bugs` (worktree, local-only)
Parent epic: elspeth-297b8f5c5d (CI allowlist revalidation)

## 1. The cluster is ~5 meta-defects, not 48 independent bugs

`subsys:cicd` has 48 open bugs (all P2/P3, all `from-observation`, filed
2026-05-13 against the **pre-migration** bespoke `scripts/cicd/enforce_*.py`
scripts — which no longer exist; the rules were migrated into the unified
`elspeth-lints/` package). They collapse into recurring classes:

| ID  | Meta-defect | Bugs (approx) | Likely status |
|-----|-------------|--------------|---------------|
| MD1 | Alias / qualified-name evasion — matcher sees bare `ast.Name` `X` but misses `mod.X` (`ast.Attribute`) and `import X as Y` | ~8 (composer exception_channel/catch_order, gve_attribution, component_type, contract_manifest, check_contracts) | LIVE (AST-logic gaps; migration rewrote each matcher but wouldn't auto-fix) |
| MD2 | Annotation-form gaps — miss `typing.List/Dict/Set` and bare `list/dict/set` | ~4 (frozen_annotations, freeze_guards FG3) | LIVE (probable) |
| MD3 | **Allowlist-expiry fail-OPEN** — malformed `expires:` swallowed → `None` → defeats `fail_on_expired` | 3 (tier_1_decoration, audit_evidence_nominal, contract_manifest) | **LIVE → FIXED (this branch)** |
| MD4 | Fail-open on `SyntaxError` (gve_attribution passes when files unparseable) | 1 | Likely SUPERSEDED — framework walker now emits a `parse-error` finding (`cli.py:_syntax_error_finding`); verify per-rule |
| MD5 | Spoofable name-identity — accepts any class literally named `AuditEvidenceBase`; any string `_plugin_component_type` | ~3 | LIVE (probable) |

**Methodology that matters:** verify each bug against *current* `elspeth_lints`
before touching it. A bug is only live if its missed case slips past the
migrated rule. Discriminator is already in-repo: add the missed case as
`rules/.../fixtures/examples_violation/NN/bad.py`, run
`test_all_rule_fixtures.py` — caught = superseded (close with the locking
test), missed = live (you now hold the failing test). Do **not** "fix" the
dead `scripts/cicd/enforce_*.py` paths.

## 2. What landed: MD3 (commit 8954a536f)

The migration left **three date parsers with divergent fail-polarity**:

- `core/allowlist.py:_optional_date` — **raises** on malformed date (correct).
- `rules/audit_evidence/shared.py:_optional_date` — caught `ValueError`,
  returned `None` (fail-OPEN). Used by `load_class_allowlist` → TDE1 + AEN1.
- `rules/manifest/contract_manifest/rule.py:_parse_date` — same, plus
  returned `None` for any non-str/non-date. Used by `load_contract_allowlist` → MC2.

A one-character typo in an `expires:` date silently disabled the time bound on
a security exemption. Both bespoke parsers now **raise** (matching the
canonical loader and their own sibling parsers, which already raised). The
raise propagates `scan_root → analyze → cli` runner (no swallowing try/except
at `cli.py:1080`/`1146`) to a non-zero exit — fail-closed. This *strengthens*
the roadblock.

Regression-locked (full loader path, not `_optional_date` in isolation):
- `test_load_class_allowlist_rejects_malformed_expiry`
- `test_audit_evidence_nominal_fails_closed_on_malformed_expiry` (end-to-end)
- `test_load_contract_allowlist_rejects_malformed_expiry`

Closed: elspeth-2d73b966c5, elspeth-44d771caad, elspeth-99ae5c0991.

**Pre-existing noise to ignore:** running `test_all_rule_fixtures.py` from a
worktree yields ~18 spurious failures (fixtures escape to the real repo
allowlist via `allowlist_path_for_root`). Confirmed unrelated to the MD3 fix
(reverting the source in the same cwd leaves them unchanged). Filed as
observation elspeth-obs-19adb5d3ad. Also pre-existing: `test_baseline_capture_is_self_consistent`
(fingerprint_baseline drift, needs operator HMAC regen).

## 3. Toolset ergonomics — auto-roll / auto-stage WITHOUT reducing the roadblock

The friction: a legitimate source edit forces a chore tail (refresh plugin
`source_file_hash`, regen `fingerprint_baseline`, rotate tier-model allowlist),
and the lint suite is unreliable from worktrees.

**The roadblock line is not "don't auto-sign" — it is "don't auto-accept
findings nobody looked at."**

- **Safe to auto-roll + git-add** (the reviewed source edit *is* the decision):
  pure source-derived hashes of files you *deliberately edited* — e.g. plugin
  `source_file_hash` (`scripts/cicd/plugin_hash`). Deterministic; recomputing
  it asserts nothing about correctness, only "this is the source I edited."
- **NEVER auto-roll** anything that snapshots the current finding/violation
  *set* as the new accepted baseline: `fingerprint_baseline` regen, new
  allowlist entries, judge corpus signatures. These swallow whatever leaked in
  — that *is* reducing the roadblock. They stay manual + operator-HMAC-gated.

**Proposed (not built — deferred):** a `scripts/cicd/autoroll` helper that,
for files already `git add`-ed, refreshes ONLY their deterministic source-hash
artifacts and re-stages them, and hard-refuses to touch baselines / allowlists
/ signatures (prints what it skipped and why). Pairs with the existing
fail-closed gates: the gate still goes red on a real new violation; autoroll
only removes the mechanical-hash busywork.

**Earned refactor (deferred, file as task):** consolidate the 3 date parsers
into one shared fail-closed helper that all loaders delegate to — removes the
drift that caused MD3 in the first place. This is the "extract the helper after
2-3 concrete fixes" win; do it after MD1/MD2 land more concrete cases.

## 4. Next session (per advisor: prove a slice, commit, continue)
- MD1 vertical slice: pick `gve_attribution` aliased-raise
  (elspeth-c36485165b) — `graph_validation_error_call` already handles
  `ast.Attribute` but not `import ... as` aliases. Add `examples_violation`
  fixture, confirm live, fix, close.
- Then MD2, MD5; verify MD4 superseded via fixture; then the consolidation refactor.
