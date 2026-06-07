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

## 1b. Progress log (verify-then-fix sweep)

**Closed 2026-06-07 (this sweep), each with TDD locking test + 3-hop audit trail:**

| Bug | Stream | Commit | Resolution |
|-----|--------|--------|------------|
| elspeth-06913f39d4 | fail-open CI (LIVE gate) | 3a8f37f23 | ci.yaml integration lane `\|\| echo` swallowed every pytest exit → capture status, tolerate ONLY exit 5 (no tests collected), propagate the rest |
| elspeth-0707b6d15b | fail-open (operator tool) | 66d3cba26 | `_structured_findings` trusted any present sidecar → now authoritative only when mtime >= report; else warn + parse Markdown |
| elspeth-4634ee39ee | fail-open (operator tool) | 66d3cba26 | 3 codex runners returned 0 after per-target failures → `exit_code_from_stats()` + `summary["failed"]`; codex_bug_hunt.py GONE (migrated) |
| elspeth-118bf5ea8c | CI build-push | 4dc2b08d2 | smoke-test hardcoded GHCR pull → registry-aware SMOKE_IMAGE (GHCR pref, ACR fallback) + per-registry logins + skip-if-nothing-pushed |
| elspeth-8cb798c3fd | CI build-push (DUP of 118bf5ea8c) | 4dc2b08d2 | same fix |
| elspeth-c36485165b | AST: gve_attribution (MD1 alias) | 9b9f22fd6 | matcher keyed on bare name → now collects ImportFrom aliases; blast radius ZERO |
| elspeth-16f41371f8 | AST: gve_attribution | 9b9f22fd6 | component_id keyword counted by presence → now flags component_id=None literal; blast radius ZERO |
| elspeth-3c73e49cc7 | AST: gve_attribution (MD4 SUPERSEDED) | (test-only) | framework CLI emits parse-error for SyntaxError → fail-closed; locking test added |
| elspeth-a586a7212e | AST: frozen_annotations (MD2) | b67f4a56e | regex missed capitalized typing aliases List[/Dict[/Set[ → AST walk; blast radius ZERO |
| elspeth-fbfb9fd634 | AST: frozen_annotations (MD2) | b67f4a56e | regex required `[` so bare list/dict/set missed → AST walk; only tightens (nested-in-immutable still flagged) |
| elspeth-20add2bd90 | AST: component_type (MD1 alias) | 9ad914feb | `_extract_base_names` saw only syntactic base → now resolves `import Base as Alias` via per-file alias map; resolved name APPENDED not substituted (never-loosen by construction); blast radius ZERO |
| elspeth-a2b240c29b | AST: component_type (MD5 spoof) | 9ad914feb | `_class_sets_component_type` accepted any str → now restricted to {source,sink,transform}; blast radius ZERO (all real labels valid). Runtime sibling filed elspeth-ce0814e726 (config_base.__init_subclass__ only checks is-None — engine/platform cluster, deferred) |
| elspeth-1e8f4ece9a | AST: contract_manifest (MD1 provenance) | 2cdec9875 | `_is_register_call` trusted textual name → now requires import from elspeth.contracts.declaration_contracts (CanonicalBindings); shadowed no-op → MC2. Fail-open closed |
| elspeth-487dfef2ce | AST: contract_manifest (MD1 provenance) | 2cdec9875 | same provenance gate for @implements_dispatch_site decorator; shadowed no-op → MC3b. Fail-open closed |
| elspeth-07d9f8a619 | AST: contract_manifest (dup) | 2cdec9875 | `compute_findings` set-deduped → now emits MC1 on duplicate contract name (key name::duplicate@L<line>), matching runtime ValueError. Fail-open closed |
| elspeth-2b5edd369e | AST: contract_manifest (FP) | 2cdec9875 | marker site read from args[0] only → now positional OR site_name= keyword; removes spurious MC3b on valid keyword form |
| elspeth-f2959957fc | AST: freeze_guards FG2 | f814fdbdb | isinstance guard matched bare Name only → now qualified Attribute (types.MappingProxyType/collections.abc.Mapping) + list/set, bare & tuple forms. Blast ZERO (no such form in real __post_init__) |
| elspeth-ec6749f8da | AST: freeze_guards FG3 nested | f814fdbdb | `_annotation_contains_container` outer-name only → recurses tuple/frozenset carriers (carrier-gated: NOT Callable), strictly additive. Surfaced 1 real (_OutputBlobFinalizationOutcome.errors → elspeth-640168fa4b, allowlisted) |
| elspeth-b872929157 | AST: freeze_guards FG3 partial | f814fdbdb | any-freeze-call → per-field coverage (freeze_fields names OR object.__setattr__ freeze-producing RHS; dynamic *splat = covers-all). 4 real partial cases all covered via object.__setattr__(tuple()) → blast ZERO |
| elspeth-b8b600e213 | AST: tier_model L1 layer | 3d47fdb66 | scan_layer_imports_file collected only ImportFrom level==0/node.module → now _resolve_relative_module (relative) + bare-elspeth package-root disambiguated via _module_name_to_path (plugins=flag, __version__=skip). Per-node emission. Blast ZERO (real L1=0) |
| elspeth-b7ef37c4a9 | AST: tier_model TC (FP) | 3d47fdb66 | _find_type_checking_lines direct-children-only → recurse node.body (NOT orelse). Nested-TC try/import → TC not L1. Real TC invariant=1 preserved |

**Ergonomics win 2 — e7ff99c39:** ruff now `extend-exclude`s `rules/**/fixtures`
(mirrors mypy). Adding a fixture using the deprecated `List[int]` form tripped
ruff UP006/UP035 (wanted to auto-upgrade it to `list[int]` — defeating the very
fixture for a586a7212e). Same root cause as the mypy hook fix: pre-commit passes
changed files explicitly, bypassing the tool's dir-exclude. **General lesson for
this sweep: adding lint-rule fixtures that intentionally use deprecated/old forms
needs BOTH mypy + ruff fixture-excludes (now both in place).**

**Ergonomics win (goal half 2) — committed 1a90a0695:** mypy pre-commit hook now
mirrors pyproject's `rules/.*/fixtures/` exclude. Without it, adding 2+ fixture
`bad.py` files in one commit fails mypy with "Duplicate module named 'bad'"
(hook passes changed files explicitly, bypassing the dir-exclude). Type-checking
deliberately-malformed fixtures has no safety value → no roadblock reduced. This
unblocks future multi-fixture rule commits in this very sweep.

**Worktree gotcha REPRODUCED + characterised:** gve fixtures (WHOLE_REPO) run
through scan_root → `allowlist_path_for_root` escapes to the real config/cicd via
cwd → spurious `actual:[]` in the harness; the PRE-EXISTING 01_missing_component_id
fails identically. Proof method that works: `scan_root(dir, allowlist_dir_override=<empty>,
emit_allowlist_governance=False)` matches expected exactly. For ANY WHOLE_REPO
allowlist-loading rule, verify with controlled override, NOT the full harness.

**Pattern for AST stream (reusable):** (1) read rule.py matcher; (2) blast-radius
grep/AST-scan src/elspeth for the missed form — ZERO hits = safe pure-hardening fix,
HITS = park `blocked` (operator allowlist) unless fixable in scope; (3) hermetic
unit test via `RULE.analyze(_tree(src), Path("x.py"), ctx)` (routes to scan_tree,
no allowlist) — red-proof against committed rule via `git show HEAD:rule.py` loaded
with `sys.modules[name]=mod` before exec (slotted dataclass needs it); (4) fixtures
+ generate expected.json with `PYTHONPATH=elspeth-lints/src` (NOT raw python — that
hits main's stale install); bump examples_violation_count.

**Method confirmed working:** verify-against-current-reality first. codex_bug_hunt.py
and the enforce_*.py scripts are gone; bugs citing them are partial-superseded —
fix the surviving instances, note the migration in root_cause. Workflow-YAML bugs
lock via tests/unit/test_ci_workflow_xdist.py + test_build_push_release_checks.py
(parse YAML, assert on steps); shell logic simulated locally with `bash -eo pipefail`.

**Remaining ~40 (streams):** CI-workflow (a57c4bd228 Dockerfile frontend-dist [+subsys:web],
a7afa79003 xdist spaced-flag guard, a1f2ef0f82 redaction-gate docs, 0def0c0404 telemetry
backfill, 313cd53771 CI policy inventory); AST-matcher ~26 (fixture discriminator +
BLAST-RADIUS check → park as `blocked` if it surfaces real in-tree violations needing
operator-HMAC allowlist); docs ~4 (refs to deleted scripts); plugin-hash 3 (still `triage`,
need --advance). **Highest-value next:** MD1 alias-evasion via gve_attribution
(c36485165b) per §4.

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

## 3c. contract_manifest family — DONE (2cdec9875, all 4 closed)

Provenance lesson generalised: name-identity ≠ provenance. The fix added a per-file
`CanonicalBindings` map (register_names / marker_names / module_aliases sourced ONLY from
`from elspeth.contracts.declaration_contracts import ...`). Proof model for THIS family was
**real-codebase finding-set invariance** (HEAD==fixed==[] on src/elspeth), NOT old⊆new — the
provenance/keyword fixes intentionally change the synthetic verdict (advisor catch). Added a
json-mode real-scan test (the rule had none, unlike component_type). Original analysis ↓ kept for
reference.

### (original analysis)

Rule: `rules/manifest/contract_manifest/rule.py`. Four bugs, coupled — land as ONE pass:

- **1e8f4ece9a (MD1, fail-OPEN):** `_is_register_call` (L225) matches any Name/Attribute
  whose final name == `register_declaration_contract`, ignoring provenance. A local no-op
  `def register_declaration_contract(x): pass` is accepted → CI certifies a fake registration.
- **487dfef2ce (MD1, fail-OPEN):** `_dispatch_site_marker` (L327) checks `_call_name(decorator)
  == _DECORATOR_NAME` textually. Local no-op `implements_dispatch_site` is accepted as a real marker.
- **07d9f8a619 (fail-OPEN):** `compute_findings` (L362,377) tracks `registered_names_found` as a
  SET, never flags two registrations sharing a `name`. Runtime `register_declaration_contract`
  raises ValueError on dup → CI passes a tree bootstrap would reject. Fix: detect duplicate
  contract_name across `registrations`, emit a finding (reuse MC1 or add a sub-id — check metadata.py).
- **2b5edd369e (P3, false-POSITIVE):** `_dispatch_site_marker` only reads `decorator.args[0]`;
  runtime decorator accepts `site_name=` kwarg → keyword-form marker missed → spurious MC3b. Fix:
  also read the `site_name` keyword. (FP removal — ensure real missing-marker detection unaffected.)

**Fix shape:** add a per-file import-provenance map (alias→canonical) for both
`register_declaration_contract` and `implements_dispatch_site`, sourced from
`elspeth.contracts.declaration_contracts`. **OPEN QUESTION — blast radius FIRST:** how does real
src/elspeth import/call these? If direct `from ...declaration_contracts import register_declaration_contract`
and bare-name calls, a provenance requirement is safe (zero FN). If Attribute-form (`module.register_...`),
the map must resolve `import ... as module` too. Grep before tightening; ZERO unresolvable real calls = safe.
Per-bug TDD + fixtures + red-proof against HEAD, same method as component_type (9ad914feb).

## 4. Next session (per advisor: prove a slice, commit, continue)
- MD1 vertical slice: pick `gve_attribution` aliased-raise
  (elspeth-c36485165b) — `graph_validation_error_call` already handles
  `ast.Attribute` but not `import ... as` aliases. Add `examples_violation`
  fixture, confirm live, fix, close.
- Then MD2, MD5; verify MD4 superseded via fixture; then the consolidation refactor.
