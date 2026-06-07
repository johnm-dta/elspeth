# RESUME PROMPT — CICD scanner-cluster bug sweep

Paste the block below to resume. Full detail lives in
`notes/cicd-scanner-cluster-2026-06-07.md` (§1b = progress log + reusable pattern).

---

Continue the `subsys:cicd` bug sweep in the **existing worktree**
`/home/john/elspeth/.worktrees/cicd-scanner-bugs` (branch `fix/cicd-scanner-bugs`,
local-only, NOT pushed/merged — operator owns that decision). Actor for filigree
calls: `claude-cicd-scanner` (ACTOR_MISMATCH warning is benign).

## The reframe (why this is tractable)
The ~48 `subsys:cicd` bugs (parent epic `elspeth-297b8f5c5d`) were filed 2026-05-13
against bespoke `scripts/cicd/enforce_*.py` scripts that **no longer exist** —
migrated into the `elspeth-lints/` package. They collapse into ~5 meta-defect
classes. So: **verify-against-current-reality before fixing.** Many bugs are
partially or fully superseded by the migration.

## Done so far (13 closed; do NOT redo)
MD3 (3, sweep 1) + this sweep (10): integration-lane fail-open (06913f39d4),
codex sidecar+partial-scan (0707b6d15b/4634ee39ee), build-push smoke dup-pair
(118bf5ea8c/8cb798c3fd), gve_attribution alias/None/syntax (c36485165b/16f41371f8/
3c73e49cc7), frozen_annotations MD2 pair (a586a7212e/fbfb9fd634). Plus 2 ergonomics
fixes: mypy (1a90a0695) + ruff (e7ff99c39) now both `exclude` rules/**/fixtures.

## Remaining (~35) — verify each, then fix-or-supersede-or-block
- **AST matchers (next):** component_type (20add2bd90 aliased subclasses, a2b240c29b
  spoofable string), contract_manifest (1e8f4ece9a/487dfef2ce shadowed regs,
  07d9f8a619 dup names, 2b5edd369e keyword-form), freeze_guards (b872929157 FG3
  partial, ec6749f8da FG3 nested mutable, f2959957fc FG2 qualified), tier_model
  (b8b600e213 upward imports, b7ef37c4a9 TYPE_CHECKING FP), guard_symmetry
  (697788a39f), audit_evidence_nominal (37879426d1 annotated to_audit_dict,
  584d4ea502 spoofed base name — MD5), tier_1_decoration (08b0336287 empty reason,
  f8650893f1 TDE2 scope), composer catch_order (c0c4f49981 aliased, eb90341cdb broad
  Exception), exception_channel (9aaa21d81c qualified/aliased), check_contracts
  (4f2471815e substring FP, 369a13a173 qualified defs, 739409bfa1 dict[str,Any]),
  check_slot_type_cross_language (83e7922d40).
- **CI-workflow/script:** a57c4bd228 (Dockerfile frontend dist — bigger, multi-stage
  npm build + .dockerignore; also subsys:web/frontend), a7afa79003 (xdist spaced-flag
  guard — test_ci_workflow_xdist.py), a1f2ef0f82 (redaction-gate branch-protection
  docs), 0def0c0404 (telemetry backfill allowlist), 313cd53771 (CI policy inventory).
- **Docs (refs to deleted scripts):** 98e02d5f0f, 4669a37774, 870ee03711, f91f5fbce7.
- **Plugin-hash (still `triage` — need `--advance`):** 93ee280da5, 4aec95f5b0, c61f4a397c.

## Method (reusable, proven)
1. Read the rule's `rule.py` matcher under
   `elspeth-lints/src/elspeth_lints/rules/<cat>/<rule>/`.
2. **Blast-radius FIRST**: AST-scan `src/elspeth` for the missed form. ZERO hits =
   safe pure-hardening fix. HITS you can't fix in scope = park `blocked` (operator
   HMAC allowlist), do NOT close. **NEVER loosen detection** — a tightening fix must
   keep every old flag (check old-minus-new, not just new-minus-old).
3. Hermetic unit test: `RULE.analyze(_tree(src), Path("x.py"), ctx)` routes to
   `scan_tree` (no allowlist) — this is the real local proof. Red-proof against the
   committed rule via `git show HEAD:<rule.py>` loaded with
   `sys.modules[name]=mod` before `exec_module` (slotted dataclass needs it).
4. Fixtures (for CI): add `examples_violation/NN_x/bad.py` + generate
   `NN_x.expected.json` with `env -u VIRTUAL_ENV PYTHONPATH=elspeth-lints/src
   .venv/bin/python` (NOT raw python — that hits main's stale install); bump
   `examples_violation_count` in metadata.py. Verify via hermetic
   `scan_root(dir, allowlist_dir_override=<empty tmp>, emit_allowlist_governance=False)`
   — the full fixture harness fails spuriously in-worktree (allowlist cwd-escape;
   pre-existing fixtures fail identically — that's the documented gotcha, not you).
5. Run tests: `env -u VIRTUAL_ENV .venv/bin/python -m pytest <file> -p no:xdist -q`.
6. Commit (hooks run; co-author trailer "Claude Opus 4.8 (1M context)
   <noreply@anthropic.com>"). Doc-only commits may use `--no-verify`.
7. Close: `work_start` (→fixing) → `issue_update status=verifying` with `root_cause`
   → `issue_close` with `fix_verification` + reason. (confirmed→fixing→verifying→
   closed; the multi-hop is required.) For superseded bugs, close with a locking test
   that proves the framework already covers it — never close blind.

## Discipline (non-negotiable)
- Every close rides an EXECUTED test, never an eyeballed matcher read.
- Mark duplicates explicitly (one fix, both closed, cross-referenced).
- The "roadblock": fixes must fail-CLOSED and never reduce operator-attention gates;
  ergonomics changes (auto-roll source hashes, fixture-excludes) are fine but NEVER
  auto-roll baseline/allowlist/signature snapshots.
- Update notes §1b progress table + memory after each chunk; keep MEMORY.md index
  line short.

Work until all remaining bugs are resolved or blocked.
