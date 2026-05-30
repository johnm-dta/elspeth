# cicd-judge-cli → RC5.2 merge plan

**Date:** 2026-05-29
**Branch:** `feat/cicd-judge-cli-prototype` @ `e31b5a394`
**Worktree:** `/home/john/elspeth/.worktrees/cicd-judge-cli`
**Goal:** land the judge-CLI prototype into `RC5.2` in a **secure and reliable** state.
**Method of this assessment:** ground-truth verification (code + tests + git + tracker), not
reliance on issue-tracker status. Every "done" claim below was checked against the source.

Companion document: `notes/cicd-judge-cli-review-2026-05-24.md` (the 57-agent pre-merge review
this plan closes out).

---

## 1. State summary

| Area | Status |
| --- | --- |
| CRITICAL remediation (review §1–§3) | **DONE — verified** |
| Security-relevant MAJORs (review §2/§3/§5) | **DONE — verified (over-delivery, not deferred)** |
| Lint / type / honesty-gate baseline | **GREEN** (ruff, mypy, `trust_boundary.{tests,scope,tier}`) |
| tier_model enforcement gate | **RED — branch-introduced** (Blocker 1) |
| RC5.2 integration | **2 conflicts** (Blocker 2) |
| Tracker | epic `elspeth-2ed3bb0f7d`: 158/158 children closed; epic stuck `in_progress` on expired `codex` claim |

### 1.1 Verified DONE

All 12 cross-cutting CRITICAL themes from the review are fixed in code (file:line evidence):

| # | Theme | Evidence |
| --- | --- | --- |
| 1 | `temperature=0` on judge call | `core/judge.py:809` |
| 2 | served model recorded (not requested) | `core/judge.py:851` (`completion.model` → `:855`) |
| 3 | `--rule` wired + mismatch-refusal | `core/cli.py:1192,1208-1220,1271-1278` |
| 4 | HMAC sign + fingerprint/ast_path binding, verified at load | `core/allowlist.py:530-581` (sign), `:624-664` (verify), `:696-728` (fp recompute), `:731-763` (ast_path / C8-3), `:931-945` (invariant 8) |
| 5 | authenticated `--operator-override` (constant-time) | `core/cli.py:112-137`, gate `:1155-1158` |
| 6 | secrets scrubber + symlink-safe path guard | `core/source_excerpt.py:215-226,554`; call sites `cli.py:1291`, `reaudit.py:887-908,960` |
| 7 | `ast.ClassDef` short-circuit (over-suppression) | `core/ast_walker.py:31`; regression tests `test_tier_model_decorator_suppression.py:959,1069` |
| 8 | atomic + locked YAML writes | `core/atomic_io.py`; used `cli.py:2220,2299`, `rotate.py:611,671` |
| 9 | `yaml.YAMLError` handled (documented exit) | `core/allowlist_io.py:30` → `judge_coverage.py:680-681` |
| 10 | `BLOCKED` rejected at load | `core/allowlist.py:850-865` (invariant 5) |
| 11 | reaudit per-entry isolation + `JUDGE_CALL_FAILED` + `entries_dispatched` | `core/reaudit.py:730-746,1004-1018,120,302,759-765` |
| 12 | test_ref **invariant liveness** (not just raising-shape) | `rules/trust_boundary/tests/rule.py:261-337,573,229-259` |

Security-relevant MAJORs also fixed (verified): markdown-injection escaping in reaudit reports
(`reaudit.py:1661-1676`), DoS caps (`JUDGE_SURROUNDING_CODE_CHAR_LIMIT=12_000` judge.py:520,637;
`_MAX_ALLOWLIST_YAML_BYTES=5MB` allowlist.py:364; `_MAX_TEST_REF_BYTES=5MB` tests/rule.py:426),
`policy_hash` recording (judge.py:537,634,862), `trust_env=False`, path guard on `_resolve_test_ref`
(tests/rule.py:410), workflow threshold pinned + `concurrency:` + `timeout-minutes:` in
`enforce-allowlist-judge-gates.yaml`. Hostile-case tests (round-trip writer→loader, path traversal,
quartet transplant, override-auth) exist and pass. 629 relevant tests pass.

### 1.2 Cleared non-issues (investigated; do NOT re-raise as blockers)

- **"CI HMAC-key gap" in `enforce-allowlist-judge-gates.yaml`** — the C1/C3/rotation-audit gates load
  with `--allowlist-root`/`--repo-root` only, **no `--source-root`**, so they are structural/report-only
  loads that never trigger HMAC verification (`enforce-allowlist-judge-gates.yaml:120-166,251-264`).
  The only HMAC-verified (`REQUIRED`-mode) load is the `tier_model` gate in `ci.yaml`, which already has
  the key wired. **Not a blocker.**
- **8 currently-failing tests** (2 lint-unit: `test_plugin_hashes_json_mode_succeeds_on_current_codebase`,
  `test_baseline_capture_is_self_consistent`; 6 web: `test_redaction_policy_snapshot_matches_live_manifest`
  + 5 skill-prompt tests) **reproduce identically on `RC5.2` main** → inherited baseline debt, not
  branch-introduced. Flag, don't gate. (Per `feedback_fix_errors_you_encounter`: decide accept-vs-fix
  explicitly during Phase 3; they are RC5.2-wide, not this branch's regression.)

---

## 2. Blockers

### Blocker 1 — RELIABILITY (branch-introduced): tier_model gate crashes on load

**Symptom:** `elspeth-lints check --rules trust_tier.tier_model --root src/elspeth` exits 1 with
`file_fingerprint mismatch` before producing findings (local AND `ci.yaml`).

**Precise scope (enumerated across all of `config/cicd/enforce_tier_model/`):** exactly **5** entries
carry `file_fingerprint`; **all 5 mismatch live source; all 5 are SIGNED** (carry
`judge_metadata_signature`). They are the **dogfood** entries from `df3463583`, covering 3 files —
`web/execution/service.py`, `web/composer/tools/_common.py`, `web/composer/tools/generation.py` (×3) —
which were edited on this branch *after* the entries were signed.

**Why it is operator-gated:** `file_fingerprint` is part of the HMAC-signed payload
(`allowlist.py:560`). A stale fingerprint cannot be corrected without re-signing, which needs
`ELSPETH_JUDGE_METADATA_HMAC_KEY`. The entries are **load-bearing** (they suppress real R1 findings),
so deletion is not free — the findings would re-fire.

**Chosen approach (operator decision, 2026-05-29): root-cause the refresh tool BEFORE fixing.**
HEAD commit `54cc4a6fd "chore(cicd): refresh judge allowlist fingerprints"` rewrote `web.yaml`
(−588 lines) and deleted `fingerprint_baseline.json` (−560 lines), yet the signed `file_fingerprint`
values are still stale at HEAD. Do not blindly re-sign; first determine why the refresh produced/left
wrong fingerprints, because that is a defect against the branch's own integrity tooling and will recur.

**Leading hypothesis (to confirm, not assume):** the refresh recomputed the finding-level `fp=`
(in the entry key) but did **not** recompute/re-sign the `file_fingerprint` field — because re-signing
needs the HMAC key a `chore` step wouldn't hold — so `file_fingerprint` stayed bound to the
dogfood-era bytes while the source kept changing. Alternative to rule out: the refresh hashed against
a different root/CWD (e.g. main's checkout) or different normalization than the load-time
`_compute_file_fingerprint` (plain `sha256(read_bytes())`, `allowlist.py:516`).

**Root-cause investigation steps:**
1. Read the refresh implementation (`rules/trust_tier/tier_model/rotate.py` + the `rotate` CLI path)
   and confirm whether it touches `file_fingerprint` / `judge_metadata_signature` at all, or only the
   finding-level `fp=`.
2. Inspect `54cc4a6fd` diff for the 5 signed entries: did `file_fingerprint` change in that commit?
   If unchanged, the refresh never recomputed it (hypothesis confirmed).
3. Check `.elspeth/rotations.log` entry added by `54cc4a6fd` for what it claimed to do.
4. Confirm the load-time hash is path/CWD-independent (it reads `source_root / relpath`); rule out a
   root mismatch between refresh-time and load-time.

**Fix (after root cause, sequenced AFTER the RC5.2 merge so bytes are final):** re-justify/re-sign the
5 entries against post-merge bytes with the HMAC key, OR (if the tool is the cause) fix the refresh
tool to recompute+re-sign `file_fingerprint` and add a regression test that a refreshed signed entry
loads clean. Migration of these 5 to `@trust_boundary` decorators (Pillar B) remains a future option
but is out of scope for the merge.

### Blocker 2 — INTEGRATION: 2 merge conflicts with RC5.2

`feat/...` is 29 ahead / 3 behind RC5.2 (merge-base `f77096d4e`). `git merge-tree` predicts exactly
**2 content conflicts**, both caused solely by RC5.2 behind-commit `1d5e5ac53` ("Fix duplicate
prompt-shield fact surfacing"):

- `src/elspeth/web/composer/tools/sessions.py`
- `src/elspeth/web/interpretation_state.py`

(The other two behind-commits — `9eaec9520`, `40193c02e` — do not overlap branch files.
`ci.yaml` has NOT diverged on RC5.2 since the merge-base, so the workflow edits apply cleanly;
the new `enforce-allowlist-judge-gates.yaml` is net-new.) Conflicts are small (RC5.2 reworked
`interpretation_state.py` −56/+16; feature added +37/−13 in the same prompt-shield region).
**Resolution must confirm the feature's additions do not reintroduce the duplicate-fact-surfacing
bug `1d5e5ac53` fixed.**

---

## 3. Should-do for a *secure* merge (not strict blockers)

- **Engine/contracts scope review.** The branch touches `engine/orchestrator/core.py`,
  `engine/orchestrator/types.py`, `engine/processor.py`, and `contracts/{__init__.py,
  config/protocols.py, config/runtime.py, contexts.py, plugin_context.py}` — merges clean against
  RC5.2 but is unexplained for a "CICD judge CLI" branch. A secure merge wants a focused functional
  diff read of these (intent + correctness), not just the conflict check they already pass.
- **SSH-excerpt historical artifact audit** — `elspeth-ae32d3537d` (P2, OPEN, sibling epic
  `elspeth-41c92e76e2`). The scrubber fix (T8c/T8d) closed the live code gap, but reaudit YAML
  *generated during the vulnerable window* (`a726fae44` → T8c) in dev/staging may contain unredacted
  SSH key bodies. Confirm absent or remediate. Concerns on-disk artifacts, not the merged code.
- **Minor follow-up (non-gating):** `allowlist_io.iter_yaml_documents` lacks the 5 MB `st_size` cap its
  sibling loader `allowlist.py:364` applies (asymmetric; weak threat model — repo-controlled files).

## 4. Tracker hygiene

- Release the expired `codex` claim on `elspeth-2ed3bb0f7d` (claim expired 2026-05-27) and close the
  epic (158/158 children closed).
- Close grooming ticket `elspeth-96640cb37b`; note the cosmetic child-count discrepancy
  (description "~159" vs actual 158) is non-load-bearing.
- `elspeth-b0125a607d`'s premise ("no codex claim exists on the judge umbrella") is now stale — the
  umbrella IS codex-claimed; reconcile when actioning.

---

## 5. Sequenced execution plan

| Phase | Action | Gate / exit |
| --- | --- | --- |
| 0 | **Root-cause Blocker 1's refresh-tool defect** (steps in §2). Decide fix shape. | Root cause documented; fix approach chosen. |
| 1 | **Sync `RC5.2` → feature in the worktree** (`git merge RC5.2`), resolve the 2 web conflicts, run `tests/unit/web/...`. (Do FIRST so fingerprints bind final bytes.) | Conflicts resolved; no prompt-shield regression; web tests pass (modulo the 6 inherited reds). |
| 2 | **Fix the 5 dogfood entries** against post-merge bytes (re-sign with HMAC key, or fix-the-tool-then-refresh per Phase 0). | `tier_model` gate loads clean; HMAC signatures valid. |
| 3 | **Full green verification** in the worktree: `tier_model` + judge gates + relevant tests + ruff + mypy. | Only reds are the 8 inherited RC5.2-wide failures; explicit accept-or-fix decision recorded. |
| 4 | **Engine/contracts focused functional review** (§3). | Changes confirmed intended + correct, or issues filed. |
| 5 | **SSH artifact audit** (operator) + **tracker hygiene** (§4). | `ae32d3537d` resolved; epic closed. |
| 6 | **Merge feature → RC5.2 `--no-ff`** (per `feedback_prefer_no_ff_merges`); bare push (non-destructive). Ensure RC5.2 CI has `OPENROUTER_API_KEY`, the `["self-hosted",…,"trusted"]` runner label, and the HMAC key for `ci.yaml`. | RC5.2 CI green; merge landed. |

### Operator-gated / decision points

- **HMAC key** (`ELSPETH_JUDGE_METADATA_HMAC_KEY`) — needed for Phase 2 re-signing; location TBD.
- **Dogfood fix shape** — pending Phase 0 root cause (re-sign vs fix-tool vs future decorator migration).
- **SSH artifact audit** — operator-run/decided.
- **Merge authorization** (Phase 6) — explicit operator go.

### CI prerequisites on RC5.2 (for the new workflow to function)

- `secrets.OPENROUTER_API_KEY` (the `check-judge-quality` VAL job hard-fails without it).
- A self-hosted runner labelled `["self-hosted","Linux","X64","nyx-ci","trusted"]`.
- `ELSPETH_JUDGE_METADATA_HMAC_KEY` available to `ci.yaml`'s `tier_model` job (already wired in the
  branch's `ci.yaml`; must exist as a repo/org secret).
