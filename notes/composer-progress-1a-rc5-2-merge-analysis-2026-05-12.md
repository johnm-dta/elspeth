# Merge Analysis — `feat/composer-progress-persistence-1a` → `RC5.2`

**Date:** 2026-05-12
**Branches compared:**
- This branch HEAD: `f5bc5dbb` (`docs(plan): Phase 3 plan rev-2 — re-baseline against Phase 2 as shipped (f54ee7e8)`)
- RC5.2 HEAD: `3e46a976` (`docs(plan): fold post-review errata (rev 2) into composer guided-mode plan`)
- Merge base: `91133f32`
**Worktree path:** `/home/john/elspeth/.worktrees/composer-progress-1a`
**Author:** Plan/merge analysis session, 2026-05-12. Reality-checked separately (see §10).

---

## 1. TL;DR

The merge is **moderately difficult but bounded**. Phase 1+2 (the substantive Phase-1A/1B/1C plus Phase-2 redaction work currently on this branch) can land on RC5.2 in a "good shape" — schema additions are inert, primitives are dormant production code, the redaction walker is unused by `_compose_loop` — provided **one prerequisite** is satisfied first:

> **Prerequisite:** reconcile this branch's *replacement-shape machinery* against RC5.2's commit `2c043a11 refactor(composer): remove dead replacement-shape machinery`. Four symbols deleted on RC5.2 are still live on this branch; the textual auto-merge would produce a clean-but-semantically-broken file (RC5.2's deletions land, this branch's callers remain).

After reconciliation, the merge ships ~half-day of textual conflicts plus a full gate rerun. Phase 3 (planned but unimplemented) is unaffected — it lives on this branch or a successor and rebases onto post-merge RC5.2.

**Timing recommendation:** merge now (before Phase 3 implementation). Doing it later compounds the conflict surface because Phase 3 modifies the same files that already have RC5.2 deltas.

---

## 2. Quantitative divergence shape

| Dimension | Value |
|---|---|
| Commits ahead (this branch vs RC5.2) | **97** |
| Commits behind (RC5.2 since fork) | **68** |
| Merge base | `91133f32` |
| Files modified on this branch since fork | 126 |
| Files modified on RC5.2 since fork | 93 |
| **Files touched by both sides (overlap candidates)** | **17** |
| Textual conflicts (per `git merge-tree --write-tree` dry-run) | **5** |
| Textual auto-merges in critical composer/sessions cluster | **6** |

Verification commands:

```bash
git rev-list --left-right --count feat/composer-progress-persistence-1a...RC5.2
# → 97  68

git merge-base feat/composer-progress-persistence-1a RC5.2
# → 91133f32f27ebf7284a35979ce5da8620153a62e

comm -12 \
  <(git diff --name-only 91133f32..feat/composer-progress-persistence-1a | sort) \
  <(git diff --name-only 91133f32..RC5.2 | sort)
# → 17 files (see §3)

git merge-tree --write-tree --name-only --messages \
  feat/composer-progress-persistence-1a RC5.2
# → 5 CONFLICT lines (see §4)
```

---

## 3. The 17 overlapping files

Files modified on both sides since `91133f32`:

```
config/cicd/contracts-whitelist.yaml
config/cicd/enforce_tier_model/web.yaml
pyproject.toml
src/elspeth/web/composer/service.py
src/elspeth/web/composer/tools.py
src/elspeth/web/execution/routes.py
src/elspeth/web/execution/schemas.py
src/elspeth/web/sessions/models.py
src/elspeth/web/sessions/protocol.py
src/elspeth/web/sessions/routes.py
src/elspeth/web/sessions/service.py
tests/unit/evals/lib/test_composer_rgr_score.py
tests/unit/web/composer/test_service.py
tests/unit/web/composer/test_tools.py
tests/unit/web/sessions/test_models.py
tests/unit/web/sessions/test_routes.py
uv.lock
```

### 3.1 Diff stats — critical files only

This branch's contribution since fork:

```
src/elspeth/web/composer/tools.py  | 668 net  (+692 -208)  [Phase 2 promotion wave + 38-entry MANIFEST plumbing]
src/elspeth/web/sessions/models.py | 232 net  [Phase 1A schema additions]
```

RC5.2's contribution since fork:

```
src/elspeth/web/composer/service.py  | 386 net  (+243 -143)  [4 commits — see §5.1]
src/elspeth/web/composer/tools.py    | 221 net               [grounding-related]
src/elspeth/web/sessions/routes.py   | 103 net (mostly -)    [single commit 2c043a11]
src/elspeth/web/sessions/models.py   |  24 net (mostly -)    [single commit 0964922c — release stamp]
src/elspeth/web/sessions/protocol.py |  18 net               [single commit 2c043a11]
src/elspeth/web/sessions/service.py  |   5 net (-)           [single commit 2c043a11]
```

**Concentration observation:** RC5.2's sessions-side risk concentrates in **one commit**: `2c043a11`. That commit alone touched sessions/routes.py, sessions/protocol.py, sessions/service.py, plus composer/service.py. The composer-side risk is split across 4 commits with `2c043a11` dominating.

---

## 4. Textual conflicts (cheap to resolve)

`git merge-tree --write-tree` dry-run reports 5 CONFLICTs:

| File | Conflict character | Resolution |
|---|---|---|
| `config/cicd/enforce_tier_model/web.yaml` | Both sides extended the allowlist | Mechanical union — sort + dedupe |
| `src/elspeth/web/execution/routes.py` | Both sides modified routing logic | Manual read; small surface |
| `tests/unit/web/sessions/test_models.py` | Schema-test changes both sides | Manual; new Phase 1A column tests vs RC5.2's row-level tests |
| `tests/unit/web/sessions/test_routes.py` | Route-test changes both sides | Manual; new Phase 1B test expectations vs RC5.2's dead-code-removal test deletions |
| `uv.lock` | Both regenerated lockfile | Post-merge `uv lock` regenerates clean |

Plus 12 files that auto-merge textually but include the semantic-conflict risk (see §5).

**Estimated effort for textual conflicts alone:** half-day.

---

## 5. Semantic conflicts (the expensive part)

### 5.1 The dominant risk: RC5.2 commit `2c043a11`

Commit metadata:

```
commit 2c043a11edc9e802704f66b76b8430bff4ebaa8a
Author: John Morrissey
Date:   Sun May 10 10:52:21 2026 +1000

    refactor(composer): remove dead replacement-shape machinery

    Follow-on to elspeth-9cfbad6901: the previous commit moved the only
    remaining replacement-shape producer to augmentation, leaving the
    replacement-side machinery dead. Per the No Legacy Code Policy
    ("when something is removed or changed, DELETE THE OLD CODE
    COMPLETELY"), this commit removes:

    - _runtime_preflight_failure_message (composer/service.py) — no caller
    - _enforce_replacement_non_prefix_invariant (composer/service.py) —
      no producer needs the symmetric guard
    - _ReplacementBranch literal (composer/service.py) — no consumer
    - _INTERCEPTED_ASSISTANT_HISTORY_PREFIX (sessions/routes.py) — no
      read-path emits the prefix anymore
    - The replacement branch in routes._composer_history_content — all
      synthesis is augmentation now; the read-path defensively rejects
      non-augmentation rows with AuditIntegrityError, matching the
      delete-old-DB migration policy in project_db_migration_policy

    Deleted tests pinning the removed behavior:
    - test_runtime_preflight_failure_message_uses_failed_check_when_errors_empty
```

The premise on RC5.2 was: a *prior* commit (`elspeth-9cfbad6901` — note this is a Filigree ID, not a git SHA; the actual RC5.2 git commit that did the migration is unnamed in this body) migrated the only producer of replacement-shape rows to augmentation. With no producer, the consumers became dead code, and per the No-Legacy-Code Policy in CLAUDE.md, they were removed.

**On this branch, the producer migration did not happen.** The replacement-shape machinery is still live, and the four symbols above are referenced by ~12 call sites across `composer/service.py`, `composer/protocol.py`, `sessions/routes.py`, `sessions/protocol.py`, and at least 2 test files.

### 5.2 Active call sites on this branch

`git grep` evidence at HEAD `f5bc5dbb`:

```
_runtime_preflight_failure_message:
  src/elspeth/web/composer/service.py:1220  (definition)
  src/elspeth/web/composer/service.py:1492  (call)
  src/elspeth/web/sessions/routes.py:433   (docstring reference)
  tests/integration/pipeline/test_composer_llm_eval_characterization.py:702  (comment)
  tests/unit/web/composer/test_service.py:4242  (test that RC5.2 deleted)

_enforce_replacement_non_prefix_invariant:
  src/elspeth/web/composer/protocol.py:75   (docstring reference)
  src/elspeth/web/composer/service.py:851   (docstring reference)
  src/elspeth/web/composer/service.py:873   (definition)
  src/elspeth/web/composer/service.py:1493  (call)
  src/elspeth/web/sessions/protocol.py:180  (docstring reference)

_ReplacementBranch:
  src/elspeth/web/composer/service.py:824   (Literal definition)
  src/elspeth/web/composer/service.py:875   (use in _enforce_*)

_INTERCEPTED_ASSISTANT_HISTORY_PREFIX:
  src/elspeth/web/composer/service.py:840   (docstring reference)
  src/elspeth/web/composer/service.py:886   (docstring reference)
  src/elspeth/web/composer/service.py:1369  (comment)
  src/elspeth/web/composer/service.py:1475  (comment in code block)
  src/elspeth/web/sessions/routes.py:573    (definition)
```

### 5.3 Why textual auto-merge would silently break

`git merge-tree` reports `composer/service.py`, `sessions/routes.py`, `sessions/protocol.py` as auto-merging (no CONFLICT markers). What that means mechanically: the hunks from each side touch different line ranges, so a 3-way merge can compose them. What it does *not* mean: the resulting file is semantically coherent.

Concretely, a textual auto-merge of `composer/service.py` produces a file where:

- RC5.2's deletions of `_runtime_preflight_failure_message`, `_enforce_replacement_non_prefix_invariant`, `_ReplacementBranch` are applied (the symbols disappear), because RC5.2 removed entire function bodies and this branch did not touch those exact line ranges.
- This branch's additions at module tail (`_arg_error_payload` from commit `70424cc1`) and the `_PROMOTED_TOOL_NAMES` frozenset are applied.
- This branch's *callers* of the deleted symbols (L1492, L1493) remain.

Result: import-time `NameError` / `AttributeError` on the merged tree. Test suite fails. mypy fails (`name-defined` errors).

### 5.4 Sessions-side delete: `_INTERCEPTED_ASSISTANT_HISTORY_PREFIX`

Same pattern in `sessions/routes.py`. RC5.2 deleted the constant at line 573 and the replacement branch in `_composer_history_content`. This branch's `composer/service.py` has comments and docstrings referencing the constant. The merged file has callers without the symbol they call.

This is more recoverable than the composer side because the references on this branch are mostly **docstrings and comments**, not live code paths. But every one needs reading to confirm.

---

## 6. Phase 1+2 in good shape — analysis

### 6.1 The bucket breakdown

The work on this branch decomposes into four buckets:

1. **Phase 1A — schema additions.** `chat_messages` new columns (`tool_call_id`, `sequence_no`, `writer_principal`, `parent_assistant_id`), `composition_states.provenance` discriminator, `audit_access_log_table` (declared INERT), partial unique index, CHECK constraints. Per `project_db_migration_policy`, staging DB is deleted and recreated on deploy — no backfill question.
2. **Phase 1B — sync primitive + DTOs.** `SessionsService.persist_compose_turn` + async dispatcher; `StatePayload`, `_ToolOutcome`, `RedactedToolRow`, `AuditOutcome` in `_persist_payload.py`; advisory-lock primitive; sequence-number reserver. **Dormant production code** — Phase 3 wires the call.
3. **Phase 1C — CI lane.** Postgres testcontainer marker registered; CL-PP-11 concurrent-session test landed (commit `eca88974`); aggregation job. **Immediately productive contribution to RC5.2** — concurrent-session coverage is real.
4. **Phase 2 — redaction walker.** `MANIFEST` (38 entries: 10 type-driven + 28 declarative per `project_phase2_implementation_complete`); `redact_tool_call_arguments`, `redact_tool_call_response`; `RedactionTelemetry` Protocol + Noop + Otel impls; `_arg_error_payload`, `canonicalize_pydantic_cause` (F2 module-tail helpers); adequacy guard. **Dormant production code** — Phase 3 wires the walker calls.

The replacement-shape machinery (§5.1–5.4) is a **fifth, parallel strand** that is *not* part of Phase 1 or Phase 2. It happens to be on the same branch.

### 6.2 Why dormant code is acceptable for a checkpoint

The auditability standard does not prohibit dormant primitives — only fabricated, half-truthful, or silently-failing ones. Phase 1B/2 primitives are:

- **Tested.** Unit tests + (for Phase 1C) integration tests against real Postgres exercise every primitive's contract.
- **Documented as dormant.** Each helper's docstring cites the spec section and the consuming phase. `SessionsService.persist_compose_turn` docstring says "Phase 3 wires the production caller." `MANIFEST` docstring cites §4.2.1.
- **Adequacy-guarded.** Phase 2's CI-time adequacy guard (`tests/unit/web/composer/test_adequacy_guard.py`) protects against drift between the manifest registration and the dispatch registry, so the dormant walker can't silently lose entries.
- **No user-visible behaviour change.** The compose loop today is byte-identical to RC5.2 on the loop body. New schema columns are present; nothing reads them outside Phase 1A test surface.

### 6.3 What "in good shape" means concretely

After reconciliation (§7) and merge:

- Staging deploy works (per `project_staging_deployment` + `project_db_migration_policy` workflow — operator archives `sessions.db`, restarts service, new DB has new schema).
- Composer end-to-end behaviour unchanged for users.
- New columns / new table exist but are inert; new primitives exist but are not called.
- CI passes (unit + integration + property + tier-model + freeze-guards + mypy + ruff).
- Phase 2 adequacy guard is the long-running quality control on the manifest while Phase 3 is in flight on the worktree.

---

## 7. Resolution options

### 7.1 Option 1 — Pre-merge reconciliation (recommended)

**Apply RC5.2's "all synthesis is augmentation" model to this branch first.** Port the four replacement-shape symbols out of use (or delete entirely, matching `2c043a11`), commit on this branch, then merge cleanly.

Steps:

1. Read RC5.2 commit `2c043a11` body carefully.
2. Identify on this branch which callers can be deleted (matches RC5.2's deletion) vs which need to be ported to augmentation (if this branch added new producers that RC5.2 didn't have).
3. Apply the change in one commit on this branch: `refactor(composer): align replacement-shape model with RC5.2 (delete dead machinery)`. Body cites `2c043a11` as the upstream model.
4. Rerun gates.
5. Merge to RC5.2. Resolve the 5 textual conflicts. Rerun gates.

Cost: ~half-day for reconciliation + half-day for merge + retest. Risk: bounded — the change to make is already documented in `2c043a11`'s body.

### 7.2 Option 2 — Cherry-pick Phase 1+2 onto a fresh RC5.2-based branch

Strip the replacement-shape work entirely by picking only the Phase-1A/1B/1C and Phase-2 commits onto a new branch off RC5.2 HEAD.

Steps:

1. Identify the exact commit set for Phase 1A/1B/1C and Phase 2 (Phase 2 alone is 28 commits between `7338e4e2`..`f54ee7e8`; Phase 1A/1B/1C predates `7338e4e2`; plan rev-2 commit `f5bc5dbb` is doc-only and can be cherry-picked separately).
2. Create a fresh branch off RC5.2 HEAD.
3. Cherry-pick in dependency order. Resolve conflicts per pick.
4. Rerun gates after each Phase boundary.
5. Open the merge PR from the fresh branch.

Cost: ~1-2 days. The benefit is provenance clarity (no replacement-shape commits in the merged history); the cost is loss of the worktree's linear history and more conflict-resolution rounds.

### 7.3 Option 3 — Defer the merge

Continue Phase 3 implementation on this branch; merge everything later. **NOT RECOMMENDED** — the conflict surface compounds because Phase 3 modifies the same files (`_compose_loop` body, sessions/routes.py, sessions/service.py, sessions/protocol.py) that RC5.2 already touched. The longer the deferral, the more 3-way merge work at the end.

### 7.4 Default recommendation

**Option 1.** Lowest cost, smallest blast radius, clearest provenance preserved on the umbrella branch (per `project_adr010_umbrella_branch` precedent — the umbrella does ship multi-phase work end-to-end, but RC5.2's policy decision means this branch needs to align with RC5.2's model before contributing).

---

## 8. Unverified caveats

The analysis above is partial. The following items deserve a 5-30 minute read each before pulling the trigger:

### 8.1 `sessions/models.py` divergence — verify low risk

RC5.2 net diff: +2 / -22 lines, single commit `0964922c` (release stamp). Body says "RC-5.1 release stamp + composer/validator/audit-integrity fixes." The 22-line deletion needs reading to confirm it does not remove a column or constraint that Phase 1A added or depended on.

This branch's contribution to `sessions/models.py`: +232 net lines (mostly Phase 1A schema additions). Auto-merge claims success; the additions and the deletions are presumed non-overlapping.

**Verification command:** `git diff 91133f32..RC5.2 -- src/elspeth/web/sessions/models.py`. Read the deletion.

### 8.2 `composer/tools.py` divergence — verify auto-merge sanity

RC5.2 net diff: 221 lines, 4 commits (grounding detector, augment-shape preflight, state-claim grounding correction, replacement-shape removal).

This branch's contribution: +668 net lines (Phase 2 promotion wave — 9 tools promoted to type-driven Pydantic argument models; the rest declarative manifest entries).

Auto-merge claims success but `composer/tools.py` is now 5736 lines. Collisions in the dispatch-dict region at lines 5250–5314 (the six function-pointer dispatch dicts cited in Phase 2 spec) would silently break manifest registry parity.

**Verification command:** `git diff 91133f32..RC5.2 -- src/elspeth/web/composer/tools.py` plus `git diff 91133f32..feat/composer-progress-persistence-1a -- src/elspeth/web/composer/tools.py | grep -E "^[+-](async def|def |    \"\w+\":)" | head -50` to see whether the changes touch the same handlers.

### 8.3 Test reruns post-reconciliation

After Option 1's reconciliation commit, the full gate suite must rerun cleanly:

```bash
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python -m pytest tests/integration -q -m "not testcontainer"
.venv/bin/python -m pytest tests/integration -q -m "testcontainer"  # Docker-enabled lane
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth
```

Two specific risks:

- Tier-model allowlist refresh after reconciliation (removing functions rotates AST fingerprints; precedent F2 commit `70424cc1` documents the mechanical pattern).
- Test files that pinned the deleted behaviour (e.g., `test_runtime_preflight_failure_message_uses_failed_check_when_errors_empty` per `2c043a11`'s body) must also be deleted on this branch.

### 8.4 Staging redeploy

Per `project_staging_deployment` and `project_db_migration_policy`:

```bash
# Operator action (cannot be automated):
sudo systemctl stop elspeth-web.service
mv /opt/elspeth/sessions.db /opt/elspeth/sessions.db.archive-2026-05-XX
# Pull the merged code:
cd /opt/elspeth && git pull
# Restart:
sudo systemctl restart elspeth-web.service
# Confirm health:
curl https://elspeth.foundryside.dev/health
```

No automated migration; operator action required. Memory `project_db_migration_policy` documents why: ELSPETH does not ship Alembic migrations; schema changes are landed by DB recreation, never row-level DELETE + structural ALTER.

### 8.5 Phase 3 plan rev-3 needed after merge

The plan I committed (`f5bc5dbb`) cites line numbers reflecting this branch's post-Phase-2 state at HEAD `f54ee7e8`: `_compose_loop` at L1668, `except ToolArgumentError` at L2518, etc. After merge, the file shape changes (RC5.2's deletions shrink `composer/service.py` by ~143 lines net). Phase 3 plan needs a rev-3 line-number refresh in Appendix A.4.

Estimated effort: 30-minute edit pass on the line-numbers table only. The task structure and TDD steps are unaffected.

---

## 9. Recommended sequence (Option 1)

1. **Decide Option 1 vs Option 2.** (Default: Option 1.)
2. **Verify caveats §8.1 and §8.2.** Read the two diffs; confirm no unexpected schema or dispatch-dict collisions. Surface anomalies before reconciliation.
3. **Reconcile replacement-shape on this branch.** Port consumers to augmentation (or delete, matching `2c043a11`). Delete the corresponding test (`test_runtime_preflight_failure_message_uses_failed_check_when_errors_empty`). Commit: `refactor(composer): align replacement-shape model with RC5.2 (delete dead machinery)`. Body cites `2c043a11` as upstream model.
4. **Rerun full gate suite on this branch.** Confirm clean.
5. **Merge `feat/composer-progress-persistence-1a` to RC5.2.** Resolve the 5 textual conflicts (tier-model allowlist union; uv.lock regenerate; 3 test files manual). Commit message names the conflict resolutions.
6. **Rerun full gate suite on the merged RC5.2.** Confirm clean.
7. **Operator: stage redeploy** per §8.4 procedure.
8. **Re-baseline the Phase 3 plan and worktree.**
   - Rebase `feat/composer-progress-persistence-1a` onto post-merge RC5.2, OR
   - Tear down and create a fresh `feat/composer-progress-persistence-phase3` branch off RC5.2 HEAD.
   - Either way, refresh Phase 3 plan rev-3 line numbers (Appendix A.4 only).
9. **Resume Phase 3 implementation** on the re-baselined worktree.

---

## 10. Reality-check status

A reality-check review pass on this document was dispatched separately via `axiom-planning:plan-review-reality` ("the hallucination hunter"). The reviewer's brief: verify every commit SHA, file path, line number, symbol name, and count claim in this document against actual codebase state at HEAD `f5bc5dbb`.

**Reviewer verdict:** [pending — see review JSON at `notes/composer-progress-1a-rc5-2-merge-analysis-2026-05-12.review.json` when complete]

**Operator action on review return:**
- **APPROVED / APPROVED_WITH_WARNINGS:** treat this document as load-bearing for the Option-1 sequence.
- **CHANGES_REQUESTED:** fold corrections back into this document (rev-2) before proceeding.

---

## Appendix A — Raw merge-tree output

```
$ git merge-tree --write-tree --name-only --messages \
    feat/composer-progress-persistence-1a RC5.2
defb8efda58e3352fbe24ce09a7074990bb0e519
config/cicd/enforce_tier_model/web.yaml
src/elspeth/web/execution/routes.py
tests/unit/web/sessions/test_models.py
tests/unit/web/sessions/test_routes.py
uv.lock

Auto-merging config/cicd/contracts-whitelist.yaml
Auto-merging config/cicd/enforce_tier_model/web.yaml
CONFLICT (content): Merge conflict in config/cicd/enforce_tier_model/web.yaml
Auto-merging pyproject.toml
Auto-merging src/elspeth/web/composer/service.py
Auto-merging src/elspeth/web/composer/tools.py
Auto-merging src/elspeth/web/execution/routes.py
CONFLICT (content): Merge conflict in src/elspeth/web/execution/routes.py
Auto-merging src/elspeth/web/execution/schemas.py
Auto-merging src/elspeth/web/sessions/models.py
Auto-merging src/elspeth/web/sessions/protocol.py
Auto-merging src/elspeth/web/sessions/routes.py
Auto-merging src/elspeth/web/sessions/service.py
Auto-merging tests/unit/web/composer/test_service.py
Auto-merging tests/unit/web/composer/test_tools.py
Auto-merging tests/unit/web/sessions/test_models.py
CONFLICT (content): Merge conflict in tests/unit/web/sessions/test_models.py
Auto-merging tests/unit/web/sessions/test_routes.py
CONFLICT (content): Merge conflict in tests/unit/web/sessions/test_routes.py
Auto-merging uv.lock
CONFLICT (content): Merge conflict in uv.lock
```

---

## Appendix B — RC5.2 commits since merge base

Excerpted from `git log --oneline 91133f32..RC5.2`, scoped to critical files. Full list available via the same command.

`src/elspeth/web/composer/service.py`:

```
7bd55faa fix(composer): widen grounding detector + plumb past early-return
2c043a11 refactor(composer): remove dead replacement-shape machinery
fab882a0 feat(composer): augment-shape preflight failures preserve model prose
c88e2112 feat(composer): state-claim grounding correction (Path 3)
```

`src/elspeth/web/sessions/models.py`:

```
0964922c release(0.5.1): RC-5.1 release stamp + composer/validator/audit-integrity fixes
```

`src/elspeth/web/sessions/protocol.py`, `sessions/routes.py`, `sessions/service.py`:

```
2c043a11 refactor(composer): remove dead replacement-shape machinery
```

(All three sessions-side files share the single commit.)

---

## Appendix C — Document provenance

| Field | Value |
|---|---|
| File path | `notes/composer-progress-1a-rc5-2-merge-analysis-2026-05-12.md` |
| Worktree | `/home/john/elspeth/.worktrees/composer-progress-1a` |
| Branch HEAD when authored | `f5bc5dbb` |
| Merge base used | `91133f32` |
| RC5.2 HEAD compared against | `3e46a976` |
| Author session | 2026-05-12 |
| Verification status | Reality-check review pending (see §10) |

---

End of merge-difficulty analysis.
