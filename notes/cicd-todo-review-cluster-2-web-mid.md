# Cluster 2 review — web-mid (execution/sessions/interpretation_state)

## 1. Verdict

**PASS**

The FIX agent's claims hold up to independent verification. 38 cluster-2 TODOs are
gone, no cluster 2 finding remains in the lint output, key rewrites and net-new
entries are canonical, budget math reconciles exactly, and 1608 tests pass.
Justifications are individually argued, not paraphrased from neighbours. The 0%
FIX rate is the correct call for this cluster: every spot-checked line is
either a closed-sum-type discrimination, an external-options boundary
validation, or a filesystem/asyncio boundary — none should be "fixed" in
source.

## 2. Check-by-check result

**Check 1 — Report read.** The FIX agent's report at
`/home/john/elspeth/notes/cicd-todo-fix-cluster-2-web-mid.md` enumerates per-entry
verdicts, key rewrites, net-new entries, the reverted match attempt, and budget
delta. Every claim there is verifiable from the diff and source.

**Check 2 — Git diff scope.** `git diff HEAD -- config/cicd/enforce_tier_model/web.yaml`
shows 38 cluster-2 entries with `owner: TODO`/`reason: TODO`/`safety: TODO`/`expires:
null` replaced by per-entry owners, paragraph-length reasons, paragraph-length
safety notes, and `expires: '2026-08-23'` (bounded, 90 days). No edits touch
source code. The remaining 76 lines reporting `owner: TODO` in the file are all
under `web/composer/tools/*` (cluster 3 territory).

**Check 3 — Source spot-check (5 entries).** See section 3. All 5 AGREE.

**Check 4 — Fresh justifications.** Reasons share a "Tier-3 …" lead phrase but
each one names the specific data field (e.g. `options['prompt_template']` vs
`options['resolved_prompt_template_hash']`), the control-flow context (which
line on either side type-checks, which branch raises which exception type), the
specific Python idiom (`asyncio.shield`+`contextlib.suppress`, `pop(key, None)`,
`primary_exc.add_note`), and the affected_node_id propagation. No
"shape-validation badge" pattern. No verbatim copy-from-neighbour.

**Check 5 — Net-new entries.** See section 5. All 4 AGREE.

**Check 6 — Reverted match attempt.** See section 6. Revert is defensible.

**Check 7 — Tests.**
`pytest tests/unit/web/sessions/ tests/unit/web/execution/ tests/integration/web/`
returns `1608 passed, 4 warnings in 63.09s`. Matches the agent's claim
exactly.

**Check 8 — Lint cleanliness in cluster 2 scope.**
`elspeth-lints check --rules trust_tier.tier_model --format text | grep -E "web/(execution|sessions|interpretation_state)"`
returns no output. 0/31 remaining lint findings are in cluster 2 scope; all 31
are in `web/composer/*` (cluster 3).

**Check 9 — Budget delta sanity.** Reconciled exactly:
- HEAD: 363 total / 283 permanent / 80 bounded.
- After: 367 total / 249 permanent / 118 bounded.
- Delta: +4 / −34 / +38.
- The −34 (not −38) is explained by the +4 net-new entries: 4 of the +38
  bounded come from net-new keys (not previously counted), so only 34 keys
  transitioned from permanent → bounded. Algebraically: −34 perm + 38 bounded
  = +4 net.

## 3. Spot-check details

| fp | location | rule | verdict | reasoning |
|----|----------|------|---------|-----------|
| `976876634c13da21` | `web/interpretation_state.py::interpretation_sites` L87 | R1 | AGREE | `node.options.get("prompt_template")` — `options` is composer-authored Mapping[str, Any]; absent key (alternate prompt-parts shape) must yield None and skip the legacy branch. Next line type-checks. Genuine Tier-3 boundary read. |
| `f19fa1c749d0d1ce` | `interpretation_state.py::_coerce_requirement` L212 | R5 | AGREE | `isinstance(user_term, str)` validates a TypedDict field from external composer-authored data; raises `TypeError` on shape violation. Textbook Tier-3 validate-and-raise; no coercion, no fabrication. |
| `f65ffe2b6eacead7` | `sessions/service.py::archive_session::_sync` L1682 | R6 | AGREE | `except OSError as restore_exc` on a filesystem `staged_blob_dir.rename(blob_dir)` rollback path. Narrow OSError catch annotates `primary_exc` via `add_note(...)` and re-raises the primary on L1689. Original DB failure never swallowed. Filesystem boundary. |
| `2776823d39a788ae` | `archive_session::_sync` L1702 | R7 | AGREE | `with contextlib.suppress(OSError): quarantine_root.rmdir()` — best-effort parent rmdir after successful session-scoped rmtree. Parent may legitimately be non-empty (other sessions quarantined). Narrowed to OSError only. Correct boundary handling. |
| `4cfc107b881f53c8` | `execution/service.py::_execute_locked` L604 | R9 | AGREE | `self._shutdown_events.pop(str(run_id), None)` in a `BaseException` setup-failure cleanup path. Idempotent remove-if-present; key may or may not have been inserted before the failure. Raising `KeyError` here would mask the actual setup failure being re-raised. Canonical idiom. |

All five are real trust-tier boundaries. None is fixable without weakening the
audit signal.

## 4. Cargo-culting indicators

None found. The opening "Tier-3 …" phrase is a legitimate categorisation
marker, not paraphrase. Every reason names:
- The specific external system whose data crosses the boundary
  (composer-authored options, asyncio cancel chain, filesystem state).
- The specific field or call.
- The next-line or sibling-line type-check that ensures absent/wrong values
  cannot fabricate downstream data.
- The exception type raised (or the cancel re-raise on L795 of messages.py).

The two text-fragment risks I checked for explicitly — "Returns None on miss,
handled by 404 response" filler badges, and "Fingerprint shifted from <old> by
the F-17/F-21 …" rotation-only justifications — are NOT present in the new
entries. (The agent itself called them out in §"Neighbour entries with weak
justifications" of its own report as follow-up audit candidates for non-TODO
entries.)

## 5. Net-new entry assessment

| fp | location | verdict | reasoning |
|----|----------|---------|-----------|
| `2776823d39a788ae` | archive_session L1702 R7 | AGREE | `contextlib.suppress(OSError)` on a post-rmtree parent rmdir. Best-effort cleanup; raising would falsely report archive failure. Entry should exist; reason is specific to the post-rmtree gate path. |
| `be313c147c5a50be` | send_message L769 R7 | AGREE | `contextlib.suppress(asyncio.CancelledError)` wrapping `asyncio.shield(_persist_llm_calls(...))`. The outer `raise` on L795 restores the cancel chain. Inline comment L757-766 documents the invariant. Entry necessary; reason names the shield-and-resume pattern. |
| `1a85561c51205f0c` | send_message L779 R7 | AGREE | Same cancel-shield idiom for `_publish_progress`. Per-line entry justified because they're two distinct shielded calls. Reason differentiates by what's shielded. |
| `4cfc107b881f53c8` | _execute_locked L604 R9 | AGREE | Idempotent `pop(key, None)` in `BaseException` cleanup. Reason correctly identifies why a strict `del` would mask the actual setup failure. |

All 4 should exist (not fix-the-code candidates) and the reason texts are
appropriately specific.

## 6. The reverted match-statement attempt (fp `039bf42792d84027`)

Source unmodified — confirmed by `git diff HEAD -- src/elspeth/web/execution/service.py`
(no output). The corresponding allowlist entry exists with
`owner: web-execution`, a reason naming the `CompositionState |
InterpretationReviewPending` discriminated sum, and the standard
`expires: '2026-08-23'`. No half-state.

**Was the revert justified?** The R5 rule's "use `match` instead" suggestion
is genuinely stylistic for a closed sum type — both `isinstance` and `match`
are exhaustive over `CompositionState | InterpretationReviewPending`, and
neither fabricates data. The cited precedent
(`source_inspection._declared_field_name`) is a legitimate ELSPETH allowlist
pattern for this exact shape.

The cascade rationale is legitimate: AST shifts caused by introducing a
`match`/`case` block change `body[N]` indices for subsequent statements, which
rotates fingerprints on entries the cluster-2 agent has no authority to
touch (they're not on the cluster-2 work list and one of them is in cluster 3
territory). The right move when an AST-shift refactor cascades outside scope
is to either (a) bundle the cascade fingerprint updates into the same commit
explicitly, or (b) defer the stylistic refactor. The agent chose (b), which
is consistent with the operator's "don't dump scope" / "no unilateral
deferral" rules — this isn't deferring required work, it's deferring an
optional style change. **Acceptable.**

The one nit: the agent could have noted explicitly which 4-5 fingerprints
would have rotated, so a future quarterly review can co-land the refactor
plus the rotation co-edit deliberately. Not a blocker.

## 7. Tests + lint cleanliness — confirmed

- `pytest tests/unit/web/sessions/ tests/unit/web/execution/ tests/integration/web/`
  → 1608 pass, 0 fail. Matches the agent's report exactly.
- `elspeth-lints check --rules trust_tier.tier_model | grep "web/(execution|sessions|interpretation_state)"`
  → 0 findings. Cluster 2 source scope is fully lint-clean against the
  current allowlist.
- 31 remaining un-allowlisted findings are all in `web/composer/*`
  (cluster 3 territory).

## 8. Budget delta sanity — math reconciles

| metric | HEAD | after | delta | reconciles? |
|--------|------|-------|-------|-------------|
| total | 363 | 367 | +4 | ✓ (4 net-new entries) |
| permanent (null) | 283 | 249 | −34 | ✓ (34 of the 38 replacement entries were permanent at HEAD; the other 4 were the +4 net-new) |
| bounded | 80 | 118 | +38 | ✓ (34 transitioned perm→bounded + 4 net-new bounded) |

The −34 (not −38) on permanent is the right number. The 4 of 38 that didn't
contribute to the permanent decrement are the 4 net-new keys — they did not
previously exist as `expires: null` entries to be transitioned, so they
contribute to bounded but not to the permanent decrement.

Headroom against budgets:
- `max_allow_hits: 542` — current 367. Ample.
- `max_permanent_allow_hits: 392` — current 249. Decreasing. Good direction.

No ceiling bump needed.
