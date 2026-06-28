# Design: Scope-Fingerprint Binding for Judge-Gated Suppressions

**Date:** 2026-05-31
**Status:** Draft — awaiting operator review
**Topic:** Replace the whole-file `file_fingerprint` binding on judge-gated
tier-model allowlist entries with an **enclosing-scope** AST fingerprint, so
editing code *near* (but not *at*) a signed suppression no longer forces an
operator-only re-signing.

---

## 1. Problem

A judge-gated tier-model allowlist entry carries three binding primitives, all
inside the HMAC-signed payload (`compute_judge_metadata_signature`,
`elspeth-lints/src/elspeth_lints/core/allowlist.py:547`):

| Primitive | What it hashes | Verified at | Brittleness |
|-----------|----------------|-------------|-------------|
| `fp=` (in `key`) | AST content of the **suppressed node** (`_fingerprint_node`, `rule.py:533`) | finding match | changes only if the suppressed code's own AST changes — *the meaningful signal* |
| `ast_path` | structural address of the node | `verify_entry_binding_against_finding` (`allowlist.py:748`) | changes on **statement insertion above** (body-index shift) |
| `file_fingerprint` | **SHA-256 of the whole file's bytes** | `_verify_file_fingerprint_at_load` (`allowlist.py:713`) — **hard crash** on mismatch | changes on **any byte edit anywhere in the file** |

`file_fingerprint` is the problem. It is double-locked: a load-time crash gate
*and* part of the signed payload, so the field cannot even be mechanically
updated without the operator-held HMAC key.

**Observed cost.** This coupling has blocked two independent workstreams in two
days:
- The 2026-05-30 reify campaign (memory `project_reify_file_fingerprint_coupling_2026-05-30.md`):
  only 17/49 files were reifiable by a keyless agent, the rest blocked solely by
  signed *neighbours* in the same file.
- The 2026-05-31 F1/F2 resume merge: 7 signed entries detonated even though
  their suppressed code was byte-identical — the edits were elsewhere in the
  files.

**Audit-integrity cost (the deeper problem).** A whole-file hash fires on 100%
of edits and discriminates nothing — it cannot tell "context that matters to
this suppression changed" from "an unrelated function 500 lines away changed."
A control that forces re-signing on every unrelated edit trains operators to
**rubber-stamp** re-justifications without re-reading the rationale. For an
audit system that is *worse* than the gap it claims to close: a thoughtless
re-sign is a recorded judge verdict nobody actually re-examined.

## 2. Goals / Non-goals

**Goals**
- Re-justification fires when the **enclosing function/class** of the
  suppression changes — neither too broad (file: noise + alarm fatigue) nor too
  narrow (node-only: silent context drift).
- Editing a *different* scope in the same file is free (no re-sign).
- The change is auditable and the security delta is explicit.
- Existing signed entries migrate without an all-red gate window.

**Non-goals (explicitly out of scope for this change)**
- `ast_path`'s insert-above brittleness. It is a *separate*, rarer tax (fires
  only when a suppression's statement position structurally shifts). Bundling it
  is scope creep. Noted as a candidate Phase 2.
- The keyless debts the 2026-05-31 `--no-verify` merge left on RC5.2
  (`IncompleteTokenSpec` contracts-layer move; `serialization.py` per-file rule
  26/13 ceiling). These are independent of the signing scheme and proceed on a
  **separate track** — they must not wait on this change shipping.
- Re-judging any suppression. This changes the *binding mechanism*, not any
  verdict.

## 3. The change: `scope_fingerprint`

Replace `file_fingerprint` with `scope_fingerprint` — the AST-content
fingerprint of the **innermost enclosing scope** of the suppressed node, using
the same normalization `_fingerprint_node` already applies (`ast.dump` with
`include_attributes=False` — formatting, whitespace, and comments are free).

Bindings after this change:
- `fp` (in `key`) — primary match + suppressed-node content drift. *Unchanged.*
- `ast_path` — structural location / transplant defence. *Unchanged.*
- `scope_fingerprint` — **local context drift. Replaces `file_fingerprint`.**

### 3.1 Scope-hash determinism rules (normative)

These must be documented and frozen — the hash must be reproducible at justify
time and verify time:

1. **Scope node** = the innermost enclosing `FunctionDef`, `AsyncFunctionDef`,
   or `ClassDef` of the suppressed node. A suppression directly in a class body
   binds to the `ClassDef`; a suppression in a method binds to that method's
   `FunctionDef` (not the class).
2. **Module-level fallback** = if there is no enclosing def (a module-level
   suppression), the scope degenerates to the whole module. This reproduces
   today's whole-file behavior for that entry. Module-level suppressions are
   rare; the degeneration is explicit, not accidental.
3. **Included in the hash:** the def's `name`, signature/parameters (`args`),
   `decorator_list`, and full `body`. Decorators are included deliberately —
   a `@retry` added above a def materially changes behavior the judge reasoned
   about. Parameters are part of the contract the judge read.
4. **Excluded from the hash:** the leading **docstring** (strip `body[0]` when
   it is a docstring `Expr`). A documentation edit must not force a security
   re-sign — that is exactly the alarm fatigue this change targets.
5. **No `ast_path` prefix.** Unlike `_fingerprint_node` (which prefixes the
   node's `ast_path`), the scope hash deliberately omits it, so inserting a
   *sibling* method above does not change a suppression's scope hash. This is
   the relief mechanism.
6. **Consequence — renaming the enclosing def changes the hash** (the `name` is
   in the dump). Acceptable: a rename is rare and is a genuine identity change
   worth a re-justify.

### 3.2 Where verification runs (open implementation question)

`file_fingerprint` is verified at allowlist **load** (cheap byte-hash, no
parse). `scope_fingerprint` needs the *parsed* enclosing def. The rule already
parses every source file during `check` and the matcher already computes the
live `ast_path` for `verify_entry_binding_against_finding` — so the natural
home is **finding-match time**, reusing that parse, not a re-parse at load.

Implication to resolve in writing-plans: a *stale* entry (no matching finding)
never reaches the matcher, so its scope drift is not checked there — but a stale
entry is already reported as stale (no finding matches), and the "binds to a
file that no longer exists" case stays a load-time check. Confirm this covers
the cases `_verify_file_fingerprint_at_load` covers today (source-drift,
cross-file transplant) before deleting it.

## 4. Signature versioning + migration

### 4.1 Versioning (mandatory part of this change)

The signed payload already carries `"version": 1` (`allowlist.py:575`). Define
**v2**: `scope_fingerprint` replaces `file_fingerprint` in the payload, and a
stored `judge_signature_version` field records the entry's version.

The version is **self-protecting**: it lives *inside* the signed payload, so a
v1 entry cannot masquerade as v2 (or vice versa) without a valid signature for
that version — which requires the key. Load dispatches verification by the
entry's stored version: v1 entries verify against `file_fingerprint` (old path
retained during migration); v2 against `scope_fingerprint`. `justify` always
writes v2 going forward.

### 4.2 Migration strategy — RECOMMENDED: eager, atomic, mechanical

> **Operator decision point.** The advisor proposed *lazy* migration (keep the
> v1 path indefinitely; entries migrate only when next edited). I am
> recommending **eager** migration instead, because of this repo's strict **No
> Legacy Code policy** (CLAUDE.md) and because the bulk migration is
> *mechanical* — not the operator marathon lazy migration was meant to avoid.
> This is the one point where I diverge from the advisor; please confirm.

A new keyed batch command migrates every **currently-valid** v1 entry to v2:

- **For each v1 entry whose `file_fingerprint` still matches live source**
  (proving the source is unchanged since judgment, so the current enclosing
  scope *is* what the judge saw): compute `scope_fingerprint` and re-sign as v2,
  carrying the existing verdict/rationale forward unchanged. **Mechanical — no
  judge re-review.**
- **For any v1 entry whose `file_fingerprint` does not match** (already stale —
  e.g. today's 7): **refuse to mechanically migrate.** These require genuine
  `justify`/`reaudit` against the changed source (they needed that regardless of
  this change). The command reports them.

Rollout sequence (no all-red window, ends with v1 code deleted):
1. Land v2 code + the batch-migrate command, v1 verify path still present
   (all entries still v1 → gate green).
2. Operator runs batch-migrate (key): valid entries → v2; stale ones reported.
3. Operator `justify`/`reaudit`s the reported stale entries (key).
4. Final commit **deletes** the v1 `file_fingerprint` field, payload branch, and
   `_verify_file_fingerprint_at_load` (No Legacy Code satisfied; all entries v2).

Steps 1–4 can be a single operator session (in a worktree, with the key) so no
intermediate dual-version state is ever committed.

## 5. Expected relief (honest)

Relief is proportional to `1 / scope-size` and only applies when an edit lands
in a *different* scope than the suppression:

- **Today's 7 pay a re-sign regardless** — they are already broken under v1.
  1d's benefit is **prospective**: after they re-sign as v2, the *next*
  unrelated edit will not re-break them.
- **Calibration of today's merge (per `git diff fd4830b42..b516921ee`):** mixed.
  `recovery.py` (`+502,160`) and `query_repository.py` (`+521,71`) are large
  *insertions* of new resume methods — **neighbours**; those suppressions'
  scopes are unchanged, so 1d relieves them fully going forward.
  `data_flow_repository.py` has dense hunks *through* the suppression region
  (657–909) — some of those methods' own bodies changed, so 1d will *correctly*
  still require re-justification there. That non-relief is a feature: the local
  context the judge read actually changed.
- **Giant-method caveat (advisor):** for a suppression inside a 1000+ LOC method
  (this codebase has had them — `_dispatch_tool_batch`), the enclosing-scope
  hash is nearly as brittle as the whole-file hash; for such an entry 1d ≈ drop.
  This is acceptable and self-correcting (large methods are themselves a
  refactor smell) but the spec does not claim 1d is a clean win for *every*
  entry.
- **Population level:** the reify evidence (the problem is signed *neighbours*)
  indicates the general case is strong relief.

## 6. Security delta

**Lost:** the guarantee that a judge-gated suppression's *entire file* is
byte-identical to what the judge inspected. A change elsewhere in the file that
alters the suppressed code's meaning *without* touching its enclosing scope
(e.g. a module-level import rebinding a name the suppressed call uses) no longer
forces re-justification.

**Retained / gained:** `fp` (node content) and `ast_path` (location) remain
signed and verified. `scope_fingerprint` adds a *meaningful* context binding —
re-justification fires exactly when the enclosing function/class the judge read
changes.

**Why the trade is sound:** the whole-file hash's "context" guarantee was always
an over-approximation that fired on everything and discriminated nothing; its
practical effect was alarm fatigue, which *degrades* audit integrity. The
time-based `reaudit` sweep (`elspeth-lints reaudit`) remains the backstop for
slow context drift — it re-examines entries on a cadence, which is the right
tool for "has the world around this suppression drifted," rather than a
per-edit crash.

## 7. Affected code (design-level; writing-plans enumerates exactly)

- `core/allowlist.py`: `AllowlistEntry` (+`scope_fingerprint`,
  +`judge_signature_version`; `file_fingerprint` removed after migration);
  `compute_judge_metadata_signature` (v2 payload + version dispatch);
  `_verify_judge_metadata_signature_at_load` (dispatch by version);
  `_verify_file_fingerprint_at_load` → `_verify_scope_fingerprint_at_load`
  (or move scope check to the matcher per §3.2);
  `verify_entry_binding_against_finding` (carry the live scope hash).
- `rules/trust_tier/tier_model/rule.py`: a scope-hash helper reusing
  `_fingerprint_node` normalization on the enclosing-def node; surface the
  enclosing scope to the matcher/writer.
- `core/cli.py`: `justify` writes v2; **new** batch-migrate command (§4.2);
  `reaudit`/`rotate` understand `scope_fingerprint`.
- `core/judge_coverage.py`, `core/reaudit_sidecar.py`, `core/source_excerpt.py`:
  `file_fingerprint` references retargeted/removed.
- Allowlist YAMLs under `config/cicd/enforce_tier_model/`: migrated by the keyed
  batch command, not hand-edited.

## 8. Testing

- Scope-hash determinism: same scope → same hash across reformatting, comment
  edits, docstring edits (excluded), and sibling-method insertion (no `ast_path`
  prefix). Different scope content → different hash. Decorator/param/body change
  → different hash.
- Module-level fallback reproduces whole-module behavior.
- v1 entry verifies under v1 path; v2 under v2; a v1↔v2 version flip without a
  valid signature is rejected (self-protection).
- Batch-migrate: valid v1 → v2 mechanically with verdict carried forward;
  already-stale v1 refused and reported.
- The deletion commit (step 4) leaves no `file_fingerprint` references and the
  full tier-model suite stays green.
- All via the real loader/justify/verify paths — no bypass of production code.

## 9. Out of scope / separate tracks (recorded so they are not lost)

1. **Keyless RC5.2 debt** (independent, do not block on this): `IncompleteTokenSpec`
   contracts-layer move (`recovery.py:67`); `serialization.py` per-file rule
   26/13 ceiling.
2. **Today's 7 + rotations** still need the operator's key to clear RC5.2's CI
   under the *current* (v1) scheme; this change does not retroactively un-break
   them.
3. **`ast_path` insert-above brittleness** — candidate Phase 2.
