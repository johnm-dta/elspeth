# Scoping — Allowlist key-match fallback to `scope_fingerprint`

**Ticket:** elspeth-17322022a7 — *Allowlist key-match fallback to scope_fingerprint when
ast_path drifts (unblock keyless @trust_boundary migration)*
**Status:** scoping (this doc) · **Author:** John Morrissey · **Date:** 2026-06-01
**Related:** elspeth-03fa70d3b3 (cicd-judge-cli), `notes/tier-model-bulk-remediation-playbook.md` §6b,
memory `project_judge_v2_binding_transport_campaign`, `project_reify_file_fingerprint_coupling_2026-05-30`

---

## 1. Problem (confirmed against code)

The `trust_tier.tier_model` allowlist key is
`file_path:rule_id:symbol:fp=<fingerprint>`, where
`fp = sha256(rule_id | ast_path | node_dump)[:16]` and `ast_path` is **module-root-rooted**
(`rule.py:572-575`: `ast_path = "/".join(self.path_stack)`).

A finding matches an entry only via **exact key equality** —
`AllowlistEntry.matches` (`allowlist.py:165-167`) and the tier_model-local
`_match_finding` (`rule.py:2240-2241`).

Adding a module-level statement — the
`from elspeth.contracts.trust_boundary import trust_boundary` import every
`@trust_boundary` migration needs — inserts a new `Module.body[0]` and shifts
`body[N] → body[N+1]` for every statement below it. That rotates the **key
fingerprint of every downstream entry in the file**. The v2 `scope_fingerprint`
work decoupled the **signature** (survives unrelated-scope edits) but **not the
key match** — two different surfaces. So decorator migration is keyless-safe only
on files with zero signed entries below the import insertion point.

**Empirically reconfirmed 2026-06-01:** one decorator+import on
`generation.py::_csv_source_delimiter` produced ~44 stale entries under the gate
(per the ticket's batch-3 reproduction).

### Why a drifted signed entry fails the gate two ways
- The old key no longer matches any live finding → `entry.matched` stays `False`
  → `get_unused_entries()` reports it → `fail_on_stale` crashes the load.
- The live finding's *new* key matches no entry → the violation is unsuppressed →
  the gate fails on the finding too.

The fallback fixes **both**: the entry matches (not stale) and the finding is
suppressed.

---

## 2. Two measurements that change the design

### 2a. Corpus is now 100% v2 — **no `migrate-judge-scope` precondition**

The fallback matches on the entry's **stored `scope_fingerprint`**, which only
**v2** entries carry (v1 binds `file_fingerprint`). Measured 2026-06-01 across
`config/cicd/enforce_tier_model/*.yaml`:

```
total judge entries: 242   v2: 242   v1: 0
web.yaml: 122 judge entries, 122 v2
files with judge entries but no v2 marker: 0
```

The corpus has fully flipped to v2 since the 05-31 memory (which said "all-v1,
migration dormant"). **Consequence:** the fallback is a *self-contained* unblock.
There is **no** per-file `migrate-judge-scope`-before-edit dependency — there are
no v1 entries left to migrate, and no v1 whole-file-hash load-crash
(`allowlist.py:816-830`) standing in the way of editing a signed-neighbour file.

> If a v1 entry is ever re-introduced, the three-step sequence returns
> (operator `migrate-judge-scope` **before** the edit → keyless decorator edit →
> this fallback). Documented here so a future regression is recognised, not
> rediscovered. Today it does not apply.

### 2b. Collisions are common — **the suffix discriminator is mandatory**

The ticket offered a simpler option: restrict the fallback to scopes with a
single finding for that rule. Measured: **39** `(file:rule:symbol, scope_fp)`
groups carry **>1** entry, several at ×8–×9, concentrated in the densest web
functions:

```
×9  web/execution/routes.py:R6:create_execution_router:websocket_run_progress
×9  web/composer/tools/generation.py:R5:_row_fields_referenced_by_condition
×8  web/composer/state.py:R6:_check_schema_contracts
×6  plugins/sources/azure_blob_source.py:R6:AzureBlobSource:_load_csv
...39 groups total
```

These high-density functions are *exactly* the entries that drift on an import
insertion. The single-finding restriction would leave them unprotected — the
worst-affected ones. **The corpus mandates the within-scope discriminator.**

---

## 3. Root cause of the collision (why scope_fp alone is insufficient)

`compute_scope_fingerprint` (`scope_fingerprint.py:54-77`) hashes the **entire
docstring-stripped body of the enclosing scope**. Two findings in the same
function therefore have an **identical** `scope_fingerprint`. So
`file:rule:symbol:scope_fp` cannot distinguish two `.get()` sites in one method.
A within-scope positional discriminator is required.

---

## 4. Proposed design

### 4.1 Discriminator: scope-relative `ast_path` suffix

The module-rooted `ast_path` is `body[N]/.../...`. A module-level import
insertion shifts the **leading index** but preserves the **number of path
components** (depth) and the **suffix below the enclosing scope**. So:

- **Live finding** carries `scope_depth K` = the number of `ast_path` components
  from module root down to the enclosing scope boundary. The visitor already
  knows this (it has both `path_stack` and `node_stack`); compute K (or the
  suffix directly) at `Finding` construction — never reverse-resolve it.
- **Stored entry** suffix is recovered with **no new persisted field**:
  `stored_ast_path.split("/")[K:]`, using the *live* K. Depth is shift-invariant,
  so K is valid for the pre-shift stored path.

### 4.2 Fallback match predicate (after exact-key miss)

For a candidate stored entry with the same `file:rule:symbol` as the live
finding, it is a fallback hit iff **all** hold:

1. `entry.scope_fingerprint == finding.scope_fingerprint` (non-empty both sides).
2. `len(entry.ast_path.split("/")) == len(finding.ast_path.split("/"))`
   (depth unchanged; a structural relocation fails here → re-justify).
3. `entry.ast_path.split("/")[K:] == finding.ast_path.split("/")[K:]`
   (same within-scope position).

If exactly one stored entry satisfies (1)–(3), it matches. **Zero or ≥2 → no
fallback match** (fail closed; ambiguity must never silently bind).

### 4.3 Binding verification on the fallback path

Critical interaction: the existing `verify_entry_binding_against_finding`
(`allowlist.py:833-876`) asserts `entry.ast_path == finding.ast_path` and
**crashes on mismatch** — which is *guaranteed* on the fallback path (drift is
why we are here). The fallback must **not** route through that strict assertion.

Predicate (1)–(3) **is** the fallback's binding check, and it is **at least as
strong** as the exact-match C8-3 in-file-transplant defence:

- `scope_fp` equality ⇒ the whole scope body is byte-identical AST ⇒ the node at
  the finding's position is byte-identical (strictly stronger than the exact
  path's single-node `node_dump`).
- The suffix pins within-scope position. Transplant would require same
  `file:rule:symbol` + identical body + identical within-scope position = the
  same node. No safe-vs-dangerous distinction remains to exploit.

Only the module-rooted *prefix* is no longer pinned — which is exactly the thing
we are deliberately relaxing.

### 4.4 Match-only — **no re-key, no re-sign**

The stored entry keeps its (old, pre-shift) `key` and its existing signature.
Load-time signature verification (`_verify_judge_metadata_signature_at_load`)
recomputes over `entry.key` against itself, so it **still passes** — the signature
never needs to change. The fallback only lets a *new* finding match an *old* key.

**Honest cost (document, do not auto-fix):** the stored `ast_path`/`key`
permanently lag the live node's address (audit-precision decay). Auto-refreshing
would re-key, which breaks the signature → cannot be done silently. A future
verdict-preserving re-key verb (re-sign new key, same verdict, gated on
integrity + relevance like `migrate-judge-scope`) is the clean refresh path and
is **out of scope here**. Tooling auditability is IRAP-grade, not Landscape-grade
(memory `feedback_tooling_auditability_irap_not_landscape`): a lagging-but-valid
stored address is a precision note, not evidence tampering.

---

## 5. Touch points

| Site | Change |
|------|--------|
| `core/allowlist.py` | New shared fallback helper (scope-fp + suffix predicate, single-match-or-fail-closed). New fallback-mode binding check (scope_fp + suffix, **not** ast_path equality). Extend `FindingKey` with optional `scope_fingerprint` / `ast_path` / `scope_depth` so both matchers feed the helper uniformly. |
| `tier_model/rule.py:2224` `_match_finding` | After exact-key miss, call the shared fallback helper; on fallback hit set `entry.matched = True` and return (skip strict `verify_entry_binding_against_finding`). |
| `tier_model/rule.py` visitor | Stamp `scope_depth` (or scope-relative suffix) onto each `Finding` at construction, forward from `node_stack`/`path_stack`. |
| `trust_boundary/shared.py:410` `_allowlist_match` | Uses core `Allowlist.match`. **No active work now** — trust_boundary findings carry no `scope_fingerprint` and entries are v1-only, so the fallback is a correct no-op. Ensure the core helper degrades safely on empty `scope_fingerprint` (skip fallback). Inherits the capability for free when trust_boundary goes v2. |
| `core/reaudit.py:1048` | Verify reaudit's `verify_entry_binding_against_finding` call still behaves for fallback-matched entries (reaudit re-checks live findings; confirm it does not falsely flag a scope-stable drift). |
| Stale/unused path | A fallback-matched entry must set `matched=True` so `get_unused_entries` / `fail_on_stale` do not report it. |

---

## 6. Acceptance criteria

1. **Reproduction regression (the headline):** a v2 corpus fixture mirroring the
   `generation.py::_csv_source_delimiter` import-insertion case loads **green**
   (zero stale, zero unsuppressed) after the module-level import shifts every
   downstream `ast_path`. Fails on revert of the fallback.
2. **No collapse:** two same-`rule` findings in one scope (e.g. an ×9 group) bind
   to their **respective** stored entries after a prefix shift; neither
   cross-binds. A targeted unit test over a 2-entry scope.
3. **Fail closed on real edits:** editing the scope **body** (changing
   `scope_fingerprint`) yields no fallback match → entry stale / finding active →
   re-justify. A structural relocation that changes depth (wrap in a class) also
   fails closed.
4. **Fail closed on ambiguity:** if ≥2 stored entries satisfy the predicate for
   one live finding, no match (no silent bind).
5. **Signature intact:** a v2 entry matched via the fallback still verifies at
   load (`_verify_judge_metadata_signature_at_load` passes; no re-key).
6. **Transplant still caught:** the existing C8-3 in-file-transplant tests stay
   green; add one asserting a quartet transplanted onto a different within-scope
   position (different suffix) does **not** fallback-match.
7. **trust_boundary unaffected:** trust_boundary gate behaviour is byte-identical
   (no `scope_fingerprint` on its findings → fallback skipped).
8. **rotate untouched:** `rotate` is match-time-orthogonal and is **not** modified
   in this ticket (folding it in is scope creep).

---

## 7. Out of scope (named, not deferred-by-avoidance)

- Verdict-preserving **re-key** verb (refresh the lagging stored `ast_path`/`key`
  without re-running the judge). Clean follow-up; file as its own ticket if the
  audit-precision decay is judged worth closing.
- Wiring trust_boundary's scanner to stamp `scope_fingerprint` + minting v2
  trust_boundary entries. Independent; this ticket leaves the hook safe.
- `migrate-judge-scope` ordering automation — moot while the corpus is 100% v2.

---

## 8. Open questions for implementation

- **Where K is computed:** confirm the visitor can stamp `scope_depth` without a
  second tree walk (it has `node_stack`; the enclosing scope index is the same
  lookup `enclosing_scope_node` already does). Likely trivial; verify.
- **`FindingKey` shape:** decide between widening `FindingKey` (optional fields)
  vs. passing a richer match-context object to the shared helper. Widening keeps
  both matchers uniform; weigh against the frozen-dataclass / churn cost.
- **reaudit semantics:** does reaudit *want* to surface scope-stable drift as a
  "this entry's stored address lags" advisory (precision decay), even though the
  gate now tolerates it? Possibly a reaudit report line, not a failure.
