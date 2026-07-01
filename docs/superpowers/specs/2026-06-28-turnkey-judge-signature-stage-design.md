# Turnkey judge/signature stage — the *stage → sign-bundle* seam

**Date:** 2026-06-28
**Status:** Design (approved for planning)
**Implements:** elspeth-281582acc9 (*Signature verification tooling — operator-safe signed allowlist
repair workflows*). Honours the security invariant in elspeth-b3a3335c9f (*[O1] operator-only HMAC
custody*) and closes elspeth-2b351cd004 (*CI exposes allowlist HMAC secret to PR-controlled code*).

---

## 1. Problem

The `trust_tier.tier_model` lint system gates code through an LLM "judge" whose verdicts are sealed
into the allowlist with an HMAC signature. Authoring and maintaining those signed entries is a manual
ceremony today. The `notes/060-*sign*.{md,sh}` runbooks capture the recurring dance: an operator runs
`diagnose-judge-signatures`, hand-builds a `sign-judge-signatures`/`justify` invocation per drifted
entry (each with `--file-path`/`--symbol`/`--fingerprint`/`--rationale` flags), runs
`regen_fingerprint_baseline.py`, re-runs the gate to confirm exit 0, then pushes — all inside a "cert
shell" holding `ELSPETH_JUDGE_METADATA_HMAC_KEY`. Key rotation has no tooling at all. The judge call
and the signing step are fused inside `justify`, so the expensive judgment and the cryptographic stamp
cannot be separated across the agent/operator trust boundary.

The result: every signing round is tedious, per-entry, operator-blocking, and undocumented except as a
one-off runbook.

## 2. Goal

A turnkey two-actor workflow with one cryptographic seam:

- **Agent** stages reviews through **MCP tools** — key-free. It can run a *non-authoritative preview*
  judge (riding the Claude login), diagnose drift, plan rotations, and assemble a complete plan, but it
  never holds the HMAC key and can never produce a signable verdict.
- **Operator** fires that plan through a **CLI** — `elspeth-lints sign-bundle <bundle>` — in a
  key-bearing shell. One command replaces the whole runbook.

## 3. Prior art and hard constraints

Existing primitives in `elspeth-lints/src/elspeth_lints/core/` that this design composes (does **not**
replace):

| Primitive | Role | Key needed |
|---|---|---|
| `cli.py: justify` | run judge + sign one new entry | HMAC + LLM |
| `cli.py: diagnose-judge-signatures` | read-only drift/missing/invalid diagnosis | none (verifies if key present) |
| `cli.py: sign-judge-signatures` | repair drifted entries by re-running `justify` | HMAC + LLM |
| `cli.py: migrate-judge-scope` | re-sign v1→v2 without re-judging | HMAC |
| `cli.py: rotate` / `rules/trust_tier/tier_model/rotate.py` | mechanical fingerprint re-binding | none |
| `cli.py: audit-verdict` | attach post-hoc human verdict | none |
| `core/judge.py` | the judge (`TRANSPORT_OPENROUTER`, `TRANSPORT_AGENT`) | LLM |
| `core/allowlist.py` | entry schema, `JudgeVerdict`, HMAC signature (`compute_judge_metadata_signature`) | — |
| `core/cli.py: _build_yaml_entry_text` | the signed-entry payload writer (`cli.py:2221`, **not** `allowlist.py`) | — |

**Security constraints that shape the architecture (not optional):**

1. **[O1] linchpin (elspeth-b3a3335c9f).** The HMAC is symmetric — any key holder can forge. If an
   agent ever holds the key it can skip the judge, hand-write `judge_verdict: ACCEPTED` with a
   fabricated rationale and a publicly-computable fingerprint, sign it, and pass every gate. Therefore
   the authoritative judge verdict for a *new* finding may only be produced inside the operator-keyed
   step; an agent-staged verdict the operator merely stamps is precisely this forgery vector.
2. **CI key exposure (elspeth-2b351cd004).** The HMAC key must never be reachable from PR-controlled
   code. Signing must run only in an operator-controlled shell, never in CI. CI keeps *verifying*,
   never signing.

## 4. Architecture

```
  AGENT (MCP)                              │  OPERATOR (CLI, holds the key)
  no HMAC key — fails closed if present    │  ELSPETH_JUDGE_METADATA_HMAC_KEY in a cert shell
  ─────────────────────────────────────────┼─────────────────────────────────────────────────
  stage_scan ─┐                            │
  stage_preview (non-authoritative judge)  ├──►  .elspeth/staged-reviews/<id>.json
  stage_rekey ─┘        "the bundle"       │     (a PLAN — carries ZERO authority)
  verify_signatures (read-only diagnosis)  │                   │
                                           │                   ▼
                                           │   elspeth-lints sign-bundle <bundle>
                                           │   elspeth-lints rekey --in <bundle>
                                           │      1. re-verify every binding FROM THE TREE
                                           │      2. new findings + drift_repair → REAL judge + sign
                                           │         rotation / stale_delete     → mechanical, no judge
                                           │      3. summary → confirm → atomic write → regen baseline
```

The handoff is a **review bundle** — a JSON file under `.elspeth/staged-reviews/` (already gitignored
via `.elspeth/*`). The agent stages everything into it; the operator fires it.

### 4.1 The review bundle (authority-free artifact)

A bundle records the plan and *nothing signable*. Conceptual shape:

```jsonc
{
  "bundle_id": "<slug>",
  "schema_version": 1,
  "created_at": "<ISO8601>",            // stamped by the writer, not asserted as authority
  "staged_by": "<agent/operator label>",
  "root": "src/elspeth",
  "allowlist_dir": "config/cicd/enforce_tier_model",
  "source_rev": "<git HEAD sha>",       // provenance; sign-bundle re-verifies regardless
  "source_dirty": true,
  "actions": [
    {
      "lane": "new_judgment",           // new_judgment | resign
      "kind": "justify",                // justify | drift_repair | rotation | stale_delete
      "key": "<canonical allowlist key>",
      "file_path": "...", "symbol": "...",
      "fingerprint": "...", "scope_fingerprint": "...", "ast_path": "...",
      "draft_rationale": "...",
      "preview": {                      // new_judgment only; NEVER authoritative
        "verdict": "ACCEPTED",
        "rationale": "...",
        "model": "...", "transport": "claude_agent_sdk",
        "authoritative": false          // ALWAYS false in a bundle
      }
    },
    {
      "lane": "resign",
      "kind": "drift_repair",
      "key": "...",
      "diagnosis_status": "AST_PATH_BINDING_DRIFT"   // what staging asserts; sign-bundle re-checks
    }
  ],
  "rekey": null    // populated only by stage_rekey
}
```

**Invariant:** a bundle never contains an HMAC signature, and every `preview.authoritative` is `false`.
The bundle is a *claim*; firing is where claims are checked.

### 4.2 MCP server (agent-facing, key-free)

A new `elspeth-judge` MCP server — `python -m elspeth_lints.mcp` (mirroring `legis mcp`), registered in
`.mcp.json`. Five thin, single-purpose tools:

- **`stage_scan`** — build/refresh a bundle worklist. Runs `check` for new tier-model findings lacking a
  valid signed entry; `diagnose-judge-signatures` for drift/missing/invalid; `rotate --dry-run` for
  positional rotations; flags stale entries. Key-free, no LLM, fast. Returns a per-lane summary and the
  exact `sign-bundle` command.
- **`stage_preview`** — run the **non-authoritative** preview judge (`claude_agent_sdk` transport) over
  the bundle's `new_judgment` actions, filling in `preview` verdicts and surfacing BLOCKED reasons so the
  agent fixes code/rationale *before* the operator is involved. Re-runnable as rationale is refined.
- **`stage_status`** — read a bundle back: per-lane counts, preview outcomes, exactly what `sign-bundle`
  will do, and the paste-ready operator command.
- **`verify_signatures`** — standalone read-only diagnosis. **Always shape-only** on the MCP surface: the
  agent surface is structurally key-free and *fails closed* if the HMAC key is present, so it can never
  perform an authoritative recompute. The authoritative recompute path is the CLI
  `diagnose-judge-signatures` / library `diagnose_judge_signatures` surface, not this tool. The report's
  `verification_mode` field labels it. Satisfies the elspeth-281582acc9 read-only requirement and is
  usable outside the staging flow.
- **`stage_rekey`** — produce a rekey plan: enumerate every entry currently valid under the active key,
  and flag any already-broken entries that must be repaired before a rekey (no laundering).

**Fail-closed enforcement:** every MCP tool aborts if `ELSPETH_JUDGE_METADATA_HMAC_KEY` is present in its
environment. This is structural [O1] defense-in-depth — the agent's surface must never be co-located with
the key, and the tool refuses to run if it is.

### 4.3 `elspeth-lints sign-bundle <bundle>` (operator CLI)

```
elspeth-lints sign-bundle <bundle.json> [--dry-run] [--yes] [--operator-override KEY ...]
```

1. **Load + integrity-check** the bundle; refuse if malformed.
2. **Re-verify from the source tree — the linchpin.** For every action, re-derive the binding
   fingerprints from the *current* source and re-run the `diagnose-judge-signatures` logic. If any
   action's staged claim does not match ground truth — e.g. the bundle says "positional drift only" but
   the enclosing scope content actually changed (`SCOPE_BINDING_DRIFT`) — **the entire run aborts before
   any write**: a mismatch means the bundle is stale relative to the tree, which is cheap to fix (re-run
   `stage_scan`) and unsafe to sign around. This is distinct from a real-judge BLOCK in step 3, which is
   an expected per-action outcome, not a staleness signal. **Staging asserts; firing verifies.** Previews
   are discarded for authority.
3. **Execute by lane (dry phase first, then write):**
   - `new_judgment` → run the **real** judge (authoritative) and sign, reusing the `_run_justify`
     handler (`cli.py:1443`). If the real judge BLOCKS something that previewed ACCEPTED, it is reported
     and *not* signed (the agent re-stages, or the operator supplies `--operator-override KEY` with the
     existing override token).
   - `resign` / `drift_repair` (missing/invalid/AST-path/scope/binding drift) → **re-judge** through the
     `_run_justify` ceremony, reusing the `_run_sign_judge_signatures` handler (`cli.py:3142`, which the
     §3 table classes "HMAC + LLM"). Re-judging — *not* a no-judge carry-forward — is the security property: an honest
     `SCOPE_BINDING_DRIFT` means the enclosing scope content actually changed, so carrying a stale verdict
     forward would stamp it onto code the judge never inspected (the [O1] forgery). BLOCK-not-signed and
     override-token gating are inherited from `justify`.
   - `resign` / `rotation` → mechanical key re-bind of **non-judge-gated** entries via `rotate` /
     `apply_plan` (which refuses any judge-gated entry). No verdict, no judge.
   - `stale_delete` → remove the orphaned entry.

   (v1→v2 migration stays on the standalone `migrate-judge-scope` CLI; there is **no** `migrate_v2`
   bundle kind — a per-entry bundle action cannot reuse that whole-allowlist command without
   reimplementing it.)

   **Verify is atomic; execute is per-action.** Step 2 is the all-or-nothing gate — "abort before any
   write" is a *verify-phase* property. Step 3, reusing the `_run_sign_judge_signatures`
   continue-on-failure ceremony, is **per-action non-transactional**: the real judge first runs *after*
   the confirm gate, so a mid-bundle BLOCK leaves earlier-accepted actions written, restores/skips the
   blocked one, and returns non-zero with an "M succeeded / K failed" report. This is safe — every
   committed write was judge-authorized and the run is re-runnable — but the operator must know a
   blocked 10-action run is partially applied, not rolled back.
4. **Summary + confirmation.** Print what will be judged, re-signed, deleted, and the override-rate
   impact; require confirmation (`--yes` to skip, `--dry-run` to stop here). Operator-gates a destructive,
   shared-state write.
5. **Atomic write** per allowlist file (reuse `atomic_update_text`), guarding against the known dup-key
   dataloss in the rotate path; regenerate the fingerprint baseline; report post-state diagnosis
   (expected all-OK **for the bundle's keys** — any out-of-bundle tree drift remains and is reported).
6. **Fail closed** without the HMAC key.

### 4.4 `elspeth-lints rekey --in <bundle>` (operator CLI)

```
elspeth-lints rekey --in <bundle.json> --old-key-env OLD --new-key-env NEW [--dry-run] [--yes]
```

Re-derive the **full judge-gated entry set from the tree** at fire time (the bundle's `RekeyPlan` and
the env-var names it records are advisory provenance; the CLI flags and the live allowlist are
authoritative), verify **every** such entry under the **old** key, recompute the HMAC under the **new**
key, and atomic-write. An entry present in the tree but absent from the staged plan is still re-keyed
(self-healing — never silently left under the retired old key). Refuses to rekey any entry that does not
currently verify under the old key — broken entries are not laundered into the new key; they must be
repaired (via `sign-bundle`) first. Because the canonical corpus is all-v2 (binding unchanged on
rekey), the write is a **scheme-preserving signature-only swap** — only the `judge_metadata_signature`
line changes; binding and audit lines stay byte-identical. This *generalizes* `migrate-judge-scope`'s
integrity-first two-pass *structure* to "re-sign-all-under-new-key"; it does **not** reuse that command's
v1→v2 binding converter (which only rewrites `file_fingerprint`-bound v1 entries).

### 4.5 CI integration

`enforce-allowlist-judge-gates.yaml` stays the **verification** side (override-rate C3, judge-quality
VAL). `sign-bundle` and `rekey` are the **authoring/repair** side and **never run in CI** — they are
operator-shell only. Making signing exclusively operator-local is what closes elspeth-2b351cd004: the
HMAC key never enters a PR-controlled checkout. Re-enabling the dormant C1 coverage gate (currently
unsatisfiable, per the workflow header) becomes possible once the tooling makes every covered entry
signable; that re-enable is tracked but out of this spec's write scope.

## 5. Security invariants (stated, tested)

1. **Staging asserts; firing verifies** — the bundle carries zero authority; `sign-bundle` re-derives
   all bindings from the tree and refuses on any drift mismatch.
2. **The agent never holds the HMAC key** — every MCP tool fails closed if the key is in its env.
3. **Authoritative judge runs only inside the keyed step** for new findings; previews are always
   `authoritative: false`.
4. **Signing never runs in CI** — `sign-bundle`/`rekey` are operator-shell only.
5. **No broken-entry laundering** — `rekey` only re-keys entries that currently verify under the old key;
   `drift_repair` *re-judges* (it never stamps a stale verdict onto changed content); `rotation` refuses
   judge-gated entries. No stale or broken verdict is ever carried into a freshly-signed state.

## 6. What this obsoletes

The one-off `notes/060-*` signing runbooks and `scripts/cicd/sign_accept_backlog.py` collapse into
`stage_scan` → review → `sign-bundle`, and are deleted once the flow demonstrably replaces them
(prerelease-no-tech-debt; removed, not nulled around). The `scripts/codex_tier_model_rejudge.py` driver
is *functionally* superseded too, **but its deletion is deferred**: the sibling active plan
`2026-06-28-codex-panel-review-foundation.md` (committed `b3909d73c`) declares it do-not-modify, so
retiring it requires explicit cross-plan reconciliation with that plan's owner before removal.

## 7. Testing

- `verify_signatures` works with no key (shape-only) and labels its `verification_mode`; on the MCP
  surface it is *always* shape-only (authoritative recompute is the CLI/library `diagnose` surface).
- Every MCP tool **fails closed** when the HMAC key is present in its environment.
- `sign-bundle` **refuses on tree-drift mismatch**: a bundle that claims "positional drift only" for an
  entry whose scope content actually changed is rejected, not signed.
- `sign-bundle` re-runs the real judge for `new_judgment` **and `drift_repair`** (re-judge, not a
  no-judge carry-forward — a contradicting BLOCK on `drift_repair` is reported and not signed);
  `rotation`/`stale_delete` touch no judge.
- A real-judge BLOCK that contradicts an ACCEPTED preview is reported and not signed (no override).
- `rekey` refuses to carry a non-verifying entry into the new key.
- Atomic write / dup-key safety on the allowlist file.
- `sign-bundle`/`rekey` **fail closed** without the key.
- A bundle never serializes a signature; `preview.authoritative` is always `false`.

## 8. Decomposition / phasing (for the implementation plan)

1. **Bundle + re-verify core** — bundle schema, writer/reader, and the from-tree re-verification used by
   `sign-bundle` (the linchpin), reusing `diagnose-judge-signatures` internals.
2. **`sign-bundle` CLI** — lanes, judge-on-new + re-judge-on-drift_repair, mechanical rotation/stale_delete, summary/confirm, atomic write,
   baseline regen.
3. **MCP server** — `stage_scan`, `stage_status`, `verify_signatures`, then `stage_preview`; fail-closed
   on key presence; `.mcp.json` registration.
4. **`rekey`** — `stage_rekey` + the operator `rekey` CLI; dual-key verify window.
5. **CI re-wire + cleanup** — confirm CI only verifies; delete the obsoleted runbooks/scripts; doc the
   agent/operator handoff and the HMAC custody rule.

## 9. Out of scope

- Asymmetric signatures (sign-private / verify-public). Noted in [O1] as a structural improvement that
  would let an agent verify but not sign; it does not remove the new-finding forgery surface (the agent
  could still fabricate a verdict for the operator to sign), so it is a separate future change.
- Re-enabling the C1 judge-coverage CI gate (depends on universal signability landing first).
