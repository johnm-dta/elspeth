# Design Spec: Unified Allowlist Operations Interface (CLI + MCP)

**Date:** 2026-05-31
**Status:** Approved (design) — pending implementation plan
**Owner:** ELSPETH maintainers
**Topic:** A unified CLI + MCP interface over all cicd allowlist gates, so agents and operators stop hand-rolling bash/python to introspect, validate, reconcile, and sign allowlist state.

## 1. Problem

The cicd allowlist gates (tier-model, freeze-guards, options-metadata, component-type, audit-evidence, contract-manifest, and ~11 more under `config/cicd/*`) are the suppression/exemption surface for ELSPETH's CI invariants. Operating on them — during merge-conflict resolution, fingerprint rotations, stale pruning, and re-signing after code edits — currently requires hand-written shell/Python:

- `grep` for conflict markers in a resolved allowlist YAML.
- ad-hoc `python -c` to check the YAML parses and has no duplicate keys.
- `check --rules trust_tier.tier_model` with manual `ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE` env juggling, then parsing the tail of stdout.
- manual reasoning about which entries are stale, which need re-signing, and what `justify` invocations to assemble.

This manual scripting is error-prone on an **audit-critical, HMAC-signed** surface (see the duplicate-key data-loss incident that deleted a valid entry), and it cannot be done cleanly by an agent (no structured output to consume, no custody-safe path).

This was surfaced concretely during the `fix/resume-fork-reemit` merge, whose conflict was in the signed `config/cicd/enforce_tier_model/core.yaml`.

## 2. Goals / Non-Goals

**Goals**
- One unified surface (CLI verbs + a dedicated MCP server) over **all** cicd allowlist gates.
- Structured, machine-consumable results (no stdout-tail parsing).
- A **plan → review → fire** lifecycle that splits cleanly along the existing two-key custody boundary: the agent does all the thinking (including the judge pre-screen) with its OpenRouter key; the operator fires one command in a keyed environment that reviews-drift-checks-applies-signs.
- Auto-act on the **mechanical, non-signed** changes (rotate, union-resolve, prune-unsigned), with hard safety rails.

**Non-Goals**
- No new trust mechanism. Signing remains HMAC, operator-key-only. This packages the *existing* two-key reality; it does not weaken it.
- No rewrite of the existing signed `rotate` / `justify` / `reaudit` internals (Approach A, not C). Consolidation of those into the new subpackage is a possible later increment.
- The agent never holds the HMAC key and never signs. Any agent-facing surface is read-only or propose-only.

## 3. Settled Decisions (from brainstorming)

1. **Consumer/surface:** both CLI and MCP, over one shared core (avoids CLI/MCP drift).
2. **Automation boundary:** auto-act on mechanical (non-signed) parts; signing stays operator-only.
3. **Scope:** all cicd allowlist gates (unified surface), not tier-model only. The judge/HMAC machinery is a tier-model-specific *extension* of the shared model; generic validate/diff/reconcile apply to every gate.
4. **Architecture:** Approach A — a new `allowlist_ops` service module over the existing generic `load_allowlist`, with thin CLI verbs and a thin MCP server as adapters. Signed `rotate`/`justify` internals left in place.
5. **Workflow:** plan → review → fire (job lifecycle), centerpiece of the design.
6. Jobs live in `config/cicd/_allowlist_jobs/` (gitignored; executed jobs retained as an audit record).
7. `job apply` supports an optional `--confirm <short-hash>` integrity pin (printed by `job show`).

### Grounding fact that de-risks "all gates"

`elspeth_lints.core.allowlist.load_allowlist` is already a **generic** loader (its own docstring, `tier_model/rule.py:2110`) shared by tier_model, freeze_guards, component_type, frozen_annotations, symbol_inventory, and trust_boundary — all parsing into common `Allowlist` / `AllowlistEntry` / `PerFileRule` / `FindingKey` types. Gates differ only in a few top-level shapes (`per_file_rules` / `entries` / `allow_hits` / `allow_classes`) and in tier-model's extra judge/HMAC fields. The unified surface builds on this existing abstraction.

## 4. Architecture

New subpackage **`elspeth-lints/src/elspeth_lints/core/allowlist_ops/`**, small focused units:

| Unit | Responsibility |
|------|----------------|
| `gates.py` | The **gate registry** — the single source of truth for all `config/cicd/*` gate dirs: each gate's allowlist dir, schema shape, associated rule ids, and `has_judge_hmac` flag (true only for tier-model today). All other code iterates this registry; no hard-coded gate names elsewhere. |
| `inspect.py` | **Read-only** ops over `load_allowlist`: `status()` (per-gate health: entry counts, ceiling headroom, expiring/expired, stale count), `validate()` (parse + duplicate-key + schema-shape + conflict-marker scan), `diff()` (entries stale / missing / fingerprint-rotation-needed vs current code). |
| `reconcile.py` | **Mechanical mutate** (the auto-act): `reconcile()` — rotate fingerprints, union-resolve conflicts, prune unsigned stale entries — dry-run by default, returning a planned `ChangeSet`; writes only on explicit apply, atomically. |
| `resign.py` | **Propose layer**: `needs_resign()` — which signed entries a reconcile/edit invalidated, each rendered as a drafted signing action. Never signs. |
| `jobs.py` | Job artifact: serialize/deserialize, `show` rendering, drift-precondition capture and re-check, `--confirm` hash. |
| `results.py` | Shared JSON-serializable result dataclasses consumed identically by CLI and MCP. |

**Two thin surfaces:**
- **CLI** — new `allowlist` subcommand group in `cli.py` (handlers call `allowlist_ops` and render): `status`, `validate`, `diff`, `reconcile`, `needs-resign`, `plan`, `job show`, `job list`, `job apply`. Flags: `--gate <id> | --all`, `--format text|json|markdown`, `--apply`, `--confirm <hash>`, `--allow-prune-signed`.
- **MCP** — new dedicated server `elspeth_lints.mcp` (entry point `elspeth-lints-mcp`, same `mcp.server.Server` + stdio pattern as `elspeth-mcp`). Read-only tools: `list_gates`, `allowlist_status`, `allowlist_validate`, `allowlist_diff`, `allowlist_needs_resign`, `allowlist_job_show`. One mechanical-mutate tool: `allowlist_reconcile` (dry-run unless `apply=true`). One planning tool: `allowlist_plan` (writes a job). **No signing tool exists** on the MCP surface.

**Custody boundary (structural, not advisory):** neither surface can sign. Keyless verification runs in shape-only mode (reusing the P1-fix mechanism: `ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing`). Auto-rotating a fingerprint on a *signed* tier-model entry invalidates its signature; `reconcile` performs the mechanical rotation but marks the entry `STALE_NEEDS_RESIGN` and reports it — it never computes a signature.

## 5. The Job Lifecycle (centerpiece)

A **job** is a durable, reviewable artifact capturing the agent's entire intended change.

**Stage 1 — Agent queues (keyless):** `elspeth-lints allowlist plan …` runs dry-run `reconcile` + `needs_resign` + (for new suppressions) the judge pre-screen using the agent's OpenRouter key, and writes one job file to `config/cicd/_allowlist_jobs/<timestamp>-<slug>.job.json` containing:
- `mechanical_changes`: exact ChangeSet (rotations old→new fingerprint, union-resolutions, prunes).
- `signing_actions`: each entry to be (re-)signed, fully specified — `key`, `file_fingerprint`, `ast_path`, `judge_verdict`, `judge_rationale`, `judge_policy_hash`, `owner`, etc. — everything `compute_judge_metadata_signature` needs **except the HMAC key**. For a **re-sign after a pure fingerprint rotation** (code moved, suppression unchanged), the verdict/rationale carry over from the existing entry — only the fingerprint changed, so no new judge run is needed. For a **genuinely new suppression**, the verdict/rationale come from a fresh judge pre-screen (agent OpenRouter key). The job records which case each signing action is, so `job show` makes the distinction visible.
- `preconditions`: fingerprint snapshot of the source files + allowlist state planned against, plus elspeth-lints version and git revision (for drift detection).
- `confirm_hash`: short hash of the job's normative content (for the optional pin).

**Stage 2 — Operator reviews:** `elspeth-lints allowlist job show <job>` renders it human-readably — files changed, rotations/prunes, and exactly which verdicts will be signed with what rationale. This is the trust step.

**Stage 3 — Operator fires (keyed), one command:** `elspeth-lints allowlist job apply <job> [--confirm <hash>]` in a keyed environment, atomically:
1. **Drift guard** — re-check `preconditions` against current code/allowlist. Any drift (fingerprints changed, new dup key, an entry-to-sign no longer matches) → **abort with a precise diff**, sign nothing, advise re-plan.
2. Optional `--confirm <hash>` mismatch → abort (what you reviewed ≠ what would fire).
3. Apply mechanical changes (temp → validate → atomic replace).
4. **Sign** the `signing_actions` with the now-present HMAC key.
5. Re-validate in *required* mode (key present): all signatures verify, gate passes — or roll back.
6. Retain the executed job as an audit record.

The split matches the existing two-key reality: agent (OpenRouter-key judge pre-screen + full packaging) → operator (one keyed command that reviews-drift-checks-applies-signs). No hand-assembling `justify` calls.

## 6. Reconcile Safety Rails

Each rail encodes a specific known failure mode; the mutate path is precondition-based, not best-effort.

1. **Dry-run by default; `apply` atomic-or-nothing.** Writes go temp → re-validate (parse + no dup keys + no markers + gate check shape-only) → atomic replace. Validation fail → no write.
2. **Duplicate-key refusal (data-loss guard).** Before any removal/rotation, assert no duplicate keys; if present, refuse the whole op and name them. (Directly encodes the rotation-tool dup-key data-loss lesson.)
3. **Signed entries never silently dropped or forged.** Rotating a signed entry's fingerprint rewrites the fingerprint and marks the signature `STALE_NEEDS_RESIGN`; never computes a signature. Pruning a signed stale entry requires explicit `--allow-prune-signed`.
4. **Conflict resolution additive-union ONLY.** Auto-resolve only when both sides added different entries (clean union). Competing same-key edits (different fingerprints/judge metadata) → refuse that hunk, leave markers, flag. (The `core.yaml` `explain_row` `fp=5b51…` vs `fp=b815…` case must NOT be auto-unioned.)
5. **Ambiguity surfaced, not guessed.** Reuse the existing rotate N:M pairing: symmetric N:N auto-paired; asymmetric/ambiguous flagged for review.
6. **Ceilings respected.** A reconcile that would exceed a gate's `max_allow_hits` is flagged, not silently exceeded.

Net contract: reconcile does all mechanical toil, but the moment a change touches signed trust or needs judgment, it stops and hands back a precise drafted action.

## 7. Error Handling (by trust tier)

This is an ELSPETH tool, so the tier model applies to it:
- **Job file = OUR data (Tier 1).** A malformed/garbage job at `job apply` → **crash** (no coercion). A drift *mismatch* is a legitimate "world moved" abort with a diff, not corruption.
- **Source code the fingerprints bind to is read at a boundary** → keyless verification runs shape-only, never fabricating a signature.
- **CLI/MCP boundaries return structured error results**, not leaked tracebacks — except genuine Tier-1 corruption, which crashes loudly per doctrine.

## 8. Testing

Each safety rail gets a test proving it BLOCKS the bad case:

| Rail | Test asserts |
|------|--------------|
| Dup-key refusal | Fixture with duplicate key → `reconcile` refuses, names key, mutates nothing |
| Signed-entry protection | Rotation hitting a signed entry → `STALE_NEEDS_RESIGN`, never a forged signature; prune-signed refused without opt-in |
| Additive-union-only | Competing-same-key conflict → refused & flagged; clean both-added → unioned |
| Atomic apply | Post-write validation failure → no file change (rollback), reported failed |
| Drift guard | `job apply` against a mutated precondition → aborts with diff, signs nothing |
| Custody | Keyless `job apply` of signing_actions → aborts; no path signs without the key |

Plus: **gate-shape coverage** — plan→job→apply round-trips (with a *test* HMAC key) across all four schema shapes (`per_file_rules` / `entries` / `allow_hits` / `allow_classes`); **MCP contract tests** — tools return the structured result types, read-only tools never mutate, `allowlist_reconcile` honors dry-run-by-default; tests use the **real `load_allowlist`** and real/fixture gate dirs — no bypassing production paths.

## 9. Suggested Phasing (for the implementation plan)

1. **Phase 1 — Read-only core + CLI + MCP:** `gates.py` registry, `inspect.py` (`status`/`validate`/`diff`), `results.py`, the `allowlist status|validate|diff` CLI verbs, and the read-only MCP tools. Immediately replaces the manual grep/dup-key/check scripting. Lowest risk (no mutation).
2. **Phase 2 — Reconcile (mechanical mutate) + safety rails:** `reconcile.py`, the `reconcile` verb + `allowlist_reconcile` MCP tool, all rails + their blocking tests.
3. **Phase 3 — Job lifecycle:** `jobs.py`, `resign.py`, `plan` / `job show` / `job list` / `job apply`, drift guard, `--confirm`, the keyed signing execute, two-key round-trip tests.

Each phase is independently shippable and testable.

## 10. Open Items / Future

- Possible later increment (Approach C): consolidate the existing `rotate.py` / reaudit / `check-rotation-audit` logic into `allowlist_ops` to de-bloat the 3061-line `cli.py`. Out of scope here.
- A CI fast-check using `allowlist validate --all` (catch dup keys / markers / drift early) — natural but optional.
- Whether the gate registry should also drive the pre-commit hook `files:` triggers (single source of truth for "what file changes touch which gate") — noted, not in scope.
