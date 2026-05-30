# cicd-judge-cli — end-to-end review (2026-05-30)

Branch `feat/cicd-judge-cli-prototype` @ `8598934e4`, diff baseline `f77096d4e..HEAD`
(30 commits, 120 files, +37,639/−628). Target: merge into RC5.2 **secure and reliable**.

Method: hybrid — 8 specialist agents (Opus, read-only) gathered breadth; the security
core (`judge.py`, `allowlist.py`, `source_excerpt.py` paths) and every load-bearing
finding were read/verified directly by the reviewer.

**Threat model (operator-clarified):** the LLM judge does NOT replace human review. It is
a guardrail to stop an automated coding *agent* from landing an inappropriate allowlist
suppression; human review is the authority on top. Severities are calibrated to that:
HMAC = audit-trail tamper-evidence + keyless-agent barrier, not access control against a
determined human.

---

## Top-line verdict

The engineering quality is high and the cryptographic/integrity core is **sound**. But the
system is **NOT merge-ready as-is**. It is a strong *signature-integrity* gate and, as
wired today, an incomplete *coverage* gate: it keeps genuinely-new agent suppressions
honest, but the existing 787-entry backlog is ungoverned and three grandfathering seams let
an agent bypass the gate in common cases. Plus the branch fails its own budget gate.

Blockers below are grouped by **root cause** (multiple agents converged on the same defects).

---

## P0 — merge blockers

### B1. The branch fails its own budget gate (RED on a faithful CI run)
Measured directly through the data + cross-checked vs `_defaults.yaml` caps:

| category | current | cap | over |
|---|---|---|---|
| allow_hits | 787 | 652 | +135 |
| total_entries | 876 | 744 | +132 |
| permanent_allow_hits | 370 | 320 | +50 |
| permanent_total_entries | 459 | 412 | +47 |

`trust_tier.tier_model` fails on any budget violation, so `ci.yaml`'s static-analysis job is
RED on budget alone. **Fix:** prune stale fingerprints via `rotate`/cleanup, or — if all
entries are legitimate — make the cap bump a deliberate, reviewed ratchet, not an auto-absorb.
Do not merge a tree that fails its own gate. (caps: `enforce_tier_model/_defaults.yaml`)

### B2. Grandfathering bypass — the gate's core purpose is defeatable while the backlog is unjudged
Three agents triangulated one root cause. **Pre-judge entries (`judge_verdict is None`) skip
ALL binding/HMAC verification** (`allowlist.py:451`: `if source_root is not None and
judge_verdict is not None`), and **C1 grandfathers them fp-agnostically** (`judge_coverage.py`
`_discriminator` strips the `fp=` suffix, and `_judge_metadata_payload` returns `None` for
pre-judge → `None == None` → grandfathered with no judge metadata required).

Exploitation paths (all confirmed; 0 of 787 entries are judged today → surface is large):
- **New unjudged entry via collision** (CI agent): add an `allow_hits` entry reusing an
  existing `(file, rule, symbol, owner, reason)` tuple (e.g. ubiquitous `owner: architecture`)
  with the `fp=` of any finding to silence it → grandfathered, no judge, no HMAC.
- **Fingerprint re-target onto modified code** (security agent F3): re-point a pre-judge
  entry's `key` to a new `fp=` after changing the code; C1 grandfathers fp-agnostically and
  `check-rotation-audit` is satisfied by an unsigned, agent-writable manifest line.
- **Renewal by date-edit** (economics agent): `expires` is excluded from the `allow_hits`
  discriminator (but *included* for `per_file_rules` — telling asymmetry), so the 396 entries
  expiring 2026-08 can be bulk-renewed with zero re-judgment.

**Unifying fix:** fold the binding identity (`fp=` / `file_fingerprint` / `ast_path`) and
`expires` into the new-vs-grandfathered decision in `judge_coverage.py:_discriminator`; force
any `fp` change on a pre-judge entry through `justify`. (The gate is "eventually consistent"
— once the backlog is judged, a pre-judge HEAD entry no longer matches a judged baseline and
falls to the violation path. The fix closes it regardless of migration state.)

---

## P0(operator) — must confirm before merge (not verifiable from the worktree)

### O1. Key custody is the linchpin
The HMAC is symmetric → every key holder can forge. If the autonomous coding agent's
environment contains `ELSPETH_JUDGE_METADATA_HMAC_KEY`, it can skip `call_judge`, hand-write
`judge_verdict: ACCEPTED` with a fabricated rationale + correct (publicly computable)
`file_fingerprint`/`ast_path`, sign it, and pass **every** automated gate — the guardrail
collapses and the trail contains a validly-signed fabricated Opus verdict.
**Required rule:** the signing key is **operator-only**, absent from any autonomous-agent
environment. An agent may *propose* a `justify` invocation; only an operator-held environment
signs. (Asymmetric signatures would remove this surface structurally — design direction.)

### O2. Branch protection / required checks / CODEOWNERS
The judge-gates workflow has **no aggregate success job** (unlike `ci.yaml`'s `ci-success`), so
C1/C3/VAL block merge only if individually required. Budget caps and the workflow files are
**editable in the same PR they gate**. Fork-PR merge-gating depends on these settings. Confirm:
(a) C1 (`check-judge-coverage`), C3, VAL are required status checks, or add an aggregator job;
(b) `config/cicd/**` + `.github/workflows/*` are CODEOWNERS-locked; (c) a fork PR's shape-only
green check cannot satisfy merge without a trusted `required`-mode re-run.

---

## Fix-before-merge — concrete code defects

### C1. Three Tier-1 error-semantics regressions in web dogfood call-sites
- `sessions/service.py:2642` — `cast(str, options["prompt_template"])` (runtime no-op) **removed
  a Tier-1 read guard**; a non-str now silently takes the mismatch branch and raises the *wrong*
  error. Restore the `isinstance` raise.
- `sessions/routes/sessions.py:50` — type anomaly raises `TypeError` (bypassing the structured
  `AuditIntegrityError` handler at `app.py:513`) while the *next block* correctly raises
  `AuditIntegrityError`. **Confirmed inconsistency.** Raise `AuditIntegrityError`.
- `sessions/service.py:~2124` — post-commit cleanup now propagates a misleading bare `OSError`
  (can't distinguish "delete failed" from "delete succeeded, orphan cleanup failed"). Raise a
  named exception that states the delete committed.

### C2. Path-insensitive derived-name tracking → over-suppression (rule precision)
`tier_model/rule.py` tracks boundary-derived names in a single flat set with **no branch
isolation** (no `visit_If`/`visit_Try`/`visit_While` — confirmed). Across `if`/`try` branches
the last lexical assignment wins, so a real R1/R5 on a **non-boundary** value can be silenced
inside a legitimately-decorated function — the exact over-suppression the gate exists to
prevent, and an aliasing shape an agent could construct deliberately. Untested. Mitigated by the
`R_TB_SUPPRESSED` audit stream + human review. **Fix:** intersection-at-join branch handling
(biased toward visible findings). Also delete the dead-but-tested `compute_derived_names` /
`_local_bindable_names` (a parallel dataflow production never calls — false test confidence +
No-Legacy-Code violation).

---

## Hardening — strong follow-ups (bounded by human review; not blockers)

### H1. Judge prompt-injection (judge.py)
Structural injection is effectively closed (real role split, JSON-fenced untrusted data,
fail-closed strict parsing, exact-match symbol resolution). Residual is semantic persuasion of
the verdict model via the `rationale`:
- **Rationale-blind second judge** that must concur for ACCEPT (defeats rationale-only persuasion).
- The prompt's own "decorator may be off-screen, don't assume undecorated" instruction is a
  sanctioned blind spot — **inject the linter's actual decorator list** instead of trusting the claim.
- **Fence `similar_entries`/`candidate`/`owner`** in `_UNTRUSTED_DATA_INSTRUCTIONS` (currently only
  `agent_rationale.text`/`surrounding_code.text` are named) → closes the precedent-poisoning loop.
- Cap/validate `--owner` (currently unbounded, unfenced, re-enters future prompts).

### H2. Lifecycle wiring (the system is structurally decaying as wired)
- **Wire `reaudit` into a scheduled CI job** (weekly, `--limit`-incremental, report-only) — today
  the only decay-sweep is on-demand. Start on the R1/R5 (67%) then R6 (silent-except, 163) clusters.
- **Set C3 `--max-overrides`** (absolute cap) — C3 is a vacuous PASS until ≥10 judged entries exist,
  and the rate alone is dilutable.
- **Backfill judge metadata** for the 787 backlog on a deliberate schedule (also closes B2's surface).
- **Reframe VAL** as a "prompt/model regression tripwire" (12 cases, tolerates 1 miss) — not a quality
  guarantee — and grow its adversarial coverage (add F1/F2/F3-style cases, all `expected: BLOCKED`).

### H3. No-silent-failures on the verify downgrade
Make `shape-only-when-key-missing` self-documenting: emit a prominent stderr + step-summary warning
("HMAC verification SKIPPED — shape-only; cannot detect forged judge metadata"). On fork PRs, fail
closed on any *new/changed* judged entry rather than shape-accepting it.

### H4. Minor
- `reaudit.py:1266` classifies an *unparseable* source file as `ENTRY_OBSOLETE` → a decay sweep run
  during a transient syntax error could prune load-bearing entries. Add a `SOURCE_UNPARSEABLE` divergence.
- Test theater: `test_justify.py:1829` greps ~15 prompt substrings (the anti-pattern the project's own
  doctrine names) — keep as a refactor canary, rename to advertise it, don't expand.
- Confirm the CI lane that runs the judge-transport tests installs the optional `[judge]`/`openai` extra
  (else those negative-case tests silently collect nothing).
- `generation.py:111` R5 remediation: message says "must be a mapping" but check is `type is dict` — align;
  applied inconsistently across sibling validators.

---

## Scope

- **Split out (creep):** the config-protocol extractions — `contracts/config/protocols.py`,
  `contracts/contexts.py`, `engine/orchestrator/types.py` + `core.py`/`processor.py` rewrites. Correct
  L0/L2-purity work, but unrelated to cicd-judge and **verified not required by the gate** (the branch
  does not escalate TYPE_CHECKING warnings→failures). Inflates the review surface; own PR.
- **Flag as deliberate:** the prompt-shield behavioral change in `interpretation_state.py:248-291`
  (producers now tracked for `on_error`/`routes`/`fork_to`, not just `on_success`) is a real
  security-contract change bundled in — confirm test coverage for the new evasion paths.

---

## What is genuinely strong (credit)

- **Crypto core (verified by reviewer):** full `file_fingerprint` inside the HMAC payload, symmetric
  sign/verify, deterministic canonical JSON (`sort_keys`, `separators`), `hmac.compare_digest`
  constant-time, no truncation/length-extension exposure. **Triple binding**: load-time fingerprint
  recompute + match-time `ast_path` check + HMAC over all of it. `_validate_judge_metadata_atomic`
  enforces 8 all-or-nothing invariants; persisted `BLOCKED` rejected as corruption.
- **Atomic IO:** `atomic_io.py` is a correct, durable, lost-update-safe primitive (same-fs temp,
  `O_EXCL`, short-write detection, file+dir fsync before AND after `os.replace`, flock across
  read-modify-write). Both allowlist write paths use it. Non-atomic writes are regenerable telemetry only.
- **Fail-closed everywhere:** judge transport/contract/config errors → exit 2, never an implicit ACCEPT;
  `finish_reason='length'` rejected; allowlist load crashes on any anomaly; reaudit surfaces
  `JUDGE_CALL_FAILED` as a visible divergence, not a swallowed pass.
- **Judge prompt hardening:** genuine system/user role split, untrusted text JSON-fenced as data,
  `temperature=0`, served-model recorded, `trust_env=False` on the http client.
- **Tests:** real negative cases against unmocked production code (HMAC tamper per-field, fingerprint
  drift, cross-file transplant, unsigned rejection, fail-closed transport); the `test_ref` rule itself
  rejects mocked/stub tests (anti-theater baked in).
- **History scrub complete:** no HMAC key material in HEAD or any ref; only the obvious test placeholder.
- **L0 purity:** `contracts/trust_boundary.py` imports only stdlib; metadata-only passthrough; validates
  `tier==3` + `source_param ∈ signature` at decoration; rejects stacking.

---

## Bottom line

Close **B1 + B2** (budget reconcile + grandfathering/discriminator fix), confirm **O1 + O2** (key
custody + branch protection), and fix the **C1/C2** code defects — then this is a secure, reliable
gate worth merging. The H-items convert it from "honest at the margin" to "bounded in aggregate" and
should follow close behind. The cryptographic and infrastructure foundations are excellent; the gaps
are in coverage wiring and a handful of dogfood-call-site regressions, not in the core design.

---

## B1 — CORRECTED diagnosis (2026-05-30, evidence-based)

The original B1 framing ("budget gate is red; bump caps") was **incomplete**. Root-cause
investigation (gate run at merge-base vs branch HEAD, per-symbol discriminator) establishes:

### What's actually wrong
| probe | unsuppressed | stale | budget over |
|---|---|---|---|
| RC5.2 merge-base (`f77096d4`) | **0** | 2 | marginal (322/320) |
| branch HEAD | 83 | 210 | 4 caps, large |

The branch reworked the tier_model **detection + fingerprint algorithm** (`rule.py` +594/−48,
`ast_walker.py` +212/−5, adds explicit `ast_path`). This globally re-fingerprinted the
allowlist — **byte-identical source files lost their suppression** (chat_solver.py,
telemetry_phase8.py, composer/service.py: 27 findings, unchanged source, dropped entries).
This is the documented `feedback_ast_shift_fingerprint_rotation` /
`feedback_merge_scale_cicd_pruning` desync, **not** a backfill of new trust-boundaries and
**not** offensive code-fixes. tier_model entries carry **no HMAC** → re-binding is signature-safe.

### Validated remediation METHOD (proven in /tmp, read-only)
Start from a **known-green baseline allowlist**, rotate it forward to the new algorithm:
- merge-base allowlist → forward-rotate → **23 residual unsuppressed (all interpretation_state.py), 27 stale, ~2 budget**.
- The 27 dropped-file findings re-suppress **with provenance carried forward** (merge-base reasons intact).

Per-symbol discriminator on interpretation_state.py (current findings vs merge-base entries):
**EXCESS = 0** — the new detector surfaces *zero* findings the old one didn't; 3 over-coverage
entries correspond exactly to 3 defensive patterns the branch's refactor *removed*
(`while isinstance(stream,str)`→`while stream`, etc.). So the branch is tier-*cleaner* here;
all 56 interpretation_state findings are **verified carry-forward**, not new debt.

Residual 23 don't auto-pair because rotate's symmetric-N:N guard
(`_entries_share_rotation_metadata`) correctly refuses to guess which distinct reason maps to
which new fingerprint. Resolution: snippet-match (`finding.message` ↔ entry reason), emit
current-fp entries with fresh provenance-citing reasons, append `old_key→new_key` to
`.elspeth/rotations.log` (for the `check-rotation-audit` gate).

### CRITICAL sequencing finding
The branch is **45 commits behind origin/RC5.2** (merge-base 05-26, RC5.2 tip 05-29). RC5.2 has
since done overlapping work: `747a232f6` (re-reconciled interpretation_state.py allowlist),
the engine/orchestrator god-class decomposition, and interpretation_state.py logic changes
(`2e8feffde`, `31174f10c`). **Reconciling the allowlist against the merge-base now is throwaway:**
integration with RC5.2 tip will desync RC5.2's freshly-reconciled allowlist via the branch's
new algorithm, forcing re-reconciliation. The method is correct; the baseline must be
**RC5.2-tip's allowlist applied AFTER integration**, not the stale merge-base now.

**Recommended sequence:** (1) rebase/merge branch onto RC5.2 tip → (2) reconcile allowlist
against the integrated state using the validated method → (3) budget caps (ADR-gated) → (4) verify gate exit 0.

Budget caps are identical across merge-base/RC5.2-tip/branch (`max_allow_hits: 652`); RC5.2
itself ships ~2 over on permanent caps, so a small cap bump is needed regardless and is
inherited, not branch-introduced.
