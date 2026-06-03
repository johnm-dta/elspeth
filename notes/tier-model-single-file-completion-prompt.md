# PROMPT — Drive one tier-model backlog file to completion (RC5.2)

> Paste the section below into a fresh session. It is self-contained: assume zero
> prior context. Everything you need to start is here or behind the cited memory /
> skill / notes references. Written 2026-06-03 by the session that landed commit
> `3b18825e7` (the `resolve_runtime_yaml_paths` source.options fix).

---

## Your task

The ELSPETH tier-model gate (`trust_tier.tier_model` elspeth-lints rule) currently
has **~52 unsuppressed findings** with **no allowlist entry** — untriaged
defensive-pattern hits (`.get()`/`getattr`/`isinstance`/broad-except/etc.) the
re-sign campaign has not yet reached. The web *allowlist* (`web.yaml`, 149 entries)
is fully re-signed; this backlog is the campaign's next phase.

**Select ONE file from the backlog and work every one of its findings to a terminal
disposition, then take it through CI to completion** — meaning: that file has
**zero** unsuppressed tier-model findings, every "defend" decision is judge-signed,
and tests + ruff + mypy are green. Code fixes are committed by you; judge-signed
allowlist entries are *proposed* by you and *signed by the operator* (HMAC custody,
see below).

This is NOT a bulk sweep. One file, done properly, end to end. Do not widen scope to
other files (the campaign tracks those separately). Do not defer findings within your
chosen file — every one gets fixed, defended-and-signed, or surfaced to the operator
with a concrete proposal in the same turn (no "log a ticket", no
"deferred to follow-up": see memory `feedback_default_is_fix_not_ticket`,
`feedback_no_unilateral_deferral`).

## Step 0 — Orient (read before touching anything)

Read these first — they are load-bearing:
- `CLAUDE.md` (the project one) — Three-Tier Trust Model, Offensive-not-Defensive,
  No-Legacy, HMAC custody, the justify/reaudit/migrate-judge-scope CLI reference.
- Memory: `project_tier_model_resign_campaign_2026-06-02.md` (campaign state, esp.
  UPDATE 12/13), `project_trust_tier_custody_doctrine_2026-06-01.md` (origin-not-
  courier, persisted-config-is-T3), `feedback_tooling_auditability_irap_not_landscape`
  (don't gold-plate the tooling — a stale fingerprint is a chore, not evidence
  tampering), `feedback_locked_in_buggy_expectations` (test failures after a
  structural fix are the bug landing visibly — update the tests, do not revert the
  fix), `reference_tier_model_fingerprint_rotation_tool`,
  `project_tier_model_python_version` (main `.venv` is 3.13; FPs are
  Python-version-sensitive — use main's venv).
- Skills: `tier-model-deep-dive` (the decision tables — coercion vs fabrication,
  what-am-I-reading tier table), `logging-telemetry-policy` (needed if you pick a
  telemetry file — telemetry is best-effort but "no silent failures"),
  `engine-patterns-reference`.
- Process: `notes/tier-model-bulk-remediation-playbook.md` (the campaign's worked
  process incl. the decorator/scope_fingerprint notes).

Ask the operator whether to create a worktree (default yes for code work; memory
`feedback_default_to_worktree`). If they say work in place on RC5.2, do that. If you
worktree, mind the venv/hook gotchas in the worktree memories.

## Step 1 — Enumerate the backlog and pick your file

Run the keyless gate and get the per-file finding counts:

```bash
cd /home/john/elspeth
env PYTHONPATH=elspeth-lints/src ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing \
  .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model \
  --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model 2>/dev/null \
  | grep -E "^[a-z_/]+\.py:[0-9]+:[0-9]+: R[0-9]+:" \
  | sed -E 's#^(src/elspeth/)?([a-z_/]+\.py):.*#\2#' | sort | uniq -c | sort -rn
```

For the RAW findings in JSON (top-level is a LIST; fields are `file_path`, `line`,
`rule_id`, `fingerprint`, `message`), append `--format json` and parse the list.

**Selection criteria** — pick a file with a *coherent trust-tier story*, not just a
low count. As of 2026-06-03 the backlog split was: `web/composer/tools/generation.py`
17, `plugins/sources/azure_blob_source.py` 10, `telemetry/manager.py` 5,
`web/execution/routes.py` 3, `plugins/sources/csv_source.py` 3, plus 2s and 1s
(`web/blobs/service.py`, `core/retention/purge.py`, `web/composer/tools/blobs.py`,
several singletons).

**Recommended first target: `telemetry/manager.py` (5 findings, all R6/R4 exception
swallows).** One coherent question governs all five: *is each swallow legitimate
under the best-effort telemetry policy, or is it a silent failure?* The
`logging-telemetry-policy` skill is decisive here ("no silent failures — every
emission point must send or explicitly acknowledge 'nothing to send'"). It exercises
the DEFEND path cleanly and is a single subsystem. `core/retention/purge.py` (2) or
`plugins/sources/csv_source.py` (3, a Tier-3 source-coercion story) are good smaller
alternatives. Avoid `generation.py`/`azure_blob_source.py` for a first pass — large
and already on the operator's queue.

State your pick and *why* before proceeding. Call `advisor()` before committing to
the triage approach.

## Step 2 — Triage each finding to a terminal disposition

For every finding in your file, decide one of:

- **FIX (real defect).** The pattern masks data we authored/guaranteed (T1/T2), or
  swallows a real error, or fabricates absent data. Fix it: offensive assertion +
  direct access for guaranteed data; proper error handling / row-diversion for
  recoverable faults; record-absence-as-`None` not a fabricated default. Then
  reconcile tests — a wave of failures is the bug landing visibly; **update the
  tests to the corrected contract, do NOT revert the fix** (this is exactly what
  bit the prior session; see `feedback_locked_in_buggy_expectations`). Make mocks
  realistic rather than papering over.

- **DEFEND (genuine boundary / by-design).** The value is external-origin
  (origin, not courier — an LLM/HTTP/subprocess/user-config value is T3 even inside
  our dataclass), or the pattern is the idiomatic optional read of a genuinely
  optional structure, or a telemetry best-effort swallow that *does* acknowledge
  "nothing to send". Write an HONEST rationale (the test: would it survive formal
  inquiry? — but IRAP-grade, not Landscape-grade; don't gold-plate). Then judge it
  (Step 3).

Apply the trust model rigorously: **origin sets the tier, not the container or the
store.** Persisted user/composer config read back from our DB is still T3 (re-validate
on read, quarantine/diagnostic — never crash the tool). Our own audit rows/checkpoints
are T1 (crash on anomaly). `.get/getattr/isinstance` are forbidden on T1/T2 but
legitimate at a T3 boundary. `hasattr` is unconditionally banned.

If a finding is genuinely ambiguous (is this value ours or theirs?), ASK the operator
with a concrete proposal — do not guess and do not defer.

## Step 3 — Judge the DEFEND findings (you PROPOSE, operator SIGNS)

**HMAC custody is absolute (CLAUDE.md "CICD Judge Gate"): the signing key
`ELSPETH_JUDGE_METADATA_HMAC_KEY` is operator-only and MUST NOT be in your
environment.** `justify` and `migrate-judge-scope` WRITE signed metadata → you may
only *propose* the exact command; the operator runs it. `reaudit` is read-only and
you MAY run it yourself with an OpenRouter key, on `--judge-transport openrouter`.

Workflow for each DEFEND:
1. Pre-screen with `reaudit` (read-only) to confirm the judge will likely ACCEPT and
   to refine the rationale. Use `--judge-transport openrouter` (deterministic,
   temperature=0).
2. Draft the rationale (save under `notes/resign/rationales/` or your own dir).
3. Hand the operator the exact `justify` invocation, e.g.:

```bash
env ELSPETH_JUDGE_METADATA_HMAC_KEY=<key> PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli justify \
  --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
  --file-path <relative/path.py> --rule <R#> --symbol <Dotted.Symbol> \
  --fingerprint <fp> --rationale "$(cat <rationale_file>)" --owner "$USER" \
  --judge-transport openrouter --format json
```

**Transport lesson (learned the hard way):** the `agent` transport can flake on a
complex function with `Reached maximum number of turns (12)` and emit no verdict —
re-run, or switch that entry to `--judge-transport openrouter` (no investigation
loop to exhaust; CLAUDE.md's recommended deterministic path). `--judge-tools
readonly` is agent-only; omit it on openrouter. New entries bind v2
`scope_fingerprint` automatically.

## Step 4 — Mind the fingerprint cascade

- The per-finding `fp = sha256(rule_id | ast_path | node_dump)`; `ast_path` is
  MODULE-ROOT-relative. Adding a **module-level import** shifts `body[N]` for every
  top-level node below it → changes the fp of all findings beneath. A
  `@trust_boundary` decorator is local (no sibling shift) but the import it needs
  cascades. Check the import before estimating blast radius.
- `scope_fingerprint` = `ast.dump()` of the enclosing function/class. Editing ANY
  line in a function re-hashes it → drifts the v2 signature of every *other*
  judge-signed entry in that same function → those need an operator re-sign. (This
  is the exact thing that put `42996be9` into the re-sign queue when the source.options
  fix touched `resolve_runtime_yaml_paths`.) Expect it; tell the operator which
  sibling entries you drifted.

## Step 5 — CI / commit mechanics (critical — read before committing)

The `elspeth-lints-trust-tier` pre-commit hook is `files:
^(config/cicd/enforce_tier_model/|elspeth-lints/.../trust_tier/)` with
`pass_filenames: false`. Meaning: **it only FIRES when you stage an allowlist (or
rule) file, and when it fires it scans the WHOLE tree** — so it will FAIL on the
remaining backlog until the entire campaign is green. Consequences:

- **FIX-only file** → its findings vanish, no allowlist entries needed. Stage only
  the source + test files (no `config/cicd/...`). The trust-tier hook stays dormant
  (no allowlist file staged), other hooks (ruff/mypy/secret-scan/freeze/composer)
  run on your staged files and must pass. You can commit this cleanly and green.
- **DEFEND findings** → the judge-signed entries live in `web.yaml` (or the relevant
  subsystem yaml). Staging that file trips the hook → whole-tree scan → fails on the
  backlog. So the signed entries CANNOT be committed green until the backlog is
  exhausted. Coordinate with the operator: the signed-entry commit batches with the
  rest of the campaign. Your code fixes commit independently (no allowlist file
  staged).
- The pre-commit dispatcher passes `--files <staged>` (stash=False) — safe to commit
  with other uncommitted work present; it will NOT stash/lose the working tree. But
  still stage *exactly* your files (`git add <explicit paths>`), never `git add -A`.
- Commit on RC5.2 (not the default branch `main`, so no branch needed — the campaign
  commits directly to RC5.2). Only commit/push when work is done and verified.
  End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Step 6 — Verify and report

Done criteria:
- Keyless gate shows **0 unsuppressed findings** for your file (filter the
  enumeration on your exact `file_path`).
- All FIX code committed; tests + `ruff check` + `mypy` green. Run the real suite as
  plain `pytest tests/<relevant>/` — do NOT pass `-o addopts=""` (that force-runs
  infra/slow tests; memory `feedback_pytest_default_selection`).
- All DEFEND rationales drafted, reaudit-validated, and exact `justify` commands
  handed to the operator (with any drifted-sibling re-signs flagged).
- Update memory `project_tier_model_resign_campaign_2026-06-02.md` with an honest
  UPDATE entry: what you fixed, what you defended (and its operator-sign status),
  what cascaded.
- Call `advisor()` before declaring done.

Report plainly: findings resolved, FIX vs DEFEND split, what's committed vs awaiting
operator HMAC sign, gate state for the file. "I don't know" beats a confident wrong
lineage; report failures with their output.
