# Judge-signature handoff: agent stages, operator signs

This document describes the two-actor workflow for acquiring and rotating signed
judge metadata on the `trust_tier.tier_model` allowlist, and the HMAC key custody
rule that makes it safe. It is the operator/agent handoff for the
`stage_scan -> sign-bundle` seam and the `elspeth-judge` MCP server.

## The custody rule ([O1])

The judge-metadata signature is an **HMAC** — a symmetric MAC. Any holder of
`ELSPETH_JUDGE_METADATA_HMAC_KEY` can forge a signature: hand-write
`judge_verdict: ACCEPTED` with a fabricated rationale over a publicly-computable
fingerprint, sign it, and pass every gate. The whole design follows from that
single fact (invariant elspeth-b3a3335c9f, *[O1] operator-only HMAC custody*;
the CI-exposure corollary is elspeth-2b351cd004):

- **An agent never holds the key.** Agents may *propose* work — survey the tree,
  stage a bundle, run a non-authoritative preview judge — but the authoritative
  verdict for a finding is only ever minted inside the operator-keyed step.
- **Signing never runs in CI.** The key must never be reachable from
  PR-controlled code. CI keeps *verifying* (`check-override-rate`,
  `check-judge-quality`); it never signs. This is enforced as a standing
  regression guard by `tests/unit/elspeth_lints/test_meta_ci_never_signs.py`,
  which fails if any signing verb is added to a `run:` step of
  `.github/workflows/enforce-allowlist-judge-gates.yaml`.

**Staging asserts; firing verifies.** A staged bundle carries *zero* authority.
Everything it claims (which entries drifted, which findings are orphaned, which
keys re-key) is an assertion the operator step re-derives from the live source
tree before it writes anything. The bundle is a worklist and an audit record,
not a grant.

## Actors and the seam

```
  AGENT (no key)                          OPERATOR (key-bearing shell)
  ------------------------------          ----------------------------------
  elspeth-judge MCP server                elspeth-lints CLI
    stage_scan   -> worklist bundle  -->  sign-bundle <bundle.json>
    stage_preview (advisory verdict)        (re-verifies the bundle against the
    stage_status (paste-ready cmd)           tree, THEN fires the real judge /
    stage_rekey  -> rekey bundle     -->     re-keys; aborts on any staleness
    verify_signatures (shape-only)           BEFORE a single write)
                                          rekey --in <bundle.json>
```

Bundles are written under `.elspeth/staged-reviews/<bundle_id>.json`. The agent
side is structurally key-free: the MCP server refuses to start a tool handler if
`ELSPETH_JUDGE_METADATA_HMAC_KEY` is present in its environment (fail-closed,
checked *before* any optional import), so the agent surface can never co-locate
with the key.

## The `elspeth-judge` MCP server (agent side)

Registered in `.mcp.json` as `elspeth-judge`, launched as
`python -m elspeth_lints.mcp --root src/elspeth --allowlist-dir
config/cicd/enforce_tier_model --staged-dir .elspeth/staged-reviews` (with
`PYTHONPATH=.../elspeth-lints/src`). It needs the `[mcp]` extra; `stage_preview`
additionally needs `[judge-agent]`. All five tools fail closed when the HMAC key
is present in the environment.

| Tool | Key-free? | LLM? | What it does |
| --- | --- | --- | --- |
| `verify_signatures` | yes | no | Read-only, **always shape-only** signature diagnosis of the tier_model allowlist. The authoritative HMAC recompute is the operator CLI `diagnose`, not this tool. |
| `stage_scan` | yes | no | Survey source tree + allowlist into an authority-free worklist bundle across four lanes — `drift_repair` / `rotation` / `stale_delete` / `new_judgment`. Args: optional `bundle_id`, `staged_by`. |
| `stage_status` | yes | no | Summarise a staged bundle (per-lane/kind counts, preview outcomes) and emit the paste-ready operator `sign-bundle` command. Arg: `bundle_id` (required). |
| `stage_preview` | yes | yes (read-only agent judge) | Run the read-only agent judge over each `new_judgment` action and record a **non-authoritative** preview verdict (`authoritative=False`); surfaces BLOCKED reasons. Never signs. Arg: `bundle_id` (required). Needs `[judge-agent]`. |
| `stage_rekey` | yes | no | Enumerate currently-valid judge-gated entries and flag broken ones into a rekey bundle, recording env-var **names** only — never key bytes. Args: `old_key_env`, `new_key_env` (required), optional `bundle_id`, `staged_by`. |

`stage_scan` feeds the rotation planner only non-judge-gated entries
(`exclude_judge_gated=True`), so the rotation lane serves the 17 non-judge-gated
pre-judge entries and never the 388 judge-gated ones; an fp-shifted judge-gated
entry routes to `drift_repair` only.

## Operator commands (key-bearing shell)

Run these only in an operator-controlled shell that holds
`ELSPETH_JUDGE_METADATA_HMAC_KEY` (and, for the LLM lanes, `OPENROUTER_API_KEY`).
Both commands re-derive every binding from the live tree and abort on any
staleness *before* the first write.

### `sign-bundle` — fire a staged review bundle

```
elspeth-lints sign-bundle <bundle.json> --owner <operator-id> [options]
```

This is the **only** place a judge signature is minted from a bundle. The verify
phase (re-check the whole bundle against the tree) is the all-or-nothing gate; on
any mismatch it aborts before writing anything. The execute phase is per-action,
non-transactional (a mid-bundle real-judge BLOCK after confirmation leaves
earlier-accepted actions written and restores/skips the blocked one), and exits
non-zero with a per-action report.

Per lane:
- `drift_repair` and `new_judgment` run the **real judge** — re-judging, never
  carrying a stale verdict forward over changed content (that would be the [O1]
  forgery). A contradicting BLOCK is surfaced and **not** signed; on BLOCK the
  popped stale entry is restored intact.
- `rotation` re-binds non-judge-gated keys with no judge.
- `stale_delete` removes an orphaned entry, surgically.

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `<bundle>` (positional) | — | Path to the staged review bundle JSON. |
| `--owner` | required | Audit identity recorded on freshly signed entries. |
| `--root` | `src/elspeth` | Source tree to re-scan for the entries' findings. |
| `--repo-root` | none | Repo root for trust-boundary scanners (omit for tier-model-only default). |
| `--allowlist-dir` | `config/cicd/enforce_tier_model` | Per-module allowlist YAML to repair in place. |
| `--operator-override` | off | Forward `--operator-override` to each justify call (requires the override-token env, exactly like `justify`). |
| `--max-tokens` | none | Override judge response `max_tokens` per call. |
| `--dry-run` | off | Print verify + per-lane plan; no judge call, no writes. |
| `--yes` | off | Skip the interactive confirm before the destructive write phase. |
| `--format` | `text` | Per-entry justify output (`text`/`json`). |
| `--judge-transport` | `openrouter` | Provider that produces the verdict. |
| `--judge-tools` | — | Judge tool configuration (mirrors `justify`). |

Override-token environment (only when `--operator-override` is passed):
`ELSPETH_JUDGE_OVERRIDE_TOKEN` + `ELSPETH_JUDGE_OVERRIDE_TOKEN_SHA256`.

A **dup-key** bundle (the same key occurs more than once in a file) aborts with
`return 2` and both copies intact — `sign-bundle` refuses rather than silently
deleting one.

### `rekey` — rotate the HMAC key

```
elspeth-lints rekey --in <bundle.json> --old-key-env <OLD_VAR> --new-key-env <NEW_VAR>
```

A scheme-preserving, **signature-only** swap: it verifies every judge-gated entry
under the OLD key, recomputes its signature under the NEW key, and atomically
rewrites *only* the `judge_metadata_signature` line (binding + audit lines stay
byte-identical). It re-derives the full judge-gated set from the live tree at fire
time and is idempotent/re-runnable — Pass-1 accepts an entry verifying under OLD
*or* NEW, Pass-2 skips already-NEW entries — so a partial/interrupted run
self-heals on re-run instead of bricking. An entry verifying under **neither**
key aborts the whole run (no laundering of a broken entry). An env-name
flag/bundle mismatch aborts before any write. Requires **both** key env vars
(named by `--old-key-env` / `--new-key-env`); fails closed without them. The key
bytes never appear on the CLI — only the *names* of the env vars holding them.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--in` | required | Path to the staged rekey bundle JSON (its `RekeyPlan` env-var names cross-check the flags). |
| `--old-key-env` | required | NAME of the env var holding the OLD (current) key bytes. |
| `--new-key-env` | required | NAME of the env var holding the NEW (target) key bytes. |
| `--root` | `src/elspeth` | Source tree (for the canonical-pair baseline-regen gate). |
| `--allowlist-dir` | `config/cicd/enforce_tier_model` | Per-module allowlist YAML to re-key in place. |
| `--dry-run` | off | Print verify + planned re-key count; write nothing. |
| `--yes` | off | Skip the interactive confirm before the destructive write phase. |

### `RekeyPlan.keys` / `broken_keys` are advisory preview — never a guarantee

The rekey bundle stores two lists for the operator's convenience: `keys`
(entries the agent's **shape-only** survey believed currently valid) and
`broken_keys` (entries it believed already broken). These are **display and
provenance only**, not a contract:

- The shape-only survey cannot determine HMAC validity (it has no key), so an
  entry that is shape-valid but HMAC-invalid under the old key can be mislabeled
  into `keys`.
- At fire time the `rekey` CLI **re-derives the full judge-gated set from the
  live tree** and Pass-1 (`verify_entry_signature_with_key`) is the authoritative
  HMAC gate. A tree entry absent from `keys` is still re-keyed; a `keys` member
  that fails Pass-1 still aborts the run.

Read the lists as a preview of scope, never as the set that will actually be
acted on.

## Staleness: when the operator must re-run `stage_scan`

A bundle is a point-in-time assertion about the tree. The verify gate aborts
(and the operator simply re-stages) in two expected situations:

- **AST-position cascade staleness (by design).** A bundle staged *before* an
  edit that shifts AST positions in a covered `src/elspeth` source — for example
  adding an `import` — no longer matches the tree. `verify_bundle_against_tree`
  aborts; re-run `stage_scan` against the current tree and fire the fresh bundle.
- **Dup-key in the target file.** If the same allow-hits key occurs more than
  once in a file, `sign-bundle` refuses (`return 2`, both copies preserved).
  Resolve the duplicate in the YAML, re-run `stage_scan`, and fire again.

In both cases the safe action is the same: re-stage, do not force.

## What replaced the one-off signing scripts

This seam replaces the per-release signing runbooks and the one-shot
`scripts/cicd/sign_accept_backlog.py` driver. Instead of hand-edited ceremony
scripts, an agent stages a bundle via `elspeth-judge` and the operator fires
`sign-bundle` / `rekey` in a key-bearing shell. `scripts/codex_tier_model_rejudge.py`
is retained (it belongs to a separate active workflow); the `notes/060-*` signing
runbooks were gitignored scratch and have been removed.
