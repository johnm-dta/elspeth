---
name: cicd-allowlist-audit
description: >
  End-to-end audit of all CI/CD enforcement gates and their allowlists — separates
  load-bearing exemptions (Tier 3 boundaries, Tier 1 offensive guards, deliberate
  I/O fault-isolation) from fixable debt (stale FPs, deferred refactors, rationalized
  defensive code, ticket-tagged temporary exemptions left permanent). Use periodically
  (every 4–8 weeks) or whenever the allowlist has grown noticeably between
  measurements. Don't trust author justifications — dispatch independent SME agents
  to verify them against actual code.
---

# CI/CD Allowlist Audit

ELSPETH ships ~14 CI/CD enforcement gates plus mypy/ruff. Most have allowlists for
deliberate exceptions. Without periodic audit those allowlists grow monotonically:
new code lands new exemptions, but nobody re-questions whether each exemption is
still load-bearing. This skill runs the audit.

> **CRITICAL:** Author justifications in the allowlist are self-attestations that
> have never been independently reviewed. They tend to crystallize as load-bearing
> because nobody re-questions them. The audit's value comes from **independent SME
> agent review of the code** against the claimed justification, not from reading
> the justification.

## When to run

- Routine: every 4–8 weeks (set a calendar entry, or fire from a periodic skill).
- Triggered: when `enforce_tier_model` allowlist crosses a +25% growth threshold
  vs prior baseline.
- Triggered: when expiry-bearing entries approach their expiry window.
- Triggered: before a major release, to size and triage allowlist debt.
- Triggered: when the override-rate gate (C3, below) fails — reaudit the
  override-heavy directories before considering an ADR to relax the threshold.

## The cicd-judge gate (allowlist write path)

As of the cicd-judge merge (RC5.2, 2026-05-30 — merge `fd4830b42`), new
`enforce_tier_model` allowlist entries are no longer hand-written. A proposed
suppression passes through an Opus **judge** that decides whether the
defensive-pattern finding is an *honest* trust boundary (legitimately exempt)
or *debt* (defensive code that should be fixed). The judge's verdict, rationale,
the source binding (AST path + a fingerprint), and an HMAC signature are
written into the entry. The forward binding primitive is the v2
`scope_fingerprint` (the enclosing-scope AST fingerprint, signature prefix
`hmac-sha256:v2:`) minted by `justify`; the legacy v1 `file_fingerprint`
(whole-file hash, prefix `hmac-sha256:v1:`) is still live for already-signed
entries and is being migrated away via `migrate-judge-scope`.

This reframes what the audit audits:

- **Pre-judge entries** (no `judge_recorded_at`) are the old self-attestations —
  the "don't trust the reason, read the code" rule above applies in full.
- **Judge-era entries** carry an independent verdict, but a verdict can **decay**
  as the underlying code changes out from under it. The audit's job for these is
  decay detection (`reaudit`), not re-justification.

All judge tooling is `elspeth-lints` subcommands. Run from the repo root with:

```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli <cmd> ...
```

`<cmd> --help` is the authoritative flag/verdict reference; the tables below are
orientation, not a substitute (verdict enums and flags evolve).

### Subcommands

| Subcommand | Role | Writes to allowlist? |
| ---------- | ---- | -------------------- |
| `rotate` | Mechanical fingerprint reconciliation after an AST refactor. **No judge.** | Yes — entry fingerprints |
| `justify` | Propose a new entry; the judge (Opus) returns `ACCEPTED`/`BLOCKED`; signs accepted entries. Accepts `--judge-transport {openrouter,agent}` (default `openrouter`). | Yes — new entry |
| `diagnose-judge-signatures` | Read-only stale-signature triage for signed allowlist entries. Use before asking an operator to sign or before committing allowlist churn. | No |
| `sign-judge-signatures` | Operator-only bulk repair for signable signed-entry drift. Refuses entries that cannot safely be signed, such as `NO_MATCHING_FINDING`. | Yes — re-signed entries |
| `audit-verdict` | Human post-review of a judge-**ACCEPTED** entry — confirm or reverse the judge. | Yes — review block |
| `reaudit` | Re-run the judge across existing entries to detect decay. **Read-only on YAML**; emits a triage report. Accepts `--judge-transport {openrouter,agent}` (default `openrouter`). | No |
| `migrate-judge-scope` | **Operator-only** (signs; needs the HMAC key). Re-signs v1 (`file_fingerprint`) entries whose signature verifies and whose node still matches a live finding as v2 (`scope_fingerprint`) **without re-running the judge** — it deliberately skips the file_fingerprint byte-freshness gate, so byte-drifted-but-scope-stable entries are the target set, not "CI-green" ones. Gated on integrity (existing v1 signature must verify) + relevance (key still matches a live finding). | Yes — re-signed entries |
| `check-judge-coverage` (C1) | CI gate: every new entry must carry signed judge metadata (pre-judge entries grandfathered, rotation-stable). | No |
| `check-override-rate` (C3) | CI gate: rolling-30d operator-override rate must stay under `--max-rate` (workflow-pinned 0.10). | No |
| `check-judge-quality` (VAL) | CI gate, trusted contexts only: judge accuracy on a labelled corpus ≥ 0.90 (regression tripwire, not a guarantee). | No |
| `check-rotation-audit` | CI gate: fingerprint rotations match the `.elspeth/rotations.log` manifest. | No |

### HMAC key custody — the load-bearing rule

The signature uses a **symmetric** key (`ELSPETH_JUDGE_METADATA_HMAC_KEY`). Any
key holder can forge a valid `ACCEPTED` verdict, so the gate's whole security is
custody: **the key is operator-only and MUST NOT be in any agent environment.**
An agent may *propose* a `justify` invocation; only an operator-held environment
runs it and signs. Full rationale: CLAUDE.md § "CICD Judge Gate: HMAC Key
Custody". During an audit you therefore *recommend* `justify`/`audit-verdict`
commands for the operator to run — you do not run the signing path yourself.

### Signed-entry repair workflow

Before committing allowlist churn, run read-only diagnosis with fork-compatible
signature verification:

```bash
ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing \
  uv run elspeth-lints diagnose-judge-signatures \
  --root src/elspeth \
  --allowlist-dir config/cicd/enforce_tier_model \
  --format text
```

Classify the output before touching YAML:

- `NO_MATCHING_FINDING`: the signed row has no live finding. The signer will
  refuse this. Inspect the code; if the finding is gone, remove that stale row
  from the allowlist. If the finding still matters under a new key, propose a
  fresh `justify` command for the operator.
- `SCOPE_BINDING_DRIFT`: the same finding key still exists, but the enclosing
  scope changed. Do not hand-edit metadata. Propose the emitted `justify`
  command; the operator runs it with the HMAC key.
- `AST_PATH_BINDING_DRIFT` with `repair_key`: the live finding moved to a new
  fingerprint. If an operator-signed replacement row for the `repair_key`
  already exists and diagnosis reports it `OK_SHAPE_ONLY`/valid, remove the old
  stale row. If no replacement exists, propose the emitted `justify` command and
  wait for the operator.

After operator signing, re-run `diagnose-judge-signatures`; only continue when
no drift/no-match rows remain. Then run the actual trust-tier gate:

```bash
ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing \
  uv run elspeth-lints check --rules trust_tier.tier_model --root src/elspeth
```

Do not commit backup files created during operator repair, such as temporary
`*.bak-*` copies, unless the user explicitly asks for them.

### `--judge-transport` — which LLM serves the verdict

`justify` and `reaudit` both accept `--judge-transport {openrouter,agent}`
(default `openrouter`; no behaviour change unless opted in). This selects the
judging LLM and is a *separate axis* from HMAC custody — it governs which model
produces the verdict, not who signs it.

- **`openrouter`** (default) — OpenAI-compatible SDK pointed at OpenRouter,
  `temperature=0`, reproducible. Persisted/signed `judge_transport`: `"openrouter"`.
  Use this for any deterministic re-check (and it is the only transport an agent
  can drive, since it runs on the agent's own OpenRouter key).
- **`agent`** — Claude Agent SDK (`claude_code` system-prompt preset, no tools).
  Needs the `[judge-agent]` extra and Claude Code auth (CLI login /
  `ANTHROPIC_API_KEY` / Bedrock-Vertex-Azure) — operator-held credentials an
  agent's environment does not carry. Persisted/signed `judge_transport`:
  `"claude_agent_sdk"`. The "cheaper" assumption holds only on the
  subscription/credit path; `ANTHROPIC_API_KEY` is per-token and may not beat
  OpenRouter. The SDK cannot pin `temperature`, so agent-written verdicts are
  less reproducible than the `openrouter` path.

`judge_transport` is part of the **signed v2 payload** ("how the verdict was
produced" — verdict metadata, tamper-evident with the verdict; a forged or edited
transport label fails the load-time HMAC recompute). Consequence for the audit: a
re-`justify` of the same entry under a different transport surfaces as a
metadata-diff (the `judge_transport` value, and the verdict/rationale it
produced, change). For decay-detection re-checks, keep `reaudit` on
`--judge-transport openrouter` so re-runs stay deterministic regardless of the
transport that originally wrote each entry.

### `justify` — propose a new entry (agent proposes, operator signs)

Re-runs the rule on `--file-path`, asks the judge to rule on the `--rationale`,
and on `ACCEPTED` writes the signed entry. The two model verdicts are `ACCEPTED`
(the suppression lands) and `BLOCKED` (fix the code instead — the judge holds a
conservative prior toward `BLOCKED`); `OVERRIDDEN_BY_OPERATOR` is a third verdict
set only by `--operator-override`, never by the model. Required: `--file-path`, `--symbol`
(qualified name; literal `_module_` for module scope), `--rationale`, `--owner`.
Useful: `--fingerprint` (disambiguate when `--symbol` matches several findings),
`--dry-run` (show the verdict without writing), `--rule` (default
`trust_tier.tier_model`). `--operator-override` writes
`verdict=OVERRIDDEN_BY_OPERATOR` (the judge is still called for the record) and
requires the override-token env vars — every override feeds the C3 rate gate.

### `reaudit` — decay sweeps (the audit's main judge-era tool)

Re-judges existing entries and reports which verdicts no longer hold. **This is
the read-only sweep to run as part of an audit** (it never edits YAML). Key
flags: `--limit N` / `--max-calls N` (the ~700-entry corpus must NOT be
re-judged in one pass — `--max-calls` is a spend guard that leaves the sweep
resumable), `--since` (skip entries judged recently), `--include-pre-judge` (off
by default; the pre-judge set is large), `--format markdown --output <path>`
(most triage-readable). Crash recovery: a fresh run prints a `run_id`; resume a
killed sweep with `--resume <run_id>`, or render its partial report with
`--render-incomplete <run_id>` (sidecar at
`<allowlist-dir>/.reaudit-state/<run_id>.jsonl`). Feed decayed verdicts into
Stage 5 as fixable subtickets.

### `audit-verdict` — reverse a bad ACCEPT

When a `reaudit` (or SME review) shows the judge wrongly accepted an entry, an
operator attaches a post-review verdict with `--key <exact allow_hits key>`,
`--verdict <choice>` (see `audit-verdict --help` for the enum), `--reviewer`,
and `--rationale`. This records the human reversal in the audit trail without
silently deleting the entry.

### The three CI gates (branch protection)

The gates live in `.github/workflows/enforce-allowlist-judge-gates.yaml`; the
single required status check is the aggregate `judge-gates-success` job (C1 + C3
must pass; VAL may be `skipped` on fork PRs, which cannot read the OpenRouter
secret). On fork PRs the HMAC key is also withheld, so signature verification
falls back to `shape-only-when-key-missing` mode
(`ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE`). A C3 failure is an audit
trigger (see "When to run"); a C1 failure means someone hand-wrote an entry
without going through `justify`.

## Method (5 stages)

### Stage 1 — Verify all gates green

All 14 gates must currently pass locally; otherwise the audit is meaningless
(you can't separate load-bearing from fixable if some are actively failing).

```bash
cd /home/john/elspeth
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth \
  --allowlist config/cicd/enforce_tier_model --exclude "**/__pycache__/*" --format text
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth \
  --allowlist config/cicd/enforce_freeze_guards
.venv/bin/python scripts/cicd/enforce_audit_evidence_nominal.py check --root src/elspeth \
  --allowlist config/cicd/enforce_audit_evidence_nominal
.venv/bin/python scripts/cicd/enforce_tier_1_decoration.py check \
  --file src/elspeth/contracts/errors.py --allowlist config/cicd/enforce_tier_1_decoration
.venv/bin/python scripts/cicd/enforce_composer_exception_channel.py check \
  --root src/elspeth --allowlist config/cicd/enforce_composer_exception_channel
.venv/bin/python scripts/cicd/enforce_composer_catch_order.py check --root src/elspeth \
  --allowlist config/cicd/enforce_composer_catch_order
.venv/bin/python scripts/cicd/enforce_contract_manifest.py check \
  --allowlist config/cicd/enforce_contract_manifest
.venv/bin/python scripts/cicd/enforce_frozen_annotations.py check --root src/elspeth \
  --allowlist config/cicd/enforce_frozen_annotations
.venv/bin/python -m scripts.cicd.enforce_plugin_hashes check --root src/elspeth
.venv/bin/python scripts/cicd/enforce_options_metadata.py
.venv/bin/python scripts/cicd/enforce_component_type.py check --root src/elspeth \
  --allowlist config/cicd/enforce_component_type
.venv/bin/python scripts/cicd/enforce_guard_symmetry.py check --root src/elspeth \
  --allowlist config/cicd/enforce_guard_symmetry
.venv/bin/python scripts/cicd/enforce_gve_attribution.py check --root src/elspeth \
  --allowlist config/cicd/enforce_gve_attribution
.venv/bin/python -m scripts.check_contracts
.venv/bin/python scripts/cicd/check_slot_type_cross_language.py
.venv/bin/python -m mypy src/elspeth
.venv/bin/python -m ruff check src/ tests/ scripts/ examples/
.venv/bin/python -m ruff format --check src/ tests/ scripts/ examples/
```

If any gate fails, **stop** and resolve that first before continuing the audit.

### Stage 2 — Aggregate allowlist corpus

Run the aggregation script to count entries per gate, per rule, per owner, per
expiry status. Use this Python (drop into a Bash heredoc):

```python
import yaml, glob, collections, datetime
from pathlib import Path

today = datetime.date.today()

def scan_dir(dirpath):
    out = {"allow_hits": 0, "per_file_rules": 0,
           "by_rule": collections.Counter(), "owners": collections.Counter(),
           "expired": [], "near_expiry": [],
           "expiry_status": collections.Counter()}
    for f in sorted(glob.glob(f"{dirpath}/*.yaml")):
        name = Path(f).name
        if name.startswith("_") or name == "__init__.yaml":
            continue
        d = yaml.safe_load(open(f)) or {}
        for e in (d.get("allow_hits", []) or []):
            out["allow_hits"] += 1
            parts = e.get("key", "").split(":")
            if len(parts) >= 2:
                out["by_rule"][parts[1]] += 1
            out["owners"][e.get("owner") or "(none)"] += 1
            _classify_expiry(e.get("expires"), name, e.get("key", "?"), out, today)
        for e in (d.get("per_file_rules", []) or []):
            out["per_file_rules"] += 1
            for r in (e.get("rules") or []):
                out["by_rule"][r] += 1
            _classify_expiry(e.get("expires"), name, e.get("pattern", "?"), out, today)
    return out

def _classify_expiry(exp, name, key, out, today):
    if exp is None:
        out["expiry_status"]["permanent"] += 1
        return
    try:
        d = datetime.date.fromisoformat(str(exp))
        if d < today:
            out["expired"].append((name, key, str(d)))
            out["expiry_status"]["EXPIRED"] += 1
        elif (d - today).days < 30:
            out["near_expiry"].append((name, key, str(d)))
            out["expiry_status"]["<30d"] += 1
        else:
            out["expiry_status"][">=30d"] += 1
    except ValueError:
        out["expiry_status"]["unparseable"] += 1

GATES = [
    "enforce_tier_model", "enforce_freeze_guards", "enforce_audit_evidence_nominal",
    "enforce_tier_1_decoration", "enforce_component_type", "enforce_guard_symmetry",
    "enforce_gve_attribution", "enforce_composer_catch_order",
    "enforce_composer_exception_channel", "enforce_contract_manifest",
    "enforce_options_metadata", "enforce_frozen_annotations",
]
for g in GATES:
    r = scan_dir(f"config/cicd/{g}")
    print(f"{g}: ah={r['allow_hits']} pfr={r['per_file_rules']} "
          f"expired={len(r['expired'])} near={len(r['near_expiry'])} "
          f"rules={dict(r['by_rule'])} owners={dict(r['owners'])}")
```

Record the totals for each gate and compare to the prior audit baseline (see
`docs/audit/`). Calculate the growth rate. If `enforce_tier_model` allow_hits
have grown > 25%, that's a structural finding on its own.

### Stage 3 — Identify suspect categories

For `enforce_tier_model` (always the dominant gate), categorize:

**Always load-bearing (do not review):**
- Owners starting with `web-` AND rule R5 — Tier 3 HTTP/YAML boundary validators
- Owner `architecture` AND rule R5 AND path under `contracts/*` AND function name
  in `__post_init__` — Tier 1 offensive guards (institutional pattern)
- Owner `infrastructure` AND rule R1 AND `os.environ.get` in the code — env var
  reads at the Tier 3 boundary (the OS is external)
- Owner `composer-audit`, `composer-guided`, `composer-walker`,
  `composer-cache-markers` — LLM tool-call protocol boundary (Tier 3)
- Owners `web-secrets`, `web-blobs`, `web-catalog`, `web-middleware`,
  `web-validation` — Tier 3 data ingestion
- L1 entries with a tracked refactor follow-up in filigree

**Always suspect (queue for SME review):**
- Owner is empty / `-` / `(none)` — missing attribution
- Owner contains a filigree ticket ID (e.g. `P2-2026-02-02-76r`) — temporary
  exemption that may have outlived its ticket
- Owner is `bugfix` — generic spot-fix label, often masks deferred refactor
- Owner is `feature` — review whether the pattern is repeated across many
  call sites (refactor opportunity)
- Owner is `refactor` — single-entry "same pattern as X" justifications cargo-cult
- Reason contains the literal text "false positive" — should fix the rule, not exempt
- Reason describes a fingerprint-rotation event ("FP rotated during merge…") rather
  than the actual justification — original justification was lost
- Reason is shorter than 40 characters — insufficient context
- `expires: null` AND owner is `bugfix`/`feature`/`refactor` — temporary owners
  with permanent waivers

### Stage 4 — Dispatch independent SME agents

For each suspect category, dispatch an SME agent **with anchor points already
located**. Don't ask the agent to guess where the code lives.

Agent matrix (run in parallel):

| Category                                         | Agent                                                 |
| ------------------------------------------------ | ----------------------------------------------------- |
| Rule false-positive candidates (e.g. httpx FP)   | `axiom-static-analysis-engineering:false-positive-analyst` |
| R4/R6 broad-catch / silent-swallow suppressions  | `pr-review-toolkit:silent-failure-hunter`             |
| Repeated patterns (refactor candidates)          | `axiom-python-engineering:refactoring-architect`      |
| `bugfix`/`architecture` justification spot-check | `axiom-python-engineering:python-code-reviewer`       |
| Trust-boundary classification questions          | `tier-model-deep-dive` skill + manual                 |

Agent prompt template (the agent must read the actual code, not the reason):

```
You are reviewing CI suppression justifications in ELSPETH.

CONTEXT: Each suppressed finding has a `reason:` field written by the author.
You verify whether that reason holds up by reading the code at the cited
location. Don't trust the reason at face value.

For each entry below:
1. Read the cited file at the cited function.
2. Find the specific line the rule flagged.
3. Apply CLAUDE.md's trust-boundary decision test.
4. Verdict: HOLDS / MISCHARACTERIZED / FIXABLE / STALE.

[List entries with key, owner, reason, file, function.]

DELIVERABLE: write to /home/john/elspeth/docs/audit/findings/<agent-role>.md
with per-entry verdict + aggregate signal + recommendation.

CWD: /home/john/elspeth. Use .venv/bin/python.
```

Always **write findings to a durable file under `docs/audit/findings/`** so the
session can be interrupted without losing work.

### Stage 5 — Synthesize, ticket, baseline

From the agent reports, produce:

1. **Audit summary doc** at `docs/audit/YYYY-MM-DD-cicd-allowlist-audit.md`
   with the inventory, growth rate, and per-category verdict.
2. **Filigree subtickets** under parent `elspeth-297b8f5c5d` ("CI allowlist
   revalidation") — one per confirmed-fixable category, citing the specific
   allowlist keys and the SME verdict.
   - Don't duplicate the existing 41 children — those are about *enforcer-script
     bugs* (false negatives, missing checks). This audit produces tickets
     about *exemption-corpus debt* (entries the enforcer correctly flagged
     but were exempted on weak grounds).
3. **Process tickets** for structural findings:
   - If growth rate > 25% in the audit window: ticket for a ratchet
     (per-PR allowlist-delta budget, or required burn-down lane).
   - If a rule has > 30% of all suppressions: ticket for rule split / re-tier.
   - If near-expiry entries exist: ticket for triage before expiry.
4. **Update the parent epic baseline** with the new total + date.

## Output contract

The audit produces exactly these artifacts:

- `docs/audit/YYYY-MM-DD-cicd-allowlist-audit.md` (the summary)
- `docs/audit/findings/<agent-role>.md` per dispatched SME agent
- Filigree subtickets under `elspeth-297b8f5c5d` (only for *new* fixable findings)
- Updated baseline note on the parent epic

## What this skill does NOT do

- Doesn't fix anything — only audits. Each fixable subticket goes back through
  normal claim/work/close flow with proper review.
- Doesn't modify enforcer scripts. Rule-split or rule-fix proposals become
  subtickets, not edits.
- Doesn't trust author reasons. Every suspect-category entry requires
  independent code review.
- Doesn't try to enumerate every entry. ~600+ entries is too many for a single
  audit pass — sample within each category, validate the *pattern*, and
  ticket the pattern.

## Anti-patterns observed in past audits

1. **"Same pattern as X" cascade** — one entry justifies itself by citing
   another, but X itself was never justified independently. Each entry must
   stand on its own.
2. **`expires: null` everywhere** — the lifecycle mechanism exists but isn't
   used. Push back: every `bugfix`/`feature`/`refactor`-owned entry should
   carry an expiry.
3. **Fingerprint-rotation justifications** — the reason describes when the
   fingerprint changed, not why the underlying code is exempt. These have lost
   their original justification.
4. **Ticket-tagged owners surviving ticket closure** — `P2-2026-…` owners
   should be retired when their ticket closes; periodically check that the
   referenced tickets are still open.
5. **Owner `-` or missing** — every entry must have an owner. Period.

## Cross-references

- `tier-model-deep-dive` skill — for the trust-tier decision test the SME
  agents apply.
- `CLAUDE.md` — for the "offensive programming encouraged, defensive forbidden"
  policy, the fabrication decision test, the `rotate`/`justify`/`reaudit`
  command examples, and § "CICD Judge Gate: HMAC Key Custody".
- The cicd-judge gate (write path) — see "The cicd-judge gate" section above;
  CLI in `elspeth-lints/src/elspeth_lints/core/cli.py`; CI in
  `.github/workflows/enforce-allowlist-judge-gates.yaml`; design notes in
  `notes/cicd-judge-cli-prototype-plan.md`.
- Parent filigree epic `elspeth-297b8f5c5d` — for the standing
  "CI allowlist revalidation" track.
- Prior audit: `docs/audit/2026-05-19-cicd-allowlist-audit.md` (the first
  application of this skill).
