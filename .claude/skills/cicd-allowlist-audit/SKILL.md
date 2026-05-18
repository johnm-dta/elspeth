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
  policy and the fabrication decision test.
- Parent filigree epic `elspeth-297b8f5c5d` — for the standing
  "CI allowlist revalidation" track.
- Prior audit: `docs/audit/2026-05-19-cicd-allowlist-audit.md` (the first
  application of this skill).
