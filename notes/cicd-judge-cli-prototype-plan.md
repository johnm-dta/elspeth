# cicd-judge-cli prototype — feature map

**Status:** prototype + convergent-findings remediation landed. Slice 1 (rotate), Slice 2 (judge gate), Slice 3 (reaudit decay-sweep), and the structural CI surfaces (judge-coverage on PRs, override-rate threshold, judge-quality corpus, multi-rule reaudit dispatch) ship on `feat/cicd-judge-cli-prototype`. Slice 4 (`@trust_boundary` decorator) ships alongside Slices 1–3 (3 honesty-gate rules plus the L0 decorator).

## Delivered vs Enforced

What ships as code (operator-runnable surfaces) and what fails CI now are not the same set. The table below distinguishes them so a reader does not infer mechanical enforcement from feature presence.

| Surface | Delivered (runnable) | Enforced (CI fails on violation) |
| --- | --- | --- |
| Slice 1 — `elspeth-lints rotate` | ✓ | n/a (rotation is mechanical, not a gate) |
| Slice 2 — `elspeth-lints justify` (judge gate at write time) | ✓ | n/a directly — the judge writes recorded metadata; the metadata is what CI checks |
| Slice 3 — `elspeth-lints reaudit` (decay sweep) | ✓ across all 17 supported rule packages | n/a (operator-paced sweep, not CI-blocking) |
| Slice 4 — `@trust_boundary` decorator + 3 honesty-gate rules | ✓ | ✓ via `trust_boundary.tests`, `trust_boundary.scope`, `trust_boundary.tier` |
| **C1 — Judge-coverage gate** (new-entry quartet required) | ✓ via `elspeth-lints check-judge-coverage` | ✓ via `.github/workflows/enforce-allowlist-judge-gates.yaml` (PR-only; rotation-grandfathered) |
| **C2 — Multi-rule reaudit dispatch** (no more single-rule artificial restriction) | ✓ via expanded `_supported_rules()` (17 of 19 BUILTIN_RULES) | n/a — same enforcement surface as Slice 3 (reaudit is sweep, not gate) |
| **C3 — Override-rate gate** (rolling-30d threshold) | ✓ via `elspeth-lints check-override-rate` | ✓ via the same workflow file (push + PR; insufficient-data passes with notice) |
| **VAL — Judge-quality corpus** (labelled discrimination cases) | ✓ via `elspeth-lints check-judge-quality` + `config/cicd/judge-quality-corpus/v1.jsonl` | ✓ via the same workflow file in trusted contexts with `OPENROUTER_API_KEY` |

Sequencing note for the convergent findings: C1 + C3 give judge enforcement a mechanical anchor — Pillar A no longer relies only on social norms or self-discipline for new-entry metadata and override-rate policy. VAL gives prompt/model quality a measured anchor rather than relying on parser tests. C2 closes the friction-displacement loop by extending reaudit's reach to all rule packages, so suppression debt cannot quietly route to ungated rules.

This document maps the two pillars of the audit-trail-integrity thesis driving the prototype:

- **Pillar A — Judge:** raise the *quality* of every new allowlist entry by forcing LLM-justified writes through an Opus judge that blocks but does not fix.
- **Pillar B — `@trust_boundary` decorator:** reduce the *volume* of allowlist entries by moving Tier-3 boundary suppressions out of YAML and into the code they describe.

The two are complementary, not alternatives. Judge guards what gets written; decorator changes what *needs* to be written. Decorator can ship without judge (smaller corpus, same self-attestation problem). Judge can ship without decorator (better quality, same corpus size). Together they collapse the residual allowlist toward genuinely one-off exemptions, each carrying a judge-recorded justification.

---

## Problem the prototype attacks

The current `enforce_tier_model` allowlist has roughly the following pathology (numbers as of 2026-05-22 production tree):

| Signal | Value | Interpretation |
| --- | --- | --- |
| Total allowlist entries | ~700 | Suppression set has grown beyond per-entry human review |
| Entries with `owner: TODO` | 112 | Old rotator silently created stubs; nobody re-reviewed |
| Entries duplicating identical rationale across files | ~117 (web/composer/tools/*) | Same Tier-3 boundary pattern, fragmented into per-line YAML |
| Fingerprint rotations per AST shift | proportional to imports added above | Refactor pain disincentivises imports-as-cleanup; allowlist becomes a refactoring tax |
| Self-attested justifications never re-reviewed | ~all | YAML write is the only review surface; PR review tends to skim allowlist diffs |

The 32-day +51% growth observed in the 2026-05-19 allowlist audit (memory: `project_cicd_allowlist_audit_2026-05-19.md`) is the trajectory consequence of this pathology. The lifecycle policy is mechanically bounded but the practice of justifying entries is not.

The prototype's hypothesis: putting an LLM judge inline with the write, and refactoring the most common write *out* of YAML, fixes both growth and quality at once.

---

## Pillar A — Judge (Slice 2 of prototype)

Single CLI surface (`elspeth-lints justify`) that becomes the mandatory path for adding new allowlist entries. Agent describes what changed and why; Opus judge reads the change + the surrounding code and decides whether the justification holds. Verdicts: `ACCEPTED`, `BLOCKED`, `OVERRIDDEN_BY_OPERATOR`.

### Roles (load-bearing)

- **Agent** writes the proposed entry, supplies a rationale, names the trust boundary or pattern.
- **Judge (Opus)** evaluates: does the rationale match the code? Is the suppressed rule legitimately a boundary concern at this site? Is this entry a duplicate of one a per-file rule should cover instead?
- **Judge does NOT write code fixes.** A BLOCKED verdict returns the failure mode to the agent; the agent figures out the remediation (refactor, broaden a per-file rule, move under a decorator, abandon the suppression). This is the operator-stated constraint: "the judge doesn't have to fix the problem, it just blocks the change to the whitelist and tells the agent to figure it out." The judge MAY name a structural alternative — concretely the `@trust_boundary` decorator — when a finding fits the decorator's preconditions; this is naming, not fixing.
- **Operator** can issue `OVERRIDDEN_BY_OPERATOR` — distinct from `ACCEPTED`. This is the audit signal that a human used their authority to bypass the judge; rotation rate of this verdict is itself a metric to watch (operator memory: "or it tells us that the operator is breaking his own rules to get something out the door which is a different signal but still worth catching").

### Output the judge produces

Every verdict gets recorded as a structured row, not a freeform log:

```yaml
- key: <path>:<rule>:<symbol>:fp=<hex>
  owner: <agent or human>
  reason: <freeform rationale supplied at write time>
  judge_verdict: ACCEPTED | OVERRIDDEN_BY_OPERATOR
  judge_recorded_at: 2026-05-23T...
  judge_model: claude-opus-4-7
  judge_rationale: <the model's reasoning, recorded verbatim>
  file_fingerprint: <sha256 of source file bytes judged>
  ast_path: <AST address of finding judged>
  judge_metadata_signature: hmac-sha256:v1:<hex>
  expires: <existing field>
  safety: <existing field>
```

The judge's rationale is new audit evidence attached to the allowlist entry, not a standalone proof of truth. It is independently re-readable in 6 months when someone asks "why did we exempt this?" — the YAML answers without re-running the model, and the binding fields let the loader verify that evidence has not been silently edited.

### Judge metadata tamper binding

The verdict quartet (`judge_verdict`, `judge_recorded_at`, `judge_model`, `judge_rationale`) is audit-significant evidence, not decorative metadata. Post-judge entries therefore carry two binding layers:

1. `file_fingerprint` + `ast_path` bind the verdict to the source bytes and AST node the judge inspected.
2. `judge_metadata_signature` is `hmac-sha256:v1:<hex>` over the entry key, `file_fingerprint`, `ast_path`, verdict, model verdict (if any), recorded timestamp, model id, and rationale.

The HMAC key is supplied by `ELSPETH_JUDGE_METADATA_HMAC_KEY`, must be held outside the allowlist YAML, and must be at least 32 UTF-8 bytes. `elspeth-lints justify` refuses to write signed judge metadata without it. Production source-root loads (`load_allowlist(..., source_root=...)`, used by the tier-model gate and reaudit) verify the signature and reject missing or mismatched signatures; report-only loaders without a source root can still inspect historical entries but do not claim tamper verification.

### Decay sweep (Slice 3 of prototype)

`elspeth-lints reaudit` re-runs the judge across existing entries. Used at allowlist-renewal time (when an `expires:` is bumped). Decisions to keep an entry must survive a fresh judge pass. This closes the loop on the "self-attestation never re-reviewed" failure mode.

**Rule coverage (post-C2):** the dispatch now supports every rule in `BUILTIN_RULES` except two explicit exclusions — `audit_evidence.nominal_base` (uses the legacy `allow_classes:` shape, not `allow_hits:`) and `meta.no-new-bespoke-cicd-enforcer` (project-policy gate, not a code-pattern lint). 17 of 19 rules are reaudit-targetable. Adding a new rule to `BUILTIN_RULES` makes it reaudit-targetable automatically (modulo registering its sub-rule vocabulary in `_valid_rule_ids_for`).

### Override-rate gate (convergent finding C3)

`elspeth-lints check-override-rate` measures the rolling-window ratio of `OVERRIDDEN_BY_OPERATOR` verdicts against the population of judged entries. Window default 30 days, threshold default 10%, minimum-samples default 10 (small-N windows pass with an "insufficient data" notice — naive rate gates trip mechanically on a 3-entry corpus with one override). Wired into `enforce-allowlist-judge-gates.yaml` on push + PR. The threshold is policy: changes land via ADR, not by editing the workflow file.

### C3 response protocol

When the override-rate gate fails, the operator response is not to raise the threshold first. Triage the listed override entries by rule and directory, run `elspeth-lints reaudit` against the override-heavy directories, and handle entries whose fresh result says the suppression is no longer justified by refactoring or deleting the allowlist entry. If the overrides remain legitimate after reaudit and review, record the policy decision in an ADR before changing `--max-rate`; the workflow threshold is an audit policy knob, not a quota agents can tune to pass CI. If the gate passes because of insufficient data, treat the summary as informational and do not infer calibration quality until the denominator reaches `--min-samples`.

### Judge-quality corpus (VAL)

`config/cicd/judge-quality-corpus/v1.jsonl` contains a bounded labelled corpus of allowlist-rationale examples. Each case becomes one real `JudgeRequest`; `check-judge-quality` scores exact matches on both `verdict` and `should_use_decorator`. The trusted CI job runs the corpus through the live OpenRouter-backed judge with `--min-accuracy 0.90`, so prompt edits and model changes have a discrimination gate, not just parser/schema tests.

Re-baseline cadence:

- Run the live corpus before and after judge prompt, policy-context, or model changes.
- Add cases when reviews discover new judge failure modes; keep the corpus at 10-30 cases so CI spend stays bounded.
- Change expected labels only when the underlying policy or intentionally reviewed decision boundary changes. Do not lower the workflow threshold as a workaround for drift; threshold changes are policy changes.

### Out-of-scope for the prototype

- No web UI; CLI only.
- No model fallback; Opus is named, fixed, single-vendor (the audit trail records the model used).
- No async batching of judge calls; one entry, one synchronous call. Performance is irrelevant for the volume (low tens of new entries per day at peak).

---

## Pillar B — `@trust_boundary` decorator (Slice 4 of prototype)

**Sequencing:** the decorator contract and companion honesty gates ship in this prototype. Bulk migration of existing allowlist entries ships later, after the Cluster 3 (web/composer/tools) reconciliation lands and the residual allowlist shape is known. The migration's value depends on the final corpus shape — how many of the remaining entries cluster around function-scoped Tier-3 boundaries vs. how many are genuinely orphaned. Without that measurement, the migration scope is speculation.

### Shape

```python
# src/elspeth/contracts/trust_boundary.py
@trust_boundary(
    tier=3,
    source="LLM tool-call arguments emitted by composer model",
    source_param="arguments",
    suppresses=("R1", "R5"),
    invariant="raises ToolArgumentError on shape mismatch; never coerces silently",
    test_ref="tests/unit/web/composer/tools/test_sessions.py::test_set_pipeline_rejects_malformed_nodes",
    test_fingerprint="<canonical AST fingerprint of that pytest node>",
)
def _execute_set_pipeline(self, arguments: dict[str, Any]) -> ToolResult:
    if not isinstance(arguments.get("nodes"), list):  # ← R5 suppressed
        raise ToolArgumentError("nodes must be a list")
    for raw in arguments["nodes"]:                    # ← R1 suppressed only on `arguments`
        node_id = raw.get("id")                       # ← still suppressed: came from `arguments`
        ...
    self._cache.get(node_id)                          # ← NOT suppressed: not `arguments`-rooted
```

The `source_param` field anchors a small dataflow walk inside `tier_model.rule`: names derived from `arguments` (via subscript, `.get(...)`, iteration, unpacking, or assignment from any of the above) are in the suppression set; everything else is not. This keeps the decorator from becoming a whole-function exemption cloak.

### Companion CI gates (load-bearing, not optional)

| Gate | What it asserts | Why it matters |
| --- | --- | --- |
| `enforce_trust_boundary_tests.py` | Every `@trust_boundary` has a `test_ref` pointing to a real pytest node that directly calls the decorated function through `source_param`, raises the documented exception type, and matches `test_fingerprint` | Stops the decorator from becoming a stale or irrelevant vibe-justification; behavioural evidence required |
| `enforce_trust_boundary_scope.py` | `source_param` is in the function signature; function body actually reads from it | Stops drive-by `@trust_boundary` on functions that don't use the declared boundary parameter |
| `enforce_trust_boundary_tier.py` | `tier=3` is the only accepted value | Tier-1 and Tier-2 must crash, not suppress — the decorator is not a general-purpose lint silencer |

### Honesty-gate false-positive escape protocol

The companion gates are fail-closed, but not dead-end. A demonstrated false
positive can be suppressed only through the shared
`config/cicd/enforce_trust_boundary_honesty/` allowlist. That directory accepts
exact `allow_hits` entries only: every entry must carry judge metadata,
`file_fingerprint`, `ast_path`, `judge_metadata_signature`, and an expiry.
`per_file_rules` are rejected by the rule loader.

Use `elspeth-lints justify` against the concrete sub-finding id, for example:

```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli justify \
  --root src/elspeth \
  --repo-root . \
  --allowlist-dir config/cicd/enforce_trust_boundary_honesty \
  --file-path web/composer/tools/sessions.py \
  --rule TBS2 \
  --symbol _execute_set_pipeline \
  --rationale "source_param is forwarded through a closure validated by the referenced test" \
  --owner codex
```

The package selectors `trust_boundary.tests`, `trust_boundary.scope`, and
`trust_boundary.tier` are also accepted by `justify`, but using the concrete
finding id (`TBE*`, `TBS*`, or `TBT*`) keeps the audit attribution sharper.

### What the decorator does NOT solve

- **Module-level patterns** (top-level constants, class-body assignments that touch external data) — these have no enclosing function to decorate; they stay in the YAML allowlist or get refactored to a function.
- **AST-position fingerprint rotation for the residual.** Decorators reduce the residual size; they don't change how the residual is fingerprinted. A separate change to fingerprint on `(file, qualified_symbol, line_relative_to_def_start)` would close that gap and is orthogonal to both pillars.
- **`source` remains reviewer-facing documentation.** The analyzer does not prove a whole-repository external-data call graph for the prose in `source`; it proves local suppression scope (`source_param`) and local behavioural evidence (`test_ref` + `test_fingerprint`). The judge and reviewer still own whether the source description is truthful.

### Residual allowlist migration phases

| Phase | Action | Exit gate |
| --- | --- | --- |
| 0 | Prototype foundation lands: `@trust_boundary`, dataflow scoping, and companion honesty gates | Greenfield lint tests pass; behaviour on hand-crafted fixture validated |
| 1 | Cluster 3 + budget reconciliation lands; final allowlist shape known | Operator reviews shape, decides whether to fund broad migration |
| 2 | Pick highest-density file (`web/composer/tools/sessions.py` likely); write or update the behavioural test first, add the decorator with `test_ref` + `test_fingerprint`, then delete the matching YAML entries | Allowlist count for that file goes to 0 or near-0; no new findings introduced |
| 3 | Review the migration result: YAML entries removed, decorators added, and whether the remaining entries now read as genuine one-offs | Operator judges whether the reduction is meaningful enough to continue; no CI metric currently proves the ratio mechanically |
| 4 | Iterate file-by-file until allowlist is reserved for genuinely one-off exemptions | Operator-judged stopping point: "the remaining entries are the ones that should be one-off entries" |

---

## How the two pillars interact (this is the point)

Once both ship, a new allowlist entry has to clear three independent checks:

1. **Could this be a decorator instead?** If yes, the judge BLOCKs with rationale "this pattern is covered by `@trust_boundary` at function scope — apply the decorator and delete this entry." The agent then refactors. Result: pillar A enforces pillar B's adoption.
2. **If not, is the rationale honest?** Judge reads the code and the rationale; verdict ACCEPTED or BLOCKED.
3. **If BLOCKED but operator insists?** `OVERRIDDEN_BY_OPERATOR` verdict, distinct from ACCEPTED. Rotation rate of this verdict tracked as a meta-metric — too many overrides signals either a too-strict judge or operator-side bypass of own rules.

Without pillar A, pillar B adoption is voluntary and will lag. Without pillar B, pillar A's accept rate stays low because it keeps blocking entries that should have been decorators.

---

## Wardline migration

The operator's stated end-state: this prototype validates the pattern inside ELSPETH; the production tool lives in `wardline` (sibling project). Both pillars are designed against `elspeth_lints`'s rule shape but the abstractions transfer directly:

- The judge is rule-agnostic — it reads `(file, rule_id, symbol, rationale, surrounding_code)` and decides; the input shape is independent of which lint tool produced the finding.
- The `@trust_boundary` decorator is Python-language-bound but the per-language equivalent (Java annotation, Rust attribute, TypeScript decorator) is mechanically translatable. The lint-side dataflow walk is the part that needs re-implementation per language.
- Wardline carries the audit-record schema for verdicts directly; no re-design needed.

Open question for the wardline port: whether the judge model identity (currently `claude-opus-4-7`) is configurable per-project or fixed by wardline. Operator-level decision, not designable from inside the prototype.

---

## Filigree shape (when ready to file)

- Epic: existing `elspeth-297b8f5c5d` (CI allowlist revalidation)
- Pillar A subticket: `prototype-cicd-judge` — Slices 1/2/3
- Pillar B subticket: `adr-trust-boundary-migration` — ADR-class, blocked on cluster-3 corpus shape
- Wardline migration subticket: `wardline-port-cicd-judge` — blocked on prototype validation in production

Drafting the ADR before cluster 3 lands would force decisions on the residual corpus shape from a position of ignorance. Hold until measurement exists.
