# cicd-judge-cli prototype — feature map

**Status:** prototype-in-flight (worktree: `.worktrees/cicd-judge-cli`, branch `feat/cicd-judge-cli-prototype`). Slice 1 (rotate replacement) is implemented and awaiting commit. Slices 2–3 designed but not built. Slice 4 (`@trust_boundary` decorator) is a separately-scoped follow-on — out of prototype scope, documented here so the structural endgame is visible while the prototype's surface area is decided.

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
- **Judge does NOT fix.** A BLOCKED verdict returns the failure mode to the agent; the agent figures out the remediation (refactor, broaden a per-file rule, move under a decorator, abandon the suppression). This is the operator-stated constraint: "the judge doesn't have to fix the problem, it just blocks the change to the whitelist and tells the agent to figure it out."
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
  expires: <existing field>
  safety: <existing field>
```

The judge's rationale is the new audit primitive. It is independently re-readable in 6 months when someone asks "why did we exempt this?" — the YAML answers without re-running the model.

### Decay sweep (Slice 3 of prototype)

`elspeth-lints reaudit` re-runs the judge across existing entries. Used at allowlist-renewal time (when an `expires:` is bumped). Decisions to keep an entry must survive a fresh judge pass. This closes the loop on the "self-attestation never re-reviewed" failure mode.

### Out-of-scope for the prototype

- No web UI; CLI only.
- No model fallback; Opus is named, fixed, single-vendor (the audit trail records the model used).
- No async batching of judge calls; one entry, one synchronous call. Performance is irrelevant for the volume (low tens of new entries per day at peak).

---

## Pillar B — `@trust_boundary` decorator (future work, not in prototype)

**Sequencing:** ships AFTER the Cluster 3 (web/composer/tools) reconciliation lands and the residual allowlist shape is known. The decorator's value depends on the final corpus shape — how many of the remaining entries cluster around function-scoped Tier-3 boundaries vs. how many are genuinely orphaned. Without that measurement, the decorator's reach is speculation.

### Shape (sketch — full design lives in ADR draft, not yet filed)

```python
# src/elspeth/contracts/trust_boundary.py
@trust_boundary(
    tier=3,
    source="LLM tool-call arguments emitted by composer model",
    source_param="arguments",
    suppresses=("R1", "R5"),
    invariant="raises ToolArgumentError on shape mismatch; never coerces silently",
    test_ref="tests/unit/web/composer/tools/test_sessions.py::test_set_pipeline_rejects_malformed_nodes",
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
| `enforce_trust_boundary_tests.py` | Every `@trust_boundary` has a `test_ref` pointing to a real pytest node that exercises malformed input raising the documented error | Stops the decorator from being a vibe-justification; behavioural evidence required |
| `enforce_trust_boundary_scope.py` | `source_param` is in the function signature; function body actually reads from it | Stops drive-by `@trust_boundary` on functions that don't take external data |
| `enforce_trust_boundary_tier.py` | `tier=3` is the only accepted value | Tier-1 and Tier-2 must crash, not suppress — the decorator is not a general-purpose lint silencer |

### What the decorator does NOT solve

- **Module-level patterns** (top-level constants, class-body assignments that touch external data) — these have no enclosing function to decorate; they stay in the YAML allowlist or get refactored to a function.
- **AST-position fingerprint rotation for the residual.** Decorators reduce the residual size; they don't change how the residual is fingerprinted. A separate change to fingerprint on `(file, qualified_symbol, line_relative_to_def_start)` would close that gap and is orthogonal to both pillars.
- **Author writes shallow justification.** The decorator's `source` and `invariant` fields are still self-attested. The test-ref requirement mitigates this (a behavioural test must exist) but does not eliminate it. This is the residual problem the judge solves at write time.

### Migration phases

| Phase | Action | Exit gate |
| --- | --- | --- |
| 0 | Cluster 3 + budget reconciliation lands; final allowlist shape known | Operator reviews shape, decides whether to fund pillar B |
| 1 | Implement `@trust_boundary` + companion lint rule changes | Greenfield lint tests pass; behaviour on hand-crafted fixture validated |
| 2 | Pick highest-density file (`web/composer/tools/sessions.py` likely); migrate every entry on that file into a decorator on the enclosing function; delete the YAML entries | Allowlist count for that file goes to 0 or near-0; no new findings introduced |
| 3 | Measure: YAML count drops by N, decorator count rises by M, target ratio M ≪ N | If M ≈ N, decorator is mis-shaped; pause and rethink |
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
- Pillar B subticket: `adr-trust-boundary-decorator` — ADR-class, blocked on cluster-3 corpus shape
- Wardline migration subticket: `wardline-port-cicd-judge` — blocked on prototype validation in production

Drafting the ADR before cluster 3 lands would force decisions on the residual corpus shape from a position of ignorance. Hold until measurement exists.
