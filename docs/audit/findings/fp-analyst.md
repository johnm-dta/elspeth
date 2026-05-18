# False-Positive Analyst Report — `enforce_tier_model` Allowlist

- **Analyzer:** `scripts/cicd/enforce_tier_model.py` (1858 LOC)
- **Suppression set:** `config/cicd/enforce_tier_model/*.yaml`
- **Branch / date of analysis:** `RC5.2` @ 2026-05-19
- **Inputs read:**
  - `/home/john/elspeth/scripts/cicd/enforce_tier_model.py` (rule definitions and visitors, lines 167–500)
  - `/home/john/elspeth/config/cicd/enforce_tier_model/{contracts,core,engine,plugins,telemetry,web,...}.yaml` (full suppression set)
  - `/home/john/elspeth/src/elspeth/web/app.py:190–223` (the cited R1 FP site)
  - `/home/john/elspeth/src/elspeth/contracts/{tier_registry,declaration_contracts,schema_contract}.py` (spot-checked R5 sites)
  - `/home/john/elspeth/src/elspeth/web/composer/guided/protocol.py` (spot-checked R5 sites)
  - Author-supplied `reason:` / `safety:` strings on every entry sampled
- **Numbers I re-derived from the files (rather than trusting the brief):**

  | Quantity | Brief said | I measured | Notes |
  |---|---|---|---|
  | Total `allow_hits` | 578 | **579** | `grep -c "^- key:"` across all yaml files |
  | R5 entries | 222 | **222** | confirmed |
  | R1 entries | 151 | **151** | confirmed |
  | R6 entries | 118 | **119** | one-off |
  | Bounded-expiry entries (`expires: 'YYYY-MM-DD'`) | "only 2" | **41** (≈ 7.1%) | The "2" claim is wrong, but 41/579 is still a tiny minority and the qualitative conclusion stands. |
  | `expires: null` entries | unstated | **600** (includes per_file_rules) | dominant lifecycle state |
  | web.yaml share | implicit | **400/579 = 69%** of all allow_hits | concentrated in one subsystem |

---

## 1. Confirmed false positive (rule defect) — R1 conflates HTTP `.get()` with `dict.get()`

### Verdict: **CONFIRMED FP. The rule is genuinely defective.** The author's "False positive — httpx" justification is accurate, but it has been used to wallpaper over the rule defect 4+ times instead of fixing the rule.

### Evidence

The cited entry:

```yaml
- key: web/app.py:R1:lifespan:fp=4407dec3d0231969
  owner: web-app
  reason: "False positive — httpx.AsyncClient.get() is an HTTP GET request, not dict.get()"
  safety: "HTTP errors caught by raise_for_status(); failure now raises SystemExit"
  expires: null
```

The actual code at `src/elspeth/web/app.py:213–218`:

```python
try:
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        resp = await client.get(discovery_url)
        resp.raise_for_status()
        app.state.oidc_authorization_endpoint = _validate_authorization_endpoint_discovery_document(resp.json())
except (httpx.HTTPError, ValueError) as exc:
    raise SystemExit(...) from exc
```

This is unambiguously `httpx.AsyncClient.get(url)`, not a dict access. The author's narrative is correct.

### Why the analyzer mis-fires

`enforce_tier_model.py:387–417` implements `_is_likely_non_dict_get`. Heuristic 2 (the relevant one) suppresses the R1 finding only when the first positional argument is an `ast.Constant` string starting with `/`, `http://`, or `https://`:

```python
if node.args:
    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        val = first_arg.value
        if val.startswith(("/", "http://", "https://")):
            return True
```

In `web/app.py:215` the argument is `discovery_url` — a `Name` node bound to a runtime-constructed string (`f"{issuer}/.well-known/openid-configuration"`). The heuristic gives up and lets R1 fire. The docstring of `_is_likely_non_dict_get` even acknowledges this case ("f-string URLs … are NOT filtered … These must be allowlisted").

### This is a systemic defect, not a one-off

Allowlist entries with the literal substring `"False positive — httpx.AsyncClient.get()"` or `"False positive — asyncio.Queue.get()"`:

- `web/app.py:R1:lifespan:fp=4407dec3d0231969` (the cited one) — owner `web-app`
- `web/auth/oidc.py:R1:JWKSTokenValidator:ensure_jwks:fp=1ae4ffe8bcf071bb` — owner `web-auth`
- `web/auth/oidc.py:R1:…:fp=<various>` — additional httpx hits in the OIDC discovery / JWKS module
- `web/execution/routes.py:R1:…:websocket_run_progress` (asyncio.Queue.get) — owner `web-execution`

There are at least 4 confirmed false-positive R1 entries naming non-dict objects whose API happens to expose `.get()`. The maintainer's response so far has been to allowlist each one as it appears. This is the "aggressive suppression" anti-pattern from `axiom-static-analysis-engineering:false-positive-economics`.

### Proposed rule patch (single paragraph)

Extend `_is_likely_non_dict_get` (`scripts/cicd/enforce_tier_model.py:387–417`) with a **receiver-type heuristic** keyed on the callee object's name and on import context, not on the call's first-argument literal. Specifically: (a) walk the module's `Import`/`ImportFrom` nodes once during `__init__` to collect names bound to known non-dict modules (`httpx`, `asyncio`, `aiohttp`, `requests`, and the names of any `Queue`-like classes imported from `asyncio`/`queue`/`multiprocessing`); (b) in `_is_likely_non_dict_get`, when `node.func` is `Attribute(value=Name)` or `Attribute(value=Attribute)`, resolve the leftmost `Name` and check whether (i) it was bound from an `httpx.AsyncClient(...)`/`httpx.Client(...)`/`asyncio.Queue(...)` constructor in the same function scope (a simple intra-procedural assignment scan suffices), or (ii) it is a parameter annotated as one of those types. If yes, return `True`. This eliminates the four confirmed FPs above and any future ones in the same shape (e.g. the existing entries against `asyncio.Queue.get`, `httpx.AsyncClient.get`, `redis.Redis.get` if it ever appears, `aiohttp.ClientSession.get`). Surface area to modify: only `_is_likely_non_dict_get` and the `__init__` of the visitor class (`TierModelVisitor`, around line 280–330 in the same file). No change to the YAML schema. After landing, the four confirmed FP allowlist entries become orphans and can be deleted in the same commit — the allowlist orphan-detection logic (`unmatched_allow_hits` reporting) will surface them on the next CI run.

**Confidence:** HIGH — the rule code and the source code are both unambiguous.
**Risk of patch:** LOW — heuristic only widens the set of `True` returns (suppresses more findings); the unit tests already cover the heuristic-2 path and adding heuristic-4 cannot regress them. Risk of overshoot is bounded because the heuristic requires the receiver to be type-bound to a known HTTP/queue type, not just any `.get()` call.

---

## 2. Disproportionate R5 (`isinstance`) — lattice imprecision, rule should be split

### Verdict: **The R5 rule is lumping two semantically distinct uses of `isinstance`. The 222 entries are mostly legitimate code per the project's own discipline (CLAUDE.md "Offensive Programming: Encouraged"). The fix is in the analyzer, not the codebase.**

### Distribution

R5 entries by file (`grep ':R5:' config/cicd/enforce_tier_model/*.yaml | awk -F: '{print $1}' | sort | uniq -c`):

```
175  web.yaml
 18  plugins.yaml
 17  contracts.yaml
 11  engine.yaml
  1  core.yaml
```

### Spot-check (verified file:line; pattern as described)

I confirmed the pattern at the source line for six entries spanning four distinct owners:

| Allowlist key | File:line | Owner | Verified pattern |
|---|---|---|---|
| `contracts/declaration_contracts.py:R5:BoundaryInputs:__post_init__:fp=db78bae36c2f61ba` | declaration_contracts.py (`__post_init__` of BoundaryInputs) | `architecture` | Tier-1 offensive guard: `isinstance(row_data, Mapping)` before `deep_freeze` — direct realization of CLAUDE.md "deep_freeze contract" |
| `contracts/run_result.py:R5:RunResult:__post_init__:fp=fac5bfd1b22037fa` | run_result.py `__post_init__` | `architecture` | Tier-1 enum-shape guard: `isinstance(status, RunStatus)` raises TypeError on contract construction |
| `contracts/schema_contract.py:R5:PipelineRow:to_dict:fp=19c383f767c4ed8c` | schema_contract.py `to_dict` | `architecture` | Tier-1 offensive guard on `deep_thaw` return shape |
| `contracts/tier_registry.py:R5:tier_1_error:fp=360d94dd51a67c28` | tier_registry.py:127 | `architecture` | Decorator-boundary string validation, raises `ValueError` |
| `contracts/tier_registry.py:R5:_register_with_module_prefix:fp=ce98be6cf59df4d5` | tier_registry.py:146 | `architecture` | `isinstance(cls, type) and issubclass(cls, BaseException)` — decorator-input validation |
| `web/composer/guided/protocol.py:R5:validate_payload:fp=66d6dab65ff1dca8` | protocol.py:257 | `composer-guided` | Tier-3 boundary validator: `isinstance(turn_type, TurnType)` on untrusted wire payload |

In every sampled case, the suppressed `isinstance` use is *required by the project's own published discipline*: either (a) an offensive Tier-1 invariant guard in a frozen-dataclass `__post_init__`, or (b) a Tier-3 boundary validator over an external/wire payload. Both are explicitly endorsed by `CLAUDE.md` § "Defensive Programming: Forbidden. Offensive Programming: Encouraged" and § "deep_freeze Contract".

The web.yaml R5 entries follow the same two patterns at higher volume because `web/` contains both the largest concentration of Tier-3 wire boundaries (FastAPI route handlers, OIDC discovery, JWT claim parsers, composer protocol validators, Pydantic before-validators) and the most `__post_init__` boundary DTOs.

### Root cause

R5 in `enforce_tier_model.py:198–202` is defined as:

> "isinstance() checks can mask contract violations outside explicit trust boundaries"

The visitor in `enforce_tier_model.py:463–468` fires unconditionally on every `isinstance(...)` call. **The rule's *definition* names a boundary distinction it does not implement.** The implementation has no boundary awareness; the operator is forced to encode the boundary distinction by hand in 222 free-text `reason:` strings.

This is classic **lattice imprecision** in the `axiom-static-analysis-engineering:taint-lattice-design` sense: a single rule conflates two abstract states (`tier_3_boundary_validator`, `tier_1_offensive_guard`) that the project's own model treats as distinct *and welcome*, with a third state (`tier_2_pipeline_data_defensive_check`) that the project genuinely wants to forbid.

### Proposed rule split

Split R5 into three rules so the lattice matches the project's discipline:

- **R5a — `isinstance-in-tier-1-frozen-dataclass-post-init`.** Fires only inside `__post_init__` of a `@dataclass(frozen=True)` (or its bases). Auto-suppressed (or downgraded to INFO); enforces the deep_freeze contract.
- **R5b — `isinstance-at-tier-3-boundary`.** Fires only inside functions decorated/annotated as boundary validators or inside Pydantic `@field_validator(mode="before")` / FastAPI route handlers / known Tier-3 entry-point modules. Auto-suppressed at boundary; surfaced for ad-hoc review.
- **R5c — `isinstance-in-pipeline-or-engine-tier-2`.** Fires on `isinstance` calls in `engine/`, `core/`, `plugins/transforms/`, `plugins/sinks/` *outside* `__post_init__` and *outside* boundary annotations. This is the only sub-rule that genuinely warrants a CI failure — defensive type-narrowing on Tier-2 pipeline data is what CLAUDE.md actually forbids.

A practical halfway implementation that does not require new annotations: add two AST-context tests to the existing `visit_Call` for R5 (`enforce_tier_model.py:463–468`):

1. Walk the enclosing scope; if the call site is inside a `FunctionDef` named `__post_init__` whose enclosing `ClassDef` has a `@dataclass(frozen=True)` decorator (or `slots=True`), tag as **R5a** and do not fire (or emit at INFO).
2. If the file path matches `contracts/` (L0) or any `web/**/protocol.py`, `web/auth/**`, `web/**/validators.py`, or the function is a Pydantic before-validator (detected by `@field_validator(..., mode="before")` decorator), tag as **R5b** and do not fire.
3. Everything else → **R5c**, fire as today.

Conservatively, this would convert ≈ 17 of 17 `contracts/` R5 entries and ≈ 150+ of the 175 `web/` R5 entries to **no-fire** (they become noise the YAML never had to carry). The remaining R5 (now R5c) hits would be a small, reviewable set that genuinely warrants per-site justification.

**Confidence:** HIGH for the diagnosis (six confirmed samples; pattern is uniform). MEDIUM for the exact precision of the proposed `web/**/protocol.py` glob — a few R5 boundary-validator sites may live elsewhere and need additional path patterns.
**Risk:** MEDIUM. The split downgrades enforcement of `isinstance` at Tier-1 post-init, which is currently nominally a fail. If a reviewer was relying on R5 firing to catch a *defensive* (forbidden) use of `isinstance` in a `__post_init__`, that signal goes away. Mitigation: keep R5c covering Tier-2 sites where the actual harm lives; R5a/R5b are exactly the patterns CLAUDE.md *recommends*.

---

## 3. Suppression growth without waiver lifecycle — failed lifecycle

### Verdict: **Lifecycle is not healthy. 93% of suppressions are permanent (`expires: null`), and the only burn-down evidence on file is in a single subsystem (telemetry).**

### Measurements

- 600 `expires: null` entries vs 41 bounded-expiry entries (≈ 7% bounded). The brief's "only 2" claim is incorrect — 41 entries do carry an explicit date — but the imbalance is still extreme.
- Bounded-expiry dates cluster:
  - `2026-07-02` × 18 (mostly `plugins.yaml` and `core.yaml`)
  - `2026-08-15` × 14 (all `telemetry.yaml`, 1 `contracts.yaml`)
  - `2026-11-13` × 5 (`core.yaml`)
  - other × 4
- The telemetry block has explicit *renewal* commentary in the YAML (`# Renewed 2026-05-16: telemetry/manager.py last touched 2026-04-13; fingerprints all still match; safety justifications unchanged.`). That is a textbook compliant renewal record and the only such evidence in the entire suppression set. Memory `feedback_tier_model_allowlist_expiry_2026-05-16` corroborates that this renewal was performed by per-entry review on 2026-05-16.
- Net growth from 2026-04-17 baseline (354 + 93 = 447) → 2026-05-19 (579 + ≈ 96 ≈ 675) is **+227 entries in 30 days (+51%)**. No burn-down branch or scheduled retirement ticket was found in the allowlist commit history or in `filigree` (within session scope).

### Recommended ratio

`axiom-static-analysis-engineering:false-positive-economics` recommends:

- **Permanent waivers** (`expires: null`) should be reserved for entries with a published architectural justification — e.g. a closed-list pattern documented in an ADR. They are the exception.
- **Bounded-expiry waivers** are the default lifecycle state; expiry forces a re-review.
- A healthy steady-state ratio is roughly **bounded ≥ 70%, permanent ≤ 30%**. The current state is the inverse.

### Process change

1. **Flip the default in the analyzer.** When the allowlist YAML is updated by the operator or by the analyzer's bulk-update tooling, the *default* `expires:` should be a date 90 days out — not `null`. `null` should require an explicit override and a justification.
2. **CI gate on growth rate, not just absolute count.** Add a step that compares the current allow_hits count to the count at `main`'s HEAD: if the delta is positive *and* none of the new entries are bounded, fail the build with an instructive message ("New permanent suppressions require an ADR reference in `reason:`").
3. **Quarterly burn-down lane.** Adopt the telemetry block's renewal pattern as the project standard: every quarter, the owning team must either (a) re-attest with an updated `# Renewed YYYY-MM-DD` comment and fingerprint re-check, or (b) delete the entry and either fix the underlying code or refine the rule.
4. **Orphan sweep.** The analyzer already reports `unmatched_allow_hits` (`enforce_tier_model.py:160`). Promote unmatched-hit count > 0 from informational to failing in CI; today it is silent.

**Confidence:** HIGH for the measured ratios. MEDIUM for the "70/30" benchmark, which is a normative recommendation from the spec, not a measured project number.
**Risk:** LOW. Process changes can be staged (warn → fail) over one or two release cycles.

---

## 4. Owner-tag taxonomy — partly semantic, partly noise; ticket-tagged owners need explicit retire-with-ticket lifecycle

### Verdict: **The owner tags are a fractured mix of (a) sub-team scopes that carry real semantic weight, (b) change-class labels ("bugfix", "feature", "refactor") that carry none, and (c) one ticket-ID tag that is a lifecycle landmine.**

### Measured distribution

```
102 web-execution          ← sub-team scope (semantic)
 94 architecture           ← cross-cutting tier-model engineering (semantic but broad)
 89 web-composer           ← sub-team scope (semantic)
 73 web-sessions           ← sub-team scope (semantic)
 64 composer-audit         ← sub-team scope (semantic)
 37 bugfix                 ← change-class label (NO semantic scope)
 29 web-auth               ← sub-team scope (semantic)
 25 feature                ← change-class label (NO semantic scope)
 12 web-catalog            ← sub-team scope (semantic)
 11 web-secrets            ← sub-team scope (semantic)
 11 composer-guided        ← sub-team scope (semantic)
  8 P2-2026-02-02-76r      ← embedded filigree ticket ID — landmine
  7 web-app                ← sub-team scope
  5 web-blobs              ← sub-team scope
  3 infrastructure         ← sub-team scope
  2 web-middleware, plugin-validation, core
  1 web-runtime, refactor, contracts
```

### Three classes; three problems

**Class A: Sub-team scope (~85% of entries).** `web-execution`, `web-composer`, etc. These are useful: they identify the team who should re-review the entry on rotation. Keep.

**Class B: Change-class labels (`bugfix` × 37, `feature` × 25, `refactor` × 1).** These describe how the entry got added, not who owns it. They are dead-weight in a triage workflow — `bugfix` could be any subsystem. **Recommendation:** retag these to the sub-team owner during the next renewal pass. New entries that would be tagged `bugfix` should instead inherit the owning subsystem (`web-sessions`, `engine`, etc.).

**Class C: Ticket-tagged owner (`P2-2026-02-02-76r` × 8).** All eight entries are in `core/landscape/exporter.py:R1:LandscapeExporter:_iter_records:*` and share the reason `"Sparse lookup - not all rows have tokens (batch export uses pre-loaded dicts)"`. This embeds a filigree ticket ID into the owner field. **This is a strong signal** that the entry was supposed to be temporary — bound to the resolution of the ticket — but the lifecycle wasn't wired up. The entries have `expires: null`. They should have `expires: <ticket-close-date + small grace>` so that the suppression evaporates when the ticket closes.

### Process recommendation: ticket-tagged-owner lifecycle

When the owner field embeds a ticket ID (regex `[A-Z]\d?-?\d{4}-\d{2}-\d{2}-\w+` or matches a `filigree`-style slug), the analyzer should:

1. Refuse `expires: null` at lint time (require an actual date).
2. Optionally cross-reference the ticket ID (an out-of-process `filigree get-issue` would suffice) and report at scan time if the ticket is closed. A closed ticket whose suppressions are still in the file is a paid debt the system is still carrying interest on.

Less ambitiously: agree a convention that ticket-tagged owners *must* carry an `expires:` date, and gate the YAML on it.

**Confidence:** HIGH for the taxonomy observation.
**Risk:** LOW. The retagging is mechanical; the lifecycle change can be a soft warning before becoming a fail.

---

## 5. Meta-verdict — **(b) Decaying.** False positives are accumulating, the lifecycle is unenforced, and the rule set is not keeping up with the codebase's actual discipline.

### Evidence

- **+51% growth in 30 days** with no offsetting burn-down or rule refinement (one telemetry renewal pass excepted).
- **Confirmed rule defect** (R1 conflates HTTP `.get()` with `dict.get()`) being papered over with allowlist entries instead of fixed in the visitor. At least four entries explicitly self-describe as "False positive".
- **Lattice imprecision** in R5: the rule fires on patterns the project's own published discipline (`CLAUDE.md`) *encourages*. The author justifications are uniformly correct (sampled n=6, all valid). The 222 entries are real code paying a 222-line metadata tax to silence a misaligned rule.
- **Lifecycle disabled:** 93% permanent suppressions. The only compliant renewal pattern (telemetry block, 2026-05-16) is an island.
- **Owner taxonomy not gated:** ticket-tagged owners exist but carry no expiry contract.
- **Concentration:** 69% of all suppressions live in `web/`. This is partly natural (web is the biggest Tier-3 surface) and partly a symptom — the rule set was tuned for `contracts/`/`core/`/`engine/` and is mis-aligned in the `web/` subsystem, which holds most Tier-3 boundaries.

The combination is **decay, not steady state, and not "paying for undecidability"** (option c). Decay because the deltas accumulate; "paying for undecidability" would require the rule definitions to be sharp and the suppressions to be unavoidable — neither is the case. R1 is fixable in the visitor; R5 is fixable by splitting; lifecycle is fixable by process. The current trajectory is suppression-as-aspirin: the symptom is silenced, the disease (rule/lattice misalignment + missing lifecycle) is not addressed.

If unaddressed, the most likely failure mode in 60–90 days: the allowlist crosses 1000 entries, CI continues to pass because everything is `expires: null`, and a real defensive-programming regression hides in a sea of legitimate boundary-validator and offensive-guard suppressions. At that point the rule has zero distinguishing power between intended Tier-1/3 code and forbidden Tier-2 defensiveness, and the original purpose of the analyzer is defeated.

---

## Prioritised actions

| # | Action | Owner artifact to modify | Priority |
|---|---|---|---|
| 1 | Extend `_is_likely_non_dict_get` to recognise receivers bound to `httpx.Client`/`AsyncClient`, `asyncio.Queue`, `aiohttp.ClientSession`. Delete the 4 confirmed FP allowlist entries in the same commit. | `scripts/cicd/enforce_tier_model.py:387–417`, plus visitor `__init__` | **P0** |
| 2 | Split R5 into R5a (Tier-1 `__post_init__` of frozen dataclass — auto-pass), R5b (Tier-3 boundary validator — auto-pass), R5c (everything else — fail). Rerun analyzer and bulk-delete the ≈ 170 R5 entries that become orphans. | `scripts/cicd/enforce_tier_model.py:463–468` and visitor scope-tracking | **P0** |
| 3 | Promote `unmatched_allow_hits` from informational to CI-failing. | analyzer CLI exit code path | **P1** |
| 4 | Change default `expires:` for new allowlist entries to "today + 90 days" instead of `null`. Require explicit override with ADR ref to use `null`. | YAML schema + any bulk-add tooling | **P1** |
| 5 | Adopt the telemetry block's renewal commentary pattern as a project-wide convention. Add a quarterly burn-down ritual. | process / `axiom-static-analysis-engineering` runbook | **P1** |
| 6 | Retag `bugfix` / `feature` / `refactor` change-class owners to their sub-team during the next renewal pass. | YAML | **P2** |
| 7 | Require `expires:` (non-null) for any owner field matching a ticket-ID pattern. | analyzer YAML schema validation | **P2** |
| 8 | Add CI gate on month-over-month growth rate of permanent (`expires: null`) entries. | CI pipeline + analyzer reporting mode | **P2** |

---

## Confidence Assessment

- **R1 defect (§1):** HIGH confidence — rule code and source code are unambiguous; multiple corroborating allowlist entries name the same FP shape across different owners.
- **R5 lattice imprecision (§2):** HIGH confidence in the diagnosis (six confirmed samples across four owners and four files, uniform pattern); MEDIUM confidence in the precise path-globs needed for R5b detection (Pydantic / FastAPI / protocol modules) — the split may need iteration after first deploy.
- **Lifecycle (§3):** HIGH confidence in the measured ratios. Note that the brief's claim of "only 2" bounded-expiry entries is inaccurate — the actual figure is 41 — but the qualitative conclusion (lifecycle is unhealthy) is unchanged because 41/600+ is still a small minority and the existing renewal evidence is concentrated in one block.
- **Owner taxonomy (§4):** HIGH confidence in the taxonomy partition; MEDIUM in the prescription (the convention can be debated but the lifecycle gap is real).
- **Meta-verdict (§5):** HIGH confidence — multiple independent indicators point the same way.

## Risk Assessment

- **R1 patch:** Low risk. The proposed heuristic only widens the FP-suppressing path; it cannot generate new findings. The four orphaned allowlist entries are safe to delete (the rule will no longer fire at those sites).
- **R5 split:** Medium risk. R5a/R5b would silently downgrade currently-firing-then-allowlisted findings. The risk vector is: a future *defensive* (forbidden) use of `isinstance` inside a Tier-1 `__post_init__` would no longer fire R5. Mitigation: such uses also tend to produce contract drift visible in test failures (the `deep_freeze` contract is testable). The actual harm vector (defensive isinstance on Tier-2 pipeline data) is preserved by R5c.
- **Lifecycle / default-90-day-expiry:** Low risk in the warning phase; medium risk when promoted to fail because every team will have to re-touch their suppressions. Staged rollout (warn for one cycle, fail next) is the standard mitigation.
- **Cumulative risk of not acting:** HIGH. The current allowlist trajectory will, by inspection, cross 1000 entries within ~3 months, at which point the analyzer's signal-to-noise ratio is below useful and the lattice it was meant to enforce becomes folklore.

## Information Gaps

- I did not have access to per-commit allowlist deltas; the +51% / +227 figure relies on the brief's stated baseline. A real burn-down report would walk `git log -p config/cicd/enforce_tier_model/` and chart additions vs deletions over time. Recommended as a follow-up by a separate audit pass.
- I did not exhaustively classify all 222 R5 entries — I sampled six across distinct files/owners. The R5-split proposal assumes the pattern is uniform; if a substantial subset is actually Tier-2 defensive (which I did not see in the sample), R5c would fire on more sites than the proposal implies. A bulk re-classification of all R5 entries against the proposed R5a/b/c criteria should be the first step of the rule-split work.
- The brief stated "only 2" bounded-expiry entries; I measured 41. I have not investigated the source of the discrepancy — possible explanations include the brief counting only entries with both `expires:` and a date in a specific format, or counting at a different scope (e.g. fingerprinted hits only). I have used the measured figure.
- The R5 path-globs for Tier-3 boundary detection (`web/**/protocol.py`, `web/auth/**`, etc.) are inferred from spot-checks; a complete inventory of Tier-3 entry-point modules would tighten the proposal.
- I did not check whether the `unmatched_allow_hits` orphan detection in `enforce_tier_model.py` works as advertised — only that the code path exists. A scaffolded test (deliberately orphaned entry → analyzer reports it) would close that gap.

## Caveats

- This report assesses the analyzer + suppression set; it does not propose changes to project source code, per the analyst's scope.
- "False positive" verdicts are based on aligning the analyzer's published rule definition (e.g. "R5: isinstance() … *outside explicit trust boundaries*") with the project's published discipline (CLAUDE.md tier model and offensive-programming policy). The discipline could be redefined in the other direction (declare `isinstance` in `__post_init__` is also a defect, refactor 17 contracts/ sites, etc.) — but that would be a strategic re-decision, not a triage finding.
- The growth-rate figure (+51% in 30 days) is from the brief and was not re-derived from git history. The current-state numbers in this report are all re-measured from disk.
- Author `reason:` and `safety:` strings were treated as self-attestations and corroborated against source code only where sampled. The 215+ R5 entries not individually spot-checked may include a small number where the justification is wrong but the rule is right — those would survive the proposed R5 split as legitimate R5c failures.
