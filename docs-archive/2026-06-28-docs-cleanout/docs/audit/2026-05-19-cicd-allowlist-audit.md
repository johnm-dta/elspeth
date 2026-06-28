# CI/CD Allowlist Audit — 2026-05-19

End-to-end audit of every enforcement gate in `scripts/cicd/` and its allowlist in
`config/cicd/`. Goal: separate load-bearing exemptions (Tier 3 trust boundaries,
Tier 1 offensive guards, deliberate I/O fault-isolation) from fixable debt
(stale FPs, deferred refactors, rationalized defensive code, ticket-tagged
temporary exemptions left permanent).

## Live gate state (2026-05-19)

All 14 enforcement gates currently pass locally:

| Gate                                  | Exit | per_file | allow_hits | Notes                  |
| ------------------------------------- | ---- | -------- | ---------- | ---------------------- |
| `enforce_tier_model`                  | 0    | 96       | 578        | 5 TYPE_CHECKING warns  |
| `enforce_freeze_guards`               | 0    | 15       | 0          | FG2 offensive guards   |
| `enforce_audit_evidence_nominal`      | 0    | 0        | 0          | clean                  |
| `enforce_tier_1_decoration`           | 0    | 0        | 0          | clean                  |
| `enforce_composer_exception_channel`  | 0    | 0        | 0          | clean                  |
| `enforce_composer_catch_order`        | 0    | 0        | 0          | clean                  |
| `enforce_contract_manifest`           | 0    | 0        | 0          | clean                  |
| `enforce_frozen_annotations`          | 0    | 0        | 0          | clean                  |
| `enforce_plugin_hashes`               | 0    | -        | -          | runtime hash compare   |
| `enforce_options_metadata`            | 0    | 0        | 0          | clean                  |
| `enforce_component_type`              | 0    | 0        | 0          | clean                  |
| `enforce_guard_symmetry`              | 0    | 1        | 0          | structural model_loaders |
| `enforce_gve_attribution`             | 0    | 2        | 0          | structural DAG validation |
| `check_slot_type_cross_language`      | 0    | -        | -          | live mirror check      |

Plus mypy (clean), ruff (clean), ruff format (clean), `check_contracts` (clean).

## Drift from parent epic baseline

Parent issue `elspeth-297b8f5c5d` (CI allowlist revalidation, proposed 2026-04-17)
recorded a baseline of **354 allow_hits + 93 per_file_rules = 447 entries**.

Today: **578 + 96 = 674 entries** — drift of **+227 entries (+51%) in one month**.

This is the primary structural finding: the allowlist is *growing*, not shrinking,
because new code lands with new exemptions but no countervailing burn-down lane.

## Tier-model allowlist composition (578 allow_hits + 96 per_file_rules)

### By rule code (allow_hits only — authoritative count via colon-split key)

| Rule | Description                              | Count | Mostly       |
| ---- | ---------------------------------------- | ----- | ------------ |
| R5   | `isinstance()` outside trust boundaries  | 222   | Tier 3/T1    |
| R1   | `dict.get()` defensive default           | 151   | Mixed        |
| R6   | Exception handler that swallows          | 118   | Telemetry/I/O |
| R2   | `getattr()` with default                 |  30   | Mixed        |
| R4   | Broad exception handler                  |  26   | Telemetry/I/O |
| R7   | `contextlib.suppress`                    |  15   | Cleanup paths |
| R9   | `dict.pop(key, default)`                 |  10   | Mixed        |
| R8   | `dict.setdefault()`                      |   6   | Mixed        |
| L1   | Upward layer import                      |   1 (per_file) | engine→plugins refactor (tracked) |

**R5 alone is 38% of all allow_hits** — this is a disproportion that warrants
analyzer-level review (rule split: R5a "isinstance in Tier-1 `__post_init__`"
allowed by lattice, vs R5b "isinstance in pipeline data" still flagged).

### By owner tag (authoritative tally — 21 distinct owners)

| Owner                  | Count | Classification                            |
| ---------------------- | ----- | ----------------------------------------- |
| `web-execution`        | 102   | LOAD-BEARING — HTTP request validation    |
| `architecture`         |  94   | LOAD-BEARING — Tier-1 `__post_init__` guards (spot-checked) |
| `web-composer`         |  89   | LOAD-BEARING — LLM tool-call boundaries   |
| `web-sessions`         |  73   | LOAD-BEARING — session-state validation   |
| `composer-audit`       |  64   | LOAD-BEARING — audit-trail validation     |
| `bugfix`               |  37   | **NEEDS INDEPENDENT REVIEW**              |
| `web-auth`             |  29   | LOAD-BEARING — auth payload validation    |
| `feature`              |  25   | **NEEDS INDEPENDENT REVIEW** (sink mapping fallbacks) |
| `web-catalog`          |  12   | LOAD-BEARING — catalog YAML parsing       |
| `composer-guided`      |  11   | LOAD-BEARING — protocol boundary          |
| `web-secrets`          |  11   | LOAD-BEARING — secret-store boundary      |
| `P2-2026-02-02-76r`    |   8   | **FIXABLE-CANDIDATE — ticket-tagged temporary, exporter sparse-token-lookup (8 sites in `_iter_records`)** |
| `web-app`              |   7   | MIXED — at least one confirmed FP (httpx) |
| `web-blobs`            |   5   | LOAD-BEARING — blob upload validation     |
| `infrastructure`       |   3   | LOAD-BEARING — pytest env vars            |
| `plugin-validation`    |   2   | LOAD-BEARING                              |
| `web-middleware`       |   2   | LOAD-BEARING                              |
| `contracts`            |   1   | LOAD-BEARING                              |
| `core`                 |   1   | LOAD-BEARING                              |
| `refactor`             |   1   | **FIXABLE-CANDIDATE — duplicated dict-type check between sinks** |
| `web-runtime`          |   1   | LOAD-BEARING                              |

**No-owner entries**: 0 confirmed. The earlier scan reporting "4 entries with
owner `-`" was a parsing artefact (key-string truncation in the report). All
578 `allow_hits` carry a non-empty owner.

### Expiry status

- **0 expired entries** (the 2026-05-15 expiry crisis was resolved on 2026-05-16)
- **2 near-expiry entries** (<30d) — must be triaged before they go expired:
  - `contracts/contract_builder.py:R6:ContractBuilder:process_first_row` — owner `bugfix`, expires 2026-06-15
  - `plugins/infrastructure/clients/http.py:R4:AuditedHTTPClient:_emit_telemetry_after_audit` — owner `architecture`, expires 2026-06-07
- **41 entries** (7.1%) have an `expires:` date (bounded); **537** are permanent
  (`expires: null`); plus 96 per_file_rules all permanent.
- Spec-recommended ratio for a healthy lifecycle (per
  `axiom-static-analysis-engineering`) is around 70/30 bounded/permanent. The
  project is at **~7/93**. The dominance of `expires: null` is itself a finding:
  the allowlist mechanism supports time-bounded exemptions but the project has
  effectively stopped using them, removing the natural re-review pressure.
- The telemetry subsystem (renewed 2026-05-16 per memory
  `feedback_tier_model_allowlist_expiry_2026-05-16.md`) is the *only* subsystem
  with a documented renewal cadence — its discipline should be adopted
  project-wide.

## Independent-review candidates

These are the categories where the author's `reason:` field is most suspect
and warrants independent SME review (per operator redirect):

### CAT-A: `P2-2026-02-02-76r` ticket-tagged owner (8 entries)

All 8 fingerprints are R1 (`dict.get()` with default) in
`core/landscape/exporter.py:LandscapeExporter:_iter_records`, all with the
identical reason "Sparse lookup - not all rows have tokens".

**Why suspicious:**
1. Owner field tags an explicit ticket ID — the convention is for ticket-tagged
   owners to be *temporary* exemptions pending the ticket's resolution.
2. 8 separate fingerprints in one function indicates 8 separate call sites of
   the same pattern — a refactor opportunity, not 8 independent decisions.
3. "Sparse lookup" is a *semantic* claim that could be made type-explicit
   (e.g. `Optional[Token] = lookup.get(...)` with a single guard, or a typed
   `SparseRowLookup` helper).

**Independent verdict needed:** is this an inherent semantic of the export
process, or a fixable pattern crystallized as 8 exemptions?

### CAT-B: ~~no-owner entries~~ (WITHDRAWN — parsing phantom)

Initial scan reported 4 entries with owner `-`. Re-verification with strict
yaml load found 0 such entries in `web.yaml`. The earlier finding was an
artefact of the reporting script truncating the owner field. **Withdrawn.**

### CAT-C: `bugfix` owner (37 entries)

Sample reasons:
- "TypeError from normalize_type_for_contract for unsupported types (dict, list,
  Decimal) — maps to object"
- "Type annotation objects (e.g. list[str]) may not have __name__ attribute"
- "Intentional fallback pattern - CompositeSecretLoader tries loaders in
  priority order"
- "vars(item) raises TypeError on non-object Pydantic metadata"

**Why suspicious:**
1. `bugfix` is a generic owner that says nothing about *why* the bug fix needed
   the exemption rather than removing it. Often these are spot-fixes that
   stopped a regression without fixing the underlying shape.
2. Several entries claim "intentional fallback pattern" — this is the precise
   shape the rule is designed to catch, and the claim alone doesn't prove
   the fallback is meaning-preserving (CLAUDE.md's coercion vs fabrication
   test).
3. The `CompositeSecretLoader` "priority order" justification, for example,
   conflates a fallback (next loader) with exception suppression (the rule's
   actual target).

**Independent verdict needed:** sample 8-10 entries, read the actual code,
classify each as (a) load-bearing as claimed, (b) load-bearing for a different
reason than claimed, (c) restructurable, or (d) outright unjustified.

### CAT-D: `feature` CSV/JSON sink display-mapping fallbacks (~8 entries)

Pattern: `display_mapping.get(field_name, field_name)` — returning original
name when no display override exists. Six entries in `csv_sink.py`/`json_sink.py`
with near-identical reasons.

**Why suspicious:**
1. The same pattern repeated 6+ times is a refactor opportunity:
   `DisplayMapping.display_name_for(field)` would make "no remap" explicit
   without R1 exemption.
2. The reason classifies this as a "feature" but the actual code shape is
   exactly what R1 targets — defensive defaulting.

**Independent verdict needed:** is the typed-helper refactor worth landing,
or is the current shape the most honest representation?

### CAT-E: `web-app` R1 entry with reason "False positive — httpx.AsyncClient.get() is an HTTP GET request, not dict.get()"

**Why suspicious:** the reason text literally classifies this as a rule false
positive, not an exemption. The right fix is to teach the rule to
distinguish `httpx.AsyncClient.get(...)` from `dict.get(...)`, not to
exempt it. The exemption is itself the bug.

**Independent verdict needed:** confirm rule should be fixed, and scope a
narrow rule patch.

### CAT-F: Telemetry R4/R6 fail-soft entries (~30 entries)

Reasons cluster on "telemetry emission failures must not corrupt main call
path once programmer bugs and Tier 1 errors have been propagated."

**Why suspicious in principle:** this is the legitimate shape for fail-soft
telemetry per CLAUDE.md primacy. But the rule is supposed to catch broad
catches that hide *bugs*, not telemetry I/O. Need to verify each entry
actually re-raises programmer errors before the fail-soft branch.

**Independent verdict needed:** sample 5-6 entries and confirm each catch
block's structure matches the claimed fail-soft pattern.

### CAT-G: `architecture` R5 entries in `contracts/*` (most of the 94)

These claim "Tier 1 offensive guard at __post_init__" — exactly the
institutional pattern from CLAUDE.md ("offensive programming"). Should be
load-bearing by policy. Spot-check 5-6 to confirm they really are
`__post_init__` guards on frozen DTOs, not R5 used elsewhere with a
borrowed justification.

## Output of this audit

1. Subtickets under `elspeth-297b8f5c5d` for each CAT-A through CAT-E
   confirmed-fixable category, with independent SME verdicts attached.
2. A subticket for the **growth-rate** finding: the allowlist gained
   ~212 entries in one month with no ratchet — requires a structural fix
   (e.g. per-PR allowlist-delta budget, or a periodic burn-down lane).
3. A new skill `cicd-allowlist-audit` that re-runs this audit on demand and
   dispatches independent SME agents over suspect categories.
4. An updated baseline in the parent epic.

## Method (so this is reproducible)

```bash
# 1. Verify all gates green
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth \
  --allowlist config/cicd/enforce_tier_model --exclude "**/__pycache__/*"
# ... (each gate, see live state table above)

# 2. Inventory allowlist corpus
.venv/bin/python -c "..."  # see Bash command in audit history for full script

# 3. Independent review per category — dispatch SME agents (see new skill)
```
