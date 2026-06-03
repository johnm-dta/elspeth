# Tickets to file under parent `elspeth-297b8f5c5d`

Draft list. Subtickets to create once all 4 SME agents complete. Each ticket
cites its independent-review source (SME agent report) and proposed change.

## Confirmed from completed agents (silent-failure-hunter + fp-analyst)

### T1 — R1 rule defect: HTTP `.get()` conflated with `dict.get()`

- **Source:** `fp-analyst.md` §1 — CONFIRMED FP, rule defective.
- **Surface area:** `scripts/cicd/enforce_tier_model.py:387–417`
  (`_is_likely_non_dict_get`).
- **Defect:** Only detects HTTP `.get()` when URL is a *string literal*. F-string
  or Name-derived URLs (`httpx.AsyncClient.get(discovery_url, ...)`) trigger R1.
- **Confirmed wallpaper exemptions:** 4+ entries self-describing as
  "False positive — httpx" / "asyncio.Queue.get()" across `web-app`, `web-auth`,
  `web-execution` owners.
- **Proposed fix:** Patch visitor with a receiver-type heuristic — intra-procedural
  assignment scan in visitor `__init__` to resolve receiver back to known
  `httpx.AsyncClient` / `aiohttp.ClientSession` / `asyncio.Queue` constructors.
- **Expected effect:** ~4 allowlist entries become deletable.
- **Priority:** P2, type=bug
- **Title:** `enforce_tier_model R1 misfires on httpx/asyncio .get() — receiver-type heuristic needed`

### T2 — R5 rule lattice imprecision: split R5a/R5b/R5c

- **Source:** `fp-analyst.md` §2 — Lattice imprecision, rule should be split.
- **Surface area:** `scripts/cicd/enforce_tier_model.py` R5 visitor (lines ~430+).
- **Defect:** R5 lumps three semantically distinct uses:
  - R5a — `isinstance` inside `__post_init__` of `@dataclass(frozen=True)` — this
    is the encouraged Tier-1 offensive-programming pattern (CLAUDE.md mandates it).
  - R5b — `isinstance` in a Tier-3 boundary validator (HTTP/YAML/env-var entry
    point) — also encouraged.
  - R5c — `isinstance` inside Tier-2 pipeline data path — the actual target of
    the rule.
- **Proposed fix:** Split R5 into the three sub-rules. R5a and R5b auto-pass
  (no exemption needed). R5c stays as the failure case.
- **Expected effect:** ~170 of 222 R5 entries become orphans (deletable).
- **Priority:** P1, type=feature
- **Title:** `enforce_tier_model R5 rule split — separate __post_init__ guards (R5a) from Tier-3 validators (R5b) from Tier-2 pipeline checks (R5c)`

### T3 — Allowlist lifecycle: enforce bounded expiry, ratchet permanent growth

- **Source:** `fp-analyst.md` §3 — Lifecycle is broken (7% bounded, 93% permanent
  against a 70/30 target).
- **Defect:** 537 of 578 allow_hits (93%) carry `expires: null`. Only telemetry
  subsystem has a documented renewal cadence (per memory
  `feedback_tier_model_allowlist_expiry_2026-05-16.md`).
- **Proposed changes:**
  1. Default new allow_hit entries to `expires: today + 90d` (or "next quarterly
     review"). Codify in `scripts/cicd/enforce_tier_model.py` allowlist
     scaffolding / docs.
  2. Add a CI gate (separate workflow lane, not a hard fail) that reports the
     count of permanent (`expires: null`) entries per PR and refuses to merge
     if the count grows without an exemption.
  3. Adopt telemetry subsystem's renewal commentary pattern as project-wide
     standard (annotation in `_defaults.yaml`).
- **Priority:** P2, type=feature
- **Title:** `enforce_tier_model allowlist lifecycle — ratchet permanent-waiver growth, default new entries to bounded expiry`

### T4 — Owner taxonomy: retag change-class owners, enforce ticket-ID lifecycle

- **Source:** `fp-analyst.md` §4 — Owner tags are partly semantic, partly noise.
- **Defects:**
  1. ~63 entries owned by change-class labels (`bugfix` × 37, `feature` × 25,
     `refactor` × 1) — these describe how the entry got added, not who owns it.
     Retag to subsystem owner (`web-sessions`, `engine`, `contracts`, etc.).
  2. 8 entries tagged `P2-2026-02-02-76r` (a filigree ticket ID) all in
     `core/landscape/exporter.py:_iter_records`, all with `expires: null` —
     ticket-ID owners should be bound to ticket close.
- **Proposed changes:**
  1. Sweep retag pass — one PR per subsystem, mechanical owner change with
     `expires:` ratcheted to today+90d.
  2. Add analyzer schema validation: refuse `expires: null` when owner field
     matches a filigree-ticket-ID pattern (`^[PC]\d-\d{4}-\d{2}-\d{2}-[a-f0-9]+$`).
- **Priority:** P2, type=task
- **Title:** `enforce_tier_model owner taxonomy — retag bugfix/feature/refactor entries to subsystem owners, gate ticket-tagged owners on bounded expiry`

### T5 — Refactor `normalize_type_for_contract` to sentinel/Result API

- **Source:** `silent-failure-hunter.md` Entries 2 + 5 + cross-site finding.
- **Defect:** `normalize_type_for_contract` raises `TypeError` as a protocol
  signal for "unsupported type". Two call sites
  (`contracts/contract_builder.py:process_first_row` and
  `contracts/contract_propagation.py:_infer_new_field_contract`) catch this
  with R6 exemptions; the bare `except TypeError` could also catch unrelated
  numpy/pandas lazy-import TypeErrors.
- **Proposed change:** Refactor to return a `Result[type, UnsupportedTypeSignal]`
  or `Optional[type]` + `UNSUPPORTED_TYPE_SENTINEL`. Update both call sites to
  branch on the sentinel.
- **Expected effect:** 2 R6 exemptions become deletable; residual TypeError
  risk closed.
- **Priority:** P2, type=feature
- **Title:** `Refactor normalize_type_for_contract — sentinel/Result instead of TypeError as protocol signal`

### T6 — Renew + correct the 2 near-expiry entries

- **Source:** `silent-failure-hunter.md` Recommendation on near-expiry entries.
- **Surface area:** `config/cicd/enforce_tier_model/`
  - `plugins.yaml`: `plugins/infrastructure/clients/http.py:R4:AuditedHTTPClient:_emit_telemetry_after_audit:fp=588df931d0f0d1f1`
    (expires 2026-06-07) — renew with extended TTL or mark permanent (it's the
    reference exemplar for audit-first / telemetry-best-effort pattern).
  - `contracts.yaml`: `contracts/contract_builder.py:R6:ContractBuilder:process_first_row:fp=4d3f0bdcf6fdd56f`
    (expires 2026-06-15) — renew with corrected reason text (proposed full text
    in silent-failure-hunter.md §Entry 2).
- **Priority:** P3, type=task (small, mechanical)
- **Title:** `Tier-model allowlist near-expiry renewal — 2 entries due before 2026-06-15`

### T7 — Growth-rate ratchet (meta-finding)

- **Source:** This audit + parent epic baseline.
- **Finding:** Allowlist grew from 354+93 (Apr 17 baseline) to 578+96 (May 19)
  — +227 entries (+51%) in 32 days, with no burn-down lane. Trajectory points
  to >1000 entries within 3 months.
- **Proposed change:** Either (a) a periodic burn-down lane that fires this
  audit skill quarterly and ratchets totals, or (b) a per-PR gate that requires
  any new allow_hits to come with a deletion of an old one (net-zero or
  net-negative).
- **Priority:** P1, type=feature
- **Title:** `Tier-model allowlist growth-rate ratchet — burn-down lane or per-PR net-negative requirement`

## Pending — to add from refactoring-architect + python-reviewer

(Filled in after those agents complete.)
