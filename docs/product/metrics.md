# Metrics — ELSPETH                     Last read: 2026-06-28

> Seeded at bootstrap. Targets are BASELINE → TARGET placeholders for the owner
> to set real numbers against — every one is falsifiable by construction, but the
> numbers below are not yet instrumented. The kill/keep logic behind these lives
> in product-metrics-and-experimentation.md.

## North-star

The enduring product value of a high-assurance substrate is not features
shipped — it is that wrong outputs are caught before harm and every run is
explainable. The north-star measures exactly that.

| Metric | Target (falsifiable) | Current | Read on | Trend |
|--------|----------------------|---------|---------|-------|
| Run assurance completeness — share of pipeline runs (both surfaces) producing a complete, replayable Landscape audit record with **zero** silent trust-tier coercions | ≥ TARGET% by <date> | BASELINE (uninstrumented) | 2026-06-14 | — |

## Input metrics (the levers that move the north-star)

| Metric | Target | Current | Read on |
|--------|--------|---------|---------|
| Web-GA readiness — open P0/P1 findings across the five Web-hardening clusters | → 0 by <date> | see tracker (5 epics open) | 2026-06-14 |
| Composer authoring success (activation) — % of composer sessions producing a validated, executable pipeline | ≥ TARGET% | BASELINE (uninstrumented) | 2026-06-14 |

## Guardrails (must NOT degrade)

| Metric | Floor / ceiling | Current | Read on |
|--------|-----------------|---------|---------|
| Silent Tier-3 coercions (the C1 bug class) | = 0 (floor) | 0 in-branch (C1 fixed 5190bb016) | 2026-06-14 |
| Composer-accepted pipelines that fail runtime validation (false-accept rate) | ≤ TARGET% (ceiling) | BASELINE (uninstrumented) | 2026-06-14 |
| Test battery — pass count with 0 hard failures | ≥ 5507 pass / 0 fail | ~5507 / 0 (1 pre-existing P3 fingerprint) | 2026-06-14 |
| Trust-tier red-gate — never bypassed or blessed without provenance | qualitative floor: green only on signed state | **2026-06-28: STRENGTHENED — operator-only HMAC custody now structurally enforced (stage→sign-bundle seam; MCP staging fails closed on key presence; CI-never-signs standing test). Pending live-judge e2e. (PDR-0002, proposed)** | 2026-06-28 |

> 2026-06-28 reading note: elspeth-lints unit suite green (1419 passed / 83 py-version skips) + 70 new judge/signature tests, all five security invariants non-vacuously pinned. The full ~5507 test battery was **not** re-run this session, so the test-battery guardrail above keeps its 2026-06-14 reading. No reversal trigger crossed.
