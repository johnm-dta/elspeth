# Phase 2 — Audit-Readiness Panel (umbrella)

**Goal:** Land the persistent audit-readiness panel described in
[07-audit-readiness-panel.md](07-audit-readiness-panel.md). The panel
surfaces four-to-six rows summarising whether the user's pipeline will
produce defensible audit evidence at runtime; clicking any row opens
either a narrative Explain detail or a per-row warning detail with a
jump-to-component affordance. Phase 2 also removes the standalone
Validate button from the inspector header — the panel's Validation row
subsumes it.

**Design reference:** [07-audit-readiness-panel.md](07-audit-readiness-panel.md).

**Roadmap reference:** [10-implementation-phasing.md](10-implementation-phasing.md)
(Phase 2 section).

## Split

Phase 2 is split into three implementation plans (the frontend half is
itself split to keep each plan under the 1500-line limit):

| Sub-plan | Scope | Status |
|---|---|---|
| [14a-phase-2a-backend.md](14a-phase-2a-backend.md) | New `audit_readiness` package: response models, closed-list plugin-trust classifier, aggregation service, deterministic narrative builder, two GET routes wired into `app.py`. No frontend changes. | Plan written. |
| [14b-phase-2b-frontend.md](14b-phase-2b-frontend.md) | Frontend foundations — TypeScript types mirroring 14a's Pydantic shapes, two typed GET wrappers in `api/auditReadiness.ts`, a Zustand `auditReadinessStore` with composition-version-keyed cache, and the `AuditReadinessPanel` component (six rows, all-green collapse, auto-fetch). Ships placeholder `ReadinessRowDetail` and `ExplainDialog` files so the panel's tests resolve before 14c lands. | Plan written. |
| [14c-phase-2c-frontend-integration.md](14c-phase-2c-frontend-integration.md) | Replace 14b's placeholders with the real `ReadinessRowDetail` (jump-to-component) and `ExplainDialog` (narrative modal). Mount the panel inside `InspectorPanel.tsx`. Remove the standalone Validate button. Staging smoke. | Plan written. |

All three sub-plans are TDD-shaped, follow `superpowers:writing-plans`,
and contain complete code blocks (no placeholders in the plan text —
14b's runtime placeholders are deliberate scaffolding that 14c replaces).

## Sequencing

Phase 2A **must** be merged before Phase 2C's staging smoke can succeed
(2C's vitest suite stubs the backend; the smoke at the end of 2C
requires the real routes deployed).

Phase 2B Task 1 (frontend types + API client) can be developed in
parallel with Phase 2A's last tasks because the wire shape is fully
specified by Phase 2A's Pydantic models (`AuditReadinessSnapshot`,
`ReadinessRow`, `AuditReadinessExplain` in
[14a-phase-2a-backend.md](14a-phase-2a-backend.md) §Task 1).

Phase 2C **depends on 2B being merged** — 2C replaces files 2B created
(the `ReadinessRowDetail` and `ExplainDialog` placeholders) and modifies
`InspectorPanel.tsx` to mount the panel 2B produced.

## Pre-flight reconnaissance notes

These are the load-bearing findings the sub-plans depend on. They are
captured here so a reviewer can verify them once and refer back from
either plan.

### Trust-tier exposure decision

The roadmap's Phase 7 introduces an explicit `data_trust_tier: ClassVar`
on plugin base classes
([16a-phase-7a-backend.md](16a-phase-7a-backend.md)). Phase 2 does **not**
depend on Phase 7. The audit-readiness panel uses a **closed allowlist of
external-boundary plugin names** in
`src/elspeth/web/audit_readiness/trust.py` instead of a runtime attribute
lookup. Rationale:

- The trust model in CLAUDE.md treats trust as per-data-flow, not as a
  per-plugin static attribute. A `csv_file` source is Tier-3 when reading
  the file (boundary) and Tier-1 once the row enters the pipeline. A
  plugin-attribute lookup would over-claim certainty.
- A closed allowlist makes the surface auditable: adding a plugin that
  crosses an external boundary requires updating `trust.py` in the same
  commit. The test in `tests/unit/web/audit_readiness/test_trust.py`
  fails the build when an allowlisted name doesn't resolve to a
  registered plugin (subset-of-catalog check).
- If Phase 7 lands later and exposes `data_trust_tier` as a ClassVar, the
  classifier in `trust.py` can be replaced with a one-line `getattr`
  call without changing the wire shape. The panel keeps working under
  either implementation.

This decision is documented in
[14a-phase-2a-backend.md](14a-phase-2a-backend.md) §"Plugin-trust
derivation (load-bearing — DO NOT defer)" and is **not** revisited in
Phase 2B.

### Validate button removal — coordination

Phase 2C Task 8 removes the standalone Validate button from
`InspectorPanel.tsx`. The Validation row in the audit-readiness panel
becomes the only user-facing trigger. Two consequences:

- The auto-validate-on-composition-change behaviour that the panel
  provides (Phase 2B Task 4: `useEffect` keyed on
  `compositionState?.version`) replaces the user's explicit click.
- `useExecutionStore.validate()` and `useExecutionStore.validationResult`
  remain — they are still consulted by `handleExecute` to gate the
  Execute button. Phase 2C does **not** remove or rename them.

### Telemetry

`grep -rln "telemetry.emit" src/elspeth/web/` returns no hits as of
2026-05-15. Phase 8 owns the telemetry pass. Phase 2 emits **no**
telemetry; the per-row click handler in Phase 2B simply opens the detail
view. A Phase 8 deferral marker is filed in 14c §Task 5 as a comment in
the click handler.

### Panel mounting site (pre-/post-Phase-3 compatible)

The design spec calls the panel "a persistent panel in the composer's
right rail. Visible at all times during composition." The right rail is
the area currently occupied by `InspectorPanel.tsx`, which has a
tab-strip layout. Phase 3 will reorganise this region away from tabs
([15a1-phase-3a-removals-part-1.md](15a1-phase-3a-removals-part-1.md) / [15a2-phase-3a-removals-part-2.md](15a2-phase-3a-removals-part-2.md)),
but Phase 2 must work *before* Phase 3 ships.

The Phase 2C plan mounts the panel as a fixed section in
`InspectorPanel.tsx` between the header and the tab strip, so it stays
visible regardless of the active tab. Phase 3's IA cleanup is then free
to relocate the panel within the side-rail without touching the
`AuditReadinessPanel` component itself — the mount point moves, the
component does not.

## Trust-tier check (umbrella scope)

The panel reads only data ELSPETH produced (Tier 1):

- Composition state from the session store.
- Validation result from `validate_pipeline`.
- Catalog entries from the catalog service.
- Secret references from `WebSecretService`.
- Settings from `WebSettings`.

The narrative builder writes strings — no external system contact. The
frontend Zustand store caches by `composition_version` — no defensive
coercion of cache contents (Tier 1 data we wrote).

No new trust boundary is introduced.

## Out of scope (entire Phase 2)

- The richer Explain view that Phase 7 layers on top of the catalog
  reshape ([16-phase-7-catalog-reshape.md](16-phase-7-catalog-reshape.md))
  — Phase 2's narrative is the minimum the design spec requires.
- Per-user retention preferences. Phase 2's Retention row reports the
  system default and is `not_applicable` until a future phase adds a
  per-composition retention surface (see
  [14a-phase-2a-backend.md](14a-phase-2a-backend.md) §"Retention row").
- LLM-interpretations row content. The roadmap gates this behind
  Phase 5b (question B2 in
  [11-open-questions.md](11-open-questions.md)). Phase 2 emits the row
  unconditionally with `status = not_applicable` so Phase 5b flips one
  place.
- Telemetry on panel-row clicks (Phase 8).
- New plugin attributes for trust tier (Phase 7 if it lands; Phase 2 is
  decoupled by closed allowlist — see "Trust-tier exposure decision"
  above).

## Risks (umbrella)

| Risk | Mitigation |
|---|---|
| Panel disagrees with what actually happens at runtime | Both must compose the same validation route. The aggregator in Phase 2A calls `ExecutionService.validate_pipeline()` directly; the panel never re-implements validation. |
| Closed allowlist drifts as plugins are renamed | Subset-of-catalog test fails the build. Allowlists are short by design (Phase 2A §Task 2). |
| Removing the Validate button strands users mid-keystroke | Phase 2B Task 4 wires auto-validate to `compositionState.version`; the Validation row reflects status without an explicit click. Phase 2C Task 8 verifies the keyboard navigation across the inspector tabs still works after the button is gone. |
| Phase 3 layout change breaks the panel | Phase 2C mounts the component inside `InspectorPanel.tsx` as a fixed section above the tab strip — pure presentation. Phase 3 relocates the mount point only. |
| Recomputing on every composition change is expensive | Phase 2B caches by `(session_id, composition_version)` in the Zustand store; only fires when the version changes. The aggregator calls existing routes — same order as the standalone Validate button it replaces. |
| Per-row "jump to component" requires `component_ids` to be accurate | Phase 2A populates `component_ids` from `ValidationResult.errors[*].component_id` and the identity-advisory check string. Phase 2C's click handler resolves these via `compositionState.nodes`, calls `selectNode`, and dispatches `SWITCH_TAB_EVENT` for `"spec"` — the same tab-switch pattern `InspectorPanel.tsx` already listens for. |

## Rollout coordination

Phase 2A introduces two new required fields without Python defaults
(per CLAUDE.md No-Legacy — no compat-shim defaults permitted):

- `ValidationError.error_code: str`
- `ValidationCheck.affected_nodes: tuple[str, ...]`

Every call site that constructs a `ValidationCheck` or `ValidationError`
must be updated in the same commit as the dataclass change. The call-site
sweep is owned by [14a-phase-2a-backend.md](14a-phase-2a-backend.md)
§Task 1 Step 3.

### Deployment ordering

Pick exactly one:

**Option A (recommended):** Backend (Phase 2A) ships first. Frontend
(Phase 2B) ships in a subsequent deploy once Phase 2A is confirmed stable
on staging. Frontend types may temporarily be wider than what the backend
returns — this is acceptable for a short deploy window.

**Option B:** Backend (Phase 2A) and frontend (Phase 2B) co-ship in a
single deploy. Safer for the schema co-update because the API contract
widens and narrows in lockstep.

**Forbidden:** Frontend before backend. Phase 2B's TypeScript types
reference `error_code` and `affected_nodes` — consuming these fields from
a backend that has not yet deployed Phase 2A will produce `undefined`
values at every call site that reads them.

### DB wipe requirement

Per project DB migration policy (`project_db_migration_policy.md` —
"no Alembic, no schema-version probes, no migration scripts; operator
deletes the old sessions/audit DB"):

- The operator **must** delete the sessions DB on staging before
  deploying Phase 2A.
- `src/elspeth/web/guided/audit.py:249` persists
  `dict(validation_result)` — records written before Phase 2A do not
  contain `error_code` or `affected_nodes`. Attempting to deserialize
  those records into the updated dataclasses will raise `TypeError` at
  every construction site.
- There is no migration path. There is no fallback. Delete the DB.

### Staging gate (before merging Phase 2A to main)

1. Confirm the operator has deleted the staging sessions DB.
2. Confirm `mypy` and `ruff` pass on the full call-site sweep (see 14a
   Task 1 Step 3 — coverage must include `execution/service.py`,
   `composer/service.py`, `guided/audit.py`, and the affected test files).
3. Confirm at least one `curl` against
   `/api/sessions/{id}/audit-readiness` returns a payload containing
   `composition_version`, `checked_at`, and the six canonical readiness rows.

### Cross-references

- [14a-phase-2a-backend.md](14a-phase-2a-backend.md) §Task 1 Step 3 —
  owns the call-site sweep and git-add enumeration.
- [14b-phase-2b-frontend.md](14b-phase-2b-frontend.md) — owns the
  frontend type widening; must not ship before Phase 2A (see ordering
  above).
- Project memory `project_db_migration_policy.md` — authoritative source
  for the DB-wipe-vs-migrate decision.

## Review history

- **2026-05-15** — Sub-plans 14a/14b/14c migrated their route descriptions from POST to GET in response to the audit-readiness read-only access pattern; the umbrella status table reflects the post-change shape. Cycle-1 finding closed.

## Memory references

- `project_composer_personas` — informs the panel-vocabulary mapping;
  Linda's vocabulary is the load-bearing constraint on row labels.
- `feedback_default_is_fix_not_ticket` — fix-in-session rule applies; the
  per-row warning detail surfaces fixes inline rather than logging
  follow-up tickets.
- `feedback_no_calendar_shipping_commitments` — no calendar SLAs.
- `feedback_repeated_out_of_scope_is_underscoping.md` — if the
  sub-agent finds itself filing "Phase 2 should also do X" observations
  on three tasks, surface to the operator: that's a scope signal, not
  three independent observations.
