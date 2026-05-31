# Filigree Label Schema

Authoritative registry: [`subsystems.yaml`](./subsystems.yaml). This document explains the rules.

## Diagnosis: why a schema

As of 2026-05-23, the filigree label space had **five competing canonicals for "subsystem"** (`component:*`, `subsystem:*`, `subsystem/*` (slash form), `web:*`, plus bare labels like `composer` and `frontend`), and `cluster:*` was carrying three orthogonal axes (subsystem, defect kind, review program). 67% of open bugs derived no clean subsystem under any single namespace.

The fix is **namespace-per-axis**, not better cluster names.

## Axes

A label expresses exactly one of these orthogonal questions:

| Axis | Namespace | Cardinality per issue | Rule |
|---|---|---|---|
| Subsystem (where in the code) | `subsys:*` | ≥1 at close-time | closed enumeration |
| Architectural layer | `layer:*` | 0–1 (auto-derivable from `subsys:*`) | closed enumeration |
| Defect kind | `kind:*` | 0–N | governed-open (PR-gated additions) |
| Trust tier | `tier:*` | 0–1 | closed enumeration |
| Provenance | `source:*` | 0–N | governed-open |
| Release target | `release:*` | 0–1 | naming pattern |
| Review program | `program:*` | 0–N | governed-open |
| Epic grouping | `cluster:*` | 0–1 (max one) | open, points to parent slug |
| Plugin instance | `plugin:*` | 0–N | open |

**Priority is a field, not a label.** Bare `P0`/`P1`/`P2`/`P3` labels are deprecated; use filigree's numeric priority.

## Subsystem rules

- **Closed enumeration.** Every `subsys:*` value MUST appear in `subsystems.yaml` with a `path:` mapping to a real directory (or `path: cross-cutting` for documented cross-cutting surfaces).
- **Slash-separated sub-areas** (`subsys:web/composer`, `subsys:engine/orchestrator`) so prefix queries work: `mcp__filigree__list_issues(label_prefix="subsys:web/")`.
- **New subsystems require a PR** adding them to `subsystems.yaml` with description, layer assignment, and code path.
- **Multiple `subsys:*` labels are allowed** for genuine cross-cutting issues. An issue with `subsys:composer` + `subsys:web/frontend` + `subsys:composer/mcp` is fine if the bug actually spans all three.

## Layer rules

- `layer:*` is auto-derivable from `subsys:*` via the `layer:` field in `subsystems.yaml`. CI can populate it.
- Bugs that **violate** the layer model (an upward import, a TYPE_CHECKING-only escape, a runtime coupling) carry `kind:layer-violation` plus the `subsys:*` of the violating module, plus the `subsys:*` of what it incorrectly depends on. The `layer:*` of either end is **not** an alternative.

## Kind rules

- `kind:*` is the **defect** — what's wrong, not where. A bug is `kind:silent-failure` + `subsys:engine/processor`, not `cluster:silent-failure`.
- Adding a new `kind:*` value requires a one-line PR to `subsystems.yaml`. The list is semi-closed: easy to grow with justification, hard to grow without.
- Multiple kinds allowed if the bug genuinely spans them (e.g. `kind:silent-failure` + `kind:tier-model`).

## Provenance rules

- `source:*` is **how the bug was found**, not what code it lives in. Provenance is durable evidence (we can defend a fix decision by pointing at the source).
- The bare `from-*` family (`from-observation`, `from-review`, etc.) is deprecated in favour of `source:*`. Migration table in `subsystems.yaml`.

## Observation promotion auto-tagging

When an observation is promoted to an issue:
1. If `file_path` is set, derive `subsys:*` from the path via the `path:` mapping in `subsystems.yaml` (longest-prefix match).
2. Always set `source:observation`.
3. Inherit any `kind:*` / `tier:*` labels from the observation's body if mechanically extractable.

This closes the 169-bug `from-observation` uncategorised hole identified in the 2026-05-23 audit.

## Constraint enforcement

| Constraint | Where enforced | Failure mode |
|---|---|---|
| `subsys:*` required to close | CI hook on close transition | Close blocked until a `subsys:*` label is added |
| At most one `cluster:*` | Pre-write hook (or periodic audit) | Reject the write; explain |
| `layer:*` must agree with `subsys:*` mapping | CI lint | Auto-correct or reject |
| Unknown `subsys:*` value | CI lint against `subsystems.yaml` | Reject |
| Deprecated namespace used (`component:*`, `subsystem/*` slash, `web:*`, bare `from-*`) | CI lint, with migration suggestion | Warning until cutover date, then error |

## Migration plan

Phased, lowest-risk first:

**Phase 1 — kind:\* extraction** (2026-05-24 target)
- Run `migrations[]` entries where `from: cluster:<defect>` → `to: kind:<defect>`.
- Affects ~340 issues. Non-overlapping with subsystem: `cluster:audit-integrity` becomes `kind:audit-integrity` outright.
- Validates the migration machinery before touching subsystem.

**Phase 2 — source:\* consolidation** (2026-05-25 target)
- Bare `from-*` family → `source:*`.
- `audit:source-risk-*` → `source:source-risk-sweep`.
- ~700 label updates.

**Phase 3 — subsys:\* migration** (2026-05-27 target)
- `component:*` → `subsys:*` per direct mapping.
- `subsystem/*` (slash) and `subsystem:*` (colon) → `subsys:*`.
- `web:*` → `subsys:web/*`.
- `cluster:<subsystem-shaped>` → `subsys:*` (plus any `add:` extras from the migration table).
- Bare subsystem aliases (`composer`, `frontend`, `web`, etc.) → `subsys:*`.
- The mass migration. ~1500 label updates.

**Phase 4 — tier:\*, release:\*, program:\* normalisation** (2026-05-28 target)
- Bare `tier-N-*` → `tier:*`.
- Case-drift releases → lowercase canonical.
- Bare sweep labels → `program:*`.

**Phase 5 — priority field migration** (2026-05-29 target)
- Bare `P0`–`P3` → numeric priority field.
- Then delete the labels.

**Phase 6 — CI gates on** (2026-05-30 target)
- Close-time `subsys:*` requirement.
- Unknown-value rejection.
- Deprecated-namespace warning escalates to error.

Each phase is one filigree script call against the migration table, with a dry-run first and a JSON diff written to `.filigree/backups/` before any write.

## Periodic audit

Every 4–8 weeks (mirroring `cicd-allowlist-audit`):

1. Tally distinct values per namespace.
2. Flag growth in `kind:*` and `subsys:*` against last snapshot.
3. Find new typos / case drift / aliases-that-should-be-merged.
4. Verify every `subsys:*` value still maps to an extant directory (rename detection).
5. Sample uncategorised issues and surface them.

Drift is the dominant failure mode for label schemas; this audit is what keeps it shippable.
