# Test-audit burndown status — epic `elspeth-b9a3c59654`

- **Audit date:** 2026-05-22
- **Branch:** RC5.2 @ `e4ecfd8d2`
- **Epic title:** Test audit — folder risk sweep
- **Children count:** 145 (confirmed via `list_issues parent_issue_id=elspeth-b9a3c59654 no_limit=true`)
- **Session context:** Operator paused this burn-down during a multi-token engine
  upgrade. Earlier this session we landed 5 fix-branches (`fix/test-audit-*`)
  that were carrying real test-gap work. Question: how many of the 145 are
  *actually* still real work versus already-shipped-but-ticket-open.

## Headline

The 145 children divide into **two distinct families** with very different
shapes:

| Family | Count | What it is |
| --- | --- | --- |
| **Folder sweep** (review-and-comment task) | 143 | "Audit `<folder>` — `<risk>`-risk test suite sweep". Bulk-imported 2026-05-20 by a heuristic scoring pass. Completion = inspect the folder, post a comment with confirmed rating + concrete findings + recommended remediation. |
| **Test-gap** (concrete regression to add) | 2 | Output of prior sweep work. Has `test-gap` label, narrative title, defined acceptance criteria. Both already closed. |

**The 143 sweep tickets are NOT bug-fix tickets.** They are *review tasks*:
the deliverable is a comment with findings, not a code change. So the
"shipped-but-not-closed" pattern from the `elspeth-de91358c30` epic and the
Phase 5 epic does **not** transfer — there is no code change that can
"silently close" a sweep ticket. Either the comment was written or it wasn't.
**Zero comments have been added to any of the 143** (all show
`updated_at == created_at`).

## Bucket counts

### Overall status

| status_category | status | count |
| --- | --- | --- |
| open | open | 143 |
| done | closed | 2 |

| priority | count | family |
| --- | --- | --- |
| P1 | 30 | all sweeps, all `test-risk:high` |
| P2 | 35 | 33 sweeps `test-risk:medium` + 2 closed `test-gap` |
| P3 | 80 | all sweeps, all `test-risk:low` |

### Sweep risk × status

| risk | open | closed |
| --- | --- | --- |
| high | 30 | 0 |
| medium | 33 | 0 |
| low | 80 | 0 |

### Labels

| label | count |
| --- | --- |
| `test-audit` | 145 |
| `test-suite-sweep` | 143 |
| `test-risk:low` | 80 |
| `test-risk:medium` | 33 |
| `test-risk:high` | 30 |
| `test-audit-20-05` | 6 |
| `test-gap` | 2 |

### Assignment

- Assigned + with live claim: **2** (both already closed — `codex` on the two `test-gap` tickets)
- Unassigned, never claimed: **143**
- Stale-claim count: **0** (no orphaned claims to clean)

## Pattern detection

- **Bulk-imported same day.** All 145 created on 2026-05-20 between 04:52 and
  05:06 UTC. Title generator is formulaic (`Audit <folder> — <risk>-risk test
  suite sweep`) with a fixed description template ("Initial test-risk rating /
  Mapped priority / Folder / Direct active test files / Approximate LOC /
  Heuristic score / Primary signals / Review scope / Expected output").
- **Heuristic-driven.** Each description carries `Heuristic score: N` and
  `Primary signals: ...`; this came from a programmatic risk scanner that
  walked every test-bearing folder and emitted one ticket per folder.
- **Priority maps 1:1 to risk band.** `high → P1`, `medium → P2`, `low → P3`.
  No human prioritisation post-import.
- **Risk-band shape is plausible** (30/33/80 high/med/low) — long tail of low
  is expected for a folder sweep across a large monorepo.

### Spot-check sample (high-confidence, evidence-based)

Six tickets read in full. **None** had been moved off `triage`/`open` since
creation, **none** had any comments, **none** had assignees. Folder existence
verified by `os.path.isdir`.

| Ticket | Folder | Status | Description still accurate? | Confidence |
| --- | --- | --- | --- | --- |
| `elspeth-3a50248223` | `tests/integration/pipeline` | open | folder exists, 16 files unchanged since 2026-05-20 | high |
| `elspeth-7460c257bf` | `tests/integration/web` | open | folder exists; recent commits touched 4 files inside it (`test_catalog_discovery.py`, `test_composer_tools.py`, `test_preferences_routes.py`, plus 1 audit dir) but none constitutes the folder-wide review the ticket asks for | high |
| `elspeth-401e1cd7cc` | `tests/unit/core/landscape/repository_integration` | open | folder exists, 25 files; many recent edits (recorder_* files) but again none is "post folder-level review comment" | high |
| `elspeth-95da17e14f` | `src/elspeth/web/frontend/src/stores` | open | folder exists; this is *frontend production code*, mis-titled as a test sweep — see Anomalies | high |
| `elspeth-fbfb46f72a` | `tests/unit/cli` | open | folder exists | high |
| `elspeth-0775a71770` | (n/a — `test-gap` ticket) | **closed** | `close_reason = "Regression coverage added and verified in commit b88a2c112."` confirms the gap was actually addressed | high |

Conclusion: **no shipped-but-not-closed pattern in this epic.** The 143
sweeps are genuinely untouched review tasks. The completion gesture (post a
comment) has not happened for any of them.

## Recent-commit cross-reference (RC5.2, since 2026-05-19)

Issue IDs found in commit-message bodies for the test-affecting range:

| ID | In epic? |
| --- | --- |
| `elspeth-879f6de6bd` | no (unrelated probe-failure ticket) |
| `elspeth-be398f0bcb` | no |

**No commit anywhere in recent RC5.2 history references a child of
`elspeth-b9a3c59654` by ID.** The 5 fix-branches landed today
(`fix/test-audit-{landscape-exporter-isolation,lineage-persisted,
payload-corruption,transcript-access,telemetry-exporter-failures}`) appear by
*branch-name* to be `test-gap` work spawned out of prior sweeps, but only 2
matching `test-gap` children are present in this epic (the 2 closed). The
other 3 branches' parent tickets live elsewhere (likely the earlier
`elspeth-de91358c30` test-audit epic) — verifiable by querying that epic's
children separately.

## File-area distribution of the 143 sweeps

Top-3 path-component breakdown of the 143 open sweep folders:

| count | folder prefix |
| --- | --- |
| 31 | `src/elspeth/web/frontend/...` |
| 15 | `tests/unit/plugins` |
| 13 | `tests/unit/web` |
| 8 | `tests/unit/core` |
| 7 | `tests/property/plugins` |
| 5 | `tests/unit/contracts` |
| 5 | `tests/integration/plugins` |
| 4 | `tests/integration/web` |
| 4 | `tests/performance/*` |
| 3 | `tests/unit/scripts` |
| 2 | `tests/integration/pipeline` |
| 2 | `tests/unit/engine` |
| 2 | `tests/unit/evals` |
| 2 | `tests/unit/mcp` |
| 2 | `tests/unit/telemetry` |
| 2 | `elspeth-lints/src/elspeth_lints` |
| rest (1 each) | scattered |

**Real test-folder review surface clusters in:** `tests/unit/plugins` (15),
`tests/unit/web` (13), `tests/unit/core` (8). Together with their `property`
and `integration` siblings, the plugins family adds 27 sweeps and the web
family adds 17 sweeps. If the operator wants to bite off one cluster, the
plugins family is the largest cohesive chunk.

## Anomalies

1. **31 sweep tickets target `src/elspeth/web/frontend/...` — production
   code, not tests.** The heuristic scanner appears to have included the
   frontend's `src/test/`, `tests/e2e/`, plus most of its `components/`,
   `stores/`, `hooks/`, etc. as test-bearing. These tickets are titled "test
   suite sweep" but their target folders are application source. Whoever
   executes them has to decide whether the title is wrong (they're really
   *frontend code reviews*) or the scope is wrong (they should be reduced to
   the subset that actually contains tests, e.g.
   `src/elspeth/web/frontend/src/test/`,
   `src/elspeth/web/frontend/tests/e2e/`). **Recommend batch-closing the 29
   that aren't actually test folders** and keeping only the 2 genuine
   frontend test folders (`src/test`, `src/test/a11y`, `tests/e2e`,
   `tests/e2e/helpers`, `tests/e2e/page-objects`, `tests/e2e/setup` — 6 of
   the 31 are arguably test surfaces).

2. **The 2 closed `test-gap` children are stylistically different** from the
   143 sweeps (narrative title, detailed Evidence + Risk + Acceptance
   criteria sections). They were filed at 2026-05-20T05:06 and
   2026-05-20T05:13 — *after* the sweep batch — and were closed within hours
   by `codex` with `close_reason` populated. The natural reading: somebody
   ran a sweep on `tests/unit/core/landscape/...` informally, found those 2
   gaps, filed them as `test-gap` children of this epic, and closed them.
   But the parent sweep ticket
   (`elspeth-401e1cd7cc tests/unit/core/landscape/repository_integration`)
   was never marked complete despite the gap-finding work that came out of
   it.

3. **Five fix-branches landed today reference IDs outside this epic.** The
   `fix/test-audit-*` branches we landed do `test-gap` work but their parent
   `test-gap` tickets are not children of `elspeth-b9a3c59654`. They likely
   belong to the earlier `elspeth-de91358c30` epic. Worth confirming in a
   follow-up to make sure those tickets get closed there.

## Categorised remaining-work surface

| category | count | action |
| --- | --- | --- |
| **Likely shipped, just not closed** | 0 | None — sweeps are review tasks, no code change closes them |
| **Branch in flight** | 0 | No work-in-progress branches map to any of the 143 |
| **Genuinely open, no work started** | 143 | The 143 sweeps; broken down below |
| **Stale claim** | 0 | None |
| **Anomalous / probably wrong-shape** | ~29 of 143 | The `src/elspeth/web/frontend/...` non-test folders |
| **Closed already** | 2 | The 2 `test-gap` tickets |

Within the 143 unstarted sweeps:

| sub-bucket | count |
| --- | --- |
| HIGH-risk genuine test folders (P1) | 30 |
| MEDIUM-risk genuine test folders (P2, excluding frontend code) | ~24 of 33 |
| LOW-risk genuine test folders (P3, excluding frontend code) | ~60 of 80 |
| Frontend-code mis-classification (P2/P3) | ~29 |

## Recommendations

1. **Batch-close the ~29 frontend-code mis-classification tickets.** They are
   titled "test suite sweep" but target frontend application code, not test
   folders. Closing them with a `close_reason` referencing this audit note
   removes ~20 % of the burndown immediately without losing real work. (List
   in the report tables; the 6 that genuinely contain tests should stay
   open.)
2. **Pick a single cluster, not a flat ordering.** The 30 P1 / HIGH sweeps
   are an obvious next bite, but if any cluster matters more right now (e.g.
   `tests/unit/plugins` at 15, or the landscape-adjacent ones we keep
   finding gaps in), pre-claim those as a sub-epic and discharge the rest
   later.
3. **Lower the bar for "done" on a sweep.** The template demands "add a
   comment with confirmed rating, concrete findings, recommended remediation,
   and folders that should be split". For LOW-risk folders (80 of 143), this
   is overkill — a one-line "reviewed, no findings worth filing" comment
   should suffice. Without that explicit affordance, agents will overshoot on
   trivial folders and undershoot on the rest.
4. **Confirm the 5 fix-branches' parent tickets get closed in their actual
   epic** (likely `elspeth-de91358c30`). The branch names imply
   gap-remediation completed today; if those parent tickets are still open
   over there, that's the easier win and should be cleaned up first.
5. **Treat this epic as paused-and-stale.** Created 2 days ago, 0 of 143
   progressed. The multi-token engine work is competing for the same agent
   attention. Either schedule a dedicated sweep day or set this epic
   explicitly to a deferred / parked state so it stops showing up in
   ready-work queries as 143 P1/P2/P3 items demanding attention.

## SME confidence summary

- Status counts, label counts, priority counts: **high confidence** — pulled
  directly from `list_issues`.
- Spot-check findings (6 tickets read in full + folder existence verified +
  git log cross-referenced): **high confidence**.
- "No shipped-but-not-closed pattern": **high confidence** — sweep tickets
  require a comment for closure, and zero comments have been added (all 143
  have `updated_at == created_at`).
- "Frontend mis-classification" assessment: **medium-to-high confidence** —
  31 sweep tickets target `src/elspeth/web/frontend/...` paths; spot-checked
  one (`src/elspeth/web/frontend/src/stores`) and confirmed it's a
  Zustand-style store folder, not a test folder. Did not verify all 31
  individually; the safe move is to read each before batch-closing.
- "Parent tickets of the 5 landed branches live in
  `elspeth-de91358c30`": **medium confidence** — naming-based inference only;
  not verified against that epic's children.
