# Frontend-mis-classified sweep ticket batch closure — 2026-05-23

**Source audit:** `notes/test-audit-burndown-status-2026-05-22.md`
**Epic:** `elspeth-b9a3c59654` — test-audit burndown sweep
**Open children before closure:** 143
**Open children after closure:** 118
**Closed in this batch:** 25
**Kept open as genuine frontend test surfaces:** 6

## Rationale

A heuristic scanner bulk-imported sweep tickets on 2026-05-20 against every
folder under the repo. 31 of those targeted paths under
`src/elspeth/web/frontend/`. The audit at
`notes/test-audit-burndown-status-2026-05-22.md` established that the
overwhelming majority of those paths are application source (Zustand stores,
React components, hooks, contexts, types, utils, styles, etc.) — not test
folders. The scanner had mis-classified them.

This closure batch removes the 25 mis-classifications. The 6 frontend paths
that are arguably genuine test surfaces remain open for the sweep reviewer.

## Closed tickets (25)

| issue_id | Folder |
| --- | --- |
| elspeth-333cca6eb6 | `src/elspeth/web/frontend/src/api` |
| elspeth-2ba4406486 | `src/elspeth/web/frontend/src/components/chat` |
| elspeth-c9737ed944 | `src/elspeth/web/frontend/src/components/chat/guided` |
| elspeth-4d6f873fde | `src/elspeth/web/frontend/src/components/common` |
| elspeth-1f10364836 | `src/elspeth/web/frontend/src/components/inspector` |
| elspeth-57403ebd7e | `src/elspeth/web/frontend/src/hooks` |
| elspeth-95da17e14f | `src/elspeth/web/frontend/src/stores` |
| elspeth-3078e42e3c | `src/elspeth/web/frontend/src` |
| elspeth-97d4c80976 | `src/elspeth/web/frontend/src/components/audit` |
| elspeth-8362d47c74 | `src/elspeth/web/frontend/src/components/auth` |
| elspeth-33926ecc0e | `src/elspeth/web/frontend/src/components/blobs` |
| elspeth-bc871382ec | `src/elspeth/web/frontend/src/components/catalog` |
| elspeth-bcaf250b87 | `src/elspeth/web/frontend/src/components/composer` |
| elspeth-d60f20f08b | `src/elspeth/web/frontend/src/components/execution` |
| elspeth-80d6ab9a70 | `src/elspeth/web/frontend/src/components/header` |
| elspeth-23cfca3ff6 | `src/elspeth/web/frontend/src/components/recovery` |
| elspeth-b7de27ed44 | `src/elspeth/web/frontend/src/components/sessions` |
| elspeth-dec20e20b3 | `src/elspeth/web/frontend/src/components/settings` |
| elspeth-ddf29311c8 | `src/elspeth/web/frontend/src/components/shared` |
| elspeth-f26aca78f6 | `src/elspeth/web/frontend/src/components/sidebar` |
| elspeth-af07e71407 | `src/elspeth/web/frontend/src/components/tutorial` |
| elspeth-4a2d6bf958 | `src/elspeth/web/frontend/src/contexts` |
| elspeth-738aa7269d | `src/elspeth/web/frontend/src/styles` |
| elspeth-4b2d76d33f | `src/elspeth/web/frontend/src/types` |
| elspeth-6bdfdc814e | `src/elspeth/web/frontend/src/utils` |

## Kept open as genuine frontend test surfaces (6)

| issue_id | Folder |
| --- | --- |
| elspeth-fa8c35ee01 | `src/elspeth/web/frontend/src/test` |
| elspeth-8ee8756e08 | `src/elspeth/web/frontend/src/test/a11y` |
| elspeth-a2e66acbad | `src/elspeth/web/frontend/tests/e2e` |
| elspeth-45c997ecb5 | `src/elspeth/web/frontend/tests/e2e/helpers` |
| elspeth-c4d863238d | `src/elspeth/web/frontend/tests/e2e/page-objects` |
| elspeth-5d8fe59ea6 | `src/elspeth/web/frontend/tests/e2e/setup` |

## batch_close summary

- **Succeeded:** 25
- **Failed:** 0
- **Actor:** `claude`
- **Force:** `true` (per task spec; ensures close transition succeeds regardless of workflow shape)

## Spot-check confirmations

Three randomly-selected closed IDs were re-fetched via `get_issue` and
confirmed `status_category=done`:

| issue_id | status | status_category | closed_at |
| --- | --- | --- | --- |
| elspeth-95da17e14f | closed | done | 2026-05-23T00:47:44.517485+00:00 |
| elspeth-bcaf250b87 | closed | done | 2026-05-23T00:47:44.528514+00:00 |
| elspeth-6bdfdc814e | closed | done | 2026-05-23T00:47:44.549433+00:00 |

The `close_reason` field on each closed issue carries the full audit
justification recorded above.

## Burn-down state

- Epic `elspeth-b9a3c59654` open children **before:** 143
- Epic `elspeth-b9a3c59654` open children **after:** 118
