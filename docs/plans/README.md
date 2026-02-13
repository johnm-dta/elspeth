# ELSPETH Implementation Plans

This directory tracks active implementation plans. Completed, superseded, and cancelled plans have been removed from the working tree (preserved in git history).

## Active Plans

| Plan | Description | Status |
|------|-------------|--------|
| `RC3-remediation.md` | RC-3 remaining remediation (9 items from original 75+) | Active |
| `RC3-doc-updates.md` | RC-3 documentation and project layout remediation | Complete |
| `ARCH-15-design.md` | Per-branch fork transforms architecture | Active |
| `2026-02-02-whitelist-reduction.md` | Tier model whitelist reduction | In progress |
| `2026-02-13-contract-propagation-complex-fields.md` | Preserve dict/list fields in propagated contracts | Queued |
| `2026-02-13-documentation-audit-report.md` | Documentation freshness and cross-reference audit | Reference |
| `2026-02-01-nodeinfo-typed-config.md` | Type NodeInfo.config with discriminated union | Queued |

## Historical Plans (Git History)

148 completed plans and 18 superseded plans were removed from this directory on 2026-02-14. To recover any historical plan:

```bash
# Find a specific plan by name
git log --all --diff-filter=D -- "docs/plans/completed/*field-normalization*"

# Restore a deleted plan
git show <commit>:docs/plans/completed/<filename>.md
```

## Plan Lifecycle

```
Created → In Progress → Completed (deleted from tree, kept in git)
                      → Superseded (deleted from tree, kept in git)
```
