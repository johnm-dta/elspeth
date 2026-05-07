# ELSPETH Plans and Design Notes

This directory is a curated set of design and implementation plans that are
still useful in-tree. It is **not** the system of record for day-to-day work:
active execution tracking lives in Filigree.

Most completed, superseded, or abandoned plans are removed from the working tree
and preserved in git history. A small number of completed plans remain because
code, ADRs, or later plans still cite them.

## What Lives Here

### Release and program framing

- `RC4-initiation.md` — RC4 scoping and sequencing snapshot retained for release-history context

### Architecture and design references

- `ARCH-15-design.md` — per-branch fork transform architecture
- `2026-02-26-t17-plugincontext-protocol-split-design.md` — PluginContext split design retained as architectural reference
- `2026-03-30-transport-primitive-composition-spec.md` — composition primitives design reference
- `2026-03-30-primitive-plugin-pack.md` — primitive plugin pack planning/design note

### Implementation plans still relevant for follow-up work

- `2026-02-01-nodeinfo-typed-config.md`
- `2026-02-02-whitelist-reduction.md`
- `2026-02-13-contract-propagation-complex-fields.md`
- `2026-02-25-llm-plugin-consolidation.md`
- `2026-03-09-display-header-mixin-extraction.md`
- `2026-03-09-purge-query-deduplication.md`

### Review companions and audit artifacts

- `*.review.json` files are plan-review outputs that belong next to the plan they assess
- `05-quality-assessment-t10-llm-consolidation.md` is an architecture quality review, not an executable plan
- `2026-02-13-documentation-audit-report.md` is a documentation audit report retained as reference material

## Historical Plans

Large batches of completed and superseded plans have already been removed from
this directory and kept in git history. To recover one:

```bash
# Find a specific deleted plan by name
git log --all --diff-filter=D -- "docs/plans/*field-collision*"

# Restore a deleted plan
git show <commit>:docs/plans/<filename>.md
```

Additional assistant-driven plans and specs live under `docs/superpowers/`.
