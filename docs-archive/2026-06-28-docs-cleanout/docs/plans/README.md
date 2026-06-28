# ELSPETH Plans and Design Notes

This directory is a curated set of design and implementation plans that are
still useful in-tree. It is **not** the system of record for day-to-day work:
active execution tracking lives in Filigree.

Most completed, superseded, or abandoned plans are removed from the working tree
and preserved in git history. A small number of completed plans remain because
code, ADRs, or later plans still cite them.

## What Lives Here

### Architecture and design references

- `ARCH-15-design.md` — per-branch fork transform architecture
- `2026-02-26-t17-plugincontext-protocol-split-design.md` — PluginContext split design retained as architectural reference (cited by `src/elspeth/contracts/contexts.py`)
- `2026-03-30-transport-primitive-composition-spec.md` — composition primitives design reference
- `2026-03-30-primitive-plugin-pack.md` — primitive plugin pack planning/design note
- `2026-04-20-h2-amendment-design.md` — declaration-trust framework H2 amendment retained as cited by `docs/architecture/adr/010-declaration-trust-framework.md`

### Implementation plans still relevant for follow-up work

- `2026-02-01-nodeinfo-typed-config.md`
- `2026-02-02-whitelist-reduction.md`
- `2026-02-13-contract-propagation-complex-fields.md`
- `2026-02-25-llm-plugin-consolidation.md`
- `2026-04-20-phase-2b-declaration-trust.md` — Phase 2B execution plan (filigree epic `elspeth-a3ac5d88c6` open) with reviewer companions in `reviews/`
- `2026-04-27-plugin-declared-transform-semantics.md` — original v1 design retained while v2 lands

### Review companions and audit artifacts

- `*.review.json` files are plan-review outputs that belong next to the plan they assess
- `reviews/` contains panel review artifacts that travel with their parent plan

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
