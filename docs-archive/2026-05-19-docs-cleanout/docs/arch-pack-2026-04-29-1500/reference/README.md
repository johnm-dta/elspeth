# Reference Data

Deterministic artefacts and indexes that other documents in this pack
cite. Treat the contents of this directory as **immutable evidence** —
the L3 import oracle is byte-stable across re-runs (with `--no-timestamp`)
and the ADR index is derived from the project's own ADR directory.

| File | Purpose |
|------|---------|
| [`l3-import-graph.json`](l3-import-graph.json) | The deterministic L3 import oracle in JSON. 33 nodes, 79 edges, 5 SCCs at the snapshot HEAD. Schema v1.0. |
| [`l3-import-graph.mmd`](l3-import-graph.mmd) | Same graph rendered as a Mermaid flowchart for inline diagrams. |
| [`l3-import-graph.dot`](l3-import-graph.dot) | Same graph rendered in Graphviz DOT format for layout tools (`dot`, `circo`, etc.). |
| [`tier-model-oracle.txt`](tier-model-oracle.txt) | The 4-layer schema extracted from `scripts/cicd/enforce_tier_model.py:237–248`. The CI source-of-truth for all "what layer is X?" questions. |
| [`adr-index.md`](adr-index.md) | Full index of accepted Architecture Decision Records with status. Resolves the `ARCHITECTURE.md` ADR-table-staleness defect captured in [R10](../07-improvement-roadmap.md#r10). |
| [`re-derive.md`](re-derive.md) | How to regenerate the L3 import oracle and verify it byte-equals the snapshot. |

## A note on the snapshot

The four `l3-import-graph.*` files and `tier-model-oracle.txt` are **byte
copies** of the equivalent files in
[`../../arch-analysis-2026-04-29-1500/temp/`](../../arch-analysis-2026-04-29-1500/temp/),
preserved to keep the pack self-contained.

Minor numerical drift exists between the snapshot and the live tree at
this pack's HEAD — see [`../08-known-gaps.md#6`](../08-known-gaps.md#6-the-head-drift-caveat)
for the drift summary. To work against the live tree, follow
[`re-derive.md`](re-derive.md).

The snapshot has the property that the L3 import oracle is **byte-stable
across re-runs** (with `--no-timestamp`), so a `diff` against a fresh
re-derivation shows only true import-graph drift, not noise.
