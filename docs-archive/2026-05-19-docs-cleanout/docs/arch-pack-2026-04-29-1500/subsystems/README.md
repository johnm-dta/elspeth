# Subsystem Reference

Per-subsystem detail: responsibility, dependencies, internal sub-area
shape, findings, strengths.

The architecture pack treats the 11 top-level subsystems asymmetrically,
matching their structural significance:

- **Five composite subsystems** (each ≥4 sub-pkgs OR ≥10k LOC OR ≥20
  files) get a dedicated page with cluster-level depth.
- **Six leaf subsystems** are aggregated into a single page because
  none is large enough to warrant its own.

| Page | Subsystem(s) | Layer | Depth |
|------|--------------|:----:|------|
| [`contracts.md`](contracts.md) | `contracts/` | L0 | Cluster |
| [`core.md`](core.md) | `core/` | L1 | Cluster |
| [`engine.md`](engine.md) | `engine/` | L2 | Cluster |
| [`plugins.md`](plugins.md) | `plugins/` | L3 | Cluster |
| [`web-composer.md`](web-composer.md) | `web/` + `composer_mcp/` (the composer cluster) | L3 | Cluster |
| [`leaf-subsystems.md`](leaf-subsystems.md) | `mcp/`, `telemetry/`, `tui/`, `testing/`, `cli` | L3 | Page each |

The findings, strengths, and confidence ratings on each page are the
same as those in [`../06-quality-assessment.md`](../06-quality-assessment.md);
this directory provides the cluster-internal detail (sub-area
inventory, cross-cluster handshakes, internal coupling notes) that the
quality assessment summarises.
