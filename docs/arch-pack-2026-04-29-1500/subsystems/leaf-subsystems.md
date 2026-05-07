# Leaf Subsystems

Five L3 leaves and the L3 cli-root files. None is large enough to warrant
a dedicated cluster page; each is described inline below. Together they
account for ~11% of production Python LOC.

---

## §1 `mcp/` — Read-only Landscape Audit MCP Server

| Property | Value |
|----------|-------|
| **Layer** | L3 |
| **Size** | 9 files, 4,114 LOC |
| **Composite/Leaf** | Leaf (1 sub-pkg `analyzers/`; below LOC and file thresholds) |
| **Console script** | `elspeth-mcp = "elspeth.mcp:main"` |
| **Outbound** | `{contracts, core}` (reads Landscape) plus possibly engine for query helpers |
| **Inbound** | External (MCP client invocation) — not imported by any subsystem |

### Responsibility

Read-only Model Context Protocol server for Landscape audit-DB analysis.
Exposes:

- `diagnose()` — what is broken in the most recent runs?
- `get_failure_context(run_id)` — deep dive on a specific failure.
- `explain_token(run_id, token_id)` — row-level lineage trace.
- Domain-specific analyzers under `analyzers/`.

### Distinct from `composer_mcp/`

`mcp/` is the **post-hoc audit analyser** (read-only consumer of
Landscape). `composer_mcp/` is the **interactive pipeline-construction
MCP** (stateful, mutating). Separate console scripts, separate runtime
concerns, separate dependency surfaces. They share the MCP transport
but nothing else.

### SCC #0

`mcp` ↔ `mcp/analyzers` form a 2-node strongly-connected component
(analyser sub-package re-uses parent-namespace types via a weight-29
inverted edge). Provider-registry pattern; not a concern.

### Findings

None at this pack's depth — sub-1500-LOC files only.

---

## §2 `telemetry/` — Operational Telemetry

| Property | Value |
|----------|-------|
| **Layer** | L3 |
| **Size** | 14 files, 2,884 LOC |
| **Composite/Leaf** | Leaf (1 sub-pkg `exporters/`; below all thresholds) |
| **Console script** | None |
| **Outbound** | `{contracts, core}` plus possibly engine |
| **Inbound** | At minimum `{plugins}` (audited clients emit telemetry); probably also `{engine, web, cli}` |

### Responsibility

Operational telemetry pipeline:

- Circuit breaker.
- Exporters for OTLP, Datadog, Azure Monitor.
- Filtering, manager, hookspecs, serialization.

**Audit primacy** is the load-bearing invariant: telemetry emits
**after** Landscape recording. The audit is the legal record;
telemetry is operational visibility. The two channels are not
substitutable.

### SCC #2

`telemetry` ↔ `telemetry/exporters` form a 2-node strongly-connected
component (exporter sub-package re-uses parent-namespace types via a
weight-18 inverted edge). Not a concern.

### Findings

None at this pack's depth.

---

## §3 `tui/` — Lineage Explorer

| Property | Value |
|----------|-------|
| **Layer** | L3 |
| **Size** | 9 files, 1,175 LOC |
| **Composite/Leaf** | Leaf (2 sub-pkgs `screens/`, `widgets/`; below LOC and file thresholds) |
| **Console script** | None — surfaces through `cli` as `elspeth explain --run <run_id> --row <row_id>` |
| **Outbound** | `{contracts, core}` (reads Landscape) |
| **Inbound** | `{cli}` only |

### Responsibility

Textual-based TUI for interactive lineage exploration. The primary
auditor surface for "show me the complete lineage of this output."

Entry: `tui/explain_app.py`.

### SCC #3

`tui` ↔ `tui/screens` ↔ `tui/widgets` form a 3-node strongly-connected
component (screens and widgets reach back into the `tui` root for
shared types). Standard Textual pattern; not a concern.

### Findings

None at this pack's depth.

---

## §4 `testing/` — In-tree pytest Plugin

| Property | Value |
|----------|-------|
| **Layer** | L3 |
| **Size** | 2 files, 877 LOC |
| **Composite/Leaf** | Leaf |
| **Entry point** | `elspeth-xdist-auto = "elspeth.testing.pytest_xdist_auto"` (pytest11 entry-point) |
| **Outbound** | Likely none in `{contracts, core, engine}` — pure pytest tooling |
| **Inbound** | External (pytest plugin discovery) |

### Responsibility

Provides `pytest_xdist_auto` for automatic parallel-worker
configuration. **Not the test suite** — the test suite lives at
`tests/` (out of scope for this pack — see [`../08-known-gaps.md#2`](../08-known-gaps.md#2-test-architecture-tests)).

### Documentation correctness note

The institutional documentation describes "Testing subsystem ~9,500
LOC including ChaosLLM/ChaosWeb/ChaosEngine" — that **describes
`tests/`** (the test suite), not `src/elspeth/testing/` (this pytest
plugin). The chaos servers live in the `tests/` tree, not here.

This is captured as a doc-correctness item in
[R10](../07-improvement-roadmap.md#r10).

### Findings

None at this pack's depth.

---

## §5 `composer_mcp/` — Pipeline-Construction MCP Server

| Property | Value |
|----------|-------|
| **Layer** | L3 |
| **Size** | 3 files, 824 LOC |
| **Composite/Leaf** | Leaf |
| **Console script** | `elspeth-composer = "elspeth.composer_mcp:main"` |
| **Outbound** | `{contracts, core, engine, web/composer, web/execution, web/catalog}` |
| **Inbound** | External (MCP client invocation) |

### Responsibility

L3 stateful MCP server backing the LLM-driven pipeline composer:
sessions, plugin assistance, YAML generation, validation, source /
transform / sink discovery.

### Structural relationship

`composer_mcp/` is **structurally a sibling of `web/composer/`, not
of `mcp/`** — see [`web-composer.md#5-strengths`](web-composer.md) for
the W4 finding. The MCP transport is shared; nothing else.

### Findings

See [`web-composer.md`](web-composer.md) — `composer_mcp/` is reviewed
as part of the composer cluster.

---

## §6 `cli` — Typer CLI

| Property | Value |
|----------|-------|
| **Layer** | L3 |
| **Size** | 4 files, 2,942 LOC (`cli.py` 2,357, `cli_helpers.py`, `cli_formatters.py`, `__init__.py`) |
| **Composite/Leaf** | Treated as a single subsystem despite >1,500 LOC; below file-count threshold |
| **Console script** | `elspeth = "elspeth.cli:app"` |
| **Outbound** | `{contracts, core, engine, plugins, tui, telemetry}` and possibly `{mcp, web}` |
| **Inbound** | External (shell invocation) |

### Responsibility

Typer-based CLI. Top-level commands:

- `elspeth run` — execute a pipeline.
- `elspeth resume <run_id>` — resume an interrupted run.
- `elspeth validate` — validate a configuration.
- `elspeth explain --run <run_id> --row <row_id>` — launch the TUI.
- `elspeth plugins list` — list available plugins.
- `elspeth purge --retention-days N` — purge old payload data.

Also hosts the `TRANSFORM_PLUGINS` plugin registry — a coupling that
makes `cli.py` larger than it would otherwise be.

### Open item

`cli.py` (2,357 LOC) is on the deep-dive backlog ([R5](../07-improvement-roadmap.md#r5)
priority 7). The `TRANSFORM_PLUGINS` registry coupling is a candidate
for relocation; whether the file genuinely needs to be that large after
the registry moves is open.

### Findings

None at this pack's depth — captured for deep-dive in R5.
