# Appendix A — Glossary

Vocabulary used throughout this pack. Terms are listed alphabetically.

---

### ADR (Architecture Decision Record)

A short document recording an architectural decision, the context that
motivated it, the alternatives considered, and the consequences of the
decision. ELSPETH ADRs live in `docs/architecture/adr/`. See
[`../reference/adr-index.md`](../reference/adr-index.md) for the index.

### ADR-010

The Declaration-Trust Framework ADR. Defines the 4-site dispatcher
(`pre_emission_check`, `post_emission_check`, `batch_flush_check`,
`boundary_check`) and the `DeclarationContract` protocol that adopters
implement.

### Aggregation (transform subtype)

A transform that collects N rows until a trigger fires, then emits a
result. **Stateful.** Distinct from a row transform (one-in / one-out)
or a coalesce (merges parallel-path results).

### Allowlist with justification

The pattern used by `enforce_tier_model.py` and `enforce_freeze_guards.py`
for legitimate exceptions to a CI rule. Each exemption carries a
written rationale. Pattern under `config/cicd/`.

### Attributability test

The audit-grade requirement: for any output, the operation
`explain(recorder, run_id, token_id)` must prove complete lineage back
to source data, configuration, and code version.

### Audit primacy

The ordering rule for telemetry, logging, and audit:

1. Audit fires first (synchronous, crash-on-failure).
2. Telemetry fires second (asynchronous, best-effort).
3. Logging only fires when audit and telemetry are both unavailable.

### Coalesce (transform subtype)

A transform that merges results from parallel DAG paths after a fork.

### Container view (C4)

A C4 Level-2 view: the top-level deployable / runnable units of the
system, with their dependencies. In this pack, the 11 top-level
subsystems. See [`../03-container-view.md`](../03-container-view.md).

### Component view (C4)

A C4 Level-3 view: a drilldown into one container's internal structure.
This pack uses it for the structurally interesting zones (web SCC,
plugin spine, audit backbone). See
[`../04-component-view.md`](../04-component-view.md).

### Composite subsystem

A subsystem that meets at least one of: ≥4 sub-packages, ≥10k LOC, ≥20
files. Five of the 11 ELSPETH subsystems are composite (`engine`,
`core`, `contracts`, `plugins`, `web`).

### Context view (C4)

A C4 Level-1 view: the system as a black box with external actors and
external systems. See [`../01-system-context.md`](../01-system-context.md).

### DAG (Directed Acyclic Graph)

The structure pipelines compile to. Linear pipelines are degenerate
DAGs (a single `continue` path). Owned by `core/dag/`.

### Declaration trust

A plugin authoring discipline (codified by ADR-010) where plugins
declare their behavioural properties (e.g., "may drop rows", "creates
tokens", "requires fields X, Y, Z") and the framework enforces those
declarations at runtime via the 4-site dispatcher.

### Defensive programming (forbidden)

Use of `.get()`, `getattr()`, `isinstance()`, `hasattr()`, or silent
exception handling to suppress errors from nonexistent attributes,
malformed data, or incorrect types — forbidden in ELSPETH except at
trust boundaries. The CI-detected anti-pattern.

### Deep_freeze contract

Frozen dataclass containers (`dict`, `list`, `set`, `Mapping`,
`Sequence`) must be made deeply immutable in `__post_init__` via
`contracts/freeze.py:freeze_fields`. `frozen=True` alone is insufficient
because Python's frozen dataclasses only block reassignment, not
mutation through references.

### Engine

The L2 SDA execution layer. Orchestrator, RowProcessor, executors,
RetryManager, ArtifactPipeline, SpanFactory, Triggers, TokenManager.

### Fabrication (forbidden)

Inventing data the external system never asserted (e.g., inferring an
absent field from adjacent fields). Distinct from coercion (which is
meaning-preserving). The fabrication decision test is the canonical
discriminator.

### Fork

A DAG split point where a row produces multiple downstream tokens, each
mintting a new `token_id` while inheriting `row_id`.

### Gate (transform subtype)

A transform subtype that evaluates a row and decides routing via
`continue`, `route_to_sink`, or `fork_to_paths`. Config-driven; not a
plugin.

### Landscape

The audit database. Implemented under `core/landscape/`. Persistent;
configurable backend (SQLite or Postgres). The legal record of every
operation in every run.

### Layer model (4-layer)

The strict downward import hierarchy:

```
L0 contracts → L1 core → L2 engine → L3 application surfaces
```

CI-enforced by `scripts/cicd/enforce_tier_model.py`. See
[`../02-architecture-overview.md#1-the-4-layer-model`](../02-architecture-overview.md#1-the-4-layer-model).

### MCP (Model Context Protocol)

The protocol used by LLM agents and IDEs to connect to ELSPETH's two
MCP surfaces: `elspeth-mcp` (read-only audit analysis) and
`elspeth-composer` (interactive pipeline construction).

### Offensive programming (encouraged)

Proactively detecting invalid states and throwing meaningful exceptions
with full context. Always uses `from exc` to preserve exception chains.
The opposite of defensive programming for code we own.

### Oracle (L3 import oracle)

The deterministic JSON artefact at
[`../reference/l3-import-graph.json`](../reference/l3-import-graph.json)
that enumerates every L3 import edge with weight, source, target, and
sample sites. Byte-stable across re-runs given the same source tree.

### `pluggy`

The plugin architecture library used by ELSPETH (and by `pytest`).
Used for clean architecture, **not** for accepting arbitrary user
plugins — see [`../02-architecture-overview.md#5-plugin-ownership-system-code-not-user-code`](../02-architecture-overview.md#5-plugin-ownership-system-code-not-user-code).

### Quarantine

Routing a row to a non-result destination because it could not be
coerced or validated at the source boundary. The audit trail records
"row 42 was quarantined because field X was NULL" — that is a valid
audit outcome, not a failure.

### Row transform

A transform that processes one row, emits one row. **Stateless.**
Contrast with aggregation (stateful, N-in / 1-out) and coalesce (merges
parallel paths).

### `row_id` / `token_id` / `parent_token_id`

Token-identity fields that survive forks and joins:

- `row_id` — stable source-row identity.
- `token_id` — instance of a row in a specific DAG path.
- `parent_token_id` — lineage for forks and joins.

### SCC (Strongly-Connected Component)

A maximal set of nodes in a directed graph where every node can reach
every other node. ELSPETH has 5 SCCs in the L3 import graph; the
largest (SCC #4) spans the 7 `web/*` sub-packages.

### SDA (Sense / Decide / Act)

ELSPETH's pipeline pattern:

- **Sense** — sources ingest data.
- **Decide** — transforms (including gates and aggregations) process it.
- **Act** — sinks emit results.

See [`../02-architecture-overview.md#3-the-sda-execution-pattern`](../02-architecture-overview.md#3-the-sda-execution-pattern).

### Sink

A plugin that writes results to an external system. One or more per
pipeline; each named.

### Source

A plugin that loads data from an external system. **Exactly one per
run.** The only place where Tier-3-to-Tier-2 coercion is permitted.

### Terminal state

One of seven row outcomes: `COMPLETED`, `ROUTED`, `FORKED`,
`CONSUMED_IN_BATCH`, `COALESCED`, `QUARANTINED`, `FAILED`, `EXPANDED`.
(`BUFFERED` is non-terminal — it becomes `COMPLETED` on flush.)

The terminal-state-per-token invariant says every row reaches **exactly
one** terminal state. Structurally guaranteed via
`engine/executors/state_guard.py:NodeStateGuard`.

### Tier 1 / Tier 2 / Tier 3

The three trust tiers in the data manifesto:

- **Tier 1** — Our data (audit DB, checkpoints). Full trust; crash on
  any anomaly.
- **Tier 2** — Pipeline data (post-source). Elevated trust; types are
  trustworthy; transforms and sinks expect conformance.
- **Tier 3** — External data (source input). Zero trust; validate and
  coerce at the boundary.

See [`../02-architecture-overview.md#2-the-three-tier-trust-model`](../02-architecture-overview.md#2-the-three-tier-trust-model).

### Token

An instance of a row in a specific DAG path. Forks mint new tokens; the
`token_id` and `parent_token_id` carry the lineage.

### TYPE_CHECKING import

A Python pattern where an import only happens for type checkers (under
`if TYPE_CHECKING:`). At runtime the import does not occur. Visible in
the source but not in runtime coupling. ADR-006d forbids using
TYPE_CHECKING imports as a workaround for layer violations; the
canonical marker of a deferred structural fix.
