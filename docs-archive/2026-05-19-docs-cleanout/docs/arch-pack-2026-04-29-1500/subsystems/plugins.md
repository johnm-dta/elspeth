# `plugins/` — L3 Plugin Ecosystem

**Layer:** L3 — outbound subset of `{contracts, core, engine, other L3}`.
**Size:** 98 files, 30,399 LOC. The largest single subsystem.
**Composite:** triggered by all three heuristics.
**Quality score:** **4 / 5**.

System-owned (not user-extensible) sources, transforms, sinks, plus
shared infrastructure (audited HTTP / LLM clients, hookspecs, base
classes).

---

## §1 Responsibility

`plugins/` is the **L3 plugin ecosystem**:

- **Sources** ingest external data (Tier 3 → Tier 2 coercion permitted here).
- **Transforms** process pipeline data (Tier 2; no coercion permitted).
- **Sinks** emit results to external systems.
- **Infrastructure** provides shared audited clients, hookspecs, and
  base classes for the leaf plugin packages above.

Plugins are **system-owned code**. ELSPETH uses `pluggy` for clean
architecture, not to accept arbitrary user plugins. They are developed,
tested, and deployed as part of ELSPETH with the same rigour as engine
code.

---

## §2 Internal sub-areas

| Sub-area | Files | LOC | Role |
|----------|------:|-----:|------|
| `infrastructure/` | 16 | 3,804 | Hookspecs, audited HTTP / LLM clients, base classes — the **structural spine** |
| `infrastructure/batching/` | several | 1,024 | Batching primitives |
| `infrastructure/clients/` | several | 3,790 | Client surfaces (LLM, HTTP, etc.) |
| `infrastructure/clients/retrieval/` | several | 1,031 | Retrieval-augmented client surfaces |
| `infrastructure/pooling/` | several | 1,133 | Connection / resource pooling |
| `sources/` | several | 3,519 | External-data sources |
| `transforms/` | several | 4,125 | Transform plugins |
| `transforms/llm/` | several | 5,740 | LLM transforms |
| `transforms/llm/providers/` | several | 663 | LLM provider registry |
| `transforms/azure/` | several | 1,125 | Azure-specific transforms |
| `transforms/rag/` | several | 922 | Retrieval-augmented generation |
| `sinks/` | several | 3,515 | External-data sinks |

---

## §3 Dependencies

| Direction | Edges |
|-----------|-------|
| **Outbound** | `{contracts, core, engine, other L3}` — notably to `telemetry/` for audited clients |
| **Inbound** | Other L3: `{web, mcp, composer_mcp, tui, cli, testing}` (engine instantiates plugins via the registry, but the registry is exposed through L3 surfaces) |

### The intra-cluster spine pattern

All 23 intra-cluster edges flow toward `plugins/infrastructure/`. The
heaviest single L3 edge in the codebase is
`plugins/sinks → plugins/infrastructure` (weight 45). Sources,
transforms, and sinks are **clients of infrastructure**, not peers of
one another.

```text
plugins/sinks  ──── w=45 ────► plugins/infrastructure
plugins/transforms  ─ w=40 ──► plugins/infrastructure
plugins/sources  ── w=17 ───► plugins/infrastructure
```

See [`../04-component-view.md#2-the-plugin-spine`](../04-component-view.md#2-the-plugin-spine)
for the diagram.

### Cross-cluster inbound

`plugins/infrastructure/` is the dominant cross-cluster inbound
destination:

| Source | Edge weight |
|--------|------------:|
| `web/composer` | 22 |
| `cli` | 8 |
| `web/execution` | 4 |
| `testing` | 4 |
| `web/catalog` | 3 |
| `web` (root) | 1 |

---

## §4 Findings

### P1 — `azure_batch.py` is unread at component depth · **Medium**

`plugins/transforms/llm/azure_batch.py` (1,592 LOC) is the largest
single plugin file. The LLM batch path is high-stakes:

- **Financial cost** — batch sizing decisions affect provider spend.
- **Audit coverage** — every prompt/response must be recorded.
- **Retry semantics** — partial-batch failures need careful handling.

Internal cohesion is un-assessed at this pack's depth.

**Recommendation:** [R5](../07-improvement-roadmap.md#r5).

### P2 — Trust-tier discipline is structural but not runtime-enforced · **Medium**

Every source module repeats "ONLY place coercion is allowed"; every
sink module repeats "wrong types = upstream bug = crash"; the
discipline is encoded in the `allow_coercion` config flag.

But **cross-cluster invariant tests** — for example, a fixture that
injects a transform that coerces and asserts the run fails — do **not**
exist. The contract is honoured today by author discipline; CI does not
catch a violator.

**Recommendation:** a property-based or fixture-based runtime probe,
paired with an architecture-pack decision on whether to mechanise the
discipline.

### P3 — Plugin-count drift in `ARCHITECTURE.md` · **Low**

Three statements in the institutional documentation disagree about the
number of plugins:

- "25 plugins" (one statement)
- "46 plugins" (another statement, same document)
- **29 plugins** (live verified count)

Four post-doc plugins were added without a documentation update.

**Recommendation:** [R10](../07-improvement-roadmap.md#r10).

### P4 — Module-level cycle in `plugins/transforms/llm` · **Low**

`plugins/transforms/llm` ↔ `plugins/transforms/llm/providers` form a
2-node strongly-connected component (SCC #1 in the L3 oracle).
**Provider-registry pattern with deferred runtime instantiation**;
the runtime decoupling is documented at `transform.py:9-13`.

The cycle is visible to the import system but runtime-decoupled. No
runtime impact; visible only in static analysis.

**Recommendation:** compare cost of moving shared types into
`plugins/infrastructure/` versus leaving the cycle visible.

---

## §5 Strengths

### `plugins/infrastructure/` is the structural spine and it is honoured consistently

All 23 intra-cluster edges flow toward `infrastructure/`; the heaviest
single L3 edge in the codebase terminates there. The dependency shape
matches the documented design.

### Trust-tier discipline is documented identically in every leaf module

Repetition is **not a smell** here; it is the protocol that prevents
drift. New contributors writing a new source see the same "ONLY place
coercion is allowed" notice as every existing source. Removing the
duplication would weaken the discipline.

---

## §6 Cross-cluster handshakes

| Partner | Direction | Shape |
|---------|-----------|-------|
| `contracts/` | plugins → contracts | Protocols, base classes, audit DTOs |
| `core/` | plugins → core | `core/expression_parser` for transform conditions; rate limiters; payload store |
| `engine/` | plugins → engine | Plugins implement engine-defined interfaces (`Source`, `Transform`, `Sink`) |
| `web/composer` | web → plugins | Heaviest cross-cluster inbound edge (weight 22): the composer reads plugin schemas via `infrastructure/` |
| `cli` | cli → plugins | `TRANSFORM_PLUGINS` registry consumes plugin metadata (weight 8) |
| `telemetry/` | plugins → telemetry | Audited clients emit telemetry signals |
