# `web/` + `composer_mcp/` — The Composer Cluster

**Layer:** L3 — outbound subset of `{contracts, core, engine, other L3}`.
**Size:** 75 files, ~23,400 LOC of Python (`web/` 72 / 22,558 + `composer_mcp/` 3 / 824). The `web/frontend/` SPA (~13k LOC TS/React) is **out of scope** — see [`../08-known-gaps.md#1`](../08-known-gaps.md#1-frontend-webfrontend).
**Composite:** `web/` is composite (8 backend sub-packages); `composer_mcp/` is a leaf.
**Quality score:** **3 / 5**.

The composer cluster scores lowest in the pack not because the code is
poor — by every mechanical measure it is competent — but because of
three structural features that combine to make it the system's
highest-risk architectural surface.

---

## §1 Responsibility

### `web/`

The L3 FastAPI web UI server. Eight backend sub-packages plus the
deferred frontend:

- **`web/composer/`** — LLM-driven pipeline composer; the cluster's
  data backbone (3,860 LOC `tools.py` + 1,710 LOC `state.py`).
- **`web/execution/`** — pipeline execution surfaces.
- **`web/catalog/`** — plugin discovery surface.
- **`web/auth/`** — authentication.
- **`web/sessions/`** — session and draft persistence.
- **`web/secrets/`** — credential handling.
- **`web/blobs/`** — blob storage.
- **`web/middleware/`** — request middleware.
- **`web/frontend/`** — TypeScript / React SPA. **Out of scope** for
  this pack.

The `webui` extra (PyJWT, bcrypt, websockets, FastAPI, uvicorn,
litellm) gates this subsystem's dependencies.

### `composer_mcp/`

The L3 stateful MCP server backing the LLM-driven pipeline composer:
sessions, plugin assistance, YAML generation, validation, source /
transform / sink discovery. Three files, 824 LOC.

The tool surface (`mcp__elspeth-composer__*`) is interactive and
mutating: `set_source`, `upsert_node`, `upsert_edge`, `set_output`,
`generate_yaml`. Distinct from `mcp/` (which is read-only audit
analysis).

---

## §2 Internal sub-areas — the 7-node SCC

The seven `web/*` sub-packages form a strongly-connected component
(SCC #4 in the L3 oracle). The cycle is **the FastAPI app-factory
pattern made structural** — see [`../04-component-view.md#1-the-7-node-web-scc-and-the-heavy-intra-cycle-edges`](../04-component-view.md#1-the-7-node-web-scc-and-the-heavy-intra-cycle-edges)
for the diagram.

| Edge | Weight |
|------|-------:|
| `web/sessions → web/composer` | 17 |
| `web/execution → web` (root) | 14 |
| `composer_mcp → web/composer` | 13 |
| `web/execution → web/composer` | 9 |
| `web/execution → web/sessions` | 7 |
| `web/composer → web/blobs` | 5 |
| (others ≤ 5) | … |

---

## §3 Dependencies

| Direction | Edges |
|-----------|-------|
| **Outbound (web)** | `{contracts, core, engine, plugins/infrastructure}` plus the seven internal sub-package edges that form SCC #4 |
| **Outbound (composer_mcp)** | `{contracts, core, engine, web/composer, web/execution, web/catalog}` |
| **Inbound (cluster as a whole)** | **Zero cross-cluster inbound edges.** Only the two console-script entry points (`elspeth-web`, `elspeth-composer`) consume the cluster. |

---

## §4 Findings

### W1 — 7-node strongly-connected component spans every `web/*` sub-package · **High**

The cycle covers `web ↔ web/auth ↔ web/blobs ↔ web/composer ↔ web/execution ↔ web/secrets ↔ web/sessions`.
**No acyclic decomposition is possible within `web/`.** The cycle is
structurally load-bearing — it implements the FastAPI app-factory
pattern:

- `web/app.py:create_app()` imports every sub-package's router (the
  wiring leg).
- Sub-packages reach back via `from elspeth.web.config import WebSettings`
  and `run_sync_in_worker` (the shared-infrastructure leg).

Both directions are intentional.

**Impact:** any architectural change in `web/` must reason about all
seven sub-packages simultaneously. Adding a new sub-package extends
the SCC by default.

**Recommendation:** [R2](../07-improvement-roadmap.md#r2). The probable
shape:

1. Extract `web/_core/` containing `WebSettings` and `run_sync_in_worker`
   so sub-packages depend on `_core` rather than the namespace root.
2. Make `web/app.py` the only place that imports sub-package routers.

**Until then, freeze new sub-package additions to `web/` unless
explicitly architecture-reviewed.**

### W2 — Largest concentration of composer logic · **Medium**

`web/composer/tools.py` (3,860 LOC) and `web/composer/state.py` (1,710
LOC) together carry 5,570 LOC — larger than any non-engine subsystem at
this pack's depth. The composer state machine is the most
architecturally-consequential surface in the system after the engine,
and any change has high blast radius.

**Recommendation:** decomposition is paired with the SCC#4 decision
(W1). Isolated decomposition without the SCC context risks producing a
worse cycle.

### W3 — `web/sessions/routes.py` was missed by the prior inventory · **Medium**

The prior inventory listed 12 files >1,500 LOC; `web/sessions/routes.py`
was not among them. At this pack's HEAD it is **2,067 LOC** — a +504
LOC growth (+32%) since the inventory pass. The file exists, is large,
and remains unread at component depth.

**Recommendation:** [R3](../07-improvement-roadmap.md#r3) — re-run the
≥1,500-LOC scan and add the missed entries to the deep-dive backlog.

### W4 — `composer_mcp/` is structurally a sibling of `web/composer/`, not of `mcp/` · **Medium**

The institutional layout framed `mcp/` and `composer_mcp/` as siblings
(both expose MCP transports). The L3 import oracle records:

- **Zero edges** between `mcp/` and `composer_mcp/`.
- A weight-13 edge from `composer_mcp → web/composer`.
- A weight-4 edge from `composer_mcp → web/execution` (new at this
  pack's HEAD).

`composer_mcp/` is the MCP transport that the web composer uses;
calling it a sibling of the audit-analyser `mcp/` mis-frames the
relationship.

**Recommendation:** either move `composer_mcp/` under `web/composer/`,
or document the structural relationship in `ARCHITECTURE.md`.

### W5 — `web/execution → .` (cli root) edge purpose unclear · **Low**

A weight-3 edge from `web/execution` into the cli-root namespace.
Could be benign (re-export of a public symbol from
`elspeth/__init__.py`) or a deferred-import hack to bypass an explicit
cluster dependency. Resolution takes minutes.

---

## §5 Strengths

### The composer cluster is a structural import-graph leaf

**Zero inbound edges** from any other cluster. Only the two
console-script entry points (`elspeth-web`, `elspeth-composer`) consume
the cluster. Architectural changes inside the cluster cannot break
library callers elsewhere — a remarkably clean blast-radius property
for a ~23k-LOC subsystem.

This is the single strongest cluster-level property in the codebase
and survives the SCC-internal decomposition decision.

---

## §6 Cross-cluster handshakes

| Partner | Direction | Shape |
|---------|-----------|-------|
| `plugins/infrastructure/` | web/composer → plugins | Heaviest cross-cluster inbound edge in the codebase (weight 22): the composer reads plugin schemas |
| `core/secrets.py` | web/secrets → core | Runtime secret-ref resolver consumed when threading `{"secret_ref": ...}` references through resolved configs |
| `engine/` | web/execution → engine | Pipeline execution dispatch |
| `composer_mcp` ↔ `web/composer` | composer_mcp → web/composer | Weight 13; the MCP transport for the web composer's state machine |

For the web SCC diagram and the cycle's structural cause, see
[`../04-component-view.md#1-the-7-node-web-scc-and-the-heavy-intra-cycle-edges`](../04-component-view.md#1-the-7-node-web-scc-and-the-heavy-intra-cycle-edges).
