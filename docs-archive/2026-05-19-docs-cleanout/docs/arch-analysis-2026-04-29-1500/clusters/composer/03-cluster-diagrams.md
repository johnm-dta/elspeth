# 03 — Cluster Diagrams (composer cluster: web/ + composer_mcp/)

C4 Container and Component diagrams scoped to this cluster only. Diagrams are observational; arrow weights cite the L3 import graph (`temp/intra-cluster-edges.json` and `docs/arch-analysis-2026-04-29-1500/temp/l3-import-graph.json`). Per Δ L2-4, no cross-cluster claims beyond the 6 outbound edges enumerated in `01-cluster-discovery.md` §4.

## D1. C4 Container view (this cluster within ELSPETH)

Shows the cluster's two containers and their cross-cluster boundaries. The cluster has 0 inbound and 6 outbound cluster-external edges; this is rendered as edge weights, not as arrows from outside.

```mermaid
graph LR
    subgraph cluster["composer cluster (this L2 pass)"]
        web["web/<br/>FastAPI server<br/>72 files / 22,558 LOC<br/>(Python only)"]
        cmcp["composer_mcp/<br/>MCP transport<br/>3 files / 824 LOC"]
    end

    subgraph other["Other clusters (cited only via outbound edges)"]
        plugins_infra["plugins/infrastructure"]
        telemetry["telemetry/"]
        elspeth_root["elspeth (root pkg)"]
        frontend["web/frontend/<br/>(out of scope)"]
    end

    cmcp -- "weight 12+1+1<br/>(F1: thin transport)" --> web
    web -- "weight 22+4+3+1<br/>(semantic validator, dependencies)" --> plugins_infra
    web -- "weight 1 (conditional)<br/>service.py:781" --> telemetry
    web -- "weight 3<br/>execution/service.py:30,805" --> elspeth_root
    web -. "static mount<br/>KNOW-G7" .-> frontend

    classDef cluster fill:#e8f4ff,stroke:#0070c0,color:#000
    classDef other fill:#f4f4f4,stroke:#888,color:#000
    classDef oos fill:#ffe8e8,stroke:#c00,color:#000,stroke-dasharray: 5 5
    class web,cmcp cluster
    class plugins_infra,telemetry,elspeth_root other
    class frontend oos
```

**Notes:**
- Edge weights are *aggregated* across the cluster's 6 outbound edges enumerated in `01-cluster-discovery.md` §4 (1+3+22+3+4+1=34, not all to one target — see source for breakdown).
- The dashed edge to `web/frontend/` is a *static-file mount*, not an import — it is included for completeness per [CITES KNOW-G7] but is not an oracle edge.
- 0 inbound arrows from other clusters: confirmed by `temp/intra-cluster-edges.json:cross_cluster_inbound_edges = []`.

## D2. C4 Component view (within `web/` + `composer_mcp/`)

Shows the cluster's 11 sub-package nodes (10 in-scope Python sub-packages + composer_mcp/), the SCC #4 membership, and the heaviest intra-cluster edges. Per Δ L2-7, the SCC is rendered as a contained group; non-SCC siblings (catalog/, middleware/) sit alongside.

```mermaid
graph TB
    subgraph cmcp_root["composer_mcp/ scope root"]
        cmcp[("composer_mcp/<br/>3 files, 824 LOC")]
    end

    subgraph web_root["web/ scope root"]
        subgraph scc["SCC #4 (7 nodes — acyclic decomposition not possible)"]
            wpkg[("web/ package root<br/>7 files, 1,072 LOC<br/>app.py is wiring root")]
            wauth[("web/auth/<br/>8 files, 1,224 LOC")]
            wblobs[("web/blobs/<br/>6 files, 1,952 LOC")]
            wcomp[("web/composer/<br/>11+1 files, 8,274 LOC<br/>★ tools.py 3,804 + state.py 1,710 = L3 deep-dive")]
            wexec[("web/execution/<br/>11 files, 3,748 LOC")]
            wsec[("web/secrets/<br/>6 files, 1,011 LOC")]
            wsess[("web/sessions/<br/>9 files, 4,080 LOC")]
        end

        wcat[("web/catalog/<br/>5 files, 407 LOC<br/>(acyclic — outside SCC)")]
        wmw[("web/middleware/<br/>3 files, 257 LOC<br/>(acyclic — outside SCC)")]
        wfront["web/frontend/ (~13k TS/React, OUT OF SCOPE)"]:::oos
    end

    cmcp -- "weight 12<br/>(F1)" --> wcomp
    cmcp -. "weight 1" .-> wcat
    cmcp -. "weight 1" .-> wpkg

    wpkg -- "wires sub-packages (app.py)" --> wauth
    wpkg --> wblobs
    wpkg --> wcomp
    wpkg --> wexec
    wpkg --> wsec
    wpkg --> wsess

    wauth -- "weight 6" --> wpkg
    wblobs -- "weight 1" --> wpkg
    wcomp -- "weight 4" --> wpkg
    wexec -- "weight 15<br/>(heaviest edge)" --> wpkg
    wsec -- "weight 6" --> wpkg
    wsess -- "weight 5" --> wpkg

    wsess -- "weight 15<br/>(persists composer drafts)" --> wcomp
    wexec -- "weight 9<br/>(validates composer YAML)" --> wcomp
    wexec -- "weight 7" --> wsess
    wexec -- "weight 6" --> wauth
    wexec -- "weight 5" --> wblobs
    wcomp -- "weight 5" --> wblobs
    wcomp -- "weight 4" --> wcat
    wcomp -- "weight 5" --> wsess
    wblobs -- "weight 4" --> wsess
    wblobs -- "weight 2" --> wauth
    wauth -- "weight 1" --> wmw

    wpkg --> wcat
    wpkg --> wmw

    classDef scc fill:#fff4d6,stroke:#b07700,color:#000,stroke-width:2px
    classDef acyc fill:#e8f4ff,stroke:#0070c0,color:#000
    classDef cmcp fill:#dff5e0,stroke:#0a7c2a,color:#000
    classDef oos fill:#ffe8e8,stroke:#c00,color:#000,stroke-dasharray: 5 5

    class wpkg,wauth,wblobs,wcomp,wexec,wsec,wsess scc
    class wcat,wmw acyc
    class cmcp cmcp
```

**Reading guide:**
- **Yellow (SCC #4):** the 7 nodes that form the strongly-connected component. Bidirectional arrows between `web/` package root and each sub-package signal the FastAPI app-factory cycle.
- **Blue (acyclic):** `web/catalog/` and `web/middleware/` — in cluster scope but not in the cycle.
- **Green:** `composer_mcp/` — the cluster's second scope root; thin transport to `web/composer/`.
- **Red dashed:** `web/frontend/` — out of scope, recorded only.
- **Heaviest edges:** `web/execution → web` (15), `web/sessions → web/composer` (15), `composer_mcp → web/composer` (12), `web/execution → web/composer` (9), `web/execution → web/sessions` (7).
- **The two weight-15 edges have semantically different causes:** the execution → web edge is *cycle-driven* (sub-package reaching back for shared types via the app-factory pattern); the sessions → composer edge is *data-flow-driven* (sessions persists composer drafts). Both are equally heavy, neither is reducible without restructuring.

## D3. SCC reference (oracle paste-through)

Direct citation of the L1 oracle entry that this cluster's analysis treats as a unit:

```
[ORACLE: strongly_connected_components[4]]
  members: ['web', 'web/auth', 'web/blobs', 'web/composer',
            'web/execution', 'web/secrets', 'web/sessions']
  size: 7
  all_in_cluster: true
  oracle_path: temp/l3-import-graph.json:strongly_connected_components[4]
```

Per Δ L2-7, the cluster does not propose an acyclic decomposition. The cycle's load-bearing nature is analysed in `04-cluster-report.md` §SCC analysis.
