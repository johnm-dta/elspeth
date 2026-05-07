# L2 cluster validation report — core/

## Summary verdict

**PASS** — all 6 contract criteria satisfied. No findings; no blocking issues.

## Per-criterion verdict

### Criterion 1: Sub-subsystem entries vs reality — PASS

`ls src/elspeth/core/` returns 6 sub-packages (`landscape/`, `dag/`, `checkpoint/`, `rate_limit/`, `retention/`, `security/`) and 11 standalone .py modules + `__init__.py`. The 9 catalog entries enumerate exactly this surface:

- Entries 1–6 = the 6 sub-packages (paths verified: `landscape/` 18 files, `dag/` 5 files, `checkpoint/` 5 files, `rate_limit/` 3 files, `retention/` 2 files, `security/` 4 files — all match).
- Entry 7 (`config.py + dependency_config.py + secrets.py`), Entry 8 (`canonical.py + templates.py + expression_parser.py`), Entry 9 (`events.py + identifiers.py + logging.py + operations.py + payload_store.py + __init__.py`) account for all 11 standalone modules + the cluster `__init__.py`. Coverage is complete and disjoint.

No invented entries; no missing modules.

### Criterion 2: intra-cluster-edges.json — PASS

`temp/intra-cluster-edges.json` exists, is well-formed JSON, and reports `intra_node_count: 0`, `intra_edge_count: 0`. Rationale field cites Δ L2-2 and the L1 oracle scope correctly. `byte_equality_assertion.applicable = false` per Δ L2-6 specialisation (cluster is L1, not L3). Catalog entries reference "Empty per `temp/intra-cluster-edges.json`" consistently in all 9 entries.

### Criterion 3: [CITES] / [DIVERGES FROM] resolution — PASS

Spot-checked 10 citations against `00b-existing-knowledge-map.md`:

- `KNOW-A20` (line 26: ~600 LOC checkpoint), `KNOW-A21` (line 27: ~300 LOC rate_limit), `KNOW-A24` (line 30: 21 audit tables), `KNOW-A27` (line 33: ~652 LOC expression_parser), `KNOW-A29` (line 35: facade), `KNOW-A30` (line 36: 4 repos), `KNOW-A31` (line 37: ~1,480 LOC ExecutionRepository), `KNOW-A47` (line 53), `KNOW-C57` (line 138), `KNOW-ADR-006b` (line 235) — all resolve.

Divergence claims independently re-verified:

- `wc -l landscape/execution_repository.py` → **1,750** (catalog: 1,750; KNOW-A31: ~1,480) ✓
- `wc -l checkpoint/*.py` sum → **1,237** (catalog: 1,237; KNOW-A20: ~600) ✓
- `wc -l rate_limit/*.py` sum → **470** (catalog: 470; KNOW-A21: ~300) ✓
- `wc -l expression_parser.py` → **820** (catalog: 820; KNOW-A27: ~652) ✓
- `grep '^[a-z_]*_table = Table' landscape/schema.py | wc -l` → **20** (catalog: 20; KNOW-A24: 21) ✓
- `wc -l config.py` → **2,227** ✓; `wc -l dag/graph.py` → **1,968** ✓; `wc -l data_flow_repository.py` → **1,590** ✓.

All 5 divergences claimed in §7 are factual.

### Criterion 4: Cross-cluster boundary discipline — PASS

Grep for `engine/`, `web/`, `mcp/` mentions in catalog returns 6 lines, all in **External coupling (cross-cluster)** sections naming WHO imports core/ (inbound) or WHAT contracts/ symbols core/ exports (outbound). No claims about internals of other clusters; no design or refactoring opinions about engine/web/mcp/composer_mcp/telemetry/tui/testing/cli. The "Inbound from `engine/orchestrator/`" / "consumed by `engine/executors/sink.py`" mentions are external-coupling references explicitly PERMITTED by the contract. Section 6 of the report properly uses the Δ L2-4 deferral channel for the 5 synthesis observations (Synthesis-1 through Synthesis-5) that would otherwise reach into other clusters.

### Criterion 5: SCC handling — PASS

`temp/l3-import-graph.json` `strongly_connected_components` enumerates exactly 5 SCCs: `[mcp, mcp/analyzers]`, `[plugins/transforms/llm, plugins/transforms/llm/providers]`, `[telemetry, telemetry/exporters]`, `[tui, tui/screens, tui/widgets]`, `[web, web/auth, web/blobs, web/composer, web/execution, web/secrets, web/sessions]`. No `core/*` entry. Report's claim "SCC handling (Δ L2-7): N/A" is correct.

### Criterion 6: Layer-check oracle — PASS

Both artefacts present (`temp/layer-check-core.txt` and `temp/layer-check-core-empty-allowlist.txt`). Re-ran on the empty-allowlist file:

- `grep -c '^  Rule: L1'` → **0** ✓
- `grep -c '^  Rule: TC'` → **0** ✓
- `grep '^  Rule:' | sort | uniq -c | sort -rn` →
  - 170 R5 (isinstance) ✓
  - 18 R6 (silent-except) ✓
  - 15 R1 (dict.get) ✓
  - 2 R4 (broad-except) ✓
  - **Total: 205** ✓

Histogram and total match catalog and report claims exactly.

## Findings (specific issues, if any)

None.

## Recommendations to author (optional)

1. Minor: §2 of `04-cluster-report.md` totals **20,691 LOC** against L1's 20,791 (~0.5% drift). Both report and `00-cluster-coordination.md` already document this honestly via the `find … | xargs cat` vs per-file `wc -l` accounting difference; no further action needed.
2. The catalog correctly defers internals of the 4 deep-dive candidates (`config.py`, `dag/graph.py`, `landscape/execution_repository.py`, `landscape/data_flow_repository.py`). Confidence levels (Medium for `config.py` and `expression_parser.py`) appropriately flag the deferral. No content opinion to add.
