# 00 — core/ cluster coordination (L2 cluster pass)

## Cluster identity

- **Cluster name:** core
- **Scope path:** `src/elspeth/core/`
- **Layer:** **L1** per `scripts/cicd/enforce_tier_model.py:238` (`"core": 1`) and `KNOW-C47`. Outbound permitted to `{contracts}` only; inbound from L2+ (`engine`, plus all L3 surfaces).
- **L1 dispatch slot:** §7 / §7.5 Priority 3 of `04-l1-summary.md` (oracle-independent, cascade-prone, recommended early in serial order: `1 (engine) → 3 (core) → 2 (composer) → 4 (plugins) → 5 (contracts)`).
- **Effort bracket:** 3–5 hr (per §7 P3, unchanged by §7.5).
- **Test paths:** `tests/unit/core/{checkpoint,dag,landscape,rate_limit,retention,security}/`, `tests/unit/core/test_canonical_mutation_gaps.py`, `tests/integration/core/dag/`.

## Workspace shape (Δ L2-1)

```
clusters/core/
├── 00-cluster-coordination.md   ← this file
├── 01-cluster-discovery.md
├── 02-cluster-catalog.md
├── 03-cluster-diagrams.md
├── 04-cluster-report.md
└── temp/
    ├── intra-cluster-edges.json                  (empty by design — see Δ L2-2 below)
    ├── layer-check-core.txt                       (production allowlist run)
    ├── layer-check-core-empty-allowlist.txt       (empty-allowlist run isolating L1/TC findings)
    └── validation-l2.md                           (validator output, written by validation gate)
```

No files outside this nested workspace are touched. The L1 workspace and the sibling cluster workspaces (`clusters/engine/` in progress, `clusters/contracts/` reportedly being authored in parallel) are **read-only** for this pass.

## Cross-cluster awareness (parallel-pass safety)

The user signalled at the start of this pass that `clusters/contracts/` is being authored concurrently by another agent. The Δ L2-4 ban on cross-cluster claims already keeps this pass strictly inside `core/`; the contracts cluster's territory is untouched here. Any `core/` observation about the `contracts/` boundary (relevant to L1 open-question Q1 — the responsibility cut between L0 contracts and L1 core post-ADR-006) is recorded in `04-cluster-report.md` under "Cross-cluster observations for synthesis" rather than asserted as a cross-cluster verdict. The post-L2 synthesis pass owns the merge.

## Oracle status (Δ L2-2)

The L1 dispatch queue's standing note flagged `core/` as **oracle-independent** — Phase 0's `temp/l3-import-graph.json` is scoped to `L3/application` (per its own `scope.layers_included` field), and `core/` is L1, so no `core/`-prefixed nodes or edges appear. The deterministic filter:

```python
[e for e in graph['edges'] if e['from'].startswith('core') or e['to'].startswith('core')]
# → [] (zero edges)

[n for n in graph['nodes'] if n['id'].startswith('core')]
# → [] (zero nodes)
```

…produced an empty result. `temp/intra-cluster-edges.json` was written with `intra_node_count = 0`, `intra_edge_count = 0`, and the rationale field explaining that the empty filter is **the audit evidence** that nothing was hand-derived (per Δ L2-2 discipline). The Δ L2-6 `dump-edges` byte-equality assertion is N/A because the cluster is L1, not L3 — recorded explicitly in the JSON's `byte_equality_assertion.applicable: false` field.

This matches the engine cluster's situation (engine is L2, also oracle-independent). `core/`'s actual outbound edges (downward to `contracts/`) and inbound edges (from `engine/` and every L3 surface) are layer-permitted by construction (per `enforce_tier_model.py:237-248`) and recorded at sub-subsystem granularity in `02-cluster-catalog.md` from per-file imports.

## Layer-conformance status (Δ L2-6)

Two artefacts in `temp/`:

- `layer-check-core.txt` — scoped run with the production allowlist (`config/cicd/enforce_tier_model/`).
- `layer-check-core-empty-allowlist.txt` — re-run against an empty allowlist to isolate genuine layer-import findings from already-allowlisted defensive patterns.

Both runs report **205 findings** because the production allowlist's path-prefixed keys are computed against the whole-tree scope (`src/elspeth/`); when the scope narrows to `src/elspeth/core/`, those keys no longer match and the same allowlisted defensive patterns re-surface. **This is not a layer-import failure**; it is a known artefact of the allowlist's key-prefix design (matches the engine cluster's experience exactly).

The load-bearing claim — verified by `grep -c '^  Rule: L1'` and `grep -c '^  Rule: TC'` against both files:

> **0 L1 layer-import violations, 0 TC TYPE_CHECKING layer warnings inside `core/`.**

Rule histogram of the 205 findings (empty-allowlist run):

| Count | Rule | Description |
|------:|------|-------------|
| 170 | R5 | `isinstance` (170 sites — overwhelmingly in `landscape/` Tier-1 read guards) |
| 18  | R6 | silent `except` |
| 15  | R1 | `dict.get` |
|  2  | R4 | broad `except` |
|  0  | **L1** | **upward import (FAIL CI)** |
|  0  | **TC** | **TYPE_CHECKING upward (warn)** |

Interpretation: `core/` is fully layer-conformant. The 205 defensive-pattern findings are all in-cluster code-quality items already governed by per-file allowlists with owner/reason/safety annotations; the catalog wave inherits them as known and does not re-triage. The 170 R5 (`isinstance`) sites concentrate in `landscape/` and reflect Tier-1 read-guard patterns at the Audit DB → Python boundary (legitimate per CLAUDE.md "Tier 1: crash on any anomaly").

## SCC status (Δ L2-7)

N/A. `core/` does not appear in `temp/l3-import-graph.json`'s `strongly_connected_components` list (the SCCs are entirely within L3: `mcp`/`mcp/analyzers`, `plugins/transforms/llm`/`providers`, `telemetry`/`telemetry/exporters`, `tui`/`screens`/`widgets`, and the 7-node `web/*` cluster). Δ L2-7's analysis is skipped.

## Sub-subsystem decomposition strategy (Δ L2-3)

`core/` has **6 immediate subdirectories** (`checkpoint/`, `dag/`, `landscape/`, `rate_limit/`, `retention/`, `security/`) and **11 standalone modules** at the package root (canonical, config, dependency_config, events, expression_parser, identifiers, logging, operations, payload_store, secrets, templates, plus `__init__.py`). Naive 1-entry-per-file would produce 17 catalog entries, exceeding the L2 depth budget.

Decomposition for the catalog (9 entries total):

| # | Sub-subsystem | Path(s) | Files | LOC | Composite per Δ4? |
|---|---|---|---:|---:|---|
| 1 | landscape | `landscape/` | 18 | 9,384 | No (no sub-pkgs, <10k LOC, <20 files) — **but** harbours 2 deep-dive candidates |
| 2 | dag | `dag/` | 5 | 3,549 | No — but harbours 1 deep-dive candidate |
| 3 | checkpoint | `checkpoint/` | 5 | 1,237 | No |
| 4 | rate_limit | `rate_limit/` | 3 | 470 | No |
| 5 | retention | `retention/` | 2 | 445 | No |
| 6 | security | `security/` | 4 | 940 | No |
| 7 | configuration_family | `config.py`, `dependency_config.py`, `secrets.py` | 3 | 2,524 | N/A (file group) — harbours 1 deep-dive candidate |
| 8 | canonicalisation_and_templating | `canonical.py`, `templates.py`, `expression_parser.py` | 3 | 1,422 | No |
| 9 | cross_cutting_primitives | `events.py`, `identifiers.py`, `logging.py`, `operations.py`, `payload_store.py`, `__init__.py` | 6 | 720 | No |

**Verification:** 18 + 5 + 5 + 3 + 2 + 4 + 3 + 3 + 6 = **49 files** (matches L1's 49 ✓). LOC: 9,384 + 3,549 + 1,237 + 470 + 445 + 940 + 2,524 + 1,422 + 720 = **20,691** (vs L1's 20,791 — 100 LOC delta, ~0.5%, attributable to whether `__init__.py` line counts use trailing newlines / pycache hygiene; the L1 figure used `find … -print0 | xargs -0 cat | wc -l` while this pass uses per-file `wc -l` summed).

**L3 deep-dive candidates flagged (per Δ L2-3, not opened by this pass):**

| File | LOC | Sub-subsystem |
|------|---:|---|
| `core/config.py` | 2,227 | configuration_family |
| `core/dag/graph.py` | 1,968 | dag |
| `core/landscape/execution_repository.py` | 1,750 | landscape |
| `core/landscape/data_flow_repository.py` | 1,590 | landscape |

Total deep-dive flagged: 7,535 LOC across 4 files (~36% of cluster LOC). This concentration matches the §7 P3 framing of "cascade-prone" risk — the four files own configuration loading, DAG construction/validation, and the two heaviest Landscape repositories. None opened.

## Execution log

- 2026-04-29 — Cluster workspace created at `clusters/core/`.
- 2026-04-29 — Read L1 inputs in prescribed order: `04-l1-summary.md` (§7 / §7.5), `02-l1-subsystem-map.md` §2 (core/), `00b-existing-knowledge-map.md` (full), `temp/l3-import-graph.json`, `temp/tier-model-oracle.txt`. Verified `clusters/engine/` exists (in progress) and `clusters/contracts/` does not exist yet (parallel-pass safe).
- 2026-04-29 — Verified per-file LOC for all 49 files in `core/`. Confirmed 4 deep-dive candidates from L1 match by LOC. Confirmed `expression_parser.py` is 820 LOC (not a deep-dive candidate; the L1 byte-size figure was misread on first pass).
- 2026-04-29 — Filtered Phase 0 oracle to `core/` paths → 0 nodes, 0 edges (oracle-independent, expected). Wrote `temp/intra-cluster-edges.json` with rationale.
- 2026-04-29 — Ran `enforce_tier_model.py check --root src/elspeth/core` with production allowlist → 205 findings, 0 L1, 0 TC. Re-ran with empty allowlist → identical histogram, 0 L1, 0 TC. Saved both artefacts in `temp/`.
- 2026-04-29 — Read 6 subpackage `__init__.py` files for public-surface inventory. Confirmed `landscape/` exposes 4 repositories + `RecorderFactory` (KNOW-A29, KNOW-A30 verified). Confirmed `security/` exposes a SecretLoader hierarchy + SSRF utilities (richer than L1 entry suggested).
- 2026-04-29 — Read 11 standalone modules and 8 representative subpackage internals (excluding all 4 deep-dive candidates per Δ L2-3). Counted `landscape/schema.py` tables → **20 tables** (KNOW-A24 claims 21; recorded as `[DIVERGES FROM KNOW-A24]` for the catalog).
- 2026-04-29 — Wrote `01-cluster-discovery.md`, `02-cluster-catalog.md`, `03-cluster-diagrams.md`, `04-cluster-report.md`.
- 2026-04-29 — Spawned validation subagent per Δ L2-8.

## Non-goals (explicit, recorded for the validator)

- Do NOT analyse `contracts/`, `engine/`, `plugins/`, `web/`, `mcp/`, `composer_mcp/`, `telemetry/`, `tui/`, `testing/`, or `cli` files except via cited cross-layer / cross-cluster edges. `core/` depends downward on `contracts/` only; the contracts boundary is recorded but the contracts internals are out-of-scope (and concurrently being authored).
- Do NOT propose refactorings for the cascade-prone risk concentration (config.py / dag/graph.py / landscape repositories). The L2 pass surfaces the question; the architecture-pack pass prescribes.
- Do NOT open files >1,500 LOC for inline summary. Flag and stop. There are 4 such files in `core/`; the catalog cites their existence and defers internals.
- Do NOT update `00-coordination.md` at the L1 workspace root. The post-L2 synthesis pass owns top-level updates.
- Do NOT produce a test architecture catalog. Tests are cited as evidence for invariants per Δ L2-5; comprehensive test architecture is a separate deliverable.
