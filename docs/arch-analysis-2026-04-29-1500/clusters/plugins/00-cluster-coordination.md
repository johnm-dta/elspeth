# L2 #4 — `plugins/` cluster coordination

## Configuration

- **Cluster name:** `plugins`
- **Scope:** `src/elspeth/plugins/`
- **Sub-cluster paths:** `src/elspeth/plugins/{infrastructure,sources,transforms,sinks}/`
- **Layer:** L3 (composite per L1 Δ4 — 4 sub-packages, 30,399 LOC, 98 files)
- **Effort bracket (per §7 / §7.5 Priority 4):** Large–Very Large (4–8 hr; or 4 parallel sub-passes)
- **Strategy:** **Serial, infrastructure-first** per F3 (`[ORACLE: edge plugins/sinks → plugins/infrastructure, weight 45 — heaviest single L3 edge]`). Parallelisation rejected because client-side catalog entries cite spine patterns from `infrastructure/`; parallel passes would each re-read `infrastructure/`, duplicating effort and introducing drift.
- **Oracle dependency:** APPLIES — `Δ L2-2` filter produced 23 intra-cluster edges, 12 nodes, 1 SCC.
- **Layer oracle (Δ L2-6):** APPLIES (cluster is L3) — both `check` and `dump-edges` runs executed.
- **SCC handling (Δ L2-7):** APPLIES — SCC #1 (`plugins/transforms/llm` ↔ `plugins/transforms/llm/providers`) is internal to this cluster.

## L1 inputs read (in mandated order)

| # | Input | Purpose |
|---|---|---|
| 1 | `04-l1-summary.md §7 / §7.5 / §F3 / standing note` | Dispatch queue authority + reading-order amendment |
| 2 | `02-l1-subsystem-map.md §4 plugins/` | L1 catalog entry to supplement |
| 3 | `00b-existing-knowledge-map.md` | KNOW-A35 (25), KNOW-A72 (46), KNOW-A16, KNOW-C13–C25, KNOW-P1–P33 |
| 4 | `temp/l3-import-graph.json` | L3 edge oracle (collapsed-to-subsystem; whole-tree) |
| 5 | `temp/tier-model-oracle.txt` | Layer-conformance status quo (whole-tree clean) |
| 6 | `ARCHITECTURE.md` (plugins section) | KNOW-A35 source; SDA model |
| 7 | `PLUGIN.md` | KNOW-P1–P33 source — plugin-author contract |

## Execution log

| Timestamp (approx) | Action | Outcome |
|---|---|---|
| 06:32 | Created cluster workspace `clusters/plugins/temp/` | OK |
| 06:32 | Δ L2-2: filtered L3 oracle to `plugins` prefix → `temp/intra-cluster-edges.json` | 23 intra-edges, 12 nodes, 7 inbound XC, 0 outbound XC, 1 SCC touching |
| 06:33 | Δ L2-6 part 1: `enforce_tier_model.py check --root src/elspeth/plugins` → `temp/layer-check-plugins.txt` | Exit 1 with **291 R-rule findings** (R1=66, R2=6, R4=15, R5=140, R6=52, R8=3, R9=9). Zero L1 (layer-import) findings. |
| 06:33 | Layer authoritative whole-tree run: `enforce_tier_model.py check --root src/elspeth` | "No bug-hiding patterns detected. Check passed." |
| 06:33 | Δ L2-6 part 2: `dump-edges --root src/elspeth/plugins --no-timestamp` → `temp/intra-cluster-edges-rederived.json` | 0 edges (tool emits inter-subsystem edges only; when scoped to a single L3 subsystem, intra-package edges are not L3↔L3) |
| 06:33 | **Determinism contract substitution.** Cluster-scoped `dump-edges` cannot reproduce 23 intra-plugin edges by tool design. Re-ran `dump-edges --root src/elspeth --no-timestamp` (whole-tree, mirrors Phase 0 oracle generation), filtered to plugin nodes. | **23 = 23 byte-equivalent edges** (modulo `samples` and ordering). SCC #1 identical in both runs. **Determinism contract substantively satisfied via the whole-tree run; cluster-scoped `dump-edges` is a tool limitation, not a determinism break.** |
| 06:33 | F3 reading: read `infrastructure/` first (16 files + 3 sub-pkgs); captured entry-point docstrings | OK; spine role confirmed — pluggy hookspecs, plugin manager, base classes, pooling, batching, audited clients |
| 06:34 | F3 reading: `sinks/` (7 files), `sources/` (8 files), `transforms/` (41 files) entry-point docstrings | OK; trust-tier discipline strings match KNOW-C13–C20 |
| 06:34 | Diagnosed SCC #1 import cycle: `llm/transform.py:64-65` imports providers; `llm/providers/{azure,openrouter}.py:23-25` import `llm/{base,provider,validation,tracing}` | Provider-registry pattern; surfacing only |
| 06:35 | Plugin count by `name = "..."` class attribute (`grep -rE '^    name\s*=' src/elspeth/plugins`) | **29** distinct registered plugins (6 sources + 17 transforms + 6 sinks); KNOW-A35 says 25 (drift +4); KNOW-A72 says 46 (drift -17). Recorded; no winner picked. |

## Δ L2-6 layer-check interpretation

| Artefact | Authority | Meaning |
|---|---|---|
| `temp/layer-check-plugins.txt` (exit 1, 291 R-rule findings) | **Defensive-pattern scanner output**, NOT layer-import enforcement. | The `--root src/elspeth/plugins` invocation produces allowlist-prefix mismatches (allowlist keys are project-root-relative; scoped paths drop the `plugins/` prefix) and reports R1–R9 *defensive-pattern* findings. R5=140 (`isinstance`) and R6=52 (`silent-except`) dominate, expected for an L3 boundary cluster. |
| Whole-tree `enforce_tier_model.py check --root src/elspeth` | **Authoritative — clean.** | "No bug-hiding patterns detected. Check passed." plugins/ has zero upward imports; layer conformance is intact. |
| `temp/intra-cluster-edges-rederived.json` (0 edges) | **Tool design limitation.** | `dump-edges` emits L3↔L3 edges between subsystems; with `--root src/elspeth/plugins`, the entire scope collapses to a single subsystem and no inter-subsystem edges exist. |
| Whole-tree `dump-edges --root src/elspeth --no-timestamp`, filtered to plugin nodes | **Substantive determinism check** for Δ L2-6's intent. | 23 plugin intra-edges; **byte-equivalent** with `temp/intra-cluster-edges.json` modulo `samples` and ordering. SCC #1 reproduced identically. |

## Cluster-specific priorities (per the approved specialisation block)

1. **Reading order discipline (F3).** `infrastructure/` first; clients second.
2. **System-ownership invariant** (KNOW-C21–C25): plugins are not user-extensible; pluggy is a clean-architecture mechanism.
3. **Trust-tier discipline at the boundary** (KNOW-C13–C20, KNOW-P6): coercion at sources only; transforms/sinks crash on type mismatch.
4. **SCC #1 root-cause documentation** (provider-registry pattern; surface intent, defer prescription per Δ L2-7).
5. **Plugin-count drift** (KNOW-A35 vs KNOW-A72): count and method recorded; no winner picked.
6. **Audited HTTP/LLM clients** (`infrastructure/clients/`) — the audit-trail wiring point for KNOW-C9 / KNOW-C10.

## Non-goals (per the approved specialisation block)

- Do NOT analyse `engine/`, `web/`, `mcp/`, `contracts/`, `core/`, `telemetry/` except via cited cross-cluster edges.
- Do NOT resolve KNOW-A35 vs KNOW-A72.
- Do NOT open `plugins/transforms/llm/azure_batch.py` (1,592 LOC) for inline summary — flag and stop.
- Do NOT propose decomposition refactorings for SCC #1 — surface intent only.
- Do NOT analyse `web/frontend/` — Python only.
- Do NOT extend the 4-layer model claim downward; intra-L3 edges are observational.

## Time budget

Target 4–8 hr; stop at >12 hr. Actual wall-clock to coordination plan complete: ~5 minutes (Δ L2-2 + Δ L2-6 + F3 reading + counting + tests enumeration). Likely ~1 hour total to all five deliverables — well under target.
