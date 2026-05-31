# Phase 9 Doc-Correctness Ground Truth

Recorded BEFORE editing any host document. Each entry resolves one of the L1-deferred tensions T1–T5 (`04-l1-summary.md` §4) with a re-verified value at validation time. Per Δ 9-3, a sample command is captured verbatim where the truth was derived from the live tree.

Re-measurement is intentional: doc-correctness passes routinely fail by replacing one stale number with another stale number. By re-deriving each ground truth at edit time, drift between the L1 finding and the current state of the repository is caught before it propagates into authoritative docs.

---

## T1 — Plugin-count drift

**Ground truth:** **29 plugins** are registered (6 sources + 17 transforms + 6 sinks) — derived by running the project's own `discover_all_plugins()` registry function, which is the same code path that powers `elspeth plugins list`. This is the count an auditor would see when listing the registered plugin surface.

**Diverges from L1:** the L1 deferral (§3 Q4) said "Per-category enumeration sums to 25" by adding ARCHITECTURE.md's claimed 4 sources + 13 transforms + 4 sinks + 4 audited clients. Re-measurement against `src/elspeth/plugins/` shows the per-category breakdown in ARCHITECTURE.md is itself stale (4→6 sources, 13→17 transforms, 4→6 sinks); the four AuditedHTTPClient/AuditedLLMClient/ReplayerClient/VerifierClient surfaces (KNOW-A39) are infrastructure, not Source/Transform/Sink plugins. Canonical count for the summary line is therefore the registered S/T/K total = 29, not 25 and not 46.

**Source-of-truth doc:** the live registry. The two contradictory ARCHITECTURE.md sites being reconciled are:
- Line 388: `**Total Plugin Ecosystem:** 25 plugins across 4 categories ...` (KNOW-A35)
- Line 986 (Summary > Key Metrics): `- Plugins: 46` (KNOW-A72)

**Command (verbatim):**

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from elspeth.plugins.infrastructure.discovery import discover_all_plugins
classes = discover_all_plugins()
for k in ['sources','transforms','sinks']:
    print(f'{k}: {len(classes[k])}')
total = sum(len(classes[k]) for k in ['sources','transforms','sinks'])
print('TOTAL S+T+K:', total)
"
```

**Output (verbatim):**

```
sources: 6
transforms: 17
sinks: 6
TOTAL S+T+K: 29
```

**Edit target(s):** ARCHITECTURE.md only. PLUGIN.md does not carry a plugin count (verified via `grep -nE 'plugin.*[0-9]+|[0-9]+ plugin' PLUGIN.md`). CLAUDE.md and AGENTS.md do not carry plugin counts.

---

## T2 — ADR-table staleness

**Ground truth:** there are **17 numbered ADRs** in `docs/architecture/adr/` (001–017), all in Accepted state per the ADR headers and KNOW-ADR-* index. ARCHITECTURE.md tabulates only ADR-001 through ADR-006 (KNOW-A62..KNOW-A68). The Summary key metrics line additionally claims "ADRs: 8" (line 987), which matches no count in the directory at any historical point.

**ADRs present in `docs/architecture/adr/` (verified by `ls docs/architecture/adr/`):**

```
000-template.md                                    (template, not an ADR)
001-plugin-level-concurrency.md
002-routing-copy-mode-limitation.md
003-schema-validation-lifecycle.md
004-adr-explicit-sink-routing.md
005-adr-declarative-dag-wiring.md
006-layer-dependency-remediation.md
007-pass-through-contract-propagation.md
008-runtime-contract-cross-check.md
009-pass-through-pathway-fusion.md
010-declaration-trust-framework.md
011-declared-output-fields-contract.md
012-can-drop-rows-contract.md
013-declared-required-fields-contract.md
014-schema-config-mode-contract.md
015-creates-tokens-contract.md
016-source-guaranteed-fields-contract.md
017-sink-required-fields-contract.md
README.md                                          (index, not an ADR)
```

**Delta to ARCHITECTURE.md table:** missing entries ADR-007, ADR-008, ADR-009, ADR-010, ADR-011, ADR-012, ADR-013, ADR-014, ADR-015, ADR-016, ADR-017 (eleven). The "ADRs: 8" key-metric line is also stale (real count: 17).

**Edit target(s):** ARCHITECTURE.md only. CLAUDE.md, AGENTS.md, and PLUGIN.md do not carry ADR tables (verified via `grep -nE 'ADR|adr'` returning empty for those three).

**Reconciliation policy (per Δ 9-4):** the table preserves chronological ordering; new rows for ADR-007..ADR-017 are appended in numeric order. Decision and Rationale columns adopt the one-line phrasings from the KNOW-ADR-* index where one exists; otherwise the ADR header sentence is used verbatim. The ADRs themselves are NOT modified.

---

## T3 — Schema-mode vocabulary drift

**Canonical vocabulary (chosen):** **`fixed`, `flexible`, `observed`**, with `parse` reserved for parse-error rows.

**Justification:**
- The runtime type is `mode: Literal["fixed", "flexible", "observed"]` at `src/elspeth/contracts/schema.py:421`.
- `from_dict()` at `src/elspeth/contracts/schema.py:457` rejects any other mode with a `ValueError`: `"'mode' is required for all schema configs (e.g. 'fixed', 'flexible', 'observed')"`.
- ADR-014 (Schema config mode contract, Accepted 2026-04-20, KNOW-ADR-014a) explicitly defines the contract over `(fixed, flexible, observed)` modes.
- `src/elspeth/contracts/schema_contract_factory.py:29`: `mode: Literal["fixed", "flexible", "observed"]`.
- `src/elspeth/contracts/audit.py:574`: comment `"fixed", "flexible", "observed", "parse"`.

**Alternatives seen in PLUGIN.md (both stale):**
- `dynamic` / `strict` / `free` — used in the **Schema Modes table** at PLUGIN.md:547–551 (KNOW-P23). None of these literals exist in the Python source as schema modes (`grep -rnE 'mode.*=.*"(dynamic|strict|free)"' src/elspeth --include='*.py'` returns no matches).
- `observed` / `fixed` / `free` — used in the **YAML examples block** at PLUGIN.md:553–574 (KNOW-P24). Two of the three (`observed`, `fixed`) are canonical; `free` is stale.

**Replacement map (one-to-one, value-preserving):**
- `dynamic` → `observed` (accept any fields / inferred from data)
- `strict` → `fixed` (only declared fields)
- `free` → `flexible` (declared required, extras allowed)

**Edit target(s):** PLUGIN.md only (the table at lines 547–551 and the YAML mode label at line 570). CLAUDE.md and ARCHITECTURE.md do not use the stale schema-mode vocabulary.

---

## T4 — ARCHITECTURE.md LOC drift

**Ground truth:** `src/elspeth/` is **121,408 LOC** across **359 Python files**, re-measured at validation time (`find src/elspeth -name '*.py' -print0 | xargs -0 cat | wc -l` → 121408; `find src/elspeth -name '*.py' | wc -l` → 359). The L1 discovery found 121,392 across 359 files; the 16-line drift between the L1 measurement and now is within rounding noise and is consistent with current ARCHITECTURE.md drift exceeding 17%.

**Command (verbatim):**

```bash
find src/elspeth -name '*.py' -print0 | xargs -0 cat | wc -l
find src/elspeth -name '*.py' | wc -l
```

**Output (verbatim):**

```
121408
359
```

**ARCHITECTURE.md currently states (three sites):**
- Line 20 (At a Glance): `~103,900 lines (315 Python, 46 TypeScript/TSX, 1 CSS)`
- Line 168 (after Container Responsibilities): `**Total Production LOC:** ~103,900 (315 Python + 47 frontend files)`
- Line 983 (Summary > Key Metrics): `- Production LOC: ~103,900 (315 Python + 47 frontend files)`

**Delta:** 121,408 / 103,900 = 1.169 → ~17% drift; file count 359 / 315 = 1.140 → ~14% drift. Frontend-file counts (TSX/CSS) are out of scope per L1 §6 (frontend deferred to a frontend-aware archaeologist) and are NOT re-measured here.

**Reconciliation policy (per Δ 9-4):** update only the Python LOC and Python file-count fields; preserve the existing frontend-file mention as-is (we have no replacement measurement for it). Round Python LOC to nearest hundred (`~121,400`) consistent with the doc's existing rounding convention. Cite the L1 discovery findings file inline at the first use, per Δ 9-4's "add citation when introducing a new number that drifts naturally" guidance.

**Edit target(s):** ARCHITECTURE.md only.

**Out of scope (deferred):** the per-container LOC table at ARCHITECTURE.md:153–164 (CLI ~2,200, TUI ~800, MCP ~3,600, Engine ~12,000, Plugins ~20,600, Landscape ~8,300, Telemetry ~1,200, Checkpoint ~600, Rate Limiting ~300, Core ~5,000, Contracts ~8,300) and the Component-tables LOC counts at lines 204–217 / 258–270. Each row would be an independent edit and a coordinated refresh would be wholesale rewriting (Δ 9-8(a) drift mode). Those go to the deferrals list. **Exception:** the `Testing` row on line 159 is updated as part of T5 (its description and LOC are inseparable from the conflation fix).

---

## T5 — KNOW-A18: testing/ ↔ tests/ conflation

**Conflation sites identified:**
- `ARCHITECTURE.md:159` — `| **Testing** | Python | ~9,500 | ChaosLLM, ChaosWeb, ChaosEngine test servers |` — describes test-suite content (`tests/chaos*`) under a row labelled "Testing" that maps to the production module `src/elspeth/testing/`. The L1 catalog (entry 10) verified `src/elspeth/testing/` is 877 LOC across 2 files and contains the `elspeth-xdist-auto` pytest plugin, NOT chaos servers.
- `CLAUDE.md:278` — `` `testing/` (chaosllm, chaosweb, chaosengine) `` in the Source Layout sentence — same conflation, abbreviated.

**Sites NOT carrying the conflation (verified by grep):**
- `AGENTS.md` — no `testing/` or `chaos` references in section headers; the file scope is operator/agent guidance.
- `PLUGIN.md` — no `testing/` or `chaos` references.

**Canonical distinction (one sentence form):**
> `src/elspeth/testing/` is production code that ships inside the `elspeth` package — currently the `elspeth-xdist-auto` pytest plugin (entry point in `pyproject.toml`) — and is distinct from the project's own `tests/` test suite (which is not part of the shipped package and is where the ChaosLLM / ChaosWeb / ChaosEngine test fixtures live).

**Note on prompt vs L1:** the Phase 9 prompt suggested clarification text describing `src/elspeth/testing/` as "chaos-injection utilities for downstream users." The L1 catalog (entry 10) verified this is incorrect — it is in fact the `elspeth-xdist-auto` pytest plugin (`pyproject.toml:[project.entry-points.pytest11]` registers `elspeth.testing.pytest_xdist_auto`). The canonical distinction above uses the L1-verified content, not the prompt's hypothesised content. Recording this divergence here (per Δ 9-3) so the validator can confirm the edit follows the verified L1 finding rather than the prompt's placeholder.

**Edit target(s):** ARCHITECTURE.md (line 159) and CLAUDE.md (line 278).

---

## Re-measurement summary

All five ground truths are derived from the live tree at the start of this pass, not from L1 numbers alone. Sample-verification by the validator (Δ 9-7) should re-run any one of the commands above and confirm the cited values still hold; mismatch indicates concurrent codebase change during the doc-correctness pass and is a STOP condition.
