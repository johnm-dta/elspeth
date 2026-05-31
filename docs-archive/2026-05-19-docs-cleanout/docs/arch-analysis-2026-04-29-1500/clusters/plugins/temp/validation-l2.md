# L2 #4 — `plugins/` cluster validation report

**Verdict:** **APPROVE-WITH-MINOR**

The deliverables satisfy the V1–V10 cluster contract. All structural checks pass with primary-source evidence (oracle JSON, file:line citations, knowledge-map entries, layer-check tool output). Two cosmetic minors are listed at the end; neither blocks progression.

---

## V1. Sub-subsystem inventory parity — **PASS**

- Filesystem (`ls src/elspeth/plugins/`) yields exactly 4 immediate sub-directories: `infrastructure/`, `sources/`, `transforms/`, `sinks/` (`__init__.py` is a file, not a sub-subsystem). Catalog has 4 entries (one per dir); 1:1; no invented entries; no omissions.
- File counts and LOC verified by `find … -name '*.py' -not -path '*/__pycache__/*' | xargs wc -l`:
  - `infrastructure/`: 41 files, 10,782 LOC — matches catalog Entry 1 line 18 and discovery §Sub-subsystem inventory line 15.
  - `sources/`: 8 files, 3,519 LOC — matches Entry 3 line 100 and discovery line 16.
  - `transforms/`: 41 files, 12,575 LOC — matches Entry 4 line 139 and discovery line 17.
  - `sinks/`: 7 files, 3,515 LOC — matches Entry 2 line 62 and discovery line 18.
  - Total 97 files / 30,391 LOC (excluding pycache); discovery reconciles the 1-file/8-LOC delta vs L1 §4's "98 / 30,399" with the `__pycache__` + `plugins/__init__.py` explanation (line 21).

## V2. Oracle citation resolution — **PASS**

Spot-checked 8+ citations against `temp/intra-cluster-edges.json`:

- `stats.intra_edge_count = 23` ✓ (JSON line 8)
- `stats.outbound_cross_cluster_count = 0` ✓ (JSON line 11)
- `stats.sccs_touching_cluster = 1` ✓ (JSON line 12)
- `plugins/sinks → plugins/infrastructure (w=45)` ✓ — JSON entry at lines 28–50; cited as "heaviest single L3 edge" in catalog line 64, report §1 line 17, discovery line 58.
- `plugins/transforms → plugins/infrastructure (w=40)` ✓ — JSON lines 51–72; cited in catalog line 143, report line 17, discovery line 59.
- `plugins/sources → plugins/infrastructure (w=17)` ✓ — JSON lines 73–94; cited in discovery line 60.
- `plugins/transforms/llm/providers → plugins/infrastructure/clients (w=12)` ✓ — JSON lines 117–138; cited in discovery line 62.
- File:line citations resolved against actual source:
  - `transforms/llm/transform.py:64-65` — verified: lines 64–65 are `from elspeth.plugins.transforms.llm.providers.azure import …` and `… openrouter import …` (forward SCC edges).
  - `transforms/llm/providers/azure.py:23-25` — verified: line 23 imports `LLMConfig`, line 24 imports `FinishReason, LLMQueryResult, parse_finish_reason`, line 25 imports `AzureAITracingConfig, TracingConfig` (reverse SCC edges).
  - `infrastructure/base.py:7-15` — verified: docstring contains "Plugin discovery uses issubclass() checks against base classes" and "Python's Protocol with non-method members (name, determinism, etc.) cannot support issubclass()".
  - `infrastructure/clients/__init__.py:1-15` — verified: docstring is "Audited clients that automatically record external calls to the audit trail. These clients wrap external service calls (LLM, HTTP) and ensure every request/response is recorded to the Landscape audit trail for complete traceability."

## V3. Knowledge-map citation resolution — **PASS**

Confirmed all sampled `[CITES KNOW-…]` and `[DIVERGES FROM KNOW-…]` resolve in `00b-existing-knowledge-map.md`:

- KNOW-A16 (line 22), KNOW-A35 (line 41), KNOW-A72 (line 78), KNOW-C9/C10 (lines 90–91), KNOW-C13/C14 (lines 94–95), KNOW-C18/C19/C20 (lines 99–101), KNOW-C21/C22 (lines 102–103), KNOW-C25 (line 106), KNOW-C26 (line 107), KNOW-C30 (line 111), KNOW-C47 (line 128), KNOW-P3/P4/P5/P6/P7 (lines 178–182), KNOW-P22 (line 197) — all present.
- `[DIVERGES FROM KNOW-A35]` (catalog Entry 2 line 88; report §5 line 99): justified inline with "+4 drift since the doc was written" and plausible plugin enumeration.
- `[DIVERGES FROM KNOW-A72]` (report §5 line 101): justified as "unsourced and inconsistent with KNOW-A35's per-category enumeration"; the L1 standing note already flagged it for doc-correctness pass.
- KNOW-A72 itself in the map states "differs from 25 plugin total elsewhere" — divergence acknowledged at the knowledge-map source level, consistent with the cluster's "record, do not resolve" discipline.

## V4. Cross-cluster boundary respect — **PASS**

- Inbound cross-cluster edges (`web/composer → plugins/infrastructure (w=22)`, `cli → plugins/infrastructure (w=7)`, etc.) are presented as oracle facts only. No claims about composer's, cli's, or web/execution's internal structure appear in the catalog or report.
- Report §9 (lines 147–155) and §12 (lines 179–187) "Cross-cluster observations for synthesis" contain only one-line deferrals using "Synthesis owns: …" framing. No inline verdicts. The §9 web/composer entry asks "what does composer need from infrastructure that warrants a single edge of this weight?" — phrased as a synthesis-owned question, not a cluster-pass conclusion.
- Catalog Entry 1 line 22 (External coupling) explicitly enumerates inbound cluster names with weights but no commentary on those clusters' shape.

## V5. SCC handling (Δ L2-7) — **PASS**

- Catalog Entry 4 lines 166–170 contains a dedicated "SCC #1 — member coupling notation (per Δ L2-7)" subsection that explicitly marks both members:
  - Line 168: "`plugins/transforms/llm` is a **member of SCC #1** with `plugins/transforms/llm/providers`. Acyclic decomposition not possible at L3↔L3 layer."
  - Line 169: "`plugins/transforms/llm/providers` is a **member of SCC #1** with `plugins/transforms/llm`. Acyclic decomposition not possible at L3↔L3 layer."
- `04-cluster-report.md §4` (lines 53–86) covers all four required elements: §4.1 cycle structure with file:line for both directions; §4.2 intent (provider-registry pattern); §4.3 trade-off space (three decomposition options weighed); §4.4 explicit non-prescription per Δ L2-7 ("This report surfaces the cycle's structure …, its intent …, and the trade-off space …. It does not recommend a specific resolution.").
- `03-cluster-diagrams.md §3` (lines 103–146) reproduces the SCC structure with a complete file:line table for all 8 import sites (3 azure.py reverse + 2 openrouter.py reverse + 2 transform.py forward + 1 openrouter.py reverse for `reject_nonfinite_constant`).

## V6. Layer-check oracle interpretation — **PASS**

The four documents distinguish the four artefacts consistently:

- **Authoritative whole-tree clean:** verified by `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth …` returns "No bug-hiding patterns detected. Check passed." — cited in coordination line 34 / 47, discovery line 116, report §6 line 111.
- **Cluster-scoped 291 R-rule findings:** `temp/layer-check-plugins.txt` is 4,814 lines; first violation lines confirm R5 isinstance/R6 silent-except findings. Coordination line 33 explicitly classifies these as "defensive-pattern scanner output, NOT layer-import enforcement" (R1=66, R2=6, R4=15, R5=140, R6=52, R8=3, R9=9). Same framing in discovery line 117 and report §6 line 112.
- **Cluster-scoped dump-edges = 0 edges:** verified by reading `temp/intra-cluster-edges-rederived.json` lines 1–103: `"edges": []`, `"total_edges": 0`. Coordination line 35 documents this as "tool emits inter-subsystem edges only; when scoped to a single L3 subsystem, intra-package edges are not L3↔L3" — i.e., a tool design limitation, not a determinism break.
- **Whole-tree dump-edges filtered to plugin nodes = 23 byte-equivalent edges:** documented in coordination line 36 ("23 = 23 byte-equivalent edges modulo `samples` and ordering. SCC #1 identical in both runs."). The substitution rationale is explicit: "Cluster-scoped `dump-edges` cannot reproduce 23 intra-plugin edges by tool design."
- Coordination §Δ L2-6 layer-check interpretation table (lines 42–49) consolidates all four with authority labels; report §6 mirrors this verdict.

## V7. Depth-cap compliance — **PASS**

- `azure_batch.py` (1,592 LOC) appears in 5 places across the documents:
  - Discovery line 17 (table flag), catalog line 207 (deep-dive table), catalog line 181 ("flagged, not opened"), catalog line 175 (test enumeration only), report §7 line 129 (referenced as L1 risk concern).
  - **No file body content is summarised inline; no line citations beyond the flag.** Compliant.
- `infrastructure/base.py` (1,159 LOC, under threshold but largest in `infrastructure/`):
  - Cited at `:7-15` (issubclass docstring, line 27 of catalog) and `:21-35` (lifecycle docstring, line 29) — both within the first 30 lines and confirmed by reading the actual source (lines 1–20 are the docstring; lifecycle continues from line 20). Compliant.
- Other large files cited at body level only via header docstrings (`clients/http.py:3-5`, `clients/replayer.py:1-11`, `pooling/executor.py:1-12`, `clients/__init__.py:1-15`) — all entry-point docstrings, no function-body inline summary.

## V8. Closing sections present (Δ L2-10) — **PASS**

`04-cluster-report.md` ends with the three required sections, all with substantive content:

- §10 "Highest-confidence claims" (lines 159–165): 3 numbered items, each with confidence label and oracle/file:line evidence.
- §11 "Highest-uncertainty questions" (lines 169–175): 3 numbered items, each framed as an open question with reasoning.
- §12 "Cross-cluster observations for synthesis" (lines 179–187): 5 deferrals (matches the "5 items" in the contract description), each one-line, none making cross-cluster verdicts.

## V9. Test-debt candidate count — **PASS**

- Catalog Closing §Test-debt candidates (lines 213–219) lists **5** candidates: (1) allowlist coherence in clients/pooling, (2) stable contract surface for infrastructure (no `__all__`), (3) per-version corpus regression for `field_normalization`, (4) cross-cluster invariant for "coercion only at sources", (5) negative test for `BaseAzureSafetyTransform` not in registry. All five match the contract's enumeration.
- Report §11 question 2 (lines 173) surfaces the cross-cluster invariant test-debt candidate ("Does the documented trust-tier discipline hold at runtime under all execution paths? … cross-cluster invariant tests are not [in place]") — satisfies "at least one in §11 as Highest-uncertainty question".
- Report §8 line 143 explicitly cross-references "see `02-cluster-catalog.md §Closing — Test-debt candidates`" and labels three as cross-cluster invariants and one as local. Bidirectional linkage intact.

## V10. Plugin-count discipline — **PASS**

- Re-ran `grep -rE '^    name\s*=' src/elspeth/plugins --include='*.py' | wc -l` → **29**. Matches every claim in the documents (coordination line 40, discovery line 27, report line 5 / §5 line 91, catalog Entry 4 line 139).
- Counting method documented at discovery line 25: "`grep -rE \"^    name\\s*=\" src/elspeth/plugins --include='*.py'` for `name = \"...\"` class attributes — the registration key used by `infrastructure/discovery.py` and `manager.py` to register a class as a discoverable plugin." Method is reproducible.
- Both KNOW-A35 (25) and KNOW-A72 (46) are flagged with `[DIVERGES FROM …]` markers AND justified inline (discovery lines 37–38; report §5 lines 99–101).
- **No claim** in any document picks a winner. Coordination line 40 ("Recorded; no winner picked"), discovery line 39 ("flagging without resolving, per the L1 dispatch instruction 'do not resolve, just flag'"), report §5 line 103 ("Doc correction is out of scope for this archaeology pass — flagging only.") — discipline is consistent.

---

## Minor findings

1. **Cosmetic — Report §1 weight tally is approximate but slightly understates dominance.** Report line 19 says "edges into `plugins/infrastructure` … account for ~85% of intra-cluster edge weight (181 of ~213 total intra-edge weight by my crude tally)". Recomputed from the JSON: actual total intra-edge weight is **196** (not ~213), and weight to `plugins/infrastructure*` (including all sub-packages) is **179** (not 181). The corrected ratio is 179/196 ≈ **91%**. The directional claim ("the spine pattern dominates") is empirically stronger than the report says, not weaker; the inline "by my crude tally" hedge accurately flags imprecision. **Severity: cosmetic.** Discovery line 69 makes the same "~85%" claim with no number breakdown — same issue, same severity. Fixing would require a one-sentence edit in each.

2. **Cosmetic — Executive synthesis claims "97 Python files, 30,391 LOC" but L1 says 98 / 30,399.** Report line 5 uses the pycache-excluded counts (97 / 30,391), while L1 §4 uses pycache-inclusive counts (98 / 30,399). Discovery line 21 explicitly reconciles the discrepancy with a justification ("L1 counted `__pycache__/` artefacts; this pass excludes them. The 1-file / 8-LOC delta corresponds to `plugins/__init__.py` plus pycache exclusions and is non-substantive."). The reconciliation is correct; the executive synthesis just doesn't repeat it. **Severity: cosmetic** — readers who jump straight to the report without reading discovery may be momentarily confused; not blocking.

Neither minor is structural. Neither blocks progression. The documents satisfy V1–V10 in full.

## Recommendation

**APPROVE-WITH-MINOR** — Proceed to next cluster / synthesis. The two cosmetic minors above can be addressed in a passing edit during synthesis or left as-is (the directional claims are correct; the numbers are off in a non-misleading direction).
