# L2 #4 — `plugins/` cluster report

## Executive synthesis

The `plugins/` cluster is the **L3 plugin ecosystem**: 4 sub-subsystems, 97 Python files (98 with `__pycache__` artefacts as L1 §4 counted them), 30,391 LOC excluding bytecode (30,399 in L1 §4; the 1-file / 8-LOC delta is reconciled in `01-cluster-discovery.md §Sub-subsystem inventory`), and 29 distinct registered plugin classes (6 sources + 17 transforms + 6 sinks). It is the largest single subsystem in the codebase but **cleanly partitioned**, with one structural spine (`plugins/infrastructure/`) and three coordinated client packages (`sources/`, `transforms/`, `sinks/`). Its layer position is unambiguous: zero outbound L3↔L3 edges (it is a sink in the L3 graph), seven inbound L3↔L3 edges all targeting `infrastructure/` or `sources/`, and clean whole-tree layer-conformance per `enforce_tier_model.py check`. [CITES KNOW-A16] [CITES KNOW-A35] [CITES KNOW-C47]

The cluster's **defining structural commitment** is the F3 spine pattern: `plugins/sinks → plugins/infrastructure (w=45)` and `plugins/transforms → plugins/infrastructure (w=40)` are the heaviest single L3 edges in the entire codebase. This is structural, not preferential — the F3 reading-order amendment in §7.5 was an oracle-derived correction, not a stylistic choice. The audited HTTP and LLM clients (`infrastructure/clients/`) are the audit-trail wiring point that operationalises [CITES KNOW-C9] and [CITES KNOW-C10] at the plugin layer; the pluggy-based hookspec + folder-scan discovery (`infrastructure/hookspecs.py`, `infrastructure/discovery.py`) operationalises [CITES KNOW-C21] and [CITES KNOW-C22] (system-owned, not user-extensible).

**Trust-tier discipline is the cluster's most striking cultural artefact.** Every source file repeats verbatim "Sources use allow_coercion=True … This is the ONLY place in the pipeline where coercion is allowed." Every sink file repeats verbatim "Sinks use allow_coercion=False … Wrong types = upstream bug = crash." The contract is enforced by a config flag (`allow_coercion`) but reinforced by docstring text in every plugin module — a discipline beyond what the type system mandates. [CITES KNOW-C13] [CITES KNOW-C14] [CITES KNOW-C18] [CITES KNOW-C19] [CITES KNOW-C20]

The cluster's **single intra-cluster SCC** (`transforms/llm` ↔ `transforms/llm/providers`, 2 nodes) is the only structural irregularity. It is import-time module-level only (provider-registry pattern with deferred runtime instantiation), and its surfacing in this report is observational per Δ L2-7 — prescription is deferred to the architecture pack.

---

## §1. F3 verdict — the spine pattern is structural

The L1 §7.5 amendment F3 mandated an infrastructure-first reading order based on two oracle citations: `plugins/sinks → plugins/infrastructure (w=45)` and `plugins/transforms → plugins/infrastructure (w=40)`. This pass confirms F3 as **structural**, not preferential.

Edges into `plugins/infrastructure` (and its sub-packages `clients/`, `clients/retrieval/`, `batching/`, `pooling/`) account for **91.3% of intra-cluster edge weight** — 179 of 196 total intra-edge weight, computed by summing `weight` across `temp/intra-cluster-edges.json[intra_edges]` where `to` matches `plugins/infrastructure*`. No other coupling pattern is comparable.

Operational consequence: the `infrastructure/` catalog entry is read first, and client-side entries (sinks/, sources/, transforms/) cite spine patterns ("uses pluggy hookspec", "wraps via AuditedHTTPClient", "extends BatchTransformMixin") rather than re-deriving them. The serial reading order avoided ~3× re-reading of `infrastructure/`.

The architecture-pack-relevant question this raises but does not answer: **whether `infrastructure/` is too large to be a single sub-package** (10,782 LOC, 41 files, 3 sub-packages plus a nested one). Splitting `infrastructure/` would either require splitting the spine (which would multiply F3's structural pressure) or curating a public surface (`__all__` at `infrastructure/__init__.py`, currently empty) — both decisions for the architecture pack.

---

## §2. System-ownership invariant (KNOW-C21–C25)

The cluster honours the system-owned, not-user-extensible model in three layers:

1. **Discovery is filesystem-scoped, not entry-point-scoped** (`infrastructure/discovery.py:1-30`). Plugin discovery scans `src/elspeth/plugins/` for subclasses with a `name` attribute; there is no `setuptools.entry_points` fallback, no plug-in path env-var, no dynamic import of user-supplied paths. A user who clones ELSPETH and adds a new file under `plugins/transforms/` is contributing a system-owned plugin (subject to PR review, tests, etc.), not loading an external extension.
2. **Discovery uses `issubclass()` against base classes** (`infrastructure/base.py:7-15`). The base classes (`BaseSource`, `BaseTransform`, `BaseSink`) inherit machinery, not just a Protocol — `__init_subclass__` enforces self-consistency at class-definition time. This deliberately rules out `Protocol`-with-non-method-members because subclass detection on Protocols only works on instances, not classes [CITES KNOW-C25].
3. **`__init__.py` exports are deliberate** — `plugins/sinks/__init__.py`, `plugins/sources/__init__.py`, `plugins/transforms/__init__.py` all repeat: "Plugins are accessed via PluginManager, not direct imports." Import-time access to plugin classes is a contract violation the docstrings explicitly call out [CITES KNOW-C21]. The transforms package exposes a minimal `__all__` (`TypeCoerce`, `ValueTransform`) for testing but otherwise routes through the manager.

Plugin-bug-handling discipline (KNOW-C23–C25) is supported structurally but not test-locked at the cluster level: there is no test that asserts a plugin raising an unexpected exception causes the orchestrator to crash rather than swallow. That test, if it exists, lives in `engine/`'s test suite — a cross-cluster invariant test that asserts coercion-only-at-sources and crash-on-plugin-bug from the engine side would close the discipline.

---

## §3. Trust-tier discipline at the boundary (KNOW-C13–C20, KNOW-P6)

The single most-cited claim across the catalog is the trust-tier docstring repetition: every plugin module restates its trust-tier obligation at lines 5–10. This is a discipline I did not expect to find at this density, and it goes beyond schema-level enforcement.

Specific instances:

- **Sources** (Tier 3, allow_coercion=True): `csv_source.py:5-6`, `json_source.py:5-6`, `azure_blob_source.py:5-6`, `dataverse.py:6-7`, `text_source.py:6-7`, `field_normalization.py:7-9`. The phrase "ONLY place in the pipeline where coercion is allowed" is verbatim across 5 of the 6 source modules.
- **Sinks** (Tier 2 input, allow_coercion=False): `json_sink.py:5-7`, `csv_sink.py:5-7`, `database_sink.py:5-7`, `azure_blob_sink.py:5-12`. The phrase "Wrong types = upstream bug = crash" is verbatim across 4 of the 6 sink modules. `azure_blob_sink.py` extends with the Azure-SDK-vs-OUR-CODE breakdown — the most explicit Tier-3-boundary annotation in the cluster.
- **Transforms on row data** (Tier 2 input, allow_coercion=False): `value_transform.py:5-7`, `passthrough.py:5-7`. The phrase "catch upstream bugs / crashes immediately" repeats verbatim.

This discipline aligns directly with the trust-flow diagram in CLAUDE.md and the `tier-model-deep-dive` skill. The catalog flagged this as a **debt candidate** (no cross-cluster invariant test that asserts the documented contract holds at runtime); flagging is the appropriate L2 response — locking it in is an architecture-pack task.

---

## §4. SCC analysis — `transforms/llm` ↔ `transforms/llm/providers`

Per Δ L2-7: surface intent, defer prescription.

### 4.1 What the cycle is

A 2-node module-level cycle. `llm/transform.py:64-65` imports the concrete provider classes (`AzureLLMProvider`, `OpenRouterLLMProvider`) from `llm/providers/`; the provider modules import shared base classes (`LLMConfig`), the protocol (`LLMProvider`, `LLMQueryResult`, `FinishReason`, `parse_finish_reason`), shared validation (`reject_nonfinite_constant`), and tracing helpers from `llm/`. Eight specific reverse-edge import sites are enumerated in `01-cluster-discovery.md §SCC #1 evidence` (`providers/azure.py:23-25`, `providers/openrouter.py:35-37`).

### 4.2 Why the cycle exists

The intent is the **provider-registry pattern**: `llm/transform.py` is the unified transform that dispatches to provider implementations via a `_PROVIDERS` registry; providers are concrete strategy implementations that need access to the shared protocol surface. Concretely:

- The protocol (`LLMProvider`, `LLMQueryResult`, `FinishReason`) lives in `llm/provider.py` because it's the shared narrow interface.
- The config base (`LLMConfig`) lives in `llm/base.py` because it inherits from `infrastructure/config_base.TransformDataConfig`.
- Providers extend these — they cannot live above them.
- The transform aggregates providers — it cannot live below them.

The cycle is therefore **structurally minimal**: both sides need each other, and breaking it requires either (a) moving the shared surface (protocol + config) up into `infrastructure/`, (b) moving the provider classes up into `llm/`, or (c) introducing a registry indirection so the transform lazy-loads providers at runtime instead of at import time. All three have non-trivial trade-offs the architecture pack would weigh.

### 4.3 Is the cycle load-bearing?

There is one strong piece of evidence for "yes":

> "Provider instantiation is deferred to on_start() when recorder/telemetry become available. __init__ stores provider_cls + config for later use."
> — `transforms/llm/transform.py:9-13`

This means **runtime coupling is already decoupled** from import-time coupling. The cycle exists for module-level type sharing, not for runtime call-graphs. Decomposition could collapse the import cycle without changing the runtime behaviour — at the cost of moving shared types into `infrastructure/` (which already has 10,782 LOC and is under composite-at-L2 pressure).

The architecture pack will need to weigh: "is the `llm/` cycle worse than further bloating `infrastructure/`?"

### 4.4 What this pass does NOT prescribe

Per Δ L2-7: "Do NOT propose specific decomposition refactorings — that's an architecture-pack task, not an archaeology task." This report surfaces the cycle's structure (module-level only), its intent (provider-registry pattern), and the trade-off space (where shared types could move). It does not recommend a specific resolution.

---

## §5. Plugin-count drift — record, do not resolve

**Observed count: 29 distinct registered plugins** by `name = "..."` class-attribute scan. Method documented in `01-cluster-discovery.md §Plugin count`.

| Source | Claim | Drift from observed |
|---|---:|---:|
| KNOW-A35 (ARCHITECTURE.md §3.3) | 25 | +4 |
| KNOW-A72 (ARCHITECTURE.md Summary) | 46 | -17 |
| Observed | **29** | — |

[DIVERGES FROM KNOW-A35]: 4-plugin growth since the doc was written. Plausible additions in the +4: `rag_retrieval`, `azure_content_safety`, `azure_prompt_shield`, `openrouter_batch_llm` (these match the post-25-era plugin names visible in the registry).

[DIVERGES FROM KNOW-A72]: KNOW-A72's "46" is unsourced and inconsistent with KNOW-A35's per-category enumeration. The L1 standing note already flagged this as a doc-correctness pass requirement; this pass confirms the inconsistency exists in code-vs-doc, not in code-vs-code.

Doc correction is out of scope for this archaeology pass — flagging only.

---

## §6. Layer conformance verdict

| Check | Authority | Result |
|---|---|---|
| Whole-tree `enforce_tier_model.py check --root src/elspeth` | **Authoritative** | **Clean** ("No bug-hiding patterns detected. Check passed.") |
| Cluster-scoped `--root src/elspeth/plugins` | **Defensive-pattern scanner**, not layer-import | Exit 1 with 291 R-rule findings (R1=66, R2=6, R4=15, R5=140, R6=52, R8=3, R9=9). Zero L1 (layer-import) findings. |
| Δ L2-2 filter (`temp/intra-cluster-edges.json`) | Phase 0 oracle subset | 23 intra-edges, 12 nodes, 7 inbound XC, 0 outbound XC, 1 SCC touching |
| Δ L2-6 dump-edges (cluster-scoped) | Tool design limitation | 0 edges (tool emits inter-subsystem L3↔L3 edges; scope collapses to single subsystem) |
| Δ L2-6 dump-edges (whole-tree, filtered to plugin nodes) | Substantive determinism check | **23 plugin intra-edges, byte-equivalent with the Δ L2-2 filter** modulo `samples` and ordering |

**Verdict:** plugins/ is layer-conformant. The substantive determinism contract that Δ L2-6 was designed to enforce is satisfied via the whole-tree dump-edges run; the cluster-scoped run's empty output is a tool-level constraint, not a determinism break. Documented as such in `00-cluster-coordination.md §Δ L2-6 layer-check interpretation` so the validator can confirm.

---

## §7. Audited clients — the auditability locus

`plugins/infrastructure/clients/` (9 files, 3,790 LOC) is the audit-trail wiring point at the plugin layer. Three observations:

1. **`AuditedClientBase`** (`clients/base.py:127 LOC`) carries the `ExecutionRepository` reference, `state_id`, `run_id`, and `telemetry_emit` callback; it is the parent of `AuditedHTTPClient` and `AuditedLLMClient` and ensures every external call routes through the audit trail.
2. **SSRF-safe HTTP** (`clients/http.py:854 LOC`, lines 1–5): "Provides SSRF-safe HTTP methods via get_ssrf_safe() which uses IP pinning to prevent DNS rebinding attacks." This control sits below the audit envelope — security and audit are layered, not alternated.
3. **CallReplayer** (`clients/replayer.py:290 LOC`, lines 1–11): "matches calls by request_hash (canonical hash of request data), so the same request always returns the same recorded response." This is the operationalisation of [CITES KNOW-C10] (attributability test): an `explain(recorder, run_id, token_id)` query can re-execute the call deterministically against the audit trail.

These three together — audit envelope, security control below it, replay above it — are the cluster's most architecturally load-bearing micro-pattern. The L1 catalog's "highest-risk concern" for plugins/ was `azure_batch.py` (1,592 LOC); this pass agrees with that flag but suggests the *audited-clients* sub-package deserves equal architectural attention from the architecture pack — not for risk, but for the load-bearing role it plays.

---

## §8. Test coverage observations

164 unit tests + 19 integration tests. Layout mirrors source. Selected evidence anchors used in the catalog:

- `tests/unit/plugins/test_discovery.py` — discovery contract.
- `tests/unit/plugins/test_manager.py` — pluggy registration.
- `tests/unit/plugins/test_base_sink_contract.py`, `test_base_source_contract.py` — base-class invariants.
- `tests/unit/plugins/llm/test_provider_protocol.py`, `test_provider_lifecycle.py` — SCC #1 surface tests.
- `tests/property/plugins/test_schema_coercion_properties.py` — Hypothesis-based source-coercion property tests.

Test-debt candidates surfaced (all four explicitly): see `02-cluster-catalog.md §Closing — Test-debt candidates`. Three are cross-cluster invariants (allowlist coherence, coercion-only-at-sources, BaseAzureSafetyTransform exclusion) and would belong to a cross-cluster test pass; one is local (`field_normalization` corpus regression).

---

## §9. Cross-cluster observations for synthesis

(Per Δ L2-4: deferrals to the post-all-L2 synthesis pass. One-line each; no cross-cluster verdicts.)

- **`web/composer → plugins/infrastructure (w=22)`** is the heaviest cross-cluster inbound edge to plugins/. Synthesis owns: what does composer need from infrastructure that warrants a single edge of this weight?
- **`. (cli root) → plugins/infrastructure (w=7)` and `. → plugins/sources (w=2)`** confirm KNOW-P22 (cli registry pattern). Synthesis owns: whether the cli's import surface to plugins/ is a coupling worth re-reviewing post-ADR-006.
- **`testing → plugins/infrastructure (w=4)`** suggests the testing harness has hard imports into plugin spine. Synthesis owns: whether the harness should depend on protocols (`contracts/`) instead.
- **SCC #1 vs other L3 SCCs.** The plugin SCC #1 is one of five L3 SCCs (mcp/, plugins/transforms/llm/, telemetry/, tui/, web/). Synthesis owns: do they share a common cause (import-time registry pattern) or are they incidental?
- **R-rule findings density at the L3 boundary.** plugins/ has 291 cluster-scoped R-rule findings (R5=140, R6=52). Synthesis owns: cluster-by-cluster comparison of R-rule density may identify boundary-handling discipline gradients.

---

## §10. Highest-confidence claims

1. **plugins/ is layer-conformant and structurally clean.** Whole-tree `enforce_tier_model.py check` runs clean; intra-cluster edges (23) all flow toward `infrastructure/` (the spine); 0 outbound L3↔L3 edges; F3 reading-order verified empirically. **Confidence: High** — oracle-cited at every step, byte-equivalent on re-derivation.

2. **Trust-tier discipline is documented, repeated, and structurally encoded.** Every source module repeats the "ONLY place coercion is allowed" contract; every sink module repeats the "wrong types = upstream bug = crash" contract; the discipline is also encoded in the `allow_coercion` config flag. The contract is enforced at both layers. **Confidence: High** — verbatim docstring matches across plugin files, citable file:line.

3. **SCC #1 is module-level only and structurally minimal.** Provider-registry pattern with deferred runtime instantiation; both sides need each other for type sharing; the only decomposition options touch `infrastructure/` or introduce indirection. **Confidence: High** — import sites enumerated by file:line; runtime decoupling cited from `transform.py:9-13`.

---

## §11. Highest-uncertainty questions

1. **Is the SCC #1 cycle worth breaking, given that runtime coupling is already deferred?** The cycle is import-time only; the architecture pack will need to compare the cost of moving shared types into `infrastructure/` (further bloating an already-composite spine) versus the cost of leaving the cycle visible.

2. **Does the documented trust-tier discipline hold at runtime under all execution paths?** Verbal/structural enforcement is in place; cross-cluster invariant tests are not. A targeted runtime probe (e.g., a test fixture that injects a transform observed to coerce and asserts the run fails) would close the gap. This is in the test-debt list but its priority is uncertain.

3. **Is the 29-vs-25 plugin count a doc-rot artefact or a signal of governance drift?** Four post-doc plugins were added without a doc update. The architecture-pack pass should decide whether this is acceptable churn or whether plugin-count is a controlled invariant. KNOW-A72's "46" remains unexplained.

---

## §12. Cross-cluster observations for synthesis

(Repeated from §9 for the validator; deferred items only.)

- `web/composer → plugins/infrastructure (w=22)` — what does composer need? (Synthesis owns.)
- `. (cli root) → plugins/infrastructure / sources` — registry coupling per KNOW-P22; review post-ADR-006? (Synthesis owns.)
- `testing → plugins/infrastructure (w=4)` — should harness depend on `contracts/` instead? (Synthesis owns.)
- L3 SCC commonality — five L3 SCCs (mcp/, plugins/transforms/llm/, telemetry/, tui/, web/); shared cause? (Synthesis owns.)
- R-rule density gradient across clusters — boundary-handling discipline comparison. (Synthesis owns.)
