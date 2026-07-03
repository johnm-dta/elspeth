# Specification: Formalising the Pipeline Compiler as a First-Class Boundary

**Status**: Proposed (analysis captured; no implementation started)
**Date**: 2026-07-04
**Author**: Claude (session analysis for John Morrissey)
**Related**: filigree elspeth-896fb00e37 (engine facade exports low-level execution plumbing)
**Evidence baseline**: `release/0.7.0` @ `93cbffffb`

## 1. Summary

ELSPETH already separates pipeline *compilation* (config → plugins → graph →
validated `PipelineConfig`) from *execution* (`Orchestrator.run()`), and the
composer front end already performs **real compilation** through the production
engine path via `POST /api/sessions/{id}/validate`. What is missing is not
capability but **form**: compilation has no name, no single entry point, and no
reusable artifact. The same compile sequence is hand-orchestrated in three call
sites, runs up to three times per execute, and cross-site agreement is enforced
behaviourally by a ~4,000-line parity test file rather than structurally by
shared code.

This spec captures the as-is analysis (§3), the gaps (§4), and a proposed
refactor (§5): introduce a `CompiledPipeline` contract and a single
`compile_pipeline(settings, *, mode)` entry point, make all three call sites
consume it, and stamp a compile fingerprint into the Landscape run record so a
run carries provenance that it executed exactly the configuration that passed
validation.

## 2. Motivation

Operator framing (2026-07-04): *"evaluate the possibility of separating the
'compiler' in the engine from the executor so the front end can do real
compilation into a better validated run."*

Concretely, the desired end state is:

1. The front end's validate action **is** the compiler, not an approximation of
   it (already true — see §3.2).
2. A run is verifiably the execution of a compiled-and-validated artifact
   ("validated run" provenance — not true today; the validated result is
   discarded and the run recompiles from YAML).
3. The compiler is one code path with one owner, so composer/runtime agreement
   is guaranteed by construction instead of by parity tests.

## 3. Current architecture (as-is analysis)

### 3.1 The compile pipeline

Compilation today is this stage sequence:

| # | Stage | Owner (layer) | Location |
|---|-------|---------------|----------|
| 1 | Composer state → YAML | web (L4) | `YamlGenerator.generate_yaml()` |
| 2 | YAML → `ElspethSettings` | core (L1) | `load_settings_from_yaml_string()` (`src/elspeth/core/config.py`) |
| 3 | Settings → `PluginBundle` | plugins (L3) | `instantiate_plugins_from_config()` (`src/elspeth/plugins/infrastructure/runtime_factory.py:55`) |
| 4 | Bundle → `ExecutionGraph` | core (L1) | `ExecutionGraph.from_plugin_instances()` → `build_execution_graph()` (`src/elspeth/core/dag/builder.py:116`) |
| 5 | Structural + schema validation | core (L1) | `graph.validate()` (`src/elspeth/core/dag/graph.py:282`) + `graph.validate_edge_compatibility()` |
| 6 | Assembly + route-target validation | engine (L2) | `assemble_and_validate_pipeline_config()` (`src/elspeth/engine/orchestrator/preflight.py:62`) |

Supporting facts:

- **Preflight mode.** Stage 3 takes `preflight_mode: bool`. Under
  `preflight_mode=True`, plugin constructors observe
  `plugin_preflight_mode_enabled()` (context manager
  `plugin_preflight_mode` in `src/elspeth/plugins/infrastructure/preflight.py`)
  and may defer side-effectful client setup. Known observers: `json_sink.py`,
  `csv_sink.py`, `clients/retrieval/connection.py`. This is what makes
  compile-without-side-effects possible.
- **Value-source compliance** (catalog membership, sibling derivation) is
  enforced during stage 3 inside `instantiate_plugins_from_config`; the
  per-node variant `check_config_value_sources()`
  (`engine/orchestrator/preflight.py:231`) exists for the composer's
  pre-wiring option prevalidation (authoring-time catalog checks, e.g. unknown
  OpenRouter model → `list_models` hint).
- **Stage 6 is deliberately pure and layer-constrained.** The module docstring
  (`engine/orchestrator/preflight.py:1-24`) records the contract: engine (L2)
  cannot import the L3 plugin factory, so it takes primitives; the four route
  validators are pure (no I/O, no input mutation); the orchestrator re-runs
  them at run-init and "the second call either passes again or raises the same
  error." The one intentional mutation: aggregation transforms get `node_id`
  assigned here.

### 3.2 The three call sites (and the triple compile)

**Call site A — composer validate** (`POST /api/sessions/{id}/validate`,
route at `src/elspeth/web/execution/routes.py:813`):
`validate_pipeline()` (`src/elspeth/web/execution/validation.py:710`, spans
699–2049 ≈ 1,350 lines). Docstring: *"Dry-run validation through the real
engine code path."* Sequence:

1. Empty-composition short-circuit (`empty_pipeline` result).
2. Source/sink/nested-provider path allowlist checks (C3/S2 defense-in-depth,
   `web/paths.py` helpers).
3. Secret-ref existence validation (refs resolved the same way the execution
   service resolves them).
4. Blob metadata checks (validate-time metadata only; content reads stay in
   the execution preflight).
5. Stages 1–6 of §3.1 with `preflight_mode=True`, via the thin wrappers
   `instantiate_runtime_plugins()` / `build_runtime_graph()`
   (`src/elspeth/web/execution/preflight.py:189,194`).
6. Translation of expected failure classes (`PydanticValidationError`,
   `ValueError`/`TypeError` at settings load, `PluginNotFoundError`,
   `PluginConfigError`, `FileExistsError`, `GraphValidationError`,
   `RouteValidationError`, `ValueSourceValidationError`) into a structured
   `ValidationResult`; everything else propagates as a 500 (Tier-1 invariant
   break, W18). The whole function is a declared `@trust_boundary(tier=3, …,
   non_raising=True)`.

**Call site B — execute** (`src/elspeth/web/execution/service.py`):

1. `service.py:659–677` — re-runs **the same dry-run `validate_pipeline`**
   (call site A's function) as a gate *before* `create_run`, so a run row is
   never created for a config that fails compile. 422 mirrors the `/validate`
   payload shape (`routes.py:1011-1015`).
2. `service.py:680` — regenerates YAML from the composition state.
3. `service.py:1261–1273` — `load_settings_from_yaml_string()` →
   `build_validated_runtime_graph(settings)`
   (`web/execution/preflight.py:208`: stages 3–5 with
   `preflight_mode=False`) → `assemble_and_validate_pipeline_config()`
   (stage 6). Result handed to the Orchestrator.

**Call site C — run-init inside the engine**
(`src/elspeth/engine/orchestrator/core.py`):

- `Orchestrator.run()` (`core.py:503`) **requires** a pre-built graph and
  raises `OrchestrationInvariantError` if it is absent (`core.py:546-547`):
  the executor already refuses to compile.
- Re-runs the four route validators a third time (`core.py:952–980`); the
  resume path has an equivalent site.
- Then performs the run-only work: `run_transform_runtime_preflights()`
  (`core.py:1088` → `engine/orchestrator/runtime_preflight.py`) — live,
  per-transform external-readiness checks recorded as audited operations.
  That module's docstring explicitly separates itself from the static
  preflight: *"Same word, different concern — they intentionally do not share
  a module."*

Net effect: one execute compiles the pipeline **three times** (A as gate, B
for real, C's validators as invariant recheck) and validates routes three
times. A and B additionally regenerate YAML independently.

### 3.3 How agreement is enforced today

- Shared helpers: `engine/orchestrator/preflight.py` (stage 6) and
  `web/execution/preflight.py` (stages 3–5 wrappers) are used by both A and B.
- Everything not shared (the ordering, the error translation, the
  YAML-regeneration, path resolution via `resolve_runtime_yaml_paths()`) is
  mirrored by hand and pinned by
  `tests/integration/pipeline/test_composer_runtime_agreement.py`
  (3,969 lines) — parity tests asserting composer validation and the runtime
  path reach the same verdicts (route targets, file-sink collisions,
  fixed-mode implicit-required fields, …).

### 3.4 Existing artifact shapes (and why none is THE artifact)

- `PluginBundle` (L3): live plugin instances; holds clients/handles.
- `ExecutionGraph` (L1): mutable runtime state.
- `RuntimeGraphBundle` (`web/execution/preflight.py:42`): explicitly
  *"Transient runtime setup result… not persisted and should not cross
  request boundaries."*
- `PipelineConfig` (engine types): assembled per-run, not fingerprinted.
- `runtime_preflight_settings_hash()` (`web/execution/preflight.py:175`):
  precedent for a deliberate, secret-free settings hash — currently covers
  only `data_dir`.

## 4. Gap analysis

| # | Gap | Consequence |
|---|-----|-------------|
| G1 | No `CompiledPipeline` artifact; each phase compiles and discards | "Validated" is not a property a run can prove; the proof is "the same code ran again" |
| G2 | No single `compile()` entry point; stages hand-wired in A and B | Drift risk lives in the unshared orchestration (ordering, error translation, YAML regeneration) |
| G3 | `validate_pipeline` is a 1,350-line web-layer monolith mixing composer-only checks, engine compile, and result translation | The compile core is not extractable/reusable without carving it out of this function |
| G4 | Parity by test, not construction (3,969-line agreement suite) | Ongoing maintenance tax; every new validator needs a new parity test |
| G5 | No compile fingerprint on the run record | Landscape audit cannot assert "run X executed exactly the config that passed validation Y"; TOCTOU window between validate and execute is invisible |
| G6 | Layering: L2 engine cannot import the L3 plugin factory | A full `compile_pipeline()` cannot live in `engine/`; the contract and the factory must be split across layers |

## 5. Proposed design

### 5.1 `CompiledPipeline` contract (engine-owned, L2)

A frozen dataclass in `engine/` (contract only — no L3 imports):

```python
@dataclass(frozen=True, slots=True)
class CompiledPipeline:
    fingerprint: str            # sha256 of canonical resolved YAML/settings (secret-free)
    mode: CompileMode           # PREFLIGHT | RUNTIME
    graph: ExecutionGraph       # built + validated (stages 4–5 passed)
    pipeline_config: PipelineConfig  # stage 6 output (validators passed)
    settings: ElspethSettings
    report: CompileReport       # structured checks/warnings (feeds ValidationResult)
```

Rules:

- **Never persisted, never crosses a request boundary** (inherits
  `RuntimeGraphBundle`'s rule — plugin instances hold live resources). What
  crosses boundaries is the `fingerprint` and the `report`.
- `mode=PREFLIGHT` artifacts are not executable; `Orchestrator.run()` rejects
  them (`OrchestrationInvariantError`). Only `mode=RUNTIME` artifacts run.
- The fingerprint is computed from the canonical resolved YAML (post
  `resolve_runtime_yaml_paths`, post secret-ref resolution *markers* — never
  secret values), extending the `runtime_preflight_settings_hash()` precedent.
  `core/canonical.canonical_json` already exists for canonicalisation.

### 5.2 `compile_pipeline()` entry point (L3)

Because of G6, the function that includes plugin instantiation lives at L3+
(proposed: `src/elspeth/plugins/infrastructure/compile.py`, or a new
top-level `src/elspeth/compile.py` facade that depends downward on L3 and L2):

```python
def compile_pipeline(
    settings: ElspethSettings,
    *,
    mode: CompileMode,
) -> CompiledPipeline:
    """Stages 3–6 of the compile pipeline as one owned sequence."""
```

Behaviour: instantiate plugins (`preflight_mode = (mode is PREFLIGHT)`) →
build graph → `graph.validate()` + `validate_edge_compatibility()` →
`assemble_and_validate_pipeline_config()` → assemble `CompiledPipeline` with
fingerprint + report. Raises the same typed errors the stages raise today;
does **not** translate them (translation stays a web concern).

Stages 1–2 (state → YAML → settings) remain caller-side: the CLI compiles
from a YAML file, the web compiles from `CompositionState`. Both converge on
`ElspethSettings` as the compiler input — that is the correct compiler
boundary.

### 5.3 Call-site changes

- **A (validate)**: `validate_pipeline` keeps its composer-specific pre-checks
  (path allowlist, secret refs, blob metadata, interpretation placeholders)
  and its Tier-3 error-translation contract, but replaces its hand-wired
  stage 3–6 block with `compile_pipeline(settings, mode=PREFLIGHT)` and
  renders `ValidationResult` from `CompiledPipeline.report` + caught typed
  errors. Target: the function shrinks to pre-checks + one compile call +
  translation.
- **B (execute)**: replaces `build_validated_runtime_graph` +
  `assemble_and_validate_pipeline_config` with
  `compile_pipeline(settings, mode=RUNTIME)`; passes the artifact to the
  orchestrator; records `fingerprint` on the run. The pre-`create_run` dry-run
  gate (§3.2 B1) is **retained** — it is what returns a structured 422 before
  a run row exists — but becomes the same one-liner compile in PREFLIGHT mode.
- **C (orchestrator)**: `Orchestrator.run()` gains an overload/param accepting
  `CompiledPipeline` (it already takes `config` + `graph` separately, so this
  is packaging, not new capability — see `TestOrchestratorAcceptsGraph`).
  The run-init re-validation (`core.py:952–980`) is **kept** as cheap
  invariant assertions (the validators are pure; the docstring contract in
  `preflight.py:17-23` already frames the second call as idempotent
  insurance). Rationale: TOCTOU — validate/compile and run are separated in
  time even within one request.
- **Landscape provenance (G5)**: the run record gains
  `compile_fingerprint` (and optionally `validated_at` + the id of the
  validation event that last passed with the same fingerprint). This is
  audit-doctrine-consistent: the fingerprint captures user-selected run
  config, not system config.

### 5.4 What stays runtime-only (explicit non-goals)

Compile time can never absorb, and this spec does not attempt to move:

- `transform.runtime_preflight(ctx)` — live external readiness, audited per
  node (`engine/orchestrator/runtime_preflight.py`), run-only **by design**.
- Secret **values** (validate checks refs exist; values resolve at run).
- Source data availability and content; blob content reads.
- File-sink write collisions at the moment of write (`fail_if_exists` /
  `auto_increment` races).
- Commencement gates and dependency runs (`engine/commencement.py`,
  `engine/orchestrator/preflight` consumers in `bootstrap_and_run`).
- TOCTOU generally: validate and execute are separate HTTP requests; the
  filesystem, catalogs, and secrets can change in between. The fingerprint
  makes the window *visible*; the run-init recheck keeps it *safe*.

"Better validated run" therefore means: byte-identical config provenance +
structurally guaranteed composer/runtime agreement — not "no run-time
failures."

## 6. Alternatives considered

### 6.1 Do nothing (status quo)

**Pros**: The capability the operator asked about substantially exists; parity
tests currently hold the line.
**Cons**: G1–G5 persist; every new validator or compile stage must be wired in
two places and parity-tested; no provenance.
**Verdict**: Rejected — the maintenance tax (G4) grows with every feature.

### 6.2 Persist/cache the compiled artifact between validate and execute

Reuse the actual `CompiledPipeline` object from `/validate` when the user hits
execute.
**Pros**: True "run what you validated"; skips one compile.
**Cons**: Violates the live-resource rule (plugin instances hold clients and
handles; `RuntimeGraphBundle` already forbids crossing request boundaries);
preflight-mode instances are deliberately *not* runtime instances; introduces
cache-invalidation and session-lifetime problems; TOCTOU risk gets **worse**
(stale artifact runs against a changed world).
**Verdict**: Rejected. Reuse the *verdict* (fingerprint), never the
*instances*. This is the load-bearing design constraint.

### 6.3 Move compilation fully into `engine/` (L2)

**Pros**: One obvious home; "the compiler in the engine" reads naturally.
**Cons**: Impossible without breaking the layer rule — stage 3 requires the
L3 plugin factory (`preflight.py:8-10` documents the constraint).
**Verdict**: Rejected as stated; achieved instead by splitting contract (L2)
from factory (L3), which is the same pattern stage 6 already uses.

### 6.4 Skip the orchestrator's run-init re-validation once compile is unified

**Pros**: Removes the third validation pass.
**Cons**: The validators are pure and cheap; deleting the recheck trades real
TOCTOU insurance for negligible savings; resume paths depend on the same site.
**Verdict**: Rejected — keep as invariant assertions (per §5.3 C).

## 7. Consequences

### Positive

- Composer/runtime agreement becomes structural; the agreement test suite
  shrinks toward testing `compile_pipeline` once.
- `validate_pipeline` drops from ~1,350 lines to pre-checks + translation.
- Landscape gains "validated run" provenance (fingerprint chain), a genuine
  audit-story improvement for a system whose selling point is auditability.
- Chips at elspeth-896fb00e37: a public `compile_pipeline` + `CompiledPipeline`
  is the right exported surface, letting low-level plumbing exports retreat.
- CLI and any future headless callers get real compile-only validation for
  free (`elspeth compile --check` becomes trivial).

### Negative / costs

- Carving the compile core out of `validate_pipeline` without disturbing its
  `@trust_boundary` Tier-3 translation contract is delicate; the W18 rule
  (unexpected exceptions must still propagate to 500) must survive the move.
- A new public contract (`CompiledPipeline`, `CompileMode`, `CompileReport`)
  to keep stable.
- Fingerprint definition must be chosen carefully once (canonical form,
  secret-marker handling, path-resolution ordering) — changing it later
  invalidates stored provenance.
- Small blast radius in tests: agreement tests, orchestrator init tests, and
  execution-service tests all touch the seam.

## 8. Implementation sketch (proposed slices)

1. **Slice 1 — contract + function, no callers change.** Add
   `CompileMode`/`CompileReport`/`CompiledPipeline` (L2 contract) and
   `compile_pipeline()` (L3), implemented by delegating to the existing
   stages. Unit-test it directly against the cases the agreement suite covers.
2. **Slice 2 — execute path.** Switch `service.py:1261–1273` to
   `compile_pipeline(mode=RUNTIME)`; thread the artifact into
   `Orchestrator.run(compiled=…)`; keep run-init assertions.
3. **Slice 3 — validate path.** Replace `validate_pipeline`'s stage 3–6 block
   with `compile_pipeline(mode=PREFLIGHT)`; render `ValidationResult` from the
   report; delete now-redundant hand-wiring.
4. **Slice 4 — provenance.** Add `compile_fingerprint` to the Landscape run
   record (schema epoch bump → sessions/audit DB wipe is the accepted
   prerelease migration model); surface it in run summary/audit export.
5. **Slice 5 — cleanup.** Ratchet `test_composer_runtime_agreement.py` down to
   the seams that remain genuinely dual (YAML generation, error translation);
   revisit the engine facade exports (elspeth-896fb00e37).

Each slice is independently shippable; slices 2–3 are where regressions would
surface and are fully pinned by existing suites.

## 9. Open questions

- **Fingerprint input**: canonical resolved YAML string vs canonical
  `ElspethSettings` dump? (Leaning YAML-after-`resolve_runtime_yaml_paths`,
  since that is the last common textual form both paths share.)
- **Where does the L3 facade live**: `plugins/infrastructure/compile.py` or a
  top-level `elspeth/compile.py`? (Top-level reads better as public API and
  aligns with the facade-cleanup issue.)
- **Should `/validate` return the fingerprint** so the frontend can display
  "validated @ <hash>" and the execute request can assert it matches? (Cheap,
  high-value; recommended but needs a frontend touch.)
- **Resume path**: `orchestrator/resume.py` has its own validator call site —
  does resume re-fingerprint against the persisted run's fingerprint and fail
  on mismatch (strict) or warn (advisory)? Strict fits the audit doctrine.

## 10. Evidence index (for future sessions)

- `src/elspeth/engine/orchestrator/preflight.py:1-24` — shared-contract module
  docstring: layer note, mutation note, idempotency note.
- `src/elspeth/web/execution/validation.py:699-754` — `validate_pipeline`
  trust-boundary decorator, docstring listing the six dry-run steps and the
  caught error classes.
- `src/elspeth/web/execution/preflight.py:41-51` — `RuntimeGraphBundle`
  "must not cross request boundaries"; `:175` settings-hash precedent;
  `:189-219` thin wrappers + `build_validated_runtime_graph`.
- `src/elspeth/web/execution/service.py:659-677` — pre-`create_run` dry-run
  gate; `:1261-1273` — runtime recompile.
- `src/elspeth/engine/orchestrator/core.py:503-549` — `run()` signature;
  graph required; `:952-980` — run-init validators; `:1088` — runtime
  preflights.
- `src/elspeth/engine/orchestrator/runtime_preflight.py:1-18` — run-only
  external readiness; "intentionally do not share a module."
- `src/elspeth/plugins/infrastructure/runtime_factory.py:55-72` —
  `preflight_mode` context; observers in `json_sink.py`, `csv_sink.py`,
  `clients/retrieval/connection.py`.
- `src/elspeth/core/dag/builder.py:116` — `build_execution_graph`;
  `src/elspeth/core/dag/graph.py:143,282` — `ExecutionGraph`, `validate()`.
- `tests/integration/pipeline/test_composer_runtime_agreement.py` (3,969
  lines) — the parity-test tax; includes `TestOrchestratorAcceptsGraph`
  evidence that the executor consumes pre-built graphs.
