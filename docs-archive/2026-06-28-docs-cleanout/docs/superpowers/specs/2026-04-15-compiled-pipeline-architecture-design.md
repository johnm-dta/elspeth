# Compiled Pipeline Architecture — Composer, Compiler, Executor, Landscape

**Status:** Draft
**Date:** 2026-04-15
**Relates to:** `elspeth-e3207636d0`, `elspeth-f584eb820c`
**Context:** Future-state architecture design following the current web composer and execution work

---

## Summary

The original direction was correct: ELSPETH already has an implicit compiler,
but it is hidden inside runtime assembly and is not materialized as a stable
artifact.

This revised spec hardens that idea into an executable design. It makes several
explicit decisions that the earlier draft left underspecified:

1. **There are currently three distinct assembly paths, not one.**
   Web validation, web execution, and CLI execution share substantial logic,
   but they are not identical and do not currently prove preview-to-run parity.
2. **`ExecutionGraph` is already a partly compiled object.**
   It already contains deterministic node IDs, route maps, branch metadata,
   coalesce schema propagation, pipeline ordering, and node step numbering.
   `PipelineConfig` is not the artifact boundary; it is a runtime bundle of
   live plugin instances.
3. **The compiled artifact must be audit-safe and secret-safe.**
   It cannot contain resolved secret values or other environment material that
   belongs only at execution time.
4. **Web-only execution claims must survive compilation explicitly.**
   `blob_ref` is intentionally stripped from engine YAML today, so a future
   artifact-first executor needs a separate place to carry blob ownership and
   other authoring-surface claims.
5. **Compiled pipeline provenance must use dedicated metadata, not the existing
   sink artifact table.**
   The current Landscape `artifacts` table is for sink outputs linked to
   `node_states`, not for compiler outputs.
6. **Portable artifact data is not the same thing as runtime `ExecutionGraph`
   internals.**
   The artifact must store DTOs for schemas, routing, branch metadata, and
   authoritative traversal order; the loader must hydrate runtime-only classes
   and prove public-API parity, not just table equality.

The target architecture is therefore:

- **Composer** produces authoring state and advisory feedback
- **Compiler** turns authoring intent into a sealed, portable,
  secret-safe `CompiledPipeline`
- **Executor** combines that sealed artifact with local execution policy,
  secrets, and infrastructure, then runs the orchestrator
- **Landscape** records the authoritative provenance chain from source intent to
  sealed artifact to runtime verification to run result

---

## Current-State Alignment

### Current code paths

ELSPETH does not currently have one "compile path." It has three:

| Path | Current entry point | Effective flow | Important reality |
|------|---------------------|----------------|-------------------|
| Web validate | `web/execution/validation.validate_pipeline()` | `CompositionState -> YAML -> temp file -> load_settings() -> instantiate_plugins_from_config() -> ExecutionGraph.from_plugin_instances() -> graph.validate() -> graph.validate_edge_compatibility()` | Uses file-backed settings loading and advisory web checks |
| Web execute | `web/execution/service.ExecutionServiceImpl._run_pipeline()` | `CompositionState -> path/blob checks -> YAML -> resolve_secret_refs() -> load_settings_from_yaml_string() -> instantiate_plugins_from_config() -> ExecutionGraph.from_plugin_instances() -> graph.validate() -> graph.validate_edge_compatibility() -> PipelineConfig -> Orchestrator.run()` | Uses in-memory settings loading, runtime secret resolution, web-only ownership checks, and currently does not carry the CLI path's audit-safe `resolve_config()` payload |
| CLI run | `cli.py` + `cli_helpers.py` | `settings file -> _load_settings_with_secrets() -> instantiate_plugins_from_config() -> ExecutionGraph.from_plugin_instances() -> orchestrator context -> Orchestrator.run()` | Carries full runtime configs and an audit-safe `resolve_config()` payload |

### What is already "compiled" today

`ExecutionGraph.from_plugin_instances()` is more than a light graph constructor.
`core/dag/builder.py` already does the following:

- deterministic node ID assignment from canonical JSON
- connection producer/consumer resolution
- gate route resolution map construction
- sink map, transform map, aggregation map, coalesce map, and branch info
- coalesce schema propagation
- edge compatibility validation
- topological pipeline ordering
- node step map construction for Landscape audit numbering
- final config freezing on node metadata

In other words, **the builder is already a substantial compiler pass**.

It also means duplication is worse than it first appears: some paths explicitly
call `graph.validate_edge_compatibility()` even though the builder already
performed that validation during graph construction.

### What is not compiled today

`engine/orchestrator/types.PipelineConfig` is not a compiled artifact. It is a
runtime bundle that contains:

- live plugin instances
- runtime routing/error handling settings
- aggregation settings keyed by compiled node ID
- executor-owned config payloads

It is not transportable, not sealed, and not suitable as the compile/execution
boundary.

### Additional late validation still happens in the executor

The current orchestrator still performs configuration checks during the GRAPH
phase that are compile-time concerns:

- gate route destination validation
- transform `on_error` sink validation
- source quarantine sink validation
- sink failsink validation

Those checks currently live in `engine/orchestrator/validation.py` and run from
`Orchestrator._register_graph_nodes_and_edges()`. In the target architecture,
they move to compilation. The executor may keep them temporarily as migration
shadow assertions, but not as the primary truth.

### Web-specific authoring claims are currently outside engine config

The web YAML generator intentionally strips `blob_ref` and other web-only
metadata before producing engine YAML. That is correct today, but it means a
future artifact built from engine config alone cannot reproduce current blob
ownership checks. The compiled design must preserve those claims separately.

### Current parity is weaker than it appears

The three paths above are similar, but not identical:

- web validation uses `load_settings()` against a temporary file
- web execution uses `load_settings_from_yaml_string()` after secret resolution
- CLI run uses secret loading and a richer runtime bootstrap path

A successful validation therefore does **not** currently guarantee "this exact
sealed object will execute later."

---

## Problem Statement

The current architecture has six structural problems:

1. **Compilation is duplicated and late-bound.**
   Validation, web execution, and CLI execution each rebuild overlapping
   compile-time state.

2. **The compile/runtime boundary is blurred.**
   Topology, route resolution, and audit step numbering are built alongside
   executor-only concerns like secret resolution and infrastructure bootstrap.

3. **There is no stable execution artifact.**
   The system validates and executes by rebuilding from authoring input rather
   than consuming a sealed artifact.

4. **The audit-safe config and runtime config are not cleanly separated.**
   ELSPETH already has `resolve_config()` for safe audit storage, but runtime
   plugin config and audit node registration still operate on raw config dicts.
   The compiled boundary must make this separation explicit rather than relying
   on call-site discipline.

5. **Authoring provenance and execution claims are underspecified.**
   Web execution depends on state-level facts such as blob ownership that do not
   belong in engine YAML but do belong in the execution chain.

6. **Compiler provenance has no proper home.**
   The existing Landscape artifact model is for sink outputs, not compiler
   products, and the web session database currently stores YAML snapshots rather
   than compiled references.

---

## Goals

- Make compilation an explicit, first-class phase with a stable output artifact.
- Preserve current deterministic node identity, route resolution, step numbering,
  and DAG validation semantics.
- Keep the composer lightweight and advisory.
- Allow the executor to consume a verified artifact instead of rebuilding
  topology from YAML.
- Preserve trust boundaries:
  compilation is deterministic and secret-safe,
  secret resolution stays at execution,
  audit remains the permanent record.
- Support future remote or lower-trust executors without changing the pipeline
  model again.

---

## Non-Goals

- Replacing the orchestrator or row-processing engine.
- Eliminating runtime plugin instances in the first increment.
- Replacing `ExecutionGraph` with a new graph engine.
- Embedding resolved secret values in the compiled artifact.
- Turning the first increment into a distributed scheduler.
- Using pickled Python objects or serialized NetworkX internals as the artifact
  format.

---

## Core Design

### First-class objects

The architecture introduces four explicit objects:

1. **`CompilationRequest`**
   The input contract to the compiler. It is a pure-data, layer-safe request
   carrying adapted authoring payload plus non-engine claims needed later for
   execution policy checks.

2. **`CanonicalPipelineDefinition`**
   A normalized, secret-safe, engine-level representation of pipeline intent.
   This is the compiler's internal semantic form.

3. **`CompiledPipeline`**
   The sealed, canonical, portable artifact emitted by the compiler.

4. **`RuntimeAssembly`**
   The executor-owned in-memory bundle built from a verified
   `CompiledPipeline` plus local execution policy, secret materialization, and
   infrastructure. In the first increment it wraps the current
   `PipelineConfig`; it does not require a big-bang orchestrator rewrite.

### `CompilationRequest` contract

`CompilationRequest` is the cross-layer compiler contract. It must be a frozen,
serializable DTO made only of primitives, standard containers, and compiler
contract types from `contracts/compiler.py`.

Required fields:

- `request_id`: compiler correlation ID for preview/seal diagnostics and
  telemetry
- `source_kind`: where the request came from
  (`composition_state`, `yaml`, `settings`, `internal`)
- `source_id`: optional authoring-surface identifier such as composition-state
  ID
- `source_digest`: digest of the exact adapted authoring input before semantic
  normalization
- `adapter_version`: version of the adapter rules that produced this request
- `authoring_payload`: adapted, engine-shaped payload tree used as compiler
  input
- `origin_metadata`: audit-safe adapter metadata useful for diagnostics and
  provenance
- `runtime_claims`: request-side DTO bundle for `secret_bindings`,
  `path_claims`, and `blob_claims`

Hard constraints:

- no resolved secret values
- no resolver/service objects
- no live plugin instances
- no `CompositionState`, `ElspethSettings`, or other layer-local rich objects
- no raw `{"secret_ref": ...}` markers outside declared secret-bearing field
  paths

Adapters are responsible for turning higher-layer surfaces into this pure-data
contract before L1 compiler code sees them.

### `RuntimeAssembly` contract

`RuntimeAssembly` is the executor-only runtime bundle. In the first increment it
**wraps** the current `PipelineConfig` rather than replacing it outright.

Required members:

- `compiled_pipeline`: the verified sealed artifact being executed
- `verified_binding`: result of digest/signature/compatibility verification
- `graph`: hydrated `ExecutionGraph`
- `pipeline_config`: current `PipelineConfig` or direct successor used by the
  orchestrator
- `audit_safe_config`: executor-side audit-safe config payload bound to the run
- `secret_resolution_inputs`: run-scoped secret-fingerprint inputs for audit
- `resume_context`: optional checkpoint/resume state

This makes the relationship explicit: `CompiledPipeline` is the portable
artifact, `PipelineConfig` remains the orchestrator-facing runtime bundle for
now, and `RuntimeAssembly` is the executor structure that owns both.

### Compiler result and error model

Compiler APIs must return structured result objects, not a mix of ad-hoc
exceptions and partially populated payloads.

Required result surfaces:

- `CompilationPreviewResult`
  - `request_id`
  - `status`: `succeeded`, `rejected`, or `error`
  - `diagnostics`
  - `logical_digest_version`
  - optional candidate digests / normalized summaries on success only
- `CompilationSealResult`
  - `request_id`
  - `status`: `sealed`, `rejected`, or `error`
  - `diagnostics`
  - sealed artifact metadata on success only

Required diagnostic shape:

- `code`
- `phase`
- `severity`
- `message`
- optional `path`
- optional `component_id`
- optional `component_type`
- optional `suggested_fix`

Error categories must distinguish:

- input-contract errors
- secret-binding/undeclared-path errors
- normalization/digest-version errors
- compile-time validation errors
- compatibility/signature errors
- publication/storage errors
- internal invariant failures

Composer and CLI preview surfaces consume `CompilationPreviewResult`. Expected
compile rejections are returned as diagnostics; internal invariant failures are
treated as system errors and must not be downgraded into ordinary user
validation messages.

### Design rule: source adaptation must respect package layers

`CompilationRequest` is the cross-layer handoff into the compiler.
`core/compiler/service.py` may consume `CompilationRequest` directly, or a
`CompilationInputAdapter` protocol defined in `contracts/compiler.py`, but it
must not import `CompositionState` or any other `web/` type.

Therefore:

- YAML/settings/`ElspethSettings` adapters that depend only on L0/L1 types may
  live in `core/compiler/adapters/`
- the concrete `CompositionState` adapter lives in `web/` and converts
  `CompositionState` plus web-only claims into a `CompilationRequest`
- if multiple higher-layer surfaces later need the same normalized handoff, the
  shared transport type belongs in `contracts/`, not as an L1 import of L3
  models

### Design rule: compiled data and runtime policy are separate

The compiled artifact contains pipeline semantics and validation evidence.
It does **not** carry deployment-owned settings such as:

- Landscape database URL
- payload store backend/path
- checkpoint policy
- telemetry exporters
- concurrency policy
- rate limit policy
- post-run Landscape export wiring

Those remain in executor-owned runtime policy objects. This matches the current
CLI/orchestrator split, where Runtime*Config objects are derived from settings
outside `ExecutionGraph`.

### Design rule: authoring claims are explicit sidecars

The compiler must preserve execution-relevant authoring claims that are not part
of engine config, including:

- web blob references used for ownership enforcement
- declared file paths that require local policy checks
- source provenance identifying whether the artifact came from composition state,
  YAML, or another source adapter

These are part of the compiled artifact's provenance and runtime-claim sections,
not part of the normalized engine definition.

### Design rule: semantic equivalence is adapter-independent

`CompositionState`, YAML/settings files, and internal settings objects are
different source surfaces for the **same** pipeline semantics. The compiler must
therefore normalize them into one semantic form with explicit rules:

- absent vs explicit-default fields must collapse to one canonical encoding
- order-insensitive maps must be sorted canonically before hashing/serialization
- list order must be preserved only where order is semantically meaningful
- web-only metadata such as `blob_ref` must be excluded from semantic topology
  and carried only as runtime claims
- plugin options must preserve full nested-object shape where config-model
  selection or downstream hashing depends on that shape

The result is that equivalent CLI and web pipelines can share one
`definition.logical_digest` even when their authoring surfaces differ.

### Design rule: semantic normalization must be versioned

`definition.logical_digest` is only trustworthy if the normalization rules that
produced it are named and immutable.

Therefore:

- the artifact carries `definition.logical_digest_version`
- changing semantic normalization rules creates a **new** logical-digest
  version, not a silent reinterpretation of historical digests
- cache keys and reuse lookups that depend on logical semantics must include
  `logical_digest_version`
- historical digests remain valid only within their declared version

This is the recovery protocol for normalization fixes: preserve historical
records as-is, bump the version, and recompile under the new rules instead of
pretending old and new digests are interchangeable.

### Design rule: secret bindings are normalized claims, not serialized values

ELSPETH currently has **two** secret-binding forms:

- web composer markers such as `{"secret_ref": "OPENAI_KEY"}`
- CLI/file-driven `${ENV_VAR}` resolution via settings loading

The compiled design must normalize both into a shared secret-binding model:

- the logical definition records that a field is bound to a secret reference
- the runtime claims record the reference identity and binding kind
- the compiled artifact digest depends on the reference identity, **not** the
  resolved secret value
- actual secret-resolution fingerprints remain run-time audit data, not compile
  artifact data

This keeps preview/seal parity meaningful without leaking plaintext secrets or
pretending that compile-time and run-time secret material are the same thing.

### Design rule: secret normalization is a mechanical boundary, not a call-order convention

The guarantee "compiled artifacts never contain resolved secrets" must not rely
on remembering to run normalization before resolution.

Therefore:

- preview/seal compiler entry points accept unresolved authoring input and
  structured secret-binding claims only
- compiler APIs do **not** accept `ResolvedSecret`, `WebSecretResolver`,
  `SecretLoader`, or a post-resolution config dict/tree
- any current path that calls `resolve_secret_refs()` or `${VAR}` expansion
  before config loading remains executor-only migration behavior and must never
  become compiler input

To make secret-bearing pipelines compile without plaintext values, the design
adopts a **new secret normalization protocol**, not raw sentinel strings and not
`SecretRef | str` unions in runtime config models:

- source adaptation extracts each secret-bearing leaf into a
  `SecretBindingClaim` in `contracts/compiler.py`
- the normalized logical definition stores a secret-binding token/reference, not
  the original `{"secret_ref": ...}` marker and not a resolved value
- the compiler may materialize **ephemeral compile-local placeholders** only to
  satisfy existing Pydantic/config-model shape requirements during static plugin
  validation and instance construction
- those placeholders are derived from declared placeholder kinds
  (`opaque_string`, `url`, `connection_string`, etc.), exist only in compiler
  memory, and are excluded from `definition.normalized_pipeline`,
  `definition.audit_safe_config`, and the sealed artifact bytes
- any secret-bearing field without a declared compiler placeholder kind or
  equivalent compiler normalizer is a compile-time error

This turns the boundary into a type/protocol contract instead of an ordering
assumption in whichever caller wires the compiler.

### Design rule: secret references are allowed only on declared secret-bearing paths

The current `resolve_secret_refs()` tree walk is too permissive for the compiled
architecture. Secret-bearing references must be discovered from declared config
paths, not from arbitrary nested dict/list locations.

Therefore:

- adapters/compiler secret normalization may extract secret references only from
  declared secret-bearing fields
- any raw `{"secret_ref": ...}` marker found at an undeclared path is a
  compile-time error (`SECRET_REF_AT_UNDECLARED_PATH`)
- metadata, display names, audit annotations, and other non-secret-bearing
  fields must never be subject to secret resolution by recursive tree walk

This closes the "secret ref injection into arbitrary metadata" class of bugs by
turning it into a schema/contract violation instead of a caller-discipline
problem.

### Design rule: no mixed secret-resolution modes on the same request

During migration, the legacy recursive `resolve_secret_refs()` path and the new
declared-path compiler model may coexist in the codebase, but they must not both
touch the same authoring request/run.

Required rules:

- if a request enters compiler preview/seal, no caller may run
  `resolve_secret_refs()` over that authoring tree before or after compiler
  normalization
- legacy recursive secret resolution remains allowed only for explicitly legacy,
  non-artifact execution paths
- once a run is artifact-bound, any later execution or resume of that run must
  use compiled secret-binding claims rather than the legacy tree walk
- plugin config surfaces that expose open-ended `dict[str, Any]` or equivalent
  opaque mapping leaves are compiler-ineligible until their secret-bearing
  subpaths are explicitly declared or the field is classified as non-secret data
  and rejects raw `{"secret_ref": ...}` markers

This prevents undeclared nested secret markers from being silently resolved by a
legacy tree walk during Stages 1-6.

---

## Boundary Matrix

| Concern | Composer | Compiler | Executor | Landscape |
|---------|----------|----------|----------|-----------|
| `CompositionState` editing | Owns | Consumes `CompilationRequest` from an L3-owned adapter; must not import raw state | Must not depend on raw state | Optionally referenced by digest/ID |
| YAML generation for web compatibility | Owns today | Adapter only, not artifact format | Must not rebuild from YAML in target state | May store source digest only |
| Static plugin config validation | Advisory | Authoritative | Shadow assertions only during migration | Records compile result |
| Deterministic node IDs | Must not assign | Authoritative | Must reuse exactly | Records IDs during run |
| Route resolution and topology maps | Advisory previews only | Authoritative | Must not reinterpret | Records verification outcome and run binding |
| Node step numbering | Must not own | Authoritative | Must reuse exactly | Used for node/state audit numbering |
| Schema compatibility checks | Advisory previews only | Authoritative | Shadow assertions only during migration | Records compile result |
| Blob ownership claims | May originate | Preserved as claims | Enforces | Records compile origin / run binding |
| Path claims | May originate | Preserved as claims | Enforces against local policy | Records verification result |
| Secret bindings | May originate | Preserved as structured claims only; compiler never sees resolved values | Resolves after artifact verification | Records fingerprints only |
| Resolved secret values | Must never persist | Must never persist | Exists only in local runtime memory | Never stored |
| Runtime infra settings | Reads for UX only | Must not seal into artifact | Owns | Records run-time facts, not policy source files |

---

## Compiler

### Inputs

The compiler accepts `CompilationRequest` values produced by explicit source
adapters:

- `CompositionState` from the web composer, adapted in `web/`
- YAML/settings-file input from CLI or file-driven workflows, adapted in
  `core/compiler/adapters/`
- an already-normalized `ElspethSettings` object in internal/test flows,
  adapted in `core/compiler/adapters/`

All adapters must produce the same internal `CanonicalPipelineDefinition`, and
the L1 compiler must never import `web.composer.state.CompositionState`
directly. `CompilationRequest` also must never carry resolved secret values or
resolver services. Compiler entry points must validate the request contract
before any normalization or sealing work begins.

### Preview mode vs seal mode

The compiler supports two explicit modes:

1. **Preview compile**
   - Runs normalization and validation
   - Returns `CompilationPreviewResult` with diagnostics and a candidate digest
   - Does not persist a sealed artifact
   - Used by composer validation and repair loops

2. **Seal compile**
   - Runs the same passes
   - Returns `CompilationSealResult`
   - Emits canonical bytes and digest on success
   - Persists artifact metadata and storage reference
   - Produces the authoritative object that execution may consume later

This avoids flooding the audit trail with every advisory validation while still
making the authoritative compile phase auditable.

### Compiler passes

The compiler should be organized into explicit passes, even if the first
implementation delegates heavily to existing code:

1. **Source adaptation**
   - Convert authoring input into `CanonicalPipelineDefinition`
   - Preserve non-engine execution claims such as `blob_ref`
   - Record origin metadata and source digest

2. **Normalization**
   - Normalize config structure into one canonical logical form
   - Preserve full-object semantics for nested config blobs so digests are
     stable
   - Produce an audit-safe logical definition that never stores plaintext
     secrets
   - Normalize secret bindings from web `secret_ref` markers and CLI `${VAR}`
     expansion sources into one claim/reference model
   - Collapse authoring-surface differences that are not semantic
     (`None` vs omitted collections, injected defaults, key ordering)

3. **Compile-safe secret projection**
   - Materialize compiler-local placeholder values from normalized
     `SecretBindingClaim`s only where existing config models need string-shaped
     inputs
   - Use declared placeholder kinds so URL / connection-string validators still
     receive shape-correct values without seeing real secrets
   - Fail closed if any secret-bearing field lacks a declared placeholder kind
     or compiler normalizer

4. **Static plugin descriptor instantiation**
   - Instantiate plugins exactly as needed for schema extraction and validation
   - Compiler-time plugin construction must be side-effect-free
   - External calls in `__init__` are forbidden

5. **Topology build**
   - Reuse and gradually extract `core/dag/builder.py`
   - Assign deterministic node IDs
   - Resolve producers, consumers, route maps, sink maps, coalesce metadata,
     branch info, pipeline ordering, and node step map

6. **Structural validation**
   - graph acyclicity
   - source/sink presence
   - reachability
   - connection namespace rules
   - duplicate producer / duplicate consumer checks
   - route-resolution completeness
   - terminal routing validity

7. **Contract validation**
   - edge compatibility
   - sink required-field validation
   - any compile-time checks currently deferred into executor graph setup,
     including route/error/quarantine/failsink destination checks

8. **Artifact assembly**
   - build `CompiledPipeline`
   - include topology tables, validation evidence, compatibility fingerprint,
     and execution claims

9. **Sealing**
   - canonical JSON serialization
   - SHA-256 digest
   - optional signature envelope

10. **Publication**
   - insert/update authoritative Landscape metadata with publication
     status=`sealing`
   - publish artifact bytes through the artifact store using an atomic
     temp-file + fsync + `os.replace()` pattern at the final digest path
   - update Landscape metadata to status=`sealed` only after the artifact bytes
     are durably published
   - write web session/cache references only after Landscape says the artifact
     is `sealed`

### First increment implementation rule

The first increment does **not** need a new plugin metadata protocol.
It may continue to instantiate real plugin classes during compilation, because
current DAG validation relies on instance-owned metadata such as:

- `input_schema`
- `output_schema`
- `_output_schema_config`
- `declared_required_fields`
- `plugin_version`
- `determinism`

The architectural change is not "stop instantiating plugins immediately." It is
"make the result of that work explicit, sealed, reusable, and secret-safe."

### Compiler-time plugin construction contract

The first increment depends on real plugin instantiation, so the spec must make
the constructor contract explicit:

- plugin module import and `__init__` must not perform outbound network I/O,
  filesystem mutation, database writes, or ambient secret resolution
- constructor-time validation may parse config and build pure helper objects,
  but must not depend on live infrastructure
- any plugin that currently violates this becomes a **compiler-blocking**
  migration issue; the side effect must move to `on_start()`, `load()`,
  `process()`, or `write()`

This is a concrete stage-0 audit item, not an aspirational guideline.

---

## `CompiledPipeline` Artifact Contract

### Required properties

The artifact must be:

- **canonical**: the same logical pipeline produces byte-identical serialized
  bytes
- **portable**: loadable by another process or host without Python object
  pickling
- **immutable**: never mutated after sealing
- **secret-safe**: contains refs, fingerprints, or claims only, never resolved
  secret values
- **auditable**: traceable to origin digest, compiler version, and validation
  result
- **runtime-sufficient**: contains enough information for the executor to
  rebuild runtime graph/query objects without re-deriving topology from YAML

### Required sections

```json
{
  "artifact_version": 1,
  "compiler": {
    "engine_version": "rcX",
    "compiler_version": "rcX",
    "build_profile": "default",
    "canonical_version": "sha256-rfc8785-v1"
  },
  "origin": {
    "source_kind": "composition_state",
    "source_id": "state-uuid-or-null",
    "source_digest": "sha256:..."
  },
  "definition": {
    "logical_digest_version": "logical_digest_v1",
    "logical_digest": "sha256:...",
    "normalized_pipeline": {},
    "audit_safe_config": {}
  },
  "topology": {
    "nodes": [],
    "edges": [],
    "sink_id_map": {},
    "transform_id_map": {},
    "aggregation_id_map": {},
    "config_gate_id_map": {},
    "coalesce_id_map": {},
    "route_resolution_map": [],
    "route_label_map": [],
    "branch_info": [],
    "pipeline_nodes": [],
    "node_step_map": {}
  },
  "validation": {
    "checks": [],
    "warnings": [],
    "schema_contract_evidence": []
  },
  "compatibility": {
    "plugin_catalog_fingerprint": "sha256:...",
    "node_implementations": [
      {
        "node_id": "transform_llm_ab12cd34",
        "plugin_name": "llm",
        "plugin_version": "1.0.0",
        "code_hash": "sha256:..."
      }
    ],
    "requires_engine_version": "rcX"
  },
  "runtime_claims": {
    "claims_digest": "sha256:...",
    "secret_bindings": [],
    "path_claims": [],
    "blob_claims": []
  },
  "seal": {
    "canonical_sha256": "sha256:...",
    "signature": null
  }
}
```

### Artifact section semantics

#### `definition.normalized_pipeline`

This is the engine-level, secret-safe logical definition. It contains the
pipeline semantics needed to rebuild runtime config, but it does not contain
resolved secret values.

Secret-bound leaves are represented as normalized secret-binding tokens/claims,
not as authoring-surface markers and not as compiler placeholder strings.

#### `definition.audit_safe_config`

This is the audit-safe, fingerprinted config representation analogous to today's
`resolve_config(settings)`. It exists so:

- the compiler can record a stable provenance hash
- the executor can bind the run to an audit-safe config digest
- preview-to-run parity can be checked without carrying plaintext secrets

This section is **not** the runtime config object passed into plugins.

For web secret-ref pipelines this may be derived from normalized claim metadata
rather than from a fully resolved `ElspethSettings` object. The contract is
"audit-safe and preview/run parity-stable," not "byte-identical to today's CLI
`resolve_config(settings)` output in every input mode."

Construction rule:

- `definition.audit_safe_config` must be built by a placeholder-free projection
  from `definition.normalized_pipeline` plus normalized claim metadata
- compiler-local placeholder-bearing settings objects, plugin instances, or
  transient runtime config objects are forbidden inputs to this projection
- if the implementation cannot construct `audit_safe_config` without consulting
  placeholder values, compilation fails closed rather than emitting placeholder
  strings into audit provenance

#### `definition.logical_digest_version`

This names the semantic-normalization algorithm used to compute
`definition.logical_digest`.

Rules:

- it changes only when semantic normalization rules change
- it does **not** change for ordinary compiler bug fixes that leave logical
  semantics unchanged
- caches and reuse lookups that depend on logical equivalence must include it
- historical compiled artifacts keep their original version forever

#### `topology`

The artifact stores portable tables and maps, not a pickled `ExecutionGraph`.
The loader rebuilds runtime graph/query objects from these tables.

Portable topology data and runtime graph objects are **not identical**:

- artifact data must not serialize `type[PluginSchema]` objects from
  `NodeInfo.input_schema` / `output_schema`
- artifact data must not serialize `BranchInfo`, `NodeInfo`, `RouteDestination`,
  or NetworkX internals directly
- artifact data must store portable DTOs that the loader converts back into
  runtime contracts and graph state

Runtime-only objects are hydrated on load from portable DTOs:

- `SchemaConfig` travels as `SchemaConfig.to_dict()` / `SchemaConfig.from_dict()`
- `RouteDestination` travels as an explicit tagged destination DTO
- coalesce union `PluginSchema` classes are regenerated from serialized schema
  config during load using the fixed runtime class name
  `_CoalesceSchema_{node_id}`; class identity is runtime scaffolding only, but
  the naming rule itself is part of the loader/builder parity contract

This means Stage 3 parity must prove public graph behavior, not just static map
equality.

The topology section must include, at minimum:

- node table with node ID, node type, plugin name, plugin version binding, and
  per-node config digest, with plugin-backed nodes also bound to audited
  implementation identity (`plugin_version` + `code_hash`)
- per-node schema-config DTOs (`input_schema_config`, `output_schema_config`)
  and `declared_required_fields`, not runtime Pydantic model classes
- edge table with labels and routing mode
- all currently-derived ID maps
- route resolution map encoded as portable destination DTOs
- route label map
- branch info for coalesce/fork tracking, including portable branch schema DTOs
- authoritative topological order
- pipeline node order
- branch-first-node map
- branch-to-sink map
- terminal-sink map
- node step map

Storing the step map explicitly is required so the executor does not silently
diverge from current Landscape numbering. Storing authoritative traversal order
is also required because the orchestrator currently consumes
`graph.topological_order()`, `graph.get_pipeline_node_sequence()`, and related
helpers whose behavior must not depend on NetworkX insertion order after load.

#### `compatibility.plugin_catalog_fingerprint`

This replaces the earlier draft's vague "plugin manifest digest." The current
system does not have a formal manifest file.

This field is only meaningful once the plugin identity discipline from
`docs/superpowers/specs/2026-04-15-plugin-version-audit-design.md` is in
place. The compiler must **not** hash today's raw `plugin_version` declarations
and pretend that is an implementation-compatibility signal.

The compatibility model is:

- `node_implementations`: authoritative per-node implementation bindings for
  every plugin-backed node, including at minimum `plugin_name`,
  `plugin_version`, and `code_hash`
- `plugin_catalog_fingerprint`: stable digest of the discovered executable
  plugin catalog plus structural-node implementation versions, where each plugin
  contributes audited identity fields such as `(plugin_name, plugin_version,
  code_hash)`

If audited plugin identity is missing, defaulted (`plugin_version="0.0.0"`), or
unenforced, sealing must fail closed or Stage 1 remains blocked.

CI does not need to prove that every behavioral change used the "correct"
semver bump. The mechanical guarantee is enforced `code_hash` freshness;
`plugin_version` remains human/auditor-facing context.

#### `runtime_claims`

This section exists so the artifact can stay engine-portable while still
supporting current runtime checks:

- `secret_bindings`: structured claims that identify which fields must be
  resolved at execution time and by what binding kind
- `path_claims`: declared source/sink paths that require executor policy checks
- `blob_claims`: web blob IDs or equivalent opaque claims required for ownership
  verification
- `claims_digest`: stable digest of the runtime-claim section alone so the
  system can distinguish semantic-topology equivalence from claim equivalence

Compiler-local placeholder values are **not** part of `runtime_claims`; they
are transient compilation scaffolding only.

### Digest semantics

The design uses **three different digests** plus two version stamps:

- `origin.source_digest`: digest of the exact authoring input after source-side
  serialization rules; useful for provenance and cache lookup
- `definition.logical_digest_version`: names the semantic normalization rules
  used for the logical digest
- `definition.logical_digest`: digest of the normalized pipeline semantics only
  under that named logical-digest version; use this for CLI/web parity
  assertions
- `compiler.canonical_version`: names the byte-level canonicalization algorithm
  used for seal-time serialization
- `seal.canonical_sha256`: digest of the full sealed artifact bytes, including
  topology, validation evidence, compatibility metadata, and runtime claims;
  use this for execution identity and artifact storage

`runtime_claims.claims_digest` sits between the last two and makes claim drift
explicit. Two artifacts may share a logical digest while differing in claims
(`blob_ref`, path binding, secret-ref identities), and therefore must not be
treated as the same executable artifact.

Normalization/canonicalization recovery protocol:

- semantic-normalization changes bump `definition.logical_digest_version`
- byte-canonicalization changes bump `compiler.canonical_version`
- no historical digest is rewritten in place
- cache entries keyed by an older version become cold and must be recomputed
  under the new version
- audit and provenance queries must display both the digest and its version

### Canonicalization requirements for sealing

Before seal-time hashing becomes authoritative, the canonicalization contract
must explicitly handle the value classes that would otherwise create latent
digest instability:

- `Enum` members must canonicalize as the recursively normalized form of
  `obj.value` only; the Enum class name and member name are never part of the
  canonical JSON surface
- finite `Decimal` values must canonicalize with
  `format(obj.normalize(), "f")` rather than `str(obj)`, so equivalent values
  such as `Decimal("1.0")` and `Decimal("1.00")` hash the same
- naive `time` objects must be rejected at compile/seal time unless and until
  the system defines an explicit wall-clock semantic contract for them

Stage 0 must land these exact `Enum` / `Decimal` rules in both
`src/elspeth/core/canonical.py` and `src/elspeth/contracts/hashing.py` before
any seal digests, cache keys, or audit bindings are treated as authoritative.

These rules are part of compiler correctness, not optional polish. If they
change later, the affected digest/canonical version must change with them.

### Signature model

Two modes are supported:

1. **Digest-only mode**
   - permitted only for preview flows, local development, and migration shadow
     execution before artifact-first execution becomes authoritative
   - not sufficient for long-lived authoritative execution identity once the web
     path starts consuming persisted artifacts

2. **Signed mode**
   - required no later than Stage 4, when web execution becomes
     artifact-first/authoritative
   - required for detached or lower-trust executors

The signature field belongs in the artifact contract from day one. It may remain
`null` only in pre-Stage-4 non-authoritative flows.

### Closed decision: move signature enforcement to Stage 4

Digest-only verification is acceptable only while compilation artifacts are
still migration/shadow outputs.

Once the web path switches to authoritative artifact-first execution, the
artifact must carry a verifiable signature envelope and the executor must verify
it before use. Stage 7 still extends this trust model to detached executors, but
signature enforcement itself no longer waits that long.

### Closed decision: store portable topology, not serialized `ExecutionGraph`

The artifact stores portable tables and maps, then the loader reconstructs
runtime graph/query objects. It does **not** store pickled Python objects,
NetworkX internals, or a direct serialized `ExecutionGraph`.

### Closed decision: runtime graph classes are hydrated from DTOs, not serialized directly

The initial compiled-pipeline implementation treats these as **runtime-only**
objects:

- `NodeInfo.input_schema` / `output_schema` Pydantic model classes
- coalesce-union schema classes generated from `SchemaConfig`
- `BranchInfo` and its embedded `SchemaConfig`
- `RouteDestination`
- NetworkX node/edge insertion state

The artifact stores DTOs for their portable content instead. The loader then
hydrates runtime graph state from those DTOs.

Implications:

- route destinations need an explicit artifact DTO and round-trip tests
- branch schemas travel as `SchemaConfig` dicts, not as raw dataclass objects
- `_coalesce_schema_counter`-style process-global class naming is not an
  acceptable source of parity; loader/builder hydration must use the fixed name
  `_CoalesceSchema_{node_id}` derived from the deterministic compiled node ID
- behavior-sensitive order must be stored explicitly, not inferred from how
  NetworkX happens to iterate after reconstruction

### Closed decision: use a secret normalization protocol, not raw sentinels or union-typed runtime fields

The initial compiled-pipeline implementation uses a **secret normalization
protocol** built around `SecretBindingClaim` plus compiler-local placeholder
materialization for declared secret-bearing fields.

Rejected approaches:

- **Raw sentinel strings everywhere**
  - rejected because one global placeholder shape will fail real validators
    (`str` vs URL vs connection string) and invites accidental persistence into
    artifacts or audit payloads
- **`SecretRef | str` in runtime config models**
  - rejected because it lets unresolved secret markers leak across the
    compile/runtime boundary and forces executor-only concerns into general
    plugin config types

The protocol approach keeps unresolved secret identity explicit in the artifact,
keeps plaintext values executor-only, and still lets the first increment reuse
existing Pydantic/plugin construction paths.

---

## Executor

### Input model

The executor consumes:

- a sealed `CompiledPipeline`
- local `ExecutionPolicy`
- local secret/materialization services
- local infrastructure objects such as payload store and Landscape DB
- optional cache/session hints that may help locate an artifact but are never
  authoritative

### `ExecutionPolicy`

`ExecutionPolicy` is the executor-owned runtime policy object. It contains
deployment and infrastructure configuration that the compiled artifact must not
own, including:

- Landscape DB location and encryption settings
- payload store configuration
- checkpoint policy
- concurrency policy
- rate limit policy
- telemetry/export configuration
- auth and ownership verification hooks

This mirrors the current separation between pipeline semantics and runtime
configs in the CLI path.

### Executor responsibilities

- verify digest and signature (signature optional only before Stage 4)
- verify engine and audited plugin implementation compatibility
- resolve the authoritative compiled-digest binding from Landscape before
  trusting any session/cache hint
- verify runtime claims:
  - blob ownership
  - path allowlist/policy
  - secret reference availability
- resolve secret bindings only after artifact verification
- materialize runtime plugin configs from the normalized logical definition plus
  local secret materialization
- rebuild `ExecutionGraph` and related query objects from compiled topology
  without re-deriving topology from YAML
- build runtime plugin instances and `PipelineConfig`/successor structures
- build `RuntimeAssembly`
- run the orchestrator
- record verification outcome and run binding in Landscape before processing

### Executor must not do

- regenerate topology from YAML
- assign fresh node IDs
- rebuild route resolution independently
- recompute node step numbering independently
- persist resolved secret values into artifact or audit metadata
- invoke preview/seal compilation on a post-resolution config tree
- silently "fix up" an incompatible artifact
- fall back to session/cache digest hints when Landscape authoritative lookup is
  unavailable

### Runtime-only checks that remain valid long-term

These remain executor-owned even after the compiler exists:

- secret resolution for the current environment and user scope
- blob ownership verification
- path allowlist enforcement against local policy
- local audited plugin catalog compatibility
- infrastructure availability and policy validation

### Authoritative digest source

The authoritative source of "which artifact am I executing?" is Landscape, not
the web session database.

Rules:

- Landscape `runs.compiled_digest` (or an equivalent authoritative binding
  table) is the only trusted run-to-artifact binding
- Landscape `compiled_pipelines.status='sealed'` is the only trusted indication
  that a digest is executable
- any digest value coming from the web session DB, request payload, or cache is
  only a lookup hint
- if a session/cache hint disagrees with Landscape, execution fails closed and
  records an integrity/security failure
- if Landscape is unreachable for authoritative digest lookup or seal-status
  verification, execution/resume fails closed
- retry-then-trust-hint behavior is forbidden; session/cache hints, request
  payload digests, and artifact-path naming are never fallback trust anchors

This removes dual-database ambiguity and prevents weaker-access-control stores
from becoming de facto trust anchors.

### Compile-time checks that move out of the executor

These are compile-time concerns and must migrate to the compiler:

- gate route destination validation
- transform `on_error` sink validation
- source quarantine destination validation
- sink failsink validation
- route-resolution completeness
- structural topology validation
- schema compatibility validation

The executor may keep them temporarily as shadow assertions during migration,
but compile-time output is the authoritative result.

---

## Runtime Materialization Contract

### Secret safety

The compiled artifact never contains resolved secret values.

The executor materializes runtime plugin input using:

- normalized logical definition from the artifact
- local secret resolution services
- local environment policy

Resolved values exist only in local memory during runtime assembly.

Any compiler-local placeholder strings used to satisfy compile-time config
validation are discarded before sealing and are never accepted as executor
runtime input.

### Audit safety

The executor must maintain a clean split between:

- **runtime plugin config**: may contain resolved secrets in memory
- **audit-safe config**: fingerprinted, serialized, and persisted

The compiler artifact therefore carries an audit-safe config section explicitly,
instead of relying on call sites to remember when to use `resolve_config()`.

### Loader contract

`core/compiler/loader.py` is not allowed to become a second hidden compiler. It
must:

- hydrate `ExecutionGraph` and related runtime tables from stored topology DTOs
- reconstruct runtime-only schema classes from serialized `SchemaConfig`
  definitions; never deserialize raw class objects from the artifact
- hydrate coalesce runtime schema classes with the fixed name
  `_CoalesceSchema_{node_id}` so process-local counters and call order never
  affect loaded behavior
- round-trip route destinations through an explicit DTO codec before injecting
  them into runtime structures
- set route maps, branch metadata, topological order, pipeline-node order,
  branch-first-node map, terminal-sink map, and node-step map from the artifact,
  not by re-deriving them from NetworkX traversal
- deep-freeze reconstructed config payloads before handing them to runtime code,
  matching current builder behavior
- populate the runtime assembly with an audit-safe config payload on **both**
  web and CLI paths so the current web/CLI divergence disappears

The loader may run invariant checks (`graph.validate()`, checksum comparisons)
as corruption guards, but those checks are not permitted to redefine topology.

The loaded graph must preserve the behavior of the builder graph for all public
runtime APIs the orchestrator depends on, including:

- `topological_order()`
- `get_pipeline_node_sequence()`
- `build_step_map()`
- `get_branch_first_nodes()`
- `get_branch_to_sink_map()`
- `get_terminal_sink_map()`
- `get_coalesce_branch_schemas()`
- `get_route_resolution_map()`

Shadow-mode parity must exercise those APIs directly and must not stop at raw
table equality.

---

## Resume, Retention, and Checkpoints

### Resume binding

Resume must bind to the **original sealed compiled artifact** used by the run.
It must not rebuild from current settings/YAML and claim that is the same run.

Required rules:

- each resumable run is bound in Landscape to exactly one `compiled_digest`
- checkpoint compatibility and payload recovery validate against the graph
  hydrated from that bound artifact
- artifact lookup for resume resolves through Landscape, not through the session
  DB or the current settings file
- during the Stage 4 -> Stage 5 split, CLI may use read-only Landscape artifact
  lookup only to detect that a run is artifact-bound and reject legacy resume
  with a specific transition error; it must not rebuild that run from current
  settings/YAML
- current settings/YAML may still be used temporarily for migration tooling or
  legacy-precompiled runs, but they are not the long-term authority once a run
  carries an authoritative compiled binding

### Artifact retention and purge policy

Compiled artifacts are not free to purge arbitrarily.

Retention must pin or otherwise preserve any artifact that is referenced by:

- a run in a resumable state
- a run with retained checkpoints
- an audit retention window that still promises reproducible execution identity

If an artifact has been moved to colder storage, resume may restore it by digest
first and then continue. If the artifact cannot be restored, resume fails closed
with a specific artifact-unavailable error. It must not silently recompile from
current inputs to continue the historical run.

### Legacy and post-cutover run bindings

`compiled_digest` nullability needs an explicit mechanical meaning:

- pre-Stage-2 historical runs may remain in `legacy_uncompiled` mode
- post-Stage-2 runs must be in `compiled` mode with a non-null
  authoritative `compiled_digest`

The schema should therefore carry an explicit binding mode (or equivalent
versioned provenance marker), not rely on `NULL` alone to distinguish legacy
history from a post-cutover integrity failure.

### Structural-node versioning

Config gates and coalesce nodes have no plugin class, but they still need
version binding. Their implementation version is tied to engine/compiler
versioning and must be recorded in the artifact and reused at runtime.

---

## Landscape and Storage Model

### Closed decision: do not reuse `artifacts`

The current Landscape `artifacts` table is the wrong model for compiler output.
It is defined for sink outputs and requires `produced_by_state_id`, which a
compiled pipeline does not have.

Compiled pipeline provenance therefore requires dedicated metadata.

### Recommended persistence split

#### Landscape audit database

Add dedicated compiler provenance metadata:

- **`compiled_pipelines`**
  - `compiled_digest` (PK, unique digest claim key)
  - `status` (`sealing`, `sealed`, `failed`)
  - `artifact_version`
  - `compiler_version`
  - `engine_version`
  - `canonical_version`
  - `source_kind`
  - `source_digest`
  - `logical_digest_version`
  - `logical_digest`
  - `plugin_catalog_fingerprint`
  - `node_implementations_json`
  - `artifact_uri`
  - `artifact_size_bytes`
  - `validation_summary_json`
  - `runtime_claims_json`
  - `signature_json`
  - `failure_reason`
  - `seal_started_at`
  - `sealed_at`
  - `created_at`

- **Run binding**
  - add `compiled_digest` to Landscape `runs`, or add a separate
    `run_compiled_bindings` table if the project prefers not to widen `runs`
  - add `compiled_binding_mode` (or equivalent explicit provenance marker) so
    legacy pre-compiled runs are distinguishable from post-cutover integrity
    failures
  - store artifact verification result and compatibility outcome with the run

This satisfies audit primacy: the authoritative compile and the authoritative
run binding are probative and belong in Landscape.

#### Web session database

Add cache/reference metadata needed by the web app:

- `compiled_digest_hint` on web `runs`
- optional later `compiled_pipelines` cache table keyed by
  `(composition_state_id, logical_digest_version, compiler_version, plugin_catalog_fingerprint)`

This lets the web app reuse sealed artifacts without making the session DB the
source of truth for audit provenance.

Session DB references are **non-authoritative cache pointers**:

- they are written only after Landscape `compiled_pipelines.status='sealed'`
- any read path that starts from a session-side `compiled_digest_hint` must resolve
  it through Landscape and require `status='sealed'`
- a missing or non-`sealed` Landscape row is an integrity error, not a signal to
  trust the session DB alone
- a mismatch between session-side hint and Landscape authoritative binding is a
  security/integrity failure, not a cache miss

### Closed decision: `compiled_digest` is the unique publication key

`compiled_pipelines.compiled_digest` is not just descriptive metadata; it is
the unique row identity for publication ownership.

Implications:

- duplicate `compiled_pipelines` rows for the same digest are forbidden by
  schema
- the first successful `status='sealing'` insert claims publication ownership
  for that digest
- an insert for a digest that already exists with `status='sealed'` is a
  cache-hit/no-op path, not a second publish
- a live conflict with an existing `status='sealing'` row fails closed rather
  than allowing concurrent publication against the same digest
- failed rows are not silently overwritten by a generic upsert path; retry or
  recovery must be explicit and auditable

### Closed decision: require authoritative Landscape `compiled_digest`, session-side hint only

The first implementation must add authoritative `compiled_digest` binding in
Landscape and only a non-authoritative `compiled_digest_hint` in the web
session DB.

That split is required because the web app needs a fast lookup hint, but the
security boundary requires one authoritative digest source.

A dedicated session-side `compiled_pipelines` cache table is deferred until the
web app actually needs cross-run or cross-validation artifact reuse. The schema
for that cache is therefore intentionally non-blocking for the first increment.

### Artifact bytes

Artifact bytes themselves may live in a dedicated artifact store abstraction:

- filesystem-backed in the first increment
- blob/object store later

Landscape stores metadata and references, not the canonical bytes themselves.

Artifact publication must use the repo's existing atomic-write pattern:

- write bytes to a temp file in the destination directory
- `fsync()` the temp file
- publish with same-filesystem `os.replace()` to the final digest path
- `fsync()` the parent directory after rename

Temp files are staging artifacts only. They are not published compiled
artifacts and may be cleaned up independently after crashes.

### Publication protocol

There is no distributed transaction across artifact store, Landscape, and the
web session DB. The first increment therefore uses a **Landscape-first,
two-phase publication state machine**:

1. Build canonical bytes and compute `compiled_digest` in memory.
2. In Landscape, attempt a fresh insert into `compiled_pipelines` with
   `status='sealing'`, the final digest, and the final `artifact_uri`. This
   row claims publication ownership for that digest.
3. If step 2 conflicts on `compiled_digest`:
   - existing row with `status='sealed'`: treat as a cache hit and return the
     existing sealed artifact metadata without rewriting artifact bytes
   - existing row with `status='sealing'`: fail closed with an explicit
     concurrent-seal conflict; do not publish bytes or session DB references
   - existing row with `status='failed'`: fail closed unless an explicit,
     auditable recovery path has first re-claimed the row for a new sealing
     attempt
4. Publish artifact bytes to the artifact store at that final digest path using
   atomic rename semantics.
5. If publication succeeds, update Landscape to `status='sealed'`.
6. Only after step 5 succeeds may the web session DB write
   `compiled_digest_hint` references or cache rows.

Best-effort failure handling:

- if publication fails while the process is still alive, update Landscape to
  `status='failed'` with a failure reason
- if the process crashes mid-publication, recovery reconciles stale
  `status='sealing'` rows by re-hashing the final artifact bytes, not by path
  existence alone

Required reconciliation behavior for stale `sealing` rows:

- final artifact exists and SHA-256 of its bytes equals the recorded
  `compiled_digest`:
  promote the Landscape row to `sealed`
- final artifact does not exist:
  mark the Landscape row `failed`
- final artifact exists but byte re-hash does not match the recorded digest:
  mark the Landscape row `failed`, emit an integrity/security event, and do not
  treat path naming or file presence as evidence of authenticity
- duplicate rows are impossible by schema; the reconciler operates on the
  single authoritative row keyed by `compiled_digest`
- session DB is missing a reference for an already `sealed` artifact:
  backfill or lazily repair the cache/reference row

This is the mechanism that closes the provenance gap without pretending the
three stores can commit atomically.

### Provenance chain

The required authoritative chain is:

```text
authoring input identified
  -> preview compile(s) as needed
  -> canonical bytes + digest computed in memory
  -> Landscape row created with status=sealing
  -> artifact bytes atomically published at digest path
  -> Landscape row updated to status=sealed
  -> session DB reference/cache updated
  -> executor verifies digest and compatibility
  -> run created and bound to compiled digest
  -> run completes / fails / cancels
```

### Audit primacy

The following are probative and must be audited first, telemetered second:

- compile publication started (`status='sealing'`)
- sealed compile creation (`status='sealed'`)
- compile publication failure / reconciliation outcome
- compile rejection at seal time
- compiled artifact verification result
- run binding to compiled digest
- run completion status

Preview compile attempts may remain non-authoritative unless and until the team
decides they are worth persistent audit capture.

### Operational telemetry, dashboards, and alerts

Compilation and artifact execution need explicit observability, but it must
follow audit primacy: probative events audit first, telemetry second, and no
row-level decision logging.

Required telemetry/health surface:

- compile counts by mode and outcome (`preview`, `seal`, `resume-verify`)
- compile latency by phase
- compile rejection counts by diagnostic code
- artifact publication/reconciliation counts and durations
- signature verification failures
- authoritative-binding mismatches between session hints and Landscape
- shadow mismatch counts by mismatch class
- artifact store publish/retrieve failures
- resume failures caused by missing artifact bytes or expired retention
- post-cutover runs missing authoritative compiled bindings

Required dashboards:

- compiler health and rejection trends
- artifact publication/sealing/reconciliation health
- shadow parity health and mismatch age
- artifact verification/signature health
- resume availability and artifact-retention health

Required alerts:

- any shadow mismatch in CI/staging
- repeated seal/publication failures
- stale `sealing` rows above threshold
- any post-cutover run with missing authoritative `compiled_digest`
- any signature verification failure in authoritative execution
- any resumable run blocked because artifact bytes were purged or unavailable

Preview compiles may be telemetry-only. Seal/publication/verification events are
probative and must be audited first.

---

## Detailed Decisions

### Decision 1: compiler facade first, not a big-bang rewrite

`core/dag/builder.py` remains the implementation center in the first increment.
The compiler facade wraps it, names its outputs, and seals them.

### Decision 2: artifact-first execution, but shadow mode during migration

During migration, the executor may:

- load compiled topology
- also rebuild current topology via existing code
- compare digests/maps
- compare public `ExecutionGraph` API outputs and graph-phase orchestrator
  artifacts
- fail closed on mismatch in test/staging

This is a migration technique, not the target architecture.

To prevent permanent shadow-mode drift, every shadow-mode rollout must carry a
mechanical sunset:

- an explicit expiry constant or release marker in code
- a CI check that fails once the expiry passes unless the shadow path is
  removed or explicitly renewed by design review
- a feature flag so production rollback does not depend on keeping shadow mode
  forever

### Decision 3: web authoring claims are first-class runtime claims

`blob_ref` and equivalent web-only claims are not leaked into engine config,
but they are preserved in the compiled artifact's claim section so runtime
ownership checks remain implementable.

### Decision 4: path allowlist is executor-authoritative

Path-bearing config may be checked early by composer and compiler for UX, but
the executor is authoritative because allowlist policy is deployment-local.

### Decision 5: compatibility is fingerprint-based, not hand-wavy patch logic

Compatibility is determined by:

- artifact schema version
- engine/compiler version policy
- audited plugin implementation identity (`plugin_version` + `code_hash`)
- plugin catalog fingerprint derived from those audited identities

Not by vague "same patch series should probably be okay" heuristics.

Stage 1 is blocked on the prerequisite work in
`docs/superpowers/specs/2026-04-15-plugin-version-audit-design.md`. Until that
plan lands, current plugin-version strings are descriptive at best and are not
an authoritative compatibility signal.

### Decision 6: publication is Landscape-first and crash-recoverable

The initial implementation does **not** pretend there is a transaction spanning
Landscape, the artifact store, and the session DB.

Instead it uses:

- Landscape-first `status='sealing'` metadata
- unique `compiled_digest` claim rows in `compiled_pipelines`
- atomic artifact publication at the final digest path
- a final Landscape transition to `status='sealed'`
- session DB writes only after authoritative sealing
- reconciliation for stale `sealing` rows and cache lag

This keeps audit primacy intact and ensures no published artifact bytes exist
without at least a corresponding Landscape sealing record.

---

## Migration Plan

### Stage 0: Lock in baseline parity tests

Before changing architecture:

- capture current graph/step-map parity tests around representative pipelines
- capture web validate vs web execute vs CLI parity expectations where they are
  supposed to agree
- add artifact safety tests for "no plaintext secrets in serialized artifact"
- harden seal-time canonicalization in both `src/elspeth/core/canonical.py` and
  `src/elspeth/contracts/hashing.py` so `Enum` values canonicalize through
  normalized `.value`, finite `Decimal` values use
  `format(obj.normalize(), "f")`, and naive `time` values are rejected before
  sealing becomes authoritative
- define and freeze `logical_digest_version` and `canonical_version`
  semantics before cache keys or audit records depend on them
- land the audited plugin identity prerequisite from
  `docs/superpowers/specs/2026-04-15-plugin-version-audit-design.md` or wire
  the compiler compatibility section directly to its outputs before compatibility
  metadata becomes authoritative
- inventory secret-bearing config fields and assign compiler placeholder kinds or
  compiler normalizers before secret-bearing pipelines are allowed through the
  compiler path
- inventory `ExecutionGraph` fields into portable DTOs vs runtime-only hydrated
  objects before designing the loader boundary
- identify every orchestrator-visible graph API that must match between
  builder-loaded and artifact-loaded graphs
- define rollout feature flags, rollback switches, and shadow-mode expiry checks
  before any authoritative traffic depends on compiled artifacts

This stage is mandatory. It prevents refactoring the compile boundary without a
behavioral oracle.

### Stage 1: Introduce compiler contracts and preview/seal API

- create `CompilationRequest`, `CanonicalPipelineDefinition`, and
  `CompiledPipeline`
- create `CompilationPreviewResult`, `CompilationSealResult`,
  `CompilationDiagnostic`, and explicit compiler error categories
- create `SecretBindingClaim` and the compiler-side secret normalization /
  placeholder projection protocol
- consume audited plugin identity data (`plugin_version`, `code_hash`) from the
  plugin-version audit plan when assembling artifact compatibility metadata
- define portable topology DTOs for schema configs, route destinations, branch
  metadata, and authoritative traversal order
- extract or define shared schema-hydration rules for builder/load parity
- create compiler facade that wraps current builder logic
- add preview mode and seal mode
- make compiler preview/seal accept unresolved inputs only; resolver services and
  post-resolution config dicts are executor-only
- make undeclared secret-ref paths fail compile explicitly
- do not change execution yet

Outcome:
the project has a first-class compiler surface without changing runtime
behavior.

### Stage 2: Persist sealed compile metadata

- add dedicated compiled-pipeline metadata in Landscape
- add web session cache/reference hint fields
- persist `compiled_digest`, validation summary, and storage URI
- persist `logical_digest_version`, `canonical_version`, and authoritative run
  binding mode
- enforce unique `compiled_digest` claim semantics in `compiled_pipelines`
- make already-`sealed` digest inserts return cache hits and live `sealing`
  conflicts fail closed
- implement `sealing` -> `sealed` / `failed` publication states and stale-row
  reconciliation
- ensure session DB references are written only after Landscape sealing
  completes
- emit audit/telemetry for compile publication, verification, and reconciliation
- keep existing `pipeline_yaml` persistence during migration

Outcome:
the system can point to an authoritative sealed compile record.

### Stage 3: Add artifact loader and shadow execution

- add loader that rebuilds runtime graph/query objects from compiled topology
- execute with artifact-loaded topology
- also rebuild with current runtime path in shadow mode
- add CI-enforced shadow expiry / renewal check
- compare node IDs, route maps, sink maps, topological order, pipeline order,
  branch-first-node maps, terminal-sink maps, step maps, and canonicalized
  coalesce branch schemas
- compare orchestrator GRAPH-phase artifacts derived from each graph, not just
  raw graph tables

Outcome:
artifact loading is proven equivalent at the public-API and graph-phase
behavior level before it becomes authoritative.

### Stage 4: Switch web execute to signed artifact-first execution

- web validation uses compiler preview
- web execute seals if necessary, signs/verifies the artifact, then runs from
  compiled topology
- retain web blob/path/secret checks through runtime claims and execution policy
- treat session-side `compiled_digest_hint` as a hint only; Landscape remains
  authoritative
- before Stage 5 lands, CLI resume gains a read-only authoritative binding
  lookup that rejects artifact-bound runs with a specific transition error
  instead of rebuilding them from `settings.yaml`

Outcome:
the web path no longer recompiles from YAML as its primary runtime input.

### Stage 5: Switch CLI run and resume to artifact-first

- CLI compile/execute shares the same compiler and artifact loader
- CLI runtime policy continues to own runtime configs such as checkpointing,
  telemetry, concurrency, and export
- CLI resume resolves the original compiled artifact by authoritative digest
  binding instead of rebuilding from current settings as primary truth
- remove the transitional Stage-4 fail-closed rejection path for artifact-bound
  CLI resume once authoritative artifact lookup+load is available
- deprecate direct runtime graph assembly as the normal CLI path

Outcome:
web and CLI converge on one compiler/executor architecture.

### Stage 6: Remove duplicate late validation

- remove executor-primary route/error/quarantine/failsink validation
- keep only invariant checks that defend against corrupted artifacts or loader
  bugs
- deprecate `pipeline_yaml` as the primary persisted execution input
- keep a bounded rollback window via feature flags; remove shadow/legacy paths
  only after the expiry window closes

Outcome:
compile-time and runtime responsibilities are no longer duplicated.

### Stage 7: Detached executors and archival restore

- allow execution on another host/process that trusts compiler signatures and
  compatibility policy
- allow archival restore of sealed artifacts by digest for retained resume needs

Outcome:
remote or lower-trust execution becomes a policy decision, not a redesign.

### Stage exit criteria

Each migration stage must have a hard exit gate:

- **Stage 0 exit:** parity fixtures exist for representative linear, gate,
  aggregation, fork/coalesce, DIVERT, web-blob, and secret-bearing pipelines;
  secret-bearing fields have declared compiler placeholder kinds or equivalent
  normalizers; plugin-backed nodes have enforced audited identity
  (`plugin_version` + `code_hash`) per the plugin-version audit plan; logical
  digest and canonical version semantics are frozen; shadow-expiry and rollback
  flags are defined in CI-visible configuration
- **Stage 1 exit:** preview and seal produce identical validation results for
  the same input, seal output is canonical/deterministic under repeated runs,
  secret-bearing pipelines compile from unresolved inputs without plaintext
  secrets entering the artifact, and compatibility metadata is derived from
  audited plugin identity rather than raw plugin-version strings; preview
  failures return structured diagnostics to composer/CLI surfaces; compiler-bound
  requests are proven not to invoke legacy recursive `resolve_secret_refs()`
- **Stage 2 exit:** Landscape and session persistence can answer "which sealed
  artifact did this run use?" without consulting `pipeline_yaml`, and simulated
  crashes across publication windows leave either a recoverable `sealing` row, a
  `failed` row, or a `sealed` row — never a trusted session reference to a
  digest that Landscape cannot resolve as sealed; stale-row reconciliation
  proves byte-digest equality instead of trusting file presence; post-cutover
  runs without an authoritative compiled binding fail integrity checks rather
  than masquerading as legacy history
- **Stage 3 exit:** shadow mode proves equality for node IDs, route maps,
  branch metadata, topological order, pipeline order, branch-first-node maps,
  terminal-sink maps, route-destination DTO round-trips, and step map on every
  parity fixture; public graph APIs and GRAPH-phase orchestrator artifacts match
  between builder-loaded and artifact-loaded graphs; CI fails if shadow mode
  remains past its declared expiry
- **Stage 4 exit:** integration tests prove the authoritative web execution path
  succeeds while legacy YAML-to-runtime assembly calls are monkeypatched to
  raise outside the optional shadow comparator; signed artifact verification is
  required on the authoritative path; representative end-to-end runs show no
  row-outcome or audit-graph divergence; CLI resume refuses artifact-bound runs
  with a specific transition error until Stage 5 is active; executor lookup
  fails closed when Landscape is unreachable even if session hints exist
- **Stage 5 exit:** integration tests prove CLI run and `resume` execute from
  bound compiled artifacts while legacy settings-driven rebuild calls are
  monkeypatched to raise on the authoritative path; CLI and web share the same
  compiler output format and loader path; the Stage-4 transition rejection for
  artifact-bound CLI resume is removed because authoritative lookup+load now
  exists
- **Stage 6 exit:** behavioral tests prove invalid route/error/quarantine/failsink
  configs are rejected during preview/seal before orchestrator execution starts,
  and remaining runtime checks fire only for corrupted artifacts/loader bugs;
  rollback window policy is satisfied before legacy path deletion
- **Stage 7 exit:** detached execution trusts compiler signatures, rejects
  unsigned or mismatched artifacts per policy, and archived-artifact restore by
  digest supports retained resume cases

### Rollback plan

Every migration stage needs an explicit rollback path before it takes
authoritative traffic.

- **Stage 1 rollback:** disable compiler preview/seal surfaces and fall back to
  current validation/execute assembly while preserving any non-authoritative
  diagnostics work
- **Stage 2 rollback:** stop consuming compiled metadata, keep writing legacy
  execution inputs, and preserve `compiled_pipelines` rows as inert historical
  records
- **Stage 3 rollback:** disable artifact-loaded shadow execution and return to
  builder-only runtime assembly; keep mismatch evidence for diagnosis
- **Stage 4 rollback:** flip the web execution feature flag back to legacy
  execution while preserving sealed artifacts and signatures already recorded
- **Stage 5 rollback:** flip CLI run/resume back to legacy direct assembly while
  preserving authoritative compiled bindings for runs already created
- **Stage 6 rollback:** only allowed during the bounded rollback window before
  legacy-path deletion; after permanent deletion, rollback is a release rollback
  rather than a runtime switch
- **Stage 7 rollback:** disable detached execution and archival restore features
  independently of local artifact-first execution

Rollback must not mutate or delete already-recorded compile provenance. It
changes which path is authoritative for new executions, not the historical audit
record.

---

## File Map

### New contracts and compiler surface

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/contracts/compiler.py` | `CompilationRequest`, `CompilationInputAdapter`, `SecretBindingClaim`, compiler placeholder-kind types, `CompilationDiagnostic`, preview/seal result types, `CanonicalPipelineDefinition`, `CompiledPipeline`, compatibility/signature types |
| Create | `src/elspeth/core/compiler/service.py` | Compiler facade with preview/seal entry points |
| Create | `src/elspeth/core/compiler/adapters/` | YAML/settings/`ElspethSettings` adapters into `CompilationRequest` and canonical definition; no `web/` imports |
| Create | `src/elspeth/core/compiler/topology_dto.py` | Portable DTOs and codecs for node metadata, schema configs, route destinations, branch metadata, and authoritative traversal order |
| Create | `src/elspeth/core/compiler/secrets.py` | Normalize authoring-surface secret markers into `SecretBindingClaim`s and materialize compile-local placeholders for declared secret-bearing fields |
| Create | `src/elspeth/core/compiler/audit_projection.py` | Build `definition.audit_safe_config` from normalized compiler inputs without consulting placeholder-bearing runtime objects |
| Create | `src/elspeth/core/compiler/loader.py` | Runtime reconstruction from `CompiledPipeline` |
| Create | `src/elspeth/core/compiler/artifact_store.py` | Artifact publication abstraction with temp-write/`os.replace()` semantics, byte-digest verification, and stale-publication reconciliation helpers |
| Create | `src/elspeth/core/compiler/signing.py` | Signature envelope creation/verification for authoritative artifact execution |
| Create | `src/elspeth/core/compiler/parity.py` | Shadow-mode parity snapshot/comparison helpers for migration |
| Create | `src/elspeth/web/compiler/composition_state_adapter.py` | Adapt `CompositionState` plus web-only claims into `CompilationRequest` without creating an L1->L3 import |

### Existing engine and DAG code

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/elspeth/core/canonical.py` | Harden compiler canonicalization for `Enum`, canonical `Decimal`, and naive `time` handling; expose versioned seal semantics |
| Modify | `src/elspeth/contracts/hashing.py` | Keep contracts-layer canonical JSON behavior aligned with compiler canonicalization for `Enum` values and canonical `Decimal` handling |
| Modify | `src/elspeth/core/dag/builder.py` | Expose/pass through compiler phases instead of acting as hidden runtime builder |
| Modify | `src/elspeth/core/dag/graph.py` | Add artifact-load support and parity helpers; keep runtime query methods |
| Modify | `src/elspeth/cli_helpers.py` | Shared compile + execute helpers |
| Modify | `src/elspeth/cli.py` | Move CLI run/resume to authoritative artifact-bound execution during Stage 5 and reject artifact-bound resume during the Stage-4 transition instead of rebuilding from settings |
| Modify | `src/elspeth/core/checkpoint/recovery.py` | Bind resume to authoritative compiled artifacts and fail closed when retained artifacts are unavailable |
| Modify | `src/elspeth/engine/orchestrator/validation.py` | Mark late validations as migration shadow checks, then retire them from primary flow |
| Modify | `src/elspeth/engine/orchestrator/types.py` | Introduce/consume `RuntimeAssembly` wrapper semantics around current `PipelineConfig` |
| Modify | `src/elspeth/contracts/events.py` | Add compiler/artifact publication, verification, and shadow-mismatch observability events |

### Web execution path

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/elspeth/web/execution/validation.py` | Build a `CompilationRequest` via the web-side `CompositionState` adapter, then use compiler preview |
| Modify | `src/elspeth/web/execution/service.py` | Build a `CompilationRequest` via the web-side `CompositionState` adapter, eliminate recursive secret-tree resolution from the authoritative path, fail closed if Landscape authoritative lookup is unavailable, then seal/sign/verify/execute from compiled artifact while preserving blob/path claim checks |
| Modify | `src/elspeth/web/composer/yaml_generator.py` | Treat YAML as adapter output only, not the long-term execution contract |

### Persistence and provenance

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/elspeth/core/landscape/schema.py` | Add compiled-pipeline provenance metadata including logical/canonical version stamps and explicit binding-mode markers; do not overload sink artifacts |
| Create | `src/elspeth/core/landscape/compiled_pipeline_repository.py` | Own `sealing` / `sealed` / `failed` transitions, byte-verified stale-publication reconciliation, and fail-closed concurrent-seal handling |
| Modify | `src/elspeth/core/landscape/run_lifecycle_repository.py` | Persist compiled-digest bindings and verification outcomes on runs |
| Modify | `src/elspeth/web/sessions/models.py` | Add web-side compiled-digest hint/cache references |
| Modify | `src/elspeth/web/sessions/protocol.py` | Extend run/session contracts with non-authoritative compiled-digest hint metadata |
| Modify | `src/elspeth/web/sessions/service.py` | Persist compiled-digest hints/cache metadata only after Landscape sealing completes; treat session hints as untrusted lookup aids and support backfill/repair for cache lag |
| Modify | `src/elspeth/web/sessions/models.py` and `src/elspeth/web/sessions/schema.py` | Add session-side V0 schema changes for compiled-digest hint/cache fields; pre-release session DBs are recreated rather than migrated |

### CI, rollout, and observability

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scripts/cicd/check_compiled_pipeline_shadow_expiry.py` | Fail CI when shadow mode survives past its declared expiry without renewal |
| Create | `docs/runbooks/compiled-pipeline-rollout.md` | Rollout flags, rollback procedures, alert response, and retention/recovery operations |

### Tests

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `tests/unit/contracts/test_compiler_contracts.py` | `CompilationRequest`, diagnostic, preview/seal result, and `RuntimeAssembly` contract behavior |
| Create | `tests/integration/pipeline/test_compiled_pipeline_parity.py` | Graph/artifact parity against current builder semantics |
| Create | `tests/integration/web/test_compiled_pipeline_execution.py` | Web compile -> verify -> execute flow including blob/path claims and hard-fail behavior when Landscape authoritative lookup is unavailable |
| Create | `tests/integration/cli/test_compiled_pipeline_cli.py` | CLI compile/verify/execute path without authoritative legacy rebuild calls |
| Create | `tests/integration/cli/test_compiled_pipeline_resume.py` | Resume against authoritative compiled artifact bindings, Stage-4 transition rejection for artifact-bound runs before CLI cutover, retention failures, and archival restore behavior |
| Create | `tests/unit/core/compiler/test_artifact_serialization.py` | Canonical bytes, digest stability, no plaintext secrets |
| Create | `tests/unit/core/compiler/test_compatibility.py` | Audited plugin-identity binding, `code_hash` mismatch detection, and fail-closed behavior for missing plugin identity |
| Create | `tests/unit/core/compiler/test_digest_versions.py` | `logical_digest_version` / `canonical_version` semantics, cache-key invalidation, and historical-digest preservation |
| Create | `tests/unit/core/compiler/test_error_model.py` | Structured compiler rejection/error results for preview and seal |
| Create | `tests/unit/core/compiler/test_secret_binding_paths.py` | Fail-closed behavior for undeclared secret-ref paths, metadata injection attempts, and mixed legacy-tree-walk/compiler secret handling on the same request |
| Create | `tests/unit/core/compiler/test_signing.py` | Signature creation/verification and Stage-4 authoritative enforcement behavior |
| Create | `tests/unit/core/compiler/test_topology_dto_roundtrip.py` | DTO round-trip for `SchemaConfig`, `RouteDestination`, branch metadata, and authoritative order fields |
| Create | `tests/unit/core/compiler/test_artifact_store.py` | Atomic publish, stale `sealing` reconciliation with byte re-hash verification, concurrent-seal conflict handling, and no final artifact without a Landscape sealing record |
| Create | `tests/unit/core/compiler/test_observability.py` | Compiler/artifact metrics, alerts, and audit-first event emission expectations |
| Create | `tests/unit/core/compiler/test_loader.py` | Topology/load reconstruction invariants and public `ExecutionGraph` API parity |
| Create | `tests/unit/core/test_canonical_json.py` | Canonicalization coverage for `Enum`, normalized `Decimal`, and naive `time` rejection |
| Create | `tests/unit/contracts/test_hashing.py` | Contracts-layer canonical JSON coverage for `Enum` values and canonical `Decimal` normalization |

---

## Test Matrix

The compiled architecture is not complete until all of these pass:

| Scenario | Must prove |
|----------|------------|
| Linear source -> transform -> sink | Node IDs, edge labels, and step map are unchanged |
| Gate routes directly to sinks | Route resolution map and route labels survive compile/load |
| Gate routes to processing nodes | Continue semantics and default fallthrough survive compile/load |
| Fork + coalesce identity branches | Branch info and coalesce join behavior survive compile/load |
| Fork + transformed branch + coalesce | Producer/consumer resolution and branch schema mapping survive compile/load |
| Aggregation nodes | Aggregation node IDs and aggregation settings remain bound correctly |
| Source quarantine / transform error / sink failsink | DIVERT edges and destination validation remain correct |
| Coalesce policies (`require_all`, quorum, first/last) | Compiled topology preserves merge policy and schema guarantees |
| Source-only pipeline | No-op pipeline still compiles and executes correctly |
| Web blob-backed source | `blob_ref` survives as runtime claim and ownership is enforced |
| Path-bearing source/sink | Path claims are validated by executor policy, not silently trusted |
| Secret-bearing pipeline | Artifact serialization contains no plaintext secrets |
| Secret-bearing pipeline compiled from unresolved input | Preview/seal succeed without calling secret resolution first |
| Undeclared secret-bearing field | Compiler fails closed rather than inventing a placeholder or resolving the secret |
| Secret ref placed in undeclared metadata field | Compiler rejects with declared-path secret-binding error; no resolved secret enters names, audit records, or logs |
| Compiler-bound request through migration code | Legacy `resolve_secret_refs()` tree walk is never invoked on the same request |
| Compiler-ineligible open-ended mapping field | Raw `{"secret_ref": ...}` nested inside `dict[str, Any]` config is rejected or kept on a legacy-only path; it is never silently compiler-normalized by undeclared tree walk |
| Same semantics, different secret values | The same secret reference identity yields the same logical digest while run-time secret fingerprints remain part of run audit, not compile identity |
| Web secret-ref vs CLI env-var binding | Equivalent secret bindings normalize to the same logical secret-claim model |
| CLI and web equivalent logical pipeline | Same logical pipeline produces the same compiled digest when authoring claims are equivalent or absent |
| Normalization rule change | New `logical_digest_version` is required; old cache entries are not silently reused |
| Compiler placeholder scaffolding | Placeholder values never appear in serialized artifact bytes or audit-safe config |
| Audit-safe config construction | `definition.audit_safe_config` is built without consulting placeholder-bearing settings/plugin/runtime objects |
| Plugin code change with unchanged semver | Audited `code_hash` change invalidates compatibility even if `plugin_version` is unchanged |
| Missing plugin identity declaration | Compiler/executor fail closed when a referenced plugin lacks audited `plugin_version` or `code_hash` |
| Session DB digest hint differs from Landscape binding | Execution fails closed and records an integrity/security failure |
| Post-cutover run with null compiled binding | Integrity checks fail unless the run is explicitly marked legacy/pre-compiled |
| `RouteDestination` DTO round-trip | Serialized route destinations reconstruct the same runtime routing behavior |
| `SchemaConfig` DTO round-trip | Branch schemas and node schema configs survive artifact encode/decode exactly |
| Coalesce schema hydration | Loader recreates runtime schema validation semantics without relying on process-global class counters |
| Topological order parity | Loaded graph returns the same `topological_order()` as the builder graph |
| Graph traversal parity | Loaded graph matches `get_pipeline_node_sequence()`, `get_branch_first_nodes()`, `get_branch_to_sink_map()`, and `get_terminal_sink_map()` |
| GRAPH-phase parity | `_register_graph_nodes_and_edges()`-level artifacts match between builder-loaded and artifact-loaded graphs |
| Enum-valued config leaf | Sealing canonicalizes through stable Enum value rather than raising `TypeError` |
| Equivalent finite Decimal values | `Decimal("1.0")` and `Decimal("1.00")` produce the same canonical artifact identity |
| Contracts-layer Enum/Decimal normalization | `contracts/hashing.py` matches compiler canonicalization rules instead of diverging or raising `TypeError` |
| Naive `time` object in config | Compiler rejects before sealing rather than emitting timezone-ambiguous digests |
| Compile request hits existing sealed digest | Publication returns a cache hit and does not rewrite artifact bytes or provenance rows |
| Concurrent seal attempt on same digest while status is `sealing` | Losing attempt fails closed and no duplicate `compiled_pipelines` row appears |
| Reconciler sees artifact bytes at expected digest path | It re-hashes file bytes and refuses to promote `sealing` -> `sealed` on existence/path name alone |
| Crash after Landscape `sealing` row, before artifact rename | No published artifact exists; stale Landscape row reconciles to `failed` or is resumed safely |
| Crash after artifact rename, before Landscape `sealed` update | Reconciler can promote the stale `sealing` row to `sealed` by verifying the final artifact bytes |
| Landscape unavailable during authoritative verify/execute | Execution/resume fails closed and does not trust session-side digest hints |
| Stage-4 CLI resume of web-originated artifact-bound run | CLI rejects with a transition error instead of rebuilding topology from current settings |
| Crash after Landscape `sealed`, before session DB update | Audit provenance remains authoritative and session DB reference can be backfilled without ambiguity |
| Resume of run whose compiled artifact was purged | Resume fails with artifact-unavailable error unless archival restore succeeds by digest |
| Authoritative web execution path | Integration test succeeds while legacy YAML/runtime-build calls are patched to raise outside shadow mode |
| Authoritative CLI run/resume path | Integration test succeeds while legacy settings/runtime-build calls are patched to raise outside migration-only code |
| Signed authoritative execution | Web/CLI artifact-first execution rejects unsigned or signature-mismatched artifacts once Stage 4 is active |

### Migration parity checks

During shadow mode, compare at minimum:

- node table
- edge table
- sink/transform/aggregation/gate/coalesce maps
- route resolution map
- route label map
- branch gate map / branch-to-sink map / coalesce branch schemas
- branch info
- topological order
- pipeline node order
- branch-first-node map
- terminal-sink map
- node step map
- audit-safe config digest
- `compute_full_topology_hash(graph)`
- public `ExecutionGraph` API outputs used by the orchestrator
- GRAPH-phase orchestrator artifacts derived from each graph

Any mismatch is a migration blocker.

---

## Acceptance Criteria

1. One logical pipeline compiles to one canonical `CompiledPipeline` artifact
   with a stable digest.
2. Deterministic node IDs are assigned during compilation and reused unchanged
   during execution.
3. Compiled topology includes route maps, branch info, pipeline order, and node
   step map sufficient to rebuild runtime graph/query objects without YAML
   recompilation.
4. The compiled artifact contains no resolved secret values.
5. Web-only execution claims such as blob ownership survive compilation through
   explicit runtime-claim fields rather than leaking into engine config.
6. The executor combines `CompiledPipeline` with local `ExecutionPolicy`
   instead of expecting deployment settings to be embedded in the artifact.
7. Landscape records authoritative compile provenance and run binding using
   dedicated metadata rather than the sink `artifacts` table.
8. Web execution runs from a verified compiled artifact without reparsing YAML as
   the primary runtime input.
9. CLI execution can use the same compiler/executor boundary without losing its
   current runtime config model.
10. Compile-time topology and contract validation are no longer duplicated as
    independent primary logic in validation, web execution, CLI execution, and
    orchestrator setup.
11. Shadow-parity tests prove that artifact-loaded execution matches current
    production semantics for gates, aggregations, coalesce, and DIVERT flows.
12. Equivalent web and CLI authoring inputs normalize to the same
    `definition.logical_digest` when runtime claims are equivalent.
13. Secret binding identity is preserved in compiled runtime claims, while
    resolved secret fingerprints remain run-scoped audit data only.
14. Compiler preview/seal operate on unresolved secret bindings only; secret
    resolution services and plaintext values remain executor-only.
15. Secret-bearing fields use declared compiler placeholder kinds or equivalent
    compiler normalizers; missing declarations fail compilation closed.
16. Publication across Landscape, artifact storage, and the session DB uses a
    defined `sealing` -> `sealed` / `failed` state machine rather than an
    implied multi-store transaction.
17. `compiled_pipelines.compiled_digest` is a unique publication key:
    existing `sealed` rows behave as cache hits, while live `sealing`
    conflicts fail closed without duplicate provenance rows.
18. Published artifact bytes use atomic temp-write + `os.replace()` semantics
    at the final digest path.
19. Reconciliation promotes stale `sealing` rows only after byte re-hashing the
    final artifact and proving SHA-256 equality with the recorded digest; file
    presence or path naming alone is never trusted.
20. Session DB references are cache-only and are never written before
    Landscape reports the compiled artifact as `sealed`.
21. Canonicalization rules are mechanically aligned across
    `core/canonical.py` and `contracts/hashing.py`, including `Enum` ->
    normalized `.value` and `Decimal` ->
    `format(obj.normalize(), "f")`, before seal digests become authoritative.
22. `definition.audit_safe_config` is built from a placeholder-free projection
    of normalized compiler inputs, never from placeholder-bearing runtime
    objects or transient settings/plugin instances.
23. Portable artifact topology stores DTOs for schemas, routing, branch
    metadata, and authoritative order; it does not serialize runtime-only graph
    classes or NetworkX internals directly.
24. Loader parity is defined at the public `ExecutionGraph` API and
    orchestrator graph-phase behavior level, not just by equality of static
    tables.
25. Artifact compatibility metadata is derived from audited plugin
    implementation identity (`plugin_version` + `code_hash`), not from today's
    unaudited constant/default version strings.
26. `CompilationRequest`, compiler diagnostics/results, and `RuntimeAssembly`
    are explicit contracts with named fields and machine-checkable behavior.
27. `definition.logical_digest` is version-stamped, and normalization/canonical
    changes create new declared versions rather than silently reinterpreting
    historical digests.
28. Secret references are accepted only on declared secret-bearing paths;
    undeclared-path secret refs fail closed before any resolution/materialization.
29. Compiler-bound requests never share authoring-surface secret handling with
    the legacy recursive `resolve_secret_refs()` path.
30. Landscape is the only authoritative digest source; session-side digest data
    is hint-only, and Landscape unavailability fails closed instead of
    downgrading trust to hints or cache state.
31. Resume is bound to the original sealed artifact (or restored copy by
    digest), not to a fresh rebuild from current settings/YAML.
32. During the Stage 4 -> Stage 5 split, CLI resume rejects artifact-bound runs
    rather than rebuilding them from current settings.
33. Post-cutover runs cannot silently carry null compiled bindings; legacy
    historical nulls are mechanically distinguished from integrity failures.
34. Authoritative artifact-first execution uses verified signatures no later
    than Stage 4.
35. Compiler/artifact publication, verification, shadow mismatch, and resume
    health have defined telemetry, dashboards, and alerts consistent with audit
    primacy.
36. Each migration stage has a documented rollback path and machine-verifiable
    exit criteria.

---

## Alternatives Rejected

### 1. Rename the existing builder to "compiler" and stop there

Rejected. That changes terminology, not architecture. Runtime would still
rebuild from authoring input on every execution.

### 2. Use normalized YAML as the execution artifact

Rejected. YAML does not capture the full compiled boundary:

- deterministic node IDs
- route resolution maps
- branch info
- step numbering
- validation evidence
- compatibility fingerprint

### 3. Make `PipelineConfig` the artifact

Rejected. It contains live plugin instances and executor-owned runtime material.
It is a runtime assembly, not a portable compile product.

### 4. Store compiled pipelines in the existing Landscape `artifacts` table

Rejected. That table models sink outputs and requires `produced_by_state_id`,
which does not fit compiler products.

### 5. Push path/blob/secret checks fully into compilation

Rejected. Those checks depend on local execution policy and ownership context.
They may be previewed early, but the executor remains authoritative.

### 6. Use one global placeholder string for all secret-bearing fields

Rejected. Shape-blind sentinels do not satisfy real validators and are too easy
to leak into sealed artifacts.

### 7. Widen plugin/runtime config fields to `SecretRef | str`

Rejected. That makes unresolved secret markers a general runtime type instead of
an explicit compiler/runtime boundary concern.

---

## Remaining Open Questions

No blocking architecture questions remain for the first increment.

Later implementation work may still refine:

1. the exact shape of a session-side compiled-artifact cache table if reuse
   pressure materializes
2. detached-executor key distribution, rotation, and trust-delegation policy
   once multi-host execution is introduced beyond the Stage-4 signed local/web
   model
