# L2 #5 — `contracts/` cluster catalog

## Conventions

Per Δ L2-3, this catalog lists ONE entry per immediate subdirectory or coherent file group at L2 depth. Files >1,500 LOC are flagged as L3 deep-dive candidates; their bodies are NOT opened. Citations:

- `[ORACLE: ...]` — refers to `temp/intra-cluster-edges.json` (Δ L2-2 filtered oracle), `temp/layer-conformance-contracts.json` (Δ L2-6 layer-check summary), or `temp/l3-import-graph.json` (L1 oracle, read-only context).
- `[CITES KNOW-*]` — entry in `00b-existing-knowledge-map.md`.
- `[DIVERGES FROM KNOW-*]` — same, but the catalog records contradicting evidence.
- `(file:line)` — direct file-line citation; verifiable.
- `Supplements 02-l1-subsystem-map.md §1` — the L1 catalog entry this entry deepens (without contradicting).

**Cluster-internal SCC handling:** N/A. `[ORACLE: temp/intra-cluster-edges.json stats.sccs_touching_cluster = 0]` and SCC list contains only L3 nodes; contracts/ is L0.

---

## Entry 1 — `contracts/__init__.py` — public re-export surface

**Path:** `src/elspeth/contracts/__init__.py`

**Responsibility:** The L0 public surface — re-exports 208 names from 30+ submodules; documents the leaf invariant inline; enforces the post-ADR-006 boundary by *non-export* (Settings classes are deliberately excluded).

**File count, LOC:** 1 file, 535 LOC (read in full).

**Internal coupling:** Imports from approximately 30 sibling modules (`aggregation_checkpoint`, `audit`, `batch_checkpoint`, `call_data`, `checkpoint`, `cli`, `coalesce_*`, `config`, `contexts`, `contract_builder`, `contract_propagation`, `contract_records`, `data`, `diversion`, `engine`, `enums`, `errors`, `events`, `header_modes`, `identity`, `node_state_context`, `payload_store`, `pipeline_runner`, `plugin_context`, `plugin_protocols`, `probes`, `results`, `routing`, `schema_contract`, `schema_contract_factory`, `sink`, `token_usage`, `transform_contract`, `type_normalization`, `types`, `url`). All star-style imports; the `__all__` list is closed and explicit.

**External coupling:** Inbound from every L1+ subsystem (per L1 oracle and CLAUDE.md "Layer Dependency Rules"); outbound zero `[ORACLE: temp/intra-cluster-edges.json stats.outbound_edge_count = 0]`. The `__init__.py` itself imports nothing from above L0.

**Patterns observed:**

- The `__all__` list is **closed and grouped by category** (`# audit`, `# errors`, `# schema contract violations`, `# config — Runtime protocols`, etc.) at lines 290–535. Total of 208 names (verified `grep -cE '^    "' src/elspeth/contracts/__init__.py` → 208).
- Lines 79–87 contain a comment block — load-bearing institutional memory — explicitly stating: "Settings classes (RetrySettings, ElspethSettings, etc.) are NOT here. Import them from elspeth.core.config to avoid breaking the leaf boundary." This is the post-ADR-006 boundary discipline encoded in code comment, not just in a doc.
- Lines 7–10 ("This package is Layer 0 (L0) ... has NO runtime imports from core, engine, or plugins. This is enforced by CI (`enforce_tier_model.py` with rule L1)") — the package docstring ratifies KNOW-A53 / KNOW-C47.
- A second comment block at lines 369–370 ("NOTE: Settings classes (RetrySettings, ElspethSettings, etc.) are NOT here / Import them from elspeth.core.config to avoid breaking the leaf boundary") repeats the discipline at the `__all__` site itself.

**Concerns:**

- No dedicated unit test pinning the `__all__` list. A regression here (accidental Settings re-export, or a name removed without notice) would surface only as a downstream import failure. **L2 debt candidate** — a minimal `tests/unit/contracts/test_public_surface.py` asserting expected groupings and the absence of Settings-shaped names would lock the boundary.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1` (`contracts/`, "Internal sub-areas (single line): Top-level contracts modules ... plus the config/ subpackage").

**[CITES KNOW-A53]** (zero-outbound leaf), **[CITES KNOW-C47]** (4-layer model), **[CITES KNOW-ADR-006]** (4-layer remediation).

**Confidence:** **High** — read in full; 208 `__all__` count verified mechanically.

---

## Entry 2 — `contracts/config/` — Settings → Runtime alignment

**Path:** `src/elspeth/contracts/config/`

**Responsibility:** The Settings ↔ Runtime contract sub-package — protocol definitions, runtime config dataclasses, defaults registries, and the alignment documentation that ties Pydantic Settings (in `core/`) to runtime dataclasses (here).

**File count, LOC:** 5 files, 1,231 LOC (`runtime.py` 655, `protocols.py` 209, `alignment.py` 181, `__init__.py` 112, `defaults.py` 74).

**Internal coupling:** `config/runtime.py` imports from `contracts.config.defaults`, `contracts.engine` (RetryPolicy), `contracts.enums`, `contracts.freeze` (`runtime.py:25-28`). `config/__init__.py` re-exports the sub-package surface to `contracts/__init__.py:88-105`.

**External coupling:** Inbound from `core/config/` (Pydantic Settings classes call into `from_settings()` factories defined here) and from every L2+ component that consumes runtime config. Outbound zero at runtime; one `if TYPE_CHECKING: from elspeth.core.config import (CheckpointSettings, ConcurrencySettings, RateLimitSettings, RetrySettings, TelemetrySettings)` block at `runtime.py:30-37` — annotation-only, no runtime coupling. **[CITES KNOW-A11]** (Container list — `Core` and `Contracts` named separately, with config primitives split between them by ADR-006).

**Patterns observed:**

- **Frozen + slots dataclasses with `from_settings()` factories** (`runtime.py:1-17` design principles): "1. Frozen (immutable) ... 2. Slots ... 3. Protocol compliance — implements Runtime*Protocol for structural typing 4. Factory methods — from_settings(), from_policy(), default(), no_retry()". This is the Settings → Runtime protocol pattern documented by the `config-contracts-guide` skill.
- **Field-origin tagging** (`runtime.py:11-15`): "Field Origins: Settings fields: Come from user YAML configuration via Pydantic models / Internal fields: Hardcoded implementation details, documented in INTERNAL_DEFAULTS". `defaults.py` is the registry of internal-default constants; `alignment.py` is the documentation contract.
- **Five Runtime protocols** re-exported through `__init__.py:99-104`: `RuntimeCheckpointProtocol`, `RuntimeConcurrencyProtocol`, `RuntimeRateLimitProtocol`, `RuntimeRetryProtocol`, `RuntimeTelemetryProtocol`. Defined in `protocols.py` (read length-only at 209 LOC; not opened beyond import-list confirmation).

**Concerns:**

- **`config/runtime.py:338` — shallow MappingProxyType wrap.** The line is `object.__setattr__(self, "services", MappingProxyType(dict(self.services)))`. Per CLAUDE.md §"Frozen Dataclass Immutability" §"When Shallow Wrapping IS Acceptable", this is acceptable **only when values are guaranteed immutable** — which holds here because `RuntimeServiceRateLimit` is `@dataclass(frozen=True, slots=True)` (`runtime.py:291`). The shallow wrap is therefore not a correctness violation, but it is the single non-uniform freeze pattern in the entire contracts cluster (every other dataclass uses `freeze_fields(self, ...)`). **L2 debt candidate** — switching to `freeze_fields(self, "services")` would unify the pattern across the cluster and remove one cited exception.
- The `from_settings()` factories are an architectural seam: they exist precisely so that `contracts/` can stay leaf while `core/config/` (Pydantic) calls into them. The TYPE_CHECKING-only annotation block (`runtime.py:30-37`) is the L0/L1 boundary contract made visible. KNOW-A53 holds.

**Test evidence:** `tests/unit/contracts/config/` contains 5 dedicated test files (`test_alignment.py`, `test_protocols.py`, `test_runtime_checkpoint.py`, `test_runtime_common.py`, `test_runtime_concurrency.py`, `test_runtime_rate_limit.py`, `test_runtime_retry.py`). The `services` shallow-wrap pattern is exercised by `test_runtime_rate_limit.py`. Integration test `tests/integration/contracts/test_build_runtime_consistency.py` is the single cross-layer alignment test.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1`'s naming of `config/` as a sub-area.

**[CITES KNOW-C33]** (config-contracts verification script), **[CITES KNOW-ADR-006]** (Phase 4 RuntimeServiceRateLimit relocation), **[CITES KNOW-C47]** (4-layer model — Settings live in L1, Runtime protocols in L0).

**Confidence:** **High** — header read in full; one targeted body read at `:325-345` confirmed the freeze observation.

---

## Entry 3 — `contracts/freeze.py` — immutability primitives

**Path:** `src/elspeth/contracts/freeze.py`

**Responsibility:** The canonical implementation of `deep_freeze` (recursive frozen-container conversion), `freeze_fields` (frozen-dataclass `__post_init__` helper), `deep_thaw` (reverse for JSON serialization), and `require_int` (Tier-1 numeric validation rejecting `bool`).

**File count, LOC:** 1 file, 172 LOC (read in full).

**Internal coupling:** No imports from sibling contracts modules. Only stdlib (`collections.abc.Mapping`, `types.MappingProxyType`, `typing.Any`, `dataclasses`).

**External coupling:** **Inbound from 18 contracts files** (verified via `grep -lE "freeze_fields\(self" src/elspeth/contracts/*.py src/elspeth/contracts/config/*.py`); 33 invocations of `freeze_fields(self, ...)` total. Inbound from many core/ and engine/ files as well (out-of-scope for this entry). Outbound zero.

**Patterns observed:**

- **`deep_freeze` invariants** (`freeze.py:23-75`): `dict` → `MappingProxyType` (line 47); `list` → `tuple` (line 49); `MappingProxyType` → fresh `MappingProxyType` of fresh dict (line 53–54, with the explicit comment "MappingProxyType is a READ-ONLY VIEW, not a detached copy"); `tuple` recurses with identity-preserving idempotency (lines 55–59); `set` → `frozenset` (line 61); `frozenset` recurses with set-equality identity check (lines 62–68); arbitrary `Mapping` ABCs handled at `:72` (covers `OrderedDict`, custom read-only wrappers); scalars/dataclasses pass through (`:74-75`).
- **`freeze_fields` field-name validation** (`freeze.py:99-104`): rejects names not declared on the dataclass with a sorted error message. Identity-preserving idempotency at `:108-109` skips `object.__setattr__` when `deep_freeze(value) is value`. **This is the canonical pattern** per CLAUDE.md "The Canonical Pattern".
- **`require_int`** (`freeze.py:142-171`): Tier-1 offensive validation; **rejects `bool` because `isinstance(True, int) is True`** in Python (`:168`). This is the offensive-programming primitive every numeric `__post_init__` in the cluster relies on. Cited by `audit.py:16`, `config/runtime.py:28`, `errors.py` (via downstream usage), and others.

**Concerns:**

- None observed at L2 depth. The module is small, well-documented, dependency-free, and enforces the contract it advertises. CI gate `enforce_freeze_guards.py` (KNOW-C65) externally enforces correct usage.

**Test evidence:** `tests/unit/contracts/test_freeze.py` and `test_freeze_regression.py`. The regression file's existence implies a prior bug was locked in — likely worth noting at synthesis but not opened here (would constitute opening test bodies, which is L3 work).

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1` (the "freeze primitives" sub-area mentioned in its single-line internal-areas summary).

**[CITES KNOW-C62]** (canonical pattern is `freeze_fields(self, "field1", "field2")`), **[CITES KNOW-C63]** (`deep_freeze` recursive contract), **[CITES KNOW-C65]** (CI enforcement via `enforce_freeze_guards.py`), **[CITES KNOW-C64]** (frozen-dataclass guard requirement).

**Confidence:** **High** — file read in full; usage census verified mechanically.

---

## Entry 4 — `contracts/hashing.py` — canonical JSON + stable hashing

**Path:** `src/elspeth/contracts/hashing.py`

**Responsibility:** Canonical JSON serialization (RFC 8785/JCS) and stable hashing for data containing JSON-safe primitives and their frozen equivalents (`MappingProxyType`, `tuple`). Defines `CANONICAL_VERSION = "sha256-rfc8785-v1"` — the single source of truth for hash-version identification.

**File count, LOC:** 1 file, ~150 LOC (read first 60 lines; rest cited by file:line where needed).

**Internal coupling:** stdlib (`hashlib`, `math`, `collections.abc.Mapping`, `typing.Any`) plus the third-party `rfc8785` library (line 21). No imports from sibling contracts modules.

**External coupling:** Inbound from `core/canonical.py` (per `hashing.py:8-11`: "This module exists to break the circular dependency between contracts/ and core/canonical.py. ... Single source of truth — core/canonical.py imports this constant" and `:24-25`). Inbound from `schema_contract.py:11`. Outbound zero.

**Patterns observed:**

- **Cycle-breaking by extraction** (`hashing.py:8-11`): the entire module's reason for existence is to break the contracts ↔ core/canonical cycle. This is **direct evidence of ADR-006 Phase 2** ([CITES KNOW-ADR-006b] "Phase 2 extracts `contracts/hashing.py`"). The module header explicitly cites the cycle and the partition.
- **Single-pass normalization with rejection** (`hashing.py:28-54`): `_normalize_frozen_and_reject_non_finite` does both jobs in one traversal — converts frozen → mutable equivalents AND rejects non-finite floats / `frozenset`. The rejection messages (`:39-42`, `:45-47`) are explicit about why ("Use None for missing values, not NaN", "frozenset is not JSON-serializable and has no canonical ordering").
- **Constant export** (`hashing.py:25`): `CANONICAL_VERSION` is a module-level constant; `core/canonical.py` imports it. This is a one-way constant flow up the layer stack — L0 defines, L1 consumes — which is the correct direction.

**Concerns:**

- None at L2 depth. The module is small, single-purpose, and explicitly motivated by an ADR.

**Test evidence:** `tests/unit/contracts/test_hashing.py`.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1` (sub-area: "hashing").

**[CITES KNOW-ADR-006b]** (Phase 2 extracted `contracts/hashing.py`); **[CITES KNOW-A69]** (implicit technology choices include RFC 8785 / `rfc8785` library — the implementation primitive this module wraps).

**Confidence:** **High** — header read in full; cycle-breaking rationale verified verbatim.

---

## Entry 5 — `contracts/audit.py` — strict-typed audit DTO surface

**Path:** `src/elspeth/contracts/audit.py`

**Responsibility:** The strict-typed dataclass surface for Landscape audit-trail rows — `Run`, `Token`, `Operation`, `Call`, `Batch`, `BatchMember`, `BatchOutput`, `BatchStatusUpdate`, `Checkpoint`, `Edge`, `Node`, `NodeState` and its four state subclasses (`NodeStateOpen`, `NodeStatePending`, `NodeStateCompleted`, `NodeStateFailed`), `Row`, `RowLineage`, `RoutingEvent`, `SecretResolution`/`SecretResolutionInput`, `Token`, `TokenOutcome`, `TokenParent`, `TransformErrorRecord`, `ValidationErrorRecord`, `Artifact`, `NonCanonicalMetadata`, `ExportStatusUpdate`. Header (`:1-25`) cites the Data Manifesto (Tier 1 Our Data — full trust).

**File count, LOC:** 1 file, 922 LOC. **Header-only inspection** per Δ L2-3.

**Internal coupling:** Imports from `contracts.freeze` (`:16` — `freeze_fields`, `require_int`), `contracts.enums` (`:21-` — `BatchStatus`, `CallStatus`, `CallType`, `Determinism`, et al.), and others (full list deferred — body not opened).

**External coupling:** Inbound from `core/landscape/` (the Landscape repository tier writes these dataclasses), `engine/` (executors construct them), and many others (`audit_protocols.py:25` re-imports `TokenRef` from here). Outbound zero.

**Patterns observed (header):**

- **Tier-1 / Data Manifesto framing** (`audit.py:5-8`): "Per Data Manifesto: The audit database is OUR data. If we read garbage from it, something catastrophic happened - crash immediately." This is the L0 expression of CLAUDE.md's Tier-1 trust contract — the dataclasses here are the type-shape that the Landscape repository tier must hand back unchanged.
- **All enum fields use proper enum types** (`audit.py:3-5`): "These are strict contracts - all enum fields use proper enum types. Model loader layer handles string→enum conversion for DB reads." The model-loader layer (in `core/landscape/`) is the boundary; the contracts dataclasses do not coerce.

**Concerns:**

- **`audit.py` is the largest non-flagged file** (922 LOC, ~60% of the L3 threshold). It is a proximal candidate for Δ4 sub-cluster reconsideration in any future L3 dive. **Not** an L3 deep-dive candidate at this pass; the threshold is 1,500 LOC.
- The 922 LOC distributed across ~25 dataclasses is dense but flat (no class hierarchies). Re-validating the boundary at the L3 level would be a class-by-class invariant audit — out of scope here.

**Test evidence:** `tests/unit/contracts/test_audit.py`.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1` (sub-areas: "audit_evidence" + downstream audit DTOs).

**[CITES KNOW-A17]** (Landscape ~8,300 LOC structured as facade + 4 repositories); **[CITES KNOW-A29]** (LandscapeRecorder facade + 4 repository delegation); **[CITES KNOW-A59]** (Tier-1 Audit DB full trust, never coerce, crash on anomaly).

**Confidence:** **Medium** — header inspected; body not opened. The 25 named DTOs are inferred from the `__init__.py` import list (`:28-57`) plus header skim, not from line-by-line read.

---

## Entry 6 — Audit-evidence framework (`audit_evidence.py`, `audit_protocols.py`, `declaration_contracts.py`)

**Path:** `src/elspeth/contracts/audit_evidence.py`, `audit_protocols.py`, `declaration_contracts.py`

**Responsibility:** The L0 surface of ADR-010 — nominal ABC for audit-bearing exceptions, cross-layer audit recorder protocols, and the 4-site declaration-contract dispatcher framework.

**File count, LOC:** 3 files; `audit_evidence.py` ~75 LOC (read in full), `audit_protocols.py` ~250 LOC (header read `:1-35`), `declaration_contracts.py` 1,323 LOC (header read `:1-58`, body deferred — near-threshold per Δ L2-3).

**Internal coupling:** `audit_evidence.py` imports only stdlib (`abc.ABC`, `abc.abstractmethod`, `collections.abc.Callable`, `collections.abc.Mapping`, `typing.Any`, `cast`). `audit_protocols.py` imports from `contracts.audit` (`:25` — `TokenRef`), `contracts.call_data`, `contracts.errors`, `contracts.schema_contract`, plus 8 sibling type re-exports from `contracts/__init__.py` directly (`audit_protocols.py:15-24`). `declaration_contracts.py` imports from `contracts.freeze` (per header — body not opened) and others; per the header it depends on `audit_evidence.AuditEvidenceBase` (because `DeclarationContractViolation` is described as "audit-evidence-bearing" at `:25-26`).

**External coupling:** Inbound from `engine/executors/` (the dispatcher in `engine/declaration_dispatch.py` — engine-cluster catalog entry — registers and invokes the contracts defined here), from contract-adopter files at `engine/executors/declaration_*.py`, and from CI scanner `scripts/cicd/enforce_audit_evidence_nominal.py` ([CITES KNOW-ADR-010a]). Outbound zero.

**Patterns observed:**

- **Nominal ABC over Protocol** (`audit_evidence.py:8-9`, ADR-010 §Decision 1): "A structural Protocol was rejected (see ADR-010 §Alternative 3) because single-method @runtime_checkable Protocols admit accidental duck-type matches from unrelated classes". This is the closure of the spoofing vector ([CITES KNOW-ADR-010]). The ABC enforcement uses `__init_subclass__` to install a checked `__init__` wrapper (`audit_evidence.py:53-62`) because **CPython 3.13 routes `BaseException.__new__` through a C-level fast-path that bypasses `ABCMeta.__call__`** (`audit_evidence.py:32-38`) — without this, exception subclasses with abstract methods would instantiate. The implementation note is load-bearing institutional memory.
- **4-site dispatch framework** (`declaration_contracts.py:1-13` header): `DispatchSite` StrEnum names the four sites; `@implements_dispatch_site("site_name")` decorator is "the authoritative signal under multi-level inheritance (AST scanner cannot reliably see mixin-inherited overrides)". The L0 placement is **mandatory per plan-review W4** — the CI scanner at L3 imports it.
- **Audit-complete dispatch** (`declaration_contracts.py:35-43`): "iterates every applicable contract for a given dispatch site. Each applicable contract's method runs; raised violations are collected rather than short-circuiting. ... 0 violations → return; 1 → raise `violations[0]` via reference equality (N6 regression invariant); >=2 → wrap in aggregate, raise. This closes the audit-trail silence that fail-fast first-fire would have made indistinguishable from 'checked and passed' (STRIDE Repudiation)." This is the CONTRACTS-side specification of the engine-side semantics that the engine cluster catalogued at `engine/declaration_dispatch.py:1-26`.
- **Deny-by-default payload schemas (H5)** (`declaration_contracts.py:25-26`, [CITES KNOW-ADR-010h]): "Subclasses declare `payload_schema` (H5 Layer 1)" — every `DeclarationContractViolation` payload is validated at construction (deny-by-default before deep-freeze).
- **Aggregate is a sibling class** (`declaration_contracts.py:28-31`): `AggregateDeclarationContractViolation` is a SIBLING of `DeclarationContractViolation`, not a subclass. Closure for security S2-001 — the aggregate's `is_aggregate: True` payload field replaces a sentinel-string-in-name-column anti-pattern (Spoofing surface).

**Concerns:**

- **`declaration_contracts.py` at 1,323 LOC is at ~80% of the L3 threshold.** Near-threshold; not flagged but worth noting in synthesis. The body contains four bundle-input/output dataclass families (`PreEmissionInputs`, `PostEmissionInputs`/`Outputs`, `BatchFlushInputs`/`Outputs`, `BoundaryInputs`/`Outputs`) plus the dispatcher-consumed registry mechanics. Future growth (a fifth dispatch site, or per-site aggregation policies) would push this file over.
- **Pytest-gated registry helpers** (`declaration_contracts.py:54-57` per header): `_clear_registry_for_tests`, `_snapshot_registry_for_tests`, `_restore_registry_snapshot_for_tests` are "pytest-gated; production callers raise". Standard test seam; the gating mechanism should be verified at L3 dive time but the existence is conventional.

**Test evidence:** `tests/unit/contracts/test_audit_evidence.py`, `test_audit_evidence_nominal_scanner.py` (the latter pins the CI scanner contract — KNOW-ADR-010a), `test_audit_protocols.py`, `test_declaration_contracts.py`.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1` (sub-areas: "audit_evidence", "declaration_contracts"). Engine cluster's "Cross-cluster observations for synthesis" §"engine ↔ contracts.declaration_contracts (ADR-010 payloads)" is **closed by this entry**.

**[CITES KNOW-ADR-010]**, **[CITES KNOW-ADR-010a]**, **[CITES KNOW-ADR-010h]**, **[CITES KNOW-ADR-006a]**.

**Confidence:** **High** for the audit_evidence and audit_protocols files (header + class-level inspection); **Medium** for `declaration_contracts.py` (header only; body inferred from header semantics + tests + engine cluster catalog cross-reference).

---

## Entry 7 — Tier-1 registry primitives (`tier_registry.py`, `registry_primitive.py`)

**Path:** `src/elspeth/contracts/tier_registry.py`, `src/elspeth/contracts/registry_primitive.py`

**Responsibility:** The Tier-1 exception registry (ADR-010 §Decision 2) and its underlying ordered-list+aux-map+freeze-flag primitive. Defines `FrameworkBugError` (the canonical "framework internal inconsistency" exception) and the `@tier_1_error(reason=...)` factory decorator that registers exception classes into a frozen registry.

**File count, LOC:** 2 files; `tier_registry.py` ~150 LOC (header read `:1-50`), `registry_primitive.py` ~100 LOC (header read `:1-35`).

**Internal coupling:** `tier_registry.py:26` imports `FrozenRegistry` from `contracts.registry_primitive`. `registry_primitive.py` imports only stdlib (`collections.abc.Callable`, `collections.abc.Iterator`, `contextlib.contextmanager`, `threading.RLock`, `typing.TypeVar`).

**External coupling:** Inbound from `contracts/errors.py:21-22` (re-exports `FrameworkBugError` to break circular import) and from every site that uses `@tier_1_error` (per `tier_registry.py:8-11` the module-prefix allowlist permits `elspeth.contracts.*`, `elspeth.engine.*`, `elspeth.core.*`, plus `tests.*` under pytest). Outbound zero.

**Patterns observed:**

- **Three reviewer-identified safety mechanisms** (`tier_registry.py:4-14`): (1) factory decorator requires `reason` kwarg → auditors can grep; (2) module-prefix allowlist → plugin modules cannot self-elevate; (3) `freeze_tier_registry()` at end of bootstrap → post-freeze registration raises `FrameworkBugError`. Each cites a specific reviewer (B6/F-6, B5/F-2). This is the **direct provenance of post-review hardening**.
- **Forward-declared `FrameworkBugError`** (`tier_registry.py:16-17`, `:29-48`): the class is **defined here first** so `errors.py` can apply `@tier_1_error` to it without a circular import. The errors.py side reads `from elspeth.contracts.tier_registry import FrameworkBugError as _FrameworkBugError` (`errors.py:21`) and then `FrameworkBugError = tier_1_error(reason=...)(_FrameworkBugError)` (`errors.py:24-27`). The result: external callers see one canonical `FrameworkBugError` (identical class object), but the decoration happens at the canonical Tier-1 declaration site (errors.py) without forming an import cycle.
- **Single RLock + ordered list + auxiliary map + freeze flag** (`registry_primitive.py:1-9`): the shared primitive used by both the Tier-1 registry and the declaration-contract registry. "Keep that machinery here so new registries do not re-learn the same concurrency and post-bootstrap rules by copy/paste."

**Concerns:**

- **Pytest-allowance widens the module-prefix allowlist.** `tier_registry.py:9-12` notes "Under pytest, `tests.*` is additionally allowed for test-only fixtures." This is a deliberate test-seam, but it means a runtime test discriminator gates the registration policy. If `pytest` could be detected falsely under production, plugin code could elevate. **L2 debt candidate** — a test pinning the discriminator (e.g., asserting that the pytest-detection is a one-shot import-time check, not a runtime probe) would lock the seam.

**Test evidence:** `tests/unit/contracts/test_tier_registry.py`, `test_tier_registry_migration.py`, `test_registry_primitive.py`.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1` (the "freeze primitives, hashing, security primitives" cluster line).

**[CITES KNOW-ADR-010]** (declaration-trust framework), **[CITES KNOW-ADR-010b]** (`@tier_1_error` factory + frozen registry, module-prefix allowlist), **[CITES KNOW-A59]** (Tier-1 trust contract — registered exceptions in this registry are the L0 vocabulary for Tier-1 crashes).

**Confidence:** **High** — both files' headers read in full and the cross-file cycle-breaking pattern verified by reading both ends (`errors.py:21-27` and `tier_registry.py:29-48`).

---

## Entry 8 — Schema contracts (`schema.py`, `schema_contract.py`, `schema_contract_factory.py`, `contract_builder.py`, `contract_propagation.py`, `contract_records.py`, `transform_contract.py`, `type_normalization.py`)

**Path:** Eight top-level files, ~3,500 LOC combined.

**Responsibility:** The Unified Schema Contracts subsystem — `FieldContract` / `SchemaContract` / `PipelineRow` types, factory functions (`create_contract_from_config`, `map_schema_mode`), pipeline-propagation helpers (`propagate_contract`, `narrow_contract_to_output`, `merge_contract_with_output`), transform-output validation (`create_output_contract_from_schema`, `validate_output_against_contract`), audit-trail records (`ContractAuditRecord`, `FieldAuditRecord`, `ValidationErrorWithContract`), and runtime-type normalization (`classify_runtime_type`, `normalize_type_for_contract`).

**File count, LOC:** 8 files; `schema.py` 851, `schema_contract.py` 797, `type_normalization.py` (size deferred — flagged in cluster-scoped layer-check as the file with most R5 isinstance findings, all justified at trust boundary), and 5 smaller siblings.

**Internal coupling:** `schema_contract.py` imports from `contracts.errors` (`:17-24` — `AuditIntegrityError`, `ContractMergeError`, `ContractViolation`, `ExtraFieldViolation`, `MissingFieldViolation`, `TypeMismatchViolation`), `contracts.freeze` (`:25` — `deep_freeze`, `deep_thaw`), and `contracts.type_normalization` (`:26-31`). The schema-contract subsystem is internally cohesive — all 8 files reference `FieldContract` / `SchemaContract` / `PipelineRow` defined in `schema_contract.py`.

**External coupling:** Inbound from `audit_protocols.py:28` (uses `SchemaContract`), from every plugin source/sink/transform (via `plugin_protocols.py:20`), and from `core/canonical.py` and `core/contracts.py` (the L1 canonical normalization pipeline). Outbound zero.

**Patterns observed:**

- **Frozen `FieldContract` with type-locking semantics** (`schema_contract.py:34-50`): `@dataclass(frozen=True, slots=True) class FieldContract` carries `normalized_name`, `original_name`, `python_type`, `required`, `source` (the literal `"declared"` or `"inferred"`). The dual-name pattern (normalized identifier + display original) is the schema-contract subsystem's central invariant.
- **`type_normalization.py` carries all the cluster's R5 findings** — 184 of the 225 cluster-scoped findings are R5 (`isinstance`) and the head of the layer-check file (`temp/layer-check-contracts.txt:7-25`) shows the first dozen R5 findings all on `type_normalization.py:79,83,85,87,...`. Per the CLAUDE.md tier model, **`type_normalization.py` is at a trust boundary** (it normalizes runtime types from external/unknown sources for the audit-record contract surface), so the isinstance checks are exactly where they should be. The `enforce_tier_model.py` allowlist is the place where this is explicitly authorised at the whole-tree level.
- **`PipelineRow` dual-access wrapper**: the wrapper type that lets pipeline code access `row["foo"]` (normalized) or `row.original("Foo Header!")` (display) interchangeably. Per its header (`:5-7`); the dual-name idea is asserted at the `FieldContract` `normalized_name`/`original_name` field pair (`schema_contract.py:49-50` per file body).

**Concerns:**

- **Subsystem internal cohesion is high but the partition is non-obvious from filenames.** `schema.py` (851), `schema_contract.py` (797), `schema_contract_factory.py`, `contract_builder.py`, `contract_propagation.py`, `contract_records.py`, `transform_contract.py`, `type_normalization.py` — the names don't make their layering self-evident. **L3 sub-cluster candidate** (per the L1 Δ4 heuristic: 8 files, 3,500+ LOC, internally cohesive). Not flagged in this pass because no single file >1,500 LOC, but worth noting for any future L3 deep-dive on the schema-contract subsystem itself.

**Test evidence:** `tests/unit/contracts/test_schema_contract.py`, `test_schema_contract_factory.py`, `test_contract_builder.py`, `test_contract_propagation.py`, `test_contract_records.py`, `test_compose_propagation.py`, `test_contract_narrowing.py`, `test_contract_violations.py`, `test_contract_violation_error.py`. Coverage is dense.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1` (sub-area: "schema_contract").

**[CITES KNOW-A55]** (Phase 1 schema validation: upstream `guaranteed_fields` ⊇ downstream `required_fields`); **[CITES KNOW-A56]** (Phase 2 type compatibility across plugin boundaries); **[CITES KNOW-A57]** (validation at DAG construction time, crash on failure); **[CITES KNOW-A65]** (ADR-003 — schema validation lifecycle, two-phase contract → type at DAG construction). **[DIVERGES FROM 02-l1-subsystem-map.md §1]** — the L1 entry summarises this as a single sub-area within the contracts top-level modules listing, but at L2 depth the schema-contract surface is clearly an **internally cohesive 8-file sub-cluster** of ~3,500 LOC, large enough to warrant L3 dive consideration as its own subsystem.

**Confidence:** **Medium** — header inspection plus cross-references; bodies of the 800-LOC files NOT opened. Sub-cluster size flagged.

---

## Entry 9 — Plugin-side L0 surface (`plugin_protocols.py`, `plugin_context.py`, `contexts.py`, `plugin_roles.py`, `plugin_semantics.py`, `plugin_assistance.py`)

**Path:** Six top-level files, ~1,750 LOC combined.

**Responsibility:** The L0-side of the plugin contract — `Source`/`Transform`/`Sink`/`BatchTransform` Protocols (`plugin_protocols.py`), the concrete `PluginContext` execution carrier and its tokens (`plugin_context.py`), the four phase-typed protocol facets `LifecycleContext` / `SinkContext` / `SourceContext` / `TransformContext` (`contexts.py`), plus role / semantics / assistance metadata.

**File count, LOC:** 6 files; `plugin_protocols.py` 753, `plugin_context.py` 550, others ≤ 250 LOC.

**Internal coupling:** `plugin_protocols.py` imports `contracts.enums.Determinism` (`:18`), `contracts.header_modes.HeaderMode` (`:19`), `contracts.schema.SchemaConfig` (`:20`), and **TYPE_CHECKING imports** from `contracts.contexts`, `contracts.data`, `contracts.diversion`, `contracts.results`, `contracts.schema_contract`, `contracts.sink` (`:22-28`). `plugin_context.py` imports `contracts.audit.TokenRef`, `contracts.call_data.RawCallPayload`, `contracts.freeze.deep_freeze` (`:18-20`), plus TYPE_CHECKING imports including the only **CROSS-LAYER TYPE_CHECKING** in the cluster (`plugin_context.py:31` — `from elspeth.core.rate_limit import RateLimitRegistry`).

**External coupling:** Inbound from every plugin (sources, sinks, transforms) — every plugin satisfies `SourceProtocol`/`TransformProtocol`/`SinkProtocol`/`BatchTransformProtocol`. Inbound from `engine/executors/` for `isinstance(node, TransformProtocol)` runtime discrimination. Outbound zero at runtime.

**Patterns observed:**

- **`@runtime_checkable` is the exception, not the norm** (`plugin_protocols.py:1-7`): "These protocols define what methods plugins must implement. They're primarily used for type checking ... with one exception: `TransformProtocol` is `@runtime_checkable` because the engine uses `isinstance()` to discriminate transforms from gates and coalesce nodes during DAG traversal." This is the rationale for an isinstance use-case the CI scanner accepts — the engine's DAG executor discriminates node types at runtime.
- **`PluginConfigProtocol` defined in L0 to avoid contracts → plugins/infrastructure dependency** (`plugin_protocols.py:31-49`): "Defined here in L0/contracts to avoid a structural dependency from contracts → plugins/infrastructure/config_base." This is the post-ADR-006 boundary preserved by inverting an apparent dependency — the plugin config base would naturally live in plugins/infrastructure (as it does for the runtime implementation), but the *protocol* for it sits in L0 so contracts can stay leaf.
- **Phase-typed contexts** (`contexts.py` — re-exported via `__init__.py:106-111`): `LifecycleContext`, `SinkContext`, `SourceContext`, `TransformContext`. The concrete `PluginContext` (in `plugin_context.py:36+`) structurally satisfies all four phase protocols ([CITES KNOW-A41]). Engine executors mutate it between pipeline steps; plugins see narrowed read-only views via protocol typing ([CITES KNOW-A40]).
- **One `core/` TYPE_CHECKING import** (`plugin_context.py:31`): `from elspeth.core.rate_limit import RateLimitRegistry`. This is the **cluster's only cross-layer TYPE_CHECKING import**. Per `__init__.py:8-10` ("TYPE_CHECKING imports from core exist for type annotations only and do not create runtime coupling") this is permitted and confirmed clean by the whole-tree layer-check.

**Concerns:**

- **TYPE_CHECKING import to `core/`** (`plugin_context.py:31`) — a soft architectural smell. The annotation is for the `RateLimitRegistry` type used by the `PluginContext`'s rate-limit accessor. If a future change needs the TYPE_CHECKING block to become a runtime import, the L0 leaf invariant breaks. Per ADR-006d's "Violation #11 Protocol" the resolution would be: (1) move the code down (extract the registry's protocol to L0), (2) extract the primitive (a `RateLimitRegistryProtocol` in `contracts.config.protocols`), or (3) restructure the caller (have the engine pass concrete instances rather than the registry). **L2 debt candidate / Q1 evidence** — this is one of the surfaces that "smells like core but lives in contracts," in the sense that the registry's interface is L0-shaped (and could be), but the registry implementation itself is L1.

**Test evidence:** `tests/unit/contracts/test_plugin_protocols.py`, `test_plugin_context_recording.py`, `test_context_protocols.py`.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1` (sub-areas: "plugin_context", "plugin_protocols").

**[CITES KNOW-A40]** (phase-typed protocols), **[CITES KNOW-A41]** (concrete `PluginContext` satisfies all 4); **[CITES KNOW-ADR-006d]** ("move down → extract primitive → restructure caller" as the TYPE_CHECKING-creep mitigation).

**Confidence:** **High** — both major files' headers read in full; the cross-layer TYPE_CHECKING import verified verbatim.

---

## Entry 10 — `contracts/errors.py` — error / reason / DTO surface (L3 deep-dive candidate)

**Path:** `src/elspeth/contracts/errors.py`

**Responsibility:** Error-hierarchy declarations, structured-reason TypedDicts, frozen audit DTOs, and Tier-1 violation classes. The largest file in the cluster.

**File count, LOC:** 1 file, **1,566 LOC** — **L3 deep-dive candidate** per Δ L2-3.

**Internal coupling (header `:1-40` only):** `contracts.audit_evidence.AuditEvidenceBase` (`:12`), `contracts.declaration_contracts.DeclarationContractViolation` (`:13`), `contracts.freeze.deep_freeze` + `freeze_fields` (`:14`), `contracts.tier_registry.FrameworkBugError` + `tier_1_error` (`:21-22`), TYPE_CHECKING `contracts.batch_checkpoint.BatchCheckpointState` and `contracts.coalesce_metadata.CoalesceMetadata` (`:30-31`).

**External coupling:** Inbound is broad — every L1+ file that constructs a violation, raises a Tier-1 exception, or records a structured error-payload imports from here. Outbound zero.

**Patterns observed (header):**

- **Forward-declared `FrameworkBugError` re-export pattern** (`errors.py:16-27`): the class object lives in `tier_registry.py` (to break the import cycle); `errors.py` imports it as `_FrameworkBugError`, applies `@tier_1_error(reason="ADR-008: framework internal inconsistency — engine bug", caller_module=__name__)`, then exports the decorated result as `FrameworkBugError`. **Identity preservation** is explicit: "The re-exported name is identical to the class object in tier_registry — `isinstance`/`except` identity is preserved." This is **the canonical pattern** for cycle-breaking-while-preserving-identity in the cluster.
- **Tier-1 vs Tier-2 distinction encoded in dataclass kind** (`errors.py:34-40`): `# TIER-2: Frozen audit DTO (not a raiseable exception) — records structured error payloads to the Landscape audit trail.` followed by `@dataclass(frozen=True, slots=True) class ExecutionError`. The same file holds raiseable Tier-1 exceptions and frozen Tier-2 DTOs; the comments distinguish.

**Concerns:**

- **Body not opened (Δ L2-3 honoured).** This file is the highest-priority L3 deep-dive candidate in the cluster — at 1,566 LOC, it is dense with violation classes, reason TypedDicts, and frozen audit DTOs. Two specific questions for L3:
  - **Tier-1/Tier-2 distinction discipline.** If both are in one file, is the comment-marker discipline mechanical (a CI check) or convention? At header-level inspection it appears conventional.
  - **`@tier_1_error` decoration sites.** How many exceptions in this file are decorated? The decorator's `caller_module` parameter (`:26`) constrains who can apply it — only `elspeth.contracts.*` callers can decorate Tier-1 here, by `tier_registry.py:8-11`'s allowlist.
- **No dedicated `test_errors.py`** — coverage exists indirectly through `test_contract_violation_error.py`, `test_contract_violations.py`, `test_control_flow_exceptions.py` and others. **L2 debt candidate** — a dedicated invariant pin (e.g., asserting the Tier-1 decoration-site set is closed) would lock the file's contract.

**Test evidence:** indirect — through the violation-classes and control-flow test suites.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1`'s "Highest-risk concern" line, which **named `errors.py` as the L2-deep-dive candidate** for this cluster (verbatim: "L2-deep-dive candidate inside this subsystem: `contracts/errors.py` (1,566 LOC per discovery findings) — flag for later, do not open"). The L1 catalog's depth-cap is honoured by this entry.

**[CITES KNOW-ADR-010b]** (`@tier_1_error` decorator surface — Tier-1 elevation only permitted to `elspeth.contracts.*`, `engine.*`, `core.*`); **[CITES KNOW-A59]** (Tier-1 Audit DB full trust contract); **[CITES KNOW-A60]** (Tier-2 Pipeline elevated trust — the frozen audit-DTO surface this file holds for the Landscape audit trail); **[DIVERGES FROM]** none — the L1 entry's depth-cap framing is fully ratified.

**Confidence:** **High** for the entry's framing (L1 already named this file); **Low** for any inferences about the body. L3 deep-dive deferred.

---

## Entry 11 — Identity, lineage, secret scrub (`identity.py`, `token_usage.py`, `secret_scrub.py`)

**Path:** `src/elspeth/contracts/identity.py`, `token_usage.py`, `secret_scrub.py`

**Responsibility:** `identity.py` defines `TokenInfo` — the row/token/parent identity carrier (`row_id`, `token_id`, `parent_token_id`) that survives forks and joins ([CITES KNOW-A46]). `token_usage.py` defines `TokenUsage` — the per-call usage record (LLM token counts, cost, latency). `secret_scrub.py` is the last-line-of-defence redaction for `DeclarationContractViolation` payloads (ADR-010 §Decision 3).

**File count, LOC:** 3 files; `secret_scrub.py` ~80 LOC (header read `:1-25`), the others smaller.

**Internal coupling:** `secret_scrub.py:21` imports from `contracts.url.SENSITIVE_PARAMS`. `identity.py` and `token_usage.py` are pure dataclass declarations with stdlib-only imports.

**External coupling:** Inbound from engine (token-identity), Landscape (audit DTOs), every LLM-using transform (token usage), and the declaration-contracts dispatcher (secret scrub). Outbound zero.

**Patterns observed:**

- **`secret_scrub.py:1-12` framing:** "The Landscape audit trail is a legal record. Arbitrary `Mapping[str, Any]` payloads (allowed by the `DeclarationContractViolation` signature) could carry API keys, connection strings, or OAuth tokens from plugin `config.options`. This helper redacts values matching known secret patterns BEFORE the payload is handed to `to_audit_dict`." The header is explicit that this is **last-line-of-defence**: "Coverage is best-effort: new secret formats need new patterns here. This is the last line of defence, not the first — contract authors SHOULD structure payloads so they never carry secrets (see per-contract `TypedDict` `payload_schema`)."
- **`_REDACTED = "<redacted-secret>"` constant** (`secret_scrub.py:23`). Single-source.
- **Heuristic patterns ordered longest-first** (`secret_scrub.py:25` "Order matters — longer / more specific first"). Body deferred per Δ L2-3 (file <1,500 LOC threshold so this is conventional, but the header-only treatment matches the cluster's discipline).

**Concerns:**

- **`secret_scrub` is best-effort by admission.** This is appropriate — defence-in-depth — but it means the security guarantee depends on contract authors structuring payloads correctly (per-contract `payload_schema` H5). Two layers of defence: (1) `payload_schema` deny-by-default (H5, KNOW-ADR-010h), (2) `secret_scrub` redaction. Layer 1 is the primary; Layer 2 is the safety net. **No L2 concern** — both layers are present and tested.

**Test evidence:** `tests/unit/contracts/test_secret_scrub.py`. No dedicated test file for `identity.py` — invariants are exercised indirectly through token-flow integration tests.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1` (sub-areas: identity, security primitives).

**[CITES KNOW-A46]** (token identity invariants), **[CITES KNOW-ADR-010]** (declaration-trust framework), **[CITES KNOW-ADR-010h]** (per-contract `payload_schema`).

**Confidence:** **High** for `secret_scrub.py` (header read); **Medium** for `identity.py` and `token_usage.py` (file size and role inferred from `__init__.py` re-exports).

---

## Entry 12 — Checkpoint family (`checkpoint.py`, `batch_checkpoint.py`, `coalesce_checkpoint.py`, `aggregation_checkpoint.py`)

**Path:** Four top-level files, ~700 LOC combined.

**Responsibility:** Tier-1 audit DTOs for resume — `ResumeCheck`, `ResumePoint`, `BatchCheckpointState`, `CoalesceCheckpointState`, `CoalescePendingCheckpoint`, `CoalesceTokenCheckpoint`, `AggregationCheckpointState`, `AggregationNodeCheckpoint`, `AggregationTokenCheckpoint`, `RowMappingEntry`. These are the persistent state-snapshots that the engine writes to checkpoint storage and reads back at resume.

**File count, LOC:** 4 files. Per L1's discovery findings the largest is `aggregation_checkpoint.py` (296 LOC); others are smaller. Header inspection only.

**Internal coupling:** Each file uses `contracts.freeze.freeze_fields` for the `__post_init__` immutability guards. Cross-file: `batch_checkpoint.py` and `coalesce_checkpoint.py` reference token / row identity types from `identity.py` and `audit.py`.

**External coupling:** Inbound from `core/checkpoint/` (the L1 repository tier writes/reads these DTOs) and from `engine/orchestrator/` (the bootstrap/resume path constructs them). Outbound zero.

**Patterns observed:**

- **Frozen + freeze_fields throughout.** All four files appear in the freeze inventory (verified: `aggregation_checkpoint.py`, `batch_checkpoint.py`, `coalesce_checkpoint.py` all use `freeze_fields(self, ...)` per the `grep` census). The fourth file (`checkpoint.py`) holds the smaller `ResumeCheck`/`ResumePoint` types.
- **TIER-1 framing** (per CLAUDE.md "Our Data" trust tier — these are persisted as audit-side state-snapshots; a bug here corrupts the resume path).

**Concerns:**

- None observed at L2 depth. Coverage is dense (`tests/unit/contracts/test_batch_checkpoint.py`, `test_checkpoint.py`, `test_checkpoint_post_init.py`, `test_coalesce_checkpoint.py`).

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1` (sub-area: "checkpoint").

**[CITES KNOW-A20]** (Checkpoint subsystem ~600 LOC, crash recovery with topology validation — these L0 DTOs are the persisted shape).

**Confidence:** **High** at the framing level (file roles + freeze pattern verified by grep); **Medium** for individual class invariants (bodies not opened, but L1 already cited the surface).

---

## Entry 13 — `contracts/pipeline_runner.py` — runner protocol

**Path:** `src/elspeth/contracts/pipeline_runner.py`

**Responsibility:** Define the `PipelineRunner` Protocol — a callable type signature `(settings_path: Path) -> RunResult` that lets L2 (engine) accept a pipeline-execution callback from L3 (application layer) without creating an upward import dependency.

**File count, LOC:** 1 file, ~25 LOC (read in full).

**Internal coupling:** `contracts.run_result.RunResult` (`:12`), stdlib `pathlib.Path` (`:9`), `typing.Protocol` (`:10`).

**External coupling:** Inbound from `engine/bootstrap.py` and `engine/dependency_resolver.py` (per the engine cluster's "Cross-cluster observations for synthesis" entry §"engine ↔ contracts.pipeline_runner protocol": "The runner protocol is the contract that lets `bootstrap.resolve_preflight()` be the shared CLI/programmatic entry"). Inbound from `cli.py` (which provides the L3 implementation that conforms to the protocol). Outbound zero.

**Patterns observed:**

- **Inverted-dependency Protocol idiom** (`pipeline_runner.py:1-5`): "Used by the dependency resolver (L2) to accept a pipeline execution callback from the application layer (L3) without creating an upward import dependency." This is the **canonical** L2-accepts-L3-via-L0-protocol pattern. The Protocol's body is just `def __call__(self, settings_path: Path) -> RunResult: ...` (line 22) — minimal, single-method, structural.

**Concerns:**

- None observed. The file is trivially small, single-purpose, and citably documented.

**Test evidence:** No dedicated test file. The protocol is exercised through integration and CLI test paths. **L2 debt candidate (mild)** — a 5-line test pinning the `__call__` signature would lock the protocol; absence is conventional for single-method Protocols but not zero-cost.

**L1 cross-reference:** Engine cluster's "Cross-cluster observations for synthesis" §"engine ↔ contracts.pipeline_runner protocol" is **closed by this entry**.

**[CITES KNOW-A53]** (zero-outbound leaf — this is one of the patterns that makes the leaf invariant practicable).

**Confidence:** **High** — read in full.

---

## Entry 14 — Misc top-level types (~30 files)

**Path:** Approximately 30 top-level files, each ≤ 632 LOC: `data.py`, `types.py`, `enums.py`, `results.py`, `run_result.py`, `events.py`, `node_state_context.py`, `header_modes.py`, `diversion.py`, `routing.py`, `probes.py`, `coalesce_enums.py`, `coalesce_metadata.py`, `call_data.py`, `engine.py`, `sink.py`, `cli.py`, `payload_store.py`, `database_url.py`, `url.py`, `runtime_val_manifest.py`, `export_records.py`, `secrets.py`, `security.py`, **`guarantee_propagation.py`** (the propagation helper used by `contracts/contract_propagation.py`'s narrowing/merging functions — kept here because it's a single-purpose pure-function utility), **`reorder_primitives.py`** (small set of immutable reordering helpers used by the schema-contract subsystem), plus a handful of smaller siblings.

**Responsibility:** The cluster's "small types and protocols" tail — frozen audit DTOs, error reasons (TypedDicts), enum families, sanitised URL types, payload-store contracts, CLI types (`ExecutionResult`, `ProgressEvent`), call-data records (`HTTPCall*`, `LLMCall*`), event types (`PhaseAction`, `RunStarted`, `RowCompleted`, etc.), and sink/source-side types.

**File count, LOC:** ~30 files; largest is `results.py` (632), then `events.py` (435), `url.py` (396), `data.py` (385). Most are 100–300 LOC.

**Internal coupling:** Most files re-import `freeze_fields` from `contracts.freeze` and enum types from `contracts.enums`. `routing.py`, `diversion.py`, `events.py`, and `run_result.py` are also part of the freeze inventory (each uses `freeze_fields(self, ...)`).

**External coupling:** Inbound from across L1+. Outbound zero.

**Patterns observed:**

- **Frozen-dataclass uniform pattern.** 18 of the cluster's files use `freeze_fields(self, ...)` per the grep census (Entry 3). The misc-types tail is roughly half of those (`call_data.py`, `coalesce_metadata.py`, `diversion.py`, `events.py`, `plugin_assistance.py`, `plugin_semantics.py`, `results.py`, `routing.py`, `run_result.py`, `sink.py`, plus the checkpoint family already enumerated under Entry 12).
- **`diversion.py:62-69` offensive validation.** Constructor checks `isinstance(self.artifact, ArtifactDescriptor)` and `isinstance(self.diversions, tuple)` and raises `PluginContractViolation` on failure. This is the **offensive-programming** pattern (assert invalid states crash with maximally-informative errors) and is the **correct** counterpart to the freeze guards. **Not** the forbidden `isinstance(self.x, tuple) to skip` anti-pattern from CLAUDE.md — this guard rejects the wrong type, it does not skip freezing.
- **`url.py` — `SanitizedDatabaseUrl` and `SanitizedWebhookUrl` types** plus `SENSITIVE_PARAMS` constant. The constant is consumed by `secret_scrub.py:21` (Entry 11); separation here makes the URL-side and the violation-payload-side independently testable.

**Concerns:**

- **Header-only inspection per Δ L2-3.** Bodies are not opened. The 30-file tail is conventionally clean (frozen-dataclass + freeze_fields), but invariant verification is deferred. **No L2 debt** identified beyond what is already noted in Entries 1–13.
- **Cluster total of 33 `freeze_fields` invocations** is the freeze surface area for L0. CI gate `enforce_freeze_guards.py` (KNOW-C65) covers all of them. Verified by whole-tree layer-check (clean).

**Test evidence:** Per-file unit tests exist for the substantive entries (`test_call_data.py`, `test_coalesce_metadata.py`, `test_routing.py` — actually `test_data.py`, `test_compose_propagation.py` etc.; full per-file coverage was confirmed by `ls tests/unit/contracts/` showing 87 files). For the smallest types (e.g., `enums.py`) coverage is via downstream usage in violation tests.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §1`'s "Top-level contracts modules ..." line; this entry is the catch-all for everything not covered by Entries 1–13.

**[CITES KNOW-C62]** (canonical `freeze_fields` pattern), **[CITES KNOW-C63]** (recursive `deep_freeze` semantics that this entry's 18-file usage census exercises).

**Confidence:** **Medium** — file roles inferred from `__init__.py` re-exports plus header skims of a handful (`diversion.py`, `url.py`); per-file body-level invariants not opened. Sufficient for L2 framing.

---

## Closing — test-debt candidates surfaced (per Δ L2-5)

Three concrete test-debt candidates flagged across the catalog:

1. **`tests/unit/contracts/test_public_surface.py` does not exist** (Entry 1). The 208-name `__all__` list has no dedicated stability test. **Risk:** an accidental Settings re-export or an inadvertently dropped name surfaces only as a downstream import failure.
2. **`tests/unit/contracts/test_tier_registry_pytest_gating.py` does not exist** (Entry 7). The pytest-allowance widening of the module-prefix allowlist is a deliberate test seam, but the discriminator's robustness (e.g., is it a one-shot import-time check?) has no pinning test. **Risk:** a future change to pytest-detection logic could permit production-time elevation.
3. **`tests/unit/contracts/test_errors.py` does not exist** (Entry 10). `errors.py` (1,566 LOC) is the cluster's only L3 deep-dive candidate; coverage is indirect through violation-class tests. **Risk:** the Tier-1 decoration-site set is not pinned as a closed contract; a stray decoration in a non-`errors.py` site is currently caught only by the `caller_module` allowlist mechanism, not by an explicit test.

Item 3 (test_errors absence) maps directly to Report Q2 ("Should `errors.py` (1,566 LOC) be split, and along which seam?") — the test-debt would lock the file's contract before any split is attempted. Items 1 and 2 are independent of the report's three uncertainty questions but thematically support Q1's "post-ADR-006 boundary" framing: test_public_surface would pin the closed-set `__all__` against accidental Settings re-export; test_tier_registry_pytest_gating would pin the registration-allowlist seam.

## Closing — concrete observations for synthesis

Three observations that the post-L2 synthesis pass may want to surface verbatim:

A. **`config/runtime.py:338` — single non-uniform freeze pattern.** `MappingProxyType(dict(self.services))` is the cluster's only exception to the `freeze_fields(self, ...)` pattern. Acceptable per CLAUDE.md (values are frozen-dataclass `RuntimeServiceRateLimit`); recommend unifying via `freeze_fields(self, "services")` for pattern uniformity.

B. **`plugin_context.py:31` — single TYPE_CHECKING cross-layer import.** `from elspeth.core.rate_limit import RateLimitRegistry` is the only TYPE_CHECKING import to L1+ in the cluster. Permitted by `__init__.py:8-10`; flagged as a Q1-evidence surface (an L0-shaped interface is annotated against an L1 implementation).

C. **`type_normalization.py` carries 184 of 225 cluster-scoped R5 isinstance findings — all justified.** The file normalizes runtime types from external/unknown sources for the audit-record contract surface; isinstance is the correct primitive at this trust boundary; the whole-tree layer-check accepts these via the standing allowlist.
