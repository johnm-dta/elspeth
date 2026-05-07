# H2 ADR-010 amendment — Phase 1 design sketch

**Date:** 2026-04-20
**Author:** claude-opus-4-7 (H2 cluster implementation)
**Cluster tickets:** H2 (`elspeth-425047a599`), N1 (`elspeth-10dc0b747f`),
N3 (`elspeth-60890a7388`), F2 (`elspeth-f52d7c5a47`), F3 (`elspeth-5fc876138d`),
F4 (`elspeth-b513c01cff`), F5 (`elspeth-121b268aec`), H1-amend
(`elspeth-5dae105959`).
**Decision anchor:** comment #417 on `elspeth-425047a599` — ADR-010 §Semantics
audit-complete posture.
**Scope:** single-PR landing per H2 §Acceptance "ADR amendment landing manifest
(H2-B)". W9 escape hatch declined (see §8 rationale).

---

## 1. Catch-site survey (plan-review B2 — delivered up-front)

Required before the PR is marked ready. Running the surveys now surfaces any
design gap before code is written.

### 1.1 `except DeclarationContractViolation` sites under `src/elspeth/`

| File | Line | Purpose | Handles aggregate? | Code change required |
|------|------|---------|--------------------|----------------------|
| `src/elspeth/engine/executors/declaration_dispatch.py` | 38 | Dispatcher attaches `contract_name` to the caught violation and re-raises | **N/A** — this is the site being restructured. The new dispatcher catches broadly, aggregates, and raises the single-or-aggregate form. | YES — complete rewrite into collect-then-raise. |

**Survey conclusion:** only the dispatcher itself catches
`DeclarationContractViolation` in production code. No downstream executor,
recorder, orchestrator, sink, or plugin code relies on the exception type
to trigger routing, audit recording, or recovery. The dispatcher is the
single authoritative catch point; restructuring it does not require
coordinated changes elsewhere.

### 1.2 `except (...DeclarationContractViolation...)` sites under `src/elspeth/`

Zero matches. No tuple-catch forms exist.

### 1.3 `except PassThroughContractViolation` — related, worth documenting

| File | Line | Purpose |
|------|------|---------|
| `src/elspeth/engine/processor.py` | 785–787 | `_cross_check_flush_output` catches `PassThroughContractViolation` after `run_runtime_checks` returns, calls `_record_flush_violation` for per-token FAILED audit entries, re-raises. |

Analysis: this catch is **outside** the dispatcher — it catches what the
dispatcher re-raises. Under audit-complete, if PassThrough + another contract
both fire on a batch-flush TRANSFORM row, the dispatcher now raises an
`AggregateDeclarationContractViolation` (not a `PassThroughContractViolation`
directly). The processor's catch must therefore ALSO handle the aggregate case:

- Aggregate containing only PassThroughContractViolation child(ren): route
  through `_record_flush_violation` for each child so audit entries match
  existing SQL shape.
- Aggregate containing mixed child types: `_record_flush_violation` for the
  PassThrough child(ren); the DCV child(ren) flow through the normal
  `NodeStateGuard.__exit__` audit path (the aggregate's `to_audit_dict()`
  writes the full violations list to `ExecutionError.context`).

This is a **real code change** required in `_cross_check_flush_output`. It is
NOT a test-only concern. Documented in §5.6.

### 1.4 Test-level catches of `DeclarationContractViolation`

Test files reference the class for `pytest.raises`. Each test's assertion needs
review for the aggregate-vs-single case:

- `tests/unit/engine/test_declaration_dispatch.py:199` — single-violation case;
  unchanged under reference-equality N=1 path.
- `tests/invariants/test_contract_negative_examples_fire.py:40,92` — iterates
  registered contracts; `runtime_check` is invoked directly (bypassing the
  dispatcher), so aggregate never arises here. Unchanged at contract-body
  level, but the harness must update its method-invocation pattern per §5.3.
- `tests/invariants/test_framework_accepts_second_contract.py:298` — asserts
  `CreatesTokensViolation` is a `DeclarationContractViolation` subclass.
  Unchanged.
- `tests/invariants/test_contract_non_fire.py:72` — asserts no DCV raised.
  Unchanged.
- `tests/integration/audit/test_declaration_contract_landscape_roundtrip.py` —
  exercises single-violation round-trip. Add aggregate round-trip test
  under the F5 / N3 acceptance bullets.
- `tests/unit/contracts/test_declaration_contracts.py` — various plumbing
  tests; each reviewed and updated where they assert the exception shape.

**No test file catches `except DeclarationContractViolation` in a form that
would fail-closed under aggregation.** All tests that `pytest.raises(DCV)` on
a single-violation scenario continue to pass because the N=1 path raises the
original violation unchanged.

---

## 2. Layer placement and the decorator (H2-A, plan-review W4)

### 2.1 `@implements_dispatch_site` decorator

**Location:** `src/elspeth/contracts/declaration_contracts.py` (layer L0).

Defining the decorator anywhere above L0 creates an upward import from
contract implementations (L2 `engine/executors/` and L3 `plugins/`) into the
decorator's defining module, which `scripts/cicd/enforce_tier_model.py`
would reject. L0 is the only viable placement for a marker the scanner at
L3 imports and concrete contracts at L2/L3 apply.

### 2.2 Signature

```python
DispatchSiteName: TypeAlias = Literal[
    "pre_emission_check",
    "post_emission_check",
    "batch_flush_check",
    "boundary_check",
]


class DispatchSite(StrEnum):
    PRE_EMISSION = "pre_emission_check"
    POST_EMISSION = "post_emission_check"
    BATCH_FLUSH = "batch_flush_check"
    BOUNDARY = "boundary_check"


F = TypeVar("F", bound=Callable[..., Any])


def implements_dispatch_site(site_name: DispatchSiteName) -> Callable[[F], F]:
    """Mark a method as implementing a named dispatch site.

    Two purposes:
    1. Runtime (L1 dispatcher): ``register_declaration_contract`` inspects
       class methods for this marker to compute the contract's per-site
       registration map ``{site → list[contract]}``. Methods without the
       marker are NOT invoked by the dispatcher for any site, even if they
       happen to share a name with a DispatchSite member.
    2. Static (L3 tooling): ``scripts/cicd/enforce_contract_manifest.py``
       MC3a/b/c rules AST-detect the decorator on concrete contract classes,
       required for multi-level-inheritance detection (per the D1 correction
       in comment #418 on H2 — ``subclass.__dict__`` alone does not see
       methods inherited from mixin bases).

    Validates ``site_name`` at decoration time against DispatchSite.
    Non-matching values raise ``ValueError`` immediately — the decorator
    cannot be applied with a typo, and the offence surfaces at module
    import rather than first dispatcher call.
    """
```

The decorator sets `method._declaration_dispatch_site = site_name` on the
function object. At `register_declaration_contract` time, the registry
walks `type(contract).__mro__` (skipping `object` and the base ABC) and
finds every method with the marker. This gives an authoritative
`frozenset[DispatchSiteName]` per contract, stored alongside the
contract in the registry.

### 2.3 Base ABC

```python
class DeclarationContract(ABC):
    """Nominal ABC every declaration-trust contract inherits.

    Four dispatch methods carry default no-op bodies (``pass``). Concrete
    contracts override the methods they implement AND decorate each override
    with ``@implements_dispatch_site("<site>")``. The decorator is the
    authoritative signal; the default-body override is the secondary signal
    for flat (non-mixin) contracts.

    The ABC is NOT runtime-checkable (Protocol). The ADR-010 §Alternative 3
    rejection of structural Protocol matching stands: nominal inheritance
    closes the STRIDE Spoofing vector where any class exposing a coincidental
    method signature could claim to be a declaration contract.
    """

    name: ClassVar[str]
    payload_schema: ClassVar[type]

    @abstractmethod
    def applies_to(self, plugin: Any) -> bool: ...

    @classmethod
    @abstractmethod
    def negative_example(cls) -> "ExampleBundle": ...

    @classmethod
    @abstractmethod
    def positive_example_does_not_apply(cls) -> "ExampleBundle": ...

    # Dispatch methods. Default no-op bodies. Concrete contracts override
    # and decorate with @implements_dispatch_site. MC3c CI rule forbids
    # trivial override bodies (pass / ... / bare return / literal).
    def pre_emission_check(self, inputs: "PreEmissionInputs") -> None:
        return None

    def post_emission_check(
        self,
        inputs: "PostEmissionInputs",
        outputs: "PostEmissionOutputs",
    ) -> None:
        return None

    def batch_flush_check(
        self,
        inputs: "BatchFlushInputs",
        outputs: "BatchFlushOutputs",
    ) -> None:
        return None

    def boundary_check(
        self,
        inputs: "BoundaryInputs",
        outputs: "BoundaryOutputs",
    ) -> None:
        return None
```

### 2.4 Example bundle (site-tagged harness example)

```python
@dataclass(frozen=True, slots=True)
class ExampleBundle:
    """Site-tagged bundle returned by ``negative_example`` /
    ``positive_example_does_not_apply``.

    Under the 4-method ABC the harness cannot know in advance which dispatch
    method to invoke. The tagged bundle lets the harness dispatch per site:

        bundle = type(contract).negative_example()
        method = getattr(contract, bundle.site.value)
        method(*bundle.args)

    The ``args`` tuple shape depends on the site:
    - PRE_EMISSION: ``(PreEmissionInputs,)``
    - POST_EMISSION: ``(PostEmissionInputs, PostEmissionOutputs)``
    - BATCH_FLUSH: ``(BatchFlushInputs, BatchFlushOutputs)``
    - BOUNDARY: ``(BoundaryInputs, BoundaryOutputs)``

    A contract implementing multiple sites returns the bundle for whichever
    site its negative example is most representative of. Contracts with
    negative examples for multiple sites subclass the bundle with
    per-site methods (not needed at 2A).
    """

    site: DispatchSite
    args: tuple[Any, ...]
```

---

## 3. Bundle types (field-by-field, H2 §Fix direction)

### 3.1 PreEmissionInputs

For F2's pre-emission dispatch site. `DeclaredRequiredFieldsContract` and
future pre-emission adopters.

```python
@dataclass(frozen=True, slots=True)
class PreEmissionInputs:
    """Bundle passed to pre-emission contracts (runs BEFORE transform.process()).

    No ``emitted_rows`` — emission hasn't happened yet. ``input_row`` is the
    to-be-processed row; ``effective_input_fields`` is the normalised field
    set derived by the caller (``TransformExecutor``) from
    ``input_row.contract.fields``. Contracts MUST use
    ``effective_input_fields`` and NOT derive from ``input_row.contract``
    themselves — the caller-derivation pattern avoids the B-antipattern
    where each contract re-implements the field-set derivation and they
    drift.
    """

    plugin: Any
    node_id: str
    run_id: str
    row_id: str
    token_id: str
    input_row: Any
    static_contract: frozenset[str]
    effective_input_fields: frozenset[str]
```

No `__post_init__` required — every field is a scalar, enum-like, or frozen
collection (CLAUDE.md §Frozen Dataclass Immutability: scalar-only fields
need no guard). `frozenset` is intrinsically immutable.

### 3.2 PostEmissionInputs

Replaces the existing `RuntimeCheckInputs`. Panel F1 (`override_input_fields`
as B-antipattern-in-miniature) fixed by replacing the nullable override with
a caller-computed `effective_input_fields`.

```python
@dataclass(frozen=True, slots=True)
class PostEmissionInputs:
    """Bundle passed to post-emission contracts (runs AFTER transform.process()).

    Panel F1 resolution: the caller (``TransformExecutor`` /
    ``RowProcessor._cross_check_flush_output`` PASSTHROUGH branch) derives
    ``effective_input_fields`` once from ``input_row.contract.fields`` and
    passes it in the bundle. Contracts reading field semantics MUST use
    ``effective_input_fields`` and MUST NOT derive from ``input_row.contract``
    themselves. The 2A-era ``override_input_fields: frozenset | None`` is
    deleted entirely — no sentinel, no branching.
    """

    plugin: Any
    node_id: str
    run_id: str
    row_id: str
    token_id: str
    input_row: Any
    static_contract: frozenset[str]
    effective_input_fields: frozenset[str]
```

### 3.3 PostEmissionOutputs

Replaces the existing `RuntimeCheckOutputs`. Same shape.

```python
@dataclass(frozen=True, slots=True)
class PostEmissionOutputs:
    """Emitted-rows bundle for post-emission dispatch.

    ``emitted_rows`` is normalised to a deep-frozen ``tuple`` in
    ``__post_init__`` — preserves the existing ``RuntimeCheckOutputs``
    contract which crashes on non-list/non-tuple inputs per CLAUDE.md
    §Offensive Programming.
    """

    emitted_rows: tuple[Any, ...]

    def __post_init__(self) -> None:  # identical to today's RuntimeCheckOutputs
        ...
```

### 3.4 BatchFlushInputs / BatchFlushOutputs

For `_cross_check_flush_output` TRANSFORM mode. The batch-homogeneous
intersection of every buffered token's contract, carried as
`effective_input_fields`.

```python
@dataclass(frozen=True, slots=True)
class BatchFlushInputs:
    """Bundle passed to batch-flush contracts (TRANSFORM mode, ADR-009 §Clause 2).

    Unlike post-emission's single ``input_row``, batch-flush has a tuple of
    buffered tokens. The identity fields (``row_id``, ``token_id``) anchor
    the violation to the triggering token (or first buffered token on
    timeout flushes — ``_cross_check_flush_output`` already computes this
    and passes its choice in).

    ``effective_input_fields`` is the INTERSECTION across every buffered
    token's contract — the weakest shared guarantee that every emitted
    row must preserve. Dispatcher caller computes this once.
    """

    plugin: Any
    node_id: str
    run_id: str
    row_id: str  # identity anchor; first-buffered if triggering is None
    token_id: str  # identity anchor
    buffered_tokens: tuple[Any, ...]  # for per-token audit recording
    static_contract: frozenset[str]
    effective_input_fields: frozenset[str]


@dataclass(frozen=True, slots=True)
class BatchFlushOutputs:
    emitted_rows: tuple[Any, ...]
    # Identical __post_init__ to PostEmissionOutputs — same offensive guards.
```

### 3.5 BoundaryInputs / BoundaryOutputs

For 2C adopters (source-side and sink-side). Per §10 this cluster's landing
does NOT land 2C adopters (they are paired per F4's rule), but the dispatch
site + bundle types must exist so the N1 manifest can record coverage.

```python
@dataclass(frozen=True, slots=True)
class BoundaryInputs:
    """Bundle passed to boundary contracts (source emission / sink consumption).

    Carries a single ``rows`` tuple whose meaning is context-dependent:
    - Source-side adopter: ``rows`` = rows the source produced (plural).
    - Sink-side adopter: ``rows`` = rows the sink consumed (plural).

    The contract's ``applies_to`` discriminates source-side vs sink-side
    based on the plugin's concrete class; each adopter's ``runtime_check``
    operates on ``rows`` with its own semantics. No ``input_row`` singular
    — sources have no input row, sinks have plural inputs.

    2C adopters may propose splitting this into ``SourceBoundaryInputs`` /
    ``SinkBoundaryInputs`` at their landing PR. The present bundle is
    sufficient for the N1 manifest + dispatcher wiring; 2C will validate or
    refine.
    """

    plugin: Any
    node_id: str
    run_id: str
    static_contract: frozenset[str]
    rows: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class BoundaryOutputs:
    """Outputs bundle for boundary dispatch.

    For most boundary contracts this is vestigial — sources' emitted rows
    ARE the ``BoundaryInputs.rows``, and sinks produce no rows. The bundle
    exists so the dispatcher signature
    ``boundary_check(inputs, outputs)`` matches the others structurally.
    2C will refine if needed.
    """

    rows: tuple[Any, ...] = ()
```

---

## 4. Dispatcher — collect-then-raise (N3 primary mitigation)

### 4.1 Shared helper

```python
def _dispatch(
    *,
    site: DispatchSite,
    plugin: Any,
    invoke: Callable[[DeclarationContract], None],
) -> None:
    """Collect-then-raise orchestration for any dispatch site.

    The SINGLE shared helper for all 4 public dispatch functions (F2's
    "single shared helper with the post-emission dispatcher, not parallel
    implementation" constraint). Each public function is a thin site-tagged
    wrapper.

    Audit-complete (ADR-010 §Semantics per comment #417): every applicable
    contract's method runs. A contract raising an audit-evidence-bearing
    exception does NOT short-circuit iteration. Violations are collected;
    at loop end the aggregation rule fires:
      - 0 violations: return normally.
      - 1 violation: raise violations[0] via reference equality
                     (N6 regression test asserts this).
      - >=2 violations: wrap in AggregateDeclarationContractViolation,
                        raise.
    """
    violations: list[AuditEvidenceBase] = []
    for contract in _registry_for_site(site):
        if not contract.applies_to(plugin):
            continue
        try:
            invoke(contract)
        except DeclarationContractViolation as exc:
            exc._attach_contract_name(contract.name)
            violations.append(exc)
        except PluginContractViolation as exc:
            # PassThroughContractViolation inherits PluginContractViolation
            # (predates ADR-010 DCV hierarchy). It carries its own 9-key
            # payload and is registered in TIER_1_ERRORS. Catch at this
            # boundary so audit-complete is enforced across BOTH violation
            # hierarchies — otherwise PassThrough's raise would short-circuit
            # the loop and shadow every later contract on the same row.
            #
            # PluginContractViolation does NOT support _attach_contract_name
            # (no C4 one-shot flag). The authoritative contract_name is
            # recorded only on DCV subclasses; PassThroughContractViolation
            # carries ``transform`` and ``transform_node_id`` in its 9-key
            # payload, which is semantically equivalent for that specific
            # exception class.
            violations.append(exc)

    if not violations:
        return
    if len(violations) == 1:
        raise violations[0]  # reference-equality fast path (N6)
    agg = AggregateDeclarationContractViolation(
        plugin=_serialize_plugin_name(plugin),
        violations=tuple(violations),
        message=_build_aggregate_message(violations),
    )
    agg._attach_by_dispatcher()
    raise agg


# Per-site registry filter. Populated at register_declaration_contract time
# from the contract's decorated-method set.
def _registry_for_site(site: DispatchSite) -> Sequence[DeclarationContract]:
    """Return contracts that implement ``site`` (marked with the decorator)."""
```

### 4.2 Public dispatch functions

```python
def run_pre_emission_checks(inputs: PreEmissionInputs) -> None:
    _dispatch(
        site=DispatchSite.PRE_EMISSION,
        plugin=inputs.plugin,
        invoke=lambda c: c.pre_emission_check(inputs),
    )


def run_post_emission_checks(
    inputs: PostEmissionInputs,
    outputs: PostEmissionOutputs,
) -> None:
    _dispatch(
        site=DispatchSite.POST_EMISSION,
        plugin=inputs.plugin,
        invoke=lambda c: c.post_emission_check(inputs, outputs),
    )


def run_batch_flush_checks(
    inputs: BatchFlushInputs,
    outputs: BatchFlushOutputs,
) -> None:
    _dispatch(
        site=DispatchSite.BATCH_FLUSH,
        plugin=inputs.plugin,
        invoke=lambda c: c.batch_flush_check(inputs, outputs),
    )


def run_boundary_checks(
    inputs: BoundaryInputs,
    outputs: BoundaryOutputs,
) -> None:
    _dispatch(
        site=DispatchSite.BOUNDARY,
        plugin=inputs.plugin,
        invoke=lambda c: c.boundary_check(inputs, outputs),
    )
```

**No-legacy rename:** the 2A symbol `run_runtime_checks` is DELETED. Every
caller (TransformExecutor line 376, processor lines 737 & 772) is updated
in the same commit. No deprecation shim.

### 4.3 Module docstring revision (M1 — plan-review B2)

Replaces the current "pure orchestration / the only catch-and-enrich is the
one narrow case" framing with audit-complete language citing ADR-010
§Semantics amendment and comment #417.

### 4.4 Tier-1 registration

`AggregateDeclarationContractViolation` gets `@tier_1_error(reason=...)`
so `TIER_1_ERRORS` includes it — on_error routing cannot absorb it, and
downstream `except TIER_1_ERRORS: raise` sites propagate it as intended.

---

## 5. AggregateDeclarationContractViolation (C5 closure)

### 5.1 Shape

```python
@tier_1_error(
    reason="ADR-010 §Semantics: audit-complete dispatch aggregation",
    caller_module=__name__,
)
class AggregateDeclarationContractViolation(AuditEvidenceBase, RuntimeError):
    """Aggregate wrapper for multi-violation (row, call-site) tuples.

    SIBLING class of DeclarationContractViolation (NOT subclass) — a generic
    ``except DeclarationContractViolation`` elsewhere does not absorb this.
    Per comment #417 §Semantics + N3 §Acceptance C5 closure + S2-001.

    Triage SQL: ``WHERE is_aggregate = true`` distinguishes multi-fire rows.
    Individual violation lookup remains via ``exception_type = '<name>'``
    against child rows — but with aggregate wrapping, the audit table stores
    ONE row with ``is_aggregate=true`` and ``violations=[...]`` inside the
    ExecutionError.context JSON. SQL queries that filter on exception_type
    = 'DeclarationContractViolation' MUST be updated to also match the
    aggregate when callers want all audit-boundary failures.
    """

    __slots__ = ("_attached_by_dispatcher", "plugin", "violations")

    def __init__(
        self,
        *,
        plugin: str,
        violations: tuple[AuditEvidenceBase, ...],
        message: str,
    ) -> None:
        super().__init__(message)
        if len(violations) < 2:
            raise ValueError(
                "AggregateDeclarationContractViolation requires >=2 violations; "
                "single-violation case must raise violations[0] via reference "
                "equality (N6 regression invariant)."
            )
        self._attached_by_dispatcher: bool = False
        self.plugin = plugin
        self.violations = violations  # frozen tuple (__slots__, frozen-semantics)

    def _attach_by_dispatcher(self) -> None:
        """One-shot attribution flag. Mirrors the C4 closure on DCV children:
        the dispatcher is the only code path that raises this class; a non-
        dispatcher raise would bypass audit-complete invariants.
        """
        if self._attached_by_dispatcher:
            raise RuntimeError(
                "AggregateDeclarationContractViolation._attach_by_dispatcher "
                "called twice — dispatcher bug or double-raise attempt."
            )
        self._attached_by_dispatcher = True

    def to_audit_dict(self) -> Mapping[str, Any]:
        if not self._attached_by_dispatcher:
            raise RuntimeError(
                "AggregateDeclarationContractViolation.to_audit_dict accessed "
                "before dispatcher attribution. Aggregate was raised outside "
                "the audit-complete dispatcher path — this is a framework bug."
            )
        return {
            "exception_type": "AggregateDeclarationContractViolation",
            "is_aggregate": True,
            "plugin": self.plugin,
            "violations": tuple(v.to_audit_dict() for v in self.violations),
            "message": str(self),
        }
```

**Note:** `is_aggregate: True` and no `contract_name` — per C5/S2-001, the
sentinel-string-in-name-column pattern is a Spoofing surface and rejected.
Triage via the explicit `is_aggregate` boolean.

### 5.2 Message composition

```python
def _build_aggregate_message(violations: tuple[AuditEvidenceBase, ...]) -> str:
    """Compose the aggregate's message from child messages.

    Format: "M contracts fired on row: <child-message-1>; <child-message-2>; ..."
    Keeps the message human-triage-readable while the structured violations
    list in to_audit_dict() carries authoritative per-child payloads.
    """
```

### 5.3 Harness pattern update

The N2 harnesses iterate `registered_declaration_contracts()` and today call
`contract.runtime_check(inputs, outputs)`. Under the new shape, the harness
reads the site-tagged `ExampleBundle`:

```python
def _invoke_example(contract: DeclarationContract, bundle: ExampleBundle) -> None:
    method = getattr(contract, bundle.site.value)
    method(*bundle.args)
```

The harness asserts `AuditEvidenceBase` rather than `DeclarationContractViolation`
to accommodate the PassThrough case (its direct raise is a `PluginContractViolation`).

---

## 6. Registry restructure

### 6.1 Registry state

```python
# Module-global. Populated at module-import time via register_declaration_contract.
_REGISTRY: list[DeclarationContract] = []
_REGISTRY_BY_SITE: dict[DispatchSite, list[DeclarationContract]] = {
    site: [] for site in DispatchSite
}
_FROZEN: bool = False
```

### 6.2 `register_declaration_contract`

Extended to:
1. Validate all existing invariants (unique name, payload_schema type, example callables).
2. **NEW:** walk the contract's class hierarchy for `@implements_dispatch_site` markers.
3. **NEW:** require at least one site (a contract with zero implemented sites is a config bug — raise TypeError).
4. **NEW:** append to `_REGISTRY_BY_SITE[site]` for each implemented site, in addition to `_REGISTRY`.

### 6.3 Registry introspection for manifest scanner

`_REGISTRY_BY_SITE` is the runtime source of truth. At bootstrap, the manifest
equality check becomes per-(name, site) — every registered
`(contract.name, site)` pair is asserted present in the extended manifest,
and vice-versa.

### 6.4 `EXPECTED_CONTRACTS` extension (N1 Fix direction)

```python
# Before (2A): frozenset of names.
EXPECTED_CONTRACTS: frozenset[str] = frozenset({"passes_through_input"})

# After (H2+N1): FrozenDict name -> frozenset[DispatchSiteName].
EXPECTED_CONTRACT_SITES: Mapping[str, frozenset[DispatchSiteName]] = MappingProxyType({
    "passes_through_input": frozenset({"post_emission_check", "batch_flush_check"}),
})
```

The 2A `EXPECTED_CONTRACTS` name is replaced; the manifest scanner's
`_MANIFEST_SYMBOL = "EXPECTED_CONTRACTS"` constant is renamed to
`"EXPECTED_CONTRACT_SITES"`. Not a legacy-shim — the only consumer is the
scanner and the `prepare_for_run()` bootstrap assertion, both updated in the
same commit.

**PassThroughDeclarationContract claims both `post_emission_check` and
`batch_flush_check`** because `_cross_check_flush_output` uses the same
contract for batch-flush TRANSFORM mode. Under the 2A single-site dispatcher
this was invisible; under multi-site dispatch the claim is explicit.

Actually, let me reconsider: does pass-through really have TWO dispatch
sites, or is batch-flush just a second *caller* of the same post-emission
check? The ADR-009 §Clause 2 batch-homogeneous semantics pass
`effective_input_fields = intersection(per_token_fields)` but invoke the
same contract body (`PassThroughDeclarationContract.runtime_check`) with the
same logic. Under the new shape, batch-flush is a SEPARATE dispatch site
because it has a different bundle (`BatchFlushInputs` with
`buffered_tokens`). Pass-through's contract needs
`batch_flush_check(inputs, outputs)` with the same intersection semantics
as its `post_emission_check`.

Resolution: pass-through implements BOTH sites, with the batch-flush method
internally building a `PostEmissionInputs` equivalent from the batch bundle
and invoking its own `post_emission_check`. Or — simpler — both methods
delegate to a shared private `_check(input_fields, emitted_rows)` helper.
Final decision at implementation time in Phase 2; the design point is that
BOTH sites are claimed in `EXPECTED_CONTRACT_SITES`.

---

## 7. MC3a/b/c scanner (N1 §Acceptance)

### 7.1 Detection mechanism

The existing scanner (`scripts/cicd/enforce_contract_manifest.py`) walks
files for `register_declaration_contract(...)` calls and extracts
`class.name = "..."` via AST.

Extensions:

```python
# --- MC3 data structures ---
@dataclass(frozen=True)
class SiteClaim:
    """A (contract_class, site, source) triple discovered by the scanner.

    Two sources:
    - MARKER: @implements_dispatch_site("site") decorator on a method.
    - MANIFEST: entry in EXPECTED_CONTRACT_SITES manifest.
    """
    contract_class: str
    site: DispatchSiteName
    source: Literal["MARKER", "MANIFEST"]
    file_path: str
    line: int

# --- MC3 rules ---
# MC3a: MARKER claim has no corresponding MANIFEST claim.
# MC3b: MANIFEST claim has no corresponding MARKER claim.
# MC3c: MARKER claim points at a method whose body is structurally trivial.
```

### 7.2 MC3c — structurally-trivial body

Per plan-review W5 + Security S2-003, a body is structurally trivial if it
consists ONLY of:

1. `pass` statements.
2. `return None` / bare `return`.
3. `...` (Ellipsis literal — AST `Expr(value=Constant(value=...))`).
4. Docstring — a leading `Expr(value=Constant(value=<str>))` is permitted
   ONLY as the first statement; subsequent bare literal expressions are
   trivial.
5. Bare literal expression statements (e.g. `Expr(value=Constant(value=1))`,
   `Expr(value=Constant(value=True))`).

A body is non-trivial if it contains at least ONE:
- `Raise`, `Return` with non-`None`-Constant value.
- `Call` (any function or method call except `Constant(Ellipsis)`).
- `Attribute` read on a non-self name.
- `Assign` / `AugAssign`.
- `If`, `For`, `While`, `With`, `Try`, `AsyncFor`, etc. (control-flow).

Implementation: AST visitor that walks the method body, returns False on
first non-trivial node.

### 7.3 Marker discovery

Method decorators on `ast.FunctionDef.decorator_list`:

```python
def _extract_marker_sites(class_node: ast.ClassDef) -> dict[str, str]:
    """Return ``{method_name: site_name}`` for every decorated method.

    Only inspects DIRECT class body methods (per the existing scanner's
    canonical-form constraint). Cross-module/mixin inheritance is not
    resolved — contracts with shared method bodies via mixins MUST carry
    the marker on the concrete class (per Python Engineer F1 / D1
    correction).
    """
    sites: dict[str, str] = {}
    for stmt in class_node.body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        for decorator in stmt.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            if name != "implements_dispatch_site":
                continue
            if not decorator.args:
                continue
            site_arg = decorator.args[0]
            if not (isinstance(site_arg, ast.Constant) and isinstance(site_arg.value, str)):
                continue
            sites[stmt.name] = site_arg.value
    return sites
```

### 7.4 OVERLAP_DETECTION_FIXTURES manifest

```python
# New manifest in declaration_contracts.py (L0).
# Maps plugin-protocol name (e.g., "TransformProtocol") to a representative
# fixture plugin instance whose applies_to results are fed into a second-level
# CI gate that detects whether contracts claim applies_to=True for plugins
# they shouldn't cover.
#
# CLOSED SET — every plugin protocol that has registered contracts MUST be
# represented. Scanner fails CI if a new plugin protocol lands without
# fixture coverage. (Reverses N3-C's polarity per D2 correction.)

OVERLAP_DETECTION_FIXTURES: Mapping[str, Callable[[], Any]] = MappingProxyType({
    "TransformProtocol": _make_transform_fixture,
    "SourceProtocol": _make_source_fixture,  # stub until 2C lands
    "SinkProtocol": _make_sink_fixture,      # stub until 2C lands
})
```

The manifest's enforcement:
- AST inspection of `applies_to` body for statically-detectable type checks.
- Runtime fixture sweep: instantiate each fixture, call every contract's
  `applies_to`, record the resulting cross-product. Dispatcher-level
  overlap is OK (design intent); fixture-level false positives (contract
  claims to apply to wrong plugin kind) fail CI.

### 7.5 File layout

New scanner module OR extension of the existing script. **Decision:**
extend the existing `scripts/cicd/enforce_contract_manifest.py`. The new
rules share the fundamental infra (AST parsing, allowlist, reporting).
Adding a `check` subcommand flag (or new subcommand `check-sites`) keeps
the existing MC1/MC2 checks behaviour unchanged while layering MC3a/b/c.

---

## 8. F2 — pre-emission dispatcher call site

### 8.1 TransformExecutor wiring

Insert AFTER input validation (line 249), BEFORE `transform.process()`
(line 318):

```python
# Pre-emission declaration-contract dispatch (ADR-010 §Decision 3, F2).
# Fires BEFORE process() so adopters like DeclaredRequiredFieldsContract can
# validate input-row field presence without the transform already having
# failed on a missing field (which would attribute the failure to the
# transform's process() body rather than the declaration violation).
effective_input_fields = frozenset(
    fc.normalized_name for fc in token.row_data.contract.fields
)
run_pre_emission_checks(
    inputs=PreEmissionInputs(
        plugin=transform,
        node_id=transform.node_id,
        run_id=ctx.run_id,
        row_id=token.row_id,
        token_id=token.token_id,
        input_row=token.row_data,
        static_contract=<...same derivation as post-emission...>,
        effective_input_fields=effective_input_fields,
    ),
)
```

The post-emission dispatch (existing block at line 376-387) is updated to
reuse `effective_input_fields` (no longer derived inside contracts) and use
the new bundle types:

```python
run_post_emission_checks(
    inputs=PostEmissionInputs(
        plugin=transform,
        ...
        effective_input_fields=effective_input_fields,  # reused from pre-emission
    ),
    outputs=PostEmissionOutputs(emitted_rows=emitted_rows),
)
```

### 8.2 Shared helper — dispatch invocation

Per F2 §Acceptance "single shared helper with the post-emission dispatcher,
not parallel implementation", §4.1's `_dispatch` IS the shared helper. The
four public functions (`run_pre_emission_checks`, `run_post_emission_checks`,
`run_batch_flush_checks`, `run_boundary_checks`) are thin wrappers. This
satisfies the constraint.

---

## 9. F3 — inline sink check reclassification

### 9.1 New exception class

```python
@tier_1_error(
    reason="ADR-010 F3: sink transactional-boundary invariant distinct from pre-write VAL",
    caller_module=__name__,
)
class SinkTransactionalInvariantError(PluginContractViolation):
    """Raised by sink inline checks AT the sink's commit boundary.

    Distinct audit signature from ``SinkRequiredFieldsContract``'s
    ``DeclarationContractViolation`` subclass (when 2C lands) — the contract
    runs pre-write (dispatcher-owned), this class runs at the commit
    boundary (inline, after contract passed). An auditor querying
    ``exception_type = 'SinkTransactionalInvariantError'`` gets transactional-
    backstop failures; ``exception_type = 'SinkRequiredFieldsViolation'``
    (or whichever DCV subclass 2C introduces) gets pre-write contract
    failures. Before F3, both paths raised ``PluginContractViolation`` and
    the audit table conflated them.

    F3 fix direction acceptance bullet 2: "renamed to a transactional-
    backstop-specific class".
    """
```

Inheriting from `PluginContractViolation` (not `DeclarationContractViolation`)
preserves the existing sink-executor semantics (row-level failure, legitimate
Tier-1 but not registry-owned). `PluginContractViolation` already inherits
`AuditEvidenceBase` + `RuntimeError`.

### 9.2 Sink-executor changes

In `src/elspeth/engine/executors/sink.py`, `_validate_sink_input`:

- Input-schema validation (line 179-181): continues to raise
  `PluginContractViolation` (this is Tier-2 upstream-schema-bug territory,
  not a transactional invariant). Unchanged.
- `declared_required_fields` check (line 218-221): RECLASSIFY to
  `SinkTransactionalInvariantError`. Same message, new class.

Comment added at the site (F3 §Fix direction bullet 3):

```python
# TWO-LAYER SINK INVARIANT ARCHITECTURE (ADR-010 F3)
# ---------------------------------------------------
# Layer 1 (pre-write): SinkRequiredFieldsContract (when 2C lands,
#   elspeth-ea5e9e4759). Dispatcher-owned, fires before sink.write(),
#   raises DeclarationContractViolation subclass. Auditor query:
#   WHERE exception_type = 'SinkRequiredFieldsViolation' (future class name).
#
# Layer 2 (transactional backstop, this site): catches the rare case where
#   state diverges between contract evaluation and commit. Raises
#   SinkTransactionalInvariantError. Auditor query:
#   WHERE exception_type = 'SinkTransactionalInvariantError'.
#
# Both are Tier-1 (cannot be absorbed by on_error routing); the exception
# class distinguishes WHICH layer fired so triage SQL can target either
# without conflation.
```

### 9.3 Integration test

Per F3 §Acceptance: exercise BOTH layers on one sink.
- Row violates required_fields at contract layer → pre-write contract fires →
  audit record with exception_type = `SinkRequiredFieldsViolation`.
- Row passes contract (all fields present) but some external mutation between
  contract and commit (simulated in test fixture) → inline check fires →
  audit record with exception_type = `SinkTransactionalInvariantError`.

**2C-dependency note:** the pre-write contract (`SinkRequiredFieldsContract`)
lands with the paired 2C PR per F4. THIS PR reclassifies the inline check
and adds the exception class; the pre-write contract comes later. The
integration test exercising BOTH layers defers to 2C. This PR's F3 test
covers only the reclassification (the inline check, given a row violating
required fields today, raises the NEW exception class).

F3 acceptance bullet 5 ("exercises both layers") is therefore PARTIALLY met
by this PR and COMPLETED by the 2C paired-landing PR. Noted in F3's ticket
amendment.

---

## 10. F4 — per-surface rule-of-three

### 10.1 §Adoption State section in ADR-010 amendment

```
## Adoption State (per dispatch surface) — F4 amendment

| Surface               | Count | Rule-of-three satisfied? | Adopters                                |
|-----------------------|-------|--------------------------|-----------------------------------------|
| pre_emission_check    |   0   | NO — 3 needed            | (F2 lands the site; first adopter is    |
|                       |       |                          | DeclaredRequiredFieldsContract in 2B)   |
| post_emission_check   |   1   | NO — 2 more needed        | PassThroughDeclarationContract           |
| batch_flush_check     |   1   | NO — 2 more needed        | PassThroughDeclarationContract           |
| boundary_check        |   0   | NO — 3 needed            | (2C paired adopters will add source +   |
|                       |       |                          | sink simultaneously per F4 rule)         |
```

### 10.2 Paired-landing rule (boundary subtype)

Documented in the ADR amendment: the boundary subtype lands with BOTH
`SourceGuaranteedFieldsContract` AND `SinkRequiredFieldsContract` in a
single commit/PR. Staggered one-boundary-adopter landings are rejected.

---

## 11. Single PR vs W9 split — recommendation and rationale

**Recommendation: SINGLE PR.** Close all 8 tickets in one landing.

Rationale:
1. The contracts module, dispatcher, bundle types, and decorator are
   mutually referential at the IMPORT level. Splitting into (H2 + N1) +
   (F2 + F3 + F4) creates a first sub-PR where the bundle types exist but
   the callers still use `run_runtime_checks` — incoherent framework state,
   failing tests, no middle ground.
2. The `override_input_fields` → `effective_input_fields` rename touches
   every call site in one pass. Splitting would require a multi-sub-PR
   rename with transitional both-work code (no-legacy-code policy rejects
   this).
3. The catch-site survey and aggregate-handling changes in
   `_cross_check_flush_output` cross the H2 / N3 / F3 boundaries. A split
   would fragment the survey across PRs.
4. The single-PR path is 8 tickets × ~10 LOC each for acceptance + ~500 LOC
   contracts module refactor + ~300 LOC dispatcher + ~400 LOC scanner +
   ~200 LOC tests ≈ 1400 LOC new + ~200 LOC touched in existing files.
   This is within the bounds the panel anticipated for a coherent H2
   landing.

**W9 escape hatch declined unless a specific reviewer objection demands it.**
The escape hatch requires a consistency integration test as a hard gate in
the first sub-PR — which is LARGER scope than landing the whole thing in one
coherent PR.

---

## 12. Verification matrix (Phase 5)

| Check | Command | Expected |
|-------|---------|----------|
| Full pytest | `.venv/bin/python -m pytest tests/` | all pass |
| Mypy | `.venv/bin/python -m mypy src/` | clean |
| Ruff lint | `.venv/bin/python -m ruff check src/` | clean |
| Tier model | `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` | clean |
| Contract manifest | `.venv/bin/python scripts/cicd/enforce_contract_manifest.py check` | clean (MC3a/b/c all pass) |
| Benchmark | `.venv/bin/python -m pytest tests/performance/benchmarks/test_cross_check_overhead.py` | Budget met at N ∈ {1,2,4,8,16} |
| N3 regression N=1 | `tests/unit/engine/test_declaration_dispatch.py::test_single_violation_reference_identity` | type() + id() equality |
| N3 order-independence | `pytest -p no:randomly --forked tests/unit/engine/test_aggregate_violation.py` | aggregate content equal across registration orders |
| N3 property test | `tests/unit/contracts/test_registry_snapshot_property.py` | Hypothesis 100-example run passes |
| F5 E2E round-trip | `tests/integration/audit/test_declaration_contract_landscape_roundtrip.py::test_aggregate_round_trip` | is_aggregate=true + violations list recoverable via explain() |
| MC3c fixtures | `tests/cicd/test_enforce_contract_manifest_mc3c.py` | pass-only body fails CI; ...-only body fails CI |

---

## 13. Out-of-scope / deferred

- **Checkpoint + resume of aggregate violations.** Aggregate exceptions
  don't need special checkpoint handling — they're terminal failures that
  complete the row as FAILED with `ExecutionError.context` carrying the
  full `violations: [...]` list. Resume replays from the last checkpoint;
  aggregate violations from prior runs are static audit records at that
  point. No Alembic migration needed.
- **Landscape schema migration.** `ExecutionError.context` is already a JSON
  column supporting arbitrary shape. `is_aggregate: True` + `violations: [...]`
  is a new JSON-internal shape, not a new column. No migration.
- **MCP server / `explain_token` surfacing aggregates.** The `explain()`
  function already returns `NodeStateFailed.error_json` verbatim; aggregate
  audit records appear under the same surface. Human readability of the
  aggregate's `violations` list can improve in a follow-up; no code change
  required for correctness.
- **Telemetry emission of aggregates.** Per-child counters (e.g.
  pass-through's `_VIOLATIONS_COUNTER`) increment BEFORE the child's
  violation raises from inside the contract body. Aggregation happens
  AFTER. Counters are correct by construction. No code change.
- **`DeclaredRequiredFieldsContract` implementation.** Phase 2B work,
  blocked on this landing. F2 landing only WIRES the dispatcher call
  site; the first pre-emission contract is separately tracked.
- **2C adopters (`Source/SinkBoundaryContract`).** Blocked on this landing;
  separate paired-landing PR per F4.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| `effective_input_fields` derivation at caller diverges from contract expectations | Single private helper in `declaration_contracts.py`: `derive_effective_input_fields(row)` — every caller uses it. |
| AggregateDeclarationContractViolation's `violations` list is concurrently mutated after dispatcher raise | Stored as `tuple` (immutable). Each child's payload is already deep-frozen at its own `__init__` (H5 Layer 1). |
| Dispatcher catches too broadly (swallows non-contract exceptions) | Catch restricted to `DeclarationContractViolation` and `PluginContractViolation`. Explicit — no `Exception` fallback. Per-catch test case verifies non-contract `RuntimeError` propagates unmodified. |
| `@implements_dispatch_site` site-name typo at decoration time | Validated at decoration via `site_name not in DispatchSite` → `ValueError` at import. |
| MC3c false positives on legitimate single-line method bodies (e.g., `raise MyViolation(...)`) | MC3c rules ONLY match `pass`, `return`/`return None`, `...`, bare literal expressions. A single `raise` statement is non-trivial → passes. AST unit tests exercise the edge cases. |
| Scanner fails on cross-module contract classes | Preserved constraint: canonical form is `register_declaration_contract(SomeClass())` with `SomeClass` in the same file, OR `SomeClass` carries the marker decorator (CI detects via the class-level `_declaration_dispatch_sites` attribute the decorator sets). Cross-module without either → MC1 finding per existing rule. |

---

## 15. Implementation order (Phase 2 dependencies)

1. `contracts/declaration_contracts.py` — bundle types, `DispatchSite` enum,
   `@implements_dispatch_site`, base ABC, `AggregateDeclarationContractViolation`,
   registry restructure, `EXPECTED_CONTRACT_SITES` manifest.
2. `engine/executors/declaration_dispatch.py` — `_dispatch` shared helper,
   4 public functions, module docstring rewrite.
3. `engine/executors/pass_through.py` — `PassThroughDeclarationContract`
   inherits `DeclarationContract` ABC, claims both
   `post_emission_check` and `batch_flush_check` via marker decorator,
   implementations use `effective_input_fields`.
4. `engine/executors/transform.py` — replace `run_runtime_checks` with
   `run_post_emission_checks` + add `run_pre_emission_checks` call site (F2).
5. `engine/processor.py` — update `_cross_check_flush_output` to use
   `run_batch_flush_checks` + bundle types. Update catch clause to handle
   both `PassThroughContractViolation` single-violation AND
   `AggregateDeclarationContractViolation` multi-violation cases.
6. `engine/executors/sink.py` — reclassify inline check's raise class to
   `SinkTransactionalInvariantError` (F3).
7. `contracts/errors.py` — add `SinkTransactionalInvariantError` class
   (F3).
8. `scripts/cicd/enforce_contract_manifest.py` — add MC3a/b/c rules, marker
   AST detection, OVERLAP_DETECTION_FIXTURES enforcement.
9. Tests — updated fixtures, new aggregate tests, N3 regression, N3
   property test, F5 round-trip, MC3c CI fixtures.
10. ADR-010 amendment — §Semantics subsection, §Adoption State table,
    cross-references.

Each step compiles and passes existing tests (where applicable) before
moving to the next. Tests written alongside implementation per TDD.

---

**End of Phase 1 design sketch.**
