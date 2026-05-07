"""Property tests for ADR-010 dispatch surfaces (H2 §Acceptance F-QA-5).

Closes the final ADR-010 H2 acceptance bullet. The prior H2-B landing at
commit 009b6009 delivered the bundle types
(``PreEmissionInputs`` / ``PostEmissionInputs`` / ``BatchFlushInputs`` /
``BoundaryInputs``) as frozen slots dataclasses with no ``Optional``/``None``
fields, satisfying the *structural* half of F-QA-5. This file delivers the
*evidentiary* half:

    Quality Engineer F-QA-5, ticket §Acceptance:
      "each dispatch method's input bundle type MUST be a concrete dataclass
       or TypedDict whose ``@given(...)`` strategy can be derived without
       conditional ``assume(x is not None)`` guards. Validate by writing one
       Hypothesis property test per dispatch surface before accepting H2 as
       closed."

Three properties per surface x 4 surfaces = 12 ``@given`` tests.

* **Property A** — empty applicable-registry ⇒ dispatcher returns ``None``.
  Proves the strategy derives cleanly across the full bundle-type inhabitation
  space (no ``assume()`` filters, no shrinker dead ends) and that the 0-raise
  branch of ``_dispatch`` is total.

* **Property B** — N=1 raiser ⇒ dispatcher raises via reference equality
  (N6 regression invariant — ADR-010 §Semantics). Asserts
  ``type(raised) is <original subclass>`` AND
  ``id(raised) == id(contract.last_raised)`` so an accidental
  "aggregation-of-one" wrapper regresses this test.

* **Property C** — N=2 raisers ⇒ dispatcher raises
  ``AggregateDeclarationContractViolation`` whose ``.violations`` tuple
  carries both child exceptions in registration order, each with authoritative
  ``contract_name`` attribution (C4 closure). Asserts
  ``isinstance(raised, DeclarationContractViolation)`` is **False** — the
  aggregate is a SIBLING class per N3 §Acceptance C5, not a subclass.

§Acceptance — Test-helper discipline (S2-008): any test-only helper that
mutates the contract registry gates on ``_require_pytest_process(name)``. The
underlying helpers imported from ``declaration_contracts`` already gate; this
module's ``_isolate_registry`` wrapper adds an explicit gate call at the
wrapper boundary so the discipline is self-evident without tracing through
two imports.
"""

from __future__ import annotations

from typing import Any, ClassVar, TypedDict

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    BatchFlushInputs,
    BatchFlushOutputs,
    BoundaryInputs,
    BoundaryOutputs,
    DeclarationContract,
    DeclarationContractViolation,
    DispatchSite,
    ExampleBundle,
    PostEmissionInputs,
    PostEmissionOutputs,
    PreEmissionInputs,
    _clear_registry_for_tests,
    _require_pytest_process,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.engine.executors.declaration_dispatch import (
    run_batch_flush_checks,
    run_boundary_checks,
    run_post_emission_checks,
    run_pre_emission_checks,
)

# =============================================================================
# TypedDict payload + test-only violation subclasses
# =============================================================================


class _Payload(TypedDict):
    note: str


class _TestViolationA(DeclarationContractViolation):
    payload_schema = _Payload


class _TestViolationB(DeclarationContractViolation):
    payload_schema = _Payload


# =============================================================================
# Hypothesis strategies — one per bundle type
# =============================================================================
#
# Every strategy below is a plain ``st.builds(...)`` call. No ``assume()``
# filter, no ``@st.composite`` workaround, no ``.filter(...)`` shrinker tax.
# ``BoundaryInputs.row_contract`` is intentionally nullable because a
# failsink-enriched row may have no corresponding primary contract; the
# strategy models that optionality directly rather than by filtering.

_IDENTITY_CHARS = st.characters(
    whitelist_categories=("Ll", "Lu", "Nd"),
    whitelist_characters="_-",
)
_IDENTITY_STR = st.text(_IDENTITY_CHARS, min_size=1, max_size=24)
_FIELD_NAME = st.text(_IDENTITY_CHARS, min_size=1, max_size=16)
_FIELD_SET = st.frozensets(_FIELD_NAME, max_size=6)


def _named_plugin(name: str) -> Any:
    plugin = type("PropertyPlugin", (), {})()
    plugin.name = name
    return plugin


def _plugin_strategy() -> st.SearchStrategy[Any]:
    """Minimal plugin-like objects with the mandatory ``name`` attribute.

    ADR-010 treats a missing ``plugin.name`` as a framework bug, so property
    strategies that exercise the aggregate path must populate it explicitly.
    Contracts under test still decide all other applicability semantics.
    """
    return st.builds(_named_plugin, _IDENTITY_STR)


def _row_like_strategy() -> st.SearchStrategy[dict[str, Any]]:
    return st.dictionaries(
        _FIELD_NAME,
        st.integers() | st.text(max_size=24) | st.booleans(),
        max_size=4,
    )


def pre_emission_inputs_strategy() -> st.SearchStrategy[PreEmissionInputs]:
    return st.builds(
        PreEmissionInputs,
        plugin=_plugin_strategy(),
        node_id=_IDENTITY_STR,
        run_id=_IDENTITY_STR,
        row_id=_IDENTITY_STR,
        token_id=_IDENTITY_STR,
        input_row=_row_like_strategy(),
        static_contract=_FIELD_SET,
        effective_input_fields=_FIELD_SET,
    )


def post_emission_inputs_strategy() -> st.SearchStrategy[PostEmissionInputs]:
    return st.builds(
        PostEmissionInputs,
        plugin=_plugin_strategy(),
        node_id=_IDENTITY_STR,
        run_id=_IDENTITY_STR,
        row_id=_IDENTITY_STR,
        token_id=_IDENTITY_STR,
        input_row=_row_like_strategy(),
        static_contract=_FIELD_SET,
        effective_input_fields=_FIELD_SET,
    )


def post_emission_outputs_strategy() -> st.SearchStrategy[PostEmissionOutputs]:
    return st.builds(
        PostEmissionOutputs,
        emitted_rows=st.lists(_row_like_strategy(), max_size=4).map(tuple),
    )


def batch_flush_inputs_strategy() -> st.SearchStrategy[BatchFlushInputs]:
    return st.builds(
        BatchFlushInputs,
        plugin=_plugin_strategy(),
        node_id=_IDENTITY_STR,
        run_id=_IDENTITY_STR,
        row_id=_IDENTITY_STR,
        token_id=_IDENTITY_STR,
        buffered_tokens=st.lists(_row_like_strategy(), max_size=4).map(tuple),
        static_contract=_FIELD_SET,
        effective_input_fields=_FIELD_SET,
    )


def batch_flush_outputs_strategy() -> st.SearchStrategy[BatchFlushOutputs]:
    return st.builds(
        BatchFlushOutputs,
        emitted_rows=st.lists(_row_like_strategy(), max_size=4).map(tuple),
    )


def boundary_inputs_strategy() -> st.SearchStrategy[BoundaryInputs]:
    return st.builds(
        BoundaryInputs,
        plugin=_plugin_strategy(),
        node_id=_IDENTITY_STR,
        run_id=_IDENTITY_STR,
        row_id=_IDENTITY_STR,
        token_id=_IDENTITY_STR,
        static_contract=_FIELD_SET,
        row_data=_row_like_strategy(),
        row_contract=st.none() | st.builds(object),
    )


def boundary_outputs_strategy() -> st.SearchStrategy[BoundaryOutputs]:
    return st.builds(
        BoundaryOutputs,
        rows=st.lists(_row_like_strategy(), max_size=4).map(tuple),
    )


def _seed_pre_emission_bundle() -> PreEmissionInputs:
    return PreEmissionInputs(
        plugin=_named_plugin("seed-pre"),
        node_id="node_pre",
        run_id="run_pre",
        row_id="row_pre",
        token_id="token_pre",
        input_row={"customer_id": 7, "amount": "42"},
        static_contract=frozenset({"customer_id", "amount"}),
        effective_input_fields=frozenset({"customer_id", "amount"}),
    )


def _seed_post_emission_inputs() -> PostEmissionInputs:
    return PostEmissionInputs(
        plugin=_named_plugin("seed-post"),
        node_id="node_post",
        run_id="run_post",
        row_id="row_post",
        token_id="token_post",
        input_row={"customer_id": 7, "amount": "42"},
        static_contract=frozenset({"customer_id", "amount"}),
        effective_input_fields=frozenset({"customer_id", "amount"}),
    )


def _seed_post_emission_outputs() -> PostEmissionOutputs:
    return PostEmissionOutputs(
        emitted_rows=({"customer_id": 7, "status": "ok"},),
    )


def _seed_batch_flush_inputs() -> BatchFlushInputs:
    return BatchFlushInputs(
        plugin=_named_plugin("seed-batch"),
        node_id="node_batch",
        run_id="run_batch",
        row_id="row_batch",
        token_id="token_batch",
        buffered_tokens=(
            {"customer_id": 7, "amount": "42"},
            {"customer_id": 8, "amount": "99"},
        ),
        static_contract=frozenset({"customer_id", "amount"}),
        effective_input_fields=frozenset({"customer_id", "amount"}),
    )


def _seed_batch_flush_outputs() -> BatchFlushOutputs:
    return BatchFlushOutputs(
        emitted_rows=({"batch_size": 2, "status": "ok"},),
    )


def _seed_boundary_inputs() -> BoundaryInputs:
    return BoundaryInputs(
        plugin=_named_plugin("seed-boundary"),
        node_id="node_boundary",
        run_id="run_boundary",
        row_id="row_boundary",
        token_id="token_boundary",
        static_contract=frozenset({"customer_id"}),
        row_data={"customer_id": 7, "raw": {"source": "csv"}},
        row_contract=None,
    )


def _seed_boundary_outputs() -> BoundaryOutputs:
    return BoundaryOutputs(rows=({"customer_id": 7},))


# =============================================================================
# ExampleBundle constructors — registry requires these to be callable classmethods
# =============================================================================
#
# The N2 Layer A/B harness iterates ``registered_declaration_contracts()`` and
# invokes these. The per-test registry-snapshot fixture clears the registry
# before any test contract is registered and restores afterwards, so harness
# invariants never observe these contracts. Still — the bundles here are
# well-formed so a future harness change that iterates during a test does not
# surface a bogus failure.


def _example_pre_emission() -> ExampleBundle:
    return ExampleBundle(
        site=DispatchSite.PRE_EMISSION,
        args=(
            PreEmissionInputs(
                plugin=object(),
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                input_row={},
                static_contract=frozenset(),
                effective_input_fields=frozenset(),
            ),
        ),
    )


def _example_post_emission() -> ExampleBundle:
    return ExampleBundle(
        site=DispatchSite.POST_EMISSION,
        args=(
            PostEmissionInputs(
                plugin=object(),
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                input_row={},
                static_contract=frozenset(),
                effective_input_fields=frozenset(),
            ),
            PostEmissionOutputs(emitted_rows=()),
        ),
    )


def _example_batch_flush() -> ExampleBundle:
    return ExampleBundle(
        site=DispatchSite.BATCH_FLUSH,
        args=(
            BatchFlushInputs(
                plugin=object(),
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                buffered_tokens=(),
                static_contract=frozenset(),
                effective_input_fields=frozenset(),
            ),
            BatchFlushOutputs(emitted_rows=()),
        ),
    )


def _example_boundary() -> ExampleBundle:
    return ExampleBundle(
        site=DispatchSite.BOUNDARY,
        args=(
            BoundaryInputs(
                plugin=object(),
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                static_contract=frozenset(),
                row_data={},
                row_contract=None,
            ),
            BoundaryOutputs(rows=()),
        ),
    )


# =============================================================================
# Raising-contract fixtures — one class per (site, violation-subclass) pair
# =============================================================================
#
# Each class:
#   * Always applies (``applies_to`` returns True).
#   * Raises a FRESH violation on every dispatch invocation (required —
#     ``_attach_contract_name`` is one-shot; reusing a violation across
#     Hypothesis examples would raise ``RuntimeError`` on the 2nd example).
#   * Stashes the just-raised violation on ``last_raised`` (class attribute)
#     so the Property B assertion can verify reference equality post-raise.
#
# Twelve classes (4 sites x 2 violation subclasses + 4 sites x 1 applies_to=
# False skip probe) is boilerplate-heavy. Factoring through ``type()`` is
# possible but obscures the ``@implements_dispatch_site`` decoration (the AST
# scanner + runtime marker-walker both depend on the decorator's applied-state
# being inspectable); explicit classes keep that transparency.


def _mk_violation_a(message: str) -> _TestViolationA:
    return _TestViolationA(
        plugin="P",
        node_id="n",
        run_id="r",
        row_id="rw",
        token_id="t",
        payload={"note": message},
        message=message,
    )


def _mk_violation_b(message: str) -> _TestViolationB:
    return _TestViolationB(
        plugin="P",
        node_id="n",
        run_id="r",
        row_id="rw",
        token_id="t",
        payload={"note": message},
        message=message,
    )


class _PreEmissionRaiserA(DeclarationContract):
    name: ClassVar[str] = "test_pre_emission_raiser_a"
    payload_schema: ClassVar[type] = _Payload
    last_raised: ClassVar[DeclarationContractViolation | None] = None

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("pre_emission_check")
    def pre_emission_check(self, inputs: PreEmissionInputs) -> None:
        v = _mk_violation_a("pre_a")
        type(self).last_raised = v
        raise v

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_pre_emission()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_pre_emission()


class _PreEmissionRaiserB(DeclarationContract):
    name: ClassVar[str] = "test_pre_emission_raiser_b"
    payload_schema: ClassVar[type] = _Payload
    last_raised: ClassVar[DeclarationContractViolation | None] = None

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("pre_emission_check")
    def pre_emission_check(self, inputs: PreEmissionInputs) -> None:
        v = _mk_violation_b("pre_b")
        type(self).last_raised = v
        raise v

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_pre_emission()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_pre_emission()


class _PostEmissionRaiserA(DeclarationContract):
    name: ClassVar[str] = "test_post_emission_raiser_a"
    payload_schema: ClassVar[type] = _Payload
    last_raised: ClassVar[DeclarationContractViolation | None] = None

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        v = _mk_violation_a("post_a")
        type(self).last_raised = v
        raise v

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_post_emission()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_post_emission()


class _PostEmissionRaiserB(DeclarationContract):
    name: ClassVar[str] = "test_post_emission_raiser_b"
    payload_schema: ClassVar[type] = _Payload
    last_raised: ClassVar[DeclarationContractViolation | None] = None

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        v = _mk_violation_b("post_b")
        type(self).last_raised = v
        raise v

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_post_emission()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_post_emission()


class _BatchFlushRaiserA(DeclarationContract):
    name: ClassVar[str] = "test_batch_flush_raiser_a"
    payload_schema: ClassVar[type] = _Payload
    last_raised: ClassVar[DeclarationContractViolation | None] = None

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("batch_flush_check")
    def batch_flush_check(
        self,
        inputs: BatchFlushInputs,
        outputs: BatchFlushOutputs,
    ) -> None:
        v = _mk_violation_a("flush_a")
        type(self).last_raised = v
        raise v

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_batch_flush()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_batch_flush()


class _BatchFlushRaiserB(DeclarationContract):
    name: ClassVar[str] = "test_batch_flush_raiser_b"
    payload_schema: ClassVar[type] = _Payload
    last_raised: ClassVar[DeclarationContractViolation | None] = None

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("batch_flush_check")
    def batch_flush_check(
        self,
        inputs: BatchFlushInputs,
        outputs: BatchFlushOutputs,
    ) -> None:
        v = _mk_violation_b("flush_b")
        type(self).last_raised = v
        raise v

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_batch_flush()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_batch_flush()


class _BoundaryRaiserA(DeclarationContract):
    name: ClassVar[str] = "test_boundary_raiser_a"
    payload_schema: ClassVar[type] = _Payload
    last_raised: ClassVar[DeclarationContractViolation | None] = None

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("boundary_check")
    def boundary_check(
        self,
        inputs: BoundaryInputs,
        outputs: BoundaryOutputs,
    ) -> None:
        v = _mk_violation_a("boundary_a")
        type(self).last_raised = v
        raise v

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_boundary()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_boundary()


class _BoundaryRaiserB(DeclarationContract):
    name: ClassVar[str] = "test_boundary_raiser_b"
    payload_schema: ClassVar[type] = _Payload
    last_raised: ClassVar[DeclarationContractViolation | None] = None

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("boundary_check")
    def boundary_check(
        self,
        inputs: BoundaryInputs,
        outputs: BoundaryOutputs,
    ) -> None:
        v = _mk_violation_b("boundary_b")
        type(self).last_raised = v
        raise v

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_boundary()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_boundary()


class _PreEmissionNeverRaises(DeclarationContract):
    name: ClassVar[str] = "test_pre_emission_never_raises"
    payload_schema: ClassVar[type] = _Payload

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("pre_emission_check")
    def pre_emission_check(self, inputs: PreEmissionInputs) -> None:
        return None

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_pre_emission()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_pre_emission()


class _PostEmissionNeverRaises(DeclarationContract):
    name: ClassVar[str] = "test_post_emission_never_raises"
    payload_schema: ClassVar[type] = _Payload

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs: PostEmissionInputs, outputs: PostEmissionOutputs) -> None:
        return None

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_post_emission()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_post_emission()


class _BatchFlushNeverRaises(DeclarationContract):
    name: ClassVar[str] = "test_batch_flush_never_raises"
    payload_schema: ClassVar[type] = _Payload

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("batch_flush_check")
    def batch_flush_check(self, inputs: BatchFlushInputs, outputs: BatchFlushOutputs) -> None:
        return None

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_batch_flush()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_batch_flush()


class _BoundaryNeverRaises(DeclarationContract):
    name: ClassVar[str] = "test_boundary_never_raises"
    payload_schema: ClassVar[type] = _Payload

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("boundary_check")
    def boundary_check(self, inputs: BoundaryInputs, outputs: BoundaryOutputs) -> None:
        return None

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _example_boundary()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _example_boundary()


# =============================================================================
# Registry-isolation helpers (S2-008 gated)
# =============================================================================


def _isolate_registry() -> tuple[Any, ...]:
    """Snapshot + clear the declaration-contract registry.

    S2-008 gate (§Acceptance "Test-helper discipline"): every test-only
    registry-mutating helper gates on ``_require_pytest_process``. The
    underlying ``_snapshot_registry_for_tests`` and ``_clear_registry_for_tests``
    already gate; the explicit call at this wrapper boundary documents the
    discipline without relying on transitive enforcement.
    """
    _require_pytest_process("_isolate_registry (dispatch property tests)")
    snapshot = _snapshot_registry_for_tests()
    _clear_registry_for_tests()
    return snapshot


@pytest.fixture
def _empty_registry():
    snapshot = _isolate_registry()
    yield
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _pre_emission_single_raiser():
    snapshot = _isolate_registry()
    contract = _PreEmissionRaiserA()
    _PreEmissionRaiserA.last_raised = None
    register_declaration_contract(contract)
    yield contract
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _pre_emission_two_raisers():
    snapshot = _isolate_registry()
    a = _PreEmissionRaiserA()
    b = _PreEmissionRaiserB()
    _PreEmissionRaiserA.last_raised = None
    _PreEmissionRaiserB.last_raised = None
    register_declaration_contract(a)
    register_declaration_contract(b)
    yield (a, b)
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _post_emission_single_raiser():
    snapshot = _isolate_registry()
    contract = _PostEmissionRaiserA()
    _PostEmissionRaiserA.last_raised = None
    register_declaration_contract(contract)
    yield contract
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _post_emission_two_raisers():
    snapshot = _isolate_registry()
    a = _PostEmissionRaiserA()
    b = _PostEmissionRaiserB()
    _PostEmissionRaiserA.last_raised = None
    _PostEmissionRaiserB.last_raised = None
    register_declaration_contract(a)
    register_declaration_contract(b)
    yield (a, b)
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _batch_flush_single_raiser():
    snapshot = _isolate_registry()
    contract = _BatchFlushRaiserA()
    _BatchFlushRaiserA.last_raised = None
    register_declaration_contract(contract)
    yield contract
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _batch_flush_two_raisers():
    snapshot = _isolate_registry()
    a = _BatchFlushRaiserA()
    b = _BatchFlushRaiserB()
    _BatchFlushRaiserA.last_raised = None
    _BatchFlushRaiserB.last_raised = None
    register_declaration_contract(a)
    register_declaration_contract(b)
    yield (a, b)
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _boundary_single_raiser():
    snapshot = _isolate_registry()
    contract = _BoundaryRaiserA()
    _BoundaryRaiserA.last_raised = None
    register_declaration_contract(contract)
    yield contract
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _boundary_two_raisers():
    snapshot = _isolate_registry()
    a = _BoundaryRaiserA()
    b = _BoundaryRaiserB()
    _BoundaryRaiserA.last_raised = None
    _BoundaryRaiserB.last_raised = None
    register_declaration_contract(a)
    register_declaration_contract(b)
    yield (a, b)
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _pre_emission_never_raises():
    snapshot = _isolate_registry()
    register_declaration_contract(_PreEmissionNeverRaises())
    yield
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _post_emission_never_raises():
    snapshot = _isolate_registry()
    register_declaration_contract(_PostEmissionNeverRaises())
    yield
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _batch_flush_never_raises():
    snapshot = _isolate_registry()
    register_declaration_contract(_BatchFlushNeverRaises())
    yield
    _restore_registry_snapshot_for_tests(snapshot)


@pytest.fixture
def _boundary_never_raises():
    snapshot = _isolate_registry()
    register_declaration_contract(_BoundaryNeverRaises())
    yield
    _restore_registry_snapshot_for_tests(snapshot)


# Hypothesis default is 100 examples; the dispatcher is pure-Python, no I/O,
# microsecond-scale per-example. Default — 100 x 12 tests = 1200 dispatcher
# invocations per suite run, under a second total.
#
# ``function_scoped_fixture`` health check is suppressed INTENTIONALLY: the
# registry fixture sets up once per test function and persists across all
# Hypothesis examples by design. Registering per-example would (a) hit the
# duplicate-name guard in ``register_declaration_contract`` on example #2 and
# (b) misrepresent the property under test — the property is "for any bundle,
# with a FIXED registry state, the dispatcher holds invariant X", not "the
# registry survives setup/teardown cycles."
_DISPATCH_PROPERTY_SETTINGS = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# =============================================================================
# Property tests — PRE-EMISSION dispatch surface
# =============================================================================


@given(bundle=pre_emission_inputs_strategy())  # -> PreEmissionInputs
@_DISPATCH_PROPERTY_SETTINGS
def test_pre_emission_empty_registry_returns_none(_empty_registry, bundle):
    """Property A — no applicable contracts ⇒ dispatcher returns None.

    Exercises the strategy's derivability (F-QA-5 structural half) and the
    0-violation branch of ``_dispatch``.
    """
    assert run_pre_emission_checks(bundle) is None


@given(bundle=pre_emission_inputs_strategy())  # -> PreEmissionInputs
@_DISPATCH_PROPERTY_SETTINGS
def test_pre_emission_applicable_non_raising_registry_returns_none(
    _pre_emission_never_raises,
    bundle,
):
    """Property A' — applicable contracts that all pass still return None."""
    assert run_pre_emission_checks(bundle) is None


@example(bundle=_seed_pre_emission_bundle())
@given(bundle=pre_emission_inputs_strategy())  # -> PreEmissionInputs
@_DISPATCH_PROPERTY_SETTINGS
def test_pre_emission_n1_raises_via_reference_identity(
    _pre_emission_single_raiser,
    bundle,
):
    """Property B — N=1 ⇒ dispatcher re-raises the same violation object.

    N6 regression: the raise path is ``raise violations[0]`` (identity), NOT
    an aggregation-of-one wrapper. If the dispatcher regresses to wrapping,
    ``type(exc_info.value) is _TestViolationA`` fails; if it regresses to
    constructing a fresh wrapper, the ``id()`` comparison fails.
    """
    with pytest.raises(_TestViolationA) as exc_info:
        run_pre_emission_checks(bundle)
    assert type(exc_info.value) is _TestViolationA
    assert _PreEmissionRaiserA.last_raised is not None
    assert id(exc_info.value) == id(_PreEmissionRaiserA.last_raised)
    assert not isinstance(exc_info.value, AggregateDeclarationContractViolation)


@example(bundle=_seed_pre_emission_bundle())
@given(bundle=pre_emission_inputs_strategy())  # -> PreEmissionInputs
@_DISPATCH_PROPERTY_SETTINGS
def test_pre_emission_n2_raises_aggregate(_pre_emission_two_raisers, bundle):
    """Property C — N=2 applicable raisers ⇒ aggregate with 2 children.

    Asserts the aggregate is a SIBLING of ``DeclarationContractViolation``,
    not a subclass (N3 §Acceptance C5). A ``except
    DeclarationContractViolation`` elsewhere must not absorb the aggregate.
    """
    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_pre_emission_checks(bundle)
    agg = exc_info.value
    assert len(agg.violations) == 2
    assert isinstance(agg.violations[0], _TestViolationA)
    assert isinstance(agg.violations[1], _TestViolationB)
    # Sibling, not subclass — triage SQL filters on is_aggregate, not on
    # exception_type being DCV.
    assert not isinstance(agg, DeclarationContractViolation)
    # C4 closure: the dispatcher attached the authoritative name on each child.
    assert agg.violations[0].contract_name == "test_pre_emission_raiser_a"
    assert agg.violations[1].contract_name == "test_pre_emission_raiser_b"


# =============================================================================
# Property tests — POST-EMISSION dispatch surface
# =============================================================================


@example(
    inputs=_seed_post_emission_inputs(),
    outputs=_seed_post_emission_outputs(),
)
@given(  # -> PostEmissionInputs, PostEmissionOutputs
    inputs=post_emission_inputs_strategy(),
    outputs=post_emission_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_post_emission_empty_registry_returns_none(_empty_registry, inputs, outputs):
    """Property A (post-emission) — see pre-emission analogue."""
    assert run_post_emission_checks(inputs, outputs) is None


@example(
    inputs=_seed_post_emission_inputs(),
    outputs=_seed_post_emission_outputs(),
)
@given(  # -> PostEmissionInputs, PostEmissionOutputs
    inputs=post_emission_inputs_strategy(),
    outputs=post_emission_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_post_emission_applicable_non_raising_registry_returns_none(
    _post_emission_never_raises,
    inputs,
    outputs,
):
    """Property A' (post-emission) — see pre-emission analogue."""
    assert run_post_emission_checks(inputs, outputs) is None


@given(  # -> PostEmissionInputs, PostEmissionOutputs
    inputs=post_emission_inputs_strategy(),
    outputs=post_emission_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_post_emission_n1_raises_via_reference_identity(
    _post_emission_single_raiser,
    inputs,
    outputs,
):
    """Property B (post-emission) — see pre-emission analogue."""
    with pytest.raises(_TestViolationA) as exc_info:
        run_post_emission_checks(inputs, outputs)
    assert type(exc_info.value) is _TestViolationA
    assert _PostEmissionRaiserA.last_raised is not None
    assert id(exc_info.value) == id(_PostEmissionRaiserA.last_raised)
    assert not isinstance(exc_info.value, AggregateDeclarationContractViolation)


@given(  # -> PostEmissionInputs, PostEmissionOutputs
    inputs=post_emission_inputs_strategy(),
    outputs=post_emission_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_post_emission_n2_raises_aggregate(
    _post_emission_two_raisers,
    inputs,
    outputs,
):
    """Property C (post-emission) — see pre-emission analogue."""
    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_post_emission_checks(inputs, outputs)
    agg = exc_info.value
    assert len(agg.violations) == 2
    assert isinstance(agg.violations[0], _TestViolationA)
    assert isinstance(agg.violations[1], _TestViolationB)
    assert not isinstance(agg, DeclarationContractViolation)
    assert agg.violations[0].contract_name == "test_post_emission_raiser_a"
    assert agg.violations[1].contract_name == "test_post_emission_raiser_b"


# =============================================================================
# Property tests — BATCH-FLUSH dispatch surface
# =============================================================================


@example(
    inputs=_seed_batch_flush_inputs(),
    outputs=_seed_batch_flush_outputs(),
)
@given(  # -> BatchFlushInputs, BatchFlushOutputs
    inputs=batch_flush_inputs_strategy(),
    outputs=batch_flush_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_batch_flush_empty_registry_returns_none(_empty_registry, inputs, outputs):
    """Property A (batch-flush) — see pre-emission analogue."""
    assert run_batch_flush_checks(inputs, outputs) is None


@example(
    inputs=_seed_batch_flush_inputs(),
    outputs=_seed_batch_flush_outputs(),
)
@given(  # -> BatchFlushInputs, BatchFlushOutputs
    inputs=batch_flush_inputs_strategy(),
    outputs=batch_flush_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_batch_flush_applicable_non_raising_registry_returns_none(
    _batch_flush_never_raises,
    inputs,
    outputs,
):
    """Property A' (batch-flush) — see pre-emission analogue."""
    assert run_batch_flush_checks(inputs, outputs) is None


@given(  # -> BatchFlushInputs, BatchFlushOutputs
    inputs=batch_flush_inputs_strategy(),
    outputs=batch_flush_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_batch_flush_n1_raises_via_reference_identity(
    _batch_flush_single_raiser,
    inputs,
    outputs,
):
    """Property B (batch-flush) — see pre-emission analogue."""
    with pytest.raises(_TestViolationA) as exc_info:
        run_batch_flush_checks(inputs, outputs)
    assert type(exc_info.value) is _TestViolationA
    assert _BatchFlushRaiserA.last_raised is not None
    assert id(exc_info.value) == id(_BatchFlushRaiserA.last_raised)
    assert not isinstance(exc_info.value, AggregateDeclarationContractViolation)


@given(  # -> BatchFlushInputs, BatchFlushOutputs
    inputs=batch_flush_inputs_strategy(),
    outputs=batch_flush_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_batch_flush_n2_raises_aggregate(
    _batch_flush_two_raisers,
    inputs,
    outputs,
):
    """Property C (batch-flush) — see pre-emission analogue."""
    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_batch_flush_checks(inputs, outputs)
    agg = exc_info.value
    assert len(agg.violations) == 2
    assert isinstance(agg.violations[0], _TestViolationA)
    assert isinstance(agg.violations[1], _TestViolationB)
    assert not isinstance(agg, DeclarationContractViolation)
    assert agg.violations[0].contract_name == "test_batch_flush_raiser_a"
    assert agg.violations[1].contract_name == "test_batch_flush_raiser_b"


# =============================================================================
# Property tests — BOUNDARY dispatch surface
# =============================================================================


@example(
    inputs=_seed_boundary_inputs(),
    outputs=_seed_boundary_outputs(),
)
@given(  # -> BoundaryInputs, BoundaryOutputs
    inputs=boundary_inputs_strategy(),
    outputs=boundary_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_boundary_empty_registry_returns_none(_empty_registry, inputs, outputs):
    """Property A (boundary) — see pre-emission analogue.

    Boundary is exercised-but-not-yet-adopted at the H2 landing: the site +
    bundles exist for N1 manifest coverage; the 2C paired adopters land later.
    Property tests here lock the dispatcher shape in before the adopters
    start depending on it.
    """
    assert run_boundary_checks(inputs, outputs) is None


@example(
    inputs=_seed_boundary_inputs(),
    outputs=_seed_boundary_outputs(),
)
@given(  # -> BoundaryInputs, BoundaryOutputs
    inputs=boundary_inputs_strategy(),
    outputs=boundary_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_boundary_applicable_non_raising_registry_returns_none(
    _boundary_never_raises,
    inputs,
    outputs,
):
    """Property A' (boundary) — see pre-emission analogue."""
    assert run_boundary_checks(inputs, outputs) is None


@given(  # -> BoundaryInputs, BoundaryOutputs
    inputs=boundary_inputs_strategy(),
    outputs=boundary_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_boundary_n1_raises_via_reference_identity(
    _boundary_single_raiser,
    inputs,
    outputs,
):
    """Property B (boundary) — see pre-emission analogue."""
    with pytest.raises(_TestViolationA) as exc_info:
        run_boundary_checks(inputs, outputs)
    assert type(exc_info.value) is _TestViolationA
    assert _BoundaryRaiserA.last_raised is not None
    assert id(exc_info.value) == id(_BoundaryRaiserA.last_raised)
    assert not isinstance(exc_info.value, AggregateDeclarationContractViolation)


@given(  # -> BoundaryInputs, BoundaryOutputs
    inputs=boundary_inputs_strategy(),
    outputs=boundary_outputs_strategy(),
)
@_DISPATCH_PROPERTY_SETTINGS
def test_boundary_n2_raises_aggregate(
    _boundary_two_raisers,
    inputs,
    outputs,
):
    """Property C (boundary) — see pre-emission analogue."""
    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_boundary_checks(inputs, outputs)
    agg = exc_info.value
    assert len(agg.violations) == 2
    assert isinstance(agg.violations[0], _TestViolationA)
    assert isinstance(agg.violations[1], _TestViolationB)
    assert not isinstance(agg, DeclarationContractViolation)
    assert agg.violations[0].contract_name == "test_boundary_raiser_a"
    assert agg.violations[1].contract_name == "test_boundary_raiser_b"
