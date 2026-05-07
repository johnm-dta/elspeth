"""Declaration-contract dispatcher tests (ADR-010 §Decision 3 + §Semantics).

Exercises the audit-complete collect-then-raise dispatcher introduced by the
H2 cluster landing (2026-04-20). The dispatcher at
``src/elspeth/engine/executors/declaration_dispatch.py`` iterates every
applicable contract for a given dispatch site, collects raised violations
rather than short-circuiting on first-fire, and at loop end:

  * 0 violations → returns normally.
  * 1 violation  → raises ``violations[0]`` via reference equality
                   (N6 regression test asserts identity).
  * >=2 violations → wraps in ``AggregateDeclarationContractViolation``.

Plugin-bug exceptions (non-audit-evidence ``RuntimeError`` / ``KeyError``)
propagate unmodified per CLAUDE.md plugin-ownership posture.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    DeclarationContract,
    DeclarationContractViolation,
    DispatchSite,
    ExampleBundle,
    PostEmissionInputs,
    PostEmissionOutputs,
    _clear_registry_for_tests,
    _mark_aggregate_dispatched,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.engine.executors.declaration_dispatch import _serialize_plugin_name, run_post_emission_checks


class _Payload(TypedDict):
    note: str


class _TestViolationA(DeclarationContractViolation):
    payload_schema = _Payload


class _TestViolationB(DeclarationContractViolation):
    payload_schema = _Payload


def _inputs() -> PostEmissionInputs:
    plugin = type("NamedPlugin", (), {})()
    plugin.name = "named-plugin"
    return PostEmissionInputs(
        plugin=plugin,
        node_id="n",
        run_id="r",
        row_id="rw",
        token_id="t",
        input_row=object(),
        static_contract=frozenset(),
        effective_input_fields=frozenset(),
    )


def _outputs() -> PostEmissionOutputs:
    return PostEmissionOutputs(emitted_rows=(object(),))


def _empty_example_bundle() -> ExampleBundle:
    """Placeholder bundle used by contracts whose tests exercise dispatcher
    routing, not harness coverage. The registry requires both example
    classmethods to be callable; returning a well-typed bundle satisfies
    registration without the harness being invoked in this scope."""
    return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(_inputs(), _outputs()))


class _AppliesContract(DeclarationContract):
    name = "applies"
    payload_schema: type = _Payload
    invoked: bool = False

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        _AppliesContract.invoked = True

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _empty_example_bundle()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _empty_example_bundle()


class _SkipsContract(DeclarationContract):
    name = "skips"
    payload_schema: type = _Payload
    invoked: bool = False

    def applies_to(self, plugin: Any) -> bool:
        return False

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        _SkipsContract.invoked = True

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _empty_example_bundle()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _empty_example_bundle()


class _RaisesViolationContract(DeclarationContract):
    name = "raises_violation"
    payload_schema: type = _Payload

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        raise _TestViolationA(
            plugin="P",
            node_id="n",
            run_id="r",
            row_id="rw",
            token_id="t",
            payload={"note": "boom"},
            message="boom",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _empty_example_bundle()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _empty_example_bundle()


class _RaisesSecondViolationContract(DeclarationContract):
    """Second violation-raising contract for aggregate coverage."""

    name = "raises_second_violation"
    payload_schema: type = _Payload

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        raise _TestViolationB(
            plugin="P",
            node_id="n",
            run_id="r",
            row_id="rw",
            token_id="t",
            payload={"note": "second"},
            message="second violation",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _empty_example_bundle()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _empty_example_bundle()


class _ApplyRaisesContract(DeclarationContract):
    """Simulates a buggy contract whose applies_to raises — must propagate
    unmodified per CLAUDE.md plugin-ownership posture."""

    name = "apply_raises"
    payload_schema: type = _Payload

    def applies_to(self, plugin: Any) -> bool:
        raise KeyError("bug in applies_to")

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None: ...

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _empty_example_bundle()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _empty_example_bundle()


class _CheckRaisesContract(DeclarationContract):
    """Contract whose dispatch method raises a generic RuntimeError —
    propagates unmodified (not caught by audit-complete collection)."""

    name = "check_raises"
    payload_schema: type = _Payload

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        raise RuntimeError("bug in post_emission_check")

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return _empty_example_bundle()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return _empty_example_bundle()


# -----------------------------------------------------------------------------
# Fixture — registry isolation
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate():
    snapshot = _snapshot_registry_for_tests()
    _clear_registry_for_tests()
    _AppliesContract.invoked = False
    _SkipsContract.invoked = False
    yield
    _restore_registry_snapshot_for_tests(snapshot)


# =============================================================================
# Basic routing
# =============================================================================


def test_dispatch_invokes_applicable_contracts() -> None:
    register_declaration_contract(_AppliesContract())
    register_declaration_contract(_SkipsContract())
    run_post_emission_checks(inputs=_inputs(), outputs=_outputs())
    assert _AppliesContract.invoked and not _SkipsContract.invoked


def test_dispatch_propagates_violation() -> None:
    register_declaration_contract(_RaisesViolationContract())
    with pytest.raises(DeclarationContractViolation):
        run_post_emission_checks(inputs=_inputs(), outputs=_outputs())


def test_dispatch_propagates_unexpected_exception_from_applies_to() -> None:
    """Reviewer B17 / CLAUDE.md plugin-ownership posture: a buggy contract
    must crash loudly. Non-audit-evidence exceptions from ``applies_to``
    propagate unmodified — they are NOT caught by the audit-complete
    collection branch."""
    register_declaration_contract(_ApplyRaisesContract())
    with pytest.raises(KeyError, match="applies_to"):
        run_post_emission_checks(inputs=_inputs(), outputs=_outputs())


def test_dispatch_propagates_unexpected_exception_from_dispatch_method() -> None:
    register_declaration_contract(_CheckRaisesContract())
    with pytest.raises(RuntimeError, match="post_emission_check"):
        run_post_emission_checks(inputs=_inputs(), outputs=_outputs())


def test_empty_registry_still_runs() -> None:
    """Bootstrap (prepare_for_run) asserts the registry is non-empty; the
    dispatcher itself does NOT — it is pure iteration, no-op on empty."""
    run_post_emission_checks(inputs=_inputs(), outputs=_outputs())


# =============================================================================
# Audit-complete collection (N3 primary mitigation)
# =============================================================================


def test_single_violation_raises_via_reference_identity() -> None:
    """N6 regression invariant (N3 §Acceptance): at N=1 the dispatcher
    raises the original violation via ``raise violations[0]`` — not an
    aggregation-of-one wrapper.

    Concretely: ``type(raised)`` is the concrete DCV subclass (not
    ``AggregateDeclarationContractViolation``) AND ``id(raised)`` equals
    the id of the exception the contract raised — proving the fast-path
    reference-equality branch was taken.
    """

    class _CapturingContract(DeclarationContract):
        name = "capturing"
        payload_schema: type = _Payload
        last_raised: DeclarationContractViolation | None = None

        def applies_to(self, plugin: Any) -> bool:
            return True

        @implements_dispatch_site("post_emission_check")
        def post_emission_check(
            self,
            inputs: PostEmissionInputs,
            outputs: PostEmissionOutputs,
        ) -> None:
            exc = _TestViolationA(
                plugin="P",
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                payload={"note": "capture"},
                message="captured for identity check",
            )
            type(self).last_raised = exc
            raise exc

        @classmethod
        def negative_example(cls) -> ExampleBundle:
            return _empty_example_bundle()

        @classmethod
        def positive_example_does_not_apply(cls) -> ExampleBundle:
            return _empty_example_bundle()

    register_declaration_contract(_CapturingContract())

    with pytest.raises(DeclarationContractViolation) as exc_info:
        run_post_emission_checks(inputs=_inputs(), outputs=_outputs())

    raised = exc_info.value
    # Type equality: MUST be the concrete subclass, NOT the aggregate.
    assert type(raised) is _TestViolationA, (
        f"N6 regression: expected _TestViolationA, got {type(raised).__name__}. "
        f"The dispatcher must take the reference-equality fast path at N=1, "
        f"not wrap the single violation in AggregateDeclarationContractViolation."
    )
    # Reference equality: the raised object IS the contract-raised object,
    # not a copy or aggregation-of-one wrapper.
    assert id(raised) == id(_CapturingContract.last_raised), (
        "N6 regression: dispatcher MUST raise violations[0] via reference "
        "equality at N=1. An object-identity mismatch means the dispatcher "
        "constructed a new exception around the original — which would "
        "break triage SQL filtering by exception_type."
    )


def test_multiple_violations_wrapped_in_aggregate() -> None:
    """N3 §Acceptance: at N>=2 applicable contracts that fire, the
    dispatcher collects every violation and raises an aggregate."""
    register_declaration_contract(_RaisesViolationContract())
    register_declaration_contract(_RaisesSecondViolationContract())

    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_post_emission_checks(inputs=_inputs(), outputs=_outputs())

    aggregate = exc_info.value
    assert len(aggregate.violations) == 2
    exception_types = {type(v).__name__ for v in aggregate.violations}
    assert exception_types == {"_TestViolationA", "_TestViolationB"}

    # Each child was attributed by the dispatcher.
    for child in aggregate.violations:
        if isinstance(child, DeclarationContractViolation):
            assert child.contract_name in {"raises_violation", "raises_second_violation"}


def test_serialize_plugin_name_raises_when_plugin_name_missing() -> None:
    class _NamelessPlugin:
        pass

    with pytest.raises(AttributeError):
        _serialize_plugin_name(_NamelessPlugin())


def test_serialize_plugin_name_raises_when_plugin_name_empty() -> None:
    from elspeth.contracts.errors import FrameworkBugError

    class _EmptyNamePlugin:
        name = ""

    with pytest.raises(FrameworkBugError, match=r"empty plugin\.name"):
        _serialize_plugin_name(_EmptyNamePlugin())


def test_aggregate_to_audit_dict_carries_children() -> None:
    """Aggregate's to_audit_dict emits is_aggregate=True and the full
    violations list — no contract_name field (C5 closure)."""
    register_declaration_contract(_RaisesViolationContract())
    register_declaration_contract(_RaisesSecondViolationContract())

    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_post_emission_checks(inputs=_inputs(), outputs=_outputs())

    audit = exc_info.value.to_audit_dict()
    assert audit["is_aggregate"] is True
    assert audit["exception_type"] == "AggregateDeclarationContractViolation"
    assert "contract_name" not in audit  # sentinel-in-name rejected (S2-001)
    assert len(audit["violations"]) == 2
    child_types = {v["exception_type"] for v in audit["violations"]}
    assert child_types == {"_TestViolationA", "_TestViolationB"}


def test_aggregate_attribution_one_shot() -> None:
    """C5 mirror of C4 closure: _attach_by_dispatcher refuses a second call."""
    aggregate = AggregateDeclarationContractViolation(
        plugin="P",
        violations=(
            _TestViolationA(
                plugin="P",
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                payload={"note": "a"},
                message="a",
            ),
            _TestViolationB(
                plugin="P",
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                payload={"note": "b"},
                message="b",
            ),
        ),
        message="2 violations",
    )
    _mark_aggregate_dispatched(aggregate)
    with pytest.raises(RuntimeError, match="twice"):
        _mark_aggregate_dispatched(aggregate)


def test_aggregate_attribution_requires_dispatcher_helper() -> None:
    aggregate = AggregateDeclarationContractViolation(
        plugin="P",
        violations=(
            _TestViolationA(
                plugin="P",
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                payload={"note": "a"},
                message="a",
            ),
            _TestViolationB(
                plugin="P",
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                payload={"note": "b"},
                message="b",
            ),
        ),
        message="2 violations",
    )
    with pytest.raises(RuntimeError, match="dispatcher"):
        aggregate._attach_by_dispatcher()


def test_aggregate_to_audit_dict_without_dispatcher_attribution_raises() -> None:
    """C5: reading to_audit_dict before dispatcher attribution raises.

    Mirrors the C4 guard on DCV.contract_name — prevents a stray raise
    path from bypassing the audit-complete dispatcher.
    """
    aggregate = AggregateDeclarationContractViolation(
        plugin="P",
        violations=(
            _TestViolationA(
                plugin="P",
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                payload={"note": "a"},
                message="a",
            ),
            _TestViolationB(
                plugin="P",
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                payload={"note": "b"},
                message="b",
            ),
        ),
        message="2 violations",
    )
    # _attach_by_dispatcher not called — to_audit_dict must refuse.
    with pytest.raises(RuntimeError, match="before dispatcher attribution"):
        aggregate.to_audit_dict()


def test_aggregate_rejects_fewer_than_two_violations() -> None:
    """Single violation cases MUST take the reference-equality fast path
    (N6 regression). Constructing the aggregate with <2 violations is a
    framework bug — rejected at __init__."""
    child = _TestViolationA(
        plugin="P",
        node_id="n",
        run_id="r",
        row_id="rw",
        token_id="t",
        payload={"note": "solo"},
        message="solo",
    )
    with pytest.raises(ValueError, match="at least 2 violations"):
        AggregateDeclarationContractViolation(
            plugin="P",
            violations=(child,),
            message="should not reach here",
        )


def test_audit_complete_runs_every_applicable_contract() -> None:
    """Even when the first contract raises, subsequent contracts still run
    — that's the whole point of audit-complete vs first-fire semantics."""

    second_invoked = False

    class _TracingContract(DeclarationContract):
        name = "tracing_second"
        payload_schema: type = _Payload

        def applies_to(self, plugin: Any) -> bool:
            return True

        @implements_dispatch_site("post_emission_check")
        def post_emission_check(
            self,
            inputs: PostEmissionInputs,
            outputs: PostEmissionOutputs,
        ) -> None:
            nonlocal second_invoked
            second_invoked = True
            raise _TestViolationB(
                plugin="P",
                node_id="n",
                run_id="r",
                row_id="rw",
                token_id="t",
                payload={"note": "traced"},
                message="second contract did run",
            )

        @classmethod
        def negative_example(cls) -> ExampleBundle:
            return _empty_example_bundle()

        @classmethod
        def positive_example_does_not_apply(cls) -> ExampleBundle:
            return _empty_example_bundle()

    register_declaration_contract(_RaisesViolationContract())
    register_declaration_contract(_TracingContract())

    with pytest.raises(AggregateDeclarationContractViolation):
        run_post_emission_checks(inputs=_inputs(), outputs=_outputs())

    assert second_invoked, (
        "Audit-complete invariant violated: the second contract did NOT run "
        "after the first contract raised. This regresses to fail-fast "
        "first-fire semantics — ADR-010 §Semantics amendment 2026-04-20 "
        "requires every applicable contract's method to execute."
    )


# =============================================================================
# Per-site filtering (the decorator marker controls which site fires)
# =============================================================================


def test_contract_without_post_emission_marker_not_invoked_by_post_emission_dispatch() -> None:
    """A contract claiming only batch_flush_check MUST NOT be invoked by
    the post_emission dispatcher — the per-site registry filter enforces
    this."""
    invoked = False

    class _BatchOnlyContract(DeclarationContract):
        name = "batch_only"
        payload_schema: type = _Payload

        def applies_to(self, plugin: Any) -> bool:
            nonlocal invoked
            invoked = True  # should NEVER fire under post-emission dispatch
            return True

        @implements_dispatch_site("batch_flush_check")
        def batch_flush_check(
            self,
            inputs: Any,
            outputs: Any,
        ) -> None:
            return None

        @classmethod
        def negative_example(cls) -> ExampleBundle:
            return _empty_example_bundle()

        @classmethod
        def positive_example_does_not_apply(cls) -> ExampleBundle:
            return _empty_example_bundle()

    register_declaration_contract(_BatchOnlyContract())
    run_post_emission_checks(inputs=_inputs(), outputs=_outputs())
    assert not invoked, (
        "Per-site registry filter failed: a contract marked only for batch_flush_check was invoked by the post-emission dispatcher."
    )
