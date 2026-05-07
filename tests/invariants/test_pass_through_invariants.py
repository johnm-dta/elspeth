"""Governance harness for ADR-009 §Clause 4 — pass-through annotation invariants.

Three tests here:

- **Forward invariant** (`test_annotated_transforms_preserve_input_fields`):
  For every registered ``passes_through_input=True`` transform, runs
  Hypothesis-generated probe rows through the transform-owned invariant
  execution hook and asserts every emitted row preserves every input field
  (contract AND payload). Fails CI on mis-annotation.

- **Backward invariant** (``test_non_pass_through_transforms_do_drop_fields``):
  Fails CI when a non-annotated transform that opted into probing (i.e.
  implements ``probe_config()``) preserves all input fields on every probe
  row. Remediation is either adding ``passes_through_input=True`` or
  supplying a ``probe_config()`` that exercises a case the transform
  demonstrably does not preserve.

- **Skip-rate budget** (``test_harness_skip_rate_budget``): asserts
  ``skip_rate ≤ 25%`` across the annotated plugin set. Track 2 additions
  that slip the budget must implement ``probe_config()`` per the contract.

The harness uses ``pytest_generate_tests`` to parametrize over registered
transforms at collection time — with a guard that crashes if the plugin
list is empty (silent "0 tests" passes are the worst kind of theatre).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from hypothesis import HealthCheck, given, settings

from elspeth.contracts.declaration_contracts import (
    DeclarationContract,
    registered_declaration_contracts,
)
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from tests.invariants.conftest import probe_row


def _iter_contracts_for_invariant_harness() -> list[DeclarationContract]:
    """Contracts the harness exercises. One-to-one with the registry today."""
    return list(registered_declaration_contracts())


class _UnprobeableTransform(Exception):
    """Raised when probe_config() is not implemented or the constructor rejects its output."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _registered_transform_classes() -> list[type[BaseTransform]]:
    """Return registered transforms as ``BaseTransform`` subclasses.

    ``PluginManager.get_transforms()`` is typed ``list[type[TransformProtocol]]``
    because the public surface is the protocol. In practice every registered
    plugin subclasses ``BaseTransform`` to inherit the
    ``passes_through_input``, ``is_batch_aware``, and ``probe_config()``
    machinery the harness relies on. The cast documents that framework
    invariant; if a non-``BaseTransform`` plugin were ever registered the
    harness would still typecheck and fail loudly at first use.
    """
    return cast(
        "list[type[BaseTransform]]",
        get_shared_plugin_manager().get_transforms(),
    )


def _annotated_pass_through_plugins() -> list[type[BaseTransform]]:
    """Every registered transform class with ``passes_through_input=True``.

    Reads ``cls.passes_through_input`` directly — no ``getattr`` default.
    Missing attribute is a framework bug (``BaseTransform`` supplies it) and
    a silent ``False`` coercion would hide the annotation from governance.
    """
    return [cls for cls in _registered_transform_classes() if cls.passes_through_input]


def _non_pass_through_plugins() -> list[type[BaseTransform]]:
    """Every registered transform class without pass-through annotation."""
    return [cls for cls in _registered_transform_classes() if not cls.passes_through_input]


def _probe_instantiate(cls: type[BaseTransform]) -> BaseTransform:
    """Build a transform instance via its ``probe_config()`` declaration.

    Narrow exception catches only ``NotImplementedError`` (missing
    implementation — legitimate skip) and ``TypeError`` (wrong constructor
    args — config shape mismatch). Any other exception is a plugin bug and
    must propagate (CLAUDE.md: plugin bugs must crash).
    """
    try:
        config = cls.probe_config()
    except NotImplementedError as exc:
        if cls.passes_through_input:
            reason = f"{cls.__name__}.probe_config() not implemented: {exc}"
        else:
            reason = f"{cls.__name__}.probe_config() not implemented (non-pass-through transform has not opted into invariant probing)."
        raise _UnprobeableTransform(reason=reason) from exc
    try:
        return cls(config)
    except TypeError as exc:
        raise _UnprobeableTransform(reason=f"{cls.__name__}.__init__ rejected probe_config() output: {exc}") from exc


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamic parametrization — resolves after plugin registration fixtures run.

    Using a fixture-less ``pytest.mark.parametrize`` at module scope would
    evaluate ``_annotated_pass_through_plugins()`` at collection time,
    before plugin registration side effects have fired. The guard here
    crashes loudly if the list is empty.
    """
    if "_annotated_cls" in metafunc.fixturenames:
        plugins = _annotated_pass_through_plugins()
        assert plugins, (
            "Expected at least 1 passes_through_input=True transform; found "
            f"{[cls.__name__ for cls in plugins]!r}. Plugin registration "
            "may have failed — the invariant harness would silently pass."
        )
        metafunc.parametrize("_annotated_cls", plugins, ids=lambda c: c.__name__)

    if "_non_pass_through_cls" in metafunc.fixturenames:
        plugins = _non_pass_through_plugins()
        metafunc.parametrize("_non_pass_through_cls", plugins, ids=lambda c: c.__name__)


def _probe_context(transform: BaseTransform) -> Any:
    """Minimal TransformContext for probe invocations.

    The harness runs transforms in isolation without a real run; the context
    is a lightweight stub with a mock landscape recorder.
    """
    from tests.fixtures.factories import make_context

    return make_context()


def _emitted_rows_from_result(result: Any) -> list[PipelineRow]:
    if result.row is not None:
        return [result.row]
    if result.rows is not None:
        return list(result.rows)
    return []


def _observed_fields(row: PipelineRow) -> frozenset[str]:
    """Fields present in both the contract and payload for ``row``."""
    contract_fields = frozenset(fc.normalized_name for fc in row.contract.fields)
    payload_fields = frozenset(row.keys())
    return contract_fields & payload_fields


def _effective_input_fields(probe_rows: list[PipelineRow]) -> frozenset[str]:
    """Mirror runtime pass-through input-field semantics for probe rows."""
    if not probe_rows:
        return frozenset()
    observed_sets = [_observed_fields(row) for row in probe_rows]
    effective = observed_sets[0]
    for observed in observed_sets[1:]:
        effective = effective & observed
    return effective


@given(row=probe_row())
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_annotated_transforms_preserve_input_fields(
    _annotated_cls: type[BaseTransform],
    row: PipelineRow,
) -> None:
    """Forward invariant — ADR-009 §Clause 4.

    Every emitted row from a ``passes_through_input=True`` transform must
    preserve every input field in both its contract and its payload
    (runtime observation is the intersection of the two).

    Failures are actionable: Hypothesis shrinks to a minimal probe row,
    producing the smallest repro case for the plugin author. The remediation
    is clear — either fix the implementation or remove the annotation.
    """
    try:
        transform = _probe_instantiate(_annotated_cls)
    except _UnprobeableTransform as exc:
        pytest.skip(f"{_annotated_cls.__name__}: {exc.reason}")

    probe_rows = transform.forward_invariant_probe_rows(row)

    result = transform.execute_forward_invariant_probe(
        probe_rows,
        _probe_context(transform),
    )

    if result.status != "success":
        # Legitimate processing error on this probe (e.g., quarantine). Not
        # a pass-through contract violation.
        return

    emitted_rows = _emitted_rows_from_result(result)
    if not emitted_rows:
        # Empty emission — ADR-009 §Clause 3 carve-out. Drops nothing.
        return

    input_fields = _effective_input_fields(probe_rows)
    for emitted in emitted_rows:
        runtime_contract = frozenset(fc.normalized_name for fc in emitted.contract.fields)
        runtime_payload = frozenset(emitted.keys())
        runtime_observed = runtime_contract & runtime_payload
        dropped = input_fields - runtime_observed
        assert not dropped, (
            f"{_annotated_cls.__name__} is annotated passes_through_input=True "
            f"but dropped fields {sorted(dropped)!r} from probe row "
            f"{row.to_dict()!r}. Either fix the implementation or remove "
            "the annotation."
        )


def test_harness_skip_rate_budget() -> None:
    """Skip-rate budget — ADR-009 §Clause 4.

    Assert ``skip_rate ≤ 25%`` across the annotated plugin set. Track 2
    additions that can't be probed in isolation must implement
    ``probe_config()`` per the contract; raising the budget is not an
    acceptable response.
    """
    transforms = _annotated_pass_through_plugins()
    if not transforms:
        pytest.skip("No annotated transforms registered.")

    unprobeable: list[str] = []
    for cls in transforms:
        try:
            _probe_instantiate(cls)
        except _UnprobeableTransform as exc:
            unprobeable.append(f"{cls.__name__}: {exc.reason}")

    skip_rate = len(unprobeable) / len(transforms)
    assert skip_rate <= 0.25, (
        f"Harness skip rate {skip_rate:.0%} exceeds 25% budget "
        f"({len(unprobeable)}/{len(transforms)} annotated transforms unprobeable). "
        f"Implement probe_config() on: {unprobeable!r}"
    )


# Backward-invariant sweep budget. Scalar-only probes — bounded to keep
# invariant runs fast; the per-transform forward invariant carries the
# correctness load, this one is a sanity check on non-annotated probeable
# transforms. ``_SWEEP_MIN_PROBES`` guards against Hypothesis strategy
# exhaustion masquerading as clean runs.
_SWEEP_EXAMPLES = 15
_SWEEP_MIN_PROBES = 5


def test_non_pass_through_transforms_do_drop_fields(
    _non_pass_through_cls: type[BaseTransform],
) -> None:
    """Backward invariant — ADR-009 §Clause 4.

    For every non-annotated transform that opted into probing (i.e.,
    implements ``probe_config()``), run ``_SWEEP_EXAMPLES`` probe rows and
    assert at least one probe produces an emission that drops a field. A
    transform that preserves all fields on every probe is either
    mis-annotated (should carry ``passes_through_input=True``) or its
    ``probe_config()`` does not exercise a case where fields are dropped
    — both are governance defects that must be addressed in this PR.

    Non-annotated transforms WITHOUT ``probe_config()`` are skipped with a
    diagnostic reason:
    probing is opt-in, and the forward invariant + skip-rate budget are the
    load-bearing governance for annotated transforms. Transforms whose
    ``probe_config()`` raises or whose constructor rejects it are also
    skipped — those are diagnostic signals, not governance gates.

    Scalar-only probes may miss structured-data or mixed-validity batch
    drops, so remediation options are either (a) add the annotation if the
    transform really is pass-through, or (b) override
    ``backward_invariant_probe_rows()`` to return a representative input
    shape that triggers the actual drop path.
    """
    try:
        transform = _probe_instantiate(_non_pass_through_cls)
    except _UnprobeableTransform as exc:
        # Non-annotated transforms that did not opt into probing — or whose
        # probe config is incompatible with their constructor — are out of
        # scope for the backward invariant. The forward invariant and
        # skip-rate budget cover the annotated set; this test only
        # exercises transforms that explicitly declared a probe config.
        pytest.skip(f"{_non_pass_through_cls.__name__}: {exc.reason}")

    probes_preserved = True
    probe_count = 0

    @given(probe=probe_row())
    @settings(
        max_examples=_SWEEP_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def _sweep(probe: PipelineRow) -> None:
        nonlocal probes_preserved, probe_count
        probe_count += 1
        probe_rows = transform.backward_invariant_probe_rows(probe)
        result = transform.execute_backward_invariant_probe(
            probe_rows,
            _probe_context(transform),
        )
        if result.status != "success":
            return
        emitted_rows = _emitted_rows_from_result(result)
        if not emitted_rows:
            return
        input_fields = frozenset(field_name for input_row in probe_rows for field_name in _observed_fields(input_row))
        for emitted in emitted_rows:
            observed = _observed_fields(emitted)
            if input_fields - observed:
                probes_preserved = False
                return

    _sweep()

    if probe_count < _SWEEP_MIN_PROBES:
        # Strategy exhaustion is a harness failure, not a plugin failure —
        # make the operator aware without blaming the transform.
        pytest.fail(
            f"{_non_pass_through_cls.__name__}: only {probe_count} probe rows "
            f"exercised (expected ≥ {_SWEEP_MIN_PROBES}). Harness probe generation "
            "is under-powered for this transform."
        )

    if probes_preserved:
        pytest.fail(
            f"{_non_pass_through_cls.__name__} is NOT annotated "
            f"passes_through_input=True but preserved every input field in "
            f"{probe_count} probe rows. Either (a) add passes_through_input=True "
            "if the transform is in fact pass-through, or (b) override "
            f"{_non_pass_through_cls.__name__}.backward_invariant_probe_rows() "
            "to return a shape that triggers the field-dropping code path."
        )
