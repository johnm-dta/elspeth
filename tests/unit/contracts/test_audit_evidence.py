"""AuditEvidenceBase tests (ADR-010 §Decision 1).

Nominal base class: classes must inherit AuditEvidenceBase to contribute
structured audit context. Structural duck-typing against to_audit_dict is
REJECTED by design — see ADR-010 §Alternative 3 for the security rationale
(accidental-match spoofing of the audit trail).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from elspeth.contracts.audit_evidence import AuditEvidenceBase
from elspeth.contracts.errors import PassThroughContractViolation, PluginContractViolation


def test_plugin_contract_violation_is_audit_evidence() -> None:
    # PluginContractViolation inherits AuditEvidenceBase (Task 4 migration).
    err = PluginContractViolation("hello")
    assert isinstance(err, AuditEvidenceBase)


def test_pass_through_violation_is_audit_evidence() -> None:
    err = PassThroughContractViolation(
        transform="x",
        transform_node_id="n",
        run_id="r",
        row_id="rw",
        token_id="t",
        static_contract=frozenset(),
        runtime_observed=frozenset(),
        divergence_set=frozenset({"a"}),
        message="m",
    )
    assert isinstance(err, AuditEvidenceBase)


def test_duck_typed_class_is_NOT_audit_evidence() -> None:
    """Nominal-only: having to_audit_dict is not enough — you must inherit."""

    class _Mimic(RuntimeError):
        def to_audit_dict(self) -> Mapping[str, Any]:
            return {"k": "v"}

    assert not isinstance(_Mimic("x"), AuditEvidenceBase)


def test_subclass_must_implement_to_audit_dict() -> None:
    """Abstract enforcement: subclass without to_audit_dict cannot be instantiated."""

    class _Incomplete(AuditEvidenceBase, RuntimeError):
        pass

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete("x")


def test_subclass_cannot_bypass_abstract_check_with_non_cooperative_init() -> None:
    """Direct ``RuntimeError.__init__`` calls must not bypass abstract enforcement."""

    class _Incomplete(AuditEvidenceBase, RuntimeError):
        def __init__(self, message: str) -> None:
            RuntimeError.__init__(self, message)

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete("x")


def test_subclass_cannot_bypass_abstract_check_with_reversed_base_order() -> None:
    """Reversing the MRO must not bypass abstract enforcement."""

    class _Incomplete(RuntimeError, AuditEvidenceBase):
        pass

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete("x")


def test_dataclass_subclass_cannot_overwrite_abstract_guard() -> None:
    """A dataclass-generated ``__init__`` must not bypass abstract enforcement."""

    @dataclass
    class _Incomplete(AuditEvidenceBase, RuntimeError):
        message: str

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete("x")


def test_base_class_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AuditEvidenceBase()  # type: ignore[abstract]
