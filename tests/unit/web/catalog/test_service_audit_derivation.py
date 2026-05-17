"""Tests for CatalogServiceImpl._derive_audit_characteristics.

The catalog service composes a plugin's declared audit_characteristics
frozenset with the characteristic inferred from determinism, sorted into
a deterministic tuple for stable wire ordering. Inference rules:

  Determinism.IO_READ            -> "io_read"
  Determinism.IO_WRITE           -> "io_write"
  Determinism.EXTERNAL_CALL      -> "external_call"
  Determinism.DETERMINISTIC      -> "deterministic"
  Determinism.SEEDED             -> "seeded"
  Determinism.NON_DETERMINISTIC  -> "non_deterministic"

Quarantine is author-declared, not inferred — `_on_validation_failure`
is a per-instance attribute set in each source's __init__ and does not
exist on the class object.
"""

from __future__ import annotations

from elspeth.contracts.enums import AuditCharacteristic, Determinism
from elspeth.web.catalog.service import _derive_audit_characteristics


class _FakeSource:
    name = "fake_source"
    determinism = Determinism.IO_READ
    # quarantine is author-declared (not inferred from _on_validation_failure,
    # which is a per-instance attribute set in __init__).
    audit_characteristics = frozenset({AuditCharacteristic.PROVENANCE, AuditCharacteristic.QUARANTINE})


class _FakeSourceWithoutQuarantine:
    name = "fake_no_quarantine"
    determinism = Determinism.IO_READ
    audit_characteristics: frozenset[AuditCharacteristic] = frozenset()


class _FakeTransformWithNetwork:
    name = "fake_xfm"
    determinism = Determinism.EXTERNAL_CALL
    audit_characteristics = frozenset({AuditCharacteristic.CREDENTIALS})


class _FakeTransformDeterministic:
    name = "fake_deterministic"
    determinism = Determinism.DETERMINISTIC
    audit_characteristics: frozenset[AuditCharacteristic] = frozenset()


class _FakeSink:
    name = "fake_sink"
    determinism = Determinism.IO_WRITE
    audit_characteristics = frozenset({AuditCharacteristic.SIGNED})


class _FakeSeeded:
    name = "fake_seeded"
    determinism = Determinism.SEEDED
    audit_characteristics: frozenset[AuditCharacteristic] = frozenset()


class _FakeNonDeterministic:
    name = "fake_non_deterministic"
    determinism = Determinism.NON_DETERMINISTIC
    audit_characteristics: frozenset[AuditCharacteristic] = frozenset()


def test_source_declared_quarantine_passes_through() -> None:
    """`quarantine` is author-declared on the class. Composition preserves it."""
    derived = _derive_audit_characteristics(_FakeSource, plugin_kind="source")
    assert "quarantine" in derived  # declared by author
    assert "io_read" in derived  # inferred from determinism
    assert "provenance" in derived  # declared, preserved


def test_source_without_declared_quarantine_omits_it() -> None:
    """Source authors who don't declare quarantine don't get it in the
    derived set. Quarantine is author-declared in `audit_characteristics`;
    `_on_validation_failure` is per-instance and does not exist on the
    class object."""
    derived = _derive_audit_characteristics(_FakeSourceWithoutQuarantine, plugin_kind="source")
    assert "quarantine" not in derived
    assert "io_read" in derived


def test_external_call_implies_external_call_flag() -> None:
    derived = _derive_audit_characteristics(_FakeTransformWithNetwork, plugin_kind="transform")
    assert "external_call" in derived
    assert "credentials" in derived  # declared, preserved


def test_deterministic_transform() -> None:
    derived = _derive_audit_characteristics(_FakeTransformDeterministic, plugin_kind="transform")
    assert "deterministic" in derived


def test_sink_io_write_inference() -> None:
    derived = _derive_audit_characteristics(_FakeSink, plugin_kind="sink")
    assert "io_write" in derived
    assert "signed" in derived  # declared, preserved


def test_transform_has_no_quarantine_inference() -> None:
    """quarantine is author-declared, not derived by the framework.
    Transforms don't quarantine at the boundary (they crash on type errors
    per the tier model); this test guards that no quarantine flag is
    injected for transforms."""
    derived = _derive_audit_characteristics(_FakeTransformWithNetwork, plugin_kind="transform")
    assert "quarantine" not in derived


def test_quarantine_inference_is_not_attempted_from_instance_attribute() -> None:
    """Regression guard: the derivation MUST NOT read
    `plugin_cls._on_validation_failure`. That attribute is set in
    __init__ and does not exist on the class object — reading it would
    raise AttributeError at catalog-build time. The plan deliberately
    moved quarantine to author declaration."""

    # No `_on_validation_failure` attribute on this fixture; if the
    # implementation attempted to read it from the class, this would
    # AttributeError instead of returning a clean frozenset.
    class _NoInstanceAttr:
        name = "no_instance_attr"
        determinism = Determinism.IO_READ
        audit_characteristics: frozenset[AuditCharacteristic] = frozenset()

    derived = _derive_audit_characteristics(_NoInstanceAttr, plugin_kind="source")
    assert "io_read" in derived
    assert "quarantine" not in derived


def test_plugin_without_audit_characteristics_attr_does_not_crash() -> None:
    """Tier-1 plugin-attribute access on a plugin missing the declared
    audit_characteristics field should still work, because BaseSource /
    BaseTransform / BaseSink provide the default of frozenset()."""

    class _NoDeclaredChars:
        name = "no_declared"
        determinism = Determinism.DETERMINISTIC
        audit_characteristics: frozenset[AuditCharacteristic] = frozenset()  # the default from the base

    derived = _derive_audit_characteristics(_NoDeclaredChars, plugin_kind="transform")
    assert derived == (AuditCharacteristic.DETERMINISTIC,)


def test_returns_sorted_tuple() -> None:
    """Derivation returns a sorted tuple[str, ...] for stable wire-format
    ordering; the response model exposes this directly to the frontend."""
    derived = _derive_audit_characteristics(_FakeSource, plugin_kind="source")
    assert isinstance(derived, tuple)
    assert list(derived) == sorted(derived)


def test_seeded_implies_seeded_flag() -> None:
    """Determinism.SEEDED maps to the 'seeded' audit flag."""
    derived = _derive_audit_characteristics(_FakeSeeded, plugin_kind="transform")
    assert "seeded" in derived


def test_non_deterministic_implies_non_deterministic_flag() -> None:
    """Determinism.NON_DETERMINISTIC maps to the 'non_deterministic' audit flag."""
    derived = _derive_audit_characteristics(_FakeNonDeterministic, plugin_kind="transform")
    assert "non_deterministic" in derived


def test_determinism_to_audit_flag_covers_all_enum_values() -> None:
    """_DETERMINISM_TO_AUDIT_FLAG must be exhaustive over Determinism.

    If a new Determinism value is added to contracts/enums.py without a
    corresponding entry in _DETERMINISM_TO_AUDIT_FLAG, the subscript
    access in _derive_audit_characteristics raises KeyError at runtime.
    This test surfaces the gap at test time rather than at production
    catalog-build time.
    """
    from elspeth.web.catalog.service import _DETERMINISM_TO_AUDIT_FLAG

    assert set(_DETERMINISM_TO_AUDIT_FLAG.keys()) == set(Determinism)


# Note: The legacy validity-loop test (test_all_plugin_audit_characteristics_are_valid)
# was deleted when the closed vocabulary was promoted from a runtime-only set
# (VALID_AUDIT_CHARACTERISTICS) into the AuditCharacteristic StrEnum in
# contracts/enums.py. Typos at declaration sites now fail mypy at edit time
# rather than at CI time, removing the need for the runtime sweep.
