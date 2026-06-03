"""Tests for CatalogServiceImpl._derive_audit_characteristics.

The catalog service composes a plugin's declared `audit_characteristics`
frozenset with the characteristic inferred from `determinism`, sorted
into a deterministic tuple for stable wire ordering.

Inference is **suppressed** when the plugin's determinism matches the
kind default (BaseSource → IO_READ, BaseTransform → DETERMINISTIC,
BaseSink → IO_WRITE). Surfacing a flag that's true for every plugin of
a kind teaches the user nothing about that plugin specifically and
fails the "each tag must represent a meaningful per-plugin decision"
test. When the author **overrides** the kind default the override is a
deliberate audit-relevant claim and is surfaced.

  Override IO_READ        -> "io_read"          (Sources only; rare)
  Override IO_WRITE       -> "io_write"         (Sinks only; rare)
  Override EXTERNAL_CALL  -> "external_call"
  Override DETERMINISTIC  -> "deterministic"    (Sources/Sinks)
  Override SEEDED         -> "seeded"
  Override NON_DETERMINISTIC -> "non_deterministic"

Quarantine is author-declared, not inferred — `_on_validation_failure`
is a per-instance attribute set in each source's __init__ and does not
exist on the class object.
"""

from __future__ import annotations

from elspeth.contracts.enums import AuditCharacteristic, Determinism
from elspeth.web.catalog.service import _derive_audit_characteristics

# ── Source fakes ────────────────────────────────────────────────────


class _SourceKindDefault:
    """A source inheriting the kind default (Determinism.IO_READ).

    Authors who don't override determinism contribute no derived flag;
    only their declared characteristics appear on the card.
    """

    name = "source_default"
    determinism = Determinism.IO_READ
    audit_characteristics = frozenset({AuditCharacteristic.PROVENANCE, AuditCharacteristic.QUARANTINE})


class _SourceKindDefaultNoDeclared:
    name = "source_default_bare"
    determinism = Determinism.IO_READ
    audit_characteristics: frozenset[AuditCharacteristic] = frozenset()


class _SourceOverriddenToDeterministic:
    """A source that overrides to DETERMINISTIC (e.g. NullSource pattern).

    Overriding the kind default IS a deliberate author claim, so the
    derived `deterministic` flag IS emitted.
    """

    name = "source_overridden"
    determinism = Determinism.DETERMINISTIC
    audit_characteristics: frozenset[AuditCharacteristic] = frozenset()


# ── Transform fakes ─────────────────────────────────────────────────


class _TransformKindDefault:
    """A transform inheriting Determinism.DETERMINISTIC — pure transform."""

    name = "xfm_default"
    determinism = Determinism.DETERMINISTIC
    audit_characteristics: frozenset[AuditCharacteristic] = frozenset()


class _TransformExternalCall:
    name = "xfm_external"
    determinism = Determinism.EXTERNAL_CALL
    audit_characteristics = frozenset({AuditCharacteristic.CREDENTIALS})


class _TransformSeeded:
    name = "xfm_seeded"
    determinism = Determinism.SEEDED
    audit_characteristics: frozenset[AuditCharacteristic] = frozenset()


class _TransformNonDeterministic:
    name = "xfm_non_det"
    determinism = Determinism.NON_DETERMINISTIC
    audit_characteristics: frozenset[AuditCharacteristic] = frozenset()


# ── Sink fakes ──────────────────────────────────────────────────────


class _SinkKindDefault:
    """A sink inheriting Determinism.IO_WRITE — kind default for sinks."""

    name = "sink_default"
    determinism = Determinism.IO_WRITE
    audit_characteristics = frozenset({AuditCharacteristic.SIGNED})


class _SinkOverriddenToExternal:
    """A sink that overrides to EXTERNAL_CALL — a boundary sink that
    makes a network write (Azure Blob, Chroma, etc.)."""

    name = "sink_external"
    determinism = Determinism.EXTERNAL_CALL
    audit_characteristics: frozenset[AuditCharacteristic] = frozenset()


# ── Kind-default suppression behaviour ──────────────────────────────


def test_source_with_kind_default_emits_no_determinism_flag() -> None:
    """A source inheriting Determinism.IO_READ (the BaseSource default)
    contributes no derived flag. Only declared characteristics surface."""
    derived = _derive_audit_characteristics(_SourceKindDefault, plugin_kind="source")
    assert "io_read" not in derived  # suppressed — kind default
    assert "quarantine" in derived  # author-declared
    assert "provenance" in derived  # author-declared


def test_source_with_kind_default_no_declared_emits_empty_tuple() -> None:
    """A source with the kind default and no declared characteristics
    surfaces nothing on the card. That's truthful: the author made no
    audit-relevant per-plugin decision."""
    derived = _derive_audit_characteristics(_SourceKindDefaultNoDeclared, plugin_kind="source")
    assert derived == ()


def test_transform_with_kind_default_emits_no_determinism_flag() -> None:
    """BaseTransform default is Determinism.DETERMINISTIC. A transform
    inheriting it contributes no `deterministic` flag — that's the
    architectural baseline, not a per-plugin claim."""
    derived = _derive_audit_characteristics(_TransformKindDefault, plugin_kind="transform")
    assert derived == ()


def test_sink_with_kind_default_emits_no_determinism_flag() -> None:
    """BaseSink default is Determinism.IO_WRITE. Suppressed because
    every sink emits to some destination — the flag would be a
    kind-constant tautology rather than a per-plugin signal."""
    derived = _derive_audit_characteristics(_SinkKindDefault, plugin_kind="sink")
    assert "io_write" not in derived  # suppressed
    assert "signed" in derived  # author-declared


# ── Override behaviour (author makes a deliberate claim) ────────────


def test_source_overriding_default_emits_the_flag() -> None:
    """When a source author actively overrides determinism from the
    kind default (e.g. NullSource declares DETERMINISTIC) the override
    IS surfaced — it's a positive claim the author chose to make."""
    derived = _derive_audit_characteristics(_SourceOverriddenToDeterministic, plugin_kind="source")
    assert "deterministic" in derived


def test_transform_external_call_emits_flag() -> None:
    """Determinism.EXTERNAL_CALL is never a kind default — every
    plugin declaring it has made a deliberate per-plugin choice and
    the flag MUST surface."""
    derived = _derive_audit_characteristics(_TransformExternalCall, plugin_kind="transform")
    assert "external_call" in derived
    assert "credentials" in derived


def test_transform_seeded_emits_flag() -> None:
    derived = _derive_audit_characteristics(_TransformSeeded, plugin_kind="transform")
    assert "seeded" in derived


def test_transform_non_deterministic_emits_flag() -> None:
    derived = _derive_audit_characteristics(_TransformNonDeterministic, plugin_kind="transform")
    assert "non_deterministic" in derived


def test_sink_overriding_default_emits_the_flag() -> None:
    """A sink that overrides to EXTERNAL_CALL (the boundary-sink
    pattern) gets `external_call` on the card."""
    derived = _derive_audit_characteristics(_SinkOverriddenToExternal, plugin_kind="sink")
    assert "external_call" in derived


# ── Quarantine remains author-declared, not inferred ────────────────


def test_quarantine_inference_is_not_attempted_from_instance_attribute() -> None:
    """Regression guard: the derivation MUST NOT read
    `plugin_cls._on_validation_failure`. That attribute is set in
    __init__ and does not exist on the class object — reading it would
    raise AttributeError at catalog-build time."""

    class _NoInstanceAttr:
        name = "no_instance_attr"
        determinism = Determinism.SEEDED  # non-default; should emit
        audit_characteristics: frozenset[AuditCharacteristic] = frozenset()

    derived = _derive_audit_characteristics(_NoInstanceAttr, plugin_kind="source")
    assert "seeded" in derived
    assert "quarantine" not in derived


# ── Sort/order + closed-vocabulary exhaustiveness ───────────────────


def test_returns_sorted_tuple() -> None:
    """Derivation returns a sorted tuple[str, ...] for stable wire-format
    ordering; the response model exposes this directly to the frontend."""
    derived = _derive_audit_characteristics(_SourceKindDefault, plugin_kind="source")
    assert isinstance(derived, tuple)
    assert list(derived) == sorted(derived)


def test_determinism_to_audit_flag_covers_all_enum_values() -> None:
    """_DETERMINISM_TO_AUDIT_FLAG must be exhaustive over Determinism.

    If a new Determinism value is added to contracts/enums.py without a
    corresponding entry in _DETERMINISM_TO_AUDIT_FLAG, the subscript
    access in _derive_audit_characteristics raises KeyError at runtime.
    """
    from elspeth.web.catalog.service import _DETERMINISM_TO_AUDIT_FLAG

    assert set(_DETERMINISM_TO_AUDIT_FLAG.keys()) == set(Determinism)


def test_kind_default_determinism_table_covers_all_plugin_kinds() -> None:
    """_KIND_DEFAULT_DETERMINISM must cover every PluginKind value.

    A future PluginKind addition without a corresponding entry would
    KeyError at catalog-build time; this test surfaces the gap up front.
    """
    from elspeth.web.catalog.service import _KIND_DEFAULT_DETERMINISM

    # PluginKind is a Literal type — enumerate its values to assert coverage.
    expected_kinds = {"source", "transform", "sink"}
    assert set(_KIND_DEFAULT_DETERMINISM.keys()) == expected_kinds
