"""Parity guard for the (kind, determinism) boundary predicate.

Replaces the prior ``test_boundary_attribute_parity.py`` (deleted alongside
the ``data_trust_tier`` class attribute). The earlier file pinned a
hand-curated ``_INTERNAL_PLUGIN_CLASSES`` list against the class-attribute
declaration; this file pins the expected boundary/internal partition
against the new predicate that derives boundary status from
``(plugin_kind, plugin_cls.determinism)``.

The expected partition is sourced from
``elspeth.web.audit_readiness.boundary_expectations`` — a production
module whose frozensets exist precisely so that catalog changes which
add a Tier-3 crossing surface as a production-code diff in PR review
(see that module's docstring for the audit-discoverability rationale).
"""

from __future__ import annotations

from elspeth.contracts.enums import Determinism
from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.web.audit_readiness.boundary_expectations import (
    EXPECTED_BOUNDARY_SINKS,
    EXPECTED_BOUNDARY_SOURCES,
    EXPECTED_BOUNDARY_TRANSFORMS,
    EXPECTED_SINK_DETERMINISMS,
    EXPECTED_SOURCE_DETERMINISMS,
    EXPECTED_TRANSFORM_DETERMINISMS,
)
from elspeth.web.audit_readiness.service import _AUDIT_FLAGGED_DETERMINISMS
from elspeth.web.catalog.schemas import PluginKind


def _predicate_says_boundary(kind: PluginKind, plugin_cls: type) -> bool:
    """Mirror of ``_build_plugin_trust_row`` in
    ``elspeth.web.audit_readiness.service``. Lift the predicate into the
    test so a future change to the production helper that drifts from
    the parity-set lists below fails this test, not just integration
    tests that happen to exercise an affected plugin."""
    return kind in ("source", "sink") or plugin_cls.determinism in _AUDIT_FLAGGED_DETERMINISMS


def _make_manager() -> PluginManager:
    manager = PluginManager()
    manager.register_builtin_plugins()
    return manager


def _registered_determinisms(classes: list[type]) -> dict[str, Determinism]:
    return {cls.name: cls.determinism for cls in classes}


def _assert_per_plugin_parity(actual: dict[str, Determinism], expected: dict[str, Determinism], kind_label: str) -> None:
    """Per-plugin determinism equality.

    Stronger than set-equality of names: catches the case where two
    offsetting drifts (e.g. rename + readd, or determinism change on a
    plugin whose kind short-circuits the boundary predicate) would
    cancel under set-equality. Per-plugin pinning means a value drift
    on any single name fails immediately with a named diff.
    """
    missing_in_actual = expected.keys() - actual.keys()
    extra_in_actual = actual.keys() - expected.keys()
    mismatched = {name: (actual[name], expected[name]) for name in expected.keys() & actual.keys() if actual[name] is not expected[name]}
    assert not (missing_in_actual or extra_in_actual or mismatched), (
        f"{kind_label} catalog drifted from declared expectations.\n"
        f"  missing from catalog (declared in expectations but no longer registered): {sorted(missing_in_actual)}\n"
        f"  unexpected in catalog (registered but not in expectations): {sorted(extra_in_actual)}\n"
        f"  determinism value changed on registered plugin (name: (actual, expected)): {mismatched}\n"
        f"Update src/elspeth/web/audit_readiness/boundary_expectations.py "
        f"to reflect the catalog change AS PART OF THE SAME COMMIT — "
        f"the production-code diff is the audit-discoverability signal."
    )


def test_every_source_classifies_as_boundary() -> None:
    """Every registered source must classify as boundary under the
    (kind, determinism) predicate. The per-plugin determinism map
    additionally pins each source's declared determinism so a silent
    drift (e.g. a Source whose determinism flips from IO_READ to
    NON_DETERMINISTIC while remaining boundary by kind) fails here."""
    manager = _make_manager()
    actual = _registered_determinisms(list(manager.get_sources()))
    _assert_per_plugin_parity(actual, EXPECTED_SOURCE_DETERMINISMS, "Source")
    # Set-equality assertion preserved alongside per-plugin parity so the
    # name-only invariant (every registered source appears in the boundary
    # set, since sources short-circuit boundary by kind) stays explicit.
    registered_names = frozenset(actual)
    assert registered_names == EXPECTED_BOUNDARY_SOURCES, (
        f"Expected boundary sources drifted from catalog: "
        f"missing={EXPECTED_BOUNDARY_SOURCES - registered_names}, "
        f"extra={registered_names - EXPECTED_BOUNDARY_SOURCES}"
    )
    for cls in manager.get_sources():
        assert _predicate_says_boundary("source", cls), (
            f"Source {cls.name!r} did not classify as boundary under the (kind, determinism) predicate. Determinism: {cls.determinism!r}"
        )


def test_every_sink_classifies_as_boundary() -> None:
    """Every registered sink must classify as boundary — writing data
    out of the pipeline is a boundary crossing regardless of whether
    the destination is local file or remote service. The per-plugin
    determinism map pins each sink's declared determinism for the
    same anti-drift reason documented on the sources test."""
    manager = _make_manager()
    actual = _registered_determinisms(list(manager.get_sinks()))
    _assert_per_plugin_parity(actual, EXPECTED_SINK_DETERMINISMS, "Sink")
    registered_names = frozenset(actual)
    assert registered_names == EXPECTED_BOUNDARY_SINKS, (
        f"Expected boundary sinks drifted from catalog: "
        f"missing={EXPECTED_BOUNDARY_SINKS - registered_names}, "
        f"extra={registered_names - EXPECTED_BOUNDARY_SINKS}"
    )
    for cls in manager.get_sinks():
        assert _predicate_says_boundary("sink", cls), (
            f"Sink {cls.name!r} did not classify as boundary under the (kind, determinism) predicate. Determinism: {cls.determinism!r}"
        )


def test_audit_flagged_transforms_classify_as_boundary() -> None:
    """Transforms declaring an audit-flagged determinism are boundary;
    every other Transform is internal-only. The per-plugin determinism
    map pins each transform's declared value so any drift (an internal
    transform mistakenly flagged IO_READ, EXTERNAL_CALL, or
    NON_DETERMINISTIC, or any silent value change) fails here with a
    per-name diff."""
    manager = _make_manager()
    actual = _registered_determinisms(list(manager.get_transforms()))
    _assert_per_plugin_parity(actual, EXPECTED_TRANSFORM_DETERMINISMS, "Transform")
    # Boundary-set assertion preserved alongside per-plugin parity so
    # the predicate's downstream classification is also pinned.
    boundary_actual = frozenset(cls.name for cls in manager.get_transforms() if _predicate_says_boundary("transform", cls))
    assert boundary_actual == EXPECTED_BOUNDARY_TRANSFORMS, (
        f"Boundary transform set drifted: "
        f"missing={EXPECTED_BOUNDARY_TRANSFORMS - boundary_actual}, "
        f"extra={boundary_actual - EXPECTED_BOUNDARY_TRANSFORMS}"
    )


def test_internal_transforms_classify_as_non_boundary() -> None:
    """Inverse of the boundary test: every Transform not in the
    expected boundary set must be classified as internal-only. Catches
    the case where a new pure transform accidentally inherits the
    wrong determinism (e.g. from a copy-pasted EXTERNAL_CALL declaration)."""
    manager = _make_manager()
    internal_actual = frozenset(cls.name for cls in manager.get_transforms() if not _predicate_says_boundary("transform", cls))
    overlap = internal_actual & EXPECTED_BOUNDARY_TRANSFORMS
    assert not overlap, (
        f"Transform(s) {sorted(overlap)} appear in both the internal "
        f"set (predicate says non-boundary) and the expected-boundary "
        f"set. Determinism declaration is inconsistent."
    )
