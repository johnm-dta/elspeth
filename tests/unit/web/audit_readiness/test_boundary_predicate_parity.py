"""Parity guard for the (kind, determinism) boundary predicate.

Replaces the prior `test_boundary_attribute_parity.py` (deleted alongside
the `data_trust_tier` class attribute).  The earlier file pinned a
hand-curated `_INTERNAL_PLUGIN_CLASSES` list against the class-attribute
declaration; this file pins the same expected boundary/internal partition
against the *new* predicate that derives boundary status from
`(plugin_kind, plugin_cls.determinism)`.

The expected partition is lifted verbatim from the deleted file's hand-
curated lists so a future regression that misclassifies any of the named
plugins fails CI here.
"""

from __future__ import annotations

from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.web.audit_readiness.service import _BOUNDARY_DETERMINISMS
from elspeth.web.catalog.schemas import PluginKind

# Every Source in the registered catalog is a boundary plugin: by
# definition a Source reads external data into the pipeline. There is no
# such thing as an "internal source" in ELSPETH's architecture.
_EXPECTED_BOUNDARY_SOURCES: frozenset[str] = frozenset(
    {
        "azure_blob",
        "csv",
        "dataverse",
        "json",
        "null",
        "text",
    },
)

# Every Sink is a boundary plugin: writing data out of the pipeline (to
# a file, database, blob store, or downstream service) crosses an
# external trust boundary regardless of whether the destination is local
# or remote.
_EXPECTED_BOUNDARY_SINKS: frozenset[str] = frozenset(
    {
        "azure_blob",
        "chroma_sink",
        "csv",
        "database",
        "dataverse",
        "json",
    },
)

# Boundary Transforms — determinism is EXTERNAL_CALL or NON_DETERMINISTIC
# per ``_BOUNDARY_DETERMINISMS`` in
# ``elspeth.web.audit_readiness.service``. Every other Transform in the
# catalog is internal-only.
_EXPECTED_BOUNDARY_TRANSFORMS: frozenset[str] = frozenset(
    {
        "azure_content_safety",
        "azure_prompt_shield",
        "llm",
        "rag_retrieval",
        "web_scrape",
    },
)


def _predicate_says_boundary(kind: PluginKind, plugin_cls: type) -> bool:
    """Mirror of ``_build_plugin_trust_row`` in
    ``elspeth.web.audit_readiness.service``. Lift the predicate into the
    test so a future change to the production helper that drifts from
    the parity-set lists below fails this test, not just integration
    tests that happen to exercise an affected plugin."""
    return kind in ("source", "sink") or plugin_cls.determinism in _BOUNDARY_DETERMINISMS


def _make_manager() -> PluginManager:
    manager = PluginManager()
    manager.register_builtin_plugins()
    return manager


def test_every_source_classifies_as_boundary() -> None:
    """Every registered source must classify as boundary under the
    (kind, determinism) predicate. A source that fails to classify
    indicates either a missing class in the catalog or a determinism
    classification that doesn't make sense for a Source."""
    manager = _make_manager()
    registered_names = frozenset(cls.name for cls in manager.get_sources())
    assert registered_names == _EXPECTED_BOUNDARY_SOURCES, (
        f"Expected boundary sources drifted from catalog: "
        f"missing={_EXPECTED_BOUNDARY_SOURCES - registered_names}, "
        f"extra={registered_names - _EXPECTED_BOUNDARY_SOURCES}"
    )
    for cls in manager.get_sources():
        assert _predicate_says_boundary("source", cls), (
            f"Source {cls.name!r} did not classify as boundary under the (kind, determinism) predicate. Determinism: {cls.determinism!r}"
        )


def test_every_sink_classifies_as_boundary() -> None:
    """Every registered sink must classify as boundary — writing data
    out of the pipeline is a boundary crossing regardless of whether
    the destination is local file or remote service."""
    manager = _make_manager()
    registered_names = frozenset(cls.name for cls in manager.get_sinks())
    assert registered_names == _EXPECTED_BOUNDARY_SINKS, (
        f"Expected boundary sinks drifted from catalog: "
        f"missing={_EXPECTED_BOUNDARY_SINKS - registered_names}, "
        f"extra={registered_names - _EXPECTED_BOUNDARY_SINKS}"
    )
    for cls in manager.get_sinks():
        assert _predicate_says_boundary("sink", cls), (
            f"Sink {cls.name!r} did not classify as boundary under the (kind, determinism) predicate. Determinism: {cls.determinism!r}"
        )


def test_external_call_transforms_classify_as_boundary() -> None:
    """Transforms declaring Determinism.EXTERNAL_CALL are boundary;
    every other Transform is internal-only. The lists below pin the
    current partition so a drift (an internal transform mistakenly
    flagged EXTERNAL_CALL, or vice versa) fails here."""
    manager = _make_manager()
    boundary_actual = frozenset(cls.name for cls in manager.get_transforms() if _predicate_says_boundary("transform", cls))
    assert boundary_actual == _EXPECTED_BOUNDARY_TRANSFORMS, (
        f"Boundary transform set drifted: "
        f"missing={_EXPECTED_BOUNDARY_TRANSFORMS - boundary_actual}, "
        f"extra={boundary_actual - _EXPECTED_BOUNDARY_TRANSFORMS}"
    )


def test_internal_transforms_classify_as_non_boundary() -> None:
    """Inverse of the boundary test: every Transform not in the
    expected boundary set must be classified as internal-only. Catches
    the case where a new pure transform accidentally inherits the
    wrong determinism (e.g. from a copy-pasted EXTERNAL_CALL declaration)."""
    manager = _make_manager()
    internal_actual = frozenset(cls.name for cls in manager.get_transforms() if not _predicate_says_boundary("transform", cls))
    overlap = internal_actual & _EXPECTED_BOUNDARY_TRANSFORMS
    assert not overlap, (
        f"Transform(s) {sorted(overlap)} appear in both the internal "
        f"set (predicate says non-boundary) and the expected-boundary "
        f"set. Determinism declaration is inconsistent."
    )
