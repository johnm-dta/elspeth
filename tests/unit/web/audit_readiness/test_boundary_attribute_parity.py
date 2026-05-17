"""Completeness and drift guards for data_trust_tier declarations on the
plugin catalog. Successor to the deleted trust.py allowlist tests.

Three guards in this module:

  1. test_boundary_plugin_has_data_trust_tier_three — positive
     completeness: every plugin previously classified BOUNDARY by the
     deleted trust.py must declare data_trust_tier == 3. Hardcoded
     import list (15 plugins).

  2. test_every_external_call_plugin_declares_boundary_tier —
     structural drift guard: iterates the LIVE catalog and asserts
     every Determinism.EXTERNAL_CALL transform/sink declares
     data_trust_tier == 3. Catches newly-added external-call plugins
     that the hardcoded list above cannot see. Successor to the
     deleted test_trust.py completeness test.

  3. test_internal_plugin_has_no_boundary_tier — negative regression
     guard: known-internal plugins must NOT declare data_trust_tier ==
     3. Catches accidental over-declarations that would silently render
     internal plugins as BOUNDARY in audit-readiness.

Do NOT use getattr() to read data_trust_tier — the field is declared
on BaseSource / BaseTransform / BaseSink (and on the corresponding
contracts/plugin_protocols.py protocols); direct attribute access is
correct and will AttributeError loudly on missing fields, which is the
desired behaviour.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.enums import Determinism

# Plugin class imports — flat, alphabetised by ruff. Authoritative membership
# of each plugin in either the BOUNDARY or INTERNAL bucket lives in
# ``_BOUNDARY_PLUGIN_CLASSES`` / ``_INTERNAL_PLUGIN_CLASSES`` below — the
# imports themselves intentionally do not annotate membership (a Sink
# imported here may be either boundary-tier or internal; the literals carry
# that signal). All six sources are BOUNDARY; sinks and transforms split.
from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
from elspeth.plugins.sinks.chroma_sink import ChromaSink
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.database_sink import DatabaseSink
from elspeth.plugins.sinks.dataverse import DataverseSink
from elspeth.plugins.sinks.json_sink import JSONSink
from elspeth.plugins.sources.azure_blob_source import AzureBlobSource
from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.plugins.sources.dataverse import DataverseSource
from elspeth.plugins.sources.json_source import JSONSource
from elspeth.plugins.sources.null_source import NullSource
from elspeth.plugins.sources.text_source import TextSource
from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield
from elspeth.plugins.transforms.field_mapper import FieldMapper
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.plugins.transforms.passthrough import PassThrough
from elspeth.plugins.transforms.rag.transform import RAGRetrievalTransform
from elspeth.plugins.transforms.truncate import Truncate
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

_BOUNDARY_PLUGIN_CLASSES = [
    # Sources
    CSVSource,
    JSONSource,
    TextSource,
    AzureBlobSource,
    DataverseSource,
    NullSource,
    # Transforms
    LLMTransform,
    WebScrapeTransform,
    RAGRetrievalTransform,
    AzureContentSafety,
    AzurePromptShield,
    # Sinks
    AzureBlobSink,
    ChromaSink,
    DatabaseSink,
    DataverseSink,
]


@pytest.mark.parametrize("plugin_cls", _BOUNDARY_PLUGIN_CLASSES, ids=lambda c: c.name)
def test_boundary_plugin_has_data_trust_tier_three(plugin_cls) -> None:
    """Every previously-BOUNDARY plugin must declare data_trust_tier == 3.

    This test gates the trust.py deletion in Task 6. It must be fully
    green before the deletion commit proceeds.
    """
    assert plugin_cls.data_trust_tier == 3, (
        f"{plugin_cls.__name__} (name={plugin_cls.name!r}) has "
        f"data_trust_tier={plugin_cls.data_trust_tier!r}; expected 3. "
        f"Author 'data_trust_tier: int | None = 3' in the class body."
    )


# Closed exception set for the structural drift guard below. Empty by
# design — extend with a written rationale (and a code-review-visible
# comment) if a future Determinism.EXTERNAL_CALL plugin legitimately
# should NOT be Tier-3 boundary (e.g., a self-loop or pure-compute
# transform that happens to use EXTERNAL_CALL determinism for
# cache-invalidation reasons). The pattern is inherited from the deleted
# test_trust.py allowlist exception mechanism.
_EXTERNAL_CALL_EXCEPTIONS: frozenset[str] = frozenset()


def test_every_external_call_plugin_declares_boundary_tier() -> None:
    """Structural drift guard: every plugin with Determinism.EXTERNAL_CALL
    must declare data_trust_tier == 3.

    A new external-call transform/sink that lacks the declaration would
    otherwise silently render as INTERNAL in audit-readiness snapshots —
    exactly the silent-misclassification failure mode trust.py's
    allowlist used to catch at CI time. The hardcoded
    _BOUNDARY_PLUGIN_CLASSES list above does not iterate the live
    catalog, so it cannot catch a newly-added plugin; this test does.

    The closed exception set ``_EXTERNAL_CALL_EXCEPTIONS`` is the
    successor to the deleted test_trust.py allowlist: extensions require
    a written rationale.
    """
    from elspeth.plugins.infrastructure.manager import PluginManager

    manager = PluginManager()
    manager.register_builtin_plugins()

    violations: list[str] = []
    for cls in list(manager.get_transforms()) + list(manager.get_sinks()):
        if cls.determinism is Determinism.EXTERNAL_CALL and cls.name not in _EXTERNAL_CALL_EXCEPTIONS and cls.data_trust_tier != 3:
            violations.append(
                f"{cls.__name__} (name={cls.name!r}) declares "
                f"Determinism.EXTERNAL_CALL but data_trust_tier="
                f"{cls.data_trust_tier!r}; expected 3."
            )

    assert not violations, "\n".join(violations)


# Negative-coverage list: known-internal plugins selected from the live
# catalog. These must NOT declare data_trust_tier == 3 — mis-declaring
# an internal plugin as boundary would silently render it as BOUNDARY
# in audit-readiness, falsifying the trust narrative the snapshot
# documents.
#
# Selection rationale (verified via PluginManager().get_transforms() /
# get_sinks() filtering on data_trust_tier != 3):
#   - CSVSink / JSONSink: local file writes, no external system
#   - FieldMapper / Truncate / PassThrough: pure row-level transforms,
#     no external calls and no Tier-3 surface
_INTERNAL_PLUGIN_CLASSES = [
    # Sinks
    CSVSink,
    JSONSink,
    # Transforms
    FieldMapper,
    Truncate,
    PassThrough,
]


@pytest.mark.parametrize("plugin_cls", _INTERNAL_PLUGIN_CLASSES, ids=lambda c: c.name)
def test_internal_plugin_has_no_boundary_tier(plugin_cls) -> None:
    """Negative regression guard: known-internal plugins must NOT
    declare data_trust_tier == 3.

    Mis-declaring an internal plugin as data_trust_tier == 3 would
    silently render it as BOUNDARY in audit-readiness, falsifying the
    trust narrative the snapshot documents. This test catches such
    drift at CI time.
    """
    assert plugin_cls.data_trust_tier != 3, (
        f"{plugin_cls.__name__} (name={plugin_cls.name!r}) declares "
        f"data_trust_tier=3 but is not on the EXTERNAL_BOUNDARY allowlist. "
        f"Internal plugins should keep the base-class default (None)."
    )
