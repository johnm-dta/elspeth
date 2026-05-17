"""Plugin-trust classification for the audit-readiness panel.

CLAUDE.md treats trust as a per-data-flow doctrine: sources cross Tier-3
(external input), transforms that make external calls (HTTP, LLM, blob
store) cross Tier-3 at the call boundary, everything else is Tier-2.

This module collapses that into a per-component classification:

  - BOUNDARY: crosses Tier-3. Sources are uniformly BOUNDARY; transforms
    and sinks are BOUNDARY only when on the closed allowlists below.
  - INTERNAL: operates only on pipeline data.

The allowlists are closed by design. Adding a plugin that crosses Tier-3
requires updating this file in the same commit. The subset-of-catalog
tests fail the build when an entry doesn't resolve to a registered
plugin — a rename without an update fails CI rather than silently
breaking the panel.

Two-tier rationale:
  - EXTERNAL_CALL determinism (automated): web_scrape, rag_retrieval,
    azure_content_safety, azure_prompt_shield are Determinism.EXTERNAL_CALL.
    The completeness test (test_every_external_call_plugin_is_on_allowlist_or_explicitly_excepted)
    catches new EXTERNAL_CALL plugins added without allowlist update.
  - Manual curation (LLM-class): "llm" is Determinism.NON_DETERMINISTIC
    but crosses an LLM API boundary and must be visible to auditors as
    BOUNDARY. Manually curated; not caught by the completeness test.
    Document any future additions to this tier with an explicit comment.

Layer: L3 (application).

Phase 7 deletion commitment (required verbatim — do not paraphrase):
When Phase 7 adds `data_trust_tier: ClassVar` to plugin base classes,
delete this module entirely and replace `classify_plugin()` callers with
direct attribute lookup. CLAUDE.md No-Legacy requires same-commit
replacement.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

PluginKind = Literal["source", "transform", "sink"]


class PluginTrust(StrEnum):
    BOUNDARY = "boundary"
    INTERNAL = "internal"


# Transform plugins that cross an external boundary (Tier-3).
# Two categories — see module docstring two-tier rationale:
#   (a) Determinism.EXTERNAL_CALL: web_scrape, rag_retrieval,
#       azure_content_safety, azure_prompt_shield (verified via grep).
#   (b) Manual curation (NON_DETERMINISTIC + LLM boundary): llm.
# Adding a new entry:
#   1. Add the plugin's `.name` value here (verified via grep, not guessed).
#   2. Confirm the plugin module documents the external surface.
#   3. The subset-of-catalog test in test_trust.py validates the name is real.
EXTERNAL_BOUNDARY_TRANSFORMS: frozenset[str] = frozenset(
    {
        "llm",  # plugins/transforms/llm/transform.py — NON_DETERMINISTIC, manually curated (LLM API boundary)
        "web_scrape",  # plugins/transforms/web_scrape.py — Determinism.EXTERNAL_CALL
        "rag_retrieval",  # plugins/transforms/rag/transform.py — Determinism.EXTERNAL_CALL
        "azure_content_safety",  # plugins/transforms/azure/content_safety.py — Determinism.EXTERNAL_CALL
        "azure_prompt_shield",  # plugins/transforms/azure/prompt_shield.py — Determinism.EXTERNAL_CALL
    }
)

# Sink plugins that write to external systems (Determinism.EXTERNAL_CALL).
# Adding a new entry: follow the same 3-step process as transforms above.
EXTERNAL_BOUNDARY_SINKS: frozenset[str] = frozenset(
    {
        "azure_blob",  # plugins/sinks/azure_blob_sink.py — external Azure Blob Storage write
        "chroma_sink",  # plugins/sinks/chroma_sink.py — ChromaDB itself is an external system
        "database",  # plugins/sinks/database_sink.py — writes rows to a database table using SQLAlchemy Core
        "dataverse",  # plugins/sinks/dataverse.py — Determinism.EXTERNAL_CALL
    }
)


@lru_cache(maxsize=1)
def _registered_plugin_names() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Return source/transform/sink plugin names from the live builtin catalog."""
    from elspeth.plugins.infrastructure.manager import PluginManager

    manager = PluginManager()
    manager.register_builtin_plugins()
    return (
        frozenset(cls.name for cls in manager.get_sources()),
        frozenset(cls.name for cls in manager.get_transforms()),
        frozenset(cls.name for cls in manager.get_sinks()),
    )


def is_registered_plugin(kind: PluginKind, name: str) -> bool:
    """Return True when ``name`` exists in the live catalog for ``kind``."""
    sources, transforms, sinks = _registered_plugin_names()
    if kind == "source":
        return name in sources
    if kind == "transform":
        return name in transforms
    if kind == "sink":
        return name in sinks
    raise ValueError(f"unknown plugin kind: {kind!r}")


def classify_plugin(kind: PluginKind, name: str) -> PluginTrust:
    """Classify a plugin by kind + name.

    Raises:
        ValueError: when ``kind`` is not one of the three known values.
            Tier-1 invariant — the aggregator dispatches kinds taken from
            the typed CompositionState.
    """
    if kind == "source":
        return PluginTrust.BOUNDARY
    if kind == "transform":
        return PluginTrust.BOUNDARY if name in EXTERNAL_BOUNDARY_TRANSFORMS else PluginTrust.INTERNAL
    if kind == "sink":
        return PluginTrust.BOUNDARY if name in EXTERNAL_BOUNDARY_SINKS else PluginTrust.INTERNAL
    raise ValueError(f"unknown plugin kind: {kind!r}")
