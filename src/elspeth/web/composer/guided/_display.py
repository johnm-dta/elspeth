"""Human display labels for plugin ids in guided-mode option lists.

The guided source/sink pickers are backend-authored: the emitters build
``_Option`` entries whose ``label`` used to echo the raw plugin id
(``azure_blob``, ``batch_top_k``) — machine register for a novice-facing
picker (ux review elspeth-5ee1f76e39, backend half). This module derives a
human display label; the option ``id`` (the VALUE the client submits) stays
the raw plugin id — only the presentation label changes.

Mirrors the frontend catalog module
``frontend/src/components/catalog/pluginDisplayName.ts`` (curated overrides +
humaniser). The two are hand-mirrored — deliberately NOT imported across the
stack — so update both when a curated override changes.

Derivation, in order:
  1. Curated override — for ids whose plain title-casing is wrong or
     unhelpful (``azure_blob`` is Azure Blob Storage, not "Azure Blob").
  2. Humanised fallback — underscores to spaces, Title Case, with a closed
     acronym set upper-cased (``json_explode`` → "JSON Explode").

Update discipline: new plugins get a sensible humanised name for free; add
an override only when that name misleads.
"""

from __future__ import annotations

from typing import Final

_ACRONYMS: Final[frozenset[str]] = frozenset(
    {
        "ai",
        "api",
        "csv",
        "db",
        "http",
        "https",
        "id",
        "io",
        "json",
        "llm",
        "rag",
        "sql",
        "url",
        "yaml",
    }
)
"""Words rendered fully upper-case by the humanised fallback."""

_DISPLAY_NAME_OVERRIDES: Final[dict[str, str]] = {
    "azure_blob": "Azure Blob Storage",
    "dataverse": "Microsoft Dataverse",
    "chroma_sink": "Chroma Vector Store",
    "batch_top_k": "Batch Top-K",
    # The resume-only placeholder source. Its id is literally "null"; the
    # display name says what it is for instead of echoing a developer value
    # at end users. (The guided source picker hides it anyway — see
    # ``emitters._GUIDED_HIDDEN_SOURCES`` — but the mirror stays complete.)
    "null": "Resume Placeholder",
}
"""Curated display names, keyed by plugin id (frontend mirror)."""


def _title_case_word(word: str) -> str:
    if word.lower() in _ACRONYMS:
        return word.upper()
    return word[:1].upper() + word[1:]


def plugin_display_label(plugin_id: str) -> str:
    """Human display label for a plugin id.

    Presentation only — the guided option ``id`` (the value the client
    submits back) must remain the raw plugin id.
    """
    override = _DISPLAY_NAME_OVERRIDES.get(plugin_id)
    if override is not None:
        return override
    words = [word for word in plugin_id.replace("_", " ").split() if word]
    return " ".join(_title_case_word(word) for word in words)
