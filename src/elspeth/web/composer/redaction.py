"""Redaction utilities for composer state serialization.

Functions that strip internal implementation details (storage paths, blob
locations) from serialized state dicts before they reach external consumers
(LLM prompts, HTTP responses, MCP tool results).
"""

from __future__ import annotations

from typing import Any

REDACTED_BLOB_SOURCE_PATH = "<redacted-blob-source-path>"


def redact_source_storage_path(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Redact internal storage paths from a serialized state dict.

    When source options contain a ``blob_ref``, ``path`` points at an
    internal storage location that must not be exposed to agents or users.
    Preserve the key with a sentinel value so external consumers can still
    tell that the source path contract is satisfied.

    Returns a shallow copy with source options redacted. Does not mutate
    the input dict.
    """
    source = state_dict.get("source")
    if source is None:
        return state_dict

    options = source.get("options")
    if options is None or "blob_ref" not in options:
        return state_dict

    # Shallow copy the chain to avoid mutating the original
    redacted = dict(state_dict)
    redacted_source = dict(source)
    redacted_options = dict(options)
    if "path" in redacted_options:
        redacted_options["path"] = REDACTED_BLOB_SOURCE_PATH
    redacted_source["options"] = redacted_options
    redacted["source"] = redacted_source
    return redacted
