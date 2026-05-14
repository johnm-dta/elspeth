"""Deterministic freeform intent routing for registered composer recipes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from elspeth.contracts.freeze import freeze_fields

_FORK_COALESCE_RECIPE = "fork-coalesce-truncate-jsonl"
_CSV_MARKER_RE = re.compile(r"(?:customer\s+rows\s*)?\(csv\):\s*\n(?P<csv>.+)\Z", re.IGNORECASE | re.DOTALL)
_OUTPUT_PATH_RE = re.compile(r"\bat\s+(?P<path>[^\s:]+\.jsonl)\b", re.IGNORECASE)
_TRUNCATE_RE = re.compile(
    r"\btruncates?\s+the\s+(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s+field\s+to\s+(?P<max_chars>\d+)\s+characters?",
    re.IGNORECASE,
)
_SUFFIX_RE = re.compile(r"\bsuffix\s+(?P<quote>['\"])(?P<suffix>.*?)(?P=quote)", re.IGNORECASE)
_KEYS_RE = re.compile(
    r"under\s+separate\s+keys\s+`(?P<key_a>[A-Za-z_][A-Za-z0-9_]*)`\s+and\s+`(?P<key_b>[A-Za-z_][A-Za-z0-9_]*)`",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class InlineRecipeBlob:
    filename: str
    mime_type: str
    content: str


@dataclass(frozen=True, slots=True)
class FreeformRecipeIntentMatch:
    recipe_name: str
    slots: Mapping[str, object]
    inline_blob: InlineRecipeBlob | None = None

    def __post_init__(self) -> None:
        freeze_fields(self, "slots")


def match_freeform_recipe_intent(message: str) -> FreeformRecipeIntentMatch | None:
    """Return a deterministic recipe match for a freeform composer request."""
    lower = message.lower()
    if not all(needle in lower for needle in ("two ways in parallel", "combined into a single merged output row", "truncat")):
        return None

    csv_match = _CSV_MARKER_RE.search(message)
    output_match = _OUTPUT_PATH_RE.search(message)
    truncate_match = _TRUNCATE_RE.search(message)
    suffix_match = _SUFFIX_RE.search(message)
    keys_match = _KEYS_RE.search(message)
    if csv_match is None or output_match is None or truncate_match is None or suffix_match is None or keys_match is None:
        return None

    csv_content = csv_match.group("csv").strip()
    if "\n" not in csv_content:
        return None

    return FreeformRecipeIntentMatch(
        recipe_name=_FORK_COALESCE_RECIPE,
        inline_blob=InlineRecipeBlob(
            filename="inline-fork-coalesce.csv",
            mime_type="text/csv",
            content=csv_content,
        ),
        slots={
            "truncate_field": truncate_match.group("field"),
            "max_chars": int(truncate_match.group("max_chars")),
            "truncation_suffix": suffix_match.group("suffix"),
            "output_path": output_match.group("path"),
            "key_a": keys_match.group("key_a"),
            "key_b": keys_match.group("key_b"),
        },
    )
