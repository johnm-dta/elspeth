"""Shared input validation utilities for the web layer.

Unicode visibility checks are used by both auth models (identity validation)
and secret schemas (invisible-value rejection). Secret-name constants and
validation also live here so config loading, request validation, and runtime
secret stores share one contract instead of drifting independently.
"""

from __future__ import annotations

import re
import unicodedata

# Major Unicode categories that produce visible glyphs.  A string composed
# entirely of characters outside these categories (whitespace, control chars,
# zero-width joiners, format chars, etc.) is "invisible" and rejected.
#
# "M" (Mark/combining) is excluded — combining marks modify a base character
# but a string of only combining marks has no visible base and should be
# rejected.
_VISIBLE_CATEGORIES = frozenset({"L", "N", "P", "S"})

SECRET_NAME_MAX_LENGTH = 256
SECRET_NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*$"
_SECRET_NAME_RE = re.compile(SECRET_NAME_PATTERN)
SERVER_SECRET_RESERVED_PREFIX = "ELSPETH_"


def has_visible_content(s: str) -> bool:
    """Return True if *s* contains at least one visible character.

    Catches zero-width spaces (U+200B), BOM (U+FEFF), soft hyphens,
    and other invisible characters that ``str.strip()`` does not remove.
    """
    return any(unicodedata.category(c)[0] in _VISIBLE_CATEGORIES for c in s)


def validate_secret_name(name: str, *, field_name: str = "Secret name") -> str:
    """Validate a secret name against the shared web secret contract.

    The contract intentionally matches the existing user-secret API shape:
    a non-empty identifier-style name that fits into audit metadata without
    exceeding the declared schema width.
    """
    if not name:
        raise ValueError(f"{field_name} must not be empty")
    if len(name) > SECRET_NAME_MAX_LENGTH:
        raise ValueError(f"{field_name} must be <= {SECRET_NAME_MAX_LENGTH} characters")
    if _SECRET_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"{field_name} must match {SECRET_NAME_PATTERN}")
    return name


def is_reserved_server_secret_name(name: str) -> bool:
    """Return True when a server-secret name targets ELSPETH internals."""
    return name.startswith(SERVER_SECRET_RESERVED_PREFIX)
