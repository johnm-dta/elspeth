"""Session title conventions: default minting and default recognition.

One default-name scheme app-wide (elspeth-ef8c18a6cb): a session created
without an explicit title is named "Session — 2 Jul 2026" (the creation
date), auto-disambiguated per user with " (2)", " (3)", … suffixes so two
same-day sessions never render as identical rows in the session switcher.

``is_default_session_title`` is the single predicate for "this title is
still a mint-time default" — the first-message auto-titler uses it to
decide whether a session is eligible for a content-derived title. It also
recognises the two legacy defaults ("New session" from the old frontend
create payload, "Untitled" from the old switcher fallback) so sessions
created before this convention still graduate to content-derived titles.

Month names are an explicit English table, not ``strftime("%b")``, so the
minted title is stable regardless of the server process locale.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime

DEFAULT_SESSION_TITLE_PREFIX = "Session — "

_LEGACY_DEFAULT_TITLES = frozenset({"New session", "Untitled"})

_MONTH_ABBREVIATIONS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

_DEFAULT_TITLE_PATTERN = re.compile(
    r"^Session — \d{1,2} "
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"\d{4}(?: \(\d+\))?$"
)


def _human_date(when: datetime) -> str:
    """Render ``when`` as a compact human date, e.g. ``2 Jul 2026``."""
    return f"{when.day} {_MONTH_ABBREVIATIONS[when.month - 1]} {when.year}"


def format_default_session_title(when: datetime) -> str:
    """The undisambiguated default title for a session created at ``when``."""
    return f"{DEFAULT_SESSION_TITLE_PREFIX}{_human_date(when)}"


def mint_default_session_title(when: datetime, existing_titles: Iterable[str]) -> str:
    """Mint the default title for a new session created at ``when``.

    ``existing_titles`` is the caller's view of the user's current session
    titles (archived included — an unarchive must not resurface a
    collision). The first free variant of "Session — <date>" wins:
    the bare title, then " (2)", " (3)", …
    """
    base = format_default_session_title(when)
    taken = set(existing_titles)
    if base not in taken:
        return base
    counter = 2
    while f"{base} ({counter})" in taken:
        counter += 1
    return f"{base} ({counter})"


def is_default_session_title(title: str) -> bool:
    """True when ``title`` is a mint-time default (current or legacy scheme).

    Used by the first-message auto-titler to decide whether the session is
    still eligible for a content-derived title.
    """
    return title in _LEGACY_DEFAULT_TITLES or _DEFAULT_TITLE_PATTERN.match(title) is not None


def abandoned_tutorial_session_title(when: datetime) -> str:
    """Human title for a tutorial session swept as abandoned at ``when``.

    Replaces the machine-register ``abandoned-<title>-<ISO>``
    rename that leaked raw timestamps into the session switcher
    (elspeth-ef8c18a6cb). Same-day duplicates are acceptable: the switcher
    shows last-modified metadata to disambiguate.
    """
    return f"First-run tutorial (abandoned {_human_date(when)})"
