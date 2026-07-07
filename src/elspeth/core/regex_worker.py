"""Stable process-pool worker helpers for regex isolation."""

from __future__ import annotations

import re
from typing import NamedTuple


class RegexMatchResult(NamedTuple):
    matched: bool
    full_match: str | None
    groups: tuple[str | None, ...]


def run_regex_worker(
    pattern: re.Pattern[str],
    text: str,
) -> RegexMatchResult:
    """Run regex search in a worker process and return neutral match data."""
    match = pattern.search(text)
    if match is None:
        return RegexMatchResult(False, None, ())
    return RegexMatchResult(True, match.group(0), match.groups())
