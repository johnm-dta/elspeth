"""Stable process-pool worker helpers for regex isolation."""

from __future__ import annotations

import re


def run_regex_worker(
    pattern: re.Pattern[str],
    text: str,
) -> tuple[bool, str | None, str | None]:
    """Run regex search in a worker process and return picklable match data."""
    match = pattern.search(text)
    if match is None:
        return (False, None, None)
    group0 = match.group(0)
    group1 = match.group(1) if pattern.groups else None
    return (True, group0, group1)
