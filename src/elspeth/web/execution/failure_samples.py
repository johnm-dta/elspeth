"""Top distinct per-row failure samples for run-level error messages.

The composer-side run row exposes a single ``error`` text field on the runs
table.  Bare structural facts (``rows_succeeded=0``) tell the operator *that*
the run failed but not *why*.  This module reads the audit DB after the run
finishes and aggregates the most common ``transform_errors`` rows so the
user-visible error string can carry an actionable sample without forcing the
operator to drill into the per-run diagnostics panel.

Tier-1 read discipline: the audit DB is full-trust, so JSON shapes are
validated strictly here (KeyError / TypeError on bad shape will surface).
The caller is responsible for catching enrichment failures and degrading to
the bare message — see ``ExecutionServiceImpl`` for the wrap pattern.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import transform_errors_table


@dataclass(frozen=True, slots=True)
class FailureSample:
    """One distinct (transform, error_type, message) tuple with row count."""

    transform_id: str
    error_type: str
    message: str
    count: int


def load_top_failure_samples(
    db: LandscapeDB,
    landscape_run_id: str,
    *,
    limit: int = 3,
    scan_cap: int = 500,
) -> list[FailureSample]:
    """Return the top ``limit`` distinct failure samples for a run.

    Aggregates ``transform_errors`` rows by ``(transform_id, error_type,
    error)`` and returns the most common combinations in descending order.
    The scan is bounded by ``scan_cap`` to keep this safe on runs with very
    large error counts; aggregation accuracy on the long tail is not the
    goal — the goal is naming the dominant failure modes for the operator.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if scan_cap < limit:
        raise ValueError("scan_cap must be >= limit")

    counter: Counter[tuple[str, str, str]] = Counter()
    with db.read_only_connection() as conn:
        stmt = (
            select(
                transform_errors_table.c.transform_id,
                transform_errors_table.c.error_details_json,
            )
            .where(transform_errors_table.c.run_id == landscape_run_id)
            .limit(scan_cap)
        )
        for transform_id, error_details_json in conn.execute(stmt):
            details = json.loads(error_details_json) if error_details_json else {}
            error_type = str(details.get("error_type", "UnknownError"))
            message = str(details.get("error", details.get("reason", "")))
            counter[(transform_id, error_type, message)] += 1

    return [
        FailureSample(transform_id=tid, error_type=etype, message=msg, count=count)
        for (tid, etype, msg), count in counter.most_common(limit)
    ]


def format_failure_samples(samples: list[FailureSample], *, message_chars: int = 240) -> str:
    """Render samples as a single multi-line string suitable for inlining.

    Each sample renders on its own bullet with the count, the error type,
    and the message (truncated to ``message_chars`` to keep the parent
    error field bounded). The transform_id is included only when distinct
    transforms appear, to keep single-transform runs concise.
    """
    if not samples:
        return ""
    distinct_transforms = {s.transform_id for s in samples}
    show_transform = len(distinct_transforms) > 1
    lines: list[str] = []
    for sample in samples:
        truncated = sample.message
        if len(truncated) > message_chars:
            truncated = truncated[: message_chars - 1].rstrip() + "…"
        prefix = f"[{sample.transform_id}] " if show_transform else ""
        lines.append(f"  • {sample.count}x {prefix}{sample.error_type}: {truncated}")
    return "\n".join(lines)
