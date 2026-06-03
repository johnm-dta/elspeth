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


def _classify(details: dict[str, object]) -> tuple[str, str]:
    """Discriminate the two documented audit shapes and return (error_type, message).

    ``error_details_json`` is written by
    ``DataFlowRepository._canonical_or_recorded_error_details_json`` and is one of
    exactly two shapes (both Tier-1 — we authored them):

    1. The canonical :class:`~elspeth.contracts.errors.TransformErrorReason`. The
       ``reason`` field is REQUIRED and is an error *category* (e.g. ``"api_error"``,
       ``"missing_field"``) — not a human message — and is validated at the write
       boundary, so it is always present here. The human message lives in the
       optional ``error`` / ``message`` fields. ``error_type`` is an optional
       sub-category; when absent we fall back to the required ``reason`` category
       rather than fabricating an ``"UnknownError"`` token.

    2. The non-canonical envelope emitted when ``error_details`` could not be
       canonically serialized: ``{"__non_canonical__": True, "repr": ...,
       "serialization_error": ...}``. Its human-meaningful message is the
       serialization error; ``repr`` carries the salvaged payload preview.

    Both shapes are read with direct subscription, not ``.get()`` — a missing
    required key is Tier-1 corruption and KeyError surfaces it, per the module
    pledge. Messageless failures are represented distinctly (carrying the
    discriminating category / repr) so unrelated failures never collapse into one
    aggregation bucket.
    """
    if "__non_canonical__" in details:
        # Non-canonical envelope: the serialization error is the message; the
        # repr preview discriminates otherwise-identical envelopes.
        message = str(details["serialization_error"])
        return "NonCanonicalErrorDetails", f"{message} (repr={details['repr']})"

    # Canonical TransformErrorReason. ``reason`` is required (KeyError = Tier-1
    # corruption). ``error_type`` and the human-message fields are optional.
    reason = str(details["reason"])
    error_type = str(details["error_type"]) if "error_type" in details else reason
    if "error" in details:
        return error_type, str(details["error"])
    if "message" in details:
        return error_type, str(details["message"])
    # Genuinely messageless: represent the absence distinctly, carrying the
    # required category so two different reasons never merge into one bucket.
    return error_type, f"(no message; category={reason})"


def load_top_failure_samples(
    db: LandscapeDB,
    landscape_run_id: str,
    *,
    limit: int = 3,
    scan_cap: int = 500,
) -> list[FailureSample]:
    """Return the top ``limit`` distinct failure samples for a run.

    Aggregates ``transform_errors`` rows by ``(transform_id, error_type,
    message)``, where ``error_type`` and ``message`` are derived from the
    persisted ``error_details_json`` by :func:`_classify` (which discriminates
    the canonical :class:`TransformErrorReason` shape from the non-canonical
    fallback envelope). The most common combinations are returned in descending
    order. The scan is bounded by ``scan_cap`` to keep this safe on runs with
    very large error counts; aggregation accuracy on the long tail is not the
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
            # No NULL guard and no ``WHERE error_details_json IS NOT NULL``:
            # the column is nominally nullable (schema.py: ``Column(..., Text)``)
            # but its sole writer — ``DataFlowRepository.
            # _canonical_or_recorded_error_details_json`` (typed ``-> str``) —
            # always persists a non-NULL canonical JSON object on every
            # transform_errors INSERT. A NULL or non-JSON value read here is
            # therefore Tier-1 corruption of our own audit data, and the bare
            # ``json.loads`` raising is the prescribed loud failure — not a
            # case to filter away or default past.
            details = json.loads(error_details_json)
            error_type, message = _classify(details)
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
