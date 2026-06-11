"""Read virtual discard sink summaries from the Landscape audit database."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.engine.url import make_url

from elspeth.contracts.audit import DISCARD_SINK_NAME
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    token_outcomes_table,
    transform_errors_table,
    validation_errors_table,
)
from elspeth.web.config import WebSettings
from elspeth.web.execution.schemas import DiscardStageSummary, DiscardSummary

DISCARD_DESTINATION = "discard"


def load_discard_summaries_for_settings(
    settings: WebSettings,
    landscape_run_ids: Iterable[str | None],
) -> dict[str, DiscardSummary]:
    """Load discard summaries for run IDs using web runtime settings.

    Missing SQLite audit files return an empty result so session-list tests
    and fresh local deployments do not create empty audit databases while
    rendering runs that predate Landscape execution. Existing but invalid
    audit databases still raise through ``LandscapeDB.from_url``.
    """
    run_ids = _unique_run_ids(landscape_run_ids)
    if not run_ids:
        return {}

    landscape_url = settings.get_landscape_url()
    if _sqlite_database_file_missing(landscape_url):
        return {}

    with LandscapeDB.from_url(
        landscape_url,
        passphrase=settings.landscape_passphrase,
        create_tables=False,
        read_only=True,
    ) as db:
        return load_discard_summaries_from_db(db, run_ids)


def load_discard_summaries_from_db(
    db: LandscapeDB,
    landscape_run_ids: Iterable[str],
) -> dict[str, DiscardSummary]:
    """Load discard summaries from an already-open Landscape database."""
    run_ids = _unique_run_ids(landscape_run_ids)
    if not run_ids:
        return {}

    counts: dict[str, dict[str, int]] = {
        run_id: {
            "validation_errors": 0,
            "transform_errors": 0,
            "sink_discards": 0,
        }
        for run_id in run_ids
    }
    stages: dict[str, list[DiscardStageSummary]] = {run_id: [] for run_id in run_ids}

    with db.read_only_connection() as conn:
        validation_query = (
            select(
                validation_errors_table.c.run_id,
                validation_errors_table.c.node_id,
                func.count().label("count"),
            )
            .where(validation_errors_table.c.run_id.in_(run_ids))
            .where(validation_errors_table.c.destination == DISCARD_DESTINATION)
            .group_by(validation_errors_table.c.run_id, validation_errors_table.c.node_id)
            .order_by(validation_errors_table.c.run_id.asc(), validation_errors_table.c.node_id.asc())
        )
        for run_id, node_id, count in conn.execute(validation_query):
            count_value = int(count)
            counts[run_id]["validation_errors"] += count_value
            stages[run_id].append(
                DiscardStageSummary(
                    stage="source_validation",
                    node_id=node_id,
                    count=count_value,
                )
            )

        transform_query = (
            select(
                transform_errors_table.c.run_id,
                transform_errors_table.c.transform_id,
                func.count().label("count"),
            )
            .where(transform_errors_table.c.run_id.in_(run_ids))
            .where(transform_errors_table.c.destination == DISCARD_DESTINATION)
            .group_by(transform_errors_table.c.run_id, transform_errors_table.c.transform_id)
            .order_by(transform_errors_table.c.run_id.asc(), transform_errors_table.c.transform_id.asc())
        )
        for run_id, transform_id, count in conn.execute(transform_query):
            count_value = int(count)
            counts[run_id]["transform_errors"] += count_value
            stages[run_id].append(
                DiscardStageSummary(
                    stage="transform_validation",
                    node_id=transform_id,
                    count=count_value,
                )
            )

        sink_query = (
            select(token_outcomes_table.c.run_id, func.count().label("count"))
            .where(token_outcomes_table.c.run_id.in_(run_ids))
            .where(token_outcomes_table.c.sink_name == DISCARD_SINK_NAME)
            .where(token_outcomes_table.c.completed == 1)
            .group_by(token_outcomes_table.c.run_id)
        )
        for run_id, count in conn.execute(sink_query):
            count_value = int(count)
            counts[run_id]["sink_discards"] = count_value
            stages[run_id].append(
                DiscardStageSummary(
                    stage="sink_discard",
                    node_id=None,
                    count=count_value,
                )
            )

    summaries: dict[str, DiscardSummary] = {}
    for run_id, run_counts in counts.items():
        total = run_counts["validation_errors"] + run_counts["transform_errors"] + run_counts["sink_discards"]
        if total > 0:
            summaries[run_id] = DiscardSummary(total=total, stages=tuple(stages[run_id]), **run_counts)
    return summaries


def _unique_run_ids(landscape_run_ids: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(sorted({run_id for run_id in landscape_run_ids if run_id}))


def _sqlite_database_file_missing(landscape_url: str) -> bool:
    parsed = make_url(landscape_url)
    if not parsed.drivername.startswith("sqlite"):
        return False
    database = parsed.database
    if database is None or database == ":memory:":
        return False
    return not Path(database).exists()
