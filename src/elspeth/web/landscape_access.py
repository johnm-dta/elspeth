"""Fail-closed web-layer access to writable Landscape databases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.core.landscape.database import LandscapeDB
from elspeth.web.deployment_contract import DEPLOYMENT_TARGET_AWS_ECS
from elspeth.web.schema_probe import postgres_engine_kwargs

if TYPE_CHECKING:
    from elspeth.web.config import WebSettings


def landscape_create_tables_allowed(settings: WebSettings) -> bool:
    """Return whether this deployment may lazily create Landscape schema."""
    if settings.deployment_target == DEPLOYMENT_TARGET_AWS_ECS:
        return False
    if settings.deployment_target == "default":
        return True
    raise ValueError("unsupported deployment_target for Landscape schema policy")


def open_landscape_db(settings: WebSettings) -> LandscapeDB:
    """Open a writable Landscape database with the deployment schema policy."""
    create_tables = landscape_create_tables_allowed(settings)
    url = settings.get_landscape_url()
    return LandscapeDB.from_url(
        url,
        passphrase=settings.landscape_passphrase,
        create_tables=create_tables,
        **postgres_engine_kwargs(url),
    )
