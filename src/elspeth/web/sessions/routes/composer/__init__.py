from __future__ import annotations

from .._helpers import APIRouter
from . import compose, guided, proposals, state

__all__ = ["register_composer_routes"]


def register_composer_routes(router: APIRouter) -> None:
    """Register all composer routes onto the session router.

    Handlers live in area sub-modules (state / proposals / compose / guided);
    each owns its own ``APIRouter`` and is merged here so the routes inherit
    the parent router's ``/api/sessions`` prefix and tags unchanged.
    """
    router.include_router(state.router)
    router.include_router(proposals.router)
    router.include_router(compose.router)
    router.include_router(guided.router)
