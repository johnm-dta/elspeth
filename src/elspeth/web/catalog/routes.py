"""Catalog API routes — read-only plugin browsing."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from elspeth.contracts.plugin_capabilities import PluginCapability
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import (
    PluginKind,
    PluginPolicyCapabilityGroup,
    PluginPolicyControlMode,
    PluginPolicyResponse,
    PluginPolicySelection,
    PluginSchemaInfo,
    PluginSummary,
)
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId

catalog_router = APIRouter(tags=["catalog"])

# Map plural REST path segments to singular protocol values.
# The CatalogService protocol uses singular forms; REST paths use plural
# (see Seam Contract C in web-ux-seam-contracts.md).
_PLURAL_TO_SINGULAR: dict[str, PluginKind] = {"sources": "source", "transforms": "transform", "sinks": "sink"}


def _get_catalog(request: Request, user: UserIdentity) -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    """Build one principal snapshot and policy view for this request."""
    catalog: CatalogService = request.app.state.catalog_service
    snapshot: PluginAvailabilitySnapshot = request.app.state.plugin_snapshot_factory(user)
    view = PolicyCatalogView(catalog, snapshot, request.app.state.operator_profile_registry)
    return view, snapshot


def _set_private_headers(response: Response, snapshot: PluginAvailabilitySnapshot) -> None:
    response.headers.update(_private_headers(snapshot))


def _private_headers(snapshot: PluginAvailabilitySnapshot) -> dict[str, str]:
    return {
        "Cache-Control": "private, no-store",
        "Vary": "Authorization, Cookie",
        "X-ELSPETH-Plugin-Snapshot": snapshot.snapshot_hash,
    }


@catalog_router.get("/sources", response_model=list[PluginSummary])
async def list_sources(
    request: Request,
    response: Response,
    user: Annotated[UserIdentity, Depends(get_current_user)],
) -> list[PluginSummary]:
    """List all registered source plugins."""
    catalog, snapshot = _get_catalog(request, user)
    _set_private_headers(response, snapshot)
    return catalog.list_sources()


@catalog_router.get("/transforms", response_model=list[PluginSummary])
async def list_transforms(
    request: Request,
    response: Response,
    user: Annotated[UserIdentity, Depends(get_current_user)],
) -> list[PluginSummary]:
    """List all registered transform plugins."""
    catalog, snapshot = _get_catalog(request, user)
    _set_private_headers(response, snapshot)
    return catalog.list_transforms()


@catalog_router.get("/sinks", response_model=list[PluginSummary])
async def list_sinks(
    request: Request,
    response: Response,
    user: Annotated[UserIdentity, Depends(get_current_user)],
) -> list[PluginSummary]:
    """List all registered sink plugins."""
    catalog, snapshot = _get_catalog(request, user)
    _set_private_headers(response, snapshot)
    return catalog.list_sinks()


@catalog_router.get("/policy", response_model=PluginPolicyResponse)
async def get_policy(
    request: Request,
    response: Response,
    user: Annotated[UserIdentity, Depends(get_current_user)],
) -> PluginPolicyResponse:
    catalog, snapshot = _get_catalog(request, user)
    _set_private_headers(response, snapshot)
    summaries = (*catalog.list_sources(), *catalog.list_transforms(), *catalog.list_sinks())
    grouped: dict[PluginCapability, list[str]] = {}
    for summary in summaries:
        plugin_id = str(PluginId(summary.plugin_type, summary.name))
        for declaration in summary.policy_capabilities:
            grouped.setdefault(declaration.capability, []).append(plugin_id)
    return PluginPolicyResponse(
        principal_scope=snapshot.principal_scope,
        snapshot_fingerprint=snapshot.snapshot_hash,
        policy_hash=snapshot.policy_hash,
        available_plugin_ids=tuple(sorted(map(str, snapshot.available))),
        capability_groups=tuple(
            PluginPolicyCapabilityGroup(capability=capability, available_plugin_ids=tuple(sorted(plugin_ids)))
            for capability, plugin_ids in sorted(grouped.items(), key=lambda item: str(item[0]))
        ),
        selections=tuple(
            PluginPolicySelection(capability=capability, plugin_id=None if plugin_id is None else str(plugin_id))
            for capability, plugin_id in snapshot.selected
        ),
        control_modes=tuple(
            PluginPolicyControlMode(capability=capability, mode=mode)
            for capability, mode in request.app.state.web_plugin_policy.control_modes
        ),
    )


@catalog_router.get("/{plugin_type}/{name}/schema", response_model=PluginSchemaInfo)
async def get_schema(
    plugin_type: str,
    name: str,
    request: Request,
    response: Response,
    user: Annotated[UserIdentity, Depends(get_current_user)],
) -> PluginSchemaInfo:
    """Get full JSON schema for a plugin's configuration."""
    singular = _PLURAL_TO_SINGULAR.get(plugin_type)
    if singular is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown plugin type: {plugin_type}. Must be one of: {sorted(_PLURAL_TO_SINGULAR)}",
        )
    catalog, snapshot = _get_catalog(request, user)
    try:
        result = catalog.get_schema(singular, name)
        _set_private_headers(response, snapshot)
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail="plugin_not_enabled",
            headers=_private_headers(snapshot),
        ) from exc
