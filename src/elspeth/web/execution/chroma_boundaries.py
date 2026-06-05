"""Web execution boundary checks for Chroma-backed plugins.

Chroma plugins are available when the optional ``chromadb`` dependency is
installed. In the web app their storage and network targets are still Tier-3
user-controlled execution inputs, so they must pass the same filesystem and
SSRF boundaries as first-class web source/sink options before the runtime SDK
is allowed to touch disk or open sockets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from elspeth.core.security.web import NetworkError, SSRFBlockedError, validate_url_for_ssrf
from elspeth.web.composer.state import CompositionState
from elspeth.web.paths import allowed_sink_directories, resolve_data_path

_CHROMA_SINK_PLUGIN = "chroma_sink"
_RAG_RETRIEVAL_PLUGIN = "rag_retrieval"
_CHROMA_PROVIDER = "chroma"

ComponentType = Literal["sink", "transform"]


@dataclass(frozen=True, slots=True)
class ChromaBoundaryViolation:
    """A web-authored Chroma target escaped a web boundary."""

    component_id: str
    component_type: ComponentType
    message: str
    detail: str
    suggestion: str


def find_chroma_boundary_violations(
    state: CompositionState,
    *,
    data_dir: str,
) -> tuple[ChromaBoundaryViolation, ...]:
    """Return Chroma filesystem/network boundary violations in a composition.

    The checks are deliberately web-scoped. CLI and test pipelines may still
    configure Chroma as general infrastructure, but authenticated web users may
    only use persistent storage under the web sink directories and may only use
    client hosts that pass the central SSRF DNS/IP policy.
    """
    violations: list[ChromaBoundaryViolation] = []

    for output in state.outputs or ():
        if output.plugin != _CHROMA_SINK_PLUGIN:
            continue
        violations.extend(
            _validate_chroma_options(
                options=output.options,
                component_id=output.name,
                component_type="sink",
                data_dir=data_dir,
            )
        )

    for node in state.nodes or ():
        if node.plugin != _RAG_RETRIEVAL_PLUGIN:
            continue
        options = node.options
        if options.get("provider") != _CHROMA_PROVIDER:
            continue
        provider_config = options.get("provider_config")
        if not isinstance(provider_config, Mapping):
            # Runtime config validation owns malformed provider_config shape.
            continue
        violations.extend(
            _validate_chroma_options(
                options=provider_config,
                component_id=node.id,
                component_type="transform",
                data_dir=data_dir,
            )
        )

    return tuple(violations)


def _validate_chroma_options(
    *,
    options: Mapping[str, Any],
    component_id: str,
    component_type: ComponentType,
    data_dir: str,
) -> tuple[ChromaBoundaryViolation, ...]:
    mode = options.get("mode")
    if mode == "persistent":
        persist_directory = options.get("persist_directory")
        if persist_directory is None:
            return ()
        return _validate_persist_directory(
            value=str(persist_directory),
            component_id=component_id,
            component_type=component_type,
            data_dir=data_dir,
        )
    if mode == "client":
        host = options.get("host")
        if host is None:
            return ()
        return _validate_client_target(
            host=str(host),
            port=options.get("port", 8000),
            ssl=options.get("ssl", True),
            component_id=component_id,
            component_type=component_type,
        )
    return ()


def _validate_persist_directory(
    *,
    value: str,
    component_id: str,
    component_type: ComponentType,
    data_dir: str,
) -> tuple[ChromaBoundaryViolation, ...]:
    resolved = resolve_data_path(value, data_dir)
    allowed_dirs = allowed_sink_directories(data_dir)
    if any(resolved.is_relative_to(allowed) for allowed in allowed_dirs):
        return ()
    return (
        ChromaBoundaryViolation(
            component_id=component_id,
            component_type=component_type,
            message=(
                f"Chroma persist_directory blocked: {component_type} "
                f"'{component_id}' persist_directory='{value}' resolves outside allowed output directories"
            ),
            detail=(
                f"{component_type} {component_id} persist_directory resolves outside "
                "data_dir/outputs and data_dir/blobs"
            ),
            suggestion="Use a Chroma persist_directory under the outputs or blobs directory.",
        ),
    )


def _validate_client_target(
    *,
    host: str,
    port: object,
    ssl: object,
    component_id: str,
    component_type: ComponentType,
) -> tuple[ChromaBoundaryViolation, ...]:
    scheme = "https" if ssl is not False else "http"
    host_for_url = host
    if "://" not in host_for_url and ":" in host_for_url and not host_for_url.startswith("["):
        host_for_url = f"[{host_for_url}]"
    url = f"{scheme}://{host_for_url}:{port}/"
    try:
        validate_url_for_ssrf(url)
    except (SSRFBlockedError, NetworkError) as exc:
        return (
            ChromaBoundaryViolation(
                component_id=component_id,
                component_type=component_type,
                message=(
                    f"Chroma client target blocked: {component_type} "
                    f"'{component_id}' host='{host}' port='{port}' failed SSRF validation"
                ),
                detail=f"{component_type} {component_id} Chroma client target failed SSRF validation: {exc}",
                suggestion="Use a publicly resolvable Chroma endpoint that passes the web SSRF policy.",
            ),
        )
    return ()
