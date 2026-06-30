"""Pure Tier-3 parsing helpers for Azure Document Intelligence analyzeResult.

No I/O. Every function fail-closes (MalformedResponseError) on a structurally
invalid result; absent optional facets return an empty container, never
fabricated. These functions are the only place the external analyzeResult shape
is interpreted, so they are unit-tested in isolation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from elspeth.plugins.transforms.azure.errors import MalformedResponseError


def extract_content(analyze_result: Mapping[str, Any]) -> str:
    """Return analyzeResult.content, or '' when absent. Fail closed on wrong type."""
    content = analyze_result.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise MalformedResponseError(f"analyzeResult.content must be str, got {type(content).__name__}")
    return content


def extract_facet_list(analyze_result: Mapping[str, Any], azure_key: str) -> list[Any]:
    """Return analyzeResult[azure_key] (a list), or [] when absent. Fail closed on wrong type."""
    value = analyze_result.get(azure_key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise MalformedResponseError(f"analyzeResult.{azure_key} must be a list, got {type(value).__name__}")
    return value


def count_pages(analyze_result: Mapping[str, Any]) -> int:
    """Return the number of analyzed pages, or 0 when absent. Fail closed on wrong type."""
    return len(extract_facet_list(analyze_result, "pages"))


def operation_location_host_matches(operation_url: str, endpoint: str) -> bool:
    """True iff operation_url is a well-formed HTTPS URL on the same host AND port as endpoint.

    Guards against the polled Operation-Location (which carries our api-key header)
    being pointed elsewhere by a malformed/compromised 202 response. We refuse to
    follow it unless its scheme is https and its host AND port both match the
    operator-configured endpoint, so the api-key is never sent to a different
    origin (host:port) than the operator configured.
    """
    try:
        op = urlparse(operation_url)
        ep = urlparse(endpoint)
    except ValueError:
        return False
    if op.scheme != "https" or not op.hostname:
        return False
    if op.hostname.lower() != (ep.hostname or "").lower():
        return False
    # Compare effective ports, normalizing the https default (443) so an explicit
    # ":443" matches an absent port. A same-host but attacker-port URL is rejected.
    op_port = op.port if op.port is not None else 443
    ep_port = ep.port if ep.port is not None else 443
    return op_port == ep_port


def operation_id_from_url(operation_url: str) -> str | None:
    """Extract the result id from .../analyzeResults/{id}[?...], or None when absent."""
    path = urlparse(operation_url).path
    segments = [segment for segment in path.split("/") if segment]
    for index, segment in enumerate(segments):
        if segment == "analyzeResults" and index + 1 < len(segments):
            return segments[index + 1]
    return None
