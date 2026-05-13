# tests/unit/contracts/transform_contracts/_azure_batch_helpers.py
"""Shared helpers for Azure batch transform contract tests.

Three Azure transform contract test files (content_safety, prompt_shield,
multi_query) share the same BatchTransformMixin contract surface. The
generic plumbing — submit a row, wait for the OutputPort, collect a single
TransformResult — was triplicated. This module hosts the common helpers.

Per-service response shape factories deliberately remain in their respective
test files because they encode each Azure service's distinct contract
surface (severity threshold / dual attack types / JSON parse failure). Do
not move them here.

Leading underscore on the module name marks it as a test-internal helper
(pytest will not collect it as a test module).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import httpx

from elspeth.contracts import TransformResult
from elspeth.plugins.infrastructure.batching.mixin import BatchTransformMixin
from elspeth.testing import make_pipeline_row

from .test_batch_transform_protocol import CollectingOutputPort

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import PluginContext

# Bound at import time so the spec_set target is the real httpx.Client class
# rather than a string lookup against a possibly-patched module attribute.
_HTTPX_CLIENT_CLASS = httpx.Client


def create_http_response(
    response_data: dict[str, Any],
    *,
    url: str,
    status_code: int = 200,
) -> httpx.Response:
    """Build an httpx.Response(status, json=...) bound to a POST request at ``url``.

    Used by tests that need to override the per-call response on the mocked
    httpx.Client to exercise specific Azure-service response shapes.
    """
    return httpx.Response(
        status_code,
        json=response_data,
        request=httpx.Request("POST", url),
    )


@contextmanager
def patch_httpx_client_with_default(
    default_response_factory: Callable[[], dict[str, Any]],
) -> Iterator[MagicMock]:
    """Patch httpx.Client; mocked client.post returns ``default_response_factory()``.

    Yields the patched class (``mock_client_class``). Per-test overrides should
    update ``mock_client_class.return_value.post.side_effect`` with a new
    callable — typically built around :func:`create_http_response`.

    The default factory is a callable rather than a precomputed dict so each
    request returns a fresh response object (httpx.Response is single-use for
    body-stream consumption in some integrations).
    """
    with patch("httpx.Client", autospec=True) as mock_client_class:
        mock_client_instance = MagicMock(spec_set=_HTTPX_CLIENT_CLASS)

        def _mocked_post(url: str, **_: object) -> httpx.Response:
            return create_http_response(default_response_factory(), url=url)

        mock_client_instance.post.side_effect = _mocked_post
        mock_client_class.return_value = mock_client_instance
        yield mock_client_class


def set_httpx_response(
    mock_client_class: MagicMock,
    response_data: dict[str, Any],
    *,
    status_code: int = 200,
) -> None:
    """Override the mocked httpx.Client to return ``response_data`` for every POST.

    Use inside a test after the autouse fixture has patched httpx.Client, when
    the test wants a specific response shape rather than the fixture's default.
    """
    mock_client_instance = mock_client_class.return_value

    def _mocked_post(url: str, **_: object) -> httpx.Response:
        return create_http_response(response_data, url=url, status_code=status_code)

    mock_client_instance.post.side_effect = _mocked_post


def submit_and_collect_single_result(
    started_transform: BatchTransformMixin,
    row_data: dict[str, Any],
    ctx: PluginContext,
    output_port: CollectingOutputPort,
) -> TransformResult:
    """Submit one row through a started batch transform, return the single result.

    Wraps the common pattern: ``accept(row, ctx)`` then wait on the OutputPort
    for exactly one result, asserting both arrival and shape (one
    ``TransformResult`` tuple). Used by Azure batch contract tests that
    exercise per-service response handling on a single submission.
    """
    started_transform.accept(make_pipeline_row(row_data), ctx)

    arrived = output_port.wait_for_results(1, timeout=10.0)
    assert arrived, "Result did not arrive via OutputPort within timeout"

    results = output_port.get_results()
    assert len(results) == 1
    _token, result, _state_id = results[0]
    assert isinstance(result, TransformResult)
    return result
