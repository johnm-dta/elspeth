"""Shared composer-service test helpers.

These were originally module-private in ``test_service.py``; sibling test
files (``test_provider_cache_markers.py``, ``test_compose_loop_envelope.py``)
imported them via ``from tests.unit.web.composer.test_service import ...``
which is fragile cross-test-file coupling — a rename or refactor inside
``test_service.py`` silently breaks unrelated modules. Extracted here per
elspeth-59cdfcaf67.

The leading underscore on the filename (``_helpers.py``) prevents pytest's
default collection from treating this as a test module — it carries
fixtures-by-function and dataclasses, no ``test_*`` callables.

Distinct from ``tests/unit/web/composer/conftest.py``, which also defines
``_mock_catalog`` and ``_make_settings`` helpers — those have a different
shape (``_make_settings`` requires a positional ``data_dir``; the conftest's
``_mock_catalog`` returns a similar-but-not-identical ``MagicMock``) and
serve the phase-3 compose-loop harness specifically. Renaming to merge
would change call-sites in ~30 tests across both surfaces; keeping the two
surfaces separate (a) preserves the conftest's auto-magic fixture injection
for the phase-3 tests and (b) keeps these helpers as plain callables that
``from ... import`` users can read at the call site.

Test-dispatch convention (I7 — test-analyst review remediation)
---------------------------------------------------------------
Two distinct invocation paths exist for composer tools, and new tests must
pick the right one deliberately:

1. **Through-dispatch** (``execute_tool(name, args, state, catalog)``) —
   exercises the production dispatch wrapper, which runs
   ``_inject_prior_validation`` and ``_augment_with_plugin_schemas`` around
   the handler.  This is the path the compose loop takes at runtime.
   Canonical examples: ``test_tools.py``, ``test_failure_schema_augmentation.py``.

2. **Handler-direct** (``_execute_<tool>(args, state, ctx)`` with a
   hand-built ``ToolContext``) — calls the handler in isolation, bypassing
   the dispatch wrapper.  The ``test_promote_*.py`` family deliberately
   uses this path because it is targeting the
   ``ValidationError`` → ``ToolArgumentError`` re-raise contract inside
   each promoted handler (rev-2 BLOCKER_A — a bare ``ValidationError``
   escaping the handler routes to ``ComposerPluginCrashError`` /
   HTTP 500 instead of ARG_ERROR).  Going through ``execute_tool`` would
   couple the contract test to dispatch-wrapper behaviour and obscure the
   exception-class invariant under test.

If you are writing a new test and want catalog augmentation or
prior-validation injection covered, use path 1.  If you are pinning a
handler-level invariant (argument validation, exception chaining,
post-mutation state shape), use path 2 — but be aware that
``_inject_prior_validation`` / ``_augment_with_plugin_schemas`` will NOT
run, so any assertion that depends on their effects must use path 1.
Don't copy the handler-direct pattern from a ``test_promote_*.py`` file
without first checking that your assertion does not depend on dispatch.

Catalog mock convention
-----------------------
Every ``MagicMock`` standing in for ``CatalogService`` MUST be constructed
with ``spec=CatalogService`` (see ``_mock_catalog`` below).  Unspecced
catalog mocks happily respond to any attribute access, so a rename or
removal of a real ``CatalogService`` method silently passes through tests
while production crashes.  ``spec=CatalogService`` causes ``AttributeError``
the instant a test tries to set a return value on a method that no longer
exists — i.e. the test breaks at the boundary where the schema drift
actually happened, not in some distant assertion (I8 — test-analyst
review remediation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.config import WebSettings


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeToolCall] | None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeLLMResponse:
    choices: list[FakeChoice]


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _mock_catalog() -> MagicMock:
    """Mock CatalogService with real PluginSummary/PluginSchemaInfo instances.

    AC #16: Tests must use real PluginSummary and PluginSchemaInfo instances,
    not plain dicts. Mock return types must match the CatalogService protocol.
    """
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(
            name="csv",
            description="CSV source",
            plugin_type="source",
            config_fields=[],
        ),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(
            name="passthrough",
            description="Uppercase",
            plugin_type="transform",
            config_fields=[],
        ),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(
            name="csv",
            description="CSV sink",
            plugin_type="sink",
            config_fields=[],
        ),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"title": "Config", "properties": {}},
        knob_schema={"fields": []},
    )
    return catalog


def _make_llm_response(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> FakeLLMResponse:
    """Build a typed fake LiteLLM response.

    Uses typed dataclasses instead of MagicMock so tests fail if production
    code accesses an attribute that doesn't exist on the real response shape.
    """
    fake_tool_calls: list[FakeToolCall] | None = None
    if tool_calls:
        fake_tool_calls = [
            FakeToolCall(
                id=tc["id"],
                function=FakeFunction(
                    name=tc["name"],
                    arguments=json.dumps(tc["arguments"]),
                ),
            )
            for tc in tool_calls
        ]

    message = FakeMessage(content=content, tool_calls=fake_tool_calls)
    return FakeLLMResponse(choices=[FakeChoice(message=message)])


def _make_settings(**overrides: Any) -> WebSettings:
    """Build WebSettings with Pydantic-enforced defaults.

    Use keyword arguments to override specific fields for a test.
    Defaults come from the Pydantic model — no drift possible.

    data_dir defaults to /data (absolute) so test paths like
    /data/blobs/file.csv pass S2 path validation.
    """
    defaults: dict[str, Any] = {
        "data_dir": Path("/data"),
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }
    defaults.update(overrides)
    return WebSettings(**defaults)
