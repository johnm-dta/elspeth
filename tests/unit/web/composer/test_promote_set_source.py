"""ARG_ERROR routing for set_source ToolArgumentError (spec §11 done-when).

Rev-2 BLOCKER_A: promoted handlers MUST catch ``pydantic.ValidationError``
and re-raise as :class:`ToolArgumentError`.  The compose loop's
``ToolArgumentError`` handler at ``service.py:2480`` routes to ARG_ERROR.
A bare ``ValidationError`` escaping the handler hits ``service.py:2564``
(→ :class:`ComposerPluginCrashError` → HTTP 500) — the wrong disposition
for Tier-3 input.

This test pins the exception-class and ``__cause__`` chain so a future
refactor that drops the ``try/except PydanticValidationError`` wrapper or
the ``from exc`` clause fails immediately rather than silently regressing
the routing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError as PydanticValidationError

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import (
    ConfigFieldSummary,
    PluginSummary,
)
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import _execute_set_source
from elspeth.web.composer.tools._common import ToolContext


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
    """Minimal CatalogService mock — only ``list_sources`` is reached by
    the path under test, and only the ``csv`` plugin name is referenced.
    """
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(
            name="csv",
            description="CSV file source",
            plugin_type="source",
            config_fields=[
                ConfigFieldSummary(
                    name="path",
                    type="string",
                    required=True,
                    description="File path",
                    default=None,
                ),
            ],
        ),
    ]
    return catalog


class TestPromoteSetSourceArgErrorRouting:
    def test_empty_arguments_raise_tool_argument_error(self) -> None:
        """A bare ``{}`` is missing all four required fields."""
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_source({}, _empty_state(), ToolContext(catalog=_mock_catalog()))
        # __cause__ chain MUST preserve the underlying ValidationError
        # so auditors can inspect missing fields without the LLM-facing
        # message exposing the raw Tier-3 argument values.
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_wrong_type_raises_tool_argument_error(self) -> None:
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_source(
                {
                    "plugin": 42,
                    "options": {},
                    "on_success": "rows",
                    "on_validation_failure": "discard",
                },
                _empty_state(),
                ToolContext(catalog=_mock_catalog()),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_extra_field_raises_tool_argument_error(self) -> None:
        """extra='forbid' on the model rejects stray fields at Tier-3."""
        with pytest.raises(ToolArgumentError) as exc_info:
            _execute_set_source(
                {
                    "plugin": "csv",
                    "options": {"path": "/tmp/x.csv"},
                    "on_success": "rows",
                    "on_validation_failure": "discard",
                    "inline_blob": {"foo": "bar"},
                },
                _empty_state(),
                ToolContext(catalog=_mock_catalog()),
            )
        assert isinstance(exc_info.value.__cause__, PydanticValidationError)

    def test_valid_arguments_dispatch_normally(self) -> None:
        result = _execute_set_source(
            {
                "plugin": "csv",
                "options": {"path": "/tmp/x.csv", "schema": {"mode": "observed"}},
                "on_success": "rows",
                "on_validation_failure": "discard",
            },
            _empty_state(),
            ToolContext(catalog=_mock_catalog()),
        )
        assert result.success is True
        assert result.updated_state.sources["source"].plugin == "csv"
