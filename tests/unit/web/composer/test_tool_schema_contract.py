"""Directional compatibility tests for the planner's canonical schema."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from elspeth.contracts.blobs import ALLOWED_MIME_TYPES
from elspeth.web.composer.redaction import SetPipelineArgumentsModel
from elspeth.web.composer.tools._dispatch import get_tool_definitions


def _registered_set_pipeline_schema() -> dict[str, Any]:
    return next(definition["parameters"] for definition in get_tool_definitions() if definition["name"] == "set_pipeline")


_BASE_PIPELINE: dict[str, Any] = {
    "source": {"plugin": "csv", "on_success": "rows"},
    "nodes": [],
    "edges": [],
    "outputs": [],
}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("metadata",), None),
        (("source",), None),
        (("sources",), None),
        (("source", "blob_id"), None),
        (("source", "on_validation_failure"), None),
        (("source", "inline_blob"), None),
        (("nodes",), [{"id": "gate", "node_type": "gate", "input": "rows", "plugin": None}]),
        (
            ("nodes",),
            [
                {
                    "id": "aggregate",
                    "node_type": "aggregation",
                    "input": "rows",
                    "plugin": "batch_stats",
                    "on_success": None,
                    "on_error": None,
                    "condition": None,
                    "routes": None,
                    "fork_to": None,
                    "branches": None,
                    "policy": None,
                    "merge": None,
                    "trigger": {"count": None, "timeout_seconds": None, "condition": None},
                    "output_mode": None,
                    "expected_output_count": None,
                }
            ],
        ),
        (
            ("edges",),
            [{"id": "edge", "from_node": "source", "to_node": "sink", "edge_type": "on_success", "label": None}],
        ),
        (("outputs",), [{"sink_name": "rows", "plugin": "json", "on_write_failure": None}]),
        (("metadata",), {"name": None, "description": None}),
    ],
)
def test_registered_schema_advertises_explicit_null_for_omission_equivalent_fields(
    path: tuple[str, ...],
    value: object,
) -> None:
    payload = deepcopy(_BASE_PIPELINE)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    SetPipelineArgumentsModel.model_validate(payload)

    errors = list(Draft202012Validator(_registered_set_pipeline_schema()).iter_errors(payload))
    assert errors == []


def test_inline_blob_mime_types_share_the_blob_contract_closed_set() -> None:
    schema = _registered_set_pipeline_schema()
    advertised = schema["properties"]["source"]["properties"]["inline_blob"]["properties"]["mime_type"]["enum"]
    assert set(advertised) == ALLOWED_MIME_TYPES

    for mime_type in sorted(ALLOWED_MIME_TYPES):
        payload = deepcopy(_BASE_PIPELINE)
        payload["source"]["inline_blob"] = {
            "filename": "input.txt",
            "mime_type": mime_type,
            "content": "one\n",
        }
        SetPipelineArgumentsModel.model_validate(payload)

    unsupported = deepcopy(_BASE_PIPELINE)
    unsupported["source"]["inline_blob"] = {
        "filename": "input.xml",
        "mime_type": "application/xml",
        "content": "<one />",
    }
    with pytest.raises(ValueError):
        SetPipelineArgumentsModel.model_validate(unsupported)


def test_canonical_schema_accessor_returns_isolated_registered_copies() -> None:
    spec = importlib.util.find_spec("elspeth.web.composer.tools.schema_contract")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    first = module.canonical_set_pipeline_schema()
    second = module.canonical_set_pipeline_schema()
    registered = _registered_set_pipeline_schema()
    assert first == second == registered

    first["required"].append("attacker_added")
    first["properties"]["source"]["properties"]["plugin"]["type"] = "integer"

    assert module.canonical_set_pipeline_schema() == registered
