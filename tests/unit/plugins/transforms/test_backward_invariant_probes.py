from __future__ import annotations

from typing import Any

import pytest

from elspeth.plugins.transforms.batch_replicate import BatchReplicate
from elspeth.plugins.transforms.batch_stats import BatchStats
from elspeth.plugins.transforms.field_mapper import FieldMapper
from elspeth.plugins.transforms.json_explode import JSONExplode
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context


def _emitted_rows(result: Any) -> list[Any]:
    if result.row is not None:
        return [result.row]
    assert result.rows is not None
    return list(result.rows)


@pytest.mark.parametrize(
    ("transform_cls", "required_dropped_field"),
    [
        pytest.param(BatchReplicate, "quarantined_only_marker", id="BatchReplicate-control"),
        pytest.param(BatchStats, "baseline", id="BatchStats"),
        pytest.param(FieldMapper, "field_mapper_probe_source", id="FieldMapper"),
        pytest.param(JSONExplode, "json_explode_items", id="JSONExplode"),
    ],
)
def test_backward_probe_helpers_exercise_a_real_drop_path(
    transform_cls: type[Any],
    required_dropped_field: str,
) -> None:
    transform = transform_cls(transform_cls.probe_config())
    base_row = make_pipeline_row({"baseline": "kept"})

    probe_rows = transform.backward_invariant_probe_rows(base_row)
    result = transform.execute_backward_invariant_probe(probe_rows, make_context())

    assert result.status == "success"
    assert all(required_dropped_field not in row.to_dict() for row in _emitted_rows(result))
