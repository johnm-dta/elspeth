"""Boundary tests for DAG wiring-only types."""

from __future__ import annotations

import importlib


def test_wired_transform_lives_in_wiring_module_not_dag_models_or_facades() -> None:
    wiring = importlib.import_module("elspeth.core.dag.wiring")
    dag_models = importlib.import_module("elspeth.core.dag.models")
    dag_facade = importlib.import_module("elspeth.core.dag")
    core_facade = importlib.import_module("elspeth.core")

    assert hasattr(wiring, "WiredTransform")
    assert not hasattr(dag_models, "WiredTransform")
    assert "WiredTransform" not in dag_facade.__all__
    assert not hasattr(dag_facade, "WiredTransform")
    assert "WiredTransform" not in core_facade.__all__
    assert not hasattr(core_facade, "WiredTransform")
