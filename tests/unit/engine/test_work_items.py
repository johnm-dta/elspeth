"""Unit tests for WorkItem cursor objects and factory helpers."""

from __future__ import annotations

from elspeth.engine.dag_navigator import DAGNavigator
from elspeth.engine.work_items import WorkItem, WorkItemFactory


def test_work_item_boundary_is_outside_dag_navigator() -> None:
    assert WorkItem.__module__ == "elspeth.engine.work_items"
    assert WorkItemFactory.__module__ == "elspeth.engine.work_items"
    assert not hasattr(DAGNavigator, "create_work_item")
    assert not hasattr(DAGNavigator, "create_continuation_work_item")
