"""Focused lifecycle wiring tests for post-run audit export resources."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from elspeth.engine.orchestrator.run_lifecycle import RunLifecycleCoordinator


def test_execute_export_phase_forwards_explicit_payload_and_content_stores() -> None:
    payload_store = object()
    content_store = object()
    settings = SimpleNamespace(
        landscape=SimpleNamespace(export=SimpleNamespace(sink="archive", format="json")),
    )
    factory = SimpleNamespace(run_lifecycle=SimpleNamespace(set_export_status=Mock()))
    binding = object()
    admission = object()
    coordinator = object.__new__(RunLifecycleCoordinator)
    coordinator._db = object()
    coordinator._events = Mock()
    coordinator._ceremony = Mock()

    with (
        patch(
            "elspeth.engine.orchestrator.run_lifecycle.prepare_audit_export_binding",
            return_value=(binding, admission),
        ),
        patch("elspeth.engine.orchestrator.run_lifecycle._validate_audit_export_binding_provenance"),
        patch("elspeth.engine.orchestrator.run_lifecycle.export_landscape") as export,
    ):
        coordinator.execute_export_phase(
            factory,
            "run-1",
            settings,
            Mock(),
            payload_store=payload_store,
            audit_export_content_store=content_store,
        )

    assert export.call_args.kwargs["payload_store"] is payload_store
    assert export.call_args.kwargs["audit_export_content_store"] is content_store
