"""Focused lifecycle wiring tests for post-run audit export resources."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from elspeth.contracts.audit_export import AuditExportContentStoreResolver
from elspeth.engine.orchestrator.run_lifecycle import RunLifecycleCoordinator


def test_execute_export_phase_forwards_explicit_payload_and_content_stores() -> None:
    payload_store = object()
    content_store = object()
    content_store_resolver = AuditExportContentStoreResolver()
    settings = SimpleNamespace(
        landscape=SimpleNamespace(export=SimpleNamespace(sink="archive", format="json")),
    )
    factory = SimpleNamespace(run_lifecycle=SimpleNamespace(set_export_status=lambda *_args, **_kwargs: None))
    binding = object()
    admission = object()
    coordinator = object.__new__(RunLifecycleCoordinator)
    coordinator._db = object()
    coordinator._events = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    coordinator._ceremony = SimpleNamespace(
        emit_telemetry=lambda *_args, **_kwargs: None,
        emit_phase_error=lambda *_args, **_kwargs: None,
    )

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
            lambda _name: binding,
            payload_store=payload_store,
            audit_export_content_store=content_store,
            audit_export_content_store_resolver=content_store_resolver,
            worker_id="worker:run-1:unique",
        )

    assert export.call_args.kwargs["payload_store"] is payload_store
    assert export.call_args.kwargs["audit_export_content_store"] is content_store
    assert export.call_args.kwargs["audit_export_content_store_resolver"] is content_store_resolver
    assert export.call_args.kwargs["worker_id"] == "worker:run-1:unique"


class _DatabasePhaseReached(Exception):
    """Sentinel: the DATABASE phase seam was invoked."""


def _export_enabled_settings() -> SimpleNamespace:
    return SimpleNamespace(landscape=SimpleNamespace(export=SimpleNamespace(enabled=True)))


_EXPORT_RESOURCES: dict[str, object] = {
    "sink_factory": lambda _name: object(),
    "audit_export_content_store": object(),
    "audit_export_content_store_resolver": AuditExportContentStoreResolver(),
}


@pytest.mark.parametrize(
    ("omitted", "match"),
    [
        ("sink_factory", "no sink_factory was provided"),
        ("audit_export_content_store", "no audit_export_content_store was provided"),
        ("audit_export_content_store_resolver", "no audit_export_content_store_resolver was provided"),
    ],
)
def test_run_validates_export_resources_before_any_irreversible_work(omitted: str, match: str) -> None:
    """elspeth-749e75a59b: export-enabled runs must fail fast on missing export
    resources BEFORE the run is bootstrapped, processed, finalized, its
    checkpoints deleted, or its leader seat released — not after."""
    coordinator = object.__new__(RunLifecycleCoordinator)
    coordinator._checkpoints = SimpleNamespace(reset_sequence=lambda: None)
    initialize_database_phase = Mock(side_effect=_DatabasePhaseReached)
    execute_run = Mock(side_effect=AssertionError("run body must never start"))
    resources = dict(_EXPORT_RESOURCES)
    resources[omitted] = None

    with (
        patch("elspeth.engine.orchestrator.run_lifecycle.prepare_for_run"),
        pytest.raises(ValueError, match=match),
    ):
        coordinator.run(
            SimpleNamespace(),
            SimpleNamespace(),
            _export_enabled_settings(),
            payload_store=object(),
            openrouter_catalog_sha256="catalog-sha",
            openrouter_catalog_source="catalog-source",
            initialize_database_phase=initialize_database_phase,
            execute_run=execute_run,
            **resources,  # type: ignore[arg-type]
        )

    initialize_database_phase.assert_not_called()
    execute_run.assert_not_called()


def test_run_with_export_disabled_does_not_require_export_resources() -> None:
    """Control: export-disabled runs keep working without export resources
    (validation stays scoped to export-enabled runs)."""
    coordinator = object.__new__(RunLifecycleCoordinator)
    coordinator._checkpoints = SimpleNamespace(reset_sequence=lambda: None)
    initialize_database_phase = Mock(side_effect=_DatabasePhaseReached)
    execute_run = Mock(side_effect=AssertionError("run body must never start"))
    settings = SimpleNamespace(landscape=SimpleNamespace(export=SimpleNamespace(enabled=False)))

    with (
        patch("elspeth.engine.orchestrator.run_lifecycle.prepare_for_run"),
        pytest.raises(_DatabasePhaseReached),
    ):
        coordinator.run(
            SimpleNamespace(),
            SimpleNamespace(),
            settings,
            payload_store=object(),
            openrouter_catalog_sha256="catalog-sha",
            openrouter_catalog_source="catalog-source",
            initialize_database_phase=initialize_database_phase,
            execute_run=execute_run,
        )

    initialize_database_phase.assert_called_once()
