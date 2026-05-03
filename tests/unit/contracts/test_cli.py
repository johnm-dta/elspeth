"""Contract tests for ``elspeth.contracts.cli``.

Pins the structural invariants on the engine-side ``ProgressEvent``
dataclass — distinct from the wire-side ``ProgressData`` Pydantic model
(see ``tests/unit/web/execution/test_schemas.py``).  The two halves
together close the producer-drift door S-8 identified: the consumer
schema requires every counter, AND the producer dataclass requires every
counter, so a future emitter cannot silently substitute ``0`` anywhere
along the path from ``orchestrator/core.py`` to the WebSocket.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.cli import ProgressEvent


class TestS8ProgressEventFabricationGuard:
    """S-8 sibling fix: engine ``ProgressEvent`` requires all counters.

    The pre-fix dataclass had ``rows_routed_success: int = 0`` and
    ``rows_routed_failure: int = 0`` as default values — explicitly added
    on 2026-05-03 (commit e8c9fbff4) as a CLAUDE.md-violation repair shim
    with the docstring stating "engine emitters MUST pass real values".
    Removing the defaults completes that migration: the constraint is now
    type-enforced at the producer site rather than relying on emitter
    discipline.
    """

    _REQUIRED_FIELDS = (
        "rows_processed",
        "rows_succeeded",
        "rows_failed",
        "rows_quarantined",
        "rows_routed_success",
        "rows_routed_failure",
        "elapsed_seconds",
    )

    def test_progress_event_with_all_fields_constructs(self) -> None:
        """Happy path — every counter explicitly supplied."""
        event = ProgressEvent(
            rows_processed=100,
            rows_succeeded=80,
            rows_failed=5,
            rows_quarantined=10,
            rows_routed_success=3,
            rows_routed_failure=2,
            elapsed_seconds=12.5,
        )
        assert event.rows_processed == 100
        assert event.rows_succeeded == 80
        assert event.rows_failed == 5
        assert event.rows_quarantined == 10
        assert event.rows_routed_success == 3
        assert event.rows_routed_failure == 2

    @pytest.mark.parametrize("missing_field", _REQUIRED_FIELDS)
    def test_progress_event_omitting_any_field_raises(self, missing_field: str) -> None:
        """Omitting any required dataclass field crashes at construction.

        With ``@dataclass(frozen=True, slots=True)`` and no defaults, the
        generated ``__init__`` raises ``TypeError`` for missing positional
        arguments.  This is the structural defense against producer drift:
        no future emitter can silently substitute ``0`` for a counter
        because the constructor itself rejects the omission.
        """
        kwargs: dict[str, float | int] = {
            "rows_processed": 100,
            "rows_succeeded": 80,
            "rows_failed": 5,
            "rows_quarantined": 10,
            "rows_routed_success": 3,
            "rows_routed_failure": 2,
            "elapsed_seconds": 12.5,
        }
        del kwargs[missing_field]
        with pytest.raises(TypeError, match=missing_field):
            ProgressEvent(**kwargs)  # type: ignore[arg-type]

    def test_progress_event_no_default_zero(self) -> None:
        """No counter silently defaults to 0 — construction with only
        ``rows_processed`` must fail.

        Pre-fix the two routed counters had ``= 0`` defaults; this test
        pins the regression door so a future "convenience default" PR
        is caught at review time.
        """
        with pytest.raises(TypeError):
            ProgressEvent(rows_processed=100)  # type: ignore[call-arg]

    def test_progress_event_is_frozen(self) -> None:
        """Tier 1 producer record — fields must not be re-assignable.

        The dataclass is declared ``frozen=True``; reassigning any field
        raises ``dataclasses.FrozenInstanceError``.  Pinning this here
        protects the audit-record-shape invariant: an emitted event is
        a snapshot, never a mutable handle.
        """
        import dataclasses

        event = ProgressEvent(
            rows_processed=10,
            rows_succeeded=10,
            rows_failed=0,
            rows_quarantined=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            elapsed_seconds=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.rows_processed = 20  # type: ignore[misc]
