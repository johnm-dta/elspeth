"""Tests for shared engine exception-boundary policy."""

from __future__ import annotations

import pytest

from elspeth.contracts.errors import AuditIntegrityError, GracefulShutdownError
from elspeth.engine.error_boundary import reraise_if_engine_crash_through


def _shutdown_error() -> GracefulShutdownError:
    return GracefulShutdownError(rows_processed=0, run_id="run-1")


@pytest.mark.parametrize(
    "exc",
    [
        KeyboardInterrupt(),
        _shutdown_error(),
        AuditIntegrityError("audit corrupt"),
        TypeError("bad type"),
        AttributeError("missing attr"),
        NotImplementedError("not implemented"),
        AssertionError("assert failed"),
        NameError("missing name"),
        KeyError("missing key"),
        RecursionError("recursive"),
    ],
)
def test_crash_through_errors_are_reraised(exc: BaseException) -> None:
    with pytest.raises(type(exc)):
        reraise_if_engine_crash_through(exc)


def test_operational_exception_is_left_for_caller_wrapper() -> None:
    assert reraise_if_engine_crash_through(RuntimeError("service unavailable")) is None
