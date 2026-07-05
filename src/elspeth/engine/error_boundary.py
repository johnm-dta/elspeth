"""Shared exception policy for engine-owned execution boundaries."""

from __future__ import annotations

import elspeth.contracts.errors as contract_errors
from elspeth.contracts.errors import GracefulShutdownError

_PROGRAMMING_ERRORS: tuple[type[Exception], ...] = (
    TypeError,
    AttributeError,
    NotImplementedError,
    AssertionError,
    NameError,
    KeyError,
    RecursionError,
)


def reraise_if_engine_crash_through(exc: BaseException) -> None:
    """Re-raise errors that must not be wrapped as operational failures."""
    if not isinstance(exc, Exception):
        raise exc
    if isinstance(exc, contract_errors.TIER_1_ERRORS):
        raise exc
    if isinstance(exc, GracefulShutdownError):
        raise exc
    if isinstance(exc, _PROGRAMMING_ERRORS):
        raise exc
