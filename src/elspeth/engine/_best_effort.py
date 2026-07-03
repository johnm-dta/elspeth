"""Best-effort post-audit ceremony helper.

This module provides a single context manager, ``best_effort``, that runs
post-audit ceremony or telemetry without masking an in-flight exception.

It exists so that 18+ scattered ``try/except Exception:`` blocks across
``engine/processor.py`` and ``engine/orchestrator/core.py`` — every one of
which followed the same pattern — can collapse to one reviewable surface.
The single broad-catch in this module carries one tier-model allowlist
entry instead of the previous fan-out.

Per CLAUDE.md logging-telemetry-policy primacy (audit > telemetry > logger):
this helper logs at the **logger** layer because the work it wraps is
post-audit ceremony or telemetry, and any failure here means the ceremony
itself (the secondary channel) is the failing system. Logger is the
recovery path of last resort. It logs exception classes only; raw ceremony
exception strings can carry SQL parameters, exporter internals, paths, or
provider details that do not belong in structured logs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

import structlog

_slog = structlog.get_logger(__name__)


@contextmanager
def best_effort(operation: str, /, **context: Any) -> Iterator[None]:
    """Run post-audit ceremony/telemetry without masking an in-flight exception.

    Use ONLY in error-handling paths where:
      1. An audit-load-bearing event has already been recorded (or an audit-
         load-bearing exception is propagating to the caller).
      2. The body's work is post-audit ceremony, telemetry, or cleanup that
         must attempt completion but must NEVER replace the in-flight event.
      3. Failure of the body must be logged for operator visibility but must
         not propagate.

    Args:
        operation: Short label describing the ceremony work, e.g.
            ``"interrupted ceremony"`` or ``"TokenCompleted telemetry after
            FAILED audit"``. Logged on failure for triage.
        **context: Structured fields (run_id, token_id, transform_node_id, …)
            attached to the failure log entry.

    Why a single broad-except is the right shape here:
        Any Exception subclass during ceremony is by definition a
        ceremony-system failure. The original event has already been
        recorded; preserving it is more important than discriminating on
        the ceremony failure type. CLAUDE.md "Plugin Ownership" — broad
        catches are forbidden for plugin/engine bugs in the primary
        execution path, but post-audit ceremony is *secondary* to the
        primary event.

    EXCEPTION — Tier-1 always escapes: audit-integrity / framework-invariant
    errors (``TIER_1_ERRORS``) re-raise before the broad catch. Audit
    corruption during ceremony outranks even the in-flight primary event
    (errors.py policy: ``except TIER_1_ERRORS: raise`` before any
    ``except Exception``). Tier-2 coordination refusals
    (``RunLeadershipLostError``, ``RunWorkerEvictedError``) are NOT Tier-1
    and stay suppressed — the finalize call sites rely on that.
    """
    # Live attribute access of the lazily materialized TIER_1_ERRORS tuple —
    # never a from-import snapshot (which would capture an empty/stale tuple
    # and let Tier-1 errors fall through to the broad suppression below).
    import elspeth.contracts.errors as contract_errors

    try:
        yield
    except contract_errors.TIER_1_ERRORS:
        raise
    except Exception as ceremony_failure:
        fields = {
            **context,
            "operation": operation,
            "error_type": type(ceremony_failure).__name__,
        }
        with suppress(Exception):
            _slog.warning(
                "best-effort ceremony failed during error propagation; original event preserved",
                **fields,
            )


__all__ = ["best_effort"]
