"""Single source of truth for terminal-pair counter effects (elspeth-feeb4482fc).

Three sites previously each maintained their own switch over the
``(TerminalOutcome, TerminalPath)`` vocabulary — the live accumulator
(``outcomes.accumulate_row_outcomes``), the sink-write diversion reconciler
(``outcomes.reconcile_sink_write_diversions``), and the audit derive
(``run_status.derive_terminal_status_from_audit``) — so a terminal-pair
semantic change had to be hand-synced across all three or
``assert_terminal_counter_parity`` would crash the run. The COUNTER EFFECT of
each pair now lives here as data; the three sites keep only their
branch-specific side effects (sink routing, error-hash construction, Tier-1
invariant checks, audit-only discriminators).

The legal pair SET is owned by ``contracts.enums._LEGAL_TERMINAL_PAIRS``;
import-time guards below keep this table in lockstep with it, mirroring the
exhaustiveness assertions in ``contracts/enums.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

from elspeth.contracts.enums import _LEGAL_TERMINAL_PAIRS, TerminalOutcome, TerminalPath
from elspeth.engine.orchestrator.types import ExecutionCounters

#: Key vocabulary: every legal terminal pair plus the single non-terminal
#: buffered marker ``(None, BUFFERED)`` (outcome deferred to flush time).
TerminalPairKey = tuple[TerminalOutcome | None, TerminalPath]


@dataclass(frozen=True, slots=True)
class TerminalPairCounterEffect:
    """Counter effect of one terminal pair on :class:`ExecutionCounters`.

    ``increments`` names the ``ExecutionCounters`` fields bumped by +1 per
    record. ``counts_routed_destination`` additionally tallies
    ``routed_destinations[sink_name]``. ``routes_to_sink`` marks pairs the
    processing loop routes into ``pending_tokens`` via ``_route_to_sink``
    (quarantine is sink-reconcilable but reaches its sink on a different
    path, so the flags are independent). ``sink_reconcilable`` marks pairs
    whose provisional counts are subtracted when a sink write reveals
    diversions. ``forbidden_in_processing_results`` marks pairs that must
    never appear in processing-loop results (their live counting belongs to
    ``SinkExecutor``); the audit derive still applies their increments.
    """

    increments: tuple[str, ...]
    counts_routed_destination: bool = False
    routes_to_sink: bool = False
    sink_reconcilable: bool = False
    forbidden_in_processing_results: bool = False


TERMINAL_PAIR_COUNTER_EFFECTS: Mapping[TerminalPairKey, TerminalPairCounterEffect] = MappingProxyType(
    {
        (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW): TerminalPairCounterEffect(
            increments=("rows_succeeded",),
            routes_to_sink=True,
            sink_reconcilable=True,
        ),
        (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED): TerminalPairCounterEffect(
            increments=("rows_succeeded", "rows_routed_success"),
            counts_routed_destination=True,
            routes_to_sink=True,
            sink_reconcilable=True,
        ),
        (TerminalOutcome.SUCCESS, TerminalPath.GATE_DISCARDED): TerminalPairCounterEffect(
            increments=("rows_succeeded",),
        ),
        (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED): TerminalPairCounterEffect(
            increments=("rows_succeeded",),
        ),
        (TerminalOutcome.SUCCESS, TerminalPath.COALESCED): TerminalPairCounterEffect(
            increments=("rows_succeeded", "rows_coalesced"),
            routes_to_sink=True,
            sink_reconcilable=True,
        ),
        (TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED): TerminalPairCounterEffect(
            increments=("rows_failed", "rows_routed_failure"),
            counts_routed_destination=True,
            routes_to_sink=True,
            sink_reconcilable=True,
        ),
        (TerminalOutcome.FAILURE, TerminalPath.UNROUTED): TerminalPairCounterEffect(
            increments=("rows_failed",),
        ),
        (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE): TerminalPairCounterEffect(
            increments=("rows_failed", "rows_quarantined"),
            sink_reconcilable=True,
        ),
        (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED): TerminalPairCounterEffect(
            increments=("rows_failed", "rows_diverted"),
            forbidden_in_processing_results=True,
        ),
        (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK): TerminalPairCounterEffect(
            increments=("rows_diverted",),
            forbidden_in_processing_results=True,
        ),
        (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT): TerminalPairCounterEffect(
            # Parent tokens delegate predicate counters to their children;
            # the structural fork count belongs to the parent record.
            increments=("rows_forked",),
        ),
        (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT): TerminalPairCounterEffect(
            increments=("rows_expanded",),
        ),
        (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED): TerminalPairCounterEffect(
            # No dedicated RunResult counter; the BUFFERED record captures
            # the structural row and the flush result carries the outcome.
            increments=(),
        ),
        (None, TerminalPath.BUFFERED): TerminalPairCounterEffect(
            # Non-terminal: token accepted into an aggregation buffer;
            # terminal outcome deferred to flush time.
            increments=("rows_buffered",),
        ),
    }
)


def apply_counter_increments(counters: ExecutionCounters, effect: TerminalPairCounterEffect) -> None:
    """Apply one record's +1 counter increments (routed_destinations excluded).

    ``routed_destinations`` needs the record's sink name and stays at the call
    site, gated on ``effect.counts_routed_destination``.
    """
    for field_name in effect.increments:
        setattr(counters, field_name, getattr(counters, field_name) + 1)


# ---------------------------------------------------------------------------
# Import-time lockstep guards (mirroring contracts/enums.py exhaustiveness).
# ---------------------------------------------------------------------------

_terminal_keys = frozenset(key for key in TERMINAL_PAIR_COUNTER_EFFECTS if key[0] is not None)
if _terminal_keys != _LEGAL_TERMINAL_PAIRS:
    _missing = _LEGAL_TERMINAL_PAIRS - _terminal_keys
    _extra = _terminal_keys - _LEGAL_TERMINAL_PAIRS
    raise AssertionError(
        f"TERMINAL_PAIR_COUNTER_EFFECTS is out of lockstep with contracts.enums._LEGAL_TERMINAL_PAIRS: "
        f"missing={sorted(str(k) for k in _missing)}, extra={sorted(str(k) for k in _extra)}. "
        "Every legal terminal pair must have exactly one counter-effect entry."
    )

_non_terminal_keys = frozenset(key for key in TERMINAL_PAIR_COUNTER_EFFECTS if key[0] is None)
if _non_terminal_keys != frozenset({(None, TerminalPath.BUFFERED)}):
    raise AssertionError(
        f"TERMINAL_PAIR_COUNTER_EFFECTS non-terminal keys must be exactly {{(None, BUFFERED)}}; got {sorted(str(k) for k in _non_terminal_keys)}."
    )

_counter_field_names = frozenset(f.name for f in fields(ExecutionCounters) if f.name != "routed_destinations")
for _key, _effect in TERMINAL_PAIR_COUNTER_EFFECTS.items():
    _unknown = frozenset(_effect.increments) - _counter_field_names
    if _unknown:
        raise AssertionError(f"TERMINAL_PAIR_COUNTER_EFFECTS[{_key!r}] names non-existent ExecutionCounters field(s): {sorted(_unknown)}.")
