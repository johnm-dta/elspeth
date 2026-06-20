"""Barrier scalar state for post-F1 checkpoint rows.

This module defines the ONLY barrier metadata the checkpoint row carries after
F1 durability unification.  The two trigger latches (count_fire_offset,
condition_fire_offset) and lost_branches records are the only underivable
scalars — everything else (counters, batch ids, buffered tokens) either lives
in journal BLOCKED rows or derives from audit tables at restore time (design D3).

Type hierarchy::

    BarrierScalars                          (top-level — serialized as barrier_scalars_json)
      aggregation: dict[str, AggregationNodeScalars]   keyed by aggregation node_id
      coalesce:    dict[tuple[str,str], CoalescePendingScalars]
                                            keyed by (coalesce_name, row_id)

Wire format (JSON column ``checkpoints.barrier_scalars_json``, epoch 20+)::

    {
        "_version": "1.0",
        "aggregation": {
            "<node_id>": {
                "_version": "1.0",
                "count_fire_offset": <float|null>,
                "condition_fire_offset": <float|null>
            },
            ...
        },
        "coalesce": [
            [[<coalesce_name>, <row_id>], {"_version": "1.0", "lost_branches": {...}}],
            ...
        ]
    }

The ``coalesce`` dict serializes as a **list of [key, value] pairs** rather than
a JSON object because JSON object keys must be strings and the coalesce key is a
two-element ``[name, row_id]`` array.  Both ``name`` and ``row_id`` have no
charset constraint (see ``CoalesceSettings.name`` in ``core/config.py``), so a
delimiter-joined string approach would require escaping and is fragile.

Trust-tier notes
----------------
* ``to_dict()`` — serialization boundary for ``checkpoint_dumps()``.
* ``from_dict()`` — Tier 1 reconstruction.  Checkpoints are our data: crash on
  any structural corruption, no coercion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import freeze_fields

BARRIER_SCALARS_VERSION = "1.0"


def _validate_envelope(data: object, expected_keys: frozenset[str], type_name: str) -> dict[str, Any]:
    """Validate the shared ``from_dict`` envelope: dict shape, exact key set, ``_version``.

    Every barrier-scalars ``from_dict`` uses the same envelope discipline —
    reject non-dicts, unknown keys, missing keys, and version mismatches with
    consistent messages.  ``expected_keys`` must include ``"_version"``.

    Returns:
        ``data`` narrowed to ``dict`` so callers can index it.

    Raises:
        AuditIntegrityError: On any envelope violation.
    """
    if not isinstance(data, dict):
        raise AuditIntegrityError(f"Corrupted {type_name}: top-level value must be a dict, got {type(data).__name__}.")
    unknown = set(data.keys()) - expected_keys
    if unknown:
        raise AuditIntegrityError(f"Corrupted {type_name}: unknown keys {sorted(unknown)}. Expected: {sorted(expected_keys)}.")
    missing = expected_keys - set(data.keys())
    if missing:
        raise AuditIntegrityError(f"Corrupted {type_name}: missing required fields {sorted(missing)}. Found: {sorted(data.keys())}.")
    version = data["_version"]
    if version != BARRIER_SCALARS_VERSION:
        raise AuditIntegrityError(f"Corrupted {type_name}: unsupported version {version!r}. Expected {BARRIER_SCALARS_VERSION!r}.")
    return data


# ---------------------------------------------------------------------------
# AggregationNodeScalars
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AggregationNodeScalars:
    """Scalar checkpoint state for one aggregation node.

    Only the trigger fire-time latches are stored — counters
    (``accepted_count_total``, ``completed_flush_count``) derive from audit
    tables at restore time and are NOT persisted here (design D3).

    Attributes:
        count_fire_offset: Elapsed-seconds latch for the count trigger at the
            time of the last checkpoint, or ``None`` if the count trigger has
            not fired in this batch.
        condition_fire_offset: Elapsed-seconds latch for the condition trigger,
            or ``None`` if not fired.
    """

    count_fire_offset: float | None
    condition_fire_offset: float | None

    def __post_init__(self) -> None:
        if self.count_fire_offset is not None and (not math.isfinite(self.count_fire_offset) or self.count_fire_offset < 0):
            raise ValueError(f"AggregationNodeScalars.count_fire_offset must be non-negative and finite, got {self.count_fire_offset!r}")
        if self.condition_fire_offset is not None and (not math.isfinite(self.condition_fire_offset) or self.condition_fire_offset < 0):
            raise ValueError(
                f"AggregationNodeScalars.condition_fire_offset must be non-negative and finite, got {self.condition_fire_offset!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to checkpoint dict format."""
        return {
            "_version": BARRIER_SCALARS_VERSION,
            "count_fire_offset": self.count_fire_offset,
            "condition_fire_offset": self.condition_fire_offset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregationNodeScalars:
        """Reconstruct from checkpoint dict (Tier 1 — crash on corruption).

        Args:
            data: Node scalars dict from checkpoint.

        Raises:
            AuditIntegrityError: If required keys are missing, unknown keys
                present, ``_version`` does not match, or an offset value has
                the wrong type / is negative / is non-finite.
        """
        data = _validate_envelope(data, frozenset({"_version", "count_fire_offset", "condition_fire_offset"}), "AggregationNodeScalars")
        # Tier 1 value-type guards — corrupted JSON arriving via from_dict is an
        # audit-integrity fault, not a programming error (ValueError stays for
        # direct construction).  bool is an int subclass in Python and
        # math.isfinite(True) passes, so it must be excluded explicitly
        # (cf. require_int doctrine in contracts/freeze.py).
        for field_name in ("count_fire_offset", "condition_fire_offset"):
            value = data[field_name]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AuditIntegrityError(
                    f"Corrupted AggregationNodeScalars: '{field_name}' must be a number or None, got {type(value).__name__}: {value!r}."
                )
            if not math.isfinite(value) or value < 0:
                raise AuditIntegrityError(
                    f"Corrupted AggregationNodeScalars: '{field_name}' must be non-negative and finite, got {value!r}."
                )
        return cls(
            count_fire_offset=data["count_fire_offset"],
            condition_fire_offset=data["condition_fire_offset"],
        )


# ---------------------------------------------------------------------------
# CoalescePendingScalars
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoalescePendingScalars:
    """Scalar checkpoint state for one pending coalesce (coalesce_name, row_id) key.

    Only the lost_branches record is stored — arrived-branch token payloads
    now live in journal BLOCKED rows; counters and state ids derive from audit
    tables at restore (design D3).

    Attributes:
        lost_branches: Branch-name → loss-reason string for every branch that
            was declared lost before the checkpoint.  May be empty.
    """

    lost_branches: Mapping[str, str]

    def __post_init__(self) -> None:
        lost_branches = self.lost_branches
        if not isinstance(lost_branches, Mapping):
            raise TypeError(f"CoalescePendingScalars.lost_branches must be a Mapping, got {type(lost_branches).__name__}")
        freeze_fields(self, "lost_branches")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to checkpoint dict format."""
        return {
            "_version": BARRIER_SCALARS_VERSION,
            "lost_branches": dict(self.lost_branches),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoalescePendingScalars:
        """Reconstruct from checkpoint dict (Tier 1 — crash on corruption).

        Args:
            data: Coalesce pending scalars dict from checkpoint.

        Raises:
            AuditIntegrityError: If required keys are missing, unknown keys
                present, ``_version`` does not match, or any lost_branches
                entry is not str → str.
        """
        data = _validate_envelope(data, frozenset({"_version", "lost_branches"}), "CoalescePendingScalars")
        lost_branches = data["lost_branches"]
        if not isinstance(lost_branches, dict):
            raise AuditIntegrityError(
                f"Corrupted CoalescePendingScalars: 'lost_branches' must be a dict, got {type(lost_branches).__name__}."
            )
        # Tier 1 value-type guards — every entry must be str → str. Anything else
        # (int, None, nested dict) is corruption; without this guard a nested dict
        # would re-serialize silently and propagate through future checkpoints.
        for lb_key, lb_val in lost_branches.items():
            if not isinstance(lb_key, str):
                raise AuditIntegrityError(
                    f"Corrupted CoalescePendingScalars: lost_branches key must be a str, got {type(lb_key).__name__}: {lb_key!r}."
                )
            if not isinstance(lb_val, str):
                raise AuditIntegrityError(
                    f"Corrupted CoalescePendingScalars: lost_branches[{lb_key!r}] must be a str, got {type(lb_val).__name__}: {lb_val!r}."
                )
        return cls(lost_branches=lost_branches)


# ---------------------------------------------------------------------------
# BarrierScalars  (top-level checkpoint value)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BarrierScalars:
    """Full barrier scalar state for a single checkpoint row.

    This is the value stored in ``checkpoints.barrier_scalars_json`` (epoch 20+).
    It carries only the underivable scalars for all aggregation and coalesce
    barriers that were mid-flight at checkpoint time.

    Attributes:
        aggregation: Per-node trigger latch scalars, keyed by aggregation node_id.
        coalesce: Per-pending-key scalars, keyed by ``(coalesce_name, row_id)``
            tuples.  Both components have no charset constraint.

    Properties:
        has_state: ``False`` when both dicts are empty (no in-flight barriers).
    """

    aggregation: Mapping[str, AggregationNodeScalars]
    coalesce: Mapping[tuple[str, str], CoalescePendingScalars]

    def __post_init__(self) -> None:
        aggregation = self.aggregation
        coalesce = self.coalesce
        if not isinstance(aggregation, Mapping):
            raise TypeError(f"BarrierScalars.aggregation must be a Mapping, got {type(aggregation).__name__}")
        if not isinstance(coalesce, Mapping):
            raise TypeError(f"BarrierScalars.coalesce must be a Mapping, got {type(coalesce).__name__}")
        # Validate aggregation keys are non-empty strings
        for key in aggregation:
            if not isinstance(key, str):
                raise TypeError(f"BarrierScalars.aggregation key must be a str, got {type(key).__name__}: {key!r}")
            if not key:
                raise ValueError("BarrierScalars.aggregation key must be non-empty")
        # Validate coalesce keys are (str, str) tuples (runtime guard — from_dict also
        # validates, but direct construction bypasses from_dict).
        raw_key: Any
        for raw_key in coalesce:
            if not isinstance(raw_key, tuple) or len(raw_key) != 2 or not all(isinstance(s, str) for s in raw_key):
                raise TypeError(f"BarrierScalars.coalesce key must be a (str, str) tuple, got {type(raw_key).__name__}: {raw_key!r}")
        freeze_fields(self, "aggregation", "coalesce")

    @property
    def has_state(self) -> bool:
        """True when any in-flight barrier scalars are present."""
        return bool(self.aggregation) or bool(self.coalesce)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to checkpoint column format.

        The ``coalesce`` dict serializes as a list of ``[[name, row_id], scalars]``
        pairs because JSON object keys must be strings and the coalesce key is a
        two-element array.
        """
        return {
            "_version": BARRIER_SCALARS_VERSION,
            "aggregation": {node_id: scalars.to_dict() for node_id, scalars in self.aggregation.items()},
            "coalesce": [[[name, row_id], scalars.to_dict()] for (name, row_id), scalars in self.coalesce.items()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BarrierScalars:
        """Reconstruct from checkpoint column value (Tier 1 — crash on corruption).

        Args:
            data: Top-level barrier scalars dict from the checkpoint column.

        Raises:
            AuditIntegrityError: On structural corruption (missing/unknown keys,
                wrong version, bad coalesce-key shape, duplicate coalesce keys, etc.).
        """
        data = _validate_envelope(data, frozenset({"_version", "aggregation", "coalesce"}), "BarrierScalars")

        # Aggregation dict
        raw_agg = data["aggregation"]
        if not isinstance(raw_agg, dict):
            raise AuditIntegrityError(f"Corrupted BarrierScalars: 'aggregation' must be a dict, got {type(raw_agg).__name__}.")
        aggregation: dict[str, AggregationNodeScalars] = {}
        for node_id, node_data in raw_agg.items():
            if not isinstance(node_data, dict):
                raise AuditIntegrityError(
                    f"Corrupted BarrierScalars: aggregation[{node_id!r}] must be a dict, got {type(node_data).__name__}."
                )
            aggregation[node_id] = AggregationNodeScalars.from_dict(node_data)

        # Coalesce list-of-pairs
        raw_coalesce = data["coalesce"]
        if not isinstance(raw_coalesce, list):
            raise AuditIntegrityError(f"Corrupted BarrierScalars: 'coalesce' must be a list, got {type(raw_coalesce).__name__}.")
        coalesce: dict[tuple[str, str], CoalescePendingScalars] = {}
        for i, entry in enumerate(raw_coalesce):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise AuditIntegrityError(
                    f"Corrupted BarrierScalars: coalesce[{i}] must be a 2-element "
                    f"[key, scalars] pair, got {type(entry).__name__}: {entry!r}."
                )
            raw_key, raw_scalars = entry
            if not isinstance(raw_key, (list, tuple)) or len(raw_key) != 2 or not all(isinstance(s, str) for s in raw_key):
                raise AuditIntegrityError(
                    f"Corrupted BarrierScalars: coalesce[{i}] key must be a 2-element "
                    f"[str, str] array, got {type(raw_key).__name__}: {raw_key!r}."
                )
            if not isinstance(raw_scalars, dict):
                raise AuditIntegrityError(
                    f"Corrupted BarrierScalars: coalesce[{i}] scalars must be a dict, got {type(raw_scalars).__name__}."
                )
            coalesce_name: str = raw_key[0]
            coalesce_row_id: str = raw_key[1]
            coalesce_key = (coalesce_name, coalesce_row_id)
            # to_dict can never emit a duplicate key (it iterates a dict), so a
            # duplicate in the list-of-pairs is by-definition corruption — reject
            # rather than silently last-wins.
            if coalesce_key in coalesce:
                raise AuditIntegrityError(f"Corrupted BarrierScalars: duplicate coalesce key {coalesce_key!r} at coalesce[{i}].")
            coalesce[coalesce_key] = CoalescePendingScalars.from_dict(raw_scalars)

        return cls(aggregation=aggregation, coalesce=coalesce)
