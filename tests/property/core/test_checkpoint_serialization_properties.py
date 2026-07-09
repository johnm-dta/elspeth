# tests/property/core/test_checkpoint_serialization_properties.py
"""Property-based tests for checkpoint serialization round-trip fidelity.

This module tests the checkpoint_dumps / checkpoint_loads serialization path
which is SEPARATE from the CheckpointManager path tested in
test_checkpoint_properties.py.

The key distinction:
- CheckpointManager uses json.dumps(allow_nan=False) for aggregation state
- checkpoint/serialization.py uses CheckpointEncoder + _reject_nan_infinity()
  with datetime type-tags for round-trip fidelity

The serialization path is exercised during RESUME (checkpoint_loads), so these
tests protect against data corruption during crash recovery.

Properties tested:
- Round-trip fidelity: checkpoint_loads(checkpoint_dumps(x)) == x
- NaN/Infinity rejection at any nesting depth
- Type tag isolation (datetime tags vs coincidental dict keys)
- Nested structure preservation
- Timezone handling for naive/aware datetimes
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.checkpoint.serialization import (
    CheckpointEncoder,
    _reject_nan_infinity,
    _restore_types,
    checkpoint_dumps,
    checkpoint_loads,
)
from tests.strategies.json import json_primitives, json_values

# =============================================================================
# Strategies for checkpoint serialization
# =============================================================================

# Timezone-aware datetimes (UTC or fixed offset)
aware_datetimes = st.datetimes(
    min_value=datetime(2000, 1, 1),  # noqa: DTZ001 — Hypothesis requires naive bounds
    max_value=datetime(2100, 1, 1),  # noqa: DTZ001
    timezones=st.just(UTC),
)

# Fixed-offset timezones for broader coverage
offset_hours = st.integers(min_value=-12, max_value=14)
fixed_tz_datetimes = st.builds(
    lambda dt, h: dt.replace(tzinfo=timezone(timedelta(hours=h))),
    st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1)),  # noqa: DTZ001
    offset_hours,
)

# Naive datetimes (no tzinfo - should get UTC attached)
naive_datetimes = st.datetimes(
    min_value=datetime(2000, 1, 1),  # noqa: DTZ001 — intentionally naive for testing UTC attachment
    max_value=datetime(2100, 1, 1),  # noqa: DTZ001
)

# Values that can appear in checkpoint aggregation state
# (everything json_values has, PLUS datetimes)
checkpoint_leaf_values = json_primitives | aware_datetimes

checkpoint_values = st.recursive(
    checkpoint_leaf_values,
    lambda children: (
        st.lists(children, max_size=5)
        | st.dictionaries(
            st.text(max_size=15).filter(lambda k: k != "__datetime__"),
            children,
            max_size=5,
        )
    ),
    max_leaves=30,
)

# Aggregation-state-shaped dicts (string keys, checkpoint values)
aggregation_states = st.dictionaries(
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    checkpoint_values,
    min_size=0,
    max_size=5,
)


# Nesting depth generators for NaN/Infinity tests
def _nest_value_at_depth(value, depth: int, container: str = "dict"):
    """Wrap a value in nested dicts/lists to a given depth."""
    result = value
    for i in range(depth):
        if container == "dict":
            result = {f"level_{i}": result}
        else:
            result = [result]
    return result


# =============================================================================
# Round-Trip Fidelity Properties
# =============================================================================


class TestRoundTripFidelity:
    """checkpoint_loads(checkpoint_dumps(x)) == x for all valid inputs."""

    @given(state=aggregation_states)
    @settings(max_examples=300)
    def test_aggregation_state_roundtrip(self, state: dict[str, Any]) -> None:
        """Property: Aggregation state survives serialization round-trip."""
        serialized = checkpoint_dumps(state)
        restored = checkpoint_loads(serialized)
        assert restored == state

    @given(dt=aware_datetimes)
    @settings(max_examples=200)
    def test_datetime_roundtrip_preserves_value(self, dt: datetime) -> None:
        """Property: UTC datetimes round-trip with exact value preservation."""
        data = {"timestamp": dt}
        restored = checkpoint_loads(checkpoint_dumps(data))
        assert restored["timestamp"] == dt
        assert isinstance(restored["timestamp"], datetime)

    @given(dt=fixed_tz_datetimes)
    @settings(max_examples=100)
    def test_fixed_offset_datetime_roundtrip(self, dt: datetime) -> None:
        """Property: Fixed-offset datetimes survive round-trip.

        The ISO format preserves offset info, so fromisoformat() restores it.
        """
        data = {"ts": dt}
        restored = checkpoint_loads(checkpoint_dumps(data))
        assert restored["ts"] == dt
        assert isinstance(restored["ts"], datetime)

    @given(dt=naive_datetimes)
    @settings(max_examples=100)
    def test_naive_datetime_gets_utc(self, dt: datetime) -> None:
        """Property: Naive datetimes get UTC attached during serialization.

        CheckpointEncoder replaces tzinfo=None with UTC, so the round-trip
        value has UTC timezone even if the input was naive.
        """
        data = {"ts": dt}
        restored = checkpoint_loads(checkpoint_dumps(data))
        expected = dt.replace(tzinfo=UTC)
        assert restored["ts"] == expected
        assert restored["ts"].tzinfo is not None

    @given(values=st.lists(checkpoint_leaf_values, min_size=0, max_size=10))
    @settings(max_examples=200)
    def test_list_roundtrip(self, values: list[Any]) -> None:
        """Property: Lists of mixed types (including datetimes) round-trip."""
        data = {"items": values}
        restored = checkpoint_loads(checkpoint_dumps(data))
        # Naive datetimes get UTC, so normalize for comparison
        expected_items = []
        for v in values:
            if isinstance(v, datetime) and v.tzinfo is None:
                expected_items.append(v.replace(tzinfo=UTC))
            else:
                expected_items.append(v)
        assert restored["items"] == expected_items

    @given(data=json_values)
    @settings(max_examples=200)
    def test_json_values_without_datetime_roundtrip(self, data) -> None:
        """Property: Plain JSON values (no datetimes) round-trip identically."""
        wrapped = {"payload": data}
        restored = checkpoint_loads(checkpoint_dumps(wrapped))
        assert restored == wrapped

    def test_empty_dict_roundtrip(self) -> None:
        """Edge case: Empty dict round-trips."""
        assert checkpoint_loads(checkpoint_dumps({})) == {}

    def test_empty_list_roundtrip(self) -> None:
        """Edge case: Empty list round-trips."""
        assert checkpoint_loads(checkpoint_dumps([])) == []

    def test_none_roundtrip(self) -> None:
        """Edge case: None round-trips."""
        assert checkpoint_loads(checkpoint_dumps(None)) is None


# =============================================================================
# NaN/Infinity Rejection Properties
# =============================================================================


class TestNaNInfinityRejection:
    """NaN and Infinity must be rejected at ANY nesting depth."""

    @given(depth=st.integers(min_value=0, max_value=8))
    @settings(max_examples=50)
    def test_nan_rejected_at_any_dict_depth(self, depth: int) -> None:
        """Property: NaN nested in dicts at any depth raises ValueError."""
        data = _nest_value_at_depth(float("nan"), depth, "dict")
        with pytest.raises(ValueError, match="non-finite float"):
            checkpoint_dumps(data)

    @given(depth=st.integers(min_value=0, max_value=8))
    @settings(max_examples=50)
    def test_infinity_rejected_at_any_dict_depth(self, depth: int) -> None:
        """Property: Infinity nested in dicts at any depth raises ValueError."""
        data = _nest_value_at_depth(float("inf"), depth, "dict")
        with pytest.raises(ValueError, match="non-finite float"):
            checkpoint_dumps(data)

    @given(depth=st.integers(min_value=0, max_value=8))
    @settings(max_examples=50)
    def test_neg_infinity_rejected_at_any_dict_depth(self, depth: int) -> None:
        """Property: -Infinity nested in dicts at any depth raises ValueError."""
        data = _nest_value_at_depth(float("-inf"), depth, "dict")
        with pytest.raises(ValueError, match="non-finite float"):
            checkpoint_dumps(data)

    @given(depth=st.integers(min_value=1, max_value=8))
    @settings(max_examples=50)
    def test_nan_rejected_at_any_list_depth(self, depth: int) -> None:
        """Property: NaN nested in lists at any depth raises ValueError."""
        data = _nest_value_at_depth(float("nan"), depth, "list")
        with pytest.raises(ValueError, match="non-finite float"):
            checkpoint_dumps(data)

    @given(
        good_keys=st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L",))),  # type: ignore[arg-type]  # hypothesis stubs accept tuple
            min_size=1,
            max_size=3,
            unique=True,
        ),
        bad_position=st.integers(min_value=0),
    )
    @settings(max_examples=100)
    def test_nan_among_valid_values_still_rejected(self, good_keys: list[str], bad_position: int) -> None:
        """Property: A single NaN among many valid values still raises."""
        data = {k: i * 1.5 for i, k in enumerate(good_keys)}
        # Insert NaN at a random key position
        poison_key = good_keys[bad_position % len(good_keys)]
        data[poison_key] = float("nan")
        with pytest.raises(ValueError, match="non-finite float"):
            checkpoint_dumps(data)

    def test_reject_nan_infinity_function_directly(self) -> None:
        """Verify _reject_nan_infinity returns obj when valid."""
        obj = {"a": 1, "b": [2.0, 3.0], "c": {"d": "hello"}}
        result = _reject_nan_infinity(obj)
        assert result is obj  # Same object, not a copy


# =============================================================================
# Type Tag Isolation Properties
# =============================================================================


class TestTypeTagIsolation:
    """Type tags must only trigger for exact datetime tag dicts."""

    def test_datetime_tag_with_extra_keys_raises(self) -> None:
        """Property: a reserved envelope polluted with extra keys fails closed.

        Only dicts with EXACTLY __elspeth_type__ and __elspeth_value__ are valid
        envelopes. A legitimate checkpoint never produces this shape (user dicts
        carrying the reserved key are escaped on the write side), so any extra key
        alongside the reserved pair signals a corrupted/tampered checkpoint and
        must crash rather than silently round-trip as an ordinary dict.
        """
        data = {"__elspeth_type__": "datetime", "__elspeth_value__": "2024-01-01T00:00:00+00:00", "extra": "value"}
        with pytest.raises(AuditIntegrityError, match="invalid envelope shape"):
            _restore_types(data)

    @given(
        extra_key=st.text(min_size=1, max_size=20).filter(lambda s: s not in ("__elspeth_type__", "__elspeth_value__")),
        extra_val=json_primitives,
    )
    @settings(max_examples=100)
    def test_datetime_tag_with_any_extra_key_raises(self, extra_key: str, extra_val) -> None:
        """Property: adding ANY extra key to a type envelope fails closed."""
        data = {"__elspeth_type__": "datetime", "__elspeth_value__": "2024-06-15T12:30:00+00:00", extra_key: extra_val}
        with pytest.raises(AuditIntegrityError, match="invalid envelope shape"):
            _restore_types(data)

    @given(value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_non_iso_datetime_tag_raises_on_restore(self, value: str) -> None:
        """Property: Invalid ISO string in datetime envelope raises AuditIntegrityError.

        If someone crafts a {"__elspeth_type__": "datetime", "__elspeth_value__": "garbage"},
        restore must fail at the Tier 1 audit-corruption boundary instead of leaking
        a raw parser exception.
        """
        assume(not _is_valid_isoformat(value))
        data = {"__elspeth_type__": "datetime", "__elspeth_value__": value}
        with pytest.raises(AuditIntegrityError, match="datetime envelope"):
            _restore_types(data)

    def test_nested_datetime_tags_restored(self) -> None:
        """Property: Datetime envelopes nested in lists/dicts all get restored."""
        data = {
            "outer": {"__elspeth_type__": "datetime", "__elspeth_value__": "2024-01-01T00:00:00+00:00"},
            "list": [
                {"__elspeth_type__": "datetime", "__elspeth_value__": "2024-06-15T12:00:00+00:00"},
                "plain_string",
            ],
        }
        restored = _restore_types(data)
        assert isinstance(restored["outer"], datetime)
        assert isinstance(restored["list"][0], datetime)
        assert restored["list"][1] == "plain_string"

    @given(
        n=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30)
    def test_multiple_datetimes_all_roundtrip(self, n: int) -> None:
        """Property: Multiple datetimes in one structure all round-trip."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        data = {f"ts_{i}": base + timedelta(days=i) for i in range(n)}
        restored = checkpoint_loads(checkpoint_dumps(data))
        assert restored == data


# =============================================================================
# Serialization Determinism Properties
# =============================================================================


class TestSerializationDeterminism:
    """Same input must always produce the same serialized output."""

    @given(state=aggregation_states)
    @settings(max_examples=200)
    def test_dumps_is_deterministic(self, state: dict[str, Any]) -> None:
        """Property: checkpoint_dumps produces identical output for identical input."""
        s1 = checkpoint_dumps(state)
        s2 = checkpoint_dumps(state)
        assert s1 == s2

    @given(dt=aware_datetimes)
    @settings(max_examples=100)
    def test_datetime_serialization_deterministic(self, dt: datetime) -> None:
        """Property: Same datetime always serializes to the same string."""
        data = {"ts": dt}
        s1 = checkpoint_dumps(data)
        s2 = checkpoint_dumps(data)
        assert s1 == s2


# =============================================================================
# Encoder Edge Cases
# =============================================================================


class TestCheckpointEncoderEdgeCases:
    """Edge cases for CheckpointEncoder behavior."""

    def test_encoder_rejects_non_serializable_types(self) -> None:
        """Property: Non-serializable types raise TypeError, not silent corruption."""
        import json

        with pytest.raises(TypeError):
            json.dumps({"obj": object()}, cls=CheckpointEncoder)

    def test_encoder_handles_bool_not_confused_with_int(self) -> None:
        """Property: Booleans are preserved as booleans, not converted to ints."""
        data = {"flag": True, "count": 1}
        restored = checkpoint_loads(checkpoint_dumps(data))
        assert restored["flag"] is True
        assert restored["count"] == 1
        assert isinstance(restored["flag"], bool)

    @given(
        f=st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-1e15,
            max_value=1e15,
        )
    )
    @settings(max_examples=200)
    def test_finite_floats_roundtrip(self, f: float) -> None:
        """Property: All finite floats survive round-trip."""
        data = {"value": f}
        restored = checkpoint_loads(checkpoint_dumps(data))
        if math.isnan(f):
            assert math.isnan(restored["value"])
        else:
            assert restored["value"] == f


# =============================================================================
# Helpers
# =============================================================================


def _is_valid_isoformat(s: str) -> bool:
    """Check if a string is a valid ISO format datetime."""
    try:
        datetime.fromisoformat(s)
        return True
    except (ValueError, TypeError):
        return False
