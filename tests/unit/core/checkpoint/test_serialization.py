"""Unit tests for checkpoint serialization edge cases."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.checkpoint.serialization import checkpoint_dumps, checkpoint_loads


def test_checkpoint_dumps_sets_utc_on_naive_datetime() -> None:
    naive = datetime(2026, 2, 8, 10, 15, 30, tzinfo=UTC).replace(tzinfo=None)
    result = checkpoint_loads(checkpoint_dumps({"created_at": naive}))
    restored = result["created_at"]

    assert isinstance(restored, datetime)
    assert restored.tzinfo is not None
    assert restored.replace(tzinfo=None) == naive


def test_checkpoint_dumps_preserves_aware_datetime() -> None:
    aware = datetime(2026, 2, 8, 10, 15, 30, tzinfo=UTC)
    result = checkpoint_loads(checkpoint_dumps({"created_at": aware}))
    assert result["created_at"] == aware


def test_checkpoint_dumps_raises_for_unserializable_type() -> None:
    with pytest.raises(TypeError):
        checkpoint_dumps({"bad": {1, 2, 3}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_checkpoint_dumps_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite float"):
        checkpoint_dumps({"v": value})


def test_checkpoint_dumps_rejects_non_finite_values_in_nested_list() -> None:
    with pytest.raises(ValueError, match="non-finite float"):
        checkpoint_dumps({"values": [1.0, {"inner": float("nan")}]})


def test_checkpoint_loads_restores_new_envelope() -> None:
    """New envelope format restores datetime correctly."""
    payload = '{"ts":{"__elspeth_type__":"datetime","__elspeth_value__":"2026-02-08T10:15:30+00:00"}}'
    result = checkpoint_loads(payload)

    assert isinstance(result["ts"], datetime)
    assert result["ts"] == datetime(2026, 2, 8, 10, 15, 30, tzinfo=UTC)


def test_checkpoint_loads_old_datetime_tag_is_not_restored() -> None:
    """Old shape-based tag is NOT restored (no legacy code per CLAUDE.md)."""
    payload = '{"ts":{"__datetime__":"2026-02-08T10:15:30+00:00"}}'
    result = checkpoint_loads(payload)

    # Should remain as a plain dict, NOT be converted to datetime
    assert isinstance(result["ts"], dict)
    assert result["ts"]["__datetime__"] == "2026-02-08T10:15:30+00:00"


def test_checkpoint_loads_does_not_restore_lookalike_tag_with_extra_keys() -> None:
    payload = '{"ts":{"__datetime__":"2026-02-08T10:15:30+00:00","extra":1}}'
    result = checkpoint_loads(payload)

    assert isinstance(result["ts"], dict)
    assert result["ts"]["__datetime__"] == "2026-02-08T10:15:30+00:00"
    assert result["ts"]["extra"] == 1


# ===========================================================================
# Bug 7.1: Collision-safe type envelopes
# ===========================================================================


def test_checkpoint_roundtrip_user_dict_matching_old_datetime_shape() -> None:
    """User dict matching the OLD shape-based tag must NOT be deserialized as datetime.

    Bug 7.1: A user dict like {"__datetime__": "2026-02-08T10:15:30+00:00"} with
    exactly 1 key would previously be incorrectly deserialized as a datetime object.
    The new envelope format prevents this collision.
    """
    user_data = {"field": {"__datetime__": "2026-02-08T10:15:30+00:00"}}
    result = checkpoint_loads(checkpoint_dumps(user_data))

    # The value should remain a dict, not be converted to datetime
    assert isinstance(result["field"], dict)
    assert result["field"]["__datetime__"] == "2026-02-08T10:15:30+00:00"


def test_checkpoint_roundtrip_user_dict_with_reserved_key() -> None:
    """User dict containing __elspeth_type__ must survive round-trip as a dict.

    The _escape_reserved_keys() function wraps such dicts in an escape envelope
    so they aren't confused with real type envelopes during deserialization.
    """
    user_data = {
        "config": {
            "__elspeth_type__": "some_user_value",
            "other_key": 42,
        }
    }
    result = checkpoint_loads(checkpoint_dumps(user_data))

    assert isinstance(result["config"], dict)
    assert result["config"]["__elspeth_type__"] == "some_user_value"
    assert result["config"]["other_key"] == 42


def test_checkpoint_roundtrip_datetime_still_works_with_new_envelope() -> None:
    """Datetime round-trip via the new collision-safe envelope."""
    dt = datetime(2026, 2, 8, 10, 15, 30, tzinfo=UTC)
    result = checkpoint_loads(checkpoint_dumps({"ts": dt}))

    assert isinstance(result["ts"], datetime)
    assert result["ts"] == dt


def test_checkpoint_roundtrip_nested_datetime_and_user_data() -> None:
    """Complex structure with both datetime and user dict containing reserved key."""
    dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    data = {
        "real_datetime": dt,
        "user_dict_with_reserved": {"__elspeth_type__": "user_label"},
        "user_dict_with_old_tag": {"__datetime__": "not-a-real-timestamp"},
        "normal": "value",
    }
    result = checkpoint_loads(checkpoint_dumps(data))

    assert isinstance(result["real_datetime"], datetime)
    assert result["real_datetime"] == dt
    assert isinstance(result["user_dict_with_reserved"], dict)
    assert result["user_dict_with_reserved"]["__elspeth_type__"] == "user_label"
    assert isinstance(result["user_dict_with_old_tag"], dict)
    assert result["user_dict_with_old_tag"]["__datetime__"] == "not-a-real-timestamp"
    assert result["normal"] == "value"


# ===========================================================================
# Tuple round-trip tests
# ===========================================================================


def test_checkpoint_roundtrip_tuple() -> None:
    """Tuples must survive checkpoint serialization round-trip.

    Regression: JSON has no tuple type — tuples must be encoded via
    the __elspeth_type__/tuple envelope and restored on loads.
    """
    data = {"key": (1, "x", True)}
    result = checkpoint_loads(checkpoint_dumps(data))

    assert result["key"] == (1, "x", True)
    assert isinstance(result["key"], tuple)


def test_checkpoint_roundtrip_nested_tuple() -> None:
    """Tuples nested inside dicts and lists must survive round-trip."""
    data = {"outer": [(1, 2), (3, 4)]}
    result = checkpoint_loads(checkpoint_dumps(data))

    assert result["outer"] == [(1, 2), (3, 4)]
    assert isinstance(result["outer"][0], tuple)
    assert isinstance(result["outer"][1], tuple)


def test_checkpoint_roundtrip_tuple_with_datetime() -> None:
    """Tuples containing datetime values must survive round-trip."""
    dt = datetime(2024, 1, 1, tzinfo=UTC)
    data = {"key": (dt, "value")}
    result = checkpoint_loads(checkpoint_dumps(data))

    assert isinstance(result["key"], tuple)
    assert result["key"][0] == dt
    assert isinstance(result["key"][0], datetime)
    assert result["key"][1] == "value"


def test_checkpoint_dumps_rejects_nan_in_tuple() -> None:
    """Non-finite floats inside tuples must be rejected like any other container."""
    with pytest.raises(ValueError, match="non-finite float"):
        checkpoint_dumps({"k": (float("nan"),)})


def test_checkpoint_new_envelope_used_in_dumps_output() -> None:
    """Verify the serialized form uses __elspeth_type__ not __datetime__."""
    import json

    dt = datetime(2026, 2, 8, 10, 15, 30, tzinfo=UTC)
    serialized = checkpoint_dumps({"ts": dt})
    raw = json.loads(serialized)

    assert "__elspeth_type__" in raw["ts"]
    assert raw["ts"]["__elspeth_type__"] == "datetime"
    assert "__elspeth_value__" in raw["ts"]


# ── Envelope corruption guards ──────────────────────────────────────────────


def test_unknown_envelope_type_raises() -> None:
    """Unknown envelope types must crash — Tier 1 corruption guard."""
    import json

    tampered = json.dumps({"data": {"__elspeth_type__": "evil", "__elspeth_value__": "payload"}})
    with pytest.raises(AuditIntegrityError, match="Unknown checkpoint envelope type 'evil'"):
        checkpoint_loads(tampered)


def test_known_envelope_wrong_value_type_datetime_raises() -> None:
    """datetime envelope with non-string value must crash — type corruption."""
    import json

    corrupted = json.dumps({"data": {"__elspeth_type__": "datetime", "__elspeth_value__": 42}})
    with pytest.raises(AuditIntegrityError, match="invalid value type"):
        checkpoint_loads(corrupted)


def test_known_envelope_wrong_value_type_tuple_raises() -> None:
    """tuple envelope with non-list value must crash — type corruption."""
    import json

    corrupted = json.dumps({"data": {"__elspeth_type__": "tuple", "__elspeth_value__": "not a list"}})
    with pytest.raises(AuditIntegrityError, match="invalid value type"):
        checkpoint_loads(corrupted)


def test_naive_datetime_in_envelope_raises() -> None:
    """Regression: naive datetime strings in checkpoint envelopes must crash (Tier 1)."""
    import json

    corrupted = json.dumps({"ts": {"__elspeth_type__": "datetime", "__elspeth_value__": "2026-03-15T10:00:00"}})
    with pytest.raises(AuditIntegrityError, match="naive datetime"):
        checkpoint_loads(corrupted)


def test_aware_datetime_in_envelope_accepted() -> None:
    """Aware datetime strings must round-trip correctly."""
    import json

    data = json.dumps({"ts": {"__elspeth_type__": "datetime", "__elspeth_value__": "2026-03-15T10:00:00+00:00"}})
    result = checkpoint_loads(data)
    assert result["ts"].tzinfo is not None
