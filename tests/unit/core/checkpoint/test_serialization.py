"""Unit tests for checkpoint serialization edge cases."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time
from decimal import Decimal
from uuid import UUID

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
    # A set is not JSON-native and has no envelope tag — the offensive error must
    # NAME the unsupported type (not the cryptic stdlib "Object of type set ...").
    with pytest.raises(TypeError, match="'set'") as exc_info:
        checkpoint_dumps({"bad": {1, 2, 3}})
    assert "Tier-1 audit-fidelity boundary" in str(exc_info.value)


def test_checkpoint_dumps_raises_for_custom_class_naming_type() -> None:
    """A genuinely unserializable custom instance raises a clear typed error naming the type."""

    class CustomThing:
        def __repr__(self) -> str:
            return "<CustomThing instance>"

    with pytest.raises(TypeError, match="'CustomThing'") as exc_info:
        checkpoint_dumps({"obj": CustomThing()})
    message = str(exc_info.value)
    assert "<CustomThing instance>" in message  # value repr included
    assert "JSON-native" in message


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


@pytest.mark.parametrize("reserved_key", ["__elspeth_type__", "__elspeth_value__"])
def test_checkpoint_roundtrip_user_dict_with_reserved_envelope_key(reserved_key: str) -> None:
    """User dicts containing reserved envelope keys must survive round-trip."""
    user_data = {"config": {reserved_key: "some_user_value", "other_key": 42}}
    result = checkpoint_loads(checkpoint_dumps(user_data))

    assert isinstance(result["config"], dict)
    assert result["config"][reserved_key] == "some_user_value"
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


def test_reserved_envelope_with_extra_keys_raises() -> None:
    """Reserved envelope keys with polluted shape must not restore as an ordinary dict."""
    import json

    tampered = json.dumps({"data": {"__elspeth_type__": "evil", "__elspeth_value__": "payload", "extra": 1}})
    with pytest.raises(AuditIntegrityError, match="invalid envelope shape"):
        checkpoint_loads(tampered)


@pytest.mark.parametrize(
    "tampered",
    [
        {"__elspeth_type__": "date"},
        {"__elspeth_value__": "2026-01-01"},
    ],
)
def test_partial_reserved_envelope_shape_raises(tampered: dict[str, object]) -> None:
    """A raw partial reserved envelope shape must crash, not restore as user data."""
    import json

    corrupted = json.dumps({"data": tampered})
    with pytest.raises(AuditIntegrityError, match="invalid envelope shape"):
        checkpoint_loads(corrupted)


def test_checkpoint_loads_rejects_duplicate_json_object_keys() -> None:
    """Ambiguous duplicate keys must crash before JSON object collapse."""
    corrupted = '{"data": {"safe": 1, "safe": 2}}'

    with pytest.raises(AuditIntegrityError, match="duplicate JSON object key"):
        checkpoint_loads(corrupted)


def test_checkpoint_loads_rejects_duplicate_envelope_keys() -> None:
    """Duplicate envelope keys must not select one type tag before restore."""
    corrupted = '{"data": {"__elspeth_type__": "bytes", "__elspeth_type__": "date", "__elspeth_value__": "2026-01-01"}}'

    with pytest.raises(AuditIntegrityError, match="duplicate JSON object key"):
        checkpoint_loads(corrupted)


def test_envelope_type_tag_must_be_string() -> None:
    """Envelope type tags must be strings before dispatch."""
    import json

    corrupted = json.dumps({"data": {"__elspeth_type__": 42, "__elspeth_value__": "payload"}})
    with pytest.raises(AuditIntegrityError, match="type tag"):
        checkpoint_loads(corrupted)


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


@pytest.mark.parametrize(
    ("envelope_type", "bad"),
    [
        ("datetime", "not-a-datetime"),
        ("decimal", "not-a-decimal"),
        ("date", "not-a-date"),
        ("time", "not-a-time"),
        ("uuid", "not-a-uuid"),
    ],
)
def test_known_string_envelope_parse_failures_raise_audit_integrity(envelope_type: str, bad: str) -> None:
    """Known string envelope parser failures must not leak raw parser errors."""
    import json

    corrupted = json.dumps({"data": {"__elspeth_type__": envelope_type, "__elspeth_value__": bad}})
    with pytest.raises(AuditIntegrityError, match=envelope_type):
        checkpoint_loads(corrupted)


# ===========================================================================
# F1 regression: fidelity-preservable delta types
#
# The F1 token_data_ref envelope runs checkpoint_dumps on row payloads during
# NORMAL execution. canonical_json accepted these types (lossily); the previous
# checkpoint serializer crashed. They must now round-trip EXACTLY (type + value).
# ===========================================================================


def test_checkpoint_roundtrip_decimal() -> None:
    """Decimal must round-trip as a Decimal instance with exact value."""
    result = checkpoint_loads(checkpoint_dumps({"amount": Decimal("99.25")}))
    restored = result["amount"]
    assert isinstance(restored, Decimal)
    assert restored == Decimal("99.25")


def test_checkpoint_roundtrip_decimal_preserves_precision() -> None:
    """str(Decimal) round-trip preserves trailing-zero precision (exact, not float-lossy)."""
    result = checkpoint_loads(checkpoint_dumps({"amount": Decimal("1.50")}))
    restored = result["amount"]
    assert isinstance(restored, Decimal)
    assert str(restored) == "1.50"


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_checkpoint_dumps_rejects_non_finite_decimal(bad: Decimal) -> None:
    """Non-finite Decimals must be rejected — same audit-integrity invariant as float."""
    with pytest.raises(ValueError, match="non-finite Decimal"):
        checkpoint_dumps({"amount": bad})


def test_checkpoint_roundtrip_date() -> None:
    """date (not datetime) must round-trip as a date instance."""
    result = checkpoint_loads(checkpoint_dumps({"d": date(2021, 1, 1)}))
    restored = result["d"]
    assert isinstance(restored, date)
    assert not isinstance(restored, datetime)  # date, not datetime
    assert restored == date(2021, 1, 1)


def test_checkpoint_roundtrip_time() -> None:
    """time must round-trip as a time instance."""
    result = checkpoint_loads(checkpoint_dumps({"t": time(12, 30, 0)}))
    restored = result["t"]
    assert isinstance(restored, time)
    assert restored == time(12, 30, 0)


def test_checkpoint_roundtrip_bytes() -> None:
    """bytes must round-trip exactly via base64."""
    result = checkpoint_loads(checkpoint_dumps({"b": b"\x00\xff"}))
    restored = result["b"]
    assert isinstance(restored, bytes)
    assert restored == b"\x00\xff"


@pytest.mark.parametrize("bad", ["Zm9v!!", "AA==$", "é"])
def test_checkpoint_loads_rejects_bytes_envelope_with_non_base64_characters(bad: str) -> None:
    """A bytes envelope with discarded non-base64 characters must crash."""
    import json

    corrupted = json.dumps({"b": {"__elspeth_type__": "bytes", "__elspeth_value__": bad}})
    with pytest.raises(AuditIntegrityError, match="bytes envelope"):
        checkpoint_loads(corrupted)


@pytest.mark.parametrize("bad", ["AB==", "//=="])
def test_checkpoint_loads_rejects_non_canonical_bytes_envelope(bad: str) -> None:
    """A bytes envelope must use the canonical Base64 spelling for its bytes."""
    import json

    corrupted = json.dumps({"b": {"__elspeth_type__": "bytes", "__elspeth_value__": bad}})
    with pytest.raises(AuditIntegrityError, match="non-canonical"):
        checkpoint_loads(corrupted)


def test_checkpoint_roundtrip_uuid() -> None:
    """UUID must round-trip as a UUID instance."""
    u = UUID("12345678-1234-5678-1234-567812345678")
    result = checkpoint_loads(checkpoint_dumps({"id": u}))
    restored = result["id"]
    assert isinstance(restored, UUID)
    assert restored == u


def test_checkpoint_roundtrip_datetime_not_confused_with_date() -> None:
    """datetime is a date subclass — it must still round-trip as datetime, not date."""
    dt = datetime(2021, 1, 1, 12, 0, 0, tzinfo=UTC)
    result = checkpoint_loads(checkpoint_dumps({"dt": dt}))
    restored = result["dt"]
    assert isinstance(restored, datetime)
    assert restored == dt


def test_checkpoint_roundtrip_nested_fidelity_types() -> None:
    """Fidelity types nested inside dicts and lists must all round-trip."""
    u = UUID("12345678-1234-5678-1234-567812345678")
    data = {
        "items": [
            {"amount": Decimal("1.5"), "d": date(2020, 5, 1)},
            {"amount": Decimal("99.25"), "t": time(8, 0, 0)},
        ],
        "blob": b"\x01\x02",
        "id": u,
    }
    result = checkpoint_loads(checkpoint_dumps(data))
    assert result["items"][0]["amount"] == Decimal("1.5")
    assert isinstance(result["items"][0]["amount"], Decimal)
    assert result["items"][0]["d"] == date(2020, 5, 1)
    assert result["items"][1]["amount"] == Decimal("99.25")
    assert result["items"][1]["t"] == time(8, 0, 0)
    assert result["blob"] == b"\x01\x02"
    assert result["id"] == u


def test_checkpoint_decimal_tuple_roundtrip() -> None:
    """A Decimal nested inside a tuple round-trips (exercises _escape_reserved_keys path)."""
    result = checkpoint_loads(checkpoint_dumps({"k": (Decimal("3.14"), b"\xab")}))
    restored = result["k"]
    assert isinstance(restored, tuple)
    assert restored[0] == Decimal("3.14")
    assert isinstance(restored[0], Decimal)
    assert restored[1] == b"\xab"


def test_checkpoint_corrupted_decimal_envelope_wrong_value_type_raises() -> None:
    """A decimal envelope with a non-string value must crash — Tier 1 corruption guard."""
    import json

    corrupted = json.dumps({"d": {"__elspeth_type__": "decimal", "__elspeth_value__": 42}})
    with pytest.raises(AuditIntegrityError, match="invalid value type"):
        checkpoint_loads(corrupted)


# ── Read-side non-finite guards ─────────────────────────────────────────────
# The write side rejects non-finite values twice (``_reject_nan_infinity`` then
# ``json.dumps(allow_nan=False)``). The read side must be symmetric: a corrupted
# or tampered Tier-1 checkpoint must NOT silently restore a live NaN/Infinity.
# Two distinct paths reach ``checkpoint_loads``:
#   1. Decimal surfaces through an envelope arm — Decimal("NaN") constructs fine.
#   2. float surfaces as a bare JSON literal — json.loads accepts NaN/Infinity
#      by default (a Python extension to strict JSON).


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "sNaN"])
def test_non_finite_decimal_in_envelope_raises(bad: str) -> None:
    """A non-finite decimal envelope value must crash — Tier 1 corruption guard.

    The value is a valid Decimal string (so it survives construction) but violates
    the same audit-integrity invariant the write side enforces at line 115. Mirrors
    ``test_naive_datetime_in_envelope_raises`` — a valid-shape envelope whose
    restored value breaks a semantic invariant.
    """
    import json

    corrupted = json.dumps({"d": {"__elspeth_type__": "decimal", "__elspeth_value__": bad}})
    with pytest.raises(AuditIntegrityError, match="non-finite"):
        checkpoint_loads(corrupted)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_float_literal_in_checkpoint_raises(literal: str) -> None:
    """A bare non-finite float literal must crash — symmetric with allow_nan=False on write.

    ``json.loads`` accepts NaN/Infinity/-Infinity float literals by default; without a
    read-side guard a tampered checkpoint round-trips a live float('nan') into Tier 1.
    """
    corrupted = '{"x": ' + literal + "}"
    with pytest.raises(AuditIntegrityError, match="non-finite"):
        checkpoint_loads(corrupted)


def test_checkpoint_numpy_scalar_normalizes_to_primitive() -> None:
    """numpy scalars serialize to and restore as Python primitives (mirror canonical_json)."""
    np = pytest.importorskip("numpy")

    result = checkpoint_loads(checkpoint_dumps({"i": np.int64(7), "f": np.float64(2.5), "b": np.bool_(True)}))
    assert result["i"] == 7
    assert type(result["i"]) is int
    assert result["f"] == 2.5
    assert type(result["f"]) is float
    assert result["b"] is True
    assert type(result["b"]) is bool


def test_checkpoint_numpy_non_finite_float_rejected() -> None:
    """Non-finite numpy floats must be rejected like Python floats."""
    np = pytest.importorskip("numpy")

    with pytest.raises(ValueError, match="non-finite float"):
        checkpoint_dumps({"f": np.float32("nan")})


def test_checkpoint_numpy_longdouble_overflow_rejected() -> None:
    """A np.longdouble finite in native form but overflowing float() must raise the
    clear ELSPETH error, not json's 'Out of range float values' (canonical parity).
    """
    np = pytest.importorskip("numpy")

    # On platforms where longdouble == float64 (e.g. some ARM/Windows builds) no
    # value is finite-in-native-but-overflowing-on-narrowing, so the guard is
    # unreachable there — skip rather than assert a platform-specific behaviour.
    max_ld = np.longdouble(np.finfo(np.longdouble).max)
    if math.isfinite(float(max_ld)):
        pytest.skip("longdouble has no values outside IEEE-754 double range on this platform")

    with pytest.raises(ValueError, match="exceeds IEEE 754 double range"):
        checkpoint_dumps({"f": max_ld})
