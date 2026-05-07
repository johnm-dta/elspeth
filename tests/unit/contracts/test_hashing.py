"""Tests for contracts/hashing.py — NaN/Infinity rejection and primitives."""

from datetime import UTC, datetime

import pytest

from elspeth.contracts import hashing as contracts_hashing
from elspeth.contracts.hashing import CANONICAL_VERSION, canonical_json, repr_hash, stable_hash
from elspeth.core import canonical as core_canonical
from elspeth.core.canonical import CANONICAL_VERSION as CORE_VERSION


class TestCanonicalJsonNanRejection:
    """NaN and Infinity must be rejected with clear error messages."""

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_json(float("nan"))

    def test_rejects_positive_infinity(self) -> None:
        with pytest.raises(ValueError, match="Infinity"):
            canonical_json(float("inf"))

    def test_rejects_negative_infinity(self) -> None:
        with pytest.raises(ValueError, match="Infinity"):
            canonical_json(float("-inf"))

    def test_rejects_nan_nested_in_dict(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_json({"key": float("nan")})

    def test_rejects_nan_nested_in_list(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_json([1, 2, float("nan")])

    def test_rejects_nan_deeply_nested(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_json({"a": {"b": [{"c": float("nan")}]}})

    def test_rejects_nan_nested_in_tuple(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_json({"key": (1, 2, float("nan"))})

    def test_rejects_nan_deeply_nested_in_tuple(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_json({"a": ({"b": (float("nan"),)},)})

    def test_rejects_infinity_in_dict_value(self) -> None:
        with pytest.raises(ValueError, match="Infinity"):
            canonical_json({"amount": float("inf")})

    def test_accepts_normal_floats(self) -> None:
        result = canonical_json({"value": 3.14, "zero": 0.0, "neg": -1.5})
        assert "3.14" in result

    def test_accepts_primitives(self) -> None:
        result = canonical_json({"str": "hello", "int": 42, "bool": True, "null": None})
        assert '"hello"' in result


class TestStableHashNanRejection:
    """stable_hash delegates to canonical_json, so NaN rejection propagates."""

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            stable_hash({"field": float("nan")})

    def test_accepts_normal_data(self) -> None:
        h = stable_hash({"key": "value"})
        assert len(h) == 64  # SHA-256 hex digest


class TestReprHash:
    """Tests for repr_hash()."""

    def test_produces_sha256_hex(self) -> None:
        result = repr_hash({"key": "value"})
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self) -> None:
        obj = {"key": "value", "num": 42}
        assert repr_hash(obj) == repr_hash(obj)

    def test_handles_non_serializable(self) -> None:
        repr_hash(datetime.now(UTC))  # must not raise


class TestCanonicalVersion:
    """Tests for the CANONICAL_VERSION constant."""

    def test_is_string(self) -> None:
        assert isinstance(CANONICAL_VERSION, str)

    def test_contains_sha256(self) -> None:
        assert "sha256" in CANONICAL_VERSION

    def test_contains_rfc8785(self) -> None:
        assert "rfc8785" in CANONICAL_VERSION

    def test_matches_core_canonical(self) -> None:
        assert CANONICAL_VERSION == CORE_VERSION


class TestCanonicalJsonTypeRejection:
    """Tests for non-JSON-safe types passed to canonical_json."""

    def test_rejects_datetime(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            canonical_json({"ts": datetime.now(UTC)})

    def test_rejects_set(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            canonical_json({"s": {1, 2, 3}})

    def test_rejects_bytes(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            canonical_json({"b": b"hello"})


class TestCanonicalJsonConsistency:
    """Tests that contracts hashing matches core canonical for primitive data."""

    def test_matches_core_canonical_for_primitives(self) -> None:
        data = {"str": "hello", "int": 42, "bool": True, "null": None, "float": 3.14}
        assert contracts_hashing.canonical_json(data) == core_canonical.canonical_json(data)

    def test_stable_hash_matches_core(self) -> None:
        data = {"str": "hello", "int": 42, "bool": True, "null": None, "float": 3.14}
        assert contracts_hashing.stable_hash(data) == core_canonical.stable_hash(data)

    def test_matches_core_canonical_for_mapping_proxy(self) -> None:
        from types import MappingProxyType

        frozen = MappingProxyType({"str": "hello", "int": 42, "nested": MappingProxyType({"x": 1})})
        assert contracts_hashing.canonical_json(frozen) == core_canonical.canonical_json(frozen)

    def test_stable_hash_matches_core_for_frozen(self) -> None:
        from types import MappingProxyType

        frozen = MappingProxyType({"key": "value", "list": (1, 2, 3)})
        assert contracts_hashing.stable_hash(frozen) == core_canonical.stable_hash(frozen)

    def test_matches_core_canonical_for_deeply_nested_frozen(self) -> None:
        from types import MappingProxyType

        frozen = MappingProxyType(
            {
                "a": MappingProxyType({"b": (MappingProxyType({"c": 3}),)}),
            }
        )
        assert contracts_hashing.canonical_json(frozen) == core_canonical.canonical_json(frozen)


class TestTelemetryLandscapeHashAlignment:
    """Verify there is only one hash computation path for call recording.

    After the architectural fix, telemetry reads hashes from the recorded Call
    object rather than recomputing them. This test validates that
    core.canonical.stable_hash (used by the recorder) handles all payload types
    that appear in practice, ensuring the single-source-of-truth design works.
    """

    def test_primitive_payload_hashes(self) -> None:
        """Primitive-only payloads hash successfully."""
        payload = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}
        h = core_canonical.stable_hash(payload)
        assert len(h) == 64
        assert h == core_canonical.stable_hash(payload)  # deterministic

    def test_datetime_payload_hashes(self) -> None:
        """Payloads containing datetime hash successfully via normalization."""
        payload = {"model": "gpt-4", "timestamp": datetime(2026, 1, 1, tzinfo=UTC)}
        h = core_canonical.stable_hash(payload)
        assert len(h) == 64

    def test_bytes_payload_hashes(self) -> None:
        """Payloads containing bytes hash successfully via normalization."""
        payload = {"content": b"binary-data", "status": "ok"}
        h = core_canonical.stable_hash(payload)
        assert len(h) == 64

    def test_decimal_payload_hashes(self) -> None:
        """Payloads containing Decimal hash successfully via normalization."""
        from decimal import Decimal

        payload = {"cost": Decimal("0.0042"), "model": "gpt-4"}
        h = core_canonical.stable_hash(payload)
        assert len(h) == 64

    def test_contracts_and_core_agree_on_primitives(self) -> None:
        """Both hash implementations agree for primitive data (regression guard)."""
        payload = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}
        assert contracts_hashing.stable_hash(payload) == core_canonical.stable_hash(payload)


class TestRejectNonFiniteMappingProxyType:
    """Regression: _reject_non_finite must recurse into MappingProxyType.

    deep_freeze() converts dict → MappingProxyType. If a frozen mapping
    containing NaN flows to canonical_json(), the NaN guard must still fire.
    Bugs: elspeth-cfa0007836, elspeth-6e8999df4e.
    """

    def test_rejects_nan_in_mapping_proxy(self) -> None:
        from types import MappingProxyType

        frozen = MappingProxyType({"value": float("nan")})
        with pytest.raises(ValueError, match="NaN"):
            canonical_json(frozen)

    def test_rejects_nan_in_nested_mapping_proxy(self) -> None:
        from types import MappingProxyType

        frozen = MappingProxyType({"inner": MappingProxyType({"x": float("nan")})})
        with pytest.raises(ValueError, match="NaN"):
            canonical_json(frozen)

    def test_rejects_infinity_in_mapping_proxy(self) -> None:
        from types import MappingProxyType

        frozen = MappingProxyType({"value": float("inf")})
        with pytest.raises(ValueError, match="Infinity"):
            canonical_json(frozen)

    def test_accepts_mapping_proxy_with_normal_values(self) -> None:
        from types import MappingProxyType

        from elspeth.contracts.hashing import _normalize_frozen_and_reject_non_finite

        frozen = MappingProxyType({"key": "value", "num": 42})
        _normalize_frozen_and_reject_non_finite(frozen)  # must not raise

    def test_rejects_nan_in_deep_frozen_structure(self) -> None:
        """Simulates what deep_freeze() produces from routing reasons."""
        from elspeth.contracts.freeze import deep_freeze

        data = {"reason": "gate_match", "details": {"score": float("nan")}}
        frozen = deep_freeze(data)
        with pytest.raises(ValueError, match="NaN"):
            canonical_json(frozen)


class TestFrozenTypeHandling:
    """contracts/hashing must serialize frozen container types from deep_freeze.

    NaN/Infinity rejection inside MappingProxyType is already covered by
    TestRejectNonFiniteMappingProxyType above. These tests verify the
    serialization path: canonical_json must produce correct output from
    frozen containers, not just validate them.
    """

    def test_mapping_proxy_simple(self) -> None:
        from types import MappingProxyType

        frozen = MappingProxyType({"a": 1, "b": "hello"})
        assert canonical_json(frozen) == canonical_json({"a": 1, "b": "hello"})

    def test_mapping_proxy_nested(self) -> None:
        from types import MappingProxyType

        frozen = MappingProxyType({"a": MappingProxyType({"b": 2})})
        assert canonical_json(frozen) == canonical_json({"a": {"b": 2}})

    def test_mapping_proxy_with_tuple(self) -> None:
        from types import MappingProxyType

        frozen = MappingProxyType({"items": (1, 2, 3)})
        assert canonical_json(frozen) == canonical_json({"items": [1, 2, 3]})

    def test_deeply_nested_frozen(self) -> None:
        from types import MappingProxyType

        frozen = MappingProxyType(
            {
                "level1": MappingProxyType(
                    {
                        "level2": (MappingProxyType({"level3": "deep"}),),
                    }
                ),
            }
        )
        expected = {"level1": {"level2": [{"level3": "deep"}]}}
        assert canonical_json(frozen) == canonical_json(expected)

    def test_stable_hash_frozen_equals_unfrozen(self) -> None:
        from types import MappingProxyType

        data = {"key": "value", "nested": {"inner": [1, 2]}}
        frozen = MappingProxyType(
            {
                "key": "value",
                "nested": MappingProxyType({"inner": (1, 2)}),
            }
        )
        assert stable_hash(frozen) == stable_hash(data)

    def test_rejects_frozenset_with_type_error(self) -> None:
        with pytest.raises(TypeError, match="frozenset"):
            canonical_json({"s": frozenset({1, 2})})


class TestFrozenRoundTripContracts:
    """Hashing thawed output must equal hashing the frozen equivalent.

    This guarantees that to_dict() (which thaws) and direct frozen access
    produce identical hashes — the coherence invariant.
    """

    def test_simple_dict_round_trip(self) -> None:
        from elspeth.contracts.freeze import deep_freeze

        original = {"key": "value", "num": 42}
        frozen = deep_freeze(original)
        assert canonical_json(original) == canonical_json(frozen)

    def test_nested_dict_round_trip(self) -> None:
        from elspeth.contracts.freeze import deep_freeze

        original = {"outer": {"inner": [1, 2, {"deep": True}]}}
        frozen = deep_freeze(original)
        assert canonical_json(original) == canonical_json(frozen)

    def test_list_of_dicts_round_trip(self) -> None:
        from elspeth.contracts.freeze import deep_freeze

        original = [{"a": 1}, {"b": 2}, {"c": [3, 4]}]
        frozen = deep_freeze(original)
        assert canonical_json(original) == canonical_json(frozen)

    def test_stable_hash_round_trip(self) -> None:
        from elspeth.contracts.freeze import deep_freeze

        original = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        frozen = deep_freeze(original)
        assert stable_hash(original) == stable_hash(frozen)

    def test_empty_containers_round_trip(self) -> None:
        from elspeth.contracts.freeze import deep_freeze

        original: dict[str, dict[str, object] | list[object]] = {"empty_dict": {}, "empty_list": []}
        frozen = deep_freeze(original)
        assert canonical_json(original) == canonical_json(frozen)
