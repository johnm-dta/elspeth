"""Tests for contracts.config.protocols -- runtime-checkable protocol verification."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any

import pytest

from elspeth.contracts.config import protocols as _protocols_module
from elspeth.contracts.config import runtime as _runtime_module
from elspeth.contracts.config.protocols import (
    RuntimeCheckpointProtocol,
    RuntimeConcurrencyProtocol,
    RuntimeRateLimitProtocol,
    RuntimeRetryProtocol,
    RuntimeTelemetryProtocol,
    ServiceRateLimitProtocol,
)
from elspeth.contracts.config.runtime import (
    ExporterConfig,
    RuntimeCheckpointConfig,
    RuntimeRetryConfig,
    RuntimeTelemetryConfig,
)
from elspeth.contracts.enums import BackpressureMode, TelemetryGranularity

# ============================================================================
# Protocol / Runtime config pairs for parametrized tests
# ============================================================================
#
# ``ALL_PROTOCOLS`` and ``PROTOCOL_CONFIG_PAIRS`` are derived from runtime
# introspection of ``elspeth.contracts.config.protocols`` and ``...runtime``.
# A new ``Runtime*Protocol`` class added in production is automatically
# picked up by every parametrised test below — no test-side edit required.
#
# Naming convention:
#   - Runtime protocols are top-level ``Runtime<Name>Protocol`` classes that
#     are ``@runtime_checkable``. ``ServiceRateLimitProtocol`` is intentionally
#     excluded (no ``Runtime`` prefix because it describes a sub-component,
#     not a top-level engine boundary).
#   - Each ``Runtime<Name>Protocol`` MUST have a corresponding
#     ``Runtime<Name>Config`` class in ``contracts.config.runtime``.
#     Discovery raises ``AttributeError`` at module-load time if a protocol
#     has no matching config — making the convention violation a hard error
#     rather than a silent test gap.


def _is_runtime_protocol(obj: Any) -> bool:
    """Match top-level ``Runtime*Protocol`` runtime-checkable Protocol classes."""
    if not inspect.isclass(obj):
        return False
    if not (obj.__name__.startswith("Runtime") and obj.__name__.endswith("Protocol")):
        return False
    # Runtime-checkable Protocols expose ``__protocol_attrs__`` on recent
    # CPython and the legacy ``_is_runtime_protocol`` attribute. Accept either,
    # mirroring the compound check in ``test_protocol_is_runtime_checkable``.
    return getattr(obj, "__protocol_attrs__", None) is not None or getattr(obj, "_is_runtime_protocol", False)


def _discover_runtime_protocols() -> list[type]:
    """Return all top-level ``Runtime*Protocol`` classes from the protocols module."""
    discovered = [obj for _name, obj in inspect.getmembers(_protocols_module) if _is_runtime_protocol(obj)]
    # Sort by class name so test order is deterministic across CPython versions.
    return sorted(discovered, key=lambda cls: cls.__name__)


def _config_class_for(protocol: type) -> type:
    """Resolve the ``Runtime<Name>Config`` class paired with a ``Runtime<Name>Protocol``."""
    config_name = protocol.__name__.removesuffix("Protocol") + "Config"
    config_cls = getattr(_runtime_module, config_name, None)
    if config_cls is None:
        raise AttributeError(
            f"Protocol {protocol.__name__!r} has no matching runtime config class. "
            f"Expected ``elspeth.contracts.config.runtime.{config_name}`` per the "
            "Runtime<Name>Protocol → Runtime<Name>Config naming convention. Add the "
            "config class or rename the protocol."
        )
    return config_cls


def _param_id_for(protocol: type) -> str:
    """Convert ``RuntimeRateLimitProtocol`` → ``rate_limit`` for parametrise IDs."""
    name = protocol.__name__.removeprefix("Runtime").removesuffix("Protocol")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


ALL_PROTOCOLS: list[type] = _discover_runtime_protocols()

PROTOCOL_CONFIG_PAIRS = [pytest.param(protocol, _config_class_for(protocol), id=_param_id_for(protocol)) for protocol in ALL_PROTOCOLS]

# Expected property names for each protocol (completeness checks)
PROTOCOL_EXPECTED_PROPERTIES: dict[type, set[str]] = {
    RuntimeRetryProtocol: {
        "max_attempts",
        "base_delay",
        "max_delay",
        "exponential_base",
        "jitter",
    },
    RuntimeRateLimitProtocol: {
        "enabled",
        "default_requests_per_minute",
        "persistence_path",
    },
    RuntimeConcurrencyProtocol: {
        "max_workers",
    },
    RuntimeCheckpointProtocol: {
        "enabled",
        "frequency",
    },
    RuntimeTelemetryProtocol: {
        "enabled",
        "granularity",
        "backpressure_mode",
        "fail_on_total_exporter_failure",
        "max_consecutive_failures",
        "exporter_configs",
    },
}

PROTOCOL_EXPECTED_METHODS: dict[type, set[str]] = {
    RuntimeRetryProtocol: set(),
    RuntimeRateLimitProtocol: {"get_service_config"},
    RuntimeConcurrencyProtocol: set(),
    RuntimeCheckpointProtocol: set(),
    RuntimeTelemetryProtocol: set(),
}


def _get_protocol_property_names(protocol: type) -> set[str]:
    """Extract property names defined in a Protocol class (excludes dunder)."""
    return {name for name in dir(protocol) if not name.startswith("_") and isinstance(getattr(protocol, name, None), property)}


def _get_protocol_property_return_type(protocol: type, prop_name: str) -> Any:
    """Get the return type annotation from a Protocol property getter."""
    prop = getattr(protocol, prop_name)
    assert isinstance(prop, property)
    fget = prop.fget
    assert fget is not None
    sig = inspect.signature(fget)
    return sig.return_annotation


def _normalize_type_name(t: Any) -> str:
    """Normalize a type annotation to a comparable base name string.

    Handles: actual types (int, bool), forward refs ('BackpressureMode'),
    enum types, and generic aliases (tuple[ExporterConfig, ...]).
    """
    if isinstance(t, str):
        # Forward reference -- strip quotes, extract base name before '['
        clean = t.strip("'\"")
        return clean.split("[")[0] if "[" in clean else clean
    if isinstance(t, type):
        return t.__name__
    # GenericAlias like tuple[ExporterConfig, ...]
    origin = getattr(t, "__origin__", None)
    if origin is not None:
        return origin.__name__ if isinstance(origin, type) else str(origin)
    return str(t)


# ============================================================================
# Stub classes for structural typing tests
# ============================================================================


@dataclass
class _FakeRetryComplete:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 1.0


@dataclass
class _FakeRetryMissingJitter:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


@dataclass
class _FakeRetryWrongTypes:
    """All attributes present but with wrong types -- runtime_checkable still passes."""

    max_attempts: str = "not_an_int"
    base_delay: str = "not_a_float"
    max_delay: str = "not_a_float"
    exponential_base: str = "not_a_float"
    jitter: str = "not_a_float"


@dataclass
class _FakeRateLimitComplete:
    enabled: bool = True
    default_requests_per_minute: int = 60
    persistence_path: str | None = None

    def get_service_config(self, service_name: str) -> ServiceRateLimitProtocol:
        return _FakeServiceRateLimit(requests_per_minute=self.default_requests_per_minute)


@dataclass
class _FakeRateLimitMissingEnabled:
    default_requests_per_minute: int = 60
    persistence_path: str | None = None

    def get_service_config(self, service_name: str) -> ServiceRateLimitProtocol:
        return _FakeServiceRateLimit(requests_per_minute=self.default_requests_per_minute)


@dataclass
class _FakeRateLimitLegacyOnly:
    enabled: bool = True
    default_requests_per_minute: int = 60


@dataclass
class _FakeConcurrencyComplete:
    max_workers: int = 4


@dataclass
class _FakeConcurrencyMissing:
    pass


@dataclass
class _FakeCheckpointComplete:
    enabled: bool = True
    frequency: int = 1


@dataclass
class _FakeCheckpointMissingFrequency:
    enabled: bool = True


@dataclass
class _FakeTelemetryComplete:
    enabled: bool = False
    granularity: TelemetryGranularity = TelemetryGranularity.LIFECYCLE
    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK
    fail_on_total_exporter_failure: bool = True
    max_consecutive_failures: int = 10
    exporter_configs: tuple[ExporterConfig, ...] = ()


@dataclass
class _FakeTelemetryMissingGranularity:
    enabled: bool = False
    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK
    fail_on_total_exporter_failure: bool = True
    max_consecutive_failures: int = 10
    exporter_configs: tuple[ExporterConfig, ...] = ()


@dataclass
class _FakeServiceRateLimit:
    requests_per_minute: int


# ============================================================================
# Tests
# ============================================================================


class TestRuntimeCheckable:
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=_param_id_for)
    def test_protocol_is_runtime_checkable(self, protocol: type) -> None:
        # A non-@runtime_checkable Protocol raises TypeError when used with
        # isinstance(). Probe the live behaviour rather than introspecting
        # CPython-private attributes (which differ across versions and may
        # OR-fall-back into a false pass).
        try:
            isinstance(object(), protocol)
        except TypeError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{protocol.__name__} is not runtime_checkable: {exc}")

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=_param_id_for)
    def test_protocol_usable_in_isinstance(self, protocol: type) -> None:
        # Verifies isinstance() can be called (does not raise TypeError)
        result = isinstance(object(), protocol)
        assert result is False

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=_param_id_for)
    def test_protocol_rejects_plain_object(self, protocol: type) -> None:
        assert not isinstance(object(), protocol)

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=_param_id_for)
    def test_protocol_rejects_empty_dict(self, protocol: type) -> None:
        assert not isinstance({}, protocol)


class TestRealImplementations:
    @pytest.mark.parametrize("protocol,config_cls", PROTOCOL_CONFIG_PAIRS)
    def test_default_instance_satisfies_protocol(self, protocol: type, config_cls: Any) -> None:
        instance = config_cls.default()
        assert isinstance(instance, protocol)

    def test_retry_no_retry_satisfies_protocol(self) -> None:
        config = RuntimeRetryConfig.no_retry()
        assert isinstance(config, RuntimeRetryProtocol)

    @pytest.mark.parametrize("protocol,config_cls", PROTOCOL_CONFIG_PAIRS)
    def test_real_instance_does_not_satisfy_wrong_protocol(self, protocol: type, config_cls: Any) -> None:
        instance = config_cls.default()
        # Sanity: the matching protocol must accept the instance.
        assert isinstance(instance, protocol), f"{config_cls.__name__}.default() failed identity check against {protocol.__name__}"
        # Every other protocol in ALL_PROTOCOLS must reject this instance.
        # Verified empirically: each runtime config has a distinct required-attribute
        # set under @runtime_checkable structural typing -- shared attribute names
        # like `enabled` do not cause cross-protocol false matches because the other
        # protocols always require additional attributes the instance does not expose.
        # If a future protocol shrinks to a strict subset of another, this test will
        # surface the structural collision rather than silently pass.
        wrong_protocols = [p for p in ALL_PROTOCOLS if p is not protocol]
        for wrong_protocol in wrong_protocols:
            assert not isinstance(instance, wrong_protocol), (
                f"{config_cls.__name__}.default() unexpectedly satisfies "
                f"{wrong_protocol.__name__}; structural typing collision -- "
                f"required-attribute sets are no longer disjoint"
            )


class TestStructuralTyping:
    def test_matching_dataclass_satisfies_retry_protocol(self) -> None:
        assert isinstance(_FakeRetryComplete(), RuntimeRetryProtocol)

    def test_missing_property_fails_retry_protocol(self) -> None:
        assert not isinstance(_FakeRetryMissingJitter(), RuntimeRetryProtocol)

    def test_wrong_types_still_satisfies_retry_protocol(self) -> None:
        # runtime_checkable only checks attribute existence, not types
        assert isinstance(_FakeRetryWrongTypes(), RuntimeRetryProtocol)

    def test_matching_dataclass_satisfies_rate_limit_protocol(self) -> None:
        assert isinstance(_FakeRateLimitComplete(), RuntimeRateLimitProtocol)

    def test_missing_property_fails_rate_limit_protocol(self) -> None:
        assert not isinstance(_FakeRateLimitMissingEnabled(), RuntimeRateLimitProtocol)

    def test_legacy_only_rate_limit_shape_fails_protocol(self) -> None:
        """Protocol must include all attributes used by RateLimitRegistry."""
        assert not isinstance(_FakeRateLimitLegacyOnly(), RuntimeRateLimitProtocol)

    def test_matching_dataclass_satisfies_concurrency_protocol(self) -> None:
        assert isinstance(_FakeConcurrencyComplete(), RuntimeConcurrencyProtocol)

    def test_missing_property_fails_concurrency_protocol(self) -> None:
        assert not isinstance(_FakeConcurrencyMissing(), RuntimeConcurrencyProtocol)

    def test_matching_dataclass_satisfies_checkpoint_protocol(self) -> None:
        assert isinstance(_FakeCheckpointComplete(), RuntimeCheckpointProtocol)

    def test_missing_property_fails_checkpoint_protocol(self) -> None:
        assert not isinstance(_FakeCheckpointMissingFrequency(), RuntimeCheckpointProtocol)

    def test_matching_dataclass_satisfies_telemetry_protocol(self) -> None:
        assert isinstance(_FakeTelemetryComplete(), RuntimeTelemetryProtocol)

    def test_missing_property_fails_telemetry_protocol(self) -> None:
        assert not isinstance(_FakeTelemetryMissingGranularity(), RuntimeTelemetryProtocol)

    def test_unrelated_class_fails_all_protocols(self) -> None:
        class _Unrelated:
            x: int = 42

        obj = _Unrelated()
        for protocol in ALL_PROTOCOLS:
            assert not isinstance(obj, protocol)

    def test_extra_attributes_do_not_prevent_satisfaction(self) -> None:
        @dataclass
        class _WithExtras:
            max_workers: int = 4
            extra_field: str = "extra"

        assert isinstance(_WithExtras(), RuntimeConcurrencyProtocol)


class TestProtocolCompleteness:
    @pytest.mark.parametrize(
        "protocol,expected_props",
        [pytest.param(p, PROTOCOL_EXPECTED_PROPERTIES[p], id=p.__name__) for p in ALL_PROTOCOLS],
    )
    def test_protocol_has_expected_properties(self, protocol: type, expected_props: set[str]) -> None:
        actual = _get_protocol_property_names(protocol)
        assert actual == expected_props, (
            f"{protocol.__name__} properties mismatch.\nMissing: {expected_props - actual}\nExtra: {actual - expected_props}"
        )

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=_param_id_for)
    def test_no_non_property_public_methods(self, protocol: type) -> None:
        # Protocols may define explicit methods when required by runtime call sites.
        public_attrs = {name for name in dir(protocol) if not name.startswith("_")}
        property_attrs = _get_protocol_property_names(protocol)
        allowed_methods = PROTOCOL_EXPECTED_METHODS[protocol]
        non_property = public_attrs - property_attrs - allowed_methods
        assert not non_property, f"{protocol.__name__} has unexpected non-property public attributes: {non_property}"

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=_param_id_for)
    def test_protocol_properties_have_type_annotations(self, protocol: type) -> None:
        expected_props = PROTOCOL_EXPECTED_PROPERTIES[protocol]
        for prop_name in expected_props:
            ret_type = _get_protocol_property_return_type(protocol, prop_name)
            assert ret_type is not inspect.Parameter.empty, f"{protocol.__name__}.{prop_name} is missing a return type annotation"


class TestCrossValidation:
    @pytest.mark.parametrize("protocol,config_cls", PROTOCOL_CONFIG_PAIRS)
    def test_runtime_config_has_all_protocol_properties(self, protocol: type, config_cls: Any) -> None:
        protocol_props = _get_protocol_property_names(protocol)
        config_fields = set(config_cls.__dataclass_fields__.keys())
        missing = protocol_props - config_fields
        assert not missing, f"{config_cls.__name__} is missing protocol properties: {missing}"

    @pytest.mark.parametrize("protocol,config_cls", PROTOCOL_CONFIG_PAIRS)
    def test_protocol_property_types_match_config_annotations(self, protocol: type, config_cls: Any) -> None:
        protocol_props = _get_protocol_property_names(protocol)
        config_fields = config_cls.__dataclass_fields__

        for prop_name in protocol_props:
            assert prop_name in config_fields, f"{config_cls.__name__} missing field for {prop_name}"
            proto_ret = _get_protocol_property_return_type(protocol, prop_name)
            config_type = config_fields[prop_name].type
            # Normalize both sides to a base type name for comparison.
            # Protocol uses forward refs (strings) and config uses actual types.
            proto_name = _normalize_type_name(proto_ret)
            config_name = _normalize_type_name(config_type)
            assert proto_name == config_name, (
                f"{config_cls.__name__}.{prop_name} type mismatch: protocol expects {proto_name}, config has {config_name}"
            )

    @pytest.mark.parametrize("protocol,config_cls", PROTOCOL_CONFIG_PAIRS)
    def test_default_instance_exposes_all_protocol_properties(self, protocol: type, config_cls: Any) -> None:
        instance = config_cls.default()
        protocol_props = _get_protocol_property_names(protocol)
        for prop_name in protocol_props:
            # Access should not raise; value should not be sentinel
            value = getattr(instance, prop_name)
            assert value is not None or prop_name in {
                "persistence_path",
            }, f"{config_cls.__name__}.default().{prop_name} returned None unexpectedly"

    def test_retry_config_protocol_property_values_accessible(self) -> None:
        config = RuntimeRetryConfig.default()
        assert config.max_attempts >= 1
        assert config.base_delay > 0
        assert config.max_delay > 0
        assert config.exponential_base > 0
        assert config.jitter >= 0

    def test_telemetry_config_protocol_property_values_accessible(self) -> None:
        config = RuntimeTelemetryConfig.default()
        assert isinstance(config.enabled, bool)
        assert isinstance(config.granularity, TelemetryGranularity)
        assert isinstance(config.backpressure_mode, BackpressureMode)
        assert isinstance(config.fail_on_total_exporter_failure, bool)
        assert isinstance(config.exporter_configs, tuple)

    def test_checkpoint_config_protocol_property_values_accessible(self) -> None:
        config = RuntimeCheckpointConfig.default()
        assert isinstance(config.enabled, bool)
        assert isinstance(config.frequency, int)
