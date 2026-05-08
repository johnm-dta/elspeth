# tests/unit/contracts/config/test_alignment.py
"""Tests for contracts.config.alignment — field mapping documentation."""

from typing import ClassVar

import pytest

from elspeth.contracts.config.alignment import (
    EXEMPT_SETTINGS,
    FIELD_MAPPINGS,
    RUNTIME_TO_SUBSYSTEM,
    SETTINGS_TO_RUNTIME,
    get_runtime_field_name,
    get_settings_field_name,
    is_exempt_settings,
)

# ---------------------------------------------------------------------------
# FIELD_MAPPINGS
# ---------------------------------------------------------------------------


class TestFieldMappings:
    def test_retry_settings_has_two_mappings(self) -> None:
        # KEEP — unique drift detection. Verified 2026-05-08:
        # test_top_level_keys_are_expected_set only checks the OUTER
        # FIELD_MAPPINGS keys; an addition to this INNER dict is caught
        # by NO other test. Do not delete in tautology sweeps.
        assert len(FIELD_MAPPINGS["RetrySettings"]) == 2, (
            "FIELD_MAPPINGS['RetrySettings'] inner-dict size drifted; an entry was added or removed and no cross-validation test caught it"
        )

    def test_retry_settings_initial_delay_mapped(self) -> None:
        assert FIELD_MAPPINGS["RetrySettings"]["initial_delay_seconds"] == "base_delay"

    def test_retry_settings_max_delay_mapped(self) -> None:
        assert FIELD_MAPPINGS["RetrySettings"]["max_delay_seconds"] == "max_delay"

    def test_telemetry_settings_has_one_mapping(self) -> None:
        # KEEP — unique drift detection. Same rationale as
        # test_retry_settings_has_two_mappings: surviving cross-validation
        # tests check OUTER keys only, so additions to this INNER dict
        # would otherwise be silent.
        assert len(FIELD_MAPPINGS["TelemetrySettings"]) == 1, (
            "FIELD_MAPPINGS['TelemetrySettings'] inner-dict size drifted; "
            "an entry was added or removed and no cross-validation test caught it"
        )

    def test_telemetry_settings_exporters_mapped(self) -> None:
        assert FIELD_MAPPINGS["TelemetrySettings"]["exporters"] == "exporter_configs"

    def test_top_level_keys_are_expected_set(self) -> None:
        assert set(FIELD_MAPPINGS.keys()) == {"RetrySettings", "TelemetrySettings"}

    @pytest.mark.parametrize("cls", list(FIELD_MAPPINGS.keys()))
    def test_all_settings_field_names_are_nonempty_strings(self, cls: str) -> None:
        for settings_field, runtime_field in FIELD_MAPPINGS[cls].items():
            assert isinstance(settings_field, str) and settings_field
            assert isinstance(runtime_field, str) and runtime_field

    @pytest.mark.parametrize("cls", list(FIELD_MAPPINGS.keys()))
    def test_no_identity_mappings(self, cls: str) -> None:
        for settings_field, runtime_field in FIELD_MAPPINGS[cls].items():
            assert settings_field != runtime_field, (
                f"Identity mapping in {cls}: {settings_field} -> {runtime_field} (same-name fields should not be in FIELD_MAPPINGS)"
            )


# ---------------------------------------------------------------------------
# SETTINGS_TO_RUNTIME
# ---------------------------------------------------------------------------


class TestSettingsToRuntime:
    EXPECTED_PAIRS: ClassVar[dict[str, str]] = {
        "RetrySettings": "RuntimeRetryConfig",
        "RateLimitSettings": "RuntimeRateLimitConfig",
        "ConcurrencySettings": "RuntimeConcurrencyConfig",
        "CheckpointSettings": "RuntimeCheckpointConfig",
        "TelemetrySettings": "RuntimeTelemetryConfig",
    }

    def test_contains_all_expected_entries(self) -> None:
        for key, value in self.EXPECTED_PAIRS.items():
            assert SETTINGS_TO_RUNTIME[key] == value

    def test_has_exactly_five_entries(self) -> None:
        # KEEP — unique drift detection. Verified 2026-05-08:
        # test_contains_all_expected_entries (line 70) is a SUBSET check
        # (asserts each expected pair is present); it cannot detect
        # ADDITIONS to SETTINGS_TO_RUNTIME. This count test is the only
        # gate against an unintended new entry.
        assert len(SETTINGS_TO_RUNTIME) == 5, (
            "SETTINGS_TO_RUNTIME size drifted; an entry was added or removed "
            "and the subset-check test_contains_all_expected_entries did not "
            "catch it (it only verifies expected entries are present)"
        )

    @pytest.mark.parametrize(
        "settings_cls,runtime_cls",
        [
            ("RetrySettings", "RuntimeRetryConfig"),
            ("RateLimitSettings", "RuntimeRateLimitConfig"),
            ("ConcurrencySettings", "RuntimeConcurrencyConfig"),
            ("CheckpointSettings", "RuntimeCheckpointConfig"),
            ("TelemetrySettings", "RuntimeTelemetryConfig"),
        ],
    )
    def test_individual_mapping(self, settings_cls: str, runtime_cls: str) -> None:
        assert SETTINGS_TO_RUNTIME[settings_cls] == runtime_cls

    def test_all_keys_end_with_settings(self) -> None:
        for key in SETTINGS_TO_RUNTIME:
            assert key.endswith("Settings"), f"{key} does not end with 'Settings'"

    def test_all_values_start_with_runtime(self) -> None:
        for value in SETTINGS_TO_RUNTIME.values():
            assert value.startswith("Runtime"), f"{value} does not start with 'Runtime'"


# ---------------------------------------------------------------------------
# EXEMPT_SETTINGS
# ---------------------------------------------------------------------------


class TestExemptSettings:
    EXPECTED_EXEMPT: ClassVar[set[str]] = {
        "SourceSettings",
        "TransformSettings",
        "SinkSettings",
        "AggregationSettings",
        "GateSettings",
        "CoalesceSettings",
        "TriggerConfig",
        "DatabaseSettings",
        "LandscapeSettings",
        "LandscapeExportSettings",
        "PayloadStoreSettings",
        "ServiceRateLimit",
        "ExporterSettings",
        "ElspethSettings",
    }

    def test_contains_all_expected_entries(self) -> None:
        for entry in self.EXPECTED_EXEMPT:
            assert entry in EXEMPT_SETTINGS, f"{entry} missing from EXEMPT_SETTINGS"

    def test_all_entries_are_nonempty_strings(self) -> None:
        for entry in EXEMPT_SETTINGS:
            assert isinstance(entry, str) and entry

    @pytest.mark.parametrize(
        "mapped_cls",
        list(SETTINGS_TO_RUNTIME.keys()),
    )
    def test_mapped_settings_not_exempt(self, mapped_cls: str) -> None:
        assert mapped_cls not in EXEMPT_SETTINGS, f"{mapped_cls} appears in both SETTINGS_TO_RUNTIME and EXEMPT_SETTINGS"

    def test_expected_subset_present(self) -> None:
        assert self.EXPECTED_EXEMPT <= EXEMPT_SETTINGS


# ---------------------------------------------------------------------------
# RUNTIME_TO_SUBSYSTEM
# ---------------------------------------------------------------------------


class TestRuntimeToSubsystem:
    def test_retry_config_maps_to_retry(self) -> None:
        assert RUNTIME_TO_SUBSYSTEM["RuntimeRetryConfig"] == "retry"

    def test_has_at_least_one_entry(self) -> None:
        assert len(RUNTIME_TO_SUBSYSTEM) >= 1

    def test_all_keys_are_nonempty_strings(self) -> None:
        for key in RUNTIME_TO_SUBSYSTEM:
            assert isinstance(key, str) and key

    def test_all_values_are_nonempty_strings(self) -> None:
        for value in RUNTIME_TO_SUBSYSTEM.values():
            assert isinstance(value, str) and value

    def test_all_keys_are_runtime_class_names(self) -> None:
        for key in RUNTIME_TO_SUBSYSTEM:
            assert key.startswith("Runtime"), f"{key} does not start with 'Runtime'"


# ---------------------------------------------------------------------------
# get_runtime_field_name()
# ---------------------------------------------------------------------------


class TestGetRuntimeFieldName:
    @pytest.mark.parametrize(
        "settings_field,expected",
        [
            ("initial_delay_seconds", "base_delay"),
            ("max_delay_seconds", "max_delay"),
        ],
    )
    def test_mapped_retry_fields(self, settings_field: str, expected: str) -> None:
        assert get_runtime_field_name("RetrySettings", settings_field) == expected

    def test_mapped_telemetry_field(self) -> None:
        assert get_runtime_field_name("TelemetrySettings", "exporters") == "exporter_configs"

    def test_unmapped_field_in_mapped_class_returns_same_name(self) -> None:
        assert get_runtime_field_name("RetrySettings", "max_attempts") == "max_attempts"

    def test_unmapped_class_returns_same_name(self) -> None:
        assert get_runtime_field_name("ConcurrencySettings", "max_workers") == "max_workers"

    def test_unknown_class_returns_same_name(self) -> None:
        assert get_runtime_field_name("FakeSettings", "some_field") == "some_field"

    def test_unknown_field_in_known_class_returns_same_name(self) -> None:
        assert get_runtime_field_name("RetrySettings", "nonexistent") == "nonexistent"

    def test_empty_field_name_returns_empty(self) -> None:
        assert get_runtime_field_name("RetrySettings", "") == ""

    def test_empty_class_name_returns_field_as_is(self) -> None:
        assert get_runtime_field_name("", "some_field") == "some_field"


# ---------------------------------------------------------------------------
# get_settings_field_name()
# ---------------------------------------------------------------------------


class TestGetSettingsFieldName:
    @pytest.mark.parametrize(
        "runtime_field,expected",
        [
            ("base_delay", "initial_delay_seconds"),
            ("max_delay", "max_delay_seconds"),
        ],
    )
    def test_mapped_retry_fields_reverse(self, runtime_field: str, expected: str) -> None:
        assert get_settings_field_name("RetrySettings", runtime_field) == expected

    def test_mapped_telemetry_field_reverse(self) -> None:
        assert get_settings_field_name("TelemetrySettings", "exporter_configs") == "exporters"

    def test_unmapped_field_in_mapped_class_returns_same_name(self) -> None:
        assert get_settings_field_name("RetrySettings", "max_attempts") == "max_attempts"

    def test_unmapped_class_returns_same_name(self) -> None:
        assert get_settings_field_name("ConcurrencySettings", "max_workers") == "max_workers"

    def test_unknown_class_returns_same_name(self) -> None:
        assert get_settings_field_name("FakeSettings", "some_field") == "some_field"

    def test_unknown_field_in_known_class_returns_same_name(self) -> None:
        assert get_settings_field_name("RetrySettings", "nonexistent") == "nonexistent"

    def test_empty_runtime_field_returns_empty(self) -> None:
        assert get_settings_field_name("RetrySettings", "") == ""

    def test_empty_class_returns_field_as_is(self) -> None:
        assert get_settings_field_name("", "some_field") == "some_field"


# ---------------------------------------------------------------------------
# is_exempt_settings()
# ---------------------------------------------------------------------------


class TestIsExemptSettings:
    @pytest.mark.parametrize(
        "cls",
        [
            "SourceSettings",
            "TransformSettings",
            "SinkSettings",
            "ElspethSettings",
            "DatabaseSettings",
            "LandscapeSettings",
            "TriggerConfig",
            "ExporterSettings",
        ],
    )
    def test_returns_true_for_exempt_classes(self, cls: str) -> None:
        assert is_exempt_settings(cls) is True

    @pytest.mark.parametrize(
        "cls",
        [
            "RetrySettings",
            "RateLimitSettings",
            "ConcurrencySettings",
            "CheckpointSettings",
            "TelemetrySettings",
        ],
    )
    def test_returns_false_for_mapped_classes(self, cls: str) -> None:
        assert is_exempt_settings(cls) is False

    def test_returns_false_for_unknown_class(self) -> None:
        assert is_exempt_settings("CompletelyMadeUp") is False

    def test_returns_false_for_empty_string(self) -> None:
        assert is_exempt_settings("") is False


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


class TestCrossValidation:
    def test_mapped_and_exempt_are_disjoint(self) -> None:
        overlap = set(SETTINGS_TO_RUNTIME.keys()) & EXEMPT_SETTINGS
        assert overlap == set(), f"Classes in both SETTINGS_TO_RUNTIME and EXEMPT_SETTINGS: {overlap}"

    def test_field_mappings_keys_subset_of_settings_to_runtime(self) -> None:
        orphaned = set(FIELD_MAPPINGS.keys()) - set(SETTINGS_TO_RUNTIME.keys())
        assert orphaned == set(), f"FIELD_MAPPINGS has classes not in SETTINGS_TO_RUNTIME: {orphaned}"

    def test_runtime_to_subsystem_keys_are_valid_runtime_classes(self) -> None:
        valid_runtimes = set(SETTINGS_TO_RUNTIME.values())
        for key in RUNTIME_TO_SUBSYSTEM:
            assert key in valid_runtimes, f"RUNTIME_TO_SUBSYSTEM key '{key}' not in SETTINGS_TO_RUNTIME values"

    @pytest.mark.parametrize("cls", list(FIELD_MAPPINGS.keys()))
    def test_get_runtime_and_get_settings_are_inverses(self, cls: str) -> None:
        for settings_field, runtime_field in FIELD_MAPPINGS[cls].items():
            assert get_runtime_field_name(cls, settings_field) == runtime_field
            assert get_settings_field_name(cls, runtime_field) == settings_field

    def test_all_mapped_fields_roundtrip(self) -> None:
        for cls, mappings in FIELD_MAPPINGS.items():
            for settings_field in mappings:
                runtime_field = get_runtime_field_name(cls, settings_field)
                back = get_settings_field_name(cls, runtime_field)
                assert back == settings_field, f"Roundtrip failed for {cls}.{settings_field}: -> {runtime_field} -> {back}"

    def test_unmapped_fields_roundtrip_as_identity(self) -> None:
        field = "unchanged_field"
        for cls in SETTINGS_TO_RUNTIME:
            rt = get_runtime_field_name(cls, field)
            back = get_settings_field_name(cls, rt)
            assert rt == field
            assert back == field
