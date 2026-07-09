"""Tests for display_headers module."""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.header_modes import HeaderMode


class _StubSink:
    """Minimal sink that satisfies DisplayHeaderHost protocol for testing."""

    _headers_mode: HeaderMode
    _headers_custom_mapping: dict[str, str] | None
    _resolved_display_headers: dict[str, str] | None
    _display_headers_resolved: bool
    _needs_resume_field_resolution: bool
    _output_contract: Any

    def __init__(
        self,
        headers_mode: HeaderMode,
        headers_custom_mapping: dict[str, str] | None = None,
    ) -> None:
        # BaseSink.__init__ sets _output_contract BEFORE subclass init.
        # Must match this ordering — get_effective_display_headers reads _output_contract.
        self._output_contract = None

        from elspeth.plugins.infrastructure.display_headers import init_display_headers

        init_display_headers(self, headers_mode, headers_custom_mapping)

    def set_output_contract(self, contract: Any) -> None:
        self._output_contract = contract


class _GetSourceFieldResolution:
    def __init__(
        self,
        *,
        return_value: dict[str, str] | None = None,
        side_effect: BaseException | None = None,
    ) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls: list[str] = []

    def __call__(self, run_id: str) -> dict[str, str] | None:
        self.calls.append(run_id)
        if self.side_effect is not None:
            raise self.side_effect
        return self.return_value

    def assert_not_called(self) -> None:
        assert self.calls == []


class _LandscapeDouble:
    def __init__(
        self,
        *,
        field_resolution: dict[str, str] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.get_source_field_resolution = _GetSourceFieldResolution(return_value=field_resolution, side_effect=error)


_DEFAULT_LANDSCAPE = object()


class _ContextDouble:
    def __init__(
        self,
        *,
        contract: Any = None,
        landscape: _LandscapeDouble | None | object = _DEFAULT_LANDSCAPE,
        run_id: str = "run-1",
    ) -> None:
        self.contract = contract
        self.landscape = _LandscapeDouble() if landscape is _DEFAULT_LANDSCAPE else landscape
        self.run_id = run_id


class TestGetEffectiveDisplayHeaders:
    """Test get_effective_display_headers priority logic."""

    def test_normalized_mode_returns_none(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            get_effective_display_headers,
        )

        sink = _StubSink(HeaderMode.NORMALIZED)
        assert get_effective_display_headers(sink) is None

    def test_custom_mode_returns_mapping(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            get_effective_display_headers,
        )

        mapping = {"field_a": "Field A", "field_b": "Field B"}
        sink = _StubSink(HeaderMode.CUSTOM, mapping)
        assert get_effective_display_headers(sink) == mapping

    def test_custom_mode_with_none_mapping_raises(self) -> None:
        """CUSTOM mode with None mapping crashes at init — not a silent fallback."""
        with pytest.raises(ValueError, match="CUSTOM header mode requires an explicit headers_custom_mapping"):
            _StubSink(HeaderMode.CUSTOM, None)

    def test_original_mode_with_contract(self) -> None:
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract
        from elspeth.plugins.infrastructure.display_headers import (
            get_effective_display_headers,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="amount",
                    original_name="Amount USD",
                    python_type=float,
                    required=True,
                    source="declared",
                ),
            ),
        )
        sink.set_output_contract(contract)
        result = get_effective_display_headers(sink)
        assert result is not None
        assert result["amount"] == "Amount USD"

    def test_original_mode_with_resolved_headers_fallback(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            get_effective_display_headers,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        sink._resolved_display_headers = {"amount": "Amount USD"}
        assert get_effective_display_headers(sink) == {"amount": "Amount USD"}

    def test_original_mode_no_contract_no_resolved_returns_none(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            get_effective_display_headers,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        assert get_effective_display_headers(sink) is None


class TestDisplayNameFor:
    """Test display-name lookup helper."""

    def test_returns_field_when_no_mapping(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import display_name_for

        assert display_name_for(None, "amount_usd") == "amount_usd"

    def test_returns_mapped_display_name(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import display_name_for

        assert display_name_for({"amount_usd": "Amount USD"}, "amount_usd") == "Amount USD"

    def test_returns_field_when_mapping_omits_transform_added_field(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import display_name_for

        assert display_name_for({"amount_usd": "Amount USD"}, "derived_score") == "derived_score"


class TestResolveContractFromContext:
    """Test resolve_contract_from_context_if_needed."""

    def test_captures_context_contract_in_original_mode(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_contract_from_context_if_needed,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        ctx = _ContextDouble(contract=object())
        resolve_contract_from_context_if_needed(sink, ctx)
        assert sink._output_contract is ctx.contract

    def test_skips_if_not_original_mode(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_contract_from_context_if_needed,
        )

        sink = _StubSink(HeaderMode.NORMALIZED)
        ctx = _ContextDouble(contract=object())
        resolve_contract_from_context_if_needed(sink, ctx)
        assert sink._output_contract is None

    def test_skips_if_contract_already_set(self) -> None:
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_contract_from_context_if_needed,
        )

        existing = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name="amount",
                    original_name="Amount USD",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )
        sink = _StubSink(HeaderMode.ORIGINAL)
        sink._output_contract = existing
        ctx = _ContextDouble(contract=existing)
        resolve_contract_from_context_if_needed(sink, ctx)
        assert sink._output_contract is existing

    def test_raises_if_context_contract_conflicts_with_cached_original_header(self) -> None:
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_contract_from_context_if_needed,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        sink._output_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name="amount",
                    original_name="Amount A",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )
        ctx = _ContextDouble(
            contract=SchemaContract(
                mode="FLEXIBLE",
                fields=(
                    FieldContract(
                        normalized_name="amount",
                        original_name="Amount B",
                        python_type=int,
                        required=True,
                        source="declared",
                    ),
                ),
                locked=True,
            )
        )

        with pytest.raises(ValueError, match="same normalized field maps to different original headers"):
            resolve_contract_from_context_if_needed(sink, ctx)

    def test_merges_non_conflicting_context_contract_fields(self) -> None:
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_contract_from_context_if_needed,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        sink._output_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name="amount",
                    original_name="Amount USD",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )
        ctx = _ContextDouble(
            contract=SchemaContract(
                mode="FLEXIBLE",
                fields=(
                    FieldContract(
                        normalized_name="customer_id",
                        original_name="Customer ID",
                        python_type=str,
                        required=True,
                        source="declared",
                    ),
                ),
                locked=True,
            )
        )

        resolve_contract_from_context_if_needed(sink, ctx)

        assert sink._output_contract.find_field("amount").original_name == "Amount USD"
        assert sink._output_contract.find_field("customer_id").original_name == "Customer ID"


class TestResolveDisplayHeadersIfNeeded:
    """Test resolve_display_headers_if_needed."""

    def test_resolves_from_landscape(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_display_headers_if_needed,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        ctx = _ContextDouble(
            landscape=_LandscapeDouble(
                field_resolution={
                    "Amount USD": "amount_usd",
                    "Customer ID": "customer_id",
                }
            )
        )
        resolve_display_headers_if_needed(sink, ctx)
        # Should build reverse: normalized -> original
        assert sink._resolved_display_headers == {
            "amount_usd": "Amount USD",
            "customer_id": "Customer ID",
        }
        assert sink._display_headers_resolved is True

    def test_skips_when_already_resolved(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_display_headers_if_needed,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        sink._display_headers_resolved = True
        ctx = _ContextDouble()
        resolve_display_headers_if_needed(sink, ctx)
        # Should not call landscape
        ctx.landscape.get_source_field_resolution.assert_not_called()

    def test_errors_if_no_landscape(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_display_headers_if_needed,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        ctx = _ContextDouble(landscape=None)
        with pytest.raises(ValueError, match="requires Landscape"):
            resolve_display_headers_if_needed(sink, ctx)

    def test_errors_if_no_field_resolution(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_display_headers_if_needed,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        ctx = _ContextDouble(landscape=_LandscapeDouble(field_resolution=None))
        with pytest.raises(ValueError, match="did not record field resolution"):
            resolve_display_headers_if_needed(sink, ctx)

    def test_landscape_failure_does_not_permanently_trip_guard(self) -> None:
        """If the Landscape query raises, the next write() must retry resolution.

        This tests the fix for a pre-existing bug where _display_headers_resolved
        was set True BEFORE the query, permanently blocking retry on failure.
        """
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_display_headers_if_needed,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)

        # First call: Landscape query raises
        ctx = _ContextDouble(landscape=_LandscapeDouble(error=RuntimeError("DB locked")))
        with pytest.raises(RuntimeError, match="DB locked"):
            resolve_display_headers_if_needed(sink, ctx)

        # Guard must NOT be tripped — next call should retry
        assert sink._display_headers_resolved is False

        # Second call: Landscape query succeeds
        ctx.landscape.get_source_field_resolution.side_effect = None
        ctx.landscape.get_source_field_resolution.return_value = {
            "Amount USD": "amount_usd",
        }
        resolve_display_headers_if_needed(sink, ctx)
        assert sink._resolved_display_headers == {"amount_usd": "Amount USD"}
        assert sink._display_headers_resolved is True

    def test_duplicate_normalized_names_raises(self) -> None:
        """Landscape returns mapping where two originals collapse to same normalized name."""
        from elspeth.plugins.infrastructure.display_headers import (
            resolve_display_headers_if_needed,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        ctx = _ContextDouble(
            landscape=_LandscapeDouble(
                field_resolution={
                    "Amount A": "amount",
                    "Amount B": "amount",  # Duplicate normalized name
                }
            )
        )
        with pytest.raises(ValueError, match="duplicate normalized names"):
            resolve_display_headers_if_needed(sink, ctx)


class TestSetResumeFieldResolution:
    """Test set_resume_field_resolution."""

    def test_builds_reverse_mapping_in_original_mode(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            set_resume_field_resolution,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        set_resume_field_resolution(sink, {"Amount USD": "amount_usd"})
        assert sink._resolved_display_headers == {"amount_usd": "Amount USD"}
        assert sink._display_headers_resolved is True

    def test_noop_in_normalized_mode(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            set_resume_field_resolution,
        )

        sink = _StubSink(HeaderMode.NORMALIZED)
        set_resume_field_resolution(sink, {"Amount USD": "amount_usd"})
        assert sink._resolved_display_headers is None

    def test_duplicate_normalized_names_raises(self) -> None:
        """Two original names mapping to same normalized name is a data integrity issue."""
        from elspeth.plugins.infrastructure.display_headers import (
            set_resume_field_resolution,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        with pytest.raises(ValueError, match="duplicate normalized names"):
            set_resume_field_resolution(sink, {"A": "same", "B": "same"})


class TestApplyDisplayHeaders:
    """Test apply_display_headers."""

    def test_maps_keys(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            apply_display_headers,
        )

        sink = _StubSink(HeaderMode.CUSTOM, {"a": "Alpha", "b": "Beta"})
        rows = [{"a": 1, "b": 2}]
        result = apply_display_headers(sink, rows)
        assert result == [{"Alpha": 1, "Beta": 2}]

    def test_custom_mode_rejects_unmapped_keys(self) -> None:
        """CUSTOM mode must map every field — unmapped keys are a config error."""
        from elspeth.plugins.infrastructure.display_headers import (
            apply_display_headers,
        )

        sink = _StubSink(HeaderMode.CUSTOM, {"a": "Alpha"})
        rows = [{"a": 1, "extra": 99}]
        with pytest.raises(ValueError, match="CUSTOM header mode has no mapping for field 'extra'"):
            apply_display_headers(sink, rows)

    def test_collision_raises(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            apply_display_headers,
        )

        sink = _StubSink(HeaderMode.CUSTOM, {"a": "Same", "b": "Same"})
        with pytest.raises(ValueError, match="Header collision"):
            apply_display_headers(sink, [{"a": 1, "b": 2}])

    def test_no_mapping_returns_original(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            apply_display_headers,
        )

        sink = _StubSink(HeaderMode.NORMALIZED)
        rows = [{"a": 1}]
        assert apply_display_headers(sink, rows) is rows

    def test_empty_rows_returns_empty(self) -> None:
        from elspeth.plugins.infrastructure.display_headers import (
            apply_display_headers,
        )

        sink = _StubSink(HeaderMode.CUSTOM, {"a": "Alpha"})
        result = apply_display_headers(sink, [])
        assert result == []

    def test_custom_mode_empty_mapping_rejects_any_field(self) -> None:
        """CUSTOM mode with empty mapping dict rejects all fields."""
        from elspeth.plugins.infrastructure.display_headers import (
            apply_display_headers,
        )

        sink = _StubSink(HeaderMode.CUSTOM, {})
        rows = [{"a": 1, "b": 2}]
        with pytest.raises(ValueError, match="CUSTOM header mode has no mapping for field"):
            apply_display_headers(sink, rows)

    def test_custom_mode_unmapped_detected_before_collision(self) -> None:
        """CUSTOM mode detects unmapped field even if it would also collide."""
        from elspeth.plugins.infrastructure.display_headers import (
            apply_display_headers,
        )

        sink = _StubSink(HeaderMode.CUSTOM, {"a": "Alpha"})
        with pytest.raises(ValueError, match="CUSTOM header mode has no mapping for field"):
            apply_display_headers(sink, [{"a": 1, "Alpha": 2}])

    def test_original_mode_unmapped_keys_pass_through(self) -> None:
        """ORIGINAL mode allows unmapped keys (transform-added fields)."""
        from elspeth.plugins.infrastructure.display_headers import (
            apply_display_headers,
        )

        sink = _StubSink(HeaderMode.ORIGINAL)
        sink._resolved_display_headers = {"a": "Amount"}
        rows = [{"a": 1, "extra": 99}]
        result = apply_display_headers(sink, rows)
        assert result == [{"Amount": 1, "extra": 99}]

    def test_custom_mode_empty_rows_no_error(self) -> None:
        """CUSTOM mode with empty rows doesn't raise — nothing to validate."""
        from elspeth.plugins.infrastructure.display_headers import (
            apply_display_headers,
        )

        sink = _StubSink(HeaderMode.CUSTOM, {"a": "Alpha"})
        assert apply_display_headers(sink, []) == []


class TestInitDisplayHeaders:
    """Test init_display_headers precondition."""

    def test_raises_if_output_contract_not_set(self) -> None:
        """Sinks must call super().__init__() before init_display_headers().

        Per CLAUDE.md offensive programming: direct attribute access crashes
        with AttributeError if _output_contract is missing — no defensive
        try/except wrapping.
        """
        from elspeth.plugins.infrastructure.display_headers import init_display_headers

        class _BadSink:
            pass  # No _output_contract — forgot super().__init__()

        with pytest.raises(AttributeError):
            init_display_headers(_BadSink(), HeaderMode.NORMALIZED)  # type: ignore[arg-type]
