"""Validation tests for reserved system-prefix routing names."""

import pytest
from pydantic import ValidationError


class TestConnectionNameValidation:
    """Connection/destination names must not use system-reserved prefixes."""

    def test_source_on_success_rejects_dunder_prefix(self) -> None:
        from elspeth.core.config import SourceSettings

        with pytest.raises(ValidationError, match="starts with '__'"):
            SourceSettings(plugin="csv", on_success="__quarantine__", options={})

    def test_transform_input_rejects_dunder_prefix(self) -> None:
        from elspeth.core.config import TransformSettings

        with pytest.raises(ValidationError, match="starts with '__'"):
            TransformSettings(name="t0", plugin="passthrough", input="__internal__", on_success="output", on_error="discard", options={})

    def test_transform_on_success_rejects_dunder_prefix(self) -> None:
        from elspeth.core.config import TransformSettings

        with pytest.raises(ValidationError, match="starts with '__'"):
            TransformSettings(
                name="t0",
                plugin="passthrough",
                input="source_out",
                on_success="__quarantine__",
                on_error="discard",
                options={},
            )

    def test_transform_on_error_rejects_dunder_prefix(self) -> None:
        from elspeth.core.config import TransformSettings

        with pytest.raises(ValidationError, match="starts with '__'"):
            TransformSettings(
                name="t0",
                plugin="passthrough",
                input="source_out",
                on_success="output",
                on_error="__error_0__",
                options={},
            )

    def test_transform_on_error_rejects_empty_string(self) -> None:
        from elspeth.core.config import TransformSettings

        with pytest.raises(ValidationError, match="on_error must be a sink name or 'discard'"):
            TransformSettings(
                name="t0",
                plugin="passthrough",
                input="source_out",
                on_success="output",
                on_error="",
                options={},
            )

    def test_transform_on_error_discard_remains_valid(self) -> None:
        from elspeth.core.config import TransformSettings

        settings = TransformSettings(
            name="t0",
            plugin="passthrough",
            input="source_out",
            on_success="output",
            on_error="discard",
            options={},
        )
        assert settings.on_error == "discard"

    def test_gate_input_rejects_dunder_prefix(self) -> None:
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError, match="starts with '__'"):
            GateSettings(
                name="g0",
                input="__internal__",
                condition="True",
                routes={"true": "next_a", "false": "next_b"},
            )

    def test_aggregation_input_rejects_dunder_prefix(self) -> None:
        from elspeth.core.config import AggregationSettings, TriggerConfig

        with pytest.raises(ValidationError, match="starts with '__'"):
            AggregationSettings(
                name="agg0",
                plugin="batch_stats",
                input="__internal__",
                on_error="discard",
                trigger=TriggerConfig(count=5),
                options={},
            )

    def test_aggregation_on_success_rejects_dunder_prefix(self) -> None:
        from elspeth.core.config import AggregationSettings, TriggerConfig

        with pytest.raises(ValidationError, match="starts with '__'"):
            AggregationSettings(
                name="agg0",
                plugin="batch_stats",
                input="agg_in",
                on_success="__quarantine__",
                on_error="discard",
                trigger=TriggerConfig(count=5),
                options={},
            )

    def test_coalesce_on_success_rejects_dunder_prefix(self) -> None:
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError, match="starts with '__'"):
            CoalesceSettings(
                name="merge0",
                branches=["path_a", "path_b"],
                policy="require_all",
                merge="union",
                on_success="__quarantine__",
            )

    def test_source_on_success_rejects_reserved_label(self) -> None:
        from elspeth.core.config import SourceSettings

        with pytest.raises(ValidationError, match="reserved"):
            SourceSettings(plugin="csv", on_success="on_success", options={})

    def test_transform_name_rejects_too_long_identifier(self) -> None:
        from elspeth.core.config import TransformSettings

        with pytest.raises(ValidationError, match="exceeds max length"):
            TransformSettings(
                name="t" * 39,
                plugin="passthrough",
                input="source_out",
                on_success="output",
                on_error="discard",
                options={},
            )

    def test_connection_name_rejects_too_long_identifier(self) -> None:
        from elspeth.core.config import SourceSettings

        with pytest.raises(ValidationError, match="exceeds max length"):
            SourceSettings(plugin="csv", on_success="c" * 65, options={})

    def test_gate_route_label_rejects_reserved_on_success(self) -> None:
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError, match="reserved"):
            GateSettings(
                name="g0",
                input="source_out",
                condition="row['route']",
                routes={"on_success": "output"},
            )

    def test_gate_route_destination_rejects_reserved_on_success(self) -> None:
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError, match="reserved"):
            GateSettings(
                name="g0",
                input="source_out",
                condition="row['route']",
                routes={"route_a": "on_success"},
            )

    def test_sink_name_rejects_reserved_on_success(self) -> None:
        from elspeth.core.config import ElspethSettings

        with pytest.raises(ValidationError, match="reserved"):
            ElspethSettings(
                sources={"primary": {"plugin": "csv", "on_success": "source_out"}},
                sinks={"on_success": {"plugin": "json", "on_write_failure": "discard"}},
            )
