"""Tests for dry-run validation using real engine code paths.

Validation calls the actual engine functions: load_settings_from_yaml_string(),
instantiate_plugins_from_config(), ExecutionGraph.from_plugin_instances(),
graph.validate(). No parallel validation logic exists.

W18 fix: Only typed exceptions are caught — no bare except Exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.secrets import ResolvedSecret, SecretInventoryItem
from elspeth.core.dag.models import GraphValidationError
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.manager import PluginNotFoundError
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.config import WebSettings
from elspeth.web.execution.validation import (
    _collect_secret_refs,
    _infer_component_type_from_plugin_error,
    validate_pipeline,
)


def _make_source(options: dict[str, Any] | None = None) -> SourceSpec:
    """Build a SourceSpec with sensible defaults for validation tests."""
    return SourceSpec(
        plugin="csv",
        on_success="transform_in",
        options=options or {},
        on_validation_failure="discard",
    )


def _make_node(options: dict[str, Any] | None = None) -> NodeSpec:
    """Build a NodeSpec with sensible defaults for validation tests."""
    return NodeSpec(
        id="test_node",
        node_type="transform",
        plugin="value_transform",
        input="transform_in",
        on_success="results",
        on_error="discard",
        options=options or {},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def _make_output(
    options: dict[str, Any] | None = None,
    name: str = "primary",
) -> OutputSpec:
    """Build an OutputSpec with sensible defaults for validation tests."""
    return OutputSpec(
        name=name,
        plugin="csv",
        options=options or {},
        on_write_failure="discard",
    )


def _make_state(
    source_options: dict[str, Any] | None = None,
    nodes: tuple[NodeSpec, ...] | None = None,
    outputs: tuple[OutputSpec, ...] | None = None,
) -> CompositionState:
    """Build a CompositionState with sensible defaults for validation tests.

    When source_options is not None, a SourceSpec is created with those options.
    When source_options is None, source is set to None.
    """
    source = _make_source(source_options) if source_options is not None else None
    return CompositionState(
        source=source,
        nodes=nodes or (),
        edges=(),
        outputs=outputs or (),
        metadata=PipelineMetadata(),
        version=1,
    )


def _make_settings(data_dir: str = "/tmp/test_data") -> WebSettings:
    """Build a WebSettings with sensible defaults for validation tests."""
    return WebSettings(
        data_dir=Path(data_dir),
        composer_max_composition_turns=10,
        composer_max_discovery_turns=5,
        composer_timeout_seconds=30.0,
        composer_rate_limit_per_minute=60,
    )


def _check(result, name: str):
    """Look up a validation check by name, not position."""
    return next(c for c in result.checks if c.name == name)


class TestValidatePipelinePathAllowlist:
    """C3/S2: Source path allowlist check — defense-in-depth."""

    def test_path_within_blobs_passes(self) -> None:
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/data.csv"},
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        # B11: path check is always recorded — verify it passed
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True

    def test_path_outside_blobs_blocked(self) -> None:
        state = _make_state(
            source_options={"path": "/etc/passwd"},
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False
        assert _check(result, "path_allowlist").passed is False
        assert any("Path traversal" in e.message for e in result.errors)

    def test_path_traversal_via_dotdot_blocked(self) -> None:
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/../../secret.csv"},
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False

    def test_no_path_option_records_skipped_check(self) -> None:
        """B11 fix: path allowlist check is always recorded, even when skipped."""
        state = _make_state(source_options={})
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        # B11: check IS recorded with passed=True and "skipped" detail
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True
        assert "skipped" in path_check.detail.lower()


class TestValidatePipelineBatchTransformOptions:
    """ADR-013 composer/runtime agreement for batch-aware transform options."""

    def test_required_input_fields_returns_structured_validation_error(self) -> None:
        source = SourceSpec(
            plugin="csv",
            on_success="agg1",
            options={"schema": {"mode": "fixed", "fields": ["amount: float"]}},
            on_validation_failure="discard",
        )
        agg = NodeSpec(
            id="agg1",
            node_type="aggregation",
            plugin="batch_stats",
            input="agg1",
            on_success="primary",
            on_error="discard",
            options={
                "schema": {"mode": "observed"},
                "value_field": "amount",
                "required_input_fields": ["amount"],
            },
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = CompositionState(
            source=source,
            nodes=(agg,),
            edges=(),
            outputs=(_make_output({"schema": {"mode": "observed"}}),),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n"

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "batch_transform_options").passed is False
        messages = "\n".join(error.message for error in result.errors)
        assert "required_input_fields" in messages
        assert "batch-aware" in messages
        mock_yaml_gen.generate_yaml.assert_not_called()


class TestValidatePipelineSinkPathAllowlist:
    """Sink path allowlist — prevents arbitrary file writes via sink options."""

    def test_sink_path_outside_outputs_blocked(self) -> None:
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="evil_sink", options={"path": "/etc/cron.d/backdoor.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False
        assert any("Path traversal" in e.message for e in result.errors)
        assert any("evil_sink" in e.message for e in result.errors)

    def test_sink_path_traversal_blocked(self) -> None:
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="tricky", options={"path": "/tmp/test_data/outputs/../../etc/passwd"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False

    def test_sink_path_under_outputs_passes(self) -> None:
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="primary", options={"path": "/tmp/test_data/outputs/result.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True
        assert "All paths within allowed directories" in path_check.detail

    def test_sink_path_under_blobs_passes(self) -> None:
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="blob_out", options={"path": "/tmp/test_data/blobs/out.json"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True

    def test_sink_without_path_passes(self) -> None:
        """Sinks without path/file options (e.g. database) skip the check."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="db_sink", options={"connection_string": "sqlite:///out.db"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True


class TestValidatePipelineSemanticContractsLegacy:
    """Validation must catch transform pairings that violate line framing.

    Renamed from TestValidatePipelineTransformFraming when the
    transform_framing check was replaced with the generic
    semantic_contracts check (Phase 4 Task 4.3). The web_scrape ->
    line_explode regression surface remains the same; only the check
    name in the response changes.
    """

    @staticmethod
    def _make_web_scrape_line_explode_state(
        *,
        scrape_options: dict[str, Any] | None = None,
    ) -> CompositionState:
        web_scrape_options = {
            "schema": {"mode": "flexible", "fields": ["url: str"]},
            "required_input_fields": ["url"],
            "url_field": "url",
            "content_field": "content",
            "fingerprint_field": "content_fingerprint",
            "format": "text",
            "fingerprint_mode": "content",
            "http": {
                "abuse_contact": "pipeline@example.com",
                "scraping_reason": "test scrape",
                "allowed_hosts": "public_only",
            },
        }
        web_scrape_options.update(scrape_options or {})
        return CompositionState(
            source=SourceSpec(
                plugin="text",
                on_success="scrape_in",
                options={
                    "path": "/tmp/test_data/blobs/urls.txt",
                    "column": "url",
                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                },
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="scrape_page",
                    node_type="transform",
                    plugin="web_scrape",
                    input="scrape_in",
                    on_success="explode_in",
                    on_error="discard",
                    options=web_scrape_options,
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="split_lines",
                    node_type="transform",
                    plugin="line_explode",
                    input="explode_in",
                    on_success="results",
                    on_error="discard",
                    options={
                        "schema": {
                            "mode": "flexible",
                            "fields": [
                                "url: str",
                                "content: str",
                                "content_fingerprint: str",
                            ],
                        },
                        "required_input_fields": ["content"],
                        "source_field": "content",
                        "output_field": "line",
                        "include_index": True,
                        "index_field": "line_index",
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(
                OutputSpec(
                    name="results",
                    plugin="json",
                    options={"path": "/tmp/test_data/outputs/lines.json", "format": "json"},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )

    def test_compact_web_scrape_text_fails_before_yaml_generation(self) -> None:
        state = self._make_web_scrape_line_explode_state()
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "semantic_contracts").passed is False
        # The semantic validator's diagnostic names the requirement code
        # and the observed framing; the legacy framing validator named
        # "text_separator". Both validators run in Phase 4-5; whichever
        # produced an error first short-circuits, so we accept either
        # surface form. Phase 6 deletes the legacy validator.
        assert any(
            "text_separator" in error.message or "line_framed_text" in error.message or "text_framing" in error.message
            for error in result.errors
        )
        mock_yaml_gen.generate_yaml.assert_not_called()

    def test_newline_framed_web_scrape_text_reaches_yaml_generation(self) -> None:
        state = self._make_web_scrape_line_explode_state(
            scrape_options={"text_separator": "\n"},
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert _check(result, "semantic_contracts").passed is True
        mock_yaml_gen.generate_yaml.assert_called_once_with(state)


class TestValidatePipelineSemanticContracts:
    """The /validate route must surface the semantic_contracts check.

    Uses a wardline-shape state (web_scrape -> line_explode -> sink) but
    with paths that pass the path_allowlist, so semantic_contracts is
    actually exercised. The Phase 3 _wardline_state fixture is composer-
    test-shaped (paths like ``data/url.csv``) and would short-circuit at
    path_allowlist when fed through validate_pipeline.
    """

    @staticmethod
    def _make_state(text_separator: str = " ") -> CompositionState:
        return CompositionState(
            metadata=PipelineMetadata(name="wardline"),
            version=1,
            edges=(),
            source=SourceSpec(
                plugin="csv",
                on_success="scrape_in",
                options={
                    "path": "/tmp/test_data/blobs/url.csv",
                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                },
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="scrape",
                    node_type="transform",
                    plugin="web_scrape",
                    input="scrape_in",
                    on_success="explode_in",
                    on_error="errors",
                    options={
                        "schema": {"mode": "flexible", "fields": ["url: str"]},
                        "required_input_fields": ["url"],
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "fingerprint",
                        "format": "text",
                        "text_separator": text_separator,
                        "http": {
                            "abuse_contact": "x@example.com",
                            "scraping_reason": "t",
                            "timeout": 5,
                            "allowed_hosts": "public_only",
                        },
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="explode",
                    node_type="transform",
                    plugin="line_explode",
                    input="explode_in",
                    on_success="sink",
                    on_error="errors",
                    options={
                        "schema": {"mode": "flexible", "fields": ["content: str"]},
                        "source_field": "content",
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            outputs=(
                OutputSpec(
                    name="sink",
                    plugin="json",
                    options={"path": "/tmp/test_data/outputs/out.json"},
                    on_write_failure="discard",
                ),
                OutputSpec(
                    name="errors",
                    plugin="json",
                    options={"path": "/tmp/test_data/outputs/err.json"},
                    on_write_failure="discard",
                ),
            ),
        )

    def test_compact_text_fails_with_semantic_contracts_check_name(self) -> None:
        state = self._make_state(text_separator=" ")
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "semantic_contracts").passed is False

    def test_newline_text_passes_semantic_contracts_check(self) -> None:
        state = self._make_state(text_separator="\n")
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        # Subsequent checks may still fail (depends on fixture); we only
        # assert semantic_contracts itself passed.
        assert _check(result, "semantic_contracts").passed is True


class TestValidatePipelineRelativePaths:
    """Relative paths must resolve against data_dir, not CWD."""

    def test_relative_sink_path_resolves_against_data_dir(self) -> None:
        """outputs/result.csv should resolve under {data_dir}/outputs/."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="primary", options={"path": "outputs/result.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True

    def test_relative_source_path_resolves_against_data_dir(self) -> None:
        """blobs/data.csv should resolve under {data_dir}/blobs/."""
        state = _make_state(
            source_options={"path": "blobs/data.csv"},
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True

    def test_relative_traversal_still_blocked(self) -> None:
        """../etc/passwd relative to data_dir must still be blocked."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="evil", options={"path": "../etc/passwd"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False
        assert any("Path traversal" in e.message for e in result.errors)

    def test_relative_sink_path_under_blobs(self) -> None:
        """blobs/out.json should resolve under {data_dir}/blobs/."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="blob_out", options={"path": "blobs/out.json"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True


class TestValidatePipelineSuccess:
    @patch("elspeth.web.execution.validation.assemble_and_validate_pipeline_config")
    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_valid_pipeline_returns_all_checks_passed(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_assemble: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        mock_settings = MagicMock()
        mock_load.return_value = mock_settings

        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_assemble.return_value = MagicMock()

        state = _make_state()
        settings = _make_settings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is True
        assert len(result.checks) == 10
        assert all(c.passed for c in result.checks)
        # B11 fix: path_allowlist check is always recorded
        assert _check(result, "path_allowlist").passed is True
        assert _check(result, "secret_refs").passed is True
        assert _check(result, "semantic_contracts").passed is True
        assert _check(result, "batch_transform_options").passed is True
        assert _check(result, "value_source_compliance").passed is True
        assert _check(result, "route_target_resolution").passed is True
        assert result.errors == []

        # Verify real engine functions were called
        mock_load.assert_called_once()
        mock_instantiate.assert_called_once_with(mock_settings, preflight_mode=True)
        mock_build_graph.assert_called_once()
        mock_graph.validate.assert_called_once()
        mock_assemble.assert_called_once()
        mock_graph.validate_edge_compatibility.assert_called_once()


class TestValidatePipelineSettingsFailure:
    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    def test_pydantic_validation_error_short_circuits(
        self,
        mock_load: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "bad: yaml"
        # Note: from_exception_data() is a Pydantic v2 internal API. If this breaks
        # on a Pydantic upgrade, replace with: `ElspethSettings(bad_field="x")` to
        # trigger a real PydanticValidationError.
        mock_load.side_effect = PydanticValidationError.from_exception_data(
            title="ElspethSettings",
            line_errors=[
                {
                    "type": "missing",
                    "loc": ("source",),
                    "input": {},
                }
            ],
        )

        state = _make_state()
        settings = _make_settings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "path_allowlist").passed is True
        assert _check(result, "secret_refs").passed is True
        assert _check(result, "settings_load").passed is False
        # Downstream checks are skipped but recorded
        skipped = [c for c in result.checks if "Skipped" in c.detail]
        assert len(skipped) >= 1
        assert all(not c.passed for c in skipped)
        assert len(result.errors) >= 1

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    def test_file_not_found_error_from_settings(
        self,
        mock_load: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source: {}"
        mock_load.side_effect = ValueError("invalid settings")

        state = _make_state()
        settings = _make_settings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "settings_load").passed is False


class TestValidatePipelinePluginFailure:
    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    def test_unknown_plugin_returns_attributed_error(
        self,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: unknown"
        mock_load.return_value = MagicMock()
        from elspeth.plugins.infrastructure.manager import PluginNotFoundError

        mock_instantiate.side_effect = PluginNotFoundError("Unknown source plugin: 'unknown'")

        state = _make_state()
        settings = _make_settings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "settings_load").passed is True
        assert _check(result, "plugin_instantiation").passed is False
        assert any("unknown" in e.message.lower() for e in result.errors)

    def test_real_text_source_config_error_returns_validation_result(self) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: text
  on_success: transform_in
  options:
    on_validation_failure: discard
transforms:
- name: append_world
  plugin: value_transform
  input: transform_in
  on_success: results
  on_error: results
  options:
    schema:
      mode: fixed
      fields:
      - 'line: str'
      - 'result: str'
    operations:
    - target: result
      expression: row['line'] + ' world'
sinks:
  results:
    plugin: csv
    on_write_failure: discard
    options:
      schema:
        mode: fixed
        fields:
        - 'line: str'
        - 'result: str'
      path: outputs/hello_world.csv
      mode: write
"""

        state = _make_state()
        settings = _make_settings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "settings_load").passed is True
        assert _check(result, "plugin_instantiation").passed is False
        assert any("source 'text'" in e.message.lower() for e in result.errors)
        assert all("textsourceconfig" not in e.message.lower() for e in result.errors)
        assert all("pydantic.dev" not in e.message.lower() for e in result.errors)
        assert any("path" in e.message.lower() for e in result.errors)
        assert any("schema" in e.message.lower() for e in result.errors)
        assert any("column" in e.message.lower() for e in result.errors)


class TestValidatePipelineGraphFailure:
    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_graph_validation_error_attributed_to_node(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        mock_load.return_value = MagicMock()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.validate.side_effect = GraphValidationError("Route destination 'nonexistent' in gate_1 not found")

        state = _make_state()
        settings = _make_settings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "graph_structure").passed is False
        assert len(result.errors) >= 1

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_edge_compatibility_error(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        mock_load.return_value = MagicMock()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.validate.return_value = None  # structural check passes
        mock_graph.validate_edge_compatibility.side_effect = GraphValidationError("Schema mismatch on edge transform_1 -> sink_primary")

        state = _make_state()
        settings = _make_settings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "graph_structure").passed is True
        assert _check(result, "schema_compatibility").passed is False


class TestValidatePipelineNoBareCatch:
    """W18 fix: unexpected exceptions propagate — no bare except Exception."""

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    def test_unexpected_exception_propagates(
        self,
        mock_load: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        mock_load.side_effect = RuntimeError("Unexpected engine bug")

        state = _make_state()
        settings = _make_settings()
        # RuntimeError is NOT in the typed exception list — it must propagate
        with pytest.raises(RuntimeError, match="Unexpected engine bug"):
            validate_pipeline(state, settings, mock_yaml_gen)


class TestValidatePipelineInMemoryLoading:
    """Verify settings loading uses in-memory loader, matching execution service."""

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_settings_loaded_from_yaml_string(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Settings are loaded via load_settings_from_yaml_string, not file-based."""
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        mock_settings = MagicMock()
        mock_load.return_value = mock_settings
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph

        state = _make_state()
        settings = _make_settings()
        validate_pipeline(state, settings, mock_yaml_gen)

        # In-memory loader called with YAML string content
        mock_load.assert_called_once()
        loaded_yaml = mock_load.call_args.args[0]
        assert isinstance(loaded_yaml, str)
        assert "csv_source" in loaded_yaml


# ── Secret Ref Helpers ────────────────────────────────────────────────


class FakeSecretService:
    """Minimal WebSecretResolver stand-in for validation tests."""

    _VALID_FINGERPRINT = "a" * 64

    def __init__(self, available_refs: set[str], inventory_refs: set[str] | None = None) -> None:
        self._available = available_refs
        self._inventory = available_refs | (inventory_refs or set())

    def list_refs(self, user_id: str) -> list[SecretInventoryItem]:
        return [
            SecretInventoryItem(
                name=name,
                scope="user",
                available=name in self._available,
                reason=None if name in self._available else "value_decryption_failed",
            )
            for name in sorted(self._inventory)
        ]

    def has_ref(self, user_id: str, name: str) -> bool:
        return name in self._available

    def resolve(self, user_id: str, name: str) -> ResolvedSecret | None:
        if name in self._available:
            return ResolvedSecret(name=name, value="fake", scope="user", fingerprint=self._VALID_FINGERPRINT)
        return None


class TestCollectSecretRefs:
    """Unit tests for _collect_secret_refs helper."""

    def test_empty_dict(self) -> None:
        assert _collect_secret_refs({}) == []

    def test_single_secret_ref(self) -> None:
        assert _collect_secret_refs({"secret_ref": "API_KEY"}) == ["API_KEY"]

    def test_nested_secret_ref(self) -> None:
        data = {"source": {"options": {"api_key": {"secret_ref": "MY_KEY"}}}}
        assert _collect_secret_refs(data) == ["MY_KEY"]

    def test_multiple_refs(self) -> None:
        data = {
            "auth": {"secret_ref": "TOKEN"},
            "db": {"password": {"secret_ref": "DB_PASS"}},
        }
        refs = _collect_secret_refs(data)
        assert sorted(refs) == ["DB_PASS", "TOKEN"]

    def test_list_with_refs(self) -> None:
        data = [{"secret_ref": "A"}, {"secret_ref": "B"}]
        assert _collect_secret_refs(data) == ["A", "B"]

    def test_non_secret_dict(self) -> None:
        data = {"secret_ref": "KEY", "extra": "field"}  # len > 1, not a secret ref
        assert _collect_secret_refs(data) == []

    def test_mapping_proxy_type(self) -> None:
        """Frozen dataclass fields use MappingProxyType — must be walkable."""
        from types import MappingProxyType

        data = MappingProxyType({"api_key": MappingProxyType({"secret_ref": "KEY"})})
        assert _collect_secret_refs(data) == ["KEY"]


class TestValidatePipelineSecretRefs:
    """Secret ref validation check in validate_pipeline()."""

    def test_missing_refs_fail_validation(self) -> None:
        """Validation fails when secret refs can't be resolved."""
        state = _make_state(
            source_options={"api_key": {"secret_ref": "MISSING_KEY"}},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        secret_svc = FakeSecretService(available_refs=set())

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        secret_check = next(c for c in result.checks if c.name == "secret_refs")
        assert secret_check.passed is False
        assert "MISSING_KEY" in secret_check.detail
        assert any("MISSING_KEY" in e.message for e in result.errors)
        # Downstream checks should be skipped
        assert any("Skipped" in c.detail for c in result.checks if c.name == "settings_load")

    def test_all_refs_present_passes(self) -> None:
        """Validation passes when all secret refs are resolvable."""
        state = _make_state(
            source_options={"api_key": {"secret_ref": "MY_KEY"}},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        secret_svc = FakeSecretService(available_refs={"MY_KEY"})

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(
                state,
                settings,
                mock_yaml_gen,
                secret_service=secret_svc,
                user_id="user-1",
            )

        secret_check = next(c for c in result.checks if c.name == "secret_refs")
        assert secret_check.passed is True
        assert "1 secret reference(s) resolved" in secret_check.detail

    def test_no_secret_service_skips_check(self) -> None:
        """Without secret_service, the check is skipped (passed=True)."""
        state = _make_state(
            source_options={"api_key": {"secret_ref": "KEY"}},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        secret_check = next(c for c in result.checks if c.name == "secret_refs")
        assert secret_check.passed is True
        assert "skipped" in secret_check.detail.lower()

    def test_refs_in_node_options_detected(self) -> None:
        """Secret refs in node options are found and validated."""
        state = _make_state(
            source_options={},
            nodes=(_make_node(options={"token": {"secret_ref": "NODE_TOKEN"}}),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        secret_svc = FakeSecretService(available_refs=set())

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        assert any("NODE_TOKEN" in e.message for e in result.errors)

    def test_refs_in_output_options_detected(self) -> None:
        """Secret refs in output options are found and validated."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(options={"password": {"secret_ref": "DB_PASS"}}),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        secret_svc = FakeSecretService(available_refs=set())

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        assert any("DB_PASS" in e.message for e in result.errors)

    def test_multiple_missing_refs_listed(self) -> None:
        """All missing refs are collected and reported at once."""
        state = _make_state(
            source_options={
                "key1": {"secret_ref": "REF_A"},
                "key2": {"secret_ref": "REF_B"},
            },
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        secret_svc = FakeSecretService(available_refs={"REF_A"})  # REF_B missing

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        secret_check = next(c for c in result.checks if c.name == "secret_refs")
        assert "REF_B" in secret_check.detail
        assert "REF_A" not in secret_check.detail  # REF_A resolved fine

    def test_raw_env_marker_for_inventory_secret_uses_secret_ref_preflight(self) -> None:
        """Known web secret names must not bypass preflight via ${VAR} syntax."""
        state = _make_state(
            source_options={"api_key": "${OPENROUTER_API_KEY}"},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        secret_svc = FakeSecretService(
            available_refs=set(),
            inventory_refs={"OPENROUTER_API_KEY"},
        )

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        secret_check = next(c for c in result.checks if c.name == "secret_refs")
        assert secret_check.passed is False
        assert "OPENROUTER_API_KEY" in secret_check.detail
        assert any("OPENROUTER_API_KEY" in e.message for e in result.errors)


class TestValidatePipelineFabricatedCredentials:
    """Issue elspeth-72d1dccd44 / S1A regression: literal placeholder strings in
    credential-bearing fields must be rejected by the secret_refs check.

    The runtime treats fields like ``api_key`` / ``password`` / ``*_token`` as
    credential-bearing (see ``elspeth.core.secrets.is_secret_field``). The
    composer's ``secret_refs`` validator must enforce the same contract: such
    fields hold a wired ``{secret_ref: ...}`` marker, an inventory env-marker,
    or are absent — never a literal placeholder string.

    Source: notes/composer-llm-eval-2026-05-01.md S1A architectural finding #3.
    """

    _PLACEHOLDER = "WILL_BE_WIRED_FROM_OPENROUTER_API_KEY"

    def _assert_value_redacted(self, result, *, value: str) -> None:
        """Audit hygiene: the literal placeholder value MUST NOT be echoed
        into the validation response. A value that *looks* like a placeholder
        in this test may be a near-miss real secret in production; reflecting
        it back to the operator surface is data leakage."""
        for check in result.checks:
            assert value not in check.detail, f"placeholder value leaked into check '{check.name}' detail"
        for error in result.errors:
            assert value not in error.message, "placeholder value leaked into error.message"
            if error.suggestion is not None:
                assert value not in error.suggestion, "placeholder value leaked into error.suggestion"

    def test_literal_placeholder_in_transform_api_key_rejected(self) -> None:
        """The exact S1A shape: an LLM transform whose ``api_key`` is a literal
        placeholder string passes 0 wired secret_refs and validates as
        ``is_valid: true`` today — this test pins the post-fix behavior.
        """
        state = _make_state(
            source_options={},
            nodes=(
                _make_node(
                    options={
                        "provider": "openrouter",
                        "model": "openai/gpt-4.1-nano",
                        "api_key": self._PLACEHOLDER,
                    }
                ),
            ),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        secret_svc = FakeSecretService(available_refs=set())

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        secret_check = _check(result, "secret_refs")
        assert secret_check.passed is False
        assert "api_key" in secret_check.detail
        # Structured ValidationError points at the offending node + field.
        api_key_errors = [e for e in result.errors if "api_key" in e.message]
        assert api_key_errors, "expected at least one error naming the api_key field"
        assert any(e.component_id == "test_node" for e in api_key_errors)
        assert any(e.component_type == "transform" for e in api_key_errors)
        # Downstream checks skipped — no point loading settings on a known-bad shape.
        assert any("Skipped" in c.detail for c in result.checks if c.name == "settings_load")
        # Settings load must NOT be invoked: the secret-shape gate fires first.
        mock_yaml_gen.generate_yaml.assert_not_called()
        # Audit hygiene.
        self._assert_value_redacted(result, value=self._PLACEHOLDER)

    def test_literal_placeholder_in_source_api_key_rejected(self) -> None:
        """Same defect on a source plugin's credential field."""
        state = _make_state(
            source_options={"api_key": "PLACEHOLDER_TOKEN"},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        secret_svc = FakeSecretService(available_refs=set())

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        assert _check(result, "secret_refs").passed is False
        api_key_errors = [e for e in result.errors if "api_key" in e.message]
        assert any(e.component_id == "source" and e.component_type == "source" for e in api_key_errors)
        self._assert_value_redacted(result, value="PLACEHOLDER_TOKEN")

    def test_literal_placeholder_in_sink_password_rejected(self) -> None:
        """Suffix-matched secret field (``password``) on a sink."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(options={"password": "fake-pwd"}, name="db_sink"),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        secret_svc = FakeSecretService(available_refs=set())

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        assert _check(result, "secret_refs").passed is False
        sink_errors = [e for e in result.errors if "password" in e.message]
        assert any(e.component_id == "db_sink" and e.component_type == "sink" for e in sink_errors)
        self._assert_value_redacted(result, value="fake-pwd")

    def test_suffix_match_field_rejected(self) -> None:
        """Suffix heuristic: ``custom_token`` ends with ``_token`` — credential."""
        state = _make_state(
            source_options={"custom_token": "literal-token-string"},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        secret_svc = FakeSecretService(available_refs=set())

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        assert _check(result, "secret_refs").passed is False
        assert any("custom_token" in e.message for e in result.errors)
        self._assert_value_redacted(result, value="literal-token-string")

    def test_nested_credential_field_rejected(self) -> None:
        """Walk recurses into nested option dicts — a credential anywhere is a
        credential."""
        state = _make_state(
            source_options={"auth": {"api_key": "BOGUS"}},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        secret_svc = FakeSecretService(available_refs=set())

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        assert _check(result, "secret_refs").passed is False
        assert any("api_key" in e.message for e in result.errors)
        self._assert_value_redacted(result, value="BOGUS")

    def test_env_marker_outside_inventory_rejected(self) -> None:
        """A ``${NAME}`` shape where NAME is NOT in the secret inventory means
        the user typed a placeholder that won't resolve. Same UX defect class
        as a literal string — flag it.
        """
        state = _make_state(
            source_options={"api_key": "${SOME_UNREGISTERED_NAME}"},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        # No matching inventory entry — env-marker name unknown.
        secret_svc = FakeSecretService(available_refs=set(), inventory_refs=set())

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        assert _check(result, "secret_refs").passed is False
        assert any("api_key" in e.message for e in result.errors)
        # Audit hygiene: the env-marker text echoes a name the user typed; do
        # not reflect it back into the response either.
        self._assert_value_redacted(result, value="${SOME_UNREGISTERED_NAME}")

    def test_wired_secret_ref_passes(self) -> None:
        """Positive control: ``{"secret_ref": NAME}`` is the wired shape."""
        state = _make_state(
            source_options={"api_key": {"secret_ref": "REAL_KEY"}},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        secret_svc = FakeSecretService(available_refs={"REAL_KEY"})

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(
                state,
                settings,
                mock_yaml_gen,
                secret_service=secret_svc,
                user_id="user-1",
            )

        assert _check(result, "secret_refs").passed is True

    def test_inventory_env_marker_passes(self) -> None:
        """Positive control: ``${NAME}`` with NAME in inventory is wired."""
        state = _make_state(
            source_options={"api_key": "${OPENROUTER_API_KEY}"},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        secret_svc = FakeSecretService(
            available_refs={"OPENROUTER_API_KEY"},
            inventory_refs={"OPENROUTER_API_KEY"},
        )

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(
                state,
                settings,
                mock_yaml_gen,
                secret_service=secret_svc,
                user_id="user-1",
            )

        assert _check(result, "secret_refs").passed is True

    def test_empty_string_in_credential_field_passes(self) -> None:
        """Empty string is "not provided" — not a fabricated literal. Settings
        load decides whether the field is required."""
        state = _make_state(
            source_options={"api_key": ""},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        secret_svc = FakeSecretService(available_refs=set())

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(
                state,
                settings,
                mock_yaml_gen,
                secret_service=secret_svc,
                user_id="user-1",
            )

        assert _check(result, "secret_refs").passed is True

    def test_non_credential_string_unaffected(self) -> None:
        """Literal strings in non-credential fields pass — only credential
        fields are scope of the fabrication check."""
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/data.csv", "delimiter": ","},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        secret_svc = FakeSecretService(available_refs=set())

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(
                state,
                settings,
                mock_yaml_gen,
                secret_service=secret_svc,
                user_id="user-1",
            )

        assert _check(result, "secret_refs").passed is True


class TestReservedNameSecretRefPreflight:
    """Regression: pipeline validation must report reserved-name refs as
    missing, not crash on the raise that used to fall out of
    ServerSecretStore.has_secret.

    Uses a real WebSecretService (not the FakeSecretService stand-in)
    because the regression lives in the production composition path:
    UserSecretStore.has_secret returns False → OR falls through to
    ServerSecretStore.has_secret → which used to raise for ELSPETH_*
    names, propagating out of WebSecretService.has_ref and turning this
    validation pass into an uncaught 500.
    """

    def test_elspeth_prefixed_secret_ref_surfaces_as_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Reserved-name ref must appear in missing_refs, not raise.

        The pipeline references {"secret_ref": "ELSPETH_FINGERPRINT_KEY"}.
        Validation should complete with is_valid=False and a
        missing_refs entry for that name — the same outcome as any
        other unresolvable ref.
        """
        import sqlalchemy as sa

        from elspeth.web.secrets.server_store import ServerSecretStore
        from elspeth.web.secrets.service import ScopedSecretResolver, WebSecretService
        from elspeth.web.secrets.user_store import UserSecretStore
        from elspeth.web.sessions.models import metadata as session_metadata

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "validation-regression-fp-key")

        # Build a real session DB and real secret service wiring.
        db_path = tmp_path / "reserved_ref_validation.db"
        engine = sa.create_engine(f"sqlite:///{db_path}")
        session_metadata.create_all(engine)

        user_store = UserSecretStore(engine=engine, master_key="master-32-chars-minimum-length!!")
        # Empty allowlist — the reserved-name path is independent of allowlist,
        # but keeping it empty focuses the test on the reserved-name branch.
        server_store = ServerSecretStore(allowlist=())
        service = WebSecretService(user_store=user_store, server_store=server_store)
        resolver = ScopedSecretResolver(service, auth_provider_type="local")

        state = _make_state(
            source_options={"api_key": {"secret_ref": "ELSPETH_FINGERPRINT_KEY"}},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()

        # Must NOT raise — regression would have surfaced as SecretNotFoundError
        # propagating out of has_ref inside the missing_refs comprehension.
        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=resolver,
            user_id="user-1",
        )

        assert result.is_valid is False
        secret_check = next(c for c in result.checks if c.name == "secret_refs")
        assert secret_check.passed is False
        assert "ELSPETH_FINGERPRINT_KEY" in secret_check.detail
        assert any("ELSPETH_FINGERPRINT_KEY" in e.message for e in result.errors)


class TestSecretRefResolutionBeforeSettingsLoad:
    """Regression: secret_ref markers must be resolved before settings loading.

    Without resolution, raw {"secret_ref": "NAME"} markers reach plugin
    instantiation and fail with PluginConfigError because plugin configs
    expect string values, not dicts.
    """

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_secret_refs_resolved_before_settings_load(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load_string: MagicMock,
    ) -> None:
        """When secrets are present, validation resolves them in-memory."""
        state = _make_state(
            source_options={"api_key": {"secret_ref": "MY_KEY"}},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = (
            "source:\n  plugin: csv\n  on_success: transform_in\n"
            "  on_validation_failure: discard\n  options:\n"
            "    api_key:\n      secret_ref: MY_KEY\n"
        )
        secret_svc = FakeSecretService(available_refs={"MY_KEY"})

        mock_settings = MagicMock()
        mock_load_string.return_value = mock_settings
        mock_bundle = MagicMock()
        mock_instantiate.return_value = mock_bundle
        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        # In-memory loader was used
        mock_load_string.assert_called_once()
        # Parse the resolved YAML to verify secret was replaced (not string-scan)
        resolved_yaml = mock_load_string.call_args.args[0]
        parsed = yaml.safe_load(resolved_yaml)
        assert parsed["source"]["options"]["api_key"] == "fake"
        # Settings load check passed
        assert _check(result, "settings_load").passed is True

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_raw_env_marker_for_inventory_secret_resolves_before_settings_load(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load_string: MagicMock,
    ) -> None:
        """Exact ${NAME} markers for known secrets use resolver, not blind env expansion."""
        state = _make_state(
            source_options={"api_key": "${OPENROUTER_API_KEY}"},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = (
            "source:\n  plugin: csv\n  on_success: transform_in\n"
            "  on_validation_failure: discard\n  options:\n"
            "    api_key: ${OPENROUTER_API_KEY}\n"
        )
        secret_svc = FakeSecretService(available_refs={"OPENROUTER_API_KEY"})

        mock_settings = MagicMock()
        mock_load_string.return_value = mock_settings
        mock_bundle = MagicMock()
        mock_instantiate.return_value = mock_bundle
        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        mock_load_string.assert_called_once()
        resolved_yaml = mock_load_string.call_args.args[0]
        parsed = yaml.safe_load(resolved_yaml)
        assert parsed["source"]["options"]["api_key"] == "fake"
        assert _check(result, "settings_load").passed is True

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_no_secrets_also_uses_in_memory_loader(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load_string: MagicMock,
    ) -> None:
        """Without secret refs, validation still uses load_settings_from_yaml_string.

        Both paths (with and without secrets) use the same in-memory loader
        to ensure validation exercises the exact same code path as execution.
        """
        state = _make_state(source_options={"url": "https://example.com/data"})
        settings = _make_settings()
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = (
            "source:\n  plugin: csv\n  on_success: transform_in\n"
            "  on_validation_failure: discard\n  options:\n"
            "    url: https://example.com/data\n"
        )

        mock_settings = MagicMock()
        mock_load_string.return_value = mock_settings
        mock_bundle = MagicMock()
        mock_instantiate.return_value = mock_bundle
        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=FakeSecretService(available_refs=set()),
            user_id="user-1",
        )

        # In-memory loader was used — same path as execution service
        mock_load_string.assert_called_once()
        assert _check(result, "settings_load").passed is True


class TestInferComponentTypeFromPluginError:
    """Tests for _infer_component_type_from_plugin_error dispatch."""

    def test_plugin_config_error_with_source_type(self) -> None:
        """PluginConfigError with component_type='source' returns 'source'."""
        exc = PluginConfigError(
            "Invalid CSV config",
            cause="missing path",
            plugin_class="CsvSourceConfig",
            component_type="source",
        )
        assert _infer_component_type_from_plugin_error(exc) == "source"

    def test_plugin_config_error_with_sink_type(self) -> None:
        """PluginConfigError with component_type='sink' returns 'sink'."""
        exc = PluginConfigError(
            "Invalid JSON config",
            cause="bad format",
            plugin_class="JsonSinkConfig",
            component_type="sink",
        )
        assert _infer_component_type_from_plugin_error(exc) == "sink"

    def test_plugin_config_error_with_transform_type(self) -> None:
        """PluginConfigError with component_type='transform' returns 'transform'."""
        exc = PluginConfigError(
            "Invalid field mapper config",
            cause="missing mappings",
            plugin_class="FieldMapperConfig",
            component_type="transform",
        )
        assert _infer_component_type_from_plugin_error(exc) == "transform"

    def test_plugin_config_error_without_component_type(self) -> None:
        """PluginConfigError raised outside from_dict() has no component_type."""
        exc = PluginConfigError("Generic config error")
        assert _infer_component_type_from_plugin_error(exc) is None

    def test_plugin_not_found_error_returns_none(self) -> None:
        """PluginNotFoundError always returns None — no component_type attribute."""
        exc = PluginNotFoundError("No plugin named 'foobar'")
        assert _infer_component_type_from_plugin_error(exc) is None


class TestValidatePipelineRuntimePathResolution:
    @staticmethod
    def _loaded_yaml_from_settings_loader(mock_load: MagicMock) -> str:
        call = mock_load.call_args
        if call.args:
            return call.args[0]
        return call.kwargs["yaml_content"]

    def test_validate_pipeline_resolves_relative_source_and_sink_paths_before_settings_load(self) -> None:
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "blobs/session/input.csv"},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "outputs/out.csv"},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: csv
  on_success: main
  options:
    path: blobs/session/input.csv
    on_validation_failure: discard
sinks:
  main:
    plugin: csv
    options:
      path: outputs/out.csv
"""

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("stop after settings-load input capture")
            validate_pipeline(state, settings, mock_yaml_gen)

        loaded_yaml = self._loaded_yaml_from_settings_loader(mock_load)
        parsed = yaml.safe_load(loaded_yaml)
        assert parsed["source"]["options"]["path"] == "/tmp/test_data/blobs/session/input.csv"
        assert parsed["sinks"]["main"]["options"]["path"] == "/tmp/test_data/outputs/out.csv"

    def test_validate_pipeline_preserves_absolute_paths_before_settings_load(self) -> None:
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "/tmp/test_data/blobs/input.csv"},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/tmp/test_data/outputs/out.csv"},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: csv
  on_success: main
  options:
    path: /tmp/test_data/blobs/input.csv
    on_validation_failure: discard
sinks:
  main:
    plugin: csv
    options:
      path: /tmp/test_data/outputs/out.csv
"""

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("stop after settings-load input capture")
            validate_pipeline(state, settings, mock_yaml_gen)

        loaded_yaml = self._loaded_yaml_from_settings_loader(mock_load)
        parsed = yaml.safe_load(loaded_yaml)
        assert parsed["source"]["options"]["path"] == "/tmp/test_data/blobs/input.csv"
        assert parsed["sinks"]["main"]["options"]["path"] == "/tmp/test_data/outputs/out.csv"


class TestValidatePipelineRuntimeCheckBoundaries:
    def test_runtime_graph_validation_check_order_matches_named_constants(self) -> None:
        from elspeth.web.execution.preflight import (
            RUNTIME_CHECK_GRAPH_STRUCTURE,
            RUNTIME_CHECK_PLUGIN_INSTANTIATION,
            RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
            RUNTIME_GRAPH_VALIDATION_CHECKS,
        )

        assert RUNTIME_GRAPH_VALIDATION_CHECKS == (
            RUNTIME_CHECK_PLUGIN_INSTANTIATION,
            RUNTIME_CHECK_GRAPH_STRUCTURE,
            RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
        )

    def test_validate_pipeline_success_surfaces_declared_runtime_graph_checks(self) -> None:
        from elspeth.web.execution.preflight import RUNTIME_GRAPH_VALIDATION_CHECKS

        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/input.csv"},
            outputs=(_make_output({"path": "/tmp/test_data/outputs/out.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: csv
  on_success: primary
  options:
    path: /tmp/test_data/blobs/input.csv
    on_validation_failure: discard
sinks:
  primary:
    plugin: csv
    options:
      path: /tmp/test_data/outputs/out.csv
"""
        fake_graph = MagicMock()

        with (
            patch("elspeth.web.execution.validation.load_settings_from_yaml_string", return_value=MagicMock()),
            patch("elspeth.web.execution.validation.instantiate_runtime_plugins", return_value=MagicMock()) as mock_instantiate,
            patch("elspeth.web.execution.validation.build_runtime_graph", return_value=fake_graph),
            patch("elspeth.web.execution.validation.assemble_and_validate_pipeline_config", return_value=MagicMock()),
        ):
            result = validate_pipeline(state, settings, mock_yaml_gen)

        passed_names = {check.name for check in result.checks if check.passed}
        assert set(RUNTIME_GRAPH_VALIDATION_CHECKS).issubset(passed_names)
        assert mock_instantiate.call_args.kwargs == {"preflight_mode": True}
        fake_graph.validate.assert_called_once_with()
        fake_graph.validate_edge_compatibility.assert_called_once_with()

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_graph_structure_failure_uses_graph_check(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/input.csv"},
            outputs=(_make_output({"path": "/tmp/test_data/outputs/out.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: csv
  on_success: primary
  options:
    path: /tmp/test_data/blobs/input.csv
    on_validation_failure: discard
sinks:
  primary:
    plugin: csv
    options:
      path: /tmp/test_data/outputs/out.csv
"""
        fake_settings = MagicMock()
        fake_graph = MagicMock()
        fake_graph.validate.side_effect = GraphValidationError("bad graph")
        mock_load.return_value = fake_settings
        mock_instantiate.return_value = MagicMock()
        mock_build_graph.return_value = fake_graph

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "graph_structure").passed is False
        assert any(check.name == "schema_compatibility" and not check.passed for check in result.checks)
        fake_graph.validate_edge_compatibility.assert_not_called()

    @patch("elspeth.web.execution.validation.assemble_and_validate_pipeline_config")
    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_schema_failure_uses_schema_check(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_assemble: MagicMock,
    ) -> None:
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/input.csv"},
            outputs=(_make_output({"path": "/tmp/test_data/outputs/out.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: csv
  on_success: primary
  options:
    path: /tmp/test_data/blobs/input.csv
    on_validation_failure: discard
sinks:
  primary:
    plugin: csv
    options:
      path: /tmp/test_data/outputs/out.csv
"""
        fake_settings = MagicMock()
        fake_graph = MagicMock()
        fake_graph.validate_edge_compatibility.side_effect = GraphValidationError("schema mismatch")
        mock_load.return_value = fake_settings
        mock_instantiate.return_value = MagicMock()
        mock_build_graph.return_value = fake_graph
        mock_assemble.return_value = MagicMock()

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "graph_structure").passed is True
        assert _check(result, "route_target_resolution").passed is True
        assert _check(result, "schema_compatibility").passed is False
