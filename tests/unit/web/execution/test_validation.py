"""Tests for dry-run validation using real engine code paths.

Validation calls the actual engine functions: load_settings_from_yaml_string(),
instantiate_plugins_from_config(), ExecutionGraph.from_plugin_instances(),
graph.validate(). No parallel validation logic exists.

W18 fix: Only typed exceptions are caught — no bare except Exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import yaml
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.data import CompatibilityResult
from elspeth.contracts.secrets import ResolvedSecret, SecretInventoryItem
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import EdgeContractError, GraphValidationError, GraphValidationWarning
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
from elspeth.web.execution.protocol import YamlGenerator
from elspeth.web.execution.schemas import CHECK_OUTCOME_SKIPPED_AFTER_FAILURE, ValidationCheck
from elspeth.web.execution.validation import (
    _ALL_CHECKS,
    _CHECK_SETTINGS,
    _append_skipped_checks,
    _build_edge_contract_suggestion,
    _collect_secret_refs,
    _format_edge_contract_failure,
    _infer_component_type_from_plugin_error,
    _reframe_settings_missing_parts,
    validate_pipeline,
)
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY, PROMPT_TEMPLATE_PARTS_KEY, SOURCE_AUTHORING_KEY


def _make_source(options: dict[str, Any] | None = None) -> SourceSpec:
    """Build a SourceSpec with sensible defaults for validation tests."""
    return SourceSpec(
        plugin="csv",
        on_success="transform_in",
        options=options or {},
        on_validation_failure="discard",
    )


def _make_node(options: dict[str, Any] | None = None, plugin: str = "value_transform") -> NodeSpec:
    """Build a NodeSpec with sensible defaults for validation tests."""
    return NodeSpec(
        id="test_node",
        node_type="transform",
        plugin=plugin,
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
    plugin: str = "csv",
) -> OutputSpec:
    """Build an OutputSpec with sensible defaults for validation tests."""
    return OutputSpec(
        name=name,
        plugin=plugin,
        options=options or {},
        on_write_failure="discard",
    )


_MAKE_STATE_DEFAULT_SOURCE = object()


def _make_state(
    source_options: dict[str, Any] | None | object = _MAKE_STATE_DEFAULT_SOURCE,
    nodes: tuple[NodeSpec, ...] | None = None,
    outputs: tuple[OutputSpec, ...] | None = None,
) -> CompositionState:
    """Build a CompositionState with sensible defaults for validation tests.

    Default: a state with a placeholder source so callers bypass the
    ``validate_pipeline`` empty-pipeline short-circuit (which returns the
    structured ``empty_pipeline`` outcome before any of the steps these
    tests exercise via mocks). Pass ``source_options=None`` explicitly to
    test the empty-source path; pass a dict to customise source options.
    """
    if source_options is _MAKE_STATE_DEFAULT_SOURCE:
        source = _make_source({})
    elif source_options is None:
        source = None
    else:
        source = _make_source(cast(dict[str, Any], source_options))
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
        shareable_link_signing_key=b"\x00" * 32,
    )


def _check(result, name: str):
    """Look up a validation check by name, not position."""
    return next(c for c in result.checks if c.name == name)


@dataclass
class _FakeYamlGenerator:
    """Small YamlGenerator test double that records the state it rendered."""

    yaml_text: str = "source:\n  plugin: csv_source\n  options: {}"
    rendered_states: list[CompositionState] = field(default_factory=list)

    def generate_yaml(self, state: CompositionState) -> str:
        self.rendered_states.append(state)
        return self.yaml_text


@dataclass
class _FakeSettings:
    gates: tuple[Any, ...] = ()
    coalesce: tuple[Any, ...] = ()

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        assert mode == "json"
        return {}


@dataclass
class _FakeSourcePlugin:
    name: str = "source"
    _on_validation_failure: str = "discard"


@dataclass
class _FakeSinkPlugin:
    name: str = "csv"
    _on_write_failure: str = "discard"


@dataclass
class _FakeRuntimeBundle:
    """Runtime bundle shape consumed by validation after plugin instantiation."""

    sources: dict[str, Any] = field(default_factory=lambda: {"source": _FakeSourcePlugin()})
    source_settings_map: dict[str, Any] = field(default_factory=lambda: {"source": object()})
    transforms: tuple[Any, ...] = ()
    sinks: dict[str, Any] = field(default_factory=lambda: {"primary": _FakeSinkPlugin()})
    aggregations: dict[str, Any] = field(default_factory=dict)


def _fake_settings() -> _FakeSettings:
    return _FakeSettings()


def _fake_pipeline_config() -> SimpleNamespace:
    return SimpleNamespace()


def _runtime_graph_mock(
    *,
    validation_warnings: tuple[GraphValidationWarning, ...] = (),
    validate_side_effect: Exception | None = None,
    edge_validation_side_effect: Exception | None = None,
) -> MagicMock:
    graph = create_autospec(ExecutionGraph, instance=True)
    graph.validation_warnings = validation_warnings
    graph.validate.side_effect = validate_side_effect
    graph.validate_edge_compatibility.side_effect = edge_validation_side_effect
    graph.get_route_resolution_map.return_value = {}
    graph.get_transform_id_map.return_value = {}
    graph.get_config_gate_id_map.return_value = {}
    graph.get_aggregation_id_map.return_value = {}
    graph.get_coalesce_id_map.return_value = {}
    return graph


@dataclass(frozen=True)
class _FakeNodeInfo:
    config: dict[str, Any]


class _ForbiddenExactSourceLookup:
    def __init__(self, message: str) -> None:
        self._message = message
        self.called = False

    def __call__(self) -> None:
        self.called = True
        raise AssertionError(self._message)

    def assert_not_called(self) -> None:
        assert self.called is False


class _EdgeSuggestionGraph:
    """Graph fake for mapping runtime DAG IDs back to composer patch targets."""

    def __init__(
        self,
        *,
        sources: tuple[str, ...],
        node_configs: dict[str, dict[str, Any]],
        transform_id_map: dict[int, str],
        sink_id_map: dict[str, str],
        config_gate_id_map: dict[str, str] | None = None,
        aggregation_id_map: dict[str, str] | None = None,
        coalesce_id_map: dict[str, str] | None = None,
    ) -> None:
        self._sources = sources
        self._node_configs = node_configs
        self._transform_id_map = transform_id_map
        self._sink_id_map = sink_id_map
        self._config_gate_id_map = config_gate_id_map or {}
        self._aggregation_id_map = aggregation_id_map or {}
        self._coalesce_id_map = coalesce_id_map or {}
        self.get_source = _ForbiddenExactSourceLookup("exact-one source API must not be called")

    def get_sources(self) -> list[str]:
        return list(self._sources)

    def get_node_info(self, node_id: str) -> _FakeNodeInfo:
        return _FakeNodeInfo(config=self._node_configs[node_id])

    def get_transform_id_map(self) -> dict[int, str]:
        return self._transform_id_map

    def get_config_gate_id_map(self) -> dict[str, str]:
        return self._config_gate_id_map

    def get_aggregation_id_map(self) -> dict[str, str]:
        return self._aggregation_id_map

    def get_coalesce_id_map(self) -> dict[str, str]:
        return self._coalesce_id_map

    def get_sink_id_map(self) -> dict[str, str]:
        return self._sink_id_map


class TestValidatePipelineEmptyComposition:
    """Empty-composition short-circuit at the top of validate_pipeline.

    A CompositionState with no source, transforms, or outputs cannot be
    assembled into ElspethSettings — pydantic would otherwise emit a raw
    "2 validation errors for ElspethSettings: source/sinks Field required"
    stack trace leaking the internal model name to a user-facing UI. The
    short-circuit replaces that with a structured ``empty_pipeline`` error.

    Surface that exercises this path: a guided session immediately after
    ``exit_to_freeform`` (the wire response sets composition_state to a
    version-bumped state with source=null nodes=[] outputs=[]).
    """

    def test_empty_pipeline_returns_structured_error(self) -> None:
        state = _make_state(source_options=None)
        settings = _make_settings()

        result = validate_pipeline(state, settings, _FakeYamlGenerator())

        assert result.is_valid is False
        assert len(result.errors) == 1
        err = result.errors[0]
        assert err.error_code == "empty_pipeline"
        # The user-facing message MUST NOT contain pydantic internals.
        assert "ElspethSettings" not in err.message
        assert "Field required" not in err.message
        assert err.suggestion is not None

    def test_empty_pipeline_skips_pydantic_invocation(self) -> None:
        """The short-circuit returns before any of the engine code paths
        the mocks would intercept — confirms we are NOT relying on
        ``load_settings_from_yaml_string`` raising PydanticValidationError
        to detect this case."""
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            state = _make_state(source_options=None)
            settings = _make_settings()
            result = validate_pipeline(state, settings, _FakeYamlGenerator())

        assert result.is_valid is False
        assert result.errors[0].error_code == "empty_pipeline"
        mock_load.assert_not_called()

    def test_source_alone_bypasses_empty_check(self) -> None:
        """A source without outputs is not 'empty' — the user has started
        composing. The empty_pipeline short-circuit must NOT fire; normal
        validation proceeds (here to a mocked plugin failure). The missing-sink
        reframe happens later, at the settings-load catch — see
        TestValidatePipelineMissingPartReframe."""
        state = _make_state()  # default: source set, nodes/outputs empty
        settings = _make_settings()

        with (
            patch("elspeth.web.execution.validation.load_settings_from_yaml_string"),
            patch("elspeth.web.execution.validation.instantiate_runtime_plugins") as mock_inst,
        ):
            mock_inst.side_effect = PluginNotFoundError("placeholder")
            mock_yaml = MagicMock(spec=YamlGenerator)
            mock_yaml.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}"
            result = validate_pipeline(state, settings, mock_yaml)

        assert result.is_valid is False
        assert all(err.error_code != "empty_pipeline" for err in result.errors)


def _elspeth_settings_missing_parts_error(present: set[str]) -> PydanticValidationError:
    """Build a real PydanticValidationError shaped like ElspethSettings
    rejecting missing required top-level parts. ``present`` names the parts to
    supply (each valid), so the omitted parts report ``type == "missing"`` with
    ``loc[0]`` in {"sources", "sinks"} — the exact structure
    ``_reframe_settings_missing_parts`` keys on. A throwaway model with the same
    field names keeps the fixture independent of ElspethSettings' nested plugin
    validation."""

    class _RequiredParts(BaseModel):
        sources: dict[str, Any]
        sinks: dict[str, Any]

    data = {name: {"placeholder": {}} for name in present}
    try:
        _RequiredParts.model_validate(data)
    except PydanticValidationError as exc:
        return exc
    raise AssertionError("expected a validation error for the missing parts")


class TestReframeSettingsMissingParts:
    """`_reframe_settings_missing_parts` maps ElspethSettings 'Field required'
    failures on sources/sinks into novice-register findings — the reframe logic
    behind elspeth-901a404926, over structured ``exc.errors()`` (never str(exc))."""

    def _assert_no_pydantic_leak(self, message: str) -> None:
        assert "ElspethSettings" not in message
        assert "Field required" not in message
        assert "pydantic" not in message.lower()

    def test_missing_sink_only(self) -> None:
        exc = _elspeth_settings_missing_parts_error(present={"sources"})
        reframed = _reframe_settings_missing_parts(exc)
        assert [e.error_code for e in reframed] == ["missing_sink"]
        assert "output step" in reframed[0].message
        assert reframed[0].suggestion is not None
        self._assert_no_pydantic_leak(reframed[0].message)

    def test_missing_source_only(self) -> None:
        exc = _elspeth_settings_missing_parts_error(present={"sinks"})
        reframed = _reframe_settings_missing_parts(exc)
        assert [e.error_code for e in reframed] == ["missing_source"]
        assert "data source" in reframed[0].message
        self._assert_no_pydantic_leak(reframed[0].message)

    def test_both_missing_are_source_before_sink(self) -> None:
        """A lone-transform composition omits both parts; findings must read in
        a stable source-before-sink order regardless of pydantic's ordering."""
        exc = _elspeth_settings_missing_parts_error(present=set())
        reframed = _reframe_settings_missing_parts(exc)
        assert [e.error_code for e in reframed] == ["missing_source", "missing_sink"]

    def test_non_missing_error_is_not_reframed(self) -> None:
        """A non-'missing' settings failure (e.g. a type error) returns [] so
        the caller falls back to str(exc)."""

        class _Model(BaseModel):
            sources: dict[str, Any]
            sinks: dict[str, Any]

        try:
            _Model.model_validate({"sources": "not-a-dict", "sinks": {"x": {}}})
        except PydanticValidationError as exc:
            assert _reframe_settings_missing_parts(exc) == []
        else:
            raise AssertionError("expected a validation error")

    def test_parity_with_real_elspeth_settings(self) -> None:
        """Guard against ElspethSettings field renames: a real missing-parts
        failure of the actual settings model must reframe to both findings —
        confirms the throwaway fixture's ``loc`` structure matches reality."""
        from elspeth.core.config import ElspethSettings

        try:
            ElspethSettings.model_validate({})
        except PydanticValidationError as exc:
            reframed = _reframe_settings_missing_parts(exc)
            assert [e.error_code for e in reframed] == ["missing_source", "missing_sink"]
        else:
            raise AssertionError("expected ElspethSettings to reject an empty config")


class TestValidatePipelineMissingPartReframe:
    """The settings-load catch in validate_pipeline routes a missing
    source/sink pydantic failure through the reframe, keeping the raw dump in
    the settings ValidationCheck detail for the engineer read (elspeth-901a404926)."""

    def test_missing_sink_surfaces_reframed_finding_not_raw_dump(self) -> None:
        state = _make_state()  # source present, no output
        mock_yaml = MagicMock(spec=YamlGenerator)
        mock_yaml.generate_yaml.return_value = "sources:\n  source:\n    plugin: csv\n    options: {}"
        exc = _elspeth_settings_missing_parts_error(present={"sources"})

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string", side_effect=exc):
            result = validate_pipeline(state, _make_settings(), mock_yaml)

        assert result.is_valid is False
        assert [e.error_code for e in result.errors] == ["missing_sink"]
        assert "ElspethSettings" not in result.errors[0].message
        assert result.readiness.blockers[0].code == "incomplete_pipeline"
        # The raw dump is retained for the record in the settings check detail.
        settings_check = _check(result, _CHECK_SETTINGS)
        assert "Field required" in settings_check.detail

    def test_other_settings_error_still_surfaces_raw_message(self) -> None:
        """A non-missing-part settings failure keeps its str(exc) message and
        the settings_load readiness code (unchanged behaviour)."""
        state = _make_state()
        mock_yaml = MagicMock(spec=YamlGenerator)
        mock_yaml.generate_yaml.return_value = "sources:\n  source:\n    plugin: csv\n    options: {}"

        with patch(
            "elspeth.web.execution.validation.load_settings_from_yaml_string",
            side_effect=ValueError("some other settings problem"),
        ):
            result = validate_pipeline(state, _make_settings(), mock_yaml)

        assert result.is_valid is False
        assert result.errors[0].error_code is None
        assert "some other settings problem" in result.errors[0].message
        assert result.readiness.blockers[0].code == "settings_load"


class TestValidatePipelinePathAllowlist:
    """C3/S2: Source path allowlist check — defense-in-depth."""

    def test_path_within_blobs_passes(self) -> None:
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/data.csv"},
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False
        assert _check(result, "path_allowlist").passed is False
        assert any("Path traversal" in e.message for e in result.errors)

    def test_second_named_source_path_outside_blobs_blocked(self) -> None:
        state = CompositionState(
            source=None,
            sources={
                "orders": _make_source({"path": "/tmp/test_data/blobs/orders.csv"}),
                "refunds": _make_source({"path": "/etc/passwd"}),
            },
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = _FakeYamlGenerator()

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "path_allowlist").passed is False
        assert any(error.component_id == "source:refunds" and "refunds" in error.message for error in result.errors)

    def test_path_traversal_via_dotdot_blocked(self) -> None:
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/../../secret.csv"},
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False

    def test_no_path_option_records_skipped_check(self) -> None:
        """B11 fix: path allowlist check is always recorded, even when skipped."""
        state = _make_state(source_options={})
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        # B11: check IS recorded with passed=True and "skipped" detail
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True
        assert "skipped" in path_check.detail.lower()


class TestSkippedCheckDeduplication:
    """``_append_skipped_checks`` must not emit a second, contradictory
    "skipped" record for a check that was already recorded earlier in the
    same ``validate_pipeline`` pass.

    Because checks are *emitted* during ``validate_pipeline`` in a different
    order than the canonical ``_ALL_CHECKS`` ordering, a check that already has
    a record can fall inside the "skip everything after me" range of a later
    gate failure.  Without the ``already_emitted`` guard it would then gain a
    second, contradictory ``passed=False`` skipped record; the guard exists to
    prevent exactly that.
    """

    def test_does_not_duplicate_an_already_emitted_check(self) -> None:
        from_check = _ALL_CHECKS[0]
        already_emitted_name = _ALL_CHECKS[-1]  # positioned strictly after from_check
        assert already_emitted_name != from_check

        original = ValidationCheck(
            name=already_emitted_name,
            passed=True,
            detail="recorded before the skip-after sweep",
            affected_nodes=(),
            outcome_code=None,
        )
        checks = [original]

        _append_skipped_checks(checks, from_check)

        names = [c.name for c in checks]
        # The already-emitted record survives exactly once — not shadowed by a
        # contradictory "skipped" entry for the same check name.
        assert names.count(already_emitted_name) == 1
        survivor = next(c for c in checks if c.name == already_emitted_name)
        assert survivor is original
        assert survivor.passed is True

    def test_still_records_skips_for_not_yet_emitted_checks(self) -> None:
        """The dedup guard must not suppress genuine skips for checks that
        were not already emitted — otherwise the audit trail loses coverage."""
        from_check = _ALL_CHECKS[0]
        not_emitted = _ALL_CHECKS[1]  # after from_check, never pre-seeded

        checks: list[ValidationCheck] = []
        _append_skipped_checks(checks, from_check)

        skipped = next(c for c in checks if c.name == not_emitted)
        assert skipped.passed is False
        assert "skipped" in skipped.detail.lower()


class TestValidatePipelineWebScrapeNetworkPolicy:
    """Web-authored web_scrape configs must not widen SSRF allowlists."""

    @staticmethod
    def _web_scrape_options(allowed_hosts: object = "public_only") -> dict[str, object]:
        return {
            "schema": {"mode": "fixed", "fields": ["url: str"]},
            "url_field": "url",
            "content_field": "content",
            "fingerprint_field": "fingerprint",
            "http": {
                "abuse_contact": "ops@somecompany.gov.au",
                "scraping_reason": "User-authorised public web fetch",
                "allowed_hosts": allowed_hosts,
            },
        }

    def test_web_scrape_allow_private_rejected_before_yaml_generation(self) -> None:
        state = _make_state(
            nodes=(
                _make_node(
                    plugin="web_scrape",
                    options=self._web_scrape_options("allow_private"),
                ),
            ),
            outputs=(_make_output(name="results"),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}\n"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "web_scrape_network_policy").passed is False
        assert result.errors[0].component_id == "test_node"
        assert result.errors[0].error_code == "web_scrape_private_network_not_allowed"
        assert "allow_private" in result.errors[0].message
        mock_yaml_gen.generate_yaml.assert_not_called()

    def test_web_scrape_explicit_cidr_allowlist_rejected_before_yaml_generation(self) -> None:
        state = _make_state(
            nodes=(
                _make_node(
                    plugin="web_scrape",
                    options=self._web_scrape_options(["10.0.0.0/8"]),
                ),
            ),
            outputs=(_make_output(name="results"),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}\n"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "web_scrape_network_policy").passed is False
        assert result.errors[0].error_code == "web_scrape_private_network_not_allowed"
        assert "CIDR" in result.errors[0].message
        mock_yaml_gen.generate_yaml.assert_not_called()

    def test_web_scrape_failure_skips_later_managed_identity_and_llm_retry_checks(self) -> None:
        # Regression: managed_identity_policy (declared #8) and llm_retry_budget_policy
        # (declared #9) must NOT be reported as passed when web_scrape_network_policy
        # (declared #2) fails earlier in the contract order. They previously executed
        # before web_scrape in code, so their pass records were already emitted and the
        # skipped-after-failure record was suppressed — the trail then showed a
        # later-declared gate passing under an earlier-declared failure.
        state = _make_state(
            nodes=(
                _make_node(
                    plugin="web_scrape",
                    options=self._web_scrape_options(["10.0.0.0/8"]),
                ),
            ),
            outputs=(_make_output(name="results"),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}\n"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "web_scrape_network_policy").passed is False
        for later_check in ("managed_identity_policy", "llm_retry_budget_policy", "llm_base_url_policy"):
            check = _check(result, later_check)
            assert check.passed is False, f"{later_check} must not pass after an earlier gate failed"
            assert check.outcome_code == CHECK_OUTCOME_SKIPPED_AFTER_FAILURE

    def test_web_scrape_public_only_allowed_to_reach_yaml_generation(self) -> None:
        state = _make_state(
            nodes=(
                _make_node(
                    plugin="web_scrape",
                    options=self._web_scrape_options("public_only"),
                ),
            ),
            outputs=(_make_output(name="results"),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}\n"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert _check(result, "web_scrape_network_policy").passed is True
        assert all(error.error_code != "web_scrape_private_network_not_allowed" for error in result.errors)
        mock_yaml_gen.generate_yaml.assert_called_once_with(state)


class TestWebLlmBaseUrlPolicyHelper:
    """Unit coverage for the web-authored OpenRouter base_url policy helper."""

    def test_non_llm_plugin_is_ignored(self) -> None:
        from elspeth.web.provider_config_policy import web_llm_base_url_policy_error

        assert web_llm_base_url_policy_error("web_scrape", {"base_url": "http://127.0.0.1/v1"}) is None

    def test_unset_base_url_is_allowed(self) -> None:
        from elspeth.web.provider_config_policy import web_llm_base_url_policy_error

        assert web_llm_base_url_policy_error("llm", {"model": "openai/gpt-4o"}) is None

    def test_canonical_base_url_is_allowed_with_or_without_trailing_slash(self) -> None:
        from elspeth.web.provider_config_policy import OPENROUTER_BASE_URL, web_llm_base_url_policy_error

        assert web_llm_base_url_policy_error("llm", {"base_url": OPENROUTER_BASE_URL}) is None
        assert web_llm_base_url_policy_error("llm", {"base_url": OPENROUTER_BASE_URL + "/"}) is None

    def test_http_loopback_base_url_is_blocked(self) -> None:
        # The reviewer's finding: loopback HTTP turns a web-authored config into
        # a credential-egress / SSRF path.
        from elspeth.web.provider_config_policy import web_llm_base_url_policy_error

        assert web_llm_base_url_policy_error("llm", {"base_url": "http://127.0.0.1:8199/v1"}) is not None
        assert web_llm_base_url_policy_error("llm", {"base_url": "http://localhost/v1"}) is not None

    def test_arbitrary_https_host_is_blocked(self) -> None:
        # Superset of the loopback finding: an HTTPS host the author controls
        # would still exfiltrate the server-held bearer credential.
        from elspeth.web.provider_config_policy import web_llm_base_url_policy_error

        assert web_llm_base_url_policy_error("llm", {"base_url": "https://evil.example.com/v1"}) is not None
        assert web_llm_base_url_policy_error("llm", {"base_url": "https://10.0.0.5/v1"}) is not None


class TestValidatePipelineLlmBaseUrlPolicy:
    """Web-authored OpenRouter LLM nodes may not override base_url (credential egress)."""

    @staticmethod
    def _llm_options(base_url: object | None = None) -> dict[str, object]:
        # No ``model`` — that would stage an llm_model_choice interpretation
        # review (gate #6) which fires before the base_url gate (#10). The
        # base_url gate inspects only base_url, so a model is irrelevant here.
        options: dict[str, object] = {}
        if base_url is not None:
            options["base_url"] = base_url
        return options

    def test_loopback_base_url_rejected_before_yaml_generation(self) -> None:
        state = _make_state(
            nodes=(_make_node(plugin="llm", options=self._llm_options("http://127.0.0.1:8199/v1")),),
            outputs=(_make_output(name="results"),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}\n"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "llm_base_url_policy").passed is False
        assert result.errors[0].component_id == "test_node"
        assert result.errors[0].error_code == "llm_base_url_not_allowed"

    def test_arbitrary_https_base_url_rejected_before_yaml_generation(self) -> None:
        state = _make_state(
            nodes=(_make_node(plugin="llm", options=self._llm_options("https://evil.example.com/v1")),),
            outputs=(_make_output(name="results"),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}\n"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "llm_base_url_policy").passed is False
        assert result.errors[0].error_code == "llm_base_url_not_allowed"

    def test_canonical_base_url_passes_gate_and_reaches_yaml_generation(self) -> None:
        from elspeth.web.provider_config_policy import OPENROUTER_BASE_URL

        state = _make_state(
            nodes=(_make_node(plugin="llm", options=self._llm_options(OPENROUTER_BASE_URL)),),
            outputs=(_make_output(name="results"),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}\n"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert _check(result, "llm_base_url_policy").passed is True
        assert all(error.error_code != "llm_base_url_not_allowed" for error in result.errors)
        mock_yaml_gen.generate_yaml.assert_called_once_with(state)

    def test_unset_base_url_passes_gate(self) -> None:
        state = _make_state(
            nodes=(_make_node(plugin="llm", options=self._llm_options(None)),),
            outputs=(_make_output(name="results"),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}\n"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert _check(result, "llm_base_url_policy").passed is True
        mock_yaml_gen.generate_yaml.assert_called_once_with(state)


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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}\n"

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "batch_transform_options").passed is False
        messages = "\n".join(error.message for error in result.errors)
        assert "required_input_fields" in messages
        assert "batch-aware" in messages
        mock_yaml_gen.generate_yaml.assert_not_called()

    def test_batch_replicate_plain_transform_returns_structured_validation_error(self) -> None:
        source = SourceSpec(
            plugin="csv",
            on_success="replicate_in",
            options={"schema": {"mode": "fixed", "fields": ["region: str"]}},
            on_validation_failure="discard",
        )
        replicate = NodeSpec(
            id="replicate",
            node_type="transform",
            plugin="batch_replicate",
            input="replicate_in",
            on_success="primary",
            on_error="discard",
            options={
                "schema": {"mode": "observed"},
                "replications": [
                    {"source_field": "region", "output_field": "region_copy"},
                ],
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
            nodes=(replicate,),
            edges=(),
            outputs=(_make_output({"schema": {"mode": "observed"}}),),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}\n"

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "batch_transform_options").passed is False
        messages = "\n".join(error.message for error in result.errors)
        assert "batch_replicate" in messages
        assert "aggregation" in messages
        assert "output_mode: transform" in messages
        mock_yaml_gen.generate_yaml.assert_not_called()

    def test_batch_distribution_profile_rejects_string_value_field_before_runtime(self) -> None:
        source = SourceSpec(
            plugin="csv",
            on_success="profile_in",
            options={
                "schema": {
                    "mode": "fixed",
                    "fields": [
                        "community: str",
                        "wave: str",
                        "financial_barrier: str",
                    ],
                }
            },
            on_validation_failure="discard",
        )
        profile = NodeSpec(
            id="profile_barriers",
            node_type="aggregation",
            plugin="batch_distribution_profile",
            input="profile_in",
            on_success="primary",
            on_error="discard",
            options={
                "schema": {"mode": "observed"},
                "value_field": "financial_barrier",
                "group_by": "community",
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
            nodes=(profile,),
            edges=(),
            outputs=(_make_output({"schema": {"mode": "observed"}}),),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}\n"

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "batch_transform_options").passed is False
        messages = "\n".join(error.message for error in result.errors)
        assert "batch_distribution_profile.value_field" in messages
        assert "numeric-only" in messages
        assert "financial_barrier" in messages
        assert "upstream declares type str" in messages
        assert "batch_top_k" in messages
        mock_yaml_gen.generate_yaml.assert_not_called()


class TestValidatePipelinePendingInterpretationPlaceholders:
    """Runtime preflight must distinguish composer authoring from execution."""

    def test_pending_structured_interpretation_returns_typed_readiness(self) -> None:
        state = _make_state(
            nodes=(
                _make_node(
                    plugin="llm",
                    options={
                        "prompt_template": "Rate pending interpretation: {{ row.text }}",
                        PROMPT_TEMPLATE_PARTS_KEY: [
                            {"kind": "text", "text": "Rate "},
                            {"kind": "interpretation_ref", "requirement_id": "coolness"},
                            {"kind": "text", "text": ": {{ row.text }}"},
                        ],
                        INTERPRETATION_REQUIREMENTS_KEY: [
                            {
                                "id": "coolness",
                                "kind": "vague_term",
                                "user_term": "coolness",
                                "status": "pending",
                                "draft": "well-designed and useful",
                                "event_id": "event-1",
                                "accepted_value": None,
                                "accepted_artifact_hash": None,
                                "resolved_prompt_template_hash": None,
                            }
                        ],
                    },
                ),
            )
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert result.errors[0].error_code == "interpretation_review_pending"
        assert result.errors[1].error_code == "interpretation_review_pending"
        assert "Invalid Jinja2 template" not in result.errors[0].message
        assert result.readiness.authoring_valid is True
        assert result.readiness.execution_ready is False
        assert result.readiness.completion_ready is True
        assert result.readiness.blockers[0].code == "interpretation_review_pending"
        assert result.readiness.blockers[0].component_id is None
        assert result.readiness.blockers[0].component_type is None
        assert "llm_prompt_template:test_node" in result.readiness.blockers[0].detail
        assert "vague_term" in result.errors[0].message
        assert "transform 'test_node'" in result.errors[0].message
        assert "llm_prompt_template" in result.errors[1].message
        assert "llm_prompt_template:test_node" in result.errors[1].message
        mock_yaml_gen.generate_yaml.assert_not_called()

    def test_pending_invented_source_review_returns_source_readiness(self) -> None:
        state = _make_state(
            source_options={
                SOURCE_AUTHORING_KEY: {
                    "modality": "llm_generated",
                    "content_hash": "a" * 64,
                    "review_event_id": None,
                    "resolved_kind": None,
                },
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "source-urls",
                        "kind": "invented_source",
                        "user_term": "inline_source_url_list",
                        "status": "pending",
                        "draft": "https://example.gov.au",
                        "event_id": None,
                        "accepted_value": None,
                        "accepted_artifact_hash": None,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            }
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert result.errors[0].error_code == "interpretation_review_pending"
        assert result.errors[0].component_id == "source"
        assert result.errors[0].component_type == "source"
        assert "invented_source" in result.errors[0].message
        assert "source 'source'" in result.errors[0].message
        assert result.readiness.blockers[0].component_id == "source"
        assert result.readiness.blockers[0].component_type == "source"
        mock_yaml_gen.generate_yaml.assert_not_called()

    def test_resolved_invented_source_drift_returns_source_readiness(self) -> None:
        """elspeth-5a94855935: a RESOLVED invented_source whose
        accepted_artifact_hash drifted from the current source content_hash is a
        readiness blocker, NOT an uncaught ValueError that escapes
        validate_pipeline as an HTTP 500."""
        state = _make_state(
            source_options={
                SOURCE_AUTHORING_KEY: {
                    "modality": "llm_generated",
                    "content_hash": "a" * 64,
                    "review_event_id": "event-1",
                    "resolved_kind": "invented_source",
                },
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "source-urls",
                        "kind": "invented_source",
                        "user_term": "inline_source_url_list",
                        "status": "resolved",
                        "draft": "https://example.gov.au",
                        "event_id": "event-1",
                        "accepted_value": "accepted source artifact",
                        "accepted_artifact_hash": "b" * 64,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            }
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert result.errors[0].error_code == "interpretation_review_pending"
        assert result.errors[0].component_id == "source"
        assert result.errors[0].component_type == "source"
        assert "invented_source" in result.errors[0].message
        assert result.readiness.blockers[0].code == "interpretation_review_pending"
        assert result.readiness.blockers[0].component_id == "source"
        assert result.readiness.blockers[0].component_type == "source"
        mock_yaml_gen.generate_yaml.assert_not_called()

    def test_legacy_pending_interpretation_placeholder_returns_typed_readiness_by_default(self) -> None:
        state = _make_state(
            nodes=(
                _make_node(
                    plugin="llm",
                    options={"prompt_template": "Rate how {{ interpretation: cool }} this row is."},
                ),
            )
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec_set=YamlGenerator)

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert result.errors[0].error_code == "interpretation_review_pending"
        assert result.readiness.authoring_valid is True
        assert result.readiness.execution_ready is False
        assert result.readiness.completion_ready is True
        mock_yaml_gen.generate_yaml.assert_not_called()

    def test_pending_interpretation_placeholder_is_masked_for_authoring_preflight(self) -> None:
        state = _make_state(
            nodes=(
                _make_node(
                    plugin="llm",
                    options={"prompt_template": "Rate how {{ interpretation: cool }} this row is."},
                ),
            )
        )
        settings = _make_settings()
        captured_states: list[CompositionState] = []
        mock_yaml_gen = MagicMock(spec=YamlGenerator)

        def _generate_yaml(candidate: CompositionState) -> str:
            captured_states.append(candidate)
            return "source:\n  plugin: csv_source\n  options: {}"

        mock_yaml_gen.generate_yaml.side_effect = _generate_yaml
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(
                state,
                settings,
                mock_yaml_gen,
                allow_pending_interpretation_placeholders=True,
            )

        assert result.is_valid is False
        assert captured_states[0].version == state.version
        assert captured_states[0].nodes[0].options["prompt_template"] == "Rate how pending interpretation this row is."
        assert state.nodes[0].options["prompt_template"] == "Rate how {{ interpretation: cool }} this row is."

    def test_resolved_prompt_hash_keeps_placeholder_strict_even_in_authoring_preflight(self) -> None:
        state = _make_state(
            nodes=(
                _make_node(
                    plugin="llm",
                    options={
                        "prompt_template": "Rate how {{ interpretation: cool }} this row is.",
                        "resolved_prompt_template_hash": "sha256-rfc8785-v1:abc123",
                    },
                ),
            )
        )
        settings = _make_settings()
        captured_states: list[CompositionState] = []
        mock_yaml_gen = MagicMock(spec=YamlGenerator)

        def _generate_yaml(candidate: CompositionState) -> str:
            captured_states.append(candidate)
            return "source:\n  plugin: csv_source\n  options: {}"

        mock_yaml_gen.generate_yaml.side_effect = _generate_yaml
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("settings stop")
            result = validate_pipeline(
                state,
                settings,
                mock_yaml_gen,
                allow_pending_interpretation_placeholders=True,
            )

        assert result.is_valid is False
        assert captured_states[0].nodes[0].options["prompt_template"] == "Rate how {{ interpretation: cool }} this row is."


class TestValidatePipelineSinkPathAllowlist:
    """Sink path allowlist — prevents arbitrary file writes via sink options."""

    def test_sink_path_outside_outputs_blocked(self) -> None:
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="evil_sink", options={"path": "/etc/cron.d/backdoor.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False

    def test_sink_path_under_outputs_passes(self) -> None:
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="primary", options={"path": "/tmp/test_data/outputs/result.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True
        assert "All paths within allowed directories" in path_check.detail

    def test_sink_path_under_own_session_blobs_passes(self) -> None:
        """elspeth-bdc17cfdb1: a sink may target the caller's own blob subtree."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="blob_out", options={"path": "/tmp/test_data/blobs/sess-1/out.json"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen, session_id="sess-1")
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True

    def test_sink_path_under_other_session_blobs_blocked(self) -> None:
        """elspeth-bdc17cfdb1: a sink targeting another session's blob subtree
        must fail the path allowlist — this is the cross-session write defect."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="blob_out", options={"path": "/tmp/test_data/blobs/sess-2/out.json"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        result = validate_pipeline(state, settings, mock_yaml_gen, session_id="sess-1")
        assert result.is_valid is False
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is False

    def test_sink_path_under_blobs_blocked_without_session(self) -> None:
        """No session identity fails closed: blob-targeted sinks are rejected,
        never allowed against the blobs root."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="blob_out", options={"path": "/tmp/test_data/blobs/sess-1/out.json"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is False

    def test_second_named_source_path_outside_allowed_dirs_is_blocked(self) -> None:
        """Every named source path must pass the validation allowlist."""
        state = CompositionState(
            source=None,
            sources={
                "orders": _make_source({"path": "/tmp/test_data/blobs/orders.csv"}),
                "refunds": _make_source({"path": "/etc/passwd"}),
            },
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = _FakeYamlGenerator()

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is False
        assert path_check.affected_nodes == ("source:refunds",)
        assert any("source 'refunds'" in error.message for error in result.errors)

    def test_sink_without_path_passes(self) -> None:
        """Sinks without path/file options (e.g. database) skip the check."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="db_sink", options={"connection_string": "sqlite:///out.db"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True


class TestValidatePipelineTransformProviderConfigPathAllowlist:
    """Nested transform provider_config path allowlist — RAG retrieval
    transforms carry a local Chroma persist_directory under
    options.provider_config; it must be confined like a sink path."""

    def test_transform_provider_persist_directory_outside_blocked(self) -> None:
        node = _make_node(
            plugin="rag_retrieval",
            options={"provider": "chroma", "provider_config": {"persist_directory": "/etc/cron.d/backdoor"}},
        )
        state = _make_state(source_options={}, nodes=(node,))
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False
        assert any("Path traversal" in e.message for e in result.errors)
        assert any(e.component_type == "transform" for e in result.errors)
        assert any("persist_directory" in e.message for e in result.errors)

    def test_transform_provider_persist_directory_traversal_blocked(self) -> None:
        node = _make_node(
            plugin="rag_retrieval",
            options={
                "provider": "chroma",
                "provider_config": {"persist_directory": "/tmp/test_data/outputs/../../etc/secret"},
            },
        )
        state = _make_state(source_options={}, nodes=(node,))
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False

    def test_transform_provider_persist_directory_under_outputs_passes(self) -> None:
        node = _make_node(
            plugin="rag_retrieval",
            options={"provider": "chroma", "provider_config": {"persist_directory": "/tmp/test_data/outputs/chroma"}},
        )
        state = _make_state(source_options={}, nodes=(node,))
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True

    def test_non_rag_transform_without_provider_config_skips_check(self) -> None:
        node = _make_node(plugin="value_transform", options={"some_field": "value"})
        state = _make_state(source_options={}, nodes=(node,))
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        # No path option anywhere → check is recorded as passed (skipped-style).
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True


class TestValidatePipelineTransformProviderConfigManagedIdentityPolicy:
    """Web-authored RAG provider configs must not enable server managed identity."""

    def test_azure_search_managed_identity_provider_config_blocked(self) -> None:
        node = _make_node(
            plugin="rag_retrieval",
            options={
                "provider": "azure_search",
                "provider_config": {
                    "endpoint": "https://tenant-b.search.windows.net",
                    "index": "payroll",
                    "use_managed_identity": True,
                },
            },
        )
        state = _make_state(source_options={}, nodes=(node,))
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")

            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert any(c.name == "managed_identity_policy" and c.passed is False for c in result.checks)
        assert any("managed identity" in e.message.lower() for e in result.errors)
        assert result.readiness is not None
        assert any(b.code == "managed_identity_policy" for b in result.readiness.blockers)

    def test_azure_search_api_key_provider_config_remains_allowed(self) -> None:
        node = _make_node(
            plugin="rag_retrieval",
            options={
                "provider": "azure_search",
                "provider_config": {
                    "endpoint": "https://tenant-a.search.windows.net",
                    "index": "docs",
                    "api_key": "test-key",
                },
            },
        )
        state = _make_state(source_options={}, nodes=(node,))
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        managed_identity_check = next(c for c in result.checks if c.name == "managed_identity_policy")
        assert managed_identity_check.passed is True


class TestValidatePipelineLlmRetryBudgetPolicy:
    """Web-authored sequential multi-query LLM retries must be bounded."""

    @staticmethod
    def _llm_multi_query_options(**overrides: Any) -> dict[str, Any]:
        options: dict[str, Any] = {
            "provider": "openrouter",
            "model": "openai/gpt-4o-mini",
            "api_key": "test-key",
            "prompt_template": "Classify {{ text }}.",
            "schema": {"mode": "observed"},
            "required_input_fields": [],
            "queries": [
                {
                    "name": "classify",
                    "input_fields": {"text": "body"},
                }
            ],
        }
        options.update(overrides)
        return options

    def test_sequential_multi_query_llm_default_retry_budget_blocked(self) -> None:
        node = _make_node(plugin="llm", options=self._llm_multi_query_options())
        state = _make_state(source_options={}, nodes=(node,))
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")

            # Authoring-preflight path (allow_pending_interpretation_placeholders=True):
            # masks the llm node's pending llm_prompt_template/llm_model_choice review so
            # validation reaches llm_retry_budget_policy (declared #9, after
            # interpretation_review #6). This is the same flag the composer authoring
            # preflight uses; without it interpretation_review blocks first.
            result = validate_pipeline(state, settings, mock_yaml_gen, allow_pending_interpretation_placeholders=True)

        assert result.is_valid is False
        assert any(c.name == "llm_retry_budget_policy" and c.passed is False for c in result.checks)
        assert any("sequential multi-query LLM" in e.message for e in result.errors)
        assert result.readiness is not None
        assert any(b.code == "llm_retry_budget_policy" for b in result.readiness.blockers)

    def test_sequential_multi_query_llm_small_retry_budget_allowed(self) -> None:
        node = _make_node(
            plugin="llm",
            options=self._llm_multi_query_options(max_capacity_retry_seconds="30.0"),
        )
        state = _make_state(source_options={}, nodes=(node,))
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            # Authoring-preflight path so llm_retry_budget_policy (#9) is reached past
            # interpretation_review (#6); see the blocked-budget test above.
            result = validate_pipeline(state, settings, mock_yaml_gen, allow_pending_interpretation_placeholders=True)

        retry_budget_check = next(c for c in result.checks if c.name == "llm_retry_budget_policy")
        assert retry_budget_check.passed is True

    def test_pooled_multi_query_llm_numeric_string_pool_size_allowed(self) -> None:
        node = _make_node(plugin="llm", options=self._llm_multi_query_options(pool_size="2.0"))
        state = _make_state(source_options={}, nodes=(node,))
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            # Authoring-preflight path so llm_retry_budget_policy (#9) is reached past
            # interpretation_review (#6); see the blocked-budget test above.
            result = validate_pipeline(state, settings, mock_yaml_gen, allow_pending_interpretation_placeholders=True)

        retry_budget_check = next(c for c in result.checks if c.name == "llm_retry_budget_policy")
        assert retry_budget_check.passed is True


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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"

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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "semantic_contracts").passed is False

    def test_newline_text_passes_semantic_contracts_check(self) -> None:
        state = self._make_state(text_separator="\n")
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        # Subsequent checks may still fail (depends on fixture); we only
        # assert semantic_contracts itself passed.
        assert _check(result, "semantic_contracts").passed is True

    def test_single_query_llm_string_rejected_before_json_explode_runtime(self) -> None:
        """A single-query LLM response_field is str, not a json_explode array."""
        state = CompositionState(
            metadata=PipelineMetadata(name="llm-json-explode-contract"),
            version=1,
            edges=(),
            source=SourceSpec(
                plugin="csv",
                on_success="llm_in",
                options={
                    "path": "/tmp/test_data/blobs/prompts.csv",
                    "schema": {"mode": "fixed", "fields": ["topic: str"]},
                },
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="ask_llm",
                    node_type="transform",
                    plugin="llm",
                    input="llm_in",
                    on_success="explode_in",
                    on_error="discard",
                    options={
                        "provider": "openrouter",
                        "api_key": "test-key",
                        "model": "openai/gpt-4o",
                        "prompt_template": "Return three themes.",
                        "response_field": "llm_response",
                        "schema": {"mode": "observed"},
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="explode_themes",
                    node_type="transform",
                    plugin="json_explode",
                    input="explode_in",
                    on_success="results",
                    on_error="discard",
                    options={
                        "schema": {
                            "mode": "flexible",
                            "fields": ["llm_response: str"],
                        },
                        "array_field": "llm_response",
                        "output_field": "theme",
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
                    name="results",
                    plugin="json",
                    options={"path": "/tmp/test_data/outputs/themes.json"},
                    on_write_failure="discard",
                ),
            ),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "semantic_contracts").passed is False
        assert any(
            error.component_id == "explode_themes"
            and "json_explode.array_field.list" in error.message
            and "value_type=str" in error.message
            and (
                "json_explode requires list field; LLM response_field is str. "
                "Use structured-output parsing or a transform that parses JSON object/string first."
            )
            in error.message
            for error in result.errors
        )
        mock_yaml_gen.generate_yaml.assert_not_called()


class TestValidatePipelineRelativePaths:
    """Relative paths must resolve against data_dir, not CWD."""

    def test_relative_sink_path_resolves_against_data_dir(self) -> None:
        """outputs/result.csv should resolve under {data_dir}/outputs/."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="primary", options={"path": "outputs/result.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False
        assert any("Path traversal" in e.message for e in result.errors)

    def test_relative_sink_path_under_own_session_blobs(self) -> None:
        """blobs/<session>/out.json resolves under the caller's own subtree
        (elspeth-bdc17cfdb1); the session segment is load-bearing."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="blob_out", options={"path": "blobs/sess-1/out.json"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen, session_id="sess-1")
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is True

    def test_relative_sink_path_at_blobs_root_blocked(self) -> None:
        """blobs/out.json (no session segment) resolves to the shared blobs
        root, which no session may write — the pre-fix contract this class
        used to pin."""
        state = _make_state(
            source_options={},
            outputs=(_make_output(name="blob_out", options={"path": "blobs/out.json"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        result = validate_pipeline(state, settings, mock_yaml_gen, session_id="sess-1")
        assert result.is_valid is False
        path_check = next(c for c in result.checks if c.name == "path_allowlist")
        assert path_check.passed is False


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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        mock_settings = _fake_settings()
        mock_load.return_value = mock_settings

        mock_bundle = _FakeRuntimeBundle()
        mock_instantiate.return_value = mock_bundle

        mock_graph = _runtime_graph_mock()
        mock_build_graph.return_value = mock_graph
        mock_assemble.return_value = _fake_pipeline_config()

        state = _make_state()
        settings = _make_settings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is True
        assert len(result.checks) == 16
        assert all(c.passed for c in result.checks)
        # B11 fix: path_allowlist check is always recorded
        assert _check(result, "path_allowlist").passed is True
        assert _check(result, "web_scrape_network_policy").passed is True
        assert _check(result, "llm_retry_budget_policy").passed is True
        assert _check(result, "llm_base_url_policy").passed is True
        assert _check(result, "secret_refs").passed is True
        assert _check(result, "blob_inline_refs").passed is True
        assert _check(result, "semantic_contracts").passed is True
        assert _check(result, "batch_transform_options").passed is True
        assert _check(result, "value_source_compliance").passed is True
        assert _check(result, "route_target_resolution").passed is True
        assert _check(result, "interpretation_review").passed is True
        assert result.readiness.authoring_valid is True
        assert result.readiness.execution_ready is True
        assert result.readiness.completion_ready is True
        assert result.errors == []
        assert result.warnings == []

        # Verify real engine functions were called
        mock_load.assert_called_once()
        mock_instantiate.assert_called_once_with(mock_settings, preflight_mode=True)
        mock_build_graph.assert_called_once()
        mock_graph.validate.assert_called_once()
        mock_assemble.assert_called_once()
        assert mock_assemble.call_args.kwargs["sources"] == mock_bundle.sources
        mock_graph.validate_edge_compatibility.assert_called_once()

    @patch("elspeth.web.execution.validation.assemble_and_validate_pipeline_config")
    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_validate_pipeline_surfaces_graph_warnings(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_assemble: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        mock_settings = _fake_settings()
        mock_load.return_value = mock_settings

        mock_bundle = _FakeRuntimeBundle()
        mock_instantiate.return_value = mock_bundle

        mock_graph = _runtime_graph_mock(
            validation_warnings=(
                GraphValidationWarning(
                    code="DIVERT_COALESCE_REQUIRE_ALL",
                    message="Transform 't_a' has on_error routing and feeds require_all coalesce 'join'.",
                    node_ids=("t_a", "join"),
                ),
            )
        )
        mock_build_graph.return_value = mock_graph
        mock_assemble.return_value = _fake_pipeline_config()

        state = _make_state()
        settings = _make_settings()

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0].component_id == "t_a"
        assert result.warnings[0].component_type == "graph"
        assert result.warnings[0].warning_code == "DIVERT_COALESCE_REQUIRE_ALL"
        assert "require_all coalesce" in result.warnings[0].message
        graph_check = _check(result, "graph_structure")
        assert graph_check.passed is True
        assert graph_check.affected_nodes == ("t_a",)


class TestValidatePipelineExportNoResolverSecretRef:
    """Export-YAML preflight runs ``validate_pipeline`` with ``secret_service=None``.

    A credential field wired as a ``{secret_ref: NAME}`` marker must not cause a
    plugin-instantiation false-failure in the no-resolver path. Regression guard
    for the bug where the YAML-export button reported "Current composition state
    failed runtime preflight" for a pipeline that *ran successfully*: run
    preflight resolves the secret, but export preflight (no resolver, by design,
    so resolved secret values can never reach error prose) tried to instantiate
    the LLM plugin from an unresolved ``{secret_ref}`` dict and hit
    ``api_key: Input should be a valid string``.
    """

    def _secret_ref_llm_state(self, tmp_path: Path) -> CompositionState:
        from elspeth.contracts.hashing import stable_hash

        # Source paths must resolve under data_dir/blobs, sink paths under
        # data_dir/outputs (allowed_source_directories / allowed_sink_directories).
        blobs_dir = tmp_path / "blobs"
        blobs_dir.mkdir()
        text_path = blobs_dir / "input.txt"
        text_path.write_text("hello world\n", encoding="utf-8")
        prompt_template = "Summarise: {{ row.text }}"
        model = "openai/gpt-4o-mini"
        # An LLM node with a prompt_template and a model carries resolved
        # prompt-template and model-choice review requirements once the guided
        # composer has surfaced them — mirror that so interpretation_review
        # passes and validation reaches the plugin-instantiation step (where the
        # secret-marker bug lives).
        interpretation_requirements = [
            {
                "id": "prompt_template_review:llm",
                "kind": "llm_prompt_template",
                "user_term": "llm_prompt_template:llm",
                "status": "resolved",
                "draft": prompt_template,
                "event_id": "evt-prompt",
                "accepted_value": prompt_template,
                "accepted_artifact_hash": None,
                "resolved_prompt_template_hash": stable_hash(prompt_template),
            },
            {
                "id": "model_choice_review:llm",
                "kind": "llm_model_choice",
                "user_term": "llm_model_choice:llm",
                "status": "resolved",
                "draft": model,
                "event_id": "evt-model",
                "accepted_value": model,
                "accepted_artifact_hash": None,
                "resolved_prompt_template_hash": stable_hash(model),
            },
        ]
        return CompositionState(
            source=SourceSpec(
                plugin="text",
                on_success="llm_in",
                options={"path": str(text_path), "column": "text", "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="llm",
                    node_type="transform",
                    plugin="llm",
                    input="llm_in",
                    on_success="main",
                    on_error="discard",
                    options={
                        "provider": "openrouter",
                        "model": model,
                        "prompt_template": prompt_template,
                        "response_field": "summary",
                        "required_input_fields": ["text"],
                        # The credential is wired as a secret_ref marker, exactly
                        # as the guided composer persists an LLM api_key.
                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                        "schema": {"mode": "observed", "guaranteed_fields": ["summary"]},
                        INTERPRETATION_REQUIREMENTS_KEY: interpretation_requirements,
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
                    name="main",
                    plugin="json",
                    options={"path": "outputs/out.json", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )

    def test_unresolved_secret_ref_does_not_block_export_preflight(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from elspeth.web.composer import yaml_generator as composer_yaml_generator

        state = self._secret_ref_llm_state(tmp_path)
        result = validate_pipeline(
            state,
            SimpleNamespace(data_dir=tmp_path),
            composer_yaml_generator,
            secret_service=None,
            user_id=None,
        )

        api_key_errors = [e for e in result.errors if "api_key" in (e.message or "")]
        assert not api_key_errors, f"export preflight false-failed on a wired secret_ref marker: {[e.message for e in api_key_errors]}"
        checks = {c.name: c for c in result.checks}
        assert checks["plugin_instantiation"].passed is True, checks["plugin_instantiation"].detail

    def test_export_yaml_still_serializes_the_secret_ref_marker(self, tmp_path: Path) -> None:
        """Leak invariant: the placeholder substituted for preflight must never
        reach the exported YAML, which is generated separately from the original
        state and must keep the real ``{secret_ref}`` marker.
        """
        from elspeth.core.secrets import SECRET_REF_VALIDATION_PLACEHOLDER
        from elspeth.web.composer.yaml_generator import generate_public_yaml

        state = self._secret_ref_llm_state(tmp_path)
        public_yaml = generate_public_yaml(state)
        assert "secret_ref: OPENROUTER_API_KEY" in public_yaml
        assert SECRET_REF_VALIDATION_PLACEHOLDER not in public_yaml


class TestValidatePipelineSettingsFailure:
    def test_file_backed_template_options_fail_during_web_settings_load_before_plugins(self) -> None:
        pipeline_yaml = """
sources:
  source:
    plugin: csv
    on_success: transform_in
    options: {}
transforms:
  - name: classify
    plugin: llm
    input: transform_in
    on_success: results
    on_error: results
    options:
      template_file: prompt.txt
      lookup_file: lookup.yaml
      system_prompt_file: system.txt
sinks:
  primary:
    plugin: json
    on_write_failure: discard
    options:
      path: output.jsonl
"""
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = pipeline_yaml
        state = _make_state(nodes=(_make_node(plugin="llm"),), outputs=(_make_output(),))
        settings = _make_settings()

        with patch("elspeth.web.execution.validation.instantiate_runtime_plugins") as mock_instantiate:
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "settings_load").passed is False
        assert any("template_file" in error.message and "load_settings()" in error.message for error in result.errors)
        mock_instantiate.assert_not_called()

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    def test_pydantic_validation_error_short_circuits(
        self,
        mock_load: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}"
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: unknown\n  options: {}"
        mock_load.return_value = _fake_settings()
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = """
sources:
  primary:
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        mock_load.return_value = _fake_settings()
        mock_bundle = _FakeRuntimeBundle()
        mock_instantiate.return_value = mock_bundle

        mock_graph = _runtime_graph_mock(validate_side_effect=GraphValidationError("Route destination 'nonexistent' in gate_1 not found"))
        mock_build_graph.return_value = mock_graph

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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        mock_load.return_value = _fake_settings()
        mock_bundle = _FakeRuntimeBundle()
        mock_instantiate.return_value = mock_bundle

        mock_graph = _runtime_graph_mock(
            edge_validation_side_effect=GraphValidationError("Schema mismatch on edge transform_1 -> sink_primary")
        )
        mock_build_graph.return_value = mock_graph

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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        mock_settings = _fake_settings()
        mock_load.return_value = mock_settings
        mock_bundle = _FakeRuntimeBundle()
        mock_instantiate.return_value = mock_bundle
        mock_graph = _runtime_graph_mock()
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
        data = {"sources": {"primary": {"options": {"api_key": {"secret_ref": "MY_KEY"}}}}}
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        assert secret_check.outcome_code == "secret_refs.unresolved"
        assert "MISSING_KEY" in secret_check.detail
        assert any("MISSING_KEY" in e.message for e in result.errors)
        # Downstream checks should be skipped
        assert any("Skipped" in c.detail for c in result.checks if c.name == "settings_load")
        assert _check(result, "settings_load").outcome_code == "validation.skipped_after_failure"

    def test_second_named_source_missing_ref_fails_validation(self) -> None:
        state = CompositionState(
            source=None,
            sources={
                "orders": _make_source({"api_key": {"secret_ref": "ORDERS_KEY"}}),
                "refunds": _make_source({"api_key": {"secret_ref": "MISSING_REFUNDS_KEY"}}),
            },
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = _make_settings()
        mock_yaml_gen = _FakeYamlGenerator()
        secret_svc = FakeSecretService(available_refs={"ORDERS_KEY"})

        result = validate_pipeline(
            state,
            settings,
            mock_yaml_gen,
            secret_service=secret_svc,
            user_id="user-1",
        )

        assert result.is_valid is False
        assert _check(result, "secret_refs").passed is False
        assert "MISSING_REFUNDS_KEY" in _check(result, "secret_refs").detail
        assert any("MISSING_REFUNDS_KEY" in error.message for error in result.errors)

    def test_all_refs_present_passes(self) -> None:
        """Validation passes when all secret refs are resolvable."""
        state = _make_state(
            source_options={"api_key": {"secret_ref": "MY_KEY"}},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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
        assert secret_check.outcome_code == "secret_refs.resolved"
        assert "1 secret reference(s) resolved" in secret_check.detail

    def test_no_secret_refs_emits_structured_no_refs_outcome(self) -> None:
        """A composition with no secret markers records that fact structurally."""
        state = _make_state(source_options={})
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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

        secret_check = next(c for c in result.checks if c.name == "secret_refs")
        assert secret_check.passed is True
        assert secret_check.outcome_code == "secret_refs.no_refs"

    def test_no_secret_service_skips_check(self) -> None:
        """Without secret_service, the check is skipped (passed=True)."""
        state = _make_state(
            source_options={"api_key": {"secret_ref": "KEY"}},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("invalid settings")
            result = validate_pipeline(state, settings, mock_yaml_gen)

        secret_check = next(c for c in result.checks if c.name == "secret_refs")
        assert secret_check.passed is True
        assert secret_check.outcome_code == "secret_refs.skipped_no_service"
        assert "skipped" in secret_check.detail.lower()

    def test_refs_in_node_options_detected(self) -> None:
        """Secret refs in node options are found and validated."""
        state = _make_state(
            source_options={},
            nodes=(_make_node(options={"token": {"secret_ref": "NODE_TOKEN"}}),),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
                "api_key": {"secret_ref": "REF_A"},
                "token": {"secret_ref": "REF_B"},
            },
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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

    Source: docs/composer/evidence/composer-llm-eval-2026-05-01.md S1A architectural finding #3.
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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

    def test_secret_ref_in_non_credential_web_scrape_field_rejected(self) -> None:
        """A wired secret marker in wire-visible non-credential text is a leak."""
        state = _make_state(
            source_options={},
            nodes=(
                _make_node(
                    plugin="web_scrape",
                    options={
                        "url_field": "url",
                        "http": {
                            "abuse_contact": {"secret_ref": "ANY_SECRET"},
                            "scraping_reason": "research",
                            "allowed_hosts": "public_only",
                        },
                    },
                ),
            ),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        secret_svc = FakeSecretService(available_refs={"ANY_SECRET"})

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
        assert "web_scrape" in secret_check.detail
        assert "http.abuse_contact" in secret_check.detail
        assert "ANY_SECRET" in secret_check.detail
        errors = [e for e in result.errors if "secret_ref" in e.message]
        assert errors
        assert any(e.component_id == "test_node" and e.component_type == "transform" for e in errors)
        assert any("credential-bearing fields" in e.message for e in errors)
        assert any("api_key" in e.message for e in errors)
        mock_yaml_gen.generate_yaml.assert_not_called()

    def test_database_sink_url_secret_ref_passes(self) -> None:
        """Database sink URL is a credential-bearing DSN field."""
        state = _make_state(
            source_options={},
            outputs=(
                _make_output(
                    name="db_out",
                    plugin="database",
                    options={
                        "url": {"secret_ref": "DATABASE_URL"},
                        "table": "audit_rows",
                        "schema": {"mode": "observed"},
                    },
                ),
            ),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        secret_svc = FakeSecretService(available_refs={"DATABASE_URL"})

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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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

    def test_json_source_data_key_literal_passes(self) -> None:
        """Regression (elspeth-61f2c0732e): ``data_key`` is a structural JSON
        extraction key, not a credential — a literal value must not trip the
        fabricated_secret check and block /validate + /execute."""
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/data.json", "data_key": "results"},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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

        secret_check = _check(result, "secret_refs")
        assert secret_check.passed is True, secret_check.detail
        assert not any(e.error_code == "fabricated_secret" for e in result.errors)

    def test_dataverse_alternate_key_literal_passes(self) -> None:
        """Regression (elspeth-61f2c0732e): ``alternate_key`` names the
        Dataverse upsert routing column — structural, never a credential."""
        state = _make_state(
            source_options={},
            outputs=(
                _make_output(
                    name="dv_out",
                    plugin="dataverse",
                    options={"alternate_key": "crabc_code"},
                ),
            ),
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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

        secret_check = _check(result, "secret_refs")
        assert secret_check.passed is True, secret_check.detail
        assert not any(e.error_code == "fabricated_secret" for e in result.errors)

    def test_non_credential_string_unaffected(self) -> None:
        """Literal strings in non-credential fields pass — only credential
        fields are scope of the fabrication check."""
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/data.csv", "delimiter": ","},
        )
        settings = _make_settings()
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)

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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = (
            "source:\n  plugin: csv\n  on_success: transform_in\n"
            "  on_validation_failure: discard\n  options:\n"
            "    api_key:\n      secret_ref: MY_KEY\n"
        )
        secret_svc = FakeSecretService(available_refs={"MY_KEY"})

        mock_settings = _fake_settings()
        mock_load_string.return_value = mock_settings
        mock_bundle = _FakeRuntimeBundle()
        mock_instantiate.return_value = mock_bundle
        mock_graph = _runtime_graph_mock()
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = (
            "source:\n  plugin: csv\n  on_success: transform_in\n"
            "  on_validation_failure: discard\n  options:\n"
            "    api_key: ${OPENROUTER_API_KEY}\n"
        )
        secret_svc = FakeSecretService(available_refs={"OPENROUTER_API_KEY"})

        mock_settings = _fake_settings()
        mock_load_string.return_value = mock_settings
        mock_bundle = _FakeRuntimeBundle()
        mock_instantiate.return_value = mock_bundle
        mock_graph = _runtime_graph_mock()
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = (
            "source:\n  plugin: csv\n  on_success: transform_in\n"
            "  on_validation_failure: discard\n  options:\n"
            "    url: https://example.com/data\n"
        )

        mock_settings = _fake_settings()
        mock_load_string.return_value = mock_settings
        mock_bundle = _FakeRuntimeBundle()
        mock_instantiate.return_value = mock_bundle
        mock_graph = _runtime_graph_mock()
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = """
sources:
  primary:
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
        assert parsed["sources"]["primary"]["options"]["path"] == "/tmp/test_data/blobs/session/input.csv"
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = """
sources:
  primary:
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
        assert parsed["sources"]["primary"]["options"]["path"] == "/tmp/test_data/blobs/input.csv"
        assert parsed["sinks"]["main"]["options"]["path"] == "/tmp/test_data/outputs/out.csv"


class TestValidatePipelineRuntimeCheckBoundaries:
    def test_runtime_graph_validation_check_order_matches_named_constants(self) -> None:
        from elspeth.web.execution.preflight import (
            RUNTIME_CHECK_GRAPH_STRUCTURE,
            RUNTIME_CHECK_PLUGIN_INSTANTIATION,
            RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
            RUNTIME_GRAPH_VALIDATION_CHECKS,
        )
        from elspeth.web.execution.schemas import VALIDATION_BLOCKING_CHECK_NAMES

        assert RUNTIME_GRAPH_VALIDATION_CHECKS == (
            RUNTIME_CHECK_PLUGIN_INSTANTIATION,
            RUNTIME_CHECK_GRAPH_STRUCTURE,
            RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
        )
        assert tuple(_ALL_CHECKS) == VALIDATION_BLOCKING_CHECK_NAMES

    def test_validate_pipeline_success_surfaces_declared_runtime_graph_checks(self) -> None:
        from elspeth.web.execution.preflight import RUNTIME_GRAPH_VALIDATION_CHECKS

        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/input.csv"},
            outputs=(_make_output({"path": "/tmp/test_data/outputs/out.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = """
sources:
  primary:
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
        fake_graph = _runtime_graph_mock()

        with (
            patch("elspeth.web.execution.validation.load_settings_from_yaml_string", return_value=_fake_settings()),
            patch("elspeth.web.execution.validation.instantiate_runtime_plugins", return_value=_FakeRuntimeBundle()) as mock_instantiate,
            patch("elspeth.web.execution.validation.build_runtime_graph", return_value=fake_graph),
            patch("elspeth.web.execution.validation.assemble_and_validate_pipeline_config", return_value=_fake_pipeline_config()),
        ):
            result = validate_pipeline(state, settings, mock_yaml_gen)

        passed_names = {check.name for check in result.checks if check.passed}
        assert set(RUNTIME_GRAPH_VALIDATION_CHECKS).issubset(passed_names)
        assert mock_instantiate.call_args.kwargs == {"preflight_mode": True}
        fake_graph.validate.assert_called_once_with()
        fake_graph.validate_edge_compatibility.assert_called_once_with()

    def test_validate_pipeline_emits_checks_in_declared_order(self) -> None:
        # Regression guard for the *bug class*, not just one case: the physical
        # emission order of the blocking checks must match
        # VALIDATION_BLOCKING_CHECK_NAMES. A check appended at the wrong physical
        # spot (as managed_identity_policy/llm_retry_budget_policy once were —
        # emitted before web_scrape_network_policy despite being declared after it)
        # leaves a later-declared gate's pass record ahead of an earlier-declared
        # gate, which corrupts the skipped-after-failure trail on any earlier
        # failure. The order-vs-constant test only checks the declared tuple; this
        # asserts the live emission sequence on a fully-passing pipeline.
        from elspeth.web.execution.schemas import VALIDATION_BLOCKING_CHECK_NAMES

        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/input.csv"},
            outputs=(_make_output({"path": "/tmp/test_data/outputs/out.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "sources: {}\nsinks: {}\n"
        fake_graph = _runtime_graph_mock()
        with (
            patch("elspeth.web.execution.validation.load_settings_from_yaml_string", return_value=_fake_settings()),
            patch("elspeth.web.execution.validation.instantiate_runtime_plugins", return_value=_FakeRuntimeBundle()),
            patch("elspeth.web.execution.validation.build_runtime_graph", return_value=fake_graph),
            patch("elspeth.web.execution.validation.assemble_and_validate_pipeline_config", return_value=_fake_pipeline_config()),
        ):
            result = validate_pipeline(state, settings, mock_yaml_gen)

        declared_index = {name: i for i, name in enumerate(VALIDATION_BLOCKING_CHECK_NAMES)}
        emitted = [check.name for check in result.checks if check.name in declared_index]
        emitted_indices = [declared_index[name] for name in emitted]
        assert emitted_indices == sorted(emitted_indices), f"blocking checks emitted out of declared order: {emitted}"
        # And the relocation is concretely asserted: the two policy checks sit
        # after blob_inline_refs (their declared #8/#9 home), not before web_scrape.
        assert emitted.index("managed_identity_policy") > emitted.index("blob_inline_refs")
        assert emitted.index("llm_retry_budget_policy") > emitted.index("managed_identity_policy")
        assert emitted.index("web_scrape_network_policy") < emitted.index("managed_identity_policy")

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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = """
sources:
  primary:
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
        fake_settings = _fake_settings()
        fake_graph = _runtime_graph_mock(validate_side_effect=GraphValidationError("bad graph"))
        mock_load.return_value = fake_settings
        mock_instantiate.return_value = _FakeRuntimeBundle()
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
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = """
sources:
  primary:
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
        fake_settings = _fake_settings()
        fake_graph = _runtime_graph_mock(edge_validation_side_effect=GraphValidationError("schema mismatch"))
        mock_load.return_value = fake_settings
        mock_instantiate.return_value = _FakeRuntimeBundle()
        mock_build_graph.return_value = fake_graph
        mock_assemble.return_value = _fake_pipeline_config()

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "graph_structure").passed is True
        assert _check(result, "route_target_resolution").passed is True
        assert _check(result, "schema_compatibility").passed is False


class TestEdgeContractFailureFormatting:
    """Tier 1.5 final hardening — composer LLM-actionable preflight error format.

    The cohort report 2026-05-07 identified intermediate-edge schema-contract
    drift as the dominant pre-existing failure mode (e.g. WebScrapeOutput.fetch_status:int
    vs LineExplodeInput.fetch_status:str|None on a model-overdeclared consumer
    schema). The structural fix preserves CompatibilityResult through the
    GraphValidationError → ValidationError translation so the error message
    surfaced to the composer LLM is named, per-field, and points at concrete
    tool-call shapes.

    These tests pin the message and suggestion shapes so prose drift is
    visible in PRs. The exact wording is allowed to evolve, but the contract
    points (named producer/consumer node IDs, per-field detail, both fix
    options, patch_node_options arguments) are load-bearing.
    """

    @staticmethod
    def _make_edge_error(
        *,
        from_node_id: str = "web_scrape_a1b2c3",
        to_node_id: str = "line_explode_d4e5f6",
        producer_schema_name: str = "WebScrapeOutput",
        consumer_schema_name: str = "LineExplodeInput",
        missing_fields: tuple[str, ...] = (),
        type_mismatches: tuple[tuple[str, str, str], ...] = (),
        extra_fields: tuple[str, ...] = (),
        constraint_mismatches: tuple[tuple[str, str], ...] = (),
    ) -> EdgeContractError:
        result = CompatibilityResult(
            compatible=False,
            missing_fields=missing_fields,
            type_mismatches=type_mismatches,
            extra_fields=extra_fields,
            constraint_mismatches=constraint_mismatches,
        )
        return EdgeContractError(
            f"Edge from '{from_node_id}' to '{to_node_id}' invalid",
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            producer_schema_name=producer_schema_name,
            consumer_schema_name=consumer_schema_name,
            compatibility_result=result,
            component_type="transform",
        )

    def test_edge_contract_error_is_graph_validation_error(self) -> None:
        """Subclass: existing catch-GraphValidationError sites keep working."""
        exc = self._make_edge_error(
            type_mismatches=(("fetch_status", "str | None", "int"),),
        )
        assert isinstance(exc, GraphValidationError)
        # component_id is the consumer (preserves existing semantics).
        assert exc.component_id == "line_explode_d4e5f6"
        assert exc.component_type == "transform"

    def test_edge_contract_error_carries_structured_fields(self) -> None:
        exc = self._make_edge_error(
            type_mismatches=(("fetch_status", "str | None", "int"),),
        )
        assert exc.from_node_id == "web_scrape_a1b2c3"
        assert exc.to_node_id == "line_explode_d4e5f6"
        assert exc.producer_schema_name == "WebScrapeOutput"
        assert exc.consumer_schema_name == "LineExplodeInput"
        assert exc.compatibility_result.compatible is False
        assert exc.compatibility_result.type_mismatches == (("fetch_status", "str | None", "int"),)

    # ── Captured-RED scenario regression ─────────────────────────────────

    def test_format_captured_red_type_mismatch_names_nodes_and_field(self) -> None:
        """The exact captured-RED case from session 20260506T160557Z."""
        exc = self._make_edge_error(
            from_node_id="transform_fetch_rules_46d77f2bcb4a",
            to_node_id="transform_split_lines_c3322ba122ca",
            type_mismatches=(("fetch_status", "str | None", "int"),),
        )
        message, _suggestion = _format_edge_contract_failure(exc)

        # Both nodes named (the model uses these as node_id= arguments).
        assert "transform_fetch_rules_46d77f2bcb4a" in message
        assert "transform_split_lines_c3322ba122ca" in message
        # Schema names surfaced (informational).
        assert "WebScrapeOutput" in message
        assert "LineExplodeInput" in message
        # Per-field detail with data-flow nomenclature.
        assert "fetch_status" in message
        assert "consumer requires 'str | None'" in message
        assert "producer emits 'int'" in message
        # Producer/consumer roles labelled at the top (not just field-level).
        assert "producer node 'transform_fetch_rules_46d77f2bcb4a'" in message
        assert "consumer node 'transform_split_lines_c3322ba122ca'" in message

    def test_suggestion_lists_both_options_with_node_ids(self) -> None:
        exc = self._make_edge_error(
            type_mismatches=(("fetch_status", "str | None", "int"),),
        )
        suggestion = _build_edge_contract_suggestion(exc)

        # Option (a): patch consumer.
        assert "(a)" in suggestion
        assert "patch_node_options(node_id='line_explode_d4e5f6'" in suggestion
        # Option (b): patch producer.
        assert "(b)" in suggestion
        assert "patch_node_options(node_id='web_scrape_a1b2c3'" in suggestion
        # Steers toward (a) — most failures are consumer over-declaration.
        assert "Try option (a) first" in suggestion or "option (a) first" in suggestion

    def test_suggestion_maps_dag_transform_ids_to_composer_node_ids(self) -> None:
        exc = self._make_edge_error(
            from_node_id="source_csv_a1b2c3",
            to_node_id="transform_split_lines_d4e5f6",
            type_mismatches=(("fetch_status", "str | None", "int"),),
        )
        state = _make_state(
            source_options={"schema": {"mode": "observed"}},
            nodes=(
                NodeSpec(
                    id="split_lines",
                    node_type="transform",
                    plugin="line_explode",
                    input="source_out",
                    on_success="results",
                    on_error="discard",
                    options={},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            outputs=(_make_output(name="results"),),
        )
        graph = _EdgeSuggestionGraph(
            sources=("source_csv_a1b2c3",),
            node_configs={"source_csv_a1b2c3": {"source_name": "source"}},
            transform_id_map={0: "transform_split_lines_d4e5f6"},
            sink_id_map={"results": "sink_results_f7g8h9"},
        )

        suggestion = _build_edge_contract_suggestion(exc, state=state, graph=graph)

        assert "patch_node_options(node_id='split_lines'" in suggestion
        assert "patch_node_options(node_id='transform_split_lines_d4e5f6'" not in suggestion
        assert "patch_source_options(patch={'schema': {...}})" in suggestion
        assert "patch_node_options(node_id='source_csv_a1b2c3'" not in suggestion

    def test_suggestion_maps_multi_source_dag_ids_to_named_source_patch_tool(self) -> None:
        exc = self._make_edge_error(
            from_node_id="source_csv_refunds_a1b2c3",
            to_node_id="transform_normalize_d4e5f6",
            missing_fields=("refund_id",),
        )
        orders = _make_source({"schema": {"mode": "observed"}})
        refunds = SourceSpec(
            plugin="csv",
            on_success="normalize",
            options={"schema": {"mode": "observed"}},
            on_validation_failure="discard",
        )
        state = CompositionState(
            sources={"orders": orders, "refunds": refunds},
            nodes=(
                NodeSpec(
                    id="normalize",
                    node_type="transform",
                    plugin="field_mapper",
                    input="normalize",
                    on_success="results",
                    on_error="discard",
                    options={},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(_make_output(name="results"),),
            metadata=PipelineMetadata(),
            version=1,
        )
        graph = _EdgeSuggestionGraph(
            sources=("source_csv_orders_z9y8x7", "source_csv_refunds_a1b2c3"),
            node_configs={
                "source_csv_orders_z9y8x7": {"source_name": "orders"},
                "source_csv_refunds_a1b2c3": {"source_name": "refunds"},
            },
            transform_id_map={0: "transform_normalize_d4e5f6"},
            sink_id_map={"results": "sink_results_f7g8h9"},
        )

        suggestion = _build_edge_contract_suggestion(exc, state=state, graph=graph)

        graph.get_source.assert_not_called()
        assert "source 'refunds' (csv)" in suggestion
        assert "patch_source_options(source_name='refunds', patch={'schema': {...}})" in suggestion
        assert "patch_source_options(patch={'schema': {...}})" not in suggestion
        assert "patch_node_options(node_id='source_csv_refunds_a1b2c3'" not in suggestion

    def test_suggestion_maps_dag_sink_ids_to_output_patch_tool(self) -> None:
        exc = self._make_edge_error(
            from_node_id="transform_clean_text_a1b2c3",
            to_node_id="sink_results_d4e5f6",
            missing_fields=("content",),
        )
        state = _make_state(
            source_options={"schema": {"mode": "observed"}},
            nodes=(
                NodeSpec(
                    id="clean_text",
                    node_type="transform",
                    plugin="field_mapper",
                    input="source_out",
                    on_success="results",
                    on_error="discard",
                    options={},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            outputs=(_make_output(name="results"),),
        )
        graph = _EdgeSuggestionGraph(
            sources=("source_csv_z9y8x7",),
            node_configs={"source_csv_z9y8x7": {"source_name": "source"}},
            transform_id_map={0: "transform_clean_text_a1b2c3"},
            sink_id_map={"results": "sink_results_d4e5f6"},
        )

        suggestion = _build_edge_contract_suggestion(exc, state=state, graph=graph)

        assert "patch_output_options(sink_name='results'" in suggestion
        assert "patch_node_options(node_id='sink_results_d4e5f6'" not in suggestion
        assert "patch_node_options(node_id='clean_text'" in suggestion

    def test_suggestion_for_type_mismatch_mentions_changing_declared_type(self) -> None:
        exc = self._make_edge_error(
            type_mismatches=(("fetch_status", "str | None", "int"),),
        )
        suggestion = _build_edge_contract_suggestion(exc)
        assert "declared field type" in suggestion or "Change the declared field type" in suggestion

    def test_suggestion_for_missing_fields_mentions_dropping_required(self) -> None:
        exc = self._make_edge_error(
            missing_fields=("content",),
        )
        suggestion = _build_edge_contract_suggestion(exc)
        assert "Drop missing required fields" in suggestion

    def test_suggestion_for_extra_fields_mentions_flexible_mode(self) -> None:
        exc = self._make_edge_error(
            extra_fields=("debug_field",),
        )
        suggestion = _build_edge_contract_suggestion(exc)
        # Flexible mode is the canonical fix for extra-field rejection.
        assert "'flexible'" in suggestion

    # ── All issue categories ─────────────────────────────────────────────

    def test_missing_fields_block_present_when_applicable(self) -> None:
        exc = self._make_edge_error(missing_fields=("content", "fingerprint"))
        message, _ = _format_edge_contract_failure(exc)
        assert "Missing required fields" in message
        assert "'content'" in message
        assert "'fingerprint'" in message

    def test_constraint_mismatches_block_present_when_applicable(self) -> None:
        exc = self._make_edge_error(
            constraint_mismatches=(("price", "consumer requires finite floats"),),
        )
        message, _ = _format_edge_contract_failure(exc)
        assert "Constraint mismatches" in message
        assert "'price'" in message
        assert "consumer requires finite floats" in message

    def test_extra_fields_block_present_when_applicable(self) -> None:
        exc = self._make_edge_error(extra_fields=("debug_field",))
        message, _ = _format_edge_contract_failure(exc)
        assert "Extra fields forbidden by consumer" in message
        assert "'debug_field'" in message

    def test_combined_issue_categories_all_appear(self) -> None:
        exc = self._make_edge_error(
            missing_fields=("content",),
            type_mismatches=(("count", "int", "str"),),
            extra_fields=("debug",),
            constraint_mismatches=(("price", "consumer requires finite floats"),),
        )
        message, _ = _format_edge_contract_failure(exc)
        assert "Missing required fields" in message
        assert "Type mismatches" in message
        assert "Constraint mismatches" in message
        assert "Extra fields forbidden by consumer" in message

    # ── End-to-end through validate_pipeline ─────────────────────────────

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    @patch("elspeth.web.execution.validation.assemble_and_validate_pipeline_config")
    def test_edge_contract_error_produces_rich_validation_error(
        self,
        mock_assemble: MagicMock,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """validate_pipeline must surface the rich message + suggestion when
        graph.validate_edge_compatibility() raises EdgeContractError."""
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        mock_load.return_value = _fake_settings()
        mock_bundle = _FakeRuntimeBundle()
        mock_instantiate.return_value = mock_bundle

        edge_exc = self._make_edge_error(
            type_mismatches=(("fetch_status", "str | None", "int"),),
        )
        mock_graph = _runtime_graph_mock(edge_validation_side_effect=edge_exc)
        mock_build_graph.return_value = mock_graph
        mock_assemble.return_value = _fake_pipeline_config()

        # source_options=None: the test exercises the rich-suggestion fall-back
        # path (``_edge_patch_target_for_node_id`` returns
        # ``_node_schema_patch_target`` when the targets dict is empty), so
        # the state must have no source. The default ``_make_state()`` now
        # adds a placeholder source to bypass the ``empty_pipeline``
        # short-circuit at the top of ``validate_pipeline``; that source
        # would populate the targets dict and route the lookup through
        # ``_unmapped_schema_patch_target`` instead — emitting a
        # ``get_pipeline_state`` suggestion rather than the expected
        # ``patch_node_options`` one.
        state = _make_state(source_options=None, nodes=(_make_node(),))
        settings = _make_settings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "schema_compatibility").passed is False
        # Exactly one error emitted from the schema-compat path.
        assert len(result.errors) == 1
        err = result.errors[0]
        # Rich multi-line message — not the legacy single-line str(exc).
        assert "producer node" in err.message
        assert "consumer node" in err.message
        assert "fetch_status" in err.message
        # Suggestion populated with concrete tool-call shapes.
        assert err.suggestion is not None
        assert "patch_node_options" in err.suggestion
        # Component attribution preserved.
        assert err.component_id == "line_explode_d4e5f6"
        assert err.component_type == "transform"

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    @patch("elspeth.web.execution.validation.assemble_and_validate_pipeline_config")
    def test_plain_graph_validation_error_falls_back_to_legacy_format(
        self,
        mock_assemble: MagicMock,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Non-edge-contract GraphValidationError keeps legacy str(exc)
        message and suggestion=None (other failure modes don't have
        structured per-field detail to enrich from)."""
        mock_yaml_gen = MagicMock(spec=YamlGenerator)
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"
        mock_load.return_value = _fake_settings()
        mock_bundle = _FakeRuntimeBundle()
        mock_instantiate.return_value = mock_bundle

        # Plain GraphValidationError — not the EdgeContractError subclass.
        mock_graph = _runtime_graph_mock(
            edge_validation_side_effect=GraphValidationError(
                "some other graph problem",
                component_id="node_x",
                component_type="transform",
            )
        )
        mock_build_graph.return_value = mock_graph
        mock_assemble.return_value = _fake_pipeline_config()

        state = _make_state()
        settings = _make_settings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert len(result.errors) == 1
        err = result.errors[0]
        # Legacy single-line message preserved.
        assert err.message == "some other graph problem"
        # No suggestion synthesized (we don't have structured fields to use).
        assert err.suggestion is None
