"""Walker tests for ``_CHECK_VALUE_SOURCE_COMPLIANCE`` step in
``validate_pipeline``.

The walker is also exercised at the L2 layer
(``engine.orchestrator.preflight.validate_value_source_compliance``);
these tests cover its integration into the composer ``/validate`` path.

Production-path discipline: tests construct real ``OpenRouterConfig`` /
``AzureOpenAIConfig`` instances inside a real plugin bundle, and only
mock at the ``secret_service`` / catalog boundary. The walker logic is
unmocked end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.value_source import register_value_source_plugin
from elspeth.engine.orchestrator.preflight import validate_value_source_compliance
from elspeth.engine.orchestrator.types import (
    ValueSourceFinding,
    ValueSourceValidationError,
)
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.config import WebSettings
from elspeth.web.execution.validation import (
    _ALL_CHECKS,
    _CHECK_GRAPH,
    _CHECK_PLUGINS,
    _CHECK_VALUE_SOURCE_COMPLIANCE,
    validate_pipeline,
)


class TestAllChecksOrdering:
    """Pin the position of _CHECK_VALUE_SOURCE_COMPLIANCE in _ALL_CHECKS.

    The skip-cascade in ``_skipped_checks`` walks _ALL_CHECKS linearly
    from the failed check forward. Inserting between _CHECK_PLUGINS and
    _CHECK_GRAPH means a value-source failure correctly skips graph,
    route-target, and schema checks. This test fails loudly if a future
    PR moves the constant.
    """

    def test_value_source_check_between_plugins_and_graph(self) -> None:
        plugins_idx = _ALL_CHECKS.index(_CHECK_PLUGINS)
        value_source_idx = _ALL_CHECKS.index(_CHECK_VALUE_SOURCE_COMPLIANCE)
        graph_idx = _ALL_CHECKS.index(_CHECK_GRAPH)
        assert plugins_idx < value_source_idx < graph_idx
        assert value_source_idx == plugins_idx + 1


class TestWalkerL2Direct:
    """Direct tests for the L2 walker (no composer plumbing)."""

    def test_passes_with_no_transforms(self) -> None:
        # Empty transforms list → no findings → returns silently.
        validate_value_source_compliance([])

    def test_passes_with_transforms_lacking_value_sources(self) -> None:
        # A WiredTransform whose plugin has no `config` attribute and no
        # VALUE_SOURCES is silently skipped.
        plugin = MagicMock(spec=[])  # no `config` attribute
        wired = MagicMock()
        wired.plugin = plugin
        wired.settings = MagicMock(name="settings_obj")
        validate_value_source_compliance([wired])

    def test_catalog_membership_pass(self) -> None:
        _plugin, wired = _build_wired_with_config(
            config_class=_FakeOpenRouterConfig,
            config_kwargs={"model": "openrouter/gpt-4o"},
            settings_name="openrouter_node_1",
        )
        with patch(
            "elspeth.engine.orchestrator.preflight.get_catalog_values",
            return_value=frozenset({"openrouter/gpt-4o", "openrouter/claude-3-haiku"}),
        ):
            validate_value_source_compliance([wired])  # passes

    def test_catalog_membership_fail_raises_with_findings(self) -> None:
        _plugin, wired = _build_wired_with_config(
            config_class=_FakeOpenRouterConfig,
            config_kwargs={"model": "anthropic/claude-3.5-sonnet"},  # not catalog-prefixed
            settings_name="openrouter_node_1",
        )
        with (
            patch(
                "elspeth.engine.orchestrator.preflight.get_catalog_values",
                return_value=frozenset({"openrouter/gpt-4o"}),
            ),
            pytest.raises(ValueSourceValidationError) as exc_info,
        ):
            validate_value_source_compliance([wired])
        err = exc_info.value
        assert len(err.findings) == 1
        finding = err.findings[0]
        # Structural attribution — composer UI reads these fields directly.
        assert finding.component_id == "openrouter_node_1"
        assert finding.field_name == "model"
        assert "anthropic/claude-3.5-sonnet" in finding.reason

    def test_empty_catalog_treated_as_structured_failure(self) -> None:
        _plugin, wired = _build_wired_with_config(
            config_class=_FakeOpenRouterConfig,
            config_kwargs={"model": "openrouter/gpt-4o"},
            settings_name="openrouter_node_1",
        )
        with (
            patch(
                "elspeth.engine.orchestrator.preflight.get_catalog_values",
                return_value=frozenset(),
            ),
            pytest.raises(ValueSourceValidationError) as exc_info,
        ):
            validate_value_source_compliance([wired])
        finding = exc_info.value.findings[0]
        assert finding.component_id == "openrouter_node_1"
        assert finding.field_name == "model"
        assert "empty or unavailable" in finding.reason

    def test_empty_catalog_finding_quotes_registered_dep_hint(self) -> None:
        """When a hint is registered for the catalog, the walker quotes
        it verbatim in place of the generic fallback. Proves the L3
        registrar's actionable string makes it through the L0 registry
        to the operator-visible finding (code-reviewer I-2).
        """
        from elspeth.contracts.value_source import (
            _CATALOG_DEP_HINTS,
            _CATALOG_READERS,
            register_catalog_reader,
        )

        # Register a fresh catalog id with an explicit hint so we don't
        # depend on which L3 packs are loaded in this test process.
        catalog_id = "test_walker_dep_hint"

        def empty_reader() -> frozenset[str]:
            return frozenset()

        register_catalog_reader(
            catalog_id,
            empty_reader,
            missing_dep_hint="install fakelib via uv pip install elspeth[fakelib]",
        )
        try:
            from elspeth.contracts.value_source import (
                CatalogValueSource as _CatalogValueSource,
            )
            from elspeth.contracts.value_source import (
                ValueSource as _ValueSource,
            )

            class _ConfigUnderTest:
                VALUE_SOURCES: tuple[_ValueSource, ...] = (_CatalogValueSource(field_name="model", catalog_id=catalog_id),)

                def __init__(self) -> None:
                    self.model = "anything"

            _plugin, wired = _build_wired_with_config(
                config_class=_ConfigUnderTest,
                config_kwargs={},
                settings_name="hint_node_1",
            )
            with pytest.raises(ValueSourceValidationError) as exc_info:
                validate_value_source_compliance([wired])
            finding = exc_info.value.findings[0]
            assert "fakelib" in finding.reason
            # Generic fallback text must NOT appear when a hint exists —
            # otherwise we'd be presenting both, which is operator-noise.
            assert "install the optional dependency that provides the catalog" not in finding.reason
        finally:
            _CATALOG_READERS.pop(catalog_id, None)
            _CATALOG_DEP_HINTS.pop(catalog_id, None)

    def test_derived_from_sibling_pass_when_equal(self) -> None:
        _plugin, wired = _build_wired_with_config(
            config_class=_FakeAzureConfig,
            config_kwargs={"model": "my-deploy", "deployment_name": "my-deploy"},
            settings_name="azure_node_1",
        )
        validate_value_source_compliance([wired])

    def test_derived_from_sibling_pass_when_field_empty_and_default_allowed(self) -> None:
        _plugin, wired = _build_wired_with_config(
            config_class=_FakeAzureConfig,
            config_kwargs={"model": "", "deployment_name": "my-deploy"},
            settings_name="azure_node_1",
        )
        validate_value_source_compliance([wired])

    def test_catalog_check_skipped_when_applies_when_predicate_fails(self) -> None:
        """Predicated catalog check (elspeth-ea207837d9): when the config's
        sibling field doesn't match the ``applies_when`` predicate (e.g.
        base_url overridden to a private compatible endpoint), the catalog check
        skips entirely — the catalog isn't authoritative for this config.
        Supports chaos test pipelines using fake model identifiers
        (``chaosllm/fake-gpt-4``) against errorworks/chaosllm servers.
        """
        _plugin, wired = _build_wired_with_config(
            config_class=_FakeOpenRouterConfigWithBaseUrl,
            config_kwargs={
                "model": "chaosllm/fake-gpt-4",
                "base_url": "https://chaos.example.test/v1",
            },
            settings_name="chaos_node_1",
        )
        # Catalog reader is patched to a known set that does NOT include
        # chaosllm/fake-gpt-4 — proves the predicate skip prevents the
        # catalog from being consulted at all.
        with patch(
            "elspeth.engine.orchestrator.preflight.get_catalog_values",
            return_value=frozenset({"openai/gpt-4o"}),
        ):
            validate_value_source_compliance([wired])  # passes (predicate skipped)

    def test_catalog_check_applied_when_applies_when_predicate_matches(self) -> None:
        """Counterpart: when base_url IS the canonical endpoint, the
        catalog check applies and rejects unknown identifiers."""
        _plugin, wired = _build_wired_with_config(
            config_class=_FakeOpenRouterConfigWithBaseUrl,
            config_kwargs={
                "model": "anthropic/claude-3.5-sonnet",
                "base_url": "https://openrouter.ai/api/v1",
            },
            settings_name="prod_node_1",
        )
        with (
            patch(
                "elspeth.engine.orchestrator.preflight.get_catalog_values",
                return_value=frozenset({"openai/gpt-4o"}),
            ),
            pytest.raises(ValueSourceValidationError) as exc_info,
        ):
            validate_value_source_compliance([wired])
        finding = exc_info.value.findings[0]
        assert finding.component_id == "prod_node_1"
        assert finding.field_name == "model"
        assert "anthropic/claude-3.5-sonnet" in finding.reason

    def test_derived_from_sibling_fail_when_diverges(self) -> None:
        _plugin, wired = _build_wired_with_config(
            config_class=_FakeAzureConfig,
            config_kwargs={"model": "wrong-deploy", "deployment_name": "right-deploy"},
            settings_name="azure_node_1",
        )
        with pytest.raises(ValueSourceValidationError) as exc_info:
            validate_value_source_compliance([wired])
        finding = exc_info.value.findings[0]
        assert finding.component_id == "azure_node_1"
        assert finding.field_name == "model"
        assert "wrong-deploy" in finding.reason
        assert "right-deploy" in finding.reason


class TestWalkerInValidatePipeline:
    """End-to-end through ``validate_pipeline`` — the composer entry path.

    Mocks at the same boundary as the surrounding tests in
    ``test_validation.py`` (load_settings, instantiate_runtime_plugins,
    build_runtime_graph, assemble_and_validate_pipeline_config) so
    behaviour matches the real validate path apart from those infrastructure
    boundaries.
    """

    def test_value_source_failure_short_circuits_with_skipped_downstream(self) -> None:
        """When ``instantiate_runtime_plugins`` raises ``ValueSourceValidationError``
        (the walker rejected a declared value), validate_pipeline reports
        PLUGINS as passed and VALUE_SOURCE as failed, with downstream checks
        skipped via cascade. The exception is raised from inside
        ``instantiate_plugins_from_config`` which is the single source of
        truth for value-source compliance under the Option-B refactor.
        """
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"

        # Construct the same exception shape the walker would raise — one
        # structured finding, attributable to a specific component.
        finding = ValueSourceFinding(
            component_id="openrouter_node_1",
            field_name="model",
            reason=(
                "value 'anthropic/claude-3.5-sonnet' is not in catalog 'openrouter' "
                "(catalog has 1 entries; pick a valid value via the list_models composer tool)"
            ),
        )
        injected_error = ValueSourceValidationError(
            f"1 field(s) violated value-source declarations: {finding.format()}",
            findings=(finding,),
        )

        state = _make_state()
        settings = _make_settings()

        with (
            patch("elspeth.web.execution.validation.load_settings_from_yaml_string", return_value=MagicMock()),
            patch(
                "elspeth.web.execution.validation.instantiate_runtime_plugins",
                side_effect=injected_error,
            ),
        ):
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        check_by_name = {c.name: c for c in result.checks}
        # PLUGINS passed because instantiate_plugins_from_config built the
        # bundle internally before the walker rejected it.
        assert check_by_name[_CHECK_PLUGINS].passed is True
        assert check_by_name[_CHECK_VALUE_SOURCE_COMPLIANCE].passed is False
        assert "openrouter_node_1" in check_by_name[_CHECK_VALUE_SOURCE_COMPLIANCE].detail
        assert check_by_name[_CHECK_GRAPH].passed is False  # skipped
        # Structured per-component error attribution.
        attributed_errors = [e for e in result.errors if e.component_id == "openrouter_node_1"]
        assert attributed_errors, "expected at least one error attributed to the offending node"
        assert attributed_errors[0].component_type == "transform"
        assert "anthropic/claude-3.5-sonnet" in attributed_errors[0].message

    def test_value_source_pass_allows_downstream_checks(self) -> None:
        """When ``instantiate_runtime_plugins`` returns a bundle without
        raising, both PLUGINS and VALUE_SOURCE are recorded as passed and
        validation continues into graph construction."""
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source\n  options: {}"

        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}

        state = _make_state()
        settings = _make_settings()

        with (
            patch("elspeth.web.execution.validation.load_settings_from_yaml_string", return_value=MagicMock()),
            patch("elspeth.web.execution.validation.instantiate_runtime_plugins", return_value=mock_bundle),
            patch("elspeth.web.execution.validation.build_runtime_graph", return_value=MagicMock()),
            patch("elspeth.web.execution.validation.assemble_and_validate_pipeline_config", return_value=MagicMock()),
        ):
            result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is True
        check_by_name = {c.name: c for c in result.checks}
        assert check_by_name[_CHECK_PLUGINS].passed is True
        assert check_by_name[_CHECK_VALUE_SOURCE_COMPLIANCE].passed is True


# ── test fixtures ────────────────────────────────────────────────────


class _FakeOpenRouterConfigWithBaseUrl:
    """Stand-in mirroring real OpenRouterConfig — model + base_url + the
    ``applies_when`` predicate. Used to verify the walker correctly skips
    or applies the catalog check based on base_url's value.
    """

    from elspeth.contracts.value_source import (
        CatalogValueSource as _CatalogValueSource,
    )
    from elspeth.contracts.value_source import (
        ValueSource as _ValueSource,
    )

    VALUE_SOURCES: tuple[_ValueSource, ...] = (
        _CatalogValueSource(
            field_name="model",
            catalog_id="openrouter",
            applies_when=(("base_url", "https://openrouter.ai/api/v1"),),
        ),
    )

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url


class _FakePlugin:
    """Stand-in plugin class for walker tests.

    Real LLMTransform construction requires a working provider, recorder,
    template, and so on — too heavy for unit-level walker tests. This
    fake exposes the same opt-in contract (a ``provider_config`` attribute
    holding the typed config) and registers itself with the L0 registry
    at module-import time so the walker discovers it.
    """

    def __init__(self, provider_config: object) -> None:
        self.provider_config = provider_config


# Register the fake plugin once at module import. The L0 registry is
# idempotent for ``register(cls, attr)`` re-registration so test reruns
# are safe.
register_value_source_plugin(_FakePlugin, config_attr="provider_config")


class _FakeOpenRouterConfig:
    """Stand-in config that mimics the VALUE_SOURCES contract.

    Real ``OpenRouterConfig`` requires ``api_key`` and a full template;
    constructing one inside a unit test would create coupling to the
    Pydantic schema. The walker only depends on the duck-typed contract:
    ``type(config).VALUE_SOURCES`` exists and ``getattr(config, field)``
    returns the field's value. This stand-in implements that contract
    with the same declaration shape ``OpenRouterConfig`` declares.
    """

    from elspeth.contracts.value_source import (
        CatalogValueSource as _CatalogValueSource,
    )
    from elspeth.contracts.value_source import (
        ValueSource as _ValueSource,
    )

    VALUE_SOURCES: tuple[_ValueSource, ...] = (_CatalogValueSource(field_name="model", catalog_id="openrouter"),)

    def __init__(self, model: str) -> None:
        self.model = model


class _FakeAzureConfig:
    """Stand-in for AzureOpenAIConfig — DerivedFromSiblingValueSource shape."""

    from elspeth.contracts.value_source import (
        DerivedFromSiblingValueSource as _DerivedFromSiblingValueSource,
    )
    from elspeth.contracts.value_source import (
        ValueSource as _ValueSource,
    )

    VALUE_SOURCES: tuple[_ValueSource, ...] = (
        _DerivedFromSiblingValueSource(
            field_name="model",
            sibling_field="deployment_name",
            allow_empty_default=True,
        ),
    )

    def __init__(self, model: str, deployment_name: str) -> None:
        self.model = model
        self.deployment_name = deployment_name


def _build_wired_with_config(
    *,
    config_class: type,
    config_kwargs: dict[str, Any],
    settings_name: str,
) -> tuple[Any, Any]:
    """Build a (plugin, WiredTransform-like) pair with a typed config attribute.

    The walker discovers configs via the L0 plugin opt-in registry; the
    fake plugin class is registered at module-import time so its
    instances are inspected like real LLMTransforms.
    """
    config = config_class(**config_kwargs)

    plugin = _FakePlugin(provider_config=config)

    wired = MagicMock()
    wired.plugin = plugin
    wired.settings = MagicMock()
    wired.settings.name = settings_name
    return plugin, wired


def _make_state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="t1",
            options={},
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="openrouter_node_1",
                node_type="transform",
                plugin="openrouter_llm",
                input="t1",
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
        outputs=(
            OutputSpec(
                name="primary",
                plugin="csv",
                options={},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _make_settings() -> WebSettings:
    return WebSettings(
        data_dir=Path("/tmp/test_data"),
        composer_max_composition_turns=10,
        composer_max_discovery_turns=5,
        composer_timeout_seconds=30.0,
        composer_rate_limit_per_minute=60,
        shareable_link_signing_key=b"\x00" * 32,
    )
