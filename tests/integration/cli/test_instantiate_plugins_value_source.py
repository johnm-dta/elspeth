"""Regression test pinning the value-source compliance call site at
``cli_helpers.instantiate_plugins_from_config``.

The CLI/runtime parity claim of the value-source PR rests on a single
call inside :func:`instantiate_plugins_from_config` —
``validate_value_source_compliance(bundle.transforms)``. Removing that
call would let hand-authored YAML containing a hallucinated model
identifier slip through every CLI path that doesn't go through the
composer ``/validate`` endpoint.

Existing bootstrap tests (``tests/integration/pipeline/test_bootstrap_preflight.py``)
patch :func:`instantiate_plugins_from_config` with
``MagicMock(spec=PluginBundle)`` and therefore never exercise the
walker. This test deliberately bypasses that mock seam: it loads real
settings via :func:`load_settings_from_yaml_string`, calls
:func:`instantiate_plugins_from_config` for real, and asserts a
:class:`ValueSourceValidationError` is raised. Removing the call site
breaks the test — exactly the regression-locking property
pr-test-analyzer Gap 2 asked for.

The catalog reader is patched to a fixed set so the test is independent
of whether ``litellm`` is installed locally and which version is
present. The patch site is the walker's import (``preflight.get_catalog_values``),
not the registry — same boundary the unit walker tests use, so the
test isolates the integration question (does the call happen?) from
the catalog-content question (which is unit-level coverage).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import load_settings_from_yaml_string
from elspeth.engine.orchestrator.value_source_validation import ValueSourceValidationError


def _build_yaml_with_model(model: str, csv_path: Path, sink_path: Path) -> str:
    """Minimal pipeline YAML with one OpenRouter LLM transform.

    Mirrors the wiring shape used in ``examples/transform_pipeline``:
    plugin names are short (``csv``, not ``csv_source``); each node
    declares ``on_success`` / ``on_error`` to wire the DAG; the
    transform's ``input`` matches its own ``name`` (the convention the
    settings loader expects). ``csv_path`` and ``sink_path`` are real
    on-disk paths so plugin instantiation completes; the file contents
    are irrelevant because we never run the pipeline.
    """
    return textwrap.dedent(
        f"""
        landscape:
          url: "sqlite:///audit.db"
        sources:
          primary:
            plugin: csv
            on_success: enrich
            options:
              path: "{csv_path}"
              on_validation_failure: discard
              schema:
                mode: observed
        transforms:
          - name: enrich
            plugin: llm
            input: enrich
            on_success: output
            on_error: output
            options:
              provider: openrouter
              model: "{model}"
              api_key: "placeholder"
              prompt_template: "Hello"
              schema:
                mode: observed
        sinks:
          output:
            plugin: csv
            on_write_failure: discard
            options:
              path: "{sink_path}"
              schema:
                mode: observed
        """
    ).strip()


@pytest.fixture
def csv_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Tmp paths used for source CSV and sink CSV."""
    src = tmp_path / "in.csv"
    src.write_text("col\nvalue\n", encoding="utf-8")
    sink = tmp_path / "out.csv"
    return src, sink


class TestInstantiatePluginsValueSourceRegression:
    """Locks the value-source compliance call site against silent regression."""

    def test_hallucinated_openrouter_model_is_rejected_at_instantiation(self, csv_paths: tuple[Path, Path]) -> None:
        """Removing the walker call inside
        :func:`instantiate_plugins_from_config` causes this test to fail.
        """
        src, sink = csv_paths
        yaml_text = _build_yaml_with_model("anthropic/claude-bogus-not-in-catalog", src, sink)
        config = load_settings_from_yaml_string(yaml_text)
        with (
            patch(
                "elspeth.engine.orchestrator.preflight.get_catalog_values",
                return_value=frozenset({"openai/gpt-4o"}),
            ),
            pytest.raises(ValueSourceValidationError) as exc_info,
        ):
            instantiate_plugins_from_config(config)
        # Structured attribution — the finding names the field whose
        # value violates the catalog declaration.
        offending = [f for f in exc_info.value.findings if f.field_name == "model" and "anthropic/claude-bogus-not-in-catalog" in f.reason]
        assert offending, (
            f"expected a finding naming the hallucinated model; got "
            f"{[(f.component_id, f.field_name, f.reason) for f in exc_info.value.findings]}"
        )

    def test_valid_openrouter_model_passes_instantiation(self, csv_paths: tuple[Path, Path]) -> None:
        """Negative control. When the model is in the patched catalog,
        the same path returns a bundle without raising. Proves the
        positive test above is testing the right gate (not just any
        instantiation failure).
        """
        src, sink = csv_paths
        yaml_text = _build_yaml_with_model("openai/gpt-4o", src, sink)
        config = load_settings_from_yaml_string(yaml_text)
        with patch(
            "elspeth.engine.orchestrator.preflight.get_catalog_values",
            return_value=frozenset({"openai/gpt-4o", "anthropic/claude-3.5-sonnet"}),
        ):
            bundle = instantiate_plugins_from_config(config)
        assert len(bundle.transforms) == 1

    def test_empty_catalog_finding_quotes_registered_dep_hint(self, csv_paths: tuple[Path, Path]) -> None:
        """When the catalog is empty (e.g. the deployment didn't
        install ``litellm``), the walker must surface the registrar's
        actionable ``missing_dep_hint`` verbatim rather than a generic
        "install the optional dependency" — the L0 walker doesn't know
        which dependency, but the L3 registrar does.

        Patching :func:`get_catalog_values` to ``frozenset()`` simulates
        a deployment without litellm; the registered hint
        (``elspeth[llm]``, set by ``model_catalog.py``) must appear in
        the finding reason so an operator has an actionable next step.
        """
        src, sink = csv_paths
        yaml_text = _build_yaml_with_model("openai/gpt-4o", src, sink)
        config = load_settings_from_yaml_string(yaml_text)
        with (
            patch(
                "elspeth.engine.orchestrator.preflight.get_catalog_values",
                return_value=frozenset(),
            ),
            pytest.raises(ValueSourceValidationError) as exc_info,
        ):
            instantiate_plugins_from_config(config)
        # Hint text from model_catalog.py registration must appear in
        # the finding's reason — quoted verbatim by the walker.
        finding = exc_info.value.findings[0]
        assert "elspeth[llm]" in finding.reason, f"expected the registered missing_dep_hint to surface; got: {finding.reason!r}"
