# Web UX Task-Plan 5B: Dry-Run Validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement dry-run pipeline validation using real engine code (ExecutionGraph, graph.validate)
**Parent Plan:** `plans/2026-03-28-web-ux-sub5-execution.md`
**Spec:** `specs/2026-03-28-web-ux-sub5-execution-design.md`
**Depends On:** Task-Plan 5A (Models -- ValidationResult schema), Sub-Plan 4 (Composer -- CompositionState, generate_yaml)
**Blocks:** Task-Plan 5D (Routes & Integration)
**Can run in parallel with:** Task-Plan 5C (ExecutionServiceImpl)

---

## File Map

| Action | Path |
|--------|------|
| Create | `src/elspeth/web/execution/validation.py` |
| Create | `tests/unit/web/execution/test_validation.py` |

---

## Task 5.4: Dry-Run Validation

Dry-run validation calls the real engine code path. No parallel validation logic. Only typed exceptions are caught (W18 fix).

- [ ] **Step 1: Write validation tests**

```python
# tests/unit/web/execution/test_validation.py
"""Tests for dry-run validation using real engine code paths.

Validation calls the actual engine functions: load_settings(),
instantiate_plugins_from_config(), ExecutionGraph.from_plugin_instances(),
graph.validate(). No parallel validation logic exists.

W18 fix: Only typed exceptions are caught — no bare except Exception.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

from elspeth.core.dag.models import GraphValidationError
from elspeth.web.execution.schemas import ValidationResult
from elspeth.web.execution.validation import validate_pipeline


class FakeCompositionState:
    """Minimal stand-in for CompositionState during validation tests.

    B-5B-3 fix: Uses state.source (a dict with "options" key) instead of
    state.source_options, matching Sub-4's CompositionStateRecord structure.
    """

    def __init__(self, yaml_content: str = "", source_options: dict | None = None) -> None:
        self.yaml_content = yaml_content
        # source is the full source dict; options is nested within it
        self.source: dict | None = {"options": source_options} if source_options else None


class FakeWebSettings:
    """Minimal stand-in for WebSettings during validation tests."""

    def __init__(self, data_dir: str = "/tmp/test_data") -> None:
        self.data_dir = data_dir


class TestValidatePipelinePathAllowlist:
    """C3/S2: Source path allowlist check — defense-in-depth."""

    def test_path_within_uploads_passes(self) -> None:
        state = FakeCompositionState(
            source_options={"path": "/tmp/test_data/uploads/data.csv"},
        )
        settings = FakeWebSettings(data_dir="/tmp/test_data")
        # Path check passes — validation continues to settings_load
        # (which will fail because no real engine, but the path check itself passes)
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings") as mock_load:
            mock_load.side_effect = FileNotFoundError("no temp file")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        # B11: path check is always recorded — verify it passed
        path_check = next(c for c in result.checks if c.name == "source_path_allowlist")
        assert path_check.passed is True

    def test_path_outside_uploads_blocked(self) -> None:
        state = FakeCompositionState(
            source_options={"path": "/etc/passwd"},
        )
        settings = FakeWebSettings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False
        assert result.checks[0].name == "source_path_allowlist"
        assert result.checks[0].passed is False
        assert any("Path traversal" in e.message for e in result.errors)

    def test_path_traversal_via_dotdot_blocked(self) -> None:
        state = FakeCompositionState(
            source_options={"path": "/tmp/test_data/uploads/../../secret.csv"},
        )
        settings = FakeWebSettings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        result = validate_pipeline(state, settings, mock_yaml_gen)
        assert result.is_valid is False

    def test_no_path_option_records_skipped_check(self) -> None:
        """B11 fix: path allowlist check is always recorded, even when skipped."""
        state = FakeCompositionState(source_options={})
        settings = FakeWebSettings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        with patch("elspeth.web.execution.validation.load_settings") as mock_load:
            mock_load.side_effect = FileNotFoundError("no temp file")
            result = validate_pipeline(state, settings, mock_yaml_gen)
        # B11: check IS recorded with passed=True and "skipped" detail
        path_check = next(c for c in result.checks if c.name == "source_path_allowlist")
        assert path_check.passed is True
        assert "skipped" in path_check.detail.lower()


class TestValidatePipelineSuccess:
    @patch("elspeth.web.execution.validation.load_settings")
    @patch("elspeth.web.execution.validation.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.validation.ExecutionGraph")
    def test_valid_pipeline_returns_all_checks_passed(
        self,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
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
        mock_graph_cls.from_plugin_instances.return_value = mock_graph

        state = FakeCompositionState()
        settings = FakeWebSettings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is True
        assert len(result.checks) == 5
        assert all(c.passed for c in result.checks)
        # B11 fix: source_path_allowlist check is always recorded
        assert result.checks[0].name == "source_path_allowlist"
        assert result.checks[0].passed is True
        assert result.errors == []

        # Verify real engine functions were called
        mock_load.assert_called_once()
        mock_instantiate.assert_called_once_with(mock_settings)
        mock_graph_cls.from_plugin_instances.assert_called_once()
        mock_graph.validate.assert_called_once()
        mock_graph.validate_edge_compatibility.assert_called_once()


class TestValidatePipelineSettingsFailure:
    @patch("elspeth.web.execution.validation.load_settings")
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
                    "msg": "Field required",
                    "input": {},
                }
            ],
        )

        state = FakeCompositionState()
        settings = FakeWebSettings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        # B11: index 0 is source_path_allowlist (passed), index 1 is settings_load
        assert result.checks[0].name == "source_path_allowlist"
        assert result.checks[0].passed is True
        assert result.checks[1].name == "settings_load"
        assert result.checks[1].passed is False
        # Downstream checks are skipped but recorded
        assert all(not c.passed for c in result.checks[2:])
        assert any("Skipped" in c.detail for c in result.checks[2:])
        assert len(result.errors) >= 1

    @patch("elspeth.web.execution.validation.load_settings")
    def test_file_not_found_error_from_settings(
        self,
        mock_load: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source: {}"
        mock_load.side_effect = FileNotFoundError("temp file missing")

        state = FakeCompositionState()
        settings = FakeWebSettings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        # B11: index 1 is settings_load (index 0 is source_path_allowlist)
        assert result.checks[1].passed is False


class TestValidatePipelinePluginFailure:
    @patch("elspeth.web.execution.validation.load_settings")
    @patch("elspeth.web.execution.validation.instantiate_plugins_from_config")
    def test_unknown_plugin_returns_attributed_error(
        self,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: unknown"
        mock_load.return_value = MagicMock()
        mock_instantiate.side_effect = ValueError(
            "Unknown source plugin: 'unknown'"
        )

        state = FakeCompositionState()
        settings = FakeWebSettings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        # B11: index 0=path_allowlist, 1=settings_load, 2=plugin_instantiation
        assert result.checks[1].passed is True  # settings_load passed
        assert result.checks[2].passed is False  # plugin_instantiation failed
        assert any("unknown" in e.message.lower() for e in result.errors)


class TestValidatePipelineGraphFailure:
    @patch("elspeth.web.execution.validation.load_settings")
    @patch("elspeth.web.execution.validation.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.validation.ExecutionGraph")
    def test_graph_validation_error_attributed_to_node(
        self,
        mock_graph_cls: MagicMock,
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
        mock_graph_cls.from_plugin_instances.return_value = mock_graph
        mock_graph.validate.side_effect = GraphValidationError(
            "Route destination 'nonexistent' in gate_1 not found"
        )

        state = FakeCompositionState()
        settings = FakeWebSettings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        # B11: index 0=path_allowlist, 1=settings_load, 2=plugins, 3=graph_structure
        assert result.checks[3].passed is False  # graph_structure failed
        assert len(result.errors) >= 1

    @patch("elspeth.web.execution.validation.load_settings")
    @patch("elspeth.web.execution.validation.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.validation.ExecutionGraph")
    def test_edge_compatibility_error(
        self,
        mock_graph_cls: MagicMock,
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
        mock_graph_cls.from_plugin_instances.return_value = mock_graph
        mock_graph.validate.return_value = None  # structural check passes
        mock_graph.validate_edge_compatibility.side_effect = GraphValidationError(
            "Schema mismatch on edge transform_1 -> sink_primary"
        )

        state = FakeCompositionState()
        settings = FakeWebSettings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        # B11: index 0=path_allowlist, 1=settings, 2=plugins, 3=graph, 4=schema
        assert result.checks[3].passed is True  # graph_structure passed
        assert result.checks[4].passed is False  # schema_compatibility failed


class TestValidatePipelineNoBareCatch:
    """W18 fix: unexpected exceptions propagate — no bare except Exception."""

    @patch("elspeth.web.execution.validation.load_settings")
    def test_unexpected_exception_propagates(
        self,
        mock_load: MagicMock,
    ) -> None:
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = "source:\n  plugin: csv_source"
        mock_load.side_effect = RuntimeError("Unexpected engine bug")

        state = FakeCompositionState()

        settings = FakeWebSettings()
        # RuntimeError is NOT in the typed exception list — it must propagate
        with pytest.raises(RuntimeError, match="Unexpected engine bug"):
            validate_pipeline(state, settings, mock_yaml_gen)


class TestValidatePipelineTempFileCleanup:
    """Verify temp file is created and cleaned up in finally block."""

    @patch("elspeth.web.execution.validation.load_settings")
    @patch("elspeth.web.execution.validation.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.validation.ExecutionGraph")
    def test_temp_file_cleaned_up_on_success(
        self,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        tmp_path: Path,
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
        mock_graph_cls.from_plugin_instances.return_value = mock_graph

        state = FakeCompositionState()
        settings = FakeWebSettings()
        result = validate_pipeline(state, settings, mock_yaml_gen)

        # load_settings was called with a Path, not YAML content
        call_args = mock_load.call_args
        arg = call_args[0][0] if call_args[0] else call_args[1].get("config_path")
        assert isinstance(arg, Path)

        # The temp file should have been cleaned up
        assert not arg.exists()
```

- [ ] **Step 2: Implement validate_pipeline()**

```python
# src/elspeth/web/execution/validation.py
"""Dry-run validation using real engine code paths.

Calls the same functions as `elspeth run`: load_settings(),
instantiate_plugins_from_config(), ExecutionGraph.from_plugin_instances(),
graph.validate(), graph.validate_edge_compatibility().

W18 fix: Only typed exceptions are caught. Bare except Exception is forbidden.
Unknown exception types propagate as 500 Internal Server Error, signalling
that this function needs updating — not that the error should be swallowed.

Temp file pattern: load_settings() takes a file path, NOT yaml content.
YAML is written to a NamedTemporaryFile, the path is passed to load_settings(),
and the file is deleted in a finally block.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError as PydanticValidationError

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import load_settings
from elspeth.core.dag.graph import ExecutionGraph
from elspeth.core.dag.models import GraphValidationError
from elspeth.web.execution.schemas import (
    ValidationCheck,
    ValidationError,
    ValidationResult,
)

if TYPE_CHECKING:
    from elspeth.web.composer.yaml_generator import YamlGenerator


# ── Check names (ordered) ─────────────────────────────────────────────
_CHECK_PATH_ALLOWLIST = "source_path_allowlist"
_CHECK_SETTINGS = "settings_load"
_CHECK_PLUGINS = "plugin_instantiation"
_CHECK_GRAPH = "graph_structure"
_CHECK_SCHEMA = "schema_compatibility"

_ALL_CHECKS = [_CHECK_PATH_ALLOWLIST, _CHECK_SETTINGS, _CHECK_PLUGINS, _CHECK_GRAPH, _CHECK_SCHEMA]


def _skipped_checks(from_check: str) -> list[ValidationCheck]:
    """Generate skipped check records for all checks after from_check."""
    skipping = False
    result: list[ValidationCheck] = []
    for name in _ALL_CHECKS:
        if name == from_check:
            skipping = True
            continue
        if skipping:
            result.append(
                ValidationCheck(
                    name=name,
                    passed=False,
                    detail=f"Skipped: {from_check} failed",
                )
            )
    return result


def _extract_component_id(message: str) -> tuple[str | None, str | None]:
    """Best-effort extraction of component_id and type from error message.

    Parses node IDs like 'gate_1', 'transform_2', 'sink_primary' from
    engine error messages. Returns (component_id, component_type) or
    (None, None) for structural errors.
    """
    # NOTE: This relies on error message string format from the engine.
    # Long-term, the engine should raise structured exceptions with
    # component_id as a field, not embedded in message strings. For MVP
    # this is acceptable — attribution degrades to (None, None), not failure.
    # Common patterns: "in gate_1", "node gate_1", "'gate_1'"
    patterns = [
        r"(?:in |node |'|\")((?:gate|transform|sink|source|aggregation)_\w+)",
        r"((?:gate|transform|sink|source|aggregation)_\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            node_id = match.group(1)
            # Extract type from prefix
            for prefix in ("gate", "transform", "sink", "source", "aggregation"):
                if node_id.startswith(prefix):
                    return node_id, prefix
    return None, None


def validate_pipeline(
    state: Any, settings: Any, yaml_generator: YamlGenerator
) -> ValidationResult:
    """Dry-run validation through the real engine code path.

    Steps:
    1. Source path allowlist check (C3/S2 defense-in-depth)
    2. Generate YAML from CompositionState
    3. Write to temp file, load_settings(path) — NOT yaml content
    4. instantiate_plugins_from_config(settings)
    5. ExecutionGraph.from_plugin_instances(bundle fields)
    6. graph.validate() + graph.validate_edge_compatibility()

    Only catches: PydanticValidationError, FileNotFoundError, ValueError,
    GraphValidationError. All other exceptions propagate (W18).

    Args:
        state: CompositionState from the session.
        settings: WebSettings — used for path allowlist check.
        yaml_generator: YamlGenerator instance for converting state to YAML.
    """
    checks: list[ValidationCheck] = []
    errors: list[ValidationError] = []
    tmp_path: Path | None = None

    # Step 1: Source path allowlist check (C3/S2 defense-in-depth)
    # Any `path` or `file` key in source options must resolve under
    # {settings.data_dir}/uploads/. This duplicates the composer tool
    # check as defense-in-depth.
    uploads_dir = Path(settings.data_dir) / "uploads"
    # B-5B-3 fix: Access state.source directly — this is Tier 1 data from our DB.
    # The source dict structure is defined by Sub-4's SourceSpec.to_dict(), and the
    # "options" key may legitimately be absent in the dict, so .get() is appropriate
    # on the dict contents (not on the state object itself).
    # TODO(sub-4): verify field path is state.source["options"] or state.source.options
    source_options = state.source.get("options", {}) if state.source else {}
    path_checked = False
    for key in ("path", "file"):
        value = source_options.get(key)
        if value is not None:
            path_checked = True
            resolved = Path(value).resolve()
            if not resolved.is_relative_to(uploads_dir.resolve()):
                return ValidationResult(
                    is_valid=False,
                    checks=[
                        ValidationCheck(
                            name="source_path_allowlist",
                            passed=False,
                            detail=f"Source {key} '{value}' is outside allowed "
                            f"upload directory: {uploads_dir}",
                        ),
                        *_skipped_checks("source_path_allowlist"),
                    ],
                    errors=[
                        ValidationError(
                            component_id="source",
                            component_type="source",
                            message=f"Path traversal blocked: {key}='{value}' "
                            f"resolves outside {uploads_dir}",
                            suggestion="Use a file within the uploads directory.",
                        ),
                    ],
                )
    # B11 fix: Always record the source_path_allowlist check
    if path_checked:
        checks.append(
            ValidationCheck(
                name=_CHECK_PATH_ALLOWLIST,
                passed=True,
                detail="Source path within allowed upload directory",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name=_CHECK_PATH_ALLOWLIST,
                passed=True,
                detail="No path option — check skipped",
            )
        )

    # Step 2: Generate YAML
    pipeline_yaml = yaml_generator.generate_yaml(state)

    # Step 2: Settings loading
    try:
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        tmp_path = Path(tmp_file.name)
        tmp_file.write(pipeline_yaml)
        tmp_file.close()

        elspeth_settings = load_settings(tmp_path)
        checks.append(
            ValidationCheck(
                name=_CHECK_SETTINGS,
                passed=True,
                detail="Settings loaded successfully",
            )
        )
    except (PydanticValidationError, FileNotFoundError, ValueError) as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_SETTINGS,
                passed=False,
                detail=str(exc),
            )
        )
        errors.append(
            ValidationError(
                component_id=None,
                component_type=None,
                message=str(exc),
                suggestion=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_SETTINGS))
        return ValidationResult(is_valid=False, checks=checks, errors=errors)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    # Step 3: Plugin instantiation
    try:
        bundle = instantiate_plugins_from_config(elspeth_settings)
        checks.append(
            ValidationCheck(
                name=_CHECK_PLUGINS,
                passed=True,
                detail="All plugins instantiated",
            )
        )
    except ValueError as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_PLUGINS,
                passed=False,
                detail=str(exc),
            )
        )
        comp_id, comp_type = _extract_component_id(str(exc))
        errors.append(
            ValidationError(
                component_id=comp_id,
                component_type=comp_type,
                message=str(exc),
                suggestion=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_PLUGINS))
        return ValidationResult(is_valid=False, checks=checks, errors=errors)

    # Step 4: Graph construction + structural validation
    try:
        graph = ExecutionGraph.from_plugin_instances(
            source=bundle.source,
            source_settings=bundle.source_settings,
            transforms=bundle.transforms,
            sinks=bundle.sinks,
            aggregations=bundle.aggregations,
            gates=list(elspeth_settings.gates),
            coalesce_settings=(
                list(elspeth_settings.coalesce) if elspeth_settings.coalesce else None
            ),
        )
        graph.validate()
        checks.append(
            ValidationCheck(
                name=_CHECK_GRAPH,
                passed=True,
                detail="Graph structure is valid",
            )
        )
    except GraphValidationError as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_GRAPH,
                passed=False,
                detail=str(exc),
            )
        )
        comp_id, comp_type = _extract_component_id(str(exc))
        errors.append(
            ValidationError(
                component_id=comp_id,
                component_type=comp_type,
                message=str(exc),
                suggestion=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_GRAPH))
        return ValidationResult(is_valid=False, checks=checks, errors=errors)

    # Step 5: Schema compatibility
    try:
        graph.validate_edge_compatibility()
        checks.append(
            ValidationCheck(
                name=_CHECK_SCHEMA,
                passed=True,
                detail="All edge schemas compatible",
            )
        )
    except GraphValidationError as exc:
        checks.append(
            ValidationCheck(
                name=_CHECK_SCHEMA,
                passed=False,
                detail=str(exc),
            )
        )
        comp_id, comp_type = _extract_component_id(str(exc))
        errors.append(
            ValidationError(
                component_id=comp_id,
                component_type=comp_type,
                message=str(exc),
                suggestion=None,
            )
        )
        return ValidationResult(is_valid=False, checks=checks, errors=errors)

    return ValidationResult(is_valid=True, checks=checks, errors=errors)
```

- [ ] **Step 3: Run tests, commit**

```bash
.venv/bin/python -m pytest tests/unit/web/execution/test_validation.py -v
git commit -m "feat(web/execution): add dry-run validation using real engine code paths"
```

---

## Self-Review Checklist

| # | Check | Task |
|---|-------|------|
| 1 | `validate_pipeline()` calls real engine functions (load_settings, instantiate_plugins_from_config, ExecutionGraph.from_plugin_instances, graph.validate, graph.validate_edge_compatibility) | 5.4 |
| 2 | Only typed exceptions caught: PydanticValidationError, FileNotFoundError, ValueError, GraphValidationError -- no bare `except Exception` (W18) | 5.4 |
| 3 | Per-component error attribution via `_extract_component_id()` | 5.4 |
| 4 | Temp file written for `load_settings(path)`, cleaned up in `finally` block | 5.4 |
| 5 | Source path allowlist check (C3/S2 defense-in-depth) blocks path traversal | 5.4 |
| 6 | Failed checks short-circuit with skipped downstream checks recorded | 5.4 |
| 7 | Unexpected exceptions (e.g. RuntimeError) propagate -- not swallowed | 5.4 |
| 8 | `validate_pipeline` is sync (called via `run_in_executor` from route handler in 5D) | 5.4 |
| 9 | All test classes cover: success, settings failure, plugin failure, graph failure, edge compatibility failure, no-bare-catch, temp file cleanup, path allowlist | 5.4 |

---

## Round 5 Review Findings

**Blocking issues (fixed inline above):**

- **B11: Check count assertion mismatch.** `test_valid_pipeline_returns_all_checks_passed` asserted `len(result.checks) == 4` but the implementation has 5 checks (`source_path_allowlist` + 4 engine checks). The spec says "each check is recorded in checks regardless of whether it was reached." Fixed: assertion changed to `== 5`, implementation updated to always record the `source_path_allowlist` check (with `passed=True` and either a confirmation detail or `"No path option -- check skipped"` for the no-path case). All downstream test index references updated (+1 offset for the new check at index 0).

- **B-5B-2: `settings` variable shadowed.** `validate_pipeline` did `settings = load_settings(tmp_path)`, shadowing the `settings: WebSettings` parameter. Fixed: renamed to `elspeth_settings = load_settings(tmp_path)` and updated all subsequent references (`instantiate_plugins_from_config(elspeth_settings)`, `elspeth_settings.gates`, `elspeth_settings.coalesce`).

- **B-5B-3: `getattr(state, "source_options", None)` violates CLAUDE.md.** Replaced with direct attribute access: `state.source.get("options", {}) if state.source else {}`. The `state.source` is Tier 1 data from our DB, but the source dict structure is defined by Sub-4's `SourceSpec.to_dict()`, and the `options` key may legitimately be absent in the dict -- so `.get()` on the dict contents is appropriate here. Added explanatory comment and `TODO(sub-4)` marker. `FakeCompositionState` updated to use `self.source` (a dict with nested `"options"` key) instead of `self.source_options`.

**Warnings:**
- **W-5B-1: `source_options` attribute assumed on state.** The validation function accesses source options but the exact field path depends on Sub-4's `CompositionState` structure. When Sub-4 lands, verify the path is `state.source["options"]` (or `state.source.options` if it's a `SourceSpec` object). Mark with `# TODO(sub-4): verify field path`.
