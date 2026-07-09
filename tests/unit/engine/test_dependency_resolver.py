"""Tests for pipeline dependency resolution."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import DependencyFailedError
from elspeth.contracts.run_result import RunResult
from elspeth.core.dependency_config import DependencyConfig
from elspeth.engine.dependency_resolver import _hash_settings_file, _load_depends_on, detect_cycles, resolve_dependencies
from tests.fixtures.audit_hashing import assert_prefixed_canonical_sha256


def _run_result(run_id: str, status: RunStatus) -> RunResult:
    return RunResult(
        run_id=run_id,
        status=status,
        rows_processed=1,
        rows_succeeded=1 if status is RunStatus.COMPLETED else 0,
        rows_failed=0 if status is RunStatus.COMPLETED else 1,
        rows_routed_success=0,
        rows_routed_failure=0,
    )


class _RunnerDouble:
    def __init__(
        self,
        *,
        result: RunResult | None = None,
        error: BaseException | None = None,
        side_effect: Callable[[Path], RunResult] | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.side_effect = side_effect
        self.paths: list[Path] = []

    def __call__(self, path: Path) -> RunResult:
        self.paths.append(path)
        if self.error is not None:
            raise self.error
        if self.side_effect is not None:
            return self.side_effect(path)
        if self.result is None:
            raise AssertionError("RunnerDouble result was not configured")
        return self.result


class TestLoadDependsOnValidation:
    """Tests for Tier 3 validation in _load_depends_on (review finding #2)."""

    def test_non_mapping_document_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("[]\n")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            _load_depends_on(f)

    def test_non_list_depends_on_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("depends_on: not_a_list\n")
        with pytest.raises(ValueError, match="must be a list"):
            _load_depends_on(f)

    def test_non_dict_entry_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("depends_on:\n  - just_a_string\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            _load_depends_on(f)

    def test_missing_settings_key_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("depends_on:\n  - name: dep\n")
        with pytest.raises(ValueError, match="missing required key 'settings'"):
            _load_depends_on(f)

    def test_missing_name_key_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("depends_on:\n  - settings: ./dep.yaml\n")
        with pytest.raises(ValueError, match="missing required key 'name'"):
            _load_depends_on(f)

    def test_valid_entry_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "good.yaml"
        f.write_text("depends_on:\n  - name: dep\n    settings: ./dep.yaml\n")
        deps = _load_depends_on(f)
        assert len(deps) == 1
        assert deps[0]["name"] == "dep"

    def test_yaml_syntax_error_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("depends_on: [\n")
        with pytest.raises(ValueError, match="Invalid YAML"):
            _load_depends_on(f)

    def test_valid_entry_uses_dependency_contract_normalization(self, tmp_path: Path) -> None:
        f = tmp_path / "good.yaml"
        f.write_text('depends_on:\n  - name: " dep "\n    settings: " ./dep.yaml "\n')
        deps = _load_depends_on(f)
        assert deps == [{"name": "dep", "settings": "./dep.yaml"}]

    def test_absent_depends_on_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "no_deps.yaml"
        f.write_text("source:\n  plugin: null_source\n")
        assert _load_depends_on(f) == []


class TestCycleDetection:
    def test_absolute_dependency_path_rejected(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        outside = tmp_path / "outside.yaml"
        outside.write_text("source:\n  plugin: null_source\n", encoding="utf-8")
        main = project / "main.yaml"
        main.write_text(
            f"depends_on:\n  - name: outside\n    settings: {outside}\nsource:\n  plugin: null_source\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=r"Dependency settings path.*must be relative"):
            detect_cycles(main)

    def test_traversal_dependency_path_rejected(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        outside = tmp_path / "outside.yaml"
        outside.write_text("source:\n  plugin: null_source\n", encoding="utf-8")
        main = project / "main.yaml"
        main.write_text(
            "depends_on:\n  - name: outside\n    settings: ../outside.yaml\nsource:\n  plugin: null_source\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="escapes allowed root"):
            detect_cycles(main)

    def test_no_cycle_returns_none(self, tmp_path: Path) -> None:
        # A -> B, no cycle
        b = tmp_path / "b.yaml"
        b.write_text("source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\nlandscape:\n  url: sqlite:///test.db\n")
        a = tmp_path / "a.yaml"
        a.write_text(
            f"depends_on:\n  - name: b\n    settings: {b.name}\nsource:\n  plugin: null_source\n"
            "sinks:\n  out:\n    plugin: json_sink\nlandscape:\n  url: sqlite:///test.db\n"
        )

        # Should not raise
        detect_cycles(a)

    def test_self_loop_detected(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        a.write_text(
            f"depends_on:\n  - name: self\n    settings: {a.name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        with pytest.raises(ValueError, match=r"[Cc]ircular|[Cc]ycle"):
            detect_cycles(a)

    def test_two_hop_cycle_detected(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(
            f"depends_on:\n  - name: b\n    settings: {b.name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )
        b.write_text(
            f"depends_on:\n  - name: a\n    settings: {a.name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        with pytest.raises(ValueError, match=r"[Cc]ircular|[Cc]ycle"):
            detect_cycles(a)

    def test_three_hop_cycle_detected(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        c = tmp_path / "c.yaml"
        a.write_text(
            f"depends_on:\n  - name: b\n    settings: {b.name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )
        b.write_text(
            f"depends_on:\n  - name: c\n    settings: {c.name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )
        c.write_text(
            f"depends_on:\n  - name: a\n    settings: {a.name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        with pytest.raises(ValueError, match=r"[Cc]ircular|[Cc]ycle"):
            detect_cycles(a)

    def test_depth_limit_exceeded(self, tmp_path: Path) -> None:
        # Create a chain: a -> b -> c -> d (depth 4, exceeds limit of 3)
        files: dict[str, Path] = {}
        for name in ["d", "c", "b", "a"]:
            files[name] = tmp_path / f"{name}.yaml"

        files["d"].write_text("source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")
        files["c"].write_text(
            f"depends_on:\n  - name: d\n    settings: {files['d'].name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )
        files["b"].write_text(
            f"depends_on:\n  - name: c\n    settings: {files['c'].name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )
        files["a"].write_text(
            f"depends_on:\n  - name: b\n    settings: {files['b'].name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        with pytest.raises(ValueError, match=r"[Dd]epth"):
            detect_cycles(files["a"], max_depth=3)

    def test_uses_resolved_paths(self, tmp_path: Path) -> None:
        """Symlinks resolve to the same canonical path."""
        real = tmp_path / "real.yaml"
        real.write_text("source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")
        link = tmp_path / "link.yaml"
        link.symlink_to(real)

        main = tmp_path / "main.yaml"
        main.write_text(
            f"depends_on:\n  - name: dep\n    settings: {link.name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        # Should not raise — link resolves to real, no cycle
        detect_cycles(main)

    def test_diamond_dependency_no_cycle(self, tmp_path: Path) -> None:
        """Diamond shape: A -> B, A -> C, B -> D, C -> D. No cycle."""
        d = tmp_path / "d.yaml"
        d.write_text("source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")

        b = tmp_path / "b.yaml"
        b.write_text(
            f"depends_on:\n  - name: d\n    settings: {d.name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        c = tmp_path / "c.yaml"
        c.write_text(
            f"depends_on:\n  - name: d\n    settings: {d.name}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        a = tmp_path / "a.yaml"
        a.write_text(
            f"depends_on:\n  - name: b\n    settings: {b.name}\n  - name: c\n    settings: {c.name}\n"
            "source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        # Should not raise — diamond is not a cycle
        detect_cycles(a)


class TestResolveDependencies:
    def test_absolute_dependency_path_rejected_before_runner(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.yaml"
        dep = DependencyConfig(name="outside", settings=str(outside))
        parent_path = tmp_path / "project" / "query.yaml"
        parent_path.parent.mkdir()
        mock_runner = _RunnerDouble(result=_run_result("unused", RunStatus.COMPLETED))

        with (
            patch("elspeth.engine.dependency_resolver._hash_settings_file") as mock_hash,
            pytest.raises(ValueError, match=r"Dependency settings path.*must be relative"),
        ):
            resolve_dependencies(depends_on=[dep], parent_settings_path=parent_path, runner=mock_runner)

        mock_hash.assert_not_called()
        assert mock_runner.paths == []

    def test_traversal_dependency_path_rejected_before_runner(self, tmp_path: Path) -> None:
        dep = DependencyConfig(name="outside", settings="../outside.yaml")
        parent_path = tmp_path / "project" / "query.yaml"
        parent_path.parent.mkdir()
        mock_runner = _RunnerDouble(result=_run_result("unused", RunStatus.COMPLETED))

        with (
            patch("elspeth.engine.dependency_resolver._hash_settings_file") as mock_hash,
            pytest.raises(ValueError, match="escapes allowed root"),
        ):
            resolve_dependencies(depends_on=[dep], parent_settings_path=parent_path, runner=mock_runner)

        mock_hash.assert_not_called()
        assert mock_runner.paths == []

    def test_single_dependency_success(self, tmp_path: Path) -> None:
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = _RunnerDouble(result=_run_result("dep-run-123", RunStatus.COMPLETED))

        with patch("elspeth.engine.dependency_resolver._hash_settings_file", return_value="sha256:abc"):
            results = resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
                runner=mock_runner,
            )

        assert len(results) == 1
        assert results[0].name == "index"
        assert results[0].run_id == "dep-run-123"

    def test_dependency_result_hashes_settings_before_runner_side_effects(self, tmp_path: Path) -> None:
        """Dependency audit hash must describe the config handed to the runner."""
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"
        dep_path = tmp_path / "index.yaml"
        dep_path.write_text("source:\n  plugin: before\n", encoding="utf-8")
        before_hash = _hash_settings_file(dep_path)

        def mutate_after_execution(settings_path: Path) -> RunResult:
            assert settings_path == dep_path
            settings_path.write_text("source:\n  plugin: after\n", encoding="utf-8")
            return RunResult(
                run_id="dep-run-123",
                status=RunStatus.COMPLETED,
                rows_processed=1,
                rows_succeeded=1,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
            )

        results = resolve_dependencies(
            depends_on=[dep],
            parent_settings_path=parent_path,
            runner=mutate_after_execution,
        )

        assert _hash_settings_file(dep_path) != before_hash
        assert results[0].settings_hash == before_hash

    def test_dependency_failure_raises(self, tmp_path: Path) -> None:
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = _RunnerDouble(result=_run_result("dep-run-fail", RunStatus.FAILED))

        with pytest.raises(DependencyFailedError, match="index"):
            resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
                runner=mock_runner,
            )

    def test_keyboard_interrupt_propagated(self, tmp_path: Path) -> None:
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = _RunnerDouble(error=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
                runner=mock_runner,
            )

    def test_graceful_shutdown_propagates_unwrapped(self, tmp_path: Path) -> None:
        from elspeth.contracts.errors import GracefulShutdownError

        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"
        shutdown = GracefulShutdownError(rows_processed=0, run_id="run-1")
        mock_runner = _RunnerDouble(error=shutdown)

        with pytest.raises(GracefulShutdownError):
            resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
                runner=mock_runner,
            )

    def test_runner_exception_wrapped_in_dependency_failed_error(self, tmp_path: Path) -> None:
        """Runner exceptions (other than KeyboardInterrupt) are wrapped in DependencyFailedError."""
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = _RunnerDouble(error=RuntimeError("config loading failed"))

        with pytest.raises(DependencyFailedError, match="RuntimeError") as exc_info:
            resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
                runner=mock_runner,
            )

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_framework_bug_error_propagates_unwrapped(self, tmp_path: Path) -> None:
        """FrameworkBugError must NOT be wrapped in DependencyFailedError."""
        from elspeth.contracts.errors import FrameworkBugError

        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = _RunnerDouble(error=FrameworkBugError("invariant violated"))

        with pytest.raises(FrameworkBugError, match="invariant violated"):
            resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
                runner=mock_runner,
            )

    def test_audit_integrity_error_propagates_unwrapped(self, tmp_path: Path) -> None:
        """AuditIntegrityError must NOT be wrapped in DependencyFailedError."""
        from elspeth.contracts.errors import AuditIntegrityError

        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = _RunnerDouble(error=AuditIntegrityError("corrupt audit trail"))

        with pytest.raises(AuditIntegrityError, match="corrupt audit trail"):
            resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
                runner=mock_runner,
            )

    def test_type_error_crashes_through(self, tmp_path: Path) -> None:
        """TypeError from code bug must NOT be wrapped in DependencyFailedError."""
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = _RunnerDouble(error=TypeError("bad argument"))

        with pytest.raises(TypeError, match="bad argument"):
            resolve_dependencies(depends_on=[dep], parent_settings_path=parent_path, runner=mock_runner)

    def test_attribute_error_crashes_through(self, tmp_path: Path) -> None:
        """AttributeError from code bug must NOT be wrapped in DependencyFailedError."""
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = _RunnerDouble(error=AttributeError("no such attr"))

        with pytest.raises(AttributeError, match="no such attr"):
            resolve_dependencies(depends_on=[dep], parent_settings_path=parent_path, runner=mock_runner)

    def test_runner_boundary_catches_exception_not_base_exception(self) -> None:
        """Crash-through policy lives in the helper; this wrapper only catches Exceptions."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(resolve_dependencies)))
        caught_names = {
            handler.type.id for handler in ast.walk(tree) if isinstance(handler, ast.ExceptHandler) and isinstance(handler.type, ast.Name)
        }

        assert "Exception" in caught_names
        assert "BaseException" not in caught_names

    def test_multiple_dependencies_sequential(self, tmp_path: Path) -> None:
        deps = [
            DependencyConfig(name="first", settings="./first.yaml"),
            DependencyConfig(name="second", settings="./second.yaml"),
        ]
        parent_path = tmp_path / "main.yaml"
        call_order: list[str] = []

        def track_calls(path: Path) -> RunResult:
            call_order.append(path.name)
            return _run_result(f"run-{path.name}", RunStatus.COMPLETED)

        mock_runner = _RunnerDouble(side_effect=track_calls)

        with patch("elspeth.engine.dependency_resolver._hash_settings_file", return_value="sha256:abc"):
            resolve_dependencies(depends_on=deps, parent_settings_path=parent_path, runner=mock_runner)

        assert call_order == ["first.yaml", "second.yaml"]

    def test_empty_depends_on_returns_empty(self, tmp_path: Path) -> None:
        parent_path = tmp_path / "main.yaml"
        mock_runner = _RunnerDouble(result=_run_result("unused", RunStatus.COMPLETED))
        results = resolve_dependencies(depends_on=[], parent_settings_path=parent_path, runner=mock_runner)
        assert results == []


class TestHashSettingsFile:
    """End-to-end tests for _hash_settings_file with real YAML."""

    def test_hash_binds_to_canonical_yaml_payload(self, tmp_path: Path) -> None:
        settings = tmp_path / "pipeline.yaml"
        settings.write_text("plugins:\n  source:\n    path: data.csv\n")
        result = _hash_settings_file(settings)
        # _hash_settings_file is a byte-integrity hash over the canonical JSON of
        # the dependency file AS WRITTEN — it does NOT load/normalise the config
        # model, so the singular `source:` key is hashed verbatim. Normalising
        # singular→plural before hashing would make the audit trail assert the
        # file contained `sources: {primary}` when it literally said `source:`
        # (fabrication). The expected payload must mirror the raw YAML.
        expected_payload = {
            "plugins": {
                "source": {
                    "path": "data.csv",
                },
            },
        }
        assert_prefixed_canonical_sha256(result, expected_payload)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        content = "depends_on:\n  - name: index\n    settings: ./index.yaml\n"
        a.write_text(content)
        b.write_text(content)
        assert _hash_settings_file(a) == _hash_settings_file(b)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("key: value1\n")
        b.write_text("key: value2\n")
        assert _hash_settings_file(a) != _hash_settings_file(b)

    def test_key_order_independent(self, tmp_path: Path) -> None:
        """Canonical JSON normalises key order — reordered YAML produces same hash."""
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("alpha: 1\nbeta: 2\n")
        b.write_text("beta: 2\nalpha: 1\n")
        assert _hash_settings_file(a) == _hash_settings_file(b)
