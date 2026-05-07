"""Tests for pipeline dependency resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import DependencyFailedError
from elspeth.core.dependency_config import DependencyConfig
from elspeth.engine.dependency_resolver import _hash_settings_file, _load_depends_on, detect_cycles, resolve_dependencies


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
    def test_no_cycle_returns_none(self, tmp_path: Path) -> None:
        # A -> B, no cycle
        b = tmp_path / "b.yaml"
        b.write_text("source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\nlandscape:\n  url: sqlite:///test.db\n")
        a = tmp_path / "a.yaml"
        a.write_text(
            f"depends_on:\n  - name: b\n    settings: {b}\nsource:\n  plugin: null_source\n"
            "sinks:\n  out:\n    plugin: json_sink\nlandscape:\n  url: sqlite:///test.db\n"
        )

        # Should not raise
        detect_cycles(a)

    def test_self_loop_detected(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        a.write_text(
            f"depends_on:\n  - name: self\n    settings: {a}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        with pytest.raises(ValueError, match=r"[Cc]ircular|[Cc]ycle"):
            detect_cycles(a)

    def test_two_hop_cycle_detected(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(
            f"depends_on:\n  - name: b\n    settings: {b}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )
        b.write_text(
            f"depends_on:\n  - name: a\n    settings: {a}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        with pytest.raises(ValueError, match=r"[Cc]ircular|[Cc]ycle"):
            detect_cycles(a)

    def test_three_hop_cycle_detected(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        c = tmp_path / "c.yaml"
        a.write_text(
            f"depends_on:\n  - name: b\n    settings: {b}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )
        b.write_text(
            f"depends_on:\n  - name: c\n    settings: {c}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )
        c.write_text(
            f"depends_on:\n  - name: a\n    settings: {a}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
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
            f"depends_on:\n  - name: d\n    settings: {files['d']}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )
        files["b"].write_text(
            f"depends_on:\n  - name: c\n    settings: {files['c']}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )
        files["a"].write_text(
            f"depends_on:\n  - name: b\n    settings: {files['b']}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
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
            f"depends_on:\n  - name: dep\n    settings: {link}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        # Should not raise — link resolves to real, no cycle
        detect_cycles(main)

    def test_diamond_dependency_no_cycle(self, tmp_path: Path) -> None:
        """Diamond shape: A -> B, A -> C, B -> D, C -> D. No cycle."""
        d = tmp_path / "d.yaml"
        d.write_text("source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")

        b = tmp_path / "b.yaml"
        b.write_text(
            f"depends_on:\n  - name: d\n    settings: {d}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        c = tmp_path / "c.yaml"
        c.write_text(
            f"depends_on:\n  - name: d\n    settings: {d}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        a = tmp_path / "a.yaml"
        a.write_text(
            f"depends_on:\n  - name: b\n    settings: {b}\n  - name: c\n    settings: {c}\n"
            "source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n"
        )

        # Should not raise — diamond is not a cycle
        detect_cycles(a)


class TestResolveDependencies:
    def test_single_dependency_success(self, tmp_path: Path) -> None:
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_result = MagicMock()
        mock_result.status = RunStatus.COMPLETED
        mock_result.run_id = "dep-run-123"

        mock_runner = MagicMock(return_value=mock_result)

        with patch("elspeth.engine.dependency_resolver._hash_settings_file", return_value="sha256:abc"):
            results = resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
                runner=mock_runner,
            )

        assert len(results) == 1
        assert results[0].name == "index"
        assert results[0].run_id == "dep-run-123"

    def test_dependency_failure_raises(self, tmp_path: Path) -> None:
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_result = MagicMock()
        mock_result.status = RunStatus.FAILED
        mock_result.run_id = "dep-run-fail"

        mock_runner = MagicMock(return_value=mock_result)

        with pytest.raises(DependencyFailedError, match="index"):
            resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
                runner=mock_runner,
            )

    def test_keyboard_interrupt_propagated(self, tmp_path: Path) -> None:
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = MagicMock(side_effect=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
                runner=mock_runner,
            )

    def test_runner_exception_wrapped_in_dependency_failed_error(self, tmp_path: Path) -> None:
        """Runner exceptions (other than KeyboardInterrupt) are wrapped in DependencyFailedError."""
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = MagicMock(side_effect=RuntimeError("config loading failed"))

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

        mock_runner = MagicMock(side_effect=FrameworkBugError("invariant violated"))

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

        mock_runner = MagicMock(side_effect=AuditIntegrityError("corrupt audit trail"))

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

        mock_runner = MagicMock(side_effect=TypeError("bad argument"))

        with pytest.raises(TypeError, match="bad argument"):
            resolve_dependencies(depends_on=[dep], parent_settings_path=parent_path, runner=mock_runner)

    def test_attribute_error_crashes_through(self, tmp_path: Path) -> None:
        """AttributeError from code bug must NOT be wrapped in DependencyFailedError."""
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_runner = MagicMock(side_effect=AttributeError("no such attr"))

        with pytest.raises(AttributeError, match="no such attr"):
            resolve_dependencies(depends_on=[dep], parent_settings_path=parent_path, runner=mock_runner)

    def test_multiple_dependencies_sequential(self, tmp_path: Path) -> None:
        deps = [
            DependencyConfig(name="first", settings="./first.yaml"),
            DependencyConfig(name="second", settings="./second.yaml"),
        ]
        parent_path = tmp_path / "main.yaml"
        call_order: list[str] = []

        def track_calls(path: Path) -> MagicMock:
            call_order.append(path.name)
            result = MagicMock()
            result.status = RunStatus.COMPLETED
            result.run_id = f"run-{path.name}"
            return result

        mock_runner = MagicMock(side_effect=track_calls)

        with patch("elspeth.engine.dependency_resolver._hash_settings_file", return_value="sha256:abc"):
            resolve_dependencies(depends_on=deps, parent_settings_path=parent_path, runner=mock_runner)

        assert call_order == ["first.yaml", "second.yaml"]

    def test_empty_depends_on_returns_empty(self, tmp_path: Path) -> None:
        parent_path = tmp_path / "main.yaml"
        mock_runner = MagicMock()
        results = resolve_dependencies(depends_on=[], parent_settings_path=parent_path, runner=mock_runner)
        assert results == []


class TestHashSettingsFile:
    """End-to-end tests for _hash_settings_file with real YAML."""

    def test_produces_sha256_prefixed_hash(self, tmp_path: Path) -> None:
        settings = tmp_path / "pipeline.yaml"
        settings.write_text("plugins:\n  source:\n    path: data.csv\n")
        result = _hash_settings_file(settings)
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64  # SHA-256 hex

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
