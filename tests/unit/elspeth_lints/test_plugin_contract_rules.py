"""Tests for plugin-contract elspeth-lints rules."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.plugin_contract.component_type import RULE as COMPONENT_TYPE_RULE
from elspeth_lints.rules.plugin_contract.plugin_hashes import RULE as PLUGIN_HASHES_RULE
from elspeth_lints.rules.plugin_contract.plugin_hashes.rule import (
    PluginHashesRule,
    compute_source_file_hash,
)
from elspeth_lints.rules.plugin_contract.plugin_hashes.rule import (
    scan_root as scan_plugin_hashes_root,
)


def test_component_type_reports_data_config_without_component_type() -> None:
    findings = list(
        COMPONENT_TYPE_RULE.analyze(
            _tree("""
            class BadConfig(DataPluginConfig):
                path: str
            """),
            Path("bad.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["CT1"]
    assert "BadConfig" in findings[0].message


def test_component_type_accepts_declared_type_and_known_typed_bases(tmp_path: Path) -> None:
    _write(
        tmp_path / "good.py",
        """
        from elspeth.plugins.infrastructure.config_base import DataPluginConfig, SourceDataConfig
        from typing import ClassVar

        class DeclaredConfig(DataPluginConfig):
            _plugin_component_type: ClassVar[str | None] = "transform"
            path: str

        class InheritedSourceConfig(SourceDataConfig):
            extra: str = "default"
        """,
    )

    assert _component_findings(tmp_path) == []


def test_component_type_exempt_base_does_not_exempt_child(tmp_path: Path) -> None:
    _write(
        tmp_path / "hierarchy.py",
        """
        from elspeth.plugins.infrastructure.config_base import DataPluginConfig
        from typing import ClassVar

        class MiddleConfig(DataPluginConfig):
            _component_type_exempt: ClassVar[bool] = True

        class ChildConfig(MiddleConfig):
            pass
        """,
    )

    findings = _component_findings(tmp_path)

    assert [finding.rule_id for finding in findings] == ["CT1"]
    assert "ChildConfig" in findings[0].message


def test_component_type_exemption_requires_literal_true(tmp_path: Path) -> None:
    _write(
        tmp_path / "not_exempt.py",
        """
        from elspeth.plugins.infrastructure.config_base import DataPluginConfig
        from typing import ClassVar

        class NotExemptConfig(DataPluginConfig):
            _component_type_exempt: ClassVar[bool] = False
        """,
    )

    findings = _component_findings(tmp_path)

    assert [finding.rule_id for finding in findings] == ["CT1"]
    assert "NotExemptConfig" in findings[0].message


def test_component_type_resolves_cross_file_base_with_declared_type(tmp_path: Path) -> None:
    _write(
        tmp_path / "base_config.py",
        """
        from elspeth.plugins.infrastructure.config_base import DataPluginConfig
        from typing import ClassVar

        class MyBaseConfig(DataPluginConfig):
            _plugin_component_type: ClassVar[str | None] = "sink"
        """,
    )
    _write(
        tmp_path / "leaf_config.py",
        """
        from base_config import MyBaseConfig

        class LeafConfig(MyBaseConfig):
            output_format: str = "json"
        """,
    )

    assert _component_findings(tmp_path) == []


def test_component_type_reports_relative_paths_and_multiple_classes(tmp_path: Path) -> None:
    _write(
        tmp_path / "subdir" / "multi.py",
        """
        from elspeth.plugins.infrastructure.config_base import DataPluginConfig

        class BadOne(DataPluginConfig):
            pass

        class BadTwo(DataPluginConfig):
            pass
        """,
    )

    findings = _component_findings(tmp_path)

    assert [finding.file_path for finding in findings] == ["subdir/multi.py", "subdir/multi.py"]
    assert {finding.message.split(" ", 1)[0] for finding in findings} == {"BadOne", "BadTwo"}


def test_component_type_skips_syntax_error_files(tmp_path: Path) -> None:
    _write(tmp_path / "bad_syntax.py", "def broken(:\n    pass\n")
    _write(
        tmp_path / "good.py",
        """
        from elspeth.plugins.infrastructure.config_base import DataPluginConfig
        from typing import ClassVar

        class GoodConfig(DataPluginConfig):
            _plugin_component_type: ClassVar[str | None] = "source"
        """,
    )

    assert _component_findings(tmp_path) == []


def test_component_type_uses_directory_allowlist(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        from elspeth.plugins.infrastructure.config_base import DataPluginConfig

        class BadConfig(DataPluginConfig):
            pass
        """,
    )
    allowlist = tmp_path / "config" / "cicd" / "enforce_component_type"
    allowlist.mkdir(parents=True)
    (allowlist / "rules.yaml").write_text(
        """
per_file_rules:
  - pattern: bad.py
    rules: [CT1]
    reason: synthetic fixture
    max_hits: 1
""",
        encoding="utf-8",
    )

    assert _component_findings(tmp_path) == []


def test_component_type_json_mode_succeeds_on_current_codebase(
    elspeth_lints_subprocess_env: dict[str, str],
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "elspeth_lints.core.cli",
            "check",
            "--rules",
            "plugin_contract.component_type",
            "--root",
            "src/elspeth",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
        env=elspeth_lints_subprocess_env,
    )

    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert json.loads(result.stdout) == []


def test_component_type_flags_alias_imported_base() -> None:
    # elspeth-20add2bd90: a base imported under an alias must still be resolved
    # back to its known config-base name so the descendant is checked.
    findings = list(
        COMPONENT_TYPE_RULE.analyze(
            _tree("""
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig as DPC

            class AliasMissing(DPC):
                path: str
            """),
            Path("bad.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["CT1"]
    assert "AliasMissing" in findings[0].message


def test_component_type_aliased_base_with_declared_type_is_clean() -> None:
    # The alias resolution must not over-flag: a properly declared type on an
    # aliased-base subclass stays clean.
    findings = list(
        COMPONENT_TYPE_RULE.analyze(
            _tree("""
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig as DPC

            class AliasDeclared(DPC):
                _plugin_component_type = "transform"
                path: str
            """),
            Path("good.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert findings == []


def test_component_type_flags_invalid_component_type_value() -> None:
    # elspeth-a2b240c29b: an out-of-contract string label must be flagged, not
    # silently accepted. The documented contract allows only source/sink/transform.
    findings = list(
        COMPONENT_TYPE_RULE.analyze(
            _tree("""
            from elspeth.plugins.infrastructure.config_base import DataPluginConfig

            class BadValue(DataPluginConfig):
                _plugin_component_type = "nonsense"
            """),
            Path("bad.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["CT1"]
    assert "BadValue" in findings[0].message


def test_component_type_accepts_each_valid_value() -> None:
    # The tightening must keep every legitimate label clean.
    for value in ("source", "sink", "transform"):
        findings = list(
            COMPONENT_TYPE_RULE.analyze(
                _tree(f"""
                from elspeth.plugins.infrastructure.config_base import DataPluginConfig

                class ValidValue(DataPluginConfig):
                    _plugin_component_type = "{value}"
                """),
                Path("good.py"),
                RuleContext(root=Path(".")),
            )
        )
        assert findings == [], f"value {value!r} should be accepted"


def test_plugin_hashes_reports_missing_source_file_hash(tmp_path: Path) -> None:
    _write(
        tmp_path / "plugins" / "sources" / "missing_hash.py",
        """
        class MissingHashSource:
            name = "missing-hash"
            plugin_version = "1.0.0"
        """,
    )

    findings = list(
        PLUGIN_HASHES_RULE.analyze(
            ast.Module(body=[], type_ignores=[]),
            tmp_path,
            RuleContext(root=tmp_path),
        )
    )

    assert [finding.rule_id for finding in findings] == ["PH2"]
    assert "source_file_hash" in findings[0].message


def test_plugin_hashes_reports_missing_hash_for_module_constant_name(tmp_path: Path) -> None:
    _write(
        tmp_path / "plugins" / "sources" / "dynamic_name.py",
        """
        PLUGIN_NAME = "dynamic"

        class DynamicSource:
            name = PLUGIN_NAME
            plugin_version = "1.0.0"
            source_file_hash = None
        """,
    )

    findings = scan_plugin_hashes_root(tmp_path)

    assert [finding.rule_id for finding in findings] == ["PH2"]
    assert "DynamicSource" in findings[0].message


def test_plugin_hashes_passes_on_correct_hashes(tmp_path: Path) -> None:
    _write_hashed_plugin(tmp_path, class_name="GoodSource", name="good", version="1.0.0")

    assert scan_plugin_hashes_root(tmp_path) == []


def test_plugin_hashes_reports_stale_hash(tmp_path: Path) -> None:
    plugin = _write_hashed_plugin(tmp_path, class_name="GoodSource", name="good", version="1.0.0")
    plugin.write_text(plugin.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    findings = scan_plugin_hashes_root(tmp_path)

    assert [finding.rule_id for finding in findings] == ["PH3"]
    assert "stale source_file_hash" in findings[0].message


def test_plugin_hashes_reports_missing_plugin_version(tmp_path: Path) -> None:
    _write(
        tmp_path / "plugins" / "sources" / "missing_version.py",
        """
        class MissingVersionSource:
            name = "missing-version"
            source_file_hash = "sha256:0000000000000000"
        """,
    )

    findings = scan_plugin_hashes_root(tmp_path)

    assert [finding.rule_id for finding in findings] == ["PH1", "PH3"]
    assert "no version declaration" in findings[0].message


def test_plugin_hashes_enforces_minimum_discovery_count(tmp_path: Path) -> None:
    _write_hashed_plugin(tmp_path, class_name="GoodSource", name="good", version="1.0.0")

    findings = list(
        PluginHashesRule(min_plugins=2).analyze(
            ast.Module(body=[], type_ignores=[]),
            tmp_path,
            RuleContext(root=tmp_path),
        )
    )

    assert [finding.rule_id for finding in findings] == ["plugin_contract.plugin_hashes"]
    assert "DISCOVERY ERROR" in findings[0].message


def test_plugin_hashes_ignores_excluded_helper_files(tmp_path: Path) -> None:
    _write(
        tmp_path / "plugins" / "sources" / "base.py",
        """
        class HelperBase:
            name = "helper"
        """,
    )

    assert scan_plugin_hashes_root(tmp_path) == []


def test_plugin_hashes_json_mode_succeeds_on_current_codebase(
    elspeth_lints_subprocess_env: dict[str, str],
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "elspeth_lints.core.cli",
            "check",
            "--rules",
            "plugin_contract.plugin_hashes",
            "--root",
            "src/elspeth",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
        env=elspeth_lints_subprocess_env,
    )

    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert json.loads(result.stdout) == []


def _tree(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


def _component_findings(root: Path):
    return list(
        COMPONENT_TYPE_RULE.analyze(
            ast.Module(body=[], type_ignores=[]),
            root,
            RuleContext(root=root),
        )
    )


def _write_hashed_plugin(tmp_path: Path, *, class_name: str, name: str, version: str) -> Path:
    plugin = tmp_path / "plugins" / "sources" / f"{name}_source.py"
    _write(
        plugin,
        f'''
        class {class_name}:
            name = "{name}"
            plugin_version = "{version}"
            source_file_hash = "sha256:0000000000000000"
        ''',
    )
    correct_hash = compute_source_file_hash(plugin)
    plugin.write_text(
        plugin.read_text(encoding="utf-8").replace("sha256:0000000000000000", correct_hash),
        encoding="utf-8",
    )
    return plugin
