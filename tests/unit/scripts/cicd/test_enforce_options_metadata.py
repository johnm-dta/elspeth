"""Tests for the plugin configuration metadata CI lint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_enforce_metadata_succeeds_on_current_plugins() -> None:
    """Every current plugin config field has title and description metadata."""
    result = subprocess.run(
        [sys.executable, "scripts/cicd/enforce_options_metadata.py"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[4],
    )
    assert result.returncode == 0, f"Lint failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_enforce_metadata_fails_on_missing_title() -> None:
    """A config field without title is reported with a stable identifier."""
    from scripts.cicd.enforce_options_metadata import run_metadata_lint

    failures = run_metadata_lint(plugin_manager=_fake_manager_with_missing_title(), allowlist=set())

    assert "source/metadata_gap:missing_title: missing title" in failures


def test_enforce_metadata_fails_on_missing_description() -> None:
    """A config field without description is reported with a stable identifier."""
    from scripts.cicd.enforce_options_metadata import run_metadata_lint

    failures = run_metadata_lint(plugin_manager=_fake_manager_with_missing_description(), allowlist=set())

    assert "sink/metadata_gap:missing_description: missing description" in failures


def test_enforce_metadata_checks_discriminated_variants() -> None:
    """Provider-specific variant models are checked, not only config_model."""
    from scripts.cicd.enforce_options_metadata import run_metadata_lint

    failures = run_metadata_lint(plugin_manager=_fake_manager_with_broken_variant(), allowlist=set())

    assert "transform/metadata_variant[alpha]:variant_field: missing title" in failures


def test_enforce_metadata_allowlist_suppresses_matching_identifier() -> None:
    """An exact allowlist identifier suppresses its matching metadata finding."""
    from scripts.cicd.enforce_options_metadata import run_metadata_lint

    failures = run_metadata_lint(
        plugin_manager=_fake_manager_with_missing_title(),
        allowlist={"source/metadata_gap:missing_title"},
    )

    assert failures == []


def test_allowlist_requires_reason(tmp_path: Path) -> None:
    """Allowlist entries need an audit reason, not just an identifier."""
    from scripts.cicd.enforce_options_metadata import load_allowlist

    allowlist = tmp_path / "allowlist.yaml"
    allowlist.write_text(
        'entries:\n  - id: "source/metadata_gap:missing_title"\n    reason: ""\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reason"):
        load_allowlist(allowlist)


def _fake_manager_with_missing_title() -> object:
    from pydantic import BaseModel, Field

    class _Options(BaseModel):
        missing_title: str = Field(description="Has description but no title")

    class _Source:
        name = "metadata_gap"
        config_model = _Options

    return _FakePluginManager(sources=[_Source])


def _fake_manager_with_missing_description() -> object:
    from pydantic import BaseModel, Field

    class _Options(BaseModel):
        missing_description: str = Field(title="Missing description")

    class _Sink:
        name = "metadata_gap"
        config_model = _Options

    return _FakePluginManager(sinks=[_Sink])


def _fake_manager_with_broken_variant() -> object:
    from pydantic import BaseModel, Field

    class _AlphaOptions(BaseModel):
        variant_field: str = Field(description="Has description but no title")

    class _Transform:
        name = "metadata_variant"
        config_model = None

        @classmethod
        def discriminated_variants(cls) -> tuple[str, dict[str, type[BaseModel]]]:
            return "provider", {"alpha": _AlphaOptions}

    return _FakePluginManager(transforms=[_Transform])


class _FakePluginManager:
    def __init__(self, *, sources: list[type] | None = None, transforms: list[type] | None = None, sinks: list[type] | None = None) -> None:
        self._sources = sources or []
        self._transforms = transforms or []
        self._sinks = sinks or []

    def get_sources(self) -> list[type]:
        return self._sources

    def get_transforms(self) -> list[type]:
        return self._transforms

    def get_sinks(self) -> list[type]:
        return self._sinks
