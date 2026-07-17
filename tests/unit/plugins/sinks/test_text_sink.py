"""Tests for the strict, line-oriented local text sink."""

from __future__ import annotations

from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode
from elspeth.plugins.sinks.text_sink import TextSink, TextSinkConfig
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory


def _config(path: Path, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "path": str(path),
        "field": "line_text",
        "encoding": "utf-8",
        "mode": "write",
        "schema": {"mode": "observed"},
    }
    config.update(overrides)
    return config


def _sink_context() -> PluginContext:
    return make_context(landscape=make_factory().plugin_audit_writer())


@pytest.mark.parametrize("encoding", ["utf-8", "ascii", "latin-1", "cp1252"])
def test_text_sink_accepts_only_supported_ascii_compatible_encodings(encoding: str) -> None:
    parsed = TextSinkConfig.from_dict(_config(Path("out.txt"), encoding=encoding), plugin_name="text")
    assert parsed.encoding == encoding


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32", "utf-7", "shift_jis", "unknown-codec"])
def test_text_sink_rejects_encodings_outside_closed_set(encoding: str) -> None:
    with pytest.raises(PluginConfigError, match="encoding"):
        TextSinkConfig.from_dict(_config(Path("out.txt"), encoding=encoding), plugin_name="text")


@pytest.mark.parametrize("field", ["", "two words", "with-hyphen", "9lives", "class"])
def test_text_sink_rejects_non_identifier_or_keyword_field(field: str) -> None:
    with pytest.raises(PluginConfigError, match="identifier"):
        TextSinkConfig.from_dict(_config(Path("out.txt"), field=field), plugin_name="text")


def test_text_sink_config_schema_has_no_headers_property() -> None:
    assert "headers" not in TextSinkConfig.model_json_schema()["properties"]


@pytest.mark.parametrize(
    ("mode", "collision_policy"),
    [
        ("append", "fail_if_exists"),
        ("append", "auto_increment"),
        ("write", "append_or_create"),
    ],
)
def test_text_sink_rejects_incompatible_mode_and_collision_policy(mode: str, collision_policy: str) -> None:
    with pytest.raises(PluginConfigError, match="collision_policy"):
        TextSinkConfig.from_dict(
            _config(Path("out.txt"), mode=mode, collision_policy=collision_policy),
            plugin_name="text",
        )


def test_preflight_construction_never_mutates_or_resolves_output(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"winner\n")

    with plugin_preflight_mode(True):
        sink = TextSink(_config(path, collision_policy="auto_increment"))

    assert sink._path == path
    assert list(tmp_path.iterdir()) == [path]


def test_configure_for_resume_switches_to_append_contract(tmp_path: Path) -> None:
    sink = inject_write_failure(TextSink(_config(tmp_path / "out.txt")))

    sink.configure_for_resume()

    assert sink._mode == "append"
    assert sink._collision_policy == "append_or_create"


def test_text_sink_assistance_pins_strict_line_and_resume_contract() -> None:
    assistance = TextSink.get_agent_assistance(issue_code=None)

    assert assistance is not None
    hints = " ".join(assistance.composer_hints)
    assert "configured field" in hints
    assert "string" in hints
    assert "CR or LF" in hints
    assert "append" in hints
    assert "resume" in hints


class TestTextSinkResumeModeResolution:
    """elspeth-fc9906e398: resolved effect mode must claim what resume executes."""

    def test_resume_purpose_resolves_post_resume_append_mode(self) -> None:
        """A write-configured text sink resumes in append mode; the resolver must say so."""
        from elspeth.contracts.sink_effects import ResolvedSinkEffectMode, SinkEffectExecutionPurpose

        config = {"path": "/tmp/out.txt", "field": "line_text", "schema": {"mode": "observed"}, "mode": "write"}

        resolved = TextSink._resolve_sink_effect_mode(config, purpose=SinkEffectExecutionPurpose.RESUME)

        assert resolved == ResolvedSinkEffectMode("append")

    def test_fresh_purpose_keeps_configured_mode(self) -> None:
        from elspeth.contracts.sink_effects import ResolvedSinkEffectMode, SinkEffectExecutionPurpose

        config = {"path": "/tmp/out.txt", "field": "line_text", "schema": {"mode": "observed"}, "mode": "write"}

        resolved = TextSink._resolve_sink_effect_mode(config, purpose=SinkEffectExecutionPurpose.FRESH)

        assert resolved == ResolvedSinkEffectMode("write")
