"""Tests for line_explode deaggregation transform."""

from __future__ import annotations

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

DYNAMIC_SCHEMA = {"mode": "observed"}


@pytest.fixture
def ctx() -> PluginContext:
    return make_context()


def test_line_explode_splits_string_into_indexed_rows(ctx: PluginContext) -> None:
    from elspeth.plugins.transforms.line_explode import LineExplode

    transform = LineExplode(
        {
            "schema": DYNAMIC_SCHEMA,
            "source_field": "html",
            "output_field": "html_line",
        }
    )

    result = transform.process(
        make_pipeline_row(
            {
                "source_url": "https://www.wardline.dev/",
                "html": "<html>\n<body>\n<h1>Wardline</h1>",
            }
        ),
        ctx,
    )

    assert result.status == "success"
    assert result.is_multi_row
    assert result.rows is not None
    assert [row.to_dict() for row in result.rows] == [
        {"source_url": "https://www.wardline.dev/", "html_line": "<html>", "line_index": 0},
        {"source_url": "https://www.wardline.dev/", "html_line": "<body>", "line_index": 1},
        {"source_url": "https://www.wardline.dev/", "html_line": "<h1>Wardline</h1>", "line_index": 2},
    ]


def test_line_explode_preserves_empty_lines(ctx: PluginContext) -> None:
    from elspeth.plugins.transforms.line_explode import LineExplode

    transform = LineExplode(
        {
            "schema": DYNAMIC_SCHEMA,
            "source_field": "html",
            "output_field": "html_line",
        }
    )

    result = transform.process(make_pipeline_row({"html": "a\n\nb"}), ctx)

    assert result.rows is not None
    assert [row.to_dict()["html_line"] for row in result.rows] == ["a", "", "b"]


def test_line_explode_can_omit_index(ctx: PluginContext) -> None:
    from elspeth.plugins.transforms.line_explode import LineExplode

    transform = LineExplode(
        {
            "schema": DYNAMIC_SCHEMA,
            "source_field": "html",
            "output_field": "html_line",
            "include_index": False,
        }
    )

    result = transform.process(make_pipeline_row({"html": "a\nb"}), ctx)

    assert result.rows is not None
    assert [row.to_dict() for row in result.rows] == [{"html_line": "a"}, {"html_line": "b"}]


def test_line_explode_output_contract_matches_declared_fields(ctx: PluginContext) -> None:
    from elspeth.plugins.transforms.line_explode import LineExplode

    transform = LineExplode(
        {
            "schema": DYNAMIC_SCHEMA,
            "source_field": "html",
            "output_field": "html_line",
        }
    )

    result = transform.process(make_pipeline_row({"html": "a\nb"}), ctx)

    assert result.rows is not None
    field_by_name = {field.normalized_name: field for field in result.rows[0].contract.fields}
    for field_name in ("html_line", "line_index"):
        assert field_by_name[field_name].required is True
        assert field_by_name[field_name].source == "declared"


def test_line_explode_empty_string_returns_error(ctx: PluginContext) -> None:
    from elspeth.plugins.transforms.line_explode import LineExplode

    transform = LineExplode(
        {
            "schema": DYNAMIC_SCHEMA,
            "source_field": "html",
            "output_field": "html_line",
        }
    )

    result = transform.process(make_pipeline_row({"html": ""}), ctx)

    assert result.status == "error"
    assert result.reason is not None
    assert result.reason["reason"] == "invalid_input"
    assert result.reason["field"] == "html"
    assert not result.retryable


def test_line_explode_crashes_on_non_string_input(ctx: PluginContext) -> None:
    from elspeth.plugins.transforms.line_explode import LineExplode

    transform = LineExplode(
        {
            "schema": DYNAMIC_SCHEMA,
            "source_field": "html",
            "output_field": "html_line",
        }
    )

    with pytest.raises(TypeError, match="must be a string"):
        transform.process(make_pipeline_row({"html": ["not", "a", "string"]}), ctx)


def test_line_explode_is_discovered_by_plugin_manager() -> None:
    from elspeth.plugins.infrastructure.manager import PluginManager

    manager = PluginManager()
    manager.register_builtin_plugins()

    assert manager.get_transform_by_name("line_explode").name == "line_explode"


class TestLineExplodeInputSemanticRequirements:
    def _build(self, **opts):
        # LineExplodeConfig is a TransformDataConfig subclass — schema
        # is REQUIRED. Omission raises PluginConfigError → validator
        # silently skips → vacuous test. See web_scrape helper above
        # for the same pattern.
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

        defaults = {
            "schema": {"mode": "flexible", "fields": ["content: str"]},
            "source_field": "content",
        }
        defaults.update(opts)
        return get_shared_plugin_manager().create_transform("line_explode", defaults)

    def test_default_source_field_requirement(self):
        from elspeth.contracts.plugin_semantics import (
            ContentKind,
            TextFraming,
            UnknownSemanticPolicy,
        )

        plugin = self._build()
        reqs = plugin.input_semantic_requirements()
        assert len(reqs.fields) == 1
        req = reqs.fields[0]
        assert req.field_name == "content"
        assert req.accepted_content_kinds == frozenset({ContentKind.PLAIN_TEXT, ContentKind.MARKDOWN})
        assert req.accepted_text_framings == frozenset({TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE})
        assert req.requirement_code == "line_explode.source_field.line_framed_text"
        assert req.unknown_policy is UnknownSemanticPolicy.FAIL
        assert req.configured_by == ("source_field",)

    def test_custom_source_field_changes_requirement_field_name(self):
        plugin = self._build(
            schema={"mode": "flexible", "fields": ["body: str"]},
            source_field="body",
        )
        req = plugin.input_semantic_requirements().fields[0]
        assert req.field_name == "body"


class TestLineExplodeAssistance:
    def test_returns_assistance_for_line_framed_requirement(self):
        from elspeth.plugins.transforms.line_explode import LineExplode

        a = LineExplode.get_agent_assistance(
            issue_code="line_explode.source_field.line_framed_text",
        )
        assert a is not None
        assert a.plugin_name == "line_explode"
        assert "splitlines" in a.summary or "line" in a.summary.lower()

    def test_returns_none_for_unknown_issue(self):
        from elspeth.plugins.transforms.line_explode import LineExplode

        assert LineExplode.get_agent_assistance(issue_code="nope") is None

    def test_assistance_does_not_leak_secret_options(self):
        """Sentinel test: configured option values must not bleed into assistance prose.

        get_agent_assistance() is a classmethod; the build-time options never
        flow into it. We construct a sentinel-laced plugin and assert both
        input_semantic_requirements() and get_agent_assistance() carry no raw
        sentinel — the only configured-leak surface is the configured_by
        tuple, which the contract limits to safe option NAMES.
        """
        from elspeth.contracts.plugin_assistance import (
            PluginAssistance,
            PluginAssistanceExample,
        )
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
        from elspeth.plugins.transforms.line_explode import LineExplode

        # Sentinel-shaped (still a valid identifier so config validators accept it).
        sentinel_field = "SENTINEL_LEAK_FIELD_credential_marker"

        plugin = get_shared_plugin_manager().create_transform(
            "line_explode",
            {
                "schema": {
                    "mode": "flexible",
                    "fields": [f"{sentinel_field}: str"],
                },
                "source_field": sentinel_field,
            },
        )

        # input_semantic_requirements echoes field_name (must) — that's the
        # configured contract. configured_by must NOT echo values, only
        # option NAMES (e.g. "source_field").
        reqs = plugin.input_semantic_requirements()
        for req in reqs.fields:
            # field_name legitimately echoes the configured field name; that
            # is the requirement's identity. configured_by must contain only
            # OPTION NAMES (not VALUES).
            for entry in req.configured_by:
                assert sentinel_field not in entry, (
                    f"configured_by entry {entry!r} leaks the source_field VALUE; configured_by must list option names only."
                )

        # get_agent_assistance is sentinel-free — it's a classmethod with no
        # access to instance config. Verify by scanning all prose fields.
        a = LineExplode.get_agent_assistance(
            issue_code="line_explode.source_field.line_framed_text",
        )
        assert a is not None
        assert isinstance(a, PluginAssistance)

        def _scan(text: str) -> None:
            assert sentinel_field not in text

        _scan(a.plugin_name)
        if a.issue_code is not None:
            _scan(a.issue_code)
        _scan(a.summary)
        for fix in a.suggested_fixes:
            _scan(fix)
        for hint in a.composer_hints:
            _scan(hint)
        for example in a.examples:
            assert isinstance(example, PluginAssistanceExample)
            _scan(example.title)
            for mapping_field in (example.before, example.after):
                if mapping_field is None:
                    continue
                for key, value in mapping_field.items():
                    _scan(str(key))
                    _scan(str(value))
