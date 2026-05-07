"""Tests for plugin assistance contract types — including deep-freeze guards."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from elspeth.contracts.plugin_assistance import (
    PluginAssistance,
    PluginAssistanceExample,
)


class TestPluginAssistanceExampleFreeze:
    def test_title_is_required(self) -> None:
        kwargs: dict[str, Any] = {"before": None, "after": None}

        with pytest.raises(TypeError, match="missing 1 required positional argument: 'title'"):
            PluginAssistanceExample(**kwargs)

    def test_dict_fields_become_mapping_proxy(self) -> None:
        example = PluginAssistanceExample(
            title="t",
            before={"format": "text", "text_separator": " "},
            after={"format": "text", "text_separator": "\n"},
        )
        assert isinstance(example.before, MappingProxyType)
        assert isinstance(example.after, MappingProxyType)

    def test_none_fields_are_left_none(self) -> None:
        example = PluginAssistanceExample(title="t", before=None, after=None)
        assert example.before is None
        assert example.after is None

    def test_inner_mutation_is_blocked(self) -> None:
        original = {"format": "text"}
        example = PluginAssistanceExample(title="t", before=original)
        assert example.before is not None
        with pytest.raises(TypeError):
            example.before["format"] = "markdown"  # type: ignore[index]
        # And mutating the source dict does NOT affect the frozen field
        original["format"] = "markdown"
        assert example.before["format"] == "text"


class TestPluginAssistanceFreeze:
    def test_examples_field_freezes_inner_dicts(self) -> None:
        example = PluginAssistanceExample(title="t", before={"k": "v"})
        assistance = PluginAssistance(
            plugin_name="web_scrape",
            issue_code="line_explode.source_field.line_framed_text",
            summary="Set text_separator to '\\n'.",
            suggested_fixes=("Set text_separator: '\\n'", "Or use format: markdown"),
            examples=(example,),
        )
        # Both examples and inner dicts are frozen.
        assert isinstance(assistance.examples, tuple)
        assert isinstance(assistance.examples[0].before, MappingProxyType)

    @pytest.mark.parametrize("missing_field", ["plugin_name", "issue_code", "summary"])
    def test_required_fields_have_no_defaults(self, missing_field: str) -> None:
        kwargs: dict[str, Any] = {
            "plugin_name": "p",
            "issue_code": None,
            "summary": "s",
        }
        del kwargs[missing_field]

        with pytest.raises(
            TypeError,
            match=rf"missing 1 required positional argument: '{missing_field}'",
        ):
            PluginAssistance(**kwargs)

    def test_optional_collection_fields_have_empty_defaults(self) -> None:
        assistance = PluginAssistance(
            plugin_name="p",
            issue_code=None,
            summary="s",
        )
        assert assistance.suggested_fixes == ()
        assert assistance.examples == ()
        assert assistance.composer_hints == ()

    def test_list_inputs_are_coerced_to_tuples(self) -> None:
        # Type annotations name tuples, but Python does not enforce that.
        # A caller passing a list must produce an immutable tuple field
        # rather than a live list reference back to caller-mutable state.
        suggested = ["fix-a", "fix-b"]
        examples = [PluginAssistanceExample(title="t")]
        hints = ["hint-a"]
        assistance = PluginAssistance(
            plugin_name="p",
            issue_code=None,
            summary="s",
            suggested_fixes=suggested,  # type: ignore[arg-type]
            examples=examples,  # type: ignore[arg-type]
            composer_hints=hints,  # type: ignore[arg-type]
        )
        assert isinstance(assistance.suggested_fixes, tuple)
        assert isinstance(assistance.examples, tuple)
        assert isinstance(assistance.composer_hints, tuple)
        # Mutating the original list MUST NOT affect the frozen field.
        suggested.append("fix-c")
        assert "fix-c" not in assistance.suggested_fixes
