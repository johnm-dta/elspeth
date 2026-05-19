"""Tests for tutorial run service hardening."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, cast

from elspeth.core.canonical import stable_hash
from elspeth.web.composer.tutorial_service import _coalesce_run_source_hashes, _normalise_bare_required_field_templates


def test_normalise_bare_required_field_templates_uses_row_namespace() -> None:
    original_template = "URL: {{url}}\nContent: {{ content }}\nOther: {{ ignored }}"
    nodes: list[dict[str, Any]] = [
        {
            "id": "rate_coolness",
            "plugin": "llm",
            "options": {
                "prompt_template": original_template,
                "required_input_fields": ["url", "content"],
                "resolved_prompt_template_hash": stable_hash(original_template),
            },
        }
    ]

    normalised, changed = _normalise_bare_required_field_templates(nodes)

    assert changed is True
    assert normalised is not None
    options = cast(dict[str, Any], normalised[0]["options"])
    prompt = cast(str, options["prompt_template"])
    assert prompt == "URL: {{ row.url }}\nContent: {{ row.content }}\nOther: {{ ignored }}"
    assert options["resolved_prompt_template_hash"] == stable_hash(prompt)
    original_options = cast(dict[str, Any], nodes[0]["options"])
    assert original_options["prompt_template"] == original_template


def test_normalise_bare_required_field_templates_leaves_row_namespace_alone() -> None:
    nodes: list[dict[str, Any]] = [
        {
            "id": "rate_coolness",
            "plugin": "llm",
            "options": {
                "prompt_template": "URL: {{ row.url }}",
                "required_input_fields": ["url"],
                "resolved_prompt_template_hash": stable_hash("URL: {{ row.url }}"),
            },
        }
    ]

    normalised, changed = _normalise_bare_required_field_templates(nodes)

    assert changed is False
    assert normalised == nodes


def test_normalise_bare_required_field_templates_replaces_interpretation_placeholders() -> None:
    original_template = "Rate how {{interpretation:cool}} this is, then explain why it is {{ interpretation: cool }}."
    nodes: list[dict[str, Any]] = [
        {
            "id": "rate_coolness",
            "plugin": "llm",
            "options": {
                "prompt_template": original_template,
                "required_input_fields": ["content"],
                "resolved_prompt_template_hash": stable_hash(original_template),
            },
        }
    ]

    normalised, changed = _normalise_bare_required_field_templates(nodes)

    assert changed is True
    assert normalised is not None
    options = cast(dict[str, Any], normalised[0]["options"])
    prompt = cast(str, options["prompt_template"])
    assert prompt == "Rate how cool this is, then explain why it is cool."
    assert options["resolved_prompt_template_hash"] == stable_hash(prompt)


def test_normalise_bare_required_field_templates_thaws_frozen_state_nodes() -> None:
    original_template = "Content: {{ content }}"
    nodes = (
        MappingProxyType(
            {
                "id": "rate_coolness",
                "plugin": "llm",
                "options": MappingProxyType(
                    {
                        "prompt_template": original_template,
                        "required_input_fields": ("content",),
                    }
                ),
            }
        ),
    )

    normalised, changed = _normalise_bare_required_field_templates(nodes)

    assert changed is True
    assert normalised is not None
    options = cast(dict[str, Any], normalised[0]["options"])
    assert options["prompt_template"] == "Content: {{ row.content }}"


def test_coalesce_run_source_hashes_aggregates_row_hashes() -> None:
    hashes = ("a" * 64, "b" * 64)

    assert _coalesce_run_source_hashes(hashes, run_id="run-1") == stable_hash({"source_data_hashes": list(hashes)})
