"""Coverage assertion for builtin plugin discovery-time assistance.

Every builtin plugin that reaches the catalog/composer surface should publish
discovery-time hints through ``get_agent_assistance(issue_code=None)``. The
composer uses these hints as just-in-time guardrails; an empty tuple means the
LLM is left to infer plugin-specific contracts from schema names alone.

The hint content is advisory, but the presence, shape, and authoring discipline
are contract: every discovered plugin gets non-empty short hints and a summary.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from elspeth.plugins.infrastructure.discovery import discover_all_plugins


def _discovered_builtin_plugins() -> list[tuple[str, type]]:
    discovered = discover_all_plugins()
    return [(plugin_type.rstrip("s"), plugin_cls) for plugin_type, plugin_classes in discovered.items() for plugin_cls in plugin_classes]


@pytest.mark.parametrize(("plugin_type", "plugin_cls"), _discovered_builtin_plugins(), ids=lambda value: getattr(value, "name", str(value)))
def test_builtin_plugin_publishes_discovery_hints(plugin_type: str, plugin_cls: type) -> None:
    """Every discovered builtin plugin returns a non-empty composer_hints tuple."""
    assistance = plugin_cls.get_agent_assistance(issue_code=None)
    label = f"{plugin_type}/{plugin_cls.name}"
    assert assistance is not None, f"{label}.get_agent_assistance(issue_code=None) returned None"
    assert assistance.composer_hints, f"{label}.composer_hints is empty"
    assert isinstance(assistance.composer_hints, tuple)
    # Authoring discipline: each hint is a non-empty short imperative.
    for hint in assistance.composer_hints:
        assert isinstance(hint, str), f"{label}: composer_hint is not a string: {hint!r}"
        assert hint.strip(), f"{label}: composer_hint is blank"
        assert len(hint) <= 280, f"{label}: composer_hint exceeds 280 chars: {hint!r}"


@pytest.mark.parametrize(("plugin_type", "plugin_cls"), _discovered_builtin_plugins(), ids=lambda value: getattr(value, "name", str(value)))
def test_builtin_plugin_publishes_summary(plugin_type: str, plugin_cls: type) -> None:
    """Every discovered builtin plugin returns a non-empty summary."""
    assistance = plugin_cls.get_agent_assistance(issue_code=None)
    label = f"{plugin_type}/{plugin_cls.name}"
    assert assistance is not None
    assert assistance.summary, f"{label}.summary is empty"
    assert isinstance(assistance.summary, str)


def test_llm_assistance_includes_structured_multi_query_example() -> None:
    """The shared LLM assistance carries one concrete structured multi-query example.

    Catalog, freeform, guided-full, guided-staged, and tutorial all consume this
    same discovery-time assistance, so the structured-output authoring shape
    (``queries`` mapping with typed ``output_fields``) must live here — not be
    copied into any guided prompt.
    """
    pytest.importorskip(
        "litellm",
        reason="LLM transform requires the [llm] extra; discovery skips it otherwise.",
    )
    from elspeth.plugins.transforms.llm.transform import LLMTransform

    assistance = LLMTransform.get_agent_assistance(issue_code=None)
    assert assistance is not None

    structured_examples = [example for example in assistance.examples if isinstance(example.after, Mapping) and "queries" in example.after]
    assert structured_examples, "LLM assistance has no structured multi-query (queries) example"

    after = structured_examples[0].after
    assert after is not None
    queries = after["queries"]
    assert isinstance(queries, Mapping) and queries, "structured example queries must be a non-empty mapping"
    # The example must demonstrate typed output_fields (suffix + type), the
    # discovery contract Task 1 makes public.
    first_query = next(iter(queries.values()))
    output_fields = first_query["output_fields"]
    assert output_fields, "structured example query must declare output_fields"
    assert {"suffix", "type"} <= set(output_fields[0])
