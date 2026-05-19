"""Coverage assertion for the Phase 1 priority-17 plugin hint set.

The composer-jit-hints Phase 1 plan commits each of these 17 plugins to
publishing discovery-time hints (a non-empty ``composer_hints`` tuple
on ``get_agent_assistance(issue_code=None)``). This test pins the set
so a regression (someone refactors the hook and drops the override)
fails CI rather than silently demoting the catalog response.

The hints content itself is *advisory*, not contract — these
assertions intentionally do not validate prose, only presence and
shape.
"""

from __future__ import annotations

import importlib

import pytest

# (module_path, class_name) — keep this list in lockstep with the
# priority-17 table in the plan file.
PRIORITY_PLUGINS: list[tuple[str, str]] = [
    # Sources
    ("elspeth.plugins.sources.csv_source", "CSVSource"),
    ("elspeth.plugins.sources.json_source", "JSONSource"),
    ("elspeth.plugins.sources.dataverse", "DataverseSource"),
    # Transforms
    ("elspeth.plugins.transforms.llm.transform", "LLMTransform"),
    ("elspeth.plugins.transforms.web_scrape", "WebScrapeTransform"),
    ("elspeth.plugins.transforms.line_explode", "LineExplode"),
    ("elspeth.plugins.transforms.json_explode", "JSONExplode"),
    ("elspeth.plugins.transforms.field_mapper", "FieldMapper"),
    ("elspeth.plugins.transforms.value_transform", "ValueTransform"),
    ("elspeth.plugins.transforms.type_coerce", "TypeCoerce"),
    ("elspeth.plugins.transforms.truncate", "Truncate"),
    ("elspeth.plugins.transforms.batch_distribution_profile", "BatchDistributionProfile"),
    ("elspeth.plugins.transforms.rag.transform", "RAGRetrievalTransform"),
    ("elspeth.plugins.transforms.azure.content_safety", "AzureContentSafety"),
    # Sinks
    ("elspeth.plugins.sinks.json_sink", "JSONSink"),
    ("elspeth.plugins.sinks.csv_sink", "CSVSink"),
    ("elspeth.plugins.sinks.database_sink", "DatabaseSink"),
]


@pytest.mark.parametrize(("module_path", "class_name"), PRIORITY_PLUGINS)
def test_priority_plugin_publishes_discovery_hints(module_path: str, class_name: str) -> None:
    """Every priority-17 plugin returns a non-empty composer_hints tuple at issue_code=None."""
    module = importlib.import_module(module_path)
    plugin_cls = getattr(module, class_name)
    assistance = plugin_cls.get_agent_assistance(issue_code=None)
    assert assistance is not None, f"{class_name}.get_agent_assistance(issue_code=None) returned None"
    assert assistance.composer_hints, f"{class_name}.composer_hints is empty"
    assert isinstance(assistance.composer_hints, tuple)
    # Authoring discipline: each hint is a non-empty short imperative.
    for hint in assistance.composer_hints:
        assert isinstance(hint, str), f"{class_name}: composer_hint is not a string: {hint!r}"
        assert hint.strip(), f"{class_name}: composer_hint is blank"
        assert len(hint) <= 280, f"{class_name}: composer_hint exceeds 280 chars: {hint!r}"


@pytest.mark.parametrize(("module_path", "class_name"), PRIORITY_PLUGINS)
def test_priority_plugin_publishes_summary(module_path: str, class_name: str) -> None:
    """Every priority-17 plugin returns a non-empty summary at issue_code=None."""
    module = importlib.import_module(module_path)
    plugin_cls = getattr(module, class_name)
    assistance = plugin_cls.get_agent_assistance(issue_code=None)
    assert assistance is not None
    assert assistance.summary, f"{class_name}.summary is empty"
    assert isinstance(assistance.summary, str)
