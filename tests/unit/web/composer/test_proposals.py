from __future__ import annotations

from elspeth.web.composer.proposals import build_tool_proposal_summary
from elspeth.web.composer.tools import is_mutation_tool


def test_is_mutation_tool_uses_closed_registries() -> None:
    assert is_mutation_tool("set_pipeline") is True
    assert is_mutation_tool("set_source_from_blob") is True
    assert is_mutation_tool("get_pipeline_state") is False
    assert is_mutation_tool("preview_pipeline") is False


def test_set_pipeline_summary_is_plain_language() -> None:
    summary = build_tool_proposal_summary(
        tool_name="set_pipeline",
        arguments={
            "source": {"plugin": "csv", "options": {}},
            "nodes": [{"id": "classify_severity", "plugin": "llm_classifier"}],
            "outputs": [{"name": "out", "plugin": "json"}],
        },
        redacted_arguments={
            "source": {"plugin": "csv", "options": {}},
            "nodes": [{"id": "classify_severity", "plugin": "llm_classifier"}],
            "outputs": [{"name": "out", "plugin": "json"}],
        },
    )

    assert summary.summary == "Replace the pipeline with csv input, 1 transform, and 1 output."
    assert summary.rationale == "Requested by the current composer turn."
    assert summary.affects == ("graph", "validation", "yaml")


def test_patch_node_options_summary_names_target_node() -> None:
    summary = build_tool_proposal_summary(
        tool_name="patch_node_options",
        arguments={"node_id": "classify_severity", "patch": {"model": "claude-haiku-4-5"}},
        redacted_arguments={"node_id": "classify_severity", "patch": {"model": "claude-haiku-4-5"}},
    )

    assert summary.summary == 'Update options for transform "classify_severity".'
    assert summary.affects == ("graph", "validation", "yaml")
