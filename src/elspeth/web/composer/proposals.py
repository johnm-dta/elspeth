"""Plain-language summaries for pending composer tool proposals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class ToolProposalSummary:
    summary: str
    rationale: str
    affects: tuple[str, ...]
    arguments_redacted_json: Mapping[str, Any]

    def __post_init__(self) -> None:
        freeze_fields(self, "affects", "arguments_redacted_json")


def _count_items(value: object) -> int:
    if type(value) is list:
        return len(value)
    if type(value) is tuple:
        return len(value)
    return 0


def _plural(count: int, singular: str) -> str:
    return f"{count} {singular}" if count == 1 else f"{count} {singular}s"


def _string_argument(arguments: Mapping[str, Any], key: str) -> str | None:
    if key not in arguments:
        return None
    value = arguments[key]
    return value if type(value) is str and value else None


def build_tool_proposal_summary(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    redacted_arguments: Mapping[str, Any],
) -> ToolProposalSummary:
    rationale = "Requested by the current composer turn."
    affects = ("graph", "validation", "yaml")

    if tool_name == "set_pipeline":
        source = arguments["source"] if "source" in arguments else None
        source_plugin = "new"
        if type(source) is dict and "plugin" in source and type(source["plugin"]) is str:
            source_plugin = source["plugin"]
        node_count = _count_items(arguments["nodes"] if "nodes" in arguments else ())
        output_count = _count_items(arguments["outputs"] if "outputs" in arguments else ())
        return ToolProposalSummary(
            summary=(
                f"Replace the pipeline with {source_plugin} input, "
                f"{_plural(node_count, 'transform')}, and {_plural(output_count, 'output')}."
            ),
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=redacted_arguments,
        )

    if tool_name == "set_source":
        label = _string_argument(arguments, "plugin") or "source"
        return ToolProposalSummary(
            summary=f"Set the pipeline source to {label}.",
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=redacted_arguments,
        )

    if tool_name == "patch_node_options":
        label = _string_argument(arguments, "node_id") or "selected transform"
        return ToolProposalSummary(
            summary=f'Update options for transform "{label}".',
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=redacted_arguments,
        )

    if tool_name == "patch_source_options":
        return ToolProposalSummary(
            summary="Update source options.",
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=redacted_arguments,
        )

    if tool_name == "patch_output_options":
        label = _string_argument(arguments, "sink_name") or "selected output"
        return ToolProposalSummary(
            summary=f'Update options for output "{label}".',
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=redacted_arguments,
        )

    return ToolProposalSummary(
        summary=f"Apply composer tool {tool_name}.",
        rationale=rationale,
        affects=affects,
        arguments_redacted_json=redacted_arguments,
    )
