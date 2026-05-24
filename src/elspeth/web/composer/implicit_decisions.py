"""Structured implicit-decision report for composer-authored states.

Layer: L3 (application).

The composer skill asks the model to tell the operator which choices it made
on their behalf. This module provides the persisted counterpart: a compact,
state-derived report that survives reload and can be inspected by auditors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from elspeth.contracts.freeze import deep_thaw
from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, SourceSpec

DecisionCategory = Literal[
    "error_routing",
    "identity",
    "model",
    "output",
    "plugin_option",
    "source",
]
DecisionProvenance = Literal[
    "composer_selected",
    "default",
    "explicit_source_required",
    "picked",
]


class ImplicitDecisionEntry(TypedDict, total=False):
    path: str
    value: object
    category: DecisionCategory
    provenance: DecisionProvenance
    candidate_alternatives: list[object]
    note: str


class ImplicitDecisionsReport(TypedDict):
    schema_version: int
    entries: list[ImplicitDecisionEntry]
    normalization_events: list[dict[str, object]]


_FORMAT_ALTERNATIVES = ["html", "markdown", "text"]
_ALLOWED_HOSTS_ALTERNATIVES = ["public_only", "same_site", "explicit_allowlist"]
_COLLISION_POLICY_ALTERNATIVES = ["fail", "overwrite", "auto_increment"]
_ROUTING_ALTERNATIVES = ["discard", "named_sink"]
_MODEL_PROVIDER_ALTERNATIVES = ["openrouter", "azure_openai"]


def build_implicit_decisions_report(state: CompositionState) -> ImplicitDecisionsReport:
    """Return a machine-readable disclosure report for a composer state.

    The report is intentionally derived from the final state rather than from
    model prose. That means it is conservative about provenance: when the final
    state alone cannot prove whether a value came from the operator or a
    deployment identity record, the entry says ``explicit_source_required``.
    """

    entries: list[ImplicitDecisionEntry] = []
    for source in state.sources.values():
        entries.extend(_source_entries(source))
    for node in state.nodes:
        entries.extend(_node_entries(node))
    for output in state.outputs:
        entries.extend(_output_entries(output))

    return {
        "schema_version": 1,
        "entries": entries,
        "normalization_events": [],
    }


def merge_implicit_decisions_meta(
    composer_meta: Mapping[str, Any] | None,
    state: CompositionState,
) -> dict[str, object]:
    """Merge the current implicit-decision report into ``composer_meta``."""

    merged: dict[str, object] = dict(deep_thaw(composer_meta)) if composer_meta is not None else {}
    merged["implicit_decisions"] = build_implicit_decisions_report(state)
    return merged


def _source_entries(source: SourceSpec) -> list[ImplicitDecisionEntry]:
    entries = [
        _entry(
            f"source.{field_path}",
            value,
            category="source",
            provenance=_provenance_for_path(f"source.{field_path}", value),
        )
        for field_path, value in _flatten_options(source.options)
    ]
    entries.append(
        _entry(
            "source.on_validation_failure",
            source.on_validation_failure,
            category="error_routing",
            provenance=_routing_provenance(source.on_validation_failure),
            candidate_alternatives=_ROUTING_ALTERNATIVES,
        )
    )
    return entries


def _node_entries(node: NodeSpec) -> list[ImplicitDecisionEntry]:
    node_id = node.id
    entries = [
        _entry(
            f"node.{node_id}.options.{field_path}",
            value,
            category=_category_for_node_option(node, field_path),
            provenance=_provenance_for_path(f"node.{node_id}.options.{field_path}", value),
            candidate_alternatives=_candidate_alternatives(field_path),
            note=_note_for_node_option(node, field_path),
        )
        for field_path, value in _flatten_options(node.options)
    ]
    if node.on_error is not None:
        entries.append(
            _entry(
                f"node.{node_id}.on_error",
                node.on_error,
                category="error_routing",
                provenance=_routing_provenance(node.on_error),
                candidate_alternatives=_ROUTING_ALTERNATIVES,
            )
        )
    return entries


def _output_entries(output: OutputSpec) -> list[ImplicitDecisionEntry]:
    output_name = output.name
    entries = [
        _entry(
            f"output.{output_name}.options.{field_path}",
            value,
            category="output",
            provenance=_provenance_for_path(f"output.{output_name}.options.{field_path}", value),
            candidate_alternatives=_candidate_alternatives(field_path),
        )
        for field_path, value in _flatten_options(output.options)
    ]
    entries.append(
        _entry(
            f"output.{output_name}.on_write_failure",
            output.on_write_failure,
            category="error_routing",
            provenance=_routing_provenance(output.on_write_failure),
            candidate_alternatives=_ROUTING_ALTERNATIVES,
        )
    )
    return entries


def _flatten_options(options: Mapping[str, Any], prefix: str = "") -> list[tuple[str, object]]:
    flattened: list[tuple[str, object]] = []
    for key in sorted(options):
        value = options[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.extend(_flatten_options(value, path))
        else:
            flattened.append((path, deep_thaw(value)))
    return flattened


def _entry(
    path: str,
    value: object,
    *,
    category: DecisionCategory,
    provenance: DecisionProvenance,
    candidate_alternatives: Sequence[object] | None = None,
    note: str | None = None,
) -> ImplicitDecisionEntry:
    entry: ImplicitDecisionEntry = {
        "path": path,
        "value": deep_thaw(value),
        "category": category,
        "provenance": provenance,
    }
    if candidate_alternatives is not None:
        entry["candidate_alternatives"] = list(candidate_alternatives)
    if note is not None:
        entry["note"] = note
    return entry


def _category_for_node_option(node: NodeSpec, field_path: str) -> DecisionCategory:
    if node.plugin == "web_scrape" and field_path in {"http.abuse_contact", "http.scraping_reason"}:
        return "identity"
    if node.plugin == "llm" and field_path in {"provider", "model", "temperature", "pool_size"}:
        return "model"
    return "plugin_option"


def _provenance_for_path(path: str, value: object) -> DecisionProvenance:
    if path.endswith(".http.abuse_contact") or path.endswith(".http.scraping_reason"):
        return "explicit_source_required"
    if path.endswith(".allowed_hosts") and value == "public_only":
        return "default"
    if path.endswith(".collision_policy") and value == "auto_increment":
        return "default"
    if path.endswith(".temperature") or path.endswith(".pool_size") or path.endswith(".model") or path.endswith(".provider"):
        return "picked"
    return "composer_selected"


def _routing_provenance(value: object) -> DecisionProvenance:
    if value == "discard":
        return "picked"
    return "composer_selected"


def _candidate_alternatives(field_path: str) -> list[object] | None:
    if field_path == "format":
        return list(_FORMAT_ALTERNATIVES)
    if field_path.endswith("allowed_hosts"):
        return list(_ALLOWED_HOSTS_ALTERNATIVES)
    if field_path == "collision_policy":
        return list(_COLLISION_POLICY_ALTERNATIVES)
    if field_path == "provider":
        return list(_MODEL_PROVIDER_ALTERNATIVES)
    return None


def _note_for_node_option(node: NodeSpec, field_path: str) -> str | None:
    if node.plugin == "web_scrape" and field_path in {"http.abuse_contact", "http.scraping_reason"}:
        return "Wire-visible identity value; must come from the operator or deployment identity, never a fabricated default."
    return None
