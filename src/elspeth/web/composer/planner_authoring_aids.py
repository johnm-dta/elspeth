"""Live-rendered planner authoring aids: worked exemplars from the live catalog.

The static skill pack deliberately carries no deployment plugin inventory (the
``no_deployment_plugin_facts`` gate enforces this), so worked ``set_pipeline``
exemplars — which must name real plugins — are rendered here at prompt-build
from the policy-visible catalog and ride in the planner's reviewed-context
user message. The exact objects rendered into the prompt are validated
through ``build_set_pipeline_candidate`` in
``tests/unit/web/composer/test_planner_authoring_aids.py``; an exemplar the
current validator rejects fails CI rather than teaching planners a dead shape.

Evidence base: the 2026-07-22 pack stress test (0/6 cold planners converged;
5/6 fabricated a ``blob_id``, 1/6 missed the source options contract). See
``scratch/planner-skill-pack-assessment.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Final

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.schemas import PluginSummary

# The prompt never models a fabricated identifier — provenance is the lesson.
PLACEHOLDER_BLOB_ID: Final[str] = "<blob_id copied verbatim from a list_blobs or create_blob result>"

_SOURCE_CUSTODY_RULES: Final[tuple[str, ...]] = (
    "A blob_id comes ONLY from blob-tool output in this session (list_blobs, "
    "list_composer_blobs, create_blob, get_blob_metadata). Copy it verbatim.",
    "If no tool returned the identifier, bind the data with source.inline_blob "
    "(filename, mime_type, content) or create_blob first. Never fabricate a "
    "blob_id, secret reference, model identifier, or any other identifier.",
    "inline_blob.content must be the user's data verbatim, exactly as it "
    "appears in their message; custody records it against that message.",
    "Custody owns the storage binding: author schema.mode and on_validation_failure on a blob-bound source, never path or blob_ref.",
)

_INLINE_EXEMPLAR_FILENAME: Final[str] = "quarterly_totals.csv"
_INLINE_EXEMPLAR_MIME: Final[str] = "text/csv"
_INLINE_EXEMPLAR_CONTENT: Final[str] = "region,total\nnorth,412\nsouth,388\n"

_FORK_COALESCE_RULES: Final[tuple[str, ...]] = (
    "When the user asks for separate LLM nodes per branch (one model call per "
    "aspect, compared side by side), use THIS shape — a gate fanning out to "
    "one llm transform per branch, rejoined by a coalesce — not a queries map "
    "on a single llm node.",
    "Key the coalesce branches map by FORK BRANCH NAME; each value names the "
    "connection arriving at the coalesce after that branch's transforms.",
    "A coalesce publishes its merged rows under its own node id — the "
    "downstream consumer sets input to the coalesce id. Do not author "
    "on_success on a coalesce unless it routes directly to a sink.",
    "Give each branch's llm node its own response_field so the union merge carries both results on one row.",
    "Author llm-node interpretation_requirements in the short form {kind, user_term, draft} exactly as the exemplar shows — user_term is mandatory; a row without it is rejected. Omitting the whole block is also legal (required reviews auto-stage).",
)

_FORK_EXEMPLAR_CONTENT: Final[str] = "color_name,hex\ncerulean,#2A52BE\nsaffron,#F4C430\n"


def _usable_llm_profile_alias(catalog: PolicyCatalogView) -> str | None:
    """Return the selected (else first usable) llm operator-profile alias."""
    snapshot = catalog.snapshot
    llm_id = next(
        (
            plugin_id
            for plugin_id, aliases in snapshot.usable_profile_aliases
            if plugin_id.kind == "transform" and plugin_id.name == "llm" and aliases
        ),
        None,
    )
    if llm_id is None:
        return None
    selected = dict(snapshot.selected_profile_aliases).get(llm_id)
    if selected is not None:
        return selected
    return dict(snapshot.usable_profile_aliases)[llm_id][0]


def _plugin_summaries(catalog: PolicyCatalogView) -> dict[str, list[PluginSummary]]:
    """One catalog sweep shared by every aid — the expensive step of a build."""
    return {
        "source": catalog.list_sources(),
        "transform": catalog.list_transforms(),
        "sink": catalog.list_sinks(),
    }


def _visible_plugin_names(
    catalog: PolicyCatalogView, summaries: Mapping[str, list[PluginSummary]] | None = None
) -> dict[str, frozenset[str]]:
    if summaries is None:
        summaries = _plugin_summaries(catalog)
    return {kind: frozenset(plugin.name for plugin in plugins) for kind, plugins in summaries.items()}


def _digest_entries(plugins: list[PluginSummary]) -> list[dict[str, Any]]:
    return [
        {
            "name": plugin.name,
            "purpose": plugin.description,
            "required_options": [field.name for field in plugin.config_fields if field.required],
            "composer_hints": list(plugin.composer_hints),
        }
        for plugin in plugins
    ]


def discovery_digest(
    catalog: PolicyCatalogView,
    *,
    summaries: Mapping[str, list[PluginSummary]] | None = None,
) -> dict[str, Any]:
    """Per-plugin digest of the policy-visible catalog for the planner prompt.

    Targets ``planner_code=DISCOVERY_CYCLE`` churn: a significant share of
    planner calls were ``list_*``/``get_plugin_schema`` rounds re-learning the
    same catalog every session. Each entry carries the plugin's name, one-line
    purpose, required knobs, and its ``composer_hints`` verbatim — the hints
    are the designated live channel for web-policy facts that plugin schemas
    cannot express.
    """
    if summaries is None:
        summaries = _plugin_summaries(catalog)
    return {
        "sources": _digest_entries(summaries["source"]),
        "transforms": _digest_entries(summaries["transform"]),
        "sinks": _digest_entries(summaries["sink"]),
    }


_DISCOVERY_DIGEST_GUIDANCE: Final[str] = (
    "This digest is rendered from the live policy-visible catalog at prompt "
    "build and is current for this deployment: you rarely need "
    "list_sources/list_transforms/list_sinks or get_plugin_schema calls — "
    "plan directly from it. Model identifiers still come only from "
    "list_models, and blob/secret discovery is unchanged. Use "
    "get_plugin_assistance and explain_validation_error for structured "
    "repair when a proposal is rejected."
)


def source_custody_exemplar_args(
    catalog: PolicyCatalogView,
    *,
    blob_id: str | None = None,
    visible: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, Any] | None:
    """Complete ``set_pipeline`` args showing one legal source custody binding.

    With ``blob_id=None`` the source binds literal user data via
    ``inline_blob``; passing a ``blob_id`` (the prompt passes
    :data:`PLACEHOLDER_BLOB_ID`; the validation test passes a real created
    blob's id) shows the existing-blob binding instead. Everything outside
    ``source`` is byte-identical between the variants. Returns ``None`` when
    the plugins the exemplar names are not policy-visible. ``visible`` lets
    the payload builder share one catalog sweep across every exemplar.
    """
    if visible is None:
        visible = _visible_plugin_names(catalog)
    if "csv" not in visible["source"] or "json" not in visible["sink"]:
        return None
    if blob_id is None:
        binding: dict[str, Any] = {
            "inline_blob": {
                "filename": _INLINE_EXEMPLAR_FILENAME,
                "mime_type": _INLINE_EXEMPLAR_MIME,
                "content": _INLINE_EXEMPLAR_CONTENT,
                "description": "Literal rows the user pasted into chat",
            }
        }
    else:
        binding = {"blob_id": blob_id}
    return {
        "source": {
            "plugin": "csv",
            "on_success": "main",
            "options": {"schema": {"mode": "observed"}},
            "on_validation_failure": "discard",
            **binding,
        },
        "nodes": [],
        "edges": [],
        "outputs": [
            {
                "sink_name": "main",
                "plugin": "json",
                "options": {
                    "path": "outputs/quarterly_totals.json",
                    "format": "json",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
        "metadata": {
            "name": "Save pasted rows",
            "description": "Bind user-provided rows through blob custody and write them to one JSON output.",
        },
    }


def fork_coalesce_exemplar_args(
    catalog: PolicyCatalogView,
    *,
    visible: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, Any] | None:
    """Complete ``set_pipeline`` args for the fork -> two-llm -> coalesce shape.

    The operator-ruled A/B topology: a gate fans identical rows out to two
    branches, each branch runs its own llm transform (own prompt, own
    ``response_field``), a coalesce rejoins them under ``require_all``/
    ``union``, one cleanup transform consumes the coalesce id, and a sink
    receives the tidied rows. Returns ``None`` when the plugins it names or a
    usable llm operator profile are not visible — an exemplar must never model
    an invented identifier.
    """
    if visible is None:
        visible = _visible_plugin_names(catalog)
    if "csv" not in visible["source"] or "json" not in visible["sink"]:
        return None
    if not {"llm", "field_mapper"} <= visible["transform"]:
        return None
    profile_alias = _usable_llm_profile_alias(catalog)
    if profile_alias is None:
        return None

    def _branch_llm(node_id: str, branch: str, response_field: str, question: str) -> dict[str, Any]:
        return {
            "id": node_id,
            "node_type": "transform",
            "plugin": "llm",
            "input": branch,
            "on_success": f"{response_field}_done",
            "on_error": "discard",
            "options": {
                "profile": profile_alias,
                "prompt_template": question,
                "required_input_fields": ["color_name", "hex"],
                "response_field": response_field,
                "schema": {"mode": "observed"},
                # The short review-requirement form: id/status are synthesized
                # at the boundary; user_term is MANDATORY (never author a
                # requirement row without kind + user_term + draft).
                "interpretation_requirements": [
                    {
                        "kind": "llm_prompt_template",
                        "user_term": f"llm_prompt_template:{node_id}",
                        "draft": question,
                    }
                ],
            },
        }

    return {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {
                "schema": {
                    "mode": "flexible",
                    "fields": ["color_name: str", "hex: str"],
                    "guaranteed_fields": ["color_name", "hex"],
                }
            },
            "on_validation_failure": "discard",
            "inline_blob": {
                "filename": "colours.csv",
                "mime_type": "text/csv",
                "content": _FORK_EXEMPLAR_CONTENT,
                "description": "Literal rows the user pasted into chat",
            },
        },
        "nodes": [
            {
                "id": "fan_out",
                "node_type": "gate",
                "input": "rows",
                "condition": "True",
                "routes": {"true": "fork", "false": "fork"},
                "fork_to": ["branch_a", "branch_b"],
            },
            _branch_llm(
                "assess_tone",
                "branch_a",
                "tone",
                "What is the emotional tone of the colour {{ row.color_name }} ({{ row.hex }})? Reply with one short phrase.",
            ),
            _branch_llm(
                "assess_usage",
                "branch_b",
                "usage",
                "Name one design usage for the colour {{ row.color_name }} ({{ row.hex }}). Reply with one short phrase.",
            ),
            {
                "id": "merge_branches",
                "node_type": "coalesce",
                "input": "branches",
                "branches": {"branch_a": "tone_done", "branch_b": "usage_done"},
                "policy": "require_all",
                "merge": "union",
                "options": {"schema": {"mode": "observed"}},
            },
            {
                "id": "tidy_columns",
                "node_type": "transform",
                "plugin": "field_mapper",
                "input": "merge_branches",
                "on_success": "main",
                "on_error": "discard",
                "options": {
                    "schema": {"mode": "observed"},
                    "mapping": {
                        "color_name": "color_name",
                        "hex": "hex",
                        "tone": "tone",
                        "usage": "usage",
                    },
                    "select_only": True,
                },
            },
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "main",
                "plugin": "json",
                "options": {
                    "path": "outputs/colour_assessments.json",
                    "format": "json",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
        "metadata": {
            "name": "Per-branch LLM assessment",
            "description": "Fan rows out to one llm transform per branch, rejoin with a coalesce, tidy, and save.",
        },
    }


# Payload memo keyed by plugin-policy snapshot hash. The aids depend only on
# the policy-visible catalog projection (plugin classes are static per
# process; visibility and profile aliases are exactly what the snapshot hash
# covers), and a cold build costs a full catalog sweep (~50ms) — too much to
# repeat inside every planner call's wall-clock budget. Bounded so snapshot
# rotation cannot grow it without limit; callers receive a deep copy so no
# caller can poison the cached payload.
_AIDS_MEMO: dict[str, dict[str, Any]] = {}
_AIDS_MEMO_MAX: Final[int] = 8


def build_planner_authoring_aids(catalog: PolicyCatalogView) -> dict[str, Any]:
    """Assemble the live authoring-aids payload for one planner call.

    Rendered from the policy-visible catalog (memoized per snapshot hash), so
    it can never drift from the deployment. Sections whose plugins are
    policy-hidden are omitted rather than rendered with invented names.
    """
    key = catalog.snapshot.snapshot_hash
    cached = _AIDS_MEMO.get(key)
    if cached is None:
        cached = _build_planner_authoring_aids(catalog)
        if len(_AIDS_MEMO) >= _AIDS_MEMO_MAX:
            _AIDS_MEMO.pop(next(iter(_AIDS_MEMO)))
        _AIDS_MEMO[key] = cached
    return deepcopy(cached)


def _build_planner_authoring_aids(catalog: PolicyCatalogView) -> dict[str, Any]:
    summaries = _plugin_summaries(catalog)
    visible = _visible_plugin_names(catalog, summaries)
    aids: dict[str, Any] = {
        "purpose": (
            "Server-rendered worked exemplars and catalog digest from the live "
            "policy-visible catalog. These shapes validate against the current deployment."
        ),
    }
    custody = source_custody_exemplar_args(catalog, visible=visible)
    custody_blob_variant = source_custody_exemplar_args(catalog, blob_id=PLACEHOLDER_BLOB_ID, visible=visible)
    if custody is not None and custody_blob_variant is not None:
        aids["source_custody"] = {
            "rules": list(_SOURCE_CUSTODY_RULES),
            "set_pipeline_exemplar_inline_blob": custody,
            "existing_blob_source_binding": custody_blob_variant["source"],
        }
    fork_coalesce = fork_coalesce_exemplar_args(catalog, visible=visible)
    if fork_coalesce is not None:
        aids["fork_coalesce"] = {
            "rules": list(_FORK_COALESCE_RULES),
            "set_pipeline_exemplar": fork_coalesce,
        }
    aids["discovery_digest"] = {
        "guidance": _DISCOVERY_DIGEST_GUIDANCE,
        "plugins": discovery_digest(catalog, summaries=summaries),
    }
    return aids


__all__ = [
    "PLACEHOLDER_BLOB_ID",
    "build_planner_authoring_aids",
    "discovery_digest",
    "fork_coalesce_exemplar_args",
    "source_custody_exemplar_args",
]
