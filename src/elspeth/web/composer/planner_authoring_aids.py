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

Exemplars are structural teaching, not solutions: they demonstrate wiring
(gate/fork/coalesce), custody binding, and review-row shapes in a neutral
domain deliberately disjoint from every live acceptance test. If a live
test's domain vocabulary ever appears in an exemplar, that test stops
measuring planner capability and starts measuring pack-lookup — the exemplar
has become the test's answer key and the acceptance signal is contaminated.
The rules text states principles generically and never names the exemplar's
domain fields as if they were required.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Final

from elspeth.contracts.plugin_capabilities import PluginCapability
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.schemas import PluginSummary

# The registered shield-review constants and the untrusted-producer set are the
# contract's single source of truth (interpretation_state); importing them —
# private set included — is deliberate, so the taught row can never drift.
from elspeth.web.interpretation_state import (
    _UNTRUSTED_REMOTE_CONTENT_PRODUCER_PLUGINS,
    PROMPT_SHIELD_AVAILABLE_DRAFT,
    PROMPT_SHIELD_USER_TERM,
    PROMPT_SHIELD_WARNING_DRAFT,
)

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

_INLINE_EXEMPLAR_FILENAME: Final[str] = "stock_levels.csv"
_INLINE_EXEMPLAR_MIME: Final[str] = "text/csv"
_INLINE_EXEMPLAR_CONTENT: Final[str] = "sku,on_hand\nAX-100,12\nBX-204,7\n"

_FORK_COALESCE_RULES: Final[tuple[str, ...]] = (
    "When the user asks for separate per-branch processing compared side by "
    "side (e.g. separate LLM nodes, one model call per aspect), use THIS "
    "shape — a gate fanning out to one branch transform per branch, rejoined "
    "by a coalesce — not a queries map on a single llm node.",
    "Key the coalesce branches map by FORK BRANCH NAME; each value names the "
    "connection arriving at the coalesce after that branch's transforms.",
    "When branches rejoin at a coalesce, each branch transform MUST publish "
    "the connection named in the coalesce's branches values. A branch "
    "transform must never publish to a sink — only the coalesce's downstream "
    "path reaches sinks.",
    "A coalesce publishes its merged rows under its own node id — the "
    "downstream consumer sets input to the coalesce id. Do not author "
    "on_success on a coalesce unless it routes directly to a sink.",
    "Give each branch transform its own output field (an llm node's response_field) so the union merge carries every branch's result on one row.",
    "When a branch transform is an llm node, author its interpretation_requirements in the short form {kind, user_term, draft} — user_term is mandatory; a row without it is rejected. Omitting the whole block is also legal (required reviews auto-stage).",
)

_FORK_EXEMPLAR_CONTENT: Final[str] = "ticket_id,body\nT-1001,Cannot log in since the update\nT-1002,Invoice totals look wrong\n"


def _prompt_shield_rules(*, shield_available: bool, untrusted_producers: tuple[str, ...]) -> list[str]:
    """Shield-staging rules quoting the registered review constants verbatim.

    The prompt-injection shield review is ADVISORY end-to-end (warnings only,
    excluded from the blocking contract), so no rejection code ever teaches it
    on a repair turn — these aids are the only lever. Tutorial finalizer
    battery (dim_c under-flag): the replan planner non-deterministically
    omitted the row on the scrape→summarize llm node. Constants are imported
    from ``interpretation_state`` so the taught row can never drift from the
    contract (the 52322ebe1 discipline); the draft is chosen by the LIVE
    snapshot's shield selection, mirroring the warning→available upgrade the
    server itself applies, and memoizes correctly because the aids cache is
    keyed by snapshot hash.
    """
    draft = PROMPT_SHIELD_AVAILABLE_DRAFT if shield_available else PROMPT_SHIELD_WARNING_DRAFT
    producers = " or ".join(sorted(untrusted_producers))
    return [
        f"When an llm transform consumes externally-fetched content (any path from a {producers} "
        "output reaches its input), stage the prompt-injection shield review ON THAT LLM NODE: "
        "add one pending pipeline_decision entry to its options.interpretation_requirements "
        "(a sibling of the node's other options).",
        f'Use exactly: {{"kind": "pipeline_decision", "user_term": "{PROMPT_SHIELD_USER_TERM}", "draft": "{draft}"}} '
        "— copy the user_term and draft strings verbatim.",
        "The review is advisory and never blocks the pipeline, but omitting it hides a "
        "prompt-injection exposure decision from the operator's review cards.",
        "Skip the row only when an authorized prompt-injection shield transform is already wired between the fetch step and the llm node.",
    ]


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
                    "path": "outputs/stock_levels.json",
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


def _renderable_branch_plugins(transforms: list[PluginSummary]) -> list[str]:
    """Non-LLM transforms authorable with only the universal ``schema`` option.

    The profile-less exemplar variant needs branch transforms it can configure
    generically: a plugin whose required options go beyond ``schema`` would
    demand invented values, and a batch-aware plugin authored as
    ``node_type='transform'`` needs the aggregation path (or extra row-mode
    options) the exemplar cannot generically supply. Sorted so the pick is
    deterministic per snapshot; plugin classes are static per process, so the
    batch-aware sweep cannot drift within a memo entry's lifetime.
    """
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    batch_aware = frozenset(cls.name for cls in get_shared_plugin_manager().get_transforms() if cls.is_batch_aware)
    return sorted(
        plugin.name
        for plugin in transforms
        if plugin.name != "llm"
        and plugin.name not in batch_aware
        and {field.name for field in plugin.config_fields if field.required} <= {"schema"}
    )


def fork_coalesce_exemplar_args(
    catalog: PolicyCatalogView,
    *,
    visible: Mapping[str, frozenset[str]] | None = None,
    summaries: Mapping[str, list[PluginSummary]] | None = None,
) -> dict[str, Any] | None:
    """Complete ``set_pipeline`` args for the fork -> branches -> coalesce shape.

    The operator-ruled A/B topology: a gate fans identical rows out to two
    branches, each branch runs its own transform, a coalesce rejoins them
    under ``require_all``/``union``, one cleanup transform consumes the
    coalesce id, and a sink receives the tidied rows. The WIRING is pure
    topology and renders regardless of LLM availability; only the branch
    contents vary. With a usable llm operator profile the branches are two
    llm transforms (own prompt, own ``response_field``, short-form
    interpretation_requirements); without one the SAME topology renders with
    two policy-visible non-LLM transforms picked deterministically (first two
    alphabetically whose only required option is ``schema``; a single
    candidate serves both branches under distinct node ids). Returns ``None``
    only when the fixed plugins (csv/json/field_mapper) or every branch
    candidate are policy-hidden — an exemplar must never model an invented
    identifier.
    """
    if summaries is None:
        summaries = _plugin_summaries(catalog)
    if visible is None:
        visible = _visible_plugin_names(catalog, summaries)
    if "csv" not in visible["source"] or "json" not in visible["sink"]:
        return None
    if "field_mapper" not in visible["transform"]:
        return None
    profile_alias = _usable_llm_profile_alias(catalog) if "llm" in visible["transform"] else None

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
                "required_input_fields": ["ticket_id", "body"],
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

    if profile_alias is not None:
        branch_nodes = [
            _branch_llm(
                "assess_sentiment",
                "branch_a",
                "sentiment",
                "What is the sentiment of support ticket {{ row.ticket_id }}: {{ row.body }}? Reply with one short phrase.",
            ),
            _branch_llm(
                "assess_urgency",
                "branch_b",
                "urgency",
                "Classify the urgency of support ticket {{ row.ticket_id }}: {{ row.body }}. Reply with a single category word.",
            ),
        ]
        coalesce_branches = {"branch_a": "sentiment_done", "branch_b": "urgency_done"}
        tidy_mapping = {
            "ticket_id": "ticket_id",
            "body": "body",
            "sentiment": "sentiment",
            "urgency": "urgency",
        }
        metadata = {
            "name": "Per-branch LLM assessment",
            "description": "Fan rows out to one llm transform per branch, rejoin with a coalesce, tidy, and save.",
        }
    else:
        branch_pool = _renderable_branch_plugins(summaries["transform"])
        if not branch_pool:
            return None
        plugin_a = branch_pool[0]
        plugin_b = branch_pool[1] if len(branch_pool) > 1 else branch_pool[0]
        branch_nodes = [
            {
                "id": "process_branch_a",
                "node_type": "transform",
                "plugin": plugin_a,
                "input": "branch_a",
                "on_success": "branch_a_done",
                "on_error": "discard",
                "options": {"schema": {"mode": "observed"}},
            },
            {
                "id": "process_branch_b",
                "node_type": "transform",
                "plugin": plugin_b,
                "input": "branch_b",
                "on_success": "branch_b_done",
                "on_error": "discard",
                "options": {"schema": {"mode": "observed"}},
            },
        ]
        coalesce_branches = {"branch_a": "branch_a_done", "branch_b": "branch_b_done"}
        tidy_mapping = {"ticket_id": "ticket_id", "body": "body"}
        metadata = {
            "name": "Per-branch fan-out and rejoin",
            "description": "Fan rows out to one transform per branch, rejoin with a coalesce, tidy, and save.",
        }

    return {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {
                "schema": {
                    "mode": "flexible",
                    "fields": ["ticket_id: str", "body: str"],
                    "guaranteed_fields": ["ticket_id", "body"],
                }
            },
            "on_validation_failure": "discard",
            "inline_blob": {
                "filename": "support_tickets.csv",
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
            *branch_nodes,
            {
                "id": "merge_branches",
                "node_type": "coalesce",
                "input": "branches",
                "branches": coalesce_branches,
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
                    "mapping": tidy_mapping,
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
                    "path": "outputs/ticket_assessments.json",
                    "format": "json",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
        "metadata": metadata,
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
    fork_coalesce = fork_coalesce_exemplar_args(catalog, visible=visible, summaries=summaries)
    if fork_coalesce is not None:
        aids["fork_coalesce"] = {
            "rules": list(_FORK_COALESCE_RULES),
            "set_pipeline_exemplar": fork_coalesce,
        }
    visible_untrusted_producers = tuple(sorted(_UNTRUSTED_REMOTE_CONTENT_PRODUCER_PLUGINS & visible["transform"]))
    if visible_untrusted_producers and "llm" in visible["transform"]:
        aids["prompt_shield"] = {
            "rules": _prompt_shield_rules(
                shield_available=dict(catalog.snapshot.selected).get(PluginCapability.PROMPT_SHIELD) is not None,
                untrusted_producers=visible_untrusted_producers,
            ),
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
