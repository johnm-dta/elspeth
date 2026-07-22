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
    RAW_HTML_CLEANUP_REVIEW_DRAFT,
    RAW_HTML_CLEANUP_USER_TERM,
    REGISTERED_PIPELINE_DECISION_USER_TERMS,
)
from elspeth.web.provider_config_policy import WEB_LLM_SEQUENTIAL_MULTI_QUERY_MAX_RETRY_SECONDS

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
    "A file the user NAMES but never uploaded, whose content is not in the "
    "conversation, has NO legal binding: source paths must resolve to real "
    "session-blob storage, so an invented path is always rejected. Discover "
    "first (list_blobs / list_composer_blobs); if nothing matches, ask one "
    "narrow upload/selection question — or, on a surface with no question "
    "channel, decline with a named gap. Never fabricate a path, blob id, or "
    "stand-in rows.",
    # run-2 G4: the digest hands a discovered blobs/<name> PATH but nothing
    # said which slot it binds to; sims put it in blob_id (UUID-typed) and died.
    "A DISCOVERED operator file path (a blobs/<name> the session's discovery "
    "facts or blob metadata handed you) binds through source options.path, "
    "copied verbatim. source.blob_id accepts ONLY the UUID a blob tool "
    "returned this session — a path in blob_id is always rejected.",
)

_INLINE_EXEMPLAR_FILENAME: Final[str] = "stock_levels.csv"
_INLINE_EXEMPLAR_MIME: Final[str] = "text/csv"
_INLINE_EXEMPLAR_CONTENT: Final[str] = "sku,on_hand\nAX-100,12\nBX-204,7\n"

_FORK_COALESCE_RULES: Final[tuple[str, ...]] = (
    # run-3 E1: the old unconditional fork-over-queries preference produced a
    # wrong-topology V-pass and 0/3 multi_query adoption where it fit. Shape
    # SELECTION is decided by what varies per assessment, not by preference.
    "SHAPE SELECTION: several assessments of the SAME input field (aspects, "
    "angles, questions about one piece of content) belong on a SINGLE llm "
    "node's queries map (multi_query) — one node, one pass, prefixed output "
    "fields. Use THIS fork/coalesce shape only when the branches take "
    "genuinely INDEPENDENT inputs or independent per-branch processing "
    "chains (different fields, different upstream transforms, different "
    "plugins per branch).",
    "When the user explicitly asks for separate nodes or one model call per "
    "branch over independent inputs, fork with a gate fanning out to one "
    "branch transform per branch, rejoined by a coalesce.",
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
    "Do not author interpretation_requirements rows for llm_prompt_template "
    "or llm_model_choice — required LLM reviews auto-stage on every llm "
    "node. Author rows only for the planner-owned kinds (vague_term wired "
    "via prompt_template_parts, registered pipeline_decision, "
    "invented_source), each in the short form {kind, user_term, draft} with "
    "user_term mandatory.",
    "Coalesce policy and merge are the engine's closed vocabularies: policy "
    "is one of require_all, quorum, best_effort, first; merge is one of "
    "union, nested, select. Use best_effort when some branches may "
    "legitimately produce no row for an id — it merges whichever branches "
    "arrive, where require_all would drop the whole row.",
    "A coalesce consumes ONLY the connections named in its branches values. "
    "Its input field is required by the schema but is not a consuming "
    "binding — set it to the first branch's arriving connection by "
    "convention, never to a name nothing publishes.",
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


def _raw_html_cleanup_rules(*, untrusted_producers: tuple[str, ...]) -> list[str]:
    """Raw-HTML cleanup review rules quoting the registered constants verbatim.

    run-2 G6: the shield draft shipped verbatim (reproduced 3/3) while the
    cleanup draft did not — sims paraphrased and the contract's marker
    recognition ("raw html" + "fingerprint" substrings) treated the row as
    absent, re-firing interpretation_review_contract_unsatisfied. Same
    imported-constants discipline as the shield rules.
    """
    producers = " or ".join(sorted(untrusted_producers))
    return [
        f"When a field_mapper with select_only=true drops {producers} raw fields "
        "(raw content / fingerprint) before the sink, stage the cleanup review ON "
        "THAT field_mapper node: one pending pipeline_decision entry in its "
        "options.interpretation_requirements (a SIBLING of mapping, never inside it).",
        f'Use exactly: {{"kind": "pipeline_decision", "user_term": "{RAW_HTML_CLEANUP_USER_TERM}", "draft": "{RAW_HTML_CLEANUP_REVIEW_DRAFT}"}} '
        "— copy the user_term and draft strings verbatim.",
        "The row is RECOGNIZED only when the draft text names both the raw HTML "
        "and the fingerprint fields — a paraphrased draft is treated as absent "
        "and the same rejection fires again.",
        "An explicit user instruction to drop the fields does NOT waive the row — it records that decision for the audit trail.",
    ]


_WEB_SCRAPE_HTTP_IDENTITY_RULES: Final[tuple[str, ...]] = (
    # run-3 E3 (mechanical half; hard-fail-vs-review doctrine is an operator
    # item — enforcement is NOT changed here).
    "http.abuse_contact and http.scraping_reason are IDENTITY CLAIMS made to "
    "remote site operators. Bind them ONLY from identities discoverable in "
    "this session — operator-provided discovery facts, session context, or "
    "the user's own words — copied verbatim.",
    "If no discoverable identity exists, OMIT the http block entirely and "
    "name the gap in metadata.description; the coded rejection for a missing "
    "required block is the correct outcome. NEVER invent a plausible "
    "identity: validation enforcement is reserved-list-only (known-reserved "
    "domains hard-fail), so a fabricated contact can pass validation and "
    "SHIP a false claim — passing the validator does not make it yours to "
    "assert.",
)


def _model_custody_rules(profile_alias: str | None) -> list[str]:
    """Model-provisioning custody with the sanctioned alternative rendered live.

    Suite run 1 G2 (8/8 problems): the pack's never-invent-a-slug rule had no
    sanctioned alternative — obeying it (omitting the model binding) was
    validator-fatal while inventing a literal slug passed. The operator-profile
    path is that alternative: ``options.profile`` binds the model through
    operator policy and exempts the ``llm_model_choice`` review
    (``interpretation_state.materialize_state_for_execution`` derives the
    exemption from the profile binding). The live alias is rendered here so the
    rule is actionable offline, not a demand for aliases the deployment does
    not serve.
    """
    rules: list[str] = []
    if profile_alias is not None:
        rules.append(
            f"This deployment serves the llm operator profile alias '{profile_alias}'. "
            "Author llm nodes with options.profile set to that alias and OMIT "
            "model/provider/credential options entirely: operator policy supplies "
            "the concrete model, and a profile-bound node carries NO "
            "llm_model_choice review card."
        )
    else:
        rules.append(
            "No llm operator profile is currently usable in this deployment: "
            "bind a model only through a literal slug that list_models served "
            "in THIS session."
        )
    rules.append(
        "Author options.model ONLY with a slug served by a list_models call — "
        "never invented, never recalled from training. A literal slug "
        "auto-stages the llm_model_choice review, which must be surfaced and "
        "resolved before the pipeline can run."
    )
    rules.append(
        "Omitting the model binding entirely is not compliance — an llm node "
        "needs either options.profile or a discovery-served options.model."
    )
    return rules


# run-2 G9: web policy rejects sequential (pool_size 1) multi_query llm nodes
# with unbounded capacity retries; the ceiling is imported so the taught
# number can never drift from provider_config_policy.
# run-3 P2 correction: the previous unconditional form of this rule steered
# PROFILE-bound authors into an operator-private option (pool_size /
# max_capacity_retry_seconds) and a profile_unavailable rejection — the
# operator layer auto-injects the web-safe retry bound on profile-bound
# multi_query nodes (profiles.lower_options). The rule is form-conditional.
_WEB_MULTI_QUERY_RETRY_RULE: Final[str] = (
    "On a PROFILE-bound llm node (options.profile), never author pacing or "
    "retry options — pool_size and max_capacity_retry_seconds are "
    "operator-private and the profile layer injects the web-safe retry bound "
    "for queries automatically. Only a provider-form llm node must bound "
    f"sequential multi_query retries itself: max_capacity_retry_seconds <= {WEB_LLM_SEQUENTIAL_MULTI_QUERY_MAX_RETRY_SECONDS} "
    "or pool_size > 1."
)

_LLM_OUTPUT_CONTRACT_RULES: Final[tuple[str, ...]] = (
    "An llm node writes the model's reply as ONE raw string into the field "
    "named by options.response_field (default llm_response). Prompt text that "
    "asks for JSON or named keys does NOT create row fields — nothing is "
    "flattened out of the reply.",
    "Downstream nodes may require only that response field (plus fields "
    "passed through from the node's input). To obtain several named result "
    "fields from one llm node, use the plugin's multi_query mechanism — its "
    "schema declares the per-query output fields, and it is the ONLY blessed "
    "multi-field shape.",
    "If a prompt asks for structured JSON anyway, the JSON arrives as one "
    "string in the response field; wire a schema-proven parser transform "
    "when downstream nodes need its keys as row fields.",
    # ── multi_query QueryDefinition contract (run-2 G2: the blessed-shape
    # mandate above shipped without the shape's contract) ─────────────────
    "queries is a mapping of query name to a query OBJECT (list form needs "
    "name in each entry). Every query object REQUIRES input_fields — a "
    "mapping of template variable name to row column name, e.g. "
    '{"field_a": "field_a", "field_b": "field_b"} — a query without '
    "input_fields is rejected.",
    "The per-query prompt key is 'template' (a Jinja2 override), NOT "
    "prompt_template. The top-level options.prompt_template is STILL "
    "required and is the fallback for any query that omits template.",
    # run-3 E2: the output contract — each query KEY prefixes its output row
    # fields, so downstream nodes can require them by exact name.
    "Each query key names its output row fields by PREFIX: the raw reply "
    "lands in <query_key>_<response_field>, and each typed output_fields "
    "entry lands in <query_key>_<suffix>. Downstream mappers and sinks "
    "reference those exact prefixed names.",
    "Sink hygiene: the auto-appended <response_field>_usage / _model audit "
    "fields ride the row automatically — do not map or require them into "
    "sinks unless the user asked for token/model reporting.",
    "on_error='discard' silently drops failed rows. When the user needs "
    "failures retained or inspected, route on_error to a dedicated "
    "quarantine sink instead of discard.",
    _WEB_MULTI_QUERY_RETRY_RULE,
)


_REVIEW_REGISTRY_RULES: Final[tuple[str, ...]] = (
    "pipeline_decision user_term values are a CLOSED registry — choose ONLY "
    "from registered_pipeline_decision_user_terms above. A minted term is "
    "unresolvable and poisons its review card.",
    "A decision outside the registry is not reviewable as a "
    "pipeline_decision: record it in metadata.description instead — never "
    "invent a new user_term for it.",
    "A registered pipeline_decision demanded by policy or the pack is NEVER "
    "waived because the user's instruction already made the decision — the "
    "row RECORDS that decision for the audit trail. User authorship changes "
    "the draft's provenance, not whether the row is staged.",
    "Do not author rows for llm_prompt_template or llm_model_choice — "
    "required LLM reviews auto-stage on every llm node. The planner-owned "
    "kinds are vague_term (wired via prompt_template_parts), registered "
    "pipeline_decision, and invented_source.",
)


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
    digest = {
        "sources": _digest_entries(summaries["source"]),
        "transforms": _digest_entries(summaries["transform"]),
        "sinks": _digest_entries(summaries["sink"]),
    }
    # run-3 E4: the llm entry carries the LIVE profile-alias enum so
    # profile-first authoring never needs a discovery round to learn the
    # aliases (they are already policy-public via the knob schema's choices).
    llm_aliases = sorted(
        alias
        for plugin_id, aliases in catalog.snapshot.usable_profile_aliases
        if plugin_id.kind == "transform" and plugin_id.name == "llm"
        for alias in aliases
    )
    if llm_aliases:
        for entry in digest["transforms"]:
            if entry["name"] == "llm":
                entry["profile_aliases"] = llm_aliases
    return digest


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

    def _branch_llm(
        node_id: str,
        branch: str,
        response_field: str,
        question: str,
        *,
        prompt_template_parts: list[dict[str, Any]] | None = None,
        interpretation_requirements: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "profile": profile_alias,
            "prompt_template": question,
            "required_input_fields": ["ticket_id", "body"],
            "response_field": response_field,
            "schema": {"mode": "observed"},
            # llm_prompt_template and llm_model_choice reviews are backend
            # auto-staged on every llm node — never hand-authored. Planner-
            # owned kinds (vague_term, registered pipeline_decision,
            # invented_source) ARE authored when the node calls for them:
            # the urgency branch below authors category semantics and must
            # stage its own wired vague_term review (run-3 E6: an exemplar
            # demonstrating an un-reviewed classification was imported as
            # precedent to skip review-staging).
        }
        if prompt_template_parts is not None:
            options["prompt_template_parts"] = prompt_template_parts
        if interpretation_requirements is not None:
            options["interpretation_requirements"] = interpretation_requirements
        return {
            "id": node_id,
            "node_type": "transform",
            "plugin": "llm",
            "input": branch,
            "on_success": f"{response_field}_done",
            "on_error": "discard",
            "options": options,
        }

    _URGENCY_RUBRIC = (
        "urgency categories: blocking = the user cannot work at all; degraded = "
        "work continues with a broken feature; routine = a question or cosmetic issue"
    )

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
                # Authored CLASSIFICATION semantics: the category set is the
                # planner's invention, so the vague_term review is staged and
                # wired below — the review-staging pattern in miniature.
                f"Classify the urgency of support ticket {{{{ row.ticket_id }}}}: {{{{ row.body }}}} using these {_URGENCY_RUBRIC}. Reply with the single category word.",
                prompt_template_parts=[
                    {"kind": "text", "text": "Classify the urgency of support ticket {{ row.ticket_id }}: {{ row.body }} using these "},
                    {"kind": "interpretation_ref", "requirement_id": "urgency_semantics_review"},
                    {"kind": "text", "text": ". Reply with the single category word."},
                ],
                interpretation_requirements=[
                    {
                        "id": "urgency_semantics_review",
                        "kind": "vague_term",
                        "user_term": "urgency",
                        "draft": _URGENCY_RUBRIC,
                    }
                ],
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
                # input is schema-required but not a consuming binding for a
                # coalesce (consumption is the branches values) — first
                # branch's arriving connection, by convention.
                "input": next(iter(coalesce_branches.values())),
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
    if "llm" in visible["transform"]:
        aids["model_custody"] = {
            "rules": _model_custody_rules(_usable_llm_profile_alias(catalog)),
        }
        aids["llm_output_contract"] = {"rules": list(_LLM_OUTPUT_CONTRACT_RULES)}
    aids["review_registry"] = {
        # Imported from interpretation_state so the taught vocabulary can
        # never drift from the resolve-time registry (52322ebe1 discipline).
        "registered_pipeline_decision_user_terms": sorted(REGISTERED_PIPELINE_DECISION_USER_TERMS),
        "rules": list(_REVIEW_REGISTRY_RULES),
    }
    visible_untrusted_producers = tuple(sorted(_UNTRUSTED_REMOTE_CONTENT_PRODUCER_PLUGINS & visible["transform"]))
    if visible_untrusted_producers and "llm" in visible["transform"]:
        aids["prompt_shield"] = {
            "rules": _prompt_shield_rules(
                shield_available=dict(catalog.snapshot.selected).get(PluginCapability.PROMPT_SHIELD) is not None,
                untrusted_producers=visible_untrusted_producers,
            ),
        }
    if visible_untrusted_producers and "field_mapper" in visible["transform"]:
        aids["raw_html_cleanup"] = {"rules": _raw_html_cleanup_rules(untrusted_producers=visible_untrusted_producers)}
    if "web_scrape" in visible["transform"]:
        aids["web_scrape_http_identity"] = {"rules": list(_WEB_SCRAPE_HTTP_IDENTITY_RULES)}
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
