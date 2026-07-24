"""Recipe scaffolding for the composer. Tier-3 boundary; see CLAUDE.md.

Slot validation runs *before* scaffolding, so a wrong-shape slot value (e.g., a
URL passed where ``blob_id`` is required) is rejected at the recipe boundary
rather than silently producing a config that fails at runtime — operators get a
diagnostic that points at the recipe input, not at a downstream plugin.

Boundary contract: recipes never read blob bytes. Resolving and inspecting blob
content lives in ``source_inspection.py``; recipes only manipulate the typed
slot values they were given.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final
from uuid import UUID

from elspeth.contracts.composer_slots import SlotSpec
from elspeth.contracts.composer_slots import SlotType as SlotType
from elspeth.contracts.freeze import freeze_fields
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    """Declares a recipe — its slot schema and the scaffold function."""

    name: str
    description: str
    slots: Mapping[str, SlotSpec]
    build: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    """Pure function: validated slots → set_pipeline-compatible args dict."""
    required_plugins: frozenset[PluginId] = frozenset()
    alternative_plugin_groups: tuple[frozenset[PluginId], ...] = ()

    def __post_init__(self) -> None:
        # ``frozen=True`` only blocks attribute reassignment; the underlying
        # mapping is mutable through the attribute reference. ``freeze_fields``
        # converts the dict to a MappingProxyType (recursively, including the
        # SlotSpec values which are themselves frozen) so registry consumers
        # cannot mutate a recipe's slot table after construction.
        freeze_fields(self, "slots")


class RecipeValidationError(ValueError):
    """Raised when operator-supplied slots fail validation."""


def _coerce_slot(name: str, spec: SlotSpec, raw: Any) -> Any:
    """Validate and coerce one slot value against its declared type."""
    if spec.slot_type == "blob_id":
        if not isinstance(raw, str):
            raise RecipeValidationError(
                f"slot '{name}' must be a UUID string for a session blob "
                f"(got type {type(raw).__name__}). To use a URL, first call "
                "create_blob with mime_type='text/plain' to wrap it."
            )
        try:
            UUID(raw)
        except ValueError as exc:
            raise RecipeValidationError(
                f"slot '{name}' must be a valid UUID (got {raw!r}). To use a "
                "URL, first call create_blob with mime_type='text/plain' to "
                "wrap it; the returned blob_id is what this slot accepts."
            ) from exc
        return raw

    if spec.slot_type == "str":
        if not isinstance(raw, str):
            raise RecipeValidationError(f"slot '{name}' must be a string (got type {type(raw).__name__})")
        return raw

    if spec.slot_type == "float":
        if isinstance(raw, bool):
            raise RecipeValidationError(f"slot '{name}' must be a number (got bool — use 0.0 or 1.0 explicitly)")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError as exc:
                raise RecipeValidationError(f"slot '{name}' must be a number; could not coerce {raw!r} to float") from exc
        raise RecipeValidationError(f"slot '{name}' must be a number (got type {type(raw).__name__})")

    if spec.slot_type == "int":
        if isinstance(raw, bool):
            raise RecipeValidationError(f"slot '{name}' must be an integer (got bool — use 0 or 1 explicitly)")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str):
            try:
                return int(raw)
            except ValueError as exc:
                raise RecipeValidationError(f"slot '{name}' must be an integer; could not coerce {raw!r}") from exc
        raise RecipeValidationError(f"slot '{name}' must be an integer (got type {type(raw).__name__})")

    if spec.slot_type == "str_list":
        # Operator-supplied list of strings. Accept only a list/tuple of
        # str entries — no string-splitting, because a single comma-separated
        # value would be ambiguous (is "a,b" one field or two?). The slot
        # caller is the LLM agent, which can construct lists natively.
        #
        # Returns a tuple, not a list: the coerced value may end up in a
        # ``SlotSpec.default`` on a ``frozen=True`` dataclass, where a
        # mutable list would silently bypass the frozen contract. Recipes
        # that need a list rebind via ``list(...)`` at the build-function
        # boundary (see _build_classify_recipe).
        if not isinstance(raw, (list, tuple)):
            raise RecipeValidationError(f"slot '{name}' must be a JSON array of strings (got type {type(raw).__name__})")
        items: list[str] = []
        for index, item in enumerate(raw):
            if not isinstance(item, str):
                raise RecipeValidationError(f"slot '{name}'[{index}] must be a string (got type {type(item).__name__})")
            items.append(item)
        return tuple(items)

    raise RecipeValidationError(f"recipe slot type {spec.slot_type!r} is not implemented")


def validate_slots(recipe: RecipeSpec, raw_slots: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a raw slots dict against a recipe's declared schema.

    Returns a new dict containing only the recipe's declared slots,
    coerced to their declared types. Raises ``RecipeValidationError``
    on missing required slots, unknown slot names, or type-coercion
    failures.
    """
    unknown = set(raw_slots) - set(recipe.slots)
    if unknown:
        raise RecipeValidationError(f"recipe '{recipe.name}' does not accept slot(s): {sorted(unknown)}. Accepted: {sorted(recipe.slots)}.")
    coerced: dict[str, Any] = {}
    for slot_name, spec in recipe.slots.items():
        if slot_name in raw_slots:
            coerced[slot_name] = _coerce_slot(slot_name, spec, raw_slots[slot_name])
        elif spec.required:
            raise RecipeValidationError(
                f"recipe '{recipe.name}' is missing required slot '{slot_name}': {spec.description or spec.slot_type}"
            )
        else:
            coerced[slot_name] = spec.default
    return coerced


# ---------------------------------------------------------------------------
# Recipe 1: classify-rows-llm-jsonl
#
#   csv source (blob)  →  llm transform (response stored in label_field)
#                       →  jsonl sink (single output)
# ---------------------------------------------------------------------------


_RECIPE1_SLOTS: Final[dict[str, SlotSpec]] = {
    "source_blob_id": SlotSpec(
        slot_type="blob_id",
        description="UUID of the operator-supplied CSV blob (use create_blob to wrap inline content first)",
    ),
    "classifier_template": SlotSpec(
        slot_type="str",
        description="Jinja2 template for the LLM prompt; reference row fields as {{ row['col'] }}",
    ),
    "profile": SlotSpec(
        slot_type="str",
        description="Opaque operator-approved LLM profile alias from the public llm schema",
    ),
    "label_field": SlotSpec(
        slot_type="str",
        required=False,
        default="classification",
        description="Row field name where the LLM response is written",
    ),
    "required_input_fields": SlotSpec(
        slot_type="str_list",
        required=False,
        default=(),
        description=(
            "Row field names the classifier_template depends on. The LLMConfig "
            "validator demands an explicit list when the template references "
            "row.* — pass the field names you reference in classifier_template, "
            "or accept the recipe default (empty list) which is the "
            "documented opt-out ('accept runtime risk') and refine later via "
            "patch_node_options."
        ),
    ),
    "output_path": SlotSpec(
        slot_type="str",
        required=False,
        default="outputs/classified.jsonl",
        description="JSONL output path",
    ),
}


def _build_classify_recipe(slots: Mapping[str, Any]) -> dict[str, Any]:
    """Build set_pipeline args for the classify-rows-llm-jsonl recipe."""
    # ``blob_id`` is a TOP-LEVEL key of ``source`` (sibling of ``options``),
    # NOT a member of ``options``. ``_execute_set_pipeline`` reads it via
    # ``src_args.get("blob_id")`` and feeds it to ``_resolve_source_blob``,
    # which authoritatively materialises ``options["path"]`` and the
    # canonical ``options["blob_ref"]``. Putting ``blob_id`` inside
    # ``options`` would skip resolution and leave the source unbound — the
    # proof step (``compute_proof_diagnostics`` reads ``options["blob_ref"]``)
    # would then silently report no diagnostics.
    required_input_fields = list(slots["required_input_fields"])
    return {
        "source": {
            "plugin": "csv",
            "blob_id": slots["source_blob_id"],
            "on_success": "rows",
            "options": {
                "schema": {"mode": "observed"},
            },
            "on_validation_failure": "discard",
        },
        "nodes": [
            {
                "id": "classifier",
                "node_type": "transform",
                "plugin": "llm",
                "input": "rows",
                "on_success": "labelled",
                "on_error": "discard",
                "options": {
                    "profile": slots["profile"],
                    "prompt_template": slots["classifier_template"],
                    "response_field": slots["label_field"],
                    "schema": {"mode": "observed", "fields": None},
                    "required_input_fields": required_input_fields,
                },
            }
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "labelled",
                "plugin": "json",
                "options": {
                    "path": slots["output_path"],
                    "format": "jsonl",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
        "metadata": {
            "name": "classify-rows-llm-jsonl",
            "description": (
                f"LLM classification of CSV rows; classification stored in field "
                f"'{slots['label_field']}', written to {slots['output_path']}"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Recipe 2: split-by-numeric-threshold
#
#   csv source (blob)  →  type_coerce (numeric field)
#                       →  gate (row[field] >= threshold)
#                       →  above_output sink + below_output sink
# ---------------------------------------------------------------------------


_RECIPE2_SLOTS: Final[dict[str, SlotSpec]] = {
    "source_blob_id": SlotSpec(
        slot_type="blob_id",
        description="UUID of the operator-supplied CSV blob",
    ),
    "field": SlotSpec(
        slot_type="str",
        description="Column to compare against the threshold (must be numeric or coercible)",
    ),
    "threshold": SlotSpec(
        slot_type="float",
        description="Numeric threshold; rows with field >= threshold go to above_output_path",
    ),
    "above_output_path": SlotSpec(
        slot_type="str",
        required=False,
        default="outputs/above.jsonl",
        description="JSONL output for rows meeting/exceeding the threshold",
    ),
    "below_output_path": SlotSpec(
        slot_type="str",
        required=False,
        default="outputs/below.jsonl",
        description="JSONL output for rows below the threshold",
    ),
}


def _build_threshold_recipe(slots: Mapping[str, Any]) -> dict[str, Any]:
    """Build set_pipeline args for the split-by-numeric-threshold recipe."""
    field = slots["field"]
    threshold = slots["threshold"]
    return {
        "source": {
            "plugin": "csv",
            "blob_id": slots["source_blob_id"],
            "on_success": "rows",
            "options": {
                "schema": {"mode": "observed"},
            },
            "on_validation_failure": "discard",
        },
        "nodes": [
            {
                "id": "coerce_numeric",
                "node_type": "transform",
                "plugin": "type_coerce",
                "input": "rows",
                "on_success": "numeric_rows",
                "on_error": "discard",
                "options": {
                    # type_coerce extends DataPluginConfig, which makes
                    # ``schema`` a required field. Recipes use observed
                    # mode so any input columns flow through; the operator
                    # can refine to a fixed schema via patch_node_options
                    # once inspect_source has surfaced the actual headers.
                    "schema": {"mode": "observed"},
                    "conversions": [{"field": field, "to": "float"}],
                },
            },
            {
                "id": "threshold_gate",
                "node_type": "gate",
                "input": "numeric_rows",
                "condition": f"row[{field!r}] >= {threshold}",
                "routes": {"true": "above", "false": "below"},
            },
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "above",
                "plugin": "json",
                "options": {
                    "path": slots["above_output_path"],
                    "format": "jsonl",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
            {
                "sink_name": "below",
                "plugin": "json",
                "options": {
                    "path": slots["below_output_path"],
                    "format": "jsonl",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
        ],
        "metadata": {
            "name": "split-by-numeric-threshold",
            "description": (
                f"CSV rows split by {field} >= {threshold}; above → {slots['above_output_path']}, below → {slots['below_output_path']}"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Recipe 3: fork-coalesce-truncate-jsonl
#
#   csv source (blob)  →  fork gate (routes:{all:fork}, fork_to:[a, b])
#                       →  passthrough (path A)        + truncate (path B)
#                       →  coalesce (merge=nested, {key_a:a_out, key_b:b_out})
#                       →  jsonl sink (one merged output)
#
# Wiring discipline (gate.fork_to ↔ path.input/on_success ↔ coalesce.branches)
# is encoded once here so the LLM agent never has to maintain it. Slot-fillable
# axes: which CSV blob, which field to truncate, max length, suffix, output
# path, and the two top-level merge keys. Path A is fixed as ``passthrough``
# because the canonical use case is "keep the original row alongside a
# transformed copy"; alternative path-A transforms would be a different
# recipe.
# ---------------------------------------------------------------------------


_RECIPE3_SLOTS: Final[dict[str, SlotSpec]] = {
    "source_blob_id": SlotSpec(
        slot_type="blob_id",
        description="UUID of the operator-supplied CSV blob (use create_blob to wrap inline content first)",
    ),
    "truncate_field": SlotSpec(
        slot_type="str",
        description="Name of the row field that path B truncates (e.g., 'description'). Path A leaves the row unchanged.",
    ),
    "max_chars": SlotSpec(
        slot_type="int",
        description=(
            "Maximum length of the truncated field on path B (suffix counts toward this length, "
            "so it must be strictly greater than the suffix length)."
        ),
    ),
    "truncation_suffix": SlotSpec(
        slot_type="str",
        required=False,
        default="...",
        description="Suffix appended when truncation occurs on path B (e.g., '...').",
    ),
    "output_path": SlotSpec(
        slot_type="str",
        required=False,
        default="outputs/merged.jsonl",
        description="JSONL output path for the merged rows.",
    ),
    "key_a": SlotSpec(
        slot_type="str",
        required=False,
        default="path_a",
        description="Top-level field in each merged output row that holds the unchanged-path row body.",
    ),
    "key_b": SlotSpec(
        slot_type="str",
        required=False,
        default="path_b",
        description="Top-level field in each merged output row that holds the truncated-path row body.",
    ),
}


def _build_fork_coalesce_truncate_recipe(slots: Mapping[str, Any]) -> dict[str, Any]:
    """Build set_pipeline args for the fork-coalesce-truncate-jsonl recipe.

    Path A is ``passthrough`` (row unchanged); path B is ``truncate`` with the
    operator-named field clipped to ``max_chars`` (with optional suffix). The
    coalesce node merges both paths under operator-supplied keys via
    ``merge: nested``, so the output rows are ``{key_a: <full row>, key_b:
    <truncated row>}``.

    Wiring discipline: ``coalesce.branches`` is mapping-form:
    ``{branch_name: input_connection}``. The branch names are the operator-
    supplied ``key_a`` / ``key_b`` values (and become nested output keys);
    the input connections are the post-transform path outputs. This is the
    runtime-required representation for transformed fork branches.
    """
    key_a = slots["key_a"]
    key_b = slots["key_b"]
    truncate_field = slots["truncate_field"]
    max_chars = slots["max_chars"]
    suffix = slots["truncation_suffix"]
    branch_a_output = f"{key_a}_out"
    branch_b_output = f"{key_b}_out"
    return {
        "source": {
            "plugin": "csv",
            "blob_id": slots["source_blob_id"],
            "on_success": "rows",
            "options": {
                "schema": {"mode": "observed"},
            },
            "on_validation_failure": "discard",
        },
        "nodes": [
            {
                "id": "fork_gate",
                "node_type": "gate",
                "input": "rows",
                # validate_boolean_routes contract: boolean predicates require
                # "true"/"false" labels. This recipe intentionally returns the
                # string literal "all" so the single route label is runtime-valid
                # while still forking every row.
                "condition": "'all'",
                "routes": {"all": "fork"},
                # fork_to publishes one connection per branch. The branch names
                # are the user-visible coalesce output keys; the branch mapping
                # below points each key at the post-transform connection that the
                # coalesce node consumes.
                "fork_to": [key_a, key_b],
            },
            {
                "id": "path_a_passthrough",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": key_a,
                "on_success": branch_a_output,
                "on_error": "discard",
                "options": {
                    "schema": {"mode": "observed"},
                },
            },
            {
                "id": "path_b_truncate",
                "node_type": "transform",
                "plugin": "truncate",
                "input": key_b,
                "on_success": branch_b_output,
                "on_error": "discard",
                "options": {
                    "schema": {"mode": "observed"},
                    "fields": {truncate_field: max_chars},
                    "suffix": suffix,
                },
            },
            {
                "id": "merge_paths",
                "node_type": "coalesce",
                # ``input`` is required by NodeSpec for every node, but the
                # producer-resolver special-cases coalesce (it walks
                # ``branches`` for routing, not ``input``). The literal
                # sentinel ``"branches"`` is the established convention
                # (see tests/unit/web/composer/test_producer_resolver.py).
                "input": "branches",
                # Mapping form: branch names are the nested output keys, while
                # values are the post-transform connections consumed by coalesce.
                "branches": {key_a: branch_a_output, key_b: branch_b_output},
                "policy": "require_all",
                "merge": "nested",
                "on_success": "merged_rows",
                "on_error": "discard",
                "options": {"schema": {"mode": "observed"}},
            },
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "merged_rows",
                "plugin": "json",
                "options": {
                    "path": slots["output_path"],
                    "format": "jsonl",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
        "metadata": {
            "name": "fork-coalesce-truncate-jsonl",
            "description": (
                f"Fork+coalesce: each row produces one merged output row with "
                f"'{key_a}' (unchanged) and '{key_b}' (field '{truncate_field}' "
                f"truncated to {max_chars} chars with suffix {suffix!r}); "
                f"written to {slots['output_path']}"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Recipe 4: web-scrape-llm-rate-jsonl (D11)
#
#   json/csv URL-row source (blob)  →  web_scrape (fetch page content)
#                                    →  llm (rate the page)
#                                    →  field_mapper(select_only) cleanup
#                                    →  jsonl sink (single output)
#
# web_scrape is a TRANSFORM, not a source: the head source is a json/csv blob
# of {url: ...} rows. The field_mapper drops the raw scraped content/fingerprint
# (data minimization) and stages the kind=pipeline_decision raw-HTML cleanup
# requirement so the blocking cleanup contract (raw_html_cleanup_review_contract_error,
# interpretation_state.py) passes deterministically.
#
# Prompt-injection shield (rev 4, re-polarized): the recipe OMITS an unbuildable
# azure_prompt_shield hard node (the composer cannot instantiate it without
# configured endpoint+api_key secrets — elspeth-abb2cb0931, a CONDITIONAL
# security ticket, NOT a licence to remove all shield signal). It does NOT
# suppress the existing medium-severity prompt-shield advisory warning
# (prompt_shield_recommendation_warning_pairs), which surfaces at the wire
# stage from validate(). See test_no_azure_prompt_shield_hard_node AND the P4.3
# advisory-presence test.
# ---------------------------------------------------------------------------


_RECIPE_WEB_SCRAPE_SLOTS: Final[dict[str, SlotSpec]] = {
    "source_blob_id": SlotSpec(
        slot_type="blob_id",
        description="UUID of the operator-supplied URL-list blob (json/csv rows of {url: ...}; use create_blob to wrap inline content first)",
    ),
    "source_plugin": SlotSpec(
        slot_type="str",
        description="Resolved URL-row source plugin carried from match_recipe; must be 'json' or 'csv'. Direct apply callers pass the materialised source plugin explicitly.",
    ),
    "profile": SlotSpec(
        slot_type="str",
        description="Opaque operator-approved LLM profile alias",
    ),
    "rating_template": SlotSpec(
        slot_type="str",
        required=False,
        default="Rate the appeal of this government web page from 1-10 and explain briefly:\n\n{{ row['content'] }}",
        description="Jinja2 template for the rating prompt; reference scraped content as {{ row['content'] }}",
    ),
    "abuse_contact": SlotSpec(
        slot_type="str",
        description=(
            "Operator-owned monitored contact address sent in web_scrape HTTP metadata. "
            "If the public-fetch request omits it and no deployment/tool identity is visible, "
            "use abuse-contact-unset@elspeth.foundryside.dev and surface that default after "
            "the recipe/tool call with a pipeline_decision interpretation review."
        ),
    ),
    "scraping_reason": SlotSpec(
        slot_type="str",
        description=(
            "Operator-authored reason for scraping, sent in web_scrape HTTP metadata. "
            "If the public-fetch request omits it, derive an explicit reason from the user's "
            "requested public fetch and surface that default after the recipe/tool call with "
            "a pipeline_decision interpretation review."
        ),
    ),
    "output_path": SlotSpec(
        slot_type="str",
        required=False,
        default="outputs/ratings.jsonl",
        description="JSONL output path",
    ),
    "allowed_hosts": SlotSpec(
        slot_type="str_list",
        required=False,
        # Tuple default (not []): recipes.py warns a mutable list default would
        # silently bypass the frozen contract; the sibling str_list slot
        # required_input_fields uses default=() (recipes.py:190). _coerce_slot
        # (recipes.py:51) returns tuple(items) for a SUPPLIED str_list; an omitted
        # slot gets spec.default verbatim (recipes.py:142) — () is already a tuple.
        default=(),
        description=(
            "SSRF allowlist for the web_scrape node, as a list of CIDR strings. "
            "Empty (the default) omits the key so the web_scrape field default "
            "'public_only' applies — the correct value for a public host, including "
            "the tutorial's publicly-hosted synthetic pages. SSRF safety comes "
            "from the web_scrape enforcement boundary (CidrStr validation + the "
            "'public_only' field default), not from the slot being unreachable — "
            "apply_pipeline_recipe is a Tier-3 boundary that can forward an "
            "LLM-authored value, which the enforcement boundary still constrains."
        ),
    ),
}


def _build_web_scrape_recipe(slots: Mapping[str, Any]) -> dict[str, Any]:
    """Build set_pipeline args for the web-scrape-llm-rate-jsonl recipe.

    Emits source → web_scrape → llm → field_mapper(cleanup) → jsonl, named by
    connection labels (NOT EdgeSpec objects — guided passes edges=[]). The
    field_mapper drops the raw scraped content/fingerprint and stages the
    kind=pipeline_decision raw-HTML cleanup requirement so the blocking cleanup
    contract passes. The unbuildable azure_prompt_shield hard node is omitted
    (elspeth-abb2cb0931); the existing medium-severity prompt-shield advisory
    is left to fire from validate() — the recipe MUST NOT suppress it.
    """
    # Function-level imports keep the recipe builder's dependency boundary
    # narrow and avoid circular imports with the tools plane.
    from elspeth.contracts.composer_interpretation import InterpretationKind
    from elspeth.web.composer.tools._common import _pending_interpretation_requirement
    from elspeth.web.interpretation_state import (
        INTERPRETATION_REQUIREMENTS_KEY,
        RAW_HTML_CLEANUP_REVIEW_DRAFT,
        RAW_HTML_CLEANUP_USER_TERM,
    )

    content_field = "content"
    fingerprint_field = "content_fingerprint"
    cleanup_requirement = _pending_interpretation_requirement(
        requirement_id="drop_raw_html_review",
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term=RAW_HTML_CLEANUP_USER_TERM,
        draft=RAW_HTML_CLEANUP_REVIEW_DRAFT,
    )
    web_scrape_options: dict[str, Any] = {
        "schema": {"mode": "observed"},
        "url_field": "url",
        "content_field": content_field,
        "fingerprint_field": fingerprint_field,
        "format": "markdown",
        "http": {
            # OPERATOR: these values are visible to scraped third
            # parties. They are required slots, not tutorial
            # defaults: use a monitored operator-owned inbox and an
            # accurate reason, or do not apply the recipe.
            "abuse_contact": slots["abuse_contact"],
            "scraping_reason": slots["scraping_reason"],
        },
    }
    allowed_hosts = slots["allowed_hosts"]
    if allowed_hosts:
        # SSRF allowlist supplied by the deterministic seam (tutorial loopback
        # CIDR). Empty -> omitted -> the web_scrape field default public_only.
        # The allowlist is a field of ``WebScrapeHTTPConfig`` (web_scrape.py),
        # so it MUST nest under ``http`` beside abuse_contact/scraping_reason —
        # a top-level key is rejected by the plugin (extra:forbid).
        web_scrape_options["http"]["allowed_hosts"] = list(allowed_hosts)
    return {
        "source": {
            "plugin": slots["source_plugin"],
            "blob_id": slots["source_blob_id"],
            "on_success": "rows",
            "options": {
                "schema": {"mode": "observed"},
            },
            "on_validation_failure": "discard",
        },
        "nodes": [
            {
                "id": "url_rows",
                "node_type": "transform",
                "plugin": "web_scrape",
                "input": "rows",
                "on_success": "scraped",
                "on_error": "discard",
                "options": web_scrape_options,
            },
            {
                "id": "rate_pages",
                "node_type": "transform",
                "plugin": "llm",
                "input": "scraped",
                "on_success": "rated",
                "on_error": "discard",
                "options": {
                    "profile": slots["profile"],
                    "prompt_template": slots["rating_template"],
                    "response_field": "rating",
                    "schema": {"mode": "observed"},
                    "required_input_fields": [content_field],
                },
            },
            {
                "id": "drop_raw_html",
                "node_type": "transform",
                "plugin": "field_mapper",
                "input": "rated",
                "on_success": "clean",
                "on_error": "discard",
                "options": {
                    "schema": {"mode": "observed"},
                    "select_only": True,
                    # mapping preserves ONLY the user-facing fields; the raw
                    # content/fingerprint are intentionally absent (dropped).
                    "mapping": {
                        "url": "url",
                        "rating": "rating",
                    },
                    INTERPRETATION_REQUIREMENTS_KEY: [cleanup_requirement],
                },
            },
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "clean",
                "plugin": "json",
                "options": {
                    "path": slots["output_path"],
                    "format": "jsonl",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
        "metadata": {
            "name": "web-scrape-llm-rate-jsonl",
            "description": (
                f"Scrape each URL, rate the page with an LLM, drop the raw HTML/fingerprint, and write ratings to {slots['output_path']}"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_RECIPES: Final[dict[str, RecipeSpec]] = {
    "classify-rows-llm-jsonl": RecipeSpec(
        name="classify-rows-llm-jsonl",
        description=(
            "Apply an LLM classifier to every row of a CSV blob and write a "
            "JSONL output with each row's classification stored in a named "
            "field. Use for: 'classify these tickets as high/medium/low', "
            "'tag these reviews as positive/negative', 'pick a category for "
            "each row'. The CSV must already be uploaded as a session blob."
        ),
        slots=_RECIPE1_SLOTS,
        build=_build_classify_recipe,
        required_plugins=frozenset({PluginId("source", "csv"), PluginId("transform", "llm"), PluginId("sink", "json")}),
    ),
    "split-by-numeric-threshold": RecipeSpec(
        name="split-by-numeric-threshold",
        description=(
            "Split CSV rows by a numeric threshold into two JSONL outputs. "
            "Coerces the field to float before comparison so a string-typed "
            "CSV column is handled correctly. Use for: 'route prices above "
            "100 to high.jsonl', 'separate scores >= 0.8 from the rest', "
            "'split orders by amount'."
        ),
        slots=_RECIPE2_SLOTS,
        build=_build_threshold_recipe,
        required_plugins=frozenset({PluginId("source", "csv"), PluginId("transform", "type_coerce"), PluginId("sink", "json")}),
    ),
    "fork-coalesce-truncate-jsonl": RecipeSpec(
        name="fork-coalesce-truncate-jsonl",
        description=(
            "Fork+coalesce: process each CSV row two ways in parallel and "
            "merge into a single output row. Path A keeps the row unchanged; "
            "path B truncates a named field to a maximum length (with optional "
            "suffix). The merged output row exposes both paths as named "
            "top-level fields. Use for: 'process each row two ways and combine', "
            "'keep the original alongside a truncated copy', 'fan out then "
            "rejoin under separate keys'. Wiring (gate.fork_to ↔ path.on_success "
            "↔ coalesce.branches naming invariants) is server-side and not the "
            "agent's responsibility."
        ),
        slots=_RECIPE3_SLOTS,
        build=_build_fork_coalesce_truncate_recipe,
        required_plugins=frozenset(
            {
                PluginId("source", "csv"),
                PluginId("transform", "passthrough"),
                PluginId("transform", "truncate"),
                PluginId("sink", "json"),
            }
        ),
    ),
    "web-scrape-llm-rate-jsonl": RecipeSpec(
        name="web-scrape-llm-rate-jsonl",
        description=(
            "Fetch each URL in a blob of {url: ...} rows, rate the page with an "
            "LLM, drop the raw scraped HTML and fingerprint, and write a JSONL "
            "output of url + rating. Use for: 'scrape these pages and rate them', "
            "'fetch each site and score it'. The URL list must already be uploaded "
            "as a session blob (json or csv rows with a url column). The resolved "
            "source_plugin slot preserves whether the materialised source is json "
            "or csv. The raw-HTML cleanup is staged as a pipeline_decision so the "
            "data-minimization contract passes deterministically."
        ),
        slots=_RECIPE_WEB_SCRAPE_SLOTS,
        build=_build_web_scrape_recipe,
        required_plugins=frozenset(
            {
                PluginId("transform", "web_scrape"),
                PluginId("transform", "llm"),
                PluginId("transform", "field_mapper"),
                PluginId("sink", "json"),
            }
        ),
        alternative_plugin_groups=(frozenset({PluginId("source", "csv"), PluginId("source", "json")}),),
    ),
}


def unavailable_recipe_plugin(
    recipe: RecipeSpec,
    snapshot: PluginAvailabilitySnapshot,
    *,
    raw_slots: Mapping[str, Any] | None = None,
) -> PluginId | None:
    """Return the first unavailable dependency, or ``None`` when usable."""
    for plugin_id in sorted(recipe.required_plugins):
        if plugin_id not in snapshot.available:
            return plugin_id
    if raw_slots is not None and isinstance(raw_slots.get("source_plugin"), str):
        try:
            selected_source = PluginId("source", raw_slots["source_plugin"])
        except ValueError as exc:
            raise RecipeValidationError("recipe source_plugin must be a registered source plugin id") from exc
        if selected_source not in snapshot.available:
            return selected_source
    if "profile" in recipe.slots:
        llm_id = PluginId("transform", "llm")
        usable_aliases = dict(snapshot.usable_profile_aliases).get(llm_id, ())
        if not usable_aliases:
            return llm_id
        if raw_slots is not None and isinstance(raw_slots.get("profile"), str) and raw_slots["profile"] not in usable_aliases:
            return llm_id
    for alternatives in recipe.alternative_plugin_groups:
        if alternatives.isdisjoint(snapshot.available):
            return min(alternatives)
    return None


def list_recipes(snapshot: PluginAvailabilitySnapshot) -> list[dict[str, Any]]:
    """Return discovery metadata for recipes usable in this snapshot."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "slots": {
                slot_name: {
                    "type": s.slot_type,
                    "required": s.required,
                    "default": s.default,
                    "description": s.description,
                    **(
                        {"choices": list(dict(snapshot.usable_profile_aliases).get(PluginId("transform", "llm"), ()))}
                        if slot_name == "profile"
                        else {}
                    ),
                }
                for slot_name, s in spec.slots.items()
            },
        }
        for spec in _RECIPES.values()
        if unavailable_recipe_plugin(spec, snapshot) is None
    ]


def get_recipe(name: str) -> RecipeSpec | None:
    """Return a recipe spec by name, or None if not registered."""
    # ``name`` is external (composer-LLM-authored); "no such recipe" is a real
    # answer, so ``None`` is honest absence, not a fabricated default. Explicit
    # membership keeps that absence signal structural rather than a swallow.
    if name in _RECIPES:
        return _RECIPES[name]
    return None


def apply_recipe(name: str, raw_slots: Mapping[str, Any]) -> dict[str, Any]:
    """Validate slots and return the set_pipeline args for a recipe.

    Raises ``RecipeValidationError`` if the recipe is unknown or the
    slots fail validation. The returned dict is consumable directly by
    ``set_pipeline``.
    """
    # ``name`` is external (composer-LLM-authored). Convert an unknown-recipe
    # KeyError directly into the typed RecipeValidationError the caller routes,
    # preserving the exception chain.
    try:
        recipe = _RECIPES[name]
    except KeyError as exc:
        raise RecipeValidationError(f"recipe '{name}' is not registered. Available recipes: {sorted(_RECIPES)}.") from exc
    coerced = validate_slots(recipe, raw_slots)
    # Concrete recipe builders return dict; the Mapping return type on the
    # RecipeSpec contract is the looser superset (Mapping ⊇ dict). Convert
    # to the concrete dict the caller (set_pipeline executor) requires.
    return dict(recipe.build(coerced))


@cache  # Process-scoped: module source on disk is immutable for the process lifetime.
def recipe_catalog_content_hash() -> str:
    """Hex SHA-256 over recipes.py byte content.

    Cache input #5 of the tutorial run-cache key (C2). The recipe registry +
    builders here author the cached pipeline's option-level content (provider,
    model, prompt_template, response_field, schema mode, output format).
    ``_state_matches_cached_topology`` is option-blind by design and cannot
    catch that drift, so option fidelity is guaranteed by keying this module's
    source here.
    """
    recipes_path = Path(__file__)
    digest = hashlib.sha256()
    digest.update(recipes_path.read_bytes())
    return digest.hexdigest()
