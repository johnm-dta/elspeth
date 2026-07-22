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

from typing import Any, Final

from elspeth.web.catalog.policy_view import PolicyCatalogView

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


def _visible_plugin_names(catalog: PolicyCatalogView) -> dict[str, frozenset[str]]:
    return {
        "source": frozenset(plugin.name for plugin in catalog.list_sources()),
        "transform": frozenset(plugin.name for plugin in catalog.list_transforms()),
        "sink": frozenset(plugin.name for plugin in catalog.list_sinks()),
    }


def source_custody_exemplar_args(
    catalog: PolicyCatalogView,
    *,
    blob_id: str | None = None,
) -> dict[str, Any] | None:
    """Complete ``set_pipeline`` args showing one legal source custody binding.

    With ``blob_id=None`` the source binds literal user data via
    ``inline_blob``; passing a ``blob_id`` (the prompt passes
    :data:`PLACEHOLDER_BLOB_ID`; the validation test passes a real created
    blob's id) shows the existing-blob binding instead. Everything outside
    ``source`` is byte-identical between the variants. Returns ``None`` when
    the plugins the exemplar names are not policy-visible.
    """
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


def build_planner_authoring_aids(catalog: PolicyCatalogView) -> dict[str, Any]:
    """Assemble the live authoring-aids payload for one planner call.

    Rendered fresh per call from the policy-visible catalog, so it can never
    drift from the deployment. Sections whose plugins are policy-hidden are
    omitted rather than rendered with invented names.
    """
    aids: dict[str, Any] = {
        "purpose": (
            "Server-rendered worked exemplars from the live policy-visible catalog. These shapes validate against the current deployment."
        ),
    }
    custody = source_custody_exemplar_args(catalog)
    custody_blob_variant = source_custody_exemplar_args(catalog, blob_id=PLACEHOLDER_BLOB_ID)
    if custody is not None and custody_blob_variant is not None:
        aids["source_custody"] = {
            "rules": list(_SOURCE_CUSTODY_RULES),
            "set_pipeline_exemplar_inline_blob": custody,
            "existing_blob_source_binding": custody_blob_variant["source"],
        }
    return aids


__all__ = [
    "PLACEHOLDER_BLOB_ID",
    "build_planner_authoring_aids",
    "source_custody_exemplar_args",
]
