"""Generate panel-cohort scenario criteria from an example's settings.yaml.

Reads `examples/<name>/settings.yaml`, extracts the structural shape of the
pipeline (source plugin, transform chain, gate/coalesce/aggregation presence,
sink plugins and count), and emits a dict that fits the
`evals.lib.composer_rgr_score` scoring contract:

  {
    "structural_target": {...},      # human-readable extraction summary
    "red_criteria":  {...},          # standard passivity + build-failure sentinels
    "green_criteria": {...},         # derived from structural_target
  }

Usage as a module:

    from evals.lib.scenario_from_example import build_criteria_from_example
    criteria = build_criteria_from_example(pathlib.Path("examples/boolean_routing"))

CLI for smoke-testing extraction:

    python3 -m evals.lib.scenario_from_example examples/boolean_routing

Design notes:
  * Permissive matching. Examples list literal plugin kinds (`csv`, `llm`,
    `rag_retrieval`, etc.); composer-produced pipelines may use lexically
    different but semantically equivalent kinds (`csv_source` vs `csv`,
    `text_source` for a URL-driven path). Substring matching on a small list
    of "shape tokens" is generous enough to catch close variants without
    over-fitting to exact plugin kind strings.

  * One-canonical-shape extraction. We do NOT attempt to enumerate all
    semantically equivalent shapes; we extract the shape this example
    declares and let the composer match it. If the composer chooses an
    alternative valid shape, calibration on the smoke cohort tells us
    whether to broaden the criteria (e.g. add `web_scrape` as an
    alternative to `csv` for URL-driven inputs) or whether the composer
    should be steered toward the example's shape via prompt design.

  * Schema fields → observed_columns. When a source declares fixed/observed
    schema fields, we surface them so a `must_include_observed_columns`
    criterion can be added per scenario. The convergence-suite uses this
    to pin the composer to the actual CSV columns rather than a guess.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Standard red criteria — copy the convergence-suite pattern verbatim so the
# panel cohort and the convergence suite share the same passivity/sentinel
# definitions. Edits here propagate to every panel scenario.
# ---------------------------------------------------------------------------

PASSIVITY_PHRASES: tuple[str, ...] = (
    "if you want, i can",
    "if you'd like, i can",
    "should i ",
    "do you want me to",
    "would you like me to",
    "shall i ",
    "let me know if",
)

BUILD_FAILURE_SENTINELS: tuple[str, ...] = (
    "i cannot mark this pipeline complete",
    "runtime preflight failed",
)

# ---------------------------------------------------------------------------
# Shape-token extraction
# ---------------------------------------------------------------------------


def _shape_token(plugin: str | None) -> str | None:
    """Reduce a plugin kind to a short shape token for substring matching.

    The shape token is intentionally short (1-2 lexemes) so the green criteria
    match composer-produced kinds that may include suffixes (`csv_source`,
    `csv_sink`) or pack prefixes (`openrouter_llm`, `azure_content_safety`).

    Returns None for plugin names we don't have a mapping for; the caller
    decides whether to surface them as a structural_target literal or to
    drop them from the criteria entirely.
    """
    if not isinstance(plugin, str) or not plugin:
        return None
    p = plugin.lower()
    # Sources / sinks
    if p in {"csv"}:
        return "csv"
    if p in {"json", "jsonl"}:
        return "json"
    if p in {"text"}:
        return "text"
    if p == "blob":
        return "blob"
    if p == "database":
        return "database"
    if p == "landscape_journal":
        return "landscape_journal"
    # Transforms
    if "rag" in p or "chroma" in p:
        return "rag"
    if p in {"web_scrape"} or "web_scrape" in p:
        return "web_scrape"
    if "llm" in p:
        return "llm"
    if p == "type_coerce":
        return "type_coerce"
    if p == "value_transform":
        return "value_transform"
    if p == "truncate":
        return "truncate"
    if "explode" in p:
        return p  # line_explode, json_explode — already short
    if p == "deaggregation":
        return "deaggregation"
    if p == "checkpoint":
        return "checkpoint"
    # Aggregations / batch
    if "batch" in p or p == "stats":
        return "batch"
    # Content safety / moderation
    if "content_safety" in p or "moderation" in p:
        return "content_safety"
    # Default: emit the plugin name as-is — caller can decide whether to use it
    return p


def _extract_source(yaml_doc: dict[str, Any]) -> dict[str, Any]:
    """Pull source kind + path + observed schema fields."""
    src = yaml_doc.get("source") or {}
    plugin = src.get("plugin")
    options = src.get("options") or {}
    schema = options.get("schema") or {}
    fields_raw = schema.get("fields") or []

    # `fields` entries are strings of the form "column: type" — strip type for
    # the column-name list.
    columns = []
    for f in fields_raw:
        if isinstance(f, str):
            name = f.split(":", 1)[0].strip()
            if name:
                columns.append(name)

    return {
        "plugin": plugin,
        "shape_token": _shape_token(plugin),
        "path": options.get("path"),
        "schema_mode": schema.get("mode"),
        "columns": columns,
    }


def _extract_transforms(yaml_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull ordered transform chain (transforms[].plugin)."""
    transforms = yaml_doc.get("transforms") or []
    out = []
    for t in transforms:
        if not isinstance(t, dict):
            continue
        plugin = t.get("plugin")
        out.append(
            {
                "name": t.get("name"),
                "plugin": plugin,
                "shape_token": _shape_token(plugin),
            }
        )
    return out


def _extract_gates(yaml_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull gates — top-level list. Detect fork (fork_to present)."""
    gates = yaml_doc.get("gates") or []
    out = []
    for g in gates:
        if not isinstance(g, dict):
            continue
        forks_to = g.get("fork_to") or []
        out.append(
            {
                "name": g.get("name"),
                "condition": g.get("condition"),
                "is_fork": bool(forks_to),
                "fork_paths": list(forks_to) if isinstance(forks_to, list) else [],
                "routes": dict(g.get("routes") or {}),
            }
        )
    return out


def _extract_aggregations(yaml_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull aggregations — top-level list with .plugin."""
    aggs = yaml_doc.get("aggregations") or []
    out = []
    for a in aggs:
        if not isinstance(a, dict):
            continue
        plugin = a.get("plugin")
        out.append(
            {
                "name": a.get("name"),
                "plugin": plugin,
                "shape_token": _shape_token(plugin),
            }
        )
    return out


def _extract_coalesce(yaml_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull coalesce nodes — top-level list with .name (no plugin field)."""
    coa = yaml_doc.get("coalesce") or []
    out = []
    for c in coa:
        if not isinstance(c, dict):
            continue
        out.append({"name": c.get("name")})
    return out


def _extract_sinks(yaml_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull sinks — top-level *dict* keyed by sink name; each value has .plugin.

    Some examples may also use `outputs:` instead of `sinks:`. Handle both —
    the live composer supports either; our extractor accepts whichever the
    example uses. If both exist, sinks wins (it's the canonical key in
    contemporary settings.yamls).
    """
    sinks_dict = yaml_doc.get("sinks") or yaml_doc.get("outputs") or {}
    if not isinstance(sinks_dict, dict):
        return []
    out = []
    for name, body in sinks_dict.items():
        if not isinstance(body, dict):
            continue
        plugin = body.get("plugin")
        out.append(
            {
                "name": name,
                "plugin": plugin,
                "shape_token": _shape_token(plugin),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Settings file selection
# ---------------------------------------------------------------------------


def _pick_settings_yaml(example_dir: pathlib.Path, variant: str | None) -> pathlib.Path:
    """Return the settings.yaml path for an example.

    Some examples ship multiple `settings_*.yaml` variants
    (fork_coalesce, statistical_batch_plugins, openrouter_*).
    `variant=None` picks `settings.yaml` (the default); a non-None variant
    picks `settings_<variant>.yaml`.
    """
    if variant:
        candidate = example_dir / f"settings_{variant}.yaml"
    else:
        candidate = example_dir / "settings.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"settings yaml not found: {candidate}")
    return candidate


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_structural_target(
    example_dir: pathlib.Path,
    variant: str | None = None,
) -> dict[str, Any]:
    """Read settings.yaml and emit the human-readable shape extraction.

    This is the load-bearing extractor. The output is a dict suitable both
    for the scenario.json `structural_target` field (review aid for the
    operator) and for green_criteria derivation by `build_criteria(...)`.
    """
    settings_path = _pick_settings_yaml(example_dir, variant)
    yaml_doc = yaml.safe_load(settings_path.read_text()) or {}
    if not isinstance(yaml_doc, dict):
        raise ValueError(f"{settings_path} did not parse to a mapping")

    return {
        "example_name": example_dir.name,
        "variant": variant,
        "settings_path": str(settings_path),
        "source": _extract_source(yaml_doc),
        "transforms": _extract_transforms(yaml_doc),
        "gates": _extract_gates(yaml_doc),
        "aggregations": _extract_aggregations(yaml_doc),
        "coalesce_nodes": _extract_coalesce(yaml_doc),
        "sinks": _extract_sinks(yaml_doc),
    }


# ---------------------------------------------------------------------------
# Green-criteria derivation
# ---------------------------------------------------------------------------


def _ordered_chain_tokens(target: dict[str, Any]) -> list[str]:
    """Compose the ordered shape-token chain: source → transforms → gates/agg/coalesce → sinks.

    Drops None tokens (plugins with no mapping). Preserves order. Collapses
    consecutive duplicates (e.g. multiple sinks with the same shape token
    only contribute one occurrence to the chain — count is captured
    separately by `must_have_outputs_min`).
    """
    chain: list[str] = []

    def push(tok: str | None) -> None:
        if not tok:
            return
        if not chain or chain[-1] != tok:
            chain.append(tok)

    push(target["source"].get("shape_token"))
    for t in target["transforms"]:
        push(t.get("shape_token"))
    for g in target["gates"]:
        push("fork" if g.get("is_fork") else "gate")
    for a in target["aggregations"]:
        push(a.get("shape_token") or "batch")
    for _c in target["coalesce_nodes"]:
        push("coalesce")
    # Collapse all sinks into a single representative token. Sinks vary in
    # type but the chain assertion is "the pipeline ends with at least one
    # sink"; the count is in `must_have_outputs_min`. Pick the most common
    # sink shape token.
    sink_tokens = [s.get("shape_token") for s in target["sinks"] if s.get("shape_token")]
    if sink_tokens:
        # Use the first sink's shape token as the chain endpoint — order in
        # the dict roughly tracks declaration order in YAML.
        push(sink_tokens[0])

    return chain


def _required_kind_set(target: dict[str, Any]) -> list[str]:
    """The flat set of shape tokens that must be present somewhere in the pipeline.

    Same union as the chain but as an unordered set; suitable for
    `must_have_node_kinds_substring_any_of` where order isn't load-bearing.
    """
    tokens: list[str] = []

    def add(tok: str | None) -> None:
        if tok and tok not in tokens:
            tokens.append(tok)

    add(target["source"].get("shape_token"))
    for t in target["transforms"]:
        add(t.get("shape_token"))
    for g in target["gates"]:
        add("fork" if g.get("is_fork") else "gate")
    for a in target["aggregations"]:
        add(a.get("shape_token") or "batch")
    for _c in target["coalesce_nodes"]:
        add("coalesce")
    for s in target["sinks"]:
        add(s.get("shape_token"))
    return tokens


def build_criteria_from_target(target: dict[str, Any]) -> dict[str, Any]:
    """Compose red + green criteria dicts from an extracted structural target.

    Output shape matches `evals.lib.composer_rgr_score.score(...)` expectations.
    """
    sink_count = len(target["sinks"])
    chain = _ordered_chain_tokens(target)
    required = _required_kind_set(target)

    green: dict[str, Any] = {
        "must_be_valid": True,
        "must_not_contain_passivity": True,
        "must_have_outputs_min": max(1, sink_count),
    }
    if required:
        green["must_have_node_kinds_substring_any_of"] = [required]
    if len(chain) >= 2:
        # Only assert chain order when it has at least source→sink shape; a
        # single-token chain is uninformative.
        green["must_have_node_chain_in_order"] = chain

    output_plugins = [s.get("shape_token") or s.get("plugin") for s in target["sinks"]]
    output_plugins = [plugin for plugin in output_plugins if isinstance(plugin, str) and plugin]
    if output_plugins:
        green["must_have_output_plugins"] = output_plugins

    columns = target["source"].get("columns") or []
    if columns:
        green["must_include_observed_columns"] = columns

    red: dict[str, Any] = {
        "passivity_phrases": list(PASSIVITY_PHRASES),
        "build_failure_sentinels": list(BUILD_FAILURE_SENTINELS),
        "must_be_valid": True,
        "passivity_phrases_note": (
            "Lower-case substring match against final assistant content. The skill "
            "(anti-passivity section) explicitly forbids these phrases — any hit is "
            "RED. Standard list shared with the convergence suite."
        ),
        "build_failure_note": (
            "These strings are server-injected by service.py:_build_runtime_preflight_message() "
            "when the model declares completion but the pipeline fails preflight. Their presence "
            "is definitive evidence the model failed to converge."
        ),
    }
    return {"red_criteria": red, "green_criteria": green}


def build_criteria_from_example(
    example_dir: pathlib.Path,
    variant: str | None = None,
) -> dict[str, Any]:
    """End-to-end: load settings.yaml → emit (structural_target, red, green)."""
    target = extract_structural_target(example_dir, variant)
    criteria = build_criteria_from_target(target)
    return {
        "structural_target": target,
        **criteria,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        sys.stderr.write("usage: python -m evals.lib.scenario_from_example <example_dir> [--variant <name>]\n")
        return 64
    example_dir = pathlib.Path(argv[1])
    if not example_dir.is_dir():
        sys.stderr.write(f"not a directory: {example_dir}\n")
        return 67
    variant: str | None = None
    if len(argv) >= 4 and argv[2] == "--variant":
        variant = argv[3]
    try:
        out = build_criteria_from_example(example_dir, variant)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"extraction failed: {exc}\n")
        return 67
    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
