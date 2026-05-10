#!/usr/bin/env bash
# evals/composer-harness/hardmode/generate_cohort.sh
#
# Walks a cohort.yaml × examples/ × personas/ and emits scenario JSONs.
#
# Usage:
#   generate_cohort.sh <cohort.yaml> [--smoke-only] [--force] [--dry-run]
#
# Flags:
#   --smoke-only   Generate only the smoke_subset cells (12 cells, ~$0.012
#                  in drafter cost). Use this to spot-check Phase A
#                  discipline before committing to the broad cohort.
#   --force        Overwrite existing scenario JSONs. Default: skip cells
#                  whose scenario.json already exists.
#   --dry-run      List planned cells; do not call the LLM or write files.
#
# Output:
#   <repo>/evals/composer-harness/<cohort.scenario_root>/<example>__<persona>.json
#   <repo>/evals/composer-harness/<cohort.scenario_root>/_cohort_manifest.json
#
# Requires OPENROUTER_API_KEY or ANTHROPIC_API_KEY in env (drafter call).
# Pass --dry-run to inspect the matrix without an LLM call.

set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
LIB_DIR="$(cd "$HARNESS_ROOT/../lib" && pwd)"
REPO_ROOT="$(cd "$HARNESS_ROOT/../.." && pwd)"
# shellcheck source=../../lib/common.sh
source "$LIB_DIR/common.sh"

evals_require_tools

cohort_file=""
smoke_only=0
force=0
dry_run=0
while (( $# > 0 )); do
  case $1 in
    --smoke-only) smoke_only=1; shift;;
    --force)      force=1; shift;;
    --dry-run)    dry_run=1; shift;;
    -*)           evals_die 64 "unknown flag: $1";;
    *)
      if [[ -n "$cohort_file" ]]; then
        evals_die 64 "extra positional arg: $1"
      fi
      cohort_file=$1; shift
      ;;
  esac
done

[[ -n "$cohort_file" ]] || evals_die 64 "usage: generate_cohort.sh <cohort.yaml> [--smoke-only] [--force] [--dry-run]"
[[ -f "$cohort_file" ]] || evals_die 67 "cohort file not found: $cohort_file"

evals_log INFO "generating from cohort=$cohort_file smoke_only=$smoke_only force=$force dry_run=$dry_run"

python3 - "$cohort_file" "$HARNESS_ROOT" "$REPO_ROOT" "$smoke_only" "$force" "$dry_run" <<'PY'
"""Cohort scenario generator.

Reads a cohort.yaml, expands the matrix (smoke_subset / sparse / diagnostic
plus per-cell overrides), and writes one scenario JSON per (example,
persona, variant) cell.

Each scenario JSON contains:
    scenario_id        — used by harness.sh to bootstrap a run dir
    persona            — used by validate_drift.sh / judge_persona.sh / harness
    cohort             — back-pointer to cohort.yaml
    example            — back-pointer to examples/<name>
    variant            — settings_<variant>.yaml or null
    summary            — one-line description
    opening_prompt     — LLM-drafted persona-flavoured ask
    structural_target  — extracted from settings.yaml (review aid)
    red_criteria       — standard passivity + build-failure sentinels
    green_criteria     — derived from structural_target
"""
from __future__ import annotations

import json
import pathlib
import sys

import yaml

# evals/lib is a Python package; add repo root so the import resolves.
cohort_path = pathlib.Path(sys.argv[1]).resolve()
harness_root = pathlib.Path(sys.argv[2]).resolve()
repo_root = pathlib.Path(sys.argv[3]).resolve()
smoke_only = sys.argv[4] == "1"
force = sys.argv[5] == "1"
dry_run = sys.argv[6] == "1"

sys.path.insert(0, str(repo_root))
from evals.lib.scenario_from_example import build_criteria_from_example  # noqa: E402
from evals.lib.prompt_drafter import draft_opening_prompt, DrafterError  # noqa: E402

cohort = yaml.safe_load(cohort_path.read_text())
if not isinstance(cohort, dict):
    sys.stderr.write(f"cohort {cohort_path} did not parse to a mapping\n")
    sys.exit(67)

cohort_name = cohort.get("cohort_name") or cohort_path.stem
scenario_root_rel = cohort.get("scenario_root") or f"scenarios/panel/{cohort_name}"
scenario_root = harness_root / scenario_root_rel
scenario_root.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Matrix expansion
# --------------------------------------------------------------------------


def _normalize_cells(block: object, default_personas: list[str] | None = None) -> list[dict]:
    """Convert a sparse-list / diagnostic-block into a flat list of cells.

    sparse: list of {example, personas: [...], variant?}
    diagnostic: {personas: [...], examples: [...]} expanded to one cell per
                (example, persona)
    smoke_subset: same shape as sparse
    """
    cells: list[dict] = []
    if isinstance(block, list):
        for entry in block:
            if not isinstance(entry, dict):
                continue
            example = entry.get("example")
            personas = entry.get("personas") or default_personas or []
            variant = entry.get("variant")
            for p in personas:
                cells.append({"example": example, "persona": p, "variant": variant})
    elif isinstance(block, dict):
        personas = block.get("personas") or default_personas or []
        for example in block.get("examples") or []:
            for p in personas:
                cells.append({"example": example, "persona": p, "variant": None})
    return cells


smoke_cells = _normalize_cells(cohort.get("smoke_subset") or [])
sparse_cells = _normalize_cells(cohort.get("sparse") or [])
diagnostic_cells = _normalize_cells(cohort.get("diagnostic") or {})
overrides = cohort.get("overrides") or []

if smoke_only:
    raw_cells = smoke_cells
else:
    raw_cells = smoke_cells + sparse_cells + diagnostic_cells

# Apply overrides (skip + criteria_overrides + variant) and dedupe by
# (example, persona, variant).
override_index: dict[tuple[str, str, str | None], dict] = {}
for ov in overrides:
    if not isinstance(ov, dict):
        continue
    key = (ov.get("example"), ov.get("persona"), ov.get("variant"))
    override_index[key] = ov

seen: set[tuple[str, str, str | None]] = set()
cells: list[dict] = []
skipped: list[dict] = []
for c in raw_cells:
    key = (c["example"], c["persona"], c["variant"])
    if key in seen:
        continue
    seen.add(key)
    ov = override_index.get(key)
    if ov and ov.get("skip_reason"):
        skipped.append({**c, "skip_reason": ov["skip_reason"]})
        continue
    if ov and ov.get("variant") is not None:
        c["variant"] = ov["variant"]
    if ov and ov.get("criteria_overrides"):
        c["criteria_overrides"] = ov["criteria_overrides"]
    cells.append(c)

print(f"cohort {cohort_name}: {len(cells)} cells planned, {len(skipped)} skipped by override")

if dry_run:
    print("--- planned cells ---")
    for c in cells:
        v = f"({c['variant']})" if c["variant"] else ""
        print(f"  {c['example']}{v} × {c['persona']}")
    if skipped:
        print("--- skipped ---")
        for c in skipped:
            print(f"  {c['example']} × {c['persona']}: {c['skip_reason']}")
    sys.exit(0)


# --------------------------------------------------------------------------
# Cell expansion → scenario JSON
# --------------------------------------------------------------------------

generated = 0
skipped_existing = 0
errors: list[dict] = []

for c in cells:
    example_name = c["example"]
    persona_id = c["persona"]
    variant = c["variant"]

    suffix = f"__{variant}" if variant else ""
    scenario_id = f"{example_name}{suffix}__{persona_id}"
    out_path = scenario_root / f"{scenario_id}.json"

    if out_path.exists() and not force:
        skipped_existing += 1
        continue

    example_dir = repo_root / "examples" / example_name
    if not example_dir.is_dir():
        errors.append({"cell": c, "error": f"examples/{example_name} not found"})
        continue

    persona_path = harness_root / "personas" / f"{persona_id}.md"
    if not persona_path.is_file():
        errors.append({"cell": c, "error": f"persona spec not found: {persona_path}"})
        continue

    try:
        criteria = build_criteria_from_example(example_dir, variant=variant)
    except (FileNotFoundError, ValueError) as exc:
        errors.append({"cell": c, "error": f"criteria extraction failed: {exc}"})
        continue

    settings_yaml_text = pathlib.Path(criteria["structural_target"]["settings_path"]).read_text()
    persona_spec_text = persona_path.read_text()

    try:
        opening_prompt = draft_opening_prompt(
            persona_spec=persona_spec_text,
            settings_yaml=settings_yaml_text,
            example_name=example_name,
        )
    except DrafterError as exc:
        errors.append({"cell": c, "error": f"draft failed: {exc}"})
        continue

    scenario = {
        "scenario_id": scenario_id,
        "cohort": cohort_name,
        "example": example_name,
        "variant": variant,
        "persona": persona_id,
        "task_class": "panel",
        # `task_summary` matches the field name harness.sh:146 reads for log
        # output; using the same key as hardmode scenarios avoids divergent
        # contracts. Rendered in the run manifest and used in finalize_scenario.sh
        # ledger building.
        "task_summary": (
            f"Panel cohort cell: {persona_id} asks composer to build the "
            f"{example_name}{suffix} pipeline shape. Generated from "
            f"{cohort_path.name} on the panel-broad cohort."
        ),
        "opening_prompt": opening_prompt,
        "structural_target": criteria["structural_target"],
        "red_criteria": criteria["red_criteria"],
        "green_criteria": criteria["green_criteria"],
    }
    if c.get("criteria_overrides"):
        # Operator-supplied overrides take precedence — merge into derived criteria.
        for k, v in (c["criteria_overrides"].get("green_criteria") or {}).items():
            scenario["green_criteria"][k] = v
        for k, v in (c["criteria_overrides"].get("red_criteria") or {}).items():
            scenario["red_criteria"][k] = v

    out_path.write_text(json.dumps(scenario, indent=2))
    generated += 1
    print(f"  + {scenario_id}")


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

manifest = {
    "cohort_name": cohort_name,
    "cohort_file": str(cohort_path),
    "scenario_root": str(scenario_root),
    "smoke_only": smoke_only,
    "cells_planned": len(cells),
    "cells_generated": generated,
    "cells_skipped_existing": skipped_existing,
    "cells_skipped_by_override": len(skipped),
    "errors": errors,
}
(scenario_root / "_cohort_manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nmanifest: {scenario_root / '_cohort_manifest.json'}")
print(f"generated={generated} skipped_existing={skipped_existing} errors={len(errors)}")

# Operator hint: the harness reads ELSPETH_EVAL_RUNS_DIR + ELSPETH_EVAL_SCENARIO_ROOT
# from env, NOT from cohort.yaml. Print the exact exports the operator needs
# to run this cohort so there's no ambiguity about runs_root being decorative.
runs_root_rel = cohort.get("runs_root") or f"runs/{cohort_name}"
runs_root_abs = harness_root / runs_root_rel
print()
print("# To run this cohort, export these env vars first:")
print(f"export ELSPETH_EVAL_SCENARIO_ROOT={scenario_root}")
print(f"export ELSPETH_EVAL_RUNS_DIR={runs_root_abs}")
if errors:
    print("--- errors ---")
    for e in errors:
        print(f"  {e['cell']['example']} × {e['cell']['persona']}: {e['error']}")
    sys.exit(1)
PY
