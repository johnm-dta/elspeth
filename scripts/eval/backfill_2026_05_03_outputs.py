"""Retroactive backfill of per-row engine outputs into evals/2026-05-03-composer/.

For each scenario directory under
``evals/2026-05-03-composer/{hardmode/results,basic}/<scenario>/`` with a
captured ``final_yaml.json`` AND ``run.json``, this tool:

1. Reads ``final_yaml.json`` to find every sink's configured output path.
2. Reads ``run.json`` for the run start/finish window.
3. Enumerates candidate files at each configured path (including
   auto-increment siblings — see
   ``elspeth.plugins.infrastructure.output_paths.next_available_output_path``).
4. Timestamp-correlates each candidate against the run window.
5. In ``--dry-run`` mode: prints what would be archived.
6. In ``--apply`` mode: copies HIGH-confidence files into
   ``<scenario>/outputs/`` and writes ``outputs/MANIFEST.json``.

**Known limitation** — ``scenario_dir/run.json`` captures the FIRST
/execute attempt per session. When a v3 proof-of-fix rerun happened
within the same session (e.g., the model swap producing run
``023eb897-...`` for ``p1_t1_happy``), this tool does NOT capture the
v3 evidence — that requires a separate rerun-mode driver fed from the
README's "Audit-trail evidence outside this folder" run-id table.
Tracked in ``elspeth-obs-e87152484a``.

Phase A.5 of ``docs/superpowers/plans/2026-05-06-eval-per-row-output-archival.md``.
Issue: elspeth-77d2641032.

Usage::

    .venv/bin/python -m scripts.eval.backfill_2026_05_03_outputs --dry-run
    .venv/bin/python -m scripts.eval.backfill_2026_05_03_outputs --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.eval._backfill_lib import (
    build_scenario_manifest,
    extract_sink_paths_from_final_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "evals/2026-05-03-composer"

# The staging deploy elspeth.foundryside.dev runs out of this checkout
# (per memory: staging_deployment.md) and writes sink outputs to
# ``data/outputs/`` relative to repo root. The relative paths in each
# scenario's final_yaml.json (e.g. ``outputs/q3_*.csv``) are resolved
# against this directory.
STAGING_DATA_DIR = REPO_ROOT / "data"


def _parse_iso(s: str) -> datetime:
    """Parse an ISO timestamp tolerating trailing ``Z`` for UTC."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def process_scenario(scenario_dir: Path, *, apply: bool) -> dict[str, Any]:
    """Build (and optionally apply) the backfill manifest for one scenario."""
    final_yaml = json.loads((scenario_dir / "final_yaml.json").read_text())
    run = json.loads((scenario_dir / "run.json").read_text())
    sinks = extract_sink_paths_from_final_yaml(final_yaml, data_dir=STAGING_DATA_DIR)
    manifest = build_scenario_manifest(
        scenario_id=scenario_dir.name,
        run_id=run["run_id"],
        run_started_at=_parse_iso(run["started_at"]),
        run_finished_at=_parse_iso(run["finished_at"]),
        sinks=sinks,
    )
    outputs_dir = scenario_dir / "outputs"
    if apply:
        outputs_dir.mkdir(exist_ok=True)
        for record in manifest["files"]:
            src = Path(record["actual_path"])
            dst = outputs_dir / src.name
            shutil.copy2(src, dst)
        (outputs_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _scenario_dirs() -> list[tuple[str, Path]]:
    """Return (tree_label, scenario_dir) for every backfill candidate.

    Hardmode scenarios live under ``hardmode/results/<scen>/``.
    Basic-mode scenarios live under ``basic/<scen>/``. Both must contain
    both ``final_yaml.json`` and ``run.json`` to be backfillable.
    """
    targets: list[tuple[str, Path]] = []
    hardmode_results = EVAL_ROOT / "hardmode/results"
    if hardmode_results.is_dir():
        for entry in sorted(hardmode_results.iterdir()):
            if entry.is_dir():
                targets.append(("hardmode", entry))
    basic_root = EVAL_ROOT / "basic"
    if basic_root.is_dir():
        for entry in sorted(basic_root.iterdir()):
            if entry.is_dir() and entry.name not in {"catalog"}:
                targets.append(("basic", entry))
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    summary: list[dict[str, Any]] = []
    for tree_label, scenario_dir in _scenario_dirs():
        if not (scenario_dir / "final_yaml.json").exists():
            print(
                f"[{tree_label}/{scenario_dir.name}] skipped — no final_yaml.json",
                file=sys.stderr,
            )
            continue
        if not (scenario_dir / "run.json").exists():
            print(
                f"[{tree_label}/{scenario_dir.name}] skipped — no run.json",
                file=sys.stderr,
            )
            continue
        manifest = process_scenario(scenario_dir, apply=args.apply)
        summary.append(
            {
                "tree": tree_label,
                "scenario": scenario_dir.name,
                "high_confidence": len(manifest["files"]),
                "low_confidence_skipped": len(manifest["skipped_low_confidence"]),
            }
        )
        print(
            f"[{tree_label}/{scenario_dir.name}] high={len(manifest['files'])} skipped_low={len(manifest['skipped_low_confidence'])}",
            file=sys.stderr,
        )

    print(
        json.dumps(
            {"mode": "apply" if args.apply else "dry-run", "scenarios": summary},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
