# Eval Per-Row Output Archival Implementation Plan

## Status (2026-05-06): Infrastructure landed, archival deferred

Phases B, C, D, and E.1/E.3 landed on branch
`feat/eval-per-row-output-archival` (commits `17ba6c13..99986128`). Phase A
**library + driver landed but `--apply` was NOT run** — a dry-run revealed
that `scenario_dir/run.json` captures only the FIRST `/execute` attempt
per session (per `elspeth-obs-e87152484a`), so the bytes the walkthrough's
INT-1001/INT-1002 routing claim rests on (the `023eb897-...` v3 proof-of-fix
run for `p1_t1_happy`) are NOT what the as-written driver targets.

The **per-row engine output evidence gap for the 2026-05-03 evals
remains open**. Closing it requires a **v3-rerun mode** for the driver
that takes a `(scenario, run_id)` mapping driven from the README's
"Audit-trail evidence outside this folder" run-id table, fetches each
run's window from staging via the (now-deployed) `/api/runs/{rid}` and
`/api/runs/{rid}/outputs` endpoints, and emits sibling
`outputs_v3/MANIFEST.json` directories. That work needs an operator JWT
against staging and is tracked in the closeout comment on
`elspeth-77d2641032`.

**Tasks A.6, A.7, A.8 in the plan body below are stale** — A.6 in
particular ("Apply hardmode backfill") would archive failed-run
errors-sink data that's largely redundant with the existing
`diagnostics.json`. Read the body below as historical context; refer to
the closeout comment on `elspeth-77d2641032` for current state.

---

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the per-row engine output evidence gap in `evals/2026-05-03-composer/` by retroactively backfilling row-level outputs from staging disk, adding a durable `/api/runs/{rid}/outputs` endpoint, mirroring the harness across hardmode and basic-mode, and updating the README so any reader can verify row-level claims from in-tree evidence alone.

**Architecture:** Three independent layers stacked. (1) **Backfill layer** — a one-shot Python tool that parses each scenario's `final_yaml.json` (or recovers it from session state for basic-mode), maps configured sink paths to candidate files on staging disk, timestamp-correlates against the run window, and writes `outputs/MANIFEST.json` per scenario. (2) **API layer** — a new `GET /api/runs/{rid}/outputs` manifest endpoint plus `GET /api/runs/{rid}/outputs/{artifact_id}/content` streaming endpoint, both DB-driven via the existing `artifacts` + `operations` Landscape tables, with a possible `written_path` column addition. (3) **Harness layer** — extend the hardmode `finalize_scenario.sh` and create a parallel basic-mode finalize step that pulls outputs through the new API.

**Tech Stack:** Python 3.13, FastAPI (web), SQLAlchemy Core (Landscape audit DB), pyyaml (config parsing), pytest + Hypothesis (tests), bash + curl + jq (harness), Pydantic (response schemas).

**Sequencing rationale:** Phase A (backfill) MUST run first. Staging is a live system writing to a shared, run-id-unstamped output directory; every hour of delay risks an unrelated run overwriting the bytes that substantiate the 2026-05-03 walkthrough's row-level claims.

**Final filename hint:** Once approved, copy this plan to
`docs/superpowers/plans/2026-05-06-eval-per-row-output-archival.md` so it
lives with the other plans (the writing-plans canonical location). Plan
mode blocks me from creating it there directly.

---

## File Structure

| Path | Role | Phase |
|---|---|---|
| `scripts/eval/backfill_2026_05_03_outputs.py` | NEW — one-shot retroactive backfill tool | A |
| `scripts/eval/_backfill_lib.py` | NEW — pure logic (parse YAML, correlate, classify) | A |
| `tests/unit/scripts/eval/test_backfill_lib.py` | NEW — unit tests for backfill logic | A |
| `evals/2026-05-03-composer/<basic\|hardmode>/<scenario>/outputs/` | NEW — backfilled per-scenario outputs + MANIFEST.json | A |
| `src/elspeth/web/execution/schemas.py` | MODIFY — add `RunOutputArtifact`, `RunOutputsResponse` | C |
| `src/elspeth/web/execution/outputs.py` | NEW — loader (mirrors `diagnostics.py`) | C |
| `src/elspeth/web/execution/routes.py` | MODIFY — add manifest + content endpoints | C |
| `src/elspeth/core/landscape/schema.py` | MODIFY (conditional) — add `written_path` to `artifacts` | B/C |
| `src/elspeth/core/landscape/data_flow_repository.py` | MODIFY (conditional) — populate `written_path` at sink-write time | B/C |
| `tests/unit/web/execution/test_outputs_loader.py` | NEW — loader unit tests | C |
| `tests/integration/web/test_run_outputs_endpoint.py` | NEW — endpoint integration tests | C |
| `evals/2026-05-03-composer/hardmode/finalize_scenario.sh` | MODIFY — pull outputs after diagnostics | D |
| `evals/2026-05-03-composer/basic/finalize_scenario.sh` | NEW — mirror hardmode | D |
| `evals/2026-05-03-composer/README.md` | MODIFY — outputs tree, verify recipes, backfill provenance | E |
| `evals/2026-05-03-composer/hardmode/README.md` | MODIFY — outputs/ inventory + verify recipes | E |
| `evals/2026-05-03-composer/basic/README.md` | NEW — mirror hardmode | E |

---

## Phase A — Retroactive Backfill (TIME-SENSITIVE: do first)

### Task A1: Pure-logic helpers — extract sink output paths from a final_yaml.json

**Files:**
- Create: `scripts/eval/_backfill_lib.py`
- Test: `tests/unit/scripts/eval/test_backfill_lib.py`

- [ ] **Step A1.1: Write the failing test for sink-path extraction**

```python
# tests/unit/scripts/eval/test_backfill_lib.py
from scripts.eval._backfill_lib import extract_sink_paths_from_final_yaml

def test_extract_sink_paths_from_final_yaml_returns_name_and_path_for_each_sink():
    final_yaml_dict = {
        "yaml": (
            "outputs:\n"
            "  - plugin: jsonl\n"
            "    name: results\n"
            "    options:\n"
            "      path: /home/john/elspeth/data/outputs/results.jsonl\n"
            "  - plugin: csv\n"
            "    name: fraud_only\n"
            "    options:\n"
            "      path: /home/john/elspeth/data/outputs/q3_fraud_security_flags.csv\n"
        )
    }
    sinks = extract_sink_paths_from_final_yaml(final_yaml_dict)
    assert sinks == [
        ("results", "/home/john/elspeth/data/outputs/results.jsonl"),
        ("fraud_only", "/home/john/elspeth/data/outputs/q3_fraud_security_flags.csv"),
    ]


def test_extract_sink_paths_skips_outputs_without_path_options():
    final_yaml_dict = {
        "yaml": "outputs:\n  - plugin: noop\n    name: ignore\n    options: {}\n"
    }
    assert extract_sink_paths_from_final_yaml(final_yaml_dict) == []
```

- [ ] **Step A1.2: Run test; expect import error**

Run: `.venv/bin/python -m pytest tests/unit/scripts/eval/test_backfill_lib.py -xvs`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval._backfill_lib'`

- [ ] **Step A1.3: Implement extract_sink_paths_from_final_yaml**

```python
# scripts/eval/_backfill_lib.py
"""Pure-logic helpers for the 2026-05-03 eval per-row output backfill."""

from __future__ import annotations

from typing import Mapping, Sequence

import yaml


def extract_sink_paths_from_final_yaml(
    final_yaml_dict: Mapping[str, object],
) -> list[tuple[str, str]]:
    """Parse the YAML string inside a final_yaml.json blob and return
    (sink_name, configured_path) for each output that has a path option.
    """
    yaml_text = final_yaml_dict.get("yaml")
    if not isinstance(yaml_text, str):
        return []
    config = yaml.safe_load(yaml_text) or {}
    outputs = config.get("outputs") or []
    if not isinstance(outputs, Sequence):
        return []
    result: list[tuple[str, str]] = []
    for entry in outputs:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        options = entry.get("options") or {}
        path = options.get("path") if isinstance(options, Mapping) else None
        if isinstance(name, str) and isinstance(path, str):
            result.append((name, path))
    return result
```

- [ ] **Step A1.4: Run test; expect pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/eval/test_backfill_lib.py -xvs`
Expected: PASS (2 tests)

- [ ] **Step A1.5: Commit**

```bash
git add scripts/eval/_backfill_lib.py tests/unit/scripts/eval/test_backfill_lib.py
git commit -m "feat(eval): add sink-path extractor for 2026-05-03 backfill"
```

---

### Task A2: Pure-logic — enumerate auto-increment candidate files for a configured path

**Files:**
- Modify: `scripts/eval/_backfill_lib.py`
- Modify: `tests/unit/scripts/eval/test_backfill_lib.py`

- [ ] **Step A2.1: Write the failing test for candidate enumeration**

```python
# Append to tests/unit/scripts/eval/test_backfill_lib.py
from scripts.eval._backfill_lib import enumerate_candidate_files


def test_enumerate_candidate_files_includes_base_and_auto_increment_siblings(tmp_path):
    base = tmp_path / "high_priority.jsonl"
    base.write_text("base\n")
    (tmp_path / "high_priority-1.jsonl").write_text("one\n")
    (tmp_path / "high_priority-2.jsonl").write_text("two\n")
    (tmp_path / "unrelated.jsonl").write_text("nope\n")

    candidates = enumerate_candidate_files(str(base))
    assert sorted(c.name for c in candidates) == [
        "high_priority-1.jsonl",
        "high_priority-2.jsonl",
        "high_priority.jsonl",
    ]


def test_enumerate_candidate_files_returns_empty_when_directory_missing(tmp_path):
    missing = tmp_path / "nope" / "results.jsonl"
    assert enumerate_candidate_files(str(missing)) == []
```

- [ ] **Step A2.2: Run test; expect ImportError**

Run: `.venv/bin/python -m pytest tests/unit/scripts/eval/test_backfill_lib.py::test_enumerate_candidate_files_includes_base_and_auto_increment_siblings -xvs`
Expected: FAIL — `ImportError: cannot import name 'enumerate_candidate_files'`

- [ ] **Step A2.3: Implement enumerate_candidate_files**

```python
# Append to scripts/eval/_backfill_lib.py
import re
from pathlib import Path


def enumerate_candidate_files(configured_path: str) -> list[Path]:
    """Return the base file (if it exists) plus any auto-increment siblings
    matching ``stem-N.ext`` in the same directory.

    Mirrors the rename behaviour of
    ``elspeth.plugins.infrastructure.output_paths.next_available_output_path``.
    """
    base = Path(configured_path)
    parent = base.parent
    if not parent.is_dir():
        return []
    suffix = "".join(base.suffixes)
    if suffix:
        stem = base.name[: -len(suffix)]
    else:
        stem = base.name
    sibling_re = re.compile(rf"^{re.escape(stem)}-\d+{re.escape(suffix)}$")
    candidates: list[Path] = []
    if base.exists():
        candidates.append(base)
    for entry in parent.iterdir():
        if entry.is_file() and sibling_re.match(entry.name):
            candidates.append(entry)
    return sorted(candidates, key=lambda p: p.name)
```

- [ ] **Step A2.4: Run test; expect pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/eval/test_backfill_lib.py -xvs`
Expected: PASS (4 tests total)

- [ ] **Step A2.5: Commit**

```bash
git add scripts/eval/_backfill_lib.py tests/unit/scripts/eval/test_backfill_lib.py
git commit -m "feat(eval): enumerate auto-increment candidate files for backfill"
```

---

### Task A3: Pure-logic — classify timestamp-correlation confidence

**Files:**
- Modify: `scripts/eval/_backfill_lib.py`
- Modify: `tests/unit/scripts/eval/test_backfill_lib.py`

- [ ] **Step A3.1: Write the failing test for confidence classification**

```python
# Append to tests/unit/scripts/eval/test_backfill_lib.py
from datetime import datetime, timedelta, timezone

from scripts.eval._backfill_lib import classify_correlation_confidence


def test_classify_correlation_confidence_high_when_mtime_inside_run_window():
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=timezone.utc)
    file_mtime = datetime(2026, 5, 3, 13, 28, 14, tzinfo=timezone.utc)
    assert classify_correlation_confidence(file_mtime, started, finished) == "high"


def test_classify_correlation_confidence_high_within_grace_after_finish():
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=timezone.utc)
    file_mtime = finished + timedelta(seconds=45)
    assert classify_correlation_confidence(file_mtime, started, finished) == "high"


def test_classify_correlation_confidence_low_when_outside_window():
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=timezone.utc)
    file_mtime = datetime(2026, 4, 28, 5, 47, 0, tzinfo=timezone.utc)
    assert classify_correlation_confidence(file_mtime, started, finished) == "low"


def test_classify_correlation_confidence_low_when_mtime_before_start():
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=timezone.utc)
    file_mtime = started - timedelta(seconds=10)
    assert classify_correlation_confidence(file_mtime, started, finished) == "low"
```

- [ ] **Step A3.2: Run test; expect ImportError**

Run: `.venv/bin/python -m pytest tests/unit/scripts/eval/test_backfill_lib.py -xvs -k classify_correlation`
Expected: FAIL — ImportError

- [ ] **Step A3.3: Implement classify_correlation_confidence**

```python
# Append to scripts/eval/_backfill_lib.py
from datetime import datetime, timedelta
from typing import Literal

CONFIDENCE_GRACE_SECONDS = 60

Confidence = Literal["high", "low"]


def classify_correlation_confidence(
    file_mtime: datetime,
    run_started_at: datetime,
    run_finished_at: datetime,
) -> Confidence:
    """Return 'high' when ``file_mtime`` falls inside the run window
    (``[run_started_at, run_finished_at + grace]``), else 'low'.

    Why a grace window: sinks may flush slightly after the engine reports
    finish; a 60-second grace tolerates clock skew and post-loop flush.
    Mtimes BEFORE start are unambiguously a different run.
    """
    if file_mtime < run_started_at:
        return "low"
    if file_mtime > run_finished_at + timedelta(seconds=CONFIDENCE_GRACE_SECONDS):
        return "low"
    return "high"
```

- [ ] **Step A3.4: Run test; expect pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/eval/test_backfill_lib.py -xvs`
Expected: PASS (8 tests total)

- [ ] **Step A3.5: Commit**

```bash
git add scripts/eval/_backfill_lib.py tests/unit/scripts/eval/test_backfill_lib.py
git commit -m "feat(eval): classify timestamp-correlation confidence for backfill"
```

---

### Task A4: Pure-logic — produce the per-scenario manifest record

**Files:**
- Modify: `scripts/eval/_backfill_lib.py`
- Modify: `tests/unit/scripts/eval/test_backfill_lib.py`

- [ ] **Step A4.1: Write the failing test for build_scenario_manifest**

```python
# Append to tests/unit/scripts/eval/test_backfill_lib.py
import hashlib

from scripts.eval._backfill_lib import build_scenario_manifest


def test_build_scenario_manifest_classifies_files_and_records_hashes(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    inside = outputs_dir / "results.jsonl"
    inside.write_text('{"interaction_id":"INT-1001"}\n')
    outside = outputs_dir / "results-1.jsonl"
    outside.write_text('{"interaction_id":"OLD"}\n')

    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=timezone.utc)

    import os
    inside_ts = (started + timedelta(seconds=14)).timestamp()
    outside_ts = (started - timedelta(days=5)).timestamp()
    os.utime(inside, (inside_ts, inside_ts))
    os.utime(outside, (outside_ts, outside_ts))

    manifest = build_scenario_manifest(
        scenario_id="p1_t1_happy",
        run_id="e9912276-8be5-4ccc-b74f-dd5f3c401946",
        run_started_at=started,
        run_finished_at=finished,
        sinks=[("results", str(inside))],
    )

    assert manifest["scenario_id"] == "p1_t1_happy"
    assert manifest["run_id"] == "e9912276-8be5-4ccc-b74f-dd5f3c401946"
    assert len(manifest["files"]) == 1
    file_record = manifest["files"][0]
    assert file_record["sink_name"] == "results"
    assert file_record["correlation_confidence"] == "high"
    assert file_record["sha256"] == hashlib.sha256(inside.read_bytes()).hexdigest()
    assert len(manifest["skipped_low_confidence"]) == 1
    assert manifest["skipped_low_confidence"][0]["actual_path"] == str(outside)
```

- [ ] **Step A4.2: Run test; expect ImportError**

Run: `.venv/bin/python -m pytest tests/unit/scripts/eval/test_backfill_lib.py -xvs -k build_scenario_manifest`
Expected: FAIL — ImportError

- [ ] **Step A4.3: Implement build_scenario_manifest**

```python
# Append to scripts/eval/_backfill_lib.py
import hashlib
from datetime import timezone


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_scenario_manifest(
    *,
    scenario_id: str,
    run_id: str,
    run_started_at: datetime,
    run_finished_at: datetime,
    sinks: list[tuple[str, str]],
    captured_by: str = "scripts/eval/backfill_2026_05_03_outputs.py",
) -> dict:
    """Build a manifest dict for one scenario from a sink list.

    Files with HIGH-confidence timestamp correlation go into ``files``;
    LOW-confidence candidates go into ``skipped_low_confidence`` (recorded
    but NOT recommended for archive copy).
    """
    captured_at = datetime.now(timezone.utc).isoformat()
    files: list[dict] = []
    skipped: list[dict] = []
    for sink_name, configured_path in sinks:
        for candidate in enumerate_candidate_files(configured_path):
            mtime = datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc)
            confidence = classify_correlation_confidence(
                mtime, run_started_at, run_finished_at
            )
            record = {
                "sink_name": sink_name,
                "configured_path": configured_path,
                "actual_path": str(candidate),
                "archived_as": f"outputs/{candidate.name}",
                "size": candidate.stat().st_size,
                "sha256": _sha256(candidate),
                "mtime": mtime.isoformat(),
                "correlation_confidence": confidence,
                "captured_at": captured_at,
                "captured_by": captured_by,
            }
            if confidence == "high":
                files.append(record)
            else:
                skipped.append({
                    "sink_name": sink_name,
                    "configured_path": configured_path,
                    "actual_path": str(candidate),
                    "mtime": mtime.isoformat(),
                    "reason": (
                        f"mtime {mtime.date()} outside run window "
                        f"[{run_started_at.isoformat()}, {run_finished_at.isoformat()}]"
                    ),
                })
    return {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "run_window": {
            "started_at": run_started_at.isoformat(),
            "finished_at": run_finished_at.isoformat(),
        },
        "files": files,
        "skipped_low_confidence": skipped,
    }
```

- [ ] **Step A4.4: Run test; expect pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/eval/test_backfill_lib.py -xvs`
Expected: PASS (9 tests total)

- [ ] **Step A4.5: Commit**

```bash
git add scripts/eval/_backfill_lib.py tests/unit/scripts/eval/test_backfill_lib.py
git commit -m "feat(eval): build per-scenario backfill manifest with confidence flags"
```

---

### Task A5: One-shot driver script (hardmode path)

**Files:**
- Create: `scripts/eval/backfill_2026_05_03_outputs.py`

- [ ] **Step A5.1: Write the driver script**

```python
# scripts/eval/backfill_2026_05_03_outputs.py
"""Retroactive backfill of per-row engine outputs into evals/2026-05-03-composer/.

Time-sensitive: staging is live and the data/outputs/ directory is shared
across runs. Run ASAP. Use --dry-run first to inspect classifications.

Usage:
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

from scripts.eval._backfill_lib import (
    build_scenario_manifest,
    extract_sink_paths_from_final_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "evals/2026-05-03-composer"


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def process_hardmode_scenario(scenario_dir: Path, *, apply: bool) -> dict:
    final_yaml = json.loads((scenario_dir / "final_yaml.json").read_text())
    run = json.loads((scenario_dir / "run.json").read_text())
    sinks = extract_sink_paths_from_final_yaml(final_yaml)
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
        (outputs_dir / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    hardmode_results = EVAL_ROOT / "hardmode/results"
    summary = []
    for scenario_dir in sorted(hardmode_results.iterdir()):
        if not scenario_dir.is_dir():
            continue
        if not (scenario_dir / "final_yaml.json").exists():
            continue
        if not (scenario_dir / "run.json").exists():
            continue
        manifest = process_hardmode_scenario(scenario_dir, apply=args.apply)
        summary.append({
            "scenario": scenario_dir.name,
            "high_confidence": len(manifest["files"]),
            "low_confidence_skipped": len(manifest["skipped_low_confidence"]),
        })
        print(f"[{scenario_dir.name}] high={len(manifest['files'])} "
              f"skipped_low={len(manifest['skipped_low_confidence'])}",
              file=sys.stderr)

    print(json.dumps({"mode": "apply" if args.apply else "dry-run",
                      "scenarios": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step A5.2: Verify dry-run executes against current eval tree**

Run: `.venv/bin/python -m scripts.eval.backfill_2026_05_03_outputs --dry-run`
Expected: prints per-scenario counts to stderr; JSON summary to stdout.
Manually inspect: at least the scenarios from README §"Audit-trail evidence" table (P1-T1 v3 etc.) report `high_confidence > 0`. If a scenario reports `high_confidence == 0`, the run window may not match disk mtimes — investigate before applying.

- [ ] **Step A5.3: Commit the script (apply step is separate)**

```bash
git add scripts/eval/backfill_2026_05_03_outputs.py
git commit -m "feat(eval): one-shot backfill driver for 2026-05-03 hardmode outputs"
```

---

### Task A6: Apply hardmode backfill (gated manual ritual)

**Files:**
- New under: `evals/2026-05-03-composer/hardmode/results/<scenario>/outputs/`

- [ ] **Step A6.1: Manual ritual — operator inspection of dry-run output**

Run: `.venv/bin/python -m scripts.eval.backfill_2026_05_03_outputs --dry-run | tee /tmp/backfill-dry.json`
Operator MUST read `/tmp/backfill-dry.json` and confirm:
- Each scenario referenced in `evals/2026-05-03-composer/hardmode/README.md` produces a non-empty file set.
- `low_confidence_skipped` counts are reasonable (small for active scenarios; outliers >5 deserve investigation).

If anything looks wrong, STOP, investigate, do not proceed to A6.2.

- [ ] **Step A6.2: Apply the backfill**

Run: `.venv/bin/python -m scripts.eval.backfill_2026_05_03_outputs --apply`
Expected: stderr lines per scenario; each scenario dir now has `outputs/MANIFEST.json` + the high-confidence files.

- [ ] **Step A6.3: Verify counts match run.json**

Run:
```bash
for d in evals/2026-05-03-composer/hardmode/results/*/; do
  if [ -d "$d/outputs" ]; then
    rows_in_jsonl=$(cat "$d"/outputs/*.jsonl 2>/dev/null | wc -l)
    routed=$(jq '(.rows_routed_success // 0) + (.rows_routed_failure // 0) + (.rows_quarantined // 0)' "$d/run.json")
    echo "$(basename $d): jsonl_lines=$rows_in_jsonl routed_total=$routed"
  fi
done
```
Operator inspects: for scenarios with `ran_engine: true` in their ledger, `jsonl_lines == routed_total` (or higher if multiple sinks share rows via fork; document the distribution in §E1 README update).

- [ ] **Step A6.4: Commit the backfilled evidence**

```bash
git add evals/2026-05-03-composer/hardmode/results/*/outputs/
git commit -m "evidence(eval): backfill hardmode 2026-05-03 per-row outputs

HIGH-confidence timestamp-correlated copies from staging-disk
data/outputs/ at backfill time. See per-scenario MANIFEST.json
for sha256, mtime, and confidence rationale."
```

---

### Task A7: Recover basic-mode final_yaml from session state

**Files:**
- Create: `scripts/eval/recover_basic_final_yaml.sh`

Why a separate task: basic-mode scenarios under `basic/s*/` lack
`final_yaml.json` (only HTTP request/response captures are in-tree). We
need to retroactively fetch the composed pipeline config so the same
backfill machinery (Task A4–A6) can apply.

- [ ] **Step A7.1: Write the recovery script**

```bash
#!/usr/bin/env bash
# Recover final_yaml.json + run.json + diagnostics.json for each basic
# scenario by querying staging via the session ID stored in sid.txt.
#
# Requires: $JWT exported (short-lived dta_user JWT against staging).
#
# Usage: env JWT=<token> bash scripts/eval/recover_basic_final_yaml.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASIC="$ROOT/evals/2026-05-03-composer/basic"

declare -A RUN_IDS=(
  [s2]="45a592e1-7a8e-416b-8650-906118b9c96d"
  [s3_prime]="a419c8a8-e07d-4618-b2c1-515b257bbb07"
  [s4]="b50133e4-cf17-45bb-9e83-f04b1011b0cf"
)

for scenario_dir in "$BASIC"/s*/; do
  scen=$(basename "$scenario_dir")
  if [[ ! -f "$scenario_dir/sid.txt" ]]; then
    echo "[$scen] no sid.txt — skipping" >&2
    continue
  fi
  sid=$(cat "$scenario_dir/sid.txt")
  echo "[$scen] sid=$sid" >&2

  curl -fsS -H "Authorization: Bearer $JWT" \
    "https://elspeth.foundryside.dev/api/sessions/$sid/state/yaml" \
    -o "$scenario_dir/final_yaml.json" || \
    echo "[$scen] state/yaml fetch failed" >&2

  rid="${RUN_IDS[$scen]:-}"
  if [[ -n "$rid" ]]; then
    curl -fsS -H "Authorization: Bearer $JWT" \
      "https://elspeth.foundryside.dev/api/runs/$rid" \
      -o "$scenario_dir/run.json" || true
    curl -fsS -H "Authorization: Bearer $JWT" \
      "https://elspeth.foundryside.dev/api/runs/$rid/diagnostics" \
      -o "$scenario_dir/diagnostics.json" || true
  fi
done
```

- [ ] **Step A7.2: Operator obtains a fresh JWT against staging**

Operator runs the staging login flow (out-of-band; same procedure used to
generate the JWTs that produced the original eval evidence). Exports `$JWT`.

- [ ] **Step A7.3: Run the recovery script**

Run: `chmod +x scripts/eval/recover_basic_final_yaml.sh && JWT=$JWT bash scripts/eval/recover_basic_final_yaml.sh`
Expected: each scenario in `basic/s2/`, `basic/s3_prime/`, `basic/s4/` now has `final_yaml.json` and `run.json` (and possibly `diagnostics.json`). Some sessions may be GC'd from staging — record those gaps explicitly.

- [ ] **Step A7.4: Commit recovered metadata**

```bash
git add scripts/eval/recover_basic_final_yaml.sh \
        evals/2026-05-03-composer/basic/*/final_yaml.json \
        evals/2026-05-03-composer/basic/*/run.json \
        evals/2026-05-03-composer/basic/*/diagnostics.json
git commit -m "evidence(eval): recover basic-mode final_yaml + run.json from staging"
```

---

### Task A8: Apply backfill to basic-mode

**Files:**
- Modify: `scripts/eval/backfill_2026_05_03_outputs.py`

- [ ] **Step A8.1: Extend the driver to walk basic/s*/ as well**

```python
# Modify scripts/eval/backfill_2026_05_03_outputs.py — add a basic-mode loop
# alongside the hardmode loop in main().
def main(argv=None):
    # ... existing arg parsing ...

    summary = []
    for tree_label, scenarios_root in [
        ("hardmode", EVAL_ROOT / "hardmode/results"),
        ("basic", EVAL_ROOT / "basic"),
    ]:
        for scenario_dir in sorted(scenarios_root.iterdir()):
            if not scenario_dir.is_dir():
                continue
            if not (scenario_dir / "final_yaml.json").exists():
                continue
            if not (scenario_dir / "run.json").exists():
                continue
            manifest = process_hardmode_scenario(scenario_dir, apply=args.apply)
            summary.append({
                "tree": tree_label,
                "scenario": scenario_dir.name,
                "high_confidence": len(manifest["files"]),
                "low_confidence_skipped": len(manifest["skipped_low_confidence"]),
            })
            print(f"[{tree_label}/{scenario_dir.name}] "
                  f"high={len(manifest['files'])} "
                  f"skipped_low={len(manifest['skipped_low_confidence'])}",
                  file=sys.stderr)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run",
                      "scenarios": summary}, indent=2))
    return 0
```

(Rename `process_hardmode_scenario` → `process_scenario` to reflect the
generalisation; the implementation works unchanged for both trees.)

- [ ] **Step A8.2: Dry-run, inspect, apply**

Run: `.venv/bin/python -m scripts.eval.backfill_2026_05_03_outputs --dry-run`
Operator inspects for basic/s2, basic/s3_prime, basic/s4. After confirmation:
Run: `.venv/bin/python -m scripts.eval.backfill_2026_05_03_outputs --apply`

- [ ] **Step A8.3: Verify and commit**

```bash
for d in evals/2026-05-03-composer/basic/s*/outputs/; do
  if [ -d "$d" ]; then
    echo "=== $d ==="
    ls -la "$d"
  fi
done

git add scripts/eval/backfill_2026_05_03_outputs.py \
        evals/2026-05-03-composer/basic/*/outputs/
git commit -m "evidence(eval): backfill basic-mode 2026-05-03 per-row outputs"
```

---

## Phase B — Audit DB schema investigation (block-resolve before C)

### Task B1: Determine whether `artifacts` already records the written path

**Files:**
- Read-only: `src/elspeth/core/landscape/schema.py`, `src/elspeth/core/landscape/data_flow_repository.py`

- [ ] **Step B1.1: Read the schema and decide**

Read: `src/elspeth/core/landscape/schema.py` lines 350–470 (the `artifacts`
table definition).

For each column in `artifacts`, determine:
- Is the *actually-written* filesystem path (post-collision-policy
  resolution) recorded? Likely candidates: a column named `path`,
  `output_path`, `file_path`, `written_path`, or carried in
  `artifact_id` if the ID is path-derived.

Then read `src/elspeth/core/landscape/data_flow_repository.py` for
artifact-write calls to see what's passed in at sink-write time.

Decision:
- **(a) Already recorded** → skip B2/B3, proceed to Phase C with the
  existing column.
- **(b) NOT recorded** → proceed to Task B2.

Document the decision in a short note inside this plan (replace this
checkbox with the decision) so downstream tasks reference the chosen
path.

---

### Task B2: (CONDITIONAL on B1=b) Add `written_path` column to `artifacts`

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py`

- [ ] **Step B2.1: Write the failing test**

```python
# tests/unit/core/landscape/test_artifacts_written_path.py
from elspeth.core.landscape.schema import artifacts_table

def test_artifacts_table_has_written_path_column():
    assert "written_path" in artifacts_table.c
    assert artifacts_table.c.written_path.type.length is None or \
           artifacts_table.c.written_path.type.length >= 1024
```

- [ ] **Step B2.2: Run; expect fail**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_artifacts_written_path.py -xvs`
Expected: FAIL — KeyError or AttributeError.

- [ ] **Step B2.3: Add the column to the schema**

```python
# In src/elspeth/core/landscape/schema.py inside the artifacts_table definition:
Column("written_path", Text, nullable=True),  # post-collision-policy actual path
```

(Place it near the existing path/hash columns.)

- [ ] **Step B2.4: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_artifacts_written_path.py -xvs`
Expected: PASS.

- [ ] **Step B2.5: Commit**

```bash
git add src/elspeth/core/landscape/schema.py tests/unit/core/landscape/test_artifacts_written_path.py
git commit -m "feat(landscape): add artifacts.written_path column for post-collision path"
```

---

### Task B3: (CONDITIONAL on B2) Populate written_path at sink-write time

**Files:**
- Modify: `src/elspeth/core/landscape/data_flow_repository.py`
- Test: `tests/unit/core/landscape/test_data_flow_repository.py` (extend existing)

- [ ] **Step B3.1: Write the failing test for the artifact-write path**

(Plan note: pinpoint the exact `INSERT INTO artifacts ... ` site by reading
`data_flow_repository.py`. Open the function and add a test that passes a
mock-resolved path and asserts the written_path column is populated.
Concrete code blocked on B1 reading.)

- [ ] **Step B3.2–B3.5:** mirror Task B2 pattern (fail → implement → pass → commit). Commit message: `feat(landscape): populate artifacts.written_path at sink-write time`.

---

## Phase C — `/api/runs/{rid}/outputs` endpoint

### Task C1: Response schemas

**Files:**
- Modify: `src/elspeth/web/execution/schemas.py`
- Test: `tests/unit/web/execution/test_schemas.py`

- [ ] **Step C1.1: Write failing test for RunOutputArtifact + RunOutputsResponse**

```python
# tests/unit/web/execution/test_schemas.py — append
from datetime import datetime, timezone
from uuid import UUID

from elspeth.web.execution.schemas import RunOutputArtifact, RunOutputsResponse


def test_run_output_artifact_round_trips_through_pydantic():
    art = RunOutputArtifact(
        artifact_id="abc-123",
        sink_name="results",
        artifact_type="jsonl",
        configured_path="/data/outputs/results.jsonl",
        written_path="/data/outputs/results.jsonl",
        written_at=datetime(2026, 5, 3, 13, 28, 14, tzinfo=timezone.utc),
        size=2414,
        sha256="0" * 64,
        exists_now=True,
    )
    assert art.model_dump()["sink_name"] == "results"


def test_run_outputs_response_carries_artifacts():
    rsp = RunOutputsResponse(
        run_id=UUID("e9912276-8be5-4ccc-b74f-dd5f3c401946"),
        landscape_run_id=UUID("e9912276-8be5-4ccc-b74f-dd5f3c401946"),
        artifacts=[],
    )
    assert rsp.artifacts == []
```

- [ ] **Step C1.2: Run; expect ImportError**

Run: `.venv/bin/python -m pytest tests/unit/web/execution/test_schemas.py -xvs -k run_output`
Expected: FAIL — ImportError.

- [ ] **Step C1.3: Implement schemas**

```python
# Append to src/elspeth/web/execution/schemas.py

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel


class RunOutputArtifact(BaseModel):
    artifact_id: str
    sink_name: str
    artifact_type: str
    configured_path: str
    written_path: str
    written_at: datetime
    size: int
    sha256: str
    exists_now: bool


class RunOutputsResponse(BaseModel):
    run_id: UUID
    landscape_run_id: UUID
    artifacts: list[RunOutputArtifact]
```

(Add the imports at the top of the file if not already present.)

- [ ] **Step C1.4: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/unit/web/execution/test_schemas.py -xvs -k run_output`
Expected: PASS.

- [ ] **Step C1.5: Commit**

```bash
git add src/elspeth/web/execution/schemas.py tests/unit/web/execution/test_schemas.py
git commit -m "feat(web): add RunOutputArtifact + RunOutputsResponse schemas"
```

---

### Task C2: Loader module (DB query + filesystem stat)

**Files:**
- Create: `src/elspeth/web/execution/outputs.py`
- Test: `tests/unit/web/execution/test_outputs_loader.py`

- [ ] **Step C2.1: Write failing test for loader against an in-memory DB**

```python
# tests/unit/web/execution/test_outputs_loader.py
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from elspeth.core.landscape.factory import create_test_landscape  # see existing test helpers
from elspeth.web.execution.outputs import load_run_outputs_for_settings


@pytest.fixture
def fake_settings(tmp_path, monkeypatch):
    # Use an existing settings-builder helper from tests/conftest.py if present;
    # otherwise build minimum settings with data_dir=tmp_path.
    from tests.conftest import build_test_settings  # adapt to actual helper
    return build_test_settings(data_dir=str(tmp_path))


def test_load_run_outputs_returns_artifact_with_filesystem_stats(fake_settings, tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    sink_file = outputs_dir / "results.jsonl"
    sink_file.write_bytes(b'{"interaction_id":"INT-1001"}\n')

    run_id = uuid4()
    # Insert a runs row + nodes row + artifacts row referencing sink_file.
    # Helper insert_artifact_row(landscape, run_id, sink_node_id, written_path)
    # is added in Phase B if not already present.
    landscape = create_test_landscape(fake_settings)
    landscape.insert_test_artifact(
        run_id=run_id,
        sink_node_id="results",
        artifact_type="jsonl",
        written_path=str(sink_file),
        configured_path=str(sink_file),
        sha256=hashlib.sha256(sink_file.read_bytes()).hexdigest(),
    )

    response = load_run_outputs_for_settings(
        fake_settings, run_id=run_id, landscape_run_id=run_id
    )
    assert len(response.artifacts) == 1
    art = response.artifacts[0]
    assert art.sink_name == "results"
    assert art.written_path == str(sink_file)
    assert art.exists_now is True
    assert art.size == sink_file.stat().st_size


def test_load_run_outputs_marks_missing_files_exists_now_false(fake_settings, tmp_path):
    run_id = uuid4()
    landscape = create_test_landscape(fake_settings)
    landscape.insert_test_artifact(
        run_id=run_id,
        sink_node_id="results",
        artifact_type="jsonl",
        written_path="/nonexistent/results.jsonl",
        configured_path="/nonexistent/results.jsonl",
        sha256="0" * 64,
    )
    response = load_run_outputs_for_settings(
        fake_settings, run_id=run_id, landscape_run_id=run_id
    )
    assert len(response.artifacts) == 1
    assert response.artifacts[0].exists_now is False
```

(Note: `landscape.insert_test_artifact` is a test helper added if not
already present — see existing patterns in `tests/integration/landscape/`
for the canonical insert helpers.)

- [ ] **Step C2.2: Run; expect ImportError on outputs module**

Run: `.venv/bin/python -m pytest tests/unit/web/execution/test_outputs_loader.py -xvs`
Expected: FAIL — `ModuleNotFoundError: elspeth.web.execution.outputs`.

- [ ] **Step C2.3: Implement the loader**

```python
# src/elspeth/web/execution/outputs.py
"""Loader for /api/runs/{rid}/outputs — returns the manifest of artefacts
written by a run, sourced from the Landscape audit DB.

Mirrors the structure of src/elspeth/web/execution/diagnostics.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

from elspeth.core.landscape.factory import landscape_for_settings
from elspeth.web.execution.schemas import RunOutputArtifact, RunOutputsResponse


def load_run_outputs_for_settings(
    settings,
    *,
    run_id: UUID,
    landscape_run_id: UUID,
) -> RunOutputsResponse:
    landscape = landscape_for_settings(settings)
    rows = landscape.query_repository.list_artifacts_for_run(landscape_run_id)
    artifacts: list[RunOutputArtifact] = []
    for row in rows:
        written = Path(row.written_path) if row.written_path else None
        exists = bool(written and written.exists())
        size = written.stat().st_size if exists else 0
        artifacts.append(
            RunOutputArtifact(
                artifact_id=row.artifact_id,
                sink_name=row.sink_node_id,
                artifact_type=row.artifact_type,
                configured_path=row.configured_path or "",
                written_path=row.written_path or "",
                written_at=row.created_at,
                size=size,
                sha256=row.output_data_hash or "",
                exists_now=exists,
            )
        )
    return RunOutputsResponse(
        run_id=run_id,
        landscape_run_id=landscape_run_id,
        artifacts=artifacts,
    )
```

(Note: `query_repository.list_artifacts_for_run` is the new query method.
If it doesn't yet exist, add it as the next sub-step before the test passes.)

- [ ] **Step C2.4: Add list_artifacts_for_run to query_repository**

```python
# src/elspeth/core/landscape/query_repository.py — append a method
def list_artifacts_for_run(self, run_id: UUID) -> list[ArtifactRow]:
    """Return all sink-write artefacts belonging to ``run_id``."""
    stmt = select(artifacts_table).where(artifacts_table.c.run_id == str(run_id))
    with self._engine.connect() as conn:
        result = conn.execute(stmt).mappings().all()
    return [ArtifactRow(**row) for row in result]
```

(The exact `ArtifactRow` dataclass and `_engine` access mirrors patterns
already in `query_repository.py` — read the existing file and copy its
conventions.)

- [ ] **Step C2.5: Run loader test; expect pass**

Run: `.venv/bin/python -m pytest tests/unit/web/execution/test_outputs_loader.py -xvs`
Expected: PASS.

- [ ] **Step C2.6: Commit**

```bash
git add src/elspeth/web/execution/outputs.py \
        src/elspeth/core/landscape/query_repository.py \
        tests/unit/web/execution/test_outputs_loader.py
git commit -m "feat(web): add load_run_outputs_for_settings loader"
```

---

### Task C3: Manifest endpoint route

**Files:**
- Modify: `src/elspeth/web/execution/routes.py`
- Test: `tests/integration/web/test_run_outputs_endpoint.py`

- [ ] **Step C3.1: Write failing integration test**

```python
# tests/integration/web/test_run_outputs_endpoint.py
import hashlib
from pathlib import Path

import pytest
from httpx import AsyncClient

# Reuse existing test fixtures from tests/integration/web/conftest.py.

@pytest.mark.asyncio
async def test_get_run_outputs_returns_manifest(
    test_app, authenticated_user, test_landscape, tmp_path
):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    sink_file = outputs_dir / "results.jsonl"
    sink_file.write_bytes(b'{"interaction_id":"INT-1001"}\n')

    run_id = test_landscape.insert_test_run(owner=authenticated_user.user_id)
    test_landscape.insert_test_artifact(
        run_id=run_id,
        sink_node_id="results",
        artifact_type="jsonl",
        written_path=str(sink_file),
        configured_path=str(sink_file),
        sha256=hashlib.sha256(sink_file.read_bytes()).hexdigest(),
    )

    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        rsp = await ac.get(
            f"/api/runs/{run_id}/outputs",
            headers={"Authorization": f"Bearer {authenticated_user.jwt}"},
        )
    assert rsp.status_code == 200
    body = rsp.json()
    assert body["run_id"] == str(run_id)
    assert len(body["artifacts"]) == 1
    assert body["artifacts"][0]["sink_name"] == "results"
    assert body["artifacts"][0]["exists_now"] is True


@pytest.mark.asyncio
async def test_get_run_outputs_404_for_unknown_run(test_app, authenticated_user):
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        rsp = await ac.get(
            "/api/runs/00000000-0000-0000-0000-000000000000/outputs",
            headers={"Authorization": f"Bearer {authenticated_user.jwt}"},
        )
    assert rsp.status_code == 404


@pytest.mark.asyncio
async def test_get_run_outputs_403_when_other_user(
    test_app, authenticated_user, other_user, test_landscape
):
    run_id = test_landscape.insert_test_run(owner=other_user.user_id)
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        rsp = await ac.get(
            f"/api/runs/{run_id}/outputs",
            headers={"Authorization": f"Bearer {authenticated_user.jwt}"},
        )
    assert rsp.status_code == 403
```

- [ ] **Step C3.2: Run; expect 404 from FastAPI (route not registered)**

Run: `.venv/bin/python -m pytest tests/integration/web/test_run_outputs_endpoint.py -xvs`
Expected: FAIL with status_code == 404 because the route doesn't exist yet.

- [ ] **Step C3.3: Add the route**

```python
# src/elspeth/web/execution/routes.py — insert after get_run_diagnostics
# (around line 498)

from elspeth.web.execution.outputs import load_run_outputs_for_settings
from elspeth.web.execution.schemas import RunOutputsResponse  # add to existing imports

@router.get(
    "/api/runs/{run_id}/outputs",
    response_model=RunOutputsResponse,
)
async def get_run_outputs(
    run_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
    service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
) -> RunOutputsResponse:
    """Return the manifest of sink artefacts written by a run."""
    await _verify_run_ownership(run_id, user, request)
    try:
        status = await service.get_status(run_id)
    except ValueError:
        raise _run_not_found_http() from None
    landscape_run_id = status.landscape_run_id or status.run_id
    return await run_sync_in_worker(
        load_run_outputs_for_settings,
        request.app.state.settings,
        run_id=status.run_id,
        landscape_run_id=landscape_run_id,
    )
```

- [ ] **Step C3.4: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/integration/web/test_run_outputs_endpoint.py -xvs`
Expected: PASS (3 tests).

- [ ] **Step C3.5: Commit**

```bash
git add src/elspeth/web/execution/routes.py \
        tests/integration/web/test_run_outputs_endpoint.py
git commit -m "feat(web): add GET /api/runs/{rid}/outputs manifest endpoint"
```

---

### Task C4: Content-streaming endpoint with path-allowlist enforcement

**Files:**
- Modify: `src/elspeth/web/execution/routes.py`
- Modify: `tests/integration/web/test_run_outputs_endpoint.py`

- [ ] **Step C4.1: Write failing test for content fetch + path-allowlist refusal**

```python
# Append to tests/integration/web/test_run_outputs_endpoint.py
@pytest.mark.asyncio
async def test_get_run_output_content_streams_file(
    test_app, authenticated_user, test_landscape, tmp_path, fake_settings
):
    # fake_settings.data_dir == tmp_path
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    sink_file = outputs_dir / "results.jsonl"
    sink_file.write_bytes(b'{"interaction_id":"INT-1001"}\n')

    run_id = test_landscape.insert_test_run(owner=authenticated_user.user_id)
    artifact_id = test_landscape.insert_test_artifact(
        run_id=run_id,
        sink_node_id="results",
        artifact_type="jsonl",
        written_path=str(sink_file),
        configured_path=str(sink_file),
        sha256="…",
    )

    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        rsp = await ac.get(
            f"/api/runs/{run_id}/outputs/{artifact_id}/content",
            headers={"Authorization": f"Bearer {authenticated_user.jwt}"},
        )
    assert rsp.status_code == 200
    assert rsp.content == b'{"interaction_id":"INT-1001"}\n'


@pytest.mark.asyncio
async def test_get_run_output_content_refuses_path_outside_allowlist(
    test_app, authenticated_user, test_landscape, tmp_path
):
    outside = tmp_path / "elsewhere/results.jsonl"
    outside.parent.mkdir()
    outside.write_bytes(b"escaped\n")

    run_id = test_landscape.insert_test_run(owner=authenticated_user.user_id)
    artifact_id = test_landscape.insert_test_artifact(
        run_id=run_id,
        sink_node_id="results",
        artifact_type="jsonl",
        written_path=str(outside),
        configured_path=str(outside),
        sha256="0" * 64,
    )

    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        rsp = await ac.get(
            f"/api/runs/{run_id}/outputs/{artifact_id}/content",
            headers={"Authorization": f"Bearer {authenticated_user.jwt}"},
        )
    assert rsp.status_code == 403
    assert rsp.json()["detail"]["error_type"] == "output_path_outside_allowlist"
```

- [ ] **Step C4.2: Run; expect 404**

Run: `.venv/bin/python -m pytest tests/integration/web/test_run_outputs_endpoint.py::test_get_run_output_content_streams_file -xvs`
Expected: FAIL with status_code == 404.

- [ ] **Step C4.3: Implement content endpoint**

```python
# src/elspeth/web/execution/routes.py — append after get_run_outputs

from fastapi.responses import FileResponse
from elspeth.web.paths import allowed_sink_directories

@router.get("/api/runs/{run_id}/outputs/{artifact_id}/content")
async def get_run_output_content(
    run_id: UUID,
    artifact_id: str,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
    service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
) -> FileResponse:
    """Stream the bytes of one artefact written by a run.

    Path-allowlist guard: refuses any artefact whose ``written_path`` is
    not under ``data_dir/{outputs,blobs}`` (the canonical sink targets).
    Defence-in-depth — the path was already allowlisted at write time,
    but the audit row is read-mutable in principle and the read-side
    guard MUST NOT trust it.
    """
    await _verify_run_ownership(run_id, user, request)
    try:
        status = await service.get_status(run_id)
    except ValueError:
        raise _run_not_found_http() from None
    landscape_run_id = status.landscape_run_id or status.run_id

    response = await run_sync_in_worker(
        load_run_outputs_for_settings,
        request.app.state.settings,
        run_id=status.run_id,
        landscape_run_id=landscape_run_id,
    )
    artifact = next(
        (a for a in response.artifacts if a.artifact_id == artifact_id), None
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail={"error_type": "artifact_not_found"})

    settings = request.app.state.settings
    data_dir = settings.data_dir
    written = Path(artifact.written_path).resolve()
    allowed = allowed_sink_directories(data_dir)
    if not any(written.is_relative_to(base) for base in allowed):
        raise HTTPException(
            status_code=403,
            detail={"error_type": "output_path_outside_allowlist",
                    "written_path": str(written)},
        )
    if not written.exists():
        raise HTTPException(
            status_code=410,
            detail={"error_type": "artifact_purged_or_moved",
                    "written_path": str(written)},
        )
    return FileResponse(written, filename=written.name)
```

- [ ] **Step C4.4: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/integration/web/test_run_outputs_endpoint.py -xvs`
Expected: PASS (5 tests).

- [ ] **Step C4.5: Commit**

```bash
git add src/elspeth/web/execution/routes.py tests/integration/web/test_run_outputs_endpoint.py
git commit -m "feat(web): add content-streaming endpoint with path-allowlist guard"
```

---

### Task C5: mypy + ruff sweep

- [ ] **Step C5.1: Run typecheck and lint**

Run:
```bash
.venv/bin/python -m mypy src/elspeth/web/execution/
.venv/bin/python -m ruff check src/elspeth/web/execution/ tests/unit/web/execution/ tests/integration/web/test_run_outputs_endpoint.py
.venv/bin/python -m ruff check src/elspeth/core/landscape/
```
Expected: clean. Fix any new findings before proceeding.

- [ ] **Step C5.2: Tier-model enforcement check**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: no new violations. The new `outputs.py` is L3 (web/) importing L1 (core/landscape/); compliant with the layer rules.

- [ ] **Step C5.3: Commit any fixes**

```bash
git commit -am "chore: typecheck + lint clean for run-outputs endpoint" || true
```

---

## Phase D — Harness wiring

### Task D1: Extend hardmode finalize_scenario.sh

**Files:**
- Modify: `evals/2026-05-03-composer/hardmode/finalize_scenario.sh`

- [ ] **Step D1.1: Add the outputs-pull block before the ledger-Python block**

Insert after the `/diagnostics` curl (around line 32 of the existing file):

```bash
# Outputs manifest + content
mkdir -p $out/outputs
curl -fsS -H "Authorization: Bearer $J" \
  "https://elspeth.foundryside.dev/api/runs/$rid/outputs" \
  -o $out/outputs/MANIFEST.json 2>/dev/null || \
  echo '{"artifacts":[]}' > $out/outputs/MANIFEST.json

jq -r '.artifacts[]? | "\(.artifact_id) \(.written_path | split("/") | .[-1])"' \
  $out/outputs/MANIFEST.json | while read aid name; do
  [ -z "$aid" ] && continue
  curl -fsS -H "Authorization: Bearer $J" \
    "https://elspeth.foundryside.dev/api/runs/$rid/outputs/$aid/content" \
    -o "$out/outputs/$name" 2>/dev/null || echo "[$scenario_id] failed to fetch $aid" >&2
done
```

- [ ] **Step D1.2: Extend the Python block to record artefact summary in ledger**

Inside the embedded `python3 - <<PY` block, after the existing run-outcome
section, add:

```python
# Outputs summary
manifest_path = out/"outputs"/"MANIFEST.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    ledger["artifacts_count"] = len(manifest.get("artifacts", []))
    ledger["artifacts"] = [
        {"sink_name": a["sink_name"], "size": a["size"], "exists_now": a["exists_now"]}
        for a in manifest.get("artifacts", [])
    ]
```

- [ ] **Step D1.3: Manual smoke against staging (optional, requires fresh run)**

If the operator runs a fresh hardmode scenario end-to-end, the new
finalize step should produce `outputs/` with files + manifest in
addition to the existing files. Skip if no fresh run is convenient —
this code path is also exercised by Task D3.

- [ ] **Step D1.4: Commit**

```bash
git add evals/2026-05-03-composer/hardmode/finalize_scenario.sh
git commit -m "feat(eval-harness): hardmode finalize pulls per-run outputs"
```

---

### Task D2: Create basic-mode finalize_scenario.sh

**Files:**
- Create: `evals/2026-05-03-composer/basic/finalize_scenario.sh`

- [ ] **Step D2.1: Write the new script**

```bash
#!/usr/bin/env bash
# Basic-mode equivalent of hardmode/finalize_scenario.sh.
# Captures run.json, diagnostics.json, final_yaml.json, messages.json, and the
# new outputs/{MANIFEST.json,*} for one basic-mode scenario after /execute.
#
# Usage: env JWT=<token> finalize_scenario.sh <scenario_id>

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scenario_id=$1
out=$ROOT/$scenario_id
sid=$(cat $out/sid.txt)
J=${JWT:?JWT must be exported}

# /messages
curl -fsS -H "Authorization: Bearer $J" \
  "https://elspeth.foundryside.dev/api/sessions/$sid/messages" \
  -o $out/messages.json 2>/dev/null || true

# /state/yaml — captures the final composed pipeline
curl -fsS -H "Authorization: Bearer $J" \
  "https://elspeth.foundryside.dev/api/sessions/$sid/state/yaml" \
  -o $out/final_yaml.json 2>/dev/null || echo '{}' > $out/final_yaml.json

# Find the run_id for this session (latest run)
runs=$(curl -fsS -H "Authorization: Bearer $J" \
  "https://elspeth.foundryside.dev/api/sessions/$sid/runs" 2>/dev/null || echo '[]')
rid=$(echo "$runs" | jq -r '.[0].run_id // empty')

if [[ -n "$rid" ]]; then
  curl -fsS -H "Authorization: Bearer $J" \
    "https://elspeth.foundryside.dev/api/runs/$rid" -o $out/run.json
  curl -fsS -H "Authorization: Bearer $J" \
    "https://elspeth.foundryside.dev/api/runs/$rid/diagnostics" \
    -o $out/diagnostics.json 2>/dev/null || true

  mkdir -p $out/outputs
  curl -fsS -H "Authorization: Bearer $J" \
    "https://elspeth.foundryside.dev/api/runs/$rid/outputs" \
    -o $out/outputs/MANIFEST.json 2>/dev/null || \
    echo '{"artifacts":[]}' > $out/outputs/MANIFEST.json

  jq -r '.artifacts[]? | "\(.artifact_id) \(.written_path | split("/") | .[-1])"' \
    $out/outputs/MANIFEST.json | while read aid name; do
    [ -z "$aid" ] && continue
    curl -fsS -H "Authorization: Bearer $J" \
      "https://elspeth.foundryside.dev/api/runs/$rid/outputs/$aid/content" \
      -o "$out/outputs/$name" 2>/dev/null || \
      echo "[$scenario_id] failed to fetch $aid" >&2
  done
fi

echo "[$scenario_id] done (rid=${rid:-NONE})"
```

- [ ] **Step D2.2: Make executable + commit**

```bash
chmod +x evals/2026-05-03-composer/basic/finalize_scenario.sh
git add evals/2026-05-03-composer/basic/finalize_scenario.sh
git commit -m "feat(eval-harness): basic-mode finalize_scenario.sh mirrors hardmode"
```

Note: this script depends on `GET /api/sessions/{sid}/runs` existing
on the server. If the route is not present, add it as a minor extension
of the session routes (file an issue/sub-task and stub here). For the
current 2026-05-03 evals, run-ids are also available in the README's
"Audit-trail evidence" table; the script can be modified to take `--rid`
as an override for retroactive use.

---

### Task D3: End-to-end harness smoke (optional, requires staging access)

- [ ] **Step D3.1: Pick one scenario, run it through harness.sh → finalize**

This task verifies the harness wiring in production-like conditions.
Skip if the operator does not have a convenient fresh staging
environment; the unit + integration tests already prove the API
behaviour.

```bash
cd evals/2026-05-03-composer/hardmode
JWT=$JWT bash harness.sh
JWT=$JWT bash post_message.sh p1_t1_happy "Build a pipeline that …" 1
# ... iterate post_message until DONE ...
JWT=$JWT bash finalize_scenario.sh p1_t1_happy

# Check the new outputs/ subdirectory was populated.
ls -la results/p1_t1_happy/outputs/
```

- [ ] **Step D3.2: No commit needed unless harness changes were required.**

---

## Phase E — Documentation

### Task E1: Update evals/2026-05-03-composer/README.md

**Files:**
- Modify: `evals/2026-05-03-composer/README.md`

- [ ] **Step E1.1: Update §Contents tree to show outputs/ subdirectories**

Replace lines 7–31 (the Contents tree) so each scenario block notes the
new `outputs/` subdirectory. Example for hardmode block:

```
└── hardmode/                 Hard-mode eval evidence (persona-driven, 3×3 matrix)
    ├── README.md
    ├── personas/
    ├── scenarios/
    ├── results/              9 per-scenario subdirectories with full session/message/state/run records
    │   └── p<N>_t<M>_<class>/
    │       ├── outputs/      backfilled per-row engine outputs + MANIFEST.json
    │       ├── ledger.json
    │       └── ...
    ├── harness.sh
    ├── post_message.sh
    ├── finalize_scenario.sh
    └── aggregate.json
```

Mirror for the basic block.

- [ ] **Step E1.2: Add §Backfilled vs live-captured outputs**

Insert a new section after §"Provenance and what was redacted" (around line 71):

```markdown
## Backfilled vs live-captured outputs

The per-scenario `outputs/` subdirectories were populated in two phases:

1. **Retroactive backfill (2026-05-06)** — for the 2026-05-03 runs in this
   tree, files were copied from the staging deploy's `data/outputs/`
   directory and timestamp-correlated against each run's start/finish
   window. Each `outputs/MANIFEST.json` records the configured path, the
   actually-written path (which can differ via `auto_increment` collision
   policy), sha256, mtime, and a `correlation_confidence` flag. Only
   HIGH-confidence files were copied; LOW-confidence candidates are
   recorded in `skipped_low_confidence` for forensic completeness but not
   archived.
2. **Live capture (future evals)** — `evals/<eval-id>/<basic|hardmode>/`
   harness scripts now call `GET /api/runs/{rid}/outputs` (manifest) and
   `GET /api/runs/{rid}/outputs/{artifact_id}/content` (bytes) at finalize
   time. These are run-id-stamped through the audit DB and do not need
   timestamp correlation. See `hardmode/finalize_scenario.sh` and
   `basic/finalize_scenario.sh`.

Backfilled files are **byte-equivalent to what the engine wrote** when
sha256 matches the audit-DB `output_data_hash` for the same run. Where
hash equivalence cannot be confirmed (the original run did not record a
hash, or staging-disk overwrites have lost the bytes), the manifest's
`correlation_confidence` is the best-available evidence.
```

- [ ] **Step E1.3: Update §How to verify any claim with row-level recipes**

After the existing worked examples (around line 68), add:

```bash
# "Did INT-1002 really route to fraud_only.jsonl?"
jq 'select(.interaction_id=="INT-1002")' \
  evals/2026-05-03-composer/basic/s4/outputs/q3_fraud_security_flags.csv \
  || head -2 evals/2026-05-03-composer/basic/s4/outputs/q3_fraud_security_flags.csv

# "Which rows did the hardmode P1-T1 v3 fix actually classify?"
cat evals/2026-05-03-composer/hardmode/results/p1_t1_happy/outputs/MANIFEST.json | \
  jq '.artifacts[] | {sink_name, written_path, sha256}'

# "Verify no auto-increment collision divergence for this run"
jq '.files[] | select(.configured_path != .actual_path)' \
  evals/2026-05-03-composer/hardmode/results/p1_t1_happy/outputs/MANIFEST.json
```

- [ ] **Step E1.4: Update §Audit-trail evidence outside this folder**

After the run-id table (around line 109), add:

```markdown
For each of the runs above, per-row outputs are now also archived
in-tree at:
- `basic/s2/outputs/`
- `basic/s3_prime/outputs/`
- `basic/s4/outputs/`
- `hardmode/results/p1_t1_happy/outputs/` (proof-of-fix)
- (etc.)
```

- [ ] **Step E1.5: Commit**

```bash
git add evals/2026-05-03-composer/README.md
git commit -m "docs(eval): README points at backfilled outputs and live-capture flow"
```

---

### Task E2: Update hardmode/README.md and create basic/README.md

**Files:**
- Modify: `evals/2026-05-03-composer/hardmode/README.md`
- Create: `evals/2026-05-03-composer/basic/README.md`

- [ ] **Step E2.1: Add per-scenario file inventory to hardmode README**

Identify the existing inventory section and append the new `outputs/`
files (matches Phase A6 + D1 outputs):

```markdown
### Per-scenario file inventory (post 2026-05-06)

Each `results/<scen>/` directory contains:

| File | Source | Description |
|---|---|---|
| `outputs/MANIFEST.json` | backfill 2026-05-06 OR live D1 capture | Manifest of sink artefacts for this run |
| `outputs/<sink>.jsonl` | (same) | Per-row engine output stream(s) |
| `outputs/<sink>.csv`   | (same) | Per-row engine output stream(s) |
| ... existing files ... | ... | ... |
```

- [ ] **Step E2.2: Create basic-mode README**

```markdown
# Basic-mode eval — 2026-05-03

Mirror of hardmode/README.md but adapted to the LLM-driver basic harness.

## Per-scenario file inventory

Each `s<N>/` directory contains:

| File | Source | Description |
|---|---|---|
| `sid.txt` | original eval | Composer session ID |
| `blob.json` / `create.json` / `execute.json` | original eval | HTTP request/response captures |
| `msg<N>.body` / `msg<N>.json` | original eval | Per-turn composer prompts + responses |
| `final_yaml.json` | recovered 2026-05-06 (Task A7) | Final composed pipeline YAML |
| `run.json` | recovered 2026-05-06 (Task A7) | Run summary for the run_id |
| `diagnostics.json` | recovered 2026-05-06 (Task A7) | Per-row failure details |
| `outputs/MANIFEST.json` | backfill 2026-05-06 (Task A8) OR live D2 capture | Manifest of sink artefacts |
| `outputs/<sink>.jsonl` | (same) | Per-row engine output stream(s) |

## Verification recipe

```bash
# "What did the LLM classifier output for INT-1001 in S2?"
grep '"INT-1001"' evals/2026-05-03-composer/basic/s2/outputs/results.jsonl
```
```

- [ ] **Step E2.3: Commit**

```bash
git add evals/2026-05-03-composer/hardmode/README.md \
        evals/2026-05-03-composer/basic/README.md
git commit -m "docs(eval): per-scenario inventories include outputs/ directory"
```

---

### Task E3: File follow-up issues

- [ ] **Step E3.1: File follow-up for run-id-stamped output dirs**

```bash
filigree create "Run-id-stamped sink output directories — eliminate timestamp correlation in eval evidence" \
  --type=task --priority=3 \
  --description "Today, sinks write to data/outputs/<configured_path> with auto_increment collision. Adding a {run_id}/ prefix would make eval-evidence backfill deterministic (no timestamp correlation needed) and eliminate the cross-run overwrite risk. Architectural change touching plugins/sinks/* and the path-allowlist (web/paths.py:41 allowed_sink_directories). Consider as part of the next sink-config refactor."
```

- [ ] **Step E3.2: Close elspeth-77d2641032 with a summary**

```bash
filigree add-comment elspeth-77d2641032 \
  "Closed by 2026-05-06 multi-phase work: backfilled hardmode + basic outputs (Phase A), added /api/runs/{rid}/outputs endpoint (Phase C), extended both finalize harnesses (Phase D), updated README + per-tree READMEs (Phase E). Follow-up filed for run-id-stamped output dirs."

filigree close elspeth-77d2641032 --reason="Per-row outputs archived; durable harness API in place; README updated."
```

---

## Verification (end-to-end, post-implementation)

### A — Phase A landed

- `find evals/2026-05-03-composer -name MANIFEST.json | wc -l` ≥ 9
  (hardmode) + however many basic scenarios were recoverable.
- For at least the proof-of-fix run `023eb897-...`:
  `jq '.files | length' evals/2026-05-03-composer/hardmode/results/p1_t1_happy/outputs/MANIFEST.json`
  ≥ 1.
- For at least the basic-S4 run: row count in
  `evals/2026-05-03-composer/basic/s4/outputs/<routed-jsonl>` matches
  `rows_routed_success + rows_routed_failure` from the run.json.

### B — Phase C landed

- `.venv/bin/python -m pytest tests/unit/web/execution/ tests/integration/web/test_run_outputs_endpoint.py -xvs`
  passes.
- Manual against staging:
  ```bash
  RID=023eb897-3049-4ad5-a502-e9eb81a4faee
  curl -s -H "Authorization: Bearer $JWT" \
    https://elspeth.foundryside.dev/api/runs/$RID/outputs | \
    jq '.artifacts | length'
  ```
  Returns ≥ 1.

### C — Phase D landed

- A fresh hardmode scenario run (`harness.sh` + N × `post_message.sh` +
  `finalize_scenario.sh`) populates `results/<scen>/outputs/` with
  `MANIFEST.json` + content files, end-to-end without manual
  intervention.

### D — Phase E landed

- A fresh reader given only the `evals/2026-05-03-composer/` directory
  (no audit-DB access, no staging access) can:
  - Find the per-row JSONL backing the walkthrough's INT-1001/INT-1002
    routing claim.
  - Confirm the row-level claim by running the recipe in §"How to verify
    any claim".

---

## Self-review notes

- **Spec coverage:** Phase A covers issue Option 3 (retroactive backfill)
  + part of Option 1 (script). Phase B–C covers Option 1's durable
  architectural fix. Phase D covers the harness symmetry asked for in
  the AskUserQuestion ("mirror hardmode"). Phase E covers Option 2
  (documentation).
- **Type consistency:** `RunOutputArtifact` field names are consistent
  across schemas, loader, route, and tests. Backfill `MANIFEST.json`
  fields match the API response shape (sink_name, configured_path,
  written_path, sha256, size) so a future tool can ingest both
  uniformly.
- **No placeholders:** every step has either complete code or a
  documented decision branch (Phase B's conditional). The conditional
  in B1 explicitly tells the executor what to read and how to decide.
