"""Pure-logic helpers for the 2026-05-03 eval per-row output backfill.

Used by ``scripts/eval/backfill_2026_05_03_outputs.py`` to extract sink output
paths from a captured ``final_yaml.json``, enumerate auto-increment-renamed
candidate files on disk, and timestamp-correlate them against a run window.

See ``docs/superpowers/plans/2026-05-06-eval-per-row-output-archival.md``
(Phase A) for the full design rationale, and the ELSPETH auditability
standard in CLAUDE.md for why audit-grade evidence demands the
``correlation_confidence`` flag.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml

CONFIDENCE_GRACE_SECONDS = 60
DEFAULT_CAPTURED_BY = "scripts/eval/backfill_2026_05_03_outputs.py"
SHA256_CHUNK_BYTES = 65536

Confidence = Literal["high", "low"]


def extract_sink_paths_from_final_yaml(
    final_yaml_dict: Mapping[str, object],
    *,
    data_dir: str | Path,
) -> list[tuple[str, str]]:
    """Parse the YAML string inside a ``final_yaml.json`` blob and return
    ``(sink_name, absolute_configured_path)`` for each sink that has a path
    option.

    The ``final_yaml.json`` shape comes from
    ``GET /api/sessions/{sid}/state/yaml`` — a JSON object with a single
    ``"yaml"`` string field carrying the composed pipeline.

    Pipeline schema (verified against the 2026-05-03 captured YAMLs):
    ``sinks`` is a top-level **mapping** of ``sink_name → sink_config``,
    where each ``sink_config`` carries ``options.path`` as a path that may
    be absolute or relative to ``data_dir``. We resolve relative paths the
    same way the runtime does (see ``elspeth.web.paths.resolve_data_path``):
    relative paths are joined to ``data_dir`` and resolved.

    Why ``data_dir`` is a parameter (not hard-coded): we want this function
    testable with a ``tmp_path`` fixture, and it's the same primitive the
    backfill driver and the future ``/api/runs/{rid}/outputs`` endpoint
    will share.
    """
    yaml_text = final_yaml_dict.get("yaml")
    if not isinstance(yaml_text, str):
        return []
    config = yaml.safe_load(yaml_text) or {}
    if not isinstance(config, Mapping):
        return []
    sinks = config.get("sinks")
    if not isinstance(sinks, Mapping):
        return []
    base = Path(data_dir).resolve()
    result: list[tuple[str, str]] = []
    for sink_name, sink_config in sinks.items():
        if not isinstance(sink_name, str) or not isinstance(sink_config, Mapping):
            continue
        options = sink_config.get("options") or {}
        if not isinstance(options, Mapping):
            continue
        path = options.get("path")
        if not isinstance(path, str):
            continue
        path_obj = Path(path)
        absolute = path_obj.resolve() if path_obj.is_absolute() else (base / path_obj).resolve()
        result.append((sink_name, str(absolute)))
    return result


def enumerate_candidate_files(configured_path: str) -> list[Path]:
    """Return the base file (if it exists) plus any auto-increment siblings
    matching ``stem-N.ext`` in the same directory.

    Mirrors the rename behaviour of
    ``elspeth.plugins.infrastructure.output_paths.next_available_output_path``:
    when ``collision_policy='auto_increment'``, the actually-written path
    becomes ``stem-1.ext`` if the configured path already exists, then
    ``stem-2.ext``, etc. Backfill must consider all of these as candidates
    so a downstream confidence-classification step can pick the right one.
    """
    base = Path(configured_path)
    parent = base.parent
    if not parent.is_dir():
        return []
    suffix = "".join(base.suffixes)
    stem = base.name[: -len(suffix)] if suffix else base.name
    sibling_re = re.compile(rf"^{re.escape(stem)}-\d+{re.escape(suffix)}$")
    candidates: list[Path] = []
    if base.exists():
        candidates.append(base)
    for entry in parent.iterdir():
        if entry.is_file() and sibling_re.match(entry.name):
            candidates.append(entry)
    return sorted(candidates, key=lambda p: p.name)


def classify_correlation_confidence(
    file_mtime: datetime,
    run_started_at: datetime,
    run_finished_at: datetime,
) -> Confidence:
    """Return ``"high"`` when ``file_mtime`` falls inside the run window
    (``[run_started_at, run_finished_at + grace]``), else ``"low"``.

    Why a grace window: sinks may flush slightly after the engine reports
    finish; the 60-second grace tolerates clock skew and post-loop flush.
    Mtimes BEFORE start are unambiguously a different run.

    The classification is the only mechanism standing between
    "byte-equivalent eval evidence" and "fabricated audit evidence" — be
    strict about LOW classifications. Per CLAUDE.md ("inference from
    adjacent fields is still fabrication"), borderline cases should
    classify LOW so the operator surfaces the ambiguity in the manifest's
    ``skipped_low_confidence`` list rather than silently archiving bytes
    of unclear provenance.
    """
    if file_mtime < run_started_at:
        return "low"
    if file_mtime > run_finished_at + timedelta(seconds=CONFIDENCE_GRACE_SECONDS):
        return "low"
    return "high"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(SHA256_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def build_scenario_manifest(
    *,
    scenario_id: str,
    run_id: str,
    run_started_at: datetime,
    run_finished_at: datetime,
    sinks: Sequence[tuple[str, str]],
    captured_by: str = DEFAULT_CAPTURED_BY,
) -> dict[str, Any]:
    """Build a manifest dict for one scenario from a (sink_name, configured_path) list.

    HIGH-confidence files go into ``files`` (eligible for archive copy).
    LOW-confidence files go into ``skipped_low_confidence`` (recorded for
    forensic completeness but NOT recommended for archive copy — their
    bytes may not be from the run we're trying to evidence).
    """
    captured_at = datetime.now(UTC).isoformat()
    files: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for sink_name, configured_path in sinks:
        for candidate in enumerate_candidate_files(configured_path):
            stat = candidate.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            confidence = classify_correlation_confidence(mtime, run_started_at, run_finished_at)
            if confidence == "high":
                files.append(
                    {
                        "sink_name": sink_name,
                        "configured_path": configured_path,
                        "actual_path": str(candidate),
                        "archived_as": f"outputs/{candidate.name}",
                        "size": stat.st_size,
                        "sha256": _sha256_of_file(candidate),
                        "mtime": mtime.isoformat(),
                        "correlation_confidence": confidence,
                        "captured_at": captured_at,
                        "captured_by": captured_by,
                    }
                )
            else:
                skipped.append(
                    {
                        "sink_name": sink_name,
                        "configured_path": configured_path,
                        "actual_path": str(candidate),
                        "mtime": mtime.isoformat(),
                        "reason": (
                            f"mtime {mtime.isoformat()} outside run window [{run_started_at.isoformat()}, {run_finished_at.isoformat()}]"
                        ),
                    }
                )

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
