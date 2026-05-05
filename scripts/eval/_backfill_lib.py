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

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


def extract_sink_paths_from_final_yaml(
    final_yaml_dict: Mapping[str, object],
) -> list[tuple[str, str]]:
    """Parse the YAML string inside a ``final_yaml.json`` blob and return
    ``(sink_name, configured_path)`` for each output that has a path option.

    The ``final_yaml.json`` shape comes from
    ``GET /api/sessions/{sid}/state/yaml`` — a JSON object with a single
    ``"yaml"`` string field carrying the composed pipeline.
    """
    yaml_text = final_yaml_dict.get("yaml")
    if not isinstance(yaml_text, str):
        return []
    config = yaml.safe_load(yaml_text) or {}
    if not isinstance(config, Mapping):
        return []
    outputs = config.get("outputs") or []
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
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
