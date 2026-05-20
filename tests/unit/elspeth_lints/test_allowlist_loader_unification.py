"""Cross-rule tests for the unified allowlist loader (Plan A).

Task 0 (this file): captures every shipped elspeth-lints rule's findings against
the worktree *before* the Plan A remediation edits land. Plan A's later tasks
edit 8+ rule files (deleting helper functions, changing imports). Per project
memory ``feedback_ast_shift_fingerprint_rotation``, adding ``ImportFrom`` nodes
shifts ``Module.body`` indices and silently rotates fingerprints — invalidating
allowlist suppressions downstream.

The committed ``fixtures/fingerprint_baseline.json`` is the load-bearing
artifact: Task 9 re-runs ``capture_all()`` and diffs against the baseline to
prove no fingerprint shifted unintentionally.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ELSPETH_LINTS_SRC = REPO_ROOT / "elspeth-lints" / "src"
BASELINE = Path(__file__).parent / "fixtures" / "fingerprint_baseline.json"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# CLOSED LIST — derived from RULE_ID constants in each rule's metadata.py,
# paired with the canonical scan root that .github/workflows/ci.yaml uses for
# each rule. Rules differ in whether they scan src/elspeth (engine code only) or
# the full repo root (rules whose contract spans tests/, scripts/, config/,
# elspeth-lints/, etc.). Using the wrong root either misses findings or crashes
# on non-UTF8 bytes inside .venv/site-packages. Update only when adding/removing
# a shipped rule OR when CI changes the canonical root; Task 9 diff depends on
# this mapping matching CI.
ALL_RULE_ROOTS: tuple[tuple[str, str], ...] = (
    ("audit_evidence.guard_symmetry", "src/elspeth"),
    ("audit_evidence.gve_attribution", "src/elspeth"),
    ("audit_evidence.nominal_base", "src/elspeth"),
    ("audit_evidence.tier_1_decoration", "src/elspeth"),
    ("composer.catch_order", "src/elspeth"),
    ("composer.exception_channel", "src/elspeth"),
    ("immutability.freeze_guards", "src/elspeth"),
    ("immutability.frozen_annotations", "src/elspeth"),
    ("manifest.contract_manifest", "src/elspeth"),
    ("manifest.symbol_inventory", "."),
    ("manifest.test_to_source_mapping", "."),
    ("meta.no-new-bespoke-cicd-enforcer", "."),
    ("plugin_contract.component_type", "src/elspeth"),
    ("plugin_contract.options_metadata", "."),
    ("plugin_contract.plugin_hashes", "src/elspeth"),
    ("trust_tier.tier_model", "src/elspeth"),
)
ALL_RULE_IDS: tuple[str, ...] = tuple(rid for rid, _ in ALL_RULE_ROOTS)
RAW_FINGERPRINT_RULES: frozenset[str] = frozenset({"trust_tier.tier_model"})


def _run_rule(rule_id: str, root: str, *, allowlist_dir: Path) -> list[dict[str, object]]:
    """Run a single rule against ``root`` and return parsed JSON findings.

    ``root`` is interpreted relative to ``REPO_ROOT`` (the CLI honours both
    absolute paths and paths relative to ``cwd``). Rules listed in
    ``RAW_FINGERPRINT_RULES`` run against an empty allowlist so their committed
    baseline can protect raw fingerprint stability even when the CI gate is
    green after allowlist suppression. The CLI exits non-zero whenever a rule
    emits at least one finding; we ignore the exit code and parse stdout,
    matching the CI's separation of "did the lint find anything" from "did the
    lint process crash".
    """
    env = {**os.environ, "PYTHONPATH": str(ELSPETH_LINTS_SRC)}
    resolved_root = root if Path(root).is_absolute() else str(REPO_ROOT / root)
    allowlist_args = ["--allowlist-dir", str(allowlist_dir)] if rule_id in RAW_FINGERPRINT_RULES else []
    result = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "elspeth_lints.core.cli",
            "check",
            "--rules",
            rule_id,
            *allowlist_args,
            "--format",
            "json",
            "--root",
            resolved_root,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(REPO_ROOT),
    )
    stdout = result.stdout.strip()
    if not stdout:
        return []
    parsed = json.loads(stdout)
    assert isinstance(parsed, list), f"{rule_id}: expected JSON array, got {type(parsed).__name__}"
    return parsed


def capture_all(*, allowlist_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Run every shipped rule, project to ``{file_path, fingerprint}``, sort deterministically.

    ``file_path`` and ``fingerprint`` are always strings in the CLI's JSON
    output (see ``elspeth_lints.core.findings``); the ``cast`` narrows from the
    ``object`` value-type that ``json.loads`` exposes.
    """
    captured: dict[str, list[dict[str, str]]] = {}
    for rule_id, root in ALL_RULE_ROOTS:
        findings = _run_rule(rule_id, root, allowlist_dir=allowlist_dir)
        captured[rule_id] = sorted(
            (
                {
                    "file_path": cast(str, f["file_path"]),
                    "fingerprint": cast(str, f["fingerprint"]),
                }
                for f in findings
            ),
            key=lambda r: (r["file_path"], r["fingerprint"]),
        )
    return captured


_MAX_DIFFS_PER_RULE = 10


@pytest.mark.fingerprint_baseline
@pytest.mark.skipif(
    sys.version_info[:2] != (3, 13),
    reason="elspeth-lints raw fingerprint baselines are version-specific; Python 3.13 is canonical",
)
def test_baseline_capture_is_self_consistent(tmp_path: Path) -> None:
    """Re-run every rule; the result must equal the committed baseline byte-for-byte.

    On failure, emit a per-rule diff of (file_path, fingerprint) pairs that were
    added or removed. Counts alone are insufficient: an AST shift that rotates
    fingerprints without changing the finding count would slip through silently
    with a "was 5, now 5" diagnostic. Capped at ``_MAX_DIFFS_PER_RULE`` per rule
    so a regression in many files doesn't dump megabytes into the test log.
    """
    allowlist_dir = tmp_path / "empty-allowlist"
    allowlist_dir.mkdir()
    committed = json.loads(BASELINE.read_text(encoding="utf-8"))
    fresh = capture_all(allowlist_dir=allowlist_dir)
    if fresh == committed:
        return
    lines: list[str] = ["Fingerprints shifted vs baseline:"]
    for rid in ALL_RULE_IDS:
        old = committed.get(rid, [])
        new = fresh.get(rid, [])
        if old == new:
            continue
        old_set = {(e["file_path"], e["fingerprint"]) for e in old}
        new_set = {(e["file_path"], e["fingerprint"]) for e in new}
        added = sorted(new_set - old_set)
        removed = sorted(old_set - new_set)
        lines.append(f"  {rid}: was {len(old)} findings, now {len(new)}")
        for path, fp in added[:_MAX_DIFFS_PER_RULE]:
            lines.append(f"    + {path}  fp={fp}")
        if len(added) > _MAX_DIFFS_PER_RULE:
            lines.append(f"    + ... and {len(added) - _MAX_DIFFS_PER_RULE} more added")
        for path, fp in removed[:_MAX_DIFFS_PER_RULE]:
            lines.append(f"    - {path}  fp={fp}")
        if len(removed) > _MAX_DIFFS_PER_RULE:
            lines.append(f"    - ... and {len(removed) - _MAX_DIFFS_PER_RULE} more removed")
    pytest.fail("\n".join(lines))
