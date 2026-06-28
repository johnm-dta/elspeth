"""From-tree re-verification for ``sign-bundle`` -- the linchpin.

``verify_bundle_against_tree`` is the all-or-nothing gate ([O1] "staging
asserts; firing verifies"): it re-derives every binding from the *current*
source and refuses on any staleness mismatch *before a single write*. It is a
pure read -- no writes, no HMAC key required (diagnosis runs shape-only; the
binding checks still fire).

Every action ``kind`` in the vocabulary has its own from-tree verify rule,
because the ``diagnose_judge_signatures`` index only covers *existing signable*
entries:

* ``drift_repair`` -- compare the action's ``diagnosis_status`` to the live
  diagnosis row (must be present and in ``_SIGNABLE_DIAGNOSIS_STATUSES``);
* ``justify`` (new_judgment) -- the key has no entry yet, so scan the action's
  file via the shared single-file scan helper and confirm the live finding
  still exists at the staged fingerprint (coverage delta is ignored -- a
  now-covered finding is a benign redundant re-judge, not a failure);
* ``stale_delete`` -- confirm the tree still reports the key as a non-signable
  orphan; a reappeared live finding is a mismatch (never delete a live entry);
* ``rotation`` -- re-derive via ``scan_for_rotations(exclude_judge_gated=True)``
  (the same filtered source ``stage_scan`` uses) and confirm the key is still a
  rotation ``old_key``.

The report carries the computed ``diagnosis`` (reused by the ``sign-bundle``
execute phase's drift_repair lane -- one diagnose call per run) and the filtered
``rotation_plan`` (the *single* whole-dir ``exclude_judge_gated=True`` scan,
computed once iff the bundle has >=1 rotation action, reused by the execute
rotation lane so it never re-scans at write time). ``rotation_plan`` is ``None``
exactly when the bundle has no rotation action. On a filtered plan only
``.rotations`` is authoritative -- the filtered-out judge-gated findings land in
``.new_findings``/``.ambiguous`` as pollution; no consumer may read those.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elspeth_lints.core.judge_signature_diagnosis import (
    _SIGNABLE_DIAGNOSIS_STATUSES,
    JudgeSignatureDiagnosisReport,
    diagnose_judge_signatures,
)
from elspeth_lints.core.review_bundle import BundleAction, ReviewBundle
from elspeth_lints.core.tier_model_scan import scan_single_file_findings
from elspeth_lints.rules.trust_tier.tier_model.rotate import RotationPlan, scan_for_rotations

# A ``stale_delete`` target is safe to remove only while the tree still reports
# it as one of these non-signable orphan statuses (neither is in
# ``_SIGNABLE_DIAGNOSIS_STATUSES``).
_STALE_DELETE_ORPHAN_STATUSES = frozenset({"NO_MATCHING_FINDING", "SOURCE_FILE_MISSING"})


@dataclass(frozen=True, slots=True)
class BundleVerificationReport:
    """Result of re-deriving a bundle's claims from the source tree."""

    mismatches: tuple[str, ...]
    diagnosis: JudgeSignatureDiagnosisReport
    rotation_plan: RotationPlan | None

    @property
    def ok(self) -> bool:
        """True only when no staged claim disagrees with ground truth."""
        return not self.mismatches


def verify_bundle_against_tree(
    bundle: ReviewBundle,
    *,
    root: Path,
    allowlist_dir: Path,
) -> BundleVerificationReport:
    """Re-derive every bundle action's binding from the tree (pure read)."""
    root = Path(root)
    allowlist_dir = Path(allowlist_dir)

    diagnosis = diagnose_judge_signatures(root=root, allowlist_dir=allowlist_dir)
    index: dict[str, Any] = {item.key: item for item in diagnosis.items}

    # Compute the filtered rotation plan exactly once, iff there is a rotation
    # action. Pass the directory as ``allowlist_path`` (load_allowlist accepts a
    # dir); ``exclude_judge_gated=True`` keeps the scan from raising on the
    # canonical mostly-judge-gated corpus.
    rotation_plan: RotationPlan | None = None
    if any(action.kind == "rotation" for action in bundle.actions):
        rotation_plan = scan_for_rotations(
            source_root=root,
            allowlist_path=allowlist_dir,
            exclude_judge_gated=True,
        )

    mismatches: list[str] = []
    for action in bundle.actions:
        if action.kind == "drift_repair":
            mismatches.extend(_verify_drift_repair(action, index))
        elif action.kind == "justify":
            mismatches.extend(_verify_new_judgment(action, root=root))
        elif action.kind == "stale_delete":
            mismatches.extend(_verify_stale_delete(action, index))
        elif action.kind == "rotation":
            mismatches.extend(_verify_rotation(action, rotation_plan))
        else:  # pragma: no cover - BundleAction.__post_init__ rejects unknown kinds
            mismatches.append(f"action {action.key!r}: unknown kind {action.kind!r}")

    return BundleVerificationReport(
        mismatches=tuple(mismatches),
        diagnosis=diagnosis,
        rotation_plan=rotation_plan,
    )


def _verify_drift_repair(action: BundleAction, index: dict[str, Any]) -> list[str]:
    item = index.get(action.key)
    if item is None:
        return [f"drift_repair {action.key!r}: no diagnosis row in the tree (staged status {action.diagnosis_status!r}); re-run stage_scan"]
    if item.status not in _SIGNABLE_DIAGNOSIS_STATUSES:
        return [f"drift_repair {action.key!r}: tree status {item.status!r} is not a signable drift (staged {action.diagnosis_status!r})"]
    if item.status != action.diagnosis_status:
        return [f"drift_repair {action.key!r}: staged diagnosis_status {action.diagnosis_status!r} but the tree reports {item.status!r}"]
    return []


def _verify_new_judgment(action: BundleAction, *, root: Path) -> list[str]:
    if not action.file_path:  # pragma: no cover - enforced by BundleAction.__post_init__
        return [f"new_judgment {action.key!r}: missing file_path"]
    target_file = (root / action.file_path).resolve()
    try:
        findings = scan_single_file_findings(target_file=target_file, root=root)
    except (OSError, ValueError):
        return [f"new_judgment {action.key!r}: source {action.file_path} could not be scanned"]
    live_keys = {_finding_canonical_key(finding) for finding in findings}
    if action.key not in live_keys:
        return [
            f"new_judgment {action.key!r}: no live finding at the staged fingerprint in "
            f"{action.file_path} (vanished or fingerprint-shifted)"
        ]
    return []


def _verify_stale_delete(action: BundleAction, index: dict[str, Any]) -> list[str]:
    item = index.get(action.key)
    if item is None:
        return [f"stale_delete {action.key!r}: no diagnosis row in the tree; cannot confirm the entry is an orphan"]
    if item.status not in _STALE_DELETE_ORPHAN_STATUSES:
        return [f"stale_delete {action.key!r}: tree reports {item.status!r}, not an orphan; the covered finding reappeared"]
    return []


def _verify_rotation(action: BundleAction, rotation_plan: RotationPlan | None) -> list[str]:
    if rotation_plan is None:  # pragma: no cover - a rotation action guarantees a computed plan
        return [f"rotation {action.key!r}: no filtered rotation plan was computed"]
    old_keys = {rotation.old_key for rotation in rotation_plan.rotations}
    if action.key not in old_keys:
        return [f"rotation {action.key!r}: no longer an applicable rotation old_key in a fresh non-judge-gated scan"]
    return []


def _finding_canonical_key(finding: Any) -> str:
    key = finding.canonical_key
    if callable(key):
        key = key()
    if not isinstance(key, str):
        raise ValueError(f"finding.canonical_key must be str; got {type(key).__name__}")
    return key
