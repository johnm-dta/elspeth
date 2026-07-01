"""Tests for shared Codex audit helpers — fail-closed summary + exit semantics.

Covers two CI/CD fail-open defects:

* elspeth-0707b6d15b — ``generate_summary``/``write_findings_index`` trusted a
  structured ``.structured.json`` sidecar over the current Markdown report even
  when the sidecar was stale (left behind by a ``--no-structured-output`` rerun
  or a manual rewrite). A "No concrete bug found" report could then be counted
  as a P1 from an old sidecar, or a real P1 could be masked by an empty one.

* elspeth-4634ee39ee — the four Codex audit runners collected per-target
  analysis exceptions, printed a warning, then returned 0 from ``main``. A
  partial scan looked complete and green to CI/operators.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.codex_audit_common import (
    exit_code_from_stats,
    generate_summary,
    structured_output_path_for_report,
    write_summary_file,
)

NO_DEFECT_MARKER = "No concrete bug found"


def _write_report(output_dir: Path, name: str, markdown: str) -> Path:
    report = output_dir / name
    report.write_text(markdown, encoding="utf-8")
    return report


def _write_sidecar(report: Path, findings: list[dict[str, object]]) -> Path:
    sidecar = structured_output_path_for_report(report)
    sidecar.write_text(json.dumps({"findings": findings}), encoding="utf-8")
    return sidecar


def _make_stale(sidecar: Path, report: Path) -> None:
    """Force the sidecar to predate the Markdown report."""
    report_mtime = report.stat().st_mtime
    os.utime(sidecar, (report_mtime - 100, report_mtime - 100))


# --------------------------------------------------------------------------- #
# elspeth-0707b6d15b — stale sidecar must not override the current Markdown
# --------------------------------------------------------------------------- #


def test_stale_sidecar_is_ignored_in_favor_of_current_markdown(tmp_path: Path) -> None:
    """A sidecar older than its Markdown report must be ignored (fall back to MD)."""
    report = _write_report(tmp_path, "finding.md", f"{NO_DEFECT_MARKER}\n")
    sidecar = _write_sidecar(report, [{"priority": "P1", "summary": "stale ghost"}])
    _make_stale(sidecar, report)

    stats = generate_summary(tmp_path, no_defect_marker=NO_DEFECT_MARKER)

    # The Markdown says "no defect"; the stale sidecar's P1 must not be counted.
    assert stats.get("P1", 0) == 0
    assert stats.get("no_defect", 0) == 1


def test_stale_empty_sidecar_does_not_mask_a_real_markdown_finding(tmp_path: Path) -> None:
    """The dangerous direction: an empty stale sidecar must not hide a real P1."""
    report = _write_report(tmp_path, "finding.md", "Priority: P1\nReal regression here.\n")
    sidecar = _write_sidecar(report, [])  # empty -> would count as no_defect if trusted
    _make_stale(sidecar, report)

    stats = generate_summary(tmp_path, no_defect_marker=NO_DEFECT_MARKER)

    assert stats.get("P1", 0) == 1
    assert stats.get("no_defect", 0) == 0


def test_fresh_sidecar_is_still_authoritative(tmp_path: Path) -> None:
    """A sidecar at least as new as its report keeps the structured fast path."""
    report = _write_report(tmp_path, "finding.md", f"{NO_DEFECT_MARKER}\n")
    sidecar = _write_sidecar(report, [{"priority": "P1", "summary": "fresh real finding"}])
    # Make the sidecar strictly newer than the report.
    report_mtime = report.stat().st_mtime
    os.utime(sidecar, (report_mtime + 5, report_mtime + 5))

    stats = generate_summary(tmp_path, no_defect_marker=NO_DEFECT_MARKER)

    assert stats.get("P1", 0) == 1


def test_sidecar_written_shortly_before_report_is_still_authoritative(tmp_path: Path) -> None:
    """Structured runs write the sidecar before extracting the Markdown report."""
    report = tmp_path / "finding.md"
    sidecar = _write_sidecar(
        report,
        [
            {"priority": "P1", "summary": "first structured finding"},
            {"priority": "P2", "summary": "second structured finding"},
        ],
    )
    report.write_text("Priority: P3\nMarkdown fallback would undercount this report.\n", encoding="utf-8")
    base_mtime = 1_700_000_000
    os.utime(sidecar, (base_mtime, base_mtime))
    os.utime(report, (base_mtime + 1, base_mtime + 1))

    stats = generate_summary(tmp_path, no_defect_marker=NO_DEFECT_MARKER)

    assert stats.get("P1", 0) == 1
    assert stats.get("P2", 0) == 1
    assert stats.get("P3", 0) == 0


# --------------------------------------------------------------------------- #
# elspeth-4634ee39ee — partial scan must exit non-zero
# --------------------------------------------------------------------------- #


def test_exit_code_nonzero_when_targets_failed() -> None:
    assert exit_code_from_stats({"failed": 1, "P1": 2}) == 1
    assert exit_code_from_stats({"failed": 3}) == 1


def test_exit_code_zero_on_clean_run() -> None:
    assert exit_code_from_stats({"P1": 2, "no_defect": 5}) == 0
    assert exit_code_from_stats({"failed": 0}) == 0
    assert exit_code_from_stats({}) == 0


def test_failed_count_not_reported_as_a_defect(tmp_path: Path) -> None:
    """The 'failed' bookkeeping key must not inflate the defect total in SUMMARY.md."""
    write_summary_file(
        output_dir=tmp_path,
        stats={"failed": 4, "P1": 1, "no_defect": 2},
        total_files=7,
        title="T",
        defects_label="Defects",
        clean_label="Clean",
    )
    summary_text = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    # Only the single P1 is a real defect; the 4 failures are not defects.
    assert "| **Defects** | 1 |" in summary_text
