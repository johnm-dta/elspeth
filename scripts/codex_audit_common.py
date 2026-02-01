#!/usr/bin/env python3
"""Common utilities for Codex audit scripts.

This module provides shared infrastructure for the three Codex audit scripts:
- codex_test_defect_hunt.py
- codex_integration_seam_hunt.py
- codex_exemption_validator.py

Extracted to eliminate ~1,200 lines of duplication across scripts.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pyrate_limiter import Limiter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from tqdm.asyncio import tqdm as async_tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class AsyncTqdm:  # type: ignore
        """Fallback progress bar when tqdm is not available."""

        def __init__(self, *args: Any, **kwargs: Any):
            self.total = kwargs.get("total", 0)
            self.desc = kwargs.get("desc", "")
            self.n = 0

        def update(self, n: int = 1) -> None:
            self.n += n
            if self.total > 0:
                pct = (self.n / self.total) * 100
                print(f"\r{self.desc}: {self.n}/{self.total} ({pct:.1f}%)", end="", file=sys.stderr)

        def close(self) -> None:
            if self.total > 0:
                print(file=sys.stderr)


# Export the correct class based on import success
if HAS_TQDM:
    AsyncTqdm = async_tqdm  # type: ignore


# === Constants ===

MAX_RETRIES = 3
RETRY_MIN_WAIT_S = 2
RETRY_MAX_WAIT_S = 60
RETRY_MULTIPLIER = 2
STDERR_TRUNCATE_CHARS = 500

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".tox",
    ".nox",
    ".hypothesis",
    "node_modules",
    ".venv",
    "venv",
    ".eggs",
    "htmlcov",
    ".coverage",
}

EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


# === Infrastructure Functions ===


def resolve_path(repo_root: Path, value: str) -> Path:
    """Resolve a path relative to repo_root if not absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def is_cache_path(path: Path) -> bool:
    """Check if path is a cache file or inside a cache directory."""
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True
    return path.suffix in EXCLUDE_SUFFIXES


def utc_now() -> str:
    """Return current UTC timestamp as ISO format string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def escape_cell(value: str) -> str:
    """Escape markdown table cell content."""
    return value.replace("|", "\\|").replace("\n", "\\n").replace("\r", "")


def get_git_commit(repo_root: Path) -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


async def append_log(
    *,
    log_path: Path,
    log_lock: asyncio.Lock,
    timestamp: str,
    status: str,
    file_display: str,
    output_display: str,
    model: str,
    duration_s: float,
    note: str,
) -> None:
    """Append a log entry to the execution log file."""
    line = (
        f"| {escape_cell(timestamp)} | {escape_cell(status)} | "
        f"{escape_cell(file_display)} | {escape_cell(output_display)} | "
        f"{escape_cell(model)} | {duration_s:.2f} | {escape_cell(note)} |\n"
    )
    async with log_lock:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def ensure_log_file(log_path: Path, *, header_title: str) -> None:
    """Ensure log file exists with proper header."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        return
    header = (
        f"# {header_title}\n\n"
        "| Timestamp (UTC) | Status | File | Output | Model | Duration_s | Note |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    log_path.write_text(header, encoding="utf-8")


# === Evidence Processing Functions ===


def load_context(repo_root: Path, extra_files: list[str] | None = None) -> str:
    """Load context from CLAUDE.md and optional extra files for agent prompts.

    Returns concatenated content of CLAUDE.md and any additional context files,
    separated by headers showing filename.
    """
    parts = []

    claude_md = repo_root / "CLAUDE.md"
    if claude_md.exists():
        parts.append(f"--- CLAUDE.md ---\n{claude_md.read_text(encoding='utf-8')}")

    if extra_files:
        for filename in extra_files:
            path = repo_root / filename
            if path.exists():
                parts.append(f"--- {filename} ---\n{path.read_text(encoding='utf-8')}")
            else:
                print(f"Warning: Context file not found: {filename}", file=sys.stderr)

    return "\n\n".join(parts)


def extract_section(report: str, heading: str) -> str:
    """Extract content from a markdown section with the given heading.

    Returns the text between `## {heading}` and the next `##` heading.
    """
    lines = report.splitlines()
    in_section = False
    collected: list[str] = []
    for line in lines:
        if line.strip().startswith("## "):
            if in_section:
                break
            if line.strip() == f"## {heading}":
                in_section = True
                continue
        if in_section:
            collected.append(line)
    return "\n".join(collected).strip()


def replace_section(report: str, heading: str, new_lines: list[str]) -> str:
    """Replace content in a markdown section with the given heading.

    Keeps the heading line, replaces everything until the next heading with new_lines.
    """
    lines = report.splitlines()
    out: list[str] = []
    in_target = False
    for line in lines:
        if line.strip().startswith("## "):
            if in_target:
                in_target = False
            if line.strip() == f"## {heading}":
                out.append(line)
                out.extend(new_lines)
                in_target = True
                continue
        if in_target:
            continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def has_file_line_evidence(evidence: str) -> bool:
    """Check if evidence contains file:line citations."""
    patterns = [
        r"\b[\w./-]+\.[\w]+:\d+\b",  # file.py:123
        r"\b[\w./-]+#L\d+\b",  # file.py#L123
        r"\bline\s+\d+\b",  # line 123
        r"\b[\w./-]+\(line\s+\d+\)",  # file.py (line 123)
        r"\bat\s+[\w./-]+\.[\w]+:\d+",  # at file.py:123
        r"\bin\s+[\w./-]+\.[\w]+:\d+",  # in file.py:123
    ]
    return any(re.search(pattern, evidence, re.IGNORECASE) for pattern in patterns)


def evidence_quality_score(evidence: str) -> int:
    """Score evidence quality 0-3 based on specificity."""
    score = 0
    if has_file_line_evidence(evidence):
        score += 1
    if re.search(r"```python", evidence, re.IGNORECASE):
        score += 1
    if len(evidence) > 100:
        score += 1
    return score


def apply_evidence_gate(output_path: Path, *, summary_prefix: str = "") -> int:
    """Apply evidence gate: downgrade reports without file:line citations.

    Args:
        output_path: Path to the report file to process
        summary_prefix: Prefix for summary message (e.g., "both sides of seam")

    Returns:
        Number of reports that were gated/downgraded
    """
    text = output_path.read_text(encoding="utf-8")
    reports: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "---":
            reports.append("\n".join(current).strip())
            current = []
            continue
        current.append(line)
    if current:
        reports.append("\n".join(current).strip())

    gated_count = 0
    new_reports: list[str] = []
    for report in reports:
        if not report.strip():
            continue

        evidence = extract_section(report, "Evidence")
        quality = evidence_quality_score(evidence)

        if quality == 0:
            gated_count += 1
            suffix = f" {summary_prefix}" if summary_prefix else ""
            report = replace_section(
                report,
                "Summary",
                ["", f"- Needs verification: missing file/line evidence{suffix}."],
            )
            report = replace_section(
                report,
                "Severity",
                ["", "- Severity: trivial", "- Priority: P3"],
            )
        elif quality == 1:
            gated_count += 1
            severity = extract_section(report, "Severity")
            # Use word boundaries to avoid false positives like "The P0 ticket"
            if re.search(r"\bP[01]\b", severity):
                report = replace_section(
                    report,
                    "Severity",
                    ["", "- Severity: minor", "- Priority: P2 (downgraded: minimal evidence)"],
                )

        new_reports.append(report.strip())

    new_text = "\n---\n".join(new_reports).rstrip() + "\n"
    if new_text != text:
        output_path.write_text(new_text, encoding="utf-8")

    return gated_count


# === Subprocess Management Functions ===


async def run_codex_once(
    *,
    file_path: Path,
    output_path: Path,
    model: str | None,
    prompt: str,
    repo_root: Path,
    file_display: str,
    output_display: str,
) -> None:
    """Run codex exec once. Raises on failure. Does not handle retries or logging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "-c",
        'approval_policy="never"',
        "--output-last-message",
        str(output_path),
    ]
    if model is not None:
        cmd.extend(["--model", model])
    cmd.append(prompt)

    # Create subprocess with stdout/stderr capture for diagnostics
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=repo_root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        _stdout, stderr = await process.communicate()

        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")
            # Truncate stderr for log readability
            if len(stderr_text) > STDERR_TRUNCATE_CHARS:
                stderr_text = stderr_text[:STDERR_TRUNCATE_CHARS] + "... (truncated)"

            raise RuntimeError(f"codex exec failed for {file_display} with code {process.returncode}\nstderr: {stderr_text}")

    finally:
        # Ensure subprocess cleanup if still running
        if process.returncode is None:
            process.terminate()
            await process.wait()


async def run_codex_with_retry_and_logging(
    *,
    file_path: Path,
    output_path: Path,
    model: str | None,
    prompt: str,
    repo_root: Path,
    log_path: Path,
    log_lock: asyncio.Lock,
    file_display: str,
    output_display: str,
    rate_limiter: Limiter | None,
    evidence_gate_summary_prefix: str = "",
) -> dict[str, int]:
    """Run codex with retry logic, rate limiting, evidence gate, and logging. Returns stats."""
    if rate_limiter:
        rate_limiter.try_acquire("codex_api")

    start_time = time.monotonic()
    status = "ok"
    note = ""
    gated_count = 0

    # Create retry decorator dynamically
    retry_decorator = retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_MIN_WAIT_S, max=RETRY_MAX_WAIT_S),
        retry=retry_if_exception_type((RuntimeError, asyncio.TimeoutError)),
        reraise=True,
    )

    try:
        # Apply retry decorator to the function call
        await retry_decorator(run_codex_once)(
            file_path=file_path,
            output_path=output_path,
            model=model,
            prompt=prompt,
            repo_root=repo_root,
            file_display=file_display,
            output_display=output_display,
        )

        # Apply evidence gate after successful completion
        gated_count = apply_evidence_gate(output_path, summary_prefix=evidence_gate_summary_prefix)
        if gated_count > 0:
            note = f"evidence_gate={gated_count}"

    except Exception as exc:
        status = "failed"
        note = str(exc)[:200]  # Truncate long error messages
        raise

    finally:
        duration_s = time.monotonic() - start_time
        await append_log(
            log_path=log_path,
            log_lock=log_lock,
            timestamp=utc_now(),
            status=status,
            file_display=file_display,
            output_display=output_display,
            model=model or "",
            duration_s=duration_s,
            note=note,
        )

    return {"gated": gated_count}


T = TypeVar("T")


def chunked[T](items: list[T], size: int) -> list[list[T]]:
    """Split items into chunks of at most `size` elements for batched processing."""
    return [items[i : i + size] for i in range(0, len(items), size)]


# === Reporting Functions ===


def priority_from_report(text: str) -> str:
    """Extract priority level from report text."""
    if match := re.search(r"Priority:\s*(P\d)", text, re.IGNORECASE):
        return match.group(1).upper()
    return "unknown"


def generate_summary(output_dir: Path, *, no_defect_marker: str) -> dict[str, int]:
    """Parse all outputs and count defects by priority.

    Args:
        output_dir: Directory containing markdown reports
        no_defect_marker: String that indicates a clean report (e.g., "No test defect found")

    Returns:
        Dictionary with counts by priority
    """
    stats: Counter[str] = Counter()

    for md_file in output_dir.rglob("*.md"):
        text = md_file.read_text(encoding="utf-8")

        if no_defect_marker in text:
            stats["no_defect"] += 1
        else:
            priority = priority_from_report(text)
            stats[priority] += 1

    return dict(stats)


def write_run_metadata(
    *,
    output_dir: Path,
    repo_root: Path,
    start_time: str,
    end_time: str,
    duration_s: float,
    files_scanned: int,
    model: str | None,
    batch_size: int,
    rate_limit: int | None,
    git_commit: str,
    title: str,
    script_name: str,
) -> None:
    """Write run metadata file for triage tracking."""
    metadata_path = output_dir / "RUN_METADATA.md"
    content = f"""# {title}

## Execution Details

- **Start Time (UTC):** {start_time}
- **End Time (UTC):** {end_time}
- **Duration:** {duration_s:.1f}s
- **Git Commit:** {git_commit}

## Scan Parameters

- **Files Scanned:** {files_scanned}
- **Model:** {model or "default (from codex config)"}
- **Batch Size:** {batch_size}
- **Rate Limit:** {rate_limit if rate_limit else "none"}

## Output Structure

```
{output_dir.relative_to(repo_root)}/
├── RUN_METADATA.md          # This file
├── SUMMARY.md               # Statistics and triage dashboard
├── FINDINGS_INDEX.md        # Table of all findings
├── <file-path>.md           # Individual finding reports
└── ...
```

## Triage Workflow

1. Review `SUMMARY.md` for overview
2. Triage high-priority findings first (P0, P1)
3. Use `FINDINGS_INDEX.md` to track triage status
4. Update individual finding files with triage decisions

---
Generated by: `{script_name}`
"""
    metadata_path.write_text(content, encoding="utf-8")


def write_summary_file(
    *,
    output_dir: Path,
    stats: dict[str, int],
    total_files: int,
    title: str,
    defects_label: str,
    clean_label: str,
) -> None:
    """Write summary statistics file for triage dashboard."""
    summary_path = output_dir / "SUMMARY.md"

    total_defects = sum(v for k, v in stats.items() if k not in ["no_defect", "gated", "unknown"])
    no_defect_count = stats.get("no_defect", 0)
    gated = stats.get("gated", 0)
    unknown = stats.get("unknown", 0)

    # Build priority breakdown
    priority_lines = []
    for priority in ["P0", "P1", "P2", "P3"]:
        count = stats.get(priority, 0)
        if count > 0:
            bar = "█" * min(count, 40)
            priority_lines.append(f"- **{priority}**: {count:3d} {bar}")

    content = f"""# {title}

## Overview

| Metric | Count |
|--------|-------|
| **Total Files Scanned** | {total_files} |
| **{defects_label}** | {total_defects} |
| **{clean_label}** | {no_defect_count} |
| **Downgraded (Evidence Gate)** | {gated} |
| **Unknown Priority** | {unknown} |

## Priority Breakdown

{chr(10).join(priority_lines) if priority_lines else "No defects found."}

## Triage Status

- [ ] Review all P0 findings (critical)
- [ ] Review all P1 findings (high)
- [ ] Triage P2 findings (medium)
- [ ] Triage P3 findings (low)

## Next Steps

1. Open `FINDINGS_INDEX.md` to see all findings in table format
2. Start with P0/P1 findings
3. For each finding:
   - Review the evidence
   - Verify the defect is real (not hallucinated)
   - Create GitHub issue or fix immediately
   - Update finding file with triage decision

---
Last updated: {utc_now()}
"""
    summary_path.write_text(content, encoding="utf-8")


def write_findings_index(
    *,
    output_dir: Path,
    repo_root: Path,
    title: str,
    file_column_label: str,
    no_defect_marker: str,
    clean_section_title: str,
) -> None:
    """Write findings index with table of all reports for easy triage."""
    findings: list[dict[str, str]] = []

    for md_file in sorted(output_dir.rglob("*.md")):
        # Skip metadata files
        if md_file.name in ["RUN_METADATA.md", "SUMMARY.md", "FINDINGS_INDEX.md"]:
            continue

        text = md_file.read_text(encoding="utf-8")

        # Extract summary (first non-heading line after ## Summary)
        summary = "No summary found"
        lines = text.splitlines()
        in_summary = False
        for line in lines:
            if line.strip() == "## Summary":
                in_summary = True
                continue
            if in_summary:
                if line.strip().startswith("## "):
                    break
                if line.strip() and not line.strip().startswith("-"):
                    summary = line.strip()[:100]  # Truncate long summaries
                    break

        # Extract priority
        priority = priority_from_report(text)

        # Check if it's a "no defect" report
        is_clean = no_defect_marker in text

        findings.append(
            {
                "file": str(md_file.relative_to(output_dir)),
                "source_file": str(md_file.relative_to(output_dir).with_suffix("").with_suffix("")),
                "priority": priority if not is_clean else "clean",
                "summary": summary,
                "status": "pending",  # For manual triage updates
            }
        )

    # Sort by priority (P0 > P1 > P2 > P3 > clean)
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "unknown": 4, "clean": 5}
    findings.sort(key=lambda f: (priority_order.get(f["priority"], 99), f["source_file"]))

    # Build table
    index_path = output_dir / "FINDINGS_INDEX.md"
    content = f"""# {title}

Total findings: {len(findings)}

## Triage Table

| Priority | {file_column_label} | Summary | Status | Report |
|----------|{"---" * (len(file_column_label) // 3)}|---------|--------|--------|
"""

    for finding in findings:
        # Skip "clean" reports for triage focus
        if finding["priority"] == "clean":
            continue

        content += (
            f"| {finding['priority']} | `{finding['source_file']}` | "
            f"{finding['summary']} | {finding['status']} | [{finding['file']}]({finding['file']}) |\n"
        )

    content += f"""

## {clean_section_title} ({sum(1 for f in findings if f["priority"] == "clean")})

"""

    clean_files = [f for f in findings if f["priority"] == "clean"]
    if clean_files:
        content += f"| {file_column_label} |\n|{'---' * (len(file_column_label) // 3)}|\n"
        for finding in clean_files:
            content += f"| `{finding['source_file']}` |\n"
    else:
        content += "_No clean files found._\n"

    content += (
        """

## Triage Status Legend

- **pending**: Not yet reviewed
- **confirmed**: Defect verified, issue created
- **invalid**: False positive, not a real defect
- **duplicate**: Already tracked elsewhere
- **fixed**: Defect fixed

---
Last updated: """
        + utc_now()
        + "\n"
    )

    index_path.write_text(content, encoding="utf-8")


def print_summary(stats: dict[str, int], *, icon: str, title: str) -> None:
    """Print formatted summary statistics to stdout."""
    print("\n" + "=" * 60)
    print(f"{icon} {title}")
    print("=" * 60)

    total_defects = sum(v for k, v in stats.items() if k not in ["no_defect", "gated", "unknown"])

    if total_defects > 0:
        print(f"\n🔍 Defects Found: {total_defects}")
        for priority in ["P0", "P1", "P2", "P3"]:
            count = stats.get(priority, 0)
            if count > 0:
                bar = "█" * min(count, 40)
                print(f"  {priority}: {count:3d} {bar}")

    no_defect_count = stats.get("no_defect", 0)
    if no_defect_count > 0:
        print(f"\n✅ Clean Files: {no_defect_count}")

    gated = stats.get("gated", 0)
    if gated > 0:
        print(f"\n⚠️  Downgraded (evidence gate): {gated}")

    unknown = stats.get("unknown", 0)
    if unknown > 0:
        print(f"\n❓ Unknown Priority: {unknown}")

    print("=" * 60 + "\n")
