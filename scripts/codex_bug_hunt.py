#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from codex_audit_common import (  # type: ignore[import-not-found]
    chunked,
    ensure_log_file,
    extract_section,
    generate_summary,
    is_cache_path,
    load_context,
    print_summary,
    priority_from_report,
    resolve_path,
    run_codex_with_retry_and_logging,
)
from pyrate_limiter import Duration, Limiter, Rate


def _is_python_file(path: Path) -> bool:
    """Check if path is a Python source file (not test)."""
    return path.suffix == ".py" and not path.name.startswith("test_")


def _build_prompt(file_path: Path, template: str, context: str) -> str:
    return (
        "You are a static analysis agent doing a deep bug audit.\n"
        f"Target file: {file_path}\n\n"
        "Instructions:\n"
        "- Use the bug report template below verbatim.\n"
        "- Fill in every section. If unknown, write 'Unknown'.\n"
        "- You may read any repo file to confirm integration behavior. Prefer\n"
        "  verification over speculation.\n"
        "- Report bugs only if the primary fix belongs in the target file.\n"
        "  If the root cause is in another file, do not report it unless the\n"
        "  severity is P0. If you report a P0 outside the target file, explain\n"
        "  why and cite the true root-cause file.\n"
        "- Integration issues can reference other files for evidence, but the\n"
        "  actionable fix must be in the target file (unless P0 as above).\n"
        "- If you find multiple distinct bugs, output one full template per bug,\n"
        "  separated by a line with only '---'.\n"
        "- If you find no credible bug, output one template with Summary set to\n"
        f"  'No concrete bug found in {file_path}', Severity 'trivial', Priority 'P3',\n"
        "  and Root Cause Hypothesis 'No bug identified'.\n"
        "- Evidence should cite file paths and line numbers when possible.\n\n"
        "Bug Categories to Check:\n"
        "1. **Audit Trail Violations**:\n"
        "   - Missing payload recording for external API calls\n"
        "   - Incomplete state transitions (missing terminal states)\n"
        "   - Hash computation without payload persistence\n"
        "   - Token lineage breaks (missing parent_token_id)\n"
        "   - Silent data loss (rows disappearing without terminal state)\n"
        "\n"
        "2. **Data Tier Confusion** (CLAUDE.md Three-Tier Trust Model):\n"
        "   - Type coercion in transforms/sinks (only allowed in sources)\n"
        "   - Missing error handling on row operations (division, parsing, etc.)\n"
        "   - Bug-hiding patterns: .get(), getattr(), hasattr() for OUR code\n"
        "   - Defensive programming masking bugs in system-owned data\n"
        "   - Silent exceptions on internal state access\n"
        "\n"
        "3. **Protocol/Contract Violations**:\n"
        "   - Missing required hookimpl decorators or specs\n"
        "   - Schema mismatch (output_schema != actual output types)\n"
        "   - Plugin interface violations (wrong return types, missing methods)\n"
        "   - Config validation gaps (Pydantic validators not enforcing constraints)\n"
        "   - Breaking plugin contracts (e.g., stateless transforms with state)\n"
        "\n"
        "4. **State Management Issues**:\n"
        "   - Race conditions in async code\n"
        "   - Shared mutable state in stateless plugins\n"
        "   - Missing synchronization on concurrent access\n"
        "   - Incomplete cleanup in context managers\n"
        "   - Resource leaks (unclosed connections, files, locks)\n"
        "\n"
        "5. **Error Handling Gaps**:\n"
        "   - External calls without try/except (API, file I/O, network)\n"
        "   - Row operations without error handling (their data can fail)\n"
        "   - Missing quarantine on validation failures\n"
        "   - Silent failures (empty except blocks, catch-all handlers)\n"
        "   - Error messages without actionable context\n"
        "\n"
        "6. **Validation Gaps**:\n"
        "   - Source input not validated against schemas\n"
        "   - Missing type checks at trust boundaries\n"
        "   - No bounds checking on numeric inputs\n"
        "   - Unchecked assumptions about data structure\n"
        "   - Missing null/None checks on optional fields\n"
        "\n"
        "7. **Integration Issues**:\n"
        "   - Mismatched async/sync boundaries\n"
        "   - Incorrect context usage (missing required context fields)\n"
        "   - DAG edge compatibility issues (schema mismatches)\n"
        "   - Plugin registration gaps (missing entry points)\n"
        "   - Configuration precedence violations\n"
        "\n"
        "8. **Observability Blind Spots**:\n"
        "   - Missing span creation for long operations\n"
        "   - No structured logging on critical paths\n"
        "   - Metrics not recorded for quota/rate tracking\n"
        "   - Error context lost across boundaries\n"
        "   - Missing retry attempt tracking\n"
        "\n"
        "9. **Performance/Resource Issues**:\n"
        "   - O(n²) algorithms where O(n log n) possible\n"
        "   - Redundant database queries (missing caching)\n"
        "   - Memory leaks (growing unbounded collections)\n"
        "   - Blocking I/O in async contexts\n"
        "   - Missing batch processing for bulk operations\n"
        "\n"
        "10. **Architectural Deviations**:\n"
        "    - CLAUDE.md standard violations (check against repository context)\n"
        "    - Backwards compatibility code (prohibited - see CLAUDE.md)\n"
        "    - Legacy shims or deprecated adapters\n"
        "    - Plugin ownership violations (treating plugins as untrusted)\n"
        "    - Canonical JSON violations (NaN/Infinity not rejected)\n"
        "\n"
        "Analysis Depth Checklist:\n"
        "- [ ] Read CLAUDE.md sections relevant to this file's subsystem\n"
        "- [ ] Check plugin protocol compliance (hookspec signatures)\n"
        "- [ ] Verify schema contracts match implementation\n"
        "- [ ] Trace data flow for audit trail completeness\n"
        "- [ ] Check error paths for quarantine vs crash decisions\n"
        "- [ ] Validate integration tests exist for edge cases\n"
        "- [ ] Look for untested error conditions\n"
        "- [ ] Check for missing type annotations\n"
        "\n"
        "Repository context (read-only):\n"
        f"{context}\n\n"
        "Bug report template:\n"
        f"{template}\n"
    )


def _extract_file_references(text: str) -> set[str]:
    """Extract all file paths referenced in the text."""
    patterns = [
        r"\b([\w./-]+/[\w./-]+\.py):\d+",  # path/to/file.py:123
        r"\b(src/[\w./-]+\.py)\b",  # src/path/file.py
        r"`([\w./-]+\.py)`",  # `file.py`
    ]
    files = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            files.add(match.group(1))
    return files


def _calculate_bug_similarity(report1: str, report2: str) -> float:
    """Calculate similarity score between two bug reports (0.0 to 1.0)."""
    # Extract key sections
    summary1 = extract_section(report1, "Summary").lower()
    summary2 = extract_section(report2, "Summary").lower()
    evidence1 = extract_section(report1, "Evidence").lower()
    evidence2 = extract_section(report2, "Evidence").lower()

    # Extract file references
    files1 = _extract_file_references(report1)
    files2 = _extract_file_references(report2)

    # Calculate file overlap
    if files1 and files2:
        file_overlap = len(files1 & files2) / len(files1 | files2)
    else:
        file_overlap = 0.0

    # Calculate text similarity (simple word overlap)
    def word_similarity(text1: str, text2: str) -> float:
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    summary_sim = word_similarity(summary1, summary2)
    evidence_sim = word_similarity(evidence1, evidence2)

    # Weighted average: files matter most, then summary, then evidence
    return 0.5 * file_overlap + 0.3 * summary_sim + 0.2 * evidence_sim


def _find_similar_bug(report: str, bugs_dir: Path, threshold: float = 0.6) -> Path | None:
    """Search for similar bug in docs/bugs/open/. Returns path if found."""
    if not bugs_dir.exists():
        return None

    # Extract target file from the report to narrow search
    target_files = _extract_file_references(report)
    if not target_files:
        return None

    best_match: tuple[Path, float] | None = None

    # Search all bug files in open/
    for bug_file in bugs_dir.rglob("*.md"):
        # Skip README files
        if bug_file.name == "README.md":
            continue

        existing_text = bug_file.read_text(encoding="utf-8")
        similarity = _calculate_bug_similarity(report, existing_text)

        if similarity >= threshold and (best_match is None or similarity > best_match[1]):
            best_match = (bug_file, similarity)

    return best_match[0] if best_match else None


def _merge_bug_reports(existing_path: Path, new_report: str, repo_root: Path) -> str:
    """Merge new analysis into existing bug report. Returns log message."""
    existing_text = existing_path.read_text(encoding="utf-8")

    # Extract sections from new report
    new_evidence = extract_section(new_report, "Evidence")
    new_root_cause = extract_section(new_report, "Root Cause Hypothesis")

    # Add a verification section with the new analysis
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    verification_section = (
        f"\n\n---\n\n## Re-verification ({timestamp})\n\n"
        f"**Status: RE-ANALYZED**\n\n"
        f"### New Analysis\n\n"
        f"Re-ran static analysis on {timestamp}. Key findings:\n\n"
        f"**Evidence:**\n{new_evidence}\n\n"
        f"**Root Cause:**\n{new_root_cause}\n"
    )

    # Append to existing bug
    updated_text = existing_text.rstrip() + verification_section
    existing_path.write_text(updated_text, encoding="utf-8")

    rel_path = existing_path.relative_to(repo_root) if repo_root else existing_path
    return f"Updated existing bug: {rel_path}"


def _deduplicate_and_merge(
    output_path: Path,
    bugs_open_dir: Path,
    repo_root: Path,
    similarity_threshold: float = 0.6,
) -> int:
    """
    Check generated reports against docs/bugs/open/ and merge duplicates.
    Returns count of bugs merged into existing reports.
    """
    if not output_path.exists():
        return 0

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

    merged_count = 0
    kept_reports = []

    for report in reports:
        if not report.strip():
            continue

        # Check if this is a "no bug found" report
        summary = extract_section(report, "Summary")
        if "no concrete bug found" in summary.lower():
            continue  # Don't keep these

        # Look for similar existing bug
        similar_bug = _find_similar_bug(report, bugs_open_dir, similarity_threshold)

        if similar_bug:
            # Merge into existing bug
            _merge_bug_reports(similar_bug, report, repo_root)
            merged_count += 1
        else:
            # Keep as new bug
            kept_reports.append(report)

    # Rewrite output with only new (non-duplicate) bugs
    if kept_reports:
        new_text = "\n---\n".join(kept_reports).rstrip() + "\n"
        output_path.write_text(new_text, encoding="utf-8")
    elif output_path.exists():
        # All bugs were duplicates or "no bug found" - remove the file
        output_path.unlink()

    return merged_count


def _organize_by_priority(output_dir: Path) -> None:
    """Organize outputs into by-priority/ subdirectories."""
    by_priority_dir = output_dir / "by-priority"

    for md_file in output_dir.rglob("*.md"):
        # Skip files already in by-priority
        if "by-priority" in md_file.parts:
            continue

        text = md_file.read_text(encoding="utf-8")
        priority = priority_from_report(text)

        # Copy to priority subdirectory
        dest_dir = by_priority_dir / priority
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / md_file.name
        shutil.copy2(md_file, dest_path)


async def _run_batches(
    *,
    files: list[Path],
    output_dir: Path,
    model: str | None,
    prompt_template: str,
    repo_root: Path,
    skip_existing: bool,
    batch_size: int,
    root_dir: Path,
    log_path: Path,
    context: str,
    rate_limit: int | None,
    organize_by_priority: bool,
    bugs_open_dir: Path | None,
    deduplicate: bool,
) -> dict[str, int]:
    """Run analysis in batches. Returns statistics."""
    import time as time_module

    log_lock = asyncio.Lock()
    failed_files: list[tuple[Path, Exception]] = []
    total_merged = 0
    total_gated = 0
    completed_count = 0

    # Use pyrate-limiter for rate limiting
    rate_limiter = Limiter(Rate(rate_limit, Duration.MINUTE)) if rate_limit else None

    # Print header
    print(f"\n{'─' * 60}", file=sys.stderr)
    print(f"🔍 Codex Bug Hunt: {len(files)} files to analyze", file=sys.stderr)
    print(f"{'─' * 60}\n", file=sys.stderr)

    for batch in chunked(files, batch_size):
        tasks: list[asyncio.Task[dict[str, int]]] = []
        batch_files: list[Path] = []
        task_to_file: dict[asyncio.Task[dict[str, int]], tuple[Path, float]] = {}

        for file_path in batch:
            relative = file_path.relative_to(root_dir)
            output_path = output_dir / relative
            output_path = output_path.with_suffix(output_path.suffix + ".md")

            if skip_existing and output_path.exists():
                completed_count += 1
                rel_display = file_path.relative_to(repo_root)
                print(f"  ⏭️  [{completed_count}/{len(files)}] {rel_display} (cached)", file=sys.stderr)
                continue

            prompt = _build_prompt(file_path, prompt_template, context)
            batch_files.append(file_path)

            # Print start message
            rel_display = file_path.relative_to(repo_root)
            print(f"  🔄 [{completed_count + len(batch_files)}/{len(files)}] {rel_display} ...", file=sys.stderr)

            task = asyncio.create_task(
                run_codex_with_retry_and_logging(
                    file_path=file_path,
                    output_path=output_path,
                    model=model,
                    prompt=prompt,
                    repo_root=repo_root,
                    log_path=log_path,
                    log_lock=log_lock,
                    file_display=str(file_path.relative_to(repo_root).as_posix()),
                    output_display=str(output_path.relative_to(repo_root).as_posix()),
                    rate_limiter=rate_limiter,
                    evidence_gate_summary_prefix="",
                )
            )
            tasks.append(task)
            task_to_file[task] = (file_path, time_module.monotonic())

        # Process results as they complete (not waiting for all)
        if tasks:
            for completed_task in asyncio.as_completed(tasks):
                file_path, start_time = task_to_file[completed_task]
                rel_display = file_path.relative_to(repo_root)
                duration = time_module.monotonic() - start_time

                try:
                    result = await completed_task
                    completed_count += 1
                    total_gated += result["gated"]

                    # Status indicator based on evidence gate
                    status = "✅" if result["gated"] == 0 else f"⚠️  (gated:{result['gated']})"
                    print(f"  {status} [{completed_count}/{len(files)}] {rel_display} ({duration:.1f}s)", file=sys.stderr)

                    # Deduplicate after successful run
                    if deduplicate and bugs_open_dir and bugs_open_dir.exists():
                        relative = file_path.relative_to(root_dir)
                        output_path = output_dir / relative
                        output_path = output_path.with_suffix(output_path.suffix + ".md")
                        merged_count = _deduplicate_and_merge(output_path, bugs_open_dir, repo_root)
                        total_merged += merged_count
                        if merged_count > 0:
                            print(f"      🔗 Merged {merged_count} duplicate(s)", file=sys.stderr)

                except Exception as exc:
                    completed_count += 1
                    failed_files.append((file_path, exc))
                    print(f"  ❌ [{completed_count}/{len(files)}] {rel_display} ({duration:.1f}s) - {str(exc)[:50]}", file=sys.stderr)

    print(f"\n{'─' * 60}", file=sys.stderr)

    # Report failures
    if failed_files:
        print(f"\n⚠️  {len(failed_files)} files failed:", file=sys.stderr)
        for path, exc in failed_files[:10]:
            print(f"  {path.relative_to(repo_root)}: {exc}", file=sys.stderr)
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more (see {log_path})", file=sys.stderr)

    # Report deduplication stats
    if deduplicate and total_merged > 0:
        print(f"\n🔗 {total_merged} bugs merged into existing reports in docs/bugs/open/")

    # Organize outputs by priority if requested
    if organize_by_priority:
        _organize_by_priority(output_dir)

    # Generate summary statistics
    summary: dict[str, int] = generate_summary(output_dir, no_defect_marker="No concrete bug found")
    summary["merged"] = total_merged
    summary["gated"] = total_gated
    return summary


def _paths_from_file(path_file: Path, repo_root: Path, root_dir: Path) -> list[Path]:
    selected: list[Path] = []
    lines = path_file.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        raw_path = Path(stripped)
        path = raw_path if raw_path.is_absolute() else (repo_root / raw_path).resolve()
        if not path.exists():
            raise RuntimeError(f"paths-from entry does not exist: {raw_path}")
        if path.is_dir():
            selected.extend([p for p in path.rglob("*") if p.is_file() and not is_cache_path(p)])
        else:
            if not is_cache_path(path):
                selected.append(path)
    return [path for path in selected if _is_under_root(path, root_dir)]


def _is_under_root(path: Path, root_dir: Path) -> bool:
    try:
        path.relative_to(root_dir)
        return True
    except ValueError:
        return False


def _changed_files_since(repo_root: Path, root_dir: Path, git_ref: str) -> list[Path]:
    try:
        root_rel = root_dir.relative_to(repo_root)
    except ValueError:
        root_rel = root_dir
    cmd = ["git", "diff", "--name-only", git_ref, "--", str(root_rel)]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    selected = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        path = (repo_root / rel).resolve()
        if path.is_file() and _is_under_root(path, root_dir) and not is_cache_path(path):
            selected.append(path)
    return selected


def _changed_files_on_branch(repo_root: Path, root_dir: Path, base_branch: str) -> list[Path]:
    """Get files changed on current branch vs base branch using merge-base."""
    # Find merge base
    merge_base_cmd = ["git", "merge-base", base_branch, "HEAD"]
    result = subprocess.run(merge_base_cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git merge-base failed for {base_branch}")
    merge_base = result.stdout.strip()

    # Get diff from merge base to HEAD
    try:
        root_rel = root_dir.relative_to(repo_root)
    except ValueError:
        root_rel = root_dir
    cmd = ["git", "diff", "--name-only", f"{merge_base}..HEAD", "--", str(root_rel)]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")

    selected = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        path = (repo_root / rel).resolve()
        if path.is_file() and _is_under_root(path, root_dir) and not is_cache_path(path):
            selected.append(path)
    return selected


def _changed_files_in_range(repo_root: Path, root_dir: Path, commit_range: str) -> list[Path]:
    """Get files changed in commit range (e.g., 'abc123..def456')."""
    # Validate range format
    if ".." not in commit_range:
        raise ValueError(f"Invalid commit range format: {commit_range}. Expected format: 'START..END'")

    # Get diff for the range
    try:
        root_rel = root_dir.relative_to(repo_root)
    except ValueError:
        root_rel = root_dir
    cmd = ["git", "diff", "--name-only", commit_range, "--", str(root_rel)]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git diff failed for range {commit_range}")

    selected = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        path = (repo_root / rel).resolve()
        if path.is_file() and _is_under_root(path, root_dir) and not is_cache_path(path):
            selected.append(path)
    return selected


def _list_files(
    *,
    root_dir: Path,
    repo_root: Path,
    changed_since: str | None,
    branch: str | None,
    commit_range: str | None,
    paths_from: Path | None,
    file_type: str,
) -> list[Path]:
    # Check mutual exclusivity of git filters
    git_filters = [changed_since, branch, commit_range]
    active_filters = [f for f in git_filters if f is not None]
    if len(active_filters) > 1:
        raise ValueError("Only one of --changed-since, --branch, or --commit-range can be used at a time")

    selected: set[Path] | None = None

    if changed_since:
        changed = set(_changed_files_since(repo_root, root_dir, changed_since))
        selected = changed if selected is None else selected & changed

    if branch:
        changed = set(_changed_files_on_branch(repo_root, root_dir, branch))
        selected = changed if selected is None else selected & changed

    if commit_range:
        changed = set(_changed_files_in_range(repo_root, root_dir, commit_range))
        selected = changed if selected is None else selected & changed

    if paths_from:
        listed = set(_paths_from_file(paths_from, repo_root, root_dir))
        selected = listed if selected is None else selected & listed

    if selected is None:
        selected = {path for path in root_dir.rglob("*") if path.is_file() and not is_cache_path(path)}

    # Apply file type filter
    if file_type == "python":
        selected = {p for p in selected if _is_python_file(p)}

    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Codex bug audits per file in batches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all Python files in src/elspeth
  %(prog)s

  # Scan only changed files since HEAD~5
  %(prog)s --changed-since HEAD~5

  # Scan files changed on current branch vs main
  %(prog)s --branch main

  # Scan files changed in a specific commit range
  %(prog)s --commit-range abc123..def456

  # Dry run to see what would be scanned
  %(prog)s --dry-run

  # Use rate limiting for API quota management
  %(prog)s --rate-limit 30

  # Organize outputs by priority
  %(prog)s --organize-by-priority
        """,
    )
    parser.add_argument(
        "--root",
        default="src/elspeth",
        help="Root directory to scan for files (default: src/elspeth).",
    )
    parser.add_argument(
        "--template",
        default="docs/bugs/BUGS.md",
        help="Bug report template path (default: docs/bugs/BUGS.md).",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/bugs/generated",
        help="Directory to write bug reports (default: docs/bugs/generated).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Maximum concurrent Codex runs per batch (default: 10).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override Codex model (passes --model to codex exec).",
    )
    parser.add_argument(
        "--changed-since",
        default=None,
        help="Only scan files changed since this git ref (e.g. HEAD~1).",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Compare against base branch to get files changed on current branch (e.g. 'main').",
    )
    parser.add_argument(
        "--commit-range",
        default=None,
        help="Only scan files changed in commit range (e.g. 'abc123..def456').",
    )
    parser.add_argument(
        "--paths-from",
        default=None,
        help="Path to a file containing newline-separated paths to scan.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have an output report.",
    )
    parser.add_argument(
        "--file-type",
        default="python",
        choices=["python", "all"],
        help="Filter by file type (default: python, excludes tests).",
    )
    parser.add_argument(
        "--context-files",
        nargs="+",
        default=None,
        help="Additional context files beyond CLAUDE.md/ARCHITECTURE.md.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        help="Max requests per minute (e.g., 30 for API quota management).",
    )
    parser.add_argument(
        "--organize-by-priority",
        action="store_true",
        help="Organize outputs into by-priority/ subdirectories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be scanned without running analysis.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Check generated bugs against docs/bugs/open/ and merge duplicates.",
    )
    parser.add_argument(
        "--bugs-dir",
        default="docs/bugs/open",
        help="Directory to search for existing bugs (default: docs/bugs/open).",
    )

    args = parser.parse_args()

    # Validation
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.rate_limit is not None and args.rate_limit < 1:
        raise ValueError("--rate-limit must be >= 1")

    if shutil.which("codex") is None:
        raise RuntimeError("codex CLI not found on PATH")

    repo_root = Path(__file__).resolve().parents[1]
    root_dir = resolve_path(repo_root, args.root)
    template_path = resolve_path(repo_root, args.template)
    output_dir = resolve_path(repo_root, args.output_dir)
    log_path = resolve_path(repo_root, "docs/bugs/process/CODEX_LOG.md")
    bugs_open_dir = resolve_path(repo_root, args.bugs_dir) if args.deduplicate else None

    # List files to scan
    paths_from = resolve_path(repo_root, args.paths_from) if args.paths_from else None
    files = _list_files(
        root_dir=root_dir,
        repo_root=repo_root,
        changed_since=args.changed_since,
        branch=args.branch,
        commit_range=args.commit_range,
        paths_from=paths_from,
        file_type=args.file_type,
    )

    if not files:
        print(f"No files found under {root_dir}", file=sys.stderr)
        return 1

    # Dry run mode
    if args.dry_run:
        print(f"Would analyze {len(files)} files:")
        for f in files[:20]:
            print(f"  {f.relative_to(repo_root)}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return 0

    # Load template and context
    template_text = template_path.read_text(encoding="utf-8")
    context_text = load_context(repo_root, extra_files=args.context_files)
    ensure_log_file(log_path, header_title="Codex Bug Hunt Log")

    # Run analysis
    stats = asyncio.run(
        _run_batches(
            files=files,
            output_dir=output_dir,
            model=args.model,
            prompt_template=template_text,
            repo_root=repo_root,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
            root_dir=root_dir,
            log_path=log_path,
            context=context_text,
            rate_limit=args.rate_limit,
            organize_by_priority=args.organize_by_priority,
            bugs_open_dir=bugs_open_dir,
            deduplicate=args.deduplicate,
        )
    )

    # Print summary
    print_summary(stats, icon="🐛", title="Bug Hunt Summary")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
