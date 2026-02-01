#!/usr/bin/env python3
"""Test quality defect scanner using Codex.

Scans test files for systematic quality issues like:
- Sleepy assertions (time.sleep in tests)
- Missing audit trail verification
- Fixture duplication
- Weak/incomplete assertions
- Missing property-based tests
- Misclassified tests
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

from codex_audit_common import (  # type: ignore[import-not-found]
    AsyncTqdm,
    chunked,
    ensure_log_file,
    generate_summary,
    get_git_commit,
    is_cache_path,
    load_context,
    print_summary,
    resolve_path,
    run_codex_with_retry_and_logging,
    utc_now,
    write_findings_index,
    write_run_metadata,
    write_summary_file,
)
from pyrate_limiter import Duration, Limiter, Rate


def _is_test_file(path: Path) -> bool:
    """Check if path is a test file."""
    return path.suffix == ".py" and path.name.startswith("test_")


def _build_test_quality_prompt(file_path: Path, template: str, context: str) -> str:
    return (
        "You are a test quality auditor analyzing test files for systematic defects.\n"
        f"Target file: {file_path}\n\n"
        "Instructions:\n"
        "- Use the test defect template below verbatim.\n"
        "- Fill in every section. If unknown, write 'Unknown'.\n"
        "- You MUST read the target test file completely before analyzing.\n"
        "- You SHOULD read the code under test to understand what's being tested.\n"
        "- You MAY read any other repo file to understand test context.\n"
        "- Report defects only if the primary fix belongs in the target test file.\n"
        "- If you find multiple distinct defects, output one full template per defect,\n"
        "  separated by a line with only '---'.\n"
        "- If you find no credible defect, output one template with Summary set to\n"
        f"  'No test defect found in {file_path}', Severity 'trivial', Priority 'P3',\n"
        "  and Root Cause Hypothesis 'No defect identified'.\n"
        "- Evidence MUST cite file paths and line numbers (e.g., 'tests/core/test_dag.py:123').\n"
        "- Evidence SHOULD include code snippets showing the actual defect.\n"
        "- Vague claims without evidence will be automatically downgraded to P3.\n\n"
        "Severity Guidelines:\n"
        "- P0 (Critical): Crashes, data corruption, security holes, missing audit trail recording\n"
        "- P1 (High): Flaky tests, missing edge cases for core logic, weak assertions on critical paths\n"
        "- P2 (Medium): Code quality issues, missing tests for error paths, fixture duplication\n"
        "- P3 (Low): Documentation, minor improvements, nice-to-have tests\n\n"
        "Test Quality Anti-Patterns to Check:\n"
        "\n"
        "1. **Sleepy Assertions**:\n"
        "   - Tests using time.sleep() for timing/synchronization\n"
        "   - Should mock time.monotonic() or use condition-based waits\n"
        "   - Causes flaky tests in slow CI environments\n"
        "   - Example: test waits 0.5s for aggregation timeout instead of mocking clock\n"
        "\n"
        "2. **Missing Audit Trail Verification** (CRITICAL for ELSPETH):\n"
        "   - Tests verify business logic but not Landscape recording\n"
        "   - SPECIFIC CHECKS REQUIRED:\n"
        "     a) node_states table: verify state, input_hash, output_hash, error fields\n"
        "     b) token_outcomes table: verify terminal_state matches expected (COMPLETED/ROUTED/QUARANTINED/etc.)\n"
        "     c) artifacts table: verify content_hash, payload_id, artifact_type\n"
        "     d) Hash integrity: verify hashes are deterministic and match canonical JSON\n"
        "     e) Lineage: verify parent_token_id, row_id relationships for forks/joins\n"
        "   - Missing verification that ALL operations are recorded (not just success paths)\n"
        "   - Missing verification of error recording in node_states.error field\n"
        "   - CLI/integration tests that don't query the audit database\n"
        "   - Tests that mock Landscape instead of verifying actual database writes\n"
        "\n"
        "3. **Fixture Duplication**:\n"
        "   - Same setup code repeated across 10+ test methods\n"
        "   - Inline class definitions (ListSource, CollectSink) duplicated\n"
        "   - Should use pytest fixtures or conftest.py\n"
        "   - 100+ lines of identical boilerplate per file\n"
        "\n"
        "4. **Weak/Incomplete Assertions**:\n"
        "   - Checking field exists without validating value (assert 'x' in obj)\n"
        "   - Vacuous assertions (assert len(rows) >= 0)\n"
        "   - Only checking success, not correctness\n"
        "   - Assertion-free tests (no assert statements)\n"
        "   - Substring matching on error messages instead of structured checks\n"
        "\n"
        "5. **Missing Tier 1 Corruption Tests** (CLAUDE.md: Three-Tier Trust Model):\n"
        "   - No tests verify crash on NULL in required fields\n"
        "   - No tests verify crash on invalid enum values\n"
        "   - No tests verify crash on wrong types in audit data\n"
        "   - No tests for checkpoint/Landscape data corruption\n"
        "   - Tier 1 'our data' must crash on anomalies, not coerce\n"
        "\n"
        "6. **Missing Property-Based Tests**:\n"
        "   - Hypothesis is in tech stack but unused\n"
        "   - Determinism claims without property tests (hash stability, ordering)\n"
        "   - Invariants tested with hand-written examples only\n"
        "   - Edge cases not systematically explored\n"
        "\n"
        "7. **Misclassified Tests**:\n"
        "   - Integration tests in unit test files (uses real DB, filesystem, plugins)\n"
        "   - Unit tests in integration test directories\n"
        "   - Contract tests that are really smoke tests\n"
        "   - Fuzz/property tests in regular test files (should be separate suite)\n"
        "\n"
        "8. **Bug-Hiding Defensive Patterns** (CLAUDE.md prohibition):\n"
        "   - Tests using hasattr(), isinstance(), .get() on system code\n"
        "   - Tests validating defensive patterns instead of letting bugs crash\n"
        "   - Protocol compliance checked at runtime instead of statically\n"
        "   - Example: if hasattr(plugin, 'close'): plugin.close() # WRONG\n"
        "\n"
        "9. **Missing Thread Safety Tests**:\n"
        "   - Code has locks/threading but no concurrency tests\n"
        "   - call_index, cache mutations, shared state untested under concurrent access\n"
        "   - Race conditions in audit trail recording\n"
        "\n"
        "10. **Infrastructure Gaps**:\n"
        "    - No shared fixtures for common setup\n"
        "    - No test data builders (magic numbers everywhere)\n"
        "    - No integration with actual database constraints\n"
        "    - Missing cleanup verification\n"
        "    - Tests accessing private attributes (_field) instead of public API\n"
        "\n"
        "11. **Missing Edge Cases**:\n"
        "    - No tests for NaN/Infinity rejection (CLAUDE.md requirement)\n"
        "    - No tests for empty inputs, None values, zero counts\n"
        "    - No tests for very large payloads\n"
        "    - No tests for Unicode/encoding issues\n"
        "    - No tests for error paths (exceptions, failures)\n"
        "\n"
        "12. **Incomplete Contract Coverage**:\n"
        "    - Plugin protocol methods untested\n"
        "    - Schema validation gaps\n"
        "    - Lifecycle hooks (on_start, on_complete, close) untested\n"
        "    - Error routing untested\n"
        "    - Batch-aware processing untested\n"
        "\n"
        "13. **Missing Recovery/Checkpoint Testing** (CRITICAL for ELSPETH):\n"
        "    - No tests verify checkpoint creation at correct intervals\n"
        "    - No tests verify recovery from checkpoint restores exact state\n"
        "    - No tests verify aggregation timeout age is preserved on resume\n"
        "    - No tests verify idempotency (running same data twice gives same result)\n"
        "    - No tests for checkpoint corruption detection\n"
        "    - No tests verify run_id, attempt counters after recovery\n"
        "    - Missing tests for partial batch recovery\n"
        "\n"
        "14. **Missing Negative Tests** (What Happens When Things Go Wrong?):\n"
        "    - No tests for database connection failures\n"
        "    - No tests for plugin exceptions during processing\n"
        "    - No tests for invalid configuration\n"
        "    - No tests for schema violations (wrong types, missing fields)\n"
        "    - No tests for external API failures (timeout, 500 errors, malformed responses)\n"
        "    - No tests for filesystem errors (disk full, permission denied)\n"
        "    - Missing quarantine verification (how are bad rows handled?)\n"
        "\n"
        "15. **Test Isolation Issues**:\n"
        "    - Tests depend on execution order (test_001, test_002 pattern)\n"
        "    - Tests share mutable state between test methods\n"
        "    - Tests leave database records that affect other tests\n"
        "    - Tests don't clean up created files/directories\n"
        "    - Tests use global singletons or class variables\n"
        "    - Integration tests don't use isolated database schemas/tables\n"
        "\n"
        "Analysis Depth Checklist:\n"
        "- [ ] Read CLAUDE.md sections on Auditability Standard and Three-Tier Trust Model\n"
        "- [ ] Read the target test file COMPLETELY before analyzing\n"
        "- [ ] Read the code under test to understand what SHOULD be tested\n"
        "- [ ] Check for sleepy assertions (time.sleep, asyncio.sleep in non-timeout tests)\n"
        "- [ ] Verify audit trail recording is tested with SPECIFIC checks (see anti-pattern #2)\n"
        "- [ ] Verify recovery/checkpoint testing exists (see anti-pattern #13)\n"
        "- [ ] Look for repeated setup code (fixture duplication)\n"
        "- [ ] Check assertion strength (presence vs correctness, vacuous assertions)\n"
        "- [ ] Verify Tier 1 corruption detection tests exist (crash on bad audit data)\n"
        "- [ ] Check if Hypothesis is used for property testing (it's in the tech stack)\n"
        "- [ ] Validate test classification (unit/integration/contract in correct directories)\n"
        "- [ ] Look for bug-hiding defensive patterns (hasattr, isinstance on system code)\n"
        "- [ ] Check for missing concurrency tests (locks, threading, race conditions)\n"
        "- [ ] Check for missing negative tests (what happens when things fail?)\n"
        "- [ ] Check for test isolation issues (execution order, shared state)\n"
        "- [ ] Provide SPECIFIC line numbers and code snippets in Evidence section\n"
        "\n"
        "Repository context (read-only):\n"
        f"{context}\n\n"
        "Test defect template:\n"
        f"{template}\n"
    )


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
) -> dict[str, int]:
    """Run analysis in batches. Returns statistics."""
    log_lock = asyncio.Lock()
    failed_files: list[tuple[Path, Exception]] = []
    total_gated = 0

    # Use pyrate-limiter for rate limiting
    rate_limiter = Limiter(Rate(rate_limit, Duration.MINUTE)) if rate_limit else None
    pbar = AsyncTqdm(total=len(files), desc="Analyzing test files", unit="file")

    for batch in chunked(files, batch_size):
        tasks: list[asyncio.Task[dict[str, int]]] = []
        batch_files: list[Path] = []

        for file_path in batch:
            relative = file_path.relative_to(root_dir)
            output_path = output_dir / relative
            output_path = output_path.with_suffix(output_path.suffix + ".md")

            if skip_existing and output_path.exists():
                pbar.update(1)
                continue

            prompt = _build_test_quality_prompt(file_path, prompt_template, context)
            batch_files.append(file_path)
            tasks.append(
                asyncio.create_task(
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
            )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for file_path, result in zip(batch_files, results, strict=True):
                if isinstance(result, Exception):
                    failed_files.append((file_path, result))
                elif isinstance(result, dict):
                    total_gated += result["gated"]  # Fixed: removed .get() - our data (Tier 2)
                pbar.update(1)

    pbar.close()

    if failed_files:
        print(f"\n⚠️  {len(failed_files)} files failed:", file=sys.stderr)
        for path, exc in failed_files[:10]:
            print(f"  {path.relative_to(repo_root)}: {exc}", file=sys.stderr)
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more (see {log_path})", file=sys.stderr)

    summary: dict[str, int] = generate_summary(output_dir, no_defect_marker="No test defect found")
    summary["gated"] = total_gated
    return summary


def _list_test_files(
    *,
    root_dir: Path,
) -> list[Path]:
    """List all test files under root_dir."""
    selected = {path for path in root_dir.rglob("test_*.py") if path.is_file() and not is_cache_path(path)}
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Codex test quality audits on test files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all test files
  %(prog)s

  # Scan specific test directory
  %(prog)s --root tests/core

  # Dry run to see what would be scanned
  %(prog)s --dry-run

  # Use rate limiting
  %(prog)s --rate-limit 30
        """,
    )
    parser.add_argument(
        "--root",
        default="tests",
        help="Root directory to scan for test files (default: tests).",
    )
    parser.add_argument(
        "--template",
        default="docs/quality-audit/TEST_DEFECT_TEMPLATE.md",
        help="Test defect template path.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/quality-audit/findings-codex",
        help="Directory to write defect reports.",
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
        "--skip-existing",
        action="store_true",
        help="Skip files that already have an output report.",
    )
    parser.add_argument(
        "--context-files",
        nargs="+",
        default=None,
        help="Additional context files beyond CLAUDE.md.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        help="Max requests per minute (e.g., 30 for API quota management).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be scanned without running analysis.",
    )

    args = parser.parse_args()

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
    log_path = resolve_path(repo_root, "docs/quality-audit/TEST_DEFECT_LOG.md")

    # List test files
    files = _list_test_files(root_dir=root_dir)

    if not files:
        print(f"No test files found under {root_dir}", file=sys.stderr)
        return 1

    # Dry run mode
    if args.dry_run:
        print(f"Would analyze {len(files)} test files:")
        for f in files[:20]:
            print(f"  {f.relative_to(repo_root)}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return 0

    # Get git commit for metadata
    git_commit = get_git_commit(repo_root)

    # Load template and context
    template_text = template_path.read_text(encoding="utf-8")
    context_text = load_context(repo_root, extra_files=args.context_files)
    ensure_log_file(log_path, header_title="Codex Test Defect Hunt Log")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track execution time
    start_time = utc_now()
    start_monotonic = time.monotonic()

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
        )
    )

    # Track completion time
    end_time = utc_now()
    duration_s = time.monotonic() - start_monotonic

    # Write structured output files for triage
    write_run_metadata(
        output_dir=output_dir,
        repo_root=repo_root,
        start_time=start_time,
        end_time=end_time,
        duration_s=duration_s,
        files_scanned=len(files),
        model=args.model,
        batch_size=args.batch_size,
        rate_limit=args.rate_limit,
        git_commit=git_commit,
        title="Test Quality Audit Run Metadata",
        script_name="scripts/codex_test_defect_hunt.py",
    )

    write_summary_file(
        output_dir=output_dir,
        stats=stats,
        total_files=len(files),
        title="Test Quality Audit Summary",
        defects_label="Test Defects Found",
        clean_label="Clean Test Files",
    )

    write_findings_index(
        output_dir=output_dir,
        repo_root=repo_root,
        title="Test Quality Findings Index",
        file_column_label="Test File",
        no_defect_marker="No test defect found",
        clean_section_title="Clean Test Files",
    )

    # Print summary to stdout for immediate feedback
    print_summary(stats, icon="📊", title="Test Quality Audit Summary")

    # Tell user where to find the results
    print(f"📁 Detailed results written to: {output_dir.relative_to(repo_root)}/")
    print("   - RUN_METADATA.md: Run details and parameters")
    print("   - SUMMARY.md: Triage dashboard and statistics")
    print("   - FINDINGS_INDEX.md: Sortable table of all findings")
    print("   - TEST_DEFECT_LOG.md: Execution log")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
