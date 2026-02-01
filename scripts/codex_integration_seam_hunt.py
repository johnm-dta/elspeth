#!/usr/bin/env python3
"""Integration seam and architecture health analyzer using Codex.

Scans source code for architectural anti-patterns at integration boundaries:
- Parallel type evolution (duplicate definitions)
- Impedance mismatch (complex translations)
- Leaky abstractions
- Contract violations
- Shared mutable state
- God objects
- Stringly-typed interfaces
- Missing facades
- Protocol drift
- Callback hell
- Missing error translation
- Implicit state dependencies
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


def _is_source_file(path: Path) -> bool:
    """Check if path is a Python source file (not test, not migration)."""
    if path.suffix != ".py":
        return False

    # Exclude test files
    if path.name.startswith("test_"):
        return False
    if "tests" in path.parts:
        return False

    # Exclude special files
    return path.name not in ["__init__.py", "conftest.py"]


def _build_integration_seam_prompt(file_path: Path, template: str, context: str) -> str:
    return (
        "You are an architecture health auditor analyzing source code for integration seam defects.\n"
        f"Target file: {file_path}\n\n"
        "Instructions:\n"
        "- Use the integration seam defect template below verbatim.\n"
        "- Fill in every section. If unknown, write 'Unknown'.\n"
        "- You MUST read the target file completely before analyzing.\n"
        "- You SHOULD read related files to understand integration boundaries.\n"
        "- You MAY read any repo file to understand architecture context.\n"
        "- Report defects only if they involve the target file as one side of a seam.\n"
        "- If you find multiple distinct seam defects, output one full template per defect,\n"
        "  separated by a line with only '---'.\n"
        "- If you find no credible defect, output one template with Summary set to\n"
        f"  'No integration seam defect found in {file_path}', Severity 'trivial', Priority 'P3',\n"
        "  and Anti-Pattern Classification 'None identified'.\n"
        "- Evidence MUST cite file paths and line numbers for BOTH sides of the seam.\n"
        "- Evidence MUST show code snippets from both sides demonstrating the defect.\n"
        "- Vague claims without evidence will be automatically downgraded to P3.\n\n"
        "Severity Guidelines:\n"
        "- P0 (Critical): Parallel evolution of core types, missing error translation exposing internals\n"
        "- P1 (High): God objects at boundaries, contract violations, leaky abstractions\n"
        "- P2 (Medium): Impedance mismatch requiring translation, stringly-typed interfaces\n"
        "- P3 (Low): Missing facade, callback chains, minor coupling\n\n"
        "Integration Anti-Patterns to Check:\n"
        "\n"
        "1. **Parallel Type Evolution** (CRITICAL):\n"
        "   - Two or more classes/types representing the same concept but defined separately\n"
        "   - Example: `core.models.RowData` vs `engine.processor.ProcessingRow`\n"
        "   - Look for: Similar class names across modules, fields with same semantics\n"
        "   - Evidence: Show both definitions side-by-side\n"
        "   - Common causes: Independent development, missing contracts module\n"
        "   - Fix: Create canonical type in `contracts/`, remove duplicates\n"
        "\n"
        "2. **Impedance Mismatch at Boundaries**:\n"
        "   - Data requires non-trivial transformation when crossing boundaries\n"
        "   - Look for: `adapt_*`, `convert_*`, `from_*`, `to_*` functions\n"
        "   - Look for: Manual field mapping between similar structures\n"
        "   - Evidence: Show translation layer code\n"
        "   - Common causes: Incompatible interfaces, DTO proliferation\n"
        "   - Fix: Align types or create proper adapter pattern\n"
        "\n"
        "3. **Leaky Abstractions** (CRITICAL):\n"
        "   - Implementation details leak across boundaries\n"
        "   - Example: Landscape exposing SQLAlchemy sessions to Orchestrator\n"
        "   - Look for: Framework-specific types in high-level code (Session, Connection, etc.)\n"
        "   - Look for: Private attributes (`_session`) accessed from outside\n"
        "   - Evidence: Show where implementation detail crosses boundary\n"
        "   - Fix: Hide implementation details behind interface\n"
        "\n"
        "4. **Contract Violations** (CRITICAL):\n"
        "   - Code assumes properties not enforced by contract\n"
        "   - Example: Assuming `result.row` is never None when contract allows it\n"
        "   - Look for: `.get()` calls with no default, unchecked attribute access\n"
        "   - Look for: Comments like '# Assume X is always Y'\n"
        "   - Evidence: Show assumption and what contract actually says\n"
        "   - Fix: Encode assumption in type system or add runtime check\n"
        "\n"
        "5. **Unnecessary Shared Mutable State**:\n"
        "   - Two systems share mutable state without clear ownership\n"
        "   - Example: Plugin context with mutable `_internal_state` dict\n"
        "   - Look for: Mutable fields in context objects, shared caches\n"
        "   - Look for: No ownership documentation (who can modify this?)\n"
        "   - Evidence: Show mutable state and multiple access points\n"
        "   - Fix: Make immutable or clarify ownership\n"
        "\n"
        "6. **God Objects Crossing Boundaries**:\n"
        "   - Large objects passed everywhere, carrying more than needed\n"
        "   - Example: Entire `PipelineState` passed to plugin that only needs `config`\n"
        "   - Look for: Context objects with >5 fields\n"
        "   - Look for: Functions using only 1-2 fields from large parameter\n"
        "   - Evidence: Show god object definition and minimal usage\n"
        "   - Fix: Pass only what's needed\n"
        "\n"
        "7. **Stringly-Typed Interfaces**:\n"
        "   - Using strings where enums/types should be used\n"
        "   - Example: `ctx.set_routing('route_to_sink', 'error_sink')`\n"
        "   - Look for: String literals as identifiers, magic strings in conditionals\n"
        "   - Look for: Comments explaining what strings are valid\n"
        "   - Evidence: Show string-based dispatch/routing\n"
        "   - Fix: Replace with enum or typed constant\n"
        "\n"
        "8. **Missing Facade at Complex Boundaries**:\n"
        "   - Complex subsystem exposed directly without simplified interface\n"
        "   - Example: Checkpoint with 5 classes, every caller orchestrates manually\n"
        "   - Look for: Multiple classes from same subsystem imported together\n"
        "   - Look for: Repeated orchestration patterns in multiple callers\n"
        "   - Evidence: Show complex usage pattern repeated\n"
        "   - Fix: Create facade with simple interface\n"
        "\n"
        "9. **Protocol Drift**:\n"
        "   - Two sides of interface evolved independently, no compatibility check\n"
        "   - Example: Engine expects `row_id` field, old plugins don't provide it\n"
        "   - Look for: Version-specific code paths, compatibility shims\n"
        "   - Look for: Comments like 'legacy plugins use X'\n"
        "   - Evidence: Show incompatible expectations\n"
        "   - Fix: Add versioning and compatibility checks\n"
        "\n"
        "10. **Callback Hell at Boundaries**:\n"
        "    - Complex callback chains between systems (3+ levels)\n"
        "    - Example: Engine → Plugin → Landscape → Plugin callback\n"
        "    - Look for: Nested callback definitions, `on_*` parameter explosion\n"
        "    - Evidence: Show callback chain structure\n"
        "    - Fix: Use async/await or synchronous calls\n"
        "\n"
        "11. **Missing Error Translation** (CRITICAL):\n"
        "    - Low-level errors leak to high-level callers\n"
        "    - Example: Landscape raises `IntegrityError` (SQLAlchemy) to Orchestrator\n"
        "    - Look for: Framework exceptions in high-level try/except\n"
        "    - Look for: No exception wrapping at boundary\n"
        "    - Evidence: Show low-level exception crossing boundary\n"
        "    - Fix: Translate to domain exceptions at boundary\n"
        "\n"
        "12. **Implicit State Dependencies**:\n"
        "    - Method call order matters but isn't enforced\n"
        "    - Example: Must call `initialize()` before `run()` but no check\n"
        "    - Look for: Comments like 'call X before Y', sequential setup methods\n"
        "    - Look for: Attributes checked without initialization check\n"
        "    - Evidence: Show usage assumptions\n"
        "    - Fix: State machine pattern or builder pattern\n"
        "\n"
        "Analysis Strategy:\n"
        "- [ ] Read CLAUDE.md to understand subsystem boundaries\n"
        "- [ ] Identify what subsystem the target file belongs to\n"
        "- [ ] Map imports to understand integration points\n"
        "- [ ] Search for similar type names across boundaries (Levenshtein distance < 3)\n"
        "- [ ] Look for translation functions (adapt_, convert_, from_, to_)\n"
        "- [ ] Check context objects for god object pattern (>5 fields)\n"
        "- [ ] Examine exception handling for error translation\n"
        "- [ ] Verify contracts match usage (type annotations vs actual usage)\n"
        "- [ ] Provide SPECIFIC line numbers and code snippets from BOTH sides\n"
        "\n"
        "Focus Areas by Subsystem:\n"
        "- **core/ ↔ engine/**: Check for parallel type evolution, error translation\n"
        "- **engine/ ↔ plugins/**: Check for leaky abstractions, god objects, contract violations\n"
        "- **plugins/ ↔ contracts/**: Check for protocol compliance, stringly-typed interfaces\n"
        "- **Any ↔ landscape/**: Check for SQLAlchemy leaks, session management\n"
        "\n"
        "Repository context (read-only):\n"
        f"{context}\n\n"
        "Integration seam defect template:\n"
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
    pbar = AsyncTqdm(total=len(files), desc="Analyzing source files", unit="file")

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

            prompt = _build_integration_seam_prompt(file_path, prompt_template, context)
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
                        evidence_gate_summary_prefix="for both sides of seam",
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

    summary: dict[str, int] = generate_summary(output_dir, no_defect_marker="No integration seam defect found")
    summary["gated"] = total_gated
    return summary


def _list_source_files(
    *,
    root_dir: Path,
) -> list[Path]:
    """List all source files under root_dir."""
    selected = {path for path in root_dir.rglob("*.py") if path.is_file() and _is_source_file(path) and not is_cache_path(path)}
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Codex integration seam and architecture health analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all source files
  %(prog)s

  # Scan specific subsystem
  %(prog)s --root src/elspeth/engine

  # Dry run to see what would be scanned
  %(prog)s --dry-run

  # Use rate limiting
  %(prog)s --rate-limit 30
        """,
    )
    parser.add_argument(
        "--root",
        default="src/elspeth",
        help="Root directory to scan for source files (default: src/elspeth).",
    )
    parser.add_argument(
        "--template",
        default="docs/quality-audit/INTEGRATION_SEAM_DEFECT_TEMPLATE.md",
        help="Integration seam defect template path.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/quality-audit/findings-integration",
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
    log_path = resolve_path(repo_root, "docs/quality-audit/INTEGRATION_SEAM_LOG.md")

    # List source files
    files = _list_source_files(root_dir=root_dir)

    if not files:
        print(f"No source files found under {root_dir}", file=sys.stderr)
        return 1

    # Dry run mode
    if args.dry_run:
        print(f"Would analyze {len(files)} source files:")
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
    ensure_log_file(log_path, header_title="Codex Integration Seam Hunt Log")

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
        title="Integration Seam Analysis Run Metadata",
        script_name="scripts/codex_integration_seam_hunt.py",
    )

    write_summary_file(
        output_dir=output_dir,
        stats=stats,
        total_files=len(files),
        title="Integration Seam Analysis Summary",
        defects_label="Integration Seam Defects Found",
        clean_label="Clean Files",
    )

    write_findings_index(
        output_dir=output_dir,
        repo_root=repo_root,
        title="Integration Seam Findings Index",
        file_column_label="Source File",
        no_defect_marker="No integration seam defect found",
        clean_section_title="Clean Files",
    )

    # Print summary to stdout for immediate feedback
    print_summary(stats, icon="🏗️", title="Integration Seam Analysis Summary")

    # Tell user where to find the results
    print(f"📁 Detailed results written to: {output_dir.relative_to(repo_root)}/")
    print("   - RUN_METADATA.md: Run details and parameters")
    print("   - SUMMARY.md: Triage dashboard and statistics")
    print("   - FINDINGS_INDEX.md: Sortable table of all findings")
    print("   - INTEGRATION_SEAM_LOG.md: Execution log")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
