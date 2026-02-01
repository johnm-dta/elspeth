#!/usr/bin/env python3
"""Defensive programming exemption validator using Codex.

Validates that whitelist entries in config/cicd/contracts-whitelist.yaml are:
- Actually present in the codebase (not stale)
- Legitimately exempt per CLAUDE.md defensive programming rules
- Properly classified according to three-tier trust model
- Sufficiently justified

This ensures the whitelist doesn't hide bugs or violate audit integrity.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import sys
import time
from pathlib import Path

import yaml
from codex_audit_common import (  # type: ignore[import-not-found]
    AsyncTqdm,
    chunked,
    ensure_log_file,
    generate_summary,
    get_git_commit,
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


def _parse_dict_pattern_entry(entry: str) -> tuple[str, str, str]:
    """Parse dict pattern entry like 'file.py:function:param_name'.

    Returns (file_path, context, param_name).
    """
    parts = entry.split(":")
    if len(parts) < 3:
        raise ValueError(f"Invalid dict pattern entry: {entry}")

    file_path = parts[0]
    context = ":".join(parts[1:-1])  # Handle nested classes/methods
    param_name = parts[-1]

    return file_path, context, param_name


def _parse_external_type_entry(entry: str) -> tuple[str, str]:
    """Parse external type entry like 'module/path:TypeName'.

    Returns (module_path, type_name).
    """
    parts = entry.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid external type entry: {entry}")

    return parts[0], parts[1]


def _build_exemption_validation_prompt(
    exemption_entry: str,
    exemption_type: str,
    template: str,
    context: str,
    repo_root: Path,
) -> str:
    return (
        "You are a defensive programming exemption auditor validating whitelist entries.\n"
        f"Exemption Entry: {exemption_entry}\n"
        f"Exemption Type: {exemption_type}\n\n"
        "Instructions:\n"
        "- Use the exemption validation defect template below verbatim.\n"
        "- Fill in every section. If unknown, write 'Unknown'.\n"
        "- You MUST read the actual code to verify the exemption is used correctly.\n"
        "- You MUST read CLAUDE.md sections on defensive programming and three-tier trust model.\n"
        "- Report defect if exemption violates CLAUDE.md rules or is stale.\n"
        "- If exemption is valid and properly used, output one template with Summary set to\n"
        f"  'Exemption validated: {exemption_entry}', Severity 'trivial', Priority 'P3',\n"
        "  and Exemption Type 'None - Valid exemption'.\n"
        "- Evidence MUST cite actual code with file paths and line numbers.\n"
        "- Evidence MUST quote relevant CLAUDE.md rules.\n"
        "- Vague claims without evidence will be automatically downgraded to P3.\n\n"
        "Severity Guidelines:\n"
        "- P0 (Critical): Bug-hiding pattern, violates audit integrity, wrong trust tier classification\n"
        "- P1 (High): Invalid exemption per CLAUDE.md, stale entry that no longer exists\n"
        "- P2 (Medium): Insufficient justification, unclear trust tier\n"
        "- P3 (Low): Valid but poorly documented\n\n"
        "CLAUDE.md Defensive Programming Rules:\n"
        "\n"
        "**PROHIBITION:** Do not use .get(), getattr(), hasattr(), isinstance(), or silent exception\n"
        "handling to suppress errors from nonexistent attributes, malformed data, or incorrect types.\n"
        "\n"
        "**Legitimate Uses (from CLAUDE.md):**\n"
        "\n"
        "1. **Operations on Row Values (Their Data)**:\n"
        "   - Even type-valid row data can cause operation failures\n"
        "   - Example: `row['numerator'] / row['denominator']` (ZeroDivisionError from their data)\n"
        "   - Rule: Wrap operations on `row[...]` values, NOT our internal state\n"
        "   - Distinction: `row['x'] / row['y']` (their data, OK) vs `self._x / self._y` (our bug, CRASH)\n"
        "\n"
        "2. **External System Boundaries**:\n"
        "   - External API responses (validating JSON structure from LLM/HTTP)\n"
        "   - Source plugin input (coercing/validating external data at ingestion)\n"
        "   - Trust Boundary: Data from systems we don't control\n"
        "\n"
        "3. **Framework Boundaries**:\n"
        "   - Plugin schema contracts (type checking at plugin boundaries)\n"
        "   - Configuration validation (Pydantic validators rejecting malformed config)\n"
        "   - Interface compliance checks\n"
        "\n"
        "4. **Serialization**:\n"
        "   - Pandas dtype normalization (numpy.int64 → int)\n"
        "   - Serialization polymorphism (datetime, Decimal, bytes)\n"
        "   - Canonical JSON conversion\n"
        "\n"
        "**Three-Tier Trust Model (from CLAUDE.md):**\n"
        "\n"
        "**Tier 1: Our Data (Audit Database / Landscape) - FULL TRUST**\n"
        "- Must be 100% pristine at all times\n"
        "- Bad data in audit trail = **CRASH IMMEDIATELY**\n"
        "- No coercion, no defaults, no silent recovery\n"
        "- Wrong type = crash, NULL where unexpected = crash, invalid enum = crash\n"
        "- Reason: Audit trail is legal record, coercing bad data is evidence tampering\n"
        "\n"
        "**Tier 2: Pipeline Data (Post-Source) - ELEVATED TRUST**\n"
        "- Type-valid but potentially operation-unsafe\n"
        "- Data passed source validation\n"
        "- Types are trustworthy, values might cause operation failures\n"
        "- **No coercion** at transform/sink level - if types wrong, upstream bug\n"
        "- **Wrap operations** on row values (division, parsing, etc.)\n"
        "- If transform receives wrong type, that's a bug in source/upstream transform\n"
        "\n"
        "**Tier 3: External Data (Source Input) - ZERO TRUST**\n"
        "- Can be literal trash - we don't control external systems\n"
        "- Malformed CSV, NULLs everywhere, wrong types, unexpected JSON\n"
        "- **Validate at boundary, coerce where possible, record what we got**\n"
        "- Sources MAY coerce: '42' → 42, 'true' → True\n"
        "- Quarantine rows that can't be coerced/validated\n"
        "\n"
        "**Decision Test (from CLAUDE.md):**\n"
        "\n"
        "| Question | If Yes | If No |\n"
        "|----------|--------|-------|\n"
        "| Is this protecting against user-provided data values? | ✅ Wrap it | — |\n"
        "| Is this at an external system boundary (API, file, DB)? | ✅ Wrap it | — |\n"
        "| Would this fail due to a bug in code we control? | — | ❌ Let it crash |\n"
        "| Am I adding this because 'something might be None'? | — | ❌ Fix root cause |\n"
        "\n"
        "**Common Invalid Exemptions:**\n"
        "- Using .get() on Landscape query results (Tier 1 - must crash on anomaly)\n"
        "- hasattr() on plugin protocol methods (system code - let it crash)\n"
        "- Silent except catching bugs in our code (hides bugs)\n"
        "- getattr() with default for internal state (masks initialization bugs)\n"
        "\n"
        "**Common Valid Exemptions:**\n"
        "- dict[str, Any] for row data (user pipeline data, inherently dynamic)\n"
        "- dict[str, Any] for plugin config (validated by Pydantic before use)\n"
        "- try/except on row value operations (their data can fail operations)\n"
        "- Validating external API responses (zero trust boundary)\n"
        "\n"
        "Validation Strategy:\n"
        "- [ ] Read CLAUDE.md defensive programming section completely\n"
        "- [ ] Read CLAUDE.md three-tier trust model section completely\n"
        "- [ ] Parse exemption entry to extract file path and context\n"
        "- [ ] Read the actual source code at that location\n"
        "- [ ] Verify the code pattern exists (not stale)\n"
        "- [ ] Classify the data according to three-tier model\n"
        "- [ ] Apply decision test to determine if exemption is legitimate\n"
        "- [ ] Check if justification in whitelist matches actual usage\n"
        "- [ ] Provide SPECIFIC line numbers and code snippets\n"
        "- [ ] Quote relevant CLAUDE.md rules\n"
        "\n"
        f"Repository context (read-only):\n"
        f"{context}\n\n"
        "Exemption validation defect template:\n"
        f"{template}\n"
    )


async def _run_batches(
    *,
    exemptions: list[tuple[str, str]],  # (entry, type)
    output_dir: Path,
    model: str | None,
    prompt_template: str,
    repo_root: Path,
    skip_existing: bool,
    batch_size: int,
    log_path: Path,
    context: str,
    rate_limit: int | None,
) -> dict[str, int]:
    """Run analysis in batches. Returns statistics."""
    log_lock = asyncio.Lock()
    failed_entries: list[tuple[str, Exception]] = []
    total_gated = 0

    # Use pyrate-limiter for rate limiting
    rate_limiter = Limiter(Rate(rate_limit, Duration.MINUTE)) if rate_limit else None
    pbar = AsyncTqdm(total=len(exemptions), desc="Validating exemptions", unit="exemption")

    for batch in chunked([e[0] for e in exemptions], batch_size):
        tasks: list[asyncio.Task[dict[str, int]]] = []
        batch_entries: list[str] = []

        for entry in batch:
            # Find the exemption type for this entry
            exemption_type = next((t for e, t in exemptions if e == entry), "unknown")

            # Create output path based on entry
            # Sanitize entry for filename (replace special chars)
            safe_entry = re.sub(r"[^\w\-_.]", "_", entry)
            output_path = output_dir / f"{safe_entry}.md"

            if skip_existing and output_path.exists():
                pbar.update(1)
                continue

            prompt = _build_exemption_validation_prompt(entry, exemption_type, prompt_template, context, repo_root)
            batch_entries.append(entry)
            # Note: exemption_entry parameter matches file_path in signature (accepts both names)
            tasks.append(
                asyncio.create_task(
                    run_codex_with_retry_and_logging(
                        file_path=Path(entry),  # Dummy path for compatibility
                        output_path=output_path,
                        model=model,
                        prompt=prompt,
                        repo_root=repo_root,
                        log_path=log_path,
                        log_lock=log_lock,
                        file_display=entry,
                        output_display=str(output_path.relative_to(repo_root).as_posix()),
                        rate_limiter=rate_limiter,
                        evidence_gate_summary_prefix="and code snippets",
                    )
                )
            )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for entry, result in zip(batch_entries, results, strict=True):
                if isinstance(result, Exception):
                    failed_entries.append((entry, result))
                elif isinstance(result, dict):
                    total_gated += result["gated"]  # Fixed: removed .get() - our data (Tier 2)
                pbar.update(1)

    pbar.close()

    if failed_entries:
        print(f"\n⚠️  {len(failed_entries)} exemptions failed:", file=sys.stderr)
        for entry, exc in failed_entries[:10]:
            print(f"  {entry}: {exc}", file=sys.stderr)
        if len(failed_entries) > 10:
            print(f"  ... and {len(failed_entries) - 10} more (see {log_path})", file=sys.stderr)

    summary: dict[str, int] = generate_summary(output_dir, no_defect_marker="Exemption validated:")
    summary["gated"] = total_gated
    return summary


def _load_whitelist(whitelist_path: Path) -> list[tuple[str, str]]:
    """Load whitelist file and return list of (entry, type) tuples."""
    with whitelist_path.open() as f:
        whitelist_data = yaml.safe_load(f)

    exemptions: list[tuple[str, str]] = []

    # Load allowed_external_types
    if "allowed_external_types" in whitelist_data:
        for entry in whitelist_data["allowed_external_types"]:
            exemptions.append((entry, "external_type"))

    # Load allowed_dict_patterns
    if "allowed_dict_patterns" in whitelist_data:
        for entry in whitelist_data["allowed_dict_patterns"]:
            exemptions.append((entry, "dict_pattern"))

    return exemptions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Codex defensive programming exemption validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all whitelist entries
  %(prog)s

  # Dry run to see what would be validated
  %(prog)s --dry-run

  # Use rate limiting
  %(prog)s --rate-limit 30
        """,
    )
    parser.add_argument(
        "--whitelist",
        default="config/cicd/contracts-whitelist.yaml",
        help="Whitelist file to validate (default: config/cicd/contracts-whitelist.yaml).",
    )
    parser.add_argument(
        "--template",
        default="docs/quality-audit/EXEMPTION_VALIDATION_DEFECT_TEMPLATE.md",
        help="Exemption validation defect template path.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/quality-audit/findings-exemptions",
        help="Directory to write validation reports.",
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
        help="Skip exemptions that already have an output report.",
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
        help="Show which exemptions would be validated without running analysis.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.rate_limit is not None and args.rate_limit < 1:
        raise ValueError("--rate-limit must be >= 1")

    if shutil.which("codex") is None:
        raise RuntimeError("codex CLI not found on PATH")

    repo_root = Path(__file__).resolve().parents[1]
    whitelist_path = resolve_path(repo_root, args.whitelist)
    template_path = resolve_path(repo_root, args.template)
    output_dir = resolve_path(repo_root, args.output_dir)
    log_path = resolve_path(repo_root, "docs/quality-audit/EXEMPTION_VALIDATION_LOG.md")

    # Load whitelist
    exemptions = _load_whitelist(whitelist_path)

    if not exemptions:
        print(f"No exemptions found in {whitelist_path}", file=sys.stderr)
        return 1

    # Dry run mode
    if args.dry_run:
        print(f"Would validate {len(exemptions)} exemptions:")
        for entry, etype in exemptions[:20]:
            print(f"  [{etype}] {entry}")
        if len(exemptions) > 20:
            print(f"  ... and {len(exemptions) - 20} more")
        return 0

    # Get git commit for metadata
    git_commit = get_git_commit(repo_root)

    # Load template and context (include whitelist as extra file)
    template_text = template_path.read_text(encoding="utf-8")
    extra_files = ["config/cicd/contracts-whitelist.yaml"]
    if args.context_files:
        extra_files.extend(args.context_files)
    context_text = load_context(repo_root, extra_files=extra_files)
    ensure_log_file(log_path, header_title="Codex Exemption Validation Log")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track execution time
    start_time = utc_now()
    start_monotonic = time.monotonic()

    # Run analysis
    stats = asyncio.run(
        _run_batches(
            exemptions=exemptions,
            output_dir=output_dir,
            model=args.model,
            prompt_template=template_text,
            repo_root=repo_root,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
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
        files_scanned=len(exemptions),
        model=args.model,
        batch_size=args.batch_size,
        rate_limit=args.rate_limit,
        git_commit=git_commit,
        title="Exemption Validation Run Metadata",
        script_name="scripts/codex_exemption_validator.py",
    )

    write_summary_file(
        output_dir=output_dir,
        stats=stats,
        total_files=len(exemptions),
        title="Exemption Validation Summary",
        defects_label="Invalid Exemptions Found",
        clean_label="Valid Exemptions",
    )

    write_findings_index(
        output_dir=output_dir,
        repo_root=repo_root,
        title="Exemption Validation Findings Index",
        file_column_label="Exemption Entry",
        no_defect_marker="Exemption validated:",
        clean_section_title="Valid Exemptions",
    )

    # Print summary to stdout for immediate feedback
    print_summary(stats, icon="🛡️", title="Exemption Validation Summary")

    # Tell user where to find the results
    print(f"📁 Detailed results written to: {output_dir.relative_to(repo_root)}/")
    print("   - RUN_METADATA.md: Run details and parameters")
    print("   - SUMMARY.md: Triage dashboard and statistics")
    print("   - FINDINGS_INDEX.md: Sortable table of all findings")
    print("   - EXEMPTION_VALIDATION_LOG.md: Execution log")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
