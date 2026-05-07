#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import asyncio
import fnmatch
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from codex_audit_common import (  # type: ignore[import-not-found]
    USAGE_STAT_KEYS,
    AsyncTqdm,
    chunked,
    ensure_log_file,
    extract_section,
    generate_summary,
    get_codex_version,
    get_git_commit,
    is_cache_path,
    iter_report_files,
    load_context,
    make_codex_rate_limiter,
    print_summary,
    priority_from_report,
    resolve_path,
    run_codex_with_retry_and_logging,
    sha256_text,
    utc_now,
    write_findings_index,
    write_run_metadata,
    write_summary_file,
)


def _is_python_file(path: Path) -> bool:
    """Check if path is a Python source file (not test)."""
    return path.suffix == ".py" and not path.name.startswith("test_")


def _load_tier_allowlist(repo_root: Path) -> list[dict[str, Any]]:
    """Load tier model per-file rules from all module allowlist YAML files."""
    allowlist_dir = repo_root / "config" / "cicd" / "enforce_tier_model"
    if not allowlist_dir.exists():
        return []

    if allowlist_dir.is_dir():
        yaml_files = sorted(path for path in allowlist_dir.glob("*.yaml") if path.name != "_defaults.yaml")
    else:
        yaml_files = [allowlist_dir]

    rules: list[dict[str, Any]] = []
    for path in yaml_files:
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in data.get("per_file_rules", []):
            if isinstance(entry, dict):
                rules.append(entry)
    return rules


def _allowlist_entries_for_file(file_path: Path, allowlist_root: Path, allowlist: list[dict[str, Any]]) -> str | None:
    """Extract tier-model allowlist entries relevant to a specific target file.

    Returns formatted text for prompt injection, or None if no entries match.
    """
    try:
        relative = file_path.relative_to(allowlist_root).as_posix()
    except ValueError:
        relative = file_path.as_posix()
    matches: list[str] = []
    for entry in allowlist:
        pattern = str(entry.get("pattern", "")).replace("\\", "/")
        if fnmatch.fnmatchcase(relative, pattern):
            rules = ", ".join(str(rule) for rule in entry.get("rules", []))
            reason = entry.get("reason", "no reason given")
            max_hits = entry.get("max_hits")
            line = f"  - Rules {rules}: {reason}"
            if max_hits is not None:
                line += f" (max_hits={max_hits})"
            matches.append(line)
    if not matches:
        return None
    return (
        "Tier Model Allowlist Entries for This File:\n"
        "The following defensive patterns are ALLOWLISTED in this file and should\n"
        "NOT be reported as bugs unless they exceed their max_hits count:\n" + "\n".join(matches)
    )


def _node_line(source_lines: list[str], lineno: int) -> str:
    if lineno < 1 or lineno > len(source_lines):
        return ""
    return source_lines[lineno - 1].strip()


def _build_static_prepass_context(file_path: Path, *, repo_root: Path, root_dir: Path) -> str:
    """Build deterministic target-file facts for Codex to verify, not assume."""
    del root_dir
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "Static Pre-pass Context:\n- Could not decode target as UTF-8."

    source_lines = text.splitlines()
    imports: list[str] = []
    risk_lines: list[str] = []

    try:
        tree = ast.parse(text, filename=str(file_path))
    except SyntaxError as exc:
        return f"Static Pre-pass Context:\n- SyntaxError while parsing target: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            imports.append(f"  - {file_path.relative_to(repo_root)}:{node.lineno}: {_node_line(source_lines, node.lineno)}")
        elif isinstance(node, ast.ExceptHandler):
            broad = node.type is None
            if isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
                broad = True
            if broad:
                risk_lines.append(
                    f"  - Broad exception handler at {file_path.relative_to(repo_root)}:{node.lineno}: "
                    f"{_node_line(source_lines, node.lineno)}"
                )

    pattern_checks = [
        ("defensive .get usage", re.compile(r"\.get\(")),
        ("getattr usage", re.compile(r"\bgetattr\(")),
        ("hasattr usage", re.compile(r"\bhasattr\(")),
        ("logger usage", re.compile(r"\blogging\.getLogger\(|\blogger\.")),
        ("blocking sleep", re.compile(r"\btime\.sleep\(")),
        ("subprocess call", re.compile(r"\bsubprocess\.(run|Popen|check_call|check_output)\(")),
    ]
    rel_path = file_path.relative_to(repo_root)
    for line_no, line in enumerate(source_lines, start=1):
        stripped = line.strip()
        for label, pattern in pattern_checks:
            if pattern.search(stripped):
                risk_lines.append(f"  - {label} at {rel_path}:{line_no}: {stripped}")

    sibling_tests = sorted((repo_root / "tests").rglob(f"test_{file_path.stem}.py")) if (repo_root / "tests").exists() else []

    sections = ["Static Pre-pass Context:"]
    if imports:
        sections.append("Imports:\n" + "\n".join(imports[:20]))
    if risk_lines:
        sections.append("Static risk leads to verify:\n" + "\n".join(risk_lines[:30]))
    if sibling_tests:
        test_lines = [f"  - {path.relative_to(repo_root)}" for path in sibling_tests[:10]]
        sections.append("Potential sibling tests:\n" + "\n".join(test_lines))
    if len(sections) == 1:
        sections.append("- No deterministic static leads found by the pre-pass.")
    return "\n".join(sections)


def _safe_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _imported_module_from_node(node: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(node, ast.ImportFrom):
        if node.module:
            return "." * node.level + node.module if node.level else node.module
        return "." * node.level if node.level else None
    if node.names:
        return node.names[0].name
    return None


def _module_path_for_import(module: str, repo_root: Path) -> Path | None:
    if not module.startswith("elspeth."):
        return None
    relative = Path("src") / Path(*module.split("."))
    module_path = repo_root / relative.with_suffix(".py")
    if module_path.exists():
        return module_path.resolve()
    package_path = repo_root / relative / "__init__.py"
    if package_path.exists():
        return package_path.resolve()
    return None


def _build_subsystem_context(root_dir: Path, *, repo_root: Path) -> str:
    """Build a deterministic subsystem map once per run for cacheable context."""
    source_root = root_dir if root_dir.is_dir() else root_dir.parent
    python_files = sorted(path for path in source_root.rglob("*.py") if path.is_file() and not is_cache_path(path))

    imports: list[str] = []
    definitions: list[str] = []
    reverse_imports: list[str] = []
    markers: list[str] = []
    tests: list[str] = []
    import_edges: list[tuple[Path, Path]] = []
    test_root = repo_root / "tests"

    for path in python_files[:200]:
        rel_path = _safe_relative(path, repo_root)
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            markers.append(f"  - {rel_path}: parse skipped ({type(exc).__name__})")
            continue

        source_lines = text.splitlines()
        for node in tree.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                module = _imported_module_from_node(node)
                if module and (module.startswith("elspeth") or module.startswith(".")):
                    imports.append(f"  - {rel_path}:{node.lineno}: {_node_line(source_lines, node.lineno)}")
                    imported_path = _module_path_for_import(module, repo_root)
                    if imported_path is not None and _is_under_root(imported_path, source_root):
                        import_edges.append((imported_path, path.resolve()))
            elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                if not node.name.startswith("_"):
                    keyword = "class" if isinstance(node, ast.ClassDef) else "def"
                    definitions.append(f"  - {rel_path}:{node.lineno}: {keyword} {node.name}")
                    if isinstance(node, ast.ClassDef):
                        base_names = [ast.unparse(base) for base in node.bases]
                        if any(token in base for base in base_names for token in ("Config", "Settings", "Protocol", "Schema", "BaseModel")):
                            markers.append(f"  - {rel_path}:{node.lineno}: {keyword} {node.name}({', '.join(base_names)})")

        for line_no, line in enumerate(source_lines, start=1):
            stripped = line.strip()
            if "@hookimpl" in stripped or "hookimpl(" in stripped or "entry_points" in stripped or "SchemaContract" in stripped:
                markers.append(f"  - {rel_path}:{line_no}: {stripped}")

        if test_root.exists():
            for test_path in sorted(test_root.rglob(f"test_{path.stem}.py"))[:5]:
                tests.append(f"  - {_safe_relative(test_path, repo_root)} covers {rel_path}")

    for imported_path, importer_path in sorted(set(import_edges)):
        reverse_imports.append(f"  - {_safe_relative(imported_path, repo_root)} <- {_safe_relative(importer_path, repo_root)}")

    sections = [
        "Subsystem Static Map:",
        f"- Root: {_safe_relative(source_root, repo_root)}",
        f"- Python files discovered: {len(python_files)}",
    ]
    if definitions:
        sections.append("Public definitions:\n" + "\n".join(definitions[:80]))
    if imports:
        sections.append("Internal imports:\n" + "\n".join(imports[:80]))
    if reverse_imports:
        sections.append("Reverse import hints:\n" + "\n".join(reverse_imports[:80]))
    if markers:
        sections.append("Plugin/contract markers:\n" + "\n".join(markers[:80]))
    if tests:
        sections.append("Adjacent tests:\n" + "\n".join(tests[:80]))
    return "\n".join(sections)


def _resolve_structured_output_schema(
    repo_root: Path,
    *,
    structured_output: bool,
    structured_output_schema: str | None,
    no_structured_output: bool,
) -> Path | None:
    """Resolve the schema path. Bug hunt defaults to structured output."""
    if no_structured_output:
        return None
    if structured_output_schema:
        return cast(Path, resolve_path(repo_root, structured_output_schema))
    # The explicit flag is retained for backward-compatible CLI habits; structured
    # output is now the default for this scanner.
    _ = structured_output
    return repo_root / "scripts" / "schemas" / "codex_bug_hunt_report.schema.json"


def _build_prompt(
    file_path: Path,
    template: str,
    context: str,
    extra_message: str | None = None,
    target_context: str | None = None,
    allowlist_context: str | None = None,
    structured_output: bool = False,
) -> str:
    structured_instruction = (
        "- Return JSON matching the supplied output schema. Put the complete\n"
        "  Markdown bug report content in the `markdown_report` field and\n"
        "  machine-readable finding summaries in `findings`. Use an empty\n"
        "  `findings` array when no concrete bug is found.\n"
        if structured_output
        else ""
    )
    run_level_context = f"Run-level context:\n{extra_message}\n\n" if extra_message else ""
    return (
        "You are a static analysis agent doing a deep bug audit.\n"
        "\n"
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
        "  'No concrete bug found in the target file', Severity 'trivial', Priority 'P3',\n"
        "  and Root Cause Hypothesis 'No bug identified'.\n"
        "- Evidence should cite file paths and line numbers when possible.\n" + structured_instruction + "\n"
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
        "- [ ] Consult the Skill references (tier model, engine patterns, config\n"
        "      contracts, logging/telemetry) included in the repository context below\n"
        "- [ ] Check plugin protocol compliance (hookspec signatures)\n"
        "- [ ] Verify schema contracts match implementation\n"
        "- [ ] Trace data flow for audit trail completeness\n"
        "- [ ] Check error paths for quarantine vs crash decisions\n"
        "- [ ] Validate integration tests exist for edge cases\n"
        "- [ ] Look for untested error conditions\n"
        "- [ ] Check for missing type annotations\n"
        "\n" + run_level_context + "Repository context (read-only):\n"
        f"{context}\n\n"
        "Bug report template:\n"
        f"{template}\n\n"
        "Target-specific context:\n"
        f"Target file: {file_path}\n"
        + (f"\n{target_context}\n" if target_context else "")
        + (f"\n{allowlist_context}\n" if allowlist_context else "")
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
    if by_priority_dir.exists():
        shutil.rmtree(by_priority_dir)

    for md_file in iter_report_files(output_dir):
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
    extra_message: str | None = None,
    tier_allowlist: list[dict[str, Any]] | None = None,
    allowlist_root: Path | None = None,
    structured_output_schema: Path | None = None,
    warm_up_cache: bool = False,
    profile: str | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    ephemeral: bool = False,
    timeout_s: float | None = 1800.0,
) -> dict[str, int]:
    """Run analysis in batches. Returns statistics."""
    log_lock = asyncio.Lock()
    failed_files: list[tuple[Path, Exception]] = []
    total_merged = 0
    total_gated = 0
    usage_totals = dict.fromkeys(USAGE_STAT_KEYS, 0)

    rate_limiter = make_codex_rate_limiter(rate_limit)

    # Progress bar
    pbar = AsyncTqdm(total=len(files), desc="Analyzing files", unit="file")

    async def run_one_file(file_path: Path) -> dict[str, int] | None:
        try:
            relative = file_path.relative_to(root_dir)
            output_path = output_dir / relative
            output_path = output_path.with_suffix(output_path.suffix + ".md")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if skip_existing and output_path.exists():
                return None

            effective_allowlist_root = allowlist_root or root_dir
            allowlist_ctx = _allowlist_entries_for_file(file_path, effective_allowlist_root, tier_allowlist) if tier_allowlist else None
            target_context_parts = [_build_static_prepass_context(file_path, repo_root=repo_root, root_dir=root_dir)]
            target_extra_message = "\n\n".join(part for part in target_context_parts if part)
            prompt = _build_prompt(
                file_path,
                prompt_template,
                context,
                extra_message=extra_message,
                target_context=target_extra_message,
                allowlist_context=allowlist_ctx,
                structured_output=structured_output_schema is not None,
            )

            result = cast(
                dict[str, int],
                await run_codex_with_retry_and_logging(
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
                    output_schema=structured_output_schema,
                    structured_markdown_field="markdown_report",
                    profile=profile,
                    reasoning_effort=reasoning_effort,
                    service_tier=service_tier,
                    ephemeral=ephemeral,
                    timeout_s=timeout_s,
                ),
            )

            # Deduplicate after successful run
            if deduplicate and bugs_open_dir and bugs_open_dir.exists():
                merged_count = _deduplicate_and_merge(output_path, bugs_open_dir, repo_root)
                result["merged"] = merged_count

            return result
        finally:
            pbar.update(1)

    async def run_and_capture(file_path: Path) -> tuple[Path, dict[str, int] | Exception | None]:
        try:
            return file_path, await run_one_file(file_path)
        except Exception as exc:
            return file_path, exc

    def record_result(file_path: Path, result: dict[str, int] | Exception | None) -> None:
        nonlocal total_gated, total_merged
        if result is None:
            return
        if isinstance(result, Exception):
            failed_files.append((file_path, result))
            return
        total_gated += result.get("gated", 0)
        total_merged += result.get("merged", 0)
        for key in USAGE_STAT_KEYS:
            usage_totals[key] += result.get(key, 0)

    remaining_files = list(files)
    if warm_up_cache and batch_size > 1 and len(remaining_files) > 1:
        warm_file = remaining_files.pop(0)
        file_path, result = await run_and_capture(warm_file)
        record_result(file_path, result)

    for batch in chunked(remaining_files, batch_size):
        tasks: list[asyncio.Task[tuple[Path, dict[str, int] | Exception | None]]] = []

        for file_path in batch:
            task = asyncio.create_task(run_and_capture(file_path))
            tasks.append(task)

        # Wait for all tasks in batch to complete
        if tasks:
            results = await asyncio.gather(*tasks)
            for file_path, result in results:
                record_result(file_path, result)

    pbar.close()

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
    for key in USAGE_STAT_KEYS:
        summary[key] = usage_totals[key]
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

  # Add extra context message (e.g., migration notes)
  %(prog)s --extra-message "Please note recent PipelineRow migration - see docs/plans/..."
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
        "--profile",
        default=None,
        help="Codex config profile to use for each codex exec run.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["minimal", "low", "medium", "high", "xhigh"],
        help="Override Codex model_reasoning_effort for each run.",
    )
    parser.add_argument(
        "--service-tier",
        default=None,
        choices=["flex", "fast"],
        help="Override Codex service_tier for each run.",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Run codex exec without persisting session rollout files.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1800.0,
        help="Per-file codex exec timeout in seconds (default: 1800).",
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
    parser.add_argument(
        "--extra-message",
        default=None,
        help="Additional context message to include in the analysis prompt (e.g., migration notes).",
    )
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Use Codex --output-schema and extract markdown_report back into the normal .md output (default).",
    )
    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        help="Disable Codex --output-schema and request raw Markdown output.",
    )
    parser.add_argument(
        "--structured-output-schema",
        default=None,
        help="Custom JSON schema for --structured-output (default: scripts/schemas/codex_bug_hunt_report.schema.json).",
    )
    parser.add_argument(
        "--no-cache-warm-up",
        action="store_true",
        help="Disable the initial serial run that warms Codex prompt caching before parallel batches.",
    )

    args = parser.parse_args()

    # Validation
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.rate_limit is not None and args.rate_limit < 1:
        raise ValueError("--rate-limit must be >= 1")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be > 0")

    if shutil.which("codex") is None:
        raise RuntimeError("codex CLI not found on PATH")

    repo_root = Path(__file__).resolve().parents[1]
    root_dir = resolve_path(repo_root, args.root)
    template_path = resolve_path(repo_root, args.template)
    output_dir = resolve_path(repo_root, args.output_dir)
    log_path = resolve_path(repo_root, "docs/bugs/process/CODEX_LOG.md")
    bugs_open_dir = resolve_path(repo_root, args.bugs_dir) if args.deduplicate else None
    structured_output_schema = _resolve_structured_output_schema(
        repo_root,
        structured_output=args.structured_output,
        structured_output_schema=args.structured_output_schema,
        no_structured_output=args.no_structured_output,
    )
    if structured_output_schema is not None and not structured_output_schema.exists():
        raise RuntimeError(f"Structured output schema not found: {structured_output_schema}")

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

    git_commit = get_git_commit(repo_root)

    # Load template, context (with skills), and tier model allowlist
    template_text = template_path.read_text(encoding="utf-8")
    context_text = load_context(repo_root, extra_files=args.context_files, include_skills=True)
    subsystem_context = _build_subsystem_context(root_dir, repo_root=repo_root)
    context_text = "\n\n".join(part for part in (context_text, subsystem_context) if part)
    tier_allowlist = _load_tier_allowlist(repo_root)
    ensure_log_file(log_path, header_title="Codex Bug Hunt Log")
    output_dir.mkdir(parents=True, exist_ok=True)

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
            organize_by_priority=args.organize_by_priority,
            bugs_open_dir=bugs_open_dir,
            deduplicate=args.deduplicate,
            extra_message=args.extra_message,
            tier_allowlist=tier_allowlist,
            allowlist_root=repo_root / "src" / "elspeth",
            structured_output_schema=structured_output_schema,
            warm_up_cache=not args.no_cache_warm_up,
            profile=args.profile,
            reasoning_effort=args.reasoning_effort,
            service_tier=args.service_tier,
            ephemeral=args.ephemeral,
            timeout_s=args.timeout_s,
        )
    )

    end_time = utc_now()
    duration_s = time.monotonic() - start_monotonic

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
        title="Codex Bug Hunt Run Metadata",
        script_name="scripts/codex_bug_hunt.py",
        extra_parameters={
            "Codex CLI": get_codex_version(repo_root),
            "Runner": "codex exec (Pro-backed CLI)",
            "Output Mode": "structured-json-to-markdown" if structured_output_schema else "markdown",
            "Prompt Cache Warm-up": "enabled" if not args.no_cache_warm_up else "disabled",
            "Codex Profile": args.profile or "default",
            "Reasoning Effort": args.reasoning_effort or "default",
            "Service Tier": args.service_tier or "default",
            "Ephemeral Sessions": "enabled" if args.ephemeral else "disabled",
            "Per-file Timeout Seconds": f"{args.timeout_s:g}",
            "Structured Output Schema": (str(structured_output_schema.relative_to(repo_root)) if structured_output_schema else "disabled"),
            "Scan Root": str(root_dir.relative_to(repo_root)) if _is_under_root(root_dir, repo_root) else str(root_dir),
            "Context SHA256": sha256_text(context_text),
            "Template SHA256": sha256_text(template_text),
        },
    )

    write_summary_file(
        output_dir=output_dir,
        stats=stats,
        total_files=len(files),
        title="Codex Bug Hunt Summary",
        defects_label="Bugs Found",
        clean_label="Clean Files",
    )

    write_findings_index(
        output_dir=output_dir,
        repo_root=repo_root,
        title="Codex Bug Hunt Findings Index",
        file_column_label="Source File",
        no_defect_marker="No concrete bug found",
        clean_section_title="Clean Files",
    )

    # Print summary
    print_summary(stats, icon="🐛", title="Bug Hunt Summary")
    print(f"Detailed results written to: {output_dir.relative_to(repo_root)}/")
    print("   - RUN_METADATA.md: Run details and parameters")
    print("   - SUMMARY.md: Triage dashboard and statistics")
    print("   - FINDINGS_INDEX.md: Sortable table of all findings")
    print("   - CODEX_LOG.md: Execution log")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
