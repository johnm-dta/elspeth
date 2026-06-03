#!/usr/bin/env python3
"""Generate test bug tickets from test audit files.

This script parses all audit files in docs/test_audit/ that have ISSUES_FOUND status
and generates structured bug tickets in docs/test_bugs/open/.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class AuditFinding:
    """A single finding from an audit file."""

    level: str  # Warning, Info
    line_ref: str
    description: str


@dataclass
class AuditFile:
    """Parsed audit file content."""

    filename: str
    test_file: str
    lines: int
    test_count: int
    summary: str
    findings: list[AuditFinding]
    verdict: str
    verdict_detail: str
    category: str  # Derived from test path


def parse_test_path(filename: str) -> tuple[str, str]:
    """Extract category and actual test file path from audit filename."""
    # Remove .audit.md suffix and convert underscores to path separators
    name = filename.replace(".audit.md", "").replace(".audit", "")

    # Map common prefixes to categories
    if name.startswith("tests_cli_"):
        return "cli", f"tests/cli/{name.replace('tests_cli_', '')}"
    elif name.startswith("tests_contracts_config_"):
        return "contracts", f"tests/contracts/config/{name.replace('tests_contracts_config_', '')}"
    elif name.startswith("tests_contracts_sink_contracts_"):
        return "contracts", f"tests/contracts/sink_contracts/{name.replace('tests_contracts_sink_contracts_', '')}"
    elif name.startswith("tests_contracts_source_contracts_"):
        return "contracts", f"tests/contracts/source_contracts/{name.replace('tests_contracts_source_contracts_', '')}"
    elif name.startswith("tests_contracts_transform_contracts_"):
        return "contracts", f"tests/contracts/transform_contracts/{name.replace('tests_contracts_transform_contracts_', '')}"
    elif name.startswith("tests_contracts_"):
        return "contracts", f"tests/contracts/{name.replace('tests_contracts_', '')}"
    elif name.startswith("tests_core_checkpoint_"):
        return "core-checkpoint", f"tests/core/checkpoint/{name.replace('tests_core_checkpoint_', '')}"
    elif name.startswith("tests_core_landscape_"):
        return "core-landscape", f"tests/core/landscape/{name.replace('tests_core_landscape_', '')}"
    elif name.startswith("tests_core_security_"):
        return "core-security", f"tests/core/security/{name.replace('tests_core_security_', '')}"
    elif name.startswith("tests_core_"):
        return "core-config", f"tests/core/{name.replace('tests_core_', '')}"
    elif name.startswith("tests_engine_"):
        return "engine", f"tests/engine/{name.replace('tests_engine_', '')}"
    elif name.startswith("tests_audit_"):
        return "audit", f"tests/audit/{name.replace('tests_audit_', '')}"
    elif name.startswith("tests_plugins_"):
        return "plugins", f"tests/plugins/{name.replace('tests_plugins_', '')}"
    elif name.startswith("test_"):
        return "cli", f"tests/cli/{name}"
    else:
        return "misc", f"tests/{name}"


def extract_verdict(content: str) -> tuple[str, str]:
    """Extract verdict type and detail from audit content."""
    verdict_section = re.search(r"## Verdict\s*\n(.*?)(?=\n#|$)", content, re.DOTALL)
    if not verdict_section:
        return "UNKNOWN", ""

    section_text = verdict_section.group(1).strip()

    # Check for bold verdict first
    for v in ["REWRITE", "SPLIT", "DELETE", "MERGE", "KEEP"]:
        pattern = rf"\*\*{v}[^*]*\*\*"
        if re.search(pattern, section_text):
            return v, section_text

    # Check for non-bold verdict
    for v in ["REWRITE", "SPLIT", "DELETE", "MERGE", "KEEP"]:
        if section_text.upper().startswith(v):
            return v, section_text

    return "UNKNOWN", section_text


def parse_findings(content: str) -> list[AuditFinding]:
    """Extract findings from audit content."""
    findings: list[AuditFinding] = []

    # Find the Findings section
    findings_section = re.search(r"## Findings\s*\n(.*?)(?=\n## |$)", content, re.DOTALL)
    if not findings_section:
        return findings

    section_text = findings_section.group(1)

    # Parse warning items (🟡)
    warning_pattern = r"-\s+\*\*([^*]+)\*\*[:\s]*(.*?)(?=\n-|\n###|\n##|$)"
    for match in re.finditer(warning_pattern, section_text, re.DOTALL):
        line_ref = match.group(1).strip()
        description = match.group(2).strip()
        # Clean up description - remove extra whitespace
        description = " ".join(description.split())
        if description:
            findings.append(AuditFinding("Warning", line_ref, description))

    return findings


def parse_audit_file(filepath: Path) -> AuditFile | None:
    """Parse a single audit file."""
    content = filepath.read_text()

    # Skip if not ISSUES_FOUND
    if "ISSUES_FOUND" not in content:
        return None

    filename = filepath.name

    # Extract metadata
    lines_match = re.search(r"\*\*Lines:\*\*\s*(\d+)", content)
    test_count_match = re.search(r"\*\*Test count:\*\*\s*(\d+)", content)

    # Extract summary
    summary_match = re.search(r"## Summary\s*\n(.*?)(?=\n## )", content, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else ""

    # Extract verdict
    verdict, verdict_detail = extract_verdict(content)

    # Extract findings
    findings = parse_findings(content)

    # Get category and test path
    category, test_file = parse_test_path(filename)

    return AuditFile(
        filename=filename,
        test_file=test_file,
        lines=int(lines_match.group(1)) if lines_match else 0,
        test_count=int(test_count_match.group(1)) if test_count_match else 0,
        summary=summary,
        findings=findings,
        verdict=verdict,
        verdict_detail=verdict_detail,
        category=category,
    )


def verdict_to_priority(verdict: str, findings: list[AuditFinding]) -> str:
    """Map verdict to priority level."""
    if verdict == "DELETE":
        return "P3"  # Low priority - just delete
    elif verdict == "REWRITE":
        return "P2"  # Medium - needs significant work
    elif verdict == "SPLIT":
        return "P2"  # Medium - structural change needed
    elif verdict == "KEEP":
        # Check severity of findings
        warning_count = sum(1 for f in findings if f.level == "Warning")
        if warning_count >= 5:
            return "P2"  # Many issues
        elif warning_count >= 2:
            return "P3"  # Some issues
        else:
            return "P3"  # Minor issues
    return "P3"


def verdict_to_severity(verdict: str) -> str:
    """Map verdict to severity."""
    if verdict in ("REWRITE", "DELETE", "SPLIT"):
        return "minor"
    else:
        return "trivial"


def generate_ticket(audit: AuditFile) -> str:
    """Generate a bug ticket from parsed audit."""
    priority = verdict_to_priority(audit.verdict, audit.findings)
    severity = verdict_to_severity(audit.verdict)
    today = datetime.now(UTC).date().isoformat()

    # Format findings as bullet points
    findings_text = ""
    for f in audit.findings:
        findings_text += f"- **{f.line_ref}**: {f.description}\n"

    if not findings_text:
        findings_text = "- See audit file for details\n"

    # Build title from verdict and file
    short_name = audit.test_file.split("/")[-1].replace(".py", "").replace("test_", "")
    title_map = {
        "REWRITE": f"Rewrite weak assertions in {short_name}",
        "SPLIT": f"Split monolithic test file {short_name}",
        "DELETE": f"Remove redundant test file {short_name}",
        "KEEP": f"Fix weak assertions in {short_name}",
    }
    title = title_map.get(audit.verdict, f"Review test issues in {short_name}")

    # Build acceptance criteria from verdict
    ac_map = {
        "REWRITE": """- [ ] Tests have specific, non-permissive assertions
- [ ] Each test verifies the exact expected behavior
- [ ] No "or 'error' in output" fallback patterns
- [ ] Tests fail when actual behavior differs from expected""",
        "SPLIT": """- [ ] Large test file split into focused modules
- [ ] Each module has a single responsibility
- [ ] Shared fixtures extracted to conftest.py
- [ ] All original test coverage preserved""",
        "DELETE": """- [ ] Redundant tests removed
- [ ] Essential coverage preserved in canonical location
- [ ] No test duplication remains""",
        "KEEP": """- [ ] Weak assertions strengthened
- [ ] Redundant tests consolidated
- [ ] Test intent clearly expressed in assertions""",
    }
    acceptance = ac_map.get(audit.verdict, "- [ ] Issues from audit addressed")

    return f"""# Test Bug Report: {title}

## Summary

- {audit.summary}

## Severity

- Severity: {severity}
- Priority: {priority}
- Verdict: **{audit.verdict}**

## Reporter

- Name or handle: Test Audit
- Date: {today}
- Audit file: docs/test_audit/{audit.filename}

## Test File

- **File:** `{audit.test_file}`
- **Lines:** {audit.lines}
- **Test count:** {audit.test_count}

## Findings

{findings_text}

## Verdict Detail

{audit.verdict_detail}

## Proposed Fix

{acceptance}

## Tests

- Run after fix: `.venv/bin/python -m pytest {audit.test_file} -v`

## Notes

- Source audit: `docs/test_audit/{audit.filename}`
"""


def slugify(text: str, max_len: int = 50) -> str:
    """Convert text to filename-safe slug."""
    # Lowercase and replace spaces with hyphens
    slug = text.lower().replace(" ", "-")
    # Remove non-alphanumeric except hyphens
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)
    # Truncate
    return slug[:max_len].rstrip("-")


def main() -> None:
    audit_dir = Path("docs/test_audit")
    output_dir = Path("docs/test_bugs/open")

    # Ensure output directories exist
    for subdir in [
        "cli",
        "contracts",
        "core-checkpoint",
        "core-landscape",
        "core-security",
        "core-config",
        "engine",
        "audit",
        "plugins",
        "misc",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Process all audit files
    total = 0
    processed = 0
    by_verdict: dict[str, int] = {}
    by_category: dict[str, int] = {}

    for audit_path in sorted(audit_dir.glob("*.audit.md")):
        total += 1

        audit = parse_audit_file(audit_path)
        if audit is None:
            continue

        processed += 1
        by_verdict[audit.verdict] = by_verdict.get(audit.verdict, 0) + 1
        by_category[audit.category] = by_category.get(audit.category, 0) + 1

        # Generate ticket
        ticket_content = generate_ticket(audit)

        # Generate filename
        priority = verdict_to_priority(audit.verdict, audit.findings)
        today = datetime.now(UTC).date().isoformat()
        short_name = audit.test_file.split("/")[-1].replace(".py", "").replace("test_", "")
        slug = slugify(short_name)
        filename = f"{priority}-{today}-{slug}.md"

        # Write ticket
        ticket_path = output_dir / audit.category / filename
        ticket_path.write_text(ticket_content)
        print(f"Created: {ticket_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Test Bug Generation Summary")
    print(f"{'=' * 60}")
    print(f"Total audit files: {total}")
    print(f"Files with issues: {processed}")
    print("\nBy verdict:")
    for v, count in sorted(by_verdict.items(), key=lambda x: -x[1]):
        print(f"  {v}: {count}")
    print("\nBy category:")
    for c, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {c}: {count}")


if __name__ == "__main__":
    main()
