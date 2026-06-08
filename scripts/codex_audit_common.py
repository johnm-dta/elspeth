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
import hashlib
import inspect
import json
import re
import subprocess
import sys
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from tqdm.asyncio import tqdm as async_tqdm  # type: ignore[import-untyped]

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class AsyncTqdm:
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
USAGE_STAT_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
CODEX_RATE_LIMIT_PERIOD_S = 60.0

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

# Project instruction documents and skills that encode project-specific local rules
# (tier model, engine patterns, etc.). These are auto-loaded into agent context
# so static analysis prompts have the full authoritative rule set, not just the
# CLAUDE.md summary. Prefer Codex-native AGENTS.md and `.agents/skills`, while
# retaining CLAUDE.md / `.claude/skills` as migration fallbacks.
PROJECT_CONTEXT_FILES = ["AGENTS.md", "CLAUDE.md"]
SKILL_NAMES = [
    "tier-model-deep-dive",
    "engine-patterns-reference",
    "config-contracts-guide",
    "logging-telemetry-policy",
]
SKILL_ROOTS = [".agents/skills", ".claude/skills"]

REPORT_METADATA_FILENAMES = frozenset({"RUN_METADATA.md", "SUMMARY.md", "FINDINGS_INDEX.md"})
PRIORITY_COPY_DIR = "by-priority"


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


def get_codex_version(repo_root: Path) -> str:
    """Get the installed Codex CLI version, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["codex", "--version"],
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


def sha256_text(value: str) -> str:
    """Return a stable SHA-256 digest for run metadata."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _empty_usage_summary() -> dict[str, int]:
    return dict.fromkeys(USAGE_STAT_KEYS, 0)


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _usage_from_event(event: object) -> dict[str, int]:
    """Extract token usage from a Codex JSONL event when present."""
    summary = _empty_usage_summary()
    if not isinstance(event, dict):
        return summary

    raw_usage = event.get("usage")
    if not isinstance(raw_usage, dict):
        msg = event.get("msg")
        if isinstance(msg, dict):
            raw_usage = msg.get("usage")
    if not isinstance(raw_usage, dict):
        return summary

    prompt_details = raw_usage.get("prompt_tokens_details")
    cached_from_details = 0
    if isinstance(prompt_details, dict):
        cached_from_details = _safe_int(prompt_details.get("cached_tokens"))

    input_tokens = _safe_int(raw_usage.get("input_tokens")) or _safe_int(raw_usage.get("prompt_tokens"))
    output_tokens = _safe_int(raw_usage.get("output_tokens")) or _safe_int(raw_usage.get("completion_tokens"))
    total_tokens = _safe_int(raw_usage.get("total_tokens")) or input_tokens + output_tokens
    cached_input_tokens = _safe_int(raw_usage.get("cached_input_tokens")) or cached_from_details

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _parse_codex_jsonl_usage(stdout: bytes) -> dict[str, int]:
    """Parse Codex --json stdout and aggregate usage from completed turns."""
    summary = _empty_usage_summary()
    stdout_text = stdout.decode("utf-8", errors="replace")
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        event_usage = _usage_from_event(event)
        for key in USAGE_STAT_KEYS:
            summary[key] += event_usage[key]
    return summary


def _write_usage_summary(output_path: Path, usage: Mapping[str, int]) -> None:
    usage_path = output_path.with_suffix(output_path.suffix + ".usage.json")
    payload = {key: int(usage.get(key, 0)) for key in USAGE_STAT_KEYS}
    usage_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def structured_output_path_for_report(output_path: Path) -> Path:
    """Return the structured Codex sidecar path for a Markdown report path."""
    return output_path.with_suffix(output_path.suffix + ".structured.json")


class AsyncRequestRateLimiter:
    """Async sliding-window rate limiter for Codex subprocess starts."""

    def __init__(
        self,
        *,
        max_calls: int,
        period_s: float = CODEX_RATE_LIMIT_PERIOD_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if period_s <= 0:
            raise ValueError("period_s must be > 0")

        self._max_calls = max_calls
        self._period_s = period_s
        self._clock = clock
        self._sleep = sleep
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available, then reserve it."""
        while True:
            async with self._lock:
                now = self._clock()
                self._discard_expired(now)
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return

                wait_s = self._period_s - (now - self._timestamps[0])

            await self._sleep(max(wait_s, 0.0))

    def _discard_expired(self, now: float) -> None:
        while self._timestamps and now - self._timestamps[0] >= self._period_s:
            self._timestamps.popleft()


def make_codex_rate_limiter(rate_limit: int | None) -> AsyncRequestRateLimiter | None:
    """Build the async Codex subprocess rate limiter from a per-minute limit."""
    if rate_limit is None:
        return None
    return AsyncRequestRateLimiter(max_calls=rate_limit)


async def _await_rate_limiter(rate_limiter: Any) -> None:
    """Acquire a rate-limit slot, preferring the shared async wrapper API."""
    acquire = getattr(rate_limiter, "acquire", None)
    if acquire is not None:
        acquired = acquire()
        if inspect.isawaitable(acquired):
            await acquired
        return

    acquire_async = getattr(rate_limiter, "try_acquire_async", None)
    if acquire_async is not None:
        await acquire_async("codex_api")
        return

    acquired = rate_limiter.try_acquire("codex_api")
    if asyncio.iscoroutine(acquired):
        await acquired


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


def _extract_structured_markdown(raw_output_path: Path, output_path: Path, field: str) -> None:
    """Extract a Markdown report field from Codex structured JSON output."""
    try:
        raw_data = json.loads(raw_output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Codex structured output was not valid JSON: {raw_output_path}") from exc

    if not isinstance(raw_data, dict):
        raise RuntimeError(f"Codex structured output must be a JSON object: {raw_output_path}")

    markdown = raw_data.get(field)
    if not isinstance(markdown, str) or not markdown.strip():
        raise RuntimeError(f"Codex structured output missing non-empty string field: {field}")

    output_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")


def _apply_structured_evidence_gate(
    output_path: Path,
    downgrades: list[tuple[int, str, str]],
) -> None:
    """Keep structured finding priorities aligned with Markdown evidence gating."""
    if not downgrades:
        return

    sidecar = structured_output_path_for_report(output_path)
    if not sidecar.exists():
        return

    try:
        raw_data = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Structured finding sidecar is not valid JSON: {sidecar}") from exc

    if not isinstance(raw_data, dict):
        raise RuntimeError(f"Structured finding sidecar must contain a JSON object: {sidecar}")

    findings = raw_data.get("findings")
    if not isinstance(findings, list):
        raise RuntimeError(f"Structured finding sidecar missing findings array: {sidecar}")

    for report_index, severity, priority in downgrades:
        if report_index >= len(findings):
            continue
        finding = findings[report_index]
        if not isinstance(finding, dict):
            raise RuntimeError(f"Structured finding {report_index} must be an object: {sidecar}")
        finding["severity"] = severity
        finding["priority"] = priority

    sidecar.write_text(json.dumps(raw_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# === Evidence Processing Functions ===


def load_context(
    repo_root: Path,
    extra_files: list[str] | None = None,
    *,
    include_skills: bool = False,
) -> str:
    """Load project instructions, skills, and optional extra files for agent prompts.

    Returns concatenated content separated by headers showing filename.

    Args:
        repo_root: Repository root directory.
        extra_files: Additional context files (relative to repo root).
        include_skills: If True, auto-load project skill files (tier model,
            engine patterns, config contracts, logging/telemetry policy).
    """
    parts = []

    for filename in PROJECT_CONTEXT_FILES:
        path = repo_root / filename
        if path.exists():
            parts.append(f"--- {filename} ---\n{path.read_text(encoding='utf-8')}")

    if include_skills:
        for skill_name in SKILL_NAMES:
            for skill_root in SKILL_ROOTS:
                full_path = repo_root / skill_root / skill_name / "SKILL.md"
                if full_path.exists():
                    parts.append(f"--- Skill: {skill_name} ---\n{full_path.read_text(encoding='utf-8')}")
                    break

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
    structured_downgrades: list[tuple[int, str, str]] = []
    for report in reports:
        if not report.strip():
            continue

        evidence = extract_section(report, "Evidence")
        quality = evidence_quality_score(evidence)

        if quality == 0:
            gated_count += 1
            structured_downgrades.append((len(new_reports), "trivial", "P3"))
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
                structured_downgrades.append((len(new_reports), "minor", "P2"))

        new_reports.append(report.strip())

    new_text = "\n---\n".join(new_reports).rstrip() + "\n"
    if new_text != text:
        output_path.write_text(new_text, encoding="utf-8")
    _apply_structured_evidence_gate(output_path, structured_downgrades)

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
    output_schema: Path | None = None,
    structured_markdown_field: str | None = None,
    profile: str | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    ephemeral: bool = False,
    timeout_s: float | None = 1800.0,
) -> dict[str, int]:
    """Run codex exec once. Raises on failure. Does not handle retries or logging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codex_output_path = output_path
    if output_schema is not None:
        codex_output_path = structured_output_path_for_report(output_path)

    cmd = [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "-c",
        'approval_policy="never"',
        "--json",
        "--cd",
        str(repo_root),
        "--output-last-message",
        str(codex_output_path),
    ]
    if profile is not None:
        cmd.extend(["--profile", profile])
    if reasoning_effort is not None:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    if service_tier is not None:
        cmd.extend(["-c", f'service_tier="{service_tier}"'])
    if ephemeral:
        cmd.append("--ephemeral")
    if model is not None:
        cmd.extend(["--model", model])
    if output_schema is not None:
        cmd.extend(["--output-schema", str(output_schema)])
    cmd.append("-")

    # Create subprocess with stdout/stderr capture for diagnostics
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=repo_root,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        communicate = process.communicate(input=prompt.encode("utf-8"))
        if timeout_s is None:
            stdout, stderr = await communicate
        else:
            stdout, stderr = await asyncio.wait_for(communicate, timeout=timeout_s)

        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")
            # Truncate stderr for log readability
            if len(stderr_text) > STDERR_TRUNCATE_CHARS:
                stderr_text = stderr_text[:STDERR_TRUNCATE_CHARS] + "... (truncated)"

            raise RuntimeError(f"codex exec failed for {file_display} with code {process.returncode}\nstderr: {stderr_text}")

        if output_schema is not None:
            _extract_structured_markdown(
                codex_output_path,
                output_path,
                structured_markdown_field or "markdown_report",
            )

        usage = _parse_codex_jsonl_usage(stdout)
        _write_usage_summary(output_path, usage)
        return usage

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
    rate_limiter: Any | None,
    evidence_gate_summary_prefix: str = "",
    output_schema: Path | None = None,
    structured_markdown_field: str | None = None,
    profile: str | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    ephemeral: bool = False,
    timeout_s: float | None = 1800.0,
) -> dict[str, int]:
    """Run codex with retry logic, rate limiting, evidence gate, and logging. Returns stats."""
    start_time = time.monotonic()
    status = "ok"
    note = ""
    gated_count = 0
    usage = _empty_usage_summary()

    async def rate_limited_run_codex_once(**kwargs: Any) -> dict[str, int]:
        if rate_limiter:
            await _await_rate_limiter(rate_limiter)
        return await run_codex_once(**kwargs)

    # Create retry decorator dynamically
    retry_decorator = retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_MIN_WAIT_S, max=RETRY_MAX_WAIT_S),
        retry=retry_if_exception_type((RuntimeError, asyncio.TimeoutError)),
        reraise=True,
    )

    try:
        # Apply retry decorator to the function call
        usage = await retry_decorator(rate_limited_run_codex_once)(
            file_path=file_path,
            output_path=output_path,
            model=model,
            prompt=prompt,
            repo_root=repo_root,
            file_display=file_display,
            output_display=output_display,
            output_schema=output_schema,
            structured_markdown_field=structured_markdown_field,
            profile=profile,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            ephemeral=ephemeral,
            timeout_s=timeout_s,
        )

        # Apply evidence gate after successful completion
        gated_count = apply_evidence_gate(output_path, summary_prefix=evidence_gate_summary_prefix)
        if gated_count > 0:
            note = f"evidence_gate={gated_count}"
        usage_note = ", ".join(f"{key}={usage[key]}" for key in USAGE_STAT_KEYS if usage[key])
        if usage_note:
            note = f"{note}; {usage_note}" if note else usage_note

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
            model=model or "default (from codex config)",
            duration_s=duration_s,
            note=note,
        )

    return {"gated": gated_count, **usage}


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


def iter_report_files(output_dir: Path) -> list[Path]:
    """Return primary Markdown report files, excluding generated indexes/copies."""
    files: list[Path] = []
    for md_file in sorted(output_dir.rglob("*.md")):
        relative_parts = md_file.relative_to(output_dir).parts
        if PRIORITY_COPY_DIR in relative_parts:
            continue
        if md_file.name in REPORT_METADATA_FILENAMES:
            continue
        files.append(md_file)
    return files


def _source_file_from_report(output_dir: Path, report_path: Path) -> str:
    return str(report_path.relative_to(output_dir).with_suffix(""))


def _structured_findings(report_path: Path) -> list[dict[str, Any]] | None:
    sidecar = structured_output_path_for_report(report_path)
    if not sidecar.exists():
        return None
    # Fail closed against stale sidecars: a structured sidecar is authoritative
    # only when it is at least as new as its Markdown report. A rerun with
    # --no-structured-output, a manual rewrite, or a partial cleanup can leave an
    # old sidecar beside fresh Markdown — trusting it would count findings the
    # current report no longer makes (false positive) or, worse, mask a real
    # finding behind an empty sidecar (false negative). When the sidecar predates
    # the report, ignore it and let the caller parse the authoritative Markdown.
    if report_path.exists() and sidecar.stat().st_mtime < report_path.stat().st_mtime:
        print(
            f"⚠️  Ignoring stale structured sidecar (older than its report): {sidecar}",
            file=sys.stderr,
        )
        return None
    try:
        raw_data = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Structured finding sidecar is not valid JSON: {sidecar}") from exc
    if not isinstance(raw_data, dict):
        raise RuntimeError(f"Structured finding sidecar must contain a JSON object: {sidecar}")
    findings = raw_data.get("findings")
    if not isinstance(findings, list):
        raise RuntimeError(f"Structured finding sidecar missing findings array: {sidecar}")

    validated: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise RuntimeError(f"Structured finding {index} must be an object: {sidecar}")
        validated.append(finding)
    return validated


def _priority_from_structured_finding(finding: Mapping[str, Any]) -> str:
    priority = finding.get("priority")
    if isinstance(priority, str) and priority in {"P0", "P1", "P2", "P3"}:
        return priority
    return "unknown"


def _summary_from_structured_finding(finding: Mapping[str, Any]) -> str:
    summary = finding.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()[:100]
    return "No summary found"


def _confidence_from_structured_finding(finding: Mapping[str, Any]) -> str:
    confidence = finding.get("confidence")
    if isinstance(confidence, str) and confidence.strip():
        return confidence.strip()
    return "unknown"


def _evidence_from_structured_finding(finding: Mapping[str, Any]) -> str:
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        return ""
    entries: list[str] = []
    for entry in evidence[:3]:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        line = entry.get("line")
        claim = entry.get("claim")
        if not isinstance(path, str) or not path:
            continue
        location = path
        if isinstance(line, int):
            location = f"{path}:{line}"
        if isinstance(claim, str) and claim:
            entries.append(f"{location} {claim}")
        else:
            entries.append(location)
    return "; ".join(entries)


def exit_code_from_stats(stats: Mapping[str, int]) -> int:
    """Return a non-zero exit code when any per-target analysis failed.

    Fail closed: when files/entries raised during analysis the scan is partial.
    A partial scan must not present to CI or operators as a successful, complete
    run — missing reports would otherwise hide behind a green process exit. Each
    runner's ``main`` returns this so a partial run exits non-zero.
    """
    return 1 if stats.get("failed", 0) else 0


def generate_summary(output_dir: Path, *, no_defect_marker: str) -> dict[str, int]:
    """Parse all outputs and count defects by priority.

    Args:
        output_dir: Directory containing markdown reports
        no_defect_marker: String that indicates a clean report (e.g., "No test defect found")

    Returns:
        Dictionary with counts by priority
    """
    stats: Counter[str] = Counter()

    for md_file in iter_report_files(output_dir):
        structured = _structured_findings(md_file)
        if structured is not None:
            if not structured:
                stats["no_defect"] += 1
                continue
            for finding in structured:
                stats[_priority_from_structured_finding(finding)] += 1
            continue

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
    extra_parameters: Mapping[str, str] | None = None,
) -> None:
    """Write run metadata file for triage tracking."""
    metadata_path = output_dir / "RUN_METADATA.md"
    extra_lines = ""
    if extra_parameters:
        extra_lines = "\n".join(f"- **{key}:** {value}" for key, value in extra_parameters.items())
        extra_lines = f"\n{extra_lines}"

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
{extra_lines}

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

    non_finding_keys = {"no_defect", "gated", "unknown", "merged", "failed", *USAGE_STAT_KEYS}
    total_defects = sum(v for k, v in stats.items() if k not in non_finding_keys)
    no_defect_count = stats.get("no_defect", 0)
    gated = stats.get("gated", 0)
    unknown = stats.get("unknown", 0)
    input_tokens = stats.get("input_tokens", 0)
    cached_input_tokens = stats.get("cached_input_tokens", 0)
    output_tokens = stats.get("output_tokens", 0)
    cache_hit_rate = (cached_input_tokens / input_tokens * 100) if input_tokens else 0.0

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
| **Input Tokens** | {input_tokens} |
| **Cached Input Tokens** | {cached_input_tokens} ({cache_hit_rate:.1f}%) |
| **Output Tokens** | {output_tokens} |

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

    for md_file in iter_report_files(output_dir):
        text = md_file.read_text(encoding="utf-8")
        report_file = str(md_file.relative_to(output_dir))
        fallback_source_file = _source_file_from_report(output_dir, md_file)
        structured = _structured_findings(md_file)

        if structured is not None:
            if not structured:
                findings.append(
                    {
                        "file": report_file,
                        "source_file": fallback_source_file,
                        "priority": "clean",
                        "summary": "No concrete bug found",
                        "status": "pending",
                        "confidence": "",
                        "evidence": "",
                    }
                )
                continue
            for finding in structured:
                target_file = finding.get("target_file")
                source_file = target_file if isinstance(target_file, str) and target_file else fallback_source_file
                findings.append(
                    {
                        "file": report_file,
                        "source_file": source_file,
                        "priority": _priority_from_structured_finding(finding),
                        "summary": _summary_from_structured_finding(finding),
                        "status": "pending",
                        "confidence": _confidence_from_structured_finding(finding),
                        "evidence": _evidence_from_structured_finding(finding),
                    }
                )
            continue

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
                "source_file": fallback_source_file,
                "priority": priority if not is_clean else "clean",
                "summary": summary,
                "status": "pending",  # For manual triage updates
                "confidence": "",
                "evidence": "",
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

| Priority | {file_column_label} | Summary | Confidence | Evidence | Status | Report |
|----------|{"---" * (len(file_column_label) // 3)}|---------|------------|----------|--------|--------|
"""

    for finding in findings:
        # Skip "clean" reports for triage focus
        if finding["priority"] == "clean":
            continue

        content += (
            f"| {finding['priority']} | `{finding['source_file']}` | "
            f"{finding['summary']} | {finding['confidence']} | {escape_cell(finding['evidence'])} | "
            f"{finding['status']} | [{finding['file']}]({finding['file']}) |\n"
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

    non_finding_keys = {"no_defect", "gated", "unknown", "merged", "failed", *USAGE_STAT_KEYS}
    total_defects = sum(v for k, v in stats.items() if k not in non_finding_keys)

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

    input_tokens = stats.get("input_tokens", 0)
    cached_input_tokens = stats.get("cached_input_tokens", 0)
    output_tokens = stats.get("output_tokens", 0)
    if input_tokens or output_tokens:
        cache_hit_rate = (cached_input_tokens / input_tokens * 100) if input_tokens else 0.0
        print(f"\nToken usage: input={input_tokens}, cached_input={cached_input_tokens} ({cache_hit_rate:.1f}%), output={output_tokens}")

    print("=" * 60 + "\n")
