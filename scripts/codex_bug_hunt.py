#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


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
        "Repository context (read-only):\n"
        f"{context}\n\n"
        "Bug report template:\n"
        f"{template}\n"
    )


async def _run_codex(
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
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    status = "ok"
    note = ""
    gated_count = 0
    cmd = [
        "codex",
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "never",
    ]
    if model is not None:
        cmd.extend(["--model", model])
    cmd.extend(
        [
            "exec",
            prompt,
            "--output-last-message",
            str(output_path),
        ]
    )

    try:
        process = await asyncio.create_subprocess_exec(*cmd, cwd=repo_root)
        return_code = await process.wait()
        if return_code != 0:
            raise RuntimeError(f"codex exec failed for {file_path} with code {return_code}")
        gated_count = _apply_evidence_gate(output_path)
        if gated_count > 0:
            note = f"evidence_gate={gated_count}"
    except Exception as exc:
        status = "failed"
        note = str(exc)
        raise
    finally:
        duration_s = time.monotonic() - start_time
        await _append_log(
            log_path=log_path,
            log_lock=log_lock,
            timestamp=_utc_now(),
            status=status,
            file_display=file_display,
            output_display=output_display,
            model=model or "",
            duration_s=duration_s,
            note=note,
        )


def _chunked(paths: list[Path], size: int) -> list[list[Path]]:
    return [paths[i : i + size] for i in range(0, len(paths), size)]


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
) -> None:
    log_lock = asyncio.Lock()
    errors: list[Exception] = []
    for batch in _chunked(files, batch_size):
        tasks: list[asyncio.Task[None]] = []
        for file_path in batch:
            relative = file_path.relative_to(root_dir)
            output_path = output_dir / relative
            output_path = output_path.with_suffix(output_path.suffix + ".md")
            if skip_existing and output_path.exists():
                continue
            prompt = _build_prompt(file_path, prompt_template, context)
            tasks.append(
                asyncio.create_task(
                    _run_codex(
                        file_path=file_path,
                        output_path=output_path,
                        model=model,
                        prompt=prompt,
                        repo_root=repo_root,
                        log_path=log_path,
                        log_lock=log_lock,
                        file_display=str(file_path.relative_to(repo_root).as_posix()),
                        output_display=str(output_path.relative_to(repo_root).as_posix()),
                    )
                )
            )
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    errors.append(result)
    if errors:
        raise RuntimeError(f"{len(errors)} codex runs failed; first error: {errors[0]}")


def _ensure_log_file(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        return
    header = (
        "# Codex Bug Hunt Log\n\n"
        "| Timestamp (UTC) | Status | File | Output | Model | Duration_s | Note |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    log_path.write_text(header, encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "\\n").replace("\r", "")


def _load_context(repo_root: Path) -> str:
    claude_path = repo_root / "CLAUDE.md"
    architecture_path = repo_root / "ARCHITECTURE.md"
    claude_text = claude_path.read_text(encoding="utf-8")
    architecture_text = architecture_path.read_text(encoding="utf-8")
    return f"--- CLAUDE.md ---\n{claude_text}\n--- ARCHITECTURE.md ---\n{architecture_text}"


def _extract_section(report: str, heading: str) -> str:
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


def _replace_section(report: str, heading: str, new_lines: list[str]) -> str:
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


def _has_file_line_evidence(evidence: str) -> bool:
    patterns = [
        r"\b[\w./-]+\.[\w]+:\d+\b",
        r"\b[\w./-]+#L\d+\b",
        r"\bline\s+\d+\b",
    ]
    return any(re.search(pattern, evidence) for pattern in patterns)


def _apply_evidence_gate(output_path: Path) -> int:
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
        evidence = _extract_section(report, "Evidence")
        if not _has_file_line_evidence(evidence):
            gated_count += 1
            report = _replace_section(
                report,
                "Summary",
                ["", "- Needs verification: missing file/line evidence."],
            )
            report = _replace_section(
                report,
                "Severity",
                ["", "- Severity: trivial", "- Priority: P3"],
            )
            report = _replace_section(
                report,
                "Root Cause Hypothesis",
                ["", "- Unverified; no file/line evidence provided."],
            )
            report = _replace_section(
                report,
                "Evidence",
                ["", "- No file/line evidence provided."],
            )
        new_reports.append(report.strip())

    new_text = "\n---\n".join(new_reports).rstrip() + "\n"
    if new_text != text:
        output_path.write_text(new_text, encoding="utf-8")

    return gated_count


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
            selected.extend([p for p in path.rglob("*") if p.is_file()])
        else:
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
        if path.is_file() and _is_under_root(path, root_dir):
            selected.append(path)
    return selected


def _list_files(
    *,
    root_dir: Path,
    repo_root: Path,
    changed_since: str | None,
    paths_from: Path | None,
) -> list[Path]:
    selected: set[Path] | None = None

    if changed_since:
        changed = set(_changed_files_since(repo_root, root_dir, changed_since))
        selected = changed if selected is None else selected & changed

    if paths_from:
        listed = set(_paths_from_file(paths_from, repo_root, root_dir))
        selected = listed if selected is None else selected & listed

    if selected is None:
        selected = {path for path in root_dir.rglob("*") if path.is_file()}

    return sorted(selected)


async def _append_log(
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
    line = (
        f"| {_escape_cell(timestamp)} | {_escape_cell(status)} | "
        f"{_escape_cell(file_display)} | {_escape_cell(output_display)} | "
        f"{_escape_cell(model)} | {duration_s:.2f} | {_escape_cell(note)} |\n"
    )
    async with log_lock:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Codex bug audits per file in batches.")
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
        "--paths-from",
        default=None,
        help="Path to a file containing newline-separated paths to scan.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have an output report.",
    )

    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if shutil.which("codex") is None:
        raise RuntimeError("codex CLI not found on PATH")

    repo_root = Path(__file__).resolve().parents[1]
    root_dir = _resolve_path(repo_root, args.root)
    template_path = _resolve_path(repo_root, args.template)
    output_dir = _resolve_path(repo_root, args.output_dir)
    log_path = _resolve_path(repo_root, "docs/bugs/process/CODEX_LOG.md")

    template_text = template_path.read_text(encoding="utf-8")
    context_text = _load_context(repo_root)
    _ensure_log_file(log_path)

    paths_from = _resolve_path(repo_root, args.paths_from) if args.paths_from else None
    files = _list_files(
        root_dir=root_dir,
        repo_root=repo_root,
        changed_since=args.changed_since,
        paths_from=paths_from,
    )
    if not files:
        raise RuntimeError(f"No files found under {root_dir}")

    asyncio.run(
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
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
