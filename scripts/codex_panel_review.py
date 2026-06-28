#!/usr/bin/env python3
"""Pre-1.0 sandblasting SME review fleet (foundation: single-file, serial lenses).

See docs/superpowers/specs/2026-06-28-codex-panel-review-design.md.

Helpers from ``codex_audit_common`` are bound at module scope (required so the
runner test can monkeypatch ``run_codex_once``) via a dual-import shim, so this
module runs both as a script (``scripts/`` on ``sys.path``) and as the package
``scripts.codex_panel_review`` under pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Dual-import shim: bare import works when run as a script (scripts/ on sys.path);
# the package form works when pytest imports this as scripts.codex_panel_review.
# All helpers are bound at module scope (every name is used below) — also required
# for the runner test's monkeypatch of run_codex_once. See
# docs/superpowers/plans/2026-06-28-codex-panel-review-foundation.md.
try:
    from codex_audit_common import (  # type: ignore[import-not-found]
        append_log,
        ensure_log_file,
        load_context,
        make_codex_rate_limiter,
        run_codex_once,
        structured_output_path_for_report,
        utc_now,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.codex_audit_common import (
        append_log,
        ensure_log_file,
        load_context,
        make_codex_rate_limiter,
        run_codex_once,
        structured_output_path_for_report,
        utc_now,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
LENSES_DIR = Path(__file__).resolve().parent / "codex_lenses"
PANEL_FINDING_SCHEMA = LENSES_DIR / "panel_finding.schema.json"
# Per-lens output goes in the *-raw sibling, NOT the counted `findings-panel/`
# tree. iter_report_files (codex_audit_common.py:792) only skips `by-priority/`
# + the metadata filenames — it does NOT skip an arbitrary subdir — so per-lens
# `.md` files written under the counted tree would be double-counted by Plan 2's
# generate_summary. Keeping raw detail in its own sibling from day one matches
# the spec's output layout and avoids seeding that double-count.
DEFAULT_OUTPUT_DIR = "docs/quality-audit/findings-panel-raw"

STRICT_CATEGORIES = frozenset({"bug", "correctness", "security", "smell"})
RELAXED_CATEGORIES = frozenset({"improvement", "efficiency"})
ANCHOR_REQUIRED_CATEGORIES = STRICT_CATEGORIES | frozenset({"easy-win"})


def build_layered_prompt(
    *,
    context: str,
    file_source: str,
    file_path: str,
    persona: str,
    lens: str,
) -> str:
    """Layer the prompt most-shared -> least-shared to ENABLE the codex prompt
    cache to key on [context][source] across every lens of a file (necessary, not
    sufficient — actual cross-process reuse is verified empirically in Task 9). The
    per-call tail (file_path, lens) is the only variable content and comes last."""
    return (
        f"{context}\n\n"
        "=== TARGET FILE SOURCE (inlined; you MAY read any other repo file via the "
        "read-only sandbox for investigation) ===\n"
        f"```\n{file_source}\n```\n\n"
        "=== REVIEW LENS ===\n"
        f"{persona}\n\n"
        "Output: a JSON object matching the provided schema — a non-empty "
        "`markdown_report` narration string and a `findings` array. Every finding "
        f'MUST set `lens` to "{lens}" and `priority` to one of P0|P1|P2|P3. For '
        "categories bug/correctness/security/smell/easy-win, `evidence` MUST cite a "
        "real path and integer line. The schema is STRICT: every finding MUST "
        "include ALL fields — for any field you have no value for (confidence, "
        "effort, impact, suggested_fix, target_file, gate_note, or an evidence "
        "`line`), emit JSON `null`; do NOT omit the key. If you find nothing, "
        "return an empty findings array and say so in markdown_report.\n"
        f"--- review target: {file_path} · lens: {lens} ---\n"
    )


# Plan 1 roster (file-predicate pairs). Full routing table is Plan 2.
_EVERY_FILE = ("solution-architect", "security-architect")


def load_persona(lens: str, *, lenses_dir: Path = LENSES_DIR) -> str:
    path = lenses_dir / f"{lens}.md"
    if not path.exists():
        raise FileNotFoundError(f"persona prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def route_lenses(file_path: Path, *, override: list[str] | None = None) -> list[str]:
    if override:
        return list(override)
    lenses = list(_EVERY_FILE)
    if file_path.suffix == ".py" and (LENSES_DIR / "python-engineer.md").exists():
        lenses.append("python-engineer")
    return lenses


def _has_line_anchor(finding: dict[str, Any]) -> bool:
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        return False
    return any(isinstance(e, dict) and isinstance(e.get("line"), int) and bool(e.get("path")) for e in evidence)


def _has_impact(finding: dict[str, Any]) -> bool:
    # `impact` is nullable in the strict schema, so finding.get("impact") may be
    # None. Coerce with `or ""` BEFORE strip() — str(None) == "None" reads truthy,
    # which would silently defeat the relaxed/fail-closed downgrade.
    return bool(str(finding.get("impact") or "").strip())


def apply_panel_evidence_gate(sidecar_path: Path, *, lens: str) -> int:
    """Category-aware, fail-CLOSED structured gate. Downgrades anchor-required
    findings lacking a path+line, and any other priority-bearing finding lacking
    an `impact` rationale (relaxed AND uncovered categories like `design` or a
    typo'd/future enum value — nothing rides through ungated). Stamps each
    finding's `lens` from the known-correct caller value (never trusts the model's
    self-reported lens). Always rewrites the sidecar so its mtime exceeds the
    per-lens .md (defeats the staleness guard at codex_audit_common.py:820).
    Returns the downgrade count."""
    raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"panel sidecar must be a JSON object: {sidecar_path}")
    findings = raw.get("findings")
    if not isinstance(findings, list):
        raise RuntimeError(f"panel sidecar missing findings array: {sidecar_path}")

    downgraded = 0
    for finding in findings:
        if not isinstance(finding, dict):
            raise RuntimeError(f"panel finding must be an object: {sidecar_path}")
        finding["lens"] = lens  # deterministic stamp; do not trust the model's value
        category = finding.get("category")
        if category in ANCHOR_REQUIRED_CATEGORIES and not _has_line_anchor(finding):
            finding["priority"] = "P3"
            finding["confidence"] = "low"
            finding["gate_note"] = "needs verification: no file:line anchor"
            downgraded += 1
        elif category in RELAXED_CATEGORIES and not _has_impact(finding):
            finding["priority"] = "P3"
            finding["gate_note"] = "needs rationale: empty impact"
            downgraded += 1
        elif category not in ANCHOR_REQUIRED_CATEGORIES and category not in RELAXED_CATEGORIES and not _has_impact(finding):
            # Fail-closed: `design` and any unknown/typo/future category need at
            # least an `impact` rationale, or they are downgraded — no silent
            # pass-through of an unsubstantiated high-priority claim.
            finding["priority"] = "P3"
            finding["gate_note"] = f"uncovered category {category!r}: needs impact rationale"
            downgraded += 1

    # Unconditional rewrite (last) — see docstring.
    sidecar_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return downgraded


async def run_file_lenses(
    *,
    file_path: Path,
    lenses: list[str],
    output_dir: Path,
    repo_root: Path,
    context: str,
    model: str | None,
    reasoning_effort: str | None,
    rate_limiter: Any | None,
    log_path: Path,
    log_lock: asyncio.Lock,
) -> dict[str, int]:
    """Run a file's lenses SERIALLY (file-major), so lens 2..N reuse lens 1's
    warm [context][source] prompt-cache prefix. Stock run_codex_once; the panel
    gate runs after each lens."""
    file_source = file_path.read_text(encoding="utf-8")
    relative = file_path.relative_to(repo_root)
    agg = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "gated": 0, "failed": 0}

    for lens in lenses:
        output_path = (output_dir / relative).with_suffix((output_dir / relative).suffix + f".{lens}.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        persona = load_persona(lens)
        prompt = build_layered_prompt(
            context=context,
            file_source=file_source,
            file_path=str(relative.as_posix()),
            persona=persona,
            lens=lens,
        )
        start = time.monotonic()
        status, note = "ok", ""
        try:
            if rate_limiter is not None:
                # public method on AsyncRequestRateLimiter (codex_audit_common.py:274);
                # avoids importing the private _await_rate_limiter, which has no
                # dual-import shim and would ModuleNotFoundError under pytest.
                await rate_limiter.acquire()
            usage = await run_codex_once(
                file_path=file_path,
                output_path=output_path,
                model=model,
                prompt=prompt,
                repo_root=repo_root,
                file_display=str(relative.as_posix()),
                output_display=str(output_path.relative_to(repo_root).as_posix())
                if output_path.is_relative_to(repo_root)
                else str(output_path),
                output_schema=PANEL_FINDING_SCHEMA,
                structured_markdown_field="markdown_report",
                reasoning_effort=reasoning_effort,
            )
            gated = apply_panel_evidence_gate(structured_output_path_for_report(output_path), lens=lens)
            agg["gated"] += gated
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
                agg[key] += int(usage.get(key, 0))
            note = f"gated={gated}; cached={usage.get('cached_input_tokens', 0)}"
        except Exception as exc:  # capture-and-continue so the pilot completes
            status, note = "failed", str(exc)[:200]
            agg["failed"] += 1
        finally:
            await append_log(
                log_path=log_path,
                log_lock=log_lock,
                timestamp=utc_now(),
                status=status,
                file_display=str(relative.as_posix()),
                output_display=f"lens={lens}",
                model=model or "default",
                duration_s=time.monotonic() - start,
                note=note,
            )
    return agg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Codex panel review (foundation: single file).")
    p.add_argument("--file", required=True, help="Source file to review.")
    p.add_argument("--lenses", default=None, help="Comma-separated lens override.")
    p.add_argument("--model", default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--context-files", nargs="+", default=None)
    p.add_argument("--rate-limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"file not found: {file_path}", file=sys.stderr)
        return 1
    if not file_path.is_relative_to(REPO_ROOT):
        # The codex sandbox is scoped to REPO_ROOT (--cd) and run_file_lenses takes
        # relative_to(repo_root); a target outside the repo would raise an uncaught
        # ValueError. Fail fast with a clean message instead.
        print(f"file is outside the repo ({REPO_ROOT}): {file_path}", file=sys.stderr)
        return 1
    override = [s.strip() for s in args.lenses.split(",")] if args.lenses else None
    lenses = route_lenses(file_path, override=override)
    missing = [lens for lens in lenses if not (LENSES_DIR / f"{lens}.md").exists()]
    if missing:
        # Fail fast BEFORE any (paid) codex call rather than crashing mid-run on a
        # --lenses typo — capture-and-continue would still spend on the valid lenses.
        available = ", ".join(sorted(p.stem for p in LENSES_DIR.glob("*.md")))
        print(f"unknown lens(es): {', '.join(missing)}; available: {available}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Would review {file_path.name} through {len(lenses)} lenses:")
        for lens in lenses:
            print(f"  - {lens}")
        return 0

    if shutil.which("codex") is None:
        raise RuntimeError("codex CLI not found on PATH")
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    log_path = output_dir / "PANEL_LOG.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_log_file(log_path, header_title="Codex Panel Review Log")
    context = load_context(REPO_ROOT, extra_files=args.context_files, include_skills=True)
    rate_limiter = make_codex_rate_limiter(args.rate_limit)

    stats = asyncio.run(
        run_file_lenses(
            file_path=file_path,
            lenses=lenses,
            output_dir=output_dir,
            repo_root=REPO_ROOT,
            context=context,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            rate_limiter=rate_limiter,
            log_path=log_path,
            log_lock=asyncio.Lock(),
        )
    )
    cached = stats["cached_input_tokens"]
    total_in = stats["input_tokens"]
    rate = (cached / total_in * 100) if total_in else 0.0
    print(f"lenses={len(lenses)} gated={stats['gated']} failed={stats['failed']} cache_hit={rate:.1f}% (cached_input={cached}/{total_in})")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
