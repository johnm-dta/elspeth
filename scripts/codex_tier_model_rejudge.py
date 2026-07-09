#!/usr/bin/env python3
"""Re-judge drifted / newly-uncovered tier-model findings via Codex, emit a sign manifest.

Reconciliation pipeline for a release branch whose feature work has drifted the
signed `enforce_tier_model` allowlist past the point where `sign-judge-signatures`
can auto-repair it (it refuses while NO_MATCHING entries exist). Three steps:

  1. build-worklist : diagnose + a decorator/per-file-rule-aware "stripped gate"
                      run -> the set of live findings that need a FRESH judge
                      verdict (moved symbols + genuinely new findings). Drift
                      that `sign-judge-signatures` repairs on its own is excluded,
                      as are R_TB_SUPPRESSED rows (decorator-covered = green) and
                      per_file_rule bookkeeping (handled out of band).
  2. rejudge        : drive `codex exec` (read-only sandbox) once per target ->
                      a per-finding ACCEPT/BLOCK verdict + a prescribed-form
                      rationale. Resumable (skips findings already reported),
                      rate-limited, retried, logged -- all via codex_audit_common.
  3. build-manifest : collect the ACCEPTed rationales -> manifest.yaml shaped for
                      `sign-judge-signatures --manifest`.

CUSTODY BOUNDARY -- this script signs NOTHING. It produces PROPOSED rationales
only, using the agent's OpenRouter/Codex access. The authoritative judge+sign is
the OPERATOR step, run in the cert shell with the HMAC key:

    # 1) drop the stale NO_MATCHING entries the worklist enumerated
    python scripts/codex_tier_model_rejudge.py emit-deletes --out <DIR>   # prints the keys
    #    (delete those keys from config/cicd/enforce_tier_model/*.yaml)
    # 2) auto-repair drift + sign the manifest entries (re-runs the real judge per entry)
    ELSPETH_JUDGE_METADATA_HMAC_KEY=... \
      sign-judge-signatures --root src/elspeth \
        --allowlist-dir config/cicd/enforce_tier_model \
        --owner john --manifest <DIR>/manifest.yaml --dry-run     # then drop --dry-run
    # 3) regenerate the fingerprint baseline (needs the green signed gate + the key)
    python scripts/cicd/regen_fingerprint_baseline.py

`sign-judge-signatures` re-judges every manifest entry itself and signs only the
ones the judge ACCEPTS; this script's verdict is a pre-screen to make that run
mostly-green and cheap, not a substitute for it.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import yaml
from codex_audit_common import (  # type: ignore[import-not-found]
    AsyncTqdm,
    chunked,
    ensure_log_file,
    extract_section,
    load_context,
    make_codex_rate_limiter,
    run_codex_with_retry_and_logging,
)

ALLOWLIST_DIR = "config/cicd/enforce_tier_model"
# Decorator-covered findings: green, never re-judged (see release-train playbook).
IGNORE_RULES = {"R_TB_SUPPRESSED"}
SHAPE_ONLY_ENV = {"ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE": "shape-only-when-key-missing"}
_JsonObject = dict[str, Any]
_JsonList = list[_JsonObject]
_ScopeSpan = tuple[int, int, list[str]]


# --------------------------------------------------------------------------- #
# worklist construction
# --------------------------------------------------------------------------- #
def _lints_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root / 'elspeth-lints' / 'src'}:{env.get('PYTHONPATH', '')}"
    env.update(SHAPE_ONLY_ENV)
    return env


def _run_diagnose(repo_root: Path, root: Path, allowlist: Path) -> _JsonObject:
    out = subprocess.run(
        [
            str(repo_root / ".venv/bin/diagnose-judge-signatures"),
            "--root",
            str(root),
            "--allowlist-dir",
            str(allowlist),
            "--format",
            "json",
        ],
        cwd=repo_root,
        env=_lints_env(repo_root),
        capture_output=True,
        text=True,
    )
    # diagnose exits non-zero when entries require action; stdout still holds the JSON.
    return cast(_JsonObject, json.loads(out.stdout))


def _strip_allowlist(src: Path, dst: Path, strip_keys: set[str]) -> int:
    """Byte-preserving removal of `- key:` blocks whose key is in strip_keys."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    key_re = re.compile(r"^- key: (.+?)\s*$")
    removed = 0
    for path in dst.glob("*.yaml"):
        lines = path.read_text().split("\n")
        out: list[str] = []
        i, n = 0, len(lines)
        while i < n:
            m = key_re.match(lines[i])
            if m and m.group(1) in strip_keys:
                i += 1
                while i < n and (lines[i].startswith(" ") or lines[i] == ""):
                    if lines[i] == "":
                        j = i + 1
                        while j < n and lines[j] == "":
                            j += 1
                        if j < n and (key_re.match(lines[j]) or not lines[j].startswith(" ")):
                            break
                    i += 1
                removed += 1
            else:
                out.append(lines[i])
                i += 1
        stripped_lines = "\n".join(out).splitlines()
        allowlist_headers = [
            line_index for line_index, line in enumerate(stripped_lines) if line.strip() in {"allow_hits:", "allow_hits: []"}
        ]
        if len(allowlist_headers) > 1:
            normalised_lines: list[str] = []
            for line_index, line in enumerate(stripped_lines):
                if line_index == allowlist_headers[0] and line.strip() == "allow_hits: []":
                    continue
                if line_index > 0 and line.strip() == "" and line_index - 1 == allowlist_headers[0]:
                    continue
                normalised_lines.append(line)
            stripped_lines = normalised_lines
        for line_index, line in enumerate(stripped_lines):
            if line.strip() != "allow_hits:":
                continue
            next_index = line_index + 1
            while next_index < len(stripped_lines) and stripped_lines[next_index].strip() == "":
                next_index += 1
            if next_index == len(stripped_lines) or not stripped_lines[next_index].startswith("- "):
                stripped_lines[line_index] = "allow_hits: []"
        path.write_text("\n".join(stripped_lines) + "\n")
    return removed


def _run_stripped_gate(repo_root: Path, root: Path, stripped_allowlist: Path) -> _JsonList:
    out = subprocess.run(
        [
            str(repo_root / ".venv/bin/python"),
            "-m",
            "elspeth_lints.core.cli",
            "check",
            "--rules",
            "trust_tier.tier_model",
            "--root",
            str(root),
            "--allowlist-dir",
            str(stripped_allowlist),
            "--format",
            "json",
        ],
        cwd=repo_root,
        env=_lints_env(repo_root),
        capture_output=True,
        text=True,
    )
    return cast(_JsonList, json.loads(out.stdout))


class _ScopeIndex:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, list[_ScopeSpan] | None] = {}

    def symbol(self, relpath: str, line: int) -> str:
        if relpath not in self._cache:
            spans: list[_ScopeSpan] = []
            try:
                tree = ast.parse((self.root / relpath).read_text())
            except (OSError, SyntaxError):
                self._cache[relpath] = None
            else:

                def walk(node: ast.AST, prefix: list[str]) -> None:
                    for ch in ast.iter_child_nodes(node):
                        if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            spans.append((ch.lineno, getattr(ch, "end_lineno", ch.lineno), [*prefix, ch.name]))
                            walk(ch, [*prefix, ch.name])
                        else:
                            walk(ch, prefix)

                walk(tree, [])
                self._cache[relpath] = spans
        cached_spans = self._cache[relpath]
        if cached_spans is None:
            return "_module_"
        best: list[str] | None = None
        for start, end, chain in cached_spans:
            if start <= line <= end and (best is None or len(chain) > len(best)):
                best = chain
        return ":".join(best) if best else "_module_"


def build_worklist(repo_root: Path, out_dir: Path) -> _JsonObject:
    root = repo_root / "src/elspeth"
    allowlist = repo_root / ALLOWLIST_DIR
    diag = _run_diagnose(repo_root, root, allowlist)

    drift_syms: set[tuple[str, str, str]] = set()
    strip_keys: set[str] = set()
    nomatch_keys: list[str] = []
    for entry in diag["entries"]:
        if not entry.get("requires_action"):
            continue
        strip_keys.add(entry["key"])
        parts = entry["key"].split(":")
        f, r, sym = parts[0], parts[1], ":".join(parts[2:-1])
        if entry["status"] in ("SCOPE_BINDING_DRIFT", "AST_PATH_BINDING_DRIFT"):
            drift_syms.add((f, r, sym))
        elif entry["status"] == "NO_MATCHING_FINDING":
            nomatch_keys.append(entry["key"])

    stripped_dir = out_dir / "_stripped_allowlist"
    _strip_allowlist(allowlist, stripped_dir, strip_keys)
    violations = _run_stripped_gate(repo_root, root, stripped_dir)

    scope = _ScopeIndex(root)
    targets: _JsonList = []
    auto_drift = pfr = ignored = 0
    for v in violations:
        f = v["file_path"]
        if f.startswith(str(out_dir)) or f.endswith(".yaml"):
            pfr += 1
            continue
        if v["rule_id"] in IGNORE_RULES:
            ignored += 1
            continue
        sym = scope.symbol(f, v["line"])
        if (f, v["rule_id"], sym) in drift_syms:
            auto_drift += 1
            continue
        targets.append(
            {
                "file_path": f,
                "rule": v["rule_id"],
                "symbol": sym,
                "fingerprint": v["fingerprint"],
                "line": v["line"],
                "message": v["message"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "worklist.json").write_text(json.dumps(targets, indent=1))
    (out_dir / "nomatch_deletes.json").write_text(json.dumps(nomatch_keys, indent=1))
    summary = {
        "targets": len(targets),
        "auto_repaired_drift": auto_drift,
        "decorator_ignored": ignored,
        "per_file_rule": pfr,
        "nomatch_to_delete": len(nomatch_keys),
    }
    (out_dir / "worklist_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# --------------------------------------------------------------------------- #
# codex rejudge
# --------------------------------------------------------------------------- #
def _safe_name(target: _JsonObject) -> str:
    raw = f"{target['file_path']}__{target['rule']}__{target['symbol']}__{target['fingerprint']}"
    return re.sub(r"[^\w.\-]", "_", raw)[:180]


def _build_rejudge_prompt(target: _JsonObject, context: str) -> str:
    return (
        "You are the cicd-judge adjudicating whether a single tier-model finding may be\n"
        "suppressed in the signed allowlist. Decide ACCEPT (legitimate Tier-3 boundary or\n"
        "other prescribed-form suppression) or BLOCK (a real defensive-programming defect /\n"
        "Tier-1 audit-data violation that must crash, not be suppressed).\n\n"
        f"Finding:\n"
        f"- file: {target['file_path']}\n"
        f"- rule: {target['rule']}\n"
        f"- symbol: {target['symbol'].replace(':', '.')}\n"
        f"- line: {target['line']}\n"
        f"- fingerprint: {target['fingerprint']}\n"
        f"- detector message: {target['message']}\n\n"
        "Instructions:\n"
        "- You MUST Read the actual source at that file/line and the enclosing function before judging.\n"
        "- Apply the three-tier trust model below. ACCEPT only a genuine boundary/prescribed pattern.\n"
        "- The rationale must be SITE-SPECIFIC (cite file:line and quote the code), <= 8192 bytes,\n"
        "  and written so a future drift would invalidate it. No generic boilerplate.\n"
        "- If you would BLOCK, say so plainly and say what the correct fix is (reify / let-it-crash).\n\n"
        "Output EXACTLY this markdown, nothing else:\n"
        "## Verdict\n"
        "ACCEPT   (or: BLOCK)\n"
        "## Rationale\n"
        "<prescribed-form justification citing file:line and quoting the code>\n"
        "## Evidence\n"
        "<the specific lines you read, with line numbers>\n\n"
        "Three-tier trust model and prescribed boundary forms (project policy):\n"
        f"{context}\n"
    )


async def _rejudge_batches(
    *,
    targets: _JsonList,
    reports_dir: Path,
    model: str | None,
    repo_root: Path,
    context: str,
    log_path: Path,
    rate_limit: int | None,
    batch_size: int,
    reasoning_effort: str | None,
    skip_existing: bool,
) -> dict[str, int]:
    log_lock = asyncio.Lock()
    rate_limiter = make_codex_rate_limiter(rate_limit)
    pbar = AsyncTqdm(total=len(targets), desc="Re-judging findings", unit="finding")
    failed: list[tuple[str, Exception]] = []
    reports_dir.mkdir(parents=True, exist_ok=True)

    for batch in chunked(targets, batch_size):
        tasks = []
        labels = []
        for t in batch:
            output_path = reports_dir / f"{_safe_name(t)}.md"
            if skip_existing and output_path.exists():
                pbar.update(1)
                continue
            label = f"{t['file_path']}:{t['rule']}:{t['symbol']}"
            labels.append(label)
            tasks.append(
                asyncio.create_task(
                    run_codex_with_retry_and_logging(
                        file_path=Path(t["file_path"]),
                        output_path=output_path,
                        model=model,
                        prompt=_build_rejudge_prompt(t, context),
                        repo_root=repo_root,
                        log_path=log_path,
                        log_lock=log_lock,
                        file_display=label,
                        output_display=str(output_path.relative_to(repo_root) if output_path.is_relative_to(repo_root) else output_path),
                        rate_limiter=rate_limiter,
                        reasoning_effort=reasoning_effort,
                    )
                )
            )
        if tasks:
            for label, res in zip(labels, await asyncio.gather(*tasks, return_exceptions=True), strict=True):
                if isinstance(res, Exception):
                    failed.append((label, res))
                pbar.update(1)
    pbar.close()
    if failed:
        print(f"\n{len(failed)} findings failed (see {log_path}):", file=sys.stderr)
        for label, exc in failed[:10]:
            print(f"  {label}: {str(exc)[:160]}", file=sys.stderr)
    return {"failed": len(failed)}


# --------------------------------------------------------------------------- #
# manifest assembly
# --------------------------------------------------------------------------- #
def build_manifest(out_dir: Path) -> dict[str, int]:
    targets = cast(_JsonList, json.loads((out_dir / "worklist.json").read_text()))
    reports_dir = out_dir / "reports"
    accepted: _JsonList = []
    counts = {"ACCEPT": 0, "BLOCK": 0, "MISSING": 0, "UNPARSED": 0}
    blocked: list[str] = []
    for t in targets:
        report = reports_dir / f"{_safe_name(t)}.md"
        if not report.exists():
            counts["MISSING"] += 1
            continue
        text = report.read_text()
        verdict = extract_section(text, "Verdict").strip().upper()
        rationale = extract_section(text, "Rationale").strip()
        if verdict.startswith("ACCEPT") and rationale:
            counts["ACCEPT"] += 1
            accepted.append(
                {
                    "file_path": t["file_path"],
                    "rule": t["rule"],
                    "symbol": t["symbol"].replace(":", "."),
                    "fingerprint": t["fingerprint"],
                    "rationale": rationale,
                }
            )
        elif verdict.startswith("BLOCK"):
            counts["BLOCK"] += 1
            blocked.append(f"{t['file_path']}:{t['rule']}:{t['symbol']}")
        else:
            counts["UNPARSED"] += 1
    # sign-judge-signatures expects a top-level mapping with an `entries:` list,
    # each entry {file_path, rule, symbol, fingerprint, rationale}.
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump({"entries": accepted}, sort_keys=False, width=10_000))
    (out_dir / "blocked.txt").write_text("\n".join(blocked))
    return counts


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["build-worklist", "rejudge", "build-manifest", "emit-deletes"])
    p.add_argument("--out", default="notes/060-rejudge", help="output dir for worklist/reports/manifest")
    p.add_argument("--repo-root", default=".", help="repository root")
    p.add_argument("--model", default=None, help="codex model (default: codex config)")
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--rate-limit", type=int, default=20, help="max codex starts per minute")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-skip-existing", action="store_true", help="re-run findings already reported")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out).resolve()

    if args.step == "build-worklist":
        summary = build_worklist(repo_root, out_dir)
        print(json.dumps(summary, indent=2))
        print(f"\nwrote {out_dir}/worklist.json  ({summary['targets']} codex targets)")
        return 0

    if args.step == "emit-deletes":
        keys = json.loads((out_dir / "nomatch_deletes.json").read_text())
        print(f"# {len(keys)} stale NO_MATCHING keys to delete from {ALLOWLIST_DIR}/*.yaml before signing:")
        for k in keys:
            print(k)
        return 0

    if args.step == "rejudge":
        targets = json.loads((out_dir / "worklist.json").read_text())
        log_path = out_dir / "rejudge.log"
        ensure_log_file(log_path, header_title="tier-model codex rejudge")
        context = load_context(repo_root, include_skills=True)
        stats = asyncio.run(
            _rejudge_batches(
                targets=targets,
                reports_dir=out_dir / "reports",
                model=args.model,
                repo_root=repo_root,
                context=context,
                log_path=log_path,
                rate_limit=args.rate_limit,
                batch_size=args.batch_size,
                reasoning_effort=args.reasoning_effort,
                skip_existing=not args.no_skip_existing,
            )
        )
        print(json.dumps(stats, indent=2))
        return 1 if stats["failed"] else 0

    if args.step == "build-manifest":
        counts = build_manifest(out_dir)
        print(json.dumps(counts, indent=2))
        print(f"\nwrote {out_dir}/manifest.yaml  ({counts['ACCEPT']} ACCEPTED entries)")
        print(f"      {out_dir}/blocked.txt     ({counts['BLOCK']} BLOCKED -- need a code fix, not a suppression)")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
