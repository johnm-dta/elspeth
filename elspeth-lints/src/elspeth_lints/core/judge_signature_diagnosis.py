"""Read-only diagnosis for signed judge allowlist metadata.

This module is the keyless side of the cicd-judge repair workflow. It may tell
an agent or operator what is stale, malformed, or unverifiable, but it never
writes YAML and never needs ``ELSPETH_JUDGE_METADATA_HMAC_KEY``. When that key is
present in an operator-held shell, the diagnosis upgrades to authoritative HMAC
verification; otherwise it reports shape/source-binding findings only and emits
commands with a placeholder for the operator-held key.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values

from elspeth_lints.core.allowlist import (
    _JUDGE_METADATA_SIGNATURE_ENV_VAR,
    _JUDGE_METADATA_SIGNATURE_VERIFY_MODE_ENV_VAR,
    AllowlistEntry,
    _compute_file_fingerprint,
    _judge_metadata_hmac_key,
    _parse_allow_hits,
    _verify_judge_metadata_signature_at_load,
    find_scope_fallback_entry,
    verify_entry_binding_against_finding,
)
from elspeth_lints.core.reaudit import (
    AmbiguousFindingMatchError,
    _canonical_key_for_finding,
    _find_matching_finding,
    _parse_entry_key,
    _scan_tier_model,
)
from elspeth_lints.core.source_excerpt import (
    SourceExcerptPathOutsideRootError,
    resolve_safe_excerpt_path,
)

VerificationMode = Literal["shape-only", "authoritative"]

_OPERATOR_KEY_PLACEHOLDER = f"{_JUDGE_METADATA_SIGNATURE_ENV_VAR}=<operator-held-key>"
_OK_STATUSES = frozenset({"OK_SHAPE_ONLY", "OK_AUTHORITATIVE", "PRE_JUDGE"})
_DIAGNOSIS_ENV_FILE_KEYS = frozenset(
    {
        _JUDGE_METADATA_SIGNATURE_ENV_VAR,
        _JUDGE_METADATA_SIGNATURE_VERIFY_MODE_ENV_VAR,
    }
)


@dataclass(frozen=True, slots=True)
class JudgeSignatureDiagnosis:
    """One operator-actionable diagnosis for one allowlist entry."""

    status: str
    key: str
    source_file: str
    note: str
    repair_command: str | None = None
    repair_key: str | None = None
    detail: str | None = None

    @property
    def requires_action(self) -> bool:
        """Return whether this item should make the CLI exit non-zero."""
        return self.status not in _OK_STATUSES


@dataclass(frozen=True, slots=True)
class JudgeSignatureDiagnosisReport:
    """Aggregate diagnosis output for the CLI/MCP surface."""

    verification_mode: VerificationMode
    allowlist_dir: Path
    root: Path
    items: tuple[JudgeSignatureDiagnosis, ...]

    @property
    def requires_action(self) -> bool:
        """Return whether any entry needs an operator repair action."""
        return any(item.requires_action for item in self.items)


def diagnose_judge_signatures(*, root: Path, allowlist_dir: Path) -> JudgeSignatureDiagnosisReport:
    """Inspect signed allowlist metadata without writing or requiring HMAC custody."""
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"--root: {resolved_root} is not a directory")
    if not allowlist_dir.is_dir():
        raise ValueError(f"--allowlist-dir: {allowlist_dir} is not a directory")

    mode = _verification_mode()
    findings_by_file: dict[Path, list[Any]] = {}
    items: list[JudgeSignatureDiagnosis] = []
    for yaml_file in sorted(path for path in allowlist_dir.glob("*.yaml") if path.name != "_defaults.yaml"):
        items.extend(
            _diagnose_yaml_file(
                yaml_file=yaml_file,
                root=resolved_root,
                allowlist_dir=allowlist_dir,
                mode=mode,
                findings_by_file=findings_by_file,
            )
        )
    return JudgeSignatureDiagnosisReport(
        verification_mode=mode,
        allowlist_dir=allowlist_dir,
        root=resolved_root,
        items=tuple(items),
    )


def load_diagnosis_env_file(env_file: Path | None) -> None:
    """Load diagnosis-relevant keys from a dotenv file without printing secrets.

    Existing process environment values win. The command only reads the keys it
    needs for judge-signature verification, so a general operator dotenv file
    does not pull unrelated secrets into this process.
    """
    if env_file is None:
        return
    resolved = env_file.resolve()
    if not resolved.is_file():
        raise ValueError(f"--env-file: {resolved} is not a file")

    values = dotenv_values(resolved)
    for key, value in values.items():
        if key not in _DIAGNOSIS_ENV_FILE_KEYS or value is None or key in os.environ:
            continue
        os.environ[key] = value


def render_judge_signature_diagnosis_text(report: JudgeSignatureDiagnosisReport) -> str:
    """Render the report in a compact operator-readable text format."""
    lines = [
        "diagnose-judge-signatures",
        f"verification_mode: {report.verification_mode}",
        f"root: {report.root}",
        f"allowlist_dir: {report.allowlist_dir}",
        f"entries: {len(report.items)}",
        "hmac_key_custody: read-only diagnosis does not require the operator key; repair commands keep it as a placeholder",
    ]
    if not report.items:
        lines.append("no allow_hits entries found")
    for item in report.items:
        lines.append(f"- {item.status}: {item.key}")
        lines.append(f"  source_file: {item.source_file}")
        lines.append(f"  note: {item.note}")
        if item.detail is not None:
            lines.append(f"  detail: {item.detail}")
        if item.repair_command is not None:
            lines.append(f"  repair: {item.repair_command}")
        if item.repair_key is not None and item.repair_key != item.key:
            lines.append(f"  repair_key: {item.repair_key}")
    return "\n".join(lines) + "\n"


def render_judge_signature_diagnosis_json(report: JudgeSignatureDiagnosisReport) -> str:
    """Render the report as stable JSON for future MCP/tool consumers."""
    payload = {
        "verification_mode": report.verification_mode,
        "root": str(report.root),
        "allowlist_dir": str(report.allowlist_dir),
        "entries": [
            {
                "status": item.status,
                "key": item.key,
                "source_file": item.source_file,
                "note": item.note,
                "detail": item.detail,
                "repair_command": item.repair_command,
                "repair_key": item.repair_key,
                "requires_action": item.requires_action,
            }
            for item in report.items
        ],
        "requires_action": report.requires_action,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the standalone ``diagnose-judge-signatures`` console command."""
    parser = argparse.ArgumentParser(prog="diagnose-judge-signatures")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("src/elspeth"),
        help="Source tree to scan for the entries' underlying findings",
    )
    parser.add_argument(
        "--allowlist-dir",
        type=Path,
        default=Path("config/cicd/enforce_tier_model"),
        help="Directory of per-module allowlist YAML files to inspect read-only",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Dotenv file containing diagnosis-relevant keys such as "
            "ELSPETH_JUDGE_METADATA_HMAC_KEY. Existing environment values win; "
            "unrelated keys are ignored."
        ),
    )
    parser.add_argument(
        "--format",
        dest="diagnose_format",
        choices=("text", "json"),
        default="text",
        help="Output format for the diagnosis report",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        load_diagnosis_env_file(args.env_file)
        report = diagnose_judge_signatures(root=args.root, allowlist_dir=args.allowlist_dir)
    except ValueError as exc:
        sys.stderr.write(f"diagnose-judge-signatures error: {exc}\n")
        return 2

    if args.diagnose_format == "json":
        sys.stdout.write(render_judge_signature_diagnosis_json(report))
    else:
        sys.stdout.write(render_judge_signature_diagnosis_text(report))
    return 1 if report.requires_action else 0


def _verification_mode() -> VerificationMode:
    raw_key = os.environ.get(_JUDGE_METADATA_SIGNATURE_ENV_VAR)
    if raw_key is None or raw_key == "":
        return "shape-only"
    _judge_metadata_hmac_key()
    return "authoritative"


def _diagnose_yaml_file(
    *,
    yaml_file: Path,
    root: Path,
    allowlist_dir: Path,
    mode: VerificationMode,
    findings_by_file: dict[Path, list[Any]],
) -> list[JudgeSignatureDiagnosis]:
    try:
        raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [
            JudgeSignatureDiagnosis(
                status="ALLOWLIST_PARSE_ERROR",
                key="<unparsed>",
                source_file=yaml_file.name,
                note="allowlist YAML could not be read or parsed",
                detail=str(exc),
            )
        ]
    if not isinstance(raw, dict):
        return [
            JudgeSignatureDiagnosis(
                status="ALLOWLIST_PARSE_ERROR",
                key="<unparsed>",
                source_file=yaml_file.name,
                note="allowlist YAML must be a mapping",
                detail=f"got {type(raw).__name__}",
            )
        ]

    raw_entries = raw.get("allow_hits", [])
    if raw_entries is None:
        raw_entries = []
    if not isinstance(raw_entries, list):
        return [
            JudgeSignatureDiagnosis(
                status="ALLOWLIST_PARSE_ERROR",
                key="<unparsed>",
                source_file=yaml_file.name,
                note="allow_hits must be a list",
                detail=f"got {type(raw_entries).__name__}",
            )
        ]

    items: list[JudgeSignatureDiagnosis] = []
    for index, raw_entry in enumerate(raw_entries):
        items.append(
            _diagnose_raw_entry(
                raw_entry=raw_entry,
                index=index,
                source_file=yaml_file.name,
                root=root,
                allowlist_dir=allowlist_dir,
                mode=mode,
                findings_by_file=findings_by_file,
            )
        )
    return items


def _diagnose_raw_entry(
    *,
    raw_entry: object,
    index: int,
    source_file: str,
    root: Path,
    allowlist_dir: Path,
    mode: VerificationMode,
    findings_by_file: dict[Path, list[Any]],
) -> JudgeSignatureDiagnosis:
    raw_key = _raw_entry_key(raw_entry)
    try:
        parsed = _parse_allow_hits({"allow_hits": [raw_entry]}, source_file=source_file, source_root=None)
    except ValueError as exc:
        return JudgeSignatureDiagnosis(
            status="METADATA_SHAPE_INVALID",
            key=raw_key,
            source_file=source_file,
            note=f"allow_hits[{index}] has internally inconsistent judge metadata; operator review required",
            detail=str(exc),
        )
    if len(parsed) != 1:
        return JudgeSignatureDiagnosis(
            status="METADATA_SHAPE_INVALID",
            key=raw_key,
            source_file=source_file,
            note=f"allow_hits[{index}] did not parse into exactly one entry",
            detail=f"parsed_count={len(parsed)}",
        )
    entry = parsed[0]
    if entry.judge_verdict is None:
        return JudgeSignatureDiagnosis(
            status="PRE_JUDGE",
            key=entry.key,
            source_file=source_file,
            note="entry predates signed judge metadata; no signed metadata repair command is applicable",
        )
    if entry.judge_metadata_signature is None:
        return JudgeSignatureDiagnosis(
            status="MISSING_SIGNATURE",
            key=entry.key,
            source_file=source_file,
            note="re-justify required; post-judge metadata is unsigned editable text",
            repair_command=_justify_command(entry=entry, root=root, allowlist_dir=allowlist_dir),
        )

    if mode == "authoritative":
        try:
            _verify_judge_metadata_signature_at_load(entry, context=f"{source_file}:allow_hits[{index}]")
        except ValueError as exc:
            return JudgeSignatureDiagnosis(
                status="INVALID_SIGNATURE",
                key=entry.key,
                source_file=source_file,
                note="re-justify required; persisted signed judge metadata does not match the operator HMAC recompute",
                detail=str(exc),
                repair_command=_justify_command(entry=entry, root=root, allowlist_dir=allowlist_dir),
            )

    key_parts = _parse_entry_key(entry.key)
    if key_parts is None:
        return JudgeSignatureDiagnosis(
            status="ENTRY_KEY_INVALID",
            key=entry.key,
            source_file=source_file,
            note="entry key is not in canonical form; operator must inspect and re-justify or remove it",
            repair_command=_justify_command(entry=entry, root=root, allowlist_dir=allowlist_dir),
        )
    file_path, _rule_id, _symbol_context, _fingerprint = key_parts
    try:
        target_file = resolve_safe_excerpt_path(root=root, target_file=root / file_path)
    except FileNotFoundError as exc:
        return JudgeSignatureDiagnosis(
            status="SOURCE_FILE_MISSING",
            key=entry.key,
            source_file=source_file,
            note="bound source file is missing; re-justify cannot run until the live finding path exists",
            detail=str(exc),
        )
    except SourceExcerptPathOutsideRootError as exc:
        return JudgeSignatureDiagnosis(
            status="SOURCE_PATH_OUTSIDE_ROOT",
            key=entry.key,
            source_file=source_file,
            note="entry key resolves outside --root; investigate YAML for tampering",
            detail=str(exc),
        )

    findings = _findings_for_file(target_file=target_file, root=root, cache=findings_by_file)
    try:
        matching_finding = _find_matching_finding(findings=findings, entry_key=entry.key)
    except AmbiguousFindingMatchError as exc:
        return JudgeSignatureDiagnosis(
            status="AMBIGUOUS_FINDING_MATCH",
            key=entry.key,
            source_file=source_file,
            note="more than one live finding has this canonical key; scanner ambiguity must be resolved before repair",
            detail=str(exc),
        )

    version = entry.judge_signature_version if entry.judge_signature_version is not None else 1
    if version == 1:
        return _diagnose_v1_entry(
            entry=entry,
            source_file=source_file,
            target_file=target_file,
            matching_finding=matching_finding,
            mode=mode,
            root=root,
            allowlist_dir=allowlist_dir,
        )
    return _diagnose_v2_entry(
        entry=entry,
        source_file=source_file,
        findings=findings,
        matching_finding=matching_finding,
        mode=mode,
        root=root,
        allowlist_dir=allowlist_dir,
    )


def _diagnose_v1_entry(
    *,
    entry: AllowlistEntry,
    source_file: str,
    target_file: Path,
    matching_finding: Any | None,
    mode: VerificationMode,
    root: Path,
    allowlist_dir: Path,
) -> JudgeSignatureDiagnosis:
    assert entry.file_fingerprint is not None
    live_fingerprint = _compute_file_fingerprint(target_file)
    if entry.file_fingerprint != live_fingerprint:
        if matching_finding is None:
            return JudgeSignatureDiagnosis(
                status="NO_MATCHING_FINDING",
                key=entry.key,
                source_file=source_file,
                note="source bytes drifted and no live finding still matches this entry; remove or re-justify after inspection",
                repair_command=_justify_command(entry=entry, root=root, allowlist_dir=allowlist_dir),
            )
        return JudgeSignatureDiagnosis(
            status="V1_FILE_FINGERPRINT_DRIFT",
            key=entry.key,
            source_file=source_file,
            note="migrate-judge-scope required; v1 file_fingerprint drifted but the live finding still exists",
            repair_command=_migrate_command(root=root, allowlist_dir=allowlist_dir),
        )

    if matching_finding is None:
        return JudgeSignatureDiagnosis(
            status="NO_MATCHING_FINDING",
            key=entry.key,
            source_file=source_file,
            note="no live finding has this canonical key; remove or re-justify after inspection",
            repair_command=_justify_command(entry=entry, root=root, allowlist_dir=allowlist_dir),
        )

    try:
        verify_entry_binding_against_finding(
            entry,
            file_path=_entry_file_path(entry),
            ast_path=_finding_string(matching_finding, "ast_path"),
            scope_fingerprint=_finding_string(matching_finding, "scope_fingerprint"),
        )
    except ValueError as exc:
        return JudgeSignatureDiagnosis(
            status="AST_PATH_BINDING_DRIFT",
            key=entry.key,
            source_file=source_file,
            note="re-justify required; the stored AST binding no longer matches the live finding",
            detail=str(exc),
            repair_command=_justify_command(
                entry=entry,
                root=root,
                allowlist_dir=allowlist_dir,
                repair_key=_canonical_key_for_finding(matching_finding),
            ),
            repair_key=_canonical_key_for_finding(matching_finding),
        )
    return _ok(entry=entry, source_file=source_file, mode=mode)


def _diagnose_v2_entry(
    *,
    entry: AllowlistEntry,
    source_file: str,
    findings: list[Any],
    matching_finding: Any | None,
    mode: VerificationMode,
    root: Path,
    allowlist_dir: Path,
) -> JudgeSignatureDiagnosis:
    finding = matching_finding
    if finding is None:
        fallback_matches = [
            candidate
            for candidate in findings
            if find_scope_fallback_entry(
                [entry],
                canonical_key=_canonical_key_for_finding(candidate),
                scope_fingerprint=_finding_string(candidate, "scope_fingerprint"),
                ast_path=_finding_string(candidate, "ast_path"),
                scope_depth=_finding_int(candidate, "scope_depth"),
            )
            is entry
        ]
        if len(fallback_matches) > 1:
            return JudgeSignatureDiagnosis(
                status="AMBIGUOUS_SCOPE_FALLBACK",
                key=entry.key,
                source_file=source_file,
                note="more than one live finding matched the v2 scope fallback; operator inspection required",
                repair_command=_justify_command(entry=entry, root=root, allowlist_dir=allowlist_dir),
            )
        if len(fallback_matches) == 1:
            finding = fallback_matches[0]
        else:
            return JudgeSignatureDiagnosis(
                status="NO_MATCHING_FINDING",
                key=entry.key,
                source_file=source_file,
                note="no live finding has this canonical key or v2 scope fallback; remove or re-justify after inspection",
                repair_command=_justify_command(entry=entry, root=root, allowlist_dir=allowlist_dir),
            )

    try:
        verify_entry_binding_against_finding(
            entry,
            file_path=_entry_file_path(entry),
            ast_path=_finding_string(finding, "ast_path"),
            scope_fingerprint=_finding_string(finding, "scope_fingerprint"),
        )
    except ValueError as exc:
        message = str(exc)
        if "scope_fingerprint" in message:
            status = "SCOPE_BINDING_DRIFT"
        elif "ast_path" in message:
            status = "AST_PATH_BINDING_DRIFT"
        else:
            status = "BINDING_DRIFT"
        return JudgeSignatureDiagnosis(
            status=status,
            key=entry.key,
            source_file=source_file,
            note="re-justify required; signed source binding no longer matches the live finding",
            detail=message,
            repair_command=_justify_command(
                entry=entry,
                root=root,
                allowlist_dir=allowlist_dir,
                repair_key=_canonical_key_for_finding(finding),
            ),
            repair_key=_canonical_key_for_finding(finding),
        )
    return _ok(entry=entry, source_file=source_file, mode=mode)


def _findings_for_file(*, target_file: Path, root: Path, cache: dict[Path, list[Any]]) -> list[Any]:
    cached = cache.get(target_file)
    if cached is None:
        cached = _scan_tier_model(target_file=target_file, root=root)
        cache[target_file] = cached
    return cached


def _ok(*, entry: AllowlistEntry, source_file: str, mode: VerificationMode) -> JudgeSignatureDiagnosis:
    if mode == "authoritative":
        return JudgeSignatureDiagnosis(
            status="OK_AUTHORITATIVE",
            key=entry.key,
            source_file=source_file,
            note="signature HMAC and live source binding verified",
        )
    return JudgeSignatureDiagnosis(
        status="OK_SHAPE_ONLY",
        key=entry.key,
        source_file=source_file,
        note="signature shape and live source binding verified; HMAC recompute skipped because operator key is absent",
    )


def _raw_entry_key(raw_entry: object) -> str:
    if isinstance(raw_entry, dict):
        raw_key = raw_entry.get("key")
        if isinstance(raw_key, str):
            return raw_key
    return "<unparsed>"


def _entry_file_path(entry: AllowlistEntry) -> str:
    parsed = _parse_entry_key(entry.key)
    if parsed is None:
        return "<invalid>"
    return parsed[0]


def _finding_string(finding: Any, field_name: str) -> str:
    value = getattr(finding, field_name)
    if not isinstance(value, str):
        raise ValueError(f"finding.{field_name} must be str; got {type(value).__name__}")
    return value


def _finding_int(finding: Any, field_name: str) -> int:
    value = getattr(finding, field_name)
    if not isinstance(value, int):
        raise ValueError(f"finding.{field_name} must be int; got {type(value).__name__}")
    return value


def _justify_command(*, entry: AllowlistEntry, root: Path, allowlist_dir: Path, repair_key: str | None = None) -> str:
    parsed = _parse_entry_key(repair_key or entry.key)
    if parsed is None:
        return (
            "# Cannot emit an exact justify command because the entry key is malformed; "
            "inspect the YAML and rerun justify against the live finding."
        )
    file_path, rule_id, symbol_context, fingerprint = parsed
    symbol = ".".join(symbol_context) if symbol_context else "_module_"
    rationale = entry.reason.strip() if entry.reason.strip() else "Re-sign stale judge metadata after operator inspection."
    parts = [
        "env",
        _OPERATOR_KEY_PLACEHOLDER,
        "PYTHONPATH=elspeth-lints/src",
        ".venv/bin/python",
        "-m",
        "elspeth_lints.core.cli",
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        file_path,
        "--rule",
        rule_id,
        "--symbol",
        symbol,
        "--fingerprint",
        fingerprint,
        "--rationale",
        rationale,
        "--owner",
        '"$USER"',
    ]
    return _shell_join_keep_user(parts)


def _migrate_command(*, root: Path, allowlist_dir: Path) -> str:
    parts = [
        "env",
        _OPERATOR_KEY_PLACEHOLDER,
        "PYTHONPATH=elspeth-lints/src",
        ".venv/bin/python",
        "-m",
        "elspeth_lints.core.cli",
        "migrate-judge-scope",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--owner",
        '"$USER"',
    ]
    return _shell_join_keep_user(parts)


def _shell_join_keep_user(parts: list[str]) -> str:
    return " ".join(part if part == '"$USER"' else shlex.quote(part) for part in parts)
