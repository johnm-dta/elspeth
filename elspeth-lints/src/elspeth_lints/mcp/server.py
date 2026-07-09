"""``elspeth-judge`` MCP server -- the key-free agent staging surface.

This server lets a key-free agent assemble and inspect an authority-free review
*bundle* (``core.review_bundle``); the operator fires it with the key-bearing
``elspeth-lints sign-bundle`` / ``rekey`` CLI. It mirrors the protocol shape of
``src/elspeth/mcp/server.py`` (``Server``, ``@server.list_tools()``,
``@server.call_tool()``, ``stdio_server()``).

[O1] linchpin -- **the agent never holds the HMAC key.** Every tool handler
calls ``_assert_no_hmac_key_in_env()`` as its *first statement*, before any
optional-dependency import (the ``mcp`` SDK or ``claude-agent-sdk``). If the
key-check sat after a lazy optional-dep import, a handler invoked with the key
set but the extra absent would trip the install-hint ``ImportError`` instead of
failing closed -- silently making the structural guarantee contingent on an
optional dependency. Fail-closed precedes everything.

There is **no authoritative MCP path**: ``verify_signatures`` is always
shape-only. The authoritative HMAC recompute lives on the CLI/library
``diagnose_judge_signatures`` surface, which upgrades only when the operator's
shell holds the key.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from elspeth_lints.core.allowlist import _JUDGE_METADATA_SIGNATURE_ENV_VAR

__all__ = [
    "HmacKeyPresentError",
    "create_server",
    "main",
    "run_server",
]


class HmacKeyPresentError(RuntimeError):
    """Raised when the operator-only HMAC key is present in the agent's env.

    The message names the offending env var so the fail-closed result the
    dispatcher surfaces is self-describing.
    """


def _assert_no_hmac_key_in_env() -> None:
    """Fail closed if the operator-only HMAC key is in the environment.

    Must be the **first statement** of every tool handler, before any
    optional-dependency import. ``raw`` is treated as absent when unset or
    empty (matching ``_verification_mode``'s shape-only branch).
    """
    raw = os.environ.get(_JUDGE_METADATA_SIGNATURE_ENV_VAR)
    if raw:
        raise HmacKeyPresentError(
            f"refusing to run: {_JUDGE_METADATA_SIGNATURE_ENV_VAR} is present in this environment. "
            "The elspeth-judge MCP surface is structurally key-free -- staging asserts, only the "
            "operator CLI (sign-bundle / rekey) holds the key and mints signatures."
        )


@dataclass(frozen=True, slots=True)
class _ServerContext:
    """Resolved roots a tool handler operates over."""

    root: Path
    allowlist_dir: Path
    staged_dir: Path


@dataclass(frozen=True, slots=True)
class _ToolOutcome:
    """An mcp-independent tool result the dispatcher returns.

    ``create_server`` translates this into the ``mcp.types`` protocol objects;
    keeping the dispatcher free of ``mcp`` imports lets the structural
    fail-closed test reach every handler without the SDK installed.
    """

    text: str
    is_error: bool


@dataclass(frozen=True, slots=True)
class _ToolSpec:
    """One registered tool: description, JSON Schema, and handler."""

    description: str
    input_schema: dict[str, Any]
    handler: Callable[[_ServerContext, dict[str, Any]], str]


# Registry of every tool the server exposes. ``create_server`` registers all of
# these; the structural fail-closed test enumerates this exact table, so a
# future tool added here without routing through ``_assert_no_hmac_key_in_env``
# is caught automatically.
_TOOLS: dict[str, _ToolSpec] = {}


def _run_tool(ctx: _ServerContext, name: str, arguments: dict[str, Any]) -> _ToolOutcome:
    """Synchronous, mcp-independent dispatcher.

    Fail-closed (``HmacKeyPresentError``) and operational/argument errors are
    converted into error ``_ToolOutcome``s; genuinely unexpected errors
    propagate so they surface as protocol errors rather than silent text.
    """
    spec = _TOOLS.get(name)
    if spec is None:
        return _ToolOutcome(text=f"unknown tool: {name!r}", is_error=True)
    try:
        text = spec.handler(ctx, arguments)
    except HmacKeyPresentError as exc:
        return _ToolOutcome(text=str(exc), is_error=True)
    except ImportError as exc:
        # An optional-extra-absent install hint (e.g. stage_preview without
        # [judge-agent]) -- a clean error result, never a crash.
        return _ToolOutcome(text=str(exc), is_error=True)
    except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
        return _ToolOutcome(text=f"{name}: {exc}", is_error=True)
    return _ToolOutcome(text=text, is_error=False)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

_OPERATOR_KEY_PLACEHOLDER = f"{_JUDGE_METADATA_SIGNATURE_ENV_VAR}=<operator-held-key>"


def _require_str_arg(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"argument {name!r} is required and must be a non-empty string")
    return value


def _resolve_bundle_path(ctx: _ServerContext, arguments: dict[str, Any]) -> Path:
    """Resolve ``<staged_dir>/<bundle_id>.json`` from the ``bundle_id`` arg."""
    bundle_id = _require_str_arg(arguments, "bundle_id")
    return ctx.staged_dir / f"{bundle_id}.json"


def _shell_join_keep_user(parts: list[str]) -> str:
    import shlex

    return " ".join(part if part == '"$USER"' else shlex.quote(part) for part in parts)


def _sign_bundle_command(ctx: _ServerContext, bundle_path: Path) -> str:
    """The paste-ready operator ``sign-bundle`` command for ``bundle_path``.

    Mirrors ``judge_signature_diagnosis._justify_command``: the operator key is
    an ``env`` placeholder (never a real value), and ``--owner "$USER"`` is left
    unquoted so the operator's shell expands it.
    """
    parts = [
        "env",
        _OPERATOR_KEY_PLACEHOLDER,
        "PYTHONPATH=elspeth-lints/src",
        ".venv/bin/python",
        "-m",
        "elspeth_lints.core.cli",
        "sign-bundle",
        str(bundle_path),
        "--root",
        str(ctx.root),
        "--allowlist-dir",
        str(ctx.allowlist_dir),
        "--owner",
        '"$USER"',
    ]
    return _shell_join_keep_user(parts)


def _rekey_command(ctx: _ServerContext, bundle_path: Path, *, old_key_env: str, new_key_env: str) -> str:
    """The paste-ready operator ``rekey`` command for a staged rekey bundle.

    The two keys are supplied through the operator-named env vars (placeholders
    here, never values); the CLI reads them by the ``--old-key-env`` /
    ``--new-key-env`` names.
    """
    parts = [
        "env",
        f"{old_key_env}=<old-operator-key>",
        f"{new_key_env}=<new-operator-key>",
        "PYTHONPATH=elspeth-lints/src",
        ".venv/bin/python",
        "-m",
        "elspeth_lints.core.cli",
        "rekey",
        "--in",
        str(bundle_path),
        "--old-key-env",
        old_key_env,
        "--new-key-env",
        new_key_env,
        "--root",
        str(ctx.root),
        "--allowlist-dir",
        str(ctx.allowlist_dir),
    ]
    return _shell_join_keep_user(parts)


# --------------------------------------------------------------------------- #
# Tool handlers
# --------------------------------------------------------------------------- #


def _tool_verify_signatures(ctx: _ServerContext, arguments: dict[str, Any]) -> str:
    """Structurally shape-only read-only signature diagnosis (never authoritative).

    Unlike the agent-shell ``diagnose-judge-signatures`` script (shape-only only
    because the shell happens to lack the key), this is **structurally** key-free
    -- ``_assert_no_hmac_key_in_env()`` aborts if a key is ever present -- so it
    is a provably-unprivileged read regardless of the surrounding env.
    """
    _assert_no_hmac_key_in_env()
    from elspeth_lints.core.judge_signature_diagnosis import (
        diagnose_judge_signatures,
        render_judge_signature_diagnosis_json,
    )

    report = diagnose_judge_signatures(root=ctx.root, allowlist_dir=ctx.allowlist_dir)
    return render_judge_signature_diagnosis_json(report)


def _tool_stage_status(ctx: _ServerContext, arguments: dict[str, Any]) -> str:
    """Summarise a staged bundle: per-lane/kind counts + the operator command."""
    _assert_no_hmac_key_in_env()
    from elspeth_lints.core.review_bundle import read_bundle

    bundle_path = _resolve_bundle_path(ctx, arguments)
    bundle = read_bundle(bundle_path)

    lane_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    preview_outcomes: dict[str, int] = {}
    for action in bundle.actions:
        lane_counts[action.lane] = lane_counts.get(action.lane, 0) + 1
        kind_counts[action.kind] = kind_counts.get(action.kind, 0) + 1
        outcome = action.preview.verdict if action.preview is not None else "none"
        preview_outcomes[outcome] = preview_outcomes.get(outcome, 0) + 1

    payload = {
        "bundle_id": bundle.bundle_id,
        "root": bundle.root,
        "allowlist_dir": bundle.allowlist_dir,
        "staged_by": bundle.staged_by,
        "created_at": bundle.created_at,
        "actions_total": len(bundle.actions),
        "lane_counts": lane_counts,
        "kind_counts": kind_counts,
        "preview_outcomes": preview_outcomes,
        "has_rekey_plan": bundle.rekey is not None,
        "sign_bundle_command": _sign_bundle_command(ctx, bundle_path),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _finding_canonical_key(finding: Any) -> str:
    key = finding.canonical_key
    if callable(key):
        key = key()
    if not isinstance(key, str):
        raise ValueError(f"finding.canonical_key must be str; got {type(key).__name__}")
    return key


def _live_findings_for_tree(root: Path, excluded_dirs: tuple[str, ...]) -> list[Any]:
    """Enumerate live tier_model findings across ``root`` via the *shared* helper.

    Uses the same ``scan_single_file_findings`` (scan_file + scan_layer_imports_file)
    that ``verify_bundle_against_tree`` calls per action -- so stage_scan and the
    from-tree verify cannot drift on the scanner set -- and mirrors
    ``scan_directory``'s file discovery (rglob ``*.py``, excluding vendored dirs).
    """
    from elspeth_lints.core.tier_model_scan import scan_single_file_findings

    findings: list[Any] = []
    for py_file in sorted(root.rglob("*.py")):
        relative = py_file.relative_to(root)
        if any(part in excluded_dirs for part in relative.parts):
            continue
        findings.extend(scan_single_file_findings(target_file=py_file, root=root))
    return findings


def _new_judgment_action_from_finding(finding: Any) -> Any:
    from elspeth_lints.core.review_bundle import BundleAction

    symbol_context = finding.symbol_context
    symbol = ".".join(symbol_context) if symbol_context else "_module_"
    return BundleAction(
        lane="new_judgment",
        kind="justify",
        key=_finding_canonical_key(finding),
        file_path=finding.file_path,
        symbol=symbol,
        rule=finding.rule_id,
        fingerprint=finding.fingerprint,
        scope_fingerprint=finding.scope_fingerprint,
        ast_path=finding.ast_path,
    )


def _build_scan_actions(ctx: _ServerContext) -> list[Any]:
    """Survey the tree+allowlist into non-overlapping bundle actions (key-free, no LLM).

    Routing is explicit and non-overlapping:

    * ``drift_repair`` / ``stale_delete`` -- from ``diagnose_judge_signatures``
      (signable drift vs non-signable orphan statuses on judge-gated entries);
    * ``rotation`` -- ``scan_for_rotations(exclude_judge_gated=True).rotations``
      ONLY (the filtered scan cannot raise on the mostly-judge-gated corpus;
      ``.new_findings``/``.ambiguous`` on a *filtered* plan are pollution and are
      never read);
    * ``new_judgment`` -- live findings whose identity-prefix is owned by NO
      entry in the **full, unfiltered** allowlist (the double-route guard: an
      fp-shifted judge-gated entry's live finding shares its prefix with the
      drifted entry, so it routes to ``drift_repair`` alone, never also to a
      spurious ``new_judgment``).
    """
    from elspeth_lints.core.bundle_verify import _STALE_DELETE_ORPHAN_STATUSES
    from elspeth_lints.core.judge_signature_diagnosis import (
        _SIGNABLE_DIAGNOSIS_STATUSES,
        diagnose_judge_signatures,
    )
    from elspeth_lints.core.review_bundle import BundleAction
    from elspeth_lints.rules.trust_tier.tier_model.rotate import identity_prefix, scan_for_rotations
    from elspeth_lints.rules.trust_tier.tier_model.rule import _ALWAYS_EXCLUDED_DIRS, _load_tier_model_allowlist

    actions: list[Any] = []

    # drift_repair + stale_delete from the keyless diagnosis index.
    diagnosis = diagnose_judge_signatures(root=ctx.root, allowlist_dir=ctx.allowlist_dir)
    for item in diagnosis.items:
        if item.status in _SIGNABLE_DIAGNOSIS_STATUSES:
            actions.append(
                BundleAction(
                    lane="resign",
                    kind="drift_repair",
                    key=item.key,
                    source_file=item.source_file,
                    diagnosis_status=item.status,
                )
            )
        elif item.status in _STALE_DELETE_ORPHAN_STATUSES:
            actions.append(BundleAction(lane="resign", kind="stale_delete", key=item.key, source_file=item.source_file))

    # rotation lane: non-judge-gated re-binds only. Only ``.rotations`` is
    # authoritative on a filtered scan.
    rotation_plan = scan_for_rotations(
        source_root=ctx.root,
        allowlist_path=ctx.allowlist_dir,
        exclude_judge_gated=True,
    )
    for rotation in rotation_plan.rotations:
        actions.append(BundleAction(lane="resign", kind="rotation", key=rotation.old_key, source_file=rotation.entry_source_file))

    # new_judgment lane: coverage check against the FULL, unfiltered allowlist.
    allowlist = _load_tier_model_allowlist(ctx.allowlist_dir)
    covered_prefixes: set[str] = set()
    for entry in allowlist.entries:
        try:
            covered_prefixes.add(identity_prefix(entry.key))
        except ValueError:
            continue  # a malformed (non-canonical) key cannot own a prefix
    seen_new: set[str] = set()
    for finding in _live_findings_for_tree(ctx.root, _ALWAYS_EXCLUDED_DIRS):
        canonical_key = _finding_canonical_key(finding)
        try:
            prefix = identity_prefix(canonical_key)
        except ValueError:
            continue
        if prefix in covered_prefixes or canonical_key in seen_new:
            continue
        seen_new.add(canonical_key)
        actions.append(_new_judgment_action_from_finding(finding))

    actions.sort(key=lambda action: (action.kind, action.key))
    return actions


def _tool_stage_scan(ctx: _ServerContext, arguments: dict[str, Any]) -> str:
    """Build/refresh the authority-free worklist bundle. Key-free, no LLM, fast."""
    _assert_no_hmac_key_in_env()
    import uuid
    from datetime import datetime

    from elspeth_lints.core.review_bundle import ReviewBundle, write_bundle

    bundle_id_arg = arguments.get("bundle_id")
    bundle_id = bundle_id_arg if isinstance(bundle_id_arg, str) and bundle_id_arg else f"stage-scan-{uuid.uuid4().hex[:12]}"
    staged_by_arg = arguments.get("staged_by")
    staged_by = staged_by_arg if isinstance(staged_by_arg, str) and staged_by_arg else "elspeth-judge-agent"

    actions = _build_scan_actions(ctx)
    bundle = ReviewBundle(
        bundle_id=bundle_id,
        schema_version=1,
        created_at=datetime.now(UTC).isoformat(),
        staged_by=staged_by,
        root=str(ctx.root),
        allowlist_dir=str(ctx.allowlist_dir),
        source_rev=None,
        source_dirty=False,
        actions=tuple(actions),
    )
    path = write_bundle(bundle, staged_dir=ctx.staged_dir)

    lane_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for action in bundle.actions:
        lane_counts[action.lane] = lane_counts.get(action.lane, 0) + 1
        kind_counts[action.kind] = kind_counts.get(action.kind, 0) + 1

    payload = {
        "bundle_id": bundle.bundle_id,
        "written_path": str(path),
        "actions_total": len(bundle.actions),
        "lane_counts": lane_counts,
        "kind_counts": kind_counts,
        "sign_bundle_command": _sign_bundle_command(ctx, path),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _surrounding_code_for(ctx: _ServerContext, action: Any) -> str:
    """Scrubbed source excerpt for the preview prompt (trust-boundary gate).

    ``surrounding_code`` is shipped to the LLM, so it MUST funnel through the
    secret scrubber ``extract_safe_excerpt`` (path-contains + reads + scrubs in
    one call) -- the same chokepoint ``justify`` uses (cli.py:1649). A raw read
    here would leak un-scrubbed source bytes (the C2-2 leak the structural
    bypass gate forbids). Re-scan the file to locate the staged finding's line
    for a targeted excerpt; any failure degrades to an empty excerpt (the agent
    transport reads source via its read-only ``tool_scope`` regardless).
    """
    from elspeth_lints.core.judge import JUDGE_EXCERPT_CONTEXT_LINES
    from elspeth_lints.core.source_excerpt import (
        SourceExcerptPathOutsideRootError,
        extract_safe_excerpt,
        resolve_safe_excerpt_path,
    )
    from elspeth_lints.core.tier_model_scan import scan_single_file_findings

    if not action.file_path:
        return ""
    try:
        target_file = resolve_safe_excerpt_path(root=ctx.root, target_file=ctx.root / action.file_path)
    except (FileNotFoundError, SourceExcerptPathOutsideRootError):
        return ""
    line = 1
    try:
        for finding in scan_single_file_findings(target_file=target_file, root=ctx.root):
            if _finding_canonical_key(finding) == action.key:
                line = finding.line
                break
    except (OSError, ValueError):
        pass
    try:
        excerpt = extract_safe_excerpt(
            root=ctx.root,
            target_file=target_file,
            line=line,
            context_lines=JUDGE_EXCERPT_CONTEXT_LINES,
        )
    except (OSError, ValueError, SourceExcerptPathOutsideRootError):
        return ""
    return excerpt.text


def _verdict_str(verdict: Any) -> str:
    value = getattr(verdict, "value", None)
    return value if isinstance(value, str) else str(verdict)


def _tool_stage_preview(ctx: _ServerContext, arguments: dict[str, Any]) -> str:
    """Populate each ``new_judgment`` action with a NON-authoritative agent verdict.

    [O1] ordering: ``_assert_no_hmac_key_in_env()`` is the first line, BEFORE the
    lazy judge import -- so a key-present call fails closed even when the
    ``[judge-agent]`` extra is absent. ``ActionPreview.authoritative`` is
    structurally ``False``; the bundle stays signature-free. BLOCKED previews are
    surfaced so the agent can fix code/rationale before the operator step.
    """
    _assert_no_hmac_key_in_env()
    from dataclasses import replace

    from elspeth_lints.core import judge as judge_mod
    from elspeth_lints.core.review_bundle import ActionPreview, read_bundle, write_bundle

    bundle_path = _resolve_bundle_path(ctx, arguments)
    bundle = read_bundle(bundle_path)

    has_justify = any(action.kind == "justify" for action in bundle.actions)
    tool_scope = judge_mod.build_readonly_tool_scope(root=ctx.root, allowlist_dir=ctx.allowlist_dir) if has_justify else None

    new_actions: list[Any] = []
    blocked: list[dict[str, str]] = []
    previewed = 0
    for action in bundle.actions:
        if action.kind != "justify":
            new_actions.append(action)
            continue
        request = judge_mod.JudgeRequest(
            file_path=action.file_path or "",
            rule_id=action.rule or "",
            symbol=action.symbol or "",
            fingerprint=action.fingerprint or "",
            rationale=action.draft_rationale or "",
            surrounding_code=_surrounding_code_for(ctx, action),
        )
        try:
            response = judge_mod.call_judge(request, transport=judge_mod.TRANSPORT_AGENT, tool_scope=tool_scope)
        except (ModuleNotFoundError, judge_mod.JudgeConfigurationError) as exc:
            # ``call_judge`` wraps the missing ``claude_agent_sdk`` import in a
            # ``JudgeConfigurationError`` (judge.py:1465), so that -- not a bare
            # ModuleNotFoundError -- is the real "extra absent" signal. Surface an
            # actionable install hint rather than a raw traceback; the other four
            # key-free tools keep working without the [judge-agent] extra.
            message = str(exc)
            if "claude_agent_sdk" in message or "claude-agent-sdk" in message:
                raise ImportError(
                    "stage_preview requires the [judge-agent] extra: uv pip install -e '.[judge-agent]' from elspeth-lints/"
                ) from exc
            raise
        verdict = _verdict_str(response.verdict)
        preview = ActionPreview(
            verdict=verdict,
            rationale=response.judge_rationale,
            model=response.model_id,
            transport=response.judge_transport,
            authoritative=False,
        )
        new_actions.append(replace(action, preview=preview))
        previewed += 1
        if verdict == "BLOCKED":
            blocked.append({"key": action.key, "rationale": response.judge_rationale})

    new_bundle = replace(bundle, actions=tuple(new_actions))
    path = write_bundle(new_bundle, staged_dir=ctx.staged_dir)
    payload = {
        "bundle_id": new_bundle.bundle_id,
        "written_path": str(path),
        "previewed": previewed,
        "blocked": blocked,
        "sign_bundle_command": _sign_bundle_command(ctx, path),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# Shape-only "currently valid" statuses (a key-free diagnosis cannot prove HMAC
# validity, only shape + source binding -- so the partition is advisory; the
# operator ``rekey`` CLI's Pass-1 is the authoritative gate).
_REKEY_VALID_STATUSES = frozenset({"OK_SHAPE_ONLY", "OK_AUTHORITATIVE"})


def _tool_stage_rekey(ctx: _ServerContext, arguments: dict[str, Any]) -> str:
    """Enumerate currently-valid judge-gated entries and flag broken ones.

    Shape-only and *advisory*: without the key, HMAC validity cannot be
    determined, so a shape-valid-but-HMAC-invalid entry may be mislabeled into
    ``rekey.keys``. The operator ``rekey`` CLI's Pass-1 (keyed verify) is the
    authoritative gate. Only the env-var NAMES are recorded -- never key bytes.
    """
    _assert_no_hmac_key_in_env()
    import uuid
    from datetime import datetime

    from elspeth_lints.core.judge_signature_diagnosis import _OK_STATUSES, diagnose_judge_signatures
    from elspeth_lints.core.review_bundle import RekeyPlan, ReviewBundle, write_bundle

    old_key_env = _require_str_arg(arguments, "old_key_env")
    new_key_env = _require_str_arg(arguments, "new_key_env")
    bundle_id_arg = arguments.get("bundle_id")
    bundle_id = bundle_id_arg if isinstance(bundle_id_arg, str) and bundle_id_arg else f"stage-rekey-{uuid.uuid4().hex[:12]}"
    staged_by_arg = arguments.get("staged_by")
    staged_by = staged_by_arg if isinstance(staged_by_arg, str) and staged_by_arg else "elspeth-judge-agent"

    diagnosis = diagnose_judge_signatures(root=ctx.root, allowlist_dir=ctx.allowlist_dir)
    valid_keys: list[str] = []
    broken_keys: list[str] = []
    for item in diagnosis.items:
        if item.status in _REKEY_VALID_STATUSES:
            valid_keys.append(item.key)
        elif item.status not in _OK_STATUSES:
            # Not OK and not PRE_JUDGE -> a judge-gated entry that does not
            # currently verify shape/binding. (PRE_JUDGE is non-judge-gated and
            # is not part of the rekey set.)
            broken_keys.append(item.key)

    rekey = RekeyPlan(
        old_key_env=old_key_env,
        new_key_env=new_key_env,
        keys=tuple(sorted(valid_keys)),
        broken_keys=tuple(sorted(broken_keys)),
    )
    bundle = ReviewBundle(
        bundle_id=bundle_id,
        schema_version=1,
        created_at=datetime.now(UTC).isoformat(),
        staged_by=staged_by,
        root=str(ctx.root),
        allowlist_dir=str(ctx.allowlist_dir),
        source_rev=None,
        source_dirty=False,
        actions=(),
        rekey=rekey,
    )
    path = write_bundle(bundle, staged_dir=ctx.staged_dir)
    payload = {
        "bundle_id": bundle.bundle_id,
        "written_path": str(path),
        "old_key_env": old_key_env,
        "new_key_env": new_key_env,
        "valid_count": len(rekey.keys),
        "broken_count": len(rekey.broken_keys),
        "rekey_command": _rekey_command(ctx, path, old_key_env=old_key_env, new_key_env=new_key_env),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


_TOOLS.update(
    {
        "verify_signatures": _ToolSpec(
            description=(
                "Read-only, structurally key-free signature diagnosis of the tier_model allowlist "
                "(always shape-only; the authoritative HMAC recompute is the operator CLI)."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_tool_verify_signatures,
        ),
        "stage_status": _ToolSpec(
            description=(
                "Summarise a staged review bundle (per-lane/kind counts, preview outcomes) and emit "
                "the paste-ready operator sign-bundle command."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "bundle_id": {"type": "string", "description": "Bundle id (file is <staged-dir>/<id>.json)"},
                },
                "required": ["bundle_id"],
            },
            handler=_tool_stage_status,
        ),
        "stage_scan": _ToolSpec(
            description=(
                "Survey the source tree + tier_model allowlist into an authority-free worklist bundle "
                "(drift_repair / rotation / stale_delete / new_judgment lanes). Key-free, no LLM."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "bundle_id": {"type": "string", "description": "Bundle id to write (default: generated)"},
                    "staged_by": {"type": "string", "description": "Agent/operator label recorded on the bundle"},
                },
            },
            handler=_tool_stage_scan,
        ),
        "stage_preview": _ToolSpec(
            description=(
                "Run the read-only agent judge over each new_judgment action and record a "
                "NON-authoritative preview verdict (never signs; surfaces BLOCKED reasons). "
                "Requires the [judge-agent] extra."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "bundle_id": {"type": "string", "description": "Bundle id (file is <staged-dir>/<id>.json)"},
                },
                "required": ["bundle_id"],
            },
            handler=_tool_stage_preview,
        ),
        "stage_rekey": _ToolSpec(
            description=(
                "Enumerate currently-valid judge-gated entries and flag broken ones into a rekey "
                "bundle (records env-var NAMES only -- never key bytes). Shape-only/advisory; the "
                "operator rekey CLI's Pass-1 is the authoritative gate."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "old_key_env": {"type": "string", "description": "NAME of the env var holding the OLD key"},
                    "new_key_env": {"type": "string", "description": "NAME of the env var holding the NEW key"},
                    "bundle_id": {"type": "string", "description": "Bundle id to write (default: generated)"},
                    "staged_by": {"type": "string", "description": "Agent/operator label recorded on the bundle"},
                },
                "required": ["old_key_env", "new_key_env"],
            },
            handler=_tool_stage_rekey,
        ),
    }
)


def create_server(
    *,
    root: Path,
    allowlist_dir: Path,
    staged_dir: Path,
) -> Any:
    """Create the ``elspeth-judge`` MCP server bound to the given roots."""
    from mcp.server import Server
    from mcp.types import CallToolResult, TextContent, Tool

    ctx = _ServerContext(root=Path(root), allowlist_dir=Path(allowlist_dir), staged_dir=Path(staged_dir))
    server = Server("elspeth-judge")

    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def list_tools() -> list[Tool]:
        return [Tool(name=name, description=spec.description, inputSchema=spec.input_schema) for name, spec in _TOOLS.items()]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult | list[TextContent]:
        outcome = _run_tool(ctx, name, arguments)
        if outcome.is_error:
            return CallToolResult(content=[TextContent(type="text", text=outcome.text)], isError=True)
        return [TextContent(type="text", text=outcome.text)]

    return server


async def run_server(*, root: Path, allowlist_dir: Path, staged_dir: Path) -> None:
    """Run the server over stdio."""
    from mcp.server.stdio import stdio_server

    server = create_server(root=root, allowlist_dir=allowlist_dir, staged_dir=staged_dir)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m elspeth_lints.mcp`` / the console script."""
    import asyncio

    parser = argparse.ArgumentParser(
        prog="elspeth-judge-mcp",
        description="ELSPETH key-free judge/signature staging MCP server",
    )
    parser.add_argument("--root", type=Path, default=Path("src/elspeth"), help="Source tree to scan")
    parser.add_argument(
        "--allowlist-dir",
        type=Path,
        default=Path("config/cicd/enforce_tier_model"),
        help="Directory of per-module tier_model allowlist YAML files",
    )
    parser.add_argument(
        "--staged-dir",
        type=Path,
        default=Path(".elspeth/staged-reviews"),
        help="Directory where staged review bundles are written/read",
    )
    args = parser.parse_args(argv)
    asyncio.run(run_server(root=args.root, allowlist_dir=args.allowlist_dir, staged_dir=args.staged_dir))


if __name__ == "__main__":
    main()
