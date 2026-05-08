"""Decode `chat_messages.tool_calls` audit envelopes for a composer session.

Diagnostic helper used by composer reliability investigations. Reads the
sessions DB read-only (SQLite `mode=ro` URI) and crashes loud when the DB or
session is missing — silent defaults would be evidence tampering on a Tier 1
audit-derived dataset.

Usage (CLI)::

    python -m evals.lib.decode_tools data/sessions.db <session_id>

Usage (library)::

    from evals.lib.decode_tools import decode_tool_sequence
    seq = decode_tool_sequence("data/sessions.db", session_id)
    for entry in seq:
        print(entry["ts"], entry["role"], entry["tool_name"])

Each returned dict has keys ``ts``, ``role``, ``tool_name``,
``arguments_canonical`` (parsed JSON object or ``None``), and
``result_summary`` (string truncated to 300 chars or ``None``). Rows whose
``tool_calls`` envelope is not an `_kind == "audit"` invocation envelope (e.g.,
`llm_call_audit` LLM-call records or rows with no envelope) are returned with
``tool_name=None``, ``arguments_canonical=None``, ``result_summary=None`` so
callers see the full chronology, not just the audit subset.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote

_RESULT_SUMMARY_MAX = 300

DecodedRow = dict[str, object]


def decode_tool_sequence(db_path: str, session_id: str) -> list[DecodedRow]:
    """Return the ordered chat-message sequence for ``session_id``.

    Crashes loud (FileNotFoundError / ValueError) on missing DB or session.
    The SQLite connection is opened with ``mode=ro`` so the helper can never
    mutate the audit DB even by accident.
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"sessions DB not found: {db_path}")

    uri = f"file:{quote(str(path))}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT created_at, role, content, tool_calls
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY sequence_no
            """,
            (session_id,),
        ).fetchall()

    if not rows:
        raise ValueError(f"no chat_messages rows for session {session_id!r}")

    decoded: list[DecodedRow] = []
    for row in rows:
        invocation = _extract_audit_invocation(row["tool_calls"])
        tool_name = invocation.get("tool_name") if invocation is not None else None
        args_raw = invocation.get("arguments_canonical") if invocation is not None else None
        result_raw = invocation.get("result_canonical") if invocation is not None else None

        decoded.append(
            {
                "ts": row["created_at"],
                "role": row["role"],
                "tool_name": tool_name,
                "arguments_canonical": _safe_json_parse(args_raw),
                "result_summary": _truncate(result_raw, _RESULT_SUMMARY_MAX),
            }
        )
    return decoded


def _extract_audit_invocation(tool_calls_json: str | None) -> dict[str, str] | None:
    """Return the audit envelope's ``invocation`` block iff envelope kind is "audit".

    Returns a dict whose values are stringly-typed (the on-disk shape: tool_name,
    arguments_canonical, result_canonical, … are all stored as strings; nested
    JSON payloads remain serialised). Other envelope kinds (notably
    ``llm_call_audit``) carry LLM-call records and are not what diagnosis cares
    about; they yield None.
    """
    if not tool_calls_json:
        return None
    parsed = json.loads(tool_calls_json)
    if not isinstance(parsed, list) or not parsed:
        return None
    first = parsed[0]
    if not isinstance(first, dict) or first.get("_kind") != "audit":
        return None
    invocation = first.get("invocation")
    if not isinstance(invocation, dict):
        return None
    # Narrow value types: every documented invocation field is a string (or null
    # for error_class/error_message). Coerce non-string values to None to keep
    # the static type honest without lying about content.
    return {k: v for k, v in invocation.items() if isinstance(v, str)}


def _safe_json_parse(raw: str | None) -> object | None:
    if raw is None:
        return None
    parsed: object = json.loads(raw)
    return parsed


def _truncate(raw: str | None, limit: int) -> str | None:
    if raw is None:
        return None
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "…"


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: python -m evals.lib.decode_tools <db_path> <session_id>\n")
        return 64  # EX_USAGE
    db_path, session_id = argv
    sequence = decode_tool_sequence(db_path, session_id)
    json.dump(sequence, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via CLI smoke test
    raise SystemExit(_main(sys.argv[1:]))
