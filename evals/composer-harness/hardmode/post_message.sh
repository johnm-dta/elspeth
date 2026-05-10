#!/usr/bin/env bash
# evals/composer-harness/hardmode/post_message.sh — send one user turn, capture metrics.
#
# Usage: post_message.sh <scenario_id> <turn_index> <user_message_file>
#
# user_message_file: plain text, NOT JSON-wrapped. Saved verbatim into msg.t<N>.req.json
#                    (under .content) and into the run dir as turn<N>.user.txt for audit.
#
# Side-effects per turn:
#   state.before.t<N>.json, state.after.t<N>.json  (mutation/version diff)
#   msg.t<N>.req.json, msg.t<N>.resp.json, msg.t<N>.curl_meta
#   progress.t<N>.json                              (composer-progress events post-call)
#   metrics.t<N>.json                               (computed metrics — see notes below)
#   turn<N>.user.txt                                (verbatim copy of input msg)
#
# Heuristic metrics — names suffixed with `_keyword_match` are NOISY signals.
# Treat them as filtering aids, not authoritative judgments. The authoritative
# read comes from reading msg.t<N>.resp.json's .message.content directly.
#
# Exit codes: 0 ok / 64 bad usage / 67 input file missing / pass-through HTTP code via curl_meta

set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_SCRIPT_DIR="$HARNESS_DIR"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
LIB_DIR="$(cd "$HARNESS_ROOT/../lib" && pwd)"
# shellcheck source=../../lib/common.sh
source "$LIB_DIR/common.sh"

if (( $# != 3 )); then
  evals_die 64 "usage: post_message.sh <scenario_id> <turn_index> <user_message_file>"
fi
scenario_id=$1
turn=$2
msg_file=$3

[[ "$turn" =~ ^[0-9]+$ ]] || evals_die 64 "turn_index must be a positive integer (got: $turn)"
[[ -f "$msg_file" ]] || evals_die 67 "user message file not found: $msg_file"

evals_load_env --require-creds
evals_require_tools

# Locate the active run dir. Convention: most recent runs/<date>-hardmode/<scenario_id>/
# unless ELSPETH_EVAL_RUNS_DIR pins a specific date.
runs_root="${ELSPETH_EVAL_RUNS_DIR:-}"
if [[ -z "$runs_root" ]]; then
  # Pick newest dated dir matching *-hardmode/.
  runs_root=$(ls -1d "$HARNESS_ROOT"/runs/*-hardmode 2>/dev/null | sort -r | head -1 || true)
fi
[[ -n "$runs_root" ]] || evals_die 67 "no runs dir found under $HARNESS_ROOT/runs (run harness.sh first or set ELSPETH_EVAL_RUNS_DIR)"

out="$runs_root/$scenario_id"
[[ -d "$out" ]] || evals_die 67 "scenario run dir not found: $out (run harness.sh first)"

export EVALS_OUT_DIR="$out"
export EVALS_LOG_FILE="$out/harness.log"
export EVALS_JWT_FILE="$out/jwt.txt"

[[ -s "$EVALS_JWT_FILE" ]] || evals_die 67 "jwt.txt missing in $out (re-run harness.sh)"

sid=$(cat "$out/sid.txt")
evals_log INFO "post_message scenario=$scenario_id turn=$turn sid=$sid"

# Snapshot the user-side message for audit (separate from the JSON envelope).
# When the caller already wrote the message into the run dir at this exact
# path (the RUNBOOK shows this idiom for turn 1), skip the cp — copying a
# file onto itself is a fatal error under set -e.
target_user_txt="$out/turn${turn}.user.txt"
if [[ "$(readlink -f "$msg_file")" != "$(readlink -f "$target_user_txt")" ]]; then
  cp "$msg_file" "$target_user_txt"
fi

# State before
evals_get_state "$sid" "$out/state.before.t${turn}.json"

# Send the message (writes msg.t<N>.{req,resp,curl_meta})
evals_post_message "$sid" "$turn" "$msg_file"

# Progress events
evals_get_progress "$sid" "$out/progress.t${turn}.json"

# State after
evals_get_state "$sid" "$out/state.after.t${turn}.json"

# Capture the model's pre-synthesis prose for this turn. When the
# empty-state synthesizer at service.py:_finalize_no_tool_response replaces visible content,
# the model's actual final text is preserved server-side in raw_content
# and is invisible to msg.t<N>.resp.json. Without this, eval analysis
# cannot tell whether the model converged on useful output that the
# synthesizer hid (cf. elspeth-861b0c58f5).
evals_get_messages_with_raw "$sid" "$out/messages.with_raw.t${turn}.json"

# Compute metrics. Use python3 — the JSON shapes are too gnarly for jq one-liners.
python3 - "$out" "$turn" "$sid" "$scenario_id" <<'PY'
import json, pathlib, sys, re
out = pathlib.Path(sys.argv[1])
turn = int(sys.argv[2])
sid = sys.argv[3]
scenario_id = sys.argv[4]

def load_json(p, default):
    try:
        text = p.read_text().strip()
        return json.loads(text) if text else default
    except (FileNotFoundError, json.JSONDecodeError):
        return default

resp = load_json(out / f"msg.t{turn}.resp.json", {})
prog = load_json(out / f"progress.t{turn}.json", {})
sb = load_json(out / f"state.before.t{turn}.json", None)
sa = load_json(out / f"state.after.t{turn}.json", None)
messages_with_raw = load_json(out / f"messages.with_raw.t{turn}.json", [])

# Find the most recent assistant message at-or-before this turn's response
# created_at and persist its (id, content, raw_content, created_at) tuple as
# msg.t<N>.raw.json. The "or after" guard catches the race where the API call
# returns before the persist transaction commits — fall back to the latest
# assistant message in the history.
def _pick_assistant_for_turn(messages, resp_payload):
    if not isinstance(messages, list) or not messages:
        return None
    target_id = (resp_payload.get("message") or {}).get("id")
    if target_id:
        for m in messages:
            if isinstance(m, dict) and m.get("id") == target_id and m.get("role") == "assistant":
                return m
    # Fallback: latest assistant message in history
    asst = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    return asst[-1] if asst else None

assistant_with_raw = _pick_assistant_for_turn(messages_with_raw, resp)
raw_artefact = {
    "turn": turn,
    "session_id": sid,
    "scenario_id": scenario_id,
    "id": (assistant_with_raw or {}).get("id"),
    "role": "assistant",
    "content": (assistant_with_raw or {}).get("content"),
    "raw_content": (assistant_with_raw or {}).get("raw_content"),
    "created_at": (assistant_with_raw or {}).get("created_at"),
    "note": (
        "raw_content is the model's pre-synthesis prose; content is what the "
        "user/SPA saw. When raw_content differs from content, the empty-state "
        "synthesizer at service.py:_finalize_no_tool_response intercepted the model's output. "
        "raw_content=null means no synthesis occurred (or no assistant message "
        "was persisted yet)."
    ),
}
(out / f"msg.t{turn}.raw.json").write_text(json.dumps(raw_artefact, indent=2))

# curl_meta is two lines: "<http_code> <time_total>" and a wall-time line.
curl_meta = (out / f"msg.t{turn}.curl_meta").read_text().strip().splitlines()
http_code = curl_meta[0].split()[0] if curl_meta else "?"
wall = float(curl_meta[1]) if len(curl_meta) > 1 else None
http_ok = http_code.isdigit() and 200 <= int(http_code) < 300
turn_status = {
    "status": "ok" if http_ok else "transport_error",
    "http_ok": http_ok,
    "http_code": http_code,
    "transport_error": None if http_ok else f"post_message HTTP {http_code}",
}

events = (prog.get("events") or [])
tool_calls = [e for e in events if e.get("kind") == "tool_call"]
val_errs = [e for e in events if e.get("kind") in ("validation_error", "tool_error")]

ver_before = sb.get("version") if isinstance(sb, dict) else None
ver_after = sa.get("version") if isinstance(sa, dict) else None
mutated = (ver_after is not None) and (ver_before != ver_after)
is_valid_after = sa.get("is_valid") if isinstance(sa, dict) else None

asst_content = (resp.get("message") or {}).get("content") or ""
asst_lower = asst_content.lower()

# Heuristic metrics — explicitly named *_keyword_match to flag they're noisy.
clarifying_keywords = ("could you", "would you", "do you want", "what kind",
                       "which", "clarify", "could i ask", "can you confirm")
limit_keywords = ("can't", "cannot", "unable to", "not supported",
                  "doesn't support", "isn't supported", "won't",
                  "does not support", "not currently")
# Suppress limit-match if the message also says "but I will"/"I can still" etc.
limit_negation_pattern = re.compile(
    r"\b(but|however|that said|still|instead)\b.*\b(i\s+(can|will|am|have|am able))\b",
    re.IGNORECASE | re.DOTALL)

asked_clarifying_keyword_match = ("?" in asst_content) and any(
    k in asst_lower for k in clarifying_keywords)
volunteered_limit_keyword_match = (
    any(k in asst_lower for k in limit_keywords)
    and not limit_negation_pattern.search(asst_content))

# Synthesizer-interception flags: distinguish the two server-side
# augmentation shapes (see service._finalize_no_tool_response):
#
#   synthesizer_replaced  — content does NOT start with raw_content. The
#       model produced a real prose response that the synthesizer hid
#       behind a synthetic [ELSPETH-SYSTEM] blocker. This is *the* signal
#       cohort scoring needs to distinguish "model converged on a useful
#       summary that was hidden" from "model genuinely failed to converge"
#       (cf. elspeth-861b0c58f5 reframing).
#   synthesizer_augmented — content starts with raw_content and is
#       strictly longer. The model's prose is preserved verbatim; the
#       synthesizer appended an operator-facing suffix. The model is not
#       "hidden" but the visible message is not pure model output either.
#   synthesizer_intercepted — union (replaced OR augmented). Retained as a
#       coarse "did synthesis happen at all" signal for diagnostic dumps.
raw_content_value = raw_artefact.get("raw_content")
content_value = raw_artefact.get("content") or ""
# raw_content is only persisted when synthesis happened (cf.
# protocol.ChatMessageRecord.raw_content), so any non-None raw_content
# is a synthesis signal. The shape — startswith vs. not — discriminates
# augment from replace; len > len excludes the no-op equality case
# (synthesis recorded but content unchanged) which is incoherent with
# _compose_empty_state_message but cheap to guard against.
synthesizer_augmented = (
    raw_content_value is not None
    and content_value.startswith(raw_content_value)
    and len(content_value) > len(raw_content_value)
)
synthesizer_replaced = (
    raw_content_value is not None
    and not content_value.startswith(raw_content_value)
)
synthesizer_intercepted = synthesizer_augmented or synthesizer_replaced

metrics = dict(
    turn=turn,
    http_code=http_code,
    turn_status=turn_status,
    wall_seconds=wall,
    tool_call_count=len(tool_calls),
    in_loop_recovery_count=len(val_errs),
    state_version_before=ver_before,
    state_version_after=ver_after,
    mutated_state=mutated,
    is_valid_after=is_valid_after,
    asked_clarifying_keyword_match=asked_clarifying_keyword_match,
    volunteered_limit_keyword_match=volunteered_limit_keyword_match,
    synthesizer_intercepted=synthesizer_intercepted,
    synthesizer_augmented=synthesizer_augmented,
    synthesizer_replaced=synthesizer_replaced,
    raw_content_excerpt=(raw_content_value or "")[:500] if raw_content_value else None,
    assistant_content_excerpt=asst_content[:500],
    note="*_keyword_match metrics are heuristic; verify by reading msg.t{N}.{resp,raw}.json directly.",
)
(out / f"metrics.t{turn}.json").write_text(json.dumps(metrics, indent=2))

turn_manifest = {
    "schema_version": 1,
    "scenario_id": scenario_id,
    "session_id": sid,
    "turn": turn,
    "user_message_path": f"turn{turn}.user.txt",
    "request_path": f"msg.t{turn}.req.json",
    "response_path": f"msg.t{turn}.resp.json",
    "raw_content_path": f"msg.t{turn}.raw.json",
    "curl_meta_path": f"msg.t{turn}.curl_meta",
    "progress_path": f"progress.t{turn}.json",
    "state_before_path": f"state.before.t{turn}.json",
    "state_after_path": f"state.after.t{turn}.json",
    "metrics_path": f"metrics.t{turn}.json",
    "status": turn_status,
    "persona_dispatch": {
        "required": turn >= 2,
        "verified_by_harness": False,
        "evidence": None,
    },
}
(out / f"turn{turn}.manifest.json").write_text(json.dumps(turn_manifest, indent=2))
print(json.dumps(metrics, indent=2))
PY
