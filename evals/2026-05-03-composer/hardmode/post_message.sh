#!/usr/bin/env bash
# Post a single user message to the composer and record the response + metrics.
#
# Usage: post_message.sh <scenario_id> <turn_index> <user_message_file>
#   - user_message_file is a plain text file containing the user message (NOT JSON-wrapped)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scenario_id=$1
turn_index=$2
msg_file=$3

out=$ROOT/results/$scenario_id
sid=$(cat $out/sid.txt)
J=$(cat $out/jwt.txt)

# Capture state BEFORE the message (for mutation/version diff)
curl -fsS -H "Authorization: Bearer $J" \
  "https://elspeth.foundryside.dev/api/sessions/$sid/state" \
  -o $out/state.before.t${turn_index}.json 2>/dev/null || echo 'null' > $out/state.before.t${turn_index}.json

# Wrap the plain user message into the JSON envelope the API expects
jq -n --rawfile c "$msg_file" '{content:$c}' > $out/msg.t${turn_index}.req.json

# Time the call
start=$(date +%s.%N)
curl -sS -X POST "https://elspeth.foundryside.dev/api/sessions/$sid/messages" \
  -H "Authorization: Bearer $J" -H 'Content-Type: application/json' \
  --data @$out/msg.t${turn_index}.req.json \
  -o $out/msg.t${turn_index}.resp.json \
  -w '%{http_code} %{time_total}\n' > $out/msg.t${turn_index}.curl_meta
end=$(date +%s.%N)
wall=$(awk "BEGIN{printf \"%.2f\", $end - $start}")

# Capture progress (post-call) for tool-call/recovery counts
curl -fsS -H "Authorization: Bearer $J" \
  "https://elspeth.foundryside.dev/api/sessions/$sid/composer-progress" \
  -o $out/progress.t${turn_index}.json 2>/dev/null || echo '{}' > $out/progress.t${turn_index}.json

# Capture state AFTER (for mutation detection)
curl -fsS -H "Authorization: Bearer $J" \
  "https://elspeth.foundryside.dev/api/sessions/$sid/state" \
  -o $out/state.after.t${turn_index}.json 2>/dev/null || echo 'null' > $out/state.after.t${turn_index}.json

# Compute per-turn metrics: tool calls, mutations, version delta
python3 - <<PY
import json, pathlib
out = pathlib.Path("$out")
turn = $turn_index
wall = $wall

resp = json.loads((out/f"msg.t{turn}.resp.json").read_text())
prog = json.loads((out/f"progress.t{turn}.json").read_text())
sb_path = out/f"state.before.t{turn}.json"
sa_path = out/f"state.after.t{turn}.json"
sb = json.loads(sb_path.read_text()) if sb_path.read_text().strip() else None
sa = json.loads(sa_path.read_text()) if sa_path.read_text().strip() else None

events = prog.get("events") or []
tool_calls = [e for e in events if e.get("kind") == "tool_call"]
val_errs = [e for e in events if e.get("kind") in ("validation_error","tool_error")]

ver_before = (sb or {}).get("version") if isinstance(sb, dict) else None
ver_after = (sa or {}).get("version") if isinstance(sa, dict) else None
mutated = (ver_after is not None) and (ver_before != ver_after)

is_valid_after = (sa or {}).get("is_valid") if isinstance(sa, dict) else None

asst_content = (resp.get("message") or {}).get("content") or ""
asst_text_lower = asst_content.lower()
asked_clarifying = "?" in asst_content and any(k in asst_text_lower for k in ("could you","would you","do you want","what kind","which","clarify","could i ask"))
volunteered_limit = any(k in asst_text_lower for k in ("can't","cannot","unable to","not supported","doesn't support","isn't supported","won't","does not support","not currently"))

metrics = dict(
    turn=turn,
    wall_seconds=wall,
    tool_call_count=len(tool_calls),
    in_loop_recovery_count=len(val_errs),
    state_version_before=ver_before,
    state_version_after=ver_after,
    mutated_state=mutated,
    is_valid_after=is_valid_after,
    asked_clarifying_question=asked_clarifying,
    volunteered_limit=volunteered_limit,
    assistant_content_excerpt=asst_content[:500],
)
(out/f"metrics.t{turn}.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
PY
