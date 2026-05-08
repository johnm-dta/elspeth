#!/usr/bin/env bash
# evals/composer-harness/hardmode/judge_persona.sh — Channel 2 LLM judge.
#
# Usage: judge_persona.sh <run_dir> [--skip-if-no-key]
#
# Calls Claude Haiku 4.5 (via OpenRouter or Anthropic API, whichever has a
# key in env) with the persona spec + full transcript and asks whether the
# user persona stayed in character. Returns structured JSON.
#
# Why this is complementary to Channel 1 (validate_drift.sh):
#   * Channel 1 catches snake_case echoes, MCP-tool name adoption, plugin-kind
#     adoption — high precision, low recall.
#   * Channel 2 catches:
#       - Marcus adopting the composer's CORRECT meanings of "schema",
#         "trigger", "webhook", "field mapping" (his pseudo-technical vocab
#         retains its tokens but loses its erroneous semantics — Channel 1
#         can't see this).
#       - Linda producing confident architectural sentences without using
#         a single banned token (the "the part that handles the routing"
#         construction — paraphrasing system components into English).
#       - Sarah dropping the narrative research-question voice for terse
#         business-mode replies.
#       - Dev becoming polite, hedge-y, or business-justification-y.
#
# Auth: reads OPENROUTER_API_KEY or ANTHROPIC_API_KEY from env (auto-detect).
#
# Cost: ~$0.001-$0.003 per run on Haiku 4.5 (one shot; persona spec ~200
# lines + transcript a few turns). Negligible in cohort budgets.
#
# Output: writes <run_dir>/judge.json with structured verdict.
#
# Exit codes:
#   0   judge.json written (verdict is data, not pass/fail)
#   3   skipped (--skip-if-no-key set and no key in env)
#   64  usage error
#   67  prerequisite missing
#   71  API call failed after retries
#   72  API returned malformed JSON

set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
LIB_DIR="$(cd "$HARNESS_ROOT/../lib" && pwd)"
# shellcheck source=../../lib/common.sh
source "$LIB_DIR/common.sh"

skip_if_no_key=0
if (( $# < 1 )); then
  evals_die 64 "usage: judge_persona.sh <run_dir> [--skip-if-no-key]"
fi
run_dir=$1
shift
while (( $# > 0 )); do
  case $1 in
    --skip-if-no-key) skip_if_no_key=1; shift;;
    *) evals_die 64 "unknown arg: $1";;
  esac
done

[[ -d "$run_dir" ]] || evals_die 67 "run dir not found: $run_dir"
[[ -s "$run_dir/scenario.json" ]] || evals_die 67 "scenario.json missing in $run_dir"

evals_require_tools

persona_id=$(jq -r '.persona' "$run_dir/scenario.json")
persona_spec="$HARNESS_ROOT/personas/${persona_id}.md"
[[ -f "$persona_spec" ]] || evals_die 67 "persona spec not found: $persona_spec"

# Auth probe — pick the first key that's set.
api_provider=""
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  api_provider="openrouter"
elif [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  api_provider="anthropic"
fi

if [[ -z "$api_provider" ]]; then
  if (( skip_if_no_key )); then
    cat > "$run_dir/judge.json" <<'JSON'
{
  "skipped": true,
  "reason": "no OPENROUTER_API_KEY or ANTHROPIC_API_KEY in env; --skip-if-no-key was set",
  "in_character": null,
  "confidence": null,
  "drift_events": [],
  "rationale": "Channel 2 not run; rely on Channel 1 (drift.json) only for this run."
}
JSON
    echo "judge_persona: skipped (no API key)" >&2
    exit 3
  fi
  evals_die 67 "no OPENROUTER_API_KEY or ANTHROPIC_API_KEY in env (pass --skip-if-no-key to soft-skip)"
fi

python3 - "$run_dir" "$persona_spec" "$api_provider" <<'PY'
"""Channel 2 LLM judge.

Builds a prompt from the persona spec + transcript, calls Haiku 4.5,
parses JSON, writes judge.json.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

run_dir = pathlib.Path(sys.argv[1])
spec_path = pathlib.Path(sys.argv[2])
provider = sys.argv[3]


# ---------- Build transcript ----------
def load_composer_response(turn_n: int) -> str | None:
    p = run_dir / f"msg.t{turn_n}.resp.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    msg = d.get("message")
    if not isinstance(msg, dict):
        return None
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, dict):
                txt = seg.get("text") or seg.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts)
    return None


turn_files = sorted(
    run_dir.glob("turn*.user.txt"),
    key=lambda p: int(re.search(r"turn(\d+)", p.name).group(1)),
)

transcript_lines: list[str] = []
for tf in turn_files:
    n = int(re.search(r"turn(\d+)", tf.name).group(1))
    transcript_lines.append(f"[Turn {n} — USER]:\n{tf.read_text().rstrip()}\n")
    composer_text = load_composer_response(n)
    if composer_text:
        # Truncate composer responses heavily — they are 5-50× the user text and
        # the judge primarily needs the gist (what concepts did the composer
        # introduce that the user might have absorbed). 4000 chars per turn is
        # plenty of context without exploding the prompt.
        if len(composer_text) > 4000:
            composer_text = composer_text[:4000] + "\n...[truncated for judge prompt]..."
        transcript_lines.append(f"[Turn {n} — COMPOSER]:\n{composer_text}\n")

transcript = "\n".join(transcript_lines)
spec_text = spec_path.read_text()

# ---------- Prompt ----------
SYSTEM_PROMPT = """You are a critic evaluating whether a USER PERSONA in a chat transcript \
stayed in character. You read a persona spec (which describes the persona's \
voice, vocabulary, and competence ceiling) and the actual transcript, then \
judge whether the user's turns drifted out of character.

Your judgement should be precise. Drift takes several forms:

1. VOCABULARY DRIFT — the persona uses technical terminology they shouldn't \
have, especially when those terms first appeared in the composer's previous \
response. Channel 1 of the drift detector catches snake_case identifiers; \
your job is to catch English-language drift Channel 1 misses.

2. SEMANTIC DRIFT — the persona uses pseudo-technical vocabulary they would \
naturally use, but with the COMPOSER'S CORRECT MEANING rather than the \
persona's original mistaken meaning. (Example: Marcus says "schema" meaning \
"the column list of my CSV". If the composer corrects him and explains schemas \
are typed contracts, and Marcus then uses "schema" with the new typed-contract \
meaning, that is semantic drift even though no new token appears.)

3. TONAL DRIFT — the persona's voice changes. A terse engineer becoming polite. \
A narrative researcher becoming terse-imperative. A hedge-driven compliance \
officer becoming direct. An assertive marketing-ops manager becoming deferential.

4. COMPETENCE DRIFT — the persona suddenly demonstrates understanding above their \
ceiling. They paraphrase a system component as a noun ("the routing rule", \
"the validation step", "the part that does the conversion") rather than \
describing outcomes ("flagged entries kept separate"). Naming a system component, \
even via English alias, is competence drift for amateur personas.

5. VOICE DRIFT — the persona drops their characteristic moves (Linda's hedges, \
Sarah's narrative framing, Marcus's stack shibboleths like HubSpot/Zapier, \
Dev's terseness and pushback).

For each drift event, cite:
- turn: 1-indexed turn number
- kind: vocabulary | semantic | tonal | competence | voice
- evidence: a verbatim quote from the user turn (max 200 chars)
- severity: minor (single phrase) | moderate (sustained for one turn) | severe (sustained drift across turns)

Output JSON ONLY. No prose before or after. Schema:
{
  "in_character": boolean,
  "confidence": number between 0 and 1,
  "drift_events": [
    {"turn": int, "kind": string, "evidence": string, "severity": string}
  ],
  "rationale": string (1-3 sentences explaining the verdict)
}

If the persona has competence_ceiling: **none** (Dev), the vocabulary, semantic, \
and competence categories are not applicable — focus on tonal and voice drift only.
"""

USER_PROMPT = f"""PERSONA SPEC:
=============
{spec_text}

TRANSCRIPT:
===========
{transcript}

Judge whether the user stayed in character. Output JSON only."""


# ---------- API call ----------
def call_openrouter(api_key: str) -> str:
    body = {
        "model": "anthropic/claude-haiku-4.5",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "temperature": 0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://elspeth.foundryside.dev",
            "X-Title": "elspeth-composer-harness-judge",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"]


def call_anthropic(api_key: str) -> str:
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1500,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": USER_PROMPT}],
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    # Anthropic returns content as a list of content blocks.
    blocks = d.get("content", [])
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    return "".join(parts)


def call_with_retry(provider: str, api_key: str, max_tries: int = 3) -> str:
    last_err: Exception | None = None
    for attempt in range(1, max_tries + 1):
        try:
            if provider == "openrouter":
                return call_openrouter(api_key)
            return call_anthropic(api_key)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = exc
            sys.stderr.write(f"judge_persona: API attempt {attempt} failed: {exc}\n")
            if attempt < max_tries:
                import time
                time.sleep(2 ** attempt)
    raise SystemExit(f"judge_persona: API failed after {max_tries} attempts: {last_err}")


api_key = (
    os.environ.get("OPENROUTER_API_KEY") if provider == "openrouter"
    else os.environ.get("ANTHROPIC_API_KEY")
)
assert api_key, f"provider={provider} but no key in env (caller should have caught this)"

raw_response = call_with_retry(provider, api_key)

# ---------- Parse JSON ----------
# The judge is asked for JSON-only output. Be defensive about markdown
# code fences if the model adds them anyway.
parsed_text = raw_response.strip()
fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", parsed_text, re.DOTALL)
if fence_match:
    parsed_text = fence_match.group(1)

try:
    verdict = json.loads(parsed_text)
except json.JSONDecodeError as exc:
    (run_dir / "judge.json").write_text(json.dumps({
        "error": f"malformed JSON from judge: {exc}",
        "raw_response": raw_response,
        "provider": provider,
    }, indent=2))
    sys.stderr.write(f"judge_persona: malformed JSON from judge — see judge.json\n")
    sys.exit(72)

# ---------- Annotate and write ----------
verdict["_meta"] = {
    "provider": provider,
    "model": (
        "anthropic/claude-haiku-4.5" if provider == "openrouter"
        else "claude-haiku-4-5-20251001"
    ),
    "run_dir": str(run_dir),
    "persona": spec_path.stem,
    "turn_count": len(turn_files),
}

(run_dir / "judge.json").write_text(json.dumps(verdict, indent=2))

# Print a tight summary on stdout for harness consumers.
summary = {
    "persona": spec_path.stem,
    "in_character": verdict.get("in_character"),
    "confidence": verdict.get("confidence"),
    "drift_events": len(verdict.get("drift_events", [])),
    "rationale": verdict.get("rationale", "")[:200],
}
print(json.dumps(summary, indent=2))
PY
