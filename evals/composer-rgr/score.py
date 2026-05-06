"""Score a captured composer-rgr run against scenario criteria.

Usage:
    score.py scenario.json messages.json state.json    # prints scoring JSON

The scoring JSON's `verdict` field is the authoritative result. Exit code
is always 0 — the operator decides what to do with RED.

Detection rules (ordered by reliability):
  1. Build-failure sentinels in final assistant content. These are
     server-injected (service.py:_build_runtime_preflight_message) and
     only appear when the model declared completion but the pipeline
     failed preflight. Most reliable RED signal.
  2. is_valid=false on the final composition state. Independent of
     message content, catches cases where the model gets stuck in
     tool loops without ever surfacing a sentinel.
  3. Passivity phrases in final assistant content. Catches the
     'If you want, I can…' / 'Should I…' rationalisation pattern that
     the skill's anti-passivity section explicitly forbids.

Persisted chat history (`tool_calls` field) is NOT a useful signal —
the composer drops internal tool-call assistant turns before persisting,
so even a successful build shows zero persisted tool calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    scenario = json.loads(Path(sys.argv[1]).read_text())
    messages = json.loads(Path(sys.argv[2]).read_text())
    state = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None

    red = scenario["red_criteria"]
    green = scenario["green_criteria"]

    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    final = assistant_msgs[-1] if assistant_msgs else {"content": ""}
    final_body = (final.get("content") or "").lower()

    red_reasons: list[str] = []

    # 1. Build-failure sentinels.
    sentinel_hits = [s for s in red["build_failure_sentinels"] if s in final_body]
    if sentinel_hits:
        red_reasons.append(f"build-failure sentinel(s) present in final message: {sentinel_hits}")

    # 2. State validity. Treat both is_valid=false AND missing/null state as RED:
    # null state means no mutation ever committed (model failed before producing
    # a saveable composition).
    is_valid = bool(state.get("is_valid")) if isinstance(state, dict) else None
    if red.get("must_be_valid", True):
        if is_valid is False:
            red_reasons.append("final composition state has is_valid=false")
        elif state is None or state == "null":
            red_reasons.append("final composition state is null (no committed pipeline)")

    # 3. Passivity phrases.
    phrase_hits = [p for p in red["passivity_phrases"] if p in final_body]
    if phrase_hits:
        red_reasons.append(f"forbidden passivity phrases in final message: {phrase_hits}")

    # ------------------------------------------------------------------
    # GREEN positive checks (only meaningful when not RED).
    # ------------------------------------------------------------------
    amber_reasons: list[str] = []
    if isinstance(state, dict):
        nodes = state.get("nodes") or []
        node_plugins: list[str] = []
        for n in nodes:
            if isinstance(n, dict):
                node_plugins.append((n.get("plugin") or n.get("type") or "").lower())
        node_blob = " ".join(node_plugins)

        kind_groups = green.get("must_have_node_kinds_substring_any_of") or []
        if kind_groups:
            ok = False
            for group in kind_groups:
                if all(needle.lower() in node_blob for needle in group):
                    ok = True
                    break
            if not ok:
                amber_reasons.append(
                    f"no expected node combo present (need one of {kind_groups}); "
                    f"found node plugins {node_plugins}"
                )

        outputs = state.get("outputs")
        out_count = len(outputs) if isinstance(outputs, (list, dict)) else 0
        min_outputs = green.get("must_have_outputs_min", 0)
        if out_count < min_outputs:
            amber_reasons.append(f"only {out_count} outputs (need >= {min_outputs})")

    verdict = "RED" if red_reasons else ("GREEN" if not amber_reasons else "AMBER")

    out = {
        "verdict": verdict,
        "red_reasons": red_reasons,
        "amber_reasons": amber_reasons,
        "stats": {
            "assistant_message_count": len(assistant_msgs),
            "final_content_chars": len(final_body),
            "final_content_preview": (final.get("content") or "")[:300],
            "is_valid": is_valid,
            "state_node_count": len(state.get("nodes", [])) if isinstance(state, dict) else None,
            "state_output_count": (
                len(state.get("outputs", [])) if isinstance(state, dict) else None
            ),
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
