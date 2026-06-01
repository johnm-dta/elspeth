#!/usr/bin/env bash
# Re-audit the cicd-judge allowlist corpus under the 2026-06-01 policy,
# using the CLAUDE AGENT SDK transport (--judge-transport agent) instead of
# OpenRouter. Functionally identical to judge_reaudit_new_policy_2026-06-01.sh;
# only the LLM transport differs.
#
# TRANSPORT DIFFERENCE (openrouter vs agent) — read before running:
#   - Auth: the agent transport does NOT use OPENROUTER_API_KEY. It needs
#     (a) the [judge-agent] extra installed:
#           uv pip install -e 'elspeth-lints/[judge-agent]'
#     AND (b) Claude Code auth available in the environment — ONE of:
#           * a logged-in Claude Code CLI / Claude-subscription or Agent-SDK
#             credit pool, OR
#           * ANTHROPIC_API_KEY, OR
#           * Bedrock / Vertex / Azure credentials.
#     (claude_code system-prompt preset, no tools.)
#   - Reproducibility: the agent transport CANNOT pin temperature, so its
#     verdicts are LESS reproducible than the openrouter temperature=0 path.
#     For deterministic re-checks/decay sweeps prefer the openrouter script;
#     use this one when you specifically want the Agent-SDK to serve verdicts.
#   - Persisted/signed value: entries judged this way record
#     judge_transport = "claude_agent_sdk" (vs "openrouter").
#   - Cost: the "agent is cheaper" assumption holds ONLY on the subscription /
#     credit path. ANTHROPIC_API_KEY is per-token Anthropic billing and may NOT
#     be cheaper than OpenRouter.
#
# IMPORTANT (same as the openrouter script): there is NO CI gate comparing a
# stored entry's judge_policy_hash to the current one, and each entry's HMAC
# signature still verifies against the hash it was signed with — so the corpus
# does NOT break CI and a blanket re-sign is NOT required. reaudit is READ-ONLY
# (writes no signed metadata); it needs no HMAC key.
#
# What this surfaces: any EXISTING suppression the corrected/role-clarified
# policy would now decide differently. Remediate divergences (code fix,
# re-justify, or remove the entry) — do not re-sign wholesale.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LINTS="env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli"

echo "Re-auditing (Claude Agent SDK transport) under policy hash:"
env PYTHONPATH=elspeth-lints/src .venv/bin/python -c \
  "import elspeth_lints.core.judge as j; print(' ', j.JUDGE_POLICY_HASH)"

echo
echo "Transport: claude_agent_sdk (needs the [judge-agent] extra + Claude Code"
echo "auth: logged-in CLI / subscription / ANTHROPIC_API_KEY / Bedrock-Vertex-Azure)."
echo "Re-runs the judge on every judged entry; temperature is NOT pinned."
echo

$LINTS reaudit \
  --root src/elspeth \
  --allowlist-dir config/cicd/enforce_tier_model \
  --judge-transport agent \
  --format markdown \
  --output /tmp/reaudit-agent-2026-06-01.md

echo
echo "Report written to /tmp/reaudit-agent-2026-06-01.md"
echo
echo "TRIAGE — bucket every WAS_ACCEPTED_NOW_BLOCKED flip into ONE of these"
echo "(a flip at the judge level is genuine policy drift, NOT sampling noise —"
echo "though note: the agent transport's UNPINNED temperature means a borderline"
echo "entry can flip on model variance alone, so confirm a flip reproduces before"
echo "treating it as policy drift; for deterministic confirmation re-run the"
echo "openrouter (temp=0) script on the flipped subset):"
echo "  1. WAS ALWAYS WRONG          -> remediate code (guard redundant from the start)"
echo "  2. STANDARD LEGITIMATELY RAISED -> remediate code (was fine before; bar moved)"
echo "  3. JUDGE OVER-CORRECTING     -> keep code; if 3s cluster, dampen the prompt"
echo "  4. CODE CORRECT, RATIONALE DISHONEST/MIS-TARGETED -> re-justify wording; NO code change"
echo "  5. OLD POLICY OVER-SWEPT (WAS_BLOCKED_NOW_ACCEPTED) -> accept; no code change"
echo "  6. JUDGE UNDER-CORRECTING (false ACCEPT) -> investigate; if 6s cluster prompt loosened"
echo
echo "  Helper: list+bucket the flips with code locations via"
echo "    .venv/bin/python notes/reaudit_flip_triage.py <run_id>"
echo
echo "If interrupted: resume with --resume <run_id>, or inspect a partial report"
echo "with --render-incomplete <run_id>."
