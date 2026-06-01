#!/usr/bin/env bash
# Re-audit the cicd-judge allowlist corpus under the current policy, using the
# CLAUDE AGENT SDK transport (--judge-transport agent) WITH the read-only
# tool-augmented investigation mode (--judge-tools readonly).
#
# AUTH (agent transport does NOT use OPENROUTER_API_KEY) — read before running:
#   (a) the [judge-agent] extra installed:
#         uv pip install -e 'elspeth-lints/[judge-agent]'
#   AND (b) Claude Code auth in the environment — ONE of:
#         * a logged-in Claude Code CLI / Claude-subscription or Agent-SDK
#           credit pool, OR
#         * ANTHROPIC_API_KEY, OR
#         * Bedrock / Vertex / Azure credentials.
#
# INVESTIGATION MODE (--judge-tools readonly) — what it changes:
#   The judge is no longer blind to the static excerpt. It may Read/Grep/Glob
#   to resolve a question the excerpt can't answer ("where does this parameter
#   come from?", "is the audit event recorded before this ceremony?"). This
#   dissolves the "blind BLOCK-PENDING" false-blocks — verdicts a blinded judge
#   raised only because it couldn't see the caller. The judge cites file:line it
#   read in its rationale.
#   Guard (load-bearing): a fail-closed PreToolUse hook confines reads to
#   src/elspeth + config/cicd/enforce_tier_model, denies .env by basename and
#   anything resolving (realpath) outside the scope, and blocks every non-read
#   tool. reaudit is READ-ONLY and never signs — zero CI / corpus impact.
#
# REPRODUCIBILITY (a property, not a reason to switch): the agent transport
#   cannot pin temperature, and investigation adds a second source of variance
#   (which files it chose to read), so a borderline entry can move between runs.
#   Confirm a flip reproduces before treating it as genuine policy drift.
#
# Cost note: the per-entry uncached cost is comparable to the blinded path
#   (~42k uncached tokens; the bulk of a tool run's token total is CACHE READS,
#   not new spend). "Agent is cheaper" holds on the subscription / credit path;
#   ANTHROPIC_API_KEY is per-token Anthropic billing.
#
# Persisted value: entries judged this way record judge_transport =
#   "claude_agent_sdk". (reaudit writes no signed metadata; no HMAC key needed.)
#   There is NO CI gate comparing a stored entry's judge_policy_hash to the
#   current one, so the corpus does NOT break CI and no blanket re-sign is
#   required. What this surfaces: existing suppressions the current policy would
#   now decide differently — remediate via code fix, re-justify, or removal.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LINTS="env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli"

echo "Re-auditing (Claude Agent SDK transport, read-only investigation tools) under policy hash:"
env PYTHONPATH=elspeth-lints/src .venv/bin/python -c \
  "import elspeth_lints.core.judge as j; print(' ', j.JUDGE_POLICY_HASH)"

echo
echo "Transport: claude_agent_sdk + --judge-tools readonly (needs the [judge-agent]"
echo "extra + Claude Code auth: logged-in CLI / subscription / ANTHROPIC_API_KEY /"
echo "Bedrock-Vertex-Azure). The judge may read src/elspeth + the allowlist dir to"
echo "investigate; temperature is NOT pinned and the read path adds variance."
echo

$LINTS reaudit \
  --root src/elspeth \
  --allowlist-dir config/cicd/enforce_tier_model \
  --judge-transport agent \
  --judge-tools readonly \
  --format markdown \
  --output /tmp/reaudit-agent-tools-2026-06-02.md

echo
echo "Report written to /tmp/reaudit-agent-tools-2026-06-02.md"
echo
echo "TRIAGE — bucket every WAS_ACCEPTED_NOW_BLOCKED flip into ONE of these."
echo "(With investigation on, a BLOCK is less likely to be mere blindness — the"
echo "judge could read the callers — so a surviving block is more probative. Still"
echo "confirm a flip reproduces before calling it policy drift; unpinned temp +"
echo "the read path can move a borderline entry on variance alone.)"
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
