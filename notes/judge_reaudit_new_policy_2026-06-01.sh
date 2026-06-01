#!/usr/bin/env bash
# Re-audit the cicd-judge allowlist corpus under the 2026-06-01 policy.
#
# IMPORTANT CORRECTION to the "must re-sign everything" assumption:
#   - Changing the judge system prompt changed JUDGE_POLICY_HASH.
#   - But there is NO CI gate comparing a stored entry's judge_policy_hash to
#     the current one (verified). Each entry's HMAC signature still verifies
#     against the hash it was signed with, so the corpus does NOT break CI.
#   - Therefore a blanket re-sign of every ACCEPTED entry is NOT required and
#     would be gold-plating (CI tooling is IRAP-grade, not Landscape-grade).
#
# What the prompt change DOES warrant: a reaudit sweep under the new policy to
# surface any EXISTING suppression the corrected rules would now decide
# differently — especially other "persisted external config re-read claimed as
# Tier-1/2" entries of the same class as the bug we just fixed. Remediate only
# those divergences (code fix, re-justify, or remove the entry).
#
# reaudit is READ-ONLY (writes no signed metadata). Agent/operator-runnable with
# an OpenRouter key; keep --judge-transport openrouter for deterministic temp=0
# re-checks regardless of how each entry was originally written.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LINTS="env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli"

echo "Re-auditing under policy hash:"
$LINTS --help >/dev/null 2>&1 || true
env PYTHONPATH=elspeth-lints/src .venv/bin/python -c \
  "import elspeth_lints.core.judge as j; print(' ', j.JUDGE_POLICY_HASH)"

echo
echo "Running reaudit (needs OPENROUTER_API_KEY in env; re-runs the judge on every"
echo "judged entry). This costs LLM tokens (~\$5-11 for a full sweep, ~77% cached)."
echo

# env OPENROUTER_API_KEY=sk-or-... \
$LINTS reaudit \
  --root src/elspeth \
  --allowlist-dir config/cicd/enforce_tier_model \
  --judge-transport openrouter \
  --format markdown \
  --output /tmp/reaudit-2026-06-01.md

echo
echo "Report written to /tmp/reaudit-2026-06-01.md"
echo
echo "TRIAGE — bucket every WAS_ACCEPTED_NOW_BLOCKED flip into ONE of three"
echo "(a flip is genuine policy drift at temp=0, NOT sampling noise, so read the code):"
echo "  1. WAS ALWAYS WRONG — the old ACCEPT was a mistake; the guard was redundant"
echo "     from the start.                       -> remediate code (remove guard / let crash)"
echo "  2. STANDARD LEGITIMATELY RAISED — the code was correct under the OLD, laxer"
echo "     policy, but the bar deliberately moved; the same code must now change to"
echo "     meet the stricter standard. The judge is right AND the old verdict was"
echo "     right-at-the-time. (no-legacy: previously-fine code changing is the POINT)"
echo "                                            -> remediate code to meet the new bar"
echo "  3. JUDGE OVER-CORRECTING — the new rationale misreads the data's origin or the"
echo "     boundary; a genuine false BLOCK.       -> keep the code; if 3s CLUSTER, the"
echo "                                               prompt over-rotated -> dampening tweak"
echo "  4. CODE CORRECT, RATIONALE DISHONEST/MIS-TARGETED — the guard is fine (e.g. it"
echo "     DOES quarantine), but the recorded rationale argues the wrong point or names"
echo "     a pattern the analyzer didn't flag, so the audit record is misleading. The"
echo "     BLOCK is correct (an allowlist entry is an audit record)."
echo "                                            -> re-justify with honest wording; NO code change"
echo
echo "For the rarer WAS_BLOCKED_NOW_ACCEPTED flips (none seen so far; the shift is"
echo "uniformly stricter — but the persisted-config clarification CAN legitimately"
echo "create space, so check):"
echo "  5. OLD POLICY OVER-SWEPT — the old BLOCK was wrongly strict; the clarified"
echo "     policy rightly makes room (e.g. a guard on persisted Tier-3 config the old"
echo "     reading dismissed as 'redundant on Tier-1 data'). The new ACCEPT is correct;"
echo "     the previously-demanded code change was never needed. (mirror of #2)"
echo "                                            -> accept; no code change"
echo "  6. JUDGE UNDER-CORRECTING — a false ACCEPT (bad twin of #5). The guard really"
echo "     is redundant/incoherent but the new prompt now waves it through."
echo "                                            -> the ACCEPT is wrong; investigate; if 6s"
echo "                                               cluster the prompt loosened too far"
echo "  Reading the code separates 1+2 (remediate) from 3 (push back). A uniformly"
echo "  one-directional ACCEPTED->BLOCKED shift is what raising a standard SHOULD do —"
echo "  high flip count is not by itself evidence of over-correction. Watch the C3"
echo "  override-rate gate for systemic 3-clustering."
echo "  - Entries still ACCEPTED (STILL_AGREES) need no action: old signature valid;"
echo "    re-stamp is optional freshness, not required."
echo
echo "  Helper: list+bucket the flips with code locations via"
echo "    .venv/bin/python notes/reaudit_flip_triage.py <run_id>"
echo
echo "If interrupted: resume with --resume <run_id>, or inspect a partial report"
echo "with --render-incomplete <run_id>."
