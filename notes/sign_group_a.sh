#!/usr/bin/env bash
#
# sign_group_a.sh — Group A tier-model judge signing (4 reify-independent entries)
#
# Run in your OPERATOR cert shell (the only environment that holds
# ELSPETH_JUDGE_METADATA_HMAC_KEY). An agent never runs this.
#
# These 4 findings live in scopes UNTOUCHED by the pending sink:482 reify, so
# their v2 scope_fingerprint binds stable code. (The 4 SinkExecutor.write
# entries are Group B — sign those only after the reify is reviewed + committed.)
#
# Usage:
#   ./notes/sign_group_a.sh              # sign for real
#   ./notes/sign_group_a.sh --dry-run    # preview judge verdict + entry, write nothing
#
# Transport: --judge-transport agent (Claude Agent SDK, the cheaper-via-subscription
# path). Needs the [judge-agent] extra AND Claude auth in this shell. If a call
# errors on claude_agent_sdk/auth, change TRANSPORT below to "openrouter"
# (needs OPENROUTER_API_KEY).

set -euo pipefail

REPO_ROOT="/home/john/elspeth"
# openrouter: temperature=0, single served model, reproducible, clean attribution
# bound into the signed payload. (The 'agent' transport failed here: the claude_code
# preset's background small-fast model makes ResultMessage.model_usage report two
# models, which the judge's single-served-model contract refuses. Needs OPENROUTER_API_KEY.)
TRANSPORT="openrouter"

if [[ -z "${ELSPETH_JUDGE_METADATA_HMAC_KEY:-}" ]]; then
  echo "ERROR: ELSPETH_JUDGE_METADATA_HMAC_KEY is not set in this shell." >&2
  echo "       This script signs audit metadata and must run in your cert shell." >&2
  exit 1
fi

cd "$REPO_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv/bin/python not found under $REPO_ROOT" >&2
  exit 1
fi

# Pass --dry-run (or any extra flags) straight through to every justify call.
EXTRA_ARGS=("$@")

J() {
  PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli justify \
    --root src/elspeth \
    --allowlist-dir config/cicd/enforce_tier_model \
    --judge-transport "$TRANSPORT" \
    --owner "$USER" \
    "${EXTRA_ARGS[@]}" \
    "$@"
}

echo "==> [1/4] recovery.py:110  IncompleteTokenSpec.__post_init__ (R5, fp=427a1fed)"
J --file-path core/checkpoint/recovery.py --rule R5 \
  --symbol 'IncompleteTokenSpec.__post_init__' --fingerprint 427a1fed4efd849c \
  --rationale 'Tier-1 identity guard. IncompleteTokenSpec is built exclusively from DB row reads (get_incomplete_tokens_by_row, a SQLAlchemy fetchall over the tokens table). token_id and row_id are the fundamental audit-identity fields: every audit record references them, and token_data_ref is dereferenced as a payload-store key before any TokenInfo guard exists. A non-str value read from a corrupt or tampered Tier-1 row must crash here (crash on any anomaly; serialization does not change trust tier). The isinstance raises TypeError as offensive detect-and-crash; it does not branch into silent recovery.'

echo "==> [2/4] recovery.py:117  IncompleteTokenSpec.__post_init__ (R5, fp=0b8d412d)"
J --file-path core/checkpoint/recovery.py --rule R5 \
  --symbol 'IncompleteTokenSpec.__post_init__' --fingerprint 0b8d412d4c743e6e \
  --rationale 'Same Tier-1 construction site as fp=427a1fed4efd849c, the optional-lineage-field branch. The optional fields (branch_name, fork_group_id, join_group_id, expand_group_id, token_data_ref) are either NULL (legitimately not applicable) or a non-empty str; an empty or wrong-typed value from a corrupt DB row is anomalous, and token_data_ref is dereferenced as a payload-store key. The is-not-None-gated isinstance raises TypeError as an offensive Tier-1 invariant assertion, not silent recovery.'

echo "==> [3/4] state_guard.py:270  NodeStateGuard._extract_audit_evidence_context (R5, fp=86d1554e)"
J --file-path engine/executors/state_guard.py --rule R5 \
  --symbol 'NodeStateGuard._extract_audit_evidence_context' --fingerprint 86d1554e15f3f1f6 \
  --rationale 'ADR-010 Decision-1 nominal dispatch check: a class must explicitly inherit AuditEvidenceBase to contribute structured audit context, a deliberate nominal (not structural-Protocol) design chosen to prevent accidental-match spoofing. A non-AuditEvidenceBase exception has no structured evidence to contribute; returning (None, None) is honest absence, not a swallowed error. The raw exception is still fully recorded as ExecutionError(exception=str(exc_val), ...) and the FAILED node_state is durably persisted before __exit__ returns. This line is distinct from the inner Mapping guard (fp=81df) and broad-except (fp=d9eb) entries already signed for this method.'

echo "==> [4/4] sink.py:163  SinkExecutor._best_effort_cleanup (R4, fp=1b066df6)"
J --file-path engine/executors/sink.py --rule R4 \
  --symbol 'SinkExecutor._best_effort_cleanup' --fingerprint 1b066df6010227f2 \
  --rationale 'Single broad-except inside the canonical post-audit cleanup helper. It re-raises contract_errors.TIER_1_ERRORS first (audit corruption during cleanup outranks the original error) and only logs genuinely-ignorable cleanup failures, so a crashing system does not strand node_states permanently OPEN, which would itself be a Tier-1 violation. Honest fault isolation matching the accepted engine/_best_effort.py:R4 pattern; logger is the last-resort channel per the primacy order when the cleanup ceremony itself is failing.'

echo
echo "==> Group A done."
echo "    Verify keyed:  env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check \\"
echo "                     --rules trust_tier.tier_model --root src/elspeth \\"
echo "                     --allowlist-dir config/cicd/enforce_tier_model --format json \\"
echo "                     --files core/checkpoint/recovery.py --files engine/executors/state_guard.py \\"
echo "                     --files engine/executors/sink.py"
echo "    (Expect: the 4 Group A findings gone; the 4 SinkExecutor.write findings remain until Group B.)"
