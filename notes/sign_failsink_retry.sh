#!/usr/bin/env bash
#
# sign_failsink_retry.sh — re-justify the 2 failsink guards (Group B #3/#4)
# that the judge BLOCKED on the first pass (flip-flop: it ACCEPTED the identical
# main-path guards #1/#2 @0.90).
#
# Run in your OPERATOR cert shell (holds ELSPETH_JUDGE_METADATA_HMAC_KEY).
# PREREQUISITE: the fp-neutral guard comments are on disk (unstaged) and the
# rationale below now matches the accepted main-path wording ("never coerce").
# No policy change — the systemic guidance fix is tracked as P1 elspeth-48aa57c3ef.
#
# Usage:
#   ./notes/sign_failsink_retry.sh --dry-run   # preview verdict, write nothing
#   ./notes/sign_failsink_retry.sh             # sign
#
# If the judge STILL blocks one (stochastic re-flip), the honest fallback is
# justify --operator-override (records verdict=OVERRIDDEN_BY_OPERATOR + the judge
# dissent; needs ELSPETH_JUDGE_OVERRIDE_TOKEN[_SHA256]) — but try this first.

set -euo pipefail

REPO_ROOT="/home/john/elspeth"
TRANSPORT="openrouter"

if [[ -z "${ELSPETH_JUDGE_METADATA_HMAC_KEY:-}" ]]; then
  echo "ERROR: ELSPETH_JUDGE_METADATA_HMAC_KEY is not set in this shell." >&2
  exit 1
fi

cd "$REPO_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv/bin/python not found under $REPO_ROOT" >&2
  exit 1
fi

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

echo "==> [1/2] sink.py  SinkExecutor.write — failsink write_result type (R5, fp=6bf4f058)"
J --file-path engine/executors/sink.py --rule R5 \
  --symbol 'SinkExecutor.write' --fingerprint 6bf4f058589b3c6f \
  --rationale 'Offensive plugin-contract guard on the failsink write path: a failsink returning a non-SinkWriteResult is a sink-plugin bug. Per Plugin-Ownership, a plugin returning the wrong type must crash (PluginContractViolation), never coerce. The isinstance raises as assert-and-crash, not coercion and not silent recovery.'

echo "==> [2/2] sink.py  SinkExecutor.write — failsink artifact descriptor type (R5, fp=00755313)"
J --file-path engine/executors/sink.py --rule R5 \
  --symbol 'SinkExecutor.write' --fingerprint 00755313e7e139d5 \
  --rationale 'Offensive plugin-contract guard on the failsink write path: a failsink SinkWriteResult whose artifact is not an ArtifactDescriptor is a sink-plugin bug. Per Plugin-Ownership, a plugin returning the wrong type must crash (PluginContractViolation), never coerce. The isinstance raises as assert-and-crash, not coercion and not silent recovery.'

echo
echo "==> Failsink retry done. Full keyed verify (expect exit 0 — all 10 resolved):"
echo "    env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check \\"
echo "      --rules trust_tier.tier_model --root src/elspeth \\"
echo "      --allowlist-dir config/cicd/enforce_tier_model --format json"
