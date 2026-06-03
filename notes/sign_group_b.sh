#!/usr/bin/env bash
#
# sign_group_b.sh — Group B tier-model judge signing (4 SinkExecutor.write guards)
#
# Run in your OPERATOR cert shell (holds ELSPETH_JUDGE_METADATA_HMAC_KEY).
#
# These 4 findings live inside SinkExecutor.write, whose AST scope the sink:482
# reify also edits. Their v2 scope_fingerprint therefore binds the CURRENT (post-
# reify) write() body. Per the arm's-length tooling reframe, a later reify change
# that staled these is just a cheap re-sign, not an audit event — so no need to
# wait for the reify review before signing. Sign Group A first (sign_group_a.sh).
#
# Usage:
#   ./notes/sign_group_b.sh --dry-run   # preview, write nothing
#   ./notes/sign_group_b.sh             # sign
#
# Transport openrouter (single-model, temp=0, clean attribution). Needs OPENROUTER_API_KEY.

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

echo "==> [1/4] sink.py  SinkExecutor.write — sink write_result type (R5, fp=8ae24dda)"
J --file-path engine/executors/sink.py --rule R5 \
  --symbol 'SinkExecutor.write' --fingerprint 8ae24dda2209d609 \
  --rationale 'Offensive plugin-contract guard: a sink returning a non-SinkWriteResult is a sink-plugin bug. Per Plugin-Ownership, a plugin returning the wrong type must crash (PluginContractViolation), never coerce. The isinstance raises as assert-and-crash, not silent recovery.'

echo "==> [2/4] sink.py  SinkExecutor.write — artifact descriptor type (R5, fp=333193ed)"
J --file-path engine/executors/sink.py --rule R5 \
  --symbol 'SinkExecutor.write' --fingerprint 333193edcab82132 \
  --rationale 'Offensive plugin-contract guard: a SinkWriteResult whose artifact is not an ArtifactDescriptor is a sink-plugin bug and must crash with PluginContractViolation, never coerce. Same justification class as fp=8ae24dda2209d609.'

echo "==> [3/4] sink.py  SinkExecutor.write — failsink write_result type (R5, fp=6bf4f058)"
J --file-path engine/executors/sink.py --rule R5 \
  --symbol 'SinkExecutor.write' --fingerprint 6bf4f058589b3c6f \
  --rationale 'Offensive plugin-contract guard on the failsink write path: a failsink returning a non-SinkWriteResult is a plugin bug and must crash with PluginContractViolation. Assert-and-crash, not silent recovery.'

echo "==> [4/4] sink.py  SinkExecutor.write — failsink artifact descriptor type (R5, fp=00755313)"
J --file-path engine/executors/sink.py --rule R5 \
  --symbol 'SinkExecutor.write' --fingerprint 00755313e7e139d5 \
  --rationale 'Offensive plugin-contract guard on the failsink write path: a failsink SinkWriteResult whose artifact is not an ArtifactDescriptor is a plugin bug and must crash with PluginContractViolation. Assert-and-crash, not silent recovery.'

echo
echo "==> Group B done. Full keyed verify:"
echo "    env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check \\"
echo "      --rules trust_tier.tier_model --root src/elspeth \\"
echo "      --allowlist-dir config/cicd/enforce_tier_model --format json"
echo "    (Expect exit 0: all 10 of the original findings resolved — 2 reified, 8 signed.)"
