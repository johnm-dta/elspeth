#!/usr/bin/env bash
# Install the commit-msg dispatcher for the Phase 8 B4-r3
# telemetry-backfill cohort-attribution rule.
#
# Why this is a separate install script: the project may set core.hooksPath so
# the active hook directory is not the default `.git/hooks` path. This script
# writes a tiny dispatcher into the real hooks directory; the dispatcher then
# execs the worktree-local `.githooks/commit-msg-telemetry-backfill` script,
# which is the version-controlled enforcement logic.
#
# Fresh-clone setup steps for full local enforcement:
#     1. scripts/git-hooks/install-pre-commit-dispatcher.sh      # pre-commit
#     2. scripts/git-hooks/install-commit-msg-dispatcher.sh      # commit-msg
#
# CI runs an equivalent predicate against PR commits as a backstop, so
# the rule is enforced even for contributors who skip step 2.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$(git config --get core.hooksPath || git rev-parse --git-path hooks)"
DISPATCHER_SOURCE="${REPO_ROOT}/scripts/git-hooks/commit-msg-dispatcher.sh"
DISPATCHER_TARGET="${HOOKS_DIR}/commit-msg"
SPEC_REF='docs/composer/ux-redesign-2026-05/20-phase-8-polish-and-telemetry.md §"Cohort attribution via commit trailers (A4 — load-bearing)"'

if [[ ! -d "$HOOKS_DIR" ]]; then
    mkdir -p "$HOOKS_DIR"
fi

if [[ ! -f "$DISPATCHER_SOURCE" ]]; then
    echo "Error: dispatcher source missing at ${DISPATCHER_SOURCE}" >&2
    echo "       Ensure your checkout is current with the .githooks tree." >&2
    exit 1
fi

cp "$DISPATCHER_SOURCE" "$DISPATCHER_TARGET"
chmod +x "$DISPATCHER_TARGET"

echo "Installed commit-msg dispatcher at: ${DISPATCHER_TARGET}"
echo "Enforces:  ${REPO_ROOT}/.githooks/commit-msg-telemetry-backfill"
echo ""
echo "Spec: ${SPEC_REF}"
