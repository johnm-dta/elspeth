#!/usr/bin/env bash
# Install the no-stash, staged-files pre-commit dispatcher for this checkout.
#
# This is intentionally separate from `pre-commit install`: some ELSPETH
# checkouts set core.hooksPath so the active hook directory is not the default
# `.git/hooks` path, and the dispatcher is part of the project contract rather
# than generated pre-commit boilerplate.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$(git config --get core.hooksPath || git rev-parse --git-path hooks)"
DISPATCHER_SOURCE="${REPO_ROOT}/scripts/git-hooks/pre-commit-dispatcher.sh"
DISPATCHER_TARGET="${HOOKS_DIR}/pre-commit"

if [[ ! -d "$HOOKS_DIR" ]]; then
    mkdir -p "$HOOKS_DIR"
fi

if [[ ! -f "$DISPATCHER_SOURCE" ]]; then
    echo "Error: dispatcher source missing at ${DISPATCHER_SOURCE}" >&2
    echo "       Ensure your checkout is current with scripts/git-hooks/." >&2
    exit 1
fi

cp "$DISPATCHER_SOURCE" "$DISPATCHER_TARGET"
chmod +x "$DISPATCHER_TARGET"

echo "Installed pre-commit dispatcher at: ${DISPATCHER_TARGET}"
echo "Runs: pre-commit run --files <staged paths>"
