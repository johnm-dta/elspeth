#!/usr/bin/env bash
# Dispatcher: routes the pre-commit hook through pre-commit with the staged
# path list supplied explicitly.
#
# Why this exists: the default pre-commit staged-file isolation can stash and
# restore unstaged changes. The older ELSPETH local hook avoided that by running
# `pre-commit run --all-files`, but that made pre-commit a slow whole-repo gate.
# Passing `--files <staged>` keeps stash=False while preserving the incremental
# pre-commit contract. PR CI remains the full-codebase gate.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
INSTALL_PYTHON="${ELSPETH_PRE_COMMIT_PYTHON:-${REPO_ROOT}/.venv/bin/python3}"

mapfile -d '' -t STAGED_FILES < <(
    git diff --cached --name-only -z --diff-filter=ACMRTUXB
)

if [[ ${#STAGED_FILES[@]} -eq 0 ]]; then
    exit 0
fi

if [[ -x "$INSTALL_PYTHON" ]]; then
    exec "$INSTALL_PYTHON" -mpre_commit run --files "${STAGED_FILES[@]}"
elif command -v pre-commit >/dev/null; then
    exec pre-commit run --files "${STAGED_FILES[@]}"
else
    echo '`pre-commit` not found. Did you forget to install project tooling?' >&2
    exit 1
fi
