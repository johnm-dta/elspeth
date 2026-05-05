#!/usr/bin/env bash
# evals/lib/preflight.sh — doctor checks for eval harnesses.
#
# Usage:
#   evals/lib/preflight.sh                       # use $EVALS_ENV_FILE or auto-detected .env
#   EVALS_ENV_FILE=path/to/.env evals/lib/preflight.sh
#
# Verifies, in order:
#   1. Required CLI tools (jq, python3, curl).
#   2. Required env vars (ELSPETH_EVAL_BASE_URL/USER/PASS).
#   3. Base URL is reachable (HTTP HEAD /api/auth/login should not 5xx).
#   4. Login works (POST /api/auth/login returns 200 + access_token).
#   5. JWT exp claim is decodable.
#   6. /api/catalog is reachable with the JWT (cheap auth-roundtrip).
#
# Exit codes:
#   0 = all checks passed
#   64 = bad env / missing required var
#   69 = missing CLI tool
#   70 = login failure
#   71 = base URL unreachable or post-login API call failed
#
# Safe to run repeatedly. Does not create any sessions.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_SCRIPT_DIR="$SCRIPT_DIR"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

step() { evals_log INFO "preflight: $*"; }

step "checking required CLI tools"
evals_require_tools

step "loading env (creds required)"
evals_load_env --require-creds

step "base URL reachable: $ELSPETH_EVAL_BASE_URL"
http=$(curl -sS -o /dev/null --max-time 10 -w '%{http_code}' \
         "$ELSPETH_EVAL_BASE_URL/api/auth/login" -X OPTIONS || echo "000")
if [[ "$http" == "000" ]]; then
  evals_die 71 "could not connect to $ELSPETH_EVAL_BASE_URL"
fi
# Any 1xx-4xx is fine here (we just need the host to respond).
if [[ "$http" == 5* ]]; then
  evals_die 71 "base URL returns $http on /api/auth/login OPTIONS"
fi

step "logging in (will not be reused — testing only)"
TMP_JWT=$(mktemp -t evals_preflight_jwt.XXXXXX)
trap 'rm -f "$TMP_JWT"' EXIT
EVALS_JWT_FILE="$TMP_JWT" evals_login

step "decoding JWT exp"
remaining=$(evals_jwt_exp_seconds_remaining "$(cat "$TMP_JWT")")
if [[ -z "$remaining" ]]; then
  evals_log WARN "JWT exp claim not decodable — auto-refresh logic will fall back to file-existence checks"
else
  evals_log INFO "JWT exp: ${remaining}s remaining"
fi

step "post-login API call (/api/catalog)"
EVALS_JWT_FILE="$TMP_JWT" _evals_http_get \
  "$ELSPETH_EVAL_BASE_URL/api/catalog" \
  /dev/null

evals_log INFO "all preflight checks passed"
