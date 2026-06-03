#!/usr/bin/env bash
# Tier-model allowlist reconciliation for the 2026-06-01 trust-tier fix.
#
# Covers:
#   - generation.py fingerprint rotations (AST line-shift from the Finding-1 fix:
#     ~13 stale fps on byte-identical functions below the inserted code), AND
#   - Finding 3 (stale `_declared_field_name` entries in source_inspection.py) —
#     `rotate` reconciles the WHOLE allowlist-dir, so both are handled in one pass.
#   - the ONE genuinely-new R6 (the get_raw_schema fall-through catch in
#     compute_proof_diagnostics) which `rotate` cannot create — it needs `justify`.
#
# Run from the repo root. Steps 1-2 and 4 are read-only/no-write. Steps 3, 5b
# WRITE — read their env requirements first. Do NOT run blind.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LINTS="env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli"
ROOT="src/elspeth"
ALLOWDIR="config/cicd/enforce_tier_model"

echo "############################################################"
echo "# STEP 1 (read-only): current tier-model state as JSON"
echo "#   - stale_allowlist_entries: entries whose fingerprint no longer matches"
echo "#   - uncovered findings: live findings with no allowlist entry"
echo "############################################################"
$LINTS check --rules trust_tier.tier_model --root "$ROOT" --allowlist "$ALLOWDIR" --format json \
  > /tmp/tier_model_before.json || true
echo "Wrote /tmp/tier_model_before.json"
echo "generation.py + source_inspection.py mentions:"
grep -o '"[^"]*\(generation\.py\|source_inspection\.py\)[^"]*"' /tmp/tier_model_before.json | sort -u | head -60 || true

echo
echo "############################################################"
echo "# STEP 2 (read-only): rotation dry-run — review BEFORE applying"
echo "#   Expect: ~13 generation.py fp rotations (line-shift), plus any"
echo "#   stale _declared_field_name rotations. Confirm none are surprising"
echo "#   (a rotation pairs an old fp to a new fp for the same file:rule:symbol)."
echo "############################################################"
$LINTS rotate --root "$ROOT" --allowlist-dir "$ALLOWDIR" --dry-run

echo
echo "############################################################"
echo "# STEP 3 (WRITES the allowlist YAML; NO HMAC key needed — rotation"
echo "# preserves existing signed metadata, it only repoints fingerprints):"
echo "#   Uncomment to apply once the dry-run looks right."
echo "############################################################"
# $LINTS rotate --root "$ROOT" --allowlist-dir "$ALLOWDIR"
echo "(step 3 commented out — uncomment in the script to apply)"

echo
echo "############################################################"
echo "# STEP 4 (read-only): find the genuinely-NEW uncovered R6 fingerprint"
echo "# on compute_proof_diagnostics (the get_raw_schema fall-through catch)."
echo "# rotate cannot create this entry; you must justify it. Capture its fp:"
echo "############################################################"
$LINTS check --rules trust_tier.tier_model --root "$ROOT" --allowlist "$ALLOWDIR" --format json \
  > /tmp/tier_model_after_rotate.json || true
echo "Uncovered generation.py findings (look for R6 / compute_proof_diagnostics):"
grep -o '"[^"]*generation\.py[^"]*compute_proof_diagnostics[^"]*"' /tmp/tier_model_after_rotate.json | sort -u || true
echo
echo ">>> Set NEW_FP to the new R6 fingerprint shown above (bare hex or fp=<hex>):"
echo ">>>   NEW_FP=<paste-here>"

echo
echo "############################################################"
echo "# STEP 5a (read-only verdict preview; needs OPENROUTER_API_KEY, NO HMAC,"
echo "# does NOT write): see what the judge says before committing to the write."
echo "############################################################"
RATIONALE='source.options is composer/operator-authored pipeline config re-read from persisted session state — Tier-3 origin, not Tier-1. get_raw_schema_config can raise ValueError on a drifted/hand-edited persisted schema block; the catch records a per-blob blocking diagnostic (quarantine) and falls through with schema_config=None so the existing is-not-None guard skips dependent analysis. This is the honest Tier-3 quarantine pattern (record-and-divert), not silent recovery, and prevents an unhandled ValueError from crashing preview_pipeline. Mirrors the existing field_mapping/derive_* catches in the same function.'
cat <<'EONOTE'
Run (fill NEW_FP first):

  env OPENROUTER_API_KEY=sk-or-... PYTHONPATH=elspeth-lints/src \
    .venv/bin/python -m elspeth_lints.core.cli justify \
      --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
      --file-path web/composer/tools/generation.py \
      --symbol compute_proof_diagnostics \
      --fingerprint "$NEW_FP" \
      --rationale "$RATIONALE" \
      --owner "$USER" \
      --dry-run
EONOTE

echo
echo "############################################################"
echo "# STEP 5b (WRITES signed allowlist metadata; needs BOTH"
echo "# ELSPETH_JUDGE_METADATA_HMAC_KEY and OPENROUTER_API_KEY): drop --dry-run."
echo "# OPERATOR-ONLY per HMAC custody. Only run if 5a's verdict is ACCEPTED."
echo "############################################################"
cat <<'EONOTE'
  env ELSPETH_JUDGE_METADATA_HMAC_KEY=<32+byte-secret> OPENROUTER_API_KEY=sk-or-... \
    PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli justify \
      --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
      --file-path web/composer/tools/generation.py \
      --symbol compute_proof_diagnostics \
      --fingerprint "$NEW_FP" \
      --rationale "$RATIONALE" \
      --owner "$USER"

Then re-run STEP 1's check (exit 0 = generation.py gate clean) and run the
full gate: env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
  check --rules trust_tier.tier_model --root src/elspeth --allowlist config/cicd/enforce_tier_model
EONOTE
