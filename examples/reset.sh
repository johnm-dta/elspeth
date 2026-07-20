#!/usr/bin/env bash
# Reset every shipped example to a clean baseline: remove the local audit
# trails and embedded vector stores that accumulate across runs.
#
# ELSPETH is pre-1.0 and does NOT migrate audit databases in place, so a stale
# runs/audit.db written under an earlier schema epoch makes a bare `elspeth run`
# fail with SchemaCompatibilityError. Run this after upgrading ELSPETH, or any
# time you want a fresh baseline. Every removed path is a gitignored artifact.
#
# Usage:
#   ./examples/reset.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Per-example audit trails (the SchemaCompatibilityError culprit).
find . -type d -name runs -prune -print0 | while IFS= read -r -d '' d; do
  rm -f "$d"/*.db "$d"/*.db-wal "$d"/*.db-shm
done

# Embedded ChromaDB vector stores (chroma_rag* examples).
find . -type d -name chroma_data -prune -exec rm -rf {} +

echo "Examples reset: cleared runs/*.db* and chroma_data/ under $SCRIPT_DIR"
