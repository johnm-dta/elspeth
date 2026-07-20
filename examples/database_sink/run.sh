#!/usr/bin/env bash
# =============================================================================
# Database Sink Example — recoverable exactly-once publication to SQLite
#
# The `database` sink appends to a target table you own and records each
# committed batch in a target-side `_elspeth_*` effect ledger. The runtime
# never creates those tables — you provision them once (seed.py), exactly as
# you would in production. This script provisions them, then runs the pipeline.
#
# Usage:
#   ./examples/database_sink/run.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Clean slate so each run is a fresh 0-baseline demo (safe to delete: outputs +
# the local audit trail; never touches anything tracked).
rm -f examples/database_sink/output/deals.db examples/database_sink/output/deals.db-wal examples/database_sink/output/deals.db-shm
rm -f examples/database_sink/output/standard_deals.csv
rm -f examples/database_sink/runs/audit.db examples/database_sink/runs/audit.db-wal examples/database_sink/runs/audit.db-shm

echo "=== Database Sink Example ==="
echo ""

# --- Provision operator-owned target table + effect ledger ---
echo "Provisioning target table + effect ledger..."
.venv/bin/python examples/database_sink/seed.py
echo ""

# --- Run pipeline ---
echo "Running ELSPETH pipeline..."
echo ""
.venv/bin/elspeth run --settings examples/database_sink/settings.yaml --execute

echo ""
echo "=== Pipeline Complete ==="
echo ""

# --- Show what was published ---
echo "High-value deals published to the database (amount >= 5000):"
.venv/bin/python - <<'PY'
import sqlite3
from pathlib import Path

db = Path("examples/database_sink/output/deals.db")
conn = sqlite3.connect(db)
try:
    deals = conn.execute(
        "SELECT id, customer, amount FROM high_value_deals ORDER BY id"
    ).fetchall()
    for row in deals:
        print(f"  id={row[0]:<3} {row[1]:<16} amount={row[2]}")
    print(f"  ({len(deals)} rows written)")
    (effects,) = conn.execute("SELECT COUNT(*) FROM _elspeth_sink_effects").fetchone()
    print(f"\nEffect ledger: {effects} committed effect(s) recorded (exactly-once recovery marker).")
finally:
    conn.close()
PY

echo ""
echo "Done. Audit trail: examples/database_sink/runs/audit.db"
