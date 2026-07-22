# Database Sink Example

Demonstrates **recoverable, exactly-once** publication of pipeline output to a
relational database using the `database` sink plugin.

## What This Shows

A pipeline reads customer deals from CSV, splits them by value at a gate, writes
high-value deals to a SQLite database, and sends standard deals to CSV.

```
source ─(validated)─> [value_gate] ─┬─ SQLite database (amount >= 5000)
                                     └─ standard.csv   (amount < 5000)
```

The `database` sink is a durable, exactly-once publisher. It **appends** to a
target table you own and records every committed batch in a target-side
`_elspeth_*` **effect ledger**, so a resumed run never republishes rows it
already wrote. Because that ledger is an operational asset, the sink **never
creates it (or the target table) for you** — you provision both once, up front.
`seed.py` is that provisioning step for this example; in production the
equivalent DDL is part of your deployment.

## Running

```bash
./examples/database_sink/run.sh
```

`run.sh` provisions the target table and effect ledger (`seed.py`), then runs
the pipeline. A plain `elspeth run` will fail preflight until the tables exist —
that is the exactly-once contract, not a bug.

## Output

- `output/deals.db` — SQLite database with:
  - `high_value_deals` — the 4 deals with `amount >= 5000` (ids 1, 4, 6, 7)
  - `_elspeth_sink_effects` — the effect ledger (one committed-effect marker)
- `output/standard_deals.csv` — the 4 deals under $5,000

### Querying the Database

```bash
# High-value deals
sqlite3 examples/database_sink/output/deals.db "SELECT * FROM high_value_deals"

# The recovery ledger — one row per committed publication batch
sqlite3 examples/database_sink/output/deals.db \
  "SELECT effect_id, completed FROM _elspeth_sink_effects"
```

## Database Sink Configuration

```yaml
sinks:
  high_value_db:
    plugin: database
    options:
      url: sqlite:///examples/database_sink/output/deals.db
      table: high_value_deals          # operator-owned target table
      schema:
        mode: fixed
        fields:
        - 'id: int'
        - 'customer: str'
        - 'amount: int'
      if_exists: append                 # exactly-once publication is append-only
      effect_ledger:
        table: _elspeth_sink_effects    # must be a namespaced `_elspeth_` table
        schema_version: 1
        permissions: [select, insert]   # the runtime identity's grants on the ledger
```

### Options

| Option | Required | Description |
|--------|----------|-------------|
| `url` | yes | SQLAlchemy connection URL (SQLite or PostgreSQL) |
| `table` | yes | Operator-provisioned target table (must exist before the run) |
| `if_exists` | `append` | Only `append` is supported — exactly-once publication never drops the target |
| `effect_ledger.table` | yes | Operator-provisioned `_elspeth_*` recovery ledger |
| `effect_ledger.schema_version` | yes | Ledger schema version the sink expects (currently `1`) |
| `effect_ledger.permissions` | yes | The grants the runtime identity holds on the ledger (`select`, `insert`) |

### Provisioning (`seed.py`)

`seed.py` builds two tables into one SQLAlchemy `MetaData` and creates them:

- the target table `high_value_deals` (columns matched by **name** to the
  sink's `fixed` schema), and
- the effect ledger `_elspeth_sink_effects`, built from the sink's own
  `database_effect_ledger_table()` factory so its schema is exactly what the
  preflight requires.

### Supported Databases

Exactly-once markers are transactional, so the effect ledger is supported on
**SQLite** and **PostgreSQL**:

```yaml
# SQLite (no server needed) — this example
url: sqlite:///examples/database_sink/output/deals.db

# PostgreSQL (the production path) — set `url` to your postgresql:// DSN,
# provision the target + `_elspeth_*` ledger, and grant the runtime identity
# SELECT + INSERT on the ledger
url: postgresql://…
```

## Key Concepts

- **Operator-owned tables**: you provision the target and the `_elspeth_*`
  ledger; the runtime only reads their shape and appends — it never issues DDL.
- **Exactly-once recovery**: each committed batch is fenced by an `effect_id`
  in the ledger, so resuming an interrupted run republishes nothing.
- **Append-only**: `replace` is unsupported by design — durable publication
  never drops the target table underneath committed effects.
- **Audit integrity**: a content hash is computed before insert (proves what
  was written).
