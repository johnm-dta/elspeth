# Database Sink Example

Demonstrates writing pipeline output to a relational database using the `database` sink plugin.

## What This Shows

A pipeline reads customer deals from CSV, splits them by value at a gate, and writes high-value deals to a SQLite database while standard deals go to CSV.

```
source ─(validated)─> [value_gate] ─┬─ SQLite database (amount >= 5000)
                                     └─ standard.csv   (amount < 5000)
```

The database sink creates the table automatically on first write.

## Running

```bash
elspeth run --settings examples/database_sink/settings.yaml --execute
```

## Output

- `output/deals.db` — SQLite database with `high_value_deals` table
- `output/standard_deals.csv` — Deals under $5,000

### Querying the Database

```bash
# View the high-value deals
sqlite3 examples/database_sink/output/deals.db "SELECT * FROM high_value_deals"

# Count by category
sqlite3 examples/database_sink/output/deals.db \
  "SELECT category, COUNT(*), SUM(amount) FROM high_value_deals GROUP BY category"
```

## Database Sink Configuration

```yaml
sinks:
  high_value_db:
    plugin: database
    options:
      url: sqlite:///examples/database_sink/output/deals.db
      table: high_value_deals
      schema:
        mode: fixed
        fields:
        - 'id: int'
        - 'customer: str'
        - 'amount: int'
      if_exists: replace    # or "append"
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `url` | Required | SQLAlchemy connection URL |
| `table` | Required | Target table name |
| `if_exists` | `append` | `append` (add rows) or `replace` (drop and recreate) |
| `validate_input` | `false` | Validate rows against schema before insert |

### Supported Databases

Any SQLAlchemy-compatible database:

```yaml
# SQLite (no server needed)
url: sqlite:///./output/data.db

# PostgreSQL
url: postgresql://user:password@localhost/database

# MySQL
url: mysql+pymysql://user:password@localhost/database
```

## Key Concepts

- **Auto-creates table**: Columns derived from schema fields with proper type mapping
- **Type mapping**: `str` → TEXT, `int` → INTEGER, `float` → REAL, `bool` → BOOLEAN
- **Resume support**: On resume, the sink switches to append mode automatically
- **Audit integrity**: Content hash computed before insert (proves what was written)
