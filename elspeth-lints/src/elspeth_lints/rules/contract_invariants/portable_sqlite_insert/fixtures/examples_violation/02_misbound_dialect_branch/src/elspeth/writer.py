from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert


def write_history(conn, table, values):
    dialect = conn.dialect.name
    if dialect == "sqlite":
        stmt = sqlite_insert(table).values(**values)
    elif dialect == "postgresql":
        stmt = sqlite_insert(table).values(**values).on_conflict_do_update(index_elements=["id"])
    else:
        raise NotImplementedError(dialect)
    return conn.execute(stmt)


def write_summary(conn, table, values):
    dialect = conn.dialect.name
    if dialect == "sqlite":
        stmt = sqlite_insert(table).values(**values)
    elif dialect == "postgresql":
        stmt = postgresql_insert(table).values(**values)
    else:
        raise NotImplementedError(dialect)
    return conn.execute(stmt)
