from sqlalchemy.dialects import postgresql, sqlite


def write(conn, table, values):
    dialect = conn.dialect.name
    if dialect == "sqlite":
        stmt = sqlite.insert(table).values(**values)
    elif dialect == "postgresql":
        stmt = postgresql.insert(table).values(**values)
    else:
        raise NotImplementedError(dialect)
    return conn.execute(stmt)
