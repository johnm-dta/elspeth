from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert


def write(conn, table, values):
    if conn.dialect.name == "sqlite":
        stmt = sqlite_insert(table).values(**values)
    elif conn.dialect.name == "postgresql":
        stmt = postgresql_insert(table).values(**values)
    else:
        raise NotImplementedError(conn.dialect.name)
    return conn.execute(stmt)
