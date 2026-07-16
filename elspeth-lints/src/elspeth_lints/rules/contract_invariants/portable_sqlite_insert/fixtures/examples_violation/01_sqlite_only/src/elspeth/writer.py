from sqlalchemy.dialects.sqlite import insert as sqlite_insert


def write(conn, table, values):
    return conn.execute(sqlite_insert(table).values(**values))
