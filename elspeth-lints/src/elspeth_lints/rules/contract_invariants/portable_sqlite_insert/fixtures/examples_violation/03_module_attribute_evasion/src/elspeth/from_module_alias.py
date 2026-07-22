from sqlalchemy.dialects import sqlite as sqlite_dialect


def write(conn, table, values):
    return conn.execute(sqlite_dialect.insert(table).values(**values))
