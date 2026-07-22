from sqlalchemy.dialects import sqlite


def write(conn, table, values):
    return conn.execute(sqlite.insert(table).values(**values))
