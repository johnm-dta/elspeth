import sqlalchemy.dialects.sqlite


def write(conn, table, values):
    return conn.execute(sqlalchemy.dialects.sqlite.insert(table).values(**values))
