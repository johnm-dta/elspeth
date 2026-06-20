import sqlalchemy as sa


def build_engine(session_db_url: str):
    return sa.create_engine(session_db_url)
