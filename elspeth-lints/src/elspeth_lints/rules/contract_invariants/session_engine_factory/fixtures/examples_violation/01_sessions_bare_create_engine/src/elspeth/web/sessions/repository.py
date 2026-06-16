from sqlalchemy import create_engine


def build_engine():
    return create_engine("sqlite:///sessions.db")
