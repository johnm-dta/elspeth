from sqlalchemy import create_engine


def create_session_engine(url: str):
    return create_engine(url)
