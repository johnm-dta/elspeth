from elspeth.web.sessions.engine import create_session_engine


def build_engine(session_db_url: str):
    return create_session_engine(session_db_url)
