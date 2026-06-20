from sqlalchemy import create_engine


def build_cache_engine(cache_db_url: str):
    return create_engine(cache_db_url)
