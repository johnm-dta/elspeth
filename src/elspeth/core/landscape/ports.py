"""Landscape persistence port protocols."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol

from sqlalchemy.engine import Connection


class LandscapeConnectionProvider(Protocol):
    """Broad Landscape database connection surface for repositories."""

    @property
    def engine(self) -> Any:
        """The underlying Tier-1 engine for fenced transactions."""
        ...

    def read_only_connection(self) -> AbstractContextManager[Connection]:
        raise NotImplementedError

    def connection(self) -> AbstractContextManager[Connection]:
        raise NotImplementedError

    def write_connection(self) -> AbstractContextManager[Connection]:
        raise NotImplementedError


__all__ = ["LandscapeConnectionProvider"]
