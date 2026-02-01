# tests/engine/conftest.py
"""Shared fixtures for engine tests.

Provides common test schemas and configuration used across processor test files.
"""

from typing import ClassVar

import pytest
from pydantic import ConfigDict

from elspeth.contracts import PluginSchema
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


# Shared schema for test plugins
class _TestSchema(PluginSchema):
    """Dynamic schema for test plugins."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


@pytest.fixture(scope="module")
def landscape_db() -> LandscapeDB:
    """Module-scoped in-memory database for engine tests.

    Each test gets a unique run_id via recorder.begin_run(), ensuring
    test isolation while sharing the database connection overhead.
    """
    return LandscapeDB.in_memory()
