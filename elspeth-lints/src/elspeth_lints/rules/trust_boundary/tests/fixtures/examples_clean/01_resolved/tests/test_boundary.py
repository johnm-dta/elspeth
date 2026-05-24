# mypy: ignore-errors
"""Fixture-side test that the rule resolves and accepts."""

import pytest
from good import execute


def test_rejects_malformed():
    with pytest.raises(ValueError):
        execute({"malformed": object()})
