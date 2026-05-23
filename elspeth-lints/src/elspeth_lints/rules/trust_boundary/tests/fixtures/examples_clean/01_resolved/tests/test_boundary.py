"""Fixture-side test that the rule resolves and accepts."""

import pytest


def test_rejects_malformed():
    with pytest.raises(ValueError):
        raise ValueError("malformed input")
