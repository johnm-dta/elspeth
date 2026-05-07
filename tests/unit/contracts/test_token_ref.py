"""Unit tests for TokenRef — bundled token_id + run_id reference."""

from __future__ import annotations

import pytest

from elspeth.contracts.audit import TokenRef


class TestTokenRef:
    """Tests for TokenRef frozen dataclass."""

    def test_construction(self) -> None:
        ref = TokenRef(token_id="tok-1", run_id="run-1")
        assert ref.token_id == "tok-1"
        assert ref.run_id == "run-1"

    def test_frozen(self) -> None:
        ref = TokenRef(token_id="tok-1", run_id="run-1")
        with pytest.raises(AttributeError):
            ref.token_id = "tok-2"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = TokenRef(token_id="tok-1", run_id="run-1")
        b = TokenRef(token_id="tok-1", run_id="run-1")
        assert a == b

    def test_inequality_token(self) -> None:
        a = TokenRef(token_id="tok-1", run_id="run-1")
        b = TokenRef(token_id="tok-2", run_id="run-1")
        assert a != b

    def test_inequality_run(self) -> None:
        a = TokenRef(token_id="tok-1", run_id="run-1")
        b = TokenRef(token_id="tok-1", run_id="run-2")
        assert a != b

    def test_hashable(self) -> None:
        ref = TokenRef(token_id="tok-1", run_id="run-1")
        s = {ref}
        assert ref in s

    def test_repr(self) -> None:
        ref = TokenRef(token_id="tok-1", run_id="run-1")
        r = repr(ref)
        assert "tok-1" in r
        assert "run-1" in r
