"""Property proofs for deterministic sink-effect identity."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

from elspeth.contracts.sink_effects import SinkEffectMemberCandidate, SinkEffectRole
from elspeth.core.landscape.execution.sink_effect_identity import (
    compute_pipeline_effect_identity,
    resolve_sink_effect_members,
)


@dataclass(frozen=True)
class _Token:
    token_id: str
    row_id: str
    run_id: str = "run-1"
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None


@dataclass(frozen=True)
class _Row:
    row_id: str
    run_id: str
    ingest_sequence: int


@dataclass(frozen=True)
class _Parent:
    token_id: str
    parent_token_id: str
    ordinal: int


class _Source:
    def __init__(self, size: int) -> None:
        self.query = self
        self.tokens = {f"t{i}": _Token(f"t{i}", "r0") for i in range(size)}
        self.rows = {"r0": _Row("r0", "run-1", 0)}

    def get_token(self, token_id: str) -> _Token | None:
        return self.tokens.get(token_id)

    def get_tokens_by_ids(self, token_ids: tuple[str, ...]) -> list[_Token]:
        return [self.tokens[token_id] for token_id in token_ids if token_id in self.tokens]

    def get_token_parents(self, token_id: str) -> list[_Parent]:
        index = int(token_id[1:])
        return [] if index == 0 else [_Parent(token_id, f"t{index - 1}", index % 2)]

    def get_row(self, row_id: str) -> _Row | None:
        return self.rows.get(row_id)


@given(st.integers(min_value=1, max_value=40))
def test_reversed_arrival_converges_on_membership_and_effect(size: int) -> None:
    source = _Source(size)
    candidates = tuple(SinkEffectMemberCandidate(token_id=f"t{i}", row={"index": i}) for i in range(size))
    first_members = resolve_sink_effect_members(source, candidates)
    second_members = resolve_sink_effect_members(source, reversed(candidates))
    assert first_members == second_members
    kwargs = {
        "run_id": "run-1",
        "sink_node_id": "sink-1",
        "role": SinkEffectRole.PRIMARY,
        "sink_config": {"format": "json"},
        "target_config": {"path": "safe/out.json"},
    }
    assert compute_pipeline_effect_identity(members=first_members, **kwargs) == compute_pipeline_effect_identity(
        members=second_members, **kwargs
    )
