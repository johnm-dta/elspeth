"""Hypothesis strategies for Phase 3 compose-loop audit persistence.

Spec §5.5 cancellation-row mapping:

| §5.5 rows | st_cancellation_arrival_time value |
|---|---|
| 1 | before_llm_call |
| 2 | during_llm_call |
| 3 | after_llm_before_tool |
| 4 | during_tool_dispatch |
| 5 | after_tool_before_sync_dispatch |
| 6 | during_run_sync_between_insert_and_commit |
| 7 | during_advisory_lock_acquisition |
| 8 | after_commit_before_response_yielded |
| 9 | after_response_yielded |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hypothesis import strategies as st

ToolCallName = Literal["set_metadata", "get_pipeline_state"]
RedactionPolicy = Literal["manifest", "unknown_response_key"]
FailureInjectionPoint = Literal[
    "none",
    "audit_raises_OperationalError_on_commit",
    "advisory_lock_unavailable",
    "tool_call_cap_exceeded",
    "unknown_response_key",
]
CancellationArrivalTime = Literal[
    "before_llm_call",
    "during_llm_call",
    "after_llm_before_tool",
    "during_tool_dispatch",
    "after_tool_before_sync_dispatch",
    "during_run_sync_between_insert_and_commit",
    "during_advisory_lock_acquisition",
    "after_commit_before_response_yielded",
    "after_response_yielded",
]
SessionState = Literal["empty", "has_prior_state"]

CANCELLATION_ARRIVAL_TIMES: tuple[CancellationArrivalTime, ...] = (
    "before_llm_call",
    "during_llm_call",
    "after_llm_before_tool",
    "during_tool_dispatch",
    "after_tool_before_sync_dispatch",
    "during_run_sync_between_insert_and_commit",
    "during_advisory_lock_acquisition",
    "after_commit_before_response_yielded",
    "after_response_yielded",
)

FAILURE_INJECTION_POINTS: tuple[FailureInjectionPoint, ...] = (
    "none",
    "audit_raises_OperationalError_on_commit",
    "advisory_lock_unavailable",
    "tool_call_cap_exceeded",
    "unknown_response_key",
)


@dataclass(frozen=True, slots=True)
class ToolCallSpec:
    name: ToolCallName
    call_id: str


def st_tool_call() -> st.SearchStrategy[ToolCallSpec]:
    return st.builds(
        ToolCallSpec,
        name=st.sampled_from(("set_metadata", "get_pipeline_state")),
        call_id=st.from_regex(r"call_[a-z0-9_]{1,16}", fullmatch=True),
    )


def st_argument_dict() -> st.SearchStrategy[dict[str, object]]:
    safe_text = st.text(
        alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters=("\x00",)),
        min_size=1,
        max_size=32,
    )
    return st.one_of(
        st.just({}),
        st.fixed_dictionaries({"patch": st.fixed_dictionaries({"name": safe_text})}),
    )


def st_redaction_policy() -> st.SearchStrategy[RedactionPolicy]:
    return st.sampled_from(("manifest", "unknown_response_key"))


def st_failure_injection_point() -> st.SearchStrategy[FailureInjectionPoint]:
    return st.sampled_from(FAILURE_INJECTION_POINTS)


def st_cancellation_arrival_time() -> st.SearchStrategy[CancellationArrivalTime]:
    return st.sampled_from(CANCELLATION_ARRIVAL_TIMES)


def st_session_state() -> st.SearchStrategy[SessionState]:
    return st.sampled_from(("empty", "has_prior_state"))
