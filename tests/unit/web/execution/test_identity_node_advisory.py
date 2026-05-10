"""Tests for identity_node_advisory — flags dead-weight passthrough transforms.

The advisory fires for ``transform → passthrough → sink`` chains where the
passthrough has no schema declaration to anchor an unsatisfied edge contract
and is not on a gate-fork branch. See plan:
``/home/john/.claude/plans/dispatch-prompt-floofy-noodle.md``.
"""

from __future__ import annotations

from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
)
from elspeth.web.execution.validation import (
    _CHECK_IDENTITY_NODE_ADVISORY,
    _find_identity_node_advisories,
)


def test_check_constant_value() -> None:
    """The check name string is the public contract — frontend and LLM both read it."""
    assert _CHECK_IDENTITY_NODE_ADVISORY == "identity_node_advisory"


def test_helper_returns_empty_list_for_empty_state() -> None:
    """Stub: helper exists, returns empty list when state has no nodes."""
    state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    findings = _find_identity_node_advisories(state)
    assert findings == []
