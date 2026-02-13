# ADR-002: COPY Mode Limited to FORK_TO_PATHS Only

**Date:** 2026-01-24
**Status:** Accepted
**Deciders:** Architecture team, Code reviewers
**Tags:** routing, audit-integrity, tokens, terminal-states

## Context

The `RoutingMode` enum defines two semantics for routing decisions:

- **MOVE:** Token exits current path and goes to destination only (terminal)
- **COPY:** Token clones to destination AND continues on current path (non-terminal)

### The Original Design

The `RoutingAction` contract was designed with COPY mode support for all routing kinds:

```python
@dataclass(frozen=True)
class RoutingAction:
    """
    CRITICAL: The `mode` field determines move vs copy semantics:
    - MOVE: Token exits current path, goes to destination only
    - COPY: Token clones to destination AND continues on current path
    """
    kind: RoutingKind
    mode: RoutingMode  # Supports COPY for all kinds
```

The `route()` factory method accepts `mode` as a parameter:

```python
def route(label: str, *, mode: RoutingMode = RoutingMode.MOVE) -> RoutingAction:
    """Route to a specific labeled destination."""
```

### The Problem

Bug P1-2026-01-15-routing-copy-ignored revealed that while COPY mode was specified in contracts, it was never implemented for single-destination routing (`RoutingKind.ROUTE`).

Investigation found:

1. **Executor layer** hardcodes `RoutingMode.MOVE` when routing to sinks (executors.py:627)
2. **Processor layer** treats any sink routing as terminal - immediately returns with `ROUTED` outcome (processor.py:681-696)
3. **FORK_TO_PATHS** correctly implements COPY by creating child tokens, each with their own terminal state

### Two Proposed Solutions

**Option A:** Implement COPY mode for ROUTE
- Add orchestrator buffering for mid-pipeline sink writes
- Allow single token to have multiple terminal states (ROUTED + COMPLETED)
- Modify processor to continue after sink routing when mode=COPY

**Option B:** Explicitly reject COPY mode for ROUTE
- Document COPY as only valid for FORK_TO_PATHS
- Add validation to prevent ROUTE + COPY combination
- Users should use fork_to_paths() for "route to sink and continue" semantics

### Critical Constraint: Single Terminal State Per Token

From CLAUDE.md:

> Every row reaches exactly one terminal state - no silent drops

Terminal states: `COMPLETED`, `ROUTED`, `FORKED`, `CONSUMED_IN_BATCH`, `COALESCED`, `QUARANTINED`, `FAILED`, `EXPANDED`

This is an **architectural invariant** enforced throughout the audit trail:

- `token_outcomes` table records ONE terminal outcome per token_id
- Lineage queries assume single terminal state
- Audit exports expect deterministic final state per token

**COPY mode for ROUTE would violate this invariant:**

A single token routed to sink with COPY mode would need TWO terminal states:
1. `ROUTED` when sent to mid-pipeline sink
2. `COMPLETED` when reaching final output sink

This breaks the single-terminal-state model.

### How FORK Correctly Handles COPY

FORK_TO_PATHS achieves "copy and continue" semantics by creating **child tokens**:

```python
# Parent token (terminals.py:697-722)
self._recorder.record_token_outcome(
    token_id=current_token.token_id,
    outcome=RowOutcome.FORKED,  # Parent terminal state
)

# Child tokens (each with their own terminal state)
for child_token in outcome.child_tokens:
    child_items.append(_WorkItem(token=child_token, ...))
    # Child eventually reaches COMPLETED, ROUTED, etc.
```

Each token (parent + children) has exactly ONE terminal state. The audit trail remains consistent.

## Decision

**COPY mode is only valid for FORK_TO_PATHS. ROUTE kind must use MOVE mode.**

Rationale:

1. **Preserves audit integrity:** Single terminal state per token remains invariant
2. **Working pattern exists:** FORK_TO_PATHS already provides "route and continue" semantics correctly
3. **No semantic value:** COPY for ROUTE would just be "fork to one sink" - use fork instead
4. **Avoids complexity:** No orchestrator buffering, no dual terminal states, no checkpoint redesign

## Consequences

### Positive

- **Audit trail remains simple and deterministic:** One terminal state per token, always
- **No breaking changes:** Existing code never used ROUTE + COPY (hardcoded to MOVE)
- **Clear user guidance:** "Use fork_to_paths() to route to sink and continue processing"
- **Fail-fast validation:** Contract-level `ValueError` prevents invalid usage at creation time

### Negative

- **Perceived feature gap:** Users might expect COPY to work with ROUTE (like the contract suggested)
- **Less intuitive for simple cases:** "Route to sink and continue" requires fork_to_paths() with single destination
- **Contract misleading:** Original documentation suggested COPY worked for all routing kinds

### Implementation

**1. Contract validation (routing.py):**

```python
def __post_init__(self) -> None:
    # Existing validations...

    if self.kind == RoutingKind.ROUTE and self.mode == RoutingMode.COPY:
        raise ValueError(
            "COPY mode not supported for ROUTE kind. "
            "Use FORK_TO_PATHS to route to sink and continue processing. "
            "Reason: ELSPETH's audit model enforces single terminal state per token; "
            "COPY would require dual terminal states (ROUTED + COMPLETED)."
        )
```

**2. Updated documentation:**

Clarify in `RoutingAction` docstring:

```
COPY: Token clones to destination AND continues on current path
     (ONLY valid for FORK_TO_PATHS - creates child tokens)
```

**3. Test coverage:**

```python
def test_route_with_copy_raises(self) -> None:
    """route with COPY mode raises ValueError (architectural limitation)."""
    with pytest.raises(ValueError, match="COPY mode not supported for ROUTE"):
        RoutingAction.route("above", mode=RoutingMode.COPY)
```

## Alternatives Considered

### Alternative 1: Implement COPY for ROUTE (Rejected)

**Why rejected:** Violates single-terminal-state invariant. Would require:

- Redesigning token/outcome model to support multiple terminal states per token
- Orchestrator buffering for mid-pipeline sink writes
- Checkpoint redesign (how to resume token with dual outputs?)
- Audit trail schema changes
- Risk of audit integrity bugs

**Estimated effort:** 2-3 weeks

**Risk assessment:** HIGH - fundamental architectural change to audit model

### Alternative 2: Remove COPY from RoutingMode Entirely (Considered but Deferred)

**Rationale:** If COPY is only valid for FORK, why have it on RoutingMode at all?

**Counter-argument:** COPY on edges (not actions) has semantic meaning for audit trail. Fork edges use COPY mode to record "this token was cloned, not moved."

**Decision:** Keep COPY in RoutingMode enum, but restrict to FORK_TO_PATHS at RoutingAction level.

**Follow-up task:** Evaluate whether RoutingMode.COPY is needed at all (separate analysis).

## References

- Bug report: `docs/bugs/pending/P1-2026-01-15-routing-copy-ignored.md`
- Architecture review: Agent a2a1113 (architecture-critic)
- Code review: Agent af71460 (python-code-reviewer)
- CLAUDE.md: "Every row reaches exactly one terminal state"
- ARCHITECTURE.md: Token lifecycle state machine

## Status History

- **2026-01-24:** Accepted - Validation added, tests updated, ADR created
