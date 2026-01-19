# WP-08: Coalesce Executor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge tokens from parallel fork paths back into a single token with configurable policies and merge strategies.

**Architecture:** Create `CoalesceExecutor` that tracks pending tokens by `row_id` (same source row), waits for expected branches per policy, merges row data per strategy, and produces a merged token that continues through the pipeline. Integrates with the WP-07 work queue - coalesce acts as a stateful "barrier" that holds tokens until merge conditions are met.

**Tech Stack:** Python `dict` for pending token tracking (keyed by `row_id`), existing `TokenManager.coalesce_tokens()`, `LandscapeRecorder`, `CoalesceProtocol`, and `CoalescePolicy` infrastructure.

---

## Background: Current State Analysis

### What Already Exists

| Component | Location | Status |
|-----------|----------|--------|
| `CoalesceProtocol` | protocols.py:382-432 | ✅ Complete |
| `CoalescePolicy` enum | protocols.py:260-266 | ✅ Has REQUIRE_ALL, QUORUM, BEST_EFFORT |
| `TokenManager.coalesce_tokens()` | tokens.py:128-160 | ✅ Complete |
| `LandscapeRecorder.coalesce_tokens()` | recorder.py:847+ | ✅ Complete |
| `RowOutcome.COALESCED` | enums.py:135 | ✅ Complete |

### What's Missing

| Component | Action |
|-----------|--------|
| `CoalesceExecutor` | CREATE - main executor class |
| `CoalesceSettings` | CREATE - config model for YAML-defined coalesce |
| Merge strategies | IMPLEMENT - union, nested, select |
| `FIRST` policy | ADD - take first arrival (per contract) |
| Processor integration | MODIFY - handle coalesce in work queue |
| Timeout handling | IMPLEMENT - for best_effort and quorum |

### Design Decisions

1. **Token correlation**: Tokens are correlated by `row_id` (same source row that was forked)
   - Fork creates children with same `row_id` but different `token_id` and `branch_name`
   - Coalesce groups arriving tokens by `row_id`

2. **Stateful barrier**: CoalesceExecutor maintains state:
   - `_pending: dict[str, dict[str, TokenInfo]]` - row_id → {branch_name → token}
   - `_arrival_times: dict[str, dict[str, float]]` - for timeout calculation

3. **Integration with WP-07 work queue**:
   - Coalesce is NOT a transform in the transforms list
   - It's a special step that intercepts tokens after transforms complete
   - When a token with a `branch_name` arrives at coalesce point, it's held
   - When merge conditions are met, merged token is added to work queue

4. **Return type**: `accept()` returns either:
   - `None` if token is being held (waiting for more branches)
   - `TokenInfo` if merge is complete (merged token to continue)

---

## Task 1: Add FIRST Policy to CoalescePolicy Enum

**Files:**
- Modify: `src/elspeth/plugins/protocols.py`
- Test: `tests/plugins/test_protocols.py`

### Step 1: Write failing test for FIRST policy

```python
# tests/plugins/test_protocols.py - add to existing test class

def test_coalesce_policy_has_first():
    """CoalescePolicy should have FIRST variant."""
    from elspeth.plugins.protocols import CoalescePolicy

    assert hasattr(CoalescePolicy, "FIRST")
    assert CoalescePolicy.FIRST.value == "first"
```

### Step 2: Run test to verify it fails

```bash
pytest tests/plugins/test_protocols.py::test_coalesce_policy_has_first -v
```

Expected: FAIL with "AttributeError: FIRST"

### Step 3: Add FIRST to CoalescePolicy

```python
# src/elspeth/plugins/protocols.py - update CoalescePolicy enum (around line 260)

class CoalescePolicy(Enum):
    """How coalesce handles partial arrivals."""

    REQUIRE_ALL = "require_all"  # Wait for all branches; any failure fails
    QUORUM = "quorum"  # Merge if >= n branches succeed
    BEST_EFFORT = "best_effort"  # Merge whatever arrives by timeout
    FIRST = "first"  # Take first arrival, don't wait for others
```

### Step 4: Run test to verify it passes

```bash
pytest tests/plugins/test_protocols.py::test_coalesce_policy_has_first -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/protocols.py tests/plugins/test_protocols.py
git commit -m "feat(protocols): add FIRST policy to CoalescePolicy enum (WP-08)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create CoalesceSettings Config Model

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

### Step 1: Write failing test for CoalesceSettings

```python
# tests/core/test_config.py - add new test class

class TestCoalesceSettings:
    """Test CoalesceSettings configuration model."""

    def test_coalesce_settings_basic(self):
        """Basic coalesce configuration should validate."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="merge_results",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )

        assert settings.name == "merge_results"
        assert settings.branches == ["path_a", "path_b"]
        assert settings.policy == "require_all"
        assert settings.merge == "union"
        assert settings.timeout_seconds is None
        assert settings.quorum_count is None

    def test_coalesce_settings_quorum_requires_count(self):
        """Quorum policy requires quorum_count."""
        from elspeth.core.config import CoalesceSettings
        import pytest

        with pytest.raises(ValueError, match="quorum_count"):
            CoalesceSettings(
                name="quorum_merge",
                branches=["a", "b", "c"],
                policy="quorum",
                merge="union",
                # Missing quorum_count
            )

    def test_coalesce_settings_quorum_with_count(self):
        """Quorum policy with count should validate."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="quorum_merge",
            branches=["a", "b", "c"],
            policy="quorum",
            merge="union",
            quorum_count=2,
        )

        assert settings.quorum_count == 2

    def test_coalesce_settings_best_effort_requires_timeout(self):
        """Best effort policy requires timeout."""
        from elspeth.core.config import CoalesceSettings
        import pytest

        with pytest.raises(ValueError, match="timeout"):
            CoalesceSettings(
                name="best_effort_merge",
                branches=["a", "b"],
                policy="best_effort",
                merge="union",
                # Missing timeout_seconds
            )

    def test_coalesce_settings_nested_merge_strategy(self):
        """Nested merge strategy should validate."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="nested_merge",
            branches=["sentiment", "entities"],
            policy="require_all",
            merge="nested",
        )

        assert settings.merge == "nested"

    def test_coalesce_settings_select_merge_strategy(self):
        """Select merge requires select_branch."""
        from elspeth.core.config import CoalesceSettings
        import pytest

        with pytest.raises(ValueError, match="select_branch"):
            CoalesceSettings(
                name="select_merge",
                branches=["a", "b"],
                policy="require_all",
                merge="select",
                # Missing select_branch
            )

    def test_coalesce_settings_select_with_branch(self):
        """Select merge with branch should validate."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="select_merge",
            branches=["primary", "fallback"],
            policy="require_all",
            merge="select",
            select_branch="primary",
        )

        assert settings.select_branch == "primary"
```

### Step 2: Run test to verify it fails

```bash
pytest tests/core/test_config.py::TestCoalesceSettings -v
```

Expected: FAIL with "cannot import name 'CoalesceSettings'"

### Step 3: Create CoalesceSettings model

```python
# src/elspeth/core/config.py - add after GateSettings class (around line 100)

class CoalesceSettings(BaseModel):
    """Configuration for coalesce (token merging) operations.

    Coalesce merges tokens from parallel fork paths back into a single token.
    Tokens are correlated by row_id (same source row that was forked).

    Example YAML:
        coalesce:
          - name: merge_analysis
            branches:
              - sentiment_path
              - entity_path
            policy: require_all
            merge: union

          - name: quorum_merge
            branches:
              - fast_model
              - slow_model
              - fallback_model
            policy: quorum
            quorum_count: 2
            merge: nested
            timeout_seconds: 30
    """

    model_config = {"frozen": True}

    name: str = Field(description="Unique identifier for this coalesce point")
    branches: list[str] = Field(
        min_length=2,
        description="Branch names to wait for (from fork_to paths)",
    )
    policy: Literal["require_all", "quorum", "best_effort", "first"] = Field(
        default="require_all",
        description="How to handle partial arrivals",
    )
    merge: Literal["union", "nested", "select"] = Field(
        default="union",
        description="How to combine row data from branches",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Max wait time (required for best_effort, optional for quorum)",
    )
    quorum_count: int | None = Field(
        default=None,
        gt=0,
        description="Minimum branches required (required for quorum policy)",
    )
    select_branch: str | None = Field(
        default=None,
        description="Which branch to take for 'select' merge strategy",
    )

    @model_validator(mode="after")
    def validate_policy_requirements(self) -> "CoalesceSettings":
        """Validate policy-specific requirements."""
        if self.policy == "quorum" and self.quorum_count is None:
            raise ValueError(
                f"Coalesce '{self.name}': quorum policy requires quorum_count"
            )
        if self.policy == "quorum" and self.quorum_count is not None:
            if self.quorum_count > len(self.branches):
                raise ValueError(
                    f"Coalesce '{self.name}': quorum_count ({self.quorum_count}) "
                    f"cannot exceed number of branches ({len(self.branches)})"
                )
        if self.policy == "best_effort" and self.timeout_seconds is None:
            raise ValueError(
                f"Coalesce '{self.name}': best_effort policy requires timeout_seconds"
            )
        return self

    @model_validator(mode="after")
    def validate_merge_requirements(self) -> "CoalesceSettings":
        """Validate merge strategy requirements."""
        if self.merge == "select" and self.select_branch is None:
            raise ValueError(
                f"Coalesce '{self.name}': select merge strategy requires select_branch"
            )
        if self.select_branch is not None and self.select_branch not in self.branches:
            raise ValueError(
                f"Coalesce '{self.name}': select_branch '{self.select_branch}' "
                f"must be one of the expected branches: {self.branches}"
            )
        return self
```

### Step 4: Run test to verify it passes

```bash
pytest tests/core/test_config.py::TestCoalesceSettings -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/config.py tests/core/test_config.py
git commit -m "feat(config): add CoalesceSettings for token merging configuration (WP-08)

- Supports 4 policies: require_all, quorum, best_effort, first
- Supports 3 merge strategies: union, nested, select
- Validates policy-specific requirements (quorum_count, timeout)
- Validates merge strategy requirements (select_branch)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Create CoalesceExecutor Core Class

**Files:**
- Create: `src/elspeth/engine/coalesce_executor.py`
- Test: `tests/engine/test_coalesce_executor.py`

### Step 1: Write failing test for CoalesceExecutor initialization

```python
# tests/engine/test_coalesce_executor.py

"""Tests for CoalesceExecutor."""

import pytest
import time
from typing import Any

from elspeth.contracts import TokenInfo
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.spans import SpanFactory


@pytest.fixture
def db() -> LandscapeDB:
    return LandscapeDB.in_memory()


@pytest.fixture
def run(db: LandscapeDB) -> Any:
    recorder = LandscapeRecorder(db)
    return recorder.create_run(settings_hash="test_hash")


@pytest.fixture
def executor_setup(db: LandscapeDB, run: Any) -> tuple[
    LandscapeRecorder, SpanFactory, "TokenManager", str
]:
    """Common setup for executor tests - reduces boilerplate.

    Returns:
        Tuple of (recorder, span_factory, token_manager, run_id)
    """
    from elspeth.engine.tokens import TokenManager

    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()
    token_manager = TokenManager(recorder)
    return recorder, span_factory, token_manager, run.run_id


class TestCoalesceExecutorInit:
    """Test CoalesceExecutor initialization."""

    def test_executor_initializes(self, db: LandscapeDB, run: Any) -> None:
        """Executor should initialize with recorder and span factory."""
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )

        assert executor is not None
```

### Step 2: Run test to verify it fails

```bash
pytest tests/engine/test_coalesce_executor.py::TestCoalesceExecutorInit -v
```

Expected: FAIL with "No module named 'elspeth.engine.coalesce_executor'"

### Step 3: Create CoalesceExecutor skeleton

```python
# src/elspeth/engine/coalesce_executor.py

"""CoalesceExecutor: Merges tokens from parallel fork paths.

Coalesce is a stateful barrier that holds tokens until merge conditions are met.
Tokens are correlated by row_id (same source row that was forked).
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.contracts import ExecutionError, TokenInfo
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.engine.tokens import TokenManager


@dataclass
class CoalesceOutcome:
    """Result of a coalesce accept operation.

    Attributes:
        held: True if token is being held waiting for more branches
        merged_token: The merged token if merge is complete, None if held
        consumed_tokens: Tokens that were merged (marked COALESCED)
    """

    held: bool
    merged_token: TokenInfo | None = None
    consumed_tokens: list[TokenInfo] = field(default_factory=list)


@dataclass
class _PendingCoalesce:
    """Tracks pending tokens for a single row_id at a coalesce point."""

    arrived: dict[str, TokenInfo]  # branch_name -> token
    arrival_times: dict[str, float]  # branch_name -> monotonic time
    first_arrival: float  # For timeout calculation


class CoalesceExecutor:
    """Executes coalesce operations with audit recording.

    Maintains state for pending coalesce operations:
    - Tracks which tokens have arrived for each row_id
    - Evaluates merge conditions based on policy
    - Merges row data according to strategy
    - Records audit trail via LandscapeRecorder

    Example:
        executor = CoalesceExecutor(recorder, span_factory, token_manager, run_id)

        # Configure coalesce point
        executor.register_coalesce(settings, node_id)

        # Accept tokens as they arrive
        for token in arriving_tokens:
            outcome = executor.accept(token, "coalesce_name", step_in_pipeline)
            if outcome.merged_token:
                # Merged token continues through pipeline
                work_queue.append(outcome.merged_token)
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        token_manager: "TokenManager",
        run_id: str,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            token_manager: TokenManager for creating merged tokens
            run_id: Run identifier for audit context
        """
        self._recorder = recorder
        self._spans = span_factory
        self._token_manager = token_manager
        self._run_id = run_id

        # Coalesce configuration: name -> settings
        self._settings: dict[str, CoalesceSettings] = {}
        # Node IDs: coalesce_name -> node_id
        self._node_ids: dict[str, str] = {}
        # Pending tokens: (coalesce_name, row_id) -> _PendingCoalesce
        self._pending: dict[tuple[str, str], _PendingCoalesce] = {}

    def register_coalesce(
        self,
        settings: CoalesceSettings,
        node_id: str,
    ) -> None:
        """Register a coalesce point.

        Args:
            settings: Coalesce configuration
            node_id: Node ID assigned by orchestrator
        """
        self._settings[settings.name] = settings
        self._node_ids[settings.name] = node_id

    def get_registered_names(self) -> list[str]:
        """Get names of all registered coalesce points.

        Used by processor for timeout checking loop.

        Returns:
            List of registered coalesce names
        """
        return list(self._settings.keys())
```

### Step 4: Run test to verify it passes

```bash
pytest tests/engine/test_coalesce_executor.py::TestCoalesceExecutorInit -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/coalesce_executor.py tests/engine/test_coalesce_executor.py
git commit -m "feat(engine): create CoalesceExecutor skeleton (WP-08 Task 3)

- Add CoalesceOutcome and _PendingCoalesce dataclasses
- Initialize with recorder, span_factory, token_manager, run_id
- Add register_coalesce() for configuration
- Add get_registered_names() for timeout polling loop

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement accept() with REQUIRE_ALL Policy

**Files:**
- Modify: `src/elspeth/engine/coalesce_executor.py`
- Test: `tests/engine/test_coalesce_executor.py`

### Step 1: Write failing test for accept with require_all

```python
# tests/engine/test_coalesce_executor.py - add new test class

class TestCoalesceExecutorRequireAll:
    """Test require_all policy."""

    def test_accept_holds_first_token(self, db: LandscapeDB, run: Any) -> None:
        """First token should be held, waiting for others."""
        from elspeth.contracts import NodeType
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        # Register source and coalesce nodes
        source_node = recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            node_id="coalesce_1",
            plugin_name="merge_results",
            node_type=NodeType.COALESCE,
        )

        settings = CoalesceSettings(
            name="merge_results",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        # Create a token from path_a
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"value": 42},
        )
        # Fork creates children with branch names
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
        )
        token_a = children[0]  # path_a

        # Accept first token
        outcome = executor.accept(
            token=token_a,
            coalesce_name="merge_results",
            step_in_pipeline=2,
        )

        # Should be held
        assert outcome.held is True
        assert outcome.merged_token is None
        assert outcome.consumed_tokens == []

    def test_accept_merges_when_all_arrive(self, db: LandscapeDB, run: Any) -> None:
        """When all branches arrive, should merge and return merged token."""
        from elspeth.contracts import NodeType
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        # Register nodes
        source_node = recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            node_id="coalesce_1",
            plugin_name="merge_results",
            node_type=NodeType.COALESCE,
        )

        settings = CoalesceSettings(
            name="merge_results",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )

        executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        # Create tokens from both paths with different data
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data={"original": True},
        )
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
        )

        # Simulate different processing on each branch
        token_a = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"sentiment": "positive"},
            branch_name="path_a",
        )
        token_b = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"entities": ["ACME"]},
            branch_name="path_b",
        )

        # Accept first token - should hold
        outcome1 = executor.accept(token_a, "merge_results", step_in_pipeline=2)
        assert outcome1.held is True

        # Accept second token - should merge
        outcome2 = executor.accept(token_b, "merge_results", step_in_pipeline=2)
        assert outcome2.held is False
        assert outcome2.merged_token is not None
        assert outcome2.merged_token.row_data == {
            "sentiment": "positive",
            "entities": ["ACME"],
        }
        assert len(outcome2.consumed_tokens) == 2
```

### Step 2: Run test to verify it fails

```bash
pytest tests/engine/test_coalesce_executor.py::TestCoalesceExecutorRequireAll -v
```

Expected: FAIL with "CoalesceExecutor has no attribute 'accept'"

### Step 3: Implement accept() method with require_all

```python
# src/elspeth/engine/coalesce_executor.py - add to CoalesceExecutor class

    def accept(
        self,
        token: TokenInfo,
        coalesce_name: str,
        step_in_pipeline: int,
    ) -> CoalesceOutcome:
        """Accept a token at a coalesce point.

        If merge conditions are met, returns the merged token.
        Otherwise, holds the token and returns held=True.

        Args:
            token: Token arriving at coalesce point (must have branch_name)
            coalesce_name: Name of the coalesce configuration
            step_in_pipeline: Current position in DAG

        Returns:
            CoalesceOutcome indicating whether token was held or merged

        Raises:
            ValueError: If coalesce_name not registered or token has no branch_name
        """
        if coalesce_name not in self._settings:
            raise ValueError(f"Coalesce '{coalesce_name}' not registered")

        if token.branch_name is None:
            raise ValueError(
                f"Token {token.token_id} has no branch_name - "
                "only forked tokens can be coalesced"
            )

        settings = self._settings[coalesce_name]
        node_id = self._node_ids[coalesce_name]

        # Validate branch is expected
        if token.branch_name not in settings.branches:
            raise ValueError(
                f"Token branch '{token.branch_name}' not in expected branches "
                f"for coalesce '{coalesce_name}': {settings.branches}"
            )

        # Get or create pending state for this row
        key = (coalesce_name, token.row_id)
        now = time.monotonic()

        if key not in self._pending:
            self._pending[key] = _PendingCoalesce(
                arrived={},
                arrival_times={},
                first_arrival=now,
            )

        pending = self._pending[key]

        # Record arrival
        pending.arrived[token.branch_name] = token
        pending.arrival_times[token.branch_name] = now

        # Check if merge conditions are met
        if self._should_merge(settings, pending):
            return self._execute_merge(
                settings=settings,
                node_id=node_id,
                pending=pending,
                step_in_pipeline=step_in_pipeline,
                key=key,
            )

        # Hold token
        return CoalesceOutcome(held=True)

    def _should_merge(
        self,
        settings: CoalesceSettings,
        pending: "_PendingCoalesce",
    ) -> bool:
        """Check if merge conditions are met based on policy."""
        arrived_count = len(pending.arrived)
        expected_count = len(settings.branches)

        if settings.policy == "require_all":
            return arrived_count == expected_count

        elif settings.policy == "first":
            return arrived_count >= 1

        elif settings.policy == "quorum":
            assert settings.quorum_count is not None
            return arrived_count >= settings.quorum_count

        elif settings.policy == "best_effort":
            # Only merge on timeout (checked elsewhere) or if all arrived
            return arrived_count == expected_count

        return False

    def _execute_merge(
        self,
        settings: CoalesceSettings,
        node_id: str,
        pending: "_PendingCoalesce",
        step_in_pipeline: int,
        key: tuple[str, str],
    ) -> CoalesceOutcome:
        """Execute the merge and create merged token."""
        # Merge row data according to strategy
        merged_data = self._merge_data(settings, pending.arrived)

        # Get list of consumed tokens
        consumed_tokens = list(pending.arrived.values())

        # Create merged token via TokenManager
        merged_token = self._token_manager.coalesce_tokens(
            parents=consumed_tokens,
            merged_data=merged_data,
            step_in_pipeline=step_in_pipeline,
        )

        # Record node states for consumed tokens
        for token in consumed_tokens:
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=node_id,
                step_index=step_in_pipeline,
                input_data=token.row_data,
            )
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="coalesced",
                output_data={"merged_into": merged_token.token_id},
                duration_ms=0,
            )

        # Clean up pending state
        del self._pending[key]

        return CoalesceOutcome(
            held=False,
            merged_token=merged_token,
            consumed_tokens=consumed_tokens,
        )

    def _merge_data(
        self,
        settings: CoalesceSettings,
        arrived: dict[str, TokenInfo],
    ) -> dict[str, Any]:
        """Merge row data from arrived tokens based on strategy."""
        if settings.merge == "union":
            # Combine all fields (later branches override earlier)
            merged: dict[str, Any] = {}
            for branch_name in settings.branches:
                if branch_name in arrived:
                    merged.update(arrived[branch_name].row_data)
            return merged

        elif settings.merge == "nested":
            # Each branch as nested object
            return {
                branch_name: arrived[branch_name].row_data
                for branch_name in settings.branches
                if branch_name in arrived
            }

        elif settings.merge == "select":
            # Take specific branch output
            assert settings.select_branch is not None
            if settings.select_branch in arrived:
                return arrived[settings.select_branch].row_data.copy()
            # Fallback to first arrived if select branch not present
            return next(iter(arrived.values())).row_data.copy()

        # Default to union
        merged = {}
        for token in arrived.values():
            merged.update(token.row_data)
        return merged
```

### Step 4: Run test to verify it passes

```bash
pytest tests/engine/test_coalesce_executor.py::TestCoalesceExecutorRequireAll -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/coalesce_executor.py tests/engine/test_coalesce_executor.py
git commit -m "feat(coalesce): implement accept() with require_all policy (WP-08 Task 4)

- Add accept() method for receiving tokens at coalesce points
- Implement require_all policy (wait for all branches)
- Add merge strategies: union, nested, select
- Record audit trail for consumed tokens

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Implement FIRST, QUORUM, and BEST_EFFORT Policies

**Files:**
- Modify: `src/elspeth/engine/coalesce_executor.py`
- Test: `tests/engine/test_coalesce_executor.py`

### Step 1: Write failing tests for other policies

```python
# tests/engine/test_coalesce_executor.py - add new test classes

class TestCoalesceExecutorFirst:
    """Test FIRST policy."""

    def test_first_merges_immediately(self, db: LandscapeDB, run: Any) -> None:
        """FIRST policy should merge as soon as one token arrives."""
        from elspeth.contracts import NodeType
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        source_node = recorder.register_node(
            run_id=run.run_id, node_id="source_1",
            plugin_name="test_source", node_type=NodeType.SOURCE,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id, node_id="coalesce_1",
            plugin_name="first_wins", node_type=NodeType.COALESCE,
        )

        settings = CoalesceSettings(
            name="first_wins",
            branches=["fast", "slow"],
            policy="first",
            merge="select",
            select_branch="fast",  # Prefer fast, but take whatever arrives first
        )

        executor = CoalesceExecutor(
            recorder=recorder, span_factory=span_factory,
            token_manager=token_manager, run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id, source_node_id=source_node.node_id,
            row_index=0, row_data={"original": True},
        )
        children = token_manager.fork_token(
            parent_token=initial_token, branches=["fast", "slow"],
            step_in_pipeline=1,
        )

        # Simulate slow arriving first (fast is delayed)
        token_slow = TokenInfo(
            row_id=children[1].row_id, token_id=children[1].token_id,
            row_data={"result": "from_slow"}, branch_name="slow",
        )

        # Accept slow token - should merge immediately
        outcome = executor.accept(token_slow, "first_wins", step_in_pipeline=2)

        assert outcome.held is False
        assert outcome.merged_token is not None
        assert outcome.merged_token.row_data == {"result": "from_slow"}


class TestCoalesceExecutorQuorum:
    """Test QUORUM policy."""

    def test_quorum_merges_at_threshold(self, db: LandscapeDB, run: Any) -> None:
        """QUORUM should merge when quorum_count branches arrive."""
        from elspeth.contracts import NodeType
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        source_node = recorder.register_node(
            run_id=run.run_id, node_id="source_1",
            plugin_name="test_source", node_type=NodeType.SOURCE,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id, node_id="coalesce_1",
            plugin_name="quorum_merge", node_type=NodeType.COALESCE,
        )

        settings = CoalesceSettings(
            name="quorum_merge",
            branches=["model_a", "model_b", "model_c"],
            policy="quorum",
            quorum_count=2,  # Merge when 2 of 3 arrive
            merge="nested",
        )

        executor = CoalesceExecutor(
            recorder=recorder, span_factory=span_factory,
            token_manager=token_manager, run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id, source_node_id=source_node.node_id,
            row_index=0, row_data={},
        )
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["model_a", "model_b", "model_c"],
            step_in_pipeline=1,
        )

        token_a = TokenInfo(
            row_id=children[0].row_id, token_id=children[0].token_id,
            row_data={"score": 0.9}, branch_name="model_a",
        )
        token_b = TokenInfo(
            row_id=children[1].row_id, token_id=children[1].token_id,
            row_data={"score": 0.85}, branch_name="model_b",
        )

        # Accept first - should hold (1 < 2)
        outcome1 = executor.accept(token_a, "quorum_merge", step_in_pipeline=2)
        assert outcome1.held is True

        # Accept second - should merge (2 >= 2)
        outcome2 = executor.accept(token_b, "quorum_merge", step_in_pipeline=2)
        assert outcome2.held is False
        assert outcome2.merged_token is not None
        # Nested merge strategy
        assert outcome2.merged_token.row_data == {
            "model_a": {"score": 0.9},
            "model_b": {"score": 0.85},
        }


class TestCoalesceExecutorBestEffort:
    """Test BEST_EFFORT policy with timeout."""

    def test_best_effort_merges_on_timeout(self, db: LandscapeDB, run: Any) -> None:
        """BEST_EFFORT should merge whatever arrived when timeout expires."""
        from elspeth.contracts import NodeType
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        source_node = recorder.register_node(
            run_id=run.run_id, node_id="source_1",
            plugin_name="test_source", node_type=NodeType.SOURCE,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id, node_id="coalesce_1",
            plugin_name="best_effort_merge", node_type=NodeType.COALESCE,
        )

        settings = CoalesceSettings(
            name="best_effort_merge",
            branches=["path_a", "path_b", "path_c"],
            policy="best_effort",
            timeout_seconds=0.1,  # Short timeout for testing
            merge="union",
        )

        executor = CoalesceExecutor(
            recorder=recorder, span_factory=span_factory,
            token_manager=token_manager, run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        initial_token = token_manager.create_initial_token(
            run_id=run.run_id, source_node_id=source_node.node_id,
            row_index=0, row_data={},
        )
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b", "path_c"],
            step_in_pipeline=1,
        )

        token_a = TokenInfo(
            row_id=children[0].row_id, token_id=children[0].token_id,
            row_data={"a_result": 1}, branch_name="path_a",
        )

        # Accept one token
        outcome1 = executor.accept(token_a, "best_effort_merge", step_in_pipeline=2)
        assert outcome1.held is True

        # Wait for timeout
        time.sleep(0.15)

        # Check timeout and force merge
        timed_out = executor.check_timeouts("best_effort_merge", step_in_pipeline=2)

        # Should have one merged result
        assert len(timed_out) == 1
        assert timed_out[0].merged_token is not None
        assert timed_out[0].merged_token.row_data == {"a_result": 1}
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/engine/test_coalesce_executor.py::TestCoalesceExecutorFirst -v
pytest tests/engine/test_coalesce_executor.py::TestCoalesceExecutorQuorum -v
pytest tests/engine/test_coalesce_executor.py::TestCoalesceExecutorBestEffort -v
```

Expected: Various failures (check_timeouts not implemented, etc.)

### Step 3: Implement check_timeouts() method

```python
# src/elspeth/engine/coalesce_executor.py - add to CoalesceExecutor class

    def check_timeouts(
        self,
        coalesce_name: str,
        step_in_pipeline: int,
    ) -> list[CoalesceOutcome]:
        """Check for timed-out pending coalesces and merge them.

        For best_effort policy, merges whatever has arrived when timeout expires.
        For quorum policy with timeout, merges if quorum met when timeout expires.

        Args:
            coalesce_name: Name of the coalesce configuration
            step_in_pipeline: Current position in DAG

        Returns:
            List of CoalesceOutcomes for any merges triggered by timeout
        """
        if coalesce_name not in self._settings:
            raise ValueError(f"Coalesce '{coalesce_name}' not registered")

        settings = self._settings[coalesce_name]
        node_id = self._node_ids[coalesce_name]

        if settings.timeout_seconds is None:
            return []

        now = time.monotonic()
        results: list[CoalesceOutcome] = []
        keys_to_process: list[tuple[str, str]] = []

        # Find timed-out entries
        for key, pending in self._pending.items():
            if key[0] != coalesce_name:
                continue

            elapsed = now - pending.first_arrival
            if elapsed >= settings.timeout_seconds:
                keys_to_process.append(key)

        # Process timed-out entries
        for key in keys_to_process:
            pending = self._pending[key]

            # For best_effort, always merge on timeout if anything arrived
            if settings.policy == "best_effort" and len(pending.arrived) > 0:
                outcome = self._execute_merge(
                    settings=settings,
                    node_id=node_id,
                    pending=pending,
                    step_in_pipeline=step_in_pipeline,
                    key=key,
                )
                results.append(outcome)

            # For quorum, merge on timeout only if quorum met
            elif settings.policy == "quorum":
                assert settings.quorum_count is not None
                if len(pending.arrived) >= settings.quorum_count:
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step_in_pipeline=step_in_pipeline,
                        key=key,
                    )
                    results.append(outcome)

        return results
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/engine/test_coalesce_executor.py -v
```

Expected: All tests pass

### Step 5: Commit

```bash
git add src/elspeth/engine/coalesce_executor.py tests/engine/test_coalesce_executor.py
git commit -m "feat(coalesce): implement FIRST, QUORUM, BEST_EFFORT policies (WP-08 Task 5)

- FIRST: merge immediately on first arrival
- QUORUM: merge when quorum_count branches arrive
- BEST_EFFORT: merge on timeout with whatever arrived
- Add check_timeouts() for timeout-based merging

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5.5: Record Coalesce Audit Metadata

**Goal:** Per the contract (lines 699-744), the audit trail must record coalesce event details including policy, strategy, timing, and arrival order.

**Files:**
- Modify: `src/elspeth/engine/coalesce_executor.py`
- Test: `tests/engine/test_coalesce_executor.py`

### Step 1: Write failing test for audit metadata

```python
# tests/engine/test_coalesce_executor.py - add to TestCoalesceIntegration

def test_coalesce_records_audit_metadata(self, db: LandscapeDB, run: Any) -> None:
    """Coalesce should record policy, strategy, and timing in audit trail."""
    from elspeth.contracts import NodeType
    from elspeth.engine.coalesce_executor import CoalesceExecutor
    from elspeth.engine.tokens import TokenManager

    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()
    token_manager = TokenManager(recorder)

    source_node = recorder.register_node(
        run_id=run.run_id, node_id="source",
        plugin_name="csv", node_type=NodeType.SOURCE,
    )
    coalesce_node = recorder.register_node(
        run_id=run.run_id, node_id="merge",
        plugin_name="merge_results", node_type=NodeType.COALESCE,
    )

    settings = CoalesceSettings(
        name="merge_results",
        branches=["path_a", "path_b"],
        policy="require_all",
        merge="union",
    )

    executor = CoalesceExecutor(
        recorder=recorder, span_factory=span_factory,
        token_manager=token_manager, run_id=run.run_id,
    )
    executor.register_coalesce(settings, coalesce_node.node_id)

    initial_token = token_manager.create_initial_token(
        run_id=run.run_id, source_node_id=source_node.node_id,
        row_index=0, row_data={"original": True},
    )
    children = token_manager.fork_token(
        parent_token=initial_token, branches=["path_a", "path_b"],
        step_in_pipeline=1,
    )

    token_a = TokenInfo(
        row_id=children[0].row_id, token_id=children[0].token_id,
        row_data={"from_a": 1}, branch_name="path_a",
    )
    token_b = TokenInfo(
        row_id=children[1].row_id, token_id=children[1].token_id,
        row_data={"from_b": 2}, branch_name="path_b",
    )

    executor.accept(token_a, "merge_results", step_in_pipeline=2)
    outcome = executor.accept(token_b, "merge_results", step_in_pipeline=2)

    # Verify audit metadata was recorded
    # The merged token's node_state should have context_after with coalesce details
    assert outcome.merged_token is not None
    assert outcome.coalesce_metadata is not None
    assert outcome.coalesce_metadata["policy"] == "require_all"
    assert outcome.coalesce_metadata["merge_strategy"] == "union"
    assert outcome.coalesce_metadata["wait_duration_ms"] >= 0
    assert set(outcome.coalesce_metadata["branches_arrived"]) == {"path_a", "path_b"}
```

### Step 2: Update CoalesceOutcome to include metadata

```python
# src/elspeth/engine/coalesce_executor.py - update CoalesceOutcome

@dataclass
class CoalesceOutcome:
    """Result of a coalesce accept operation.

    Attributes:
        held: True if token is being held waiting for more branches
        merged_token: The merged token if merge is complete, None if held
        consumed_tokens: Tokens that were merged (marked COALESCED)
        coalesce_metadata: Audit metadata about the merge (policy, strategy, timing)
    """

    held: bool
    merged_token: TokenInfo | None = None
    consumed_tokens: list[TokenInfo] = field(default_factory=list)
    coalesce_metadata: dict[str, Any] | None = None
```

### Step 3: Update _execute_merge to capture and return metadata

```python
# src/elspeth/engine/coalesce_executor.py - update _execute_merge

    def _execute_merge(
        self,
        settings: CoalesceSettings,
        node_id: str,
        pending: "_PendingCoalesce",
        step_in_pipeline: int,
        key: tuple[str, str],
    ) -> CoalesceOutcome:
        """Execute the merge and create merged token."""
        now = time.monotonic()

        # Capture audit metadata
        coalesce_metadata = {
            "policy": settings.policy,
            "merge_strategy": settings.merge,
            "expected_branches": settings.branches,
            "branches_arrived": list(pending.arrived.keys()),
            "arrival_order": [
                {"branch": branch, "arrival_offset_ms": (t - pending.first_arrival) * 1000}
                for branch, t in sorted(pending.arrival_times.items(), key=lambda x: x[1])
            ],
            "wait_duration_ms": (now - pending.first_arrival) * 1000,
        }

        # Merge row data according to strategy
        merged_data = self._merge_data(settings, pending.arrived)

        # Get list of consumed tokens
        consumed_tokens = list(pending.arrived.values())

        # Create merged token via TokenManager
        merged_token = self._token_manager.coalesce_tokens(
            parents=consumed_tokens,
            merged_data=merged_data,
            step_in_pipeline=step_in_pipeline,
        )

        # Record node states for consumed tokens with coalesce metadata
        for token in consumed_tokens:
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=node_id,
                step_index=step_in_pipeline,
                input_data=token.row_data,
            )
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="coalesced",
                output_data={"merged_into": merged_token.token_id},
                duration_ms=coalesce_metadata["wait_duration_ms"],
                context_after=coalesce_metadata,  # Store metadata in context_after
            )

        # Clean up pending state
        del self._pending[key]

        return CoalesceOutcome(
            held=False,
            merged_token=merged_token,
            consumed_tokens=consumed_tokens,
            coalesce_metadata=coalesce_metadata,
        )
```

### Step 4: Run test to verify it passes

```bash
pytest tests/engine/test_coalesce_executor.py::TestCoalesceIntegration::test_coalesce_records_audit_metadata -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/coalesce_executor.py tests/engine/test_coalesce_executor.py
git commit -m "feat(coalesce): record audit metadata for coalesce events (WP-08 Task 5.5)

- Add coalesce_metadata to CoalesceOutcome
- Record policy, merge_strategy, branches_arrived, arrival_order, wait_duration_ms
- Store metadata in node_state.context_after for audit trail

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Export CoalesceExecutor

**Files:**
- Modify: `src/elspeth/engine/__init__.py`
- Test: Verify imports work

> **Note:** `NodeType.COALESCE` already exists in `enums.py:66` (verified 2026-01-18).

### Step 1: Export CoalesceExecutor

```python
# src/elspeth/engine/__init__.py - add to exports

from elspeth.engine.coalesce_executor import CoalesceExecutor, CoalesceOutcome

# Add to __all__
__all__ = [
    # ... existing exports ...
    "CoalesceExecutor",
    "CoalesceOutcome",
]
```

### Step 2: Verify imports

```bash
python -c "from elspeth.engine import CoalesceExecutor, CoalesceOutcome; print('OK')"
python -c "from elspeth.contracts import NodeType; print(NodeType.COALESCE)"
```

Expected: Both print successfully

### Step 3: Commit

```bash
git add src/elspeth/engine/__init__.py
git commit -m "feat(engine): export CoalesceExecutor (WP-08 Task 6)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Add coalesce Field to ElspethSettings

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

### Step 1: Write test for coalesce in ElspethSettings

```python
# tests/core/test_config.py - add to existing test class

def test_elspeth_settings_with_coalesce(self):
    """ElspethSettings should accept coalesce configuration."""
    from elspeth.core.config import (
        ElspethSettings,
        DatasourceSettings,
        SinkSettings,
        CoalesceSettings,
    )

    settings = ElspethSettings(
        datasource=DatasourceSettings(plugin="csv_local", options={"path": "test.csv"}),
        sinks={"default": SinkSettings(plugin="csv", options={"path": "out.csv"})},
        output_sink="default",
        coalesce=[
            CoalesceSettings(
                name="merge_results",
                branches=["path_a", "path_b"],
                policy="require_all",
                merge="union",
            ),
        ],
    )

    assert len(settings.coalesce) == 1
    assert settings.coalesce[0].name == "merge_results"
```

### Step 2: Add coalesce field to ElspethSettings

```python
# src/elspeth/core/config.py - add to ElspethSettings class

class ElspethSettings(BaseModel):
    # ... existing fields ...

    # Optional - coalesce configuration (for merging fork paths)
    coalesce: list[CoalesceSettings] = Field(
        default_factory=list,
        description="Coalesce configurations for merging forked paths",
    )
```

### Step 3: Run test

```bash
pytest tests/core/test_config.py::test_elspeth_settings_with_coalesce -v
```

Expected: PASS

### Step 4: Commit

```bash
git add src/elspeth/core/config.py tests/core/test_config.py
git commit -m "feat(config): add coalesce field to ElspethSettings (WP-08 Task 7)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Integration Test - Full Fork/Coalesce Pipeline

**Files:**
- Test: `tests/engine/test_coalesce_executor.py`

### Step 1: Write integration test

```python
# tests/engine/test_coalesce_executor.py - add new test class

class TestCoalesceIntegration:
    """Integration tests for full fork -> process -> coalesce flow."""

    def test_fork_process_coalesce_full_flow(self, db: LandscapeDB, run: Any) -> None:
        """Full flow: fork -> different transforms -> coalesce."""
        from elspeth.contracts import NodeType, RowOutcome
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        # Register all nodes
        source_node = recorder.register_node(
            run_id=run.run_id, node_id="source",
            plugin_name="csv", node_type=NodeType.SOURCE,
        )
        gate_node = recorder.register_node(
            run_id=run.run_id, node_id="fork_gate",
            plugin_name="parallel_analysis", node_type=NodeType.GATE,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id, node_id="merge",
            plugin_name="merge_results", node_type=NodeType.COALESCE,
        )

        settings = CoalesceSettings(
            name="merge_results",
            branches=["sentiment", "entities"],
            policy="require_all",
            merge="nested",
        )

        executor = CoalesceExecutor(
            recorder=recorder, span_factory=span_factory,
            token_manager=token_manager, run_id=run.run_id,
        )
        executor.register_coalesce(settings, coalesce_node.node_id)

        # Simulate source row
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id, source_node_id=source_node.node_id,
            row_index=0, row_data={"text": "ACME Corp reported positive earnings"},
        )

        # Simulate fork
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["sentiment", "entities"],
            step_in_pipeline=1,
        )

        # Simulate different processing on each branch
        sentiment_token = TokenInfo(
            row_id=children[0].row_id, token_id=children[0].token_id,
            row_data={"sentiment": "positive", "confidence": 0.92},
            branch_name="sentiment",
        )
        entities_token = TokenInfo(
            row_id=children[1].row_id, token_id=children[1].token_id,
            row_data={"entities": [{"name": "ACME Corp", "type": "ORG"}]},
            branch_name="entities",
        )

        # Coalesce
        outcome1 = executor.accept(sentiment_token, "merge_results", step_in_pipeline=3)
        assert outcome1.held is True

        outcome2 = executor.accept(entities_token, "merge_results", step_in_pipeline=3)
        assert outcome2.held is False
        assert outcome2.merged_token is not None

        # Verify merged data has nested structure
        merged = outcome2.merged_token.row_data
        assert merged == {
            "sentiment": {"sentiment": "positive", "confidence": 0.92},
            "entities": {"entities": [{"name": "ACME Corp", "type": "ORG"}]},
        }

        # Verify consumed tokens
        assert len(outcome2.consumed_tokens) == 2
        consumed_branches = {t.branch_name for t in outcome2.consumed_tokens}
        assert consumed_branches == {"sentiment", "entities"}
```

### Step 2: Run test

```bash
pytest tests/engine/test_coalesce_executor.py::TestCoalesceIntegration -v
```

Expected: PASS

### Step 3: Commit

```bash
git add tests/engine/test_coalesce_executor.py
git commit -m "test(coalesce): add fork/process/coalesce integration test (WP-08 Task 8)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8.5: Add flush_pending() for Graceful Shutdown

**Goal:** Handle end-of-source and graceful shutdown by processing any pending coalesces.

**Files:**
- Modify: `src/elspeth/engine/coalesce_executor.py`
- Test: `tests/engine/test_coalesce_executor.py`

### Step 1: Write failing test for flush_pending

```python
# tests/engine/test_coalesce_executor.py - add to TestCoalesceExecutorBestEffort

def test_flush_pending_merges_incomplete(self, db: LandscapeDB, run: Any) -> None:
    """flush_pending should merge incomplete pending coalesces at end-of-source."""
    from elspeth.contracts import NodeType
    from elspeth.engine.coalesce_executor import CoalesceExecutor
    from elspeth.engine.tokens import TokenManager

    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()
    token_manager = TokenManager(recorder)

    source_node = recorder.register_node(
        run_id=run.run_id, node_id="source_1",
        plugin_name="test_source", node_type=NodeType.SOURCE,
    )
    coalesce_node = recorder.register_node(
        run_id=run.run_id, node_id="coalesce_1",
        plugin_name="best_effort_merge", node_type=NodeType.COALESCE,
    )

    settings = CoalesceSettings(
        name="best_effort_merge",
        branches=["path_a", "path_b", "path_c"],
        policy="best_effort",
        timeout_seconds=60,  # Long timeout
        merge="union",
    )

    executor = CoalesceExecutor(
        recorder=recorder, span_factory=span_factory,
        token_manager=token_manager, run_id=run.run_id,
    )
    executor.register_coalesce(settings, coalesce_node.node_id)

    initial_token = token_manager.create_initial_token(
        run_id=run.run_id, source_node_id=source_node.node_id,
        row_index=0, row_data={},
    )
    children = token_manager.fork_token(
        parent_token=initial_token,
        branches=["path_a", "path_b", "path_c"],
        step_in_pipeline=1,
    )

    # Only two of three branches arrive
    token_a = TokenInfo(
        row_id=children[0].row_id, token_id=children[0].token_id,
        row_data={"a_result": 1}, branch_name="path_a",
    )
    token_b = TokenInfo(
        row_id=children[1].row_id, token_id=children[1].token_id,
        row_data={"b_result": 2}, branch_name="path_b",
    )

    outcome1 = executor.accept(token_a, "best_effort_merge", step_in_pipeline=2)
    assert outcome1.held is True
    outcome2 = executor.accept(token_b, "best_effort_merge", step_in_pipeline=2)
    assert outcome2.held is True  # Still waiting for path_c

    # Source exhausts - flush pending coalesces
    flushed = executor.flush_pending(step_in_pipeline=2)

    # Should have merged what was available
    assert len(flushed) == 1
    assert flushed[0].merged_token is not None
    assert flushed[0].merged_token.row_data == {"a_result": 1, "b_result": 2}
    assert flushed[0].coalesce_metadata["branches_arrived"] == ["path_a", "path_b"]

def test_flush_pending_respects_quorum(self, db: LandscapeDB, run: Any) -> None:
    """flush_pending should NOT merge quorum policy if quorum not met."""
    from elspeth.contracts import NodeType
    from elspeth.engine.coalesce_executor import CoalesceExecutor
    from elspeth.engine.tokens import TokenManager

    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()
    token_manager = TokenManager(recorder)

    source_node = recorder.register_node(
        run_id=run.run_id, node_id="source_1",
        plugin_name="test_source", node_type=NodeType.SOURCE,
    )
    coalesce_node = recorder.register_node(
        run_id=run.run_id, node_id="coalesce_1",
        plugin_name="quorum_merge", node_type=NodeType.COALESCE,
    )

    settings = CoalesceSettings(
        name="quorum_merge",
        branches=["model_a", "model_b", "model_c"],
        policy="quorum",
        quorum_count=3,  # Need all 3
        merge="union",
    )

    executor = CoalesceExecutor(
        recorder=recorder, span_factory=span_factory,
        token_manager=token_manager, run_id=run.run_id,
    )
    executor.register_coalesce(settings, coalesce_node.node_id)

    initial_token = token_manager.create_initial_token(
        run_id=run.run_id, source_node_id=source_node.node_id,
        row_index=0, row_data={},
    )
    children = token_manager.fork_token(
        parent_token=initial_token,
        branches=["model_a", "model_b", "model_c"],
        step_in_pipeline=1,
    )

    # Only one arrives - doesn't meet quorum
    token_a = TokenInfo(
        row_id=children[0].row_id, token_id=children[0].token_id,
        row_data={"score": 0.9}, branch_name="model_a",
    )
    executor.accept(token_a, "quorum_merge", step_in_pipeline=2)

    # flush_pending returns failed entries (quorum not met)
    flushed = executor.flush_pending(step_in_pipeline=2)

    # Should return failure info, not a merged result
    assert len(flushed) == 1
    assert flushed[0].merged_token is None  # No merge - quorum not met
    assert flushed[0].failure_reason == "quorum_not_met"
```

### Step 2: Update CoalesceOutcome to include failure_reason

```python
# src/elspeth/engine/coalesce_executor.py - update CoalesceOutcome

@dataclass
class CoalesceOutcome:
    """Result of a coalesce accept operation."""

    held: bool
    merged_token: TokenInfo | None = None
    consumed_tokens: list[TokenInfo] = field(default_factory=list)
    coalesce_metadata: dict[str, Any] | None = None
    failure_reason: str | None = None  # For flush_pending when merge not possible
```

### Step 3: Implement flush_pending method

```python
# src/elspeth/engine/coalesce_executor.py - add to CoalesceExecutor class

    def flush_pending(
        self,
        step_in_pipeline: int,
    ) -> list[CoalesceOutcome]:
        """Flush all pending coalesces (called at end-of-source or shutdown).

        For best_effort policy: merges whatever arrived.
        For quorum policy: merges if quorum met, returns failure otherwise.
        For require_all policy: returns failure (never partial merge).
        For first policy: should never have pending (merges immediately).

        Args:
            step_in_pipeline: Current position in DAG

        Returns:
            List of CoalesceOutcomes for all pending coalesces
        """
        results: list[CoalesceOutcome] = []
        keys_to_process = list(self._pending.keys())

        for key in keys_to_process:
            coalesce_name, row_id = key
            settings = self._settings[coalesce_name]
            node_id = self._node_ids[coalesce_name]
            pending = self._pending[key]

            if settings.policy == "best_effort":
                # Always merge whatever arrived
                if len(pending.arrived) > 0:
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step_in_pipeline=step_in_pipeline,
                        key=key,
                    )
                    results.append(outcome)

            elif settings.policy == "quorum":
                assert settings.quorum_count is not None
                if len(pending.arrived) >= settings.quorum_count:
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step_in_pipeline=step_in_pipeline,
                        key=key,
                    )
                    results.append(outcome)
                else:
                    # Quorum not met - record failure
                    del self._pending[key]
                    results.append(CoalesceOutcome(
                        held=False,
                        failure_reason="quorum_not_met",
                        coalesce_metadata={
                            "policy": settings.policy,
                            "quorum_required": settings.quorum_count,
                            "branches_arrived": list(pending.arrived.keys()),
                            "row_id": row_id,
                        },
                    ))

            elif settings.policy == "require_all":
                # require_all never does partial merge
                if len(pending.arrived) == len(settings.branches):
                    outcome = self._execute_merge(
                        settings=settings,
                        node_id=node_id,
                        pending=pending,
                        step_in_pipeline=step_in_pipeline,
                        key=key,
                    )
                    results.append(outcome)
                else:
                    del self._pending[key]
                    results.append(CoalesceOutcome(
                        held=False,
                        failure_reason="incomplete_branches",
                        coalesce_metadata={
                            "policy": settings.policy,
                            "expected_branches": settings.branches,
                            "branches_arrived": list(pending.arrived.keys()),
                            "row_id": row_id,
                        },
                    ))

            # FIRST policy should never have pending entries

        return results

    def get_pending_count(self, coalesce_name: str | None = None) -> int:
        """Get count of pending coalesces (for monitoring/debugging).

        Args:
            coalesce_name: Filter by coalesce name, or None for all

        Returns:
            Number of pending coalesce operations
        """
        if coalesce_name is None:
            return len(self._pending)
        return sum(1 for k in self._pending if k[0] == coalesce_name)
```

### Step 4: Run tests

```bash
pytest tests/engine/test_coalesce_executor.py -v -k "flush_pending"
```

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/coalesce_executor.py tests/engine/test_coalesce_executor.py
git commit -m "feat(coalesce): add flush_pending() for graceful shutdown (WP-08 Task 8.5)

- flush_pending() processes all pending coalesces at end-of-source
- best_effort: merges whatever arrived
- quorum: merges if met, returns failure_reason if not
- require_all: returns failure_reason if incomplete
- Add get_pending_count() for monitoring

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Update Tracker

**Files:**
- Modify: `docs/plans/2026-01-17-plugin-refactor-tracker.md`

### Step 1: Mark WP-08 complete

Update tracker with completion status and verification results.

### Step 2: Commit

```bash
git add docs/plans/2026-01-17-plugin-refactor-tracker.md
git commit -m "docs(tracker): mark WP-08 complete

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification Checklist

After all tasks complete, verify:

- [ ] `CoalescePolicy.FIRST` exists in protocols.py
- [ ] `CoalesceSettings` validates all policy/merge requirements
- [ ] `CoalesceExecutor` initializes with recorder, spans, token_manager
- [ ] `accept()` holds tokens until merge conditions met
- [ ] All 4 policies work:
  - [ ] `require_all` - waits for all branches
  - [ ] `quorum` - waits for N branches
  - [ ] `best_effort` - merges on timeout
  - [ ] `first` - merges immediately
- [ ] All 3 merge strategies work:
  - [ ] `union` - combines all fields
  - [ ] `nested` - branches as nested objects
  - [ ] `select` - takes specific branch
- [ ] `check_timeouts()` triggers timeout-based merges
- [ ] `flush_pending()` handles end-of-source gracefully
- [ ] `get_registered_names()` returns list of coalesce names
- [ ] `get_pending_count()` returns pending count for monitoring
- [ ] `NodeType.COALESCE` exists (verified: already in enums.py:66)
- [ ] `CoalesceExecutor` exported from engine module
- [ ] `ElspethSettings.coalesce` field exists
- [ ] Audit metadata recorded in `coalesce_metadata`:
  - [ ] policy, merge_strategy, expected_branches
  - [ ] branches_arrived, arrival_order, wait_duration_ms
- [ ] Audit trail records consumed tokens with COALESCED status
- [ ] `failure_reason` set for incomplete coalesces
- [ ] `mypy --strict` passes
- [ ] All tests pass

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Timeout race conditions | Use `time.monotonic()` (not wall clock) |
| Memory leak from pending tokens | Cleanup on merge, timeout, or flush_pending |
| Branch name mismatch | Validate branch in expected list |
| Orphaned pending state | `flush_pending()` at end-of-source handles all remaining |
| End-of-source with pending tokens | `flush_pending()` called by orchestrator after source exhausts |
| Pipeline crash mid-coalesce | Tokens held in memory only; recovery uses checkpoint system |

---

## Notes

1. **Processor integration**: This plan creates `CoalesceExecutor` as a standalone component. Integration into `processor.py` work queue requires WP-07 to be complete first. The integration would:
   - Detect when a token with `branch_name` reaches a coalesce point
   - Call `executor.accept()`
   - If merged, add merged token to work queue
   - If held, don't add to results yet

2. **Orchestrator setup**: The orchestrator will need to:
   - Register coalesce nodes in Landscape
   - Create CoalesceExecutor
   - Call `register_coalesce()` for each config entry
   - Call `check_timeouts()` and `flush_pending()` at appropriate points

3. **Timeout polling strategy**: The `check_timeouts()` method uses a polling model. There are three integration points:

   | When | Method | Caller |
   |------|--------|--------|
   | After processing each token batch | `check_timeouts(coalesce_name)` | RowProcessor work queue loop |
   | After source exhausts | `flush_pending()` | Orchestrator.run() |
   | On graceful shutdown | `flush_pending()` | Orchestrator.shutdown() |

   **Recommended polling approach in processor work queue:**
   ```python
   while work_queue:
       token = work_queue.popleft()
       result = self._process_single_token(token, ...)

       # After each token, check timeouts for any coalesce points
       for coalesce_name in self._coalesce_executor.get_registered_names():
           timed_out = self._coalesce_executor.check_timeouts(coalesce_name, step)
           for outcome in timed_out:
               if outcome.merged_token:
                   work_queue.append(outcome.merged_token)

   # After queue drains (source exhausted), flush pending
   flushed = self._coalesce_executor.flush_pending(step)
   for outcome in flushed:
       if outcome.merged_token:
           final_results.append(outcome.merged_token)
       elif outcome.failure_reason:
           failed_coalesces.append(outcome)
   ```

4. **COALESCED terminal state**: Parent tokens that are coalesced get status "coalesced" in node_states. The `RowOutcome.COALESCED` enum value is for the processor to return.
